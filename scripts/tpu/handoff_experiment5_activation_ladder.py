#!/usr/bin/env python3
"""Durably hand a sealed Experiment-5 core campaign to the activation ladder.

This is an operations-plane program, not part of the scientific source
revision.  It deliberately deploys the already-frozen scientific commit
``4472ff228b43d272caeac6a31b547dbe11e7e2be`` from an independently staged
checkout and bundle.

The only state-changing path is ``run``.  Its ordering is fail closed:

1. validate the staged payload without touching the core checkout or TPUs;
2. wait for the exact core plan to expose a validated final record;
3. prove that no core controller unit or TPU worker remains active;
4. update the clean canonical controller and worker checkouts;
5. regenerate commit-scoped incarnation evidence;
6. build H references and the disjoint activation-ladder plan; and
7. enable and start one persistent systemd user service for that plan.

Every completed phase is recorded beneath the extension namespace as a
self-hashed, create-only JSON record.  Re-execution proves existing evidence
instead of overwriting it.  A partial create-only probe attempt is preserved
and the next lexicographic attempt is selected.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


CORE_COMMIT = "674cecbb19ba9130a81a4e27249c9862de08ad17"
EXTENSION_COMMIT = "4472ff228b43d272caeac6a31b547dbe11e7e2be"
CORE_PLAN = Path(
    "/mnt/moe-sae-control/experiment5/contracts/"
    "campaign-ready-three-v1-674cecbb19ba-a1.yaml"
)
CORE_UNIT = "moe-sae-experiment5-674cecbb19ba-a1.service"
STAGED_CHECKOUT = Path("/home/tpuops/src/moe-sae-activation-4472ff228b43")
CANONICAL_CHECKOUT = Path("/home/tpuops/src/moe-sae")
DEPLOY_BUNDLE = Path(
    "/home/tpuops/.cache/moe-sae-deploy/4472ff228b43/repository.bundle"
)
R8_PLAN = Path(
    "/mnt/moe-sae-control/horizon-10m/contracts/direct-campaign-2f778bb-r8.yaml"
)
EXTENSION_ROOT = Path(
    "/mnt/moe-sae-control/experiment5/extensions/"
    "routed-activation-ladder-v1"
)
HANDOFF_ROOT = EXTENSION_ROOT / "handoff-4472ff228b43-a1"
CONTRACT_ROOT = EXTENSION_ROOT / "contracts"
H_REFERENCE_MANIFEST = (
    CONTRACT_ROOT / "h-references-core-674cecbb19ba.yaml"
)

PROJECT = "neural-dragnet"
EXECUTION_PROFILE = "ready-three-v1"
REMOTE_USER = "abistras"
REMOTE_CHECKOUT = "/home/abistras/moe-sae"
REMOTE_UV = "/home/abistras/.local/bin/uv"
REMOTE_ARTIFACT_ROOT = (
    "/home/abistras/moe-sae/artifacts/gemma4_v1/production"
)
CACHE_ROOT = "/dev/shm/moe-sae-local/horizon-10m/cache"
REQUIRED_LANE_BYTES = 159_000_000_000
REQUIRED_FREE_BYTES = 188_978_561_024
EXPECTED_RESOURCE_COUNT = 7
EXPECTED_TRAINING_COUNT = 12
EXPECTED_EVALUATION_COUNT = 54
EXPECTED_H_REFERENCE_COUNT = 18
POLL_SECONDS = 30
MAX_EVIDENCE_ATTEMPTS = 99
SCHEMA_VERSION = "gemma4-experiment5-activation-handoff-v1"

_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ATTEMPT = re.compile(r"^a[1-9][0-9]*$")
_QUIESCENCE_PATTERN = (
    "[t]rain_contribution.py|"
    "[c]ontribution_training_worker|"
    "[e]xperiment5_evaluation_worker|"
    "[e]xperiment5_routed_ladder_evaluation_worker|"
    "[e]valuate_experiment5|"
    "[r]un_experiment5_campaign|"
    "[r]un_experiment5_routed_ladder_campaign|"
    "[p]robe_controller_identity"
)


class HandoffError(RuntimeError):
    """The core-to-extension transition is unsafe or incomplete."""


@dataclass(frozen=True, slots=True)
class Resource:
    resource_id: str
    tpu_name: str
    zone: str
    accelerator_type: str
    provisioning_model: str
    controller_mount: str


RESOURCES = (
    Resource(
        "slice-sae-a",
        "gemma4-horizon-v5e-usc-a",
        "us-central1-a",
        "v5litepod-16",
        "SPOT",
        "/mnt/moe-sae-usc1",
    ),
    Resource(
        "slice-sae-b",
        "gemma4-horizon-harvest-0728",
        "us-central2-b",
        "v4-32",
        "ON_DEMAND",
        "/mnt/moe-sae-control",
    ),
    Resource(
        "slice-transcoder-b",
        "gemma4-horizon-v4-spot-trans-b",
        "us-central2-b",
        "v4-32",
        "SPOT",
        "/mnt/moe-sae-control",
    ),
)


@dataclass(frozen=True, slots=True)
class ProductionSpec:
    core_commit: str = CORE_COMMIT
    extension_commit: str = EXTENSION_COMMIT
    core_plan: Path = CORE_PLAN
    core_unit: str = CORE_UNIT
    staged_checkout: Path = STAGED_CHECKOUT
    canonical_checkout: Path = CANONICAL_CHECKOUT
    deploy_bundle: Path = DEPLOY_BUNDLE
    r8_plan: Path = R8_PLAN
    extension_root: Path = EXTENSION_ROOT
    handoff_root: Path = HANDOFF_ROOT
    contract_root: Path = CONTRACT_ROOT
    h_reference_manifest: Path = H_REFERENCE_MANIFEST
    project: str = PROJECT
    resources: tuple[Resource, ...] = RESOURCES

    @property
    def short_extension_commit(self) -> str:
        return self.extension_commit[:12]

    @property
    def staged_python(self) -> Path:
        return self.staged_checkout / ".venv/bin/python"

    @property
    def canonical_python(self) -> Path:
        return self.canonical_checkout / ".venv/bin/python-direct"


@dataclass(frozen=True, slots=True)
class CompletedCommand:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class CommandRunner:
    """Logged subprocess boundary, injectable in focused tests."""

    def __call__(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        check: bool = True,
    ) -> CompletedCommand:
        command = tuple(os.fspath(value) for value in args)
        print(f"[activation-handoff] $ {shlex.join(command)}", flush=True)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr, flush=True)
        result = CompletedCommand(
            command,
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            raise HandoffError(
                f"command failed with exit {result.returncode}{suffix}"
            )
        return result


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise HandoffError(f"immutable file is absent or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HandoffError(f"JSON evidence is absent or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HandoffError(f"cannot parse JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise HandoffError(f"JSON evidence is not an object: {path}")
    return value


def _load_phase(path: Path, expected_phase: str | None = None) -> dict[str, Any]:
    value = _load_json(path)
    digest = value.get("record_sha256")
    unsigned = {key: item for key, item in value.items() if key != "record_sha256"}
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or _canonical_hash(unsigned) != digest
        or (
            expected_phase is not None
            and value.get("phase") != expected_phase
        )
    ):
        raise HandoffError(f"handoff phase record is invalid: {path}")
    return value


def _seal_phase(path: Path, phase: str, body: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        **dict(body),
    }
    expected = {**unsigned, "record_sha256": _canonical_hash(unsigned)}
    payload = _canonical_bytes(expected)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        observed = _load_phase(path, phase)
        if observed != expected:
            raise HandoffError(f"immutable handoff phase changed: {path}")
        return observed
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return _seal_phase(path, phase, body)
        directory = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return expected


def _phase_path(spec: ProductionSpec, phase: str) -> Path:
    return spec.handoff_root / "phases" / f"{phase}.json"


def _parse_json_stdout(result: CompletedCommand, name: str) -> dict[str, Any]:
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise HandoffError(f"{name} did not emit one JSON object") from error
    if not isinstance(value, dict):
        raise HandoffError(f"{name} did not emit a JSON object")
    return value


def _git_head(
    runner: Callable[..., CompletedCommand],
    checkout: Path,
) -> str:
    result = runner(("git", "-C", checkout, "rev-parse", "HEAD"))
    head = result.stdout.strip()
    if _SHA.fullmatch(head) is None:
        raise HandoffError(f"checkout emitted an invalid Git SHA: {checkout}")
    return head


def _git_clean(
    runner: Callable[..., CompletedCommand],
    checkout: Path,
) -> bool:
    result = runner(
        (
            "git",
            "-C",
            checkout,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
    )
    return not result.stdout.strip()


def _remote(
    resource: Resource,
    command: str,
) -> tuple[str, ...]:
    return (
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        f"{REMOTE_USER}@{resource.tpu_name}",
        f"--project={PROJECT}",
        f"--zone={resource.zone}",
        "--worker=all",
        "--internal-ip",
        "--quiet",
        f"--command={command}",
    )


def _remote_worker_zero(
    resource: Resource,
    command: str,
) -> tuple[str, ...]:
    return (
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        f"{REMOTE_USER}@{resource.tpu_name}",
        f"--project={PROJECT}",
        f"--zone={resource.zone}",
        "--worker=0",
        "--internal-ip",
        "--quiet",
        f"--command={command}",
    )


def _remote_quiescence_command() -> str:
    return (
        "set -eu; "
        f"! pgrep -af {shlex.quote(_QUIESCENCE_PATTERN)}; "
        f"test -z \"$(git -C {shlex.quote(REMOTE_CHECKOUT)} "
        "status --porcelain=v1 --untracked-files=all)\""
    )


def campaign_unit_name(attempt: str) -> str:
    if _SAFE_ATTEMPT.fullmatch(attempt) is None:
        raise HandoffError(f"invalid evidence attempt: {attempt!r}")
    return (
        "moe-sae-experiment5-activation-"
        f"{EXTENSION_COMMIT[:12]}-{attempt}.service"
    )


def campaign_unit_text(
    *,
    plan: Path,
    attempt: str,
    spec: ProductionSpec,
) -> str:
    unit = campaign_unit_name(attempt)
    del unit
    return (
        "[Unit]\n"
        "Description=Gemma Experiment 5 activation ladder "
        f"{spec.short_extension_commit} {attempt}\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "StartLimitIntervalSec=3600\n"
        "StartLimitBurst=3\n"
        "\n"
        "[Service]\n"
        "Type=exec\n"
        f"WorkingDirectory={spec.canonical_checkout}\n"
        "Environment=PYTHONUNBUFFERED=1\n"
        f"ExecStart={spec.canonical_python} "
        f"{spec.canonical_checkout}/scripts/tpu/"
        "run_experiment5_routed_ladder_campaign.py "
        f"run --plan {plan}\n"
        "Restart=on-failure\n"
        "RestartSec=30s\n"
        "TimeoutStopSec=120s\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


class ActivationHandoff:
    def __init__(
        self,
        spec: ProductionSpec = ProductionSpec(),
        *,
        runner: Callable[..., CompletedCommand] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.spec = spec
        self.runner = runner or CommandRunner()
        self.sleep = sleep

    def validate_payload(self) -> dict[str, Any]:
        spec = self.spec
        if (
            _SHA.fullmatch(spec.core_commit) is None
            or _SHA.fullmatch(spec.extension_commit) is None
        ):
            raise HandoffError("production source revisions are invalid")
        if spec.extension_commit != EXTENSION_COMMIT:
            raise HandoffError("scientific extension commit changed")
        if len(spec.resources) != 3 or {
            row.resource_id for row in spec.resources
        } != {"slice-sae-a", "slice-sae-b", "slice-transcoder-b"}:
            raise HandoffError("ready-three resource inventory changed")
        if _git_head(self.runner, spec.staged_checkout) != spec.extension_commit:
            raise HandoffError("staged checkout is not the frozen extension commit")
        if not _git_clean(self.runner, spec.staged_checkout):
            raise HandoffError("staged extension checkout is dirty")
        if not spec.staged_python.is_file() or not os.access(
            spec.staged_python, os.X_OK
        ):
            raise HandoffError("staged extension interpreter is absent or not executable")
        if not spec.core_plan.is_file() or spec.core_plan.is_symlink():
            raise HandoffError("exact sealed core plan is absent or unsafe")
        if not spec.r8_plan.is_file() or spec.r8_plan.is_symlink():
            raise HandoffError("sealed r8 plan is absent or unsafe")
        bundle_heads = self.runner(
            ("git", "bundle", "list-heads", spec.deploy_bundle)
        ).stdout.splitlines()
        if not any(
            line.split(maxsplit=1)[0] == spec.extension_commit
            for line in bundle_heads
            if line.strip()
        ):
            raise HandoffError("deployment bundle does not advertise extension SHA")
        self.runner(
            (
                "git",
                "-C",
                spec.staged_checkout,
                "bundle",
                "verify",
                spec.deploy_bundle,
            )
        )
        plan_identity = self.runner(
            (
                spec.staged_python,
                "-c",
                (
                    "import sys;"
                    "from moe_sae.experiment5_campaign import "
                    "load_experiment5_campaign_plan;"
                    "print(load_experiment5_campaign_plan(sys.argv[1])."
                    "project_commit)"
                ),
                spec.core_plan,
            )
        ).stdout.strip()
        if plan_identity != spec.core_commit:
            raise HandoffError("sealed core plan source revision changed")
        return {
            "operator_path": str(Path(__file__).resolve()),
            "operator_file_sha256": _sha256_file(Path(__file__).resolve()),
            "core_plan_sha256": _sha256_file(spec.core_plan),
            "r8_plan_sha256": _sha256_file(spec.r8_plan),
            "bundle_sha256": _sha256_file(spec.deploy_bundle),
            "extension_commit": spec.extension_commit,
        }

    def _core_status(self) -> dict[str, Any]:
        result = self.runner(
            (
                self.spec.staged_python,
                self.spec.staged_checkout
                / "scripts/tpu/run_experiment5_campaign.py",
                "status",
                "--plan",
                self.spec.core_plan,
            )
        )
        return _parse_json_stdout(result, "core campaign status")

    def _active_core_units(self) -> list[str]:
        result = self.runner(
            (
                "systemctl",
                "--user",
                "is-active",
                self.spec.core_unit,
            ),
            check=False,
        )
        state = result.stdout.strip()
        if state in {"active", "activating", "reloading", "deactivating"}:
            return [f"{self.spec.core_unit}:{state}"]
        if state in {"inactive", "failed", "unknown", ""}:
            return []
        raise HandoffError(f"cannot classify core unit state: {state!r}")

    def _validated_core_final(
        self,
        status: Mapping[str, Any],
    ) -> tuple[dict[str, Any], float]:
        final = status.get("final")
        if not isinstance(final, dict) or final.get("status") != "completed":
            raise HandoffError("core campaign has no sealed completed final record")
        if status.get("campaign_id") != final.get("campaign_id"):
            raise HandoffError("core campaign identity differs from final record")
        if status.get("plan_sha256") != final.get("plan_sha256"):
            raise HandoffError("core plan identity differs from final record")
        report = final.get("report")
        if not isinstance(report, dict):
            raise HandoffError("core final record lacks its report identity")
        ledger_path_raw = report.get("budget_ledger_path")
        ledger_file_sha = report.get("budget_ledger_file_sha256")
        if (
            not isinstance(ledger_path_raw, str)
            or _SHA256.fullmatch(str(ledger_file_sha)) is None
        ):
            raise HandoffError("core final record lacks a valid budget ledger")
        ledger_path = Path(ledger_path_raw)
        if _sha256_file(ledger_path) != ledger_file_sha:
            raise HandoffError("core budget ledger bytes changed after final seal")
        validator = (
            "import json,sys;"
            "from moe_sae.experiment5_report import "
            "validate_experiment5_budget_ledger;"
            "v=validate_experiment5_budget_ledger("
            "json.load(open(sys.argv[1],encoding='ascii')));"
            "print(json.dumps(v,sort_keys=True))"
        )
        ledger = _parse_json_stdout(
            self.runner((self.spec.staged_python, "-c", validator, ledger_path)),
            "core budget ledger validator",
        )
        spent = ledger.get("incremental_cash_spent")
        if (
            isinstance(spent, bool)
            or not isinstance(spent, (int, float))
            or not 0 <= float(spent) <= 50
        ):
            raise HandoffError("core incremental cash spend is invalid")
        return final, float(spent)

    def wait_for_core(self, *, wait: bool) -> dict[str, Any]:
        while True:
            status = self._core_status()
            try:
                final, spent = self._validated_core_final(status)
            except HandoffError:
                active = self._active_core_units()
                if not wait:
                    raise
                if not active:
                    raise HandoffError(
                        "core is not sealed and no core controller unit is active"
                    )
                print(
                    "[activation-handoff] core is still active; "
                    f"polling again in {POLL_SECONDS}s",
                    flush=True,
                )
                self.sleep(POLL_SECONDS)
                continue
            active = self._active_core_units()
            if active:
                if not wait:
                    raise HandoffError(
                        "core final exists but a core controller unit is still active"
                    )
                print(
                    "[activation-handoff] core final is sealed; waiting for "
                    "controller unit quiescence",
                    flush=True,
                )
                self.sleep(POLL_SECONDS)
                continue
            for resource in self.spec.resources:
                self.runner(_remote(resource, _remote_quiescence_command()))
            return {
                "core_campaign_id": final["campaign_id"],
                "core_plan_sha256": status["plan_sha256"],
                "core_final_record_sha256": final["record_sha256"],
                "core_report_sha256": final["report"]["report_sha256"],
                "core_budget_ledger_sha256": final["report"][
                    "budget_ledger_sha256"
                ],
                "already_spent_usd": spent,
            }

    def seal_core_gate(self, *, wait: bool) -> dict[str, Any]:
        body = self.wait_for_core(wait=wait)
        return _seal_phase(
            _phase_path(self.spec, "01-core-sealed-and-quiescent"),
            "core-sealed-and-quiescent",
            {
                "core_commit": self.spec.core_commit,
                "extension_commit": self.spec.extension_commit,
                **body,
            },
        )

    def _canonical_checkout_ready(self) -> str:
        head = _git_head(self.runner, self.spec.canonical_checkout)
        if head not in {self.spec.core_commit, self.spec.extension_commit}:
            raise HandoffError(
                "canonical checkout is neither frozen core nor extension source"
            )
        if not _git_clean(self.runner, self.spec.canonical_checkout):
            raise HandoffError("canonical controller checkout is dirty")
        return head

    def deploy(self, gate: Mapping[str, Any]) -> dict[str, Any]:
        if gate.get("phase") != "core-sealed-and-quiescent":
            raise HandoffError("deployment lacks the immutable core gate")
        # Re-prove the live boundary immediately before the first mutation.
        self.wait_for_core(wait=False)
        before = self._canonical_checkout_ready()
        for resource in self.spec.resources:
            self.runner(_remote(resource, _remote_quiescence_command()))

        if before != self.spec.extension_commit:
            self.runner(
                (
                    "git",
                    "-C",
                    self.spec.canonical_checkout,
                    "fetch",
                    self.spec.deploy_bundle,
                    "HEAD",
                )
            )
            self.runner(
                (
                    "git",
                    "-C",
                    self.spec.canonical_checkout,
                    "checkout",
                    "--detach",
                    "FETCH_HEAD",
                )
            )
        if (
            _git_head(self.runner, self.spec.canonical_checkout)
            != self.spec.extension_commit
            or not _git_clean(self.runner, self.spec.canonical_checkout)
        ):
            raise HandoffError("canonical controller deployment did not seal")

        remote_bundle = f"/tmp/moe-sae-{self.spec.short_extension_commit}.bundle"
        for resource in self.spec.resources:
            self.runner(
                (
                    "gcloud",
                    "compute",
                    "tpus",
                    "tpu-vm",
                    "scp",
                    self.spec.deploy_bundle,
                    f"{REMOTE_USER}@{resource.tpu_name}:{remote_bundle}",
                    f"--project={PROJECT}",
                    f"--zone={resource.zone}",
                    "--worker=all",
                    "--internal-ip",
                    "--quiet",
                )
            )
        for resource in self.spec.resources:
            command = (
                "set -eu; "
                f"! pgrep -af {shlex.quote(_QUIESCENCE_PATTERN)}; "
                f"test -z \"$(git -C {shlex.quote(REMOTE_CHECKOUT)} "
                "status --porcelain=v1 --untracked-files=all)\"; "
                f"git -C {shlex.quote(REMOTE_CHECKOUT)} fetch "
                f"{shlex.quote(remote_bundle)} HEAD; "
                f"git -C {shlex.quote(REMOTE_CHECKOUT)} checkout --detach "
                "FETCH_HEAD; "
                f"test \"$(git -C {shlex.quote(REMOTE_CHECKOUT)} rev-parse HEAD)\" "
                f"= {shlex.quote(self.spec.extension_commit)}; "
                f"test -z \"$(git -C {shlex.quote(REMOTE_CHECKOUT)} "
                "status --porcelain=v1 --untracked-files=all)\""
            )
            self.runner(_remote(resource, command))
        body = {
            "core_gate_record_sha256": gate["record_sha256"],
            "controller_previous_commit": before,
            "deployed_commit": self.spec.extension_commit,
            "bundle_sha256": _sha256_file(self.spec.deploy_bundle),
            "targets": [
                {
                    "resource_id": row.resource_id,
                    "tpu_name": row.tpu_name,
                    "zone": row.zone,
                }
                for row in self.spec.resources
            ],
        }
        return _seal_phase(
            _phase_path(self.spec, "02-extension-deployed"),
            "extension-deployed",
            body,
        )

    def _attempt_paths(
        self,
        attempt: str,
    ) -> dict[str, tuple[str, str, Path]]:
        short = self.spec.short_extension_commit
        result: dict[str, tuple[str, str, Path]] = {}
        for resource in self.spec.resources:
            assignment_id = (
                f"experiment5-{EXECUTION_PROFILE}-{short}-{attempt}-"
                f"{resource.resource_id}"
            )
            probe_id = f"probe-{assignment_id}"
            remote_assignment = (
                f"{REMOTE_ARTIFACT_ROOT}/horizon-10m/topology/"
                f"{resource.tpu_name}/assignments/{assignment_id}"
            )
            remote_probe = f"{remote_assignment}/probes/{probe_id}"
            controller_probe = Path(resource.controller_mount) / Path(
                remote_probe
            ).relative_to(REMOTE_ARTIFACT_ROOT)
            result[resource.resource_id] = (
                remote_assignment,
                remote_probe,
                controller_probe,
            )
        return result

    @staticmethod
    def _probe_state(
        paths: Mapping[str, tuple[str, str, Path]],
    ) -> str:
        expected: list[Path] = []
        roots: list[Path] = []
        for assignment, _remote_probe, probe in paths.values():
            controller_assignment = probe.parent.parent
            del assignment
            roots.append(controller_assignment)
            expected.extend(
                (
                    controller_assignment / "tpu-description.json",
                    controller_assignment / "accelerator-contract.json",
                    controller_assignment / "spot-attestation.json",
                    probe / "controller-mapping.json",
                    probe / "cache-inventory.json",
                )
            )
        if all(not root.exists() for root in roots):
            return "absent"
        if all(path.is_file() and not path.is_symlink() for path in expected):
            return "complete"
        return "partial"

    def select_evidence_attempt(
        self,
    ) -> tuple[str, dict[str, tuple[str, str, Path]], str]:
        for number in range(1, MAX_EVIDENCE_ATTEMPTS + 1):
            attempt = f"a{number}"
            paths = self._attempt_paths(attempt)
            state = self._probe_state(paths)
            if state in {"absent", "complete"}:
                return attempt, paths, state
            print(
                "[activation-handoff] preserving partial create-only "
                f"evidence attempt {attempt}; trying the next identity",
                flush=True,
            )
        raise HandoffError("all bounded evidence attempt identities are occupied")

    def _clear_caches(self) -> None:
        for resource in self.spec.resources:
            self.runner(_remote(resource, _remote_quiescence_command()))
        command = (
            "set -eu; "
            f"sudo rm -rf -- {shlex.quote(CACHE_ROOT)}; "
            "sudo mount -o remount,size=180G /dev/shm; "
            "sudo install -d -m 0750 -o \"$(id -un)\" -g \"$(id -gn)\" "
            f"{shlex.quote(CACHE_ROOT)}; "
            f"test \"$(find {shlex.quote(CACHE_ROOT)} -mindepth 1 "
            "-maxdepth 1 -print -quit)\" = ''; "
            f"test \"$(df -B1 --output=avail {shlex.quote(CACHE_ROOT)} "
            "| tail -1 | tr -d ' ')\" -ge "
            f"{REQUIRED_FREE_BYTES}"
        )
        for resource in self.spec.resources:
            self.runner(_remote(resource, command))

    def _attest_and_probe(
        self,
        *,
        attempt: str,
        paths: Mapping[str, tuple[str, str, Path]],
    ) -> None:
        run_id = (
            f"gemma4-experiment5-{EXECUTION_PROFILE}-"
            f"{self.spec.short_extension_commit}"
        )
        for resource in self.spec.resources:
            assignment, probe, _controller_probe = paths[resource.resource_id]
            command = shlex.join(
                (
                    "cd",
                    REMOTE_CHECKOUT,
                )
            )
            # `cd` cannot be expressed as an argv command over gcloud's remote
            # shell, so compose only fixed, shell-quoted production literals.
            command = (
                f"cd {shlex.quote(REMOTE_CHECKOUT)} && "
                f"{shlex.quote(REMOTE_UV)} run --frozen --extra tpu python "
                "scripts/tpu/attest_accelerator.py "
                f"--run-id {shlex.quote(run_id)} "
                "--scientific-model hybrid --condition all "
                f"--project {shlex.quote(PROJECT)} "
                f"--tpu-name {shlex.quote(resource.tpu_name)} "
                f"--zone {shlex.quote(resource.zone)} "
                f"--accelerator-type {shlex.quote(resource.accelerator_type)} "
                f"--provisioning-model "
                f"{shlex.quote(resource.provisioning_model)} "
                f"--output-root {shlex.quote(assignment)}"
            )
            self.runner(_remote_worker_zero(resource, command))
            probe_id = Path(probe).name
            self.runner(
                (
                    self.spec.canonical_python,
                    self.spec.canonical_checkout
                    / "scripts/tpu/probe_controller_identity.py",
                    "--project",
                    PROJECT,
                    "--tpu-name",
                    resource.tpu_name,
                    "--zone",
                    resource.zone,
                    "--project-commit",
                    self.spec.extension_commit,
                    "--project-root",
                    REMOTE_CHECKOUT,
                    "--remote-uv",
                    REMOTE_UV,
                    "--contract",
                    f"{assignment}/accelerator-contract.json",
                    "--probe-id",
                    probe_id,
                    "--probe-root",
                    probe,
                    "--cache-root",
                    CACHE_ROOT,
                    "--required-lane-bytes",
                    str(REQUIRED_LANE_BYTES),
                    "--required-free-bytes",
                    str(REQUIRED_FREE_BYTES),
                    "--ssh-user",
                    REMOTE_USER,
                    "--internal-ip",
                )
            )
        if self._probe_state(paths) != "complete":
            raise HandoffError(
                f"evidence attempt {attempt} did not produce all sealed witnesses"
            )

    def evidence(self, deployed: Mapping[str, Any]) -> dict[str, Any]:
        if deployed.get("phase") != "extension-deployed":
            raise HandoffError("evidence generation lacks deployment proof")
        if (
            _git_head(self.runner, self.spec.canonical_checkout)
            != self.spec.extension_commit
        ):
            raise HandoffError("canonical checkout moved after deployment")
        phase_path = _phase_path(self.spec, "03-extension-evidence")
        if phase_path.exists():
            observed = _load_phase(phase_path, "extension-evidence")
            attempt = str(observed.get("evidence_attempt"))
            paths = self._attempt_paths(attempt)
            override = Path(str(observed.get("evidence_override")))
            if (
                self._probe_state(paths) != "complete"
                or _sha256_file(override)
                != observed.get("evidence_override_file_sha256")
            ):
                raise HandoffError("sealed extension evidence changed")
            return observed

        attempt, paths, state = self.select_evidence_attempt()
        if state == "absent":
            self._clear_caches()
            self._attest_and_probe(attempt=attempt, paths=paths)
        override = (
            self.spec.contract_root
            / f"evidence-{EXECUTION_PROFILE}-"
            f"{self.spec.short_extension_commit}-{attempt}.yaml"
        )
        if not override.exists():
            command: list[str | os.PathLike[str]] = [
                self.spec.canonical_python,
                self.spec.canonical_checkout
                / "scripts/tpu/build_experiment5_evidence_overrides.py",
                "--r8-plan",
                self.spec.r8_plan,
                "--project-commit",
                self.spec.extension_commit,
                "--execution-profile",
                EXECUTION_PROFILE,
            ]
            for resource in self.spec.resources:
                command.extend(
                    (
                        "--probe-root",
                        f"{resource.resource_id}="
                        f"{paths[resource.resource_id][2]}",
                    )
                )
            command.extend(("--output", override))
            self.runner(command)
        override_hash = _sha256_file(override)
        body = {
            "deployment_record_sha256": deployed["record_sha256"],
            "evidence_attempt": attempt,
            "evidence_override": str(override),
            "evidence_override_file_sha256": override_hash,
            "probe_roots": {
                key: str(value[2]) for key, value in sorted(paths.items())
            },
        }
        return _seal_phase(
            phase_path,
            "extension-evidence",
            body,
        )

    def _build_h_references(self) -> None:
        if self.spec.h_reference_manifest.exists():
            # The plan builder's strict loader below revalidates it; never
            # invoke the create-only writer over an existing artifact.
            return
        self.runner(
            (
                self.spec.canonical_python,
                self.spec.canonical_checkout
                / "scripts/tpu/build_experiment5_routed_ladder_h_references.py",
                "--experiment5-plan",
                self.spec.core_plan,
                "--output",
                self.spec.h_reference_manifest,
            )
        )

    def contracts(
        self,
        evidence: Mapping[str, Any],
        gate: Mapping[str, Any],
    ) -> dict[str, Any]:
        if (
            evidence.get("phase") != "extension-evidence"
            or gate.get("phase") != "core-sealed-and-quiescent"
        ):
            raise HandoffError("contract build lacks its immutable parents")
        phase_path = _phase_path(self.spec, "04-extension-contracts")
        if phase_path.exists():
            observed = _load_phase(phase_path, "extension-contracts")
            for field in (
                "h_reference_manifest",
                "activation_plan",
            ):
                path = Path(str(observed.get(field)))
                expected = observed.get(f"{field}_file_sha256")
                if _sha256_file(path) != expected:
                    raise HandoffError(f"sealed {field} changed")
            return observed

        self._build_h_references()
        attempt = str(evidence["evidence_attempt"])
        plan = (
            self.spec.contract_root
            / f"campaign-{EXECUTION_PROFILE}-"
            f"{self.spec.short_extension_commit}-{attempt}.yaml"
        )
        if not plan.exists():
            self.runner(
                (
                    self.spec.canonical_python,
                    self.spec.canonical_checkout
                    / "scripts/tpu/build_experiment5_routed_ladder_plan.py",
                    "--r8-plan",
                    self.spec.r8_plan,
                    "--project-commit",
                    self.spec.extension_commit,
                    "--execution-profile",
                    EXECUTION_PROFILE,
                    "--resource-evidence-overrides",
                    evidence["evidence_override"],
                    "--h-reference-manifest",
                    self.spec.h_reference_manifest,
                    "--already-spent-usd",
                    str(gate["already_spent_usd"]),
                    "--output",
                    plan,
                )
            )
        validation = _parse_json_stdout(
            self.runner(
                (
                    self.spec.canonical_python,
                    self.spec.canonical_checkout
                    / "scripts/tpu/run_experiment5_routed_ladder_campaign.py",
                    "validate",
                    "--plan",
                    plan,
                )
            ),
            "activation-ladder plan validation",
        )
        expected = {
            "resources": EXPECTED_RESOURCE_COUNT,
            "trajectories": EXPECTED_TRAINING_COUNT,
            "evaluations": EXPECTED_EVALUATION_COUNT,
            "h_references": EXPECTED_H_REFERENCE_COUNT,
        }
        if any(validation.get(key) != value for key, value in expected.items()):
            raise HandoffError("activation-ladder plan matrix changed")
        body = {
            "core_gate_record_sha256": gate["record_sha256"],
            "evidence_record_sha256": evidence["record_sha256"],
            "h_reference_manifest": str(self.spec.h_reference_manifest),
            "h_reference_manifest_file_sha256": _sha256_file(
                self.spec.h_reference_manifest
            ),
            "activation_plan": str(plan),
            "activation_plan_file_sha256": _sha256_file(plan),
            "activation_plan_sha256": validation["plan_sha256"],
            **expected,
        }
        return _seal_phase(
            phase_path,
            "extension-contracts",
            body,
        )

    def launch(
        self,
        contracts: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        if contracts.get("phase") != "extension-contracts":
            raise HandoffError("campaign launch lacks validated contracts")
        attempt = str(evidence["evidence_attempt"])
        unit_name = campaign_unit_name(attempt)
        plan = Path(str(contracts["activation_plan"]))
        unit_payload = campaign_unit_text(
            plan=plan,
            attempt=attempt,
            spec=self.spec,
        ).encode("utf-8")
        unit_path = (
            Path.home() / ".config/systemd/user" / unit_name
        )
        unit_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if unit_path.exists():
            if unit_path.is_symlink() or unit_path.read_bytes() != unit_payload:
                raise HandoffError("existing activation campaign unit differs")
        else:
            descriptor = os.open(
                unit_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(unit_payload)
                handle.flush()
                os.fsync(handle.fileno())
        linger = self.runner(
            (
                "loginctl",
                "show-user",
                str(os.getuid()),
                "-p",
                "Linger",
                "--value",
            )
        ).stdout.strip()
        if linger != "yes":
            raise HandoffError("tpuops systemd lingering is not enabled")
        self.runner(("systemctl", "--user", "daemon-reload"))
        self.runner(("systemctl", "--user", "enable", "--now", unit_name))
        active = self.runner(
            ("systemctl", "--user", "is-active", unit_name),
            check=False,
        )
        if active.returncode != 0 or active.stdout.strip() not in {
            "active",
            "activating",
        }:
            # A very short fully-completed replay may already be inactive; in
            # that case the read-only campaign status must prove final.
            status = _parse_json_stdout(
                self.runner(
                    (
                        self.spec.canonical_python,
                        self.spec.canonical_checkout
                        / "scripts/tpu/"
                        "run_experiment5_routed_ladder_campaign.py",
                        "status",
                        "--plan",
                        plan,
                    )
                ),
                "activation campaign status",
            )
            if not isinstance(status.get("final"), dict):
                raise HandoffError("activation campaign unit did not stay active")
        body = {
            "contracts_record_sha256": contracts["record_sha256"],
            "unit_name": unit_name,
            "unit_path": str(unit_path),
            "unit_file_sha256": hashlib.sha256(unit_payload).hexdigest(),
            "activation_plan": str(plan),
            "activation_plan_sha256": contracts["activation_plan_sha256"],
            "status_command": (
                f"{self.spec.canonical_python} "
                f"{self.spec.canonical_checkout}/scripts/tpu/"
                "run_experiment5_routed_ladder_campaign.py status "
                f"--plan {plan}"
            ),
        }
        return _seal_phase(
            _phase_path(self.spec, "05-campaign-launched"),
            "campaign-launched",
            body,
        )

    def run(self, *, wait: bool = True) -> dict[str, Any]:
        self.spec.handoff_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = self.spec.handoff_root / "handoff.lock"
        with lock_path.open("a+b") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise HandoffError("another activation handoff owns the lock") from error
            payload = self.validate_payload()
            _seal_phase(
                _phase_path(self.spec, "00-payload-validated"),
                "payload-validated",
                payload,
            )
            launched_path = _phase_path(self.spec, "05-campaign-launched")
            if launched_path.exists():
                return _load_phase(launched_path, "campaign-launched")
            # Reconcile the narrow crash window after systemd accepted the
            # campaign unit but before phase 05 was sealed.  At that point TPU
            # work may already be active, so repeating the pre-deployment
            # quiescence gate would be both misleading and unrecoverable.
            contracts_path = _phase_path(
                self.spec, "04-extension-contracts"
            )
            evidence_path = _phase_path(
                self.spec, "03-extension-evidence"
            )
            if contracts_path.exists() and evidence_path.exists():
                contracts = _load_phase(
                    contracts_path, "extension-contracts"
                )
                evidence = _load_phase(
                    evidence_path, "extension-evidence"
                )
                candidate_unit = (
                    Path.home()
                    / ".config/systemd/user"
                    / campaign_unit_name(str(evidence["evidence_attempt"]))
                )
                if candidate_unit.exists():
                    return self.launch(contracts, evidence)
            gate = self.seal_core_gate(wait=wait)
            deployed_path = _phase_path(self.spec, "02-extension-deployed")
            if deployed_path.exists():
                deployed = _load_phase(deployed_path, "extension-deployed")
                if (
                    deployed.get("deployed_commit")
                    != self.spec.extension_commit
                ):
                    raise HandoffError("deployment phase source changed")
            else:
                deployed = self.deploy(gate)
            evidence = self.evidence(deployed)
            contracts = self.contracts(evidence, gate)
            return self.launch(contracts, evidence)

    def status(self) -> dict[str, Any]:
        phases: dict[str, Any] = {}
        phase_root = self.spec.handoff_root / "phases"
        if phase_root.is_dir():
            for path in sorted(phase_root.glob("*.json")):
                phases[path.stem] = _load_phase(path)
        return {
            "schema_version": SCHEMA_VERSION,
            "core_commit": self.spec.core_commit,
            "extension_commit": self.spec.extension_commit,
            "core_plan": str(self.spec.core_plan),
            "handoff_root": str(self.spec.handoff_root),
            "phases": phases,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("status")
    run = subparsers.add_parser("run")
    run.add_argument(
        "--no-wait",
        action="store_true",
        help="fail immediately unless the core is already sealed and quiescent",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    operator = ActivationHandoff()
    try:
        if args.action == "validate":
            print(
                json.dumps(operator.validate_payload(), indent=2, sort_keys=True)
            )
        elif args.action == "status":
            print(json.dumps(operator.status(), indent=2, sort_keys=True))
        else:
            print(
                json.dumps(
                    operator.run(wait=not args.no_wait),
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
        return 0
    except (HandoffError, OSError, subprocess.SubprocessError) as error:
        print(f"[activation-handoff] ERROR: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
