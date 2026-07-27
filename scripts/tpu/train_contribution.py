#!/usr/bin/env python3
"""Four-controller operator for Experiment-2 SAE qualification and training.

The public command runs on the Mac.  Worker 0 first seals an immutable attempt
plan from the v4 harvester-qualification bundle.  The Mac then uses the
bundle's physical-controller -> observed-JAX-process permutation to:

1. prefetch and hash-prove exactly one contiguous contribution lane on each
   mapped controller; and
2. fan out a bounded direct resident experiment, a hard-600-second two-mode
   qualifier, or a longer exactly receipt-authorized training run.

The operator never assumes that ``--worker=N`` implies ``jax.process_index=N``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

PROCESS_COUNT = 4
QUALIFICATION_SECONDS = 600
DIRECT_EXPERIMENT_SECONDS = 3_600
QUALIFICATION_TRANSPORT_WAIT_SECONDS = 720
PREFETCH_SECONDS = 600
PREFETCH_TRANSPORT_WAIT_SECONDS = 720
TRAIN_TRANSPORT_GRACE_SECONDS = 300

COORDINATOR_BUNDLE_SCHEMA_VERSION = (
    "gemma4-contribution-qualification-coordinator-bundle-v4"
)
QUALIFICATION_RECEIPT_SCHEMA_VERSION = (
    "gemma4-contribution-performance-qualification-v2"
)
RUN_CONTRACT_SCHEMA_VERSION = "gemma4-contribution-sae-operator-run-v1"
ATTEMPT_PLAN_SCHEMA_VERSION = "gemma4-contribution-sae-operator-attempt-v1"

_COORDINATOR_BUNDLE_FIELDS = {
    "schema_version",
    "identity_relationship",
    "authorization_contract_sha256",
    "host_evidence_sha256",
    "controller_to_jax_process_index",
    "observed_executable_identity",
    "recomposition_sidecar_sha256",
    "recomposition_sidecar_canonical_sha256",
    "source_evidence",
    "useful_flops_ledger",
    "qualification_receipt",
    "coordinator_bundle_sha256",
}

CONDITIONS = ("A", "R", "B", "F", "Q")
SCIENTIFIC_MODELS = ("sae", "transcoder")
ACTIONS = ("qualify", "train")
DIRECT_BOUNDED_ADMISSION = "bounded_direct_resident"
QUALIFIED_ADMISSION = "qualified_mode_selection"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_SAFE_PHASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class OperatorError(RuntimeError):
    """The training operator cannot continue safely."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Explicit non-secret inputs for one mapped four-controller attempt."""

    gcp_project: str
    tpu_name: str
    tpu_zone: str
    project_commit: str
    project_root: str
    remote_uv: str
    capture_control_root: str
    object_prefix: str
    recomposition_sidecar: str
    qualification_bundle: str
    cache_root: str
    cache_max_bytes: int
    sae_control_root: str
    checkpoint_payload_root: str
    ssh_key_file: str | None
    action: str
    phase: str
    condition: str
    seed: int
    run_id: str
    attempt_id: str
    maximum_runtime_seconds: int
    local_log_root: Path
    scientific_model: str = "sae"
    source_overlay_control_root: str | None = None
    source_overlay_object_prefix: str | None = None

    @property
    def config_path(self) -> str:
        filename = (
            "gemma4_transcoder_v1.yaml"
            if self.scientific_model == "transcoder"
            else "gemma4_contribution_v1.yaml"
        )
        return f"{self.project_root}/configs/{filename}"

    @property
    def run_root(self) -> str:
        return f"{self.sae_control_root}/runs/{self.run_id}"

    @property
    def attempt_root(self) -> str:
        return (
            f"{self.run_root}/operator-attempts/"
            f"{self.action}-{self.attempt_id}"
        )

    @property
    def selection_path(self) -> str:
        return f"{self.run_root}/mode-selection.json"

    @property
    def checkpoint_control_root(self) -> str:
        return f"{self.run_root}/checkpoints"

    @property
    def result_path(self) -> str:
        return f"{self.run_root}/training-result.json"

    @property
    def metric_path(self) -> str:
        return f"{self.run_root}/metrics.jsonl"

    def process_cache(self, process_index: int) -> str:
        # Stable across conditions and seeds: one sealed phase capture should
        # be downloaded only once per process/controller.
        return f"{self.cache_root}/process-p{process_index:05d}"

    def qualified_prefetch_receipt(
        self,
        controller_index: int,
        process_index: int,
    ) -> str:
        return (
            f"{self.run_root}/qualified-prefetch/"
            f"controller-c{controller_index:05d}-p{process_index:05d}.json"
        )

    def refresh_prefetch_receipt(
        self,
        controller_index: int,
        process_index: int,
    ) -> str:
        return (
            f"{self.attempt_root}/prefetch-refresh/"
            f"controller-c{controller_index:05d}-p{process_index:05d}.json"
        )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise OperatorError("operator evidence is not finite canonical JSON") from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "")
    if not value:
        raise OperatorError(f"required environment variable is unset: {name}")
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise OperatorError(f"{name} contains an unsafe control character")
    return value


def _positive_int(name: str, value: object, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise OperatorError(f"{name} must be an integer")
    try:
        selected = int(value)
    except (TypeError, ValueError) as error:
        raise OperatorError(f"{name} must be an integer") from error
    minimum = 0 if allow_zero else 1
    if selected < minimum or str(selected) != str(value):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise OperatorError(f"{name} must be a canonical {qualifier} integer")
    return selected


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OperatorError(f"{name} must be a lowercase SHA-256")
    return value


def _absolute_remote_path(name: str, value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or value == "/":
        raise OperatorError(f"{name} must be a non-root absolute POSIX path")
    return str(path)


def _gcs_prefix(name: str, value: str) -> str:
    if (
        not value.startswith("gs://")
        or value.count("/") < 3
        or any(character.isspace() for character in value)
        or any(character in value for character in "*?[]")
    ):
        raise OperatorError(f"{name} must be a non-root, non-glob gs:// prefix")
    return value.rstrip("/")


def _settings(
    environment: Mapping[str, str],
    args: argparse.Namespace,
) -> Settings:
    project = _required(environment, "GCP_PROJECT")
    tpu_name = _required(environment, "TPU_NAME")
    zone = _required(environment, "TPU_ZONE")
    commit = _required(environment, "PROJECT_COMMIT")
    if re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project) is None:
        raise OperatorError("GCP_PROJECT is invalid")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", tpu_name) is None:
        raise OperatorError("TPU_NAME is invalid")
    if re.fullmatch(r"[a-z0-9-]+", zone) is None:
        raise OperatorError("TPU_ZONE is invalid")
    if _GIT_SHA_RE.fullmatch(commit) is None:
        raise OperatorError("PROJECT_COMMIT must be a full lowercase Git SHA")
    if args.action not in ACTIONS:
        raise OperatorError("--action is invalid")
    if args.scientific_model not in SCIENTIFIC_MODELS:
        raise OperatorError("--scientific-model is invalid")
    if args.condition not in CONDITIONS:
        raise OperatorError("--condition is invalid")
    if args.scientific_model == "transcoder" and args.condition == "Q":
        raise OperatorError("transcoder mode supports routed A/R/B/F only")
    if _SAFE_PHASE_RE.fullmatch(args.phase) is None:
        raise OperatorError("--phase is invalid")
    if _SAFE_ID_RE.fullmatch(args.run_id) is None:
        raise OperatorError("--run-id is invalid")
    if _SAFE_ID_RE.fullmatch(args.attempt_id) is None:
        raise OperatorError("--attempt-id is invalid")
    seed = _positive_int("--seed", args.seed, allow_zero=True)
    maximum = _positive_int(
        "--maximum-runtime-seconds",
        args.maximum_runtime_seconds,
    )
    if args.action == "qualify" and maximum != QUALIFICATION_SECONDS:
        raise OperatorError(
            "SAE qualification maximum-runtime-seconds is fixed at 600"
        )
    default_logs = (
        Path.cwd()
        / "artifacts"
        / "gemma4_v1"
        / "operator-results"
        / (
            "experiment3-transcoder"
            if args.scientific_model == "transcoder"
            else "experiment2-sae"
        )
        / args.run_id
        / f"{args.action}-{args.attempt_id}"
    )
    local_logs = Path(
        environment.get("CONTRIBUTION_SAE_LOCAL_LOG_ROOT", str(default_logs))
    ).expanduser().resolve()
    return Settings(
        gcp_project=project,
        tpu_name=tpu_name,
        tpu_zone=zone,
        project_commit=commit,
        project_root=_absolute_remote_path(
            "TPU_PROJECT_ROOT",
            _required(environment, "TPU_PROJECT_ROOT"),
        ),
        remote_uv=_absolute_remote_path(
            "REMOTE_UV",
            _required(environment, "REMOTE_UV"),
        ),
        scientific_model=args.scientific_model,
        capture_control_root=_absolute_remote_path(
            "CONTRIBUTION_TRAIN_CAPTURE_ROOT",
            _required(environment, "CONTRIBUTION_TRAIN_CAPTURE_ROOT"),
        ),
        object_prefix=_gcs_prefix(
            "CONTRIBUTION_OBJECT_PREFIX",
            _required(environment, "CONTRIBUTION_OBJECT_PREFIX"),
        ),
        source_overlay_control_root=(
            _absolute_remote_path(
                "CONTRIBUTION_SOURCE_OVERLAY_ROOT",
                _required(environment, "CONTRIBUTION_SOURCE_OVERLAY_ROOT"),
            )
            if args.scientific_model == "transcoder"
            else None
        ),
        source_overlay_object_prefix=(
            _gcs_prefix(
                "CONTRIBUTION_SOURCE_OVERLAY_PREFIX",
                _required(environment, "CONTRIBUTION_SOURCE_OVERLAY_PREFIX"),
            )
            if args.scientific_model == "transcoder"
            else None
        ),
        recomposition_sidecar=_absolute_remote_path(
            "CONTRIBUTION_RECOMPOSITION_SIDECAR",
            _required(environment, "CONTRIBUTION_RECOMPOSITION_SIDECAR"),
        ),
        qualification_bundle=_absolute_remote_path(
            "CONTRIBUTION_QUALIFICATION_BUNDLE",
            _required(environment, "CONTRIBUTION_QUALIFICATION_BUNDLE"),
        ),
        cache_root=_absolute_remote_path(
            "CONTRIBUTION_SAE_CACHE_ROOT",
            _required(environment, "CONTRIBUTION_SAE_CACHE_ROOT"),
        ),
        cache_max_bytes=_positive_int(
            "CONTRIBUTION_SAE_CACHE_MAX_BYTES",
            _required(environment, "CONTRIBUTION_SAE_CACHE_MAX_BYTES"),
        ),
        sae_control_root=_absolute_remote_path(
            "CONTRIBUTION_SAE_CONTROL_ROOT",
            _required(environment, "CONTRIBUTION_SAE_CONTROL_ROOT"),
        ),
        checkpoint_payload_root=_gcs_prefix(
            "CONTRIBUTION_SAE_CHECKPOINT_ROOT",
            _required(environment, "CONTRIBUTION_SAE_CHECKPOINT_ROOT"),
        ),
        ssh_key_file=environment.get("SSH_KEY_FILE") or None,
        action=args.action,
        phase=args.phase,
        condition=args.condition,
        seed=seed,
        run_id=args.run_id,
        attempt_id=args.attempt_id,
        maximum_runtime_seconds=maximum,
        local_log_root=local_logs,
    )


def _admission_mode(action: str, maximum_runtime_seconds: int) -> str:
    """Select the explicit training admission path.

    A bounded train is itself the at-most-one-hour experiment and therefore
    lowers the fixed resident graph directly.  Longer runs preserve the
    existing two-mode qualification contract.
    """

    if action == "train" and maximum_runtime_seconds <= DIRECT_EXPERIMENT_SECONDS:
        return DIRECT_BOUNDED_ADMISSION
    return QUALIFIED_ADMISSION


def _read_json(path: Path, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperatorError(f"cannot read {name}: {path}") from error
    if not isinstance(value, dict):
        raise OperatorError(f"{name} must be a JSON object")
    return value


def _validate_self_hash(
    value: Mapping[str, object],
    *,
    hash_field: str,
    name: str,
) -> tuple[dict[str, object], str]:
    unsigned = dict(value)
    claimed = _sha256(name, unsigned.pop(hash_field, None))
    if _canonical_sha256(unsigned) != claimed:
        raise OperatorError(f"{name} self-hash is invalid")
    return unsigned, claimed


def _controller_mapping(bundle: Mapping[str, object]) -> dict[int, int]:
    raw = bundle.get("controller_to_jax_process_index")
    if not isinstance(raw, list) or len(raw) != PROCESS_COUNT:
        raise OperatorError(
            "qualification bundle lacks four controller/JAX mappings"
        )
    result: dict[int, int] = {}
    processes: set[int] = set()
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {
            "controller_index",
            "process_index",
        }:
            raise OperatorError("qualification controller mapping is malformed")
        controller = _positive_int(
            "controller_index",
            item["controller_index"],
            allow_zero=True,
        )
        process = _positive_int(
            "process_index",
            item["process_index"],
            allow_zero=True,
        )
        if (
            controller >= PROCESS_COUNT
            or process >= PROCESS_COUNT
            or controller in result
            or process in processes
        ):
            raise OperatorError(
                "qualification controller mapping is not a permutation"
            )
        result[controller] = process
        processes.add(process)
    if set(result) != set(range(PROCESS_COUNT)) or processes != set(
        range(PROCESS_COUNT)
    ):
        raise OperatorError(
            "qualification controller mapping is not the exact 0..3 permutation"
        )
    return dict(sorted(result.items()))


def _absolute_file(value: str, *, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise OperatorError(f"{name} must be absolute")
    selected = path.resolve()
    if not selected.is_file() or selected.is_symlink():
        raise OperatorError(f"{name} is missing, non-regular, or a symlink")
    return selected


def _qualification_lineage(
    bundle_path: Path,
    sidecar_path: Path,
) -> dict[str, object]:
    bundle = _read_json(bundle_path, name="qualification coordinator bundle")
    if set(bundle) != _COORDINATOR_BUNDLE_FIELDS:
        raise OperatorError(
            "qualification coordinator bundle fields are not the exact v4 "
            "contract"
        )
    unsigned, bundle_sha256 = _validate_self_hash(
        bundle,
        hash_field="coordinator_bundle_sha256",
        name="qualification coordinator bundle",
    )
    if unsigned.get("schema_version") != COORDINATOR_BUNDLE_SCHEMA_VERSION:
        raise OperatorError("qualification coordinator bundle is not v4")
    mapping = _controller_mapping(unsigned)
    sidecar = _read_json(sidecar_path, name="recomposition sidecar")
    try:
        from moe_sae.contribution_artifacts import (
            validate_recomposition_sidecar,
        )
        from moe_sae.contribution_performance import (
            validate_qualification_receipt,
        )

        checked_sidecar = validate_recomposition_sidecar(sidecar)
        checked_receipt = validate_qualification_receipt(
            unsigned.get("qualification_receipt")
        )
    except (TypeError, ValueError) as error:
        raise OperatorError(
            "qualification bundle contains invalid passing evidence"
        ) from error
    semantic_sha256 = _sha256(
        "recomposition sidecar semantic SHA-256",
        checked_sidecar.get("sidecar_sha256"),
    )
    canonical_sha256 = _canonical_sha256(checked_sidecar)
    observed_identity = unsigned.get("observed_executable_identity")
    receipt_identity = checked_receipt.get("identity")
    if (
        not isinstance(observed_identity, Mapping)
        or observed_identity.get("identity_sha256")
        != checked_receipt.get("identity_sha256")
        or {
            name: value
            for name, value in observed_identity.items()
            if name != "identity_sha256"
        }
        != receipt_identity
    ):
        raise OperatorError(
            "qualification bundle receipt and observed executable identity "
            "differ"
        )
    host_hashes = unsigned.get("host_evidence_sha256")
    if (
        not isinstance(host_hashes, list)
        or len(host_hashes) != PROCESS_COUNT
        or any(
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
            for value in host_hashes
        )
        or len(set(host_hashes)) != PROCESS_COUNT
    ):
        raise OperatorError(
            "qualification bundle host evidence inventory is incomplete"
        )
    if (
        unsigned.get("recomposition_sidecar_sha256") != semantic_sha256
        or unsigned.get("recomposition_sidecar_canonical_sha256")
        != canonical_sha256
    ):
        raise OperatorError(
            "qualification bundle does not bind the exact recomposition sidecar"
        )
    return {
        "qualification_bundle_sha256": bundle_sha256,
        "qualification_receipt_sha256": checked_receipt["receipt_sha256"],
        "recomposition_sidecar_sha256": semantic_sha256,
        "recomposition_sidecar_canonical_sha256": canonical_sha256,
        "controller_to_jax_process_index": [
            {"controller_index": controller, "process_index": process}
            for controller, process in mapping.items()
        ],
    }


def _manifest_sha256(capture_root: Path) -> str:
    manifest = _read_json(capture_root / "manifest.json", name="capture manifest")
    _unsigned, claimed = _validate_self_hash(
        manifest,
        hash_field="manifest_sha256",
        name="capture manifest",
    )
    return claimed


def _atomic_create(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = _canonical_json_bytes(dict(value)) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OperatorError("immutable JSON write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _create_or_validate(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        if path.is_symlink() or _read_json(path, name=path.name) != dict(value):
            raise OperatorError(f"existing immutable contract differs: {path}")
        return
    _atomic_create(path, value)


def initialize_attempt(args: argparse.Namespace) -> dict[str, object]:
    """Seal one run identity and one action-attempt plan on shared POSIX."""

    config = _absolute_file(args.config, name="config")
    sidecar = _absolute_file(args.recomposition_sidecar, name="sidecar")
    bundle = _absolute_file(args.qualification_bundle, name="bundle")
    capture_root = Path(args.capture_control_root).expanduser().resolve()
    if (
        not capture_root.is_dir()
        or capture_root.is_symlink()
        or not (capture_root / "manifest.json").is_file()
    ):
        raise OperatorError("capture_control_root is not one sealed directory")
    scientific_model = getattr(args, "scientific_model", "sae")
    if scientific_model not in SCIENTIFIC_MODELS:
        raise OperatorError("scientific_model is invalid")
    source_overlay_root: Path | None = None
    if scientific_model == "transcoder":
        source_control = getattr(args, "source_overlay_control_root", None)
        source_prefix = getattr(args, "source_overlay_object_prefix", None)
        if not source_control or not source_prefix:
            raise OperatorError(
                "transcoder initialization requires the routed-input overlay"
            )
        source_overlay_root = Path(source_control).expanduser().resolve()
        if (
            not source_overlay_root.is_dir()
            or source_overlay_root.is_symlink()
            or not (source_overlay_root / "manifest.json").is_file()
        ):
            raise OperatorError(
                "source_overlay_control_root is not one sealed directory"
            )
    sae_control_root = Path(args.sae_control_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    attempt_root = Path(args.attempt_root).expanduser().resolve()
    expected_run = sae_control_root / "runs" / args.run_id
    expected_attempt = (
        expected_run / "operator-attempts" / f"{args.action}-{args.attempt_id}"
    )
    if run_root != expected_run or attempt_root != expected_attempt:
        raise OperatorError("run/attempt paths are not canonical")
    if sae_control_root.is_symlink() or run_root.is_symlink():
        raise OperatorError("SAE control paths must not be symbolic links")
    lineage = _qualification_lineage(bundle, sidecar)
    admission_mode = _admission_mode(
        args.action,
        args.maximum_runtime_seconds,
    )
    direct_bounded = admission_mode == DIRECT_BOUNDED_ADMISSION
    mapping = {
        int(item["controller_index"]): int(item["process_index"])
        for item in lineage["controller_to_jax_process_index"]
    }
    contract_body: dict[str, object] = {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "project_commit": args.project_commit,
        "config": str(config),
        "config_file_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "phase": args.phase,
        "condition": args.condition,
        "seed": args.seed,
        "capture_control_root": str(capture_root),
        "capture_manifest_sha256": _manifest_sha256(capture_root),
        "object_prefix": args.object_prefix.rstrip("/"),
        "recomposition_sidecar": str(sidecar),
        "qualification_bundle": str(bundle),
        **lineage,
        "process_count": PROCESS_COUNT,
        "cache_root": args.cache_root,
        "cache_max_bytes": args.cache_max_bytes,
        "checkpoint_payload_root": args.checkpoint_payload_root.rstrip("/"),
        "admission_mode": admission_mode,
    }
    if source_overlay_root is not None:
        contract_body.update(
            {
                "scientific_model": scientific_model,
                "source_overlay_control_root": str(source_overlay_root),
                "source_overlay_manifest_sha256": _manifest_sha256(
                    source_overlay_root
                ),
                "source_overlay_object_prefix": (
                    source_prefix.rstrip("/")
                ),
            }
        )
    contract = {
        **contract_body,
        "run_contract_sha256": _canonical_sha256(contract_body),
    }
    contract_path = run_root / "run-contract.json"
    if (
        args.action == "train"
        and not direct_bounded
        and not contract_path.is_file()
    ):
        raise OperatorError("training requires a prior qualified run contract")
    sae_control_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    _create_or_validate(contract_path, contract)
    selection = run_root / "mode-selection.json"
    result = run_root / "training-result.json"
    if args.action == "qualify" and selection.exists():
        raise OperatorError("run already has a completed mode selection")
    if args.action == "train":
        if direct_bounded and selection.exists():
            raise OperatorError(
                "bounded direct training must not inherit a mode selection"
            )
        if not direct_bounded and not selection.is_file():
            raise OperatorError("training requires mode-selection.json")
        if result.exists():
            raise OperatorError("run already has a final training result")
        if not direct_bounded:
            for controller, process in mapping.items():
                receipt = (
                    run_root
                    / "qualified-prefetch"
                    / f"controller-c{controller:05d}-p{process:05d}.json"
                )
                if not receipt.is_file():
                    raise OperatorError(
                        "training requires every qualification-bound prefetch "
                        "receipt"
                    )
    try:
        attempt_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    except FileExistsError as error:
        raise OperatorError(
            f"immutable operator attempt already exists: {attempt_root}"
        ) from error
    plan_body: dict[str, object] = {
        "schema_version": ATTEMPT_PLAN_SCHEMA_VERSION,
        "action": args.action,
        "attempt_id": args.attempt_id,
        "attempt_root": str(attempt_root),
        "run_root": str(run_root),
        "run_contract_sha256": contract["run_contract_sha256"],
        "admission_mode": admission_mode,
        "controller_to_jax_process_index": (
            lineage["controller_to_jax_process_index"]
        ),
        "maximum_runtime_seconds": args.maximum_runtime_seconds,
        "qualification_hard_deadline_seconds": QUALIFICATION_SECONDS,
        "prefetch_hard_deadline_seconds": PREFETCH_SECONDS,
        "terminal_policy": (
            "four_mapped_prefetches_then_four_concurrent_supervised_workers"
        ),
    }
    plan = {
        **plan_body,
        "attempt_plan_sha256": _canonical_sha256(plan_body),
    }
    _atomic_create(attempt_root / "operator-plan.json", plan)
    return plan


def _validate_attempt_plan(
    value: Mapping[str, object],
    settings: Settings,
) -> tuple[dict[str, object], dict[int, int]]:
    unsigned, _claimed = _validate_self_hash(
        value,
        hash_field="attempt_plan_sha256",
        name="SAE attempt plan",
    )
    if (
        unsigned.get("schema_version") != ATTEMPT_PLAN_SCHEMA_VERSION
        or unsigned.get("action") != settings.action
        or unsigned.get("attempt_id") != settings.attempt_id
        or unsigned.get("attempt_root") != settings.attempt_root
        or unsigned.get("run_root") != settings.run_root
        or unsigned.get("maximum_runtime_seconds")
        != settings.maximum_runtime_seconds
        or unsigned.get("admission_mode")
        != _admission_mode(
            settings.action,
            settings.maximum_runtime_seconds,
        )
        or unsigned.get("qualification_hard_deadline_seconds")
        != QUALIFICATION_SECONDS
        or unsigned.get("prefetch_hard_deadline_seconds")
        != PREFETCH_SECONDS
    ):
        raise OperatorError("remote SAE attempt plan differs from this request")
    mapping = _controller_mapping(unsigned)
    return dict(value), mapping


def _gcloud(settings: Settings, *, worker: int, remote_command: str) -> list[str]:
    command = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        settings.tpu_name,
        f"--project={settings.gcp_project}",
        f"--zone={settings.tpu_zone}",
        f"--worker={worker}",
        f"--command={remote_command}",
        "--ssh-flag=-oBatchMode=yes",
        "--ssh-flag=-oConnectTimeout=30",
        "--ssh-flag=-oConnectionAttempts=3",
        "--ssh-flag=-oServerAliveInterval=30",
        "--ssh-flag=-oServerAliveCountMax=4",
        "--quiet",
    ]
    if settings.ssh_key_file:
        command.insert(-1, f"--ssh-key-file={settings.ssh_key_file}")
    return command


def _bash_payload(lines: Sequence[str]) -> str:
    return "bash -lc " + shlex.quote("\n".join(lines))


def _clean_checkout_lines(settings: Settings) -> list[str]:
    return [
        f"cd {shlex.quote(settings.project_root)}",
        ('test "$(git rev-parse HEAD)" = ' f"{shlex.quote(settings.project_commit)}"),
        'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
    ]


def _init_command(settings: Settings) -> list[str]:
    helper = f"{settings.project_root}/scripts/tpu/train_contribution.py"
    command = [
        settings.remote_uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        helper,
        "_init-attempt",
        "--action",
        settings.action,
        "--run-id",
        settings.run_id,
        "--attempt-id",
        settings.attempt_id,
        "--project-commit",
        settings.project_commit,
        "--scientific-model",
        settings.scientific_model,
        "--config",
        settings.config_path,
        "--phase",
        settings.phase,
        "--condition",
        settings.condition,
        "--seed",
        str(settings.seed),
        "--capture-control-root",
        settings.capture_control_root,
        "--object-prefix",
        settings.object_prefix,
        "--recomposition-sidecar",
        settings.recomposition_sidecar,
        "--qualification-bundle",
        settings.qualification_bundle,
        "--cache-root",
        settings.cache_root,
        "--cache-max-bytes",
        str(settings.cache_max_bytes),
        "--sae-control-root",
        settings.sae_control_root,
        "--run-root",
        settings.run_root,
        "--attempt-root",
        settings.attempt_root,
        "--checkpoint-payload-root",
        settings.checkpoint_payload_root,
        "--maximum-runtime-seconds",
        str(settings.maximum_runtime_seconds),
    ]
    if settings.scientific_model == "transcoder":
        if (
            settings.source_overlay_control_root is None
            or settings.source_overlay_object_prefix is None
        ):
            raise OperatorError("transcoder settings lack their source overlay")
        command.extend(
            (
                "--source-overlay-control-root",
                settings.source_overlay_control_root,
                "--source-overlay-object-prefix",
                settings.source_overlay_object_prefix,
            )
        )
    return _gcloud(
        settings,
        worker=0,
        remote_command=_bash_payload(
            ["set -eu", *_clean_checkout_lines(settings), shlex.join(command)]
        ),
    )


def _common_worker_arguments(
    settings: Settings,
    *,
    controller_index: int,
    process_index: int,
) -> list[str]:
    result = [
        "--scientific-model",
        settings.scientific_model,
        "--config",
        settings.config_path,
        "--phase",
        settings.phase,
        "--condition",
        settings.condition,
        "--seed",
        str(settings.seed),
        "--capture-control-root",
        settings.capture_control_root,
        "--object-prefix",
        settings.object_prefix,
        "--resident-cache-root",
        settings.process_cache(process_index),
        "--resident-cache-max-bytes",
        str(settings.cache_max_bytes),
        "--prefetch-receipt-path",
        settings.qualified_prefetch_receipt(controller_index, process_index),
        "--recomposition-sidecar",
        settings.recomposition_sidecar,
        "--source-revision",
        settings.project_commit,
    ]
    if settings.scientific_model == "transcoder":
        if (
            settings.source_overlay_control_root is None
            or settings.source_overlay_object_prefix is None
        ):
            raise OperatorError("transcoder settings lack their source overlay")
        result.extend(
            (
                "--source-overlay-control-root",
                settings.source_overlay_control_root,
                "--source-overlay-object-prefix",
                settings.source_overlay_object_prefix,
            )
        )
    return result


def _prefetch_command(
    settings: Settings,
    *,
    controller_index: int,
    process_index: int,
) -> list[str]:
    stable = settings.qualified_prefetch_receipt(
        controller_index,
        process_index,
    )
    refresh = settings.refresh_prefetch_receipt(
        controller_index,
        process_index,
    )
    cache = settings.process_cache(process_index)
    shared_log = (
        f"{settings.attempt_root}/logs/"
        f"prefetch-c{controller_index:05d}-p{process_index:05d}.log"
    )
    command = [
        settings.remote_uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        "-m",
        "moe_sae.contribution_training_worker",
        "prefetch",
        "--scientific-model",
        settings.scientific_model,
        "--config",
        settings.config_path,
        "--phase",
        settings.phase,
        "--capture-control-root",
        settings.capture_control_root,
        "--object-prefix",
        settings.object_prefix,
        "--resident-cache-root",
        cache,
        "--resident-cache-max-bytes",
        str(settings.cache_max_bytes),
        "--process-count",
        str(PROCESS_COUNT),
        "--process-index",
        str(process_index),
        "--recomposition-sidecar",
        settings.recomposition_sidecar,
        "--source-revision",
        settings.project_commit,
        "--supervision-dir",
        f"{settings.attempt_root}/supervision",
        "--host-id",
        f"controller-c{controller_index:05d}-p{process_index:05d}",
        "--working-directory",
        settings.project_root,
        "--maximum-runtime-seconds",
        str(PREFETCH_SECONDS),
    ]
    if settings.scientific_model == "transcoder":
        if (
            settings.source_overlay_control_root is None
            or settings.source_overlay_object_prefix is None
        ):
            raise OperatorError("transcoder settings lack their source overlay")
        command.extend(
            (
                "--source-overlay-control-root",
                settings.source_overlay_control_root,
                "--source-overlay-object-prefix",
                settings.source_overlay_object_prefix,
            )
        )
    lines = [
        "set -eu",
        *_clean_checkout_lines(settings),
        f"test -s {shlex.quote(settings.attempt_root + '/operator-plan.json')}",
        (
            "install -d -m 0700 "
            f"{shlex.quote(cache)} "
            f"{shlex.quote(str(PurePosixPath(stable).parent))} "
            f"{shlex.quote(str(PurePosixPath(refresh).parent))} "
            f"{shlex.quote(str(PurePosixPath(shared_log).parent))}"
        ),
        (
            f"if test -s {shlex.quote(stable)}; then "
            f"prefetch_receipt={shlex.quote(refresh)}; else "
            f"prefetch_receipt={shlex.quote(stable)}; fi"
        ),
        "set +e",
        "set -C",
        (
            shlex.join(command)
            + ' --prefetch-receipt-path "$prefetch_receipt"'
            + f" > {shlex.quote(shared_log)} 2>&1"
        ),
        "prefetch_status=$?",
        "set +C",
        "set -e",
        'test -s "$prefetch_receipt" || exit 91',
        (
            'printf "terminal prefetch controller=%s process=%s status=%s\\n" '
            f"{controller_index} {process_index} \"$prefetch_status\""
        ),
        'exit "$prefetch_status"',
    ]
    return _gcloud(
        settings,
        worker=controller_index,
        remote_command=_bash_payload(lines),
    )


def _action_command(
    settings: Settings,
    *,
    controller_index: int,
    process_index: int,
) -> list[str]:
    supervision = f"{settings.attempt_root}/supervision"
    shared_log = (
        f"{settings.attempt_root}/logs/"
        f"{settings.action}-c{controller_index:05d}-p{process_index:05d}.log"
    )
    command = [
        settings.remote_uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        "-m",
        "moe_sae.contribution_training_worker",
        settings.action,
        *_common_worker_arguments(
            settings,
            controller_index=controller_index,
            process_index=process_index,
        ),
        "--supervision-dir",
        supervision,
        "--host-id",
        f"controller-c{controller_index:05d}-p{process_index:05d}",
        "--working-directory",
        settings.project_root,
        "--selection-path",
        settings.selection_path,
    ]
    terminal_path = settings.selection_path
    if settings.action == "train":
        command.extend(
            (
                "--checkpoint-control-root",
                settings.checkpoint_control_root,
                "--checkpoint-payload-root",
                settings.checkpoint_payload_root,
                "--maximum-runtime-seconds",
                str(settings.maximum_runtime_seconds),
                "--result-path",
                settings.result_path,
                "--metric-path",
                settings.metric_path,
            )
        )
        if (
            _admission_mode(
                settings.action,
                settings.maximum_runtime_seconds,
            )
            == DIRECT_BOUNDED_ADMISSION
        ):
            command.append("--direct-bounded-resident")
        terminal_path = settings.result_path
    lines = [
        "set -eu",
        *_clean_checkout_lines(settings),
        f"test -s {shlex.quote(settings.attempt_root + '/operator-plan.json')}",
        f"test -s {shlex.quote(settings.qualified_prefetch_receipt(controller_index, process_index))}",
        (
            "install -d -m 0700 "
            f"{shlex.quote(supervision)} "
            f"{shlex.quote(settings.checkpoint_control_root)} "
            f"{shlex.quote(str(PurePosixPath(shared_log).parent))}"
        ),
        "set +e",
        "set -C",
        shlex.join(command) + f" > {shlex.quote(shared_log)} 2>&1",
        "action_status=$?",
        "set +C",
        "set -e",
        (
            f"if test \"$action_status\" -eq 0; then "
            f"test -s {shlex.quote(terminal_path)} || exit 92; fi"
        ),
        (
            'printf "terminal action=%s controller=%s process=%s status=%s\\n" '
            f"{shlex.quote(settings.action)} {controller_index} "
            f"{process_index} \"$action_status\""
        ),
        'exit "$action_status"',
    ]
    return _gcloud(
        settings,
        worker=controller_index,
        remote_command=_bash_payload(lines),
    )


def _run_capture(
    command: Sequence[str],
    log_path: Path,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    with log_path.open("xb") as handle:
        handle.write(completed.stdout)
        handle.flush()
        os.fsync(handle.fileno())
    return completed


def _plan_from_output(payload: bytes) -> dict[str, object]:
    for raw in reversed(payload.decode("utf-8", errors="replace").splitlines()):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and value.get("schema_version") == ATTEMPT_PLAN_SCHEMA_VERSION
        ):
            return value
    raise OperatorError("remote initializer did not return its SAE attempt plan")


def _verify_local_checkout(settings: Settings) -> None:
    repository = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        raise OperatorError("cannot verify the local Git checkout") from error
    if head != settings.project_commit:
        raise OperatorError(f"local HEAD {head!r} differs from PROJECT_COMMIT")
    if status:
        raise OperatorError(
            "local checkout is dirty; commit and deploy the exact operator first"
        )


def _terminate_processes(
    processes: Sequence[tuple[subprocess.Popen[bytes], BinaryIO]],
) -> None:
    pending = [process for process, _handle in processes if process.poll() is None]
    for process in pending:
        process.terminate()
    for process in pending:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _fanout(
    commands: Sequence[Sequence[str]],
    log_paths: Sequence[Path],
    *,
    maximum_wait_seconds: int,
    stage: str,
) -> list[int]:
    if len(commands) != PROCESS_COUNT or len(log_paths) != PROCESS_COUNT:
        raise OperatorError(f"{stage} fanout requires exactly four commands")
    processes: list[tuple[subprocess.Popen[bytes], BinaryIO]] = []
    try:
        for command, log_path in zip(commands, log_paths, strict=True):
            handle = log_path.open("xb")
            try:
                process = subprocess.Popen(
                    list(command),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            except BaseException:
                handle.close()
                _terminate_processes(processes)
                raise
            processes.append((process, handle))
        deadline = time.monotonic() + maximum_wait_seconds
        pending = {process for process, _handle in processes}
        while pending:
            for process in tuple(pending):
                if process.poll() is not None:
                    pending.remove(process)
            if not pending:
                break
            if time.monotonic() >= deadline:
                _terminate_processes(processes)
                raise OperatorError(
                    f"{stage} gcloud transports exceeded "
                    f"{maximum_wait_seconds}s; no later stage was launched"
                )
            time.sleep(0.25)
        return [int(process.returncode) for process, _handle in processes]
    finally:
        for _process, handle in processes:
            handle.close()


def _terminal_worker_status(
    log_path: Path,
    *,
    settings: Settings,
    stage: str,
    controller_index: int,
    process_index: int,
) -> int:
    """Read one exact remote worker terminal marker from an SSH log.

    ``gcloud ... ssh`` can return nonzero after the remote shell has already
    completed successfully.  The remote command prints its marker only after
    the stage's required immutable evidence exists, so the marker—not the
    later local transport teardown—is the workload completion authority.
    """

    if stage == "prefetch":
        marker_start = b"terminal prefetch "
        expected_prefix = (
            f"terminal prefetch controller={controller_index} "
            f"process={process_index} status="
        ).encode("ascii")
    elif stage == "action":
        marker_start = b"terminal action="
        expected_prefix = (
            f"terminal action={settings.action} "
            f"controller={controller_index} process={process_index} status="
        ).encode("ascii")
    else:
        raise OperatorError(f"unknown terminal-marker stage {stage!r}")

    try:
        lines = log_path.read_bytes().splitlines()
    except OSError as error:
        raise OperatorError(
            f"cannot read controller {controller_index} {stage} SSH log: "
            f"{log_path}"
        ) from error
    marker_lines = [line for line in lines if line.startswith(marker_start)]
    if len(marker_lines) != 1:
        raise OperatorError(
            f"controller {controller_index} {stage} SSH log has "
            f"{len(marker_lines)} terminal markers; exactly one is required"
        )
    marker = marker_lines[0]
    if not marker.startswith(expected_prefix):
        raise OperatorError(
            f"controller {controller_index} {stage} SSH terminal marker "
            "does not match the exact action/controller/process"
        )
    raw_status = marker[len(expected_prefix) :]
    if re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})", raw_status) is None:
        raise OperatorError(
            f"controller {controller_index} {stage} SSH terminal marker "
            "has an invalid worker status"
        )
    status = int(raw_status)
    if status > 255:
        raise OperatorError(
            f"controller {controller_index} {stage} SSH terminal marker "
            "status is outside the shell exit-code range"
        )
    return status


def _validated_fanout_completion(
    *,
    settings: Settings,
    mapping: Mapping[int, int],
    stage: str,
    returncodes: Sequence[int],
    log_paths: Sequence[Path],
) -> list[int]:
    """Require four exact remote completions before advancing a stage."""

    if (
        len(mapping) != PROCESS_COUNT
        or len(returncodes) != PROCESS_COUNT
        or len(log_paths) != PROCESS_COUNT
        or set(mapping) != set(range(PROCESS_COUNT))
        or set(mapping.values()) != set(range(PROCESS_COUNT))
    ):
        raise OperatorError(
            f"{stage} completion requires one exact four-controller/process "
            "permutation and exactly four codes and logs"
        )
    statuses = [
        _terminal_worker_status(
            log_path,
            settings=settings,
            stage=stage,
            controller_index=controller_index,
            process_index=process_index,
        )
        for (controller_index, process_index), log_path in zip(
            mapping.items(),
            log_paths,
            strict=True,
        )
    ]
    if any(int(code) != 0 for code in returncodes):
        print(
            "[exp2-sae] local gcloud transport returned nonzero codes "
            f"{list(returncodes)} after all four exact {stage} terminal "
            "markers; using the bound remote worker statuses",
            flush=True,
        )
    return statuses


def _dry_run(settings: Settings) -> int:
    payload = {
        "action": settings.action,
        "run_id": settings.run_id,
        "attempt_id": settings.attempt_id,
        "phase": settings.phase,
        "condition": settings.condition,
        "seed": settings.seed,
        "qualification_hard_deadline_seconds": QUALIFICATION_SECONDS,
        "admission_mode": _admission_mode(
            settings.action,
            settings.maximum_runtime_seconds,
        ),
        "prefetch_hard_deadline_seconds": PREFETCH_SECONDS,
        "mapping_source": settings.qualification_bundle,
        "mapping_policy": (
            "init returns sealed controller_to_jax_process_index; Mac builds "
            "prefetch/action fanout from that exact permutation"
        ),
        "init_attempt": _init_command(settings),
        "fanout_controllers": [
            {
                "controller_index": index,
                "process_index": "<from-sealed-v4-bundle>",
                "stages": ("prefetch", settings.action),
            }
            for index in range(PROCESS_COUNT)
        ],
        "run_root": settings.run_root,
        "attempt_root": settings.attempt_root,
        "selection_path": settings.selection_path,
        "checkpoint_control_root": settings.checkpoint_control_root,
        "checkpoint_payload_root": settings.checkpoint_payload_root,
        "local_logs": str(settings.local_log_root),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run(settings: Settings) -> int:
    if shutil.which("gcloud") is None:
        raise OperatorError("gcloud is unavailable")
    _verify_local_checkout(settings)
    settings.local_log_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    init_log = settings.local_log_root / "init-attempt.ssh.log"
    initialized = _run_capture(_init_command(settings), init_log)
    if initialized.returncode != 0:
        raise OperatorError(
            f"could not seal the SAE attempt plan; inspect {init_log}"
        )
    plan, mapping = _validate_attempt_plan(
        _plan_from_output(initialized.stdout),
        settings,
    )
    local_plan = settings.local_log_root / "attempt-plan.json"
    _atomic_create(local_plan, plan)

    print(
        "[exp2-sae] prefetching mapped process lanes concurrently: "
        + ", ".join(
            f"controller-{controller}->process-{process}"
            for controller, process in mapping.items()
        ),
        flush=True,
    )
    prefetch_logs = [
        settings.local_log_root
        / f"prefetch-c{controller:05d}-p{process:05d}.ssh.log"
        for controller, process in mapping.items()
    ]
    prefetch_codes = _fanout(
        [
            _prefetch_command(
                settings,
                controller_index=controller,
                process_index=process,
            )
            for controller, process in mapping.items()
        ],
        prefetch_logs,
        maximum_wait_seconds=PREFETCH_TRANSPORT_WAIT_SECONDS,
        stage="prefetch",
    )
    prefetch_statuses = _validated_fanout_completion(
        settings=settings,
        mapping=mapping,
        stage="prefetch",
        returncodes=prefetch_codes,
        log_paths=prefetch_logs,
    )
    if any(status != 0 for status in prefetch_statuses):
        raise OperatorError(
            "one or more mapped lane prefetches failed "
            f"(statuses={prefetch_statuses}); accelerator work was not "
            "launched; "
            f"inspect {settings.local_log_root}"
        )

    action_wait = (
        QUALIFICATION_TRANSPORT_WAIT_SECONDS
        if settings.action == "qualify"
        else settings.maximum_runtime_seconds + TRAIN_TRANSPORT_GRACE_SECONDS
    )
    print(
        f"[exp2-sae] launching four mapped {settings.action} workers concurrently",
        flush=True,
    )
    action_logs = [
        settings.local_log_root
        / (
            f"{settings.action}-c{controller:05d}-"
            f"p{process:05d}.ssh.log"
        )
        for controller, process in mapping.items()
    ]
    action_codes = _fanout(
        [
            _action_command(
                settings,
                controller_index=controller,
                process_index=process,
            )
            for controller, process in mapping.items()
        ],
        action_logs,
        maximum_wait_seconds=action_wait,
        stage=settings.action,
    )
    action_statuses = _validated_fanout_completion(
        settings=settings,
        mapping=mapping,
        stage="action",
        returncodes=action_codes,
        log_paths=action_logs,
    )
    if any(status != 0 for status in action_statuses):
        raise OperatorError(
            f"one or more mapped {settings.action} workers failed "
            f"(statuses={action_statuses}); inspect "
            f"{settings.local_log_root} and {settings.attempt_root}"
        )
    terminal = (
        settings.selection_path
        if settings.action == "qualify"
        else settings.result_path
    )
    print(
        f"[exp2-sae] {settings.action.upper()} COMPLETE\n"
        f"controller/JAX map: {mapping}\n"
        f"terminal evidence: {terminal}\n"
        f"shared attempt: {settings.attempt_root}\n"
        f"Mac logs: {settings.local_log_root}",
        flush=True,
    )
    return 0


def _public_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map, prefetch, and fan out one Experiment-2 SAE qualification "
            "or training attempt"
        )
    )
    parser.add_argument("--action", choices=ACTIONS, required=True)
    parser.add_argument(
        "--scientific-model",
        choices=SCIENTIFIC_MODELS,
        default="sae",
    )
    parser.add_argument("--phase", required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--maximum-runtime-seconds", type=int, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the init and mapped-fanout contract without invoking gcloud",
    )
    return parser


def _init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--action", choices=ACTIONS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument(
        "--scientific-model",
        choices=SCIENTIFIC_MODELS,
        default="sae",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--capture-control-root", required=True)
    parser.add_argument("--object-prefix", required=True)
    parser.add_argument("--source-overlay-control-root")
    parser.add_argument("--source-overlay-object-prefix")
    parser.add_argument("--recomposition-sidecar", required=True)
    parser.add_argument("--qualification-bundle", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cache-max-bytes", type=int, required=True)
    parser.add_argument("--sae-control-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--attempt-root", required=True)
    parser.add_argument("--checkpoint-payload-root", required=True)
    parser.add_argument("--maximum-runtime-seconds", type=int, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    selected = list(sys.argv[1:] if argv is None else argv)
    try:
        if selected and selected[0] == "_init-attempt":
            args = _init_parser().parse_args(selected[1:])
            plan = initialize_attempt(args)
            print(
                _canonical_json_bytes(plan).decode("ascii"),
                flush=True,
            )
            return 0
        args = _public_parser().parse_args(selected)
        settings = _settings(os.environ if environment is None else environment, args)
        if args.dry_run:
            return _dry_run(settings)
        return run(settings)
    except (OperatorError, OSError, ValueError) as error:
        print(f"[exp2-sae] ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
