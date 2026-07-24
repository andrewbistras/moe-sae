#!/usr/bin/env python3
"""Qualification-only four-controller launcher for Experiment 2.

Run this executable from zsh on the operator machine.  It starts the same
bounded ``launch-host`` command on TPU workers 0--3 concurrently, waits for
all four terminal records, and asks transport controller 0 to seal the exact
aggregate.  TPU worker/controller ordinals address SSH and control paths only;
each child records its independently observed JAX process index.  It stops
after the first passing receipt and contains no workload launch path.
"""

from __future__ import annotations

import argparse
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
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

QUALIFICATION_SECONDS = 600
TRANSPORT_WAIT_SECONDS = 720
PROCESS_COUNT = 4
MESH = "ep8-fsdp2-tp1"

PASS_EXIT = 0
FALLBACK_EXIT = 20
ABORT_EXIT = 30


@dataclass(frozen=True, slots=True)
class Candidate:
    identifier: str
    per_device_sequence_batch: int

    @property
    def global_sequence_batch(self) -> int:
        return self.per_device_sequence_batch * 16

    @property
    def declaration(self) -> str:
        return (
            f"{self.identifier}:{MESH}:{self.per_device_sequence_batch}:"
            f"{self.global_sequence_batch}"
        )


# This is intentionally not a Cartesian benchmark sweep.  The known-best mesh
# is held fixed while static batch falls only when a previous candidate has an
# explicit, hash-verified OOM or performance-headroom admission failure.
CANDIDATES = (
    Candidate("b128-ep8-fsdp2", 8),
    Candidate("b64-ep8-fsdp2", 4),
    Candidate("b32-ep8-fsdp2", 2),
)


class OperatorError(RuntimeError):
    """The qualification operator cannot continue safely."""


@dataclass(frozen=True, slots=True)
class Settings:
    gcp_project: str
    tpu_name: str
    tpu_zone: str
    project_commit: str
    project_root: str
    local_buffer_root: str
    artifact_root: str
    model_checkpoint: str
    checkpoint_sha256: str
    remote_uv: str
    ssh_key_file: str | None
    run_id: str
    local_log_root: Path

    @property
    def config_path(self) -> str:
        return f"{self.project_root}/configs/gemma4_contribution_v1.yaml"

    @property
    def shared_run_root(self) -> str:
        return f"{self.artifact_root}/experiment2/qualification/{self.run_id}"

    @property
    def local_run_root(self) -> str:
        return f"{self.local_buffer_root}/experiment2/qualification/{self.run_id}"


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "")
    if not value:
        raise OperatorError(f"required environment variable is unset: {name}")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise OperatorError(f"{name} contains an unsafe control character")
    return value


def _absolute_remote_path(name: str, value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or value == "/":
        raise OperatorError(f"{name} must be a non-root absolute POSIX path")
    return str(path)


def _settings(environment: Mapping[str, str]) -> Settings:
    project = _required(environment, "GCP_PROJECT")
    tpu_name = _required(environment, "TPU_NAME")
    zone = _required(environment, "TPU_ZONE")
    commit = _required(environment, "PROJECT_COMMIT")
    checkpoint_sha256 = _required(environment, "CHECKPOINT_SHA256")
    if re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project) is None:
        raise OperatorError("GCP_PROJECT is invalid")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", tpu_name) is None:
        raise OperatorError("TPU_NAME is invalid")
    if re.fullmatch(r"[a-z0-9-]+", zone) is None:
        raise OperatorError("TPU_ZONE is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise OperatorError("PROJECT_COMMIT must be a full lowercase Git SHA")
    if re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None:
        raise OperatorError("CHECKPOINT_SHA256 must be lowercase hexadecimal")

    run_id = environment.get("QUALIFICATION_RUN_ID") or (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + commit[:12]
    )
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id) is None:
        raise OperatorError("QUALIFICATION_RUN_ID is invalid")
    default_log_root = (
        Path.cwd()
        / "artifacts"
        / "gemma4_v1"
        / "operator-results"
        / "experiment2-qualification"
        / run_id
    )
    log_root = Path(
        environment.get("QUALIFICATION_LOCAL_LOG_ROOT", str(default_log_root))
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
        local_buffer_root=_absolute_remote_path(
            "LOCAL_BUFFER_ROOT",
            _required(environment, "LOCAL_BUFFER_ROOT"),
        ),
        artifact_root=_absolute_remote_path(
            "ARTIFACT_ROOT",
            _required(environment, "ARTIFACT_ROOT"),
        ),
        model_checkpoint=_required(environment, "MODEL_CHECKPOINT"),
        checkpoint_sha256=checkpoint_sha256,
        remote_uv=_absolute_remote_path(
            "REMOTE_UV",
            _required(environment, "REMOTE_UV"),
        ),
        ssh_key_file=environment.get("SSH_KEY_FILE"),
        run_id=run_id,
        local_log_root=log_root,
    )


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
        "--ssh-flag=-oConnectTimeout=30",
        "--ssh-flag=-oServerAliveInterval=30",
        "--ssh-flag=-oServerAliveCountMax=4",
        "--quiet",
    ]
    if settings.ssh_key_file:
        command.insert(-1, f"--ssh-key-file={settings.ssh_key_file}")
    return command


def _bash_payload(lines: Sequence[str]) -> str:
    return "bash -lc " + shlex.quote("\n".join(lines))


def _qualification_command(
    settings: Settings,
    candidate: Candidate,
    controller_index: int,
) -> tuple[list[str], str, str]:
    candidate_control = f"{settings.shared_run_root}/{candidate.identifier}"
    candidate_local = f"{settings.local_run_root}/{candidate.identifier}"
    # Keep the persistent XLA cache stable across immutable operator attempts
    # and identical to the production harvester path.  Cache keys still bind
    # the executable; the path itself is host-local and never authorizes work.
    cache = (
        f"{settings.local_buffer_root}/experiment2/jax-cache/"
        f"{settings.project_commit}/{MESH}/"
        f"b{candidate.per_device_sequence_batch}/"
        f"controller-c{controller_index:05d}"
    )
    scratch = f"{candidate_local}/scratch"
    evidence = (
        f"{candidate_control}/controller-c{controller_index:05d}/"
        "host-evidence.json"
    )
    qualification = [
        settings.remote_uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        "-m",
        "moe_sae.contribution_qualification_worker",
        "launch-host",
        "--config",
        settings.config_path,
        "--checkpoint",
        settings.model_checkpoint,
        "--checkpoint-sha256",
        settings.checkpoint_sha256,
        "--repo-root",
        settings.project_root,
        "--jax-cache-dir",
        cache,
        "--control-root",
        candidate_control,
        "--scratch-root",
        scratch,
        "--mesh",
        MESH,
        "--per-device-sequence-batch",
        str(candidate.per_device_sequence_batch),
        "--controller-index",
        str(controller_index),
        "--warmup-steps",
        "5",
        "--writer-disabled-steps",
        "10",
        "--measured-steps",
        "10",
        "--maximum-seconds",
        str(QUALIFICATION_SECONDS),
        "--termination-grace-seconds",
        "10",
    ]
    lines = [
        "set -u",
        f"cd {shlex.quote(settings.project_root)}",
        (
            'test "$(git rev-parse HEAD)" = '
            f"{shlex.quote(settings.project_commit)}"
        ),
        'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
        (
            "install -d -m 0700 "
            f"{shlex.quote(candidate_control)} "
            f"{shlex.quote(cache)} "
            f"{shlex.quote(scratch)}"
        ),
        "set +e",
        shlex.join(qualification),
        "qualification_status=$?",
        "set -e",
        (
            f"if test ! -s {shlex.quote(evidence)}; then "
            'echo "missing terminal host evidence" >&2; '
            "exit 91; fi"
        ),
        (
            'printf "terminal candidate=%s controller=%s child_status=%s\\n" '
            f"{shlex.quote(candidate.identifier)} "
            f"{controller_index} \"$qualification_status\""
        ),
        "exit 0",
    ]
    return (
        _gcloud(
            settings,
            worker=controller_index,
            remote_command=_bash_payload(lines),
        ),
        evidence,
        candidate_control,
    )


def _plan_command(settings: Settings) -> list[str]:
    plan_path = f"{settings.shared_run_root}/operator-plan.json"
    helper = f"{settings.project_root}/scripts/tpu/contribution_qualification_decide.py"
    command = [
        settings.remote_uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        helper,
        "init-plan",
        "--run-id",
        settings.run_id,
        "--project-commit",
        settings.project_commit,
        "--config",
        settings.config_path,
        "--checkpoint",
        settings.model_checkpoint,
        "--checkpoint-sha256",
        settings.checkpoint_sha256,
    ]
    for candidate in CANDIDATES:
        command.extend(("--candidate", candidate.declaration))
    command.extend(("--output", plan_path))
    payload = _bash_payload(
        [
            "set -eu",
            f"cd {shlex.quote(settings.project_root)}",
            f"install -d -m 0700 {shlex.quote(settings.shared_run_root)}",
            shlex.join(command),
        ]
    )
    return _gcloud(settings, worker=0, remote_command=payload)


def _decision_command(
    settings: Settings,
    candidate: Candidate,
    evidence_paths: Sequence[str],
    candidate_control: str,
) -> list[str]:
    helper = f"{settings.project_root}/scripts/tpu/contribution_qualification_decide.py"
    command = [
        settings.remote_uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        helper,
        "decide",
        "--candidate-id",
        candidate.identifier,
        "--mesh",
        MESH,
        "--per-device-sequence-batch",
        str(candidate.per_device_sequence_batch),
        "--batch-sequences",
        str(candidate.global_sequence_batch),
    ]
    for path in evidence_paths:
        command.extend(("--host-evidence", path))
    command.extend(
        (
            "--output",
            f"{candidate_control}/coordinator-bundle.json",
            "--decision",
            f"{candidate_control}/operator-decision.json",
            "--qualification-receipt",
            f"{candidate_control}/qualification-receipt.json",
            "--recomposition-sidecar",
            f"{candidate_control}/recomposition-sidecar.json",
        )
    )
    return _gcloud(
        settings,
        worker=0,
        remote_command=_bash_payload(
            [
                "set -u",
                f"cd {shlex.quote(settings.project_root)}",
                shlex.join(command),
            ]
        ),
    )


def _run_logged(
    command: Sequence[str],
    log_path: Path,
) -> subprocess.CompletedProcess[bytes]:
    with log_path.open("xb") as log:
        return subprocess.run(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )


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
        raise OperatorError(
            f"local HEAD {head!r} differs from PROJECT_COMMIT"
        )
    if status:
        raise OperatorError(
            "local checkout is dirty; commit and deploy the exact operator "
            "before qualification"
        )


def _fanout(
    commands: Sequence[Sequence[str]],
    log_paths: Sequence[Path],
) -> list[int]:
    if len(commands) != PROCESS_COUNT or len(log_paths) != PROCESS_COUNT:
        raise OperatorError("qualification fanout requires exactly four commands")
    processes: list[tuple[subprocess.Popen[bytes], object]] = []
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
                raise
            processes.append((process, handle))
        deadline = time.monotonic() + TRANSPORT_WAIT_SECONDS
        pending = {process for process, _ in processes}
        while pending:
            for process in tuple(pending):
                if process.poll() is not None:
                    pending.remove(process)
            if not pending:
                break
            if time.monotonic() >= deadline:
                for process in pending:
                    process.terminate()
                for process in pending:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise OperatorError(
                    "gcloud transport exceeded 720 seconds; no fallback was "
                    "started, and each remote supervisor retains its own "
                    "600-second hard deadline"
                )
            time.sleep(0.25)
        return [int(process.returncode) for process, _ in processes]
    finally:
        for _, handle in processes:
            handle.close()


def _terminal_child_status(
    log_path: Path,
    *,
    candidate: Candidate,
    controller_index: int,
) -> int:
    """Read one exact remote terminal marker from a completed SSH log.

    ``gcloud ... ssh`` can return a local transport status of one after the
    remote shell has completed successfully.  The marker authorizes only a
    coordinator validation attempt; it never substitutes for the immutable,
    self-hashed host evidence checked by that coordinator.
    """

    try:
        lines = log_path.read_bytes().splitlines()
    except OSError as error:
        raise OperatorError(
            f"cannot read controller {controller_index} SSH log: {log_path}"
        ) from error
    marker_lines = [
        line for line in lines if line.startswith(b"terminal candidate=")
    ]
    if len(marker_lines) != 1:
        raise OperatorError(
            f"controller {controller_index} SSH log has "
            f"{len(marker_lines)} terminal markers; exactly one is required"
        )
    expected_prefix = (
        f"terminal candidate={candidate.identifier} "
        f"controller={controller_index} child_status="
    ).encode("ascii")
    marker = marker_lines[0]
    if not marker.startswith(expected_prefix):
        raise OperatorError(
            f"controller {controller_index} SSH terminal marker does not "
            "match the exact candidate/controller"
        )
    raw_status = marker[len(expected_prefix) :]
    if re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})", raw_status) is None:
        raise OperatorError(
            f"controller {controller_index} SSH terminal marker has an "
            "invalid child status"
        )
    status = int(raw_status)
    if status > 255:
        raise OperatorError(
            f"controller {controller_index} SSH terminal marker child status "
            "is outside the shell exit-code range"
        )
    return status


def _validated_transport_completion(
    *,
    candidate: Candidate,
    returncodes: Sequence[int],
    log_paths: Sequence[Path],
) -> list[int]:
    """Require exact remote completion before attempting coordinator checks."""

    if len(returncodes) != PROCESS_COUNT or len(log_paths) != PROCESS_COUNT:
        raise OperatorError(
            "transport completion requires exactly four codes and logs"
        )
    child_statuses = [
        _terminal_child_status(
            log_path,
            candidate=candidate,
            controller_index=controller_index,
        )
        for controller_index, log_path in enumerate(log_paths)
    ]
    nonzero = [int(code) for code in returncodes if int(code) != 0]
    if nonzero:
        print(
            "[exp2-qualification] local gcloud transport returned nonzero "
            f"codes {list(returncodes)} after all four exact terminal markers; "
            "continuing only to immutable coordinator validation",
            flush=True,
        )
    return child_statuses


def _dry_run(settings: Settings) -> int:
    payload: dict[str, object] = {
        "run_id": settings.run_id,
        "hard_deadline_seconds": QUALIFICATION_SECONDS,
        "process_count": PROCESS_COUNT,
        "terminal_action": "stop_after_first_pass_no_workload_launch",
        "candidates": [],
    }
    candidates: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        commands = [
            _qualification_command(settings, candidate, index)[0]
            for index in range(PROCESS_COUNT)
        ]
        candidates.append(
            {
                "candidate_id": candidate.identifier,
                "mesh": MESH,
                "per_device_sequence_batch": (
                    candidate.per_device_sequence_batch
                ),
                "global_sequence_batch": candidate.global_sequence_batch,
                "fanout": commands,
            }
        )
    payload["candidates"] = candidates
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run(settings: Settings) -> int:
    if shutil.which("gcloud") is None:
        raise OperatorError("gcloud is unavailable")
    _verify_local_checkout(settings)
    settings.local_log_root.mkdir(parents=True, exist_ok=False)
    plan_log = settings.local_log_root / "plan.log"
    plan = _run_logged(_plan_command(settings), plan_log)
    if plan.returncode != 0:
        raise OperatorError(
            f"could not seal the immutable operator plan; inspect {plan_log}"
        )

    for candidate in CANDIDATES:
        candidate_logs = settings.local_log_root / candidate.identifier
        candidate_logs.mkdir(mode=0o700)
        built = [
            _qualification_command(settings, candidate, index)
            for index in range(PROCESS_COUNT)
        ]
        commands = [item[0] for item in built]
        evidence_paths = [item[1] for item in built]
        candidate_control = built[0][2]
        print(
            "[exp2-qualification] launching four controllers concurrently: "
            f"{candidate.identifier} {MESH} "
            f"per-device={candidate.per_device_sequence_batch} "
            f"global-B={candidate.global_sequence_batch}",
            flush=True,
        )
        log_paths = [
            candidate_logs / f"host-{index:05d}.ssh.log"
            for index in range(PROCESS_COUNT)
        ]
        returncodes = _fanout(commands, log_paths)
        _validated_transport_completion(
            candidate=candidate,
            returncodes=returncodes,
            log_paths=log_paths,
        )

        decision_log = candidate_logs / "decision.log"
        decision = _run_logged(
            _decision_command(
                settings,
                candidate,
                evidence_paths,
                candidate_control,
            ),
            decision_log,
        )
        if decision.returncode == PASS_EXIT:
            print(
                "[exp2-qualification] PASSED; no workload was launched.\n"
                f"receipt bundle: {candidate_control}/coordinator-bundle.json\n"
                f"workload receipt: "
                f"{candidate_control}/qualification-receipt.json\n"
                f"recomposition sidecar: "
                f"{candidate_control}/recomposition-sidecar.json\n"
                f"shared evidence: {candidate_control}\n"
                f"operator logs: {candidate_logs}",
                flush=True,
            )
            return PASS_EXIT
        if decision.returncode == FALLBACK_EXIT:
            print(
                "[exp2-qualification] candidate received an explicit bounded "
                f"fallback decision; inspect {candidate_control}/"
                "operator-decision.json",
                flush=True,
            )
            continue
        raise OperatorError(
            "candidate decision was not an explicit OOM/headroom fallback "
            f"(code={decision.returncode}); aborting; inspect {decision_log}; "
            "sealed decision (when available): "
            f"{candidate_control}/operator-decision.json"
        )

    raise OperatorError(
        "all three declared candidates exhausted explicit fallback admission; "
        "no workload was launched"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fan out only the bounded Experiment-2 qualification and stop "
            "after the first passing receipt"
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print exact fanout commands without invoking gcloud",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = _settings(os.environ if environment is None else environment)
        if args.dry_run:
            return _dry_run(settings)
        return run(settings)
    except (OperatorError, OSError, ValueError) as error:
        print(f"[exp2-qualification] ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
