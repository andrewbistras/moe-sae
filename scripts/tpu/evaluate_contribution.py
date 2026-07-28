#!/usr/bin/env python3
"""Fan out the Experiment-2 final evaluator across four TPU controllers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Sequence

from moe_sae.runtime import (
    REGIONAL_ACCELERATOR_TYPES,
    SUPPORTED_TRAINING_ACCELERATOR_TYPES,
    V5LITEPOD_DEVICE_KIND_ENV,
    RuntimeDependencyError,
    accelerator_peak_flops_per_second,
    v5litepod_device_kind_from_environment,
)

PROCESS_COUNT = 4
V4_32_PEAK_FLOPS_PER_SECOND = 4_400_000_000_000_000
REGIONAL_PROCESS_CACHE_MAX_BYTES = 160 * 1024**3
_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SSH_USER = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
RESULT_SCHEMA_VERSION = "gemma4-contribution-phase2-scientific-result-v2"
LEARNING_HORIZON_RESULT_SCHEMA_VERSION = (
    "gemma4-contribution-learning-horizon-curve-point-v1"
)
LEARNING_HORIZON_PROTOCOL_ID = "gemma4-pilot-learning-horizon-v1-20260728"


class OperatorError(RuntimeError):
    pass


def _required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise OperatorError(f"required environment variable {name} is unset")
    return value


@dataclass(frozen=True)
class Settings:
    project: str
    tpu: str
    zone: str
    commit: str
    root: str
    uv: str
    bundle: str
    sidecar: str
    train_capture: str
    eval_capture: str
    train_objects: str
    eval_objects: str
    cache_root: str
    cache_bytes: int
    sae_control: str
    checkpoint_root: str
    result_root: str
    log_root: Path
    ssh_key: str | None
    accelerator_type: str = "v4-32"
    peak_flops_per_second: int = V4_32_PEAK_FLOPS_PER_SECOND
    regional_mirror_seal: str | None = None
    internal_ip: bool = False
    config_path: str | None = None
    phase: str = "phase2_mechanism_pilot"
    seed: int = 0
    checkpoint_tokens: int | None = None
    run_roots: dict[str, str] | None = None
    ssh_user: str | None = None
    v5litepod_device_kind: str | None = None

    @classmethod
    def from_environment(cls) -> Settings:
        commit = _required("PROJECT_COMMIT")
        if _SHA.fullmatch(commit) is None:
            raise OperatorError("PROJECT_COMMIT must be a full Git SHA")
        cache_bytes = int(_required("CONTRIBUTION_SAE_CACHE_MAX_BYTES"))
        if cache_bytes <= 0:
            raise OperatorError("CONTRIBUTION_SAE_CACHE_MAX_BYTES must be positive")
        artifact_root = _required("ARTIFACT_ROOT")
        bulk_root = _required("BULK_GCS_ROOT").rstrip("/")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        accelerator_type = os.environ.get("TPU_ACCELERATOR_TYPE", "v4-32")
        if accelerator_type not in SUPPORTED_TRAINING_ACCELERATOR_TYPES:
            raise OperatorError("TPU_ACCELERATOR_TYPE is unsupported")
        try:
            v5litepod_device_kind = (
                v5litepod_device_kind_from_environment()
                if accelerator_type == "v5litepod-16"
                else None
            )
        except RuntimeDependencyError as error:
            raise OperatorError(str(error)) from error
        mapping_receipt = (
            _required("CONTRIBUTION_FRESH_CONTROLLER_MAPPING")
            if accelerator_type in REGIONAL_ACCELERATOR_TYPES
            else _required("CONTRIBUTION_QUALIFICATION_BUNDLE")
        )
        regional_mirror = os.environ.get("CONTRIBUTION_REGIONAL_MIRROR_SEAL")
        if accelerator_type in REGIONAL_ACCELERATOR_TYPES and not regional_mirror:
            raise OperatorError(
                f"{accelerator_type} evaluation requires a regional mirror seal"
            )
        if (
            accelerator_type in REGIONAL_ACCELERATOR_TYPES
            and cache_bytes != REGIONAL_PROCESS_CACHE_MAX_BYTES
        ):
            raise OperatorError(
                f"{accelerator_type} cache bound is fixed at 160GiB"
            )
        internal_ip_raw = os.environ.get("TPU_USE_INTERNAL_IP", "0")
        if internal_ip_raw not in {"0", "1"}:
            raise OperatorError("TPU_USE_INTERNAL_IP must be exactly 0 or 1")
        internal_ip = internal_ip_raw == "1"
        if accelerator_type in REGIONAL_ACCELERATOR_TYPES and not internal_ip:
            raise OperatorError(
                f"{accelerator_type} evaluation requires explicit "
                "TPU_USE_INTERNAL_IP=1"
            )
        ssh_user = os.environ.get("TPU_SSH_USER") or None
        if ssh_user is not None and _SAFE_SSH_USER.fullmatch(ssh_user) is None:
            raise OperatorError(
                "TPU_SSH_USER must be a safe lowercase Linux username"
            )
        return cls(
            project=_required("GCP_PROJECT"),
            tpu=_required("TPU_NAME"),
            zone=_required("TPU_ZONE"),
            commit=commit,
            root=_required("TPU_PROJECT_ROOT"),
            uv=_required("REMOTE_UV"),
            bundle=mapping_receipt,
            sidecar=_required("CONTRIBUTION_RECOMPOSITION_SIDECAR"),
            train_capture=_required("CONTRIBUTION_TRAIN_CAPTURE_ROOT"),
            eval_capture=os.environ.get(
                "CONTRIBUTION_EVAL_CAPTURE_ROOT",
                (
                    f"{artifact_root}/experiment2/contributions/"
                    "phase2_mechanism_pilot/eval"
                ),
            ),
            train_objects=_required("CONTRIBUTION_OBJECT_PREFIX"),
            eval_objects=os.environ.get(
                "CONTRIBUTION_EVAL_OBJECT_PREFIX",
                f"{bulk_root}/experiment2/contributions/eval",
            ),
            cache_root=_required("CONTRIBUTION_SAE_CACHE_ROOT"),
            cache_bytes=cache_bytes,
            sae_control=_required("CONTRIBUTION_SAE_CONTROL_ROOT"),
            checkpoint_root=_required("CONTRIBUTION_SAE_CHECKPOINT_ROOT"),
            result_root=os.environ.get(
                "CONTRIBUTION_EVALUATION_ROOT",
                f"{artifact_root}/experiment2/evaluation/phase2-seed0",
            ),
            log_root=Path(
                os.environ.get(
                    "CONTRIBUTION_EVALUATION_LOG_ROOT",
                    (
                        f"{artifact_root}/experiment2/evaluation/"
                        f"operator-{stamp}"
                    ),
                )
            ),
            ssh_key=os.environ.get("CONTRIBUTION_SSH_KEY_FILE") or None,
            accelerator_type=accelerator_type,
            peak_flops_per_second=accelerator_peak_flops_per_second(
                accelerator_type
            ),
            regional_mirror_seal=regional_mirror or None,
            internal_ip=internal_ip,
            ssh_user=ssh_user,
            v5litepod_device_kind=v5litepod_device_kind,
        )


def _mapping(bundle_path: str, *, accelerator_type: str = "v4-32") -> dict[int, int]:
    try:
        bundle = json.loads(Path(bundle_path).read_text(encoding="ascii"))
        if accelerator_type in REGIONAL_ACCELERATOR_TYPES:
            from moe_sae.regional_execution import (
                validate_controller_mapping_receipt,
            )

            bundle = validate_controller_mapping_receipt(bundle)
            if (
                bundle["accelerator_type"] != accelerator_type
                or bundle["process_count"] != PROCESS_COUNT
                or bundle["global_device_count"] != 16
                or bundle["local_device_count"] != 4
            ):
                raise OperatorError(
                    f"fresh {accelerator_type} controller mapping topology changed"
                )
        rows = bundle["controller_to_jax_process_index"]
        result = {
            int(row["controller_index"]): int(row["process_index"])
            for row in rows
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise OperatorError("qualification bundle has no controller mapping") from error
    if set(result) != set(range(PROCESS_COUNT)) or set(result.values()) != set(
        range(PROCESS_COUNT)
    ):
        raise OperatorError("controller/JAX process mapping is not a permutation")
    return result


def _worker(settings: Settings, process: int) -> list[str]:
    worker = [
        settings.uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        "-m",
        "moe_sae.contribution_evaluation_worker",
        "--config",
        (
            settings.config_path
            or f"{settings.root}/configs/gemma4_contribution_v1.yaml"
        ),
        "--accelerator-type",
        settings.accelerator_type,
        "--peak-flops-per-second",
        str(settings.peak_flops_per_second),
        "--phase",
        settings.phase,
        "--seed",
        str(settings.seed),
        "--train-capture-root",
        settings.train_capture,
        "--eval-capture-root",
        settings.eval_capture,
        "--train-object-prefix",
        settings.train_objects,
        "--eval-object-prefix",
        settings.eval_objects,
        "--cache-root",
        settings.cache_root,
        "--cache-max-bytes",
        str(settings.cache_bytes),
        "--sae-control-root",
        settings.sae_control,
        "--checkpoint-payload-root",
        settings.checkpoint_root,
        "--recomposition-sidecar",
        settings.sidecar,
        "--result-root",
        settings.result_root,
        "--process-index",
        str(process),
    ]
    if settings.v5litepod_device_kind is not None:
        worker[:0] = [
            "env",
            (
                f"{V5LITEPOD_DEVICE_KIND_ENV}="
                f"{settings.v5litepod_device_kind}"
            ),
        ]
    if settings.checkpoint_tokens is not None:
        worker.extend(
            ("--checkpoint-tokens", str(settings.checkpoint_tokens))
        )
    for condition, run_root in (settings.run_roots or {}).items():
        worker.extend((f"--run-root-{condition.lower()}", run_root))
    if settings.regional_mirror_seal is not None:
        worker.extend(
            ("--regional-mirror-seal", settings.regional_mirror_seal)
        )
    return worker


def _command(settings: Settings, controller: int, process: int) -> list[str]:
    worker = _worker(settings, process)
    body = "\n".join(
        (
            "set -euo pipefail",
            f"cd {shlex.quote(settings.root)}",
            f'test "$(git rev-parse HEAD)" = {shlex.quote(settings.commit)}',
            'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
            "set +e",
            shlex.join(worker),
            "evaluation_status=$?",
            "set -e",
            (
                'if test "$evaluation_status" -ne 0; then '
                'exit "$evaluation_status"; fi'
            ),
            (
                "test -s "
                f"{shlex.quote(settings.result_root.rstrip('/') + '/scientific-result.json')}"
            ),
            (
                'printf "terminal action=evaluate controller=%s process=%s '
                'status=%s\\n" '
                f"{controller} {process} \"$evaluation_status\""
            ),
            'exit "$evaluation_status"',
        )
    )
    target = (
        f"{settings.ssh_user}@{settings.tpu}"
        if settings.ssh_user is not None
        else settings.tpu
    )
    command = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        target,
        f"--project={settings.project}",
        f"--zone={settings.zone}",
        f"--worker={controller}",
        "--quiet",
        "--command",
        "bash -lc " + shlex.quote(body),
    ]
    if settings.ssh_key:
        command.insert(-2, f"--ssh-key-file={settings.ssh_key}")
    if settings.internal_ip:
        command.insert(-2, "--internal-ip")
    return command


def _run(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for name in ("LD_PRELOAD", "GLIBCXX_FORCE_NEW", "GLIBCPP_FORCE_NEW"):
        environment.pop(name, None)
    with log.open("wb") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        try:
            return process.wait(timeout=7_500)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            return 124


def _terminal_evaluation_status(
    log: Path,
    *,
    controller: int,
    process: int,
) -> int:
    try:
        lines = log.read_bytes().splitlines()
    except OSError as error:
        raise OperatorError(f"cannot read evaluation SSH log: {log}") from error
    markers = [
        line for line in lines if line.startswith(b"terminal action=evaluate ")
    ]
    if len(markers) != 1:
        raise OperatorError(
            f"controller {controller} evaluation log has {len(markers)} "
            "terminal markers; exactly one is required"
        )
    prefix = (
        f"terminal action=evaluate controller={controller} "
        f"process={process} status="
    ).encode("ascii")
    marker = markers[0]
    if not marker.startswith(prefix):
        raise OperatorError(
            f"controller {controller} evaluation terminal marker is not "
            "bound to the expected controller/process"
        )
    raw_status = marker[len(prefix) :]
    if re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})", raw_status) is None:
        raise OperatorError("evaluation terminal marker has an invalid status")
    status = int(raw_status)
    if status > 255:
        raise OperatorError(
            "evaluation terminal marker status is outside the shell range"
        )
    return status


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise OperatorError("scientific result is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _validate_scientific_result(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperatorError(f"cannot read scientific result: {path}") from error
    if not isinstance(value, dict):
        raise OperatorError("scientific result must be a JSON object")
    body = dict(value)
    claimed = body.pop("result_sha256", None)
    schema = body.get("schema_version")
    schema_is_valid = schema == RESULT_SCHEMA_VERSION or (
        schema == LEARNING_HORIZON_RESULT_SCHEMA_VERSION
        and body.get("protocol_id") == LEARNING_HORIZON_PROTOCOL_ID
    )
    if (
        not schema_is_valid
        or not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or _canonical_sha256(body) != claimed
    ):
        raise OperatorError("scientific result identity or self-hash is invalid")
    return value


def _absolute_remote_path(name: str, value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or value == "/":
        raise OperatorError(f"{name} must be a non-root absolute POSIX path")
    return str(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help=(
            "absolute finalized config path on every TPU VM; defaults to "
            "configs/gemma4_contribution_v1.yaml"
        ),
    )
    parser.add_argument("--phase", default="phase2_mechanism_pilot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-tokens", type=int)
    parser.add_argument("--result-root")
    for condition in ("A", "R", "B", "F", "Q"):
        parser.add_argument(f"--run-root-{condition.lower()}")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the evaluator worker contract without invoking gcloud",
    )
    return parser


def _apply_public_arguments(
    settings: Settings,
    args: argparse.Namespace,
) -> Settings:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.phase) is None:
        raise OperatorError("--phase is invalid")
    if args.seed < 0:
        raise OperatorError("--seed must be nonnegative")
    if args.checkpoint_tokens is not None and args.checkpoint_tokens <= 0:
        raise OperatorError("--checkpoint-tokens must be positive")
    run_roots = dict(settings.run_roots or {})
    for condition in ("A", "R", "B", "F", "Q"):
        selected = getattr(args, f"run_root_{condition.lower()}")
        if selected is not None:
            run_roots[condition] = _absolute_remote_path(
                f"--run-root-{condition.lower()}",
                selected,
            )
    return replace(
        settings,
        config_path=(
            _absolute_remote_path("--config", args.config)
            if args.config is not None
            else settings.config_path
        ),
        phase=args.phase,
        seed=args.seed,
        checkpoint_tokens=args.checkpoint_tokens,
        run_roots=run_roots or None,
        result_root=(
            _absolute_remote_path("--result-root", args.result_root)
            if args.result_root is not None
            else settings.result_root
        ),
    )


def main(argv: Sequence[str] = ()) -> int:
    args = _parser().parse_args(argv)
    settings = _apply_public_arguments(Settings.from_environment(), args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "checkpoint_tokens": settings.checkpoint_tokens,
                    "config": (
                        settings.config_path
                        or (
                            f"{settings.root}/configs/"
                            "gemma4_contribution_v1.yaml"
                        )
                    ),
                    "phase": settings.phase,
                    "result_root": settings.result_root,
                    "run_roots": settings.run_roots,
                    "seed": settings.seed,
                    "worker_process_zero": _worker(settings, 0),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    mapping = (
        _mapping(settings.bundle, accelerator_type=settings.accelerator_type)
        if settings.accelerator_type in REGIONAL_ACCELERATOR_TYPES
        else _mapping(settings.bundle)
    )
    settings.log_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    logs = {
        controller: settings.log_root
        / f"evaluation-c{controller:05d}-p{process:05d}.log"
        for controller, process in mapping.items()
    }
    with ThreadPoolExecutor(max_workers=PROCESS_COUNT) as pool:
        futures = {
            controller: pool.submit(
                _run,
                _command(settings, controller, process),
                logs[controller],
            )
            for controller, process in mapping.items()
        }
        codes = {controller: future.result() for controller, future in futures.items()}
    statuses = {
        controller: _terminal_evaluation_status(
            logs[controller],
            controller=controller,
            process=process,
        )
        for controller, process in mapping.items()
    }
    if any(code != 0 for code in codes.values()):
        print(
            "local gcloud transports returned nonzero after all four exact "
            f"evaluation terminal markers: {codes}",
            flush=True,
        )
    if any(status != 0 for status in statuses.values()):
        raise OperatorError(
            f"one or more evaluation workers failed: {statuses}; "
            f"inspect {settings.log_root}"
        )
    result = Path(settings.result_root) / "scientific-result.json"
    _validate_scientific_result(result)
    print(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as error:
        print(f"EXPERIMENT-2 EVALUATION OPERATOR FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error
