#!/usr/bin/env python3
"""Fan out the Experiment-3 final evaluator across four TPU controllers."""

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

PROCESS_COUNT = 4
V4_32_PEAK_FLOPS_PER_SECOND = 4_400_000_000_000_000
V6E_16_PEAK_FLOPS_PER_SECOND = 14_688_000_000_000_000
V6E_PROCESS_CACHE_MAX_BYTES = 160 * 1024**3
RESULT_SCHEMA_VERSION = "gemma4-contribution-transcoder-scientific-result-v2"
PROTOCOL_ID = "gemma4-contribution-transcoder-v2-20260727"
LEARNING_HORIZON_RESULT_SCHEMA_VERSION = (
    "gemma4-contribution-transcoder-learning-horizon-result-v1"
)
LEARNING_HORIZON_PROTOCOL_ID = "gemma4-pilot-learning-horizon-v1-20260728"
_GIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OperatorError(RuntimeError):
    """The final transcoder evaluation cannot be launched or verified."""


def _required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise OperatorError(f"required environment variable {name} is unset")
    if any(character in value for character in "\x00\n\r"):
        raise OperatorError(f"{name} contains an unsafe control character")
    return value


def _positive_int(name: str, value: str) -> int:
    try:
        selected = int(value)
    except ValueError as error:
        raise OperatorError(f"{name} must be an integer") from error
    if selected <= 0 or str(selected) != value:
        raise OperatorError(f"{name} must be a canonical positive integer")
    return selected


def _nonnegative_float(name: str, value: str) -> float:
    try:
        selected = float(value)
    except ValueError as error:
        raise OperatorError(f"{name} must be numeric") from error
    if not selected >= 0 or selected == float("inf"):
        raise OperatorError(f"{name} must be finite and nonnegative")
    return selected


@dataclass(frozen=True, slots=True)
class Settings:
    project: str
    tpu: str
    zone: str
    commit: str
    training_commit: str
    root: str
    uv: str
    bundle: str
    sidecar: str
    train_target_root: str
    eval_target_root: str
    train_target_objects: str
    eval_target_objects: str
    train_source_root: str
    eval_source_root: str
    train_source_objects: str
    eval_source_objects: str
    cache_root: str
    cache_bytes: int
    sae_control_root: str
    checkpoint_root: str
    run_roots: dict[str, str]
    result_root: str
    log_root: Path
    peak_flops_per_second: int
    harvest_receipt_sha256: str | None
    harvest_chip_seconds: float
    ssh_key: str | None
    accelerator_type: str = "v4-32"
    regional_mirror_seal: str | None = None
    internal_ip: bool = False
    config_path: str | None = None
    phase: str = "mechanism_pilot"
    seed: int = 0
    checkpoint_tokens: int | None = None

    @classmethod
    def from_environment(cls) -> "Settings":
        commit = _required("PROJECT_COMMIT")
        if _GIT.fullmatch(commit) is None:
            raise OperatorError("PROJECT_COMMIT must be a full lowercase Git SHA")
        training_commit = _required("TRAINING_PROJECT_COMMIT")
        if _GIT.fullmatch(training_commit) is None:
            raise OperatorError(
                "TRAINING_PROJECT_COMMIT must be a full lowercase Git SHA"
            )
        control = _required("CONTRIBUTION_SAE_CONTROL_ROOT").rstrip("/")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        harvest_sha = os.environ.get("TRANSCODER_HARVEST_RECEIPT_SHA256")
        if harvest_sha and _SHA256.fullmatch(harvest_sha) is None:
            raise OperatorError(
                "TRANSCODER_HARVEST_RECEIPT_SHA256 must be a SHA-256"
            )
        run_roots = {
            condition: (
                f"{control}/runs/"
                + os.environ.get(
                    f"TRANSCODER_RUN_ID_{condition}",
                    f"experiment3-{condition.lower()}-seed0",
                )
            )
            for condition in ("A", "R", "B", "F")
        }
        accelerator_type = os.environ.get("TPU_ACCELERATOR_TYPE", "v4-32")
        if accelerator_type not in {"v4-32", "v6e-16"}:
            raise OperatorError("TPU_ACCELERATOR_TYPE is unsupported")
        mapping_receipt = (
            _required("CONTRIBUTION_FRESH_CONTROLLER_MAPPING")
            if accelerator_type == "v6e-16"
            else _required("CONTRIBUTION_QUALIFICATION_BUNDLE")
        )
        regional_mirror = os.environ.get("CONTRIBUTION_REGIONAL_MIRROR_SEAL")
        if accelerator_type == "v6e-16" and not regional_mirror:
            raise OperatorError("v6e-16 evaluation requires a regional mirror seal")
        expected_peak = (
            V6E_16_PEAK_FLOPS_PER_SECOND
            if accelerator_type == "v6e-16"
            else V4_32_PEAK_FLOPS_PER_SECOND
        )
        cache_bytes = _positive_int(
            "CONTRIBUTION_SAE_CACHE_MAX_BYTES",
            _required("CONTRIBUTION_SAE_CACHE_MAX_BYTES"),
        )
        if (
            accelerator_type == "v6e-16"
            and cache_bytes != V6E_PROCESS_CACHE_MAX_BYTES
        ):
            raise OperatorError("v6e-16 cache bound is fixed at 160GiB")
        supplied_peak = _positive_int(
            "TRANSCODER_PEAK_FLOPS_PER_SECOND",
            os.environ.get(
                "TRANSCODER_PEAK_FLOPS_PER_SECOND",
                str(expected_peak),
            ),
        )
        if supplied_peak != expected_peak:
            raise OperatorError(
                f"{accelerator_type} peak FLOPs must be exactly {expected_peak}"
            )
        internal_ip_raw = os.environ.get("TPU_USE_INTERNAL_IP", "0")
        if internal_ip_raw not in {"0", "1"}:
            raise OperatorError("TPU_USE_INTERNAL_IP must be exactly 0 or 1")
        internal_ip = internal_ip_raw == "1"
        if accelerator_type == "v6e-16" and not internal_ip:
            raise OperatorError(
                "v6e-16 evaluation requires explicit TPU_USE_INTERNAL_IP=1"
            )
        return cls(
            project=_required("GCP_PROJECT"),
            tpu=_required("TPU_NAME"),
            zone=_required("TPU_ZONE"),
            commit=commit,
            training_commit=training_commit,
            root=_required("TPU_PROJECT_ROOT"),
            uv=_required("REMOTE_UV"),
            bundle=mapping_receipt,
            sidecar=_required("CONTRIBUTION_RECOMPOSITION_SIDECAR"),
            train_target_root=_required("CONTRIBUTION_TRAIN_CAPTURE_ROOT"),
            eval_target_root=_required("CONTRIBUTION_EVAL_CAPTURE_ROOT"),
            train_target_objects=_required("CONTRIBUTION_OBJECT_PREFIX"),
            eval_target_objects=_required("CONTRIBUTION_EVAL_OBJECT_PREFIX"),
            train_source_root=_required("CONTRIBUTION_SOURCE_OVERLAY_ROOT"),
            eval_source_root=_required("CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT"),
            train_source_objects=_required(
                "CONTRIBUTION_SOURCE_OVERLAY_PREFIX"
            ),
            eval_source_objects=_required(
                "CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX"
            ),
            cache_root=_required("CONTRIBUTION_SAE_CACHE_ROOT"),
            cache_bytes=cache_bytes,
            sae_control_root=control,
            checkpoint_root=_required("CONTRIBUTION_SAE_CHECKPOINT_ROOT"),
            run_roots=run_roots,
            result_root=os.environ.get(
                "TRANSCODER_EVALUATION_ROOT",
                f"{control}/evaluation/experiment3-seed0",
            ),
            log_root=Path(
                os.environ.get(
                    "TRANSCODER_EVALUATION_LOG_ROOT",
                    str(
                        Path.cwd()
                        / "artifacts"
                        / "gemma4_v1"
                        / "operator-results"
                        / "experiment3-evaluation"
                        / f"operator-{stamp}"
                    ),
                )
            ),
            peak_flops_per_second=supplied_peak,
            harvest_receipt_sha256=harvest_sha or None,
            harvest_chip_seconds=_nonnegative_float(
                "TRANSCODER_HARVEST_CHIP_SECONDS",
                os.environ.get("TRANSCODER_HARVEST_CHIP_SECONDS", "0"),
            ),
            ssh_key=os.environ.get("CONTRIBUTION_SSH_KEY_FILE") or None,
            accelerator_type=accelerator_type,
            regional_mirror_seal=regional_mirror or None,
            internal_ip=internal_ip,
        )


def _mapping_payload(
    payload: bytes,
    *,
    accelerator_type: str = "v4-32",
) -> dict[int, int]:
    try:
        bundle = json.loads(payload)
        if accelerator_type == "v6e-16":
            from moe_sae.regional_execution import (
                validate_controller_mapping_receipt,
            )

            bundle = validate_controller_mapping_receipt(bundle)
            if (
                bundle["accelerator_type"] != "v6e-16"
                or bundle["process_count"] != PROCESS_COUNT
                or bundle["global_device_count"] != 16
                or bundle["local_device_count"] != 4
            ):
                raise OperatorError(
                    "fresh v6e controller mapping topology changed"
                )
        rows = bundle["controller_to_jax_process_index"]
        result = {
            int(row["controller_index"]): int(row["process_index"])
            for row in rows
        }
    except (
        UnicodeDecodeError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        raise OperatorError("qualification bundle has no controller mapping") from error
    if set(result) != set(range(PROCESS_COUNT)) or set(result.values()) != set(
        range(PROCESS_COUNT)
    ):
        raise OperatorError("controller/JAX process mapping is not a permutation")
    return result


def _remote_bytes(
    settings: Settings,
    *,
    controller: int,
    remote_path: str,
) -> bytes:
    """Read one remote control artifact from the Mac through gcloud."""

    command = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        settings.tpu,
        f"--project={settings.project}",
        f"--zone={settings.zone}",
        f"--worker={controller}",
        "--quiet",
        "--command",
        "cat " + shlex.quote(remote_path),
    ]
    if settings.ssh_key:
        command.insert(-2, f"--ssh-key-file={settings.ssh_key}")
    if settings.internal_ip:
        command.insert(-2, "--internal-ip")
    environment = os.environ.copy()
    for name in ("LD_PRELOAD", "GLIBCXX_FORCE_NEW", "GLIBCPP_FORCE_NEW"):
        environment.pop(name, None)
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if completed.returncode != 0:
        raise OperatorError(
            f"cannot read remote control artifact {remote_path}: "
            + completed.stderr.decode("utf-8", errors="replace")[-2_000:]
        )
    return completed.stdout


def _mapping(settings: Settings) -> dict[int, int]:
    return _mapping_payload(
        _remote_bytes(
            settings,
            controller=0,
            remote_path=settings.bundle,
        ),
        accelerator_type=settings.accelerator_type,
    )


def _worker(settings: Settings, process: int) -> list[str]:
    command = [
        settings.uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        "-m",
        "moe_sae.contribution_transcoder_evaluation_worker",
        "--config",
        (
            settings.config_path
            or f"{settings.root}/configs/gemma4_transcoder_v2.yaml"
        ),
        "--accelerator-type",
        settings.accelerator_type,
        "--phase",
        settings.phase,
        "--seed",
        str(settings.seed),
        "--source-revision",
        settings.commit,
        "--training-source-revision",
        settings.training_commit,
        "--train-target-root",
        settings.train_target_root,
        "--eval-target-root",
        settings.eval_target_root,
        "--train-target-object-prefix",
        settings.train_target_objects,
        "--eval-target-object-prefix",
        settings.eval_target_objects,
        "--train-source-root",
        settings.train_source_root,
        "--eval-source-root",
        settings.eval_source_root,
        "--train-source-object-prefix",
        settings.train_source_objects,
        "--eval-source-object-prefix",
        settings.eval_source_objects,
        "--cache-root",
        settings.cache_root,
        "--cache-max-bytes",
        str(settings.cache_bytes),
        "--sae-control-root",
        settings.sae_control_root,
        "--checkpoint-payload-root",
        settings.checkpoint_root,
        "--recomposition-sidecar",
        settings.sidecar,
        "--result-root",
        settings.result_root,
        "--process-index",
        str(process),
        "--peak-flops-per-second",
        str(settings.peak_flops_per_second),
        "--shared-harvest-chip-seconds",
        str(settings.harvest_chip_seconds),
    ]
    if settings.checkpoint_tokens is not None:
        command.extend(
            ("--checkpoint-tokens", str(settings.checkpoint_tokens))
        )
    if settings.regional_mirror_seal is not None:
        command.extend(
            ("--regional-mirror-seal", settings.regional_mirror_seal)
        )
    for condition in ("A", "R", "B", "F"):
        command.extend(
            (
                f"--run-root-{condition.lower()}",
                settings.run_roots[condition],
            )
        )
    if settings.harvest_receipt_sha256:
        command.extend(
            (
                "--shared-harvest-receipt-sha256",
                settings.harvest_receipt_sha256,
            )
        )
    return command


def _command(
    settings: Settings,
    controller: int,
    process: int,
) -> list[str]:
    worker = _worker(settings, process)
    result_path = settings.result_root.rstrip("/") + "/scientific-result.json"
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
                'if test "$evaluation_status" -eq 0; then '
                f"test -s {shlex.quote(result_path)}; fi"
            ),
            (
                'printf "terminal action=evaluate-transcoder controller=%s '
                'process=%s status=%s\\n" '
                f"{controller} {process} \"$evaluation_status\""
            ),
            'exit "$evaluation_status"',
        )
    )
    command = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        settings.tpu,
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


def _terminal_status(
    log: Path,
    *,
    controller: int,
    process: int,
) -> int:
    try:
        lines = log.read_bytes().splitlines()
    except OSError as error:
        raise OperatorError(f"cannot read evaluator SSH log: {log}") from error
    prefix = (
        f"terminal action=evaluate-transcoder controller={controller} "
        f"process={process} status="
    ).encode("ascii")
    rows = [line for line in lines if line.startswith(prefix)]
    if len(rows) != 1 or re.fullmatch(rb"[0-9]{1,3}", rows[0][len(prefix) :]) is None:
        raise OperatorError(
            f"controller {controller} has no unique valid terminal marker"
        )
    status = int(rows[0][len(prefix) :])
    if status > 255:
        raise OperatorError("terminal status exceeds shell range")
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


def _validate_result(
    path: Path,
    *,
    expected_source_revision: str,
    expected_training_source_revision: str,
) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperatorError(f"cannot read scientific result: {path}") from error
    if not isinstance(value, dict):
        raise OperatorError("scientific result must be a JSON object")
    body = dict(value)
    claimed = body.pop("result_sha256", None)
    result_identity = (
        body.get("schema_version"),
        body.get("protocol_id"),
    )
    if (
        result_identity
        not in {
            (RESULT_SCHEMA_VERSION, PROTOCOL_ID),
            (
                LEARNING_HORIZON_RESULT_SCHEMA_VERSION,
                LEARNING_HORIZON_PROTOCOL_ID,
            ),
        }
        or body.get("source_revision") != expected_source_revision
        or body.get("training_source_revision")
        != expected_training_source_revision
        or not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or _canonical_sha256(body) != claimed
    ):
        raise OperatorError("scientific result identity or self-hash is invalid")
    parents = body.get("parents")
    contracts = (
        parents.get("run_contract_sha256")
        if isinstance(parents, dict)
        else None
    )
    if (
        not isinstance(contracts, dict)
        or set(contracts) != {"A", "R", "B", "F"}
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in contracts.values()
        )
    ):
        raise OperatorError(
            "scientific result lacks exact training run-contract provenance"
        )
    return value


def _fetch_result(
    settings: Settings,
    *,
    controller: int,
) -> Path:
    """Fetch the process-zero result without assuming the Mac mounts NFS."""

    remote = settings.result_root.rstrip("/") + "/scientific-result.json"
    payload = _remote_bytes(
        settings,
        controller=controller,
        remote_path=remote,
    )
    local = settings.log_root / "scientific-result.json"
    local.write_bytes(payload)
    return local


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
            "configs/gemma4_transcoder_v2.yaml"
        ),
    )
    parser.add_argument("--phase", default="mechanism_pilot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-tokens", type=int)
    parser.add_argument("--result-root")
    for condition in ("A", "R", "B", "F"):
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
    run_roots = dict(settings.run_roots)
    for condition in ("A", "R", "B", "F"):
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
        run_roots=run_roots,
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
                            "gemma4_transcoder_v2.yaml"
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
    mapping = _mapping(settings)
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
        codes = {
            controller: future.result()
            for controller, future in futures.items()
        }
    statuses = {
        controller: _terminal_status(
            logs[controller],
            controller=controller,
            process=process,
        )
        for controller, process in mapping.items()
    }
    if any(status != 0 for status in statuses.values()):
        raise OperatorError(
            f"one or more evaluator workers failed: {statuses}; "
            f"transports={codes}; logs={settings.log_root}"
        )
    process_zero_controller = next(
        controller for controller, process in mapping.items() if process == 0
    )
    result = _fetch_result(
        settings,
        controller=process_zero_controller,
    )
    sealed = _validate_result(
        result,
        expected_source_revision=settings.commit,
        expected_training_source_revision=settings.training_commit,
    )
    print(
        json.dumps(
            {
                "result": str(result),
                "result_sha256": sealed["result_sha256"],
                "status": sealed["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as error:
        print(
            f"EXPERIMENT-3 EVALUATION OPERATOR FAILED: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
