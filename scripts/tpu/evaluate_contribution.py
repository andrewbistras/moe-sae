#!/usr/bin/env python3
"""Fan out the Experiment-2 final evaluator across four TPU controllers."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROCESS_COUNT = 4
_SHA = re.compile(r"^[0-9a-f]{40}$")


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
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return cls(
            project=_required("GCP_PROJECT"),
            tpu=_required("TPU_NAME"),
            zone=_required("TPU_ZONE"),
            commit=commit,
            root=_required("TPU_PROJECT_ROOT"),
            uv=_required("REMOTE_UV"),
            bundle=_required("CONTRIBUTION_QUALIFICATION_BUNDLE"),
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
        )


def _mapping(bundle_path: str) -> dict[int, int]:
    try:
        bundle = json.loads(Path(bundle_path).read_text(encoding="ascii"))
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


def _command(settings: Settings, controller: int, process: int) -> list[str]:
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
        f"{settings.root}/configs/gemma4_contribution_v1.yaml",
        "--phase",
        "phase2_mechanism_pilot",
        "--seed",
        "0",
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
    body = "\n".join(
        (
            "set -euo pipefail",
            f"cd {shlex.quote(settings.root)}",
            f'test "$(git rev-parse HEAD)" = {shlex.quote(settings.commit)}',
            'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
            shlex.join(worker),
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


def main() -> int:
    settings = Settings.from_environment()
    mapping = _mapping(settings.bundle)
    settings.log_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    with ThreadPoolExecutor(max_workers=PROCESS_COUNT) as pool:
        futures = {
            controller: pool.submit(
                _run,
                _command(settings, controller, process),
                settings.log_root
                / f"evaluation-c{controller:05d}-p{process:05d}.log",
            )
            for controller, process in mapping.items()
        }
        codes = {controller: future.result() for controller, future in futures.items()}
    if any(code != 0 for code in codes.values()):
        raise OperatorError(
            f"one or more evaluation workers failed: {codes}; "
            f"inspect {settings.log_root}"
        )
    result = Path(settings.result_root) / "scientific-result.json"
    if not result.is_file():
        raise OperatorError(f"evaluation completed without {result}")
    print(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"EXPERIMENT-2 EVALUATION OPERATOR FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error
