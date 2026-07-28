#!/usr/bin/env python3
"""Fan out one bounded routed-input overlay harvest.

This operator runs on the Mac/control machine.  It starts the identical worker
module on TPU VM workers 0--3 concurrently; the worker module itself performs
the JAX barrier and lets JAX process zero seal the manifest.  All scientific
inputs are explicit CLI arguments.  Environment variables are used only by
``gcloud`` for its ordinary account/SDK configuration.

The frozen v2 experiment reuses its two already-sealed overlay-v1 manifests.
Its recovery path retains the original ten-minute limit.  The 10M learning
horizon may instead supply a strict target-bound capture contract whose sole
pending identity is the overlay this command will seal.  That train-only path
has a one-hour supervised hard limit.  In both cases a new capture requires an
explicit acknowledgement and cannot target a sealed control root.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import time

from moe_sae.contribution_transcoder_capture_config import (
    TranscoderOverlayCaptureConfigError,
    load_transcoder_overlay_capture_config,
    overlay_capture_runtime_cap_seconds,
)


PROCESS_COUNT = 4
TRANSPORT_GRACE_SECONDS = 180
DEFAULT_CONFIG = "configs/gemma4_transcoder_v2.yaml"
_SAFE_SSH_USER = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")


class OperatorError(RuntimeError):
    """The bounded four-controller launch could not complete safely."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Harvest and seal one Experiment-3 routed-input overlay"
    )
    parser.add_argument("--gcp-project", required=True)
    parser.add_argument("--tpu-name", required=True)
    parser.add_argument("--tpu-zone", required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=(
            "absolute path shared by the operator and TPU workers, or a path "
            "relative to both checkouts; the 10M path requires its strict "
            "target-bound capture contract"
        ),
    )
    parser.add_argument("--remote-uv", default="uv")
    parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--grain-files", required=True)
    parser.add_argument("--staged-manifest", required=True)
    parser.add_argument("--source-v1-root", required=True)
    parser.add_argument("--source-cache-max-bytes", type=int, required=True)
    parser.add_argument("--target-control-root", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--local-buffer-root", required=True)
    parser.add_argument("--object-prefix", required=True)
    parser.add_argument(
        "--mesh",
        choices=("ep16-fsdp1-tp1", "ep8-fsdp2-tp1", "ep8-fsdp1-tp2"),
        required=True,
    )
    parser.add_argument("--per-device-sequence-batch", type=int, required=True)
    parser.add_argument("--maximum-runtime-seconds", type=int, default=600)
    parser.add_argument("--local-log-root", required=True)
    parser.add_argument("--ssh-key-file")
    parser.add_argument(
        "--ssh-user",
        default=os.environ.get("TPU_SSH_USER") or None,
        help=(
            "explicit remote TPU VM Linux username; defaults to "
            "TPU_SSH_USER, otherwise gcloud's normal default"
        ),
    )
    parser.add_argument(
        "--new-overlay",
        action="store_true",
        help=(
            "acknowledge that this is a new overlay capture using a fresh "
            "control root; frozen v2 production reuses its sealed v1 overlays"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _remote_config_path(args: argparse.Namespace) -> str:
    value = str(getattr(args, "config", DEFAULT_CONFIG))
    selected = PurePosixPath(value)
    if selected.is_absolute():
        return str(selected)
    if not value or ".." in selected.parts:
        raise OperatorError(
            "--config must be absolute or a safe project-relative path"
        )
    return str(PurePosixPath(args.project_root) / selected)


def _local_config_path(args: argparse.Namespace) -> Path:
    value = str(getattr(args, "config", DEFAULT_CONFIG))
    selected = Path(value).expanduser()
    if selected.is_absolute():
        return selected.resolve()
    if not value or ".." in selected.parts:
        raise OperatorError(
            "--config must be absolute or a safe project-relative path"
        )
    return (Path(__file__).resolve().parents[2] / selected).resolve()


def _expected_config_sha256(args: argparse.Namespace) -> str:
    selected = getattr(args, "expected_config_sha256", None)
    if isinstance(selected, str) and selected:
        return selected
    try:
        return load_transcoder_overlay_capture_config(
            _local_config_path(args)
        ).config_hash
    except TranscoderOverlayCaptureConfigError as error:
        raise OperatorError(str(error)) from error


def _worker_command(args: argparse.Namespace, controller: int) -> list[str]:
    source_cache = (
        f"{args.local_buffer_root}/experiment3/source-cache/"
        f"{args.split}/controller-c{controller:05d}"
    )
    local_buffer = (
        f"{args.local_buffer_root}/experiment3/overlay/"
        f"{args.split}/controller-c{controller:05d}"
    )
    jax_cache = (
        f"{args.local_buffer_root}/experiment3/jax-cache/"
        f"{args.project_commit}/{args.mesh}/"
        f"b{args.per_device_sequence_batch}/controller-c{controller:05d}"
    )
    command = [
        args.remote_uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        "-m",
        "moe_sae.contribution_transcoder_harvest_worker",
        "--config",
        _remote_config_path(args),
        "--expected-config-sha256",
        _expected_config_sha256(args),
        "--split",
        args.split,
        "--checkpoint",
        args.checkpoint,
        "--checkpoint-sha256",
        args.checkpoint_sha256,
        "--grain-files",
        args.grain_files,
        "--staged-manifest",
        args.staged_manifest,
        "--source-v1-root",
        args.source_v1_root,
        "--source-cache-root",
        source_cache,
        "--source-cache-max-bytes",
        str(args.source_cache_max_bytes),
        "--target-control-root",
        args.target_control_root,
        "--control-root",
        args.control_root,
        "--local-buffer-root",
        local_buffer,
        "--object-prefix",
        args.object_prefix,
        "--repo-root",
        args.project_root,
        "--jax-cache-dir",
        jax_cache,
        "--mesh",
        args.mesh,
        "--per-device-sequence-batch",
        str(args.per_device_sequence_batch),
        "--maximum-runtime-seconds",
        str(args.maximum_runtime_seconds),
    ]
    lines = [
        "set -euo pipefail",
        f"cd {shlex.quote(args.project_root)}",
        (
            'test "$(git rev-parse HEAD)" = '
            f"{shlex.quote(args.project_commit)}"
        ),
        'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
        (
            f"if test -e {shlex.quote(str(Path(args.control_root) / 'manifest.json'))}; "
            "then echo 'sealed overlay already exists; reuse it or choose a "
            "fresh --control-root' >&2; exit 2; fi"
        ),
        (
            "install -d -m 0700 "
            f"{shlex.quote(source_cache)} "
            f"{shlex.quote(local_buffer)} "
            f"{shlex.quote(jax_cache)}"
        ),
        (
            f"timeout --signal=TERM --kill-after=30s "
            f"{args.maximum_runtime_seconds}s {shlex.join(command)}"
        ),
    ]
    remote = "bash -lc " + shlex.quote("\n".join(lines))
    ssh_user = getattr(args, "ssh_user", None)
    target = f"{ssh_user}@{args.tpu_name}" if ssh_user else args.tpu_name
    result = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        target,
        f"--project={args.gcp_project}",
        f"--zone={args.tpu_zone}",
        f"--worker={controller}",
        f"--command={remote}",
        "--ssh-flag=-oBatchMode=yes",
        "--ssh-flag=-oConnectTimeout=30",
        "--ssh-flag=-oConnectionAttempts=3",
        "--ssh-flag=-oServerAliveInterval=30",
        "--ssh-flag=-oServerAliveCountMax=4",
        "--quiet",
    ]
    if args.ssh_key_file:
        result.insert(-1, f"--ssh-key-file={args.ssh_key_file}")
    return result


def run(args: argparse.Namespace) -> int:
    if not args.new_overlay:
        raise OperatorError(
            "frozen v2 reuses its sealed overlay-v1 manifests; a new capture "
            "requires --new-overlay and a fresh --control-root"
        )
    ssh_user = getattr(args, "ssh_user", None)
    if ssh_user is not None and _SAFE_SSH_USER.fullmatch(ssh_user) is None:
        raise OperatorError(
            "TPU_SSH_USER/--ssh-user must be a safe lowercase Linux username"
        )
    try:
        capture_config = load_transcoder_overlay_capture_config(
            _local_config_path(args)
        )
        runtime_cap = overlay_capture_runtime_cap_seconds(
            capture_config,
            split=args.split,
        )
    except TranscoderOverlayCaptureConfigError as error:
        raise OperatorError(str(error)) from error
    args.expected_config_sha256 = capture_config.config_hash
    if not 0 < args.maximum_runtime_seconds <= runtime_cap:
        raise OperatorError(
            "maximum runtime must lie in "
            f"[1,{runtime_cap}] seconds for the selected protocol"
        )
    if args.source_cache_max_bytes <= 0 or args.per_device_sequence_batch <= 0:
        raise OperatorError("cache bytes and sequence batch must be positive")
    commands = [_worker_command(args, index) for index in range(PROCESS_COUNT)]
    if args.dry_run:
        for index, command in enumerate(commands):
            print(f"# controller {index}\n{shlex.join(command)}")
        return 0
    if shutil.which("gcloud") is None:
        raise OperatorError("gcloud is unavailable")

    log_root = Path(args.local_log_root).expanduser().resolve()
    log_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    processes: list[tuple[subprocess.Popen[bytes], object, Path]] = []
    try:
        for index, command in enumerate(commands):
            log = log_root / f"controller-{index:05d}.ssh.log"
            handle = log.open("xb")
            try:
                process = subprocess.Popen(
                    command, stdout=handle, stderr=subprocess.STDOUT
                )
            except BaseException:
                handle.close()
                raise
            processes.append((process, handle, log))
        deadline = (
            time.monotonic()
            + args.maximum_runtime_seconds
            + TRANSPORT_GRACE_SECONDS
        )
        pending = {process for process, _, _ in processes}
        while pending and time.monotonic() < deadline:
            pending = {process for process in pending if process.poll() is None}
            if pending:
                time.sleep(0.25)
        if pending:
            for process in pending:
                process.terminate()
            raise OperatorError(
                "SSH transports exceeded the supervised workload bound plus grace"
            )
        failed = [
            (index, int(process.returncode), log)
            for index, (process, _, log) in enumerate(processes)
            if process.returncode != 0
        ]
        if failed:
            detail = ", ".join(
                f"controller {index}: status {status}, log {log}"
                for index, status, log in failed
            )
            raise OperatorError(f"overlay harvest failed: {detail}")
    finally:
        for process, handle, _ in processes:
            if process.poll() is None:
                process.terminate()
            handle.close()
    print(
        "SEALED routed-input overlay\n"
        f"manifest: {args.control_root}/manifest.json\n"
        f"capture contract SHA-256: {capture_config.config_hash}\n"
        f"logs: {log_root}"
    )
    return 0


def main() -> int:
    try:
        return run(_parser().parse_args())
    except OperatorError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
