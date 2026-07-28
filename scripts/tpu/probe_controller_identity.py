#!/usr/bin/env python3
"""Fast read-only controller/JAX identity probe for one fresh TPU slice.

Run this on the operator machine after the TPU description has passed the
Spot/shape attestation.  Four SSH commands initialize JAX concurrently, each
physical controller writes one immutable observation to the shared regional
control plane, and controller 0 seals the exact permutation.  No performance
qualification and no cloud mutation occur here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys


PROCESS_COUNT = 4
PROBE_TIMEOUT_SECONDS = 420
MINIMUM_CACHE_FREE_BYTES = 192 * 1024**3
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ProbeOperatorError(RuntimeError):
    """The fresh controller identity probe could not be sealed."""


def _absolute_remote(name: str, value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) == "/":
        raise ProbeOperatorError(f"{name} must be a non-root absolute path")
    return str(path)


def _gcloud(
    args: argparse.Namespace,
    *,
    worker: int,
    remote_command: str,
) -> list[str]:
    command = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        args.tpu_name,
        f"--project={args.project}",
        f"--zone={args.zone}",
        f"--worker={worker}",
        f"--command={remote_command}",
        "--ssh-flag=-oBatchMode=yes",
        "--ssh-flag=-oConnectTimeout=30",
        "--ssh-flag=-oConnectionAttempts=3",
        "--ssh-flag=-oServerAliveInterval=30",
        "--ssh-flag=-oServerAliveCountMax=4",
        "--quiet",
    ]
    if args.internal_ip:
        command.insert(-1, "--internal-ip")
    if args.ssh_key_file:
        command.insert(-1, f"--ssh-key-file={args.ssh_key_file}")
    return command


def _remote_payload(lines: Sequence[str]) -> str:
    return "bash -lc " + shlex.quote("\n".join(lines))


def _clean_checkout(args: argparse.Namespace) -> list[str]:
    return [
        "set -eu",
        f"cd {shlex.quote(args.project_root)}",
        (
            'test "$(git rev-parse HEAD)" = '
            f"{shlex.quote(args.project_commit)}"
        ),
        'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
    ]


def _observation_command(
    args: argparse.Namespace,
    controller_index: int,
) -> list[str]:
    observation_root = f"{args.probe_root}/observations"
    output = f"{observation_root}/controller-{controller_index:05d}.json"
    command = [
        args.remote_uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        "-m",
        "moe_sae.regional_execution",
        "observe-controller",
        "--contract",
        args.contract,
        "--probe-id",
        args.probe_id,
        "--controller-index",
        str(controller_index),
        "--output",
        output,
    ]
    return _gcloud(
        args,
        worker=controller_index,
        remote_command=_remote_payload(
            [
                *_clean_checkout(args),
                f"install -d -m 0700 {shlex.quote(observation_root)}",
                shlex.join(command),
            ]
        ),
    )


def _seal_command(args: argparse.Namespace) -> list[str]:
    output = f"{args.probe_root}/controller-mapping.json"
    command = [
        args.remote_uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        "-m",
        "moe_sae.regional_execution",
        "seal-controller-map",
        "--contract",
        args.contract,
        "--probe-id",
        args.probe_id,
        "--observation-root",
        f"{args.probe_root}/observations",
        "--output",
        output,
    ]
    return _gcloud(
        args,
        worker=0,
        remote_command=_remote_payload(
            [*_clean_checkout(args), shlex.join(command)]
        ),
    )


def _cache_observation_command(
    args: argparse.Namespace,
    controller_index: int,
) -> list[str]:
    root = f"{args.probe_root}/cache-observations"
    output = f"{root}/controller-{controller_index:05d}.json"
    command = [
        args.remote_uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        "-m",
        "moe_sae.regional_execution",
        "observe-cache",
        "--contract",
        args.contract,
        "--controller-mapping",
        f"{args.probe_root}/controller-mapping.json",
        "--controller-index",
        str(controller_index),
        "--cache-root",
        args.cache_root,
        "--required-lane-bytes",
        str(args.required_lane_bytes),
        "--required-free-bytes",
        str(args.required_free_bytes),
        "--output",
        output,
    ]
    return _gcloud(
        args,
        worker=controller_index,
        remote_command=_remote_payload(
            [
                *_clean_checkout(args),
                f"install -d -m 0700 {shlex.quote(root)}",
                shlex.join(command),
            ]
        ),
    )


def _seal_cache_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.remote_uv,
        "run",
        "--frozen",
        "--extra",
        "tpu",
        "python",
        "-m",
        "moe_sae.regional_execution",
        "seal-cache-inventory",
        "--contract",
        args.contract,
        "--controller-mapping",
        f"{args.probe_root}/controller-mapping.json",
        "--observation-root",
        f"{args.probe_root}/cache-observations",
        "--output",
        f"{args.probe_root}/cache-inventory.json",
    ]
    return _gcloud(
        args,
        worker=0,
        remote_command=_remote_payload(
            [*_clean_checkout(args), shlex.join(command)]
        ),
    )


def _cat_receipt_command(args: argparse.Namespace, name: str) -> list[str]:
    return _gcloud(
        args,
        worker=0,
        remote_command=_remote_payload(
            [
                "set -eu",
                f"cat {shlex.quote(args.probe_root + '/' + name)}",
            ]
        ),
    )


def _validate_args(args: argparse.Namespace) -> None:
    if not args.internal_ip:
        raise ProbeOperatorError(
            "--internal-ip is required for the private v6e-16 identity probe"
        )
    if (
        re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", args.project) is None
        or _SAFE_ID.fullmatch(args.tpu_name) is None
        or re.fullmatch(r"^[a-z]+-[a-z0-9]+[0-9]-[a-z]$", args.zone) is None
        or _GIT_SHA.fullmatch(args.project_commit) is None
        or _SAFE_ID.fullmatch(args.probe_id) is None
    ):
        raise ProbeOperatorError("project, TPU, zone, commit, or probe ID is invalid")
    args.project_root = _absolute_remote("project_root", args.project_root)
    args.remote_uv = _absolute_remote("remote_uv", args.remote_uv)
    args.contract = _absolute_remote("contract", args.contract)
    args.probe_root = _absolute_remote("probe_root", args.probe_root)
    args.cache_root = _absolute_remote("cache_root", args.cache_root)
    if (
        args.required_lane_bytes <= 0
        or args.required_free_bytes < MINIMUM_CACHE_FREE_BYTES
        or args.required_free_bytes
        < (args.required_lane_bytes * 11 + 9) // 10
    ):
        raise ProbeOperatorError(
            "cache requires its lane plus 10% and at least 192GiB free"
        )
    if args.ssh_key_file:
        key = Path(args.ssh_key_file).expanduser().resolve()
        if not key.is_file():
            raise ProbeOperatorError("ssh_key_file is not a regular file")
        args.ssh_key_file = str(key)


def _run_probe(args: argparse.Namespace) -> dict[str, object]:
    processes = [
        subprocess.Popen(
            _observation_command(args, controller),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for controller in range(PROCESS_COUNT)
    ]
    failures: list[str] = []
    for controller, process in enumerate(processes):
        try:
            output, _ = process.communicate(timeout=PROBE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            failures.append(f"controller-{controller}: timeout\n{output}")
            continue
        if process.returncode != 0:
            failures.append(
                f"controller-{controller}: exit={process.returncode}\n{output}"
            )
    if failures:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        raise ProbeOperatorError(
            "fresh distributed identity probe failed:\n" + "\n".join(failures)
        )
    sealed = subprocess.run(
        _seal_command(args),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if sealed.returncode != 0:
        raise ProbeOperatorError(
            "could not seal fresh controller mapping:\n"
            + sealed.stdout
            + sealed.stderr
        )
    cache_processes = [
        subprocess.Popen(
            _cache_observation_command(args, controller),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for controller in range(PROCESS_COUNT)
    ]
    cache_failures: list[str] = []
    for controller, process in enumerate(cache_processes):
        try:
            output, _ = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            cache_failures.append(
                f"cache-controller-{controller}: timeout\n{output}"
            )
            continue
        if process.returncode != 0:
            cache_failures.append(
                f"cache-controller-{controller}: "
                f"exit={process.returncode}\n{output}"
            )
    if cache_failures:
        raise ProbeOperatorError(
            "fresh local-cache capacity probe failed:\n"
            + "\n".join(cache_failures)
        )
    sealed_cache = subprocess.run(
        _seal_cache_command(args),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if sealed_cache.returncode != 0:
        raise ProbeOperatorError(
            "could not seal fresh cache inventory:\n"
            + sealed_cache.stdout
            + sealed_cache.stderr
        )
    fetched = subprocess.run(
        _cat_receipt_command(args, "controller-mapping.json"),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if fetched.returncode != 0:
        raise ProbeOperatorError(
            "could not read sealed controller mapping:\n"
            + fetched.stdout
            + fetched.stderr
        )
    try:
        value = json.loads(fetched.stdout)
    except json.JSONDecodeError as error:
        raise ProbeOperatorError(
            "sealed controller mapping output is not JSON"
        ) from error
    if not isinstance(value, dict):
        raise ProbeOperatorError("sealed controller mapping is not an object")
    cache_fetched = subprocess.run(
        _cat_receipt_command(args, "cache-inventory.json"),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if cache_fetched.returncode != 0:
        raise ProbeOperatorError("could not read sealed cache inventory")
    try:
        cache_value = json.loads(cache_fetched.stdout)
    except json.JSONDecodeError as error:
        raise ProbeOperatorError("sealed cache inventory is not JSON") from error
    if not isinstance(cache_value, dict):
        raise ProbeOperatorError("sealed cache inventory is not an object")
    return {
        "controller_mapping": value,
        "cache_inventory": cache_value,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/tpu/probe_controller_identity.py",
        description=(
            "Read-only fresh physical-controller/JAX identity probe for one "
            "already-created v6e-16 slice."
        ),
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--tpu-name", required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--remote-uv", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--probe-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--required-lane-bytes", required=True, type=int)
    parser.add_argument(
        "--required-free-bytes",
        type=int,
        default=MINIMUM_CACHE_FREE_BYTES,
    )
    parser.add_argument("--ssh-key-file")
    parser.add_argument(
        "--internal-ip",
        action="store_true",
        help=(
            "explicitly connect through TPU VM internal addresses; required "
            "for the private v6e-16 regional slices"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact four observation commands without invoking gcloud",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    del environment
    args = _parser().parse_args(argv)
    try:
        _validate_args(args)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "process_count": PROCESS_COUNT,
                        "commands": [
                            _observation_command(args, controller)
                            for controller in range(PROCESS_COUNT)
                        ],
                        "seal_command": _seal_command(args),
                        "cache_commands": [
                            _cache_observation_command(args, controller)
                            for controller in range(PROCESS_COUNT)
                        ],
                        "seal_cache_command": _seal_cache_command(args),
                        "cloud_mutation": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if not shutil_which("gcloud"):
            raise ProbeOperatorError("gcloud is unavailable")
        mapping = _run_probe(args)
        print(json.dumps(mapping, sort_keys=True))
        return 0
    except (ProbeOperatorError, OSError, subprocess.SubprocessError) as error:
        print(f"[controller-probe] ERROR: {error}", file=sys.stderr)
        return 1


def shutil_which(command: str) -> str | None:
    """Tiny injectable seam without importing the larger operator modules."""

    path = os.environ.get("PATH", "")
    for directory in path.split(os.pathsep):
        candidate = Path(directory) / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
