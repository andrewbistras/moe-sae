#!/usr/bin/env python3
"""Build and materialize one immutable capture mirror per TPU region.

Run ``plan`` on a controller which can read the sealed scientific control
directories.  Run ``copy-and-seal`` with an explicit ``--apply`` to perform
generation-pinned, create-only server-side GCS copies.  Replays accept an
already-present destination only after its exact generation, size, CRC32C,
and ``sha256`` custom metadata have been observed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from moe_sae.contribution_artifacts import canonical_json_bytes
from moe_sae.contribution_storage import ContributionDataset
from moe_sae.contribution_transcoder_storage import RoutedInputDataset
from moe_sae.object_store import ObjectIdentity, ShardCache
from moe_sae.regional_execution import (
    MirrorDataset,
    RegionalExecutionError,
    atomic_write_evidence,
    build_regional_mirror_specification,
    seal_regional_mirror,
    validate_regional_mirror_specification,
)


class MirrorOperatorError(RuntimeError):
    """The regional mirror operator cannot continue safely."""


class _InventoryOnlyStore:
    """Object-store seam which proves planning cannot transfer bytes."""

    @staticmethod
    def _refuse(*_args: Any, **_kwargs: Any) -> None:
        raise MirrorOperatorError("mirror planning attempted an object transfer")

    put_if_absent = _refuse
    download = _refuse
    verify = _refuse
    delete = _refuse


def _read_json(path: str | Path, *, name: str) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    if not selected.is_file() or selected.is_symlink():
        raise MirrorOperatorError(f"{name} is not one regular file")
    try:
        value = json.loads(selected.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MirrorOperatorError(f"cannot read {name}") from error
    if not isinstance(value, dict):
        raise MirrorOperatorError(f"{name} must be a JSON object")
    return value


def _planning_cache(root: Path) -> ShardCache:
    return ShardCache(root, _InventoryOnlyStore(), max_bytes=1)


def _capture_datasets(args: argparse.Namespace) -> tuple[MirrorDataset, ...]:
    sidecar = _read_json(args.recomposition_sidecar, name="recomposition sidecar")
    with tempfile.TemporaryDirectory(prefix="gemma4-mirror-inventory-") as scratch:
        train_target = ContributionDataset(
            args.target_control_root,
            object_prefix=args.target_object_prefix,
            cache=_planning_cache(Path(scratch) / "target-cache"),
            recomposition_sidecar=sidecar,
        )
        eval_target = ContributionDataset(
            args.eval_target_control_root,
            object_prefix=args.eval_target_object_prefix,
            cache=_planning_cache(Path(scratch) / "eval-target-cache"),
            recomposition_sidecar=sidecar,
        )
        target_datasets = (
            MirrorDataset(
                name="train-target",
                manifest_sha256=train_target.manifest_sha256,
                source_object_prefix=args.target_object_prefix,
                objects=train_target.token_range_object_identities(
                    0,
                    train_target.total_tokens,
                ),
            ),
            MirrorDataset(
                name="eval-target",
                manifest_sha256=eval_target.manifest_sha256,
                source_object_prefix=args.eval_target_object_prefix,
                objects=eval_target.token_range_object_identities(
                    0,
                    eval_target.total_tokens,
                ),
            ),
        )
        if args.overlay_control_root is None:
            return target_datasets
        train_overlay = RoutedInputDataset(
            args.overlay_control_root,
            object_prefix=args.overlay_object_prefix,
            cache=_planning_cache(Path(scratch) / "overlay-cache"),
            target_manifest=train_target.manifest,
        )
        eval_overlay = RoutedInputDataset(
            args.eval_overlay_control_root,
            object_prefix=args.eval_overlay_object_prefix,
            cache=_planning_cache(Path(scratch) / "eval-overlay-cache"),
            target_manifest=eval_target.manifest,
        )
        return (
            *target_datasets,
            MirrorDataset(
                name="train-overlay",
                manifest_sha256=train_overlay.manifest_sha256,
                source_object_prefix=args.overlay_object_prefix,
                objects=train_overlay.token_range_object_identities(
                    0,
                    train_overlay.total_tokens,
                ),
            ),
            MirrorDataset(
                name="eval-overlay",
                manifest_sha256=eval_overlay.manifest_sha256,
                source_object_prefix=args.eval_overlay_object_prefix,
                objects=eval_overlay.token_range_object_identities(
                    0,
                    eval_overlay.total_tokens,
                ),
            ),
        )


def _plan(args: argparse.Namespace) -> int:
    overlay_values = (
        args.overlay_control_root,
        args.overlay_object_prefix,
        args.eval_overlay_control_root,
        args.eval_overlay_object_prefix,
    )
    if any(value is not None for value in overlay_values) and not all(
        value is not None for value in overlay_values
    ):
        raise MirrorOperatorError(
            "all train/eval overlay roots and prefixes must be supplied together"
        )
    plan = build_regional_mirror_specification(
        run_id=args.run_id,
        region=args.region,
        mirror_prefix=args.mirror_prefix,
        datasets=_capture_datasets(args),
    )
    atomic_write_evidence(args.output, plan)
    print(canonical_json_bytes(plan).decode("ascii"), flush=True)
    return 0


def _copy_command(
    *,
    project: str,
    source: ObjectIdentity,
    destination: str,
) -> list[str]:
    return [
        "gcloud",
        "storage",
        "cp",
        f"{source.uri}#{source.generation}",
        destination,
        f"--project={project}",
        "--if-generation-match=0",
        "--quiet",
    ]


def _describe_command(*, project: str, destination: str) -> list[str]:
    return [
        "gcloud",
        "storage",
        "objects",
        "describe",
        destination,
        f"--project={project}",
        "--format=json",
    ]


def _identity_from_describe(
    destination: str,
    source: ObjectIdentity,
    value: Mapping[str, Any],
) -> ObjectIdentity:
    try:
        # ``gcloud storage objects describe --format=json`` has emitted two
        # field layouts across SDK releases.  The JSON-API-shaped layout uses
        # ``crc32c``/``metadata``; current SDKs use
        # ``crc32c_hash``/``custom_fields``.  Accept either representation,
        # but fail closed if a response supplies contradictory aliases.
        api_crc32c = value.get("crc32c")
        cli_crc32c = value.get("crc32c_hash")
        if (
            api_crc32c is not None
            and cli_crc32c is not None
            and str(api_crc32c) != str(cli_crc32c)
        ):
            raise ValueError("conflicting crc32c aliases")
        crc32c = api_crc32c if api_crc32c is not None else cli_crc32c

        api_metadata = value.get("metadata")
        cli_metadata = value.get("custom_fields")
        if (
            api_metadata is not None
            and cli_metadata is not None
            and api_metadata != cli_metadata
        ):
            raise ValueError("conflicting metadata aliases")
        metadata = api_metadata if api_metadata is not None else cli_metadata
        if not isinstance(metadata, Mapping) or crc32c is None:
            raise ValueError("missing object identity fields")
        result = ObjectIdentity(
            uri=destination,
            generation=int(value["generation"]),
            size_bytes=int(value["size"]),
            crc32c=str(crc32c),
            sha256=str(metadata["sha256"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise MirrorOperatorError(
            f"mirror object metadata is incomplete: {destination}"
        ) from error
    if (
        result.size_bytes,
        result.crc32c,
        result.sha256,
    ) != (
        source.size_bytes,
        source.crc32c,
        source.sha256,
    ):
        raise MirrorOperatorError(
            f"mirror object differs from source identity: {destination}"
        )
    return result


def _describe(
    *,
    project: str,
    destination: str,
    source: ObjectIdentity,
) -> ObjectIdentity:
    completed = subprocess.run(
        _describe_command(project=project, destination=destination),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise MirrorOperatorError(
            f"cannot describe mirror destination {destination}{suffix}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise MirrorOperatorError(
            f"mirror describe returned non-JSON: {destination}"
        ) from error
    if not isinstance(value, Mapping):
        raise MirrorOperatorError("mirror describe must return an object")
    return _identity_from_describe(destination, source, value)


def _copy_one(
    *,
    project: str,
    source: ObjectIdentity,
    destination: str,
) -> ObjectIdentity:
    completed = subprocess.run(
        _copy_command(
            project=project,
            source=source,
            destination=destination,
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    # A replay sees the create-only copy fail.  It is accepted only if the
    # exact immutable destination can subsequently be described and matched.
    try:
        return _describe(
            project=project,
            destination=destination,
            source=source,
        )
    except MirrorOperatorError as error:
        if completed.returncode == 0:
            raise
        detail = completed.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise MirrorOperatorError(
            f"create-only mirror copy failed for {destination}{suffix}"
        ) from error


def _copy_and_seal(args: argparse.Namespace) -> int:
    if not args.apply:
        raise MirrorOperatorError(
            "copy-and-seal mutates GCS and requires explicit --apply"
        )
    plan = validate_regional_mirror_specification(
        _read_json(args.plan, name="regional mirror plan")
    )
    if plan["region"] != args.region:
        raise MirrorOperatorError("plan region differs from --region")
    observations: dict[str, ObjectIdentity] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _copy_one,
                project=args.project,
                source=ObjectIdentity(**entry["source"]),
                destination=entry["mirror_uri"],
            ): entry["mirror_uri"]
            for entry in plan["objects"]
        }
        try:
            for future in as_completed(futures):
                destination = futures[future]
                observations[destination] = future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    seal = seal_regional_mirror(plan, observations)
    atomic_write_evidence(args.output, seal)
    print(canonical_json_bytes(seal).decode("ascii"), flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generation-pinned regional capture mirror operator"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--region", required=True)
    plan.add_argument("--mirror-prefix", required=True)
    plan.add_argument("--target-control-root", required=True)
    plan.add_argument("--target-object-prefix", required=True)
    plan.add_argument("--eval-target-control-root", required=True)
    plan.add_argument("--eval-target-object-prefix", required=True)
    plan.add_argument("--recomposition-sidecar", required=True)
    plan.add_argument("--overlay-control-root")
    plan.add_argument("--overlay-object-prefix")
    plan.add_argument("--eval-overlay-control-root")
    plan.add_argument("--eval-overlay-object-prefix")
    plan.add_argument("--output", required=True)
    plan.set_defaults(handler=_plan)

    copy = commands.add_parser("copy-and-seal")
    copy.add_argument("--project", required=True)
    copy.add_argument("--region", required=True)
    copy.add_argument("--plan", required=True)
    copy.add_argument("--output", required=True)
    copy.add_argument("--workers", type=int, default=16)
    copy.add_argument(
        "--apply",
        action="store_true",
        help="explicitly authorize the planned create-only GCS copies",
    )
    copy.set_defaults(handler=_copy_and_seal)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if getattr(args, "workers", 1) <= 0:
            raise MirrorOperatorError("--workers must be positive")
        return int(args.handler(args))
    except (
        MirrorOperatorError,
        RegionalExecutionError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(f"[regional-mirror] ERROR: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
