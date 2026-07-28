"""Create-only delivery of small control files to a regional TPU Filestore.

Bulk capture shards belong in the generation-pinned regional GCS mirror.  A
worker nevertheless needs a handful of POSIX files before it can open that
mirror: sealed manifests, the recomposition sidecar, finalized run configs,
and the regional mirror seal.  This module packages only explicitly named
regular files and applies them without ever replacing an existing path.

The bundle is safe to send to worker zero after ``deploy_workers.sh`` has
mounted the region's Filestore.  All four controllers then observe the same
files through that mount.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from typing import Any, Mapping

from moe_sae.contribution_artifacts import canonical_json_bytes, canonical_sha256


STAGE_SCHEMA_VERSION = "gemma4-regional-control-stage-v1"
_PLAN_NAME = "stage-plan.json"
_PAYLOAD_ROOT = "payload"


class RegionalControlStageError(RuntimeError):
    """A regional control-plane delivery is ambiguous or differs on replay."""


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != value
    ):
        raise RegionalControlStageError(
            "control-stage paths must be normalized relative POSIX paths"
        )
    return path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: Path, *, owner: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RegionalControlStageError(f"{owner} is not one regular file: {path}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise RegionalControlStageError(f"cannot read {owner}: {path}") from error


def _validate_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RegionalControlStageError("control-stage plan must be an object")
    selected = dict(value)
    if set(selected) != {
        "schema_version",
        "files",
        "unique_payload_bytes",
        "stage_sha256",
    }:
        raise RegionalControlStageError("control-stage plan fields changed")
    claimed = selected.pop("stage_sha256")
    if (
        not isinstance(claimed, str)
        or len(claimed) != 64
        or any(character not in "0123456789abcdef" for character in claimed)
        or canonical_sha256(selected) != claimed
    ):
        raise RegionalControlStageError("control-stage plan self-hash is invalid")
    if selected["schema_version"] != STAGE_SCHEMA_VERSION:
        raise RegionalControlStageError("control-stage schema is unsupported")
    rows = selected["files"]
    if not isinstance(rows, list) or not rows:
        raise RegionalControlStageError("control-stage plan has no files")
    paths: set[str] = set()
    payloads: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise RegionalControlStageError("control-stage file row changed")
        path = str(_relative_path(row["path"]))
        digest = row["sha256"]
        size = row["size_bytes"]
        if (
            path in paths
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise RegionalControlStageError("control-stage file row is invalid")
        prior = payloads.get(digest)
        if prior is not None and prior != size:
            raise RegionalControlStageError(
                "one payload digest has conflicting byte counts"
            )
        paths.add(path)
        payloads[digest] = size
    if rows != sorted(rows, key=lambda row: row["path"]):
        raise RegionalControlStageError("control-stage file rows are not ordered")
    expected_bytes = sum(payloads.values())
    if selected["unique_payload_bytes"] != expected_bytes:
        raise RegionalControlStageError("control-stage byte accounting changed")
    return {**selected, "stage_sha256": claimed}


def build_control_stage(
    source_root: str | Path,
    relative_files: Sequence[str],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build a strict plan and de-duplicated payload map without writing."""

    selected_root = Path(source_root).expanduser()
    if selected_root.is_symlink():
        raise RegionalControlStageError("source root must not be a symlink")
    root = selected_root.resolve()
    if not root.is_dir():
        raise RegionalControlStageError("source root is not one directory")
    if not relative_files or len(set(relative_files)) != len(relative_files):
        raise RegionalControlStageError(
            "relative files must be a nonempty duplicate-free sequence"
        )
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for raw in relative_files:
        relative = _relative_path(raw)
        candidate = root.joinpath(*relative.parts)
        try:
            candidate.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as error:
            raise RegionalControlStageError(
                f"control-stage source escapes or is absent: {raw}"
            ) from error
        payload = _read_regular(candidate, owner=f"control-stage source {raw}")
        digest = _sha256_bytes(payload)
        prior = payloads.get(digest)
        if prior is not None and prior != payload:
            raise RegionalControlStageError("SHA-256 collision in control-stage inputs")
        payloads[digest] = payload
        rows.append(
            {
                "path": str(relative),
                "sha256": digest,
                "size_bytes": len(payload),
            }
        )
    unsigned = {
        "schema_version": STAGE_SCHEMA_VERSION,
        "files": sorted(rows, key=lambda row: row["path"]),
        "unique_payload_bytes": sum(len(payload) for payload in payloads.values()),
    }
    plan = {**unsigned, "stage_sha256": canonical_sha256(unsigned)}
    return _validate_plan(plan), payloads


def write_control_stage_bundle(
    source_root: str | Path,
    relative_files: Sequence[str],
    destination: str | Path,
) -> dict[str, Any]:
    """Create one immutable tar bundle containing only named control files."""

    plan, payloads = build_control_stage(source_root, relative_files)
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RegionalControlStageError(f"bundle destination exists: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with tarfile.open(temporary, "x") as archive:
            encoded = canonical_json_bytes(plan)
            info = tarfile.TarInfo(_PLAN_NAME)
            info.size = len(encoded)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(encoded))
            for digest, payload in sorted(payloads.items()):
                info = tarfile.TarInfo(f"{_PAYLOAD_ROOT}/{digest}")
                info.size = len(payload)
                info.mode = 0o600
                archive.addfile(info, io.BytesIO(payload))
        os.link(temporary, output)
    except FileExistsError as error:
        raise RegionalControlStageError(
            f"bundle destination was concurrently created: {output}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return plan


def _read_bundle(
    bundle: str | Path,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    selected = Path(bundle).expanduser().resolve()
    if selected.is_symlink() or not selected.is_file():
        raise RegionalControlStageError("control-stage bundle is not a regular file")
    try:
        with tarfile.open(selected, "r:*") as archive:
            members = archive.getmembers()
            if any(
                not member.isfile()
                or member.name.startswith("/")
                or ".." in PurePosixPath(member.name).parts
                for member in members
            ):
                raise RegionalControlStageError(
                    "control-stage archive contains a non-regular member"
                )
            names = [member.name for member in members]
            if len(names) != len(set(names)) or _PLAN_NAME not in names:
                raise RegionalControlStageError(
                    "control-stage archive inventory is invalid"
                )
            plan_member = archive.extractfile(_PLAN_NAME)
            assert plan_member is not None
            try:
                plan = _validate_plan(json.loads(plan_member.read()))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RegionalControlStageError(
                    "control-stage plan is not canonical JSON"
                ) from error
            expected = {
                f"{_PAYLOAD_ROOT}/{row['sha256']}"
                for row in plan["files"]
            }
            if set(names) != {_PLAN_NAME, *expected}:
                raise RegionalControlStageError(
                    "control-stage archive payload inventory changed"
                )
            payloads: dict[str, bytes] = {}
            for row in plan["files"]:
                digest = row["sha256"]
                if digest in payloads:
                    continue
                handle = archive.extractfile(f"{_PAYLOAD_ROOT}/{digest}")
                assert handle is not None
                payload = handle.read()
                if len(payload) != row["size_bytes"] or _sha256_bytes(payload) != digest:
                    raise RegionalControlStageError(
                        "control-stage payload identity is invalid"
                    )
                payloads[digest] = payload
    except (OSError, tarfile.TarError) as error:
        raise RegionalControlStageError("cannot read control-stage bundle") from error
    return plan, payloads


def _publish_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RegionalControlStageError(f"destination is a symlink: {path}")
    if path.exists():
        existing = _read_regular(path, owner="existing control-stage destination")
        if existing != payload:
            raise RegionalControlStageError(
                f"existing regional control file differs: {path}"
            )
        return
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.tmp-",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = _read_regular(
                path, owner="concurrent control-stage destination"
            )
            if existing != payload:
                raise RegionalControlStageError(
                    f"concurrent regional control file differs: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def apply_control_stage_bundle(
    bundle: str | Path,
    destination_root: str | Path,
) -> dict[str, Any]:
    """Apply a bundle idempotently without replacing any existing file."""

    plan, payloads = _read_bundle(bundle)
    selected_root = Path(destination_root).expanduser()
    if selected_root.is_symlink():
        raise RegionalControlStageError("destination root must not be a symlink")
    root = selected_root.resolve()
    if not root.is_dir():
        raise RegionalControlStageError("destination root is not one directory")
    for row in plan["files"]:
        relative = _relative_path(row["path"])
        destination = root.joinpath(*relative.parts)
        try:
            destination.parent.resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            # Parents may not exist yet.  Existing ancestors are checked again
            # after mkdir; no symlink may redirect the resolved parent.
            destination.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = destination.parent.resolve(strict=True)
        try:
            resolved_parent.relative_to(root)
        except ValueError as error:
            raise RegionalControlStageError(
                f"control-stage destination escapes root: {relative}"
            ) from error
        _publish_create_only(destination, payloads[row["sha256"]])
    return plan


__all__ = [
    "RegionalControlStageError",
    "STAGE_SCHEMA_VERSION",
    "apply_control_stage_bundle",
    "build_control_stage",
    "write_control_stage_bundle",
]
