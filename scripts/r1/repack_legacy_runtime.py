#!/usr/bin/env python3
"""Derive a legacy TPU runtime without rebuilding the pinned base image.

This program is deliberately an offline, single-host transformation.  It is
intended to run as root on a disposable VM booted from the exact qualified
custom image.  The caller supplies the deterministic ``build_context.py`` tar
for a clean Git HEAD and the immutable identities resolved by the controller.

Only the ``moe-sae-r1-worker`` wheel and ``/opt/moe-sae-r1/source`` may change.
Every other installed distribution is measured before and after and must
remain byte-identical.  The output is a deterministic, content-addressable
``tar.gz`` containing the complete ``/opt/moe-sae-r1`` tree; publication and
GCS generation pinning are intentionally outside this program.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import email.parser
import fcntl
import functools
import gzip
import hashlib
import io
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ATTESTATION_SCHEMA = "moe-sae.r1.derived-runtime-attestation.v1"
RESULT_SCHEMA = "moe-sae.r1.derived-runtime-result.v1"
REPACKER_SCHEMA = "moe-sae.r1.legacy-runtime-repacker.v1"
REPACKER_PROTOCOL_VERSION = 1
CONTEXT_MANIFEST_SCHEMA = "moe-sae.r1.runtime-context-manifest.v1"
WORKER_DISTRIBUTION = "moe-sae-r1-worker"
WORKER_VERSION = "0.1.0"
SOURCE_DATE_EPOCH = 315_532_800  # 1980-01-01, also the ZIP timestamp floor.
MAX_CONTEXT_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_CONTEXT_FILE_BYTES = 2 * 1024 * 1024
MAX_CONTEXT_PAYLOAD_BYTES = 8 * 1024 * 1024
MAX_CONTEXT_FILES = 128
MAX_RUNTIME_BYTES = 12 * 1024 * 1024 * 1024
MAX_RUNTIME_MEMBERS = 250_000
METADATA_IMAGE_URL = "http://metadata.google.internal/computeMetadata/v1/instance/image"
RENAME_EXCHANGE = 2
AT_FDCWD = -100
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_IMAGE_RESOURCE = re.compile(
    r"projects/(?:[a-z][a-z0-9-]{4,61}[a-z0-9]|[1-9][0-9]{5,29})"
    r"/global/images/"
    r"[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?"
)
_SAFE_ARCHIVE_MODE = {0o644, 0o755}
_CRITICAL_PROTECTED = {
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "libtpu": "0.0.43",
    "maxtext": "0.2.3",
}
_INERT_CONTEXT_DELTA_ALLOWLIST = frozenset(
    {
        "image/measure_image.py",
        "image/provision.sh",
    }
)
_IMPORT_PROBES = (
    "moe_sae.r1.runtime_entrypoint",
    "moe_sae.r1.legacy_slice_runtime",
    "moe_sae.r1.jax_prefix_restore",
    "moe_sae.r1.maxtext_prefix_runtime",
    "moe_sae.r1.prefix_benchmark",
    "moe_sae.r1.prefix_benchmark_runner",
    "moe_sae.r1.runtime_supervisor",
)
_WORK_ROOT_MARKER = b"moe-sae.r1.runtime-repack-work-root.v1\n"


class RepackError(RuntimeError):
    """The requested runtime transformation did not satisfy its contract."""


CommandRunner = Callable[
    [Sequence[str], Mapping[str, str] | None], subprocess.CompletedProcess[str]
]


def _remove_owned_work_root(path: Path) -> None:
    marker = path / ".repacker-owned"
    try:
        root_stat = path.lstat()
        marker_stat = marker.lstat()
        owned = (
            stat.S_ISDIR(root_stat.st_mode)
            and not path.is_symlink()
            and stat.S_ISREG(marker_stat.st_mode)
            and not marker.is_symlink()
            and root_stat.st_uid == os.geteuid()
            and marker_stat.st_uid == os.geteuid()
            and marker.read_bytes() == _WORK_ROOT_MARKER
        )
    except FileNotFoundError:
        owned = False
    if owned:
        shutil.rmtree(path)


def require_pristine_derived_state(prefix: Path) -> None:
    attestation = prefix / "derived-runtime-attestation.json"
    retained = prefix / "derived"
    if attestation.exists() or retained.exists():
        raise RepackError("refusing to reuse a runtime after overlay mutation began")


def _cleanup_owned_work_root(
    function: Callable[..., Mapping[str, object]],
) -> Callable[..., Mapping[str, object]]:
    """Clean only a work root proven to have been created by this process."""

    @functools.wraps(function)
    def wrapped(
        request: RepackRequest,
        *args: object,
        **kwargs: object,
    ) -> Mapping[str, object]:
        try:
            return function(request, *args, **kwargs)
        finally:
            _remove_owned_work_root(request.work_root)

    return wrapped


@dataclass(frozen=True, slots=True)
class ContextIdentity:
    commit: str
    manifest_sha256: str
    archive_sha256: str
    file_count: int
    total_payload_bytes: int
    records: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class WheelIdentity:
    path: Path
    name: str
    sha256: str
    size: int
    record_sha256: str
    payload_sha256: str
    package_payload_sha256: str


@dataclass(frozen=True, slots=True)
class ArchiveIdentity:
    sha256: str
    size: int
    member_count: int
    regular_bytes: int
    max_member_name_bytes: int
    max_symlink_target_bytes: int


@dataclass(frozen=True, slots=True)
class RepackRequest:
    context_archive: Path
    context_archive_sha256: str
    expected_repacker_sha256: str
    expected_boot_image: str
    controller_admitted_boot_source_image_id: int
    expected_image_recipe_sha256: str
    expected_image_observation_sha256: str
    controller_admitted_base_runtime_archive_sha256: str
    expected_base_context_commit: str
    prefix: Path
    identity_root: Path
    uv: Path
    output: Path
    result: Path
    work_root: Path


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise RepackError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _canonical_json(
    raw: bytes,
    *,
    name: str,
    trailing_newline: bool = True,
) -> dict[str, object]:
    try:
        value = json.loads(raw, object_pairs_hook=_strict_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RepackError(f"{name} is not strict JSON") from error
    if not isinstance(value, dict):
        raise RepackError(f"{name} must be a JSON object")
    canonical = _canonical_bytes(value)
    if not trailing_newline:
        canonical = canonical.removesuffix(b"\n")
    if canonical != raw:
        raise RepackError(f"{name} is not canonical JSON")
    return value


def _require_sha256(value: str, *, name: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise RepackError(f"{name} is not a lowercase SHA-256")
    return value


def _require_revision(value: str, *, name: str) -> str:
    if _REVISION.fullmatch(value) is None:
        raise RepackError(f"{name} is not a full lowercase Git revision")
    return value


def _safe_archive_name(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 0x20 for character in value)
        or "\\" in value
        or "//" in value
        or re.fullmatch(r"[A-Za-z0-9._/-]+", value) is None
        or len(value.encode("ascii")) > 240
    ):
        raise RepackError(f"{name} is not a safe relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RepackError(f"{name} is not a normalized relative path")
    if posixpath.normpath(value) != value:
        raise RepackError(f"{name} is not canonical")
    return value


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _require_safe_output_parent(path: Path, *, name: str) -> None:
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise RepackError(f"{name} parent must be an existing real directory")
    cursor = parent
    while True:
        observed = cursor.lstat()
        if not stat.S_ISDIR(observed.st_mode) or cursor.is_symlink():
            raise RepackError(f"{name} parent chain contains a symlink")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    parent_stat = parent.lstat()
    parent_mode = stat.S_IMODE(parent_stat.st_mode)
    if parent_stat.st_uid != os.geteuid():
        raise RepackError(f"{name} parent is not owned by the repacker uid")
    if parent_mode & 0o022 and not parent_mode & stat.S_ISVTX:
        raise RepackError(f"{name} parent is writable without the sticky bit")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, raw: bytes, *, mode: int) -> None:
    """Publish one fresh file without ever replacing an existing name."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise RepackError(f"refusing to replace existing output: {path}") from error
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> None:
        del request, file_pointer, code, message, headers, new_url


def _read_metadata_image(url: str) -> str:
    if url != METADATA_IMAGE_URL:
        raise RepackError("metadata reader accepts only the exact GCE image URL")
    request = urllib.request.Request(
        url,
        headers={"Metadata-Flavor": "Google"},
        method="GET",
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect(),
    )
    try:
        with opener.open(request, timeout=10) as response:
            if response.headers.get("Metadata-Flavor") != "Google":
                raise RepackError("metadata response lacks Metadata-Flavor")
            raw = response.read(1024)
            if response.read(1):
                raise RepackError("metadata image response exceeds its bound")
    except RepackError:
        raise
    except Exception as error:
        raise RepackError("could not read the boot-image metadata") from error
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise RepackError("boot-image metadata is not ASCII") from error
    if _IMAGE_RESOURCE.fullmatch(value) is None:
        raise RepackError("boot-image metadata is malformed")
    return value


def _validate_context_manifest(
    raw: bytes,
    *,
    payloads: Mapping[str, bytes],
    modes: Mapping[str, int],
) -> ContextIdentity:
    manifest = _canonical_json(raw, name="CONTEXT-MANIFEST.json")
    if set(manifest) != {
        "schema",
        "commit",
        "files",
        "production_qualified",
    }:
        raise RepackError("context manifest fields drifted")
    if manifest["schema"] != CONTEXT_MANIFEST_SCHEMA:
        raise RepackError("context manifest schema drifted")
    raw_commit = manifest["commit"]
    if not isinstance(raw_commit, str):
        raise RepackError("context commit must be a string")
    commit = _require_revision(raw_commit, name="context commit")
    if manifest["production_qualified"] is not False:
        raise RepackError("a build context cannot claim production qualification")
    records = manifest["files"]
    if not isinstance(records, list) or not (1 <= len(records) <= MAX_CONTEXT_FILES):
        raise RepackError("context manifest file count is invalid")
    seen_sources: set[str] = set()
    seen_archives: set[str] = set()
    normalized: list[Mapping[str, object]] = []
    total = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "source",
            "archive",
            "mode",
            "size",
            "sha256",
            "git_blob",
        }:
            raise RepackError(f"context record {index} is malformed")
        source = _safe_archive_name(record["source"], name="context source")
        archive = _safe_archive_name(record["archive"], name="context archive")
        if source in seen_sources or archive in seen_archives:
            raise RepackError("context records contain a duplicate path")
        if archive.casefold() in {item.casefold() for item in seen_archives}:
            raise RepackError("context archive paths case-fold collide")
        seen_sources.add(source)
        seen_archives.add(archive)
        mode_text = record["mode"]
        if mode_text not in {"0644", "0755"}:
            raise RepackError("context record mode is invalid")
        expected_mode = int(mode_text, 8)
        payload = payloads.get(archive)
        if payload is None or modes.get(archive) != expected_mode:
            raise RepackError(f"context payload/mode is absent: {archive}")
        if (
            type(record["size"]) is not int
            or record["size"] != len(payload)
            or record["sha256"] != _sha256_bytes(payload)
            or not isinstance(record["git_blob"], str)
            or record["git_blob"] != _git_blob_sha1(payload)
        ):
            raise RepackError(f"context record does not match payload: {archive}")
        total += len(payload)
        if total > MAX_CONTEXT_PAYLOAD_BYTES:
            raise RepackError("context payload exceeds its total bound")
        normalized.append(record)
    if seen_archives != set(payloads):
        raise RepackError("context manifest does not own the exact payload set")
    return ContextIdentity(
        commit=commit,
        manifest_sha256=_sha256_bytes(raw),
        archive_sha256="",
        file_count=len(records),
        total_payload_bytes=total,
        records=tuple(sorted(normalized, key=lambda item: str(item["archive"]))),
    )


def validate_and_extract_context(
    archive_path: Path,
    *,
    expected_sha256: str,
    destination: Path,
) -> ContextIdentity:
    """Validate the deterministic build-context tar and extract regular files."""

    expected_sha256 = _require_sha256(expected_sha256, name="context archive SHA-256")
    try:
        descriptor = os.open(
            archive_path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise RepackError(
            "context archive cannot be opened without following links"
        ) from error
    with os.fdopen(descriptor, "rb") as archive_stream:
        before = os.fstat(archive_stream.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > MAX_CONTEXT_ARCHIVE_BYTES
        ):
            raise RepackError("context archive is not a bounded regular file")
        archive_raw = archive_stream.read(MAX_CONTEXT_ARCHIVE_BYTES + 1)
        after = os.fstat(archive_stream.fileno())
        if (
            len(archive_raw) != before.st_size
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
        ):
            raise RepackError("context archive changed while it was read")
    actual_sha256 = _sha256_bytes(archive_raw)
    if actual_sha256 != expected_sha256:
        raise RepackError("context archive SHA-256 mismatch")
    if destination.exists():
        raise RepackError("context extraction destination already exists")
    destination.mkdir(mode=0o700, parents=False)
    payloads: dict[str, bytes] = {}
    modes: dict[str, int] = {}
    manifest_raw: bytes | None = None
    ordered_names: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:") as archive:
        members = archive.getmembers()
        if not (2 <= len(members) <= MAX_CONTEXT_FILES + 1):
            raise RepackError("context archive member count is invalid")
        for member in members:
            name = _safe_archive_name(member.name, name="context tar member")
            if name in ordered_names:
                raise RepackError("context tar contains a duplicate member")
            ordered_names.append(name)
            if (
                not member.isreg()
                or member.linkname
                or member.pax_headers
                or member.uid != 0
                or member.gid != 0
                or member.uname
                or member.gname
                or member.mtime != 0
                or member.size < 0
                or member.size > MAX_CONTEXT_FILE_BYTES
            ):
                raise RepackError(f"unsafe context tar metadata: {name}")
            expected_mode = 0o644 if name == "CONTEXT-MANIFEST.json" else member.mode
            if member.mode != expected_mode or member.mode not in _SAFE_ARCHIVE_MODE:
                raise RepackError(f"unsafe context tar mode: {name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise RepackError(f"context member is unreadable: {name}")
            raw = stream.read(MAX_CONTEXT_FILE_BYTES + 1)
            if len(raw) != member.size:
                raise RepackError(f"context member size drifted: {name}")
            if name == "CONTEXT-MANIFEST.json":
                manifest_raw = raw
            else:
                payloads[name] = raw
                modes[name] = member.mode
        if ordered_names != sorted(ordered_names):
            raise RepackError("context tar members are not canonically ordered")
    if manifest_raw is None:
        raise RepackError("context archive lacks its manifest")
    identity = _validate_context_manifest(
        manifest_raw,
        payloads=payloads,
        modes=modes,
    )
    for name in sorted(payloads):
        target = destination.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payloads[name])
        os.chmod(target, modes[name])
    manifest_path = destination / "CONTEXT-MANIFEST.json"
    manifest_path.write_bytes(manifest_raw)
    os.chmod(manifest_path, 0o644)
    return ContextIdentity(
        commit=identity.commit,
        manifest_sha256=identity.manifest_sha256,
        archive_sha256=actual_sha256,
        file_count=identity.file_count,
        total_payload_bytes=identity.total_payload_bytes,
        records=identity.records,
    )


def validate_context_directory(path: Path) -> ContextIdentity:
    """Validate an already-installed context and reject unowned files."""

    manifest_path = path / "CONTEXT-MANIFEST.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RepackError("installed context manifest is absent")
    manifest_raw = manifest_path.read_bytes()
    manifest = _canonical_json(manifest_raw, name="installed context manifest")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise RepackError("installed context records are malformed")
    payloads: dict[str, bytes] = {}
    modes: dict[str, int] = {}
    expected_files = {"CONTEXT-MANIFEST.json"}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("archive"), str):
            raise RepackError("installed context record is malformed")
        name = _safe_archive_name(record["archive"], name="installed context path")
        file_path = path.joinpath(*PurePosixPath(name).parts)
        if not file_path.is_file() or file_path.is_symlink():
            raise RepackError(f"installed context file is absent: {name}")
        payloads[name] = file_path.read_bytes()
        modes[name] = _mode(file_path)
        expected_files.add(name)
    actual_files: set[str] = set()
    for root, directories, files in os.walk(path, followlinks=False):
        directories.sort()
        files.sort()
        root_path = Path(root)
        for directory in directories:
            candidate = root_path / directory
            if candidate.is_symlink():
                raise RepackError("installed context contains a directory symlink")
        for file_name in files:
            candidate = root_path / file_name
            relative = candidate.relative_to(path).as_posix()
            actual_files.add(relative)
    if actual_files != expected_files:
        raise RepackError("installed context contains unowned or missing files")
    identity = _validate_context_manifest(
        manifest_raw,
        payloads=payloads,
        modes=modes,
    )
    return ContextIdentity(
        commit=identity.commit,
        manifest_sha256=identity.manifest_sha256,
        archive_sha256="",
        file_count=identity.file_count,
        total_payload_bytes=identity.total_payload_bytes,
        records=identity.records,
    )


def _normalize_context_directory_modes(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise RepackError("context root is not a real directory")
    directories = [path]
    for root, child_directories, _ in os.walk(path, followlinks=False):
        child_directories.sort()
        root_path = Path(root)
        for child in child_directories:
            candidate = root_path / child
            if candidate.is_symlink() or not candidate.is_dir():
                raise RepackError("context contains an unsafe directory")
            directories.append(candidate)
    for directory in directories:
        os.chmod(directory, 0o755)


def require_worker_only_context_delta(
    base: Path, target: Path
) -> tuple[Mapping[str, object], ...]:
    """Validate the closed base-to-overlay context mutation surface.

    Worker Python may be added, removed, or changed.  The non-worker path set
    and every mode remain exact.  Two image-bake validators may change because
    they describe a future full bake but are inert in this overlay: neither is
    executed or installed here.  Their exact before/after hashes are returned
    for the derived attestation.
    """

    base_identity = validate_context_directory(base)
    target_identity = validate_context_directory(target)
    base_records = {str(item["archive"]): item for item in base_identity.records}
    target_records = {str(item["archive"]): item for item in target_identity.records}
    base_non_worker = {
        name: record
        for name, record in base_records.items()
        if not name.startswith("src/moe_sae/")
    }
    target_non_worker = {
        name: record
        for name, record in target_records.items()
        if not name.startswith("src/moe_sae/")
    }
    if set(base_non_worker) != set(target_non_worker):
        raise RepackError("target added or removed a non-worker context input")
    inert_deltas: list[Mapping[str, object]] = []
    for name in sorted(base_non_worker):
        base_record = base_non_worker[name]
        target_record = target_non_worker[name]
        if base_record["mode"] != target_record["mode"]:
            raise RepackError(f"target changed a non-worker context input: {name}")
        if base_record["sha256"] != target_record["sha256"]:
            if name not in _INERT_CONTEXT_DELTA_ALLOWLIST:
                raise RepackError(f"target changed a non-worker context input: {name}")
            inert_deltas.append(
                {
                    "path": name,
                    "mode": base_record["mode"],
                    "base_sha256": base_record["sha256"],
                    "target_sha256": target_record["sha256"],
                }
            )
    if not any(name.startswith("src/moe_sae/") for name in target_records):
        raise RepackError("target context contains no worker Python")
    return tuple(inert_deltas)


def _context_worker_package_payload_sha256(
    identity: ContextIdentity,
) -> str:
    payload: list[dict[str, object]] = []
    for record in identity.records:
        archive = record["archive"]
        if isinstance(archive, str) and archive.startswith("src/moe_sae/"):
            digest = record["sha256"]
            size = record["size"]
            if (
                not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
                or type(size) is not int
            ):
                raise RepackError("context worker payload identity is malformed")
            payload.append(
                {
                    "name": archive.removeprefix("src/"),
                    "sha256": digest,
                    "size": size,
                }
            )
    if not payload:
        raise RepackError("context contains no worker package payload")
    return _sha256_bytes(
        _canonical_bytes(sorted(payload, key=lambda item: str(item["name"])))
    )


def _default_runner(
    argv: Sequence[str], environment: Mapping[str, str] | None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=True,
        text=True,
        capture_output=True,
        env=None if environment is None else dict(environment),
    )


def _build_environment(work_root: Path) -> dict[str, str]:
    return {
        "HOME": str(work_root / "home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "SOURCE_DATE_EPOCH": str(SOURCE_DATE_EPOCH),
        "UV_CACHE_DIR": str(work_root / "uv-cache"),
        "UV_LINK_MODE": "copy",
        "UV_NO_CONFIG": "1",
        "UV_OFFLINE": "1",
        "UV_PYTHON_DOWNLOADS": "never",
    }


def _wheel_files(directory: Path) -> list[Path]:
    return sorted(
        (item for item in directory.iterdir() if item.is_file()),
        key=lambda item: item.name,
    )


def build_worker_wheel(
    source: Path,
    *,
    runtime_python: Path,
    uv: Path,
    work_root: Path,
    runner: CommandRunner = _default_runner,
) -> WheelIdentity:
    """Build the worker wheel twice and require byte determinism."""

    build_a = work_root / "wheel-a"
    build_b = work_root / "wheel-b"
    build_a.mkdir(mode=0o700)
    build_b.mkdir(mode=0o700)
    environment = _build_environment(work_root)
    for output in (build_a, build_b):
        runner(
            (
                str(uv),
                "build",
                "--offline",
                "--no-config",
                "--no-build-isolation",
                "--no-create-gitignore",
                "--no-python-downloads",
                "--wheel",
                "--python",
                str(runtime_python),
                "--out-dir",
                str(output),
                str(source),
            ),
            environment,
        )
    files_a = _wheel_files(build_a)
    files_b = _wheel_files(build_b)
    if len(files_a) != 1 or len(files_b) != 1:
        raise RepackError("worker builds did not each emit one wheel")
    if files_a[0].name != files_b[0].name:
        raise RepackError("worker wheel names are not deterministic")
    raw_a = files_a[0].read_bytes()
    raw_b = files_b[0].read_bytes()
    if raw_a != raw_b:
        raise RepackError("worker wheel build is not byte deterministic")
    identity = validate_worker_wheel(files_a[0])
    return identity


def _wheel_record_rows(raw: bytes) -> list[list[str]]:
    import csv

    try:
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as error:
        raise RepackError("wheel RECORD is malformed") from error
    if any(len(row) != 3 for row in rows):
        raise RepackError("wheel RECORD row is malformed")
    return rows


def validate_worker_wheel(path: Path) -> WheelIdentity:
    """Require a pure, internally hashed worker wheel."""

    expected_name = "moe_sae_r1_worker-0.1.0-py3-none-any.whl"
    if path.name != expected_name:
        raise RepackError(f"worker wheel name must be {expected_name}")
    raw = path.read_bytes()
    payload_digests: list[tuple[str, str, int]] = []
    package_digests: list[tuple[str, str, int]] = []
    with zipfile.ZipFile(io.BytesIO(raw), mode="r") as wheel:
        infos = wheel.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise RepackError("wheel members are duplicated")
        dist_info = "moe_sae_r1_worker-0.1.0.dist-info/"
        allowed_prefixes = ("moe_sae/", dist_info)
        for info in infos:
            name = _safe_archive_name(info.filename.rstrip("/"), name="wheel member")
            if not any(
                name == prefix.rstrip("/") or name.startswith(prefix)
                for prefix in allowed_prefixes
            ):
                raise RepackError(f"wheel contains a foreign payload: {name}")
            if name.startswith("moe_sae/") and not (
                info.is_dir() or name.endswith(".py")
            ):
                raise RepackError(
                    f"worker wheel contains non-Python package data: {name}"
                )
            file_type = (info.external_attr >> 16) & 0o170000
            if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise RepackError(f"wheel contains a non-regular payload: {name}")
            if info.file_size > MAX_CONTEXT_FILE_BYTES:
                raise RepackError(f"wheel member is oversized: {name}")
        required = {
            f"{dist_info}METADATA",
            f"{dist_info}WHEEL",
            f"{dist_info}RECORD",
        }
        if not required.issubset(names):
            raise RepackError("wheel metadata is incomplete")
        wheel_metadata = email.parser.BytesParser().parsebytes(
            wheel.read(f"{dist_info}WHEEL")
        )
        if wheel_metadata.get("Root-Is-Purelib") != "true" or wheel_metadata.get_all(
            "Tag"
        ) != ["py3-none-any"]:
            raise RepackError("worker wheel is not pure py3-none-any")
        metadata = email.parser.BytesParser().parsebytes(
            wheel.read(f"{dist_info}METADATA")
        )
        if (
            metadata.get("Name") != WORKER_DISTRIBUTION
            or metadata.get("Version") != WORKER_VERSION
        ):
            raise RepackError("worker wheel name/version drifted")
        record_name = f"{dist_info}RECORD"
        record_raw = wheel.read(record_name)
        rows = _wheel_record_rows(record_raw)
        row_map = {row[0]: row[1:] for row in rows}
        if len(row_map) != len(rows) or set(row_map) != {
            name for name in names if not name.endswith("/")
        }:
            raise RepackError("wheel RECORD does not own every file")
        for name in sorted(row_map):
            hash_text, size_text = row_map[name]
            member_raw = wheel.read(name)
            if name == record_name:
                if hash_text or size_text:
                    raise RepackError("wheel RECORD self-row must be unhashed")
            else:
                digest = (
                    base64.urlsafe_b64encode(hashlib.sha256(member_raw).digest())
                    .rstrip(b"=")
                    .decode("ascii")
                )
                if hash_text != f"sha256={digest}" or size_text != str(len(member_raw)):
                    raise RepackError(f"wheel RECORD hash mismatch: {name}")
            payload_digests.append((name, _sha256_bytes(member_raw), len(member_raw)))
            if name.startswith("moe_sae/"):
                package_digests.append(
                    (name, _sha256_bytes(member_raw), len(member_raw))
                )
    payload_sha256 = _sha256_bytes(
        _canonical_bytes(
            [
                {"name": name, "sha256": digest, "size": size}
                for name, digest, size in payload_digests
            ]
        )
    )
    package_payload_sha256 = _sha256_bytes(
        _canonical_bytes(
            [
                {"name": name, "sha256": digest, "size": size}
                for name, digest, size in package_digests
            ]
        )
    )
    return WheelIdentity(
        path=path,
        name=path.name,
        sha256=_sha256_bytes(raw),
        size=len(raw),
        record_sha256=_sha256_bytes(record_raw),
        payload_sha256=payload_sha256,
        package_payload_sha256=package_payload_sha256,
    )


_INVENTORY_PROGRAM = r"""
from __future__ import annotations
import base64
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import sys

environment = Path(sys.argv[1]).resolve(strict=True)
result = []
seen = set()
for distribution in distributions():
    raw_name = distribution.metadata["Name"]
    if not isinstance(raw_name, str) or not raw_name:
        raise RuntimeError("distribution has no name")
    name = raw_name.lower().replace("_", "-")
    if name in seen:
        raise RuntimeError("duplicate normalized distribution")
    seen.add(name)
    record = distribution.read_text("RECORD")
    files = distribution.files
    if record is None or files is None:
        raise RuntimeError("distribution has no RECORD inventory")
    direct = distribution.read_text("direct_url.json")
    installed = []
    for item in files:
        located = Path(distribution.locate_file(item)).resolve(strict=True)
        relative = located.relative_to(environment).as_posix()
        raw = located.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if item.size is not None and item.size != len(raw):
            raise RuntimeError("RECORD size mismatch")
        if item.hash is not None:
            if item.hash.mode != "sha256":
                raise RuntimeError("non-SHA256 RECORD")
            encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=")
            if encoded.decode("ascii") != item.hash.value:
                raise RuntimeError("RECORD hash mismatch")
        installed.append({
            "logical_path": str(item),
            "installed_path": relative,
            "size": len(raw),
            "sha256": digest,
            "record_sha256": None if item.hash is None else item.hash.value,
        })
    result.append({
        "name": name,
        "version": distribution.version,
        "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
        "direct_url_sha256": None if direct is None else hashlib.sha256(direct.encode("utf-8")).hexdigest(),
        "files": sorted(installed, key=lambda item: item["logical_path"]),
    })
result.sort(key=lambda item: (item["name"], item["version"]))
print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
"""


def installed_inventory(
    runtime_python: Path,
    environment: Path,
    *,
    runner: CommandRunner = _default_runner,
) -> list[dict[str, object]]:
    result = runner(
        (
            str(runtime_python),
            "-I",
            "-B",
            "-c",
            _INVENTORY_PROGRAM,
            str(environment),
        ),
        {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
    )
    try:
        value = json.loads(result.stdout, object_pairs_hook=_strict_pairs)
    except json.JSONDecodeError as error:
        raise RepackError(
            "installed inventory probe returned malformed JSON"
        ) from error
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise RepackError("installed inventory probe returned the wrong shape")
    return value


def _inventory_digest(inventory: Sequence[Mapping[str, object]]) -> str:
    return _sha256_bytes(_canonical_bytes(list(inventory)))


def _protected_inventory(
    inventory: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    return [item for item in inventory if item.get("name") != WORKER_DISTRIBUTION]


def _distribution(
    inventory: Sequence[Mapping[str, object]], name: str
) -> Mapping[str, object]:
    matches = [item for item in inventory if item.get("name") == name]
    if len(matches) != 1:
        raise RepackError(f"installed distribution is absent or duplicated: {name}")
    return matches[0]


def require_critical_versions(
    inventory: Sequence[Mapping[str, object]],
) -> None:
    for name, version in _CRITICAL_PROTECTED.items():
        if _distribution(inventory, name).get("version") != version:
            raise RepackError(f"critical protected version changed: {name}")
    if _distribution(inventory, WORKER_DISTRIBUTION).get("version") != WORKER_VERSION:
        raise RepackError("worker distribution version changed")


def _require_release_inventory_unchanged(
    post_install: Sequence[Mapping[str, object]],
    release_boundary: Sequence[Mapping[str, object]],
) -> None:
    if list(release_boundary) != list(post_install):
        raise RepackError(
            "import probes mutated the installed environment after verification"
        )


def install_worker_wheel(
    wheel: Path,
    *,
    runtime_python: Path,
    uv: Path,
    work_root: Path,
    runner: CommandRunner = _default_runner,
) -> None:
    """Offline reinstall of exactly one local wheel, explicitly without deps."""

    runner(
        (
            str(uv),
            "pip",
            "install",
            "--offline",
            "--no-config",
            "--no-python-downloads",
            "--no-deps",
            "--reinstall",
            "--python",
            str(runtime_python),
            str(wheel),
        ),
        _build_environment(work_root),
    )


def _installed_worker_package_payload_sha256(
    worker: Mapping[str, object],
) -> str:
    raw_files = worker.get("files")
    if not isinstance(raw_files, list):
        raise RepackError("installed worker inventory has no file list")
    payload: list[dict[str, object]] = []
    for item in raw_files:
        if not isinstance(item, dict):
            raise RepackError("installed worker file inventory is malformed")
        logical_path = item.get("logical_path")
        digest = item.get("sha256")
        size = item.get("size")
        if isinstance(logical_path, str) and logical_path.startswith("moe_sae/"):
            if (
                not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
                or type(size) is not int
            ):
                raise RepackError("installed worker payload identity is malformed")
            payload.append({"name": logical_path, "sha256": digest, "size": size})
    return _sha256_bytes(
        _canonical_bytes(sorted(payload, key=lambda item: str(item["name"])))
    )


def _remove_worker_bytecode(environment: Path) -> None:
    candidates = list(environment.glob("lib/python*/site-packages/moe_sae"))
    if len(candidates) != 1:
        raise RepackError("installed worker package root is absent or ambiguous")
    package_root = candidates[0].resolve(strict=True)
    try:
        package_root.relative_to(environment.resolve(strict=True))
    except ValueError as error:
        raise RepackError("installed worker package escapes the runtime") from error
    for candidate in sorted(
        package_root.rglob("__pycache__"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        if candidate.is_symlink() or not candidate.is_dir():
            raise RepackError("worker bytecode cache path is unsafe")
        shutil.rmtree(candidate)


def _require_worker_filesystem_exact(
    environment: Path,
    worker: Mapping[str, object],
) -> None:
    candidates = list(environment.glob("lib/python*/site-packages/moe_sae"))
    if len(candidates) != 1:
        raise RepackError("installed worker package root is absent or ambiguous")
    environment_root = environment.resolve(strict=True)
    package_root = candidates[0].resolve(strict=True)
    try:
        package_root.relative_to(environment_root)
    except ValueError as error:
        raise RepackError("installed worker package escapes the runtime") from error
    raw_files = worker.get("files")
    if not isinstance(raw_files, list):
        raise RepackError("installed worker inventory has no file list")
    expected: set[str] = set()
    for item in raw_files:
        if (
            isinstance(item, dict)
            and isinstance(item.get("logical_path"), str)
            and str(item["logical_path"]).startswith("moe_sae/")
        ):
            installed_path = item.get("installed_path")
            if not isinstance(installed_path, str):
                raise RepackError("installed worker path is malformed")
            expected.add(installed_path)
    actual: set[str] = set()
    for root, directories, files in os.walk(package_root, followlinks=False):
        directories.sort()
        files.sort()
        root_path = Path(root)
        for directory in directories:
            if (root_path / directory).is_symlink():
                raise RepackError("installed worker contains a directory symlink")
        for file_name in files:
            candidate = root_path / file_name
            if candidate.is_symlink() or not candidate.is_file():
                raise RepackError("installed worker contains a non-regular file")
            actual.add(
                candidate.resolve(strict=True).relative_to(environment_root).as_posix()
            )
    if actual != expected:
        raise RepackError("installed worker tree differs from its RECORD inventory")


_IMPORT_PROGRAM = r"""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import sys

environment = Path(sys.argv[1]).resolve(strict=True)
source = Path(sys.argv[2]).resolve(strict=True)
result = []
for name in sys.argv[3:]:
    module = importlib.import_module(name)
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        raise RuntimeError(f"module has no origin: {name}")
    installed = Path(raw_path).resolve(strict=True)
    installed_relative = installed.relative_to(environment).as_posix()
    source_path = source / ("src/" + name.replace(".", "/") + ".py")
    if not source_path.is_file():
        if name.endswith(".__init__"):
            source_path = source / ("src/" + name.removesuffix(".__init__").replace(".", "/") + "/__init__.py")
        else:
            raise RuntimeError(f"source module is absent: {name}")
    installed_raw = installed.read_bytes()
    source_raw = source_path.read_bytes()
    if installed_raw != source_raw:
        raise RuntimeError(f"installed/source module mismatch: {name}")
    result.append({
        "module": name,
        "installed_path": installed_relative,
        "sha256": hashlib.sha256(installed_raw).hexdigest(),
    })
print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
"""


def verify_import_identity(
    runtime_python: Path,
    environment: Path,
    source: Path,
    *,
    runner: CommandRunner = _default_runner,
    modules: Sequence[str] = _IMPORT_PROBES,
) -> list[Mapping[str, object]]:
    result = runner(
        (
            str(runtime_python),
            "-I",
            "-B",
            "-c",
            _IMPORT_PROGRAM,
            str(environment),
            str(source),
            *modules,
        ),
        {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
    )
    try:
        value = json.loads(result.stdout, object_pairs_hook=_strict_pairs)
    except json.JSONDecodeError as error:
        raise RepackError("import probe returned malformed JSON") from error
    if not isinstance(value, list) or len(value) != len(modules):
        raise RepackError("import probe returned an incomplete module set")
    return value


def _rename_exchange(left: Path, right: Path) -> None:
    """Atomically exchange two same-filesystem directory names on Linux."""

    if (
        left.is_symlink()
        or right.is_symlink()
        or not left.is_dir()
        or not right.is_dir()
    ):
        raise RepackError("source exchange operands must be real directories")
    if left.stat().st_dev != right.stat().st_dev:
        raise RepackError("source exchange operands are on different filesystems")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RepackError("kernel libc has no renameat2; refusing non-atomic fallback")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(left),
        AT_FDCWD,
        os.fsencode(right),
        RENAME_EXCHANGE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise RepackError(f"atomic source exchange failed: {os.strerror(error_number)}")
    _fsync_directory(left.parent)


def _safe_symlink_target(relative_name: str, target: str) -> None:
    if not target or "\x00" in target:
        raise RepackError(f"runtime symlink is malformed: {relative_name}")
    if target.startswith("/"):
        resolved = posixpath.normpath(target)
        if not (
            resolved == "/opt/moe-sae-r1" or resolved.startswith("/opt/moe-sae-r1/")
        ):
            raise RepackError(f"runtime symlink escapes /opt: {relative_name}")
    else:
        resolved = posixpath.normpath(
            posixpath.join(posixpath.dirname(relative_name), target)
        )
        if not (resolved == "opt/moe-sae-r1" or resolved.startswith("opt/moe-sae-r1/")):
            raise RepackError(f"runtime symlink escapes /opt: {relative_name}")


def _runtime_entries(prefix: Path) -> Iterator[tuple[str, Path, os.stat_result]]:
    opt = prefix.parent
    if opt.as_posix() != "/opt":
        # Synthetic tests use another absolute root but retain a virtual "opt".
        virtual_root = "opt"
    else:
        virtual_root = "opt"
    yield virtual_root, opt, opt.lstat()
    prefix_name = f"{virtual_root}/{prefix.name}"
    yield prefix_name, prefix, prefix.lstat()

    def walk(
        directory: Path, relative: str
    ) -> Iterator[tuple[str, Path, os.stat_result]]:
        with os.scandir(directory) as scan:
            children = sorted(scan, key=lambda item: os.fsencode(item.name))
        for child in children:
            child_path = Path(child.path)
            child_relative = f"{relative}/{child.name}"
            child_stat = child_path.lstat()
            yield child_relative, child_path, child_stat
            if stat.S_ISDIR(child_stat.st_mode):
                yield from walk(child_path, child_relative)

    yield from walk(prefix, prefix_name)


class _HashingWriter:
    def __init__(self, stream: io.BufferedWriter) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes) -> int:
        written = self.stream.write(data)
        if written != len(data):
            raise RepackError("short archive write")
        self.digest.update(data)
        self.size += written
        return written

    def flush(self) -> None:
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def write_deterministic_runtime_archive(
    prefix: Path,
    destination: Path,
) -> ArchiveIdentity:
    """Write canonical GNU-tar+gzip bytes for one safe runtime tree."""

    if not prefix.is_dir() or prefix.is_symlink():
        raise RepackError("runtime prefix is not a real directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    member_count = 0
    regular_bytes = 0
    max_member_name_bytes = 0
    max_symlink_target_bytes = 0
    try:
        with temporary.open("wb") as raw_stream:
            hashing = _HashingWriter(raw_stream)
            with (
                gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=6,
                    fileobj=hashing,
                    mtime=0,
                ) as compressed,
                tarfile.open(
                    fileobj=compressed,
                    mode="w|",
                    format=tarfile.GNU_FORMAT,
                ) as archive,
            ):
                for name, path, observed in _runtime_entries(prefix):
                    member_count += 1
                    max_member_name_bytes = max(
                        max_member_name_bytes,
                        len(name.encode("utf-8")),
                    )
                    if member_count > MAX_RUNTIME_MEMBERS:
                        raise RepackError("runtime archive has too many members")
                    info = tarfile.TarInfo(name)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    mode = stat.S_IMODE(observed.st_mode)
                    if mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
                        raise RepackError(f"runtime has privileged mode: {name}")
                    if stat.S_ISDIR(observed.st_mode):
                        if path.is_symlink():
                            raise RepackError(
                                f"runtime directory became a link: {name}"
                            )
                        info.type = tarfile.DIRTYPE
                        info.mode = mode
                        info.size = 0
                        archive.addfile(info)
                    elif stat.S_ISLNK(observed.st_mode):
                        target = os.readlink(path)
                        _safe_symlink_target(name, target)
                        max_symlink_target_bytes = max(
                            max_symlink_target_bytes,
                            len(target.encode("utf-8")),
                        )
                        info.type = tarfile.SYMTYPE
                        info.mode = 0o777
                        info.size = 0
                        info.linkname = target
                        archive.addfile(info)
                    elif stat.S_ISREG(observed.st_mode):
                        if observed.st_nlink != 1:
                            raise RepackError(
                                f"runtime regular file is hard-linked: {name}"
                            )
                        if observed.st_size < 0 or observed.st_size > MAX_RUNTIME_BYTES:
                            raise RepackError(
                                f"runtime file size is out of bounds: {name}"
                            )
                        regular_bytes += observed.st_size
                        if regular_bytes > MAX_RUNTIME_BYTES:
                            raise RepackError("runtime expanded bytes exceed bound")
                        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                        with os.fdopen(descriptor, "rb") as stream:
                            opened = os.fstat(stream.fileno())
                            if (
                                opened.st_dev != observed.st_dev
                                or opened.st_ino != observed.st_ino
                                or opened.st_size != observed.st_size
                                or stat.S_IMODE(opened.st_mode) != mode
                            ):
                                raise RepackError(
                                    f"runtime file raced before read: {name}"
                                )
                            info.type = tarfile.REGTYPE
                            info.mode = mode
                            info.size = observed.st_size
                            archive.addfile(info, stream)
                            after = os.fstat(stream.fileno())
                            if (
                                after.st_size != opened.st_size
                                or after.st_mtime_ns != opened.st_mtime_ns
                                or after.st_ctime_ns != opened.st_ctime_ns
                            ):
                                raise RepackError(
                                    f"runtime file changed during read: {name}"
                                )
                    else:
                        raise RepackError(f"runtime member type is forbidden: {name}")
            hashing.flush()
            os.fsync(raw_stream.fileno())
            archive_sha256 = hashing.digest.hexdigest()
            archive_size = hashing.size
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise RepackError(
                f"refusing to replace existing archive: {destination}"
            ) from error
        temporary.unlink()
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return ArchiveIdentity(
        sha256=archive_sha256,
        size=archive_size,
        member_count=member_count,
        regular_bytes=regular_bytes,
        max_member_name_bytes=max_member_name_bytes,
        max_symlink_target_bytes=max_symlink_target_bytes,
    )


def _validate_request(request: RepackRequest) -> None:
    _require_sha256(
        request.expected_repacker_sha256,
        name="expected repacker SHA-256",
    )
    script_path = Path(__file__).absolute()
    script_stat = script_path.lstat()
    if not stat.S_ISREG(script_stat.st_mode) or script_path.is_symlink():
        raise RepackError("executed repacker is not a regular non-symlink file")
    if _sha256_file(script_path) != request.expected_repacker_sha256:
        raise RepackError("executed repacker SHA-256 mismatch")
    _require_sha256(request.context_archive_sha256, name="context archive SHA-256")
    _require_sha256(
        request.expected_image_recipe_sha256,
        name="image recipe SHA-256",
    )
    _require_sha256(
        request.expected_image_observation_sha256,
        name="image observation SHA-256",
    )
    _require_sha256(
        request.controller_admitted_base_runtime_archive_sha256,
        name="controller-admitted base runtime archive SHA-256",
    )
    _require_revision(
        request.expected_base_context_commit,
        name="base context commit",
    )
    if _IMAGE_RESOURCE.fullmatch(request.expected_boot_image) is None:
        raise RepackError("expected boot image resource is malformed")
    if (
        type(request.controller_admitted_boot_source_image_id) is not int
        or request.controller_admitted_boot_source_image_id <= 0
    ):
        raise RepackError(
            "controller-admitted boot source image ID must be a positive integer"
        )
    for name, path in (
        ("prefix", request.prefix),
        ("identity root", request.identity_root),
        ("uv", request.uv),
        ("context archive", request.context_archive),
        ("output", request.output),
        ("result", request.result),
        ("work root", request.work_root),
    ):
        if not path.is_absolute() or path != Path(os.path.abspath(path)):
            raise RepackError(f"{name} must be a normalized absolute path")
    if len({request.output, request.result, request.work_root}) != 3:
        raise RepackError("output, result, and work root must be pairwise distinct")
    if (
        request.work_root in request.output.parents
        or request.work_root in request.result.parents
    ):
        raise RepackError("output/result cannot be inside the work root")
    _require_safe_output_parent(request.output, name="output")
    _require_safe_output_parent(request.result, name="result")
    _require_safe_output_parent(request.work_root, name="work root")
    for output in (request.output, request.result, request.work_root):
        if output == request.prefix or request.prefix in output.parents:
            raise RepackError("output/work paths must be outside the runtime prefix")


@_cleanup_owned_work_root
def repack(
    request: RepackRequest,
    *,
    runner: CommandRunner = _default_runner,
    metadata_reader: Callable[[str], str] = _read_metadata_image,
) -> Mapping[str, object]:
    """Execute one fail-closed base-image-to-derived-runtime transformation."""

    _validate_request(request)
    if metadata_reader(METADATA_IMAGE_URL) != request.expected_boot_image:
        raise RepackError("guest boot-image metadata differs from admitted image")
    if request.output.exists() or request.result.exists() or request.work_root.exists():
        raise RepackError("output, result, and work paths must all be fresh")
    prefix = request.prefix
    source = prefix / "source"
    environment = prefix / "runtime"
    runtime_python = environment / "bin/python"
    try:
        resolved_runtime_python = runtime_python.resolve(strict=True)
        resolved_runtime_python.relative_to(prefix.resolve(strict=True))
    except (FileNotFoundError, ValueError) as error:
        raise RepackError("runtime Python escapes the sealed prefix") from error
    if (
        not prefix.is_dir()
        or prefix.is_symlink()
        or not resolved_runtime_python.is_file()
        or not request.uv.is_file()
        or request.uv.is_symlink()
    ):
        raise RepackError("base runtime executables are absent or unsafe")
    derived_attestation_path = prefix / "derived-runtime-attestation.json"
    retained_root = prefix / "derived"
    require_pristine_derived_state(prefix)

    recipe_path = request.identity_root / "image-recipe.json"
    observation_path = request.identity_root / "image-observation.json"
    recipe_raw = recipe_path.read_bytes()
    observation_raw = observation_path.read_bytes()
    if _sha256_bytes(recipe_raw) != request.expected_image_recipe_sha256:
        raise RepackError("base image recipe hash mismatch")
    if _sha256_bytes(observation_raw) != request.expected_image_observation_sha256:
        raise RepackError("base image observation hash mismatch")
    # ImageRecipeV1 uses the protocol's canonical wire form, which deliberately
    # has no record-separating newline.  The diagnostic observation is emitted
    # as a canonical JSON line and therefore retains the default newline.
    recipe = _canonical_json(
        recipe_raw,
        name="base image recipe",
        trailing_newline=False,
    )
    observation = _canonical_json(observation_raw, name="base image observation")
    if (
        recipe.get("schema") != "moe-sae.r1.image-recipe.v1"
        or observation.get("schema") != "moe-sae.r1.image-observation-diagnostic.v1"
        or observation.get("production_qualified") is not False
    ):
        raise RepackError("base image recipe/observation schema drifted")
    if observation.get("image_recipe_sha256") != request.expected_image_recipe_sha256:
        raise RepackError("base observation does not bind the image recipe")

    base_context = validate_context_directory(source)
    if (
        base_context.commit != request.expected_base_context_commit
        or recipe.get("repository_commit") != base_context.commit
        or observation.get("context_manifest_sha256") != base_context.manifest_sha256
        or observation.get("pyproject_sha256")
        != _sha256_file(source / "pyproject.toml")
        or observation.get("uv_lock_sha256") != _sha256_file(source / "uv.lock")
    ):
        raise RepackError("base context identity disagrees with baked authorities")
    if observation.get("uv_executable_sha256") != _sha256_file(request.uv):
        raise RepackError("installed uv differs from the base observation")
    if observation.get("python_executable_sha256") != _sha256_file(
        resolved_runtime_python
    ):
        raise RepackError("runtime Python differs from the base observation")

    before = installed_inventory(runtime_python, environment, runner=runner)
    require_critical_versions(before)
    if observation.get("packages") != before:
        raise RepackError("installed packages differ from the baked observation")
    before_digest = _inventory_digest(before)
    protected_before = _protected_inventory(before)
    protected_digest = _inventory_digest(protected_before)

    request.work_root.mkdir(mode=0o700, parents=True)
    _atomic_write(
        request.work_root / ".repacker-owned",
        _WORK_ROOT_MARKER,
        mode=0o400,
    )
    (request.work_root / "home").mkdir(mode=0o700)
    target_source = request.work_root / "source-next"
    target_context = validate_and_extract_context(
        request.context_archive,
        expected_sha256=request.context_archive_sha256,
        destination=target_source,
    )
    inert_context_deltas = require_worker_only_context_delta(source, target_source)

    wheel = build_worker_wheel(
        target_source,
        runtime_python=runtime_python,
        uv=request.uv,
        work_root=request.work_root,
        runner=runner,
    )
    if wheel.package_payload_sha256 != _context_worker_package_payload_sha256(
        target_context
    ):
        raise RepackError(
            "worker wheel does not exactly cover the target worker source"
        )
    retained_wheel_dir = retained_root / "wheels"
    retained_wheel_dir.mkdir(mode=0o755, parents=True)
    retained_wheel = retained_wheel_dir / f"{wheel.sha256}-{wheel.name}"
    with (
        wheel.path.open("rb") as source_wheel,
        retained_wheel.open("xb") as output_wheel,
    ):
        shutil.copyfileobj(source_wheel, output_wheel)
        output_wheel.flush()
        os.fsync(output_wheel.fileno())
    os.chmod(retained_wheel, 0o444)
    if _sha256_file(retained_wheel) != wheel.sha256:
        raise RepackError("retained worker wheel changed during copy")
    incomplete_path = retained_root / "OVERLAY-INCOMPLETE.json"
    incomplete_raw = _canonical_bytes(
        {
            "schema": "moe-sae.r1.derived-runtime-incomplete.v1",
            "state": "worker-install-started-no-derived-attestation",
            "base_repository_commit": base_context.commit,
            "target_repository_commit": target_context.commit,
            "worker_wheel_sha256": wheel.sha256,
        }
    )
    _atomic_write(incomplete_path, incomplete_raw, mode=0o444)

    install_worker_wheel(
        retained_wheel,
        runtime_python=runtime_python,
        uv=request.uv,
        work_root=request.work_root,
        runner=runner,
    )
    _remove_worker_bytecode(environment)
    after = installed_inventory(runtime_python, environment, runner=runner)
    require_critical_versions(after)
    if _protected_inventory(after) != protected_before:
        raise RepackError("worker install mutated a protected distribution")
    worker_after = _distribution(after, WORKER_DISTRIBUTION)
    _require_worker_filesystem_exact(environment, worker_after)
    if (
        _installed_worker_package_payload_sha256(worker_after)
        != wheel.package_payload_sha256
    ):
        raise RepackError("installed worker payload differs from the retained wheel")

    import_probe = verify_import_identity(
        runtime_python,
        environment,
        target_source,
        runner=runner,
    )
    release_inventory = installed_inventory(
        runtime_python,
        environment,
        runner=runner,
    )
    _require_release_inventory_unchanged(after, release_inventory)
    require_critical_versions(release_inventory)
    if _protected_inventory(release_inventory) != protected_before:
        raise RepackError("release-boundary protected inventory changed")
    worker_release = _distribution(release_inventory, WORKER_DISTRIBUTION)
    _require_worker_filesystem_exact(environment, worker_release)
    if (
        _installed_worker_package_payload_sha256(worker_release)
        != wheel.package_payload_sha256
    ):
        raise RepackError("release-boundary worker payload changed")
    # Build backends and probes may not add files to the committed context.
    rebuilt_target_context = validate_context_directory(target_source)
    if (
        rebuilt_target_context.commit != target_context.commit
        or rebuilt_target_context.manifest_sha256 != target_context.manifest_sha256
        or rebuilt_target_context.records != target_context.records
    ):
        raise RepackError("worker build mutated the target source context")
    _normalize_context_directory_modes(target_source)

    _rename_exchange(source, target_source)
    try:
        installed_target = validate_context_directory(source)
        if (
            installed_target.commit != target_context.commit
            or installed_target.manifest_sha256 != target_context.manifest_sha256
            or installed_target.records != target_context.records
        ):
            raise RepackError(
                "atomic source replacement did not preserve target identity"
            )
        # target_source now names the old source; remove it only after proving exchange.
        shutil.rmtree(target_source)
        _fsync_directory(target_source.parent)

        attestation = {
            "schema": ATTESTATION_SCHEMA,
            "production_qualified": False,
            "transformer": {
                "schema": REPACKER_SCHEMA,
                "protocol_version": REPACKER_PROTOCOL_VERSION,
                "repacker_sha256": request.expected_repacker_sha256,
            },
            "observed_boot_image": {
                "metadata_resource": request.expected_boot_image,
            },
            "controller_admitted": {
                "boot_source_image_id": (
                    request.controller_admitted_boot_source_image_id
                ),
                "base_runtime_archive_sha256": (
                    request.controller_admitted_base_runtime_archive_sha256
                ),
            },
            "base": {
                "image_recipe_sha256": request.expected_image_recipe_sha256,
                "image_observation_sha256": request.expected_image_observation_sha256,
                "repository_commit": base_context.commit,
                "repository_archive_sha256": recipe.get("repository_archive_sha256"),
                "context_manifest_sha256": base_context.manifest_sha256,
                "installed_inventory_sha256": before_digest,
                "protected_inventory_sha256": protected_digest,
                "python_executable_sha256": observation.get("python_executable_sha256"),
                "uv_executable_sha256": observation.get("uv_executable_sha256"),
            },
            "target": {
                "repository_commit": target_context.commit,
                "context_archive_sha256": target_context.archive_sha256,
                "context_manifest_sha256": target_context.manifest_sha256,
                "source_date_epoch": SOURCE_DATE_EPOCH,
                "worker_wheel_name": wheel.name,
                "worker_wheel_sha256": wheel.sha256,
                "worker_wheel_size": wheel.size,
                "worker_wheel_payload_sha256": wheel.payload_sha256,
                "worker_wheel_record_sha256": wheel.record_sha256,
                "installed_worker_record_sha256": worker_release.get("record_sha256"),
                "installed_worker_inventory_sha256": _inventory_digest(
                    [worker_release]
                ),
                "release_inventory_sha256": _inventory_digest(release_inventory),
                "protected_inventory_sha256": _inventory_digest(
                    _protected_inventory(release_inventory)
                ),
                "import_probe_sha256": _sha256_bytes(_canonical_bytes(import_probe)),
            },
            "invariants": {
                "context_delta": "src/moe_sae/** plus exact inert image validators",
                "inert_context_deltas": list(inert_context_deltas),
                "install": "uv pip install --offline --no-deps --reinstall",
                "archive": "GNU-tar+gzip; uid/gid/mtime=0; gzip-mtime=0",
                "python_version": observation.get("python_version"),
                "uv_version": recipe.get("uv_version"),
                "critical_protected_versions": dict(
                    sorted(_CRITICAL_PROTECTED.items())
                ),
            },
        }
        attestation_raw = _canonical_bytes(attestation)
        _atomic_write(derived_attestation_path, attestation_raw, mode=0o444)
        if (
            not incomplete_path.is_file()
            or incomplete_path.is_symlink()
            or incomplete_path.read_bytes() != incomplete_raw
        ):
            raise RepackError("incomplete overlay marker changed before release")
        incomplete_path.unlink()
        _fsync_directory(incomplete_path.parent)
        archive_identity = write_deterministic_runtime_archive(prefix, request.output)
        result = {
            "schema": RESULT_SCHEMA,
            "production_qualified": False,
            "repacker_sha256": request.expected_repacker_sha256,
            "repository_commit": target_context.commit,
            "derived_attestation_sha256": _sha256_bytes(attestation_raw),
            "archive": {
                "media_type": "application/gzip",
                "format": "GNU-tar+gzip",
                "sha256": archive_identity.sha256,
                "size_bytes": archive_identity.size,
                "member_count": archive_identity.member_count,
                "expanded_regular_bytes": archive_identity.regular_bytes,
                "max_member_name_bytes": archive_identity.max_member_name_bytes,
                "max_symlink_target_bytes": (archive_identity.max_symlink_target_bytes),
                "publication": "external-generation-pinned-regional-object",
            },
        }
        _atomic_write(request.result, _canonical_bytes(result), mode=0o444)
        return result
    finally:
        _remove_owned_work_root(request.work_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--context-archive", type=Path, required=True)
    parser.add_argument("--context-archive-sha256", required=True)
    parser.add_argument("--expected-repacker-sha256", required=True)
    parser.add_argument("--expected-boot-image", required=True)
    parser.add_argument(
        "--controller-admitted-boot-source-image-id",
        type=int,
        required=True,
    )
    parser.add_argument("--expected-image-recipe-sha256", required=True)
    parser.add_argument("--expected-image-observation-sha256", required=True)
    parser.add_argument(
        "--controller-admitted-base-runtime-archive-sha256",
        required=True,
    )
    parser.add_argument("--expected-base-context-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, default=Path("/opt/moe-sae-r1"))
    parser.add_argument(
        "--identity-root",
        type=Path,
        default=Path("/usr/local/lib/moe-sae-r1"),
    )
    parser.add_argument("--uv", type=Path, default=Path("/usr/local/bin/uv"))
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("/var/tmp/moe-sae-r1-runtime-repack"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.umask(0o077)
    if os.geteuid() != 0:
        raise RepackError("runtime repacker must run as root")
    if sys.platform != "linux":
        raise RepackError("runtime repacker requires Linux renameat2")
    lock_path = Path("/run/lock/moe-sae-r1-runtime-repack.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RepackError("another runtime repacker owns this VM") from error
        request = RepackRequest(
            context_archive=args.context_archive,
            context_archive_sha256=args.context_archive_sha256,
            expected_repacker_sha256=args.expected_repacker_sha256,
            expected_boot_image=args.expected_boot_image,
            controller_admitted_boot_source_image_id=(
                args.controller_admitted_boot_source_image_id
            ),
            expected_image_recipe_sha256=args.expected_image_recipe_sha256,
            expected_image_observation_sha256=args.expected_image_observation_sha256,
            controller_admitted_base_runtime_archive_sha256=(
                args.controller_admitted_base_runtime_archive_sha256
            ),
            expected_base_context_commit=args.expected_base_context_commit,
            prefix=args.prefix,
            identity_root=args.identity_root,
            uv=args.uv,
            output=args.output,
            result=args.result,
            work_root=args.work_root,
        )
        result = repack(request)
        sys.stdout.buffer.write(_canonical_bytes(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RepackError as error:
        print(f"[r1-runtime-repack] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
