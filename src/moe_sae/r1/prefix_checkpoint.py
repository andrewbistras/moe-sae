"""Diagnostic, immutable byte storage for converted R1 prefix chunks.

This module is a correctness-first local data plane between the bounded
conversion executor and a future distributed restore implementation.  It is
deliberately *not* Orbax, TensorStore, a production checkpoint, or a qualified
cloud artifact.  It provides:

* create-only, exact-path local objects with durable canonical receipts;
* an output sink implementing :class:`ConversionOutputSink`;
* a canonical, plan-bound diagnostic storage manifest; and
* bounded flat-region reads without materialising a whole checkpoint.

The storage manifest is independent of which source shards happened to be
opened in the final conversion invocation.  Actual source observations grow
monotonically at exact plan-derived paths and, only when complete, produce a
separate content-addressed :class:`ObservationSnapshotV1`.  Storage restore
therefore converges regardless of finalization order; callers opt into the
stronger complete-observation gate explicitly.

Object discovery is structural: callers derive every path from a conversion
plan.  Directory listing is never an authority.  A chunk binding records the
authenticated source identities embedded in its conversion identity.  Those
bindings are explicitly plan-derived lineage, not fabricated claims that a
source blob was observed in the final invocation; this distinction keeps a
crash after a shard's final chunk restoreable without overclaiming evidence.

Production sealing remains disabled.  Missing tranches are a generation- and
CRC32C-pinned GCS byte-store adapter, a JAX addressable-shard tree mapper and
restore path, real-payload throughput qualification, interoperability with an
actual checkpoint framework, and full numerical/logit/prefix parity evidence.
In particular, this local reader full-hashes roughly 650 GB at open and again
for each touched chunk read; that is diagnostic integrity behavior, not a
claim of production restore scalability.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import hashlib
import os
from pathlib import PurePosixPath
import re
import stat
import threading
from typing import Any, ClassVar, Protocol, Self

import ml_dtypes
import numpy as np

from .checkpoint_conversion import (
    CHUNKING_CONTRACT,
    CONVERSION_ENGINE_CONTRACT,
    DEQUANTIZATION_CONTRACT,
    SINK_VERIFICATION_CONTRACT,
    SOURCE_VERIFICATION_CONTRACT,
    CheckpointConversionError,
    ConversionAttemptSummaryV1,
    ConversionChunkIdentityV1,
    ConversionOutputSink,
    OutputChunkReceiptV1,
    R1RuntimeLockIdentityV1,
    SourceAdapterIdentityV1,
    StreamingConversionPlanV1,
    VerifiedSourceShardReceiptV1,
)
from .protocol import (
    CanonicalRecord,
    IDENTIFIER_PATTERN,
    ProtocolError,
    SHA256_PATTERN,
    canonical_sha256,
    require_integer,
    require_record_object,
    require_string,
)


LOCAL_BYTE_STORE_CONTRACT = "local-create-only-fsync-immutable-bytes-v1"
LOCAL_STORE_IDENTITY_SCOPE = "local-path-diagnostic-only-v1"
PREFIX_SINK_CONTRACT = "diagnostic-prefix-chunk-sink-v1"
SOURCE_LINEAGE_CONTRACT = (
    "plan-authenticated-source-identity-not-final-invocation-observation-v1"
)
OBJECT_DISCOVERY_CONTRACT = "exact-plan-derived-paths-no-listing-v1"

_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_LOCAL_VERSION = re.compile(
    r"local-v1-[0-9a-f]+-[0-9a-f]+-[0-9a-f]+-[0-9a-f]+-[0-9a-f]+"
)
_MAX_OBJECT_ID_BYTES = 1024
_MAX_RECEIPT_BYTES = 1024**2
_MAX_MANIFEST_BYTES = 64 * 1024**2
_DEFAULT_IO_CHUNK_BYTES = 8 * 1024**2
_MAX_IO_CHUNK_BYTES = 64 * 1024**2
_DEFAULT_MAX_READ_BYTES = 64 * 1024**2
_DTYPES = {
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
    "float32": np.dtype("<f4"),
}
UNMET_PRODUCTION_GAPS = (
    "gcs-generation-and-crc32c-byte-store-adapter",
    "jax-addressable-shard-tree-mapper-and-restore",
    "real-payload-throughput-qualification",
    "checkpoint-framework-interoperability",
    "independent-numerical-logit-and-prefix-parity",
)


class PrefixCheckpointError(CheckpointConversionError):
    """The local immutable checkpoint data plane failed closed."""


class ImmutableObjectConflict(PrefixCheckpointError):
    """An immutable exact path exists with different bytes or identity."""


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ProtocolError(f"{name} must be a JSON boolean")
    return value


def _optional_record(value: Any, name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be an object or null")
    return value


def _record_list(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise ProtocolError(f"{name} must be a JSON array")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ProtocolError(f"{name}[{index}] must be an object")
        result.append(item)
    return tuple(result)


def _string_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProtocolError(f"{name} must be a JSON array")
    return tuple(
        require_string(item, f"{name}[{index}]") for index, item in enumerate(value)
    )


def _shape(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ProtocolError(f"{name} must be a non-empty JSON array")
    return tuple(
        require_integer(item, f"{name}[{index}]", minimum=1)
        for index, item in enumerate(value)
    )


def _validate_object_id(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise PrefixCheckpointError("object_id must be a non-empty string")
    if len(value.encode("utf-8")) > _MAX_OBJECT_ID_BYTES:
        raise PrefixCheckpointError("object_id exceeds its byte bound")
    if "\\" in value or value.startswith("/") or value.endswith("/"):
        raise PrefixCheckpointError(
            "object_id must be a normalized relative POSIX path"
        )
    path = PurePosixPath(value)
    parts = path.parts
    if (
        not parts
        or str(path) != value
        or any(
            part in {"", ".", ".."} or _COMPONENT.fullmatch(part) is None
            for part in parts
        )
    ):
        raise PrefixCheckpointError("object_id contains an unsafe path component")
    return value


def _chunk_object_id(plan_id: str, chunk_id: str) -> str:
    return _validate_object_id(f"plans/{plan_id}/chunks/{chunk_id}.bin")


def _binding_object_id(plan_id: str, chunk_id: str) -> str:
    return _validate_object_id(f"plans/{plan_id}/receipts/{chunk_id}.json")


def _manifest_object_id(plan_id: str) -> str:
    return _validate_object_id(f"plans/{plan_id}/manifest.json")


def _source_observation_object_id(plan_id: str, shard: str) -> str:
    shard_id = hashlib.sha256(shard.encode("utf-8")).hexdigest()
    return _validate_object_id(f"plans/{plan_id}/source-observations/{shard_id}.json")


@dataclass(frozen=True, slots=True)
class ImmutablePhysicalObjectReceiptV1(CanonicalRecord):
    """Exact local physical identity for one immutable regular file."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.local-physical-object-receipt.v1"

    store_id: str
    object_id: str
    version_token: str
    size_bytes: int
    sha256: str
    storage_contract: str = LOCAL_BYTE_STORE_CONTRACT

    def __post_init__(self) -> None:
        require_string(self.store_id, "store_id", pattern=IDENTIFIER_PATTERN)
        _validate_object_id(self.object_id)
        require_string(self.version_token, "version_token", pattern=_LOCAL_VERSION)
        require_integer(self.size_bytes, "size_bytes", minimum=1)
        require_string(self.sha256, "sha256", pattern=SHA256_PATTERN)
        if self.storage_contract != LOCAL_BYTE_STORE_CONTRACT:
            raise ProtocolError("unsupported local immutable storage contract")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "store_id": self.store_id,
            "object_id": self.object_id,
            "version_token": self.version_token,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "storage_contract": self.storage_contract,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = require_record_object(
            value,
            schema=cls.SCHEMA,
            fields=(
                "store_id",
                "object_id",
                "version_token",
                "size_bytes",
                "sha256",
                "storage_contract",
            ),
        )
        return cls(
            store_id=require_string(
                obj["store_id"], "store_id", pattern=IDENTIFIER_PATTERN
            ),
            object_id=_validate_object_id(
                require_string(obj["object_id"], "object_id")
            ),
            version_token=require_string(
                obj["version_token"], "version_token", pattern=_LOCAL_VERSION
            ),
            size_bytes=require_integer(obj["size_bytes"], "size_bytes", minimum=1),
            sha256=require_string(obj["sha256"], "sha256", pattern=SHA256_PATTERN),
            storage_contract=require_string(
                obj["storage_contract"], "storage_contract"
            ),
        )


class ImmutableByteStore(Protocol):
    """No-listing immutable byte boundary used by the diagnostic sink."""

    @property
    def store_id(self) -> str: ...

    @property
    def stream_buffer_bytes(self) -> int: ...

    @property
    def distributed_identity_qualified(self) -> bool: ...

    def create_or_verify(
        self,
        object_id: str,
        payload: bytes | bytearray | memoryview,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> ImmutablePhysicalObjectReceiptV1: ...

    def probe_bounded(
        self, object_id: str, *, maximum_bytes: int
    ) -> tuple[bytes, ImmutablePhysicalObjectReceiptV1] | None: ...

    def verify(
        self, receipt: ImmutablePhysicalObjectReceiptV1
    ) -> ImmutablePhysicalObjectReceiptV1: ...

    def read_exact(
        self,
        receipt: ImmutablePhysicalObjectReceiptV1,
        *,
        offset: int,
        length: int,
        maximum_bytes: int,
    ) -> bytes: ...


@dataclass(frozen=True, slots=True)
class _FileState:
    device: int
    inode: int
    mode: int
    nlink: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> Self:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            mode=value.st_mode,
            nlink=value.st_nlink,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
        )

    @property
    def version_token(self) -> str:
        return (
            f"local-v1-{self.device:x}-{self.inode:x}-{self.ctime_ns:x}-"
            f"{self.mtime_ns:x}-{stat.S_IMODE(self.mode):x}"
        )


class LocalImmutableByteStore:
    """Create-only local store pinned to one non-symlink directory inode.

    Publication writes and fsyncs an exact content-derived staging inode,
    links it into the exact final name without replacement, unlinks staging,
    and fsyncs the parent directory.  The exact staging identity makes every
    instruction-boundary publication state recoverable without listing.  A
    process-wide file lock on the root
    directory plus an in-process lock makes identical concurrent publication
    convergent while divergent publication fails.  ``store_id`` is only a
    hash of the local absolute path: it is deliberately not a distributed or
    portable storage identity.  Every authoritative read
    reopens with ``O_NOFOLLOW``, requires link count one, hashes the complete
    file in bounded chunks, and compares pre/post inode state.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        io_chunk_bytes: int = _DEFAULT_IO_CHUNK_BYTES,
    ) -> None:
        if (
            type(io_chunk_bytes) is not int
            or not 1 <= io_chunk_bytes <= _MAX_IO_CHUNK_BYTES
        ):
            raise PrefixCheckpointError(
                f"io_chunk_bytes must be an integer in [1, {_MAX_IO_CHUNK_BYTES}]"
            )
        raw_root = os.path.abspath(os.fspath(root))
        try:
            root_lstat = os.lstat(raw_root)
        except OSError as error:
            raise PrefixCheckpointError(
                "immutable store root must already exist"
            ) from error
        if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
            raise PrefixCheckpointError(
                "immutable store root must be a non-symlink directory"
            )
        flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            root_fd = os.open(raw_root, flags)
        except OSError as error:
            raise PrefixCheckpointError("cannot pin immutable store root") from error
        root_fstat = os.fstat(root_fd)
        if (root_fstat.st_dev, root_fstat.st_ino) != (
            root_lstat.st_dev,
            root_lstat.st_ino,
        ):
            os.close(root_fd)
            raise PrefixCheckpointError("immutable store root changed while opening")
        self._root = raw_root
        self._root_fd = root_fd
        self._root_identity = (root_fstat.st_dev, root_fstat.st_ino)
        self._store_id = (
            "local-prefix-byte-store-v1:"
            + hashlib.sha256(raw_root.encode("utf-8")).hexdigest()
        )
        self._io_chunk_bytes = io_chunk_bytes
        self._lock = threading.RLock()
        self._closed = False

    @property
    def store_id(self) -> str:
        return self._store_id

    @property
    def stream_buffer_bytes(self) -> int:
        """Maximum transient stream slice, not a bound on total Python RSS."""

        return self._io_chunk_bytes

    @property
    def distributed_identity_qualified(self) -> bool:
        return False

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                os.close(self._root_fd)
                self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise PrefixCheckpointError("immutable store is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _check_root(self) -> None:
        if self._closed:
            raise PrefixCheckpointError("immutable store is closed")
        try:
            visible = os.lstat(self._root)
            pinned = os.fstat(self._root_fd)
        except OSError as error:
            raise PrefixCheckpointError(
                "immutable store root is no longer stable"
            ) from error
        if (
            stat.S_ISLNK(visible.st_mode)
            or not stat.S_ISDIR(visible.st_mode)
            or (visible.st_dev, visible.st_ino) != self._root_identity
            or (pinned.st_dev, pinned.st_ino) != self._root_identity
        ):
            raise PrefixCheckpointError("immutable store root was substituted")

    @contextmanager
    def _serialized(self):
        with self._lock:
            self._check_root()
            try:
                fcntl.flock(self._root_fd, fcntl.LOCK_EX)
            except OSError as error:
                raise PrefixCheckpointError(
                    "cannot lock immutable store root"
                ) from error
            completed = False
            try:
                self._check_root()
                yield
                completed = True
            finally:
                try:
                    if completed:
                        self._check_root()
                finally:
                    fcntl.flock(self._root_fd, fcntl.LOCK_UN)

    def _open_parent(self, object_id: str, *, create: bool) -> tuple[int, str]:
        parts = _validate_object_id(object_id).split("/")
        current = os.dup(self._root_fd)
        flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            for component in parts[:-1]:
                if create:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=current)
                        os.fsync(current)
                    except FileExistsError:
                        pass
                try:
                    child = os.open(component, flags, dir_fd=current)
                except FileNotFoundError:
                    os.close(current)
                    return -1, parts[-1]
                except OSError as error:
                    raise PrefixCheckpointError(
                        f"unsafe immutable object parent for {object_id}"
                    ) from error
                state = os.fstat(child)
                if not stat.S_ISDIR(state.st_mode):
                    os.close(child)
                    raise PrefixCheckpointError(
                        f"immutable object parent is not a directory: {object_id}"
                    )
                os.close(current)
                current = child
            return current, parts[-1]
        except Exception:
            if current >= 0:
                os.close(current)
            raise

    @staticmethod
    def _regular_state(value: os.stat_result, object_id: str) -> _FileState:
        result = _FileState.from_stat(value)
        if not stat.S_ISREG(result.mode) or result.nlink != 1:
            raise PrefixCheckpointError(
                f"immutable object must be a single-link regular file: {object_id}"
            )
        if stat.S_IMODE(result.mode) != 0o600:
            raise PrefixCheckpointError(
                f"immutable object mode changed for {object_id}"
            )
        if result.size <= 0:
            raise PrefixCheckpointError(f"immutable object is empty: {object_id}")
        return result

    def _open_object(self, object_id: str) -> tuple[int, int] | None:
        parent_fd, leaf = self._open_parent(object_id, create=False)
        if parent_fd < 0:
            return None
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            try:
                fd = os.open(leaf, flags, dir_fd=parent_fd)
            except FileNotFoundError:
                os.close(parent_fd)
                return None
            except OSError as error:
                raise PrefixCheckpointError(
                    f"cannot safely open immutable object {object_id}"
                ) from error
            return parent_fd, fd
        except Exception:
            os.close(parent_fd)
            raise

    def _measure(
        self,
        object_id: str,
        *,
        capture_offset: int | None = None,
        capture_length: int = 0,
        maximum_capture_bytes: int = 0,
        maximum_size_bytes: int | None = None,
    ) -> tuple[ImmutablePhysicalObjectReceiptV1, bytes | None] | None:
        opened = self._open_object(object_id)
        if opened is None:
            return None
        parent_fd, fd = opened
        try:
            before = self._regular_state(os.fstat(fd), object_id)
            if maximum_size_bytes is not None:
                if type(maximum_size_bytes) is not int or maximum_size_bytes <= 0:
                    raise PrefixCheckpointError(
                        "maximum object size must be a positive integer"
                    )
                if before.size > maximum_size_bytes:
                    raise PrefixCheckpointError(
                        "immutable object exceeds bounded probe size before hashing"
                    )
            if capture_offset is not None:
                if type(capture_offset) is not int or capture_offset < 0:
                    raise PrefixCheckpointError(
                        "read offset must be a nonnegative integer"
                    )
                if type(capture_length) is not int or capture_length <= 0:
                    raise PrefixCheckpointError(
                        "read length must be a positive integer"
                    )
                if type(maximum_capture_bytes) is not int or maximum_capture_bytes <= 0:
                    raise PrefixCheckpointError("maximum read bytes must be positive")
                if capture_length > maximum_capture_bytes:
                    raise PrefixCheckpointError("requested read exceeds its bound")
                if capture_offset + capture_length > before.size:
                    raise PrefixCheckpointError(
                        "requested read exceeds immutable object size"
                    )
                captured = bytearray(capture_length)
            else:
                captured = None
            digest = hashlib.sha256()
            cursor = 0
            while cursor < before.size:
                request = min(self._io_chunk_bytes, before.size - cursor)
                raw = os.pread(fd, request, cursor)
                if len(raw) != request:
                    raise PrefixCheckpointError(
                        f"short immutable object read for {object_id}"
                    )
                digest.update(raw)
                if captured is not None and capture_offset is not None:
                    left = max(cursor, capture_offset)
                    right = min(cursor + request, capture_offset + capture_length)
                    if left < right:
                        source_start = left - cursor
                        target_start = left - capture_offset
                        captured[target_start : target_start + (right - left)] = raw[
                            source_start : source_start + (right - left)
                        ]
                cursor += request
            after = self._regular_state(os.fstat(fd), object_id)
            if before != after:
                raise PrefixCheckpointError(
                    f"immutable object mutated while being read: {object_id}"
                )
            # The opened inode is insufficient authority: an attacker can
            # rename it away and substitute the exact path while hashing is in
            # progress.  Re-traverse from the pinned root and compare the full
            # final file state before issuing a receipt.
            reopened = self._open_object(object_id)
            if reopened is None:
                raise PrefixCheckpointError(
                    f"immutable object path vanished while being read: {object_id}"
                )
            reopened_parent_fd, reopened_fd = reopened
            try:
                path_state = self._regular_state(os.fstat(reopened_fd), object_id)
            finally:
                os.close(reopened_fd)
                os.close(reopened_parent_fd)
            if path_state != after:
                raise PrefixCheckpointError(
                    f"immutable object path was substituted while being read: {object_id}"
                )
            receipt = ImmutablePhysicalObjectReceiptV1(
                store_id=self.store_id,
                object_id=object_id,
                version_token=after.version_token,
                size_bytes=after.size,
                sha256=digest.hexdigest(),
            )
            return receipt, None if captured is None else bytes(captured)
        finally:
            os.close(fd)
            os.close(parent_fd)

    @staticmethod
    def _staging_leaf(leaf: str, expected_sha256: str) -> str:
        # This exact, content-specific staging identity is derivable without a
        # listing after process death.  Even the longest admitted leaf remains
        # below common NAME_MAX limits.
        return f".{leaf}.stage-{expected_sha256}"

    @staticmethod
    def _open_optional_leaf(parent_fd: int, leaf: str) -> int | None:
        try:
            return os.open(
                leaf,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            return None
        except OSError as error:
            raise PrefixCheckpointError(
                f"cannot safely open staging path {leaf}"
            ) from error

    def _measure_staging_fd(
        self,
        fd: int,
        *,
        label: str,
        allowed_nlinks: set[int],
    ) -> tuple[_FileState, str]:
        before = _FileState.from_stat(os.fstat(fd))
        if (
            not stat.S_ISREG(before.mode)
            or before.nlink not in allowed_nlinks
            or stat.S_IMODE(before.mode) != 0o600
        ):
            raise PrefixCheckpointError(
                f"staging object has unsafe file identity: {label}"
            )
        digest = hashlib.sha256()
        cursor = 0
        while cursor < before.size:
            request = min(self._io_chunk_bytes, before.size - cursor)
            raw = os.pread(fd, request, cursor)
            if len(raw) != request:
                raise PrefixCheckpointError(f"short staging read: {label}")
            digest.update(raw)
            cursor += request
        after = _FileState.from_stat(os.fstat(fd))
        if before != after:
            raise PrefixCheckpointError(f"staging object mutated while read: {label}")
        return after, digest.hexdigest()

    def _recover_exact_staging(
        self,
        object_id: str,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> None:
        """Resolve only the one content-derived staging name for this write.

        This covers death while writing, after staging fsync, and after the
        final hard-link but before staging unlink.  It never lists or adopts a
        different content identity.  An incomplete same-content staging file
        can be discarded because the root lock proves no cooperating writer
        remains active; a foreign link topology still fails closed.
        """

        parent_fd, leaf = self._open_parent(object_id, create=True)
        if parent_fd < 0:  # pragma: no cover - create=True invariant
            raise PrefixCheckpointError("cannot open immutable staging parent")
        staging = self._staging_leaf(leaf, expected_sha256)
        final_fd = self._open_optional_leaf(parent_fd, leaf)
        staging_fd = self._open_optional_leaf(parent_fd, staging)
        try:
            if staging_fd is None:
                return
            staging_state, staging_sha = self._measure_staging_fd(
                staging_fd,
                label=staging,
                allowed_nlinks={1, 2},
            )
            if final_fd is None:
                if staging_state.nlink != 1:
                    raise PrefixCheckpointError(
                        "orphan staging object has an unexplained hard link"
                    )
                if (
                    staging_state.size != expected_size_bytes
                    or staging_sha != expected_sha256
                ):
                    # The exact staging name identifies the intended digest,
                    # so a mismatching single-link file is a partial/corrupt
                    # write from a dead holder, never a divergent authority.
                    os.unlink(staging, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                    return
                os.link(
                    staging,
                    leaf,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                final_fd = self._open_optional_leaf(parent_fd, leaf)
                if final_fd is None:  # pragma: no cover - link invariant
                    raise PrefixCheckpointError("recovered final object vanished")
                staging_state, staging_sha = self._measure_staging_fd(
                    staging_fd,
                    label=staging,
                    allowed_nlinks={2},
                )
            final_state, final_sha = self._measure_staging_fd(
                final_fd,
                label=leaf,
                allowed_nlinks={2},
            )
            if (
                (final_state.device, final_state.inode)
                != (staging_state.device, staging_state.inode)
                or final_state != staging_state
                or final_state.size != expected_size_bytes
                or final_sha != expected_sha256
                or staging_sha != expected_sha256
            ):
                raise ImmutableObjectConflict(
                    "final and deterministic staging identities conflict"
                )
            os.unlink(staging, dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            if final_fd is not None:
                os.close(final_fd)
            if staging_fd is not None:
                os.close(staging_fd)
            os.close(parent_fd)

    def create_or_verify(
        self,
        object_id: str,
        payload: bytes | bytearray | memoryview,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> ImmutablePhysicalObjectReceiptV1:
        object_id = _validate_object_id(object_id)
        require_string(expected_sha256, "expected_sha256", pattern=SHA256_PATTERN)
        if type(expected_size_bytes) is not int or expected_size_bytes <= 0:
            raise PrefixCheckpointError(
                "expected_size_bytes must be a positive integer"
            )
        try:
            view = memoryview(payload).cast("B")
        except (TypeError, ValueError) as error:
            raise PrefixCheckpointError(
                "immutable payload must expose contiguous bytes"
            ) from error
        try:
            if len(view) != expected_size_bytes:
                raise PrefixCheckpointError(
                    "immutable payload size differs from expectation"
                )
            with self._serialized():
                self._recover_exact_staging(
                    object_id,
                    expected_sha256=expected_sha256,
                    expected_size_bytes=expected_size_bytes,
                )
                existing = self._measure(object_id)
                if existing is not None:
                    receipt = existing[0]
                    if (
                        receipt.size_bytes != expected_size_bytes
                        or receipt.sha256 != expected_sha256
                    ):
                        raise ImmutableObjectConflict(
                            f"immutable object exists with divergent bytes: {object_id}"
                        )
                    return receipt

                parent_fd, leaf = self._open_parent(object_id, create=True)
                if parent_fd < 0:  # pragma: no cover - create=True invariant
                    raise PrefixCheckpointError("cannot create immutable object parent")
                temporary = self._staging_leaf(leaf, expected_sha256)
                flags = (
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                )
                temp_fd = -1
                linked = False
                try:
                    temp_fd = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
                    digest = hashlib.sha256()
                    cursor = 0
                    while cursor < len(view):
                        stop = min(cursor + self._io_chunk_bytes, len(view))
                        piece = view[cursor:stop]
                        wrote = os.write(temp_fd, piece)
                        if wrote <= 0:
                            raise PrefixCheckpointError("short immutable object write")
                        digest.update(piece[:wrote])
                        cursor += wrote
                    os.fsync(temp_fd)
                    state = os.fstat(temp_fd)
                    if (
                        not stat.S_ISREG(state.st_mode)
                        or state.st_size != expected_size_bytes
                    ):
                        raise PrefixCheckpointError(
                            "temporary immutable object is inconsistent"
                        )
                    if digest.hexdigest() != expected_sha256:
                        raise PrefixCheckpointError(
                            "immutable payload SHA-256 differs from expectation"
                        )
                    os.close(temp_fd)
                    temp_fd = -1
                    try:
                        os.link(
                            temporary,
                            leaf,
                            src_dir_fd=parent_fd,
                            dst_dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                        linked = True
                    except FileExistsError:
                        linked = False
                    os.unlink(temporary, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                except Exception:
                    if temp_fd >= 0:
                        os.close(temp_fd)
                    try:
                        os.unlink(temporary, dir_fd=parent_fd)
                        os.fsync(parent_fd)
                    except FileNotFoundError:
                        pass
                    finally:
                        os.close(parent_fd)
                    raise
                os.close(parent_fd)
                measured = self._measure(object_id)
                if measured is None:  # pragma: no cover - link/create invariant
                    raise PrefixCheckpointError("published immutable object vanished")
                receipt = measured[0]
                if (
                    receipt.size_bytes != expected_size_bytes
                    or receipt.sha256 != expected_sha256
                ):
                    action = "published" if linked else "concurrent"
                    raise ImmutableObjectConflict(
                        f"{action} immutable object has divergent bytes: {object_id}"
                    )
                return receipt
        finally:
            view.release()

    def _exact_path_size(self, object_id: str) -> int | None:
        opened = self._open_object(object_id)
        if opened is None:
            return None
        parent_fd, fd = opened
        try:
            return self._regular_state(os.fstat(fd), object_id).size
        finally:
            os.close(fd)
            os.close(parent_fd)

    def probe_bounded(
        self, object_id: str, *, maximum_bytes: int
    ) -> tuple[bytes, ImmutablePhysicalObjectReceiptV1] | None:
        object_id = _validate_object_id(object_id)
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise PrefixCheckpointError("maximum_bytes must be a positive integer")
        with self._serialized():
            observed_size = self._exact_path_size(object_id)
            if observed_size is None:
                return None
            if observed_size > maximum_bytes:
                raise PrefixCheckpointError(
                    "immutable object exceeds bounded probe size"
                )
            captured = self._measure(
                object_id,
                capture_offset=0,
                capture_length=observed_size,
                maximum_capture_bytes=maximum_bytes,
                maximum_size_bytes=maximum_bytes,
            )
            if captured is None:  # pragma: no cover - serialized exact path
                raise PrefixCheckpointError("immutable object vanished during probe")
            receipt, raw = captured
            if raw is None or receipt.size_bytes != observed_size:
                raise PrefixCheckpointError(
                    "immutable object changed during bounded probe"
                )
            return raw, receipt

    def verify(
        self, receipt: ImmutablePhysicalObjectReceiptV1
    ) -> ImmutablePhysicalObjectReceiptV1:
        if type(receipt) is not ImmutablePhysicalObjectReceiptV1:
            raise PrefixCheckpointError("physical receipt has the wrong type")
        if receipt.store_id != self.store_id:
            raise PrefixCheckpointError("physical receipt belongs to another store")
        with self._serialized():
            measured = self._measure(receipt.object_id)
            if measured is None or measured[0] != receipt:
                raise PrefixCheckpointError(
                    f"physical receipt no longer matches {receipt.object_id}"
                )
            return measured[0]

    def read_exact(
        self,
        receipt: ImmutablePhysicalObjectReceiptV1,
        *,
        offset: int,
        length: int,
        maximum_bytes: int,
    ) -> bytes:
        if type(receipt) is not ImmutablePhysicalObjectReceiptV1:
            raise PrefixCheckpointError("physical receipt has the wrong type")
        if receipt.store_id != self.store_id:
            raise PrefixCheckpointError("physical receipt belongs to another store")
        with self._serialized():
            measured = self._measure(
                receipt.object_id,
                capture_offset=offset,
                capture_length=length,
                maximum_capture_bytes=maximum_bytes,
            )
            if measured is None or measured[0] != receipt or measured[1] is None:
                raise PrefixCheckpointError(
                    f"physical receipt no longer matches {receipt.object_id}"
                )
            return measured[1]


@dataclass(frozen=True, slots=True)
class PlanSourceIdentityBindingV1(CanonicalRecord):
    """Plan-authenticated lineage, explicitly not an I/O observation receipt."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.plan-source-identity-binding.v1"

    role: str
    shard: str
    sha256: str
    size_bytes: int
    source_adapter_identity: SourceAdapterIdentityV1
    evidence_contract: str = SOURCE_LINEAGE_CONTRACT

    def __post_init__(self) -> None:
        if self.role not in {"source", "scale"}:
            raise ProtocolError("source lineage role must be source or scale")
        require_string(self.shard, "shard", pattern=IDENTIFIER_PATTERN)
        require_string(self.sha256, "sha256", pattern=SHA256_PATTERN)
        require_integer(self.size_bytes, "size_bytes", minimum=1)
        if type(self.source_adapter_identity) is not SourceAdapterIdentityV1:
            raise ProtocolError("source adapter identity has the wrong type")
        if self.evidence_contract != SOURCE_LINEAGE_CONTRACT:
            raise ProtocolError("unsupported source lineage evidence contract")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "role": self.role,
            "shard": self.shard,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source_adapter_identity": self.source_adapter_identity.as_dict(),
            "evidence_contract": self.evidence_contract,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = require_record_object(
            value,
            schema=cls.SCHEMA,
            fields=(
                "role",
                "shard",
                "sha256",
                "size_bytes",
                "source_adapter_identity",
                "evidence_contract",
            ),
        )
        adapter = obj["source_adapter_identity"]
        if not isinstance(adapter, Mapping) or set(adapter) != {
            "adapter_id",
            "implementation_sha256",
        }:
            raise ProtocolError("source_adapter_identity has invalid fields")
        return cls(
            role=require_string(obj["role"], "role"),
            shard=require_string(obj["shard"], "shard", pattern=IDENTIFIER_PATTERN),
            sha256=require_string(obj["sha256"], "sha256", pattern=SHA256_PATTERN),
            size_bytes=require_integer(obj["size_bytes"], "size_bytes", minimum=1),
            source_adapter_identity=SourceAdapterIdentityV1(
                adapter_id=require_string(adapter["adapter_id"], "adapter_id"),
                implementation_sha256=require_string(
                    adapter["implementation_sha256"],
                    "implementation_sha256",
                    pattern=SHA256_PATTERN,
                ),
            ),
            evidence_contract=require_string(
                obj["evidence_contract"], "evidence_contract"
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceObservationEvidenceV1(CanonicalRecord):
    """One actually observed source receipt accumulated under an exact path."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.source-observation-evidence.v1"

    plan_id: str
    receipt: VerifiedSourceShardReceiptV1

    def __post_init__(self) -> None:
        require_string(self.plan_id, "plan_id", pattern=SHA256_PATTERN)
        if type(self.receipt) is not VerifiedSourceShardReceiptV1:
            raise ProtocolError("observed source receipt has the wrong type")
        if self.receipt.verification_contract != SOURCE_VERIFICATION_CONTRACT:
            raise ProtocolError("observed source receipt contract is incompatible")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "plan_id": self.plan_id,
            "receipt": self.receipt.as_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = require_record_object(
            value,
            schema=cls.SCHEMA,
            fields=("plan_id", "receipt"),
        )
        raw = obj["receipt"]
        expected = {
            "shard",
            "observed_sha256",
            "observed_size_bytes",
            "source_adapter_identity",
            "verification_contract",
        }
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ProtocolError("observed source receipt has invalid fields")
        adapter = raw["source_adapter_identity"]
        if not isinstance(adapter, Mapping) or set(adapter) != {
            "adapter_id",
            "implementation_sha256",
        }:
            raise ProtocolError("observed source adapter identity has invalid fields")
        return cls(
            plan_id=require_string(obj["plan_id"], "plan_id", pattern=SHA256_PATTERN),
            receipt=VerifiedSourceShardReceiptV1(
                shard=require_string(raw["shard"], "shard", pattern=IDENTIFIER_PATTERN),
                observed_sha256=require_string(
                    raw["observed_sha256"],
                    "observed_sha256",
                    pattern=SHA256_PATTERN,
                ),
                observed_size_bytes=require_integer(
                    raw["observed_size_bytes"], "observed_size_bytes", minimum=1
                ),
                source_adapter_identity=SourceAdapterIdentityV1(
                    adapter_id=require_string(adapter["adapter_id"], "adapter_id"),
                    implementation_sha256=require_string(
                        adapter["implementation_sha256"],
                        "implementation_sha256",
                        pattern=SHA256_PATTERN,
                    ),
                ),
                verification_contract=require_string(
                    raw["verification_contract"], "verification_contract"
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class ObservationSnapshotV1(CanonicalRecord):
    """Complete, content-addressed source observations for one exact plan.

    This evidence object is intentionally separate from the stable storage
    manifest.  It can appear monotonically after chunk storage was finalized,
    without changing checkpoint byte identity or making finalization order
    significant.
    """

    SCHEMA: ClassVar[str] = "moe-sae.r1.prefix-observation-snapshot.v1"

    plan_id: str
    source_adapter_identity: SourceAdapterIdentityV1
    expected_source_shards: tuple[str, ...]
    observations: tuple[SourceObservationEvidenceV1, ...]
    complete: bool = True
    production_sealable: bool = False
    snapshot_id: str = field(init=False)

    def __post_init__(self) -> None:
        require_string(self.plan_id, "plan_id", pattern=SHA256_PATTERN)
        if type(self.source_adapter_identity) is not SourceAdapterIdentityV1:
            raise ProtocolError("snapshot source adapter identity has the wrong type")
        if (
            type(self.expected_source_shards) is not tuple
            or not self.expected_source_shards
        ):
            raise ProtocolError("snapshot expected source shards must be non-empty")
        if self.expected_source_shards != tuple(
            sorted(self.expected_source_shards)
        ) or len(self.expected_source_shards) != len(set(self.expected_source_shards)):
            raise ProtocolError(
                "snapshot expected source shards must be unique and sorted"
            )
        if type(self.observations) is not tuple or any(
            type(item) is not SourceObservationEvidenceV1 for item in self.observations
        ):
            raise ProtocolError("snapshot observations have the wrong type")
        observed = tuple(item.receipt.shard for item in self.observations)
        if observed != self.expected_source_shards:
            raise ProtocolError("snapshot observations are incomplete or out of order")
        if any(item.plan_id != self.plan_id for item in self.observations):
            raise ProtocolError("snapshot contains an observation from another plan")
        if any(
            item.receipt.source_adapter_identity != self.source_adapter_identity
            for item in self.observations
        ):
            raise ProtocolError("snapshot source adapter identities differ")
        if self.complete is not True or self.production_sealable is not False:
            raise ProtocolError("snapshot diagnostic qualification flags are invalid")
        object.__setattr__(self, "snapshot_id", canonical_sha256(self.identity_dict()))

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "plan_id": self.plan_id,
            "source_adapter_identity": self.source_adapter_identity.as_dict(),
            "expected_source_shards": list(self.expected_source_shards),
            "observations": [item.as_dict() for item in self.observations],
            "complete": self.complete,
            "production_sealable": self.production_sealable,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_dict(), "snapshot_id": self.snapshot_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = require_record_object(
            value,
            schema=cls.SCHEMA,
            fields=(
                "snapshot_id",
                "plan_id",
                "source_adapter_identity",
                "expected_source_shards",
                "observations",
                "complete",
                "production_sealable",
            ),
        )
        adapter = obj["source_adapter_identity"]
        if not isinstance(adapter, Mapping) or set(adapter) != {
            "adapter_id",
            "implementation_sha256",
        }:
            raise ProtocolError("snapshot source adapter identity has invalid fields")
        result = cls(
            plan_id=require_string(obj["plan_id"], "plan_id", pattern=SHA256_PATTERN),
            source_adapter_identity=SourceAdapterIdentityV1(
                adapter_id=require_string(adapter["adapter_id"], "adapter_id"),
                implementation_sha256=require_string(
                    adapter["implementation_sha256"],
                    "implementation_sha256",
                    pattern=SHA256_PATTERN,
                ),
            ),
            expected_source_shards=_string_list(
                obj["expected_source_shards"], "expected_source_shards"
            ),
            observations=tuple(
                SourceObservationEvidenceV1.from_dict(item)
                for item in _record_list(obj["observations"], "observations")
            ),
            complete=_strict_bool(obj["complete"], "complete"),
            production_sealable=_strict_bool(
                obj["production_sealable"], "production_sealable"
            ),
        )
        supplied = require_string(
            obj["snapshot_id"], "snapshot_id", pattern=SHA256_PATTERN
        )
        if supplied != result.snapshot_id:
            raise ProtocolError("snapshot_id does not match canonical snapshot content")
        return result

    def validate_for_plan(self, plan: StreamingConversionPlanV1) -> None:
        if type(plan) is not StreamingConversionPlanV1 or self.plan_id != plan.plan_id:
            raise PrefixCheckpointError("observation snapshot belongs to another plan")
        descriptors = _source_descriptors(plan)
        if (
            self.source_adapter_identity != plan.source_adapter_identity
            or self.expected_source_shards != tuple(sorted(descriptors))
        ):
            raise PrefixCheckpointError(
                "observation snapshot identity differs from plan"
            )
        for evidence in self.observations:
            evidence.receipt.validate_for(
                descriptors[evidence.receipt.shard], plan.source_adapter_identity
            )


def _source_descriptors(plan: StreamingConversionPlanV1) -> dict[str, Any]:
    return {
        descriptor.shard: descriptor
        for chunk in plan.chunks
        for descriptor in (chunk.source_descriptor, chunk.scale_descriptor)
        if descriptor is not None
    }


def _lineage_for_chunk(
    plan: StreamingConversionPlanV1, chunk: ConversionChunkIdentityV1
) -> tuple[PlanSourceIdentityBindingV1, ...]:
    result = [
        PlanSourceIdentityBindingV1(
            role="source",
            shard=chunk.source_descriptor.shard,
            sha256=chunk.source_descriptor.expected_lfs_sha256,
            size_bytes=chunk.source_descriptor.expected_blob_size_bytes,
            source_adapter_identity=plan.source_adapter_identity,
        )
    ]
    if (
        chunk.scale_descriptor is not None
        and chunk.scale_descriptor != chunk.source_descriptor
    ):
        result.append(
            PlanSourceIdentityBindingV1(
                role="scale",
                shard=chunk.scale_descriptor.shard,
                sha256=chunk.scale_descriptor.expected_lfs_sha256,
                size_bytes=chunk.scale_descriptor.expected_blob_size_bytes,
                source_adapter_identity=plan.source_adapter_identity,
            )
        )
    return tuple(result)


def _output_receipt_from_dict(value: Mapping[str, Any]) -> OutputChunkReceiptV1:
    expected = {
        "plan_id",
        "chunk_id",
        "sink_id",
        "destination_key",
        "destination_flat_start",
        "destination_flat_stop",
        "dtype",
        "output_sha256",
        "output_size_bytes",
        "immutable_object_id",
        "verification_contract",
    }
    if set(value) != expected:
        raise ProtocolError("output receipt has missing or unknown fields")
    return OutputChunkReceiptV1(
        plan_id=require_string(value["plan_id"], "plan_id", pattern=SHA256_PATTERN),
        chunk_id=require_string(value["chunk_id"], "chunk_id", pattern=SHA256_PATTERN),
        sink_id=require_string(value["sink_id"], "sink_id", pattern=IDENTIFIER_PATTERN),
        destination_key=require_string(value["destination_key"], "destination_key"),
        destination_flat_start=require_integer(
            value["destination_flat_start"], "destination_flat_start", minimum=0
        ),
        destination_flat_stop=require_integer(
            value["destination_flat_stop"], "destination_flat_stop", minimum=1
        ),
        dtype=require_string(value["dtype"], "dtype"),
        output_sha256=require_string(
            value["output_sha256"], "output_sha256", pattern=SHA256_PATTERN
        ),
        output_size_bytes=require_integer(
            value["output_size_bytes"], "output_size_bytes", minimum=1
        ),
        immutable_object_id=_validate_object_id(
            require_string(value["immutable_object_id"], "immutable_object_id")
        ),
        verification_contract=require_string(
            value["verification_contract"], "verification_contract"
        ),
    )


def _observation_snapshot_object_id(plan_id: str, snapshot_id: str) -> str:
    return _validate_object_id(
        f"plans/{plan_id}/observation-snapshots/{snapshot_id}.json"
    )


@dataclass(frozen=True, slots=True)
class PrefixChunkBindingV1(CanonicalRecord):
    """Durable binding between a logical output receipt and physical bytes."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.prefix-chunk-binding.v1"

    plan_id: str
    chunk_id: str
    output_receipt: OutputChunkReceiptV1
    physical_receipt: ImmutablePhysicalObjectReceiptV1
    source_lineage: tuple[PlanSourceIdentityBindingV1, ...]
    sink_contract: str = PREFIX_SINK_CONTRACT

    def __post_init__(self) -> None:
        require_string(self.plan_id, "plan_id", pattern=SHA256_PATTERN)
        require_string(self.chunk_id, "chunk_id", pattern=SHA256_PATTERN)
        if type(self.output_receipt) is not OutputChunkReceiptV1:
            raise ProtocolError("output receipt has the wrong type")
        if type(self.physical_receipt) is not ImmutablePhysicalObjectReceiptV1:
            raise ProtocolError("physical receipt has the wrong type")
        if type(self.source_lineage) is not tuple or not self.source_lineage:
            raise ProtocolError("source lineage must be a non-empty tuple")
        if any(
            type(item) is not PlanSourceIdentityBindingV1
            for item in self.source_lineage
        ):
            raise ProtocolError("source lineage item has the wrong type")
        if (
            self.plan_id != self.output_receipt.plan_id
            or self.chunk_id != self.output_receipt.chunk_id
        ):
            raise ProtocolError("chunk binding identity differs from output receipt")
        if (
            self.output_receipt.immutable_object_id != self.physical_receipt.object_id
            or self.output_receipt.output_size_bytes != self.physical_receipt.size_bytes
            or self.output_receipt.output_sha256 != self.physical_receipt.sha256
        ):
            raise ProtocolError("logical and physical chunk receipts do not bind")
        if self.sink_contract != PREFIX_SINK_CONTRACT:
            raise ProtocolError("unsupported prefix sink contract")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "plan_id": self.plan_id,
            "chunk_id": self.chunk_id,
            "output_receipt": self.output_receipt.as_dict(),
            "physical_receipt": self.physical_receipt.as_dict(),
            "source_lineage": [item.as_dict() for item in self.source_lineage],
            "sink_contract": self.sink_contract,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = require_record_object(
            value,
            schema=cls.SCHEMA,
            fields=(
                "plan_id",
                "chunk_id",
                "output_receipt",
                "physical_receipt",
                "source_lineage",
                "sink_contract",
            ),
        )
        output = obj["output_receipt"]
        physical = obj["physical_receipt"]
        if not isinstance(output, Mapping) or not isinstance(physical, Mapping):
            raise ProtocolError("chunk binding receipts must be objects")
        return cls(
            plan_id=require_string(obj["plan_id"], "plan_id", pattern=SHA256_PATTERN),
            chunk_id=require_string(
                obj["chunk_id"], "chunk_id", pattern=SHA256_PATTERN
            ),
            output_receipt=_output_receipt_from_dict(output),
            physical_receipt=ImmutablePhysicalObjectReceiptV1.from_dict(physical),
            source_lineage=tuple(
                PlanSourceIdentityBindingV1.from_dict(item)
                for item in _record_list(obj["source_lineage"], "source_lineage")
            ),
            sink_contract=require_string(obj["sink_contract"], "sink_contract"),
        )


class ImmutablePrefixChunkSinkV1(ConversionOutputSink):
    """Conversion sink backed by exact-path immutable local byte objects."""

    def __init__(
        self, store: ImmutableByteStore, plan: StreamingConversionPlanV1
    ) -> None:
        if type(plan) is not StreamingConversionPlanV1:
            raise PrefixCheckpointError("prefix sink requires an exact conversion plan")
        store_id = store.store_id
        require_string(store_id, "store_id", pattern=IDENTIFIER_PATTERN)
        stream_buffer_bytes = store.stream_buffer_bytes
        if (
            type(stream_buffer_bytes) is not int
            or stream_buffer_bytes <= 0
            or stream_buffer_bytes > plan.limits.sink_staging_buffer_bytes
        ):
            raise PrefixCheckpointError(
                "immutable store stream buffer exceeds the plan sink-staging bound"
            )
        if store.distributed_identity_qualified is not False:
            raise PrefixCheckpointError(
                "the local diagnostic sink cannot accept a distributed store identity"
            )
        self.store = store
        self._plan_id = plan.plan_id
        self._sink_id = (
            "local-prefix-chunk-sink-v1:"
            + hashlib.sha256(store_id.encode("utf-8")).hexdigest()
        )

    @property
    def sink_id(self) -> str:
        return self._sink_id

    def _validate_plan(self, plan: StreamingConversionPlanV1) -> None:
        stream_buffer_bytes = self.store.stream_buffer_bytes
        if (
            type(plan) is not StreamingConversionPlanV1
            or plan.plan_id != self._plan_id
            or type(stream_buffer_bytes) is not int
            or stream_buffer_bytes <= 0
            or stream_buffer_bytes > plan.limits.sink_staging_buffer_bytes
            or self.store.distributed_identity_qualified is not False
        ):
            raise PrefixCheckpointError(
                "prefix sink plan or stream-buffer admission changed"
            )

    def _load_binding(
        self, plan: StreamingConversionPlanV1, chunk: ConversionChunkIdentityV1
    ) -> PrefixChunkBindingV1 | None:
        self._validate_plan(plan)
        binding_id = _binding_object_id(plan.plan_id, chunk.chunk_id)
        probed = self.store.probe_bounded(binding_id, maximum_bytes=_MAX_RECEIPT_BYTES)
        if probed is None:
            return None
        raw, _binding_physical = probed
        try:
            binding = PrefixChunkBindingV1.from_json(raw)
        except (ProtocolError, CheckpointConversionError) as error:
            raise PrefixCheckpointError(
                f"stored chunk binding is malformed: {chunk.chunk_id}"
            ) from error
        expected_object_id = _chunk_object_id(plan.plan_id, chunk.chunk_id)
        binding.output_receipt.validate_for(plan, chunk, sink_id=self.sink_id)
        if (
            binding.plan_id != plan.plan_id
            or binding.chunk_id != chunk.chunk_id
            or binding.physical_receipt.object_id != expected_object_id
            or binding.physical_receipt.store_id != self.store.store_id
            or binding.source_lineage != _lineage_for_chunk(plan, chunk)
        ):
            raise PrefixCheckpointError(
                f"stored chunk binding conflicts with plan: {chunk.chunk_id}"
            )
        self.store.verify(binding.physical_receipt)
        return binding

    def probe_committed(
        self, plan: StreamingConversionPlanV1, chunk: ConversionChunkIdentityV1
    ) -> OutputChunkReceiptV1 | None:
        if (
            type(plan) is not StreamingConversionPlanV1
            or type(chunk) is not ConversionChunkIdentityV1
        ):
            raise PrefixCheckpointError(
                "prefix sink requires exact plan and chunk types"
            )
        self._validate_plan(plan)
        binding = self._load_binding(plan, chunk)
        return None if binding is None else binding.output_receipt

    def commit_chunk(
        self,
        plan: StreamingConversionPlanV1,
        chunk: ConversionChunkIdentityV1,
        payload: memoryview,
        *,
        output_sha256: str,
    ) -> OutputChunkReceiptV1:
        if (
            type(plan) is not StreamingConversionPlanV1
            or type(chunk) is not ConversionChunkIdentityV1
        ):
            raise PrefixCheckpointError(
                "prefix sink requires exact plan and chunk types"
            )
        self._validate_plan(plan)
        if type(payload) is not memoryview:
            raise PrefixCheckpointError("prefix sink requires a memoryview payload")
        require_string(output_sha256, "output_sha256", pattern=SHA256_PATTERN)
        object_id = _chunk_object_id(plan.plan_id, chunk.chunk_id)
        physical = self.store.create_or_verify(
            object_id,
            payload,
            expected_sha256=output_sha256,
            expected_size_bytes=chunk.output_bytes,
        )
        output = OutputChunkReceiptV1(
            plan_id=plan.plan_id,
            chunk_id=chunk.chunk_id,
            sink_id=self.sink_id,
            destination_key=chunk.mapping.destination_key,
            destination_flat_start=chunk.destination_flat_start,
            destination_flat_stop=chunk.destination_flat_stop,
            dtype=chunk.mapping.destination_dtype,
            output_sha256=physical.sha256,
            output_size_bytes=physical.size_bytes,
            immutable_object_id=physical.object_id,
            verification_contract=SINK_VERIFICATION_CONTRACT,
        )
        binding = PrefixChunkBindingV1(
            plan_id=plan.plan_id,
            chunk_id=chunk.chunk_id,
            output_receipt=output,
            physical_receipt=physical,
            source_lineage=_lineage_for_chunk(plan, chunk),
        )
        encoded = binding.canonical_bytes()
        self.store.create_or_verify(
            _binding_object_id(plan.plan_id, chunk.chunk_id),
            encoded,
            expected_sha256=hashlib.sha256(encoded).hexdigest(),
            expected_size_bytes=len(encoded),
        )
        loaded = self._load_binding(plan, chunk)
        if loaded != binding:
            raise ImmutableObjectConflict(
                f"concurrent chunk binding differs for {chunk.chunk_id}"
            )
        return output

    def binding_for(
        self, plan: StreamingConversionPlanV1, chunk: ConversionChunkIdentityV1
    ) -> PrefixChunkBindingV1:
        binding = self._load_binding(plan, chunk)
        if binding is None:
            raise PrefixCheckpointError(f"missing committed chunk {chunk.chunk_id}")
        return binding


@dataclass(frozen=True, slots=True)
class StoredDestinationV1(CanonicalRecord):
    SCHEMA: ClassVar[str] = "moe-sae.r1.stored-prefix-destination.v1"

    destination_key: str
    dtype: str
    shape: tuple[int, ...]
    payload_bytes: int
    chunks: tuple[PrefixChunkBindingV1, ...]

    def __post_init__(self) -> None:
        require_string(
            self.destination_key, "destination_key", pattern=IDENTIFIER_PATTERN
        )
        if self.dtype not in _DTYPES:
            raise ProtocolError("stored destination dtype is unsupported")
        if (
            type(self.shape) is not tuple
            or not self.shape
            or any(type(x) is not int or x <= 0 for x in self.shape)
        ):
            raise ProtocolError("stored destination shape is invalid")
        require_integer(self.payload_bytes, "payload_bytes", minimum=1)
        if (
            self.payload_bytes
            != int(np.prod(self.shape)) * _DTYPES[self.dtype].itemsize
        ):
            raise ProtocolError("stored destination byte total is inconsistent")
        if type(self.chunks) is not tuple or not self.chunks:
            raise ProtocolError("stored destination chunks must be non-empty")
        cursor = 0
        for binding in self.chunks:
            if type(binding) is not PrefixChunkBindingV1:
                raise ProtocolError("stored destination chunk type is invalid")
            receipt = binding.output_receipt
            if (
                receipt.destination_key != self.destination_key
                or receipt.dtype != self.dtype
                or receipt.destination_flat_start != cursor
                or receipt.destination_flat_stop <= cursor
            ):
                raise ProtocolError(
                    "stored destination has a gap, overlap, or foreign chunk"
                )
            cursor = receipt.destination_flat_stop
        if cursor != int(np.prod(self.shape)):
            raise ProtocolError(
                "stored destination does not cover its full flat interval"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "destination_key": self.destination_key,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "payload_bytes": self.payload_bytes,
            "chunks": [item.as_dict() for item in self.chunks],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = require_record_object(
            value,
            schema=cls.SCHEMA,
            fields=("destination_key", "dtype", "shape", "payload_bytes", "chunks"),
        )
        return cls(
            destination_key=require_string(obj["destination_key"], "destination_key"),
            dtype=require_string(obj["dtype"], "dtype"),
            shape=_shape(obj["shape"], "shape"),
            payload_bytes=require_integer(
                obj["payload_bytes"], "payload_bytes", minimum=1
            ),
            chunks=tuple(
                PrefixChunkBindingV1.from_dict(item)
                for item in _record_list(obj["chunks"], "chunks")
            ),
        )


def _runtime_dict(value: R1RuntimeLockIdentityV1 | None) -> dict[str, str] | None:
    return None if value is None else value.as_dict()


@dataclass(frozen=True, slots=True)
class PrefixCheckpointManifestV1(CanonicalRecord):
    """Canonical diagnostic index over an exactly completed conversion plan."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.prefix-checkpoint-manifest.v1"

    plan_id: str
    mode: str
    selection_sha256: str
    full_inventory_sha256: str
    selected_inventory_sha256: str
    source_revision: str
    maxtext_revision: str
    geometry_sha256: str
    engine_sha256: str
    numeric_codec_sha256: str
    source_adapter_identity: SourceAdapterIdentityV1
    runtime_lock_identity: R1RuntimeLockIdentityV1 | None
    sink_id: str
    store_id: str
    total_chunks: int
    converted_payload_bytes: int
    max_estimated_resident_bytes: int
    destinations: tuple[StoredDestinationV1, ...]
    store_identity_scope: str = LOCAL_STORE_IDENTITY_SCOPE
    missing_chunk_ids: tuple[str, ...] = ()
    storage_complete: bool = True
    production_sealable: bool = False
    unplanned_object_absence_asserted: bool = False
    object_discovery_contract: str = OBJECT_DISCOVERY_CONTRACT
    engine_contract: str = CONVERSION_ENGINE_CONTRACT
    chunking_contract: str = CHUNKING_CONTRACT
    dequantization_contract: str = DEQUANTIZATION_CONTRACT
    unmet_production_gaps: tuple[str, ...] = UNMET_PRODUCTION_GAPS
    manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "plan_id",
            "selection_sha256",
            "full_inventory_sha256",
            "selected_inventory_sha256",
            "geometry_sha256",
            "engine_sha256",
            "numeric_codec_sha256",
        ):
            require_string(getattr(self, name), name, pattern=SHA256_PATTERN)
        require_string(self.mode, "mode", pattern=IDENTIFIER_PATTERN)
        require_string(
            self.source_revision, "source_revision", pattern=IDENTIFIER_PATTERN
        )
        require_string(
            self.maxtext_revision, "maxtext_revision", pattern=IDENTIFIER_PATTERN
        )
        if type(self.source_adapter_identity) is not SourceAdapterIdentityV1:
            raise ProtocolError("manifest source adapter identity has the wrong type")
        if (
            self.runtime_lock_identity is not None
            and type(self.runtime_lock_identity) is not R1RuntimeLockIdentityV1
        ):
            raise ProtocolError("manifest runtime lock identity has the wrong type")
        require_string(self.sink_id, "sink_id", pattern=IDENTIFIER_PATTERN)
        require_string(self.store_id, "store_id", pattern=IDENTIFIER_PATTERN)
        if self.store_identity_scope != LOCAL_STORE_IDENTITY_SCOPE:
            raise ProtocolError("manifest store identity is not explicitly local-only")
        require_integer(self.total_chunks, "total_chunks", minimum=1)
        require_integer(
            self.converted_payload_bytes, "converted_payload_bytes", minimum=1
        )
        require_integer(
            self.max_estimated_resident_bytes, "max_estimated_resident_bytes", minimum=1
        )
        if type(self.destinations) is not tuple or not self.destinations:
            raise ProtocolError("manifest destinations must be a non-empty tuple")
        if any(type(item) is not StoredDestinationV1 for item in self.destinations):
            raise ProtocolError("manifest destination type is invalid")
        keys = tuple(item.destination_key for item in self.destinations)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ProtocolError("manifest destinations must be unique and sorted")
        chunks = tuple(
            chunk for destination in self.destinations for chunk in destination.chunks
        )
        ids = tuple(chunk.chunk_id for chunk in chunks)
        if len(chunks) != self.total_chunks or len(ids) != len(set(ids)):
            raise ProtocolError("manifest chunk count or uniqueness is inconsistent")
        if (
            sum(item.payload_bytes for item in self.destinations)
            != self.converted_payload_bytes
        ):
            raise ProtocolError("manifest payload total is inconsistent")
        if type(self.missing_chunk_ids) is not tuple or self.missing_chunk_ids:
            raise ProtocolError("a complete manifest cannot contain missing chunk IDs")
        if (
            self.storage_complete is not True
            or self.production_sealable is not False
            or self.unplanned_object_absence_asserted is not False
        ):
            raise ProtocolError("diagnostic manifest evidence flags are invalid")
        if self.object_discovery_contract != OBJECT_DISCOVERY_CONTRACT:
            raise ProtocolError("unsupported object discovery contract")
        if (
            self.engine_contract != CONVERSION_ENGINE_CONTRACT
            or self.chunking_contract != CHUNKING_CONTRACT
            or self.dequantization_contract != DEQUANTIZATION_CONTRACT
        ):
            raise ProtocolError("manifest conversion contracts are incompatible")
        if self.unmet_production_gaps != UNMET_PRODUCTION_GAPS:
            raise ProtocolError(
                "manifest unmet production gaps are not the exact protocol set"
            )
        object.__setattr__(self, "manifest_id", canonical_sha256(self.identity_dict()))

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "plan_id": self.plan_id,
            "mode": self.mode,
            "selection_sha256": self.selection_sha256,
            "full_inventory_sha256": self.full_inventory_sha256,
            "selected_inventory_sha256": self.selected_inventory_sha256,
            "source_revision": self.source_revision,
            "maxtext_revision": self.maxtext_revision,
            "geometry_sha256": self.geometry_sha256,
            "engine_sha256": self.engine_sha256,
            "numeric_codec_sha256": self.numeric_codec_sha256,
            "source_adapter_identity": self.source_adapter_identity.as_dict(),
            "runtime_lock_identity": _runtime_dict(self.runtime_lock_identity),
            "sink_id": self.sink_id,
            "store_id": self.store_id,
            "store_identity_scope": self.store_identity_scope,
            "total_chunks": self.total_chunks,
            "converted_payload_bytes": self.converted_payload_bytes,
            "max_estimated_resident_bytes": self.max_estimated_resident_bytes,
            "destinations": [item.as_dict() for item in self.destinations],
            "missing_chunk_ids": list(self.missing_chunk_ids),
            "storage_complete": self.storage_complete,
            "production_sealable": self.production_sealable,
            "unplanned_object_absence_asserted": self.unplanned_object_absence_asserted,
            "object_discovery_contract": self.object_discovery_contract,
            "engine_contract": self.engine_contract,
            "chunking_contract": self.chunking_contract,
            "dequantization_contract": self.dequantization_contract,
            "unmet_production_gaps": list(self.unmet_production_gaps),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity_dict(), "manifest_id": self.manifest_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "manifest_id",
            "plan_id",
            "mode",
            "selection_sha256",
            "full_inventory_sha256",
            "selected_inventory_sha256",
            "source_revision",
            "maxtext_revision",
            "geometry_sha256",
            "engine_sha256",
            "numeric_codec_sha256",
            "source_adapter_identity",
            "runtime_lock_identity",
            "sink_id",
            "store_id",
            "store_identity_scope",
            "total_chunks",
            "converted_payload_bytes",
            "max_estimated_resident_bytes",
            "destinations",
            "missing_chunk_ids",
            "storage_complete",
            "production_sealable",
            "unplanned_object_absence_asserted",
            "object_discovery_contract",
            "engine_contract",
            "chunking_contract",
            "dequantization_contract",
            "unmet_production_gaps",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        adapter = obj["source_adapter_identity"]
        if not isinstance(adapter, Mapping) or set(adapter) != {
            "adapter_id",
            "implementation_sha256",
        }:
            raise ProtocolError("manifest source adapter identity has invalid fields")
        runtime_raw = _optional_record(
            obj["runtime_lock_identity"], "runtime_lock_identity"
        )
        runtime = None
        if runtime_raw is not None:
            if set(runtime_raw) != {
                "maxtext_revision",
                "pyproject_sha256",
                "uv_lock_sha256",
            }:
                raise ProtocolError("manifest runtime lock identity has invalid fields")
            runtime = R1RuntimeLockIdentityV1(
                maxtext_revision=require_string(
                    runtime_raw["maxtext_revision"], "maxtext_revision"
                ),
                pyproject_sha256=require_string(
                    runtime_raw["pyproject_sha256"],
                    "pyproject_sha256",
                    pattern=SHA256_PATTERN,
                ),
                uv_lock_sha256=require_string(
                    runtime_raw["uv_lock_sha256"],
                    "uv_lock_sha256",
                    pattern=SHA256_PATTERN,
                ),
            )
        result = cls(
            plan_id=require_string(obj["plan_id"], "plan_id", pattern=SHA256_PATTERN),
            mode=require_string(obj["mode"], "mode"),
            selection_sha256=require_string(
                obj["selection_sha256"], "selection_sha256", pattern=SHA256_PATTERN
            ),
            full_inventory_sha256=require_string(
                obj["full_inventory_sha256"],
                "full_inventory_sha256",
                pattern=SHA256_PATTERN,
            ),
            selected_inventory_sha256=require_string(
                obj["selected_inventory_sha256"],
                "selected_inventory_sha256",
                pattern=SHA256_PATTERN,
            ),
            source_revision=require_string(obj["source_revision"], "source_revision"),
            maxtext_revision=require_string(
                obj["maxtext_revision"], "maxtext_revision"
            ),
            geometry_sha256=require_string(
                obj["geometry_sha256"], "geometry_sha256", pattern=SHA256_PATTERN
            ),
            engine_sha256=require_string(
                obj["engine_sha256"], "engine_sha256", pattern=SHA256_PATTERN
            ),
            numeric_codec_sha256=require_string(
                obj["numeric_codec_sha256"],
                "numeric_codec_sha256",
                pattern=SHA256_PATTERN,
            ),
            source_adapter_identity=SourceAdapterIdentityV1(
                adapter_id=require_string(adapter["adapter_id"], "adapter_id"),
                implementation_sha256=require_string(
                    adapter["implementation_sha256"],
                    "implementation_sha256",
                    pattern=SHA256_PATTERN,
                ),
            ),
            runtime_lock_identity=runtime,
            sink_id=require_string(obj["sink_id"], "sink_id"),
            store_id=require_string(obj["store_id"], "store_id"),
            store_identity_scope=require_string(
                obj["store_identity_scope"], "store_identity_scope"
            ),
            total_chunks=require_integer(
                obj["total_chunks"], "total_chunks", minimum=1
            ),
            converted_payload_bytes=require_integer(
                obj["converted_payload_bytes"], "converted_payload_bytes", minimum=1
            ),
            max_estimated_resident_bytes=require_integer(
                obj["max_estimated_resident_bytes"],
                "max_estimated_resident_bytes",
                minimum=1,
            ),
            destinations=tuple(
                StoredDestinationV1.from_dict(item)
                for item in _record_list(obj["destinations"], "destinations")
            ),
            missing_chunk_ids=_string_list(
                obj["missing_chunk_ids"], "missing_chunk_ids"
            ),
            storage_complete=_strict_bool(obj["storage_complete"], "storage_complete"),
            production_sealable=_strict_bool(
                obj["production_sealable"], "production_sealable"
            ),
            unplanned_object_absence_asserted=_strict_bool(
                obj["unplanned_object_absence_asserted"],
                "unplanned_object_absence_asserted",
            ),
            object_discovery_contract=require_string(
                obj["object_discovery_contract"], "object_discovery_contract"
            ),
            engine_contract=require_string(obj["engine_contract"], "engine_contract"),
            chunking_contract=require_string(
                obj["chunking_contract"], "chunking_contract"
            ),
            dequantization_contract=require_string(
                obj["dequantization_contract"], "dequantization_contract"
            ),
            unmet_production_gaps=_string_list(
                obj["unmet_production_gaps"], "unmet_production_gaps"
            ),
        )
        supplied = require_string(
            obj["manifest_id"], "manifest_id", pattern=SHA256_PATTERN
        )
        if supplied != result.manifest_id:
            raise ProtocolError("manifest_id does not match canonical manifest content")
        return result

    def validate_for_plan(
        self, plan: StreamingConversionPlanV1, *, sink_id: str, store_id: str
    ) -> None:
        if type(plan) is not StreamingConversionPlanV1:
            raise PrefixCheckpointError("manifest validation requires an exact plan")
        expected_header = (
            plan.plan_id,
            plan.mode,
            plan.selection_sha256,
            plan.full_inventory_sha256,
            plan.selected_inventory_sha256,
            plan.source_revision,
            plan.maxtext_revision,
            plan.geometry_sha256,
            plan.engine_sha256,
            plan.numeric_codec_sha256,
            plan.source_adapter_identity,
            plan.runtime_lock_identity,
            sink_id,
            store_id,
            LOCAL_STORE_IDENTITY_SCOPE,
            len(plan.chunks),
            plan.converted_payload_bytes,
            max(item.estimated_peak_resident_bytes for item in plan.shard_work),
            UNMET_PRODUCTION_GAPS,
        )
        actual_header = (
            self.plan_id,
            self.mode,
            self.selection_sha256,
            self.full_inventory_sha256,
            self.selected_inventory_sha256,
            self.source_revision,
            self.maxtext_revision,
            self.geometry_sha256,
            self.engine_sha256,
            self.numeric_codec_sha256,
            self.source_adapter_identity,
            self.runtime_lock_identity,
            self.sink_id,
            self.store_id,
            self.store_identity_scope,
            self.total_chunks,
            self.converted_payload_bytes,
            self.max_estimated_resident_bytes,
            self.unmet_production_gaps,
        )
        if actual_header != expected_header:
            raise PrefixCheckpointError(
                "manifest identity conflicts with conversion plan"
            )
        by_id = {chunk.chunk_id: chunk for chunk in plan.chunks}
        if len(by_id) != self.total_chunks:
            raise PrefixCheckpointError("plan chunk identity collision")
        if len(self.destinations) != len(plan.destinations):
            raise PrefixCheckpointError("manifest destination count differs from plan")
        for stored, destination in zip(
            self.destinations, plan.destinations, strict=True
        ):
            if (
                stored.destination_key != destination.destination_key
                or stored.dtype != destination.dtype
                or stored.shape != destination.shape
                or stored.payload_bytes != destination.payload_bytes
                or len(stored.chunks) != len(destination.chunks)
            ):
                raise PrefixCheckpointError("manifest destination differs from plan")
            for binding, interval in zip(
                stored.chunks, destination.chunks, strict=True
            ):
                start, stop, chunk_id = interval
                chunk = by_id.get(chunk_id)
                if chunk is None:
                    raise PrefixCheckpointError("manifest references a foreign chunk")
                binding.output_receipt.validate_for(plan, chunk, sink_id=sink_id)
                if (
                    binding.chunk_id != chunk_id
                    or binding.output_receipt.destination_flat_start != start
                    or binding.output_receipt.destination_flat_stop != stop
                    or binding.physical_receipt.object_id
                    != _chunk_object_id(plan.plan_id, chunk_id)
                    or binding.physical_receipt.store_id != store_id
                    or binding.source_lineage != _lineage_for_chunk(plan, chunk)
                ):
                    raise PrefixCheckpointError(
                        "manifest chunk binding differs from plan"
                    )


def _validate_summary(
    plan: StreamingConversionPlanV1,
    summary: ConversionAttemptSummaryV1,
    sink: ImmutablePrefixChunkSinkV1,
) -> None:
    if type(summary) is not ConversionAttemptSummaryV1:
        raise PrefixCheckpointError("finalization requires an exact attempt summary")
    if (
        summary.plan_id != plan.plan_id
        or summary.sink_id != sink.sink_id
        or summary.total_chunks != len(plan.chunks)
        or summary.converted_payload_bytes != plan.converted_payload_bytes
        or summary.max_estimated_resident_bytes
        != max(item.estimated_peak_resident_bytes for item in plan.shard_work)
        or len(summary.receipts) != len(plan.chunks)
    ):
        raise PrefixCheckpointError("attempt summary conflicts with conversion plan")
    for receipt, chunk in zip(summary.receipts, plan.chunks, strict=True):
        receipt.validate_for(plan, chunk, sink_id=sink.sink_id)
    # These are observations from this invocation only.  Validate every one
    # that exists, but do not elevate a partial resumed set into a complete
    # source-observation claim in the canonical storage manifest.
    descriptors = _source_descriptors(plan)
    for receipt in summary.source_receipts:
        descriptor = descriptors.get(receipt.shard)
        if descriptor is None:
            raise PrefixCheckpointError("summary contains a foreign source receipt")
        receipt.validate_for(descriptor, plan.source_adapter_identity)


def _accumulate_source_observations(
    plan: StreamingConversionPlanV1,
    summary: ConversionAttemptSummaryV1,
    store: ImmutableByteStore,
) -> tuple[SourceObservationEvidenceV1, ...]:
    """Persist current observations and read the cumulative exact-path set."""

    for receipt in summary.source_receipts:
        evidence = SourceObservationEvidenceV1(plan_id=plan.plan_id, receipt=receipt)
        encoded = evidence.canonical_bytes()
        store.create_or_verify(
            _source_observation_object_id(plan.plan_id, receipt.shard),
            encoded,
            expected_sha256=hashlib.sha256(encoded).hexdigest(),
            expected_size_bytes=len(encoded),
        )
    return _read_source_observations(plan, store)


def _read_source_observations(
    plan: StreamingConversionPlanV1, store: ImmutableByteStore
) -> tuple[SourceObservationEvidenceV1, ...]:
    """Read only exact plan-derived paths; partial evidence remains diagnostic."""

    descriptors = _source_descriptors(plan)
    accumulated: list[SourceObservationEvidenceV1] = []
    for shard in sorted(descriptors):
        probed = store.probe_bounded(
            _source_observation_object_id(plan.plan_id, shard),
            maximum_bytes=_MAX_RECEIPT_BYTES,
        )
        if probed is None:
            continue
        raw, _physical = probed
        try:
            evidence = SourceObservationEvidenceV1.from_json(raw)
        except (ProtocolError, CheckpointConversionError) as error:
            raise PrefixCheckpointError(
                f"stored source observation is malformed: {shard}"
            ) from error
        if evidence.plan_id != plan.plan_id or evidence.receipt.shard != shard:
            raise PrefixCheckpointError(
                f"stored source observation conflicts with plan: {shard}"
            )
        evidence.receipt.validate_for(descriptors[shard], plan.source_adapter_identity)
        accumulated.append(evidence)
    return tuple(accumulated)


def _publish_complete_observation_snapshot(
    plan: StreamingConversionPlanV1,
    observations: tuple[SourceObservationEvidenceV1, ...],
    store: ImmutableByteStore,
) -> ObservationSnapshotV1 | None:
    expected_shards = tuple(sorted(_source_descriptors(plan)))
    if tuple(item.receipt.shard for item in observations) != expected_shards:
        return None
    snapshot = ObservationSnapshotV1(
        plan_id=plan.plan_id,
        source_adapter_identity=plan.source_adapter_identity,
        expected_source_shards=expected_shards,
        observations=observations,
    )
    snapshot.validate_for_plan(plan)
    encoded = snapshot.canonical_bytes()
    if len(encoded) > _MAX_RECEIPT_BYTES:
        raise PrefixCheckpointError(
            "complete observation snapshot exceeds its admission bound"
        )
    store.create_or_verify(
        _observation_snapshot_object_id(plan.plan_id, snapshot.snapshot_id),
        encoded,
        expected_sha256=hashlib.sha256(encoded).hexdigest(),
        expected_size_bytes=len(encoded),
    )
    return snapshot


def finalize_prefix_checkpoint(
    plan: StreamingConversionPlanV1,
    summary: ConversionAttemptSummaryV1,
    sink: ImmutablePrefixChunkSinkV1,
) -> PrefixCheckpointManifestV1:
    """Verify every exact object and publish a diagnostic manifest once."""

    if (
        type(plan) is not StreamingConversionPlanV1
        or type(sink) is not ImmutablePrefixChunkSinkV1
    ):
        raise PrefixCheckpointError("finalization requires exact plan and sink types")
    _validate_summary(plan, summary, sink)
    source_observations = _accumulate_source_observations(plan, summary, sink.store)
    _publish_complete_observation_snapshot(plan, source_observations, sink.store)
    chunks = {chunk.chunk_id: chunk for chunk in plan.chunks}
    summary_receipts = {receipt.chunk_id: receipt for receipt in summary.receipts}
    destinations: list[StoredDestinationV1] = []
    for destination in plan.destinations:
        bindings = tuple(
            sink.binding_for(plan, chunks[chunk_id])
            for _start, _stop, chunk_id in destination.chunks
        )
        if any(
            summary_receipts.get(binding.chunk_id) != binding.output_receipt
            for binding in bindings
        ):
            raise PrefixCheckpointError(
                "attempt summary output receipt differs from durable chunk binding"
            )
        destinations.append(
            StoredDestinationV1(
                destination_key=destination.destination_key,
                dtype=destination.dtype,
                shape=destination.shape,
                payload_bytes=destination.payload_bytes,
                chunks=bindings,
            )
        )
    manifest = PrefixCheckpointManifestV1(
        plan_id=plan.plan_id,
        mode=plan.mode,
        selection_sha256=plan.selection_sha256,
        full_inventory_sha256=plan.full_inventory_sha256,
        selected_inventory_sha256=plan.selected_inventory_sha256,
        source_revision=plan.source_revision,
        maxtext_revision=plan.maxtext_revision,
        geometry_sha256=plan.geometry_sha256,
        engine_sha256=plan.engine_sha256,
        numeric_codec_sha256=plan.numeric_codec_sha256,
        source_adapter_identity=plan.source_adapter_identity,
        runtime_lock_identity=plan.runtime_lock_identity,
        sink_id=sink.sink_id,
        store_id=sink.store.store_id,
        total_chunks=len(plan.chunks),
        converted_payload_bytes=plan.converted_payload_bytes,
        max_estimated_resident_bytes=summary.max_estimated_resident_bytes,
        destinations=tuple(destinations),
    )
    manifest.validate_for_plan(plan, sink_id=sink.sink_id, store_id=sink.store.store_id)
    encoded = manifest.canonical_bytes()
    if len(encoded) > _MAX_MANIFEST_BYTES:
        raise PrefixCheckpointError(
            "prefix checkpoint manifest exceeds reader admission bound"
        )
    sink.store.create_or_verify(
        _manifest_object_id(plan.plan_id),
        encoded,
        expected_sha256=hashlib.sha256(encoded).hexdigest(),
        expected_size_bytes=len(encoded),
    )
    return manifest


class PrefixCheckpointReaderV1:
    """Plan-bound, full-hash-verifying reader for flat tensor intervals."""

    def __init__(
        self,
        store: ImmutableByteStore,
        plan: StreamingConversionPlanV1,
        manifest: PrefixCheckpointManifestV1,
        *,
        maximum_read_bytes: int = _DEFAULT_MAX_READ_BYTES,
    ) -> None:
        if (
            type(plan) is not StreamingConversionPlanV1
            or type(manifest) is not PrefixCheckpointManifestV1
        ):
            raise PrefixCheckpointError("reader requires exact plan and manifest types")
        if type(maximum_read_bytes) is not int or maximum_read_bytes <= 0:
            raise PrefixCheckpointError("maximum_read_bytes must be a positive integer")
        if store.distributed_identity_qualified is not False:
            raise PrefixCheckpointError(
                "local prefix reader cannot use store_id as a distributed identity"
            )
        sink_id = (
            "local-prefix-chunk-sink-v1:"
            + hashlib.sha256(store.store_id.encode("utf-8")).hexdigest()
        )
        manifest.validate_for_plan(plan, sink_id=sink_id, store_id=store.store_id)
        persisted_manifest = store.probe_bounded(
            _manifest_object_id(plan.plan_id), maximum_bytes=_MAX_MANIFEST_BYTES
        )
        if (
            persisted_manifest is None
            or persisted_manifest[0] != manifest.canonical_bytes()
        ):
            raise PrefixCheckpointError(
                "manifest argument differs from the exact persisted manifest"
            )
        for destination in manifest.destinations:
            for binding in destination.chunks:
                persisted_binding = store.probe_bounded(
                    _binding_object_id(plan.plan_id, binding.chunk_id),
                    maximum_bytes=_MAX_RECEIPT_BYTES,
                )
                if (
                    persisted_binding is None
                    or persisted_binding[0] != binding.canonical_bytes()
                ):
                    raise PrefixCheckpointError(
                        "chunk binding differs from exact persisted receipt"
                    )
                store.verify(binding.physical_receipt)
        self.store = store
        self.plan = plan
        self.manifest = manifest
        self.maximum_read_bytes = maximum_read_bytes
        self._destinations = {
            item.destination_key: item for item in manifest.destinations
        }

    @classmethod
    def open(
        cls,
        store: ImmutableByteStore,
        plan: StreamingConversionPlanV1,
        *,
        maximum_read_bytes: int = _DEFAULT_MAX_READ_BYTES,
    ) -> Self:
        probed = store.probe_bounded(
            _manifest_object_id(plan.plan_id), maximum_bytes=_MAX_MANIFEST_BYTES
        )
        if probed is None:
            raise PrefixCheckpointError("prefix checkpoint manifest is missing")
        raw, _physical = probed
        try:
            manifest = PrefixCheckpointManifestV1.from_json(raw)
        except (ProtocolError, CheckpointConversionError) as error:
            raise PrefixCheckpointError(
                "prefix checkpoint manifest is malformed"
            ) from error
        return cls(
            store,
            plan,
            manifest,
            maximum_read_bytes=maximum_read_bytes,
        )

    def verify_complete_observation_evidence(self) -> ObservationSnapshotV1:
        """Require all exact source observations and their immutable snapshot.

        Opening and reading storage does not implicitly grant this stronger
        source-evidence qualification.  Callers that require complete source
        observations must invoke this explicit gate.
        """

        observations = _read_source_observations(self.plan, self.store)
        expected_shards = tuple(sorted(_source_descriptors(self.plan)))
        if tuple(item.receipt.shard for item in observations) != expected_shards:
            raise PrefixCheckpointError(
                "complete source observation evidence is unavailable"
            )
        expected = ObservationSnapshotV1(
            plan_id=self.plan.plan_id,
            source_adapter_identity=self.plan.source_adapter_identity,
            expected_source_shards=expected_shards,
            observations=observations,
        )
        probed = self.store.probe_bounded(
            _observation_snapshot_object_id(self.plan.plan_id, expected.snapshot_id),
            maximum_bytes=_MAX_RECEIPT_BYTES,
        )
        if probed is None:
            raise PrefixCheckpointError(
                "complete observation snapshot is not published"
            )
        raw, _physical = probed
        try:
            snapshot = ObservationSnapshotV1.from_json(raw)
        except (ProtocolError, CheckpointConversionError) as error:
            raise PrefixCheckpointError(
                "complete observation snapshot is malformed"
            ) from error
        if snapshot != expected:
            raise PrefixCheckpointError(
                "complete observation snapshot differs from exact evidence"
            )
        snapshot.validate_for_plan(self.plan)
        return snapshot

    def read_flat_region(
        self, destination_key: str, flat_start: int, flat_stop: int
    ) -> np.ndarray:
        require_string(destination_key, "destination_key", pattern=IDENTIFIER_PATTERN)
        if type(flat_start) is not int or type(flat_stop) is not int:
            raise PrefixCheckpointError("flat read bounds must be integers")
        destination = self._destinations.get(destination_key)
        if destination is None:
            raise PrefixCheckpointError("unknown destination key")
        numel = int(np.prod(destination.shape))
        if flat_start < 0 or flat_stop <= flat_start or flat_stop > numel:
            raise PrefixCheckpointError("flat read interval is invalid")
        dtype = _DTYPES[destination.dtype]
        output_bytes = (flat_stop - flat_start) * dtype.itemsize
        if output_bytes > self.maximum_read_bytes:
            raise PrefixCheckpointError("flat read exceeds configured byte bound")
        output = np.empty(flat_stop - flat_start, dtype=dtype)
        storage = output.view(np.uint16 if dtype.itemsize == 2 else np.uint32).view(
            np.uint8
        )
        target = memoryview(storage).cast("B")
        try:
            for binding in destination.chunks:
                receipt = binding.output_receipt
                left = max(flat_start, receipt.destination_flat_start)
                right = min(flat_stop, receipt.destination_flat_stop)
                if left >= right:
                    continue
                byte_offset = (left - receipt.destination_flat_start) * dtype.itemsize
                byte_length = (right - left) * dtype.itemsize
                raw = self.store.read_exact(
                    binding.physical_receipt,
                    offset=byte_offset,
                    length=byte_length,
                    maximum_bytes=self.maximum_read_bytes,
                )
                target_start = (left - flat_start) * dtype.itemsize
                target[target_start : target_start + byte_length] = raw
            return output
        finally:
            target.release()


__all__ = [
    "ImmutableByteStore",
    "ImmutableObjectConflict",
    "ImmutablePhysicalObjectReceiptV1",
    "ImmutablePrefixChunkSinkV1",
    "LOCAL_STORE_IDENTITY_SCOPE",
    "LocalImmutableByteStore",
    "ObservationSnapshotV1",
    "PlanSourceIdentityBindingV1",
    "PrefixCheckpointError",
    "PrefixCheckpointManifestV1",
    "PrefixCheckpointReaderV1",
    "PrefixChunkBindingV1",
    "SourceObservationEvidenceV1",
    "StoredDestinationV1",
    "UNMET_PRODUCTION_GAPS",
    "finalize_prefix_checkpoint",
]
