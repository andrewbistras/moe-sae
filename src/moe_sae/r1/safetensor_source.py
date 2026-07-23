"""Bounded local safetensor reads behind the conversion source protocol.

This adapter trusts a controller to have pre-staged an exact directory of
checkpoint shards on the local machine.  It authenticates each complete blob
against the already-authenticated inventory before exposing tensor bytes, but
it does not establish how those bytes reached the machine.  Consequently its
receipts are local conversion evidence only: they are never a production
checkpoint seal or a substitute for authenticated object-store provenance.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import reduce
import hashlib
import json
import operator
import os
from pathlib import Path
import stat
import struct
import threading
from types import TracebackType
from typing import Any, Final

import ml_dtypes
import numpy as np

from .checkpoint_conversion import (
    CheckpointConversionError,
    SourceAdapterIdentityV1,
    TensorReadRegionV1,
    VerifiedSourceShardReceiptV1,
)
from .checkpoint_inventory import (
    BF16_DTYPE,
    F32_DTYPE,
    FP8_DTYPE,
    MAX_RAW_HEADER_BYTES,
    CheckpointInventoryError,
    CheckpointInventorySelectionV1,
    ShardSourceDescriptorV1,
    SourceTensorEntryV1,
    decode_safetensor_header_bytes,
)


LOCAL_SOURCE_ADAPTER_ID: Final = "local-prestaged-verified-safetensor-source-v1"
LOCAL_SOURCE_TRUST_BOUNDARY: Final = "trusted-controller-prestaged-directory-only-v1"

_DEFAULT_MAX_BLOB_BYTES = 6 * 1024**3
_DEFAULT_MAX_REGION_BYTES = 512 * 1024**2
_DEFAULT_HASH_CHUNK_BYTES = 8 * 1024**2
_DEFAULT_IO_CHUNK_BYTES = 8 * 1024**2
_MAX_STREAM_CHUNK_BYTES = 64 * 1024**2

_SOURCE_DTYPES = {
    FP8_DTYPE: np.dtype(ml_dtypes.float8_e4m3fn),
    BF16_DTYPE: np.dtype(ml_dtypes.bfloat16),
    F32_DTYPE: np.dtype("<f4"),
}


def _implementation_sha256() -> str:
    """Measure the exact installed adapter implementation."""

    try:
        raw = Path(__file__).read_bytes()
    except OSError as error:  # pragma: no cover - broken installation
        raise CheckpointConversionError(
            "cannot hash the installed local safetensor source adapter"
        ) from error
    return hashlib.sha256(raw).hexdigest()


def local_safetensor_source_identity() -> SourceAdapterIdentityV1:
    """Return the implementation-derived identity bound into conversion plans."""

    return SourceAdapterIdentityV1(
        adapter_id=LOCAL_SOURCE_ADAPTER_ID,
        implementation_sha256=_implementation_sha256(),
    )


def _bounded_integer(
    value: int,
    name: str,
    *,
    minimum: int = 1,
    maximum: int,
) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise CheckpointConversionError(
            f"{name} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:  # pragma: no cover - inventory invariant
        raise CheckpointConversionError(
            "authenticated safetensor header is not canonical JSON"
        ) from error


@dataclass(frozen=True, slots=True)
class _FileState:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileState:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            mode=value.st_mode,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
        )


def _require_regular_file(
    value: os.stat_result,
    *,
    shard: str,
) -> _FileState:
    if not stat.S_ISREG(value.st_mode):
        raise CheckpointConversionError(
            f"local shard must be a non-symlink regular file: {shard}"
        )
    return _FileState.from_stat(value)


def _product(values: tuple[int, ...]) -> int:
    return reduce(operator.mul, values, 1)


class LocalVerifiedSafetensorSource:
    """One-reader-at-a-time source for controller-prestaged local shards.

    The complete authenticated inventory is required so that the raw header is
    checked against every entry in each opened shard, including entries that
    are not selected for the block-31 prefix.  Files are pinned by directory
    descriptor and inode before conversion begins.  This class deliberately
    has no production-sealing capability.

    Payload access is correctness-first ``pread``/``preadv`` rather than
    ``mmap``: concurrent truncation of a mapped multi-gigabyte shard can raise
    ``SIGBUS`` instead of a catchable failure.  A column slice of a rank-two
    tensor can therefore issue one bounded payload syscall per source row.
    ``payload_read_syscalls`` makes that cost observable and it must be
    benchmark-qualified before the roughly 326 GB prefix conversion.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        selection: CheckpointInventorySelectionV1,
        *,
        max_blob_bytes: int = _DEFAULT_MAX_BLOB_BYTES,
        max_region_bytes: int = _DEFAULT_MAX_REGION_BYTES,
        hash_chunk_bytes: int = _DEFAULT_HASH_CHUNK_BYTES,
        io_chunk_bytes: int = _DEFAULT_IO_CHUNK_BYTES,
        max_header_bytes: int = MAX_RAW_HEADER_BYTES,
    ) -> None:
        if type(selection) is not CheckpointInventorySelectionV1:
            raise CheckpointConversionError(
                "local safetensor source requires an exact checkpoint selection"
            )
        if not selection.metadata_authenticated:
            raise CheckpointConversionError(
                "local safetensor source requires authenticated inventory metadata"
            )
        self._max_blob_bytes = _bounded_integer(
            max_blob_bytes,
            "max_blob_bytes",
            maximum=64 * 1024**3,
        )
        self._max_region_bytes = _bounded_integer(
            max_region_bytes,
            "max_region_bytes",
            maximum=8 * 1024**3,
        )
        self._hash_chunk_bytes = _bounded_integer(
            hash_chunk_bytes,
            "hash_chunk_bytes",
            maximum=_MAX_STREAM_CHUNK_BYTES,
        )
        self._io_chunk_bytes = _bounded_integer(
            io_chunk_bytes,
            "io_chunk_bytes",
            maximum=_MAX_STREAM_CHUNK_BYTES,
        )
        self._max_header_bytes = _bounded_integer(
            max_header_bytes,
            "max_header_bytes",
            maximum=MAX_RAW_HEADER_BYTES,
        )
        self._identity = local_safetensor_source_identity()
        self._state_lock = threading.Lock()
        self._active_reader = False
        self._closed = False
        self._payload_read_syscalls = 0

        # Reconstruct the frozen records so an in-process caller cannot mutate
        # the selection objects by bypassing ``frozen=True`` after this trust
        # boundary has measured them.
        descriptors = tuple(
            ShardSourceDescriptorV1(
                shard=item.shard,
                expected_lfs_sha256=item.expected_lfs_sha256,
                expected_blob_size_bytes=item.expected_blob_size_bytes,
                raw_header_sha256=item.raw_header_sha256,
                header_length_bytes=item.header_length_bytes,
                decoded_header_sha256=item.decoded_header_sha256,
                decoder_contract=item.decoder_contract,
            )
            for item in selection.selected.shard_descriptors
        )
        descriptor_names = tuple(item.shard for item in descriptors)
        if not descriptors:
            raise CheckpointConversionError(
                "local safetensor source has no authenticated shard descriptors"
            )
        if len({name.casefold() for name in descriptor_names}) != len(descriptor_names):
            raise CheckpointConversionError(
                "local shard descriptor basenames collide case-insensitively"
            )
        for name in descriptor_names:
            if Path(name).name != name or os.path.basename(name) != name:
                raise CheckpointConversionError(
                    f"local shard descriptor is not an exact basename: {name}"
                )
        self._descriptors = {item.shard: item for item in descriptors}

        entries_by_shard: defaultdict[str, list[SourceTensorEntryV1]] = defaultdict(
            list
        )
        for raw_entry in selection.full.entries:
            entry = SourceTensorEntryV1(
                key=raw_entry.key,
                shard=raw_entry.shard,
                dtype=raw_entry.dtype,
                shape=tuple(raw_entry.shape),
                data_offsets=tuple(raw_entry.data_offsets),
            )
            if entry.shard in self._descriptors:
                entries_by_shard[entry.shard].append(entry)
        metadata = dict(selection.full.header_metadata)
        self._entries: dict[str, dict[str, SourceTensorEntryV1]] = {}
        self._expected_headers: dict[str, Mapping[str, Any]] = {}
        for descriptor in descriptors:
            shard_entries = sorted(
                entries_by_shard[descriptor.shard], key=lambda item: item.key
            )
            if not shard_entries:
                raise CheckpointConversionError(
                    f"authenticated inventory has no entries for {descriptor.shard}"
                )
            by_key = {entry.key: entry for entry in shard_entries}
            if len(by_key) != len(
                shard_entries
            ):  # pragma: no cover - inventory invariant
                raise CheckpointConversionError(
                    f"authenticated inventory repeats entries in {descriptor.shard}"
                )
            expected_header: dict[str, Any] = {}
            header_metadata = metadata.get(descriptor.shard)
            if header_metadata is None:
                raise CheckpointConversionError(
                    f"authenticated inventory is missing metadata for {descriptor.shard}"
                )
            if header_metadata:
                expected_header["__metadata__"] = dict(header_metadata)
            for entry in shard_entries:
                expected_header[entry.key] = {
                    "dtype": entry.dtype,
                    "shape": list(entry.shape),
                    "data_offsets": list(entry.data_offsets),
                }
            self._entries[descriptor.shard] = by_key
            self._expected_headers[descriptor.shard] = expected_header

        root_path = os.path.abspath(os.fspath(root))
        try:
            root_lstat = os.lstat(root_path)
        except OSError as error:
            raise CheckpointConversionError(
                "local safetensor source directory is unavailable"
            ) from error
        if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
            raise CheckpointConversionError(
                "local safetensor source root must be a non-symlink directory"
            )
        root_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        root_flags |= getattr(os, "O_DIRECTORY", 0)
        root_flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            root_fd = os.open(root_path, root_flags)
        except OSError as error:
            raise CheckpointConversionError(
                "cannot securely open local safetensor source directory"
            ) from error
        try:
            opened_root = os.fstat(root_fd)
            if (
                opened_root.st_dev != root_lstat.st_dev
                or opened_root.st_ino != root_lstat.st_ino
                or not stat.S_ISDIR(opened_root.st_mode)
            ):
                raise CheckpointConversionError(
                    "local safetensor source directory changed while opening"
                )
            baseline: dict[str, _FileState] = {}
            identities: set[tuple[int, int]] = set()
            for descriptor in descriptors:
                state = self._stat_at(root_fd, root_path, descriptor.shard)
                baseline_state = _require_regular_file(state, shard=descriptor.shard)
                if baseline_state.size != descriptor.expected_blob_size_bytes:
                    raise CheckpointConversionError(
                        f"local shard size mismatch for {descriptor.shard}"
                    )
                if baseline_state.size > self._max_blob_bytes:
                    raise CheckpointConversionError(
                        f"local shard exceeds max_blob_bytes: {descriptor.shard}"
                    )
                if descriptor.header_length_bytes > self._max_header_bytes:
                    raise CheckpointConversionError(
                        f"local shard header exceeds max_header_bytes: {descriptor.shard}"
                    )
                identity = (baseline_state.device, baseline_state.inode)
                if identity in identities:
                    raise CheckpointConversionError(
                        "multiple shard basenames resolve to one regular file"
                    )
                identities.add(identity)
                baseline[descriptor.shard] = baseline_state
        except BaseException:
            os.close(root_fd)
            raise
        self._root_path = root_path
        self._root_fd = root_fd
        self._baseline = baseline

    @staticmethod
    def _stat_at(root_fd: int, root_path: str, name: str) -> os.stat_result:
        try:
            if os.stat in os.supports_dir_fd and os.stat in os.supports_follow_symlinks:
                return os.stat(
                    name,
                    dir_fd=root_fd,
                    follow_symlinks=False,
                )
            return os.lstat(os.path.join(root_path, name))
        except OSError as error:
            raise CheckpointConversionError(
                f"cannot inspect local shard mapping for {name}"
            ) from error

    @property
    def adapter_identity(self) -> SourceAdapterIdentityV1:
        return self._identity

    @property
    def trust_boundary(self) -> str:
        return LOCAL_SOURCE_TRUST_BOUNDARY

    @property
    def production_sealable(self) -> bool:
        return False

    @property
    def payload_read_syscalls(self) -> int:
        """Number of payload pread/preadv calls across all opened readers."""

        with self._state_lock:
            return self._payload_read_syscalls

    def _record_payload_read_syscall(self) -> None:
        with self._state_lock:
            self._payload_read_syscalls += 1

    def open_verified_shard(
        self,
        descriptor: ShardSourceDescriptorV1,
    ) -> AbstractContextManager[_LocalVerifiedShardReader]:
        if type(descriptor) is not ShardSourceDescriptorV1:
            raise CheckpointConversionError(
                "local source requires an exact shard source descriptor"
            )
        expected = self._descriptors.get(descriptor.shard)
        if expected is None or descriptor != expected:
            raise CheckpointConversionError(
                f"local source descriptor is not authorized: {descriptor.shard}"
            )
        with self._state_lock:
            if self._closed:
                raise CheckpointConversionError("local safetensor source is closed")
        return _LocalShardContext(self, descriptor)

    def _reserve_reader(self) -> None:
        with self._state_lock:
            if self._closed:
                raise CheckpointConversionError("local safetensor source is closed")
            if self._active_reader:
                raise CheckpointConversionError(
                    "local safetensor source permits only one active reader"
                )
            self._active_reader = True

    def _release_reader(self) -> None:
        with self._state_lock:
            self._active_reader = False

    def _open_fd(self, descriptor: ShardSourceDescriptorV1) -> int:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_BINARY", 0)
        try:
            if os.open in os.supports_dir_fd:
                fd = os.open(descriptor.shard, flags, dir_fd=self._root_fd)
            else:  # pragma: no cover - Linux/macOS deployment uses openat
                fd = os.open(os.path.join(self._root_path, descriptor.shard), flags)
        except OSError as error:
            raise CheckpointConversionError(
                f"cannot securely open local shard {descriptor.shard}"
            ) from error
        try:
            state = _require_regular_file(os.fstat(fd), shard=descriptor.shard)
            if state != self._baseline[descriptor.shard]:
                raise CheckpointConversionError(
                    f"local shard changed since source initialization: {descriptor.shard}"
                )
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _build_reader(
        self,
        descriptor: ShardSourceDescriptorV1,
    ) -> _LocalVerifiedShardReader:
        fd = self._open_fd(descriptor)
        try:
            verified_state = self._baseline[descriptor.shard]
            before = _FileState.from_stat(os.fstat(fd))
            if before != verified_state:
                raise CheckpointConversionError(
                    f"local shard changed before verification: {descriptor.shard}"
                )
            observed_sha256 = self._hash_fd(
                fd,
                descriptor.expected_blob_size_bytes,
                descriptor.shard,
            )
            after_hash = _FileState.from_stat(os.fstat(fd))
            if after_hash != verified_state:
                raise CheckpointConversionError(
                    f"local shard mutated during verification: {descriptor.shard}"
                )
            if observed_sha256 != descriptor.expected_lfs_sha256:
                raise CheckpointConversionError(
                    f"local shard SHA-256 mismatch for {descriptor.shard}"
                )

            raw_prefix = self._pread_exact(fd, 8, 0, descriptor.shard, chunk_limit=8)
            header_length = struct.unpack("<Q", raw_prefix)[0]
            if header_length != descriptor.header_length_bytes:
                raise CheckpointConversionError(
                    f"safetensor header length mismatch for {descriptor.shard}"
                )
            if header_length > self._max_header_bytes:
                raise CheckpointConversionError(
                    f"safetensor header exceeds configured bound for {descriptor.shard}"
                )
            if 8 + header_length >= descriptor.expected_blob_size_bytes:
                raise CheckpointConversionError(
                    f"safetensor header leaves no payload for {descriptor.shard}"
                )
            raw_header = self._pread_exact(
                fd,
                header_length,
                8,
                descriptor.shard,
                chunk_limit=self._io_chunk_bytes,
            )
            try:
                decoded, observed_descriptor = decode_safetensor_header_bytes(
                    raw_header,
                    shard=descriptor.shard,
                    expected_lfs_sha256=descriptor.expected_lfs_sha256,
                    expected_blob_size_bytes=descriptor.expected_blob_size_bytes,
                )
            except CheckpointInventoryError as error:
                raise CheckpointConversionError(
                    f"strict safetensor header validation failed for {descriptor.shard}"
                ) from error
            if observed_descriptor != descriptor.as_dict():
                raise CheckpointConversionError(
                    f"authenticated safetensor descriptor mismatch for {descriptor.shard}"
                )
            if _canonical_bytes(decoded) != _canonical_bytes(
                self._expected_headers[descriptor.shard]
            ):
                raise CheckpointConversionError(
                    f"safetensor header/inventory mismatch for {descriptor.shard}"
                )
            after_header = _FileState.from_stat(os.fstat(fd))
            if after_header != verified_state:
                raise CheckpointConversionError(
                    f"local shard mutated during header validation: {descriptor.shard}"
                )
            receipt = VerifiedSourceShardReceiptV1(
                shard=descriptor.shard,
                observed_sha256=observed_sha256,
                observed_size_bytes=descriptor.expected_blob_size_bytes,
                source_adapter_identity=self._identity,
            )
            return _LocalVerifiedShardReader(
                owner=self,
                fd=fd,
                descriptor=descriptor,
                verified_state=verified_state,
                receipt=receipt,
                entries=self._entries[descriptor.shard],
                payload_base=8 + header_length,
            )
        except BaseException:
            os.close(fd)
            raise

    def _hash_fd(self, fd: int, size: int, shard: str) -> str:
        digest = hashlib.sha256()
        offset = 0
        while offset < size:
            request = min(self._hash_chunk_bytes, size - offset)
            try:
                chunk = os.pread(fd, request, offset)
            except OSError as error:
                raise CheckpointConversionError(
                    f"cannot hash local shard {shard}"
                ) from error
            if not chunk:
                raise CheckpointConversionError(
                    f"local shard truncated while hashing: {shard}"
                )
            digest.update(chunk)
            offset += len(chunk)
        return digest.hexdigest()

    @staticmethod
    def _pread_exact(
        fd: int,
        size: int,
        offset: int,
        shard: str,
        *,
        chunk_limit: int,
    ) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        cursor = offset
        while remaining:
            try:
                chunk = os.pread(fd, min(remaining, chunk_limit), cursor)
            except OSError as error:
                raise CheckpointConversionError(
                    f"cannot read safetensor metadata for {shard}"
                ) from error
            if not chunk:
                raise CheckpointConversionError(
                    f"local shard truncated while reading metadata: {shard}"
                )
            chunks.append(chunk)
            cursor += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            if self._active_reader:
                raise CheckpointConversionError(
                    "cannot close local safetensor source with an active reader"
                )
            self._closed = True
            fd = self._root_fd
        try:
            os.close(fd)
        except OSError as error:  # pragma: no cover - external descriptor corruption
            raise CheckpointConversionError(
                "cannot close local safetensor source directory"
            ) from error

    def __enter__(self) -> LocalVerifiedSafetensorSource:
        with self._state_lock:
            if self._closed:
                raise CheckpointConversionError("local safetensor source is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - defensive descriptor cleanup
        root_fd = getattr(self, "_root_fd", None)
        closed = getattr(self, "_closed", True)
        if root_fd is not None and not closed:
            try:
                os.close(root_fd)
            except OSError:
                pass


class _LocalShardContext(AbstractContextManager["_LocalVerifiedShardReader"]):
    def __init__(
        self,
        owner: LocalVerifiedSafetensorSource,
        descriptor: ShardSourceDescriptorV1,
    ) -> None:
        self._owner = owner
        self._descriptor = descriptor
        self._reader: _LocalVerifiedShardReader | None = None
        self._used = False
        self._reserved = False

    def __enter__(self) -> _LocalVerifiedShardReader:
        if self._used:
            raise CheckpointConversionError(
                "local shard context cannot be entered more than once"
            )
        self._used = True
        self._owner._reserve_reader()
        self._reserved = True
        try:
            self._reader = self._owner._build_reader(self._descriptor)
            return self._reader
        except BaseException:
            self._owner._release_reader()
            self._reserved = False
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._reserved:
            return
        reader = self._reader
        try:
            if reader is not None:
                reader.close()
        finally:
            self._reader = None
            self._owner._release_reader()
            self._reserved = False


class _LocalVerifiedShardReader:
    def __init__(
        self,
        *,
        owner: LocalVerifiedSafetensorSource,
        fd: int,
        descriptor: ShardSourceDescriptorV1,
        verified_state: _FileState,
        receipt: VerifiedSourceShardReceiptV1,
        entries: Mapping[str, SourceTensorEntryV1],
        payload_base: int,
    ) -> None:
        self._owner = owner
        self._fd = fd
        self._descriptor = descriptor
        self._verified_state = verified_state
        self._receipt = receipt
        self._entries = entries
        self._payload_base = payload_base
        self._closed = False
        self._payload_read_syscalls = 0

    @property
    def receipt(self) -> VerifiedSourceShardReceiptV1:
        return self._receipt

    @property
    def payload_read_syscalls(self) -> int:
        """Number of bounded payload syscalls issued by this shard reader."""

        return self._payload_read_syscalls

    def _require_unchanged(self, phase: str) -> None:
        if self._closed:
            raise CheckpointConversionError("local verified shard reader is closed")
        try:
            current = _FileState.from_stat(os.fstat(self._fd))
        except OSError as error:
            raise CheckpointConversionError(
                f"cannot inspect local shard during {phase}: {self._descriptor.shard}"
            ) from error
        if current != self._verified_state:
            raise CheckpointConversionError(
                f"local shard mutated during {phase}: {self._descriptor.shard}"
            )

    def read_region(self, region: TensorReadRegionV1) -> np.ndarray:
        if type(region) is not TensorReadRegionV1:
            raise CheckpointConversionError(
                "local source read requires an exact TensorReadRegionV1"
            )
        if region.entry.shard != self._descriptor.shard:
            raise CheckpointConversionError(
                "tensor region belongs to a different local shard"
            )
        authenticated_entry = self._entries.get(region.entry.key)
        if authenticated_entry is None or region.entry != authenticated_entry:
            raise CheckpointConversionError(
                f"tensor region entry is not authenticated: {region.entry.key}"
            )
        if region.payload_bytes > self._owner._max_region_bytes:
            raise CheckpointConversionError(
                f"tensor region exceeds max_region_bytes: {region.entry.key}"
            )
        dtype = _SOURCE_DTYPES.get(region.entry.dtype)
        if dtype is None:  # pragma: no cover - SourceTensorEntryV1 invariant
            raise CheckpointConversionError(
                f"unsupported local tensor dtype: {region.entry.dtype}"
            )
        absolute_end = self._payload_base + region.entry.data_offsets[1]
        if absolute_end > self._verified_state.size:
            raise CheckpointConversionError(
                f"authenticated tensor exceeds local shard: {region.entry.key}"
            )

        self._require_unchanged("pre-read validation")
        output = np.empty(region.shape, dtype=dtype, order="C")
        output_bytes = memoryview(output.view(np.uint8).reshape(-1))
        block_axis = len(region.entry.shape) - 1
        while (
            block_axis > 0
            and region.starts[block_axis] == 0
            and region.stops[block_axis] == region.entry.shape[block_axis]
        ):
            block_axis -= 1
        trailing_elements = _product(region.entry.shape[block_axis + 1 :])
        block_elements = (
            region.stops[block_axis] - region.starts[block_axis]
        ) * trailing_elements
        block_bytes = block_elements * dtype.itemsize
        prefix_shape = region.shape[:block_axis]
        prefixes = np.ndindex(prefix_shape) if prefix_shape else ((),)
        output_offset = 0
        for output_prefix in prefixes:
            source_linear = 0
            for axis, local_index in enumerate(output_prefix):
                source_linear = (
                    source_linear * region.entry.shape[axis]
                    + region.starts[axis]
                    + local_index
                )
            source_linear = (
                source_linear * region.entry.shape[block_axis]
                + region.starts[block_axis]
            ) * trailing_elements
            source_offset = (
                self._payload_base
                + region.entry.data_offsets[0]
                + source_linear * dtype.itemsize
            )
            self._pread_into(
                output_bytes[output_offset : output_offset + block_bytes],
                source_offset,
                region.entry.key,
            )
            output_offset += block_bytes
        if (
            output_offset != region.payload_bytes
        ):  # pragma: no cover - geometry invariant
            raise CheckpointConversionError(
                f"local tensor region byte accounting failed: {region.entry.key}"
            )
        self._require_unchanged("post-read validation")
        if (
            output.shape != region.shape
            or output.dtype != dtype
            or not output.flags.c_contiguous
            or not output.flags.owndata
        ):  # pragma: no cover - NumPy allocation invariant
            raise CheckpointConversionError(
                f"local tensor region ownership invariant failed: {region.entry.key}"
            )
        return output

    def _pread_into(
        self,
        destination: memoryview,
        offset: int,
        key: str,
    ) -> None:
        cursor = 0
        while cursor < len(destination):
            request = min(self._owner._io_chunk_bytes, len(destination) - cursor)
            target = destination[cursor : cursor + request]
            try:
                self._payload_read_syscalls += 1
                self._owner._record_payload_read_syscall()
                if hasattr(os, "preadv"):
                    received = os.preadv(self._fd, [target], offset + cursor)
                else:  # pragma: no cover - Linux/macOS provide preadv
                    chunk = os.pread(self._fd, request, offset + cursor)
                    received = len(chunk)
                    target[:received] = chunk
            except OSError as error:
                raise CheckpointConversionError(
                    f"cannot read local tensor payload for {key}"
                ) from error
            if received <= 0:
                raise CheckpointConversionError(
                    f"local shard truncated while reading tensor {key}"
                )
            cursor += received

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self._fd)
        except OSError as error:  # pragma: no cover - external descriptor corruption
            raise CheckpointConversionError(
                f"cannot close local shard {self._descriptor.shard}"
            ) from error


__all__ = [
    "LOCAL_SOURCE_ADAPTER_ID",
    "LOCAL_SOURCE_TRUST_BOUNDARY",
    "LocalVerifiedSafetensorSource",
    "local_safetensor_source_identity",
]
