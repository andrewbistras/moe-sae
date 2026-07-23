"""Exact-receipt read-through cache for the JAX prefix restore.

The qualified GCS reader verifies a ranged request by hashing the complete
physical object.  That is the right authority boundary, but calling it once
per JAX shard interval would download the same object repeatedly.  This
adapter performs that complete verified read once per VM and exact receipt,
publishes the bytes into a pinned local directory, then serves later ranges
with ``pread``.

There is no object listing, discovery, eviction, or cross-region copy here.
The cache is disposable acceleration; the GCS receipt remains authority.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import stat
import threading
from typing import Any

from .gcs_immutable_byte_store import GCSPhysicalObjectReceiptV1


DEFAULT_MAXIMUM_CACHED_OBJECT_BYTES = 64 * 1024**2
_IO_BYTES = 8 * 1024**2


class GCSReadCacheError(RuntimeError):
    """The local cache or its authoritative source violated the read contract."""


class ExactReceiptReadCache:
    """Implement the generation-pinned reader protocol with one remote fill."""

    def __init__(
        self,
        source: Any,
        root: str | os.PathLike[str],
        *,
        maximum_cached_object_bytes: int = DEFAULT_MAXIMUM_CACHED_OBJECT_BYTES,
        maximum_parallel_fills: int = 8,
    ) -> None:
        if not all(
            hasattr(source, name)
            for name in (
                "store_id",
                "region",
                "distributed_identity_qualified",
                "read_exact",
            )
        ):
            raise GCSReadCacheError("source does not implement the exact reader")
        if (
            type(maximum_cached_object_bytes) is not int
            or not 1 <= maximum_cached_object_bytes <= DEFAULT_MAXIMUM_CACHED_OBJECT_BYTES
        ):
            raise GCSReadCacheError("maximum cached object bound is invalid")
        if (
            type(maximum_parallel_fills) is not int
            or not 1 <= maximum_parallel_fills <= 32
        ):
            raise GCSReadCacheError("maximum parallel fills must be in [1, 32]")
        path = Path(root).absolute()
        try:
            visible = path.lstat()
        except OSError as error:
            raise GCSReadCacheError("cache root must already exist") from error
        if stat.S_ISLNK(visible.st_mode) or not stat.S_ISDIR(visible.st_mode):
            raise GCSReadCacheError("cache root must be a non-symlink directory")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            root_fd = os.open(path, flags)
        except OSError as error:
            raise GCSReadCacheError("cannot pin cache root") from error
        opened = os.fstat(root_fd)
        if (visible.st_dev, visible.st_ino) != (opened.st_dev, opened.st_ino):
            os.close(root_fd)
            raise GCSReadCacheError("cache root changed while opening")
        self._source = source
        self._root = path
        self._root_fd = root_fd
        self._root_identity = (opened.st_dev, opened.st_ino)
        self._maximum_cached_object_bytes = maximum_cached_object_bytes
        self._lock = threading.RLock()
        self._fill_semaphore = threading.BoundedSemaphore(maximum_parallel_fills)
        self._key_locks: dict[str, threading.Lock] = {}
        self._verified: dict[str, tuple[int, ...]] = {}
        self._closed = False
        self._remote_objects = 0
        self._remote_bytes = 0
        self._local_read_calls = 0
        self._local_read_bytes = 0

    @property
    def store_id(self) -> str:
        return self._source.store_id

    @property
    def region(self) -> str:
        return self._source.region

    @property
    def distributed_identity_qualified(self) -> bool:
        return self._source.distributed_identity_qualified

    @property
    def remote_objects(self) -> int:
        with self._lock:
            return self._remote_objects

    @property
    def remote_bytes(self) -> int:
        with self._lock:
            return self._remote_bytes

    @property
    def local_read_calls(self) -> int:
        with self._lock:
            return self._local_read_calls

    @property
    def local_read_bytes(self) -> int:
        with self._lock:
            return self._local_read_bytes

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                os.close(self._root_fd)
                self._closed = True

    def __enter__(self) -> ExactReceiptReadCache:
        if self._closed:
            raise GCSReadCacheError("cache is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _check_root(self) -> None:
        if self._closed:
            raise GCSReadCacheError("cache is closed")
        try:
            visible = self._root.lstat()
            opened = os.fstat(self._root_fd)
        except OSError as error:
            raise GCSReadCacheError("cache root is unavailable") from error
        identity = (opened.st_dev, opened.st_ino)
        if (
            stat.S_ISLNK(visible.st_mode)
            or not stat.S_ISDIR(visible.st_mode)
            or identity != self._root_identity
            or (visible.st_dev, visible.st_ino) != identity
        ):
            raise GCSReadCacheError("cache root was substituted")

    @staticmethod
    def _file_state(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    @staticmethod
    def _require_regular(value: os.stat_result, size: int) -> tuple[int, ...]:
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_nlink != 1
            or stat.S_IMODE(value.st_mode) != 0o600
            or value.st_size != size
        ):
            raise GCSReadCacheError("cached object has an unsafe file identity")
        return ExactReceiptReadCache._file_state(value)

    def _open_parent(self, key: str, *, create: bool) -> int:
        component = key[:2]
        if create:
            try:
                os.mkdir(component, mode=0o700, dir_fd=self._root_fd)
                os.fsync(self._root_fd)
            except FileExistsError:
                pass
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            parent = os.open(component, flags, dir_fd=self._root_fd)
        except OSError as error:
            raise GCSReadCacheError("cannot open cache shard directory") from error
        if not stat.S_ISDIR(os.fstat(parent).st_mode):
            os.close(parent)
            raise GCSReadCacheError("cache shard path is not a directory")
        return parent

    @staticmethod
    def _open_optional(parent: int, name: str) -> int | None:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            return os.open(name, flags, dir_fd=parent)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise GCSReadCacheError("cannot safely open cached object") from error

    @staticmethod
    def _hash_fd(fd: int, size: int) -> str:
        digest = hashlib.sha256()
        cursor = 0
        while cursor < size:
            request = min(_IO_BYTES, size - cursor)
            data = os.pread(fd, request, cursor)
            if len(data) != request:
                raise GCSReadCacheError("cached object read was short")
            digest.update(data)
            cursor += request
        return digest.hexdigest()

    def _verify_open(
        self,
        fd: int,
        *,
        key: str,
        receipt: GCSPhysicalObjectReceiptV1,
    ) -> tuple[int, ...]:
        before = self._require_regular(os.fstat(fd), receipt.size_bytes)
        with self._lock:
            known = self._verified.get(key)
        if known is None:
            if self._hash_fd(fd, receipt.size_bytes) != receipt.sha256:
                raise GCSReadCacheError("cached object SHA-256 differs from receipt")
            after = self._require_regular(os.fstat(fd), receipt.size_bytes)
            if before != after:
                raise GCSReadCacheError("cached object changed while hashing")
            with self._lock:
                self._verified[key] = after
            return after
        if before != known:
            raise GCSReadCacheError("verified cached object identity changed")
        return before

    def _fill(
        self,
        parent: int,
        name: str,
        key: str,
        receipt: GCSPhysicalObjectReceiptV1,
    ) -> int:
        partial = f".{name}.partial"
        try:
            os.unlink(partial, dir_fd=parent)
        except FileNotFoundError:
            pass
        with self._fill_semaphore:
            payload = self._source.read_exact(
                receipt,
                offset=0,
                length=receipt.size_bytes,
                maximum_bytes=receipt.size_bytes,
            )
        if (
            type(payload) is not bytes
            or len(payload) != receipt.size_bytes
            or hashlib.sha256(payload).hexdigest() != receipt.sha256
        ):
            raise GCSReadCacheError("source returned bytes outside the exact receipt")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        fd = -1
        try:
            fd = os.open(partial, flags, 0o600, dir_fd=parent)
            view = memoryview(payload)
            cursor = 0
            while cursor < len(view):
                wrote = os.write(fd, view[cursor : cursor + _IO_BYTES])
                if wrote <= 0:
                    raise GCSReadCacheError("cached object write was short")
                cursor += wrote
            view.release()
            os.fsync(fd)
            os.close(fd)
            fd = -1
            try:
                os.link(
                    partial,
                    name,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileExistsError:
                pass
            os.unlink(partial, dir_fd=parent)
            os.fsync(parent)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(partial, dir_fd=parent)
            except FileNotFoundError:
                pass
        result = self._open_optional(parent, name)
        if result is None:  # pragma: no cover - link/fsync invariant
            raise GCSReadCacheError("published cached object vanished")
        with self._lock:
            self._remote_objects += 1
            self._remote_bytes += receipt.size_bytes
            self._verified.pop(key, None)
        return result

    def read_exact(
        self,
        receipt: GCSPhysicalObjectReceiptV1,
        *,
        offset: int,
        length: int,
        maximum_bytes: int,
    ) -> bytes:
        if type(receipt) is not GCSPhysicalObjectReceiptV1:
            raise GCSReadCacheError("receipt has the wrong type")
        if receipt.store_id != self.store_id or receipt.region != self.region:
            raise GCSReadCacheError("receipt belongs to another source")
        if (
            receipt.size_bytes > self._maximum_cached_object_bytes
            or type(offset) is not int
            or offset < 0
            or type(length) is not int
            or length <= 0
            or type(maximum_bytes) is not int
            or maximum_bytes <= 0
            or length > maximum_bytes
            or offset + length > receipt.size_bytes
        ):
            raise GCSReadCacheError("cached range request is invalid")
        key = receipt.digest
        name = f"{key}.bin"
        with self._lock:
            self._check_root()
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            parent = -1
            fd = -1
            lock_fd = -1
            try:
                self._check_root()
                parent = self._open_parent(key, create=True)
                lock_name = f".{name}.lock"
                lock_fd = os.open(
                    lock_name,
                    os.O_RDWR
                    | os.O_CREAT
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=parent,
                )
                lock_state = os.fstat(lock_fd)
                if (
                    not stat.S_ISREG(lock_state.st_mode)
                    or lock_state.st_nlink != 1
                    or stat.S_IMODE(lock_state.st_mode) != 0o600
                ):
                    raise GCSReadCacheError("cache lock has an unsafe identity")
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                fd = self._open_optional(parent, name)
                if fd is None:
                    fd = self._fill(parent, name, key, receipt)
                state = self._verify_open(fd, key=key, receipt=receipt)
                result = os.pread(fd, length, offset)
                if len(result) != length:
                    raise GCSReadCacheError("cached range read was short")
                if self._require_regular(os.fstat(fd), receipt.size_bytes) != state:
                    raise GCSReadCacheError("cached object changed during range read")
                with self._lock:
                    self._local_read_calls += 1
                    self._local_read_bytes += length
                return result
            finally:
                if fd >= 0:
                    os.close(fd)
                if lock_fd >= 0:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                if parent >= 0:
                    os.close(parent)


__all__ = [
    "DEFAULT_MAXIMUM_CACHED_OBJECT_BYTES",
    "ExactReceiptReadCache",
    "GCSReadCacheError",
]
