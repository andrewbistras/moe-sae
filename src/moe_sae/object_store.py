"""Immutable GCS objects and a bounded, hash-verifying local shard cache."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import fcntl
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Any, Protocol
import uuid


_SHA256 = re.compile(r"[0-9a-f]{64}")


class ObjectStoreError(RuntimeError):
    """An immutable object could not be published or materialized safely."""


@dataclass(frozen=True, slots=True)
class ObjectIdentity:
    """Immutable identity recorded in a capture/checkpoint manifest."""

    uri: str
    generation: int
    size_bytes: int
    crc32c: str
    sha256: str

    def __post_init__(self) -> None:
        try:
            parse_gcs_uri(self.uri)
        except ValueError as exc:
            raise ObjectStoreError("object identity has an invalid URI") from exc
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation <= 0
            or isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
            or not re.fullmatch(r"[A-Za-z0-9+/]{6}==", self.crc32c)
            or not _SHA256.fullmatch(self.sha256)
        ):
            raise ObjectStoreError("object identity contains invalid values")

    def as_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "generation": self.generation,
            "size_bytes": self.size_bytes,
            "crc32c": self.crc32c,
            "sha256": self.sha256,
        }


class ImmutableObjectStore(Protocol):
    def put_if_absent(
        self, source: Path, uri: str, *, sha256: str
    ) -> ObjectIdentity: ...

    def download(self, identity: ObjectIdentity, destination: Path) -> None: ...

    def verify(self, identity: ObjectIdentity) -> None: ...

    def delete(self, identity: ObjectIdentity) -> None: ...


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("object URI must start with gs://")
    remainder = uri[5:]
    bucket, separator, name = remainder.partition("/")
    if not separator or not bucket or not name or name.endswith("/"):
        raise ValueError(f"object URI must name one GCS object, got {uri!r}")
    if "//" in name or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError(f"unsafe GCS object name in {uri!r}")
    return bucket, name


def join_gcs_uri(prefix: str, *parts: str) -> str:
    if not prefix.startswith("gs://"):
        raise ValueError("object prefix must start with gs://")
    remainder = prefix[5:].rstrip("/")
    bucket, separator, base = remainder.partition("/")
    if not bucket:
        raise ValueError("object prefix must include a bucket")
    if not separator:
        base = ""
    elif any(part in {"", ".", ".."} for part in base.split("/")):
        raise ValueError(f"unsafe GCS object prefix {prefix!r}")
    clean: list[str] = []
    for part in parts:
        if not part or part in {".", ".."} or "/" in part:
            raise ValueError(f"unsafe object path component {part!r}")
        clean.append(part)
    suffix = "/".join(clean)
    name = f"{base}/{suffix}" if base else suffix
    return f"gs://{bucket}/{name}"


def object_identity_from_mapping(value: Any) -> ObjectIdentity:
    if not isinstance(value, dict) or set(value) != {
        "uri",
        "generation",
        "size_bytes",
        "crc32c",
        "sha256",
    }:
        raise ObjectStoreError("object identity has incompatible fields")
    uri = str(value.get("uri", ""))
    parse_gcs_uri(uri)
    generation = value.get("generation")
    size = value.get("size_bytes")
    digest = str(value.get("sha256", ""))
    crc32c = str(value.get("crc32c", ""))
    return ObjectIdentity(uri, generation, size, crc32c, digest)


class GCSObjectStore:
    """Publish and fetch generation-pinned Google Cloud Storage objects."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        parallel_upload_threshold_bytes: int = 32 * 1024**2,
        parallel_upload_chunk_bytes: int = 8 * 1024**2,
        parallel_upload_workers: int = 8,
        transfer_manager_module: Any | None = None,
    ):
        if client is None:
            try:
                from google.cloud import storage
            except ImportError as error:
                raise ObjectStoreError(
                    "GCS object storage requires google-cloud-storage (`--extra tpu`)"
                ) from error
            client = storage.Client()
        self.client = client
        self.parallel_upload_threshold_bytes = int(
            parallel_upload_threshold_bytes
        )
        self.parallel_upload_chunk_bytes = int(parallel_upload_chunk_bytes)
        self.parallel_upload_workers = int(parallel_upload_workers)
        self._transfer_manager_module = transfer_manager_module
        if (
            self.parallel_upload_threshold_bytes <= 0
            or self.parallel_upload_chunk_bytes <= 0
            or self.parallel_upload_workers <= 0
        ):
            raise ValueError("parallel GCS upload settings must be positive")

    def _blob(self, uri: str) -> Any:
        bucket, name = parse_gcs_uri(uri)
        return self.client.bucket(bucket).blob(name)

    @staticmethod
    def _identity(blob: Any, uri: str, expected_sha256: str) -> ObjectIdentity:
        blob.reload()
        metadata = dict(blob.metadata or {})
        generation = int(blob.generation or 0)
        size = int(blob.size or 0)
        crc32c = str(blob.crc32c or "")
        if (
            generation <= 0
            or size <= 0
            or metadata.get("sha256") != expected_sha256
            or not re.fullmatch(r"[A-Za-z0-9+/]{6}==", crc32c)
        ):
            raise ObjectStoreError(
                f"existing GCS object lacks the expected immutable identity: {uri}"
            )
        return ObjectIdentity(uri, generation, size, crc32c, expected_sha256)

    def put_if_absent(
        self, source: Path, uri: str, *, sha256: str
    ) -> ObjectIdentity:
        if not _SHA256.fullmatch(sha256):
            raise ValueError("sha256 must be lowercase 64-character hex")
        actual_sha, size = sha256_file(source)
        if actual_sha != sha256 or size <= 0:
            raise ObjectStoreError("local upload source does not match its claimed SHA-256")
        blob = self._blob(uri)
        # Resume avoids retransmitting a content-addressed object that already
        # has the exact expected metadata and size.
        try:
            existing = self._identity(blob, uri, sha256)
        except Exception:
            existing = None
        if existing is not None:
            if existing.size_bytes != size:
                raise ObjectStoreError(f"existing GCS object size differs: {uri}")
            return existing
        blob.metadata = {"sha256": sha256}
        if size >= self.parallel_upload_threshold_bytes:
            identity = self._put_parallel_conditional(
                source, blob, uri=uri, sha256=sha256
            )
            if identity.size_bytes != size:
                raise ObjectStoreError(f"GCS object size differs after upload: {uri}")
            return identity
        try:
            blob.upload_from_filename(
                str(source),
                if_generation_match=0,
                checksum="crc32c",
            )
        except Exception as error:
            # A racing/replayed content-addressed upload is successful only if
            # the pre-existing object advertises the exact expected identity.
            try:
                identity = self._identity(blob, uri, sha256)
            except Exception:
                raise ObjectStoreError(f"cannot publish immutable GCS object {uri}") from error
        else:
            identity = self._identity(blob, uri, sha256)
        if identity.size_bytes != size:
            raise ObjectStoreError(f"GCS object size differs after upload: {uri}")
        return identity

    def _put_parallel_conditional(
        self, source: Path, final_blob: Any, *, uri: str, sha256: str
    ) -> ObjectIdentity:
        """Multipart-upload to a unique temporary, then conditionally copy.

        GCS XML multipart uploads do not support create-only preconditions.
        The temporary name is therefore unpredictable, and the server-side
        copy into the content-addressed final name is the sole commit point
        with ``if_generation_match=0``.  Threads are deliberate: forking a
        process pool after JAX initialized a TPU runtime is unsafe.
        """

        transfer_manager = self._transfer_manager_module
        if transfer_manager is None:
            try:
                from google.cloud.storage import transfer_manager
            except ImportError as error:  # pragma: no cover - same optional package
                raise ObjectStoreError("parallel GCS uploader is unavailable") from error
        temporary = final_blob.bucket.blob(
            f"{final_blob.name}.multipart-{uuid.uuid4().hex}"
        )
        temporary.metadata = {"sha256": sha256}
        temporary_generation = 0
        try:
            transfer_manager.upload_chunks_concurrently(
                str(source),
                temporary,
                chunk_size=self.parallel_upload_chunk_bytes,
                worker_type=transfer_manager.THREAD,
                max_workers=self.parallel_upload_workers,
                checksum="crc32c",
            )
            temporary.reload()
            temporary_generation = int(temporary.generation or 0)
            if temporary_generation <= 0:
                raise ObjectStoreError("multipart upload has no GCS generation")
            try:
                final_blob.bucket.copy_blob(
                    temporary,
                    final_blob.bucket,
                    new_name=final_blob.name,
                    source_generation=temporary_generation,
                    if_generation_match=0,
                    if_source_generation_match=temporary_generation,
                )
            except Exception as error:
                # A racing writer is successful only if its final immutable
                # identity is byte-for-byte compatible with ours.
                try:
                    return self._identity(final_blob, uri, sha256)
                except Exception:
                    raise ObjectStoreError(
                        f"cannot conditionally commit multipart object {uri}"
                    ) from error
            return self._identity(final_blob, uri, sha256)
        finally:
            if temporary_generation > 0:
                try:
                    temporary.delete(if_generation_match=temporary_generation)
                except Exception as error:
                    raise ObjectStoreError(
                        f"cannot remove multipart temporary for {uri}"
                    ) from error

    def download(self, identity: ObjectIdentity, destination: Path) -> None:
        blob = self._blob(identity.uri)
        blob.reload(if_generation_match=identity.generation)
        if (
            int(blob.generation or 0) != identity.generation
            or int(blob.size or 0) != identity.size_bytes
            or str(blob.crc32c or "") != identity.crc32c
            or dict(blob.metadata or {}).get("sha256") != identity.sha256
        ):
            raise ObjectStoreError(
                f"GCS object generation/metadata changed: {identity.uri}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(
            str(destination),
            if_generation_match=identity.generation,
            checksum="auto",
        )

    def verify(self, identity: ObjectIdentity) -> None:
        blob = self._blob(identity.uri)
        blob.reload(if_generation_match=identity.generation)
        if (
            int(blob.generation or 0) != identity.generation
            or int(blob.size or 0) != identity.size_bytes
            or str(blob.crc32c or "") != identity.crc32c
            or dict(blob.metadata or {}).get("sha256") != identity.sha256
        ):
            raise ObjectStoreError(f"immutable GCS object identity changed: {identity.uri}")

    def delete(self, identity: ObjectIdentity) -> None:
        """Delete only the exact probe generation named by ``identity``."""

        self.verify(identity)
        blob = self._blob(identity.uri)
        try:
            blob.delete(if_generation_match=identity.generation)
        except Exception as exc:
            raise ObjectStoreError(
                f"cannot delete generation-pinned GCS object {identity.uri}"
            ) from exc


class LocalObjectStore:
    """Filesystem-backed immutable object store used by deterministic tests."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, uri: str) -> Path:
        bucket, name = parse_gcs_uri(uri)
        return self.root / bucket / name

    @staticmethod
    def _crc32c(path: Path) -> str:
        checksum = 0xFFFFFFFF
        polynomial = 0x82F63B78
        table = []
        for index in range(256):
            value = index
            for _ in range(8):
                value = (value >> 1) ^ (polynomial if value & 1 else 0)
            table.append(value)
        with path.open("rb") as handle:
            while chunk := handle.read(8 << 20):
                for byte in chunk:
                    checksum = table[(checksum ^ byte) & 0xFF] ^ (checksum >> 8)
        encoded = ((checksum ^ 0xFFFFFFFF) & 0xFFFFFFFF).to_bytes(4, "big")
        return base64.b64encode(encoded).decode("ascii")

    def _identity(self, uri: str, path: Path, sha256: str) -> ObjectIdentity:
        return ObjectIdentity(
            uri=uri,
            generation=max(1, int(path.stat().st_ino)),
            size_bytes=int(path.stat().st_size),
            crc32c=self._crc32c(path),
            sha256=sha256,
        )

    def _require_identity(self, identity: ObjectIdentity) -> Path:
        path = self._path(identity.uri)
        try:
            actual_sha, actual_size = sha256_file(path)
            actual = self._identity(identity.uri, path, actual_sha)
        except OSError as exc:
            raise ObjectStoreError(
                f"immutable local object is unavailable: {identity.uri}"
            ) from exc
        if actual != identity or actual_size != identity.size_bytes:
            raise ObjectStoreError(f"immutable local object changed: {identity.uri}")
        return path

    def put_if_absent(
        self, source: Path, uri: str, *, sha256: str
    ) -> ObjectIdentity:
        actual, size = sha256_file(source)
        if actual != sha256 or size <= 0:
            raise ObjectStoreError("local object source hash mismatch")
        path = self._path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", dir=path.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                shutil.copyfile(source, temporary)
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                try:
                    # A hard link publishes the fully written temporary file
                    # only if the immutable name is still absent.  Unlike
                    # os.replace, it can never overwrite a racing writer.
                    os.link(temporary, path)
                except FileExistsError:
                    pass
                else:
                    _fsync_directory(path.parent)
            finally:
                temporary.unlink(missing_ok=True)
        existing, existing_size = sha256_file(path)
        if (existing, existing_size) != (sha256, size):
            raise ObjectStoreError(f"immutable local object collision: {uri}")
        identity = self._identity(uri, path, sha256)
        if identity.size_bytes != size:
            raise ObjectStoreError(f"immutable local object size changed: {uri}")
        return identity

    def download(self, identity: ObjectIdentity, destination: Path) -> None:
        source = self._require_identity(identity)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def verify(self, identity: ObjectIdentity) -> None:
        self._require_identity(identity)

    def delete(self, identity: ObjectIdentity) -> None:
        self._require_identity(identity).unlink()


class ShardCache:
    """Size-bounded local cache which verifies every object before use."""

    def __init__(
        self,
        root: str | Path,
        store: ImmutableObjectStore,
        *,
        max_bytes: int,
    ):
        if max_bytes <= 0:
            raise ValueError("cache max_bytes must be positive")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._lock_path = self.root / ".cache.lock"

    def _process_lock(self) -> Any:
        """Return a context manager serializing cache mutation across processes."""

        class _Lock:
            def __init__(self, path: Path):
                self.path = path
                self.handle: Any | None = None

            def __enter__(self) -> None:
                self.handle = self.path.open("a+b", buffering=0)
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)

            def __exit__(self, *_args: Any) -> None:
                assert self.handle is not None
                try:
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
                finally:
                    self.handle.close()

        return _Lock(self._lock_path)

    def _evict_for(self, required_bytes: int, keep: Path) -> None:
        if required_bytes > self.max_bytes:
            raise ObjectStoreError(
                f"one shard ({required_bytes} bytes) exceeds the local cache bound"
            )
        candidates = [
            path
            for path in self.root.glob("*.safetensors")
            if path.is_file() and path != keep
        ]
        total = sum(path.stat().st_size for path in candidates)
        if keep.is_file():
            total += keep.stat().st_size
        for path in sorted(candidates, key=lambda item: item.stat().st_mtime_ns):
            if total + required_bytes <= self.max_bytes:
                break
            size = path.stat().st_size
            path.unlink()
            total -= size
        if total + required_bytes > self.max_bytes:
            raise ObjectStoreError("local cache cannot free enough bounded capacity")
        free = shutil.disk_usage(self.root).free
        if free < required_bytes:
            raise ObjectStoreError(
                f"local cache has only {free} free bytes; need {required_bytes}"
            )

    def materialize(self, value: Any) -> Path:
        identity = (
            value if isinstance(value, ObjectIdentity) else object_identity_from_mapping(value)
        )
        target = self.root / f"{identity.sha256}.safetensors"
        with self._lock:
            with self._process_lock():
                if target.is_file():
                    digest, size = sha256_file(target)
                    if digest == identity.sha256 and size == identity.size_bytes:
                        os.utime(target, None)
                        return target
                    target.unlink()
                self._evict_for(identity.size_bytes, target)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{identity.sha256}.", suffix=".partial", dir=self.root
                )
                os.close(descriptor)
                temporary = Path(temporary_name)
                try:
                    self.store.download(identity, temporary)
                    digest, size = sha256_file(temporary)
                    if digest != identity.sha256 or size != identity.size_bytes:
                        raise ObjectStoreError(
                            "downloaded object failed SHA-256/size verification: "
                            f"{identity.uri}"
                        )
                    with temporary.open("rb") as handle:
                        os.fsync(handle.fileno())
                    os.replace(temporary, target)
                    _fsync_directory(self.root)
                finally:
                    temporary.unlink(missing_ok=True)
                return target


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            # Cache contents are always recoverable from their immutable
            # object identity; directory fsync is a durability optimization.
            pass
    finally:
        os.close(descriptor)


__all__ = [
    "GCSObjectStore",
    "ImmutableObjectStore",
    "LocalObjectStore",
    "ObjectIdentity",
    "ObjectStoreError",
    "ShardCache",
    "join_gcs_uri",
    "object_identity_from_mapping",
    "parse_gcs_uri",
    "sha256_file",
]
