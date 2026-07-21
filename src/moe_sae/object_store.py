"""Immutable GCS objects and a bounded, hash-verifying local shard cache."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import fcntl
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any, Protocol
import uuid


_SHA256 = re.compile(r"[0-9a-f]{64}")
_CRC32C = re.compile(r"[A-Za-z0-9+/]{6}==")
_ISOLATED_TRANSFER_EXIT_GRACE_SECONDS = 60.0


class ObjectStoreError(RuntimeError):
    """An immutable object could not be published or materialized safely."""


@dataclass(frozen=True, slots=True)
class _FileState:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _file_state(value: os.stat_result) -> _FileState:
    return _FileState(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        size=int(value.st_size),
        mtime_ns=int(value.st_mtime_ns),
        ctime_ns=int(value.st_ctime_ns),
    )


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    """One content digest bound to an exact immutable local file state.

    Producers that already scan or generate a file can carry this fingerprint
    into object publication.  The store still compares the device, inode,
    size, mtime, and ctime before and after transfer, while avoiding a second
    full-file digest pass on the critical upload path.
    """

    sha256: str
    crc32c: str
    size_bytes: int
    state: _FileState

    def __post_init__(self) -> None:
        if (
            not _SHA256.fullmatch(self.sha256)
            or not _CRC32C.fullmatch(self.crc32c)
            or isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
            or not isinstance(self.state, _FileState)
            or self.state.size != self.size_bytes
        ):
            raise ValueError("source fingerprint is invalid")


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
        self,
        source: Path,
        uri: str,
        *,
        sha256: str,
        source_fingerprint: SourceFingerprint | None = None,
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


def fingerprint_file(
    path: Path, chunk_bytes: int = 8 << 20
) -> SourceFingerprint:
    """Hash one stable source pass with SHA-256 and whole-file CRC32C."""

    try:
        import google_crc32c
    except ImportError as error:  # pragma: no cover - installed by GCS extra
        raise ObjectStoreError(
            "whole-object GCS verification requires google-crc32c"
        ) from error
    sha256 = hashlib.sha256()
    crc32c = google_crc32c.Checksum()
    size = 0
    with path.open("rb") as handle:
        before = _file_state(os.fstat(handle.fileno()))
        while chunk := handle.read(chunk_bytes):
            sha256.update(chunk)
            crc32c.update(chunk)
            size += len(chunk)
        after = _file_state(os.fstat(handle.fileno()))
    current = _file_state(path.stat())
    if before != after or after != current or size != current.size or size <= 0:
        raise ObjectStoreError("local upload source changed while it was hashed")
    encoded_crc32c = base64.b64encode(crc32c.digest()).decode("ascii")
    return SourceFingerprint(
        sha256=sha256.hexdigest(),
        crc32c=encoded_crc32c,
        size_bytes=size,
        state=current,
    )


def _fingerprint_from_components(
    path: Path,
    *,
    sha256: str,
    crc32c: str,
    size_bytes: int,
) -> SourceFingerprint:
    """Bind exclusive-writer digests to the file the producer finalized."""

    try:
        state = _file_state(path.stat())
    except OSError as error:
        raise ObjectStoreError("cannot stat fingerprinted upload source") from error
    try:
        fingerprint = SourceFingerprint(
            sha256=sha256,
            crc32c=crc32c,
            size_bytes=size_bytes,
            state=state,
        )
    except ValueError as error:
        raise ObjectStoreError("producer supplied an invalid source fingerprint") from error
    return fingerprint


def _resolve_source_fingerprint(
    source: Path,
    claimed_sha256: str,
    supplied: SourceFingerprint | None,
) -> SourceFingerprint:
    fingerprint = fingerprint_file(source) if supplied is None else supplied
    if fingerprint.sha256 != claimed_sha256:
        raise ObjectStoreError("local upload source does not match its claimed SHA-256")
    try:
        current = _file_state(source.stat())
    except OSError as error:
        raise ObjectStoreError("cannot stat fingerprinted upload source") from error
    if current != fingerprint.state:
        raise ObjectStoreError("local upload source changed after it was fingerprinted")
    return fingerprint


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
        parallel_upload_chunk_bytes: int = 16 * 1024**2,
        parallel_upload_workers: int = 32,
        parallel_download_threshold_bytes: int = 32 * 1024**2,
        parallel_download_chunk_bytes: int = 32 * 1024**2,
        parallel_download_workers: int = 16,
        transfer_timeout_seconds: float = 30 * 60.0,
        isolated_large_transfers: bool | None = None,
        parallel_worker_type: str = "thread",
        transfer_manager_module: Any | None = None,
    ):
        supplied_client = client is not None
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
        self.parallel_download_threshold_bytes = int(
            parallel_download_threshold_bytes
        )
        self.parallel_download_chunk_bytes = int(parallel_download_chunk_bytes)
        self.parallel_download_workers = int(parallel_download_workers)
        self.transfer_timeout_seconds = float(transfer_timeout_seconds)
        self.isolated_large_transfers = (
            not supplied_client and transfer_manager_module is None
            if isolated_large_transfers is None
            else bool(isolated_large_transfers)
        )
        self.parallel_worker_type = str(parallel_worker_type).lower()
        self._transfer_manager_module = transfer_manager_module
        if (
            self.parallel_upload_threshold_bytes <= 0
            or self.parallel_upload_chunk_bytes <= 0
            or self.parallel_upload_workers <= 0
            or self.parallel_download_threshold_bytes <= 0
            or self.parallel_download_chunk_bytes <= 0
            or self.parallel_download_workers <= 0
        ):
            raise ValueError("parallel GCS transfer settings must be positive")
        if (
            not math.isfinite(self.transfer_timeout_seconds)
            or self.transfer_timeout_seconds <= 0
        ):
            raise ValueError("transfer_timeout_seconds must be finite and positive")
        if self.parallel_worker_type not in {"thread", "process"}:
            raise ValueError("parallel_worker_type must be 'thread' or 'process'")

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
        self,
        source: Path,
        uri: str,
        *,
        sha256: str,
        source_fingerprint: SourceFingerprint | None = None,
    ) -> ObjectIdentity:
        if not _SHA256.fullmatch(sha256):
            raise ValueError("sha256 must be lowercase 64-character hex")
        fingerprint = _resolve_source_fingerprint(
            source, sha256, source_fingerprint
        )
        crc32c = fingerprint.crc32c
        size = fingerprint.size_bytes
        source_state = fingerprint.state
        blob = self._blob(uri)
        # Resume avoids retransmitting a content-addressed object that already
        # has the exact expected metadata and size.
        try:
            existing = self._identity(blob, uri, sha256)
        except Exception:
            existing = None
        if existing is not None:
            if existing.size_bytes != size or existing.crc32c != crc32c:
                raise ObjectStoreError(
                    f"existing GCS object content identity differs: {uri}"
                )
            return existing
        blob.metadata = {"sha256": sha256}
        if size >= self.parallel_upload_threshold_bytes:
            identity = self._put_parallel_conditional(
                source,
                blob,
                uri=uri,
                sha256=sha256,
                crc32c=crc32c,
                source_state=source_state,
            )
            if identity.size_bytes != size or identity.crc32c != crc32c:
                raise ObjectStoreError(
                    f"GCS object content identity differs after upload: {uri}"
                )
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
        if _file_state(source.stat()) != source_state:
            raise ObjectStoreError("local upload source changed during transfer")
        if identity.size_bytes != size or identity.crc32c != crc32c:
            raise ObjectStoreError(
                f"GCS object content identity differs after upload: {uri}"
            )
        return identity

    def _put_parallel_conditional(
        self,
        source: Path,
        final_blob: Any,
        *,
        uri: str,
        sha256: str,
        crc32c: str,
        source_state: _FileState,
    ) -> ObjectIdentity:
        """Multipart-upload to a unique temporary, then conditionally copy.

        GCS XML multipart uploads do not support create-only preconditions.
        The temporary name is therefore unpredictable, and the server-side
        copy into the content-addressed final name is the sole commit point
        with ``if_generation_match=0``.  Production launches the multipart
        transfer in a fresh, JAX-cold interpreter.  The process pool is
        created only inside that helper, never inside the live TPU process.
        """

        temporary = final_blob.bucket.blob(
            f"{final_blob.name}.multipart-{uuid.uuid4().hex}"
        )
        temporary.metadata = {"sha256": sha256}
        temporary_generation = 0
        try:
            if self.isolated_large_transfers:
                bucket_name, _ = parse_gcs_uri(uri)
                self._run_isolated_transfer(
                    "upload-temp",
                    "--source",
                    str(source),
                    "--uri",
                    f"gs://{bucket_name}/{temporary.name}",
                    "--sha256",
                    sha256,
                    "--crc32c",
                    crc32c,
                    "--chunk-bytes",
                    str(self.parallel_upload_chunk_bytes),
                    "--workers",
                    str(self.parallel_upload_workers),
                    "--deadline-seconds",
                    str(self.transfer_timeout_seconds),
                )
            else:
                transfer_manager = self._transfer_manager()
                transfer_manager.upload_chunks_concurrently(
                    str(source),
                    temporary,
                    chunk_size=self.parallel_upload_chunk_bytes,
                    worker_type=self._worker_type(transfer_manager),
                    max_workers=self.parallel_upload_workers,
                    checksum="crc32c",
                    deadline=self.transfer_timeout_seconds,
                )
            if _file_state(source.stat()) != source_state:
                raise ObjectStoreError(
                    "local upload source changed during multipart transfer"
                )
            temporary.reload()
            temporary_generation = int(temporary.generation or 0)
            if (
                temporary_generation <= 0
                or int(temporary.size or 0) != source_state.size
                or dict(temporary.metadata or {}).get("sha256") != sha256
                or str(temporary.crc32c or "") != crc32c
            ):
                raise ObjectStoreError(
                    "multipart upload has no valid temporary GCS identity"
                )
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
                    identity = self._identity(final_blob, uri, sha256)
                    self._require_content_identity(
                        identity,
                        size_bytes=source_state.size,
                        crc32c=crc32c,
                    )
                    return identity
                except Exception:
                    raise ObjectStoreError(
                        f"cannot conditionally commit multipart object {uri}"
                    ) from error
            identity = self._identity(final_blob, uri, sha256)
            self._require_content_identity(
                identity,
                size_bytes=source_state.size,
                crc32c=crc32c,
            )
            return identity
        finally:
            if temporary_generation <= 0:
                try:
                    temporary.reload()
                    temporary_generation = int(temporary.generation or 0)
                except Exception:
                    temporary_generation = 0
            if temporary_generation > 0:
                try:
                    temporary.delete(if_generation_match=temporary_generation)
                except Exception as error:
                    raise ObjectStoreError(
                        f"cannot remove multipart temporary for {uri}"
                    ) from error

    @staticmethod
    def _require_content_identity(
        identity: ObjectIdentity, *, size_bytes: int, crc32c: str
    ) -> None:
        if identity.size_bytes != size_bytes or identity.crc32c != crc32c:
            raise ObjectStoreError(
                f"GCS object content identity differs: {identity.uri}"
            )

    def _transfer_manager(self) -> Any:
        transfer_manager = self._transfer_manager_module
        if transfer_manager is None:
            try:
                from google.cloud.storage import transfer_manager
            except ImportError as error:  # pragma: no cover - optional package
                raise ObjectStoreError(
                    "parallel GCS transfer support is unavailable"
                ) from error
        return transfer_manager

    def _worker_type(self, transfer_manager: Any) -> Any:
        attribute = self.parallel_worker_type.upper()
        try:
            return getattr(transfer_manager, attribute)
        except AttributeError as error:
            raise ObjectStoreError(
                f"parallel GCS transfer manager has no {attribute} worker type"
            ) from error

    def _run_isolated_transfer(self, operation: str, *arguments: str) -> None:
        command = [
            sys.executable,
            "-m",
            "moe_sae._gcs_transfer",
            operation,
            *arguments,
        ]
        outer_timeout = (
            self.transfer_timeout_seconds
            + _ISOLATED_TRANSFER_EXIT_GRACE_SECONDS
        )
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=outer_timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise ObjectStoreError(
                f"isolated GCS {operation} helper exceeded "
                f"its {outer_timeout:.0f}-second hard deadline"
            ) from error
        except OSError as error:
            raise ObjectStoreError(
                f"cannot launch isolated GCS {operation} helper"
            ) from error
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            raise ObjectStoreError(
                f"isolated GCS {operation} helper failed "
                f"with exit code {completed.returncode}{suffix}"
            )

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
        if identity.size_bytes >= self.parallel_download_threshold_bytes:
            try:
                if self.isolated_large_transfers:
                    self._run_isolated_transfer(
                        "download",
                        "--uri",
                        identity.uri,
                        "--generation",
                        str(identity.generation),
                        "--size-bytes",
                        str(identity.size_bytes),
                        "--crc32c",
                        identity.crc32c,
                        "--sha256",
                        identity.sha256,
                        "--destination",
                        str(destination),
                        "--chunk-bytes",
                        str(self.parallel_download_chunk_bytes),
                        "--workers",
                        str(self.parallel_download_workers),
                        "--deadline-seconds",
                        str(self.transfer_timeout_seconds),
                    )
                else:
                    transfer_manager = self._transfer_manager()
                    transfer_manager.download_chunks_concurrently(
                        blob,
                        str(destination),
                        chunk_size=self.parallel_download_chunk_bytes,
                        download_kwargs={
                            "if_generation_match": identity.generation
                        },
                        worker_type=self._worker_type(transfer_manager),
                        max_workers=self.parallel_download_workers,
                        crc32c_checksum=True,
                        deadline=self.transfer_timeout_seconds,
                    )
            except Exception:
                destination.unlink(missing_ok=True)
                raise
        else:
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
        self,
        source: Path,
        uri: str,
        *,
        sha256: str,
        source_fingerprint: SourceFingerprint | None = None,
    ) -> ObjectIdentity:
        fingerprint = _resolve_source_fingerprint(
            source, sha256, source_fingerprint
        )
        size = fingerprint.size_bytes
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
        if _file_state(source.stat()) != fingerprint.state:
            raise ObjectStoreError("local object source changed during transfer")
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
    "SourceFingerprint",
    "fingerprint_file",
    "join_gcs_uri",
    "object_identity_from_mapping",
    "parse_gcs_uri",
    "sha256_file",
]
