"""Create-only, generation-pinned immutable bytes in Google Cloud Storage.

This module is the production storage primitive for converted R1 bytes.  It
does not publish a prefix-checkpoint manifest and is not itself an Orbax or
TensorStore checkpoint.  Callers provide exact object identities; payloads
may use :func:`content_addressed_object_id`, while a later checkpoint tranche
will use plan-derived identities for small bindings and regional envelopes.

The adapter deliberately exposes no listing operation.  Every mutation uses
``if_generation_match=0`` and every accepted object is re-read from its exact
generation and metageneration, then checked with local SHA-256 and CRC32C.
Every caller-supplied memoryview is copied into exact immutable bytes and
cryptographically checked before a canonical object path is opened for
upload.  The application-owned immutable snapshot is bounded at 64 MiB and
individual application stream slices are bounded at 8 MiB.  These are
separate allocation bounds, not a measured total-process RSS bound and not a
bound on multipart bodies or other copies inside the pinned HTTP client.

The pinned 3.13.0 client computes multipart CRC32C itself.  For resumable
objects this adapter supplies the precomputed CRC32C for server finalization
while disabling the client's checksum mode, which prevents its unqualified
``DataCorruption`` deletion path.  Buckets with a default CMEK are rejected
because they do not expose the required server CRC32C receipt.  Bucket region
and encryption are attested once; trusted operators must not update bucket
configuration while a store instance is live.

``google-cloud-storage`` is imported lazily.  Deterministic tests inject the
narrow ``bucket(...).blob(...)`` client surface and a retry-policy sentinel,
so the base CPU test environment does not need the optional cloud package.
Injected clients are explicitly test-only: they receive a different store
identity and cannot claim a distributed production identity.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import math
from pathlib import PurePosixPath
import re
from typing import Any, ClassVar, Mapping, Self

import google_crc32c

from .checkpoint_conversion import CheckpointConversionError
from .protocol import (
    CanonicalRecord,
    IDENTIFIER_PATTERN,
    ProtocolError,
    SHA256_PATTERN,
    canonical_json_bytes,
    require_integer,
    require_record_object,
    require_string,
)


GCS_CLIENT_VERSION = "3.13.0"
GCS_IMMUTABLE_STORAGE_CONTRACT = "gcs-json-create-only-generation-crc32c-sha256-v2"

_CONTENT_TYPE_BINARY = "application/octet-stream"
_CONTENT_TYPE_JSON = "application/json"
_CONTENT_TYPES = {_CONTENT_TYPE_BINARY, _CONTENT_TYPE_JSON}
_METADATA_SCHEMA_KEY = "moe-sae-r1-immutable-schema"
_METADATA_STORE_ID_KEY = "moe-sae-r1-immutable-store-id"
_METADATA_OBJECT_ID_SHA256_KEY = "moe-sae-r1-immutable-object-id-sha256"
_METADATA_SHA256_KEY = "moe-sae-r1-immutable-sha256"
_METADATA_SIZE_KEY = "moe-sae-r1-immutable-size-bytes"
# The metadata field layout remains v1.  The v2 storage contract is carried in
# the derived store ID, which is itself mandatory metadata, so objects written
# by the unsafe v1 adapter cannot be adopted by this adapter.
_METADATA_SCHEMA = "v1"

_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]")
# Regional buckets have hyphenated locations ending in a numeric revision
# (for example, ``us-central1``).  This deliberately rejects multi-region and
# predefined dual-region spellings such as ``us`` and ``nam4``.
_REGION = re.compile(r"[a-z]+(?:-[a-z0-9]+)+[0-9]")
_DECIMAL_GENERATION = re.compile(r"[1-9][0-9]*")
_CRC32C = re.compile(r"[A-Za-z0-9+/]{6}==")
_MAX_OBJECT_ID_BYTES = 1024
_MAX_GCS_NAME_BYTES = 1024
_MAX_GCS_LONG = (1 << 63) - 1
_MAX_GCS_UNSIGNED_LONG = (1 << 64) - 1
_MIN_STREAM_CHUNK_BYTES = 256 * 1024
MAX_APPLICATION_STREAM_SLICE_BYTES = 8 * 1024**2
_MAX_STREAM_CHUNK_BYTES = MAX_APPLICATION_STREAM_SLICE_BYTES
_MAX_MULTIPART_SIZE_BYTES = 8 * 1024**2
MAX_IMMUTABLE_SNAPSHOT_BYTES = 64 * 1024**2
_DEFAULT_MAXIMUM_OBJECT_BYTES = MAX_IMMUTABLE_SNAPSHOT_BYTES
_RETRYABLE_OR_AMBIGUOUS_STATUSES = {408, 412, 429, 500, 502, 503, 504}
_AUTO_RETRY = object()


class GCSImmutableByteStoreError(CheckpointConversionError):
    """An immutable GCS operation failed without an acceptable receipt."""


class GCSImmutableObjectConflict(GCSImmutableByteStoreError):
    """A create-only exact path contains incompatible bytes or metadata."""


class CorruptGCSImmutableObject(GCSImmutableByteStoreError):
    """A physical object no longer satisfies its immutable receipt."""


def _status_code(error: BaseException) -> int | None:
    """Extract an HTTP status without importing google-api-core in tests."""

    candidates = [getattr(error, "code", None)]
    response = getattr(error, "response", None)
    if response is not None:
        candidates.extend(
            (getattr(response, "status_code", None), getattr(response, "status", None))
        )
    for candidate in candidates:
        if callable(candidate):
            try:
                candidate = candidate()
            except TypeError:
                continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _validate_object_id(value: str) -> str:
    if type(value) is not str or not value:
        raise GCSImmutableByteStoreError("object_id must be a non-empty string")
    if len(value.encode("utf-8")) > _MAX_OBJECT_ID_BYTES:
        raise GCSImmutableByteStoreError("object_id exceeds its byte bound")
    if "\\" in value or value.startswith("/") or value.endswith("/"):
        raise GCSImmutableByteStoreError(
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
        raise GCSImmutableByteStoreError("object_id contains an unsafe path component")
    return value


def _protocol_object_id(value: str) -> str:
    try:
        return _validate_object_id(value)
    except GCSImmutableByteStoreError as error:
        raise ProtocolError("object_id is not a safe relative POSIX path") from error


def _validate_prefix(value: str) -> str:
    try:
        return _validate_object_id(value)
    except GCSImmutableByteStoreError as error:
        raise ValueError("prefix must be a normalized non-empty POSIX path") from error


def _validate_bucket(value: str) -> str:
    if type(value) is not str or _BUCKET.fullmatch(value) is None:
        raise ValueError("bucket must be a normalized GCS bucket name")
    return value


def _validate_region(value: str, *, protocol: bool = False) -> str:
    error_type: type[ValueError] = ProtocolError if protocol else ValueError
    if type(value) is not str or _REGION.fullmatch(value) is None:
        raise error_type("region must be a normalized lowercase GCS location")
    return value


def _decimal_generation(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or _DECIMAL_GENERATION.fullmatch(value) is None
        or len(value) > 19
        or int(value) > _MAX_GCS_LONG
    ):
        raise ProtocolError(f"{name} must be a canonical positive decimal string")
    return value


def _generation_integer(value: str, name: str) -> int:
    try:
        canonical = _decimal_generation(value, name)
    except ProtocolError as error:
        raise GCSImmutableByteStoreError(
            f"{name} is outside the accepted GCS long range"
        ) from error
    return int(canonical)


def _remote_decimal_generation(value: Any, name: str, uri: str) -> str:
    if type(value) is int:
        if not 1 <= value <= _MAX_GCS_LONG:
            raise CorruptGCSImmutableObject(f"{name} is invalid for {uri}")
        return str(value)
    if type(value) is not str:
        raise CorruptGCSImmutableObject(f"{name} is invalid for {uri}")
    if (
        _DECIMAL_GENERATION.fullmatch(value) is None
        or len(value) > 19
        or int(value) > _MAX_GCS_LONG
    ):
        raise CorruptGCSImmutableObject(f"{name} is invalid for {uri}")
    return value


def _remote_positive_integer(value: Any, name: str, uri: str) -> int:
    if type(value) is int:
        result = value
    elif type(value) is str:
        if (
            _DECIMAL_GENERATION.fullmatch(value) is None
            or len(value) > 20
            or (len(value) == 20 and value > str(_MAX_GCS_UNSIGNED_LONG))
        ):
            raise CorruptGCSImmutableObject(f"{name} is invalid for {uri}")
        result = int(value)
    else:
        raise CorruptGCSImmutableObject(f"{name} is invalid for {uri}")
    if not 1 <= result <= _MAX_GCS_UNSIGNED_LONG:
        raise CorruptGCSImmutableObject(f"{name} is invalid for {uri}")
    return result


def _validate_crc32c(value: Any, *, protocol: bool = False) -> str:
    error_type: type[ValueError] = (
        ProtocolError if protocol else CorruptGCSImmutableObject
    )
    if type(value) is not str or _CRC32C.fullmatch(value) is None:
        raise error_type("crc32c must be canonical base64 for four bytes")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise error_type("crc32c must be canonical base64 for four bytes") from error
    if len(decoded) != 4 or base64.b64encode(decoded).decode("ascii") != value:
        raise error_type("crc32c must be canonical base64 for four bytes")
    return value


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if type(uri) is not str or not uri.startswith("gs://"):
        raise ProtocolError("uri must be a gs:// object URI")
    bucket, separator, name = uri[5:].partition("/")
    if (
        not separator
        or _BUCKET.fullmatch(bucket) is None
        or not name
        or name.endswith("/")
        or "//" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ProtocolError("uri must be a normalized gs:// object URI")
    if len(name.encode("utf-8")) > _MAX_GCS_NAME_BYTES:
        raise ProtocolError("GCS object name exceeds its byte bound")
    return bucket, name


def _content_type(object_id: str) -> str:
    if object_id.endswith(".bin"):
        return _CONTENT_TYPE_BINARY
    if object_id.endswith(".json"):
        return _CONTENT_TYPE_JSON
    raise GCSImmutableByteStoreError("immutable object_id must end in .bin or .json")


def _protocol_content_type(object_id: str) -> str:
    try:
        return _content_type(object_id)
    except GCSImmutableByteStoreError as error:
        raise ProtocolError("object_id must end in .bin or .json") from error


def content_addressed_object_id(sha256: str) -> str:
    """Return the canonical relative payload path for one SHA-256 digest."""

    if type(sha256) is not str:
        raise ProtocolError("sha256 must be an exact immutable string")
    digest = require_string(sha256, "sha256", pattern=SHA256_PATTERN)
    return _validate_object_id(f"objects/sha256/{digest[:2]}/{digest}.bin")


def content_addressed_record_object_id(sha256: str) -> str:
    """Return the canonical relative JSON-record path for one SHA-256 digest."""

    if type(sha256) is not str:
        raise ProtocolError("sha256 must be an exact immutable string")
    digest = require_string(sha256, "sha256", pattern=SHA256_PATTERN)
    return _validate_object_id(f"records/sha256/{digest[:2]}/{digest}.json")


def _validate_reserved_content_address(
    object_id: str,
    sha256: str,
    *,
    protocol: bool = False,
) -> None:
    """Protect canonical binary and JSON SHA-256 namespaces from squatting."""

    if object_id == "objects/sha256" or object_id.startswith("objects/sha256/"):
        expected = f"objects/sha256/{sha256[:2]}/{sha256}.bin"
    elif object_id == "records/sha256" or object_id.startswith("records/sha256/"):
        expected = f"records/sha256/{sha256[:2]}/{sha256}.json"
    else:
        return
    if object_id != expected:
        error_type: type[ValueError] = (
            ProtocolError if protocol else GCSImmutableByteStoreError
        )
        raise error_type(
            "SHA-256 namespaces are reserved for canonical content-addressed objects"
        )


def _crc32c_base64(checksum: google_crc32c.Checksum) -> str:
    return base64.b64encode(checksum.digest()).decode("ascii")


def _metadata(
    *, store_id: str, object_id: str, sha256: str, size_bytes: int
) -> dict[str, str]:
    return {
        _METADATA_SCHEMA_KEY: _METADATA_SCHEMA,
        _METADATA_STORE_ID_KEY: store_id,
        _METADATA_OBJECT_ID_SHA256_KEY: hashlib.sha256(
            object_id.encode("utf-8")
        ).hexdigest(),
        _METADATA_SHA256_KEY: sha256,
        _METADATA_SIZE_KEY: str(size_bytes),
    }


@dataclass(frozen=True, slots=True)
class GCSPhysicalObjectReceiptV1(CanonicalRecord):
    """Exact generation, metadata generation, CRC32C, and SHA evidence."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.gcs-physical-object-receipt.v1"

    store_id: str
    object_id: str
    uri: str
    region: str
    generation: str
    metageneration: str
    size_bytes: int
    crc32c: str
    sha256: str
    content_type: str
    storage_contract: str = GCS_IMMUTABLE_STORAGE_CONTRACT

    def __post_init__(self) -> None:
        for name in (
            "store_id",
            "object_id",
            "uri",
            "region",
            "generation",
            "metageneration",
            "crc32c",
            "sha256",
            "content_type",
            "storage_contract",
        ):
            if type(getattr(self, name)) is not str:
                raise ProtocolError(f"{name} must be an exact immutable string")
        require_string(self.store_id, "store_id", pattern=IDENTIFIER_PATTERN)
        object_id = _protocol_object_id(self.object_id)
        sha256 = require_string(self.sha256, "sha256", pattern=SHA256_PATTERN)
        _validate_reserved_content_address(object_id, sha256, protocol=True)
        _bucket, name = _parse_gs_uri(self.uri)
        if name != object_id and not name.endswith(f"/{object_id}"):
            raise ProtocolError("uri does not end at object_id")
        _validate_region(self.region, protocol=True)
        _decimal_generation(self.generation, "generation")
        _decimal_generation(self.metageneration, "metageneration")
        require_integer(self.size_bytes, "size_bytes", minimum=1)
        _validate_crc32c(self.crc32c, protocol=True)
        if self.content_type not in _CONTENT_TYPES:
            raise ProtocolError("unsupported immutable object content_type")
        if self.content_type != _protocol_content_type(object_id):
            raise ProtocolError("content_type does not match object_id suffix")
        if self.storage_contract != GCS_IMMUTABLE_STORAGE_CONTRACT:
            raise ProtocolError("unsupported GCS immutable storage contract")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "store_id": self.store_id,
            "object_id": self.object_id,
            "uri": self.uri,
            "region": self.region,
            "generation": self.generation,
            "metageneration": self.metageneration,
            "size_bytes": self.size_bytes,
            "crc32c": self.crc32c,
            "sha256": self.sha256,
            "content_type": self.content_type,
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
                "uri",
                "region",
                "generation",
                "metageneration",
                "size_bytes",
                "crc32c",
                "sha256",
                "content_type",
                "storage_contract",
            ),
        )
        return cls(
            store_id=require_string(
                obj["store_id"], "store_id", pattern=IDENTIFIER_PATTERN
            ),
            object_id=_protocol_object_id(
                require_string(obj["object_id"], "object_id")
            ),
            uri=require_string(obj["uri"], "uri"),
            region=require_string(obj["region"], "region"),
            generation=require_string(obj["generation"], "generation"),
            metageneration=require_string(obj["metageneration"], "metageneration"),
            size_bytes=require_integer(obj["size_bytes"], "size_bytes", minimum=1),
            crc32c=require_string(obj["crc32c"], "crc32c"),
            sha256=require_string(obj["sha256"], "sha256", pattern=SHA256_PATTERN),
            content_type=require_string(obj["content_type"], "content_type"),
            storage_contract=require_string(
                obj["storage_contract"], "storage_contract"
            ),
        )


class _MemoryViewReader:
    """Seekable byte reader that never returns more than one stream chunk."""

    def __init__(
        self,
        view: memoryview,
        *,
        maximum_read_bytes: int,
        exact_bytes: bytes,
    ) -> None:
        if (
            type(exact_bytes) is not bytes
            or not view.readonly
            or view.obj is not exact_bytes
            or len(exact_bytes) != len(view)
        ):
            raise GCSImmutableByteStoreError(
                "upload reader requires an exact immutable bytes view"
            )
        self._view = view
        self._maximum_read_bytes = maximum_read_bytes
        self._exact_bytes = exact_bytes
        self._position = 0

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self._position + offset
        elif whence == 2:
            position = len(self._view) + offset
        else:
            raise ValueError("invalid seek origin")
        if type(position) is not int or not 0 <= position <= len(self._view):
            raise ValueError("seek exceeds immutable payload")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        remaining = len(self._view) - self._position
        if size is None or size < 0:
            size = remaining
        if type(size) is not int:
            raise TypeError("read size must be an integer")
        request = min(size, remaining)
        if request > self._maximum_read_bytes:
            raise GCSImmutableByteStoreError(
                "GCS client requested an upload stream slice above the bound"
            )
        start = self._position
        self._position += request
        if start == 0 and request == len(self._exact_bytes):
            # Pinned 3.13.0 reads a multipart object in one call.  Reusing the
            # exact immutable object avoids a second 8-MiB Python allocation
            # while the bounded memoryview snapshot is live.
            return self._exact_bytes
        return self._view[start : start + request].tobytes()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True


class _HashingCaptureWriter:
    """Hash a bounded download while retaining only the requested subrange."""

    def __init__(
        self,
        *,
        expected_size_bytes: int,
        maximum_write_bytes: int,
        capture_offset: int | None = None,
        capture_length: int = 0,
    ) -> None:
        self._expected_size_bytes = expected_size_bytes
        self._maximum_write_bytes = maximum_write_bytes
        self._capture_offset = capture_offset
        self._capture_length = capture_length
        self._captured = None if capture_offset is None else bytearray(capture_length)
        self._sha256 = hashlib.sha256()
        self._crc32c = google_crc32c.Checksum()
        self._position = 0

    def write(self, value: bytes | bytearray | memoryview) -> int:
        try:
            view = memoryview(value).cast("B")
        except (TypeError, ValueError) as error:
            raise CorruptGCSImmutableObject(
                "GCS client produced a non-contiguous download chunk"
            ) from error
        try:
            count = len(view)
            if count > self._maximum_write_bytes:
                raise CorruptGCSImmutableObject(
                    "GCS client produced a download stream slice above the bound"
                )
            if self._position + count > self._expected_size_bytes:
                raise CorruptGCSImmutableObject(
                    "GCS download exceeded immutable object size"
                )
            self._sha256.update(view)
            if type(value) is bytes:
                # The pinned client writes exact bytes.  Reuse that immutable
                # stream slice instead of making an application-side copy.
                self._crc32c.update(value)
            else:
                # google-crc32c's C extension does not accept every
                # buffer-protocol value.  This fallback copy is independently
                # capped by maximum_write_bytes.
                self._crc32c.update(view.tobytes())
            if self._captured is not None and self._capture_offset is not None:
                left = max(self._position, self._capture_offset)
                right = min(
                    self._position + count,
                    self._capture_offset + self._capture_length,
                )
                if left < right:
                    source_start = left - self._position
                    target_start = left - self._capture_offset
                    self._captured[target_start : target_start + right - left] = view[
                        source_start : source_start + right - left
                    ]
            self._position += count
            return count
        finally:
            view.release()

    def finish(self) -> tuple[str, str, bytes | None]:
        if self._position != self._expected_size_bytes:
            raise CorruptGCSImmutableObject(
                "GCS download ended before immutable object size"
            )
        return (
            self._sha256.hexdigest(),
            _crc32c_base64(self._crc32c),
            None if self._captured is None else bytes(self._captured),
        )


class GCSImmutableByteStore:
    """No-listing create-only byte store on one region-pinned GCS prefix."""

    __slots__ = (
        "_bucket_name_snapshot",
        "_bucket_snapshot",
        "_client_snapshot",
        "_io_chunk_bytes",
        "_maximum_object_bytes",
        "_prefix_snapshot",
        "_production_identity_qualified",
        "_region_snapshot",
        "_resolution_attempts",
        "_retry_policy",
        "_store_id",
        "_stream_buffer_bytes",
        "_timeout_seconds",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__ and hasattr(self, name):
            raise AttributeError(f"{name} is an immutable store snapshot")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in self.__slots__:
            raise AttributeError(f"{name} is an immutable store snapshot")
        object.__delattr__(self, name)

    def __init__(
        self,
        bucket: str,
        prefix: str,
        expected_region: str,
        client: Any | None = None,
        *,
        project: str | None = None,
        io_chunk_bytes: int = _MAX_STREAM_CHUNK_BYTES,
        maximum_object_bytes: int = _DEFAULT_MAXIMUM_OBJECT_BYTES,
        timeout_seconds: float = 60.0,
        retry_deadline_seconds: float = 120.0,
        resolution_attempts: int = 4,
        retry_policy: Any = _AUTO_RETRY,
    ) -> None:
        bucket_name = _validate_bucket(bucket)
        normalized_prefix = _validate_prefix(prefix)
        normalized_region = _validate_region(expected_region)
        if project is not None and (type(project) is not str or not project):
            raise ValueError("project must be a non-empty string when supplied")
        if (
            type(io_chunk_bytes) is not int
            or not _MIN_STREAM_CHUNK_BYTES <= io_chunk_bytes <= _MAX_STREAM_CHUNK_BYTES
            or io_chunk_bytes % _MIN_STREAM_CHUNK_BYTES != 0
        ):
            raise ValueError(
                "io_chunk_bytes must be a 256-KiB multiple in [256 KiB, 8 MiB]"
            )
        if (
            type(maximum_object_bytes) is not int
            or not 1 <= maximum_object_bytes <= MAX_IMMUTABLE_SNAPSHOT_BYTES
        ):
            raise ValueError(
                "maximum_object_bytes must be in [1, 64 MiB] so mutable inputs "
                "have a deterministic immutable-snapshot bound"
            )
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive")
        if isinstance(retry_deadline_seconds, bool) or not isinstance(
            retry_deadline_seconds, (int, float)
        ):
            raise ValueError("retry_deadline_seconds must be finite and positive")
        retry_deadline = float(retry_deadline_seconds)
        if not math.isfinite(retry_deadline) or retry_deadline <= 0.0:
            raise ValueError("retry_deadline_seconds must be finite and positive")
        if type(resolution_attempts) is not int or resolution_attempts <= 0:
            raise ValueError("resolution_attempts must be a positive integer")

        internally_constructed_client = client is None
        storage_module: Any | None = None
        if client is None or retry_policy is _AUTO_RETRY:
            try:
                from google.cloud import storage
                from google.cloud.storage.retry import DEFAULT_RETRY
            except ImportError as error:
                raise GCSImmutableByteStoreError(
                    "GCS immutable storage requires google-cloud-storage==3.13.0 "
                    "or an injected client and retry policy"
                ) from error
            storage_module = storage
            if retry_policy is _AUTO_RETRY:
                retry_policy = DEFAULT_RETRY
        if client is None:
            if getattr(storage_module, "__version__", None) != GCS_CLIENT_VERSION:
                raise GCSImmutableByteStoreError(
                    "GCS immutable storage requires google-cloud-storage==3.13.0"
                )
            client = storage_module.Client(project=project)
        if retry_policy is None:
            raise ValueError("retry_policy must not disable bounded retries")
        with_timeout = getattr(retry_policy, "with_timeout", None)
        if not callable(with_timeout):
            raise ValueError("retry_policy must expose with_timeout")
        retry_policy = with_timeout(retry_deadline)
        if retry_policy is None:
            raise ValueError("retry_policy.with_timeout returned no policy")

        self._client_snapshot = client
        self._bucket_name_snapshot = bucket_name
        self._prefix_snapshot = normalized_prefix
        self._region_snapshot = normalized_region
        self._io_chunk_bytes = io_chunk_bytes
        # The pinned Blob API always materialises a complete multipart
        # payload at or below 8 MiB even when its resumable chunk is smaller.
        # Therefore 8 MiB is the honest explicit stream-slice ceiling.
        self._stream_buffer_bytes = _MAX_STREAM_CHUNK_BYTES
        self._maximum_object_bytes = maximum_object_bytes
        self._timeout_seconds = timeout
        self._retry_policy = retry_policy
        self._resolution_attempts = resolution_attempts
        self._bucket_snapshot = client.bucket(bucket_name)
        try:
            self._bucket_snapshot.reload(
                projection="noAcl",
                timeout=self._timeout_seconds,
                retry=self._retry_policy,
            )
        except Exception as error:
            raise GCSImmutableByteStoreError(
                f"cannot attest GCS bucket location: {bucket_name}"
            ) from error
        observed_location = getattr(self._bucket_snapshot, "location", None)
        if (
            type(observed_location) is not str
            or observed_location.lower() != normalized_region
        ):
            raise GCSImmutableByteStoreError(
                "GCS bucket location does not match expected_region"
            )
        if getattr(self._bucket_snapshot, "default_kms_key_name", None) is not None:
            raise GCSImmutableByteStoreError(
                "GCS immutable storage does not support a default CMEK because "
                "the required server CRC32C receipt is unavailable"
            )
        self._production_identity_qualified = internally_constructed_client
        identity_scope = (
            "production-exact-client"
            if internally_constructed_client
            else "injected-test-client-unqualified"
        )
        identity = {
            "schema": "moe-sae.r1.gcs-immutable-byte-store-identity.v1",
            "bucket": bucket_name,
            "prefix": normalized_prefix,
            "region": normalized_region,
            "storage_contract": GCS_IMMUTABLE_STORAGE_CONTRACT,
            "client_contract": (
                f"google-cloud-storage=={GCS_CLIENT_VERSION}"
                if internally_constructed_client
                else "injected-test-client"
            ),
            "identity_scope": identity_scope,
        }
        store_id_prefix = (
            "gcs-prefix-byte-store-v2:"
            if internally_constructed_client
            else "gcs-test-prefix-byte-store-v2:"
        )
        self._store_id = (
            store_id_prefix + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        )

    @property
    def store_id(self) -> str:
        return self._store_id

    @property
    def client(self) -> Any:
        """The construction-time client handle; the property is read-only."""

        return self._client_snapshot

    @property
    def bucket_name(self) -> str:
        return self._bucket_name_snapshot

    @property
    def prefix(self) -> str:
        return self._prefix_snapshot

    @property
    def region(self) -> str:
        return self._region_snapshot

    @property
    def immutable_snapshot_bytes(self) -> int:
        """Maximum exact-bytes snapshot made from caller-mutable storage."""

        return self._maximum_object_bytes

    @property
    def stream_buffer_bytes(self) -> int:
        """Maximum application stream slice, separate from snapshot bytes."""

        return self._stream_buffer_bytes

    @property
    def maximum_object_bytes(self) -> int:
        return self._maximum_object_bytes

    @property
    def distributed_identity_qualified(self) -> bool:
        return self._production_identity_qualified

    def _require_address_state(self) -> None:
        self._require_bucket_address(self._bucket_snapshot)

    def _require_bucket_address(self, bucket: Any) -> None:
        bucket_client = getattr(bucket, "client", getattr(bucket, "_client", None))
        bucket_name = getattr(bucket, "name", None)
        if (
            type(bucket_name) is not str
            or bucket_name != self._bucket_name_snapshot
            or bucket_client is not self._client_snapshot
        ):
            raise GCSImmutableByteStoreError(
                "GCS bucket/client handle drifted from the immutable store identity"
            )

    def _require_blob_address(self, blob: Any, object_name: str) -> None:
        blob_name = getattr(blob, "name", None)
        blob_bucket = getattr(blob, "bucket", None)
        if type(blob_bucket) is str:
            blob_bucket_name = blob_bucket
            blob_bucket_client = self._client_snapshot
        else:
            blob_bucket_name = getattr(blob_bucket, "name", None)
            blob_bucket_client = getattr(
                blob_bucket, "client", getattr(blob_bucket, "_client", None)
            )
        if (
            type(blob_name) is not str
            or blob_name != object_name
            or type(blob_bucket_name) is not str
            or blob_bucket_name != self._bucket_name_snapshot
            or blob_bucket_client is not self._client_snapshot
        ):
            raise GCSImmutableByteStoreError(
                "GCS blob address drifted from the immutable store identity"
            )

    def object_uri(self, object_id: str) -> str:
        self._require_address_state()
        object_id = _validate_object_id(object_id)
        return f"gs://{self._bucket_name_snapshot}/{self._object_name(object_id)}"

    def _object_name(self, object_id: str) -> str:
        self._require_address_state()
        object_id = _validate_object_id(object_id)
        name = f"{self._prefix_snapshot}/{object_id}"
        if len(name.encode("utf-8")) > _MAX_GCS_NAME_BYTES:
            raise GCSImmutableByteStoreError("GCS object name exceeds its byte bound")
        return name

    def _blob(self, object_id: str, *, generation: int | None = None) -> Any:
        self._require_address_state()
        object_name = self._object_name(object_id)
        # Construct a private operation-local handle from the frozen address.
        # The attested construction-time bucket remains a configuration
        # snapshot only; accidental mutation of that retained object can
        # therefore never redirect an object request after the state check.
        bucket = self._client_snapshot.bucket(self._bucket_name_snapshot)
        self._require_bucket_address(bucket)
        blob = bucket.blob(
            object_name,
            chunk_size=self._io_chunk_bytes,
            generation=generation,
        )
        self._require_blob_address(blob, object_name)
        return blob

    def _receipt_from_blob(
        self, object_id: str, blob: Any
    ) -> GCSPhysicalObjectReceiptV1:
        uri = self.object_uri(object_id)
        object_name = self._object_name(object_id)
        try:
            self._require_blob_address(blob, object_name)
        except GCSImmutableByteStoreError as error:
            raise CorruptGCSImmutableObject(
                f"blob address is invalid for {uri}"
            ) from error
        generation = _remote_decimal_generation(blob.generation, "generation", uri)
        metageneration = _remote_decimal_generation(
            blob.metageneration, "metageneration", uri
        )
        size_bytes = _remote_positive_integer(blob.size, "size", uri)
        if size_bytes > self.maximum_object_bytes:
            raise CorruptGCSImmutableObject(
                f"immutable object exceeds store size bound: {uri}"
            )
        crc32c = _validate_crc32c(blob.crc32c)
        content_type = getattr(blob, "content_type", None)
        if type(content_type) is not str or content_type != _content_type(object_id):
            raise CorruptGCSImmutableObject(
                f"immutable object has unexpected content type: {uri}"
            )
        if getattr(blob, "content_encoding", None) is not None:
            raise CorruptGCSImmutableObject(
                f"immutable object must not have content encoding: {uri}"
            )
        metadata = getattr(blob, "metadata", None)
        if not isinstance(metadata, Mapping):
            raise CorruptGCSImmutableObject(
                f"immutable object metadata is missing: {uri}"
            )
        metadata_dict = dict(metadata)
        expected_keys = {
            _METADATA_SCHEMA_KEY,
            _METADATA_STORE_ID_KEY,
            _METADATA_OBJECT_ID_SHA256_KEY,
            _METADATA_SHA256_KEY,
            _METADATA_SIZE_KEY,
        }
        sha256 = metadata_dict.get(_METADATA_SHA256_KEY)
        if (
            set(metadata_dict) != expected_keys
            or metadata_dict.get(_METADATA_SCHEMA_KEY) != _METADATA_SCHEMA
            or metadata_dict.get(_METADATA_STORE_ID_KEY) != self.store_id
            or metadata_dict.get(_METADATA_OBJECT_ID_SHA256_KEY)
            != hashlib.sha256(object_id.encode("utf-8")).hexdigest()
            or metadata_dict.get(_METADATA_SIZE_KEY) != str(size_bytes)
            or type(sha256) is not str
            or SHA256_PATTERN.fullmatch(sha256) is None
        ):
            raise CorruptGCSImmutableObject(
                f"immutable object metadata is incompatible: {uri}"
            )
        try:
            _validate_reserved_content_address(object_id, sha256)
        except GCSImmutableByteStoreError as error:
            raise CorruptGCSImmutableObject(
                f"immutable object violates the reserved SHA-256 namespace: {uri}"
            ) from error
        return GCSPhysicalObjectReceiptV1(
            store_id=self.store_id,
            object_id=object_id,
            uri=uri,
            region=self._region_snapshot,
            generation=generation,
            metageneration=metageneration,
            size_bytes=size_bytes,
            crc32c=crc32c,
            sha256=sha256,
            content_type=content_type,
        )

    def _head(
        self,
        object_id: str,
        *,
        generation: str | None = None,
        metageneration: str | None = None,
        absent_ok: bool = False,
    ) -> tuple[Any, GCSPhysicalObjectReceiptV1] | None:
        numeric_generation = (
            None
            if generation is None
            else _generation_integer(generation, "generation")
        )
        blob = self._blob(object_id, generation=numeric_generation)
        kwargs: dict[str, Any] = {
            "projection": "noAcl",
            "timeout": self._timeout_seconds,
            "retry": self._retry_policy,
        }
        if generation is not None:
            kwargs["if_generation_match"] = numeric_generation
        if metageneration is not None:
            kwargs["if_metageneration_match"] = _generation_integer(
                metageneration, "metageneration"
            )
        try:
            blob.reload(**kwargs)
        except Exception as error:
            status = _status_code(error)
            if status == 404 and absent_ok:
                return None
            if status in {404, 412} and generation is not None:
                raise CorruptGCSImmutableObject(
                    f"physical receipt no longer resolves: {self.object_uri(object_id)}"
                ) from error
            raise GCSImmutableByteStoreError(
                f"cannot read immutable GCS metadata: {self.object_uri(object_id)}"
            ) from error
        receipt = self._receipt_from_blob(object_id, blob)
        if generation is not None and (
            receipt.generation != generation or receipt.metageneration != metageneration
        ):
            raise CorruptGCSImmutableObject(
                f"physical receipt metadata changed: {receipt.uri}"
            )
        return blob, receipt

    def _validate_receipt(self, receipt: GCSPhysicalObjectReceiptV1) -> None:
        if type(receipt) is not GCSPhysicalObjectReceiptV1:
            raise GCSImmutableByteStoreError("physical receipt has the wrong type")
        if receipt.store_id != self.store_id or receipt.region != self._region_snapshot:
            raise GCSImmutableByteStoreError(
                "physical receipt belongs to another GCS store"
            )
        if receipt.uri != self.object_uri(receipt.object_id):
            raise GCSImmutableByteStoreError(
                "physical receipt URI does not match this GCS store"
            )
        if receipt.size_bytes > self.maximum_object_bytes:
            raise GCSImmutableByteStoreError(
                "physical receipt exceeds the GCS store size bound"
            )

    def _measure_receipt(
        self,
        receipt: GCSPhysicalObjectReceiptV1,
        *,
        capture_offset: int | None = None,
        capture_length: int = 0,
    ) -> tuple[GCSPhysicalObjectReceiptV1, bytes | None]:
        self._validate_receipt(receipt)
        exact = self._head(
            receipt.object_id,
            generation=receipt.generation,
            metageneration=receipt.metageneration,
        )
        if exact is None:  # pragma: no cover - absent_ok is false
            raise CorruptGCSImmutableObject("exact immutable generation vanished")
        blob, observed = exact
        if observed != receipt:
            raise CorruptGCSImmutableObject(
                f"physical receipt no longer matches {receipt.object_id}"
            )
        writer = _HashingCaptureWriter(
            expected_size_bytes=receipt.size_bytes,
            maximum_write_bytes=self._io_chunk_bytes,
            capture_offset=capture_offset,
            capture_length=capture_length,
        )
        try:
            blob.download_to_file(
                writer,
                raw_download=True,
                if_generation_match=_generation_integer(
                    receipt.generation, "generation"
                ),
                if_metageneration_match=_generation_integer(
                    receipt.metageneration, "metageneration"
                ),
                checksum=None,
                timeout=self._timeout_seconds,
                retry=self._retry_policy,
                single_shot_download=False,
            )
        except GCSImmutableByteStoreError:
            raise
        except Exception as error:
            if _status_code(error) in {404, 412}:
                raise CorruptGCSImmutableObject(
                    f"physical receipt changed during read: {receipt.object_id}"
                ) from error
            raise GCSImmutableByteStoreError(
                f"cannot download immutable GCS bytes: {receipt.object_id}"
            ) from error
        sha256, crc32c, captured = writer.finish()
        if sha256 != receipt.sha256 or crc32c != receipt.crc32c:
            raise CorruptGCSImmutableObject(
                f"immutable GCS bytes fail SHA-256 or CRC32C: {receipt.object_id}"
            )
        live = self._head(receipt.object_id, absent_ok=True)
        if live is None or live[1] != receipt:
            raise CorruptGCSImmutableObject(
                f"immutable GCS live path changed during read: {receipt.object_id}"
            )
        return receipt, captured

    def _resolve_expected(
        self,
        object_id: str,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
        expected_crc32c: str,
    ) -> GCSPhysicalObjectReceiptV1:
        last_error: GCSImmutableByteStoreError | None = None
        for _attempt in range(self._resolution_attempts):
            try:
                current = self._head(object_id, absent_ok=True)
            except CorruptGCSImmutableObject as error:
                raise GCSImmutableObjectConflict(
                    f"immutable GCS object has incompatible metadata: {object_id}"
                ) from error
            except GCSImmutableByteStoreError as error:
                last_error = error
                continue
            if current is None:
                continue
            receipt = current[1]
            if (
                receipt.sha256 != expected_sha256
                or receipt.size_bytes != expected_size_bytes
                or receipt.crc32c != expected_crc32c
            ):
                raise GCSImmutableObjectConflict(
                    f"immutable GCS object exists with divergent identity: {object_id}"
                )
            try:
                return self._measure_receipt(receipt)[0]
            except CorruptGCSImmutableObject as error:
                raise GCSImmutableObjectConflict(
                    f"immutable GCS object exists but fails verification: {object_id}"
                ) from error
            except GCSImmutableByteStoreError as error:
                last_error = error
                continue
        raise GCSImmutableByteStoreError(
            f"cannot resolve create-only GCS upload: {object_id}"
        ) from last_error

    def create_content_or_verify(
        self,
        payload: bytes,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> GCSPhysicalObjectReceiptV1:
        return self.create_or_verify(
            content_addressed_object_id(expected_sha256),
            payload,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )

    def create_memoryview_or_verify(
        self,
        object_id: str,
        payload: memoryview,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> GCSPhysicalObjectReceiptV1:
        """Snapshot mutable storage before opening a canonical upload path.

        The exact immutable snapshot is bounded by
        :attr:`immutable_snapshot_bytes`.  SHA-256 is checked from that
        snapshot before upload, so concurrent caller mutation can only produce
        either the expected immutable bytes or a pre-upload failure.  The
        method is synchronous and never retains ``payload`` after return.
        """

        if type(payload) is not memoryview:
            raise GCSImmutableByteStoreError(
                "mutable immutable-upload payload must be an exact memoryview"
            )
        try:
            view = payload.cast("B")
        except (TypeError, ValueError) as error:
            raise GCSImmutableByteStoreError(
                "immutable upload memoryview must be contiguous byte storage"
            ) from error
        try:
            if (
                type(expected_size_bytes) is not int
                or not 1 <= expected_size_bytes <= self.immutable_snapshot_bytes
            ):
                raise GCSImmutableByteStoreError(
                    "expected_size_bytes exceeds the immutable snapshot bound"
                )
            if len(view) != expected_size_bytes:
                raise GCSImmutableByteStoreError(
                    "immutable payload size differs from expectation"
                )
            snapshot = view.tobytes()
            return self.create_or_verify(
                object_id,
                snapshot,
                expected_sha256=expected_sha256,
                expected_size_bytes=expected_size_bytes,
            )
        finally:
            view.release()

    def create_or_verify(
        self,
        object_id: str,
        payload: bytes,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> GCSPhysicalObjectReceiptV1:
        if type(payload) is not bytes:
            raise GCSImmutableByteStoreError(
                "immutable payload must be exact bytes, not mutable buffer storage"
            )
        view = memoryview(payload)
        try:
            return self._create_view_or_verify(
                object_id,
                view,
                expected_sha256=expected_sha256,
                expected_size_bytes=expected_size_bytes,
                exact_bytes=payload,
            )
        finally:
            view.release()

    def _create_view_or_verify(
        self,
        object_id: str,
        view: memoryview,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
        exact_bytes: bytes,
    ) -> GCSPhysicalObjectReceiptV1:
        object_id = _validate_object_id(object_id)
        if type(expected_sha256) is not str:
            raise GCSImmutableByteStoreError(
                "expected_sha256 must be an exact immutable string"
            )
        expected_sha256 = require_string(
            expected_sha256, "expected_sha256", pattern=SHA256_PATTERN
        )
        _validate_reserved_content_address(object_id, expected_sha256)
        content_type = _content_type(object_id)
        if (
            type(expected_size_bytes) is not int
            or not 1 <= expected_size_bytes <= self.maximum_object_bytes
        ):
            raise GCSImmutableByteStoreError(
                "expected_size_bytes exceeds the positive store bound"
            )
        if len(view) != expected_size_bytes:
            raise GCSImmutableByteStoreError(
                "immutable payload size differs from expectation"
            )
        if (
            type(exact_bytes) is not bytes
            or not view.readonly
            or view.obj is not exact_bytes
            or len(exact_bytes) != len(view)
        ):
            raise GCSImmutableByteStoreError(
                "canonical upload requires an exact immutable bytes view"
            )
        sha256 = hashlib.sha256()
        cursor = 0
        while cursor < len(view):
            stop = min(cursor + self.stream_buffer_bytes, len(view))
            piece = view[cursor:stop]
            try:
                sha256.update(piece)
            finally:
                piece.release()
            cursor = stop
        if sha256.hexdigest() != expected_sha256:
            raise GCSImmutableByteStoreError(
                "immutable payload SHA-256 differs from expectation"
            )
        crc32c = google_crc32c.Checksum()
        crc32c.update(exact_bytes)
        expected_crc32c = _crc32c_base64(crc32c)
        blob = self._blob(object_id)
        blob.metadata = _metadata(
            store_id=self.store_id,
            object_id=object_id,
            sha256=expected_sha256,
            size_bytes=expected_size_bytes,
        )
        reader = _MemoryViewReader(
            view,
            maximum_read_bytes=self.stream_buffer_bytes,
            exact_bytes=exact_bytes,
        )
        try:
            # In 3.13.0 crc32c_checksum_value is ignored by multipart
            # uploads, whose client checksum is safe because payload is
            # exact bytes.  Resumable uploads send the precomputed value
            # for server validation with no client checksum, avoiding the
            # client's generation-unqualified DataCorruption deletion.
            checksum = (
                "crc32c" if expected_size_bytes <= _MAX_MULTIPART_SIZE_BYTES else None
            )
            blob.upload_from_file(
                reader,
                rewind=False,
                size=expected_size_bytes,
                content_type=content_type,
                if_generation_match=0,
                checksum=checksum,
                retry=self._retry_policy,
                timeout=self._timeout_seconds,
                crc32c_checksum_value=expected_crc32c,
            )
        except Exception as error:
            status = _status_code(error)
            if status is None or status in _RETRYABLE_OR_AMBIGUOUS_STATUSES:
                try:
                    return self._resolve_expected(
                        object_id,
                        expected_sha256=expected_sha256,
                        expected_size_bytes=expected_size_bytes,
                        expected_crc32c=expected_crc32c,
                    )
                except GCSImmutableObjectConflict:
                    raise
                except GCSImmutableByteStoreError as resolution_error:
                    raise GCSImmutableByteStoreError(
                        f"cannot create immutable GCS object: {object_id}"
                    ) from resolution_error
            raise GCSImmutableByteStoreError(
                f"cannot create immutable GCS object: {object_id}"
            ) from error
        return self._resolve_expected(
            object_id,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
            expected_crc32c=expected_crc32c,
        )

    def probe_bounded(
        self, object_id: str, *, maximum_bytes: int
    ) -> tuple[bytes, GCSPhysicalObjectReceiptV1] | None:
        object_id = _validate_object_id(object_id)
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise GCSImmutableByteStoreError("maximum_bytes must be a positive integer")
        current = self._head(object_id, absent_ok=True)
        if current is None:
            return None
        receipt = current[1]
        if receipt.size_bytes > maximum_bytes:
            raise GCSImmutableByteStoreError(
                "immutable GCS object exceeds bounded probe size"
            )
        measured, captured = self._measure_receipt(
            receipt,
            capture_offset=0,
            capture_length=receipt.size_bytes,
        )
        if captured is None:  # pragma: no cover - capture was requested
            raise CorruptGCSImmutableObject("bounded GCS probe captured no bytes")
        return captured, measured

    def verify(self, receipt: GCSPhysicalObjectReceiptV1) -> GCSPhysicalObjectReceiptV1:
        return self._measure_receipt(receipt)[0]

    def read_exact(
        self,
        receipt: GCSPhysicalObjectReceiptV1,
        *,
        offset: int,
        length: int,
        maximum_bytes: int,
    ) -> bytes:
        self._validate_receipt(receipt)
        if type(offset) is not int or offset < 0:
            raise GCSImmutableByteStoreError("offset must be a non-negative integer")
        if type(length) is not int or length <= 0:
            raise GCSImmutableByteStoreError("length must be a positive integer")
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise GCSImmutableByteStoreError("maximum_bytes must be a positive integer")
        if length > maximum_bytes:
            raise GCSImmutableByteStoreError("requested GCS read exceeds its bound")
        if offset + length > receipt.size_bytes:
            raise GCSImmutableByteStoreError(
                "requested GCS read exceeds immutable object size"
            )
        _measured, captured = self._measure_receipt(
            receipt,
            capture_offset=offset,
            capture_length=length,
        )
        if captured is None:  # pragma: no cover - capture was requested
            raise CorruptGCSImmutableObject("exact GCS read captured no bytes")
        return captured
