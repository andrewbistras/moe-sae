#!/usr/bin/env python3
"""Publish one derived R1 runtime as an immutable regional GCS object.

The repacker deliberately stops at a local deterministic archive.  This
one-shot publisher is the narrow bridge from that archive to one direct GCS
object.  It never lists, composes, rewrites, copies, deletes, or overwrites.

The local archive is measured from one ``O_NOFOLLOW`` descriptor before any
cloud mutation.  A create-only upload supplies the precomputed CRC32C to GCS
and uses google-cloud-storage's generation-conditional retry policy.  A
successful or ambiguous upload is accepted only after the exact live object
has been re-read and matches the complete expected identity.  The canonical
receipt is created last and never replaced.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import secrets
import stat
import sys
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import google_crc32c

RESULT_SCHEMA = "moe-sae.r1.derived-runtime-result.v1"
RECEIPT_SCHEMA = "moe-sae.r1.derived-runtime-publication-receipt.v1"
STORAGE_CLIENT_VERSION = "3.13.0"
STORAGE_CONTRACT = (
    "gcs-direct-create-only-generation-precomputed-crc32c-derived-runtime-v1"
)
CONTENT_TYPE = "application/gzip"
CHUNK_BYTES = 8 * 1024 * 1024
MAX_RESULT_BYTES = 64 * 1024
MAX_ARCHIVE_BYTES = 12 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 250_000
MAX_GCS_LONG = (1 << 63) - 1

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]")
_REGION = re.compile(r"[a-z]+(?:-[a-z0-9]+)+[0-9]")
_OBJECT_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
_AMBIGUOUS_STATUSES = {408, 412, 429, 500, 502, 503, 504}

_METADATA_SCHEMA_KEY = "moe-sae-r1-derived-runtime-schema"
_METADATA_SHA256_KEY = "moe-sae-r1-derived-runtime-sha256"
_METADATA_SIZE_KEY = "moe-sae-r1-derived-runtime-size-bytes"
_METADATA_MEMBER_COUNT_KEY = "moe-sae-r1-derived-runtime-member-count"
_METADATA_COMMIT_KEY = "moe-sae-r1-derived-runtime-repository-commit"
_METADATA_RESULT_KEY = "moe-sae-r1-derived-runtime-result-sha256"
_METADATA_ATTESTATION_KEY = "moe-sae-r1-derived-runtime-attestation-sha256"
_METADATA_SCHEMA = "v1"


class PublicationError(RuntimeError):
    """The requested publication cannot produce an authoritative receipt."""


class PublicationConflict(PublicationError):
    """The content-addressed path contains an incompatible live object."""


@dataclass(frozen=True, slots=True)
class RepackResult:
    raw_sha256: str
    repacker_sha256: str
    repository_commit: str
    derived_attestation_sha256: str
    archive_sha256: str
    archive_size_bytes: int
    archive_member_count: int


@dataclass(frozen=True, slots=True)
class ArchiveMeasurement:
    sha256: str
    crc32c: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class PublishRequest:
    archive: Path
    result: Path
    destination_uri: str
    expected_region: str
    receipt: Path
    timeout_seconds: float = 120.0
    resolution_attempts: int = 3


@dataclass(frozen=True, slots=True)
class _ObjectIdentity:
    generation: int
    metageneration: int
    size_bytes: int
    crc32c: str
    metadata: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _BucketIdentity:
    region: str
    metageneration: int


@dataclass(frozen=True, slots=True)
class _ReceiptDirectory:
    descriptor: int
    path: Path
    name: str
    observed: os.stat_result


def _canonical_bytes(value: object) -> bytes:
    try:
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
    except (TypeError, ValueError) as error:
        raise PublicationError("value cannot be encoded as canonical JSON") from error


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise PublicationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_object(
    value: object,
    *,
    name: str,
    fields: frozenset[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PublicationError(f"{name} must be an object")
    if set(value) != fields:
        raise PublicationError(f"{name} has missing or unknown fields")
    return value


def _string(
    value: object,
    *,
    name: str,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if type(value) is not str or not value:
        raise PublicationError(f"{name} must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise PublicationError(f"{name} has an invalid format")
    return value


def _integer(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise PublicationError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def parse_repack_result(raw: bytes) -> RepackResult:
    """Parse the exact canonical result emitted by the pinned repacker."""

    if type(raw) is not bytes or not raw or len(raw) > MAX_RESULT_BYTES:
        raise PublicationError("repacker result exceeds its byte contract")
    try:
        value = json.loads(raw, object_pairs_hook=_strict_pairs)
    except PublicationError:
        raise
    except (UnicodeDecodeError, ValueError) as error:
        raise PublicationError("repacker result is not strict JSON") from error
    if _canonical_bytes(value) != raw:
        raise PublicationError("repacker result is not canonical JSON")
    top = _strict_object(
        value,
        name="repacker result",
        fields=frozenset(
            {
                "schema",
                "production_qualified",
                "repacker_sha256",
                "repository_commit",
                "derived_attestation_sha256",
                "archive",
            }
        ),
    )
    if top["schema"] != RESULT_SCHEMA or top["production_qualified"] is not False:
        raise PublicationError("repacker result schema or qualification drifted")
    archive = _strict_object(
        top["archive"],
        name="repacker archive",
        fields=frozenset(
            {
                "media_type",
                "format",
                "sha256",
                "size_bytes",
                "member_count",
                "expanded_regular_bytes",
                "max_member_name_bytes",
                "max_symlink_target_bytes",
                "publication",
            }
        ),
    )
    if (
        archive["media_type"] != CONTENT_TYPE
        or archive["format"] != "GNU-tar+gzip"
        or archive["publication"] != "external-generation-pinned-regional-object"
    ):
        raise PublicationError("repacker archive contract drifted")
    size_bytes = _integer(
        archive["size_bytes"],
        name="archive.size_bytes",
        minimum=1,
        maximum=MAX_ARCHIVE_BYTES,
    )
    _integer(
        archive["member_count"],
        name="archive.member_count",
        minimum=1,
        maximum=MAX_ARCHIVE_MEMBERS,
    )
    _integer(
        archive["expanded_regular_bytes"],
        name="archive.expanded_regular_bytes",
        minimum=1,
        maximum=MAX_ARCHIVE_BYTES,
    )
    _integer(
        archive["max_member_name_bytes"],
        name="archive.max_member_name_bytes",
        minimum=1,
        maximum=1024 * 1024,
    )
    _integer(
        archive["max_symlink_target_bytes"],
        name="archive.max_symlink_target_bytes",
        minimum=0,
        maximum=1024 * 1024,
    )
    return RepackResult(
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        repacker_sha256=_string(
            top["repacker_sha256"],
            name="repacker_sha256",
            pattern=_SHA256,
        ),
        repository_commit=_string(
            top["repository_commit"],
            name="repository_commit",
            pattern=_REVISION,
        ),
        derived_attestation_sha256=_string(
            top["derived_attestation_sha256"],
            name="derived_attestation_sha256",
            pattern=_SHA256,
        ),
        archive_sha256=_string(
            archive["sha256"],
            name="archive.sha256",
            pattern=_SHA256,
        ),
        archive_size_bytes=size_bytes,
        archive_member_count=int(archive["member_count"]),
    )


def _parse_destination(uri: str, archive_sha256: str) -> tuple[str, str]:
    if type(uri) is not str or not uri.startswith("gs://"):
        raise PublicationError("destination must be a gs:// object URI")
    bucket, separator, object_name = uri[5:].partition("/")
    if (
        not separator
        or _BUCKET.fullmatch(bucket) is None
        or object_name != f"runtime/{archive_sha256}.tar.gz"
        or len(object_name.encode("ascii")) > 1024
        or any(
            _OBJECT_COMPONENT.fullmatch(component) is None
            for component in object_name.split("/")
        )
    ):
        raise PublicationError("destination must be gs://BUCKET/runtime/SHA256.tar.gz")
    return bucket, object_name


def _normalized_absolute(path: Path, *, name: str) -> Path:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path != Path(os.path.abspath(path))
    ):
        raise PublicationError(f"{name} must be a normalized absolute path")
    return path


def _open_readonly_regular(
    stack: ExitStack,
    path: Path,
    *,
    name: str,
) -> tuple[BinaryIO, os.stat_result]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise PublicationError(f"cannot open {name} without following links") from error
    stream = stack.enter_context(os.fdopen(descriptor, "rb"))
    observed = os.fstat(stream.fileno())
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) & 0o222
    ):
        raise PublicationError(f"{name} must be one read-only regular file")
    return stream, observed


def _require_unchanged(
    stream: BinaryIO,
    observed: os.stat_result,
    *,
    name: str,
) -> None:
    after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns")
    if any(getattr(after, field) != getattr(observed, field) for field in fields):
        raise PublicationError(f"{name} changed during publication")
    # ctime is checked separately because it is not available on every
    # alternative Python platform supported by static unit tests.
    if getattr(after, "st_ctime_ns", None) != getattr(observed, "st_ctime_ns", None):
        raise PublicationError(f"{name} changed during publication")


def _measure_archive(
    stream: BinaryIO,
    *,
    expected_size_bytes: int,
) -> ArchiveMeasurement:
    stream.seek(0)
    sha256 = hashlib.sha256()
    crc32c = google_crc32c.Checksum()
    measured_size = 0
    while chunk := stream.read(CHUNK_BYTES):
        if type(chunk) is not bytes or len(chunk) > CHUNK_BYTES:
            raise PublicationError("archive reader violated its bounded byte contract")
        measured_size += len(chunk)
        if measured_size > expected_size_bytes:
            raise PublicationError("archive is larger than the repacker result")
        sha256.update(chunk)
        crc32c.update(chunk)
    if measured_size != expected_size_bytes:
        raise PublicationError("archive size differs from the repacker result")
    return ArchiveMeasurement(
        sha256=sha256.hexdigest(),
        crc32c=base64.b64encode(crc32c.digest()).decode("ascii"),
        size_bytes=measured_size,
    )


class _HashingDownloadWriter:
    """Measure a bounded raw download without retaining object bytes."""

    def __init__(self, *, expected_size_bytes: int) -> None:
        self._expected_size_bytes = expected_size_bytes
        self._sha256 = hashlib.sha256()
        self._crc32c = google_crc32c.Checksum()
        self._position = 0

    def write(self, value: bytes | bytearray | memoryview) -> int:
        try:
            view = memoryview(value).cast("B")
        except (TypeError, ValueError) as error:
            raise PublicationConflict(
                "GCS client produced a non-contiguous download chunk"
            ) from error
        try:
            count = len(view)
            if count > CHUNK_BYTES:
                raise PublicationConflict(
                    "GCS client produced an oversized download chunk"
                )
            if self._position + count > self._expected_size_bytes:
                raise PublicationConflict("GCS download exceeded the expected size")
            self._sha256.update(view)
            if type(value) is bytes:
                self._crc32c.update(value)
            else:
                self._crc32c.update(view.tobytes())
            self._position += count
            return count
        finally:
            view.release()

    def tell(self) -> int:
        return self._position

    def writable(self) -> bool:
        return True

    def flush(self) -> None:
        return None

    def finish(self) -> ArchiveMeasurement:
        if self._position != self._expected_size_bytes:
            raise PublicationConflict("GCS download ended before the expected size")
        return ArchiveMeasurement(
            sha256=self._sha256.hexdigest(),
            crc32c=base64.b64encode(self._crc32c.digest()).decode("ascii"),
            size_bytes=self._position,
        )


def _metadata(
    result: RepackResult,
    measurement: ArchiveMeasurement,
) -> dict[str, str]:
    return {
        _METADATA_SCHEMA_KEY: _METADATA_SCHEMA,
        _METADATA_SHA256_KEY: measurement.sha256,
        _METADATA_SIZE_KEY: str(measurement.size_bytes),
        _METADATA_MEMBER_COUNT_KEY: str(result.archive_member_count),
        _METADATA_COMMIT_KEY: result.repository_commit,
        _METADATA_RESULT_KEY: result.raw_sha256,
        _METADATA_ATTESTATION_KEY: result.derived_attestation_sha256,
    }


def _status_code(error: BaseException) -> int | None:
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


def _remote_integer(value: object, *, name: str, uri: str) -> int:
    if type(value) is int:
        result = value
    elif type(value) is str and re.fullmatch(r"[1-9][0-9]{0,18}", value):
        result = int(value)
    else:
        raise PublicationConflict(f"{name} is invalid for {uri}")
    if not 1 <= result <= MAX_GCS_LONG:
        raise PublicationConflict(f"{name} is invalid for {uri}")
    return result


def _require_blob_address(
    blob: Any,
    *,
    client: Any,
    bucket_name: str,
    object_name: str,
) -> None:
    blob_bucket = getattr(blob, "bucket", None)
    if (
        getattr(blob, "name", None) != object_name
        or getattr(blob_bucket, "name", None) != bucket_name
        or getattr(blob_bucket, "client", getattr(blob_bucket, "_client", None))
        is not client
    ):
        raise PublicationError("GCS blob address drifted from the requested object")


def _object_identity(
    blob: Any,
    *,
    client: Any,
    bucket_name: str,
    object_name: str,
    uri: str,
    measurement: ArchiveMeasurement,
    expected_metadata: Mapping[str, str],
) -> _ObjectIdentity:
    _require_blob_address(
        blob,
        client=client,
        bucket_name=bucket_name,
        object_name=object_name,
    )
    generation = _remote_integer(blob.generation, name="generation", uri=uri)
    metageneration = _remote_integer(
        blob.metageneration,
        name="metageneration",
        uri=uri,
    )
    size_bytes = _remote_integer(blob.size, name="size", uri=uri)
    metadata = getattr(blob, "metadata", None)
    if not isinstance(metadata, Mapping) or any(
        type(key) is not str or type(value) is not str
        for key, value in metadata.items()
    ):
        raise PublicationConflict(f"custom metadata is invalid for {uri}")
    metadata_copy = dict(metadata)
    if (
        size_bytes != measurement.size_bytes
        or getattr(blob, "crc32c", None) != measurement.crc32c
        or getattr(blob, "content_type", None) != CONTENT_TYPE
        or getattr(blob, "content_encoding", None) is not None
        or getattr(blob, "component_count", None) is not None
        or metadata_copy != dict(expected_metadata)
    ):
        raise PublicationConflict(f"live object identity is incompatible: {uri}")
    return _ObjectIdentity(
        generation=generation,
        metageneration=metageneration,
        size_bytes=size_bytes,
        crc32c=measurement.crc32c,
        metadata=metadata_copy,
    )


def _new_blob(
    client: Any,
    *,
    bucket_name: str,
    object_name: str,
    generation: int | None = None,
) -> Any:
    bucket = client.bucket(bucket_name)
    if (
        getattr(bucket, "name", None) != bucket_name
        or getattr(bucket, "client", getattr(bucket, "_client", None)) is not client
    ):
        raise PublicationError("GCS bucket address drifted")
    blob = bucket.blob(
        object_name,
        chunk_size=CHUNK_BYTES,
        generation=generation,
    )
    _require_blob_address(
        blob,
        client=client,
        bucket_name=bucket_name,
        object_name=object_name,
    )
    return blob


def _head(
    client: Any,
    *,
    bucket_name: str,
    object_name: str,
    read_retry_policy: Any,
    timeout_seconds: float,
    generation: int | None = None,
    metageneration: int | None = None,
    absent_ok: bool = False,
) -> Any | None:
    blob = _new_blob(
        client,
        bucket_name=bucket_name,
        object_name=object_name,
        generation=generation,
    )
    kwargs: dict[str, object] = {
        "projection": "noAcl",
        "retry": read_retry_policy,
        "timeout": timeout_seconds,
    }
    if generation is not None:
        kwargs["if_generation_match"] = generation
    if metageneration is not None:
        kwargs["if_metageneration_match"] = metageneration
    try:
        blob.reload(**kwargs)
    except Exception as error:
        if absent_ok and _status_code(error) == 404:
            return None
        raise PublicationError("cannot read exact GCS object metadata") from error
    return blob


def _measure_exact_generation(
    blob: Any,
    *,
    identity: _ObjectIdentity,
    expected: ArchiveMeasurement,
    read_retry_policy: Any,
    timeout_seconds: float,
) -> None:
    writer = _HashingDownloadWriter(expected_size_bytes=expected.size_bytes)
    try:
        blob.download_to_file(
            writer,
            raw_download=True,
            if_generation_match=identity.generation,
            if_metageneration_match=identity.metageneration,
            checksum=None,
            retry=read_retry_policy,
            timeout=timeout_seconds,
            single_shot_download=False,
        )
    except PublicationConflict:
        raise
    except Exception as error:
        if _status_code(error) in {404, 412}:
            raise PublicationConflict(
                "exact GCS generation changed during byte verification"
            ) from error
        raise PublicationError("cannot read exact GCS object bytes") from error
    measured = writer.finish()
    if measured != expected:
        raise PublicationConflict(
            "exact GCS generation bytes fail SHA-256, CRC32C, or size verification"
        )


def _verify_live_object(
    client: Any,
    *,
    bucket_name: str,
    object_name: str,
    uri: str,
    measurement: ArchiveMeasurement,
    expected_metadata: Mapping[str, str],
    read_retry_policy: Any,
    timeout_seconds: float,
    initial_blob: Any,
) -> _ObjectIdentity:
    initial = _object_identity(
        initial_blob,
        client=client,
        bucket_name=bucket_name,
        object_name=object_name,
        uri=uri,
        measurement=measurement,
        expected_metadata=expected_metadata,
    )
    exact_blob = _head(
        client,
        bucket_name=bucket_name,
        object_name=object_name,
        generation=initial.generation,
        metageneration=initial.metageneration,
        read_retry_policy=read_retry_policy,
        timeout_seconds=timeout_seconds,
    )
    assert exact_blob is not None
    exact = _object_identity(
        exact_blob,
        client=client,
        bucket_name=bucket_name,
        object_name=object_name,
        uri=uri,
        measurement=measurement,
        expected_metadata=expected_metadata,
    )
    if exact != initial:
        raise PublicationConflict(f"exact object metadata changed: {uri}")
    _measure_exact_generation(
        exact_blob,
        identity=exact,
        expected=measurement,
        read_retry_policy=read_retry_policy,
        timeout_seconds=timeout_seconds,
    )
    exact_after_blob = _head(
        client,
        bucket_name=bucket_name,
        object_name=object_name,
        generation=initial.generation,
        metageneration=initial.metageneration,
        read_retry_policy=read_retry_policy,
        timeout_seconds=timeout_seconds,
    )
    assert exact_after_blob is not None
    exact_after = _object_identity(
        exact_after_blob,
        client=client,
        bucket_name=bucket_name,
        object_name=object_name,
        uri=uri,
        measurement=measurement,
        expected_metadata=expected_metadata,
    )
    live_blob = _head(
        client,
        bucket_name=bucket_name,
        object_name=object_name,
        read_retry_policy=read_retry_policy,
        timeout_seconds=timeout_seconds,
    )
    if live_blob is None:  # pragma: no cover - absent_ok is false
        raise PublicationError("live GCS object vanished")
    live = _object_identity(
        live_blob,
        client=client,
        bucket_name=bucket_name,
        object_name=object_name,
        uri=uri,
        measurement=measurement,
        expected_metadata=expected_metadata,
    )
    if exact_after != initial or live != initial:
        raise PublicationConflict(f"live object generation changed: {uri}")
    return exact


def _resolve_expected(
    client: Any,
    *,
    request: PublishRequest,
    bucket_name: str,
    object_name: str,
    measurement: ArchiveMeasurement,
    expected_metadata: Mapping[str, str],
    read_retry_policy: Any,
) -> _ObjectIdentity:
    last_error: PublicationError | None = None
    for _ in range(request.resolution_attempts):
        try:
            blob = _head(
                client,
                bucket_name=bucket_name,
                object_name=object_name,
                read_retry_policy=read_retry_policy,
                timeout_seconds=request.timeout_seconds,
                absent_ok=True,
            )
            if blob is None:
                continue
            return _verify_live_object(
                client,
                bucket_name=bucket_name,
                object_name=object_name,
                uri=request.destination_uri,
                measurement=measurement,
                expected_metadata=expected_metadata,
                read_retry_policy=read_retry_policy,
                timeout_seconds=request.timeout_seconds,
                initial_blob=blob,
            )
        except PublicationConflict:
            raise
        except PublicationError as error:
            last_error = error
    raise PublicationError(
        "cannot reconcile the create-only upload to an exact live generation"
    ) from last_error


def _validate_request(request: PublishRequest) -> None:
    archive = _normalized_absolute(request.archive, name="archive")
    result = _normalized_absolute(request.result, name="result")
    receipt = _normalized_absolute(request.receipt, name="receipt")
    if len({archive, result, receipt}) != 3:
        raise PublicationError("archive, result, and receipt must be distinct")
    if _OBJECT_COMPONENT.fullmatch(receipt.name) is None:
        raise PublicationError("receipt name must be one safe path component")
    if (
        type(request.expected_region) is not str
        or _REGION.fullmatch(request.expected_region) is None
    ):
        raise PublicationError("expected_region must be a lowercase GCS region")
    if isinstance(request.timeout_seconds, bool) or not isinstance(
        request.timeout_seconds, (int, float)
    ):
        raise PublicationError("timeout_seconds must be finite and positive")
    timeout = float(request.timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise PublicationError("timeout_seconds must be finite and positive")
    if (
        type(request.resolution_attempts) is not int
        or not 1 <= request.resolution_attempts <= 10
    ):
        raise PublicationError("resolution_attempts must be in [1, 10]")


def _require_generation_conditional_retry(policy: Any) -> None:
    resolver = getattr(policy, "get_retry_policy_if_conditions_met", None)
    if not callable(resolver):
        raise PublicationError("upload retry policy is not generation-conditional")
    without_precondition = resolver(
        query_params={
            "ifGenerationMatch": None,
            "ifMetagenerationMatch": None,
        }
    )
    with_create_precondition = resolver(
        query_params={
            "ifGenerationMatch": 0,
            "ifMetagenerationMatch": None,
        }
    )
    if without_precondition is not None or with_create_precondition is None:
        raise PublicationError("upload retry policy is not generation-conditional")


def _verify_bucket(
    client: Any,
    *,
    bucket_name: str,
    expected_region: str,
    read_retry_policy: Any,
    timeout_seconds: float,
) -> _BucketIdentity:
    bucket = client.bucket(bucket_name)
    if (
        getattr(bucket, "name", None) != bucket_name
        or getattr(bucket, "client", getattr(bucket, "_client", None)) is not client
    ):
        raise PublicationError("GCS bucket address drifted")
    try:
        bucket.reload(
            projection="noAcl",
            retry=read_retry_policy,
            timeout=timeout_seconds,
        )
    except Exception as error:
        raise PublicationError("cannot attest destination bucket") from error
    location = getattr(bucket, "location", None)
    if type(location) is not str or location.lower() != expected_region:
        raise PublicationError("destination bucket is not in the expected region")
    return _BucketIdentity(
        region=expected_region,
        metageneration=_remote_integer(
            getattr(bucket, "metageneration", None),
            name="bucket metageneration",
            uri=f"gs://{bucket_name}",
        ),
    )


def _open_receipt_directory(path: Path) -> _ReceiptDirectory:
    parent = path.parent
    try:
        before = parent.lstat()
        descriptor = os.open(
            parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise PublicationError(
            "receipt parent must be an existing real directory"
        ) from error
    try:
        observed = os.fstat(descriptor)
    except OSError:
        os.close(descriptor)
        raise
    mode = stat.S_IMODE(observed.st_mode)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_dev != before.st_dev
        or observed.st_ino != before.st_ino
        or observed.st_uid != os.geteuid()
        or mode & 0o022
    ):
        os.close(descriptor)
        raise PublicationError(
            "receipt parent must be a stable, uid-owned private directory"
        )
    try:
        os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError as error:
        os.close(descriptor)
        raise PublicationError("cannot attest the fresh receipt name") from error
    else:
        os.close(descriptor)
        raise PublicationError("receipt path must be fresh")
    return _ReceiptDirectory(
        descriptor=descriptor,
        path=parent,
        name=path.name,
        observed=observed,
    )


def _require_receipt_directory_stable(directory: _ReceiptDirectory) -> None:
    try:
        current = directory.path.lstat()
    except OSError as error:
        raise PublicationError("receipt parent pathname changed") from error
    if (
        not stat.S_ISDIR(current.st_mode)
        or current.st_dev != directory.observed.st_dev
        or current.st_ino != directory.observed.st_ino
        or current.st_uid != directory.observed.st_uid
        or stat.S_IMODE(current.st_mode) != stat.S_IMODE(directory.observed.st_mode)
    ):
        raise PublicationError("receipt parent pathname changed")


def _write_all(descriptor: int, raw: bytes) -> None:
    position = 0
    while position < len(raw):
        written = os.write(descriptor, raw[position:])
        if written <= 0:
            raise PublicationError("receipt write made no progress")
        position += written


def _write_receipt_last(directory: _ReceiptDirectory, raw: bytes) -> None:
    temporary_name = f".{directory.name}.{secrets.token_hex(16)}.tmp"
    temporary_exists = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
            dir_fd=directory.descriptor,
        )
        temporary_exists = True
        try:
            _write_all(descriptor, raw)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            temporary_stat = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary_name,
                directory.name,
                src_dir_fd=directory.descriptor,
                dst_dir_fd=directory.descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise PublicationError("refusing to replace an existing receipt") from error
        os.unlink(temporary_name, dir_fd=directory.descriptor)
        temporary_exists = False
        os.fsync(directory.descriptor)
        final_descriptor = os.open(
            directory.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory.descriptor,
        )
        try:
            final_stat = os.fstat(final_descriptor)
            if (
                not stat.S_ISREG(final_stat.st_mode)
                or final_stat.st_dev != temporary_stat.st_dev
                or final_stat.st_ino != temporary_stat.st_ino
                or final_stat.st_nlink != 1
                or stat.S_IMODE(final_stat.st_mode) != 0o444
                or final_stat.st_size != len(raw)
            ):
                raise PublicationError(
                    "final receipt inode differs from its staging file"
                )
            observed_raw = b""
            while len(observed_raw) <= len(raw):
                chunk = os.read(
                    final_descriptor,
                    min(CHUNK_BYTES, len(raw) + 1 - len(observed_raw)),
                )
                if not chunk:
                    break
                observed_raw += chunk
            if observed_raw != raw:
                raise PublicationError("final receipt bytes changed")
            after = os.fstat(final_descriptor)
            if (
                after.st_dev != final_stat.st_dev
                or after.st_ino != final_stat.st_ino
                or after.st_size != final_stat.st_size
                or after.st_mtime_ns != final_stat.st_mtime_ns
                or getattr(after, "st_ctime_ns", None)
                != getattr(final_stat, "st_ctime_ns", None)
            ):
                raise PublicationError("final receipt changed during verification")
        finally:
            os.close(final_descriptor)
        path_stat = os.stat(
            directory.name,
            dir_fd=directory.descriptor,
            follow_symlinks=False,
        )
        if (
            path_stat.st_dev != final_stat.st_dev
            or path_stat.st_ino != final_stat.st_ino
        ):
            raise PublicationError("final receipt name changed during verification")
        _require_receipt_directory_stable(directory)
    finally:
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=directory.descriptor)
            except FileNotFoundError:
                pass


def publish(
    request: PublishRequest,
    *,
    client: Any,
    upload_retry_policy: Any,
    read_retry_policy: Any,
    client_version: str,
) -> Mapping[str, object]:
    """Publish or reconcile one exact archive and emit its receipt last."""

    _validate_request(request)
    if client_version != STORAGE_CLIENT_VERSION:
        raise PublicationError(
            f"publisher requires google-cloud-storage=={STORAGE_CLIENT_VERSION}"
        )
    if upload_retry_policy is None or read_retry_policy is None:
        raise PublicationError("bounded upload and read retry policies are required")
    _require_generation_conditional_retry(upload_retry_policy)

    with ExitStack() as stack:
        receipt_directory = _open_receipt_directory(request.receipt)
        stack.callback(os.close, receipt_directory.descriptor)
        result_stream, result_stat = _open_readonly_regular(
            stack,
            request.result,
            name="repacker result",
        )
        if not 1 <= result_stat.st_size <= MAX_RESULT_BYTES:
            raise PublicationError("repacker result size is out of bounds")
        result_raw = result_stream.read(MAX_RESULT_BYTES + 1)
        result = parse_repack_result(result_raw)
        archive_stream, archive_stat = _open_readonly_regular(
            stack,
            request.archive,
            name="runtime archive",
        )
        if archive_stat.st_size != result.archive_size_bytes:
            raise PublicationError("runtime archive stat differs from repacker result")
        measurement = _measure_archive(
            archive_stream,
            expected_size_bytes=result.archive_size_bytes,
        )
        if measurement.sha256 != result.archive_sha256:
            raise PublicationError(
                "runtime archive SHA-256 differs from repacker result"
            )
        bucket_name, object_name = _parse_destination(
            request.destination_uri,
            measurement.sha256,
        )
        expected_metadata = _metadata(result, measurement)
        _require_unchanged(result_stream, result_stat, name="repacker result")
        _require_unchanged(archive_stream, archive_stat, name="runtime archive")

        bucket_identity = _verify_bucket(
            client,
            bucket_name=bucket_name,
            expected_region=request.expected_region,
            read_retry_policy=read_retry_policy,
            timeout_seconds=request.timeout_seconds,
        )
        current = _head(
            client,
            bucket_name=bucket_name,
            object_name=object_name,
            read_retry_policy=read_retry_policy,
            timeout_seconds=request.timeout_seconds,
            absent_ok=True,
        )
        disposition = "reconciled-existing"
        if current is None:
            blob = _new_blob(
                client,
                bucket_name=bucket_name,
                object_name=object_name,
            )
            blob.metadata = dict(expected_metadata)
            archive_stream.seek(0)
            try:
                blob.upload_from_file(
                    archive_stream,
                    rewind=False,
                    size=measurement.size_bytes,
                    content_type=CONTENT_TYPE,
                    if_generation_match=0,
                    checksum=(
                        "crc32c" if measurement.size_bytes <= CHUNK_BYTES else None
                    ),
                    crc32c_checksum_value=measurement.crc32c,
                    retry=upload_retry_policy,
                    timeout=request.timeout_seconds,
                )
            except Exception as error:
                status = _status_code(error)
                if status not in _AMBIGUOUS_STATUSES and status is not None:
                    raise PublicationError("create-only GCS upload failed") from error
                identity = _resolve_expected(
                    client,
                    request=request,
                    bucket_name=bucket_name,
                    object_name=object_name,
                    measurement=measurement,
                    expected_metadata=expected_metadata,
                    read_retry_policy=read_retry_policy,
                )
            else:
                if archive_stream.tell() != measurement.size_bytes:
                    raise PublicationError(
                        "GCS client did not consume the exact archive"
                    )
                identity = _verify_live_object(
                    client,
                    bucket_name=bucket_name,
                    object_name=object_name,
                    uri=request.destination_uri,
                    measurement=measurement,
                    expected_metadata=expected_metadata,
                    read_retry_policy=read_retry_policy,
                    timeout_seconds=request.timeout_seconds,
                    initial_blob=blob,
                )
                disposition = "created"
        else:
            identity = _verify_live_object(
                client,
                bucket_name=bucket_name,
                object_name=object_name,
                uri=request.destination_uri,
                measurement=measurement,
                expected_metadata=expected_metadata,
                read_retry_policy=read_retry_policy,
                timeout_seconds=request.timeout_seconds,
                initial_blob=current,
            )

        bucket_after = _verify_bucket(
            client,
            bucket_name=bucket_name,
            expected_region=request.expected_region,
            read_retry_policy=read_retry_policy,
            timeout_seconds=request.timeout_seconds,
        )
        if bucket_after != bucket_identity:
            raise PublicationConflict(
                "destination bucket location or metageneration changed"
            )
        _require_unchanged(result_stream, result_stat, name="repacker result")
        _require_unchanged(archive_stream, archive_stat, name="runtime archive")
        receipt: Mapping[str, object] = {
            "schema": RECEIPT_SCHEMA,
            "production_qualified": False,
            "storage_contract": STORAGE_CONTRACT,
            "client": f"google-cloud-storage=={STORAGE_CLIENT_VERSION}",
            "disposition": disposition,
            "repository_commit": result.repository_commit,
            "repacker_sha256": result.repacker_sha256,
            "repacker_result_sha256": result.raw_sha256,
            "derived_attestation_sha256": result.derived_attestation_sha256,
            "bucket": {
                "name": bucket_name,
                "region": bucket_identity.region,
                "metageneration": bucket_identity.metageneration,
            },
            "archive": {
                "sha256": measurement.sha256,
                "size_bytes": measurement.size_bytes,
                "member_count": result.archive_member_count,
                "crc32c": measurement.crc32c,
                "media_type": CONTENT_TYPE,
            },
            "object": {
                "uri": request.destination_uri,
                "region": request.expected_region,
                "generation": identity.generation,
                "metageneration": identity.metageneration,
                "size_bytes": identity.size_bytes,
                "crc32c": identity.crc32c,
                "content_type": CONTENT_TYPE,
                "component_count": None,
                "metadata": dict(identity.metadata),
            },
        }
        raw_receipt = _canonical_bytes(receipt)
        _write_receipt_last(receipt_directory, raw_receipt)
        return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--destination-uri", required=True)
    parser.add_argument("--expected-region", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--project")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--resolution-attempts", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        from google.cloud import storage
        from google.cloud.storage.retry import (
            DEFAULT_RETRY,
            DEFAULT_RETRY_IF_GENERATION_SPECIFIED,
        )
    except ImportError as error:
        raise PublicationError(
            "publisher requires google-cloud-storage==3.13.0"
        ) from error
    if getattr(storage, "__version__", None) != STORAGE_CLIENT_VERSION:
        raise PublicationError(
            f"publisher requires google-cloud-storage=={STORAGE_CLIENT_VERSION}"
        )
    request = PublishRequest(
        archive=args.archive,
        result=args.result,
        destination_uri=args.destination_uri,
        expected_region=args.expected_region,
        receipt=args.receipt,
        timeout_seconds=args.timeout_seconds,
        resolution_attempts=args.resolution_attempts,
    )
    receipt = publish(
        request,
        client=storage.Client(project=args.project),
        upload_retry_policy=DEFAULT_RETRY_IF_GENERATION_SPECIFIED,
        read_retry_policy=DEFAULT_RETRY.with_timeout(args.timeout_seconds),
        client_version=storage.__version__,
    )
    sys.stdout.buffer.write(_canonical_bytes(receipt))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PublicationError as error:
        print(f"[r1-runtime-publish] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
