"""Generation-fenced Google Cloud Storage for R1 controller JSON.

The controller stores only small canonical JSON documents through this
backend.  Every mutation uses a GCS object-generation precondition; immutable
documents use ``if_generation_match=0`` and the single mutable campaign head
uses the generation returned by its preceding read.  Reads download bytes from
the exact generation and metageneration whose metadata was inspected.

This module intentionally has no object-listing operation.  Campaign replay
must follow the accepted head and its explicit content-addressed references,
never infer authority from a bucket listing.

``google-cloud-storage`` is imported only when no client is injected.  The R1
protocol and its deterministic in-memory oracle therefore remain usable in a
minimal CPU environment without cloud dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

from .control_store import (
    CASConflict,
    ControlStoreError,
    ImmutableConflict,
    VersionedJSON,
    _canonicalize_value,
)


_CONTENT_TYPE = "application/json"
_METADATA_SCHEMA_KEY = "moe-sae-r1-control-schema"
_METADATA_KIND_KEY = "moe-sae-r1-control-kind"
_METADATA_SHA256_KEY = "moe-sae-r1-control-sha256"
_METADATA_SCHEMA = "v1"
_MUTABLE = "mutable"
_IMMUTABLE = "immutable"


class GCSControlStoreError(ControlStoreError):
    """A GCS control-plane operation failed without a known CAS outcome."""


class CorruptControlObject(ControlStoreError):
    """A control object does not satisfy the canonical byte/metadata contract."""


@dataclass(frozen=True, slots=True)
class _StoredJSON:
    version: VersionedJSON
    kind: str
    metageneration: int


def _parse_gcs_object_uri(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri or uri != uri.strip():
        raise ValueError("GCS control URI must be a non-empty string")
    if any(ord(character) < 0x20 for character in uri):
        raise ValueError("GCS control URI must not contain control characters")
    if not uri.startswith("gs://"):
        raise ValueError("GCS control URI must start with gs://")
    bucket, separator, name = uri[5:].partition("/")
    if not separator or not bucket or not name or name.endswith("/"):
        raise ValueError(f"GCS control URI must name one object, got {uri!r}")
    if "//" in name or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError(f"unsafe GCS control object name in {uri!r}")
    return bucket, name


def _status_code(error: BaseException) -> int | None:
    """Extract an HTTP status without importing google-api-core in unit tests."""

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


def _positive_remote_integer(value: Any, field: str, uri: str) -> int:
    if isinstance(value, bool):
        raise CorruptControlObject(f"{field} is invalid for {uri}")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise CorruptControlObject(f"{field} is invalid for {uri}") from error
    if result <= 0 or str(result) != str(value):
        raise CorruptControlObject(f"{field} is invalid for {uri}")
    return result


def _metadata(kind: str, digest: str) -> dict[str, str]:
    return {
        _METADATA_SCHEMA_KEY: _METADATA_SCHEMA,
        _METADATA_KIND_KEY: kind,
        _METADATA_SHA256_KEY: digest,
    }


class GCSJSONCASStore:
    """Production :class:`~moe_sae.r1.control_store.JSONCASStore` on GCS.

    The injected client need only expose ``bucket(name).blob(name)`` with the
    normal ``Blob.reload``, ``Blob.download_as_bytes``, and
    ``Blob.upload_from_string`` methods.  This narrow surface keeps unit tests
    deterministic and makes the absence of object listing structural.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        project: str | None = None,
        timeout_seconds: float = 60.0,
        consistent_read_attempts: int = 4,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive")
        if (
            isinstance(consistent_read_attempts, bool)
            or not isinstance(consistent_read_attempts, int)
            or consistent_read_attempts <= 0
        ):
            raise ValueError("consistent_read_attempts must be a positive integer")
        if project is not None and (not isinstance(project, str) or not project):
            raise ValueError("project must be a non-empty string when supplied")
        if client is None:
            try:
                from google.cloud import storage
            except ImportError as error:
                raise GCSControlStoreError(
                    "GCS control storage requires google-cloud-storage (`--extra tpu`)"
                ) from error
            client = storage.Client(project=project)
        self.client = client
        self.timeout_seconds = timeout
        self.consistent_read_attempts = consistent_read_attempts

    @staticmethod
    def _version(uri: str, value: bytes, generation: int) -> VersionedJSON:
        return VersionedJSON(
            uri=uri,
            value=value,
            generation=generation,
            sha256=hashlib.sha256(value).hexdigest(),
        )

    def _blob(self, uri: str) -> Any:
        bucket, name = _parse_gcs_object_uri(uri)
        return self.client.bucket(bucket).blob(name)

    def _read_stored(self, uri: str) -> _StoredJSON | None:
        _parse_gcs_object_uri(uri)
        last_race: BaseException | None = None
        for _attempt in range(self.consistent_read_attempts):
            blob = self._blob(uri)
            try:
                blob.reload(timeout=self.timeout_seconds)
            except Exception as error:
                if _status_code(error) == 404:
                    return None
                raise GCSControlStoreError(
                    f"cannot read GCS control metadata: {uri}"
                ) from error

            generation = _positive_remote_integer(blob.generation, "generation", uri)
            metageneration = _positive_remote_integer(
                blob.metageneration, "metageneration", uri
            )
            metadata = blob.metadata
            if not isinstance(metadata, Mapping):
                raise CorruptControlObject(f"control metadata is missing for {uri}")
            metadata_dict = dict(metadata)
            kind = metadata_dict.get(_METADATA_KIND_KEY)
            digest_metadata = metadata_dict.get(_METADATA_SHA256_KEY)
            expected_keys = {
                _METADATA_SCHEMA_KEY,
                _METADATA_KIND_KEY,
                _METADATA_SHA256_KEY,
            }
            if (
                set(metadata_dict) != expected_keys
                or metadata_dict.get(_METADATA_SCHEMA_KEY) != _METADATA_SCHEMA
                or kind not in {_MUTABLE, _IMMUTABLE}
                or not isinstance(digest_metadata, str)
            ):
                raise CorruptControlObject(
                    f"control metadata is incompatible or has unknown fields: {uri}"
                )
            if blob.content_type != _CONTENT_TYPE:
                raise CorruptControlObject(
                    f"control object has unexpected content type: {uri}"
                )

            try:
                value = blob.download_as_bytes(
                    if_generation_match=generation,
                    if_metageneration_match=metageneration,
                    checksum="crc32c",
                    timeout=self.timeout_seconds,
                )
            except Exception as error:
                if _status_code(error) in {404, 412}:
                    last_race = error
                    continue
                raise GCSControlStoreError(
                    f"cannot download GCS control bytes: {uri}"
                ) from error
            if type(value) is not bytes:
                raise CorruptControlObject(
                    f"GCS client returned non-bytes control content for {uri}"
                )
            size = _positive_remote_integer(blob.size, "size", uri)
            if size != len(value):
                raise CorruptControlObject(
                    f"control object size does not match downloaded bytes: {uri}"
                )
            digest = hashlib.sha256(value).hexdigest()
            if digest_metadata != digest:
                raise CorruptControlObject(
                    f"control object SHA-256 metadata does not match bytes: {uri}"
                )
            try:
                version = self._version(uri, value, generation)
            except (TypeError, ValueError) as error:
                raise CorruptControlObject(
                    f"control object is not canonical versioned JSON: {uri}"
                ) from error
            return _StoredJSON(version, str(kind), metageneration)

        raise GCSControlStoreError(
            f"GCS control object changed during {self.consistent_read_attempts} "
            f"generation-pinned read attempts: {uri}"
        ) from last_race

    def read(self, uri: str) -> VersionedJSON | None:
        stored = self._read_stored(uri)
        return None if stored is None else stored.version

    def _upload(
        self,
        uri: str,
        value: bytes,
        *,
        kind: str,
        if_generation_match: int,
        if_metageneration_match: int | None = None,
    ) -> VersionedJSON:
        blob = self._blob(uri)
        digest = hashlib.sha256(value).hexdigest()
        blob.metadata = _metadata(kind, digest)
        kwargs: dict[str, Any] = {
            "content_type": _CONTENT_TYPE,
            "if_generation_match": if_generation_match,
            "checksum": "crc32c",
            "timeout": self.timeout_seconds,
        }
        if if_metageneration_match is not None:
            kwargs["if_metageneration_match"] = if_metageneration_match
        blob.upload_from_string(value, **kwargs)
        generation = _positive_remote_integer(blob.generation, "generation", uri)
        if if_generation_match > 0 and generation == if_generation_match:
            raise CorruptControlObject(
                f"GCS replacement did not advance object generation: {uri}"
            )
        return self._version(uri, value, generation)

    def _current_after_precondition(self, uri: str) -> _StoredJSON | None:
        try:
            return self._read_stored(uri)
        except ControlStoreError:
            raise
        except Exception as error:  # pragma: no cover - defensive client boundary
            raise GCSControlStoreError(
                f"cannot resolve failed GCS precondition: {uri}"
            ) from error

    def create(self, uri: str, value: Any) -> VersionedJSON:
        _parse_gcs_object_uri(uri)
        encoded = _canonicalize_value(value)
        try:
            return self._upload(
                uri,
                encoded,
                kind=_MUTABLE,
                if_generation_match=0,
            )
        except ControlStoreError:
            raise
        except Exception as error:
            if _status_code(error) != 412:
                raise GCSControlStoreError(
                    f"cannot create mutable GCS control object: {uri}"
                ) from error
            current = self._current_after_precondition(uri)
            raise CASConflict(
                uri,
                expected_generation=None,
                actual_generation=(
                    None if current is None else current.version.generation
                ),
            ) from None

    def compare_and_swap(
        self,
        uri: str,
        expected_generation: int,
        value: Any,
    ) -> VersionedJSON:
        _parse_gcs_object_uri(uri)
        if (
            isinstance(expected_generation, bool)
            or not isinstance(expected_generation, int)
            or expected_generation <= 0
        ):
            raise ValueError("expected_generation must be a positive integer")
        encoded = _canonicalize_value(value)
        current = self._read_stored(uri)
        actual = None if current is None else current.version.generation
        if current is None or actual != expected_generation:
            raise CASConflict(
                uri,
                expected_generation=expected_generation,
                actual_generation=actual,
            )
        if current.kind == _IMMUTABLE:
            raise ImmutableConflict(f"immutable JSON cannot be replaced: {uri}")

        try:
            return self._upload(
                uri,
                encoded,
                kind=_MUTABLE,
                if_generation_match=expected_generation,
                if_metageneration_match=current.metageneration,
            )
        except ControlStoreError:
            raise
        except Exception as error:
            if _status_code(error) == 412:
                latest = self._current_after_precondition(uri)
                # The GCS client may safely retry a generation-conditional
                # request after losing its first response.  If the first RPC
                # committed, that internal retry receives 412.  Treat the
                # exact requested bytes at a new mutable generation as the
                # idempotent result of this CAS; a different value remains a
                # genuine race loss.
                if (
                    latest is not None
                    and latest.kind == _MUTABLE
                    and latest.version.generation != expected_generation
                    and latest.version.value == encoded
                ):
                    return latest.version
                if latest is not None and latest.kind == _IMMUTABLE:
                    raise ImmutableConflict(
                        f"immutable JSON cannot be replaced: {uri}"
                    ) from None
                raise CASConflict(
                    uri,
                    expected_generation=expected_generation,
                    actual_generation=(
                        None if latest is None else latest.version.generation
                    ),
                ) from None

            # A timeout can arrive after GCS durably accepted a conditional
            # upload.  Reconcile only this ambiguous transport outcome, and
            # only when the exact requested bytes occupy a new generation.
            try:
                latest = self._read_stored(uri)
            except ControlStoreError:
                latest = None
            if (
                latest is not None
                and latest.kind == _MUTABLE
                and latest.version.generation != expected_generation
                and latest.version.value == encoded
            ):
                return latest.version
            raise GCSControlStoreError(
                f"cannot replace mutable GCS control object: {uri}"
            ) from error

    def put_immutable(self, uri: str, value: Any) -> VersionedJSON:
        _parse_gcs_object_uri(uri)
        encoded = _canonicalize_value(value)
        try:
            return self._upload(
                uri,
                encoded,
                kind=_IMMUTABLE,
                if_generation_match=0,
            )
        except ControlStoreError:
            raise
        except Exception as error:
            precondition_failed = _status_code(error) == 412
            try:
                current = self._read_stored(uri)
            except ControlStoreError:
                if precondition_failed:
                    raise
                current = None
            if current is not None:
                if current.kind != _IMMUTABLE:
                    raise ImmutableConflict(
                        f"name already belongs to a mutable JSON document: {uri}"
                    ) from None
                if current.version.value != encoded:
                    raise ImmutableConflict(
                        f"immutable JSON exists with different content: {uri}"
                    ) from None
                return current.version
            if precondition_failed:
                raise GCSControlStoreError(
                    f"GCS immutable create lost a race but no object is readable: {uri}"
                ) from error
            raise GCSControlStoreError(
                f"cannot publish immutable GCS control object: {uri}"
            ) from error


__all__ = [
    "CorruptControlObject",
    "GCSControlStoreError",
    "GCSJSONCASStore",
]
