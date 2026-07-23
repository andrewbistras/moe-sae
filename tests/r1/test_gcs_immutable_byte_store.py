from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import base64
import hashlib
import inspect
import threading
from typing import Any, Callable

import google_crc32c
import pytest

from moe_sae.r1.gcs_immutable_byte_store import (
    MAX_APPLICATION_STREAM_SLICE_BYTES,
    MAX_IMMUTABLE_SNAPSHOT_BYTES,
    CorruptGCSImmutableObject,
    GCS_CLIENT_VERSION,
    GCS_IMMUTABLE_STORAGE_CONTRACT,
    GCSImmutableByteStore,
    GCSImmutableByteStoreError,
    GCSImmutableObjectConflict,
    GCSPhysicalObjectReceiptV1,
    content_addressed_object_id,
    content_addressed_record_object_id,
)
from moe_sae.r1.protocol import ProtocolError


_MIB = 1024**2


class _RetryPolicy:
    def __init__(self) -> None:
        self.timeout: float | None = None

    def with_timeout(self, timeout: float) -> "_RetryPolicy":
        self.timeout = timeout
        return self


_RETRY_POLICY = _RetryPolicy()


class _HTTPError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        self.code = code
        super().__init__(message)


class _DriftStr(str):
    """A string whose formatted representation can drift after validation."""

    def __new__(cls, value: str) -> "_DriftStr":
        instance = super().__new__(cls, value)
        instance.render = value
        return instance

    def __format__(self, format_spec: str) -> str:
        return format(self.render, format_spec)


def _crc32c(value: bytes) -> str:
    checksum = google_crc32c.Checksum()
    checksum.update(value)
    return base64.b64encode(checksum.digest()).decode("ascii")


@dataclass
class _Object:
    value: bytes
    generation: int
    metageneration: int
    metadata: dict[str, str]
    content_type: str
    crc32c: str
    content_encoding: str | None = None


class _FakeClient:
    """A generation-aware fake with intentionally no listing method."""

    def __init__(
        self,
        *,
        location: str = "US-CENTRAL1",
        default_kms_key_name: str | None = None,
    ) -> None:
        self.location = location
        self.default_kms_key_name = default_kms_key_name
        self.objects: dict[tuple[str, str], _Object] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.lock = threading.RLock()
        self.next_generation = 100
        self.before_blob: Callable[["_FakeBucket"], None] | None = None
        self.before_upload: Callable[[], None] | None = None
        self.before_download: Callable[[str, str], None] | None = None
        self.after_download: Callable[[str, str], None] | None = None
        self.fail_upload_before_status: int | None = None
        self.fail_upload_after_status: int | None = None
        self.fail_blob_reload_status: int | None = None
        self.download_override: bytes | None = None
        self.download_piece_bytes: int | None = None
        self.upload_read_sizes: list[int] = []
        self.downloaded_bytes = 0

    def bucket(self, name: str) -> "_FakeBucket":
        return _FakeBucket(self, name)

    def current(self, bucket: str, name: str) -> _Object:
        with self.lock:
            return self.objects[(bucket, name)]

    def delete(self, bucket: str, name: str) -> None:
        with self.lock:
            self.objects.pop((bucket, name), None)

    def mutate_metadata(self, bucket: str, name: str) -> None:
        with self.lock:
            obj = self.objects[(bucket, name)]
            obj.metadata["foreign-field"] = "forbidden"
            obj.metageneration += 1

    def replace_live(self, bucket: str, name: str) -> int:
        with self.lock:
            obj = self.objects[(bucket, name)]
            generation = self.next_generation
            self.next_generation += 1
            self.objects[(bucket, name)] = _Object(
                value=obj.value,
                generation=generation,
                metageneration=1,
                metadata=dict(obj.metadata),
                content_type=obj.content_type,
                crc32c=obj.crc32c,
                content_encoding=obj.content_encoding,
            )
            return generation


class _FakeBucket:
    def __init__(self, client: _FakeClient, name: str) -> None:
        self.client = client
        self.name = name
        self.location: str | None = None
        self.default_kms_key_name: str | None = None

    def reload(self, **kwargs: Any) -> None:
        self.client.calls.append(("bucket_reload", self.name, kwargs))
        self.location = self.client.location
        self.default_kms_key_name = self.client.default_kms_key_name

    def blob(
        self,
        name: str,
        *,
        chunk_size: int,
        generation: int | None = None,
    ) -> "_FakeBlob":
        hook = self.client.before_blob
        if hook is not None:
            self.client.before_blob = None
            hook(self)
        return _FakeBlob(
            self.client,
            self.name,
            name,
            chunk_size=chunk_size,
            requested_generation=generation,
        )


class _FakeBlob:
    def __init__(
        self,
        client: _FakeClient,
        bucket: str,
        name: str,
        *,
        chunk_size: int,
        requested_generation: int | None,
    ) -> None:
        self.client = client
        self.bucket = bucket
        self.name = name
        self.chunk_size = chunk_size
        self.requested_generation = requested_generation
        self.metadata: dict[str, str] | None = None
        self.content_type: str | None = None
        self.content_encoding: str | None = None
        self.generation: int | None = requested_generation
        self.metageneration: int | None = None
        self.size: int | None = None
        self.crc32c: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.bucket, self.name

    def _refresh(self, obj: _Object) -> None:
        self.metadata = dict(obj.metadata)
        self.content_type = obj.content_type
        self.content_encoding = obj.content_encoding
        self.generation = obj.generation
        self.metageneration = obj.metageneration
        self.size = len(obj.value)
        self.crc32c = obj.crc32c

    def reload(self, **kwargs: Any) -> None:
        self.client.calls.append(("blob_reload", f"{self.bucket}/{self.name}", kwargs))
        if self.client.fail_blob_reload_status is not None:
            status = self.client.fail_blob_reload_status
            self.client.fail_blob_reload_status = None
            raise _HTTPError(status, "injected metadata failure")
        with self.client.lock:
            obj = self.client.objects.get(self.key)
            if obj is None:
                raise _HTTPError(404, "not found")
            if (
                self.requested_generation is not None
                and self.requested_generation != obj.generation
            ):
                raise _HTTPError(404, "requested generation unavailable")
            if (
                kwargs.get("if_generation_match") is not None
                and kwargs["if_generation_match"] != obj.generation
            ):
                raise _HTTPError(412, "generation mismatch")
            if (
                kwargs.get("if_metageneration_match") is not None
                and kwargs["if_metageneration_match"] != obj.metageneration
            ):
                raise _HTTPError(412, "metageneration mismatch")
            self._refresh(obj)

    def upload_from_file(self, file_obj: Any, **kwargs: Any) -> None:
        self.client.calls.append(("upload", f"{self.bucket}/{self.name}", kwargs))
        if self.client.fail_upload_before_status is not None:
            status = self.client.fail_upload_before_status
            self.client.fail_upload_before_status = None
            raise _HTTPError(status, "injected pre-commit upload failure")
        hook = self.client.before_upload
        if hook is not None:
            hook()
        size = kwargs["size"]
        value = bytearray()
        # google-cloud-storage 3.13.0 reads a multipart object in one call at
        # or below 8 MiB; larger objects use Blob.chunk_size resumable reads.
        upload_piece_bytes = size if size <= 8 * _MIB else self.chunk_size
        while len(value) < size:
            request = min(upload_piece_bytes, size - len(value))
            piece = file_obj.read(request)
            assert type(piece) is bytes
            assert len(piece) <= 8 * _MIB
            self.client.upload_read_sizes.append(len(piece))
            if not piece:
                break
            value.extend(piece)
        if len(value) != size:
            raise ValueError("short fake upload stream")
        raw = bytes(value)
        if size <= 8 * _MIB:
            # Pinned google-cloud-storage 3.13.0 ignores the caller's
            # crc32c_checksum_value for multipart uploads and computes one
            # from the bytes it reads. The immutable-bytes API makes those
            # bytes identical to the premeasured payload.
            if kwargs.get("checksum") != "crc32c":
                raise AssertionError("multipart upload must request client CRC32C")
        else:
            # For resumable uploads the precomputed value is placed in object
            # metadata for server finalization validation while checksum=None
            # prevents the pinned client's unqualified corruption delete.
            if kwargs.get("checksum") is not None:
                raise AssertionError("resumable upload must disable client deletion")
            if kwargs.get("crc32c_checksum_value") != _crc32c(raw):
                raise _HTTPError(400, "server CRC32C rejection")
        with self.client.lock:
            current = self.client.objects.get(self.key)
            if kwargs.get("if_generation_match") != 0:
                raise AssertionError("fake accepts only create preconditions")
            if current is not None:
                raise _HTTPError(412, "create precondition failed")
            generation = self.client.next_generation
            self.client.next_generation += 1
            obj = _Object(
                value=raw,
                generation=generation,
                metageneration=1,
                metadata=dict(self.metadata or {}),
                content_type=str(kwargs["content_type"]),
                crc32c=_crc32c(raw),
            )
            self.client.objects[self.key] = obj
            self._refresh(obj)
        if self.client.fail_upload_after_status is not None:
            status = self.client.fail_upload_after_status
            self.client.fail_upload_after_status = None
            raise _HTTPError(status, "injected post-commit upload failure")

    def download_to_file(self, file_obj: Any, **kwargs: Any) -> None:
        self.client.calls.append(("download", f"{self.bucket}/{self.name}", kwargs))
        hook = self.client.before_download
        if hook is not None:
            self.client.before_download = None
            hook(self.bucket, self.name)
        with self.client.lock:
            obj = self.client.objects.get(self.key)
            if obj is None:
                raise _HTTPError(404, "not found")
            if self.requested_generation != obj.generation:
                raise _HTTPError(412, "requested generation mismatch")
            if kwargs.get("if_generation_match") != obj.generation:
                raise _HTTPError(412, "generation mismatch")
            if kwargs.get("if_metageneration_match") != obj.metageneration:
                raise _HTTPError(412, "metageneration mismatch")
            value = (
                obj.value
                if self.client.download_override is None
                else self.client.download_override
            )
        piece_bytes = self.client.download_piece_bytes or self.chunk_size
        for start in range(0, len(value), piece_bytes):
            piece = value[start : start + piece_bytes]
            self.client.downloaded_bytes += len(piece)
            assert file_obj.write(piece) == len(piece)
        hook = self.client.after_download
        if hook is not None:
            self.client.after_download = None
            hook(self.bucket, self.name)


def _store(
    client: _FakeClient | None = None,
    *,
    prefix: str = "converted/r1",
    region: str = "us-central1",
    io_chunk_bytes: int = 256 * 1024,
    maximum_object_bytes: int = MAX_IMMUTABLE_SNAPSHOT_BYTES,
) -> tuple[GCSImmutableByteStore, _FakeClient]:
    fake = _FakeClient() if client is None else client
    return (
        GCSImmutableByteStore(
            "r1-checkpoints",
            prefix,
            region,
            fake,
            io_chunk_bytes=io_chunk_bytes,
            maximum_object_bytes=maximum_object_bytes,
            timeout_seconds=17,
            retry_deadline_seconds=29,
            resolution_attempts=4,
            retry_policy=_RETRY_POLICY,
        ),
        fake,
    )


def _create(
    store: GCSImmutableByteStore,
    payload: bytes,
) -> GCSPhysicalObjectReceiptV1:
    return store.create_content_or_verify(
        payload,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
    )


def test_receipt_is_strict_canonical_with_decimal_generation_fields() -> None:
    store, client = _store()
    payload = b"immutable receipt"
    receipt = _create(store, payload)

    assert GCSPhysicalObjectReceiptV1.from_json(receipt.canonical_bytes()) == receipt
    assert receipt.generation == "100"
    assert receipt.metageneration == "1"
    assert receipt.object_id == content_addressed_object_id(receipt.sha256)
    assert receipt.uri == f"gs://r1-checkpoints/converted/r1/{receipt.object_id}"
    assert receipt.region == "us-central1"
    assert receipt.crc32c == _crc32c(payload)
    assert (
        receipt.storage_contract
        == GCS_IMMUTABLE_STORAGE_CONTRACT
        == "gcs-json-create-only-generation-crc32c-sha256-v2"
    )
    assert not hasattr(client, "list_blobs")

    forged = receipt.as_dict()
    forged["generation"] = 100
    with pytest.raises(ProtocolError, match="string"):
        GCSPhysicalObjectReceiptV1.from_dict(forged)
    forged = receipt.as_dict()
    forged["generation"] = "0100"
    with pytest.raises(ProtocolError, match="canonical"):
        GCSPhysicalObjectReceiptV1.from_dict(forged)
    forged = receipt.as_dict()
    forged["crc32c"] = "AAAAAAAA"
    with pytest.raises(ProtocolError, match="crc32c"):
        GCSPhysicalObjectReceiptV1.from_dict(forged)
    forged = receipt.as_dict()
    forged["unknown"] = "forbidden"
    with pytest.raises(ProtocolError, match="unknown"):
        GCSPhysicalObjectReceiptV1.from_dict(forged)
    forged = receipt.as_dict()
    forged["storage_contract"] = "gcs-json-create-only-generation-crc32c-sha256-v1"
    with pytest.raises(ProtocolError, match="unsupported"):
        GCSPhysicalObjectReceiptV1.from_dict(forged)
    forged = receipt.as_dict()
    forged["generation"] = "9" * 5000
    with pytest.raises(ProtocolError, match="canonical"):
        GCSPhysicalObjectReceiptV1.from_dict(forged)
    forged = receipt.as_dict()
    forged["metageneration"] = str(1 << 63)
    with pytest.raises(ProtocolError, match="canonical"):
        GCSPhysicalObjectReceiptV1.from_dict(forged)
    forged = receipt.as_dict()
    foreign_sha256 = hashlib.sha256(b"foreign payload").hexdigest()
    foreign_object_id = content_addressed_object_id(foreign_sha256)
    forged["object_id"] = foreign_object_id
    forged["uri"] = f"gs://r1-checkpoints/converted/r1/{foreign_object_id}"
    with pytest.raises(ProtocolError, match="reserved"):
        GCSPhysicalObjectReceiptV1.from_dict(forged)


def test_store_identity_bucket_location_and_constructor_bounds_are_exact() -> None:
    store, client = _store()
    same, _ = _store(client)
    different, _ = _store(_FakeClient(), prefix="converted/other")

    assert store.store_id == same.store_id
    assert store.store_id != different.store_id
    assert store.distributed_identity_qualified is False
    assert store.store_id.startswith("gcs-test-prefix-byte-store-v2:")
    assert store.immutable_snapshot_bytes == MAX_IMMUTABLE_SNAPSHOT_BYTES
    assert store.maximum_object_bytes == MAX_IMMUTABLE_SNAPSHOT_BYTES
    assert store.stream_buffer_bytes == MAX_APPLICATION_STREAM_SLICE_BYTES
    assert _RETRY_POLICY.timeout == 29.0
    bucket_reload = client.calls[0]
    assert bucket_reload == (
        "bucket_reload",
        "r1-checkpoints",
        {"projection": "noAcl", "timeout": 17.0, "retry": _RETRY_POLICY},
    )

    with pytest.raises(GCSImmutableByteStoreError, match="location"):
        _store(_FakeClient(location="EUROPE-WEST4"))
    with pytest.raises(GCSImmutableByteStoreError, match="CMEK"):
        _store(
            _FakeClient(
                default_kms_key_name="projects/p/locations/us/keyRings/r/cryptoKeys/k"
            )
        )


@pytest.mark.parametrize(
    ("field", "argument_name", "value"),
    [
        ("bucket", "bucket", "r1-checkpoints"),
        ("prefix", "prefix", "converted/r1"),
        ("region", "expected_region", "us-central1"),
    ],
)
def test_store_rejects_render_mutable_string_subclasses_before_object_io(
    field: str,
    argument_name: str,
    value: str,
) -> None:
    client = _FakeClient()
    arguments: dict[str, str] = {
        "bucket": "r1-checkpoints",
        "prefix": "converted/r1",
        "expected_region": "us-central1",
    }
    arguments[argument_name] = _DriftStr(value)

    with pytest.raises(ValueError, match=field):
        GCSImmutableByteStore(
            **arguments,
            client=client,
            retry_policy=_RETRY_POLICY,
        )

    assert client.calls == []
    assert client.objects == {}


def test_operation_local_blob_address_drift_fails_before_object_io() -> None:
    store, client = _store()
    client.calls.clear()
    client.before_blob = lambda bucket: setattr(bucket, "name", "other-bucket")

    with pytest.raises(GCSImmutableByteStoreError, match="blob address drifted"):
        _create(store, b"must stay on the frozen bucket")

    assert client.calls == []
    assert client.objects == {}


@pytest.mark.parametrize("field", ["object_id", "expected_sha256"])
def test_create_rejects_render_mutable_string_subclasses_before_object_io(
    field: str,
) -> None:
    store, client = _store()
    payload = b"must not poison a canonical SHA path"
    sha256 = hashlib.sha256(payload).hexdigest()
    arguments: dict[str, object] = {
        "object_id": content_addressed_object_id(sha256),
        "payload": payload,
        "expected_sha256": sha256,
        "expected_size_bytes": len(payload),
    }
    arguments[field] = _DriftStr(str(arguments[field]))
    client.calls.clear()

    with pytest.raises((ProtocolError, GCSImmutableByteStoreError), match=field):
        store.create_or_verify(**arguments)  # type: ignore[arg-type]

    assert client.calls == []
    assert client.objects == {}
    with pytest.raises(ValueError, match="region"):
        _store(_FakeClient(location="NAM4"), region="nam4")
    with pytest.raises(ValueError, match="256-KiB"):
        GCSImmutableByteStore(
            "r1-checkpoints",
            "converted/r1",
            "us-central1",
            _FakeClient(),
            io_chunk_bytes=123,
            retry_policy=_RETRY_POLICY,
        )
    with pytest.raises(ValueError, match="64 MiB"):
        _store(maximum_object_bytes=MAX_IMMUTABLE_SNAPSHOT_BYTES + 1)
    with pytest.raises(ValueError, match="retry_policy"):
        GCSImmutableByteStore(
            "r1-checkpoints",
            "converted/r1",
            "us-central1",
            _FakeClient(),
            retry_policy=None,
        )


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [
        ("bucket_name", "other-bucket"),
        ("prefix", "converted/other"),
        ("region", "us-east1"),
        ("client", _FakeClient()),
        ("_bucket", object()),
        ("_bucket_name_snapshot", "other-bucket"),
        ("_prefix_snapshot", "converted/other"),
        ("_region_snapshot", "us-east1"),
    ],
)
def test_real_adapter_address_identity_snapshots_are_read_only(
    attribute: str,
    replacement: object,
) -> None:
    store, _client = _store()

    with pytest.raises(AttributeError):
        setattr(store, attribute, replacement)

    payload = b"still uses the frozen bucket and prefix"
    receipt = _create(store, payload)
    assert receipt.uri == (f"gs://r1-checkpoints/converted/r1/{receipt.object_id}")


def test_store_snapshot_cannot_be_deleted_then_reassigned() -> None:
    store, client = _store()
    client.calls.clear()

    with pytest.raises(AttributeError, match="immutable store snapshot"):
        del store._prefix_snapshot  # noqa: SLF001 - exact freeze invariant
    with pytest.raises(AttributeError, match="immutable store snapshot"):
        store._prefix_snapshot = "converted/redirected"  # noqa: SLF001

    assert store.prefix == "converted/r1"
    assert client.calls == []
    assert client.objects == {}


@pytest.mark.parametrize("drift", ["bucket-name", "bucket-client"])
def test_real_adapter_retained_bucket_handle_drift_fails_before_object_io(
    drift: str,
) -> None:
    store, client = _store()
    bucket = store._bucket_snapshot  # noqa: SLF001 - exact adapter invariant
    if drift == "bucket-name":
        bucket.name = "other-bucket"
    else:
        bucket.client = _FakeClient()
    client.calls.clear()

    with pytest.raises(GCSImmutableByteStoreError, match="handle drifted"):
        _create(store, b"must not cross the frozen address boundary")

    assert client.calls == []
    assert client.objects == {}


def test_exact_313_client_contract_and_internal_client_production_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = pytest.importorskip("google.cloud.storage")
    from google.cloud.storage import blob as blob_module

    assert storage.__version__ == GCS_CLIENT_VERSION
    assert blob_module._MAX_MULTIPART_SIZE == 8 * _MIB
    assert (
        "crc32c_checksum_value"
        not in inspect.signature(blob_module.Blob._do_multipart_upload).parameters
    )
    assert (
        "crc32c_checksum_value"
        in inspect.signature(blob_module.Blob._do_resumable_upload).parameters
    )
    assert "delete" in blob_module.Blob._do_resumable_upload.__code__.co_names
    fake = _FakeClient()
    monkeypatch.setattr(storage, "Client", lambda *, project=None: fake)

    production = GCSImmutableByteStore(
        "r1-checkpoints",
        "converted/r1",
        "us-central1",
        retry_policy=_RETRY_POLICY,
    )
    injected, _client = _store(_FakeClient())

    assert production.distributed_identity_qualified is True
    assert production.store_id.startswith("gcs-prefix-byte-store-v2:")
    assert production.store_id != injected.store_id


def test_resumable_upload_uses_server_crc_without_client_delete_and_bounded_chunks() -> (
    None
):
    store, client = _store(io_chunk_bytes=1 * _MIB)
    payload = b"x" * (8 * _MIB + 17)
    receipt = _create(store, payload)

    upload = next(call for call in client.calls if call[0] == "upload")
    assert upload[2] == {
        "rewind": False,
        "size": len(payload),
        "content_type": "application/octet-stream",
        "if_generation_match": 0,
        "checksum": None,
        "retry": _RETRY_POLICY,
        "timeout": 17.0,
        "crc32c_checksum_value": _crc32c(payload),
    }
    assert max(client.upload_read_sizes) <= store.stream_buffer_bytes
    obj = client.current("r1-checkpoints", f"converted/r1/{receipt.object_id}")
    assert obj.metadata == {
        "moe-sae-r1-immutable-schema": "v1",
        "moe-sae-r1-immutable-store-id": store.store_id,
        "moe-sae-r1-immutable-object-id-sha256": hashlib.sha256(
            receipt.object_id.encode()
        ).hexdigest(),
        "moe-sae-r1-immutable-sha256": receipt.sha256,
        "moe-sae-r1-immutable-size-bytes": str(len(payload)),
    }
    assert obj.content_encoding is None
    downloads = [call for call in client.calls if call[0] == "download"]
    assert downloads
    assert all(
        call[2]
        == {
            "raw_download": True,
            "if_generation_match": 100,
            "if_metageneration_match": 1,
            "checksum": None,
            "timeout": 17.0,
            "retry": _RETRY_POLICY,
            "single_shot_download": False,
        }
        for call in downloads
    )


def test_pinned_313_multipart_upload_computes_crc_from_exact_immutable_bytes() -> None:
    store, client = _store()
    payload = b"small immutable multipart payload"
    receipt = _create(store, payload)

    upload = next(call for call in client.calls if call[0] == "upload")
    assert GCS_CLIENT_VERSION == "3.13.0"
    assert upload[2]["size"] == len(payload)
    assert upload[2]["checksum"] == "crc32c"
    assert upload[2]["crc32c_checksum_value"] == _crc32c(payload)
    assert store.verify(receipt) == receipt


def test_memoryview_multipart_snapshot_cannot_poison_canonical_path() -> None:
    store, client = _store()
    mutable = bytearray(b"a" * (2 * _MIB))
    expected = bytes(mutable)
    digest = hashlib.sha256(expected).hexdigest()

    def mutate_source() -> None:
        mutable[:] = b"b" * len(mutable)

    client.before_upload = mutate_source
    view = memoryview(mutable)
    try:
        receipt = store.create_memoryview_or_verify(
            content_addressed_object_id(digest),
            view,
            expected_sha256=digest,
            expected_size_bytes=len(view),
        )
    finally:
        view.release()

    stored = client.current(
        "r1-checkpoints",
        f"converted/r1/{content_addressed_object_id(digest)}",
    )
    assert stored.value == expected
    assert stored.value != bytes(mutable)
    assert receipt.sha256 == digest
    assert receipt.crc32c == _crc32c(expected)


def test_memoryview_resumable_same_crc_mutation_cannot_poison_canonical_path() -> None:
    store, client = _store(io_chunk_bytes=256 * 1024)
    prefix = b"a" * (8 * _MIB)
    # These fixed equal-length suffixes are a deterministic CRC32C collision.
    expected = prefix + bytes.fromhex("6c7fd5e46cf48184")
    colliding = prefix + bytes.fromhex("5f479afde2b0d701")
    assert _crc32c(expected) == _crc32c(colliding)
    assert hashlib.sha256(expected).digest() != hashlib.sha256(colliding).digest()
    mutable = bytearray(expected)
    digest = hashlib.sha256(expected).hexdigest()
    object_id = content_addressed_object_id(digest)

    def mutate_source() -> None:
        mutable[:] = colliding

    client.before_upload = mutate_source
    view = memoryview(mutable)
    try:
        receipt = store.create_memoryview_or_verify(
            object_id,
            view,
            expected_sha256=digest,
            expected_size_bytes=len(view),
        )
    finally:
        view.release()

    stored = client.current("r1-checkpoints", f"converted/r1/{object_id}")
    assert stored.value == expected
    assert stored.value != bytes(mutable)
    assert receipt.sha256 == digest
    assert receipt.crc32c == _crc32c(expected)


def test_read_exact_downloads_and_hashes_whole_generation_but_captures_range() -> None:
    store, client = _store()
    payload = bytes(range(251)) * 3000
    receipt = _create(store, payload)
    client.downloaded_bytes = 0
    client.calls.clear()

    selected = store.read_exact(
        receipt,
        offset=12345,
        length=6789,
        maximum_bytes=6789,
    )

    assert selected == payload[12345 : 12345 + 6789]
    assert client.downloaded_bytes == len(payload)
    downloads = [call for call in client.calls if call[0] == "download"]
    assert len(downloads) == 1
    assert downloads[0][2]["if_generation_match"] == int(receipt.generation)
    assert downloads[0][2]["if_metageneration_match"] == int(receipt.metageneration)


def test_identical_create_is_idempotent_and_concurrent_writers_converge() -> None:
    store, client = _store()
    payload = b"same immutable bytes" * 1000
    digest = hashlib.sha256(payload).hexdigest()
    barrier = threading.Barrier(2)
    client.before_upload = barrier.wait

    def publish() -> GCSPhysicalObjectReceiptV1:
        return store.create_content_or_verify(
            payload,
            expected_sha256=digest,
            expected_size_bytes=len(payload),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = list(executor.map(lambda _unused: publish(), range(2)))

    assert receipts[0] == receipts[1]
    assert len(client.objects) == 1
    client.before_upload = None
    assert publish() == receipts[0]
    assert all(
        call[2]["if_generation_match"] == 0
        for call in client.calls
        if call[0] == "upload"
    )


def test_divergent_exact_path_fails_closed_without_overwrite() -> None:
    store, client = _store()
    first = b"first"
    second = b"other"
    object_id = "plans/test/receipts/chunk.json"
    first_receipt = store.create_or_verify(
        object_id,
        first,
        expected_sha256=hashlib.sha256(first).hexdigest(),
        expected_size_bytes=len(first),
    )

    with pytest.raises(GCSImmutableObjectConflict, match="divergent"):
        store.create_or_verify(
            object_id,
            second,
            expected_sha256=hashlib.sha256(second).hexdigest(),
            expected_size_bytes=len(second),
        )

    assert store.verify(first_receipt) == first_receipt
    assert client.current("r1-checkpoints", f"converted/r1/{object_id}").value == first


def test_reserved_content_address_namespaces_reject_foreign_or_malformed_paths() -> (
    None
):
    store, client = _store()
    intended = b"intended content"
    foreign = b"foreign content"
    intended_sha256 = hashlib.sha256(intended).hexdigest()
    foreign_sha256 = hashlib.sha256(foreign).hexdigest()
    client.calls.clear()

    with pytest.raises(GCSImmutableByteStoreError, match="reserved"):
        store.create_or_verify(
            content_addressed_object_id(intended_sha256),
            foreign,
            expected_sha256=foreign_sha256,
            expected_size_bytes=len(foreign),
        )
    with pytest.raises(GCSImmutableByteStoreError, match="reserved"):
        store.create_or_verify(
            "objects/sha256/foreign-binding.json",
            foreign,
            expected_sha256=foreign_sha256,
            expected_size_bytes=len(foreign),
        )
    with pytest.raises(GCSImmutableByteStoreError, match="reserved"):
        store.create_or_verify(
            content_addressed_record_object_id(intended_sha256),
            foreign,
            expected_sha256=foreign_sha256,
            expected_size_bytes=len(foreign),
        )
    with pytest.raises(GCSImmutableByteStoreError, match="reserved"):
        store.create_or_verify(
            "records/sha256/foreign-binding.json",
            foreign,
            expected_sha256=foreign_sha256,
            expected_size_bytes=len(foreign),
        )

    assert not [call for call in client.calls if call[0] == "upload"]
    assert not client.objects


def test_content_addressed_json_record_is_idempotent_and_reserved() -> None:
    store, _client = _store()
    payload = b'{"schema":"example.v1"}'
    digest = hashlib.sha256(payload).hexdigest()
    object_id = content_addressed_record_object_id(digest)

    first = store.create_or_verify(
        object_id,
        payload,
        expected_sha256=digest,
        expected_size_bytes=len(payload),
    )
    second = store.create_or_verify(
        object_id,
        payload,
        expected_sha256=digest,
        expected_size_bytes=len(payload),
    )

    assert first == second
    assert first.content_type == "application/json"
    assert first.object_id == object_id


def test_existing_exact_bytes_with_foreign_metadata_are_a_conflict() -> None:
    store, client = _store()
    payload = b"bytes whose binding was altered"
    receipt = _create(store, payload)
    obj = client.current("r1-checkpoints", f"converted/r1/{receipt.object_id}")
    obj.metadata["foreign-field"] = "forbidden"

    with pytest.raises(GCSImmutableObjectConflict, match="metadata"):
        _create(store, payload)


@pytest.mark.parametrize("status", [412, 503])
def test_ambiguous_post_commit_upload_adopts_only_exact_verified_object(
    status: int,
) -> None:
    store, client = _store()
    client.fail_upload_after_status = status
    payload = b"committed before response loss"

    receipt = _create(store, payload)

    assert receipt.sha256 == hashlib.sha256(payload).hexdigest()
    assert store.verify(receipt) == receipt


def test_precommit_transport_failure_has_bounded_exact_path_resolution() -> None:
    store, client = _store()
    client.calls.clear()
    client.fail_upload_before_status = 503

    with pytest.raises(GCSImmutableByteStoreError, match="cannot create"):
        _create(store, b"never committed")

    reloads = [call for call in client.calls if call[0] == "blob_reload"]
    assert len(reloads) == 4
    assert not client.objects


def test_transient_existing_object_verification_is_retried_not_conflicted() -> None:
    store, client = _store()
    payload = b"healthy object behind a transient read"
    receipt = _create(store, payload)

    def fail_once(_bucket: str, _name: str) -> None:
        raise _HTTPError(503, "one transient download failure")

    client.before_download = fail_once
    assert _create(store, payload) == receipt


def test_nonambiguous_authorization_failure_is_not_misreported_as_absence() -> None:
    store, client = _store()
    client.calls.clear()
    client.fail_upload_before_status = 403

    with pytest.raises(GCSImmutableByteStoreError, match="cannot create"):
        _create(store, b"forbidden")

    assert not [call for call in client.calls if call[0] == "blob_reload"]


@pytest.mark.parametrize("failure", ["metadata", "replace", "delete"])
def test_receipt_fails_closed_on_mutation_replacement_or_deletion(failure: str) -> None:
    store, client = _store()
    receipt = _create(store, b"stable generation")
    bucket, name = "r1-checkpoints", f"converted/r1/{receipt.object_id}"
    if failure == "metadata":
        client.mutate_metadata(bucket, name)
    elif failure == "replace":
        client.replace_live(bucket, name)
    else:
        client.delete(bucket, name)

    with pytest.raises(CorruptGCSImmutableObject):
        store.verify(receipt)


def test_change_during_download_and_wrong_download_bytes_fail_closed() -> None:
    store, client = _store()
    payload = b"authenticated bytes" * 20_000
    receipt = _create(store, payload)
    bucket, name = "r1-checkpoints", f"converted/r1/{receipt.object_id}"

    client.download_override = b"x" * len(payload)
    with pytest.raises(CorruptGCSImmutableObject, match="SHA-256 or CRC32C"):
        store.verify(receipt)
    client.download_override = None

    client.after_download = lambda _bucket, _name: client.delete(bucket, name)
    with pytest.raises(CorruptGCSImmutableObject, match="live path"):
        store.verify(receipt)


def test_probe_rejects_oversize_before_download_and_absence_is_precise() -> None:
    store, client = _store()
    receipt = _create(store, b"oversized probe")
    client.calls.clear()

    with pytest.raises(GCSImmutableByteStoreError, match="bounded probe"):
        store.probe_bounded(receipt.object_id, maximum_bytes=2)
    assert not [call for call in client.calls if call[0] == "download"]
    assert (
        store.probe_bounded("plans/test/receipts/absent.json", maximum_bytes=10) is None
    )


def test_remote_metadata_content_encoding_and_stream_piece_bounds_are_strict() -> None:
    store, client = _store()
    receipt = _create(store, b"strict remote representation" * 20_000)
    bucket, name = "r1-checkpoints", f"converted/r1/{receipt.object_id}"
    obj = client.current(bucket, name)

    obj.metadata["unknown"] = "not allowlisted"
    with pytest.raises(CorruptGCSImmutableObject, match="metadata"):
        store.probe_bounded(receipt.object_id, maximum_bytes=receipt.size_bytes)
    obj.metadata.pop("unknown")
    obj.content_encoding = "gzip"
    with pytest.raises(CorruptGCSImmutableObject, match="content encoding"):
        store.probe_bounded(receipt.object_id, maximum_bytes=receipt.size_bytes)
    obj.content_encoding = None

    client.download_piece_bytes = store.stream_buffer_bytes + 1
    with pytest.raises(CorruptGCSImmutableObject, match="slice above the bound"):
        store.verify(receipt)


def test_payload_and_read_bounds_fail_before_remote_payload_io() -> None:
    store, client = _store()
    payload = b"abcdef"
    digest = hashlib.sha256(payload).hexdigest()
    client.calls.clear()
    with pytest.raises(GCSImmutableByteStoreError, match="SHA-256"):
        store.create_content_or_verify(
            payload,
            expected_sha256="0" * 64,
            expected_size_bytes=len(payload),
        )
    with pytest.raises(GCSImmutableByteStoreError, match="size"):
        store.create_content_or_verify(
            payload,
            expected_sha256=digest,
            expected_size_bytes=len(payload) + 1,
        )
    for mutable_payload in (bytearray(payload), memoryview(payload)):
        with pytest.raises(GCSImmutableByteStoreError, match="exact bytes"):
            store.create_content_or_verify(
                mutable_payload,  # type: ignore[arg-type]
                expected_sha256=digest,
                expected_size_bytes=len(payload),
            )
    assert not [call for call in client.calls if call[0] == "upload"]

    receipt = _create(store, payload)
    client.calls.clear()
    with pytest.raises(GCSImmutableByteStoreError, match="exceeds its bound"):
        store.read_exact(receipt, offset=0, length=4, maximum_bytes=3)
    with pytest.raises(GCSImmutableByteStoreError, match="object size"):
        store.read_exact(receipt, offset=4, length=3, maximum_bytes=3)
    assert not [call for call in client.calls if call[0] == "download"]
