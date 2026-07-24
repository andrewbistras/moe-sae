from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import google_crc32c
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/r1/publish_derived_runtime.py"
SPEC = importlib.util.spec_from_file_location("r1_derived_runtime_publisher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
publisher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = publisher
SPEC.loader.exec_module(publisher)


class _HTTPError(RuntimeError):
    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(f"injected HTTP {code}")


class _ConditionalRetry:
    def __init__(self, *, unsafe: bool = False) -> None:
        self.unsafe = unsafe
        self.enabled = object()

    def get_retry_policy_if_conditions_met(
        self,
        *,
        query_params: dict[str, object],
    ) -> object | None:
        if self.unsafe:
            return self.enabled
        if query_params.get("ifGenerationMatch") is not None:
            return self.enabled
        return None


READ_RETRY = object()
UPLOAD_RETRY = _ConditionalRetry()


def _crc32c(raw: bytes) -> str:
    checksum = google_crc32c.Checksum()
    checksum.update(raw)
    return base64.b64encode(checksum.digest()).decode("ascii")


@dataclass
class _StoredObject:
    raw: bytes
    generation: int
    metageneration: int
    metadata: dict[str, str]
    content_type: str
    crc32c: str
    component_count: int | None = None
    content_encoding: str | None = None


class _FakeClient:
    def __init__(
        self,
        *,
        receipt: Path,
        location: str = "EUROPE-WEST4",
    ) -> None:
        self.receipt = receipt
        self.location = location
        self.bucket_metageneration = 11
        self.object: _StoredObject | None = None
        self.next_generation = 101
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.fail_upload_before: int | None = None
        self.fail_upload_after: int | None = None
        self.before_upload: Callable[[], None] | None = None
        self.after_upload_commit: Callable[[_StoredObject], None] | None = None
        self.after_reload: Callable[[str, int | None], None] | None = None
        self.after_bucket_reload: Callable[[], None] | None = None
        self.upload_count = 0
        self.download_count = 0

    def _record(self, name: str, kwargs: dict[str, object]) -> None:
        assert not self.receipt.exists(), (
            "receipt was visible before verification ended"
        )
        self.calls.append((name, dict(kwargs)))

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(self, name)


class _FakeBucket:
    def __init__(self, client: _FakeClient, name: str) -> None:
        self.client = client
        self.name = name
        self.location: str | None = None
        self.metageneration: int | None = None

    def reload(self, **kwargs: object) -> None:
        self.client._record("bucket_reload", dict(kwargs))
        self.location = self.client.location
        self.metageneration = self.client.bucket_metageneration
        hook = self.client.after_bucket_reload
        if hook is not None:
            hook()

    def blob(
        self,
        name: str,
        *,
        chunk_size: int,
        generation: int | None = None,
    ) -> _FakeBlob:
        assert chunk_size == publisher.CHUNK_BYTES
        return _FakeBlob(self, name=name, requested_generation=generation)


class _FakeBlob:
    def __init__(
        self,
        bucket: _FakeBucket,
        *,
        name: str,
        requested_generation: int | None,
    ) -> None:
        self.bucket = bucket
        self.name = name
        self.requested_generation = requested_generation
        self.metadata: dict[str, str] | None = None
        self.content_type: str | None = None
        self.content_encoding: str | None = None
        self.component_count: int | None = None
        self.generation: int | None = requested_generation
        self.metageneration: int | None = None
        self.size: int | None = None
        self.crc32c: str | None = None

    @property
    def client(self) -> _FakeClient:
        return self.bucket.client

    def _refresh(self, stored: _StoredObject) -> None:
        self.metadata = dict(stored.metadata)
        self.content_type = stored.content_type
        self.content_encoding = stored.content_encoding
        self.component_count = stored.component_count
        self.generation = stored.generation
        self.metageneration = stored.metageneration
        self.size = len(stored.raw)
        self.crc32c = stored.crc32c

    def reload(self, **kwargs: object) -> None:
        self.client._record("reload", dict(kwargs))
        stored = self.client.object
        if stored is None:
            raise _HTTPError(404)
        if (
            self.requested_generation is not None
            and stored.generation != self.requested_generation
        ):
            raise _HTTPError(404)
        if (
            kwargs.get("if_generation_match") is not None
            and kwargs["if_generation_match"] != stored.generation
        ):
            raise _HTTPError(412)
        if (
            kwargs.get("if_metageneration_match") is not None
            and kwargs["if_metageneration_match"] != stored.metageneration
        ):
            raise _HTTPError(412)
        self._refresh(stored)
        hook = self.client.after_reload
        if hook is not None:
            hook(self.name, self.requested_generation)

    def upload_from_file(self, stream: Any, **kwargs: object) -> None:
        self.client._record("upload", dict(kwargs))
        self.client.upload_count += 1
        if self.client.fail_upload_before is not None:
            raise _HTTPError(self.client.fail_upload_before)
        hook = self.client.before_upload
        if hook is not None:
            hook()
        assert kwargs["if_generation_match"] == 0
        assert kwargs["retry"] is UPLOAD_RETRY
        assert kwargs["content_type"] == publisher.CONTENT_TYPE
        size = int(kwargs["size"])
        raw = stream.read(size)
        assert len(raw) == size
        expected_crc32c = str(kwargs["crc32c_checksum_value"])
        if _crc32c(raw) != expected_crc32c:
            raise _HTTPError(400)
        if size <= publisher.CHUNK_BYTES:
            assert kwargs["checksum"] == "crc32c"
        else:
            assert kwargs["checksum"] is None
        if self.client.object is not None:
            raise _HTTPError(412)
        stored = _StoredObject(
            raw=raw,
            generation=self.client.next_generation,
            metageneration=1,
            metadata=dict(self.metadata or {}),
            content_type=str(kwargs["content_type"]),
            crc32c=_crc32c(raw),
        )
        self.client.next_generation += 1
        self.client.object = stored
        self._refresh(stored)
        after_commit = self.client.after_upload_commit
        if after_commit is not None:
            after_commit(stored)
        if self.client.fail_upload_after is not None:
            raise _HTTPError(self.client.fail_upload_after)

    def download_to_file(self, stream: Any, **kwargs: object) -> None:
        self.client._record("download", dict(kwargs))
        self.client.download_count += 1
        stored = self.client.object
        if stored is None:
            raise _HTTPError(404)
        if (
            self.requested_generation != stored.generation
            or kwargs.get("if_generation_match") != stored.generation
            or kwargs.get("if_metageneration_match") != stored.metageneration
        ):
            raise _HTTPError(412)
        assert kwargs["raw_download"] is True
        assert kwargs["checksum"] is None
        assert kwargs["single_shot_download"] is False
        assert kwargs["retry"] is READ_RETRY
        position = 0
        while position < len(stored.raw):
            chunk = stored.raw[position : position + publisher.CHUNK_BYTES]
            assert stream.write(chunk) == len(chunk)
            position += len(chunk)


def _canonical(value: object) -> bytes:
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


def _result(raw: bytes, *, commit: str = "a" * 40) -> dict[str, object]:
    return {
        "schema": publisher.RESULT_SCHEMA,
        "production_qualified": False,
        "repacker_sha256": "b" * 64,
        "repository_commit": commit,
        "derived_attestation_sha256": "c" * 64,
        "archive": {
            "media_type": publisher.CONTENT_TYPE,
            "format": "GNU-tar+gzip",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "member_count": 17,
            "expanded_regular_bytes": len(raw) * 2,
            "max_member_name_bytes": 71,
            "max_symlink_target_bytes": 0,
            "publication": "external-generation-pinned-regional-object",
        },
    }


@dataclass(frozen=True)
class _Fixture:
    archive: Path
    result: Path
    receipt: Path
    uri: str
    raw: bytes

    def request(self, **overrides: object) -> Any:
        values: dict[str, object] = {
            "archive": self.archive,
            "result": self.result,
            "destination_uri": self.uri,
            "expected_region": "europe-west4",
            "receipt": self.receipt,
        }
        values.update(overrides)
        return publisher.PublishRequest(**values)


def _runtime_fixture(tmp_path: Path, raw: bytes) -> _Fixture:
    digest = hashlib.sha256(raw).hexdigest()
    archive = tmp_path / "runtime.tar.gz"
    result = tmp_path / "result.json"
    receipt = tmp_path / "receipt.json"
    archive.write_bytes(raw)
    result.write_bytes(_canonical(_result(raw)))
    archive.chmod(0o444)
    result.chmod(0o444)
    return _Fixture(
        archive=archive,
        result=result,
        receipt=receipt,
        uri=f"gs://neural-dragnet-r1-euw4-655343590563/runtime/{digest}.tar.gz",
        raw=raw,
    )


@pytest.fixture
def runtime(tmp_path: Path) -> _Fixture:
    return _runtime_fixture(
        tmp_path,
        (b"derived-runtime\n" * 4096) + b"end\n",
    )


def _publish(runtime: _Fixture, client: _FakeClient, **kwargs: object) -> Any:
    return publisher.publish(
        runtime.request(**kwargs),
        client=client,
        upload_retry_policy=UPLOAD_RETRY,
        read_retry_policy=READ_RETRY,
        client_version=publisher.STORAGE_CLIENT_VERSION,
    )


def _expected_metadata(runtime: _Fixture) -> dict[str, str]:
    parsed = publisher.parse_repack_result(runtime.result.read_bytes())
    measurement = publisher.ArchiveMeasurement(
        sha256=hashlib.sha256(runtime.raw).hexdigest(),
        crc32c=_crc32c(runtime.raw),
        size_bytes=len(runtime.raw),
    )
    return publisher._metadata(parsed, measurement)


def _preexisting(
    runtime: _Fixture,
    *,
    metadata: dict[str, str] | None = None,
    component_count: int | None = None,
    content_type: str = publisher.CONTENT_TYPE,
) -> _StoredObject:
    return _StoredObject(
        raw=runtime.raw,
        generation=77,
        metageneration=3,
        metadata=_expected_metadata(runtime) if metadata is None else metadata,
        content_type=content_type,
        crc32c=_crc32c(runtime.raw),
        component_count=component_count,
    )


def test_fresh_create_is_direct_conditional_and_receipt_is_last(
    runtime: _Fixture,
) -> None:
    client = _FakeClient(receipt=runtime.receipt)

    receipt = _publish(runtime, client)

    assert client.upload_count == 1
    assert client.download_count == 1
    assert client.object is not None
    assert client.object.raw == runtime.raw
    assert client.object.component_count is None
    upload = next(kwargs for name, kwargs in client.calls if name == "upload")
    assert upload["if_generation_match"] == 0
    assert upload["retry"] is UPLOAD_RETRY
    assert upload["crc32c_checksum_value"] == _crc32c(runtime.raw)
    assert receipt["disposition"] == "created"
    assert receipt["object"]["generation"] == 101
    assert receipt["object"]["component_count"] is None
    assert runtime.receipt.read_bytes() == publisher._canonical_bytes(receipt)
    assert runtime.receipt.stat().st_mode & 0o777 == 0o444
    assert SCRIPT.stat().st_mode & 0o111
    source = SCRIPT.read_text(encoding="utf-8")
    assert ".compose(" not in source
    assert ".rewrite(" not in source
    assert ".delete(" not in source
    assert "if_generation_match=0" in source
    assert "DEFAULT_RETRY_IF_GENERATION_SPECIFIED" in source


def test_resumable_path_disables_unqualified_client_delete(
    runtime: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(publisher, "CHUNK_BYTES", 4096)
    client = _FakeClient(receipt=runtime.receipt)

    _publish(runtime, client)

    upload = next(kwargs for name, kwargs in client.calls if name == "upload")
    assert upload["checksum"] is None
    assert upload["crc32c_checksum_value"] == _crc32c(runtime.raw)
    assert client.object is not None
    assert client.object.component_count is None


def test_ambiguous_post_commit_is_reconciled_without_second_upload(
    runtime: _Fixture,
) -> None:
    client = _FakeClient(receipt=runtime.receipt)
    client.fail_upload_after = 503

    receipt = _publish(runtime, client)

    assert client.upload_count == 1
    assert client.download_count == 1
    assert receipt["disposition"] == "reconciled-existing"
    assert receipt["object"]["generation"] == 101


def test_success_cannot_discard_the_upload_response_generation(
    runtime: _Fixture,
) -> None:
    client = _FakeClient(receipt=runtime.receipt)

    def replace_with_exact_looking_generation(stored: _StoredObject) -> None:
        client.object = _StoredObject(
            raw=stored.raw,
            generation=stored.generation + 1,
            metageneration=1,
            metadata=dict(stored.metadata),
            content_type=stored.content_type,
            crc32c=stored.crc32c,
        )

    client.after_upload_commit = replace_with_exact_looking_generation

    with pytest.raises(publisher.PublicationError, match="exact GCS"):
        _publish(runtime, client)

    assert client.upload_count == 1
    assert client.download_count == 0
    assert not runtime.receipt.exists()


def test_exact_preexisting_object_is_adopted_without_upload(
    runtime: _Fixture,
) -> None:
    client = _FakeClient(receipt=runtime.receipt)
    client.object = _preexisting(runtime)

    receipt = _publish(runtime, client)

    assert client.upload_count == 0
    assert client.download_count == 1
    assert receipt["disposition"] == "reconciled-existing"
    assert receipt["object"]["generation"] == 77


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda stored: stored.metadata.update({"foreign": "bad"}),
            "incompatible",
        ),
        (
            lambda stored: setattr(stored, "component_count", 2),
            "incompatible",
        ),
        (
            lambda stored: setattr(stored, "content_type", "application/octet-stream"),
            "incompatible",
        ),
        (
            lambda stored: setattr(stored, "content_encoding", "gzip"),
            "incompatible",
        ),
        (
            lambda stored: setattr(stored, "crc32c", "AAAAAA=="),
            "incompatible",
        ),
    ],
)
def test_preexisting_mismatch_is_rejected_without_receipt(
    runtime: _Fixture,
    mutation: Callable[[_StoredObject], None],
    message: str,
) -> None:
    client = _FakeClient(receipt=runtime.receipt)
    client.object = _preexisting(runtime)
    mutation(client.object)

    with pytest.raises(publisher.PublicationConflict, match=message):
        _publish(runtime, client)

    assert client.upload_count == 0
    assert not runtime.receipt.exists()


def test_full_readback_rejects_a_real_crc32c_collision(tmp_path: Path) -> None:
    first = bytes.fromhex(
        "6a3b39f73b241c5b49a1a3d1cab2434470931181662d41f219d9c6ab68bb619d"
    )
    collision = bytes.fromhex(
        "2d8b79b064313266672f9c08fb465fcaf232bd1f27333ae842fba24eb35d97de"
    )
    assert first != collision
    assert _crc32c(first) == _crc32c(collision)
    runtime = _runtime_fixture(tmp_path, first)
    client = _FakeClient(receipt=runtime.receipt)
    client.object = _preexisting(runtime)
    client.object.raw = collision

    with pytest.raises(publisher.PublicationConflict, match="SHA-256"):
        _publish(runtime, client)

    assert client.upload_count == 0
    assert client.download_count == 1
    assert not runtime.receipt.exists()


@pytest.mark.parametrize("status", [408, 429, 500, 503, None])
def test_ambiguous_upload_without_exact_object_emits_no_receipt(
    runtime: _Fixture,
    status: int | None,
) -> None:
    client = _FakeClient(receipt=runtime.receipt)
    if status is None:
        client.fail_upload_before = 503

        def raise_without_status() -> None:
            raise OSError("connection reset")

        client.before_upload = raise_without_status
        client.fail_upload_before = None
    else:
        client.fail_upload_before = status

    with pytest.raises(publisher.PublicationError, match="cannot reconcile"):
        _publish(runtime, client)

    assert client.upload_count == 1
    assert client.object is None
    assert not runtime.receipt.exists()


def test_nonambiguous_upload_error_is_not_adopted(runtime: _Fixture) -> None:
    client = _FakeClient(receipt=runtime.receipt)
    client.fail_upload_before = 400

    with pytest.raises(publisher.PublicationError, match="upload failed"):
        _publish(runtime, client)

    assert sum(name == "reload" for name, _ in client.calls) == 1
    assert not runtime.receipt.exists()


def test_live_generation_change_is_fail_closed(runtime: _Fixture) -> None:
    client = _FakeClient(receipt=runtime.receipt)
    client.object = _preexisting(runtime)
    changed = False

    def replace_after_first_reload(_name: str, generation: int | None) -> None:
        nonlocal changed
        if generation is None and not changed:
            changed = True
            assert client.object is not None
            client.object.generation += 1

    client.after_reload = replace_after_first_reload

    with pytest.raises(publisher.PublicationError, match="cannot read exact"):
        _publish(runtime, client)

    assert not runtime.receipt.exists()


def test_source_mutation_during_upload_cannot_produce_receipt(
    runtime: _Fixture,
) -> None:
    client = _FakeClient(receipt=runtime.receipt)

    def mutate() -> None:
        runtime.archive.chmod(0o644)
        runtime.archive.write_bytes(b"x" * len(runtime.raw))

    client.before_upload = mutate

    with pytest.raises(publisher.PublicationError):
        _publish(runtime, client)

    assert client.object is None
    assert not runtime.receipt.exists()


def test_receipt_race_never_replaces_an_existing_file(runtime: _Fixture) -> None:
    client = _FakeClient(receipt=runtime.receipt)
    bucket_reloads = 0

    def occupy_after_final_verification() -> None:
        nonlocal bucket_reloads
        bucket_reloads += 1
        if bucket_reloads == 2:
            runtime.receipt.write_bytes(b"foreign\n")

    client.after_bucket_reload = occupy_after_final_verification

    with pytest.raises(publisher.PublicationError, match="replace"):
        _publish(runtime, client)

    assert runtime.receipt.read_bytes() == b"foreign\n"
    assert client.object is not None


def test_receipt_parent_swap_cannot_redirect_or_claim_success(
    runtime: _Fixture,
) -> None:
    client = _FakeClient(receipt=runtime.receipt)
    original_parent = runtime.receipt.parent
    moved_parent = original_parent.with_name(f"{original_parent.name}-held")
    bucket_reloads = 0

    def swap_parent_after_final_verification() -> None:
        nonlocal bucket_reloads
        bucket_reloads += 1
        if bucket_reloads == 2:
            original_parent.rename(moved_parent)
            original_parent.mkdir(mode=0o700)
            (original_parent / runtime.receipt.name).write_bytes(b"foreign\n")

    client.after_bucket_reload = swap_parent_after_final_verification

    with pytest.raises(publisher.PublicationError, match="parent pathname changed"):
        _publish(runtime, client)

    assert runtime.receipt.read_bytes() == b"foreign\n"
    anchored = moved_parent / runtime.receipt.name
    assert anchored.is_file()
    assert anchored.read_bytes() != b"foreign\n"
    assert client.object is not None


def test_wrong_region_or_unsafe_retry_fails_before_upload(runtime: _Fixture) -> None:
    wrong_region = _FakeClient(receipt=runtime.receipt, location="US-EAST1")
    with pytest.raises(publisher.PublicationError, match="expected region"):
        _publish(runtime, wrong_region)
    assert wrong_region.upload_count == 0

    unsafe = _FakeClient(receipt=runtime.receipt)
    with pytest.raises(publisher.PublicationError, match="generation-conditional"):
        publisher.publish(
            runtime.request(),
            client=unsafe,
            upload_retry_policy=_ConditionalRetry(unsafe=True),
            read_retry_policy=READ_RETRY,
            client_version=publisher.STORAGE_CLIENT_VERSION,
        )
    assert unsafe.calls == []


def test_bucket_location_and_metageneration_are_re_attested(
    runtime: _Fixture,
) -> None:
    client = _FakeClient(receipt=runtime.receipt)
    reloads = 0

    def mutate_bucket_after_initial_attestation() -> None:
        nonlocal reloads
        reloads += 1
        if reloads == 1:
            client.bucket_metageneration += 1

    client.after_bucket_reload = mutate_bucket_after_initial_attestation

    with pytest.raises(publisher.PublicationConflict, match="metageneration"):
        _publish(runtime, client)

    assert client.upload_count == 1
    assert client.download_count == 1
    assert not runtime.receipt.exists()


def test_result_and_content_address_are_exact(runtime: _Fixture) -> None:
    parsed = publisher.parse_repack_result(runtime.result.read_bytes())
    assert parsed.archive_sha256 == hashlib.sha256(runtime.raw).hexdigest()

    noncanonical = runtime.result.read_bytes().replace(b":", b": ", 1)
    with pytest.raises(publisher.PublicationError, match="canonical"):
        publisher.parse_repack_result(noncanonical)

    value = _result(runtime.raw)
    value["unknown"] = True
    with pytest.raises(publisher.PublicationError, match="unknown"):
        publisher.parse_repack_result(_canonical(value))

    client = _FakeClient(receipt=runtime.receipt)
    bad_uri = runtime.uri.replace("/runtime/", "/uploads/")
    with pytest.raises(publisher.PublicationError, match="SHA256.tar.gz"):
        _publish(runtime, client, destination_uri=bad_uri)
    assert client.calls == []


def test_symlink_or_writable_input_and_existing_receipt_are_rejected(
    runtime: _Fixture,
    tmp_path: Path,
) -> None:
    linked = tmp_path / "linked.tar.gz"
    linked.symlink_to(runtime.archive)
    client = _FakeClient(receipt=runtime.receipt)
    with pytest.raises(publisher.PublicationError, match="without following"):
        _publish(runtime, client, archive=linked)
    assert client.calls == []

    runtime.archive.chmod(0o644)
    with pytest.raises(publisher.PublicationError, match="read-only"):
        _publish(runtime, client)
    runtime.archive.chmod(0o444)

    runtime.receipt.write_text("occupied", encoding="utf-8")
    with pytest.raises(publisher.PublicationError, match="fresh"):
        _publish(runtime, client)
    assert client.calls == []


def test_client_version_is_exact(runtime: _Fixture) -> None:
    client = _FakeClient(receipt=runtime.receipt)
    with pytest.raises(publisher.PublicationError, match="3.13.0"):
        publisher.publish(
            runtime.request(),
            client=client,
            upload_retry_policy=UPLOAD_RETRY,
            read_retry_policy=READ_RETRY,
            client_version="3.14.0",
        )
    assert client.calls == []
