from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import threading
from typing import Any, Callable

import pytest

from moe_sae.r1.control_store import CASConflict, ImmutableConflict
from moe_sae.r1.controller import CampaignController
from moe_sae.r1.gcs_control_store import (
    CorruptControlObject,
    GCSControlStoreError,
    GCSJSONCASStore,
)
from moe_sae.r1.protocol import HarvestRangeV1


def _state(value: int) -> dict[str, object]:
    return {"schema": "test.gcs-controller-state.v1", "value": value}


class _HTTPError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass
class _Object:
    value: bytes
    generation: int
    metageneration: int
    metadata: dict[str, str]
    content_type: str


class _FakeClient:
    """A generation-aware fake exposing no listing method."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], _Object] = {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.lock = threading.RLock()
        self.next_generation = 100
        self.before_conditional_upload: Callable[[], None] | None = None
        self.before_download: Callable[[str, str], None] | None = None
        self.fail_reload_status: int | None = None
        self.fail_after_upload_status: int | None = None
        self.download_override: Any | None = None

    def bucket(self, name: str) -> "_FakeBucket":
        return _FakeBucket(self, name)

    def force_replace(
        self,
        bucket: str,
        name: str,
        value: bytes,
        *,
        metadata: dict[str, str],
        content_type: str = "application/json",
    ) -> int:
        with self.lock:
            generation = self.next_generation
            self.next_generation += 1
            self.objects[(bucket, name)] = _Object(
                value=value,
                generation=generation,
                metageneration=1,
                metadata=dict(metadata),
                content_type=content_type,
            )
            return generation


class _FakeBucket:
    def __init__(self, client: _FakeClient, name: str) -> None:
        self.client = client
        self.name = name

    def blob(self, name: str) -> "_FakeBlob":
        return _FakeBlob(self.client, self.name, name)


class _FakeBlob:
    def __init__(self, client: _FakeClient, bucket: str, name: str) -> None:
        self.client = client
        self.bucket = bucket
        self.name = name
        self.metadata: dict[str, str] | None = None
        self.content_type: str | None = None
        self.generation: int | None = None
        self.metageneration: int | None = None
        self.size: int | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.bucket, self.name

    def _refresh(self, obj: _Object) -> None:
        self.metadata = dict(obj.metadata)
        self.content_type = obj.content_type
        self.generation = obj.generation
        self.metageneration = obj.metageneration
        self.size = len(obj.value)

    def reload(self, **kwargs: Any) -> None:
        self.client.calls.append(("reload", f"{self.bucket}/{self.name}", kwargs))
        if self.client.fail_reload_status is not None:
            status = self.client.fail_reload_status
            self.client.fail_reload_status = None
            raise _HTTPError(status, "injected reload failure")
        with self.client.lock:
            obj = self.client.objects.get(self.key)
            if obj is None:
                raise _HTTPError(404, "not found")
            self._refresh(obj)

    def download_as_bytes(self, **kwargs: Any) -> bytes:
        self.client.calls.append(("download", f"{self.bucket}/{self.name}", kwargs))
        hook = self.client.before_download
        if hook is not None:
            self.client.before_download = None
            hook(self.bucket, self.name)
        with self.client.lock:
            obj = self.client.objects.get(self.key)
            if obj is None:
                raise _HTTPError(404, "not found")
            if kwargs.get("if_generation_match") != obj.generation:
                raise _HTTPError(412, "generation mismatch")
            if kwargs.get("if_metageneration_match") != obj.metageneration:
                raise _HTTPError(412, "metageneration mismatch")
            if self.client.download_override is not None:
                return self.client.download_override
            return bytes(obj.value)

    def upload_from_string(self, value: bytes, **kwargs: Any) -> None:
        assert type(value) is bytes
        self.client.calls.append(("upload", f"{self.bucket}/{self.name}", kwargs))
        expected = kwargs.get("if_generation_match")
        hook = self.client.before_conditional_upload
        if expected != 0 and hook is not None:
            hook()
        with self.client.lock:
            current = self.client.objects.get(self.key)
            if expected == 0:
                if current is not None:
                    raise _HTTPError(412, "create precondition failed")
            elif current is None or current.generation != expected:
                raise _HTTPError(412, "replace generation precondition failed")
            expected_meta = kwargs.get("if_metageneration_match")
            if expected_meta is not None and (
                current is None or current.metageneration != expected_meta
            ):
                raise _HTTPError(412, "replace metageneration precondition failed")
            generation = self.client.next_generation
            self.client.next_generation += 1
            obj = _Object(
                value=bytes(value),
                generation=generation,
                metageneration=1,
                metadata=dict(self.metadata or {}),
                content_type=str(kwargs["content_type"]),
            )
            self.client.objects[self.key] = obj
            self._refresh(obj)
        if self.client.fail_after_upload_status is not None:
            status = self.client.fail_after_upload_status
            self.client.fail_after_upload_status = None
            raise _HTTPError(status, "injected post-commit transport failure")


def _store() -> tuple[GCSJSONCASStore, _FakeClient]:
    client = _FakeClient()
    return GCSJSONCASStore(client, timeout_seconds=17), client


def _control_metadata(kind: str, payload: bytes) -> dict[str, str]:
    return {
        "moe-sae-r1-control-schema": "v1",
        "moe-sae-r1-control-kind": kind,
        "moe-sae-r1-control-sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_create_read_and_cas_use_exact_bytes_and_generation_preconditions() -> None:
    store, client = _store()
    uri = "gs://r1-control/campaigns/test/head.json"

    created = store.create(uri, _state(1))
    updated = store.compare_and_swap(uri, created.generation, _state(2))
    read = store.read(uri)

    assert created.value == b'{"schema":"test.gcs-controller-state.v1","value":1}'
    assert updated.value == b'{"schema":"test.gcs-controller-state.v1","value":2}'
    assert read == updated
    assert updated.generation != created.generation
    uploads = [call for call in client.calls if call[0] == "upload"]
    assert uploads[0][2] == {
        "content_type": "application/json",
        "if_generation_match": 0,
        "checksum": "crc32c",
        "timeout": 17.0,
    }
    assert uploads[1][2]["if_generation_match"] == created.generation
    assert uploads[1][2]["if_metageneration_match"] == 1
    downloads = [call for call in client.calls if call[0] == "download"]
    assert all(call[2]["checksum"] == "crc32c" for call in downloads)
    assert all("if_generation_match" in call[2] for call in downloads)
    assert not hasattr(client, "list_blobs")


def test_absent_and_stale_generations_are_precise_cas_conflicts() -> None:
    store, _client = _store()
    uri = "gs://r1-control/campaigns/test/head.json"

    with pytest.raises(CASConflict) as absent:
        store.compare_and_swap(uri, 99, _state(2))
    assert absent.value.actual_generation is None

    created = store.create(uri, _state(1))
    with pytest.raises(CASConflict) as duplicate:
        store.create(uri, _state(2))
    assert duplicate.value.expected_generation is None
    assert duplicate.value.actual_generation == created.generation

    updated = store.compare_and_swap(uri, created.generation, _state(2))
    with pytest.raises(CASConflict) as stale:
        store.compare_and_swap(uri, created.generation, _state(3))
    assert stale.value.expected_generation == created.generation
    assert stale.value.actual_generation == updated.generation


def test_concurrent_cas_has_one_generation_fenced_winner() -> None:
    store, client = _store()
    uri = "gs://r1-control/campaigns/test/head.json"
    initial = store.create(uri, _state(0))
    barrier = threading.Barrier(2)
    client.before_conditional_upload = barrier.wait

    def contender(value: int) -> tuple[str, int]:
        try:
            won = store.compare_and_swap(uri, initial.generation, _state(value))
            return "won", won.generation
        except CASConflict as error:
            assert error.actual_generation is not None
            return "lost", error.actual_generation

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(contender, (1, 2)))

    assert sorted(result for result, _generation in results) == ["lost", "won"]
    assert len({generation for _result, generation in results}) == 1
    assert store.read(uri).decoded()["value"] in {1, 2}  # type: ignore[union-attr]


def test_immutable_publication_is_idempotent_and_cannot_be_reclassified() -> None:
    store, _client = _store()
    event_uri = "gs://r1-control/campaigns/test/events/event.json"
    head_uri = "gs://r1-control/campaigns/test/head.json"

    first = store.put_immutable(event_uri, _state(1))
    retry = store.put_immutable(
        event_uri,
        b'{ "value": 1, "schema": "test.gcs-controller-state.v1" }',
    )
    assert retry == first
    with pytest.raises(ImmutableConflict, match="different content"):
        store.put_immutable(event_uri, _state(2))
    with pytest.raises(ImmutableConflict, match="cannot be replaced"):
        store.compare_and_swap(event_uri, first.generation, _state(2))

    store.create(head_uri, _state(1))
    with pytest.raises(ImmutableConflict, match="mutable"):
        store.put_immutable(head_uri, _state(1))


def test_generation_pinned_read_retries_a_replace_between_metadata_and_bytes() -> None:
    store, client = _store()
    uri = "gs://r1-control/campaigns/test/head.json"
    store.create(uri, _state(1))
    replacement = b'{"schema":"test.gcs-controller-state.v1","value":2}'
    replacement_generation: list[int] = []

    def replace(bucket: str, name: str) -> None:
        replacement_generation.append(
            client.force_replace(
                bucket,
                name,
                replacement,
                metadata=_control_metadata("mutable", replacement),
            )
        )

    client.before_download = replace
    read = store.read(uri)

    assert read is not None
    assert read.generation == replacement_generation[0]
    assert read.value == replacement
    downloads = [call for call in client.calls if call[0] == "download"]
    assert len(downloads) == 2


@pytest.mark.parametrize(
    "uri",
    [
        "memory://control/head.json",
        "gs:///head.json",
        "gs://bucket",
        "gs://bucket/path/",
        "gs://bucket/a//b.json",
        "gs://bucket/a/../b.json",
    ],
)
def test_gcs_store_rejects_non_object_or_unsafe_uris(uri: str) -> None:
    store, _client = _store()
    with pytest.raises(ValueError):
        store.read(uri)


def test_corrupt_metadata_noncanonical_json_and_nonbytes_download_fail_closed() -> None:
    store, client = _store()
    uri = "gs://r1-control/campaigns/test/head.json"
    bucket, name = "r1-control", "campaigns/test/head.json"
    noncanonical = b'{ "schema": "test.gcs-controller-state.v1", "value": 1 }'
    client.force_replace(
        bucket,
        name,
        noncanonical,
        metadata=_control_metadata("mutable", noncanonical),
    )
    with pytest.raises(CorruptControlObject, match="not canonical"):
        store.read(uri)

    canonical = b'{"schema":"test.gcs-controller-state.v1","value":1}'
    client.force_replace(bucket, name, canonical, metadata={})
    with pytest.raises(CorruptControlObject, match="metadata"):
        store.read(uri)

    client.force_replace(
        bucket,
        name,
        canonical,
        metadata=_control_metadata("mutable", canonical),
    )
    client.download_override = bytearray(canonical)
    with pytest.raises(CorruptControlObject, match="non-bytes"):
        store.read(uri)


def test_transport_failure_is_not_treated_as_absence() -> None:
    store, client = _store()
    client.fail_reload_status = 503
    with pytest.raises(GCSControlStoreError, match="metadata"):
        store.read("gs://r1-control/campaigns/test/head.json")


@pytest.mark.parametrize("status", [412, 503])
def test_ambiguous_post_commit_cas_is_reconciled_only_at_new_exact_generation(
    status: int,
) -> None:
    store, client = _store()
    uri = "gs://r1-control/campaigns/test/head.json"
    created = store.create(uri, _state(1))
    client.fail_after_upload_status = status

    updated = store.compare_and_swap(uri, created.generation, _state(2))

    assert updated.generation != created.generation
    assert updated.value == b'{"schema":"test.gcs-controller-state.v1","value":2}'
    assert store.read(uri) == updated


def test_campaign_controller_replays_entire_accepted_chain_through_gcs_api() -> None:
    store, client = _store()
    clock_values = iter((1_000, 1_100))
    controller = CampaignController(
        store,
        base_uri="gs://r1-control/control",
        campaign_id="integration",
        clock_ms=clock_values.__next__,
        content_verifier=lambda _content: True,
    )
    work = HarvestRangeV1(
        scientific_identity_sha256="a" * 64,
        corpus_fingerprint_sha256="b" * 64,
        split="train",
        record_start=0,
        record_end=100,
        scientific_token_start=0,
        scientific_token_end=100_000,
    )

    initialized = controller.initialize((work,))
    lease = controller.acquire(
        work.work_id,
        attempt_id="attempt-1",
        executor_id="executor-1",
        deployment_identity_sha256="c" * 64,
        lease_duration_ms=10_000,
    )
    replayed = controller.replay()

    assert initialized.revision == 0
    assert replayed.revision == 1
    assert replayed.work(work.work_id).lease == lease
    assert replayed.event_count == 2
    assert all(call[0] in {"reload", "download", "upload"} for call in client.calls)


def test_constructor_and_expected_generation_reject_bools_and_nonfinite_values() -> (
    None
):
    client = _FakeClient()
    with pytest.raises(ValueError, match="timeout_seconds"):
        GCSJSONCASStore(client, timeout_seconds=float("nan"))
    with pytest.raises(ValueError, match="positive integer"):
        GCSJSONCASStore(client, consistent_read_attempts=True)
    store = GCSJSONCASStore(client)
    with pytest.raises(ValueError, match="positive integer"):
        store.compare_and_swap(
            "gs://r1-control/campaigns/test/head.json", True, _state(1)
        )
