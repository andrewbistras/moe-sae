from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import uuid

import pytest

from moe_sae.object_store import (
    GCSObjectStore,
    LocalObjectStore,
    ObjectIdentity,
    ObjectStoreError,
    ShardCache,
    join_gcs_uri,
    object_identity_from_mapping,
    parse_gcs_uri,
    sha256_file,
)


class _FakeBlob:
    def __init__(
        self,
        state: dict[str, object],
        calls: list[tuple[object, ...]],
        bucket: "_FakeBucket",
        name: str,
    ):
        self._state = state
        self._calls = calls
        self.bucket = bucket
        self.name = name
        self.metadata: dict[str, str] | None = None
        self.generation: int | None = None
        self.size: int | None = None
        self.crc32c: str | None = None

    def _refresh(self) -> None:
        if "content" not in self._state:
            raise FileNotFoundError("object absent")
        self.metadata = dict(self._state["metadata"])
        self.generation = int(self._state["generation"])
        self.size = len(self._state["content"])
        self.crc32c = str(self._state["crc32c"])

    def upload_from_filename(self, filename: str, **kwargs: object) -> None:
        self._calls.append(("upload", kwargs))
        if "content" in self._state:
            raise RuntimeError("precondition failed")
        self._state.update(
            content=Path(filename).read_bytes(),
            metadata=dict(self.metadata or {}),
            generation=41,
            crc32c="AAAAAA==",
        )
        self._refresh()

    def reload(self, **kwargs: object) -> None:
        self._calls.append(("reload", kwargs))
        self._refresh()
        expected = kwargs.get("if_generation_match")
        if expected is not None and expected != self.generation:
            raise RuntimeError("generation mismatch")

    def download_to_filename(self, filename: str, **kwargs: object) -> None:
        self._calls.append(("download", kwargs))
        self.reload(if_generation_match=kwargs.get("if_generation_match"))
        Path(filename).write_bytes(self._state["content"])

    def delete(self, **kwargs: object) -> None:
        self._calls.append(("delete", kwargs))
        if kwargs.get("if_generation_match") != self._state.get("generation"):
            raise RuntimeError("generation mismatch")
        self._state.clear()


class _FakeBucket:
    def __init__(self, client: "_FakeClient", bucket: str):
        self.client = client
        self.bucket = bucket

    def blob(self, name: str) -> _FakeBlob:
        key = f"{self.bucket}/{name}"
        state = self.client.states.setdefault(key, {})
        return _FakeBlob(state, self.client.calls, self, name)

    def copy_blob(
        self,
        source: _FakeBlob,
        destination_bucket: "_FakeBucket",
        *,
        new_name: str,
        **kwargs: object,
    ) -> _FakeBlob:
        self.client.calls.append(("copy", kwargs))
        if kwargs.get("source_generation") != source._state.get("generation"):
            raise RuntimeError("source generation mismatch")
        if kwargs.get("if_source_generation_match") != source._state.get("generation"):
            raise RuntimeError("source precondition mismatch")
        destination = destination_bucket.blob(new_name)
        if "content" in destination._state or kwargs.get("if_generation_match") != 0:
            raise RuntimeError("destination precondition failed")
        destination._state.update(
            content=source._state["content"],
            metadata=dict(source._state["metadata"]),
            generation=73,
            crc32c=source._state["crc32c"],
        )
        destination._refresh()
        return destination


class _FakeClient:
    def __init__(self):
        self.states: dict[str, dict[str, object]] = {}
        self.calls: list[tuple[object, ...]] = []

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(self, name)


class _FakeTransferManager:
    THREAD = "thread"

    def __init__(self, calls: list[tuple[object, ...]]):
        self.calls = calls

    def upload_chunks_concurrently(
        self, filename: str, blob: _FakeBlob, **kwargs: object
    ) -> None:
        self.calls.append(("multipart", kwargs))
        if "content" in blob._state:
            raise RuntimeError("temporary collision")
        blob._state.update(
            content=Path(filename).read_bytes(),
            metadata=dict(blob.metadata or {}),
            generation=59,
            crc32c="AAAAAA==",
        )
        blob._refresh()


def _write(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_local_store_is_conditional_and_generation_bound(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    digest = _write(source, b"immutable payload")
    store = LocalObjectStore(tmp_path / "objects")
    uri = f"gs://bucket/capture/{digest}.safetensors"

    first = store.put_if_absent(source, uri, sha256=digest)
    second = store.put_if_absent(source, uri, sha256=digest)
    assert first == second
    assert object_identity_from_mapping(first.as_dict()) == first
    assert base64.b64decode(first.crc32c)
    store.verify(first)

    different = tmp_path / "different.bin"
    other_digest = _write(different, b"different payload")
    with pytest.raises(ObjectStoreError, match="collision"):
        store.put_if_absent(different, uri, sha256=other_digest)

    wrong_generation = ObjectIdentity(
        uri=first.uri,
        generation=first.generation + 1,
        size_bytes=first.size_bytes,
        crc32c=first.crc32c,
        sha256=first.sha256,
    )
    with pytest.raises(ObjectStoreError, match="changed"):
        store.verify(wrong_generation)


def test_gcs_store_uses_preconditions_checksums_and_pinned_reads(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    digest = _write(source, b"immutable payload")
    client = _FakeClient()
    store = GCSObjectStore(client)
    uri = f"gs://bucket/capture/{digest}.bin"

    identity = store.put_if_absent(source, uri, sha256=digest)
    assert identity.generation == 41
    assert ("upload", {"if_generation_match": 0, "checksum": "crc32c"}) in client.calls
    # Replaying the exact content takes the conditional-upload conflict path
    # and succeeds only after the existing object's identity is reloaded.
    assert store.put_if_absent(source, uri, sha256=digest) == identity
    destination = tmp_path / "download.bin"
    store.download(identity, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert (
        "download",
        {"if_generation_match": 41, "checksum": "auto"},
    ) in client.calls
    store.verify(identity)
    store.delete(identity)
    assert ("delete", {"if_generation_match": 41}) in client.calls


def test_gcs_large_upload_multipart_commits_conditionally_and_cleans_temp(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    digest = _write(source, b"parallel immutable payload")
    client = _FakeClient()
    transfer_manager = _FakeTransferManager(client.calls)
    store = GCSObjectStore(
        client,
        parallel_upload_threshold_bytes=1,
        parallel_upload_chunk_bytes=7,
        parallel_upload_workers=3,
        transfer_manager_module=transfer_manager,
    )
    uri = f"gs://bucket/capture/{digest}.bin"

    identity = store.put_if_absent(source, uri, sha256=digest)

    assert identity.generation == 73
    assert identity.sha256 == digest
    assert (
        "multipart",
        {
            "chunk_size": 7,
            "worker_type": transfer_manager.THREAD,
            "max_workers": 3,
            "checksum": "crc32c",
        },
    ) in client.calls
    assert (
        "copy",
        {
            "source_generation": 59,
            "if_generation_match": 0,
            "if_source_generation_match": 59,
        },
    ) in client.calls
    temporary_states = [
        state
        for name, state in client.states.items()
        if ".multipart-" in name
    ]
    assert temporary_states and all("content" not in state for state in temporary_states)
    assert client.states[f"bucket/capture/{digest}.bin"]["content"] == source.read_bytes()


def test_store_detects_corruption_and_generation_pinned_delete(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    digest = _write(source, b"payload")
    root = tmp_path / "objects"
    store = LocalObjectStore(root)
    identity = store.put_if_absent(
        source, "gs://bucket/path/object.bin", sha256=digest
    )
    object_path = root / "bucket/path/object.bin"
    object_path.write_bytes(b"corrupt")
    with pytest.raises(ObjectStoreError, match="changed"):
        store.verify(identity)
    with pytest.raises(ObjectStoreError, match="changed"):
        store.delete(identity)


def test_bounded_cache_rehashes_hits_and_evicts_lru(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    sources = []
    identities = []
    for index, payload in enumerate((b"a" * 16, b"b" * 16, b"c" * 16)):
        source = tmp_path / f"source-{index}.bin"
        digest = _write(source, payload)
        sources.append(source)
        identities.append(
            store.put_if_absent(
                source,
                f"gs://bucket/capture/{digest}.safetensors",
                sha256=digest,
            )
        )
    cache = ShardCache(tmp_path / "cache", store, max_bytes=32)
    first = cache.materialize(identities[0])
    second = cache.materialize(identities[1].as_dict())
    assert first.read_bytes() == b"a" * 16
    assert second.read_bytes() == b"b" * 16

    # A corrupt hit is deleted and fetched again before being returned.
    first.write_bytes(b"x" * 16)
    assert cache.materialize(identities[0]).read_bytes() == b"a" * 16
    third = cache.materialize(identities[2])
    assert third.read_bytes() == b"c" * 16
    cached = list((tmp_path / "cache").glob("*.safetensors"))
    assert sum(path.stat().st_size for path in cached) <= 32
    assert len(cached) == 2


def test_cache_rejects_one_object_larger_than_bound(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    digest = _write(source, b"too large")
    store = LocalObjectStore(tmp_path / "objects")
    identity = store.put_if_absent(
        source, "gs://bucket/path/object.bin", sha256=digest
    )
    cache = ShardCache(tmp_path / "cache", store, max_bytes=1)
    with pytest.raises(ObjectStoreError, match="exceeds the local cache bound"):
        cache.materialize(identity)


@pytest.mark.parametrize(
    "uri",
    ["bucket/path", "gs:///path", "gs://bucket", "gs://bucket/a/../b", "gs://bucket/a//b"],
)
def test_parse_gcs_uri_rejects_unsafe_names(uri: str) -> None:
    with pytest.raises(ValueError):
        parse_gcs_uri(uri)


def test_join_and_hash_helpers(tmp_path: Path) -> None:
    assert join_gcs_uri("gs://bucket/base/", "host-00000", "x.bin") == (
        "gs://bucket/base/host-00000/x.bin"
    )
    assert join_gcs_uri("gs://bucket", "x.bin") == "gs://bucket/x.bin"
    path = tmp_path / "data"
    path.write_bytes(b"abc")
    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        3,
    )


@pytest.mark.skipif(
    not os.environ.get("MOE_SAE_TEST_GCS_PREFIX"),
    reason="set MOE_SAE_TEST_GCS_PREFIX to opt into a real GCS round trip",
)
def test_real_gcs_round_trip_is_explicitly_opt_in(tmp_path: Path) -> None:
    prefix = os.environ["MOE_SAE_TEST_GCS_PREFIX"]
    source = tmp_path / "real-gcs-probe.bin"
    digest = _write(source, os.urandom(10 * 1024**2))
    uri = join_gcs_uri(prefix, "pytest", f"{uuid.uuid4().hex}-{digest}.bin")
    store = GCSObjectStore(parallel_upload_threshold_bytes=1)
    identity = store.put_if_absent(source, uri, sha256=digest)
    try:
        destination = tmp_path / "download.bin"
        store.download(identity, destination)
        assert sha256_file(destination) == (digest, source.stat().st_size)
        store.verify(identity)
    finally:
        store.delete(identity)
