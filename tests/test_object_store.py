from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace
import uuid

import google_crc32c
import pytest
import moe_sae._gcs_transfer as gcs_transfer_module
import moe_sae.object_store as object_store_module

from moe_sae.object_store import (
    GCSObjectStore,
    LocalObjectStore,
    ObjectIdentity,
    ObjectStoreError,
    ShardCache,
    fingerprint_file,
    join_gcs_uri,
    object_identity_from_mapping,
    parse_gcs_uri,
    sha256_file,
)


def _crc32c(payload: bytes) -> str:
    return base64.b64encode(
        google_crc32c.value(payload).to_bytes(4, "big")
    ).decode("ascii")


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
        content = Path(filename).read_bytes()
        self._state.update(
            content=content,
            metadata=dict(self.metadata or {}),
            generation=41,
            crc32c=_crc32c(content),
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
    PROCESS = "process"

    def __init__(self, calls: list[tuple[object, ...]]):
        self.calls = calls

    def upload_chunks_concurrently(
        self, filename: str, blob: _FakeBlob, **kwargs: object
    ) -> None:
        self.calls.append(("multipart", kwargs))
        if "content" in blob._state:
            raise RuntimeError("temporary collision")
        content = Path(filename).read_bytes()
        blob._state.update(
            content=content,
            metadata=dict(blob.metadata or {}),
            generation=59,
            crc32c=_crc32c(content),
        )
        blob._refresh()

    def download_chunks_concurrently(
        self, blob: _FakeBlob, filename: str, **kwargs: object
    ) -> None:
        self.calls.append(("multipart-download", kwargs))
        blob._refresh()
        Path(filename).write_bytes(blob._state["content"])


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


def test_precomputed_source_fingerprint_skips_rehash_and_is_state_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.bin"
    digest = _write(source, b"preverified immutable payload")
    fingerprint = fingerprint_file(source)
    client = _FakeClient()
    transfer_manager = _FakeTransferManager(client.calls)
    store = GCSObjectStore(
        client,
        parallel_upload_threshold_bytes=1,
        transfer_manager_module=transfer_manager,
    )
    uri = f"gs://bucket/capture/{digest}.bin"

    def unexpected_rehash(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("fingerprinted source was redundantly rehashed")

    monkeypatch.setattr(object_store_module, "fingerprint_file", unexpected_rehash)
    identity = store.put_if_absent(
        source,
        uri,
        sha256=digest,
        source_fingerprint=fingerprint,
    )
    assert identity.sha256 == digest
    assert any(call[0] == "multipart" for call in client.calls)

    changed = tmp_path / "changed.bin"
    changed.write_bytes(b"first")
    stale = object_store_module._fingerprint_from_components(
        changed,
        sha256=hashlib.sha256(b"first").hexdigest(),
        crc32c=_crc32c(b"first"),
        size_bytes=5,
    )
    changed.write_bytes(b"other")
    with pytest.raises(ObjectStoreError, match="changed after it was fingerprinted"):
        store.put_if_absent(
            changed,
            "gs://bucket/capture/changed.bin",
            sha256=stale.sha256,
            source_fingerprint=stale,
        )


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
            "deadline": 1800.0,
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


def test_gcs_rejects_wrong_local_sha_before_upload(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    _write(source, b"local identity")
    client = _FakeClient()
    store = GCSObjectStore(client)

    with pytest.raises(ObjectStoreError, match="claimed SHA-256"):
        store.put_if_absent(
            source,
            "gs://bucket/capture/wrong.bin",
            sha256="0" * 64,
        )

    assert client.calls == []


def test_isolated_upload_helper_rejects_whole_object_crc_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    digest = _write(source, b"whole object checksum")
    client = _FakeClient()
    blob = client.bucket("bucket").blob("temporary")

    class _BadCRCTransferManager(_FakeTransferManager):
        def upload_chunks_concurrently(
            self, filename: str, target: _FakeBlob, **kwargs: object
        ) -> None:
            super().upload_chunks_concurrently(filename, target, **kwargs)
            target._state["crc32c"] = "AAAAAA=="
            target._refresh()

    manager = _BadCRCTransferManager(client.calls)
    monkeypatch.setattr(
        gcs_transfer_module,
        "_client_and_blob",
        lambda _uri: (client, blob),
    )
    monkeypatch.setattr(
        gcs_transfer_module,
        "_transfer_manager",
        lambda: manager,
    )
    args = SimpleNamespace(
        source=str(source),
        uri="gs://bucket/temporary",
        sha256=digest,
        crc32c=_crc32c(source.read_bytes()),
        chunk_bytes=7,
        workers=3,
        deadline_seconds=30.0,
    )

    with pytest.raises(RuntimeError, match="invalid GCS identity"):
        gcs_transfer_module._upload_temporary(args)

    assert "content" not in client.states["bucket/temporary"]
    assert (
        "delete",
        {"if_generation_match": 59},
    ) in client.calls


def test_parent_rejects_bad_multipart_crc_before_conditional_commit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    digest = _write(source, b"bad multipart checksum")
    client = _FakeClient()

    class _BadCRCTransferManager(_FakeTransferManager):
        def upload_chunks_concurrently(
            self, filename: str, target: _FakeBlob, **kwargs: object
        ) -> None:
            super().upload_chunks_concurrently(filename, target, **kwargs)
            target._state["crc32c"] = "AAAAAA=="
            target._refresh()

    store = GCSObjectStore(
        client,
        parallel_upload_threshold_bytes=1,
        transfer_manager_module=_BadCRCTransferManager(client.calls),
    )
    uri = f"gs://bucket/capture/{digest}.bin"

    with pytest.raises(ObjectStoreError, match="temporary GCS identity"):
        store.put_if_absent(source, uri, sha256=digest)

    assert not any(call[0] == "copy" for call in client.calls)
    assert all(
        "content" not in state
        for name, state in client.states.items()
        if ".multipart-" in name
    )


def test_gcs_large_download_uses_parallel_generation_pinned_ranges(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    digest = _write(source, b"parallel immutable download")
    client = _FakeClient()
    transfer_manager = _FakeTransferManager(client.calls)
    upload_store = GCSObjectStore(client)
    uri = f"gs://bucket/capture/{digest}.bin"
    identity = upload_store.put_if_absent(source, uri, sha256=digest)
    store = GCSObjectStore(
        client,
        parallel_download_threshold_bytes=1,
        parallel_download_chunk_bytes=11,
        parallel_download_workers=5,
        parallel_worker_type="process",
        transfer_manager_module=transfer_manager,
    )

    destination = tmp_path / "download.bin"
    store.download(identity, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert (
        "multipart-download",
        {
            "chunk_size": 11,
            "download_kwargs": {"if_generation_match": 41},
            "worker_type": transfer_manager.PROCESS,
            "max_workers": 5,
            "crc32c_checksum": True,
            "deadline": 1800.0,
        },
    ) in client.calls


def test_gcs_isolated_large_upload_keeps_conditional_commit_in_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    digest = _write(source, b"isolated immutable upload")
    client = _FakeClient()
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object):
        commands.append(command)
        uri = command[command.index("--uri") + 1]
        bucket, name = parse_gcs_uri(uri)
        state = client.states.setdefault(f"{bucket}/{name}", {})
        state.update(
            content=source.read_bytes(),
            metadata={"sha256": digest},
            generation=59,
            crc32c=_crc32c(source.read_bytes()),
        )
        return type("Completed", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr("moe_sae.object_store.subprocess.run", run)
    store = GCSObjectStore(
        client,
        parallel_upload_threshold_bytes=1,
        isolated_large_transfers=True,
    )
    uri = f"gs://bucket/capture/{digest}.bin"

    identity = store.put_if_absent(source, uri, sha256=digest)

    assert identity.generation == 73
    assert commands and commands[0][3] == "upload-temp"
    assert commands[0][commands[0].index("--chunk-bytes") + 1] == str(16 * 1024**2)
    assert commands[0][commands[0].index("--workers") + 1] == "32"
    assert any(call[0] == "copy" for call in client.calls)
    assert all(
        "content" not in state
        for name, state in client.states.items()
        if ".multipart-" in name
    )


def test_gcs_isolated_large_download_is_generation_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    digest = _write(source, b"isolated immutable download")
    client = _FakeClient()
    upload_store = GCSObjectStore(client)
    uri = f"gs://bucket/capture/{digest}.bin"
    identity = upload_store.put_if_absent(source, uri, sha256=digest)
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object):
        commands.append(command)
        destination = Path(command[command.index("--destination") + 1])
        destination.write_bytes(source.read_bytes())
        return type("Completed", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr("moe_sae.object_store.subprocess.run", run)
    store = GCSObjectStore(
        client,
        parallel_download_threshold_bytes=1,
        isolated_large_transfers=True,
    )
    destination = tmp_path / "download.bin"

    store.download(identity, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert commands and commands[0][3] == "download"
    assert commands[0][commands[0].index("--generation") + 1] == "41"
    assert commands[0][commands[0].index("--chunk-bytes") + 1] == str(32 * 1024**2)
    assert commands[0][commands[0].index("--workers") + 1] == "16"


def test_gcs_isolated_transfer_timeout_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    digest = _write(source, b"bounded helper")
    client = _FakeClient()
    observed: dict[str, float] = {}

    def run(command: list[str], **kwargs: object):
        observed["timeout"] = float(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("moe_sae.object_store.subprocess.run", run)
    store = GCSObjectStore(
        client,
        parallel_upload_threshold_bytes=1,
        transfer_timeout_seconds=30.0,
        isolated_large_transfers=True,
    )

    with pytest.raises(ObjectStoreError, match="90-second hard deadline"):
        store.put_if_absent(
            source,
            f"gs://bucket/capture/{digest}.bin",
            sha256=digest,
        )

    assert observed == {"timeout": 90.0}
    assert not any(call[0] == "copy" for call in client.calls)


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


def test_cache_materialize_many_downloads_four_objects_concurrently(
    tmp_path: Path,
) -> None:
    local = LocalObjectStore(tmp_path / "objects")
    identities = []
    for index in range(4):
        source = tmp_path / f"source-{index}.bin"
        digest = _write(source, bytes([index + 1]) * 64)
        identities.append(
            local.put_if_absent(
                source,
                f"gs://bucket/concurrent/{digest}.safetensors",
                sha256=digest,
            )
        )

    class ConcurrentStore:
        def __init__(self) -> None:
            self.barrier = threading.Barrier(4)
            self.lock = threading.Lock()
            self.active = 0
            self.peak = 0

        def download(self, identity, destination):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            self.barrier.wait(timeout=5)
            time.sleep(0.01)
            try:
                local.download(identity, destination)
            finally:
                with self.lock:
                    self.active -= 1

    store = ConcurrentStore()
    cache = ShardCache(tmp_path / "cache", store, max_bytes=4 * 64)
    paths = cache.materialize_many(identities, max_workers=4)

    assert store.peak == 4
    assert [path.read_bytes() for path in paths] == [
        bytes([index + 1]) * 64 for index in range(4)
    ]
    assert not list((tmp_path / "cache").glob("*.partial"))


def test_resident_only_cache_never_refills_missing_or_corrupt_objects(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    digest = _write(source, b"qualified resident object")
    store = LocalObjectStore(tmp_path / "objects")
    identity = store.put_if_absent(
        source,
        f"gs://bucket/qualified/{digest}.safetensors",
        sha256=digest,
    )
    cache_root = tmp_path / "cache"
    cached = ShardCache(cache_root, store, max_bytes=1024).materialize(identity)

    class NoDownloadStore:
        def download(self, *_args, **_kwargs):
            pytest.fail("resident-only cache attempted an unqualified refill")

    resident = ShardCache(
        cache_root,
        NoDownloadStore(),
        max_bytes=1024,
        resident_only=True,
    )
    assert resident.materialize_many((identity,)) == (cached,)

    cached.write_bytes(b"corrupt")
    with pytest.raises(ObjectStoreError, match="requalify a fresh SAE cache"):
        resident.materialize(identity)
    assert cached.read_bytes() == b"corrupt"

    cached.unlink()
    with pytest.raises(ObjectStoreError, match="requalify a fresh SAE cache"):
        resident.materialize_many((identity,))


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
    store = GCSObjectStore(
        parallel_upload_threshold_bytes=1,
        parallel_download_threshold_bytes=1,
    )
    identity = store.put_if_absent(source, uri, sha256=digest)
    try:
        destination = tmp_path / "download.bin"
        store.download(identity, destination)
        assert sha256_file(destination) == (digest, source.stat().st_size)
        store.verify(identity)
    finally:
        store.delete(identity)
