from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import threading

import ml_dtypes
import numpy as np
import pytest

from moe_sae.r1.checkpoint_conversion import (
    ConversionMemoryLimitsV1,
    build_synthetic_prefix_conversion_plan,
    execute_streaming_conversion,
)
from moe_sae.r1.checkpoint_inventory import (
    BF16_DTYPE,
    FP8_DTYPE,
    CheckpointInventorySelectionV1,
    DeepSeekR1CheckpointGeometryV1,
    build_synthetic_checkpoint_inventory_selection,
    decode_safetensor_header_bytes,
    decode_strict_json_bytes_with_receipt,
    expected_full_source_specs,
)
from moe_sae.r1.prefix_checkpoint import (
    ImmutableObjectConflict,
    ImmutablePhysicalObjectReceiptV1,
    ImmutablePrefixChunkSinkV1,
    LocalImmutableByteStore,
    ObservationSnapshotV1,
    PrefixCheckpointError,
    PrefixCheckpointManifestV1,
    PrefixCheckpointReaderV1,
    finalize_prefix_checkpoint,
)
from moe_sae.r1.protocol import ProtocolError, canonical_sha256
from moe_sae.r1.safetensor_source import (
    LocalVerifiedSafetensorSource,
    local_safetensor_source_identity,
)
import moe_sae.r1.prefix_checkpoint as prefix_checkpoint_module


BF16 = np.dtype(ml_dtypes.bfloat16)
FP8 = np.dtype(ml_dtypes.float8_e4m3fn)
F32 = np.dtype("<f4")


class _Fixture:
    def __init__(
        self,
        root: Path,
        selection: CheckpointInventorySelectionV1,
        arrays: dict[str, np.ndarray],
    ) -> None:
        self.root = root
        self.selection = selection
        self.arrays = arrays


def _shard_for(key: str, geometry: DeepSeekR1CheckpointGeometryV1) -> str:
    logical = key.removesuffix("_scale_inv")
    if logical == "model.embed_tokens.weight":
        number = 1
    elif logical.startswith(f"model.layers.{geometry.target_layer_zero_indexed}."):
        number = 2
    elif logical.startswith("model.layers."):
        layer = int(logical.split(".")[2])
        number = 1 if layer < geometry.target_layer_zero_indexed else 3
    else:
        number = 3
    return f"model-{number:05d}-of-000003.safetensors"


def _array_for(key: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    values = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    salt = sum(key.encode("utf-8")) % 11
    if dtype == FP8_DTYPE:
        return ((values + salt) % 15 - 7).astype(FP8)
    if dtype == BF16_DTYPE:
        return ((values + salt) / 8 - 3).astype(BF16)
    if key.endswith("_scale_inv"):
        return (0.75 + ((values + salt) % 7) / 16).astype(F32)
    return ((values.astype(np.float32) + salt) / 13 - 2).astype(F32)


def _build_fixture(root: Path) -> _Fixture:
    root.mkdir(parents=True)
    geometry = DeepSeekR1CheckpointGeometryV1(
        vocab_size=11,
        hidden_dim=6,
        main_layers=4,
        first_dense_layers=1,
        target_layer_zero_indexed=2,
        attention_heads=2,
        q_lora_rank=3,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        value_head_dim=2,
        dense_mlp_dim=8,
        routed_mlp_dim=5,
        routed_experts=2,
        shared_experts=1,
        fp8_block_size=4,
    )
    by_shard: defaultdict[str, list[tuple[str, str, tuple[int, ...]]]] = defaultdict(
        list
    )
    for spec in expected_full_source_specs(geometry):
        by_shard[_shard_for(spec.key, geometry)].append(
            (spec.key, spec.dtype, spec.shape)
        )
    arrays: dict[str, np.ndarray] = {}
    headers: dict[str, dict[str, object]] = {}
    descriptors: dict[str, dict[str, object]] = {}
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard in sorted(by_shard):
        header: dict[str, object] = {"__metadata__": {"format": "pt"}}
        payloads: list[bytes] = []
        cursor = 0
        for key, dtype, shape in sorted(by_shard[shard]):
            array = _array_for(key, dtype, shape)
            raw = array.tobytes(order="C")
            arrays[key] = array
            header[key] = {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [cursor, cursor + len(raw)],
            }
            weight_map[key] = shard
            payloads.append(raw)
            cursor += len(raw)
            total_size += len(raw)
        encoded = json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        raw_header = encoded + b" " * ((-len(encoded)) % 8)
        blob = struct.pack("<Q", len(raw_header)) + raw_header + b"".join(payloads)
        (root / shard).write_bytes(blob)
        decoded, descriptor = decode_safetensor_header_bytes(
            raw_header,
            shard=shard,
            expected_lfs_sha256=hashlib.sha256(blob).hexdigest(),
            expected_blob_size_bytes=len(blob),
        )
        assert decoded == header
        headers[shard] = header
        descriptors[shard] = descriptor
    index: dict[str, object] = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    raw_index = json.dumps(
        index, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    index_sha = hashlib.sha256(raw_index).hexdigest()
    _, index_receipt = decode_strict_json_bytes_with_receipt(
        raw_index, expected_sha256=index_sha, maximum_bytes=len(raw_index)
    )
    selection = build_synthetic_checkpoint_inventory_selection(
        index,
        headers,
        verified_index_sha256=index_sha,
        index_decode_receipt=index_receipt,
        shard_descriptors=descriptors,
        geometry=geometry,
    )
    return _Fixture(root, selection, arrays)


@pytest.fixture
def wired(tmp_path: Path) -> _Fixture:
    return _build_fixture(tmp_path / "source")


def _plan(
    fixture: _Fixture,
    *,
    max_output_chunk_bytes: int = 64,
    max_resident_bytes: int = 1024,
):
    return build_synthetic_prefix_conversion_plan(
        fixture.selection,
        ConversionMemoryLimitsV1(
            max_source_blob_bytes=1024**2,
            max_resident_bytes=max_resident_bytes,
            max_output_chunk_bytes=max_output_chunk_bytes,
            sink_staging_buffer_bytes=32,
        ),
        source_adapter_identity=local_safetensor_source_identity(),
    )


class _CrashAfterCommitted:
    def __init__(self, delegate: ImmutablePrefixChunkSinkV1, count: int) -> None:
        self.delegate = delegate
        self.count = count
        self.committed = 0

    @property
    def sink_id(self) -> str:
        return self.delegate.sink_id

    def probe_committed(self, plan, chunk):
        return self.delegate.probe_committed(plan, chunk)

    def commit_chunk(self, plan, chunk, payload, *, output_sha256):
        receipt = self.delegate.commit_chunk(
            plan, chunk, payload, output_sha256=output_sha256
        )
        self.committed += 1
        if self.committed == self.count:
            raise RuntimeError("injected crash after durable chunk commit")
        return receipt


class _FailFirstBindingPublication:
    def __init__(self, delegate: LocalImmutableByteStore) -> None:
        self.delegate = delegate
        self.failed = False

    @property
    def store_id(self) -> str:
        return self.delegate.store_id

    @property
    def stream_buffer_bytes(self) -> int:
        return self.delegate.stream_buffer_bytes

    @property
    def distributed_identity_qualified(self) -> bool:
        return self.delegate.distributed_identity_qualified

    def create_or_verify(
        self, object_id, payload, *, expected_sha256, expected_size_bytes
    ):
        if "/receipts/" in object_id and not self.failed:
            self.failed = True
            raise RuntimeError("injected crash before binding publication")
        return self.delegate.create_or_verify(
            object_id,
            payload,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )

    def probe_bounded(self, object_id, *, maximum_bytes):
        return self.delegate.probe_bounded(object_id, maximum_bytes=maximum_bytes)

    def verify(self, receipt):
        return self.delegate.verify(receipt)

    def read_exact(self, receipt, *, offset, length, maximum_bytes):
        return self.delegate.read_exact(
            receipt,
            offset=offset,
            length=length,
            maximum_bytes=maximum_bytes,
        )


def _expected_destinations(fixture: _Fixture) -> dict[str, np.ndarray]:
    geometry = fixture.selection.full.geometry
    result: dict[str, np.ndarray] = {}
    for mapping in fixture.selection.mappings:
        dtype = BF16 if mapping.destination_dtype == "bfloat16" else F32
        target = result.setdefault(
            mapping.destination_key,
            np.zeros(int(np.prod(mapping.destination_shape)), dtype=dtype),
        )
        source = fixture.arrays[mapping.source_key]
        if mapping.transform == "identity-v1":
            converted = source.reshape(-1)
        elif mapping.transform == "transpose-reshape-v1":
            converted = np.ascontiguousarray(source.T).reshape(-1)
        else:
            scale = fixture.arrays[mapping.scale_key]
            matrix = np.empty((source.shape[1], source.shape[0]), dtype=BF16)
            for row in range(source.shape[0]):
                for column in range(source.shape[1]):
                    matrix[column, row] = ml_dtypes.bfloat16(
                        np.float32(source[row, column])
                        * np.float32(
                            scale[
                                row // geometry.fp8_block_size,
                                column // geometry.fp8_block_size,
                            ]
                        )
                    )
            converted = matrix.reshape(-1)
        if mapping.destination_slice_axis is None:
            start = 0
        else:
            assert mapping.destination_slice_axis == 0
            slice_size = int(np.prod(mapping.destination_shape[1:]))
            start = mapping.destination_slice_index * slice_size
        target[start : start + converted.size] = converted
    return result


def test_physical_receipt_is_strict_canonical_and_rejects_paths_and_bools() -> None:
    receipt = ImmutablePhysicalObjectReceiptV1(
        store_id="local:test",
        object_id=f"plans/{'a' * 64}/chunks/{'b' * 64}.bin",
        version_token="local-v1-1-2-3-4-180",
        size_bytes=7,
        sha256="c" * 64,
    )
    assert (
        ImmutablePhysicalObjectReceiptV1.from_json(receipt.canonical_bytes()) == receipt
    )
    forged = receipt.as_dict()
    forged["unknown"] = 1
    with pytest.raises(ProtocolError, match="unknown"):
        ImmutablePhysicalObjectReceiptV1.from_dict(forged)
    forged = receipt.as_dict()
    forged["size_bytes"] = True
    with pytest.raises(ProtocolError, match="booleans"):
        ImmutablePhysicalObjectReceiptV1.from_dict(forged)
    forged = receipt.as_dict()
    forged["object_id"] = "../escape"
    with pytest.raises(PrefixCheckpointError, match="unsafe"):
        ImmutablePhysicalObjectReceiptV1.from_dict(forged)


def test_local_store_create_reopen_mutation_and_path_attacks(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    root.mkdir()
    store = LocalImmutableByteStore(root, io_chunk_bytes=3)
    payload = bytearray(b"abcdefghijk")
    object_id = "plans/test/chunks/value.bin"
    digest = hashlib.sha256(payload).hexdigest()
    receipt = store.create_or_verify(
        object_id,
        memoryview(payload),
        expected_sha256=digest,
        expected_size_bytes=len(payload),
    )
    payload[:] = b"x" * len(payload)
    assert store.read_exact(receipt, offset=2, length=5, maximum_bytes=5) == b"cdefg"
    store.close()

    reopened = LocalImmutableByteStore(root, io_chunk_bytes=2)
    assert reopened.verify(receipt) == receipt
    assert (
        reopened.create_or_verify(
            object_id,
            b"abcdefghijk",
            expected_sha256=digest,
            expected_size_bytes=11,
        )
        == receipt
    )
    with pytest.raises(ImmutableObjectConflict, match="divergent"):
        reopened.create_or_verify(
            object_id,
            b"different!!",
            expected_sha256=hashlib.sha256(b"different!!").hexdigest(),
            expected_size_bytes=11,
        )
    with pytest.raises(PrefixCheckpointError, match="unsafe"):
        reopened.create_or_verify(
            "../escape",
            b"a",
            expected_sha256=hashlib.sha256(b"a").hexdigest(),
            expected_size_bytes=1,
        )

    path = root / object_id
    before = path.read_bytes()
    path.write_bytes(before)
    os.chmod(path, 0o600)
    with pytest.raises(PrefixCheckpointError, match="no longer matches"):
        reopened.verify(receipt)
    reopened.close()

    symlink_root = tmp_path / "symlink-root"
    symlink_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(PrefixCheckpointError, match="non-symlink"):
        LocalImmutableByteStore(symlink_root)


def test_local_store_identity_is_nonportable_and_oversize_probe_prehash_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = LocalImmutableByteStore(first_root, io_chunk_bytes=4)
    second = LocalImmutableByteStore(second_root, io_chunk_bytes=4)
    assert first.distributed_identity_qualified is False
    assert second.distributed_identity_qualified is False
    assert first.store_id != second.store_id
    assert first.stream_buffer_bytes == 4
    payload = b"oversized"
    first.create_or_verify(
        "plans/p/chunks/large.bin",
        payload,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
    )
    small_id = "plans/p/chunks/grow.bin"
    first.create_or_verify(
        small_id,
        b"ok",
        expected_sha256=hashlib.sha256(b"ok").hexdigest(),
        expected_size_bytes=2,
    )
    original_size = first._exact_path_size

    def admit_then_grow(object_id: str):
        result = original_size(object_id)
        if object_id == small_id:
            (first_root / small_id).write_bytes(b"grew-beyond-bound")
        return result

    monkeypatch.setattr(first, "_exact_path_size", admit_then_grow)

    def forbidden_pread(*_args, **_kwargs):
        raise AssertionError("oversized probe hashed payload bytes")

    monkeypatch.setattr(os, "pread", forbidden_pread)
    with pytest.raises(PrefixCheckpointError, match="exceeds bounded probe"):
        first.probe_bounded("plans/p/chunks/large.bin", maximum_bytes=2)
    with pytest.raises(PrefixCheckpointError, match="before hashing"):
        first.probe_bounded(small_id, maximum_bytes=2)
    first.close()
    second.close()


def test_sink_stream_buffer_admission_is_exact_and_rechecked(
    wired: _Fixture, tmp_path: Path
) -> None:
    plan = _plan(wired)
    accepted_root = tmp_path / "accepted"
    rejected_root = tmp_path / "rejected"
    accepted_root.mkdir()
    rejected_root.mkdir()
    accepted = LocalImmutableByteStore(accepted_root, io_chunk_bytes=32)
    sink = ImmutablePrefixChunkSinkV1(accepted, plan)
    rejected = LocalImmutableByteStore(rejected_root, io_chunk_bytes=33)
    with pytest.raises(PrefixCheckpointError, match="stream buffer exceeds"):
        ImmutablePrefixChunkSinkV1(rejected, plan)
    accepted._io_chunk_bytes = 33
    with pytest.raises(PrefixCheckpointError, match="admission changed"):
        sink.probe_committed(plan, plan.chunks[0])
    accepted.close()
    rejected.close()


def test_local_store_rejects_symlink_hardlink_and_root_substitution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / "plans" / "x" / "chunks").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    (root / "plans" / "x" / "chunks" / "evil.bin").symlink_to(outside)
    store = LocalImmutableByteStore(root)
    with pytest.raises(PrefixCheckpointError, match="safely open"):
        store.create_or_verify(
            "plans/x/chunks/evil.bin",
            b"outside",
            expected_sha256=hashlib.sha256(b"outside").hexdigest(),
            expected_size_bytes=7,
        )
    receipt = store.create_or_verify(
        "plans/x/chunks/good.bin",
        b"good",
        expected_sha256=hashlib.sha256(b"good").hexdigest(),
        expected_size_bytes=4,
    )
    os.link(root / receipt.object_id, tmp_path / "second-link")
    with pytest.raises(PrefixCheckpointError, match="single-link"):
        store.verify(receipt)
    os.unlink(tmp_path / "second-link")
    moved = tmp_path / "moved"
    root.rename(moved)
    root.mkdir()
    with pytest.raises(PrefixCheckpointError, match="substituted"):
        store.verify(receipt)
    store.close()


def test_measure_rejects_exact_path_replacement_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store"
    root.mkdir()
    store = LocalImmutableByteStore(root, io_chunk_bytes=2)
    object_id = "plans/p/chunks/race.bin"
    raw = b"abcdefgh"
    receipt = store.create_or_verify(
        object_id,
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_size_bytes=len(raw),
    )
    path = root / object_id
    old_path = path.with_name("old.bin")
    entered = threading.Event()
    replaced = threading.Event()
    original_pread = os.pread

    def pausing_pread(fd: int, length: int, offset: int) -> bytes:
        result = original_pread(fd, length, offset)
        if not entered.is_set():
            entered.set()
            assert replaced.wait(timeout=5)
        return result

    def replace_path() -> None:
        assert entered.wait(timeout=5)
        path.rename(old_path)
        path.write_bytes(raw)
        os.chmod(path, 0o600)
        replaced.set()

    monkeypatch.setattr(os, "pread", pausing_pread)
    worker = threading.Thread(target=replace_path)
    worker.start()
    with pytest.raises(PrefixCheckpointError, match="mutated|substituted"):
        store.verify(receipt)
    worker.join(timeout=5)
    assert not worker.is_alive()
    store.close()


def test_deterministic_staging_recovers_exact_crash_windows(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    store = LocalImmutableByteStore(root, io_chunk_bytes=2)
    payload = b"durable"
    digest = hashlib.sha256(payload).hexdigest()
    parent = root / "plans" / "p" / "chunks"
    parent.mkdir(parents=True)

    for leaf, state in (
        ("during-write.bin", "partial"),
        ("before-link.bin", "staged"),
        ("after-link.bin", "linked"),
    ):
        staging = parent / f".{leaf}.stage-{digest}"
        staging.write_bytes(payload[:2] if state == "partial" else payload)
        os.chmod(staging, 0o600)
        final = parent / leaf
        if state == "linked":
            os.link(staging, final)
            assert staging.stat().st_nlink == 2
        store.close()
        store = LocalImmutableByteStore(root, io_chunk_bytes=2)
        receipt = store.create_or_verify(
            f"plans/p/chunks/{leaf}",
            payload,
            expected_sha256=digest,
            expected_size_bytes=len(payload),
        )
        assert final.read_bytes() == payload
        assert final.stat().st_nlink == 1
        assert not staging.exists()
        assert store.verify(receipt) == receipt
    store.close()


def test_local_store_concurrent_identical_converges_and_divergent_fails(
    tmp_path: Path,
) -> None:
    root = tmp_path / "objects"
    root.mkdir()
    store = LocalImmutableByteStore(root)
    barrier = threading.Barrier(2)
    receipts: list[ImmutablePhysicalObjectReceiptV1] = []
    errors: list[BaseException] = []

    def identical() -> None:
        barrier.wait()
        try:
            receipts.append(
                store.create_or_verify(
                    "plans/p/chunks/same.bin",
                    b"same",
                    expected_sha256=hashlib.sha256(b"same").hexdigest(),
                    expected_size_bytes=4,
                )
            )
        except BaseException as error:  # pragma: no cover - assertion capture
            errors.append(error)

    threads = [threading.Thread(target=identical) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(receipts) == 2 and receipts[0] == receipts[1]

    barrier = threading.Barrier(2)
    results: list[object] = []

    def divergent(raw: bytes) -> None:
        barrier.wait()
        try:
            results.append(
                store.create_or_verify(
                    "plans/p/chunks/race.bin",
                    raw,
                    expected_sha256=hashlib.sha256(raw).hexdigest(),
                    expected_size_bytes=len(raw),
                )
            )
        except BaseException as error:
            results.append(error)

    threads = [
        threading.Thread(target=divergent, args=(b"first",)),
        threading.Thread(target=divergent, args=(b"other",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert (
        sum(isinstance(item, ImmutablePhysicalObjectReceiptV1) for item in results) == 1
    )
    assert sum(isinstance(item, ImmutableObjectConflict) for item in results) == 1
    store.close()


def test_true_conversion_crash_after_complete_shard_resume_manifest_and_restore(
    wired: _Fixture, tmp_path: Path
) -> None:
    plan = _plan(wired)
    object_root = tmp_path / "checkpoint"
    object_root.mkdir()
    store = LocalImmutableByteStore(object_root, io_chunk_bytes=17)
    sink = ImmutablePrefixChunkSinkV1(store, plan)
    source = LocalVerifiedSafetensorSource(
        wired.root,
        wired.selection,
        max_blob_bytes=1024**2,
        max_region_bytes=1024,
        hash_chunk_bytes=37,
        io_chunk_bytes=19,
    )
    first_shard_chunks = len(plan.shard_work[0].chunks)
    crashing = _CrashAfterCommitted(sink, first_shard_chunks)
    with pytest.raises(RuntimeError, match="injected crash"):
        execute_streaming_conversion(plan, source, crashing)
    source.close()

    source = LocalVerifiedSafetensorSource(
        wired.root,
        wired.selection,
        max_blob_bytes=1024**2,
        max_region_bytes=1024,
        hash_chunk_bytes=31,
        io_chunk_bytes=23,
    )
    resumed = execute_streaming_conversion(plan, source, sink)
    assert resumed.reused_chunks == first_shard_chunks
    assert plan.shard_work[0].source_descriptor.shard not in {
        item.shard for item in resumed.source_receipts
    }
    manifest = finalize_prefix_checkpoint(plan, resumed, sink)
    assert manifest.storage_complete is True
    assert manifest.production_sealable is False
    assert manifest.unplanned_object_absence_asserted is False
    assert not manifest.missing_chunk_ids
    assert all(
        binding.source_lineage
        for destination in manifest.destinations
        for binding in destination.chunks
    )

    # A clean invocation over the same immutable physical objects changes only
    # per-attempt accounting, which is deliberately not part of storage identity.
    repeated = execute_streaming_conversion(plan, source, sink)
    assert repeated.reused_chunks == len(plan.chunks)
    assert not repeated.source_receipts
    repeated_manifest = finalize_prefix_checkpoint(plan, repeated, sink)
    assert repeated_manifest.canonical_bytes() == manifest.canonical_bytes()
    source.close()

    parsed = PrefixCheckpointManifestV1.from_json(manifest.canonical_bytes())
    assert parsed == manifest
    reader = PrefixCheckpointReaderV1(store, plan, parsed, maximum_read_bytes=4096)
    with pytest.raises(PrefixCheckpointError, match="complete source observation"):
        reader.verify_complete_observation_evidence()
    expected = _expected_destinations(wired)
    saw_bf16 = saw_f32 = False
    for destination in plan.destinations:
        actual = reader.read_flat_region(
            destination.destination_key, 0, int(np.prod(destination.shape))
        )
        reference = expected[destination.destination_key]
        assert actual.flags.owndata and actual.flags.c_contiguous
        assert actual.dtype == reference.dtype
        assert actual.tobytes() == reference.tobytes()
        saw_bf16 |= actual.dtype == BF16
        saw_f32 |= actual.dtype == F32
    assert saw_bf16 and saw_f32

    # Fresh store instance and fresh interpreter restore the same exact bytes.
    store.close()
    fresh = LocalImmutableByteStore(object_root, io_chunk_bytes=13)
    fresh_reader = PrefixCheckpointReaderV1.open(fresh, plan, maximum_read_bytes=4096)
    key = plan.destinations[0].destination_key
    expected_raw = expected[key].tobytes()
    assert (
        fresh_reader.read_flat_region(key, 0, expected[key].size).tobytes()
        == expected_raw
    )
    plan_pickle = tmp_path / "plan.pkl"
    plan_pickle.write_bytes(pickle.dumps(plan))
    code = """
import hashlib, pickle, sys
from moe_sae.r1.prefix_checkpoint import LocalImmutableByteStore, PrefixCheckpointReaderV1
plan = pickle.loads(open(sys.argv[1], 'rb').read())
store = LocalImmutableByteStore(sys.argv[2], io_chunk_bytes=11)
reader = PrefixCheckpointReaderV1.open(store, plan, maximum_read_bytes=4096)
key = plan.destinations[0].destination_key
value = reader.read_flat_region(key, 0, int(__import__('numpy').prod(plan.destinations[0].shape)))
print(hashlib.sha256(value.tobytes()).hexdigest())
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(plan_pickle), str(object_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == hashlib.sha256(expected_raw).hexdigest()
    fresh.close()


def test_orphan_content_is_adopted_only_after_recompute_and_exact_rehash(
    wired: _Fixture, tmp_path: Path
) -> None:
    plan = _plan(wired)
    root = tmp_path / "checkpoint"
    root.mkdir()
    store = LocalImmutableByteStore(root, io_chunk_bytes=16)
    faulting_sink = ImmutablePrefixChunkSinkV1(
        _FailFirstBindingPublication(store), plan
    )
    source = LocalVerifiedSafetensorSource(
        wired.root,
        wired.selection,
        max_blob_bytes=1024**2,
        max_region_bytes=1024,
    )
    with pytest.raises(RuntimeError, match="before binding"):
        execute_streaming_conversion(plan, source, faulting_sink)
    source.close()

    sink = ImmutablePrefixChunkSinkV1(store, plan)
    first = plan.chunks[0]
    assert sink.probe_committed(plan, first) is None
    source = LocalVerifiedSafetensorSource(
        wired.root,
        wired.selection,
        max_blob_bytes=1024**2,
        max_region_bytes=1024,
    )
    summary = execute_streaming_conversion(plan, source, sink)
    manifest = finalize_prefix_checkpoint(plan, summary, sink)
    assert manifest.storage_complete is True
    assert sink.probe_committed(plan, first) == summary.receipts[0]
    source.close()
    store.close()


def _completed_checkpoint(wired: _Fixture, tmp_path: Path):
    plan = _plan(wired)
    root = tmp_path / "checkpoint"
    root.mkdir()
    store = LocalImmutableByteStore(root, io_chunk_bytes=16)
    sink = ImmutablePrefixChunkSinkV1(store, plan)
    source = LocalVerifiedSafetensorSource(
        wired.root,
        wired.selection,
        max_blob_bytes=1024**2,
        max_region_bytes=1024,
    )
    summary = execute_streaming_conversion(plan, source, sink)
    manifest = finalize_prefix_checkpoint(plan, summary, sink)
    source.close()
    return plan, root, store, manifest


def _full_and_reused_summaries(wired: _Fixture, root: Path):
    plan = _plan(wired)
    root.mkdir()
    store = LocalImmutableByteStore(root, io_chunk_bytes=16)
    sink = ImmutablePrefixChunkSinkV1(store, plan)
    source = LocalVerifiedSafetensorSource(
        wired.root,
        wired.selection,
        max_blob_bytes=1024**2,
        max_region_bytes=1024,
    )
    full = execute_streaming_conversion(plan, source, sink)
    reused = execute_streaming_conversion(plan, source, sink)
    source.close()
    assert full.written_chunks == len(plan.chunks)
    assert reused.reused_chunks == len(plan.chunks)
    return plan, store, sink, full, reused


def test_storage_manifest_and_evidence_converge_in_both_finalization_orders(
    wired: _Fixture, tmp_path: Path
) -> None:
    plan, first_store, first_sink, full, reused = _full_and_reused_summaries(
        wired, tmp_path / "reused-first"
    )
    reused_first = finalize_prefix_checkpoint(plan, reused, first_sink)
    first_reader = PrefixCheckpointReaderV1.open(
        first_store, plan, maximum_read_bytes=4096
    )
    with pytest.raises(PrefixCheckpointError, match="complete source observation"):
        first_reader.verify_complete_observation_evidence()
    full_second = finalize_prefix_checkpoint(plan, full, first_sink)
    assert full_second.canonical_bytes() == reused_first.canonical_bytes()
    first_snapshot = first_reader.verify_complete_observation_evidence()
    assert type(first_snapshot) is ObservationSnapshotV1

    second_plan, second_store, second_sink, second_full, second_reused = (
        _full_and_reused_summaries(wired, tmp_path / "full-first")
    )
    full_first = finalize_prefix_checkpoint(second_plan, second_full, second_sink)
    second_reader = PrefixCheckpointReaderV1.open(
        second_store, second_plan, maximum_read_bytes=4096
    )
    second_snapshot = second_reader.verify_complete_observation_evidence()
    reused_second = finalize_prefix_checkpoint(second_plan, second_reused, second_sink)
    assert reused_second.canonical_bytes() == full_first.canonical_bytes()
    assert second_reader.verify_complete_observation_evidence() == second_snapshot
    assert first_snapshot.canonical_bytes() == second_snapshot.canonical_bytes()
    first_store.close()
    second_store.close()


def test_oversized_manifest_is_rejected_before_publication(
    wired: _Fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, store, sink, full, _reused = _full_and_reused_summaries(
        wired, tmp_path / "oversized-manifest"
    )
    monkeypatch.setattr(prefix_checkpoint_module, "_MAX_MANIFEST_BYTES", 1)
    with pytest.raises(PrefixCheckpointError, match="reader admission"):
        finalize_prefix_checkpoint(plan, full, sink)
    assert (
        store.probe_bounded(f"plans/{plan.plan_id}/manifest.json", maximum_bytes=1024)
        is None
    )
    store.close()


def test_reader_rejects_forged_manifest_foreign_plan_and_bounds(
    wired: _Fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _root, store, manifest = _completed_checkpoint(wired, tmp_path)
    raw = manifest.as_dict()
    raw["total_chunks"] += 1
    with pytest.raises(ProtocolError):
        PrefixCheckpointManifestV1.from_dict(raw)
    raw = manifest.as_dict()
    raw["unknown"] = True
    with pytest.raises(ProtocolError, match="unknown"):
        PrefixCheckpointManifestV1.from_dict(raw)
    raw = manifest.as_dict()
    raw["storage_complete"] = 1
    with pytest.raises(ProtocolError, match="boolean"):
        PrefixCheckpointManifestV1.from_dict(raw)

    raw = manifest.as_dict()
    raw["max_estimated_resident_bytes"] += 1
    raw["manifest_id"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "manifest_id"}
    )
    self_hashed_peak_forgery = PrefixCheckpointManifestV1.from_dict(raw)
    with pytest.raises(PrefixCheckpointError, match="identity conflicts"):
        self_hashed_peak_forgery.validate_for_plan(
            plan, sink_id=manifest.sink_id, store_id=manifest.store_id
        )

    raw = manifest.as_dict()
    raw["unmet_production_gaps"] = raw["unmet_production_gaps"][:-1]
    raw["manifest_id"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "manifest_id"}
    )
    with pytest.raises(ProtocolError, match="exact protocol set"):
        PrefixCheckpointManifestV1.from_dict(raw)

    raw = manifest.as_dict()
    raw["store_identity_scope"] = "distributed-qualified-v1"
    raw["manifest_id"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "manifest_id"}
    )
    with pytest.raises(ProtocolError, match="local-only"):
        PrefixCheckpointManifestV1.from_dict(raw)

    foreign = _plan(wired, max_resident_bytes=2048)
    with pytest.raises(PrefixCheckpointError, match="conflicts"):
        PrefixCheckpointReaderV1(store, foreign, manifest)
    reader = PrefixCheckpointReaderV1(store, plan, manifest, maximum_read_bytes=4)
    snapshot = reader.verify_complete_observation_evidence()
    assert snapshot.complete is True
    assert manifest.store_identity_scope == "local-path-diagnostic-only-v1"
    destination = plan.destinations[0]
    with pytest.raises(PrefixCheckpointError, match="exceeds"):
        reader.read_flat_region(destination.destination_key, 0, 3)
    with pytest.raises(PrefixCheckpointError, match="integers"):
        reader.read_flat_region(destination.destination_key, True, 2)

    # Exact-path restore has no dependency on directory enumeration.
    monkeypatch.setattr(
        os, "listdir", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("listed"))
    )
    monkeypatch.setattr(
        os,
        "scandir",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("scanned")),
    )
    bounded = PrefixCheckpointReaderV1(store, plan, manifest, maximum_read_bytes=4)
    assert bounded.verify_complete_observation_evidence() == snapshot
    assert bounded.read_flat_region(destination.destination_key, 0, 1).size == 1
    store.close()


def test_corruption_forged_binding_and_physical_receipt_fail_closed(
    wired: _Fixture, tmp_path: Path
) -> None:
    plan, root, store, manifest = _completed_checkpoint(wired, tmp_path)
    binding = manifest.destinations[0].chunks[0]
    forged_physical = replace(
        binding.physical_receipt,
        version_token="local-v1-1-2-3-4-180",
    )
    forged_binding = replace(binding, physical_receipt=forged_physical)
    forged_destination = replace(
        manifest.destinations[0],
        chunks=(forged_binding, *manifest.destinations[0].chunks[1:]),
    )
    forged_manifest = replace(
        manifest,
        destinations=(forged_destination, *manifest.destinations[1:]),
    )
    with pytest.raises(PrefixCheckpointError, match="persisted manifest"):
        PrefixCheckpointReaderV1(store, plan, forged_manifest)

    content = root / binding.physical_receipt.object_id
    raw = bytearray(content.read_bytes())
    raw[0] ^= 0x01
    content.write_bytes(raw)
    os.chmod(content, 0o600)
    with pytest.raises(PrefixCheckpointError, match="no longer matches"):
        PrefixCheckpointReaderV1(store, plan, manifest)
    store.close()


def test_probe_rejects_forged_binding_sidecar(wired: _Fixture, tmp_path: Path) -> None:
    plan, root, store, manifest = _completed_checkpoint(wired, tmp_path)
    binding = manifest.destinations[0].chunks[0]
    binding_path = (
        root / "plans" / plan.plan_id / "receipts" / f"{binding.chunk_id}.json"
    )
    forged = binding.as_dict()
    forged["chunk_id"] = "f" * 64
    binding_path.write_bytes(
        json.dumps(forged, sort_keys=True, separators=(",", ":")).encode()
    )
    os.chmod(binding_path, 0o600)
    sink = ImmutablePrefixChunkSinkV1(store, plan)
    chunk = next(item for item in plan.chunks if item.chunk_id == binding.chunk_id)
    with pytest.raises(PrefixCheckpointError, match="malformed"):
        sink.probe_committed(plan, chunk)
    store.close()
