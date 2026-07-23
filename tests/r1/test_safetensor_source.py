from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import struct

import ml_dtypes
import numpy as np
import pytest

from moe_sae.r1.checkpoint_conversion import (
    CheckpointConversionError,
    ConversionMemoryLimitsV1,
    OutputChunkReceiptV1,
    TensorReadRegionV1,
    build_synthetic_prefix_conversion_plan,
    execute_streaming_conversion,
)
from moe_sae.r1.checkpoint_inventory import (
    BF16_DTYPE,
    FP8_DTYPE,
    CheckpointInventorySelectionV1,
    DeepSeekR1CheckpointGeometryV1,
    SourceTensorEntryV1,
    build_synthetic_checkpoint_inventory_selection,
    decode_safetensor_header_bytes,
    decode_strict_json_bytes_with_receipt,
    expected_full_source_specs,
)
from moe_sae.r1.safetensor_source import (
    LOCAL_SOURCE_ADAPTER_ID,
    LOCAL_SOURCE_TRUST_BOUNDARY,
    LocalVerifiedSafetensorSource,
    local_safetensor_source_identity,
)
import moe_sae.r1.safetensor_source as safetensor_source_module


BF16 = np.dtype(ml_dtypes.bfloat16)
FP8 = np.dtype(ml_dtypes.float8_e4m3fn)
F32 = np.dtype("<f4")


@dataclass(frozen=True)
class _Fixture:
    root: Path
    selection: CheckpointInventorySelectionV1
    arrays: dict[str, np.ndarray]


class _RehashingMemorySink:
    def __init__(self) -> None:
        self._sink_id = "local-source-integration-memory-sink-v1"
        self.objects: dict[str, bytes] = {}
        self.receipts: dict[str, OutputChunkReceiptV1] = {}

    @property
    def sink_id(self) -> str:
        return self._sink_id

    def probe_committed(self, plan, chunk):
        receipt = self.receipts.get(chunk.chunk_id)
        if receipt is None:
            return None
        payload = self.objects[chunk.chunk_id]
        if hashlib.sha256(payload).hexdigest() != receipt.output_sha256:
            raise CheckpointConversionError("in-memory object failed resume re-hash")
        return receipt

    def commit_chunk(self, plan, chunk, payload, *, output_sha256):
        if chunk.chunk_id in self.objects:
            raise CheckpointConversionError("in-memory sink is create-only")
        raw = bytes(payload)
        if hashlib.sha256(raw).hexdigest() != output_sha256:
            raise CheckpointConversionError("in-memory sink received a bad digest")
        receipt = OutputChunkReceiptV1(
            plan_id=plan.plan_id,
            chunk_id=chunk.chunk_id,
            sink_id=self.sink_id,
            destination_key=chunk.mapping.destination_key,
            destination_flat_start=chunk.destination_flat_start,
            destination_flat_stop=chunk.destination_flat_stop,
            dtype=chunk.mapping.destination_dtype,
            output_sha256=output_sha256,
            output_size_bytes=len(raw),
            immutable_object_id=f"memory://{chunk.chunk_id}",
        )
        self.objects[chunk.chunk_id] = raw
        self.receipts[chunk.chunk_id] = receipt
        return receipt


def _shard_for(key: str, geometry: DeepSeekR1CheckpointGeometryV1) -> str:
    logical_key = key.removesuffix("_scale_inv")
    if logical_key == "model.embed_tokens.weight":
        number = 1
    elif logical_key.startswith(f"model.layers.{geometry.target_layer_zero_indexed}."):
        number = 2
    elif logical_key.startswith("model.layers."):
        layer = int(logical_key.split(".")[2])
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
    return ((values.astype(np.float32) + salt) / 13 - 2).astype(F32)


def _build_fixture(tmp_path: Path) -> _Fixture:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
        ).encode("utf-8")
        raw_header = encoded + b" " * ((-len(encoded)) % 8)
        blob = struct.pack("<Q", len(raw_header)) + raw_header + b"".join(payloads)
        (tmp_path / shard).write_bytes(blob)
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
        index,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    index_sha256 = hashlib.sha256(raw_index).hexdigest()
    _, index_receipt = decode_strict_json_bytes_with_receipt(
        raw_index,
        expected_sha256=index_sha256,
        maximum_bytes=len(raw_index),
    )
    selection = build_synthetic_checkpoint_inventory_selection(
        index,
        headers,
        verified_index_sha256=index_sha256,
        index_decode_receipt=index_receipt,
        shard_descriptors=descriptors,
        geometry=geometry,
    )
    return _Fixture(tmp_path, selection, arrays)


@pytest.fixture
def wired(tmp_path: Path) -> _Fixture:
    return _build_fixture(tmp_path)


def _entry(wired: _Fixture, key: str) -> SourceTensorEntryV1:
    return next(item for item in wired.selection.full.entries if item.key == key)


def _descriptor(wired: _Fixture, shard: str):
    return next(
        item
        for item in wired.selection.selected.shard_descriptors
        if item.shard == shard
    )


def _region(
    entry: SourceTensorEntryV1,
    starts: tuple[int, ...],
    stops: tuple[int, ...],
) -> TensorReadRegionV1:
    return TensorReadRegionV1(entry=entry, starts=starts, stops=stops)


def _assembled_destination(plan, sink: _RehashingMemorySink, key: str) -> np.ndarray:
    destination = next(
        item for item in plan.destinations if item.destination_key == key
    )
    dtype = BF16 if destination.dtype == "bfloat16" else F32
    raw = bytearray(destination.payload_bytes)
    for start, stop, chunk_id in destination.chunks:
        payload = sink.objects[chunk_id]
        raw[start * dtype.itemsize : stop * dtype.itemsize] = payload
    return np.frombuffer(bytes(raw), dtype=dtype).reshape(destination.shape)


def _scalar_fp8_oracle(
    weight: np.ndarray,
    scale: np.ndarray,
    block_size: int,
) -> np.ndarray:
    result = np.empty((weight.shape[1], weight.shape[0]), dtype=BF16)
    for row in range(weight.shape[0]):
        for column in range(weight.shape[1]):
            product = np.float32(weight[row, column]) * np.float32(
                scale[row // block_size, column // block_size]
            )
            result[column, row] = ml_dtypes.bfloat16(product)
    return result


def test_reads_real_bf16_fp8_and_f32_regions_with_bound_receipts(
    wired: _Fixture,
) -> None:
    geometry = wired.selection.full.geometry
    bf16_key = "model.embed_tokens.weight"
    fp8_key = "model.layers.0.mlp.gate_proj.weight"
    f32_key = (
        f"model.layers.{geometry.target_layer_zero_indexed}.mlp.gate."
        "e_score_correction_bias"
    )
    requests = (
        (bf16_key, (2, 1), (9, 5)),
        (fp8_key, (1, 2), (7, 5)),
        (f32_key, (0,), (2,)),
    )

    identity = local_safetensor_source_identity()
    implementation = Path(safetensor_source_module.__file__).read_bytes()
    assert identity.adapter_id == LOCAL_SOURCE_ADAPTER_ID
    assert identity.implementation_sha256 == hashlib.sha256(implementation).hexdigest()

    with LocalVerifiedSafetensorSource(wired.root, wired.selection) as source:
        assert source.adapter_identity == identity
        assert source.trust_boundary == LOCAL_SOURCE_TRUST_BOUNDARY
        assert source.production_sealable is False
        for key, starts, stops in requests:
            entry = _entry(wired, key)
            descriptor = _descriptor(wired, entry.shard)
            with source.open_verified_shard(descriptor) as reader:
                receipt = reader.receipt
                receipt.validate_for(descriptor, source.adapter_identity)
                actual = reader.read_region(_region(entry, starts, stops))
            slices = tuple(slice(start, stop) for start, stop in zip(starts, stops))
            expected = wired.arrays[key][slices]
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == expected.dtype
            assert actual.flags.c_contiguous
            assert actual.flags.owndata


def test_real_local_source_executes_the_streaming_conversion_plan_end_to_end(
    wired: _Fixture,
) -> None:
    limits = ConversionMemoryLimitsV1(
        max_source_blob_bytes=1024**2,
        max_resident_bytes=512,
        max_output_chunk_bytes=64,
        sink_staging_buffer_bytes=16,
    )
    source = LocalVerifiedSafetensorSource(
        wired.root,
        wired.selection,
        max_blob_bytes=limits.max_source_blob_bytes,
        max_region_bytes=limits.max_resident_bytes,
    )
    plan = build_synthetic_prefix_conversion_plan(
        wired.selection,
        limits,
        source_adapter_identity=source.adapter_identity,
    )
    sink = _RehashingMemorySink()
    with source:
        summary = execute_streaming_conversion(plan, source, sink)
        resumed = execute_streaming_conversion(plan, source, sink)

    chunk_ids = {chunk.chunk_id for chunk in plan.chunks}
    assert summary.written_chunks == len(plan.chunks)
    assert summary.reused_chunks == 0
    assert summary.converted_payload_bytes == wired.selection.converted_payload_bytes
    assert {receipt.chunk_id for receipt in summary.receipts} == chunk_ids
    assert set(sink.objects) == chunk_ids
    assert set(sink.receipts) == chunk_ids
    assert sum(len(payload) for payload in sink.objects.values()) == (
        plan.converted_payload_bytes
    )
    assert {receipt.shard for receipt in summary.source_receipts} == set(
        wired.selection.selected.required_shards
    )
    assert all(
        destination.chunks[0][0] == 0
        and destination.chunks[-1][1] == int(np.prod(destination.shape))
        and sum(stop - start for start, stop, _ in destination.chunks)
        == int(np.prod(destination.shape))
        for destination in plan.destinations
    )
    assert resumed.reused_chunks == len(plan.chunks)
    assert resumed.written_chunks == 0
    assert resumed.source_receipts == ()

    embedding_key = "model.embed_tokens.weight"
    embedding_mapping = next(
        item for item in wired.selection.mappings if item.source_key == embedding_key
    )
    embedding = _assembled_destination(plan, sink, embedding_mapping.destination_key)
    np.testing.assert_array_equal(
        embedding.view(np.uint16), wired.arrays[embedding_key].view(np.uint16)
    )

    fp8_key = "model.layers.0.mlp.gate_proj.weight"
    fp8_mapping = next(
        item for item in wired.selection.mappings if item.source_key == fp8_key
    )
    fp8_actual = _assembled_destination(plan, sink, fp8_mapping.destination_key)
    fp8_expected = _scalar_fp8_oracle(
        wired.arrays[fp8_key],
        wired.arrays[fp8_key + "_scale_inv"],
        wired.selection.full.geometry.fp8_block_size,
    ).reshape(fp8_mapping.destination_shape)
    np.testing.assert_array_equal(
        fp8_actual.view(np.uint16), fp8_expected.view(np.uint16)
    )


def test_region_reads_are_chunk_bounded_and_do_not_read_a_whole_tensor(
    wired: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "model.embed_tokens.weight"
    entry = _entry(wired, key)
    descriptor = _descriptor(wired, entry.shard)
    pread_requests: list[int] = []
    preadv_requests: list[int] = []
    real_pread = os.pread
    real_preadv = os.preadv

    def measured_pread(fd: int, size: int, offset: int) -> bytes:
        pread_requests.append(size)
        return real_pread(fd, size, offset)

    def measured_preadv(fd: int, buffers, offset: int) -> int:
        preadv_requests.extend(len(buffer) for buffer in buffers)
        return real_preadv(fd, buffers, offset)

    monkeypatch.setattr(safetensor_source_module.os, "pread", measured_pread)
    monkeypatch.setattr(safetensor_source_module.os, "preadv", measured_preadv)
    with LocalVerifiedSafetensorSource(
        wired.root,
        wired.selection,
        hash_chunk_bytes=17,
        io_chunk_bytes=7,
        max_region_bytes=32,
    ) as source:
        with source.open_verified_shard(descriptor) as reader:
            requested = _region(entry, (1, 2), (3, 5))
            actual = reader.read_region(requested)
            assert actual.nbytes == requested.payload_bytes == 12
            # A non-contiguous column slice is intentionally read as one safe,
            # bounded segment per selected source row rather than with mmap.
            assert reader.payload_read_syscalls == 2
            assert source.payload_read_syscalls == 2
    assert pread_requests
    assert max(pread_requests) <= 17
    assert preadv_requests
    assert max(preadv_requests) <= 7
    assert entry.payload_bytes > actual.nbytes


@pytest.mark.parametrize("corruption", ["size", "payload", "header"])
def test_size_hash_and_header_corruption_fail_closed(
    tmp_path: Path,
    corruption: str,
) -> None:
    wired = _build_fixture(tmp_path)
    descriptor = wired.selection.selected.shard_descriptors[0]
    path = wired.root / descriptor.shard
    raw = path.read_bytes()
    if corruption == "size":
        path.write_bytes(raw[:-1])
        with pytest.raises(CheckpointConversionError, match="size mismatch"):
            LocalVerifiedSafetensorSource(wired.root, wired.selection)
        return
    if corruption == "payload":
        changed = bytearray(raw)
        changed[-1] ^= 1
    else:
        changed = bytearray(raw)
        changed[0] ^= 8
    path.write_bytes(changed)
    with LocalVerifiedSafetensorSource(wired.root, wired.selection) as source:
        with pytest.raises(CheckpointConversionError, match="SHA-256 mismatch"):
            with source.open_verified_shard(descriptor):
                pass


def test_header_must_match_the_complete_authenticated_inventory(
    wired: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = wired.selection.selected.shard_descriptors[0]
    real_decode = safetensor_source_module.decode_safetensor_header_bytes

    def mismatched_decode(*args, **kwargs):
        decoded, observed = real_decode(*args, **kwargs)
        changed = dict(decoded)
        changed["__metadata__"] = {"format": "not-pt"}
        return changed, observed

    monkeypatch.setattr(
        safetensor_source_module,
        "decode_safetensor_header_bytes",
        mismatched_decode,
    )
    with LocalVerifiedSafetensorSource(wired.root, wired.selection) as source:
        with pytest.raises(
            CheckpointConversionError, match="header/inventory mismatch"
        ):
            with source.open_verified_shard(descriptor):
                pass


def test_traversal_case_collision_symlink_and_hardlink_are_rejected(
    tmp_path: Path,
) -> None:
    traversal_fixture = _build_fixture(tmp_path / "traversal")
    source = LocalVerifiedSafetensorSource(
        traversal_fixture.root, traversal_fixture.selection
    )
    descriptor = traversal_fixture.selection.selected.shard_descriptors[0]
    object.__setattr__(descriptor, "shard", "../escape.safetensors")
    with pytest.raises(CheckpointConversionError, match="not authorized"):
        source.open_verified_shard(descriptor)
    source.close()

    collision_fixture = _build_fixture(tmp_path / "collision")
    first = collision_fixture.selection.selected.shard_descriptors[0]
    collision = replace(
        first,
        shard=first.shard.removesuffix(".safetensors").upper() + ".safetensors",
    )
    object.__setattr__(
        collision_fixture.selection.selected,
        "shard_descriptors",
        (first, collision),
    )
    with pytest.raises(CheckpointConversionError, match="collide case-insensitively"):
        LocalVerifiedSafetensorSource(
            collision_fixture.root, collision_fixture.selection
        )

    symlink_fixture = _build_fixture(tmp_path / "symlink")
    symlink_descriptor = symlink_fixture.selection.selected.shard_descriptors[0]
    symlink_path = symlink_fixture.root / symlink_descriptor.shard
    target = symlink_fixture.root / "target.bin"
    target.write_bytes(symlink_path.read_bytes())
    symlink_path.unlink()
    symlink_path.symlink_to(target.name)
    with pytest.raises(CheckpointConversionError, match="non-symlink regular file"):
        LocalVerifiedSafetensorSource(symlink_fixture.root, symlink_fixture.selection)

    hardlink_fixture = _build_fixture(tmp_path / "hardlink")
    descriptors = hardlink_fixture.selection.selected.shard_descriptors
    first_path = hardlink_fixture.root / descriptors[0].shard
    second_path = hardlink_fixture.root / descriptors[1].shard
    second_path.unlink()
    os.link(first_path, second_path)
    with pytest.raises(
        CheckpointConversionError, match="size mismatch|one regular file"
    ):
        LocalVerifiedSafetensorSource(hardlink_fixture.root, hardlink_fixture.selection)


@pytest.mark.parametrize("mutation", ["overwrite", "truncate"])
def test_mutation_or_truncation_after_verification_is_detected_before_read(
    tmp_path: Path,
    mutation: str,
) -> None:
    wired = _build_fixture(tmp_path)
    key = "model.embed_tokens.weight"
    entry = _entry(wired, key)
    descriptor = _descriptor(wired, entry.shard)
    path = wired.root / descriptor.shard
    with LocalVerifiedSafetensorSource(wired.root, wired.selection) as source:
        with source.open_verified_shard(descriptor) as reader:
            if mutation == "overwrite":
                with path.open("r+b") as handle:
                    handle.seek(-1, os.SEEK_END)
                    byte = handle.read(1)
                    handle.seek(-1, os.SEEK_END)
                    handle.write(bytes([byte[0] ^ 1]))
                    handle.flush()
                    os.fsync(handle.fileno())
            else:
                with path.open("r+b") as handle:
                    handle.truncate(path.stat().st_size - 1)
                    handle.flush()
                    os.fsync(handle.fileno())
            with pytest.raises(CheckpointConversionError, match="mutated"):
                reader.read_region(_region(entry, (0, 0), (1, 2)))


def test_cross_shard_forged_entry_limits_and_close_semantics(
    wired: _Fixture,
) -> None:
    first_descriptor, second_descriptor = wired.selection.selected.shard_descriptors[:2]
    first_entry = next(
        item
        for item in wired.selection.selected.entries
        if item.shard == first_descriptor.shard and len(item.shape) == 2
    )
    second_entry = next(
        item
        for item in wired.selection.selected.entries
        if item.shard == second_descriptor.shard
    )
    source = LocalVerifiedSafetensorSource(
        wired.root,
        wired.selection,
        max_region_bytes=8,
    )
    context = source.open_verified_shard(first_descriptor)
    with context as reader:
        with pytest.raises(CheckpointConversionError, match="different local shard"):
            reader.read_region(
                _region(
                    second_entry,
                    tuple(0 for _ in second_entry.shape),
                    tuple(1 for _ in second_entry.shape),
                )
            )
        forged_shape = (1, first_entry.numel)
        forged = replace(first_entry, shape=forged_shape)
        with pytest.raises(CheckpointConversionError, match="not authenticated"):
            reader.read_region(_region(forged, (0, 0), forged_shape))
        with pytest.raises(CheckpointConversionError, match="max_region_bytes"):
            reader.read_region(
                _region(
                    first_entry,
                    tuple(0 for _ in first_entry.shape),
                    first_entry.shape,
                )
            )
        with pytest.raises(CheckpointConversionError, match="only one active reader"):
            with source.open_verified_shard(second_descriptor):
                pass
        with pytest.raises(CheckpointConversionError, match="active reader"):
            source.close()

    with pytest.raises(CheckpointConversionError, match="closed"):
        reader.read_region(
            _region(
                first_entry,
                tuple(0 for _ in first_entry.shape),
                tuple(1 for _ in first_entry.shape),
            )
        )
    source.close()
    source.close()
    with pytest.raises(CheckpointConversionError, match="closed"):
        source.open_verified_shard(first_descriptor)


def test_source_and_reader_close_when_the_context_body_raises(
    wired: _Fixture,
) -> None:
    entry = _entry(wired, "model.embed_tokens.weight")
    descriptor = _descriptor(wired, entry.shard)
    source = LocalVerifiedSafetensorSource(wired.root, wired.selection)
    reader = None

    with pytest.raises(RuntimeError, match="intentional context failure"):
        with source:
            with source.open_verified_shard(descriptor) as active_reader:
                reader = active_reader
                raise RuntimeError("intentional context failure")

    assert reader is not None
    with pytest.raises(CheckpointConversionError, match="closed"):
        reader.read_region(_region(entry, (0, 0), (1, 1)))
    with pytest.raises(CheckpointConversionError, match="closed"):
        source.open_verified_shard(descriptor)
