from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import struct
from typing import Any
import weakref

import ml_dtypes
import numpy as np
import pytest

from moe_sae.r1.checkpoint_conversion import (
    CheckpointConversionError,
    ConversionMemoryLimitsV1,
    OutputChunkReceiptV1,
    ProductionSealUnavailable,
    SourceAdapterIdentityV1,
    TensorReadRegionV1,
    VerifiedSourceShardReceiptV1,
    attest_r1_runtime_lock,
    build_pinned_prefix_conversion_plan,
    build_synthetic_prefix_conversion_plan,
    execute_streaming_conversion,
    numeric_codec_identity,
    numeric_codec_identity_sha256,
    seal_production_conversion,
)
from moe_sae.r1.checkpoint_inventory import (
    BF16_DTYPE,
    F32_DTYPE,
    FP8_DTYPE,
    DeepSeekR1CheckpointGeometryV1,
    build_synthetic_checkpoint_inventory_selection,
    decode_safetensor_header_bytes,
    decode_strict_json_bytes_with_receipt,
    expected_full_source_specs,
)


BF16 = np.dtype(ml_dtypes.bfloat16)
FP8 = np.dtype(ml_dtypes.float8_e4m3fn)
F32 = np.dtype("<f4")
SYNTHETIC_SOURCE_IDENTITY = SourceAdapterIdentityV1(
    adapter_id="synthetic-verified-tensor-source-v1",
    implementation_sha256="a" * 64,
)


@pytest.fixture(scope="module")
def tiny_geometry() -> DeepSeekR1CheckpointGeometryV1:
    return DeepSeekR1CheckpointGeometryV1(
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


def _source_shard(
    key: str,
    geometry: DeepSeekR1CheckpointGeometryV1,
    relocated_scale: str | None,
) -> str:
    if key == relocated_scale:
        return "model-00003-of-000003.safetensors"
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


def _wire_inventory(
    geometry: DeepSeekR1CheckpointGeometryV1,
    *,
    relocated_scale: str | None = None,
):
    by_shard: dict[str, list[tuple[str, str, tuple[int, ...]]]] = defaultdict(list)
    for spec in expected_full_source_specs(geometry):
        by_shard[_source_shard(spec.key, geometry, relocated_scale)].append(
            (spec.key, spec.dtype, spec.shape)
        )

    dtype_bytes = {FP8_DTYPE: 1, BF16_DTYPE: 2, F32_DTYPE: 4}
    headers: dict[str, dict[str, object]] = {}
    weight_map: dict[str, str] = {}
    source_numel = 0
    descriptors: dict[str, dict[str, object]] = {}
    for shard in sorted(by_shard):
        header: dict[str, object] = {"__metadata__": {"format": "pt"}}
        cursor = 0
        for key, dtype, shape in sorted(by_shard[shard]):
            numel = int(np.prod(shape))
            size = numel * dtype_bytes[dtype]
            header[key] = {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [cursor, cursor + size],
            }
            cursor += size
            source_numel += numel
            weight_map[key] = shard
        headers[shard] = header
        encoded = json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        raw_header = encoded + b" " * ((-len(encoded)) % 8)
        blob = struct.pack("<Q", len(raw_header)) + raw_header + bytes(cursor)
        decoded, descriptor = decode_safetensor_header_bytes(
            raw_header,
            shard=shard,
            expected_lfs_sha256=hashlib.sha256(blob).hexdigest(),
            expected_blob_size_bytes=len(blob),
        )
        assert decoded == header
        descriptors[shard] = descriptor

    index: dict[str, object] = {
        "metadata": {"total_size": 2 * source_numel},
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
    return selection


def _fixture_arrays(selection) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for entry in selection.selected.entries:
        offset = sum(entry.key.encode("utf-8")) % 7
        values = np.arange(entry.numel, dtype=np.int64).reshape(entry.shape)
        if entry.dtype == FP8_DTYPE:
            arrays[entry.key] = ((values + offset) % 9 - 4).astype(FP8)
        elif entry.dtype == BF16_DTYPE:
            arrays[entry.key] = ((values + offset) / 8 - 2).astype(BF16)
        elif entry.key.endswith("_scale_inv"):
            arrays[entry.key] = (
                0.75 + ((values + offset) % 7).astype(np.float32) / 16
            ).astype(F32)
        else:
            arrays[entry.key] = (
                (values.astype(np.float32) - entry.numel / 2) / 7
            ).astype(F32)
    dense = "model.layers.0.mlp.gate_proj.weight"
    arrays[dense][0, 0] = ml_dtypes.float8_e4m3fn(448)
    arrays[dense + "_scale_inv"][0, 0] = np.float32(1.003)
    return arrays


@dataclass
class _Reader:
    descriptor: Any
    arrays: dict[str, np.ndarray]
    owner: "_Source"

    @property
    def receipt(self) -> VerifiedSourceShardReceiptV1:
        sha256 = self.descriptor.expected_lfs_sha256
        size = self.descriptor.expected_blob_size_bytes
        if self.owner.bad_receipt_shard == self.descriptor.shard:
            sha256 = "f" * 64 if sha256 != "f" * 64 else "e" * 64
        return VerifiedSourceShardReceiptV1(
            shard=self.descriptor.shard,
            observed_sha256=sha256,
            observed_size_bytes=size,
            source_adapter_identity=self.owner.adapter_identity,
        )

    def read_region(self, region: TensorReadRegionV1) -> np.ndarray:
        if (
            self.owner.prior_payload_ref is not None
            and self.owner.prior_payload_ref() is not None
        ):
            raise AssertionError("executor retained the preceding chunk payload")
        if region.entry.shard != self.descriptor.shard:
            raise AssertionError("reader was asked for a tensor in another shard")
        self.owner.read_count += 1
        source = self.arrays[region.entry.key]
        slices = tuple(
            slice(start, stop)
            for start, stop in zip(region.starts, region.stops, strict=True)
        )
        result = np.array(source[slices], copy=True, order="C")
        if self.owner.bad_dtype_key == region.entry.key:
            result = result.astype(np.float64)
        self.owner.max_returned_bytes = max(
            self.owner.max_returned_bytes, result.nbytes
        )
        return result


class _ReaderContext(AbstractContextManager[_Reader]):
    def __init__(self, reader: _Reader, owner: "_Source") -> None:
        self.reader = reader
        self.owner = owner

    def __enter__(self) -> _Reader:
        if self.owner.open_readers:
            raise AssertionError("executor nested verified source shards")
        self.owner.open_readers += 1
        self.owner.opened_shards.append(self.reader.descriptor.shard)
        return self.reader

    def __exit__(self, *_: object) -> None:
        self.owner.open_readers -= 1


class _Source:
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        *,
        adapter_identity: SourceAdapterIdentityV1 = SYNTHETIC_SOURCE_IDENTITY,
    ) -> None:
        self.arrays = arrays
        self.read_count = 0
        self.max_returned_bytes = 0
        self.open_readers = 0
        self.opened_shards: list[str] = []
        self.bad_receipt_shard: str | None = None
        self.bad_dtype_key: str | None = None
        self.prior_payload_ref: weakref.ReferenceType[Any] | None = None
        self._adapter_identity = adapter_identity

    @property
    def adapter_identity(self) -> SourceAdapterIdentityV1:
        return self._adapter_identity

    def open_verified_shard(self, descriptor):
        return _ReaderContext(_Reader(descriptor, self.arrays, self), self)


class _Sink:
    def __init__(
        self,
        *,
        crash_after: int | None = None,
        lifetime_source: _Source | None = None,
    ) -> None:
        self._sink_id = "synthetic-create-only-sink-v1"
        self.crash_after = crash_after
        self.receipts: dict[str, OutputChunkReceiptV1] = {}
        self.objects: dict[str, bytes] = {}
        self.commits = 0
        self.probes = 0
        self.corrupt_probe_for: str | None = None
        self.lifetime_source = lifetime_source

    @property
    def sink_id(self) -> str:
        return self._sink_id

    def probe_committed(self, plan, chunk):
        self.probes += 1
        receipt = self.receipts.get(chunk.chunk_id)
        if receipt is None:
            return None
        payload = self.objects[chunk.chunk_id]
        if hashlib.sha256(payload).hexdigest() != receipt.output_sha256:
            raise CheckpointConversionError("synthetic durable object hash mismatch")
        if self.corrupt_probe_for == chunk.chunk_id:
            return replace(
                receipt, destination_flat_stop=receipt.destination_flat_stop + 1
            )
        return receipt

    def commit_chunk(self, plan, chunk, payload, *, output_sha256):
        if self.crash_after is not None and self.commits >= self.crash_after:
            raise RuntimeError("synthetic process crash")
        raw = bytes(payload)
        if self.lifetime_source is not None:
            self.lifetime_source.prior_payload_ref = weakref.ref(payload.obj)
        if hashlib.sha256(raw).hexdigest() != output_sha256:
            raise AssertionError("executor supplied the wrong output digest")
        if chunk.chunk_id in self.objects:
            raise AssertionError("executor attempted to overwrite an immutable chunk")
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
        self.commits += 1
        return receipt


@pytest.fixture
def limits() -> ConversionMemoryLimitsV1:
    return ConversionMemoryLimitsV1(
        max_source_blob_bytes=1024**2,
        max_resident_bytes=512,
        max_output_chunk_bytes=64,
        sink_staging_buffer_bytes=16,
    )


def _destination_array(plan, sink: _Sink, key: str) -> np.ndarray:
    destination = next(
        item for item in plan.destinations if item.destination_key == key
    )
    dtype = BF16 if destination.dtype == "bfloat16" else F32
    raw = bytearray(destination.payload_bytes)
    itemsize = dtype.itemsize
    for start, stop, chunk_id in destination.chunks:
        payload = sink.objects[chunk_id]
        raw[start * itemsize : stop * itemsize] = payload
    return np.frombuffer(bytes(raw), dtype=dtype).reshape(destination.shape)


def _mapping(selection, source_key: str):
    return next(item for item in selection.mappings if item.source_key == source_key)


def _plan(selection, limits: ConversionMemoryLimitsV1):
    return build_synthetic_prefix_conversion_plan(
        selection,
        limits,
        source_adapter_identity=SYNTHETIC_SOURCE_IDENTITY,
    )


def _scalar_fp8_reference(
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


def test_plan_is_exact_deterministic_bounded_and_explicitly_nonsealable(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
    limits: ConversionMemoryLimitsV1,
) -> None:
    selection = _wire_inventory(tiny_geometry)
    first = _plan(selection, limits)
    second = _plan(selection, limits)

    assert first.plan_id == second.plan_id
    assert [item.chunk_id for item in first.chunks] == [
        item.chunk_id for item in second.chunks
    ]
    assert first.production_input_qualified is False
    assert first.conversion_sealable is False
    assert sum(item.payload_bytes for item in first.destinations) == (
        selection.converted_payload_bytes
    )
    assert max(item.output_bytes for item in first.chunks) <= 64
    assert max(item.estimated_peak_resident_bytes for item in first.chunks) <= 512
    assert first.numeric_codec_sha256 == numeric_codec_identity_sha256()
    assert {
        "numpy_core_extension_sha256",
        "numpy_record_sha256",
        "ml_dtypes_extension_sha256",
        "ml_dtypes_record_sha256",
    } <= set(numeric_codec_identity())
    assert all(
        destination.chunks[0][0] == 0
        and destination.chunks[-1][1] == int(np.prod(destination.shape))
        and all(
            left[1] == right[0]
            for left, right in zip(
                destination.chunks, destination.chunks[1:], strict=False
            )
        )
        for destination in first.destinations
    )
    target = tiny_geometry.target_layer_zero_indexed
    assert not any(
        f"model.layers.{target}.mlp.shared_experts" in item.source_entry.key
        for item in first.chunks
    )
    with pytest.raises(CheckpointConversionError, match="pinned R1-0528"):
        build_pinned_prefix_conversion_plan(
            selection,
            limits,
            source_adapter_identity=SYNTHETIC_SOURCE_IDENTITY,
            runtime_project_root=Path(__file__).parents[2],
        )
    with pytest.raises(CheckpointConversionError, match="authoritative inventory"):
        replace(first, selection_sha256="f" * 64)

    with pytest.raises(CheckpointConversionError, match="audited R1 revision"):
        attest_r1_runtime_lock(Path(__file__).parents[2])

    swappable_index, swappable = next(
        (index, destination)
        for index, destination in enumerate(first.destinations)
        if len(destination.chunks) >= 2
        and destination.chunks[0][1] - destination.chunks[0][0]
        == destination.chunks[1][1] - destination.chunks[1][0]
    )
    forged_chunks = list(swappable.chunks)
    first_interval, second_interval = forged_chunks[:2]
    forged_chunks[0] = (*first_interval[:2], second_interval[2])
    forged_chunks[1] = (*second_interval[:2], first_interval[2])
    forged_destination = replace(swappable, chunks=tuple(forged_chunks))
    forged_destinations = list(first.destinations)
    forged_destinations[swappable_index] = forged_destination
    with pytest.raises(CheckpointConversionError, match="exactly match"):
        replace(first, destinations=tuple(forged_destinations))


def test_runtime_lock_attestation_accepts_only_the_exact_r1_pin(tmp_path: Path) -> None:
    revision = "c03f8bd307d18049fa814fc8c104bd9ec8c292aa"
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            (
                "[tool.uv.sources]",
                'maxtext = { git = "https://github.com/AI-Hypercomputer/maxtext.git", '
                f'rev = "{revision}" }}',
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text(
        "\n".join(
            (
                "version = 1",
                "revision = 1",
                'requires-python = ">=3.12"',
                "[[package]]",
                'name = "maxtext"',
                'version = "0.0.0"',
                'source = { git = "https://github.com/AI-Hypercomputer/maxtext.git'
                f'?rev={revision}#{revision}" }}',
            )
        ),
        encoding="utf-8",
    )

    identity = attest_r1_runtime_lock(tmp_path)
    assert identity.maxtext_revision == revision
    assert len(identity.pyproject_sha256) == 64
    assert len(identity.uv_lock_sha256) == 64

    stale = (
        (tmp_path / "pyproject.toml")
        .read_text(encoding="utf-8")
        .replace(revision, "48c7da3e28e8f57fe36ddf09d00eec53072a2ea0")
    )
    (tmp_path / "pyproject.toml").write_text(stale, encoding="utf-8")
    with pytest.raises(CheckpointConversionError, match="audited R1 revision"):
        attest_r1_runtime_lock(tmp_path)


def test_fp8_multiply_order_layouts_and_passthrough_bits(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
    limits: ConversionMemoryLimitsV1,
) -> None:
    selection = _wire_inventory(tiny_geometry)
    arrays = _fixture_arrays(selection)
    plan = _plan(selection, limits)
    source = _Source(arrays)
    sink = _Sink()

    summary = execute_streaming_conversion(plan, source, sink)
    assert summary.written_chunks == len(plan.chunks)
    assert summary.reused_chunks == 0
    assert summary.converted_payload_bytes == selection.converted_payload_bytes
    assert summary.conversion_sealable is False
    assert summary.max_estimated_resident_bytes <= limits.max_resident_bytes
    assert summary.source_receipts
    assert source.max_returned_bytes <= limits.max_resident_bytes

    dense_key = "model.layers.0.mlp.gate_proj.weight"
    dense_mapping = _mapping(selection, dense_key)
    dense_actual = _destination_array(plan, sink, dense_mapping.destination_key)
    dense_expected = _scalar_fp8_reference(
        arrays[dense_key],
        arrays[dense_key + "_scale_inv"],
        tiny_geometry.fp8_block_size,
    ).reshape(dense_mapping.destination_shape)
    np.testing.assert_array_equal(
        dense_actual.view(np.uint16), dense_expected.view(np.uint16)
    )
    early_rounded = ml_dtypes.bfloat16(
        np.float32(ml_dtypes.bfloat16(np.float32(arrays[dense_key][0, 0])))
        * np.float32(ml_dtypes.bfloat16(arrays[dense_key + "_scale_inv"][0, 0]))
    )
    assert dense_actual[0, 0] != early_rounded

    target = tiny_geometry.target_layer_zero_indexed
    q_b_key = f"model.layers.{target}.self_attn.q_b_proj.weight"
    q_b_mapping = _mapping(selection, q_b_key)
    q_b_actual = _destination_array(plan, sink, q_b_mapping.destination_key)
    q_b_expected = _scalar_fp8_reference(
        arrays[q_b_key], arrays[q_b_key + "_scale_inv"], 4
    ).reshape(q_b_mapping.destination_shape)
    np.testing.assert_array_equal(
        q_b_actual.view(np.uint16), q_b_expected.view(np.uint16)
    )

    expert_mappings = [
        _mapping(
            selection,
            f"model.layers.{target}.mlp.experts.{expert}.gate_proj.weight",
        )
        for expert in range(tiny_geometry.routed_experts)
    ]
    expert_actual = _destination_array(plan, sink, expert_mappings[0].destination_key)
    expert_expected = np.stack(
        [
            _scalar_fp8_reference(
                arrays[mapping.source_key],
                arrays[mapping.source_key + "_scale_inv"],
                4,
            )
            for mapping in expert_mappings
        ]
    )
    np.testing.assert_array_equal(
        expert_actual.view(np.uint16), expert_expected.view(np.uint16)
    )

    embedding_key = "model.embed_tokens.weight"
    embedding_mapping = _mapping(selection, embedding_key)
    embedding_actual = _destination_array(plan, sink, embedding_mapping.destination_key)
    np.testing.assert_array_equal(
        embedding_actual.view(np.uint16), arrays[embedding_key].view(np.uint16)
    )

    router_key = f"model.layers.{target}.mlp.gate.weight"
    router_mapping = _mapping(selection, router_key)
    router_actual = _destination_array(plan, sink, router_mapping.destination_key)
    np.testing.assert_array_equal(
        router_actual.view(np.uint16), arrays[router_key].T.view(np.uint16)
    )
    bias_key = f"model.layers.{target}.mlp.gate.e_score_correction_bias"
    bias_mapping = _mapping(selection, bias_key)
    bias_actual = _destination_array(plan, sink, bias_mapping.destination_key)
    np.testing.assert_array_equal(
        bias_actual.view(np.uint32), arrays[bias_key].view(np.uint32)
    )

    with pytest.raises(ProductionSealUnavailable, match="production sealing"):
        seal_production_conversion(summary)


def test_crash_resume_is_exact_and_fully_resumed_run_reads_nothing(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
    limits: ConversionMemoryLimitsV1,
) -> None:
    selection = _wire_inventory(tiny_geometry)
    arrays = _fixture_arrays(selection)
    plan = _plan(selection, limits)

    interrupted_source = _Source(arrays)
    interrupted_sink = _Sink(crash_after=5)
    with pytest.raises(RuntimeError, match="synthetic process crash"):
        execute_streaming_conversion(plan, interrupted_source, interrupted_sink)
    assert interrupted_sink.commits == 5
    assert len(interrupted_sink.objects) == 5

    interrupted_sink.crash_after = None
    reads_before_resume = interrupted_source.read_count
    resumed = execute_streaming_conversion(plan, interrupted_source, interrupted_sink)
    assert resumed.reused_chunks == 5
    assert resumed.written_chunks == len(plan.chunks) - 5
    assert interrupted_source.read_count > reads_before_resume

    uninterrupted_sink = _Sink()
    execute_streaming_conversion(plan, _Source(arrays), uninterrupted_sink)
    assert interrupted_sink.objects == uninterrupted_sink.objects

    reads_before_noop = interrupted_source.read_count
    commits_before_noop = interrupted_sink.commits
    no_op = execute_streaming_conversion(plan, interrupted_source, interrupted_sink)
    assert no_op.reused_chunks == len(plan.chunks)
    assert no_op.written_chunks == 0
    assert no_op.source_receipts == ()
    assert interrupted_source.read_count == reads_before_noop
    assert interrupted_sink.commits == commits_before_noop


def test_previous_payload_is_released_before_the_next_source_read(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
    limits: ConversionMemoryLimitsV1,
) -> None:
    selection = _wire_inventory(tiny_geometry)
    plan = _plan(selection, limits)
    source = _Source(_fixture_arrays(selection))
    sink = _Sink(lifetime_source=source)

    execute_streaming_conversion(plan, source, sink)
    assert source.prior_payload_ref is not None
    assert source.prior_payload_ref() is None


def test_source_adapter_identity_is_bound_before_resume_or_io(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
    limits: ConversionMemoryLimitsV1,
) -> None:
    selection = _wire_inventory(tiny_geometry)
    plan = _plan(selection, limits)
    source = _Source(
        _fixture_arrays(selection),
        adapter_identity=SourceAdapterIdentityV1(
            adapter_id="different-source-adapter-v1",
            implementation_sha256="b" * 64,
        ),
    )
    sink = _Sink()

    with pytest.raises(CheckpointConversionError, match="does not match"):
        execute_streaming_conversion(plan, source, sink)
    assert source.read_count == 0
    assert sink.probes == 0


def test_conflicting_resume_receipt_and_corrupt_object_fail_closed(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
    limits: ConversionMemoryLimitsV1,
) -> None:
    selection = _wire_inventory(tiny_geometry)
    arrays = _fixture_arrays(selection)
    plan = _plan(selection, limits)
    sink = _Sink()
    execute_streaming_conversion(plan, _Source(arrays), sink)

    first = plan.chunks[0]
    sink.corrupt_probe_for = first.chunk_id
    with pytest.raises(
        CheckpointConversionError, match="conflicts with chunk identity"
    ):
        execute_streaming_conversion(plan, _Source(arrays), sink)

    sink.corrupt_probe_for = None
    original = sink.objects[first.chunk_id]
    sink.objects[first.chunk_id] = bytes([original[0] ^ 1]) + original[1:]
    with pytest.raises(CheckpointConversionError, match="durable object hash mismatch"):
        execute_streaming_conversion(plan, _Source(arrays), sink)


def test_bad_source_receipt_fails_before_tensor_read_or_commit(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
    limits: ConversionMemoryLimitsV1,
) -> None:
    selection = _wire_inventory(tiny_geometry)
    plan = _plan(selection, limits)
    source = _Source(_fixture_arrays(selection))
    source.bad_receipt_shard = plan.shard_work[0].source_descriptor.shard
    sink = _Sink()

    with pytest.raises(CheckpointConversionError, match="source blob receipt mismatch"):
        execute_streaming_conversion(plan, source, sink)
    assert source.read_count == 0
    assert sink.commits == 0


def test_malformed_source_array_is_rejected_before_sink_commit(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
    limits: ConversionMemoryLimitsV1,
) -> None:
    selection = _wire_inventory(tiny_geometry)
    plan = _plan(selection, limits)
    source = _Source(_fixture_arrays(selection))
    first = plan.chunks[0]
    source.bad_dtype_key = first.source_entry.key
    sink = _Sink()

    with pytest.raises(CheckpointConversionError, match="source region dtype mismatch"):
        execute_streaming_conversion(plan, source, sink)
    assert sink.commits == 0


def test_cross_shard_scale_is_verified_sequentially_and_accounted(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
    limits: ConversionMemoryLimitsV1,
) -> None:
    relocated = "model.layers.0.self_attn.q_a_proj.weight_scale_inv"
    selection = _wire_inventory(tiny_geometry, relocated_scale=relocated)
    plan = _plan(selection, limits)
    source = _Source(_fixture_arrays(selection))
    sink = _Sink()
    summary = execute_streaming_conversion(plan, source, sink)

    assert summary.written_chunks == len(plan.chunks)
    assert source.open_readers == 0
    assert "model-00003-of-000003.safetensors" in source.opened_shards
    assert any(item.cross_shard_scale_cache_bytes > 0 for item in plan.shard_work)
    assert summary.max_estimated_resident_bytes <= limits.max_resident_bytes


def test_impossible_memory_limit_fails_during_planning_before_any_io(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    selection = _wire_inventory(tiny_geometry)
    impossible = ConversionMemoryLimitsV1(
        max_source_blob_bytes=1024**2,
        max_resident_bytes=32,
        max_output_chunk_bytes=16,
        sink_staging_buffer_bytes=0,
    )
    with pytest.raises(CheckpointConversionError, match="cannot hold"):
        _plan(selection, impossible)


def test_chunk_contract_change_changes_plan_and_chunk_identities(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
    limits: ConversionMemoryLimitsV1,
) -> None:
    selection = _wire_inventory(tiny_geometry)
    first = _plan(selection, limits)
    second = _plan(selection, replace(limits, max_output_chunk_bytes=96))
    assert first.plan_id != second.plan_id
    assert {item.chunk_id for item in first.chunks} != {
        item.chunk_id for item in second.chunks
    }


def test_peak_estimate_charges_the_complete_declared_sink_staging_bound(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
    limits: ConversionMemoryLimitsV1,
) -> None:
    selection = _wire_inventory(tiny_geometry)
    first = _plan(selection, limits)
    second = _plan(
        selection,
        replace(limits, sink_staging_buffer_bytes=limits.sink_staging_buffer_bytes + 1),
    )

    def identity(chunk):
        return (
            chunk.mapping.source_key,
            chunk.source_start,
            chunk.source_stop,
            chunk.destination_flat_start,
            chunk.destination_flat_stop,
        )

    first_by_identity = {identity(chunk): chunk for chunk in first.chunks}
    second_by_identity = {identity(chunk): chunk for chunk in second.chunks}
    assert set(first_by_identity) == set(second_by_identity)
    assert all(
        second_by_identity[key].estimated_peak_resident_bytes
        == first_by_identity[key].estimated_peak_resident_bytes + 1
        for key in first_by_identity
    )


def test_nonfinite_scale_fails_without_creating_that_chunk(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
    limits: ConversionMemoryLimitsV1,
) -> None:
    selection = _wire_inventory(tiny_geometry)
    arrays = _fixture_arrays(selection)
    scale_key = "model.layers.0.mlp.gate_proj.weight_scale_inv"
    arrays[scale_key][0, 0] = np.float32(np.inf)
    plan = _plan(selection, limits)
    sink = _Sink()

    with pytest.raises(CheckpointConversionError, match="must be finite"):
        execute_streaming_conversion(plan, _Source(arrays), sink)
    affected = {
        item.chunk_id
        for item in plan.chunks
        if item.scale_entry is not None and item.scale_entry.key == scale_key
    }
    assert affected.isdisjoint(sink.objects)


def test_unverified_scale_positivity_is_not_invented_as_a_gate(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
    limits: ConversionMemoryLimitsV1,
) -> None:
    selection = _wire_inventory(tiny_geometry)
    arrays = _fixture_arrays(selection)
    key = "model.layers.0.mlp.gate_proj.weight"
    arrays[key + "_scale_inv"][0, 0] = np.float32(0)
    arrays[key + "_scale_inv"][0, 1] = np.float32(-0.5)
    plan = _plan(selection, limits)
    sink = _Sink()
    execute_streaming_conversion(plan, _Source(arrays), sink)

    mapping = _mapping(selection, key)
    actual = _destination_array(plan, sink, mapping.destination_key)
    expected = _scalar_fp8_reference(arrays[key], arrays[key + "_scale_inv"], 4)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


def test_synthetic_selection_mapping_tampering_is_rejected_upstream(
    tiny_geometry: DeepSeekR1CheckpointGeometryV1,
) -> None:
    selection = _wire_inventory(tiny_geometry)
    forged = replace(selection.mappings[0], destination_key="forged-destination")
    with pytest.raises(ValueError, match="canonical HF-to-MaxText"):
        replace(selection, mappings=(forged, *selection.mappings[1:]))
