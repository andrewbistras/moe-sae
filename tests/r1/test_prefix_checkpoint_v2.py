from __future__ import annotations

import base64
from collections import defaultdict
from dataclasses import FrozenInstanceError, dataclass, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

import google_crc32c
import ml_dtypes
import numpy as np
import pytest

from moe_sae.r1.checkpoint_conversion import (
    ConversionMemoryLimitsV1,
    OutputChunkReceiptV1,
    R1RuntimeLockIdentityV1,
    SourceAdapterIdentityV1,
    execute_streaming_conversion,
)
from moe_sae.r1.checkpoint_inventory import (
    BF16_DTYPE,
    F32_DTYPE,
    FP8_DTYPE,
    HF_REVISION,
    MAXTEXT_REVISION,
    DeepSeekR1CheckpointGeometryV1,
    SourceTensorEntryV1,
    expected_prefix_logical_specs,
)
from moe_sae.r1.gcs_immutable_byte_store import (
    GCSPhysicalObjectReceiptV1,
    content_addressed_object_id,
    content_addressed_record_object_id,
)
from moe_sae.r1.prefix_checkpoint import PlanSourceIdentityBindingV1
from moe_sae.r1.prefix_checkpoint_v2 import (
    MAX_BINDING_BYTES,
    MAX_DESTINATIONS,
    MAX_LOGICAL_MANIFEST_BYTES,
    MAX_REGIONAL_ENVELOPE_BYTES,
    PINNED_CONVERTED_PARAMETERS,
    PINNED_CONVERTED_PAYLOAD_BYTES,
    PINNED_DESTINATION_TENSORS,
    PINNED_SOURCE_ENTRIES,
    PINNED_SOURCE_LOGICAL_TENSORS,
    PINNED_SOURCE_SHARDS,
    PINNED_TARGET_LAYER_ONE_INDEXED,
    PINNED_TOTAL_CHUNKS,
    PRODUCTION_PREFIX_SINK_ID,
    LazyRegionalPrefixCheckpointReaderV1,
    LogicalPrefixCheckpointManifestV2,
    LogicalStoredDestinationV2,
    PrefixCheckpointV2Error,
    RegionalChunkReplicaV1,
    RegionalPrefixCheckpointEnvelopeV1,
    RegionalPrefixChunkBindingV1,
    build_logical_prefix_checkpoint_manifest_v2,
    build_regional_prefix_checkpoint_envelope_v1,
    regional_chunk_binding_object_id,
)
from moe_sae.r1.protocol import ProtocolError, canonical_json_bytes
import moe_sae.r1.prefix_checkpoint_v2 as prefix_v2
import moe_sae.r1.checkpoint_conversion as checkpoint_conversion
import moe_sae.r1.checkpoint_inventory as checkpoint_inventory
from tests.r1.test_checkpoint_conversion import (
    _Source,
    _fixture_arrays,
    _plan,
    _wire_inventory,
)


BF16 = np.dtype(ml_dtypes.bfloat16)
F32 = np.dtype("<f4")


class _ContentAddressedSink:
    def __init__(self) -> None:
        self.receipts: dict[str, OutputChunkReceiptV1] = {}
        self.objects: dict[str, bytes] = {}

    @property
    def sink_id(self) -> str:
        return PRODUCTION_PREFIX_SINK_ID

    def probe_committed(self, plan, chunk):
        receipt = self.receipts.get(chunk.chunk_id)
        if receipt is not None:
            assert hashlib.sha256(self.objects[chunk.chunk_id]).hexdigest() == (
                receipt.output_sha256
            )
        return receipt

    def commit_chunk(self, plan, chunk, payload, *, output_sha256):
        raw = bytes(payload)
        assert hashlib.sha256(raw).hexdigest() == output_sha256
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
            immutable_object_id=content_addressed_object_id(output_sha256),
        )
        self.receipts[chunk.chunk_id] = receipt
        self.objects[chunk.chunk_id] = raw
        return receipt


@dataclass(frozen=True)
class _Artifacts:
    plan: Any
    summary: Any
    manifest: LogicalPrefixCheckpointManifestV2
    sink: _ContentAddressedSink


@pytest.fixture(scope="module")
def artifacts() -> _Artifacts:
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
    selection = _wire_inventory(geometry)
    plan = _plan(
        selection,
        ConversionMemoryLimitsV1(
            max_source_blob_bytes=1024**2,
            max_resident_bytes=512,
            max_output_chunk_bytes=64,
            sink_staging_buffer_bytes=16,
        ),
    )
    source = _Source(_fixture_arrays(selection))
    sink = _ContentAddressedSink()
    summary = execute_streaming_conversion(plan, source, sink)
    manifest = build_logical_prefix_checkpoint_manifest_v2(plan, summary)
    return _Artifacts(plan=plan, summary=summary, manifest=manifest, sink=sink)


def _crc32c(raw: bytes) -> str:
    checksum = google_crc32c.Checksum(raw)
    return base64.b64encode(checksum.digest()).decode("ascii")


def _physical_receipt(
    raw: bytes,
    *,
    object_id: str,
    store_id: str,
    region: str,
    generation: int,
) -> GCSPhysicalObjectReceiptV1:
    digest = hashlib.sha256(raw).hexdigest()
    suffix = (
        "application/json"
        if object_id.endswith(".json")
        else "application/octet-stream"
    )
    return GCSPhysicalObjectReceiptV1(
        store_id=store_id,
        object_id=object_id,
        uri=f"gs://r1-test-bucket/prefix/{object_id}",
        region=region,
        generation=str(generation),
        metageneration="1",
        size_bytes=len(raw),
        crc32c=_crc32c(raw),
        sha256=digest,
        content_type=suffix,
    )


def _lineage(plan, chunk) -> tuple[PlanSourceIdentityBindingV1, ...]:
    result = [
        PlanSourceIdentityBindingV1(
            role="source",
            shard=chunk.source_descriptor.shard,
            sha256=chunk.source_descriptor.expected_lfs_sha256,
            size_bytes=chunk.source_descriptor.expected_blob_size_bytes,
            source_adapter_identity=plan.source_adapter_identity,
        )
    ]
    if (
        chunk.scale_descriptor is not None
        and chunk.scale_descriptor != chunk.source_descriptor
    ):
        result.append(
            PlanSourceIdentityBindingV1(
                role="scale",
                shard=chunk.scale_descriptor.shard,
                sha256=chunk.scale_descriptor.expected_lfs_sha256,
                size_bytes=chunk.scale_descriptor.expected_blob_size_bytes,
                source_adapter_identity=plan.source_adapter_identity,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class _RegionalArtifacts:
    envelope: RegionalPrefixCheckpointEnvelopeV1
    envelope_receipt: GCSPhysicalObjectReceiptV1
    manifest_receipt: GCSPhysicalObjectReceiptV1
    bindings: tuple[RegionalPrefixChunkBindingV1, ...]
    binding_receipts: tuple[GCSPhysicalObjectReceiptV1, ...]
    objects: dict[GCSPhysicalObjectReceiptV1, bytes]


def _regional_artifacts(
    artifacts: _Artifacts,
    *,
    region: str = "us-central1",
    store_id: str = "gcs-prefix-byte-store-v1:test",
    generation_base: int = 100,
) -> _RegionalArtifacts:
    objects: dict[GCSPhysicalObjectReceiptV1, bytes] = {}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in artifacts.plan.chunks}
    logical = tuple(
        receipt
        for destination in artifacts.manifest.destinations
        for receipt in destination.chunks
    )
    bindings: list[RegionalPrefixChunkBindingV1] = []
    binding_receipts: list[GCSPhysicalObjectReceiptV1] = []
    for index, output in enumerate(logical):
        payload = artifacts.sink.objects[output.chunk_id]
        payload_receipt = _physical_receipt(
            payload,
            object_id=output.immutable_object_id,
            store_id=store_id,
            region=region,
            generation=generation_base + index * 2,
        )
        objects[payload_receipt] = payload
        binding = RegionalPrefixChunkBindingV1(
            plan_id=artifacts.plan.plan_id,
            chunk_id=output.chunk_id,
            output_receipt=output,
            physical_receipt=payload_receipt,
            source_lineage=_lineage(artifacts.plan, chunks_by_id[output.chunk_id]),
        )
        binding_raw = binding.canonical_bytes()
        binding_receipt = _physical_receipt(
            binding_raw,
            object_id=regional_chunk_binding_object_id(
                artifacts.plan.plan_id, output.chunk_id
            ),
            store_id=store_id,
            region=region,
            generation=generation_base + index * 2 + 1,
        )
        objects[binding_receipt] = binding_raw
        bindings.append(binding)
        binding_receipts.append(binding_receipt)
    manifest_raw = artifacts.manifest.canonical_bytes()
    manifest_receipt = _physical_receipt(
        manifest_raw,
        object_id=content_addressed_record_object_id(
            hashlib.sha256(manifest_raw).hexdigest()
        ),
        store_id=store_id,
        region=region,
        generation=generation_base + len(logical) * 2,
    )
    objects[manifest_receipt] = manifest_raw
    envelope = build_regional_prefix_checkpoint_envelope_v1(
        artifacts.plan,
        artifacts.manifest,
        manifest_receipt,
        tuple(bindings),
        tuple(binding_receipts),
    )
    envelope_raw = envelope.canonical_bytes()
    envelope_receipt = _physical_receipt(
        envelope_raw,
        object_id=content_addressed_record_object_id(
            hashlib.sha256(envelope_raw).hexdigest()
        ),
        store_id=store_id,
        region=region,
        generation=generation_base + len(logical) * 2 + 1,
    )
    objects[envelope_receipt] = envelope_raw
    return _RegionalArtifacts(
        envelope=envelope,
        envelope_receipt=envelope_receipt,
        manifest_receipt=manifest_receipt,
        bindings=tuple(bindings),
        binding_receipts=tuple(binding_receipts),
        objects=objects,
    )


class _ExactStore:
    def __init__(
        self,
        regional: _RegionalArtifacts,
        *,
        qualified: bool = True,
    ) -> None:
        self.store_id = regional.envelope.store_id
        self.region = regional.envelope.region
        self.distributed_identity_qualified = qualified
        self.objects = dict(regional.objects)
        self.calls: list[tuple[GCSPhysicalObjectReceiptV1, int, int, int]] = []
        self.after_read = None

    def read_exact(self, receipt, *, offset, length, maximum_bytes):
        assert receipt.store_id == self.store_id
        assert receipt.region == self.region
        assert length <= maximum_bytes
        raw = self.objects.get(receipt)
        if raw is None:
            raise PrefixCheckpointV2Error("exact generation is absent")
        if (
            len(raw) != receipt.size_bytes
            or hashlib.sha256(raw).hexdigest() != receipt.sha256
            or _crc32c(raw) != receipt.crc32c
        ):
            raise PrefixCheckpointV2Error("exact generation is corrupt")
        if offset < 0 or length <= 0 or offset + length > len(raw):
            raise PrefixCheckpointV2Error("exact read range is invalid")
        self.calls.append((receipt, offset, length, maximum_bytes))
        result = raw[offset : offset + length]
        if self.after_read is not None:
            self.after_read(self, receipt, len(self.calls))
        return result


def test_logical_manifest_is_topology_neutral_exact_and_observation_independent(
    artifacts: _Artifacts,
) -> None:
    manifest = artifacts.manifest
    manifest.validate_for_plan(artifacts.plan)
    assert manifest.production_sealable is False
    assert manifest.pinned_production_profile is False
    assert "complete-source-observation-snapshot-evidence" in (
        manifest.external_qualification_required
    )
    assert LogicalPrefixCheckpointManifestV2.from_json(manifest.canonical_bytes()) == (
        manifest
    )

    forbidden = {
        "bucket",
        "prefix",
        "store_id",
        "uri",
        "region",
        "generation",
        "metageneration",
        "crc32c",
        "physical_receipt",
        "attempt",
        "host",
    }

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(manifest.as_dict())
    assert {
        receipt.sink_id
        for destination in manifest.destinations
        for receipt in destination.chunks
    } == {PRODUCTION_PREFIX_SINK_ID}
    assert all(
        receipt.immutable_object_id
        == content_addressed_object_id(receipt.output_sha256)
        for destination in manifest.destinations
        for receipt in destination.chunks
    )

    no_observations = replace(artifacts.summary, source_receipts=())
    rebuilt = build_logical_prefix_checkpoint_manifest_v2(
        artifacts.plan, no_observations
    )
    assert rebuilt.canonical_bytes() == manifest.canonical_bytes()


def test_tree_commitment_is_honestly_ordered_and_rejects_bad_coverage(
    artifacts: _Artifacts,
) -> None:
    destination = next(
        item for item in artifacts.manifest.destinations if len(item.chunks) >= 2
    )
    first = destination.chunks[0]
    changed_hash = "f" * 64 if first.output_sha256 != "f" * 64 else "e" * 64
    changed = replace(
        first,
        output_sha256=changed_hash,
        immutable_object_id=content_addressed_object_id(changed_hash),
    )
    changed_destination = LogicalStoredDestinationV2.build(
        destination_key=destination.destination_key,
        dtype=destination.dtype,
        shape=destination.shape,
        chunks=(changed, *destination.chunks[1:]),
    )
    assert changed_destination.logical_tensor_tree_sha256 != (
        destination.logical_tensor_tree_sha256
    )
    assert LogicalStoredDestinationV2.from_json(destination.canonical_bytes()) == (
        destination
    )

    with pytest.raises(ProtocolError, match="gap|overlap|duplicate|foreign"):
        replace(destination, chunks=tuple(reversed(destination.chunks)))
    with pytest.raises(ProtocolError, match="gap|overlap|duplicate|foreign"):
        replace(
            destination,
            chunks=(
                replace(
                    first,
                    destination_flat_stop=first.destination_flat_stop - 1,
                    output_size_bytes=first.output_size_bytes - 2,
                ),
                *destination.chunks[1:],
            ),
        )


def test_public_destination_builder_rejects_unbounded_inputs_before_hashing(
    artifacts: _Artifacts, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = artifacts.manifest.destinations[0].chunks[0]
    called = False

    def forbidden_tree(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("tree hashing must not run")

    monkeypatch.setattr(prefix_v2, "_logical_tensor_tree_sha256", forbidden_tree)
    with pytest.raises(ProtocolError, match="parameter bound"):
        LogicalStoredDestinationV2.build(
            destination_key="x",
            dtype="bfloat16",
            shape=(10**10_000,),
            chunks=(receipt,),
        )
    assert called is False


def test_wire_decode_is_bounded_canonical_and_normalizes_integer_errors(
    artifacts: _Artifacts,
) -> None:
    raw = artifacts.manifest.canonical_bytes()
    with pytest.raises(ProtocolError, match="canonical JSON"):
        LogicalPrefixCheckpointManifestV2.from_json(raw + b"\n")
    with pytest.raises(ProtocolError, match="strict JSON"):
        LogicalPrefixCheckpointManifestV2.from_json(b'{"schema":"x","schema":"x"}')
    huge_integer = (
        b'{"schema":"moe-sae.r1.logical-prefix-checkpoint-manifest.v2",'
        b'"source_shard_count":' + b"9" * 5_000 + b"}"
    )
    with pytest.raises(ProtocolError, match="strict JSON"):
        LogicalPrefixCheckpointManifestV2.from_json(huge_integer)
    with pytest.raises(ProtocolError, match="byte bound"):
        RegionalPrefixChunkBindingV1.from_json(b" " * (MAX_BINDING_BYTES + 1))

    decoded = json.loads(raw)
    decoded["production_sealable"] = True
    with pytest.raises(ProtocolError, match="evidence flags"):
        LogicalPrefixCheckpointManifestV2.from_json(canonical_json_bytes(decoded))
    decoded = json.loads(raw)
    decoded["unknown"] = 1
    with pytest.raises(ProtocolError, match="unknown fields"):
        LogicalPrefixCheckpointManifestV2.from_json(canonical_json_bytes(decoded))

    decoded = json.loads(raw)
    decoded["destinations"] = [{}] * (MAX_DESTINATIONS + 1)
    with pytest.raises(ProtocolError, match="bounded"):
        LogicalPrefixCheckpointManifestV2.from_dict(decoded)


def test_binding_lineage_is_plan_derived_not_observation_evidence(
    artifacts: _Artifacts,
) -> None:
    regional = _regional_artifacts(artifacts)
    chunks = {chunk.chunk_id: chunk for chunk in artifacts.plan.chunks}
    for binding in regional.bindings:
        binding.validate_for(artifacts.plan, chunks[binding.chunk_id])
        assert tuple(item.role for item in binding.source_lineage) in {
            ("source",),
            ("source", "scale"),
        }
        assert "observed" not in binding.canonical_json()
        assert RegionalPrefixChunkBindingV1.from_json(binding.canonical_bytes()) == (
            binding
        )

    first = regional.bindings[0]
    with pytest.raises(ProtocolError, match="roles are ambiguous"):
        replace(
            first,
            source_lineage=(first.source_lineage[0], first.source_lineage[0]),
        )


def test_regional_envelope_requires_durable_exact_binding_records(
    artifacts: _Artifacts,
) -> None:
    regional = _regional_artifacts(artifacts)
    regional.envelope.validate_for_manifest(artifacts.manifest)
    assert regional.envelope.production_sealable is False
    assert (
        RegionalPrefixCheckpointEnvelopeV1.from_json(
            regional.envelope.canonical_bytes()
        )
        == regional.envelope
    )
    assert (
        RegionalChunkReplicaV1.from_json(
            regional.envelope.chunk_replicas[0].canonical_bytes()
        )
        == regional.envelope.chunk_replicas[0]
    )

    with pytest.raises(PrefixCheckpointV2Error, match="cardinality"):
        build_regional_prefix_checkpoint_envelope_v1(
            artifacts.plan,
            artifacts.manifest,
            regional.manifest_receipt,
            regional.bindings,
            regional.binding_receipts[:-1],
        )
    bad_receipt = replace(
        regional.binding_receipts[0],
        generation="999",
        sha256="f" * 64,
    )
    with pytest.raises(PrefixCheckpointV2Error, match="exact durable receipt"):
        build_regional_prefix_checkpoint_envelope_v1(
            artifacts.plan,
            artifacts.manifest,
            regional.manifest_receipt,
            regional.bindings,
            (bad_receipt, *regional.binding_receipts[1:]),
        )

    reordered = replace(
        regional.envelope,
        chunk_replicas=tuple(reversed(regional.envelope.chunk_replicas)),
    )
    with pytest.raises(PrefixCheckpointV2Error, match="ordered logical content"):
        reordered.validate_for_manifest(artifacts.manifest)


def test_equal_size_swaps_foreign_stores_and_latest_paths_fail_closed(
    artifacts: _Artifacts,
) -> None:
    regional = _regional_artifacts(artifacts)
    by_size: defaultdict[int, list[RegionalPrefixChunkBindingV1]] = defaultdict(list)
    for binding in regional.bindings:
        by_size[binding.output_receipt.output_size_bytes].append(binding)
    first, second = next(items[:2] for items in by_size.values() if len(items) >= 2)
    with pytest.raises(ProtocolError, match="cross-bind exact bytes"):
        replace(first, physical_receipt=second.physical_receipt)

    payload = artifacts.sink.objects[first.chunk_id]
    wrong_path = _physical_receipt(
        payload,
        object_id=f"foreign/{first.output_receipt.output_sha256}.bin",
        store_id=regional.envelope.store_id,
        region=regional.envelope.region,
        generation=777,
    )
    with pytest.raises(ProtocolError, match="cross-bind exact bytes"):
        replace(first, physical_receipt=wrong_path)

    foreign_physical = replace(
        first.physical_receipt,
        store_id="gcs-prefix-byte-store-v1:foreign",
    )
    foreign_binding = replace(first, physical_receipt=foreign_physical)
    foreign_bindings = tuple(
        foreign_binding if item.chunk_id == first.chunk_id else item
        for item in regional.bindings
    )
    with pytest.raises(PrefixCheckpointV2Error, match="another store"):
        build_regional_prefix_checkpoint_envelope_v1(
            artifacts.plan,
            artifacts.manifest,
            regional.manifest_receipt,
            foreign_bindings,
            regional.binding_receipts,
        )

    with pytest.raises(PrefixCheckpointV2Error, match="duplicate"):
        build_regional_prefix_checkpoint_envelope_v1(
            artifacts.plan,
            artifacts.manifest,
            regional.manifest_receipt,
            (regional.bindings[0], regional.bindings[0], *regional.bindings[2:]),
            regional.binding_receipts,
        )

    manifest_raw = artifacts.manifest.canonical_bytes()
    latest_receipt = _physical_receipt(
        manifest_raw,
        object_id=f"plans/{artifacts.plan.plan_id}/latest.json",
        store_id=regional.envelope.store_id,
        region=regional.envelope.region,
        generation=778,
    )
    with pytest.raises(PrefixCheckpointV2Error, match="not exact"):
        build_regional_prefix_checkpoint_envelope_v1(
            artifacts.plan,
            artifacts.manifest,
            latest_receipt,
            regional.bindings,
            regional.binding_receipts,
        )

    with pytest.raises(ProtocolError, match="unique and sorted"):
        replace(
            artifacts.manifest,
            destinations=tuple(reversed(artifacts.manifest.destinations)),
        )


def test_region_copy_changes_envelope_but_not_logical_identity(
    artifacts: _Artifacts,
) -> None:
    first = _regional_artifacts(
        artifacts,
        region="us-central1",
        store_id="gcs-prefix-byte-store-v1:first",
    )
    second = _regional_artifacts(
        artifacts,
        region="us-east1",
        store_id="gcs-prefix-byte-store-v1:second",
        generation_base=10_000,
    )
    assert first.envelope.logical_manifest_sha256 == (
        second.envelope.logical_manifest_sha256
    )
    assert artifacts.manifest.digest == first.envelope.logical_manifest_sha256
    assert first.envelope.digest != second.envelope.digest


def test_synthetic_manifest_cannot_cross_the_pinned_production_gate(
    artifacts: _Artifacts,
) -> None:
    runtime = R1RuntimeLockIdentityV1(
        maxtext_revision=MAXTEXT_REVISION,
        pyproject_sha256="a" * 64,
        uv_lock_sha256="b" * 64,
    )
    with pytest.raises(ProtocolError, match="pinned R1 prefix invariants"):
        replace(
            artifacts.manifest,
            mode="pinned-r1-prefix",
            runtime_lock_identity=runtime,
        )


def test_lazy_open_is_exactly_two_control_reads_then_touched_bindings_and_payloads(
    artifacts: _Artifacts,
) -> None:
    regional = _regional_artifacts(artifacts)
    store = _ExactStore(regional)
    reader = LazyRegionalPrefixCheckpointReaderV1.open(
        store,
        regional.envelope_receipt,
        expected_logical_manifest_sha256=artifacts.manifest.digest,
    )
    assert len(store.calls) == 2
    assert [call[0] for call in store.calls] == [
        regional.envelope_receipt,
        regional.manifest_receipt,
    ]
    assert all(call[0].content_type == "application/json" for call in store.calls)
    assert not any(call[0].object_id.endswith(".bin") for call in store.calls)

    destination = next(
        item for item in artifacts.manifest.destinations if len(item.chunks) >= 2
    )
    boundary = destination.chunks[0].destination_flat_stop
    actual = reader.read_flat_region(
        destination.destination_key, boundary - 1, boundary + 1
    )
    dtype = BF16 if destination.dtype == "bfloat16" else F32
    complete = b"".join(
        artifacts.sink.objects[receipt.chunk_id] for receipt in destination.chunks
    )
    expected = np.frombuffer(complete, dtype=dtype)[boundary - 1 : boundary + 1]
    np.testing.assert_array_equal(actual, expected)
    touched = store.calls[2:]
    assert len(touched) == 4
    assert [item[0].content_type for item in touched] == [
        "application/json",
        "application/octet-stream",
        "application/json",
        "application/octet-stream",
    ]


def test_lazy_open_rejects_unqualified_oversized_and_raced_state(
    artifacts: _Artifacts,
) -> None:
    regional = _regional_artifacts(artifacts)
    unqualified = _ExactStore(regional, qualified=False)
    with pytest.raises(PrefixCheckpointV2Error, match="qualified"):
        LazyRegionalPrefixCheckpointReaderV1.open(
            unqualified,
            regional.envelope_receipt,
            expected_logical_manifest_sha256=artifacts.manifest.digest,
        )
    assert unqualified.calls == []

    oversized_store = _ExactStore(regional)
    oversized = replace(
        regional.envelope_receipt,
        size_bytes=MAX_REGIONAL_ENVELOPE_BYTES + 1,
    )
    with pytest.raises((PrefixCheckpointV2Error, ProtocolError), match="size|bound"):
        LazyRegionalPrefixCheckpointReaderV1.open(
            oversized_store,
            oversized,
            expected_logical_manifest_sha256=artifacts.manifest.digest,
        )
    assert oversized_store.calls == []

    raced = _ExactStore(regional)

    def delete_manifest(store, _receipt, call_count):
        if call_count == 1:
            del store.objects[regional.manifest_receipt]

    raced.after_read = delete_manifest
    with pytest.raises(PrefixCheckpointV2Error, match="absent"):
        LazyRegionalPrefixCheckpointReaderV1.open(
            raced,
            regional.envelope_receipt,
            expected_logical_manifest_sha256=artifacts.manifest.digest,
        )
    assert len(raced.calls) == 1


def test_reader_cannot_bypass_open_or_mutate_authority_and_rechecks_each_touch(
    artifacts: _Artifacts,
) -> None:
    regional = _regional_artifacts(artifacts)
    store = _ExactStore(regional)
    with pytest.raises(PrefixCheckpointV2Error, match="only through open"):
        LazyRegionalPrefixCheckpointReaderV1(
            store,
            regional.envelope,
            artifacts.manifest,
            maximum_read_bytes=64,
        )
    reader = LazyRegionalPrefixCheckpointReaderV1.open(
        store,
        regional.envelope_receipt,
        expected_logical_manifest_sha256=artifacts.manifest.digest,
    )
    with pytest.raises(FrozenInstanceError):
        reader.store = object()  # type: ignore[misc]
    first = artifacts.manifest.destinations[0].chunks[0]
    with pytest.raises(TypeError):
        reader._binding_receipts[first.chunk_id] = regional.binding_receipts[0]  # type: ignore[index]

    before = len(store.calls)
    store.store_id = "gcs-prefix-byte-store-v1:changed"
    with pytest.raises(PrefixCheckpointV2Error, match="changed after open"):
        reader.read_flat_region(
            first.destination_key,
            first.destination_flat_start,
            first.destination_flat_start + 1,
        )
    assert len(store.calls) == before


def test_touched_read_defends_against_post_open_binding_substitution(
    artifacts: _Artifacts,
) -> None:
    regional = _regional_artifacts(artifacts)
    store = _ExactStore(regional)
    reader = LazyRegionalPrefixCheckpointReaderV1.open(
        store,
        regional.envelope_receipt,
        expected_logical_manifest_sha256=artifacts.manifest.digest,
    )
    logical = tuple(
        receipt
        for destination in artifacts.manifest.destinations
        for receipt in destination.chunks
    )
    first, second = logical[:2]
    foreign_binding = next(
        item for item in regional.bindings if item.chunk_id == second.chunk_id
    )
    foreign_raw = foreign_binding.canonical_bytes()
    substituted_receipt = _physical_receipt(
        foreign_raw,
        object_id=regional_chunk_binding_object_id(
            artifacts.manifest.plan_id, first.chunk_id
        ),
        store_id=regional.envelope.store_id,
        region=regional.envelope.region,
        generation=999_999,
    )
    store.objects[substituted_receipt] = foreign_raw
    replacement = dict(reader._binding_receipts)
    replacement[first.chunk_id] = substituted_receipt
    object.__setattr__(reader, "_binding_receipts", MappingProxyType(replacement))
    with pytest.raises(PrefixCheckpointV2Error, match="substitutes"):
        reader.read_flat_region(
            first.destination_key,
            first.destination_flat_start,
            first.destination_flat_start + 1,
        )


def _real_pinned_destination_topology(
    plan_id: str,
) -> tuple[tuple[LogicalStoredDestinationV2, ...], int]:
    """Re-derive real keys, shapes, slices, and default conversion chunks."""

    geometry = DeepSeekR1CheckpointGeometryV1()
    limits = ConversionMemoryLimitsV1()
    specs = tuple(sorted(expected_prefix_logical_specs(geometry), key=lambda x: x.key))
    assert len(specs) == PINNED_SOURCE_LOGICAL_TENSORS
    dtype_bytes = {FP8_DTYPE: 1, BF16_DTYPE: 2, F32_DTYPE: 4}
    shard = "model-00001-of-000163.safetensors"
    by_destination: defaultdict[str, list[OutputChunkReceiptV1]] = defaultdict(list)
    destination_metadata: dict[str, tuple[str, tuple[int, ...]]] = {}
    chunk_count = 0

    for spec in specs:
        mapping = checkpoint_inventory._mapping_descriptor(spec, geometry)
        source_entry = SourceTensorEntryV1(
            key=spec.key,
            shard=shard,
            dtype=spec.dtype,
            shape=spec.shape,
            data_offsets=(0, math.prod(spec.shape) * dtype_bytes[spec.dtype]),
        )
        if mapping.scale_key is None:
            scale_entry = None
        else:
            scale_shape = tuple(
                (dimension + geometry.fp8_block_size - 1) // geometry.fp8_block_size
                for dimension in spec.shape
            )
            scale_entry = SourceTensorEntryV1(
                key=mapping.scale_key,
                shard=shard,
                dtype=F32_DTYPE,
                shape=scale_shape,
                data_offsets=(0, math.prod(scale_shape) * 4),
            )
        width = checkpoint_conversion._chunk_width(
            mapping,
            source_entry,
            scale_entry,
            limits,
            geometry.fp8_block_size,
        )
        stream_axis = 0 if mapping.transform == "identity-v1" else 1
        extent = spec.shape[stream_axis]
        trailing = (
            math.prod(spec.shape[1:])
            if mapping.transform == "identity-v1"
            else spec.shape[0]
        )
        if mapping.destination_slice_axis is None:
            destination_base = 0
        else:
            destination_base = (
                mapping.destination_slice_index
                * math.prod(mapping.destination_shape)
                // mapping.destination_shape[mapping.destination_slice_axis]
            )
        destination_metadata.setdefault(
            mapping.destination_key,
            (mapping.destination_dtype, mapping.destination_shape),
        )
        assert destination_metadata[mapping.destination_key] == (
            mapping.destination_dtype,
            mapping.destination_shape,
        )
        start = 0
        while start < extent:
            stop = min(start + width, extent)
            flat_start = destination_base + start * trailing
            flat_stop = destination_base + stop * trailing
            identity = f"{mapping.source_key}:{start}:{stop}".encode()
            chunk_id = hashlib.sha256(b"chunk:" + identity).hexdigest()
            payload_sha = hashlib.sha256(b"payload:" + identity).hexdigest()
            itemsize = 2 if mapping.destination_dtype == "bfloat16" else 4
            by_destination[mapping.destination_key].append(
                OutputChunkReceiptV1(
                    plan_id=plan_id,
                    chunk_id=chunk_id,
                    sink_id=PRODUCTION_PREFIX_SINK_ID,
                    destination_key=mapping.destination_key,
                    destination_flat_start=flat_start,
                    destination_flat_stop=flat_stop,
                    dtype=mapping.destination_dtype,
                    output_sha256=payload_sha,
                    output_size_bytes=(flat_stop - flat_start) * itemsize,
                    immutable_object_id=content_addressed_object_id(payload_sha),
                )
            )
            chunk_count += 1
            start = stop

    destinations: list[LogicalStoredDestinationV2] = []
    for destination_key in sorted(by_destination):
        dtype, shape = destination_metadata[destination_key]
        chunks = tuple(
            sorted(
                by_destination[destination_key],
                key=lambda item: item.destination_flat_start,
            )
        )
        destinations.append(
            LogicalStoredDestinationV2.build(
                destination_key=destination_key,
                dtype=dtype,
                shape=shape,
                chunks=chunks,
            )
        )
    return tuple(destinations), chunk_count


def test_full_pinned_profile_and_22108_leaf_control_records_fit_32_mib() -> None:
    plan_id = hashlib.sha256(b"pinned-plan").hexdigest()
    destinations, chunk_count = _real_pinned_destination_topology(plan_id)
    assert len(destinations) == PINNED_DESTINATION_TENSORS
    assert chunk_count == PINNED_TOTAL_CHUNKS
    assert sum(math.prod(item.shape) for item in destinations) == (
        PINNED_CONVERTED_PARAMETERS
    )
    assert sum(item.payload_bytes for item in destinations) == (
        PINNED_CONVERTED_PAYLOAD_BYTES
    )
    runtime = R1RuntimeLockIdentityV1(
        maxtext_revision=MAXTEXT_REVISION,
        pyproject_sha256="a" * 64,
        uv_lock_sha256="b" * 64,
    )
    manifest = LogicalPrefixCheckpointManifestV2(
        plan_id=plan_id,
        mode="pinned-r1-prefix",
        selection_sha256="1" * 64,
        full_inventory_sha256="2" * 64,
        selected_inventory_sha256="3" * 64,
        source_revision=HF_REVISION,
        maxtext_revision=MAXTEXT_REVISION,
        geometry_sha256="4" * 64,
        engine_sha256="5" * 64,
        numeric_codec_sha256="6" * 64,
        source_adapter_identity=SourceAdapterIdentityV1(
            adapter_id="pinned-source-adapter-v1",
            implementation_sha256="7" * 64,
        ),
        runtime_lock_identity=runtime,
        sink_id=PRODUCTION_PREFIX_SINK_ID,
        source_shard_count=PINNED_SOURCE_SHARDS,
        source_entry_count=PINNED_SOURCE_ENTRIES,
        source_logical_tensor_count=PINNED_SOURCE_LOGICAL_TENSORS,
        destination_tensor_count=PINNED_DESTINATION_TENSORS,
        converted_parameter_count=PINNED_CONVERTED_PARAMETERS,
        total_chunks=PINNED_TOTAL_CHUNKS,
        converted_payload_bytes=PINNED_CONVERTED_PAYLOAD_BYTES,
        max_estimated_resident_bytes=64 * 1024**2,
        target_layer_one_indexed=PINNED_TARGET_LAYER_ONE_INDEXED,
        target_semantics=prefix_v2.TARGET_SEMANTICS,
        destinations=destinations,
    )
    manifest.require_pinned_production_invariants()
    manifest_raw = manifest.canonical_bytes()
    assert len(manifest.destinations) == PINNED_DESTINATION_TENSORS
    assert sum(len(item.chunks) for item in manifest.destinations) == (
        PINNED_TOTAL_CHUNKS
    )
    assert manifest.source_entry_count == 43_679
    assert manifest.source_logical_tensor_count == 21_930
    assert manifest.destination_tensor_count == 510
    assert len(manifest_raw) > 14 * 1024**2
    assert len(manifest_raw) <= MAX_LOGICAL_MANIFEST_BYTES
    assert LogicalPrefixCheckpointManifestV2.from_json(manifest_raw) == manifest

    store_id = "gcs-prefix-byte-store-v1:pinned-size-test"
    region = "us-central1"
    manifest_receipt = _physical_receipt(
        manifest_raw,
        object_id=content_addressed_record_object_id(manifest.digest),
        store_id=store_id,
        region=region,
        generation=1,
    )
    replicas: list[RegionalChunkReplicaV1] = []
    replica_index = 0
    for destination in manifest.destinations:
        for receipt in destination.chunks:
            binding_raw = f"binding-{receipt.chunk_id}".encode().ljust(2_048, b"x")
            binding_receipt = _physical_receipt(
                binding_raw,
                object_id=regional_chunk_binding_object_id(
                    manifest.plan_id, receipt.chunk_id
                ),
                store_id=store_id,
                region=region,
                generation=replica_index + 2,
            )
            replicas.append(
                RegionalChunkReplicaV1(
                    chunk_id=receipt.chunk_id,
                    binding_receipt=binding_receipt,
                )
            )
            replica_index += 1
    envelope = RegionalPrefixCheckpointEnvelopeV1(
        logical_manifest_sha256=manifest.digest,
        logical_manifest_size_bytes=len(manifest_raw),
        plan_id=manifest.plan_id,
        region=region,
        store_id=store_id,
        manifest_receipt=manifest_receipt,
        chunk_replicas=tuple(replicas),
        total_chunks=manifest.total_chunks,
        converted_payload_bytes=manifest.converted_payload_bytes,
    )
    envelope.validate_for_manifest(manifest)
    envelope_raw = envelope.canonical_bytes()
    assert len(envelope_raw) > 16 * 1024**2
    assert len(envelope_raw) <= MAX_REGIONAL_ENVELOPE_BYTES
    assert RegionalPrefixCheckpointEnvelopeV1.from_json(envelope_raw) == envelope
