from __future__ import annotations

import ast
import base64
from dataclasses import FrozenInstanceError, dataclass, replace
import hashlib
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import google_crc32c
import numpy as np
import pytest

import moe_sae.r1.prefix_restore_authority as restore_authority_module
from moe_sae.r1.checkpoint_conversion import (
    R1RuntimeLockIdentityV1,
    SourceAdapterIdentityV1,
    conversion_engine_sha256,
    numeric_codec_identity_sha256,
)
from moe_sae.r1.checkpoint_inventory import (
    HF_REVISION,
    MAXTEXT_REVISION,
    PINNED_GEOMETRY,
)
from moe_sae.r1.gcs_immutable_byte_store import (
    GCSPhysicalObjectReceiptV1,
    content_addressed_object_id,
    content_addressed_record_object_id,
)
from moe_sae.r1.prefix_batchsplit_contract import (
    PINNED_UNBOUND_PREFIX_PERFORMANCE_TREE_PLAN_SHA256,
)
from moe_sae.r1.prefix_checkpoint import PlanSourceIdentityBindingV1
from moe_sae.r1.prefix_checkpoint_v2 import (
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
    LogicalPrefixCheckpointManifestV2,
    LogicalStoredDestinationV2,
    PrefixCheckpointV2Error,
    RegionalChunkReplicaV1,
    RegionalPrefixCheckpointEnvelopeV1,
    RegionalPrefixChunkBindingV1,
    regional_chunk_binding_object_id,
)
from moe_sae.r1.prefix_restore_authority import (
    LazyPerformanceLeafV1,
    PrefixDirectRestoreAuthorityV1,
    PrefixRestoreAuthorityError,
    PrefixRestoreBindingV1,
    PrefixRestoreIdentityExpectationV1,
)
from tests.r1.test_prefix_checkpoint_v2 import (
    _physical_receipt,
    _real_pinned_destination_topology,
)


def _crc32c(raw: bytes) -> str:
    checksum = google_crc32c.Checksum(raw)
    return base64.b64encode(checksum.digest()).decode("ascii")


class _EqualString(str):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    __hash__ = str.__hash__


class _SwitchingDestination(str):
    """Change dictionary identity between the leaf and reader lookups."""

    def __new__(
        cls,
        value: str,
        first_key: str,
        second_key: str,
    ) -> "_SwitchingDestination":
        instance = super().__new__(cls, value)
        instance._hashes = iter((hash(first_key), hash(second_key)))
        return instance

    def __hash__(self) -> int:
        return next(self._hashes)

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


class _ExactStore:
    def __init__(
        self,
        *,
        store_id: str,
        region: str,
        objects: dict[GCSPhysicalObjectReceiptV1, bytes],
    ) -> None:
        self.store_id = store_id
        self.region = region
        self.distributed_identity_qualified = True
        self.objects = objects
        self.calls: list[tuple[GCSPhysicalObjectReceiptV1, int, int, int]] = []

    def read_exact(
        self,
        receipt: GCSPhysicalObjectReceiptV1,
        *,
        offset: int,
        length: int,
        maximum_bytes: int,
    ) -> bytes:
        self.calls.append((receipt, offset, length, maximum_bytes))
        if receipt.store_id != self.store_id or receipt.region != self.region:
            raise PrefixCheckpointV2Error("foreign exact receipt")
        if type(offset) is not int or type(length) is not int:
            raise PrefixCheckpointV2Error("invalid exact interval")
        if length > maximum_bytes or offset < 0 or offset + length > receipt.size_bytes:
            raise PrefixCheckpointV2Error("unbounded exact interval")
        raw = self.objects.get(receipt)
        if raw is None:
            raise PrefixCheckpointV2Error("exact generation is absent")
        if len(raw) != receipt.size_bytes:
            raise PrefixCheckpointV2Error("stored size differs from exact receipt")
        return raw[offset : offset + length]


@dataclass(frozen=True)
class _PinnedControlFixture:
    authority: PrefixDirectRestoreAuthorityV1
    expectation: PrefixRestoreIdentityExpectationV1
    store: _ExactStore
    envelope_receipt: GCSPhysicalObjectReceiptV1
    manifest: LogicalPrefixCheckpointManifestV2
    selected_destination_key: str
    selected_payload: bytes
    selected_binding_receipt: GCSPhysicalObjectReceiptV1
    selected_payload_receipt: GCSPhysicalObjectReceiptV1


@pytest.fixture(scope="module")
def pinned_control() -> _PinnedControlFixture:
    plan_id = hashlib.sha256(b"restore-authority-test-plan").hexdigest()
    destinations, chunk_count = _real_pinned_destination_topology(plan_id)
    assert len(destinations) == PINNED_DESTINATION_TENSORS
    assert chunk_count == PINNED_TOTAL_CHUNKS

    selected_index = min(
        range(len(destinations)),
        key=lambda index: destinations[index].payload_bytes,
    )
    selected = destinations[selected_index]
    selected_receipt = selected.chunks[0]
    selected_payload = bytes(selected_receipt.output_size_bytes)
    selected_sha256 = hashlib.sha256(selected_payload).hexdigest()
    selected_receipt = replace(
        selected_receipt,
        output_sha256=selected_sha256,
        immutable_object_id=content_addressed_object_id(selected_sha256),
    )
    selected_chunks = (selected_receipt, *selected.chunks[1:])
    selected = LogicalStoredDestinationV2.build(
        destination_key=selected.destination_key,
        dtype=selected.dtype,
        shape=selected.shape,
        chunks=selected_chunks,
    )
    destinations = (
        *destinations[:selected_index],
        selected,
        *destinations[selected_index + 1 :],
    )

    source_adapter = SourceAdapterIdentityV1(
        adapter_id="restore-authority-test-source-adapter-v1",
        implementation_sha256="7" * 64,
    )
    manifest = LogicalPrefixCheckpointManifestV2(
        plan_id=plan_id,
        mode="pinned-r1-prefix",
        selection_sha256="1" * 64,
        full_inventory_sha256="2" * 64,
        selected_inventory_sha256="3" * 64,
        source_revision=HF_REVISION,
        maxtext_revision=MAXTEXT_REVISION,
        geometry_sha256=PINNED_GEOMETRY.digest,
        engine_sha256=conversion_engine_sha256(),
        numeric_codec_sha256=numeric_codec_identity_sha256(),
        source_adapter_identity=source_adapter,
        runtime_lock_identity=R1RuntimeLockIdentityV1(
            maxtext_revision=MAXTEXT_REVISION,
            pyproject_sha256="a" * 64,
            uv_lock_sha256="b" * 64,
        ),
        sink_id=PRODUCTION_PREFIX_SINK_ID,
        source_shard_count=PINNED_SOURCE_SHARDS,
        source_entry_count=PINNED_SOURCE_ENTRIES,
        source_logical_tensor_count=PINNED_SOURCE_LOGICAL_TENSORS,
        destination_tensor_count=PINNED_DESTINATION_TENSORS,
        converted_parameter_count=PINNED_CONVERTED_PARAMETERS,
        total_chunks=PINNED_TOTAL_CHUNKS,
        converted_payload_bytes=PINNED_CONVERTED_PAYLOAD_BYTES,
        max_estimated_resident_bytes=512 * 1024**2,
        target_layer_one_indexed=PINNED_TARGET_LAYER_ONE_INDEXED,
        target_semantics="attention-router-and-routed-experts-only-before-shared-expert",
        destinations=destinations,
    )

    store_id = "gcs-prefix-byte-store-v2:restore-authority-test"
    region = "us-central1"
    manifest_raw = manifest.canonical_bytes()
    manifest_receipt = _physical_receipt(
        manifest_raw,
        object_id=content_addressed_record_object_id(manifest.digest),
        store_id=store_id,
        region=region,
        generation=10,
    )

    selected_payload_receipt = _physical_receipt(
        selected_payload,
        object_id=selected_receipt.immutable_object_id,
        store_id=store_id,
        region=region,
        generation=20,
    )
    selected_binding = RegionalPrefixChunkBindingV1(
        plan_id=plan_id,
        chunk_id=selected_receipt.chunk_id,
        output_receipt=selected_receipt,
        physical_receipt=selected_payload_receipt,
        source_lineage=(
            PlanSourceIdentityBindingV1(
                role="source",
                shard="model-00001-of-000163.safetensors",
                sha256="8" * 64,
                size_bytes=1,
                source_adapter_identity=source_adapter,
            ),
        ),
    )
    selected_binding_raw = selected_binding.canonical_bytes()

    replicas: list[RegionalChunkReplicaV1] = []
    selected_binding_receipt: GCSPhysicalObjectReceiptV1 | None = None
    generation = 100
    for destination in manifest.destinations:
        for receipt in destination.chunks:
            object_id = regional_chunk_binding_object_id(plan_id, receipt.chunk_id)
            if receipt.chunk_id == selected_receipt.chunk_id:
                binding_raw = selected_binding_raw
            else:
                binding_raw = f"unread-binding:{receipt.chunk_id}".encode("ascii")
            binding_receipt = _physical_receipt(
                binding_raw,
                object_id=object_id,
                store_id=store_id,
                region=region,
                generation=generation,
            )
            generation += 1
            if receipt.chunk_id == selected_receipt.chunk_id:
                selected_binding_receipt = binding_receipt
            replicas.append(
                RegionalChunkReplicaV1(
                    chunk_id=receipt.chunk_id,
                    binding_receipt=binding_receipt,
                )
            )
    assert selected_binding_receipt is not None

    envelope = RegionalPrefixCheckpointEnvelopeV1(
        logical_manifest_sha256=manifest.digest,
        logical_manifest_size_bytes=len(manifest_raw),
        plan_id=plan_id,
        region=region,
        store_id=store_id,
        manifest_receipt=manifest_receipt,
        chunk_replicas=tuple(replicas),
        total_chunks=PINNED_TOTAL_CHUNKS,
        converted_payload_bytes=PINNED_CONVERTED_PAYLOAD_BYTES,
    )
    envelope_raw = envelope.canonical_bytes()
    envelope_receipt = _physical_receipt(
        envelope_raw,
        object_id=content_addressed_record_object_id(envelope.digest),
        store_id=store_id,
        region=region,
        generation=99_999,
    )
    store = _ExactStore(
        store_id=store_id,
        region=region,
        objects={
            envelope_receipt: envelope_raw,
            manifest_receipt: manifest_raw,
            selected_binding_receipt: selected_binding_raw,
            selected_payload_receipt: selected_payload,
        },
    )
    expectation = PrefixRestoreIdentityExpectationV1(
        logical_manifest_sha256=manifest.digest,
        selection_sha256=manifest.selection_sha256,
        full_inventory_sha256=manifest.full_inventory_sha256,
        selected_inventory_sha256=manifest.selected_inventory_sha256,
    )
    authority = PrefixDirectRestoreAuthorityV1.open(
        store,
        envelope_receipt,
        expected_identity=expectation,
        maximum_control_bytes=MAX_REGIONAL_ENVELOPE_BYTES,
        maximum_read_bytes=64,
    )
    return _PinnedControlFixture(
        authority=authority,
        expectation=expectation,
        store=store,
        envelope_receipt=envelope_receipt,
        manifest=manifest,
        selected_destination_key=selected.destination_key,
        selected_payload=selected_payload,
        selected_binding_receipt=selected_binding_receipt,
        selected_payload_receipt=selected_payload_receipt,
    )


def test_import_boundary_is_jax_and_maxtext_free() -> None:
    path = (
        Path(__file__).parents[2]
        / "src"
        / "moe_sae"
        / "r1"
        / "prefix_restore_authority.py"
    )
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(parsed)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(parsed)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        name == "jax"
        or name.startswith("jax.")
        or name == "maxtext"
        or name.startswith("maxtext.")
        for name in imported
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys; sys.path.insert(0, {str(path.parents[2])!r}); "
                "import moe_sae.r1.prefix_restore_authority; "
                "assert 'jax' not in sys.modules; "
                "assert 'maxtext' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_open_binds_exact_inventory_and_derives_510_to_44_without_payload_reads(
    pinned_control: _PinnedControlFixture,
) -> None:
    authority = pinned_control.authority
    assert len(pinned_control.store.calls) == 2
    assert [call[0] for call in pinned_control.store.calls] == [
        pinned_control.envelope_receipt,
        authority._reader.envelope.manifest_receipt,
    ]
    assert all(
        call[0].content_type == "application/json"
        for call in pinned_control.store.calls
    )

    binding = authority.binding
    assert binding.inventory_bound is True
    assert binding.control_plane_verified is True
    assert binding.released_checkpoint_observed is False
    assert binding.payload_restore_completed is False
    assert binding.qualification_complete is False
    assert binding.production_sealable is False
    assert binding.production_admission_enabled is False
    assert binding.logical_manifest_sha256 == pinned_control.manifest.digest
    assert (
        binding.regional_envelope_receipt_sha256
        == pinned_control.envelope_receipt.digest
    )
    assert (
        binding.performance_plan_sha256
        == PINNED_UNBOUND_PREFIX_PERFORMANCE_TREE_PLAN_SHA256
    )
    assert binding.regional_envelope_generation == "99999"
    assert len(authority.plan.destinations) == 510
    assert len(authority.leaves) == 44
    assert tuple(view.spec.leaf_key for view in authority.leaves) == tuple(
        leaf.leaf_key for leaf in authority.plan.leaves
    )
    for view, planned_leaf in zip(authority.leaves, authority.plan.leaves, strict=True):
        assert view.spec.logical_shape == planned_leaf.logical_shape
        assert view.spec.scan_axis == planned_leaf.scan_axis
        assert tuple(item.destination_key for item in view.spec.sources) == (
            planned_leaf.source_destination_keys
        )
    assert tuple(authority.iter_leaves()) == authority.leaves


def test_binding_rejects_unbound_restore_and_production_claims(
    pinned_control: _PinnedControlFixture,
) -> None:
    binding = pinned_control.authority.binding
    with pytest.raises(PrefixRestoreAuthorityError, match="unbound"):
        replace(binding, inventory_bound=False)
    with pytest.raises(PrefixRestoreAuthorityError, match="source observation"):
        replace(binding, released_checkpoint_observed=True)
    with pytest.raises(PrefixRestoreAuthorityError, match="source observation"):
        replace(binding, payload_restore_completed=True)
    with pytest.raises(PrefixRestoreAuthorityError, match="source observation"):
        replace(binding, qualification_complete=True)
    with pytest.raises(PrefixRestoreAuthorityError, match="source observation"):
        replace(binding, production_sealable=True)
    with pytest.raises(PrefixRestoreAuthorityError, match="source observation"):
        replace(binding, production_admission_enabled=True)
    with pytest.raises(PrefixRestoreAuthorityError, match="pinned authority"):
        replace(binding, maxtext_revision="deadbeef")
    with pytest.raises(FrozenInstanceError):
        binding.inventory_bound = False  # type: ignore[misc]

    with pytest.raises(PrefixRestoreAuthorityError, match="only through open"):
        PrefixDirectRestoreAuthorityV1(
            binding,
            pinned_control.authority.plan,
            pinned_control.authority._reader,
            pinned_control.authority.leaves,
        )
    with pytest.raises(
        PrefixRestoreAuthorityError, match="only through restore authority"
    ):
        LazyPerformanceLeafV1(
            pinned_control.authority.leaves[0].spec,
            pinned_control.authority._reader,
            maximum_read_bytes=64,
        )


def test_identity_and_binding_reject_equality_overriding_string_subclasses(
    pinned_control: _PinnedControlFixture,
) -> None:
    expectation = pinned_control.expectation
    with pytest.raises(PrefixRestoreAuthorityError, match="exact immutable string"):
        PrefixRestoreIdentityExpectationV1(
            logical_manifest_sha256=_EqualString("f" * 64),
            selection_sha256=expectation.selection_sha256,
            full_inventory_sha256=expectation.full_inventory_sha256,
            selected_inventory_sha256=expectation.selected_inventory_sha256,
        )

    binding = pinned_control.authority.binding
    values = binding.as_dict()
    values.pop("schema")
    values["performance_plan_sha256"] = _EqualString("f" * 64)
    with pytest.raises(PrefixRestoreAuthorityError, match="exact immutable string"):
        PrefixRestoreBindingV1(**values)


def test_switching_hash_cannot_cross_leaf_or_touch_object_storage(
    pinned_control: _PinnedControlFixture,
) -> None:
    authority = pinned_control.authority
    source_leaf = authority.leaves[0]
    source_key = source_leaf.spec.sources[0].destination_key
    foreign_key = next(
        source.destination_key
        for leaf in authority.leaves[1:]
        for source in leaf.spec.sources
    )
    switching = _SwitchingDestination(foreign_key, source_key, foreign_key)
    before = len(pinned_control.store.calls)

    with pytest.raises(PrefixRestoreAuthorityError, match="exact immutable string"):
        source_leaf.read_destination_flat(switching, 0, 1)

    assert len(pinned_control.store.calls) == before


def test_private_constructor_revalidates_complete_binding_and_leaf_specs(
    pinned_control: _PinnedControlFixture,
) -> None:
    authority = pinned_control.authority
    forged_binding = replace(
        authority.binding,
        selection_sha256="4" * 64,
        full_inventory_sha256="5" * 64,
        selected_inventory_sha256="6" * 64,
        regional_envelope_sha256="7" * 64,
        regional_envelope_receipt_sha256="8" * 64,
        regional_envelope_generation="1",
        store_id="attacker-store",
        region="us-east1",
    )
    with pytest.raises(PrefixRestoreAuthorityError, match="cross-bound"):
        PrefixDirectRestoreAuthorityV1(
            forged_binding,
            authority.plan,
            authority._reader,
            authority.leaves,
            envelope_receipt=pinned_control.envelope_receipt,
            _capability=restore_authority_module._AUTHORITY_OPEN_CAPABILITY,
        )

    receipt_prefix = pinned_control.envelope_receipt.uri.rsplit("/", 1)[0]
    unread_receipt = replace(
        pinned_control.envelope_receipt,
        object_id="attacker-envelope.json",
        uri=f"{receipt_prefix}/attacker-envelope.json",
        generation="123456",
    )
    unread_binding = replace(
        authority.binding,
        regional_envelope_receipt_sha256=unread_receipt.digest,
        regional_envelope_generation=unread_receipt.generation,
    )
    with pytest.raises(PrefixRestoreAuthorityError, match="cross-bound"):
        PrefixDirectRestoreAuthorityV1(
            unread_binding,
            authority.plan,
            authority._reader,
            authority.leaves,
            envelope_receipt=unread_receipt,
            _capability=restore_authority_module._AUTHORITY_OPEN_CAPABILITY,
        )

    first = authority.leaves[0]
    foreign = authority.leaves[1]
    forged_leaf = object.__new__(LazyPerformanceLeafV1)
    object.__setattr__(forged_leaf, "spec", first.spec)
    object.__setattr__(forged_leaf, "maximum_read_bytes", first.maximum_read_bytes)
    object.__setattr__(forged_leaf, "_reader", authority._reader)
    object.__setattr__(forged_leaf, "_sources", foreign._sources)
    forged_leaves = (forged_leaf, *authority.leaves[1:])

    with pytest.raises(PrefixRestoreAuthorityError, match="cross-bound"):
        PrefixDirectRestoreAuthorityV1(
            authority.binding,
            authority.plan,
            authority._reader,
            forged_leaves,
            envelope_receipt=pinned_control.envelope_receipt,
            _capability=restore_authority_module._AUTHORITY_OPEN_CAPABILITY,
        )


def test_leaf_read_is_lazy_bounded_and_exactly_generation_pinned(
    pinned_control: _PinnedControlFixture,
) -> None:
    authority = pinned_control.authority
    selected_leaf = next(
        view
        for view in authority.leaves
        if pinned_control.selected_destination_key
        in {source.destination_key for source in view.spec.sources}
    )
    selected_index = tuple(
        source.destination_key for source in selected_leaf.spec.sources
    ).index(pinned_control.selected_destination_key)
    source = selected_leaf.source(selected_index)
    initial_calls = len(pinned_control.store.calls)
    itemsize = 2 if source.dtype == "bfloat16" else 4

    actual = selected_leaf.read_source_flat(selected_index, 0, 2)
    assert actual.shape == (2,)
    assert not np.any(actual)
    touched = pinned_control.store.calls[initial_calls:]
    assert [call[0] for call in touched] == [
        pinned_control.selected_binding_receipt,
        pinned_control.selected_payload_receipt,
    ]
    assert touched[1][1:3] == (0, 2 * itemsize)
    assert all(call[2] <= call[3] for call in touched)
    assert touched[1][3] == 64

    before = len(pinned_control.store.calls)
    maximum_elements = 64 // itemsize
    with pytest.raises(PrefixRestoreAuthorityError, match="exceeds"):
        selected_leaf.read_source_flat(selected_index, 0, maximum_elements + 1)
    assert len(pinned_control.store.calls) == before

    foreign = next(
        source
        for leaf in authority.leaves
        if leaf is not selected_leaf
        for source in leaf.spec.sources
    )
    with pytest.raises(PrefixRestoreAuthorityError, match="outside"):
        selected_leaf.read_destination_flat(foreign.destination_key, 0, 1)
    assert len(pinned_control.store.calls) == before


def test_region_iterator_and_adapter_boundary_do_not_create_evidence(
    pinned_control: _PinnedControlFixture,
) -> None:
    authority = pinned_control.authority
    selected_leaf = next(
        view
        for view in authority.leaves
        if pinned_control.selected_destination_key
        in {source.destination_key for source in view.spec.sources}
    )
    selected_index = tuple(
        source.destination_key for source in selected_leaf.spec.sources
    ).index(pinned_control.selected_destination_key)
    source = selected_leaf.source(selected_index)
    itemsize = 2 if source.dtype == "bfloat16" else 4
    regions = selected_leaf.iter_source_regions(
        selected_index,
        maximum_region_bytes=itemsize,
    )
    start, stop, values = next(regions)
    regions.close()
    assert (start, stop) == (0, 1)
    assert values.shape == (1,)
    assert not np.any(values)

    class _Adapter:
        def arrange_leaf(self, leaf: LazyPerformanceLeafV1) -> Any:
            return (
                leaf.spec.leaf_key,
                tuple(source.destination_key for source in leaf.spec.sources),
            )

    result = authority.arrange_leaf(selected_leaf.spec.leaf_key, _Adapter())
    assert result == (
        selected_leaf.spec.leaf_key,
        tuple(source.destination_key for source in selected_leaf.spec.sources),
    )
    assert authority.binding.payload_restore_completed is False
    assert authority.binding.qualification_complete is False
    assert authority.binding.production_admission_enabled is False
    with pytest.raises(PrefixRestoreAuthorityError, match="implement arrange_leaf"):
        authority.arrange_leaf(selected_leaf.spec.leaf_key, object())
    with pytest.raises(PrefixRestoreAuthorityError, match="unknown performance leaf"):
        authority.leaf("not-a-real-leaf")


def test_open_rejects_missing_or_mismatched_inventory_binding_before_authority(
    pinned_control: _PinnedControlFixture,
) -> None:
    empty_store = _ExactStore(
        store_id=pinned_control.store.store_id,
        region=pinned_control.store.region,
        objects=dict(pinned_control.store.objects),
    )
    with pytest.raises(PrefixRestoreAuthorityError, match="exact bound identity"):
        PrefixDirectRestoreAuthorityV1.open(
            empty_store,
            pinned_control.envelope_receipt,
            expected_identity=None,  # type: ignore[arg-type]
            maximum_control_bytes=MAX_REGIONAL_ENVELOPE_BYTES,
        )
    assert empty_store.calls == []

    mismatch = replace(
        pinned_control.expectation,
        selected_inventory_sha256="f" * 64,
    )
    before = len(pinned_control.store.calls)
    with pytest.raises(PrefixRestoreAuthorityError, match="expected inventory binding"):
        PrefixDirectRestoreAuthorityV1._validate_opened_control_plane(
            pinned_control.authority._reader,
            pinned_control.envelope_receipt,
            mismatch,
            pinned_control.authority.plan,
        )
    assert len(pinned_control.store.calls) == before


def test_test_fixture_payload_receipt_is_internally_exact(
    pinned_control: _PinnedControlFixture,
) -> None:
    receipt = pinned_control.selected_payload_receipt
    assert receipt.sha256 == hashlib.sha256(pinned_control.selected_payload).hexdigest()
    assert receipt.crc32c == _crc32c(pinned_control.selected_payload)
    assert receipt.size_bytes == len(pinned_control.selected_payload)
    source = next(
        source
        for leaf in pinned_control.authority.leaves
        for source in leaf.spec.sources
        if source.destination_key == pinned_control.selected_destination_key
    )
    itemsize = 2 if source.dtype == "bfloat16" else 4
    assert math.prod(
        next(
            source.shape
            for leaf in pinned_control.authority.leaves
            for source in leaf.spec.sources
            if source.destination_key == pinned_control.selected_destination_key
        )
    ) * itemsize >= len(pinned_control.selected_payload)
