from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable

import google_crc32c
import numpy as np
import pytest

from moe_sae.r1.checkpoint_conversion import (
    CheckpointConversionError,
    ConversionMemoryLimitsV1,
    execute_streaming_conversion,
)
from moe_sae.r1.checkpoint_inventory import DeepSeekR1CheckpointGeometryV1
from moe_sae.r1.gcs_immutable_byte_store import (
    MAX_APPLICATION_STREAM_SLICE_BYTES,
    MAX_IMMUTABLE_SNAPSHOT_BYTES,
    GCSPhysicalObjectReceiptV1,
)
from moe_sae.r1.gcs_prefix_checkpoint_sink import (
    GCSPrefixCheckpointSinkError,
    GCSRegionalPrefixCheckpointSinkV1,
)
from moe_sae.r1.prefix_checkpoint_v2 import (
    MAX_LOGICAL_MANIFEST_BYTES,
    MAX_REGIONAL_ENVELOPE_BYTES,
    LazyRegionalPrefixCheckpointReaderV1,
    build_regional_prefix_chunk_binding_v1,
    regional_chunk_binding_object_id,
)
from tests.r1.test_checkpoint_conversion import (
    _Source,
    _fixture_arrays,
    _plan,
    _wire_inventory,
)
from tests.r1.test_gcs_immutable_byte_store import (
    _DriftStr,
    _store as _unqualified_gcs_store,
)


_MIB = 1024**2


def _crc32c(raw: bytes) -> str:
    checksum = google_crc32c.Checksum(raw)
    return base64.b64encode(checksum.digest()).decode("ascii")


class _InjectedCrash(CheckpointConversionError):
    pass


class _ExplicitTestOnlyQualifiedStore:
    """Exact-path fake; qualification is explicit and confined to this test."""

    def __init__(
        self,
        *,
        qualified: bool = True,
        stream_buffer_bytes: int = 16,
        immutable_snapshot_bytes: int = MAX_IMMUTABLE_SNAPSHOT_BYTES,
        maximum_object_bytes: int = 64 * _MIB,
    ) -> None:
        self.store_id = "gcs-prefix-byte-store-v2:explicit-test-only"
        self.region = "us-central1"
        self.distributed_identity_qualified = qualified
        self.stream_buffer_bytes = stream_buffer_bytes
        self.immutable_snapshot_bytes = immutable_snapshot_bytes
        self.maximum_object_bytes = maximum_object_bytes
        self.objects: dict[str, tuple[bytes, GCSPhysicalObjectReceiptV1]] = {}
        self.calls: list[tuple[str, str, str]] = []
        self.next_generation = 100
        self.fail_once: tuple[str, str] | None = None
        self.after_probe: Callable[["_ExplicitTestOnlyQualifiedStore"], None] | None = (
            None
        )
        self.before_create: (
            Callable[["_ExplicitTestOnlyQualifiedStore", str], None] | None
        ) = None

    @staticmethod
    def _stage(object_id: str, raw: bytes) -> str:
        if object_id.endswith(".bin"):
            return "payload"
        if "/bindings/" in object_id:
            return "binding"
        try:
            schema = json.loads(raw)["schema"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return "unknown-json"
        if schema == "moe-sae.r1.logical-prefix-checkpoint-manifest.v2":
            return "manifest"
        if schema == "moe-sae.r1.regional-prefix-checkpoint-envelope.v1":
            return "envelope"
        return "unknown-json"

    def _maybe_crash(self, stage: str, boundary: str) -> None:
        if self.fail_once == (stage, boundary):
            self.fail_once = None
            raise _InjectedCrash(f"injected {stage} {boundary} crash")

    def _receipt(self, object_id: str, raw: bytes) -> GCSPhysicalObjectReceiptV1:
        digest = hashlib.sha256(raw).hexdigest()
        receipt = GCSPhysicalObjectReceiptV1(
            store_id=self.store_id,
            object_id=object_id,
            uri=f"gs://r1-test-bucket/prefix/{object_id}",
            region=self.region,
            generation=str(self.next_generation),
            metageneration="1",
            size_bytes=len(raw),
            crc32c=_crc32c(raw),
            sha256=digest,
            content_type=(
                "application/octet-stream"
                if object_id.endswith(".bin")
                else "application/json"
            ),
        )
        self.next_generation += 1
        return receipt

    def _create(
        self,
        object_id: str,
        raw: bytes,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> GCSPhysicalObjectReceiptV1:
        stage = self._stage(object_id, raw)
        self.calls.append(("create", stage, object_id))
        if self.before_create is not None:
            hook = self.before_create
            self.before_create = None
            hook(self, stage)
        self._maybe_crash(stage, "before")
        if (
            len(raw) != expected_size_bytes
            or hashlib.sha256(raw).hexdigest() != expected_sha256
            or len(raw) > self.maximum_object_bytes
        ):
            raise CheckpointConversionError("fake create identity differs")
        existing = self.objects.get(object_id)
        if existing is None:
            receipt = self._receipt(object_id, raw)
            self.objects[object_id] = (raw, receipt)
        elif existing[0] != raw:
            raise CheckpointConversionError("immutable fake object conflicts")
        else:
            receipt = existing[1]
        self._maybe_crash(stage, "after")
        return receipt

    def create_memoryview_or_verify(
        self,
        object_id: str,
        payload: memoryview,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> GCSPhysicalObjectReceiptV1:
        if type(payload) is not memoryview:
            raise CheckpointConversionError("fake requires a memoryview")
        return self._create(
            object_id,
            payload.tobytes(),
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )

    def create_or_verify(
        self,
        object_id: str,
        payload: bytes,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> GCSPhysicalObjectReceiptV1:
        if type(payload) is not bytes:
            raise CheckpointConversionError("fake requires exact bytes")
        return self._create(
            object_id,
            payload,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )

    def probe_bounded(
        self,
        object_id: str,
        *,
        maximum_bytes: int,
    ) -> tuple[bytes, GCSPhysicalObjectReceiptV1] | None:
        self.calls.append(("probe", "binding", object_id))
        current = self.objects.get(object_id)
        if current is not None and len(current[0]) > maximum_bytes:
            raise CheckpointConversionError("fake bounded probe overflow")
        if self.after_probe is not None:
            hook = self.after_probe
            self.after_probe = None
            hook(self)
        return current

    def verify(
        self,
        receipt: GCSPhysicalObjectReceiptV1,
    ) -> GCSPhysicalObjectReceiptV1:
        self.calls.append(("verify", "payload", receipt.object_id))
        current = self.objects.get(receipt.object_id)
        if (
            current is None
            or current[1] != receipt
            or hashlib.sha256(current[0]).hexdigest() != receipt.sha256
            or _crc32c(current[0]) != receipt.crc32c
        ):
            raise CheckpointConversionError("fake exact generation changed")
        return receipt

    def read_exact(
        self,
        receipt: GCSPhysicalObjectReceiptV1,
        *,
        offset: int,
        length: int,
        maximum_bytes: int,
    ) -> bytes:
        current = self.objects.get(receipt.object_id)
        if (
            current is None
            or current[1] != receipt
            or hashlib.sha256(current[0]).hexdigest() != receipt.sha256
            or _crc32c(current[0]) != receipt.crc32c
            or length > maximum_bytes
            or offset < 0
            or length <= 0
            or offset + length > len(current[0])
        ):
            raise CheckpointConversionError("fake exact read failed")
        stage = self._stage(receipt.object_id, current[0])
        self.calls.append(("read", stage, receipt.object_id))
        return current[0][offset : offset + length]

    def inject(self, object_id: str, raw: bytes) -> GCSPhysicalObjectReceiptV1:
        receipt = self._receipt(object_id, raw)
        self.objects[object_id] = (raw, receipt)
        return receipt

    def replace_live_generation(self, object_id: str) -> None:
        raw = self.objects[object_id][0]
        self.objects[object_id] = (raw, self._receipt(object_id, raw))

    def corrupt_live_bytes(self, object_id: str) -> None:
        raw, receipt = self.objects[object_id]
        changed = bytes([raw[0] ^ 1]) + raw[1:]
        self.objects[object_id] = (changed, receipt)

    def count_stage(self, stage: str) -> int:
        return sum(
            1
            for object_id, (raw, _receipt) in self.objects.items()
            if self._stage(object_id, raw) == stage
        )


@dataclass(frozen=True)
class _Fixture:
    selection: Any
    plan: Any


@pytest.fixture(scope="module")
def fixture() -> _Fixture:
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
            sink_staging_buffer_bytes=80,
        ),
    )
    return _Fixture(selection=selection, plan=plan)


def _execute(
    fixture: _Fixture,
    store: _ExplicitTestOnlyQualifiedStore,
):
    sink = GCSRegionalPrefixCheckpointSinkV1(store, fixture.plan)
    summary = execute_streaming_conversion(
        fixture.plan,
        _Source(_fixture_arrays(fixture.selection)),
        sink,
    )
    return sink, summary


def test_admission_is_qualified_bounded_and_topology_neutral(
    fixture: _Fixture,
) -> None:
    real_unqualified_store, _client = _unqualified_gcs_store()
    assert real_unqualified_store.distributed_identity_qualified is False
    with pytest.raises(GCSPrefixCheckpointSinkError, match="qualified"):
        GCSRegionalPrefixCheckpointSinkV1(
            real_unqualified_store,
            fixture.plan,
        )
    with pytest.raises(GCSPrefixCheckpointSinkError, match="qualified"):
        GCSRegionalPrefixCheckpointSinkV1(
            _ExplicitTestOnlyQualifiedStore(qualified=False),
            fixture.plan,
        )
    under_staged = _plan(
        fixture.selection,
        ConversionMemoryLimitsV1(
            max_source_blob_bytes=1024**2,
            max_resident_bytes=512,
            max_output_chunk_bytes=64,
            sink_staging_buffer_bytes=79,
        ),
    )
    with pytest.raises(GCSPrefixCheckpointSinkError, match="staging"):
        GCSRegionalPrefixCheckpointSinkV1(
            _ExplicitTestOnlyQualifiedStore(),
            under_staged,
        )
    with pytest.raises(GCSPrefixCheckpointSinkError, match="stream-slice"):
        GCSRegionalPrefixCheckpointSinkV1(
            _ExplicitTestOnlyQualifiedStore(
                stream_buffer_bytes=MAX_APPLICATION_STREAM_SLICE_BYTES + 1
            ),
            fixture.plan,
        )
    with pytest.raises(GCSPrefixCheckpointSinkError, match="snapshot"):
        GCSRegionalPrefixCheckpointSinkV1(
            _ExplicitTestOnlyQualifiedStore(immutable_snapshot_bytes=63),
            fixture.plan,
        )
    with pytest.raises(GCSPrefixCheckpointSinkError, match="control"):
        GCSRegionalPrefixCheckpointSinkV1(
            _ExplicitTestOnlyQualifiedStore(
                maximum_object_bytes=max(
                    MAX_LOGICAL_MANIFEST_BYTES,
                    MAX_REGIONAL_ENVELOPE_BYTES,
                )
                - 1
            ),
            fixture.plan,
        )

    store = _ExplicitTestOnlyQualifiedStore()
    sink = GCSRegionalPrefixCheckpointSinkV1(store, fixture.plan)
    assert fixture.plan.limits.sink_staging_buffer_bytes == (
        fixture.plan.limits.max_output_chunk_bytes + store.stream_buffer_bytes
    )
    assert (
        fixture.plan.limits.max_output_chunk_bytes
        <= store.immutable_snapshot_bytes
        <= MAX_IMMUTABLE_SNAPSHOT_BYTES
    )
    assert sink.sink_id == "r1-prefix-content-addressed-output-v1"
    with pytest.raises(AttributeError):
        setattr(sink, "store", _ExplicitTestOnlyQualifiedStore())
    assert not hasattr(store, "list")
    assert not hasattr(store, "delete")


@pytest.mark.parametrize("attribute", ["store_id", "region"])
def test_admission_rejects_render_mutable_store_identity_strings(
    fixture: _Fixture,
    attribute: str,
) -> None:
    store = _ExplicitTestOnlyQualifiedStore()
    setattr(store, attribute, _DriftStr(str(getattr(store, attribute))))

    with pytest.raises(GCSPrefixCheckpointSinkError, match="identity"):
        GCSRegionalPrefixCheckpointSinkV1(store, fixture.plan)

    assert store.calls == []


def test_commit_rejects_render_mutable_digest_before_object_io(
    fixture: _Fixture,
) -> None:
    store = _ExplicitTestOnlyQualifiedStore()
    sink = GCSRegionalPrefixCheckpointSinkV1(store, fixture.plan)
    chunk = fixture.plan.chunks[0]
    raw = bytes(chunk.output_bytes)
    payload = memoryview(raw)
    try:
        with pytest.raises(GCSPrefixCheckpointSinkError, match="SHA-256"):
            sink.commit_chunk(
                fixture.plan,
                chunk,
                payload,
                output_sha256=_DriftStr(hashlib.sha256(raw).hexdigest()),
            )
    finally:
        payload.release()

    assert store.calls == []
    assert store.objects == {}


def test_chunk_payload_is_immutable_and_hashed_before_canonical_create(
    fixture: _Fixture,
) -> None:
    store = _ExplicitTestOnlyQualifiedStore()
    sink = GCSRegionalPrefixCheckpointSinkV1(store, fixture.plan)
    chunk = fixture.plan.chunks[0]
    mutable = bytearray(index % 251 for index in range(chunk.output_bytes))
    expected = bytes(mutable)
    digest = hashlib.sha256(expected).hexdigest()

    def mutate_caller(
        _store: _ExplicitTestOnlyQualifiedStore,
        stage: str,
    ) -> None:
        assert stage == "payload"
        mutable[:] = b"\xff" * len(mutable)

    store.before_create = mutate_caller
    payload = memoryview(mutable)
    try:
        receipt = sink.commit_chunk(
            fixture.plan,
            chunk,
            payload,
            output_sha256=digest,
        )
    finally:
        payload.release()

    assert store.objects[receipt.immutable_object_id][0] == expected
    assert store.objects[receipt.immutable_object_id][0] != bytes(mutable)


@pytest.mark.parametrize(
    ("stage", "boundary", "payloads_after", "bindings_after"),
    [
        ("payload", "before", 0, 0),
        ("payload", "after", 1, 0),
        ("binding", "before", 1, 0),
        ("binding", "after", 1, 1),
    ],
)
def test_chunk_crash_boundaries_resume_idempotently(
    fixture: _Fixture,
    stage: str,
    boundary: str,
    payloads_after: int,
    bindings_after: int,
) -> None:
    store = _ExplicitTestOnlyQualifiedStore()
    store.fail_once = (stage, boundary)
    sink = GCSRegionalPrefixCheckpointSinkV1(store, fixture.plan)
    with pytest.raises(GCSPrefixCheckpointSinkError):
        execute_streaming_conversion(
            fixture.plan,
            _Source(_fixture_arrays(fixture.selection)),
            sink,
        )
    assert store.count_stage("payload") == payloads_after
    assert store.count_stage("binding") == bindings_after
    assert store.count_stage("manifest") == 0
    assert store.count_stage("envelope") == 0

    resumed_sink, summary = _execute(fixture, store)
    assert summary.total_chunks == len(fixture.plan.chunks)
    assert store.count_stage("binding") == len(fixture.plan.chunks)
    second = resumed_sink.publish_final_artifacts(fixture.plan, summary)
    third = resumed_sink.publish_final_artifacts(fixture.plan, summary)
    assert second == third


@pytest.mark.parametrize(
    ("stage", "boundary", "manifests_after", "envelopes_after"),
    [
        ("manifest", "before", 0, 0),
        ("manifest", "after", 1, 0),
        ("envelope", "before", 1, 0),
        ("envelope", "after", 1, 1),
    ],
)
def test_final_publication_crash_boundaries_are_publish_last(
    fixture: _Fixture,
    stage: str,
    boundary: str,
    manifests_after: int,
    envelopes_after: int,
) -> None:
    store = _ExplicitTestOnlyQualifiedStore()
    sink, summary = _execute(fixture, store)
    store.fail_once = (stage, boundary)
    with pytest.raises(GCSPrefixCheckpointSinkError):
        sink.publish_final_artifacts(fixture.plan, summary)
    assert store.count_stage("manifest") == manifests_after
    assert store.count_stage("envelope") == envelopes_after

    publication = sink.publish_final_artifacts(fixture.plan, summary)
    repeated = sink.publish_final_artifacts(fixture.plan, summary)
    assert publication == repeated
    assert publication.production_sealable is False
    assert publication.regional_envelope.production_sealable is False


def test_finalizer_reloads_all_bindings_and_envelope_is_last(
    fixture: _Fixture,
) -> None:
    store = _ExplicitTestOnlyQualifiedStore()
    sink, summary = _execute(fixture, store)
    store.calls.clear()

    publication = sink.publish_final_artifacts(fixture.plan, summary)

    mutations = [stage for operation, stage, _ in store.calls if operation == "create"]
    assert mutations == ["manifest", "envelope"]
    manifest_index = next(
        index
        for index, call in enumerate(store.calls)
        if call[0:2] == ("create", "manifest")
    )
    assert sum(
        1
        for operation, stage, _ in store.calls[:manifest_index]
        if (operation, stage) == ("probe", "binding")
    ) == len(fixture.plan.chunks)
    assert sum(
        1
        for operation, stage, _ in store.calls[:manifest_index]
        if (operation, stage) == ("verify", "payload")
    ) == len(fixture.plan.chunks)
    assert store.calls[-1][0:2] == ("create", "envelope")
    assert publication.regional_envelope_receipt.object_id in store.objects


def test_public_binding_builder_and_lazy_reader_are_compatible(
    fixture: _Fixture,
) -> None:
    store = _ExplicitTestOnlyQualifiedStore()
    sink, summary = _execute(fixture, store)
    first_chunk = fixture.plan.chunks[0]
    loaded = sink.binding_for(fixture.plan, first_chunk)
    assert (
        build_regional_prefix_chunk_binding_v1(
            fixture.plan,
            first_chunk,
            loaded.output_receipt,
            loaded.physical_receipt,
        )
        == loaded
    )
    publication = sink.publish_final_artifacts(fixture.plan, summary)
    store.calls.clear()

    reader = LazyRegionalPrefixCheckpointReaderV1.open(
        store,
        publication.regional_envelope_receipt,
        expected_logical_manifest_sha256=(
            publication.logical_manifest.logical_checkpoint_id
        ),
    )
    assert [stage for operation, stage, _ in store.calls] == [
        "envelope",
        "manifest",
    ]
    destination = publication.logical_manifest.destinations[0]
    observed = reader.read_flat_region(
        destination.destination_key,
        0,
        int(np.prod(destination.shape)),
    )
    expected = b"".join(
        store.objects[receipt.immutable_object_id][0] for receipt in destination.chunks
    )
    assert observed.tobytes(order="C") == expected
    assert any(call[0:2] == ("read", "binding") for call in store.calls)
    assert any(call[0:2] == ("read", "payload") for call in store.calls)


@pytest.mark.parametrize("foreign_kind", ["malformed", "other-chunk"])
def test_foreign_or_squatted_binding_is_never_treated_as_absent(
    fixture: _Fixture,
    foreign_kind: str,
) -> None:
    donor = _ExplicitTestOnlyQualifiedStore()
    donor_sink, _summary = _execute(fixture, donor)
    first, second = fixture.plan.chunks[:2]
    target_id = regional_chunk_binding_object_id(fixture.plan.plan_id, first.chunk_id)
    if foreign_kind == "malformed":
        raw = b"{}"
    else:
        other_id = regional_chunk_binding_object_id(
            fixture.plan.plan_id,
            second.chunk_id,
        )
        raw = donor.objects[other_id][0]

    store = _ExplicitTestOnlyQualifiedStore()
    store.inject(target_id, raw)
    sink = GCSRegionalPrefixCheckpointSinkV1(store, fixture.plan)
    with pytest.raises(GCSPrefixCheckpointSinkError, match="foreign|malformed"):
        sink.probe_committed(fixture.plan, first)
    assert donor_sink.binding_for(fixture.plan, first).chunk_id == first.chunk_id


@pytest.mark.parametrize("mutation", ["generation", "bytes"])
def test_resume_rejects_payload_generation_or_byte_mutation(
    fixture: _Fixture,
    mutation: str,
) -> None:
    store = _ExplicitTestOnlyQualifiedStore()
    sink, _summary = _execute(fixture, store)
    chunk = fixture.plan.chunks[0]
    binding = sink.binding_for(fixture.plan, chunk)
    payload_id = binding.physical_receipt.object_id
    if mutation == "generation":
        store.replace_live_generation(payload_id)
    else:
        store.corrupt_live_bytes(payload_id)

    with pytest.raises(GCSPrefixCheckpointSinkError, match="payload"):
        sink.probe_committed(fixture.plan, chunk)


def test_store_identity_drift_fails_before_or_immediately_after_io(
    fixture: _Fixture,
) -> None:
    chunk = fixture.plan.chunks[0]
    store = _ExplicitTestOnlyQualifiedStore()
    sink = GCSRegionalPrefixCheckpointSinkV1(store, fixture.plan)
    store.region = "us-east1"
    with pytest.raises(GCSPrefixCheckpointSinkError, match="identity"):
        sink.probe_committed(fixture.plan, chunk)
    assert store.calls == []

    store = _ExplicitTestOnlyQualifiedStore()
    sink = GCSRegionalPrefixCheckpointSinkV1(store, fixture.plan)
    store.after_probe = lambda value: setattr(value, "store_id", "changed")
    with pytest.raises(GCSPrefixCheckpointSinkError, match="identity"):
        sink.probe_committed(fixture.plan, chunk)
    assert store.calls == [
        (
            "probe",
            "binding",
            regional_chunk_binding_object_id(fixture.plan.plan_id, chunk.chunk_id),
        )
    ]


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("distributed_identity_qualified", False),
        ("stream_buffer_bytes", 8),
        ("immutable_snapshot_bytes", 63 * _MIB),
        ("maximum_object_bytes", 63 * _MIB),
    ],
)
def test_admitted_store_qualification_and_bounds_are_snapshotted(
    fixture: _Fixture,
    attribute: str,
    value: Any,
) -> None:
    store = _ExplicitTestOnlyQualifiedStore()
    sink = GCSRegionalPrefixCheckpointSinkV1(store, fixture.plan)
    setattr(store, attribute, value)
    with pytest.raises(GCSPrefixCheckpointSinkError, match="changed"):
        sink.probe_committed(fixture.plan, fixture.plan.chunks[0])
    assert store.calls == []


def test_identity_drift_inside_create_is_detected_before_next_object(
    fixture: _Fixture,
) -> None:
    store = _ExplicitTestOnlyQualifiedStore()
    sink = GCSRegionalPrefixCheckpointSinkV1(store, fixture.plan)
    store.before_create = lambda value, _stage: setattr(
        value,
        "region",
        "us-east1",
    )
    with pytest.raises(GCSPrefixCheckpointSinkError, match="changed"):
        execute_streaming_conversion(
            fixture.plan,
            _Source(_fixture_arrays(fixture.selection)),
            sink,
        )
    assert store.count_stage("payload") == 1
    assert store.count_stage("binding") == 0
