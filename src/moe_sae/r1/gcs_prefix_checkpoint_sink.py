"""Generation-pinned GCS publication for the V2 R1 prefix checkpoint.

The conversion sink writes only two objects per chunk, in this order:

1. a canonical content-addressed binary payload;
2. one exact plan/chunk regional binding.

Resume discovers work only through the binding's deterministic path.  Final
publication reopens every binding, publishes the topology-neutral logical
manifest, and publishes the content-addressed regional envelope last.  This
module deliberately does not update controller CAS state and none of its
records claim production sealing.

Before the canonical payload create, the sink snapshots and SHA-256 checks
the complete conversion memoryview as exact immutable bytes.  Admission
therefore requires the plan to account for one maximum output snapshot plus
one bounded application stream slice.  This is not a bound on opaque GCS
client allocations or total process RSS.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Mapping, Protocol

from .checkpoint_conversion import (
    CheckpointConversionError,
    ConversionAttemptSummaryV1,
    ConversionChunkIdentityV1,
    ConversionOutputSink,
    OutputChunkReceiptV1,
    SINK_VERIFICATION_CONTRACT,
    StreamingConversionPlanV1,
)
from .gcs_immutable_byte_store import (
    MAX_APPLICATION_STREAM_SLICE_BYTES,
    MAX_IMMUTABLE_SNAPSHOT_BYTES,
    GCSPhysicalObjectReceiptV1,
    content_addressed_object_id,
    content_addressed_record_object_id,
)
from .prefix_checkpoint_v2 import (
    MAX_BINDING_BYTES,
    MAX_LOGICAL_MANIFEST_BYTES,
    MAX_REGIONAL_ENVELOPE_BYTES,
    PRODUCTION_PREFIX_SINK_ID,
    LogicalPrefixCheckpointManifestV2,
    PrefixCheckpointV2Error,
    RegionalPrefixCheckpointEnvelopeV1,
    RegionalPrefixChunkBindingV1,
    build_logical_prefix_checkpoint_manifest_v2,
    build_regional_prefix_checkpoint_envelope_v1,
    build_regional_prefix_chunk_binding_v1,
    regional_chunk_binding_object_id,
)
from .protocol import IDENTIFIER_PATTERN, ProtocolError, SHA256_PATTERN, require_string


class GCSPrefixCheckpointSinkError(PrefixCheckpointV2Error):
    """A qualified regional GCS sink failed closed."""


class QualifiedGCSCheckpointStore(Protocol):
    """Narrow no-list surface required by the production checkpoint sink."""

    @property
    def store_id(self) -> str: ...

    @property
    def region(self) -> str: ...

    @property
    def distributed_identity_qualified(self) -> bool: ...

    @property
    def stream_buffer_bytes(self) -> int: ...

    @property
    def immutable_snapshot_bytes(self) -> int: ...

    @property
    def maximum_object_bytes(self) -> int: ...

    def create_or_verify(
        self,
        object_id: str,
        payload: bytes,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> GCSPhysicalObjectReceiptV1: ...

    def probe_bounded(
        self,
        object_id: str,
        *,
        maximum_bytes: int,
    ) -> tuple[bytes, GCSPhysicalObjectReceiptV1] | None: ...

    def verify(
        self, receipt: GCSPhysicalObjectReceiptV1
    ) -> GCSPhysicalObjectReceiptV1: ...

    def read_exact(
        self,
        receipt: GCSPhysicalObjectReceiptV1,
        *,
        offset: int,
        length: int,
        maximum_bytes: int,
    ) -> bytes: ...


@dataclass(frozen=True, slots=True)
class RegionalPrefixCheckpointPublicationV1:
    """Exact in-memory result of a publish-last regional commit."""

    logical_manifest: LogicalPrefixCheckpointManifestV2
    logical_manifest_receipt: GCSPhysicalObjectReceiptV1
    regional_envelope: RegionalPrefixCheckpointEnvelopeV1
    regional_envelope_receipt: GCSPhysicalObjectReceiptV1
    production_sealable: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.logical_manifest) is not LogicalPrefixCheckpointManifestV2
            or type(self.logical_manifest_receipt) is not GCSPhysicalObjectReceiptV1
            or type(self.regional_envelope) is not RegionalPrefixCheckpointEnvelopeV1
            or type(self.regional_envelope_receipt) is not GCSPhysicalObjectReceiptV1
        ):
            raise ProtocolError("regional publication contains a foreign record type")
        manifest_raw = self.logical_manifest.canonical_bytes()
        envelope_raw = self.regional_envelope.canonical_bytes()
        if (
            self.logical_manifest_receipt.sha256
            != hashlib.sha256(manifest_raw).hexdigest()
            or self.logical_manifest_receipt.size_bytes != len(manifest_raw)
            or self.logical_manifest_receipt.object_id
            != content_addressed_record_object_id(self.logical_manifest_receipt.sha256)
            or self.regional_envelope_receipt.sha256
            != hashlib.sha256(envelope_raw).hexdigest()
            or self.regional_envelope_receipt.size_bytes != len(envelope_raw)
            or self.regional_envelope_receipt.object_id
            != content_addressed_record_object_id(self.regional_envelope_receipt.sha256)
            or self.regional_envelope.manifest_receipt != self.logical_manifest_receipt
            or self.regional_envelope_receipt.store_id
            != self.regional_envelope.store_id
            or self.regional_envelope_receipt.region != self.regional_envelope.region
            or self.production_sealable is not False
        ):
            raise ProtocolError("regional publication receipts are not exact")
        self.regional_envelope.validate_for_manifest(self.logical_manifest)


@dataclass(frozen=True, slots=True)
class _LoadedBinding:
    binding: RegionalPrefixChunkBindingV1
    binding_receipt: GCSPhysicalObjectReceiptV1


class GCSRegionalPrefixCheckpointSinkV1(ConversionOutputSink):
    """Qualified GCS conversion sink and publish-last V2 finalizer."""

    __slots__ = (
        "_chunks",
        "_immutable_snapshot_bytes_snapshot",
        "_maximum_object_bytes_snapshot",
        "_plan_id",
        "_region_snapshot",
        "_required_object_bytes",
        "_required_sink_staging_bytes",
        "_store",
        "_store_id_snapshot",
        "_stream_buffer_bytes_snapshot",
    )

    def __init__(
        self,
        store: QualifiedGCSCheckpointStore,
        plan: StreamingConversionPlanV1,
    ) -> None:
        if type(plan) is not StreamingConversionPlanV1:
            raise GCSPrefixCheckpointSinkError(
                "GCS prefix sink requires an exact conversion plan"
            )
        try:
            raw_store_id = store.store_id
            raw_region = store.region
            if type(raw_store_id) is not str or type(raw_region) is not str:
                raise ProtocolError(
                    "store identity fields must be exact immutable strings"
                )
            store_id = require_string(
                raw_store_id,
                "store_id",
                pattern=IDENTIFIER_PATTERN,
            )
            region = require_string(
                raw_region,
                "region",
                pattern=IDENTIFIER_PATTERN,
            )
            qualified = store.distributed_identity_qualified
            stream_buffer_bytes = store.stream_buffer_bytes
            immutable_snapshot_bytes = store.immutable_snapshot_bytes
            maximum_object_bytes = store.maximum_object_bytes
        except (AttributeError, ProtocolError) as error:
            raise GCSPrefixCheckpointSinkError(
                "GCS prefix sink store identity is malformed"
            ) from error
        required_methods = (
            "create_or_verify",
            "probe_bounded",
            "verify",
            "read_exact",
        )
        if any(not callable(getattr(store, name, None)) for name in required_methods):
            raise GCSPrefixCheckpointSinkError(
                "GCS prefix sink store lacks its exact no-list I/O surface"
            )
        if qualified is not True:
            raise GCSPrefixCheckpointSinkError(
                "GCS prefix sink requires a distributed-qualified store identity"
            )
        if (
            type(stream_buffer_bytes) is not int
            or not 1 <= stream_buffer_bytes <= MAX_APPLICATION_STREAM_SLICE_BYTES
        ):
            raise GCSPrefixCheckpointSinkError(
                "GCS application stream-slice bound is invalid"
            )
        if (
            type(immutable_snapshot_bytes) is not int
            or not plan.limits.max_output_chunk_bytes
            <= immutable_snapshot_bytes
            <= MAX_IMMUTABLE_SNAPSHOT_BYTES
        ):
            raise GCSPrefixCheckpointSinkError(
                "GCS immutable-snapshot bound cannot hold a maximum output chunk"
            )
        required_sink_staging_bytes = (
            plan.limits.max_output_chunk_bytes + stream_buffer_bytes
        )
        if plan.limits.sink_staging_buffer_bytes < required_sink_staging_bytes:
            raise GCSPrefixCheckpointSinkError(
                "plan sink staging cannot hold the immutable payload snapshot "
                "and application stream slice"
            )
        required_object_bytes = max(
            plan.limits.max_output_chunk_bytes,
            MAX_LOGICAL_MANIFEST_BYTES,
            MAX_REGIONAL_ENVELOPE_BYTES,
        )
        if (
            type(maximum_object_bytes) is not int
            or maximum_object_bytes < required_object_bytes
            or maximum_object_bytes > MAX_IMMUTABLE_SNAPSHOT_BYTES
            or immutable_snapshot_bytes > maximum_object_bytes
        ):
            raise GCSPrefixCheckpointSinkError(
                "GCS object bound cannot hold all payload and control records"
            )
        chunks = {chunk.chunk_id: chunk for chunk in plan.chunks}
        if len(chunks) != len(plan.chunks):
            raise GCSPrefixCheckpointSinkError("conversion plan chunk IDs collide")
        self._store = store
        self._plan_id = plan.plan_id
        self._store_id_snapshot = store_id
        self._region_snapshot = region
        self._stream_buffer_bytes_snapshot = stream_buffer_bytes
        self._immutable_snapshot_bytes_snapshot = immutable_snapshot_bytes
        self._maximum_object_bytes_snapshot = maximum_object_bytes
        self._required_object_bytes = required_object_bytes
        self._required_sink_staging_bytes = required_sink_staging_bytes
        self._chunks: Mapping[str, ConversionChunkIdentityV1] = MappingProxyType(chunks)

    @property
    def store(self) -> QualifiedGCSCheckpointStore:
        return self._store

    @property
    def sink_id(self) -> str:
        self._require_store_state()
        return PRODUCTION_PREFIX_SINK_ID

    def _require_store_state(self) -> None:
        try:
            current = (
                self.store.store_id,
                self.store.region,
                self.store.distributed_identity_qualified,
                self.store.stream_buffer_bytes,
                self.store.immutable_snapshot_bytes,
                self.store.maximum_object_bytes,
            )
        except AttributeError as error:
            raise GCSPrefixCheckpointSinkError(
                "qualified GCS store surface changed"
            ) from error
        expected = (
            self._store_id_snapshot,
            self._region_snapshot,
            True,
            self._stream_buffer_bytes_snapshot,
            self._immutable_snapshot_bytes_snapshot,
            self._maximum_object_bytes_snapshot,
        )
        if (
            current != expected
            or type(current[0]) is not str
            or type(current[1]) is not str
            or current[2] is not True
            or type(current[3]) is not int
            or not 1 <= current[3] <= MAX_APPLICATION_STREAM_SLICE_BYTES
            or type(current[4]) is not int
            or not self._immutable_snapshot_bytes_snapshot
            <= current[4]
            <= MAX_IMMUTABLE_SNAPSHOT_BYTES
            or type(current[5]) is not int
            or not self._required_object_bytes
            <= current[5]
            <= MAX_IMMUTABLE_SNAPSHOT_BYTES
            or current[4] > current[5]
        ):
            raise GCSPrefixCheckpointSinkError(
                "qualified GCS store identity or admitted bounds changed"
            )

    def _validate_plan(
        self,
        plan: StreamingConversionPlanV1,
    ) -> None:
        self._require_store_state()
        if (
            type(plan) is not StreamingConversionPlanV1
            or plan.plan_id != self._plan_id
            or plan.limits.sink_staging_buffer_bytes < self._required_sink_staging_bytes
            or plan.limits.max_output_chunk_bytes
            > self._immutable_snapshot_bytes_snapshot
            or max(
                plan.limits.max_output_chunk_bytes,
                MAX_LOGICAL_MANIFEST_BYTES,
                MAX_REGIONAL_ENVELOPE_BYTES,
            )
            > self._maximum_object_bytes_snapshot
        ):
            raise GCSPrefixCheckpointSinkError(
                "GCS prefix sink received another conversion plan"
            )

    def _validate_chunk(
        self,
        plan: StreamingConversionPlanV1,
        chunk: ConversionChunkIdentityV1,
    ) -> None:
        self._validate_plan(plan)
        if (
            type(chunk) is not ConversionChunkIdentityV1
            or self._chunks.get(chunk.chunk_id) != chunk
        ):
            raise GCSPrefixCheckpointSinkError(
                "GCS prefix sink received a foreign conversion chunk"
            )

    def _require_store_receipt(
        self,
        receipt: GCSPhysicalObjectReceiptV1,
        *,
        object_id: str,
        content_type: str,
        maximum_bytes: int,
    ) -> None:
        if (
            type(receipt) is not GCSPhysicalObjectReceiptV1
            or receipt.store_id != self._store_id_snapshot
            or receipt.region != self._region_snapshot
            or receipt.object_id != object_id
            or receipt.content_type != content_type
            or receipt.size_bytes > maximum_bytes
            or receipt.size_bytes > self._maximum_object_bytes_snapshot
        ):
            raise GCSPrefixCheckpointSinkError(
                "GCS physical receipt differs from the admitted exact store"
            )

    def _load_binding(
        self,
        plan: StreamingConversionPlanV1,
        chunk: ConversionChunkIdentityV1,
    ) -> _LoadedBinding | None:
        self._validate_chunk(plan, chunk)
        binding_id = regional_chunk_binding_object_id(plan.plan_id, chunk.chunk_id)
        try:
            probed = self.store.probe_bounded(
                binding_id,
                maximum_bytes=MAX_BINDING_BYTES,
            )
        except CheckpointConversionError as error:
            raise GCSPrefixCheckpointSinkError(
                f"cannot probe exact GCS chunk binding: {chunk.chunk_id}"
            ) from error
        self._require_store_state()
        if probed is None:
            return None
        if type(probed) is not tuple or len(probed) != 2:
            raise GCSPrefixCheckpointSinkError(
                "GCS binding probe returned an invalid result"
            )
        raw, binding_receipt = probed
        if type(raw) is not bytes:
            raise GCSPrefixCheckpointSinkError("GCS binding probe returned non-bytes")
        self._require_store_receipt(
            binding_receipt,
            object_id=binding_id,
            content_type="application/json",
            maximum_bytes=MAX_BINDING_BYTES,
        )
        if (
            len(raw) != binding_receipt.size_bytes
            or hashlib.sha256(raw).hexdigest() != binding_receipt.sha256
        ):
            raise GCSPrefixCheckpointSinkError(
                "GCS binding bytes differ from their exact receipt"
            )
        try:
            binding = RegionalPrefixChunkBindingV1.from_json(raw)
            binding.validate_for(plan, chunk)
        except (CheckpointConversionError, ProtocolError) as error:
            raise GCSPrefixCheckpointSinkError(
                f"stored GCS binding is foreign or malformed: {chunk.chunk_id}"
            ) from error
        self._require_store_receipt(
            binding.physical_receipt,
            object_id=content_addressed_object_id(binding.output_receipt.output_sha256),
            content_type="application/octet-stream",
            maximum_bytes=plan.limits.max_output_chunk_bytes,
        )
        self._require_store_state()
        try:
            verified_payload = self.store.verify(binding.physical_receipt)
        except CheckpointConversionError as error:
            raise GCSPrefixCheckpointSinkError(
                f"stored GCS payload is not exact: {chunk.chunk_id}"
            ) from error
        self._require_store_state()
        if verified_payload != binding.physical_receipt:
            raise GCSPrefixCheckpointSinkError(
                "GCS payload verification returned another generation"
            )
        return _LoadedBinding(
            binding=binding,
            binding_receipt=binding_receipt,
        )

    def probe_committed(
        self,
        plan: StreamingConversionPlanV1,
        chunk: ConversionChunkIdentityV1,
    ) -> OutputChunkReceiptV1 | None:
        loaded = self._load_binding(plan, chunk)
        return None if loaded is None else loaded.binding.output_receipt

    def commit_chunk(
        self,
        plan: StreamingConversionPlanV1,
        chunk: ConversionChunkIdentityV1,
        payload: memoryview,
        *,
        output_sha256: str,
    ) -> OutputChunkReceiptV1:
        self._validate_chunk(plan, chunk)
        if type(payload) is not memoryview:
            raise GCSPrefixCheckpointSinkError(
                "GCS prefix sink requires an exact memoryview payload"
            )
        if not payload.c_contiguous or payload.nbytes != chunk.output_bytes:
            raise GCSPrefixCheckpointSinkError(
                "GCS prefix payload is non-contiguous or has the wrong byte count"
            )
        try:
            if type(output_sha256) is not str:
                raise ProtocolError("output_sha256 must be an exact immutable string")
            digest = require_string(
                output_sha256,
                "output_sha256",
                pattern=SHA256_PATTERN,
            )
        except ProtocolError as error:
            raise GCSPrefixCheckpointSinkError(
                "GCS prefix output SHA-256 is malformed"
            ) from error
        existing = self._load_binding(plan, chunk)
        if existing is not None:
            try:
                existing.binding.output_receipt.validate_for(
                    plan,
                    chunk,
                    sink_id=PRODUCTION_PREFIX_SINK_ID,
                    expected_output_sha256=digest,
                )
            except CheckpointConversionError as error:
                raise GCSPrefixCheckpointSinkError(
                    "existing GCS binding conflicts with conversion output"
                ) from error
            return existing.binding.output_receipt

        payload_id = content_addressed_object_id(digest)
        try:
            immutable_payload = payload.cast("B").tobytes()
        except (TypeError, ValueError) as error:
            raise GCSPrefixCheckpointSinkError(
                "cannot snapshot exact GCS prefix payload bytes"
            ) from error
        if (
            len(immutable_payload) != chunk.output_bytes
            or hashlib.sha256(immutable_payload).hexdigest() != digest
        ):
            raise GCSPrefixCheckpointSinkError(
                "immutable GCS prefix payload snapshot differs from conversion output"
            )
        self._require_store_state()
        try:
            physical = self.store.create_or_verify(
                payload_id,
                immutable_payload,
                expected_sha256=digest,
                expected_size_bytes=chunk.output_bytes,
            )
        except CheckpointConversionError as error:
            raise GCSPrefixCheckpointSinkError(
                f"cannot publish GCS payload: {chunk.chunk_id}"
            ) from error
        self._require_store_state()
        self._require_store_receipt(
            physical,
            object_id=payload_id,
            content_type="application/octet-stream",
            maximum_bytes=plan.limits.max_output_chunk_bytes,
        )
        if physical.sha256 != digest or physical.size_bytes != chunk.output_bytes:
            raise GCSPrefixCheckpointSinkError(
                "published GCS payload receipt differs from conversion output"
            )
        # The application-owned object-sized snapshot must not overlap binding
        # construction or the exact payload remeasurement in _load_binding.
        del immutable_payload
        output = OutputChunkReceiptV1(
            plan_id=plan.plan_id,
            chunk_id=chunk.chunk_id,
            sink_id=PRODUCTION_PREFIX_SINK_ID,
            destination_key=chunk.mapping.destination_key,
            destination_flat_start=chunk.destination_flat_start,
            destination_flat_stop=chunk.destination_flat_stop,
            dtype=chunk.mapping.destination_dtype,
            output_sha256=physical.sha256,
            output_size_bytes=physical.size_bytes,
            immutable_object_id=physical.object_id,
            verification_contract=SINK_VERIFICATION_CONTRACT,
        )
        binding = build_regional_prefix_chunk_binding_v1(
            plan,
            chunk,
            output,
            physical,
        )
        binding_raw = binding.canonical_bytes()
        binding_id = regional_chunk_binding_object_id(plan.plan_id, chunk.chunk_id)
        self._require_store_state()
        try:
            binding_receipt = self.store.create_or_verify(
                binding_id,
                binding_raw,
                expected_sha256=hashlib.sha256(binding_raw).hexdigest(),
                expected_size_bytes=len(binding_raw),
            )
        except CheckpointConversionError as error:
            raise GCSPrefixCheckpointSinkError(
                f"cannot publish GCS binding: {chunk.chunk_id}"
            ) from error
        self._require_store_state()
        self._require_store_receipt(
            binding_receipt,
            object_id=binding_id,
            content_type="application/json",
            maximum_bytes=MAX_BINDING_BYTES,
        )
        loaded = self._load_binding(plan, chunk)
        if (
            loaded is None
            or loaded.binding != binding
            or loaded.binding_receipt != binding_receipt
        ):
            raise GCSPrefixCheckpointSinkError(
                "concurrent GCS binding publication did not converge exactly"
            )
        return loaded.binding.output_receipt

    def binding_for(
        self,
        plan: StreamingConversionPlanV1,
        chunk: ConversionChunkIdentityV1,
    ) -> RegionalPrefixChunkBindingV1:
        loaded = self._load_binding(plan, chunk)
        if loaded is None:
            raise GCSPrefixCheckpointSinkError(
                f"missing exact GCS binding: {chunk.chunk_id}"
            )
        return loaded.binding

    def publish_final_artifacts(
        self,
        plan: StreamingConversionPlanV1,
        summary: ConversionAttemptSummaryV1,
    ) -> RegionalPrefixCheckpointPublicationV1:
        """Reload all bindings, then publish manifest and envelope in order."""

        self._validate_plan(plan)
        if type(summary) is not ConversionAttemptSummaryV1:
            raise GCSPrefixCheckpointSinkError(
                "GCS finalization requires an exact conversion summary"
            )
        summary_by_id = {receipt.chunk_id: receipt for receipt in summary.receipts}
        if (
            summary.plan_id != plan.plan_id
            or summary.sink_id != PRODUCTION_PREFIX_SINK_ID
            or len(summary_by_id) != len(summary.receipts)
            or set(summary_by_id) != set(self._chunks)
        ):
            raise GCSPrefixCheckpointSinkError(
                "GCS finalization summary differs from the exact plan"
            )

        bindings: list[RegionalPrefixChunkBindingV1] = []
        binding_receipts: list[GCSPhysicalObjectReceiptV1] = []
        for chunk in plan.chunks:
            loaded = self._load_binding(plan, chunk)
            if loaded is None:
                raise GCSPrefixCheckpointSinkError(
                    f"GCS finalization found an absent chunk: {chunk.chunk_id}"
                )
            if loaded.binding.output_receipt != summary_by_id[chunk.chunk_id]:
                raise GCSPrefixCheckpointSinkError(
                    "GCS finalization summary substitutes a reloaded binding"
                )
            bindings.append(loaded.binding)
            binding_receipts.append(loaded.binding_receipt)

        try:
            manifest = build_logical_prefix_checkpoint_manifest_v2(plan, summary)
        except (CheckpointConversionError, ProtocolError) as error:
            raise GCSPrefixCheckpointSinkError(
                "cannot build logical prefix manifest from exact bindings"
            ) from error
        manifest_raw = manifest.canonical_bytes()
        manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
        manifest_id = content_addressed_record_object_id(manifest_sha256)
        self._require_store_state()
        try:
            manifest_receipt = self.store.create_or_verify(
                manifest_id,
                manifest_raw,
                expected_sha256=manifest_sha256,
                expected_size_bytes=len(manifest_raw),
            )
        except CheckpointConversionError as error:
            raise GCSPrefixCheckpointSinkError(
                "cannot publish topology-neutral logical manifest"
            ) from error
        self._require_store_state()
        self._require_store_receipt(
            manifest_receipt,
            object_id=manifest_id,
            content_type="application/json",
            maximum_bytes=MAX_LOGICAL_MANIFEST_BYTES,
        )

        try:
            envelope = build_regional_prefix_checkpoint_envelope_v1(
                plan,
                manifest,
                manifest_receipt,
                tuple(bindings),
                tuple(binding_receipts),
            )
        except (CheckpointConversionError, ProtocolError) as error:
            raise GCSPrefixCheckpointSinkError(
                "cannot build exact regional prefix envelope"
            ) from error
        envelope_raw = envelope.canonical_bytes()
        envelope_sha256 = hashlib.sha256(envelope_raw).hexdigest()
        envelope_id = content_addressed_record_object_id(envelope_sha256)
        self._require_store_state()
        try:
            envelope_receipt = self.store.create_or_verify(
                envelope_id,
                envelope_raw,
                expected_sha256=envelope_sha256,
                expected_size_bytes=len(envelope_raw),
            )
        except CheckpointConversionError as error:
            raise GCSPrefixCheckpointSinkError(
                "cannot publish regional prefix envelope"
            ) from error
        self._require_store_state()
        self._require_store_receipt(
            envelope_receipt,
            object_id=envelope_id,
            content_type="application/json",
            maximum_bytes=MAX_REGIONAL_ENVELOPE_BYTES,
        )
        return RegionalPrefixCheckpointPublicationV1(
            logical_manifest=manifest,
            logical_manifest_receipt=manifest_receipt,
            regional_envelope=envelope,
            regional_envelope_receipt=envelope_receipt,
        )


__all__ = [
    "GCSPrefixCheckpointSinkError",
    "GCSRegionalPrefixCheckpointSinkV1",
    "QualifiedGCSCheckpointStore",
    "RegionalPrefixCheckpointPublicationV1",
]
