"""Topology-neutral R1 prefix manifests and generation-pinned GCS envelopes.

This additive protocol deliberately leaves :mod:`moe_sae.r1.prefix_checkpoint`
unchanged.  The V2 logical manifest commits to converted tensor content and
plan lineage only.  A separate regional envelope is the sole commit point for
GCS locations, generations, CRC32C values, and store identity.

Opening a regional checkpoint is intentionally lazy: callers must supply the
exact physical receipt for a content-addressed envelope and the independently
expected logical-manifest SHA-256.  Opening performs exactly two bounded JSON
reads (envelope, then its embedded exact manifest generation), performs no
listing or per-chunk HEAD/verification, and only reads payload objects touched
by a later flat-region request.

These records cannot seal a production checkpoint.  A later qualification
record must join fresh-process restore, framework interoperability, independent
numeric-oracle, parity, and measured resource evidence without rewriting this
logical identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, Self

import ml_dtypes
import numpy as np

from .checkpoint_conversion import (
    CHUNKING_CONTRACT,
    CONVERSION_ENGINE_CONTRACT,
    DEQUANTIZATION_CONTRACT,
    SINK_VERIFICATION_CONTRACT,
    CheckpointConversionError,
    ConversionAttemptSummaryV1,
    ConversionChunkIdentityV1,
    OutputChunkReceiptV1,
    R1RuntimeLockIdentityV1,
    SourceAdapterIdentityV1,
    StreamingConversionPlanV1,
)
from .gcs_immutable_byte_store import (
    GCSPhysicalObjectReceiptV1,
    content_addressed_object_id,
    content_addressed_record_object_id,
)
from .checkpoint_inventory import HF_REVISION, MAXTEXT_REVISION
from .prefix_checkpoint import (
    PlanSourceIdentityBindingV1,
    SOURCE_LINEAGE_CONTRACT,
)
from .protocol import (
    CanonicalRecord,
    IDENTIFIER_PATTERN,
    ProtocolError,
    SHA256_PATTERN,
    canonical_json_bytes,
    canonical_sha256,
    parse_json,
    require_integer,
    require_record_object,
    require_string,
)


PRODUCTION_PREFIX_SINK_ID = "r1-prefix-content-addressed-output-v1"
PRODUCTION_PREFIX_SINK_CONTRACT = (
    "topology-neutral-content-addressed-conversion-output-v1"
)
LOGICAL_IDENTITY_SCOPE = "topology-neutral-converted-prefix-content-v2"
LOGICAL_TENSOR_TREE_CONTRACT = "ordered-flat-interval-sha256-content-commitment-tree-v1"
OBJECT_ADDRESSING_CONTRACT = "sha256-content-addressed-relative-object-path-v1"
OBJECT_DISCOVERY_CONTRACT = "exact-receipt-paths-no-listing-no-latest-v1"
REGIONAL_BINDING_CONTRACT = "logical-output-to-generation-pinned-regional-object-v1"
REGIONAL_ENVELOPE_CONTRACT = "content-addressed-regional-envelope-published-last-v1"
REGIONAL_COPY_CONTRACT = (
    "byte-identical-cross-region-copy-with-new-physical-receipts-v1"
)

TARGET_SEMANTICS = "attention-router-and-routed-experts-only-before-shared-expert"

EXTERNAL_QUALIFICATION_REQUIRED = (
    "complete-source-observation-snapshot-evidence",
    "fresh-process-generation-pinned-restore",
    "orbax-or-tensorstore-framework-interoperability",
    "independent-r1-fp8-and-bf16-conversion-oracle",
    "full-model-r1-logit-parity",
    "prefix-versus-full-routed-aggregate-parity",
    "real-payload-throughput-and-peak-rss-qualification",
)

MAX_BINDING_BYTES = 16 * 1024
MAX_LOGICAL_MANIFEST_BYTES = 32 * 1024**2
MAX_REGIONAL_ENVELOPE_BYTES = 32 * 1024**2
MAX_PAYLOAD_READ_BYTES = 64 * 1024**2
MAX_PHYSICAL_PAYLOAD_BYTES = 64 * 1024**2
MAX_CHUNKS = 22_108
MAX_SOURCE_LOGICAL_TENSORS = 21_930
MAX_DESTINATIONS = 510
MAX_RANK = 8
MAX_CONVERTED_PARAMETERS = 324_837_104_640
MAX_CONTROL_STRING_BYTES = 2_048

PINNED_SOURCE_SHARDS = 78
PINNED_SOURCE_ENTRIES = 43_679
PINNED_SOURCE_LOGICAL_TENSORS = 21_930
PINNED_DESTINATION_TENSORS = 510
PINNED_CONVERTED_PARAMETERS = 324_837_104_640
PINNED_TOTAL_CHUNKS = 22_108
PINNED_CONVERTED_PAYLOAD_BYTES = 649_674_223_616
PINNED_TARGET_LAYER_ONE_INDEXED = 31

_DTYPES = {
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
    "float32": np.dtype("<f4"),
}
_DTYPE_BYTES = {name: dtype.itemsize for name, dtype in _DTYPES.items()}


class PrefixCheckpointV2Error(CheckpointConversionError):
    """A V2 logical or regional prefix-checkpoint invariant failed."""


def _raw_json_bytes(value: str | bytes, *, maximum_bytes: int, name: str) -> bytes:
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ProtocolError(f"{name} maximum_bytes must be a positive integer")
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        # Every admitted protocol field is ASCII.  Checking this before encode
        # keeps the byte bound identical to the already measured character bound.
        if len(value) > maximum_bytes or not value.isascii():
            raise ProtocolError(f"{name} exceeds its canonical ASCII byte bound")
        raw = value.encode("ascii")
    else:
        raise ProtocolError(f"{name} JSON input must be exact str or bytes")
    if len(raw) > maximum_bytes:
        raise ProtocolError(f"{name} exceeds its JSON byte bound")
    return raw


def _bounded_parse_record(
    value: str | bytes, *, maximum_bytes: int, name: str
) -> tuple[bytes, Mapping[str, Any]]:
    raw = _raw_json_bytes(value, maximum_bytes=maximum_bytes, name=name)
    try:
        parsed = parse_json(raw)
    except (ProtocolError, ValueError) as error:
        raise ProtocolError(f"{name} is not strict JSON") from error
    if not isinstance(parsed, Mapping):
        raise ProtocolError(f"{name} must be a JSON object")
    return raw, parsed


def _require_canonical_wire(raw: bytes, record: CanonicalRecord, name: str) -> None:
    if raw != record.canonical_bytes():
        raise ProtocolError(f"{name} is not the canonical JSON encoding")


def _bounded_string(
    value: Any,
    name: str,
    *,
    maximum_bytes: int = MAX_CONTROL_STRING_BYTES,
    pattern: Any | None = None,
) -> str:
    result = require_string(value, name, pattern=pattern)
    if not result.isascii() or len(result) > maximum_bytes:
        raise ProtocolError(f"{name} exceeds its canonical ASCII byte bound")
    return result


def _identifier(value: Any, name: str) -> str:
    return _bounded_string(value, name, maximum_bytes=512, pattern=IDENTIFIER_PATTERN)


def _sha256(value: Any, name: str) -> str:
    return _bounded_string(value, name, maximum_bytes=64, pattern=SHA256_PATTERN)


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ProtocolError(f"{name} must be a JSON boolean")
    return value


def _record_array(
    value: Any, name: str, *, maximum_items: int, nonempty: bool = True
) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ProtocolError(f"{name} must be a JSON array")
    if (nonempty and not value) or len(value) > maximum_items:
        qualifier = "non-empty and " if nonempty else ""
        raise ProtocolError(
            f"{name} must be {qualifier}bounded to {maximum_items} items"
        )
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ProtocolError(f"{name}[{index}] must be an object")
    return value


def _string_array(value: Any, name: str, *, maximum_items: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise ProtocolError(f"{name} must be an array bounded to {maximum_items}")
    return tuple(
        _identifier(item, f"{name}[{index}]") for index, item in enumerate(value)
    )


def _shape(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or len(value) > MAX_RANK:
        raise ProtocolError(f"{name} must have between 1 and {MAX_RANK} dimensions")
    result: list[int] = []
    numel = 1
    for index, item in enumerate(value):
        dimension = require_integer(
            item,
            f"{name}[{index}]",
            minimum=1,
            maximum=MAX_CONVERTED_PARAMETERS,
        )
        if numel > MAX_CONVERTED_PARAMETERS // dimension:
            raise ProtocolError(f"{name} exceeds the parameter-count bound")
        numel *= dimension
        result.append(dimension)
    return tuple(result)


def _source_adapter_from_dict(value: Any) -> SourceAdapterIdentityV1:
    if not isinstance(value, Mapping) or set(value) != {
        "adapter_id",
        "implementation_sha256",
    }:
        raise ProtocolError("source_adapter_identity has invalid fields")
    return SourceAdapterIdentityV1(
        adapter_id=_identifier(value["adapter_id"], "source adapter_id"),
        implementation_sha256=_sha256(
            value["implementation_sha256"], "source implementation_sha256"
        ),
    )


def _runtime_from_dict(value: Any) -> R1RuntimeLockIdentityV1 | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "maxtext_revision",
        "pyproject_sha256",
        "uv_lock_sha256",
    }:
        raise ProtocolError("runtime_lock_identity has invalid fields")
    return R1RuntimeLockIdentityV1(
        maxtext_revision=_identifier(value["maxtext_revision"], "maxtext_revision"),
        pyproject_sha256=_sha256(value["pyproject_sha256"], "pyproject_sha256"),
        uv_lock_sha256=_sha256(value["uv_lock_sha256"], "uv_lock_sha256"),
    )


def _runtime_dict(
    value: R1RuntimeLockIdentityV1 | None,
) -> dict[str, str] | None:
    return None if value is None else value.as_dict()


def _output_receipt_from_dict(value: Any) -> OutputChunkReceiptV1:
    fields = {
        "plan_id",
        "chunk_id",
        "sink_id",
        "destination_key",
        "destination_flat_start",
        "destination_flat_stop",
        "dtype",
        "output_sha256",
        "output_size_bytes",
        "immutable_object_id",
        "verification_contract",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ProtocolError("output_receipt has invalid fields")
    return OutputChunkReceiptV1(
        plan_id=_sha256(value["plan_id"], "output plan_id"),
        chunk_id=_sha256(value["chunk_id"], "output chunk_id"),
        sink_id=_identifier(value["sink_id"], "output sink_id"),
        destination_key=_identifier(value["destination_key"], "output destination_key"),
        destination_flat_start=require_integer(
            value["destination_flat_start"],
            "output destination_flat_start",
            minimum=0,
            maximum=MAX_CONVERTED_PARAMETERS - 1,
        ),
        destination_flat_stop=require_integer(
            value["destination_flat_stop"],
            "output destination_flat_stop",
            minimum=1,
            maximum=MAX_CONVERTED_PARAMETERS,
        ),
        dtype=_identifier(value["dtype"], "output dtype"),
        output_sha256=_sha256(value["output_sha256"], "output SHA-256"),
        output_size_bytes=require_integer(
            value["output_size_bytes"],
            "output_size_bytes",
            minimum=1,
            maximum=MAX_PHYSICAL_PAYLOAD_BYTES,
        ),
        immutable_object_id=_bounded_string(
            value["immutable_object_id"],
            "immutable_object_id",
            maximum_bytes=1024,
        ),
        verification_contract=_identifier(
            value["verification_contract"], "verification_contract"
        ),
    )


def _validate_physical_bounds(
    receipt: GCSPhysicalObjectReceiptV1,
    *,
    maximum_size_bytes: int = MAX_PHYSICAL_PAYLOAD_BYTES,
) -> None:
    if type(receipt) is not GCSPhysicalObjectReceiptV1:
        raise ProtocolError("physical receipt has the wrong type")
    _bounded_string(receipt.store_id, "physical store_id", maximum_bytes=512)
    _bounded_string(receipt.object_id, "physical object_id", maximum_bytes=1024)
    _bounded_string(receipt.uri, "physical uri", maximum_bytes=2_048)
    _bounded_string(receipt.region, "physical region", maximum_bytes=64)
    require_integer(
        receipt.size_bytes,
        "physical size_bytes",
        minimum=1,
        maximum=maximum_size_bytes,
    )


def _physical_receipt_from_dict(
    value: Any,
    *,
    maximum_size_bytes: int = MAX_PHYSICAL_PAYLOAD_BYTES,
) -> GCSPhysicalObjectReceiptV1:
    if not isinstance(value, Mapping):
        raise ProtocolError("physical receipt must be an object")
    # Bound the parsed integer before constructing a receipt or using it in
    # arithmetic.  JSON parsing itself is already raw-byte bounded.
    require_integer(
        value.get("size_bytes"),
        "physical size_bytes",
        minimum=1,
        maximum=maximum_size_bytes,
    )
    receipt = GCSPhysicalObjectReceiptV1.from_dict(value)
    _validate_physical_bounds(receipt, maximum_size_bytes=maximum_size_bytes)
    return receipt


def _source_lineage_from_dict(value: Any) -> PlanSourceIdentityBindingV1:
    if not isinstance(value, Mapping):
        raise ProtocolError("source lineage item must be an object")
    require_integer(
        value.get("size_bytes"),
        "source lineage size_bytes",
        minimum=1,
        maximum=1 << 40,
    )
    result = PlanSourceIdentityBindingV1.from_dict(value)
    _identifier(result.role, "source lineage role")
    _identifier(result.shard, "source lineage shard")
    _validate_source_adapter_bounds(result.source_adapter_identity)
    return result


def _validate_source_adapter_bounds(value: SourceAdapterIdentityV1) -> None:
    if type(value) is not SourceAdapterIdentityV1:
        raise ProtocolError("source adapter identity has the wrong type")
    _identifier(value.adapter_id, "source adapter_id")
    _sha256(value.implementation_sha256, "source implementation_sha256")


def _lineage_for_chunk(
    plan: StreamingConversionPlanV1,
    chunk: ConversionChunkIdentityV1,
) -> tuple[PlanSourceIdentityBindingV1, ...]:
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


def _validate_output_receipt_profile(receipt: OutputChunkReceiptV1) -> None:
    if type(receipt) is not OutputChunkReceiptV1:
        raise ProtocolError("logical output receipt has the wrong type")
    _sha256(receipt.plan_id, "logical output plan_id")
    _sha256(receipt.chunk_id, "logical output chunk_id")
    _identifier(receipt.sink_id, "logical output sink_id")
    _identifier(receipt.destination_key, "logical output destination_key")
    require_integer(
        receipt.destination_flat_start,
        "logical output destination_flat_start",
        minimum=0,
        maximum=MAX_CONVERTED_PARAMETERS - 1,
    )
    require_integer(
        receipt.destination_flat_stop,
        "logical output destination_flat_stop",
        minimum=1,
        maximum=MAX_CONVERTED_PARAMETERS,
    )
    _identifier(receipt.dtype, "logical output dtype")
    _sha256(receipt.output_sha256, "logical output SHA-256")
    require_integer(
        receipt.output_size_bytes,
        "logical output size_bytes",
        minimum=1,
        maximum=MAX_PHYSICAL_PAYLOAD_BYTES,
    )
    _bounded_string(
        receipt.immutable_object_id,
        "logical output immutable_object_id",
        maximum_bytes=1024,
    )
    _identifier(receipt.verification_contract, "logical output verification_contract")
    if receipt.sink_id != PRODUCTION_PREFIX_SINK_ID:
        raise ProtocolError("logical output receipt has a topology-dependent sink")
    if receipt.verification_contract != SINK_VERIFICATION_CONTRACT:
        raise ProtocolError("logical output receipt verification contract changed")
    expected_object_id = content_addressed_object_id(receipt.output_sha256)
    if receipt.immutable_object_id != expected_object_id:
        raise ProtocolError("logical output receipt is not content addressed")
    itemsize = _DTYPE_BYTES.get(receipt.dtype)
    if itemsize is None:
        raise ProtocolError("logical output receipt has an unsupported dtype")
    expected_size = (
        receipt.destination_flat_stop - receipt.destination_flat_start
    ) * itemsize
    if receipt.output_size_bytes != expected_size:
        raise ProtocolError("logical output receipt byte count is inconsistent")
    if receipt.output_size_bytes > MAX_PHYSICAL_PAYLOAD_BYTES:
        raise ProtocolError("logical output receipt exceeds the payload-object bound")


def regional_chunk_binding_object_id(plan_id: str, chunk_id: str) -> str:
    """Return the exact no-list path for one plan-derived regional binding."""

    plan = _sha256(plan_id, "plan_id")
    chunk = _sha256(chunk_id, "chunk_id")
    return f"plans/{plan}/bindings/{chunk}.json"


def _logical_tensor_tree_sha256(
    *,
    destination_key: str,
    dtype: str,
    shape: tuple[int, ...],
    chunks: tuple[OutputChunkReceiptV1, ...],
) -> str:
    """Commit to ordered chunk content; this is not a flat tensor byte hash."""

    return canonical_sha256(
        {
            "schema": LOGICAL_TENSOR_TREE_CONTRACT,
            "destination_key": destination_key,
            "dtype": dtype,
            "shape": list(shape),
            "ordered_leaves": [
                {
                    "chunk_id": receipt.chunk_id,
                    "flat_start": receipt.destination_flat_start,
                    "flat_stop": receipt.destination_flat_stop,
                    "output_sha256": receipt.output_sha256,
                    "output_size_bytes": receipt.output_size_bytes,
                }
                for receipt in chunks
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class LogicalStoredDestinationV2(CanonicalRecord):
    """One logical tensor committed as ordered, content-addressed leaves."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.logical-stored-destination.v2"

    destination_key: str
    dtype: str
    shape: tuple[int, ...]
    payload_bytes: int
    chunks: tuple[OutputChunkReceiptV1, ...]
    logical_tensor_tree_sha256: str
    logical_tensor_tree_contract: str = LOGICAL_TENSOR_TREE_CONTRACT

    def __post_init__(self) -> None:
        _identifier(self.destination_key, "destination_key")
        if self.dtype not in _DTYPES:
            raise ProtocolError("logical destination dtype is unsupported")
        if (
            type(self.shape) is not tuple
            or not self.shape
            or len(self.shape) > MAX_RANK
            or any(type(item) is not int or item <= 0 for item in self.shape)
        ):
            raise ProtocolError("logical destination shape is invalid")
        numel = 1
        for dimension in self.shape:
            if dimension > MAX_CONVERTED_PARAMETERS // numel:
                raise ProtocolError("logical destination exceeds the parameter bound")
            numel *= dimension
        require_integer(
            self.payload_bytes,
            "payload_bytes",
            minimum=1,
            maximum=PINNED_CONVERTED_PAYLOAD_BYTES,
        )
        if self.payload_bytes != numel * _DTYPE_BYTES[self.dtype]:
            raise ProtocolError("logical destination byte total is inconsistent")
        if (
            type(self.chunks) is not tuple
            or not self.chunks
            or len(self.chunks) > MAX_CHUNKS
        ):
            raise ProtocolError("logical destination chunks are invalid or unbounded")
        cursor = 0
        ids: set[str] = set()
        for receipt in self.chunks:
            _validate_output_receipt_profile(receipt)
            if (
                receipt.destination_key != self.destination_key
                or receipt.dtype != self.dtype
                or receipt.destination_flat_start != cursor
                or receipt.destination_flat_stop <= cursor
                or receipt.chunk_id in ids
            ):
                raise ProtocolError(
                    "logical destination has a gap, overlap, duplicate, or foreign leaf"
                )
            ids.add(receipt.chunk_id)
            cursor = receipt.destination_flat_stop
        if cursor != numel:
            raise ProtocolError("logical destination leaves do not cover the tensor")
        _sha256(self.logical_tensor_tree_sha256, "logical_tensor_tree_sha256")
        if self.logical_tensor_tree_contract != LOGICAL_TENSOR_TREE_CONTRACT:
            raise ProtocolError("logical tensor tree contract changed")
        expected = _logical_tensor_tree_sha256(
            destination_key=self.destination_key,
            dtype=self.dtype,
            shape=self.shape,
            chunks=self.chunks,
        )
        if self.logical_tensor_tree_sha256 != expected:
            raise ProtocolError("logical tensor tree commitment does not match leaves")

    @classmethod
    def build(
        cls,
        *,
        destination_key: str,
        dtype: str,
        shape: tuple[int, ...],
        chunks: tuple[OutputChunkReceiptV1, ...],
    ) -> Self:
        _identifier(destination_key, "destination_key")
        if dtype not in _DTYPES:
            raise ProtocolError("logical destination dtype is unsupported")
        if (
            type(shape) is not tuple
            or not shape
            or len(shape) > MAX_RANK
            or any(type(item) is not int or item <= 0 for item in shape)
        ):
            raise ProtocolError("logical destination shape is invalid")
        numel = 1
        for dimension in shape:
            if dimension > MAX_CONVERTED_PARAMETERS // numel:
                raise ProtocolError("logical destination exceeds the parameter bound")
            numel *= dimension
        if (
            type(chunks) is not tuple
            or not chunks
            or len(chunks) > MAX_CHUNKS
            or any(type(item) is not OutputChunkReceiptV1 for item in chunks)
        ):
            raise ProtocolError("logical destination chunks are invalid or unbounded")
        cursor = 0
        chunk_ids: set[str] = set()
        for receipt in chunks:
            _validate_output_receipt_profile(receipt)
            if (
                receipt.destination_key != destination_key
                or receipt.dtype != dtype
                or receipt.destination_flat_start != cursor
                or receipt.destination_flat_stop <= cursor
                or receipt.chunk_id in chunk_ids
            ):
                raise ProtocolError(
                    "logical destination has a gap, overlap, duplicate, or foreign leaf"
                )
            chunk_ids.add(receipt.chunk_id)
            cursor = receipt.destination_flat_stop
        if cursor != numel:
            raise ProtocolError("logical destination leaves do not cover the tensor")
        payload_bytes = numel * _DTYPE_BYTES[dtype]
        tree = _logical_tensor_tree_sha256(
            destination_key=destination_key,
            dtype=dtype,
            shape=shape,
            chunks=chunks,
        )
        return cls(
            destination_key=destination_key,
            dtype=dtype,
            shape=shape,
            payload_bytes=payload_bytes,
            chunks=chunks,
            logical_tensor_tree_sha256=tree,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "destination_key": self.destination_key,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "payload_bytes": self.payload_bytes,
            "chunks": [receipt.as_dict() for receipt in self.chunks],
            "logical_tensor_tree_sha256": self.logical_tensor_tree_sha256,
            "logical_tensor_tree_contract": self.logical_tensor_tree_contract,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = require_record_object(
            value,
            schema=cls.SCHEMA,
            fields=(
                "destination_key",
                "dtype",
                "shape",
                "payload_bytes",
                "chunks",
                "logical_tensor_tree_sha256",
                "logical_tensor_tree_contract",
            ),
        )
        raw_chunks = _record_array(obj["chunks"], "chunks", maximum_items=MAX_CHUNKS)
        return cls(
            destination_key=_identifier(obj["destination_key"], "destination_key"),
            dtype=_identifier(obj["dtype"], "dtype"),
            shape=_shape(obj["shape"], "shape"),
            payload_bytes=require_integer(
                obj["payload_bytes"],
                "payload_bytes",
                minimum=1,
                maximum=PINNED_CONVERTED_PAYLOAD_BYTES,
            ),
            chunks=tuple(_output_receipt_from_dict(item) for item in raw_chunks),
            logical_tensor_tree_sha256=_sha256(
                obj["logical_tensor_tree_sha256"], "logical_tensor_tree_sha256"
            ),
            logical_tensor_tree_contract=_identifier(
                obj["logical_tensor_tree_contract"],
                "logical_tensor_tree_contract",
            ),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> Self:
        raw, parsed = _bounded_parse_record(
            value,
            maximum_bytes=MAX_LOGICAL_MANIFEST_BYTES,
            name="logical stored destination",
        )
        result = cls.from_dict(parsed)
        _require_canonical_wire(raw, result, "logical stored destination")
        return result


@dataclass(frozen=True, slots=True)
class LogicalPrefixCheckpointManifestV2(CanonicalRecord):
    """Topology-neutral logical identity for a complete converted prefix."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.logical-prefix-checkpoint-manifest.v2"

    plan_id: str
    mode: str
    selection_sha256: str
    full_inventory_sha256: str
    selected_inventory_sha256: str
    source_revision: str
    maxtext_revision: str
    geometry_sha256: str
    engine_sha256: str
    numeric_codec_sha256: str
    source_adapter_identity: SourceAdapterIdentityV1
    runtime_lock_identity: R1RuntimeLockIdentityV1 | None
    sink_id: str
    source_shard_count: int
    source_entry_count: int
    source_logical_tensor_count: int
    destination_tensor_count: int
    converted_parameter_count: int
    total_chunks: int
    converted_payload_bytes: int
    max_estimated_resident_bytes: int
    target_layer_one_indexed: int
    target_semantics: str
    destinations: tuple[LogicalStoredDestinationV2, ...]
    logical_identity_scope: str = LOGICAL_IDENTITY_SCOPE
    source_lineage_contract: str = SOURCE_LINEAGE_CONTRACT
    sink_contract: str = PRODUCTION_PREFIX_SINK_CONTRACT
    object_addressing_contract: str = OBJECT_ADDRESSING_CONTRACT
    object_discovery_contract: str = OBJECT_DISCOVERY_CONTRACT
    engine_contract: str = CONVERSION_ENGINE_CONTRACT
    chunking_contract: str = CHUNKING_CONTRACT
    dequantization_contract: str = DEQUANTIZATION_CONTRACT
    storage_complete: bool = True
    production_sealable: bool = False
    external_qualification_required: tuple[str, ...] = EXTERNAL_QUALIFICATION_REQUIRED

    def __post_init__(self) -> None:
        for name in (
            "plan_id",
            "selection_sha256",
            "full_inventory_sha256",
            "selected_inventory_sha256",
            "geometry_sha256",
            "engine_sha256",
            "numeric_codec_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.mode not in {"pinned-r1-prefix", "synthetic-test-prefix"}:
            raise ProtocolError("logical manifest mode is unsupported")
        _identifier(self.source_revision, "source_revision")
        _identifier(self.maxtext_revision, "maxtext_revision")
        if type(self.source_adapter_identity) is not SourceAdapterIdentityV1:
            raise ProtocolError("source adapter identity has the wrong type")
        _validate_source_adapter_bounds(self.source_adapter_identity)
        if (
            self.runtime_lock_identity is not None
            and type(self.runtime_lock_identity) is not R1RuntimeLockIdentityV1
        ):
            raise ProtocolError("runtime lock identity has the wrong type")
        if self.runtime_lock_identity is not None:
            _identifier(
                self.runtime_lock_identity.maxtext_revision,
                "runtime maxtext_revision",
            )
            _sha256(
                self.runtime_lock_identity.pyproject_sha256,
                "runtime pyproject_sha256",
            )
            _sha256(
                self.runtime_lock_identity.uv_lock_sha256,
                "runtime uv_lock_sha256",
            )
        if self.sink_id != PRODUCTION_PREFIX_SINK_ID:
            raise ProtocolError("logical manifest sink ID is not topology neutral")
        for name, maximum in (
            ("source_shard_count", PINNED_SOURCE_SHARDS),
            ("source_entry_count", PINNED_SOURCE_ENTRIES),
            ("source_logical_tensor_count", MAX_SOURCE_LOGICAL_TENSORS),
            ("destination_tensor_count", MAX_DESTINATIONS),
            ("converted_parameter_count", MAX_CONVERTED_PARAMETERS),
            ("total_chunks", MAX_CHUNKS),
            ("converted_payload_bytes", PINNED_CONVERTED_PAYLOAD_BYTES),
            ("max_estimated_resident_bytes", PINNED_CONVERTED_PAYLOAD_BYTES),
            ("target_layer_one_indexed", PINNED_TARGET_LAYER_ONE_INDEXED),
        ):
            require_integer(getattr(self, name), name, minimum=1, maximum=maximum)
        if self.target_semantics != TARGET_SEMANTICS:
            raise ProtocolError("target routed-prefix semantics changed")
        if (
            type(self.destinations) is not tuple
            or not self.destinations
            or len(self.destinations) > MAX_DESTINATIONS
            or any(
                type(item) is not LogicalStoredDestinationV2
                for item in self.destinations
            )
        ):
            raise ProtocolError("manifest destinations are invalid or unbounded")
        keys = tuple(item.destination_key for item in self.destinations)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ProtocolError("manifest destinations must be unique and sorted")
        if self.destination_tensor_count != len(self.destinations):
            raise ProtocolError("manifest destination count is inconsistent")
        chunks = [
            receipt
            for destination in self.destinations
            for receipt in destination.chunks
        ]
        chunk_ids = [receipt.chunk_id for receipt in chunks]
        if (
            len(chunks) != self.total_chunks
            or len(chunk_ids) != len(set(chunk_ids))
            or any(receipt.plan_id != self.plan_id for receipt in chunks)
        ):
            raise ProtocolError("manifest chunk identity or count is inconsistent")
        if sum(item.payload_bytes for item in self.destinations) != (
            self.converted_payload_bytes
        ):
            raise ProtocolError("manifest payload byte total is inconsistent")
        if sum(math.prod(item.shape) for item in self.destinations) != (
            self.converted_parameter_count
        ):
            raise ProtocolError("manifest parameter total is inconsistent")
        if (
            self.logical_identity_scope != LOGICAL_IDENTITY_SCOPE
            or self.source_lineage_contract != SOURCE_LINEAGE_CONTRACT
            or self.sink_contract != PRODUCTION_PREFIX_SINK_CONTRACT
            or self.object_addressing_contract != OBJECT_ADDRESSING_CONTRACT
            or self.object_discovery_contract != OBJECT_DISCOVERY_CONTRACT
            or self.engine_contract != CONVERSION_ENGINE_CONTRACT
            or self.chunking_contract != CHUNKING_CONTRACT
            or self.dequantization_contract != DEQUANTIZATION_CONTRACT
        ):
            raise ProtocolError("logical manifest protocol contracts changed")
        if self.storage_complete is not True or self.production_sealable is not False:
            raise ProtocolError("logical manifest evidence flags are invalid")
        if self.external_qualification_required != EXTERNAL_QUALIFICATION_REQUIRED:
            raise ProtocolError("logical manifest qualification gates changed")
        if self.mode == "pinned-r1-prefix":
            self.require_pinned_production_invariants()
        elif self.runtime_lock_identity is not None:
            raise ProtocolError("synthetic manifest cannot claim a runtime lock")
        if len(canonical_json_bytes(self)) > MAX_LOGICAL_MANIFEST_BYTES:
            raise ProtocolError("logical manifest exceeds its canonical byte bound")

    @property
    def logical_checkpoint_id(self) -> str:
        """The canonical manifest digest; no self-hash field is serialized."""

        return self.digest

    @property
    def pinned_production_profile(self) -> bool:
        return self.mode == "pinned-r1-prefix"

    def require_pinned_production_invariants(self) -> None:
        expected = (
            PINNED_SOURCE_SHARDS,
            PINNED_SOURCE_ENTRIES,
            PINNED_SOURCE_LOGICAL_TENSORS,
            PINNED_DESTINATION_TENSORS,
            PINNED_CONVERTED_PARAMETERS,
            PINNED_TOTAL_CHUNKS,
            PINNED_CONVERTED_PAYLOAD_BYTES,
            PINNED_TARGET_LAYER_ONE_INDEXED,
            TARGET_SEMANTICS,
        )
        actual = (
            self.source_shard_count,
            self.source_entry_count,
            self.source_logical_tensor_count,
            self.destination_tensor_count,
            self.converted_parameter_count,
            self.total_chunks,
            self.converted_payload_bytes,
            self.target_layer_one_indexed,
            self.target_semantics,
        )
        if (
            actual != expected
            or self.runtime_lock_identity is None
            or self.source_revision != HF_REVISION
            or self.maxtext_revision != MAXTEXT_REVISION
            or self.runtime_lock_identity.maxtext_revision != MAXTEXT_REVISION
        ):
            raise ProtocolError("pinned R1 prefix invariants are not exact")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "plan_id": self.plan_id,
            "mode": self.mode,
            "selection_sha256": self.selection_sha256,
            "full_inventory_sha256": self.full_inventory_sha256,
            "selected_inventory_sha256": self.selected_inventory_sha256,
            "source_revision": self.source_revision,
            "maxtext_revision": self.maxtext_revision,
            "geometry_sha256": self.geometry_sha256,
            "engine_sha256": self.engine_sha256,
            "numeric_codec_sha256": self.numeric_codec_sha256,
            "source_adapter_identity": self.source_adapter_identity.as_dict(),
            "runtime_lock_identity": _runtime_dict(self.runtime_lock_identity),
            "sink_id": self.sink_id,
            "source_shard_count": self.source_shard_count,
            "source_entry_count": self.source_entry_count,
            "source_logical_tensor_count": self.source_logical_tensor_count,
            "destination_tensor_count": self.destination_tensor_count,
            "converted_parameter_count": self.converted_parameter_count,
            "total_chunks": self.total_chunks,
            "converted_payload_bytes": self.converted_payload_bytes,
            "max_estimated_resident_bytes": self.max_estimated_resident_bytes,
            "target_layer_one_indexed": self.target_layer_one_indexed,
            "target_semantics": self.target_semantics,
            "destinations": [item.as_dict() for item in self.destinations],
            "logical_identity_scope": self.logical_identity_scope,
            "source_lineage_contract": self.source_lineage_contract,
            "sink_contract": self.sink_contract,
            "object_addressing_contract": self.object_addressing_contract,
            "object_discovery_contract": self.object_discovery_contract,
            "engine_contract": self.engine_contract,
            "chunking_contract": self.chunking_contract,
            "dequantization_contract": self.dequantization_contract,
            "storage_complete": self.storage_complete,
            "production_sealable": self.production_sealable,
            "external_qualification_required": list(
                self.external_qualification_required
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "plan_id",
            "mode",
            "selection_sha256",
            "full_inventory_sha256",
            "selected_inventory_sha256",
            "source_revision",
            "maxtext_revision",
            "geometry_sha256",
            "engine_sha256",
            "numeric_codec_sha256",
            "source_adapter_identity",
            "runtime_lock_identity",
            "sink_id",
            "source_shard_count",
            "source_entry_count",
            "source_logical_tensor_count",
            "destination_tensor_count",
            "converted_parameter_count",
            "total_chunks",
            "converted_payload_bytes",
            "max_estimated_resident_bytes",
            "target_layer_one_indexed",
            "target_semantics",
            "destinations",
            "logical_identity_scope",
            "source_lineage_contract",
            "sink_contract",
            "object_addressing_contract",
            "object_discovery_contract",
            "engine_contract",
            "chunking_contract",
            "dequantization_contract",
            "storage_complete",
            "production_sealable",
            "external_qualification_required",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        raw_destinations = _record_array(
            obj["destinations"],
            "destinations",
            maximum_items=MAX_DESTINATIONS,
        )
        return cls(
            plan_id=_sha256(obj["plan_id"], "plan_id"),
            mode=_identifier(obj["mode"], "mode"),
            selection_sha256=_sha256(obj["selection_sha256"], "selection_sha256"),
            full_inventory_sha256=_sha256(
                obj["full_inventory_sha256"], "full_inventory_sha256"
            ),
            selected_inventory_sha256=_sha256(
                obj["selected_inventory_sha256"], "selected_inventory_sha256"
            ),
            source_revision=_identifier(obj["source_revision"], "source_revision"),
            maxtext_revision=_identifier(obj["maxtext_revision"], "maxtext_revision"),
            geometry_sha256=_sha256(obj["geometry_sha256"], "geometry_sha256"),
            engine_sha256=_sha256(obj["engine_sha256"], "engine_sha256"),
            numeric_codec_sha256=_sha256(
                obj["numeric_codec_sha256"], "numeric_codec_sha256"
            ),
            source_adapter_identity=_source_adapter_from_dict(
                obj["source_adapter_identity"]
            ),
            runtime_lock_identity=_runtime_from_dict(obj["runtime_lock_identity"]),
            sink_id=_identifier(obj["sink_id"], "sink_id"),
            source_shard_count=require_integer(
                obj["source_shard_count"],
                "source_shard_count",
                minimum=1,
                maximum=PINNED_SOURCE_SHARDS,
            ),
            source_entry_count=require_integer(
                obj["source_entry_count"],
                "source_entry_count",
                minimum=1,
                maximum=PINNED_SOURCE_ENTRIES,
            ),
            source_logical_tensor_count=require_integer(
                obj["source_logical_tensor_count"],
                "source_logical_tensor_count",
                minimum=1,
                maximum=MAX_SOURCE_LOGICAL_TENSORS,
            ),
            destination_tensor_count=require_integer(
                obj["destination_tensor_count"],
                "destination_tensor_count",
                minimum=1,
                maximum=MAX_DESTINATIONS,
            ),
            converted_parameter_count=require_integer(
                obj["converted_parameter_count"],
                "converted_parameter_count",
                minimum=1,
                maximum=MAX_CONVERTED_PARAMETERS,
            ),
            total_chunks=require_integer(
                obj["total_chunks"],
                "total_chunks",
                minimum=1,
                maximum=MAX_CHUNKS,
            ),
            converted_payload_bytes=require_integer(
                obj["converted_payload_bytes"],
                "converted_payload_bytes",
                minimum=1,
                maximum=PINNED_CONVERTED_PAYLOAD_BYTES,
            ),
            max_estimated_resident_bytes=require_integer(
                obj["max_estimated_resident_bytes"],
                "max_estimated_resident_bytes",
                minimum=1,
                maximum=PINNED_CONVERTED_PAYLOAD_BYTES,
            ),
            target_layer_one_indexed=require_integer(
                obj["target_layer_one_indexed"],
                "target_layer_one_indexed",
                minimum=1,
                maximum=PINNED_TARGET_LAYER_ONE_INDEXED,
            ),
            target_semantics=_identifier(obj["target_semantics"], "target_semantics"),
            destinations=tuple(
                LogicalStoredDestinationV2.from_dict(item) for item in raw_destinations
            ),
            logical_identity_scope=_identifier(
                obj["logical_identity_scope"], "logical_identity_scope"
            ),
            source_lineage_contract=_identifier(
                obj["source_lineage_contract"], "source_lineage_contract"
            ),
            sink_contract=_identifier(obj["sink_contract"], "sink_contract"),
            object_addressing_contract=_identifier(
                obj["object_addressing_contract"], "object_addressing_contract"
            ),
            object_discovery_contract=_identifier(
                obj["object_discovery_contract"], "object_discovery_contract"
            ),
            engine_contract=_identifier(obj["engine_contract"], "engine_contract"),
            chunking_contract=_identifier(
                obj["chunking_contract"], "chunking_contract"
            ),
            dequantization_contract=_identifier(
                obj["dequantization_contract"], "dequantization_contract"
            ),
            storage_complete=_strict_bool(obj["storage_complete"], "storage_complete"),
            production_sealable=_strict_bool(
                obj["production_sealable"], "production_sealable"
            ),
            external_qualification_required=_string_array(
                obj["external_qualification_required"],
                "external_qualification_required",
                maximum_items=len(EXTERNAL_QUALIFICATION_REQUIRED),
            ),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> Self:
        raw, parsed = _bounded_parse_record(
            value,
            maximum_bytes=MAX_LOGICAL_MANIFEST_BYTES,
            name="logical manifest",
        )
        result = cls.from_dict(parsed)
        _require_canonical_wire(raw, result, "logical manifest")
        return result

    def validate_for_plan(self, plan: StreamingConversionPlanV1) -> None:
        """Re-derive every field and relation without truncating paired arrays."""

        if type(plan) is not StreamingConversionPlanV1:
            raise PrefixCheckpointV2Error(
                "validation requires an exact conversion plan"
            )
        totals = plan.selection.selected.totals
        expected_header = (
            plan.plan_id,
            plan.mode,
            plan.selection_sha256,
            plan.full_inventory_sha256,
            plan.selected_inventory_sha256,
            plan.source_revision,
            plan.maxtext_revision,
            plan.geometry_sha256,
            plan.engine_sha256,
            plan.numeric_codec_sha256,
            plan.source_adapter_identity,
            plan.runtime_lock_identity,
            PRODUCTION_PREFIX_SINK_ID,
            len(plan.selection.selected.required_shards),
            totals.source_entry_count,
            totals.logical_tensor_count,
            len(plan.destinations),
            sum(math.prod(item.shape) for item in plan.destinations),
            len(plan.chunks),
            plan.converted_payload_bytes,
            max(item.estimated_peak_resident_bytes for item in plan.shard_work),
            plan.selection.full.geometry.target_layer_one_indexed,
            TARGET_SEMANTICS,
        )
        actual_header = (
            self.plan_id,
            self.mode,
            self.selection_sha256,
            self.full_inventory_sha256,
            self.selected_inventory_sha256,
            self.source_revision,
            self.maxtext_revision,
            self.geometry_sha256,
            self.engine_sha256,
            self.numeric_codec_sha256,
            self.source_adapter_identity,
            self.runtime_lock_identity,
            self.sink_id,
            self.source_shard_count,
            self.source_entry_count,
            self.source_logical_tensor_count,
            self.destination_tensor_count,
            self.converted_parameter_count,
            self.total_chunks,
            self.converted_payload_bytes,
            self.max_estimated_resident_bytes,
            self.target_layer_one_indexed,
            self.target_semantics,
        )
        if actual_header != expected_header:
            raise PrefixCheckpointV2Error(
                "logical manifest differs from conversion plan"
            )
        if len(self.destinations) != len(plan.destinations):
            raise PrefixCheckpointV2Error("destination cardinality differs from plan")
        chunks_by_id = {chunk.chunk_id: chunk for chunk in plan.chunks}
        if len(chunks_by_id) != len(plan.chunks):
            raise PrefixCheckpointV2Error("conversion plan chunk IDs collide")
        for destination_index in range(len(plan.destinations)):
            expected_destination = plan.destinations[destination_index]
            actual_destination = self.destinations[destination_index]
            if (
                actual_destination.destination_key
                != expected_destination.destination_key
                or actual_destination.dtype != expected_destination.dtype
                or actual_destination.shape != expected_destination.shape
                or actual_destination.payload_bytes
                != expected_destination.payload_bytes
                or len(actual_destination.chunks) != len(expected_destination.chunks)
            ):
                raise PrefixCheckpointV2Error("destination differs from exact plan")
            for chunk_index in range(len(expected_destination.chunks)):
                start, stop, chunk_id = expected_destination.chunks[chunk_index]
                receipt = actual_destination.chunks[chunk_index]
                chunk = chunks_by_id.get(chunk_id)
                if chunk is None:
                    raise PrefixCheckpointV2Error("manifest references a foreign chunk")
                try:
                    receipt.validate_for(
                        plan,
                        chunk,
                        sink_id=PRODUCTION_PREFIX_SINK_ID,
                    )
                    _validate_output_receipt_profile(receipt)
                except (CheckpointConversionError, ProtocolError) as error:
                    raise PrefixCheckpointV2Error(
                        "logical output receipt differs from exact plan"
                    ) from error
                if (
                    receipt.chunk_id != chunk_id
                    or receipt.destination_flat_start != start
                    or receipt.destination_flat_stop != stop
                ):
                    raise PrefixCheckpointV2Error(
                        "logical output interval differs from exact plan"
                    )


@dataclass(frozen=True, slots=True)
class RegionalPrefixChunkBindingV1(CanonicalRecord):
    """Cross-bind one logical output receipt to one regional GCS generation."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.regional-prefix-chunk-binding.v1"

    plan_id: str
    chunk_id: str
    output_receipt: OutputChunkReceiptV1
    physical_receipt: GCSPhysicalObjectReceiptV1
    source_lineage: tuple[PlanSourceIdentityBindingV1, ...]
    source_lineage_contract: str = SOURCE_LINEAGE_CONTRACT
    regional_binding_contract: str = REGIONAL_BINDING_CONTRACT

    def __post_init__(self) -> None:
        _sha256(self.plan_id, "plan_id")
        _sha256(self.chunk_id, "chunk_id")
        _validate_output_receipt_profile(self.output_receipt)
        _validate_physical_bounds(
            self.physical_receipt,
            maximum_size_bytes=MAX_PHYSICAL_PAYLOAD_BYTES,
        )
        if (
            type(self.source_lineage) is not tuple
            or not 1 <= len(self.source_lineage) <= 2
            or any(
                type(item) is not PlanSourceIdentityBindingV1
                for item in self.source_lineage
            )
        ):
            raise ProtocolError("regional binding source lineage is invalid")
        for item in self.source_lineage:
            _identifier(item.role, "source lineage role")
            _identifier(item.shard, "source lineage shard")
            _sha256(item.sha256, "source lineage SHA-256")
            require_integer(
                item.size_bytes,
                "source lineage size_bytes",
                minimum=1,
                maximum=1 << 40,
            )
            _validate_source_adapter_bounds(item.source_adapter_identity)
            if item.evidence_contract != SOURCE_LINEAGE_CONTRACT:
                raise ProtocolError("regional binding source lineage contract changed")
        roles = tuple(item.role for item in self.source_lineage)
        if roles not in {("source",), ("source", "scale")}:
            raise ProtocolError("regional binding source lineage roles are ambiguous")
        if len({item.shard for item in self.source_lineage}) != len(
            self.source_lineage
        ) or any(
            item.source_adapter_identity
            != self.source_lineage[0].source_adapter_identity
            for item in self.source_lineage
        ):
            raise ProtocolError(
                "regional binding source lineage identities are ambiguous"
            )
        if (
            self.plan_id != self.output_receipt.plan_id
            or self.chunk_id != self.output_receipt.chunk_id
            or self.output_receipt.immutable_object_id
            != self.physical_receipt.object_id
            or self.output_receipt.output_sha256 != self.physical_receipt.sha256
            or self.output_receipt.output_size_bytes != self.physical_receipt.size_bytes
            or self.physical_receipt.content_type != "application/octet-stream"
        ):
            raise ProtocolError("regional binding does not cross-bind exact bytes")
        if (
            self.source_lineage_contract != SOURCE_LINEAGE_CONTRACT
            or self.regional_binding_contract != REGIONAL_BINDING_CONTRACT
        ):
            raise ProtocolError("regional binding contracts changed")
        if len(canonical_json_bytes(self)) > MAX_BINDING_BYTES:
            raise ProtocolError("regional binding exceeds its canonical byte bound")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "plan_id": self.plan_id,
            "chunk_id": self.chunk_id,
            "output_receipt": self.output_receipt.as_dict(),
            "physical_receipt": self.physical_receipt.as_dict(),
            "source_lineage": [item.as_dict() for item in self.source_lineage],
            "source_lineage_contract": self.source_lineage_contract,
            "regional_binding_contract": self.regional_binding_contract,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = require_record_object(
            value,
            schema=cls.SCHEMA,
            fields=(
                "plan_id",
                "chunk_id",
                "output_receipt",
                "physical_receipt",
                "source_lineage",
                "source_lineage_contract",
                "regional_binding_contract",
            ),
        )
        raw_lineage = _record_array(
            obj["source_lineage"],
            "source_lineage",
            maximum_items=2,
        )
        return cls(
            plan_id=_sha256(obj["plan_id"], "plan_id"),
            chunk_id=_sha256(obj["chunk_id"], "chunk_id"),
            output_receipt=_output_receipt_from_dict(obj["output_receipt"]),
            physical_receipt=_physical_receipt_from_dict(
                obj["physical_receipt"],
                maximum_size_bytes=MAX_PHYSICAL_PAYLOAD_BYTES,
            ),
            source_lineage=tuple(
                _source_lineage_from_dict(item) for item in raw_lineage
            ),
            source_lineage_contract=_identifier(
                obj["source_lineage_contract"], "source_lineage_contract"
            ),
            regional_binding_contract=_identifier(
                obj["regional_binding_contract"], "regional_binding_contract"
            ),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> Self:
        raw, parsed = _bounded_parse_record(
            value, maximum_bytes=MAX_BINDING_BYTES, name="regional chunk binding"
        )
        result = cls.from_dict(parsed)
        _require_canonical_wire(raw, result, "regional chunk binding")
        return result

    def validate_for(
        self,
        plan: StreamingConversionPlanV1,
        chunk: ConversionChunkIdentityV1,
    ) -> None:
        if (
            type(plan) is not StreamingConversionPlanV1
            or type(chunk) is not ConversionChunkIdentityV1
        ):
            raise PrefixCheckpointV2Error(
                "binding validation requires exact plan types"
            )
        try:
            self.output_receipt.validate_for(
                plan, chunk, sink_id=PRODUCTION_PREFIX_SINK_ID
            )
        except CheckpointConversionError as error:
            raise PrefixCheckpointV2Error(
                "regional binding differs from plan"
            ) from error
        if self.plan_id != plan.plan_id or self.chunk_id != chunk.chunk_id:
            raise PrefixCheckpointV2Error("regional binding identity differs from plan")
        if self.source_lineage != _lineage_for_chunk(plan, chunk):
            raise PrefixCheckpointV2Error(
                "regional binding lineage differs from exact plan"
            )


def build_regional_prefix_chunk_binding_v1(
    plan: StreamingConversionPlanV1,
    chunk: ConversionChunkIdentityV1,
    output_receipt: OutputChunkReceiptV1,
    physical_receipt: GCSPhysicalObjectReceiptV1,
) -> RegionalPrefixChunkBindingV1:
    """Build one binding with exact plan-derived source lineage."""

    if (
        type(plan) is not StreamingConversionPlanV1
        or type(chunk) is not ConversionChunkIdentityV1
        or type(output_receipt) is not OutputChunkReceiptV1
        or type(physical_receipt) is not GCSPhysicalObjectReceiptV1
    ):
        raise PrefixCheckpointV2Error(
            "regional binding construction requires exact protocol types"
        )
    try:
        output_receipt.validate_for(
            plan,
            chunk,
            sink_id=PRODUCTION_PREFIX_SINK_ID,
        )
        binding = RegionalPrefixChunkBindingV1(
            plan_id=plan.plan_id,
            chunk_id=chunk.chunk_id,
            output_receipt=output_receipt,
            physical_receipt=physical_receipt,
            source_lineage=_lineage_for_chunk(plan, chunk),
        )
        binding.validate_for(plan, chunk)
    except (CheckpointConversionError, ProtocolError) as error:
        raise PrefixCheckpointV2Error(
            "regional binding cannot be derived from the exact plan"
        ) from error
    return binding


@dataclass(frozen=True, slots=True)
class RegionalChunkReplicaV1(CanonicalRecord):
    """Exact regional receipt for one durably published binding record."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.regional-chunk-replica.v1"

    chunk_id: str
    binding_receipt: GCSPhysicalObjectReceiptV1

    def __post_init__(self) -> None:
        _sha256(self.chunk_id, "chunk_id")
        _validate_physical_bounds(
            self.binding_receipt,
            maximum_size_bytes=MAX_BINDING_BYTES,
        )
        if self.binding_receipt.content_type != "application/json":
            raise ProtocolError("regional chunk replica is not a JSON binding")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "chunk_id": self.chunk_id,
            "binding_receipt": self.binding_receipt.as_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = require_record_object(
            value,
            schema=cls.SCHEMA,
            fields=("chunk_id", "binding_receipt"),
        )
        return cls(
            chunk_id=_sha256(obj["chunk_id"], "chunk_id"),
            binding_receipt=_physical_receipt_from_dict(
                obj["binding_receipt"],
                maximum_size_bytes=MAX_BINDING_BYTES,
            ),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> Self:
        raw, parsed = _bounded_parse_record(
            value,
            maximum_bytes=MAX_BINDING_BYTES,
            name="regional chunk replica",
        )
        result = cls.from_dict(parsed)
        _require_canonical_wire(raw, result, "regional chunk replica")
        return result


@dataclass(frozen=True, slots=True)
class RegionalPrefixCheckpointEnvelopeV1(CanonicalRecord):
    """Sole regional commit point for one topology-neutral logical manifest."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.regional-prefix-checkpoint-envelope.v1"

    logical_manifest_sha256: str
    logical_manifest_size_bytes: int
    plan_id: str
    region: str
    store_id: str
    manifest_receipt: GCSPhysicalObjectReceiptV1
    chunk_replicas: tuple[RegionalChunkReplicaV1, ...]
    total_chunks: int
    converted_payload_bytes: int
    regional_envelope_contract: str = REGIONAL_ENVELOPE_CONTRACT
    regional_copy_contract: str = REGIONAL_COPY_CONTRACT
    object_discovery_contract: str = OBJECT_DISCOVERY_CONTRACT
    storage_complete: bool = True
    production_sealable: bool = False
    external_qualification_required: tuple[str, ...] = EXTERNAL_QUALIFICATION_REQUIRED

    def __post_init__(self) -> None:
        _sha256(self.logical_manifest_sha256, "logical_manifest_sha256")
        _sha256(self.plan_id, "plan_id")
        _identifier(self.region, "region")
        _identifier(self.store_id, "store_id")
        require_integer(
            self.logical_manifest_size_bytes,
            "logical_manifest_size_bytes",
            minimum=1,
            maximum=MAX_LOGICAL_MANIFEST_BYTES,
        )
        _validate_physical_bounds(
            self.manifest_receipt,
            maximum_size_bytes=MAX_LOGICAL_MANIFEST_BYTES,
        )
        if (
            self.manifest_receipt.sha256 != self.logical_manifest_sha256
            or self.manifest_receipt.size_bytes != self.logical_manifest_size_bytes
            or self.manifest_receipt.object_id
            != content_addressed_record_object_id(self.logical_manifest_sha256)
            or self.manifest_receipt.content_type != "application/json"
            or self.manifest_receipt.region != self.region
            or self.manifest_receipt.store_id != self.store_id
        ):
            raise ProtocolError("envelope does not bind the exact logical manifest")
        if (
            type(self.chunk_replicas) is not tuple
            or not self.chunk_replicas
            or len(self.chunk_replicas) > MAX_CHUNKS
            or any(
                type(item) is not RegionalChunkReplicaV1 for item in self.chunk_replicas
            )
        ):
            raise ProtocolError("envelope chunk replicas are invalid or unbounded")
        ids = tuple(item.chunk_id for item in self.chunk_replicas)
        if len(ids) != len(set(ids)):
            raise ProtocolError("envelope chunk IDs are duplicated")
        if any(
            item.binding_receipt.region != self.region
            or item.binding_receipt.store_id != self.store_id
            or item.binding_receipt.object_id
            != regional_chunk_binding_object_id(self.plan_id, item.chunk_id)
            for item in self.chunk_replicas
        ):
            raise ProtocolError("envelope contains another regional store identity")
        require_integer(
            self.total_chunks,
            "total_chunks",
            minimum=1,
            maximum=MAX_CHUNKS,
        )
        require_integer(
            self.converted_payload_bytes,
            "converted_payload_bytes",
            minimum=1,
            maximum=PINNED_CONVERTED_PAYLOAD_BYTES,
        )
        if self.total_chunks != len(self.chunk_replicas):
            raise ProtocolError("envelope chunk count is inconsistent")
        if (
            self.regional_envelope_contract != REGIONAL_ENVELOPE_CONTRACT
            or self.regional_copy_contract != REGIONAL_COPY_CONTRACT
            or self.object_discovery_contract != OBJECT_DISCOVERY_CONTRACT
        ):
            raise ProtocolError("regional envelope contracts changed")
        if self.storage_complete is not True or self.production_sealable is not False:
            raise ProtocolError("regional envelope evidence flags are invalid")
        if self.external_qualification_required != EXTERNAL_QUALIFICATION_REQUIRED:
            raise ProtocolError("regional envelope qualification gates changed")
        if len(canonical_json_bytes(self)) > MAX_REGIONAL_ENVELOPE_BYTES:
            raise ProtocolError("regional envelope exceeds its canonical byte bound")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "logical_manifest_sha256": self.logical_manifest_sha256,
            "logical_manifest_size_bytes": self.logical_manifest_size_bytes,
            "plan_id": self.plan_id,
            "region": self.region,
            "store_id": self.store_id,
            "manifest_receipt": self.manifest_receipt.as_dict(),
            "chunk_replicas": [item.as_dict() for item in self.chunk_replicas],
            "total_chunks": self.total_chunks,
            "converted_payload_bytes": self.converted_payload_bytes,
            "regional_envelope_contract": self.regional_envelope_contract,
            "regional_copy_contract": self.regional_copy_contract,
            "object_discovery_contract": self.object_discovery_contract,
            "storage_complete": self.storage_complete,
            "production_sealable": self.production_sealable,
            "external_qualification_required": list(
                self.external_qualification_required
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = require_record_object(
            value,
            schema=cls.SCHEMA,
            fields=(
                "logical_manifest_sha256",
                "logical_manifest_size_bytes",
                "plan_id",
                "region",
                "store_id",
                "manifest_receipt",
                "chunk_replicas",
                "total_chunks",
                "converted_payload_bytes",
                "regional_envelope_contract",
                "regional_copy_contract",
                "object_discovery_contract",
                "storage_complete",
                "production_sealable",
                "external_qualification_required",
            ),
        )
        raw_replicas = _record_array(
            obj["chunk_replicas"],
            "chunk_replicas",
            maximum_items=MAX_CHUNKS,
        )
        return cls(
            logical_manifest_sha256=_sha256(
                obj["logical_manifest_sha256"], "logical_manifest_sha256"
            ),
            logical_manifest_size_bytes=require_integer(
                obj["logical_manifest_size_bytes"],
                "logical_manifest_size_bytes",
                minimum=1,
                maximum=MAX_LOGICAL_MANIFEST_BYTES,
            ),
            plan_id=_sha256(obj["plan_id"], "plan_id"),
            region=_identifier(obj["region"], "region"),
            store_id=_identifier(obj["store_id"], "store_id"),
            manifest_receipt=_physical_receipt_from_dict(
                obj["manifest_receipt"],
                maximum_size_bytes=MAX_LOGICAL_MANIFEST_BYTES,
            ),
            chunk_replicas=tuple(
                RegionalChunkReplicaV1.from_dict(item) for item in raw_replicas
            ),
            total_chunks=require_integer(
                obj["total_chunks"],
                "total_chunks",
                minimum=1,
                maximum=MAX_CHUNKS,
            ),
            converted_payload_bytes=require_integer(
                obj["converted_payload_bytes"],
                "converted_payload_bytes",
                minimum=1,
                maximum=PINNED_CONVERTED_PAYLOAD_BYTES,
            ),
            regional_envelope_contract=_identifier(
                obj["regional_envelope_contract"], "regional_envelope_contract"
            ),
            regional_copy_contract=_identifier(
                obj["regional_copy_contract"], "regional_copy_contract"
            ),
            object_discovery_contract=_identifier(
                obj["object_discovery_contract"], "object_discovery_contract"
            ),
            storage_complete=_strict_bool(obj["storage_complete"], "storage_complete"),
            production_sealable=_strict_bool(
                obj["production_sealable"], "production_sealable"
            ),
            external_qualification_required=_string_array(
                obj["external_qualification_required"],
                "external_qualification_required",
                maximum_items=len(EXTERNAL_QUALIFICATION_REQUIRED),
            ),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> Self:
        raw, parsed = _bounded_parse_record(
            value,
            maximum_bytes=MAX_REGIONAL_ENVELOPE_BYTES,
            name="regional envelope",
        )
        result = cls.from_dict(parsed)
        _require_canonical_wire(raw, result, "regional envelope")
        return result

    def validate_for_manifest(
        self, manifest: LogicalPrefixCheckpointManifestV2
    ) -> None:
        if type(manifest) is not LogicalPrefixCheckpointManifestV2:
            raise PrefixCheckpointV2Error(
                "envelope validation requires an exact logical manifest"
            )
        encoded = manifest.canonical_bytes()
        if (
            self.logical_manifest_sha256 != hashlib.sha256(encoded).hexdigest()
            or self.logical_manifest_size_bytes != len(encoded)
            or self.plan_id != manifest.plan_id
            or self.total_chunks != manifest.total_chunks
            or self.converted_payload_bytes != manifest.converted_payload_bytes
        ):
            raise PrefixCheckpointV2Error("envelope differs from logical manifest")
        logical_receipts = [
            receipt
            for destination in manifest.destinations
            for receipt in destination.chunks
        ]
        if len(self.chunk_replicas) != len(logical_receipts):
            raise PrefixCheckpointV2Error("envelope replica cardinality differs")
        for index in range(len(logical_receipts)):
            logical = logical_receipts[index]
            replica = self.chunk_replicas[index]
            if (
                replica.chunk_id != logical.chunk_id
                or replica.binding_receipt.object_id
                != regional_chunk_binding_object_id(self.plan_id, logical.chunk_id)
                or replica.binding_receipt.store_id != self.store_id
                or replica.binding_receipt.region != self.region
            ):
                raise PrefixCheckpointV2Error(
                    "envelope binding replica differs from ordered logical content"
                )


def build_logical_prefix_checkpoint_manifest_v2(
    plan: StreamingConversionPlanV1,
    summary: ConversionAttemptSummaryV1,
) -> LogicalPrefixCheckpointManifestV2:
    """Build deterministic logical content, intentionally ignoring observations."""

    if type(plan) is not StreamingConversionPlanV1:
        raise PrefixCheckpointV2Error("logical finalization requires an exact plan")
    if type(summary) is not ConversionAttemptSummaryV1:
        raise PrefixCheckpointV2Error("logical finalization requires an exact summary")
    if (
        summary.plan_id != plan.plan_id
        or summary.sink_id != PRODUCTION_PREFIX_SINK_ID
        or summary.total_chunks != len(plan.chunks)
        or summary.converted_payload_bytes != plan.converted_payload_bytes
        or summary.max_estimated_resident_bytes
        != max(item.estimated_peak_resident_bytes for item in plan.shard_work)
    ):
        raise PrefixCheckpointV2Error("conversion summary differs from exact plan")
    receipts_by_id = {receipt.chunk_id: receipt for receipt in summary.receipts}
    plan_chunks_by_id = {chunk.chunk_id: chunk for chunk in plan.chunks}
    if (
        len(receipts_by_id) != len(summary.receipts)
        or len(plan_chunks_by_id) != len(plan.chunks)
        or set(receipts_by_id) != set(plan_chunks_by_id)
    ):
        raise PrefixCheckpointV2Error("summary receipts are not an exact chunk set")
    for chunk_id, chunk in plan_chunks_by_id.items():
        receipt = receipts_by_id[chunk_id]
        try:
            receipt.validate_for(plan, chunk, sink_id=PRODUCTION_PREFIX_SINK_ID)
            _validate_output_receipt_profile(receipt)
        except (CheckpointConversionError, ProtocolError) as error:
            raise PrefixCheckpointV2Error(
                f"summary receipt differs from plan: {chunk_id}"
            ) from error
    destinations: list[LogicalStoredDestinationV2] = []
    for destination in plan.destinations:
        receipts = tuple(
            receipts_by_id[chunk_id] for _start, _stop, chunk_id in destination.chunks
        )
        destinations.append(
            LogicalStoredDestinationV2.build(
                destination_key=destination.destination_key,
                dtype=destination.dtype,
                shape=destination.shape,
                chunks=receipts,
            )
        )
    totals = plan.selection.selected.totals
    manifest = LogicalPrefixCheckpointManifestV2(
        plan_id=plan.plan_id,
        mode=plan.mode,
        selection_sha256=plan.selection_sha256,
        full_inventory_sha256=plan.full_inventory_sha256,
        selected_inventory_sha256=plan.selected_inventory_sha256,
        source_revision=plan.source_revision,
        maxtext_revision=plan.maxtext_revision,
        geometry_sha256=plan.geometry_sha256,
        engine_sha256=plan.engine_sha256,
        numeric_codec_sha256=plan.numeric_codec_sha256,
        source_adapter_identity=plan.source_adapter_identity,
        runtime_lock_identity=plan.runtime_lock_identity,
        sink_id=PRODUCTION_PREFIX_SINK_ID,
        source_shard_count=len(plan.selection.selected.required_shards),
        source_entry_count=totals.source_entry_count,
        source_logical_tensor_count=totals.logical_tensor_count,
        destination_tensor_count=len(plan.destinations),
        converted_parameter_count=sum(
            math.prod(item.shape) for item in plan.destinations
        ),
        total_chunks=len(plan.chunks),
        converted_payload_bytes=plan.converted_payload_bytes,
        max_estimated_resident_bytes=summary.max_estimated_resident_bytes,
        target_layer_one_indexed=(
            plan.selection.full.geometry.target_layer_one_indexed
        ),
        target_semantics=TARGET_SEMANTICS,
        destinations=tuple(destinations),
    )
    manifest.validate_for_plan(plan)
    return manifest


def build_regional_prefix_checkpoint_envelope_v1(
    plan: StreamingConversionPlanV1,
    manifest: LogicalPrefixCheckpointManifestV2,
    manifest_receipt: GCSPhysicalObjectReceiptV1,
    bindings: tuple[RegionalPrefixChunkBindingV1, ...],
    binding_receipts: tuple[GCSPhysicalObjectReceiptV1, ...],
) -> RegionalPrefixCheckpointEnvelopeV1:
    """Cross-bind one regional replica set; publish its envelope only last."""

    if type(plan) is not StreamingConversionPlanV1:
        raise PrefixCheckpointV2Error("regional finalization requires an exact plan")
    if type(manifest) is not LogicalPrefixCheckpointManifestV2:
        raise PrefixCheckpointV2Error(
            "regional finalization requires a logical manifest"
        )
    if type(manifest_receipt) is not GCSPhysicalObjectReceiptV1:
        raise PrefixCheckpointV2Error("regional finalization requires a GCS receipt")
    if (
        type(bindings) is not tuple
        or not 1 <= len(bindings) <= MAX_CHUNKS
        or any(type(item) is not RegionalPrefixChunkBindingV1 for item in bindings)
    ):
        raise PrefixCheckpointV2Error("regional bindings must be an exact tuple")
    if (
        type(binding_receipts) is not tuple
        or not 1 <= len(binding_receipts) <= MAX_CHUNKS
        or any(
            type(item) is not GCSPhysicalObjectReceiptV1 for item in binding_receipts
        )
    ):
        raise PrefixCheckpointV2Error(
            "regional binding receipts must be an exact tuple"
        )
    manifest.validate_for_plan(plan)
    logical_receipts = [
        receipt
        for destination in manifest.destinations
        for receipt in destination.chunks
    ]
    if len(bindings) != len(logical_receipts):
        raise PrefixCheckpointV2Error("regional binding cardinality differs")
    if len(binding_receipts) != len(logical_receipts):
        raise PrefixCheckpointV2Error("regional binding-receipt cardinality differs")
    bindings_by_id = {binding.chunk_id: binding for binding in bindings}
    if len(bindings_by_id) != len(bindings):
        raise PrefixCheckpointV2Error("regional bindings contain duplicate chunk IDs")
    expected_ids = {receipt.chunk_id for receipt in logical_receipts}
    if set(bindings_by_id) != expected_ids:
        raise PrefixCheckpointV2Error("regional bindings are not the exact chunk set")
    binding_receipts_by_path = {
        receipt.object_id: receipt for receipt in binding_receipts
    }
    if len(binding_receipts_by_path) != len(binding_receipts):
        raise PrefixCheckpointV2Error("regional binding receipts are duplicated")
    manifest_raw = manifest.canonical_bytes()
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if (
        manifest_receipt.sha256 != manifest_sha256
        or manifest_receipt.size_bytes != len(manifest_raw)
        or manifest_receipt.object_id
        != content_addressed_record_object_id(manifest_sha256)
        or manifest_receipt.content_type != "application/json"
    ):
        raise PrefixCheckpointV2Error("logical manifest physical receipt is not exact")
    replicas: list[RegionalChunkReplicaV1] = []
    plan_chunks_by_id = {chunk.chunk_id: chunk for chunk in plan.chunks}
    if len(plan_chunks_by_id) != len(plan.chunks):
        raise PrefixCheckpointV2Error("conversion plan chunk IDs collide")
    for logical in logical_receipts:
        binding = bindings_by_id[logical.chunk_id]
        if binding.output_receipt != logical or binding.plan_id != manifest.plan_id:
            raise PrefixCheckpointV2Error(
                "regional binding substitutes logical content"
            )
        chunk = plan_chunks_by_id.get(logical.chunk_id)
        if chunk is None:
            raise PrefixCheckpointV2Error("regional binding references a foreign chunk")
        binding.validate_for(plan, chunk)
        if (
            binding.physical_receipt.store_id != manifest_receipt.store_id
            or binding.physical_receipt.region != manifest_receipt.region
        ):
            raise PrefixCheckpointV2Error("regional binding belongs to another store")
        binding_path = regional_chunk_binding_object_id(
            manifest.plan_id, logical.chunk_id
        )
        binding_receipt = binding_receipts_by_path.get(binding_path)
        binding_raw = binding.canonical_bytes()
        if (
            binding_receipt is None
            or binding_receipt.content_type != "application/json"
            or binding_receipt.sha256 != hashlib.sha256(binding_raw).hexdigest()
            or binding_receipt.size_bytes != len(binding_raw)
            or binding_receipt.store_id != manifest_receipt.store_id
            or binding_receipt.region != manifest_receipt.region
        ):
            raise PrefixCheckpointV2Error(
                "regional binding JSON lacks an exact durable receipt"
            )
        replicas.append(
            RegionalChunkReplicaV1(
                chunk_id=logical.chunk_id,
                binding_receipt=binding_receipt,
            )
        )
    if set(binding_receipts_by_path) != {
        regional_chunk_binding_object_id(manifest.plan_id, receipt.chunk_id)
        for receipt in logical_receipts
    }:
        raise PrefixCheckpointV2Error("regional binding receipts contain foreign paths")
    envelope = RegionalPrefixCheckpointEnvelopeV1(
        logical_manifest_sha256=manifest_sha256,
        logical_manifest_size_bytes=len(manifest_raw),
        plan_id=manifest.plan_id,
        region=manifest_receipt.region,
        store_id=manifest_receipt.store_id,
        manifest_receipt=manifest_receipt,
        chunk_replicas=tuple(replicas),
        total_chunks=manifest.total_chunks,
        converted_payload_bytes=manifest.converted_payload_bytes,
    )
    envelope.validate_for_manifest(manifest)
    return envelope


class GenerationPinnedPrefixObjectReader(Protocol):
    """Narrow no-list reader implemented by the qualified GCS byte store."""

    @property
    def store_id(self) -> str: ...

    @property
    def region(self) -> str: ...

    @property
    def distributed_identity_qualified(self) -> bool: ...

    def read_exact(
        self,
        receipt: GCSPhysicalObjectReceiptV1,
        *,
        offset: int,
        length: int,
        maximum_bytes: int,
    ) -> bytes: ...


def _read_exact_control_record(
    store: GenerationPinnedPrefixObjectReader,
    receipt: GCSPhysicalObjectReceiptV1,
    *,
    maximum_bytes: int,
    name: str,
) -> bytes:
    if receipt.size_bytes > maximum_bytes:
        raise PrefixCheckpointV2Error(f"{name} exceeds its read bound")
    raw = store.read_exact(
        receipt,
        offset=0,
        length=receipt.size_bytes,
        maximum_bytes=maximum_bytes,
    )
    if type(raw) is not bytes:
        raise PrefixCheckpointV2Error(f"{name} reader returned non-bytes")
    if (
        len(raw) != receipt.size_bytes
        or hashlib.sha256(raw).hexdigest() != receipt.sha256
    ):
        raise PrefixCheckpointV2Error(f"{name} bytes differ from exact receipt")
    return raw


_READER_OPEN_CAPABILITY = object()


@dataclass(frozen=True, slots=True, init=False)
class LazyRegionalPrefixCheckpointReaderV1:
    """Two-control-read open followed by touched-chunk-only payload reads."""

    store: GenerationPinnedPrefixObjectReader
    envelope: RegionalPrefixCheckpointEnvelopeV1
    manifest: LogicalPrefixCheckpointManifestV2
    maximum_read_bytes: int
    _opening_envelope_receipt: GCSPhysicalObjectReceiptV1
    _store_id_snapshot: str
    _region_snapshot: str
    _destinations: Mapping[str, LogicalStoredDestinationV2]
    _binding_receipts: Mapping[str, GCSPhysicalObjectReceiptV1]

    def __init__(
        self,
        store: GenerationPinnedPrefixObjectReader,
        envelope: RegionalPrefixCheckpointEnvelopeV1,
        manifest: LogicalPrefixCheckpointManifestV2,
        *,
        envelope_receipt: GCSPhysicalObjectReceiptV1 | None = None,
        maximum_read_bytes: int,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _READER_OPEN_CAPABILITY:
            raise PrefixCheckpointV2Error(
                "regional reader construction is available only through open()"
            )
        if (
            type(envelope) is not RegionalPrefixCheckpointEnvelopeV1
            or type(manifest) is not LogicalPrefixCheckpointManifestV2
            or type(envelope_receipt) is not GCSPhysicalObjectReceiptV1
            or type(maximum_read_bytes) is not int
            or not 1 <= maximum_read_bytes <= MAX_PAYLOAD_READ_BYTES
        ):
            raise PrefixCheckpointV2Error("regional reader state is invalid")
        if (
            store.distributed_identity_qualified is not True
            or store.store_id != envelope.store_id
            or store.region != envelope.region
        ):
            raise PrefixCheckpointV2Error("regional reader store identity is invalid")
        envelope.validate_for_manifest(manifest)
        envelope_raw = envelope.canonical_bytes()
        if (
            manifest.digest != envelope.logical_manifest_sha256
            or envelope_receipt.sha256 != envelope.digest
            or envelope_receipt.size_bytes != len(envelope_raw)
            or envelope_receipt.store_id != envelope.store_id
            or envelope_receipt.region != envelope.region
            or envelope_receipt.content_type != "application/json"
            or envelope_receipt.object_id
            != content_addressed_record_object_id(envelope_receipt.sha256)
        ):
            raise PrefixCheckpointV2Error("regional reader logical identity changed")
        destinations = {item.destination_key: item for item in manifest.destinations}
        binding_receipts = {
            item.chunk_id: item.binding_receipt for item in envelope.chunk_replicas
        }
        if (
            len(destinations) != len(manifest.destinations)
            or len(binding_receipts) != envelope.total_chunks
        ):
            raise PrefixCheckpointV2Error("regional reader indexes are not exact")
        object.__setattr__(self, "store", store)
        object.__setattr__(self, "envelope", envelope)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "maximum_read_bytes", maximum_read_bytes)
        object.__setattr__(self, "_opening_envelope_receipt", envelope_receipt)
        object.__setattr__(self, "_store_id_snapshot", envelope.store_id)
        object.__setattr__(self, "_region_snapshot", envelope.region)
        object.__setattr__(self, "_destinations", MappingProxyType(destinations))
        object.__setattr__(
            self,
            "_binding_receipts",
            MappingProxyType(binding_receipts),
        )

    def _require_live_store_identity(self) -> None:
        if (
            self.store.distributed_identity_qualified is not True
            or self.store.store_id != self._store_id_snapshot
            or self.store.region != self._region_snapshot
        ):
            raise PrefixCheckpointV2Error(
                "qualified regional store identity changed after open"
            )

    @classmethod
    def open(
        cls,
        store: GenerationPinnedPrefixObjectReader,
        envelope_receipt: GCSPhysicalObjectReceiptV1,
        *,
        expected_logical_manifest_sha256: str,
        maximum_control_bytes: int = MAX_REGIONAL_ENVELOPE_BYTES,
        maximum_read_bytes: int = MAX_PAYLOAD_READ_BYTES,
    ) -> Self:
        expected_manifest_sha256 = _sha256(
            expected_logical_manifest_sha256,
            "expected_logical_manifest_sha256",
        )
        if (
            type(maximum_control_bytes) is not int
            or not 1 <= maximum_control_bytes <= MAX_REGIONAL_ENVELOPE_BYTES
            or type(maximum_read_bytes) is not int
            or not 1 <= maximum_read_bytes <= MAX_PAYLOAD_READ_BYTES
        ):
            raise PrefixCheckpointV2Error("reader byte bounds are invalid")
        if store.distributed_identity_qualified is not True:
            raise PrefixCheckpointV2Error(
                "regional reader requires a qualified distributed store identity"
            )
        if type(envelope_receipt) is not GCSPhysicalObjectReceiptV1:
            raise PrefixCheckpointV2Error("reader requires an exact envelope receipt")
        _validate_physical_bounds(
            envelope_receipt,
            maximum_size_bytes=MAX_REGIONAL_ENVELOPE_BYTES,
        )
        if (
            envelope_receipt.store_id != store.store_id
            or envelope_receipt.region != store.region
            or envelope_receipt.content_type != "application/json"
            or envelope_receipt.object_id
            != content_addressed_record_object_id(envelope_receipt.sha256)
        ):
            raise PrefixCheckpointV2Error(
                "envelope receipt does not belong to the qualified exact store"
            )
        envelope_raw = _read_exact_control_record(
            store,
            envelope_receipt,
            maximum_bytes=maximum_control_bytes,
            name="regional envelope",
        )
        if (
            store.distributed_identity_qualified is not True
            or store.store_id != envelope_receipt.store_id
            or store.region != envelope_receipt.region
        ):
            raise PrefixCheckpointV2Error(
                "qualified store identity changed during envelope read"
            )
        try:
            envelope = RegionalPrefixCheckpointEnvelopeV1.from_json(envelope_raw)
        except (ProtocolError, CheckpointConversionError) as error:
            raise PrefixCheckpointV2Error("regional envelope is malformed") from error
        if (
            envelope.digest != envelope_receipt.sha256
            or envelope.store_id != store.store_id
            or envelope.region != store.region
            or envelope.logical_manifest_sha256 != expected_manifest_sha256
        ):
            raise PrefixCheckpointV2Error(
                "regional envelope differs from its authority inputs"
            )
        manifest_receipt = envelope.manifest_receipt
        if manifest_receipt.size_bytes > maximum_control_bytes:
            raise PrefixCheckpointV2Error("logical manifest exceeds the open bound")
        manifest_raw = _read_exact_control_record(
            store,
            manifest_receipt,
            maximum_bytes=maximum_control_bytes,
            name="logical manifest",
        )
        if (
            store.distributed_identity_qualified is not True
            or store.store_id != envelope.store_id
            or store.region != envelope.region
        ):
            raise PrefixCheckpointV2Error(
                "qualified store identity changed during manifest read"
            )
        try:
            manifest = LogicalPrefixCheckpointManifestV2.from_json(manifest_raw)
        except (ProtocolError, CheckpointConversionError) as error:
            raise PrefixCheckpointV2Error("logical manifest is malformed") from error
        if manifest.digest != expected_manifest_sha256:
            raise PrefixCheckpointV2Error("logical manifest digest is unexpected")
        envelope.validate_for_manifest(manifest)
        return cls(
            store,
            envelope,
            manifest,
            envelope_receipt=envelope_receipt,
            maximum_read_bytes=maximum_read_bytes,
            _capability=_READER_OPEN_CAPABILITY,
        )

    def read_flat_region(
        self, destination_key: str, flat_start: int, flat_stop: int
    ) -> np.ndarray:
        _identifier(destination_key, "destination_key")
        if type(flat_start) is not int or type(flat_stop) is not int:
            raise PrefixCheckpointV2Error("flat read bounds must be integers")
        destination = self._destinations.get(destination_key)
        if destination is None:
            raise PrefixCheckpointV2Error("unknown destination key")
        numel = math.prod(destination.shape)
        if flat_start < 0 or flat_stop <= flat_start or flat_stop > numel:
            raise PrefixCheckpointV2Error("flat read interval is invalid")
        dtype = _DTYPES[destination.dtype]
        output_bytes = (flat_stop - flat_start) * dtype.itemsize
        if output_bytes > self.maximum_read_bytes:
            raise PrefixCheckpointV2Error("flat read exceeds configured byte bound")
        self._require_live_store_identity()
        output = np.empty(flat_stop - flat_start, dtype=dtype)
        storage = output.view(np.uint16 if dtype.itemsize == 2 else np.uint32).view(
            np.uint8
        )
        target = memoryview(storage).cast("B")
        try:
            for receipt in destination.chunks:
                left = max(flat_start, receipt.destination_flat_start)
                right = min(flat_stop, receipt.destination_flat_stop)
                if left >= right:
                    continue
                self._require_live_store_identity()
                binding_receipt = self._binding_receipts.get(receipt.chunk_id)
                if binding_receipt is None:
                    raise PrefixCheckpointV2Error(
                        "regional binding-receipt map is incomplete"
                    )
                if (
                    binding_receipt.store_id != self._store_id_snapshot
                    or binding_receipt.region != self._region_snapshot
                    or binding_receipt.object_id
                    != regional_chunk_binding_object_id(
                        self.manifest.plan_id, receipt.chunk_id
                    )
                    or binding_receipt.content_type != "application/json"
                    or binding_receipt.size_bytes > MAX_BINDING_BYTES
                ):
                    raise PrefixCheckpointV2Error(
                        "regional binding receipt changed after open"
                    )
                binding_raw = _read_exact_control_record(
                    self.store,
                    binding_receipt,
                    maximum_bytes=MAX_BINDING_BYTES,
                    name="regional chunk binding",
                )
                self._require_live_store_identity()
                try:
                    binding = RegionalPrefixChunkBindingV1.from_json(binding_raw)
                except (ProtocolError, CheckpointConversionError) as error:
                    raise PrefixCheckpointV2Error(
                        "regional chunk binding is malformed"
                    ) from error
                physical = binding.physical_receipt
                if (
                    binding.plan_id != self.manifest.plan_id
                    or binding.chunk_id != receipt.chunk_id
                    or binding.output_receipt != receipt
                    or any(
                        item.source_adapter_identity
                        != self.manifest.source_adapter_identity
                        for item in binding.source_lineage
                    )
                    or physical.object_id != receipt.immutable_object_id
                    or physical.sha256 != receipt.output_sha256
                    or physical.size_bytes != receipt.output_size_bytes
                    or physical.store_id != self._store_id_snapshot
                    or physical.region != self._region_snapshot
                ):
                    raise PrefixCheckpointV2Error(
                        "regional binding substitutes logical or physical content"
                    )
                byte_offset = (left - receipt.destination_flat_start) * dtype.itemsize
                byte_length = (right - left) * dtype.itemsize
                self._require_live_store_identity()
                raw = self.store.read_exact(
                    physical,
                    offset=byte_offset,
                    length=byte_length,
                    maximum_bytes=self.maximum_read_bytes,
                )
                self._require_live_store_identity()
                if type(raw) is not bytes or len(raw) != byte_length:
                    raise PrefixCheckpointV2Error(
                        "payload reader returned an inexact byte interval"
                    )
                target_start = (left - flat_start) * dtype.itemsize
                target[target_start : target_start + byte_length] = raw
            return output
        finally:
            target.release()


__all__ = [
    "EXTERNAL_QUALIFICATION_REQUIRED",
    "GenerationPinnedPrefixObjectReader",
    "LOGICAL_IDENTITY_SCOPE",
    "LOGICAL_TENSOR_TREE_CONTRACT",
    "LazyRegionalPrefixCheckpointReaderV1",
    "LogicalPrefixCheckpointManifestV2",
    "LogicalStoredDestinationV2",
    "MAX_BINDING_BYTES",
    "MAX_CHUNKS",
    "MAX_DESTINATIONS",
    "MAX_LOGICAL_MANIFEST_BYTES",
    "MAX_REGIONAL_ENVELOPE_BYTES",
    "MAX_SOURCE_LOGICAL_TENSORS",
    "OBJECT_DISCOVERY_CONTRACT",
    "PINNED_CONVERTED_PARAMETERS",
    "PINNED_CONVERTED_PAYLOAD_BYTES",
    "PINNED_DESTINATION_TENSORS",
    "PINNED_SOURCE_ENTRIES",
    "PINNED_SOURCE_LOGICAL_TENSORS",
    "PINNED_SOURCE_SHARDS",
    "PINNED_TARGET_LAYER_ONE_INDEXED",
    "PINNED_TOTAL_CHUNKS",
    "PRODUCTION_PREFIX_SINK_ID",
    "PrefixCheckpointV2Error",
    "REGIONAL_ENVELOPE_CONTRACT",
    "RegionalChunkReplicaV1",
    "RegionalPrefixCheckpointEnvelopeV1",
    "RegionalPrefixChunkBindingV1",
    "SOURCE_LINEAGE_CONTRACT",
    "TARGET_SEMANTICS",
    "build_logical_prefix_checkpoint_manifest_v2",
    "build_regional_prefix_chunk_binding_v1",
    "build_regional_prefix_checkpoint_envelope_v1",
    "content_addressed_record_object_id",
    "regional_chunk_binding_object_id",
]
