"""Fail-closed, content-bound MFU evidence for the R1 prefix harvester.

V1 protects the numerical gate from caller-supplied weaker thresholds, but its
measurement fields are intentionally retained for compatibility.  This V2
module is a parallel evidence path: useful FLOPs, rates, MFU, percentiles, and
instrumentation deltas are derived from immutable inputs rather than accepted
as producer scalars.

Large artifacts (HLO, Xprof, packed segment ledgers, and host traces) remain
content-addressed binary objects.  Canonical JSON contains their
:class:`~moe_sae.r1.protocol.ContentRefV1` identities plus compact segment
histograms and per-step integer nanosecond summaries; it never contains a
per-token ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
from fractions import Fraction
import math
from typing import Any, ClassVar, Mapping, Protocol as TypingProtocol, Self, Sequence

from .image_protocol import DeviceInventoryV1
from .mfu import (
    V6E64_BF16_PEAK_FLOPS,
    V6E64_CHIPS,
    V6E_CHIP_BF16_PEAK_FLOPS,
)
from .flops import (
    DEEPSEEK_R1_PREFIX_FORMULA_SHA256,
    DEEPSEEK_R1_PREFIX_FORMULA_VERSION,
    DEEPSEEK_R1_PREFIX_GEOMETRY_V1,
    deepseek_r1_fp32_router_flops,
    deepseek_r1_prefix_flop_terms,
)
from .protocol import (
    CanonicalRecord,
    ContentRefV1,
    IDENTIFIER_PATTERN,
    ProtocolError,
    SHA256_PATTERN,
    canonical_json_bytes,
    parse_json,
    require_integer as _integer,
    require_record_object as _strict_object,
    require_string as _string,
)


V6E64_CHIPS_V2 = V6E64_CHIPS
V6E_CHIP_BF16_PEAK_FLOPS_V2 = V6E_CHIP_BF16_PEAK_FLOPS
V6E64_BF16_PEAK_FLOPS_V2 = V6E64_BF16_PEAK_FLOPS
V6E_CHIP_HBM_BYTES_V2 = 32_000_000_000
PREFIX_SEQUENCE_LENGTH_V2 = 1_024
PREFIX_SELECTED_EXPERTS_V2 = DEEPSEEK_R1_PREFIX_GEOMETRY_V1.selected_experts
PREFIX_ROUTER_INVOCATIONS_V2 = DEEPSEEK_R1_PREFIX_GEOMETRY_V1.completed_moe_blocks + 1
PREFIX_MINIMUM_WINDOWS_V2 = 3
PREFIX_MINIMUM_STEPS_PER_WINDOW_V2 = 500
PREFIX_MINIMUM_WALL_NS_V2 = 60_000_000_000
PREFIX_MINIMUM_WARMUP_STEPS_V2 = 5

_REAL = "REAL"
_SYNTHETIC = "SYNTHETIC"
_BASELINE = "BASELINE"
_INSTRUMENTED = "INSTRUMENTED"
_DATA_KINDS = (_REAL, _SYNTHETIC)
_INSTRUMENTATION_KINDS = (_BASELINE, _INSTRUMENTED)

MEDIA_SCIENTIFIC_IDENTITY_V2 = "application/vnd.moe-sae.r1.scientific-identity.v1+json"
MEDIA_IMPLEMENTATION_IDENTITY_V2 = (
    "application/vnd.moe-sae.r1.implementation-identity.v1+json"
)
MEDIA_DEPLOYMENT_IDENTITY_V2 = "application/vnd.moe-sae.r1.deployment-identity.v1+json"
MEDIA_DEVICE_INVENTORY_V2 = "application/vnd.moe-sae.r1.device-inventory.v1+json"
MEDIA_TENSOR_INVENTORY_V2 = "application/vnd.moe-sae.r1.tensor-inventory.v1+json"
MEDIA_EXECUTABLE_V2 = "application/vnd.moe-sae.r1.xla-executable.v1+binary"
MEDIA_HLO_V2 = "application/vnd.moe-sae.r1.hlo-module.v1+binary"
MEDIA_FLOP_FORMULA_V2 = "application/vnd.moe-sae.r1.prefix-flop-formula.v2+json"
MEDIA_OPERATION_INVENTORY_V2 = (
    "application/vnd.moe-sae.r1.hlo-operation-inventory.v2+json"
)
MEDIA_INPUT_MANIFEST_V2 = "application/vnd.moe-sae.r1.packed-input-manifest.v2+json"
MEDIA_SEGMENT_LEDGER_V2 = "application/vnd.moe-sae.r1.packed-segment-ledger.v2+binary"
MEDIA_STEP_TRACE_V2 = "application/vnd.moe-sae.r1.step-trace.v2+binary"
MEDIA_TIMING_BLOCKS_V2 = "application/vnd.moe-sae.r1.timing-blocks.v2+binary"
MEDIA_PROFILER_TRACE_V2 = "application/vnd.moe-sae.r1.xprof-trace.v2+binary"
MEDIA_HOST_METRICS_V2 = "application/vnd.moe-sae.r1.host-metrics.v2+binary"
MEDIA_INTEGRITY_EVENTS_V2 = "application/vnd.moe-sae.r1.integrity-events.v2+json"
MEDIA_ROUTING_CAPACITY_LEDGER_V2 = (
    "application/vnd.moe-sae.r1.routing-capacity-ledger.v2+binary"
)
MEDIA_PERSISTENCE_RECEIPT_V2 = "application/vnd.moe-sae.r1.persistence-receipt.v2+json"
MEDIA_SEMANTIC_AUTHORITY_V2 = (
    "application/vnd.moe-sae.r1.mfu-semantic-authority.v2+json"
)
MEDIA_PREDECLARATION_RECEIPT_V2 = (
    "application/vnd.moe-sae.r1.mfu-predeclaration-receipt.v2+json"
)
MEDIA_WARMUP_RECEIPT_V2 = (
    "application/vnd.moe-sae.r1.mfu-compile-warmup-receipt.v2+json"
)
MEDIA_FORMULA_RECONCILIATION_RECEIPT_V2 = (
    "application/vnd.moe-sae.r1.mfu-formula-reconciliation.v2+json"
)
MEDIA_INPUT_REDUCTION_RECEIPT_V2 = (
    "application/vnd.moe-sae.r1.mfu-input-reduction.v2+json"
)
MEDIA_WINDOW_REDUCTION_RECEIPT_V2 = (
    "application/vnd.moe-sae.r1.mfu-window-reduction.v2+json"
)


class MFUV2ContractError(ProtocolError):
    """V2 evidence is malformed, inconsistent, unresolved, or below gate."""


def _content(value: Any, name: str, media_type: str) -> ContentRefV1:
    if not isinstance(value, ContentRefV1):
        raise MFUV2ContractError(f"{name} must be ContentRefV1")
    if value.media_type != media_type:
        raise MFUV2ContractError(f"{name} media_type must be exactly {media_type!r}")
    return value


def _sha(value: Any, name: str) -> str:
    return _string(value, name, pattern=SHA256_PATTERN)


def _identifier(value: Any, name: str) -> str:
    return _string(value, name, pattern=IDENTIFIER_PATTERN)


def _int_tuple(
    value: Any,
    name: str,
    *,
    length: int | None = None,
    minimum: int = 0,
    nonempty: bool = True,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise MFUV2ContractError(f"{name} must be an array")
    result = tuple(_integer(item, f"{name}[]", minimum=minimum) for item in value)
    if nonempty and not result:
        raise MFUV2ContractError(f"{name} must not be empty")
    if length is not None and len(result) != length:
        raise MFUV2ContractError(f"{name} must contain exactly {length} entries")
    return result


def _record_ref_matches(
    reference: ContentRefV1,
    record: CanonicalRecord,
    *,
    name: str,
) -> None:
    if reference.sha256 != record.digest:
        raise MFUV2ContractError(f"{name} SHA-256 does not match canonical bytes")
    if reference.size_bytes != len(record.canonical_bytes()):
        raise MFUV2ContractError(f"{name} size does not match canonical bytes")


def _same_payload(left: ContentRefV1, right: ContentRefV1) -> bool:
    return (
        left.sha256,
        left.size_bytes,
        left.media_type,
    ) == (
        right.sha256,
        right.size_bytes,
        right.media_type,
    )


def _wire_value(value: Any) -> Any:
    if isinstance(value, CanonicalRecord):
        return value.as_dict()
    if isinstance(value, tuple):
        return [_wire_value(item) for item in value]
    if value is None or isinstance(value, (str, int)):
        return value
    raise MFUV2ContractError(f"unsupported record field value: {type(value).__name__}")


def _record_dict(record: CanonicalRecord) -> dict[str, Any]:
    return {
        "schema": record.SCHEMA,
        **{
            item.name: _wire_value(getattr(record, item.name))
            for item in dataclass_fields(record)
            if item.init and not item.name.startswith("_")
        },
    }


def _record_object(value: Any, record_type: type[CanonicalRecord]) -> Mapping[str, Any]:
    names = tuple(
        item.name
        for item in dataclass_fields(record_type)
        if item.init and not item.name.startswith("_")
    )
    return _strict_object(value, schema=record_type.SCHEMA, fields=names)


def _nearest_rank(values: tuple[int, ...], percentile: int) -> int:
    ordered = sorted(values)
    index = math.ceil(percentile * len(ordered) / 100) - 1
    return ordered[index]


@dataclass(frozen=True, slots=True)
class PrefixFlopFormulaV2(CanonicalRecord):
    """Exact BF16 useful-FLOP formula reconciled to one executable/HLO."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.prefix-flop-formula.v2"

    tensor_inventory: ContentRefV1
    executable: ContentRefV1
    hlo: ContentRefV1
    counted_operation_inventory: ContentRefV1
    sequences_per_step: int
    sequence_length: int
    counted_matmul_dtype: str
    denominator_dtype: str
    fp32_router_policy: str
    analytic_formula_version: str
    analytic_formula_sha256: str
    reducer_version: str

    def __post_init__(self) -> None:
        _content(
            self.tensor_inventory,
            "tensor_inventory",
            MEDIA_TENSOR_INVENTORY_V2,
        )
        _content(self.executable, "executable", MEDIA_EXECUTABLE_V2)
        _content(self.hlo, "hlo", MEDIA_HLO_V2)
        _content(
            self.counted_operation_inventory,
            "counted_operation_inventory",
            MEDIA_OPERATION_INVENTORY_V2,
        )
        _integer(self.sequences_per_step, "sequences_per_step", minimum=1)
        if self.sequence_length != PREFIX_SEQUENCE_LENGTH_V2:
            raise MFUV2ContractError("sequence_length must be exactly 1024")
        if self.counted_matmul_dtype != "bfloat16":
            raise MFUV2ContractError("counted_matmul_dtype must be bfloat16")
        if self.denominator_dtype != "bfloat16":
            raise MFUV2ContractError("denominator_dtype must be bfloat16")
        if self.fp32_router_policy != "FP32_ROUTER_EXCLUDED_FROM_BF16_NUMERATOR":
            raise MFUV2ContractError(
                "FP32 router projections must be excluded from the BF16 numerator"
            )
        if self.analytic_formula_version != DEEPSEEK_R1_PREFIX_FORMULA_VERSION:
            raise MFUV2ContractError("analytic FLOP formula version is not pinned")
        if self.analytic_formula_sha256 != DEEPSEEK_R1_PREFIX_FORMULA_SHA256:
            raise MFUV2ContractError("analytic FLOP formula digest is not pinned")
        if self.reducer_version != "deepseek-r1-prefix-bf16-v2":
            raise MFUV2ContractError("unsupported prefix FLOP reducer version")

    @classmethod
    def build(
        cls,
        *,
        tensor_inventory: ContentRefV1,
        executable: ContentRefV1,
        hlo: ContentRefV1,
        counted_operation_inventory: ContentRefV1,
        sequences_per_step: int,
    ) -> Self:
        return cls(
            tensor_inventory=tensor_inventory,
            executable=executable,
            hlo=hlo,
            counted_operation_inventory=counted_operation_inventory,
            sequences_per_step=sequences_per_step,
            sequence_length=PREFIX_SEQUENCE_LENGTH_V2,
            counted_matmul_dtype="bfloat16",
            denominator_dtype="bfloat16",
            fp32_router_policy="FP32_ROUTER_EXCLUDED_FROM_BF16_NUMERATOR",
            analytic_formula_version=DEEPSEEK_R1_PREFIX_FORMULA_VERSION,
            analytic_formula_sha256=DEEPSEEK_R1_PREFIX_FORMULA_SHA256,
            reducer_version="deepseek-r1-prefix-bf16-v2",
        )

    def useful_flops(self, histogram: PackedSegmentHistogramV2) -> int:
        if not isinstance(histogram, PackedSegmentHistogramV2):
            raise MFUV2ContractError("histogram must be PackedSegmentHistogramV2")
        return sum(
            term.useful_flops
            for term in deepseek_r1_prefix_flop_terms(
                valid_tokens=histogram.valid_tokens,
                causal_pairs=histogram.causal_pairs,
            )
        )

    def excluded_fp32_router_flops(self, histogram: PackedSegmentHistogramV2) -> int:
        if not isinstance(histogram, PackedSegmentHistogramV2):
            raise MFUV2ContractError("histogram must be PackedSegmentHistogramV2")
        return deepseek_r1_fp32_router_flops(valid_tokens=histogram.valid_tokens)

    def as_dict(self) -> dict[str, Any]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _record_object(value, cls)
        return cls(
            tensor_inventory=ContentRefV1.from_dict(obj["tensor_inventory"]),
            executable=ContentRefV1.from_dict(obj["executable"]),
            hlo=ContentRefV1.from_dict(obj["hlo"]),
            counted_operation_inventory=ContentRefV1.from_dict(
                obj["counted_operation_inventory"]
            ),
            sequences_per_step=_integer(
                obj["sequences_per_step"], "sequences_per_step", minimum=1
            ),
            sequence_length=_integer(
                obj["sequence_length"], "sequence_length", minimum=1
            ),
            counted_matmul_dtype=_string(
                obj["counted_matmul_dtype"], "counted_matmul_dtype"
            ),
            denominator_dtype=_string(obj["denominator_dtype"], "denominator_dtype"),
            fp32_router_policy=_string(obj["fp32_router_policy"], "fp32_router_policy"),
            analytic_formula_version=_string(
                obj["analytic_formula_version"], "analytic_formula_version"
            ),
            analytic_formula_sha256=_sha(
                obj["analytic_formula_sha256"], "analytic_formula_sha256"
            ),
            reducer_version=_string(obj["reducer_version"], "reducer_version"),
        )


@dataclass(frozen=True, slots=True, order=True)
class SegmentLengthBinV2(CanonicalRecord):
    """One nonzero bin in the packed-segment length histogram."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.segment-length-bin.v2"

    length: int
    count: int

    def __post_init__(self) -> None:
        _integer(
            self.length,
            "length",
            minimum=1,
            maximum=PREFIX_SEQUENCE_LENGTH_V2,
        )
        _integer(self.count, "count", minimum=1)

    def as_dict(self) -> dict[str, Any]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _record_object(value, cls)
        return cls(
            length=_integer(
                obj["length"],
                "length",
                minimum=1,
                maximum=PREFIX_SEQUENCE_LENGTH_V2,
            ),
            count=_integer(obj["count"], "count", minimum=1),
        )


@dataclass(frozen=True, slots=True)
class PackedSegmentHistogramV2(CanonicalRecord):
    """Compact exact derivation of valid tokens and causal attention pairs."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.packed-segment-histogram.v2"

    bins: tuple[SegmentLengthBinV2, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.bins, (list, tuple)):
            raise MFUV2ContractError("bins must be an array")
        bins = tuple(self.bins)
        if not bins or not all(isinstance(item, SegmentLengthBinV2) for item in bins):
            raise MFUV2ContractError(
                "bins must contain at least one SegmentLengthBinV2"
            )
        lengths = tuple(item.length for item in bins)
        if lengths != tuple(sorted(lengths)) or len(set(lengths)) != len(lengths):
            raise MFUV2ContractError("segment bins must be unique and sorted")
        object.__setattr__(self, "bins", bins)

    @property
    def segment_count(self) -> int:
        return sum(item.count for item in self.bins)

    @property
    def valid_tokens(self) -> int:
        return sum(item.length * item.count for item in self.bins)

    @property
    def causal_pairs(self) -> int:
        return sum(
            item.length * (item.length + 1) // 2 * item.count for item in self.bins
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "bins": [item.as_dict() for item in self.bins],
            "segment_count": self.segment_count,
            "valid_tokens": self.valid_tokens,
            "causal_pairs": self.causal_pairs,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _strict_object(
            value,
            schema=cls.SCHEMA,
            fields=("bins", "segment_count", "valid_tokens", "causal_pairs"),
        )
        raw_bins = obj["bins"]
        if not isinstance(raw_bins, list):
            raise MFUV2ContractError("bins must be an array")
        result = cls(tuple(SegmentLengthBinV2.from_dict(item) for item in raw_bins))
        expected = {
            "segment_count": result.segment_count,
            "valid_tokens": result.valid_tokens,
            "causal_pairs": result.causal_pairs,
        }
        for field_name, expected_value in expected.items():
            if _integer(obj[field_name], field_name, minimum=1) != expected_value:
                raise MFUV2ContractError(f"{field_name} is not derived from histogram")
        return result


@dataclass(frozen=True, slots=True)
class PackedInputSliceV2(CanonicalRecord):
    """One predeclared input interval and its independently reducible layout."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.packed-input-slice.v2"

    group_index: int
    data_kind: str
    parent_input_manifest_sha256: str
    batch_start: int
    batch_end: int
    packed_sequence_count: int
    histogram: PackedSegmentHistogramV2
    segment_ledger: ContentRefV1

    def __post_init__(self) -> None:
        _integer(
            self.group_index,
            "group_index",
            minimum=0,
            maximum=PREFIX_MINIMUM_WINDOWS_V2 - 1,
        )
        if self.data_kind not in _DATA_KINDS:
            raise MFUV2ContractError("data_kind must be REAL or SYNTHETIC")
        _sha(self.parent_input_manifest_sha256, "parent_input_manifest_sha256")
        _integer(self.batch_start, "batch_start", minimum=0)
        _integer(self.batch_end, "batch_end", minimum=1)
        if self.batch_end <= self.batch_start:
            raise MFUV2ContractError("input batch interval must be non-empty")
        _integer(self.packed_sequence_count, "packed_sequence_count", minimum=1)
        if not isinstance(self.histogram, PackedSegmentHistogramV2):
            raise MFUV2ContractError("histogram must be PackedSegmentHistogramV2")
        if (
            self.histogram.valid_tokens
            > self.packed_sequence_count * PREFIX_SEQUENCE_LENGTH_V2
        ):
            raise MFUV2ContractError("valid tokens exceed packed sequence capacity")
        _content(
            self.segment_ledger,
            "segment_ledger",
            MEDIA_SEGMENT_LEDGER_V2,
        )

    def as_dict(self) -> dict[str, Any]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _record_object(value, cls)
        return cls(
            group_index=_integer(
                obj["group_index"],
                "group_index",
                minimum=0,
                maximum=PREFIX_MINIMUM_WINDOWS_V2 - 1,
            ),
            data_kind=_string(obj["data_kind"], "data_kind"),
            parent_input_manifest_sha256=_sha(
                obj["parent_input_manifest_sha256"],
                "parent_input_manifest_sha256",
            ),
            batch_start=_integer(obj["batch_start"], "batch_start", minimum=0),
            batch_end=_integer(obj["batch_end"], "batch_end", minimum=1),
            packed_sequence_count=_integer(
                obj["packed_sequence_count"],
                "packed_sequence_count",
                minimum=1,
            ),
            histogram=PackedSegmentHistogramV2.from_dict(obj["histogram"]),
            segment_ledger=ContentRefV1.from_dict(obj["segment_ledger"]),
        )


@dataclass(frozen=True, slots=True)
class MFUWindowPlanEntryV2(CanonicalRecord):
    """One predeclared measurement interval and immutable input slice."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.mfu-window-plan-entry.v2"

    window_id: str
    group_index: int
    data_kind: str
    instrumentation_kind: str
    step_start: int
    step_end: int
    input_slice_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.window_id, "window_id")
        _integer(
            self.group_index,
            "group_index",
            minimum=0,
            maximum=PREFIX_MINIMUM_WINDOWS_V2 - 1,
        )
        if self.data_kind not in _DATA_KINDS:
            raise MFUV2ContractError("data_kind must be REAL or SYNTHETIC")
        if self.instrumentation_kind not in _INSTRUMENTATION_KINDS:
            raise MFUV2ContractError(
                "instrumentation_kind must be BASELINE or INSTRUMENTED"
            )
        _integer(self.step_start, "step_start", minimum=0)
        _integer(self.step_end, "step_end", minimum=1)
        if self.step_end - self.step_start < PREFIX_MINIMUM_STEPS_PER_WINDOW_V2:
            raise MFUV2ContractError("planned windows require at least 500 steps")
        _sha(self.input_slice_sha256, "input_slice_sha256")

    @property
    def measured_steps(self) -> int:
        return self.step_end - self.step_start

    @property
    def ordering_key(self) -> tuple[int, int, int]:
        return (
            self.group_index,
            _DATA_KINDS.index(self.data_kind),
            _INSTRUMENTATION_KINDS.index(self.instrumentation_kind),
        )

    def as_dict(self) -> dict[str, Any]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _record_object(value, cls)
        return cls(
            window_id=_identifier(obj["window_id"], "window_id"),
            group_index=_integer(obj["group_index"], "group_index", minimum=0),
            data_kind=_string(obj["data_kind"], "data_kind"),
            instrumentation_kind=_string(
                obj["instrumentation_kind"], "instrumentation_kind"
            ),
            step_start=_integer(obj["step_start"], "step_start", minimum=0),
            step_end=_integer(obj["step_end"], "step_end", minimum=1),
            input_slice_sha256=_sha(obj["input_slice_sha256"], "input_slice_sha256"),
        )


@dataclass(frozen=True, slots=True)
class PrefixMFUWindowPlanV2(CanonicalRecord):
    """Exactly three predeclared 2x2 real/synthetic/instrumentation groups."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.prefix-mfu-window-plan.v2"

    input_slices: tuple[PackedInputSliceV2, ...]
    entries: tuple[MFUWindowPlanEntryV2, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.input_slices, (list, tuple)):
            raise MFUV2ContractError("input_slices must be an array")
        slices = tuple(self.input_slices)
        if len(slices) != PREFIX_MINIMUM_WINDOWS_V2 * len(_DATA_KINDS) or not all(
            isinstance(item, PackedInputSliceV2) for item in slices
        ):
            raise MFUV2ContractError("input_slices must contain exactly six slices")
        expected_slice_order = tuple(
            sorted(
                slices,
                key=lambda item: (
                    item.group_index,
                    _DATA_KINDS.index(item.data_kind),
                ),
            )
        )
        if slices != expected_slice_order:
            raise MFUV2ContractError("input_slices must be in canonical order")
        slice_keys = {(item.group_index, item.data_kind) for item in slices}
        expected_slice_keys = {
            (group, kind)
            for group in range(PREFIX_MINIMUM_WINDOWS_V2)
            for kind in _DATA_KINDS
        }
        if slice_keys != expected_slice_keys:
            raise MFUV2ContractError("input_slices do not cover every planned group")
        slice_digests = [item.digest for item in slices]
        if len(set(slice_digests)) != len(slice_digests):
            raise MFUV2ContractError("input slice identities must be unique")
        ledger_digests = [item.segment_ledger.sha256 for item in slices]
        if len(set(ledger_digests)) != len(ledger_digests):
            raise MFUV2ContractError("segment ledgers must be unique per input slice")

        if not isinstance(self.entries, (list, tuple)):
            raise MFUV2ContractError("entries must be an array")
        entries = tuple(self.entries)
        expected_count = (
            PREFIX_MINIMUM_WINDOWS_V2 * len(_DATA_KINDS) * len(_INSTRUMENTATION_KINDS)
        )
        if len(entries) != expected_count or not all(
            isinstance(item, MFUWindowPlanEntryV2) for item in entries
        ):
            raise MFUV2ContractError("entries must contain exactly twelve windows")
        if tuple(item.ordering_key for item in entries) != tuple(
            sorted(item.ordering_key for item in entries)
        ):
            raise MFUV2ContractError("window plan entries must be in canonical order")
        expected_entry_keys = {
            (group, data, instrumentation)
            for group in range(PREFIX_MINIMUM_WINDOWS_V2)
            for data in _DATA_KINDS
            for instrumentation in _INSTRUMENTATION_KINDS
        }
        entry_keys = {
            (item.group_index, item.data_kind, item.instrumentation_kind)
            for item in entries
        }
        if entry_keys != expected_entry_keys:
            raise MFUV2ContractError("window plan does not cover the exact 3x2x2 grid")
        ids = [item.window_id for item in entries]
        if len(set(ids)) != len(ids):
            raise MFUV2ContractError("window IDs must be unique")
        by_slice_key = {(item.group_index, item.data_kind): item for item in slices}
        for entry in entries:
            input_slice = by_slice_key[(entry.group_index, entry.data_kind)]
            if entry.input_slice_sha256 != input_slice.digest:
                raise MFUV2ContractError("plan entry names the wrong input slice")
            if entry.measured_steps != input_slice.batch_end - input_slice.batch_start:
                raise MFUV2ContractError(
                    "planned step count must equal input batch count"
                )
        intervals = [(item.step_start, item.step_end) for item in entries]
        if intervals != sorted(intervals):
            raise MFUV2ContractError(
                "measurement intervals must follow canonical plan order"
            )
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                raise MFUV2ContractError("planned measurement step ranges overlap")
        for data_kind in _DATA_KINDS:
            ranges = [
                (item.batch_start, item.batch_end)
                for item in slices
                if item.data_kind == data_kind
            ]
            if ranges != sorted(ranges):
                raise MFUV2ContractError(
                    "input ranges must follow canonical group order"
                )
            for previous, current in zip(ranges, ranges[1:]):
                if current[0] < previous[1]:
                    raise MFUV2ContractError("planned input slice ranges overlap")
        for group in range(PREFIX_MINIMUM_WINDOWS_V2):
            real = by_slice_key[(group, _REAL)]
            synthetic = by_slice_key[(group, _SYNTHETIC)]
            if (
                real.histogram != synthetic.histogram
                or real.packed_sequence_count != synthetic.packed_sequence_count
            ):
                raise MFUV2ContractError(
                    "real and synthetic slices must be exactly density matched"
                )
        object.__setattr__(self, "input_slices", slices)
        object.__setattr__(self, "entries", entries)

    def entry(self, window_id: str) -> MFUWindowPlanEntryV2:
        for item in self.entries:
            if item.window_id == window_id:
                return item
        raise MFUV2ContractError(f"window {window_id!r} is not predeclared")

    def input_slice(self, sha256: str) -> PackedInputSliceV2:
        for item in self.input_slices:
            if item.digest == sha256:
                return item
        raise MFUV2ContractError("input slice is not in the predeclared plan")

    def as_dict(self) -> dict[str, Any]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _record_object(value, cls)
        raw_slices = obj["input_slices"]
        raw_entries = obj["entries"]
        if not isinstance(raw_slices, list) or not isinstance(raw_entries, list):
            raise MFUV2ContractError("window plan arrays must be JSON arrays")
        return cls(
            input_slices=tuple(PackedInputSliceV2.from_dict(x) for x in raw_slices),
            entries=tuple(MFUWindowPlanEntryV2.from_dict(x) for x in raw_entries),
        )


@dataclass(frozen=True, slots=True)
class PrefixMFURunContractV2(CanonicalRecord):
    """Immutable identities, denominator, inputs, formula, and window plan."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.prefix-mfu-run-contract.v2"

    attempt_id: str
    scientific_identity: ContentRefV1
    implementation_identity: ContentRefV1
    deployment_identity: ContentRefV1
    semantic_authority: ContentRefV1
    device_inventory: ContentRefV1
    tensor_inventory: ContentRefV1
    executable: ContentRefV1
    hlo: ContentRefV1
    flop_formula: ContentRefV1
    real_input_manifest: ContentRefV1
    synthetic_input_manifest: ContentRefV1
    window_plan: PrefixMFUWindowPlanV2
    sequences_per_step: int
    warmup_steps: int
    accelerator_type: str
    topology: str
    physical_chips: int
    denominator_dtype: str
    peak_flops_per_chip: int

    def __post_init__(self) -> None:
        _identifier(self.attempt_id, "attempt_id")
        roles = (
            (
                self.scientific_identity,
                "scientific_identity",
                MEDIA_SCIENTIFIC_IDENTITY_V2,
            ),
            (
                self.implementation_identity,
                "implementation_identity",
                MEDIA_IMPLEMENTATION_IDENTITY_V2,
            ),
            (
                self.deployment_identity,
                "deployment_identity",
                MEDIA_DEPLOYMENT_IDENTITY_V2,
            ),
            (
                self.semantic_authority,
                "semantic_authority",
                MEDIA_SEMANTIC_AUTHORITY_V2,
            ),
            (
                self.device_inventory,
                "device_inventory",
                MEDIA_DEVICE_INVENTORY_V2,
            ),
            (
                self.tensor_inventory,
                "tensor_inventory",
                MEDIA_TENSOR_INVENTORY_V2,
            ),
            (self.executable, "executable", MEDIA_EXECUTABLE_V2),
            (self.hlo, "hlo", MEDIA_HLO_V2),
            (self.flop_formula, "flop_formula", MEDIA_FLOP_FORMULA_V2),
            (
                self.real_input_manifest,
                "real_input_manifest",
                MEDIA_INPUT_MANIFEST_V2,
            ),
            (
                self.synthetic_input_manifest,
                "synthetic_input_manifest",
                MEDIA_INPUT_MANIFEST_V2,
            ),
        )
        for reference, name, media_type in roles:
            _content(reference, name, media_type)
        role_hashes = [reference.sha256 for reference, _name, _media in roles]
        if len(set(role_hashes)) != len(role_hashes):
            raise MFUV2ContractError("run-contract role content must be distinct")
        if not isinstance(self.window_plan, PrefixMFUWindowPlanV2):
            raise MFUV2ContractError("window_plan must be PrefixMFUWindowPlanV2")
        _integer(self.sequences_per_step, "sequences_per_step", minimum=1)
        _integer(
            self.warmup_steps,
            "warmup_steps",
            minimum=PREFIX_MINIMUM_WARMUP_STEPS_V2,
        )
        if self.accelerator_type != "v6e-64":
            raise MFUV2ContractError("accelerator_type must be v6e-64")
        if self.topology != "8x8":
            raise MFUV2ContractError("topology must be 8x8")
        if self.physical_chips != V6E64_CHIPS_V2:
            raise MFUV2ContractError("physical_chips must be exactly 64")
        if self.denominator_dtype != "bfloat16":
            raise MFUV2ContractError("denominator_dtype must be bfloat16")
        if self.peak_flops_per_chip != V6E_CHIP_BF16_PEAK_FLOPS_V2:
            raise MFUV2ContractError("peak_flops_per_chip must be the v6e BF16 peak")
        expected_parent = {
            _REAL: self.real_input_manifest.sha256,
            _SYNTHETIC: self.synthetic_input_manifest.sha256,
        }
        for input_slice in self.window_plan.input_slices:
            if (
                input_slice.parent_input_manifest_sha256
                != expected_parent[input_slice.data_kind]
            ):
                raise MFUV2ContractError("input slice is bound to the wrong manifest")
            expected_sequences = (
                input_slice.batch_end - input_slice.batch_start
            ) * self.sequences_per_step
            if input_slice.packed_sequence_count != expected_sequences:
                raise MFUV2ContractError(
                    "packed_sequence_count does not match static step shape"
                )
        if any(
            entry.step_start < self.warmup_steps for entry in self.window_plan.entries
        ):
            raise MFUV2ContractError(
                "every measurement window must begin after declared warmup"
            )

    def verify_dependencies(
        self,
        formula: PrefixFlopFormulaV2,
        device_inventory: DeviceInventoryV1,
    ) -> None:
        if not isinstance(formula, PrefixFlopFormulaV2):
            raise MFUV2ContractError("formula must be PrefixFlopFormulaV2")
        if not isinstance(device_inventory, DeviceInventoryV1):
            raise MFUV2ContractError("device_inventory must be DeviceInventoryV1")
        _record_ref_matches(self.flop_formula, formula, name="flop_formula")
        _record_ref_matches(
            self.device_inventory, device_inventory, name="device_inventory"
        )
        if not _same_payload(formula.tensor_inventory, self.tensor_inventory):
            raise MFUV2ContractError("formula tensor inventory does not match run")
        if not _same_payload(formula.executable, self.executable):
            raise MFUV2ContractError("formula executable does not match run")
        if not _same_payload(formula.hlo, self.hlo):
            raise MFUV2ContractError("formula HLO does not match run")
        if formula.sequences_per_step != self.sequences_per_step:
            raise MFUV2ContractError("formula static batch shape does not match run")
        if device_inventory.device_count != V6E64_CHIPS_V2:
            raise MFUV2ContractError("device inventory must contain exactly 64 devices")
        if device_inventory.topology != self.topology:
            raise MFUV2ContractError("device inventory topology does not match run")
        if {host.accelerator_type for host in device_inventory.hosts} != {
            self.accelerator_type
        }:
            raise MFUV2ContractError("device inventory is not a v6e-64 slice")

    def as_dict(self) -> dict[str, Any]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _record_object(value, cls)
        return cls(
            attempt_id=_identifier(obj["attempt_id"], "attempt_id"),
            scientific_identity=ContentRefV1.from_dict(obj["scientific_identity"]),
            implementation_identity=ContentRefV1.from_dict(
                obj["implementation_identity"]
            ),
            deployment_identity=ContentRefV1.from_dict(obj["deployment_identity"]),
            semantic_authority=ContentRefV1.from_dict(obj["semantic_authority"]),
            device_inventory=ContentRefV1.from_dict(obj["device_inventory"]),
            tensor_inventory=ContentRefV1.from_dict(obj["tensor_inventory"]),
            executable=ContentRefV1.from_dict(obj["executable"]),
            hlo=ContentRefV1.from_dict(obj["hlo"]),
            flop_formula=ContentRefV1.from_dict(obj["flop_formula"]),
            real_input_manifest=ContentRefV1.from_dict(obj["real_input_manifest"]),
            synthetic_input_manifest=ContentRefV1.from_dict(
                obj["synthetic_input_manifest"]
            ),
            window_plan=PrefixMFUWindowPlanV2.from_dict(obj["window_plan"]),
            sequences_per_step=_integer(
                obj["sequences_per_step"], "sequences_per_step", minimum=1
            ),
            warmup_steps=_integer(obj["warmup_steps"], "warmup_steps", minimum=0),
            accelerator_type=_string(obj["accelerator_type"], "accelerator_type"),
            topology=_string(obj["topology"], "topology"),
            physical_chips=_integer(obj["physical_chips"], "physical_chips", minimum=1),
            denominator_dtype=_string(obj["denominator_dtype"], "denominator_dtype"),
            peak_flops_per_chip=_integer(
                obj["peak_flops_per_chip"],
                "peak_flops_per_chip",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class MFUSemanticReceiptsV2(CanonicalRecord):
    """Content-addressed outputs of the trusted, format-specific verifier.

    The record layer deliberately does not parse Xprof, HLO, ArrayRecord, GCS
    generation metadata, or persistence receipts.  A predeclared authority
    must do so and publish one signed/content-addressed receipt for the run
    plan, formula reconciliation, every input reduction, and every window.
    Receipt order is the canonical plan order.
    """

    SCHEMA: ClassVar[str] = "moe-sae.r1.mfu-semantic-receipts.v2"

    authority_identity: ContentRefV1
    predeclaration_receipt: ContentRefV1
    compile_warmup_receipt: ContentRefV1
    formula_reconciliation_receipt: ContentRefV1
    input_reduction_receipts: tuple[ContentRefV1, ...]
    window_reduction_receipts: tuple[ContentRefV1, ...]

    def __post_init__(self) -> None:
        _content(
            self.authority_identity,
            "authority_identity",
            MEDIA_SEMANTIC_AUTHORITY_V2,
        )
        _content(
            self.predeclaration_receipt,
            "predeclaration_receipt",
            MEDIA_PREDECLARATION_RECEIPT_V2,
        )
        _content(
            self.compile_warmup_receipt,
            "compile_warmup_receipt",
            MEDIA_WARMUP_RECEIPT_V2,
        )
        _content(
            self.formula_reconciliation_receipt,
            "formula_reconciliation_receipt",
            MEDIA_FORMULA_RECONCILIATION_RECEIPT_V2,
        )
        inputs = tuple(self.input_reduction_receipts)
        windows = tuple(self.window_reduction_receipts)
        if len(inputs) != PREFIX_MINIMUM_WINDOWS_V2 * len(_DATA_KINDS):
            raise MFUV2ContractError(
                "input reductions must contain exactly six receipts"
            )
        if len(windows) != (
            PREFIX_MINIMUM_WINDOWS_V2 * len(_DATA_KINDS) * len(_INSTRUMENTATION_KINDS)
        ):
            raise MFUV2ContractError(
                "window reductions must contain exactly twelve receipts"
            )
        for reference in inputs:
            _content(
                reference,
                "input_reduction_receipts[]",
                MEDIA_INPUT_REDUCTION_RECEIPT_V2,
            )
        for reference in windows:
            _content(
                reference,
                "window_reduction_receipts[]",
                MEDIA_WINDOW_REDUCTION_RECEIPT_V2,
            )
        receipt_hashes = [
            self.predeclaration_receipt.sha256,
            self.compile_warmup_receipt.sha256,
            self.formula_reconciliation_receipt.sha256,
            *(item.sha256 for item in inputs),
            *(item.sha256 for item in windows),
        ]
        if len(set(receipt_hashes)) != len(receipt_hashes):
            raise MFUV2ContractError("semantic receipts must be distinct")
        object.__setattr__(self, "input_reduction_receipts", inputs)
        object.__setattr__(self, "window_reduction_receipts", windows)

    def as_dict(self) -> dict[str, Any]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _record_object(value, cls)
        raw_inputs = obj["input_reduction_receipts"]
        raw_windows = obj["window_reduction_receipts"]
        if not isinstance(raw_inputs, list) or not isinstance(raw_windows, list):
            raise MFUV2ContractError("semantic receipt fields must be arrays")
        return cls(
            authority_identity=ContentRefV1.from_dict(obj["authority_identity"]),
            predeclaration_receipt=ContentRefV1.from_dict(
                obj["predeclaration_receipt"]
            ),
            compile_warmup_receipt=ContentRefV1.from_dict(
                obj["compile_warmup_receipt"]
            ),
            formula_reconciliation_receipt=ContentRefV1.from_dict(
                obj["formula_reconciliation_receipt"]
            ),
            input_reduction_receipts=tuple(
                ContentRefV1.from_dict(item) for item in raw_inputs
            ),
            window_reduction_receipts=tuple(
                ContentRefV1.from_dict(item) for item in raw_windows
            ),
        )


class MFUEvidenceAuthorityV2(TypingProtocol):
    """Trust boundary for bulky, format-specific semantic verification.

    ``verify_semantics`` must independently establish all of the following:

    * the registry generation/fence predates all measured trace timestamps and
      binds the exact run contract and plan;
    * the compile/warmup receipt binds the executable and inventory, excludes
      compilation from measured durations, and proves at least the frozen
      number of warmup steps completed before the first planned window;
    * the executable, HLO and counted-operation inventory execute the pinned
      prefix/dtype and reconcile to the analytic formula;
    * every parent manifest and segment ledger reduces to the claimed slice,
      histogram, valid-token count and causal-pair count;
    * every trace binds the attempt, executable/HLO, exact 64 inventory device
      IDs, synchronized planned steps, and all integer timing/health/output
      projections in its window record;
    * persistence receipts and the per-step, per-router, per-expert capacity
      ledgers agree with the claimed output, queue, error and capacity
      observations, with no overloaded expert averaged away.

    Production and every report consumer must inject an authority whose payload
    identity is allowlisted outside the candidate run contract. The run may
    bind that identity but cannot declare it trusted. Returning ``True``
    without performing these checks violates this protocol; serialized
    qualification JSON is unauthenticated until loaded through such an
    externally trusted authority.
    """

    @property
    def identity(self) -> ContentRefV1: ...

    def verify_content(self, reference: ContentRefV1) -> bool: ...

    def verify_semantics(
        self,
        *,
        run_contract: PrefixMFURunContractV2,
        formula: PrefixFlopFormulaV2,
        device_inventory: DeviceInventoryV1,
        windows: tuple[PrefixMFUWindowEvidenceV2, ...],
        receipts: MFUSemanticReceiptsV2,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class PrefixMFUWindowEvidenceV2(CanonicalRecord):
    """Raw integer observations and immutable payload references for one window."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.prefix-mfu-window-evidence.v2"

    run_contract_sha256: str
    window_id: str
    input_slice_sha256: str
    step_start: int
    step_end: int
    step_trace: ContentRefV1
    timing_blocks: ContentRefV1
    profiler_trace: ContentRefV1
    host_metrics: ContentRefV1
    integrity_events: ContentRefV1
    routing_capacity_ledger: ContentRefV1
    persistence_receipt: ContentRefV1 | None
    step_duration_ns: tuple[int, ...]
    input_wait_duration_ns: tuple[int, ...]
    host_total_step_duration_ns: tuple[int, ...]
    peak_hbm_bytes_per_device: tuple[int, ...]
    writer_queue_depth_by_step: tuple[int, ...]
    device_output_tokens: int
    device_output_duration_ns: int
    persisted_tokens: int
    persistence_duration_ns: int
    dispatched_routed_rows_by_invocation: tuple[int, ...]
    allocated_routed_capacity_rows_by_invocation: tuple[int, ...]
    oom_events: int
    recompilations: int
    dropped_routed_tokens: int
    capacity_overflows: int
    nonfinite_events: int

    def __post_init__(self) -> None:
        _sha(self.run_contract_sha256, "run_contract_sha256")
        _identifier(self.window_id, "window_id")
        _sha(self.input_slice_sha256, "input_slice_sha256")
        _integer(self.step_start, "step_start", minimum=0)
        _integer(self.step_end, "step_end", minimum=1)
        if self.step_end <= self.step_start:
            raise MFUV2ContractError("observed step interval must be non-empty")
        for reference, name, media in (
            (self.step_trace, "step_trace", MEDIA_STEP_TRACE_V2),
            (self.timing_blocks, "timing_blocks", MEDIA_TIMING_BLOCKS_V2),
            (self.profiler_trace, "profiler_trace", MEDIA_PROFILER_TRACE_V2),
            (self.host_metrics, "host_metrics", MEDIA_HOST_METRICS_V2),
            (
                self.integrity_events,
                "integrity_events",
                MEDIA_INTEGRITY_EVENTS_V2,
            ),
            (
                self.routing_capacity_ledger,
                "routing_capacity_ledger",
                MEDIA_ROUTING_CAPACITY_LEDGER_V2,
            ),
        ):
            _content(reference, name, media)
        if self.persistence_receipt is not None:
            _content(
                self.persistence_receipt,
                "persistence_receipt",
                MEDIA_PERSISTENCE_RECEIPT_V2,
            )

        steps = self.step_end - self.step_start
        durations = _int_tuple(
            self.step_duration_ns,
            "step_duration_ns",
            length=steps,
            minimum=1,
        )
        waits = _int_tuple(
            self.input_wait_duration_ns,
            "input_wait_duration_ns",
            length=steps,
        )
        queues = _int_tuple(
            self.writer_queue_depth_by_step,
            "writer_queue_depth_by_step",
            length=steps,
        )
        hosts = _int_tuple(
            self.host_total_step_duration_ns,
            "host_total_step_duration_ns",
            length=16,
            minimum=1,
        )
        hbm = _int_tuple(
            self.peak_hbm_bytes_per_device,
            "peak_hbm_bytes_per_device",
            length=V6E64_CHIPS_V2,
            minimum=1,
        )
        dispatches = _int_tuple(
            self.dispatched_routed_rows_by_invocation,
            "dispatched_routed_rows_by_invocation",
            length=PREFIX_ROUTER_INVOCATIONS_V2,
            minimum=1,
        )
        capacities = _int_tuple(
            self.allocated_routed_capacity_rows_by_invocation,
            "allocated_routed_capacity_rows_by_invocation",
            length=PREFIX_ROUTER_INVOCATIONS_V2,
            minimum=1,
        )
        if any(
            wait > duration for wait, duration in zip(waits, durations, strict=True)
        ):
            raise MFUV2ContractError("per-step input wait exceeds step duration")
        if any(value > V6E_CHIP_HBM_BYTES_V2 for value in hbm):
            raise MFUV2ContractError("observed HBM exceeds physical device capacity")
        object.__setattr__(self, "step_duration_ns", durations)
        object.__setattr__(self, "input_wait_duration_ns", waits)
        object.__setattr__(self, "writer_queue_depth_by_step", queues)
        object.__setattr__(self, "host_total_step_duration_ns", hosts)
        object.__setattr__(self, "peak_hbm_bytes_per_device", hbm)
        object.__setattr__(self, "dispatched_routed_rows_by_invocation", dispatches)
        object.__setattr__(
            self,
            "allocated_routed_capacity_rows_by_invocation",
            capacities,
        )

        for name in (
            "device_output_tokens",
            "device_output_duration_ns",
            "persisted_tokens",
            "persistence_duration_ns",
            "oom_events",
            "recompilations",
            "dropped_routed_tokens",
            "capacity_overflows",
            "nonfinite_events",
        ):
            _integer(getattr(self, name), name, minimum=0)
        for count_name, duration_name in (
            ("device_output_tokens", "device_output_duration_ns"),
            ("persisted_tokens", "persistence_duration_ns"),
        ):
            count = getattr(self, count_name)
            duration = getattr(self, duration_name)
            if (count == 0) != (duration == 0):
                raise MFUV2ContractError(
                    f"{count_name} and {duration_name} must both be zero or positive"
                )
        if (self.persistence_receipt is None) != (self.persisted_tokens == 0):
            raise MFUV2ContractError(
                "persistence receipt must exist exactly when persisted tokens exist"
            )
        if any(
            capacity < dispatched
            for dispatched, capacity in zip(dispatches, capacities, strict=True)
        ):
            raise MFUV2ContractError("allocated routed capacity is below dispatch")

    @property
    def wall_time_ns(self) -> int:
        return sum(self.step_duration_ns)

    @property
    def input_wait_ns(self) -> int:
        return sum(self.input_wait_duration_ns)

    @property
    def p50_step_duration_ns(self) -> int:
        return _nearest_rank(self.step_duration_ns, 50)

    @property
    def p95_step_duration_ns(self) -> int:
        return _nearest_rank(self.step_duration_ns, 95)

    def as_dict(self) -> dict[str, Any]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _record_object(value, cls)
        persistence = obj["persistence_receipt"]
        integer_fields = (
            "step_start",
            "step_end",
            "device_output_tokens",
            "device_output_duration_ns",
            "persisted_tokens",
            "persistence_duration_ns",
            "oom_events",
            "recompilations",
            "dropped_routed_tokens",
            "capacity_overflows",
            "nonfinite_events",
        )
        integers = {
            name: _integer(obj[name], name, minimum=0) for name in integer_fields
        }
        return cls(
            run_contract_sha256=_sha(obj["run_contract_sha256"], "run_contract_sha256"),
            window_id=_identifier(obj["window_id"], "window_id"),
            input_slice_sha256=_sha(obj["input_slice_sha256"], "input_slice_sha256"),
            step_trace=ContentRefV1.from_dict(obj["step_trace"]),
            timing_blocks=ContentRefV1.from_dict(obj["timing_blocks"]),
            profiler_trace=ContentRefV1.from_dict(obj["profiler_trace"]),
            host_metrics=ContentRefV1.from_dict(obj["host_metrics"]),
            integrity_events=ContentRefV1.from_dict(obj["integrity_events"]),
            routing_capacity_ledger=ContentRefV1.from_dict(
                obj["routing_capacity_ledger"]
            ),
            persistence_receipt=(
                None if persistence is None else ContentRefV1.from_dict(persistence)
            ),
            step_duration_ns=_int_tuple(
                obj["step_duration_ns"], "step_duration_ns", minimum=1
            ),
            input_wait_duration_ns=_int_tuple(
                obj["input_wait_duration_ns"], "input_wait_duration_ns"
            ),
            host_total_step_duration_ns=_int_tuple(
                obj["host_total_step_duration_ns"],
                "host_total_step_duration_ns",
                minimum=1,
            ),
            peak_hbm_bytes_per_device=_int_tuple(
                obj["peak_hbm_bytes_per_device"],
                "peak_hbm_bytes_per_device",
                minimum=1,
            ),
            dispatched_routed_rows_by_invocation=_int_tuple(
                obj["dispatched_routed_rows_by_invocation"],
                "dispatched_routed_rows_by_invocation",
                minimum=1,
            ),
            allocated_routed_capacity_rows_by_invocation=_int_tuple(
                obj["allocated_routed_capacity_rows_by_invocation"],
                "allocated_routed_capacity_rows_by_invocation",
                minimum=1,
            ),
            writer_queue_depth_by_step=_int_tuple(
                obj["writer_queue_depth_by_step"],
                "writer_queue_depth_by_step",
            ),
            **integers,
        )


@dataclass(frozen=True, slots=True)
class _WindowReductionV2:
    entry: MFUWindowPlanEntryV2
    input_slice: PackedInputSliceV2
    evidence: PrefixMFUWindowEvidenceV2
    useful_flops: int
    excluded_fp32_router_flops: int
    mfu: Fraction
    input_rate: Fraction
    device_output_rate: Fraction | None
    persistence_rate: Fraction | None


@dataclass(frozen=True, slots=True)
class _QualificationReductionV2:
    windows: tuple[_WindowReductionV2, ...]
    aggregate_real_mfu: Fraction
    aggregate_real_baseline_mfu: Fraction
    aggregate_real_instrumented_mfu: Fraction
    minimum_real_window_mfu: Fraction
    instrumentation_deltas: tuple[tuple[int, str, Fraction], ...]
    real_to_synthetic_ratios: tuple[tuple[int, Fraction], ...]
    persisted_to_device_ratios: tuple[tuple[int, Fraction], ...]


def _rate(tokens: int, duration_ns: int) -> Fraction:
    return Fraction(tokens * 1_000_000_000, duration_ns)


def _mfu(useful_flops: int, wall_time_ns: int) -> Fraction:
    return Fraction(
        useful_flops * 1_000_000_000,
        wall_time_ns * V6E64_BF16_PEAK_FLOPS_V2,
    )


def _fraction_dict(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _mfu_dict(value: Fraction) -> dict[str, int]:
    result = _fraction_dict(value)
    result["parts_per_billion"] = value.numerator * 1_000_000_000 // value.denominator
    return result


def _window_key(
    window: _WindowReductionV2,
) -> tuple[int, str, str]:
    return (
        window.entry.group_index,
        window.entry.data_kind,
        window.entry.instrumentation_kind,
    )


def _aggregate_mfu(windows: Sequence[_WindowReductionV2]) -> Fraction:
    if not windows:
        raise MFUV2ContractError("cannot aggregate an empty window set")
    return _mfu(
        sum(window.useful_flops for window in windows),
        sum(window.evidence.wall_time_ns for window in windows),
    )


def _reduce_qualification(
    run: PrefixMFURunContractV2,
    formula: PrefixFlopFormulaV2,
    device_inventory: DeviceInventoryV1,
    evidence: tuple[PrefixMFUWindowEvidenceV2, ...],
) -> _QualificationReductionV2:
    run.verify_dependencies(formula, device_inventory)
    plan = run.window_plan
    if len(evidence) != len(plan.entries) or tuple(
        item.window_id for item in evidence
    ) != tuple(item.window_id for item in plan.entries):
        raise MFUV2ContractError(
            "evidence must contain every planned window in exact canonical order"
        )
    if not all(isinstance(item, PrefixMFUWindowEvidenceV2) for item in evidence):
        raise MFUV2ContractError("window evidence has the wrong record type")

    # Trace-slice identity is one-to-one with a planned window. This rejects
    # accidental reuse, selective aliases, and duplicate timing populations.
    for name in (
        "step_trace",
        "timing_blocks",
        "profiler_trace",
        "host_metrics",
        "routing_capacity_ledger",
    ):
        digests = [getattr(item, name).sha256 for item in evidence]
        if len(set(digests)) != len(digests):
            raise MFUV2ContractError(f"duplicate {name} content across windows")

    reduced: list[_WindowReductionV2] = []
    run_digest = run.digest
    zero_event_fields = (
        "oom_events",
        "recompilations",
        "dropped_routed_tokens",
        "capacity_overflows",
        "nonfinite_events",
    )
    for entry, observed in zip(plan.entries, evidence, strict=True):
        input_slice = plan.input_slice(entry.input_slice_sha256)
        if observed.run_contract_sha256 != run_digest:
            raise MFUV2ContractError("window evidence names the wrong run contract")
        if observed.input_slice_sha256 != entry.input_slice_sha256:
            raise MFUV2ContractError("window evidence names the wrong input slice")
        if (observed.step_start, observed.step_end) != (
            entry.step_start,
            entry.step_end,
        ):
            raise MFUV2ContractError("observed steps differ from the frozen plan")
        wall = observed.wall_time_ns
        if wall < PREFIX_MINIMUM_WALL_NS_V2:
            raise MFUV2ContractError(f"{entry.window_id}: wall time is below 60s")
        if observed.input_wait_ns * 100 >= wall * 5:
            raise MFUV2ContractError(f"{entry.window_id}: input wait is not below 5%")
        if observed.p95_step_duration_ns * 10 > observed.p50_step_duration_ns * 11:
            raise MFUV2ContractError(f"{entry.window_id}: p95/p50 exceeds 1.10")
        ordered_hosts = sorted(observed.host_total_step_duration_ns)
        median_host_twice = ordered_hosts[7] + ordered_hosts[8]
        if ordered_hosts[-1] * 200 > median_host_twice * 105:
            raise MFUV2ContractError(f"{entry.window_id}: host skew exceeds 1.05")
        if any(abs(host - wall) * 100 > wall * 5 for host in ordered_hosts):
            raise MFUV2ContractError(
                f"{entry.window_id}: host and synchronized wall clocks disagree"
            )
        if max(observed.peak_hbm_bytes_per_device) * 100 > (V6E_CHIP_HBM_BYTES_V2 * 85):
            raise MFUV2ContractError(f"{entry.window_id}: HBM headroom is below 15%")
        expected_dispatch = (
            input_slice.packed_sequence_count
            * PREFIX_SEQUENCE_LENGTH_V2
            * PREFIX_SELECTED_EXPERTS_V2
        )
        if any(
            dispatched != expected_dispatch
            for dispatched in observed.dispatched_routed_rows_by_invocation
        ):
            raise MFUV2ContractError(
                f"{entry.window_id}: per-invocation dispatch count is not exact"
            )
        if any(
            capacity * 10 > dispatched * 11
            for dispatched, capacity in zip(
                observed.dispatched_routed_rows_by_invocation,
                observed.allocated_routed_capacity_rows_by_invocation,
                strict=True,
            )
        ):
            raise MFUV2ContractError(
                f"{entry.window_id}: per-invocation capacity exceeds 1.10"
            )
        if any(getattr(observed, name) != 0 for name in zero_event_fields):
            raise MFUV2ContractError(f"{entry.window_id}: integrity event is nonzero")
        if max(observed.writer_queue_depth_by_step) >= 2:
            raise MFUV2ContractError(f"{entry.window_id}: writer queue reached two")

        if entry.instrumentation_kind == _BASELINE:
            if (
                observed.device_output_tokens
                or observed.device_output_duration_ns
                or observed.persisted_tokens
                or observed.persistence_duration_ns
                or observed.persistence_receipt is not None
            ):
                raise MFUV2ContractError(
                    f"{entry.window_id}: baseline contains instrumented output"
                )
        else:
            if observed.device_output_tokens != input_slice.histogram.valid_tokens:
                raise MFUV2ContractError(
                    f"{entry.window_id}: device output token count is not exact"
                )
            if observed.device_output_duration_ns != wall:
                raise MFUV2ContractError(
                    f"{entry.window_id}: device output duration differs from wall"
                )
            if entry.data_kind == _REAL:
                if observed.persisted_tokens != input_slice.histogram.valid_tokens:
                    raise MFUV2ContractError(
                        f"{entry.window_id}: persisted token count is not exact"
                    )
            elif observed.persisted_tokens or observed.persistence_receipt is not None:
                raise MFUV2ContractError(
                    f"{entry.window_id}: synthetic window must not persist output"
                )

        useful = formula.useful_flops(input_slice.histogram)
        window_mfu = _mfu(useful, wall)
        if window_mfu > 1:
            raise MFUV2ContractError(f"{entry.window_id}: MFU exceeds physical peak")
        reduced.append(
            _WindowReductionV2(
                entry=entry,
                input_slice=input_slice,
                evidence=observed,
                useful_flops=useful,
                excluded_fp32_router_flops=(
                    formula.excluded_fp32_router_flops(input_slice.histogram)
                ),
                mfu=window_mfu,
                input_rate=_rate(input_slice.histogram.valid_tokens, wall),
                device_output_rate=(
                    None
                    if observed.device_output_tokens == 0
                    else _rate(
                        observed.device_output_tokens,
                        observed.device_output_duration_ns,
                    )
                ),
                persistence_rate=(
                    None
                    if observed.persisted_tokens == 0
                    else _rate(
                        observed.persisted_tokens,
                        observed.persistence_duration_ns,
                    )
                ),
            )
        )

    windows = tuple(reduced)
    by_key = {_window_key(window): window for window in windows}
    instrumentation_deltas: list[tuple[int, str, Fraction]] = []
    real_synthetic_ratios: list[tuple[int, Fraction]] = []
    persistence_ratios: list[tuple[int, Fraction]] = []
    for group in range(PREFIX_MINIMUM_WINDOWS_V2):
        for data_kind in _DATA_KINDS:
            baseline = by_key[(group, data_kind, _BASELINE)]
            instrumented = by_key[(group, data_kind, _INSTRUMENTED)]
            delta = abs(baseline.mfu - instrumented.mfu)
            if delta > Fraction(2, 100):
                raise MFUV2ContractError(
                    f"group {group} {data_kind}: instrumentation MFU delta exceeds 2pp"
                )
            instrumentation_deltas.append((group, data_kind, delta))
        real = by_key[(group, _REAL, _INSTRUMENTED)]
        synthetic = by_key[(group, _SYNTHETIC, _INSTRUMENTED)]
        if real.device_output_rate is None or synthetic.device_output_rate is None:
            raise MFUV2ContractError("instrumented device rate is absent")
        real_synthetic = real.device_output_rate / synthetic.device_output_rate
        if real_synthetic < Fraction(9, 10):
            raise MFUV2ContractError(
                f"group {group}: real/synthetic device goodput is below 90%"
            )
        real_synthetic_ratios.append((group, real_synthetic))
        if real.persistence_rate is None:
            raise MFUV2ContractError("real persistence rate is absent")
        persisted_device = real.persistence_rate / real.device_output_rate
        if persisted_device < Fraction(9, 10):
            raise MFUV2ContractError(
                f"group {group}: persisted/device goodput is below 90%"
            )
        persistence_ratios.append((group, persisted_device))

    real_windows = tuple(
        window for window in windows if window.entry.data_kind == _REAL
    )
    real_baseline = tuple(
        window
        for window in real_windows
        if window.entry.instrumentation_kind == _BASELINE
    )
    real_instrumented = tuple(
        window
        for window in real_windows
        if window.entry.instrumentation_kind == _INSTRUMENTED
    )
    minimum_real = min(window.mfu for window in real_windows)
    aggregate_real = _aggregate_mfu(real_windows)
    aggregate_baseline = _aggregate_mfu(real_baseline)
    aggregate_instrumented = _aggregate_mfu(real_instrumented)
    for name, value in (
        ("minimum real window", minimum_real),
        ("aggregate real", aggregate_real),
        ("aggregate real baseline", aggregate_baseline),
        ("aggregate real instrumented", aggregate_instrumented),
    ):
        if value < Fraction(51, 100):
            raise MFUV2ContractError(f"{name} MFU is below 51%")

    return _QualificationReductionV2(
        windows=windows,
        aggregate_real_mfu=aggregate_real,
        aggregate_real_baseline_mfu=aggregate_baseline,
        aggregate_real_instrumented_mfu=aggregate_instrumented,
        minimum_real_window_mfu=minimum_real,
        instrumentation_deltas=tuple(instrumentation_deltas),
        real_to_synthetic_ratios=tuple(real_synthetic_ratios),
        persisted_to_device_ratios=tuple(persistence_ratios),
    )


def _evidence_content_refs(
    run: PrefixMFURunContractV2,
    formula: PrefixFlopFormulaV2,
    evidence: Sequence[PrefixMFUWindowEvidenceV2],
    receipts: MFUSemanticReceiptsV2,
) -> tuple[ContentRefV1, ...]:
    references: list[ContentRefV1] = [
        run.scientific_identity,
        run.implementation_identity,
        run.deployment_identity,
        run.semantic_authority,
        run.device_inventory,
        run.tensor_inventory,
        run.executable,
        run.hlo,
        run.flop_formula,
        run.real_input_manifest,
        run.synthetic_input_manifest,
        formula.counted_operation_inventory,
        *(item.segment_ledger for item in run.window_plan.input_slices),
        receipts.predeclaration_receipt,
        receipts.compile_warmup_receipt,
        receipts.formula_reconciliation_receipt,
        *receipts.input_reduction_receipts,
        *receipts.window_reduction_receipts,
    ]
    for item in evidence:
        references.extend(
            (
                item.step_trace,
                item.timing_blocks,
                item.profiler_trace,
                item.host_metrics,
                item.integrity_events,
                item.routing_capacity_ledger,
            )
        )
        if item.persistence_receipt is not None:
            references.append(item.persistence_receipt)
    return tuple(references)


def _verify_content(
    references: Sequence[ContentRefV1],
    authority: MFUEvidenceAuthorityV2,
) -> None:
    verifier = getattr(authority, "verify_content", None)
    if not callable(verifier):
        raise MFUV2ContractError("evidence authority cannot verify content")
    identities: dict[str, tuple[int, str]] = {}
    verified_records: set[str] = set()
    for reference in references:
        identity = (reference.size_bytes, reference.media_type)
        previous = identities.setdefault(reference.sha256, identity)
        if previous != identity:
            raise MFUV2ContractError(
                "one content digest has conflicting size or media identities"
            )
        if reference.digest in verified_records:
            continue
        try:
            resolved = verifier(reference)
        except Exception as error:
            raise MFUV2ContractError(
                f"content verification raised for {reference.sha256}"
            ) from error
        if resolved is not True:
            raise MFUV2ContractError(f"content did not verify: {reference.sha256}")
        verified_records.add(reference.digest)


def _validate_semantic_authority(
    *,
    authority: MFUEvidenceAuthorityV2,
    run: PrefixMFURunContractV2,
    receipts: MFUSemanticReceiptsV2,
) -> None:
    identity = getattr(authority, "identity", None)
    _content(identity, "evidence_authority.identity", MEDIA_SEMANTIC_AUTHORITY_V2)
    if not _same_payload(identity, run.semantic_authority):
        raise MFUV2ContractError(
            "evidence authority is not the authority frozen in the run contract"
        )
    if not _same_payload(identity, receipts.authority_identity):
        raise MFUV2ContractError(
            "semantic receipts name a different evidence authority"
        )
    verifier = getattr(authority, "verify_semantics", None)
    if not callable(verifier):
        raise MFUV2ContractError("evidence authority cannot verify semantics")


def _verify_semantic_authority(
    *,
    authority: MFUEvidenceAuthorityV2,
    run: PrefixMFURunContractV2,
    formula: PrefixFlopFormulaV2,
    device_inventory: DeviceInventoryV1,
    windows: tuple[PrefixMFUWindowEvidenceV2, ...],
    receipts: MFUSemanticReceiptsV2,
) -> None:
    _validate_semantic_authority(authority=authority, run=run, receipts=receipts)
    verifier = authority.verify_semantics
    try:
        verified = verifier(
            run_contract=run,
            formula=formula,
            device_inventory=device_inventory,
            windows=windows,
            receipts=receipts,
        )
    except Exception as error:
        raise MFUV2ContractError("semantic verification raised") from error
    if verified is not True:
        raise MFUV2ContractError("semantic evidence did not verify")


def _window_derived_dict(window: _WindowReductionV2) -> dict[str, Any]:
    evidence = window.evidence
    capacity_ratios = tuple(
        Fraction(capacity, dispatched)
        for dispatched, capacity in zip(
            evidence.dispatched_routed_rows_by_invocation,
            evidence.allocated_routed_capacity_rows_by_invocation,
            strict=True,
        )
    )
    return {
        "window_id": window.entry.window_id,
        "group_index": window.entry.group_index,
        "data_kind": window.entry.data_kind,
        "instrumentation_kind": window.entry.instrumentation_kind,
        "valid_tokens": window.input_slice.histogram.valid_tokens,
        "causal_pairs": window.input_slice.histogram.causal_pairs,
        "useful_bf16_flops": window.useful_flops,
        "excluded_fp32_router_flops": window.excluded_fp32_router_flops,
        "wall_time_ns": evidence.wall_time_ns,
        "input_wait_ns": evidence.input_wait_ns,
        "p50_step_duration_ns": evidence.p50_step_duration_ns,
        "p95_step_duration_ns": evidence.p95_step_duration_ns,
        "maximum_hbm_bytes": max(evidence.peak_hbm_bytes_per_device),
        "maximum_writer_queue_depth": max(evidence.writer_queue_depth_by_step),
        "total_dispatched_routed_rows": sum(
            evidence.dispatched_routed_rows_by_invocation
        ),
        "maximum_per_invocation_capacity_ratio": _fraction_dict(max(capacity_ratios)),
        "mfu": _mfu_dict(window.mfu),
        "input_tokens_per_second": _fraction_dict(window.input_rate),
        "device_output_tokens_per_second": (
            None
            if window.device_output_rate is None
            else _fraction_dict(window.device_output_rate)
        ),
        "persisted_tokens_per_second": (
            None
            if window.persistence_rate is None
            else _fraction_dict(window.persistence_rate)
        ),
    }


@dataclass(frozen=True, slots=True, init=False)
class PrefixMFUQualificationV2(CanonicalRecord):
    """Canonical qualification whose reported values are reducer-only outputs.

    Plain ``from_dict`` intentionally fails: remote content must be resolved
    again through ``from_dict_verified``.  The deterministic lower bound is the
    minimum of all six real-data windows.  No statistical interval is inferred
    from three groups; bound timing-block payloads support a later predeclared
    block-level reducer. Canonical bytes are not self-authenticating: consumers
    must use a verified loader with an externally allowlisted authority.
    """

    SCHEMA: ClassVar[str] = "moe-sae.r1.prefix-mfu-qualification.v2"

    run_contract: PrefixMFURunContractV2
    formula: PrefixFlopFormulaV2
    device_inventory: DeviceInventoryV1
    windows: tuple[PrefixMFUWindowEvidenceV2, ...]
    semantic_receipts: MFUSemanticReceiptsV2
    _reduction: _QualificationReductionV2 = field(init=False, repr=False, compare=False)

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise MFUV2ContractError(
            "qualification construction is private; use qualify_prefix_mfu_v2"
        )

    @classmethod
    def _from_verified(
        cls,
        *,
        run_contract: PrefixMFURunContractV2,
        formula: PrefixFlopFormulaV2,
        device_inventory: DeviceInventoryV1,
        windows: tuple[PrefixMFUWindowEvidenceV2, ...],
        semantic_receipts: MFUSemanticReceiptsV2,
    ) -> Self:
        result = object.__new__(cls)
        object.__setattr__(result, "run_contract", run_contract)
        object.__setattr__(result, "formula", formula)
        object.__setattr__(result, "device_inventory", device_inventory)
        object.__setattr__(result, "windows", windows)
        object.__setattr__(result, "semantic_receipts", semantic_receipts)
        result._initialize()
        return result

    def _initialize(self) -> None:
        if not isinstance(self.run_contract, PrefixMFURunContractV2):
            raise MFUV2ContractError("run_contract has the wrong record type")
        if not isinstance(self.formula, PrefixFlopFormulaV2):
            raise MFUV2ContractError("formula has the wrong record type")
        if not isinstance(self.device_inventory, DeviceInventoryV1):
            raise MFUV2ContractError("device_inventory has the wrong record type")
        if not isinstance(self.windows, (list, tuple)):
            raise MFUV2ContractError("windows must be an array")
        if not isinstance(self.semantic_receipts, MFUSemanticReceiptsV2):
            raise MFUV2ContractError("semantic_receipts has the wrong record type")
        if not _same_payload(
            self.semantic_receipts.authority_identity,
            self.run_contract.semantic_authority,
        ):
            raise MFUV2ContractError(
                "semantic receipts do not match the frozen authority"
            )
        windows = tuple(self.windows)
        object.__setattr__(self, "windows", windows)
        object.__setattr__(
            self,
            "_reduction",
            _reduce_qualification(
                self.run_contract,
                self.formula,
                self.device_inventory,
                windows,
            ),
        )

    @property
    def aggregate_real_mfu(self) -> Fraction:
        return self._reduction.aggregate_real_mfu

    @property
    def minimum_real_window_mfu(self) -> Fraction:
        """Deterministic conservative lower bound used for headline readiness."""

        return self._reduction.minimum_real_window_mfu

    def window_mfu(self, window_id: str) -> Fraction:
        for window in self._reduction.windows:
            if window.entry.window_id == window_id:
                return window.mfu
        raise MFUV2ContractError(f"unknown window ID: {window_id}")

    def _derived_dict(self) -> dict[str, Any]:
        reduction = self._reduction
        return {
            "overall_qualification_status": "UNVERIFIED",
            "numerical_gate_status": "SATISFIED",
            "semantic_authentication_status": ("MUST_BE_REVERIFIED_BY_CONSUMER"),
            "verification_requirement": ("EXTERNALLY_ALLOWLISTED_SEMANTIC_AUTHORITY"),
            "semantic_receipts_sha256": self.semantic_receipts.digest,
            "physical_chips": V6E64_CHIPS_V2,
            "aggregate_peak_bf16_flops_per_second": (V6E64_BF16_PEAK_FLOPS_V2),
            "windows": [_window_derived_dict(window) for window in reduction.windows],
            "aggregate_real_mfu": _mfu_dict(reduction.aggregate_real_mfu),
            "aggregate_real_baseline_mfu": _mfu_dict(
                reduction.aggregate_real_baseline_mfu
            ),
            "aggregate_real_instrumented_mfu": _mfu_dict(
                reduction.aggregate_real_instrumented_mfu
            ),
            "minimum_real_window_mfu": _mfu_dict(reduction.minimum_real_window_mfu),
            "conservative_lower_bound_method": ("MINIMUM_OF_ALL_PLANNED_REAL_WINDOWS"),
            "instrumentation_mfu_deltas": [
                {
                    "group_index": group,
                    "data_kind": data_kind,
                    "delta": _mfu_dict(delta),
                }
                for group, data_kind, delta in reduction.instrumentation_deltas
            ],
            "real_to_synthetic_device_goodput_ratios": [
                {"group_index": group, "ratio": _fraction_dict(ratio)}
                for group, ratio in reduction.real_to_synthetic_ratios
            ],
            "persisted_to_device_goodput_ratios": [
                {"group_index": group, "ratio": _fraction_dict(ratio)}
                for group, ratio in reduction.persisted_to_device_ratios
            ],
            "statistical_interval_status": (
                "NOT_DERIVED_REQUIRES_PREDECLARED_BLOCK_REDUCER"
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "run_contract": self.run_contract.as_dict(),
            "formula": self.formula.as_dict(),
            "device_inventory": self.device_inventory.as_dict(),
            "windows": [window.as_dict() for window in self.windows],
            "semantic_receipts": self.semantic_receipts.as_dict(),
            "derived": self._derived_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        del value
        raise MFUV2ContractError(
            "use PrefixMFUQualificationV2.from_dict_verified with the frozen "
            "semantic evidence authority"
        )

    @classmethod
    def from_dict_verified(
        cls,
        value: Mapping[str, Any],
        *,
        evidence_authority: MFUEvidenceAuthorityV2,
    ) -> Self:
        obj = _strict_object(
            value,
            schema=cls.SCHEMA,
            fields=(
                "run_contract",
                "formula",
                "device_inventory",
                "windows",
                "semantic_receipts",
                "derived",
            ),
        )
        raw_windows = obj["windows"]
        if not isinstance(raw_windows, list):
            raise MFUV2ContractError("windows must be a JSON array")
        result = qualify_prefix_mfu_v2(
            run_contract=PrefixMFURunContractV2.from_dict(obj["run_contract"]),
            formula=PrefixFlopFormulaV2.from_dict(obj["formula"]),
            device_inventory=DeviceInventoryV1.from_dict(obj["device_inventory"]),
            windows=tuple(
                PrefixMFUWindowEvidenceV2.from_dict(item) for item in raw_windows
            ),
            semantic_receipts=MFUSemanticReceiptsV2.from_dict(obj["semantic_receipts"]),
            evidence_authority=evidence_authority,
        )
        if canonical_json_bytes(obj["derived"]) != canonical_json_bytes(
            result._derived_dict()
        ):
            raise MFUV2ContractError(
                "serialized qualification values are not canonical derivations"
            )
        return result

    @classmethod
    def from_json_verified(
        cls,
        value: str | bytes,
        *,
        evidence_authority: MFUEvidenceAuthorityV2,
    ) -> Self:
        parsed = parse_json(value)
        if not isinstance(parsed, Mapping):
            raise MFUV2ContractError("qualification JSON must be an object")
        return cls.from_dict_verified(parsed, evidence_authority=evidence_authority)


def qualify_prefix_mfu_v2(
    *,
    run_contract: PrefixMFURunContractV2,
    formula: PrefixFlopFormulaV2,
    device_inventory: DeviceInventoryV1,
    windows: Sequence[PrefixMFUWindowEvidenceV2],
    semantic_receipts: MFUSemanticReceiptsV2,
    evidence_authority: MFUEvidenceAuthorityV2,
) -> PrefixMFUQualificationV2:
    """Verify content, reduce raw observations, and apply the immutable gates."""

    if not isinstance(run_contract, PrefixMFURunContractV2):
        raise MFUV2ContractError("run_contract has the wrong record type")
    if not isinstance(formula, PrefixFlopFormulaV2):
        raise MFUV2ContractError("formula has the wrong record type")
    if not isinstance(device_inventory, DeviceInventoryV1):
        raise MFUV2ContractError("device_inventory has the wrong record type")
    if not isinstance(semantic_receipts, MFUSemanticReceiptsV2):
        raise MFUV2ContractError("semantic_receipts has the wrong record type")
    if not isinstance(windows, (list, tuple)):
        raise MFUV2ContractError("windows must be an array")
    frozen_windows = tuple(windows)
    if not all(
        isinstance(window, PrefixMFUWindowEvidenceV2) for window in frozen_windows
    ):
        raise MFUV2ContractError(
            "windows must contain PrefixMFUWindowEvidenceV2 records"
        )
    _reduce_qualification(run_contract, formula, device_inventory, frozen_windows)
    _validate_semantic_authority(
        authority=evidence_authority,
        run=run_contract,
        receipts=semantic_receipts,
    )
    _verify_content(
        _evidence_content_refs(
            run_contract,
            formula,
            frozen_windows,
            semantic_receipts,
        ),
        evidence_authority,
    )
    _verify_semantic_authority(
        authority=evidence_authority,
        run=run_contract,
        formula=formula,
        device_inventory=device_inventory,
        windows=frozen_windows,
        receipts=semantic_receipts,
    )
    return PrefixMFUQualificationV2._from_verified(
        run_contract=run_contract,
        formula=formula,
        device_inventory=device_inventory,
        windows=frozen_windows,
        semantic_receipts=semantic_receipts,
    )


__all__ = [
    "MFUV2ContractError",
    "MFUEvidenceAuthorityV2",
    "MFUSemanticReceiptsV2",
    "MFUWindowPlanEntryV2",
    "PackedInputSliceV2",
    "PackedSegmentHistogramV2",
    "PrefixFlopFormulaV2",
    "PrefixMFUQualificationV2",
    "PrefixMFURunContractV2",
    "PrefixMFUWindowEvidenceV2",
    "PrefixMFUWindowPlanV2",
    "SegmentLengthBinV2",
    "qualify_prefix_mfu_v2",
]
