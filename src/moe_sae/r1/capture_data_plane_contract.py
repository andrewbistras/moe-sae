"""Topology-neutral capture-fragment and writer contract for the R1 prefix.

This module turns the B128 batch-split output ABI into deterministic logical
fragments without importing JAX, MaxText, or a cloud client.  One addressable
two-sequence batch shard becomes one columnar binary object.  Fragment identity
depends on the immutable input range and global batch slice, never on host
suffix, process index, local device order, zone, or upload stream.

The contract intentionally separates the numerical window from durable
qualification.  A create acknowledgement is provisional; generation-pinned
full-payload readback and manifest publication happen outside the synchronized
prefix window.  No object written under this contract is production-visible
until that later validator succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, ClassVar, Iterator

from .gcs_immutable_byte_store import (
    MAX_APPLICATION_STREAM_SLICE_BYTES,
    MAX_IMMUTABLE_SNAPSHOT_BYTES,
    content_addressed_object_id,
)
from .prefix_batchsplit_contract import (
    PINNED_CHIPS_PER_HOST,
    PINNED_GLOBAL_BATCH_SIZE,
    PINNED_HOST_COUNT,
    PINNED_PREFIX_BATCHSPLIT_CONTRACT,
    PINNED_PREFIX_BATCHSPLIT_CONTRACT_SHA256,
    PINNED_SEQUENCE_LENGTH,
)
from .protocol import (
    CanonicalRecord,
    IDENTIFIER_PATTERN,
    ProtocolError,
    SHA256_PATTERN,
    require_integer,
    require_string,
)


PINNED_CAPTURE_ROW_BYTES = 14_378
PINNED_FRAGMENT_TOKEN_CAPACITY = 2_048
PINNED_FRAGMENTS_PER_STEP = 64
PINNED_FRAGMENTS_PER_HOST = 4
PINNED_FULL_FRAGMENT_BYTES = 29_446_144
PINNED_FULL_HOST_STEP_BYTES = 117_784_576
PINNED_FULL_STEP_BYTES = 1_884_553_216

PINNED_WRITER_SLOT_COUNT = 2
PINNED_MAX_QUEUED_COMPLETE_STEPS = 1
PINNED_UPLOAD_STREAMS_PER_HOST = 4
PINNED_PLANNING_TOKENS_PER_SECOND = 801_000
PINNED_MINIMUM_PERSISTED_NUMERATOR = 9
PINNED_MINIMUM_PERSISTED_DENOMINATOR = 10

_CREATE_POLICY = "unique-attempt-create-only-conflict-is-failure"
_READBACK_POLICY = "generation-pinned-full-readback-outside-numerical-window"
_PAYLOAD_LAYOUT = "columnar-c-order-little-endian-v1"
_DTYPE_BYTES = {
    "bfloat16": 2,
    "uint16": 2,
    "int64": 8,
}


class CaptureDataPlaneContractError(ValueError):
    """A capture-fragment identity, byte layout, or queue invariant failed."""


def _exact_identifier(value: Any, name: str) -> str:
    if type(value) is not str:
        raise CaptureDataPlaneContractError(f"{name} must be an exact string")
    try:
        return require_string(value, name, pattern=IDENTIFIER_PATTERN)
    except ProtocolError as error:
        raise CaptureDataPlaneContractError(f"{name} is not an identifier") from error


def _exact_sha256(value: Any, name: str) -> str:
    if type(value) is not str:
        raise CaptureDataPlaneContractError(f"{name} must be an exact string")
    try:
        return require_string(value, name, pattern=SHA256_PATTERN)
    except ProtocolError as error:
        raise CaptureDataPlaneContractError(f"{name} is not a SHA-256") from error


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    try:
        return require_integer(value, name, minimum=minimum, maximum=maximum)
    except ProtocolError as error:
        raise CaptureDataPlaneContractError(f"{name} is invalid") from error


@dataclass(frozen=True, slots=True)
class CapturePayloadFieldV1(CanonicalRecord):
    """One column in the deterministic fragment payload."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.capture-payload-field.v1"

    name: str
    dtype: str
    elements_per_token: int
    semantics: str

    def __post_init__(self) -> None:
        _exact_identifier(self.name, "name")
        dtype = _exact_identifier(self.dtype, "dtype")
        if dtype not in _DTYPE_BYTES:
            raise CaptureDataPlaneContractError("payload field dtype is unsupported")
        _integer(self.elements_per_token, "elements_per_token", minimum=1)
        _exact_identifier(self.semantics, "semantics")

    @property
    def bytes_per_token(self) -> int:
        return self.elements_per_token * _DTYPE_BYTES[self.dtype]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "name": self.name,
            "dtype": self.dtype,
            "elements_per_token": self.elements_per_token,
            "bytes_per_token": self.bytes_per_token,
            "semantics": self.semantics,
        }


PINNED_CAPTURE_PAYLOAD_FIELDS = (
    CapturePayloadFieldV1(
        name="routed_activation",
        dtype="bfloat16",
        elements_per_token=7_168,
        semantics="target-routed-aggregate-before-shared-expert-and-residual",
    ),
    CapturePayloadFieldV1(
        name="global_expert_ids",
        dtype="uint16",
        elements_per_token=8,
        semantics="global-ids-before-shard-local-remap",
    ),
    CapturePayloadFieldV1(
        name="scaled_router_weights",
        dtype="bfloat16",
        elements_per_token=8,
        semantics="normalized-unbiased-selected-scores-times-2.5",
    ),
    CapturePayloadFieldV1(
        name="sequence_id",
        dtype="int64",
        elements_per_token=1,
        semantics="stable-sequence-id-for-each-token",
    ),
    CapturePayloadFieldV1(
        name="position",
        dtype="uint16",
        elements_per_token=1,
        semantics="zero-based-position-within-sequence",
    ),
)


@dataclass(frozen=True, slots=True)
class CaptureDataPlaneContractV1(CanonicalRecord):
    """Frozen first writer candidate; this record is not qualification evidence."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.capture-data-plane-contract.v1"

    batchsplit_contract_sha256: str = PINNED_PREFIX_BATCHSPLIT_CONTRACT_SHA256
    payload_layout: str = _PAYLOAD_LAYOUT
    payload_fields: tuple[CapturePayloadFieldV1, ...] = PINNED_CAPTURE_PAYLOAD_FIELDS
    fragment_token_capacity: int = PINNED_FRAGMENT_TOKEN_CAPACITY
    fragments_per_step: int = PINNED_FRAGMENTS_PER_STEP
    fragments_per_host: int = PINNED_FRAGMENTS_PER_HOST
    writer_slot_count: int = PINNED_WRITER_SLOT_COUNT
    maximum_queued_complete_steps: int = PINNED_MAX_QUEUED_COMPLETE_STEPS
    upload_streams_per_host: int = PINNED_UPLOAD_STREAMS_PER_HOST
    maximum_object_bytes: int = MAX_IMMUTABLE_SNAPSHOT_BYTES
    maximum_application_slice_bytes: int = MAX_APPLICATION_STREAM_SLICE_BYTES
    planning_tokens_per_second: int = PINNED_PLANNING_TOKENS_PER_SECOND
    minimum_persisted_numerator: int = PINNED_MINIMUM_PERSISTED_NUMERATOR
    minimum_persisted_denominator: int = PINNED_MINIMUM_PERSISTED_DENOMINATOR
    create_policy: str = _CREATE_POLICY
    readback_policy: str = _READBACK_POLICY
    synchronized_path_may_wait_for_writer: bool = False
    create_ack_is_durable_qualification: bool = False
    production_sealable: bool = False
    production_admission_enabled: bool = False

    def __post_init__(self) -> None:
        _exact_sha256(
            self.batchsplit_contract_sha256,
            "batchsplit_contract_sha256",
        )
        if (
            self.batchsplit_contract_sha256 != PINNED_PREFIX_BATCHSPLIT_CONTRACT_SHA256
            or PINNED_PREFIX_BATCHSPLIT_CONTRACT.digest
            != self.batchsplit_contract_sha256
        ):
            raise CaptureDataPlaneContractError(
                "capture writer is not bound to the pinned B128 ABI"
            )
        if (
            type(self.payload_layout) is not str
            or self.payload_layout != _PAYLOAD_LAYOUT
        ):
            raise CaptureDataPlaneContractError("payload layout changed")
        if (
            type(self.payload_fields) is not tuple
            or self.payload_fields != PINNED_CAPTURE_PAYLOAD_FIELDS
            or any(
                type(item) is not CapturePayloadFieldV1 for item in self.payload_fields
            )
        ):
            raise CaptureDataPlaneContractError("payload field layout changed")
        exact_integers = (
            ("fragment_token_capacity", PINNED_FRAGMENT_TOKEN_CAPACITY),
            ("fragments_per_step", PINNED_FRAGMENTS_PER_STEP),
            ("fragments_per_host", PINNED_FRAGMENTS_PER_HOST),
            ("writer_slot_count", PINNED_WRITER_SLOT_COUNT),
            (
                "maximum_queued_complete_steps",
                PINNED_MAX_QUEUED_COMPLETE_STEPS,
            ),
            ("upload_streams_per_host", PINNED_UPLOAD_STREAMS_PER_HOST),
            ("maximum_object_bytes", MAX_IMMUTABLE_SNAPSHOT_BYTES),
            (
                "maximum_application_slice_bytes",
                MAX_APPLICATION_STREAM_SLICE_BYTES,
            ),
            (
                "planning_tokens_per_second",
                PINNED_PLANNING_TOKENS_PER_SECOND,
            ),
            (
                "minimum_persisted_numerator",
                PINNED_MINIMUM_PERSISTED_NUMERATOR,
            ),
            (
                "minimum_persisted_denominator",
                PINNED_MINIMUM_PERSISTED_DENOMINATOR,
            ),
        )
        for name, expected in exact_integers:
            if _integer(getattr(self, name), name, minimum=1) != expected:
                raise CaptureDataPlaneContractError(f"{name} changed")
        if self.maximum_queued_complete_steps >= self.writer_slot_count:
            raise CaptureDataPlaneContractError(
                "admissible queue depth must stay below the two-slot capacity"
            )
        if self.fragments_per_host != PINNED_CHIPS_PER_HOST:
            raise CaptureDataPlaneContractError(
                "one host must publish one fragment per addressable chip"
            )
        if self.upload_streams_per_host != self.fragments_per_host:
            raise CaptureDataPlaneContractError(
                "the first writer candidate requires one stream per fragment"
            )
        if self.row_bytes != PINNED_CAPTURE_ROW_BYTES:
            raise CaptureDataPlaneContractError("durable row width changed")
        if (
            self.full_fragment_bytes != PINNED_FULL_FRAGMENT_BYTES
            or self.full_host_step_bytes != PINNED_FULL_HOST_STEP_BYTES
            or self.full_step_bytes != PINNED_FULL_STEP_BYTES
        ):
            raise CaptureDataPlaneContractError("capture byte arithmetic changed")
        if self.full_fragment_bytes > self.maximum_object_bytes:
            raise CaptureDataPlaneContractError(
                "one logical fragment exceeds the immutable snapshot bound"
            )
        if (
            type(self.create_policy) is not str
            or self.create_policy != _CREATE_POLICY
            or type(self.readback_policy) is not str
            or self.readback_policy != _READBACK_POLICY
        ):
            raise CaptureDataPlaneContractError("writer publication policy changed")
        for name in (
            "synchronized_path_may_wait_for_writer",
            "create_ack_is_durable_qualification",
            "production_sealable",
            "production_admission_enabled",
        ):
            if (
                type(getattr(self, name)) is not bool
                or getattr(self, name) is not False
            ):
                raise CaptureDataPlaneContractError(
                    f"{name} must remain exact false before live qualification"
                )

    @property
    def row_bytes(self) -> int:
        return sum(item.bytes_per_token for item in self.payload_fields)

    @property
    def full_fragment_bytes(self) -> int:
        return self.row_bytes * self.fragment_token_capacity

    @property
    def full_host_step_bytes(self) -> int:
        return self.full_fragment_bytes * self.fragments_per_host

    @property
    def full_step_bytes(self) -> int:
        return self.full_fragment_bytes * self.fragments_per_step

    @property
    def full_step_tokens(self) -> int:
        return self.fragment_token_capacity * self.fragments_per_step

    @property
    def planning_bytes_per_second(self) -> int:
        return self.planning_tokens_per_second * self.row_bytes

    @property
    def planning_bytes_per_second_per_host(self) -> int:
        return self.planning_bytes_per_second // PINNED_HOST_COUNT

    @property
    def planning_step_duration_ns_numerator(self) -> int:
        return self.full_step_tokens * 1_000_000_000

    @property
    def planning_step_duration_ns_denominator(self) -> int:
        return self.planning_tokens_per_second

    def field_byte_ranges(self, valid_tokens: int) -> tuple[tuple[str, int, int], ...]:
        """Return exact columnar byte intervals for one compacted fragment."""

        count = _integer(
            valid_tokens,
            "valid_tokens",
            minimum=1,
            maximum=self.fragment_token_capacity,
        )
        offset = 0
        result = []
        for field in self.payload_fields:
            stop = offset + field.bytes_per_token * count
            result.append((field.name, offset, stop))
            offset = stop
        if offset != count * self.row_bytes:
            raise CaptureDataPlaneContractError("columnar byte layout is inconsistent")
        return tuple(result)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "batchsplit_contract_sha256": self.batchsplit_contract_sha256,
            "payload_layout": self.payload_layout,
            "payload_fields": [item.as_dict() for item in self.payload_fields],
            "row_bytes": self.row_bytes,
            "fragment_token_capacity": self.fragment_token_capacity,
            "fragments_per_step": self.fragments_per_step,
            "fragments_per_host": self.fragments_per_host,
            "full_fragment_bytes": self.full_fragment_bytes,
            "full_host_step_bytes": self.full_host_step_bytes,
            "full_step_bytes": self.full_step_bytes,
            "writer_slot_count": self.writer_slot_count,
            "maximum_queued_complete_steps": self.maximum_queued_complete_steps,
            "upload_streams_per_host": self.upload_streams_per_host,
            "maximum_object_bytes": self.maximum_object_bytes,
            "maximum_application_slice_bytes": (self.maximum_application_slice_bytes),
            "planning_tokens_per_second": self.planning_tokens_per_second,
            "planning_bytes_per_second": self.planning_bytes_per_second,
            "planning_bytes_per_second_per_host": (
                self.planning_bytes_per_second_per_host
            ),
            "planning_step_duration_ns": {
                "numerator": self.planning_step_duration_ns_numerator,
                "denominator": self.planning_step_duration_ns_denominator,
            },
            "minimum_persisted_ratio": {
                "numerator": self.minimum_persisted_numerator,
                "denominator": self.minimum_persisted_denominator,
            },
            "create_policy": self.create_policy,
            "readback_policy": self.readback_policy,
            "synchronized_path_may_wait_for_writer": (
                self.synchronized_path_may_wait_for_writer
            ),
            "create_ack_is_durable_qualification": (
                self.create_ack_is_durable_qualification
            ),
            "production_sealable": self.production_sealable,
            "production_admission_enabled": self.production_admission_enabled,
        }


PINNED_CAPTURE_DATA_PLANE_CONTRACT = CaptureDataPlaneContractV1()
PINNED_CAPTURE_DATA_PLANE_CONTRACT_SHA256 = (
    "ac61594cd3dd23a27f579fc66020368e75bec6557b3860185b7a8109885a7240"
)
if (
    PINNED_CAPTURE_DATA_PLANE_CONTRACT.digest
    != PINNED_CAPTURE_DATA_PLANE_CONTRACT_SHA256
):
    raise CaptureDataPlaneContractError(
        "pinned capture data-plane contract digest changed: "
        f"{PINNED_CAPTURE_DATA_PLANE_CONTRACT.digest}"
    )


@dataclass(frozen=True, slots=True)
class CaptureFragmentPlanV1(CanonicalRecord):
    """One topology-neutral compacted object derived from a global batch slice."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.capture-fragment-plan.v1"

    work_id: str
    input_range_sha256: str
    contract_sha256: str
    work_start_token_ordinal: int
    work_token_count: int
    step_index: int
    global_batch_start: int
    global_batch_stop: int
    token_ordinal_start: int
    valid_tokens: int
    payload_bytes: int

    def __post_init__(self) -> None:
        _exact_identifier(self.work_id, "work_id")
        _exact_sha256(self.input_range_sha256, "input_range_sha256")
        _exact_sha256(self.contract_sha256, "contract_sha256")
        if self.contract_sha256 != PINNED_CAPTURE_DATA_PLANE_CONTRACT.digest:
            raise CaptureDataPlaneContractError(
                "fragment is not bound to the pinned data-plane contract"
            )
        work_start = _integer(
            self.work_start_token_ordinal,
            "work_start_token_ordinal",
        )
        work_count = _integer(self.work_token_count, "work_token_count", minimum=1)
        step = _integer(self.step_index, "step_index")
        start = _integer(
            self.global_batch_start,
            "global_batch_start",
            maximum=PINNED_GLOBAL_BATCH_SIZE - 1,
        )
        stop = _integer(
            self.global_batch_stop,
            "global_batch_stop",
            minimum=1,
            maximum=PINNED_GLOBAL_BATCH_SIZE,
        )
        if stop - start != 2 or start % 2:
            raise CaptureDataPlaneContractError(
                "fragment must be one aligned two-sequence global batch slice"
            )
        relative_start = (
            step * PINNED_CAPTURE_DATA_PLANE_CONTRACT.full_step_tokens
            + start * PINNED_SEQUENCE_LENGTH
        )
        if relative_start >= work_count:
            raise CaptureDataPlaneContractError(
                "fragment global slice is padding-only for its exact work range"
            )
        token_start = _integer(
            self.token_ordinal_start,
            "token_ordinal_start",
        )
        valid = _integer(
            self.valid_tokens,
            "valid_tokens",
            minimum=1,
            maximum=PINNED_FRAGMENT_TOKEN_CAPACITY,
        )
        expected_valid = min(
            PINNED_FRAGMENT_TOKEN_CAPACITY, work_count - relative_start
        )
        if token_start != work_start + relative_start or valid != expected_valid:
            raise CaptureDataPlaneContractError(
                "fragment token interval differs from its exact work and global slice"
            )
        payload_bytes = _integer(self.payload_bytes, "payload_bytes", minimum=1)
        if payload_bytes != valid * PINNED_CAPTURE_ROW_BYTES:
            raise CaptureDataPlaneContractError(
                "fragment payload bytes do not match compacted valid rows"
            )

    @property
    def fragment_index_within_step(self) -> int:
        return self.global_batch_start // 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "work_id": self.work_id,
            "input_range_sha256": self.input_range_sha256,
            "contract_sha256": self.contract_sha256,
            "work_start_token_ordinal": self.work_start_token_ordinal,
            "work_token_count": self.work_token_count,
            "step_index": self.step_index,
            "fragment_index_within_step": self.fragment_index_within_step,
            "global_batch_start": self.global_batch_start,
            "global_batch_stop": self.global_batch_stop,
            "token_ordinal_start": self.token_ordinal_start,
            "valid_tokens": self.valid_tokens,
            "payload_bytes": self.payload_bytes,
        }


@dataclass(frozen=True, slots=True)
class CaptureWorkPlanV1(CanonicalRecord):
    """Compact immutable input-range plan; fragments are derived lazily."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.capture-work-plan.v1"

    work_id: str
    input_range_sha256: str
    contract_sha256: str
    start_token_ordinal: int
    token_count: int
    step_count: int
    fragment_count: int
    payload_bytes: int
    production_sealable: bool = False

    def __post_init__(self) -> None:
        _exact_identifier(self.work_id, "work_id")
        _exact_sha256(self.input_range_sha256, "input_range_sha256")
        _exact_sha256(self.contract_sha256, "contract_sha256")
        if self.contract_sha256 != PINNED_CAPTURE_DATA_PLANE_CONTRACT.digest:
            raise CaptureDataPlaneContractError(
                "work is not bound to the pinned data-plane contract"
            )
        _integer(self.start_token_ordinal, "start_token_ordinal")
        count = _integer(self.token_count, "token_count", minimum=1)
        expected_steps = math.ceil(
            count / PINNED_CAPTURE_DATA_PLANE_CONTRACT.full_step_tokens
        )
        expected_fragments = math.ceil(count / PINNED_FRAGMENT_TOKEN_CAPACITY)
        steps = _integer(self.step_count, "step_count", minimum=1)
        fragments = _integer(self.fragment_count, "fragment_count", minimum=1)
        payload_bytes = _integer(self.payload_bytes, "payload_bytes", minimum=1)
        if (
            steps != expected_steps
            or fragments != expected_fragments
            or payload_bytes != count * PINNED_CAPTURE_ROW_BYTES
        ):
            raise CaptureDataPlaneContractError(
                "work summary differs from deterministic fragment arithmetic"
            )
        if type(self.production_sealable) is not bool or self.production_sealable:
            raise CaptureDataPlaneContractError(
                "an unvalidated capture work plan cannot seal production"
            )

    def fragment_for_global_batch_slice(
        self,
        step_index: int,
        global_batch_start: int,
        global_batch_stop: int,
    ) -> CaptureFragmentPlanV1 | None:
        """Map an addressable shard index to science identity, independent of host."""

        step = _integer(
            step_index,
            "step_index",
            maximum=self.step_count - 1,
        )
        start = _integer(
            global_batch_start,
            "global_batch_start",
            maximum=PINNED_GLOBAL_BATCH_SIZE - 1,
        )
        stop = _integer(
            global_batch_stop,
            "global_batch_stop",
            minimum=1,
            maximum=PINNED_GLOBAL_BATCH_SIZE,
        )
        if stop - start != 2 or start % 2:
            raise CaptureDataPlaneContractError(
                "addressable shard must cover one aligned two-sequence slice"
            )
        relative_start = (
            step * PINNED_CAPTURE_DATA_PLANE_CONTRACT.full_step_tokens
            + start * PINNED_SEQUENCE_LENGTH
        )
        remaining = self.token_count - relative_start
        if remaining <= 0:
            return None
        valid_tokens = min(PINNED_FRAGMENT_TOKEN_CAPACITY, remaining)
        return CaptureFragmentPlanV1(
            work_id=self.work_id,
            input_range_sha256=self.input_range_sha256,
            contract_sha256=self.contract_sha256,
            work_start_token_ordinal=self.start_token_ordinal,
            work_token_count=self.token_count,
            step_index=step,
            global_batch_start=start,
            global_batch_stop=stop,
            token_ordinal_start=self.start_token_ordinal + relative_start,
            valid_tokens=valid_tokens,
            payload_bytes=valid_tokens * PINNED_CAPTURE_ROW_BYTES,
        )

    def iter_fragments(self) -> Iterator[CaptureFragmentPlanV1]:
        for step_index in range(self.step_count):
            for start in range(0, PINNED_GLOBAL_BATCH_SIZE, 2):
                fragment = self.fragment_for_global_batch_slice(
                    step_index,
                    start,
                    start + 2,
                )
                if fragment is not None:
                    yield fragment

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "work_id": self.work_id,
            "input_range_sha256": self.input_range_sha256,
            "contract_sha256": self.contract_sha256,
            "start_token_ordinal": self.start_token_ordinal,
            "token_count": self.token_count,
            "step_count": self.step_count,
            "fragment_count": self.fragment_count,
            "payload_bytes": self.payload_bytes,
            "production_sealable": self.production_sealable,
        }


def build_capture_work_plan(
    *,
    work_id: str,
    input_range_sha256: str,
    start_token_ordinal: int,
    token_count: int,
) -> CaptureWorkPlanV1:
    """Build one compact exact-range plan without materializing its fragments."""

    _exact_identifier(work_id, "work_id")
    _exact_sha256(input_range_sha256, "input_range_sha256")
    start = _integer(start_token_ordinal, "start_token_ordinal")
    count = _integer(token_count, "token_count", minimum=1)
    contract = PINNED_CAPTURE_DATA_PLANE_CONTRACT
    return CaptureWorkPlanV1(
        work_id=work_id,
        input_range_sha256=input_range_sha256,
        contract_sha256=contract.digest,
        start_token_ordinal=start,
        token_count=count,
        step_count=math.ceil(count / contract.full_step_tokens),
        fragment_count=math.ceil(count / contract.fragment_token_capacity),
        payload_bytes=count * contract.row_bytes,
    )


def fragment_content_object_id(payload_sha256: str) -> str:
    """Return the byte store's reserved path; its attempt prefix is separate."""

    digest = _exact_sha256(payload_sha256, "payload_sha256")
    try:
        return content_addressed_object_id(digest)
    except ProtocolError as error:
        raise CaptureDataPlaneContractError(
            "payload SHA-256 cannot form a canonical object ID"
        ) from error


__all__ = [
    "CaptureDataPlaneContractError",
    "CaptureDataPlaneContractV1",
    "CaptureFragmentPlanV1",
    "CapturePayloadFieldV1",
    "CaptureWorkPlanV1",
    "PINNED_CAPTURE_DATA_PLANE_CONTRACT",
    "PINNED_CAPTURE_DATA_PLANE_CONTRACT_SHA256",
    "PINNED_CAPTURE_PAYLOAD_FIELDS",
    "PINNED_CAPTURE_ROW_BYTES",
    "PINNED_FRAGMENT_TOKEN_CAPACITY",
    "PINNED_FULL_FRAGMENT_BYTES",
    "PINNED_FULL_HOST_STEP_BYTES",
    "PINNED_FULL_STEP_BYTES",
    "PINNED_FRAGMENTS_PER_HOST",
    "PINNED_FRAGMENTS_PER_STEP",
    "PINNED_MAX_QUEUED_COMPLETE_STEPS",
    "PINNED_PLANNING_TOKENS_PER_SECOND",
    "PINNED_UPLOAD_STREAMS_PER_HOST",
    "PINNED_WRITER_SLOT_COUNT",
    "build_capture_work_plan",
    "fragment_content_object_id",
]
