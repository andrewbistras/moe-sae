"""Deterministic host codec for one topology-neutral R1 capture fragment.

The numerical prefix hands this module one already-host-resident addressable
two-sequence shard.  The codec validates the small router and metadata fields,
canonicalizes expert-ID/weight pairs by ascending global expert ID, casts the
durable fields, and emits the exact headerless columnar byte layout frozen by
``capture_data_plane_contract``.

This is synchronous fragment work intended for a future background writer
thread.  It imports neither JAX nor MaxText and performs no cloud action.
Successful packing is not persistence, readback, qualification, or manifest
publication evidence.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import sys
from types import MappingProxyType
from typing import Mapping

import google_crc32c
import ml_dtypes
import numpy as np

from .capture_data_plane_contract import (
    CaptureDataPlaneContractError,
    CaptureFragmentPlanV1,
    PINNED_CAPTURE_DATA_PLANE_CONTRACT,
    fragment_content_object_id,
)
from .prefix_batchsplit_contract import PINNED_SEQUENCE_LENGTH
from .protocol import ProtocolError, SHA256_PATTERN, require_string
from .router_oracle import (
    PINNED_R1_ROUTER_GEOMETRY,
    RouterOracleError,
    canonicalize_numpy_router_result,
)


_BF16 = np.dtype(ml_dtypes.bfloat16)
_F32 = np.dtype(np.float32)
_I32 = np.dtype(np.int32)
_U16 = np.dtype(np.uint16)
_I64 = np.dtype(np.int64)
_PACKED_CAPABILITY = object()
_DECODED_CAPABILITY = object()
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))

_FIELD_DTYPES = {
    "routed_activation": _BF16,
    "global_expert_ids": _U16,
    "scaled_router_weights": _BF16,
    "sequence_id": _I64,
    "position": _U16,
}
_ADDRESSABLE_SHAPES = {
    "routed_activation": (2, PINNED_SEQUENCE_LENGTH, 7_168),
    "global_expert_ids": (2, PINNED_SEQUENCE_LENGTH, 8),
    "scaled_router_weights": (2, PINNED_SEQUENCE_LENGTH, 8),
    "sequence_id": (2, PINNED_SEQUENCE_LENGTH),
    "position": (2, PINNED_SEQUENCE_LENGTH),
}
_ADDRESSABLE_DTYPES = {
    "routed_activation": _BF16,
    "global_expert_ids": _I32,
    "scaled_router_weights": _F32,
    "sequence_id": _I64,
    "position": _U16,
}


class CaptureFragmentCodecError(CaptureDataPlaneContractError):
    """A host shard or encoded payload differs from the frozen durable ABI."""


def _exact_array(value: object, name: str) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise CaptureFragmentCodecError(f"{name} must be an exact NumPy array")
    expected_shape = _ADDRESSABLE_SHAPES[name]
    expected_dtype = _ADDRESSABLE_DTYPES[name]
    if value.shape != expected_shape or value.dtype != expected_dtype:
        raise CaptureFragmentCodecError(
            f"{name} must have shape {expected_shape} and dtype {expected_dtype}"
        )
    if not value.flags.c_contiguous:
        raise CaptureFragmentCodecError(f"{name} must be C-contiguous")
    return value


@dataclass(frozen=True, slots=True)
class AddressableCaptureShardV1:
    """One host-resident shard; construction transfers read-only ownership."""

    routed_activation: np.ndarray
    global_expert_ids: np.ndarray
    scaled_router_weights: np.ndarray
    sequence_id: np.ndarray
    position: np.ndarray

    def __post_init__(self) -> None:
        for name in _ADDRESSABLE_SHAPES:
            value = _exact_array(getattr(self, name), name)
            value.setflags(write=False)
            if value.flags.writeable:
                raise CaptureFragmentCodecError(
                    f"{name} could not enter read-only writer ownership"
                )


def _little_endian_contiguous(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    target = dtype.newbyteorder("<")
    result = np.ascontiguousarray(value, dtype=target)
    if result.dtype.byteorder == ">" or (
        result.dtype.byteorder == "=" and sys.byteorder != "little"
    ):
        result = result.byteswap().view(target)
    if not result.flags.c_contiguous:
        raise CaptureFragmentCodecError("canonical durable field is not contiguous")
    return result


def _validate_metadata(
    sequence_id: np.ndarray,
    position: np.ndarray,
    valid_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequence_flat = sequence_id.reshape(-1)[:valid_tokens]
    position_flat = position.reshape(-1)[:valid_tokens]
    if np.any(sequence_flat < 0):
        raise CaptureFragmentCodecError("retained sequence IDs must be nonnegative")
    expected_positions = np.tile(
        np.arange(PINNED_SEQUENCE_LENGTH, dtype=np.uint16),
        2,
    )[:valid_tokens]
    if not np.array_equal(position_flat, expected_positions):
        raise CaptureFragmentCodecError(
            "retained positions are not the exact zero-based sequence positions"
        )
    if valid_tokens > PINNED_SEQUENCE_LENGTH:
        first = sequence_flat[:PINNED_SEQUENCE_LENGTH]
        second = sequence_flat[PINNED_SEQUENCE_LENGTH:]
        if (
            np.any(first != first[0])
            or np.any(second != second[0])
            or first[0] == second[0]
        ):
            raise CaptureFragmentCodecError(
                "sequence IDs must be constant and distinct across retained sequences"
            )
    elif np.any(sequence_flat != sequence_flat[0]):
        raise CaptureFragmentCodecError(
            "sequence ID must be constant within the retained sequence"
        )
    return (
        _little_endian_contiguous(sequence_flat, _I64),
        _little_endian_contiguous(position_flat, _U16),
    )


def _canonical_router_fields(
    expert_ids: np.ndarray,
    router_weights: np.ndarray,
    valid_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    ids_flat = expert_ids.reshape(-1, 8)[:valid_tokens]
    weights_flat = router_weights.reshape(-1, 8)[:valid_tokens]
    try:
        canonical = canonicalize_numpy_router_result(
            ids_flat,
            weights_flat,
            PINNED_R1_ROUTER_GEOMETRY,
            expected_leading_shape=(valid_tokens,),
        )
    except RouterOracleError as error:
        raise CaptureFragmentCodecError(
            "retained router fields violate the exact R1 router contract"
        ) from error
    ids = _little_endian_contiguous(canonical.expert_ids, _U16)
    weights = _little_endian_contiguous(canonical.router_weights, _BF16)
    return ids, weights


def _canonical_field_arrays(
    plan: CaptureFragmentPlanV1,
    shard: AddressableCaptureShardV1,
) -> Mapping[str, np.ndarray]:
    if type(plan) is not CaptureFragmentPlanV1:
        raise CaptureFragmentCodecError("plan must be an exact fragment plan")
    if type(shard) is not AddressableCaptureShardV1:
        raise CaptureFragmentCodecError(
            "shard must be an exact addressable capture shard"
        )
    valid = plan.valid_tokens
    activation = _little_endian_contiguous(
        shard.routed_activation.reshape(-1, 7_168)[:valid],
        _BF16,
    )
    expert_ids, router_weights = _canonical_router_fields(
        shard.global_expert_ids,
        shard.scaled_router_weights,
        valid,
    )
    sequence_id, position = _validate_metadata(
        shard.sequence_id,
        shard.position,
        valid,
    )
    result = {
        "routed_activation": activation,
        "global_expert_ids": expert_ids,
        "scaled_router_weights": router_weights,
        "sequence_id": sequence_id,
        "position": position,
    }
    if tuple(result) != tuple(
        item.name for item in PINNED_CAPTURE_DATA_PLANE_CONTRACT.payload_fields
    ):
        raise CaptureFragmentCodecError("codec field order differs from the contract")
    return MappingProxyType(result)


def _crc32c_base64(payload: bytes) -> str:
    checksum = google_crc32c.Checksum()
    checksum.update(payload)
    return base64.b64encode(checksum.digest()).decode("ascii")


@dataclass(frozen=True, slots=True, init=False)
class PackedCaptureFragmentV1:
    """Exact immutable bytes and local hashes; never a persistence receipt."""

    plan: CaptureFragmentPlanV1
    payload: bytes
    sha256: str
    crc32c: str
    object_id: str
    field_byte_ranges: tuple[tuple[str, int, int], ...]
    persisted: bool
    readback_verified: bool
    production_sealable: bool

    def __init__(
        self,
        plan: CaptureFragmentPlanV1,
        payload: bytes,
        *,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _PACKED_CAPABILITY:
            raise CaptureFragmentCodecError(
                "packed fragments are available only through the codec"
            )
        if type(plan) is not CaptureFragmentPlanV1 or type(payload) is not bytes:
            raise CaptureFragmentCodecError("packed fragment state has invalid types")
        if len(payload) != plan.payload_bytes:
            raise CaptureFragmentCodecError(
                "packed payload size differs from its fragment plan"
            )
        ranges = PINNED_CAPTURE_DATA_PLANE_CONTRACT.field_byte_ranges(plan.valid_tokens)
        if ranges[-1][2] != len(payload):
            raise CaptureFragmentCodecError("packed field ranges do not cover payload")
        sha256 = hashlib.sha256(payload).hexdigest()
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(self, "crc32c", _crc32c_base64(payload))
        object.__setattr__(self, "object_id", fragment_content_object_id(sha256))
        object.__setattr__(self, "field_byte_ranges", ranges)
        object.__setattr__(self, "persisted", False)
        object.__setattr__(self, "readback_verified", False)
        object.__setattr__(self, "production_sealable", False)


def pack_capture_fragment(
    plan: CaptureFragmentPlanV1,
    shard: AddressableCaptureShardV1,
) -> PackedCaptureFragmentV1:
    """Snapshot one shard into the exact immutable columnar payload."""

    fields = _canonical_field_arrays(plan, shard)
    payload_buffer = bytearray(plan.payload_bytes)
    destination = memoryview(payload_buffer)
    try:
        for name, start, stop in PINNED_CAPTURE_DATA_PLANE_CONTRACT.field_byte_ranges(
            plan.valid_tokens
        ):
            source_array = fields[name]
            source = memoryview(source_array.view(np.uint8).reshape(-1))
            try:
                if len(source) != stop - start:
                    raise CaptureFragmentCodecError(
                        f"{name} bytes differ from the frozen field interval"
                    )
                destination[start:stop] = source
            finally:
                source.release()
    finally:
        destination.release()
    return PackedCaptureFragmentV1(
        plan,
        bytes(payload_buffer),
        _capability=_PACKED_CAPABILITY,
    )


@dataclass(frozen=True, slots=True, init=False)
class DecodedCaptureFragmentV1:
    """Read-only zero-copy views over an independently hash-checked payload."""

    plan: CaptureFragmentPlanV1
    payload_sha256: str
    fields: Mapping[str, np.ndarray]

    def __init__(
        self,
        plan: CaptureFragmentPlanV1,
        payload_sha256: str,
        fields: Mapping[str, np.ndarray],
        *,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _DECODED_CAPABILITY:
            raise CaptureFragmentCodecError(
                "decoded fragments are available only through the codec"
            )
        if (
            type(plan) is not CaptureFragmentPlanV1
            or type(payload_sha256) is not str
            or type(fields) is not _MAPPING_PROXY_TYPE
            or tuple(fields)
            != tuple(
                item.name for item in PINNED_CAPTURE_DATA_PLANE_CONTRACT.payload_fields
            )
        ):
            raise CaptureFragmentCodecError("decoded fragment state is invalid")
        if any(
            type(value) is not np.ndarray
            or value.flags.writeable
            or not value.flags.c_contiguous
            for value in fields.values()
        ):
            raise CaptureFragmentCodecError(
                "decoded fields must be exact read-only contiguous arrays"
            )
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(self, "fields", fields)


def decode_capture_fragment(
    plan: CaptureFragmentPlanV1,
    payload: bytes,
    *,
    expected_sha256: str,
) -> DecodedCaptureFragmentV1:
    """Verify and expose one immutable payload without copying its columns."""

    if type(plan) is not CaptureFragmentPlanV1 or type(payload) is not bytes:
        raise CaptureFragmentCodecError("decode inputs have invalid types")
    if type(expected_sha256) is not str:
        raise CaptureFragmentCodecError("expected_sha256 must be an exact SHA-256")
    try:
        require_string(
            expected_sha256,
            "expected_sha256",
            pattern=SHA256_PATTERN,
        )
    except ProtocolError as error:
        raise CaptureFragmentCodecError(
            "expected_sha256 must be an exact SHA-256"
        ) from error
    if len(payload) != plan.payload_bytes:
        raise CaptureFragmentCodecError(
            "encoded payload size differs from its fragment plan"
        )
    measured_sha256 = hashlib.sha256(payload).hexdigest()
    if measured_sha256 != expected_sha256:
        raise CaptureFragmentCodecError("encoded payload SHA-256 differs")
    count = plan.valid_tokens
    shapes = {
        "routed_activation": (count, 7_168),
        "global_expert_ids": (count, 8),
        "scaled_router_weights": (count, 8),
        "sequence_id": (count,),
        "position": (count,),
    }
    fields: dict[str, np.ndarray] = {}
    for name, start, stop in PINNED_CAPTURE_DATA_PLANE_CONTRACT.field_byte_ranges(
        count
    ):
        expected_elements = math_prod(shapes[name])
        values = np.frombuffer(
            payload,
            dtype=_FIELD_DTYPES[name].newbyteorder("<"),
            count=expected_elements,
            offset=start,
        )
        if values.size != expected_elements or start + values.nbytes != stop:
            raise CaptureFragmentCodecError(
                f"{name} payload interval has the wrong element count"
            )
        values = values[:expected_elements].reshape(shapes[name])
        values.setflags(write=False)
        fields[name] = values
    ids = fields["global_expert_ids"]
    if ids.shape[-1] > 1 and np.any(np.diff(ids.astype(np.int32), axis=-1) <= 0):
        raise CaptureFragmentCodecError(
            "durable expert IDs are not unique ascending global IDs"
        )
    weights = fields["scaled_router_weights"].astype(np.float32)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise CaptureFragmentCodecError("durable router weights are invalid")
    _validate_metadata(
        fields["sequence_id"],
        fields["position"],
        count,
    )
    return DecodedCaptureFragmentV1(
        plan,
        measured_sha256,
        MappingProxyType(fields),
        _capability=_DECODED_CAPABILITY,
    )


def math_prod(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


__all__ = [
    "AddressableCaptureShardV1",
    "CaptureFragmentCodecError",
    "DecodedCaptureFragmentV1",
    "PackedCaptureFragmentV1",
    "decode_capture_fragment",
    "pack_capture_fragment",
]
