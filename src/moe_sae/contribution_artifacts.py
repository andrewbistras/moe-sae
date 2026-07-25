"""Fail-closed artifacts for Experiment 2 selected-expert contributions.

This module intentionally does not extend :mod:`moe_sae.artifacts`.  The
aggregate capture and the contribution capture are different scientific
objects, and accepting one where the other is expected would silently change
the experiment.

BF16 tensors use their exact bit-patterns in NumPy-facing code.  In other
words, ``uint16`` is the storage dtype and ``bfloat16`` is the logical dtype.
This avoids a dependency on a particular NumPy/ml_dtypes combination while
preserving every stored operand bit-for-bit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SCHEMA_VERSION = "gemma4-expert-contribution-v2"
RECOMPOSITION_SIDECAR_SCHEMA_VERSION = "gemma4-recomposition-sidecar-v1"
LEGACY_AGGREGATE_SCHEMA_VERSION = "gemma4-router-aggregate-v1"
PROTOCOL_ID = "gemma4-contribution-v1-20260725"

MODEL_ID = "google/gemma-4-26B-A4B"
MAXTEXT_BASE_COMMIT = "48c7da3e28e8f57fe36ddf09d00eec53072a2ea0"
TARGET_LAYER_ONE_INDEXED = 15
HIDDEN_SIZE = 2_816
NUM_EXPERTS = 128
TOP_K_EXPERTS = 8
SEQUENCE_LENGTH = 1_024
MAX_SHARD_TOKENS = 16_384
RMS_NORM_EPSILON = 1e-6
ROUTER_WEIGHT_SUM_ATOL = 0.005

CAPTURE_SITE = "routed_moe.selected_outputs.pre_router_weight"
SHARED_CAPTURE_SITE = "shared_expert.raw_output"

BF16_LOGICAL_DTYPE = "bfloat16"
BF16_STORAGE_DTYPE = "uint16"

TENSOR_SCHEMA: dict[str, dict[str, Any]] = {
    "expert_output": {
        "logical_dtype": BF16_LOGICAL_DTYPE,
        "storage_dtype": BF16_STORAGE_DTYPE,
        "shape": ["N", TOP_K_EXPERTS, HIDDEN_SIZE],
    },
    "expert_ids": {
        "logical_dtype": "uint8",
        "storage_dtype": "uint8",
        "shape": ["N", TOP_K_EXPERTS],
    },
    "router_weights": {
        "logical_dtype": BF16_LOGICAL_DTYPE,
        "storage_dtype": BF16_STORAGE_DTYPE,
        "shape": ["N", TOP_K_EXPERTS],
    },
    "shared_output": {
        "logical_dtype": BF16_LOGICAL_DTYPE,
        "storage_dtype": BF16_STORAGE_DTYPE,
        "shape": ["N", HIDDEN_SIZE],
    },
    "sequence_id": {
        "logical_dtype": "int64",
        "storage_dtype": "int64",
        "shape": ["N"],
    },
    "position": {
        "logical_dtype": "uint16",
        "storage_dtype": "uint16",
        "shape": ["N"],
    },
}

SIDECAR_TENSOR_SCHEMA: dict[str, dict[str, Any]] = {
    "routed_rmsnorm_scale": {
        "logical_dtype": BF16_LOGICAL_DTYPE,
        "storage_dtype": BF16_STORAGE_DTYPE,
        "shape": [HIDDEN_SIZE],
    },
    "shared_rmsnorm_scale": {
        "logical_dtype": BF16_LOGICAL_DTYPE,
        "storage_dtype": BF16_STORAGE_DTYPE,
        "shape": [HIDDEN_SIZE],
    },
    "outer_ffn_rmsnorm_scale": {
        "logical_dtype": BF16_LOGICAL_DTYPE,
        "storage_dtype": BF16_STORAGE_DTYPE,
        "shape": [HIDDEN_SIZE],
    },
    "layer_scalar": {
        "logical_dtype": BF16_LOGICAL_DTYPE,
        "storage_dtype": BF16_STORAGE_DTYPE,
        "shape": [1],
    },
}

DTYPE_POLICY: dict[str, str] = {
    "stored_expert_output": BF16_LOGICAL_DTYPE,
    "stored_router_weights": BF16_LOGICAL_DTYPE,
    "stored_shared_output": BF16_LOGICAL_DTYPE,
    "router_multiply": "float32",
    "routed_accumulator": "float32",
    "routed_matmul_precision": "default",
    "rmsnorm_reduction": "float32",
    "rmsnorm_output": BF16_LOGICAL_DTYPE,
    "outer_addition": BF16_LOGICAL_DTYPE,
    "layer_scalar_multiply": BF16_LOGICAL_DTYPE,
}

_CAPTURE_FIELDS = tuple(TENSOR_SCHEMA)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ContributionArtifactError(RuntimeError):
    """Base class for Experiment 2 artifact failures."""


class ContributionSchemaError(ContributionArtifactError):
    """A contribution object does not satisfy the v2 schema."""


class ContributionLegacySchemaError(ContributionSchemaError):
    """An aggregate, missing, or otherwise non-v2 schema was supplied."""


class ContributionIntegrityError(ContributionArtifactError):
    """A hash, count, statistic, or identity invariant failed."""


class ContributionRangeError(ContributionIntegrityError):
    """An immutable scientific token range is malformed."""


@dataclasses.dataclass(frozen=True, slots=True)
class ContributionBatch:
    """Exact NumPy wire representation for one v2 shard."""

    expert_output: np.ndarray
    expert_ids: np.ndarray
    router_weights: np.ndarray
    shared_output: np.ndarray
    sequence_id: np.ndarray
    position: np.ndarray

    @property
    def tokens(self) -> int:
        return int(np.asarray(self.expert_output).shape[0])


@dataclasses.dataclass(frozen=True, slots=True)
class ContributionMetadata:
    """Scientific identity common to every contribution shard."""

    checkpoint_sha256: str
    maxtext_fork_commit: str
    dataset_fingerprint: str
    split: str
    config_sha256: str
    environment_sha256: str
    source_v1_manifest_sha256: str
    recomposition_sidecar_sha256: str
    protocol_id: str = PROTOCOL_ID
    model_id: str = MODEL_ID
    maxtext_base_commit: str = MAXTEXT_BASE_COMMIT
    target_layer_one_indexed: int = TARGET_LAYER_ONE_INDEXED
    sequence_length: int = SEQUENCE_LENGTH
    capture_site: str = CAPTURE_SITE
    shared_capture_site: str = SHARED_CAPTURE_SITE

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value.update(
            {
                "hidden_size": HIDDEN_SIZE,
                "num_experts": NUM_EXPERTS,
                "top_k_experts": TOP_K_EXPERTS,
                "max_shard_tokens": MAX_SHARD_TOKENS,
                "tensor_schema": TENSOR_SCHEMA,
            }
        )
        return validate_metadata(value)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON encoding used for every identity hash."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ContributionSchemaError(
            "artifact identities may contain only finite canonical JSON values"
        ) from exc


def canonical_sha256(value: Any) -> str:
    """Hash a value after canonical JSON encoding."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContributionSchemaError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_positive_int(name: str, value: Any, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ContributionSchemaError(f"{name} must be a {qualifier} integer")
    return value


def float32_to_bf16_bits(values: Any) -> np.ndarray:
    """Round FP32 to BF16 (ties-to-even) and return the exact uint16 bits."""

    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_float32(values: Any) -> np.ndarray:
    """Expand exact BF16 uint16 bit-patterns to FP32."""

    array = np.asarray(values)
    if array.dtype != np.uint16:
        raise ContributionSchemaError(
            f"BF16 wire values must have dtype uint16, got {array.dtype}"
        )
    return (array.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _bf16_bits_all_finite(values: np.ndarray) -> bool:
    """Check BF16 finiteness without expanding a capture-sized FP32 array."""

    flat = np.asarray(values).reshape(-1)
    # Keep temporary masks bounded.  A full B128 controller payload contains
    # hundreds of millions of expert elements; expanding it to FP32 merely to
    # inspect the exponent doubled both host traffic and peak memory.
    block_elements = 8 * 1024 * 1024
    exponent_mask = np.uint16(0x7F80)
    for start in range(0, int(flat.size), block_elements):
        block = flat[start : start + block_elements]
        if np.any(np.bitwise_and(block, exponent_mask) == exponent_mask):
            return False
    return True


def _as_batch(value: ContributionBatch | Mapping[str, Any]) -> ContributionBatch:
    if isinstance(value, ContributionBatch):
        return value
    if not isinstance(value, Mapping):
        raise ContributionSchemaError(
            "contribution batch must be a ContributionBatch or mapping"
        )
    keys = set(value)
    if "activation" in keys:
        raise ContributionLegacySchemaError(
            "aggregate activation shards cannot be interpreted as contribution shards"
        )
    if keys != set(_CAPTURE_FIELDS):
        raise ContributionSchemaError(
            "contribution tensor keys must be exactly "
            f"{sorted(_CAPTURE_FIELDS)}, got {sorted(str(key) for key in keys)}"
        )
    return ContributionBatch(
        **{name: np.asarray(value[name]) for name in _CAPTURE_FIELDS}
    )


def _require_exact_dtype(name: str, array: np.ndarray, expected: np.dtype[Any]) -> None:
    if array.dtype != expected:
        raise ContributionSchemaError(
            f"{name} must have wire dtype {expected}, got {array.dtype}"
        )


def validate_contribution_batch(
    value: ContributionBatch | Mapping[str, Any],
    *,
    max_tokens: int = MAX_SHARD_TOKENS,
) -> ContributionBatch:
    """Validate exact v2 shapes, wire dtypes, route order, and token identity."""

    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 0 < max_tokens <= MAX_SHARD_TOKENS
    ):
        raise ContributionSchemaError(f"max_tokens must be in [1,{MAX_SHARD_TOKENS}]")
    batch = _as_batch(value)
    arrays = {field: np.asarray(getattr(batch, field)) for field in _CAPTURE_FIELDS}
    for name in ("expert_output", "router_weights", "shared_output"):
        _require_exact_dtype(name, arrays[name], np.dtype(np.uint16))
    _require_exact_dtype("expert_ids", arrays["expert_ids"], np.dtype(np.uint8))
    _require_exact_dtype("sequence_id", arrays["sequence_id"], np.dtype(np.int64))
    _require_exact_dtype("position", arrays["position"], np.dtype(np.uint16))

    expert_output = arrays["expert_output"]
    if expert_output.ndim != 3 or expert_output.shape[1:] != (
        TOP_K_EXPERTS,
        HIDDEN_SIZE,
    ):
        raise ContributionSchemaError(
            "expert_output must have shape "
            f"[N,{TOP_K_EXPERTS},{HIDDEN_SIZE}], got {expert_output.shape}"
        )
    tokens = int(expert_output.shape[0])
    if not 0 < tokens <= max_tokens:
        raise ContributionSchemaError(
            f"shard token count must be in [1,{max_tokens}], got {tokens}"
        )
    expected_shapes = {
        "expert_ids": (tokens, TOP_K_EXPERTS),
        "router_weights": (tokens, TOP_K_EXPERTS),
        "shared_output": (tokens, HIDDEN_SIZE),
        "sequence_id": (tokens,),
        "position": (tokens,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ContributionSchemaError(
                f"{name} must have shape {shape}, got {arrays[name].shape}"
            )

    decoded_weights = bf16_bits_to_float32(arrays["router_weights"])
    if not _bf16_bits_all_finite(expert_output):
        raise ContributionSchemaError("expert_output contains non-finite values")
    if not _bf16_bits_all_finite(arrays["shared_output"]):
        raise ContributionSchemaError("shared_output contains non-finite values")
    if not np.isfinite(decoded_weights).all() or np.any(decoded_weights < 0):
        raise ContributionSchemaError("router_weights must be finite and nonnegative")
    if not np.allclose(
        decoded_weights.sum(axis=1, dtype=np.float32),
        1.0,
        atol=ROUTER_WEIGHT_SUM_ATOL,
        rtol=0.0,
    ):
        raise ContributionSchemaError("router_weights must sum to one for every token")

    expert_ids = arrays["expert_ids"]
    if np.any(expert_ids >= NUM_EXPERTS):
        raise ContributionSchemaError(f"expert_ids must be in [0,{NUM_EXPERTS - 1}]")
    sorted_ids = np.sort(expert_ids, axis=1)
    if np.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
        raise ContributionSchemaError("each token must contain eight unique expert_ids")
    heavier_after = decoded_weights[:, 1:] > decoded_weights[:, :-1]
    equal_weight = decoded_weights[:, 1:] == decoded_weights[:, :-1]
    descending_id_tie = expert_ids[:, 1:] < expert_ids[:, :-1]
    if np.any(heavier_after) or np.any(equal_weight & descending_id_tie):
        raise ContributionSchemaError(
            "router slots must be ordered by descending weight with expert ID "
            "as the ascending deterministic tie-breaker"
        )

    sequence_id = arrays["sequence_id"]
    position = arrays["position"]
    if np.any(sequence_id < 0):
        raise ContributionSchemaError("sequence_id must be nonnegative")
    if np.any(position >= SEQUENCE_LENGTH):
        raise ContributionSchemaError(f"position must be in [0,{SEQUENCE_LENGTH - 1}]")
    token_keys = np.rec.fromarrays(
        [sequence_id, position], names=("sequence_id", "position")
    )
    if np.unique(token_keys).size != tokens:
        raise ContributionSchemaError(
            "a shard cannot contain duplicate (sequence_id, position) tokens"
        )

    return ContributionBatch(
        **{name: np.ascontiguousarray(arrays[name]) for name in _CAPTURE_FIELDS}
    )


def token_identity_sha256(
    sequence_id: Any,
    position: Any,
) -> str:
    """Hash the ordered scientific token-key sequence without JSON expansion.

    The byte format is explicitly versioned and little-endian.  Both the v1
    source range and the v2 payload must independently produce this digest.
    It therefore detects the otherwise invisible case where a valid payload
    is attached to the numeric bounds of a different source range.
    """

    sequence = np.asarray(sequence_id)
    positions = np.asarray(position)
    if sequence.dtype != np.int64 or positions.dtype != np.uint16:
        raise ContributionSchemaError(
            "token identity requires int64 sequence_id and uint16 position"
        )
    if sequence.ndim != 1 or positions.shape != sequence.shape or not sequence.size:
        raise ContributionSchemaError(
            "token identity arrays must be nonempty, aligned rank-one vectors"
        )
    if np.any(sequence < 0) or np.any(positions >= SEQUENCE_LENGTH):
        raise ContributionSchemaError("token identity contains an invalid key")
    records = np.empty(
        sequence.size,
        dtype=np.dtype([("sequence_id", "<i8"), ("position", "<u2")]),
    )
    records["sequence_id"] = sequence
    records["position"] = positions
    digest = hashlib.sha256()
    digest.update(b"gemma4-scientific-token-identity-v1\0")
    digest.update(int(sequence.size).to_bytes(8, "little", signed=False))
    digest.update(records.tobytes(order="C"))
    return digest.hexdigest()


def contribution_batch_token_identity_sha256(
    value: ContributionBatch | Mapping[str, Any],
) -> str:
    """Return the ordered token-key identity of a validated v2 batch."""

    batch = validate_contribution_batch(value)
    return token_identity_sha256(batch.sequence_id, batch.position)


def _finite_vector(name: str, value: Any, length: int) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ContributionIntegrityError(
            f"{name} must be a finite FP64 vector of length {length}"
        )
    return [float(item) for item in array]


def _finite_nonnegative_vector(name: str, value: Any, length: int) -> list[float]:
    result = _finite_vector(name, value, length)
    if any(item < 0 for item in result):
        raise ContributionIntegrityError(f"{name} must be nonnegative")
    return result


def _finite_nonnegative_scalar(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContributionIntegrityError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ContributionIntegrityError(f"{name} must be finite and nonnegative")
    return result


def compute_contribution_statistics(
    value: ContributionBatch | Mapping[str, Any],
    *,
    normalized_shared_output: Any,
) -> dict[str, Any]:
    """Compute FP64 sufficient statistics for the Experiment 2 objectives.

    The pooled raw-expert mean is common to A/R/B/F.  Weighted centering uses
    ``sum(g**2 * e)`` and ``sum(g**2)`` because the scientific residual is
    ``g * (e - e_hat)``.  The caller supplies the already normalized shared
    branch so this module does not duplicate the pinned MaxText RMSNorm.
    """

    batch = validate_contribution_batch(value)
    normalized_shared = np.asarray(normalized_shared_output)
    if normalized_shared.shape != (batch.tokens, HIDDEN_SIZE):
        raise ContributionSchemaError(
            "normalized_shared_output must have shape "
            f"{(batch.tokens, HIDDEN_SIZE)}, got {normalized_shared.shape}"
        )
    if normalized_shared.dtype == np.uint16:
        normalized_shared = bf16_bits_to_float32(normalized_shared)
    elif not np.issubdtype(normalized_shared.dtype, np.floating):
        raise ContributionSchemaError(
            "normalized_shared_output must be floating-point or BF16 bits"
        )
    normalized_shared = np.asarray(normalized_shared, dtype=np.float32)
    if not np.isfinite(normalized_shared).all():
        raise ContributionSchemaError(
            "normalized_shared_output contains non-finite values"
        )

    expert = bf16_bits_to_float32(batch.expert_output)
    weights = bf16_bits_to_float32(batch.router_weights)
    shared = bf16_bits_to_float32(batch.shared_output)
    contribution = weights[..., None] * expert
    weight_squared = np.square(weights, dtype=np.float32)
    contribution_energy = np.square(contribution).sum(axis=-1, dtype=np.float64)

    expert_counts = np.bincount(
        batch.expert_ids.reshape(-1), minlength=NUM_EXPERTS
    ).astype(np.int64)
    expert_energy = np.bincount(
        batch.expert_ids.reshape(-1),
        weights=contribution_energy.reshape(-1),
        minlength=NUM_EXPERTS,
    ).astype(np.float64)
    rank_counts = np.full(TOP_K_EXPERTS, batch.tokens, dtype=np.int64)
    rank_energy = contribution_energy.sum(axis=0, dtype=np.float64)

    flat_expert = expert.reshape(-1, HIDDEN_SIZE)
    flat_weight_squared = weight_squared.reshape(-1)
    weighted_output_sum = np.einsum(
        "i,ih->h",
        flat_weight_squared.astype(np.float64),
        flat_expert.astype(np.float64),
        optimize=True,
    )
    shared64 = shared.astype(np.float64)
    normalized_shared64 = normalized_shared.astype(np.float64)
    return {
        "token_count": batch.tokens,
        "component_count": batch.tokens * TOP_K_EXPERTS,
        "pooled_expert_output": {
            "count": batch.tokens * TOP_K_EXPERTS,
            "sum": flat_expert.sum(axis=0, dtype=np.float64).tolist(),
            "squared_norm": float(np.square(flat_expert).sum(dtype=np.float64)),
        },
        "weighted_contribution": {
            "weight_squared_sum": float(flat_weight_squared.sum(dtype=np.float64)),
            "weight_squared_output_sum": weighted_output_sum.tolist(),
            "squared_norm": float(contribution_energy.sum(dtype=np.float64)),
            "raw_sum": contribution.sum(axis=(0, 1), dtype=np.float64).tolist(),
        },
        "router_rank": {
            "count": rank_counts.tolist(),
            "energy": rank_energy.tolist(),
        },
        "expert_id": {
            "count": expert_counts.tolist(),
            "energy": expert_energy.tolist(),
        },
        "shared_raw": {
            "count": batch.tokens,
            "sum": shared64.sum(axis=0, dtype=np.float64).tolist(),
            "squared_norm": float(np.square(shared64).sum(dtype=np.float64)),
        },
        "shared_normalized": {
            "count": batch.tokens,
            "sum": normalized_shared64.sum(axis=0, dtype=np.float64).tolist(),
            "squared_norm": float(np.square(normalized_shared64).sum(dtype=np.float64)),
        },
    }


def validate_contribution_statistics(
    value: Mapping[str, Any],
    *,
    expected_tokens: int | None = None,
) -> dict[str, Any]:
    """Validate counts, shapes, finiteness, and cross-statistic conservation."""

    if not isinstance(value, Mapping):
        raise ContributionIntegrityError("contribution statistics must be an object")
    expected_keys = {
        "token_count",
        "component_count",
        "pooled_expert_output",
        "weighted_contribution",
        "router_rank",
        "expert_id",
        "shared_raw",
        "shared_normalized",
    }
    if set(value) != expected_keys:
        raise ContributionIntegrityError(
            "contribution statistic fields differ from the v2 contract"
        )
    tokens = value["token_count"]
    components = value["component_count"]
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise ContributionIntegrityError("statistics token_count must be positive")
    if expected_tokens is not None and tokens != expected_tokens:
        raise ContributionIntegrityError(
            f"statistics count {tokens} differs from expected {expected_tokens}"
        )
    if components != tokens * TOP_K_EXPERTS:
        raise ContributionIntegrityError(
            "statistics component_count must equal token_count * top_k"
        )

    pooled = value["pooled_expert_output"]
    weighted = value["weighted_contribution"]
    router_rank = value["router_rank"]
    expert_id = value["expert_id"]
    shared_raw = value["shared_raw"]
    shared_normalized = value["shared_normalized"]
    nested = (
        ("pooled_expert_output", pooled, {"count", "sum", "squared_norm"}),
        (
            "weighted_contribution",
            weighted,
            {
                "weight_squared_sum",
                "weight_squared_output_sum",
                "squared_norm",
                "raw_sum",
            },
        ),
        ("router_rank", router_rank, {"count", "energy"}),
        ("expert_id", expert_id, {"count", "energy"}),
        ("shared_raw", shared_raw, {"count", "sum", "squared_norm"}),
        (
            "shared_normalized",
            shared_normalized,
            {"count", "sum", "squared_norm"},
        ),
    )
    for name, nested_value, keys in nested:
        if not isinstance(nested_value, Mapping) or set(nested_value) != keys:
            raise ContributionIntegrityError(
                f"statistics {name} fields differ from the v2 contract"
            )

    if pooled["count"] != components:
        raise ContributionIntegrityError("pooled expert count is inconsistent")
    if shared_raw["count"] != tokens or shared_normalized["count"] != tokens:
        raise ContributionIntegrityError("shared branch count is inconsistent")
    pooled_sum = _finite_vector("pooled expert sum", pooled["sum"], HIDDEN_SIZE)
    pooled_squared = _finite_nonnegative_scalar(
        "pooled expert squared norm", pooled["squared_norm"]
    )
    weight_squared_sum = _finite_nonnegative_scalar(
        "weight squared sum", weighted["weight_squared_sum"]
    )
    weighted_output_sum = _finite_vector(
        "weight-squared output sum",
        weighted["weight_squared_output_sum"],
        HIDDEN_SIZE,
    )
    weighted_squared = _finite_nonnegative_scalar(
        "weighted contribution squared norm", weighted["squared_norm"]
    )
    raw_sum = _finite_vector(
        "weighted contribution raw sum", weighted["raw_sum"], HIDDEN_SIZE
    )

    rank_count = np.asarray(router_rank["count"])
    expert_count = np.asarray(expert_id["count"])
    if (
        rank_count.dtype.kind not in "iu"
        or rank_count.shape != (TOP_K_EXPERTS,)
        or np.any(rank_count != tokens)
    ):
        raise ContributionIntegrityError(
            "each router rank must contain exactly one component per token"
        )
    if (
        expert_count.dtype.kind not in "iu"
        or expert_count.shape != (NUM_EXPERTS,)
        or np.any(expert_count < 0)
        or int(expert_count.sum()) != components
    ):
        raise ContributionIntegrityError(
            "expert-ID counts must be nonnegative and sum to component_count"
        )
    rank_energy = _finite_nonnegative_vector(
        "router-rank energy", router_rank["energy"], TOP_K_EXPERTS
    )
    expert_energy = _finite_nonnegative_vector(
        "expert-ID energy", expert_id["energy"], NUM_EXPERTS
    )
    if not math.isclose(
        sum(rank_energy), weighted_squared, rel_tol=1e-12, abs_tol=1e-9
    ) or not math.isclose(
        sum(expert_energy), weighted_squared, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise ContributionIntegrityError(
            "rank/expert energy must conserve weighted contribution energy"
        )
    shared_raw_sum = _finite_vector("raw shared sum", shared_raw["sum"], HIDDEN_SIZE)
    shared_raw_squared = _finite_nonnegative_scalar(
        "raw shared squared norm", shared_raw["squared_norm"]
    )
    shared_normalized_sum = _finite_vector(
        "normalized shared sum", shared_normalized["sum"], HIDDEN_SIZE
    )
    shared_normalized_squared = _finite_nonnegative_scalar(
        "normalized shared squared norm", shared_normalized["squared_norm"]
    )
    return {
        "token_count": tokens,
        "component_count": components,
        "pooled_expert_output": {
            "count": components,
            "sum": pooled_sum,
            "squared_norm": pooled_squared,
        },
        "weighted_contribution": {
            "weight_squared_sum": weight_squared_sum,
            "weight_squared_output_sum": weighted_output_sum,
            "squared_norm": weighted_squared,
            "raw_sum": raw_sum,
        },
        "router_rank": {
            "count": [int(item) for item in rank_count],
            "energy": rank_energy,
        },
        "expert_id": {
            "count": [int(item) for item in expert_count],
            "energy": expert_energy,
        },
        "shared_raw": {
            "count": tokens,
            "sum": shared_raw_sum,
            "squared_norm": shared_raw_squared,
        },
        "shared_normalized": {
            "count": tokens,
            "sum": shared_normalized_sum,
            "squared_norm": shared_normalized_squared,
        },
    }


def merge_contribution_statistics(
    values: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge shard statistics in caller-supplied deterministic range order."""

    if not values:
        raise ContributionIntegrityError("cannot merge an empty statistic sequence")
    checked = [validate_contribution_statistics(value) for value in values]

    def sum_vectors(path: tuple[str, str], length: int) -> list[float]:
        result = np.zeros(length, dtype=np.float64)
        for statistic in checked:
            result += np.asarray(statistic[path[0]][path[1]], dtype=np.float64)
        return result.tolist()

    tokens = sum(item["token_count"] for item in checked)
    components = sum(item["component_count"] for item in checked)
    rank_counts = np.zeros(TOP_K_EXPERTS, dtype=np.int64)
    expert_counts = np.zeros(NUM_EXPERTS, dtype=np.int64)
    for item in checked:
        rank_counts += np.asarray(item["router_rank"]["count"], dtype=np.int64)
        expert_counts += np.asarray(item["expert_id"]["count"], dtype=np.int64)
    merged = {
        "token_count": tokens,
        "component_count": components,
        "pooled_expert_output": {
            "count": components,
            "sum": sum_vectors(("pooled_expert_output", "sum"), HIDDEN_SIZE),
            "squared_norm": sum(
                item["pooled_expert_output"]["squared_norm"] for item in checked
            ),
        },
        "weighted_contribution": {
            "weight_squared_sum": sum(
                item["weighted_contribution"]["weight_squared_sum"] for item in checked
            ),
            "weight_squared_output_sum": sum_vectors(
                ("weighted_contribution", "weight_squared_output_sum"), HIDDEN_SIZE
            ),
            "squared_norm": sum(
                item["weighted_contribution"]["squared_norm"] for item in checked
            ),
            "raw_sum": sum_vectors(("weighted_contribution", "raw_sum"), HIDDEN_SIZE),
        },
        "router_rank": {
            "count": rank_counts.tolist(),
            "energy": sum_vectors(("router_rank", "energy"), TOP_K_EXPERTS),
        },
        "expert_id": {
            "count": expert_counts.tolist(),
            "energy": sum_vectors(("expert_id", "energy"), NUM_EXPERTS),
        },
        "shared_raw": {
            "count": tokens,
            "sum": sum_vectors(("shared_raw", "sum"), HIDDEN_SIZE),
            "squared_norm": sum(item["shared_raw"]["squared_norm"] for item in checked),
        },
        "shared_normalized": {
            "count": tokens,
            "sum": sum_vectors(("shared_normalized", "sum"), HIDDEN_SIZE),
            "squared_norm": sum(
                item["shared_normalized"]["squared_norm"] for item in checked
            ),
        },
    }
    return validate_contribution_statistics(merged, expected_tokens=tokens)


def _validate_range_bounds(start: Any, end: Any) -> tuple[int, int]:
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise ContributionRangeError(
            f"invalid half-open scientific token range [{start},{end})"
        )
    if end - start > MAX_SHARD_TOKENS:
        raise ContributionRangeError(
            f"scientific token range exceeds {MAX_SHARD_TOKENS} tokens"
        )
    return start, end


def deterministic_range_id(
    *,
    split: str,
    range_start: int,
    range_end: int,
    source_v1_range_start: int,
    source_v1_range_end: int,
    source_v1_manifest_sha256: str,
    token_identity_sha256: str,
) -> str:
    """Derive an immutable ID from logical and source-v1 token bounds."""

    if not isinstance(split, str) or _SAFE_NAME_RE.fullmatch(split) is None:
        raise ContributionRangeError("split is not a safe nonempty identity")
    start, end = _validate_range_bounds(range_start, range_end)
    source_start, source_end = _validate_range_bounds(
        source_v1_range_start,
        source_v1_range_end,
    )
    if source_end - source_start != end - start:
        raise ContributionRangeError(
            "logical and source-v1 ranges must contain the same token count"
        )
    source_digest = _require_sha256(
        "source_v1_manifest_sha256", source_v1_manifest_sha256
    )
    token_digest = _require_sha256("token_identity_sha256", token_identity_sha256)
    identity = canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "source_v1_manifest_sha256": source_digest,
            "token_identity_sha256": token_digest,
            "split": split,
            "range_start": start,
            "range_end": end,
            "source_v1_range_start": source_start,
            "source_v1_range_end": source_end,
        }
    )
    return f"{split}-{start:012d}-{end:012d}-{identity[:16]}"


def validate_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact scientific metadata shared by every v2 shard."""

    if not isinstance(value, Mapping):
        raise ContributionSchemaError("contribution metadata must be an object")
    expected_keys = {
        "checkpoint_sha256",
        "maxtext_fork_commit",
        "dataset_fingerprint",
        "split",
        "config_sha256",
        "environment_sha256",
        "source_v1_manifest_sha256",
        "recomposition_sidecar_sha256",
        "protocol_id",
        "model_id",
        "maxtext_base_commit",
        "target_layer_one_indexed",
        "sequence_length",
        "capture_site",
        "shared_capture_site",
        "hidden_size",
        "num_experts",
        "top_k_experts",
        "max_shard_tokens",
        "tensor_schema",
    }
    if set(value) != expected_keys:
        raise ContributionSchemaError(
            "contribution metadata fields differ from the v2 contract"
        )
    for name in (
        "checkpoint_sha256",
        "dataset_fingerprint",
        "config_sha256",
        "environment_sha256",
        "source_v1_manifest_sha256",
        "recomposition_sidecar_sha256",
    ):
        _require_sha256(name, value[name])
    if (
        not isinstance(value["maxtext_fork_commit"], str)
        or _GIT_COMMIT_RE.fullmatch(value["maxtext_fork_commit"]) is None
    ):
        raise ContributionSchemaError(
            "maxtext_fork_commit must be a lowercase 40-character Git commit"
        )
    if (
        not isinstance(value["split"], str)
        or _SAFE_NAME_RE.fullmatch(value["split"]) is None
    ):
        raise ContributionSchemaError("split must be a safe nonempty identity")
    fixed = {
        "protocol_id": PROTOCOL_ID,
        "model_id": MODEL_ID,
        "maxtext_base_commit": MAXTEXT_BASE_COMMIT,
        "target_layer_one_indexed": TARGET_LAYER_ONE_INDEXED,
        "sequence_length": SEQUENCE_LENGTH,
        "capture_site": CAPTURE_SITE,
        "shared_capture_site": SHARED_CAPTURE_SITE,
        "hidden_size": HIDDEN_SIZE,
        "num_experts": NUM_EXPERTS,
        "top_k_experts": TOP_K_EXPERTS,
        "max_shard_tokens": MAX_SHARD_TOKENS,
        "tensor_schema": TENSOR_SCHEMA,
    }
    for name, expected in fixed.items():
        if value[name] != expected:
            raise ContributionSchemaError(
                f"metadata {name} must be {expected!r}, got {value[name]!r}"
            )
    return json.loads(canonical_json_bytes(dict(value)))


def build_recomposition_sidecar(
    *,
    checkpoint_sha256: str,
    maxtext_fork_commit: str,
    source_fingerprints: Mapping[str, str],
    routed_rmsnorm_scale: Any,
    shared_rmsnorm_scale: Any,
    outer_ffn_rmsnorm_scale: Any,
    layer_scalar: Any,
    accumulation_order: str,
) -> dict[str, Any]:
    """Build and self-hash the checkpoint-free Gemma recomposition sidecar."""

    tensors: dict[str, Any] = {}
    supplied = {
        "routed_rmsnorm_scale": routed_rmsnorm_scale,
        "shared_rmsnorm_scale": shared_rmsnorm_scale,
        "outer_ffn_rmsnorm_scale": outer_ffn_rmsnorm_scale,
        "layer_scalar": layer_scalar,
    }
    for name, specification in SIDECAR_TENSOR_SCHEMA.items():
        array = np.asarray(supplied[name])
        if array.dtype != np.uint16:
            raise ContributionSchemaError(
                f"sidecar tensor {name} must contain exact uint16 BF16 bits"
            )
        expected_shape = tuple(specification["shape"])
        if array.shape != expected_shape:
            raise ContributionSchemaError(
                f"sidecar tensor {name} must have shape {expected_shape}, "
                f"got {array.shape}"
            )
        if not np.isfinite(bf16_bits_to_float32(array)).all():
            raise ContributionSchemaError(
                f"sidecar tensor {name} contains non-finite values"
            )
        tensors[name] = {
            **specification,
            "data": [int(item) for item in array.reshape(-1)],
        }
    if (
        not isinstance(maxtext_fork_commit, str)
        or _GIT_COMMIT_RE.fullmatch(maxtext_fork_commit) is None
    ):
        raise ContributionSchemaError(
            "maxtext_fork_commit must be a lowercase 40-character Git commit"
        )
    if not isinstance(source_fingerprints, Mapping) or not source_fingerprints:
        raise ContributionSchemaError("source_fingerprints must be nonempty")
    normalized_fingerprints: dict[str, str] = {}
    for name, digest in source_fingerprints.items():
        if not isinstance(name, str) or _SAFE_NAME_RE.fullmatch(name) is None:
            raise ContributionSchemaError("source fingerprint name is unsafe")
        normalized_fingerprints[name] = _require_sha256(
            f"source_fingerprints.{name}", digest
        )
    if not isinstance(accumulation_order, str) or not accumulation_order.strip():
        raise ContributionSchemaError("accumulation_order must be explicit")
    unsigned = {
        "schema_version": RECOMPOSITION_SIDECAR_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "checkpoint_sha256": _require_sha256("checkpoint_sha256", checkpoint_sha256),
        "target_layer_one_indexed": TARGET_LAYER_ONE_INDEXED,
        "hidden_size": HIDDEN_SIZE,
        "maxtext_base_commit": MAXTEXT_BASE_COMMIT,
        "maxtext_fork_commit": maxtext_fork_commit,
        "rms_norm": {
            "epsilon": RMS_NORM_EPSILON,
            "reduction_axis": -1,
            "scale_offset": 0.0,
            "formula": "x_f32*rsqrt(mean(square(x_f32))+epsilon)*scale",
        },
        "dtype_policy": DTYPE_POLICY,
        "accumulation_order": accumulation_order,
        "source_fingerprints": dict(sorted(normalized_fingerprints.items())),
        "tensors": tensors,
    }
    return {**unsigned, "sidecar_sha256": canonical_sha256(unsigned)}


def validate_recomposition_sidecar(value: Mapping[str, Any]) -> dict[str, Any]:
    """Verify exact sidecar fields, tensor bits, and canonical self-hash."""

    if not isinstance(value, Mapping):
        raise ContributionSchemaError("recomposition sidecar must be an object")
    if value.get("schema_version") != RECOMPOSITION_SIDECAR_SCHEMA_VERSION:
        raise ContributionLegacySchemaError(
            "recomposition sidecar has a missing or unsupported schema"
        )
    expected_keys = {
        "schema_version",
        "model_id",
        "checkpoint_sha256",
        "target_layer_one_indexed",
        "hidden_size",
        "maxtext_base_commit",
        "maxtext_fork_commit",
        "rms_norm",
        "dtype_policy",
        "accumulation_order",
        "source_fingerprints",
        "tensors",
        "sidecar_sha256",
    }
    if set(value) != expected_keys:
        raise ContributionSchemaError(
            "recomposition sidecar fields differ from the v1 contract"
        )
    unsigned = {name: value[name] for name in expected_keys if name != "sidecar_sha256"}
    claimed = _require_sha256("sidecar_sha256", value["sidecar_sha256"])
    if canonical_sha256(unsigned) != claimed:
        raise ContributionIntegrityError(
            "recomposition sidecar canonical SHA-256 mismatch"
        )
    fixed = {
        "model_id": MODEL_ID,
        "target_layer_one_indexed": TARGET_LAYER_ONE_INDEXED,
        "hidden_size": HIDDEN_SIZE,
        "maxtext_base_commit": MAXTEXT_BASE_COMMIT,
        "rms_norm": {
            "epsilon": RMS_NORM_EPSILON,
            "reduction_axis": -1,
            "scale_offset": 0.0,
            "formula": "x_f32*rsqrt(mean(square(x_f32))+epsilon)*scale",
        },
        "dtype_policy": DTYPE_POLICY,
    }
    for name, expected in fixed.items():
        if value[name] != expected:
            raise ContributionSchemaError(
                f"sidecar {name} differs from the pinned recomposition contract"
            )
    _require_sha256("checkpoint_sha256", value["checkpoint_sha256"])
    if (
        not isinstance(value["maxtext_fork_commit"], str)
        or _GIT_COMMIT_RE.fullmatch(value["maxtext_fork_commit"]) is None
    ):
        raise ContributionSchemaError("sidecar MaxText fork identity is invalid")
    if (
        not isinstance(value["accumulation_order"], str)
        or not value["accumulation_order"].strip()
    ):
        raise ContributionSchemaError("sidecar accumulation order is missing")
    fingerprints = value["source_fingerprints"]
    if not isinstance(fingerprints, Mapping) or not fingerprints:
        raise ContributionSchemaError("sidecar source fingerprints are missing")
    for name, digest in fingerprints.items():
        if not isinstance(name, str) or _SAFE_NAME_RE.fullmatch(name) is None:
            raise ContributionSchemaError("sidecar source fingerprint name is unsafe")
        _require_sha256(f"source_fingerprints.{name}", digest)

    tensors = value["tensors"]
    if not isinstance(tensors, Mapping) or set(tensors) != set(SIDECAR_TENSOR_SCHEMA):
        raise ContributionSchemaError("sidecar tensor inventory is incomplete")
    for name, specification in SIDECAR_TENSOR_SCHEMA.items():
        tensor = tensors[name]
        if not isinstance(tensor, Mapping) or set(tensor) != {
            *specification.keys(),
            "data",
        }:
            raise ContributionSchemaError(
                f"sidecar tensor {name} fields differ from the contract"
            )
        for field, expected in specification.items():
            if tensor[field] != expected:
                raise ContributionSchemaError(
                    f"sidecar tensor {name}.{field} differs from the contract"
                )
        data = tensor["data"]
        expected_elements = math.prod(specification["shape"])
        if (
            not isinstance(data, list)
            or len(data) != expected_elements
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or not 0 <= item <= np.iinfo(np.uint16).max
                for item in data
            )
        ):
            raise ContributionSchemaError(
                f"sidecar tensor {name} does not contain exact uint16 BF16 bits"
            )
        decoded = bf16_bits_to_float32(np.asarray(data, dtype=np.uint16))
        if not np.isfinite(decoded).all():
            raise ContributionSchemaError(
                f"sidecar tensor {name} contains non-finite values"
            )
    return json.loads(canonical_json_bytes(dict(value)))


def _validate_source_v1_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "manifest_sha256",
        "split",
        "total_tokens",
    }:
        raise ContributionSchemaError(
            "source_v1_manifest identity fields differ from the contract"
        )
    if value["schema_version"] != LEGACY_AGGREGATE_SCHEMA_VERSION:
        raise ContributionSchemaError(
            "source_v1_manifest must identify the sealed aggregate v1 schema"
        )
    _require_sha256("source_v1_manifest.manifest_sha256", value["manifest_sha256"])
    if (
        not isinstance(value["split"], str)
        or _SAFE_NAME_RE.fullmatch(value["split"]) is None
    ):
        raise ContributionSchemaError("source v1 split is invalid")
    _require_positive_int("source_v1_manifest.total_tokens", value["total_tokens"])
    return dict(value)


def _validate_shard_entry(
    value: Any,
    *,
    split: str,
    source_v1_manifest_sha256: str,
    recomposition_sidecar_sha256: str,
    metadata_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContributionIntegrityError("contribution shard entry must be an object")
    expected_keys = {
        "range_id",
        "range_start",
        "range_end",
        "source_v1_range_start",
        "source_v1_range_end",
        "tokens",
        "host_id",
        "path",
        "sha256",
        "token_identity_sha256",
        "metadata_sha256",
        "source_v1_manifest_sha256",
        "recomposition_sidecar_sha256",
        "statistics",
    }
    if set(value) != expected_keys:
        raise ContributionIntegrityError(
            "contribution shard entry fields differ from the v2 contract"
        )
    start, end = _validate_range_bounds(value["range_start"], value["range_end"])
    if value["tokens"] != end - start:
        raise ContributionIntegrityError(
            "shard token count must equal its scientific token range length"
        )
    source_start, source_end = _validate_range_bounds(
        value["source_v1_range_start"],
        value["source_v1_range_end"],
    )
    if source_end - source_start != value["tokens"]:
        raise ContributionIntegrityError(
            "shard source-v1 range length differs from its token count"
        )
    token_digest = _require_sha256(
        "shard.token_identity_sha256", value["token_identity_sha256"]
    )
    expected_range_id = deterministic_range_id(
        split=split,
        range_start=start,
        range_end=end,
        source_v1_range_start=source_start,
        source_v1_range_end=source_end,
        source_v1_manifest_sha256=source_v1_manifest_sha256,
        token_identity_sha256=token_digest,
    )
    if value["range_id"] != expected_range_id:
        raise ContributionRangeError(
            "shard range_id is not the deterministic v2 range identity"
        )
    if (
        not isinstance(value["host_id"], str)
        or _SAFE_NAME_RE.fullmatch(value["host_id"]) is None
    ):
        raise ContributionIntegrityError("shard host_id is invalid")
    digest = _require_sha256("shard.sha256", value["sha256"])
    _require_sha256("shard.metadata_sha256", value["metadata_sha256"])
    _require_sha256(
        "shard.source_v1_manifest_sha256",
        value["source_v1_manifest_sha256"],
    )
    _require_sha256(
        "shard.recomposition_sidecar_sha256",
        value["recomposition_sidecar_sha256"],
    )
    if value["metadata_sha256"] != metadata_sha256:
        raise ContributionIntegrityError("shard metadata identity changed")
    if value["source_v1_manifest_sha256"] != source_v1_manifest_sha256:
        raise ContributionIntegrityError("shard source v1 identity changed")
    if value["recomposition_sidecar_sha256"] != recomposition_sidecar_sha256:
        raise ContributionIntegrityError("shard recomposition sidecar identity changed")
    path = value["path"]
    expected_path = f"shards/host-{value['host_id']}/{digest}.safetensors"
    if path != expected_path:
        raise ContributionIntegrityError(
            "shard path must be host-specific and content-addressed"
        )
    statistics = validate_contribution_statistics(
        value["statistics"], expected_tokens=value["tokens"]
    )
    return {**dict(value), "statistics": statistics}


def validate_contribution_shard_payload(
    value: ContributionBatch | Mapping[str, Any],
    entry: Mapping[str, Any],
) -> ContributionBatch:
    """Prove one v2 payload has the token keys named by its sealed entry.

    Publication must call this on the contribution payload and call
    :func:`token_identity_sha256` on the corresponding v1 source slice.  Both
    digests must equal ``entry["token_identity_sha256"]``.
    """

    batch = validate_contribution_batch(value)
    if not isinstance(entry, Mapping):
        raise ContributionIntegrityError("shard entry must be an object")
    expected_tokens = entry.get("tokens")
    if (
        isinstance(expected_tokens, bool)
        or not isinstance(expected_tokens, int)
        or expected_tokens != batch.tokens
    ):
        raise ContributionIntegrityError(
            "contribution payload token count differs from its shard entry"
        )
    expected_digest = _require_sha256(
        "shard.token_identity_sha256", entry.get("token_identity_sha256")
    )
    actual_digest = token_identity_sha256(batch.sequence_id, batch.position)
    if actual_digest != expected_digest:
        raise ContributionIntegrityError(
            "contribution payload token-key sequence differs from its bound v1 range"
        )
    return batch


def validate_source_v1_token_keys(
    sequence_id: Any,
    position: Any,
    entry: Mapping[str, Any],
) -> str:
    """Prove the corresponding v1 slice has the keys bound by a v2 entry."""

    if not isinstance(entry, Mapping):
        raise ContributionIntegrityError("shard entry must be an object")
    sequence = np.asarray(sequence_id)
    positions = np.asarray(position)
    expected_tokens = entry.get("tokens")
    if (
        isinstance(expected_tokens, bool)
        or not isinstance(expected_tokens, int)
        or sequence.ndim != 1
        or sequence.size != expected_tokens
        or positions.shape != sequence.shape
    ):
        raise ContributionIntegrityError(
            "source v1 token keys do not match the contribution range length"
        )
    expected_digest = _require_sha256(
        "shard.token_identity_sha256", entry.get("token_identity_sha256")
    )
    actual_digest = token_identity_sha256(sequence, positions)
    if actual_digest != expected_digest:
        raise ContributionIntegrityError(
            "source v1 token-key sequence differs from the contribution binding"
        )
    return actual_digest


def seal_contribution_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Attach the canonical manifest hash to an otherwise complete manifest."""

    if not isinstance(value, Mapping):
        raise ContributionSchemaError("contribution manifest must be an object")
    if "manifest_sha256" in value:
        raise ContributionSchemaError("manifest is already self-hashed")
    unsigned = dict(value)
    return {**unsigned, "manifest_sha256": canonical_sha256(unsigned)}


def validate_contribution_manifest(
    value: Mapping[str, Any],
    *,
    recomposition_sidecar: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a sealed v2 manifest and all deterministic range identities."""

    if not isinstance(value, Mapping):
        raise ContributionSchemaError("contribution manifest must be an object")
    schema = value.get("schema_version")
    if schema != SCHEMA_VERSION:
        if schema in {None, LEGACY_AGGREGATE_SCHEMA_VERSION}:
            raise ContributionLegacySchemaError(
                "aggregate or missing schemas cannot be read as contribution v2"
            )
        raise ContributionLegacySchemaError(
            f"unsupported contribution artifact schema {schema!r}"
        )
    expected_keys = {
        "schema_version",
        "protocol_id",
        "sealed",
        "metadata",
        "metadata_sha256",
        "source_v1_manifest",
        "recomposition_sidecar_sha256",
        "total_tokens",
        "shard_count",
        "statistics",
        "shards",
        "manifest_sha256",
    }
    if set(value) != expected_keys:
        raise ContributionSchemaError(
            "contribution manifest fields differ from the v2 contract"
        )
    unsigned = {
        name: value[name] for name in expected_keys if name != "manifest_sha256"
    }
    claimed = _require_sha256("manifest_sha256", value["manifest_sha256"])
    if canonical_sha256(unsigned) != claimed:
        raise ContributionIntegrityError(
            "contribution manifest canonical SHA-256 mismatch"
        )
    if value["protocol_id"] != PROTOCOL_ID:
        raise ContributionSchemaError("contribution manifest protocol changed")
    if value["sealed"] is not True:
        raise ContributionIntegrityError(
            "only a sealed contribution manifest is consumable"
        )
    metadata = validate_metadata(value["metadata"])
    metadata_sha256 = _require_sha256("metadata_sha256", value["metadata_sha256"])
    if canonical_sha256(metadata) != metadata_sha256:
        raise ContributionIntegrityError("contribution metadata SHA-256 mismatch")
    source = _validate_source_v1_identity(value["source_v1_manifest"])
    source_digest = source["manifest_sha256"]
    sidecar_digest = _require_sha256(
        "recomposition_sidecar_sha256",
        value["recomposition_sidecar_sha256"],
    )
    if metadata["source_v1_manifest_sha256"] != source_digest:
        raise ContributionIntegrityError(
            "metadata and manifest bind different aggregate v1 captures"
        )
    if metadata["recomposition_sidecar_sha256"] != sidecar_digest:
        raise ContributionIntegrityError(
            "metadata and manifest bind different recomposition sidecars"
        )
    if metadata["split"] != source["split"]:
        raise ContributionIntegrityError(
            "contribution and aggregate manifests use different splits"
        )
    if recomposition_sidecar is not None:
        checked_sidecar = validate_recomposition_sidecar(recomposition_sidecar)
        if checked_sidecar["sidecar_sha256"] != sidecar_digest:
            raise ContributionIntegrityError(
                "provided recomposition sidecar differs from manifest identity"
            )
        if checked_sidecar["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
            raise ContributionIntegrityError(
                "sidecar and contribution manifest use different checkpoints"
            )
        if checked_sidecar["maxtext_fork_commit"] != metadata["maxtext_fork_commit"]:
            raise ContributionIntegrityError(
                "sidecar and contribution manifest use different MaxText forks"
            )

    total_tokens = _require_positive_int("total_tokens", value["total_tokens"])
    shard_count = _require_positive_int("shard_count", value["shard_count"])
    raw_shards = value["shards"]
    if not isinstance(raw_shards, list) or len(raw_shards) != shard_count:
        raise ContributionIntegrityError(
            "manifest shard_count differs from its shard inventory"
        )
    shards = [
        _validate_shard_entry(
            shard,
            split=metadata["split"],
            source_v1_manifest_sha256=source_digest,
            recomposition_sidecar_sha256=sidecar_digest,
            metadata_sha256=metadata_sha256,
        )
        for shard in raw_shards
    ]
    if shards != sorted(
        shards, key=lambda shard: (shard["range_start"], shard["range_id"])
    ):
        raise ContributionRangeError(
            "manifest shards must be in deterministic scientific token order"
        )
    expected_start = 0
    seen_ids: set[str] = set()
    for shard in shards:
        if shard["range_id"] in seen_ids:
            raise ContributionRangeError("manifest contains a duplicate range_id")
        seen_ids.add(shard["range_id"])
        if shard["range_start"] != expected_start:
            raise ContributionRangeError(
                "contribution token ranges contain a gap or overlap"
            )
        expected_start = shard["range_end"]
    if expected_start != total_tokens:
        raise ContributionRangeError(
            "contribution ranges do not cover exactly total_tokens"
        )
    if total_tokens > source["total_tokens"]:
        raise ContributionIntegrityError(
            "contribution capture exceeds its bound aggregate v1 capture"
        )
    source_ranges = sorted(
        (
            int(shard["source_v1_range_start"]),
            int(shard["source_v1_range_end"]),
        )
        for shard in shards
    )
    prior_source_end = -1
    for source_start, source_end in source_ranges:
        if source_end > source["total_tokens"]:
            raise ContributionRangeError(
                "contribution source range exceeds the bound aggregate v1 capture"
            )
        if source_start < prior_source_end:
            raise ContributionRangeError(
                "contribution source-v1 ranges overlap or duplicate tokens"
            )
        prior_source_end = source_end

    merged = merge_contribution_statistics([shard["statistics"] for shard in shards])
    recorded_statistics = validate_contribution_statistics(
        value["statistics"], expected_tokens=total_tokens
    )
    if canonical_json_bytes(recorded_statistics) != canonical_json_bytes(merged):
        raise ContributionIntegrityError(
            "manifest statistics do not equal the deterministic shard merge"
        )
    return {
        **dict(value),
        "metadata": metadata,
        "source_v1_manifest": source,
        "statistics": recorded_statistics,
        "shards": shards,
    }
