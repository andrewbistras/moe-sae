"""Streaming scientific evaluation boundary for Experiment 5.

TPU workers feed this module compact host row chunks: token keys, additive
full-FFN and oracle errors, exact logical latent IDs, and (for H/R) routed
bank IDs plus branch sufficient statistics. The accumulator never needs
activation tensors after a chunk is scored.

The finalizer joins:

* four condition accumulators on one identical packed-sequence basis;
* self-hashed milestone training receipts;
* checkpoint cumulative firing masks in logical feature order; and
* evaluation-time firing and route exposure.

It emits the exact scientific-result and paired-score artifacts accepted by
``experiment5_report``. Training firing coverage and held-out nonfiring remain
separate evidence throughout.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from moe_sae.contribution_artifacts import (
    SEQUENCE_LENGTH,
    bf16_bits_to_float32,
    canonical_sha256,
    float32_to_bf16_bits,
)
from moe_sae.experiment5_hybrid import (
    Experiment5Layout,
    Experiment5Recomposition,
    numpy_recompose_experiment5_full_ffn,
)
from moe_sae.experiment5_report import (
    CONDITIONS,
    CONTRASTS,
    EVAL_SOURCE_SHA256,
    EVAL_TARGET_SHA256,
    EVAL_TOKENS,
    MILESTONES,
    PAIRED_SCHEMA_VERSION,
    PANELS,
    PROTOCOL_ID,
    RESULT_SCHEMA_VERSION,
    SEEDS,
    SIDECAR_CONTENT_SHA256,
    SIDECAR_FILE_SHA256,
    SOURCE_CAPTURE_CONTRACT_SHA256,
    TRAIN_SOURCE_SHA256,
    TRAIN_TARGET_SHA256,
    bootstrap_experiment5_paired_effect,
    validate_experiment5_paired_scores,
    validate_experiment5_result,
)

TRAINING_RECEIPT_SCHEMA_VERSION = (
    "gemma4-routed-shared-hybrid-training-receipt-v1"
)
FIRING_MASK_SCHEMA_VERSION = "gemma4-routed-shared-hybrid-firing-mask-v1"
EVALUATION_PARTIAL_SCHEMA_VERSION = (
    "gemma4-routed-shared-hybrid-evaluation-partial-v1"
)
_ENDPOINT_NAMES = (
    "full_ffn",
    "routed",
    "shared",
    "oracle_shared_full_ffn",
    "oracle_routed_full_ffn",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")


class Experiment5EvaluationError(ValueError):
    """Evaluation rows or evidence violate the frozen Experiment-5 contract."""


def _finite(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise Experiment5EvaluationError(f"{name} must be numeric")
    selected = float(value)
    if not math.isfinite(selected):
        raise Experiment5EvaluationError(f"{name} must be finite")
    if positive and selected <= 0:
        raise Experiment5EvaluationError(f"{name} must be positive")
    if minimum is not None and selected < minimum:
        raise Experiment5EvaluationError(f"{name} must be at least {minimum}")
    return selected


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        raise Experiment5EvaluationError(
            f"{name} must be an integer at least {minimum}"
        )
    return int(value)


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Experiment5EvaluationError(f"{name} must be a lowercase SHA-256")
    return value


def _float_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    nonnegative: bool = False,
) -> np.ndarray:
    selected = np.asarray(value)
    if (
        selected.shape != shape
        or not np.issubdtype(selected.dtype, np.floating)
        or not np.isfinite(selected).all()
        or (nonnegative and np.any(selected < 0))
    ):
        qualifier = "finite nonnegative" if nonnegative else "finite"
        raise Experiment5EvaluationError(
            f"{name} must be {qualifier} floating-point {shape}"
        )
    result = np.ascontiguousarray(selected, dtype=np.float64)
    result.setflags(write=False)
    return result


def _bf16(value: Any) -> np.ndarray:
    return bf16_bits_to_float32(
        float32_to_bf16_bits(np.asarray(value, dtype=np.float32))
    )


def _rmsnorm_bf16(
    value: Any,
    scale: Any,
    *,
    epsilon: float,
) -> np.ndarray:
    selected = _bf16(value)
    scale_bf16 = _bf16(scale)
    mean_square = np.mean(
        np.square(selected, dtype=np.float32),
        axis=-1,
        keepdims=True,
        dtype=np.float32,
    )
    inverse = np.reciprocal(
        np.sqrt(mean_square + np.float32(epsilon), dtype=np.float32),
        dtype=np.float32,
    )
    unit = _bf16(np.multiply(selected, inverse, dtype=np.float32))
    return _bf16(np.multiply(unit, scale_bf16, dtype=np.float32))


def _routed_branch(
    expert_output: Any,
    router_weights: Any,
    recomposition: Experiment5Recomposition,
) -> np.ndarray:
    expert = _bf16(expert_output)
    weights = _bf16(router_weights)
    raw = np.einsum(
        "nkh,nk->nh",
        expert,
        weights,
        dtype=np.float32,
        optimize=False,
    )
    return _rmsnorm_bf16(
        raw,
        recomposition.routed_scale,
        epsilon=float(recomposition.epsilon),
    )


def _token_sse(target: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    return np.sum(
        np.square(
            target.astype(np.float32) - estimate.astype(np.float32),
            dtype=np.float32,
        ),
        axis=1,
        dtype=np.float64,
    )


def _token_sst(target: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return np.sum(
        np.square(
            target.astype(np.float32) - mean.astype(np.float32)[None, :],
            dtype=np.float32,
        ),
        axis=1,
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class Experiment5FeatureMaskShard:
    """One disjoint logical-feature slice from a distributed checkpoint."""

    logical_feature_ids: np.ndarray
    fired: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.logical_feature_ids)
        fired = np.asarray(self.fired)
        if (
            ids.ndim != 1
            or ids.dtype not in (np.int32, np.int64)
            or fired.dtype != np.bool_
            or fired.shape != ids.shape
            or not ids.size
            or np.any(ids < 0)
            or len(np.unique(ids)) != ids.size
        ):
            raise Experiment5EvaluationError(
                "feature mask shard requires unique nonnegative IDs and bool values"
            )
        checked_ids = np.ascontiguousarray(ids, dtype=np.int32)
        checked_fired = np.ascontiguousarray(fired, dtype=np.bool_)
        checked_ids.setflags(write=False)
        checked_fired.setflags(write=False)
        object.__setattr__(self, "logical_feature_ids", checked_ids)
        object.__setattr__(self, "fired", checked_fired)


def merge_experiment5_firing_mask_shards(
    shards: Sequence[Experiment5FeatureMaskShard],
    *,
    total_features: int,
) -> tuple[np.ndarray, str]:
    """Merge an exact disjoint cover and hash the canonical logical mask."""

    total = _integer(total_features, name="total_features", minimum=1)
    if not shards:
        raise Experiment5EvaluationError("checkpoint firing shards are empty")
    ids = np.concatenate([row.logical_feature_ids for row in shards])
    fired = np.concatenate([row.fired for row in shards])
    if (
        ids.size != total
        or len(np.unique(ids)) != total
        or not np.array_equal(np.sort(ids), np.arange(total, dtype=np.int32))
    ):
        raise Experiment5EvaluationError(
            "checkpoint firing shards are not an exact logical-feature cover"
        )
    logical = np.zeros((total,), dtype=np.bool_)
    logical[ids] = fired
    packed = np.packbits(logical, bitorder="little")
    digest = hashlib.sha256()
    digest.update(FIRING_MASK_SCHEMA_VERSION.encode("ascii") + b"\0")
    digest.update(total.to_bytes(8, "little", signed=False))
    digest.update(packed.tobytes(order="C"))
    logical.setflags(write=False)
    return logical, digest.hexdigest()


def logicalize_experiment5_checkpoint_firing_mask(
    physical_fired: Any,
    *,
    layout: Experiment5Layout,
    feature_axis_size: int,
    expected_physical_layout_sha256: str,
) -> Experiment5FeatureMaskShard:
    """Map a gathered checkpoint mask from physical to logical feature order.

    H/R checkpoints interleave each feature shard's routed-bank slice and
    shared slice.  Reporting a physical mask as if it were logical silently
    corrupts per-bank firing coverage, so this conversion also verifies the
    mesh-bound physical-layout identity recorded by training.
    """

    # Delayed to keep the host result layer independent of JAX at import time
    # and to avoid a module cycle when the distributed evaluator imports us.
    from moe_sae.experiment5_distributed import (
        experiment5_physical_layout,
        experiment5_physical_to_logical,
    )

    selected_axis = _integer(
        feature_axis_size,
        name="feature_axis_size",
        minimum=1,
    )
    expected = _sha256(
        expected_physical_layout_sha256,
        name="expected_physical_layout_sha256",
    )
    identity = experiment5_physical_layout(
        layout,
        selected_axis,
    )
    if identity.identity_sha256 != expected:
        raise Experiment5EvaluationError(
            "checkpoint physical feature layout identity changed"
        )
    physical = np.asarray(physical_fired)
    if (
        physical.shape != (layout.total_features,)
        or physical.dtype != np.bool_
    ):
        raise Experiment5EvaluationError(
            "checkpoint physical firing mask has incompatible shape or dtype"
        )
    physical_to_logical = experiment5_physical_to_logical(
        layout,
        selected_axis,
    )
    logical = np.zeros_like(physical)
    logical[physical_to_logical] = physical
    return Experiment5FeatureMaskShard(
        logical_feature_ids=np.arange(layout.total_features, dtype=np.int32),
        fired=logical,
    )


@dataclass(frozen=True, slots=True)
class Experiment5TrainingReceipt:
    """Normalized milestone timing and route-exposure receipt."""

    panel: str
    condition: str
    seed: int
    checkpoint_target_tokens: int
    actual_training_tokens: int
    training_result_sha256: str
    checkpoint_pointer_sha256: str
    accelerator_type: str
    accelerator_generation: str
    chip_count: int
    marginal_wall_seconds: float
    marginal_chip_seconds: float
    tokens_per_second: float
    useful_mfu: float | None
    routed_bank_component_counts: tuple[int, ...] | None
    receipt_sha256: str
    schema_version: str = TRAINING_RECEIPT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        if self.routed_bank_component_counts is not None:
            value["routed_bank_component_counts"] = list(
                self.routed_bank_component_counts
            )
        return value


def build_experiment5_training_receipt(
    *,
    panel: str,
    condition: str,
    seed: int,
    checkpoint_target_tokens: int,
    actual_training_tokens: int,
    training_result_sha256: str,
    checkpoint_pointer_sha256: str,
    accelerator_type: str,
    accelerator_generation: str,
    chip_count: int,
    marginal_wall_seconds: float,
    marginal_chip_seconds: float,
    tokens_per_second: float,
    useful_mfu: float | None,
    routed_bank_component_counts: Sequence[int] | None,
) -> Experiment5TrainingReceipt:
    """Build one self-hashed normalized receipt from training/workload evidence."""

    if panel not in PANELS or condition not in CONDITIONS or seed not in SEEDS:
        raise Experiment5EvaluationError("training receipt identity is invalid")
    if checkpoint_target_tokens not in MILESTONES:
        raise Experiment5EvaluationError("training receipt milestone is invalid")
    actual = _integer(
        actual_training_tokens,
        name="actual_training_tokens",
        minimum=checkpoint_target_tokens,
    )
    chips = _integer(chip_count, name="chip_count", minimum=1)
    wall = _finite(
        marginal_wall_seconds,
        name="marginal_wall_seconds",
        positive=True,
    )
    chip_seconds = _finite(
        marginal_chip_seconds,
        name="marginal_chip_seconds",
        positive=True,
    )
    throughput = _finite(
        tokens_per_second,
        name="tokens_per_second",
        positive=True,
    )
    if not math.isclose(chip_seconds, wall * chips, rel_tol=1e-10, abs_tol=1e-8):
        raise Experiment5EvaluationError(
            "marginal chip-seconds differ from wall time x chip count"
        )
    if not math.isclose(throughput, actual / wall, rel_tol=1e-10, abs_tol=1e-8):
        raise Experiment5EvaluationError(
            "training throughput differs from token count / wall time"
        )
    if accelerator_generation not in {"v4", "v5e", "v6e"}:
        raise Experiment5EvaluationError("accelerator generation is unsupported")
    if not isinstance(accelerator_type, str) or not accelerator_type:
        raise Experiment5EvaluationError("accelerator type must be nonempty")
    selected_mfu = (
        None
        if useful_mfu is None
        else _finite(useful_mfu, name="useful_mfu", minimum=0)
    )
    if selected_mfu is not None and selected_mfu > 1:
        raise Experiment5EvaluationError("useful_mfu exceeds one")
    if condition in {"H", "R"}:
        if routed_bank_component_counts is None:
            raise Experiment5EvaluationError(
                "hybrid training receipt lacks routed bank counts"
            )
        counts = tuple(
            _integer(value, name="routed_bank_component_count")
            for value in routed_bank_component_counts
        )
        if len(counts) != 128 or sum(counts) != actual * 8:
            raise Experiment5EvaluationError(
                "routed bank component counts differ from training tokens"
            )
    else:
        if routed_bank_component_counts is not None:
            raise Experiment5EvaluationError(
                "global training receipt cannot contain routed bank counts"
            )
        counts = None
    body = {
        "schema_version": TRAINING_RECEIPT_SCHEMA_VERSION,
        "panel": panel,
        "condition": condition,
        "seed": seed,
        "checkpoint_target_tokens": checkpoint_target_tokens,
        "actual_training_tokens": actual,
        "training_result_sha256": _sha256(
            training_result_sha256,
            name="training_result_sha256",
        ),
        "checkpoint_pointer_sha256": _sha256(
            checkpoint_pointer_sha256,
            name="checkpoint_pointer_sha256",
        ),
        "accelerator_type": accelerator_type,
        "accelerator_generation": accelerator_generation,
        "chip_count": chips,
        "marginal_wall_seconds": wall,
        "marginal_chip_seconds": chip_seconds,
        "tokens_per_second": throughput,
        "useful_mfu": selected_mfu,
        "routed_bank_component_counts": (
            list(counts) if counts is not None else None
        ),
    }
    return Experiment5TrainingReceipt(
        panel=panel,
        condition=condition,
        seed=seed,
        checkpoint_target_tokens=checkpoint_target_tokens,
        actual_training_tokens=actual,
        training_result_sha256=body["training_result_sha256"],
        checkpoint_pointer_sha256=body["checkpoint_pointer_sha256"],
        accelerator_type=accelerator_type,
        accelerator_generation=accelerator_generation,
        chip_count=chips,
        marginal_wall_seconds=wall,
        marginal_chip_seconds=chip_seconds,
        tokens_per_second=throughput,
        useful_mfu=selected_mfu,
        routed_bank_component_counts=counts,
        receipt_sha256=canonical_sha256(body),
    )


def validate_experiment5_training_receipt(
    value: Experiment5TrainingReceipt | Mapping[str, Any],
) -> Experiment5TrainingReceipt:
    """Validate or normalize one training receipt."""

    if isinstance(value, Experiment5TrainingReceipt):
        raw = value.to_dict()
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise Experiment5EvaluationError("training receipt must be a mapping")
    expected = {
        "schema_version",
        "panel",
        "condition",
        "seed",
        "checkpoint_target_tokens",
        "actual_training_tokens",
        "training_result_sha256",
        "checkpoint_pointer_sha256",
        "accelerator_type",
        "accelerator_generation",
        "chip_count",
        "marginal_wall_seconds",
        "marginal_chip_seconds",
        "tokens_per_second",
        "useful_mfu",
        "routed_bank_component_counts",
        "receipt_sha256",
    }
    if set(raw) != expected:
        raise Experiment5EvaluationError("training receipt fields differ")
    claimed = _sha256(raw.pop("receipt_sha256"), name="receipt_sha256")
    if raw.get("schema_version") != TRAINING_RECEIPT_SCHEMA_VERSION:
        raise Experiment5EvaluationError("training receipt schema changed")
    if canonical_sha256(raw) != claimed:
        raise Experiment5EvaluationError("training receipt self-hash is invalid")
    rebuilt = build_experiment5_training_receipt(
        **{key: value for key, value in raw.items() if key != "schema_version"}
    )
    if rebuilt.receipt_sha256 != claimed:
        raise Experiment5EvaluationError("training receipt normalization changed")
    return rebuilt


@dataclass(frozen=True, slots=True)
class Experiment5EvaluationPartial:
    """Self-hashed compact output from one TPU evaluation host."""

    partial_id: str
    panel: str
    condition: str
    seed: int
    checkpoint_target_tokens: int
    total_features: int
    hidden_dim: int
    training_receipt_sha256: str
    checkpoint_firing_mask_sha256: str
    chunk_ids: tuple[str, ...]
    sequence_ids: tuple[int, ...]
    sequence_position_bits_hex: tuple[str, ...]
    sequence_valid_tokens: tuple[int, ...]
    sequence_sst: tuple[float, ...]
    sequence_sse: tuple[float, ...]
    heldout_fired_bits_hex: str
    evaluation_bank_component_counts: tuple[int, ...] | None
    endpoint_sse: tuple[float, ...]
    routed_target_sum: tuple[float, ...] | None
    routed_target_squared_norm_sum: float | None
    shared_target_sum: tuple[float, ...] | None
    shared_target_squared_norm_sum: float | None
    tokens: int
    partial_sha256: str
    schema_version: str = EVALUATION_PARTIAL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "partial_id": self.partial_id,
            "panel": self.panel,
            "condition": self.condition,
            "seed": self.seed,
            "checkpoint_target_tokens": self.checkpoint_target_tokens,
            "total_features": self.total_features,
            "hidden_dim": self.hidden_dim,
            "training_receipt_sha256": self.training_receipt_sha256,
            "checkpoint_firing_mask_sha256": (
                self.checkpoint_firing_mask_sha256
            ),
            "chunk_ids": list(self.chunk_ids),
            "sequence_ids": list(self.sequence_ids),
            "sequence_position_bits_hex": list(
                self.sequence_position_bits_hex
            ),
            "sequence_valid_tokens": list(self.sequence_valid_tokens),
            "sequence_sst": list(self.sequence_sst),
            "sequence_sse": list(self.sequence_sse),
            "heldout_fired_bits_hex": self.heldout_fired_bits_hex,
            "evaluation_bank_component_counts": (
                list(self.evaluation_bank_component_counts)
                if self.evaluation_bank_component_counts is not None
                else None
            ),
            "endpoint_sse": dict(
                zip(_ENDPOINT_NAMES, self.endpoint_sse, strict=True)
            ),
            "routed_target_sum": (
                list(self.routed_target_sum)
                if self.routed_target_sum is not None
                else None
            ),
            "routed_target_squared_norm_sum": (
                self.routed_target_squared_norm_sum
            ),
            "shared_target_sum": (
                list(self.shared_target_sum)
                if self.shared_target_sum is not None
                else None
            ),
            "shared_target_squared_norm_sum": (
                self.shared_target_squared_norm_sum
            ),
            "tokens": self.tokens,
            "partial_sha256": self.partial_sha256,
        }


def _position_bits_hex(value: int) -> str:
    if value < 0 or value >= 1 << SEQUENCE_LENGTH:
        raise Experiment5EvaluationError("sequence position bitmap is invalid")
    return value.to_bytes(
        SEQUENCE_LENGTH // 8,
        "little",
        signed=False,
    ).hex()


def _position_bits_from_hex(value: Any, *, name: str) -> int:
    if (
        not isinstance(value, str)
        or len(value) != SEQUENCE_LENGTH // 4
        or re.fullmatch(r"[0-9a-f]+", value) is None
    ):
        raise Experiment5EvaluationError(
            f"{name} must be a canonical {SEQUENCE_LENGTH}-bit hex bitmap"
        )
    return int.from_bytes(bytes.fromhex(value), "little", signed=False)


def _feature_bits_hex(value: np.ndarray) -> str:
    return np.packbits(value, bitorder="little").tobytes(order="C").hex()


def _feature_bits_from_hex(
    value: Any,
    *,
    total_features: int,
) -> np.ndarray:
    expected_bytes = (total_features + 7) // 8
    if (
        not isinstance(value, str)
        or len(value) != expected_bytes * 2
        or re.fullmatch(r"[0-9a-f]+", value) is None
    ):
        raise Experiment5EvaluationError(
            "heldout firing bitmap has incompatible canonical encoding"
        )
    packed = np.frombuffer(bytes.fromhex(value), dtype=np.uint8)
    result = np.unpackbits(
        packed,
        bitorder="little",
        count=expected_bytes * 8,
    )
    if np.any(result[total_features:]):
        raise Experiment5EvaluationError(
            "heldout firing bitmap has nonzero padding bits"
        )
    logical = np.ascontiguousarray(result[:total_features], dtype=np.bool_)
    logical.setflags(write=False)
    return logical


def _build_experiment5_evaluation_partial(
    *,
    partial_id: str,
    panel: str,
    condition: str,
    seed: int,
    checkpoint_target_tokens: int,
    total_features: int,
    hidden_dim: int,
    training_receipt_sha256: str,
    checkpoint_firing_mask_sha256: str,
    chunk_ids: Sequence[str],
    sequence_ids: Sequence[int],
    sequence_position_bits_hex: Sequence[str],
    sequence_valid_tokens: Sequence[int],
    sequence_sst: Sequence[float],
    sequence_sse: Sequence[float],
    heldout_fired_bits_hex: str,
    evaluation_bank_component_counts: Sequence[int] | None,
    endpoint_sse: Mapping[str, Any],
    routed_target_sum: Sequence[float] | None,
    routed_target_squared_norm_sum: float | None,
    shared_target_sum: Sequence[float] | None,
    shared_target_squared_norm_sum: float | None,
    tokens: int,
) -> Experiment5EvaluationPartial:
    if (
        not isinstance(partial_id, str)
        or not partial_id
        or panel not in PANELS
        or condition not in CONDITIONS
        or seed not in SEEDS
        or checkpoint_target_tokens not in MILESTONES
    ):
        raise Experiment5EvaluationError("evaluation partial identity is invalid")
    selected_total = _integer(
        total_features,
        name="total_features",
        minimum=1,
    )
    expected_total = 8_192 if condition == "B" else 69_632
    if selected_total != expected_total:
        raise Experiment5EvaluationError(
            "evaluation partial feature width changed"
        )
    selected_hidden = _integer(hidden_dim, name="hidden_dim", minimum=1)
    selected_tokens = _integer(tokens, name="tokens", minimum=1)
    chunks = tuple(chunk_ids)
    if (
        not chunks
        or tuple(sorted(chunks)) != chunks
        or len(set(chunks)) != len(chunks)
        or any(not isinstance(item, str) or not item for item in chunks)
    ):
        raise Experiment5EvaluationError(
            "evaluation partial chunk IDs must be unique and sorted"
        )
    ids = tuple(
        _integer(value, name="sequence_id") for value in sequence_ids
    )
    position_hex = tuple(sequence_position_bits_hex)
    counts = tuple(
        _integer(value, name="sequence_valid_tokens", minimum=1)
        for value in sequence_valid_tokens
    )
    sst = tuple(
        _finite(value, name="sequence_sst", minimum=0)
        for value in sequence_sst
    )
    sse = tuple(
        _finite(value, name="sequence_sse", minimum=0)
        for value in sequence_sse
    )
    if (
        not ids
        or tuple(sorted(ids)) != ids
        or len(set(ids)) != len(ids)
        or not (
            len(ids)
            == len(position_hex)
            == len(counts)
            == len(sst)
            == len(sse)
        )
        or sum(counts) != selected_tokens
    ):
        raise Experiment5EvaluationError(
            "evaluation partial sequence vectors are incompatible"
        )
    for index, (encoded, count) in enumerate(
        zip(position_hex, counts, strict=True)
    ):
        bits = _position_bits_from_hex(
            encoded,
            name=f"sequence_position_bits_hex[{index}]",
        )
        if bits.bit_count() != count:
            raise Experiment5EvaluationError(
                "sequence position bitmap differs from its valid-token count"
            )
    heldout = _feature_bits_from_hex(
        heldout_fired_bits_hex,
        total_features=selected_total,
    )
    if condition in {"H", "R"}:
        if evaluation_bank_component_counts is None:
            raise Experiment5EvaluationError(
                "hybrid evaluation partial lacks routed bank counts"
            )
        bank_counts = tuple(
            _integer(value, name="evaluation_bank_component_count")
            for value in evaluation_bank_component_counts
        )
        if len(bank_counts) != 128 or sum(bank_counts) != selected_tokens * 8:
            raise Experiment5EvaluationError(
                "evaluation bank counts differ from partial token count"
            )
    else:
        if evaluation_bank_component_counts is not None:
            raise Experiment5EvaluationError(
                "global evaluation partial has routed bank counts"
            )
        bank_counts = None
    if set(endpoint_sse) != set(_ENDPOINT_NAMES):
        raise Experiment5EvaluationError("evaluation partial endpoint fields differ")
    endpoints = tuple(
        _finite(endpoint_sse[name], name=f"endpoint_sse.{name}", minimum=0)
        for name in _ENDPOINT_NAMES
    )
    if condition in {"H", "R"}:
        if routed_target_sum is None or shared_target_sum is None:
            raise Experiment5EvaluationError(
                "hybrid evaluation partial lacks branch sufficient statistics"
            )
        routed_sum = tuple(
            _finite(value, name="routed_target_sum")
            for value in routed_target_sum
        )
        shared_sum = tuple(
            _finite(value, name="shared_target_sum")
            for value in shared_target_sum
        )
        if (
            len(routed_sum) != selected_hidden
            or len(shared_sum) != selected_hidden
        ):
            raise Experiment5EvaluationError(
                "branch target sums differ from hidden dimension"
            )
        routed_squared = _finite(
            routed_target_squared_norm_sum,
            name="routed_target_squared_norm_sum",
            minimum=0,
        )
        shared_squared = _finite(
            shared_target_squared_norm_sum,
            name="shared_target_squared_norm_sum",
            minimum=0,
        )
    else:
        if (
            any(value != 0 for value in endpoints[1:])
            or routed_target_sum is not None
            or routed_target_squared_norm_sum is not None
            or shared_target_sum is not None
            or shared_target_squared_norm_sum is not None
        ):
            raise Experiment5EvaluationError(
                "global evaluation partial has hybrid branch statistics"
            )
        routed_sum = None
        shared_sum = None
        routed_squared = None
        shared_squared = None
    body = {
        "schema_version": EVALUATION_PARTIAL_SCHEMA_VERSION,
        "partial_id": partial_id,
        "panel": panel,
        "condition": condition,
        "seed": seed,
        "checkpoint_target_tokens": checkpoint_target_tokens,
        "total_features": selected_total,
        "hidden_dim": selected_hidden,
        "training_receipt_sha256": _sha256(
            training_receipt_sha256,
            name="training_receipt_sha256",
        ),
        "checkpoint_firing_mask_sha256": _sha256(
            checkpoint_firing_mask_sha256,
            name="checkpoint_firing_mask_sha256",
        ),
        "chunk_ids": list(chunks),
        "sequence_ids": list(ids),
        "sequence_position_bits_hex": list(position_hex),
        "sequence_valid_tokens": list(counts),
        "sequence_sst": list(sst),
        "sequence_sse": list(sse),
        "heldout_fired_bits_hex": _feature_bits_hex(heldout),
        "evaluation_bank_component_counts": (
            list(bank_counts) if bank_counts is not None else None
        ),
        "endpoint_sse": dict(
            zip(_ENDPOINT_NAMES, endpoints, strict=True)
        ),
        "routed_target_sum": (
            list(routed_sum) if routed_sum is not None else None
        ),
        "routed_target_squared_norm_sum": routed_squared,
        "shared_target_sum": (
            list(shared_sum) if shared_sum is not None else None
        ),
        "shared_target_squared_norm_sum": shared_squared,
        "tokens": selected_tokens,
    }
    return Experiment5EvaluationPartial(
        partial_id=partial_id,
        panel=panel,
        condition=condition,
        seed=seed,
        checkpoint_target_tokens=checkpoint_target_tokens,
        total_features=selected_total,
        hidden_dim=selected_hidden,
        training_receipt_sha256=body["training_receipt_sha256"],
        checkpoint_firing_mask_sha256=body[
            "checkpoint_firing_mask_sha256"
        ],
        chunk_ids=chunks,
        sequence_ids=ids,
        sequence_position_bits_hex=position_hex,
        sequence_valid_tokens=counts,
        sequence_sst=sst,
        sequence_sse=sse,
        heldout_fired_bits_hex=body["heldout_fired_bits_hex"],
        evaluation_bank_component_counts=bank_counts,
        endpoint_sse=endpoints,
        routed_target_sum=routed_sum,
        routed_target_squared_norm_sum=routed_squared,
        shared_target_sum=shared_sum,
        shared_target_squared_norm_sum=shared_squared,
        tokens=selected_tokens,
        partial_sha256=canonical_sha256(body),
    )


def validate_experiment5_evaluation_partial(
    value: Experiment5EvaluationPartial | Mapping[str, Any],
) -> Experiment5EvaluationPartial:
    """Validate and normalize one self-hashed cross-process partial."""

    raw = value.to_dict() if isinstance(value, Experiment5EvaluationPartial) else dict(value)
    expected = {
        "schema_version",
        "partial_id",
        "panel",
        "condition",
        "seed",
        "checkpoint_target_tokens",
        "total_features",
        "hidden_dim",
        "training_receipt_sha256",
        "checkpoint_firing_mask_sha256",
        "chunk_ids",
        "sequence_ids",
        "sequence_position_bits_hex",
        "sequence_valid_tokens",
        "sequence_sst",
        "sequence_sse",
        "heldout_fired_bits_hex",
        "evaluation_bank_component_counts",
        "endpoint_sse",
        "routed_target_sum",
        "routed_target_squared_norm_sum",
        "shared_target_sum",
        "shared_target_squared_norm_sum",
        "tokens",
        "partial_sha256",
    }
    if set(raw) != expected:
        raise Experiment5EvaluationError("evaluation partial fields differ")
    claimed = _sha256(raw.pop("partial_sha256"), name="partial_sha256")
    if raw.get("schema_version") != EVALUATION_PARTIAL_SCHEMA_VERSION:
        raise Experiment5EvaluationError("evaluation partial schema changed")
    if canonical_sha256(raw) != claimed:
        raise Experiment5EvaluationError("evaluation partial self-hash is invalid")
    rebuilt = _build_experiment5_evaluation_partial(
        **{key: item for key, item in raw.items() if key != "schema_version"}
    )
    if rebuilt.partial_sha256 != claimed:
        raise Experiment5EvaluationError(
            "evaluation partial normalization changed"
        )
    return rebuilt


@dataclass(frozen=True, slots=True)
class Experiment5EvaluationRows:
    """Compact per-token sufficient rows from one distributed evaluator chunk."""

    chunk_id: str
    panel: str
    condition: str
    seed: int
    checkpoint_target_tokens: int
    sequence_id: np.ndarray
    position: np.ndarray
    valid_mask: np.ndarray
    selected_latent_ids: np.ndarray
    bank_ids: np.ndarray | None
    full_sse: np.ndarray
    full_sst: np.ndarray
    routed_sse: np.ndarray | None
    shared_sse: np.ndarray | None
    oracle_shared_full_sse: np.ndarray | None
    oracle_routed_full_sse: np.ndarray | None
    routed_target_sum: np.ndarray | None
    routed_target_squared_norm_sum: float | None
    shared_target_sum: np.ndarray | None
    shared_target_squared_norm_sum: float | None

    def __post_init__(self) -> None:
        """Validate the compact TPU-to-host scientific boundary."""

        if (
            not isinstance(self.chunk_id, str)
            or not self.chunk_id
            or self.panel not in PANELS
            or self.condition not in CONDITIONS
            or self.seed not in SEEDS
            or self.checkpoint_target_tokens not in MILESTONES
        ):
            raise Experiment5EvaluationError("evaluation row identity is invalid")
        sequence = np.asarray(self.sequence_id)
        position = np.asarray(self.position)
        valid = np.asarray(self.valid_mask)
        latent = np.asarray(self.selected_latent_ids)
        if (
            sequence.ndim != 1
            or not sequence.size
            or sequence.dtype != np.int64
            or position.shape != sequence.shape
            or position.dtype != np.uint16
            or valid.shape != sequence.shape
            or valid.dtype != np.bool_
            or not np.any(valid)
            or np.any(sequence[valid] < 0)
            or np.any(position[valid] >= SEQUENCE_LENGTH)
            or latent.shape != (sequence.size, 64)
            or latent.dtype not in (np.int32, np.int64)
        ):
            raise Experiment5EvaluationError(
                "compact token identities or latent IDs are invalid"
            )
        total_features = 8_192 if self.condition == "B" else 69_632
        valid_latent = latent[valid]
        if (
            np.any(valid_latent < 0)
            or np.any(valid_latent >= total_features)
            or np.any(np.diff(np.sort(valid_latent, axis=1), axis=1) == 0)
            or np.any(latent[~valid] != -1)
        ):
            raise Experiment5EvaluationError(
                "selected latent IDs violate exact logical L0 or feature bounds"
            )

        checked: dict[str, np.ndarray] = {
            "sequence_id": np.ascontiguousarray(sequence, dtype=np.int64),
            "position": np.ascontiguousarray(position, dtype=np.uint16),
            "valid_mask": np.ascontiguousarray(valid, dtype=np.bool_),
            "selected_latent_ids": np.ascontiguousarray(latent, dtype=np.int32),
        }
        hybrid_metric_names = {
            "routed_sse",
            "shared_sse",
            "oracle_shared_full_sse",
            "oracle_routed_full_sse",
        }
        for name in (
            "full_sse",
            "full_sst",
            "routed_sse",
            "shared_sse",
            "oracle_shared_full_sse",
            "oracle_routed_full_sse",
        ):
            raw = getattr(self, name)
            required = (
                name in {"full_sse", "full_sst"}
                or self.condition in {"H", "R"}
            )
            if raw is None:
                if required:
                    raise Experiment5EvaluationError(
                        f"{name} is required for {self.condition}"
                    )
                continue
            if not required and name in hybrid_metric_names:
                raise Experiment5EvaluationError(
                    f"global evaluation rows cannot contain {name}"
                )
            array = np.asarray(raw)
            if (
                array.shape != sequence.shape
                or not np.issubdtype(array.dtype, np.floating)
                or not np.isfinite(array).all()
                or np.any(array < 0)
                or np.any(array[~valid] != 0)
            ):
                raise Experiment5EvaluationError(
                    f"{name} must be finite nonnegative [tokens] with zero padding"
                )
            checked[name] = np.ascontiguousarray(array, dtype=np.float64)

        hybrid = self.condition in {"H", "R"}
        if hybrid:
            selected_banks = np.asarray(self.bank_ids)
            if (
                selected_banks.shape != (sequence.size, 8)
                or selected_banks.dtype not in (np.int32, np.int64)
                or np.any(selected_banks[valid] < 0)
                or np.any(selected_banks[valid] >= 128)
                or np.any(
                    np.diff(np.sort(selected_banks[valid], axis=1), axis=1) == 0
                )
                or np.any(selected_banks[~valid] != -1)
            ):
                raise Experiment5EvaluationError(
                    "hybrid bank IDs must be eight unique logical banks per token"
                )
            routed_ids = valid_latent[:, :32].reshape(-1, 8, 4)
            selected = selected_banks[valid]
            starts = selected[:, :, None] * 512
            if (
                np.any(routed_ids < starts)
                or np.any(routed_ids >= starts + 512)
                or np.any(valid_latent[:, 32:] < 65_536)
            ):
                raise Experiment5EvaluationError(
                    "hybrid latent IDs do not match routed-bank/shared ordering"
                )
            checked["bank_ids"] = np.ascontiguousarray(
                selected_banks,
                dtype=np.int32,
            )
            routed_sum = np.asarray(self.routed_target_sum)
            shared_sum = np.asarray(self.shared_target_sum)
            if (
                routed_sum.ndim != 1
                or not routed_sum.size
                or shared_sum.shape != routed_sum.shape
                or not np.issubdtype(routed_sum.dtype, np.floating)
                or not np.issubdtype(shared_sum.dtype, np.floating)
                or not np.isfinite(routed_sum).all()
                or not np.isfinite(shared_sum).all()
            ):
                raise Experiment5EvaluationError(
                    "hybrid branch target sums are invalid"
                )
            checked["routed_target_sum"] = np.ascontiguousarray(
                routed_sum,
                dtype=np.float64,
            )
            checked["shared_target_sum"] = np.ascontiguousarray(
                shared_sum,
                dtype=np.float64,
            )
            for name in (
                "routed_target_squared_norm_sum",
                "shared_target_squared_norm_sum",
            ):
                _finite(getattr(self, name), name=name, minimum=0)
        elif any(
            value is not None
            for value in (
                self.bank_ids,
                self.routed_sse,
                self.shared_sse,
                self.oracle_shared_full_sse,
                self.oracle_routed_full_sse,
                self.routed_target_sum,
                self.routed_target_squared_norm_sum,
                self.shared_target_sum,
                self.shared_target_squared_norm_sum,
            )
        ):
            raise Experiment5EvaluationError(
                "global evaluation rows contain hybrid-only fields"
            )

        for name, array in checked.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if hybrid:
            object.__setattr__(
                self,
                "routed_target_squared_norm_sum",
                float(self.routed_target_squared_norm_sum),
            )
            object.__setattr__(
                self,
                "shared_target_squared_norm_sum",
                float(self.shared_target_squared_norm_sum),
            )

    @property
    def tokens(self) -> int:
        return int(np.count_nonzero(self.valid_mask))


def _device_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _device_scalar(value: Any, name: str) -> float:
    selected = np.asarray(_device_field(value, name))
    if selected.shape != () or not np.issubdtype(selected.dtype, np.number):
        raise Experiment5EvaluationError(f"{name} must be a device scalar")
    return float(selected)


def pack_experiment5_device_rows(
    *,
    chunk_id: str,
    panel: str,
    condition: str,
    seed: int,
    checkpoint_target_tokens: int,
    sequence_id: Any,
    position: Any,
    valid_mask: Any,
    device_scores: Any,
) -> Experiment5EvaluationRows:
    """Materialize compact feature-sharded TPU scores on the host.

    The large checkpoint and decoded reconstructions never cross this
    boundary. ``device_scores`` may be a mapping or JAX ``NamedTuple``.
    """

    hybrid = condition in {"H", "R"}
    return Experiment5EvaluationRows(
        chunk_id=chunk_id,
        panel=panel,
        condition=condition,
        seed=seed,
        checkpoint_target_tokens=checkpoint_target_tokens,
        sequence_id=np.asarray(sequence_id),
        position=np.asarray(position),
        valid_mask=np.asarray(valid_mask),
        selected_latent_ids=np.asarray(
            _device_field(device_scores, "selected_latent_ids")
        ),
        bank_ids=(
            np.asarray(_device_field(device_scores, "bank_ids"))
            if hybrid
            else None
        ),
        full_sse=np.asarray(_device_field(device_scores, "full_sse")),
        full_sst=np.asarray(_device_field(device_scores, "full_sst")),
        routed_sse=(
            np.asarray(_device_field(device_scores, "routed_sse"))
            if hybrid
            else None
        ),
        shared_sse=(
            np.asarray(_device_field(device_scores, "shared_sse"))
            if hybrid
            else None
        ),
        oracle_shared_full_sse=(
            np.asarray(_device_field(device_scores, "oracle_shared_full_sse"))
            if hybrid
            else None
        ),
        oracle_routed_full_sse=(
            np.asarray(_device_field(device_scores, "oracle_routed_full_sse"))
            if hybrid
            else None
        ),
        routed_target_sum=(
            np.asarray(_device_field(device_scores, "routed_target_sum"))
            if hybrid
            else None
        ),
        routed_target_squared_norm_sum=(
            _device_scalar(device_scores, "routed_target_squared_norm_sum")
            if hybrid
            else None
        ),
        shared_target_sum=(
            np.asarray(_device_field(device_scores, "shared_target_sum"))
            if hybrid
            else None
        ),
        shared_target_squared_norm_sum=(
            _device_scalar(device_scores, "shared_target_squared_norm_sum")
            if hybrid
            else None
        ),
    )


def _mask_metric(value: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.where(valid, value, 0.0).astype(np.float64)
    output.setflags(write=False)
    return output


def score_experiment5_rows(
    *,
    chunk_id: str,
    panel: str,
    seed: int,
    checkpoint_target_tokens: int,
    layout: Experiment5Layout,
    sequence_id: Any,
    position: Any,
    valid_mask: Any,
    selected_latent_ids: Any,
    expert_target: Any,
    shared_target: Any,
    router_weights: Any,
    full_target_mean: Any,
    recomposition: Experiment5Recomposition,
    bank_ids: Any | None = None,
    routed_reconstruction: Any | None = None,
    shared_reconstruction: Any | None = None,
    full_reconstruction: Any | None = None,
) -> Experiment5EvaluationRows:
    """NumPy reference-score one chunk from exact decoded predictions.

    H/R provide reconstructed raw expert outputs and normalized shared output.
    B/F provide only the reconstructed full-FFN target. Logical latent IDs are
    always the exact 64 selected features for each scientific token. Production
    evaluation must use the feature-sharded TPU scorer and
    :func:`pack_experiment5_device_rows`; this oracle exists for parity tests.
    """

    if (
        not isinstance(chunk_id, str)
        or not chunk_id
        or panel not in PANELS
        or seed not in SEEDS
        or checkpoint_target_tokens not in MILESTONES
        or layout.condition not in CONDITIONS
    ):
        raise Experiment5EvaluationError("evaluation row identity is invalid")
    sequence = np.asarray(sequence_id)
    positions = np.asarray(position)
    valid = np.asarray(valid_mask)
    expert = np.asarray(expert_target)
    shared = np.asarray(shared_target)
    weights = np.asarray(router_weights)
    if (
        sequence.ndim != 1
        or sequence.dtype != np.int64
        or positions.dtype != np.uint16
        or positions.shape != sequence.shape
        or valid.dtype != np.bool_
        or valid.shape != sequence.shape
        or not np.any(valid)
        or np.any(sequence[valid] < 0)
        or np.any(positions[valid] >= SEQUENCE_LENGTH)
    ):
        raise Experiment5EvaluationError("token keys or valid mask are invalid")
    tokens = sequence.size
    hidden = layout.hidden_dim
    if (
        expert.shape != (tokens, layout.top_routed, hidden)
        or shared.shape != (tokens, hidden)
        or weights.shape != (tokens, layout.top_routed)
    ):
        raise Experiment5EvaluationError("target operands have incompatible shapes")
    for name, array in (
        ("expert_target", expert),
        ("shared_target", shared),
        ("router_weights", weights),
    ):
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(
            array
        ).all():
            raise Experiment5EvaluationError(f"{name} must be finite")
    if np.any(weights[valid] < 0) or not np.all(
        np.abs(weights[valid].sum(axis=1, dtype=np.float32) - 1.0) <= 0.005
    ):
        raise Experiment5EvaluationError("router weights are invalid")
    mean = np.asarray(full_target_mean)
    if (
        mean.shape != (hidden,)
        or not np.issubdtype(mean.dtype, np.floating)
        or not np.isfinite(mean).all()
    ):
        raise Experiment5EvaluationError("full target mean is invalid")
    latent = np.asarray(selected_latent_ids)
    if latent.shape != (tokens, 64) or latent.dtype not in (np.int32, np.int64):
        raise Experiment5EvaluationError(
            "selected_latent_ids must be integer [tokens,64]"
        )
    if (
        np.any(latent[valid] < 0)
        or np.any(latent[valid] >= layout.total_features)
        or any(len(np.unique(row)) != 64 for row in latent[valid])
        or np.any(latent[~valid] != -1)
    ):
        raise Experiment5EvaluationError(
            "selected latent IDs violate exact L0 or feature bounds"
        )
    target_full = numpy_recompose_experiment5_full_ffn(
        expert,
        shared,
        weights,
        recomposition,
    )
    if layout.is_hybrid:
        routed_hat = np.asarray(routed_reconstruction)
        shared_hat = np.asarray(shared_reconstruction)
        if (
            routed_hat.shape != expert.shape
            or shared_hat.shape != shared.shape
            or not np.isfinite(routed_hat).all()
            or not np.isfinite(shared_hat).all()
        ):
            raise Experiment5EvaluationError(
                "hybrid reconstruction operands are invalid"
            )
        selected_banks = np.asarray(bank_ids)
        if (
            selected_banks.shape != weights.shape
            or selected_banks.dtype not in (np.int32, np.int64)
            or np.any(selected_banks[valid] < 0)
            or np.any(selected_banks[valid] >= layout.num_experts)
            or any(
                len(np.unique(row)) != layout.top_routed
                for row in selected_banks[valid]
            )
            or np.any(selected_banks[~valid] != -1)
        ):
            raise Experiment5EvaluationError("hybrid bank IDs are invalid")
        reconstructed_full = numpy_recompose_experiment5_full_ffn(
            routed_hat,
            shared_hat,
            weights,
            recomposition,
        )
        target_routed = _routed_branch(expert, weights, recomposition)
        reconstructed_routed = _routed_branch(
            routed_hat,
            weights,
            recomposition,
        )
        oracle_shared = numpy_recompose_experiment5_full_ffn(
            routed_hat,
            shared,
            weights,
            recomposition,
        )
        oracle_routed = numpy_recompose_experiment5_full_ffn(
            expert,
            shared_hat,
            weights,
            recomposition,
        )
        routed_sse = _mask_metric(
            _token_sse(target_routed, reconstructed_routed),
            valid,
        )
        shared_sse = _mask_metric(_token_sse(shared, shared_hat), valid)
        oracle_shared_sse = _mask_metric(
            _token_sse(target_full, oracle_shared),
            valid,
        )
        oracle_routed_sse = _mask_metric(
            _token_sse(target_full, oracle_routed),
            valid,
        )
        routed_valid = target_routed[valid].astype(np.float64)
        shared_valid = shared[valid].astype(np.float64)
        routed_sum = np.sum(routed_valid, axis=0, dtype=np.float64)
        shared_sum = np.sum(shared_valid, axis=0, dtype=np.float64)
        routed_squared = float(
            np.sum(np.square(routed_valid), dtype=np.float64)
        )
        shared_squared = float(
            np.sum(np.square(shared_valid), dtype=np.float64)
        )
        checked_banks: np.ndarray | None = np.ascontiguousarray(
            selected_banks,
            dtype=np.int32,
        )
    else:
        reconstructed_full = np.asarray(full_reconstruction)
        if (
            reconstructed_full.shape != target_full.shape
            or not np.issubdtype(reconstructed_full.dtype, np.floating)
            or not np.isfinite(reconstructed_full).all()
            or bank_ids is not None
            or routed_reconstruction is not None
            or shared_reconstruction is not None
        ):
            raise Experiment5EvaluationError(
                "global reconstruction inputs are invalid"
            )
        routed_sse = None
        shared_sse = None
        oracle_shared_sse = None
        oracle_routed_sse = None
        routed_sum = None
        shared_sum = None
        routed_squared = None
        shared_squared = None
        checked_banks = None
    full_sse = _mask_metric(_token_sse(target_full, reconstructed_full), valid)
    full_sst = _mask_metric(_token_sst(target_full, mean), valid)
    checked_sequence = np.ascontiguousarray(sequence, dtype=np.int64)
    checked_position = np.ascontiguousarray(positions, dtype=np.uint16)
    checked_valid = np.ascontiguousarray(valid, dtype=np.bool_)
    checked_latent = np.ascontiguousarray(latent, dtype=np.int32)
    for array in (
        checked_sequence,
        checked_position,
        checked_valid,
        checked_latent,
        checked_banks,
        routed_sum,
        shared_sum,
    ):
        if isinstance(array, np.ndarray):
            array.setflags(write=False)
    return Experiment5EvaluationRows(
        chunk_id=chunk_id,
        panel=panel,
        condition=layout.condition,
        seed=seed,
        checkpoint_target_tokens=checkpoint_target_tokens,
        sequence_id=checked_sequence,
        position=checked_position,
        valid_mask=checked_valid,
        selected_latent_ids=checked_latent,
        bank_ids=checked_banks,
        full_sse=full_sse,
        full_sst=full_sst,
        routed_sse=routed_sse,
        shared_sse=shared_sse,
        oracle_shared_full_sse=oracle_shared_sse,
        oracle_routed_full_sse=oracle_routed_sse,
        routed_target_sum=routed_sum,
        routed_target_squared_norm_sum=routed_squared,
        shared_target_sum=shared_sum,
        shared_target_squared_norm_sum=shared_squared,
    )


@dataclass(frozen=True, slots=True)
class Experiment5ConditionEvaluation:
    panel: str
    condition: str
    seed: int
    milestone: int
    actual_training_tokens: int
    condition_result: dict[str, Any]
    sequence_id: np.ndarray
    valid_tokens: np.ndarray
    sequence_position_bits: tuple[int, ...]
    sequence_sst: np.ndarray
    sequence_sse: np.ndarray


class Experiment5EvaluationAccumulator:
    """Streaming, duplicate-detecting accumulator for one condition."""

    def __init__(
        self,
        *,
        panel: str,
        layout: Experiment5Layout,
        seed: int,
        milestone: int,
        training_receipt: Experiment5TrainingReceipt | Mapping[str, Any],
        checkpoint_firing_shards: Sequence[Experiment5FeatureMaskShard],
        expected_evaluation_tokens: int = EVAL_TOKENS,
    ) -> None:
        self.panel = panel
        self.layout = layout
        self.condition = layout.condition
        self.seed = seed
        self.milestone = milestone
        self.expected_tokens = _integer(
            expected_evaluation_tokens,
            name="expected_evaluation_tokens",
            minimum=1,
        )
        expected_features = 8_192 if self.condition == "B" else 69_632
        expected_candidates = (
            69_632 if self.condition == "F" else 8_192
        )
        if (
            layout.total_features != expected_features
            or layout.candidates_per_token != expected_candidates
            or layout.total_l0 != 64
            or (
                layout.is_hybrid
                and (
                    layout.num_experts != 128
                    or layout.top_routed != 8
                    or layout.routed_bank_width != 512
                    or layout.routed_k != 4
                    or layout.shared_width != 4_096
                    or layout.shared_k != 32
                )
            )
        ):
            raise Experiment5EvaluationError(
                "accumulator layout differs from the frozen Experiment-5 geometry"
            )
        self.training = validate_experiment5_training_receipt(training_receipt)
        if (
            panel not in PANELS
            or seed not in SEEDS
            or milestone not in MILESTONES
            or self.training.panel != panel
            or self.training.condition != self.condition
            or self.training.seed != seed
            or self.training.checkpoint_target_tokens != milestone
        ):
            raise Experiment5EvaluationError(
                "accumulator and training receipt identities differ"
            )
        self.training_fired, self.training_firing_sha256 = (
            merge_experiment5_firing_mask_shards(
                checkpoint_firing_shards,
                total_features=layout.total_features,
            )
        )
        self._chunk_ids: set[str] = set()
        self._partial_ids: set[str] = set()
        self._position_bits: dict[int, int] = {}
        self._sequence_count: dict[int, int] = {}
        self._sequence_sst: dict[int, float] = {}
        self._sequence_sse: dict[int, float] = {}
        self._tokens = 0
        self._heldout_fired = np.zeros(
            (layout.total_features,),
            dtype=np.bool_,
        )
        self._eval_bank_counts = (
            np.zeros((layout.num_experts,), dtype=np.int64)
            if layout.is_hybrid
            else None
        )
        self._endpoint_sse = {
            "full_ffn": 0.0,
            "routed": 0.0,
            "shared": 0.0,
            "oracle_shared_full_ffn": 0.0,
            "oracle_routed_full_ffn": 0.0,
        }
        self._routed_sum = (
            np.zeros((layout.hidden_dim,), dtype=np.float64)
            if layout.is_hybrid
            else None
        )
        self._shared_sum = (
            np.zeros((layout.hidden_dim,), dtype=np.float64)
            if layout.is_hybrid
            else None
        )
        self._routed_squared = 0.0
        self._shared_squared = 0.0

    def add(self, rows: Experiment5EvaluationRows) -> None:
        """Add one disjoint process/update row chunk."""

        if (
            rows.panel != self.panel
            or rows.condition != self.condition
            or rows.seed != self.seed
            or rows.checkpoint_target_tokens != self.milestone
        ):
            raise Experiment5EvaluationError("evaluation chunk identity differs")
        if rows.chunk_id in self._chunk_ids:
            raise Experiment5EvaluationError("duplicate evaluation chunk ID")
        self._chunk_ids.add(rows.chunk_id)
        valid_indices = np.flatnonzero(rows.valid_mask)
        if not valid_indices.size:
            raise Experiment5EvaluationError("evaluation chunk has no valid rows")
        for index in valid_indices:
            sequence = int(rows.sequence_id[index])
            position = int(rows.position[index])
            bit = 1 << position
            if self._position_bits.get(sequence, 0) & bit:
                raise Experiment5EvaluationError(
                    "evaluation chunks contain duplicate token keys"
                )
            self._position_bits[sequence] = (
                self._position_bits.get(sequence, 0) | bit
            )
            self._sequence_count[sequence] = self._sequence_count.get(sequence, 0) + 1
            self._sequence_sst[sequence] = (
                self._sequence_sst.get(sequence, 0.0)
                + float(rows.full_sst[index])
            )
            self._sequence_sse[sequence] = (
                self._sequence_sse.get(sequence, 0.0)
                + float(rows.full_sse[index])
            )
        self._tokens += int(valid_indices.size)
        if self._tokens > self.expected_tokens:
            raise Experiment5EvaluationError(
                "evaluation accumulator exceeded its exact token count"
            )
        selected_ids = rows.selected_latent_ids[rows.valid_mask].reshape(-1)
        self._heldout_fired[selected_ids] = True
        self._endpoint_sse["full_ffn"] += float(
            np.sum(rows.full_sse, dtype=np.float64)
        )
        if self.layout.is_hybrid:
            if (
                rows.bank_ids is None
                or rows.routed_sse is None
                or rows.shared_sse is None
                or rows.oracle_shared_full_sse is None
                or rows.oracle_routed_full_sse is None
                or rows.routed_target_sum is None
                or rows.shared_target_sum is None
                or rows.routed_target_squared_norm_sum is None
                or rows.shared_target_squared_norm_sum is None
                or self._eval_bank_counts is None
                or self._routed_sum is None
                or self._shared_sum is None
            ):
                raise Experiment5EvaluationError(
                    "hybrid compact rows lack required diagnostics"
                )
            self._eval_bank_counts += np.bincount(
                rows.bank_ids[rows.valid_mask].reshape(-1),
                minlength=self.layout.num_experts,
            )
            for name, value in (
                ("routed", rows.routed_sse),
                ("shared", rows.shared_sse),
                ("oracle_shared_full_ffn", rows.oracle_shared_full_sse),
                ("oracle_routed_full_ffn", rows.oracle_routed_full_sse),
            ):
                self._endpoint_sse[name] += float(
                    np.sum(value, dtype=np.float64)
                )
            self._routed_sum += rows.routed_target_sum
            self._shared_sum += rows.shared_target_sum
            self._routed_squared += rows.routed_target_squared_norm_sum
            self._shared_squared += rows.shared_target_squared_norm_sum
        elif any(
            value is not None
            for value in (
                rows.bank_ids,
                rows.routed_sse,
                rows.shared_sse,
                rows.oracle_shared_full_sse,
                rows.oracle_routed_full_sse,
            )
        ):
            raise Experiment5EvaluationError(
                "global evaluation chunk has hybrid diagnostics"
            )

    def export_partial(self, *, partial_id: str) -> Experiment5EvaluationPartial:
        """Seal the current host state as a compact self-hashed partial."""

        if not self._tokens:
            raise Experiment5EvaluationError(
                "cannot export an empty evaluation partial"
            )
        sequence_ids = tuple(sorted(self._sequence_count))
        endpoint = {
            name: self._endpoint_sse[name] for name in _ENDPOINT_NAMES
        }
        return _build_experiment5_evaluation_partial(
            partial_id=partial_id,
            panel=self.panel,
            condition=self.condition,
            seed=self.seed,
            checkpoint_target_tokens=self.milestone,
            total_features=self.layout.total_features,
            hidden_dim=self.layout.hidden_dim,
            training_receipt_sha256=self.training.receipt_sha256,
            checkpoint_firing_mask_sha256=self.training_firing_sha256,
            chunk_ids=tuple(sorted(self._chunk_ids)),
            sequence_ids=sequence_ids,
            sequence_position_bits_hex=tuple(
                _position_bits_hex(self._position_bits[key])
                for key in sequence_ids
            ),
            sequence_valid_tokens=tuple(
                self._sequence_count[key] for key in sequence_ids
            ),
            sequence_sst=tuple(
                self._sequence_sst[key] for key in sequence_ids
            ),
            sequence_sse=tuple(
                self._sequence_sse[key] for key in sequence_ids
            ),
            heldout_fired_bits_hex=_feature_bits_hex(self._heldout_fired),
            evaluation_bank_component_counts=(
                tuple(int(value) for value in self._eval_bank_counts)
                if self._eval_bank_counts is not None
                else None
            ),
            endpoint_sse=endpoint,
            routed_target_sum=(
                tuple(float(value) for value in self._routed_sum)
                if self._routed_sum is not None
                else None
            ),
            routed_target_squared_norm_sum=(
                self._routed_squared if self.layout.is_hybrid else None
            ),
            shared_target_sum=(
                tuple(float(value) for value in self._shared_sum)
                if self._shared_sum is not None
                else None
            ),
            shared_target_squared_norm_sum=(
                self._shared_squared if self.layout.is_hybrid else None
            ),
            tokens=self._tokens,
        )

    def merge_partial(
        self,
        value: Experiment5EvaluationPartial | Mapping[str, Any],
    ) -> None:
        """Merge one disjoint host partial, failing before any overlap mutation."""

        partial = validate_experiment5_evaluation_partial(value)
        if (
            partial.panel != self.panel
            or partial.condition != self.condition
            or partial.seed != self.seed
            or partial.checkpoint_target_tokens != self.milestone
            or partial.total_features != self.layout.total_features
            or partial.hidden_dim != self.layout.hidden_dim
            or partial.training_receipt_sha256
            != self.training.receipt_sha256
            or partial.checkpoint_firing_mask_sha256
            != self.training_firing_sha256
        ):
            raise Experiment5EvaluationError(
                "evaluation partial and accumulator evidence differ"
            )
        if partial.partial_id in self._partial_ids:
            raise Experiment5EvaluationError("duplicate evaluation partial ID")
        if self._chunk_ids.intersection(partial.chunk_ids):
            raise Experiment5EvaluationError(
                "evaluation partial contains duplicate chunk IDs"
            )
        if self._tokens + partial.tokens > self.expected_tokens:
            raise Experiment5EvaluationError(
                "merged evaluation exceeds its exact token count"
            )
        decoded_bits = tuple(
            _position_bits_from_hex(
                encoded,
                name=f"sequence_position_bits_hex[{index}]",
            )
            for index, encoded in enumerate(
                partial.sequence_position_bits_hex
            )
        )
        for sequence, bits in zip(
            partial.sequence_ids,
            decoded_bits,
            strict=True,
        ):
            if self._position_bits.get(sequence, 0) & bits:
                raise Experiment5EvaluationError(
                    "evaluation partial contains duplicate token keys"
                )
        heldout = _feature_bits_from_hex(
            partial.heldout_fired_bits_hex,
            total_features=self.layout.total_features,
        )

        self._partial_ids.add(partial.partial_id)
        self._chunk_ids.update(partial.chunk_ids)
        for sequence, bits, count, sst, sse in zip(
            partial.sequence_ids,
            decoded_bits,
            partial.sequence_valid_tokens,
            partial.sequence_sst,
            partial.sequence_sse,
            strict=True,
        ):
            self._position_bits[sequence] = (
                self._position_bits.get(sequence, 0) | bits
            )
            self._sequence_count[sequence] = (
                self._sequence_count.get(sequence, 0) + count
            )
            self._sequence_sst[sequence] = (
                self._sequence_sst.get(sequence, 0.0) + sst
            )
            self._sequence_sse[sequence] = (
                self._sequence_sse.get(sequence, 0.0) + sse
            )
        self._tokens += partial.tokens
        self._heldout_fired |= heldout
        for name, amount in zip(
            _ENDPOINT_NAMES,
            partial.endpoint_sse,
            strict=True,
        ):
            self._endpoint_sse[name] += amount
        if self.layout.is_hybrid:
            if (
                partial.evaluation_bank_component_counts is None
                or partial.routed_target_sum is None
                or partial.routed_target_squared_norm_sum is None
                or partial.shared_target_sum is None
                or partial.shared_target_squared_norm_sum is None
                or self._eval_bank_counts is None
                or self._routed_sum is None
                or self._shared_sum is None
            ):
                raise Experiment5EvaluationError(
                    "hybrid evaluation partial lacks merge statistics"
                )
            self._eval_bank_counts += np.asarray(
                partial.evaluation_bank_component_counts,
                dtype=np.int64,
            )
            self._routed_sum += np.asarray(
                partial.routed_target_sum,
                dtype=np.float64,
            )
            self._shared_sum += np.asarray(
                partial.shared_target_sum,
                dtype=np.float64,
            )
            self._routed_squared += partial.routed_target_squared_norm_sum
            self._shared_squared += partial.shared_target_squared_norm_sum

    @staticmethod
    def _score(sse: float, sst: float) -> dict[str, float]:
        if not math.isfinite(sse) or sse < 0 or not math.isfinite(sst) or sst <= 0:
            raise Experiment5EvaluationError("endpoint SSE/SST is invalid")
        nmse = sse / sst
        return {
            "sse": sse,
            "sst": sst,
            "nmse": nmse,
            "ev": 1.0 - nmse,
        }

    def finalize(self) -> Experiment5ConditionEvaluation:
        """Seal one condition's result row and packed-sequence vectors."""

        if self._tokens != self.expected_tokens:
            raise Experiment5EvaluationError(
                f"evaluation has {self._tokens} tokens, expected {self.expected_tokens}"
            )
        sequence_ids = np.asarray(sorted(self._sequence_count), dtype=np.int64)
        counts = np.asarray(
            [self._sequence_count[int(key)] for key in sequence_ids],
            dtype=np.int64,
        )
        sequence_sst = np.asarray(
            [self._sequence_sst[int(key)] for key in sequence_ids],
            dtype=np.float64,
        )
        sequence_sse = np.asarray(
            [self._sequence_sse[int(key)] for key in sequence_ids],
            dtype=np.float64,
        )
        if (
            int(counts.sum()) != self.expected_tokens
            or np.any(counts <= 0)
            or np.any(counts > SEQUENCE_LENGTH)
            or np.any(sequence_sst <= 0)
        ):
            raise Experiment5EvaluationError(
                "packed-sequence evaluation rows are invalid"
            )
        common_sst = float(np.sum(sequence_sst, dtype=np.float64))
        full = self._score(self._endpoint_sse["full_ffn"], common_sst)
        if self.layout.is_hybrid:
            if self._routed_sum is None or self._shared_sum is None:
                raise Experiment5EvaluationError(
                    "hybrid branch accumulators are unavailable"
                )
            routed_sst = self._routed_squared - float(
                np.dot(self._routed_sum, self._routed_sum) / self.expected_tokens
            )
            shared_sst = self._shared_squared - float(
                np.dot(self._shared_sum, self._shared_sum) / self.expected_tokens
            )
            routed = self._score(self._endpoint_sse["routed"], routed_sst)
            shared = self._score(self._endpoint_sse["shared"], shared_sst)
            oracle_shared = self._score(
                self._endpoint_sse["oracle_shared_full_ffn"],
                common_sst,
            )
            oracle_routed = self._score(
                self._endpoint_sse["oracle_routed_full_ffn"],
                common_sst,
            )
            cancellation = (
                self._endpoint_sse["full_ffn"]
                - self._endpoint_sse["oracle_shared_full_ffn"]
                - self._endpoint_sse["oracle_routed_full_ffn"]
            )
            if (
                self._eval_bank_counts is None
                or self.training.routed_bank_component_counts is None
            ):
                raise Experiment5EvaluationError(
                    "hybrid routed exposure evidence is unavailable"
                )
            bank_rows = []
            for bank_id in range(self.layout.num_experts):
                start = bank_id * self.layout.routed_bank_width
                stop = start + self.layout.routed_bank_width
                bank_rows.append(
                    {
                        "bank_id": bank_id,
                        "routed_training_components": (
                            self.training.routed_bank_component_counts[bank_id]
                        ),
                        "routed_evaluation_components": int(
                            self._eval_bank_counts[bank_id]
                        ),
                        "cumulative_training_fired_features": int(
                            np.count_nonzero(self.training_fired[start:stop])
                        ),
                        "heldout_fired_features": int(
                            np.count_nonzero(self._heldout_fired[start:stop])
                        ),
                        "bank_features": self.layout.routed_bank_width,
                    }
                )
        else:
            routed = None
            shared = None
            oracle_shared = None
            oracle_routed = None
            cancellation = None
            bank_rows = []
        training_fired = int(np.count_nonzero(self.training_fired))
        heldout_fired = int(np.count_nonzero(self._heldout_fired))
        training = self.training
        condition_result = {
            "condition": self.condition,
            "total_features": self.layout.total_features,
            "candidates_per_token": self.layout.candidates_per_token,
            "nominal_exposure_fraction": (
                self.layout.candidates_per_token / self.layout.total_features
            ),
            "exact_l0": 64,
            "finite": True,
            "full_ffn": full,
            "routed": routed,
            "shared": shared,
            "oracle_shared_full_ffn": oracle_shared,
            "oracle_routed_full_ffn": oracle_routed,
            "error_cancellation_sse_delta": cancellation,
            "cumulative_training_coverage": {
                "fired_features": training_fired,
                "total_features": self.layout.total_features,
                "firing_fraction": training_fired / self.layout.total_features,
            },
            "heldout_nonfiring": {
                "fired_features": heldout_fired,
                "total_features": self.layout.total_features,
                "nonfiring_fraction": (
                    1.0 - heldout_fired / self.layout.total_features
                ),
            },
            "routed_bank_exposure": bank_rows,
            "training": {
                "marginal_chip_seconds": training.marginal_chip_seconds,
                "wall_seconds": training.marginal_wall_seconds,
                "tokens_per_second": training.tokens_per_second,
                "accelerator_type": training.accelerator_type,
                "accelerator_generation": training.accelerator_generation,
                "chip_count": training.chip_count,
                "useful_mfu": training.useful_mfu,
                "training_receipt_sha256": training.receipt_sha256,
                "checkpoint_firing_mask_sha256": (
                    self.training_firing_sha256
                ),
            },
        }
        for array in (sequence_ids, counts, sequence_sst, sequence_sse):
            array.setflags(write=False)
        return Experiment5ConditionEvaluation(
            panel=self.panel,
            condition=self.condition,
            seed=self.seed,
            milestone=self.milestone,
            actual_training_tokens=training.actual_training_tokens,
            condition_result=condition_result,
            sequence_id=sequence_ids,
            valid_tokens=counts,
            sequence_position_bits=tuple(
                self._position_bits[int(key)] for key in sequence_ids
            ),
            sequence_sst=sequence_sst,
            sequence_sse=sequence_sse,
        )


def _parent_identities(panel: str) -> dict[str, str | None]:
    return {
        "train_target_manifest_sha256": TRAIN_TARGET_SHA256,
        "eval_target_manifest_sha256": EVAL_TARGET_SHA256,
        "train_source_overlay_manifest_sha256": (
            TRAIN_SOURCE_SHA256 if panel == "transcoder" else None
        ),
        "eval_source_overlay_manifest_sha256": (
            EVAL_SOURCE_SHA256 if panel == "transcoder" else None
        ),
        "source_overlay_capture_contract_sha256": (
            SOURCE_CAPTURE_CONTRACT_SHA256 if panel == "transcoder" else None
        ),
        "recomposition_sidecar_content_sha256": SIDECAR_CONTENT_SHA256,
        "recomposition_sidecar_file_sha256": SIDECAR_FILE_SHA256,
    }


def build_experiment5_evaluation_result(
    *,
    accumulators: Mapping[str, Experiment5EvaluationAccumulator],
    source_revision: str,
    config_sha256: str,
    completed_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Finalize H/R/B/F into one scientific result and paired-score artifact."""

    if set(accumulators) != set(CONDITIONS):
        raise Experiment5EvaluationError("finalization requires exactly H/R/B/F")
    if not isinstance(source_revision, str) or _GIT_RE.fullmatch(source_revision) is None:
        raise Experiment5EvaluationError("source_revision must be a Git SHA")
    _sha256(config_sha256, name="config_sha256")
    rows = {
        condition: accumulators[condition].finalize()
        for condition in CONDITIONS
    }
    identity = {
        (row.panel, row.seed, row.milestone) for row in rows.values()
    }
    if len(identity) != 1:
        raise Experiment5EvaluationError("condition evaluation identities differ")
    panel, seed, milestone = next(iter(identity))
    if any(row.valid_tokens.sum() != EVAL_TOKENS for row in rows.values()):
        raise Experiment5EvaluationError(
            "production scientific result requires exactly 250,000 tokens"
        )
    canonical = rows["H"]
    for condition in CONDITIONS[1:]:
        row = rows[condition]
        if (
            not np.array_equal(row.sequence_id, canonical.sequence_id)
            or not np.array_equal(row.valid_tokens, canonical.valid_tokens)
            or row.sequence_position_bits != canonical.sequence_position_bits
            or not np.allclose(
                row.sequence_sst,
                canonical.sequence_sst,
                rtol=1e-10,
                atol=1e-7,
            )
        ):
            raise Experiment5EvaluationError(
                f"{condition} does not share the packed-sequence target basis"
            )
    paired_body = {
        "schema_version": PAIRED_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "panel": panel,
        "seed": seed,
        "checkpoint_target_tokens": milestone,
        "packed_sequence_ids": canonical.sequence_id.tolist(),
        "valid_tokens": canonical.valid_tokens.tolist(),
        "sequence_sst": canonical.sequence_sst.tolist(),
        "sequence_sse": {
            condition: rows[condition].sequence_sse.tolist()
            for condition in CONDITIONS
        },
    }
    paired = {
        **paired_body,
        "artifact_sha256": canonical_sha256(paired_body),
    }
    effects = {}
    for contrast, (left, right) in CONTRASTS.items():
        estimate, lower, upper, _samples = bootstrap_experiment5_paired_effect(
            panel=panel,
            seed=seed,
            milestone=milestone,
            contrast=contrast,
            sequence_sst=canonical.sequence_sst,
            left_sequence_sse=rows[left].sequence_sse,
            right_sequence_sse=rows[right].sequence_sse,
        )
        effects[contrast] = {
            "endpoint": "full_ffn",
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
            "confidence_level": 0.95,
            "bootstrap_unit": "packed_sequence",
            "bootstrap_replicates": 20_000,
        }
    result_body = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "curve_point_measured",
        "panel": panel,
        "seed": seed,
        "checkpoint_target_tokens": milestone,
        "checkpoint_actual_tokens": {
            condition: rows[condition].actual_training_tokens
            for condition in CONDITIONS
        },
        "evaluation_tokens": EVAL_TOKENS,
        "source_revision": source_revision,
        "completed_at": completed_at,
        "config_sha256": config_sha256,
        "parents": _parent_identities(panel),
        "common_full_ffn_sst": float(
            np.sum(canonical.sequence_sst, dtype=np.float64)
        ),
        "conditions": {
            condition: rows[condition].condition_result
            for condition in CONDITIONS
        },
        "paired_effects": effects,
        "predictions_artifact_sha256": paired["artifact_sha256"],
    }
    result = {
        **result_body,
        "result_sha256": canonical_sha256(result_body),
    }
    checked_result = validate_experiment5_result(result)
    validate_experiment5_paired_scores(paired, result=checked_result)
    return result, paired


__all__ = [
    "Experiment5ConditionEvaluation",
    "Experiment5EvaluationAccumulator",
    "Experiment5EvaluationError",
    "Experiment5EvaluationPartial",
    "Experiment5EvaluationRows",
    "Experiment5FeatureMaskShard",
    "Experiment5TrainingReceipt",
    "build_experiment5_evaluation_result",
    "build_experiment5_training_receipt",
    "logicalize_experiment5_checkpoint_firing_mask",
    "merge_experiment5_firing_mask_shards",
    "pack_experiment5_device_rows",
    "score_experiment5_rows",
    "validate_experiment5_evaluation_partial",
    "validate_experiment5_training_receipt",
]
