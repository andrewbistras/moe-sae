"""Pure scientific primitives for expert-aligned contribution SAEs.

Experiment 2 reconstructs the eight selected expert outputs independently,
then applies Gemma's native router weights and routed-branch normalization.
This module defines the condition matrix and host-side correctness oracle.  It
deliberately contains no JAX, MaxText, checkpoint, or storage code.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ContributionCondition = Literal["A", "R", "B", "F"]

CONTRIBUTION_CONDITIONS = ("A", "R", "B", "F")
CONTRIBUTION_RANDOMIZATION_DOMAIN = "gemma4-contribution-sequence-bank-permutation-v1"
DEFAULT_HIDDEN_DIM = 2_816
DEFAULT_NUM_EXPERTS = 128
DEFAULT_TOP_ROUTED = 8
DEFAULT_BANK_WIDTH = 1_024
DEFAULT_K_PER_CONTRIBUTION = 8
PILOT_BANK_WIDTH = 256
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_UINT64_MAX = 2**64 - 1


class ContributionSAEError(ValueError):
    """An Experiment-2 scientific input violates the frozen contract."""


def _integer(
    name: str,
    value: Any,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ContributionSAEError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ContributionSAEError(f"{name} must be in [{minimum}, {maximum}]")
    return result


@dataclass(frozen=True, slots=True)
class ContributionSAEConfig:
    """Static dictionary layout for one objective-matched condition."""

    condition: ContributionCondition
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_experts: int = DEFAULT_NUM_EXPERTS
    top_routed: int = DEFAULT_TOP_ROUTED
    bank_width: int = DEFAULT_BANK_WIDTH
    k_per_contribution: int = DEFAULT_K_PER_CONTRIBUTION
    randomization_key: int = 0

    def __post_init__(self) -> None:
        if self.condition not in CONTRIBUTION_CONDITIONS:
            raise ContributionSAEError(
                "condition must be one of "
                f"{CONTRIBUTION_CONDITIONS}, got {self.condition!r}"
            )
        for name in (
            "hidden_dim",
            "num_experts",
            "top_routed",
            "bank_width",
            "k_per_contribution",
        ):
            _integer(name, getattr(self, name), minimum=1, maximum=2**31 - 1)
        _integer(
            "randomization_key",
            self.randomization_key,
            minimum=0,
            maximum=_UINT64_MAX,
        )
        if self.top_routed > self.num_experts:
            raise ContributionSAEError("top_routed cannot exceed num_experts")
        if self.k_per_contribution > self.bank_width:
            raise ContributionSAEError("k_per_contribution cannot exceed bank_width")

    @property
    def bank_count(self) -> int:
        if self.condition in ("A", "R"):
            return self.num_experts
        return self.top_routed

    @property
    def total_features(self) -> int:
        return self.bank_count * self.bank_width

    @property
    def candidates_per_token(self) -> int:
        return self.top_routed * self.bank_width

    @property
    def total_l0(self) -> int:
        return self.top_routed * self.k_per_contribution

    @property
    def parameter_count(self) -> int:
        # Encoder + decoder + encoder bias.  The common centering mean is fixed.
        return 2 * self.hidden_dim * self.total_features + self.total_features


def production_contribution_config(
    condition: ContributionCondition,
    *,
    randomization_key: int,
) -> ContributionSAEConfig:
    """Return the frozen full-width A/R/B/F layout."""

    widths = {
        "A": DEFAULT_BANK_WIDTH,
        "R": DEFAULT_BANK_WIDTH,
        "B": DEFAULT_BANK_WIDTH,
        "F": DEFAULT_BANK_WIDTH * 16,
    }
    try:
        width = widths[condition]
    except KeyError as error:
        raise ContributionSAEError(
            f"unsupported contribution condition: {condition!r}"
        ) from error
    return ContributionSAEConfig(
        condition=condition,
        bank_width=width,
        randomization_key=randomization_key,
    )


def pilot_contribution_config(
    condition: ContributionCondition,
    *,
    randomization_key: int,
) -> ContributionSAEConfig:
    """Return the frozen seed-0 pilot A0/R0/B0/F0 layout."""

    widths = {
        "A": PILOT_BANK_WIDTH,
        "R": PILOT_BANK_WIDTH,
        "B": PILOT_BANK_WIDTH,
        "F": PILOT_BANK_WIDTH * 16,
    }
    try:
        width = widths[condition]
    except KeyError as error:
        raise ContributionSAEError(
            f"unsupported contribution condition: {condition!r}"
        ) from error
    return ContributionSAEConfig(
        condition=condition,
        bank_width=width,
        k_per_contribution=2,
        randomization_key=randomization_key,
    )


def canonical_router_order(
    expert_ids: Any,
    router_weights: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort every token by descending weight, then ascending expert ID.

    Returns ``(ordered_ids, ordered_weights, source_slots)``.  The source-slot
    indices allow callers to reorder selected expert outputs with exactly the
    same permutation.
    """

    ids = np.asarray(expert_ids)
    weights = np.asarray(router_weights)
    if ids.ndim != 2 or ids.shape[1] != DEFAULT_TOP_ROUTED:
        raise ContributionSAEError(
            f"expert_ids must have shape [N, {DEFAULT_TOP_ROUTED}], got {ids.shape}"
        )
    if weights.shape != ids.shape:
        raise ContributionSAEError(
            f"router_weights must have shape {ids.shape}, got {weights.shape}"
        )
    if not np.issubdtype(ids.dtype, np.integer):
        raise ContributionSAEError("expert_ids must use an integer dtype")
    if not np.issubdtype(weights.dtype, np.floating):
        raise ContributionSAEError("router_weights must use a floating dtype")
    if ids.size and (np.any(ids < 0) or np.any(ids >= DEFAULT_NUM_EXPERTS)):
        raise ContributionSAEError("expert_ids are outside [0, 127]")
    if ids.shape[1] > 1 and np.any(
        np.diff(np.sort(ids.astype(np.int64, copy=False), axis=1), axis=1) == 0
    ):
        raise ContributionSAEError("every token must contain unique expert IDs")
    weights64 = weights.astype(np.float64, copy=False)
    if not np.isfinite(weights64).all():
        raise ContributionSAEError("router_weights must be finite")
    if np.any(weights64 < 0):
        raise ContributionSAEError("router_weights must be nonnegative")
    if ids.shape[0] and not np.allclose(
        np.sum(weights64, axis=1), 1.0, rtol=2e-3, atol=2e-3
    ):
        raise ContributionSAEError("router_weights must sum to one per token")

    # np.lexsort uses the last key as primary.  The explicit ID key supplies a
    # total deterministic ordering for equal BF16 router weights.
    source_slots = np.lexsort(
        (ids.astype(np.int64, copy=False), -weights64),
        axis=1,
    )
    return (
        np.take_along_axis(ids, source_slots, axis=1),
        np.take_along_axis(weights, source_slots, axis=1),
        source_slots,
    )


def sequence_bank_permutation(
    sequence_id: int,
    *,
    key: int,
    num_experts: int = DEFAULT_NUM_EXPERTS,
) -> np.ndarray:
    """Return a stable keyed expert-to-bank bijection for one sequence."""

    selected_sequence = _integer(
        "sequence_id",
        sequence_id,
        minimum=_INT64_MIN,
        maximum=_INT64_MAX,
    )
    selected_key = _integer(
        "key",
        key,
        minimum=0,
        maximum=_UINT64_MAX,
    )
    selected_experts = _integer(
        "num_experts",
        num_experts,
        minimum=2,
        maximum=2**31 - 1,
    )
    prefix = (
        CONTRIBUTION_RANDOMIZATION_DOMAIN.encode("ascii")
        + b"\0"
        + selected_key.to_bytes(8, "little", signed=False)
        + selected_sequence.to_bytes(8, "little", signed=True)
    )
    labels = list(range(selected_experts))
    labels.sort(
        key=lambda expert: (
            hashlib.blake2b(
                prefix + expert.to_bytes(4, "little", signed=False),
                digest_size=16,
            ).digest(),
            expert,
        )
    )
    permutation = np.empty(selected_experts, dtype=np.int64)
    permutation[np.asarray(labels, dtype=np.int64)] = np.arange(
        selected_experts,
        dtype=np.int64,
    )
    return permutation


def condition_bank_ids(
    ordered_expert_ids: Any,
    sequence_ids: Any,
    config: ContributionSAEConfig,
) -> np.ndarray:
    """Map canonical router slots to the condition's dictionary banks."""

    ids = np.asarray(ordered_expert_ids)
    sequences = np.asarray(sequence_ids)
    if sequences.ndim != 1:
        raise ContributionSAEError("sequence_ids must have shape [N]")
    if ids.shape != (sequences.shape[0], config.top_routed):
        raise ContributionSAEError(
            "ordered_expert_ids and sequence_ids do not match configured routes"
        )
    if not np.issubdtype(ids.dtype, np.integer):
        raise ContributionSAEError("ordered_expert_ids must use an integer dtype")
    if not np.issubdtype(sequences.dtype, np.signedinteger):
        raise ContributionSAEError("sequence_ids must use a signed integer dtype")
    if ids.size and (np.any(ids < 0) or np.any(ids >= config.num_experts)):
        raise ContributionSAEError(
            "ordered_expert_ids are outside the configured expert range"
        )
    if config.condition == "A":
        return ids.astype(np.int32, copy=True)
    if config.condition == "R":
        output = np.empty(ids.shape, dtype=np.int32)
        for sequence in np.unique(sequences):
            sequence_value = _integer(
                "sequence_id",
                sequence,
                minimum=_INT64_MIN,
                maximum=_INT64_MAX,
            )
            permutation = sequence_bank_permutation(
                sequence_value,
                key=config.randomization_key,
                num_experts=config.num_experts,
            )
            selected = sequences == sequence
            output[selected] = permutation[ids[selected]]
        if ids.shape[1] > 1 and np.any(np.diff(np.sort(output, axis=1), axis=1) == 0):
            raise ContributionSAEError(
                "randomized bank mapping did not preserve token uniqueness"
            )
        return output

    ranks = np.arange(config.top_routed, dtype=np.int32)
    return np.broadcast_to(ranks, ids.shape).copy()


@dataclass(frozen=True, slots=True)
class ContributionTokenScores:
    """Per-source-token errors for clustered evaluation."""

    token_sse: np.ndarray
    token_sst: np.ndarray
    valid_mask: np.ndarray

    @property
    def tokens(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def sse(self) -> float:
        return float(np.sum(self.token_sse, dtype=np.float64))

    @property
    def sst(self) -> float:
        return float(np.sum(self.token_sst, dtype=np.float64))

    @property
    def nmse(self) -> float:
        if not self.sst > 0:
            raise ContributionSAEError("weighted contribution SST must be positive")
        return self.sse / self.sst

    @property
    def ev(self) -> float:
        return 1.0 - self.nmse


def weighted_contribution_token_scores(
    expert_output: Any,
    reconstruction: Any,
    router_weights: Any,
    pooled_mean: Any,
    *,
    valid_mask: Any | None = None,
) -> ContributionTokenScores:
    """Compute FP64-accumulated SSE/SST after native FP32 router weighting."""

    target = np.asarray(expert_output)
    estimate = np.asarray(reconstruction)
    weights = np.asarray(router_weights)
    mean = np.asarray(pooled_mean)
    if target.ndim != 3 or target.shape[1] != DEFAULT_TOP_ROUTED:
        raise ContributionSAEError("expert_output must have shape [N, 8, hidden]")
    if estimate.shape != target.shape:
        raise ContributionSAEError("reconstruction must match expert_output")
    if weights.shape != target.shape[:2]:
        raise ContributionSAEError(
            "router_weights must match expert_output's token and slot axes"
        )
    if mean.shape != (target.shape[2],):
        raise ContributionSAEError(f"pooled_mean must have shape [{target.shape[2]}]")
    for name, value in (
        ("expert_output", target),
        ("reconstruction", estimate),
        ("router_weights", weights),
        ("pooled_mean", mean),
    ):
        if not np.issubdtype(value.dtype, np.floating):
            raise ContributionSAEError(f"{name} must use a floating dtype")
        if not np.isfinite(value).all():
            raise ContributionSAEError(f"{name} must be finite")
    if np.any(weights < 0):
        raise ContributionSAEError("router_weights must be nonnegative")

    if valid_mask is None:
        valid = np.ones((target.shape[0],), dtype=np.bool_)
    else:
        valid = np.asarray(valid_mask)
        if valid.shape != (target.shape[0],) or valid.dtype != np.bool_:
            raise ContributionSAEError(f"valid_mask must be bool [{target.shape[0]}]")

    target32 = target.astype(np.float32)
    estimate32 = estimate.astype(np.float32)
    weights32 = weights.astype(np.float32)
    mean32 = mean.astype(np.float32)
    error = weights32[..., None] * (target32 - estimate32)
    centered = weights32[..., None] * (target32 - mean32[None, None, :])
    token_sse = np.sum(
        np.square(error, dtype=np.float32),
        axis=(1, 2),
        dtype=np.float64,
    )
    token_sst = np.sum(
        np.square(centered, dtype=np.float32),
        axis=(1, 2),
        dtype=np.float64,
    )
    token_sse = np.where(valid, token_sse, 0.0)
    token_sst = np.where(valid, token_sst, 0.0)
    return ContributionTokenScores(token_sse, token_sst, valid.copy())


def weighted_expert_sum(expert_output: Any, router_weights: Any) -> np.ndarray:
    """Return the native logical FP32 routed sum before branch RMSNorm."""

    target = np.asarray(expert_output)
    weights = np.asarray(router_weights)
    if target.ndim != 3 or target.shape[1] != DEFAULT_TOP_ROUTED:
        raise ContributionSAEError("expert_output must have shape [N, 8, hidden]")
    if weights.shape != target.shape[:2]:
        raise ContributionSAEError(
            "router_weights must match expert_output's token and slot axes"
        )
    if not np.issubdtype(target.dtype, np.floating) or not np.issubdtype(
        weights.dtype, np.floating
    ):
        raise ContributionSAEError(
            "expert_output and router_weights must use floating dtypes"
        )
    if not np.isfinite(target).all() or not np.isfinite(weights).all():
        raise ContributionSAEError("expert_output and router_weights must be finite")
    return np.sum(
        weights.astype(np.float32)[..., None] * target.astype(np.float32),
        axis=1,
        dtype=np.float32,
    )


def validate_contribution_l0(
    per_contribution_l0: Any,
    valid_mask: Any,
    config: ContributionSAEConfig,
) -> None:
    """Require exact TopK in every valid contribution and zero on padding."""

    l0 = np.asarray(per_contribution_l0)
    valid = np.asarray(valid_mask)
    if valid.ndim != 1:
        raise ContributionSAEError("valid_mask must have shape [N]")
    if l0.shape != (valid.shape[0], config.top_routed):
        raise ContributionSAEError(
            "per_contribution_l0 and valid_mask do not match configured routes"
        )
    if not np.issubdtype(l0.dtype, np.integer):
        raise ContributionSAEError("per_contribution_l0 must be integer")
    if valid.dtype != np.bool_:
        raise ContributionSAEError("valid_mask must be boolean")
    if np.any(l0[valid] != config.k_per_contribution):
        raise ContributionSAEError(
            "valid contributions must have exact per-contribution TopK"
        )
    if np.any(l0[~valid] != 0):
        raise ContributionSAEError("padding contributions must have zero L0")
    total = np.sum(l0, axis=1, dtype=np.int64)
    if np.any(total[valid] != config.total_l0):
        raise ContributionSAEError("valid tokens must have exact total L0")


def gain_recovery(
    ev_a: float,
    ev_b: float,
    ev_f: float,
) -> float | None:
    """Return the unclipped percentage of F-over-B gain recovered by A."""

    values = (float(ev_a), float(ev_b), float(ev_f))
    if not all(math.isfinite(value) for value in values):
        raise ContributionSAEError("EV values must be finite")
    denominator = values[2] - values[1]
    if denominator <= 0:
        return None
    return 100.0 * (values[0] - values[1]) / denominator


__all__ = [
    "CONTRIBUTION_CONDITIONS",
    "CONTRIBUTION_RANDOMIZATION_DOMAIN",
    "ContributionCondition",
    "ContributionSAEConfig",
    "ContributionSAEError",
    "ContributionTokenScores",
    "canonical_router_order",
    "condition_bank_ids",
    "gain_recovery",
    "pilot_contribution_config",
    "production_contribution_config",
    "sequence_bank_permutation",
    "validate_contribution_l0",
    "weighted_contribution_token_scores",
    "weighted_expert_sum",
]
