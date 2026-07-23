"""Independent DeepSeek-R1 router-semantics correctness oracle.

This module implements the released R1 selection arithmetic twice: once with
NumPy and once with JAX.  It exists to compare native Hugging Face and MaxText
router results; it is not a production routing kernel and it never supersedes
the IDs and weights emitted by the native MaxText invocation used for capture.

The public JAX checked function returns a :mod:`jax.experimental.checkify`
error value so callers can place the complete calculation under ``jax.jit``::

    checked = jax.jit(
        lambda h, g, b: jax_router_oracle_checked(
            h, g, b, geometry=geometry
        )
    )
    error, (expert_ids, router_weights) = checked(hidden, gate, bias)
    error.throw()

``jax_router_oracle`` is the convenient eager wrapper.  It throws the same
checks on the host, but the wrapper itself should not be put under ``jax.jit``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import jax
from jax.experimental import checkify
import jax.numpy as jnp
import ml_dtypes
import numpy as np


_BF16 = np.dtype(ml_dtypes.bfloat16)
_F32 = np.dtype(np.float32)
_UINT16_EXPERT_CAPACITY = 1 << 16
_WEIGHT_SUM_ATOL = 3e-6
_WEIGHT_SUM_RTOL = 3e-6


class RouterOracleError(ValueError):
    """The oracle input or result violates the frozen router contract."""


class AmbiguousRouterSelection(RouterOracleError):
    """A selection-boundary tie has no shared HF/MaxText set semantics."""


@dataclass(frozen=True, slots=True)
class RouterGeometryV1:
    """Static equal-group geometry for R1 routing and bounded tiny fixtures."""

    hidden_dim: int
    expert_count: int
    group_count: int
    selected_group_count: int
    experts_per_token: int
    group_score_top_k: int = 2
    routed_scaling_factor: float = 2.5

    def __post_init__(self) -> None:
        integer_fields = (
            "hidden_dim",
            "expert_count",
            "group_count",
            "selected_group_count",
            "experts_per_token",
            "group_score_top_k",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise RouterOracleError(f"{name} must be a positive integer")
        if self.expert_count > _UINT16_EXPERT_CAPACITY:
            raise RouterOracleError("expert_count does not fit global uint16 IDs")
        if self.expert_count % self.group_count != 0:
            raise RouterOracleError("experts must divide into equal routing groups")
        if self.selected_group_count > self.group_count:
            raise RouterOracleError("selected_group_count exceeds group_count")
        group_size = self.expert_count // self.group_count
        if self.group_score_top_k > group_size:
            raise RouterOracleError("group_score_top_k exceeds the group size")
        eligible_experts = self.selected_group_count * group_size
        if self.experts_per_token > eligible_experts:
            raise RouterOracleError(
                "experts_per_token exceeds experts in the selected groups"
            )
        if (
            not isinstance(self.routed_scaling_factor, float)
            or not math.isfinite(self.routed_scaling_factor)
            or self.routed_scaling_factor != 2.5
        ):
            raise RouterOracleError("R1 routed_scaling_factor must be exactly 2.5")

    @property
    def group_size(self) -> int:
        return self.expert_count // self.group_count


PINNED_R1_ROUTER_GEOMETRY = RouterGeometryV1(
    hidden_dim=7168,
    expert_count=256,
    group_count=8,
    selected_group_count=4,
    experts_per_token=8,
    group_score_top_k=2,
    routed_scaling_factor=2.5,
)


def validate_pinned_r1_geometry(geometry: RouterGeometryV1) -> None:
    """Require the exact released R1 router geometry."""

    if type(geometry) is not RouterGeometryV1:
        raise RouterOracleError("geometry must be an exact RouterGeometryV1")
    if geometry != PINNED_R1_ROUTER_GEOMETRY:
        raise RouterOracleError("geometry is not the frozen DeepSeek-R1 geometry")


@dataclass(frozen=True, slots=True)
class NumpyRouterResultV1:
    """Canonical global expert-ID/weight pairs, sorted by expert ID."""

    expert_ids: np.ndarray
    router_weights: np.ndarray


def _require_numpy_inputs(
    hidden: np.ndarray,
    gate_kernel: np.ndarray,
    correction_bias: np.ndarray,
    geometry: RouterGeometryV1,
) -> tuple[int, ...]:
    if type(geometry) is not RouterGeometryV1:
        raise RouterOracleError("geometry must be an exact RouterGeometryV1")
    for name, value in (
        ("hidden", hidden),
        ("gate_kernel", gate_kernel),
        ("correction_bias", correction_bias),
    ):
        if type(value) is not np.ndarray:
            raise RouterOracleError(f"{name} must be an exact NumPy ndarray")
    if hidden.ndim < 1 or hidden.shape[-1] != geometry.hidden_dim:
        raise RouterOracleError(f"hidden must have shape [..., {geometry.hidden_dim}]")
    if hidden.size == 0:
        raise RouterOracleError("hidden must contain at least one token")
    if gate_kernel.shape != (geometry.hidden_dim, geometry.expert_count):
        raise RouterOracleError(
            "gate_kernel must have shape "
            f"[{geometry.hidden_dim}, {geometry.expert_count}]"
        )
    if correction_bias.shape != (geometry.expert_count,):
        raise RouterOracleError(
            f"correction_bias must have shape [{geometry.expert_count}]"
        )
    if hidden.dtype != _BF16:
        raise RouterOracleError("hidden must be BF16 before the FP32 router cast")
    if gate_kernel.dtype != _BF16:
        raise RouterOracleError("gate_kernel must be BF16 before the FP32 router cast")
    if correction_bias.dtype != _F32:
        raise RouterOracleError("correction_bias must be FP32")
    if not np.isfinite(hidden.astype(np.float32)).all():
        raise RouterOracleError("hidden contains non-finite values")
    if not np.isfinite(gate_kernel.astype(np.float32)).all():
        raise RouterOracleError("gate_kernel contains non-finite values")
    if not np.isfinite(correction_bias).all():
        raise RouterOracleError("correction_bias contains non-finite values")
    return hidden.shape[:-1]


def _stable_numpy_sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= np.float32(0.0)
    negative_exp = np.exp(np.where(positive, np.float32(0.0), value), dtype=np.float32)
    positive_exp = np.exp(np.where(positive, -value, np.float32(0.0)), dtype=np.float32)
    result = np.where(
        positive,
        np.float32(1.0) / (np.float32(1.0) + positive_exp),
        negative_exp / (np.float32(1.0) + negative_exp),
    )
    return np.asarray(result, dtype=np.float32)


def _numpy_boundary_tie(
    sorted_values: np.ndarray,
    selected_count: int,
) -> np.ndarray:
    candidate_count = sorted_values.shape[-1]
    if selected_count == candidate_count:
        return np.zeros(sorted_values.shape[:-1], dtype=np.bool_)
    return (
        sorted_values[..., -selected_count] == sorted_values[..., -(selected_count + 1)]
    )


def canonicalize_numpy_router_result(
    expert_ids: np.ndarray,
    router_weights: np.ndarray,
    geometry: RouterGeometryV1,
    *,
    expected_leading_shape: tuple[int, ...] | None = None,
) -> NumpyRouterResultV1:
    """Validate result arrays and sort paired records by global expert ID."""

    if type(geometry) is not RouterGeometryV1:
        raise RouterOracleError("geometry must be an exact RouterGeometryV1")
    if type(expert_ids) is not np.ndarray or type(router_weights) is not np.ndarray:
        raise RouterOracleError("router results must be exact NumPy ndarrays")
    if expert_ids.shape != router_weights.shape:
        raise RouterOracleError("expert IDs and weights have different shapes")
    if expert_ids.ndim < 1 or expert_ids.shape[-1] != geometry.experts_per_token:
        raise RouterOracleError(
            "router result must end in exactly experts_per_token records"
        )
    if (
        expected_leading_shape is not None
        and expert_ids.shape[:-1] != expected_leading_shape
    ):
        raise RouterOracleError("router result leading shape changed")
    if expert_ids.dtype.kind not in {"i", "u"}:
        raise RouterOracleError("expert IDs must have an integer dtype")
    if router_weights.dtype != _F32:
        raise RouterOracleError("router weights must be FP32")
    ids_i64 = expert_ids.astype(np.int64, copy=False)
    if np.any(ids_i64 < 0) or np.any(ids_i64 >= geometry.expert_count):
        raise RouterOracleError("router result contains an out-of-range global ID")
    if not np.isfinite(router_weights).all():
        raise RouterOracleError("router result contains non-finite weights")
    if np.any(router_weights < np.float32(0.0)):
        raise RouterOracleError("router result contains negative weights")

    order = np.argsort(ids_i64, axis=-1, kind="stable")
    sorted_ids = np.take_along_axis(ids_i64, order, axis=-1)
    sorted_weights = np.take_along_axis(router_weights, order, axis=-1)
    if geometry.experts_per_token > 1 and np.any(np.diff(sorted_ids, axis=-1) <= 0):
        raise RouterOracleError("router result contains duplicate expert IDs")
    totals = np.sum(sorted_weights, axis=-1, dtype=np.float32)
    if np.any(totals <= np.float32(0.0)):
        raise RouterOracleError("router result has a non-positive weight sum")
    if not np.allclose(
        totals,
        np.float32(geometry.routed_scaling_factor),
        rtol=_WEIGHT_SUM_RTOL,
        atol=_WEIGHT_SUM_ATOL,
    ):
        raise RouterOracleError("router weights do not sum to 2.5")

    canonical_ids = np.ascontiguousarray(sorted_ids, dtype=np.uint16)
    canonical_weights = np.ascontiguousarray(sorted_weights, dtype=np.float32)
    canonical_ids.setflags(write=False)
    canonical_weights.setflags(write=False)
    return NumpyRouterResultV1(canonical_ids, canonical_weights)


def numpy_router_oracle(
    hidden: np.ndarray,
    gate_kernel: np.ndarray,
    correction_bias: np.ndarray,
    *,
    geometry: RouterGeometryV1 = PINNED_R1_ROUTER_GEOMETRY,
) -> NumpyRouterResultV1:
    """Execute the exact R1 selection semantics with NumPy FP32 router math."""

    leading_shape = _require_numpy_inputs(
        hidden, gate_kernel, correction_bias, geometry
    )
    hidden_f32 = hidden.astype(np.float32)
    gate_f32 = gate_kernel.astype(np.float32)
    logits = np.matmul(hidden_f32, gate_f32)
    if logits.dtype != _F32 or not np.isfinite(logits).all():
        raise RouterOracleError("FP32 router projection produced non-finite logits")
    unbiased_scores = _stable_numpy_sigmoid(logits)
    if not np.isfinite(unbiased_scores).all():
        raise RouterOracleError("sigmoid produced non-finite router scores")
    selection_scores = np.add(unbiased_scores, correction_bias, dtype=np.float32)
    if not np.isfinite(selection_scores).all():
        raise RouterOracleError("biased selection scores are non-finite")

    grouped = selection_scores.reshape(
        *leading_shape, geometry.group_count, geometry.group_size
    )
    sorted_within_group = np.sort(grouped, axis=-1)
    group_scores = np.sum(
        sorted_within_group[..., -geometry.group_score_top_k :],
        axis=-1,
        dtype=np.float32,
    )
    if not np.isfinite(group_scores).all():
        raise RouterOracleError("group score reduction produced non-finite values")
    sorted_group_scores = np.sort(group_scores, axis=-1)
    if np.any(_numpy_boundary_tie(sorted_group_scores, geometry.selected_group_count)):
        raise AmbiguousRouterSelection(
            "group selection has a tie at the selected/unselected boundary"
        )
    group_order = np.argsort(group_scores, axis=-1, kind="stable")
    selected_groups = group_order[..., -geometry.selected_group_count :]

    gathered_groups = np.take_along_axis(
        grouped,
        selected_groups[..., :, None],
        axis=-2,
    )
    eligible_scores = gathered_groups.reshape(
        *leading_shape, geometry.selected_group_count * geometry.group_size
    )
    group_offsets = selected_groups[..., :, None] * geometry.group_size
    within_group = np.arange(geometry.group_size, dtype=np.int64)
    eligible_ids = (group_offsets + within_group).reshape(
        *leading_shape, geometry.selected_group_count * geometry.group_size
    )

    sorted_eligible_scores = np.sort(eligible_scores, axis=-1)
    if np.any(_numpy_boundary_tie(sorted_eligible_scores, geometry.experts_per_token)):
        raise AmbiguousRouterSelection(
            "expert selection has a tie at the selected/unselected boundary"
        )
    expert_order = np.argsort(eligible_scores, axis=-1, kind="stable")
    selected_positions = expert_order[..., -geometry.experts_per_token :]
    selected_ids = np.take_along_axis(eligible_ids, selected_positions, axis=-1)
    selected_unbiased = np.take_along_axis(unbiased_scores, selected_ids, axis=-1)
    denominator = np.sum(selected_unbiased, axis=-1, keepdims=True, dtype=np.float32)
    if not np.isfinite(denominator).all() or np.any(denominator <= np.float32(0.0)):
        raise RouterOracleError("selected unbiased scores have no positive denominator")
    weights = np.multiply(
        np.divide(selected_unbiased, denominator, dtype=np.float32),
        np.float32(geometry.routed_scaling_factor),
        dtype=np.float32,
    )
    if not np.isfinite(weights).all():
        raise RouterOracleError("normalized router weights are non-finite")
    return canonicalize_numpy_router_result(
        selected_ids,
        weights,
        geometry,
        expected_leading_shape=leading_shape,
    )


def _require_jax_inputs(
    hidden: Any,
    gate_kernel: Any,
    correction_bias: Any,
    geometry: RouterGeometryV1,
) -> tuple[int, ...]:
    if type(geometry) is not RouterGeometryV1:
        raise RouterOracleError("geometry must be an exact RouterGeometryV1")
    for name, value in (
        ("hidden", hidden),
        ("gate_kernel", gate_kernel),
        ("correction_bias", correction_bias),
    ):
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise RouterOracleError(f"{name} must be a JAX-compatible array")
    if len(hidden.shape) < 1 or hidden.shape[-1] != geometry.hidden_dim:
        raise RouterOracleError(f"hidden must have shape [..., {geometry.hidden_dim}]")
    if math.prod(hidden.shape) == 0:
        raise RouterOracleError("hidden must contain at least one token")
    if tuple(gate_kernel.shape) != (geometry.hidden_dim, geometry.expert_count):
        raise RouterOracleError(
            "gate_kernel must have shape "
            f"[{geometry.hidden_dim}, {geometry.expert_count}]"
        )
    if tuple(correction_bias.shape) != (geometry.expert_count,):
        raise RouterOracleError(
            f"correction_bias must have shape [{geometry.expert_count}]"
        )
    if jnp.dtype(hidden.dtype) != jnp.dtype(jnp.bfloat16):
        raise RouterOracleError("hidden must be BF16 before the FP32 router cast")
    if jnp.dtype(gate_kernel.dtype) != jnp.dtype(jnp.bfloat16):
        raise RouterOracleError("gate_kernel must be BF16 before the FP32 router cast")
    if jnp.dtype(correction_bias.dtype) != jnp.dtype(jnp.float32):
        raise RouterOracleError("correction_bias must be FP32")
    return tuple(hidden.shape[:-1])


def canonicalize_jax_router_result(
    expert_ids: jax.Array,
    router_weights: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sort paired JAX records by global ID without assigning tie semantics."""

    order = jnp.argsort(expert_ids, axis=-1, stable=True)
    sorted_ids = jnp.take_along_axis(expert_ids, order, axis=-1)
    sorted_weights = jnp.take_along_axis(router_weights, order, axis=-1)
    return sorted_ids.astype(jnp.uint16), sorted_weights.astype(jnp.float32)


def _jax_router_oracle_core(
    hidden: jax.Array,
    gate_kernel: jax.Array,
    correction_bias: jax.Array,
    geometry: RouterGeometryV1,
) -> tuple[jax.Array, jax.Array]:
    checkify.check(jnp.all(jnp.isfinite(hidden)), "hidden contains non-finite values")
    checkify.check(
        jnp.all(jnp.isfinite(gate_kernel)),
        "gate_kernel contains non-finite values",
    )
    checkify.check(
        jnp.all(jnp.isfinite(correction_bias)),
        "correction_bias contains non-finite values",
    )
    hidden_f32 = hidden.astype(jnp.float32)
    gate_f32 = gate_kernel.astype(jnp.float32)
    logits = jnp.matmul(hidden_f32, gate_f32)
    checkify.check(
        jnp.all(jnp.isfinite(logits)),
        "FP32 router projection produced non-finite logits",
    )
    unbiased_scores = jax.nn.sigmoid(logits).astype(jnp.float32)
    checkify.check(
        jnp.all(jnp.isfinite(unbiased_scores)),
        "sigmoid produced non-finite router scores",
    )
    selection_scores = unbiased_scores + correction_bias
    checkify.check(
        jnp.all(jnp.isfinite(selection_scores)),
        "biased selection scores are non-finite",
    )

    leading_shape = hidden.shape[:-1]
    grouped = selection_scores.reshape(
        *leading_shape, geometry.group_count, geometry.group_size
    )
    sorted_within_group = jnp.sort(grouped, axis=-1)
    group_scores = jnp.sum(
        sorted_within_group[..., -geometry.group_score_top_k :],
        axis=-1,
        dtype=jnp.float32,
    )
    checkify.check(
        jnp.all(jnp.isfinite(group_scores)),
        "group score reduction produced non-finite values",
    )
    sorted_group_scores = jnp.sort(group_scores, axis=-1)
    if geometry.selected_group_count < geometry.group_count:
        group_boundary_tie = (
            sorted_group_scores[..., -geometry.selected_group_count]
            == sorted_group_scores[..., -(geometry.selected_group_count + 1)]
        )
        checkify.check(
            ~jnp.any(group_boundary_tie),
            "group selection has a boundary tie",
        )
    group_order = jnp.argsort(group_scores, axis=-1, stable=True)
    selected_groups = group_order[..., -geometry.selected_group_count :]

    gathered_groups = jnp.take_along_axis(
        grouped,
        selected_groups[..., :, None],
        axis=-2,
    )
    eligible_count = geometry.selected_group_count * geometry.group_size
    eligible_scores = gathered_groups.reshape(*leading_shape, eligible_count)
    group_offsets = selected_groups[..., :, None] * geometry.group_size
    within_group = jnp.arange(geometry.group_size, dtype=jnp.int32)
    eligible_ids = (group_offsets + within_group).reshape(
        *leading_shape, eligible_count
    )

    sorted_eligible_scores = jnp.sort(eligible_scores, axis=-1)
    if geometry.experts_per_token < eligible_count:
        expert_boundary_tie = (
            sorted_eligible_scores[..., -geometry.experts_per_token]
            == sorted_eligible_scores[..., -(geometry.experts_per_token + 1)]
        )
        checkify.check(
            ~jnp.any(expert_boundary_tie),
            "expert selection has a boundary tie",
        )
    expert_order = jnp.argsort(eligible_scores, axis=-1, stable=True)
    selected_positions = expert_order[..., -geometry.experts_per_token :]
    selected_ids = jnp.take_along_axis(eligible_ids, selected_positions, axis=-1)
    selected_unbiased = jnp.take_along_axis(unbiased_scores, selected_ids, axis=-1)
    denominator = jnp.sum(selected_unbiased, axis=-1, keepdims=True, dtype=jnp.float32)
    checkify.check(
        jnp.all(jnp.isfinite(denominator)),
        "selected unbiased denominator is non-finite",
    )
    checkify.check(
        jnp.all(denominator > jnp.float32(0.0)),
        "selected unbiased scores have no positive denominator",
    )
    weights = (
        selected_unbiased / denominator * jnp.float32(geometry.routed_scaling_factor)
    ).astype(jnp.float32)
    checkify.check(
        jnp.all(jnp.isfinite(weights)),
        "normalized router weights are non-finite",
    )
    checkify.check(
        jnp.all(weights >= jnp.float32(0.0)),
        "normalized router weights are negative",
    )

    canonical_ids, canonical_weights = canonicalize_jax_router_result(
        selected_ids, weights
    )
    if geometry.experts_per_token > 1:
        checkify.check(
            jnp.all(jnp.diff(canonical_ids.astype(jnp.int32), axis=-1) > 0),
            "router result contains duplicate expert IDs",
        )
    totals = jnp.sum(canonical_weights, axis=-1, dtype=jnp.float32)
    checkify.check(
        jnp.all(
            jnp.isclose(
                totals,
                jnp.float32(geometry.routed_scaling_factor),
                rtol=_WEIGHT_SUM_RTOL,
                atol=_WEIGHT_SUM_ATOL,
            )
        ),
        "router weights do not sum to 2.5",
    )
    return canonical_ids, canonical_weights


_CHECKED_JAX_ROUTER_ORACLE = checkify.checkify(_jax_router_oracle_core)


def jax_router_oracle_checked(
    hidden: jax.Array,
    gate_kernel: jax.Array,
    correction_bias: jax.Array,
    *,
    geometry: RouterGeometryV1 = PINNED_R1_ROUTER_GEOMETRY,
) -> tuple[checkify.Error, tuple[jax.Array, jax.Array]]:
    """Return checkified JAX router evidence; this wrapper is safe under jit."""

    _require_jax_inputs(hidden, gate_kernel, correction_bias, geometry)
    return _CHECKED_JAX_ROUTER_ORACLE(hidden, gate_kernel, correction_bias, geometry)


def jax_router_oracle(
    hidden: jax.Array,
    gate_kernel: jax.Array,
    correction_bias: jax.Array,
    *,
    geometry: RouterGeometryV1 = PINNED_R1_ROUTER_GEOMETRY,
) -> tuple[jax.Array, jax.Array]:
    """Eager checked JAX oracle; use ``jax_router_oracle_checked`` under jit."""

    error, result = jax_router_oracle_checked(
        hidden, gate_kernel, correction_bias, geometry=geometry
    )
    try:
        error.throw()
    except checkify.JaxRuntimeError as exc:
        message = str(exc)
        if "boundary tie" in message:
            raise AmbiguousRouterSelection(message) from exc
        raise RouterOracleError(message) from exc
    return result


def validate_jax_router_result(
    expert_ids: jax.Array,
    router_weights: jax.Array,
    geometry: RouterGeometryV1,
    *,
    expected_leading_shape: tuple[int, ...] | None = None,
) -> NumpyRouterResultV1:
    """Transfer a JAX result to the host and apply strict canonical validation."""

    ids = np.asarray(jax.device_get(expert_ids))
    weights = np.asarray(jax.device_get(router_weights), dtype=np.float32)
    return canonicalize_numpy_router_result(
        ids,
        weights,
        geometry,
        expected_leading_shape=expected_leading_shape,
    )


__all__ = [
    "AmbiguousRouterSelection",
    "NumpyRouterResultV1",
    "PINNED_R1_ROUTER_GEOMETRY",
    "RouterGeometryV1",
    "RouterOracleError",
    "canonicalize_jax_router_result",
    "canonicalize_numpy_router_result",
    "jax_router_oracle",
    "jax_router_oracle_checked",
    "numpy_router_oracle",
    "validate_jax_router_result",
    "validate_pinned_r1_geometry",
]
