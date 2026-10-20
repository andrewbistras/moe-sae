"""Checkpoint-free Gemma 4 Experiment-2 recomposition oracle.

This module is a host NumPy reference for the exact *stages* around Gemma 4's
block-15 routed and shared experts.  It deliberately consumes the strict v2
wire representation and a validated, content-addressed recomposition sidecar;
it never loads MaxText, JAX, a checkpoint, or a cloud client.

The reference follows pinned MaxText commit
``48c7da3e28e8f57fe36ddf09d00eec53072a2ea0``:

* ``layers/moe.py:962-986`` restores selected outputs, performs the weighted
  reduction in FP32 when ``float32_weight_sum`` is enabled, and casts the
  routed aggregate back to the model dtype;
* ``layers/normalizations.py:69-92`` casts RMSNorm inputs to FP32, computes the
  mean square in FP32, casts the unit-normalized value to the model dtype, and
  applies the scale in that dtype;
* ``models/gemma4.py:123-145`` independently normalizes shared and routed
  outputs before adding them; and
* ``models/gemma4.py:360-379`` applies outer FFN RMSNorm, adds the residual,
  and multiplies the result by the layer scalar.

NumPy has no native BF16 array in the supported runtime, so every BF16 stage is
represented by its exact ``uint16`` bits.  Names ending in ``_f32`` are actual
FP32 arrays; names ending in ``_bf16_bits`` are exact BF16 wire arrays.

One hardware caveat cannot be hidden: the pinned routed sum is a JAX
``einsum``/dot reduction and each RMS mean-square is also a reduction.  XLA is
free to choose target-dependent reduction trees for both.  This oracle
evaluates the same expressions in NumPy FP32 and exposes every surrounding
BF16 round point.  Live qualification must compare captured native stages and
record the diagnostics; it must not claim bit identity for the pre-cast FP32
accumulator or mean-square solely from this host reference.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    RMS_NORM_EPSILON,
    TOP_K_EXPERTS,
    ContributionBatch,
    ContributionIntegrityError,
    ContributionSchemaError,
    bf16_bits_to_float32,
    float32_to_bf16_bits,
    validate_contribution_batch,
    validate_recomposition_sidecar,
)

# This is the only reduction provenance emitted by the current Experiment-2
# adapter contract.  The artifact validator intentionally accepts arbitrary
# nonempty provenance strings so it can remain a generic integrity layer; the
# numerical oracle is narrower and fails closed on an unqualified expression.
SUPPORTED_ACCUMULATION_ORDER = (
    "jax_einsum_BKE_BK_to_BE_float32_matmul_precision_default"
)


class RecompositionError(RuntimeError):
    """Base class for checkpoint-free recomposition failures."""


class RecompositionContractError(RecompositionError):
    """Inputs are valid artifacts but not a qualified numerical contract."""


class RecompositionNumericalError(RecompositionError):
    """A recomposition operation produced an invalid numerical value."""


class RecompositionMismatchError(RecompositionError):
    """An observed native stage does not meet its declared comparison gate."""


@dataclasses.dataclass(frozen=True, slots=True)
class RecompositionSidecar:
    """Decoded, validated numerical state from a recomposition sidecar."""

    sidecar_sha256: str
    accumulation_order: str
    epsilon: np.float32
    routed_rmsnorm_scale_bf16_bits: np.ndarray
    shared_rmsnorm_scale_bf16_bits: np.ndarray
    outer_ffn_rmsnorm_scale_bf16_bits: np.ndarray
    layer_scalar_bf16_bits: np.ndarray


@dataclasses.dataclass(frozen=True, slots=True)
class RecompositionStages:
    """Every material arithmetic/rounding stage in Gemma 4 recomposition."""

    sidecar_sha256: str
    weighted_contributions_f32: np.ndarray
    raw_routed_sum_f32: np.ndarray
    raw_routed_bf16_bits: np.ndarray
    routed_rms_unit_bf16_bits: np.ndarray
    routed_post_norm_bf16_bits: np.ndarray
    shared_raw_bf16_bits: np.ndarray
    shared_rms_unit_bf16_bits: np.ndarray
    shared_post_norm_bf16_bits: np.ndarray
    routed_plus_shared_bf16_bits: np.ndarray
    outer_ffn_rms_unit_bf16_bits: np.ndarray
    outer_ffn_post_norm_bf16_bits: np.ndarray
    residual_bf16_bits: np.ndarray | None
    residual_plus_ffn_bf16_bits: np.ndarray | None
    full_block_output_bf16_bits: np.ndarray | None

    @property
    def tokens(self) -> int:
        """Number of scientific tokens represented by the stages."""

        return int(self.raw_routed_sum_f32.shape[0])


@dataclasses.dataclass(frozen=True, slots=True)
class NumericalComparison:
    """Auditable diagnostics for one expected-versus-observed array."""

    stage: str
    shape: tuple[int, ...]
    representation: str
    element_count: int
    exact_element_count: int
    exact_fraction: float
    within_tolerance_count: int
    within_tolerance_fraction: float
    all_within_tolerance: bool
    max_absolute_error: float
    mean_absolute_error: float
    root_mean_squared_error: float
    max_relative_error: float
    worst_index: tuple[int, ...]
    expected_at_worst: float
    observed_at_worst: float
    atol: float
    rtol: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable diagnostics record."""

        return dataclasses.asdict(self)


def _readonly(array: np.ndarray) -> np.ndarray:
    """Return a contiguous, immutable array owned by the result object."""

    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _require_bf16_bits(
    name: str,
    value: Any,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint16:
        raise RecompositionContractError(
            f"{name} must contain exact uint16 BF16 bits, got {array.dtype}"
        )
    if array.shape != shape:
        raise RecompositionContractError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    decoded = bf16_bits_to_float32(array)
    if not np.isfinite(decoded).all():
        raise RecompositionContractError(f"{name} contains non-finite BF16 values")
    return array


def _round_f32_to_bf16_bits(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise RecompositionNumericalError(
            f"{name} must be FP32 before BF16 rounding, got {array.dtype}"
        )
    if not np.isfinite(array).all():
        raise RecompositionNumericalError(f"{name} is non-finite before BF16 rounding")
    bits = float32_to_bf16_bits(array)
    if not np.isfinite(bf16_bits_to_float32(bits)).all():
        raise RecompositionNumericalError(
            f"{name} overflowed to a non-finite BF16 value"
        )
    return bits


def _sidecar_tensor_bits(
    checked: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    tensor = checked["tensors"][name]
    bits = np.asarray(tensor["data"], dtype=np.uint16).reshape(shape)
    return _readonly(_require_bf16_bits(name, bits, shape=shape))


def decode_recomposition_sidecar(
    value: Mapping[str, Any],
) -> RecompositionSidecar:
    """Validate and decode a sidecar, rejecting unqualified arithmetic.

    The artifact validator proves schema and content identity.  This function
    additionally proves that the recorded accumulation expression is the one
    implemented by this oracle.
    """

    try:
        checked = validate_recomposition_sidecar(value)
    except (ContributionSchemaError, ContributionIntegrityError) as exc:
        raise RecompositionContractError(
            f"invalid recomposition sidecar: {exc}"
        ) from exc
    order = checked["accumulation_order"]
    if order != SUPPORTED_ACCUMULATION_ORDER:
        raise RecompositionContractError(
            "unsupported routed accumulation order "
            f"{order!r}; expected {SUPPORTED_ACCUMULATION_ORDER!r}"
        )
    epsilon = checked["rms_norm"]["epsilon"]
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or float(epsilon) != RMS_NORM_EPSILON
    ):
        # The artifact validator already freezes this, but keep the numerical
        # boundary explicit rather than trusting a future relaxed validator.
        raise RecompositionContractError(f"unsupported RMSNorm epsilon {epsilon!r}")
    return RecompositionSidecar(
        sidecar_sha256=checked["sidecar_sha256"],
        accumulation_order=order,
        epsilon=np.float32(epsilon),
        routed_rmsnorm_scale_bf16_bits=_sidecar_tensor_bits(
            checked, "routed_rmsnorm_scale", (HIDDEN_SIZE,)
        ),
        shared_rmsnorm_scale_bf16_bits=_sidecar_tensor_bits(
            checked, "shared_rmsnorm_scale", (HIDDEN_SIZE,)
        ),
        outer_ffn_rmsnorm_scale_bf16_bits=_sidecar_tensor_bits(
            checked, "outer_ffn_rmsnorm_scale", (HIDDEN_SIZE,)
        ),
        layer_scalar_bf16_bits=_sidecar_tensor_bits(checked, "layer_scalar", (1,)),
    )


def maxtext_rmsnorm_bf16(
    input_bf16_bits: Any,
    scale_bf16_bits: Any,
    *,
    epsilon: float = RMS_NORM_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply pinned MaxText RMSNorm and expose both BF16 round points.

    Returns ``(unit_normalized_bits, scaled_output_bits)``.  The first value is
    the cast at ``normalizations.py:73``; the second is the BF16 scale product
    at lines 90-92.
    """

    input_array = np.asarray(input_bf16_bits)
    scale_array = np.asarray(scale_bf16_bits)
    if input_array.dtype != np.uint16 or input_array.ndim < 1:
        raise RecompositionContractError(
            "RMSNorm input must be a rank-one-or-higher uint16 BF16-bit array"
        )
    if scale_array.dtype != np.uint16 or scale_array.shape != (input_array.shape[-1],):
        raise RecompositionContractError(
            "RMSNorm scale must be uint16 BF16 bits with shape "
            f"{(input_array.shape[-1],)}, got {scale_array.shape}/{scale_array.dtype}"
        )
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0
    ):
        raise RecompositionContractError("RMSNorm epsilon must be finite and positive")
    x = bf16_bits_to_float32(input_array)
    scale = bf16_bits_to_float32(scale_array)
    if not np.isfinite(x).all() or not np.isfinite(scale).all():
        raise RecompositionContractError(
            "RMSNorm input and scale must contain only finite values"
        )

    epsilon_f32 = np.float32(epsilon)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        squared = np.square(x, dtype=np.float32)
        mean_square = np.mean(squared, axis=-1, keepdims=True, dtype=np.float32)
        inverse_rms = np.reciprocal(
            np.sqrt(mean_square + epsilon_f32, dtype=np.float32),
            dtype=np.float32,
        )
        unit_f32 = np.multiply(x, inverse_rms, dtype=np.float32)
    if not np.isfinite(mean_square).all() or not np.isfinite(unit_f32).all():
        raise RecompositionNumericalError(
            "RMSNorm FP32 reduction or normalization produced a non-finite value"
        )

    unit_bits = _round_f32_to_bf16_bits("RMSNorm unit value", unit_f32)
    unit_bf16_f32 = bf16_bits_to_float32(unit_bits)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_f32 = np.multiply(unit_bf16_f32, scale, dtype=np.float32)
    output_bits = _round_f32_to_bf16_bits("RMSNorm scale product", scaled_f32)
    return _readonly(unit_bits), _readonly(output_bits)


def _bf16_add(name: str, left_bits: np.ndarray, right_bits: np.ndarray) -> np.ndarray:
    if left_bits.shape != right_bits.shape:
        raise RecompositionContractError(
            f"{name} operands must have identical shapes, got "
            f"{left_bits.shape} and {right_bits.shape}"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.add(
            bf16_bits_to_float32(left_bits),
            bf16_bits_to_float32(right_bits),
            dtype=np.float32,
        )
    return _readonly(_round_f32_to_bf16_bits(name, value))


def _bf16_scalar_multiply(
    name: str,
    value_bits: np.ndarray,
    scalar_bits: np.ndarray,
) -> np.ndarray:
    scalar = bf16_bits_to_float32(scalar_bits)
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.multiply(
            bf16_bits_to_float32(value_bits), scalar[0], dtype=np.float32
        )
    return _readonly(_round_f32_to_bf16_bits(name, value))


def recompose_gemma4(
    batch: ContributionBatch | Mapping[str, Any],
    sidecar: Mapping[str, Any],
    *,
    residual_bf16_bits: Any | None = None,
) -> RecompositionStages:
    """Recompose routed, shared, FFN, and optionally full-block Gemma stages.

    ``batch`` must be an exact Experiment-2 contribution batch.  Supplying a
    residual enables the final residual-addition and layer-scalar
    instrumentation; omitting it leaves all three full-block fields ``None``.
    """

    try:
        checked_batch = validate_contribution_batch(batch)
    except (ContributionSchemaError, ContributionIntegrityError) as exc:
        raise RecompositionContractError(f"invalid contribution batch: {exc}") from exc
    decoded_sidecar = decode_recomposition_sidecar(sidecar)
    tokens = checked_batch.tokens

    expert = bf16_bits_to_float32(checked_batch.expert_output)
    weights = bf16_bits_to_float32(checked_batch.router_weights)
    with np.errstate(over="ignore", invalid="ignore"):
        weighted = np.multiply(expert, weights[..., None], dtype=np.float32)
        # This is the host analogue of the pinned JAX "BKE,BK -> BE"
        # expression.  See the module-level reduction-order caveat.
        raw_routed_f32 = np.einsum(
            "nkh,nk->nh",
            expert,
            weights,
            dtype=np.float32,
            optimize=False,
        )
    if weighted.shape != (tokens, TOP_K_EXPERTS, HIDDEN_SIZE):
        raise RecompositionNumericalError(
            f"weighted contributions have unexpected shape {weighted.shape}"
        )
    if raw_routed_f32.shape != (tokens, HIDDEN_SIZE):
        raise RecompositionNumericalError(
            f"raw routed sum has unexpected shape {raw_routed_f32.shape}"
        )
    if not np.isfinite(weighted).all() or not np.isfinite(raw_routed_f32).all():
        raise RecompositionNumericalError(
            "FP32 router multiplication or routed reduction produced non-finite values"
        )

    raw_routed_bits = _readonly(
        _round_f32_to_bf16_bits("raw routed aggregate", raw_routed_f32)
    )
    routed_unit_bits, routed_post_norm_bits = maxtext_rmsnorm_bf16(
        raw_routed_bits,
        decoded_sidecar.routed_rmsnorm_scale_bf16_bits,
        epsilon=float(decoded_sidecar.epsilon),
    )
    shared_raw_bits = _readonly(np.asarray(checked_batch.shared_output).copy())
    shared_unit_bits, shared_post_norm_bits = maxtext_rmsnorm_bf16(
        shared_raw_bits,
        decoded_sidecar.shared_rmsnorm_scale_bf16_bits,
        epsilon=float(decoded_sidecar.epsilon),
    )
    routed_plus_shared_bits = _bf16_add(
        "routed-plus-shared addition",
        routed_post_norm_bits,
        shared_post_norm_bits,
    )
    outer_unit_bits, outer_post_norm_bits = maxtext_rmsnorm_bf16(
        routed_plus_shared_bits,
        decoded_sidecar.outer_ffn_rmsnorm_scale_bf16_bits,
        epsilon=float(decoded_sidecar.epsilon),
    )

    residual_bits: np.ndarray | None
    residual_plus_ffn_bits: np.ndarray | None
    full_block_bits: np.ndarray | None
    if residual_bf16_bits is None:
        residual_bits = None
        residual_plus_ffn_bits = None
        full_block_bits = None
    else:
        residual_bits = _readonly(
            _require_bf16_bits(
                "residual_bf16_bits",
                residual_bf16_bits,
                shape=(tokens, HIDDEN_SIZE),
            ).copy()
        )
        residual_plus_ffn_bits = _bf16_add(
            "FFN residual addition", outer_post_norm_bits, residual_bits
        )
        full_block_bits = _bf16_scalar_multiply(
            "layer-scalar multiplication",
            residual_plus_ffn_bits,
            decoded_sidecar.layer_scalar_bf16_bits,
        )

    return RecompositionStages(
        sidecar_sha256=decoded_sidecar.sidecar_sha256,
        weighted_contributions_f32=_readonly(weighted),
        raw_routed_sum_f32=_readonly(raw_routed_f32),
        raw_routed_bf16_bits=raw_routed_bits,
        routed_rms_unit_bf16_bits=routed_unit_bits,
        routed_post_norm_bf16_bits=routed_post_norm_bits,
        shared_raw_bf16_bits=shared_raw_bits,
        shared_rms_unit_bf16_bits=shared_unit_bits,
        shared_post_norm_bf16_bits=shared_post_norm_bits,
        routed_plus_shared_bf16_bits=routed_plus_shared_bits,
        outer_ffn_rms_unit_bf16_bits=outer_unit_bits,
        outer_ffn_post_norm_bf16_bits=outer_post_norm_bits,
        residual_bf16_bits=residual_bits,
        residual_plus_ffn_bf16_bits=residual_plus_ffn_bits,
        full_block_output_bf16_bits=full_block_bits,
    )


def compare_stage(
    expected: Any,
    observed: Any,
    *,
    stage: str,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> NumericalComparison:
    """Compare one native stage and return exact plus floating diagnostics.

    Arrays must either both be FP32 or both be exact ``uint16`` BF16 bits.
    This prevents an accidental implicit cast from weakening a parity gate.
    """

    if not isinstance(stage, str) or not stage.strip():
        raise RecompositionContractError("comparison stage must be nonempty")
    if (
        isinstance(atol, bool)
        or isinstance(rtol, bool)
        or not isinstance(atol, (int, float))
        or not isinstance(rtol, (int, float))
        or not math.isfinite(float(atol))
        or not math.isfinite(float(rtol))
        or float(atol) < 0
        or float(rtol) < 0
    ):
        raise RecompositionContractError(
            "comparison tolerances must be finite and nonnegative"
        )
    expected_array = np.asarray(expected)
    observed_array = np.asarray(observed)
    if expected_array.shape != observed_array.shape:
        raise RecompositionContractError(
            f"{stage} shapes differ: {expected_array.shape} vs {observed_array.shape}"
        )
    if expected_array.size == 0:
        raise RecompositionContractError(f"{stage} arrays must be nonempty")
    if expected_array.dtype != observed_array.dtype:
        raise RecompositionContractError(
            f"{stage} dtypes differ: {expected_array.dtype} vs {observed_array.dtype}"
        )
    if expected_array.dtype == np.uint16:
        representation = "bf16_bits"
        expected_float = bf16_bits_to_float32(expected_array)
        observed_float = bf16_bits_to_float32(observed_array)
        exact = expected_array == observed_array
    elif expected_array.dtype == np.float32:
        representation = "float32"
        expected_float = expected_array
        observed_float = observed_array
        exact = expected_array.view(np.uint32) == observed_array.view(np.uint32)
    else:
        raise RecompositionContractError(
            f"{stage} comparison requires float32 or uint16 BF16 bits, got "
            f"{expected_array.dtype}"
        )
    if not np.isfinite(expected_float).all() or not np.isfinite(observed_float).all():
        raise RecompositionContractError(
            f"{stage} comparison arrays must contain only finite values"
        )

    expected64 = expected_float.astype(np.float64)
    observed64 = observed_float.astype(np.float64)
    absolute = np.abs(observed64 - expected64)
    threshold = float(atol) + float(rtol) * np.abs(expected64)
    within = absolute <= threshold
    # A tiny finite floor keeps diagnostics valid strict JSON even when the
    # expected value is zero.  This quantity is diagnostic only; pass/fail is
    # determined by the explicit absolute/relative tolerance expression above.
    relative = absolute / np.maximum(
        np.abs(expected64), float(np.finfo(np.float32).tiny)
    )
    flat_worst = int(np.argmax(absolute))
    worst_index = tuple(
        int(index) for index in np.unravel_index(flat_worst, expected_array.shape)
    )
    elements = int(expected_array.size)
    exact_count = int(np.count_nonzero(exact))
    within_count = int(np.count_nonzero(within))
    return NumericalComparison(
        stage=stage,
        shape=tuple(int(item) for item in expected_array.shape),
        representation=representation,
        element_count=elements,
        exact_element_count=exact_count,
        exact_fraction=exact_count / elements,
        within_tolerance_count=within_count,
        within_tolerance_fraction=within_count / elements,
        all_within_tolerance=within_count == elements,
        max_absolute_error=float(np.max(absolute)),
        mean_absolute_error=float(np.mean(absolute, dtype=np.float64)),
        root_mean_squared_error=float(
            np.sqrt(np.mean(np.square(absolute), dtype=np.float64))
        ),
        max_relative_error=float(np.max(relative)),
        worst_index=worst_index,
        expected_at_worst=float(expected_float[worst_index]),
        observed_at_worst=float(observed_float[worst_index]),
        atol=float(atol),
        rtol=float(rtol),
    )


def require_stage_match(
    expected: Any,
    observed: Any,
    *,
    stage: str,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> NumericalComparison:
    """Return diagnostics or fail the parity gate with an auditable summary."""

    diagnostics = compare_stage(
        expected,
        observed,
        stage=stage,
        atol=atol,
        rtol=rtol,
    )
    if not diagnostics.all_within_tolerance:
        raise RecompositionMismatchError(
            f"{stage} parity failed: "
            f"{diagnostics.within_tolerance_count}/"
            f"{diagnostics.element_count} elements within tolerance; "
            f"max_abs={diagnostics.max_absolute_error:.9g} at "
            f"{diagnostics.worst_index}; expected="
            f"{diagnostics.expected_at_worst:.9g}, observed="
            f"{diagnostics.observed_at_worst:.9g}, "
            f"atol={diagnostics.atol:.9g}, rtol={diagnostics.rtol:.9g}"
        )
    return diagnostics
