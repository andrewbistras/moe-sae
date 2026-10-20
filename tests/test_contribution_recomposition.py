"""Checkpoint-free parity tests for Gemma Experiment-2 recomposition."""

from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pytest

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    RMS_NORM_EPSILON,
    TOP_K_EXPERTS,
    ContributionBatch,
    bf16_bits_to_float32,
    build_recomposition_sidecar,
    float32_to_bf16_bits,
)
from moe_sae.contribution_recomposition import (
    SUPPORTED_ACCUMULATION_ORDER,
    RecompositionContractError,
    RecompositionMismatchError,
    compare_stage,
    decode_recomposition_sidecar,
    maxtext_rmsnorm_bf16,
    recompose_gemma4,
    require_stage_match,
)

CHECKPOINT_SHA = "1" * 64
FORK_COMMIT = "2" * 40


def _bits(value: object) -> np.ndarray:
    return float32_to_bf16_bits(np.asarray(value, dtype=np.float32))


def _sidecar(
    *,
    routed_scale: np.ndarray | None = None,
    shared_scale: np.ndarray | None = None,
    outer_scale: np.ndarray | None = None,
    layer_scalar: float = 1.0,
    accumulation_order: str = SUPPORTED_ACCUMULATION_ORDER,
) -> dict:
    ones = _bits(np.ones(HIDDEN_SIZE, dtype=np.float32))
    return build_recomposition_sidecar(
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_fork_commit=FORK_COMMIT,
        source_fingerprints={
            "checkpoint": CHECKPOINT_SHA,
            "fork_tree": "3" * 64,
        },
        routed_rmsnorm_scale=ones if routed_scale is None else routed_scale,
        shared_rmsnorm_scale=ones if shared_scale is None else shared_scale,
        outer_ffn_rmsnorm_scale=ones if outer_scale is None else outer_scale,
        layer_scalar=_bits([layer_scalar]),
        accumulation_order=accumulation_order,
    )


def _batch(
    *,
    tokens: int = 1,
    expert_output: np.ndarray | None = None,
    shared_output: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> ContributionBatch:
    if expert_output is None:
        expert_output = np.zeros((tokens, TOP_K_EXPERTS, HIDDEN_SIZE), dtype=np.float32)
    if shared_output is None:
        shared_output = np.zeros((tokens, HIDDEN_SIZE), dtype=np.float32)
    if weights is None:
        weights = np.full(
            (tokens, TOP_K_EXPERTS), 1.0 / TOP_K_EXPERTS, dtype=np.float32
        )
    return ContributionBatch(
        expert_output=_bits(expert_output),
        expert_ids=np.tile(np.arange(TOP_K_EXPERTS, dtype=np.uint8), (tokens, 1)),
        router_weights=_bits(weights),
        shared_output=_bits(shared_output),
        sequence_id=np.arange(100, 100 + tokens, dtype=np.int64),
        position=np.zeros(tokens, dtype=np.uint16),
    )


def _decoded(bits: np.ndarray) -> np.ndarray:
    return bf16_bits_to_float32(np.asarray(bits, dtype=np.uint16))


def test_hand_computed_constant_recomposition_and_full_block() -> None:
    expert = np.empty((1, TOP_K_EXPERTS, HIDDEN_SIZE), dtype=np.float32)
    for slot in range(TOP_K_EXPERTS):
        expert[:, slot, :] = float(slot + 1)
    shared = np.full((1, HIDDEN_SIZE), 2.0, dtype=np.float32)
    residual = _bits(np.full((1, HIDDEN_SIZE), 3.0, dtype=np.float32))

    result = recompose_gemma4(
        _batch(expert_output=expert, shared_output=shared),
        _sidecar(layer_scalar=0.5),
        residual_bf16_bits=residual,
    )

    for slot in range(TOP_K_EXPERTS):
        np.testing.assert_array_equal(
            result.weighted_contributions_f32[:, slot, :],
            np.float32((slot + 1) / TOP_K_EXPERTS),
        )
    np.testing.assert_array_equal(result.raw_routed_sum_f32, np.float32(4.5))
    np.testing.assert_array_equal(
        result.raw_routed_bf16_bits, _bits([[4.5] * HIDDEN_SIZE])
    )
    np.testing.assert_array_equal(
        result.routed_rms_unit_bf16_bits, _bits([[1.0] * HIDDEN_SIZE])
    )
    np.testing.assert_array_equal(
        result.routed_post_norm_bf16_bits, _bits([[1.0] * HIDDEN_SIZE])
    )
    np.testing.assert_array_equal(
        result.shared_post_norm_bf16_bits, _bits([[1.0] * HIDDEN_SIZE])
    )
    np.testing.assert_array_equal(
        result.routed_plus_shared_bf16_bits, _bits([[2.0] * HIDDEN_SIZE])
    )
    np.testing.assert_array_equal(
        result.outer_ffn_post_norm_bf16_bits, _bits([[1.0] * HIDDEN_SIZE])
    )
    np.testing.assert_array_equal(
        result.residual_plus_ffn_bf16_bits, _bits([[4.0] * HIDDEN_SIZE])
    )
    np.testing.assert_array_equal(
        result.full_block_output_bf16_bits, _bits([[2.0] * HIDDEN_SIZE])
    )
    assert result.tokens == 1
    assert not result.full_block_output_bf16_bits.flags.writeable


def test_without_residual_stops_after_outer_ffn_norm() -> None:
    result = recompose_gemma4(_batch(), _sidecar())

    assert result.residual_bf16_bits is None
    assert result.residual_plus_ffn_bf16_bits is None
    assert result.full_block_output_bf16_bits is None
    assert result.outer_ffn_post_norm_bf16_bits.shape == (1, HIDDEN_SIZE)


def test_bf16_operands_round_trip_bit_exactly_before_arithmetic() -> None:
    expert = np.linspace(
        -3.01,
        3.01,
        TOP_K_EXPERTS * HIDDEN_SIZE,
        dtype=np.float32,
    ).reshape(1, TOP_K_EXPERTS, HIDDEN_SIZE)
    shared = np.linspace(-1.001, 1.001, HIDDEN_SIZE, dtype=np.float32)[None, :]
    batch = _batch(expert_output=expert, shared_output=shared)

    result = recompose_gemma4(batch, _sidecar())
    decoded_expert = _decoded(batch.expert_output)
    decoded_weights = _decoded(batch.router_weights)

    np.testing.assert_array_equal(result.shared_raw_bf16_bits, batch.shared_output)
    np.testing.assert_array_equal(
        result.weighted_contributions_f32,
        decoded_expert * decoded_weights[..., None],
    )
    assert result.sidecar_sha256 == _sidecar()["sidecar_sha256"]


def test_rmsnorm_exposes_unit_round_before_bf16_scale_product() -> None:
    x = _bits([[3.0, 4.0]])
    scale = _bits([2.0, 0.5])
    unit_bits, output_bits = maxtext_rmsnorm_bf16(x, scale, epsilon=RMS_NORM_EPSILON)

    rms = np.sqrt(np.float32((9.0 + 16.0) / 2.0 + RMS_NORM_EPSILON))
    expected_unit = _bits(np.asarray([[3.0 / rms, 4.0 / rms]], dtype=np.float32))
    expected_output = _bits(
        _decoded(expected_unit) * np.asarray([2.0, 0.5], dtype=np.float32)
    )
    np.testing.assert_array_equal(unit_bits, expected_unit)
    np.testing.assert_array_equal(output_bits, expected_output)


def test_numpy_oracle_matches_lightweight_jax_reference() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    rng = np.random.default_rng(123)
    expert = rng.normal(0.0, 0.2, size=(1, TOP_K_EXPERTS, HIDDEN_SIZE)).astype(
        np.float32
    )
    shared = rng.normal(0.0, 0.2, size=(1, HIDDEN_SIZE)).astype(np.float32)
    weights = np.asarray(
        [[0.25, 0.1875, 0.15625, 0.125, 0.09375, 0.078125, 0.0625, 0.046875]],
        dtype=np.float32,
    )
    routed_scale_bits = _bits(np.linspace(0.75, 1.25, HIDDEN_SIZE, dtype=np.float32))
    shared_scale_bits = _bits(np.linspace(1.25, 0.75, HIDDEN_SIZE, dtype=np.float32))
    outer_scale_bits = _bits(np.linspace(0.875, 1.125, HIDDEN_SIZE, dtype=np.float32))
    residual_bits = _bits(
        rng.normal(0.0, 0.2, size=(1, HIDDEN_SIZE)).astype(np.float32)
    )
    batch = _batch(expert_output=expert, shared_output=shared, weights=weights)
    sidecar = _sidecar(
        routed_scale=routed_scale_bits,
        shared_scale=shared_scale_bits,
        outer_scale=outer_scale_bits,
        layer_scalar=0.875,
    )
    result = recompose_gemma4(batch, sidecar, residual_bf16_bits=residual_bits)

    expert_jax = jnp.asarray(_decoded(batch.expert_output), dtype=jnp.bfloat16)
    weights_jax = jnp.asarray(_decoded(batch.router_weights), dtype=jnp.bfloat16)
    shared_jax = jnp.asarray(_decoded(batch.shared_output), dtype=jnp.bfloat16)

    def jax_norm(x, scale_bits):
        x_f32 = jnp.asarray(x, jnp.float32)
        mean2 = jnp.mean(jax.lax.square(x_f32), axis=-1, keepdims=True)
        unit = jnp.asarray(
            x_f32 * jax.lax.rsqrt(mean2 + RMS_NORM_EPSILON), jnp.bfloat16
        )
        scale = jnp.asarray(_decoded(scale_bits), dtype=jnp.bfloat16)
        return jnp.einsum("...k,k->...k", unit, scale)

    raw_f32 = jnp.einsum(
        "nkh,nk->nh",
        expert_jax.astype(jnp.float32),
        weights_jax.astype(jnp.float32),
    )
    raw_bf16 = raw_f32.astype(jnp.bfloat16)
    routed = jax_norm(raw_bf16, routed_scale_bits)
    normalized_shared = jax_norm(shared_jax, shared_scale_bits)
    branch_sum = routed + normalized_shared
    outer = jax_norm(branch_sum, outer_scale_bits)
    residual_sum = outer + jnp.asarray(_decoded(residual_bits), dtype=jnp.bfloat16)
    full_block = residual_sum * jnp.asarray(0.875, dtype=jnp.bfloat16)

    # The pre-cast dot is diagnostic, not a cross-target bit-identity promise.
    assert compare_stage(
        result.raw_routed_sum_f32,
        np.asarray(raw_f32, dtype=np.float32),
        stage="raw_routed_f32",
        atol=2e-7,
    ).all_within_tolerance
    np.testing.assert_array_equal(
        result.raw_routed_bf16_bits,
        _bits(np.asarray(raw_bf16, dtype=np.float32)),
    )
    np.testing.assert_array_equal(
        result.routed_post_norm_bf16_bits,
        _bits(np.asarray(routed, dtype=np.float32)),
    )
    np.testing.assert_array_equal(
        result.shared_post_norm_bf16_bits,
        _bits(np.asarray(normalized_shared, dtype=np.float32)),
    )
    np.testing.assert_array_equal(
        result.routed_plus_shared_bf16_bits,
        _bits(np.asarray(branch_sum, dtype=np.float32)),
    )
    np.testing.assert_array_equal(
        result.outer_ffn_post_norm_bf16_bits,
        _bits(np.asarray(outer, dtype=np.float32)),
    )
    np.testing.assert_array_equal(
        result.residual_plus_ffn_bf16_bits,
        _bits(np.asarray(residual_sum, dtype=np.float32)),
    )
    np.testing.assert_array_equal(
        result.full_block_output_bf16_bits,
        _bits(np.asarray(full_block, dtype=np.float32)),
    )


def test_sidecar_validation_rejects_tampering_and_unknown_reduction() -> None:
    sidecar = _sidecar()
    tampered = copy.deepcopy(sidecar)
    tampered["tensors"]["layer_scalar"]["data"][0] ^= 1
    with pytest.raises(RecompositionContractError, match="SHA-256 mismatch"):
        decode_recomposition_sidecar(tampered)

    unsupported = _sidecar(accumulation_order="left_to_right_guess")
    with pytest.raises(RecompositionContractError, match="unsupported routed"):
        recompose_gemma4(_batch(), unsupported)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda batch: dataclasses.replace(
                batch, expert_output=batch.expert_output.astype(np.float32)
            ),
            "wire dtype",
        ),
        (
            lambda batch: dataclasses.replace(
                batch, shared_output=batch.shared_output[:, :-1]
            ),
            "shape",
        ),
        (
            lambda batch: dataclasses.replace(
                batch,
                expert_output=np.where(
                    np.indices(batch.expert_output.shape)[2] == 0,
                    np.uint16(0x7F80),
                    batch.expert_output,
                ).astype(np.uint16),
            ),
            "non-finite",
        ),
    ],
)
def test_invalid_batch_dtype_shape_and_nonfinite_fail_closed(change, message) -> None:
    with pytest.raises(RecompositionContractError, match=message):
        recompose_gemma4(change(_batch()), _sidecar())


def test_invalid_residual_dtype_shape_and_nonfinite_fail_closed() -> None:
    valid = _bits(np.zeros((1, HIDDEN_SIZE), dtype=np.float32))
    with pytest.raises(RecompositionContractError, match="uint16"):
        recompose_gemma4(
            _batch(), _sidecar(), residual_bf16_bits=valid.astype(np.float32)
        )
    with pytest.raises(RecompositionContractError, match="shape"):
        recompose_gemma4(_batch(), _sidecar(), residual_bf16_bits=valid[:, :-1])
    nonfinite = valid.copy()
    nonfinite[0, 0] = np.uint16(0x7F80)
    with pytest.raises(RecompositionContractError, match="non-finite"):
        recompose_gemma4(_batch(), _sidecar(), residual_bf16_bits=nonfinite)


def test_comparison_diagnostics_report_exactness_and_enforce_gate() -> None:
    expected = _bits([1.0, 2.0])
    observed = _bits([1.0, 2.015625])
    diagnostics = compare_stage(
        expected,
        observed,
        stage="outer_ffn",
        atol=0.02,
    )

    assert diagnostics.representation == "bf16_bits"
    assert diagnostics.element_count == 2
    assert diagnostics.exact_element_count == 1
    assert diagnostics.exact_fraction == 0.5
    assert diagnostics.all_within_tolerance
    assert diagnostics.max_absolute_error == pytest.approx(0.015625)
    assert diagnostics.worst_index == (1,)
    assert diagnostics.to_dict()["stage"] == "outer_ffn"
    json.dumps(diagnostics.to_dict(), allow_nan=False)

    with pytest.raises(RecompositionMismatchError, match="parity failed"):
        require_stage_match(
            expected,
            observed,
            stage="outer_ffn",
        )


def test_comparison_rejects_implicit_cast_shape_nonfinite_and_bad_tolerance() -> None:
    expected = np.ones((2,), dtype=np.float32)
    with pytest.raises(RecompositionContractError, match="dtypes differ"):
        compare_stage(
            expected,
            np.ones((2,), dtype=np.float64),
            stage="raw",
        )
    with pytest.raises(RecompositionContractError, match="shapes differ"):
        compare_stage(
            expected,
            np.ones((1,), dtype=np.float32),
            stage="raw",
        )
    nonfinite = expected.copy()
    nonfinite[0] = np.inf
    with pytest.raises(RecompositionContractError, match="finite"):
        compare_stage(nonfinite, nonfinite, stage="raw")
    with pytest.raises(RecompositionContractError, match="nonnegative"):
        compare_stage(expected, expected, stage="raw", atol=-1.0)

    zero_reference = compare_stage(
        np.zeros((1,), dtype=np.float32),
        np.ones((1,), dtype=np.float32),
        stage="zero_reference",
    )
    assert np.isfinite(zero_reference.max_relative_error)
    json.dumps(zero_reference.to_dict(), allow_nan=False)
