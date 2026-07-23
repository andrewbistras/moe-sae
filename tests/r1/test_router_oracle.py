from __future__ import annotations

import math

import jax
from jax.experimental import checkify
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from moe_sae.r1.router_oracle import (
    AmbiguousRouterSelection,
    PINNED_R1_ROUTER_GEOMETRY,
    RouterGeometryV1,
    RouterOracleError,
    canonicalize_numpy_router_result,
    jax_router_oracle,
    jax_router_oracle_checked,
    numpy_router_oracle,
    validate_jax_router_result,
    validate_pinned_r1_geometry,
)


BF16 = ml_dtypes.bfloat16


def _scalar_sigmoid(value: np.float32) -> np.float32:
    scalar = float(value)
    if scalar >= 0.0:
        return np.float32(1.0 / (1.0 + math.exp(-scalar)))
    exponential = math.exp(scalar)
    return np.float32(exponential / (1.0 + exponential))


def _independent_scalar_oracle(
    hidden: np.ndarray,
    gate: np.ndarray,
    bias: np.ndarray,
    geometry: RouterGeometryV1,
) -> tuple[np.ndarray, np.ndarray]:
    """Loop-based expression independent of either implementation under test."""

    flat_hidden = hidden.reshape(-1, geometry.hidden_dim)
    all_ids: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for token in flat_hidden:
        logits = np.empty(geometry.expert_count, dtype=np.float32)
        for expert in range(geometry.expert_count):
            accumulator = np.float32(0.0)
            for feature in range(geometry.hidden_dim):
                product = np.float32(token[feature]) * np.float32(gate[feature, expert])
                accumulator = np.float32(accumulator + product)
            logits[expert] = accumulator
        unbiased = np.asarray(
            [_scalar_sigmoid(value) for value in logits], dtype=np.float32
        )
        selection = np.asarray(unbiased + bias, dtype=np.float32)
        grouped = selection.reshape(geometry.group_count, geometry.group_size)
        group_scores = np.asarray(
            [
                np.sum(
                    np.sort(group)[-geometry.group_score_top_k :],
                    dtype=np.float32,
                )
                for group in grouped
            ],
            dtype=np.float32,
        )
        sorted_groups = np.sort(group_scores)
        if geometry.selected_group_count < geometry.group_count:
            assert (
                sorted_groups[-geometry.selected_group_count]
                != sorted_groups[-(geometry.selected_group_count + 1)]
            )
        selected_groups = np.argsort(group_scores)[-geometry.selected_group_count :]
        candidate_ids = np.asarray(
            [
                group * geometry.group_size + offset
                for group in selected_groups
                for offset in range(geometry.group_size)
            ],
            dtype=np.int64,
        )
        candidate_scores = selection[candidate_ids]
        sorted_candidates = np.sort(candidate_scores)
        if geometry.experts_per_token < candidate_ids.size:
            assert (
                sorted_candidates[-geometry.experts_per_token]
                != sorted_candidates[-(geometry.experts_per_token + 1)]
            )
        selected_ids = candidate_ids[
            np.argsort(candidate_scores)[-geometry.experts_per_token :]
        ]
        selected_unbiased = unbiased[selected_ids]
        denominator = np.sum(selected_unbiased, dtype=np.float32)
        weights = np.asarray(
            selected_unbiased
            / denominator
            * np.float32(geometry.routed_scaling_factor),
            dtype=np.float32,
        )
        canonical = np.argsort(selected_ids)
        all_ids.append(selected_ids[canonical].astype(np.uint16))
        all_weights.append(weights[canonical])
    result_shape = (*hidden.shape[:-1], geometry.experts_per_token)
    return (
        np.stack(all_ids).reshape(result_shape),
        np.stack(all_weights).reshape(result_shape),
    )


def _random_fixture(
    seed: int,
) -> tuple[RouterGeometryV1, np.ndarray, np.ndarray, np.ndarray]:
    geometry = RouterGeometryV1(
        hidden_dim=8,
        expert_count=16,
        group_count=4,
        selected_group_count=2,
        experts_per_token=3,
    )
    rng = np.random.default_rng(seed)
    hidden = np.asarray(rng.normal(size=(2, 3, 8)), dtype=BF16)
    gate = np.asarray(rng.normal(scale=0.3, size=(8, 16)), dtype=BF16)
    # A continuous FP32 bias makes exact selection-boundary ties vanishingly
    # unlikely while exercising the bias-only selection path.
    bias = np.asarray(rng.normal(scale=0.03, size=16), dtype=np.float32)
    return geometry, hidden, gate, bias


def _zero_projection_fixture(
    geometry: RouterGeometryV1,
    bias: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((geometry.hidden_dim,), dtype=BF16),
        np.zeros((geometry.hidden_dim, geometry.expert_count), dtype=BF16),
        np.asarray(bias, dtype=np.float32),
    )


def test_pinned_geometry_is_frozen_and_uint16_compatible() -> None:
    validate_pinned_r1_geometry(PINNED_R1_ROUTER_GEOMETRY)
    assert PINNED_R1_ROUTER_GEOMETRY == RouterGeometryV1(
        hidden_dim=7168,
        expert_count=256,
        group_count=8,
        selected_group_count=4,
        experts_per_token=8,
        group_score_top_k=2,
        routed_scaling_factor=2.5,
    )
    with pytest.raises(RouterOracleError, match="frozen"):
        validate_pinned_r1_geometry(
            RouterGeometryV1(
                hidden_dim=8,
                expert_count=256,
                group_count=8,
                selected_group_count=4,
                experts_per_token=8,
            )
        )
    with pytest.raises(RouterOracleError, match="uint16"):
        RouterGeometryV1(
            hidden_dim=1,
            expert_count=131072,
            group_count=8,
            selected_group_count=4,
            experts_per_token=8,
        )


@pytest.mark.parametrize("seed", range(5))
def test_numpy_jax_and_independent_scalar_oracles_agree(seed: int) -> None:
    geometry, hidden, gate, bias = _random_fixture(seed)
    numpy_result = numpy_router_oracle(hidden, gate, bias, geometry=geometry)
    scalar_ids, scalar_weights = _independent_scalar_oracle(
        hidden, gate, bias, geometry
    )
    np.testing.assert_array_equal(numpy_result.expert_ids, scalar_ids)
    np.testing.assert_allclose(
        numpy_result.router_weights,
        scalar_weights,
        rtol=4e-6,
        atol=4e-6,
    )

    jax_ids, jax_weights = jax_router_oracle(
        jnp.asarray(hidden),
        jnp.asarray(gate),
        jnp.asarray(bias),
        geometry=geometry,
    )
    validated = validate_jax_router_result(
        jax_ids,
        jax_weights,
        geometry,
        expected_leading_shape=hidden.shape[:-1],
    )
    np.testing.assert_array_equal(validated.expert_ids, numpy_result.expert_ids)
    np.testing.assert_allclose(
        validated.router_weights,
        numpy_result.router_weights,
        rtol=3e-6,
        atol=3e-6,
    )


def test_group_restriction_precedes_global_expert_selection() -> None:
    geometry = RouterGeometryV1(
        hidden_dim=1,
        expert_count=8,
        group_count=2,
        selected_group_count=1,
        experts_per_token=2,
    )
    # Expert 4 is individually stronger than expert 1, but group 1 loses
    # because its second-best member is weak. Only IDs from group 0 are legal.
    hidden, gate, bias = _zero_projection_fixture(
        geometry,
        [10.0, 9.0, 0.0, -1.0, 9.5, -100.0, -101.0, -102.0],
    )
    result = numpy_router_oracle(hidden, gate, bias, geometry=geometry)
    np.testing.assert_array_equal(result.expert_ids, np.asarray([0, 1], np.uint16))
    np.testing.assert_array_equal(
        result.router_weights, np.asarray([1.25, 1.25], np.float32)
    )


def test_correction_bias_changes_ids_but_never_emitted_score_source() -> None:
    geometry = RouterGeometryV1(
        hidden_dim=2,
        expert_count=8,
        group_count=2,
        selected_group_count=1,
        experts_per_token=2,
    )
    hidden = np.asarray([1.0, 0.0], dtype=BF16)
    gate = np.asarray(
        [
            [-1.0, -0.8, -0.6, -0.4, 0.1, 0.2, 0.3, 0.4],
            [0.0] * 8,
        ],
        dtype=BF16,
    )
    no_bias = np.zeros(8, dtype=np.float32)
    shifted_bias = np.asarray([2.0] * 4 + [0.0] * 4, dtype=np.float32)
    baseline = numpy_router_oracle(hidden, gate, no_bias, geometry=geometry)
    shifted = numpy_router_oracle(hidden, gate, shifted_bias, geometry=geometry)
    assert not np.array_equal(baseline.expert_ids, shifted.expert_ids)

    logits = hidden.astype(np.float32) @ gate.astype(np.float32)
    unbiased = np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float32)
    selected = shifted.expert_ids.astype(np.int64)
    expected = unbiased[selected]
    expected = np.asarray(
        expected / np.sum(expected, dtype=np.float32) * np.float32(2.5),
        dtype=np.float32,
    )
    np.testing.assert_allclose(shifted.router_weights, expected, rtol=0.0, atol=1e-7)

    # A global correction-bias translation preserves all selected sets. It
    # must therefore leave emitted weights bitwise unchanged.
    translated = numpy_router_oracle(
        hidden,
        gate,
        shifted_bias + np.float32(100.0),
        geometry=geometry,
    )
    np.testing.assert_array_equal(translated.expert_ids, shifted.expert_ids)
    np.testing.assert_array_equal(translated.router_weights, shifted.router_weights)


def test_router_projection_is_fp32_not_bf16() -> None:
    geometry = RouterGeometryV1(
        hidden_dim=2,
        expert_count=8,
        group_count=2,
        selected_group_count=1,
        experts_per_token=2,
    )
    hidden = np.asarray([1.0, 0.1], dtype=BF16)
    gate = np.zeros((2, 8), dtype=BF16)
    gate[0] = np.asarray([-2, -3, -4, -5, -1, 0.5, 0.5, 1], dtype=BF16)
    gate[1, 5] = BF16(0.01)
    bias = np.zeros(8, dtype=np.float32)
    bias[6] = np.float32(0.0001)

    fp32_result = numpy_router_oracle(hidden, gate, bias, geometry=geometry)
    np.testing.assert_array_equal(
        fp32_result.expert_ids, np.asarray([5, 7], dtype=np.uint16)
    )

    fp32_logits = hidden.astype(np.float32) @ gate.astype(np.float32)
    rounded_logits = np.asarray(fp32_logits, dtype=BF16).astype(np.float32)
    rounded_scores = np.asarray(
        1.0 / (1.0 + np.exp(-rounded_logits)) + bias,
        dtype=np.float32,
    )
    chosen_group = int(
        np.argmax(
            np.sum(
                np.sort(rounded_scores.reshape(2, 4), axis=-1)[:, -2:],
                axis=-1,
            )
        )
    )
    candidates = np.arange(chosen_group * 4, chosen_group * 4 + 4)
    rounded_ids = np.sort(candidates[np.argsort(rounded_scores[candidates])[-2:]])
    np.testing.assert_array_equal(rounded_ids, np.asarray([6, 7]))


def test_weights_are_nonnegative_and_sum_to_routed_scale() -> None:
    geometry, hidden, gate, bias = _random_fixture(71)
    result = numpy_router_oracle(hidden, gate, bias, geometry=geometry)
    assert result.router_weights.dtype == np.float32
    assert np.all(result.router_weights >= 0.0)
    np.testing.assert_allclose(
        np.sum(result.router_weights, axis=-1, dtype=np.float32),
        np.float32(2.5),
        rtol=3e-6,
        atol=3e-6,
    )


def test_canonicalization_sorts_paired_records_and_rejects_bad_results() -> None:
    geometry = RouterGeometryV1(
        hidden_dim=1,
        expert_count=8,
        group_count=2,
        selected_group_count=1,
        experts_per_token=2,
    )
    canonical = canonicalize_numpy_router_result(
        np.asarray([[7, 2]], dtype=np.int32),
        np.asarray([[1.0, 1.5]], dtype=np.float32),
        geometry,
        expected_leading_shape=(1,),
    )
    np.testing.assert_array_equal(canonical.expert_ids, [[2, 7]])
    np.testing.assert_array_equal(canonical.router_weights, [[1.5, 1.0]])
    assert canonical.expert_ids.dtype == np.uint16
    assert not canonical.expert_ids.flags.writeable
    assert not canonical.router_weights.flags.writeable

    with pytest.raises(RouterOracleError, match="duplicate"):
        canonicalize_numpy_router_result(
            np.asarray([2, 2], dtype=np.uint16),
            np.asarray([1.0, 1.5], dtype=np.float32),
            geometry,
        )
    with pytest.raises(RouterOracleError, match="out-of-range"):
        canonicalize_numpy_router_result(
            np.asarray([2, 8], dtype=np.int32),
            np.asarray([1.0, 1.5], dtype=np.float32),
            geometry,
        )
    with pytest.raises(RouterOracleError, match="sum to 2.5"):
        canonicalize_numpy_router_result(
            np.asarray([2, 7], dtype=np.int32),
            np.asarray([0.5, 0.5], dtype=np.float32),
            geometry,
        )


def test_boundary_ties_fail_but_set_invariant_ties_are_accepted() -> None:
    geometry = RouterGeometryV1(
        hidden_dim=1,
        expert_count=16,
        group_count=4,
        selected_group_count=2,
        experts_per_token=2,
    )
    group_boundary = [
        4,
        3,
        0,
        -1,
        3,
        2,
        0,
        -1,
        3,
        2,
        0,
        -1,
        0,
        -1,
        -2,
        -3,
    ]
    hidden, gate, bias = _zero_projection_fixture(geometry, group_boundary)
    with pytest.raises(AmbiguousRouterSelection, match="group selection"):
        numpy_router_oracle(hidden, gate, bias, geometry=geometry)

    expert_boundary = [
        4,
        3,
        0,
        -1,
        3,
        2,
        0,
        -1,
        0,
        -1,
        -2,
        -3,
        -1,
        -2,
        -3,
        -4,
    ]
    hidden, gate, bias = _zero_projection_fixture(geometry, expert_boundary)
    with pytest.raises(AmbiguousRouterSelection, match="expert selection"):
        numpy_router_oracle(hidden, gate, bias, geometry=geometry)
    with pytest.raises(AmbiguousRouterSelection, match="boundary tie"):
        jax_router_oracle(
            jnp.asarray(hidden),
            jnp.asarray(gate),
            jnp.asarray(bias),
            geometry=geometry,
        )

    harmless = [
        4,
        3,
        0,
        -1,
        4,
        3,
        0,
        -1,
        1,
        0,
        -1,
        -2,
        0,
        -1,
        -2,
        -3,
    ]
    hidden, gate, bias = _zero_projection_fixture(geometry, harmless)
    accepted = numpy_router_oracle(hidden, gate, bias, geometry=geometry)
    # The tie is wholly within the selected group and expert sets; canonical
    # output makes slot order irrelevant.
    np.testing.assert_array_equal(accepted.expert_ids, np.asarray([0, 4], np.uint16))
    jax_ids, jax_weights = jax_router_oracle(
        jnp.asarray(hidden),
        jnp.asarray(gate),
        jnp.asarray(bias),
        geometry=geometry,
    )
    np.testing.assert_array_equal(np.asarray(jax_ids), accepted.expert_ids)
    np.testing.assert_allclose(np.asarray(jax_weights), accepted.router_weights)

    with pytest.raises(AmbiguousRouterSelection, match="boundary tie"):
        jax_router_oracle(
            jnp.asarray(_zero_projection_fixture(geometry, group_boundary)[0]),
            jnp.asarray(_zero_projection_fixture(geometry, group_boundary)[1]),
            jnp.asarray(_zero_projection_fixture(geometry, group_boundary)[2]),
            geometry=geometry,
        )


@pytest.mark.parametrize("invalid", ["hidden", "gate", "bias"])
def test_nonfinite_inputs_fail_in_numpy_and_checkified_jax(invalid: str) -> None:
    geometry, hidden, gate, bias = _random_fixture(83)
    hidden = hidden.copy()
    gate = gate.copy()
    bias = bias.copy()
    if invalid == "hidden":
        hidden[0, 0, 0] = BF16(np.nan)
    elif invalid == "gate":
        gate[0, 0] = BF16(np.inf)
    else:
        bias[0] = np.float32(-np.inf)
    with pytest.raises(RouterOracleError, match="non-finite"):
        numpy_router_oracle(hidden, gate, bias, geometry=geometry)

    error, _ = jax.jit(
        lambda h, g, b: jax_router_oracle_checked(h, g, b, geometry=geometry)
    )(jnp.asarray(hidden), jnp.asarray(gate), jnp.asarray(bias))
    with pytest.raises(checkify.JaxRuntimeError, match="non-finite"):
        error.throw()


def test_shape_and_dtype_contracts_fail_before_router_math() -> None:
    geometry, hidden, gate, bias = _random_fixture(91)
    with pytest.raises(RouterOracleError, match="hidden must be BF16"):
        numpy_router_oracle(hidden.astype(np.float32), gate, bias, geometry=geometry)
    with pytest.raises(RouterOracleError, match="gate_kernel must be BF16"):
        numpy_router_oracle(hidden, gate.astype(np.float32), bias, geometry=geometry)
    with pytest.raises(RouterOracleError, match="correction_bias must be FP32"):
        numpy_router_oracle(hidden, gate, bias.astype(np.float64), geometry=geometry)
    with pytest.raises(RouterOracleError, match="gate_kernel must have shape"):
        numpy_router_oracle(hidden, gate[:, :-1], bias, geometry=geometry)
    with pytest.raises(RouterOracleError, match="correction_bias must have shape"):
        numpy_router_oracle(hidden, gate, bias[:-1], geometry=geometry)
    with pytest.raises(RouterOracleError, match="at least one token"):
        numpy_router_oracle(hidden[:0], gate, bias, geometry=geometry)
    with pytest.raises(RouterOracleError, match="hidden must be BF16"):
        jax_router_oracle_checked(
            jnp.asarray(hidden, dtype=jnp.float32),
            jnp.asarray(gate),
            jnp.asarray(bias),
            geometry=geometry,
        )
    with pytest.raises(RouterOracleError, match="at least one token"):
        jax_router_oracle_checked(
            jnp.asarray(hidden[:0]),
            jnp.asarray(gate),
            jnp.asarray(bias),
            geometry=geometry,
        )


def test_zero_unbiased_denominator_fails_explicitly() -> None:
    geometry = RouterGeometryV1(
        hidden_dim=1,
        expert_count=4,
        group_count=1,
        selected_group_count=1,
        experts_per_token=2,
    )
    hidden = np.asarray([-1.0e10], dtype=BF16)
    gate = np.ones((1, 4), dtype=BF16)
    bias = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(RouterOracleError, match="positive denominator"):
        numpy_router_oracle(hidden, gate, bias, geometry=geometry)
    with pytest.raises(RouterOracleError, match="positive denominator"):
        jax_router_oracle(
            jnp.asarray(hidden),
            jnp.asarray(gate),
            jnp.asarray(bias),
            geometry=geometry,
        )


def test_pinned_256_expert_smoke() -> None:
    geometry = PINNED_R1_ROUTER_GEOMETRY
    hidden = np.zeros((geometry.hidden_dim,), dtype=BF16)
    gate = np.zeros((geometry.hidden_dim, geometry.expert_count), dtype=BF16)
    bias = np.arange(geometry.expert_count, dtype=np.float32) * np.float32(0.001)
    result = numpy_router_oracle(hidden, gate, bias, geometry=geometry)
    np.testing.assert_array_equal(
        result.expert_ids,
        np.arange(248, 256, dtype=np.uint16),
    )
    np.testing.assert_allclose(
        result.router_weights,
        np.full(8, 2.5 / 8, dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_checkified_jax_path_is_jittable_and_batch_vectorizable() -> None:
    geometry, hidden, gate, bias = _random_fixture(101)
    compiled = jax.jit(
        lambda h, g, b: jax_router_oracle_checked(h, g, b, geometry=geometry)
    )
    error, (ids, weights) = compiled(
        jnp.asarray(hidden), jnp.asarray(gate), jnp.asarray(bias)
    )
    error.throw()
    validated = validate_jax_router_result(
        ids,
        weights,
        geometry,
        expected_leading_shape=hidden.shape[:-1],
    )
    expected = numpy_router_oracle(hidden, gate, bias, geometry=geometry)
    np.testing.assert_array_equal(validated.expert_ids, expected.expert_ids)

    def single(token):
        return jax_router_oracle_checked(
            token, jnp.asarray(gate), jnp.asarray(bias), geometry=geometry
        )

    vmapped_error, (vmapped_ids, vmapped_weights) = jax.jit(jax.vmap(single))(
        jnp.asarray(hidden.reshape(-1, geometry.hidden_dim))
    )
    vmapped_error.throw()
    np.testing.assert_array_equal(
        np.asarray(vmapped_ids).reshape(expected.expert_ids.shape),
        expected.expert_ids,
    )
    np.testing.assert_allclose(
        np.asarray(vmapped_weights).reshape(expected.router_weights.shape),
        expected.router_weights,
        rtol=3e-6,
        atol=3e-6,
    )
