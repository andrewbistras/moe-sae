"""JAX reference tests for Experiment-2 contribution SAE training."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_training import (
    ContributionOptimizerConfig,
    contribution_candidate_scores,
    contribution_train_step,
    forward_contributions,
    init_contribution_train_state,
    normalized_contribution_loss,
    optimizer_sae_config,
)
from moe_sae.sae import SAEParams


def layout(
    condition: str = "A",
    *,
    hidden: int = 2,
    experts: int = 4,
    routed: int = 2,
    width: int = 3,
    k: int = 1,
) -> ContributionSAEConfig:
    return ContributionSAEConfig(
        condition=condition,  # type: ignore[arg-type]
        hidden_dim=hidden,
        num_experts=experts,
        top_routed=routed,
        bank_width=width,
        k_per_contribution=k,
        randomization_key=7,
    )


def test_optimizer_mapping_preserves_all_condition_parameter_counts() -> None:
    layouts = {
        "A": layout("A"),
        "R": layout("R"),
        "B": layout("B"),
        "F": layout("F", width=6),
    }

    for selected in layouts.values():
        config = optimizer_sae_config(selected)
        assert config.total_features == selected.total_features
        assert config.parameter_count == selected.parameter_count
        assert config.k == selected.total_l0

    assert optimizer_sae_config(layouts["A"]).condition == "A"
    assert optimizer_sae_config(layouts["R"]).condition == "A"
    assert optimizer_sae_config(layouts["B"]).condition == "B"
    assert optimizer_sae_config(layouts["F"]).condition == "F"
    assert optimizer_sae_config(layouts["F"]).num_experts == 2


def test_candidate_scores_expose_only_the_assigned_bank() -> None:
    selected = layout(experts=3, routed=2, width=2)
    # Bank b has feature scores (b+1)*x0 and (b+1)*x1.
    encoder = jnp.asarray(
        [
            [1, 0],
            [0, 1],
            [2, 0],
            [0, 2],
            [3, 0],
            [0, 3],
        ],
        dtype=jnp.float32,
    )
    params = SAEParams(
        encoder=encoder,
        decoder=encoder.T,
        bias=jnp.zeros((6,), jnp.float32),
    )
    target = jnp.asarray([[[2, 1], [2, 1]]], jnp.float32)
    banks = jnp.asarray([[0, 2]], jnp.int32)

    scores, ids = contribution_candidate_scores(
        params,
        target,
        banks,
        jnp.zeros((2,), jnp.float32),
        selected,
    )

    np.testing.assert_allclose(scores, [[2, 1], [6, 3]])
    np.testing.assert_array_equal(ids, [[0, 1], [4, 5]])


def test_forward_uses_global_latent_id_as_the_tie_break() -> None:
    selected = layout(experts=2, routed=2, width=3, k=2)
    params = SAEParams(
        encoder=jnp.ones((6, 2), jnp.float32),
        decoder=jnp.eye(2, 6, dtype=jnp.float32),
        bias=jnp.zeros((6,), jnp.float32),
    )
    target = jnp.ones((1, 2, 2), jnp.float32)
    banks = jnp.asarray([[1, 0]], jnp.int32)

    output = forward_contributions(
        params,
        target,
        banks,
        jnp.zeros((2,), jnp.float32),
        selected,
    )

    np.testing.assert_array_equal(output.selection.latent_ids, [[3, 4], [0, 1]])
    assert output.reconstruction.shape == target.shape


def test_weighted_loss_is_component_objective_not_aggregate_objective() -> None:
    selected = layout(experts=2, routed=2, width=1, k=1)
    params = SAEParams(
        encoder=jnp.asarray([[1, 0], [1, 0]], jnp.float32),
        decoder=jnp.asarray([[0, 0], [0, 0]], jnp.float32),
        bias=jnp.asarray([2, 2], jnp.float32),
    )
    target = jnp.asarray([[[1, 0], [-1, 0]]], jnp.float32)
    banks = jnp.asarray([[0, 1]], jnp.int32)
    weights = jnp.asarray([[0.75, 0.25]], jnp.float32)

    loss, sums = normalized_contribution_loss(
        params,
        target,
        banks,
        weights,
        jnp.zeros((2,), jnp.float32),
        selected,
    )

    # Both reconstructions are zero. The aggregate weighted target is 0.5 and
    # could be fit separately, but the component objective retains both errors.
    assert float(sums.sse) == pytest.approx(0.75**2 + 0.25**2)
    assert float(sums.sst) == pytest.approx(0.75**2 + 0.25**2)
    assert float(loss) == pytest.approx(1.0)
    assert int(sums.l0_sum) == 2
    assert int(sums.valid_source_tokens) == 1
    assert int(sums.valid_components) == 2


def test_padding_is_excluded_and_exact_l0_is_per_source_token() -> None:
    selected = layout(experts=2, routed=2, width=3, k=2)
    params = SAEParams(
        encoder=jnp.ones((6, 2), jnp.float32),
        decoder=jnp.zeros((2, 6), jnp.float32),
        bias=jnp.ones((6,), jnp.float32),
    )
    target = jnp.asarray(
        [
            [[1, 0], [1, 0]],
            [[0, 0], [0, 0]],
        ],
        jnp.float32,
    )
    banks = jnp.asarray([[0, 1], [0, 1]], jnp.int32)
    weights = jnp.full((2, 2), 0.5, jnp.float32)

    _, sums = normalized_contribution_loss(
        params,
        target,
        banks,
        weights,
        jnp.zeros((2,), jnp.float32),
        selected,
        valid_mask=jnp.asarray([True, False]),
    )

    assert bool(sums.topk_valid)
    assert int(sums.l0_sum) == selected.total_l0
    assert int(sums.valid_source_tokens) == 1
    assert int(sums.valid_components) == 2


def test_fewer_than_k_positive_features_fails_scientific_batch() -> None:
    selected = layout(experts=2, routed=2, width=3, k=2)
    params = SAEParams(
        encoder=jnp.zeros((6, 2), jnp.float32),
        decoder=jnp.zeros((2, 6), jnp.float32),
        bias=jnp.zeros((6,), jnp.float32),
    )
    target = jnp.ones((1, 2, 2), jnp.float32)

    loss, sums = normalized_contribution_loss(
        params,
        target,
        jnp.asarray([[0, 1]], jnp.int32),
        jnp.full((1, 2), 0.5, jnp.float32),
        jnp.zeros((2,), jnp.float32),
        selected,
    )

    assert not bool(sums.topk_valid)
    assert np.isnan(float(loss))


def test_invalid_bank_or_router_weights_fail_scientific_batch() -> None:
    selected = layout(experts=2, routed=2, width=3, k=1)
    params = SAEParams(
        encoder=jnp.ones((6, 2), jnp.float32),
        decoder=jnp.zeros((2, 6), jnp.float32),
        bias=jnp.ones((6,), jnp.float32),
    )
    target = jnp.ones((1, 2, 2), jnp.float32)
    mean = jnp.zeros((2,), jnp.float32)

    invalid_bank_loss, invalid_bank_sums = normalized_contribution_loss(
        params,
        target,
        jnp.asarray([[0, 2]], jnp.int32),
        jnp.full((1, 2), 0.5, jnp.float32),
        mean,
        selected,
    )
    invalid_router_loss, invalid_router_sums = normalized_contribution_loss(
        params,
        target,
        jnp.asarray([[0, 1]], jnp.int32),
        jnp.asarray([[0.7, 0.7]], jnp.float32),
        mean,
        selected,
    )

    assert not bool(invalid_bank_sums.topk_valid)
    assert np.isnan(float(invalid_bank_loss))
    assert not bool(invalid_router_sums.topk_valid)
    assert np.isnan(float(invalid_router_loss))


@pytest.mark.parametrize(
    "field",
    ["expert_output", "bank_ids", "router_weights", "mean", "valid_mask"],
)
def test_shape_contract_fails_closed(field: str) -> None:
    selected = layout()
    state = init_contribution_train_state(
        jax.random.key(0),
        selected,
        ContributionOptimizerConfig(parameter_dtype="float32"),
    )
    values: dict[str, jax.Array] = {
        "expert_output": jnp.ones((2, 2, 2), jnp.float32),
        "bank_ids": jnp.asarray([[0, 1], [2, 3]], jnp.int32),
        "router_weights": jnp.full((2, 2), 0.5, jnp.float32),
        "mean": jnp.zeros((2,), jnp.float32),
        "valid_mask": jnp.ones((2,), bool),
    }
    replacements = {
        "expert_output": jnp.ones((2, 2, 3), jnp.float32),
        "bank_ids": jnp.zeros((2, 1), jnp.int32),
        "router_weights": jnp.ones((2, 1), jnp.float32),
        "mean": jnp.zeros((3,), jnp.float32),
        "valid_mask": jnp.ones((3,), bool),
    }
    values[field] = replacements[field]

    with pytest.raises(ValueError):
        normalized_contribution_loss(
            state.params,
            values["expert_output"],
            values["bank_ids"],
            values["router_weights"],
            values["mean"],
            selected,
            valid_mask=values["valid_mask"],
        )


def test_reference_training_has_finite_state_and_falling_loss() -> None:
    selected = layout(
        "B",
        hidden=3,
        experts=8,
        routed=2,
        width=16,
        k=2,
    )
    optimizer = ContributionOptimizerConfig(
        parameter_dtype="float32",
        peak_lr=3e-3,
        min_lr=3e-3,
        warmup_steps=0,
        total_steps=40,
    )
    state = init_contribution_train_state(
        jax.random.key(3),
        selected,
        optimizer,
    )
    key = jax.random.key(5)
    target = jax.random.normal(key, (24, 2, 3), dtype=jnp.float32)
    banks = jnp.broadcast_to(
        jnp.arange(2, dtype=jnp.int32)[None, :],
        (24, 2),
    )
    weights = jnp.full((24, 2), 0.5, jnp.float32)
    mean = jnp.mean(target, axis=(0, 1))

    initial, initial_sums = normalized_contribution_loss(
        state.params,
        target,
        banks,
        weights,
        mean,
        selected,
    )
    assert bool(initial_sums.topk_valid)
    for _ in range(40):
        state, metrics = contribution_train_step(
            state,
            target,
            banks,
            weights,
            mean,
            selected,
            optimizer,
        )
    final, final_sums = normalized_contribution_loss(
        state.params,
        target,
        banks,
        weights,
        mean,
        selected,
    )

    assert bool(final_sums.topk_valid)
    assert np.isfinite(float(metrics.grad_norm))
    assert int(state.tokens_seen) == 40 * 24
    assert float(final) < 0.8 * float(initial)
