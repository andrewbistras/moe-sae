"""Scientific and NumPy/JAX parity tests for Experiment 5."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moe_sae.contribution_training import ContributionOptimizerConfig
from moe_sae.experiment5_hybrid import (
    Experiment5Batch,
    Experiment5Centering,
    Experiment5Error,
    Experiment5Layout,
    Experiment5Recomposition,
    experiment5_bank_ids,
    experiment5_loss_and_grad_sums,
    experiment5_token_scores,
    experiment5_train_step,
    forward_experiment5,
    init_experiment5_train_state,
    jax_recompose_experiment5_full_ffn,
    normalized_experiment5_loss,
    numpy_forward_experiment5,
    numpy_recompose_experiment5_full_ffn,
    production_experiment5_layout,
    stack_experiment5_microbatches,
    validate_experiment5_microbatch,
)
from moe_sae.sae import SAEParams


def layout(condition: str) -> Experiment5Layout:
    return Experiment5Layout(
        condition=condition,  # type: ignore[arg-type]
        hidden_dim=3,
        num_experts=4,
        top_routed=2,
        routed_bank_width=3,
        routed_k=1,
        shared_width=4,
        shared_k=2,
        randomization_key=17,
    )


def batch(selected: Experiment5Layout) -> Experiment5Batch:
    return Experiment5Batch(
        routed_input=np.asarray(
            [[0.5, -0.25, 1.0], [-0.75, 0.5, 0.25]],
            np.float32,
        ),
        expert_output=np.asarray(
            [
                [[1.0, -0.5, 0.25], [0.5, 0.75, -1.0]],
                [[-0.25, 1.0, 0.5], [0.75, -0.5, 1.25]],
            ],
            np.float32,
        ),
        shared_normalized=np.asarray(
            [[0.125, 0.75, -0.5], [1.0, -0.25, 0.375]],
            np.float32,
        ),
        bank_ids=np.asarray([[3, 1], [2, 0]], np.int32),
        router_weights=np.asarray([[0.625, 0.375], [0.25, 0.75]], np.float32),
        valid_mask=np.asarray([True, True]),
    )


def centers() -> Experiment5Centering:
    return Experiment5Centering(
        source_mean=np.asarray([0.1, -0.2, 0.05], np.float32),
        routed_target_mean=np.asarray([0.2, 0.1, -0.1], np.float32),
        shared_target_mean=np.asarray([-0.1, 0.2, 0.05], np.float32),
        full_target_mean=np.asarray([0.0, 0.1, -0.2], np.float32),
    )


def recomposition() -> Experiment5Recomposition:
    return Experiment5Recomposition(
        routed_scale=np.asarray([0.75, 1.0, 1.25], np.float32),
        outer_scale=np.asarray([1.125, 0.875, 1.0], np.float32),
        epsilon=1e-6,
    )


def params(
    selected: Experiment5Layout,
    *,
    seed: int,
    bias: float = 4.0,
    parameter_dtype: jnp.dtype = jnp.float32,
) -> SAEParams:
    rng = np.random.default_rng(seed)
    return SAEParams(
        encoder=jnp.asarray(
            rng.normal(0.0, 0.35, (selected.total_features, selected.hidden_dim)),
            parameter_dtype,
        ),
        decoder=jnp.asarray(
            rng.normal(0.0, 0.35, (selected.hidden_dim, selected.total_features)),
            parameter_dtype,
        ),
        bias=jnp.full((selected.total_features,), bias, parameter_dtype),
    )


def test_frozen_production_layout_has_exact_matches() -> None:
    layouts = {
        condition: production_experiment5_layout(
            condition,  # type: ignore[arg-type]
            randomization_key=9,
        )
        for condition in ("H", "R", "B", "F")
    }

    assert layouts["H"].total_features == 69_632
    assert layouts["R"].total_features == 69_632
    assert layouts["B"].total_features == 8_192
    assert layouts["F"].total_features == 69_632
    assert layouts["H"].candidates_per_token == 8_192
    assert layouts["R"].candidates_per_token == 8_192
    assert layouts["B"].candidates_per_token == 8_192
    assert layouts["F"].candidates_per_token == 69_632
    assert {item.total_l0 for item in layouts.values()} == {64}
    assert layouts["H"].parameter_count == layouts["F"].parameter_count
    assert layouts["H"].parameter_count == 392_237_056
    assert layouts["H"].candidate_exposure == pytest.approx(8_192 / 69_632)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"condition": "A"}, "condition must be"),
        ({"condition": "H", "top_routed": 5, "num_experts": 4}, "cannot exceed"),
        (
            {"condition": "H", "routed_bank_width": 2, "routed_k": 3},
            "routed_k",
        ),
        (
            {"condition": "H", "shared_width": 1, "shared_k": 2},
            "shared_k",
        ),
    ],
)
def test_layout_rejects_contract_changes(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(Experiment5Error, match=message):
        Experiment5Layout(**kwargs)  # type: ignore[arg-type]


def test_h_and_r_bank_ids_reuse_stable_sequence_randomization() -> None:
    expert_ids = np.asarray([[3, 1], [3, 1], [3, 1]], np.int32)
    sequences = np.asarray([10, 10, 11], np.int64)
    h = experiment5_bank_ids(expert_ids, sequences, layout("H"))
    randomized = experiment5_bank_ids(expert_ids, sequences, layout("R"))

    np.testing.assert_array_equal(h, expert_ids)
    np.testing.assert_array_equal(randomized[0], randomized[1])
    assert not np.array_equal(randomized[0], randomized[2])
    with pytest.raises(Experiment5Error, match="only H/R"):
        experiment5_bank_ids(expert_ids, sequences, layout("B"))


def test_initializer_is_one_flat_parameter_tree_and_h_matches_f() -> None:
    optimizer = ContributionOptimizerConfig(
        parameter_dtype="float32",
        total_steps=3,
        warmup_steps=1,
    )
    h_layout = layout("H")
    f_layout = layout("F")
    h = init_experiment5_train_state(jax.random.key(0), h_layout, optimizer)
    f = init_experiment5_train_state(jax.random.key(1), f_layout, optimizer)

    assert h.params.encoder.shape == (
        h_layout.total_features,
        h_layout.hidden_dim,
    )
    assert h.params.decoder.shape == (
        h_layout.hidden_dim,
        h_layout.total_features,
    )
    assert h.params.bias.shape == (h_layout.total_features,)
    assert h.params.encoder.shape == f.params.encoder.shape
    assert sum(value.size for value in h.params) == h_layout.parameter_count
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(h.params.decoder), axis=0),
        1.0,
        rtol=1e-6,
        atol=1e-6,
    )


def test_host_batch_validation_and_stacking_fail_closed() -> None:
    selected = layout("H")
    one = batch(selected)
    checked = validate_experiment5_microbatch(one, selected)
    stacked = stack_experiment5_microbatches([checked, checked], selected)
    assert stacked.expert_output.shape == (2, 2, 2, 3)
    assert stacked.valid_mask.dtype == np.bool_

    invalid = one._replace(
        router_weights=np.asarray([[0.4, 0.4], [0.25, 0.75]], np.float32)
    )
    with pytest.raises(Experiment5Error, match="sum to one"):
        validate_experiment5_microbatch(invalid, selected)
    duplicate = one._replace(bank_ids=np.asarray([[1, 1], [2, 0]], np.int32))
    with pytest.raises(Experiment5Error, match="unique"):
        validate_experiment5_microbatch(duplicate, selected)


def test_exact_full_ffn_numpy_and_jax_recomposition_match() -> None:
    selected = layout("H")
    source = batch(selected)
    expected = numpy_recompose_experiment5_full_ffn(
        source.expert_output,
        source.shared_normalized,
        source.router_weights,
        recomposition(),
    )
    actual = jax_recompose_experiment5_full_ffn(
        jnp.asarray(source.expert_output),
        jnp.asarray(source.shared_normalized),
        jnp.asarray(source.router_weights),
        recomposition(),
    )

    np.testing.assert_array_equal(np.asarray(actual, np.float32), expected)


@pytest.mark.parametrize("condition", ["H", "R"])
@pytest.mark.parametrize("encoder_kind", ["sae", "transcoder"])
@pytest.mark.parametrize("parameter_dtype", [jnp.float32, jnp.bfloat16])
def test_hybrid_numpy_and_jax_forward_match(
    condition: str,
    encoder_kind: str,
    parameter_dtype: jnp.dtype,
) -> None:
    selected = layout(condition)
    source = batch(selected)
    model = params(selected, seed=12, parameter_dtype=parameter_dtype)
    numpy_output = numpy_forward_experiment5(
        model,
        source,
        centers(),
        recomposition(),
        selected,
        encoder_kind=encoder_kind,  # type: ignore[arg-type]
    )
    jax_output = forward_experiment5(
        model,
        source,
        centers(),
        recomposition(),
        selected,
        encoder_kind=encoder_kind,  # type: ignore[arg-type]
    )

    assert numpy_output.routed_selection is not None
    assert numpy_output.shared_selection is not None
    assert jax_output.routed_selection is not None
    assert jax_output.shared_selection is not None
    np.testing.assert_array_equal(
        numpy_output.routed_selection.latent_ids,
        jax_output.routed_selection.latent_ids,
    )
    np.testing.assert_array_equal(
        numpy_output.shared_selection.latent_ids,
        jax_output.shared_selection.latent_ids,
    )
    np.testing.assert_allclose(
        numpy_output.routed_selection.values,
        jax_output.routed_selection.values,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        numpy_output.shared_selection.values,
        jax_output.shared_selection.values,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        numpy_output.routed_reconstruction,
        jax_output.routed_reconstruction,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        numpy_output.shared_reconstruction,
        jax_output.shared_reconstruction,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_array_equal(
        numpy_output.full_reconstruction,
        np.asarray(jax_output.full_reconstruction, np.float32),
    )


@pytest.mark.parametrize("condition", ["B", "F"])
@pytest.mark.parametrize("encoder_kind", ["sae", "transcoder"])
@pytest.mark.parametrize("parameter_dtype", [jnp.float32, jnp.bfloat16])
def test_global_numpy_and_jax_forward_match(
    condition: str,
    encoder_kind: str,
    parameter_dtype: jnp.dtype,
) -> None:
    selected = layout(condition)
    source = batch(selected)
    model = params(selected, seed=31, parameter_dtype=parameter_dtype)
    numpy_output = numpy_forward_experiment5(
        model,
        source,
        centers(),
        recomposition(),
        selected,
        encoder_kind=encoder_kind,  # type: ignore[arg-type]
    )
    jax_output = forward_experiment5(
        model,
        source,
        centers(),
        recomposition(),
        selected,
        encoder_kind=encoder_kind,  # type: ignore[arg-type]
    )

    assert numpy_output.global_selection is not None
    assert jax_output.global_selection is not None
    np.testing.assert_array_equal(
        numpy_output.global_selection.latent_ids,
        jax_output.global_selection.latent_ids,
    )
    np.testing.assert_allclose(
        numpy_output.global_selection.values,
        jax_output.global_selection.values,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        numpy_output.full_reconstruction,
        jax_output.full_reconstruction,
        rtol=2e-6,
        atol=2e-6,
    )


def test_all_conditions_share_target_sst_and_enforce_exact_l0() -> None:
    observed = {}
    for index, condition in enumerate(("H", "R", "B", "F")):
        selected = layout(condition)
        loss, sums = normalized_experiment5_loss(
            params(selected, seed=index + 50),
            batch(selected),
            centers(),
            recomposition(),
            selected,
            encoder_kind="transcoder",
        )
        assert np.isfinite(float(loss))
        assert bool(sums.topk_valid)
        assert int(sums.valid_tokens) == 2
        assert int(sums.l0_sum) == 2 * selected.total_l0
        observed[condition] = float(sums.sst)

    assert max(observed.values()) == pytest.approx(
        min(observed.values()),
        rel=0,
        abs=0,
    )


@pytest.mark.parametrize("condition", ["H", "B"])
@pytest.mark.parametrize("encoder_kind", ["sae", "transcoder"])
def test_loss_gradients_are_finite_and_train_step_preserves_decoder_norms(
    condition: str,
    encoder_kind: str,
) -> None:
    selected = layout(condition)
    optimizer = ContributionOptimizerConfig(
        parameter_dtype="float32",
        total_steps=3,
        warmup_steps=1,
    )
    state = init_experiment5_train_state(
        jax.random.key(71),
        selected,
        optimizer,
    )
    # Positive bias makes the SAE exact-TopK health property deterministic.
    state = state._replace(
        params=state.params._replace(
            bias=jnp.full_like(state.params.bias, 2.0)
        )
    )
    sums, gradients = experiment5_loss_and_grad_sums(
        state.params,
        batch(selected),
        centers(),
        recomposition(),
        selected,
        encoder_kind=encoder_kind,  # type: ignore[arg-type]
    )
    assert bool(sums.topk_valid)
    for gradient in gradients:
        assert np.isfinite(np.asarray(gradient)).all()

    updated, metrics = experiment5_train_step(
        state,
        batch(selected),
        centers(),
        recomposition(),
        selected,
        optimizer,
        encoder_kind=encoder_kind,  # type: ignore[arg-type]
    )
    assert bool(metrics.topk_valid)
    assert float(metrics.l0) == selected.total_l0
    assert int(updated.step) == 1
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(updated.params.decoder), axis=0),
        1.0,
        rtol=2e-5,
        atol=2e-5,
    )


def test_evaluation_scores_use_one_common_mean_and_mask_padding() -> None:
    target = np.asarray([[1.0, 2.0], [3.0, 4.0]], np.float32)
    estimate = np.asarray([[0.5, 2.5], [0.0, 0.0]], np.float32)
    scores = experiment5_token_scores(
        target,
        estimate,
        np.zeros((2,), np.float32),
        valid_mask=np.asarray([True, False]),
    )

    np.testing.assert_allclose(scores.token_sse, [0.5, 0.0])
    np.testing.assert_allclose(scores.token_sst, [5.0, 0.0])
    assert scores.tokens == 1
    assert scores.nmse == pytest.approx(0.1)
    assert scores.ev == pytest.approx(0.9)


def test_sae_softplus_preserves_exact_positive_budget_for_negative_bias() -> None:
    selected = layout("H")
    loss, sums = normalized_experiment5_loss(
        params(selected, seed=90, bias=-100.0),
        batch(selected),
        centers(),
        recomposition(),
        selected,
        encoder_kind="sae",
    )

    assert bool(sums.topk_valid)
    assert int(sums.l0_sum) == selected.total_l0 * 2
    assert np.isfinite(float(loss))
