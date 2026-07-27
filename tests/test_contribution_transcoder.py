"""Focused correctness tests for the Experiment-3 transcoder core."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moe_sae.contribution_sae import ContributionSAEConfig, condition_bank_ids
from moe_sae.contribution_training import ContributionOptimizerConfig
from moe_sae.contribution_transcoder import (
    ContributionTranscoderParams,
    ContributionTranscoderTrainState,
    contribution_transcoder_bank_ids,
    contribution_transcoder_candidate_scores,
    contribution_transcoder_loss_and_grad_sums,
    contribution_transcoder_token_scores,
    contribution_transcoder_train_step,
    forward_contribution_transcoder,
    init_contribution_transcoder_train_state,
    normalized_contribution_transcoder_loss,
    numpy_forward_contribution_transcoder,
    transcoder_params_to_sae,
    transcoder_state_from_sae,
    transcoder_state_to_sae,
)


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


def test_bank_ids_are_exactly_the_existing_a_r_b_f_layouts() -> None:
    expert_ids = np.asarray([[3, 1], [2, 0], [3, 0]], dtype=np.int32)
    sequence_ids = np.asarray([10, 10, 11], dtype=np.int64)

    for condition in ("A", "R", "B", "F"):
        selected = layout(condition)
        actual = contribution_transcoder_bank_ids(
            expert_ids,
            sequence_ids,
            selected,
        )
        expected = condition_bank_ids(expert_ids, sequence_ids, selected)
        np.testing.assert_array_equal(actual, expected)

    np.testing.assert_array_equal(
        contribution_transcoder_bank_ids(
            expert_ids,
            sequence_ids,
            layout("A"),
        ),
        expert_ids,
    )
    ranks = np.broadcast_to(np.arange(2, dtype=np.int32), expert_ids.shape)
    np.testing.assert_array_equal(
        contribution_transcoder_bank_ids(
            expert_ids,
            sequence_ids,
            layout("B"),
        ),
        ranks,
    )
    np.testing.assert_array_equal(
        contribution_transcoder_bank_ids(
            expert_ids,
            sequence_ids,
            layout("F"),
        ),
        ranks,
    )


def test_initializer_and_optimizer_compatibility_are_explicit() -> None:
    selected = layout("A", hidden=3, experts=4, routed=2, width=5, k=2)
    state = init_contribution_transcoder_train_state(
        jax.random.key(0),
        selected,
        ContributionOptimizerConfig(parameter_dtype="float32"),
    )

    assert isinstance(state, ContributionTranscoderTrainState)
    assert isinstance(state.params, ContributionTranscoderParams)
    assert state.params.encoder.shape == (selected.total_features, 3)
    assert state.params.decoder.shape == (3, selected.total_features)
    assert state.params.encoder_bias.shape == (selected.total_features,)
    assert sum(value.size for value in state.params) == selected.parameter_count
    np.testing.assert_allclose(
        jnp.linalg.norm(state.params.decoder, axis=0),
        np.ones((selected.total_features,)),
        rtol=1e-6,
        atol=1e-6,
    )

    round_trip = transcoder_state_from_sae(transcoder_state_to_sae(state))
    for before, after in zip(
        jax.tree_util.tree_leaves(state),
        jax.tree_util.tree_leaves(round_trip),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)

    with pytest.raises(TypeError, match="ContributionTranscoderParams"):
        contribution_transcoder_candidate_scores(
            transcoder_params_to_sae(state.params),  # type: ignore[arg-type]
            jnp.ones((1, 3), jnp.float32),
            jnp.zeros((1, 2), jnp.int32),
            jnp.zeros((3,), jnp.float32),
            selected,
        )


def test_candidate_scores_repeat_one_input_but_gather_only_assigned_banks() -> None:
    selected = layout("A", experts=3, routed=2, width=2)
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
    params = ContributionTranscoderParams(
        encoder=encoder,
        decoder=encoder.T,
        encoder_bias=jnp.zeros((6,), jnp.float32),
    )
    source = jnp.asarray([[3, 1]], jnp.float32)
    source_mean = jnp.asarray([1, 0], jnp.float32)
    banks = jnp.asarray([[0, 2]], jnp.int32)

    scores, latent_ids = contribution_transcoder_candidate_scores(
        params,
        source,
        banks,
        source_mean,
        selected,
    )

    # Shape is routed slots times one bank width, not all three banks.
    assert scores.shape == (2, selected.bank_width)
    np.testing.assert_allclose(
        scores,
        np.logaddexp(
            np.float32(0),
            np.asarray([[2, 1], [6, 3]], np.float32),
        ),
    )
    np.testing.assert_array_equal(latent_ids, [[0, 1], [4, 5]])


def test_forward_keeps_source_and_target_centers_distinct_and_decodes_sparse() -> None:
    selected = layout("A", experts=2, routed=2, width=3, k=1)
    params = ContributionTranscoderParams(
        encoder=jnp.asarray(
            [
                [1, 0],
                [0, 1],
                [-1, 0],
                [2, 0],
                [0, 2],
                [-2, 0],
            ],
            jnp.float32,
        ),
        # Unselected columns are deliberately enormous.  A dense latent decode
        # would visibly corrupt the expected reconstruction.
        decoder=jnp.asarray(
            [
                [5, 1_000, 1_000, -2, 1_000, 1_000],
                [6, 1_000, 1_000, 3, 1_000, 1_000],
            ],
            jnp.float32,
        ),
        encoder_bias=jnp.zeros((6,), jnp.float32),
    )

    output = forward_contribution_transcoder(
        params,
        jnp.asarray([[3, 4]], jnp.float32),
        jnp.asarray([[0, 1]], jnp.int32),
        jnp.asarray([2, 4], jnp.float32),
        jnp.asarray([10, 20], jnp.float32),
        selected,
    )

    # The source-center yields [1, 0]. Bank 0 selects softplus(1) and bank 1
    # selects softplus(2); only then is the independent target-center added.
    np.testing.assert_array_equal(output.selection.latent_ids, [[0], [3]])
    first = np.logaddexp(np.float32(0), np.float32(1))
    second = np.logaddexp(np.float32(0), np.float32(2))
    np.testing.assert_allclose(
        output.reconstruction,
        np.asarray(
            [
                [
                    [10 + 5 * first, 20 + 6 * first],
                    [10 - 2 * second, 20 + 3 * second],
                ]
            ],
            np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_forward_uses_global_latent_id_as_deterministic_tie_break() -> None:
    selected = layout("A", experts=2, routed=2, width=3, k=2)
    params = ContributionTranscoderParams(
        encoder=jnp.ones((6, 2), jnp.float32),
        decoder=jnp.eye(2, 6, dtype=jnp.float32),
        encoder_bias=jnp.zeros((6,), jnp.float32),
    )

    output = forward_contribution_transcoder(
        params,
        jnp.ones((1, 2), jnp.float32),
        jnp.asarray([[1, 0]], jnp.int32),
        jnp.zeros((2,), jnp.float32),
        jnp.zeros((2,), jnp.float32),
        selected,
    )

    np.testing.assert_array_equal(output.selection.latent_ids, [[3, 4], [0, 1]])


@pytest.mark.parametrize("condition", ["A", "R", "B", "F"])
@pytest.mark.parametrize("parameter_dtype", [jnp.float32, jnp.bfloat16])
def test_numpy_and_jax_forward_match_for_every_condition(
    condition: str,
    parameter_dtype: jnp.dtype,
) -> None:
    selected = layout(condition, experts=4, routed=2, width=5, k=2)
    rng = np.random.default_rng(19)
    params = ContributionTranscoderParams(
        encoder=jnp.asarray(
            rng.normal(size=(selected.total_features, 2)),
            parameter_dtype,
        ),
        decoder=jnp.asarray(
            rng.normal(size=(2, selected.total_features)),
            parameter_dtype,
        ),
        # Keep every row well inside the exact-positive TopK contract.
        encoder_bias=jnp.full(
            (selected.total_features,),
            4.0,
            parameter_dtype,
        ),
    )
    source = rng.normal(size=(3, 2)).astype(np.float32)
    source_mean = np.asarray([0.25, -0.5], np.float32)
    target_mean = np.asarray([10.0, -7.0], np.float32)
    expert_ids = np.asarray([[3, 1], [2, 0], [3, 0]], np.int32)
    sequences = np.asarray([10, 10, 11], np.int64)
    banks = contribution_transcoder_bank_ids(
        expert_ids,
        sequences,
        selected,
    )

    numpy_output = numpy_forward_contribution_transcoder(
        params,
        source,
        banks,
        source_mean,
        target_mean,
        selected,
    )
    jax_output = forward_contribution_transcoder(
        params,
        jnp.asarray(source),
        jnp.asarray(banks),
        jnp.asarray(source_mean),
        jnp.asarray(target_mean),
        selected,
    )

    np.testing.assert_array_equal(numpy_output.valid, jax_output.selection.valid)
    np.testing.assert_array_equal(
        numpy_output.latent_ids,
        jax_output.selection.latent_ids,
    )
    np.testing.assert_allclose(
        numpy_output.values,
        jax_output.selection.values,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        numpy_output.reconstruction,
        jax_output.reconstruction,
        rtol=2e-6,
        atol=2e-6,
    )


def test_changing_targets_cannot_change_preupdate_latent_selection() -> None:
    selected = layout(
        "B",
        hidden=1,
        experts=2,
        routed=2,
        width=2,
        k=1,
    )
    params = ContributionTranscoderParams(
        encoder=jnp.asarray([[1], [-1], [1], [-1]], jnp.float32),
        decoder=jnp.asarray([[1, -1, 1, -1]], jnp.float32),
        encoder_bias=jnp.zeros((4,), jnp.float32),
    )
    source = jnp.asarray([[2]], jnp.float32)
    banks = jnp.asarray([[0, 1]], jnp.int32)
    weights = jnp.asarray([[0.5, 0.5]], jnp.float32)
    center = jnp.zeros((1,), jnp.float32)
    valid = jnp.ones((1,), bool)
    before = forward_contribution_transcoder(
        params,
        source,
        banks,
        center,
        center,
        selected,
    )
    first_loss, _ = normalized_contribution_transcoder_loss(
        params,
        source,
        jnp.asarray([[[2], [2]]], jnp.float32),
        banks,
        weights,
        center,
        center,
        selected,
        valid_mask=valid,
    )
    second_loss, _ = normalized_contribution_transcoder_loss(
        params,
        source,
        jnp.asarray([[[-4], [7]]], jnp.float32),
        banks,
        weights,
        center,
        center,
        selected,
        valid_mask=valid,
    )
    after = forward_contribution_transcoder(
        params,
        source,
        banks,
        center,
        center,
        selected,
    )

    assert float(first_loss) != float(second_loss)
    np.testing.assert_array_equal(
        before.selection.latent_ids,
        after.selection.latent_ids,
    )
    np.testing.assert_array_equal(
        before.selection.values,
        after.selection.values,
    )


def test_weighted_loss_matches_numpy_metric_and_counts_exact_source_l0() -> None:
    selected = layout(
        "A",
        hidden=2,
        experts=8,
        routed=8,
        width=2,
        k=1,
    )
    bank_encoder = jnp.asarray([[1, 0], [0, 1]], jnp.float32)
    params = ContributionTranscoderParams(
        encoder=jnp.tile(bank_encoder, (8, 1)),
        decoder=jnp.zeros((2, 16), jnp.float32),
        encoder_bias=jnp.zeros((16,), jnp.float32),
    )
    source = jnp.ones((2, 2), jnp.float32)
    target = jnp.ones((2, 8, 2), jnp.float32)
    banks = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (2, 8))
    weights = jnp.asarray(
        [[1 / 8] * 8, [0.0] * 8],
        jnp.float32,
    )
    source_mean = jnp.zeros((2,), jnp.float32)
    target_mean = jnp.zeros((2,), jnp.float32)
    valid = jnp.asarray([True, False])

    loss, sums = normalized_contribution_transcoder_loss(
        params,
        source,
        target,
        banks,
        weights,
        source_mean,
        target_mean,
        selected,
        valid_mask=valid,
    )
    output = forward_contribution_transcoder(
        params,
        source,
        banks,
        source_mean,
        target_mean,
        selected,
    )
    host_scores = contribution_transcoder_token_scores(
        np.asarray(target),
        np.asarray(output.reconstruction),
        np.asarray(weights),
        np.asarray(target_mean),
        valid_mask=np.asarray(valid),
    )

    assert bool(sums.topk_valid)
    assert int(sums.l0_sum) == selected.total_l0
    assert int(sums.valid_source_tokens) == 1
    assert int(sums.valid_components) == 8
    assert int(jnp.sum(sums.fired_mask)) == 8
    assert float(sums.sse) == pytest.approx(0.25)
    assert float(sums.sst) == pytest.approx(0.25)
    assert float(loss) == pytest.approx(1.0)
    assert host_scores.sse == pytest.approx(float(sums.sse))
    assert host_scores.sst == pytest.approx(float(sums.sst))


def test_softplus_row_is_valid_while_invalid_bank_or_router_fails() -> None:
    selected = layout("A", experts=2, routed=2, width=2, k=1)
    params = ContributionTranscoderParams(
        encoder=jnp.zeros((4, 2), jnp.float32),
        decoder=jnp.zeros((2, 4), jnp.float32),
        encoder_bias=jnp.zeros((4,), jnp.float32),
    )
    source = jnp.ones((1, 2), jnp.float32)
    target = jnp.ones((1, 2, 2), jnp.float32)
    valid = jnp.ones((1,), bool)

    softplus_loss, softplus_sums = normalized_contribution_transcoder_loss(
        params,
        source,
        target,
        jnp.asarray([[0, 1]], jnp.int32),
        jnp.full((1, 2), 0.5, jnp.float32),
        jnp.zeros((2,), jnp.float32),
        jnp.zeros((2,), jnp.float32),
        selected,
        valid_mask=valid,
    )
    invalid_bank_loss, invalid_bank_sums = (
        normalized_contribution_transcoder_loss(
            params._replace(encoder_bias=jnp.ones((4,), jnp.float32)),
            source,
            target,
            jnp.asarray([[0, 2]], jnp.int32),
            jnp.full((1, 2), 0.5, jnp.float32),
            jnp.zeros((2,), jnp.float32),
            jnp.zeros((2,), jnp.float32),
            selected,
            valid_mask=valid,
        )
    )
    invalid_router_loss, invalid_router_sums = (
        normalized_contribution_transcoder_loss(
            params._replace(encoder_bias=jnp.ones((4,), jnp.float32)),
            source,
            target,
            jnp.asarray([[0, 1]], jnp.int32),
            jnp.asarray([[0.8, 0.8]], jnp.float32),
            jnp.zeros((2,), jnp.float32),
            jnp.zeros((2,), jnp.float32),
            selected,
            valid_mask=valid,
        )
    )

    assert bool(softplus_sums.topk_valid)
    assert np.isfinite(float(softplus_loss))
    assert not bool(invalid_bank_sums.topk_valid)
    assert np.isnan(float(invalid_bank_loss))
    assert not bool(invalid_router_sums.topk_valid)
    assert np.isnan(float(invalid_router_loss))


@pytest.mark.parametrize("condition", ["A", "R", "B", "F"])
def test_strongly_negative_finite_rows_still_select_exact_positive_topk(
    condition: str,
) -> None:
    selected = layout(
        condition,
        hidden=2,
        experts=4,
        routed=2,
        width=4,
        k=2,
    )
    params = ContributionTranscoderParams(
        encoder=jnp.zeros(
            (selected.total_features, selected.hidden_dim),
            jnp.float32,
        ),
        decoder=jnp.zeros(
            (selected.hidden_dim, selected.total_features),
            jnp.float32,
        ),
        encoder_bias=jnp.full(
            (selected.total_features,),
            -1.0e6,
            jnp.float32,
        ),
    )
    source = np.asarray([[1.0, -2.0], [-3.0, 4.0]], np.float32)
    source_mean = np.asarray([0.25, -0.5], np.float32)
    target_mean = np.asarray([0.0, 0.0], np.float32)
    if condition in {"A", "R"}:
        banks = np.asarray([[0, 3], [2, 1]], np.int32)
    else:
        banks = np.broadcast_to(
            np.arange(selected.top_routed, dtype=np.int32),
            (source.shape[0], selected.top_routed),
        ).copy()

    numpy_output = numpy_forward_contribution_transcoder(
        params,
        source,
        banks,
        source_mean,
        target_mean,
        selected,
    )
    jax_output = forward_contribution_transcoder(
        params,
        jnp.asarray(source),
        jnp.asarray(banks),
        jnp.asarray(source_mean),
        jnp.asarray(target_mean),
        selected,
    )

    assert np.all(numpy_output.valid)
    assert bool(jnp.all(jax_output.selection.valid))
    assert np.all(numpy_output.values > 0)
    assert bool(jnp.all(jax_output.selection.values > 0))
    np.testing.assert_array_equal(
        numpy_output.latent_ids,
        jax_output.selection.latent_ids,
    )
    np.testing.assert_array_equal(
        numpy_output.latent_ids,
        (
            banks.reshape(-1, 1) * selected.bank_width
            + np.arange(selected.k_per_contribution, dtype=np.int32)[None, :]
        ),
    )
    np.testing.assert_allclose(
        numpy_output.values,
        jax_output.selection.values,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("routed_input", jnp.ones((2, 3), jnp.float32)),
        ("expert_output", jnp.ones((2, 2, 3), jnp.float32)),
        ("bank_ids", jnp.zeros((2, 1), jnp.int32)),
        ("router_weights", jnp.ones((2, 1), jnp.float32)),
        ("source_mean", jnp.zeros((3,), jnp.float32)),
        ("target_mean", jnp.zeros((3,), jnp.float32)),
        ("valid_mask", jnp.ones((3,), bool)),
    ],
)
def test_shape_contract_fails_closed(
    field: str,
    replacement: jax.Array,
) -> None:
    selected = layout()
    values: dict[str, jax.Array] = {
        "routed_input": jnp.ones((2, 2), jnp.float32),
        "expert_output": jnp.ones((2, 2, 2), jnp.float32),
        "bank_ids": jnp.asarray([[0, 1], [2, 3]], jnp.int32),
        "router_weights": jnp.full((2, 2), 0.5, jnp.float32),
        "source_mean": jnp.zeros((2,), jnp.float32),
        "target_mean": jnp.zeros((2,), jnp.float32),
        "valid_mask": jnp.ones((2,), bool),
    }
    values[field] = replacement
    params = ContributionTranscoderParams(
        encoder=jnp.ones((12, 2), jnp.float32),
        decoder=jnp.ones((2, 12), jnp.float32),
        encoder_bias=jnp.ones((12,), jnp.float32),
    )

    with pytest.raises(ValueError):
        normalized_contribution_transcoder_loss(
            params,
            values["routed_input"],
            values["expert_output"],
            values["bank_ids"],
            values["router_weights"],
            values["source_mean"],
            values["target_mean"],
            selected,
            valid_mask=values["valid_mask"],
        )


def test_gradients_and_shared_adam_update_retain_transcoder_types() -> None:
    selected = layout(
        "B",
        hidden=1,
        experts=2,
        routed=2,
        width=2,
        k=1,
    )
    optimizer = ContributionOptimizerConfig(
        parameter_dtype="float32",
        peak_lr=5e-2,
        min_lr=5e-2,
        warmup_steps=0,
        total_steps=20,
    )
    state = init_contribution_transcoder_train_state(
        jax.random.key(1),
        selected,
        optimizer,
    )
    params = ContributionTranscoderParams(
        encoder=jnp.asarray([[1], [-1], [1], [-1]], jnp.float32),
        decoder=jnp.asarray([[1, -1, -1, 1]], jnp.float32),
        encoder_bias=jnp.zeros((4,), jnp.float32),
    )
    state = state._replace(params=params)
    source = jnp.asarray([[-2], [-1], [1], [2]], jnp.float32)
    target = jnp.stack((2 * source, -2 * source), axis=1)
    banks = jnp.broadcast_to(jnp.arange(2, dtype=jnp.int32), (4, 2))
    weights = jnp.full((4, 2), 0.5, jnp.float32)
    source_mean = jnp.zeros((1,), jnp.float32)
    target_mean = jnp.zeros((1,), jnp.float32)
    valid = jnp.ones((4,), bool)

    initial, _ = normalized_contribution_transcoder_loss(
        state.params,
        source,
        target,
        banks,
        weights,
        source_mean,
        target_mean,
        selected,
        valid_mask=valid,
    )
    sums, gradients = contribution_transcoder_loss_and_grad_sums(
        state.params,
        source,
        target,
        banks,
        weights,
        source_mean,
        target_mean,
        selected,
        valid_mask=valid,
    )
    assert bool(sums.topk_valid)
    assert isinstance(gradients, ContributionTranscoderParams)
    assert all(
        bool(jnp.all(jnp.isfinite(value)))
        for value in jax.tree_util.tree_leaves(gradients)
    )

    for _ in range(20):
        state, metrics = contribution_transcoder_train_step(
            state,
            source,
            target,
            banks,
            weights,
            source_mean,
            target_mean,
            selected,
            optimizer,
            valid_mask=valid,
        )
    final, final_sums = normalized_contribution_transcoder_loss(
        state.params,
        source,
        target,
        banks,
        weights,
        source_mean,
        target_mean,
        selected,
        valid_mask=valid,
    )

    assert isinstance(state, ContributionTranscoderTrainState)
    assert isinstance(state.params, ContributionTranscoderParams)
    assert bool(final_sums.topk_valid)
    assert bool(metrics.topk_valid)
    assert int(state.step) == 20
    assert int(state.tokens_seen) == 80
    assert float(metrics.l0) == pytest.approx(selected.total_l0)
    assert float(final) < 0.5 * float(initial)
    np.testing.assert_allclose(
        jnp.linalg.norm(state.params.decoder, axis=0),
        np.ones((selected.total_features,)),
        rtol=1e-6,
        atol=1e-6,
    )
