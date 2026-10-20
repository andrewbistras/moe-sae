from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np

from moe_sae.sae import (
    SAEConfig,
    SAEParams,
    Selection,
    decode_selected_columns,
    forward,
    init_train_state,
    learning_rate,
    local_topk_candidates,
    make_grouping_plan,
    merge_feature_shard_topk,
    nonnegative_topk,
    positive_topk,
    train_step,
)


def tiny_config(condition: str, *, total_steps: int = 4) -> SAEConfig:
    return SAEConfig(
        condition,
        hidden_dim=3,
        num_experts=4,
        top_routed=2,
        bank_width=2,
        k=2,
        parameter_dtype="float32",
        warmup_steps=2,
        total_steps=total_steps,
    )


def test_production_widths_and_parameter_counts() -> None:
    a = SAEConfig("A")
    b = SAEConfig("B")
    f = SAEConfig("F")
    assert (a.total_features, a.candidates_per_token) == (131_072, 8_192)
    assert (b.total_features, b.candidates_per_token) == (8_192, 8_192)
    assert (f.total_features, f.candidates_per_token) == (131_072, 131_072)
    assert a.parameter_count == f.parameter_count == 738_328_576
    assert b.parameter_count == 46_145_536


def test_positive_topk_is_strict_and_breaks_cutoff_ties_by_global_id() -> None:
    # Three equal maxima: global ids 2 and 4 must win despite shuffled input.
    scores = jnp.asarray([[3.0, 3.0, 1.0, 3.0, 0.0]])
    ids = jnp.asarray([[9, 4, 1, 2, 0]], jnp.int32)
    selected = positive_topk(scores, ids, 2)
    np.testing.assert_array_equal(selected.latent_ids, [[2, 4]])
    np.testing.assert_allclose(selected.values, [[3.0, 3.0]])
    assert bool(selected.valid[0])

    invalid = positive_topk(jnp.asarray([[1.0, 0.0, -2.0]]), jnp.arange(3), 2)
    assert not bool(invalid.valid[0])
    np.testing.assert_array_equal(invalid.values, [[0.0, 0.0]])


def test_nonnegative_topk_keeps_genuine_relu_zeros_and_excludes_sentinels() -> None:
    sentinel = np.iinfo(np.int32).max
    scores = jnp.asarray([[0.0, 2.0, 0.0, -jnp.inf, 0.0]], jnp.float32)
    ids = jnp.asarray([[7, 9, 3, 1, sentinel]], jnp.int32)
    selected = nonnegative_topk(scores, ids, 3)

    assert bool(selected.valid[0])
    np.testing.assert_array_equal(selected.latent_ids, [[3, 7, 9]])
    np.testing.assert_array_equal(selected.values, [[0.0, 0.0, 2.0]])

    local_values = jnp.asarray(
        [[[0.0, 2.0]], [[0.0, 0.0]]],
        jnp.float32,
    )
    local_ids = jnp.asarray(
        [[[3, 9]], [[7, sentinel]]],
        jnp.int32,
    )
    merged = merge_feature_shard_topk(
        local_values,
        local_ids,
        3,
        require_strictly_positive=False,
    )
    np.testing.assert_array_equal(merged.latent_ids, [[3, 7, 9]])
    np.testing.assert_array_equal(merged.values, [[0.0, 0.0, 2.0]])


def test_feature_sharded_topk_matches_unsharded_membership() -> None:
    scores = jnp.asarray(
        [
            [1.0, 5.0, 2.0, 4.0, 3.0, 0.0, 5.0, -1.0],
            [8.0, 1.0, 4.0, 2.0, 7.0, 3.0, 6.0, 5.0],
        ]
    )
    expected = positive_topk(scores, jnp.arange(8), 3)
    local_values = []
    local_ids = []
    for start in (0, 4):
        selected = local_topk_candidates(
            scores[:, start : start + 4], jnp.arange(start, start + 4), 3
        )
        # Single-axis helper already gives that shard's candidates.  Feed its
        # selected candidates through an artificial two-shard merge below.
        local_values.append(selected.values)
        local_ids.append(selected.latent_ids)
    actual = merge_feature_shard_topk(jnp.stack(local_values), jnp.stack(local_ids), 3)
    np.testing.assert_array_equal(actual.latent_ids, expected.latent_ids)
    np.testing.assert_allclose(actual.values, expected.values)


def test_ordered_local_topk_skips_redundant_candidate_argsort() -> None:
    scores = jnp.asarray(
        [
            [3.0, 1.0, 5.0, 5.0, 2.0, 4.0],
            [0.0, 7.0, 3.0, 6.0, 1.0, 5.0],
        ],
        jnp.float32,
    )
    ids = jnp.asarray([[8, 9, 10, 11, 12, 13]], jnp.int32)
    generic = local_topk_candidates(scores, ids, 3)
    ordered = local_topk_candidates(
        scores,
        ids,
        3,
        ids_are_sorted=True,
    )
    np.testing.assert_array_equal(ordered.latent_ids, generic.latent_ids)
    np.testing.assert_allclose(ordered.values, generic.values)

    jaxpr = str(
        jax.make_jaxpr(
            lambda value: local_topk_candidates(
                value,
                ids,
                3,
                ids_are_sorted=True,
            )
        )(scores)
    )
    assert "argsort" not in jaxpr
    assert "sort[" not in jaxpr


def test_condition_a_visibility_and_grouped_plan() -> None:
    cfg = tiny_config("A")
    # Every bank has distinct positive bias, so selected ids reveal exactly
    # which routed banks were visible.  Router weights cannot affect this API.
    params = SAEParams(
        encoder=jnp.zeros((8, 3)),
        decoder=jnp.eye(3, 8),
        bias=jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8], jnp.float32),
    )
    expert_ids = np.asarray([[3, 1], [0, 2]], np.int32)
    plan = make_grouping_plan(np.sort(expert_ids, axis=1), num_experts=4, capacity=2)
    result = forward(
        params,
        jnp.ones((2, 3)),
        jnp.zeros(3),
        cfg,
        expert_ids=jnp.asarray(expert_ids),
        grouping_plan=plan,
    )
    np.testing.assert_array_equal(result.selection.latent_ids, [[6, 7], [4, 5]])
    assert np.all(np.asarray(result.selection.valid))


def test_sparse_selected_column_decode() -> None:
    decoder = jnp.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    selection = Selection(
        values=jnp.asarray([[2.0, 0.5]]),
        latent_ids=jnp.asarray([[0, 3]], jnp.int32),
        valid=jnp.asarray([True]),
    )
    reconstruction = decode_selected_columns(
        decoder, selection, jnp.asarray([0.5, -0.5])
    )
    np.testing.assert_allclose(reconstruction, [[4.5, 8.0]])


def test_train_step_keeps_decoder_columns_normalized_and_exact_l0() -> None:
    cfg = tiny_config("B")
    state = init_train_state(jax.random.PRNGKey(7), cfg)
    # A positive bias makes the strict-positive TopK precondition deterministic.
    state = state._replace(
        params=state.params._replace(bias=jnp.ones_like(state.params.bias))
    )
    activation = jnp.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [1.0, 1.0, 0.0]])
    updated, metrics = train_step(state, activation, jnp.zeros(3), cfg)
    assert float(metrics.l0) == cfg.k
    assert bool(metrics.topk_valid)
    np.testing.assert_allclose(
        jnp.linalg.norm(updated.params.decoder.astype(jnp.float32), axis=0),
        np.ones(cfg.total_features),
        atol=2e-6,
    )
    assert updated.optimizer.m.encoder.dtype == jnp.float32
    assert updated.optimizer.v.decoder.dtype == jnp.float32


def test_learning_rate_schedule_endpoints() -> None:
    cfg = tiny_config("B", total_steps=6)
    assert np.isclose(float(learning_rate(0, cfg)), cfg.peak_lr / 2)
    assert np.isclose(float(learning_rate(1, cfg)), cfg.peak_lr)
    assert np.isclose(float(learning_rate(5, cfg)), cfg.min_lr)
