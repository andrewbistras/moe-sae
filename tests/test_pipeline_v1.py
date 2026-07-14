from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moe_sae.artifacts import CaptureBatch, float32_to_bf16_bits
from moe_sae.pipeline import (
    ActivationBatch,
    TrainingConfig,
    check_seed_health,
    evaluate,
    load_checkpoint,
    planned_token_mask,
    train,
)
from moe_sae.sae import SAEConfig, init_train_state


def config(condition: str = "B") -> TrainingConfig:
    sae = SAEConfig(
        condition,
        hidden_dim=4,
        num_experts=4,
        top_routed=2,
        bank_width=2,
        k=1,
        parameter_dtype="float32",
        warmup_steps=1,
        total_steps=2,
    )
    return TrainingConfig(
        sae,
        seed=11,
        total_tokens=6,
        effective_batch_tokens=4,
        microbatch_tokens=2,
        checkpoint_every_tokens=4,
        health_gate_tokens=4,
        live_feature_fraction=0.0,
        loss_ratio_threshold=2.0,
    )


def initial_state(cfg: TrainingConfig):
    state = init_train_state(jax.random.PRNGKey(cfg.seed), cfg.sae)
    # Guarantee at least k strictly positive candidates for every data row.
    return state._replace(
        params=state.params._replace(bias=jnp.ones_like(state.params.bias))
    )


def batches() -> list[ActivationBatch]:
    x = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, -0.5],
            [0.0, 0.0, 1.0, 0.25],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, -0.25],
        ],
        dtype=np.float32,
    )
    # Extra fields are intentionally ignored, including router weights.
    return [ActivationBatch(x)]


def test_production_final_mask() -> None:
    assert planned_token_mask(3050).sum() == 16_384
    final = planned_token_mask(3051)
    assert final.sum() == 12_416
    assert (~final).sum() == 3_968
    assert planned_token_mask(3052).sum() == 0


def test_train_handles_partial_final_effective_batch_and_checkpoints(
    tmp_path: Path,
) -> None:
    cfg = config()
    result = train(
        cfg,
        batches(),
        np.zeros(4, np.float32),
        state=initial_state(cfg),
        checkpoint_dir=tmp_path,
    )
    assert int(result.state.step) == 2
    assert int(result.state.tokens_seen) == 6
    assert [metric.valid_tokens for metric in result.metrics] == [4, 2]
    assert all(metric.l0 == 1.0 for metric in result.metrics)
    assert result.latest_checkpoint == tmp_path / "latest.npz"
    assert result.final_checkpoint == tmp_path / "final.npz"
    assert result.latest_checkpoint.exists()
    assert result.final_checkpoint.exists()


def test_resume_matches_uninterrupted_updates(tmp_path: Path) -> None:
    cfg = config()
    start = initial_state(cfg)
    uninterrupted = train(cfg, batches(), np.zeros(4, np.float32), state=start)
    partial = train(
        cfg,
        batches(),
        np.zeros(4, np.float32),
        state=start,
        checkpoint_dir=tmp_path,
        stop_after_tokens=4,
    )
    loaded, _ = load_checkpoint(tmp_path / "latest.npz", cfg)
    assert int(partial.state.tokens_seen) == 4
    resumed = train(cfg, batches(), np.zeros(4, np.float32), state=loaded)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(resumed.state),
        jax.tree_util.tree_leaves(uninterrupted.state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_evaluation_accumulates_common_sse_sst_and_exact_l0() -> None:
    cfg = config()
    state = initial_state(cfg)
    result = evaluate(
        state.params,
        cfg,
        batches(),
        np.zeros(4, np.float32),
        max_tokens=6,
    )
    assert result.tokens == 6
    assert result.l0 == 1.0
    assert result.topk_valid
    assert np.isclose(result.explained_variance, 1.0 - result.nmse)
    assert result.sst == result.computed_sst

    pinned = evaluate(
        state.params,
        cfg,
        batches(),
        np.zeros(4, np.float32),
        max_tokens=6,
        expected_sst=result.computed_sst * 2,
    )
    assert np.isclose(pinned.nmse, result.nmse / 2)


def test_health_gate_checks_loss_and_live_features() -> None:
    cfg = config("A")
    state = initial_state(cfg)._replace(
        tokens_seen=jnp.asarray(4, jnp.int32),
        fired_mask=jnp.ones((cfg.sae.total_features,), bool),
        loss_ema=jnp.asarray(0.7, jnp.float32),
        initial_loss_ema=jnp.asarray(1.0, jnp.float32),
        ema_updates=jnp.asarray(64, jnp.int32),
    )
    health = check_seed_health(state, cfg, l0=1.0)
    assert health.passed
    assert health.loss_ratio < cfg.loss_ratio_threshold
    assert health.live_fraction == 1.0


def test_training_fails_closed_when_fewer_than_k_scores_are_positive() -> None:
    cfg = config()
    state = initial_state(cfg)
    state = state._replace(
        params=state.params._replace(
            encoder=jnp.zeros_like(state.params.encoder),
            bias=jnp.zeros_like(state.params.bias),
        )
    )
    with pytest.raises(RuntimeError, match="fewer than k"):
        train(cfg, batches(), np.zeros(4, np.float32), state=state)


def test_capture_bf16_wire_batch_is_decoded_before_training() -> None:
    cfg = config()
    x = batches()[0].activation
    capture_batch = CaptureBatch(
        activation=float32_to_bf16_bits(x),
        expert_ids=np.zeros((6, 2), np.uint8),
        router_weights=float32_to_bf16_bits(np.full((6, 2), 0.5, np.float32)),
        sequence_id=np.arange(6, dtype=np.int64),
        position=np.arange(6, dtype=np.uint16),
    )
    result = train(
        cfg,
        [capture_batch],
        np.zeros(4, np.float32),
        state=initial_state(cfg),
    )
    assert int(result.state.tokens_seen) == 6
    assert all(np.isfinite(metric.loss) for metric in result.metrics)


def test_condition_a_padding_routes_are_balanced_not_capacity_spikes() -> None:
    from moe_sae.pipeline import _fixed_microbatches

    sae = SAEConfig(
        "A",
        hidden_dim=4,
        num_experts=8,
        top_routed=2,
        bank_width=2,
        k=1,
        parameter_dtype="float32",
        warmup_steps=1,
        total_steps=1,
    )
    cfg = TrainingConfig(
        sae,
        seed=1,
        total_tokens=1,
        effective_batch_tokens=8,
        microbatch_tokens=4,
        checkpoint_every_tokens=8,
        health_gate_tokens=1,
    )
    source = ActivationBatch(
        np.ones((1, 4), np.float32), np.asarray([[0, 1]], np.int32)
    )
    packed = list(_fixed_microbatches([source], cfg, skip_tokens=0, take_tokens=1))
    assert len(packed) == 2
    padded_ids = np.concatenate(
        [batch.expert_ids[~batch.valid_mask] for batch in packed], axis=0
    )
    loads = np.bincount(padded_ids.reshape(-1), minlength=sae.num_experts)
    assert loads.max() - loads.min() <= 1
    assert loads.max() < padded_ids.shape[0]


def test_low_rank_synthetic_training_has_finite_falling_loss() -> None:
    sae = SAEConfig(
        "B",
        hidden_dim=4,
        num_experts=4,
        top_routed=2,
        bank_width=4,
        k=2,
        parameter_dtype="float32",
        warmup_steps=5,
        total_steps=40,
        peak_lr=1e-2,
        min_lr=1e-3,
    )
    cfg = TrainingConfig(
        sae,
        seed=3,
        total_tokens=2_560,
        effective_batch_tokens=64,
        microbatch_tokens=16,
        checkpoint_every_tokens=1_024,
        health_gate_tokens=128,
        live_feature_fraction=0.0,
        loss_ratio_threshold=2.0,
    )
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(cfg.total_tokens, 2)).astype(np.float32)
    basis = np.asarray([[1.0, 0.2, -0.4, 0.8], [-0.3, 1.0, 0.7, 0.1]], np.float32)
    activation = latent @ basis
    state = init_train_state(jax.random.PRNGKey(cfg.seed), sae)
    state = state._replace(
        params=state.params._replace(bias=jnp.full_like(state.params.bias, 0.5))
    )

    result = train(
        cfg,
        [ActivationBatch(activation)],
        np.zeros(4, np.float32),
        state=state,
    )
    losses = np.asarray([metric.loss for metric in result.metrics])

    assert np.isfinite(losses).all()
    assert all(metric.l0 == sae.k and metric.topk_valid for metric in result.metrics)
    assert losses[-8:].mean() < 0.4 * losses[:8].mean()
