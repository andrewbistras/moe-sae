from __future__ import annotations

import jax
import numpy as np
import pytest

from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_training import ContributionOptimizerConfig
from moe_sae.contribution_transcoder_distributed import (
    ContributionTranscoderCentering,
    ContributionTranscoderDistributedTrainer,
    init_contribution_transcoder_distributed_state,
    stack_contribution_transcoder_microbatches,
)
from moe_sae.distributed import DistributedConfig, create_mesh


def _layout(condition: str = "B") -> ContributionSAEConfig:
    return ContributionSAEConfig(
        condition=condition,  # type: ignore[arg-type]
        hidden_dim=4,
        num_experts=4,
        top_routed=2,
        bank_width=4,
        k_per_contribution=1,
    )


def _optimizer() -> ContributionOptimizerConfig:
    return ContributionOptimizerConfig(
        parameter_dtype="float32",
        warmup_steps=0,
        total_steps=2,
    )


def _microbatch(target_offset: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "routed_input": np.asarray(
            [[1.0, -0.5, 0.25, 0.75], [0.5, 1.0, -0.25, 0.5]],
            np.float32,
        ),
        "expert_output": np.asarray(
            [
                [[1.0, 0.5, -0.5, 0.25], [0.25, 0.5, 0.75, 1.0]],
                [[0.5, -0.5, 1.0, 0.25], [1.0, 0.0, 0.5, -0.5]],
            ],
            np.float32,
        )
        + np.float32(target_offset),
        "bank_ids": np.asarray([[0, 1], [0, 1]], np.int32),
        "router_weights": np.asarray([[0.6, 0.4], [0.55, 0.45]], np.float32),
        "valid_mask": np.asarray([True, True], np.bool_),
    }


def test_stack_has_one_source_operand_and_separate_decoder_targets():
    layout = _layout()
    distributed = DistributedConfig(
        data_axis_size=1,
        feature_axis_size=1,
        microbatches_per_update=1,
    )
    left = stack_contribution_transcoder_microbatches(
        [_microbatch(0.0)],
        layout,
        distributed,
    )
    right = stack_contribution_transcoder_microbatches(
        [_microbatch(100.0)],
        layout,
        distributed,
    )

    np.testing.assert_array_equal(left.routed_input, right.routed_input)
    assert not np.array_equal(left.expert_output, right.expert_output)
    assert left.routed_input.shape == (1, 2, 4)
    assert left.expert_output.shape == (1, 2, 2, 4)


@pytest.mark.parametrize("condition", ["A", "R", "B", "F"])
def test_cpu_distributed_strongly_negative_update_is_finite_and_exact_l0(
    condition: str,
):
    layout = _layout(condition)
    distributed = DistributedConfig(
        data_axis_size=1,
        feature_axis_size=1,
        microbatches_per_update=1,
    )
    mesh = create_mesh(distributed, devices=np.asarray(jax.devices()[:1]))
    optimizer = _optimizer()
    state = init_contribution_transcoder_distributed_state(
        jax.random.key(7),
        mesh,
        layout,
        optimizer,
        distributed,
    )
    state = state._replace(
        params=state.params._replace(
            encoder=np.zeros(state.params.encoder.shape, np.float32),
            encoder_bias=np.full(
                state.params.encoder_bias.shape,
                -1.0e6,
                np.float32,
            ),
        )
    )
    trainer = ContributionTranscoderDistributedTrainer(
        mesh,
        layout,
        optimizer,
        distributed,
        bank_capacity=2 if condition in {"A", "R"} else None,
        rematerialize_microbatch=False,
    )
    updated, metrics = trainer.step(
        state,
        [_microbatch()],
        ContributionTranscoderCentering(
            source_mean=np.zeros((4,), np.float32),
            target_mean=np.zeros((4,), np.float32),
        ),
    )

    assert int(updated.step) == 1
    assert int(updated.tokens_seen) == 2
    assert float(metrics.l0) == layout.total_l0
    assert np.isfinite(float(metrics.loss))


def test_transcoder_distributed_trainer_can_select_relu_instead_of_softplus():
    layout = _layout("A")
    distributed = DistributedConfig(
        data_axis_size=1,
        feature_axis_size=1,
        microbatches_per_update=1,
    )
    mesh = create_mesh(distributed, devices=np.asarray(jax.devices()[:1]))
    optimizer = _optimizer()
    state = init_contribution_transcoder_distributed_state(
        jax.random.key(23),
        mesh,
        layout,
        optimizer,
        distributed,
    )
    state = state._replace(
        params=state.params._replace(
            encoder=np.zeros(state.params.encoder.shape, np.float32),
            encoder_bias=np.ones(state.params.encoder_bias.shape, np.float32),
        )
    )
    trainer = ContributionTranscoderDistributedTrainer(
        mesh,
        layout,
        optimizer,
        distributed,
        bank_capacity=2,
        rematerialize_microbatch=False,
        encoder_activation="relu",
    )
    _updated, metrics = trainer.step(
        state,
        [_microbatch()],
        ContributionTranscoderCentering(
            source_mean=np.zeros((4,), np.float32),
            target_mean=np.zeros((4,), np.float32),
        ),
    )

    assert trainer.encoder_activation == "relu"
    assert float(metrics.l0) == layout.total_l0
    assert np.isfinite(float(metrics.loss))
    assert bool(metrics.topk_valid)


def test_transcoder_relu_inclusive_zero_topk_survives_support_collapse():
    layout = _layout("A")
    distributed = DistributedConfig(
        data_axis_size=1,
        feature_axis_size=1,
        microbatches_per_update=1,
    )
    mesh = create_mesh(distributed, devices=np.asarray(jax.devices()[:1]))
    optimizer = _optimizer()
    state = init_contribution_transcoder_distributed_state(
        jax.random.key(29),
        mesh,
        layout,
        optimizer,
        distributed,
    )
    state = state._replace(
        params=state.params._replace(
            encoder=np.zeros(state.params.encoder.shape, np.float32),
            encoder_bias=np.full(
                state.params.encoder_bias.shape,
                -1.0e6,
                np.float32,
            ),
        )
    )
    trainer = ContributionTranscoderDistributedTrainer(
        mesh,
        layout,
        optimizer,
        distributed,
        bank_capacity=2,
        rematerialize_microbatch=False,
        encoder_activation="relu",
        topk_requires_strictly_positive=False,
    )
    updated, metrics = trainer.step(
        state,
        [_microbatch()],
        ContributionTranscoderCentering(
            source_mean=np.zeros((4,), np.float32),
            target_mean=np.zeros((4,), np.float32),
        ),
    )

    assert int(updated.step) == 1
    assert bool(metrics.topk_valid)
    assert float(metrics.l0) == 0.0
    assert not np.asarray(updated.fired_mask).any()
    assert np.isfinite(float(metrics.loss))


def test_transcoder_softplus_cannot_disable_strict_positive_selection():
    layout = _layout("A")
    distributed = DistributedConfig(
        data_axis_size=1,
        feature_axis_size=1,
        microbatches_per_update=1,
    )
    mesh = create_mesh(distributed, devices=np.asarray(jax.devices()[:1]))

    with pytest.raises(
        ValueError,
        match="softplus must retain",
    ):
        ContributionTranscoderDistributedTrainer(
            mesh,
            layout,
            _optimizer(),
            distributed,
            bank_capacity=2,
            encoder_activation="softplus",
            topk_requires_strictly_positive=False,
        )
