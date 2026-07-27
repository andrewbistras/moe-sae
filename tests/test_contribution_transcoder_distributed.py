from __future__ import annotations

import jax
import numpy as np

from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_training import ContributionOptimizerConfig
from moe_sae.contribution_transcoder_distributed import (
    ContributionTranscoderCentering,
    ContributionTranscoderDistributedTrainer,
    init_contribution_transcoder_distributed_state,
    stack_contribution_transcoder_microbatches,
)
from moe_sae.distributed import DistributedConfig, create_mesh


def _layout() -> ContributionSAEConfig:
    return ContributionSAEConfig(
        condition="B",
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


def test_cpu_distributed_transcoder_update_is_finite_and_exact_l0():
    layout = _layout()
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
    # A positive encoder bias makes the exact-positive TopK precondition
    # deterministic for this tiny production-shaped update.
    state = state._replace(
        params=state.params._replace(
            encoder_bias=np.ones(state.params.encoder_bias.shape, np.float32)
        )
    )
    trainer = ContributionTranscoderDistributedTrainer(
        mesh,
        layout,
        optimizer,
        distributed,
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
