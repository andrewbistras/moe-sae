"""Production-shaped distributed tests for Experiment 5."""

from __future__ import annotations

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moe_sae.contribution_training import ContributionOptimizerConfig
from moe_sae.distributed import DistributedConfig, create_mesh
from moe_sae.experiment5_distributed import (
    Experiment5DistributedTrainer,
    experiment5_physical_layout,
    experiment5_physical_to_logical,
    pack_experiment5_train_state,
    require_experiment5_physical_layout,
    shard_experiment5_train_state,
    stack_experiment5_distributed_microbatches,
    unpack_experiment5_train_state,
)
from moe_sae.experiment5_hybrid import (
    Experiment5Batch,
    Experiment5Centering,
    Experiment5Layout,
    Experiment5Recomposition,
    experiment5_train_step,
    init_experiment5_train_state,
)


def _layout(condition: str) -> Experiment5Layout:
    return Experiment5Layout(
        condition=condition,  # type: ignore[arg-type]
        hidden_dim=3,
        num_experts=4,
        top_routed=2,
        routed_bank_width=2,
        routed_k=1,
        shared_width=4,
        shared_k=2,
        randomization_key=17,
    )


def _optimizer() -> ContributionOptimizerConfig:
    return ContributionOptimizerConfig(
        parameter_dtype="float32",
        peak_lr=1e-3,
        min_lr=1e-3,
        warmup_steps=0,
        total_steps=2,
    )


def _batch() -> Experiment5Batch:
    return Experiment5Batch(
        routed_input=np.asarray(
            [[1.0, 0.2, -0.5], [0.3, -0.7, 0.1]],
            np.float32,
        ),
        expert_output=(
            np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 10.0
            + 0.1
        ),
        shared_normalized=np.asarray(
            [[0.2, 0.3, -0.1], [0.3, -0.2, 0.4]],
            np.float32,
        ),
        bank_ids=np.asarray([[0, 1], [2, 3]], np.int32),
        router_weights=np.asarray(
            [[0.6, 0.4], [0.25, 0.75]],
            np.float32,
        ),
        valid_mask=np.ones((2,), np.bool_),
    )


def _centering() -> Experiment5Centering:
    return Experiment5Centering(
        source_mean=np.asarray([0.1, -0.2, 0.05], np.float32),
        routed_target_mean=np.asarray([0.2, 0.1, -0.1], np.float32),
        shared_target_mean=np.asarray([-0.1, 0.2, 0.05], np.float32),
        full_target_mean=np.asarray([0.0, 0.1, -0.2], np.float32),
    )


def _recomposition() -> Experiment5Recomposition:
    return Experiment5Recomposition(
        routed_scale=np.asarray([0.75, 1.0, 1.25], np.float32),
        outer_scale=np.asarray([1.125, 0.875, 1.0], np.float32),
        epsilon=1e-6,
    )


def _assert_state_close(actual, expected, *, atol: float = 2e-4) -> None:
    # Grouped bank GEMMs and the dense oracle contract along different axes.
    # Their selected IDs/loss are exact, while FP32 reverse-mode accumulation
    # can differ slightly after the two BF16 RMSNorm round points.
    for observed, reference in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        observed_array = np.asarray(observed)
        reference_array = np.asarray(reference)
        if observed_array.dtype == np.bool_:
            np.testing.assert_array_equal(observed_array, reference_array)
        else:
            np.testing.assert_allclose(
                observed_array,
                reference_array,
                rtol=2e-5,
                atol=atol,
            )


def test_balanced_physical_layout_is_bijective_and_mesh_bound() -> None:
    selected = _layout("H")
    permutation = experiment5_physical_to_logical(selected, 2)
    physical = experiment5_physical_layout(selected, 2)

    # Each physical shard is two whole routed banks followed by two shared
    # features. Logical IDs remain canonical.
    np.testing.assert_array_equal(
        permutation,
        np.asarray([0, 1, 2, 3, 8, 9, 4, 5, 6, 7, 10, 11], np.int32),
    )
    assert physical.routed_banks_per_shard == 2
    assert physical.routed_features_per_shard == 4
    assert physical.shared_features_per_shard == 2
    assert len(physical.identity_sha256) == 64
    assert (
        experiment5_physical_layout(selected, 1).identity_sha256
        != physical.identity_sha256
    )
    require_experiment5_physical_layout(
        physical.identity_sha256,
        physical,
    )
    with pytest.raises(RuntimeError, match="offline repack"):
        require_experiment5_physical_layout("0" * 64, physical)


def test_pack_unpack_round_trip_includes_moments_and_firing_mask() -> None:
    selected = _layout("H")
    state = init_experiment5_train_state(
        jax.random.key(2),
        selected,
        _optimizer(),
    )
    state = state._replace(
        fired_mask=jnp.arange(selected.total_features) % 3 == 0,
        optimizer=state.optimizer._replace(
            m=state.optimizer.m._replace(
                bias=jnp.arange(selected.total_features, dtype=jnp.float32)
            )
        ),
    )
    packed = pack_experiment5_train_state(state, selected, 2)
    restored = unpack_experiment5_train_state(packed, selected, 2)
    _assert_state_close(restored, state, atol=0)


def test_stack_preserves_all_existing_capture_and_overlay_operands() -> None:
    selected = _layout("H")
    distributed = DistributedConfig(microbatches_per_update=2)
    stacked = stack_experiment5_distributed_microbatches(
        [_batch(), _batch()],
        selected,
        distributed,
    )

    assert stacked.routed_input.shape == (2, 2, 3)
    assert stacked.expert_output.shape == (2, 2, 2, 3)
    assert stacked.shared_normalized.shape == (2, 2, 3)
    assert stacked.bank_ids.shape == (2, 2, 2)
    assert stacked.router_weights.shape == (2, 2, 2)
    assert stacked.valid_mask.shape == (2, 2)


@pytest.mark.parametrize("condition", ["H", "R", "B", "F"])
@pytest.mark.parametrize("encoder_kind", ["sae", "transcoder"])
def test_single_device_update_matches_jitted_scientific_core(
    condition: str,
    encoder_kind: str,
) -> None:
    selected = _layout(condition)
    optimizer = _optimizer()
    distributed = DistributedConfig(
        data_axis_size=1,
        feature_axis_size=1,
        microbatches_per_update=1,
    )
    mesh = create_mesh(distributed, devices=np.asarray(jax.devices()[:1]))
    state = init_experiment5_train_state(
        jax.random.key(3),
        selected,
        optimizer,
    )
    state = state._replace(
        params=state.params._replace(
            bias=3.0 * jnp.ones_like(state.params.bias),
        )
    )
    source = _batch()
    reference, reference_metrics = jax.jit(
        lambda current: experiment5_train_step(
            current,
            source,
            _centering(),
            _recomposition(),
            selected,
            optimizer,
            encoder_kind=encoder_kind,  # type: ignore[arg-type]
        )
    )(state)
    trainer = Experiment5DistributedTrainer(
        mesh,
        selected,
        optimizer,
        distributed,
        encoder_kind=encoder_kind,  # type: ignore[arg-type]
        bank_capacity=1 if selected.is_hybrid else None,
        rematerialize_microbatch=False,
    )
    actual, metrics = trainer.step(
        shard_experiment5_train_state(
            state,
            mesh,
            selected,
            optimizer,
            distributed,
        ),
        [source],
        _centering(),
        _recomposition(),
    )

    _assert_state_close(actual, reference)
    np.testing.assert_allclose(
        metrics.loss,
        reference_metrics.loss,
        rtol=2e-5,
        atol=6e-6,
    )
    assert int(metrics.valid_tokens) == 2
    assert float(metrics.l0) == selected.total_l0
    assert bool(metrics.topk_valid)


def test_grouped_capacity_overflow_is_a_hard_error() -> None:
    selected = _layout("H")
    optimizer = _optimizer()
    distributed = DistributedConfig(
        data_axis_size=1,
        feature_axis_size=1,
        microbatches_per_update=1,
    )
    mesh = create_mesh(distributed, devices=np.asarray(jax.devices()[:1]))
    state = init_experiment5_train_state(
        jax.random.key(5),
        selected,
        optimizer,
    )
    state = state._replace(
        params=state.params._replace(
            bias=3.0 * jnp.ones_like(state.params.bias),
        )
    )
    repeated = _batch()._replace(
        bank_ids=np.asarray([[0, 1], [0, 1]], np.int32)
    )
    trainer = Experiment5DistributedTrainer(
        mesh,
        selected,
        optimizer,
        distributed,
        encoder_kind="sae",
        bank_capacity=1,
        rematerialize_microbatch=False,
    )

    with pytest.raises(RuntimeError, match="exceeded.*capacity"):
        trainer.step(
            shard_experiment5_train_state(
                state,
                mesh,
                selected,
                optimizer,
                distributed,
            ),
            [repeated],
            _centering(),
            _recomposition(),
        )


@pytest.mark.parametrize(
    ("encoder_kind", "condition"),
    [
        ("sae", "H"),
        ("sae", "R"),
        ("sae", "B"),
        ("sae", "F"),
        ("transcoder", "H"),
    ],
)
def test_softplus_retains_exact_l0_with_negative_preactivations(
    encoder_kind: str,
    condition: str,
) -> None:
    selected = _layout(condition)
    optimizer = _optimizer()
    distributed = DistributedConfig(
        data_axis_size=1,
        feature_axis_size=1,
        microbatches_per_update=1,
    )
    mesh = create_mesh(distributed, devices=np.asarray(jax.devices()[:1]))
    state = init_experiment5_train_state(
        jax.random.key(8),
        selected,
        optimizer,
    )
    state = state._replace(
        params=state.params._replace(
            encoder=jnp.zeros_like(state.params.encoder),
            bias=jnp.full_like(state.params.bias, -1.0e6),
        )
    )
    trainer = Experiment5DistributedTrainer(
        mesh,
        selected,
        optimizer,
        distributed,
        encoder_kind=encoder_kind,  # type: ignore[arg-type]
        bank_capacity=1 if selected.is_hybrid else None,
        rematerialize_microbatch=False,
    )
    updated, metrics = trainer.step(
        shard_experiment5_train_state(
            state,
            mesh,
            selected,
            optimizer,
            distributed,
        ),
        [_batch()],
        _centering(),
        _recomposition(),
    )

    assert int(updated.step) == 1
    assert float(metrics.l0) == selected.total_l0
    assert bool(metrics.topk_valid)
    assert np.isfinite(float(metrics.loss))


def test_two_feature_shards_match_canonical_pack_and_unpacked_update() -> None:
    """Exercise balanced routed/shared work and real feature collectives."""

    script = r"""
import jax
import jax.numpy as jnp
import numpy as np

from moe_sae.contribution_training import ContributionOptimizerConfig
from moe_sae.distributed import DistributedConfig, create_mesh
from moe_sae.experiment5_distributed import (
    Experiment5DistributedTrainer,
    shard_experiment5_train_state,
    unpack_experiment5_train_state,
)
from moe_sae.experiment5_hybrid import (
    Experiment5Batch,
    Experiment5Centering,
    Experiment5Layout,
    Experiment5Recomposition,
    experiment5_train_step,
    init_experiment5_train_state,
)

assert jax.device_count() == 2
distributed = DistributedConfig(
    data_axis_size=1,
    feature_axis_size=2,
    microbatches_per_update=1,
)
mesh = create_mesh(distributed)
optimizer = ContributionOptimizerConfig(
    parameter_dtype="float32",
    peak_lr=1e-3,
    min_lr=1e-3,
    warmup_steps=0,
    total_steps=2,
)
source = Experiment5Batch(
    routed_input=np.asarray([[1.0, 0.2], [0.3, -0.7]], np.float32),
    expert_output=np.asarray(
        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
        np.float32,
    ),
    shared_normalized=np.asarray([[0.2, -0.1], [0.3, 0.4]], np.float32),
    bank_ids=np.asarray([[0, 1], [2, 3]], np.int32),
    router_weights=np.asarray([[0.6, 0.4], [0.25, 0.75]], np.float32),
    valid_mask=np.ones((2,), np.bool_),
)
centering = Experiment5Centering(*([np.zeros((2,), np.float32)] * 4))
recomposition = Experiment5Recomposition(
    np.ones((2,), np.float32),
    np.ones((2,), np.float32),
)

for condition in ("H", "B"):
    layout = Experiment5Layout(
        condition=condition,
        hidden_dim=2,
        num_experts=4,
        top_routed=2,
        routed_bank_width=2,
        routed_k=1,
        shared_width=4,
        shared_k=2,
    )
    for encoder_kind in ("sae", "transcoder"):
        state = init_experiment5_train_state(
            jax.random.key(4),
            layout,
            optimizer,
        )
        state = state._replace(
            params=state.params._replace(
                bias=3.0 * jnp.ones_like(state.params.bias),
            )
        )
        reference, reference_metrics = jax.jit(
            lambda current: experiment5_train_step(
                current,
                source,
                centering,
                recomposition,
                layout,
                optimizer,
                encoder_kind=encoder_kind,
            )
        )(state)
        trainer = Experiment5DistributedTrainer(
            mesh,
            layout,
            optimizer,
            distributed,
            encoder_kind=encoder_kind,
            bank_capacity=1 if condition == "H" else None,
            rematerialize_microbatch=False,
        )
        physical, metrics = trainer.step(
            shard_experiment5_train_state(
                state,
                mesh,
                layout,
                optimizer,
                distributed,
            ),
            [source],
            centering,
            recomposition,
        )
        actual = unpack_experiment5_train_state(physical, layout, 2)
        for observed, expected in zip(
            jax.tree_util.tree_leaves(actual),
            jax.tree_util.tree_leaves(reference),
            strict=True,
        ):
            observed_array = np.asarray(observed)
            expected_array = np.asarray(expected)
            if observed_array.dtype == np.bool_:
                np.testing.assert_array_equal(observed_array, expected_array)
            else:
                np.testing.assert_allclose(
                    observed_array,
                    expected_array,
                    rtol=2e-5,
                    atol=1.2e-4,
                )
        np.testing.assert_allclose(
            metrics.loss,
            reference_metrics.loss,
            rtol=2e-5,
            atol=8e-6,
        )
        assert float(metrics.l0) == layout.total_l0
"""
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
