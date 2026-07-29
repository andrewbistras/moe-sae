"""Parity tests for the TPU-native Experiment-5 evaluation boundary."""

from __future__ import annotations

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from moe_sae.contribution_training import ContributionOptimizerConfig
from moe_sae.distributed import DistributedConfig, create_mesh
from moe_sae.experiment5_device_evaluation import (
    Experiment5CanonicalTargetEvaluator,
    Experiment5CanonicalTargetScores,
    Experiment5DistributedEvaluator,
    process_local_experiment5_device_scores,
)
from moe_sae.experiment5_distributed import shard_experiment5_train_state
from moe_sae.experiment5_hybrid import (
    Experiment5Batch,
    Experiment5Centering,
    Experiment5Layout,
    Experiment5Recomposition,
    forward_experiment5,
    init_experiment5_train_state,
    jax_experiment5_routed_branch,
    jax_recompose_experiment5_full_ffn,
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


def _batch(*, invalid_tail: bool = False) -> Experiment5Batch:
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
        valid_mask=np.asarray([True, not invalid_tail], np.bool_),
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


def _state(layout: Experiment5Layout):
    state = init_experiment5_train_state(
        jax.random.key(31),
        layout,
        _optimizer(),
    )
    return state._replace(
        params=state.params._replace(
            bias=3.0 * jnp.ones_like(state.params.bias),
        )
    )


def _sse(target: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    return np.sum(
        np.square(
            target.astype(np.float32) - estimate.astype(np.float32),
            dtype=np.float32,
        ),
        axis=-1,
        dtype=np.float32,
    )


def _sse_jax(target: jax.Array, estimate: jax.Array) -> jax.Array:
    return jnp.sum(
        jax.lax.square(
            target.astype(jnp.float32) - estimate.astype(jnp.float32)
        ),
        axis=-1,
        dtype=jnp.float32,
    )


@pytest.mark.parametrize("condition", ["H", "R", "B", "F"])
@pytest.mark.parametrize("encoder_kind", ["sae", "transcoder"])
def test_device_rows_match_scientific_forward(
    condition: str,
    encoder_kind: str,
) -> None:
    layout = _layout(condition)
    distributed = DistributedConfig(
        data_axis_size=1,
        feature_axis_size=1,
        microbatches_per_update=1,
    )
    mesh = create_mesh(distributed, devices=np.asarray(jax.devices()[:1]))
    state = _state(layout)
    batch = _batch()
    evaluator = Experiment5DistributedEvaluator(
        mesh,
        layout,
        distributed,
        encoder_kind=encoder_kind,  # type: ignore[arg-type]
        bank_capacity=1 if layout.is_hybrid else None,
    )
    canonical = evaluator.canonical_target_scores(
        [batch],
        _centering(),
        _recomposition(),
    )
    canonical_target = canonical.full_target[0]
    canonical_sst = canonical.full_sst[0]

    def reference_objective(params):
        output = forward_experiment5(
            params,
            batch,
            _centering(),
            _recomposition(),
            layout,
            encoder_kind=encoder_kind,  # type: ignore[arg-type]
            canonical_full_target=canonical_target,
        )
        target_value = canonical_target
        full_rows = jnp.sum(
            jax.lax.square(
                target_value.astype(jnp.float32)
                - output.full_reconstruction.astype(jnp.float32)
            ),
            axis=1,
            dtype=jnp.float32,
        )
        sst_rows = canonical_sst
        if layout.is_hybrid:
            assert output.routed_reconstruction is not None
            assert output.shared_reconstruction is not None
            target_routed_value = jax_experiment5_routed_branch(
                batch.expert_output,
                batch.router_weights,
                _recomposition(),
            )
            estimated_routed_value = jax_experiment5_routed_branch(
                output.routed_reconstruction,
                batch.router_weights,
                _recomposition(),
            )
            oracle_shared_value = jax_recompose_experiment5_full_ffn(
                output.routed_reconstruction,
                batch.shared_normalized,
                batch.router_weights,
                _recomposition(),
            )
            oracle_routed_value = jax_recompose_experiment5_full_ffn(
                batch.expert_output,
                output.shared_reconstruction,
                batch.router_weights,
                _recomposition(),
            )
            diagnostics = (
                _sse_jax(target_routed_value, estimated_routed_value),
                _sse_jax(
                    jnp.asarray(batch.shared_normalized),
                    output.shared_reconstruction,
                ),
                _sse_jax(target_value, oracle_shared_value),
                _sse_jax(target_value, oracle_routed_value),
                target_routed_value,
            )
        else:
            diagnostics = ()
        return jnp.sum(full_rows), (
            output,
            target_value,
            full_rows,
            sst_rows,
            diagnostics,
        )

    (_, reference_aux), _ = jax.jit(
        jax.value_and_grad(reference_objective, has_aux=True)
    )(state.params)
    (
        reference,
        target,
        reference_full_sse,
        reference_full_sst,
        diagnostics,
    ) = reference_aux
    physical = shard_experiment5_train_state(
        state,
        mesh,
        layout,
        _optimizer(),
        distributed,
    )
    compact = process_local_experiment5_device_scores(
        evaluator.score(
            physical,
            [batch],
            _centering(),
            _recomposition(),
        )
    )

    target = np.asarray(target)
    np.testing.assert_allclose(
        compact.full_sse,
        np.asarray(reference_full_sse),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        compact.full_sst,
        np.asarray(reference_full_sst),
        rtol=1e-6,
        atol=1e-6,
    )
    if layout.is_hybrid:
        assert reference.routed_selection is not None
        assert reference.shared_selection is not None
        assert reference.routed_reconstruction is not None
        assert reference.shared_reconstruction is not None
        expected_ids = np.concatenate(
            (
                np.asarray(reference.routed_selection.latent_ids).reshape(
                    batch.valid_mask.shape[0],
                    layout.top_routed * layout.routed_k,
                ),
                np.asarray(reference.shared_selection.latent_ids),
            ),
            axis=1,
        )
        np.testing.assert_array_equal(compact.selected_latent_ids, expected_ids)
        np.testing.assert_array_equal(compact.bank_ids, batch.bank_ids)
        (
            expected_routed_sse,
            expected_shared_sse,
            expected_oracle_shared_sse,
            expected_oracle_routed_sse,
            target_routed,
        ) = diagnostics
        target_routed = np.asarray(target_routed)
        np.testing.assert_allclose(
            compact.routed_sse,
            expected_routed_sse,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            compact.shared_sse,
            expected_shared_sse,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            compact.oracle_shared_full_sse,
            expected_oracle_shared_sse,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            compact.oracle_routed_full_sse,
            expected_oracle_routed_sse,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            compact.routed_target_sum,
            target_routed.sum(axis=0, dtype=np.float64),
            # XLA may reassociate the eight-term routed contraction by one
            # BF16 ULP between the dense oracle and shard_map graphs.
            rtol=2e-3,
            atol=4e-3,
        )
        np.testing.assert_allclose(
            compact.shared_target_sum,
            np.asarray(batch.shared_normalized).sum(
                axis=0,
                dtype=np.float64,
            ),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            compact.routed_target_squared_norm_sum,
            np.square(target_routed.astype(np.float32)).sum(dtype=np.float64),
            rtol=2e-3,
            atol=8e-3,
        )
        np.testing.assert_allclose(
            compact.shared_target_squared_norm_sum,
            np.square(
                np.asarray(batch.shared_normalized, dtype=np.float32)
            ).sum(dtype=np.float64),
            rtol=1e-6,
            atol=1e-6,
        )
    else:
        assert reference.global_selection is not None
        np.testing.assert_array_equal(
            compact.selected_latent_ids,
            np.asarray(reference.global_selection.latent_ids),
        )
        assert compact.bank_ids is None
        assert compact.routed_sse is None
        assert compact.shared_sse is None
        assert compact.oracle_shared_full_sse is None
        assert compact.oracle_routed_full_sse is None
        assert compact.routed_target_sum is None
        assert compact.routed_target_squared_norm_sum is None
        assert compact.shared_target_sum is None
        assert compact.shared_target_squared_norm_sum is None


def test_one_materialized_target_is_bitwise_common_across_layouts() -> None:
    distributed = DistributedConfig(
        data_axis_size=1,
        feature_axis_size=1,
        microbatches_per_update=1,
    )
    mesh = create_mesh(distributed, devices=np.asarray(jax.devices()[:1]))
    common = Experiment5CanonicalTargetEvaluator(
        mesh,
        distributed,
        hidden_dim=3,
        top_routed=2,
    )
    target_rows: list[np.ndarray] = []
    sst_rows: list[np.ndarray] = []
    condition_sst_rows: list[np.ndarray] = []

    for condition in ("H", "R", "B", "F"):
        layout = _layout(condition)
        evaluator = Experiment5DistributedEvaluator(
            mesh,
            layout,
            distributed,
            encoder_kind="sae",
            bank_capacity=1 if layout.is_hybrid else None,
            canonical_target_evaluator=common,
        )
        canonical = evaluator.canonical_target_scores(
            [_batch()],
            _centering(),
            _recomposition(),
        )
        target_rows.append(np.asarray(canonical.full_target))
        sst_rows.append(np.asarray(canonical.full_sst))
        physical = shard_experiment5_train_state(
            _state(layout),
            mesh,
            layout,
            _optimizer(),
            distributed,
        )
        condition_scores = process_local_experiment5_device_scores(
            evaluator.score(
                physical,
                [_batch()],
                _centering(),
                _recomposition(),
            )
        )
        condition_sst_rows.append(condition_scores.full_sst)

    assert common.compile_count == 1
    for target in target_rows[1:]:
        np.testing.assert_array_equal(target, target_rows[0])
    for sst in (*sst_rows[1:], *condition_sst_rows):
        np.testing.assert_array_equal(
            np.asarray(sst).reshape(-1),
            sst_rows[0].reshape(-1),
        )


@pytest.mark.parametrize("condition", ["H", "R", "B", "F"])
def test_condition_scorer_consumes_injected_canonical_basis(
    condition: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distributed = DistributedConfig(
        data_axis_size=1,
        feature_axis_size=1,
        microbatches_per_update=1,
    )
    mesh = create_mesh(distributed, devices=np.asarray(jax.devices()[:1]))
    layout = _layout(condition)
    batch = _batch()
    state = _state(layout)
    state = state._replace(
        params=state.params._replace(
            decoder=jnp.zeros_like(state.params.decoder),
        )
    )
    zero_centering = Experiment5Centering(
        *(np.zeros((layout.hidden_dim,), dtype=np.float32) for _ in range(4))
    )
    common = Experiment5CanonicalTargetEvaluator(
        mesh,
        distributed,
        hidden_dim=layout.hidden_dim,
        top_routed=layout.top_routed,
    )
    target_sharding = NamedSharding(mesh, P(None, "data", None))
    sst_sharding = NamedSharding(mesh, P(None, "data"))
    injected_target_value = [-2.0]
    injected_sst_value = [7.0]

    def injected_scores(
        placed_batch,
        _full_target_mean,
        _routed_scale,
        _outer_scale,
    ) -> Experiment5CanonicalTargetScores:
        target = jax.device_put(
            jnp.full(
                (*placed_batch.expert_output.shape[:2], layout.hidden_dim),
                injected_target_value[0],
                dtype=jnp.bfloat16,
            ),
            target_sharding,
        )
        sst = jax.device_put(
            jnp.full(
                placed_batch.expert_output.shape[:2],
                injected_sst_value[0],
                dtype=jnp.float32,
            ),
            sst_sharding,
        )
        return Experiment5CanonicalTargetScores(target, sst)

    monkeypatch.setattr(common, "score", injected_scores)
    evaluator = Experiment5DistributedEvaluator(
        mesh,
        layout,
        distributed,
        # The transcoder reconstruction is independent of the full target,
        # so changing only the injected target isolates its use in full SSE.
        encoder_kind="transcoder",
        bank_capacity=1 if layout.is_hybrid else None,
        canonical_target_evaluator=common,
    )
    physical = shard_experiment5_train_state(
        state,
        mesh,
        layout,
        _optimizer(),
        distributed,
    )

    first = process_local_experiment5_device_scores(
        evaluator.score(
            physical,
            [batch],
            zero_centering,
            _recomposition(),
        )
    )
    injected_target_value[0] = 3.0
    injected_sst_value[0] = 11.0
    second = process_local_experiment5_device_scores(
        evaluator.score(
            physical,
            [batch],
            zero_centering,
            _recomposition(),
        )
    )

    # A zero decoder and zero decoder means make every reconstruction exactly
    # zero. The only possible full-SSE values are therefore ||target||².
    np.testing.assert_array_equal(
        first.full_sse,
        np.full(
            (batch.valid_mask.shape[0],),
            layout.hidden_dim * 4.0,
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        second.full_sse,
        np.full(
            (batch.valid_mask.shape[0],),
            layout.hidden_dim * 9.0,
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        first.full_sst,
        np.full((batch.valid_mask.shape[0],), 7.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        second.full_sst,
        np.full((batch.valid_mask.shape[0],), 11.0, dtype=np.float32),
    )


def test_padding_rows_are_zeroed_and_ids_are_negative_one() -> None:
    layout = _layout("H")
    distributed = DistributedConfig(
        data_axis_size=1,
        feature_axis_size=1,
        microbatches_per_update=1,
    )
    mesh = create_mesh(distributed, devices=np.asarray(jax.devices()[:1]))
    batch = _batch(invalid_tail=True)
    state = shard_experiment5_train_state(
        _state(layout),
        mesh,
        layout,
        _optimizer(),
        distributed,
    )
    evaluator = Experiment5DistributedEvaluator(
        mesh,
        layout,
        distributed,
        encoder_kind="sae",
        bank_capacity=1,
    )
    rows = process_local_experiment5_device_scores(
        evaluator.score(
            state,
            [batch],
            _centering(),
            _recomposition(),
        )
    )

    np.testing.assert_array_equal(rows.selected_latent_ids[1], -1)
    assert rows.bank_ids is not None
    np.testing.assert_array_equal(rows.bank_ids[1], -1)
    for value in (
        rows.full_sse,
        rows.full_sst,
        rows.routed_sse,
        rows.shared_sse,
        rows.oracle_shared_full_sse,
        rows.oracle_routed_full_sse,
    ):
        assert value is not None
        assert float(value[1]) == 0.0


def test_two_feature_shards_emit_canonical_logical_rows() -> None:
    """Exercise physical packing without ever gathering the parameter tree."""

    script = r"""
import jax
import jax.numpy as jnp
import numpy as np

from moe_sae.contribution_training import ContributionOptimizerConfig
from moe_sae.distributed import DistributedConfig, create_mesh
from moe_sae.experiment5_device_evaluation import (
    Experiment5DistributedEvaluator,
    process_local_experiment5_device_scores,
)
from moe_sae.experiment5_distributed import shard_experiment5_train_state
from moe_sae.experiment5_hybrid import (
    Experiment5Batch,
    Experiment5Centering,
    Experiment5Layout,
    Experiment5Recomposition,
    forward_experiment5,
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
batch = Experiment5Batch(
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
layout = Experiment5Layout(
    condition="H",
    hidden_dim=2,
    num_experts=4,
    top_routed=2,
    routed_bank_width=2,
    routed_k=1,
    shared_width=4,
    shared_k=2,
)
state = init_experiment5_train_state(jax.random.key(9), layout, optimizer)
state = state._replace(
    params=state.params._replace(
        bias=3.0 * jnp.ones_like(state.params.bias),
    )
)
reference = forward_experiment5(
    state.params,
    batch,
    centering,
    recomposition,
    layout,
    encoder_kind="sae",
)
physical = shard_experiment5_train_state(
    state,
    mesh,
    layout,
    optimizer,
    distributed,
)
evaluator = Experiment5DistributedEvaluator(
    mesh,
    layout,
    distributed,
    encoder_kind="sae",
    bank_capacity=1,
)
rows = process_local_experiment5_device_scores(
    evaluator.score(
        physical,
        [batch],
        centering,
        recomposition,
    )
)
expected = np.concatenate(
    (
        np.asarray(reference.routed_selection.latent_ids).reshape(2, 2),
        np.asarray(reference.shared_selection.latent_ids),
    ),
    axis=1,
)
np.testing.assert_array_equal(rows.selected_latent_ids, expected)
assert rows.selected_latent_ids.dtype == np.int32
assert rows.full_sse.dtype == np.float32
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


def test_canonical_target_preserves_exact_two_axis_sharding_and_dtype() -> None:
    script = r"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from moe_sae.contribution_training import ContributionOptimizerConfig
from moe_sae.distributed import DistributedConfig, create_mesh
from moe_sae.experiment5_device_evaluation import (
    Experiment5CanonicalTargetEvaluator,
    Experiment5DistributedEvaluator,
    process_local_experiment5_device_scores,
)
from moe_sae.experiment5_distributed import shard_experiment5_train_state
from moe_sae.experiment5_hybrid import (
    Experiment5Batch,
    Experiment5Centering,
    Experiment5Layout,
    Experiment5Recomposition,
    init_experiment5_train_state,
)

assert jax.device_count() == 4
distributed = DistributedConfig(
    data_axis_size=2,
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
batch = Experiment5Batch(
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
centering = Experiment5Centering(
    source_mean=np.asarray([0.1, -0.2, 0.05], np.float32),
    routed_target_mean=np.asarray([0.2, 0.1, -0.1], np.float32),
    shared_target_mean=np.asarray([-0.1, 0.2, 0.05], np.float32),
    full_target_mean=np.asarray([0.0, 0.1, -0.2], np.float32),
)
recomposition = Experiment5Recomposition(
    routed_scale=np.asarray([0.75, 1.0, 1.25], np.float32),
    outer_scale=np.asarray([1.125, 0.875, 1.0], np.float32),
)
common = Experiment5CanonicalTargetEvaluator(
    mesh,
    distributed,
    hidden_dim=3,
    top_routed=2,
)
targets = []
canonical_ssts = []
condition_ssts = []

for condition in ("H", "R", "B", "F"):
    layout = Experiment5Layout(
        condition=condition,
        hidden_dim=3,
        num_experts=4,
        top_routed=2,
        routed_bank_width=2,
        routed_k=1,
        shared_width=4,
        shared_k=2,
        randomization_key=17,
    )
    evaluator = Experiment5DistributedEvaluator(
        mesh,
        layout,
        distributed,
        encoder_kind="sae",
        bank_capacity=1 if layout.is_hybrid else None,
        canonical_target_evaluator=common,
    )
    canonical = evaluator.canonical_target_scores(
        [batch],
        centering,
        recomposition,
    )
    assert canonical.full_target.dtype == jnp.bfloat16
    assert canonical.full_sst.dtype == jnp.float32
    assert canonical.full_target.sharding.spec == P(None, "data", None)
    assert canonical.full_sst.sharding.spec == P(None, "data")
    targets.append(np.asarray(canonical.full_target))
    canonical_ssts.append(np.asarray(canonical.full_sst))

    state = init_experiment5_train_state(
        jax.random.key(31),
        layout,
        optimizer,
    )
    state = state._replace(
        params=state.params._replace(
            bias=3.0 * jnp.ones_like(state.params.bias),
        )
    )
    physical = shard_experiment5_train_state(
        state,
        mesh,
        layout,
        optimizer,
        distributed,
    )
    scores = process_local_experiment5_device_scores(
        evaluator.score(
            physical,
            [batch],
            centering,
            recomposition,
        )
    )
    condition_ssts.append(scores.full_sst)

assert common.compile_count == 1
for target in targets[1:]:
    np.testing.assert_array_equal(target, targets[0])
for sst in (*canonical_ssts[1:], *condition_ssts):
    np.testing.assert_array_equal(
        np.asarray(sst).reshape(-1),
        canonical_ssts[0].reshape(-1),
    )
"""
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
