"""Production-shaped distributed tests for Experiment-2 contribution SAEs."""

from __future__ import annotations

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moe_sae.contribution_distributed import (
    ContributionDistributedTrainer,
    shard_contribution_train_state,
    stack_contribution_microbatches,
)
from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_training import (
    ContributionOptimizerConfig,
    contribution_train_step,
    init_contribution_train_state,
)
from moe_sae.distributed import DistributedConfig, create_mesh
from moe_sae.sae import SAEParams


def _layout(
    condition: str,
    *,
    hidden: int = 3,
    experts: int = 4,
    routes: int = 2,
    width: int = 4,
    k: int = 2,
) -> ContributionSAEConfig:
    return ContributionSAEConfig(
        condition=condition,  # type: ignore[arg-type]
        hidden_dim=hidden,
        num_experts=experts,
        top_routed=routes,
        bank_width=width,
        k_per_contribution=k,
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


def _assert_state_close(actual, expected, *, atol: float = 5e-6) -> None:
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


@pytest.mark.parametrize("condition", ["A", "R", "B", "F"])
def test_single_device_update_matches_dense_oracle(condition: str) -> None:
    layout = _layout(condition, width=8 if condition == "F" else 4)
    optimizer = _optimizer()
    distributed = DistributedConfig(microbatches_per_update=2)
    mesh = create_mesh(distributed)
    state = init_contribution_train_state(
        jax.random.key(3),
        layout,
        optimizer,
    )
    state = state._replace(
        params=state.params._replace(
            bias=10.0 * jnp.ones_like(state.params.bias),
        )
    )
    target = np.arange(30, dtype=np.float32).reshape(5, 2, 3) / 20.0 + 0.2
    expert_banks = np.asarray(
        [[0, 1], [2, 3], [0, 2], [1, 3], [3, 2]],
        np.int32,
    )
    rank_banks = np.broadcast_to(
        np.arange(2, dtype=np.int32)[None, :],
        (5, 2),
    ).copy()
    banks = expert_banks if condition in {"A", "R"} else rank_banks
    weights = np.broadcast_to(
        np.asarray([[0.625, 0.375]], np.float32),
        (5, 2),
    ).copy()
    valid = np.asarray([True, True, True, False, False])
    mean = np.asarray([0.1, -0.2, 0.05], np.float32)

    # Static microbatches contain explicit padding in the second group.
    microbatches = [
        {
            "expert_output": target[:3],
            "bank_ids": banks[:3],
            "router_weights": weights[:3],
            "valid_mask": valid[:3],
        },
        {
            "expert_output": target[2:],
            "bank_ids": banks[2:],
            "router_weights": weights[2:],
            "valid_mask": np.asarray([False, True, False]),
        },
    ]
    # The second group intentionally reuses source row 2 as padding and row 3
    # as its sole scientific row, so compare with that exact stacked order.
    stacked_target = np.concatenate((target[:3], target[2:]))
    stacked_banks = np.concatenate((banks[:3], banks[2:]))
    stacked_weights = np.concatenate((weights[:3], weights[2:]))
    stacked_valid = np.asarray([True, True, True, False, True, False])
    reference, reference_metrics = contribution_train_step(
        state,
        stacked_target,
        stacked_banks,
        stacked_weights,
        mean,
        layout,
        optimizer,
        valid_mask=stacked_valid,
    )

    trainer = ContributionDistributedTrainer(
        mesh,
        layout,
        optimizer,
        distributed,
        bank_capacity=3 if condition in {"A", "R"} else None,
    )
    sharded = shard_contribution_train_state(
        state,
        mesh,
        layout,
        optimizer,
        distributed,
    )
    actual, metrics = trainer.step(sharded, microbatches, mean)

    _assert_state_close(actual, reference)
    assert float(metrics.loss) == pytest.approx(
        float(reference_metrics.loss),
        rel=2e-5,
        abs=5e-6,
    )
    assert int(metrics.valid_tokens) == 4
    assert float(metrics.l0) == layout.total_l0
    assert bool(metrics.topk_valid)


def test_tied_scores_use_lowest_global_ids_and_padding_never_fires() -> None:
    layout = _layout("A", hidden=2, width=3, k=2)
    optimizer = _optimizer()
    distributed = DistributedConfig(microbatches_per_update=1)
    mesh = create_mesh(distributed)
    state = init_contribution_train_state(
        jax.random.key(1),
        layout,
        optimizer,
    )
    state = state._replace(
        params=SAEParams(
            encoder=jnp.zeros_like(state.params.encoder),
            decoder=state.params.decoder,
            bias=jnp.ones_like(state.params.bias),
        )
    )
    target = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[3.0, 4.0], [5.0, 6.0]],
        ],
        np.float32,
    )
    banks = np.asarray([[2, 0], [3, 1]], np.int32)
    weights = np.asarray([[0.6, 0.4], [0.0, 0.0]], np.float32)
    trainer = ContributionDistributedTrainer(
        mesh,
        layout,
        optimizer,
        distributed,
        bank_capacity=1,
    )
    result, metrics = trainer.step(
        shard_contribution_train_state(
            state,
            mesh,
            layout,
            optimizer,
            distributed,
        ),
        [
            {
                "expert_output": target,
                "bank_ids": banks,
                "router_weights": weights,
                "valid_mask": np.asarray([True, False]),
            }
        ],
        np.zeros((2,), np.float32),
    )

    expected = np.zeros((layout.total_features,), dtype=np.bool_)
    expected[[0, 1, 6, 7]] = True
    np.testing.assert_array_equal(np.asarray(result.fired_mask), expected)
    assert int(metrics.valid_tokens) == 1
    assert float(metrics.l0) == 4.0
    assert bool(metrics.topk_valid)


def test_sae_distributed_trainer_can_run_the_softplus_activation_graph() -> None:
    layout = _layout("A", hidden=2, experts=2, routes=1, width=2, k=1)
    optimizer = _optimizer()
    distributed = DistributedConfig(microbatches_per_update=1)
    mesh = create_mesh(distributed)
    state = init_contribution_train_state(
        jax.random.key(19),
        layout,
        optimizer,
    )
    state = state._replace(
        params=SAEParams(
            encoder=jnp.zeros_like(state.params.encoder),
            decoder=state.params.decoder,
            bias=jnp.full_like(state.params.bias, -1.0e6),
        )
    )
    trainer = ContributionDistributedTrainer(
        mesh,
        layout,
        optimizer,
        distributed,
        bank_capacity=1,
        rematerialize_microbatch=False,
        encoder_activation="softplus",
    )
    _updated, metrics = trainer.step(
        shard_contribution_train_state(
            state,
            mesh,
            layout,
            optimizer,
            distributed,
        ),
        [
            {
                "expert_output": np.asarray([[[1.0, -0.5]]], np.float32),
                "bank_ids": np.asarray([[0]], np.int32),
                "router_weights": np.asarray([[1.0]], np.float32),
                "valid_mask": np.asarray([True]),
            }
        ],
        np.zeros((2,), np.float32),
    )

    assert trainer.encoder_activation == "softplus"
    assert float(metrics.l0) == 1.0
    assert np.isfinite(float(metrics.loss))
    assert bool(metrics.topk_valid)


def test_no_rematerialization_preserves_the_exact_update() -> None:
    layout = _layout("B", hidden=2, routes=2, width=4, k=2)
    optimizer = _optimizer()
    distributed = DistributedConfig(microbatches_per_update=1)
    mesh = create_mesh(distributed)
    state = init_contribution_train_state(jax.random.key(13), layout, optimizer)
    state = state._replace(
        params=state.params._replace(
            bias=2.0 * jnp.ones_like(state.params.bias),
        )
    )
    state = shard_contribution_train_state(
        state,
        mesh,
        layout,
        optimizer,
        distributed,
    )
    batch = [
        {
            "expert_output": np.asarray(
                [[[0.5, 1.0], [1.5, -0.25]], [[0.25, 0.75], [1.0, 0.5]]],
                np.float32,
            ),
            "bank_ids": np.asarray([[0, 1], [0, 1]], np.int32),
            "router_weights": np.asarray([[0.6, 0.4], [0.55, 0.45]], np.float32),
            "valid_mask": np.asarray([True, True]),
        }
    ]
    mean = np.asarray([0.1, -0.2], np.float32)
    rematerialized = ContributionDistributedTrainer(
        mesh,
        layout,
        optimizer,
        distributed,
        rematerialize_microbatch=True,
    )
    resident = ContributionDistributedTrainer(
        mesh,
        layout,
        optimizer,
        distributed,
        rematerialize_microbatch=False,
    )

    remat_state, remat_metrics = rematerialized.step(state, batch, mean)
    resident_state, resident_metrics = resident.step(state, batch, mean)

    _assert_state_close(resident_state, remat_state)
    np.testing.assert_allclose(
        np.asarray(resident_metrics.loss),
        np.asarray(remat_metrics.loss),
        rtol=1e-6,
    )


def test_grouped_capacity_overflow_is_a_hard_error() -> None:
    layout = _layout(
        "A",
        hidden=2,
        experts=2,
        routes=2,
        width=2,
        k=1,
    )
    optimizer = _optimizer()
    distributed = DistributedConfig(microbatches_per_update=1)
    mesh = create_mesh(distributed)
    state = init_contribution_train_state(
        jax.random.key(2),
        layout,
        optimizer,
    )
    state = state._replace(
        params=state.params._replace(
            bias=10.0 * jnp.ones_like(state.params.bias),
        )
    )
    batch = {
        "expert_output": np.ones((2, 2, 2), np.float32),
        "bank_ids": np.asarray([[0, 1], [0, 1]], np.int32),
        "router_weights": np.full((2, 2), 0.5, np.float32),
        "valid_mask": np.ones((2,), np.bool_),
    }
    trainer = ContributionDistributedTrainer(
        mesh,
        layout,
        optimizer,
        distributed,
        bank_capacity=1,
    )

    with pytest.raises(RuntimeError, match="exceeded.*capacity"):
        trainer.step(
            shard_contribution_train_state(
                state,
                mesh,
                layout,
                optimizer,
                distributed,
            ),
            [batch],
            np.zeros((2,), np.float32),
        )


@pytest.mark.parametrize(
    "corruption",
    ["parameter", "first_moment", "second_moment"],
)
def test_dormant_nonfinite_state_is_never_accepted(corruption: str) -> None:
    layout = _layout(
        "B",
        hidden=2,
        experts=2,
        routes=2,
        width=4,
        k=1,
    )
    optimizer = _optimizer()
    distributed = DistributedConfig(microbatches_per_update=1)
    mesh = create_mesh(distributed)
    state = init_contribution_train_state(
        jax.random.key(8),
        layout,
        optimizer,
    )
    state = state._replace(
        params=SAEParams(
            encoder=jnp.zeros_like(state.params.encoder),
            decoder=state.params.decoder,
            bias=jnp.asarray([10, 0, 0, 0, 10, 0, 0, 0], jnp.float32),
        )
    )
    if corruption == "parameter":
        state = state._replace(
            params=state.params._replace(
                # Rank one always selects latent four, so the final decoder
                # column is dormant for this update.
                decoder=state.params.decoder.at[:, -1].set(jnp.nan),
            )
        )
    elif corruption == "first_moment":
        state = state._replace(
            optimizer=state.optimizer._replace(
                m=state.optimizer.m._replace(
                    encoder=state.optimizer.m.encoder.at[-1, 0].set(jnp.nan),
                )
            )
        )
    else:
        state = state._replace(
            optimizer=state.optimizer._replace(
                v=state.optimizer.v._replace(
                    encoder=state.optimizer.v.encoder.at[-1, 0].set(jnp.nan),
                )
            )
        )

    trainer = ContributionDistributedTrainer(
        mesh,
        layout,
        optimizer,
        distributed,
    )
    batch = {
        "expert_output": np.ones((1, 2, 2), np.float32),
        "bank_ids": np.asarray([[0, 1]], np.int32),
        "router_weights": np.asarray([[0.5, 0.5]], np.float32),
        "valid_mask": np.ones((1,), np.bool_),
    }

    with pytest.raises(RuntimeError, match="non-finite parameters or Adam moments"):
        trainer.step(
            shard_contribution_train_state(
                state,
                mesh,
                layout,
                optimizer,
                distributed,
            ),
            [batch],
            np.zeros((2,), np.float32),
        )
    assert int(state.step) == 0


def test_capacity_contract_is_condition_specific() -> None:
    optimizer = _optimizer()
    distributed = DistributedConfig(microbatches_per_update=1)
    mesh = create_mesh(distributed)
    with pytest.raises(ValueError, match="A/R.*requires"):
        ContributionDistributedTrainer(
            mesh,
            _layout("A"),
            optimizer,
            distributed,
        )
    with pytest.raises(ValueError, match="B/F.*does not accept"):
        ContributionDistributedTrainer(
            mesh,
            _layout("B"),
            optimizer,
            distributed,
            bank_capacity=1,
        )


def test_stack_fails_closed_on_condition_bank_contracts() -> None:
    distributed = DistributedConfig(microbatches_per_update=1)
    common = {
        "expert_output": np.ones((2, 2, 2), np.float32),
        "router_weights": np.full((2, 2), 0.5, np.float32),
        "valid_mask": np.ones((2,), np.bool_),
    }
    with pytest.raises(ValueError, match="unique"):
        stack_contribution_microbatches(
            [{**common, "bank_ids": np.asarray([[0, 0], [1, 2]], np.int32)}],
            _layout("A", hidden=2),
            distributed,
        )
    with pytest.raises(ValueError, match="canonical router ranks"):
        stack_contribution_microbatches(
            [{**common, "bank_ids": np.asarray([[1, 0], [0, 1]], np.int32)}],
            _layout("B", hidden=2),
            distributed,
        )


def test_two_dimensional_cpu_mesh_matches_oracle_for_all_conditions() -> None:
    """Exercise real data/feature collectives in a fresh four-device process."""

    script = r"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from moe_sae.contribution_distributed import (
    ContributionDistributedTrainer,
    shard_contribution_train_state,
)
from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_training import (
    ContributionOptimizerConfig,
    contribution_train_step,
    init_contribution_train_state,
)
from moe_sae.distributed import DistributedConfig, create_mesh

assert jax.device_count() == 4
distributed = DistributedConfig(
    data_axis_size=2,
    feature_axis_size=2,
    microbatches_per_update=2,
)
mesh = create_mesh(distributed)
target = np.arange(48, dtype=np.float32).reshape(8, 2, 3) / 20.0 + 0.2
weights = np.broadcast_to(
    np.asarray([[0.6, 0.4]], np.float32),
    (8, 2),
).copy()
valid = np.asarray([True, True, True, True, True, True, True, False])
mean = np.asarray([0.1, -0.2, 0.05], np.float32)
expert_banks = np.asarray(
    [[0, 1], [2, 3], [0, 2], [1, 3],
     [0, 3], [1, 2], [0, 1], [2, 3]],
    np.int32,
)
rank_banks = np.broadcast_to(
    np.arange(2, dtype=np.int32)[None, :],
    (8, 2),
).copy()
optimizer = ContributionOptimizerConfig(
    parameter_dtype="float32",
    peak_lr=1e-3,
    min_lr=1e-3,
    warmup_steps=0,
    total_steps=2,
)

for condition in ("A", "R", "B", "F"):
    layout = ContributionSAEConfig(
        condition=condition,
        hidden_dim=3,
        num_experts=4,
        top_routed=2,
        bank_width=8 if condition == "F" else 4,
        k_per_contribution=2,
        randomization_key=17,
    )
    banks = expert_banks if condition in {"A", "R"} else rank_banks
    state = init_contribution_train_state(jax.random.key(3), layout, optimizer)
    state = state._replace(
        params=state.params._replace(
            bias=10.0 * jnp.ones_like(state.params.bias),
        )
    )
    reference, reference_metrics = contribution_train_step(
        state,
        target,
        banks,
        weights,
        mean,
        layout,
        optimizer,
        valid_mask=valid,
    )
    sharded = shard_contribution_train_state(
        state,
        mesh,
        layout,
        optimizer,
        distributed,
    )
    assert sharded.params.encoder.sharding.spec == P("feature", None)
    assert sharded.params.decoder.sharding.spec == P(None, "feature")
    assert sharded.optimizer.m.encoder.sharding.spec == P("feature", None)
    trainer = ContributionDistributedTrainer(
        mesh,
        layout,
        optimizer,
        distributed,
        bank_capacity=2 if condition in {"A", "R"} else None,
    )
    microbatches = []
    for selected in (slice(0, 4), slice(4, 8)):
        microbatches.append(
            {
                "expert_output": target[selected],
                "bank_ids": banks[selected],
                "router_weights": weights[selected],
                "valid_mask": valid[selected],
            }
        )
    actual, metrics = trainer.step(sharded, microbatches, mean)
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
                atol=5e-6,
            )
    np.testing.assert_allclose(
        float(metrics.loss),
        float(reference_metrics.loss),
        rtol=2e-5,
        atol=5e-6,
    )
    assert int(metrics.valid_tokens) == 7
    assert float(metrics.l0) == layout.total_l0
    assert bool(metrics.topk_valid)

bad_layout = ContributionSAEConfig(
    condition="A",
    hidden_dim=3,
    num_experts=3,
    top_routed=2,
    bank_width=4,
    k_per_contribution=2,
)
try:
    ContributionDistributedTrainer(
        mesh,
        bad_layout,
        optimizer,
        distributed,
        bank_capacity=2,
    )
except ValueError as error:
    assert "bank count must divide" in str(error)
else:
    raise AssertionError("split bank boundary was accepted")
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


def test_bf16_four_microbatch_two_axis_lowering_keeps_fp32_gradients() -> None:
    """Production BF16 forward must retain FP32 scan/reduction cotangents."""

    script = r"""
import re

import jax
import jax.numpy as jnp
import numpy as np

from moe_sae.contribution_distributed import (
    ContributionDistributedTrainer,
    shard_contribution_batch,
    shard_contribution_train_state,
    stack_contribution_microbatches,
)
from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_training import (
    ContributionOptimizerConfig,
    init_contribution_train_state,
)
from moe_sae.distributed import DistributedConfig, create_mesh

assert jax.device_count() == 4
distributed = DistributedConfig(
    data_axis_size=2,
    feature_axis_size=2,
    microbatches_per_update=4,
)
mesh = create_mesh(distributed)
layout = ContributionSAEConfig(
    condition="A",
    hidden_dim=4,
    num_experts=4,
    top_routed=2,
    bank_width=8,
    k_per_contribution=2,
    randomization_key=17,
)
optimizer = ContributionOptimizerConfig(
    parameter_dtype="bfloat16",
    peak_lr=1e-3,
    min_lr=1e-3,
    warmup_steps=0,
    total_steps=2,
)
state = init_contribution_train_state(jax.random.key(3), layout, optimizer)
state = state._replace(
    params=state.params._replace(
        bias=4.0 * jnp.ones_like(state.params.bias),
    )
)

base = np.asarray(
    [[1.25, -0.75, 0.5, 2.0], [-0.5, 1.5, 0.25, -1.0]],
    np.float32,
)
target = np.broadcast_to(base, (4, 4, 2, 4)).copy()
bank_pattern = np.asarray(
    [[0, 1], [2, 3], [0, 2], [1, 3]],
    np.int32,
)
banks = np.broadcast_to(bank_pattern, (4, 4, 2)).copy()
weights = np.broadcast_to(
    np.asarray([0.625, 0.375], np.float32),
    (4, 4, 2),
).copy()
valid = np.ones((4, 4), np.bool_)
mean = np.asarray([0.25, -0.125, 0.5, -0.25], np.float32)
microbatches = [
    {
        "expert_output": target[index],
        "bank_ids": banks[index],
        "router_weights": weights[index],
        "valid_mask": valid[index],
    }
    for index in range(4)
]

trainer = ContributionDistributedTrainer(
    mesh,
    layout,
    optimizer,
    distributed,
    bank_capacity=2,
)
sharded_state = shard_contribution_train_state(
    state,
    mesh,
    layout,
    optimizer,
    distributed,
)
placed, placed_mean = trainer._prepare(
    microbatches,
    mean,
    process_local=False,
)
lowered = trainer._compiled.lower(
    sharded_state,
    placed,
    placed_mean,
).as_text()

# Forward encoder and decoder GEMMs remain BF16, accumulate to FP32, and have
# an explicit DEFAULT precision policy.
bf16_dots = [
    line
    for line in lowered.splitlines()
    if "stablehlo.dot_general" in line and "xbf16>" in line
]
assert len(bf16_dots) >= 2
assert all("precision = [DEFAULT, DEFAULT]" in line for line in bf16_dots)

# The reverse scan's three parameter-gradient carries must be FP32.  This
# assertion detects moving the BF16 cast back outside the scanned body.
reverse_scans = [
    line
    for line in lowered.splitlines()
    if ":14 = stablehlo.while" in line
]
assert len(reverse_scans) == 1
assert "bf16" not in reverse_scans[0]
assert reverse_scans[0].count("xf32>") >= 7

# Data-axis gradient collectives must likewise never receive or return BF16.
all_reduce_signatures = re.findall(
    r'"stablehlo\.all_reduce".*?\}\) : \(([^\n]+)\) -> ([^\n]+)',
    lowered,
    flags=re.DOTALL,
)
assert all_reduce_signatures
assert not any(
    "bf16" in source or "bf16" in result
    for source, result in all_reduce_signatures
)

updated, metrics = trainer.step(sharded_state, microbatches, mean)
assert updated.params.encoder.dtype == jnp.bfloat16
assert updated.params.decoder.dtype == jnp.bfloat16
assert updated.optimizer.m.encoder.dtype == jnp.float32
assert updated.optimizer.v.encoder.dtype == jnp.float32
assert int(metrics.valid_tokens) == 16
assert float(metrics.l0) == layout.total_l0
assert np.isfinite(float(metrics.grad_norm))

# Compare the real two-axis result with the same BF16/four-microbatch kernel
# on one device.  Collective reduction order may move FP32 moments slightly,
# but the stored BF16 parameters and scientific metrics must agree.
single_distributed = DistributedConfig(
    data_axis_size=1,
    feature_axis_size=1,
    microbatches_per_update=4,
)
single_mesh = create_mesh(single_distributed, devices=[jax.devices()[0]])
single_trainer = ContributionDistributedTrainer(
    single_mesh,
    layout,
    optimizer,
    single_distributed,
    bank_capacity=2,
)
single_updated, single_metrics = single_trainer.step(
    shard_contribution_train_state(
        state,
        single_mesh,
        layout,
        optimizer,
        single_distributed,
    ),
    microbatches,
    mean,
)
for observed, reference in zip(
    jax.tree_util.tree_leaves(updated),
    jax.tree_util.tree_leaves(single_updated),
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
            rtol=5e-3,
            atol=2e-4,
        )
np.testing.assert_allclose(
    float(metrics.loss),
    float(single_metrics.loss),
    rtol=1e-6,
    atol=1e-6,
)

# Poison only a dormant feature on the second feature shard.  The scalar
# acceptance decision must still observe it globally.
poisoned = state._replace(
    optimizer=state.optimizer._replace(
        v=state.optimizer.v._replace(
            encoder=state.optimizer.v.encoder.at[-1, 0].set(jnp.nan),
        )
    )
)
try:
    trainer.step(
        shard_contribution_train_state(
            poisoned,
            mesh,
            layout,
            optimizer,
            distributed,
        ),
        microbatches,
        mean,
    )
except RuntimeError as error:
    assert "non-finite parameters or Adam moments" in str(error)
else:
    raise AssertionError("non-finite remote feature shard was accepted")
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
