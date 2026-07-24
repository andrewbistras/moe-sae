"""Fail-closed evaluation tests for Experiment-2 contribution SAEs."""

from __future__ import annotations

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    SCHEMA_VERSION,
    TOP_K_EXPERTS,
    ContributionBatch,
    bf16_bits_to_float32,
    build_recomposition_sidecar,
    compute_contribution_statistics,
    float32_to_bf16_bits,
)
from moe_sae.contribution_distributed import shard_contribution_train_state
from moe_sae.contribution_evaluation import (
    ContributionDistributedEvaluator,
    ContributionEvaluationChunk,
    ContributionEvaluationEvidenceChunk,
    ContributionEvaluationError,
    evaluate_contribution_input,
    oracle_evaluate_microbatch,
    routed_recomposition_token_scores,
)
from moe_sae.contribution_input import ContributionProcessInput
from moe_sae.contribution_recomposition import (
    SUPPORTED_ACCUMULATION_ORDER,
    maxtext_rmsnorm_bf16,
    recompose_gemma4,
)
from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_training import (
    ContributionOptimizerConfig,
    init_contribution_train_state,
)
from moe_sae.distributed import DistributedConfig, create_mesh


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
        warmup_steps=0,
        total_steps=2,
    )


def _microbatches(condition: str):
    target = np.arange(36, dtype=np.float32).reshape(6, 2, 3) / 20.0 + 0.2
    expert_banks = np.asarray(
        [[0, 1], [2, 3], [0, 2], [1, 3], [3, 2], [0, 1]],
        np.int32,
    )
    rank_banks = np.broadcast_to(
        np.arange(2, dtype=np.int32)[None, :],
        (6, 2),
    ).copy()
    banks = expert_banks if condition in {"A", "R"} else rank_banks
    weights = np.broadcast_to(
        np.asarray([[0.625, 0.375]], np.float32),
        (6, 2),
    ).copy()
    return [
        {
            "expert_output": target[:3],
            "bank_ids": banks[:3],
            "router_weights": weights[:3],
            "valid_mask": np.asarray([True, True, True]),
        },
        {
            "expert_output": target[3:],
            "bank_ids": banks[3:],
            "router_weights": weights[3:],
            "valid_mask": np.asarray([True, False, False]),
        },
    ]


@pytest.mark.parametrize("condition", ["A", "R", "B", "F"])
def test_distributed_evaluator_matches_dense_oracle(condition: str) -> None:
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
    state = shard_contribution_train_state(
        state,
        mesh,
        layout,
        optimizer,
        distributed,
    )
    microbatches = _microbatches(condition)
    mean = np.asarray([0.1, -0.2, 0.05], np.float32)

    evaluator = ContributionDistributedEvaluator(
        mesh,
        layout,
        distributed,
        bank_capacity=3 if condition in {"A", "R"} else None,
    )
    aux, actual = evaluator.step(state.params, microbatches, mean)
    expected = [
        oracle_evaluate_microbatch(state.params, item, mean, layout)
        for item in microbatches
    ]

    for name in (
        "component_token_sse",
        "component_token_sst",
        "component_route_sse",
        "component_route_sst",
        "raw_routed_target",
        "raw_routed_reconstruction",
        "router_weights",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(actual, name)),
            np.concatenate(
                [np.asarray(getattr(item, name)) for item in expected],
                axis=0,
            ),
            rtol=3e-5,
            atol=3e-5,
        )
    for name in ("per_contribution_l0", "bank_ids", "valid_mask"):
        np.testing.assert_array_equal(
            np.asarray(getattr(actual, name)),
            np.concatenate(
                [np.asarray(getattr(item, name)) for item in expected],
                axis=0,
            ),
        )

    assert int(aux.valid_source_tokens) == 4
    assert int(aux.valid_components) == 8
    assert int(aux.l0_sum) == 4 * layout.total_l0
    assert bool(aux.topk_valid)
    assert not bool(aux.capacity_overflow)
    assert bool(aux.params_finite)
    assert float(aux.component_sse) == pytest.approx(
        actual.component_sse,
        rel=3e-5,
        abs=3e-5,
    )
    assert float(aux.component_sst) == pytest.approx(
        actual.component_sst,
        rel=3e-5,
        abs=3e-5,
    )


def test_evaluator_rejects_capacity_overflow_and_dormant_nan() -> None:
    layout = _layout("A", experts=2, routes=2, width=2, k=1)
    optimizer = _optimizer()
    distributed = DistributedConfig(microbatches_per_update=1)
    mesh = create_mesh(distributed)
    state = init_contribution_train_state(jax.random.key(4), layout, optimizer)
    state = state._replace(
        params=state.params._replace(
            bias=10.0 * jnp.ones_like(state.params.bias),
        )
    )
    state = shard_contribution_train_state(
        state,
        mesh,
        layout,
        optimizer,
        distributed,
    )
    batch = {
        "expert_output": np.ones((2, 2, 3), np.float32),
        "bank_ids": np.asarray([[0, 1], [0, 1]], np.int32),
        "router_weights": np.full((2, 2), 0.5, np.float32),
        "valid_mask": np.ones((2,), np.bool_),
    }
    evaluator = ContributionDistributedEvaluator(
        mesh,
        layout,
        distributed,
        bank_capacity=1,
    )
    with pytest.raises(ContributionEvaluationError, match="capacity"):
        evaluator.step(state.params, [batch], np.zeros((3,), np.float32))

    safe = ContributionDistributedEvaluator(
        mesh,
        layout,
        distributed,
        bank_capacity=2,
    )
    poisoned = state.params._replace(
        encoder=state.params.encoder.at[-1, -1].set(jnp.nan)
    )
    with pytest.raises(ContributionEvaluationError, match="non-finite"):
        safe.step(poisoned, [batch], np.zeros((3,), np.float32))


def _sidecar() -> dict:
    ones = float32_to_bf16_bits(np.ones((HIDDEN_SIZE,), np.float32))
    return build_recomposition_sidecar(
        checkpoint_sha256="1" * 64,
        maxtext_fork_commit="2" * 40,
        source_fingerprints={
            "checkpoint": "1" * 64,
            "fork_tree": "3" * 64,
        },
        routed_rmsnorm_scale=ones,
        shared_rmsnorm_scale=ones,
        outer_ffn_rmsnorm_scale=ones,
        layer_scalar=float32_to_bf16_bits(np.ones((1,), np.float32)),
        accumulation_order=SUPPORTED_ACCUMULATION_ORDER,
    )


def test_routed_sufficient_rows_match_exact_recomposition_oracle() -> None:
    target = np.linspace(
        0.25,
        1.25,
        HIDDEN_SIZE,
        dtype=np.float32,
    )[None, None, :]
    estimate = target * np.float32(0.7)
    valid = np.asarray([[True, False]])
    zeros_token = np.zeros((1, 2), np.float32)
    zeros_route = np.zeros((1, 2, TOP_K_EXPERTS), np.float32)
    chunk = ContributionEvaluationChunk(
        component_token_sse=zeros_token,
        component_token_sst=zeros_token,
        component_route_sse=zeros_route,
        component_route_sst=zeros_route,
        per_contribution_l0=np.where(
            valid[..., None],
            1,
            0,
        )
        * np.ones((1, 2, TOP_K_EXPERTS), np.int32),
        raw_routed_target=np.concatenate(
            (target, np.zeros_like(target)),
            axis=1,
        ),
        raw_routed_reconstruction=np.concatenate(
            (estimate, np.zeros_like(estimate)),
            axis=1,
        ),
        bank_ids=np.zeros((1, 2, TOP_K_EXPERTS), np.int32),
        router_weights=np.where(
            valid[..., None],
            1.0 / TOP_K_EXPERTS,
            0.0,
        ).astype(np.float32)
        * np.ones((1, 2, TOP_K_EXPERTS), np.float32),
        valid_mask=valid,
    )
    sidecar = _sidecar()
    routed_mean = np.zeros((HIDDEN_SIZE,), np.float32)
    scores = routed_recomposition_token_scores(chunk, sidecar, routed_mean)

    expert = np.broadcast_to(
        target.reshape(1, 1, HIDDEN_SIZE),
        (1, TOP_K_EXPERTS, HIDDEN_SIZE),
    )
    weights = np.full((1, TOP_K_EXPERTS), 1 / TOP_K_EXPERTS, np.float32)
    batch = ContributionBatch(
        expert_output=float32_to_bf16_bits(expert),
        expert_ids=np.arange(TOP_K_EXPERTS, dtype=np.uint8)[None, :],
        router_weights=float32_to_bf16_bits(weights),
        shared_output=float32_to_bf16_bits(
            np.zeros((1, HIDDEN_SIZE), np.float32)
        ),
        sequence_id=np.asarray([9], np.int64),
        position=np.asarray([0], np.uint16),
    )
    expected_target = bf16_bits_to_float32(
        recompose_gemma4(batch, sidecar).routed_post_norm_bf16_bits
    )
    _, expected_estimate_bits = maxtext_rmsnorm_bf16(
        float32_to_bf16_bits(estimate.reshape(1, HIDDEN_SIZE)),
        np.asarray(
            sidecar["tensors"]["routed_rmsnorm_scale"]["data"],
            dtype=np.uint16,
        ),
    )
    expected_estimate = bf16_bits_to_float32(expected_estimate_bits)
    expected_sse = np.sum(
        np.square(expected_target - expected_estimate, dtype=np.float32),
        dtype=np.float64,
    )
    expected_sst = np.sum(
        np.square(expected_target, dtype=np.float32),
        dtype=np.float64,
    )
    assert scores.tokens == 1
    assert scores.sse == pytest.approx(expected_sse)
    assert scores.sst == pytest.approx(expected_sst)
    assert scores.token_sse[0, 1] == 0
    assert scores.token_sst[0, 1] == 0


class _FakeDataset:
    def __init__(self, value: ContributionBatch):
        self.value = value
        self.total_tokens = value.tokens
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "manifest_sha256": "a" * 64,
            "statistics": compute_contribution_statistics(
                value,
                normalized_shared_output=np.zeros(
                    (value.tokens, HIDDEN_SIZE),
                    np.float32,
                ),
            ),
        }

    def iter_token_range(self, start: int, end: int):
        # Deliberately cross an underlying shard boundary inside the first
        # static microbatch.
        boundary = min(start + 2, end)
        for left, right in ((start, boundary), (boundary, end)):
            if left == right:
                continue
            yield ContributionBatch(
                **{
                    name: np.ascontiguousarray(
                        np.asarray(getattr(self.value, name))[left:right]
                    )
                    for name in ContributionBatch.__dataclass_fields__
                }
            )


def test_sealed_input_runner_consumes_exact_tail_and_emits_rows() -> None:
    tokens = 5
    expert = np.ones(
        (tokens, TOP_K_EXPERTS, HIDDEN_SIZE),
        dtype=np.float32,
    )
    weights = np.full(
        (tokens, TOP_K_EXPERTS),
        1.0 / TOP_K_EXPERTS,
        dtype=np.float32,
    )
    wire = ContributionBatch(
        expert_output=float32_to_bf16_bits(expert),
        expert_ids=np.broadcast_to(
            np.arange(TOP_K_EXPERTS, dtype=np.uint8),
            (tokens, TOP_K_EXPERTS),
        ).copy(),
        router_weights=float32_to_bf16_bits(weights),
        shared_output=float32_to_bf16_bits(
            np.zeros((tokens, HIDDEN_SIZE), np.float32)
        ),
        sequence_id=np.arange(tokens, dtype=np.int64),
        position=np.arange(tokens, dtype=np.uint16),
    )
    layout = ContributionSAEConfig(
        condition="B",
        hidden_dim=HIDDEN_SIZE,
        bank_width=2,
        k_per_contribution=1,
        randomization_key=17,
    )
    source = ContributionProcessInput(
        _FakeDataset(wire),
        layout,
        total_tokens=tokens,
        global_microbatch_tokens=4,
        microbatches_per_update=1,
    )
    optimizer = ContributionOptimizerConfig(
        parameter_dtype="float32",
        warmup_steps=0,
        total_steps=2,
    )
    distributed = DistributedConfig(microbatches_per_update=1)
    mesh = create_mesh(distributed)
    state = init_contribution_train_state(jax.random.key(5), layout, optimizer)
    state = state._replace(
        params=state.params._replace(
            bias=10.0 * jnp.ones_like(state.params.bias),
        )
    )
    state = shard_contribution_train_state(
        state,
        mesh,
        layout,
        optimizer,
        distributed,
    )
    chunks: list[ContributionEvaluationEvidenceChunk] = []
    result = evaluate_contribution_input(
        state.params,
        source,
        np.zeros((HIDDEN_SIZE,), np.float32),
        mesh,
        checkpoint_sha256="4" * 64,
        distributed=distributed,
        row_sink=chunks.append,
    )

    assert result.tokens == tokens
    assert result.valid_components == tokens * TOP_K_EXPERTS
    assert result.l0_sum == tokens * layout.total_l0
    assert result.l0_per_contribution == 1
    assert result.total_l0 == TOP_K_EXPERTS
    assert result.updates == 2
    assert result.local_sufficient_rows == tokens
    assert result.checkpoint_sha256 == "4" * 64
    assert len(result.centering_mean_sha256) == 64
    assert len(result.receipt_sha256) == 64
    assert sum(chunk.tokens for chunk in chunks) == tokens
    assert chunks[-1].scores.valid_mask.tolist() == [
        [True, False, False, False]
    ]
    observed_keys = [
        (int(sequence), int(position))
        for chunk in chunks
        for sequence, position in zip(
            chunk.sequence_id[chunk.scores.valid_mask],
            chunk.position[chunk.scores.valid_mask],
            strict=True,
        )
    ]
    assert observed_keys == list(
        zip(
            wire.sequence_id.tolist(),
            wire.position.tolist(),
            strict=True,
        )
    )
    assert chunks[-1].sequence_id.tolist() == [[4, -1, -1, -1]]
    assert chunks[-1].position.tolist() == [[4, 65_535, 65_535, 65_535]]
    bad_sequence = chunks[-1].sequence_id.copy()
    bad_sequence[0, 1] = 0
    with pytest.raises(ContributionEvaluationError, match="sentinels"):
        ContributionEvaluationEvidenceChunk(
            update_index=chunks[-1].update_index,
            sequence_id=bad_sequence,
            position=chunks[-1].position,
            scores=chunks[-1].scores,
        )


def test_two_axis_collectives_match_dense_oracle() -> None:
    script = r"""
import jax
import jax.numpy as jnp
import numpy as np

from moe_sae.contribution_distributed import shard_contribution_train_state
from moe_sae.contribution_evaluation import (
    ContributionDistributedEvaluator,
    oracle_evaluate_microbatch,
)
from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_training import (
    ContributionOptimizerConfig,
    init_contribution_train_state,
)
from moe_sae.distributed import DistributedConfig, create_mesh

layout = ContributionSAEConfig(
    condition="A",
    hidden_dim=3,
    num_experts=4,
    top_routed=2,
    bank_width=4,
    k_per_contribution=2,
    randomization_key=17,
)
optimizer = ContributionOptimizerConfig(
    parameter_dtype="float32",
    warmup_steps=0,
    total_steps=2,
)
distributed = DistributedConfig(
    data_axis_size=2,
    feature_axis_size=2,
    microbatches_per_update=1,
)
mesh = create_mesh(distributed)
state = init_contribution_train_state(jax.random.key(9), layout, optimizer)
state = state._replace(
    params=state.params._replace(
        bias=10.0 * jnp.ones_like(state.params.bias),
    )
)
target = np.arange(24, dtype=np.float32).reshape(4, 2, 3) / 20 + 0.2
banks = np.asarray([[0, 1], [2, 3], [0, 2], [1, 3]], np.int32)
weights = np.broadcast_to(
    np.asarray([[0.625, 0.375]], np.float32),
    (4, 2),
).copy()
batch = {
    "expert_output": target,
    "bank_ids": banks,
    "router_weights": weights,
    "valid_mask": np.asarray([True, True, True, False]),
}
mean = np.asarray([0.1, -0.2, 0.05], np.float32)
evaluator = ContributionDistributedEvaluator(
    mesh,
    layout,
    distributed,
    bank_capacity=2,
)
params = shard_contribution_train_state(
    state,
    mesh,
    layout,
    optimizer,
    distributed,
).params
aux, actual = evaluator.step(params, [batch], mean)
expected = oracle_evaluate_microbatch(state.params, batch, mean, layout)
for name in (
    "component_token_sse",
    "component_token_sst",
    "component_route_sse",
    "component_route_sst",
    "raw_routed_target",
    "raw_routed_reconstruction",
):
    np.testing.assert_allclose(
        np.asarray(getattr(actual, name)),
        np.asarray(getattr(expected, name)),
        rtol=3e-5,
        atol=3e-5,
    )
np.testing.assert_array_equal(
    actual.per_contribution_l0,
    expected.per_contribution_l0,
)
assert int(aux.valid_source_tokens) == 3
assert int(aux.l0_sum) == 3 * layout.total_l0
assert bool(aux.topk_valid)
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
