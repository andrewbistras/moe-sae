"""Common shared-Q and exact full-FFN tests for Experiment 2."""

from __future__ import annotations

import copy
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    LEGACY_AGGREGATE_SCHEMA_VERSION,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    TOP_K_EXPERTS,
    ContributionBatch,
    ContributionMetadata,
    bf16_bits_to_float32,
    build_recomposition_sidecar,
    canonical_sha256,
    compute_contribution_statistics,
    contribution_batch_token_identity_sha256,
    deterministic_range_id,
    float32_to_bf16_bits,
    seal_contribution_manifest,
)
from moe_sae.contribution_evaluation import (
    ContributionEvaluationChunk,
    ContributionEvaluationEvidenceChunk,
)
from moe_sae.contribution_recomposition import (
    SUPPORTED_ACCUMULATION_ORDER,
    maxtext_rmsnorm_bf16,
    recompose_gemma4,
)
from moe_sae.contribution_shared import (
    SHARED_Q_INPUT_SCHEMA_VERSION,
    SharedQDistributedEvaluator,
    SharedQDistributedTrainer,
    SharedQError,
    SharedQEvaluationEvidenceChunk,
    SharedQEvaluationRows,
    SharedQLayout,
    SharedQProcessInput,
    build_shared_q_centering,
    full_ffn_token_scores,
    init_shared_q_state,
    recompose_full_ffn_rows,
    shared_q_sae_config,
)
from moe_sae.contribution_storage import ContributionSharedBatch
from moe_sae.distributed import DistributedConfig, create_mesh
from moe_sae.sae import forward, init_train_state, train_step

CHECKPOINT_SHA = "1" * 64
FORK_COMMIT = "2" * 40
SOURCE_SHA = "3" * 64


def _sidecar(
    *,
    shared_scale: np.ndarray | None = None,
    outer_scale: np.ndarray | None = None,
) -> dict:
    ones = float32_to_bf16_bits(np.ones((HIDDEN_SIZE,), np.float32))
    return build_recomposition_sidecar(
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_fork_commit=FORK_COMMIT,
        source_fingerprints={
            "checkpoint": CHECKPOINT_SHA,
            "fork_tree": "4" * 64,
        },
        routed_rmsnorm_scale=ones,
        shared_rmsnorm_scale=ones if shared_scale is None else shared_scale,
        outer_ffn_rmsnorm_scale=ones if outer_scale is None else outer_scale,
        layer_scalar=float32_to_bf16_bits(np.ones((1,), np.float32)),
        accumulation_order=SUPPORTED_ACCUMULATION_ORDER,
    )


def _wire(tokens: int, *, offset: float = 0.0) -> ContributionBatch:
    feature = np.linspace(
        0.25 + offset,
        1.25 + offset,
        HIDDEN_SIZE,
        dtype=np.float32,
    )
    expert = np.broadcast_to(
        feature[None, None, :],
        (tokens, TOP_K_EXPERTS, HIDDEN_SIZE),
    ).copy()
    shared = np.stack(
        [
            feature * np.float32(0.5 + 0.1 * row)
            + np.float32((row % 2) * 0.2)
            for row in range(tokens)
        ]
    )
    weights = np.asarray(
        [0.25, 0.1875, 0.15625, 0.125, 0.09375, 0.078125, 0.0625, 0.046875],
        np.float32,
    )
    return ContributionBatch(
        expert_output=float32_to_bf16_bits(expert),
        expert_ids=np.broadcast_to(
            np.arange(TOP_K_EXPERTS, dtype=np.uint8),
            (tokens, TOP_K_EXPERTS),
        ).copy(),
        router_weights=float32_to_bf16_bits(
            np.broadcast_to(weights, (tokens, TOP_K_EXPERTS))
        ),
        shared_output=float32_to_bf16_bits(shared),
        sequence_id=np.asarray(
            [100 + row // 2 for row in range(tokens)],
            np.int64,
        ),
        position=np.arange(tokens, dtype=np.uint16),
    )


def _manifest(value: ContributionBatch, sidecar: dict, *, split: str) -> dict:
    _, normalized = maxtext_rmsnorm_bf16(
        value.shared_output,
        np.asarray(
            sidecar["tensors"]["shared_rmsnorm_scale"]["data"],
            np.uint16,
        ),
    )
    metadata = ContributionMetadata(
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_fork_commit=FORK_COMMIT,
        dataset_fingerprint="5" * 64,
        split=split,
        config_sha256="6" * 64,
        environment_sha256="7" * 64,
        source_v1_manifest_sha256=SOURCE_SHA,
        recomposition_sidecar_sha256=sidecar["sidecar_sha256"],
    ).to_dict()
    metadata_sha = canonical_sha256(metadata)
    token_sha = contribution_batch_token_identity_sha256(value)
    object_sha = "8" * 64
    entry = {
        "range_id": deterministic_range_id(
            split=split,
            range_start=0,
            range_end=value.tokens,
            source_v1_range_start=0,
            source_v1_range_end=value.tokens,
            source_v1_manifest_sha256=SOURCE_SHA,
            token_identity_sha256=token_sha,
        ),
        "range_start": 0,
        "range_end": value.tokens,
        "source_v1_range_start": 0,
        "source_v1_range_end": value.tokens,
        "tokens": value.tokens,
        "host_id": "worker-0",
        "path": f"shards/host-worker-0/{object_sha}.safetensors",
        "sha256": object_sha,
        "token_identity_sha256": token_sha,
        "metadata_sha256": metadata_sha,
        "source_v1_manifest_sha256": SOURCE_SHA,
        "recomposition_sidecar_sha256": sidecar["sidecar_sha256"],
        "statistics": compute_contribution_statistics(
            value,
            normalized_shared_output=normalized,
        ),
    }
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "sealed": True,
        "metadata": metadata,
        "metadata_sha256": metadata_sha,
        "source_v1_manifest": {
            "schema_version": LEGACY_AGGREGATE_SCHEMA_VERSION,
            "manifest_sha256": SOURCE_SHA,
            "split": split,
            "total_tokens": value.tokens,
        },
        "recomposition_sidecar_sha256": sidecar["sidecar_sha256"],
        "total_tokens": value.tokens,
        "shard_count": 1,
        "statistics": entry["statistics"],
        "shards": [entry],
    }
    return seal_contribution_manifest(unsigned)


class _Dataset:
    def __init__(self, value: ContributionBatch, sidecar: dict, *, split: str):
        self.value = value
        self.total_tokens = value.tokens
        self.manifest = _manifest(value, sidecar, split=split)

    def iter_token_range(self, start: int, end: int):
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


class _SelectiveDataset(_Dataset):
    def __init__(self, value: ContributionBatch, sidecar: dict, *, split: str):
        super().__init__(value, sidecar, split=split)
        self.shared_reads: list[tuple[int, int]] = []

    def iter_token_range(self, start: int, end: int):
        raise AssertionError("shared-Q input must prefer the selective reader")

    def iter_shared_token_range(self, start: int, end: int):
        self.shared_reads.append((start, end))
        boundary = min(start + 2, end)
        for left, right in ((start, boundary), (boundary, end)):
            if left == right:
                continue
            yield ContributionSharedBatch(
                shared_output=np.ascontiguousarray(
                    self.value.shared_output[left:right]
                ),
                sequence_id=np.ascontiguousarray(
                    self.value.sequence_id[left:right]
                ),
                position=np.ascontiguousarray(self.value.position[left:right]),
            )


def test_shared_q_input_prefers_selective_shared_tensor_reader() -> None:
    sidecar = _sidecar()
    train = _Dataset(_wire(4), sidecar, split="train")
    evaluation = _SelectiveDataset(
        _wire(3, offset=0.2),
        sidecar,
        split="validation",
    )
    source = SharedQProcessInput(
        evaluation,
        sidecar,
        build_shared_q_centering(train, sidecar),
        role="evaluation",
        total_tokens=3,
        global_microbatch_tokens=4,
        microbatches_per_update=1,
    )

    batch = next(source.iter_updates())[0]

    assert batch.valid_mask.tolist() == [True, True, True, False]
    np.testing.assert_array_equal(batch.sequence_id, [100, 100, 101, -1])
    np.testing.assert_array_equal(batch.position, [0, 1, 2, 65_535])


def test_shared_q_process_lanes_are_disjoint_exhaustive_and_static() -> None:
    sidecar = _sidecar()
    train = _Dataset(_wire(10), sidecar, split="train")
    evaluation0 = _SelectiveDataset(_wire(10), sidecar, split="validation")
    evaluation1 = _SelectiveDataset(_wire(10), sidecar, split="validation")
    centering = build_shared_q_centering(train, sidecar)
    sources = [
        SharedQProcessInput(
            dataset,
            sidecar,
            centering,
            role="evaluation",
            total_tokens=10,
            global_microbatch_tokens=8,
            microbatches_per_update=2,
            process_count=2,
            process_index=index,
        )
        for index, dataset in enumerate((evaluation0, evaluation1))
    ]
    updates = [next(source.iter_updates()) for source in sources]

    assert evaluation0.shared_reads == [(0, 4), (4, 5)]
    assert evaluation1.shared_reads == [(5, 9), (9, 10)]
    assert [source.receipt.schema_version for source in sources] == [
        SHARED_Q_INPUT_SCHEMA_VERSION,
        SHARED_Q_INPUT_SCHEMA_VERSION,
    ]
    assert [
        (source.process_range_start, source.process_range_end)
        for source in sources
    ] == [(0, 5), (5, 10)]
    positions = [
        int(position)
        for update in updates
        for batch in update
        for position in batch.position[batch.valid_mask]
    ]
    assert sorted(positions) == list(range(10))
    assert all(batch.activation.shape == (4, HIDDEN_SIZE) for update in updates for batch in update)
    assert sum(batch.valid_mask.size for update in updates for batch in update) - len(positions) == 6


def test_shared_q_input_uses_exact_sidecar_mean_sst_and_tail_identity() -> None:
    sidecar = _sidecar()
    train = _Dataset(_wire(6), sidecar, split="train")
    evaluation = _Dataset(_wire(5, offset=0.1), sidecar, split="validation")
    centering = build_shared_q_centering(train, sidecar)
    train_normalized = bf16_bits_to_float32(
        maxtext_rmsnorm_bf16(
            train.value.shared_output,
            np.asarray(
                sidecar["tensors"]["shared_rmsnorm_scale"]["data"],
                np.uint16,
            ),
        )[1]
    )
    expected_mean = train_normalized.astype(np.float64).mean(axis=0).astype(
        np.float32
    )
    expected_train_sst = np.sum(
        np.square(
            train_normalized - expected_mean[None, :],
            dtype=np.float32,
        ),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(centering.mean, expected_mean)
    assert centering.training_sst == pytest.approx(
        expected_train_sst,
        rel=5e-7,
    )
    training_source = SharedQProcessInput(
        train,
        sidecar,
        centering,
        role="train",
        total_tokens=6,
        global_microbatch_tokens=4,
        microbatches_per_update=1,
    )
    source = SharedQProcessInput(
        evaluation,
        sidecar,
        centering,
        role="evaluation",
        total_tokens=5,
        global_microbatch_tokens=4,
        microbatches_per_update=1,
    )
    updates = tuple(source.iter_updates())
    rows = [batch for update in updates for batch in update]
    assert source.planned_updates == 2
    assert centering.training_capture_manifest_sha256 == (
        train.manifest["manifest_sha256"]
    )
    assert centering.recomposition_sidecar_sha256 == sidecar["sidecar_sha256"]
    assert training_source.sealed_sst == pytest.approx(centering.training_sst)
    assert source.sealed_sst > 0

    expected_bits = maxtext_rmsnorm_bf16(
        evaluation.value.shared_output,
        np.asarray(
            sidecar["tensors"]["shared_rmsnorm_scale"]["data"],
            np.uint16,
        ),
    )[1]
    actual = np.concatenate(
        [row.activation[row.valid_mask] for row in rows],
        axis=0,
    )
    np.testing.assert_array_equal(actual, bf16_bits_to_float32(expected_bits))
    np.testing.assert_array_equal(rows[-1].sequence_id, [102, -1, -1, -1])
    np.testing.assert_array_equal(rows[-1].position, [4, 65_535, 65_535, 65_535])

    changed = copy.deepcopy(sidecar)
    changed["sidecar_sha256"] = "9" * 64
    with pytest.raises(Exception):
        SharedQProcessInput(
            evaluation,
            changed,
            centering,
            role="evaluation",
            total_tokens=5,
        )
    with pytest.raises(SharedQError, match="split"):
        SharedQProcessInput(
            evaluation,
            sidecar,
            centering,
            role="train",
            total_tokens=5,
        )


def _small_config(*, parameter_dtype: str = "float32"):
    return shared_q_sae_config(
        SharedQLayout(hidden_dim=3, total_features=8, k=2),
        parameter_dtype=parameter_dtype,
        warmup_steps=0,
        total_steps=2,
        peak_lr=1e-3,
        min_lr=1e-3,
    )


def _small_microbatches():
    target = np.arange(18, dtype=np.float32).reshape(6, 3) / 20.0 + 0.2
    return [
        {
            "activation": target[:3],
            "valid_mask": np.asarray([True, True, True]),
        },
        {
            "activation": target[3:],
            "valid_mask": np.asarray([True, False, False]),
        },
    ]


def _assert_tree_close(left, right, *, atol: float = 5e-6):
    for actual, expected in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        if actual.dtype == np.bool_:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=2e-5,
                atol=atol,
            )


def test_shared_q_trainer_and_evaluator_match_dense_global_oracle() -> None:
    config = _small_config()
    distributed = DistributedConfig(microbatches_per_update=2)
    mesh = create_mesh(distributed)
    state = init_train_state(jax.random.key(3), config)
    state = state._replace(
        params=state.params._replace(
            bias=10.0 * jnp.ones_like(state.params.bias),
        )
    )
    mean = np.asarray([0.1, -0.2, 0.05], np.float32)
    microbatches = _small_microbatches()
    target = np.concatenate(
        [item["activation"] for item in microbatches],
        axis=0,
    )
    valid = np.concatenate(
        [item["valid_mask"] for item in microbatches],
        axis=0,
    )
    expected_state, expected_metrics = train_step(
        state,
        target,
        mean,
        config,
        valid_mask=valid,
    )
    sharded = init_shared_q_state(
        jax.random.key(3),
        mesh,
        config,
        distributed,
    )
    sharded = sharded._replace(params=state.params)
    trainer = SharedQDistributedTrainer(mesh, config, distributed)
    actual_state, actual_metrics = trainer.step(
        sharded,
        microbatches,
        mean,
    )
    _assert_tree_close(actual_state, expected_state)
    assert float(actual_metrics.loss) == pytest.approx(
        float(expected_metrics.loss),
        rel=2e-5,
        abs=5e-6,
    )
    assert int(actual_metrics.valid_tokens) == 4
    assert float(actual_metrics.l0) == 2

    evaluator = SharedQDistributedEvaluator(mesh, config, distributed)
    aux, rows = evaluator.step(state.params, microbatches, mean)
    oracle = forward(state.params, target, mean, config)
    oracle_reconstruction = np.asarray(oracle.reconstruction).reshape(2, 3, 3)
    oracle_reconstruction = np.where(
        rows.valid_mask[..., None],
        oracle_reconstruction,
        0.0,
    )
    np.testing.assert_allclose(
        rows.reconstruction,
        oracle_reconstruction,
        rtol=2e-5,
        atol=5e-6,
    )
    assert int(aux.valid_tokens) == 4
    assert int(aux.l0_sum) == 8
    assert rows.tokens == 4
    np.testing.assert_array_equal(rows.l0[rows.valid_mask], 2)
    assert float(aux.sse) == pytest.approx(
        float(np.sum(rows.token_sse, dtype=np.float64)),
        rel=2e-5,
    )


def test_shared_q_resident_and_rematerialized_graphs_match() -> None:
    config = _small_config()
    distributed = DistributedConfig(microbatches_per_update=2)
    mesh = create_mesh(distributed)
    state = init_shared_q_state(
        jax.random.key(9),
        mesh,
        config,
        distributed,
    )
    state = state._replace(
        params=state.params._replace(
            bias=10.0 * jnp.ones_like(state.params.bias),
        )
    )
    mean = np.asarray([0.1, -0.2, 0.05], np.float32)
    microbatches = _small_microbatches()
    resident = SharedQDistributedTrainer(
        mesh,
        config,
        distributed,
        rematerialize_microbatch=False,
    )
    rematerialized = SharedQDistributedTrainer(
        mesh,
        config,
        distributed,
        rematerialize_microbatch=True,
    )
    assert resident.compile(state, microbatches, mean) >= 0.0
    assert rematerialized.compile(state, microbatches, mean) >= 0.0
    resident_state, resident_metrics = resident.step(
        state,
        microbatches,
        mean,
    )
    remat_state, remat_metrics = rematerialized.step(
        state,
        microbatches,
        mean,
    )
    _assert_tree_close(resident_state, remat_state)
    np.testing.assert_allclose(
        np.asarray(resident_metrics.loss),
        np.asarray(remat_metrics.loss),
        rtol=1e-6,
        atol=1e-6,
    )


def _routed_and_q_evidence(sidecar: dict):
    tokens = 2
    wire = _wire(1)
    stages = recompose_gemma4(wire, sidecar)
    raw = stages.raw_routed_sum_f32.reshape(1, 1, HIDDEN_SIZE)
    q = bf16_bits_to_float32(
        stages.shared_post_norm_bf16_bits
    ).reshape(1, 1, HIDDEN_SIZE)
    valid = np.asarray([[True, False]])
    raw_target = np.concatenate((raw, np.zeros_like(raw)), axis=1)
    raw_estimate = raw_target.copy()
    route_zeros = np.zeros((1, tokens, TOP_K_EXPERTS), np.float32)
    routed_rows = ContributionEvaluationChunk(
        component_token_sse=np.zeros((1, tokens), np.float32),
        component_token_sst=np.zeros((1, tokens), np.float32),
        component_route_sse=route_zeros,
        component_route_sst=route_zeros,
        per_contribution_l0=np.where(
            valid[..., None],
            1,
            0,
        )
        * np.ones((1, tokens, TOP_K_EXPERTS), np.int32),
        raw_routed_target=raw_target,
        raw_routed_reconstruction=raw_estimate,
        bank_ids=np.zeros((1, tokens, TOP_K_EXPERTS), np.int32),
        router_weights=np.where(
            valid[..., None],
            1.0 / TOP_K_EXPERTS,
            0.0,
        ).astype(np.float32)
        * np.ones((1, tokens, TOP_K_EXPERTS), np.float32),
        valid_mask=valid,
    )
    sequence = np.asarray([[100, -1]], np.int64)
    position = np.asarray([[0, 65_535]], np.uint16)
    routed = ContributionEvaluationEvidenceChunk(
        update_index=0,
        sequence_id=sequence,
        position=position,
        scores=routed_rows,
    )
    q_target = np.concatenate((q, np.zeros_like(q)), axis=1)
    q_estimate = q_target.copy()
    q_estimate[0, 0] = q_estimate[0, 0, ::-1]
    q_rows = SharedQEvaluationRows(
        token_sse=np.zeros((1, tokens), np.float32),
        token_sst=np.zeros((1, tokens), np.float32),
        target=q_target,
        reconstruction=q_estimate,
        l0=np.asarray([[64, 0]], np.int32),
        valid_mask=valid,
    )
    q_evidence = SharedQEvaluationEvidenceChunk(
        update_index=0,
        capture_manifest_sha256="a" * 64,
        centering_contract_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        recomposition_sidecar_sha256=sidecar["sidecar_sha256"],
        sequence_id=sequence,
        position=position,
        rows=q_rows,
    )
    return wire, routed, q_evidence


def test_full_ffn_oracle_shared_and_learned_shared_are_exact_and_paired() -> None:
    sidecar = _sidecar(
        shared_scale=float32_to_bf16_bits(
            np.linspace(0.8, 1.2, HIDDEN_SIZE, dtype=np.float32)
        ),
        outer_scale=float32_to_bf16_bits(
            np.linspace(1.1, 0.7, HIDDEN_SIZE, dtype=np.float32)
        ),
    )
    wire, routed, q_evidence = _routed_and_q_evidence(sidecar)
    rows = recompose_full_ffn_rows(routed, q_evidence, sidecar)
    assert rows.shared_q_checkpoint_sha256 == q_evidence.checkpoint_sha256
    assert (
        rows.shared_q_centering_contract_sha256
        == q_evidence.centering_contract_sha256
    )
    expected = bf16_bits_to_float32(
        recompose_gemma4(wire, sidecar).outer_ffn_post_norm_bf16_bits
    )
    np.testing.assert_array_equal(rows.target[0, 0], expected[0])
    # Routed reconstruction is exact and oracle-shared uses exact q.
    np.testing.assert_array_equal(
        rows.oracle_shared_reconstruction[0, 0],
        expected[0],
    )
    assert not np.array_equal(
        rows.learned_shared_reconstruction[0, 0],
        expected[0],
    )
    scores = full_ffn_token_scores(
        rows,
        np.zeros((HIDDEN_SIZE,), np.float32),
    )
    assert scores.tokens == 1
    assert scores.oracle_ev == pytest.approx(1.0)
    assert scores.learned_ev < 1.0
    assert scores.shared_q_checkpoint_sha256 == q_evidence.checkpoint_sha256
    assert scores.oracle_shared_sse[0, 1] == 0
    assert scores.common_sst[0, 1] == 0

    misaligned = SharedQEvaluationEvidenceChunk(
        update_index=0,
        capture_manifest_sha256=q_evidence.capture_manifest_sha256,
        centering_contract_sha256=q_evidence.centering_contract_sha256,
        checkpoint_sha256=q_evidence.checkpoint_sha256,
        recomposition_sidecar_sha256=(
            q_evidence.recomposition_sidecar_sha256
        ),
        sequence_id=np.asarray([[101, -1]], np.int64),
        position=q_evidence.position,
        rows=q_evidence.rows,
    )
    with pytest.raises(SharedQError, match="misaligned"):
        recompose_full_ffn_rows(routed, misaligned, sidecar)


def test_two_axis_shared_q_collectives_and_bf16_moments() -> None:
    script = r"""
import jax
import jax.numpy as jnp
import numpy as np

from moe_sae.contribution_shared import (
    SharedQDistributedEvaluator,
    SharedQDistributedTrainer,
    SharedQLayout,
    init_shared_q_state,
    shared_q_sae_config,
)
from moe_sae.distributed import DistributedConfig, create_mesh

config = shared_q_sae_config(
    SharedQLayout(hidden_dim=3, total_features=8, k=2),
    parameter_dtype="bfloat16",
    warmup_steps=0,
    total_steps=2,
)
distributed = DistributedConfig(
    data_axis_size=2,
    feature_axis_size=2,
    microbatches_per_update=1,
)
mesh = create_mesh(distributed)
state = init_shared_q_state(jax.random.key(7), mesh, config, distributed)
state = state._replace(
    params=state.params._replace(
        bias=10 * jnp.ones_like(state.params.bias),
    )
)
batch = {
    "activation": np.arange(12, dtype=np.float32).reshape(4, 3) / 20 + 0.2,
    "valid_mask": np.asarray([True, True, True, False]),
}
mean = np.asarray([0.1, -0.2, 0.05], np.float32)
evaluator = SharedQDistributedEvaluator(mesh, config, distributed)
aux, rows = evaluator.step(state.params, [batch], mean)
assert int(aux.valid_tokens) == 3
assert int(aux.l0_sum) == 6
assert rows.tokens == 3
trainer = SharedQDistributedTrainer(mesh, config, distributed)
updated, metrics = trainer.step(state, [batch], mean)
assert updated.params.encoder.dtype == jnp.bfloat16
assert updated.optimizer.m.encoder.dtype == jnp.float32
assert updated.optimizer.v.encoder.dtype == jnp.float32
assert int(metrics.valid_tokens) == 3

poisoned = state._replace(
    optimizer=state.optimizer._replace(
        v=state.optimizer.v._replace(
            encoder=state.optimizer.v.encoder.at[-1, -1].set(jnp.nan),
        )
    )
)
try:
    trainer.step(poisoned, [batch], mean)
except Exception as error:
    assert "non-finite" in str(error)
else:
    raise AssertionError("shared-Q trainer accepted a dormant remote NaN")
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
