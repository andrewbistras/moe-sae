from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import moe_sae.experiment5_evaluation as evaluation_module
import moe_sae.experiment5_report as report_module
from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_distributed import (
    experiment5_physical_layout,
    experiment5_physical_to_logical,
)
from moe_sae.experiment5_evaluation import (
    Experiment5EvaluationAccumulator,
    Experiment5EvaluationError,
    Experiment5FeatureMaskShard,
    build_experiment5_evaluation_result,
    build_experiment5_training_receipt,
    logicalize_experiment5_checkpoint_firing_mask,
    pack_experiment5_device_rows,
    score_experiment5_rows,
    validate_experiment5_evaluation_partial,
    validate_experiment5_training_receipt,
)
from moe_sae.experiment5_hybrid import (
    Experiment5Layout,
    Experiment5Recomposition,
    numpy_recompose_experiment5_full_ffn,
)


def _hybrid_ids(bank_ids: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rows = bank_ids.shape[0]
    selected = np.full((rows, 64), -1, dtype=np.int32)
    for row in np.flatnonzero(valid):
        routed = np.concatenate(
            [
                bank_ids[row, route] * 512 + np.arange(4, dtype=np.int32)
                for route in range(8)
            ]
        )
        selected[row] = np.concatenate(
            (routed, 65_536 + np.arange(32, dtype=np.int32))
        )
    return selected


def _receipt(condition: str):
    actual = 2_000_000
    return build_experiment5_training_receipt(
        panel="sae",
        condition=condition,
        seed=0,
        checkpoint_target_tokens=actual,
        actual_training_tokens=actual,
        training_result_sha256="1" * 64,
        checkpoint_pointer_sha256="2" * 64,
        accelerator_type="v4-32",
        accelerator_generation="v4",
        chip_count=16,
        marginal_wall_seconds=100.0,
        marginal_chip_seconds=1_600.0,
        tokens_per_second=20_000.0,
        useful_mfu=0.2,
        routed_bank_component_counts=(
            [125_000] * 128 if condition in {"H", "R"} else None
        ),
    )


def _targets():
    expert = np.asarray(
        [
            [[0.1 * (route + 1), -0.05 * (route + 1)] for route in range(8)],
            [[-0.04 * (route + 1), 0.08 * (route + 1)] for route in range(8)],
        ],
        dtype=np.float32,
    )
    shared = np.asarray([[0.2, -0.3], [-0.5, 0.4]], dtype=np.float32)
    weights = np.full((2, 8), 0.125, dtype=np.float32)
    recomposition = Experiment5Recomposition(
        routed_scale=np.ones((2,), dtype=np.float32),
        outer_scale=np.ones((2,), dtype=np.float32),
    )
    return expert, shared, weights, recomposition


def _score(condition: str):
    expert, shared, weights, recomposition = _targets()
    valid = np.ones((2,), dtype=np.bool_)
    banks = np.tile(np.arange(8, dtype=np.int32), (2, 1))
    layout = Experiment5Layout(condition=condition, hidden_dim=2)
    target = numpy_recompose_experiment5_full_ffn(
        expert,
        shared,
        weights,
        recomposition,
    )
    if condition in {"H", "R"}:
        routed_hat = expert if condition == "H" else np.zeros_like(expert)
        shared_hat = shared if condition == "H" else np.zeros_like(shared)
        selected = _hybrid_ids(banks, valid)
        return score_experiment5_rows(
            chunk_id=f"{condition}-chunk",
            panel="sae",
            seed=0,
            checkpoint_target_tokens=2_000_000,
            layout=layout,
            sequence_id=np.asarray([10, 11], dtype=np.int64),
            position=np.asarray([0, 0], dtype=np.uint16),
            valid_mask=valid,
            selected_latent_ids=selected,
            expert_target=expert,
            shared_target=shared,
            router_weights=weights,
            full_target_mean=np.zeros((2,), dtype=np.float32),
            recomposition=recomposition,
            bank_ids=banks,
            routed_reconstruction=routed_hat,
            shared_reconstruction=shared_hat,
        )
    return score_experiment5_rows(
        chunk_id=f"{condition}-chunk",
        panel="sae",
        seed=0,
        checkpoint_target_tokens=2_000_000,
        layout=layout,
        sequence_id=np.asarray([10, 11], dtype=np.int64),
        position=np.asarray([0, 0], dtype=np.uint16),
        valid_mask=valid,
        selected_latent_ids=np.tile(
            np.arange(64, dtype=np.int32),
            (2, 1),
        ),
        expert_target=expert,
        shared_target=shared,
        router_weights=weights,
        full_target_mean=np.zeros((2,), dtype=np.float32),
        recomposition=recomposition,
        full_reconstruction=(
            np.zeros_like(target) if condition == "B" else target * 0.5
        ),
    )


def _mask(layout: Experiment5Layout) -> Experiment5FeatureMaskShard:
    return Experiment5FeatureMaskShard(
        logical_feature_ids=np.arange(layout.total_features, dtype=np.int32),
        fired=np.zeros((layout.total_features,), dtype=np.bool_),
    )


def test_reference_scorer_and_device_boundary_are_exact() -> None:
    rows = _score("H")
    assert rows.tokens == 2
    assert np.array_equal(rows.full_sse, np.zeros((2,)))
    assert np.array_equal(rows.routed_sse, np.zeros((2,)))
    assert np.array_equal(rows.shared_sse, np.zeros((2,)))

    device_scores = {
        field: getattr(rows, field)
        for field in (
            "selected_latent_ids",
            "bank_ids",
            "full_sse",
            "full_sst",
            "routed_sse",
            "shared_sse",
            "oracle_shared_full_sse",
            "oracle_routed_full_sse",
            "routed_target_sum",
            "routed_target_squared_norm_sum",
            "shared_target_sum",
            "shared_target_squared_norm_sum",
        )
    }
    packed = pack_experiment5_device_rows(
        chunk_id=rows.chunk_id,
        panel=rows.panel,
        condition=rows.condition,
        seed=rows.seed,
        checkpoint_target_tokens=rows.checkpoint_target_tokens,
        sequence_id=rows.sequence_id,
        position=rows.position,
        valid_mask=rows.valid_mask,
        device_scores=device_scores,
    )
    assert np.array_equal(packed.selected_latent_ids, rows.selected_latent_ids)
    assert not packed.selected_latent_ids.flags.writeable

    broken = dict(device_scores)
    broken["selected_latent_ids"] = rows.selected_latent_ids.copy()
    broken["selected_latent_ids"][0, 0] = 520
    with pytest.raises(
        Experiment5EvaluationError,
        match="routed-bank/shared ordering",
    ):
        pack_experiment5_device_rows(
            chunk_id=rows.chunk_id,
            panel=rows.panel,
            condition=rows.condition,
            seed=rows.seed,
            checkpoint_target_tokens=rows.checkpoint_target_tokens,
            sequence_id=rows.sequence_id,
            position=rows.position,
            valid_mask=rows.valid_mask,
            device_scores=broken,
        )


def test_physical_firing_mask_is_mapped_and_identity_bound() -> None:
    layout = Experiment5Layout(condition="H", hidden_dim=2)
    identity = experiment5_physical_layout(layout, 2)
    physical = np.zeros((layout.total_features,), dtype=np.bool_)
    physical[[0, 32_768, 34_816, 67_584]] = True
    shard = logicalize_experiment5_checkpoint_firing_mask(
        physical,
        layout=layout,
        feature_axis_size=2,
        expected_physical_layout_sha256=identity.identity_sha256,
    )
    permutation = experiment5_physical_to_logical(layout, 2)
    expected = np.zeros_like(physical)
    expected[permutation] = physical
    assert np.array_equal(shard.fired, expected)
    with pytest.raises(
        Experiment5EvaluationError,
        match="layout identity changed",
    ):
        logicalize_experiment5_checkpoint_firing_mask(
            physical,
            layout=layout,
            feature_axis_size=2,
            expected_physical_layout_sha256="0" * 64,
        )


def test_training_receipt_is_self_hashed_and_milestone_specific() -> None:
    receipt = _receipt("H")
    assert sum(receipt.routed_bank_component_counts or ()) == 16_000_000
    assert validate_experiment5_training_receipt(receipt) == receipt
    tampered = receipt.to_dict()
    tampered["routed_bank_component_counts"][0] += 1
    with pytest.raises(Experiment5EvaluationError, match="self-hash"):
        validate_experiment5_training_receipt(tampered)


def test_accumulator_rejects_duplicate_token_keys() -> None:
    layout = Experiment5Layout(condition="H", hidden_dim=2)
    accumulator = Experiment5EvaluationAccumulator(
        panel="sae",
        layout=layout,
        seed=0,
        milestone=2_000_000,
        training_receipt=_receipt("H"),
        checkpoint_firing_shards=[_mask(layout)],
        expected_evaluation_tokens=4,
    )
    rows = _score("H")
    accumulator.add(rows)
    with pytest.raises(Experiment5EvaluationError, match="duplicate token keys"):
        accumulator.add(dataclasses.replace(rows, chunk_id="second"))


def test_cross_process_partial_is_self_hashed_and_merges_exactly() -> None:
    layout = Experiment5Layout(condition="H", hidden_dim=2)
    source = Experiment5EvaluationAccumulator(
        panel="sae",
        layout=layout,
        seed=0,
        milestone=2_000_000,
        training_receipt=_receipt("H"),
        checkpoint_firing_shards=[_mask(layout)],
        expected_evaluation_tokens=2,
    )
    source.add(_score("H"))
    partial = source.export_partial(partial_id="process-001")
    assert validate_experiment5_evaluation_partial(partial) == partial

    coordinator = Experiment5EvaluationAccumulator(
        panel="sae",
        layout=layout,
        seed=0,
        milestone=2_000_000,
        training_receipt=_receipt("H"),
        checkpoint_firing_shards=[_mask(layout)],
        expected_evaluation_tokens=2,
    )
    coordinator.merge_partial(partial)
    assert coordinator.finalize().condition_result == source.finalize().condition_result
    with pytest.raises(
        Experiment5EvaluationError,
        match="duplicate evaluation partial ID",
    ):
        coordinator.merge_partial(partial)

    tampered = partial.to_dict()
    tampered["sequence_sse"][0] += 1.0
    with pytest.raises(Experiment5EvaluationError, match="self-hash"):
        validate_experiment5_evaluation_partial(tampered)

    overlap = partial.to_dict()
    overlap["partial_id"] = "process-002"
    overlap["chunk_ids"] = ["other-chunk"]
    overlap.pop("partial_sha256")
    overlap["partial_sha256"] = canonical_sha256(overlap)
    overlap_coordinator = Experiment5EvaluationAccumulator(
        panel="sae",
        layout=layout,
        seed=0,
        milestone=2_000_000,
        training_receipt=_receipt("H"),
        checkpoint_firing_shards=[_mask(layout)],
        expected_evaluation_tokens=4,
    )
    overlap_coordinator.merge_partial(partial)
    with pytest.raises(Experiment5EvaluationError, match="duplicate token keys"):
        overlap_coordinator.merge_partial(overlap)


def test_joint_finalizer_emits_valid_result_and_paired_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Keep this integration test tiny while exercising the same hard token-count
    # checks in both the evaluation finalizer and report validators.
    monkeypatch.setattr(evaluation_module, "EVAL_TOKENS", 2)
    monkeypatch.setattr(report_module, "EVAL_TOKENS", 2)
    accumulators = {}
    for condition in ("H", "R", "B", "F"):
        layout = Experiment5Layout(condition=condition, hidden_dim=2)
        accumulator = Experiment5EvaluationAccumulator(
            panel="sae",
            layout=layout,
            seed=0,
            milestone=2_000_000,
            training_receipt=_receipt(condition),
            checkpoint_firing_shards=[_mask(layout)],
            expected_evaluation_tokens=2,
        )
        accumulator.add(_score(condition))
        accumulators[condition] = accumulator
    result, paired = build_experiment5_evaluation_result(
        accumulators=accumulators,
        source_revision="a" * 40,
        config_sha256="b" * 64,
        completed_at="2026-07-28T12:00:00Z",
    )
    assert result["evaluation_tokens"] == 2
    assert result["conditions"]["H"]["full_ffn"]["sse"] == 0
    assert result["predictions_artifact_sha256"] == paired["artifact_sha256"]
