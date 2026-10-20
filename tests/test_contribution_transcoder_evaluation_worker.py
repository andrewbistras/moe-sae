"""Durable worker-boundary tests for Experiment-3 evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_transcoder_evaluation import (
    ContributionTranscoderTokenScores,
)
from moe_sae.contribution_transcoder_evaluation_worker import (
    Experiment3EvaluationWorkerError,
    _concatenate_condition_evidence,
    _evaluation_capacity_roles,
    _load_process_evidence,
    _seal_process_evidence,
    _training_ledger,
    _validate_phase_datasets,
    _validate_run_contract,
)
from moe_sae.contribution_transcoder_config import (
    PHASE_NAME,
    load_contribution_transcoder_config,
)


def _scores(tokens: int, *, offset: float = 0.0):
    component_sse = np.arange(tokens, dtype=np.float64) + 1.0 + offset
    component_sst = np.full(tokens, 10.0, np.float64)
    return ContributionTranscoderTokenScores(
        component_sse=component_sse,
        component_sst=component_sst,
        routed_sse=component_sse * 1.1,
        routed_sst=component_sst * 1.2,
        oracle_shared_sse=component_sse * 0.9,
        oracle_shared_sst=component_sst * 0.8,
        route_component_sse=np.broadcast_to(
            component_sse[:, None] / 8,
            (tokens, 8),
        ).copy(),
        route_component_sst=np.broadcast_to(
            component_sst[:, None] / 8,
            (tokens, 8),
        ).copy(),
        valid_mask=np.ones(tokens, np.bool_),
        recomposition_sidecar_sha256="5" * 64,
    )


def _seal(root: Path, process: int):
    tokens = 2
    lane = root / f"process-p{process:05d}"
    return _seal_process_evidence(
        lane,
        condition="A",
        process_index=process,
        checkpoint_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        target_manifest_sha256="3" * 64,
        input_receipt_sha256=f"{process + 6:x}" * 64,
        centering_receipt_sha256="4" * 64,
        capacity_contract_sha256="9" * 64,
        evaluation_wall_seconds=3.0 + process,
        fired_mask=np.asarray(
            [True, False, process == 1, True],
            np.bool_,
        ),
        sequence_id=np.asarray(
            [10 + process * 10, 10 + process * 10],
            np.int64,
        ),
        position=np.asarray([0, 1], np.uint16),
        expert_ids=np.tile(np.arange(8, dtype=np.uint8), (tokens, 1)),
        scores=_scores(tokens, offset=float(process)),
    )


def test_process_evidence_is_content_addressed_and_concatenates(tmp_path: Path):
    first = _seal(tmp_path, 0)
    second = _seal(tmp_path, 1)
    rows = [
        _load_process_evidence(
            tmp_path / f"process-p{process:05d}" / "A-evidence.json"
        )
        for process in range(2)
    ]
    summary, sequence, position, expert_ids, scores = (
        _concatenate_condition_evidence(rows)
    )

    assert first["data_sha256"] != second["data_sha256"]
    assert summary["tokens"] == 4
    assert summary["evaluation_wall_seconds"] == 4.0
    np.testing.assert_array_equal(sequence, [10, 10, 20, 20])
    np.testing.assert_array_equal(position, [0, 1, 0, 1])
    assert expert_ids.shape == (4, 8)
    assert scores.tokens == 4
    np.testing.assert_array_equal(
        summary["fired_mask"],
        [True, False, True, True],
    )


def test_process_evidence_rejects_content_corruption(tmp_path: Path):
    metadata = _seal(tmp_path, 0)
    data = Path(metadata["data_path"])
    data.write_bytes(data.read_bytes() + b"x")
    with pytest.raises(
        Experiment3EvaluationWorkerError,
        match="content identity",
    ):
        _load_process_evidence(
            tmp_path / "process-p00000" / "A-evidence.json"
        )


def test_run_contract_proves_source_only_manifest_binding(tmp_path: Path):
    body = {
        "schema_version": "gemma4-contribution-sae-operator-run-v1",
        "scientific_model": "transcoder",
        "condition": "A",
        "seed": 0,
        "project_commit": "b" * 40,
        "source_overlay_manifest_sha256": "1" * 64,
        "capture_manifest_sha256": "2" * 64,
    }
    contract = {**body, "run_contract_sha256": canonical_sha256(body)}
    (tmp_path / "run-contract.json").write_text(json.dumps(contract))

    assert (
        _validate_run_contract(
            tmp_path,
            condition="A",
            source_manifest_sha256="1" * 64,
            target_manifest_sha256="2" * 64,
            training_source_revision="b" * 40,
        )["scientific_model"]
        == "transcoder"
    )
    with pytest.raises(
        Experiment3EvaluationWorkerError,
        match="selected source-only run",
    ):
        _validate_run_contract(
            tmp_path,
            condition="A",
            source_manifest_sha256="1" * 64,
            target_manifest_sha256="2" * 64,
            training_source_revision="a" * 40,
        )
    changed = dict(contract)
    changed["scientific_model"] = "sae"
    unsigned = {
        name: value
        for name, value in changed.items()
        if name != "run_contract_sha256"
    }
    changed["run_contract_sha256"] = canonical_sha256(unsigned)
    (tmp_path / "run-contract.json").write_text(json.dumps(changed))
    with pytest.raises(
        Experiment3EvaluationWorkerError,
        match="source-only",
    ):
        _validate_run_contract(
            tmp_path,
            condition="A",
            source_manifest_sha256="1" * 64,
            target_manifest_sha256="2" * 64,
            training_source_revision="b" * 40,
        )


def test_evaluator_rejects_a_different_train_or_eval_overlay():
    config = load_contribution_transcoder_config(
        "configs/gemma4_transcoder_v2.yaml"
    )
    phase = config.phase(PHASE_NAME)
    train_target = SimpleNamespace(
        total_tokens=phase.train_tokens,
        manifest_sha256=config.parents["train_target_manifest_sha256"],
    )
    eval_target = SimpleNamespace(
        total_tokens=phase.eval_tokens,
        manifest_sha256=config.parents["eval_target_manifest_sha256"],
    )
    train_source = SimpleNamespace(
        total_tokens=phase.train_tokens,
        manifest_sha256=config.parents[
            "train_source_overlay_manifest_sha256"
        ],
    )
    eval_source = SimpleNamespace(
        total_tokens=phase.eval_tokens,
        manifest_sha256=config.parents[
            "eval_source_overlay_manifest_sha256"
        ],
    )
    arguments = {
        "train_target": train_target,
        "train_source": train_source,
        "eval_target": eval_target,
        "eval_source": eval_source,
    }
    _validate_phase_datasets(config, phase, **arguments)

    for role in ("train_source", "eval_source"):
        changed = dict(arguments)
        selected = changed[role]
        changed[role] = SimpleNamespace(
            total_tokens=selected.total_tokens,
            manifest_sha256="0" * 64,
        )
        with pytest.raises(
            Experiment3EvaluationWorkerError,
            match="frozen phase",
        ):
            _validate_phase_datasets(config, phase, **changed)

    capture_contract = "c" * 64
    horizon_config = SimpleNamespace(
        parents={
            **config.parents,
            "source_overlay_capture_contract_sha256": capture_contract,
        }
    )
    train_source.manifest = {
        "metadata": {"config_sha256": capture_contract}
    }
    _validate_phase_datasets(horizon_config, phase, **arguments)
    train_source.manifest["metadata"]["config_sha256"] = "d" * 64
    with pytest.raises(
        Experiment3EvaluationWorkerError,
        match="config-pinned capture contract",
    ):
        _validate_phase_datasets(horizon_config, phase, **arguments)


def test_training_ledger_uses_measured_marginal_cost_and_diagnostic_mfu():
    layout = ContributionSAEConfig(
        condition="A",
        hidden_dim=3,
        num_experts=4,
        top_routed=2,
        bank_width=4,
        k_per_contribution=2,
        randomization_key=0,
    )
    result = {
        "result_sha256": "1" * 64,
        "compile_seconds": 2.0,
        "lowering_seconds": 1.0,
        "preprocessing_wall_seconds": 3.0,
        "optimizer_wall_seconds": 7.0,
        "marginal_wall_seconds": 10.0,
        "checkpoint_seconds": 4.0,
        "chip_seconds": 160.0,
        "chip_seconds_scope": "test",
        "chip_seconds_excludes": ["compile"],
        "tokens_per_second": 100.0,
        "tokens": 1_000,
    }
    ledger = _training_ledger(
        result,
        layout=layout,
        evaluation_wall_seconds=5.0,
        peak_flops_per_second=1_000_000_000,
    )

    assert ledger.chip_count == 16
    assert ledger.marginal_chip_seconds == 160.0
    assert ledger.compilation_wall_seconds == 3.0
    assert ledger.total_wall_seconds == 22.0
    assert ledger.useful_mfu > 0


def test_curve_ledger_uses_checkpoint_cumulative_timing_not_final_cost():
    layout = ContributionSAEConfig(
        condition="A",
        hidden_dim=3,
        num_experts=4,
        top_routed=2,
        bank_width=4,
        k_per_contribution=2,
        randomization_key=0,
    )
    final = {
        "result_sha256": "1" * 64,
        "compile_seconds": 20.0,
        "lowering_seconds": 10.0,
        "preprocessing_wall_seconds": 30.0,
        "optimizer_wall_seconds": 70.0,
        "marginal_wall_seconds": 100.0,
        "checkpoint_seconds": 40.0,
        "chip_seconds": 1_600.0,
        "chip_seconds_scope": "test",
        "chip_seconds_excludes": ["compile"],
        "tokens_per_second": 100_000.0,
        "tokens": 10_000_000,
    }
    pointer = {
        "pointer_sha256": "2" * 64,
        "tokens_seen": 2_015_232,
    }
    metadata = {
        "timing": {
            "compile_seconds": 2.0,
            "lowering_seconds": 1.0,
            "preprocessing_wall_seconds": 3.0,
            "optimizer_wall_seconds": 7.0,
            "checkpoint_seconds_before_save": 4.0,
            "global_chip_count": 16,
        }
    }

    ledger = _training_ledger(
        final,
        layout=layout,
        evaluation_wall_seconds=5.0,
        peak_flops_per_second=1_000_000_000,
        checkpoint_pointer=pointer,
        checkpoint_metadata=metadata,
    )

    assert ledger.marginal_wall_seconds == 10.0
    assert ledger.marginal_chip_seconds == 160.0
    assert ledger.tokens_per_second == pytest.approx(201_523.2)
    assert ledger.total_wall_seconds == 22.0


@pytest.mark.parametrize("condition", ["A", "R"])
def test_restore_keeps_training_capacity_while_eval_uses_larger_capacity(
    condition: str,
) -> None:
    roles = _evaluation_capacity_roles(
        condition,
        training_capacity=8,
        evaluation_capacity=24,
    )

    assert roles.checkpoint == 8
    assert roles.execution == 24


@pytest.mark.parametrize("condition", ["B", "F"])
def test_rank_banked_capacity_roles_remain_absent(condition: str) -> None:
    roles = _evaluation_capacity_roles(
        condition,
        training_capacity=None,
        evaluation_capacity=None,
    )

    assert roles.checkpoint is None
    assert roles.execution is None
