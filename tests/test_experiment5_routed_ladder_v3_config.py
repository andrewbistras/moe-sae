from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.contribution_performance import ExactPerformanceIdentity
from moe_sae.contribution_training_worker import (
    ContributionTrainingWorkerError,
    TrainingWorkerPlan,
    _contribution_kernel_condition,
    _load_worker_config,
    _run_identity,
    direct_resident_runtime_limit_seconds,
)
from moe_sae.experiment5_routed_ladder_v2_config import (
    load_routed_ladder_v2_config,
)
from moe_sae.experiment5_routed_ladder_evaluation_worker import (
    _encoder_activation,
    _evaluated_conditions,
    _layout,
    _load_routed_ladder_config,
    _phase_name,
    _topk_requires_strictly_positive,
    load_merged_scores,
    load_process_evidence,
    merge_process_evidence,
    seal_merged_result,
    seal_process_evidence,
)
from moe_sae.experiment5_routed_ladder_v3_config import (
    A256_NATIVE_CONFIG_SHA256,
    A256_NATIVE_LABEL,
    A256_SCALE_INTERPRETATION,
    ANALYZED_CONDITIONS,
    CONDITIONS,
    PHASE_NAME,
    PROTOCOL_ID,
    PRIMARY_CONTRAST_LABEL,
    PRIMARY_ESTIMAND,
    REQUIRED_SPARSITY_METRICS,
    SCHEMA_VERSION,
    SEEDS,
    V2_CONFIG_SHA256,
    RoutedLadderV3ConfigError,
    load_routed_ladder_v3_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "sae": ROOT / "configs" / "gemma4_experiment5_routed_ladder_sae_v3.yaml",
    "transcoder": (
        ROOT / "configs" / "gemma4_experiment5_routed_ladder_transcoder_v3.yaml"
    ),
}
V2_CONFIGS = {
    "sae": ROOT / "configs" / "gemma4_experiment5_routed_ladder_sae_v2.yaml",
    "transcoder": (
        ROOT / "configs" / "gemma4_experiment5_routed_ladder_transcoder_v2.yaml"
    ),
}
EXPECTED_CONFIG_SHA256 = {
    "sae": "725c1d846bc2189096f5742a758e6845db4d7cad1fdc203d950e50de152bafc5",
    "transcoder": (
        "059d845bbb7b1140ac50f6866efc7727fd2ee88fcc4e788d147ddcfa86c9798f"
    ),
}


@pytest.mark.parametrize(("panel", "path"), CONFIGS.items())
def test_checked_in_v3_contract_is_exact(panel: str, path: Path):
    config = load_routed_ladder_v3_config(path, expected_panel=panel)

    assert config.config_hash == EXPECTED_CONFIG_SHA256[panel]
    assert config.config_hash == canonical_sha256(config.raw)
    assert config.raw["schema_version"] == SCHEMA_VERSION
    assert config.phase(PHASE_NAME).seeds == SEEDS
    assert config.phase(PHASE_NAME).planned_updates == 611
    assert config.runtime.direct_runtime_max_seconds == 43_200
    assert tuple(item.condition for item in config.layouts) == CONDITIONS
    assert tuple(config.raw["evaluation"]["ladder_conditions"]) == (
        ANALYZED_CONDITIONS
    )
    assert _load_worker_config(path, panel).config_hash == config.config_hash
    assert (
        direct_resident_runtime_limit_seconds(
            config_path=path,
            accelerator_type="v4-32",
            phase=PHASE_NAME,
        )
        == 43_200
    )


@pytest.mark.parametrize("path", CONFIGS.values())
def test_v3_matches_geometry_and_selected_slots_but_not_exact_positive_l0(
    path: Path,
):
    config = load_routed_ladder_v3_config(path)
    relu = config.layout_for_condition("A512ReLUInclusiveTopK")
    softplus = config.layout_for_condition("A512Softplus")

    for layout in (relu, softplus):
        assert layout.worker_condition == "A"
        assert layout.bank_count == 128
        assert layout.bank_width == 512
        assert layout.components_per_token == 8
        assert layout.topk_per_component == 4
        assert layout.total_features == 65_536
        assert layout.candidates_per_token == 4_096
        assert layout.selected_slots_per_token == layout.total_l0 == 32
        assert layout.maximum_positive_l0 == 32
        assert layout.trainable_parameters == 369_164_288
    assert relu.training_seeds == SEEDS
    assert softplus.training_seeds == ()

    assert relu.encoder_activation == "relu"
    assert relu.selection_policy == "inclusive_zero_topk"
    assert relu.topk_requires_strictly_positive is False
    assert relu.positive_l0_contract == "bounded_zero_to_selected_slots"
    assert relu.softplus_positive_floor is None

    assert softplus.encoder_activation == "softplus"
    assert softplus.selection_policy == "exact_positive_topk"
    assert softplus.topk_requires_strictly_positive is True
    assert softplus.positive_l0_contract == "exact_selected_slots"
    assert softplus.softplus_positive_floor == "smallest_normal_float32"


@pytest.mark.parametrize(("panel", "path"), CONFIGS.items())
def test_v3_a256_is_panel_native_read_only_context(panel: str, path: Path):
    config = load_routed_ladder_v3_config(path)
    reference = config.existing_a256_native_reference

    assert reference.condition == "A256Native"
    assert reference.source_label == A256_NATIVE_LABEL[panel]
    assert reference.source_config_sha256 == A256_NATIVE_CONFIG_SHA256[panel]
    assert reference.reused_seeds == (0,)
    assert reference.training_policy == "reuse_existing_seed0_only"
    assert reference.bank_width == 256
    assert reference.topk_per_component == 2
    assert config.raw["evaluation"]["a256_native_training_allowed"] is False
    with pytest.raises(RoutedLadderV3ConfigError, match="trainable v3 layout"):
        config.layout_for_condition("A256Native")


@pytest.mark.parametrize(("panel", "path"), CONFIGS.items())
def test_v3_report_names_the_actual_estimand(panel: str, path: Path):
    report = load_routed_ladder_v3_config(path).report_contract

    assert report.primary_contrast_label == PRIMARY_CONTRAST_LABEL
    assert report.estimand == PRIMARY_ESTIMAND
    assert report.paired_seeds == SEEDS
    assert report.fixed_bank_width is True
    assert report.fixed_parameter_count is True
    assert report.fixed_candidate_count is True
    assert report.fixed_max_selected_slots is True
    assert report.fixed_exact_positive_l0 is False
    assert report.required_sparsity_metrics == REQUIRED_SPARSITY_METRICS
    assert report.a256_panel_native is True
    assert report.a256_scale_interpretation == A256_SCALE_INTERPRETATION[panel]


@pytest.mark.parametrize(("panel", "path"), CONFIGS.items())
def test_v3_pins_v2_as_terminal_evidence_not_quality_data(
    panel: str,
    path: Path,
):
    prior = load_routed_ladder_v3_config(path).prior_v2_reference

    assert prior.config_sha256 == V2_CONFIG_SHA256[panel]
    assert prior.artifact_policy == "immutable_read_only"
    assert prior.unmapped_quality_grid_allowed is False
    assert prior.evidentiary_role == "terminal_feasibility_evidence_only"
    assert prior.terminal_reason == "fewer_than_k_positive_candidates"


@pytest.mark.parametrize(("panel", "path"), CONFIGS.items())
def test_v3_imports_only_completed_equivalent_v2_a512_cells(
    panel: str,
    path: Path,
):
    config = load_routed_ladder_v3_config(path)
    relu, softplus = config.v2_condition_imports

    assert softplus.condition == "A512Softplus"
    assert softplus.import_policy == "reuse_completed_v2_exact_graph"
    assert softplus.reused_seeds == SEEDS
    assert softplus.new_training_seeds == ()
    assert softplus.failed_or_partial_import_allowed is False

    assert relu.import_policy == "train_v3_from_scratch"
    assert relu.reused_seeds == ()
    assert relu.new_training_seeds == SEEDS
    assert (
        relu.equivalence_basis
        == "relu_selection_policy_boundary_not_importable"
    )


@pytest.mark.parametrize(("panel", "path"), CONFIGS.items())
def test_v3_launch_plan_authorizes_only_fresh_relu_trajectories(
    panel: str,
    path: Path,
):
    config = _load_worker_config(path, panel)

    for seed in SEEDS:
        plan = TrainingWorkerPlan.from_config(
            config,
            phase=PHASE_NAME,
            condition="A512ReLUInclusiveTopK",
            seed=seed,
        )
        assert plan.condition == "A512ReLUInclusiveTopK"
        assert plan.seed == seed
        assert plan.total_tokens == 10_000_000
        assert plan.planned_updates == 611
        assert plan.checkpoint_tokens[-1] == 10_000_000
        identity = ExactPerformanceIdentity(
            graph_sha256="1" * 64,
            hardware_sha256="2" * 64,
            config_sha256=config.config_hash,
        )
        run_identity = _run_identity(
            plan,
            identity=identity,
            mode="resident",
            admission_basis="bounded_ungated_resident",
            admission_evidence_sha256=None,
        )
        assert run_identity["plan"]["condition"] == (
            "A512ReLUInclusiveTopK"
        )
        assert run_identity["plan"]["phase_name"] == PHASE_NAME
        assert run_identity["performance_identity"]["config_sha256"] == (
            config.config_hash
        )

    with pytest.raises(
        ContributionTrainingWorkerError,
        match="read-only historical reference",
    ):
        TrainingWorkerPlan.from_config(
            config,
            phase=PHASE_NAME,
            condition="A512Softplus",
            seed=0,
        )
    assert _contribution_kernel_condition("A512ReLUInclusiveTopK") == "A"


@pytest.mark.parametrize(("panel", "path"), CONFIGS.items())
def test_v3_evaluator_dispatches_the_relu_inclusive_policy(
    panel: str,
    path: Path,
):
    config = _load_routed_ladder_config(path, expected_panel=panel)

    assert _phase_name(config) == PHASE_NAME
    assert _evaluated_conditions(config) == ANALYZED_CONDITIONS
    layout = _layout(config, "A512ReLUInclusiveTopK", seed=0)
    assert layout.bank_width == 512
    assert layout.k_per_contribution == 4
    assert _encoder_activation(config, "A512ReLUInclusiveTopK") == "relu"
    assert (
        _topk_requires_strictly_positive(
            config,
            "A512ReLUInclusiveTopK",
        )
        is False
    )


def test_v3_evaluation_evidence_reports_actual_positive_l0(tmp_path: Path):
    valid = np.ones((2,), dtype=np.bool_)
    scores = SimpleNamespace(
        component_sse=np.array([1.0, 2.0]),
        component_sst=np.array([4.0, 5.0]),
        routed_sse=np.array([2.0, 3.0]),
        routed_sst=np.array([6.0, 7.0]),
        oracle_shared_sse=np.array([3.0, 4.0]),
        oracle_shared_sst=np.array([8.0, 9.0]),
        valid_mask=valid,
    )
    root = tmp_path / "evidence"
    seal_process_evidence(
        root,
        panel="sae",
        condition="A512ReLUInclusiveTopK",
        seed=0,
        checkpoint_tokens=2_000_000,
        checkpoint_actual_tokens=2_000_000,
        process_index=0,
        process_count=1,
        source_revision="a" * 40,
        training_source_revision="b" * 40,
        config_sha256="1" * 64,
        config_file_sha256="2" * 64,
        h_config_sha256=None,
        h_config_file_sha256=None,
        training_result_sha256="3" * 64,
        training_contract_sha256="4" * 64,
        checkpoint_sha256="5" * 64,
        parent_training_result_sha256="6" * 64,
        parent_training_result_file_sha256="7" * 64,
        parent_run_contract_sha256="8" * 64,
        parent_run_contract_file_sha256="9" * 64,
        parent_checkpoint_pointer_sha256="a" * 64,
        parent_checkpoint_pointer_file_sha256="b" * 64,
        target_manifest_sha256="c" * 64,
        source_manifest_sha256=None,
        recomposition_sidecar_sha256="d" * 64,
        centering_receipt_sha256="e" * 64,
        input_receipt_sha256="f" * 64,
        learned_input_receipt_sha256=None,
        evaluation_wall_seconds=1.0,
        sequence_id=np.array([1, 2], dtype=np.int64),
        position=np.array([0, 0], dtype=np.uint16),
        scores=scores,
        protocol_id=PROTOCOL_ID,
        selected_slot_count=32,
        positive_l0_sum=50,
    )
    row = load_process_evidence(root / "process-evidence.json")
    assert row[0]["positive_l0"] == 25.0
    assert row[0]["zero_selected_slot_fraction"] == pytest.approx(
        7.0 / 32.0
    )
    summary, tensors = merge_process_evidence([row])
    assert summary["selected_slot_count"] == 32
    assert summary["positive_l0_sum"] == 50
    assert summary["positive_l0"] == 25.0
    assert summary["zero_selected_slot_fraction"] == pytest.approx(
        7.0 / 32.0
    )
    seal_merged_result(
        root,
        summary=summary,
        tensors=tensors,
        expected_tokens=2,
    )
    loaded = load_merged_scores(root / "evaluation-result.json")
    assert loaded.condition == "A512ReLUInclusiveTopK"


@pytest.mark.parametrize(("panel", "path"), V2_CONFIGS.items())
def test_v2_contract_remains_independently_loadable_and_identity_pinned(
    panel: str,
    path: Path,
):
    config = load_routed_ladder_v2_config(path, expected_panel=panel)

    assert config.config_hash == V2_CONFIG_SHA256[panel]


@pytest.mark.parametrize(
    ("path_parts", "replacement", "message"),
    [
        (
            ("layouts", 0, "selection_policy"),
            "exact_positive_topk",
            r"layouts\[0\].selection_policy",
        ),
        (
            ("layouts", 0, "topk_requires_strictly_positive"),
            True,
            r"layouts\[0\].topk_requires_strictly_positive",
        ),
        (
            ("report_contract", "matched_quantities", "exact_positive_l0"),
            True,
            "report_contract.matched_quantities.exact_positive_l0",
        ),
        (
            ("evaluation", "condition_specific_v2_import_policy"),
            "all_v2_cells",
            "evaluation.condition_specific_v2_import_policy",
        ),
        (
            ("v2_condition_imports", 0, "failed_or_partial_import_allowed"),
            True,
            r"v2_condition_imports\[0\].failed_or_partial_import_allowed",
        ),
        (
            ("prior_v2_reference", "artifact_policy"),
            "resume_allowed",
            "prior_v2_reference.artifact_policy",
        ),
    ],
)
def test_v3_policy_and_evidence_drift_fail_closed(
    tmp_path: Path,
    path_parts: tuple[str | int, ...],
    replacement: object,
    message: str,
):
    raw = yaml.safe_load(CONFIGS["sae"].read_text(encoding="utf-8"))
    selected = raw
    for part in path_parts[:-1]:
        selected = selected[part]
    selected[path_parts[-1]] = replacement
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(RoutedLadderV3ConfigError, match=message):
        load_routed_ladder_v3_config(changed)


def test_cross_panel_a256_reference_cannot_be_relabelled(tmp_path: Path):
    raw = yaml.safe_load(CONFIGS["transcoder"].read_text(encoding="utf-8"))
    raw["existing_a256_native_reference"]["source_label"] = "A256ReLU"
    raw["existing_a256_native_reference"]["encoder_activation"] = "relu"
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        RoutedLadderV3ConfigError,
        match="existing_a256_native_reference.source_label",
    ):
        load_routed_ladder_v3_config(changed)


def test_v3_document_records_zero_slots_as_scientific_measurement():
    text = (
        ROOT / "docs" / "EXPERIMENT_5_LADDER_V3.md"
    ).read_text(encoding="utf-8")

    assert "Zero values are eligible" in text
    assert "may never turn zero into epsilon" in text
    assert "exact positive L0 is **not** fixed" in text
    assert "structural evidence" in text
