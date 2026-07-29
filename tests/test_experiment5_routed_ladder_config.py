from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from moe_sae.contribution_training_worker import (
    TrainingWorkerPlan,
    _contribution_kernel_condition,
    _load_worker_config,
    direct_resident_runtime_limit_seconds,
    prefetch_runtime_limit_seconds,
)
from moe_sae.experiment5_routed_ladder_config import (
    CONDITIONS,
    EXPERIMENT4_PROTOCOL_ID,
    EXPERIMENT5_PROTOCOL_ID,
    H_CONFIG_SHA256,
    PHASE_NAME,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    RoutedLadderConfigError,
    load_routed_ladder_config,
)

ROOT = Path(__file__).resolve().parents[1]
SAE_CONFIG = ROOT / "configs" / "gemma4_experiment5_routed_ladder_sae_v1.yaml"
TRANSCODER_CONFIG = (
    ROOT / "configs" / "gemma4_experiment5_routed_ladder_transcoder_v1.yaml"
)
EXPERIMENT5_CONFIG = {
    "sae": ROOT / "configs" / "gemma4_hybrid_sae_v1.yaml",
    "transcoder": ROOT / "configs" / "gemma4_hybrid_transcoder_v1.yaml",
}
PROTOCOL_DOC = ROOT / "docs" / "EXPERIMENT_5.md"
OPERATIONS_DOC = ROOT / "docs" / "EXPERIMENT_5_OPERATIONS.md"
EXPECTED_CONFIG_SHA256 = {
    "sae": "05ee90c14cb956ca18ef0038993fcb7493d0930dfa6f6df2d91cc5dc7700fd44",
    "transcoder": ("fc1e35e29dfddf7bcab135d5f669a0130742e804dc4675ba495a828181407add"),
}


@pytest.mark.parametrize(
    ("panel", "path"),
    [
        ("sae", SAE_CONFIG),
        ("transcoder", TRANSCODER_CONFIG),
    ],
)
def test_checked_in_ladder_is_exact_worker_compatible_contract(
    panel: str,
    path: Path,
):
    config = load_routed_ladder_config(path, expected_panel=panel)
    phase = config.phase(PHASE_NAME)

    assert config.config_hash == EXPECTED_CONFIG_SHA256[panel]
    assert config.protocol_id == PROTOCOL_ID
    assert config.raw["schema_version"] == SCHEMA_VERSION
    assert config.scientific_model == panel
    assert config.parent_protocol_ids == {
        "experiment_4": EXPERIMENT4_PROTOCOL_ID,
        "experiment_5": EXPERIMENT5_PROTOCOL_ID,
    }
    assert phase.layout_set == "true_id_routed_activation_ladder"
    assert phase.seeds == (0, 1, 2)
    assert phase.train_tokens == 10_000_000
    assert phase.eval_tokens == 250_000
    assert phase.planned_updates == 611
    assert phase.warmup_updates == 61
    assert phase.tail_mask_tokens == 10_624
    assert phase.checkpoint_tokens == tuple(range(1_000_000, 10_000_001, 1_000_000))
    assert phase.evaluation_checkpoint_tokens == (
        2_000_000,
        5_000_000,
        10_000_000,
    )
    assert config.runtime.effective_batch_tokens == 16_384
    assert config.runtime.microbatch_tokens == 4_096
    assert config.runtime.direct_runtime_max_seconds == 43_200


@pytest.mark.parametrize("path", [SAE_CONFIG, TRANSCODER_CONFIG])
def test_ladder_trains_only_full_sized_activation_pair_and_maps_both_to_a(
    path: Path,
):
    config = load_routed_ladder_config(path)

    assert tuple(layout.condition for layout in config.layouts) == CONDITIONS
    assert [
        (
            layout.base_condition,
            layout.worker_condition,
            layout.bank_count,
            layout.bank_width,
            layout.components_per_token,
            layout.topk_per_component,
            layout.total_features,
            layout.candidates_per_token,
            layout.total_l0,
            layout.trainable_parameters,
            layout.encoder_activation,
            layout.softplus_positive_floor,
        )
        for layout in config.layouts_for_phase(PHASE_NAME)
    ] == [
        (
            "A",
            "A",
            128,
            512,
            8,
            4,
            65_536,
            4_096,
            32,
            369_164_288,
            "relu",
            None,
        ),
        (
            "A",
            "A",
            128,
            512,
            8,
            4,
            65_536,
            4_096,
            32,
            369_164_288,
            "softplus",
            "smallest_normal_float32",
        ),
    ]
    assert all(
        layout.target == "raw_selected_expert_output"
        and layout.bank_identity == "true_expert_id"
        for layout in config.layouts
    )
    with pytest.raises(RoutedLadderConfigError, match="one phase"):
        config.phase("routed_shared_hybrid_10m")


@pytest.mark.parametrize("path", [SAE_CONFIG, TRANSCODER_CONFIG])
def test_activation_is_the_only_randomized_a_treatment(path: Path):
    config = load_routed_ladder_config(path)
    relu, softplus = config.layouts

    for field in (
        "base_condition",
        "target",
        "bank_identity",
        "bank_count",
        "bank_width",
        "components_per_token",
        "topk_per_component",
        "total_features",
        "candidates_per_token",
        "total_l0",
        "trainable_parameters",
    ):
        assert getattr(relu, field) == getattr(softplus, field)
    assert (relu.encoder_activation, softplus.encoder_activation) == (
        "relu",
        "softplus",
    )

    for seed in (0, 1, 2):
        relu_plan = TrainingWorkerPlan.from_config(
            config,
            phase=PHASE_NAME,
            condition="A512ReLU",
            seed=seed,
        )
        softplus_plan = TrainingWorkerPlan.from_config(
            config,
            phase=PHASE_NAME,
            condition="A512Softplus",
            seed=seed,
        )
        assert relu_plan.randomization_key == softplus_plan.randomization_key == seed


@pytest.mark.parametrize("path", [SAE_CONFIG, TRANSCODER_CONFIG])
@pytest.mark.parametrize("condition", CONDITIONS)
def test_training_worker_plan_accepts_every_ladder_seed(
    path: Path,
    condition: str,
):
    config = load_routed_ladder_config(path)

    for seed in (0, 1, 2):
        plan = TrainingWorkerPlan.from_config(
            config,
            phase=PHASE_NAME,
            condition=condition,
            seed=seed,
        )
        assert plan.condition == condition
        assert plan.seed == seed
        assert plan.total_tokens == 10_000_000
        assert plan.planned_updates == 611
        assert plan.global_microbatch_tokens == 4_096
        assert plan.microbatches_per_update == 4
        assert plan.checkpoint_tokens[-1] == 10_000_000


@pytest.mark.parametrize(
    ("panel", "path"),
    [("sae", SAE_CONFIG), ("transcoder", TRANSCODER_CONFIG)],
)
def test_worker_loader_and_long_horizon_policy_recognize_ladder(
    panel: str,
    path: Path,
):
    loaded = _load_worker_config(path, panel)

    assert loaded.protocol_id == PROTOCOL_ID
    assert loaded.panel == panel
    assert (
        prefetch_runtime_limit_seconds(
            config_path=path,
            accelerator_type="v5litepod-16",
            phase=PHASE_NAME,
        )
        == 1_800
    )
    assert (
        direct_resident_runtime_limit_seconds(
            config_path=path,
            accelerator_type="v5litepod-16",
            phase=PHASE_NAME,
        )
        == 43_200
    )


def test_public_ladder_conditions_map_only_to_existing_a_kernel():
    assert _contribution_kernel_condition("A512ReLU") == "A"
    assert _contribution_kernel_condition("A512Softplus") == "A"


@pytest.mark.parametrize(
    ("panel", "path"),
    [("sae", SAE_CONFIG), ("transcoder", TRANSCODER_CONFIG)],
)
def test_h_is_read_only_existing_experiment5_reference(
    panel: str,
    path: Path,
):
    config = load_routed_ladder_config(path)
    reference = config.existing_h_reference

    assert "H" not in {layout.condition for layout in config.layouts}
    assert reference.condition == "H"
    assert reference.source_protocol_id == EXPERIMENT5_PROTOCOL_ID
    assert reference.source_config_sha256 == H_CONFIG_SHA256[panel]
    assert reference.training_policy == "reuse_existing_artifacts_only"
    assert reference.encoder_activation == "softplus"
    assert reference.objective == "normalized_full_ffn_sse"
    assert reference.checkpoint_tokens == (2_000_000, 5_000_000, 10_000_000)
    assert config.raw["evaluation"]["h_reference_training_allowed"] is False


@pytest.mark.parametrize(
    ("panel", "ladder_path"),
    [("sae", SAE_CONFIG), ("transcoder", TRANSCODER_CONFIG)],
)
def test_ladder_reuses_exact_experiment5_optimizer_and_artifact_hashes(
    panel: str,
    ladder_path: Path,
):
    ladder = yaml.safe_load(ladder_path.read_text(encoding="utf-8"))
    parent = yaml.safe_load(EXPERIMENT5_CONFIG[panel].read_text(encoding="utf-8"))

    assert ladder["optimizer"] == parent["optimizer"]
    for key in (
        "target_schema_version",
        "source_overlay_schema_version",
        "train_target_manifest_sha256",
        "eval_target_manifest_sha256",
        "train_source_overlay_manifest_sha256",
        "eval_source_overlay_manifest_sha256",
        "recomposition_sidecar_content_sha256",
        "recomposition_sidecar_file_sha256",
        "train_tokens",
        "eval_tokens",
    ):
        assert ladder["artifacts"][key] == parent["artifacts"][key]
    if panel == "transcoder":
        assert (
            ladder["artifacts"]["source_overlay_capture_contract_sha256"]
            == parent["artifacts"]["source_overlay_capture_contract_sha256"]
        )


@pytest.mark.parametrize(
    ("source", "path_parts", "replacement", "message"),
    [
        (
            SAE_CONFIG,
            ("layouts", 0, "encoder_activation"),
            "softplus",
            r"layouts\[0\].encoder_activation",
        ),
        (
            TRANSCODER_CONFIG,
            ("layouts", 0, "bank_width"),
            256,
            r"layouts\[0\].bank_width",
        ),
        (
            SAE_CONFIG,
            ("layouts", 1, "topk_per_component"),
            2,
            r"layouts\[1\].topk_per_component",
        ),
        (
            SAE_CONFIG,
            ("training", "seeds"),
            [0],
            "training.seeds",
        ),
        (
            TRANSCODER_CONFIG,
            ("existing_h_reference", "training_policy"),
            "train",
            "existing_h_reference.training_policy",
        ),
        (
            SAE_CONFIG,
            ("parent_protocol_ids", "experiment_4"),
            "changed",
            "parent_protocol_ids.experiment_4",
        ),
    ],
)
def test_scientific_and_lineage_drift_fail_closed(
    tmp_path: Path,
    source: Path,
    path_parts: tuple[str | int, ...],
    replacement: object,
    message: str,
):
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    selected = raw
    for part in path_parts[:-1]:
        selected = selected[part]
    selected[path_parts[-1]] = replacement
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(RoutedLadderConfigError, match=message):
        load_routed_ladder_config(changed)


def test_unknown_fields_cross_panel_loading_and_post_load_mutation_fail_closed(
    tmp_path: Path,
):
    with pytest.raises(RoutedLadderConfigError, match="expected a transcoder"):
        load_routed_ladder_config(SAE_CONFIG, expected_panel="transcoder")

    raw = yaml.safe_load(SAE_CONFIG.read_text(encoding="utf-8"))
    raw["training"]["silent_override"] = True
    changed = tmp_path / "extra.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(RoutedLadderConfigError, match="training fields differ"):
        load_routed_ladder_config(changed)

    config = load_routed_ladder_config(SAE_CONFIG)
    config.raw["layouts"][0]["bank_width"] = 1
    with pytest.raises(RoutedLadderConfigError, match="identity changed"):
        config.validate()


def test_protocol_and_operations_document_the_disjoint_ladder_extension():
    protocol = PROTOCOL_DOC.read_text(encoding="utf-8")
    operations = OPERATIONS_DOC.read_text(encoding="utf-8")

    assert PROTOCOL_ID in protocol
    assert "| A512ReLU |" in protocol
    assert "| A512Softplus |" in protocol
    assert "extension launches 12 new trajectories" in protocol
    assert "six H trajectories are read-only" in protocol
    assert "does not retroactively alter this frozen" in protocol
    assert "experiment5/extensions/routed-activation-ladder-v1/" in protocol
    assert "Nothing in the `ready-three-v1` plan" in operations
    assert "launch all 12 new A trajectories" in operations
