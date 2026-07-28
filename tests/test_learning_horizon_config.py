from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from moe_sae.contribution_config import load_contribution_config
from moe_sae.contribution_training_worker import TrainingWorkerPlan
from moe_sae.contribution_transcoder_config import (
    load_contribution_transcoder_config,
)
from moe_sae.learning_horizon_config import (
    EVAL_SOURCE_OVERLAY_MANIFEST_SHA256,
    EVAL_TARGET_MANIFEST_SHA256,
    PHASE_NAME,
    LearningHorizonConfigError,
    finalize_learning_horizon_config,
    load_learning_horizon_config,
)

ROOT = Path(__file__).resolve().parents[1]
SAE_TEMPLATE = ROOT / "configs" / "gemma4_contribution_horizon_v1.yaml"
TRANSCODER_TEMPLATE = ROOT / "configs" / "gemma4_transcoder_horizon_v1.yaml"
TARGET_SHA = "1" * 64
SOURCE_SHA = "2" * 64
SOURCE_CAPTURE_CONTRACT_SHA = "4" * 64


@pytest.mark.parametrize(
    ("path", "loader"),
    [
        (SAE_TEMPLATE, load_contribution_config),
        (TRANSCODER_TEMPLATE, load_contribution_transcoder_config),
    ],
)
def test_checked_in_templates_are_not_executable(path: Path, loader):
    with pytest.raises(
        LearningHorizonConfigError,
        match="template is not executable",
    ):
        loader(path)


@pytest.mark.parametrize(
    ("template", "model"),
    [
        (SAE_TEMPLATE, "sae"),
        (TRANSCODER_TEMPLATE, "transcoder"),
    ],
)
def test_template_freezes_exact_10m_pilot_horizon(template: Path, model: str):
    config = load_learning_horizon_config(
        template,
        expected_scientific_model=model,
        require_finalized=False,
    )
    phase = config.phase(PHASE_NAME)

    assert phase.seeds == (0,)
    assert phase.layout_set == "pilot"
    assert phase.initialization == "from_scratch"
    assert phase.train_tokens == 10_000_000
    assert phase.eval_tokens == 250_000
    assert phase.planned_updates == 611
    assert phase.warmup_updates == 61
    assert phase.tail_mask_tokens == 10_624
    assert phase.restart_checkpoint_tokens == tuple(
        range(1_000_000, 10_000_001, 1_000_000)
    )
    assert phase.evaluation_checkpoint_tokens == (
        2_000_000,
        5_000_000,
        10_000_000,
    )
    assert phase.escalation_gates == ()
    assert phase.blind_relative_quality_until_health is False

    assert [layout.condition for layout in config.layouts] == ["A", "R", "B", "F"]
    assert [
        (layout.bank_count, layout.bank_width, layout.topk_per_component)
        for layout in config.layouts
    ] == [
        (128, 256, 2),
        (128, 256, 2),
        (8, 256, 2),
        (8, 4_096, 2),
    ]
    assert config.parents["eval_target_manifest_sha256"] == (
        EVAL_TARGET_MANIFEST_SHA256
    )
    assert config.runtime.accelerator_min_useful_mfu == 0.10


def test_finalized_configs_dispatch_through_existing_worker_loaders(tmp_path: Path):
    sae_path = tmp_path / "sae.yaml"
    transcoder_path = tmp_path / "transcoder.yaml"
    sae = finalize_learning_horizon_config(
        SAE_TEMPLATE,
        sae_path,
        train_target_manifest_sha256=TARGET_SHA,
    )
    transcoder = finalize_learning_horizon_config(
        TRANSCODER_TEMPLATE,
        transcoder_path,
        train_target_manifest_sha256=TARGET_SHA,
        train_source_overlay_manifest_sha256=SOURCE_SHA,
        source_overlay_capture_contract_sha256=SOURCE_CAPTURE_CONTRACT_SHA,
    )

    assert load_contribution_config(sae_path).config_hash == sae.config_hash
    assert (
        load_contribution_transcoder_config(transcoder_path).config_hash
        == transcoder.config_hash
    )
    assert transcoder.parents == {
        "target_schema_version": "gemma4-expert-contribution-v2",
        "train_target_manifest_sha256": TARGET_SHA,
        "eval_target_manifest_sha256": EVAL_TARGET_MANIFEST_SHA256,
        "train_source_overlay_manifest_sha256": SOURCE_SHA,
        "eval_source_overlay_manifest_sha256": (
            EVAL_SOURCE_OVERLAY_MANIFEST_SHA256
        ),
        "source_overlay_capture_contract_sha256": (
            SOURCE_CAPTURE_CONTRACT_SHA
        ),
        "train_tokens": 10_000_000,
        "eval_tokens": 250_000,
        "recomposition_sidecar_schema_version": (
            "gemma4-contribution-recomposition-v1"
        ),
    }

    for config in (sae, transcoder):
        for condition in ("A", "R", "B", "F"):
            plan = TrainingWorkerPlan.from_config(
                config,
                phase=PHASE_NAME,
                condition=condition,
                seed=0,
            )
            assert plan.total_tokens == 10_000_000
            assert plan.planned_updates == 611
            assert plan.warmup_updates == 61
            assert plan.checkpoint_tokens == tuple(
                range(1_000_000, 10_000_001, 1_000_000)
            )


def test_finalization_is_idempotent_but_never_overwrites(tmp_path: Path):
    destination = tmp_path / "final.yaml"
    first = finalize_learning_horizon_config(
        SAE_TEMPLATE,
        destination,
        train_target_manifest_sha256=TARGET_SHA,
    )
    replay = finalize_learning_horizon_config(
        SAE_TEMPLATE,
        destination,
        train_target_manifest_sha256=TARGET_SHA,
    )
    assert replay.config_hash == first.config_hash

    with pytest.raises(LearningHorizonConfigError, match="different content"):
        finalize_learning_horizon_config(
            SAE_TEMPLATE,
            destination,
            train_target_manifest_sha256="3" * 64,
        )


def test_cross_model_loading_and_incomplete_transcoder_binding_fail(tmp_path: Path):
    with pytest.raises(LearningHorizonConfigError, match="expected a sae"):
        load_learning_horizon_config(
            TRANSCODER_TEMPLATE,
            expected_scientific_model="sae",
            require_finalized=False,
        )
    with pytest.raises(
        LearningHorizonConfigError,
        match="train_source_overlay_manifest_sha256",
    ):
        finalize_learning_horizon_config(
            TRANSCODER_TEMPLATE,
            tmp_path / "bad.yaml",
            train_target_manifest_sha256=TARGET_SHA,
        )


@pytest.mark.parametrize(
    ("path", "field", "value", "message"),
    [
        (
            ("training",),
            "planned_updates",
            610,
            "training.planned_updates",
        ),
        (
            ("training",),
            "tail_mask_tokens",
            0,
            "training.tail_mask_tokens",
        ),
        (
            ("evaluation",),
            "relative_quality_gates",
            ["A-F-positive"],
            "evaluation.relative_quality_gates",
        ),
        (
            ("layouts", 0),
            "bank_width",
            1_024,
            r"layouts\[0\].bank_width",
        ),
    ],
)
def test_schedule_layout_and_quality_gate_drift_fail_closed(
    tmp_path: Path,
    path: tuple[str | int, ...],
    field: str,
    value,
    message: str,
):
    raw = yaml.safe_load(SAE_TEMPLATE.read_text(encoding="utf-8"))
    selected = raw
    for component in path:
        selected = selected[component]
    selected[field] = value
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(LearningHorizonConfigError, match=message):
        load_learning_horizon_config(changed, require_finalized=False)
