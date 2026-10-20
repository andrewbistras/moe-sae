from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from moe_sae.component_closure_config import (
    CHECKPOINT_TOKENS,
    PHASE_NAME,
    SEEDS,
    ComponentClosureConfigError,
    component_closure_config_from_mapping,
    load_component_closure_config,
)
from moe_sae.contribution_training_worker import (
    ContributionTrainingWorkerError,
    TrainingWorkerPlan,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("panel", ["sae", "transcoder"])
def test_closure_config_is_exact_and_worker_compatible(panel: str) -> None:
    config = load_component_closure_config(
        ROOT / "configs" / f"gemma4_component_closure_{panel}_v1.yaml",
        expected_panel=panel,  # type: ignore[arg-type]
    )
    assert [row.condition for row in config.layouts] == ["A", "R", "B", "F"]
    assert [row.total_features for row in config.layouts] == [
        65_536,
        65_536,
        4_096,
        65_536,
    ]
    assert [row.candidates_per_token for row in config.layouts] == [
        4_096,
        4_096,
        4_096,
        65_536,
    ]
    assert all(row.selected_slots_per_token == 32 for row in config.layouts)
    plan = TrainingWorkerPlan.from_config(
        config,
        phase=PHASE_NAME,
        condition="F",
        seed=2,
    )
    assert plan.total_tokens == 10_000_000
    assert plan.checkpoint_tokens == tuple(range(1_000_000, 10_000_001, 1_000_000))
    assert config.closure_phase.evaluation_checkpoint_tokens == CHECKPOINT_TOKENS
    assert config.training_seeds_for_condition("R") == SEEDS
    with pytest.raises(ContributionTrainingWorkerError):
        TrainingWorkerPlan.from_config(
            config,
            phase=PHASE_NAME,
            condition="A",
            seed=0,
        )


def test_closure_config_rejects_geometry_drift() -> None:
    path = ROOT / "configs/gemma4_component_closure_sae_v1.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["layouts"][3]["bank_width"] = 4_096
    with pytest.raises(ComponentClosureConfigError):
        component_closure_config_from_mapping(raw)
