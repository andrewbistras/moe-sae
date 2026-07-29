from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from moe_sae.contribution_training_worker import (
    ContributionTrainingWorkerError,
    TrainingWorkerPlan,
    _contribution_kernel_condition,
    _load_worker_config,
    direct_resident_runtime_limit_seconds,
)
from moe_sae.experiment5_routed_ladder_v2_config import (
    A256_SAE_CONFIG_SHA256,
    A256_SAE_NEW_TRAINING_SEEDS,
    A256_SAE_REUSED_SEEDS,
    ANALYZED_CONDITIONS,
    CONDITIONS,
    PHASE_NAME,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    RoutedLadderV2ConfigError,
    load_routed_ladder_v2_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "sae": ROOT / "configs" / "gemma4_experiment5_routed_ladder_sae_v2.yaml",
    "transcoder": (
        ROOT / "configs" / "gemma4_experiment5_routed_ladder_transcoder_v2.yaml"
    ),
}
EXPECTED_CONFIG_SHA256 = {
    "sae": "729902a48f48859eb0ce1aac386e4a5b16dd3c32417dc361af9633a41e8349b2",
    "transcoder": "12062fbc4073bc93d72a55643ff3031d6166996e8454b35e557596e6361a9787",
}


@pytest.mark.parametrize(("panel", "path"), CONFIGS.items())
def test_checked_in_v2_contract_is_exact_and_worker_compatible(
    panel: str,
    path: Path,
):
    config = load_routed_ladder_v2_config(path, expected_panel=panel)

    assert config.config_hash == EXPECTED_CONFIG_SHA256[panel]
    assert config.protocol_id == PROTOCOL_ID
    assert config.raw["schema_version"] == SCHEMA_VERSION
    assert config.phase(PHASE_NAME).seeds == (0, 1, 2)
    assert config.phase(PHASE_NAME).planned_updates == 611
    assert config.runtime.direct_runtime_max_seconds == 43_200
    assert _load_worker_config(path, panel).config_hash == config.config_hash
    assert (
        direct_resident_runtime_limit_seconds(
            config_path=path,
            accelerator_type="v5litepod-16",
            phase=PHASE_NAME,
        )
        == 43_200
    )


@pytest.mark.parametrize("path", CONFIGS.values())
def test_v2_ladder_contains_the_exact_three_routed_geometries(path: Path):
    config = load_routed_ladder_v2_config(path)

    assert tuple(layout.condition for layout in config.layouts) == CONDITIONS
    assert [
        (
            layout.bank_width,
            layout.topk_per_component,
            layout.total_features,
            layout.candidates_per_token,
            layout.total_l0,
            layout.trainable_parameters,
            layout.encoder_activation,
            layout.training_seeds,
        )
        for layout in config.layouts
    ] == [
        (
            256,
            2,
            32_768,
            2_048,
            16,
            184_582_144,
            "relu",
            (1, 2) if config.panel == "sae" else (0, 1, 2),
        ),
        (512, 4, 65_536, 4_096, 32, 369_164_288, "relu", (0, 1, 2)),
        (512, 4, 65_536, 4_096, 32, 369_164_288, "softplus", (0, 1, 2)),
    ]
    assert all(
        layout.base_condition == layout.worker_condition == "A"
        and layout.bank_count == 128
        and layout.components_per_token == 8
        for layout in config.layouts
    )
    assert tuple(config.raw["evaluation"]["ladder_conditions"]) == (
        ANALYZED_CONDITIONS
    )


def test_sae_a256_historical_seed_is_one_reference_not_three_seed_evidence():
    config = load_routed_ladder_v2_config(CONFIGS["sae"])
    reference = config.existing_a256_reference

    assert reference is not None
    assert reference.source_condition == "A"
    assert reference.source_config_sha256 == A256_SAE_CONFIG_SHA256
    assert reference.reused_seeds == A256_SAE_REUSED_SEEDS == (0,)
    assert (
        reference.new_training_seeds
        == A256_SAE_NEW_TRAINING_SEEDS
        == (1, 2)
    )
    assert config.training_seeds_for_condition("A256ReLU") == (1, 2)
    assert config.raw["evaluation"]["a256_reference_training_allowed"] is False

    with pytest.raises(
        ContributionTrainingWorkerError,
        match="read-only historical reference",
    ):
        TrainingWorkerPlan.from_config(
            config,
            phase=PHASE_NAME,
            condition="A256ReLU",
            seed=0,
        )
    for seed in (1, 2):
        plan = TrainingWorkerPlan.from_config(
            config,
            phase=PHASE_NAME,
            condition="A256ReLU",
            seed=seed,
        )
        assert (plan.condition, plan.seed) == ("A256ReLU", seed)


def test_transcoder_a256_relu_trains_all_seeds_and_reuses_no_softplus_reference():
    config = load_routed_ladder_v2_config(CONFIGS["transcoder"])

    assert config.existing_a256_reference is None
    assert config.raw["existing_a256_reference"] is None
    assert config.training_seeds_for_condition("A256ReLU") == (0, 1, 2)
    for seed in (0, 1, 2):
        plan = TrainingWorkerPlan.from_config(
            config,
            phase=PHASE_NAME,
            condition="A256ReLU",
            seed=seed,
        )
        assert (plan.condition, plan.seed) == ("A256ReLU", seed)


@pytest.mark.parametrize("path", CONFIGS.values())
def test_all_new_v2_conditions_map_to_the_unchanged_a_kernel(path: Path):
    config = load_routed_ladder_v2_config(path)

    for condition in CONDITIONS:
        assert _contribution_kernel_condition(condition) == "A"
    for condition in ("A512ReLU", "A512Softplus"):
        for seed in (0, 1, 2):
            plan = TrainingWorkerPlan.from_config(
                config,
                phase=PHASE_NAME,
                condition=condition,
                seed=seed,
            )
            assert plan.randomization_key == seed


@pytest.mark.parametrize(
    ("source", "path_parts", "replacement", "message"),
    [
        (
            CONFIGS["sae"],
            ("existing_a256_reference", "reused_seeds"),
            [0, 1, 2],
            "existing_a256_reference.reused_seeds",
        ),
        (
            CONFIGS["transcoder"],
            ("existing_a256_reference",),
            {
                "condition": "A256ReLU",
                "source_condition": "A",
            },
            "existing_a256_reference",
        ),
        (
            CONFIGS["sae"],
            ("layouts", 0, "training_seeds"),
            [0, 1, 2],
            r"layouts\[0\].training_seeds",
        ),
        (
            CONFIGS["transcoder"],
            ("layouts", 0, "bank_width"),
            512,
            r"layouts\[0\].bank_width",
        ),
        (
            CONFIGS["sae"],
            ("evaluation", "a256_reference_training_allowed"),
            True,
            "evaluation.a256_reference_training_allowed",
        ),
    ],
)
def test_v2_reference_and_geometry_drift_fail_closed(
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

    with pytest.raises(RoutedLadderV2ConfigError, match=message):
        load_routed_ladder_v2_config(changed)
