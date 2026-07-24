from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from moe_sae.contribution_config import (
    C4_REVISION,
    MAXTEXT_BASE_COMMIT,
    MAXTEXT_FORK_COMMIT,
    MAXTEXT_SELECTED_OUTPUTS_PATCH_SHA256,
    MODEL_REVISION,
    PHASE_NAMES,
    ContributionConfigError,
    ContributionLayout,
    ContributionModelContract,
    ContributionOptimizer,
    ContributionProvenance,
    ContributionRuntime,
    Experiment2Config,
    load_contribution_config,
    save_contribution_config,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN_CONFIG = ROOT / "configs" / "gemma4_contribution_v1.yaml"
DEFAULT_CONFIG_SHA256 = (
    "ca41145b5cc58be6f2a2c4f46efeb80c5f9621934d2bd7c31432719dd7190c4a"
)


def _gate_results(config: Experiment2Config, *phase_names: str):
    return {
        phase_name: {
            gate: True
            for gate in config.phase(phase_name).escalation_gates
        }
        for phase_name in phase_names
    }


def test_checked_in_config_is_the_exact_unqualified_default():
    config = load_contribution_config(CHECKED_IN_CONFIG)

    assert config == Experiment2Config()
    assert config.config_hash == DEFAULT_CONFIG_SHA256
    assert config.provenance.model_revision == MODEL_REVISION
    assert config.provenance.dataset_revision == C4_REVISION
    assert config.provenance.maxtext_base_commit == MAXTEXT_BASE_COMMIT
    assert (
        config.provenance.maxtext_selected_outputs_patch_sha256
        == MAXTEXT_SELECTED_OUTPUTS_PATCH_SHA256
    )
    assert config.provenance.maxtext_fork_commit == MAXTEXT_FORK_COMMIT
    assert config.provenance.authoritative_capture_ready


def test_config_identity_matches_the_checked_in_maxtext_patch_manifest():
    manifest = json.loads(
        (
            ROOT / "patches" / "maxtext" / "selected_outputs_manifest.json"
        ).read_text(encoding="utf-8")
    )
    config = Experiment2Config()
    assert manifest["base_commit"] == config.provenance.maxtext_base_commit
    assert (
        manifest["patch_sha256"]
        == config.provenance.maxtext_selected_outputs_patch_sha256
    )
    assert manifest["fork_commit"] == config.provenance.maxtext_fork_commit
    assert manifest["qualified_path"] == {
        "decoder_block": "gemma4",
        "sparse_matmul": True,
        "use_ring_of_experts": False,
        "ragged_buffer_factor_max": 0.0,
        "tensor_parallelism": 1,
        "hash_routing": False,
        "serve_quantization": False,
        "fused_vllm": False,
    }
    assert config.model.sparse_matmul
    assert not config.model.use_ring_of_experts
    assert config.model.ragged_buffer_factor <= 0.0
    assert config.model.tensor_parallelism == 1
    assert not config.model.hash_routing
    assert not config.model.serve_quantization
    assert config.model.attention != "vllm_rpa"


def test_phase_plan_freezes_scale_updates_warmups_and_tail_masks():
    config = Experiment2Config()
    assert tuple(phase.name for phase in config.phases) == PHASE_NAMES

    expected = (
        (100_000, 25_000, None, None, None),
        (2_000_000, 250_000, 123, 12, 15_232),
        (10_000_000, 1_000_000, 611, 61, 10_624),
        (50_000_000, 2_000_000, 3_052, 250, 3_968),
    )
    assert tuple(
        (
            phase.train_tokens,
            phase.eval_tokens,
            phase.planned_updates,
            phase.warmup_updates,
            phase.tail_mask_tokens,
        )
        for phase in config.phases
    ) == expected
    assert config.phase(1).layout_set == "none"
    assert config.phase(2).seeds == (0,)
    assert config.phase(3).seeds == (0,)
    assert config.phase(4).seeds == (1, 2, 3)
    assert config.phase(2).require_f_compile_smoke
    assert config.phase(3).shared_q_resume_phase == "phase2_mechanism_pilot"
    assert config.phase(4).shared_q_resume_phase is None
    assert all(
        phase.blind_relative_quality_until_health
        for phase in config.phases[1:]
    )


def test_pilot_full_and_shared_layouts_are_objective_matched():
    config = Experiment2Config()
    pilot = {layout.condition: layout for layout in config.pilot_layouts}
    full = {layout.condition: layout for layout in config.full_layouts}

    assert tuple(pilot) == ("A", "R", "B", "F")
    assert pilot["A"].total_features == 32_768
    assert pilot["R"].total_features == 32_768
    assert pilot["B"].total_features == 2_048
    assert pilot["F"].total_features == 32_768
    assert pilot["A"].candidates_per_token == 2_048
    assert pilot["B"].candidates_per_token == 2_048
    assert pilot["F"].candidates_per_token == 32_768
    assert {layout.total_l0 for layout in pilot.values()} == {16}

    assert full["A"].total_features == 131_072
    assert full["R"].total_features == 131_072
    assert full["B"].total_features == 8_192
    assert full["F"].total_features == 131_072
    assert full["A"].candidates_per_token == 8_192
    assert full["B"].candidates_per_token == 8_192
    assert full["F"].candidates_per_token == 131_072
    assert {layout.total_l0 for layout in full.values()} == {64}

    assert config.shared_q.total_features == 8_192
    assert config.shared_q.candidates_per_token == 8_192
    assert config.shared_q.total_l0 == 64
    assert config.shared_q.target == "normalized_shared_branch"
    assert config.layouts_for_phase(1) == ()
    assert tuple(
        layout.condition for layout in config.layouts_for_phase(2)
    ) == ("A", "R", "B", "F", "Q")
    assert tuple(
        layout.condition for layout in config.layouts_for_phase(4)
    ) == ("A", "R", "B", "F", "Q")


def test_runtime_does_not_reintroduce_experiment1_capture_defaults():
    runtime = Experiment2Config().runtime
    assert runtime.capture_schema_version == "gemma4-expert-contribution-v2"
    assert runtime.max_shard_tokens == 16_384
    assert runtime.accelerator_mfu_policy == (
        "hard_gate_exact_graph_hardware_config"
    )
    assert runtime.accelerator_min_useful_mfu == 0.10
    assert runtime.accelerator_ungated_max_seconds == 600
    assert runtime.accelerator_qualification_max_seconds == 600
    assert runtime.phase1_min_persisted_goodput_fraction == 0.85
    assert runtime.phase1_min_auxiliary_throughput_fraction == 0.90
    assert runtime.router_order == "weight_desc_expert_id_asc"
    assert runtime.contribution_loss == "normalized_router_weighted_sse"
    assert runtime.router_weight_math_dtype == "float32"
    assert runtime.topk_requires_strictly_positive
    assert runtime.capacity_overflow_policy == "hard_error"
    assert runtime.phase2_min_g2_percent == 10.0
    assert runtime.phase3_min_g2_percent == 10.0
    assert runtime.phase3_max_c2_percent == 50.0
    assert runtime.bootstrap_replicates == 20_000


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: ContributionProvenance(model_revision="a" * 40),
            "model_revision is pinned",
        ),
        (
            lambda: ContributionProvenance(
                maxtext_base_commit="b" * 40
            ),
            "maxtext_base_commit is pinned",
        ),
        (
            lambda: ContributionProvenance(
                maxtext_selected_outputs_patch_sha256="c" * 64
            ),
            "patch_sha256 is pinned",
        ),
        (
            lambda: ContributionModelContract(hidden_size=1_024),
            "model.hidden_size is pinned",
        ),
        (
            lambda: ContributionModelContract(tensor_parallelism=2),
            "model.tensor_parallelism is pinned",
        ),
        (
            lambda: ContributionRuntime(max_shard_tokens=131_072),
            "runtime.max_shard_tokens is pinned",
        ),
        (
            lambda: ContributionRuntime(accelerator_min_useful_mfu=0.09),
            "runtime.accelerator_min_useful_mfu is pinned",
        ),
        (
            lambda: ContributionOptimizer(weight_decay=0.1),
            "optimizer.weight_decay is pinned",
        ),
    ],
)
def test_immutable_source_runtime_and_optimizer_pins_reject_drift(
    factory,
    message,
):
    with pytest.raises(ContributionConfigError, match=message):
        factory()


@pytest.mark.parametrize("fork", ["abc", "A" * 40, "0" * 39, 7])
def test_malformed_fork_commit_is_rejected(fork):
    with pytest.raises(ContributionConfigError, match="40-character"):
        ContributionProvenance(maxtext_fork_commit=fork)


def test_unpatched_base_cannot_masquerade_as_the_fork():
    with pytest.raises(ContributionConfigError, match="unpatched base"):
        ContributionProvenance(maxtext_fork_commit=MAXTEXT_BASE_COMMIT)


def test_authoritative_tpu_capture_requires_the_observed_pinned_fork_commit():
    config = Experiment2Config()
    with pytest.raises(ContributionConfigError, match="observed MaxText"):
        config.validate_authoritative_tpu_capture(1)
    with pytest.raises(ContributionConfigError, match="observed MaxText"):
        config.validate_authoritative_tpu_capture(
            1,
            observed_maxtext_fork_commit="2" * 40,
        )
    assert (
        config.validate_authoritative_tpu_capture(
            1,
            observed_maxtext_fork_commit=MAXTEXT_FORK_COMMIT,
        ).name
        == PHASE_NAMES[0]
    )
    assert (
        config.validate_authoritative_tpu_capture(
            4,
            observed_maxtext_fork_commit=MAXTEXT_FORK_COMMIT,
        ).name
        == PHASE_NAMES[3]
    )
    assert config.with_maxtext_fork_commit(MAXTEXT_FORK_COMMIT) is config
    with pytest.raises(ContributionConfigError, match="cannot drift"):
        config.with_maxtext_fork_commit("1" * 40)


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ((), "typo"),
        (("provenance",), "typo"),
        (("model",), "typo"),
        (("runtime",), "typo"),
        (("optimizer",), "typo"),
        (("pilot_layouts", 0), "typo"),
        (("phases", 0), "typo"),
    ],
)
def test_unknown_keys_are_rejected_recursively(path, key):
    raw = Experiment2Config().as_dict()
    selected = raw
    for component in path:
        selected = selected[component]
    selected[key] = "not allowed"

    with pytest.raises(ContributionConfigError, match="unknown keys"):
        Experiment2Config.from_mapping(raw)


def test_missing_keys_are_rejected_recursively():
    raw = Experiment2Config().as_dict()
    del raw["runtime"]["max_shard_tokens"]
    with pytest.raises(ContributionConfigError, match="is missing keys"):
        Experiment2Config.from_mapping(raw)

    raw = Experiment2Config().as_dict()
    del raw["phases"][2]["warmup_updates"]
    with pytest.raises(ContributionConfigError, match="is missing keys"):
        Experiment2Config.from_mapping(raw)


def test_layout_identity_and_phase_specific_layout_drift_fail_closed():
    with pytest.raises(ContributionConfigError, match="identity/shape"):
        ContributionLayout(
            condition="A",
            target="raw_selected_expert_output",
            bank_identity="router_rank",
            bank_count=128,
            bank_width=256,
            components_per_token=8,
            topk_per_component=2,
        )

    config = Experiment2Config()
    changed = dataclasses.replace(config.pilot_layouts[0], bank_width=512)
    with pytest.raises(
        ContributionConfigError,
        match="exact A0/R0/B0/F0 matrix",
    ):
        dataclasses.replace(
            config,
            pilot_layouts=(changed, *config.pilot_layouts[1:]),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda phases: (
                dataclasses.replace(phases[0], ordinal=2),
                *phases[1:],
            ),
            "contiguous ordinals",
        ),
        (
            lambda phases: (
                phases[0],
                dataclasses.replace(phases[1], extends_phase=None),
                *phases[2:],
            ),
            "must extend",
        ),
        (
            lambda phases: (
                phases[0],
                dataclasses.replace(
                    phases[1],
                    train_tokens=100_000,
                    checkpoint_tokens=(100_000,),
                ),
                *phases[2:],
            ),
            "extend monotonically",
        ),
        (
            lambda phases: (
                phases[0],
                dataclasses.replace(phases[1], planned_updates=124),
                *phases[2:],
            ),
            "planned_updates must be 123",
        ),
        (
            lambda phases: (
                phases[0],
                dataclasses.replace(phases[1], tail_mask_tokens=0),
                *phases[2:],
            ),
            "tail_mask_tokens must be 15232",
        ),
        (
            lambda phases: (
                dataclasses.replace(phases[0], warmup_updates=1),
                *phases[1:],
            ),
            "capture-only Phase 1",
        ),
        (
            lambda phases: (
                phases[0],
                dataclasses.replace(
                    phases[1],
                    shared_q_resume_phase="phase4_confirmatory",
                ),
                *phases[2:],
            ),
            "shared Q can resume only from a predecessor",
        ),
    ],
)
def test_phase_plan_must_remain_monotonic_and_schedule_exact(mutate, message):
    config = Experiment2Config()
    with pytest.raises(ContributionConfigError, match=message):
        dataclasses.replace(config, phases=mutate(config.phases))


def test_escalation_requires_every_exact_predecessor_gate_to_pass():
    config = Experiment2Config()
    assert config.validate_escalation(1, {}).ordinal == 1

    with pytest.raises(ContributionConfigError, match="missing gate results"):
        config.validate_escalation(2, {})

    phase1 = _gate_results(config, "phase1_smoke")
    assert config.validate_escalation(2, phase1).ordinal == 2

    failed = copy.deepcopy(phase1)
    failed["phase1_smoke"][config.phase(1).escalation_gates[0]] = False
    with pytest.raises(ContributionConfigError, match="has failed gates"):
        config.validate_escalation(2, failed)

    unknown = copy.deepcopy(phase1)
    unknown["phase1_smoke"]["made_up_gate"] = True
    with pytest.raises(ContributionConfigError, match="unknown gate results"):
        config.validate_escalation(2, unknown)

    predecessors = _gate_results(
        config,
        "phase1_smoke",
        "phase2_mechanism_pilot",
        "phase3_full_width_pilot",
    )
    assert config.validate_escalation(4, predecessors).ordinal == 4


def test_config_hash_is_canonical_and_roundtrips_json_and_yaml(tmp_path):
    config = Experiment2Config()
    raw = config.as_dict()
    reversed_mapping = dict(reversed(list(raw.items())))
    assert Experiment2Config.from_mapping(reversed_mapping).config_hash == (
        config.config_hash
    )

    for suffix in (".json", ".yaml"):
        path = tmp_path / f"experiment2{suffix}"
        save_contribution_config(config, path)
        restored = load_contribution_config(path)
        assert restored == config
        assert restored.config_hash == config.config_hash

    json_path = tmp_path / "canonical.json"
    json_path.write_text(
        json.dumps(config.as_dict(), sort_keys=False),
        encoding="utf-8",
    )
    assert load_contribution_config(json_path).config_hash == config.config_hash


def test_phase_selector_rejects_ambiguous_or_unknown_values():
    config = Experiment2Config()
    assert config.phase("phase3_full_width_pilot").ordinal == 3
    with pytest.raises(ContributionConfigError, match="cannot be boolean"):
        config.phase(True)
    with pytest.raises(ContributionConfigError, match="unknown phase ordinal"):
        config.phase(5)
    with pytest.raises(ContributionConfigError, match="unknown phase name"):
        config.phase("phase5")


def test_config_is_deeply_immutable():
    config = Experiment2Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.runtime.max_shard_tokens = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.phases[1].warmup_updates = 250
