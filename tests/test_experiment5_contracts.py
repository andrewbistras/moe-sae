from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SAE_CONFIG = ROOT / "configs" / "gemma4_hybrid_sae_v1.yaml"
TRANSCODER_CONFIG = ROOT / "configs" / "gemma4_hybrid_transcoder_v1.yaml"
RESULT_SCHEMA = ROOT / "schemas" / "gemma4_hybrid_result_v1.schema.json"
REPORT_SCHEMA = ROOT / "schemas" / "gemma4_hybrid_report_v1.schema.json"
PAIRED_SCHEMA = ROOT / "schemas" / "gemma4_hybrid_paired_scores_v1.schema.json"
BUDGET_SCHEMA = ROOT / "schemas" / "gemma4_hybrid_budget_ledger_v1.schema.json"
PARTIAL_SCHEMA = (
    ROOT / "schemas" / "gemma4_hybrid_evaluation_partial_v1.schema.json"
)
PROTOCOL = ROOT / "docs" / "EXPERIMENT_5.md"

PROTOCOL_ID = "gemma4-routed-shared-hybrid-v1-20260728"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_ARTIFACTS = {
    "train_target_manifest_sha256": (
        "f29aaf45d9ff93be08cc3d175cdcea9fa4aecc37fc0e256d09720e88e1ae838c"
    ),
    "eval_target_manifest_sha256": (
        "31454e5ce8693521c41a2a95f2ef88d40c5d26c726a0efac5c26d882ed341f5e"
    ),
    "recomposition_sidecar_content_sha256": (
        "37c7b8c012f8fc0c1f63210986c0b3071a261261c88ee4bc1f502220ce3b8526"
    ),
    "recomposition_sidecar_file_sha256": (
        "6486288677e81665b21815879729c424c6acb8c7d0e45be264b961a9d652fe3d"
    ),
}


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("path", "panel", "activation"),
    [
        (SAE_CONFIG, "sae", "relu"),
        (TRANSCODER_CONFIG, "transcoder", "softplus"),
    ],
)
def test_experiment5_configs_freeze_identity_schedule_and_budget(
    path: Path,
    panel: str,
    activation: str,
):
    raw_text = path.read_text(encoding="utf-8")
    assert "REQUIRED_SEALED_" not in raw_text
    config = _load_yaml(path)

    assert config["schema_version"] == (
        "gemma4-routed-shared-hybrid-run-config-v1"
    )
    assert config["protocol_id"] == PROTOCOL_ID
    assert config["panel"] == panel
    assert config["artifacts"]["new_harvest_allowed"] is False
    for name, expected in EXPECTED_ARTIFACTS.items():
        assert config["artifacts"][name] == expected
        assert SHA256_RE.fullmatch(expected)

    training = config["training"]
    assert training["seeds"] == [0, 1, 2]
    assert training["encoder_activation"] == activation
    assert training["train_tokens"] == 10_000_000
    assert training["eval_tokens"] == 250_000
    assert training["effective_batch_tokens"] == 16_384
    assert training["microbatch_tokens"] == 4_096
    assert training["planned_updates"] == 611
    assert training["warmup_updates"] == 61
    assert training["tail_mask_tokens"] == 10_624
    assert training["objective"] == "normalized_full_ffn_sse"
    assert training["objective_target"] == (
        "exact_full_ffn_output_before_residual"
    )

    assert config["evaluation"]["checkpoint_tokens"] == [
        2_000_000,
        5_000_000,
        10_000_000,
    ]
    assert config["evaluation"]["primary_contrasts"] == [
        "H-R",
        "H-B",
        "H-F",
        "F-B",
    ]
    assert config["evaluation"]["fixed_quality_reference"] == {
        "condition": "F",
        "checkpoint_tokens": 10_000_000,
        "candidate_condition": "H",
        "interpolation_allowed": False,
    }

    assert config["execution"]["accelerator_min_useful_mfu"] == 0.10
    assert config["execution"]["accelerator_qualification_max_seconds"] == 600
    assert config["execution"]["new_bulk_capture_allowed"] is False
    assert config["execution"]["cross_region_bulk_copy_allowed"] is False
    assert config["budget"] == {
        "currency": "USD",
        "incremental_cash_hard_limit": 50.0,
        "maximum_launch_reservation": 45.0,
        "contingency": 5.0,
        "trc_tpu_time_is_quota_funded": True,
        "stop_before_projected_exceedance": True,
        "protocol_reduction_on_budget_pressure_allowed": False,
    }


@pytest.mark.parametrize("path", [SAE_CONFIG, TRANSCODER_CONFIG])
def test_experiment5_layout_matrix_is_exact(path: Path):
    config = _load_yaml(path)
    layouts = config["layouts"]
    assert [layout["condition"] for layout in layouts] == ["H", "R", "B", "F"]
    by_condition = {layout["condition"]: layout for layout in layouts}

    for condition in ("H", "R"):
        layout = by_condition[condition]
        routed_features = (
            layout["routed_bank_count"] * layout["routed_bank_width"]
        )
        shared_features = (
            layout["shared_bank_count"] * layout["shared_bank_width"]
        )
        candidates = (
            layout["routed_visible_banks_per_token"]
            * layout["routed_bank_width"]
            + layout["shared_bank_width"]
        )
        l0 = (
            layout["routed_visible_banks_per_token"]
            * layout["routed_topk_per_component"]
            + layout["shared_topk_per_token"]
        )
        assert routed_features == 65_536
        assert shared_features == 4_096
        assert layout["total_features"] == routed_features + shared_features
        assert layout["total_features"] == 69_632
        assert layout["candidates_per_token"] == candidates == 8_192
        assert layout["exact_total_l0"] == l0 == 64
        assert layout["trainable_parameters"] == (
            layout["total_features"] * (2 * 2_816 + 1)
        )
        assert layout["trainable_parameters"] == 392_237_056

    assert by_condition["H"]["routed_bank_identity"] == "true_expert_id"
    assert by_condition["R"]["routed_bank_identity"] == (
        "per_sequence_randomized_expert_id"
    )

    for condition, width in (("B", 8_192), ("F", 69_632)):
        layout = by_condition[condition]
        assert layout["architecture"] == "global"
        assert layout["bank_count"] == 1
        assert layout["bank_width"] == width
        assert layout["total_features"] == width
        assert layout["candidates_per_token"] == width
        assert layout["exact_total_l0"] == 64
        assert layout["trainable_parameters"] == width * (2 * 2_816 + 1)

    assert by_condition["H"]["candidates_per_token"] == (
        by_condition["B"]["candidates_per_token"]
    )
    assert by_condition["H"]["total_features"] == (
        by_condition["F"]["total_features"]
    )
    assert by_condition["H"]["trainable_parameters"] == (
        by_condition["F"]["trainable_parameters"]
    )
    assert (
        by_condition["H"]["candidates_per_token"]
        / by_condition["H"]["total_features"]
        == pytest.approx(0.11764705882352941)
    )


def test_transcoder_uses_one_common_source_without_target_leakage():
    config = _load_yaml(TRANSCODER_CONFIG)
    assert config["training"]["common_source"] == (
        "block15_post_attention_routed_input"
    )
    for layout in config["layouts"]:
        if layout["condition"] in {"H", "R"}:
            assert layout["routed_source"] == (
                "block15_post_attention_routed_input"
            )
            assert layout["shared_source"] == (
                "block15_post_attention_routed_input"
            )
        else:
            assert layout["source"] == "block15_post_attention_routed_input"


def test_result_schema_separates_training_coverage_from_heldout_nonfiring():
    schema = _load_json(RESULT_SCHEMA)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    properties = schema["properties"]
    assert properties["protocol_id"]["const"] == PROTOCOL_ID
    assert properties["panel"]["enum"] == ["sae", "transcoder"]
    assert set(properties["conditions"]["properties"]) == {"H", "R", "B", "F"}

    condition = schema["$defs"]["conditionResult"]
    assert "cumulative_training_coverage" in condition["required"]
    assert "heldout_nonfiring" in condition["required"]
    assert (
        condition["properties"]["cumulative_training_coverage"]["$ref"]
        == "#/$defs/coverageCount"
    )
    assert (
        condition["properties"]["heldout_nonfiring"]["$ref"]
        == "#/$defs/heldoutNonfiring"
    )
    assert condition["properties"]["exact_l0"]["const"] == 64

    parents = properties["parents"]
    assert "recomposition_sidecar_content_sha256" in parents["required"]
    assert "recomposition_sidecar_file_sha256" in parents["required"]
    sae_parent_properties = schema["allOf"][0]["then"]["properties"]["parents"][
        "properties"
    ]
    transcoder_parent_properties = schema["allOf"][0]["else"]["properties"][
        "parents"
    ]["properties"]
    assert sae_parent_properties["train_target_manifest_sha256"]["const"] == (
        EXPECTED_ARTIFACTS["train_target_manifest_sha256"]
    )
    assert (
        transcoder_parent_properties["train_source_overlay_manifest_sha256"][
            "const"
        ]
        == "8f27dcf9c1dc0c7f3ffa2e2941daec13d8a72c7204077f09420e15aacb02ffe0"
    )


def test_report_schema_freezes_complete_matrix_and_no_interpolation_q5():
    schema = _load_json(REPORT_SCHEMA)
    assert schema["properties"]["protocol_id"]["const"] == PROTOCOL_ID
    panels = schema["properties"]["panels"]
    assert set(panels["properties"]) == {"sae", "transcoder"}
    assert schema["$defs"]["nineHashes"]["minItems"] == 9
    assert schema["$defs"]["nineHashes"]["maxItems"] == 9
    assert schema["$defs"]["fixedQuality"]["properties"][
        "interpolation_used"
    ]["const"] is False
    assert schema["properties"]["budget"]["properties"]["hard_limit"][
        "const"
    ] == 50
    assert schema["properties"]["headline"]["properties"][
        "nominal_candidate_exposure_percent"
    ]["const"] == pytest.approx(11.764705882352942)


def test_paired_score_and_budget_schemas_freeze_inference_and_cash_inputs():
    paired = _load_json(PAIRED_SCHEMA)
    assert paired["properties"]["protocol_id"]["const"] == PROTOCOL_ID
    assert paired["properties"]["panel"]["enum"] == ["sae", "transcoder"]
    assert set(paired["properties"]["sequence_sse"]["properties"]) == {
        "H",
        "R",
        "B",
        "F",
    }
    assert "sequence_sst" in paired["required"]
    assert "valid_tokens" in paired["required"]

    budget = _load_json(BUDGET_SCHEMA)
    assert budget["properties"]["protocol_id"]["const"] == PROTOCOL_ID
    assert budget["properties"]["currency"]["const"] == "USD"
    assert budget["properties"]["hard_limit"]["const"] == 50
    assert "line_items" in budget["required"]


def test_evaluation_partial_schema_is_compact_self_hashed_host_evidence():
    partial = _load_json(PARTIAL_SCHEMA)
    assert partial["properties"]["schema_version"]["const"] == (
        "gemma4-routed-shared-hybrid-evaluation-partial-v1"
    )
    assert partial["properties"]["panel"]["enum"] == ["sae", "transcoder"]
    assert partial["properties"]["condition"]["enum"] == ["H", "R", "B", "F"]
    assert partial["properties"]["sequence_position_bits_hex"]["items"][
        "pattern"
    ] == "^[0-9a-f]{256}$"
    assert "partial_sha256" in partial["required"]


def test_protocol_documents_the_frozen_decisions():
    text = PROTOCOL.read_text(encoding="utf-8")
    for required in (
        "The only valid condition enum is `H/R/B/F`.",
        "The valid panel identifier is `sae`.",
        "The valid panel identifier is `transcoder`.",
        "USD 50.00",
        "cumulative training coverage",
        "held-out nonfiring rate",
        "Do not interpolate.",
        "No new model activation harvest is authorized.",
    ):
        assert required in text
