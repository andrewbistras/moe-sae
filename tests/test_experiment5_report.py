from __future__ import annotations

from pathlib import Path

import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_report import (
    BUDGET_SCHEMA_VERSION,
    CONDITIONS,
    EVAL_SOURCE_SHA256,
    EVAL_TARGET_SHA256,
    MILESTONES,
    PAIRED_SCHEMA_VERSION,
    PROTOCOL_ID,
    RESULT_SCHEMA_VERSION,
    SEEDS,
    SIDECAR_CONTENT_SHA256,
    SIDECAR_FILE_SHA256,
    SOURCE_CAPTURE_CONTRACT_SHA256,
    TRAIN_SOURCE_SHA256,
    TRAIN_TARGET_SHA256,
    Experiment5ReportError,
    build_experiment5_budget_ledger,
    build_experiment5_report,
    render_experiment5_report,
    write_experiment5_report,
)

SOURCE_REVISION = "c" * 40
CONFIG_SHAS = {"sae": "a" * 64, "transcoder": "b" * 64}
EVAL_TOKENS = 250_000
SST = 1_000.0


def _score(ev: float) -> dict[str, float]:
    nmse = 1.0 - ev
    return {
        "sse": nmse * SST,
        "sst": SST,
        "nmse": nmse,
        "ev": ev,
    }


def _ev(
    condition: str,
    milestone: int,
    seed: int,
    *,
    nonpositive_denominator: bool,
) -> float:
    values = {
        "H": {2_000_000: 0.50, 5_000_000: 0.58, 10_000_000: 0.65},
        "R": {2_000_000: 0.28, 5_000_000: 0.35, 10_000_000: 0.40},
        "B": {2_000_000: 0.39, 5_000_000: 0.46, 10_000_000: 0.50},
        "F": {2_000_000: 0.43, 5_000_000: 0.52, 10_000_000: 0.57},
    }
    value = values[condition][milestone]
    if nonpositive_denominator and condition == "F" and milestone == 10_000_000:
        value = 0.49
    return value + 0.001 * seed


def _parents(panel: str) -> dict[str, str | None]:
    return {
        "train_target_manifest_sha256": TRAIN_TARGET_SHA256,
        "eval_target_manifest_sha256": EVAL_TARGET_SHA256,
        "train_source_overlay_manifest_sha256": (
            TRAIN_SOURCE_SHA256 if panel == "transcoder" else None
        ),
        "eval_source_overlay_manifest_sha256": (
            EVAL_SOURCE_SHA256 if panel == "transcoder" else None
        ),
        "source_overlay_capture_contract_sha256": (
            SOURCE_CAPTURE_CONTRACT_SHA256 if panel == "transcoder" else None
        ),
        "recomposition_sidecar_content_sha256": SIDECAR_CONTENT_SHA256,
        "recomposition_sidecar_file_sha256": SIDECAR_FILE_SHA256,
    }


def _bank_rows(milestone: int) -> list[dict[str, int]]:
    return [
        {
            "bank_id": bank_id,
            "routed_training_components": milestone * 8 // 128,
            "routed_evaluation_components": 15_625,
            "cumulative_training_fired_features": 512,
            "heldout_fired_features": 512,
            "bank_features": 512,
        }
        for bank_id in range(128)
    ]


def _cost(condition: str, milestone: int) -> float:
    return {
        "H": {2_000_000: 20.0, 5_000_000: 45.0, 10_000_000: 80.0},
        "R": {2_000_000: 22.0, 5_000_000: 49.0, 10_000_000: 88.0},
        "B": {2_000_000: 19.0, 5_000_000: 43.0, 10_000_000: 77.0},
        "F": {2_000_000: 25.0, 5_000_000: 60.0, 10_000_000: 100.0},
    }[condition][milestone]


def _paired_scores_and_sha(
    panel: str,
    seed: int,
    milestone: int,
    *,
    nonpositive_denominator: bool,
) -> tuple[dict, str]:
    sequence_sse = {
        condition: [
            _score(
                _ev(
                    condition,
                    milestone,
                    seed,
                    nonpositive_denominator=nonpositive_denominator,
                )
            )["sse"]
            / 2,
            _score(
                _ev(
                    condition,
                    milestone,
                    seed,
                    nonpositive_denominator=nonpositive_denominator,
                )
            )["sse"]
            / 2,
        ]
        for condition in CONDITIONS
    }
    body = {
        "schema_version": PAIRED_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "panel": panel,
        "seed": seed,
        "checkpoint_target_tokens": milestone,
        "packed_sequence_ids": [1001, 1002],
        "valid_tokens": [125_000, 125_000],
        "sequence_sst": [SST / 2, SST / 2],
        "sequence_sse": sequence_sse,
    }
    artifact_sha = canonical_sha256(body)
    return {**body, "artifact_sha256": artifact_sha}, artifact_sha


def _condition_result(
    panel: str,
    condition: str,
    seed: int,
    milestone: int,
    *,
    nonpositive_denominator: bool,
    f_generation: str,
) -> dict:
    hybrid = condition in {"H", "R"}
    total = 69_632 if condition != "B" else 8_192
    candidates = (
        8_192
        if condition in {"H", "R", "B"}
        else 69_632
    )
    full = _score(
        _ev(
            condition,
            milestone,
            seed,
            nonpositive_denominator=nonpositive_denominator,
        )
    )
    fired_training = total - 10
    fired_eval = total - 20
    chip_seconds = _cost(condition, milestone)
    generation = f_generation if condition == "F" else "v5e"
    return {
        "condition": condition,
        "total_features": total,
        "candidates_per_token": candidates,
        "nominal_exposure_fraction": candidates / total,
        "exact_l0": 64,
        "finite": True,
        "full_ffn": full,
        "routed": full if hybrid else None,
        "shared": full if hybrid else None,
        "oracle_shared_full_ffn": full if hybrid else None,
        "oracle_routed_full_ffn": full if hybrid else None,
        "error_cancellation_sse_delta": 0.0 if hybrid else None,
        "cumulative_training_coverage": {
            "fired_features": fired_training,
            "total_features": total,
            "firing_fraction": fired_training / total,
        },
        "heldout_nonfiring": {
            "fired_features": fired_eval,
            "total_features": total,
            "nonfiring_fraction": 1.0 - fired_eval / total,
        },
        "routed_bank_exposure": _bank_rows(milestone) if hybrid else [],
        "training": {
            "marginal_chip_seconds": chip_seconds,
            "wall_seconds": chip_seconds / 16,
            "tokens_per_second": milestone / (chip_seconds / 16),
            "accelerator_type": (
                "v4-32" if generation == "v4" else "v5litepod-16"
            ),
            "accelerator_generation": generation,
            "chip_count": 16,
            "useful_mfu": 0.2,
            "training_receipt_sha256": "d" * 64,
            "checkpoint_firing_mask_sha256": "e" * 64,
        },
    }


def _result(
    panel: str,
    seed: int,
    milestone: int,
    prediction_sha: str,
    *,
    nonpositive_denominator: bool,
    f_generation: str,
) -> dict:
    conditions = {
        condition: _condition_result(
            panel,
            condition,
            seed,
            milestone,
            nonpositive_denominator=nonpositive_denominator,
            f_generation=f_generation,
        )
        for condition in CONDITIONS
    }
    contrasts = {
        "H-R": ("H", "R"),
        "H-B": ("H", "B"),
        "H-F": ("H", "F"),
        "F-B": ("F", "B"),
    }
    paired = {}
    for name, (left, right) in contrasts.items():
        estimate = (
            conditions[left]["full_ffn"]["ev"]
            - conditions[right]["full_ffn"]["ev"]
        )
        paired[name] = {
            "endpoint": "full_ffn",
            "estimate": estimate,
            "lower": estimate,
            "upper": estimate,
            "confidence_level": 0.95,
            "bootstrap_unit": "packed_sequence",
            "bootstrap_replicates": 20_000,
        }
    body = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "curve_point_measured",
        "panel": panel,
        "seed": seed,
        "checkpoint_target_tokens": milestone,
        "checkpoint_actual_tokens": {
            condition: milestone for condition in CONDITIONS
        },
        "evaluation_tokens": EVAL_TOKENS,
        "source_revision": SOURCE_REVISION,
        "completed_at": f"2026-07-28T{18 + seed:02d}:00:00Z",
        "config_sha256": CONFIG_SHAS[panel],
        "parents": _parents(panel),
        "common_full_ffn_sst": SST,
        "conditions": conditions,
        "paired_effects": paired,
        "predictions_artifact_sha256": prediction_sha,
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def _grid(
    *,
    nonpositive_denominator: bool = False,
    f_generation: str = "v5e",
) -> tuple[list[dict], list[dict]]:
    results = []
    paired = []
    for panel in ("sae", "transcoder"):
        for seed in SEEDS:
            for milestone in MILESTONES:
                score, score_sha = _paired_scores_and_sha(
                    panel,
                    seed,
                    milestone,
                    nonpositive_denominator=nonpositive_denominator,
                )
                paired.append(score)
                results.append(
                    _result(
                        panel,
                        seed,
                        milestone,
                        score_sha,
                        nonpositive_denominator=nonpositive_denominator,
                        f_generation=f_generation,
                    )
                )
    return results, paired


def _budget(spent: float = 5.0) -> dict:
    return build_experiment5_budget_ledger(
        line_items=[
            {"category": "controller", "amount_usd": spent * 0.6},
            {"category": "storage-and-operations", "amount_usd": spent * 0.4},
        ],
        quota_funded_tpu_chip_seconds=12_345.0,
    )


def test_report_computes_gain_hierarchical_effects_and_earliest_observed_q5():
    results, paired = _grid()
    report = build_experiment5_report(
        results=results,
        paired_scores=paired,
        budget_ledger=_budget(),
    )

    assert report["status"] == "headline_ready"
    primary = report["panels"]["transcoder"]
    assert primary["headline_ready"] is True
    assert primary["ten_million_contrasts"]["H-R"][
        "passes_headline_rule"
    ] is True
    assert primary["ten_million_contrasts"]["H-B"][
        "passes_headline_rule"
    ] is True
    assert primary["gain_recovery"]["status"] == "defined"
    assert primary["gain_recovery"]["g5_percent"] == pytest.approx(
        100 * 0.15 / 0.07
    )

    fixed = primary["fixed_quality_cost"]
    assert fixed["status"] == "defined"
    assert fixed["reaching_seed_count"] == 3
    assert [
        row["earliest_h_checkpoint_tokens"] for row in fixed["per_seed"]
    ] == [5_000_000, 5_000_000, 5_000_000]
    assert fixed["median_q5_percent"] == pytest.approx(45.0)
    assert fixed["interpolation_used"] is False
    assert report["headline"]["q5_percent"] == pytest.approx(45.0)


def test_gain_is_undefined_for_nonpositive_f_minus_b():
    results, paired = _grid(nonpositive_denominator=True)
    report = build_experiment5_report(
        results=results,
        paired_scores=paired,
        budget_ledger=_budget(),
    )
    gain = report["panels"]["transcoder"]["gain_recovery"]
    assert gain["status"] == "undefined_nonpositive_denominator"
    assert gain["g5_percent"] is None
    assert gain["denominator_mean"] < 0


def test_cross_generation_cost_is_retained_but_not_interpreted():
    results, paired = _grid(f_generation="v4")
    report = build_experiment5_report(
        results=results,
        paired_scores=paired,
        budget_ledger=_budget(),
    )
    primary = report["panels"]["transcoder"]
    assert primary["fixed_quality_cost"]["status"] == "hardware_incomparable"
    assert primary["fixed_quality_cost"]["median_q5_percent"] is None
    assert all(
        row["median_h_over_f_percent"] is None
        for row in primary["equal_token_cost"]
    )


def test_over_budget_is_recorded_and_cannot_be_headline_ready():
    results, paired = _grid()
    report = build_experiment5_report(
        results=results,
        paired_scores=paired,
        budget_ledger=_budget(50.01),
    )
    assert report["status"] == "budget_stopped"
    assert report["budget"]["within_limit"] is False
    assert report["budget"]["incremental_cash_spent"] == pytest.approx(50.01)


def test_paired_score_tampering_fails_closed():
    results, paired = _grid()
    paired[0]["sequence_sse"]["H"][0] += 1.0
    with pytest.raises(Experiment5ReportError, match="self-hash"):
        build_experiment5_report(
            results=results,
            paired_scores=paired,
            budget_ledger=_budget(),
        )


def test_report_render_and_create_only_write(tmp_path: Path):
    results, paired = _grid()
    report = build_experiment5_report(
        results=results,
        paired_scores=paired,
        budget_ledger=_budget(),
    )
    markdown = render_experiment5_report(report)
    assert "Fixed-quality Q5: 45.00%" in markdown
    assert "cumulative training fired" in markdown
    assert "held-out nonfiring" in markdown

    output = tmp_path / "report"
    json_path, markdown_path = write_experiment5_report(output, report)
    assert json_path.name == "experiment5-result.json"
    assert markdown_path.read_text(encoding="utf-8") == markdown
    with pytest.raises(Experiment5ReportError, match="already exists"):
        write_experiment5_report(output, report)
