from __future__ import annotations

import json
from pathlib import Path

import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.horizon_curve_report import (
    MILESTONES,
    PROTOCOL_ID,
    REPORT_SCHEMA_VERSION,
    RUN_CONTRACT_SCHEMA_VERSION,
    SAE_RESULT_SCHEMA_VERSION,
    TRANSCODER_RESULT_SCHEMA_VERSION,
    HorizonCurveReportError,
    build_horizon_curve_report,
    render_horizon_curve_report,
    write_horizon_curve_report,
)

CONDITIONS = ("A", "R", "B", "F")


def _digest(label: str) -> str:
    return canonical_sha256({"label": label})


def _sealed(body: dict, field: str) -> dict:
    return {**body, field: canonical_sha256(body)}


def _score(ev: float) -> dict[str, float]:
    nmse = 1.0 - ev
    return {
        "sse": 100.0 * nmse,
        "sst": 100.0,
        "nmse": nmse,
        "ev": ev,
    }


def _contracts(
    tmp_path: Path,
    *,
    accelerator_by_model_condition: dict[tuple[str, str], str] | None = None,
) -> tuple[list[str], dict[tuple[str, str], dict]]:
    accelerators = accelerator_by_model_condition or {}
    values: dict[tuple[str, str], dict] = {}
    arguments = []
    for model in ("sae", "transcoder"):
        for condition in CONDITIONS:
            body = {
                "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
                "run_id": f"horizon-{model}-{condition.lower()}",
                "project_commit": "a" * 40,
                "phase": "learning_horizon_10m",
                "condition": condition,
                "seed": 0,
                "scientific_model": model,
                "capture_manifest_sha256": _digest("train-target"),
                "source_overlay_manifest_sha256": (
                    _digest("train-source") if model == "transcoder" else None
                ),
                "accelerator_type": accelerators.get(
                    (model, condition),
                    "v6e-16",
                ),
            }
            value = _sealed(body, "run_contract_sha256")
            path = tmp_path / f"{model}-{condition}.run-contract.json"
            path.write_text(json.dumps(value), encoding="ascii")
            values[(model, condition)] = value
            arguments.append(f"{model}:{condition}={path}")
    return arguments, values


def _sae_result(target: int) -> dict:
    costs = {
        "A": float(target) / 1_000.0,
        "R": float(target) / 900.0,
        "B": float(target) / 1_800.0,
        "F": float(target) / 800.0,
    }
    component = {
        condition: _score(
            0.10
            + target / 100_000_000
            + 0.01 * CONDITIONS.index(condition)
        )
        for condition in CONDITIONS
    }
    routed = {
        condition: _score(
            -0.20
            + target / 200_000_000
            + 0.01 * CONDITIONS.index(condition)
        )
        for condition in CONDITIONS
    }
    body = {
        "schema_version": SAE_RESULT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "curve_point_measured",
        "phase": "learning_horizon_10m",
        "seed": 0,
        "training_capture_manifest_sha256": _digest("train-target"),
        "evaluation_capture_manifest_sha256": _digest("eval-target"),
        "evaluation_tokens": 250_000,
        "checkpoint_target_tokens": target,
        "checkpoint_actual_tokens": {
            condition: target + 15_232 for condition in CONDITIONS
        },
        "endpoint_centering": {
            "component_mean_sha256": _digest("component-mean"),
            "routed_mean_sha256": _digest("routed-mean"),
            "contract_sha256": _digest("centering"),
        },
        "endpoints": {"component": component, "routed": routed},
        "primary": {
            "g2_percent": 125.0,
            "c2_percent": 100.0 * costs["A"] / costs["F"],
        },
        "training_chip_seconds": costs,
    }
    return _sealed(body, "result_sha256")


def _transcoder_result(
    target: int,
    contracts: dict[tuple[str, str], dict],
) -> dict:
    costs = {
        "A": float(target) / 1_200.0,
        "R": float(target) / 1_100.0,
        "B": float(target) / 2_000.0,
        "F": float(target) / 900.0,
    }
    conditions = {}
    for index, condition in enumerate(CONDITIONS):
        component = _score(0.20 + target / 100_000_000 + 0.01 * index)
        routed = _score(-0.10 + target / 200_000_000 + 0.01 * index)
        conditions[condition] = {
            "endpoints": {
                "component": component,
                "routed": routed,
                "oracle_shared": component,
            },
            "training": {
                "marginal_chip_seconds": costs[condition],
                "marginal_wall_seconds": costs[condition] / 16.0,
                "tokens_per_second": target / (costs[condition] / 16.0),
                "useful_mfu": 0.21,
            },
        }
    body = {
        "schema_version": TRANSCODER_RESULT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "curve_point_measured",
        "seed": 0,
        "source_revision": "b" * 40,
        "training_source_revision": "a" * 40,
        "config_sha256": _digest("transcoder-config"),
        "parents": {
            "train_source_manifest_sha256": _digest("train-source"),
            "train_target_manifest_sha256": _digest("train-target"),
            "eval_source_manifest_sha256": _digest("eval-source"),
            "eval_target_manifest_sha256": _digest("eval-target"),
            "recomposition_sidecar_sha256": _digest("sidecar"),
            "common_sst_receipt_sha256": _digest(f"sst-{target}"),
            "run_contract_sha256": {
                condition: contracts[
                    ("transcoder", condition)
                ]["run_contract_sha256"]
                for condition in CONDITIONS
            },
        },
        "paired_scores": {
            "score_artifact_sha256": _digest(f"scores-{target}"),
        },
        "learning_curve_checkpoint": {
            "target_tokens": target,
            "actual_tokens_by_condition": {
                condition: target + 15_232 for condition in CONDITIONS
            },
        },
        "conditions": conditions,
        "primary": {
            "g3": {
                "gain_recovery_percent": 140.0,
                "status": "stable",
            },
            "c3_percent": 100.0 * costs["A"] / costs["F"],
        },
    }
    return _sealed(body, "result_sha256")


def _write_results(
    tmp_path: Path,
    contracts: dict[tuple[str, str], dict],
) -> tuple[list[Path], list[Path]]:
    sae_paths = []
    transcoder_paths = []
    for target in MILESTONES:
        sae = tmp_path / f"sae-{target}.json"
        sae.write_text(json.dumps(_sae_result(target)), encoding="ascii")
        sae_paths.append(sae)
        transcoder = tmp_path / f"transcoder-{target}.json"
        transcoder.write_text(
            json.dumps(_transcoder_result(target, contracts)),
            encoding="ascii",
        )
        transcoder_paths.append(transcoder)
    return sae_paths, transcoder_paths


def test_builds_one_sorted_self_hashed_six_point_report(tmp_path: Path) -> None:
    run_contracts, contract_values = _contracts(tmp_path)
    sae, transcoder = _write_results(tmp_path, contract_values)

    result = build_horizon_curve_report(
        sae_results=list(reversed(sae)),
        transcoder_results=[transcoder[1], transcoder[2], transcoder[0]],
        run_contracts=run_contracts,
    )

    assert result["schema_version"] == REPORT_SCHEMA_VERSION
    assert canonical_sha256(
        {key: value for key, value in result.items() if key != "report_sha256"}
    ) == result["report_sha256"]
    for model in ("sae", "transcoder"):
        curve = result["scientific_models"][model]["curve"]
        assert [row["target_training_tokens"] for row in curve] == list(
            MILESTONES
        )
        assert curve[-1]["conditions"]["A"]["held_out_evaluation"][
            "component"
        ]["ev"] > curve[0]["conditions"]["A"]["held_out_evaluation"][
            "component"
        ]["ev"]
        assert curve[0]["cost_interpretation"][
            "interpretable_a_over_f_percent"
        ] is not None
    assert (
        result["scientific_models"]["sae"]["curve"][0]["conditions"]["A"][
            "training"
        ]["marginal_wall_seconds"]
        == pytest.approx(125.0)
    )

    rendered = render_horizon_curve_report(result)
    assert "# Experiment 4" in rendered
    assert "10,000,000 training tokens" in rendered
    assert result["report_sha256"] in rendered


def test_mixed_generation_cost_is_retained_but_not_interpreted(
    tmp_path: Path,
) -> None:
    accelerators = {
        ("sae", "A"): "v5litepod-16",
        ("sae", "F"): "v6e-16",
    }
    run_contracts, contract_values = _contracts(
        tmp_path,
        accelerator_by_model_condition=accelerators,
    )
    sae, transcoder = _write_results(tmp_path, contract_values)

    result = build_horizon_curve_report(
        sae_results=sae,
        transcoder_results=transcoder,
        run_contracts=run_contracts,
    )
    sae_model = result["scientific_models"]["sae"]
    point = sae_model["curve"][0]

    assert not sae_model["cost_comparability"]["pairs"]["A/F"]["comparable"]
    assert point["cost_interpretation"][
        "source_measured_a_over_f_percent"
    ] == pytest.approx(80.0)
    assert point["cost_interpretation"][
        "interpretable_a_over_f_percent"
    ] is None
    rendered = render_horizon_curve_report(result)
    assert "mixed accelerator generations" in rendered
    assert "not comparable across hardware generations" in rendered


def test_transcoder_run_contract_must_match_scientific_result(
    tmp_path: Path,
) -> None:
    run_contracts, contract_values = _contracts(tmp_path)
    sae, transcoder = _write_results(tmp_path, contract_values)
    for path in transcoder:
        changed = json.loads(path.read_text(encoding="ascii"))
        body = dict(changed)
        body.pop("result_sha256")
        body["parents"]["run_contract_sha256"]["A"] = "0" * 64
        path.write_text(
            json.dumps(_sealed(body, "result_sha256")),
            encoding="ascii",
        )

    with pytest.raises(HorizonCurveReportError, match="provenance differs"):
        build_horizon_curve_report(
            sae_results=sae,
            transcoder_results=transcoder,
            run_contracts=run_contracts,
        )


def test_output_directory_is_create_only(tmp_path: Path) -> None:
    run_contracts, contract_values = _contracts(tmp_path)
    sae, transcoder = _write_results(tmp_path, contract_values)
    result = build_horizon_curve_report(
        sae_results=sae,
        transcoder_results=transcoder,
        run_contracts=run_contracts,
    )
    output = tmp_path / "report"

    json_path, markdown_path = write_horizon_curve_report(output, result)
    assert json.loads(json_path.read_text(encoding="ascii")) == result
    assert result["report_sha256"] in markdown_path.read_text(encoding="utf-8")
    with pytest.raises(HorizonCurveReportError, match="already exists"):
        write_horizon_curve_report(output, result)
