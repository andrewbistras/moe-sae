from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from moe_sae.config import Gemma4Config
from moe_sae.diagnostics import (
    PAIRED_BOOTSTRAP_REPLICATES,
    SEQUENCE_SSE_SCHEMA_VERSION,
    validate_sequence_sse_summary,
)
from moe_sae.gate1_protocol import load_gate1_protocol, validate_gate1_protocol
from moe_sae.gate1_report import (
    GATE1_CELL_RESULT_SCHEMA,
    GATE1_REPORT_SCHEMA,
    Gate1ReportError,
    build_gate1_report,
    required_gate1_cells,
    validate_gate1_cell_record,
    validate_gate1_cell_records,
    validate_gate1_report,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_gate1_protocol(ROOT / "configs" / "gemma4_gate1_audit.yaml")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _nmse(key: tuple[str, str, str]) -> float:
    checkpoint_label, split, route_mode = key
    checkpoint = PROTOCOL.checkpoints[checkpoint_label]
    split_offset = 0.01 if split == "train_stratified" else 0.0
    if checkpoint.milestone_millions == 5:
        return {"A": 0.45, "B": 0.42, "F": 0.38}[checkpoint.condition]
    if checkpoint.milestone_millions == 40:
        return 0.34
    baseline_b = 0.30 + 0.001 * (checkpoint.seed - 1) + split_offset
    if checkpoint.condition == "B":
        return baseline_b
    true_a = baseline_b + 0.002
    if route_mode == "shuffled":
        return true_a + 0.006
    if route_mode == "all_banks":
        return true_a - 0.006
    return true_a


def _summary(
    key: tuple[str, str, str],
    *,
    nmse: float | None = None,
    sst_scale: float = 1.0,
) -> dict:
    split = key[1]
    first_sequence = 100 if split == "heldout" else 200
    selected_nmse = _nmse(key) if nmse is None else nmse
    clusters = [
        {
            "sequence_id": first_sequence,
            "tokens": 1_000_000,
            "sst": 500.0 * sst_scale,
            "sse": 500.0 * sst_scale * selected_nmse,
        },
        {
            "sequence_id": first_sequence + 1,
            "tokens": 1_000_000,
            "sst": 700.0 * sst_scale,
            "sse": 700.0 * sst_scale * selected_nmse,
        },
    ]
    sst = sum(row["sst"] for row in clusters)
    sse = sum(row["sse"] for row in clusters)
    return validate_sequence_sse_summary(
        {
            "schema_version": SEQUENCE_SSE_SCHEMA_VERSION,
            "label": "|".join(key),
            "tokens": 2_000_000,
            "sequence_count": 2,
            "sst": sst,
            "sse": sse,
            "nmse": sse / sst,
            "ev": 1.0 - sse / sst,
            "clusters": clusters,
        }
    )


def _cell(key: tuple[str, str, str]) -> dict:
    checkpoint_label, split_name, route_mode = key
    checkpoint = PROTOCOL.checkpoints[checkpoint_label]
    split = PROTOCOL.splits[split_name]
    config = Gemma4Config()
    total_features = (
        config.visible_features
        if checkpoint.condition == "B"
        else config.total_features
    )
    shuffled = route_mode == "shuffled"
    routed_a = checkpoint.condition == "A" and route_mode != "all_banks"
    return {
        "schema_version": GATE1_CELL_RESULT_SCHEMA,
        "protocol_sha256": PROTOCOL.sha256,
        "cell_identity_sha256": _digest(f"cell:{key}"),
        "consumer_environment_sha256": _digest("consumer-environment"),
        "process_result_sha256s": [
            _digest(f"process:{key}:{process_index}") for process_index in range(4)
        ],
        "checkpoint_label": checkpoint_label,
        "checkpoint_expectation_sha256": checkpoint.expectation.sha256,
        "checkpoint_descriptor_sha256": _digest(f"descriptor:{checkpoint_label}"),
        "checkpoint_uri": checkpoint.checkpoint_uri,
        "capture_config_sha256": PROTOCOL.capture.config_sha256,
        "capture_checkpoint_sha256": PROTOCOL.capture.checkpoint_sha256,
        "capture_manifest_sha256": (
            PROTOCOL.capture.train_manifest_sha256
            if split.source == "train"
            else PROTOCOL.capture.eval_manifest_sha256
        ),
        "split": split_name,
        "source": split.source,
        "ranges": [item.to_list() for item in split.ranges],
        "route_mode": route_mode,
        "route_permutation_domain": (
            PROTOCOL.route_permutation.domain if shuffled else None
        ),
        "route_permutation_key": (PROTOCOL.route_permutation.key if shuffled else None),
        "tokens": split.tokens,
        "exact_l0": True,
        "topk_valid": True,
        "dead_features": min(7, total_features),
        "total_features": total_features,
        "fired_mask_sha256": _digest(f"fired:{key}"),
        "audit_capacity": (
            {
                "margin": 0.10,
                "round_to": 8,
                "local_observed_max_by_process": [100, 101, 102, 103],
                "global_observed_max": 103,
                "capacity": 120,
            }
            if routed_a
            else None
        ),
        "sequence_sse": _summary(key),
    }


def _cells() -> dict[tuple[str, str, str], dict]:
    return {key: _cell(key) for key in required_gate1_cells(PROTOCOL)}


def test_required_matrix_collapses_only_the_three_reused_a_final_cells() -> None:
    required = required_gate1_cells(PROTOCOL)

    assert len(required) == 22
    assert len(set(required)) == 22
    for seed in (1, 2, 3):
        assert required.count((f"A_seed{seed}_50m", "heldout", "true")) == 1
    report = build_gate1_report(PROTOCOL, _cells())
    assert report["matrix_accounting"] == {
        "expanded_requests": 25,
        "unique_cells": 22,
        "duplicate_requests_collapsed": 3,
    }


def test_complete_report_has_exact_pairs_learning_curve_and_cautious_decision() -> None:
    report = build_gate1_report(PROTOCOL, _cells())

    assert report["schema_version"] == GATE1_REPORT_SCHEMA
    assert report["protocol_sha256"] == PROTOCOL.sha256
    assert report["cell_count"] == len(report["cells"]) == 22
    assert len(report["comparisons"]) == 12
    assert len(report["seed_summaries"]) == 4
    assert all(
        comparison["bootstrap_95"]["replicates"] == PAIRED_BOOTSTRAP_REPLICATES
        for comparison in report["comparisons"]
    )
    assert all(
        comparison["paired_sequence_sse"]["tokens"] == 2_000_000
        for comparison in report["comparisons"]
    )
    assert all(
        comparison["paired_sequence_sse"]["sequence_count"] == 2
        for comparison in report["comparisons"]
    )

    summaries = {(row["family"], row["split"]): row for row in report["seed_summaries"]}
    heldout_ab = summaries[("A_vs_B", "heldout")]
    routing = summaries[("shuffled_to_true", "heldout")]
    all_banks = summaries[("true_to_all_banks", "heldout")]
    assert heldout_ab["all_seeds_negative"]
    assert heldout_ab["mean_point_ev"] == pytest.approx(-0.002)
    assert routing["mean_point_ev"] == pytest.approx(0.006)
    assert routing["material_positive_seed_count"] == 3
    assert routing["positive_ci_seed_count"] == 3
    assert all_banks["mean_point_ev"] == pytest.approx(0.006)
    assert all(row["token_pooling"] is False for row in summaries.values())
    assert all(len(row["per_seed"]) == 3 for row in summaries.values())
    assert all("bootstrap_95" not in row for row in summaries.values())

    curve = report["learning_curve"]
    assert [row["condition"] for row in curve["rows"]] == ["A", "B", "F", "A", "A"]
    assert [row["checkpoint_tokens_seen"] for row in curve["rows"]] == [
        5_013_504,
        5_013_504,
        5_013_504,
        40_009_728,
        50_000_000,
    ]
    assert curve["rows"][-1]["reused_final_baseline_cell"] is True
    assert curve["reused_cell"] == ["A_seed1_50m", "heldout", "true"]

    decision = report["mechanism_decision"]
    assert (
        decision["router_identity"]["evidence_grade"]
        == "material-positive-in-all-seeds"
    )
    assert (
        decision["all_banks_visibility"]["evidence_grade"]
        == "material-positive-in-all-seeds"
    )
    assert decision["router_identity"]["training_time_causality_established"] is False
    assert decision["all_banks_visibility"]["training_distribution_matched"] is False
    assert (
        decision["ordinary_overfit"]["evidence_code"]
        == "ordinary-overfit-not-supported-by-sampled-train-and-heldout"
    )
    assert decision["ordinary_overfit"]["ruled_out"] is False
    assert decision["original_expert_decomposition_thesis_tested"] is False
    assert decision["gate2_required"] is True


def test_report_is_deterministic_json_safe_and_self_hashed() -> None:
    cells = _cells()
    first = build_gate1_report(PROTOCOL, cells)
    second = build_gate1_report(PROTOCOL, dict(reversed(tuple(cells.items()))))

    assert first == second
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    assert validate_gate1_report(first, PROTOCOL) == first

    tampered = copy.deepcopy(first)
    tampered["cells"][0]["ev"] += 0.01
    with pytest.raises(Gate1ReportError, match="self-hash"):
        validate_gate1_report(tampered, PROTOCOL)


def test_report_requires_the_precommitted_twenty_thousand_bootstraps() -> None:
    mapping = PROTOCOL.to_mapping()
    mapping["statistics"]["bootstrap_replicates"] = 1_000
    changed = validate_gate1_protocol(mapping)

    with pytest.raises(Gate1ReportError, match="exactly 20000"):
        build_gate1_report(changed, {})


def test_cell_table_records_descriptor_l0_features_and_candidate_exposure() -> None:
    report = build_gate1_report(PROTOCOL, _cells())
    by_key = {tuple(row["cell"]): row for row in report["cells"]}

    a_all = by_key[("A_seed1_50m", "heldout", "all_banks")]
    a_true = by_key[("A_seed1_50m", "heldout", "true")]
    b_true = by_key[("B_seed1_50m", "heldout", "true")]
    assert a_all["total_features"] == 131_072
    assert a_all["candidates_per_token"] == 131_072
    assert a_all["candidate_exposure_vs_full_dictionary"] == 1.0
    assert a_true["total_features"] == 131_072
    assert a_true["candidates_per_token"] == 8_192
    assert a_true["candidate_exposure_vs_full_dictionary"] == 0.0625
    assert b_true["total_features"] == 8_192
    assert b_true["candidates_per_token"] == 8_192
    assert all(row["exact_l0"] is True for row in report["cells"])
    assert all(row["l0"] == 64 for row in report["cells"])
    assert all(row["topk_valid"] is True for row in report["cells"])
    assert all(
        len(row["checkpoint_descriptor_sha256"]) == 64 for row in report["cells"]
    )
    assert all(len(row["cell_record_sha256"]) == 64 for row in report["cells"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda record, _key: record.update(extra=True),
            "fields do not match schema",
        ),
        (
            lambda record, _key: record.update(protocol_sha256="0" * 64),
            "another Gate-1 protocol",
        ),
        (
            lambda record, _key: record.update(checkpoint_expectation_sha256="0" * 64),
            "expectation binding",
        ),
        (
            lambda record, _key: record.update(checkpoint_descriptor_sha256="A" * 64),
            "lowercase",
        ),
        (
            lambda record, _key: record.update(checkpoint_uri="gs://other/value"),
            "checkpoint URI",
        ),
        (
            lambda record, _key: record.update(capture_config_sha256="0" * 64),
            "capture_config_sha256",
        ),
        (
            lambda record, _key: record.update(capture_manifest_sha256="0" * 64),
            "capture manifest",
        ),
        (
            lambda record, _key: record.update(source="train"),
            "source",
        ),
        (
            lambda record, _key: record.update(ranges=[[0, 4]]),
            "frozen split",
        ),
        (
            lambda record, _key: record.update(tokens=4),
            "token count",
        ),
        (
            lambda record, _key: record.update(exact_l0=False),
            "exact_l0 must prove L0=64",
        ),
        (
            lambda record, _key: record.update(topk_valid=False),
            "topk_valid",
        ),
        (
            lambda record, _key: record.update(
                dead_features=record["total_features"] + 1
            ),
            "at most",
        ),
        (
            lambda record, _key: record.update(total_features=8_192),
            "total_features",
        ),
        (
            lambda record, _key: record["sequence_sse"].update(tokens=3),
            "sequence_sse is invalid",
        ),
    ],
)
def test_cell_validator_fails_closed_on_identity_and_scientific_drift(
    mutate, message: str
) -> None:
    key = ("A_seed1_50m", "heldout", "all_banks")
    record = _cell(key)
    mutate(record, key)

    with pytest.raises(Gate1ReportError, match=message):
        validate_gate1_cell_record(PROTOCOL, key, record)


def test_route_randomization_identity_is_present_only_for_shuffled() -> None:
    shuffled_key = ("A_seed1_50m", "heldout", "shuffled")
    shuffled = _cell(shuffled_key)
    shuffled["route_permutation_domain"] = None
    with pytest.raises(Gate1ReportError, match="domain"):
        validate_gate1_cell_record(PROTOCOL, shuffled_key, shuffled)

    true_key = ("A_seed1_50m", "heldout", "true")
    true = _cell(true_key)
    true["route_permutation_domain"] = PROTOCOL.route_permutation.domain
    true["route_permutation_key"] = PROTOCOL.route_permutation.key
    with pytest.raises(Gate1ReportError, match="must use null"):
        validate_gate1_cell_record(PROTOCOL, true_key, true)


def test_record_set_requires_every_unique_cell_and_one_common_split_measure() -> None:
    cells = _cells()
    cells.pop(("F_seed1_5m", "heldout", "true"))
    with pytest.raises(Gate1ReportError, match="missing cells"):
        validate_gate1_cell_records(PROTOCOL, cells)

    cells = _cells()
    cells[("A_seed1_5m", "heldout", "true")]["sequence_sse"] = _summary(
        ("A_seed1_5m", "heldout", "true"), sst_scale=2.0
    )
    with pytest.raises(Gate1ReportError, match="disagree on sequence/SST"):
        validate_gate1_cell_records(PROTOCOL, cells)


def test_mapping_key_must_match_record_identity() -> None:
    key = ("A_seed1_50m", "heldout", "true")
    record = _cell(key)
    record["route_mode"] = "shuffled"

    with pytest.raises(Gate1ReportError, match="mapping key"):
        validate_gate1_cell_record(PROTOCOL, key, record)


def test_report_module_has_no_jax_or_cloud_imports() -> None:
    source = (ROOT / "src" / "moe_sae" / "gate1_report.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert "jax" not in imported_roots
    assert "google" not in imported_roots
