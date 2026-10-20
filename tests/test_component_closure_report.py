from __future__ import annotations

import hashlib

import numpy as np
import pytest

from moe_sae.component_closure_config import (
    CHECKPOINT_TOKENS,
    CONDITIONS,
    PANELS,
    PROTOCOL_ID,
    SEEDS,
)
from moe_sae.component_closure_report import (
    EVIDENCE_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    SST_AGGREGATE_RELATIVE_TOLERANCE,
    SST_NORMALIZED_L1_TOLERANCE,
    SST_PER_SEQUENCE_RELATIVE_TOLERANCE,
    ComponentClosureReportError,
    _sst_equivalence_metrics,
    build_closure_report,
    render_markdown,
    validate_score_grid,
)
from moe_sae.contribution_artifacts import canonical_sha256

SHA = "a" * 64
REVISION = "b" * 40


def _evidence() -> dict:
    # EV values make F-B=0.50 and A-B=0.30, hence G=60%.
    ev = {"A": 0.70, "R": 0.55, "B": 0.40, "F": 0.90}
    cells = []
    for panel in PANELS:
        for seed in SEEDS:
            for checkpoint in CHECKPOINT_TOKENS:
                for condition in CONDITIONS:
                    sst = [1.0, 2.0, 3.0, 4.0]
                    cells.append(
                        {
                            "panel": panel,
                            "seed": seed,
                            "checkpoint_tokens": checkpoint,
                            "condition": condition,
                            "sequence_id": [10, 11, 12, 13],
                            "component_sse": [
                                value * (1.0 - ev[condition]) for value in sst
                            ],
                            "component_sst": sst,
                            "evaluation_result_sha256": SHA,
                            "evaluation_token_identity_sha256": "1" * 64,
                            "evaluation_target_manifest_sha256": "2" * 64,
                            "evaluation_token_count": 4,
                            "evaluation_packed_sequences": 4,
                            "evaluation_source_manifest_sha256": (
                                None if panel == "sae" else "e" * 64
                            ),
                            "centering_receipt_sha256": "3" * 64,
                            "centering_target_manifest_sha256": "b" * 64,
                            "target_mean_sha256": "4" * 64,
                            "endpoint_receipt_sha256": "5" * 64,
                            "endpoint_cache_key_sha256": "6" * 64,
                            "routed_mean_sha256": "7" * 64,
                            "oracle_shared_mean_sha256": "8" * 64,
                            "recomposition_sidecar_sha256": "9" * 64,
                            "source_manifest_sha256": (
                                None if panel == "sae" else "c" * 64
                            ),
                            "source_mean_sha256": (
                                None if panel == "sae" else "d" * 64
                            ),
                        }
                    )
    costs = []
    for panel in PANELS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                wall = {"A": 10.0, "R": 11.0, "B": 5.0, "F": 20.0}[condition]
                costs.append(
                    {
                        "panel": panel,
                        "seed": seed,
                        "condition": condition,
                        "accelerator_type": "v5litepod-16",
                        "chip_count": 16,
                        "marginal_wall_seconds": wall,
                        "chip_seconds": 16 * wall,
                        "chip_seconds_scope": "optimizer_loop",
                        "chip_seconds_excludes": ["compile", "evaluation"],
                        "training_source_revision": REVISION,
                    }
                )
    body = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "cells": cells,
        "costs": costs,
    }
    return {**body, "evidence_sha256": canonical_sha256(body)}


def _reseal(value: dict) -> None:
    body = {key: item for key, item in value.items() if key != "evidence_sha256"}
    value["evidence_sha256"] = canonical_sha256(body)


def _cell(
    value: dict,
    *,
    condition: str,
    panel: str = "sae",
    seed: int = 0,
    checkpoint: int = 2_000_000,
) -> dict:
    return next(
        row
        for row in value["cells"]
        if (
            row["panel"],
            row["seed"],
            row["checkpoint_tokens"],
            row["condition"],
        )
        == (panel, seed, checkpoint, condition)
    )


def _set_group_sst(value: dict, sst: list[float]) -> None:
    for condition in CONDITIONS:
        _cell(value, condition=condition)["component_sst"] = list(sst)


def test_report_computes_closure_contrasts_g_and_cost_compatibility() -> None:
    report = build_closure_report(_evidence(), bootstrap_replicates=200)
    assert REPORT_SCHEMA_VERSION.endswith("-v2")
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    row = report["panels"]["sae"]["milestones"]["10000000"]
    assert row["contrasts"]["A-R"]["estimate"] == pytest.approx(0.15)
    assert row["contrasts"]["F-B"]["estimate"] == pytest.approx(0.50)
    assert row["g"]["estimate"] == pytest.approx(60.0)
    assert row["g"]["status"] == "defined"
    assert report["panels"]["sae"]["cost_compatibility"][
        "a_over_f_all_seeds_comparable"
    ]
    assert "| 10,000,000 |" in render_markdown(report)


def test_report_seals_common_target_inventory_and_allows_receipt_provenance() -> None:
    value = _evidence()
    selected = _cell(
        value,
        panel="transcoder",
        seed=2,
        checkpoint=10_000_000,
        condition="F",
    )
    selected["centering_receipt_sha256"] = "e" * 64
    _reseal(value)

    report = build_closure_report(value, bootstrap_replicates=100)
    common = report["common_target"]
    assert common["shared_identity"] == {
        "evaluation_token_identity_sha256": "1" * 64,
        "evaluation_target_manifest_sha256": "2" * 64,
        "centering_target_manifest_sha256": "b" * 64,
        "target_mean_sha256": "4" * 64,
        "endpoint_receipt_sha256": "5" * 64,
        "endpoint_cache_key_sha256": "6" * 64,
        "routed_mean_sha256": "7" * 64,
        "oracle_shared_mean_sha256": "8" * 64,
        "recomposition_sidecar_sha256": "9" * 64,
        "evaluation_token_count": 4,
        "evaluation_packed_sequences": 4,
    }
    assert common["panel_source_identity"] == {
        "sae": {
            "evaluation_source_manifest_sha256": None,
            "source_manifest_sha256": None,
            "source_mean_sha256": None,
        },
        "transcoder": {
            "evaluation_source_manifest_sha256": "e" * 64,
            "source_manifest_sha256": "c" * 64,
            "source_mean_sha256": "d" * 64,
        },
    }
    assert len(common["representations"]) == 72
    inventory_row = next(
        row
        for row in common["representations"]
        if (
            row["panel"],
            row["seed"],
            row["checkpoint_tokens"],
            row["condition"],
        )
        == ("transcoder", 2, 10_000_000, "F")
    )
    assert inventory_row["evaluation_result_sha256"] == SHA
    assert inventory_row["centering_receipt_sha256"] == "e" * 64
    assert "Common target identity: all 72 cells share exactly" in render_markdown(
        report
    )


@pytest.mark.parametrize(
    "field",
    [
        "evaluation_token_identity_sha256",
        "target_mean_sha256",
        "recomposition_sidecar_sha256",
    ],
)
def test_report_rejects_global_target_identity_drift(field: str) -> None:
    value = _evidence()
    _cell(value, condition="R")[field] = "e" * 64
    _reseal(value)

    with pytest.raises(
        ComponentClosureReportError,
        match="one exact target identity",
    ):
        validate_score_grid(value)


def test_report_rejects_global_centering_target_manifest_drift() -> None:
    value = _evidence()
    _cell(value, condition="R")["centering_target_manifest_sha256"] = "e" * 64
    _reseal(value)

    with pytest.raises(
        ComponentClosureReportError,
        match="one exact target identity",
    ):
        validate_score_grid(value)


def test_report_rejects_global_evaluation_token_count_drift() -> None:
    value = _evidence()
    _cell(value, condition="R")["evaluation_token_count"] = 5
    _reseal(value)

    with pytest.raises(
        ComponentClosureReportError,
        match="one exact target identity",
    ):
        validate_score_grid(value)


@pytest.mark.parametrize(
    ("panel", "source_manifest", "source_mean", "message"),
    [
        ("sae", "c" * 64, "d" * 64, "SAE score cells"),
        ("transcoder", None, None, "transcoder score cells"),
    ],
)
def test_report_enforces_panel_source_identity_nullability(
    panel: str,
    source_manifest: str | None,
    source_mean: str | None,
    message: str,
) -> None:
    value = _evidence()
    selected = _cell(value, panel=panel, condition="R")
    selected["source_manifest_sha256"] = source_manifest
    selected["source_mean_sha256"] = source_mean
    _reseal(value)

    with pytest.raises(ComponentClosureReportError, match=message):
        validate_score_grid(value)


@pytest.mark.parametrize(
    "field",
    ("evaluation_source_manifest_sha256", "source_manifest_sha256"),
)
def test_report_rejects_panel_source_identity_drift(field: str) -> None:
    value = _evidence()
    _cell(
        value,
        panel="transcoder",
        condition="R",
    )[field] = "f" * 64
    _reseal(value)

    with pytest.raises(
        ComponentClosureReportError,
        match="do not share one source identity",
    ):
        validate_score_grid(value)


@pytest.mark.parametrize(
    ("reference", "candidate", "metric", "expected"),
    [
        (
            [25_000_000.0] * 4,
            [25_000_001.0, 25_000_000.0, 25_000_000.0, 25_000_000.0],
            "maximum_observed_aggregate_relative_difference",
            SST_AGGREGATE_RELATIVE_TOLERANCE,
        ),
        (
            [2_500_000.0] * 4,
            [2_500_000.5, 2_499_999.5, 2_500_000.0, 2_500_000.0],
            "maximum_observed_normalized_l1_difference",
            SST_NORMALIZED_L1_TOLERANCE,
        ),
        (
            [1_000_000.0, 100_000_000.0, 100_000_000.0, 100_000_000.0],
            [1_000_001.0, 99_999_999.0, 100_000_000.0, 100_000_000.0],
            "maximum_observed_per_sequence_relative_difference",
            SST_PER_SEQUENCE_RELATIVE_TOLERANCE,
        ),
    ],
    ids=(
        "aggregate-relative-boundary",
        "normalized-l1-boundary",
        "per-sequence-relative-boundary",
    ),
)
def test_report_accepts_common_sst_tolerance_boundaries(
    reference: list[float],
    candidate: list[float],
    metric: str,
    expected: float,
) -> None:
    value = _evidence()
    _set_group_sst(value, reference)
    _cell(value, condition="R")["component_sst"] = candidate
    _reseal(value)

    report = build_closure_report(value, bootstrap_replicates=100)
    assert report["common_sst"][metric] == pytest.approx(
        expected,
        rel=0,
        abs=1e-20,
    )


def test_report_accepts_bounded_sst_drift_and_uses_a_as_canonical() -> None:
    value = _evidence()
    raw_f_sst = [
        1.0 + 2e-7,
        2.0 - 1.5e-7,
        3.0,
        4.0,
    ]
    _cell(value, condition="F")["component_sst"] = raw_f_sst
    _reseal(value)

    checked = validate_score_grid(value)
    assert _cell(checked, condition="F")["component_sst"] == raw_f_sst
    report = build_closure_report(value, bootstrap_replicates=100)
    audit = report["common_sst"]
    assert audit["reference_condition"] == "A"
    assert audit["sequence_pairing"] == "exact_sequence_id_array_equality"
    assert "A_zero_and_unequal_is_infinite" in audit[
        "per_sequence_relative_zero_denominator_policy"
    ]
    assert audit[
        "maximum_observed_aggregate_relative_difference"
    ] == pytest.approx(5e-9)
    assert audit["maximum_observed_normalized_l1_difference"] == pytest.approx(
        3.5e-8
    )
    assert audit["maximum_observed_max_abs_difference"] == pytest.approx(2e-7)
    assert audit[
        "maximum_observed_per_sequence_relative_difference"
    ] == pytest.approx(2e-7)
    assert audit["maximum_observed_unequal_sequence_count"] == 2
    assert len(audit["representations"]) == 72
    representation = next(
        row
        for row in audit["representations"]
        if (
            row["panel"],
            row["seed"],
            row["checkpoint_tokens"],
            row["condition"],
        )
        == ("sae", 0, 2_000_000, "F")
    )
    raw_sst_sha256 = hashlib.sha256(
        np.asarray(raw_f_sst, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    assert representation["per_sequence_sst_raw_sha256"] == raw_sst_sha256
    raw_sequence_sha256 = hashlib.sha256(
        np.asarray([10, 11, 12, 13], dtype="<i8").tobytes(order="C")
    ).hexdigest()
    sequence_sha256 = canonical_sha256(
        {
            "domain": "gemma4-a512-component-closure-sequence-identity-v1",
            "encoding": "little_endian_int64_c_order",
            "length": 4,
            "raw_sha256": raw_sequence_sha256,
        }
    )
    assert representation["sequence_identity_sha256"] == sequence_sha256
    assert representation["evaluation_result_sha256"] == SHA
    assert representation["per_sequence_sst_length"] == 4
    assert representation["per_sequence_sst_identity_sha256"] == canonical_sha256(
        {
            "domain": "gemma4-a512-component-closure-sst-identity-v1",
            "role": {
                "panel": "sae",
                "seed": 0,
                "checkpoint_tokens": 2_000_000,
                "condition": "F",
            },
            "evaluation_result_sha256": SHA,
            "sequence_identity_sha256": sequence_sha256,
            "encoding": "little_endian_float64_c_order",
            "length": 4,
            "raw_sha256": raw_sst_sha256,
        }
    )
    assert representation["sst_total"] == pytest.approx(10.0 + 5e-8)
    assert representation["normalized_l1_difference"] == pytest.approx(3.5e-8)
    assert representation["unequal_sequence_count"] == 2
    a_representation = next(
        row
        for row in audit["representations"]
        if (
            row["panel"],
            row["seed"],
            row["checkpoint_tokens"],
            row["condition"],
        )
        == ("sae", 0, 2_000_000, "A")
    )
    assert a_representation["aggregate_relative_difference"] == 0.0
    assert a_representation["normalized_l1_difference"] == 0.0
    assert a_representation["max_abs_difference"] == 0.0
    assert a_representation["maximum_per_sequence_relative_difference"] == 0.0
    assert a_representation["unequal_sequence_count"] == 0
    assert (
        a_representation["canonical_a_per_sequence_sst_identity_sha256"]
        == a_representation["per_sequence_sst_identity_sha256"]
    )
    assert (
        representation["canonical_a_per_sequence_sst_identity_sha256"]
        == a_representation["per_sequence_sst_identity_sha256"]
    )
    assert audit["maximum_observed_aggregate_relative_difference"] == max(
        row["aggregate_relative_difference"] for row in audit["representations"]
    )
    assert audit["maximum_observed_normalized_l1_difference"] == max(
        row["normalized_l1_difference"] for row in audit["representations"]
    )
    assert audit["maximum_observed_max_abs_difference"] == max(
        row["max_abs_difference"] for row in audit["representations"]
    )
    assert audit["maximum_observed_per_sequence_relative_difference"] == max(
        row["maximum_per_sequence_relative_difference"]
        for row in audit["representations"]
    )
    assert audit["maximum_observed_unequal_sequence_count"] == max(
        row["unequal_sequence_count"] for row in audit["representations"]
    )
    endpoints = report["panels"]["sae"]["milestones"]["2000000"][
        "component_endpoint"
    ]
    assert endpoints["F"]["sst"] == endpoints["A"]["sst"] == 30.0
    assert endpoints["F"]["sst"] != pytest.approx(
        20.0 + representation["sst_total"],
        rel=0,
        abs=1e-10,
    )
    markdown = render_markdown(report)
    assert "condition A is canonical" in markdown
    assert "Maximum observed SST differences" in markdown
    assert "inventories all 72 condition representations" in markdown
    assert "per-sequence relative difference <= 1e-06" in markdown


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        (
            [25_000_000.0] * 4,
            [25_000_002.0, 25_000_000.0, 25_000_000.0, 25_000_000.0],
        ),
        (
            [2_500_000.0] * 4,
            [2_500_001.0, 2_499_999.0, 2_500_000.0, 2_500_000.0],
        ),
        (
            [1_000_000.0, 100_000_000.0, 100_000_000.0, 100_000_000.0],
            [1_000_002.0, 99_999_998.0, 100_000_000.0, 100_000_000.0],
        ),
    ],
    ids=(
        "aggregate-relative-over-limit",
        "normalized-l1-over-limit",
        "per-sequence-relative-over-limit",
    ),
)
def test_report_rejects_sst_drift_outside_envelope(
    reference: list[float],
    candidate: list[float],
) -> None:
    value = _evidence()
    _set_group_sst(value, reference)
    _cell(value, condition="R")["component_sst"] = candidate
    _reseal(value)

    with pytest.raises(
        ComponentClosureReportError,
        match="numerically common SST",
    ):
        validate_score_grid(value)


def test_report_still_requires_exact_sequence_pairing() -> None:
    value = _evidence()
    _cell(value, condition="R")["sequence_id"] = [10, 11, 13, 12]
    _reseal(value)

    with pytest.raises(
        ComponentClosureReportError,
        match="exact sequences",
    ):
        validate_score_grid(value)


@pytest.mark.parametrize(
    "sequence_id",
    [
        [10.0, 11, 12, 13],
        [True, 11, 12, 13],
        [2**63, 11, 12, 13],
        ["10", 11, 12, 13],
    ],
    ids=("integral-float", "bool", "int64-overflow", "string"),
)
def test_report_rejects_lossy_or_noninteger_sequence_ids(
    sequence_id: list,
) -> None:
    value = _evidence()
    _cell(value, condition="R")["sequence_id"] = sequence_id
    _reseal(value)

    with pytest.raises(
        ComponentClosureReportError,
        match="lossless integer vector",
    ):
        validate_score_grid(value)


@pytest.mark.parametrize(
    "result_sha256",
    ["A" * 64, "g" * 64, "a" * 63, True, None],
    ids=("uppercase", "nonhex", "short", "bool", "null"),
)
def test_report_rejects_invalid_evaluation_result_sha256(
    result_sha256: object,
) -> None:
    value = _evidence()
    _cell(value, condition="R")["evaluation_result_sha256"] = result_sha256
    _reseal(value)

    with pytest.raises(
        ComponentClosureReportError,
        match="evaluation_result_sha256 must be a lowercase SHA-256",
    ):
        validate_score_grid(value)


def test_per_sequence_relative_zero_denominator_policy_is_fail_closed() -> None:
    exact = _sst_equivalence_metrics(
        np.asarray([1.0, 0.0]),
        np.asarray([1.0, 0.0]),
    )
    assert exact["maximum_per_sequence_relative_difference"] == 0.0
    assert exact["unequal_sequence_count"] == 0

    with pytest.raises(ComponentClosureReportError, match="metrics must be finite"):
        _sst_equivalence_metrics(
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 1e-12]),
        )


@pytest.mark.parametrize(
    ("reference", "candidate", "message"),
    [
        ([1.0], [1.0, 2.0], "equal nonempty finite vectors"),
        ([[1.0]], [[1.0]], "equal nonempty finite vectors"),
        ([], [], "equal nonempty finite vectors"),
        ([float("inf")], [float("inf")], "equal nonempty finite vectors"),
        ([0.0, 0.0], [0.0, 0.0], "positive finite total and L1"),
        ([1.0, -1.0], [1.0, -1.0], "positive finite total and L1"),
        (
            [np.finfo(np.float64).max] * 2,
            [1.0, 1.0],
            "positive finite total and L1",
        ),
        (
            [1.0, 1.0],
            [np.finfo(np.float64).max] * 2,
            "candidate must have a finite total",
        ),
    ],
    ids=(
        "shape",
        "rank",
        "empty",
        "nonfinite",
        "zero-mass",
        "nonpositive-total",
        "reference-total-overflow",
        "candidate-total-overflow",
    ),
)
def test_sst_equivalence_metrics_independently_rejects_invalid_vectors(
    reference: list,
    candidate: list,
    message: str,
) -> None:
    with pytest.raises(ComponentClosureReportError, match=message):
        _sst_equivalence_metrics(
            np.asarray(reference, dtype=np.float64),
            np.asarray(candidate, dtype=np.float64),
        )
