from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

import moe_sae.experiment5_routed_ladder_v3_report as report_module
from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.direct_horizon_campaign import (
    PROTOCOL_ID as EXPERIMENT4_PROTOCOL_ID,
)
from moe_sae.experiment5_routed_ladder import ENDPOINTS, RoutedLadderTokenScores
from moe_sae.experiment5_routed_ladder_v3_config import (
    A256_NATIVE_CONFIG_SHA256,
    CHECKPOINT_TOKENS,
    CONDITIONS,
    PANELS,
    PROTOCOL_ID,
    SEEDS,
)
from moe_sae.experiment5_routed_ladder_v3_evaluation import (
    OPERATOR_MANIFEST_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
)
from moe_sae.experiment5_routed_ladder_v3_report import (
    BOOTSTRAP_REPLICATES,
    BundleTimingEvidence,
    PrimaryCellEvidence,
    RoutedLadderV3ReportError,
    TrainingCostEvidence,
    build_v3_primary_report_from_evidence,
    collect_a256_reference_context,
    hierarchical_v3_bootstrap,
    paired_v3_bootstrap,
    render_v3_primary_report_markdown,
    validate_v3_primary_report,
    verify_v3_primary_report_from_paths,
    write_v3_primary_report,
)
from moe_sae.experiment5_routed_ladder_v3_provenance import (
    A256_PROVENANCE_ADDENDUM_ID,
    A256_PROVENANCE_ADDENDUM_SCHEMA_VERSION,
)


def _evaluation_plan(tmp_path: Path) -> dict:
    bundles = []
    for panel in PANELS:
        for seed in SEEDS:
            cells = [
                {
                    "condition": condition,
                    "checkpoint_tokens": checkpoint,
                }
                for condition in CONDITIONS
                for checkpoint in CHECKPOINT_TOKENS
            ]
            manifest_body = {
                "schema_version": OPERATOR_MANIFEST_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "panel": panel,
                "seed": seed,
                "source_revision": "e" * 40,
                "config_remote": f"/remote/{panel}-v3.yaml",
                "bundle_result_root_remote": (
                    f"/remote/results/{panel}/seed{seed}/bundle"
                ),
                "bundle_result_root_controller": str(
                    tmp_path / panel / f"seed{seed}" / "bundle"
                ),
                "cells": cells,
            }
            manifest = {
                **manifest_body,
                "manifest_sha256": canonical_sha256(manifest_body),
            }
            bundles.append(
                {
                    "task_id": f"eval-primary-{panel}-seed{seed}",
                    "panel": panel,
                    "seed": seed,
                    "resource_id": f"owner-{panel}-{seed}",
                    "operator_manifest": manifest,
                }
            )
    body = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": "v3-primary",
        "evaluation_source_revision": "e" * 40,
        "training_source_revision": "a" * 40,
        "state_root": str(tmp_path),
        "training_plan_sha256": "1" * 64,
        "v2_source_plan_sha256": "2" * 64,
        "completed_trajectory_manifest_sha256": "3" * 64,
        "bundles": bundles,
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def _a256_addendum(tmp_path: Path) -> dict:
    panels = []
    for panel_index, panel in enumerate(PANELS):
        is_sae = panel == "sae"
        declared = A256_NATIVE_CONFIG_SHA256[panel]
        observed = declared if is_sae else "8" * 64
        raw_config_sha = f"{panel_index + 1:x}" * 64
        checkpoints = []
        for checkpoint_index, checkpoint in enumerate(CHECKPOINT_TOKENS):
            pointer_index = (
                3 + panel_index * len(CHECKPOINT_TOKENS) + checkpoint_index
            )
            checkpoints.append(
                {
                    "checkpoint_tokens": checkpoint,
                    "checkpoint_pointer": {
                        "controller_path": str(
                            (
                                tmp_path
                                / panel
                                / "checkpoints"
                                / f"{checkpoint}.json"
                            ).resolve()
                        ),
                        "remote_path": (
                            f"/remote/{panel}/checkpoints/{checkpoint}.json"
                        ),
                        "sha256": f"{pointer_index:x}" * 64,
                    },
                    "checkpoint_pointer_sha256": (
                        f"{pointer_index + 6:x}" * 64
                    ),
                }
            )
        marginal = 100.0 + panel_index
        panels.append(
            {
                "panel": panel,
                "target_condition": "A256Native",
                "source_condition": "A",
                "source_label": (
                    "A256ReLU" if is_sae else "A256Softplus"
                ),
                "seed": 0,
                "encoder_activation": "relu" if is_sae else "softplus",
                "selection_policy": (
                    "strict_positive_topk"
                    if is_sae
                    else "exact_positive_topk"
                ),
                "declared_source_config_semantic_sha256": declared,
                "observed_source_config_semantic_sha256": observed,
                "observed_source_config_file_sha256": raw_config_sha,
                "declaration_status": (
                    "exact_match" if is_sae else "mismatch"
                ),
                "disposition": (
                    "contextual_reference_verified"
                    if is_sae
                    else (
                        "contextual_reference_corrected_by_observed_lineage"
                    )
                ),
                "training_source_revision": "a" * 40,
                "source_config": {
                    "controller_path": str(
                        (tmp_path / panel / "config.yaml").resolve()
                    ),
                    "remote_path": f"/remote/{panel}/config.yaml",
                    "sha256": raw_config_sha,
                },
                "run_root_remote": f"/remote/{panel}/runs/a",
                "checkpoint_payload_root_gcs": f"gs://bucket/{panel}/a",
                "run_contract": {
                    "controller_path": str(
                        (tmp_path / panel / "run-contract.json").resolve()
                    ),
                    "remote_path": f"/remote/{panel}/run-contract.json",
                    "sha256": "b" * 64,
                },
                "run_contract_sha256": "c" * 64,
                "training_result": {
                    "controller_path": str(
                        (tmp_path / panel / "training-result.json").resolve()
                    ),
                    "remote_path": f"/remote/{panel}/training-result.json",
                    "sha256": "d" * 64,
                },
                "training_result_sha256": "e" * 64,
                "cost": {
                    "resource_id": f"slice-{panel}",
                    "accelerator_type": "v4-32",
                    "chip_count": 16,
                    "marginal_wall_seconds": marginal,
                    "chip_seconds": 16 * marginal,
                    "chip_seconds_scope": "optimizer_training",
                    "chip_seconds_excludes": [
                        "compile",
                        "checkpoint_serialization",
                    ],
                },
                "checkpoints": checkpoints,
            }
        )
    scope = {
        "condition": "A256Native",
        "evidentiary_role": "panel_native_descriptive_context_only",
        "included_in_primary_a512_inference": False,
        "included_in_primary_a512_estimand": False,
        "cross_panel_pooling_allowed": False,
        "across_seed_inference_allowed": False,
        "allowed_use": "within_panel_historical_scale_context",
        "prohibited_uses": [
            "matched_a512_activation_policy_contrast",
            "cross_panel_activation_policy_contrast",
            "three_seed_inference",
            "replacement_for_any_a512_cell",
        ],
    }
    body = {
        "schema_version": A256_PROVENANCE_ADDENDUM_SCHEMA_VERSION,
        "addendum_id": A256_PROVENANCE_ADDENDUM_ID,
        "source_protocol_id": EXPERIMENT4_PROTOCOL_ID,
        "target_protocol_id": PROTOCOL_ID,
        "source_plan_sha256": "f" * 64,
        "scientific_scope": scope,
        "panels": panels,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _scores(
    *,
    panel: str,
    condition: str,
    seed: int,
    checkpoint: int,
) -> RoutedLadderTokenScores:
    sequence = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    position = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.uint16)
    endpoint_sst = {}
    endpoint_sse = {}
    checkpoint_gain = checkpoint / 10_000_000 * 0.01
    seed_gain = seed * 0.001
    panel_gain = 0.002 if panel == "transcoder" else 0.0
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        sst = np.asarray(
            [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            dtype=np.float64,
        ) * (endpoint_index + 1)
        relu_nmse = (
            np.asarray([0.72, 0.68, 0.70, 0.66, 0.71, 0.67])
            - checkpoint_gain
            + seed_gain
            + panel_gain
        )
        relu_sse = sst * relu_nmse
        delta = np.asarray([0.02, 0.01, 0.03, 0.015, 0.025, 0.012])
        endpoint_sst[endpoint] = sst
        endpoint_sse[endpoint] = (
            relu_sse
            if condition == "A512ReLUInclusiveTopK"
            else relu_sse - sst * delta
        )
    return RoutedLadderTokenScores(
        condition=condition,
        panel=panel,
        seed=seed,
        checkpoint_tokens=checkpoint,
        source_revision="e" * 40,
        capture_manifest_sha256="4" * 64,
        recomposition_sidecar_sha256="5" * 64,
        sequence_id=sequence,
        position=position,
        endpoint_sse=endpoint_sse,
        endpoint_sst=endpoint_sst,
    )


def _evidence(
    tmp_path: Path,
) -> tuple[
    list[PrimaryCellEvidence],
    list[BundleTimingEvidence],
    list[TrainingCostEvidence],
]:
    timings = []
    timing_by_id = {}
    for panel in PANELS:
        for seed in SEEDS:
            timing = BundleTimingEvidence(
                panel=panel,
                seed=seed,
                accelerator_type="v4-32",
                chip_count=16,
                successful_attempt_wall_seconds=100.0 + seed,
                successful_attempt_chip_seconds=16 * (100.0 + seed),
                timing_scope=(
                    "successful_resident_worker_after_cli_parse_through_"
                    "cell_completion"
                ),
                bundle_result_sha256=f"{seed + 6:x}" * 64,
                bundle_result_file_sha256=f"{seed + 9:x}" * 64,
            )
            timings.append(timing)
            timing_by_id[(panel, seed)] = timing
    cells = []
    for panel in PANELS:
        for checkpoint in CHECKPOINT_TOKENS:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    scores = _scores(
                        panel=panel,
                        condition=condition,
                        seed=seed,
                        checkpoint=checkpoint,
                    )
                    positive_l0 = (
                        30.0
                        if condition == "A512ReLUInclusiveTopK"
                        else 32.0
                    )
                    result = {
                        "training_source_revision": "a" * 40,
                        "parent_training_result_sha256": "a" * 64,
                        "parent_training_result_file_sha256": "b" * 64,
                        "parent_run_contract_sha256": "c" * 64,
                        "parent_run_contract_file_sha256": "d" * 64,
                        "parent_checkpoint_pointer_sha256": "e" * 64,
                        "parent_checkpoint_pointer_file_sha256": "f" * 64,
                        "result_sha256": "0" * 64,
                        "score_artifact_sha256": "1" * 64,
                        "score_data_sha256": "2" * 64,
                        "score_data_bytes": 100,
                        "evaluation_wall_seconds": 5.0 + seed,
                        "tokens": scores.tokens,
                        "packed_sequences": scores.packed_sequences,
                        "token_identity_sha256": (
                            scores.token_identity_sha256
                        ),
                        "selected_slot_count": 32,
                        "positive_l0_sum": int(
                            positive_l0 * scores.tokens
                        ),
                        "positive_l0": positive_l0,
                        "zero_selected_slot_fraction": (
                            1.0 - positive_l0 / 32.0
                        ),
                        "positive_l0_contract": (
                            "bounded_zero_to_selected_slots"
                            if condition == "A512ReLUInclusiveTopK"
                            else "exact_selected_slots"
                        ),
                        "dead_rate": 0.25,
                        "dead_rate_scope": "heldout_all_features",
                        "total_features": 65_536,
                    }
                    cells.append(
                        PrimaryCellEvidence(
                            panel=panel,
                            condition=condition,
                            seed=seed,
                            checkpoint_tokens=checkpoint,
                            origin=(
                                "fresh_v3"
                                if condition
                                == "A512ReLUInclusiveTopK"
                                else "imported_v2_exact_graph"
                            ),
                            result_path=str(
                                tmp_path
                                / panel
                                / condition
                                / str(seed)
                                / str(checkpoint)
                            ),
                            result_file_sha256="3" * 64,
                            result=result,
                            scores=scores,
                            bundle_timing=timing_by_id[(panel, seed)],
                        )
                    )
    costs = []
    for panel in PANELS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                wall = (
                    100.0
                    + seed
                    + (
                        5.0
                        if condition == "A512Softplus"
                        else 0.0
                    )
                )
                costs.append(
                    TrainingCostEvidence(
                        panel=panel,
                        condition=condition,
                        seed=seed,
                        origin=(
                            "fresh_v3"
                            if condition == "A512ReLUInclusiveTopK"
                            else "imported_v2_exact_graph"
                        ),
                        resource_id=f"owner-{panel}-{seed}",
                        accelerator_type="v4-32",
                        chip_count=16,
                        tokens=10_000_000,
                        marginal_wall_seconds=wall,
                        chip_seconds=16 * wall,
                        tokens_per_second=10_000_000 / wall,
                        chip_seconds_scope="training",
                        chip_seconds_excludes=(
                            "compile",
                            "lowering",
                            "checkpoint serialization",
                        ),
                        compile_seconds=1.0,
                        lowering_seconds=1.0,
                        preprocessing_wall_seconds=2.0,
                        optimizer_wall_seconds=wall - 2.0,
                        checkpoint_seconds=1.0,
                        training_source_revision="a" * 40,
                        training_result_sha256="6" * 64,
                        training_result_file_sha256="7" * 64,
                    )
                )
    return cells, timings, costs


def test_v3_bootstraps_are_deterministic_and_require_three_paired_seeds() -> None:
    pairs = [
        (
            _scores(
                panel="sae",
                condition="A512ReLUInclusiveTopK",
                seed=seed,
                checkpoint=10_000_000,
            ),
            _scores(
                panel="sae",
                condition="A512Softplus",
                seed=seed,
                checkpoint=10_000_000,
            ),
        )
        for seed in SEEDS
    ]
    first = paired_v3_bootstrap(*pairs[0], endpoint="component")
    second = paired_v3_bootstrap(*pairs[0], endpoint="component")
    assert first == second
    assert first["bootstrap_replicates"] == BOOTSTRAP_REPLICATES
    assert first["estimate_ev_difference"] > 0

    hierarchical = hierarchical_v3_bootstrap(
        pairs,
        endpoint="component",
    )
    assert hierarchical == hierarchical_v3_bootstrap(
        pairs,
        endpoint="component",
    )
    assert hierarchical["positive_seed_count"] == 3
    with pytest.raises(
        RoutedLadderV3ReportError,
        match="exactly paired seeds",
    ):
        hierarchical_v3_bootstrap(pairs[:2], endpoint="component")


def test_crossed_bootstrap_preserves_common_sequence_covariance() -> None:
    sequence = np.arange(60, dtype=np.int64)
    position = np.zeros(60, dtype=np.uint16)
    sst = np.ones(60, dtype=np.float64)
    sequence_effect = np.tile(
        np.asarray([-0.15, 0.35], dtype=np.float64),
        30,
    )

    def scores(condition: str, seed: int) -> RoutedLadderTokenScores:
        relu_sse = np.full(60, 0.8, dtype=np.float64)
        selected_sse = (
            relu_sse
            if condition == "A512ReLUInclusiveTopK"
            else relu_sse - sequence_effect
        )
        return RoutedLadderTokenScores(
            condition=condition,
            panel="sae",
            seed=seed,
            checkpoint_tokens=10_000_000,
            source_revision="e" * 40,
            capture_manifest_sha256="4" * 64,
            recomposition_sidecar_sha256="5" * 64,
            sequence_id=sequence,
            position=position,
            endpoint_sse={
                endpoint: selected_sse for endpoint in ENDPOINTS
            },
            endpoint_sst={endpoint: sst for endpoint in ENDPOINTS},
        )

    pairs = [
        (
            scores("A512ReLUInclusiveTopK", seed),
            scores("A512Softplus", seed),
        )
        for seed in SEEDS
    ]
    per_seed = paired_v3_bootstrap(*pairs[0], endpoint="component")
    crossed = hierarchical_v3_bootstrap(pairs, endpoint="component")
    per_seed_width = per_seed["ci95_upper"] - per_seed["ci95_lower"]
    crossed_width = crossed["ci95_upper"] - crossed["ci95_lower"]

    # With an identical sequence-level effect in all three seeds, a crossed
    # bootstrap retains the per-seed corpus uncertainty. Independently
    # resampling sequences within each seed would shrink this by about sqrt(3).
    assert 0.85 < crossed_width / per_seed_width < 1.15
    assert (
        crossed["bootstrap_unit"]
        == "crossed_seed_and_packed_sequence"
    )


def test_v3_report_is_complete_self_hashed_and_create_only(
    tmp_path: Path,
) -> None:
    cells, timings, costs = _evidence(tmp_path)
    report = build_v3_primary_report_from_evidence(
        _evaluation_plan(tmp_path),
        cells,
        timings,
        costs,
    )
    assert len(report["cells"]) == 36
    assert len(report["contrasts"]) == 18
    assert report["primary_estimand"]["matched_quantities"][
        "exact_positive_l0"
    ] is False
    assert report["optional_context"]["H"]["status"] == "absent_nonblocking"
    assert validate_v3_primary_report(report) == report

    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    receipt = write_v3_primary_report(
        json_path=json_path,
        markdown_path=markdown_path,
        report=report,
    )
    assert receipt["report_sha256"] == report["report_sha256"]
    assert markdown_path.read_text() == render_v3_primary_report_markdown(
        report
    )
    write_v3_primary_report(
        json_path=json_path,
        markdown_path=markdown_path,
        report=report,
    )
    markdown_path.write_text("changed")
    with pytest.raises(RoutedLadderV3ReportError, match="replace immutable"):
        write_v3_primary_report(
            json_path=json_path,
            markdown_path=markdown_path,
            report=report,
        )


def test_v3_report_rejects_false_fixed_positive_l0(
    tmp_path: Path,
) -> None:
    cells, timings, costs = _evidence(tmp_path)
    report = build_v3_primary_report_from_evidence(
        _evaluation_plan(tmp_path),
        cells,
        timings,
        costs,
    )
    changed = copy.deepcopy(report)
    changed["primary_estimand"]["matched_quantities"][
        "exact_positive_l0"
    ] = True
    unsigned = {
        key: value
        for key, value in changed.items()
        if key != "report_sha256"
    }
    changed["report_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(RoutedLadderV3ReportError, match="fixed-L0"):
        validate_v3_primary_report(changed)


def test_a256_addendum_attaches_only_read_only_checkpoint_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells, timings, costs = _evidence(tmp_path)
    evaluation = _evaluation_plan(tmp_path)
    baseline = build_v3_primary_report_from_evidence(
        evaluation,
        cells,
        timings,
        costs,
    )
    validated_panels = []
    monkeypatch.setattr(
        report_module,
        "validate_a256_addendum_panel",
        validated_panels.append,
    )

    report = build_v3_primary_report_from_evidence(
        evaluation,
        cells,
        timings,
        costs,
        a256_provenance_addendum=_a256_addendum(tmp_path),
    )

    context = report["optional_context"]["A256Native"]
    assert context["status"] == "available_read_only_checkpoint_context"
    assert context["panel_reference_count"] == 2
    assert context["checkpoint_reference_count"] == 6
    assert context["a256_scores_in_primary_inference"] is False
    assert context["primary_verdict_blocked"] is False
    assert [panel.panel for panel in validated_panels] == [
        "sae",
        "transcoder",
    ]
    assert {
        row["panel"]: row["declaration_status"]
        for row in context["panel_references"]
    } == {"sae": "exact_match", "transcoder": "mismatch"}
    for field in (
        "primary_estimand",
        "cells",
        "contrasts",
        "primary_verdict",
        "training_cost_comparison",
    ):
        assert report[field] == baseline[field]
    assert "No A256 score row enters" in render_v3_primary_report_markdown(
        report
    )
    assert validate_v3_primary_report(report) == report


def test_a256_context_rejects_scope_expansion_after_sealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells, timings, costs = _evidence(tmp_path)
    monkeypatch.setattr(
        report_module,
        "validate_a256_addendum_panel",
        lambda _panel: None,
    )
    report = build_v3_primary_report_from_evidence(
        _evaluation_plan(tmp_path),
        cells,
        timings,
        costs,
        a256_provenance_addendum=_a256_addendum(tmp_path),
    )
    changed = copy.deepcopy(report)
    changed["optional_context"]["A256Native"]["scientific_scope"][
        "included_in_primary_a512_inference"
    ] = True
    changed["markdown_sha256"] = hashlib.sha256(
        report_module._render_v3_primary_markdown_core(changed).encode(
            "utf-8"
        )
    ).hexdigest()
    unsigned = {
        key: value
        for key, value in changed.items()
        if key != "report_sha256"
    }
    changed["report_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(
        RoutedLadderV3ReportError,
        match="identity or scope changed",
    ):
        validate_v3_primary_report(changed)


def test_a256_collector_rejects_changed_addendum_self_hash(
    tmp_path: Path,
) -> None:
    addendum = _a256_addendum(tmp_path)
    addendum["panels"][1][
        "observed_source_config_semantic_sha256"
    ] = "9" * 64

    with pytest.raises(
        RoutedLadderV3ReportError,
        match="provenance addendum is invalid",
    ):
        collect_a256_reference_context(addendum)


def test_cross_accelerator_costs_remain_raw_and_noncomparable(
    tmp_path: Path,
) -> None:
    cells, timings, costs = _evidence(tmp_path)
    mixed_costs = [
        (
            dataclasses.replace(
                row,
                accelerator_type="v6e-16",
                resource_id=f"v6e-{row.panel}-{row.seed}",
            )
            if row.condition == "A512Softplus"
            else row
        )
        for row in costs
    ]
    report = build_v3_primary_report_from_evidence(
        _evaluation_plan(tmp_path),
        cells,
        timings,
        mixed_costs,
    )

    for row in report["training_cost_comparison"][
        "paired_seed_comparisons"
    ]:
        assert (
            row["comparison_status"]
            == "non_comparable_accelerator_type_mismatch"
        )
        assert row["softplus_minus_relu_chip_seconds"] is None
        assert row["softplus_to_relu_chip_seconds_ratio_percent"] is None
        assert row["relu_to_softplus_chip_seconds_ratio_percent"] is None
    assert "n/a" in render_v3_primary_report_markdown(report)
    assert validate_v3_primary_report(report) == report


def test_training_cost_rejects_chip_count_accelerator_mismatch(
    tmp_path: Path,
) -> None:
    _cells, _timings, costs = _evidence(tmp_path)
    with pytest.raises(
        RoutedLadderV3ReportError,
        match="differs from accelerator type",
    ):
        dataclasses.replace(
            costs[0],
            chip_count=8,
            chip_seconds=8 * costs[0].marginal_wall_seconds,
        )


def test_integrity_validate_is_non_authoritative_for_fabricated_ci(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cells, timings, costs = _evidence(tmp_path)
    expected = build_v3_primary_report_from_evidence(
        _evaluation_plan(tmp_path),
        cells,
        timings,
        costs,
    )
    fabricated = copy.deepcopy(expected)
    contrast = next(
        row
        for row in fabricated["contrasts"]
        if row["panel"] == "sae"
        and row["checkpoint_tokens"] == 10_000_000
        and row["endpoint"] == "component"
    )
    contrast["across_seed"]["ci95_lower"] = -0.2
    contrast["across_seed"]["ci95_upper"] = -0.1
    contrast["verdict"] = "relu_inclusive_topk_better"
    primary = fabricated["primary_verdict"]["panels"][0]
    primary["ci95_lower"] = -0.2
    primary["ci95_upper"] = -0.1
    primary["verdict"] = "relu_inclusive_topk_better"
    fabricated.pop("report_sha256")
    fabricated["markdown_sha256"] = hashlib.sha256(
        report_module._render_v3_primary_markdown_core(fabricated).encode(
            "utf-8"
        )
    ).hexdigest()
    fabricated["report_sha256"] = canonical_sha256(fabricated)

    # A self-contained integrity check cannot authenticate bootstrap
    # quantiles, even though it can still verify their internal wiring.
    assert validate_v3_primary_report(fabricated) == fabricated
    report_path = tmp_path / "fabricated.json"
    markdown_path = tmp_path / "fabricated.md"
    report_path.write_bytes(report_module._json_payload(fabricated))
    markdown_path.write_text(
        render_v3_primary_report_markdown(fabricated),
        encoding="utf-8",
    )
    cli = runpy.run_path(
        "scripts/tpu/seal_experiment5_routed_ladder_v3_report.py"
    )
    assert (
        cli["main"](
            [
                "validate",
                "--report",
                str(report_path),
                "--markdown",
                str(markdown_path),
            ]
        )
        == 0
    )
    integrity_receipt = json.loads(capsys.readouterr().out)
    assert integrity_receipt == {
        **{
            "status": (
                "primary_report_integrity_valid_non_authoritative"
            ),
            "authoritative": False,
            "validation_scope": (
                "self_hash_internal_arithmetic_and_markdown_only"
            ),
            "authoritative_command": "verify",
            "report_sha256": fabricated["report_sha256"],
        },
        "json_file_sha256": hashlib.sha256(
            report_path.read_bytes()
        ).hexdigest(),
        "markdown_file_sha256": hashlib.sha256(
            markdown_path.read_bytes()
        ).hexdigest(),
    }

    monkeypatch.setattr(
        report_module,
        "build_v3_primary_report_from_paths",
        lambda **_kwargs: expected,
    )
    with pytest.raises(
        RoutedLadderV3ReportError,
        match="differs from recomputed immutable evidence",
    ):
        verify_v3_primary_report_from_paths(
            report_path=report_path,
            markdown_path=markdown_path,
            evaluation_plan_path=tmp_path / "evaluation.yaml",
            training_plan_path=tmp_path / "training.yaml",
            completed_manifest_path=tmp_path / "softplus.yaml",
        )
