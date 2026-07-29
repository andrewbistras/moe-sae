from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_routed_ladder import ENDPOINTS, RoutedLadderTokenScores
from moe_sae.experiment5_routed_ladder_v3_config import (
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
    hierarchical_v3_bootstrap,
    paired_v3_bootstrap,
    render_v3_primary_report_markdown,
    validate_v3_primary_report,
    write_v3_primary_report,
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
