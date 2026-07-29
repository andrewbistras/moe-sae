from __future__ import annotations

import copy
from pathlib import Path

import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_routed_ladder_campaign import (
    RoutedLadderCampaignPlanError,
)
from moe_sae.experiment5_routed_ladder_evaluation_worker import (
    CHECKPOINT_TOKENS,
    V3_BUNDLE_RESULT_SCHEMA_VERSION,
    V3_BUNDLE_SPEC_SCHEMA_VERSION,
    RoutedLadderEvaluationError,
    parse_resident_bundle_spec,
    seal_resident_bundle_result,
)
from moe_sae.experiment5_routed_ladder_v3_config import PROTOCOL_ID
from moe_sae.experiment5_routed_ladder_v3_evaluation import (
    BUNDLE_COUNT,
    CELL_COUNT,
    CONDITIONS,
    OPERATOR_MANIFEST_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    _expected_progress,
    validate_v3_primary_evaluation_plan,
)


def _worker_spec(tmp_path: Path) -> dict:
    cells = []
    for condition in CONDITIONS:
        for checkpoint in CHECKPOINT_TOKENS:
            imported = condition == "A512Softplus"
            cells.append(
                {
                    "condition": condition,
                    "checkpoint_tokens": checkpoint,
                    "training_source_revision": (
                        "b" * 40 if imported else "a" * 40
                    ),
                    "run_root": (
                        f"/remote/runs/{condition}/{checkpoint}"
                    ),
                    "checkpoint_payload_root": (
                        f"gs://bucket/checkpoints/{condition}"
                    ),
                    "result_root": str(
                        tmp_path / condition / str(checkpoint)
                    ),
                    "reference_kind": (
                        "completed" if imported else "trained_v3"
                    ),
                    "reference_config": (
                        "/remote/configs/ladder-v2.yaml"
                        if imported
                        else None
                    ),
                    "reference_run_contract_sha256": None,
                    "reference_run_contract_file_sha256": None,
                    "parent_training_result_sha256": "1" * 64,
                    "parent_training_result_file_sha256": "2" * 64,
                    "parent_run_contract_sha256": "3" * 64,
                    "parent_run_contract_file_sha256": "4" * 64,
                    "parent_checkpoint_pointer_sha256": "5" * 64,
                    "parent_checkpoint_pointer_file_sha256": "6" * 64,
                }
            )
    body = {
        "schema_version": V3_BUNDLE_SPEC_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "panel": "sae",
        "seed": 0,
        "source_revision": "e" * 40,
        "config": "/remote/configs/ladder-v3.yaml",
        "bundle_result_root": str(tmp_path / "bundle"),
        "operator_manifest_sha256": "7" * 64,
        "cells": cells,
    }
    return {**body, "bundle_spec_sha256": canonical_sha256(body)}


def test_v3_resident_bundle_accepts_only_exact_primary_pair(
    tmp_path: Path,
) -> None:
    bundle = parse_resident_bundle_spec(_worker_spec(tmp_path))
    assert bundle.protocol_id == PROTOCOL_ID
    assert len(bundle.cells) == 6
    assert [
        (row.condition, row.checkpoint_tokens) for row in bundle.cells
    ] == [
        (condition, checkpoint)
        for condition in CONDITIONS
        for checkpoint in CHECKPOINT_TOKENS
    ]
    assert {
        row.reference_config
        for row in bundle.cells
        if row.reference_kind == "completed"
    } == {"/remote/configs/ladder-v2.yaml"}
    assert {
        row.training_source_revision
        for row in bundle.cells
        if row.reference_kind == "trained_v3"
    } == {"a" * 40}
    assert bundle.source_revision == "e" * 40


def test_v3_milestone_progress_reads_the_real_worker_plan_shape() -> None:
    result = {
        "run_identity": {
            "plan": {
                "global_microbatch_tokens": 4096,
                "microbatches_per_update": 4,
                "planned_updates": 611,
                "total_tokens": 10_000_000,
            }
        }
    }

    assert _expected_progress(result, 2_000_000) == (123, 2_015_232)
    assert _expected_progress(result, 10_000_000) == (611, 10_000_000)


def test_v3_import_requires_its_v2_config(tmp_path: Path) -> None:
    spec = _worker_spec(tmp_path)
    spec["cells"][3]["reference_config"] = None
    unsigned = {
        key: spec[key] for key in spec if key != "bundle_spec_sha256"
    }
    spec["bundle_spec_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(RoutedLadderEvaluationError, match="absolute path"):
        parse_resident_bundle_spec(spec)


def test_v3_bundle_result_uses_v3_envelope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = parse_resident_bundle_spec(_worker_spec(tmp_path))

    def completion(_bundle, cell):
        return {
            "condition": cell.condition,
            "checkpoint_tokens": cell.checkpoint_tokens,
        }

    monkeypatch.setattr(
        "moe_sae.experiment5_routed_ladder_evaluation_worker."
        "_resident_cell_completion",
        completion,
    )
    rows = [completion(bundle, cell) for cell in bundle.cells]
    result = seal_resident_bundle_result(
        bundle,
        rows,
        timing={
            "accelerator_type": "v4-32",
            "chip_count": 16,
            "successful_attempt_wall_seconds": 2.0,
            "successful_attempt_chip_seconds": 32.0,
            "timing_scope": (
                "successful_resident_worker_after_cli_parse_through_"
                "cell_completion"
            ),
        },
    )
    assert result["protocol_id"] == PROTOCOL_ID
    assert result["schema_version"] == V3_BUNDLE_RESULT_SCHEMA_VERSION


def _evaluation_plan(tmp_path: Path) -> dict:
    bundles = []
    for panel in ("sae", "transcoder"):
        for seed in (0, 1, 2):
            cells = []
            for condition in CONDITIONS:
                for checkpoint in CHECKPOINT_TOKENS:
                    cells.append(
                        {
                            "condition": condition,
                            "checkpoint_tokens": checkpoint,
                        }
                    )
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


def test_v3_primary_plan_is_exact_six_bundle_36_cell_grid(
    tmp_path: Path,
) -> None:
    plan = validate_v3_primary_evaluation_plan(
        _evaluation_plan(tmp_path)
    )
    assert len(plan["bundles"]) == BUNDLE_COUNT
    assert sum(
        len(row["operator_manifest"]["cells"])
        for row in plan["bundles"]
    ) == CELL_COUNT
    assert plan["evaluation_source_revision"] != (
        plan["training_source_revision"]
    )


def test_v3_primary_plan_rejects_shared_owner(tmp_path: Path) -> None:
    plan = _evaluation_plan(tmp_path)
    plan["bundles"][1]["resource_id"] = plan["bundles"][0]["resource_id"]
    unsigned = {key: plan[key] for key in plan if key != "plan_sha256"}
    plan["plan_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="matrix or owners",
    ):
        validate_v3_primary_evaluation_plan(copy.deepcopy(plan))


def test_v3_primary_plan_rejects_training_commit_as_eval_commit(
    tmp_path: Path,
) -> None:
    plan = _evaluation_plan(tmp_path)
    plan["evaluation_source_revision"] = plan["training_source_revision"]
    for bundle in plan["bundles"]:
        manifest = bundle["operator_manifest"]
        manifest["source_revision"] = plan["training_source_revision"]
        unsigned_manifest = {
            key: manifest[key]
            for key in manifest
            if key != "manifest_sha256"
        }
        manifest["manifest_sha256"] = canonical_sha256(unsigned_manifest)
    unsigned = {key: plan[key] for key in plan if key != "plan_sha256"}
    plan["plan_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="identity changed",
    ):
        validate_v3_primary_evaluation_plan(plan)
