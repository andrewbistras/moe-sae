from __future__ import annotations

import copy
from pathlib import Path

import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.direct_horizon_campaign import PROTOCOL_ID as EXPERIMENT4_PROTOCOL_ID
from moe_sae.experiment5_routed_ladder_v3_a256_scores import (
    A256_CONTEXT_SCORE_MANIFEST_ID,
    A256_CONTEXT_SCORE_MANIFEST_SCHEMA_VERSION,
    A256ContextScoreError,
    parse_a256_context_score_manifest,
    write_a256_context_score_manifest,
)
from moe_sae.experiment5_routed_ladder_v3_config import PROTOCOL_ID

MILESTONES = (2_000_000, 5_000_000, 10_000_000)


def _scope() -> dict:
    return {
        "condition": "A256Native",
        "evidentiary_role": "panel_native_descriptive_score_context_only",
        "included_in_primary_a512_inference": False,
        "included_in_primary_a512_estimand": False,
        "cross_panel_pooling_allowed": False,
        "across_seed_inference_allowed": False,
        "cross_panel_activation_contrast_allowed": False,
        "allowed_use": "within_panel_historical_learning_curve_context",
        "prohibited_uses": [
            "matched_a512_activation_policy_contrast",
            "cross_panel_activation_policy_contrast",
            "three_seed_inference",
            "pooled_a256_estimate",
            "replacement_for_any_a512_cell",
        ],
    }


def _token_policy() -> str:
    return (
        "The target capture SHA-256 and retained token count are common "
        "across both panels. Ordered token and SST-content hashes are "
        "panel-local representations because the historical evaluators "
        "serialized and aggregated the same capture in different row "
        "orders; no cross-panel byte-hash equality or pooling is asserted."
    )


def _parents(panel: str, milestone_index: int) -> dict:
    representation = (
        {
            "ordered_token_identity_sha256": "1" * 64,
            "component_sst_content_sha256": "2" * 64,
            "routed_sst_content_sha256": "3" * 64,
        }
        if panel == "sae"
        else {
            "ordered_token_identity_sha256": "4" * 64,
            "component_sst_content_sha256": "5" * 64,
            "routed_sst_content_sha256": "6" * 64,
        }
    )
    is_sae = panel == "sae"
    return {
        "target_capture_manifest_sha256": "a" * 64,
        "source_capture_manifest_sha256": None if is_sae else "b" * 64,
        "training_target_manifest_sha256": "c" * 64,
        "training_source_manifest_sha256": None if is_sae else "d" * 64,
        "recomposition_sidecar_sha256": None if is_sae else "e" * 64,
        "run_contract_sha256": "7" * 64,
        "run_contract_file_sha256": "8" * 64,
        "training_result_sha256": "9" * 64,
        "training_result_file_sha256": "0" * 64,
        "checkpoint_pointer_sha256": (
            f"{milestone_index + 1:x}" * 64
        ),
        "checkpoint_pointer_file_sha256": (
            f"{milestone_index + 4:x}" * 64
        ),
        "evaluation_receipt_sha256": "b" * 64,
        "common_sst_receipt_sha256": "c" * 64,
        "common_sst_receipt_file_sha256": None if is_sae else "d" * 64,
        "score_artifact_sha256": "e" * 64,
        **representation,
    }


def _manifest(tmp_path: Path) -> dict:
    cells = []
    for panel_index, panel in enumerate(("sae", "transcoder")):
        is_sae = panel == "sae"
        for milestone_index, milestone in enumerate(MILESTONES):
            nmse_component = 0.7 - 0.05 * milestone_index
            nmse_routed = 1.2 - 0.05 * milestone_index
            cells.append(
                {
                    "panel": panel,
                    "condition": "A256Native",
                    "source_condition": "A",
                    "source_label": (
                        "A256ReLU" if is_sae else "A256Softplus"
                    ),
                    "seed": 0,
                    "checkpoint_tokens": milestone,
                    "checkpoint_actual_tokens": milestone + (
                        15_232 if milestone < 10_000_000 else 0
                    ),
                    "encoder_activation": (
                        "relu" if is_sae else "softplus"
                    ),
                    "selection_policy": (
                        "strict_positive_topk"
                        if is_sae
                        else "exact_positive_topk"
                    ),
                    "source_config_semantic_sha256": (
                        f"{panel_index + 1:x}" * 64
                    ),
                    "source_config_file_sha256": (
                        f"{panel_index + 3:x}" * 64
                    ),
                    "source_result": {
                        "controller_path": str(
                            (
                                tmp_path
                                / panel
                                / str(milestone)
                                / "scientific-result.json"
                            ).resolve()
                        ),
                        "remote_path": (
                            f"/remote/{panel}/{milestone}/"
                            "scientific-result.json"
                        ),
                        "file_sha256": (
                            f"{milestone_index + 5:x}" * 64
                        ),
                        "result_sha256": (
                            f"{milestone_index + 8:x}" * 64
                        ),
                    },
                    "parent_hashes": _parents(panel, milestone_index),
                    "endpoints": {
                        "component": {
                            "sse": 10.0 * nmse_component,
                            "sst": 10.0,
                            "nmse": nmse_component,
                            "ev": 1.0 - nmse_component,
                        },
                        "routed": {
                            "sse": 100.0 * nmse_routed,
                            "sst": 100.0,
                            "nmse": nmse_routed,
                            "ev": 1.0 - nmse_routed,
                        },
                    },
                    "sparsity": {
                        "l0_per_contribution": 2,
                        "total_l0": 16,
                        "dead_rate": 0.25 if is_sae else 0.05,
                        "fired_feature_fraction": (
                            0.75 if is_sae else 0.95
                        ),
                    },
                }
            )
    body = {
        "schema_version": A256_CONTEXT_SCORE_MANIFEST_SCHEMA_VERSION,
        "manifest_id": A256_CONTEXT_SCORE_MANIFEST_ID,
        "source_protocol_id": EXPERIMENT4_PROTOCOL_ID,
        "target_protocol_id": PROTOCOL_ID,
        "source_plan_sha256": "f" * 64,
        "provenance_addendum": {
            "addendum_id": (
                "gemma4-experiment5-v3-a256native-"
                "provenance-correction-20260729"
            ),
            "controller_path": str((tmp_path / "addendum.yaml").resolve()),
            "file_sha256": "d" * 64,
            "manifest_sha256": "e" * 64,
        },
        "scientific_scope": _scope(),
        "shared_held_out_identity": {
            "evaluation_tokens": 250_000,
            "target_capture_manifest_sha256": "a" * 64,
            "component_sst": 10.0,
            "routed_sst": 100.0,
            "panel_representations": {
                "sae": {
                    "ordered_token_identity_sha256": "1" * 64,
                    "component_sst_content_sha256": "2" * 64,
                    "routed_sst_content_sha256": "3" * 64,
                },
                "transcoder": {
                    "ordered_token_identity_sha256": "4" * 64,
                    "component_sst_content_sha256": "5" * 64,
                    "routed_sst_content_sha256": "6" * 64,
                },
            },
            "token_identity_policy": _token_policy(),
        },
        "cells": cells,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _rehash(manifest: dict) -> None:
    body = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = canonical_sha256(body)


def test_manifest_preserves_panel_native_semantics_and_six_scores(
    tmp_path: Path,
) -> None:
    parsed = parse_a256_context_score_manifest(_manifest(tmp_path))

    assert len(parsed["cells"]) == 6
    by_panel = {
        panel: [row for row in parsed["cells"] if row["panel"] == panel]
        for panel in ("sae", "transcoder")
    }
    assert {
        (row["source_label"], row["encoder_activation"])
        for row in by_panel["sae"]
    } == {("A256ReLU", "relu")}
    assert {
        (row["source_label"], row["encoder_activation"])
        for row in by_panel["transcoder"]
    } == {("A256Softplus", "softplus")}
    assert (
        parsed["scientific_scope"]["cross_panel_pooling_allowed"] is False
    )
    assert (
        parsed["scientific_scope"]["included_in_primary_a512_inference"]
        is False
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["scientific_scope"].__setitem__(
                "cross_panel_pooling_allowed", True
            ),
            "scope changed",
        ),
        (
            lambda value: value["cells"][3].__setitem__(
                "encoder_activation", "relu"
            ),
            "not panel-native",
        ),
        (
            lambda value: value["cells"][0]["parent_hashes"].__setitem__(
                "target_capture_manifest_sha256", "f" * 64
            ),
            "shared held-out identity",
        ),
        (
            lambda value: value["cells"][1]["parent_hashes"].__setitem__(
                "component_sst_content_sha256", "f" * 64
            ),
            "shared held-out identity",
        ),
        (
            lambda value: value["cells"][2]["endpoints"]["component"].__setitem__(
                "ev", 0.0
            ),
            "EV=1-NMSE",
        ),
    ],
)
def test_manifest_fails_closed_on_scope_parent_or_metric_change(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    manifest = _manifest(tmp_path)
    mutation(manifest)
    _rehash(manifest)

    with pytest.raises(A256ContextScoreError, match=match):
        parse_a256_context_score_manifest(manifest)


def test_manifest_keeps_panel_local_ordered_identities_distinct(
    tmp_path: Path,
) -> None:
    parsed = parse_a256_context_score_manifest(_manifest(tmp_path))
    representations = parsed["shared_held_out_identity"][
        "panel_representations"
    ]

    assert (
        representations["sae"]["ordered_token_identity_sha256"]
        != representations["transcoder"]["ordered_token_identity_sha256"]
    )
    assert "no cross-panel byte-hash equality or pooling" in (
        parsed["shared_held_out_identity"]["token_identity_policy"]
    )


def test_write_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "context-scores.yaml"
    manifest = _manifest(tmp_path)

    written = write_a256_context_score_manifest(output, manifest)
    assert written["manifest_sha256"] == manifest["manifest_sha256"]
    with pytest.raises(A256ContextScoreError, match="refusing to overwrite"):
        write_a256_context_score_manifest(output, manifest)
