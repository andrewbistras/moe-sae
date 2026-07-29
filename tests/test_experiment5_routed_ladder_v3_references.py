from __future__ import annotations

from pathlib import Path

import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.direct_horizon_campaign import PROTOCOL_ID as EXPERIMENT4_PROTOCOL_ID
from moe_sae.experiment5_routed_ladder_campaign import (
    RoutedLadderCampaignPlanError,
)
from moe_sae.experiment5_routed_ladder_v3_config import (
    A256_NATIVE_CONFIG_SHA256,
    A256_NATIVE_LABEL,
    PROTOCOL_ID,
)
from moe_sae.experiment5_routed_ladder_v3_references import (
    MILESTONES,
    NATIVE_A256_REFERENCE_MANIFEST_SCHEMA_VERSION,
    parse_native_a256_reference_manifest,
)


def _evidence(path: Path, remote: str, digest: str) -> dict[str, str]:
    return {
        "controller_path": str(path),
        "remote_path": remote,
        "sha256": digest,
    }


def _manifest(tmp_path: Path) -> dict:
    costs = []
    cells = []
    for panel_index, panel in enumerate(("sae", "transcoder")):
        resource = f"slice-{panel}"
        accelerator = "v4-32"
        costs.append(
            {
                "panel": panel,
                "resource_id": resource,
                "accelerator_type": accelerator,
                "chip_count": 16,
                "marginal_wall_seconds": 10.0 + panel_index,
                "chip_seconds": 16.0 * (10.0 + panel_index),
                "chip_seconds_scope": "training",
                "chip_seconds_excludes": ["compile", "checkpoint"],
            }
        )
        root = tmp_path / panel
        common = {
            "panel": panel,
            "target_condition": "A256Native",
            "source_condition": "A",
            "source_label": A256_NATIVE_LABEL[panel],
            "seed": 0,
            "training_source_revision": "a" * 40,
            "source_config": _evidence(
                root / "config.yaml",
                f"/remote/{panel}/config.yaml",
                "1" * 64,
            ),
            "source_config_sha256": A256_NATIVE_CONFIG_SHA256[panel],
            "run_root_remote": f"/remote/{panel}/runs/a",
            "checkpoint_payload_root_gcs": f"gs://bucket/{panel}/a",
            "run_contract": _evidence(
                root / "run-contract.json",
                f"/remote/{panel}/runs/a/run-contract.json",
                "2" * 64,
            ),
            "run_contract_sha256": "3" * 64,
            "training_result": _evidence(
                root / "training-result.json",
                f"/remote/{panel}/runs/a/training-result.json",
                "4" * 64,
            ),
            "training_result_sha256": "5" * 64,
            "resource_id": resource,
            "accelerator_type": accelerator,
        }
        for milestone_index, milestone in enumerate(MILESTONES):
            cells.append(
                {
                    **common,
                    "checkpoint_tokens": milestone,
                    "checkpoint_pointer": _evidence(
                        root / f"{milestone}.json",
                        f"/remote/{panel}/runs/a/checkpoints/{milestone}.json",
                        f"{6 + milestone_index + panel_index * 3:x}" * 64,
                    ),
                    "checkpoint_pointer_sha256": (
                        f"{milestone_index + panel_index * 3:x}" * 64
                    ),
                }
            )
    body = {
        "schema_version": NATIVE_A256_REFERENCE_MANIFEST_SCHEMA_VERSION,
        "source_protocol_id": EXPERIMENT4_PROTOCOL_ID,
        "target_protocol_id": PROTOCOL_ID,
        "source_plan_sha256": "f" * 64,
        "training_costs": costs,
        "cells": cells,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def test_native_a256_manifest_requires_exact_two_panel_grid(
    tmp_path: Path,
) -> None:
    _parsed, cells, costs = parse_native_a256_reference_manifest(
        _manifest(tmp_path)
    )

    assert [cell.identity for cell in cells] == [
        (panel, "A256Native", 0, milestone)
        for panel in ("sae", "transcoder")
        for milestone in MILESTONES
    ]
    assert [cost.panel for cost in costs] == ["sae", "transcoder"]


def test_native_a256_manifest_rejects_cross_panel_relabelling(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    manifest["cells"][3]["source_label"] = "A256ReLU"
    manifest["manifest_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_sha256"
        }
    )

    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="panel-native",
    ):
        parse_native_a256_reference_manifest(manifest)
