"""Read-only importer for the completed Experiment-5 v3 A trajectories."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_routed_ladder_v3_campaign import (
    RoutedLadderV3TrainingPlan,
    load_routed_ladder_v3_training_plan,
)

from moe_sae.component_closure_config import (
    CHECKPOINT_TOKENS,
    PANELS,
    PROTOCOL_ID,
    SEEDS,
    SOURCE_CONDITION,
    SOURCE_CONFIG_FILE_SHA256,
    SOURCE_CONFIG_SHA256,
    SOURCE_PHASE_NAME,
    SOURCE_PROTOCOL_ID,
)

SCHEMA_VERSION = "gemma4-a512-component-closure-a-reference-manifest-v1"


class ComponentClosureReferenceError(ValueError):
    """The A import is incomplete, mutable, or scientifically incompatible."""


def validate_a_reference_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ComponentClosureReferenceError("A reference manifest must be a mapping")
    selected = copy.deepcopy(dict(value))
    expected_keys = {
        "schema_version",
        "source_protocol_id",
        "target_protocol_id",
        "source_plan_sha256",
        "import_policy",
        "cells",
        "manifest_sha256",
    }
    if set(selected) != expected_keys:
        raise ComponentClosureReferenceError("A reference manifest fields changed")
    claimed = selected.pop("manifest_sha256")
    if (
        selected["schema_version"] != SCHEMA_VERSION
        or selected["source_protocol_id"] != SOURCE_PROTOCOL_ID
        or selected["target_protocol_id"] != PROTOCOL_ID
        or selected["import_policy"] != "immutable_read_only_no_copy_no_relabel"
        or not isinstance(claimed, str)
        or canonical_sha256(selected) != claimed
    ):
        raise ComponentClosureReferenceError("A reference manifest identity changed")
    cells = selected["cells"]
    if not isinstance(cells, list):
        raise ComponentClosureReferenceError("A reference cells must be a list")
    expected_roles = [
        (panel, seed, checkpoint)
        for panel in PANELS
        for seed in SEEDS
        for checkpoint in CHECKPOINT_TOKENS
    ]
    roles: list[tuple[str, int, int]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping) or set(cell) != {
            "panel",
            "target_condition",
            "source_condition",
            "seed",
            "checkpoint_tokens",
            "source_phase_name",
            "source_config_sha256",
            "source_config_file_sha256",
            "training_source_revision",
            "training_result_controller_path",
            "run_root_remote",
            "checkpoint_payload_root_gcs",
            "parent_training_result_sha256",
            "parent_training_result_file_sha256",
            "parent_run_contract_sha256",
            "parent_run_contract_file_sha256",
            "parent_checkpoint_pointer_sha256",
            "parent_checkpoint_pointer_file_sha256",
            "resource_id",
            "accelerator_type",
        }:
            raise ComponentClosureReferenceError(
                f"A reference cell {index} fields changed"
            )
        panel = cell["panel"]
        seed = cell["seed"]
        checkpoint = cell["checkpoint_tokens"]
        if (
            panel not in PANELS
            or seed not in SEEDS
            or checkpoint not in CHECKPOINT_TOKENS
            or cell["target_condition"] != "A"
            or cell["source_condition"] != SOURCE_CONDITION
            or cell["source_phase_name"] != SOURCE_PHASE_NAME
            or cell["source_config_sha256"] != SOURCE_CONFIG_SHA256[panel]
            or cell["source_config_file_sha256"] != SOURCE_CONFIG_FILE_SHA256[panel]
        ):
            raise ComponentClosureReferenceError(
                f"A reference cell {index} scientific role changed"
            )
        roles.append((str(panel), int(seed), int(checkpoint)))
    if roles != expected_roles:
        raise ComponentClosureReferenceError(
            "A reference manifest must contain the ordered 18-cell grid"
        )
    return {**selected, "manifest_sha256": claimed}


def build_a_reference_manifest(
    training_plan: RoutedLadderV3TrainingPlan,
) -> dict[str, Any]:
    """Hash-bind the 18 completed A cells without copying checkpoint payloads."""

    # This helper already validates the v3 result, run contract, milestone
    # pointer, source revision, exact optimizer progress, and payload root.
    from moe_sae.experiment5_routed_ladder_v3_evaluation import _fresh_parent

    cells: list[dict[str, Any]] = []
    for panel in PANELS:
        for seed in SEEDS:
            task = next(
                row
                for row in training_plan.trajectories
                if row.panel == panel and row.seed == seed
            )
            resource = training_plan.resource(task.eligible_resource_ids[0])
            for checkpoint in CHECKPOINT_TOKENS:
                parent = _fresh_parent(
                    training_plan,
                    panel=panel,
                    seed=seed,
                    checkpoint_tokens=checkpoint,
                )
                cells.append(
                    {
                        "panel": panel,
                        "target_condition": "A",
                        "source_condition": SOURCE_CONDITION,
                        "seed": seed,
                        "checkpoint_tokens": checkpoint,
                        "source_phase_name": SOURCE_PHASE_NAME,
                        "source_config_sha256": SOURCE_CONFIG_SHA256[panel],
                        "source_config_file_sha256": (SOURCE_CONFIG_FILE_SHA256[panel]),
                        "training_source_revision": parent["training_source_revision"],
                        "training_result_controller_path": str(
                            task.completion_path_controller
                        ),
                        "run_root_remote": parent["run_root_remote"],
                        "checkpoint_payload_root_gcs": parent[
                            "checkpoint_payload_root_gcs"
                        ],
                        "parent_training_result_sha256": parent[
                            "parent_training_result_sha256"
                        ],
                        "parent_training_result_file_sha256": parent[
                            "parent_training_result_file_sha256"
                        ],
                        "parent_run_contract_sha256": parent[
                            "parent_run_contract_sha256"
                        ],
                        "parent_run_contract_file_sha256": parent[
                            "parent_run_contract_file_sha256"
                        ],
                        "parent_checkpoint_pointer_sha256": parent[
                            "parent_checkpoint_pointer_sha256"
                        ],
                        "parent_checkpoint_pointer_file_sha256": parent[
                            "parent_checkpoint_pointer_file_sha256"
                        ],
                        "resource_id": resource.resource_id,
                        "accelerator_type": resource.accelerator_type,
                    }
                )
    body = {
        "schema_version": SCHEMA_VERSION,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "target_protocol_id": PROTOCOL_ID,
        "source_plan_sha256": training_plan.plan_sha256,
        "import_policy": "immutable_read_only_no_copy_no_relabel",
        "cells": cells,
    }
    return validate_a_reference_manifest(
        {**body, "manifest_sha256": canonical_sha256(body)}
    )


def build_a_reference_manifest_from_path(path: str | Path) -> dict[str, Any]:
    return build_a_reference_manifest(load_routed_ladder_v3_training_plan(path))


def load_a_reference_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComponentClosureReferenceError(
            f"cannot read A reference manifest {source}"
        ) from error
    return validate_a_reference_manifest(value)


__all__ = [
    "SCHEMA_VERSION",
    "ComponentClosureReferenceError",
    "build_a_reference_manifest",
    "build_a_reference_manifest_from_path",
    "load_a_reference_manifest",
    "validate_a_reference_manifest",
]
