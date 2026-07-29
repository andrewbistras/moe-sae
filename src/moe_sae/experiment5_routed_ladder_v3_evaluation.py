"""Primary bundled evaluation for Experiment-5 activation-policy v3.

The decisive v3 result contains only the matched A512 pair.  Optional A256
and H context deliberately remain outside this launch barrier: a historical
provenance correction must not delay evaluation of six freshly trained ReLU
trajectories against six hash-pinned, graph-identical v2 Softplus
trajectories.

This module is intentionally smaller than the training scheduler.  It builds
six immutable resident-bundle manifests (one panel/seed each), runs them in
parallel, and validates the resulting 36-cell grid.  Re-entry is monotonic:
complete bundles are revalidated and skipped; partial or changed evidence
fails closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from moe_sae.checkpointing import milestone_pointer_path, validate_pointer
from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_campaign import (
    CampaignResource,
    _read_self_hashed_json,
)
from moe_sae.experiment5_routed_ladder_campaign import (
    CompletedTrajectoryCheckpointReference,
    RoutedLadderCampaignPlan,
    RoutedLadderCampaignPlanError,
    RoutedLadderCampaignStateError,
    load_routed_ladder_campaign_plan,
    parse_completed_trajectory_reference_manifest,
    validate_completed_checkpoint_lineage,
)
from moe_sae.experiment5_routed_ladder_evaluation_worker import (
    CHECKPOINT_TOKENS,
    V3_BUNDLE_RESULT_SCHEMA_VERSION,
    V3_RESULT_SCHEMA_VERSION,
    load_merged_scores,
)
from moe_sae.experiment5_routed_ladder_v3_campaign import (
    EXTENSION_NAMESPACE,
    RoutedLadderV3TrainingPlan,
    inspect_v3_training_completion,
    load_routed_ladder_v3_training_plan,
)
from moe_sae.experiment5_routed_ladder_v3_config import (
    PHASE_NAME,
    PROTOCOL_ID,
)

PLAN_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-policy-primary-evaluation-plan-v1"
)
OPERATOR_MANIFEST_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-policy-operator-bundle-manifest-v1"
)
COMPLETION_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-policy-primary-evaluation-"
    "completion-v1"
)
PANELS = ("sae", "transcoder")
SEEDS = (0, 1, 2)
CONDITIONS = ("A512ReLUInclusiveTopK", "A512Softplus")
CELL_COUNT = len(PANELS) * len(SEEDS) * len(CONDITIONS) * len(CHECKPOINT_TOKENS)
BUNDLE_COUNT = len(PANELS) * len(SEEDS)
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_CONFIG_NAMES = {
    "sae": "gemma4_experiment5_routed_ladder_sae_v3.yaml",
    "transcoder": "gemma4_experiment5_routed_ladder_transcoder_v3.yaml",
}


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise RoutedLadderCampaignStateError(
                f"immutable v3 evaluation artifact changed: {path}"
            )
        return
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise RoutedLadderCampaignStateError(
            f"cannot hash v3 evaluation parent: {path}"
        ) from error


def _expected_progress(
    result: Mapping[str, Any],
    checkpoint_tokens: int,
) -> tuple[int, int]:
    identity = result.get("run_identity")
    run_plan = identity.get("plan") if isinstance(identity, Mapping) else None
    optimizer = run_plan.get("optimizer") if isinstance(run_plan, Mapping) else None
    if not isinstance(run_plan, Mapping) or not isinstance(optimizer, Mapping):
        raise RoutedLadderCampaignStateError(
            "training result lacks immutable optimizer progress"
        )
    microbatch = run_plan.get("global_microbatch_tokens")
    accumulation = optimizer.get("gradient_accumulation_steps")
    total = run_plan.get("total_tokens")
    if (
        isinstance(microbatch, bool)
        or not isinstance(microbatch, int)
        or microbatch <= 0
        or isinstance(accumulation, bool)
        or not isinstance(accumulation, int)
        or accumulation <= 0
        or total != 10_000_000
    ):
        raise RoutedLadderCampaignStateError(
            "training result progress contract changed"
        )
    batch = microbatch * accumulation
    step = math.ceil(checkpoint_tokens / batch)
    return step, min(step * batch, int(total))


def _fresh_parent(
    plan: RoutedLadderV3TrainingPlan,
    *,
    panel: str,
    seed: int,
    checkpoint_tokens: int,
) -> dict[str, Any]:
    task = next(
        row
        for row in plan.trajectories
        if row.panel == panel and row.seed == seed
    )
    if inspect_v3_training_completion(plan, task) is None:
        raise RoutedLadderCampaignStateError(
            f"fresh v3 parent is incomplete: {panel}/seed{seed}"
        )
    result, result_payload = _read_self_hashed_json(
        task.completion_path_controller,
        hash_field="result_sha256",
    )
    run_root = task.completion_path_controller.parent
    contract_path = run_root / "run-contract.json"
    contract, contract_payload = _read_self_hashed_json(
        contract_path,
        hash_field="run_contract_sha256",
    )
    pointer_path = milestone_pointer_path(
        run_root / "checkpoints",
        checkpoint_tokens,
    )
    pointer, pointer_payload = _read_self_hashed_json(
        pointer_path,
        hash_field="pointer_sha256",
    )
    identity = result.get("run_identity")
    if not isinstance(identity, Mapping):
        raise RoutedLadderCampaignStateError(
            f"fresh v3 run identity is absent: {panel}/seed{seed}"
        )
    try:
        checked = validate_pointer(
            pointer,
            expected_kind="latest",
            payload_root=task.checkpoint_root_gcs,
            run_identity=identity,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise RoutedLadderCampaignStateError(
            f"fresh v3 checkpoint is invalid: {panel}/seed{seed}"
        ) from error
    expected_step, expected_tokens = _expected_progress(
        result,
        checkpoint_tokens,
    )
    config_file = plan.working_directory / "configs" / _CONFIG_NAMES[panel]
    if (
        contract.get("project_commit") != plan.project_commit
        or contract.get("phase") != PHASE_NAME
        or contract.get("scientific_model") != panel
        or contract.get("condition") != "A512ReLUInclusiveTopK"
        or contract.get("seed") != seed
        or contract.get("config_file_sha256") != _file_hash(config_file)
        or contract.get("checkpoint_payload_root") != task.checkpoint_root_gcs
        or checked["step"] != expected_step
        or checked["tokens_seen"] != expected_tokens
    ):
        raise RoutedLadderCampaignStateError(
            f"fresh v3 parent lineage changed: {panel}/seed{seed}"
        )
    return {
        "training_source_revision": plan.project_commit,
        "run_root_remote": str(
            PurePosixPath(task.control_root_remote) / "runs" / task.run_id
        ),
        "checkpoint_payload_root_gcs": task.checkpoint_root_gcs,
        "reference_kind": "trained_v3",
        "reference_config_remote": None,
        "reference_run_contract_sha256": None,
        "reference_run_contract_file_sha256": None,
        "parent_training_result_sha256": str(result["result_sha256"]),
        "parent_training_result_file_sha256": hashlib.sha256(
            result_payload
        ).hexdigest(),
        "parent_run_contract_sha256": str(
            contract["run_contract_sha256"]
        ),
        "parent_run_contract_file_sha256": hashlib.sha256(
            contract_payload
        ).hexdigest(),
        "parent_checkpoint_pointer_sha256": str(
            checked["pointer_sha256"]
        ),
        "parent_checkpoint_pointer_file_sha256": hashlib.sha256(
            pointer_payload
        ).hexdigest(),
    }


def _imported_parent(
    cell: CompletedTrajectoryCheckpointReference,
) -> dict[str, Any]:
    validate_completed_checkpoint_lineage(cell)
    return {
        "training_source_revision": cell.training_source_revision,
        "run_root_remote": cell.run_root_remote,
        "checkpoint_payload_root_gcs": cell.checkpoint_payload_root_gcs,
        "reference_kind": "completed",
        "reference_config_remote": cell.config_remote,
        "reference_run_contract_sha256": None,
        "reference_run_contract_file_sha256": None,
        "parent_training_result_sha256": cell.training_result_sha256,
        "parent_training_result_file_sha256": cell.training_result.sha256,
        "parent_run_contract_sha256": cell.run_contract_sha256,
        "parent_run_contract_file_sha256": cell.run_contract.sha256,
        "parent_checkpoint_pointer_sha256": cell.checkpoint_pointer_sha256,
        "parent_checkpoint_pointer_file_sha256": cell.checkpoint_pointer.sha256,
    }


def _transcoder_overlay(
    v2_plan: RoutedLadderCampaignPlan,
    resource_id: str,
) -> dict[str, str | None]:
    candidates = [
        row
        for row in v2_plan.evaluations
        if row.panel == "transcoder"
        and row.eligible_resource_ids == (resource_id,)
    ]
    if not candidates:
        raise RoutedLadderCampaignPlanError(
            f"resource {resource_id} lacks a sealed transcoder evaluation plane"
        )
    values = {
        (
            row.train_source_root_remote,
            row.eval_source_root_remote,
            row.train_source_object_prefix_gcs,
            row.eval_source_object_prefix_gcs,
        )
        for row in candidates
    }
    if len(values) != 1 or any(value is None for value in next(iter(values))):
        raise RoutedLadderCampaignPlanError(
            f"resource {resource_id} transcoder overlay is ambiguous"
        )
    train_root, eval_root, train_prefix, eval_prefix = next(iter(values))
    return {
        "train_source_root_remote": train_root,
        "eval_source_root_remote": eval_root,
        "train_source_object_prefix_gcs": train_prefix,
        "eval_source_object_prefix_gcs": eval_prefix,
    }


def build_v3_primary_evaluation_plan(
    training_plan: RoutedLadderV3TrainingPlan,
    v2_plan: RoutedLadderCampaignPlan,
    completed_manifest: Mapping[str, Any],
    *,
    evaluation_source_revision: str,
) -> dict[str, Any]:
    """Bind the exact six-bundle/36-cell primary evaluation grid."""

    if _GIT_SHA.fullmatch(evaluation_source_revision) is None:
        raise RoutedLadderCampaignPlanError(
            "evaluation_source_revision must be a full Git SHA"
        )
    manifest, imported = parse_completed_trajectory_reference_manifest(
        completed_manifest
    )
    if (
        manifest["source_plan_sha256"] != v2_plan.plan_sha256
        or len(imported) != len(PANELS) * len(SEEDS) * len(CHECKPOINT_TOKENS)
    ):
        raise RoutedLadderCampaignPlanError(
            "Softplus imports do not bind the exact source campaign"
        )
    imported_by_id = {row.identity: row for row in imported}
    short = evaluation_source_revision[:12]
    state_root = Path(
        f"/mnt/moe-sae-control/{EXTENSION_NAMESPACE}/"
        f"primary-evaluation/controller/{short}"
    )
    bundles: list[dict[str, Any]] = []
    for panel in PANELS:
        for seed in SEEDS:
            fresh = next(
                row
                for row in training_plan.trajectories
                if row.panel == panel and row.seed == seed
            )
            owner_id = fresh.eligible_resource_ids[0]
            plane = training_plan.resource_pool.plane(owner_id)
            relative = (
                PurePosixPath(EXTENSION_NAMESPACE)
                / short
                / "evaluation"
                / panel
                / f"seed{seed}"
            )
            cells: list[dict[str, Any]] = []
            for condition in CONDITIONS:
                for checkpoint_tokens in CHECKPOINT_TOKENS:
                    parent = (
                        _fresh_parent(
                            training_plan,
                            panel=panel,
                            seed=seed,
                            checkpoint_tokens=checkpoint_tokens,
                        )
                        if condition == "A512ReLUInclusiveTopK"
                        else _imported_parent(
                            imported_by_id[
                                (panel, condition, seed, checkpoint_tokens)
                            ]
                        )
                    )
                    cell_relative = (
                        relative / condition / str(checkpoint_tokens)
                    )
                    cells.append(
                        {
                            "condition": condition,
                            "checkpoint_tokens": checkpoint_tokens,
                            **parent,
                            "result_root_remote": str(
                                PurePosixPath(plane.remote_artifact_root)
                                / cell_relative
                            ),
                            "result_root_controller": str(
                                plane.controller_artifact_root
                                / Path(*cell_relative.parts)
                            ),
                        }
                    )
            bundle_relative = relative / "_primary_bundle"
            operator_body = {
                "schema_version": OPERATOR_MANIFEST_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "panel": panel,
                "seed": seed,
                "source_revision": evaluation_source_revision,
                "config_remote": fresh.config_remote,
                "bundle_result_root_remote": str(
                    PurePosixPath(plane.remote_artifact_root)
                    / bundle_relative
                ),
                "bundle_result_root_controller": str(
                    plane.controller_artifact_root
                    / Path(*bundle_relative.parts)
                ),
                "cells": cells,
            }
            operator_manifest = {
                **operator_body,
                "manifest_sha256": canonical_sha256(operator_body),
            }
            overlay = (
                _transcoder_overlay(v2_plan, owner_id)
                if panel == "transcoder"
                else {
                    "train_source_root_remote": None,
                    "eval_source_root_remote": None,
                    "train_source_object_prefix_gcs": None,
                    "eval_source_object_prefix_gcs": None,
                }
            )
            task_id = f"eval-primary-{panel}-seed{seed}"
            bundles.append(
                {
                    "task_id": task_id,
                    "panel": panel,
                    "seed": seed,
                    "resource_id": owner_id,
                    "config_remote": fresh.config_remote,
                    "control_root_remote": fresh.control_root_remote,
                    "checkpoint_root_gcs": fresh.checkpoint_root_gcs,
                    **overlay,
                    "operator_manifest_path": str(
                        state_root
                        / "primary-evaluation"
                        / task_id
                        / "operator-manifest.json"
                    ),
                    "bundle_completion_path": (
                        operator_body["bundle_result_root_controller"]
                        + "/bundle-result.json"
                    ),
                    "operator_manifest": operator_manifest,
                }
            )
    body = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": (
            training_plan.campaign_id
            + f"-primary-evaluation-{evaluation_source_revision[:12]}"
        ),
        "evaluation_source_revision": evaluation_source_revision,
        "training_source_revision": training_plan.project_commit,
        "state_root": str(state_root),
        "training_plan_sha256": training_plan.plan_sha256,
        "v2_source_plan_sha256": v2_plan.plan_sha256,
        "completed_trajectory_manifest_sha256": manifest["manifest_sha256"],
        "bundles": bundles,
    }
    plan = {**body, "plan_sha256": canonical_sha256(body)}
    validate_v3_primary_evaluation_plan(plan)
    return plan


def validate_v3_primary_evaluation_plan(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    selected = copy.deepcopy(dict(value))
    if set(selected) != {
        "schema_version",
        "protocol_id",
        "campaign_id",
        "evaluation_source_revision",
        "training_source_revision",
        "state_root",
        "training_plan_sha256",
        "v2_source_plan_sha256",
        "completed_trajectory_manifest_sha256",
        "bundles",
        "plan_sha256",
    }:
        raise RoutedLadderCampaignPlanError(
            "v3 primary evaluation plan fields changed"
        )
    claimed = selected.pop("plan_sha256")
    if (
        selected["schema_version"] != PLAN_SCHEMA_VERSION
        or selected["protocol_id"] != PROTOCOL_ID
        or _GIT_SHA.fullmatch(
            str(selected["evaluation_source_revision"])
        )
        is None
        or _GIT_SHA.fullmatch(str(selected["training_source_revision"]))
        is None
        or selected["evaluation_source_revision"]
        == selected["training_source_revision"]
        or not isinstance(claimed, str)
        or canonical_sha256(selected) != claimed
    ):
        raise RoutedLadderCampaignPlanError(
            "v3 primary evaluation plan identity changed"
        )
    bundles = selected["bundles"]
    if not isinstance(bundles, list) or len(bundles) != BUNDLE_COUNT:
        raise RoutedLadderCampaignPlanError(
            "v3 primary evaluation plan requires six bundles"
        )
    observed: set[tuple[str, int]] = set()
    cells: set[tuple[str, str, int, int]] = set()
    owners: set[str] = set()
    for row in bundles:
        if not isinstance(row, Mapping):
            raise RoutedLadderCampaignPlanError(
                "v3 primary evaluation bundle must be an object"
            )
        panel = row.get("panel")
        seed = row.get("seed")
        owner = row.get("resource_id")
        manifest = row.get("operator_manifest")
        if (
            panel not in PANELS
            or seed not in SEEDS
            or not isinstance(owner, str)
            or not isinstance(manifest, Mapping)
            or manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("source_revision")
            != selected["evaluation_source_revision"]
            or manifest.get("panel") != panel
            or manifest.get("seed") != seed
        ):
            raise RoutedLadderCampaignPlanError(
                "v3 primary evaluation bundle identity changed"
            )
        unsigned_manifest = {
            key: manifest[key]
            for key in manifest
            if key != "manifest_sha256"
        }
        if (
            manifest.get("schema_version")
            != OPERATOR_MANIFEST_SCHEMA_VERSION
            or canonical_sha256(unsigned_manifest)
            != manifest.get("manifest_sha256")
        ):
            raise RoutedLadderCampaignPlanError(
                "v3 operator manifest identity changed"
            )
        observed.add((str(panel), int(seed)))
        owners.add(owner)
        raw_cells = manifest.get("cells")
        if not isinstance(raw_cells, list) or len(raw_cells) != (
            len(CONDITIONS) * len(CHECKPOINT_TOKENS)
        ):
            raise RoutedLadderCampaignPlanError(
                "v3 primary bundle cell count changed"
            )
        expected_order = [
            (condition, checkpoint)
            for condition in CONDITIONS
            for checkpoint in CHECKPOINT_TOKENS
        ]
        if [
            (cell.get("condition"), cell.get("checkpoint_tokens"))
            for cell in raw_cells
        ] != expected_order:
            raise RoutedLadderCampaignPlanError(
                "v3 primary bundle order changed"
            )
        cells.update(
            (str(panel), str(condition), int(seed), int(checkpoint))
            for condition, checkpoint in expected_order
        )
    expected_bundles = {
        (panel, seed) for panel in PANELS for seed in SEEDS
    }
    expected_cells = {
        (panel, condition, seed, checkpoint)
        for panel in PANELS
        for condition in CONDITIONS
        for seed in SEEDS
        for checkpoint in CHECKPOINT_TOKENS
    }
    if (
        observed != expected_bundles
        or cells != expected_cells
        or len(owners) != BUNDLE_COUNT
    ):
        raise RoutedLadderCampaignPlanError(
            "v3 primary evaluation matrix or owners are not exact"
        )
    return {**selected, "plan_sha256": claimed}


def write_v3_primary_evaluation_plan(
    path: str | Path,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    plan = validate_v3_primary_evaluation_plan(value)
    selected = Path(path).expanduser().resolve()
    _create_only(selected, _json_payload(plan))
    for bundle in plan["bundles"]:
        _create_only(
            Path(bundle["operator_manifest_path"]),
            _json_payload(bundle["operator_manifest"]),
        )
    return plan


def load_v3_primary_evaluation_plan(
    path: str | Path,
) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        value = json.loads(selected.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RoutedLadderCampaignPlanError(
            f"cannot read v3 primary evaluation plan: {selected}"
        ) from error
    if not isinstance(value, Mapping):
        raise RoutedLadderCampaignPlanError(
            "v3 primary evaluation plan must be an object"
        )
    return validate_v3_primary_evaluation_plan(value)


def _resource(
    training_plan: RoutedLadderV3TrainingPlan,
    resource_id: str,
) -> CampaignResource:
    return training_plan.resource(resource_id)


def _bundle_environment(
    training_plan: RoutedLadderV3TrainingPlan,
    bundle: Mapping[str, Any],
    *,
    evaluation_source_revision: str,
    log_root: Path,
) -> dict[str, str]:
    resource = _resource(training_plan, str(bundle["resource_id"]))
    environment = os.environ.copy()
    environment.update(training_plan.environment_dict())
    environment.update(resource.environment_dict())
    environment.update(
        {
            "GCP_PROJECT": resource.project,
            "TPU_NAME": resource.tpu_name,
            "TPU_ZONE": resource.zone,
            "PROJECT_COMMIT": evaluation_source_revision,
            "TPU_PROJECT_ROOT": training_plan.remote_project_root,
            "REMOTE_UV": training_plan.remote_uv,
            "TPU_ACCELERATOR_TYPE": resource.accelerator_type,
            "TPU_USE_INTERNAL_IP": "1" if resource.internal_ip else "0",
            "EXPERIMENT5_PANEL": str(bundle["panel"]),
            "CONTRIBUTION_SAE_CONTROL_ROOT": str(
                bundle["control_root_remote"]
            ),
            "CONTRIBUTION_SAE_CHECKPOINT_ROOT": str(
                bundle["checkpoint_root_gcs"]
            ),
            "CONTRIBUTION_FRESH_CONTROLLER_MAPPING": (
                resource.controller_mapping.remote_path
            ),
            "CONTRIBUTION_LOCAL_CACHE_INVENTORY": (
                resource.cache_inventory.remote_path
            ),
            "EXPERIMENT5_ROUTED_ACTIVATION_LADDER_RESULT_ROOT_CONTROLLER": (
                str(
                    Path(bundle["bundle_completion_path"]).parent
                )
            ),
        }
    )
    if resource.mirror_seal is not None:
        environment["CONTRIBUTION_REGIONAL_MIRROR_SEAL"] = (
            resource.mirror_seal.remote_path
        )
    if resource.ssh_user is not None:
        environment["TPU_SSH_USER"] = resource.ssh_user
    log_name = (
        "CONTRIBUTION_EVALUATION_LOG_ROOT"
        if bundle["panel"] == "sae"
        else "TRANSCODER_EVALUATION_LOG_ROOT"
    )
    environment[log_name] = str(log_root)
    if bundle["panel"] == "transcoder":
        environment.update(
            {
                "CONTRIBUTION_SOURCE_OVERLAY_ROOT": str(
                    bundle["train_source_root_remote"]
                ),
                "CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT": str(
                    bundle["eval_source_root_remote"]
                ),
                "CONTRIBUTION_SOURCE_OVERLAY_PREFIX": str(
                    bundle["train_source_object_prefix_gcs"]
                ),
                "CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX": str(
                    bundle["eval_source_object_prefix_gcs"]
                ),
            }
        )
    return environment


def _bundle_command(
    training_plan: RoutedLadderV3TrainingPlan,
    bundle: Mapping[str, Any],
    *,
    evaluation_source_revision: str,
    attempt_id: str,
) -> list[str]:
    return [
        str(training_plan.python_executable),
        str(
            training_plan.working_directory
            / "scripts/tpu/evaluate_experiment5_routed_ladder.py"
        ),
        "--panel",
        str(bundle["panel"]),
        "--config",
        str(bundle["config_remote"]),
        "--seed",
        str(bundle["seed"]),
        "--source-revision",
        evaluation_source_revision,
        "--bundle-manifest",
        str(bundle["operator_manifest_path"]),
        "--attempt-id",
        attempt_id,
    ]


def _inspect_bundle(bundle: Mapping[str, Any]) -> dict[str, Any] | None:
    path = Path(bundle["bundle_completion_path"])
    if not path.exists():
        return None
    value, payload = _read_self_hashed_json(path, hash_field="result_sha256")
    manifest = bundle["operator_manifest"]
    expected_cells = []
    for cell in manifest["cells"]:
        result_path = (
            Path(cell["result_root_controller"]) / "evaluation-result.json"
        )
        scores = load_merged_scores(result_path)
        result, result_payload = _read_self_hashed_json(
            result_path,
            hash_field="result_sha256",
        )
        if (
            result.get("schema_version") != V3_RESULT_SCHEMA_VERSION
            or result.get("protocol_id") != PROTOCOL_ID
            or scores.condition != cell["condition"]
            or scores.panel != bundle["panel"]
            or scores.seed != bundle["seed"]
            or scores.checkpoint_tokens != cell["checkpoint_tokens"]
        ):
            raise RoutedLadderCampaignStateError(
                f"v3 evaluation cell changed: {scores.identity}"
            )
        expected_cells.append(
            {
                "condition": cell["condition"],
                "checkpoint_tokens": cell["checkpoint_tokens"],
                "training_source_revision": cell[
                    "training_source_revision"
                ],
                "parent_training_result_sha256": cell[
                    "parent_training_result_sha256"
                ],
                "parent_training_result_file_sha256": cell[
                    "parent_training_result_file_sha256"
                ],
                "parent_run_contract_sha256": cell[
                    "parent_run_contract_sha256"
                ],
                "parent_run_contract_file_sha256": cell[
                    "parent_run_contract_file_sha256"
                ],
                "parent_checkpoint_pointer_sha256": cell[
                    "parent_checkpoint_pointer_sha256"
                ],
                "parent_checkpoint_pointer_file_sha256": cell[
                    "parent_checkpoint_pointer_file_sha256"
                ],
                "result_root": cell["result_root_remote"],
                "result_sha256": result["result_sha256"],
                "result_file_sha256": hashlib.sha256(
                    result_payload
                ).hexdigest(),
                "score_artifact_sha256": result["score_artifact_sha256"],
                "score_data_sha256": result["score_data_sha256"],
                "score_data_bytes": result["score_data_bytes"],
            }
        )
    if (
        value.get("schema_version") != V3_BUNDLE_RESULT_SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "bundle_evaluated"
        or value.get("panel") != bundle["panel"]
        or value.get("seed") != bundle["seed"]
        or value.get("source_revision") != manifest["source_revision"]
        or value.get("operator_manifest_sha256")
        != manifest["manifest_sha256"]
        or value.get("cell_results") != expected_cells
    ):
        raise RoutedLadderCampaignStateError(
            f"v3 primary bundle completion changed: {bundle['task_id']}"
        )
    return {
        "task_id": bundle["task_id"],
        "path": str(path),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "result_sha256": value["result_sha256"],
    }


def run_v3_primary_evaluation(
    evaluation_plan: Mapping[str, Any],
    training_plan: RoutedLadderV3TrainingPlan,
    *,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Run pending bundles in parallel and seal the exact primary matrix."""

    plan = validate_v3_primary_evaluation_plan(evaluation_plan)
    if (
        plan["training_plan_sha256"] != training_plan.plan_sha256
        or plan["training_source_revision"] != training_plan.project_commit
    ):
        raise RoutedLadderCampaignStateError(
            "v3 evaluation plan and training plan differ"
        )
    completions: dict[str, dict[str, Any]] = {}
    pending: list[Mapping[str, Any]] = []
    for bundle in plan["bundles"]:
        observed = _inspect_bundle(bundle)
        if observed is None:
            pending.append(bundle)
        else:
            completions[str(bundle["task_id"])] = observed
    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        attempt_stamp = f"v3eval-{int(time.time())}-{attempt}"
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            futures = {}
            for bundle in pending:
                log_root = (
                    Path(plan["state_root"])
                    / "primary-evaluation"
                    / str(bundle["task_id"])
                    / "operator-logs"
                    / attempt_stamp
                )
                command = _bundle_command(
                    training_plan,
                    bundle,
                    evaluation_source_revision=plan[
                        "evaluation_source_revision"
                    ],
                    attempt_id=attempt_stamp,
                )
                future = pool.submit(
                    subprocess.run,
                    command,
                    cwd=training_plan.working_directory,
                    env=_bundle_environment(
                        training_plan,
                        bundle,
                        evaluation_source_revision=plan[
                            "evaluation_source_revision"
                        ],
                        log_root=log_root,
                    ),
                    check=False,
                    capture_output=True,
                    text=True,
                )
                futures[future] = bundle
            for future in as_completed(futures):
                bundle = futures[future]
                completed = future.result()
                attempt_root = (
                    Path(plan["state_root"])
                    / "primary-evaluation"
                    / str(bundle["task_id"])
                    / "controller-attempts"
                    / attempt_stamp
                )
                _create_only(
                    attempt_root / "stdout.txt",
                    completed.stdout.encode("utf-8"),
                )
                _create_only(
                    attempt_root / "stderr.txt",
                    completed.stderr.encode("utf-8"),
                )
                if completed.returncode != 0:
                    failures.append(
                        f"{bundle['task_id']}:{completed.returncode}"
                    )
        pending = []
        for bundle in plan["bundles"]:
            observed = _inspect_bundle(bundle)
            if observed is None:
                pending.append(bundle)
            else:
                completions[str(bundle["task_id"])] = observed
        if pending and attempt == max_attempts:
            raise RoutedLadderCampaignStateError(
                "v3 primary evaluation did not complete after "
                f"{max_attempts} attempts: failures={failures}, "
                f"pending={[row['task_id'] for row in pending]}"
            )
    if len(completions) != BUNDLE_COUNT:
        raise RoutedLadderCampaignStateError(
            "v3 primary evaluation completion count changed"
        )
    body = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "status": "primary_evaluation_completed",
        "bundles": [completions[key] for key in sorted(completions)],
    }
    final = {**body, "completion_sha256": canonical_sha256(body)}
    path = (
        Path(plan["state_root"])
        / "primary-evaluation"
        / "primary-evaluation-complete.json"
    )
    _create_only(path, _json_payload(final))
    return final


def primary_evaluation_result_paths(
    evaluation_plan: Mapping[str, Any],
) -> tuple[Path, ...]:
    plan = validate_v3_primary_evaluation_plan(evaluation_plan)
    paths = [
        Path(cell["result_root_controller"]) / "evaluation-result.json"
        for bundle in plan["bundles"]
        for cell in bundle["operator_manifest"]["cells"]
    ]
    if len(paths) != CELL_COUNT or len(set(paths)) != CELL_COUNT:
        raise RoutedLadderCampaignPlanError(
            "v3 primary evaluation result paths are not exact"
        )
    return tuple(paths)


def build_from_paths(
    *,
    training_plan_path: str | Path,
    v2_plan_path: str | Path,
    completed_manifest_path: str | Path,
    evaluation_source_revision: str,
) -> dict[str, Any]:
    training = load_routed_ladder_v3_training_plan(training_plan_path)
    v2 = load_routed_ladder_campaign_plan(v2_plan_path)
    try:
        completed = yaml.safe_load(
            Path(completed_manifest_path).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RoutedLadderCampaignPlanError(
            "cannot read completed-trajectory reference manifest"
        ) from error
    if not isinstance(completed, Mapping):
        raise RoutedLadderCampaignPlanError(
            "completed-trajectory reference manifest must be an object"
        )
    return build_v3_primary_evaluation_plan(
        training,
        v2,
        completed,
        evaluation_source_revision=evaluation_source_revision,
    )


__all__ = [
    "BUNDLE_COUNT",
    "CELL_COUNT",
    "COMPLETION_SCHEMA_VERSION",
    "CONDITIONS",
    "OPERATOR_MANIFEST_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "build_from_paths",
    "build_v3_primary_evaluation_plan",
    "load_v3_primary_evaluation_plan",
    "primary_evaluation_result_paths",
    "run_v3_primary_evaluation",
    "validate_v3_primary_evaluation_plan",
    "write_v3_primary_evaluation_plan",
]
