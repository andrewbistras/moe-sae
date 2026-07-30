"""Mixed-parent resident evaluation plan and adapter for component closure."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from moe_sae.checkpointing import milestone_pointer_path, validate_pointer
from moe_sae.component_closure_campaign import validate_training_plan
from moe_sae.component_closure_config import (
    CHECKPOINT_TOKENS,
    CONDITIONS,
    PANELS,
    PHASE_NAME,
    PROTOCOL_ID,
    SEEDS,
    SOURCE_CONDITION,
    SOURCE_PHASE_NAME,
    SOURCE_PROTOCOL_ID,
    ComponentClosureConfigError,
    load_component_closure_config,
)
from moe_sae.component_closure_references import validate_a_reference_manifest
from moe_sae.contribution_artifacts import canonical_sha256, token_identity_sha256

PLAN_SCHEMA_VERSION = "gemma4-a512-component-closure-evaluation-plan-v1"
BUNDLE_SCHEMA_VERSION = "gemma4-a512-component-closure-resident-bundle-v1"
CELL_COUNT = 72
BUNDLE_COUNT = 6
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_FIELDS = {
    "resource_id",
    "a_config_remote",
    "result_root_remote",
    "result_root_controller",
    "accelerator_type",
    "peak_flops_per_second",
    "train_target_root",
    "eval_target_root",
    "train_target_object_prefix",
    "eval_target_object_prefix",
    "train_source_root",
    "eval_source_root",
    "train_source_object_prefix",
    "eval_source_object_prefix",
    "cache_root",
    "cache_max_bytes",
    "regional_mirror_seal",
    "recomposition_sidecar",
    "expected_token_identity_sha256",
}


class ComponentClosureEvaluationError(ValueError):
    """The mixed-parent closure evaluation grid is invalid."""


def _validated_runtime(
    value: Any,
    *,
    resource_id: str,
    panel: str,
    accelerator_type: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RUNTIME_FIELDS:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ComponentClosureEvaluationError(
            f"runtime for {resource_id} fields changed: {observed}"
        )
    selected = copy.deepcopy(dict(value))
    expected_peak = {
        "v5litepod-16": 3_152_000_000_000_000,
        "v4-32": 4_400_000_000_000_000,
    }.get(accelerator_type)
    if (
        selected["resource_id"] != resource_id
        or selected["accelerator_type"] != accelerator_type
        or expected_peak is None
        or isinstance(selected["peak_flops_per_second"], bool)
        or not isinstance(selected["peak_flops_per_second"], int)
        or selected["peak_flops_per_second"] != expected_peak
        or isinstance(selected["cache_max_bytes"], bool)
        or not isinstance(selected["cache_max_bytes"], int)
        or selected["cache_max_bytes"] != 171_798_691_840
        or (
            accelerator_type == "v5litepod-16"
            and selected["regional_mirror_seal"] is None
        )
        or (
            accelerator_type == "v4-32" and selected["regional_mirror_seal"] is not None
        )
    ):
        raise ComponentClosureEvaluationError(
            f"runtime identity changed for {resource_id}"
        )
    remote_fields = {
        "a_config_remote",
        "result_root_remote",
        "train_target_root",
        "eval_target_root",
        "cache_root",
        "recomposition_sidecar",
    }
    if selected["regional_mirror_seal"] is not None:
        remote_fields.add("regional_mirror_seal")
    for field in remote_fields:
        raw = selected[field]
        path = PurePosixPath(str(raw))
        if not isinstance(raw, str) or not path.is_absolute() or ".." in path.parts:
            raise ComponentClosureEvaluationError(
                f"runtime {resource_id}.{field} must be an absolute remote path"
            )
    controller = Path(str(selected["result_root_controller"]))
    if (
        not isinstance(selected["result_root_controller"], str)
        or not controller.is_absolute()
        or ".." in controller.parts
    ):
        raise ComponentClosureEvaluationError(
            f"runtime {resource_id}.result_root_controller is invalid"
        )
    for field in ("train_target_object_prefix", "eval_target_object_prefix"):
        raw = selected[field]
        if not isinstance(raw, str) or not raw.startswith("gs://") or raw.endswith("/"):
            raise ComponentClosureEvaluationError(
                f"runtime {resource_id}.{field} is not a GCS prefix"
            )
    source_fields = (
        "train_source_root",
        "eval_source_root",
        "train_source_object_prefix",
        "eval_source_object_prefix",
    )
    if panel == "sae":
        if any(selected[field] is not None for field in source_fields):
            raise ComponentClosureEvaluationError(
                f"SAE runtime {resource_id} must not declare source overlays"
            )
    else:
        for field in source_fields[:2]:
            raw = selected[field]
            path = PurePosixPath(str(raw))
            if not isinstance(raw, str) or not path.is_absolute() or ".." in path.parts:
                raise ComponentClosureEvaluationError(
                    f"transcoder runtime {resource_id}.{field} is invalid"
                )
        for field in source_fields[2:]:
            raw = selected[field]
            if (
                not isinstance(raw, str)
                or not raw.startswith("gs://")
                or raw.endswith("/")
            ):
                raise ComponentClosureEvaluationError(
                    f"transcoder runtime {resource_id}.{field} is invalid"
                )
    token_sha = selected["expected_token_identity_sha256"]
    if token_sha is not None and (
        not isinstance(token_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", token_sha) is None
    ):
        raise ComponentClosureEvaluationError(
            f"runtime {resource_id} token identity is invalid"
        )
    return selected


def _canonical_cell_result_roots(
    runtime: Mapping[str, Any],
    *,
    panel: str,
    seed: int,
    condition: str,
    checkpoint: int,
) -> tuple[str, str]:
    """Return the panel-qualified remote and controller roots for one cell."""

    remote_base = PurePosixPath(str(runtime["result_root_remote"]))
    controller_base = (
        Path(str(runtime["result_root_controller"])).expanduser().resolve()
    )
    known_panels = set(PANELS)
    remote_tail = remote_base.name
    controller_tail = controller_base.name
    if (
        remote_tail in known_panels
        and remote_tail != panel
        or controller_tail in known_panels
        and controller_tail != panel
    ):
        raise ComponentClosureEvaluationError(
            "evaluation result base has the wrong panel qualifier"
        )
    remote_qualified = remote_tail == panel
    controller_qualified = controller_tail == panel
    if remote_qualified != controller_qualified:
        raise ComponentClosureEvaluationError(
            "evaluation result bases have inconsistent panel qualification"
        )
    if remote_qualified and (
        remote_base.parent.name in known_panels
        or controller_base.parent.name in known_panels
    ):
        raise ComponentClosureEvaluationError(
            "evaluation result base has duplicate panel qualification"
        )
    suffix = (f"seed{seed}", condition.lower(), str(checkpoint))
    if not remote_qualified:
        suffix = (panel, *suffix)
    remote = remote_base.joinpath(*suffix)
    controller = controller_base.joinpath(*suffix)
    return str(remote), str(controller)


def _read_self_hashed(path: Path, field: str) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComponentClosureEvaluationError(f"cannot read parent {path}") from error
    if not isinstance(value, dict):
        raise ComponentClosureEvaluationError(f"parent {path} is not an object")
    claimed = value.get(field)
    body = {key: item for key, item in value.items() if key != field}
    if not isinstance(claimed, str) or canonical_sha256(body) != claimed:
        raise ComponentClosureEvaluationError(f"parent {path} self-hash changed")
    return value, hashlib.sha256(payload).hexdigest()


def _required_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ComponentClosureEvaluationError(f"{name} is not a SHA-256")
    return value


def _nullable_sha256(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _required_sha256(value, name=name)


def _score_cell_target_identity(
    result_path: Path,
    *,
    result: Mapping[str, Any],
    scores: Any,
    packed_sequence_count: int | None = None,
) -> dict[str, Any]:
    """Bind one score row to its ordered held-out tokens and target basis."""

    result_token_identity = _required_sha256(
        result.get("token_identity_sha256"),
        name="evaluation result token identity",
    )
    result_target_manifest = _required_sha256(
        result.get("target_manifest_sha256"),
        name="evaluation result target manifest",
    )
    result_recomposition = _required_sha256(
        result.get("recomposition_sidecar_sha256"),
        name="evaluation result recomposition sidecar",
    )
    result_centering = _required_sha256(
        result.get("centering_receipt_sha256"),
        name="evaluation result centering receipt",
    )
    result_source_manifest = _nullable_sha256(
        result.get("source_manifest_sha256"),
        name="evaluation result source manifest",
    )
    tokens = result.get("tokens")
    packed_sequences = result.get("packed_sequences")
    if (
        isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or tokens <= 0
        or isinstance(packed_sequences, bool)
        or not isinstance(packed_sequences, int)
        or packed_sequences <= 0
        or packed_sequences > tokens
    ):
        raise ComponentClosureEvaluationError(
            "evaluation result token counts are invalid"
        )
    canonical_token_identity = token_identity_sha256(
        scores.sequence_id,
        scores.position,
    )
    observed_packed_sequences = (
        scores.packed_sequences
        if packed_sequence_count is None
        else packed_sequence_count
    )
    if (
        result_token_identity != scores.token_identity_sha256
        or result_token_identity != canonical_token_identity
        or result_target_manifest != scores.capture_manifest_sha256
        or result_recomposition != scores.recomposition_sidecar_sha256
        or tokens != scores.tokens
        or packed_sequences != observed_packed_sequences
    ):
        raise ComponentClosureEvaluationError(
            "evaluation result and loaded scores target identity differ"
        )
    receipt_path = result_path.parent / "centering-receipt.json"
    receipt, _receipt_file_sha256 = _read_self_hashed(
        receipt_path,
        "receipt_sha256",
    )
    expected_receipt_fields = {
        "schema_version",
        "protocol_id",
        "panel",
        "config_sha256",
        "target_manifest_sha256",
        "source_manifest_sha256",
        "recomposition_sidecar_sha256",
        "target_mean_sha256",
        "source_mean_sha256",
        "routed_mean_sha256",
        "oracle_shared_mean_sha256",
        "endpoint_receipt_sha256",
        "endpoint_cache_key_sha256",
        "receipt_sha256",
    }
    if set(receipt) != expected_receipt_fields:
        raise ComponentClosureEvaluationError(
            "centering receipt fields changed"
        )
    receipt_sha256 = _required_sha256(
        receipt.get("receipt_sha256"),
        name="centering receipt identity",
    )
    centering_target_manifest = _required_sha256(
        receipt.get("target_manifest_sha256"),
        name="centering target manifest",
    )
    centering_recomposition = _required_sha256(
        receipt.get("recomposition_sidecar_sha256"),
        name="centering recomposition sidecar",
    )
    source_manifest = _nullable_sha256(
        receipt.get("source_manifest_sha256"),
        name="centering source manifest",
    )
    source_mean = _nullable_sha256(
        receipt.get("source_mean_sha256"),
        name="centering source mean",
    )
    required_centering = {
        "target_mean_sha256": _required_sha256(
            receipt.get("target_mean_sha256"),
            name="centering target mean",
        ),
        "endpoint_receipt_sha256": _required_sha256(
            receipt.get("endpoint_receipt_sha256"),
            name="centering endpoint receipt",
        ),
        "endpoint_cache_key_sha256": _required_sha256(
            receipt.get("endpoint_cache_key_sha256"),
            name="centering endpoint cache key",
        ),
        "routed_mean_sha256": _required_sha256(
            receipt.get("routed_mean_sha256"),
            name="centering routed mean",
        ),
        "oracle_shared_mean_sha256": _required_sha256(
            receipt.get("oracle_shared_mean_sha256"),
            name="centering oracle shared mean",
        ),
    }
    _required_sha256(
        receipt.get("config_sha256"),
        name="centering config",
    )
    if (
        receipt_sha256 != result_centering
        or receipt.get("protocol_id") != result.get("protocol_id")
        or receipt.get("panel") != result.get("panel")
        or centering_recomposition != result_recomposition
        or (source_manifest is None) is not (source_mean is None)
    ):
        raise ComponentClosureEvaluationError(
            "centering receipt and evaluation target identity differ"
        )
    return {
        "evaluation_token_identity_sha256": result_token_identity,
        "evaluation_target_manifest_sha256": result_target_manifest,
        "evaluation_token_count": tokens,
        "evaluation_packed_sequences": packed_sequences,
        "evaluation_source_manifest_sha256": result_source_manifest,
        "centering_receipt_sha256": receipt_sha256,
        "centering_target_manifest_sha256": centering_target_manifest,
        **required_centering,
        "recomposition_sidecar_sha256": centering_recomposition,
        "source_manifest_sha256": source_manifest,
        "source_mean_sha256": source_mean,
    }


def _fresh_cells(training: Mapping[str, Any]) -> dict[tuple[str, int, str, int], Any]:
    cells: dict[tuple[str, int, str, int], Any] = {}
    for task in training["tasks"]:
        result_path = Path(task["completion_path_controller"])
        result, result_file_sha = _read_self_hashed(result_path, "result_sha256")
        identity = result.get("run_identity")
        plan = identity.get("plan") if isinstance(identity, Mapping) else None
        if (
            result.get("tokens") != 10_000_000
            or not isinstance(plan, Mapping)
            or plan.get("condition") != task["condition"]
            or plan.get("seed") != task["seed"]
            or plan.get("phase_name") != PHASE_NAME
        ):
            raise ComponentClosureEvaluationError(
                f"fresh parent role changed: {task['task_id']}"
            )
        run_root = result_path.parent
        contract, contract_file_sha = _read_self_hashed(
            run_root / "run-contract.json",
            "run_contract_sha256",
        )
        config_path = Path(training["configs"][task["panel"]]["controller_path"])
        try:
            config_file_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
        except OSError as error:
            raise ComponentClosureEvaluationError(
                f"cannot hash fresh parent config {config_path}"
            ) from error
        if (
            contract.get("project_commit") != training["project_commit"]
            or contract.get("phase") != PHASE_NAME
            or contract.get("condition") != task["condition"]
            or contract.get("seed") != task["seed"]
            or contract.get("config_file_sha256") != config_file_sha
            or contract.get("checkpoint_payload_root") != task["checkpoint_root_gcs"]
        ):
            raise ComponentClosureEvaluationError(
                f"fresh run contract changed: {task['task_id']}"
            )
        remote_run_root = str(
            PurePosixPath(task["control_root_remote"]) / "runs" / task["run_id"]
        )
        for checkpoint in CHECKPOINT_TOKENS:
            pointer_path = milestone_pointer_path(
                run_root / "checkpoints",
                checkpoint,
            )
            pointer, pointer_file_sha = _read_self_hashed(
                pointer_path,
                "pointer_sha256",
            )
            try:
                checked = validate_pointer(
                    pointer,
                    expected_kind="latest",
                    payload_root=task["checkpoint_root_gcs"],
                    run_identity=identity,
                )
            except (TypeError, ValueError, RuntimeError) as error:
                raise ComponentClosureEvaluationError(
                    f"fresh checkpoint is invalid: {task['task_id']}/{checkpoint}"
                ) from error
            batch = int(plan["global_microbatch_tokens"]) * int(
                plan["microbatches_per_update"]
            )
            step = math.ceil(checkpoint / batch)
            if checked["step"] != step or checked["tokens_seen"] != min(
                step * batch, 10_000_000
            ):
                raise ComponentClosureEvaluationError(
                    f"fresh checkpoint progress changed: {task['task_id']}"
                )
            cells[(task["panel"], task["seed"], task["condition"], checkpoint)] = {
                "logical_condition": task["condition"],
                "source_condition": task["condition"],
                "source_protocol_id": PROTOCOL_ID,
                "source_phase_name": PHASE_NAME,
                "source_config": task["config_remote"],
                "training_source_revision": contract["project_commit"],
                "run_root": remote_run_root,
                "checkpoint_payload_root": task["checkpoint_root_gcs"],
                "parent_training_result_sha256": result["result_sha256"],
                "parent_training_result_file_sha256": result_file_sha,
                "parent_run_contract_sha256": contract["run_contract_sha256"],
                "parent_run_contract_file_sha256": contract_file_sha,
                "parent_checkpoint_pointer_sha256": checked["pointer_sha256"],
                "parent_checkpoint_pointer_file_sha256": pointer_file_sha,
                "checkpoint_tokens": checkpoint,
                "reference_kind": "none",
            }
    return cells


def validate_evaluation_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ComponentClosureEvaluationError("evaluation plan must be a mapping")
    selected = copy.deepcopy(dict(value))
    expected = {
        "schema_version",
        "protocol_id",
        "evaluation_source_revision",
        "training_plan_sha256",
        "a_reference_manifest_sha256",
        "state_root",
        "bundles",
        "plan_sha256",
    }
    if set(selected) != expected:
        raise ComponentClosureEvaluationError("evaluation plan fields changed")
    claimed = selected.pop("plan_sha256")
    if (
        selected["schema_version"] != PLAN_SCHEMA_VERSION
        or selected["protocol_id"] != PROTOCOL_ID
        or _GIT_RE.fullmatch(str(selected["evaluation_source_revision"])) is None
        or not isinstance(claimed, str)
        or canonical_sha256(selected) != claimed
    ):
        raise ComponentClosureEvaluationError("evaluation plan identity changed")
    bundles = selected["bundles"]
    if not isinstance(bundles, list) or len(bundles) != BUNDLE_COUNT:
        raise ComponentClosureEvaluationError("evaluation requires six bundles")
    expected_bundles = [(panel, seed) for panel in PANELS for seed in SEEDS]
    roles: list[tuple[str, int]] = []
    all_cells: list[tuple[str, int, str, int]] = []
    result_roots_remote: list[str] = []
    result_roots_controller: list[str] = []
    expected_result_roots: list[tuple[str, str, str, str]] = []
    for index, bundle in enumerate(bundles):
        if not isinstance(bundle, Mapping) or set(bundle) != {
            "schema_version",
            "protocol_id",
            "resource_id",
            "panel",
            "seed",
            "source_revision",
            "runtime",
            "cells",
            "bundle_sha256",
        }:
            raise ComponentClosureEvaluationError(
                f"evaluation bundle {index} fields changed"
            )
        bundle_body = {
            key: item for key, item in bundle.items() if key != "bundle_sha256"
        }
        if (
            bundle["schema_version"] != BUNDLE_SCHEMA_VERSION
            or bundle["protocol_id"] != PROTOCOL_ID
            or canonical_sha256(bundle_body) != bundle["bundle_sha256"]
        ):
            raise ComponentClosureEvaluationError(
                f"evaluation bundle {index} identity changed"
            )
        panel, seed = bundle["panel"], bundle["seed"]
        runtime_value = bundle["runtime"]
        if (
            not isinstance(bundle["resource_id"], str)
            or not bundle["resource_id"]
            or not isinstance(runtime_value, Mapping)
            or runtime_value.get("resource_id") != bundle["resource_id"]
        ):
            raise ComponentClosureEvaluationError(
                f"evaluation bundle {index} resource binding changed"
            )
        _validated_runtime(
            runtime_value,
            resource_id=bundle["resource_id"],
            panel=str(panel),
            accelerator_type=str(runtime_value.get("accelerator_type")),
        )
        roles.append((panel, seed))
        cells = bundle["cells"]
        expected_cells = [
            (condition, checkpoint)
            for condition in CONDITIONS
            for checkpoint in CHECKPOINT_TOKENS
        ]
        if (
            not isinstance(cells, list)
            or not all(isinstance(cell, Mapping) for cell in cells)
            or [
                (cell.get("logical_condition"), cell.get("checkpoint_tokens"))
                for cell in cells
            ]
            != expected_cells
        ):
            raise ComponentClosureEvaluationError(
                f"evaluation bundle {index} cell matrix changed"
            )
        for cell, (condition, checkpoint) in zip(cells, expected_cells, strict=True):
            remote = cell.get("result_root")
            controller = cell.get("result_root_controller")
            remote_path = PurePosixPath(str(remote))
            controller_path = Path(str(controller))
            if (
                not isinstance(remote, str)
                or not remote_path.is_absolute()
                or ".." in remote_path.parts
                or not isinstance(controller, str)
                or not controller_path.is_absolute()
                or ".." in controller_path.parts
            ):
                raise ComponentClosureEvaluationError(
                    f"evaluation bundle {index} cell result root is invalid"
                )
            expected_remote, expected_controller = _canonical_cell_result_roots(
                runtime_value,
                panel=str(panel),
                seed=int(seed),
                condition=condition,
                checkpoint=checkpoint,
            )
            result_roots_remote.append(remote)
            result_roots_controller.append(controller)
            expected_result_roots.append(
                (remote, controller, expected_remote, expected_controller)
            )
        all_cells.extend(
            (panel, seed, condition, checkpoint)
            for condition, checkpoint in expected_cells
        )
    if roles != expected_bundles or len(all_cells) != CELL_COUNT:
        raise ComponentClosureEvaluationError("evaluation matrix is not exact")
    if (
        len(result_roots_remote) != CELL_COUNT
        or len(set(result_roots_remote)) != CELL_COUNT
        or len(result_roots_controller) != CELL_COUNT
        or len(set(result_roots_controller)) != CELL_COUNT
    ):
        raise ComponentClosureEvaluationError("evaluation cell result roots collide")
    for (
        remote,
        controller,
        expected_remote,
        expected_controller,
    ) in expected_result_roots:
        if remote != expected_remote or controller != expected_controller:
            raise ComponentClosureEvaluationError(
                "evaluation cell result root is not canonically panel-qualified"
            )
    return {**selected, "plan_sha256": claimed}


def build_evaluation_plan(
    training_plan: Mapping[str, Any],
    a_reference_manifest: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    evaluation_source_revision: str,
    state_root: str | Path,
) -> dict[str, Any]:
    training = validate_training_plan(training_plan)
    a_manifest = validate_a_reference_manifest(a_reference_manifest)
    if (
        _GIT_RE.fullmatch(evaluation_source_revision) is None
        or evaluation_source_revision != training["project_commit"]
    ):
        raise ComponentClosureEvaluationError("invalid evaluation source revision")
    fresh = _fresh_cells(training)
    a_cells = {
        (row["panel"], row["seed"], row["checkpoint_tokens"]): row
        for row in a_manifest["cells"]
    }
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "assignments",
        "resources",
    }:
        raise ComponentClosureEvaluationError(
            "evaluation runtime requires assignments and resources"
        )
    assignments = runtime["assignments"]
    resource_runtime = runtime["resources"]
    if (
        not isinstance(assignments, Mapping)
        or set(assignments)
        != {f"{panel}.seed{seed}" for panel in PANELS for seed in SEEDS}
        or not isinstance(resource_runtime, Mapping)
    ):
        raise ComponentClosureEvaluationError(
            "evaluation runtime assignments are incomplete"
        )
    training_resources = {row["id"]: row for row in training["resources"]}
    cohorts = training["cohorts"]
    bundles: list[dict[str, Any]] = []
    for panel in PANELS:
        for seed in SEEDS:
            assignment_key = f"{panel}.seed{seed}"
            resource_id = assignments[assignment_key]
            compatible = cohorts[assignment_key]
            if (
                not isinstance(resource_id, str)
                or resource_id not in training_resources
                or resource_id not in compatible
                or resource_id not in resource_runtime
            ):
                raise ComponentClosureEvaluationError(
                    f"evaluation assignment {assignment_key} is incompatible"
                )
            accelerator_type = training_resources[resource_id]["accelerator_type"]
            panel_runtime = _validated_runtime(
                resource_runtime[resource_id],
                resource_id=resource_id,
                panel=panel,
                accelerator_type=accelerator_type,
            )
            resource_row = training_resources[resource_id]
            resource_environment = resource_row["environment"]
            plane = training["planes"][resource_id]
            try:
                PurePosixPath(panel_runtime["result_root_remote"]).relative_to(
                    PurePosixPath(plane["remote_artifact_root"])
                )
                Path(panel_runtime["result_root_controller"]).relative_to(
                    Path(plane["controller_artifact_root"])
                )
            except ValueError as error:
                raise ComponentClosureEvaluationError(
                    f"evaluation results cross {resource_id}'s artifact plane"
                ) from error
            mirror = resource_row["evidence"]["mirror_seal"]
            expected_mirror = (
                mirror["remote_path"] if isinstance(mirror, Mapping) else None
            )
            expected_runtime = {
                "train_target_root": resource_environment[
                    "CONTRIBUTION_TRAIN_CAPTURE_ROOT"
                ],
                "train_target_object_prefix": resource_environment[
                    "CONTRIBUTION_OBJECT_PREFIX"
                ],
                "cache_root": resource_environment["CONTRIBUTION_SAE_CACHE_ROOT"],
                "recomposition_sidecar": resource_environment[
                    "CONTRIBUTION_RECOMPOSITION_SIDECAR"
                ],
                "regional_mirror_seal": expected_mirror,
            }
            if any(
                panel_runtime[name] != expected
                for name, expected in expected_runtime.items()
            ):
                raise ComponentClosureEvaluationError(
                    f"evaluation runtime crosses {resource_id}'s capture/cache plane"
                )
            if panel == "transcoder" and (
                panel_runtime["train_source_root"]
                != plane["source_overlay_control_root_remote"]
                or panel_runtime["train_source_object_prefix"]
                != plane["source_overlay_object_prefix_gcs"]
            ):
                raise ComponentClosureEvaluationError(
                    f"evaluation runtime crosses {resource_id}'s source plane"
                )
            cells: list[dict[str, Any]] = []
            for condition in CONDITIONS:
                for checkpoint in CHECKPOINT_TOKENS:
                    if condition == "A":
                        row = a_cells[(panel, seed, checkpoint)]
                        cell = {
                            "logical_condition": "A",
                            "source_condition": SOURCE_CONDITION,
                            "source_protocol_id": SOURCE_PROTOCOL_ID,
                            "source_phase_name": SOURCE_PHASE_NAME,
                            "source_config": panel_runtime["a_config_remote"],
                            "training_source_revision": row["training_source_revision"],
                            "run_root": row["run_root_remote"],
                            "checkpoint_payload_root": row[
                                "checkpoint_payload_root_gcs"
                            ],
                            "parent_training_result_sha256": row[
                                "parent_training_result_sha256"
                            ],
                            "parent_training_result_file_sha256": row[
                                "parent_training_result_file_sha256"
                            ],
                            "parent_run_contract_sha256": row[
                                "parent_run_contract_sha256"
                            ],
                            "parent_run_contract_file_sha256": row[
                                "parent_run_contract_file_sha256"
                            ],
                            "parent_checkpoint_pointer_sha256": row[
                                "parent_checkpoint_pointer_sha256"
                            ],
                            "parent_checkpoint_pointer_file_sha256": row[
                                "parent_checkpoint_pointer_file_sha256"
                            ],
                            "checkpoint_tokens": checkpoint,
                            "reference_kind": "trained_v3",
                        }
                    else:
                        cell = fresh[(panel, seed, condition, checkpoint)]
                    result_root, result_root_controller = _canonical_cell_result_roots(
                        panel_runtime,
                        panel=panel,
                        seed=seed,
                        condition=condition,
                        checkpoint=checkpoint,
                    )
                    cell = {
                        **cell,
                        "result_root": result_root,
                        "result_root_controller": result_root_controller,
                    }
                    cells.append(cell)
            bundle_body = {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "resource_id": resource_id,
                "panel": panel,
                "seed": seed,
                "source_revision": evaluation_source_revision,
                "runtime": copy.deepcopy(dict(panel_runtime)),
                "cells": cells,
            }
            bundles.append(
                {
                    **bundle_body,
                    "bundle_sha256": canonical_sha256(bundle_body),
                }
            )
    body = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "evaluation_source_revision": evaluation_source_revision,
        "training_plan_sha256": training["plan_sha256"],
        "a_reference_manifest_sha256": a_manifest["manifest_sha256"],
        "state_root": str(Path(state_root).expanduser().resolve()),
        "bundles": bundles,
    }
    return validate_evaluation_plan({**body, "plan_sha256": canonical_sha256(body)})


def install_resident_evaluator_adapter() -> Any:
    """Teach the mature ladder evaluator only the new config and R/B/F layout."""

    from moe_sae import experiment5_routed_ladder_evaluation_worker as worker
    from moe_sae import experiment5_routed_ladder as ladder
    from moe_sae.contribution_sae import ContributionSAEConfig

    original_loader = worker._load_routed_ladder_config
    original_phase = worker._phase_name
    original_conditions = worker._evaluated_conditions
    original_layout = worker._layout

    def loader(path: str | Path, *, expected_panel: str) -> Any:
        try:
            return load_component_closure_config(
                path,
                expected_panel=expected_panel,  # type: ignore[arg-type]
            )
        except ComponentClosureConfigError:
            return original_loader(path, expected_panel=expected_panel)

    def phase(config: Any) -> str:
        return (
            PHASE_NAME if config.protocol_id == PROTOCOL_ID else original_phase(config)
        )

    def conditions(config: Any) -> tuple[str, ...]:
        return (
            CONDITIONS
            if config.protocol_id == PROTOCOL_ID
            else original_conditions(config)
        )

    def layout(config: Any, condition: str, seed: int) -> Any:
        if config.protocol_id != PROTOCOL_ID:
            return original_layout(config, condition, seed)
        selected = config.layout_for_condition(condition)
        return ContributionSAEConfig(
            condition=condition,
            hidden_dim=config.model.hidden_size,
            num_experts=config.model.num_routed_experts,
            top_routed=config.model.experts_per_token,
            bank_width=selected.bank_width,
            k_per_contribution=selected.topk_per_component,
            randomization_key=seed,
        )

    worker._load_routed_ladder_config = loader
    worker._phase_name = phase
    worker._evaluated_conditions = conditions
    worker._layout = layout
    worker.EVALUATED_CONDITIONS = tuple(
        dict.fromkeys((*worker.EVALUATED_CONDITIONS, *CONDITIONS))
    )
    worker.PROTOCOL_IDS = tuple(dict.fromkeys((*worker.PROTOCOL_IDS, PROTOCOL_ID)))
    ladder.TOKEN_SCORE_CONDITIONS = tuple(
        dict.fromkeys((*ladder.TOKEN_SCORE_CONDITIONS, *CONDITIONS))
    )
    return worker


def _canonical_cell_completion(
    worker: Any,
    bundle: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the exact immutable completion or ``None`` when truly absent."""

    root = Path(str(cell["result_root"]))
    result_path = root / "evaluation-result.json"
    if not root.exists():
        return None
    if (
        root.is_symlink()
        or not root.is_dir()
        or result_path.is_symlink()
        or not result_path.is_file()
    ):
        raise ComponentClosureEvaluationError(
            "canonical evaluation cell is partial: "
            f"{bundle['panel']}/{cell['logical_condition']}/"
            f"{bundle['seed']}/{cell['checkpoint_tokens']}"
        )
    scores = worker.load_merged_scores(result_path)
    result, result_file_sha = _read_self_hashed(result_path, "result_sha256")
    if (
        result.get("protocol_id") != cell["source_protocol_id"]
        or result.get("panel") != bundle["panel"]
        or result.get("condition") != cell["source_condition"]
        or result.get("seed") != bundle["seed"]
        or result.get("checkpoint_tokens") != cell["checkpoint_tokens"]
        or result.get("source_revision") != bundle["source_revision"]
        or result.get("training_source_revision") != cell["training_source_revision"]
        or result.get("parent_training_result_sha256")
        != cell["parent_training_result_sha256"]
        or result.get("parent_training_result_file_sha256")
        != cell["parent_training_result_file_sha256"]
        or result.get("parent_run_contract_sha256")
        != cell["parent_run_contract_sha256"]
        or result.get("parent_run_contract_file_sha256")
        != cell["parent_run_contract_file_sha256"]
        or result.get("parent_checkpoint_pointer_sha256")
        != cell["parent_checkpoint_pointer_sha256"]
        or result.get("parent_checkpoint_pointer_file_sha256")
        != cell["parent_checkpoint_pointer_file_sha256"]
        or scores.panel != bundle["panel"]
        or scores.condition != cell["source_condition"]
        or scores.seed != bundle["seed"]
        or scores.checkpoint_tokens != cell["checkpoint_tokens"]
        or scores.source_revision != bundle["source_revision"]
        or scores.score_artifact_sha256 != result.get("score_artifact_sha256")
    ):
        raise ComponentClosureEvaluationError(
            "canonical evaluation cell identity changed: "
            f"{bundle['panel']}/{cell['logical_condition']}/"
            f"{bundle['seed']}/{cell['checkpoint_tokens']}"
        )
    return {
        "logical_condition": cell["logical_condition"],
        "source_condition": cell["source_condition"],
        "checkpoint_tokens": cell["checkpoint_tokens"],
        "result_sha256": result["result_sha256"],
        "result_file_sha256": result_file_sha,
        "score_artifact_sha256": result["score_artifact_sha256"],
    }


def partition_bundle_retry(
    bundle: Mapping[str, Any],
    inspect_cell: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Partition a bundle into completed and pending indexes for safe retry."""

    completed: list[int] = []
    pending: list[int] = []
    for index, cell in enumerate(bundle["cells"]):
        if inspect_cell(cell) is None:
            pending.append(index)
        else:
            completed.append(index)
    return tuple(completed), tuple(pending)


def run_resident_bundle(
    bundle: Mapping[str, Any],
    *,
    process_index: int,
    attempt_id: str,
) -> int:
    """Run one mixed-parent 12-cell bundle in one initialized JAX process."""

    # Plan validation is normally done at the controller boundary; retain an
    # exact self-hash check here because this function is accelerator-facing.
    body = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION or canonical_sha256(
        body
    ) != bundle.get("bundle_sha256"):
        raise ComponentClosureEvaluationError("bundle self-hash changed")
    worker = install_resident_evaluator_adapter()
    import jax
    from moe_sae.runtime import initialize_jax_distributed, require_accelerator_runtime

    runtime_spec = bundle["runtime"]
    initialize_jax_distributed()
    require_accelerator_runtime(
        runtime_spec["accelerator_type"],
        peak_flops_per_second=int(runtime_spec["peak_flops_per_second"]),
    )
    if int(jax.process_index()) != process_index:
        raise ComponentClosureEvaluationError("controller/process mapping changed")
    residents = {
        SOURCE_PROTOCOL_ID: worker._ResidentRuntime(),
        PROTOCOL_ID: worker._ResidentRuntime(),
    }
    from moe_sae.contribution_training_worker import (
        _collective_barrier,
        _collective_records,
    )

    original_v3_protocol = worker.V3_PROTOCOL_ID
    try:
        for index, cell in enumerate(bundle["cells"]):
            worker.V3_PROTOCOL_ID = (
                SOURCE_PROTOCOL_ID if cell["logical_condition"] == "A" else PROTOCOL_ID
            )
            canonical_root = Path(cell["result_root"])
            existing = _canonical_cell_completion(worker, bundle, cell)
            existing_rows = _collective_records(
                {
                    "process_index": int(jax.process_index()),
                    "result_sha256": (
                        existing["result_sha256"] if existing is not None else None
                    ),
                }
            )
            observed = {row["result_sha256"] for row in existing_rows}
            if len(observed) != 1:
                raise ComponentClosureEvaluationError(
                    f"processes disagree on canonical cell {index}"
                )
            if existing is not None:
                _collective_barrier(f"component-closure-skip-complete-{index}")
                continue
            result_root = (
                canonical_root.parent / ".attempts" / canonical_root.name / attempt_id
            )
            args = argparse.Namespace(
                config=cell["source_config"],
                panel=bundle["panel"],
                condition=cell["source_condition"],
                h_config=None,
                a256_config=None,
                completed_config=None,
                reference_run_contract_sha256=None,
                reference_run_contract_file_sha256=None,
                parent_training_result_sha256=cell["parent_training_result_sha256"],
                parent_training_result_file_sha256=cell[
                    "parent_training_result_file_sha256"
                ],
                parent_run_contract_sha256=cell["parent_run_contract_sha256"],
                parent_run_contract_file_sha256=cell["parent_run_contract_file_sha256"],
                parent_checkpoint_pointer_sha256=cell[
                    "parent_checkpoint_pointer_sha256"
                ],
                parent_checkpoint_pointer_file_sha256=cell[
                    "parent_checkpoint_pointer_file_sha256"
                ],
                seed=bundle["seed"],
                checkpoint_tokens=cell["checkpoint_tokens"],
                source_revision=bundle["source_revision"],
                training_source_revision=cell["training_source_revision"],
                train_target_root=runtime_spec["train_target_root"],
                eval_target_root=runtime_spec["eval_target_root"],
                train_target_object_prefix=runtime_spec["train_target_object_prefix"],
                eval_target_object_prefix=runtime_spec["eval_target_object_prefix"],
                train_source_root=runtime_spec.get("train_source_root"),
                eval_source_root=runtime_spec.get("eval_source_root"),
                train_source_object_prefix=runtime_spec.get(
                    "train_source_object_prefix"
                ),
                eval_source_object_prefix=runtime_spec.get("eval_source_object_prefix"),
                cache_root=(
                    f"{runtime_spec['cache_root']}/" f"process-p{process_index:05d}"
                ),
                cache_max_bytes=int(runtime_spec["cache_max_bytes"]),
                regional_mirror_seal=runtime_spec.get("regional_mirror_seal"),
                run_root=cell["run_root"],
                checkpoint_payload_root=cell["checkpoint_payload_root"],
                recomposition_sidecar=runtime_spec["recomposition_sidecar"],
                result_root=str(result_root),
                bundle_spec_json=None,
                attempt_id=attempt_id,
                process_index=process_index,
                expected_token_identity_sha256=runtime_spec.get(
                    "expected_token_identity_sha256"
                ),
                accelerator_type=runtime_spec["accelerator_type"],
                peak_flops_per_second=int(runtime_spec["peak_flops_per_second"]),
                reference_kind=cell["reference_kind"],
            )
            worker.run(
                args,
                resident_runtime=residents[cell["source_protocol_id"]],
                clear_jax_caches=False,
            )
            _collective_barrier(f"component-closure-promote-{index}")
            if int(jax.process_index()) == 0:
                if canonical_root.exists():
                    raise ComponentClosureEvaluationError(
                        f"canonical result already exists: {canonical_root}"
                    )
                attempt_cell = {**cell, "result_root": str(result_root)}
                if _canonical_cell_completion(worker, bundle, attempt_cell) is None:
                    raise ComponentClosureEvaluationError(
                        f"attempt did not seal cell {index}"
                    )
                canonical_root.parent.mkdir(parents=True, exist_ok=True)
                os.rename(result_root, canonical_root)
            _collective_barrier(f"component-closure-promoted-{index}")
            if _canonical_cell_completion(worker, bundle, cell) is None:
                raise ComponentClosureEvaluationError(
                    f"promoted canonical cell {index} is absent"
                )
    finally:
        worker.V3_PROTOCOL_ID = original_v3_protocol
    return 0


def load_evaluation_plan(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComponentClosureEvaluationError("cannot read evaluation plan") from error
    return validate_evaluation_plan(value)


def collect_score_grid(
    evaluation_plan: Mapping[str, Any],
    costs: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Convert the 72 immutable cell artifacts into report-ready score rows."""

    import numpy as np

    from moe_sae.component_closure_report import (
        EVIDENCE_SCHEMA_VERSION,
        validate_score_grid,
    )

    plan = validate_evaluation_plan(evaluation_plan)
    worker = install_resident_evaluator_adapter()
    original_v3_protocol = worker.V3_PROTOCOL_ID
    cells: list[dict[str, Any]] = []
    try:
        for bundle in plan["bundles"]:
            for cell in bundle["cells"]:
                worker.V3_PROTOCOL_ID = cell["source_protocol_id"]
                result_path = (
                    Path(cell["result_root_controller"]) / "evaluation-result.json"
                )
                scores = worker.load_merged_scores(result_path)
                if (
                    scores.panel != bundle["panel"]
                    or scores.seed != bundle["seed"]
                    or scores.checkpoint_tokens != cell["checkpoint_tokens"]
                    or scores.condition != cell["source_condition"]
                ):
                    raise ComponentClosureEvaluationError(
                        f"score artifact role changed: {result_path}"
                    )
                sequence, inverse = np.unique(
                    np.asarray(scores.sequence_id, dtype=np.int64),
                    return_inverse=True,
                )
                sse = np.bincount(
                    inverse,
                    weights=np.asarray(
                        scores.endpoint_sse["component"],
                        dtype=np.float64,
                    ),
                    minlength=sequence.size,
                )
                sst = np.bincount(
                    inverse,
                    weights=np.asarray(
                        scores.endpoint_sst["component"],
                        dtype=np.float64,
                    ),
                    minlength=sequence.size,
                )
                result, _file_sha = _read_self_hashed(
                    result_path,
                    "result_sha256",
                )
                target_identity = _score_cell_target_identity(
                    result_path,
                    result=result,
                    scores=scores,
                    packed_sequence_count=int(sequence.size),
                )
                cells.append(
                    {
                        "panel": bundle["panel"],
                        "seed": bundle["seed"],
                        "checkpoint_tokens": cell["checkpoint_tokens"],
                        "condition": cell["logical_condition"],
                        "sequence_id": sequence.tolist(),
                        "component_sse": sse.tolist(),
                        "component_sst": sst.tolist(),
                        "evaluation_result_sha256": result["result_sha256"],
                        **target_identity,
                    }
                )
    finally:
        worker.V3_PROTOCOL_ID = original_v3_protocol
    panel_order = {value: index for index, value in enumerate(PANELS)}
    condition_order = {value: index for index, value in enumerate(CONDITIONS)}
    cells.sort(
        key=lambda row: (
            panel_order[row["panel"]],
            row["seed"],
            row["checkpoint_tokens"],
            condition_order[row["condition"]],
        )
    )
    body = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "cells": cells,
        "costs": copy.deepcopy(costs),
    }
    return validate_score_grid({**body, "evidence_sha256": canonical_sha256(body)})


def collect_training_costs(
    training_plan: Mapping[str, Any],
    a_reference_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Read the 24 parent results into the report's exact timing boundary."""

    training = validate_training_plan(training_plan)
    references = validate_a_reference_manifest(a_reference_manifest)
    a_rows = {
        (row["panel"], row["seed"]): row
        for row in references["cells"]
        if row["checkpoint_tokens"] == CHECKPOINT_TOKENS[0]
    }
    fresh_tasks = {
        (row["panel"], row["seed"], row["condition"]): row for row in training["tasks"]
    }
    resources = {row["id"]: row["accelerator_type"] for row in training["resources"]}
    rows: list[dict[str, Any]] = []
    for panel in PANELS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                if condition == "A":
                    source = a_rows[(panel, seed)]
                    result_path = Path(source["training_result_controller_path"])
                    expected_accelerator = source["accelerator_type"]
                else:
                    task = fresh_tasks[(panel, seed, condition)]
                    result_path = Path(task["completion_path_controller"])
                    expected_accelerator = resources[task["eligible_resource_ids"][0]]
                result, _file_sha = _read_self_hashed(
                    result_path,
                    "result_sha256",
                )
                contract, _contract_file_sha = _read_self_hashed(
                    result_path.parent / "run-contract.json",
                    "run_contract_sha256",
                )
                wall = float(result["marginal_wall_seconds"])
                chip_seconds = float(result["chip_seconds"])
                chip_count = int(round(chip_seconds / wall))
                if (
                    wall <= 0
                    or chip_seconds <= 0
                    or chip_count <= 0
                    or not math.isclose(
                        chip_seconds,
                        wall * chip_count,
                        rel_tol=1e-9,
                        abs_tol=1e-6,
                    )
                    or contract.get("accelerator_type") != expected_accelerator
                ):
                    raise ComponentClosureEvaluationError(
                        f"training cost identity changed: {panel}/{condition}/{seed}"
                    )
                rows.append(
                    {
                        "panel": panel,
                        "seed": seed,
                        "condition": condition,
                        "accelerator_type": expected_accelerator,
                        "chip_count": chip_count,
                        "marginal_wall_seconds": wall,
                        "chip_seconds": chip_seconds,
                        "chip_seconds_scope": result["chip_seconds_scope"],
                        "chip_seconds_excludes": result["chip_seconds_excludes"],
                        "training_source_revision": contract["project_commit"],
                    }
                )
    return rows


__all__ = [
    "BUNDLE_COUNT",
    "BUNDLE_SCHEMA_VERSION",
    "CELL_COUNT",
    "PLAN_SCHEMA_VERSION",
    "ComponentClosureEvaluationError",
    "build_evaluation_plan",
    "collect_score_grid",
    "collect_training_costs",
    "install_resident_evaluator_adapter",
    "load_evaluation_plan",
    "partition_bundle_retry",
    "run_resident_bundle",
    "validate_evaluation_plan",
]
