"""Immutable provenance correction for Experiment-5 v3 ``A256Native``.

The frozen v3 configuration declares one historical semantic config hash for
each panel.  The SAE declaration agrees with the completed Experiment-4
trajectory.  The transcoder declaration does not: the completed historical
trajectory is a Softplus A256 run, but its actual finalized config has a
different semantic hash.

This module does not rewrite the v3 configuration or relabel the historical
run.  It builds a create-only addendum which records both the declaration and
the observed lineage, pins every config/contract/result/checkpoint artifact,
and limits ``A256Native`` to panel-native descriptive context.  The addendum
cannot enter the matched A512 primary estimand.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from moe_sae.checkpointing import milestone_pointer_path, validate_pointer
from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.direct_horizon_campaign import (
    PROTOCOL_ID as EXPERIMENT4_PROTOCOL_ID,
    DirectCampaignPlan,
    inspect_training_completion as inspect_horizon_training_completion,
)
from moe_sae.experiment5_campaign import (
    EvidencePath,
    _evidence,
    _exact_mapping,
    _git_sha,
    _read_self_hashed_json,
    _remote_path,
    _safe_id,
    _sha256,
)
from moe_sae.experiment5_routed_ladder_campaign import (
    RoutedLadderCampaignPlanError,
    RoutedLadderCampaignStateError,
)
from moe_sae.experiment5_routed_ladder_v3_config import (
    A256_NATIVE_ACTIVATION,
    A256_NATIVE_CONFIG_SHA256,
    A256_NATIVE_LABEL,
    A256_NATIVE_SELECTION_POLICY,
    PROTOCOL_ID,
)
from moe_sae.learning_horizon_config import (
    LearningHorizonConfigError,
    load_learning_horizon_config,
)

A256_PROVENANCE_ADDENDUM_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-v3-"
    "a256native-provenance-addendum-v1"
)
A256_PROVENANCE_ADDENDUM_ID = (
    "gemma4-experiment5-v3-a256native-provenance-correction-20260729"
)
PANELS = ("sae", "transcoder")
MILESTONES = (2_000_000, 5_000_000, 10_000_000)
TRAIN_TOKENS = 10_000_000

_SCOPE = {
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


@dataclass(frozen=True, slots=True)
class A256AddendumCost:
    resource_id: str
    accelerator_type: str
    chip_count: int
    marginal_wall_seconds: float
    chip_seconds: float
    chip_seconds_scope: str
    chip_seconds_excludes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class A256AddendumCheckpoint:
    checkpoint_tokens: int
    checkpoint_pointer: EvidencePath
    checkpoint_pointer_sha256: str


@dataclass(frozen=True, slots=True)
class A256AddendumPanel:
    panel: str
    source_label: str
    seed: int
    encoder_activation: str
    selection_policy: str
    declared_source_config_semantic_sha256: str
    observed_source_config_semantic_sha256: str
    observed_source_config_file_sha256: str
    declaration_status: str
    disposition: str
    training_source_revision: str
    source_config: EvidencePath
    run_root_remote: str
    checkpoint_payload_root_gcs: str
    run_contract: EvidencePath
    run_contract_sha256: str
    training_result: EvidencePath
    training_result_sha256: str
    cost: A256AddendumCost
    checkpoints: tuple[A256AddendumCheckpoint, ...]


def _positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} must be finite and positive"
        )
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RoutedLadderCampaignPlanError(
            f"{name} must be a positive integer"
        )
    return value


def _gcs_prefix(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("gs://")
        or value.endswith("/")
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} must be a canonical GCS prefix"
        )
    return value


def _parse_cost(value: Any, name: str) -> A256AddendumCost:
    row = _exact_mapping(
        value,
        {
            "resource_id",
            "accelerator_type",
            "chip_count",
            "marginal_wall_seconds",
            "chip_seconds",
            "chip_seconds_scope",
            "chip_seconds_excludes",
        },
        name,
    )
    chip_count = _positive_int(row["chip_count"], f"{name}.chip_count")
    marginal = _positive_number(
        row["marginal_wall_seconds"],
        f"{name}.marginal_wall_seconds",
    )
    chip_seconds = _positive_number(
        row["chip_seconds"],
        f"{name}.chip_seconds",
    )
    scope = row["chip_seconds_scope"]
    excludes = row["chip_seconds_excludes"]
    if (
        not isinstance(scope, str)
        or not scope
        or not isinstance(excludes, list)
        or not excludes
        or not all(isinstance(item, str) and item for item in excludes)
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} has invalid measured-cost scope"
        )
    if not math.isclose(
        chip_seconds,
        chip_count * marginal,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} chip-seconds do not equal chips times wall-seconds"
        )
    return A256AddendumCost(
        resource_id=_safe_id(row["resource_id"], f"{name}.resource_id"),
        accelerator_type=_safe_id(
            row["accelerator_type"],
            f"{name}.accelerator_type",
        ),
        chip_count=chip_count,
        marginal_wall_seconds=marginal,
        chip_seconds=chip_seconds,
        chip_seconds_scope=scope,
        chip_seconds_excludes=tuple(excludes),
    )


def _parse_checkpoint(
    value: Any,
    name: str,
) -> A256AddendumCheckpoint:
    row = _exact_mapping(
        value,
        {
            "checkpoint_tokens",
            "checkpoint_pointer",
            "checkpoint_pointer_sha256",
        },
        name,
    )
    tokens = row["checkpoint_tokens"]
    if tokens not in MILESTONES:
        raise RoutedLadderCampaignPlanError(
            f"{name}.checkpoint_tokens is not an A256 milestone"
        )
    return A256AddendumCheckpoint(
        checkpoint_tokens=int(tokens),
        checkpoint_pointer=_evidence(
            row["checkpoint_pointer"],
            f"{name}.checkpoint_pointer",
        ),
        checkpoint_pointer_sha256=_sha256(
            row["checkpoint_pointer_sha256"],
            f"{name}.checkpoint_pointer_sha256",
        ),
    )


def _parse_panel(value: Any, index: int) -> A256AddendumPanel:
    name = f"a256_provenance_addendum.panels[{index}]"
    row = _exact_mapping(
        value,
        {
            "panel",
            "target_condition",
            "source_condition",
            "source_label",
            "seed",
            "encoder_activation",
            "selection_policy",
            "declared_source_config_semantic_sha256",
            "observed_source_config_semantic_sha256",
            "observed_source_config_file_sha256",
            "declaration_status",
            "disposition",
            "training_source_revision",
            "source_config",
            "run_root_remote",
            "checkpoint_payload_root_gcs",
            "run_contract",
            "run_contract_sha256",
            "training_result",
            "training_result_sha256",
            "cost",
            "checkpoints",
        },
        name,
    )
    panel = row["panel"]
    if (
        panel not in PANELS
        or row["target_condition"] != "A256Native"
        or row["source_condition"] != "A"
        or row["source_label"] != A256_NATIVE_LABEL.get(str(panel))
        or row["seed"] != 0
        or row["encoder_activation"]
        != A256_NATIVE_ACTIVATION.get(str(panel))
        or row["selection_policy"]
        != A256_NATIVE_SELECTION_POLICY.get(str(panel))
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} is not panel-native historical A256"
        )
    declared = _sha256(
        row["declared_source_config_semantic_sha256"],
        f"{name}.declared_source_config_semantic_sha256",
    )
    observed = _sha256(
        row["observed_source_config_semantic_sha256"],
        f"{name}.observed_source_config_semantic_sha256",
    )
    raw_sha = _sha256(
        row["observed_source_config_file_sha256"],
        f"{name}.observed_source_config_file_sha256",
    )
    if declared != A256_NATIVE_CONFIG_SHA256[panel]:
        raise RoutedLadderCampaignPlanError(
            f"{name} does not preserve the frozen v3 declaration"
        )
    expected_status = "exact_match" if declared == observed else "mismatch"
    expected_disposition = (
        "contextual_reference_verified"
        if expected_status == "exact_match"
        else "contextual_reference_corrected_by_observed_lineage"
    )
    if (
        row["declaration_status"] != expected_status
        or row["disposition"] != expected_disposition
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} does not state its declaration mismatch honestly"
        )
    source_config = _evidence(
        row["source_config"],
        f"{name}.source_config",
    )
    if source_config.sha256 != raw_sha:
        raise RoutedLadderCampaignPlanError(
            f"{name} raw config hash differs from its evidence"
        )
    raw_checkpoints = row["checkpoints"]
    if not isinstance(raw_checkpoints, list):
        raise RoutedLadderCampaignPlanError(
            f"{name}.checkpoints must be a list"
        )
    checkpoints = tuple(
        _parse_checkpoint(item, f"{name}.checkpoints[{checkpoint_index}]")
        for checkpoint_index, item in enumerate(raw_checkpoints)
    )
    if (
        len(checkpoints) != len(MILESTONES)
        or {item.checkpoint_tokens for item in checkpoints}
        != set(MILESTONES)
        or len({item.checkpoint_pointer for item in checkpoints})
        != len(MILESTONES)
        or len(
            {item.checkpoint_pointer_sha256 for item in checkpoints}
        )
        != len(MILESTONES)
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} must pin distinct 2M/5M/10M checkpoints"
        )
    return A256AddendumPanel(
        panel=str(panel),
        source_label=str(row["source_label"]),
        seed=0,
        encoder_activation=str(row["encoder_activation"]),
        selection_policy=str(row["selection_policy"]),
        declared_source_config_semantic_sha256=declared,
        observed_source_config_semantic_sha256=observed,
        observed_source_config_file_sha256=raw_sha,
        declaration_status=expected_status,
        disposition=expected_disposition,
        training_source_revision=_git_sha(
            row["training_source_revision"],
            f"{name}.training_source_revision",
        ),
        source_config=source_config,
        run_root_remote=_remote_path(
            row["run_root_remote"],
            f"{name}.run_root_remote",
        ),
        checkpoint_payload_root_gcs=_gcs_prefix(
            row["checkpoint_payload_root_gcs"],
            f"{name}.checkpoint_payload_root_gcs",
        ),
        run_contract=_evidence(
            row["run_contract"],
            f"{name}.run_contract",
        ),
        run_contract_sha256=_sha256(
            row["run_contract_sha256"],
            f"{name}.run_contract_sha256",
        ),
        training_result=_evidence(
            row["training_result"],
            f"{name}.training_result",
        ),
        training_result_sha256=_sha256(
            row["training_result_sha256"],
            f"{name}.training_result_sha256",
        ),
        cost=_parse_cost(row["cost"], f"{name}.cost"),
        checkpoints=tuple(
            sorted(
                checkpoints,
                key=lambda checkpoint: checkpoint.checkpoint_tokens,
            )
        ),
    )


def parse_a256_provenance_addendum(
    value: Any,
) -> tuple[Mapping[str, Any], tuple[A256AddendumPanel, ...]]:
    selected = _exact_mapping(
        value,
        {
            "schema_version",
            "addendum_id",
            "source_protocol_id",
            "target_protocol_id",
            "source_plan_sha256",
            "scientific_scope",
            "panels",
            "manifest_sha256",
        },
        "a256_provenance_addendum",
    )
    if (
        selected["schema_version"]
        != A256_PROVENANCE_ADDENDUM_SCHEMA_VERSION
        or selected["addendum_id"] != A256_PROVENANCE_ADDENDUM_ID
        or selected["source_protocol_id"] != EXPERIMENT4_PROTOCOL_ID
        or selected["target_protocol_id"] != PROTOCOL_ID
        or selected["scientific_scope"] != _SCOPE
    ):
        raise RoutedLadderCampaignPlanError(
            "A256 provenance addendum identity or scientific scope changed"
        )
    _sha256(
        selected["source_plan_sha256"],
        "a256_provenance_addendum.source_plan_sha256",
    )
    body = {
        key: selected[key] for key in selected if key != "manifest_sha256"
    }
    claimed = _sha256(
        selected["manifest_sha256"],
        "a256_provenance_addendum.manifest_sha256",
    )
    if canonical_sha256(body) != claimed:
        raise RoutedLadderCampaignPlanError(
            "A256 provenance addendum self-hash changed"
        )
    raw_panels = selected["panels"]
    if not isinstance(raw_panels, list):
        raise RoutedLadderCampaignPlanError(
            "A256 provenance addendum panels must be a list"
        )
    panels = tuple(
        sorted(
            (
                _parse_panel(item, index)
                for index, item in enumerate(raw_panels)
            ),
            key=lambda item: item.panel,
        )
    )
    if (
        len(panels) != len(PANELS)
        or {item.panel for item in panels} != set(PANELS)
        or len({item.training_source_revision for item in panels}) != 1
    ):
        raise RoutedLadderCampaignPlanError(
            "A256 provenance addendum requires both panels from one revision"
        )
    by_panel = {item.panel: item for item in panels}
    if (
        by_panel["sae"].declaration_status != "exact_match"
        or by_panel["transcoder"].declaration_status != "mismatch"
    ):
        raise RoutedLadderCampaignPlanError(
            "A256 provenance addendum must preserve the SAE match and "
            "transcoder mismatch"
        )
    return copy.deepcopy(selected), panels


def _expected_progress(
    result: Mapping[str, Any],
    checkpoint_tokens: int,
) -> tuple[int, int]:
    identity = result.get("run_identity")
    plan = identity.get("plan") if isinstance(identity, Mapping) else None
    if not isinstance(plan, Mapping):
        raise RoutedLadderCampaignStateError(
            "A256 historical result lacks its immutable worker plan"
        )
    microbatch = plan.get("global_microbatch_tokens")
    accumulation = plan.get("microbatches_per_update")
    total = plan.get("total_tokens")
    planned_updates = plan.get("planned_updates")
    if (
        isinstance(microbatch, bool)
        or not isinstance(microbatch, int)
        or microbatch <= 0
        or isinstance(accumulation, bool)
        or not isinstance(accumulation, int)
        or accumulation <= 0
        or total != TRAIN_TOKENS
        or planned_updates
        != math.ceil(total / (microbatch * accumulation))
    ):
        raise RoutedLadderCampaignStateError(
            "A256 historical result has an invalid exact-token plan"
        )
    batch = microbatch * accumulation
    step = math.ceil(checkpoint_tokens / batch)
    return step, min(step * batch, total)


def validate_a256_addendum_panel(
    panel: A256AddendumPanel,
) -> None:
    try:
        config_payload = panel.source_config.controller_path.read_bytes()
    except OSError as error:
        raise RoutedLadderCampaignStateError(
            f"cannot read A256 {panel.panel} source config"
        ) from error
    if (
        hashlib.sha256(config_payload).hexdigest()
        != panel.observed_source_config_file_sha256
    ):
        raise RoutedLadderCampaignStateError(
            f"A256 {panel.panel} source config file changed"
        )
    try:
        config = load_learning_horizon_config(
            panel.source_config.controller_path,
            expected_scientific_model=panel.panel,
        )
    except LearningHorizonConfigError as error:
        raise RoutedLadderCampaignStateError(
            f"A256 {panel.panel} source config is invalid"
        ) from error
    layout = next((item for item in config.layouts if item.condition == "A"), None)
    if (
        config.config_hash
        != panel.observed_source_config_semantic_sha256
        or config.encoder_activation != panel.encoder_activation
        or layout is None
        or layout.bank_identity != "true_expert_id"
        or layout.bank_count != 128
        or layout.bank_width != 256
        or layout.components_per_token != 8
        or layout.topk_per_component != 2
        or layout.target != "raw_selected_expert_output"
    ):
        raise RoutedLadderCampaignStateError(
            f"A256 {panel.panel} observed scientific config changed"
        )
    contract, contract_payload = _read_self_hashed_json(
        panel.run_contract.controller_path,
        hash_field="run_contract_sha256",
    )
    result, result_payload = _read_self_hashed_json(
        panel.training_result.controller_path,
        hash_field="result_sha256",
    )
    identity = result.get("run_identity")
    run_plan = identity.get("plan") if isinstance(identity, Mapping) else None
    if (
        hashlib.sha256(contract_payload).hexdigest()
        != panel.run_contract.sha256
        or hashlib.sha256(result_payload).hexdigest()
        != panel.training_result.sha256
        or contract.get("run_contract_sha256")
        != panel.run_contract_sha256
        or result.get("result_sha256") != panel.training_result_sha256
        or result.get("schema_version")
        != "gemma4-contribution-sae-training-result-v1"
        or result.get("status") not in {"completed", "already_complete"}
        or result.get("tokens") != TRAIN_TOKENS
        or result.get("source_revision")
        != panel.training_source_revision
        or not isinstance(identity, Mapping)
        or not isinstance(run_plan, Mapping)
        or run_plan.get("phase_name") != "learning_horizon_10m"
        or run_plan.get("condition") != "A"
        or run_plan.get("seed") != 0
        or contract.get("project_commit")
        != panel.training_source_revision
        or contract.get("phase") != "learning_horizon_10m"
        or contract.get("scientific_model") != panel.panel
        or contract.get("condition") != "A"
        or contract.get("seed") != 0
        or contract.get("config_file_sha256")
        != panel.observed_source_config_file_sha256
        or contract.get("checkpoint_payload_root")
        != panel.checkpoint_payload_root_gcs
        or not math.isclose(
            _positive_number(
                result.get("marginal_wall_seconds"),
                "historical marginal_wall_seconds",
            ),
            panel.cost.marginal_wall_seconds,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        or not math.isclose(
            _positive_number(
                result.get("chip_seconds"),
                "historical chip_seconds",
            ),
            panel.cost.chip_seconds,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        or result.get("chip_seconds_scope")
        != panel.cost.chip_seconds_scope
        or result.get("chip_seconds_excludes")
        != list(panel.cost.chip_seconds_excludes)
    ):
        raise RoutedLadderCampaignStateError(
            f"A256 {panel.panel} run lineage changed"
        )
    for checkpoint in panel.checkpoints:
        pointer, pointer_payload = _read_self_hashed_json(
            checkpoint.checkpoint_pointer.controller_path,
            hash_field="pointer_sha256",
        )
        if (
            hashlib.sha256(pointer_payload).hexdigest()
            != checkpoint.checkpoint_pointer.sha256
        ):
            raise RoutedLadderCampaignStateError(
                f"A256 {panel.panel} checkpoint file changed"
            )
        try:
            checked = validate_pointer(
                pointer,
                expected_kind="latest",
                payload_root=panel.checkpoint_payload_root_gcs,
                run_identity=identity,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise RoutedLadderCampaignStateError(
                f"A256 {panel.panel} checkpoint is invalid"
            ) from error
        expected_step, expected_tokens = _expected_progress(
            result,
            checkpoint.checkpoint_tokens,
        )
        if (
            checked["pointer_sha256"]
            != checkpoint.checkpoint_pointer_sha256
            or checked["step"] != expected_step
            or checked["tokens_seen"] != expected_tokens
        ):
            raise RoutedLadderCampaignStateError(
                f"A256 {panel.panel} checkpoint progress changed"
            )


def build_a256_provenance_addendum(
    horizon_plan: DirectCampaignPlan,
) -> dict[str, Any]:
    environment = horizon_plan.environment_dict()
    training_revision = _git_sha(
        environment.get("TRAINING_PROJECT_COMMIT"),
        "horizon_plan.environment.TRAINING_PROJECT_COMMIT",
    )
    panels: list[dict[str, Any]] = []
    for panel in PANELS:
        matches = [
            task
            for task in horizon_plan.trajectories
            if task.scientific_model == panel
            and task.condition == "A"
            and task.seed == 0
        ]
        if len(matches) != 1:
            raise RoutedLadderCampaignPlanError(
                f"horizon plan lacks one historical {panel} A seed-zero run"
            )
        task = matches[0]
        if inspect_horizon_training_completion(task) is None:
            raise RoutedLadderCampaignPlanError(
                f"historical {panel} A256 training is incomplete"
            )
        result, result_payload = _read_self_hashed_json(
            task.completion_path_controller,
            hash_field="result_sha256",
        )
        remote_horizon_root = PurePosixPath(task.control_root_remote).parents[1]
        try:
            relative_config = PurePosixPath(task.config_remote).relative_to(
                remote_horizon_root
            )
            controller_horizon_root = task.completion_path_controller.parents[4]
        except (ValueError, IndexError) as error:
            raise RoutedLadderCampaignPlanError(
                f"historical {panel} config is outside its artifact plane"
            ) from error
        controller_config = (
            controller_horizon_root / Path(*relative_config.parts)
        )
        try:
            config_payload = controller_config.read_bytes()
            config = load_learning_horizon_config(
                controller_config,
                expected_scientific_model=panel,
            )
        except (OSError, LearningHorizonConfigError) as error:
            raise RoutedLadderCampaignPlanError(
                f"historical {panel} A256 config is invalid"
            ) from error
        config_file_sha = hashlib.sha256(config_payload).hexdigest()
        declared = A256_NATIVE_CONFIG_SHA256[panel]
        status = "exact_match" if declared == config.config_hash else "mismatch"
        if (
            (panel == "sae" and status != "exact_match")
            or (panel == "transcoder" and status != "mismatch")
        ):
            raise RoutedLadderCampaignPlanError(
                "observed A256 declaration pattern changed: expected the "
                "frozen SAE declaration to match and transcoder declaration "
                "to mismatch"
            )
        resource = horizon_plan.resource(task.resource_id)
        marginal = _positive_number(
            result.get("marginal_wall_seconds"),
            f"historical {panel} A256 marginal_wall_seconds",
        )
        chip_seconds = _positive_number(
            result.get("chip_seconds"),
            f"historical {panel} A256 chip_seconds",
        )
        raw_chips = chip_seconds / marginal
        chip_count = round(raw_chips)
        exclusions = result.get("chip_seconds_excludes")
        scope = result.get("chip_seconds_scope")
        if (
            chip_count <= 0
            or not math.isclose(
                raw_chips,
                chip_count,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not isinstance(scope, str)
            or not scope
            or not isinstance(exclusions, list)
            or not exclusions
            or not all(isinstance(item, str) and item for item in exclusions)
        ):
            raise RoutedLadderCampaignPlanError(
                f"historical {panel} A256 measured cost is invalid"
            )
        run_root = str(
            PurePosixPath(task.control_root_remote) / "runs" / task.run_id
        )
        run_contract_path = (
            task.completion_path_controller.parent / "run-contract.json"
        )
        contract, contract_payload = _read_self_hashed_json(
            run_contract_path,
            hash_field="run_contract_sha256",
        )
        if (
            contract.get("project_commit") != training_revision
            or contract.get("phase") != "learning_horizon_10m"
            or contract.get("scientific_model") != panel
            or contract.get("condition") != "A"
            or contract.get("seed") != 0
            or contract.get("config_file_sha256") != config_file_sha
            or contract.get("checkpoint_payload_root")
            != task.checkpoint_root_gcs
        ):
            raise RoutedLadderCampaignPlanError(
                f"historical {panel} A256 run contract changed"
            )
        controller_checkpoints = (
            task.completion_path_controller.parent / "checkpoints"
        )
        remote_checkpoints = str(PurePosixPath(run_root) / "checkpoints")
        checkpoints: list[dict[str, Any]] = []
        for milestone in MILESTONES:
            pointer_path = milestone_pointer_path(
                controller_checkpoints,
                milestone,
            )
            pointer, pointer_payload = _read_self_hashed_json(
                pointer_path,
                hash_field="pointer_sha256",
            )
            checkpoints.append(
                {
                    "checkpoint_tokens": milestone,
                    "checkpoint_pointer": {
                        "controller_path": str(pointer_path),
                        "remote_path": str(
                            milestone_pointer_path(
                                remote_checkpoints,
                                milestone,
                            )
                        ),
                        "sha256": hashlib.sha256(
                            pointer_payload
                        ).hexdigest(),
                    },
                    "checkpoint_pointer_sha256": pointer["pointer_sha256"],
                }
            )
        panels.append(
            {
                "panel": panel,
                "target_condition": "A256Native",
                "source_condition": "A",
                "source_label": A256_NATIVE_LABEL[panel],
                "seed": 0,
                "encoder_activation": A256_NATIVE_ACTIVATION[panel],
                "selection_policy": A256_NATIVE_SELECTION_POLICY[panel],
                "declared_source_config_semantic_sha256": declared,
                "observed_source_config_semantic_sha256": config.config_hash,
                "observed_source_config_file_sha256": config_file_sha,
                "declaration_status": status,
                "disposition": (
                    "contextual_reference_verified"
                    if status == "exact_match"
                    else "contextual_reference_corrected_by_observed_lineage"
                ),
                "training_source_revision": training_revision,
                "source_config": {
                    "controller_path": str(controller_config),
                    "remote_path": task.config_remote,
                    "sha256": config_file_sha,
                },
                "run_root_remote": run_root,
                "checkpoint_payload_root_gcs": task.checkpoint_root_gcs,
                "run_contract": {
                    "controller_path": str(run_contract_path),
                    "remote_path": str(
                        PurePosixPath(run_root) / "run-contract.json"
                    ),
                    "sha256": hashlib.sha256(contract_payload).hexdigest(),
                },
                "run_contract_sha256": contract["run_contract_sha256"],
                "training_result": {
                    "controller_path": str(
                        task.completion_path_controller
                    ),
                    "remote_path": str(
                        PurePosixPath(run_root) / "training-result.json"
                    ),
                    "sha256": hashlib.sha256(result_payload).hexdigest(),
                },
                "training_result_sha256": result["result_sha256"],
                "cost": {
                    "resource_id": resource.resource_id,
                    "accelerator_type": resource.accelerator_type,
                    "chip_count": chip_count,
                    "marginal_wall_seconds": marginal,
                    "chip_seconds": chip_seconds,
                    "chip_seconds_scope": scope,
                    "chip_seconds_excludes": exclusions,
                },
                "checkpoints": checkpoints,
            }
        )
    body = {
        "schema_version": A256_PROVENANCE_ADDENDUM_SCHEMA_VERSION,
        "addendum_id": A256_PROVENANCE_ADDENDUM_ID,
        "source_protocol_id": EXPERIMENT4_PROTOCOL_ID,
        "target_protocol_id": PROTOCOL_ID,
        "source_plan_sha256": horizon_plan.plan_sha256,
        "scientific_scope": copy.deepcopy(_SCOPE),
        "panels": panels,
    }
    manifest = {**body, "manifest_sha256": canonical_sha256(body)}
    _parsed, parsed_panels = parse_a256_provenance_addendum(manifest)
    for parsed_panel in parsed_panels:
        validate_a256_addendum_panel(parsed_panel)
    return manifest


def load_a256_provenance_addendum(
    path: str | Path,
) -> Mapping[str, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RoutedLadderCampaignPlanError(
            f"cannot read A256 provenance addendum: {selected}"
        ) from error
    parsed, _panels = parse_a256_provenance_addendum(value)
    return parsed


def write_a256_provenance_addendum(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[A256AddendumPanel, ...]]:
    parsed, panels = parse_a256_provenance_addendum(manifest)
    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = yaml.safe_dump(
        dict(manifest),
        sort_keys=False,
        default_flow_style=False,
    ).encode("utf-8")
    try:
        descriptor = os.open(
            selected,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o640,
        )
    except FileExistsError as error:
        raise RoutedLadderCampaignPlanError(
            f"refusing to overwrite A256 provenance addendum: {selected}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        selected.unlink(missing_ok=True)
        raise
    return parsed, panels


__all__ = [
    "A256_PROVENANCE_ADDENDUM_ID",
    "A256_PROVENANCE_ADDENDUM_SCHEMA_VERSION",
    "A256AddendumCheckpoint",
    "A256AddendumCost",
    "A256AddendumPanel",
    "MILESTONES",
    "build_a256_provenance_addendum",
    "load_a256_provenance_addendum",
    "parse_a256_provenance_addendum",
    "validate_a256_addendum_panel",
    "write_a256_provenance_addendum",
]
