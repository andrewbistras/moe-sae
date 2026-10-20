"""Durable campaign boundary for the Experiment-5 activation ladder.

This module is deliberately a sibling of :mod:`moe_sae.experiment5_campaign`.
It reuses that controller's append-only attempt, readiness, retry, lock, and
budget machinery without changing the frozen H/R/B/F task matrix or state
namespace.

The extension schedules exactly:

* 17 new training trajectories under the protocol-v2 provenance matrix;
* 54 A256ReLU/A512ReLU/A512Softplus evaluations at 2M/5M/10M;
* 18 read-only rescorings of immutable H checkpoints; and
* 18 hash-pinned H training/checkpoint references that are never retrained.

The controller owns no cloud lifecycle.  A self-hashed resource-pool manifest
admits three to eighteen already-deployed, evidence-bound slices.  Every task
has one immutable owner, while the inherited scheduler continues to serialize
attempts per resource and to handle readiness, retry, and preemption evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_campaign import (
    CampaignResource,
    CompletionFingerprint,
    CostPolicy,
    EvidencePath,
    Experiment5Campaign,
    Experiment5CampaignPlan,
    Experiment5CampaignPlanError,
    Experiment5CampaignStateError,
    RetryPolicy,
    _controller_path,
    _evidence,
    _environment,
    _exact_mapping,
    _gcs_prefix,
    _git_sha,
    _money,
    _positive_int,
    _read_record,
    _read_self_hashed_json,
    _record,
    _remote_path,
    _resource,
    _safe_id,
    _seal_exact_bytes,
    _sha256,
    _timestamp,
    _verify_file,
    _write_create_only,
    inspect_training_completion as inspect_core_training_completion,
)
from moe_sae.checkpointing import milestone_pointer_path, validate_pointer
from moe_sae.direct_horizon_campaign import (
    PROTOCOL_ID as EXPERIMENT4_PROTOCOL_ID,
    DirectCampaignPlan,
    DirectCampaignPlanError,
    inspect_training_completion as inspect_horizon_training_completion,
    parse_direct_campaign_plan,
)
from moe_sae.experiment5_config import (
    Experiment5ConfigError,
    load_experiment5_config,
)
from moe_sae.experiment5_plan_builder import _PRESERVED_GLOBAL_ENVIRONMENT
from moe_sae.experiment5_routed_ladder import (
    PROTOCOL_ID,
    SCORE_SCHEMA_VERSION,
    Experiment5RoutedLadderError,
    ImmutableArtifactReference,
    RoutedLadderEvaluationBundleAccounting,
    RoutedLadderHTokenScores,
    RoutedLadderTokenScores,
    RoutedLadderTrainingCost,
    RoutedLadderTrajectoryAccounting,
    RoutedLadderWallCashAccounting,
    build_experiment5_routed_ladder_report,
    h_token_scores_to_existing_reference,
    validate_experiment5_routed_ladder_report,
)
from moe_sae.experiment5_routed_ladder_evaluation_worker import (
    BUNDLE_CONDITIONS,
    BUNDLE_RESULT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION as EVALUATION_RESULT_SCHEMA_VERSION,
    RoutedLadderEvaluationError,
    load_merged_scores,
)
from moe_sae.experiment5_routed_ladder_config import (
    H_CONFIG_SHA256,
)
from moe_sae.experiment5_routed_ladder_v2_config import (
    A256_SAE_CONFIG_SHA256,
    A256_SAE_NEW_TRAINING_SEEDS,
    ANALYZED_CONDITIONS,
    CONDITIONS as V2_CONDITIONS,
    PHASE_NAME,
    RoutedLadderV2ConfigError,
    load_routed_ladder_v2_config,
)
from moe_sae.experiment5_routed_ladder_v3_config import (
    PROTOCOL_ID as V3_PROTOCOL_ID,
    V2_CONFIG_SHA256,
)
from moe_sae.experiment5_routed_ladder_resources import (
    MAX_RESOURCE_COUNT,
    MIN_RESOURCE_COUNT,
    RoutedLadderResourcePool,
    RoutedLadderResourcePoolError,
    parse_routed_ladder_resource_pool,
)
from moe_sae.learning_horizon_config import (
    LearningHorizonConfigError,
    load_learning_horizon_config,
)

PLAN_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-campaign-plan-v2"
)
H_REFERENCE_MANIFEST_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-h-reference-manifest-v1"
)
A256_REFERENCE_MANIFEST_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-a256-reference-manifest-v1"
)
COMPLETED_TRAJECTORY_REFERENCE_MANIFEST_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-completed-trajectory-"
    "reference-manifest-v1"
)
EVALUATION_SCORE_SCHEMA_VERSION = SCORE_SCHEMA_VERSION
REPORT_INPUT_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-report-inputs-v2"
)

PANELS = ("sae", "transcoder")
CONDITIONS = V2_CONDITIONS
EVALUATION_CONDITIONS = ANALYZED_CONDITIONS
SEEDS = (0, 1, 2)
MILESTONES = (2_000_000, 5_000_000, 10_000_000)
TRAIN_TOKENS = 10_000_000


def training_seeds_for_condition(
    panel: str,
    condition: str,
) -> tuple[int, ...]:
    """Return the exact protocol-v2 set of newly trained seeds."""

    if panel not in PANELS or condition not in CONDITIONS:
        raise RoutedLadderCampaignPlanError(
            "training role is outside the protocol-v2 matrix"
        )
    if panel == "sae" and condition == "A256ReLU":
        return A256_SAE_NEW_TRAINING_SEEDS
    return SEEDS


EXPECTED_TRAINING_ROLES = frozenset(
    (panel, condition, seed)
    for panel in PANELS
    for condition in CONDITIONS
    for seed in training_seeds_for_condition(panel, condition)
)
EXPECTED_EVALUATION_ROLES = frozenset(
    (panel, condition, seed, milestone)
    for panel in PANELS
    for condition in EVALUATION_CONDITIONS
    for seed in SEEDS
    for milestone in MILESTONES
)
TRAINING_TASK_COUNT = len(EXPECTED_TRAINING_ROLES)
EVALUATION_TASK_COUNT = len(EXPECTED_EVALUATION_ROLES)
EVALUATION_BUNDLE_COUNT = len(PANELS) * len(SEEDS)
H_REFERENCE_COUNT = 18
A256_REFERENCE_COUNT = 3
EXTENSION_NAMESPACE = (
    "experiment5/extensions/routed-activation-ladder-v2"
)
CONTINUOUS_RUNTIME_SECONDS = 43_200
OPERATOR_BUNDLE_MANIFEST_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-operator-bundle-manifest-v1"
)
_CHIP_COUNT_BY_ACCELERATOR = {
    "v4-32": 16,
    "v6e-16": 16,
    "v5litepod-16": 16,
}
_EVALUATION_TIMING_SCOPE = (
    "successful_resident_worker_after_cli_parse_through_cell_completion"
)
_CAMPAIGN_WALL_SCOPE = (
    "durable_campaign_clock_through_evaluation_completion"
)

_CONFIG_NAMES = {
    "sae": "gemma4_experiment5_routed_ladder_sae_v2.yaml",
    "transcoder": "gemma4_experiment5_routed_ladder_transcoder_v2.yaml",
}
_RESERVED_ENVIRONMENT = frozenset(
    {
        "EXPERIMENT5_ROUTED_ACTIVATION_LADDER_RESULT_ROOT_CONTROLLER",
    }
)
_CONDITION_SLUG = {
    "A256ReLU": "a256-relu",
    "A512ReLU": "a512-relu",
    "A512Softplus": "a512-softplus",
    "H": "h",
}


class RoutedLadderCampaignPlanError(Experiment5CampaignPlanError):
    """The extension plan is incomplete, mutable, or outside its namespace."""


class RoutedLadderCampaignStateError(Experiment5CampaignStateError):
    """The extension's durable state or completion evidence is invalid."""


@dataclass(frozen=True, slots=True)
class RoutedLadderTrainingTask:
    task_id: str
    panel: str
    condition: str
    seed: int
    eligible_resource_ids: tuple[str, ...]
    run_id: str
    config_remote: str
    config_sha256: str
    control_root_remote: str
    checkpoint_root_gcs: str
    source_overlay_control_root_remote: str | None
    source_overlay_object_prefix_gcs: str | None
    completion_path_controller: Path
    environment: tuple[tuple[str, str], ...]

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)


@dataclass(frozen=True, slots=True)
class RoutedLadderEvaluationTask:
    task_id: str
    panel: str
    condition: str
    seed: int
    milestone_tokens: int
    eligible_resource_ids: tuple[str, ...]
    config_remote: str
    config_sha256: str
    h_config_remote: str | None
    h_config_sha256: str | None
    a256_config_remote: str | None
    a256_config_sha256: str | None
    reference_run_contract_sha256: str | None
    reference_run_contract_file_sha256: str | None
    training_source_revision: str
    control_root_remote: str
    checkpoint_root_gcs: str
    train_source_root_remote: str | None
    eval_source_root_remote: str | None
    train_source_object_prefix_gcs: str | None
    eval_source_object_prefix_gcs: str | None
    run_root_remote: str
    result_root_remote: str
    result_root_controller: Path
    completion_path_controller: Path
    environment: tuple[tuple[str, str], ...]

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)


@dataclass(frozen=True, slots=True)
class RoutedLadderEvaluationBundleTask:
    """One resident operator invocation covering twelve immutable cells."""

    task_id: str
    panel: str
    seed: int
    eligible_resource_ids: tuple[str, ...]
    config_remote: str
    control_root_remote: str
    checkpoint_root_gcs: str
    train_source_root_remote: str | None
    eval_source_root_remote: str | None
    train_source_object_prefix_gcs: str | None
    eval_source_object_prefix_gcs: str | None
    result_root_remote: str
    result_root_controller: Path
    completion_path_controller: Path
    cells: tuple[RoutedLadderEvaluationTask, ...]
    environment: tuple[tuple[str, str], ...] = ()

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)


RoutedLadderTask = (
    RoutedLadderTrainingTask
    | RoutedLadderEvaluationTask
    | RoutedLadderEvaluationBundleTask
)


@dataclass(frozen=True, slots=True)
class HCheckpointReference:
    """One hash-pinned, read-only H checkpoint consumed by rescoring."""

    panel: str
    seed: int
    checkpoint_tokens: int
    training_source_revision: str
    h_config_remote: str
    h_config_sha256: str
    run_root_remote: str
    checkpoint_payload_root_gcs: str
    training_result: EvidencePath
    training_result_sha256: str
    checkpoint_pointer: EvidencePath
    checkpoint_pointer_sha256: str

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.panel, self.seed, self.checkpoint_tokens


@dataclass(frozen=True, slots=True)
class A256HistoricalCost:
    """Measured cost provenance for the one reused Experiment-4 trajectory."""

    resource_id: str
    accelerator_type: str
    chip_count: int
    marginal_wall_seconds: float
    chip_seconds: float
    chip_seconds_scope: str
    chip_seconds_excludes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class A256CheckpointReference:
    """One milestone of the sole compatible historical SAE A256 trajectory."""

    checkpoint_tokens: int
    training_source_revision: str
    a256_config: EvidencePath
    a256_config_sha256: str
    run_root_remote: str
    checkpoint_payload_root_gcs: str
    run_contract: EvidencePath
    run_contract_sha256: str
    training_result: EvidencePath
    training_result_sha256: str
    checkpoint_pointer: EvidencePath
    checkpoint_pointer_sha256: str

    @property
    def identity(self) -> tuple[str, str, int, int]:
        return "sae", "A256ReLU", 0, self.checkpoint_tokens


@dataclass(frozen=True, slots=True)
class CompletedTrajectoryCheckpointReference:
    """One immutable milestone imported from an earlier ladder campaign."""

    panel: str
    source_condition: str
    target_condition: str
    seed: int
    checkpoint_tokens: int
    training_source_revision: str
    config_remote: str
    config_sha256: str
    config_file_sha256: str
    run_root_remote: str
    checkpoint_payload_root_gcs: str
    run_contract: EvidencePath
    run_contract_sha256: str
    training_result: EvidencePath
    training_result_sha256: str
    checkpoint_pointer: EvidencePath
    checkpoint_pointer_sha256: str
    resource_id: str
    accelerator_type: str

    @property
    def identity(self) -> tuple[str, str, int, int]:
        return (
            self.panel,
            self.target_condition,
            self.seed,
            self.checkpoint_tokens,
        )

    @property
    def trajectory_identity(self) -> tuple[str, str, int]:
        return self.panel, self.target_condition, self.seed

    @property
    def source_identity(self) -> tuple[str, str, int]:
        return self.panel, self.source_condition, self.seed


@dataclass(frozen=True, slots=True)
class RoutedLadderCampaignPlan:
    campaign_id: str
    state_root: Path
    working_directory: Path
    python_executable: Path
    remote_project_root: str
    remote_uv: str
    project_commit: str
    max_parallel_train: int
    max_parallel_eval: int
    retry_policy: RetryPolicy
    cost_policy: CostPolicy
    environment: tuple[tuple[str, str], ...]
    resources: tuple[CampaignResource, ...]
    resource_pool: RoutedLadderResourcePool
    trajectories: tuple[RoutedLadderTrainingTask, ...]
    evaluations: tuple[RoutedLadderEvaluationTask, ...]
    h_reference_manifest: Mapping[str, Any]
    h_references: tuple[HCheckpointReference, ...]
    a256_reference_manifest: Mapping[str, Any]
    a256_references: tuple[A256CheckpointReference, ...]
    a256_historical_cost: A256HistoricalCost
    plan_sha256: str
    source: Mapping[str, Any]

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)

    def resource(self, resource_id: str) -> CampaignResource:
        for resource in self.resources:
            if resource.resource_id == resource_id:
                return resource
        raise RoutedLadderCampaignPlanError(f"unknown resource {resource_id!r}")


def _eligible_resources(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RoutedLadderCampaignPlanError(f"{name} must be a nonempty list")
    selected = tuple(
        _safe_id(item, f"{name}[{index}]") for index, item in enumerate(value)
    )
    if len(set(selected)) != len(selected):
        raise RoutedLadderCampaignPlanError(f"{name} contains duplicates")
    return selected


def _panel_remote(value: Any, panel: str, name: str) -> str | None:
    if panel == "sae":
        if value is not None:
            raise RoutedLadderCampaignPlanError(f"{name} must be null for SAE")
        return None
    if value is None:
        raise RoutedLadderCampaignPlanError(f"{name} is required for transcoder")
    return _remote_path(value, name)


def _panel_gcs(value: Any, panel: str, name: str) -> str | None:
    if panel == "sae":
        if value is not None:
            raise RoutedLadderCampaignPlanError(f"{name} must be null for SAE")
        return None
    if value is None:
        raise RoutedLadderCampaignPlanError(f"{name} is required for transcoder")
    return _gcs_prefix(value, name)


def _task_environment(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    result = _environment(value, name)
    overlap = set(dict(result)) & _RESERVED_ENVIRONMENT
    if overlap:
        raise RoutedLadderCampaignPlanError(
            f"{name} overrides activation-ladder controller variables: "
            f"{sorted(overlap)!r}"
        )
    return result


def _positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} must be finite and positive"
        )
    return float(value)


def _training_task(value: Any, index: int) -> RoutedLadderTrainingTask:
    name = f"trajectories[{index}]"
    selected = _exact_mapping(
        value,
        {
            "id",
            "panel",
            "condition",
            "seed",
            "eligible_resource_ids",
            "run_id",
            "config_remote",
            "config_sha256",
            "phase",
            "control_root_remote",
            "checkpoint_root_gcs",
            "source_overlay_control_root_remote",
            "source_overlay_object_prefix_gcs",
            "completion_path_controller",
            "environment",
        },
        name,
    )
    panel = selected["panel"]
    condition = selected["condition"]
    seed = selected["seed"]
    if panel not in PANELS or condition not in CONDITIONS or seed not in SEEDS:
        raise RoutedLadderCampaignPlanError(f"{name} scientific role changed")
    slug = _CONDITION_SLUG[condition]
    expected_id = f"train-{panel}-{slug}-seed{seed}"
    expected_run = (
        f"experiment5-activation-ladder-{panel}-{slug}-seed{seed}"
    )
    if (
        selected["id"] != expected_id
        or selected["run_id"] != expected_run
        or selected["phase"] != PHASE_NAME
    ):
        raise RoutedLadderCampaignPlanError(f"{name} identity changed")
    completion = _controller_path(
        selected["completion_path_controller"],
        f"{name}.completion_path_controller",
    )
    if completion.name != "training-result.json" or completion.parent.name != expected_run:
        raise RoutedLadderCampaignPlanError(
            f"{name} completion is not bound to its run ID"
        )
    return RoutedLadderTrainingTask(
        task_id=expected_id,
        panel=panel,
        condition=condition,
        seed=seed,
        eligible_resource_ids=_eligible_resources(
            selected["eligible_resource_ids"],
            f"{name}.eligible_resource_ids",
        ),
        run_id=expected_run,
        config_remote=_remote_path(selected["config_remote"], f"{name}.config_remote"),
        config_sha256=_sha256(
            selected["config_sha256"], f"{name}.config_sha256"
        ),
        control_root_remote=_remote_path(
            selected["control_root_remote"], f"{name}.control_root_remote"
        ),
        checkpoint_root_gcs=_gcs_prefix(
            selected["checkpoint_root_gcs"], f"{name}.checkpoint_root_gcs"
        ),
        source_overlay_control_root_remote=_panel_remote(
            selected["source_overlay_control_root_remote"],
            panel,
            f"{name}.source_overlay_control_root_remote",
        ),
        source_overlay_object_prefix_gcs=_panel_gcs(
            selected["source_overlay_object_prefix_gcs"],
            panel,
            f"{name}.source_overlay_object_prefix_gcs",
        ),
        completion_path_controller=completion,
        environment=_task_environment(selected["environment"], f"{name}.environment"),
    )


def _evaluation_task(value: Any, index: int) -> RoutedLadderEvaluationTask:
    name = f"evaluations[{index}]"
    selected = _exact_mapping(
        value,
        {
            "id",
            "panel",
            "condition",
            "seed",
            "milestone_tokens",
            "eligible_resource_ids",
            "config_remote",
            "config_sha256",
            "h_config_remote",
            "h_config_sha256",
            "a256_config_remote",
            "a256_config_sha256",
            "reference_run_contract_sha256",
            "reference_run_contract_file_sha256",
            "training_source_revision",
            "phase",
            "control_root_remote",
            "checkpoint_root_gcs",
            "train_source_root_remote",
            "eval_source_root_remote",
            "train_source_object_prefix_gcs",
            "eval_source_object_prefix_gcs",
            "run_root_remote",
            "result_root_remote",
            "completion_path_controller",
            "result_root_controller",
            "environment",
        },
        name,
    )
    panel = selected["panel"]
    condition = selected["condition"]
    seed = selected["seed"]
    milestone = selected["milestone_tokens"]
    if (
        panel not in PANELS
        or condition not in EVALUATION_CONDITIONS
        or seed not in SEEDS
        or milestone not in MILESTONES
    ):
        raise RoutedLadderCampaignPlanError(f"{name} scientific role changed")
    slug = _CONDITION_SLUG[condition]
    expected_id = f"eval-{panel}-{slug}-seed{seed}-{milestone}"
    if selected["id"] != expected_id or selected["phase"] != PHASE_NAME:
        raise RoutedLadderCampaignPlanError(f"{name} identity changed")
    h_config_remote = selected["h_config_remote"]
    h_config_sha256 = selected["h_config_sha256"]
    a256_config_remote = selected["a256_config_remote"]
    a256_config_sha256 = selected["a256_config_sha256"]
    reference_run_contract_sha256 = selected[
        "reference_run_contract_sha256"
    ]
    reference_run_contract_file_sha256 = selected[
        "reference_run_contract_file_sha256"
    ]
    historical_a256 = (
        panel == "sae" and condition == "A256ReLU" and seed == 0
    )
    if condition == "H":
        h_config_remote = _remote_path(
            h_config_remote, f"{name}.h_config_remote"
        )
        h_config_sha256 = _sha256(
            h_config_sha256, f"{name}.h_config_sha256"
        )
        if a256_config_remote is not None or a256_config_sha256 is not None:
            raise RoutedLadderCampaignPlanError(
                f"{name} H task may not carry an A256 config"
            )
        if (
            reference_run_contract_sha256 is not None
            or reference_run_contract_file_sha256 is not None
        ):
            raise RoutedLadderCampaignPlanError(
                f"{name} H task may not carry an A256 run contract"
            )
    elif historical_a256:
        a256_config_remote = _remote_path(
            a256_config_remote,
            f"{name}.a256_config_remote",
        )
        a256_config_sha256 = _sha256(
            a256_config_sha256,
            f"{name}.a256_config_sha256",
        )
        if (
            a256_config_sha256 != A256_SAE_CONFIG_SHA256
            or h_config_remote is not None
            or h_config_sha256 is not None
        ):
            raise RoutedLadderCampaignPlanError(
                f"{name} historical A256 config identity changed"
            )
        reference_run_contract_sha256 = _sha256(
            reference_run_contract_sha256,
            f"{name}.reference_run_contract_sha256",
        )
        reference_run_contract_file_sha256 = _sha256(
            reference_run_contract_file_sha256,
            f"{name}.reference_run_contract_file_sha256",
        )
    elif h_config_remote is not None or h_config_sha256 is not None:
        raise RoutedLadderCampaignPlanError(
            f"{name} non-H task may not carry an H config"
        )
    elif a256_config_remote is not None or a256_config_sha256 is not None:
        raise RoutedLadderCampaignPlanError(
            f"{name} new task may not carry an A256 parent config"
        )
    elif (
        reference_run_contract_sha256 is not None
        or reference_run_contract_file_sha256 is not None
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} new task may not carry a historical run contract"
        )
    completion = _controller_path(
        selected["completion_path_controller"],
        f"{name}.completion_path_controller",
    )
    result_root_controller = _controller_path(
        selected["result_root_controller"],
        f"{name}.result_root_controller",
    )
    if (
        completion.name != "evaluation-result.json"
        or completion.parent != result_root_controller
        or completion.parent.name != slug
        or completion.parent.parent.name != str(milestone)
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} result paths are not condition/milestone bound"
        )
    return RoutedLadderEvaluationTask(
        task_id=expected_id,
        panel=panel,
        condition=condition,
        seed=seed,
        milestone_tokens=milestone,
        eligible_resource_ids=_eligible_resources(
            selected["eligible_resource_ids"],
            f"{name}.eligible_resource_ids",
        ),
        config_remote=_remote_path(selected["config_remote"], f"{name}.config_remote"),
        config_sha256=_sha256(
            selected["config_sha256"], f"{name}.config_sha256"
        ),
        h_config_remote=h_config_remote,
        h_config_sha256=h_config_sha256,
        a256_config_remote=a256_config_remote,
        a256_config_sha256=a256_config_sha256,
        reference_run_contract_sha256=reference_run_contract_sha256,
        reference_run_contract_file_sha256=(
            reference_run_contract_file_sha256
        ),
        training_source_revision=_git_sha(
            selected["training_source_revision"],
            f"{name}.training_source_revision",
        ),
        control_root_remote=_remote_path(
            selected["control_root_remote"], f"{name}.control_root_remote"
        ),
        checkpoint_root_gcs=_gcs_prefix(
            selected["checkpoint_root_gcs"], f"{name}.checkpoint_root_gcs"
        ),
        train_source_root_remote=_panel_remote(
            selected["train_source_root_remote"],
            panel,
            f"{name}.train_source_root_remote",
        ),
        eval_source_root_remote=_panel_remote(
            selected["eval_source_root_remote"],
            panel,
            f"{name}.eval_source_root_remote",
        ),
        train_source_object_prefix_gcs=_panel_gcs(
            selected["train_source_object_prefix_gcs"],
            panel,
            f"{name}.train_source_object_prefix_gcs",
        ),
        eval_source_object_prefix_gcs=_panel_gcs(
            selected["eval_source_object_prefix_gcs"],
            panel,
            f"{name}.eval_source_object_prefix_gcs",
        ),
        run_root_remote=_remote_path(
            selected["run_root_remote"], f"{name}.run_root_remote"
        ),
        result_root_remote=_remote_path(
            selected["result_root_remote"], f"{name}.result_root_remote"
        ),
        result_root_controller=result_root_controller,
        completion_path_controller=completion,
        environment=_task_environment(selected["environment"], f"{name}.environment"),
    )


def _cost_policy(
    value: Any,
    *,
    training_task_count: int,
    evaluation_task_count: int,
) -> CostPolicy:
    selected = _exact_mapping(
        value,
        {
            "hard_limit_usd",
            "contingency_usd",
            "already_spent_usd",
            "fixed_cost_reserve_usd",
            "training_attempt_reserve_usd",
            "evaluation_attempt_reserve_usd",
        },
        "cost_policy",
    )
    policy = CostPolicy(
        hard_limit_usd=_money(selected["hard_limit_usd"], "hard_limit_usd"),
        contingency_usd=_money(selected["contingency_usd"], "contingency_usd"),
        already_spent_usd=_money(
            selected["already_spent_usd"], "already_spent_usd"
        ),
        fixed_cost_reserve_usd=_money(
            selected["fixed_cost_reserve_usd"], "fixed_cost_reserve_usd"
        ),
        training_attempt_reserve_usd=_money(
            selected["training_attempt_reserve_usd"],
            "training_attempt_reserve_usd",
        ),
        evaluation_attempt_reserve_usd=_money(
            selected["evaluation_attempt_reserve_usd"],
            "evaluation_attempt_reserve_usd",
        ),
    )
    if policy.hard_limit_usd != 50.0 or policy.contingency_usd != 5.0:
        raise RoutedLadderCampaignPlanError(
            "activation ladder inherits the cumulative $50/$5 policy"
        )
    # Protocol v2 preserves the historical cumulative spend and the inherited
    # reserve model as measurement provenance.  It deliberately does not use
    # either value as a launch gate: TPU time is grant-funded and the operator
    # explicitly prioritized completing this extension over the retired
    # ready-three cash watchdog.
    return policy


def parse_h_reference_manifest(
    value: Any,
) -> tuple[Mapping[str, Any], tuple[HCheckpointReference, ...]]:
    """Validate the exact 18-cell read-only H checkpoint inventory."""

    selected = _exact_mapping(
        value,
        {
            "schema_version",
            "protocol_id",
            "cells",
            "manifest_sha256",
        },
        "h_reference_manifest",
    )
    if (
        selected["schema_version"] != H_REFERENCE_MANIFEST_SCHEMA_VERSION
        or selected["protocol_id"] != PROTOCOL_ID
    ):
        raise RoutedLadderCampaignPlanError("H reference manifest identity changed")
    if not isinstance(selected["cells"], list):
        raise RoutedLadderCampaignPlanError("H reference cells must be a list")
    body = {key: selected[key] for key in selected if key != "manifest_sha256"}
    claimed = _sha256(
        selected["manifest_sha256"], "h_reference_manifest.manifest_sha256"
    )
    if canonical_sha256(body) != claimed:
        raise RoutedLadderCampaignPlanError("H reference manifest self-hash changed")
    parsed: list[HCheckpointReference] = []
    for index, value_row in enumerate(selected["cells"]):
        name = f"h_reference_manifest.cells[{index}]"
        row = _exact_mapping(
            value_row,
            {
                "panel",
                "seed",
                "checkpoint_tokens",
                "training_source_revision",
                "h_config_remote",
                "h_config_sha256",
                "run_root_remote",
                "checkpoint_payload_root_gcs",
                "training_result",
                "training_result_sha256",
                "checkpoint_pointer",
                "checkpoint_pointer_sha256",
            },
            name,
        )
        panel = row["panel"]
        seed = row["seed"]
        milestone = row["checkpoint_tokens"]
        if panel not in PANELS or seed not in SEEDS or milestone not in MILESTONES:
            raise RoutedLadderCampaignPlanError(
                f"{name} scientific identity changed"
            )
        h_config_sha = _sha256(
            row["h_config_sha256"], f"{name}.h_config_sha256"
        )
        if h_config_sha != H_CONFIG_SHA256[panel]:
            raise RoutedLadderCampaignPlanError(
                f"{name} parent H config identity changed"
            )
        h_config_remote = _remote_path(
            row["h_config_remote"], f"{name}.h_config_remote"
        )
        if PurePosixPath(h_config_remote).name != f"gemma4_hybrid_{panel}_v1.yaml":
            raise RoutedLadderCampaignPlanError(
                f"{name} parent H config path changed"
            )
        parsed.append(
            HCheckpointReference(
                panel=panel,
                seed=seed,
                checkpoint_tokens=milestone,
                training_source_revision=_git_sha(
                    row["training_source_revision"],
                    f"{name}.training_source_revision",
                ),
                h_config_remote=h_config_remote,
                h_config_sha256=h_config_sha,
                run_root_remote=_remote_path(
                    row["run_root_remote"], f"{name}.run_root_remote"
                ),
                checkpoint_payload_root_gcs=_gcs_prefix(
                    row["checkpoint_payload_root_gcs"],
                    f"{name}.checkpoint_payload_root_gcs",
                ),
                training_result=_evidence(
                    row["training_result"], f"{name}.training_result"
                ),
                training_result_sha256=_sha256(
                    row["training_result_sha256"],
                    f"{name}.training_result_sha256",
                ),
                checkpoint_pointer=_evidence(
                    row["checkpoint_pointer"], f"{name}.checkpoint_pointer"
                ),
                checkpoint_pointer_sha256=_sha256(
                    row["checkpoint_pointer_sha256"],
                    f"{name}.checkpoint_pointer_sha256",
                ),
            )
        )
    cells = tuple(parsed)
    expected = {
        (panel, seed, milestone)
        for panel in PANELS
        for seed in SEEDS
        for milestone in MILESTONES
    }
    observed = {cell.identity for cell in cells}
    if len(cells) != H_REFERENCE_COUNT or observed != expected:
        raise RoutedLadderCampaignPlanError(
            "H reference manifest must contain exactly 18 unique cells"
        )
    for panel in PANELS:
        for seed in SEEDS:
            trajectory = [
                cell
                for cell in cells
                if cell.panel == panel and cell.seed == seed
            ]
            if (
                len(trajectory) != len(MILESTONES)
                or len(
                    {
                        (
                            cell.training_source_revision,
                            cell.h_config_remote,
                            cell.h_config_sha256,
                            cell.run_root_remote,
                            cell.checkpoint_payload_root_gcs,
                            cell.training_result,
                            cell.training_result_sha256,
                        )
                        for cell in trajectory
                    }
                )
                != 1
            ):
                raise RoutedLadderCampaignPlanError(
                    f"H {panel}/seed{seed} milestones do not share one "
                    "immutable training trajectory"
                )
            if (
                len({cell.checkpoint_pointer for cell in trajectory})
                != len(MILESTONES)
                or len(
                    {
                        cell.checkpoint_pointer_sha256
                        for cell in trajectory
                    }
                )
                != len(MILESTONES)
            ):
                raise RoutedLadderCampaignPlanError(
                    f"H {panel}/seed{seed} milestone pointers are not unique"
                )
    return copy.deepcopy(selected), tuple(
        sorted(cells, key=lambda item: item.identity)
    )


def parse_a256_reference_manifest(
    value: Any,
) -> tuple[
    Mapping[str, Any],
    tuple[A256CheckpointReference, ...],
    A256HistoricalCost,
]:
    """Validate the three milestones of historical SAE A256 seed zero."""

    selected = _exact_mapping(
        value,
        {
            "schema_version",
            "protocol_id",
            "source_protocol_id",
            "source_plan_sha256",
            "historical_training_cost",
            "cells",
            "manifest_sha256",
        },
        "a256_reference_manifest",
    )
    if (
        selected["schema_version"] != A256_REFERENCE_MANIFEST_SCHEMA_VERSION
        or selected["protocol_id"] != PROTOCOL_ID
        or selected["source_protocol_id"] != EXPERIMENT4_PROTOCOL_ID
    ):
        raise RoutedLadderCampaignPlanError(
            "A256 reference manifest identity changed"
        )
    _sha256(
        selected["source_plan_sha256"],
        "a256_reference_manifest.source_plan_sha256",
    )
    body = {
        key: selected[key] for key in selected if key != "manifest_sha256"
    }
    claimed = _sha256(
        selected["manifest_sha256"],
        "a256_reference_manifest.manifest_sha256",
    )
    if canonical_sha256(body) != claimed:
        raise RoutedLadderCampaignPlanError(
            "A256 reference manifest self-hash changed"
        )
    raw_cost = _exact_mapping(
        selected["historical_training_cost"],
        {
            "resource_id",
            "accelerator_type",
            "chip_count",
            "marginal_wall_seconds",
            "chip_seconds",
            "chip_seconds_scope",
            "chip_seconds_excludes",
        },
        "a256_reference_manifest.historical_training_cost",
    )
    exclusions = raw_cost["chip_seconds_excludes"]
    if (
        not isinstance(exclusions, list)
        or not exclusions
        or not all(
            isinstance(item, str) and item and "\x00" not in item
            for item in exclusions
        )
        or len(set(exclusions)) != len(exclusions)
    ):
        raise RoutedLadderCampaignPlanError(
            "A256 historical chip-second exclusions are invalid"
        )
    chip_count = _positive_int(
        raw_cost["chip_count"],
        "a256_reference_manifest.historical_training_cost.chip_count",
        maximum=1024,
    )
    marginal = _positive_number(
        raw_cost["marginal_wall_seconds"],
        "a256_reference_manifest.historical_training_cost."
        "marginal_wall_seconds",
    )
    chip_seconds = _positive_number(
        raw_cost["chip_seconds"],
        "a256_reference_manifest.historical_training_cost.chip_seconds",
    )
    if not math.isclose(
        chip_seconds,
        marginal * chip_count,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise RoutedLadderCampaignPlanError(
            "A256 historical chip-seconds do not equal wall time times chips"
        )
    resource_id = _safe_id(
        raw_cost["resource_id"],
        "a256_reference_manifest.historical_training_cost.resource_id",
    )
    accelerator_type = raw_cost["accelerator_type"]
    chip_seconds_scope = raw_cost["chip_seconds_scope"]
    if (
        not isinstance(accelerator_type, str)
        or not accelerator_type
        or not isinstance(chip_seconds_scope, str)
        or not chip_seconds_scope
    ):
        raise RoutedLadderCampaignPlanError(
            "A256 historical accelerator/cost scope is invalid"
        )
    cost = A256HistoricalCost(
        resource_id=resource_id,
        accelerator_type=accelerator_type,
        chip_count=chip_count,
        marginal_wall_seconds=marginal,
        chip_seconds=chip_seconds,
        chip_seconds_scope=chip_seconds_scope,
        chip_seconds_excludes=tuple(exclusions),
    )
    raw_cells = selected["cells"]
    if not isinstance(raw_cells, list):
        raise RoutedLadderCampaignPlanError(
            "A256 reference cells must be a list"
        )
    cells: list[A256CheckpointReference] = []
    for index, raw in enumerate(raw_cells):
        name = f"a256_reference_manifest.cells[{index}]"
        row = _exact_mapping(
            raw,
            {
                "panel",
                "condition",
                "seed",
                "checkpoint_tokens",
                "training_source_revision",
                "a256_config",
                "a256_config_sha256",
                "run_root_remote",
                "checkpoint_payload_root_gcs",
                "run_contract",
                "run_contract_sha256",
                "training_result",
                "training_result_sha256",
                "checkpoint_pointer",
                "checkpoint_pointer_sha256",
            },
            name,
        )
        if (
            row["panel"] != "sae"
            or row["condition"] != "A256ReLU"
            or row["seed"] != 0
            or row["checkpoint_tokens"] not in MILESTONES
        ):
            raise RoutedLadderCampaignPlanError(
                f"{name} is not historical SAE A256 seed zero"
            )
        config_sha = _sha256(
            row["a256_config_sha256"],
            f"{name}.a256_config_sha256",
        )
        if config_sha != A256_SAE_CONFIG_SHA256:
            raise RoutedLadderCampaignPlanError(
                f"{name} historical SAE A256 config identity changed"
            )
        config_evidence = _evidence(
            row["a256_config"],
            f"{name}.a256_config",
        )
        cells.append(
            A256CheckpointReference(
                checkpoint_tokens=row["checkpoint_tokens"],
                training_source_revision=_git_sha(
                    row["training_source_revision"],
                    f"{name}.training_source_revision",
                ),
                a256_config=config_evidence,
                a256_config_sha256=config_sha,
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
                checkpoint_pointer=_evidence(
                    row["checkpoint_pointer"],
                    f"{name}.checkpoint_pointer",
                ),
                checkpoint_pointer_sha256=_sha256(
                    row["checkpoint_pointer_sha256"],
                    f"{name}.checkpoint_pointer_sha256",
                ),
            )
        )
    identities = {cell.identity for cell in cells}
    expected = {
        ("sae", "A256ReLU", 0, milestone) for milestone in MILESTONES
    }
    if len(cells) != A256_REFERENCE_COUNT or identities != expected:
        raise RoutedLadderCampaignPlanError(
            "A256 reference manifest must contain exactly the three "
            "historical SAE seed-zero milestones"
        )
    if (
        len({cell.training_source_revision for cell in cells}) != 1
        or len({cell.a256_config for cell in cells}) != 1
        or len({cell.run_root_remote for cell in cells}) != 1
        or len({cell.checkpoint_payload_root_gcs for cell in cells}) != 1
        or len({cell.run_contract for cell in cells}) != 1
        or len({cell.run_contract_sha256 for cell in cells}) != 1
        or len({cell.training_result for cell in cells}) != 1
        or len({cell.training_result_sha256 for cell in cells}) != 1
    ):
        raise RoutedLadderCampaignPlanError(
            "A256 reference milestones do not share one historical trajectory"
        )
    if (
        len({cell.checkpoint_pointer for cell in cells}) != len(MILESTONES)
        or len({cell.checkpoint_pointer_sha256 for cell in cells})
        != len(MILESTONES)
    ):
        raise RoutedLadderCampaignPlanError(
            "A256 historical milestone pointers are not unique"
        )
    return (
        copy.deepcopy(selected),
        tuple(sorted(cells, key=lambda item: item.checkpoint_tokens)),
        cost,
    )


def parse_completed_trajectory_reference_manifest(
    value: Any,
) -> tuple[
    Mapping[str, Any],
    tuple[CompletedTrajectoryCheckpointReference, ...],
]:
    """Validate completed v2 trajectories imported read-only into recovery.

    The manifest is intentionally milestone-granular.  Every imported
    trajectory must contribute all three evaluation checkpoints and all
    lineage files are pinned by both their file hashes and scientific
    self-hashes.
    """

    selected = _exact_mapping(
        value,
        {
            "schema_version",
            "source_protocol_id",
            "target_protocol_id",
            "source_campaign_id",
            "source_plan_sha256",
            "cells",
            "manifest_sha256",
        },
        "completed_trajectory_reference_manifest",
    )
    if (
        selected["schema_version"]
        != COMPLETED_TRAJECTORY_REFERENCE_MANIFEST_SCHEMA_VERSION
        or selected["source_protocol_id"] != PROTOCOL_ID
        or selected["target_protocol_id"] != V3_PROTOCOL_ID
    ):
        raise RoutedLadderCampaignPlanError(
            "completed-trajectory reference manifest identity changed"
        )
    _safe_id(
        selected["source_campaign_id"],
        "completed_trajectory_reference_manifest.source_campaign_id",
    )
    _sha256(
        selected["source_plan_sha256"],
        "completed_trajectory_reference_manifest.source_plan_sha256",
    )
    body = {
        key: selected[key] for key in selected if key != "manifest_sha256"
    }
    claimed = _sha256(
        selected["manifest_sha256"],
        "completed_trajectory_reference_manifest.manifest_sha256",
    )
    if canonical_sha256(body) != claimed:
        raise RoutedLadderCampaignPlanError(
            "completed-trajectory reference manifest self-hash changed"
        )
    raw_cells = selected["cells"]
    if not isinstance(raw_cells, list) or not raw_cells:
        raise RoutedLadderCampaignPlanError(
            "completed-trajectory reference cells must be a nonempty list"
        )
    cells: list[CompletedTrajectoryCheckpointReference] = []
    for index, raw in enumerate(raw_cells):
        name = f"completed_trajectory_reference_manifest.cells[{index}]"
        row = _exact_mapping(
            raw,
            {
                "panel",
                "source_condition",
                "target_condition",
                "seed",
                "checkpoint_tokens",
                "training_source_revision",
                "config_remote",
                "config_sha256",
                "config_file_sha256",
                "run_root_remote",
                "checkpoint_payload_root_gcs",
                "run_contract",
                "run_contract_sha256",
                "training_result",
                "training_result_sha256",
                "checkpoint_pointer",
                "checkpoint_pointer_sha256",
                "resource_id",
                "accelerator_type",
            },
            name,
        )
        source_role = (
            row["panel"],
            row["source_condition"],
            row["seed"],
        )
        target_role = (
            row["panel"],
            row["target_condition"],
            row["seed"],
        )
        milestone = row["checkpoint_tokens"]
        if (
            source_role not in EXPECTED_TRAINING_ROLES
            or source_role[1] != "A512Softplus"
            or target_role != source_role
            or milestone not in MILESTONES
            or row["config_sha256"] != V2_CONFIG_SHA256.get(str(row["panel"]))
        ):
            raise RoutedLadderCampaignPlanError(
                f"{name} is not an exact A512Softplus v2-to-v3 import"
            )
        accelerator_type = row["accelerator_type"]
        if accelerator_type not in _CHIP_COUNT_BY_ACCELERATOR:
            raise RoutedLadderCampaignPlanError(
                f"{name} has an unsupported training accelerator"
            )
        cells.append(
            CompletedTrajectoryCheckpointReference(
                panel=str(row["panel"]),
                source_condition=str(row["source_condition"]),
                target_condition=str(row["target_condition"]),
                seed=int(row["seed"]),
                checkpoint_tokens=int(milestone),
                training_source_revision=_git_sha(
                    row["training_source_revision"],
                    f"{name}.training_source_revision",
                ),
                config_remote=_remote_path(
                    row["config_remote"], f"{name}.config_remote"
                ),
                config_sha256=_sha256(
                    row["config_sha256"], f"{name}.config_sha256"
                ),
                config_file_sha256=_sha256(
                    row["config_file_sha256"],
                    f"{name}.config_file_sha256",
                ),
                run_root_remote=_remote_path(
                    row["run_root_remote"], f"{name}.run_root_remote"
                ),
                checkpoint_payload_root_gcs=_gcs_prefix(
                    row["checkpoint_payload_root_gcs"],
                    f"{name}.checkpoint_payload_root_gcs",
                ),
                run_contract=_evidence(
                    row["run_contract"], f"{name}.run_contract"
                ),
                run_contract_sha256=_sha256(
                    row["run_contract_sha256"],
                    f"{name}.run_contract_sha256",
                ),
                training_result=_evidence(
                    row["training_result"], f"{name}.training_result"
                ),
                training_result_sha256=_sha256(
                    row["training_result_sha256"],
                    f"{name}.training_result_sha256",
                ),
                checkpoint_pointer=_evidence(
                    row["checkpoint_pointer"], f"{name}.checkpoint_pointer"
                ),
                checkpoint_pointer_sha256=_sha256(
                    row["checkpoint_pointer_sha256"],
                    f"{name}.checkpoint_pointer_sha256",
                ),
                resource_id=_safe_id(
                    row["resource_id"], f"{name}.resource_id"
                ),
                accelerator_type=str(accelerator_type),
            )
        )
    identities = {cell.identity for cell in cells}
    if len(identities) != len(cells):
        raise RoutedLadderCampaignPlanError(
            "completed-trajectory reference cells contain duplicates"
        )
    roles = {cell.trajectory_identity for cell in cells}
    expected_roles = {
        (panel, "A512Softplus", seed)
        for panel in PANELS
        for seed in SEEDS
    }
    if (
        roles != expected_roles
        or len(cells) != len(roles) * len(MILESTONES)
        or identities
        != {
            (*role, milestone)
            for role in roles
            for milestone in MILESTONES
        }
    ):
        raise RoutedLadderCampaignPlanError(
            "completed-trajectory references require all six Softplus "
            "trajectories at exactly 2M/5M/10M"
        )
    for role in roles:
        trajectory = [
            cell for cell in cells if cell.trajectory_identity == role
        ]
        shared = {
            (
                cell.training_source_revision,
                cell.config_remote,
                cell.config_sha256,
                cell.config_file_sha256,
                cell.run_root_remote,
                cell.checkpoint_payload_root_gcs,
                cell.run_contract,
                cell.run_contract_sha256,
                cell.training_result,
                cell.training_result_sha256,
                cell.resource_id,
                cell.accelerator_type,
            )
            for cell in trajectory
        }
        if len(shared) != 1:
            raise RoutedLadderCampaignPlanError(
                f"imported {role} milestones do not share one trajectory"
            )
        if (
            len({cell.checkpoint_pointer for cell in trajectory})
            != len(MILESTONES)
            or len(
                {
                    cell.checkpoint_pointer_sha256
                    for cell in trajectory
                }
            )
            != len(MILESTONES)
        ):
            raise RoutedLadderCampaignPlanError(
                f"imported {role} checkpoint pointers are not unique"
            )
    if len({cell.training_source_revision for cell in cells}) != 1:
        raise RoutedLadderCampaignPlanError(
            "completed-trajectory references must come from one source revision"
        )
    return copy.deepcopy(selected), tuple(
        sorted(cells, key=lambda item: item.identity)
    )


def parse_routed_ladder_campaign_plan(value: Any) -> RoutedLadderCampaignPlan:
    """Validate and freeze one exact routed-activation campaign plan."""

    selected = _exact_mapping(
        value,
        {
            "schema_version",
            "protocol_id",
            "campaign_id",
            "state_root",
            "working_directory",
            "python_executable",
            "remote_project_root",
            "remote_uv",
            "project_commit",
            "max_parallel_train",
            "max_parallel_eval",
            "retry_policy",
            "cost_policy",
            "environment",
            "resources",
            "resource_pool",
            "trajectories",
            "evaluations",
            "h_reference_manifest",
            "a256_reference_manifest",
        },
        "plan",
    )
    if (
        selected["schema_version"] != PLAN_SCHEMA_VERSION
        or selected["protocol_id"] != PROTOCOL_ID
    ):
        raise RoutedLadderCampaignPlanError("campaign protocol identity changed")
    project_commit = _git_sha(selected["project_commit"], "project_commit")
    short = project_commit[:12]
    state_root = _controller_path(selected["state_root"], "state_root")
    if tuple(state_root.parts[-5:]) != (
        "experiment5",
        "extensions",
        "routed-activation-ladder-v2",
        "controller",
        short,
    ):
        raise RoutedLadderCampaignPlanError(
            "state_root is outside the immutable activation-ladder namespace"
        )
    retry_raw = _exact_mapping(
        selected["retry_policy"],
        {
            "max_transient_attempts",
            "initial_backoff_seconds",
            "maximum_backoff_seconds",
        },
        "retry_policy",
    )
    retry = RetryPolicy(
        max_transient_attempts=_positive_int(
            retry_raw["max_transient_attempts"],
            "retry_policy.max_transient_attempts",
            maximum=100,
        ),
        initial_backoff_seconds=_money(
            retry_raw["initial_backoff_seconds"],
            "retry_policy.initial_backoff_seconds",
        ),
        maximum_backoff_seconds=_money(
            retry_raw["maximum_backoff_seconds"],
            "retry_policy.maximum_backoff_seconds",
        ),
    )
    if retry.maximum_backoff_seconds < retry.initial_backoff_seconds:
        raise RoutedLadderCampaignPlanError("retry backoff decreases")
    raw_resources = selected["resources"]
    raw_training = selected["trajectories"]
    raw_evaluations = selected["evaluations"]
    if (
        not isinstance(raw_resources, list)
        or not isinstance(raw_training, list)
        or not isinstance(raw_evaluations, list)
    ):
        raise RoutedLadderCampaignPlanError(
            "resources, trajectories, and evaluations must be lists"
        )
    resources = tuple(
        _resource(row, index, project_commit)
        for index, row in enumerate(raw_resources)
    )
    try:
        resource_pool = parse_routed_ladder_resource_pool(
            selected["resource_pool"],
            project_commit=project_commit,
        )
    except RoutedLadderResourcePoolError as error:
        raise RoutedLadderCampaignPlanError(
            f"resource pool is invalid: {error}"
        ) from error
    if resources != resource_pool.resources:
        raise RoutedLadderCampaignPlanError(
            "plan resources differ from the immutable resource pool"
        )
    trajectories = tuple(
        _training_task(row, index) for index, row in enumerate(raw_training)
    )
    evaluations = tuple(
        _evaluation_task(row, index) for index, row in enumerate(raw_evaluations)
    )
    h_manifest, h_references = parse_h_reference_manifest(
        selected["h_reference_manifest"]
    )
    (
        a256_manifest,
        a256_references,
        a256_historical_cost,
    ) = parse_a256_reference_manifest(selected["a256_reference_manifest"])
    environment = _task_environment(selected["environment"], "environment")
    plan = RoutedLadderCampaignPlan(
        campaign_id=_safe_id(selected["campaign_id"], "campaign_id"),
        state_root=state_root,
        working_directory=_controller_path(
            selected["working_directory"], "working_directory"
        ),
        python_executable=_controller_path(
            selected["python_executable"], "python_executable"
        ),
        remote_project_root=_remote_path(
            selected["remote_project_root"], "remote_project_root"
        ),
        remote_uv=_remote_path(selected["remote_uv"], "remote_uv"),
        project_commit=project_commit,
        max_parallel_train=_positive_int(
            selected["max_parallel_train"],
            "max_parallel_train",
            maximum=len(resources),
        ),
        max_parallel_eval=_positive_int(
            selected["max_parallel_eval"],
            "max_parallel_eval",
            maximum=len(resources),
        ),
        retry_policy=retry,
        cost_policy=_cost_policy(
            selected["cost_policy"],
            training_task_count=len(trajectories),
            evaluation_task_count=EVALUATION_BUNDLE_COUNT,
        ),
        environment=environment,
        resources=resources,
        resource_pool=resource_pool,
        trajectories=trajectories,
        evaluations=evaluations,
        h_reference_manifest=h_manifest,
        h_references=h_references,
        a256_reference_manifest=a256_manifest,
        a256_references=a256_references,
        a256_historical_cost=a256_historical_cost,
        plan_sha256=canonical_sha256(selected),
        source=copy.deepcopy(selected),
    )
    _validate_plan_relations(plan)
    return plan


def _validate_plan_relations(plan: RoutedLadderCampaignPlan) -> None:
    if (
        not MIN_RESOURCE_COUNT <= len(plan.resources) <= MAX_RESOURCE_COUNT
        or len(plan.trajectories) != TRAINING_TASK_COUNT
        or len(plan.evaluations) != EVALUATION_TASK_COUNT
        or len(plan.h_references) != H_REFERENCE_COUNT
        or len(plan.a256_references) != A256_REFERENCE_COUNT
    ):
        raise RoutedLadderCampaignPlanError(
            "campaign must contain 3..18 resources, the exact protocol-v2 "
            f"{TRAINING_TASK_COUNT}-trajectory/{EVALUATION_TASK_COUNT}-cell "
            "matrix, 18 H references, and three SAE A256 references"
        )
    resource_ids = [row.resource_id for row in plan.resources]
    tpu_names = [row.tpu_name for row in plan.resources]
    if (
        len(set(resource_ids)) != len(resource_ids)
        or len(set(tpu_names)) != len(tpu_names)
    ):
        raise RoutedLadderCampaignPlanError(
            "resource IDs and TPU names must be unique"
        )
    training_owner_count = len(
        {
            task.eligible_resource_ids[0]
            for task in plan.trajectories
            if len(task.eligible_resource_ids) == 1
        }
    )
    evaluation_owner_count = len(
        {
            task.eligible_resource_ids[0]
            for task in plan.evaluations
            if len(task.eligible_resource_ids) == 1
        }
    )
    if plan.max_parallel_train != min(
        training_owner_count,
        len(plan.trajectories),
    ):
        raise RoutedLadderCampaignPlanError(
            "max_parallel_train must consume every training owner"
        )
    if plan.max_parallel_eval != min(
        evaluation_owner_count,
        len(plan.evaluations),
    ):
        raise RoutedLadderCampaignPlanError(
            "max_parallel_eval must consume every evaluation owner"
        )
    by_id = {row.resource_id: row for row in plan.resources}
    active_ids = {
        resource_id
        for task in (*plan.trajectories, *plan.evaluations)
        for resource_id in task.eligible_resource_ids
    }
    if active_ids != set(resource_ids):
        raise RoutedLadderCampaignPlanError(
            "every admitted resource must own at least one campaign task"
        )
    if any(by_id[item].cache_profile != "paired" for item in active_ids):
        raise RoutedLadderCampaignPlanError(
            "every active activation-ladder resource requires the paired cache"
        )
    observed_training = {
        (task.panel, task.condition, task.seed) for task in plan.trajectories
    }
    if observed_training != EXPECTED_TRAINING_ROLES:
        raise RoutedLadderCampaignPlanError("training matrix is not exact")
    observed_evaluations = {
        (task.panel, task.condition, task.seed, task.milestone_tokens)
        for task in plan.evaluations
    }
    if observed_evaluations != EXPECTED_EVALUATION_ROLES:
        raise RoutedLadderCampaignPlanError("evaluation matrix is not exact")
    all_tasks = (*plan.trajectories, *plan.evaluations)
    if len({task.task_id for task in all_tasks}) != len(all_tasks):
        raise RoutedLadderCampaignPlanError("task IDs are not unique")
    for task in all_tasks:
        if (
            len(task.eligible_resource_ids) != 1
            or task.eligible_resource_ids[0] not in active_ids
        ):
            raise RoutedLadderCampaignPlanError(
                f"{task.task_id} must have one immutable resource owner"
            )
        owner_id = task.eligible_resource_ids[0]
        cohort = plan.resource_pool.cohort(task.panel, task.seed)
        if owner_id not in cohort:
            raise RoutedLadderCampaignPlanError(
                f"{task.task_id} owner is outside its immutable cohort plane"
            )
        plane = plan.resource_pool.plane(owner_id)
        namespace = f"/{EXTENSION_NAMESPACE}/{plan.project_commit[:12]}/"
        values = [task.control_root_remote, str(task.completion_path_controller)]
        if isinstance(task, RoutedLadderTrainingTask):
            values.append(task.checkpoint_root_gcs)
        else:
            values.extend(
                (task.result_root_remote, str(task.result_root_controller))
            )
            if task.condition in CONDITIONS and not (
                task.panel == "sae"
                and task.condition == "A256ReLU"
                and task.seed == 0
            ):
                values.extend((task.run_root_remote, task.checkpoint_root_gcs))
        if any(namespace not in value for value in values):
            raise RoutedLadderCampaignPlanError(
                f"{task.task_id} escapes the immutable extension namespace"
            )
        try:
            task.completion_path_controller.relative_to(
                plane.controller_artifact_root
            )
            PurePosixPath(task.control_root_remote).relative_to(
                PurePosixPath(plane.remote_artifact_root)
            )
            if isinstance(task, RoutedLadderEvaluationTask):
                task.result_root_controller.relative_to(
                    plane.controller_artifact_root
                )
                PurePosixPath(task.result_root_remote).relative_to(
                    PurePosixPath(plane.remote_artifact_root)
                )
        except ValueError as error:
            raise RoutedLadderCampaignPlanError(
                f"{task.task_id} output paths cross its owner artifact plane"
            ) from error
        if (
            isinstance(task, RoutedLadderTrainingTask)
            or (
                isinstance(task, RoutedLadderEvaluationTask)
                and task.condition in CONDITIONS
                and not (
                    task.panel == "sae"
                    and task.condition == "A256ReLU"
                    and task.seed == 0
                )
            )
        ) and not task.checkpoint_root_gcs.startswith(
            plane.bulk_gcs_root + "/"
        ):
            raise RoutedLadderCampaignPlanError(
                f"{task.task_id} checkpoint path crosses its owner object plane"
            )
    training_by_role = {
        (task.panel, task.condition, task.seed): task for task in plan.trajectories
    }
    h_by_role = {row.identity: row for row in plan.h_references}
    a256_by_milestone = {
        row.checkpoint_tokens: row for row in plan.a256_references
    }
    for task in plan.evaluations:
        historical_a256 = (
            task.panel == "sae"
            and task.condition == "A256ReLU"
            and task.seed == 0
        )
        if historical_a256:
            parent_a256 = a256_by_milestone[task.milestone_tokens]
            if (
                task.run_root_remote != parent_a256.run_root_remote
                or task.checkpoint_root_gcs
                != parent_a256.checkpoint_payload_root_gcs
                or task.a256_config_remote
                != parent_a256.a256_config.remote_path
                or task.a256_config_sha256
                != parent_a256.a256_config_sha256
                or task.reference_run_contract_sha256
                != parent_a256.run_contract_sha256
                or task.reference_run_contract_file_sha256
                != parent_a256.run_contract.sha256
                or task.training_source_revision
                != parent_a256.training_source_revision
            ):
                raise RoutedLadderCampaignPlanError(
                    f"{task.task_id} is not bound to the immutable "
                    "historical SAE A256 checkpoint"
                )
        elif task.condition in CONDITIONS:
            parent = training_by_role[(task.panel, task.condition, task.seed)]
            if (
                task.run_root_remote
                != str(
                    PurePosixPath(parent.control_root_remote)
                    / "runs"
                    / parent.run_id
                )
                or task.config_remote != parent.config_remote
                or task.config_sha256 != parent.config_sha256
                or task.checkpoint_root_gcs != parent.checkpoint_root_gcs
                or task.training_source_revision != plan.project_commit
            ):
                raise RoutedLadderCampaignPlanError(
                    f"{task.task_id} is not bound to its training trajectory"
                )
        else:
            parent_h = h_by_role[
                (task.panel, task.seed, task.milestone_tokens)
            ]
            if (
                task.run_root_remote != parent_h.run_root_remote
                or task.checkpoint_root_gcs
                != parent_h.checkpoint_payload_root_gcs
                or task.h_config_remote != parent_h.h_config_remote
                or task.h_config_sha256 != parent_h.h_config_sha256
                or task.training_source_revision
                != parent_h.training_source_revision
            ):
                raise RoutedLadderCampaignPlanError(
                    f"{task.task_id} is not bound to its immutable H checkpoint"
                )
    for panel in PANELS:
        for seed in SEEDS:
            bundle = [
                task
                for task in plan.evaluations
                if task.panel == panel and task.seed == seed
            ]
            if (
                len(bundle) != len(EVALUATION_CONDITIONS) * len(MILESTONES)
                or len(
                    {
                        task.eligible_resource_ids[0]
                        for task in bundle
                    }
                )
                != 1
            ):
                raise RoutedLadderCampaignPlanError(
                    f"{panel}/seed{seed} evaluation rows do not share one "
                    "resident bundle owner"
                )


def load_routed_ladder_campaign_plan(
    path: str | Path,
) -> RoutedLadderCampaignPlan:
    selected = Path(path)
    try:
        value = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RoutedLadderCampaignPlanError(
            f"cannot read activation-ladder plan: {selected}"
        ) from error
    return parse_routed_ladder_campaign_plan(value)


def load_h_reference_manifest(path: str | Path) -> Mapping[str, Any]:
    selected = Path(path)
    try:
        value = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RoutedLadderCampaignPlanError(
            f"cannot read H reference manifest: {selected}"
        ) from error
    manifest, _cells = parse_h_reference_manifest(value)
    return manifest


def load_a256_reference_manifest(path: str | Path) -> Mapping[str, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RoutedLadderCampaignPlanError(
            f"cannot read A256 reference manifest: {selected}"
        ) from error
    manifest, _cells, _cost = parse_a256_reference_manifest(value)
    return manifest


def load_completed_trajectory_reference_manifest(
    path: str | Path,
) -> Mapping[str, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RoutedLadderCampaignPlanError(
            "cannot read completed-trajectory reference manifest: "
            f"{selected}"
        ) from error
    manifest, _cells = parse_completed_trajectory_reference_manifest(value)
    return manifest


def _expected_milestone_progress(
    result: Mapping[str, Any],
    checkpoint_tokens: int,
) -> tuple[int, int]:
    identity = result.get("run_identity")
    plan = identity.get("plan") if isinstance(identity, Mapping) else None
    if not isinstance(plan, Mapping):
        raise RoutedLadderCampaignStateError(
            "H training result lacks its immutable worker plan"
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
        or isinstance(total, bool)
        or not isinstance(total, int)
        or total != TRAIN_TOKENS
        or isinstance(planned_updates, bool)
        or not isinstance(planned_updates, int)
        or planned_updates != math.ceil(total / (microbatch * accumulation))
    ):
        raise RoutedLadderCampaignStateError(
            "H training result has an invalid exact-token plan"
        )
    batch = microbatch * accumulation
    step = math.ceil(checkpoint_tokens / batch)
    return step, min(step * batch, total)


def _validate_h_checkpoint_lineage(
    cell: HCheckpointReference,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result, _result_payload = _read_self_hashed_json(
        cell.training_result.controller_path,
        hash_field="result_sha256",
    )
    pointer, _pointer_payload = _read_self_hashed_json(
        cell.checkpoint_pointer.controller_path,
        hash_field="pointer_sha256",
    )
    identity = result.get("run_identity")
    run_plan = identity.get("plan") if isinstance(identity, Mapping) else None
    if (
        result.get("schema_version")
        != "gemma4-contribution-sae-training-result-v1"
        or result.get("status") not in {"completed", "already_complete"}
        or result.get("source_revision") != cell.training_source_revision
        or result.get("tokens") != TRAIN_TOKENS
        or not isinstance(identity, Mapping)
        or not isinstance(run_plan, Mapping)
        or run_plan.get("phase_name") != "routed_shared_hybrid_10m"
        or run_plan.get("condition") != "H"
        or run_plan.get("seed") != cell.seed
        or run_plan.get("total_tokens") != TRAIN_TOKENS
        or result.get("result_sha256") != cell.training_result_sha256
    ):
        raise RoutedLadderCampaignStateError(
            f"H training lineage changed: {cell.identity}"
        )
    expected_step, expected_tokens = _expected_milestone_progress(
        result,
        cell.checkpoint_tokens,
    )
    try:
        checked_pointer = validate_pointer(
            pointer,
            expected_kind="latest",
            payload_root=cell.checkpoint_payload_root_gcs,
            run_identity=identity,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise RoutedLadderCampaignStateError(
            f"H checkpoint pointer is invalid: {cell.identity}"
        ) from error
    if (
        checked_pointer["pointer_sha256"] != cell.checkpoint_pointer_sha256
        or checked_pointer["step"] != expected_step
        or checked_pointer["tokens_seen"] != expected_tokens
    ):
        raise RoutedLadderCampaignStateError(
            f"H checkpoint progress changed: {cell.identity}"
        )
    return result, checked_pointer


def _validate_a256_checkpoint_lineage(
    cell: A256CheckpointReference,
    cost: A256HistoricalCost,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_contract, _contract_payload = _read_self_hashed_json(
        cell.run_contract.controller_path,
        hash_field="run_contract_sha256",
    )
    result, _result_payload = _read_self_hashed_json(
        cell.training_result.controller_path,
        hash_field="result_sha256",
    )
    pointer, _pointer_payload = _read_self_hashed_json(
        cell.checkpoint_pointer.controller_path,
        hash_field="pointer_sha256",
    )
    identity = result.get("run_identity")
    run_plan = identity.get("plan") if isinstance(identity, Mapping) else None
    if (
        result.get("schema_version")
        != "gemma4-contribution-sae-training-result-v1"
        or result.get("status") not in {"completed", "already_complete"}
        or result.get("tokens") != TRAIN_TOKENS
        or not isinstance(identity, Mapping)
        or not isinstance(run_plan, Mapping)
        or run_plan.get("phase_name") != "learning_horizon_10m"
        or run_plan.get("condition") != "A"
        or run_plan.get("seed") != 0
        or run_plan.get("total_tokens") != TRAIN_TOKENS
        or result.get("result_sha256") != cell.training_result_sha256
        or result.get("marginal_wall_seconds")
        != cost.marginal_wall_seconds
        or result.get("chip_seconds") != cost.chip_seconds
        or result.get("chip_seconds_scope") != cost.chip_seconds_scope
        or result.get("chip_seconds_excludes")
        != list(cost.chip_seconds_excludes)
        or run_contract.get("run_contract_sha256")
        != cell.run_contract_sha256
        or run_contract.get("project_commit")
        != cell.training_source_revision
        or run_contract.get("phase") != "learning_horizon_10m"
        or run_contract.get("scientific_model") != "sae"
        or run_contract.get("condition") != "A"
        or run_contract.get("seed") != 0
        or run_contract.get("config_file_sha256")
        != cell.a256_config.sha256
    ):
        raise RoutedLadderCampaignStateError(
            f"historical SAE A256 lineage changed: {cell.identity}"
        )
    expected_step, expected_tokens = _expected_milestone_progress(
        result,
        cell.checkpoint_tokens,
    )
    try:
        checked_pointer = validate_pointer(
            pointer,
            expected_kind="latest",
            payload_root=cell.checkpoint_payload_root_gcs,
            run_identity=identity,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise RoutedLadderCampaignStateError(
            f"historical SAE A256 checkpoint is invalid: {cell.identity}"
        ) from error
    if (
        checked_pointer["pointer_sha256"] != cell.checkpoint_pointer_sha256
        or checked_pointer["step"] != expected_step
        or checked_pointer["tokens_seen"] != expected_tokens
    ):
        raise RoutedLadderCampaignStateError(
            f"historical SAE A256 checkpoint progress changed: {cell.identity}"
        )
    return result, checked_pointer


def validate_completed_checkpoint_lineage(
    cell: CompletedTrajectoryCheckpointReference,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one imported v2 milestone without rebinding its identity."""

    run_contract, contract_payload = _read_self_hashed_json(
        cell.run_contract.controller_path,
        hash_field="run_contract_sha256",
    )
    result, result_payload = _read_self_hashed_json(
        cell.training_result.controller_path,
        hash_field="result_sha256",
    )
    pointer, pointer_payload = _read_self_hashed_json(
        cell.checkpoint_pointer.controller_path,
        hash_field="pointer_sha256",
    )
    identity = result.get("run_identity")
    run_plan = identity.get("plan") if isinstance(identity, Mapping) else None
    if (
        hashlib.sha256(contract_payload).hexdigest()
        != cell.run_contract.sha256
        or hashlib.sha256(result_payload).hexdigest()
        != cell.training_result.sha256
        or hashlib.sha256(pointer_payload).hexdigest()
        != cell.checkpoint_pointer.sha256
        or cell.config_sha256 != V2_CONFIG_SHA256[cell.panel]
        or result.get("schema_version")
        != "gemma4-contribution-sae-training-result-v1"
        or result.get("status") not in {"completed", "already_complete"}
        or result.get("tokens") != TRAIN_TOKENS
        or result.get("source_revision") != cell.training_source_revision
        or result.get("result_sha256") != cell.training_result_sha256
        or not isinstance(identity, Mapping)
        or not isinstance(run_plan, Mapping)
        or run_plan.get("phase_name") != PHASE_NAME
        or run_plan.get("condition") != cell.source_condition
        or run_plan.get("seed") != cell.seed
        or run_plan.get("total_tokens") != TRAIN_TOKENS
        or run_contract.get("run_contract_sha256")
        != cell.run_contract_sha256
        or run_contract.get("project_commit")
        != cell.training_source_revision
        or run_contract.get("phase") != PHASE_NAME
        or run_contract.get("scientific_model") != cell.panel
        or run_contract.get("condition") != cell.source_condition
        or run_contract.get("seed") != cell.seed
        or run_contract.get("config_file_sha256")
        != cell.config_file_sha256
        or run_contract.get("checkpoint_payload_root")
        != cell.checkpoint_payload_root_gcs
    ):
        raise RoutedLadderCampaignStateError(
            f"completed trajectory lineage changed: {cell.identity}"
        )
    _training_cost(
        result=result,
        resource_id=cell.resource_id,
        accelerator_type=cell.accelerator_type,
    )
    expected_step, expected_tokens = _expected_milestone_progress(
        result,
        cell.checkpoint_tokens,
    )
    try:
        checked_pointer = validate_pointer(
            pointer,
            expected_kind="latest",
            payload_root=cell.checkpoint_payload_root_gcs,
            run_identity=identity,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise RoutedLadderCampaignStateError(
            f"completed trajectory checkpoint is invalid: {cell.identity}"
        ) from error
    if (
        checked_pointer["pointer_sha256"]
        != cell.checkpoint_pointer_sha256
        or checked_pointer["step"] != expected_step
        or checked_pointer["tokens_seen"] != expected_tokens
    ):
        raise RoutedLadderCampaignStateError(
            f"completed trajectory checkpoint progress changed: "
            f"{cell.identity}"
        )
    return result, checked_pointer


def build_h_reference_manifest(
    core_plan: Experiment5CampaignPlan,
) -> dict[str, Any]:
    """Inventory the 18 immutable H checkpoints from one sealed core campaign.

    This is a read-only inventory operation.  It neither restores checkpoints
    nor creates cloud resources; each row is bound to the raw controller file
    hash and the artifact's scientific self-hash.
    """

    h_tasks = {
        (task.panel, task.seed): task
        for task in core_plan.trajectories
        if task.condition == "H"
    }
    expected_tasks = {(panel, seed) for panel in PANELS for seed in SEEDS}
    if set(h_tasks) != expected_tasks:
        raise RoutedLadderCampaignPlanError(
            "core campaign lacks the exact six H training trajectories"
        )
    cells: list[dict[str, Any]] = []
    for panel in PANELS:
        for seed in SEEDS:
            task = h_tasks[(panel, seed)]
            completion = inspect_core_training_completion(core_plan, task)
            if completion is None:
                raise RoutedLadderCampaignPlanError(
                    f"core H training is incomplete: {panel}/seed{seed}"
                )
            result, result_payload = _read_self_hashed_json(
                task.completion_path_controller,
                hash_field="result_sha256",
            )
            remote_run_root = str(
                PurePosixPath(task.control_root_remote) / "runs" / task.run_id
            )
            controller_checkpoint_root = (
                task.completion_path_controller.parent / "checkpoints"
            )
            remote_checkpoint_root = str(
                PurePosixPath(remote_run_root) / "checkpoints"
            )
            for milestone in MILESTONES:
                pointer_path = milestone_pointer_path(
                    controller_checkpoint_root,
                    milestone,
                )
                pointer, pointer_payload = _read_self_hashed_json(
                    pointer_path,
                    hash_field="pointer_sha256",
                )
                row = {
                    "panel": panel,
                    "seed": seed,
                    "checkpoint_tokens": milestone,
                    "training_source_revision": core_plan.project_commit,
                    "h_config_remote": task.config_remote,
                    "h_config_sha256": H_CONFIG_SHA256[panel],
                    "run_root_remote": remote_run_root,
                    "checkpoint_payload_root_gcs": task.checkpoint_root_gcs,
                    "training_result": {
                        "controller_path": str(task.completion_path_controller),
                        "remote_path": str(
                            PurePosixPath(remote_run_root)
                            / "training-result.json"
                        ),
                        "sha256": hashlib.sha256(result_payload).hexdigest(),
                    },
                    "training_result_sha256": result["result_sha256"],
                    "checkpoint_pointer": {
                        "controller_path": str(pointer_path),
                        "remote_path": str(
                            milestone_pointer_path(
                                remote_checkpoint_root,
                                milestone,
                            )
                        ),
                        "sha256": hashlib.sha256(pointer_payload).hexdigest(),
                    },
                    "checkpoint_pointer_sha256": pointer["pointer_sha256"],
                }
                cells.append(row)
    body = {
        "schema_version": H_REFERENCE_MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "cells": cells,
    }
    manifest = {**body, "manifest_sha256": canonical_sha256(body)}
    _selected, parsed = parse_h_reference_manifest(manifest)
    for cell in parsed:
        _validate_h_checkpoint_lineage(cell)
    return manifest


def write_h_reference_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[HCheckpointReference, ...]]:
    parsed, cells = parse_h_reference_manifest(manifest)
    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    if selected.exists():
        raise RoutedLadderCampaignPlanError(
            f"refusing to overwrite H reference manifest: {selected}"
        )
    selected.write_text(
        yaml.safe_dump(dict(manifest), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return parsed, cells


def build_a256_reference_manifest(
    horizon_plan: DirectCampaignPlan,
) -> dict[str, Any]:
    """Inventory historical SAE A seed zero without creating or restoring it."""

    matches = [
        task
        for task in horizon_plan.trajectories
        if task.scientific_model == "sae"
        and task.condition == "A"
        and task.seed == 0
    ]
    if len(matches) != 1:
        raise RoutedLadderCampaignPlanError(
            "horizon plan lacks one exact historical SAE A seed-zero trajectory"
        )
    task = matches[0]
    completion = inspect_horizon_training_completion(task)
    if completion is None:
        raise RoutedLadderCampaignPlanError(
            "historical SAE A256 training is incomplete"
        )
    result, result_payload = _read_self_hashed_json(
        task.completion_path_controller,
        hash_field="result_sha256",
    )
    environment = horizon_plan.environment_dict()
    training_source_revision = _git_sha(
        environment.get("TRAINING_PROJECT_COMMIT"),
        "horizon_plan.environment.TRAINING_PROJECT_COMMIT",
    )
    remote_horizon_root = PurePosixPath(task.control_root_remote).parents[1]
    try:
        relative_config = PurePosixPath(task.config_remote).relative_to(
            remote_horizon_root
        )
    except ValueError as error:
        raise RoutedLadderCampaignPlanError(
            "historical SAE config is outside its horizon artifact plane"
        ) from error
    try:
        controller_horizon_root = task.completion_path_controller.parents[4]
    except IndexError as error:
        raise RoutedLadderCampaignPlanError(
            "historical SAE completion path has no horizon artifact root"
        ) from error
    controller_config = controller_horizon_root / Path(*relative_config.parts)
    try:
        config = load_learning_horizon_config(
            controller_config,
            expected_scientific_model="sae",
        )
    except LearningHorizonConfigError as error:
        raise RoutedLadderCampaignPlanError(
            f"historical SAE A256 config is invalid: {error}"
        ) from error
    if config.config_hash != A256_SAE_CONFIG_SHA256:
        raise RoutedLadderCampaignPlanError(
            "historical SAE A256 config hash changed"
        )
    try:
        config_payload = controller_config.read_bytes()
    except OSError as error:
        raise RoutedLadderCampaignPlanError(
            "cannot read historical SAE A256 config"
        ) from error
    resource = horizon_plan.resource(task.resource_id)
    marginal = _positive_number(
        result.get("marginal_wall_seconds"),
        "historical A256 marginal_wall_seconds",
    )
    chip_seconds = _positive_number(
        result.get("chip_seconds"),
        "historical A256 chip_seconds",
    )
    raw_chip_count = chip_seconds / marginal
    chip_count = round(raw_chip_count)
    if (
        chip_count <= 0
        or not math.isclose(
            raw_chip_count,
            chip_count,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise RoutedLadderCampaignPlanError(
            "historical A256 result has non-integral chip accounting"
        )
    chip_scope = result.get("chip_seconds_scope")
    chip_excludes = result.get("chip_seconds_excludes")
    if (
        not isinstance(chip_scope, str)
        or not chip_scope
        or not isinstance(chip_excludes, list)
        or not chip_excludes
        or not all(isinstance(item, str) and item for item in chip_excludes)
    ):
        raise RoutedLadderCampaignPlanError(
            "historical A256 result lacks explicit chip-second scope"
        )
    run_root = str(
        PurePosixPath(task.control_root_remote) / "runs" / task.run_id
    )
    run_contract_path = task.completion_path_controller.parent / "run-contract.json"
    run_contract, run_contract_payload = _read_self_hashed_json(
        run_contract_path,
        hash_field="run_contract_sha256",
    )
    if (
        run_contract.get("project_commit") != training_source_revision
        or run_contract.get("phase") != "learning_horizon_10m"
        or run_contract.get("scientific_model") != "sae"
        or run_contract.get("condition") != "A"
        or run_contract.get("seed") != 0
        or run_contract.get("config_file_sha256")
        != hashlib.sha256(config_payload).hexdigest()
    ):
        raise RoutedLadderCampaignPlanError(
            "historical SAE A256 run contract changed"
        )
    controller_checkpoint_root = (
        task.completion_path_controller.parent / "checkpoints"
    )
    remote_checkpoint_root = str(PurePosixPath(run_root) / "checkpoints")
    config_evidence = {
        "controller_path": str(controller_config),
        "remote_path": task.config_remote,
        "sha256": hashlib.sha256(config_payload).hexdigest(),
    }
    training_evidence = {
        "controller_path": str(task.completion_path_controller),
        "remote_path": str(
            PurePosixPath(run_root) / "training-result.json"
        ),
        "sha256": hashlib.sha256(result_payload).hexdigest(),
    }
    cells: list[dict[str, Any]] = []
    for milestone in MILESTONES:
        pointer_path = milestone_pointer_path(
            controller_checkpoint_root,
            milestone,
        )
        pointer, pointer_payload = _read_self_hashed_json(
            pointer_path,
            hash_field="pointer_sha256",
        )
        cells.append(
            {
                "panel": "sae",
                "condition": "A256ReLU",
                "seed": 0,
                "checkpoint_tokens": milestone,
                "training_source_revision": training_source_revision,
                "a256_config": config_evidence,
                "a256_config_sha256": config.config_hash,
                "run_root_remote": run_root,
                "checkpoint_payload_root_gcs": task.checkpoint_root_gcs,
                "run_contract": {
                    "controller_path": str(run_contract_path),
                    "remote_path": str(
                        PurePosixPath(run_root) / "run-contract.json"
                    ),
                    "sha256": hashlib.sha256(
                        run_contract_payload
                    ).hexdigest(),
                },
                "run_contract_sha256": run_contract["run_contract_sha256"],
                "training_result": training_evidence,
                "training_result_sha256": result["result_sha256"],
                "checkpoint_pointer": {
                    "controller_path": str(pointer_path),
                    "remote_path": str(
                        milestone_pointer_path(
                            remote_checkpoint_root,
                            milestone,
                        )
                    ),
                    "sha256": hashlib.sha256(pointer_payload).hexdigest(),
                },
                "checkpoint_pointer_sha256": pointer["pointer_sha256"],
            }
        )
    body = {
        "schema_version": A256_REFERENCE_MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": EXPERIMENT4_PROTOCOL_ID,
        "source_plan_sha256": horizon_plan.plan_sha256,
        "historical_training_cost": {
            "resource_id": resource.resource_id,
            "accelerator_type": resource.accelerator_type,
            "chip_count": chip_count,
            "marginal_wall_seconds": marginal,
            "chip_seconds": chip_seconds,
            "chip_seconds_scope": chip_scope,
            "chip_seconds_excludes": chip_excludes,
        },
        "cells": cells,
    }
    manifest = {**body, "manifest_sha256": canonical_sha256(body)}
    _selected, parsed, cost = parse_a256_reference_manifest(manifest)
    for cell in parsed:
        _validate_a256_checkpoint_lineage(cell, cost)
    return manifest


def write_a256_reference_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    tuple[A256CheckpointReference, ...],
    A256HistoricalCost,
]:
    parsed, cells, cost = parse_a256_reference_manifest(manifest)
    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    if selected.exists():
        raise RoutedLadderCampaignPlanError(
            f"refusing to overwrite A256 reference manifest: {selected}"
        )
    selected.write_text(
        yaml.safe_dump(dict(manifest), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return parsed, cells, cost


def build_completed_trajectory_reference_manifest(
    source_plan: RoutedLadderCampaignPlan,
    *,
    roles: set[tuple[str, str, int]] | None = None,
) -> dict[str, Any]:
    """Inventory completed new trajectories from a prior immutable campaign.

    Incomplete, failed, and merely checkpointed trajectories are never
    imported.  Protocol v3 requires the exact six completed Softplus
    trajectories; ``roles`` exists only so callers can state that requirement
    explicitly and cannot be used to authorize a subset.
    """

    importable_roles = {
        (panel, "A512Softplus", seed)
        for panel in PANELS
        for seed in SEEDS
    }
    requested = None if roles is None else set(roles)
    if requested is not None and requested != importable_roles:
        raise RoutedLadderCampaignPlanError(
            "completed-trajectory import roles are invalid"
        )
    selected_tasks: list[RoutedLadderTrainingTask] = []
    for task in source_plan.trajectories:
        role = task.panel, task.condition, task.seed
        if role not in importable_roles:
            continue
        if requested is not None and role not in requested:
            continue
        completion = inspect_training_completion(source_plan, task)
        if completion is None:
            if requested is not None:
                raise RoutedLadderCampaignPlanError(
                    f"requested completed trajectory is incomplete: {role}"
                )
            continue
        selected_tasks.append(task)
    selected_roles = {
        (task.panel, task.condition, task.seed) for task in selected_tasks
    }
    if requested is not None and selected_roles != requested:
        raise RoutedLadderCampaignPlanError(
            "requested completed trajectories are absent from source plan"
        )
    if not selected_tasks:
        raise RoutedLadderCampaignPlanError(
            "source campaign has no completed trajectories to import"
        )

    cells: list[dict[str, Any]] = []
    for task in sorted(
        selected_tasks,
        key=lambda row: (row.panel, row.condition, row.seed),
    ):
        result, result_payload = _read_self_hashed_json(
            task.completion_path_controller,
            hash_field="result_sha256",
        )
        run_root_remote = str(
            PurePosixPath(task.control_root_remote) / "runs" / task.run_id
        )
        run_contract_path = (
            task.completion_path_controller.parent / "run-contract.json"
        )
        run_contract, run_contract_payload = _read_self_hashed_json(
            run_contract_path,
            hash_field="run_contract_sha256",
        )
        config_path = (
            source_plan.working_directory
            / "configs"
            / _CONFIG_NAMES[task.panel]
        )
        try:
            config_payload = config_path.read_bytes()
        except OSError as error:
            raise RoutedLadderCampaignPlanError(
                f"cannot read source config for {task.task_id}"
            ) from error
        config_file_sha = hashlib.sha256(config_payload).hexdigest()
        if (
            run_contract.get("project_commit")
            != source_plan.project_commit
            or run_contract.get("config_file_sha256") != config_file_sha
            or run_contract.get("phase") != PHASE_NAME
            or run_contract.get("scientific_model") != task.panel
            or run_contract.get("condition") != task.condition
            or run_contract.get("seed") != task.seed
            or run_contract.get("checkpoint_payload_root")
            != task.checkpoint_root_gcs
        ):
            raise RoutedLadderCampaignPlanError(
                f"completed source run contract changed: {task.task_id}"
            )
        resource = source_plan.resource(task.eligible_resource_ids[0])
        controller_checkpoint_root = (
            task.completion_path_controller.parent / "checkpoints"
        )
        remote_checkpoint_root = str(
            PurePosixPath(run_root_remote) / "checkpoints"
        )
        common = {
            "panel": task.panel,
            "source_condition": task.condition,
            "target_condition": task.condition,
            "seed": task.seed,
            "training_source_revision": source_plan.project_commit,
            "config_remote": task.config_remote,
            "config_sha256": task.config_sha256,
            "config_file_sha256": config_file_sha,
            "run_root_remote": run_root_remote,
            "checkpoint_payload_root_gcs": task.checkpoint_root_gcs,
            "run_contract": {
                "controller_path": str(run_contract_path),
                "remote_path": str(
                    PurePosixPath(run_root_remote) / "run-contract.json"
                ),
                "sha256": hashlib.sha256(
                    run_contract_payload
                ).hexdigest(),
            },
            "run_contract_sha256": run_contract["run_contract_sha256"],
            "training_result": {
                "controller_path": str(task.completion_path_controller),
                "remote_path": str(
                    PurePosixPath(run_root_remote) / "training-result.json"
                ),
                "sha256": hashlib.sha256(result_payload).hexdigest(),
            },
            "training_result_sha256": result["result_sha256"],
            "resource_id": resource.resource_id,
            "accelerator_type": resource.accelerator_type,
        }
        for milestone in MILESTONES:
            pointer_path = milestone_pointer_path(
                controller_checkpoint_root,
                milestone,
            )
            pointer, pointer_payload = _read_self_hashed_json(
                pointer_path,
                hash_field="pointer_sha256",
            )
            cells.append(
                {
                    **common,
                    "checkpoint_tokens": milestone,
                    "checkpoint_pointer": {
                        "controller_path": str(pointer_path),
                        "remote_path": str(
                            milestone_pointer_path(
                                remote_checkpoint_root,
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
    body = {
        "schema_version": (
            COMPLETED_TRAJECTORY_REFERENCE_MANIFEST_SCHEMA_VERSION
        ),
        "source_protocol_id": str(source_plan.source["protocol_id"]),
        "target_protocol_id": V3_PROTOCOL_ID,
        "source_campaign_id": source_plan.campaign_id,
        "source_plan_sha256": source_plan.plan_sha256,
        "cells": cells,
    }
    manifest = {**body, "manifest_sha256": canonical_sha256(body)}
    _parsed, references = parse_completed_trajectory_reference_manifest(
        manifest
    )
    for cell in references:
        validate_completed_checkpoint_lineage(cell)
    return manifest


def write_completed_trajectory_reference_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    tuple[CompletedTrajectoryCheckpointReference, ...],
]:
    parsed, cells = parse_completed_trajectory_reference_manifest(manifest)
    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    if selected.exists():
        raise RoutedLadderCampaignPlanError(
            "refusing to overwrite completed-trajectory reference manifest: "
            f"{selected}"
        )
    selected.write_text(
        yaml.safe_dump(dict(manifest), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return parsed, cells


def inspect_training_completion(
    plan: RoutedLadderCampaignPlan,
    task: RoutedLadderTrainingTask,
) -> CompletionFingerprint | None:
    path = task.completion_path_controller
    if not path.exists():
        return None
    value, payload = _read_self_hashed_json(path, hash_field="result_sha256")
    identity = value.get("run_identity")
    run_plan = identity.get("plan") if isinstance(identity, Mapping) else None
    if (
        value.get("schema_version")
        != "gemma4-contribution-sae-training-result-v1"
        or value.get("status") not in {"completed", "already_complete"}
        or value.get("tokens") != TRAIN_TOKENS
        or value.get("source_revision") != plan.project_commit
        or not isinstance(run_plan, Mapping)
        or run_plan.get("phase_name") != PHASE_NAME
        or run_plan.get("condition") != task.condition
        or run_plan.get("seed") != task.seed
        or run_plan.get("total_tokens") != TRAIN_TOKENS
    ):
        raise RoutedLadderCampaignStateError(
            f"training completion role changed: {task.task_id}"
        )
    return CompletionFingerprint(
        path=str(path),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        scientific_sha256=str(value["result_sha256"]),
    )


def inspect_evaluation_completion(
    plan: RoutedLadderCampaignPlan,
    task: RoutedLadderEvaluationTask,
) -> CompletionFingerprint | None:
    result_path = task.completion_path_controller
    if not result_path.exists():
        if task.result_root_controller.exists() and any(
            task.result_root_controller.iterdir()
        ):
            raise RoutedLadderCampaignStateError(
                f"evaluation completion is partial: {task.task_id}"
            )
        return None
    value, payload = _read_self_hashed_json(
        result_path, hash_field="result_sha256"
    )
    relative = value.get("score_data_object")
    if (
        not isinstance(relative, str)
        or PurePosixPath(relative).is_absolute()
        or ".." in PurePosixPath(relative).parts
        or not relative.startswith("objects/")
        or not relative.endswith(".safetensors")
    ):
        raise RoutedLadderCampaignStateError(
            f"evaluation score object is invalid: {task.task_id}"
        )
    scores_path = task.result_root_controller / relative
    if scores_path.is_symlink() or not scores_path.is_file():
        raise RoutedLadderCampaignStateError(
            f"evaluation score object is absent: {task.task_id}"
        )
    try:
        score_payload = scores_path.read_bytes()
    except OSError as error:
        raise RoutedLadderCampaignStateError(
            f"cannot read evaluation scores: {task.task_id}"
        ) from error
    score_file_sha = hashlib.sha256(score_payload).hexdigest()
    if (
        value.get("schema_version") != EVALUATION_RESULT_SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "cell_evaluated"
        or value.get("panel") != task.panel
        or value.get("condition") != task.condition
        or value.get("seed") != task.seed
        or value.get("checkpoint_tokens") != task.milestone_tokens
        or value.get("source_revision") != plan.project_commit
        or value.get("training_source_revision")
        != task.training_source_revision
        or value.get("config_sha256") != task.config_sha256
        or value.get("score_schema_version") != EVALUATION_SCORE_SCHEMA_VERSION
        or value.get("score_data_sha256") != score_file_sha
        or value.get("score_data_bytes") != len(score_payload)
        or not isinstance(value.get("score_artifact_sha256"), str)
        or not value["score_artifact_sha256"]
    ):
        raise RoutedLadderCampaignStateError(
            f"evaluation completion role changed: {task.task_id}"
        )
    if task.condition == "H":
        parent = {
            row.identity: row for row in plan.h_references
        }[(task.panel, task.seed, task.milestone_tokens)]
        if (
            value.get("h_config_sha256") != parent.h_config_sha256
            or value.get("training_result_sha256")
            != parent.training_result_sha256
            or value.get("checkpoint_sha256")
            != parent.checkpoint_pointer_sha256
        ):
            raise RoutedLadderCampaignStateError(
                f"H evaluation lineage changed: {task.task_id}"
            )
    elif (
        task.panel == "sae"
        and task.condition == "A256ReLU"
        and task.seed == 0
    ):
        parent = {
            row.checkpoint_tokens: row for row in plan.a256_references
        }[task.milestone_tokens]
        if (
            value.get("training_result_sha256")
            != parent.training_result_sha256
            or value.get("training_contract_sha256")
            != parent.run_contract_sha256
            or value.get("checkpoint_sha256")
            != parent.checkpoint_pointer_sha256
        ):
            raise RoutedLadderCampaignStateError(
                f"historical A256 evaluation lineage changed: {task.task_id}"
            )
    parent_pins = _evaluation_parent_pins(plan, task)
    if (
        value.get("training_result_sha256")
        != parent_pins["parent_training_result_sha256"]
        or value.get("checkpoint_sha256")
        != parent_pins["parent_checkpoint_pointer_sha256"]
        or (
            parent_pins["parent_run_contract_sha256"] is not None
            and value.get("training_contract_sha256")
            != parent_pins["parent_run_contract_sha256"]
        )
    ):
        raise RoutedLadderCampaignStateError(
            f"evaluation parent pins changed: {task.task_id}"
        )
    return CompletionFingerprint(
        path=str(result_path),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        scientific_sha256=str(value["result_sha256"]),
        companion_path=str(scores_path),
        companion_size_bytes=len(score_payload),
        companion_sha256=score_file_sha,
        companion_scientific_sha256=str(value["score_artifact_sha256"]),
    )


def build_train_command(
    plan: RoutedLadderCampaignPlan,
    task: RoutedLadderTrainingTask,
    attempt_id: str,
) -> list[str]:
    return [
        str(plan.python_executable),
        str(plan.working_directory / "scripts/tpu/train_contribution.py"),
        "--action",
        "train",
        "--scientific-model",
        task.panel,
        "--config",
        task.config_remote,
        "--controller-config",
        str(plan.working_directory / "configs" / _CONFIG_NAMES[task.panel]),
        "--phase",
        PHASE_NAME,
        "--condition",
        task.condition,
        "--seed",
        str(task.seed),
        "--run-id",
        task.run_id,
        "--attempt-id",
        attempt_id,
        "--maximum-runtime-seconds",
        str(CONTINUOUS_RUNTIME_SECONDS),
    ]


def build_evaluation_command(
    plan: RoutedLadderCampaignPlan,
    task: RoutedLadderEvaluationTask,
) -> list[str]:
    command = [
        str(plan.python_executable),
        str(
            plan.working_directory
            / "scripts/tpu/evaluate_experiment5_routed_ladder.py"
        ),
        "--panel",
        task.panel,
        "--config",
        task.config_remote,
        "--condition",
        task.condition,
        "--seed",
        str(task.seed),
        "--checkpoint-tokens",
        str(task.milestone_tokens),
        "--source-revision",
        plan.project_commit,
        "--training-source-revision",
        task.training_source_revision,
        "--run-root",
        task.run_root_remote,
        "--result-root",
        task.result_root_remote,
    ]
    if task.h_config_remote is not None:
        command.extend(("--h-config", task.h_config_remote))
    if task.a256_config_remote is not None:
        command.extend(("--a256-config", task.a256_config_remote))
        if (
            task.reference_run_contract_sha256 is None
            or task.reference_run_contract_file_sha256 is None
        ):
            raise RoutedLadderCampaignStateError(
                f"{task.task_id} lost its historical run-contract pins"
            )
        command.extend(
            (
                "--reference-run-contract-sha256",
                task.reference_run_contract_sha256,
                "--reference-run-contract-file-sha256",
                task.reference_run_contract_file_sha256,
            )
        )
    return command


def _evaluation_bundles(
    plan: RoutedLadderCampaignPlan,
) -> tuple[RoutedLadderEvaluationBundleTask, ...]:
    """Derive the exact six resident execution units from the 72 plan rows."""

    by_role = {
        (task.panel, task.condition, task.seed, task.milestone_tokens): task
        for task in plan.evaluations
    }
    bundles: list[RoutedLadderEvaluationBundleTask] = []
    for panel in PANELS:
        for seed in SEEDS:
            cells = tuple(
                by_role[(panel, condition, seed, milestone)]
                for condition in BUNDLE_CONDITIONS
                for milestone in MILESTONES
            )
            owners = {cell.eligible_resource_ids for cell in cells}
            configs = {cell.config_remote for cell in cells}
            controls = {cell.control_root_remote for cell in cells}
            train_roots = {cell.train_source_root_remote for cell in cells}
            eval_roots = {cell.eval_source_root_remote for cell in cells}
            train_prefixes = {
                cell.train_source_object_prefix_gcs for cell in cells
            }
            eval_prefixes = {
                cell.eval_source_object_prefix_gcs for cell in cells
            }
            environments = {cell.environment for cell in cells}
            if any(
                len(values) != 1
                for values in (
                    owners,
                    configs,
                    controls,
                    train_roots,
                    eval_roots,
                    train_prefixes,
                    eval_prefixes,
                    environments,
                )
            ):
                raise RoutedLadderCampaignPlanError(
                    f"{panel}/seed{seed} cannot form one resident bundle"
                )
            owner_id = cells[0].eligible_resource_ids[0]
            plane = plan.resource_pool.plane(owner_id)
            relative_root = (
                PurePosixPath(EXTENSION_NAMESPACE)
                / plan.project_commit[:12]
                / "evaluation"
                / panel
                / f"seed{seed}"
                / "_bundle"
            )
            remote_result = str(
                PurePosixPath(plane.remote_artifact_root) / relative_root
            )
            controller_result = (
                plane.controller_artifact_root / Path(*relative_root.parts)
            )
            # The operator replaces the per-cell roots from its manifest.  Its
            # common environment is nevertheless validated through a current
            # A512ReLU cell, so use that cell's object root as the probe value.
            probe = next(
                cell
                for cell in cells
                if cell.condition == "A512ReLU"
                and cell.milestone_tokens == MILESTONES[0]
            )
            bundles.append(
                RoutedLadderEvaluationBundleTask(
                    task_id=f"eval-bundle-{panel}-seed{seed}",
                    panel=panel,
                    seed=seed,
                    eligible_resource_ids=cells[0].eligible_resource_ids,
                    config_remote=cells[0].config_remote,
                    control_root_remote=cells[0].control_root_remote,
                    checkpoint_root_gcs=probe.checkpoint_root_gcs,
                    train_source_root_remote=cells[0].train_source_root_remote,
                    eval_source_root_remote=cells[0].eval_source_root_remote,
                    train_source_object_prefix_gcs=(
                        cells[0].train_source_object_prefix_gcs
                    ),
                    eval_source_object_prefix_gcs=(
                        cells[0].eval_source_object_prefix_gcs
                    ),
                    result_root_remote=remote_result,
                    result_root_controller=controller_result,
                    completion_path_controller=(
                        controller_result / "bundle-result.json"
                    ),
                    cells=cells,
                    environment=cells[0].environment,
                )
            )
    if (
        len(bundles) != EVALUATION_BUNDLE_COUNT
        or len({bundle.task_id for bundle in bundles})
        != EVALUATION_BUNDLE_COUNT
    ):
        raise RoutedLadderCampaignPlanError(
            "evaluation plan did not derive the exact six resident bundles"
        )
    return tuple(bundles)


def _operator_bundle_manifest(
    plan: RoutedLadderCampaignPlan,
    bundle: RoutedLadderEvaluationBundleTask,
) -> dict[str, Any]:
    """Build the exact self-hashed manifest consumed by the TPU operator."""

    cells: list[dict[str, Any]] = []
    for task in bundle.cells:
        parent_pins = _evaluation_parent_pins(plan, task)
        historical_a256 = (
            task.panel == "sae"
            and task.seed == 0
            and task.condition == "A256ReLU"
        )
        reference_kind = (
            "a256" if historical_a256 else "h" if task.condition == "H" else "none"
        )
        reference_config = (
            task.a256_config_remote
            if reference_kind == "a256"
            else task.h_config_remote
            if reference_kind == "h"
            else None
        )
        cells.append(
            {
                "condition": task.condition,
                "checkpoint_tokens": task.milestone_tokens,
                "training_source_revision": task.training_source_revision,
                "run_root_remote": task.run_root_remote,
                "checkpoint_payload_root_gcs": task.checkpoint_root_gcs,
                "result_root_remote": task.result_root_remote,
                "result_root_controller": str(task.result_root_controller),
                "reference_kind": reference_kind,
                "reference_config_remote": reference_config,
                "reference_run_contract_sha256": (
                    task.reference_run_contract_sha256
                    if reference_kind == "a256"
                    else None
                ),
                "reference_run_contract_file_sha256": (
                    task.reference_run_contract_file_sha256
                    if reference_kind == "a256"
                    else None
                ),
                **parent_pins,
            }
        )
    body = {
        "schema_version": OPERATOR_BUNDLE_MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "panel": bundle.panel,
        "seed": bundle.seed,
        "source_revision": plan.project_commit,
        "config_remote": bundle.config_remote,
        "bundle_result_root_remote": bundle.result_root_remote,
        "bundle_result_root_controller": str(bundle.result_root_controller),
        "cells": cells,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _evaluation_parent_pins(
    plan: RoutedLadderCampaignPlan,
    task: RoutedLadderEvaluationTask,
) -> dict[str, str | None]:
    """Pin every parent artifact after training and before bundle launch."""

    if task.condition == "H":
        parent = {
            row.identity: row for row in plan.h_references
        }[(task.panel, task.seed, task.milestone_tokens)]
        return {
            "parent_training_result_sha256": parent.training_result_sha256,
            "parent_training_result_file_sha256": (
                parent.training_result.sha256
            ),
            "parent_run_contract_sha256": None,
            "parent_run_contract_file_sha256": None,
            "parent_checkpoint_pointer_sha256": (
                parent.checkpoint_pointer_sha256
            ),
            "parent_checkpoint_pointer_file_sha256": (
                parent.checkpoint_pointer.sha256
            ),
        }
    historical_a256 = (
        task.panel == "sae"
        and task.condition == "A256ReLU"
        and task.seed == 0
    )
    if historical_a256:
        parent = {
            row.checkpoint_tokens: row for row in plan.a256_references
        }[task.milestone_tokens]
        return {
            "parent_training_result_sha256": parent.training_result_sha256,
            "parent_training_result_file_sha256": (
                parent.training_result.sha256
            ),
            "parent_run_contract_sha256": parent.run_contract_sha256,
            "parent_run_contract_file_sha256": parent.run_contract.sha256,
            "parent_checkpoint_pointer_sha256": (
                parent.checkpoint_pointer_sha256
            ),
            "parent_checkpoint_pointer_file_sha256": (
                parent.checkpoint_pointer.sha256
            ),
        }
    training_by_role = {
        (row.panel, row.condition, row.seed): row
        for row in plan.trajectories
    }
    parent = training_by_role[(task.panel, task.condition, task.seed)]
    completion = inspect_training_completion(plan, parent)
    if completion is None:
        raise RoutedLadderCampaignStateError(
            f"{task.task_id} parent training is incomplete"
        )
    result, result_payload = _read_self_hashed_json(
        parent.completion_path_controller,
        hash_field="result_sha256",
    )
    run_contract_path = (
        parent.completion_path_controller.parent / "run-contract.json"
    )
    run_contract, run_contract_payload = _read_self_hashed_json(
        run_contract_path,
        hash_field="run_contract_sha256",
    )
    config_path = (
        plan.working_directory / "configs" / _CONFIG_NAMES[parent.panel]
    )
    try:
        config_file_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    except OSError as error:
        raise RoutedLadderCampaignStateError(
            f"{task.task_id} cannot read its training config"
        ) from error
    if (
        run_contract.get("project_commit") != plan.project_commit
        or run_contract.get("config_file_sha256") != config_file_sha
        or run_contract.get("phase") != PHASE_NAME
        or run_contract.get("scientific_model") != parent.panel
        or run_contract.get("condition") != parent.condition
        or run_contract.get("seed") != parent.seed
        or run_contract.get("checkpoint_payload_root")
        != parent.checkpoint_root_gcs
    ):
        raise RoutedLadderCampaignStateError(
            f"{task.task_id} parent run contract changed"
        )
    pointer_path = milestone_pointer_path(
        parent.completion_path_controller.parent / "checkpoints",
        task.milestone_tokens,
    )
    pointer, pointer_payload = _read_self_hashed_json(
        pointer_path,
        hash_field="pointer_sha256",
    )
    identity = result.get("run_identity")
    if not isinstance(identity, Mapping):
        raise RoutedLadderCampaignStateError(
            f"{task.task_id} parent run identity is absent"
        )
    try:
        checked = validate_pointer(
            pointer,
            expected_kind="latest",
            payload_root=parent.checkpoint_root_gcs,
            run_identity=identity,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise RoutedLadderCampaignStateError(
            f"{task.task_id} parent milestone pointer is invalid"
        ) from error
    expected_step, expected_tokens = _expected_milestone_progress(
        result,
        task.milestone_tokens,
    )
    if (
        checked["step"] != expected_step
        or checked["tokens_seen"] != expected_tokens
    ):
        raise RoutedLadderCampaignStateError(
            f"{task.task_id} parent milestone progress changed"
        )
    return {
        "parent_training_result_sha256": str(result["result_sha256"]),
        "parent_training_result_file_sha256": hashlib.sha256(
            result_payload
        ).hexdigest(),
        "parent_run_contract_sha256": str(
            run_contract["run_contract_sha256"]
        ),
        "parent_run_contract_file_sha256": hashlib.sha256(
            run_contract_payload
        ).hexdigest(),
        "parent_checkpoint_pointer_sha256": str(
            pointer["pointer_sha256"]
        ),
        "parent_checkpoint_pointer_file_sha256": hashlib.sha256(
            pointer_payload
        ).hexdigest(),
    }


def _bundle_manifest_path(
    plan: RoutedLadderCampaignPlan,
    bundle: RoutedLadderEvaluationBundleTask,
) -> Path:
    return (
        plan.state_root
        / "evaluation-bundles"
        / bundle.task_id
        / "operator-manifest.json"
    )


def _seal_operator_bundle_manifest(
    plan: RoutedLadderCampaignPlan,
    bundle: RoutedLadderEvaluationBundleTask,
) -> tuple[Path, dict[str, Any]]:
    manifest = _operator_bundle_manifest(plan, bundle)
    payload = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    path = _bundle_manifest_path(plan, bundle)
    _seal_exact_bytes(path, payload, name=f"{bundle.task_id} operator manifest")
    return path, manifest


def build_evaluation_bundle_command(
    plan: RoutedLadderCampaignPlan,
    bundle: RoutedLadderEvaluationBundleTask,
    attempt_id: str,
) -> list[str]:
    _require_evaluation_bundle_parents(plan, bundle)
    manifest_path, _manifest = _seal_operator_bundle_manifest(plan, bundle)
    return [
        str(plan.python_executable),
        str(
            plan.working_directory
            / "scripts/tpu/evaluate_experiment5_routed_ladder.py"
        ),
        "--panel",
        bundle.panel,
        "--config",
        bundle.config_remote,
        "--seed",
        str(bundle.seed),
        "--source-revision",
        plan.project_commit,
        "--bundle-manifest",
        str(manifest_path),
        "--attempt-id",
        attempt_id,
    ]


def _require_evaluation_bundle_parents(
    plan: RoutedLadderCampaignPlan,
    bundle: RoutedLadderEvaluationBundleTask,
) -> None:
    """Fail closed unless every new trajectory parent is fully complete."""

    training = {
        (task.panel, task.condition, task.seed): task
        for task in plan.trajectories
    }
    for condition in CONDITIONS:
        historical = (
            bundle.panel == "sae"
            and bundle.seed == 0
            and condition == "A256ReLU"
        )
        if historical:
            # Preflight validates all three immutable A256 milestone parents.
            if len(plan.a256_references) != A256_REFERENCE_COUNT:
                raise RoutedLadderCampaignStateError(
                    f"{bundle.task_id} lost its historical A256 parents"
                )
            continue
        parent = training[(bundle.panel, condition, bundle.seed)]
        if inspect_training_completion(plan, parent) is None:
            raise RoutedLadderCampaignStateError(
                f"{bundle.task_id} is not runnable before "
                f"{parent.task_id} completes"
            )


def _bundle_cell_completion(
    plan: RoutedLadderCampaignPlan,
    task: RoutedLadderEvaluationTask,
) -> dict[str, Any]:
    completion = inspect_evaluation_completion(plan, task)
    if completion is None:
        raise RoutedLadderCampaignStateError(
            f"resident bundle omitted evaluation cell: {task.task_id}"
        )
    value, payload = _read_self_hashed_json(
        task.completion_path_controller,
        hash_field="result_sha256",
    )
    return {
        "condition": task.condition,
        "checkpoint_tokens": task.milestone_tokens,
        "training_source_revision": task.training_source_revision,
        **_evaluation_parent_pins(plan, task),
        "result_root": task.result_root_remote,
        "result_sha256": str(value["result_sha256"]),
        "result_file_sha256": hashlib.sha256(payload).hexdigest(),
        "score_artifact_sha256": str(value["score_artifact_sha256"]),
        "score_data_sha256": str(value["score_data_sha256"]),
        "score_data_bytes": int(value["score_data_bytes"]),
    }


def inspect_evaluation_bundle_completion(
    plan: RoutedLadderCampaignPlan,
    bundle: RoutedLadderEvaluationBundleTask,
) -> CompletionFingerprint | None:
    path = bundle.completion_path_controller
    if not path.exists():
        if bundle.result_root_controller.exists() and any(
            bundle.result_root_controller.iterdir()
        ):
            raise RoutedLadderCampaignStateError(
                f"resident bundle completion is partial: {bundle.task_id}"
            )
        return None
    value, payload = _read_self_hashed_json(path, hash_field="result_sha256")
    _manifest_path, manifest = _seal_operator_bundle_manifest(plan, bundle)
    expected_cells = [
        _bundle_cell_completion(plan, task) for task in bundle.cells
    ]
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "panel",
        "seed",
        "source_revision",
        "operator_manifest_sha256",
        "bundle_spec_sha256",
        "timing",
        "cell_results",
        "result_sha256",
    }
    timing = value.get("timing")
    resource = plan.resource(bundle.eligible_resource_ids[0])
    expected_chips = _CHIP_COUNT_BY_ACCELERATOR.get(
        resource.accelerator_type
    )
    if (
        set(value) != expected_keys
        or value.get("schema_version") != BUNDLE_RESULT_SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "bundle_evaluated"
        or value.get("panel") != bundle.panel
        or value.get("seed") != bundle.seed
        or value.get("source_revision") != plan.project_commit
        or value.get("operator_manifest_sha256")
        != manifest["manifest_sha256"]
        or re.fullmatch(r"[0-9a-f]{64}", str(value.get("bundle_spec_sha256")))
        is None
        or not isinstance(timing, Mapping)
        or set(timing)
        != {
            "accelerator_type",
            "chip_count",
            "successful_attempt_wall_seconds",
            "successful_attempt_chip_seconds",
            "timing_scope",
        }
        or timing.get("accelerator_type") != resource.accelerator_type
        or timing.get("chip_count") != expected_chips
        or timing.get("timing_scope") != _EVALUATION_TIMING_SCOPE
        or value.get("cell_results") != expected_cells
    ):
        raise RoutedLadderCampaignStateError(
            f"resident bundle completion changed: {bundle.task_id}"
        )
    try:
        wall_seconds = float(timing["successful_attempt_wall_seconds"])
        chip_seconds = float(timing["successful_attempt_chip_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise RoutedLadderCampaignStateError(
            f"resident bundle timing is invalid: {bundle.task_id}"
        ) from error
    if (
        not math.isfinite(wall_seconds)
        or wall_seconds <= 0
        or not math.isfinite(chip_seconds)
        or chip_seconds <= 0
        or expected_chips is None
        or not math.isclose(
            chip_seconds,
            wall_seconds * expected_chips,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
    ):
        raise RoutedLadderCampaignStateError(
            f"resident bundle timing is invalid: {bundle.task_id}"
        )
    return CompletionFingerprint(
        path=str(path),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        scientific_sha256=str(value["result_sha256"]),
    )


def _attempt_environment(
    plan: RoutedLadderCampaignPlan,
    resource: CampaignResource,
    task: RoutedLadderTask,
    operator_log_root: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    audited = plan.environment_dict()
    audited.update(resource.environment_dict())
    audited.update(task.environment_dict())
    audited.update(
        {
            "GCP_PROJECT": resource.project,
            "TPU_NAME": resource.tpu_name,
            "TPU_ZONE": resource.zone,
            "PROJECT_COMMIT": plan.project_commit,
            "TPU_PROJECT_ROOT": plan.remote_project_root,
            "REMOTE_UV": plan.remote_uv,
            "TPU_ACCELERATOR_TYPE": resource.accelerator_type,
            "TPU_USE_INTERNAL_IP": "1" if resource.internal_ip else "0",
            "EXPERIMENT5_PANEL": task.panel,
            "CONTRIBUTION_SAE_CONTROL_ROOT": task.control_root_remote,
            "CONTRIBUTION_SAE_CHECKPOINT_ROOT": task.checkpoint_root_gcs,
            "CONTRIBUTION_FRESH_CONTROLLER_MAPPING": (
                resource.controller_mapping.remote_path
            ),
            "CONTRIBUTION_LOCAL_CACHE_INVENTORY": (
                resource.cache_inventory.remote_path
            ),
        }
    )
    if resource.mirror_seal is not None:
        audited["CONTRIBUTION_REGIONAL_MIRROR_SEAL"] = resource.mirror_seal.remote_path
    if resource.ssh_user is not None:
        audited["TPU_SSH_USER"] = resource.ssh_user
    if task.panel == "transcoder":
        if isinstance(task, RoutedLadderTrainingTask):
            train_root = task.source_overlay_control_root_remote
            train_prefix = task.source_overlay_object_prefix_gcs
            eval_root = None
            eval_prefix = None
        else:
            train_root = task.train_source_root_remote
            train_prefix = task.train_source_object_prefix_gcs
            eval_root = task.eval_source_root_remote
            eval_prefix = task.eval_source_object_prefix_gcs
        if train_root is None or train_prefix is None:
            raise RoutedLadderCampaignStateError(
                f"{task.task_id} lost its transcoder source overlay"
            )
        audited["CONTRIBUTION_SOURCE_OVERLAY_ROOT"] = train_root
        audited["CONTRIBUTION_SOURCE_OVERLAY_PREFIX"] = train_prefix
        if eval_root is not None and eval_prefix is not None:
            audited["CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT"] = eval_root
            audited["CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX"] = eval_prefix
    if isinstance(task, RoutedLadderTrainingTask):
        audited["CONTRIBUTION_SAE_LOCAL_LOG_ROOT"] = str(operator_log_root)
    else:
        audited.update(
            {
                "EXPERIMENT5_ROUTED_ACTIVATION_LADDER_RESULT_ROOT_CONTROLLER": str(
                    task.result_root_controller
                ),
            }
        )
        log_name = (
            "CONTRIBUTION_EVALUATION_LOG_ROOT"
            if task.panel == "sae"
            else "TRANSCODER_EVALUATION_LOG_ROOT"
        )
        audited[log_name] = str(operator_log_root)
    executable = os.environ.copy()
    executable.update(audited)
    return executable, audited


def _verify_checkout(plan: RoutedLadderCampaignPlan) -> None:
    if plan.working_directory.is_symlink() or not plan.working_directory.is_dir():
        raise RoutedLadderCampaignStateError("working_directory is not real")
    if (
        plan.python_executable.is_symlink()
        or not plan.python_executable.is_file()
        or not os.access(plan.python_executable, os.X_OK)
    ):
        raise RoutedLadderCampaignStateError("python_executable is not executable")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=plan.working_directory,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=plan.working_directory,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RoutedLadderCampaignStateError(
            "cannot verify controller checkout"
        ) from error
    if head != plan.project_commit or status:
        raise RoutedLadderCampaignStateError(
            "controller checkout must be clean at project_commit"
        )
    evaluator = (
        plan.working_directory / "scripts/tpu/evaluate_experiment5_routed_ladder.py"
    )
    if evaluator.is_symlink() or not evaluator.is_file():
        raise RoutedLadderCampaignStateError(
            "activation-ladder evaluator script is absent"
        )
    expected_config_hashes = {
        panel: {
            task.config_sha256
            for task in (*plan.trajectories, *plan.evaluations)
            if task.panel == panel
        }
        for panel in PANELS
    }
    for panel in PANELS:
        path = plan.working_directory / "configs" / _CONFIG_NAMES[panel]
        try:
            config = load_routed_ladder_v2_config(
                path,
                expected_panel=panel,
            )
        except RoutedLadderV2ConfigError as error:
            raise RoutedLadderCampaignStateError(
                f"{panel} activation-ladder config is invalid: {error}"
            ) from error
        if expected_config_hashes[panel] != {config.config_hash}:
            raise RoutedLadderCampaignStateError(
                f"{panel} task config hash differs from checkout"
            )
        h_path = (
            plan.working_directory
            / "configs"
            / f"gemma4_hybrid_{panel}_v1.yaml"
        )
        try:
            h_config = load_experiment5_config(h_path, expected_panel=panel)
        except Experiment5ConfigError as error:
            raise RoutedLadderCampaignStateError(
                f"{panel} parent H config is invalid: {error}"
            ) from error
        if h_config.config_hash != H_CONFIG_SHA256[panel]:
            raise RoutedLadderCampaignStateError(
                f"{panel} parent H config identity changed"
            )


def _preflight(
    plan: RoutedLadderCampaignPlan,
    *,
    verify_checkout: bool,
) -> None:
    if verify_checkout:
        _verify_checkout(plan)
    active_ids = {
        resource_id
        for task in (*plan.trajectories, *plan.evaluations)
        for resource_id in task.eligible_resource_ids
    }
    for resource in plan.resources:
        if resource.resource_id not in active_ids:
            continue
        _verify_file(
            resource.controller_mapping.controller_path,
            resource.controller_mapping.sha256,
            f"{resource.resource_id} controller mapping",
        )
        _verify_file(
            resource.cache_inventory.controller_path,
            resource.cache_inventory.sha256,
            f"{resource.resource_id} cache inventory",
        )
        if resource.mirror_seal is not None:
            _verify_file(
                resource.mirror_seal.controller_path,
                resource.mirror_seal.sha256,
                f"{resource.resource_id} mirror seal",
            )
    for cell in plan.h_references:
        for label, reference in (
            ("training result", cell.training_result),
            ("checkpoint pointer", cell.checkpoint_pointer),
        ):
            _verify_file(
                reference.controller_path,
                reference.sha256,
                f"H {cell.panel}/seed{cell.seed}/{cell.checkpoint_tokens} "
                f"{label}",
            )
        _validate_h_checkpoint_lineage(cell)
    for cell in plan.a256_references:
        for label, reference in (
            ("config", cell.a256_config),
            ("run contract", cell.run_contract),
            ("training result", cell.training_result),
            ("checkpoint pointer", cell.checkpoint_pointer),
        ):
            _verify_file(
                reference.controller_path,
                reference.sha256,
                f"A256 SAE/seed0/{cell.checkpoint_tokens} {label}",
            )
        try:
            config = load_learning_horizon_config(
                cell.a256_config.controller_path,
                expected_scientific_model="sae",
            )
        except LearningHorizonConfigError as error:
            raise RoutedLadderCampaignStateError(
                f"historical A256 config is invalid: {error}"
            ) from error
        if config.config_hash != A256_SAE_CONFIG_SHA256:
            raise RoutedLadderCampaignStateError(
                "historical A256 config identity changed"
            )
        _validate_a256_checkpoint_lineage(
            cell,
            plan.a256_historical_cost,
        )


def _training_cost(
    *,
    result: Mapping[str, Any],
    resource_id: str,
    accelerator_type: str,
) -> RoutedLadderTrainingCost:
    """Bind one training-result timing ledger to its immutable owner."""

    expected_chips = _CHIP_COUNT_BY_ACCELERATOR.get(accelerator_type)
    if expected_chips is None:
        raise RoutedLadderCampaignStateError(
            f"unsupported training accelerator {accelerator_type!r}"
        )
    exclusions = result.get("chip_seconds_excludes")
    try:
        return RoutedLadderTrainingCost(
            resource_id=resource_id,
            accelerator_type=accelerator_type,
            chip_count=expected_chips,
            marginal_wall_seconds=float(result["marginal_wall_seconds"]),
            chip_seconds=float(result["chip_seconds"]),
            chip_seconds_scope=str(result["chip_seconds_scope"]),
            chip_seconds_excludes=(
                tuple(exclusions) if isinstance(exclusions, list) else ()
            ),
        )
    except (KeyError, TypeError, ValueError, Experiment5RoutedLadderError) as error:
        raise RoutedLadderCampaignStateError(
            f"training result has invalid cost accounting for {resource_id}"
        ) from error


def _trajectory_accounting(
    plan: RoutedLadderCampaignPlan,
) -> tuple[RoutedLadderTrajectoryAccounting, ...]:
    """Construct the exact 24-row historical/new training inventory."""

    new_by_role = {
        (task.panel, task.condition, task.seed): task
        for task in plan.trajectories
    }
    a256_by_milestone = {
        row.checkpoint_tokens: row for row in plan.a256_references
    }
    h_by_role = {
        (row.panel, row.seed, row.checkpoint_tokens): row
        for row in plan.h_references
    }
    rows: list[RoutedLadderTrajectoryAccounting] = []
    for panel in PANELS:
        for condition in EVALUATION_CONDITIONS:
            for seed in SEEDS:
                historical_a256 = (
                    panel == "sae"
                    and condition == "A256ReLU"
                    and seed == 0
                )
                if historical_a256:
                    parent = a256_by_milestone[TRAIN_TOKENS]
                    historical = plan.a256_historical_cost
                    cost = RoutedLadderTrainingCost(
                        resource_id=historical.resource_id,
                        accelerator_type=historical.accelerator_type,
                        chip_count=historical.chip_count,
                        marginal_wall_seconds=(
                            historical.marginal_wall_seconds
                        ),
                        chip_seconds=historical.chip_seconds,
                        chip_seconds_scope=historical.chip_seconds_scope,
                        chip_seconds_excludes=(
                            historical.chip_seconds_excludes
                        ),
                    )
                    row = RoutedLadderTrajectoryAccounting(
                        panel=panel,
                        condition=condition,
                        seed=seed,
                        origin=(
                            "historical_experiment4_seed0_reference"
                        ),
                        training_source_revision=(
                            parent.training_source_revision
                        ),
                        reference_manifest_kind=(
                            "a256_reference_manifest"
                        ),
                        reference_manifest_sha256=str(
                            plan.a256_reference_manifest["manifest_sha256"]
                        ),
                        parent_training_result_sha256=(
                            parent.training_result_sha256
                        ),
                        parent_training_result_file_sha256=(
                            parent.training_result.sha256
                        ),
                        cost_status=(
                            "measured_historical_training_result"
                        ),
                        training_cost=cost,
                    )
                elif condition == "H":
                    parent = h_by_role[(panel, seed, TRAIN_TOKENS)]
                    row = RoutedLadderTrajectoryAccounting(
                        panel=panel,
                        condition=condition,
                        seed=seed,
                        origin="historical_experiment5_reference",
                        training_source_revision=(
                            parent.training_source_revision
                        ),
                        reference_manifest_kind="h_reference_manifest",
                        reference_manifest_sha256=str(
                            plan.h_reference_manifest["manifest_sha256"]
                        ),
                        parent_training_result_sha256=(
                            parent.training_result_sha256
                        ),
                        parent_training_result_file_sha256=(
                            parent.training_result.sha256
                        ),
                        cost_status=(
                            "not_supplied_by_h_reference_manifest"
                        ),
                        training_cost=None,
                    )
                else:
                    task = new_by_role[(panel, condition, seed)]
                    completion = inspect_training_completion(plan, task)
                    if completion is None:
                        raise RoutedLadderCampaignStateError(
                            f"training accounting parent is incomplete: "
                            f"{task.task_id}"
                        )
                    result, payload = _read_self_hashed_json(
                        task.completion_path_controller,
                        hash_field="result_sha256",
                    )
                    resource = plan.resource(
                        task.eligible_resource_ids[0]
                    )
                    row = RoutedLadderTrajectoryAccounting(
                        panel=panel,
                        condition=condition,
                        seed=seed,
                        origin="protocol_v2_new_training",
                        training_source_revision=plan.project_commit,
                        reference_manifest_kind=None,
                        reference_manifest_sha256=None,
                        parent_training_result_sha256=str(
                            result["result_sha256"]
                        ),
                        parent_training_result_file_sha256=hashlib.sha256(
                            payload
                        ).hexdigest(),
                        cost_status="measured_v2_training_result",
                        training_cost=_training_cost(
                            result=result,
                            resource_id=resource.resource_id,
                            accelerator_type=resource.accelerator_type,
                        ),
                    )
                rows.append(row)
    return tuple(rows)


def _evaluation_bundle_accounting(
    plan: RoutedLadderCampaignPlan,
) -> tuple[RoutedLadderEvaluationBundleAccounting, ...]:
    """Construct the exact six measured resident-evaluation rows."""

    rows: list[RoutedLadderEvaluationBundleAccounting] = []
    for bundle in _evaluation_bundles(plan):
        completion = inspect_evaluation_bundle_completion(plan, bundle)
        if completion is None:
            raise RoutedLadderCampaignStateError(
                f"evaluation bundle accounting is incomplete: {bundle.task_id}"
            )
        result, payload = _read_self_hashed_json(
            bundle.completion_path_controller,
            hash_field="result_sha256",
        )
        timing = result.get("timing")
        resource = plan.resource(bundle.eligible_resource_ids[0])
        expected_chips = _CHIP_COUNT_BY_ACCELERATOR.get(
            resource.accelerator_type
        )
        if (
            not isinstance(timing, Mapping)
            or timing.get("accelerator_type") != resource.accelerator_type
            or timing.get("chip_count") != expected_chips
            or timing.get("timing_scope") != _EVALUATION_TIMING_SCOPE
        ):
            raise RoutedLadderCampaignStateError(
                f"evaluation bundle hardware timing changed: {bundle.task_id}"
            )
        try:
            row = RoutedLadderEvaluationBundleAccounting(
                panel=bundle.panel,
                seed=bundle.seed,
                source_revision=plan.project_commit,
                operator_manifest_sha256=str(
                    result["operator_manifest_sha256"]
                ),
                bundle_spec_sha256=str(result["bundle_spec_sha256"]),
                bundle_result_sha256=str(result["result_sha256"]),
                bundle_result_file_sha256=hashlib.sha256(
                    payload
                ).hexdigest(),
                accelerator_type=resource.accelerator_type,
                chip_count=int(timing["chip_count"]),
                successful_attempt_wall_seconds=float(
                    timing["successful_attempt_wall_seconds"]
                ),
                successful_attempt_chip_seconds=float(
                    timing["successful_attempt_chip_seconds"]
                ),
                timing_scope=str(timing["timing_scope"]),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            Experiment5RoutedLadderError,
        ) as error:
            raise RoutedLadderCampaignStateError(
                f"evaluation bundle accounting is invalid: {bundle.task_id}"
            ) from error
        rows.append(row)
    return tuple(rows)


def _wall_cash_accounting(
    plan: RoutedLadderCampaignPlan,
    cost_projection: Mapping[str, Any],
) -> RoutedLadderWallCashAccounting:
    """Bind report wall/cash fields to the sealed campaign projection."""

    path = plan.state_root / "cost-projection.json"
    observed = _read_record(path, record_type="cost_projection")
    if observed != dict(cost_projection):
        raise RoutedLadderCampaignStateError(
            "report cost projection differs from its sealed record"
        )
    try:
        payload = path.read_bytes()
        return RoutedLadderWallCashAccounting(
            campaign_wall_seconds=float(
                observed["campaign_elapsed_seconds"]
            ),
            wall_scope=_CAMPAIGN_WALL_SCOPE,
            cash_usd=float(observed["projected_committed_usd"]),
            cash_currency=str(observed["currency"]),
            cash_basis=(
                "sealed projected committed cash: existing spend plus "
                "the durable campaign infrastructure model and immutable "
                "attempt reservations"
            ),
            accounting_source=ImmutableArtifactReference(
                kind="cost_projection",
                schema_version=str(observed["schema_version"]),
                uri=str(path),
                sha256=hashlib.sha256(payload).hexdigest(),
            ),
        )
    except (KeyError, OSError, ValueError, Experiment5RoutedLadderError) as error:
        raise RoutedLadderCampaignStateError(
            "sealed campaign wall/cash accounting is invalid"
        ) from error


def _render_routed_ladder_report(report: Mapping[str, Any]) -> str:
    """Render a compact deterministic companion to the validated JSON report."""

    lines = [
        "# Experiment 5 routed-activation ladder",
        "",
        f"Protocol: `{report['protocol_id']}`",
        "",
        (
            "The only interval-bearing contrast is paired "
            "A512Softplus−A512ReLU. "
            "H comparisons are matched descriptive context, not causal effects."
        ),
        "",
        "## Across-seed verdict",
        "",
        "| panel | tokens | endpoint | mean A512Softplus−A512ReLU EV | "
        "hierarchical 95% CI | positive seeds |",
        "|---|---:|---|---:|---:|---:|",
    ]
    across_seed = report.get(
        "across_seed_a512_softplus_minus_a512_relu"
    )
    if not isinstance(across_seed, list):
        raise RoutedLadderCampaignStateError(
            "validated activation-ladder report lacks across-seed summaries"
        )
    for row in across_seed:
        identity = row["identity"]
        for endpoint in ("component", "routed", "oracle_shared"):
            effect = row[
                "paired_mean_a512_softplus_minus_a512_relu"
            ][endpoint]
            lines.append(
                "| {panel} | {tokens} | {endpoint} | {estimate:.8f} | "
                "[{lower:.8f}, {upper:.8f}] | {positive}/3 |".format(
                    panel=identity["panel"],
                    tokens=identity["checkpoint_tokens"],
                    endpoint=endpoint,
                    estimate=effect["estimate_mean_ev_difference"],
                    lower=effect["ci95_lower"],
                    upper=effect["ci95_upper"],
                    positive=effect["positive_seed_count"],
                )
            )
    lines.extend(
        [
            "",
            "## Per-seed detail",
            "",
        "| panel | seed | tokens | endpoint | A512ReLU EV | "
        "A512Softplus EV | Softplus−ReLU [95% CI] | H EV |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    cohorts = report.get("cohorts")
    if not isinstance(cohorts, list):
        raise RoutedLadderCampaignStateError(
            "validated activation-ladder report lacks cohorts"
        )
    for cohort in cohorts:
        identity = cohort["identity"]
        conditions = cohort["conditions"]
        paired = cohort[
            "paired_a512_softplus_minus_a512_relu"
        ]
        for endpoint in ("component", "routed", "oracle_shared"):
            effect = paired[endpoint]
            lines.append(
                "| {panel} | {seed} | {tokens} | {endpoint} | "
                "{relu:.8f} | {softplus:.8f} | {delta:.8f} "
                "[{lower:.8f}, {upper:.8f}] | {h:.8f} |".format(
                    panel=identity["panel"],
                    seed=identity["seed"],
                    tokens=identity["checkpoint_tokens"],
                    endpoint=endpoint,
                    relu=conditions["A512ReLU"]["endpoints"][endpoint]["ev"],
                    softplus=conditions["A512Softplus"]["endpoints"][
                        endpoint
                    ]["ev"],
                    delta=effect["estimate_ev_difference"],
                    lower=effect["ci95_lower"],
                    upper=effect["ci95_upper"],
                    h=conditions["H"]["endpoints"][endpoint]["ev"],
                )
            )
    lines.extend(("", "## Claim boundary", ""))
    for caveat in report["interpretation"]["caveats"]:
        lines.append(f"- {caveat}")
    lines.extend(("", f"Report SHA-256: `{report['report_sha256']}`", ""))
    return "\n".join(lines)


class RoutedLadderCampaign(Experiment5Campaign):
    """Run the isolated extension using the hardened Experiment-5 scheduler."""

    plan: RoutedLadderCampaignPlan

    def __init__(
        self,
        plan: RoutedLadderCampaignPlan,
        **kwargs: Any,
    ) -> None:
        super().__init__(plan, **kwargs)  # type: ignore[arg-type]

    def _maximum_campaign_wall_seconds(self) -> float:
        """Retain a finite reporting horizon without a cash-derived stop."""

        return float(CONTINUOUS_RUNTIME_SECONDS)

    def _check_campaign_watchdog(self) -> None:
        """Protocol v2 records cost but never blocks grant-funded TPU work."""

    def _reserve_attempt_cost(
        self,
        task: RoutedLadderTask,
        *,
        kind: str,
        attempt_number: int,
        resource: CampaignResource,
    ) -> dict[str, Any]:
        """Append the inherited cash estimate as measurement-only evidence."""

        path = (
            self.plan.state_root
            / "budget"
            / "attempt-reservations"
            / kind
            / task.task_id
            / f"attempt-{attempt_number:06d}.json"
        )
        with self._budget_lock:
            if path.exists():
                observed = _read_record(
                    path,
                    record_type="cost_reservation",
                )
                if (
                    observed.get("task_id") != task.task_id
                    or observed.get("kind") != kind
                    or observed.get("attempt_number") != attempt_number
                    or observed.get("eligible_resource_ids")
                    != list(task.eligible_resource_ids)
                    or observed.get("enforcement")
                    != "measurement_only_grant_funded_v2"
                ):
                    raise RoutedLadderCampaignStateError(
                        f"cost reservation changed for {task.task_id}"
                    )
                return observed
            amount = self.plan.cost_policy.attempt_reserve(kind)
            before = float(
                self.cost_projection()["projected_committed_usd"]
            )
            reservation = _record(
                "cost_reservation",
                {
                    "campaign_id": self.plan.campaign_id,
                    "plan_sha256": self.plan.plan_sha256,
                    "task_id": task.task_id,
                    "kind": kind,
                    "attempt_number": attempt_number,
                    "resource_id": resource.resource_id,
                    "eligible_resource_ids": list(
                        task.eligible_resource_ids
                    ),
                    "amount_usd": amount,
                    "projected_committed_after_usd": before + amount,
                    "enforcement": "measurement_only_grant_funded_v2",
                    "reserved_at": _timestamp(),
                },
            )
            _write_create_only(path, reservation)
            return reservation

    def _baseline_attempt_reserve_usd(self) -> float:
        policy = self.plan.cost_policy
        return (
            len(self.plan.trajectories)
            * policy.training_attempt_reserve_usd
            + EVALUATION_BUNDLE_COUNT
            * policy.evaluation_attempt_reserve_usd
        )

    def _preflight(self) -> None:
        _preflight(self.plan, verify_checkout=self.verify_checkout)

    def _inspector(
        self, task: RoutedLadderTask
    ) -> CompletionFingerprint | None:
        if isinstance(task, RoutedLadderTrainingTask):
            return inspect_training_completion(self.plan, task)
        if isinstance(task, RoutedLadderEvaluationBundleTask):
            return inspect_evaluation_bundle_completion(self.plan, task)
        return inspect_evaluation_completion(self.plan, task)

    def _attempt_environment(
        self,
        resource: CampaignResource,
        task: RoutedLadderTask,
        operator_log_root: Path,
    ) -> tuple[dict[str, str], dict[str, str]]:
        return _attempt_environment(
            self.plan,
            resource,
            task,
            operator_log_root,
        )

    def _attempt_command(
        self,
        task: RoutedLadderTask,
        *,
        attempt_id: str,
    ) -> list[str]:
        if isinstance(task, RoutedLadderTrainingTask):
            return build_train_command(self.plan, task, attempt_id)
        if isinstance(task, RoutedLadderEvaluationBundleTask):
            return build_evaluation_bundle_command(
                self.plan,
                task,
                attempt_id,
            )
        return build_evaluation_command(self.plan, task)

    def _attempt_id(self, attempt_number: int) -> str:
        parent = super()._attempt_id(attempt_number)
        return parent.replace("e5-", "e5rl-", 1)

    def run(self, *, stop_after_training: bool = False) -> dict[str, Any]:
        """Train 17 trajectories, execute six bundles, then seal 72 cells."""

        self._preflight()
        self._acquire_lock()
        try:
            self._bind_plan()
            self._bind_campaign_clock()
            self._start_watchdog()
            training = self._run_phase(
                self.plan.trajectories,
                kind="training",
                max_parallel=self.plan.max_parallel_train,
            )
            barrier = self._seal_phase(
                path=self.plan.state_root / "training-complete.json",
                record_type="training_phase_completion",
                status="training_completed",
                field="training",
                completions=training,
            )
            if stop_after_training:
                return barrier

            bundles = _evaluation_bundles(self.plan)
            bundle_completions = self._run_phase(
                bundles,
                kind="evaluation",
                max_parallel=self.plan.max_parallel_eval,
            )
            if set(bundle_completions) != {
                bundle.task_id for bundle in bundles
            }:
                raise RoutedLadderCampaignStateError(
                    "resident evaluation bundle matrix is incomplete"
                )

            evaluations: dict[str, CompletionFingerprint] = {}
            for task in self.plan.evaluations:
                completion = inspect_evaluation_completion(self.plan, task)
                if completion is None:
                    raise RoutedLadderCampaignStateError(
                        f"resident evaluation omitted {task.task_id}"
                    )
                self._completion_witness(
                    task,
                    kind="evaluation",
                    fingerprint=completion,
                )
                evaluations[task.task_id] = completion

            evaluation_barrier = self._seal_phase(
                path=self.plan.state_root / "evaluation-complete.json",
                record_type="evaluation_phase_completion",
                status="evaluation_completed",
                field="evaluations",
                completions=evaluations,
            )
            cost = self._seal_cost_projection(
                completed_at=evaluation_barrier["completed_at"],
            )
            report_identity = self._seal_scientific_report(
                evaluation_barrier=evaluation_barrier,
                cost_projection=cost,
            )
            final = _record(
                "campaign_completion",
                {
                    "campaign_id": self.plan.campaign_id,
                    "plan_sha256": self.plan.plan_sha256,
                    "status": "completed",
                    "training": {
                        key: training[key].to_dict()
                        for key in sorted(training)
                    },
                    "evaluations": {
                        key: evaluations[key].to_dict()
                        for key in sorted(evaluations)
                    },
                    "report": report_identity,
                    "cost_projection_record_sha256": cost["record_sha256"],
                    "completed_at": evaluation_barrier["completed_at"],
                },
            )
            path = self.plan.state_root / "final.json"
            if path.exists():
                observed = _read_record(
                    path,
                    record_type="campaign_completion",
                )
                for key in (
                    "campaign_id",
                    "plan_sha256",
                    "status",
                    "training",
                    "evaluations",
                    "report",
                    "cost_projection_record_sha256",
                    "completed_at",
                ):
                    if observed.get(key) != final.get(key):
                        raise RoutedLadderCampaignStateError(
                            "final campaign record changed"
                        )
                return observed
            _write_create_only(path, final)
            return final
        finally:
            self._stop_watchdog()
            self._release_lock()

    def _seal_scientific_report(
        self,
        *,
        evaluation_barrier: Mapping[str, Any],
        cost_projection: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build, validate, and seal the complete activation-ladder report."""

        rows = evaluation_barrier.get("evaluations")
        expected = {task.task_id for task in self.plan.evaluations}
        if (
            evaluation_barrier.get("campaign_id") != self.plan.campaign_id
            or evaluation_barrier.get("plan_sha256") != self.plan.plan_sha256
            or evaluation_barrier.get("status") != "evaluation_completed"
            or not isinstance(rows, Mapping)
            or set(rows) != expected
            or len(expected) != len(self.plan.evaluations)
        ):
            raise RoutedLadderCampaignStateError(
                "evaluation barrier is not the exact plan-bound ladder matrix"
            )
        score_rows: list[dict[str, Any]] = []
        score_sets: list[RoutedLadderTokenScores] = []
        h_references = []
        for task in sorted(
            self.plan.evaluations,
            key=lambda item: (
                item.panel,
                item.seed,
                item.milestone_tokens,
                item.condition,
            ),
        ):
            completion = inspect_evaluation_completion(self.plan, task)
            if completion is None or completion.to_dict() != rows.get(task.task_id):
                raise RoutedLadderCampaignStateError(
                    f"evaluation changed after phase seal: {task.task_id}"
                )
            score_rows.append(
                {
                    "panel": task.panel,
                    "condition": task.condition,
                    "seed": task.seed,
                    "checkpoint_tokens": task.milestone_tokens,
                    "evaluation_result": completion.to_dict(),
                }
            )
            try:
                scores = load_merged_scores(task.completion_path_controller)
            except RoutedLadderEvaluationError as error:
                raise RoutedLadderCampaignStateError(
                    f"cannot load report scores for {task.task_id}: {error}"
                ) from error
            if task.condition == "H":
                if not isinstance(scores, RoutedLadderHTokenScores):
                    raise RoutedLadderCampaignStateError(
                        f"H report cell has routed-only scores: {task.task_id}"
                    )
                result, _payload = _read_self_hashed_json(
                    task.completion_path_controller,
                    hash_field="result_sha256",
                )
                parent = {
                    row.identity: row for row in self.plan.h_references
                }[(task.panel, task.seed, task.milestone_tokens)]
                relative = str(result["score_data_object"])
                h_references.append(
                    h_token_scores_to_existing_reference(
                        scores,
                        training_source_revision=(
                            parent.training_source_revision
                        ),
                        h_config_sha256=parent.h_config_sha256,
                        training_result_sha256=(
                            parent.training_result_sha256
                        ),
                        training_result_file_sha256=(
                            parent.training_result.sha256
                        ),
                        checkpoint_pointer_sha256=(
                            parent.checkpoint_pointer_sha256
                        ),
                        checkpoint_pointer_file_sha256=(
                            parent.checkpoint_pointer.sha256
                        ),
                        scientific_result=ImmutableArtifactReference(
                            kind="scientific_result",
                            schema_version=EVALUATION_RESULT_SCHEMA_VERSION,
                            uri=str(task.completion_path_controller),
                            sha256=completion.sha256,
                        ),
                        diagnostics=(
                            ImmutableArtifactReference(
                                kind="token_scores",
                                schema_version=EVALUATION_SCORE_SCHEMA_VERSION,
                                uri=str(task.result_root_controller / relative),
                                sha256=str(result["score_data_sha256"]),
                            ),
                        ),
                    )
                )
            else:
                if not isinstance(scores, RoutedLadderTokenScores):
                    raise RoutedLadderCampaignStateError(
                        f"A report cell has H scores: {task.task_id}"
                    )
                score_sets.append(scores)
        trajectory_accounting = _trajectory_accounting(self.plan)
        bundle_accounting = _evaluation_bundle_accounting(self.plan)
        wall_cash = _wall_cash_accounting(
            self.plan,
            cost_projection,
        )
        body = {
            "schema_version": REPORT_INPUT_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "campaign_id": self.plan.campaign_id,
            "plan_sha256": self.plan.plan_sha256,
            "source_revision": self.plan.project_commit,
            "evaluation_cells": score_rows,
            "h_reference_manifest": self.plan.h_reference_manifest,
            "a256_reference_manifest_sha256": (
                self.plan.a256_reference_manifest["manifest_sha256"]
            ),
            "h_reference_manifest_sha256": (
                self.plan.h_reference_manifest["manifest_sha256"]
            ),
            "trajectory_accounting": [
                row.report_row() for row in trajectory_accounting
            ],
            "evaluation_bundle_accounting": [
                row.report_row() for row in bundle_accounting
            ],
            "wall_cash_accounting": wall_cash.report_row(),
            "interpretation": (
                "A512Softplus-minus-A512ReLU is the only paired bootstrap "
                "contrast; "
                "H gaps are descriptive and noncausal"
            ),
        }
        report_inputs = {**body, "report_inputs_sha256": canonical_sha256(body)}
        root = self.plan.state_root / "report"
        path = root / "report-inputs.json"
        payload = (
            json.dumps(
                report_inputs,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        file_sha = _seal_exact_bytes(
            path,
            payload,
            name="activation-ladder report inputs",
        )
        try:
            report = build_experiment5_routed_ladder_report(
                score_sets=score_sets,
                h_references=h_references,
                report_inputs=ImmutableArtifactReference(
                    kind="report_inputs",
                    schema_version=REPORT_INPUT_SCHEMA_VERSION,
                    uri=str(path),
                    sha256=file_sha,
                ),
                trajectory_accounting=trajectory_accounting,
                evaluation_bundle_accounting=bundle_accounting,
                a256_reference_manifest_sha256=str(
                    self.plan.a256_reference_manifest["manifest_sha256"]
                ),
                h_reference_manifest_sha256=str(
                    self.plan.h_reference_manifest["manifest_sha256"]
                ),
                wall_cash_accounting=wall_cash,
            )
            validate_experiment5_routed_ladder_report(report)
            markdown = _render_routed_ladder_report(report)
        except Experiment5RoutedLadderError as error:
            raise RoutedLadderCampaignStateError(
                f"activation-ladder evidence cannot be reported: {error}"
            ) from error
        report_path = (
            root / "experiment5-routed-activation-ladder-result.json"
        )
        report_file_sha = _seal_exact_bytes(
            report_path,
            (
                json.dumps(
                    report,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            ),
            name="activation-ladder JSON report",
        )
        markdown_path = (
            root / "EXPERIMENT_5_ROUTED_ACTIVATION_LADDER.md"
        )
        markdown_file_sha = _seal_exact_bytes(
            markdown_path,
            markdown.encode("utf-8"),
            name="activation-ladder Markdown report",
        )
        return {
            "status": "report_sealed",
            "report_inputs_path": str(path),
            "report_inputs_sha256": report_inputs["report_inputs_sha256"],
            "report_inputs_file_sha256": file_sha,
            "report_path": str(report_path),
            "report_sha256": report["report_sha256"],
            "report_file_sha256": report_file_sha,
            "markdown_path": str(markdown_path),
            "markdown_file_sha256": markdown_file_sha,
            "evaluation_completion_record_sha256": evaluation_barrier[
                "record_sha256"
            ],
        }


def routed_ladder_campaign_status(
    plan: RoutedLadderCampaignPlan,
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task in (*plan.trajectories, *plan.evaluations):
        kind = (
            "training"
            if isinstance(task, RoutedLadderTrainingTask)
            else "evaluation"
        )
        completion = (
            inspect_training_completion(plan, task)
            if isinstance(task, RoutedLadderTrainingTask)
            else inspect_evaluation_completion(plan, task)
        )
        root = plan.state_root / kind / task.task_id
        attempts = (
            len([row for row in (root / "attempts").iterdir() if row.is_dir()])
            if (root / "attempts").is_dir()
            else 0
        )
        state = "completed" if completion is not None else "pending"
        if completion is None and (root / "failure.json").exists():
            failure = _read_record(root / "failure.json", record_type="task_failure")
            state = f"failed:{failure.get('classification')}"
        elif completion is None and attempts:
            latest = max((root / "attempts").iterdir())
            state = (
                "retryable"
                if (latest / "result.json").exists()
                else "ambiguous_inflight"
            )
        tasks[task.task_id] = {
            "kind": kind,
            "state": state,
            "attempts": attempts,
            "eligible_resources": list(task.eligible_resource_ids),
            "completion": completion.to_dict() if completion else None,
        }
    controller = RoutedLadderCampaign(plan, verify_checkout=False)
    clock_path = plan.state_root / "campaign-clock.json"
    clock = (
        _read_record(clock_path, record_type="campaign_clock")
        if clock_path.exists()
        else None
    )
    if clock is not None:
        controller._activate_campaign_clock(clock)
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.plan_sha256,
        "tasks": tasks,
        "h_reference_cells": len(plan.h_references),
        "campaign_clock": clock,
        "cost_projection": controller.cost_projection(),
        "watchdog_projection": (
            controller._watchdog_projection() if clock is not None else None
        ),
        "training_barrier": (
            _read_record(
                plan.state_root / "training-complete.json",
                record_type="training_phase_completion",
            )
            if (plan.state_root / "training-complete.json").exists()
            else None
        ),
        "evaluation_barrier": (
            _read_record(
                plan.state_root / "evaluation-complete.json",
                record_type="evaluation_phase_completion",
            )
            if (plan.state_root / "evaluation-complete.json").exists()
            else None
        ),
        "final": (
            _read_record(plan.state_root / "final.json", record_type="campaign_completion")
            if (plan.state_root / "final.json").exists()
            else None
        ),
    }


def build_routed_ladder_campaign_plan(
    source: Mapping[str, Any],
    *,
    project_commit: str,
    h_reference_manifest: Mapping[str, Any],
    a256_reference_manifest: Mapping[str, Any],
    already_spent_usd: float,
    resource_pool: Mapping[str, Any],
    campaign_id: str | None = None,
) -> dict[str, Any]:
    """Build one N-way plan without discovering or mutating cloud resources."""

    if not isinstance(source, Mapping):
        raise RoutedLadderCampaignPlanError("source must be a mapping")
    try:
        selected_commit = _git_sha(project_commit, "project_commit")
        admitted = parse_routed_ladder_resource_pool(
            resource_pool,
            project_commit=selected_commit,
        )
        working_directory = _controller_path(
            source.get("working_directory"),
            "source.working_directory",
        )
        python_executable = _controller_path(
            source.get("python_executable"),
            "source.python_executable",
        )
        remote_project_root = _remote_path(
            source.get("remote_project_root"),
            "source.remote_project_root",
        )
        remote_uv = _remote_path(
            source.get("remote_uv"),
            "source.remote_uv",
        )
    except (Experiment5CampaignPlanError, RoutedLadderResourcePoolError) as error:
        raise RoutedLadderCampaignPlanError(str(error)) from error
    source_environment = source.get("environment")
    if not isinstance(source_environment, Mapping) or not all(
        isinstance(key, str) for key in source_environment
    ):
        raise RoutedLadderCampaignPlanError(
            "source.environment must be a string-keyed mapping"
        )
    environment = {
        key: copy.deepcopy(value)
        for key, value in source_environment.items()
        if key in _PRESERVED_GLOBAL_ENVIRONMENT
    }
    try:
        environment = dict(_task_environment(environment, "environment"))
        source_overlays = {
            "train_root": _remote_path(
                source_environment.get("CONTRIBUTION_SOURCE_OVERLAY_ROOT"),
                "CONTRIBUTION_SOURCE_OVERLAY_ROOT",
            ),
            "eval_root": _remote_path(
                source_environment.get("CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT"),
                "CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT",
            ),
            "train_prefix": _gcs_prefix(
                source_environment.get("CONTRIBUTION_SOURCE_OVERLAY_PREFIX"),
                "CONTRIBUTION_SOURCE_OVERLAY_PREFIX",
            ),
            "eval_prefix": _gcs_prefix(
                source_environment.get("CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX"),
                "CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX",
            ),
        }
    except Experiment5CampaignPlanError as error:
        raise RoutedLadderCampaignPlanError(str(error)) from error
    global_bulk = environment.get("BULK_GCS_ROOT")
    for resource in admitted.resources:
        if "BULK_GCS_ROOT" in resource.environment_dict():
            continue
        if global_bulk is None or admitted.plane(
            resource.resource_id
        ).bulk_gcs_root != global_bulk.removesuffix("/horizon-10m"):
            raise RoutedLadderCampaignPlanError(
                f"{resource.resource_id} object plane differs from the "
                "source default"
            )
    h_manifest, h_cells = parse_h_reference_manifest(h_reference_manifest)
    h_by_role = {cell.identity: cell for cell in h_cells}
    (
        a256_manifest,
        a256_cells,
        a256_cost,
    ) = parse_a256_reference_manifest(a256_reference_manifest)
    try:
        historical_plan = parse_direct_campaign_plan(source)
    except DirectCampaignPlanError as error:
        raise RoutedLadderCampaignPlanError(
            f"A256 source is not the strict Experiment-4 direct plan: {error}"
        ) from error
    if (
        a256_manifest["source_plan_sha256"]
        != historical_plan.plan_sha256
    ):
        raise RoutedLadderCampaignPlanError(
            "A256 reference manifest names a different Experiment-4 plan"
        )
    historical_tasks = [
        task
        for task in historical_plan.trajectories
        if task.scientific_model == "sae"
        and task.condition == "A"
        and task.seed == 0
    ]
    if len(historical_tasks) != 1:
        raise RoutedLadderCampaignPlanError(
            "Experiment-4 plan lacks the sole historical SAE A trajectory"
        )
    historical_task = historical_tasks[0]
    historical_resource = historical_plan.resource(
        historical_task.resource_id
    )
    historical_revision = historical_plan.environment_dict().get(
        "TRAINING_PROJECT_COMMIT"
    )
    expected_run_root = str(
        PurePosixPath(historical_task.control_root_remote)
        / "runs"
        / historical_task.run_id
    )
    expected_training_remote = str(
        PurePosixPath(expected_run_root) / "training-result.json"
    )
    expected_contract_controller = (
        historical_task.completion_path_controller.parent
        / "run-contract.json"
    )
    expected_contract_remote = str(
        PurePosixPath(expected_run_root) / "run-contract.json"
    )
    if (
        historical_revision is None
        or a256_cost.resource_id != historical_resource.resource_id
        or a256_cost.accelerator_type
        != historical_resource.accelerator_type
        or a256_cost.chip_count != 16
    ):
        raise RoutedLadderCampaignPlanError(
            "A256 historical cost/resource identity differs from its "
            "Experiment-4 source plan"
        )
    for cell in a256_cells:
        expected_pointer_controller = milestone_pointer_path(
            historical_task.completion_path_controller.parent
            / "checkpoints",
            cell.checkpoint_tokens,
        )
        expected_pointer_remote = str(
            milestone_pointer_path(
                str(
                    PurePosixPath(expected_run_root)
                    / "checkpoints"
                ),
                cell.checkpoint_tokens,
            )
        )
        if (
            cell.training_source_revision != historical_revision
            or cell.a256_config.remote_path
            != historical_task.config_remote
            or cell.run_root_remote != expected_run_root
            or cell.checkpoint_payload_root_gcs
            != historical_task.checkpoint_root_gcs
            or cell.training_result.controller_path
            != historical_task.completion_path_controller
            or cell.training_result.remote_path
            != expected_training_remote
            or cell.run_contract.controller_path
            != expected_contract_controller
            or cell.run_contract.remote_path
            != expected_contract_remote
            or cell.checkpoint_pointer.controller_path
            != expected_pointer_controller
            or cell.checkpoint_pointer.remote_path
            != expected_pointer_remote
        ):
            raise RoutedLadderCampaignPlanError(
                "A256 historical artifact paths differ from the exact "
                "Experiment-4 source trajectory"
            )
    a256_by_milestone = {
        cell.checkpoint_tokens: cell for cell in a256_cells
    }
    short = selected_commit[:12]
    relative_root = PurePosixPath(EXTENSION_NAMESPACE) / short
    configs: dict[str, tuple[str, str, Any]] = {}
    for panel in PANELS:
        local_path = working_directory / "configs" / _CONFIG_NAMES[panel]
        try:
            config = load_routed_ladder_v2_config(
                local_path,
                expected_panel=panel,
            )
        except RoutedLadderV2ConfigError as error:
            raise RoutedLadderCampaignPlanError(
                f"cannot bind {panel} activation-ladder config: {error}"
            ) from error
        for condition in CONDITIONS:
            if (
                config.training_seeds_for_condition(condition)
                != training_seeds_for_condition(panel, condition)
            ):
                raise RoutedLadderCampaignPlanError(
                    f"{panel}/{condition} training provenance differs from "
                    "the campaign matrix"
                )
        configs[panel] = (
            str(
                PurePosixPath(remote_project_root)
                / "configs"
                / _CONFIG_NAMES[panel]
            ),
            config.config_hash,
            config,
        )
    trajectories: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for panel in PANELS:
        config_remote, config_sha, config = configs[panel]
        for seed in SEEDS:
            cohort = admitted.cohort(panel, seed)
            panel_offset = PANELS.index(panel) * len(CONDITIONS)
            bundle_owner_id = cohort[
                (PANELS.index(panel) * len(SEEDS) + seed) % len(cohort)
            ]
            bundle_plane = admitted.plane(bundle_owner_id)
            evaluation_control_root = str(
                PurePosixPath(bundle_plane.remote_artifact_root)
                / relative_root
                / "runs"
                / panel
                / f"seed{seed}"
            )
            for condition_index, condition in enumerate(CONDITIONS):
                slug = _CONDITION_SLUG[condition]
                owner_id = cohort[
                    (panel_offset + condition_index) % len(cohort)
                ]
                owner_plane = admitted.plane(owner_id)
                control_root = str(
                    PurePosixPath(owner_plane.remote_artifact_root)
                    / relative_root
                    / "runs"
                    / panel
                    / f"seed{seed}"
                )
                controller_group = (
                    owner_plane.controller_artifact_root
                    / Path(*relative_root.parts)
                    / "runs"
                    / panel
                    / f"seed{seed}"
                )
                checkpoint_root = (
                    f"{owner_plane.bulk_gcs_root}/{relative_root}/"
                    f"checkpoints/{panel}/seed{seed}"
                )
                run_id = (
                    f"experiment5-activation-ladder-{panel}-{slug}-seed{seed}"
                )
                train_new = seed in config.training_seeds_for_condition(
                    condition
                )
                if train_new:
                    trajectories.append(
                        {
                            "id": f"train-{panel}-{slug}-seed{seed}",
                            "panel": panel,
                            "condition": condition,
                            "seed": seed,
                            "eligible_resource_ids": [owner_id],
                            "run_id": run_id,
                            "config_remote": config_remote,
                            "config_sha256": config_sha,
                            "phase": PHASE_NAME,
                            "control_root_remote": control_root,
                            "checkpoint_root_gcs": checkpoint_root,
                            "source_overlay_control_root_remote": (
                                None
                                if panel == "sae"
                                else source_overlays["train_root"]
                            ),
                            "source_overlay_object_prefix_gcs": (
                                None
                                if panel == "sae"
                                else source_overlays["train_prefix"]
                            ),
                            "completion_path_controller": str(
                                controller_group
                                / "runs"
                                / run_id
                                / "training-result.json"
                            ),
                            "environment": {},
                        }
                    )
                elif not (
                    panel == "sae"
                    and condition == "A256ReLU"
                    and seed == 0
                ):
                    raise RoutedLadderCampaignPlanError(
                        f"{panel}/{condition}/seed{seed} has no authorized "
                        "training or historical reference"
                    )
                run_root = str(PurePosixPath(control_root) / "runs" / run_id)
                for milestone in MILESTONES:
                    historical = (
                        a256_by_milestone[milestone]
                        if not train_new
                        else None
                    )
                    remote_result = str(
                        PurePosixPath(bundle_plane.remote_artifact_root)
                        / relative_root
                        / "evaluation"
                        / panel
                        / f"seed{seed}"
                        / str(milestone)
                        / slug
                    )
                    controller_result = (
                        bundle_plane.controller_artifact_root
                        / Path(*relative_root.parts)
                        / "evaluation"
                        / panel
                        / f"seed{seed}"
                        / str(milestone)
                        / slug
                    )
                    evaluations.append(
                        {
                            "id": (
                                f"eval-{panel}-{slug}-seed{seed}-"
                                f"{milestone}"
                            ),
                            "panel": panel,
                            "condition": condition,
                            "seed": seed,
                            "milestone_tokens": milestone,
                            "eligible_resource_ids": [bundle_owner_id],
                            "config_remote": config_remote,
                            "config_sha256": config_sha,
                            "h_config_remote": None,
                            "h_config_sha256": None,
                            "a256_config_remote": (
                                historical.a256_config.remote_path
                                if historical is not None
                                else None
                            ),
                            "a256_config_sha256": (
                                historical.a256_config_sha256
                                if historical is not None
                                else None
                            ),
                            "reference_run_contract_sha256": (
                                historical.run_contract_sha256
                                if historical is not None
                                else None
                            ),
                            "reference_run_contract_file_sha256": (
                                historical.run_contract.sha256
                                if historical is not None
                                else None
                            ),
                            "training_source_revision": (
                                historical.training_source_revision
                                if historical is not None
                                else selected_commit
                            ),
                            "phase": PHASE_NAME,
                            "control_root_remote": evaluation_control_root,
                            "checkpoint_root_gcs": (
                                historical.checkpoint_payload_root_gcs
                                if historical is not None
                                else checkpoint_root
                            ),
                            "train_source_root_remote": (
                                None
                                if panel == "sae"
                                else source_overlays["train_root"]
                            ),
                            "eval_source_root_remote": (
                                None
                                if panel == "sae"
                                else source_overlays["eval_root"]
                            ),
                            "train_source_object_prefix_gcs": (
                                None
                                if panel == "sae"
                                else source_overlays["train_prefix"]
                            ),
                            "eval_source_object_prefix_gcs": (
                                None
                                if panel == "sae"
                                else source_overlays["eval_prefix"]
                            ),
                            "run_root_remote": (
                                historical.run_root_remote
                                if historical is not None
                                else run_root
                            ),
                            "result_root_remote": remote_result,
                            "completion_path_controller": str(
                                controller_result / "evaluation-result.json"
                            ),
                            "result_root_controller": str(controller_result),
                            "environment": {},
                        }
                    )
            for milestone_index, milestone in enumerate(MILESTONES):
                parent_h = h_by_role[(panel, seed, milestone)]
                remote_result = str(
                    PurePosixPath(bundle_plane.remote_artifact_root)
                    / relative_root
                    / "evaluation"
                    / panel
                    / f"seed{seed}"
                    / str(milestone)
                    / "h"
                )
                controller_result = (
                    bundle_plane.controller_artifact_root
                    / Path(*relative_root.parts)
                    / "evaluation"
                    / panel
                    / f"seed{seed}"
                    / str(milestone)
                    / "h"
                )
                evaluations.append(
                    {
                        "id": f"eval-{panel}-h-seed{seed}-{milestone}",
                        "panel": panel,
                        "condition": "H",
                        "seed": seed,
                        "milestone_tokens": milestone,
                        "eligible_resource_ids": [bundle_owner_id],
                        "config_remote": config_remote,
                        "config_sha256": config_sha,
                        "h_config_remote": parent_h.h_config_remote,
                        "h_config_sha256": parent_h.h_config_sha256,
                        "a256_config_remote": None,
                        "a256_config_sha256": None,
                        "reference_run_contract_sha256": None,
                        "reference_run_contract_file_sha256": None,
                        "training_source_revision": (
                            parent_h.training_source_revision
                        ),
                        "phase": PHASE_NAME,
                        "control_root_remote": evaluation_control_root,
                        "checkpoint_root_gcs": (
                            parent_h.checkpoint_payload_root_gcs
                        ),
                        "train_source_root_remote": (
                            None
                            if panel == "sae"
                            else source_overlays["train_root"]
                        ),
                        "eval_source_root_remote": (
                            None
                            if panel == "sae"
                            else source_overlays["eval_root"]
                        ),
                        "train_source_object_prefix_gcs": (
                            None
                            if panel == "sae"
                            else source_overlays["train_prefix"]
                        ),
                        "eval_source_object_prefix_gcs": (
                            None
                            if panel == "sae"
                            else source_overlays["eval_prefix"]
                        ),
                        "run_root_remote": parent_h.run_root_remote,
                        "result_root_remote": remote_result,
                        "result_root_controller": str(controller_result),
                        "completion_path_controller": str(
                            controller_result / "evaluation-result.json"
                        ),
                        "environment": {},
                    }
                )
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": (
            campaign_id
            or f"gemma4-experiment5-routed-activation-ladder-{short}"
        ),
        "state_root": (
            f"/mnt/moe-sae-control/{EXTENSION_NAMESPACE}/controller/{short}"
        ),
        "working_directory": str(working_directory),
        "python_executable": str(python_executable),
        "remote_project_root": remote_project_root,
        "remote_uv": remote_uv,
        "project_commit": selected_commit,
        "max_parallel_train": len(
            {
                task["eligible_resource_ids"][0]
                for task in trajectories
            }
        ),
        "max_parallel_eval": len(
            {
                task["eligible_resource_ids"][0]
                for task in evaluations
            }
        ),
        "retry_policy": {
            "max_transient_attempts": 8,
            "initial_backoff_seconds": 30,
            "maximum_backoff_seconds": 600,
        },
        "cost_policy": {
            "hard_limit_usd": 50.0,
            "contingency_usd": 5.0,
            "already_spent_usd": already_spent_usd,
            "fixed_cost_reserve_usd": 5.0,
            "training_attempt_reserve_usd": 0.5,
            "evaluation_attempt_reserve_usd": 0.2,
        },
        "environment": environment,
        "resources": copy.deepcopy(list(admitted.resource_rows)),
        "resource_pool": copy.deepcopy(admitted.source),
        "trajectories": trajectories,
        "evaluations": evaluations,
        "h_reference_manifest": h_manifest,
        "a256_reference_manifest": a256_manifest,
    }
    parse_routed_ladder_campaign_plan(plan)
    return plan


def write_routed_ladder_campaign_plan(
    path: str | Path,
    plan: Mapping[str, Any],
) -> RoutedLadderCampaignPlan:
    parsed = parse_routed_ladder_campaign_plan(plan)
    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    if selected.exists():
        raise RoutedLadderCampaignPlanError(
            f"refusing to overwrite activation-ladder plan: {selected}"
        )
    selected.write_text(
        yaml.safe_dump(dict(plan), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return parsed


__all__ = [
    "COMPLETED_TRAJECTORY_REFERENCE_MANIFEST_SCHEMA_VERSION",
    "CompletedTrajectoryCheckpointReference",
    "CONDITIONS",
    "EVALUATION_TASK_COUNT",
    "H_REFERENCE_COUNT",
    "H_REFERENCE_MANIFEST_SCHEMA_VERSION",
    "MILESTONES",
    "PANELS",
    "PLAN_SCHEMA_VERSION",
    "RoutedLadderCampaign",
    "RoutedLadderCampaignPlan",
    "RoutedLadderCampaignPlanError",
    "RoutedLadderCampaignStateError",
    "RoutedLadderEvaluationTask",
    "RoutedLadderTrainingTask",
    "SEEDS",
    "TRAINING_TASK_COUNT",
    "build_completed_trajectory_reference_manifest",
    "build_evaluation_command",
    "build_h_reference_manifest",
    "build_routed_ladder_campaign_plan",
    "build_train_command",
    "inspect_evaluation_completion",
    "inspect_training_completion",
    "load_h_reference_manifest",
    "load_completed_trajectory_reference_manifest",
    "load_routed_ladder_campaign_plan",
    "parse_h_reference_manifest",
    "parse_completed_trajectory_reference_manifest",
    "parse_routed_ladder_campaign_plan",
    "routed_ladder_campaign_status",
    "validate_completed_checkpoint_lineage",
    "write_h_reference_manifest",
    "write_completed_trajectory_reference_manifest",
    "write_routed_ladder_campaign_plan",
]
