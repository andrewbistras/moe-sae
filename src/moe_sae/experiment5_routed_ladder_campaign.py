"""Durable campaign boundary for the Experiment-5 activation ladder.

This module is deliberately a sibling of :mod:`moe_sae.experiment5_campaign`.
It reuses that controller's append-only attempt, readiness, retry, lock, and
budget machinery without changing the frozen H/R/B/F task matrix or state
namespace.

The extension schedules exactly:

* 12 new training trajectories: A512ReLU/A512Softplus x two panels x seeds;
* 36 A512ReLU/A512Softplus evaluations at 2M/5M/10M;
* 18 read-only rescorings of immutable H checkpoints; and
* 18 hash-pinned H training/checkpoint references that are never retrained.

The controller owns no cloud lifecycle.  Its seven resource rows must be the
existing TRC inventory, and only the three already-authorized ready slices may
own work.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
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
    _remote_path,
    _resource,
    _safe_id,
    _seal_exact_bytes,
    _sha256,
    _verify_file,
    inspect_training_completion as inspect_core_training_completion,
)
from moe_sae.checkpointing import milestone_pointer_path, validate_pointer
from moe_sae.experiment5_config import (
    Experiment5ConfigError,
    load_experiment5_config,
)
from moe_sae.experiment5_plan_builder import (
    EXECUTION_PROFILE,
    EXPECTED_RESOURCE_EVIDENCE_OVERRIDES,
    EXPECTED_SURVIVING_TPUS,
    build_experiment5_plan,
)
from moe_sae.experiment5_routed_ladder import (
    PROTOCOL_ID,
    SCORE_SCHEMA_VERSION,
    Experiment5RoutedLadderError,
    ImmutableArtifactReference,
    RoutedLadderHTokenScores,
    RoutedLadderTokenScores,
    build_experiment5_routed_ladder_report,
    h_token_scores_to_existing_reference,
    validate_experiment5_routed_ladder_report,
)
from moe_sae.experiment5_routed_ladder_evaluation_worker import (
    RESULT_SCHEMA_VERSION as EVALUATION_RESULT_SCHEMA_VERSION,
    RoutedLadderEvaluationError,
    load_merged_scores,
)
from moe_sae.experiment5_routed_ladder_config import (
    H_CONFIG_SHA256,
    PHASE_NAME,
    RoutedLadderConfigError,
    load_routed_ladder_config,
)

PLAN_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-campaign-plan-v1"
)
H_REFERENCE_MANIFEST_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-h-reference-manifest-v1"
)
EVALUATION_SCORE_SCHEMA_VERSION = SCORE_SCHEMA_VERSION
REPORT_INPUT_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-report-inputs-v1"
)

PANELS = ("sae", "transcoder")
CONDITIONS = ("A512ReLU", "A512Softplus")
EVALUATION_CONDITIONS = (*CONDITIONS, "H")
SEEDS = (0, 1, 2)
MILESTONES = (2_000_000, 5_000_000, 10_000_000)
TRAIN_TOKENS = 10_000_000
RESOURCE_COUNT = 7
TRAINING_TASK_COUNT = 12
EVALUATION_TASK_COUNT = 54
H_REFERENCE_COUNT = 18
EXTENSION_NAMESPACE = (
    "experiment5/extensions/routed-activation-ladder-v1"
)
CONTINUOUS_RUNTIME_SECONDS = 43_200

_ACTIVE_TPU_NAMES = frozenset(EXPECTED_RESOURCE_EVIDENCE_OVERRIDES.values())
_CONFIG_NAMES = {
    "sae": "gemma4_experiment5_routed_ladder_sae_v1.yaml",
    "transcoder": "gemma4_experiment5_routed_ladder_transcoder_v1.yaml",
}
_RESERVED_ENVIRONMENT = frozenset(
    {
        "EXPERIMENT5_ROUTED_ACTIVATION_LADDER_RESULT_ROOT_CONTROLLER",
    }
)
_CONDITION_SLUG = {
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


RoutedLadderTask = RoutedLadderTrainingTask | RoutedLadderEvaluationTask


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
    trajectories: tuple[RoutedLadderTrainingTask, ...]
    evaluations: tuple[RoutedLadderEvaluationTask, ...]
    h_reference_manifest: Mapping[str, Any]
    h_references: tuple[HCheckpointReference, ...]
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
    if condition == "H":
        h_config_remote = _remote_path(
            h_config_remote, f"{name}.h_config_remote"
        )
        h_config_sha256 = _sha256(
            h_config_sha256, f"{name}.h_config_sha256"
        )
    elif h_config_remote is not None or h_config_sha256 is not None:
        raise RoutedLadderCampaignPlanError(
            f"{name} non-H task may not carry an H config"
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


def _cost_policy(value: Any) -> CostPolicy:
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
    projection = (
        policy.already_spent_usd
        + policy.fixed_cost_reserve_usd
        + TRAINING_TASK_COUNT * policy.training_attempt_reserve_usd
        + EVALUATION_TASK_COUNT * policy.evaluation_attempt_reserve_usd
    )
    if projection > policy.initial_launch_limit_usd + 1e-9:
        raise RoutedLadderCampaignPlanError(
            "one-attempt activation-ladder projection exceeds the protected "
            "$45 limit"
        )
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
            "trajectories",
            "evaluations",
            "h_reference_manifest",
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
        "routed-activation-ladder-v1",
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
    trajectories = tuple(
        _training_task(row, index) for index, row in enumerate(raw_training)
    )
    evaluations = tuple(
        _evaluation_task(row, index) for index, row in enumerate(raw_evaluations)
    )
    h_manifest, h_references = parse_h_reference_manifest(
        selected["h_reference_manifest"]
    )
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
            selected["max_parallel_train"], "max_parallel_train", maximum=RESOURCE_COUNT
        ),
        max_parallel_eval=_positive_int(
            selected["max_parallel_eval"], "max_parallel_eval", maximum=RESOURCE_COUNT
        ),
        retry_policy=retry,
        cost_policy=_cost_policy(selected["cost_policy"]),
        environment=environment,
        resources=resources,
        trajectories=trajectories,
        evaluations=evaluations,
        h_reference_manifest=h_manifest,
        h_references=h_references,
        plan_sha256=canonical_sha256(selected),
        source=copy.deepcopy(selected),
    )
    _validate_plan_relations(plan)
    return plan


def _validate_plan_relations(plan: RoutedLadderCampaignPlan) -> None:
    if (
        len(plan.resources) != RESOURCE_COUNT
        or len(plan.trajectories) != TRAINING_TASK_COUNT
        or len(plan.evaluations) != EVALUATION_TASK_COUNT
        or len(plan.h_references) != H_REFERENCE_COUNT
    ):
        raise RoutedLadderCampaignPlanError(
            "campaign must contain 7 resources, 12 training tasks, "
            "54 evaluations, and 18 H references"
        )
    resource_ids = [row.resource_id for row in plan.resources]
    tpu_names = [row.tpu_name for row in plan.resources]
    if (
        len(set(resource_ids)) != RESOURCE_COUNT
        or set(tpu_names) != set(EXPECTED_SURVIVING_TPUS)
    ):
        raise RoutedLadderCampaignPlanError(
            "resource rows differ from the existing seven-slice TRC inventory"
        )
    by_id = {row.resource_id: row for row in plan.resources}
    active_ids = {
        resource_id
        for task in (*plan.trajectories, *plan.evaluations)
        for resource_id in task.eligible_resource_ids
    }
    if {by_id[item].tpu_name for item in active_ids} != set(_ACTIVE_TPU_NAMES):
        raise RoutedLadderCampaignPlanError(
            "tasks must use only the existing three authorized ready slices"
        )
    if any(by_id[item].cache_profile != "paired" for item in active_ids):
        raise RoutedLadderCampaignPlanError(
            "every active activation-ladder resource requires the paired cache"
        )
    expected_training = {
        (panel, condition, seed)
        for panel in PANELS
        for condition in CONDITIONS
        for seed in SEEDS
    }
    observed_training = {
        (task.panel, task.condition, task.seed) for task in plan.trajectories
    }
    if observed_training != expected_training:
        raise RoutedLadderCampaignPlanError("training matrix is not exact")
    expected_evaluations = {
        (panel, condition, seed, milestone)
        for panel in PANELS
        for condition in EVALUATION_CONDITIONS
        for seed in SEEDS
        for milestone in MILESTONES
    }
    observed_evaluations = {
        (task.panel, task.condition, task.seed, task.milestone_tokens)
        for task in plan.evaluations
    }
    if observed_evaluations != expected_evaluations:
        raise RoutedLadderCampaignPlanError("evaluation matrix is not exact")
    if len({task.task_id for task in (*plan.trajectories, *plan.evaluations)}) != (
        TRAINING_TASK_COUNT + EVALUATION_TASK_COUNT
    ):
        raise RoutedLadderCampaignPlanError("task IDs are not unique")
    for task in (*plan.trajectories, *plan.evaluations):
        if (
            not task.eligible_resource_ids
            or any(item not in active_ids for item in task.eligible_resource_ids)
        ):
            raise RoutedLadderCampaignPlanError(
                f"{task.task_id} has an invalid eligible-resource set"
            )
        required_names = (
            {"gemma4-horizon-v5e-usc-a"}
            if task.seed == 0
            else {
                "gemma4-horizon-harvest-0728",
                "gemma4-horizon-v4-spot-trans-b",
            }
        )
        if {by_id[item].tpu_name for item in task.eligible_resource_ids} != required_names:
            raise RoutedLadderCampaignPlanError(
                f"{task.task_id} resource generation/seed assignment changed"
            )
        namespace = f"/{EXTENSION_NAMESPACE}/{plan.project_commit[:12]}/"
        values = [task.control_root_remote, str(task.completion_path_controller)]
        if isinstance(task, RoutedLadderTrainingTask):
            values.append(task.checkpoint_root_gcs)
        else:
            values.extend(
                (task.result_root_remote, str(task.result_root_controller))
            )
            if task.condition in CONDITIONS:
                values.extend((task.run_root_remote, task.checkpoint_root_gcs))
        if any(namespace not in value for value in values):
            raise RoutedLadderCampaignPlanError(
                f"{task.task_id} escapes the immutable extension namespace"
            )
    training_by_role = {
        (task.panel, task.condition, task.seed): task for task in plan.trajectories
    }
    h_by_role = {row.identity: row for row in plan.h_references}
    for task in plan.evaluations:
        if task.condition in CONDITIONS:
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
                or task.eligible_resource_ids != parent.eligible_resource_ids
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
    return command


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
            config = load_routed_ladder_config(path, expected_panel=panel)
        except RoutedLadderConfigError as error:
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

    def _baseline_attempt_reserve_usd(self) -> float:
        policy = self.plan.cost_policy
        return (
            TRAINING_TASK_COUNT * policy.training_attempt_reserve_usd
            + EVALUATION_TASK_COUNT * policy.evaluation_attempt_reserve_usd
        )

    def _preflight(self) -> None:
        _preflight(self.plan, verify_checkout=self.verify_checkout)

    def _inspector(
        self, task: RoutedLadderTask
    ) -> CompletionFingerprint | None:
        if isinstance(task, RoutedLadderTrainingTask):
            return inspect_training_completion(self.plan, task)
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
        return build_evaluation_command(self.plan, task)

    def _attempt_id(self, attempt_number: int) -> str:
        parent = super()._attempt_id(attempt_number)
        return parent.replace("e5-", "e5rl-", 1)

    def _seal_scientific_report(
        self,
        *,
        evaluation_barrier: Mapping[str, Any],
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
            or len(expected) != EVALUATION_TASK_COUNT
        ):
            raise RoutedLadderCampaignStateError(
                "evaluation barrier is not the exact 54-task ladder matrix"
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
        body = {
            "schema_version": REPORT_INPUT_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "campaign_id": self.plan.campaign_id,
            "plan_sha256": self.plan.plan_sha256,
            "source_revision": self.plan.project_commit,
            "evaluation_cells": score_rows,
            "h_reference_manifest": self.plan.h_reference_manifest,
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


def _replace_core_prefix(
    value: str,
    *,
    core_marker: str,
    replacement: str,
    name: str,
) -> str:
    count = value.count(core_marker)
    if count != 1:
        raise RoutedLadderCampaignPlanError(
            f"{name} does not contain one exact core namespace marker"
        )
    return value.replace(core_marker, replacement, 1)


def build_routed_ladder_campaign_plan(
    source: Mapping[str, Any],
    *,
    project_commit: str,
    execution_profile: str,
    h_reference_manifest: Mapping[str, Any],
    already_spent_usd: float,
    resource_evidence_overrides: Mapping[str, Any] | None,
    campaign_id: str | None = None,
) -> dict[str, Any]:
    """Build the extension from the same strict r8/evidence resource source."""

    if execution_profile != EXECUTION_PROFILE:
        raise RoutedLadderCampaignPlanError(
            f"execution_profile must be exactly {EXECUTION_PROFILE!r}"
        )
    # This creates no files or cloud resources.  It uses the frozen core
    # builder as the authority for the seven resource descriptors and the
    # ready-three data/cache planes at this exact commit.
    core = build_experiment5_plan(
        source,
        project_commit=project_commit,
        execution_profile=execution_profile,
        already_spent_usd=0.0,
        resource_evidence_overrides=resource_evidence_overrides,
        campaign_id=f"temporary-resource-source-{project_commit[:12]}",
    )
    core_plan = __import__(
        "moe_sae.experiment5_campaign",
        fromlist=["parse_experiment5_campaign_plan"],
    ).parse_experiment5_campaign_plan(core)
    h_manifest, h_cells = parse_h_reference_manifest(h_reference_manifest)
    h_by_role = {cell.identity: cell for cell in h_cells}
    short = project_commit[:12]
    core_marker = f"/{execution_profile}/{short}/"
    remote_replacement = f"/{EXTENSION_NAMESPACE}/{short}/"
    controller_core_marker = f"/experiment5/{execution_profile}/{short}/"
    controller_replacement = f"/{EXTENSION_NAMESPACE}/{short}/"
    configs: dict[str, tuple[str, str]] = {}
    for panel in PANELS:
        local_path = core_plan.working_directory / "configs" / _CONFIG_NAMES[panel]
        try:
            config = load_routed_ladder_config(local_path, expected_panel=panel)
        except RoutedLadderConfigError as error:
            raise RoutedLadderCampaignPlanError(
                f"cannot bind {panel} activation-ladder config: {error}"
            ) from error
        configs[panel] = (
            str(PurePosixPath(core_plan.remote_project_root) / "configs" / _CONFIG_NAMES[panel]),
            config.config_hash,
        )
    trajectories: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    core_training = {
        (task.panel, task.seed): task
        for task in core_plan.trajectories
        if task.condition == "H"
    }
    core_evaluations = {
        (task.panel, task.seed, task.milestone_tokens): task
        for task in core_plan.evaluations
    }
    for panel in PANELS:
        config_remote, config_sha = configs[panel]
        for seed in SEEDS:
            plane = core_training[(panel, seed)]
            control_root = _replace_core_prefix(
                plane.control_root_remote,
                core_marker=core_marker,
                replacement=remote_replacement,
                name=f"{panel}/seed{seed} control root",
            )
            checkpoint_root = _replace_core_prefix(
                plane.checkpoint_root_gcs,
                core_marker=controller_core_marker,
                replacement=controller_replacement,
                name=f"{panel}/seed{seed} checkpoint root",
            )
            completion_core_marker = controller_core_marker
            for condition in CONDITIONS:
                slug = _CONDITION_SLUG[condition]
                run_id = (
                    f"experiment5-activation-ladder-{panel}-{slug}-seed{seed}"
                )
                controller_group = str(plane.completion_path_controller.parents[2])
                controller_group = _replace_core_prefix(
                    controller_group,
                    core_marker=completion_core_marker,
                    replacement=controller_replacement,
                    name=f"{panel}/seed{seed} controller group",
                )
                trajectories.append(
                    {
                        "id": f"train-{panel}-{slug}-seed{seed}",
                        "panel": panel,
                        "condition": condition,
                        "seed": seed,
                        "eligible_resource_ids": list(plane.eligible_resource_ids),
                        "run_id": run_id,
                        "config_remote": config_remote,
                        "config_sha256": config_sha,
                        "phase": PHASE_NAME,
                        "control_root_remote": control_root,
                        "checkpoint_root_gcs": checkpoint_root,
                        "source_overlay_control_root_remote": (
                            plane.source_overlay_control_root_remote
                        ),
                        "source_overlay_object_prefix_gcs": (
                            plane.source_overlay_object_prefix_gcs
                        ),
                        "completion_path_controller": str(
                            Path(controller_group)
                            / "runs"
                            / run_id
                            / "training-result.json"
                        ),
                        "environment": {},
                    }
                )
                run_root = str(PurePosixPath(control_root) / "runs" / run_id)
                for milestone in MILESTONES:
                    eval_plane = core_evaluations[(panel, seed, milestone)]
                    remote_result = _replace_core_prefix(
                        eval_plane.result_root_remote,
                        core_marker=core_marker,
                        replacement=remote_replacement,
                        name=f"{panel}/seed{seed}/{milestone} result root",
                    )
                    remote_result = str(
                        PurePosixPath(remote_result) / slug
                    )
                    controller_result = _replace_core_prefix(
                        str(eval_plane.completion_path_controller.parent),
                        core_marker=controller_core_marker,
                        replacement=controller_replacement,
                        name=f"{panel}/seed{seed}/{milestone} controller result",
                    )
                    controller_result = str(
                        Path(controller_result) / slug
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
                            "eligible_resource_ids": list(
                                eval_plane.eligible_resource_ids
                            ),
                            "config_remote": config_remote,
                            "config_sha256": config_sha,
                            "h_config_remote": None,
                            "h_config_sha256": None,
                            "training_source_revision": project_commit,
                            "phase": PHASE_NAME,
                            "control_root_remote": control_root,
                            "checkpoint_root_gcs": checkpoint_root,
                            "train_source_root_remote": (
                                eval_plane.train_source_root_remote
                            ),
                            "eval_source_root_remote": (
                                eval_plane.eval_source_root_remote
                            ),
                            "train_source_object_prefix_gcs": (
                                eval_plane.train_source_object_prefix_gcs
                            ),
                            "eval_source_object_prefix_gcs": (
                                eval_plane.eval_source_object_prefix_gcs
                            ),
                            "run_root_remote": run_root,
                            "result_root_remote": remote_result,
                            "completion_path_controller": str(
                                Path(controller_result) / "evaluation-result.json"
                            ),
                            "result_root_controller": controller_result,
                            "environment": {},
                        }
                    )
            for milestone in MILESTONES:
                eval_plane = core_evaluations[(panel, seed, milestone)]
                parent_h = h_by_role[(panel, seed, milestone)]
                remote_result = _replace_core_prefix(
                    eval_plane.result_root_remote,
                    core_marker=core_marker,
                    replacement=remote_replacement,
                    name=f"{panel}/seed{seed}/{milestone} H result root",
                )
                remote_result = str(PurePosixPath(remote_result) / "h")
                controller_result = _replace_core_prefix(
                    str(eval_plane.completion_path_controller.parent),
                    core_marker=controller_core_marker,
                    replacement=controller_replacement,
                    name=f"{panel}/seed{seed}/{milestone} H controller result",
                )
                controller_result = str(Path(controller_result) / "h")
                evaluations.append(
                    {
                        "id": f"eval-{panel}-h-seed{seed}-{milestone}",
                        "panel": panel,
                        "condition": "H",
                        "seed": seed,
                        "milestone_tokens": milestone,
                        "eligible_resource_ids": list(
                            eval_plane.eligible_resource_ids
                        ),
                        "config_remote": config_remote,
                        "config_sha256": config_sha,
                        "h_config_remote": parent_h.h_config_remote,
                        "h_config_sha256": parent_h.h_config_sha256,
                        "training_source_revision": (
                            parent_h.training_source_revision
                        ),
                        "phase": PHASE_NAME,
                        "control_root_remote": control_root,
                        "checkpoint_root_gcs": (
                            parent_h.checkpoint_payload_root_gcs
                        ),
                        "train_source_root_remote": (
                            eval_plane.train_source_root_remote
                        ),
                        "eval_source_root_remote": (
                            eval_plane.eval_source_root_remote
                        ),
                        "train_source_object_prefix_gcs": (
                            eval_plane.train_source_object_prefix_gcs
                        ),
                        "eval_source_object_prefix_gcs": (
                            eval_plane.eval_source_object_prefix_gcs
                        ),
                        "run_root_remote": parent_h.run_root_remote,
                        "result_root_remote": remote_result,
                        "result_root_controller": controller_result,
                        "completion_path_controller": str(
                            Path(controller_result) / "evaluation-result.json"
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
        "working_directory": str(core_plan.working_directory),
        "python_executable": str(core_plan.python_executable),
        "remote_project_root": core_plan.remote_project_root,
        "remote_uv": core_plan.remote_uv,
        "project_commit": project_commit,
        "max_parallel_train": 3,
        "max_parallel_eval": 3,
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
        "environment": dict(core_plan.environment),
        "resources": copy.deepcopy(core["resources"]),
        "trajectories": trajectories,
        "evaluations": evaluations,
        "h_reference_manifest": h_manifest,
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
    "build_evaluation_command",
    "build_h_reference_manifest",
    "build_routed_ladder_campaign_plan",
    "build_train_command",
    "inspect_evaluation_completion",
    "inspect_training_completion",
    "load_h_reference_manifest",
    "load_routed_ladder_campaign_plan",
    "parse_h_reference_manifest",
    "parse_routed_ladder_campaign_plan",
    "routed_ladder_campaign_status",
    "write_h_reference_manifest",
    "write_routed_ladder_campaign_plan",
]
