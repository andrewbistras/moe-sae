"""Append-only evaluation recovery for the frozen Experiment-5 training farm.

The original Experiment-5 campaign is immutable.  In particular, a repair to
the evaluator must not rewrite its plan, training barrier, failed evaluation
attempts, or result namespace.  This module defines a sibling protocol which:

* parents to the exact sealed 24-trajectory training barrier;
* pins every training result and every 2M/5M/10M checkpoint pointer;
* executes only the fixed eighteen panel/seed/milestone evaluations;
* writes into disjoint controller state and result roots;
* records a control-plane observation before treating a failure as spot
  preemption; and
* seals a normal Experiment-5 report inside a recovery provenance envelope.

The scheduler machinery is inherited from :mod:`moe_sae.experiment5_campaign`.
Only the immutable protocol boundary and the evaluation-only lifecycle live
here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from moe_sae.checkpointing import milestone_pointer_path, validate_pointer
from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.direct_horizon_campaign import _log_evidence
from moe_sae.experiment5_campaign import (
    COMPLETION_VISIBILITY_POLL_SECONDS,
    COMPLETION_VISIBILITY_RETRIES,
    CONDITIONS,
    EVALUATION_TASK_COUNT,
    MILESTONES,
    PANELS,
    PHASE_NAME,
    PROTOCOL_ID,
    RETRYABLE_FAILURES,
    SEEDS,
    TRAINING_RESULT_SCHEMA_VERSION,
    TRAINING_TASK_COUNT,
    TRANSIENT_FAILURES,
    CampaignResource,
    CompletionFingerprint,
    CostPolicy,
    EvaluationTask,
    EvidencePath,
    Experiment5Campaign,
    Experiment5CampaignPlan,
    Experiment5CampaignPlanError,
    Experiment5CampaignStateError,
    ResourceReadinessObservation,
    ResourceReadinessStatus,
    RetryPolicy,
    _AttemptOutcome,
    _canonical_json_bytes,
    _controller_path,
    _environment,
    _evaluation_task,
    _exact_mapping,
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
    _verify_checkout,
    _verify_file,
    _write_create_only,
    inspect_evaluation_completion,
    inspect_training_completion,
    load_experiment5_campaign_plan,
)
from moe_sae.experiment5_report import (
    Experiment5ReportError,
    validate_experiment5_budget_ledger,
    validate_experiment5_report,
)
from moe_sae.horizon_supervisor import classify_failure

PLAN_SCHEMA_VERSION = "gemma4-experiment5-evaluation-recovery-plan-v2"
PARENT_SCHEMA_VERSION = "gemma4-experiment5-evaluation-recovery-parent-v1"
REPORT_ENVELOPE_SCHEMA_VERSION = (
    "gemma4-experiment5-evaluation-recovery-report-envelope-v1"
)
HANDOFF_SCHEMA_VERSION = "gemma4-experiment5-evaluation-recovery-handoff-v1"
FROZEN_TRAINING_REVISION = "674cecbb19ba9130a81a4e27249c9862de08ad17"
RECOVERY_NAMESPACE = "experiment5/recovery/evaluation-v1"
CHECKPOINT_POINTER_COUNT = TRAINING_TASK_COUNT * len(MILESTONES)


class EvaluationRecoveryPlanError(Experiment5CampaignPlanError):
    """The recovery plan does not exactly parent the frozen training farm."""


class EvaluationRecoveryStateError(Experiment5CampaignStateError):
    """Recovery state, lineage, or completion evidence is invalid."""


@dataclass(frozen=True, slots=True)
class CheckpointPointerParent:
    milestone_tokens: int
    artifact: EvidencePath
    pointer_sha256: str


@dataclass(frozen=True, slots=True)
class TrainingParent:
    task_id: str
    panel: str
    condition: str
    seed: int
    run_root_remote: str
    training_result: EvidencePath
    training_result_sha256: str
    checkpoint_pointers: tuple[CheckpointPointerParent, ...]

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.panel, self.condition, self.seed


@dataclass(frozen=True, slots=True)
class RecoveryParent:
    core_campaign_id: str
    core_plan: EvidencePath
    core_plan_sha256: str
    training_barrier: EvidencePath
    training_barrier_record_sha256: str
    training_source_revision: str
    training_cells: tuple[TrainingParent, ...]


@dataclass(frozen=True, slots=True)
class EvaluationRecoveryPlan:
    campaign_id: str
    state_root: Path
    result_roots_controller: tuple[tuple[str, Path], ...]
    result_root_remote: str
    handoff_path: Path
    working_directory: Path
    python_executable: Path
    remote_project_root: str
    remote_uv: str
    project_commit: str
    max_parallel_eval: int
    retry_policy: RetryPolicy
    cost_policy: CostPolicy
    environment: tuple[tuple[str, str], ...]
    resources: tuple[CampaignResource, ...]
    evaluations: tuple[EvaluationTask, ...]
    parent: RecoveryParent
    core_plan: Experiment5CampaignPlan
    plan_sha256: str
    source: Mapping[str, Any]

    @property
    def trajectories(self) -> tuple[()]:
        """The recovery protocol has no training operation by construction."""

        return ()

    @property
    def max_parallel_train(self) -> int:
        # Required only by the inherited plan interface; no training phase is
        # ever invoked.
        return 1

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)

    def result_roots_controller_dict(self) -> dict[str, Path]:
        return dict(self.result_roots_controller)

    def resource(self, resource_id: str) -> CampaignResource:
        for resource in self.resources:
            if resource.resource_id == resource_id:
                return resource
        raise EvaluationRecoveryPlanError(f"unknown resource {resource_id!r}")


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _controller_result_roots(
    value: Any,
) -> tuple[tuple[str, Path], ...]:
    if not isinstance(value, Mapping) or not value:
        raise EvaluationRecoveryPlanError(
            "result_roots_controller must be a nonempty region mapping"
        )
    if not all(isinstance(region, str) for region in value):
        raise EvaluationRecoveryPlanError(
            "controller result-root region keys must be strings"
        )
    roots = tuple(
        (
            _safe_id(region, f"result_roots_controller[{region!r}]"),
            _controller_path(
                str(path), f"result_roots_controller[{region!r}]"
            ),
        )
        for region, path in sorted(value.items())
    )
    if len({path for _region, path in roots}) != len(roots):
        raise EvaluationRecoveryPlanError(
            "controller result roots must be unique per region"
        )
    return roots


def _evidence_with_expected_path(
    value: Any,
    *,
    name: str,
    expected_controller: Path,
    expected_remote: str,
) -> EvidencePath:
    from moe_sae.experiment5_campaign import _evidence

    selected = _evidence(value, name)
    if (
        selected.controller_path != expected_controller
        or selected.remote_path != expected_remote
    ):
        raise EvaluationRecoveryPlanError(f"{name} path changed")
    return selected


def _checkpoint_parent(
    value: Any,
    *,
    name: str,
    core_task: Any,
    run_root_remote: str,
) -> CheckpointPointerParent:
    selected = _exact_mapping(
        value,
        {"milestone_tokens", "artifact", "pointer_sha256"},
        name,
    )
    milestone = selected["milestone_tokens"]
    if milestone not in MILESTONES:
        raise EvaluationRecoveryPlanError(f"{name}.milestone_tokens changed")
    controller_path = milestone_pointer_path(
        core_task.completion_path_controller.parent / "checkpoints",
        milestone,
    )
    remote_path = str(
        milestone_pointer_path(
            PurePosixPath(run_root_remote) / "checkpoints",
            milestone,
        )
    )
    return CheckpointPointerParent(
        milestone_tokens=milestone,
        artifact=_evidence_with_expected_path(
            selected["artifact"],
            name=f"{name}.artifact",
            expected_controller=controller_path,
            expected_remote=remote_path,
        ),
        pointer_sha256=_sha256(
            selected["pointer_sha256"], f"{name}.pointer_sha256"
        ),
    )


def _training_parent(
    value: Any,
    *,
    index: int,
    core_plan: Experiment5CampaignPlan,
) -> TrainingParent:
    name = f"parent.training_cells[{index}]"
    selected = _exact_mapping(
        value,
        {
            "task_id",
            "panel",
            "condition",
            "seed",
            "run_root_remote",
            "training_result",
            "training_result_sha256",
            "checkpoint_pointers",
        },
        name,
    )
    identity = (selected["panel"], selected["condition"], selected["seed"])
    by_identity = {
        (task.panel, task.condition, task.seed): task
        for task in core_plan.trajectories
    }
    task = by_identity.get(identity)
    if task is None or selected["task_id"] != task.task_id:
        raise EvaluationRecoveryPlanError(f"{name} scientific identity changed")
    run_root = str(PurePosixPath(task.control_root_remote) / "runs" / task.run_id)
    if selected["run_root_remote"] != run_root:
        raise EvaluationRecoveryPlanError(f"{name}.run_root_remote changed")
    pointers_raw = selected["checkpoint_pointers"]
    if not isinstance(pointers_raw, list) or len(pointers_raw) != len(MILESTONES):
        raise EvaluationRecoveryPlanError(
            f"{name} must bind exactly three checkpoint pointers"
        )
    pointers = tuple(
        _checkpoint_parent(
            row,
            name=f"{name}.checkpoint_pointers[{pointer_index}]",
            core_task=task,
            run_root_remote=run_root,
        )
        for pointer_index, row in enumerate(pointers_raw)
    )
    if {row.milestone_tokens for row in pointers} != set(MILESTONES):
        raise EvaluationRecoveryPlanError(
            f"{name} checkpoint milestone matrix changed"
        )
    return TrainingParent(
        task_id=task.task_id,
        panel=task.panel,
        condition=task.condition,
        seed=task.seed,
        run_root_remote=run_root,
        training_result=_evidence_with_expected_path(
            selected["training_result"],
            name=f"{name}.training_result",
            expected_controller=task.completion_path_controller,
            expected_remote=str(
                PurePosixPath(run_root) / "training-result.json"
            ),
        ),
        training_result_sha256=_sha256(
            selected["training_result_sha256"],
            f"{name}.training_result_sha256",
        ),
        checkpoint_pointers=pointers,
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
    result = CostPolicy(
        hard_limit_usd=_money(selected["hard_limit_usd"], "hard_limit_usd"),
        contingency_usd=_money(
            selected["contingency_usd"], "contingency_usd"
        ),
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
    if (
        result.hard_limit_usd != 50.0
        or result.contingency_usd != 5.0
        or result.training_attempt_reserve_usd != 0.0
    ):
        raise EvaluationRecoveryPlanError(
            "recovery requires the $50/$5 ceiling and zero training reserve"
        )
    baseline = (
        result.already_spent_usd
        + result.fixed_cost_reserve_usd
        + EVALUATION_TASK_COUNT * result.evaluation_attempt_reserve_usd
    )
    if baseline > result.initial_launch_limit_usd + 1e-9:
        raise EvaluationRecoveryPlanError(
            "recovery baseline exceeds the protected $45 launch envelope"
        )
    return result


def _parse_parent(
    value: Any,
    *,
    core_plan: Experiment5CampaignPlan,
) -> RecoveryParent:
    selected = _exact_mapping(
        value,
        {
            "schema_version",
            "core_campaign_id",
            "core_plan",
            "core_plan_sha256",
            "training_barrier",
            "training_barrier_record_sha256",
            "training_source_revision",
            "training_cells",
        },
        "parent",
    )
    if selected["schema_version"] != PARENT_SCHEMA_VERSION:
        raise EvaluationRecoveryPlanError("recovery parent schema changed")
    if (
        selected["core_campaign_id"] != core_plan.campaign_id
        or selected["core_plan_sha256"] != core_plan.plan_sha256
        or selected["training_source_revision"] != FROZEN_TRAINING_REVISION
        or core_plan.project_commit != FROZEN_TRAINING_REVISION
    ):
        raise EvaluationRecoveryPlanError("frozen core campaign identity changed")
    core_plan_path = Path(str(selected["core_plan"]["controller_path"])).resolve()
    core_plan_evidence = _evidence_with_expected_path(
        selected["core_plan"],
        name="parent.core_plan",
        expected_controller=core_plan_path,
        expected_remote=str(core_plan_path),
    )
    barrier_path = core_plan.state_root / "training-complete.json"
    barrier_evidence = _evidence_with_expected_path(
        selected["training_barrier"],
        name="parent.training_barrier",
        expected_controller=barrier_path,
        expected_remote=str(barrier_path),
    )
    raw_cells = selected["training_cells"]
    if not isinstance(raw_cells, list) or len(raw_cells) != TRAINING_TASK_COUNT:
        raise EvaluationRecoveryPlanError(
            "recovery parent must bind exactly 24 training cells"
        )
    cells = tuple(
        _training_parent(row, index=index, core_plan=core_plan)
        for index, row in enumerate(raw_cells)
    )
    expected = {
        (panel, condition, seed)
        for panel in PANELS
        for condition in CONDITIONS
        for seed in SEEDS
    }
    if {row.identity for row in cells} != expected:
        raise EvaluationRecoveryPlanError("training parent matrix changed")
    if (
        sum(len(row.checkpoint_pointers) for row in cells)
        != CHECKPOINT_POINTER_COUNT
    ):
        raise EvaluationRecoveryPlanError("checkpoint parent matrix changed")
    return RecoveryParent(
        core_campaign_id=core_plan.campaign_id,
        core_plan=core_plan_evidence,
        core_plan_sha256=core_plan.plan_sha256,
        training_barrier=barrier_evidence,
        training_barrier_record_sha256=_sha256(
            selected["training_barrier_record_sha256"],
            "parent.training_barrier_record_sha256",
        ),
        training_source_revision=FROZEN_TRAINING_REVISION,
        training_cells=cells,
    )


def _validate_plan_relations(plan: EvaluationRecoveryPlan) -> None:
    resource_ids = {resource.resource_id for resource in plan.resources}
    if len(resource_ids) != len(plan.resources):
        raise EvaluationRecoveryPlanError("resource IDs must be unique")
    physical = {
        (resource.project, resource.zone, resource.tpu_name)
        for resource in plan.resources
    }
    if len(physical) != len(plan.resources):
        raise EvaluationRecoveryPlanError("physical TPU resources must be unique")
    expected = {
        (panel, seed, milestone)
        for panel in PANELS
        for seed in SEEDS
        for milestone in MILESTONES
    }
    if {
        (task.panel, task.seed, task.milestone_tokens)
        for task in plan.evaluations
    } != expected:
        raise EvaluationRecoveryPlanError("evaluation matrix is not exact")
    if len({task.task_id for task in plan.evaluations}) != EVALUATION_TASK_COUNT:
        raise EvaluationRecoveryPlanError("evaluation task IDs are not unique")
    core_eval = {
        (task.panel, task.seed, task.milestone_tokens): task
        for task in plan.core_plan.evaluations
    }
    required_regions: set[str] = set()
    for old in plan.core_plan.evaluations:
        regions = {
            plan.core_plan.resource(resource_id).region
            for resource_id in old.eligible_resource_ids
        }
        if len(regions) != 1:
            raise EvaluationRecoveryPlanError(
                f"{old.task_id} frozen regional plane is ambiguous"
            )
        required_regions.update(regions)
    result_roots = plan.result_roots_controller_dict()
    if set(result_roots) != required_regions:
        raise EvaluationRecoveryPlanError(
            "controller result roots do not exactly cover frozen regions"
        )
    new_roots = [plan.state_root, *result_roots.values()]
    if any(
        _paths_overlap(first, second)
        for index, first in enumerate(new_roots)
        for second in new_roots[index + 1 :]
    ):
        raise EvaluationRecoveryPlanError(
            "recovery state and regional result roots must be disjoint"
        )
    old_paths = [
        plan.core_plan.state_root,
        *(
            task.completion_path_controller.parent
            for task in plan.core_plan.trajectories
        ),
        *(
            task.completion_path_controller.parent
            for task in plan.core_plan.evaluations
        ),
    ]
    if any(
        _paths_overlap(new, old)
        for new in new_roots
        for old in old_paths
    ):
        raise EvaluationRecoveryPlanError(
            "recovery state or result root overlaps a frozen namespace"
        )
    for old in plan.core_plan.evaluations:
        if (
            plan.result_root_remote == old.result_root_remote
            or plan.result_root_remote.startswith(old.result_root_remote + "/")
            or old.result_root_remote.startswith(plan.result_root_remote + "/")
        ):
            raise EvaluationRecoveryPlanError(
                "recovery remote result root overlaps the frozen result namespace"
            )
    parent_runs = {
        (row.panel, row.condition, row.seed): row.run_root_remote
        for row in plan.parent.training_cells
    }
    completion_paths: list[str] = []
    for task in plan.evaluations:
        if not set(task.eligible_resource_ids) <= resource_ids:
            raise EvaluationRecoveryPlanError(
                f"{task.task_id} references an unknown resource"
            )
        eligible = [
            plan.resource(resource_id)
            for resource_id in task.eligible_resource_ids
        ]
        if task.panel == "transcoder" and any(
            resource.cache_profile != "paired" for resource in eligible
        ):
            raise EvaluationRecoveryPlanError(
                f"{task.task_id} permits a target-only cache"
            )
        old = core_eval[(task.panel, task.seed, task.milestone_tokens)]
        frozen_resources = [
            plan.core_plan.resource(resource_id)
            for resource_id in old.eligible_resource_ids
        ]
        frozen_accelerators = {
            resource.accelerator_type for resource in frozen_resources
        }
        frozen_regions = {resource.region for resource in frozen_resources}
        if (
            len(frozen_accelerators) != 1
            or len(frozen_regions) != 1
            or {resource.accelerator_type for resource in eligible}
            != frozen_accelerators
            or {resource.region for resource in eligible} != frozen_regions
        ):
            raise EvaluationRecoveryPlanError(
                f"{task.task_id} changes its frozen accelerator or "
                "regional artifact plane"
            )
        expected_roots = {
            condition: parent_runs[(task.panel, condition, task.seed)]
            for condition in CONDITIONS
        }
        if (
            task.config_remote != old.config_remote
            or task.control_root_remote != old.control_root_remote
            or task.checkpoint_root_gcs != old.checkpoint_root_gcs
            or task.train_source_root_remote != old.train_source_root_remote
            or task.eval_source_root_remote != old.eval_source_root_remote
            or task.train_source_object_prefix_gcs
            != old.train_source_object_prefix_gcs
            or task.eval_source_object_prefix_gcs
            != old.eval_source_object_prefix_gcs
            or task.run_roots_dict() != expected_roots
        ):
            raise EvaluationRecoveryPlanError(
                f"{task.task_id} does not read the frozen training plane"
            )
        expected_controller = (
            result_roots[next(iter(frozen_regions))]
            / task.panel
            / f"seed{task.seed}"
            / str(task.milestone_tokens)
        )
        expected_remote = str(
            PurePosixPath(plan.result_root_remote)
            / task.panel
            / f"seed{task.seed}"
            / str(task.milestone_tokens)
        )
        if (
            task.completion_path_controller
            != expected_controller / "scientific-result.json"
            or task.paired_scores_path_controller
            != expected_controller / "paired-scores.json"
            or task.result_root_remote != expected_remote
        ):
            raise EvaluationRecoveryPlanError(
                f"{task.task_id} does not use the recovery result namespace"
            )
        completion_paths.extend(
            (
                str(task.completion_path_controller),
                str(task.paired_scores_path_controller),
            )
        )
    if len(set(completion_paths)) != len(completion_paths):
        raise EvaluationRecoveryPlanError("recovery result paths are not unique")


def parse_evaluation_recovery_plan(value: Any) -> EvaluationRecoveryPlan:
    selected = _exact_mapping(
        value,
        {
            "schema_version",
            "protocol_id",
            "campaign_id",
            "state_root",
            "result_roots_controller",
            "result_root_remote",
            "handoff_path",
            "working_directory",
            "python_executable",
            "remote_project_root",
            "remote_uv",
            "project_commit",
            "max_parallel_eval",
            "retry_policy",
            "cost_policy",
            "environment",
            "resources",
            "evaluations",
            "parent",
        },
        "plan",
    )
    if selected["schema_version"] != PLAN_SCHEMA_VERSION:
        raise EvaluationRecoveryPlanError("recovery plan schema changed")
    if selected["protocol_id"] != PROTOCOL_ID:
        raise EvaluationRecoveryPlanError("scientific protocol changed")
    project_commit = _git_sha(selected["project_commit"], "project_commit")
    if project_commit == FROZEN_TRAINING_REVISION:
        raise EvaluationRecoveryPlanError(
            "evaluation recovery requires a new source revision"
        )
    core_plan_path = Path(
        str(selected["parent"]["core_plan"]["controller_path"])
    ).expanduser().resolve()
    core_plan = load_experiment5_campaign_plan(core_plan_path)
    parent = _parse_parent(selected["parent"], core_plan=core_plan)
    raw_resources = selected["resources"]
    raw_evaluations = selected["evaluations"]
    if not isinstance(raw_resources, list) or not raw_resources:
        raise EvaluationRecoveryPlanError("recovery requires at least one resource")
    if len(raw_resources) > 32:
        raise EvaluationRecoveryPlanError("recovery resource inventory is unbounded")
    if (
        not isinstance(raw_evaluations, list)
        or len(raw_evaluations) != EVALUATION_TASK_COUNT
    ):
        raise EvaluationRecoveryPlanError(
            "recovery must schedule exactly 18 evaluations"
        )
    resources = tuple(
        _resource(row, index, project_commit)
        for index, row in enumerate(raw_resources)
    )
    evaluations = tuple(
        _evaluation_task(row, index)
        for index, row in enumerate(raw_evaluations)
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
        raise EvaluationRecoveryPlanError("retry backoff maximum precedes initial")
    state_root = _controller_path(selected["state_root"], "state_root")
    result_roots = _controller_result_roots(
        selected["result_roots_controller"]
    )
    handoff_path = _controller_path(
        selected["handoff_path"], "handoff_path"
    )
    if handoff_path != state_root / "recovery-handoff.json":
        raise EvaluationRecoveryPlanError("handoff path is outside recovery state")
    plan = EvaluationRecoveryPlan(
        campaign_id=_safe_id(selected["campaign_id"], "campaign_id"),
        state_root=state_root,
        result_roots_controller=result_roots,
        result_root_remote=_remote_path(
            selected["result_root_remote"], "result_root_remote"
        ),
        handoff_path=handoff_path,
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
        max_parallel_eval=_positive_int(
            selected["max_parallel_eval"],
            "max_parallel_eval",
            maximum=len(resources),
        ),
        retry_policy=retry,
        cost_policy=_cost_policy(selected["cost_policy"]),
        environment=_environment(selected["environment"], "environment"),
        resources=resources,
        evaluations=evaluations,
        parent=parent,
        core_plan=core_plan,
        plan_sha256=canonical_sha256(selected),
        source=selected,
    )
    _validate_plan_relations(plan)
    return plan


def load_evaluation_recovery_plan(
    path: str | Path,
) -> EvaluationRecoveryPlan:
    selected = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise EvaluationRecoveryPlanError(
            f"cannot read evaluation recovery plan: {selected}"
        ) from error
    return parse_evaluation_recovery_plan(value)


def _read_pointer_parent(
    parent: CheckpointPointerParent,
    *,
    training_result: Mapping[str, Any],
    checkpoint_root_gcs: str,
) -> None:
    try:
        _verify_file(
            parent.artifact.controller_path,
            parent.artifact.sha256,
            f"{parent.milestone_tokens} checkpoint pointer",
        )
    except Experiment5CampaignStateError as error:
        raise EvaluationRecoveryStateError(
            f"{parent.milestone_tokens} checkpoint pointer changed"
        ) from error
    pointer, _payload = _read_self_hashed_json(
        parent.artifact.controller_path,
        hash_field="pointer_sha256",
    )
    if pointer.get("pointer_sha256") != parent.pointer_sha256:
        raise EvaluationRecoveryStateError(
            "checkpoint pointer scientific hash changed"
        )
    try:
        checked = validate_pointer(
            pointer,
            expected_kind="latest",
            payload_root=checkpoint_root_gcs,
            run_identity=training_result["run_identity"],
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise EvaluationRecoveryStateError(
            "checkpoint pointer lineage is invalid"
        ) from error
    identity = training_result.get("run_identity")
    run_plan = identity.get("plan") if isinstance(identity, Mapping) else None
    if not isinstance(run_plan, Mapping):
        raise EvaluationRecoveryStateError(
            "training result lacks its exact-token worker plan"
        )
    microbatch = run_plan.get("global_microbatch_tokens")
    accumulation = run_plan.get("microbatches_per_update")
    total = run_plan.get("total_tokens")
    planned_updates = run_plan.get("planned_updates")
    if (
        isinstance(microbatch, bool)
        or not isinstance(microbatch, int)
        or microbatch <= 0
        or isinstance(accumulation, bool)
        or not isinstance(accumulation, int)
        or accumulation <= 0
        or total != 10_000_000
        or isinstance(planned_updates, bool)
        or not isinstance(planned_updates, int)
    ):
        raise EvaluationRecoveryStateError(
            "training result has an invalid exact-token worker plan"
        )
    batch = microbatch * accumulation
    if planned_updates != math.ceil(total / batch):
        raise EvaluationRecoveryStateError(
            "training result planned-update count changed"
        )
    expected_step = math.ceil(parent.milestone_tokens / batch)
    expected_tokens = min(expected_step * batch, total)
    if (
        checked["step"] != expected_step
        or checked["tokens_seen"] != expected_tokens
    ):
        raise EvaluationRecoveryStateError(
            "checkpoint pointer milestone progress changed"
        )


def _verify_parent(plan: EvaluationRecoveryPlan) -> dict[str, Any]:
    parent = plan.parent
    for label, evidence in (
        ("core plan", parent.core_plan),
        ("training barrier", parent.training_barrier),
    ):
        _verify_file(evidence.controller_path, evidence.sha256, label)
    loaded_core = load_experiment5_campaign_plan(parent.core_plan.controller_path)
    if (
        loaded_core.plan_sha256 != parent.core_plan_sha256
        or loaded_core.campaign_id != parent.core_campaign_id
        or loaded_core.project_commit != parent.training_source_revision
    ):
        raise EvaluationRecoveryStateError("frozen core plan changed")
    barrier = _read_record(
        parent.training_barrier.controller_path,
        record_type="training_phase_completion",
    )
    rows = barrier.get("training")
    if (
        barrier.get("campaign_id") != parent.core_campaign_id
        or barrier.get("plan_sha256") != parent.core_plan_sha256
        or barrier.get("status") != "training_completed"
        or barrier.get("record_sha256") != parent.training_barrier_record_sha256
        or not isinstance(rows, Mapping)
        or len(rows) != TRAINING_TASK_COUNT
    ):
        raise EvaluationRecoveryStateError("frozen training barrier changed")
    tasks = {task.task_id: task for task in loaded_core.trajectories}
    for cell in parent.training_cells:
        task = tasks[cell.task_id]
        completion = inspect_training_completion(loaded_core, task)
        if completion is None or completion.to_dict() != rows.get(cell.task_id):
            raise EvaluationRecoveryStateError(
                f"training barrier cell changed: {cell.task_id}"
            )
        _verify_file(
            cell.training_result.controller_path,
            cell.training_result.sha256,
            f"{cell.task_id} training result",
        )
        result, _payload = _read_self_hashed_json(
            cell.training_result.controller_path,
            hash_field="result_sha256",
        )
        if (
            result.get("schema_version") != TRAINING_RESULT_SCHEMA_VERSION
            or result.get("result_sha256") != cell.training_result_sha256
            or result.get("source_revision") != parent.training_source_revision
        ):
            raise EvaluationRecoveryStateError(
                f"training result lineage changed: {cell.task_id}"
            )
        for pointer in cell.checkpoint_pointers:
            _read_pointer_parent(
                pointer,
                training_result=result,
                checkpoint_root_gcs=task.checkpoint_root_gcs,
            )
    return barrier


def _preflight(
    plan: EvaluationRecoveryPlan,
    *,
    verify_checkout: bool,
) -> dict[str, Any]:
    if verify_checkout:
        _verify_checkout(plan)  # type: ignore[arg-type]
        evaluator = (
            plan.working_directory / "scripts/tpu/evaluate_experiment5.py"
        )
        if evaluator.is_symlink() or not evaluator.is_file():
            raise EvaluationRecoveryStateError("recovery evaluator is absent")
    active_ids = {
        resource_id
        for task in plan.evaluations
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
    return _verify_parent(plan)


def _parent_binding_body(plan: EvaluationRecoveryPlan) -> dict[str, Any]:
    return {
        "core_campaign_id": plan.parent.core_campaign_id,
        "core_plan_sha256": plan.parent.core_plan_sha256,
        "training_source_revision": plan.parent.training_source_revision,
        "training_barrier_record_sha256": (
            plan.parent.training_barrier_record_sha256
        ),
        "training_cells": {
            cell.task_id: {
                "training_result_sha256": cell.training_result_sha256,
                "training_result_file_sha256": cell.training_result.sha256,
                "checkpoint_pointers": {
                    str(pointer.milestone_tokens): {
                        "pointer_sha256": pointer.pointer_sha256,
                        "pointer_file_sha256": pointer.artifact.sha256,
                    }
                    for pointer in sorted(
                        cell.checkpoint_pointers,
                        key=lambda row: row.milestone_tokens,
                    )
                },
            }
            for cell in sorted(
                plan.parent.training_cells,
                key=lambda row: (row.panel, row.condition, row.seed),
            )
        },
    }


class EvaluationRecoveryCampaign(Experiment5Campaign):
    """Execute only the eighteen recovered evaluations."""

    plan: EvaluationRecoveryPlan

    def __init__(self, plan: EvaluationRecoveryPlan, **kwargs: Any) -> None:
        self._retired_resource_lock = threading.Lock()
        self._retired_resources: dict[
            str, ResourceReadinessObservation
        ] = {}
        super().__init__(plan, **kwargs)  # type: ignore[arg-type]

    def _baseline_attempt_reserve_usd(self) -> float:
        return (
            EVALUATION_TASK_COUNT
            * self.plan.cost_policy.evaluation_attempt_reserve_usd
        )

    def _preflight(self) -> None:
        _preflight(self.plan, verify_checkout=self.verify_checkout)

    def _inspector(
        self, task: EvaluationTask
    ) -> CompletionFingerprint | None:
        return inspect_evaluation_completion(self.plan, task)  # type: ignore[arg-type]

    def _attempt_id(self, attempt_number: int) -> str:
        parent = super()._attempt_id(attempt_number)
        return parent.replace("e5-", "e5er-", 1)

    def _probe_resource(
        self,
        resource: CampaignResource,
    ) -> ResourceReadinessObservation:
        with self._retired_resource_lock:
            retired = self._retired_resources.get(resource.resource_id)
        if retired is None:
            return super()._probe_resource(resource)
        # Never reuse an incarnation after append-only terminal evidence.  A
        # forming observation makes this inherited resource worker yield its
        # tentative task while other eligible ready slices drain the queue.
        return ResourceReadinessObservation(
            status=ResourceReadinessStatus.FORMING,
            state=retired.state,
            health=retired.health,
            create_time=retired.create_time,
            spot=retired.spot,
            absent=retired.absent,
            detail=(
                "resource retired for this recovery plan after terminal "
                f"control-plane evidence: {retired.detail}"
            ),
        )

    def _retire_if_alternative_exists(
        self,
        *,
        task: EvaluationTask,
        resource: CampaignResource,
        observation: ResourceReadinessObservation,
    ) -> bool:
        with self._retired_resource_lock:
            alternatives = set(task.eligible_resource_ids) - {
                resource.resource_id,
                *self._retired_resources,
            }
            if not alternatives:
                return False
            self._retired_resources[resource.resource_id] = observation
        return True

    def _execute_once(
        self,
        task: EvaluationTask,
        *,
        kind: str,
        resource: CampaignResource,
        pre_attempt_readiness: ResourceReadinessObservation,
    ) -> _AttemptOutcome:
        """Execute one cell and require control-plane proof for preemption."""

        existing = self._inspector(task)
        if existing is not None:
            self._completion_witness(task, kind=kind, fingerprint=existing)
            return _AttemptOutcome(existing)
        maximum, transient_count = self._prior_state(task, kind=kind)
        number = maximum + 1
        self._reserve_attempt_cost(
            task,
            kind=kind,
            attempt_number=number,
            resource=resource,
        )
        provisional = (
            self._task_root(task.task_id, kind=kind)
            / "attempts"
            / f"attempt-{number:06d}"
        )
        environment, audited = self._attempt_environment(
            resource,
            task,
            provisional / "operator",
        )
        attempt_id = self._attempt_id(number)
        command = self._attempt_command(task, attempt_id=attempt_id)
        attempt_root = self._new_attempt(
            task,
            kind=kind,
            number=number,
            attempt_id=attempt_id,
            command=command,
            audited_environment=audited,
            resource=resource,
            pre_attempt_readiness=pre_attempt_readiness,
        )
        execution = self.attempt_runner(
            task.task_id,
            command,
            environment,
            self.plan.working_directory,
            attempt_root / "controller.log",
            self.stop_event,
        )
        completion = self._inspector(task)
        if (
            completion is None
            and execution.returncode == 0
            and not self.stop_event.is_set()
        ):
            for _ in range(COMPLETION_VISIBILITY_RETRIES):
                if self.stop_event.wait(COMPLETION_VISIBILITY_POLL_SECONDS):
                    break
                completion = self._inspector(task)
                if completion is not None:
                    break
        output = _log_evidence(attempt_root)
        raw_classification = classify_failure(
            returncode=execution.returncode,
            output=output + "\n" + execution.detail,
            timed_out=execution.timed_out,
            completion_present=completion is not None,
            checkpoint_before=None,
            checkpoint_after=None,
            segment=None,
        )
        classification = raw_classification
        if self.stop_event.is_set() and completion is None:
            classification = "controller_stop"
        post: ResourceReadinessObservation | None = None
        confirmed_preemption = False
        needs_corroboration = (
            completion is None
            and raw_classification
            in (set(TRANSIENT_FAILURES) | {"spot_preemption"})
            and pre_attempt_readiness.status is ResourceReadinessStatus.READY
            and not self.stop_event.is_set()
        )
        if needs_corroboration:
            try:
                observed = self.resource_readiness_probe(resource)
                if not isinstance(observed, ResourceReadinessObservation):
                    raise TypeError(
                        "readiness probe did not return "
                        "ResourceReadinessObservation"
                    )
                post = observed
            except Exception as error:
                post = ResourceReadinessObservation(
                    status=ResourceReadinessStatus.TRANSIENT_CONTROL_PLANE,
                    state=None,
                    health=None,
                    create_time=None,
                    spot=None,
                    absent=False,
                    detail=(
                        "post-failure readiness probe failed: "
                        f"{type(error).__name__}: {error}"
                    ),
                )
            confirmed_loss = (
                post.status
                is ResourceReadinessStatus.TERMINAL_REPLACEMENT_REQUIRED
                or (
                    post.status is ResourceReadinessStatus.FORMING
                    and post.absent
                )
            )
            if pre_attempt_readiness.spot is True and confirmed_loss:
                classification = "spot_preemption"
                confirmed_preemption = True
            elif raw_classification == "spot_preemption":
                classification = (
                    "workload_failure"
                    if post.status is ResourceReadinessStatus.READY
                    else "resource_unavailable"
                )
        result = _record(
            "attempt_result",
            {
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "task_id": task.task_id,
                "kind": kind,
                "attempt_number": number,
                "attempt_id": attempt_id,
                "resource_id": resource.resource_id,
                "returncode": execution.returncode,
                "timed_out": execution.timed_out,
                "raw_classification": raw_classification,
                "classification": classification,
                "pre_attempt_readiness": pre_attempt_readiness.to_dict(),
                "post_failure_readiness": (
                    post.to_dict() if post is not None else None
                ),
                "confirmed_preemption": confirmed_preemption,
                "executed_command_sha256": canonical_sha256(list(command)),
                "completion": (
                    completion.to_dict() if completion is not None else None
                ),
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "detail": execution.detail[-4_096:],
                "finished_at": _timestamp(),
            },
        )
        _write_create_only(attempt_root / "result.json", result)
        if post is not None and (
            confirmed_preemption
            or post.status
            is ResourceReadinessStatus.TERMINAL_REPLACEMENT_REQUIRED
        ):
            self._record_resource_readiness_event(
                resource,
                task=task,
                kind=kind,
                stage=(
                    "confirmed-spot-preemption"
                    if confirmed_preemption
                    else "post-attempt-terminal"
                ),
                observation=post,
            )
            if self._retire_if_alternative_exists(
                task=task,
                resource=resource,
                observation=post,
            ):
                return _AttemptOutcome(None, retry_delay_seconds=0.0)
            raise EvaluationRecoveryStateError(
                f"{resource.resource_id} requires replacement after "
                f"{task.task_id} and no eligible recovery resource remains: "
                f"{post.detail}"
            )
        if completion is not None:
            self._completion_witness(task, kind=kind, fingerprint=completion)
            return _AttemptOutcome(completion)
        if classification == "controller_stop":
            raise EvaluationRecoveryStateError(
                f"{task.task_id} stopped before immutable completion"
            )
        if classification not in RETRYABLE_FAILURES:
            self._terminal_failure(
                task,
                kind=kind,
                classification=classification,
                attempt_number=number,
                detail=output or execution.detail,
            )
            raise EvaluationRecoveryStateError(
                f"{task.task_id} failed terminally: {classification}"
            )
        if classification in TRANSIENT_FAILURES:
            transient_count += 1
            if transient_count >= self.plan.retry_policy.max_transient_attempts:
                self._terminal_failure(
                    task,
                    kind=kind,
                    classification="transient_attempts_exhausted",
                    attempt_number=number,
                    detail=output or execution.detail,
                )
                raise EvaluationRecoveryStateError(
                    f"{task.task_id} exhausted transient retries"
                )
        delay = min(
            self.plan.retry_policy.maximum_backoff_seconds,
            self.plan.retry_policy.initial_backoff_seconds
            * (2 ** max(0, transient_count - 1)),
        )
        return _AttemptOutcome(None, retry_delay_seconds=delay)

    def _seal_parent_binding(self) -> dict[str, Any]:
        expected = _record(
            "evaluation_recovery_parent_binding",
            {
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "status": "frozen_training_parent_bound",
                **_parent_binding_body(self.plan),
            },
        )
        path = self.plan.state_root / "parent-training-bound.json"
        if path.exists():
            observed = _read_record(
                path, record_type="evaluation_recovery_parent_binding"
            )
            if observed != expected:
                raise EvaluationRecoveryStateError(
                    "recovery training-parent binding changed"
                )
            return observed
        _write_create_only(path, expected)
        return expected

    def _seal_scientific_report(
        self,
        *,
        evaluation_barrier: Mapping[str, Any],
    ) -> dict[str, Any]:
        inner = super()._seal_scientific_report(
            evaluation_barrier=evaluation_barrier
        )
        envelope_body = {
            "schema_version": REPORT_ENVELOPE_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "campaign_id": self.plan.campaign_id,
            "plan_sha256": self.plan.plan_sha256,
            "evaluation_source_revision": self.plan.project_commit,
            "parent": _parent_binding_body(self.plan),
            "evaluation_barrier_record_sha256": evaluation_barrier[
                "record_sha256"
            ],
            "scientific_report": inner,
        }
        envelope = {
            **envelope_body,
            "envelope_sha256": canonical_sha256(envelope_body),
        }
        path = self.plan.state_root / "report" / "recovery-envelope.json"
        file_sha = _seal_exact_bytes(
            path,
            _canonical_json_bytes(envelope) + b"\n",
            name="Experiment-5 recovery report envelope",
        )
        return {
            **inner,
            "recovery_envelope_path": str(path),
            "recovery_envelope_sha256": envelope["envelope_sha256"],
            "recovery_envelope_file_sha256": file_sha,
        }

    def run(self, *, stop_after_training: bool = False) -> dict[str, Any]:
        if stop_after_training:
            raise EvaluationRecoveryStateError(
                "evaluation recovery has no training phase"
            )
        self._preflight()
        self._acquire_lock()
        try:
            self._bind_plan()
            self._bind_campaign_clock()
            parent_binding = self._seal_parent_binding()
            self._start_watchdog()
            evaluations = self._run_phase(
                self.plan.evaluations,
                kind="evaluation",
                max_parallel=self.plan.max_parallel_eval,
            )
            barrier = self._seal_phase(
                path=self.plan.state_root / "evaluation-complete.json",
                record_type="evaluation_phase_completion",
                status="evaluation_completed",
                field="evaluations",
                completions=evaluations,
            )
            report = self._seal_scientific_report(
                evaluation_barrier=barrier
            )
            cost = self._seal_cost_projection(
                completed_at=barrier["completed_at"]
            )
            final = _record(
                "evaluation_recovery_completion",
                {
                    "campaign_id": self.plan.campaign_id,
                    "plan_sha256": self.plan.plan_sha256,
                    "status": "completed",
                    "evaluation_source_revision": self.plan.project_commit,
                    "parent_binding_record_sha256": parent_binding[
                        "record_sha256"
                    ],
                    "evaluations": {
                        key: evaluations[key].to_dict()
                        for key in sorted(evaluations)
                    },
                    "report": report,
                    "cost_projection_record_sha256": cost["record_sha256"],
                    "completed_at": barrier["completed_at"],
                },
            )
            path = self.plan.state_root / "final.json"
            if path.exists():
                observed = _read_record(
                    path, record_type="evaluation_recovery_completion"
                )
                if observed != final:
                    raise EvaluationRecoveryStateError(
                        "recovery final record changed"
                    )
                return observed
            _write_create_only(path, final)
            return final
        finally:
            self._stop_watchdog()
            self._release_lock()


def evaluation_recovery_status(
    plan: EvaluationRecoveryPlan,
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task in plan.evaluations:
        completion = inspect_evaluation_completion(
            plan, task  # type: ignore[arg-type]
        )
        root = plan.state_root / "evaluation" / task.task_id
        attempts = (
            len([row for row in (root / "attempts").iterdir() if row.is_dir()])
            if (root / "attempts").is_dir()
            else 0
        )
        state = "completed" if completion is not None else "pending"
        if completion is None and (root / "failure.json").exists():
            failure = _read_record(
                root / "failure.json", record_type="task_failure"
            )
            state = f"failed:{failure.get('classification')}"
        elif completion is None and attempts:
            latest = max((root / "attempts").iterdir())
            state = (
                "retryable"
                if (latest / "result.json").exists()
                else "ambiguous_inflight"
            )
        tasks[task.task_id] = {
            "kind": "evaluation",
            "state": state,
            "attempts": attempts,
            "eligible_resources": list(task.eligible_resource_ids),
            "completion": completion.to_dict() if completion else None,
        }
    controller = EvaluationRecoveryCampaign(plan, verify_checkout=False)
    clock_path = plan.state_root / "campaign-clock.json"
    clock = (
        _read_record(clock_path, record_type="campaign_clock")
        if clock_path.exists()
        else None
    )
    if clock is not None:
        controller._activate_campaign_clock(clock)
    final_path = plan.state_root / "final.json"
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.plan_sha256,
        "evaluation_source_revision": plan.project_commit,
        "training_source_revision": plan.parent.training_source_revision,
        "tasks": tasks,
        "training_parent_cells": len(plan.parent.training_cells),
        "checkpoint_parent_cells": sum(
            len(row.checkpoint_pointers)
            for row in plan.parent.training_cells
        ),
        "campaign_clock": clock,
        "cost_projection": controller.cost_projection(),
        "watchdog_projection": (
            controller._watchdog_projection() if clock is not None else None
        ),
        "parent_binding": (
            _read_record(
                plan.state_root / "parent-training-bound.json",
                record_type="evaluation_recovery_parent_binding",
            )
            if (plan.state_root / "parent-training-bound.json").exists()
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
            _read_record(
                final_path, record_type="evaluation_recovery_completion"
            )
            if final_path.exists()
            else None
        ),
    }


def _artifact_reference(
    path: Path,
    *,
    remote_path: str | None = None,
) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise EvaluationRecoveryPlanError(
            f"parent artifact is not a regular file: {path}"
        )
    payload = path.read_bytes()
    return {
        "controller_path": str(path),
        "remote_path": remote_path or str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _build_training_parents(
    core_plan: Experiment5CampaignPlan,
    barrier: Mapping[str, Any],
) -> list[dict[str, Any]]:
    barrier_rows = barrier.get("training")
    if not isinstance(barrier_rows, Mapping):
        raise EvaluationRecoveryPlanError("training barrier lacks completions")
    result: list[dict[str, Any]] = []
    for task in sorted(
        core_plan.trajectories,
        key=lambda row: (row.panel, row.condition, row.seed),
    ):
        completion = inspect_training_completion(core_plan, task)
        if completion is None or completion.to_dict() != barrier_rows.get(
            task.task_id
        ):
            raise EvaluationRecoveryPlanError(
                f"core training is not sealed: {task.task_id}"
            )
        training, _payload = _read_self_hashed_json(
            task.completion_path_controller,
            hash_field="result_sha256",
        )
        run_root = str(
            PurePosixPath(task.control_root_remote) / "runs" / task.run_id
        )
        pointers: list[dict[str, Any]] = []
        for milestone in MILESTONES:
            controller_pointer = milestone_pointer_path(
                task.completion_path_controller.parent / "checkpoints",
                milestone,
            )
            pointer, _pointer_payload = _read_self_hashed_json(
                controller_pointer,
                hash_field="pointer_sha256",
            )
            remote_pointer = str(
                milestone_pointer_path(
                    PurePosixPath(run_root) / "checkpoints",
                    milestone,
                )
            )
            pointers.append(
                {
                    "milestone_tokens": milestone,
                    "artifact": _artifact_reference(
                        controller_pointer,
                        remote_path=remote_pointer,
                    ),
                    "pointer_sha256": pointer["pointer_sha256"],
                }
            )
        result.append(
            {
                "task_id": task.task_id,
                "panel": task.panel,
                "condition": task.condition,
                "seed": task.seed,
                "run_root_remote": run_root,
                "training_result": _artifact_reference(
                    task.completion_path_controller,
                    remote_path=str(
                        PurePosixPath(run_root) / "training-result.json"
                    ),
                ),
                "training_result_sha256": training["result_sha256"],
                "checkpoint_pointers": pointers,
            }
        )
    return result


def build_evaluation_recovery_plan(
    *,
    core_plan_path: str | Path,
    evaluation_source_revision: str,
    resources: Sequence[Mapping[str, Any]],
    state_root: str | Path,
    result_roots_controller: Mapping[str, str | Path],
    result_root_remote: str,
    already_spent_usd: float,
    eligible_resource_ids: Mapping[str, Sequence[str]] | None = None,
    campaign_id: str | None = None,
    max_parallel_eval: int | None = None,
) -> dict[str, Any]:
    """Build one exact recovery plan from the sealed core training barrier."""

    core_path = Path(core_plan_path).expanduser().resolve()
    core = load_experiment5_campaign_plan(core_path)
    if core.project_commit != FROZEN_TRAINING_REVISION:
        raise EvaluationRecoveryPlanError("core training revision changed")
    barrier_path = core.state_root / "training-complete.json"
    barrier = _read_record(
        barrier_path, record_type="training_phase_completion"
    )
    if (
        barrier.get("campaign_id") != core.campaign_id
        or barrier.get("plan_sha256") != core.plan_sha256
        or barrier.get("status") != "training_completed"
        or not isinstance(barrier.get("training"), Mapping)
        or len(barrier["training"]) != TRAINING_TASK_COUNT
    ):
        raise EvaluationRecoveryPlanError(
            "core training barrier is not the exact sealed matrix"
        )
    revision = _git_sha(
        evaluation_source_revision, "evaluation_source_revision"
    )
    if revision == core.project_commit:
        raise EvaluationRecoveryPlanError(
            "recovery must use a distinct evaluator revision"
        )
    raw_resources = [copy.deepcopy(dict(row)) for row in resources]
    if not raw_resources:
        raise EvaluationRecoveryPlanError("recovery has no TPU resources")
    resource_ids = [str(row.get("id")) for row in raw_resources]
    if len(set(resource_ids)) != len(resource_ids):
        raise EvaluationRecoveryPlanError("recovery resource IDs are not unique")
    for row in raw_resources:
        if row.get("deployed_commit") != revision:
            raise EvaluationRecoveryPlanError(
                f"resource {row.get('id')!r} is not freshly evidenced at "
                "the evaluation source revision"
            )
    parsed_resources = tuple(
        _resource(row, index, revision)
        for index, row in enumerate(raw_resources)
    )
    state = Path(state_root).expanduser().resolve()
    controller_results = dict(
        _controller_result_roots(result_roots_controller)
    )
    remote_results = _remote_path(result_root_remote, "result_root_remote")
    core_eval = sorted(
        core.evaluations,
        key=lambda row: (row.panel, row.seed, row.milestone_tokens),
    )
    evaluations: list[dict[str, Any]] = []
    for old in core_eval:
        frozen_resources = [
            core.resource(resource_id)
            for resource_id in old.eligible_resource_ids
        ]
        frozen_accelerators = {
            resource.accelerator_type for resource in frozen_resources
        }
        frozen_regions = {resource.region for resource in frozen_resources}
        if len(frozen_accelerators) != 1 or len(frozen_regions) != 1:
            raise EvaluationRecoveryPlanError(
                f"{old.task_id} frozen execution plane is ambiguous"
            )
        compatible = [
            resource.resource_id
            for resource in parsed_resources
            if resource.accelerator_type in frozen_accelerators
            and resource.region in frozen_regions
            and (
                old.panel == "sae" or resource.cache_profile == "paired"
            )
        ]
        if eligible_resource_ids is not None:
            requested = eligible_resource_ids.get(old.task_id)
            if requested is None:
                raise EvaluationRecoveryPlanError(
                    f"eligibility map lacks {old.task_id}"
                )
            incompatible = set(requested) - set(compatible)
            if incompatible:
                raise EvaluationRecoveryPlanError(
                    f"{old.task_id} explicitly permits incompatible "
                    f"resources: {sorted(incompatible)!r}"
                )
            compatible = list(requested)
        if not compatible:
            raise EvaluationRecoveryPlanError(
                f"{old.task_id} has no compatible recovery resource"
            )
        frozen_region = next(iter(frozen_regions))
        if frozen_region not in controller_results:
            raise EvaluationRecoveryPlanError(
                f"{old.task_id} has no controller result root for "
                f"{frozen_region}"
            )
        destination = (
            controller_results[frozen_region]
            / old.panel
            / f"seed{old.seed}"
            / str(old.milestone_tokens)
        )
        remote_destination = str(
            PurePosixPath(remote_results)
            / old.panel
            / f"seed{old.seed}"
            / str(old.milestone_tokens)
        )
        evaluations.append(
            {
                "id": old.task_id,
                "panel": old.panel,
                "seed": old.seed,
                "milestone_tokens": old.milestone_tokens,
                "eligible_resource_ids": compatible,
                "config_remote": old.config_remote,
                "phase": PHASE_NAME,
                "control_root_remote": old.control_root_remote,
                "checkpoint_root_gcs": old.checkpoint_root_gcs,
                "train_source_root_remote": old.train_source_root_remote,
                "eval_source_root_remote": old.eval_source_root_remote,
                "train_source_object_prefix_gcs": (
                    old.train_source_object_prefix_gcs
                ),
                "eval_source_object_prefix_gcs": (
                    old.eval_source_object_prefix_gcs
                ),
                "result_root_remote": remote_destination,
                "completion_path_controller": str(
                    destination / "scientific-result.json"
                ),
                "paired_scores_path_controller": str(
                    destination / "paired-scores.json"
                ),
                "run_roots_remote": old.run_roots_dict(),
                "environment": {},
            }
        )
    short = revision[:12]
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": (
            campaign_id
            or f"gemma4-experiment5-evaluation-recovery-{short}"
        ),
        "state_root": str(state),
        "result_roots_controller": {
            region: str(path)
            for region, path in sorted(controller_results.items())
        },
        "result_root_remote": remote_results,
        "handoff_path": str(state / "recovery-handoff.json"),
        "working_directory": str(core.working_directory),
        "python_executable": str(core.python_executable),
        "remote_project_root": core.remote_project_root,
        "remote_uv": core.remote_uv,
        "project_commit": revision,
        "max_parallel_eval": (
            min(len(raw_resources), EVALUATION_TASK_COUNT)
            if max_parallel_eval is None
            else max_parallel_eval
        ),
        "retry_policy": {
            "max_transient_attempts": 8,
            "initial_backoff_seconds": 15.0,
            "maximum_backoff_seconds": 300.0,
        },
        "cost_policy": {
            "hard_limit_usd": 50.0,
            "contingency_usd": 5.0,
            "already_spent_usd": float(already_spent_usd),
            "fixed_cost_reserve_usd": 1.0,
            "training_attempt_reserve_usd": 0.0,
            "evaluation_attempt_reserve_usd": 0.2,
        },
        "environment": {
            **core.environment_dict(),
            "EXPERIMENT5_RECOVERY_TRAINING_SOURCE_REVISION": (
                FROZEN_TRAINING_REVISION
            ),
            "EXPERIMENT5_RECOVERY_TRAINING_BARRIER_SHA256": barrier[
                "record_sha256"
            ],
        },
        "resources": raw_resources,
        "evaluations": evaluations,
        "parent": {
            "schema_version": PARENT_SCHEMA_VERSION,
            "core_campaign_id": core.campaign_id,
            "core_plan": _artifact_reference(core_path),
            "core_plan_sha256": core.plan_sha256,
            "training_barrier": _artifact_reference(barrier_path),
            "training_barrier_record_sha256": barrier["record_sha256"],
            "training_source_revision": FROZEN_TRAINING_REVISION,
            "training_cells": _build_training_parents(core, barrier),
        },
    }
    parse_evaluation_recovery_plan(plan)
    return plan


def write_evaluation_recovery_plan(
    path: str | Path,
    plan: Mapping[str, Any],
) -> EvaluationRecoveryPlan:
    parsed = parse_evaluation_recovery_plan(plan)
    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    if selected.exists():
        raise EvaluationRecoveryPlanError(
            f"refusing to overwrite evaluation recovery plan: {selected}"
        )
    selected.write_text(
        yaml.safe_dump(dict(plan), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return parsed


def validate_recovery_handoff(
    value: Mapping[str, Any],
    *,
    plan: EvaluationRecoveryPlan | None = None,
) -> dict[str, Any]:
    selected = _exact_mapping(
        value,
        {
            "schema_version",
            "protocol_id",
            "status",
            "campaign_id",
            "plan_sha256",
            "evaluation_source_revision",
            "training_source_revision",
            "core_plan_sha256",
            "training_barrier_record_sha256",
            "recovery_final_record_sha256",
            "evaluation_barrier_record_sha256",
            "report_sha256",
            "report_file_sha256",
            "budget_ledger_sha256",
            "budget_ledger_file_sha256",
            "recovery_envelope_sha256",
            "recovery_envelope_file_sha256",
            "incremental_cash_spent_usd",
            "handoff_sha256",
        },
        "recovery_handoff",
    )
    claimed = selected.pop("handoff_sha256")
    for field in (
        "core_plan_sha256",
        "training_barrier_record_sha256",
        "recovery_final_record_sha256",
        "evaluation_barrier_record_sha256",
        "report_sha256",
        "report_file_sha256",
        "budget_ledger_sha256",
        "budget_ledger_file_sha256",
        "recovery_envelope_sha256",
        "recovery_envelope_file_sha256",
    ):
        _sha256(selected[field], f"recovery_handoff.{field}")
    _git_sha(
        selected["evaluation_source_revision"],
        "recovery_handoff.evaluation_source_revision",
    )
    _git_sha(
        selected["training_source_revision"],
        "recovery_handoff.training_source_revision",
    )
    spent = selected["incremental_cash_spent_usd"]
    if (
        isinstance(spent, bool)
        or not isinstance(spent, (int, float))
        or not math.isfinite(float(spent))
        or not 0 <= float(spent) <= 50.0
    ):
        raise EvaluationRecoveryStateError(
            "recovery handoff cash spend is invalid"
        )
    if (
        selected["schema_version"] != HANDOFF_SCHEMA_VERSION
        or selected["protocol_id"] != PROTOCOL_ID
        or selected["status"] != "recovery_report_sealed"
        or _sha256(claimed, "handoff_sha256") != canonical_sha256(selected)
        or selected["training_source_revision"] != FROZEN_TRAINING_REVISION
    ):
        raise EvaluationRecoveryStateError("recovery handoff is invalid")
    if plan is not None and (
        selected["campaign_id"] != plan.campaign_id
        or selected["plan_sha256"] != plan.plan_sha256
        or selected["evaluation_source_revision"] != plan.project_commit
        or selected["core_plan_sha256"] != plan.parent.core_plan_sha256
        or selected["training_barrier_record_sha256"]
        != plan.parent.training_barrier_record_sha256
    ):
        raise EvaluationRecoveryStateError(
            "recovery handoff differs from its plan"
        )
    return {**selected, "handoff_sha256": claimed}


def seal_recovery_handoff(
    plan: EvaluationRecoveryPlan,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Seal the exact recovery result for a downstream extension handoff."""

    status = evaluation_recovery_status(plan)
    final = status.get("final")
    barrier = status.get("evaluation_barrier")
    if (
        not isinstance(final, Mapping)
        or final.get("status") != "completed"
        or not isinstance(barrier, Mapping)
        or barrier.get("status") != "evaluation_completed"
        or final.get("evaluations") != barrier.get("evaluations")
    ):
        raise EvaluationRecoveryStateError(
            "recovery campaign is not completely sealed"
        )
    report = final.get("report")
    if not isinstance(report, Mapping):
        raise EvaluationRecoveryStateError("recovery final lacks its report")
    paths = {
        "report": (
            Path(str(report.get("report_path"))),
            report.get("report_file_sha256"),
        ),
        "budget ledger": (
            Path(str(report.get("budget_ledger_path"))),
            report.get("budget_ledger_file_sha256"),
        ),
        "recovery envelope": (
            Path(str(report.get("recovery_envelope_path"))),
            report.get("recovery_envelope_file_sha256"),
        ),
    }
    for label, (path, digest) in paths.items():
        if not isinstance(digest, str):
            raise EvaluationRecoveryStateError(f"{label} hash is absent")
        _verify_file(path, _sha256(digest, f"{label} hash"), label)
    try:
        report_value = validate_experiment5_report(
            json.loads(paths["report"][0].read_text(encoding="ascii"))
        )
        ledger_value = validate_experiment5_budget_ledger(
            json.loads(paths["budget ledger"][0].read_text(encoding="ascii"))
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        Experiment5ReportError,
    ) as error:
        raise EvaluationRecoveryStateError(
            "recovery report or ledger failed validation"
        ) from error
    envelope_raw = json.loads(
        paths["recovery envelope"][0].read_text(encoding="ascii")
    )
    envelope = dict(envelope_raw)
    envelope_hash = envelope.pop("envelope_sha256", None)
    if (
        envelope.get("schema_version") != REPORT_ENVELOPE_SCHEMA_VERSION
        or envelope.get("campaign_id") != plan.campaign_id
        or envelope.get("plan_sha256") != plan.plan_sha256
        or envelope_hash != canonical_sha256(envelope)
    ):
        raise EvaluationRecoveryStateError("recovery report envelope changed")
    body = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "recovery_report_sealed",
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.plan_sha256,
        "evaluation_source_revision": plan.project_commit,
        "training_source_revision": plan.parent.training_source_revision,
        "core_plan_sha256": plan.parent.core_plan_sha256,
        "training_barrier_record_sha256": (
            plan.parent.training_barrier_record_sha256
        ),
        "recovery_final_record_sha256": final["record_sha256"],
        "evaluation_barrier_record_sha256": barrier["record_sha256"],
        "report_sha256": report_value["report_sha256"],
        "report_file_sha256": report["report_file_sha256"],
        "budget_ledger_sha256": ledger_value["ledger_sha256"],
        "budget_ledger_file_sha256": report[
            "budget_ledger_file_sha256"
        ],
        "recovery_envelope_sha256": envelope_hash,
        "recovery_envelope_file_sha256": report[
            "recovery_envelope_file_sha256"
        ],
        "incremental_cash_spent_usd": ledger_value[
            "incremental_cash_spent"
        ],
    }
    handoff = {**body, "handoff_sha256": canonical_sha256(body)}
    validate_recovery_handoff(handoff, plan=plan)
    path = (
        plan.handoff_path
        if output is None
        else Path(output).expanduser().resolve()
    )
    if path != plan.handoff_path:
        raise EvaluationRecoveryStateError(
            "handoff output differs from the plan-pinned path"
        )
    _seal_exact_bytes(
        path,
        _canonical_json_bytes(handoff) + b"\n",
        name="Experiment-5 recovery handoff",
    )
    return handoff


__all__ = [
    "CHECKPOINT_POINTER_COUNT",
    "FROZEN_TRAINING_REVISION",
    "HANDOFF_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "EvaluationRecoveryCampaign",
    "EvaluationRecoveryPlan",
    "EvaluationRecoveryPlanError",
    "EvaluationRecoveryStateError",
    "build_evaluation_recovery_plan",
    "evaluation_recovery_status",
    "load_evaluation_recovery_plan",
    "parse_evaluation_recovery_plan",
    "seal_recovery_handoff",
    "validate_recovery_handoff",
    "write_evaluation_recovery_plan",
]
