"""Durable training campaign for the Experiment-5 activation-policy v3 reset.

This module deliberately owns a new commit-scoped namespace.  It launches
only the six fresh ``A512ReLUInclusiveTopK`` trajectories.  Completed v2
Softplus trajectories remain read-only parents for the later evaluation
assembly and are never interpreted as v3 training completions.
"""

from __future__ import annotations

import copy
import hashlib
import math
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
    Experiment5CampaignPlanError,
    RetryPolicy,
    _controller_path,
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
    _sha256,
    _timestamp,
    _write_create_only,
)
from moe_sae.experiment5_plan_builder import _PRESERVED_GLOBAL_ENVIRONMENT
from moe_sae.experiment5_routed_ladder_campaign import (
    RoutedLadderCampaign,
    RoutedLadderCampaignPlanError,
    RoutedLadderCampaignStateError,
    RoutedLadderTrainingTask,
    _attempt_environment,
    _task_environment,
    _verify_checkout,
)
from moe_sae.experiment5_routed_ladder_resources import (
    MAX_RESOURCE_COUNT,
    MIN_RESOURCE_COUNT,
    RoutedLadderResourcePool,
    RoutedLadderResourcePoolError,
    parse_routed_ladder_resource_pool,
)
from moe_sae.experiment5_routed_ladder_v3_config import (
    CONDITIONS,
    PHASE_NAME,
    PROTOCOL_ID,
    SEEDS,
    RoutedLadderV3ConfigError,
    load_routed_ladder_v3_config,
)

PLAN_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-v3-training-plan-v1"
)
EXTENSION_NAMESPACE = "experiment5/extensions/routed-activation-ladder-v3"
PANELS = ("sae", "transcoder")
TRAIN_CONDITION = "A512ReLUInclusiveTopK"
TRAINING_ROLES = frozenset(
    (panel, TRAIN_CONDITION, seed)
    for panel in PANELS
    for seed in SEEDS
)
TRAINING_TASK_COUNT = len(TRAINING_ROLES)
TRAIN_TOKENS = 10_000_000
CONTINUOUS_RUNTIME_SECONDS = 43_200
_CONFIG_NAMES = {
    "sae": "gemma4_experiment5_routed_ladder_sae_v3.yaml",
    "transcoder": "gemma4_experiment5_routed_ladder_transcoder_v3.yaml",
}
_SLUG = "a512-relu-inclusive-topk"


@dataclass(frozen=True, slots=True)
class RoutedLadderV3TrainingPlan:
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
    evaluations: tuple[Any, ...]
    plan_sha256: str
    source: Mapping[str, Any]

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)

    def resource(self, resource_id: str) -> CampaignResource:
        for resource in self.resources:
            if resource.resource_id == resource_id:
                return resource
        raise RoutedLadderCampaignPlanError(
            f"unknown v3 campaign resource {resource_id!r}"
        )


def _eligible_resources(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RoutedLadderCampaignPlanError(f"{name} must be a nonempty list")
    selected = tuple(
        _safe_id(item, f"{name}[{index}]")
        for index, item in enumerate(value)
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
        raise RoutedLadderCampaignPlanError(
            f"{name} is required for transcoder"
        )
    return _remote_path(value, name)


def _panel_gcs(value: Any, panel: str, name: str) -> str | None:
    if panel == "sae":
        if value is not None:
            raise RoutedLadderCampaignPlanError(f"{name} must be null for SAE")
        return None
    if (
        not isinstance(value, str)
        or not value.startswith("gs://")
        or value.endswith("/")
    ):
        raise RoutedLadderCampaignPlanError(f"{name} must be a GCS prefix")
    return value


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
    seed = selected["seed"]
    if (
        panel not in PANELS
        or selected["condition"] != TRAIN_CONDITION
        or seed not in SEEDS
        or selected["phase"] != PHASE_NAME
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} scientific identity changed"
        )
    expected_id = f"train-{panel}-{_SLUG}-seed{seed}"
    expected_run = (
        f"experiment5-activation-policy-v3-{panel}-{_SLUG}-seed{seed}"
    )
    if selected["id"] != expected_id or selected["run_id"] != expected_run:
        raise RoutedLadderCampaignPlanError(f"{name} run identity changed")
    completion = _controller_path(
        selected["completion_path_controller"],
        f"{name}.completion_path_controller",
    )
    if (
        completion.name != "training-result.json"
        or completion.parent.name != expected_run
    ):
        raise RoutedLadderCampaignPlanError(
            f"{name} completion is not bound to its run ID"
        )
    return RoutedLadderTrainingTask(
        task_id=expected_id,
        panel=str(panel),
        condition=TRAIN_CONDITION,
        seed=int(seed),
        eligible_resource_ids=_eligible_resources(
            selected["eligible_resource_ids"],
            f"{name}.eligible_resource_ids",
        ),
        run_id=expected_run,
        config_remote=_remote_path(
            selected["config_remote"], f"{name}.config_remote"
        ),
        config_sha256=_sha256(
            selected["config_sha256"], f"{name}.config_sha256"
        ),
        control_root_remote=_remote_path(
            selected["control_root_remote"],
            f"{name}.control_root_remote",
        ),
        checkpoint_root_gcs=_panel_gcs(
            selected["checkpoint_root_gcs"],
            "transcoder",
            f"{name}.checkpoint_root_gcs",
        )
        or "",
        source_overlay_control_root_remote=_panel_remote(
            selected["source_overlay_control_root_remote"],
            str(panel),
            f"{name}.source_overlay_control_root_remote",
        ),
        source_overlay_object_prefix_gcs=_panel_gcs(
            selected["source_overlay_object_prefix_gcs"],
            str(panel),
            f"{name}.source_overlay_object_prefix_gcs",
        ),
        completion_path_controller=completion,
        environment=_task_environment(
            selected["environment"], f"{name}.environment"
        ),
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
    if policy.hard_limit_usd != 50.0 or policy.contingency_usd != 5.0:
        raise RoutedLadderCampaignPlanError(
            "v3 campaign must preserve the recorded $50/$5 cost policy"
        )
    return policy


def parse_routed_ladder_v3_training_plan(
    value: Any,
) -> RoutedLadderV3TrainingPlan:
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
        },
        "v3_training_plan",
    )
    if (
        selected["schema_version"] != PLAN_SCHEMA_VERSION
        or selected["protocol_id"] != PROTOCOL_ID
    ):
        raise RoutedLadderCampaignPlanError(
            "v3 training-plan identity changed"
        )
    project_commit = _git_sha(
        selected["project_commit"], "project_commit"
    )
    short = project_commit[:12]
    state_root = _controller_path(selected["state_root"], "state_root")
    if tuple(state_root.parts[-5:]) != (
        "experiment5",
        "extensions",
        "routed-activation-ladder-v3",
        "controller",
        short,
    ):
        raise RoutedLadderCampaignPlanError(
            "v3 state root is outside its immutable namespace"
        )
    raw_resources = selected["resources"]
    raw_trajectories = selected["trajectories"]
    if not isinstance(raw_resources, list) or not isinstance(
        raw_trajectories, list
    ):
        raise RoutedLadderCampaignPlanError(
            "v3 resources and trajectories must be lists"
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
        raise RoutedLadderCampaignPlanError(str(error)) from error
    if resources != resource_pool.resources:
        raise RoutedLadderCampaignPlanError(
            "v3 resources differ from the immutable resource pool"
        )
    trajectories = tuple(
        _training_task(row, index)
        for index, row in enumerate(raw_trajectories)
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
        raise RoutedLadderCampaignPlanError("v3 retry backoff decreases")
    environment = _task_environment(
        selected["environment"], "environment"
    )
    plan = RoutedLadderV3TrainingPlan(
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
        cost_policy=_cost_policy(selected["cost_policy"]),
        environment=environment,
        resources=resources,
        resource_pool=resource_pool,
        trajectories=trajectories,
        evaluations=(),
        plan_sha256=canonical_sha256(selected),
        source=copy.deepcopy(selected),
    )
    _validate_plan(plan)
    return plan


def _validate_plan(plan: RoutedLadderV3TrainingPlan) -> None:
    roles = {
        (task.panel, task.condition, task.seed)
        for task in plan.trajectories
    }
    if (
        not MIN_RESOURCE_COUNT <= len(plan.resources) <= MAX_RESOURCE_COUNT
        or len(plan.trajectories) != TRAINING_TASK_COUNT
        or roles != TRAINING_ROLES
        or plan.max_parallel_eval != 1
    ):
        raise RoutedLadderCampaignPlanError(
            "v3 plan must contain the exact six fresh inclusive-ReLU roles"
        )
    owners = {
        task.eligible_resource_ids[0]
        for task in plan.trajectories
        if len(task.eligible_resource_ids) == 1
    }
    if (
        len(owners) != len(plan.trajectories)
        or plan.max_parallel_train != len(owners)
    ):
        raise RoutedLadderCampaignPlanError(
            "every v3 trajectory requires one distinct resource owner"
        )
    resource_ids = {resource.resource_id for resource in plan.resources}
    if owners != resource_ids:
        raise RoutedLadderCampaignPlanError(
            "every admitted v3 resource must own one trajectory"
        )
    namespace = f"/{EXTENSION_NAMESPACE}/{plan.project_commit[:12]}/"
    for task in plan.trajectories:
        owner_id = task.eligible_resource_ids[0]
        if owner_id not in plan.resource_pool.cohort(task.panel, task.seed):
            raise RoutedLadderCampaignPlanError(
                f"{task.task_id} owner is outside its immutable cohort"
            )
        owner = plan.resource_pool.plane(owner_id)
        if (
            namespace not in task.control_root_remote
            or namespace not in task.checkpoint_root_gcs
            or namespace not in str(task.completion_path_controller)
            or not task.checkpoint_root_gcs.startswith(
                owner.bulk_gcs_root + "/"
            )
        ):
            raise RoutedLadderCampaignPlanError(
                f"{task.task_id} escapes the v3 artifact namespace"
            )
        try:
            task.completion_path_controller.relative_to(
                owner.controller_artifact_root
            )
            PurePosixPath(task.control_root_remote).relative_to(
                PurePosixPath(owner.remote_artifact_root)
            )
        except ValueError as error:
            raise RoutedLadderCampaignPlanError(
                f"{task.task_id} crosses its owner artifact plane"
            ) from error


def load_routed_ladder_v3_training_plan(
    path: str | Path,
) -> RoutedLadderV3TrainingPlan:
    selected = Path(path)
    try:
        value = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RoutedLadderCampaignPlanError(
            f"cannot read v3 training plan: {selected}"
        ) from error
    return parse_routed_ladder_v3_training_plan(value)


def write_routed_ladder_v3_training_plan(
    path: str | Path,
    value: Mapping[str, Any],
) -> RoutedLadderV3TrainingPlan:
    plan = parse_routed_ladder_v3_training_plan(value)
    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    if selected.exists():
        raise RoutedLadderCampaignPlanError(
            f"refusing to overwrite v3 training plan: {selected}"
        )
    selected.write_text(
        yaml.safe_dump(dict(value), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return plan


def build_routed_ladder_v3_training_plan(
    source: Mapping[str, Any],
    *,
    project_commit: str,
    resource_pool: Mapping[str, Any],
    already_spent_usd: float,
    campaign_id: str | None = None,
) -> dict[str, Any]:
    """Build the exact six-job launch plan without cloud discovery."""

    if not isinstance(source, Mapping):
        raise RoutedLadderCampaignPlanError("source must be a mapping")
    selected_commit = _git_sha(project_commit, "project_commit")
    try:
        admitted = parse_routed_ladder_resource_pool(
            resource_pool,
            project_commit=selected_commit,
        )
    except RoutedLadderResourcePoolError as error:
        raise RoutedLadderCampaignPlanError(str(error)) from error
    working_directory = _controller_path(
        source.get("working_directory"), "source.working_directory"
    )
    python_executable = _controller_path(
        source.get("python_executable"), "source.python_executable"
    )
    remote_project_root = _remote_path(
        source.get("remote_project_root"), "source.remote_project_root"
    )
    remote_uv = _remote_path(source.get("remote_uv"), "source.remote_uv")
    raw_environment = source.get("environment")
    if not isinstance(raw_environment, Mapping):
        raise RoutedLadderCampaignPlanError(
            "source.environment must be a mapping"
        )
    environment = {
        key: copy.deepcopy(value)
        for key, value in raw_environment.items()
        if key in _PRESERVED_GLOBAL_ENVIRONMENT
    }
    environment = dict(_task_environment(environment, "environment"))
    train_overlay_root = _remote_path(
        raw_environment.get("CONTRIBUTION_SOURCE_OVERLAY_ROOT"),
        "CONTRIBUTION_SOURCE_OVERLAY_ROOT",
    )
    train_overlay_prefix = _panel_gcs(
        raw_environment.get("CONTRIBUTION_SOURCE_OVERLAY_PREFIX"),
        "transcoder",
        "CONTRIBUTION_SOURCE_OVERLAY_PREFIX",
    )
    configs: dict[str, tuple[str, str]] = {}
    for panel in PANELS:
        path = working_directory / "configs" / _CONFIG_NAMES[panel]
        try:
            config = load_routed_ladder_v3_config(
                path,
                expected_panel=panel,
            )
        except RoutedLadderV3ConfigError as error:
            raise RoutedLadderCampaignPlanError(
                f"cannot bind v3 {panel} config: {error}"
            ) from error
        layout = next(
            (
                item
                for item in config.layouts
                if item.condition == TRAIN_CONDITION
            ),
            None,
        )
        if (
            tuple(item.condition for item in config.layouts)
            != CONDITIONS
            or layout is None
            or layout.training_seeds != SEEDS
        ):
            raise RoutedLadderCampaignPlanError(
                f"v3 {panel} config does not authorize all fresh ReLU seeds"
            )
        configs[panel] = (
            str(
                PurePosixPath(remote_project_root)
                / "configs"
                / _CONFIG_NAMES[panel]
            ),
            config.config_hash,
        )
    if len(admitted.resources) != TRAINING_TASK_COUNT:
        raise RoutedLadderCampaignPlanError(
            "v3 launch requires exactly six admitted resources"
        )
    short = selected_commit[:12]
    relative_root = PurePosixPath(EXTENSION_NAMESPACE) / short
    trajectories: list[dict[str, Any]] = []
    used_owners: set[str] = set()
    for panel in PANELS:
        config_remote, config_sha = configs[panel]
        for seed in SEEDS:
            cohort = admitted.cohort(panel, seed)
            available = [
                owner_id for owner_id in cohort if owner_id not in used_owners
            ]
            if not available:
                raise RoutedLadderCampaignPlanError(
                    f"v3 {panel}/seed{seed} has no distinct resource owner"
                )
            owner_id = available[0]
            used_owners.add(owner_id)
            plane = admitted.plane(owner_id)
            control_root = str(
                PurePosixPath(plane.remote_artifact_root)
                / relative_root
                / "runs"
                / panel
                / f"seed{seed}"
            )
            controller_root = (
                plane.controller_artifact_root
                / Path(*relative_root.parts)
                / "runs"
                / panel
                / f"seed{seed}"
            )
            checkpoint_root = (
                f"{plane.bulk_gcs_root}/{relative_root}/checkpoints/"
                f"{panel}/seed{seed}"
            )
            run_id = (
                f"experiment5-activation-policy-v3-{panel}-{_SLUG}-seed{seed}"
            )
            trajectories.append(
                {
                    "id": f"train-{panel}-{_SLUG}-seed{seed}",
                    "panel": panel,
                    "condition": TRAIN_CONDITION,
                    "seed": seed,
                    "eligible_resource_ids": [owner_id],
                    "run_id": run_id,
                    "config_remote": config_remote,
                    "config_sha256": config_sha,
                    "phase": PHASE_NAME,
                    "control_root_remote": control_root,
                    "checkpoint_root_gcs": checkpoint_root,
                    "source_overlay_control_root_remote": (
                        None if panel == "sae" else train_overlay_root
                    ),
                    "source_overlay_object_prefix_gcs": (
                        None if panel == "sae" else train_overlay_prefix
                    ),
                    "completion_path_controller": str(
                        controller_root
                        / "runs"
                        / run_id
                        / "training-result.json"
                    ),
                    "environment": {},
                }
            )
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": (
            campaign_id
            or f"gemma4-experiment5-activation-policy-v3-{short}"
        ),
        "state_root": (
            f"/mnt/moe-sae-control/{EXTENSION_NAMESPACE}/controller/{short}"
        ),
        "working_directory": str(working_directory),
        "python_executable": str(python_executable),
        "remote_project_root": remote_project_root,
        "remote_uv": remote_uv,
        "project_commit": selected_commit,
        "max_parallel_train": TRAINING_TASK_COUNT,
        "max_parallel_eval": 1,
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
    }
    parse_routed_ladder_v3_training_plan(plan)
    return plan


def inspect_v3_training_completion(
    plan: RoutedLadderV3TrainingPlan,
    task: RoutedLadderTrainingTask,
) -> CompletionFingerprint | None:
    path = task.completion_path_controller
    if not path.exists():
        return None
    value, payload = _read_self_hashed_json(
        path,
        hash_field="result_sha256",
    )
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
            f"v3 training completion role changed: {task.task_id}"
        )
    return CompletionFingerprint(
        path=str(path),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        scientific_sha256=str(value["result_sha256"]),
    )


def build_v3_train_command(
    plan: RoutedLadderV3TrainingPlan,
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
        str(
            plan.working_directory
            / "configs"
            / _CONFIG_NAMES[task.panel]
        ),
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


def _preflight(plan: RoutedLadderV3TrainingPlan, *, verify_checkout: bool) -> None:
    if verify_checkout:
        _verify_checkout(plan)  # type: ignore[arg-type]
    for resource in plan.resources:
        for label, evidence in (
            ("controller mapping", resource.controller_mapping),
            ("cache inventory", resource.cache_inventory),
        ):
            try:
                payload = evidence.controller_path.read_bytes()
            except OSError as error:
                raise RoutedLadderCampaignStateError(
                    f"cannot read {resource.resource_id} {label}"
                ) from error
            if hashlib.sha256(payload).hexdigest() != evidence.sha256:
                raise RoutedLadderCampaignStateError(
                    f"{resource.resource_id} {label} changed"
                )
        if resource.mirror_seal is not None:
            try:
                payload = resource.mirror_seal.controller_path.read_bytes()
            except OSError as error:
                raise RoutedLadderCampaignStateError(
                    f"cannot read {resource.resource_id} mirror seal"
                ) from error
            if (
                hashlib.sha256(payload).hexdigest()
                != resource.mirror_seal.sha256
            ):
                raise RoutedLadderCampaignStateError(
                    f"{resource.resource_id} mirror seal changed"
                )
    for panel in PANELS:
        path = plan.working_directory / "configs" / _CONFIG_NAMES[panel]
        try:
            config = load_routed_ladder_v3_config(
                path,
                expected_panel=panel,
            )
        except RoutedLadderV3ConfigError as error:
            raise RoutedLadderCampaignStateError(
                f"v3 {panel} config is invalid: {error}"
            ) from error
        tasks = [task for task in plan.trajectories if task.panel == panel]
        if any(task.config_sha256 != config.config_hash for task in tasks):
            raise RoutedLadderCampaignStateError(
                f"v3 {panel} task config identity changed"
            )


class RoutedLadderV3TrainingCampaign(RoutedLadderCampaign):
    """Run only the six new v3 training trajectories."""

    plan: RoutedLadderV3TrainingPlan

    def _baseline_attempt_reserve_usd(self) -> float:
        return (
            len(self.plan.trajectories)
            * self.plan.cost_policy.training_attempt_reserve_usd
        )

    def _reserve_attempt_cost(
        self,
        task: RoutedLadderTrainingTask,
        *,
        kind: str,
        attempt_number: int,
        resource: CampaignResource,
    ) -> dict[str, Any]:
        """Record the inherited estimate without mislabelling it as v2."""

        path = (
            self.plan.state_root
            / "budget"
            / "attempt-reservations"
            / kind
            / task.task_id
            / f"attempt-{attempt_number:06d}.json"
        )
        enforcement = "measurement_only_grant_funded_v3"
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
                    or observed.get("enforcement") != enforcement
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
                    "enforcement": enforcement,
                    "reserved_at": _timestamp(),
                },
            )
            _write_create_only(path, reservation)
            return reservation

    def _preflight(self) -> None:
        _preflight(self.plan, verify_checkout=self.verify_checkout)

    def _inspector(
        self,
        task: RoutedLadderTrainingTask,
    ) -> CompletionFingerprint | None:
        return inspect_v3_training_completion(self.plan, task)

    def _attempt_environment(
        self,
        resource: CampaignResource,
        task: RoutedLadderTrainingTask,
        operator_log_root: Path,
    ) -> tuple[dict[str, str], dict[str, str]]:
        return _attempt_environment(
            self.plan,  # type: ignore[arg-type]
            resource,
            task,
            operator_log_root,
        )

    def _attempt_command(
        self,
        task: RoutedLadderTrainingTask,
        *,
        attempt_id: str,
    ) -> list[str]:
        return build_v3_train_command(self.plan, task, attempt_id)

    def _attempt_id(self, attempt_number: int) -> str:
        parent = super()._attempt_id(attempt_number)
        return parent.replace("e5rl-", "e5rlv3-", 1)

    def run(self) -> dict[str, Any]:
        self._preflight()
        self._acquire_lock()
        try:
            self._bind_plan()
            self._bind_campaign_clock()
            self._start_watchdog()
            completions = self._run_phase(
                self.plan.trajectories,
                kind="training",
                max_parallel=self.plan.max_parallel_train,
            )
            return self._seal_phase(
                path=self.plan.state_root / "training-complete.json",
                record_type="training_phase_completion",
                status="training_completed",
                field="training",
                completions=completions,
            )
        finally:
            self._stop_watchdog()
            self._release_lock()


def routed_ladder_v3_training_status(
    plan: RoutedLadderV3TrainingPlan,
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task in plan.trajectories:
        completion = inspect_v3_training_completion(plan, task)
        root = plan.state_root / "training" / task.task_id
        attempts_root = root / "attempts"
        attempts = (
            len(
                [
                    row
                    for row in attempts_root.iterdir()
                    if row.is_dir()
                ]
            )
            if attempts_root.is_dir()
            else 0
        )
        state = "completed" if completion is not None else "pending"
        if completion is None and (root / "failure.json").exists():
            failure = _read_record(
                root / "failure.json",
                record_type="task_failure",
            )
            state = f"failed:{failure.get('classification')}"
        elif completion is None and attempts:
            latest = max(attempts_root.iterdir())
            state = (
                "retryable"
                if (latest / "result.json").exists()
                else "ambiguous_inflight"
            )
        tasks[task.task_id] = {
            "state": state,
            "attempts": attempts,
            "eligible_resources": list(task.eligible_resource_ids),
            "completion": completion.to_dict() if completion else None,
        }
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.plan_sha256,
        "tasks": tasks,
        "training_barrier": (
            _read_record(
                plan.state_root / "training-complete.json",
                record_type="training_phase_completion",
            )
            if (plan.state_root / "training-complete.json").exists()
            else None
        ),
    }


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "RoutedLadderV3TrainingCampaign",
    "RoutedLadderV3TrainingPlan",
    "TRAINING_TASK_COUNT",
    "build_routed_ladder_v3_training_plan",
    "build_v3_train_command",
    "inspect_v3_training_completion",
    "load_routed_ladder_v3_training_plan",
    "parse_routed_ladder_v3_training_plan",
    "routed_ladder_v3_training_status",
    "write_routed_ladder_v3_training_plan",
]
