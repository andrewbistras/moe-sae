"""Durable five-resource execution for the component-closure campaign.

The public plan stays a plain, self-hashed mapping so it can be inspected and
used by the evaluation builder.  ``parse_training_campaign`` binds that mapping
to the mature Experiment-5 controller: one persistent worker per exact TPU,
append-only attempt state, readiness checks, retries, cost reservations, and a
campaign-wide wall clock. A resource may execute multiple trajectories
sequentially; the two SAE v5 slices together own all nine SAE trajectories.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from moe_sae.component_closure_config import (
    PANELS,
    PHASE_NAME,
    PROTOCOL_ID,
    SEEDS,
    TRAINED_CONDITIONS,
    TRAIN_TOKENS,
    ComponentClosureConfigError,
    load_component_closure_config,
)
from moe_sae.component_closure_references import validate_a_reference_manifest
from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_campaign import (
    CampaignResource,
    CompletionFingerprint,
    CostPolicy,
    Experiment5Campaign,
    Experiment5CampaignStateError,
    RetryPolicy,
    _read_self_hashed_json,
    _resource,
)
from moe_sae.experiment5_routed_ladder_campaign import (
    RoutedLadderTrainingTask,
    _attempt_environment as _routed_attempt_environment,
)

PLAN_SCHEMA_VERSION = "gemma4-a512-component-closure-training-plan-v2"
TASK_COUNT = 18
RESOURCE_COUNT = 5
MAXIMUM_WALL_SECONDS = 43_200
HARD_LIMIT_USD = 40.0
CONTINGENCY_USD = 4.0
TRAINING_RESULT_SCHEMA_VERSION = "gemma4-contribution-sae-training-result-v1"
RUN_CONTRACT_SCHEMA_VERSION = "gemma4-contribution-sae-operator-run-v1"

_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_CONFIG_NAMES = {
    "sae": "gemma4_component_closure_sae_v1.yaml",
    "transcoder": "gemma4_component_closure_transcoder_v1.yaml",
}
_SCHEDULING_CONDITIONS = ("F", "R", "B")
_CONDITION_PRIORITY = {
    condition: index for index, condition in enumerate(_SCHEDULING_CONDITIONS)
}
_COHORT_KEYS = tuple(f"{panel}.seed{seed}" for panel in PANELS for seed in SEEDS)
_REQUIRED_RESOURCE_ENVIRONMENT = frozenset(
    {
        "CONTRIBUTION_QUALIFICATION_BUNDLE",
        "CONTRIBUTION_TRAIN_CAPTURE_ROOT",
        "CONTRIBUTION_OBJECT_PREFIX",
        "CONTRIBUTION_RECOMPOSITION_SIDECAR",
        "CONTRIBUTION_SAE_CACHE_ROOT",
        "CONTRIBUTION_SAE_CACHE_MAX_BYTES",
    }
)
_RESERVED_PLAN_ENVIRONMENT = frozenset(
    {
        "GCP_PROJECT",
        "TPU_NAME",
        "TPU_ZONE",
        "PROJECT_COMMIT",
        "TPU_PROJECT_ROOT",
        "REMOTE_UV",
        "TPU_ACCELERATOR_TYPE",
        "TPU_USE_INTERNAL_IP",
        "TPU_SSH_USER",
        "EXPERIMENT5_PANEL",
        "CONTRIBUTION_FRESH_CONTROLLER_MAPPING",
        "CONTRIBUTION_LOCAL_CACHE_INVENTORY",
        "CONTRIBUTION_REGIONAL_MIRROR_SEAL",
        "CONTRIBUTION_SAE_CONTROL_ROOT",
        "CONTRIBUTION_SAE_CHECKPOINT_ROOT",
        "CONTRIBUTION_SAE_LOCAL_LOG_ROOT",
        "CONTRIBUTION_SOURCE_OVERLAY_ROOT",
        "CONTRIBUTION_SOURCE_OVERLAY_PREFIX",
    }
)


class ComponentClosureCampaignError(ValueError):
    """The closure launch plan is incomplete or changes scientific identity."""


class ComponentClosureCampaignStateError(Experiment5CampaignStateError):
    """Durable closure controller state or completion evidence is invalid."""


@dataclass(frozen=True, slots=True)
class ComponentClosureArtifactPlane:
    resource_id: str
    controller_artifact_root: Path
    remote_artifact_root: str
    bulk_gcs_root: str
    source_overlay_control_root_remote: str | None
    source_overlay_object_prefix_gcs: str | None


@dataclass(frozen=True, slots=True)
class ComponentClosureTrainingPlan:
    campaign_id: str
    state_root: Path
    working_directory: Path
    python_executable: Path
    remote_project_root: str
    remote_uv: str
    project_commit: str
    maximum_wall_seconds: int
    max_parallel_train: int
    max_parallel_eval: int
    retry_policy: RetryPolicy
    cost_policy: CostPolicy
    environment: tuple[tuple[str, str], ...]
    resources: tuple[CampaignResource, ...]
    planes: tuple[ComponentClosureArtifactPlane, ...]
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
        raise ComponentClosureCampaignError(f"unknown resource {resource_id!r}")

    def plane(self, resource_id: str) -> ComponentClosureArtifactPlane:
        for plane in self.planes:
            if plane.resource_id == resource_id:
                return plane
        raise ComponentClosureCampaignError(f"unknown artifact plane {resource_id!r}")


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ComponentClosureCampaignError(
            f"{name} fields differ from the strict schema: {observed}"
        )
    return dict(value)


def _absolute(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ComponentClosureCampaignError(f"{name} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or value == "/" or ".." in path.parts:
        raise ComponentClosureCampaignError(f"{name} must be an absolute path")
    return str(path)


def _remote(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ComponentClosureCampaignError(f"{name} must be an absolute path")
    path = PurePosixPath(value)
    if not path.is_absolute() or value == "/" or ".." in path.parts:
        raise ComponentClosureCampaignError(f"{name} must be an absolute path")
    return str(path)


def _gcs(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"gs://[a-z0-9][a-z0-9._-]*(?:/[^/\s].*)?",
            value,
        )
        is None
        or value.endswith("/")
    ):
        raise ComponentClosureCampaignError(f"{name} must be a GCS prefix")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ComponentClosureCampaignError(f"{name} must be a SHA-256")
    return value


def _safe(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SAFE_RE.fullmatch(value) is None:
        raise ComponentClosureCampaignError(f"{name} is not a safe identifier")
    return value


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ComponentClosureCampaignError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ComponentClosureCampaignError(f"{name} exceeds {maximum}")
    return value


def _money(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComponentClosureCampaignError(f"{name} must be numeric")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0:
        raise ComponentClosureCampaignError(f"{name} must be nonnegative")
    return selected


def _environment(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise ComponentClosureCampaignError(f"{name} must be a mapping")
    rows: list[tuple[str, str]] = []
    for key, raw in value.items():
        if (
            not isinstance(key, str)
            or _ENV_RE.fullmatch(key) is None
            or key in _RESERVED_PLAN_ENVIRONMENT
        ):
            raise ComponentClosureCampaignError(
                f"{name} has controller-owned or invalid variable {key!r}"
            )
        if not isinstance(raw, str) or any(char in raw for char in "\x00\n\r"):
            raise ComponentClosureCampaignError(
                f"{name}.{key} must be a single-line string"
            )
        rows.append((key, raw))
    return tuple(sorted(rows))


def _plane(
    value: Any,
    *,
    resource: CampaignResource,
) -> ComponentClosureArtifactPlane:
    name = f"planes.{resource.resource_id}"
    row = _exact(
        value,
        {
            "controller_artifact_root",
            "remote_artifact_root",
            "bulk_gcs_root",
            "source_overlay_control_root_remote",
            "source_overlay_object_prefix_gcs",
        },
        name,
    )
    controller_root = Path(
        _absolute(row["controller_artifact_root"], f"{name}.controller_artifact_root")
    )
    remote_root = _remote(row["remote_artifact_root"], f"{name}.remote_artifact_root")
    bulk_root = _gcs(row["bulk_gcs_root"], f"{name}.bulk_gcs_root")
    source_root_raw = row["source_overlay_control_root_remote"]
    source_prefix_raw = row["source_overlay_object_prefix_gcs"]
    if (source_root_raw is None) != (source_prefix_raw is None):
        raise ComponentClosureCampaignError(
            f"{name} source overlay root/prefix must be both present or both null"
        )
    source_root = (
        None
        if source_root_raw is None
        else _remote(source_root_raw, f"{name}.source_overlay_control_root_remote")
    )
    source_prefix = (
        None
        if source_prefix_raw is None
        else _gcs(source_prefix_raw, f"{name}.source_overlay_object_prefix_gcs")
    )
    for label, evidence in (
        ("controller mapping", resource.controller_mapping),
        ("cache inventory", resource.cache_inventory),
        ("mirror seal", resource.mirror_seal),
    ):
        if evidence is None:
            continue
        try:
            evidence.controller_path.relative_to(controller_root)
            PurePosixPath(evidence.remote_path).relative_to(PurePosixPath(remote_root))
        except ValueError as error:
            raise ComponentClosureCampaignError(
                f"{resource.resource_id} {label} crosses its artifact plane"
            ) from error
    return ComponentClosureArtifactPlane(
        resource_id=resource.resource_id,
        controller_artifact_root=controller_root,
        remote_artifact_root=remote_root,
        bulk_gcs_root=bulk_root,
        source_overlay_control_root_remote=source_root,
        source_overlay_object_prefix_gcs=source_prefix,
    )


def _parse_cost_policy(value: Any) -> CostPolicy:
    row = _exact(
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
        hard_limit_usd=_money(row["hard_limit_usd"], "cost_policy.hard_limit_usd"),
        contingency_usd=_money(row["contingency_usd"], "cost_policy.contingency_usd"),
        already_spent_usd=_money(
            row["already_spent_usd"], "cost_policy.already_spent_usd"
        ),
        fixed_cost_reserve_usd=_money(
            row["fixed_cost_reserve_usd"], "cost_policy.fixed_cost_reserve_usd"
        ),
        training_attempt_reserve_usd=_money(
            row["training_attempt_reserve_usd"],
            "cost_policy.training_attempt_reserve_usd",
        ),
        evaluation_attempt_reserve_usd=_money(
            row["evaluation_attempt_reserve_usd"],
            "cost_policy.evaluation_attempt_reserve_usd",
        ),
    )
    if (
        policy.hard_limit_usd != HARD_LIMIT_USD
        or policy.contingency_usd != CONTINGENCY_USD
        or policy.initial_launch_limit_usd <= 0
        or policy.already_spent_usd
        + policy.fixed_cost_reserve_usd
        + TASK_COUNT * policy.training_attempt_reserve_usd
        > policy.initial_launch_limit_usd
    ):
        raise ComponentClosureCampaignError(
            "cost policy must preserve the $40 hard ceiling and $4 contingency"
        )
    return policy


def _parse_retry_policy(value: Any) -> RetryPolicy:
    row = _exact(
        value,
        {
            "max_transient_attempts",
            "initial_backoff_seconds",
            "maximum_backoff_seconds",
        },
        "retry_policy",
    )
    policy = RetryPolicy(
        max_transient_attempts=_positive_int(
            row["max_transient_attempts"],
            "retry_policy.max_transient_attempts",
            maximum=20,
        ),
        initial_backoff_seconds=_money(
            row["initial_backoff_seconds"],
            "retry_policy.initial_backoff_seconds",
        ),
        maximum_backoff_seconds=_money(
            row["maximum_backoff_seconds"],
            "retry_policy.maximum_backoff_seconds",
        ),
    )
    if policy.maximum_backoff_seconds < policy.initial_backoff_seconds:
        raise ComponentClosureCampaignError("retry backoff decreases")
    return policy


def _task_from_row(row: Mapping[str, Any], index: int) -> RoutedLadderTrainingTask:
    name = f"tasks[{index}]"
    selected = _exact(
        row,
        {
            "task_id",
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
    if panel not in PANELS or condition not in TRAINED_CONDITIONS or seed not in SEEDS:
        raise ComponentClosureCampaignError(f"{name} scientific role changed")
    slug = f"{condition.lower()}-seed{seed}"
    expected_task = f"train-{panel}-p{_CONDITION_PRIORITY[condition]}-{slug}"
    expected_run = f"component-closure-{panel}-{slug}"
    owners = selected["eligible_resource_ids"]
    expected_owner_count = 2 if panel == "sae" else 1
    if (
        selected["task_id"] != expected_task
        or selected["run_id"] != expected_run
        or selected["phase"] != PHASE_NAME
        or not isinstance(owners, list)
        or len(owners) != expected_owner_count
        or len({str(owner) for owner in owners}) != expected_owner_count
        or any(_SAFE_RE.fullmatch(str(owner)) is None for owner in owners)
    ):
        raise ComponentClosureCampaignError(f"{name} execution identity changed")
    source_root = selected["source_overlay_control_root_remote"]
    source_prefix = selected["source_overlay_object_prefix_gcs"]
    if panel == "sae":
        if source_root is not None or source_prefix is not None:
            raise ComponentClosureCampaignError(f"{name} SAE overlay must be null")
    else:
        source_root = _remote(source_root, f"{name}.source_overlay_control_root_remote")
        source_prefix = _gcs(source_prefix, f"{name}.source_overlay_object_prefix_gcs")
    return RoutedLadderTrainingTask(
        task_id=expected_task,
        panel=panel,
        condition=condition,
        seed=seed,
        eligible_resource_ids=tuple(str(owner) for owner in owners),
        run_id=expected_run,
        config_remote=_remote(selected["config_remote"], f"{name}.config_remote"),
        config_sha256=_sha(selected["config_sha256"], f"{name}.config_sha256"),
        control_root_remote=_remote(
            selected["control_root_remote"], f"{name}.control_root_remote"
        ),
        checkpoint_root_gcs=_gcs(
            selected["checkpoint_root_gcs"], f"{name}.checkpoint_root_gcs"
        ),
        source_overlay_control_root_remote=source_root,
        source_overlay_object_prefix_gcs=source_prefix,
        completion_path_controller=Path(
            _absolute(
                selected["completion_path_controller"],
                f"{name}.completion_path_controller",
            )
        ),
        environment=_environment(selected["environment"], f"{name}.environment"),
    )


def _validate_relations(
    plan: ComponentClosureTrainingPlan,
    *,
    configs: Mapping[str, Any],
    cohorts: Mapping[str, Any],
) -> None:
    if (
        len(plan.resources) != RESOURCE_COUNT
        or len(plan.trajectories) != TASK_COUNT
        or plan.maximum_wall_seconds != MAXIMUM_WALL_SECONDS
        or plan.max_parallel_train != RESOURCE_COUNT
        or plan.max_parallel_eval != 1
    ):
        raise ComponentClosureCampaignError(
            "closure requires 18 tasks, five resource workers, and a 12h wall"
        )
    resource_ids = {resource.resource_id for resource in plan.resources}
    if (
        len(resource_ids) != RESOURCE_COUNT
        or {plane.resource_id for plane in plan.planes} != resource_ids
    ):
        raise ComponentClosureCampaignError("resource identities or planes changed")
    if set(cohorts) != set(_COHORT_KEYS):
        raise ComponentClosureCampaignError("cohorts must cover six panel/seed roles")
    cohorts_by_key: dict[str, tuple[str, ...]] = {}
    for key in _COHORT_KEYS:
        raw = cohorts[key]
        panel = key.split(".", 1)[0]
        expected_count = 2 if panel == "sae" else 1
        if (
            not isinstance(raw, list)
            or len(raw) != expected_count
            or len({str(item) for item in raw}) != expected_count
            or any(str(item) not in resource_ids for item in raw)
        ):
            raise ComponentClosureCampaignError(
                f"cohorts.{key} has the wrong compatible resource set"
            )
        cohorts_by_key[key] = tuple(str(item) for item in raw)
    sae_owners = {item for seed in SEEDS for item in cohorts_by_key[f"sae.seed{seed}"]}
    transcoder_owners = {cohorts_by_key[f"transcoder.seed{seed}"][0] for seed in SEEDS}
    if (
        len(sae_owners) != 2
        or len(transcoder_owners) != 3
        or len(sae_owners | transcoder_owners) != RESOURCE_COUNT
    ):
        raise ComponentClosureCampaignError(
            "ownership must be two SAE v5s plus distinct transcoder v5/v4/v4 owners"
        )
    by_id = {resource.resource_id: resource for resource in plan.resources}
    for owner in sae_owners:
        if by_id[owner].accelerator_type != "v5litepod-16":
            raise ComponentClosureCampaignError("SAE seeds require v5litepod-16")
    if by_id[cohorts_by_key["transcoder.seed0"][0]].accelerator_type != "v5litepod-16":
        raise ComponentClosureCampaignError("transcoder seed0 requires v5litepod-16")
    if any(
        by_id[cohorts_by_key[f"transcoder.seed{seed}"][0]].accelerator_type != "v4-32"
        for seed in (1, 2)
    ):
        raise ComponentClosureCampaignError("transcoder seeds1-2 require v4-32")
    planes = {plane.resource_id: plane for plane in plan.planes}
    sae_plane_keys = {
        (
            str(planes[owner].controller_artifact_root),
            planes[owner].remote_artifact_root,
            planes[owner].bulk_gcs_root,
            planes[owner].source_overlay_control_root_remote,
            planes[owner].source_overlay_object_prefix_gcs,
        )
        for owner in sae_owners
    }
    sae_regions = {by_id[owner].region for owner in sae_owners}
    sae_mirror_hashes = {
        by_id[owner].mirror_seal.sha256
        for owner in sae_owners
        if by_id[owner].mirror_seal is not None
    }
    if (
        len(sae_plane_keys) != 1
        or sae_regions != {"us-central1"}
        or len(sae_mirror_hashes) != 1
    ):
        raise ComponentClosureCampaignError(
            "the two SAE v5 workers must share one central1 artifact/mirror plane"
        )
    roles = []
    for task in plan.trajectories:
        roles.append((task.panel, task.condition, task.seed))
        compatible_owners = cohorts_by_key[f"{task.panel}.seed{task.seed}"]
        if tuple(task.eligible_resource_ids) != compatible_owners:
            raise ComponentClosureCampaignError(
                f"{task.task_id} eligible resource set changed"
            )
        plane = planes[task.eligible_resource_ids[0]]
        if (
            task.config_sha256 != configs[task.panel]["config_sha256"]
            or task.config_remote != configs[task.panel]["remote_path"]
        ):
            raise ComponentClosureCampaignError(
                f"{task.task_id} owner or config binding changed"
            )
        try:
            task.completion_path_controller.relative_to(plane.controller_artifact_root)
            PurePosixPath(task.control_root_remote).relative_to(
                PurePosixPath(plane.remote_artifact_root)
            )
        except ValueError as error:
            raise ComponentClosureCampaignError(
                f"{task.task_id} crosses its owner artifact plane"
            ) from error
        if not task.checkpoint_root_gcs.startswith(plane.bulk_gcs_root + "/"):
            raise ComponentClosureCampaignError(
                f"{task.task_id} crosses its owner GCS plane"
            )
        if task.panel == "transcoder" and (
            task.source_overlay_control_root_remote
            != plane.source_overlay_control_root_remote
            or task.source_overlay_object_prefix_gcs
            != plane.source_overlay_object_prefix_gcs
        ):
            raise ComponentClosureCampaignError(
                f"{task.task_id} transcoder overlay differs from its owner plane"
            )
    expected_roles = [
        (panel, condition, seed)
        for panel in PANELS
        for condition in TRAINED_CONDITIONS
        for seed in SEEDS
    ]
    if roles != expected_roles:
        raise ComponentClosureCampaignError("task matrix or ordering changed")


def parse_training_campaign(value: Any) -> ComponentClosureTrainingPlan:
    selected = _exact(
        value,
        {
            "schema_version",
            "protocol_id",
            "campaign_id",
            "project_commit",
            "working_directory",
            "python_executable",
            "remote_project_root",
            "remote_uv",
            "state_root",
            "maximum_wall_seconds",
            "max_parallel_train",
            "retry_policy",
            "cost_policy",
            "environment",
            "a_reference_manifest_sha256",
            "configs",
            "resources",
            "planes",
            "cohorts",
            "tasks",
            "plan_sha256",
        },
        "training plan",
    )
    claimed = _sha(selected["plan_sha256"], "plan_sha256")
    unsigned = {key: item for key, item in selected.items() if key != "plan_sha256"}
    if (
        selected["schema_version"] != PLAN_SCHEMA_VERSION
        or selected["protocol_id"] != PROTOCOL_ID
        or _GIT_RE.fullmatch(str(selected["project_commit"])) is None
        or canonical_sha256(unsigned) != claimed
    ):
        raise ComponentClosureCampaignError("training plan identity changed")
    configs = selected["configs"]
    if not isinstance(configs, Mapping) or set(configs) != set(PANELS):
        raise ComponentClosureCampaignError("plan requires both panel configs")
    for panel in PANELS:
        row = _exact(
            configs[panel],
            {"controller_path", "remote_path", "config_sha256"},
            f"configs.{panel}",
        )
        _absolute(row["controller_path"], f"configs.{panel}.controller_path")
        _remote(row["remote_path"], f"configs.{panel}.remote_path")
        _sha(row["config_sha256"], f"configs.{panel}.config_sha256")
    raw_resources = selected["resources"]
    if not isinstance(raw_resources, list) or len(raw_resources) != RESOURCE_COUNT:
        raise ComponentClosureCampaignError("plan requires exactly five resources")
    try:
        resources = tuple(
            _resource(row, index, str(selected["project_commit"]))
            for index, row in enumerate(raw_resources)
        )
    except Exception as error:
        raise ComponentClosureCampaignError(str(error)) from error
    for resource in resources:
        resource_environment = resource.environment_dict()
        missing = _REQUIRED_RESOURCE_ENVIRONMENT - set(resource_environment)
        if missing:
            raise ComponentClosureCampaignError(
                f"{resource.resource_id} lacks required train environment: "
                f"{sorted(missing)}"
            )
        if resource_environment["CONTRIBUTION_SAE_CACHE_MAX_BYTES"] != ("171798691840"):
            raise ComponentClosureCampaignError(
                f"{resource.resource_id} cache budget must be exactly 160 GiB"
            )
        if resource_environment["CONTRIBUTION_SAE_CACHE_ROOT"] != (
            "/dev/shm/moe-sae-local/horizon-10m/cache"
        ):
            raise ComponentClosureCampaignError(
                f"{resource.resource_id} must use the qualified host-private cache"
            )
    raw_planes = selected["planes"]
    if not isinstance(raw_planes, Mapping) or set(raw_planes) != {
        item.resource_id for item in resources
    }:
        raise ComponentClosureCampaignError(
            "planes must cover exactly the admitted resources"
        )
    planes = tuple(
        _plane(raw_planes[item.resource_id], resource=item) for item in resources
    )
    raw_tasks = selected["tasks"]
    if not isinstance(raw_tasks, list) or len(raw_tasks) != TASK_COUNT:
        raise ComponentClosureCampaignError("plan requires exactly 18 tasks")
    tasks = tuple(_task_from_row(row, index) for index, row in enumerate(raw_tasks))
    retry = _parse_retry_policy(selected["retry_policy"])
    maximum_wall = _positive_int(
        selected["maximum_wall_seconds"],
        "maximum_wall_seconds",
        maximum=MAXIMUM_WALL_SECONDS,
    )
    plan = ComponentClosureTrainingPlan(
        campaign_id=_safe(selected["campaign_id"], "campaign_id"),
        state_root=Path(_absolute(selected["state_root"], "state_root")),
        working_directory=Path(
            _absolute(selected["working_directory"], "working_directory")
        ),
        python_executable=Path(
            _absolute(selected["python_executable"], "python_executable")
        ),
        remote_project_root=_remote(
            selected["remote_project_root"], "remote_project_root"
        ),
        remote_uv=_remote(selected["remote_uv"], "remote_uv"),
        project_commit=str(selected["project_commit"]),
        maximum_wall_seconds=maximum_wall,
        max_parallel_train=_positive_int(
            selected["max_parallel_train"],
            "max_parallel_train",
            maximum=RESOURCE_COUNT,
        ),
        max_parallel_eval=1,
        retry_policy=retry,
        cost_policy=_parse_cost_policy(selected["cost_policy"]),
        environment=_environment(selected["environment"], "environment"),
        resources=resources,
        planes=planes,
        trajectories=tasks,
        evaluations=(),
        plan_sha256=claimed,
        source=copy.deepcopy(selected),
    )
    _validate_relations(
        plan,
        configs=configs,
        cohorts=selected["cohorts"],
    )
    return plan


def validate_training_plan(value: Any) -> dict[str, Any]:
    """Validate a public plan and return an independent plain mapping."""

    plan = parse_training_campaign(value)
    return copy.deepcopy(dict(plan.source))


def _default_retry_policy(source: Mapping[str, Any]) -> dict[str, Any]:
    raw = source.get("retry_policy")
    if raw is None:
        return {
            "max_transient_attempts": 4,
            "initial_backoff_seconds": 30.0,
            "maximum_backoff_seconds": 300.0,
        }
    if not isinstance(raw, Mapping):
        raise ComponentClosureCampaignError("retry_policy must be a mapping")
    return copy.deepcopy(dict(raw))


def _default_cost_policy(source: Mapping[str, Any]) -> dict[str, Any]:
    raw = source.get("cost_policy")
    if raw is None:
        return {
            "hard_limit_usd": HARD_LIMIT_USD,
            "contingency_usd": CONTINGENCY_USD,
            "already_spent_usd": 0.0,
            "fixed_cost_reserve_usd": 15.0,
            "training_attempt_reserve_usd": 0.5,
            "evaluation_attempt_reserve_usd": 0.0,
        }
    if not isinstance(raw, Mapping):
        raise ComponentClosureCampaignError("cost_policy must be a mapping")
    return copy.deepcopy(dict(raw))


def build_training_plan(
    source: Mapping[str, Any],
    *,
    project_commit: str,
    a_reference_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact launch plan locally; no cloud discovery occurs."""

    if _GIT_RE.fullmatch(project_commit) is None:
        raise ComponentClosureCampaignError("project_commit must be a full Git SHA")
    references = validate_a_reference_manifest(a_reference_manifest)
    working = Path(_absolute(source.get("working_directory"), "working_directory"))
    python = _absolute(source.get("python_executable"), "python_executable")
    remote_root = _remote(source.get("remote_project_root"), "remote_project_root")
    remote_uv = _remote(source.get("remote_uv"), "remote_uv")
    state_root = _absolute(source.get("state_root"), "state_root")
    raw_resources = source.get("resources")
    raw_planes = source.get("planes")
    raw_cohorts = source.get("cohorts")
    if (
        not isinstance(raw_resources, list)
        or not isinstance(raw_planes, Mapping)
        or not isinstance(raw_cohorts, Mapping)
    ):
        raise ComponentClosureCampaignError(
            "source requires resources, planes, and cohorts"
        )
    try:
        resources = tuple(
            _resource(row, index, project_commit)
            for index, row in enumerate(raw_resources)
        )
    except Exception as error:
        raise ComponentClosureCampaignError(str(error)) from error
    if len(resources) != RESOURCE_COUNT:
        raise ComponentClosureCampaignError("source requires exactly five resources")
    if set(raw_planes) != {item.resource_id for item in resources}:
        raise ComponentClosureCampaignError("source planes do not match resources")
    planes = {
        item.resource_id: _plane(raw_planes[item.resource_id], resource=item)
        for item in resources
    }
    configs: dict[str, Any] = {}
    for panel in PANELS:
        controller = working / "configs" / _CONFIG_NAMES[panel]
        try:
            config = load_component_closure_config(
                controller,
                expected_panel=panel,  # type: ignore[arg-type]
            )
        except ComponentClosureConfigError as error:
            raise ComponentClosureCampaignError(
                f"{panel} closure config is invalid: {error}"
            ) from error
        configs[panel] = {
            "controller_path": str(controller),
            "remote_path": str(
                PurePosixPath(remote_root) / "configs" / _CONFIG_NAMES[panel]
            ),
            "config_sha256": config.config_hash,
        }
    cohorts = copy.deepcopy(dict(raw_cohorts))
    tasks: list[dict[str, Any]] = []
    short = project_commit[:12]
    for panel in PANELS:
        for condition in TRAINED_CONDITIONS:
            for seed in SEEDS:
                cohort_key = f"{panel}.seed{seed}"
                compatible = cohorts.get(cohort_key)
                expected_count = 2 if panel == "sae" else 1
                if (
                    not isinstance(compatible, list)
                    or len(compatible) != expected_count
                ):
                    raise ComponentClosureCampaignError(
                        f"cohorts.{cohort_key} has the wrong resource count"
                    )
                eligible = [str(owner) for owner in compatible]
                if any(owner not in planes for owner in eligible):
                    raise ComponentClosureCampaignError(
                        f"cohorts.{cohort_key} names an unknown resource"
                    )
                plane = planes[eligible[0]]
                slug = f"{condition.lower()}-seed{seed}"
                run_id = f"component-closure-{panel}-{slug}"
                namespace = PurePosixPath(
                    "component-closure",
                    short,
                    panel,
                    f"seed{seed}",
                    condition.lower(),
                )
                control = str(PurePosixPath(plane.remote_artifact_root) / namespace)
                checkpoint = f"{plane.bulk_gcs_root}/{namespace}"
                completion = (
                    plane.controller_artifact_root
                    / Path(*namespace.parts)
                    / "runs"
                    / run_id
                    / "training-result.json"
                )
                tasks.append(
                    {
                        "task_id": (
                            f"train-{panel}-"
                            f"p{_CONDITION_PRIORITY[condition]}-{slug}"
                        ),
                        "panel": panel,
                        "condition": condition,
                        "seed": seed,
                        "eligible_resource_ids": eligible,
                        "run_id": run_id,
                        "config_remote": configs[panel]["remote_path"],
                        "config_sha256": configs[panel]["config_sha256"],
                        "phase": PHASE_NAME,
                        "control_root_remote": control,
                        "checkpoint_root_gcs": checkpoint,
                        "source_overlay_control_root_remote": (
                            plane.source_overlay_control_root_remote
                            if panel == "transcoder"
                            else None
                        ),
                        "source_overlay_object_prefix_gcs": (
                            plane.source_overlay_object_prefix_gcs
                            if panel == "transcoder"
                            else None
                        ),
                        "completion_path_controller": str(completion),
                        "environment": {},
                    }
                )
    environment = source.get("environment", {})
    if not isinstance(environment, Mapping):
        raise ComponentClosureCampaignError("environment must be a mapping")
    body = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "campaign_id": str(source.get("campaign_id", f"component-closure-{short}")),
        "project_commit": project_commit,
        "working_directory": str(working),
        "python_executable": python,
        "remote_project_root": remote_root,
        "remote_uv": remote_uv,
        "state_root": state_root,
        "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
        "max_parallel_train": int(source.get("max_parallel_train", RESOURCE_COUNT)),
        "retry_policy": _default_retry_policy(source),
        "cost_policy": _default_cost_policy(source),
        "environment": copy.deepcopy(dict(environment)),
        "a_reference_manifest_sha256": references["manifest_sha256"],
        "configs": configs,
        "resources": copy.deepcopy(raw_resources),
        "planes": copy.deepcopy(dict(raw_planes)),
        "cohorts": cohorts,
        "tasks": tasks,
    }
    return validate_training_plan({**body, "plan_sha256": canonical_sha256(body)})


def build_train_command(
    plan: ComponentClosureTrainingPlan,
    task: RoutedLadderTrainingTask,
    attempt_id: str,
) -> list[str]:
    if not isinstance(attempt_id, str) or _SAFE_RE.fullmatch(attempt_id) is None:
        raise ComponentClosureCampaignError("attempt_id is not safe")
    return [
        str(plan.python_executable),
        str(plan.working_directory / "scripts/tpu/train_component_closure.py"),
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
        str(MAXIMUM_WALL_SECONDS),
    ]


def _read_contract(path: Path) -> tuple[dict[str, Any], bytes]:
    value, payload = _read_self_hashed_json(
        path,
        hash_field="run_contract_sha256",
    )
    return value, payload


def inspect_training_completion(
    plan: ComponentClosureTrainingPlan,
    task: RoutedLadderTrainingTask,
) -> CompletionFingerprint | None:
    result_path = task.completion_path_controller
    contract_path = result_path.parent / "run-contract.json"
    if not result_path.exists() and not contract_path.exists():
        return None
    if not result_path.exists() or not contract_path.exists():
        raise ComponentClosureCampaignStateError(
            f"training completion is partial: {task.task_id}"
        )
    try:
        result, payload = _read_self_hashed_json(
            result_path,
            hash_field="result_sha256",
        )
        contract, _contract_payload = _read_contract(contract_path)
    except Experiment5CampaignStateError as error:
        raise ComponentClosureCampaignStateError(str(error)) from error
    identity = result.get("run_identity")
    identity_plan = identity.get("plan") if isinstance(identity, Mapping) else None
    if (
        result.get("schema_version") != TRAINING_RESULT_SCHEMA_VERSION
        or result.get("status") not in {"completed", "already_complete"}
        or result.get("tokens") != TRAIN_TOKENS
        or not isinstance(identity_plan, Mapping)
        or identity_plan.get("phase_name") != PHASE_NAME
        or identity_plan.get("condition") != task.condition
        or identity_plan.get("seed") != task.seed
        or identity_plan.get("total_tokens") != TRAIN_TOKENS
        or contract.get("schema_version") != RUN_CONTRACT_SCHEMA_VERSION
        or contract.get("run_id") != task.run_id
        or contract.get("project_commit") != plan.project_commit
        or contract.get("phase") != PHASE_NAME
        or contract.get("condition") != task.condition
        or contract.get("seed") != task.seed
        or contract.get("checkpoint_payload_root") != task.checkpoint_root_gcs
    ):
        raise ComponentClosureCampaignStateError(
            f"training completion role changed: {task.task_id}"
        )
    return CompletionFingerprint(
        path=str(result_path),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        scientific_sha256=str(result["result_sha256"]),
    )


def _verify_file(path: Path, expected_sha256: str, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ComponentClosureCampaignStateError(
            f"{name} is not a regular file: {path}"
        )
    try:
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ComponentClosureCampaignStateError(f"cannot read {name}") from error
    if observed != expected_sha256:
        raise ComponentClosureCampaignStateError(f"{name} SHA-256 changed")


def _preflight(
    plan: ComponentClosureTrainingPlan,
    *,
    verify_checkout: bool,
) -> None:
    if verify_checkout:
        if (
            plan.working_directory.is_symlink()
            or not plan.working_directory.is_dir()
            or plan.python_executable.is_symlink()
            or not plan.python_executable.is_file()
            or not os.access(plan.python_executable, os.X_OK)
        ):
            raise ComponentClosureCampaignStateError(
                "controller checkout or python executable is invalid"
            )
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=plan.working_directory,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                cwd=plan.working_directory,
                check=True,
                text=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise ComponentClosureCampaignStateError(
                "cannot verify controller checkout"
            ) from error
        if head != plan.project_commit or dirty:
            raise ComponentClosureCampaignStateError(
                "controller checkout must be clean at project_commit"
            )
    trainer = plan.working_directory / "scripts/tpu/train_component_closure.py"
    if trainer.is_symlink() or not trainer.is_file():
        raise ComponentClosureCampaignStateError("closure trainer script is absent")
    for resource in plan.resources:
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
    for panel in PANELS:
        path = plan.working_directory / "configs" / _CONFIG_NAMES[panel]
        try:
            config = load_component_closure_config(
                path,
                expected_panel=panel,  # type: ignore[arg-type]
            )
        except ComponentClosureConfigError as error:
            raise ComponentClosureCampaignStateError(
                f"{panel} closure config is invalid: {error}"
            ) from error
        hashes = {
            task.config_sha256 for task in plan.trajectories if task.panel == panel
        }
        if hashes != {config.config_hash}:
            raise ComponentClosureCampaignStateError(
                f"{panel} task config hash differs from checkout"
            )


class ComponentClosureTrainingCampaign(Experiment5Campaign):
    """Execute the 18 fresh trajectories over five durable resource workers."""

    plan: ComponentClosureTrainingPlan

    def _baseline_attempt_reserve_usd(self) -> float:
        return (
            len(self.plan.trajectories)
            * self.plan.cost_policy.training_attempt_reserve_usd
        )

    def _maximum_campaign_wall_seconds(self) -> float:
        return float(self.plan.maximum_wall_seconds)

    def _preflight(self) -> None:
        _preflight(self.plan, verify_checkout=self.verify_checkout)

    def _inspector(
        self,
        task: RoutedLadderTrainingTask,
    ) -> CompletionFingerprint | None:
        return inspect_training_completion(self.plan, task)

    def _attempt_environment(
        self,
        resource: CampaignResource,
        task: RoutedLadderTrainingTask,
        operator_log_root: Path,
    ) -> tuple[dict[str, str], dict[str, str]]:
        return _routed_attempt_environment(
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
        return build_train_command(self.plan, task, attempt_id)

    def _attempt_id(self, attempt_number: int) -> str:
        parent = super()._attempt_id(attempt_number)
        return parent.replace("e5-", "closure-", 1)

    def _prioritized_trajectories(
        self,
    ) -> tuple[RoutedLadderTrainingTask, ...]:
        return tuple(
            sorted(
                self.plan.trajectories,
                key=lambda task: (
                    _CONDITION_PRIORITY[task.condition],
                    PANELS.index(task.panel),
                    task.seed,
                ),
            )
        )

    def run(self) -> dict[str, Any]:
        self._preflight()
        self._acquire_lock()
        try:
            self._bind_plan()
            self._bind_campaign_clock()
            self._start_watchdog()
            completions = self._run_phase(
                self._prioritized_trajectories(),
                kind="training",
                max_parallel=self.plan.max_parallel_train,
            )
            barrier = self._seal_phase(
                path=self.plan.state_root / "training-complete.json",
                record_type="training_phase_completion",
                status="training_completed",
                field="training",
                completions=completions,
            )
            self._seal_cost_projection(completed_at=barrier["completed_at"])
            return barrier
        finally:
            self._stop_watchdog()
            self._release_lock()


def status(plan: Mapping[str, Any]) -> dict[str, Any]:
    parsed = parse_training_campaign(plan)
    rows = []
    for task in parsed.trajectories:
        try:
            completion = inspect_training_completion(parsed, task)
        except ComponentClosureCampaignStateError:
            state = "invalid"
        else:
            state = "completed" if completion is not None else "pending"
        attempts_root = parsed.state_root / "training" / task.task_id / "attempts"
        attempts = (
            len([item for item in attempts_root.iterdir() if item.is_dir()])
            if attempts_root.is_dir()
            else 0
        )
        rows.append(
            {
                "task_id": task.task_id,
                "eligible_resource_ids": list(task.eligible_resource_ids),
                "state": state,
                "attempts": attempts,
            }
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "plan_sha256": parsed.plan_sha256,
        "hard_limit_usd": parsed.cost_policy.hard_limit_usd,
        "maximum_wall_seconds": parsed.maximum_wall_seconds,
        "tasks": rows,
    }


def load_training_plan(path: str | Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ComponentClosureCampaignError("cannot read closure plan") from error
    return validate_training_plan(value)


def load_training_campaign(path: str | Path) -> ComponentClosureTrainingPlan:
    return parse_training_campaign(load_training_plan(path))


__all__ = [
    "CONTINGENCY_USD",
    "HARD_LIMIT_USD",
    "MAXIMUM_WALL_SECONDS",
    "PLAN_SCHEMA_VERSION",
    "RESOURCE_COUNT",
    "TASK_COUNT",
    "ComponentClosureArtifactPlane",
    "ComponentClosureCampaignError",
    "ComponentClosureCampaignStateError",
    "ComponentClosureTrainingCampaign",
    "ComponentClosureTrainingPlan",
    "build_train_command",
    "build_training_plan",
    "inspect_training_completion",
    "load_training_campaign",
    "load_training_plan",
    "parse_training_campaign",
    "status",
    "validate_training_plan",
]
