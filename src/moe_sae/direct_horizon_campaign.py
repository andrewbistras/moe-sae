"""Direct controller for the fixed Gemma 10M learning-horizon campaign.

This module is intentionally smaller than :mod:`moe_sae.horizon_supervisor`.
It does not allocate, delete, or discover cloud resources.  A strict plan
binds the eight scientific trajectories to eight already-existing TPU slices
and binds the six physical joint evaluations to selected slices.

The controller provides only the missing durable control-plane semantics:

* exactly SAE/transcoder x A/R/B/F training trajectories;
* one exclusive trajectory owner per preallocated TPU;
* one continuous 43,200-second direct operator invocation per attempt;
* bounded retries for observed transport/resource transients;
* fail-closed handling of numerical, integrity, and ambiguous failures;
* exact ``training-result.json`` completion recognition; and
* exactly one joint A/R/B/F evaluation per model at 2M, 5M, and 10M.

An intent without a terminal attempt record is deliberately not retried.  The
remote command may still be alive after a controller crash, so automatically
starting another owner would be less safe than requiring reconciliation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.horizon_supervisor import classify_failure

PLAN_SCHEMA_VERSION = "gemma4-horizon-direct-campaign-plan-v1"
RECORD_SCHEMA_VERSION = "gemma4-horizon-direct-controller-record-v1"
TRAINING_RESULT_SCHEMA_VERSION = "gemma4-contribution-sae-training-result-v1"
SAE_EVALUATION_SCHEMA_VERSION = "gemma4-contribution-learning-horizon-curve-point-v1"
TRANSCODER_EVALUATION_SCHEMA_VERSION = (
    "gemma4-contribution-transcoder-learning-horizon-result-v1"
)
PROTOCOL_ID = "gemma4-pilot-learning-horizon-v1-20260728"
PHASE_NAME = "learning_horizon_10m"
MODELS = ("sae", "transcoder")
CONDITIONS = ("A", "R", "B", "F")
MILESTONES = (2_000_000, 5_000_000, 10_000_000)
TRAIN_TOKENS = 10_000_000
CONTINUOUS_RUNTIME_SECONDS = 43_200
SUPPORTED_ACCELERATOR_TYPES = ("v4-32", "v6e-16", "v5litepod-16")
TRANSIENT_FAILURES = frozenset({"ssh_transport", "resource_unavailable"})
RESUMABLE_FAILURES = TRANSIENT_FAILURES | {"controller_stop"}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_ZONE = re.compile(r"^[a-z][a-z0-9-]*-[a-z]$")
_GCS_PREFIX = re.compile(r"^gs://[a-z0-9][a-z0-9._-]*/[^/\s].*$")

_RESERVED_ENVIRONMENT = frozenset(
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
        "CONTRIBUTION_FRESH_CONTROLLER_MAPPING",
        "CONTRIBUTION_LOCAL_CACHE_INVENTORY",
        "CONTRIBUTION_QUALIFICATION_BUNDLE",
        "CONTRIBUTION_REGIONAL_MIRROR_SEAL",
        "CONTRIBUTION_SAE_CONTROL_ROOT",
        "CONTRIBUTION_SAE_CHECKPOINT_ROOT",
        "CONTRIBUTION_SAE_LOCAL_LOG_ROOT",
        "CONTRIBUTION_EVALUATION_LOG_ROOT",
        "TRANSCODER_EVALUATION_LOG_ROOT",
    }
)


class DirectCampaignPlanError(RuntimeError):
    """A direct campaign plan is incomplete or unsafe."""


class DirectCampaignStateError(RuntimeError):
    """Durable controller state is corrupt, ambiguous, or terminal."""


@dataclass(frozen=True, slots=True)
class EvidencePath:
    """One evidence file as seen by the controller and by TPU workers."""

    controller_path: Path
    remote_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """One already-existing, never-created-by-this-controller TPU slice."""

    resource_id: str
    project: str
    tpu_name: str
    zone: str
    region: str
    accelerator_type: str
    internal_ip: bool
    ssh_user: str | None
    controller_mapping: EvidencePath
    cache_inventory: EvidencePath
    mirror_seal: EvidencePath | None
    environment: tuple[tuple[str, str], ...]

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)


@dataclass(frozen=True, slots=True)
class TrajectorySpec:
    """One model/condition training trajectory."""

    task_id: str
    scientific_model: str
    condition: str
    seed: int
    resource_id: str
    run_id: str
    config_remote: str
    phase: str
    control_root_remote: str
    checkpoint_root_gcs: str
    completion_path_controller: Path
    environment: tuple[tuple[str, str], ...]

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """One physical joint A/R/B/F evaluation."""

    task_id: str
    scientific_model: str
    milestone_tokens: int
    seed: int
    resource_id: str
    config_remote: str
    phase: str
    control_root_remote: str
    checkpoint_root_gcs: str
    result_root_remote: str
    completion_path_controller: Path
    run_roots_remote: tuple[tuple[str, str], ...]
    environment: tuple[tuple[str, str], ...]

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)

    def run_roots_dict(self) -> dict[str, str]:
        return dict(self.run_roots_remote)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_transient_attempts: int
    initial_backoff_seconds: float
    maximum_backoff_seconds: float


@dataclass(frozen=True, slots=True)
class DirectCampaignPlan:
    """Fully validated immutable direct-campaign plan."""

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
    environment: tuple[tuple[str, str], ...]
    resources: tuple[ResourceSpec, ...]
    trajectories: tuple[TrajectorySpec, ...]
    evaluations: tuple[EvaluationSpec, ...]
    plan_sha256: str
    source: Mapping[str, Any]

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)

    def resource(self, resource_id: str) -> ResourceSpec:
        for resource in self.resources:
            if resource.resource_id == resource_id:
                return resource
        raise DirectCampaignPlanError(f"unknown resource {resource_id!r}")


@dataclass(frozen=True, slots=True)
class CompletionFingerprint:
    path: str
    size_bytes: int
    sha256: str
    scientific_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "scientific_sha256": self.scientific_sha256,
        }


@dataclass(frozen=True, slots=True)
class AttemptExecution:
    """Result returned by an injected or real process runner."""

    returncode: int | None
    timed_out: bool = False
    detail: str = ""


AttemptRunner = Callable[
    [
        str,
        Sequence[str],
        Mapping[str, str],
        Path,
        Path,
        threading.Event,
    ],
    AttemptExecution,
]


def _exact_mapping(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise DirectCampaignPlanError(
            f"{name} fields differ from the strict schema: {observed}"
        )
    if not all(isinstance(key, str) for key in value):
        raise DirectCampaignPlanError(f"{name} keys must be strings")
    return dict(value)


def _safe_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise DirectCampaignPlanError(f"{name} is not a safe identifier")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DirectCampaignPlanError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DirectCampaignPlanError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise DirectCampaignPlanError(f"{name} exceeds {maximum}")
    return value


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DirectCampaignPlanError(f"{name} must be nonnegative")
    selected = float(value)
    if not selected >= 0.0 or selected == float("inf"):
        raise DirectCampaignPlanError(f"{name} must be finite and nonnegative")
    return selected


def _controller_path(value: Any, name: str) -> Path:
    if not isinstance(value, str):
        raise DirectCampaignPlanError(f"{name} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or value == "/" or ".." in path.parts:
        raise DirectCampaignPlanError(f"{name} must be a non-root absolute path")
    return path


def _remote_path(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise DirectCampaignPlanError(f"{name} must be an absolute POSIX path")
    path = PurePosixPath(value)
    if not path.is_absolute() or value == "/" or ".." in path.parts:
        raise DirectCampaignPlanError(f"{name} must be a non-root absolute POSIX path")
    return str(path)


def _gcs_prefix(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or _GCS_PREFIX.fullmatch(value) is None
        or value.endswith("/")
    ):
        raise DirectCampaignPlanError(f"{name} must be a non-root GCS prefix")
    return value


def _environment(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise DirectCampaignPlanError(f"{name} must be a mapping")
    result: list[tuple[str, str]] = []
    for key, raw in value.items():
        if not isinstance(key, str) or _ENV_NAME.fullmatch(key) is None:
            raise DirectCampaignPlanError(f"{name} has invalid variable {key!r}")
        if key in _RESERVED_ENVIRONMENT:
            raise DirectCampaignPlanError(
                f"{name} may not override controller-owned variable {key}"
            )
        if not isinstance(raw, str) or "\x00" in raw or "\n" in raw or "\r" in raw:
            raise DirectCampaignPlanError(
                f"{name}.{key} must be one single-line string"
            )
        result.append((key, raw))
    return tuple(sorted(result))


def _evidence(value: Any, name: str) -> EvidencePath:
    selected = _exact_mapping(
        value,
        {"controller_path", "remote_path", "sha256"},
        name,
    )
    return EvidencePath(
        controller_path=_controller_path(
            selected["controller_path"], f"{name}.controller_path"
        ),
        remote_path=_remote_path(selected["remote_path"], f"{name}.remote_path"),
        sha256=_sha256(selected["sha256"], f"{name}.sha256"),
    )


def _resource(value: Any, index: int) -> ResourceSpec:
    name = f"resources[{index}]"
    selected = _exact_mapping(
        value,
        {
            "id",
            "project",
            "tpu_name",
            "zone",
            "region",
            "accelerator_type",
            "internal_ip",
            "ssh_user",
            "evidence",
            "environment",
        },
        name,
    )
    zone = _safe_id(selected["zone"], f"{name}.zone")
    region = _safe_id(selected["region"], f"{name}.region")
    if _ZONE.fullmatch(zone) is None or not zone.startswith(region + "-"):
        raise DirectCampaignPlanError(f"{name} zone does not belong to region")
    accelerator = selected["accelerator_type"]
    if accelerator not in SUPPORTED_ACCELERATOR_TYPES:
        raise DirectCampaignPlanError(f"{name} accelerator type is unsupported")
    internal_ip = selected["internal_ip"]
    if type(internal_ip) is not bool:
        raise DirectCampaignPlanError(f"{name}.internal_ip must be boolean")
    if accelerator in {"v6e-16", "v5litepod-16"} and not internal_ip:
        raise DirectCampaignPlanError(f"{name} regional TPU requires internal IP")
    ssh_user = selected["ssh_user"]
    if ssh_user is not None:
        ssh_user = _safe_id(ssh_user, f"{name}.ssh_user")
        if ssh_user.lower() != ssh_user:
            raise DirectCampaignPlanError(f"{name}.ssh_user must be lowercase")
    evidence = _exact_mapping(
        selected["evidence"],
        {"controller_mapping", "cache_inventory", "mirror_seal"},
        f"{name}.evidence",
    )
    environment = _environment(selected["environment"], f"{name}.environment")
    environment_dict = dict(environment)
    if accelerator == "v5litepod-16":
        if not environment_dict.get("MOE_SAE_V5LITEPOD_DEVICE_KIND"):
            raise DirectCampaignPlanError(
                f"{name} v5litepod resource lacks observed device kind"
            )
    elif "MOE_SAE_V5LITEPOD_DEVICE_KIND" in environment_dict:
        raise DirectCampaignPlanError(
            f"{name} non-v5 resource declares a v5 device kind"
        )
    mirror_raw = evidence["mirror_seal"]
    if accelerator == "v4-32":
        mirror_seal = (
            None
            if mirror_raw is None
            else _evidence(mirror_raw, f"{name}.evidence.mirror_seal")
        )
    else:
        if mirror_raw is None:
            raise DirectCampaignPlanError(
                f"{name} regional TPU requires mirror-seal evidence"
            )
        mirror_seal = _evidence(mirror_raw, f"{name}.evidence.mirror_seal")
    return ResourceSpec(
        resource_id=_safe_id(selected["id"], f"{name}.id"),
        project=_safe_id(selected["project"], f"{name}.project"),
        tpu_name=_safe_id(selected["tpu_name"], f"{name}.tpu_name"),
        zone=zone,
        region=region,
        accelerator_type=accelerator,
        internal_ip=internal_ip,
        ssh_user=ssh_user,
        controller_mapping=_evidence(
            evidence["controller_mapping"],
            f"{name}.evidence.controller_mapping",
        ),
        cache_inventory=_evidence(
            evidence["cache_inventory"],
            f"{name}.evidence.cache_inventory",
        ),
        mirror_seal=mirror_seal,
        environment=environment,
    )


def _trajectory(value: Any, index: int) -> TrajectorySpec:
    name = f"trajectories[{index}]"
    selected = _exact_mapping(
        value,
        {
            "id",
            "scientific_model",
            "condition",
            "seed",
            "resource_id",
            "run_id",
            "config_remote",
            "phase",
            "control_root_remote",
            "checkpoint_root_gcs",
            "completion_path_controller",
            "environment",
        },
        name,
    )
    model = selected["scientific_model"]
    condition = selected["condition"]
    if model not in MODELS or condition not in CONDITIONS:
        raise DirectCampaignPlanError(f"{name} has an unsupported scientific role")
    expected_id = f"train-{model}-{condition.lower()}"
    if selected["id"] != expected_id:
        raise DirectCampaignPlanError(f"{name}.id must be {expected_id!r}")
    if selected["seed"] != 0:
        raise DirectCampaignPlanError(f"{name}.seed must be zero")
    expected_run_id = f"horizon10m-{model}-{condition.lower()}-seed0"
    if selected["run_id"] != expected_run_id:
        raise DirectCampaignPlanError(f"{name}.run_id must be {expected_run_id!r}")
    if selected["phase"] != PHASE_NAME:
        raise DirectCampaignPlanError(f"{name}.phase must be {PHASE_NAME!r}")
    completion = _controller_path(
        selected["completion_path_controller"],
        f"{name}.completion_path_controller",
    )
    if completion.name != "training-result.json":
        raise DirectCampaignPlanError(f"{name} completion must be training-result.json")
    if completion.parent.name != expected_run_id:
        raise DirectCampaignPlanError(
            f"{name} completion parent must be the immutable run ID"
        )
    return TrajectorySpec(
        task_id=expected_id,
        scientific_model=model,
        condition=condition,
        seed=0,
        resource_id=_safe_id(selected["resource_id"], f"{name}.resource_id"),
        run_id=expected_run_id,
        config_remote=_remote_path(selected["config_remote"], f"{name}.config_remote"),
        phase=PHASE_NAME,
        control_root_remote=_remote_path(
            selected["control_root_remote"], f"{name}.control_root_remote"
        ),
        checkpoint_root_gcs=_gcs_prefix(
            selected["checkpoint_root_gcs"], f"{name}.checkpoint_root_gcs"
        ),
        completion_path_controller=completion,
        environment=_environment(selected["environment"], f"{name}.environment"),
    )


def _run_roots(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    selected = _exact_mapping(value, set(CONDITIONS), name)
    result = tuple(
        (
            condition,
            _remote_path(selected[condition], f"{name}.{condition}"),
        )
        for condition in CONDITIONS
    )
    return result


def _evaluation(value: Any, index: int) -> EvaluationSpec:
    name = f"evaluations[{index}]"
    selected = _exact_mapping(
        value,
        {
            "id",
            "scientific_model",
            "milestone_tokens",
            "seed",
            "resource_id",
            "config_remote",
            "phase",
            "control_root_remote",
            "checkpoint_root_gcs",
            "result_root_remote",
            "completion_path_controller",
            "run_roots_remote",
            "environment",
        },
        name,
    )
    model = selected["scientific_model"]
    milestone = selected["milestone_tokens"]
    if model not in MODELS or milestone not in MILESTONES:
        raise DirectCampaignPlanError(f"{name} has an unsupported evaluation role")
    expected_id = f"eval-{model}-{milestone}"
    if selected["id"] != expected_id:
        raise DirectCampaignPlanError(f"{name}.id must be {expected_id!r}")
    if selected["seed"] != 0:
        raise DirectCampaignPlanError(f"{name}.seed must be zero")
    if selected["phase"] != PHASE_NAME:
        raise DirectCampaignPlanError(f"{name}.phase must be {PHASE_NAME!r}")
    completion = _controller_path(
        selected["completion_path_controller"],
        f"{name}.completion_path_controller",
    )
    if completion.name != "scientific-result.json":
        raise DirectCampaignPlanError(
            f"{name} completion must be scientific-result.json"
        )
    if completion.parent.name != str(milestone):
        raise DirectCampaignPlanError(
            f"{name} completion parent must be the milestone token count"
        )
    return EvaluationSpec(
        task_id=expected_id,
        scientific_model=model,
        milestone_tokens=milestone,
        seed=0,
        resource_id=_safe_id(selected["resource_id"], f"{name}.resource_id"),
        config_remote=_remote_path(selected["config_remote"], f"{name}.config_remote"),
        phase=PHASE_NAME,
        control_root_remote=_remote_path(
            selected["control_root_remote"], f"{name}.control_root_remote"
        ),
        checkpoint_root_gcs=_gcs_prefix(
            selected["checkpoint_root_gcs"], f"{name}.checkpoint_root_gcs"
        ),
        result_root_remote=_remote_path(
            selected["result_root_remote"], f"{name}.result_root_remote"
        ),
        completion_path_controller=completion,
        run_roots_remote=_run_roots(
            selected["run_roots_remote"], f"{name}.run_roots_remote"
        ),
        environment=_environment(selected["environment"], f"{name}.environment"),
    )


def parse_direct_campaign_plan(value: Any) -> DirectCampaignPlan:
    """Validate and freeze one strict direct-campaign plan."""

    selected = _exact_mapping(
        value,
        {
            "schema_version",
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
            "environment",
            "resources",
            "trajectories",
            "evaluations",
        },
        "plan",
    )
    if selected["schema_version"] != PLAN_SCHEMA_VERSION:
        raise DirectCampaignPlanError("direct campaign schema version changed")
    project_commit = selected["project_commit"]
    if (
        not isinstance(project_commit, str)
        or _GIT_SHA.fullmatch(project_commit) is None
    ):
        raise DirectCampaignPlanError("project_commit must be a full Git SHA")
    retry = _exact_mapping(
        selected["retry_policy"],
        {
            "max_transient_attempts",
            "initial_backoff_seconds",
            "maximum_backoff_seconds",
        },
        "retry_policy",
    )
    retry_policy = RetryPolicy(
        max_transient_attempts=_positive_int(
            retry["max_transient_attempts"],
            "retry_policy.max_transient_attempts",
            maximum=100,
        ),
        initial_backoff_seconds=_nonnegative_number(
            retry["initial_backoff_seconds"],
            "retry_policy.initial_backoff_seconds",
        ),
        maximum_backoff_seconds=_nonnegative_number(
            retry["maximum_backoff_seconds"],
            "retry_policy.maximum_backoff_seconds",
        ),
    )
    if retry_policy.maximum_backoff_seconds < retry_policy.initial_backoff_seconds:
        raise DirectCampaignPlanError("retry backoff maximum precedes initial")
    raw_resources = selected["resources"]
    raw_trajectories = selected["trajectories"]
    raw_evaluations = selected["evaluations"]
    if not isinstance(raw_resources, list) or len(raw_resources) != 8:
        raise DirectCampaignPlanError("plan must contain exactly eight resources")
    if not isinstance(raw_trajectories, list) or len(raw_trajectories) != 8:
        raise DirectCampaignPlanError("plan must contain exactly eight trajectories")
    if not isinstance(raw_evaluations, list) or len(raw_evaluations) != 6:
        raise DirectCampaignPlanError(
            "plan must contain exactly six physical joint evaluations"
        )
    resources = tuple(
        _resource(item, index) for index, item in enumerate(raw_resources)
    )
    trajectories = tuple(
        _trajectory(item, index) for index, item in enumerate(raw_trajectories)
    )
    evaluations = tuple(
        _evaluation(item, index) for index, item in enumerate(raw_evaluations)
    )
    _validate_plan_relations(resources, trajectories, evaluations)
    return DirectCampaignPlan(
        campaign_id=_safe_id(selected["campaign_id"], "campaign_id"),
        state_root=_controller_path(selected["state_root"], "state_root"),
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
            selected["max_parallel_train"], "max_parallel_train", maximum=8
        ),
        max_parallel_eval=_positive_int(
            selected["max_parallel_eval"], "max_parallel_eval", maximum=6
        ),
        retry_policy=retry_policy,
        environment=_environment(selected["environment"], "environment"),
        resources=resources,
        trajectories=trajectories,
        evaluations=evaluations,
        plan_sha256=canonical_sha256(selected),
        source=selected,
    )


def _validate_plan_relations(
    resources: tuple[ResourceSpec, ...],
    trajectories: tuple[TrajectorySpec, ...],
    evaluations: tuple[EvaluationSpec, ...],
) -> None:
    resource_ids = [item.resource_id for item in resources]
    if len(set(resource_ids)) != 8:
        raise DirectCampaignPlanError("resource IDs must be unique")
    physical = [(item.project, item.zone, item.tpu_name) for item in resources]
    if len(set(physical)) != 8:
        raise DirectCampaignPlanError("every resource must name a unique TPU slice")
    mapping_paths = [str(item.controller_mapping.controller_path) for item in resources]
    cache_paths = [str(item.cache_inventory.controller_path) for item in resources]
    if len(set(mapping_paths)) != 8 or len(set(cache_paths)) != 8:
        raise DirectCampaignPlanError(
            "fresh mapping/cache evidence must be unique per TPU slice"
        )
    expected_train_roles = {
        (model, condition) for model in MODELS for condition in CONDITIONS
    }
    observed_train_roles = {
        (item.scientific_model, item.condition) for item in trajectories
    }
    if observed_train_roles != expected_train_roles:
        raise DirectCampaignPlanError(
            "trajectories must cover SAE/transcoder x A/R/B/F exactly once"
        )
    trajectory_resources = [item.resource_id for item in trajectories]
    if (
        set(trajectory_resources) != set(resource_ids)
        or len(set(trajectory_resources)) != 8
    ):
        raise DirectCampaignPlanError(
            "each preallocated TPU must own exactly one training trajectory"
        )
    expected_eval_roles = {
        (model, milestone) for model in MODELS for milestone in MILESTONES
    }
    observed_eval_roles = {
        (item.scientific_model, item.milestone_tokens) for item in evaluations
    }
    if observed_eval_roles != expected_eval_roles:
        raise DirectCampaignPlanError("evaluations must be exactly model x {2M,5M,10M}")
    if any(item.resource_id not in resource_ids for item in evaluations):
        raise DirectCampaignPlanError("evaluation references an unknown resource")
    completion_paths = [
        str(item.completion_path_controller) for item in (*trajectories, *evaluations)
    ]
    if len(set(completion_paths)) != 14:
        raise DirectCampaignPlanError("completion paths must be unique")
    by_role = {(item.scientific_model, item.condition): item for item in trajectories}
    for model in MODELS:
        checkpoints = {
            by_role[(model, condition)].checkpoint_root_gcs for condition in CONDITIONS
        }
        if len(checkpoints) != 1:
            raise DirectCampaignPlanError(
                f"{model} conditions must share one checkpoint namespace"
            )
        configs = {
            by_role[(model, condition)].config_remote for condition in CONDITIONS
        }
        if len(configs) != 1:
            raise DirectCampaignPlanError(
                f"{model} conditions must share one finalized config"
            )
        expected_runs = {
            condition: by_role[(model, condition)].run_id for condition in CONDITIONS
        }
        for evaluation in (
            item for item in evaluations if item.scientific_model == model
        ):
            if evaluation.checkpoint_root_gcs not in checkpoints:
                raise DirectCampaignPlanError(
                    f"{evaluation.task_id} uses another checkpoint namespace"
                )
            if evaluation.config_remote not in configs:
                raise DirectCampaignPlanError(
                    f"{evaluation.task_id} uses another finalized config"
                )
            for condition, root in evaluation.run_roots_remote:
                if PurePosixPath(root).name != expected_runs[condition]:
                    raise DirectCampaignPlanError(
                        f"{evaluation.task_id} {condition} run root is misbound"
                    )


def load_direct_campaign_plan(path: str | Path) -> DirectCampaignPlan:
    selected = Path(path)
    try:
        decoded = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise DirectCampaignPlanError(
            f"cannot read direct campaign plan {selected}"
        ) from error
    return parse_direct_campaign_plan(decoded)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise DirectCampaignStateError("record is not canonical JSON") from error


def _record(schema_type: str, body: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "record_type": schema_type,
        **dict(body),
    }
    return {**unsigned, "record_sha256": canonical_sha256(unsigned)}


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = _canonical_json_bytes(dict(value)) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise DirectCampaignStateError(
            f"create-only state already exists: {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _read_record(path: Path, *, record_type: str | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DirectCampaignStateError(
            f"cannot read controller state {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise DirectCampaignStateError(f"controller state is not an object: {path}")
    selected = dict(value)
    claimed = selected.pop("record_sha256", None)
    if (
        selected.get("schema_version") != RECORD_SCHEMA_VERSION
        or not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or canonical_sha256(selected) != claimed
        or (record_type is not None and selected.get("record_type") != record_type)
    ):
        raise DirectCampaignStateError(f"controller state identity is invalid: {path}")
    return dict(value)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_self_hashed_json(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise DirectCampaignStateError(f"completion is not a regular file: {path}")
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DirectCampaignStateError(f"completion JSON is invalid: {path}") from error
    if not isinstance(value, Mapping):
        raise DirectCampaignStateError(f"completion is not an object: {path}")
    selected = dict(value)
    claimed = selected.pop("result_sha256", None)
    if (
        not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or canonical_sha256(selected) != claimed
    ):
        raise DirectCampaignStateError(f"completion self-hash is invalid: {path}")
    return dict(value), payload


def inspect_training_completion(
    task: TrajectorySpec,
) -> CompletionFingerprint | None:
    path = task.completion_path_controller
    if not path.exists():
        return None
    value, payload = _read_self_hashed_json(path)
    identity = value.get("run_identity")
    plan = identity.get("plan") if isinstance(identity, Mapping) else None
    if (
        value.get("schema_version") != TRAINING_RESULT_SCHEMA_VERSION
        or value.get("status") not in {"completed", "already_complete"}
        or value.get("tokens") != TRAIN_TOKENS
        or not isinstance(plan, Mapping)
        or plan.get("phase_name") != PHASE_NAME
        or plan.get("condition") != task.condition
        or plan.get("seed") != task.seed
        or plan.get("total_tokens") != TRAIN_TOKENS
    ):
        raise DirectCampaignStateError(
            f"training completion role changed: {task.task_id}"
        )
    return CompletionFingerprint(
        path=str(path),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        scientific_sha256=str(value["result_sha256"]),
    )


def inspect_evaluation_completion(
    task: EvaluationSpec,
) -> CompletionFingerprint | None:
    path = task.completion_path_controller
    if not path.exists():
        return None
    value, payload = _read_self_hashed_json(path)
    expected_schema = (
        SAE_EVALUATION_SCHEMA_VERSION
        if task.scientific_model == "sae"
        else TRANSCODER_EVALUATION_SCHEMA_VERSION
    )
    if task.scientific_model == "sae":
        observed_milestone = value.get("checkpoint_target_tokens")
    else:
        checkpoint = value.get("learning_curve_checkpoint")
        observed_milestone = (
            checkpoint.get("target_tokens") if isinstance(checkpoint, Mapping) else None
        )
    if (
        value.get("schema_version") != expected_schema
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "curve_point_measured"
        or value.get("seed") != task.seed
        or observed_milestone != task.milestone_tokens
    ):
        raise DirectCampaignStateError(
            f"evaluation completion role changed: {task.task_id}"
        )
    return CompletionFingerprint(
        path=str(path),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        scientific_sha256=str(value["result_sha256"]),
    )


def _verify_file(path: Path, expected_sha256: str, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise DirectCampaignStateError(f"{name} is not a regular file: {path}")
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise DirectCampaignStateError(f"cannot read {name}: {path}") from error
    if digest != expected_sha256:
        raise DirectCampaignStateError(f"{name} SHA-256 changed: {path}")


def _verify_checkout(plan: DirectCampaignPlan) -> None:
    if plan.working_directory.is_symlink() or not plan.working_directory.is_dir():
        raise DirectCampaignStateError("working_directory is not a real directory")
    if (
        plan.python_executable.is_symlink()
        or not plan.python_executable.is_file()
        or not os.access(plan.python_executable, os.X_OK)
    ):
        raise DirectCampaignStateError("python_executable is not executable")
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
        raise DirectCampaignStateError("cannot verify controller checkout") from error
    if head != plan.project_commit or status:
        raise DirectCampaignStateError(
            "controller checkout must be clean at project_commit"
        )


def _preflight(plan: DirectCampaignPlan, *, verify_checkout: bool) -> None:
    if verify_checkout:
        _verify_checkout(plan)
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


def build_train_command(
    plan: DirectCampaignPlan,
    task: TrajectorySpec,
    attempt_id: str,
) -> list[str]:
    return [
        str(plan.python_executable),
        str(plan.working_directory / "scripts/tpu/train_contribution.py"),
        "--action",
        "train",
        "--scientific-model",
        task.scientific_model,
        "--config",
        task.config_remote,
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
    plan: DirectCampaignPlan,
    task: EvaluationSpec,
) -> list[str]:
    operator = (
        "evaluate_contribution.py"
        if task.scientific_model == "sae"
        else "evaluate_transcoder.py"
    )
    command = [
        str(plan.python_executable),
        str(plan.working_directory / f"scripts/tpu/{operator}"),
        "--config",
        task.config_remote,
        "--phase",
        PHASE_NAME,
        "--seed",
        str(task.seed),
        "--checkpoint-tokens",
        str(task.milestone_tokens),
        "--result-root",
        task.result_root_remote,
    ]
    for condition, root in task.run_roots_remote:
        command.extend((f"--run-root-{condition.lower()}", root))
    return command


def _task_environment(
    plan: DirectCampaignPlan,
    resource: ResourceSpec,
    task: TrajectorySpec | EvaluationSpec,
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
            "CONTRIBUTION_SAE_CONTROL_ROOT": task.control_root_remote,
            "CONTRIBUTION_SAE_CHECKPOINT_ROOT": task.checkpoint_root_gcs,
        }
    )
    if resource.mirror_seal is not None:
        audited["CONTRIBUTION_REGIONAL_MIRROR_SEAL"] = resource.mirror_seal.remote_path
    if resource.ssh_user is not None:
        audited["TPU_SSH_USER"] = resource.ssh_user
    if isinstance(task, TrajectorySpec):
        audited.update(
            {
                "CONTRIBUTION_FRESH_CONTROLLER_MAPPING": (
                    resource.controller_mapping.remote_path
                ),
                "CONTRIBUTION_LOCAL_CACHE_INVENTORY": (
                    resource.cache_inventory.remote_path
                ),
                "CONTRIBUTION_SAE_LOCAL_LOG_ROOT": str(operator_log_root),
            }
        )
    else:
        if task.scientific_model == "sae":
            # The SAE evaluator parses the mapping on the controller before
            # fanning out its workers.
            mapping_path = str(resource.controller_mapping.controller_path)
            audited["CONTRIBUTION_FRESH_CONTROLLER_MAPPING"] = mapping_path
            audited["CONTRIBUTION_EVALUATION_LOG_ROOT"] = str(operator_log_root)
        else:
            # The transcoder evaluator fetches the mapping through TPU SSH, so
            # it must receive the worker-visible NFS path.
            mapping_path = resource.controller_mapping.remote_path
            audited["CONTRIBUTION_FRESH_CONTROLLER_MAPPING"] = mapping_path
            audited["TRANSCODER_EVALUATION_LOG_ROOT"] = str(operator_log_root)
        if resource.accelerator_type == "v4-32":
            # Legacy v4 evaluator settings still name this permutation a
            # qualification bundle. The fresh mapping has the exact field
            # they consume and remains independently hash-verified.
            audited["CONTRIBUTION_QUALIFICATION_BUNDLE"] = mapping_path
    executable = os.environ.copy()
    executable.update(audited)
    return executable, audited


def _default_attempt_runner(
    _task_id: str,
    command: Sequence[str],
    environment: Mapping[str, str],
    working_directory: Path,
    stdout_path: Path,
    stop_event: threading.Event,
) -> AttemptExecution:
    stdout_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    clean_environment = dict(environment)
    for name in ("LD_PRELOAD", "GLIBCXX_FORCE_NEW", "GLIBCPP_FORCE_NEW"):
        clean_environment.pop(name, None)
    try:
        with stdout_path.open("wb") as output:
            process = subprocess.Popen(
                list(command),
                cwd=working_directory,
                env=clean_environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while True:
                try:
                    return AttemptExecution(process.wait(timeout=1.0))
                except subprocess.TimeoutExpired:
                    if not stop_event.is_set():
                        continue
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        return AttemptExecution(
                            process.wait(timeout=30),
                            detail="controller stop requested",
                        )
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        return AttemptExecution(
                            process.wait(),
                            detail="controller stop forced kill",
                        )
    except OSError as error:
        try:
            with stdout_path.open("ab") as output:
                output.write(f"\ncontroller process error: {error}\n".encode())
        except OSError:
            pass
        return AttemptExecution(None, detail=str(error))


def _log_evidence(attempt_root: Path) -> str:
    chunks: list[str] = []
    for path in sorted(attempt_root.rglob("*.log")):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        chunks.append(payload[-262_144:].decode("utf-8", errors="replace"))
    return "\n".join(chunks)[-1_048_576:]


class DirectCampaign:
    """Durably execute one strict plan against preallocated TPU slices."""

    def __init__(
        self,
        plan: DirectCampaignPlan,
        *,
        stop_event: threading.Event | None = None,
        attempt_runner: AttemptRunner | None = None,
        verify_checkout: bool = True,
    ) -> None:
        self.plan = plan
        self.stop_event = stop_event or threading.Event()
        self.attempt_runner = attempt_runner or _default_attempt_runner
        self.verify_checkout = verify_checkout
        self._lock_handle: Any | None = None
        self._resource_locks = {
            resource.resource_id: threading.Lock() for resource in self.plan.resources
        }

    def _acquire_lock(self) -> None:
        self.plan.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.plan.state_root / "controller.lock"
        handle = path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise DirectCampaignStateError(
                "another direct campaign controller owns the state root"
            ) from error
        self._lock_handle = handle

    def _release_lock(self) -> None:
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None

    def _bind_plan(self) -> None:
        binding_path = self.plan.state_root / "plan-binding.json"
        body = {
            "campaign_id": self.plan.campaign_id,
            "plan_sha256": self.plan.plan_sha256,
            "plan": self.plan.source,
        }
        expected = _record("plan_binding", body)
        if binding_path.exists():
            observed = _read_record(binding_path, record_type="plan_binding")
            if observed != expected:
                raise DirectCampaignStateError(
                    "state root is bound to another direct campaign plan"
                )
        else:
            _write_create_only(binding_path, expected)
        ownership = {
            task.resource_id: {
                "task_id": task.task_id,
                "scientific_model": task.scientific_model,
                "condition": task.condition,
                "run_id": task.run_id,
            }
            for task in self.plan.trajectories
        }
        ownership_path = self.plan.state_root / "resource-ownership.json"
        expected_ownership = _record(
            "resource_ownership",
            {
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "ownership": ownership,
            },
        )
        if ownership_path.exists():
            if (
                _read_record(ownership_path, record_type="resource_ownership")
                != expected_ownership
            ):
                raise DirectCampaignStateError("resource ownership ledger changed")
        else:
            _write_create_only(ownership_path, expected_ownership)

    def _task_root(self, task_id: str, *, kind: str) -> Path:
        return self.plan.state_root / kind / task_id

    def _completion_witness(
        self,
        task_id: str,
        *,
        kind: str,
        fingerprint: CompletionFingerprint,
    ) -> None:
        path = self._task_root(task_id, kind=kind) / "completion.json"
        expected = _record(
            "task_completion",
            {
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "task_id": task_id,
                "kind": kind,
                "completion": fingerprint.to_dict(),
            },
        )
        if path.exists():
            if _read_record(path, record_type="task_completion") != expected:
                raise DirectCampaignStateError(
                    f"completion witness changed for {task_id}"
                )
        else:
            _write_create_only(path, expected)

    def _prior_state(
        self,
        task_id: str,
        *,
        kind: str,
    ) -> tuple[int, int]:
        root = self._task_root(task_id, kind=kind)
        failure = root / "failure.json"
        if failure.exists():
            record = _read_record(failure, record_type="task_failure")
            raise DirectCampaignStateError(
                f"{task_id} has terminal failure {record.get('classification')}"
            )
        attempts_root = root / "attempts"
        maximum = 0
        transient_count = 0
        if not attempts_root.exists():
            return maximum, transient_count
        for attempt in sorted(attempts_root.iterdir()):
            if not attempt.is_dir() or not attempt.name.startswith("attempt-"):
                raise DirectCampaignStateError(
                    f"unexpected attempt state under {attempts_root}"
                )
            try:
                number = int(attempt.name.removeprefix("attempt-"))
            except ValueError as error:
                raise DirectCampaignStateError(
                    f"invalid attempt directory {attempt}"
                ) from error
            maximum = max(maximum, number)
            intent = attempt / "intent.json"
            result = attempt / "result.json"
            if not intent.is_file():
                raise DirectCampaignStateError(
                    f"attempt lacks immutable intent: {attempt}"
                )
            _read_record(intent, record_type="attempt_intent")
            if not result.exists():
                raise DirectCampaignStateError(
                    f"{task_id} has an unobserved attempt {attempt.name}; "
                    "reconcile the remote owner before retrying"
                )
            record = _read_record(result, record_type="attempt_result")
            classification = record.get("classification")
            if classification in RESUMABLE_FAILURES:
                if classification in TRANSIENT_FAILURES:
                    transient_count += 1
            elif classification != "completed":
                raise DirectCampaignStateError(
                    f"{task_id} has a prior terminal attempt "
                    f"{attempt.name}: {classification}"
                )
        return maximum, transient_count

    def _new_attempt(
        self,
        task_id: str,
        *,
        kind: str,
        number: int,
        attempt_id: str,
        command: Sequence[str],
        audited_environment: Mapping[str, str],
        resource: ResourceSpec,
    ) -> tuple[Path, str]:
        root = (
            self._task_root(task_id, kind=kind) / "attempts" / f"attempt-{number:06d}"
        )
        intent = _record(
            "attempt_intent",
            {
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "task_id": task_id,
                "kind": kind,
                "attempt_number": number,
                "attempt_id": attempt_id,
                "resource_id": resource.resource_id,
                "resource_identity": {
                    "project": resource.project,
                    "zone": resource.zone,
                    "tpu_name": resource.tpu_name,
                    "accelerator_type": resource.accelerator_type,
                },
                "command": list(command),
                "command_sha256": canonical_sha256(list(command)),
                "environment_sha256": canonical_sha256(
                    dict(sorted(audited_environment.items()))
                ),
                "started_at": _timestamp(),
            },
        )
        _write_create_only(root / "intent.json", intent)
        return root, attempt_id

    def _terminal_failure(
        self,
        task_id: str,
        *,
        kind: str,
        classification: str,
        attempt_number: int,
        detail: str,
    ) -> None:
        path = self._task_root(task_id, kind=kind) / "failure.json"
        if not path.exists():
            _write_create_only(
                path,
                _record(
                    "task_failure",
                    {
                        "campaign_id": self.plan.campaign_id,
                        "plan_sha256": self.plan.plan_sha256,
                        "task_id": task_id,
                        "kind": kind,
                        "classification": classification,
                        "attempt_number": attempt_number,
                        "detail": detail[-4_096:],
                        "failed_at": _timestamp(),
                    },
                ),
            )

    def _execute_task(
        self,
        task: TrajectorySpec | EvaluationSpec,
        *,
        kind: str,
    ) -> CompletionFingerprint:
        inspector: Callable[
            [TrajectorySpec | EvaluationSpec], CompletionFingerprint | None
        ]
        if isinstance(task, TrajectorySpec):
            inspector = inspect_training_completion  # type: ignore[assignment]
        else:
            inspector = inspect_evaluation_completion  # type: ignore[assignment]
        existing = inspector(task)
        if existing is not None:
            self._completion_witness(task.task_id, kind=kind, fingerprint=existing)
            return existing
        maximum, transient_count = self._prior_state(task.task_id, kind=kind)
        resource = self.plan.resource(task.resource_id)
        while not self.stop_event.is_set():
            number = maximum + 1
            provisional_root = (
                self._task_root(task.task_id, kind=kind)
                / "attempts"
                / f"attempt-{number:06d}"
            )
            operator_log = provisional_root / "operator"
            environment, audited = _task_environment(
                self.plan, resource, task, operator_log
            )
            attempt_id = f"dc-a{number:08d}-{uuid.uuid4().hex[:16]}"
            command = (
                build_train_command(self.plan, task, attempt_id)
                if isinstance(task, TrajectorySpec)
                else build_evaluation_command(self.plan, task)
            )
            attempt_root, attempt_id = self._new_attempt(
                task.task_id,
                kind=kind,
                number=number,
                attempt_id=attempt_id,
                command=command,
                audited_environment=audited,
                resource=resource,
            )
            execution = self.attempt_runner(
                task.task_id,
                command,
                environment,
                self.plan.working_directory,
                attempt_root / "controller.log",
                self.stop_event,
            )
            completion = inspector(task)
            output = _log_evidence(attempt_root)
            classification = classify_failure(
                returncode=execution.returncode,
                output=output + "\n" + execution.detail,
                timed_out=execution.timed_out,
                completion_present=completion is not None,
                checkpoint_before=None,
                checkpoint_after=None,
                segment=None,
            )
            if self.stop_event.is_set() and completion is None:
                classification = "controller_stop"
            result = _record(
                "attempt_result",
                {
                    "campaign_id": self.plan.campaign_id,
                    "plan_sha256": self.plan.plan_sha256,
                    "task_id": task.task_id,
                    "kind": kind,
                    "attempt_number": number,
                    "attempt_id": attempt_id,
                    "returncode": execution.returncode,
                    "timed_out": execution.timed_out,
                    "classification": classification,
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
            if completion is not None:
                self._completion_witness(
                    task.task_id, kind=kind, fingerprint=completion
                )
                return completion
            if classification == "controller_stop":
                break
            maximum = number
            if classification not in TRANSIENT_FAILURES:
                self._terminal_failure(
                    task.task_id,
                    kind=kind,
                    classification=classification,
                    attempt_number=number,
                    detail=output or execution.detail,
                )
                raise DirectCampaignStateError(
                    f"{task.task_id} failed terminally: {classification}"
                )
            transient_count += 1
            if transient_count >= self.plan.retry_policy.max_transient_attempts:
                self._terminal_failure(
                    task.task_id,
                    kind=kind,
                    classification="transient_attempts_exhausted",
                    attempt_number=number,
                    detail=output or execution.detail,
                )
                raise DirectCampaignStateError(
                    f"{task.task_id} exhausted transient retries"
                )
            delay = min(
                self.plan.retry_policy.maximum_backoff_seconds,
                self.plan.retry_policy.initial_backoff_seconds
                * (2 ** max(0, transient_count - 1)),
            )
            if self.stop_event.wait(delay):
                break
        raise DirectCampaignStateError(
            f"{task.task_id} stopped before immutable completion"
        )

    def _run_train(self, task: TrajectorySpec) -> CompletionFingerprint:
        lock = self._resource_locks[task.resource_id]
        with lock:
            return self._execute_task(task, kind="training")

    def _run_evaluation_pipeline(
        self,
        model: str,
    ) -> list[CompletionFingerprint]:
        results: list[CompletionFingerprint] = []
        tasks = sorted(
            (task for task in self.plan.evaluations if task.scientific_model == model),
            key=lambda task: task.milestone_tokens,
        )
        for task in tasks:
            lock = self._resource_locks[task.resource_id]
            with lock:
                results.append(self._execute_task(task, kind="evaluation"))
        return results

    def _seal_training_barrier(
        self,
        training: Mapping[str, CompletionFingerprint],
    ) -> dict[str, Any]:
        path = self.plan.state_root / "training-complete.json"
        expected = _record(
            "training_phase_completion",
            {
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "status": "training_completed",
                "training": {key: training[key].to_dict() for key in sorted(training)},
                "completed_at": _timestamp(),
            },
        )
        if path.exists():
            observed = _read_record(path, record_type="training_phase_completion")
            for key in (
                "campaign_id",
                "plan_sha256",
                "status",
                "training",
            ):
                if observed.get(key) != expected.get(key):
                    raise DirectCampaignStateError("training-complete barrier changed")
            return observed
        _write_create_only(path, expected)
        return expected

    def run(self, *, stop_after_training: bool = False) -> dict[str, Any]:
        _preflight(self.plan, verify_checkout=self.verify_checkout)
        self._acquire_lock()
        try:
            self._bind_plan()
            training: dict[str, CompletionFingerprint] = {}
            failures: list[str] = []
            with ThreadPoolExecutor(
                max_workers=self.plan.max_parallel_train,
                thread_name_prefix="direct-train",
            ) as pool:
                futures = {
                    pool.submit(self._run_train, task): task
                    for task in self.plan.trajectories
                }
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        training[task.task_id] = future.result()
                    except DirectCampaignStateError as error:
                        failures.append(f"{task.task_id}: {error}")
            if failures:
                raise DirectCampaignStateError(
                    "training campaign has terminal failures: "
                    + "; ".join(sorted(failures))
                )
            training_barrier = self._seal_training_barrier(training)
            if stop_after_training:
                return training_barrier
            evaluation: dict[str, CompletionFingerprint] = {}
            eval_failures: list[str] = []
            with ThreadPoolExecutor(
                max_workers=min(self.plan.max_parallel_eval, len(MODELS)),
                thread_name_prefix="direct-eval",
            ) as pool:
                futures = {
                    pool.submit(self._run_evaluation_pipeline, model): model
                    for model in MODELS
                }
                for future in as_completed(futures):
                    model = futures[future]
                    try:
                        for fingerprint in future.result():
                            task = next(
                                item
                                for item in self.plan.evaluations
                                if str(item.completion_path_controller)
                                == fingerprint.path
                            )
                            evaluation[task.task_id] = fingerprint
                    except DirectCampaignStateError as error:
                        eval_failures.append(f"{model}: {error}")
            if eval_failures:
                raise DirectCampaignStateError(
                    "evaluation campaign has terminal failures: "
                    + "; ".join(sorted(eval_failures))
                )
            final = _record(
                "campaign_completion",
                {
                    "campaign_id": self.plan.campaign_id,
                    "plan_sha256": self.plan.plan_sha256,
                    "status": "completed",
                    "training": {
                        key: training[key].to_dict() for key in sorted(training)
                    },
                    "evaluations": {
                        key: evaluation[key].to_dict() for key in sorted(evaluation)
                    },
                    "completed_at": _timestamp(),
                },
            )
            final_path = self.plan.state_root / "final.json"
            if final_path.exists():
                observed = _read_record(final_path, record_type="campaign_completion")
                # The timestamp is intentionally excluded from replay equality.
                for key in (
                    "campaign_id",
                    "plan_sha256",
                    "status",
                    "training",
                    "evaluations",
                ):
                    if observed.get(key) != final.get(key):
                        raise DirectCampaignStateError("final campaign record changed")
                return observed
            _write_create_only(final_path, final)
            return final
        finally:
            self._release_lock()


def direct_campaign_status(plan: DirectCampaignPlan) -> dict[str, Any]:
    """Inspect completion and durable controller state without mutation."""

    tasks: dict[str, Any] = {}
    for task in (*plan.trajectories, *plan.evaluations):
        kind = "training" if isinstance(task, TrajectorySpec) else "evaluation"
        inspector = (
            inspect_training_completion
            if isinstance(task, TrajectorySpec)
            else inspect_evaluation_completion
        )
        completion = inspector(task)
        root = plan.state_root / kind / task.task_id
        attempts = (
            len([item for item in (root / "attempts").iterdir() if item.is_dir()])
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
            "completion": completion.to_dict() if completion else None,
        }
    final_path = plan.state_root / "final.json"
    training_path = plan.state_root / "training-complete.json"
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.plan_sha256,
        "tasks": tasks,
        "training_barrier": (
            _read_record(training_path, record_type="training_phase_completion")
            if training_path.exists()
            else None
        ),
        "final": (
            _read_record(final_path, record_type="campaign_completion")
            if final_path.exists()
            else None
        ),
    }


__all__ = [
    "CONDITIONS",
    "CONTINUOUS_RUNTIME_SECONDS",
    "MILESTONES",
    "MODELS",
    "PLAN_SCHEMA_VERSION",
    "DirectCampaign",
    "DirectCampaignPlan",
    "DirectCampaignPlanError",
    "DirectCampaignStateError",
    "EvaluationSpec",
    "ResourceSpec",
    "TrajectorySpec",
    "build_evaluation_command",
    "build_train_command",
    "direct_campaign_status",
    "inspect_evaluation_completion",
    "inspect_training_completion",
    "load_direct_campaign_plan",
    "parse_direct_campaign_plan",
]
