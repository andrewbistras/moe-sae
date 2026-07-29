"""Durable scheduler for the fixed Experiment-5 routed/shared campaign.

The controller owns no cloud lifecycle.  It schedules the frozen
``2 panels x 4 conditions x 3 seeds`` training matrix and the eighteen joint
held-out evaluations over exactly seven already-deployed TPU slices.  A
resource worker executes at most one task at a time; tasks may move only among
their plan-pinned eligible resources after a recorded transient failure.

The state directory is append-only apart from the advisory controller lock.
Every attempt has an immutable intent and terminal result, completions are
content fingerprinted, and an intent without a result is deliberately
ambiguous rather than automatically retried.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.direct_horizon_campaign import (
    AttemptExecution,
    AttemptRunner,
    _default_attempt_runner,
    _log_evidence,
)
from moe_sae.experiment5_report import (
    Experiment5ReportError,
    build_experiment5_budget_ledger,
    build_experiment5_report,
    render_experiment5_report,
    validate_experiment5_budget_ledger,
    validate_experiment5_report,
    validate_experiment5_result,
)
from moe_sae.horizon_supervisor import classify_failure
from moe_sae.regional_execution import (
    RegionalExecutionError,
    validate_controller_mapping_receipt,
)

PLAN_SCHEMA_VERSION = "gemma4-experiment5-campaign-plan-v1"
RECORD_SCHEMA_VERSION = "gemma4-experiment5-controller-record-v1"
PROTOCOL_ID = "gemma4-routed-shared-hybrid-v1-20260728"
PHASE_NAME = "routed_shared_hybrid_10m"
TRAINING_RESULT_SCHEMA_VERSION = "gemma4-contribution-sae-training-result-v1"
EVALUATION_RESULT_SCHEMA_VERSION = "gemma4-routed-shared-hybrid-scientific-result-v1"
PAIRED_SCORES_SCHEMA_VERSION = "gemma4-routed-shared-hybrid-paired-scores-v1"

PANELS = ("sae", "transcoder")
CONDITIONS = ("H", "R", "B", "F")
SEEDS = (0, 1, 2)
MILESTONES = (2_000_000, 5_000_000, 10_000_000)
TRAIN_TOKENS = 10_000_000
RESOURCE_COUNT = 7
TRAINING_TASK_COUNT = 24
EVALUATION_TASK_COUNT = 18
CONTINUOUS_RUNTIME_SECONDS = 43_200
COMPLETION_VISIBILITY_RETRIES = 45
COMPLETION_VISIBILITY_POLL_SECONDS = 2.0
RESOURCE_READINESS_POLL_SECONDS = 30.0
RESOURCE_READINESS_TIMEOUT_SECONDS = 120.0
# This deliberately rounds the measured live control-plane burn upward.  The
# durable campaign clock uses the plan's existing $45 launch ceiling (the $50
# hard limit less its $5 contingency) to derive a finite maximum wall time.
CONSERVATIVE_INFRASTRUCTURE_BURN_USD_PER_HOUR = 1.25
WATCHDOG_POLL_SECONDS = 15.0
SUPPORTED_ACCELERATOR_TYPES = ("v4-32", "v6e-16", "v5litepod-16")
TRANSIENT_FAILURES = frozenset({"ssh_transport", "resource_unavailable"})
RETRYABLE_FAILURES = TRANSIENT_FAILURES | {"spot_preemption"}
RESUMABLE_FAILURES = RETRYABLE_FAILURES | {"controller_stop"}
CACHE_PROFILES = ("target", "paired")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_ZONE = re.compile(r"^[a-z][a-z0-9-]*-[a-z]$")
_GCS_PREFIX = re.compile(r"^gs://[a-z0-9][a-z0-9._-]*/[^/\s].*$")
_RESOURCE_NOT_FOUND = re.compile(
    r"(?:\bNOT_FOUND\b|was not found|could not fetch resource.*\b404\b)",
    re.I | re.S,
)
_AUTHORIZATION_FAILURE = re.compile(
    r"(?:PERMISSION_DENIED|permission denied|forbidden|"
    r"does not have permission|insufficient authentication|"
    r"invalid_grant|reauthentication|credentialed accounts)",
    re.I,
)
_FORMING_TPU_STATES = frozenset(
    {
        "CREATING",
        "HIDDEN",
        "HIDING",
        "REIMAGING",
        "REPAIRING",
        "RESTARTING",
        "STARTING",
        "STOPPED",
        "STOPPING",
        "SUSPENDED",
        "SUSPENDING",
        "UNHIDING",
    }
)
_TERMINAL_TPU_STATES = frozenset(
    {
        "CANCELLED",
        "DELETING",
        "FAILED",
        "PREEMPTED",
        "TERMINATED",
        "UNKNOWN",
    }
)
_KNOWN_TPU_STATES = _FORMING_TPU_STATES | _TERMINAL_TPU_STATES | {"READY"}
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
        "EXPERIMENT5_PANEL",
        "CONTRIBUTION_FRESH_CONTROLLER_MAPPING",
        "CONTRIBUTION_LOCAL_CACHE_INVENTORY",
        "CONTRIBUTION_REGIONAL_MIRROR_SEAL",
        "CONTRIBUTION_SAE_CONTROL_ROOT",
        "CONTRIBUTION_SAE_CHECKPOINT_ROOT",
        "CONTRIBUTION_SAE_LOCAL_LOG_ROOT",
        "CONTRIBUTION_EVALUATION_LOG_ROOT",
        "TRANSCODER_EVALUATION_LOG_ROOT",
        "EXPERIMENT5_RESULT_PATH_CONTROLLER",
        "EXPERIMENT5_PAIRED_SCORES_PATH_CONTROLLER",
        "CONTRIBUTION_SOURCE_OVERLAY_ROOT",
        "CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT",
        "CONTRIBUTION_SOURCE_OVERLAY_PREFIX",
        "CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX",
    }
)


class Experiment5CampaignPlanError(RuntimeError):
    """The campaign plan is incomplete, inconsistent, or unsafe."""


class Experiment5CampaignStateError(RuntimeError):
    """Durable campaign state is corrupt, ambiguous, or terminal."""


class Experiment5BudgetStop(Experiment5CampaignStateError):
    """A new attempt would cross the conservative USD ceiling."""


@dataclass(frozen=True, slots=True)
class EvidencePath:
    controller_path: Path
    remote_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class CampaignResource:
    resource_id: str
    project: str
    tpu_name: str
    zone: str
    region: str
    accelerator_type: str
    internal_ip: bool
    ssh_user: str | None
    cache_profile: str
    deployed_commit: str
    controller_mapping: EvidencePath
    cache_inventory: EvidencePath
    mirror_seal: EvidencePath | None
    environment: tuple[tuple[str, str], ...]

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)


class ResourceReadinessStatus(str, Enum):
    """Controller-visible lifecycle state for one exact planned TPU."""

    READY = "ready"
    FORMING = "forming"
    TRANSIENT_CONTROL_PLANE = "transient_control_plane"
    TERMINAL_REPLACEMENT_REQUIRED = "terminal_replacement_required"


@dataclass(frozen=True, slots=True)
class ResourceReadinessObservation:
    """One typed, auditable observation of a plan-bound TPU incarnation."""

    status: ResourceReadinessStatus
    state: str | None
    health: str | None
    create_time: str | None
    spot: bool | None
    absent: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "state": self.state,
            "health": self.health,
            "create_time": self.create_time,
            "spot": self.spot,
            "absent": self.absent,
            "detail": self.detail[-4_096:],
        }


ResourceReadinessProbe = Callable[
    [CampaignResource],
    ResourceReadinessObservation,
]


@dataclass(frozen=True, slots=True)
class TrainingTask:
    task_id: str
    panel: str
    condition: str
    seed: int
    eligible_resource_ids: tuple[str, ...]
    run_id: str
    config_remote: str
    control_root_remote: str
    checkpoint_root_gcs: str
    source_overlay_control_root_remote: str | None
    source_overlay_object_prefix_gcs: str | None
    completion_path_controller: Path
    environment: tuple[tuple[str, str], ...]

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    task_id: str
    panel: str
    seed: int
    milestone_tokens: int
    eligible_resource_ids: tuple[str, ...]
    config_remote: str
    control_root_remote: str
    checkpoint_root_gcs: str
    train_source_root_remote: str | None
    eval_source_root_remote: str | None
    train_source_object_prefix_gcs: str | None
    eval_source_object_prefix_gcs: str | None
    result_root_remote: str
    completion_path_controller: Path
    paired_scores_path_controller: Path
    run_roots_remote: tuple[tuple[str, str], ...]
    environment: tuple[tuple[str, str], ...]

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)

    def run_roots_dict(self) -> dict[str, str]:
        return dict(self.run_roots_remote)


CampaignTask = TrainingTask | EvaluationTask


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_transient_attempts: int
    initial_backoff_seconds: float
    maximum_backoff_seconds: float


@dataclass(frozen=True, slots=True)
class CostPolicy:
    hard_limit_usd: float
    contingency_usd: float
    already_spent_usd: float
    fixed_cost_reserve_usd: float
    training_attempt_reserve_usd: float
    evaluation_attempt_reserve_usd: float

    @property
    def initial_launch_limit_usd(self) -> float:
        return self.hard_limit_usd - self.contingency_usd

    def attempt_reserve(self, kind: str) -> float:
        if kind == "training":
            return self.training_attempt_reserve_usd
        if kind == "evaluation":
            return self.evaluation_attempt_reserve_usd
        raise ValueError(f"unknown task kind {kind!r}")


@dataclass(frozen=True, slots=True)
class Experiment5CampaignPlan:
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
    trajectories: tuple[TrainingTask, ...]
    evaluations: tuple[EvaluationTask, ...]
    plan_sha256: str
    source: Mapping[str, Any]

    def environment_dict(self) -> dict[str, str]:
        return dict(self.environment)

    def resource(self, resource_id: str) -> CampaignResource:
        for resource in self.resources:
            if resource.resource_id == resource_id:
                return resource
        raise Experiment5CampaignPlanError(f"unknown resource {resource_id!r}")


@dataclass(frozen=True, slots=True)
class CompletionFingerprint:
    path: str
    size_bytes: int
    sha256: str
    scientific_sha256: str
    companion_path: str | None = None
    companion_size_bytes: int | None = None
    companion_sha256: str | None = None
    companion_scientific_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "scientific_sha256": self.scientific_sha256,
            "companion_path": self.companion_path,
            "companion_size_bytes": self.companion_size_bytes,
            "companion_sha256": self.companion_sha256,
            "companion_scientific_sha256": self.companion_scientific_sha256,
        }


@dataclass(frozen=True, slots=True)
class _AttemptOutcome:
    completion: CompletionFingerprint | None
    retry_delay_seconds: float = 0.0


def _exact_mapping(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise Experiment5CampaignPlanError(
            f"{name} fields differ from the strict schema: {observed}"
        )
    if not all(isinstance(key, str) for key in value):
        raise Experiment5CampaignPlanError(f"{name} keys must be strings")
    return dict(value)


def _safe_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise Experiment5CampaignPlanError(f"{name} is not a safe identifier")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Experiment5CampaignPlanError(f"{name} must be a lowercase SHA-256")
    return value


def _git_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
        raise Experiment5CampaignPlanError(f"{name} must be a full Git SHA")
    return value


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Experiment5CampaignPlanError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise Experiment5CampaignPlanError(f"{name} exceeds {maximum}")
    return value


def _money(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Experiment5CampaignPlanError(f"{name} must be numeric")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0:
        raise Experiment5CampaignPlanError(f"{name} must be finite and nonnegative")
    return selected


def _controller_path(value: Any, name: str) -> Path:
    if not isinstance(value, str):
        raise Experiment5CampaignPlanError(f"{name} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or value == "/" or ".." in path.parts:
        raise Experiment5CampaignPlanError(f"{name} must be a non-root absolute path")
    return path


def _remote_path(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise Experiment5CampaignPlanError(f"{name} must be an absolute POSIX path")
    path = PurePosixPath(value)
    if not path.is_absolute() or value == "/" or ".." in path.parts:
        raise Experiment5CampaignPlanError(
            f"{name} must be a non-root absolute POSIX path"
        )
    return str(path)


def _gcs_prefix(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or _GCS_PREFIX.fullmatch(value) is None
        or value.endswith("/")
    ):
        raise Experiment5CampaignPlanError(f"{name} must be a non-root GCS prefix")
    return value


def _panel_remote_path(value: Any, panel: str, name: str) -> str | None:
    if panel == "sae":
        if value is not None:
            raise Experiment5CampaignPlanError(f"{name} must be null for the SAE panel")
        return None
    if value is None:
        raise Experiment5CampaignPlanError(
            f"{name} is required for the transcoder panel"
        )
    return _remote_path(value, name)


def _panel_gcs_prefix(value: Any, panel: str, name: str) -> str | None:
    if panel == "sae":
        if value is not None:
            raise Experiment5CampaignPlanError(f"{name} must be null for the SAE panel")
        return None
    if value is None:
        raise Experiment5CampaignPlanError(
            f"{name} is required for the transcoder panel"
        )
    return _gcs_prefix(value, name)


def _environment(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise Experiment5CampaignPlanError(f"{name} must be a mapping")
    result: list[tuple[str, str]] = []
    for key, raw in value.items():
        if not isinstance(key, str) or _ENV_NAME.fullmatch(key) is None:
            raise Experiment5CampaignPlanError(f"{name} has invalid variable {key!r}")
        if key in _RESERVED_ENVIRONMENT:
            raise Experiment5CampaignPlanError(
                f"{name} may not override controller-owned variable {key}"
            )
        if not isinstance(raw, str) or any(char in raw for char in "\x00\n\r"):
            raise Experiment5CampaignPlanError(
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


def _resource(value: Any, index: int, project_commit: str) -> CampaignResource:
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
            "cache_profile",
            "deployed_commit",
            "evidence",
            "environment",
        },
        name,
    )
    zone = _safe_id(selected["zone"], f"{name}.zone")
    region = _safe_id(selected["region"], f"{name}.region")
    if _ZONE.fullmatch(zone) is None or not zone.startswith(region + "-"):
        raise Experiment5CampaignPlanError(f"{name} zone does not belong to region")
    accelerator = selected["accelerator_type"]
    if accelerator not in SUPPORTED_ACCELERATOR_TYPES:
        raise Experiment5CampaignPlanError(f"{name} accelerator type is unsupported")
    internal_ip = selected["internal_ip"]
    if type(internal_ip) is not bool:
        raise Experiment5CampaignPlanError(f"{name}.internal_ip must be boolean")
    if accelerator in {"v6e-16", "v5litepod-16"} and not internal_ip:
        raise Experiment5CampaignPlanError(f"{name} regional TPU requires internal IP")
    ssh_user = selected["ssh_user"]
    if ssh_user is not None:
        ssh_user = _safe_id(ssh_user, f"{name}.ssh_user")
        if ssh_user.lower() != ssh_user:
            raise Experiment5CampaignPlanError(f"{name}.ssh_user must be lowercase")
    cache_profile = selected["cache_profile"]
    if cache_profile not in CACHE_PROFILES:
        raise Experiment5CampaignPlanError(f"{name}.cache_profile is invalid")
    deployed_commit = _git_sha(selected["deployed_commit"], f"{name}.deployed_commit")
    if deployed_commit != project_commit:
        raise Experiment5CampaignPlanError(
            f"{name} is not declared deployed at project_commit"
        )
    evidence = _exact_mapping(
        selected["evidence"],
        {"controller_mapping", "cache_inventory", "mirror_seal"},
        f"{name}.evidence",
    )
    mirror_raw = evidence["mirror_seal"]
    mirror = (
        None
        if mirror_raw is None
        else _evidence(mirror_raw, f"{name}.evidence.mirror_seal")
    )
    if accelerator != "v4-32" and mirror is None:
        raise Experiment5CampaignPlanError(
            f"{name} regional TPU requires mirror-seal evidence"
        )
    environment = _environment(selected["environment"], f"{name}.environment")
    environment_dict = dict(environment)
    if accelerator == "v5litepod-16":
        if not environment_dict.get("MOE_SAE_V5LITEPOD_DEVICE_KIND"):
            raise Experiment5CampaignPlanError(
                f"{name} v5litepod resource lacks observed device kind"
            )
    elif "MOE_SAE_V5LITEPOD_DEVICE_KIND" in environment_dict:
        raise Experiment5CampaignPlanError(
            f"{name} non-v5 resource declares a v5 device kind"
        )
    return CampaignResource(
        resource_id=_safe_id(selected["id"], f"{name}.id"),
        project=_safe_id(selected["project"], f"{name}.project"),
        tpu_name=_safe_id(selected["tpu_name"], f"{name}.tpu_name"),
        zone=zone,
        region=region,
        accelerator_type=accelerator,
        internal_ip=internal_ip,
        ssh_user=ssh_user,
        cache_profile=cache_profile,
        deployed_commit=deployed_commit,
        controller_mapping=_evidence(
            evidence["controller_mapping"],
            f"{name}.evidence.controller_mapping",
        ),
        cache_inventory=_evidence(
            evidence["cache_inventory"],
            f"{name}.evidence.cache_inventory",
        ),
        mirror_seal=mirror,
        environment=environment,
    )


def _eligible_resources(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise Experiment5CampaignPlanError(f"{name} must be a nonempty list")
    selected = tuple(
        _safe_id(item, f"{name}[{index}]") for index, item in enumerate(value)
    )
    if len(set(selected)) != len(selected):
        raise Experiment5CampaignPlanError(f"{name} contains duplicates")
    return selected


def _training_task(value: Any, index: int) -> TrainingTask:
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
        raise Experiment5CampaignPlanError(f"{name} has an invalid scientific role")
    expected_id = f"train-{panel}-{condition.lower()}-seed{seed}"
    expected_run = f"experiment5-{panel}-{condition.lower()}-seed{seed}"
    if selected["id"] != expected_id or selected["run_id"] != expected_run:
        raise Experiment5CampaignPlanError(f"{name} task or run ID changed")
    if selected["phase"] != PHASE_NAME:
        raise Experiment5CampaignPlanError(f"{name}.phase changed")
    completion = _controller_path(
        selected["completion_path_controller"],
        f"{name}.completion_path_controller",
    )
    if (
        completion.name != "training-result.json"
        or completion.parent.name != expected_run
    ):
        raise Experiment5CampaignPlanError(
            f"{name} completion is not bound to its immutable run ID"
        )
    return TrainingTask(
        task_id=expected_id,
        panel=panel,
        condition=condition,
        seed=seed,
        eligible_resource_ids=_eligible_resources(
            selected["eligible_resource_ids"], f"{name}.eligible_resource_ids"
        ),
        run_id=expected_run,
        config_remote=_remote_path(selected["config_remote"], f"{name}.config_remote"),
        control_root_remote=_remote_path(
            selected["control_root_remote"], f"{name}.control_root_remote"
        ),
        checkpoint_root_gcs=_gcs_prefix(
            selected["checkpoint_root_gcs"], f"{name}.checkpoint_root_gcs"
        ),
        source_overlay_control_root_remote=_panel_remote_path(
            selected["source_overlay_control_root_remote"],
            panel,
            f"{name}.source_overlay_control_root_remote",
        ),
        source_overlay_object_prefix_gcs=_panel_gcs_prefix(
            selected["source_overlay_object_prefix_gcs"],
            panel,
            f"{name}.source_overlay_object_prefix_gcs",
        ),
        completion_path_controller=completion,
        environment=_environment(selected["environment"], f"{name}.environment"),
    )


def _run_roots(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    selected = _exact_mapping(value, set(CONDITIONS), name)
    return tuple(
        (
            condition,
            _remote_path(selected[condition], f"{name}.{condition}"),
        )
        for condition in CONDITIONS
    )


def _evaluation_task(value: Any, index: int) -> EvaluationTask:
    name = f"evaluations[{index}]"
    selected = _exact_mapping(
        value,
        {
            "id",
            "panel",
            "seed",
            "milestone_tokens",
            "eligible_resource_ids",
            "config_remote",
            "phase",
            "control_root_remote",
            "checkpoint_root_gcs",
            "train_source_root_remote",
            "eval_source_root_remote",
            "train_source_object_prefix_gcs",
            "eval_source_object_prefix_gcs",
            "result_root_remote",
            "completion_path_controller",
            "paired_scores_path_controller",
            "run_roots_remote",
            "environment",
        },
        name,
    )
    panel = selected["panel"]
    seed = selected["seed"]
    milestone = selected["milestone_tokens"]
    if panel not in PANELS or seed not in SEEDS or milestone not in MILESTONES:
        raise Experiment5CampaignPlanError(f"{name} has an invalid scientific role")
    expected_id = f"eval-{panel}-seed{seed}-{milestone}"
    if selected["id"] != expected_id or selected["phase"] != PHASE_NAME:
        raise Experiment5CampaignPlanError(f"{name} task ID or phase changed")
    completion = _controller_path(
        selected["completion_path_controller"],
        f"{name}.completion_path_controller",
    )
    paired = _controller_path(
        selected["paired_scores_path_controller"],
        f"{name}.paired_scores_path_controller",
    )
    if (
        completion.name != "scientific-result.json"
        or paired.name != "paired-scores.json"
        or completion.parent != paired.parent
        or completion.parent.name != str(milestone)
    ):
        raise Experiment5CampaignPlanError(
            f"{name} result and paired scores must share the milestone directory"
        )
    return EvaluationTask(
        task_id=expected_id,
        panel=panel,
        seed=seed,
        milestone_tokens=milestone,
        eligible_resource_ids=_eligible_resources(
            selected["eligible_resource_ids"], f"{name}.eligible_resource_ids"
        ),
        config_remote=_remote_path(selected["config_remote"], f"{name}.config_remote"),
        control_root_remote=_remote_path(
            selected["control_root_remote"], f"{name}.control_root_remote"
        ),
        checkpoint_root_gcs=_gcs_prefix(
            selected["checkpoint_root_gcs"], f"{name}.checkpoint_root_gcs"
        ),
        train_source_root_remote=_panel_remote_path(
            selected["train_source_root_remote"],
            panel,
            f"{name}.train_source_root_remote",
        ),
        eval_source_root_remote=_panel_remote_path(
            selected["eval_source_root_remote"],
            panel,
            f"{name}.eval_source_root_remote",
        ),
        train_source_object_prefix_gcs=_panel_gcs_prefix(
            selected["train_source_object_prefix_gcs"],
            panel,
            f"{name}.train_source_object_prefix_gcs",
        ),
        eval_source_object_prefix_gcs=_panel_gcs_prefix(
            selected["eval_source_object_prefix_gcs"],
            panel,
            f"{name}.eval_source_object_prefix_gcs",
        ),
        result_root_remote=_remote_path(
            selected["result_root_remote"], f"{name}.result_root_remote"
        ),
        completion_path_controller=completion,
        paired_scores_path_controller=paired,
        run_roots_remote=_run_roots(
            selected["run_roots_remote"], f"{name}.run_roots_remote"
        ),
        environment=_environment(selected["environment"], f"{name}.environment"),
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
        contingency_usd=_money(selected["contingency_usd"], "contingency_usd"),
        already_spent_usd=_money(selected["already_spent_usd"], "already_spent_usd"),
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
    if result.hard_limit_usd != 50.0 or result.contingency_usd != 5.0:
        raise Experiment5CampaignPlanError(
            "Experiment 5 requires the frozen $50 limit and $5 contingency"
        )
    initial_projection = (
        result.already_spent_usd
        + result.fixed_cost_reserve_usd
        + TRAINING_TASK_COUNT * result.training_attempt_reserve_usd
        + EVALUATION_TASK_COUNT * result.evaluation_attempt_reserve_usd
    )
    if initial_projection > result.initial_launch_limit_usd + 1e-9:
        raise Experiment5CampaignPlanError(
            "one-attempt campaign projection exceeds the $45 launch reserve"
        )
    return result


def parse_experiment5_campaign_plan(value: Any) -> Experiment5CampaignPlan:
    """Validate and freeze one exact Experiment-5 campaign plan."""

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
            "cost_policy",
            "environment",
            "resources",
            "trajectories",
            "evaluations",
        },
        "plan",
    )
    if selected["schema_version"] != PLAN_SCHEMA_VERSION:
        raise Experiment5CampaignPlanError("campaign schema version changed")
    project_commit = _git_sha(selected["project_commit"], "project_commit")
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
        raise Experiment5CampaignPlanError("retry backoff maximum precedes initial")
    raw_resources = selected["resources"]
    raw_trajectories = selected["trajectories"]
    raw_evaluations = selected["evaluations"]
    if not isinstance(raw_resources, list) or len(raw_resources) != RESOURCE_COUNT:
        raise Experiment5CampaignPlanError("plan must contain exactly seven resources")
    if (
        not isinstance(raw_trajectories, list)
        or len(raw_trajectories) != TRAINING_TASK_COUNT
    ):
        raise Experiment5CampaignPlanError(
            "plan must contain exactly 24 training trajectories"
        )
    if (
        not isinstance(raw_evaluations, list)
        or len(raw_evaluations) != EVALUATION_TASK_COUNT
    ):
        raise Experiment5CampaignPlanError(
            "plan must contain exactly 18 joint evaluations"
        )
    resources = tuple(
        _resource(item, index, project_commit)
        for index, item in enumerate(raw_resources)
    )
    trajectories = tuple(
        _training_task(item, index) for index, item in enumerate(raw_trajectories)
    )
    evaluations = tuple(
        _evaluation_task(item, index) for index, item in enumerate(raw_evaluations)
    )
    _validate_plan_relations(resources, trajectories, evaluations)
    return Experiment5CampaignPlan(
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
            selected["max_parallel_train"], "max_parallel_train", maximum=RESOURCE_COUNT
        ),
        max_parallel_eval=_positive_int(
            selected["max_parallel_eval"], "max_parallel_eval", maximum=RESOURCE_COUNT
        ),
        retry_policy=retry,
        cost_policy=_cost_policy(selected["cost_policy"]),
        environment=_environment(selected["environment"], "environment"),
        resources=resources,
        trajectories=trajectories,
        evaluations=evaluations,
        plan_sha256=canonical_sha256(selected),
        source=selected,
    )


def _validate_plan_relations(
    resources: tuple[CampaignResource, ...],
    trajectories: tuple[TrainingTask, ...],
    evaluations: tuple[EvaluationTask, ...],
) -> None:
    resource_ids = [resource.resource_id for resource in resources]
    if len(set(resource_ids)) != RESOURCE_COUNT:
        raise Experiment5CampaignPlanError("resource IDs must be unique")
    physical = [
        (resource.project, resource.zone, resource.tpu_name) for resource in resources
    ]
    if len(set(physical)) != RESOURCE_COUNT:
        raise Experiment5CampaignPlanError("physical TPU resources must be unique")
    if len({str(item.controller_mapping.controller_path) for item in resources}) != 7:
        raise Experiment5CampaignPlanError("controller mappings must be per-resource")
    if len({str(item.cache_inventory.controller_path) for item in resources}) != 7:
        raise Experiment5CampaignPlanError("cache inventories must be per-resource")
    expected_training = {
        (panel, condition, seed)
        for panel in PANELS
        for condition in CONDITIONS
        for seed in SEEDS
    }
    observed_training = {
        (task.panel, task.condition, task.seed) for task in trajectories
    }
    if observed_training != expected_training:
        raise Experiment5CampaignPlanError("training matrix is not exact")
    expected_evaluations = {
        (panel, seed, milestone)
        for panel in PANELS
        for seed in SEEDS
        for milestone in MILESTONES
    }
    observed_evaluations = {
        (task.panel, task.seed, task.milestone_tokens) for task in evaluations
    }
    if observed_evaluations != expected_evaluations:
        raise Experiment5CampaignPlanError("evaluation matrix is not exact")
    all_tasks: tuple[CampaignTask, ...] = (*trajectories, *evaluations)
    resource_by_id = {resource.resource_id: resource for resource in resources}
    for task in all_tasks:
        if not set(task.eligible_resource_ids) <= set(resource_ids):
            raise Experiment5CampaignPlanError(
                f"{task.task_id} references an unknown resource"
            )
        eligible = [resource_by_id[item] for item in task.eligible_resource_ids]
        if task.panel == "transcoder" and any(
            resource.cache_profile != "paired" for resource in eligible
        ):
            raise Experiment5CampaignPlanError(
                f"{task.task_id} permits a target-only cache"
            )
        if len({resource.region for resource in eligible}) != 1:
            raise Experiment5CampaignPlanError(
                f"{task.task_id} may not cross regional artifact planes"
            )
        if len({resource.accelerator_type for resource in eligible}) != 1:
            raise Experiment5CampaignPlanError(
                f"{task.task_id} may not change accelerator generation on retry"
            )
    completion_paths = [str(task.completion_path_controller) for task in all_tasks] + [
        str(task.paired_scores_path_controller) for task in evaluations
    ]
    if len(set(completion_paths)) != len(completion_paths):
        raise Experiment5CampaignPlanError("completion paths must be unique")

    by_role = {(task.panel, task.seed, task.condition): task for task in trajectories}
    eval_by_role = {
        (task.panel, task.seed, task.milestone_tokens): task for task in evaluations
    }
    for panel in PANELS:
        for seed in SEEDS:
            group = [by_role[(panel, seed, condition)] for condition in CONDITIONS]
            if (
                len({task.config_remote for task in group}) != 1
                or len({task.control_root_remote for task in group}) != 1
                or len({task.checkpoint_root_gcs for task in group}) != 1
                or len({task.source_overlay_control_root_remote for task in group}) != 1
                or len({task.source_overlay_object_prefix_gcs for task in group}) != 1
            ):
                raise Experiment5CampaignPlanError(
                    f"{panel}.seed{seed} conditions do not share one artifact plane"
                )
            regions = {
                resource_by_id[resource_id].region
                for task in group
                for resource_id in task.eligible_resource_ids
            }
            if len(regions) != 1:
                raise Experiment5CampaignPlanError(
                    f"{panel}.seed{seed} conditions span regions"
                )
            h_types = {
                resource_by_id[item].accelerator_type
                for item in by_role[(panel, seed, "H")].eligible_resource_ids
            }
            f_types = {
                resource_by_id[item].accelerator_type
                for item in by_role[(panel, seed, "F")].eligible_resource_ids
            }
            if len(h_types) != 1 or h_types != f_types:
                raise Experiment5CampaignPlanError(
                    f"{panel}.seed{seed} H/F are not generation matched"
                )
            expected_roots = {
                condition: str(
                    PurePosixPath(group[0].control_root_remote)
                    / "runs"
                    / by_role[(panel, seed, condition)].run_id
                )
                for condition in CONDITIONS
            }
            for milestone in MILESTONES:
                evaluation = eval_by_role[(panel, seed, milestone)]
                eval_regions = {
                    resource_by_id[item].region
                    for item in evaluation.eligible_resource_ids
                }
                if (
                    eval_regions != regions
                    or evaluation.config_remote != group[0].config_remote
                    or evaluation.control_root_remote != group[0].control_root_remote
                    or evaluation.checkpoint_root_gcs != group[0].checkpoint_root_gcs
                    or evaluation.run_roots_dict() != expected_roots
                    or evaluation.train_source_root_remote
                    != group[0].source_overlay_control_root_remote
                    or evaluation.train_source_object_prefix_gcs
                    != group[0].source_overlay_object_prefix_gcs
                ):
                    raise Experiment5CampaignPlanError(
                        f"{evaluation.task_id} differs from its training group"
                    )


def load_experiment5_campaign_plan(
    path: str | Path,
) -> Experiment5CampaignPlan:
    selected = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise Experiment5CampaignPlanError(
            f"cannot read campaign plan: {selected}"
        ) from error
    return parse_experiment5_campaign_plan(value)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise Experiment5CampaignStateError("record is not canonical JSON") from error


def _record(record_type: str, body: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "record_type": record_type,
        **dict(body),
    }
    return {**unsigned, "record_sha256": canonical_sha256(unsigned)}


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = _canonical_json_bytes(dict(value)) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise Experiment5CampaignStateError(
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


def _seal_exact_bytes(path: Path, payload: bytes, *, name: str) -> str:
    """Create one immutable artifact, or prove an existing artifact is exact."""

    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise Experiment5CampaignStateError(
                f"{name} is not a regular immutable file: {path}"
            )
        try:
            observed = path.read_bytes()
        except OSError as error:
            raise Experiment5CampaignStateError(
                f"cannot read existing {name}: {path}"
            ) from error
        if observed != payload:
            raise Experiment5CampaignStateError(f"existing {name} changed: {path}")
        return digest

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Hard-link publication is atomic and refuses to replace a final
            # artifact. A killed writer can leave only an unreferenced staging
            # file, never a partial final report.
            os.link(temporary, path)
        except FileExistsError:
            return _seal_exact_bytes(path, payload, name=name)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return digest


def _read_record(path: Path, *, record_type: str | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Experiment5CampaignStateError(
            f"cannot read controller state {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise Experiment5CampaignStateError(
            f"controller state is not an object: {path}"
        )
    selected = dict(value)
    claimed = selected.pop("record_sha256", None)
    if (
        selected.get("schema_version") != RECORD_SCHEMA_VERSION
        or not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or canonical_sha256(selected) != claimed
        or (record_type is not None and selected.get("record_type") != record_type)
    ):
        raise Experiment5CampaignStateError(
            f"controller state identity is invalid: {path}"
        )
    return dict(value)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_self_hashed_json(
    path: Path,
    *,
    hash_field: str,
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise Experiment5CampaignStateError(f"completion is not a regular file: {path}")
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Experiment5CampaignStateError(
            f"completion JSON is invalid: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise Experiment5CampaignStateError(f"completion is not an object: {path}")
    selected = dict(value)
    claimed = selected.pop(hash_field, None)
    if (
        not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or canonical_sha256(selected) != claimed
    ):
        raise Experiment5CampaignStateError(f"completion self-hash is invalid: {path}")
    return dict(value), payload


def inspect_training_completion(
    task: TrainingTask,
) -> CompletionFingerprint | None:
    path = task.completion_path_controller
    if not path.exists():
        return None
    value, payload = _read_self_hashed_json(path, hash_field="result_sha256")
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
        raise Experiment5CampaignStateError(
            f"training completion role changed: {task.task_id}"
        )
    return CompletionFingerprint(
        path=str(path),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        scientific_sha256=str(value["result_sha256"]),
    )


def inspect_evaluation_completion(
    plan: Experiment5CampaignPlan,
    task: EvaluationTask,
) -> CompletionFingerprint | None:
    path = task.completion_path_controller
    paired_path = task.paired_scores_path_controller
    if not path.exists() and not paired_path.exists():
        return None
    if not path.exists() or not paired_path.exists():
        raise Experiment5CampaignStateError(
            f"evaluation completion is partial: {task.task_id}"
        )
    value, payload = _read_self_hashed_json(path, hash_field="result_sha256")
    paired, paired_payload = _read_self_hashed_json(
        paired_path,
        hash_field="artifact_sha256",
    )
    if (
        value.get("schema_version") != EVALUATION_RESULT_SCHEMA_VERSION
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "curve_point_measured"
        or value.get("panel") != task.panel
        or value.get("seed") != task.seed
        or value.get("checkpoint_target_tokens") != task.milestone_tokens
        or value.get("source_revision") != plan.project_commit
        or paired.get("schema_version") != PAIRED_SCORES_SCHEMA_VERSION
        or paired.get("protocol_id") != PROTOCOL_ID
        or paired.get("panel") != task.panel
        or paired.get("seed") != task.seed
        or paired.get("checkpoint_target_tokens") != task.milestone_tokens
        or value.get("predictions_artifact_sha256") != paired.get("artifact_sha256")
    ):
        raise Experiment5CampaignStateError(
            f"evaluation completion role changed: {task.task_id}"
        )
    return CompletionFingerprint(
        path=str(path),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        scientific_sha256=str(value["result_sha256"]),
        companion_path=str(paired_path),
        companion_size_bytes=len(paired_payload),
        companion_sha256=hashlib.sha256(paired_payload).hexdigest(),
        companion_scientific_sha256=str(paired["artifact_sha256"]),
    )


def _verify_file(path: Path, expected_sha256: str, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise Experiment5CampaignStateError(f"{name} is not a regular file: {path}")
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise Experiment5CampaignStateError(f"cannot read {name}: {path}") from error
    if digest != expected_sha256:
        raise Experiment5CampaignStateError(f"{name} SHA-256 changed: {path}")


def _verify_checkout(plan: Experiment5CampaignPlan) -> None:
    if plan.working_directory.is_symlink() or not plan.working_directory.is_dir():
        raise Experiment5CampaignStateError("working_directory is not a real directory")
    if (
        plan.python_executable.is_symlink()
        or not plan.python_executable.is_file()
        or not os.access(plan.python_executable, os.X_OK)
    ):
        raise Experiment5CampaignStateError("python_executable is not executable")
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
        raise Experiment5CampaignStateError(
            "cannot verify controller checkout"
        ) from error
    if head != plan.project_commit or status:
        raise Experiment5CampaignStateError(
            "controller checkout must be clean at project_commit"
        )
    for panel in PANELS:
        config = (
            plan.working_directory
            / "configs"
            / f"gemma4_hybrid_{panel}_v1.yaml"
        )
        if config.is_symlink() or not config.is_file():
            raise Experiment5CampaignStateError(
                f"{panel} controller config is not a regular file"
            )


def _preflight(
    plan: Experiment5CampaignPlan,
    *,
    verify_checkout: bool,
) -> None:
    if verify_checkout:
        _verify_checkout(plan)
    active_resource_ids = {
        resource_id
        for task in (*plan.trajectories, *plan.evaluations)
        for resource_id in task.eligible_resource_ids
    }
    for resource in plan.resources:
        if resource.resource_id not in active_resource_ids:
            continue
        if resource.deployed_commit != plan.project_commit:
            raise Experiment5CampaignStateError(
                f"{resource.resource_id} deployment commitment changed"
            )
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


def _default_resource_readiness_probe(
    resource: CampaignResource,
) -> ResourceReadinessObservation:
    """Observe the exact plan-bound TPU without mutating cloud state.

    A typed result keeps resource formation, control-plane transport, and a
    terminal TPU incarnation distinct.  Malformed or unauthorized evidence
    still fails closed.
    """

    try:
        mapping_value = json.loads(
            resource.controller_mapping.controller_path.read_text(encoding="ascii")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Experiment5CampaignStateError(
            f"{resource.resource_id} controller mapping is invalid"
        ) from error
    if not isinstance(mapping_value, Mapping):
        raise Experiment5CampaignStateError(
            f"{resource.resource_id} controller mapping is not an object"
        )
    try:
        mapping = validate_controller_mapping_receipt(mapping_value)
    except RegionalExecutionError as error:
        raise Experiment5CampaignStateError(
            f"{resource.resource_id} controller mapping is invalid"
        ) from error
    if (
        mapping.get("tpu_name") != resource.tpu_name
        or mapping.get("zone") != resource.zone
        or mapping.get("accelerator_type") != resource.accelerator_type
    ):
        raise Experiment5CampaignStateError(
            f"{resource.resource_id} controller mapping identity changed"
        )

    command = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "describe",
        resource.tpu_name,
        f"--project={resource.project}",
        f"--zone={resource.zone}",
        "--format=json",
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=RESOURCE_READINESS_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ResourceReadinessObservation(
            status=ResourceReadinessStatus.TRANSIENT_CONTROL_PLANE,
            state=None,
            health=None,
            create_time=None,
            spot=None,
            absent=False,
            detail="gcloud TPU describe timed out",
        )
    except OSError as error:
        raise Experiment5CampaignStateError(
            f"cannot execute TPU readiness probe for {resource.resource_id}"
        ) from error
    output = f"{result.stdout or ''}\n{result.stderr or ''}"
    if result.returncode != 0:
        if _AUTHORIZATION_FAILURE.search(output):
            raise Experiment5CampaignStateError(
                f"TPU readiness probe is unauthorized for {resource.resource_id}"
            )
        if _RESOURCE_NOT_FOUND.search(output):
            return ResourceReadinessObservation(
                status=ResourceReadinessStatus.FORMING,
                state=None,
                health=None,
                create_time=None,
                spot=None,
                absent=True,
                detail=output.strip()[-4_096:] or "TPU node is absent",
            )
        classification = classify_failure(
            returncode=result.returncode,
            output=output,
            timed_out=False,
            completion_present=False,
            checkpoint_before=None,
            checkpoint_after=None,
            segment=None,
        )
        if classification == "spot_preemption":
            return ResourceReadinessObservation(
                status=(ResourceReadinessStatus.TERMINAL_REPLACEMENT_REQUIRED),
                state="PREEMPTED",
                health=None,
                create_time=None,
                spot=True,
                absent=False,
                detail=output.strip()[-4_096:],
            )
        if classification in TRANSIENT_FAILURES:
            return ResourceReadinessObservation(
                status=ResourceReadinessStatus.TRANSIENT_CONTROL_PLANE,
                state=None,
                health=None,
                create_time=None,
                spot=None,
                absent=False,
                detail=output.strip()[-4_096:],
            )
        raise Experiment5CampaignStateError(
            f"TPU readiness probe failed for {resource.resource_id}: "
            f"{output.strip()[-1_024:]}"
        )
    try:
        description = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise Experiment5CampaignStateError(
            f"TPU readiness probe returned malformed JSON for {resource.resource_id}"
        ) from error
    if not isinstance(description, Mapping):
        raise Experiment5CampaignStateError(
            f"TPU readiness probe is not an object for {resource.resource_id}"
        )
    full_name = description.get("name")
    expected_name = (
        f"projects/{resource.project}/locations/{resource.zone}/nodes/"
        f"{resource.tpu_name}"
    )
    state = description.get("state")
    if (
        full_name != expected_name
        or description.get("acceleratorType") != resource.accelerator_type
        or not isinstance(state, str)
        or state.upper() not in _KNOWN_TPU_STATES
    ):
        raise Experiment5CampaignStateError(
            f"TPU readiness identity or state changed for {resource.resource_id}"
        )
    state = state.upper()
    create_time = description.get("createTime")
    if not isinstance(create_time, str):
        raise Experiment5CampaignStateError(
            f"TPU readiness create time is invalid for {resource.resource_id}"
        )
    scheduling = description.get("schedulingConfig")
    if scheduling is None:
        spot = False
    elif not isinstance(scheduling, Mapping) or (
        "spot" in scheduling and type(scheduling.get("spot")) is not bool
    ):
        raise Experiment5CampaignStateError(
            f"TPU readiness scheduling mode is invalid for {resource.resource_id}"
        )
    else:
        spot = scheduling.get("spot") is True
    health = description.get("health")
    if create_time != mapping.get("resource_create_time"):
        return ResourceReadinessObservation(
            status=ResourceReadinessStatus.TERMINAL_REPLACEMENT_REQUIRED,
            state=state,
            health=health if isinstance(health, str) else None,
            create_time=create_time,
            spot=spot,
            absent=False,
            detail=(
                "observed TPU createTime differs from the plan-bound "
                "controller mapping"
            ),
        )
    if state in _TERMINAL_TPU_STATES:
        return ResourceReadinessObservation(
            status=ResourceReadinessStatus.TERMINAL_REPLACEMENT_REQUIRED,
            state=state,
            health=health if isinstance(health, str) else None,
            create_time=create_time,
            spot=spot,
            absent=False,
            detail=f"TPU entered terminal state {state}",
        )
    if state != "READY":
        return ResourceReadinessObservation(
            status=ResourceReadinessStatus.FORMING,
            state=state,
            health=health if isinstance(health, str) else None,
            create_time=create_time,
            spot=spot,
            absent=False,
            detail=f"TPU is not ready: state={state}",
        )
    if not isinstance(health, str) or not health:
        raise Experiment5CampaignStateError(
            f"READY TPU lacks a canonical health state for {resource.resource_id}"
        )
    if health != "HEALTHY":
        return ResourceReadinessObservation(
            status=ResourceReadinessStatus.FORMING,
            state=state,
            health=health,
            create_time=create_time,
            spot=spot,
            absent=False,
            detail=f"TPU is READY but not healthy: health={health}",
        )
    return ResourceReadinessObservation(
        status=ResourceReadinessStatus.READY,
        state=state,
        health=health,
        create_time=create_time,
        spot=spot,
        absent=False,
        detail="exact plan-bound TPU is READY and HEALTHY",
    )


def build_train_command(
    plan: Experiment5CampaignPlan,
    task: TrainingTask,
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
            / f"gemma4_hybrid_{task.panel}_v1.yaml"
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


def build_evaluation_command(
    plan: Experiment5CampaignPlan,
    task: EvaluationTask,
) -> list[str]:
    command = [
        str(plan.python_executable),
        str(plan.working_directory / "scripts/tpu/evaluate_experiment5.py"),
        "--panel",
        task.panel,
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
    plan: Experiment5CampaignPlan,
    resource: CampaignResource,
    task: CampaignTask,
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
        }
    )
    if resource.mirror_seal is not None:
        audited["CONTRIBUTION_REGIONAL_MIRROR_SEAL"] = resource.mirror_seal.remote_path
    if resource.ssh_user is not None:
        audited["TPU_SSH_USER"] = resource.ssh_user
    if isinstance(task, TrainingTask):
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
        if task.panel == "transcoder":
            if (
                task.source_overlay_control_root_remote is None
                or task.source_overlay_object_prefix_gcs is None
            ):
                raise Experiment5CampaignStateError(
                    f"{task.task_id} lost its transcoder source overlay"
                )
            audited.update(
                {
                    "CONTRIBUTION_SOURCE_OVERLAY_ROOT": (
                        task.source_overlay_control_root_remote
                    ),
                    "CONTRIBUTION_SOURCE_OVERLAY_PREFIX": (
                        task.source_overlay_object_prefix_gcs
                    ),
                }
            )
    else:
        audited["CONTRIBUTION_FRESH_CONTROLLER_MAPPING"] = str(
            resource.controller_mapping.controller_path
        )
        audited.update(
            {
                "EXPERIMENT5_RESULT_PATH_CONTROLLER": str(
                    task.completion_path_controller
                ),
                "EXPERIMENT5_PAIRED_SCORES_PATH_CONTROLLER": str(
                    task.paired_scores_path_controller
                ),
            }
        )
        if task.panel == "sae":
            audited["CONTRIBUTION_EVALUATION_LOG_ROOT"] = str(operator_log_root)
        else:
            if any(
                value is None
                for value in (
                    task.train_source_root_remote,
                    task.eval_source_root_remote,
                    task.train_source_object_prefix_gcs,
                    task.eval_source_object_prefix_gcs,
                )
            ):
                raise Experiment5CampaignStateError(
                    f"{task.task_id} lost its transcoder source overlays"
                )
            audited.update(
                {
                    "CONTRIBUTION_SOURCE_OVERLAY_ROOT": (task.train_source_root_remote),
                    "CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT": (
                        task.eval_source_root_remote
                    ),
                    "CONTRIBUTION_SOURCE_OVERLAY_PREFIX": (
                        task.train_source_object_prefix_gcs
                    ),
                    "CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX": (
                        task.eval_source_object_prefix_gcs
                    ),
                }
            )
            audited["TRANSCODER_EVALUATION_LOG_ROOT"] = str(operator_log_root)
    executable = os.environ.copy()
    executable.update(audited)
    return executable, audited


class Experiment5Campaign:
    """Execute the fixed matrix over a seven-resource durable work queue."""

    def __init__(
        self,
        plan: Experiment5CampaignPlan,
        *,
        stop_event: threading.Event | None = None,
        attempt_runner: AttemptRunner | None = None,
        resource_readiness_probe: ResourceReadinessProbe | None = None,
        resource_readiness_poll_seconds: float = RESOURCE_READINESS_POLL_SECONDS,
        verify_checkout: bool = True,
    ) -> None:
        if (
            isinstance(resource_readiness_poll_seconds, bool)
            or not isinstance(resource_readiness_poll_seconds, (int, float))
            or not math.isfinite(float(resource_readiness_poll_seconds))
            or resource_readiness_poll_seconds < 0
        ):
            raise ValueError(
                "resource_readiness_poll_seconds must be finite and nonnegative"
            )
        self.plan = plan
        self.stop_event = stop_event or threading.Event()
        self.attempt_runner = attempt_runner or _default_attempt_runner
        self.resource_readiness_probe = (
            resource_readiness_probe or _default_resource_readiness_probe
        )
        self.resource_readiness_poll_seconds = float(resource_readiness_poll_seconds)
        self.verify_checkout = verify_checkout
        self._lock_handle: Any | None = None
        self._budget_lock = threading.Lock()
        self._campaign_started_at = datetime.now(UTC)
        self._campaign_clock_bound = False
        self._watchdog_shutdown = threading.Event()
        self._watchdog_lock = threading.Lock()
        self._watchdog_error: Experiment5CampaignStateError | None = None
        self._watchdog_thread: threading.Thread | None = None

    def _baseline_attempt_reserve_usd(self) -> float:
        policy = self.plan.cost_policy
        return (
            TRAINING_TASK_COUNT * policy.training_attempt_reserve_usd
            + EVALUATION_TASK_COUNT * policy.evaluation_attempt_reserve_usd
        )

    def _maximum_campaign_wall_seconds(self) -> float:
        """Derive a finite wall bound while preserving the $5 contingency."""

        policy = self.plan.cost_policy
        infrastructure_budget = (
            policy.initial_launch_limit_usd
            - policy.already_spent_usd
            - self._baseline_attempt_reserve_usd()
        )
        if infrastructure_budget + 1e-9 < policy.fixed_cost_reserve_usd:
            raise Experiment5CampaignStateError(
                "campaign cost policy has no infrastructure wall-time reserve"
            )
        return max(
            1.0,
            math.floor(
                infrastructure_budget
                / CONSERVATIVE_INFRASTRUCTURE_BURN_USD_PER_HOUR
                * 3_600
            ),
        )

    def _campaign_elapsed_seconds(self) -> float:
        elapsed = (datetime.now(UTC) - self._campaign_started_at).total_seconds()
        if elapsed < -300:
            raise Experiment5CampaignStateError(
                "campaign clock starts implausibly in the future"
            )
        return max(0.0, elapsed)

    def _bind_campaign_clock(self) -> None:
        path = self.plan.state_root / "campaign-clock.json"
        maximum = self._maximum_campaign_wall_seconds()
        if path.exists():
            observed = _read_record(path, record_type="campaign_clock")
        else:
            started_at = _timestamp()
            observed = _record(
                "campaign_clock",
                {
                    "campaign_id": self.plan.campaign_id,
                    "plan_sha256": self.plan.plan_sha256,
                    "started_at": started_at,
                    "infrastructure_burn_usd_per_hour": (
                        CONSERVATIVE_INFRASTRUCTURE_BURN_USD_PER_HOUR
                    ),
                    "baseline_attempt_reserve_usd": (
                        self._baseline_attempt_reserve_usd()
                    ),
                    "maximum_campaign_wall_seconds": maximum,
                    "protected_launch_limit_usd": (
                        self.plan.cost_policy.initial_launch_limit_usd
                    ),
                },
            )
            _write_create_only(path, observed)
        self._activate_campaign_clock(observed)

    def _activate_campaign_clock(self, observed: Mapping[str, Any]) -> None:
        """Validate and activate one already self-hash-verified clock record."""

        maximum = self._maximum_campaign_wall_seconds()
        if (
            observed.get("campaign_id") != self.plan.campaign_id
            or observed.get("plan_sha256") != self.plan.plan_sha256
            or observed.get("infrastructure_burn_usd_per_hour")
            != CONSERVATIVE_INFRASTRUCTURE_BURN_USD_PER_HOUR
            or observed.get("maximum_campaign_wall_seconds") != maximum
            or observed.get("baseline_attempt_reserve_usd")
            != self._baseline_attempt_reserve_usd()
            or observed.get("protected_launch_limit_usd")
            != self.plan.cost_policy.initial_launch_limit_usd
        ):
            raise Experiment5CampaignStateError("durable campaign clock changed")
        started_at = observed.get("started_at")
        if not isinstance(started_at, str):
            raise Experiment5CampaignStateError(
                "durable campaign clock lacks started_at"
            )
        try:
            parsed = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise Experiment5CampaignStateError(
                "durable campaign clock has invalid started_at"
            ) from error
        if parsed.tzinfo is None:
            raise Experiment5CampaignStateError(
                "durable campaign clock is not timezone-aware"
            )
        self._campaign_started_at = parsed.astimezone(UTC)
        self._campaign_clock_bound = True

    def _watchdog_projection(self) -> dict[str, float]:
        elapsed = self._campaign_elapsed_seconds()
        policy = self.plan.cost_policy
        with self._budget_lock:
            reservations = self._reservation_records()
        actual_attempts = math.fsum(
            float(record["amount_usd"]) for record in reservations
        )
        protected_attempts = max(
            actual_attempts,
            self._baseline_attempt_reserve_usd(),
        )
        accrued_infrastructure = (
            elapsed / 3_600 * CONSERVATIVE_INFRASTRUCTURE_BURN_USD_PER_HOUR
        )
        accounted_infrastructure = max(
            policy.fixed_cost_reserve_usd,
            accrued_infrastructure,
        )
        return {
            "elapsed_seconds": elapsed,
            "maximum_campaign_wall_seconds": (self._maximum_campaign_wall_seconds()),
            "actual_attempt_reservations_usd": actual_attempts,
            "protected_attempt_reservations_usd": protected_attempts,
            "accrued_infrastructure_usd": accrued_infrastructure,
            "accounted_infrastructure_usd": accounted_infrastructure,
            "projected_committed_usd": (
                policy.already_spent_usd + accounted_infrastructure + protected_attempts
            ),
        }

    def _check_campaign_watchdog(self) -> None:
        if not self._campaign_clock_bound:
            return
        projection = self._watchdog_projection()
        policy = self.plan.cost_policy
        if (
            projection["elapsed_seconds"] <= projection["maximum_campaign_wall_seconds"]
            and projection["projected_committed_usd"]
            <= policy.initial_launch_limit_usd + 1e-9
        ):
            return
        self._raise_campaign_watchdog_stop(
            projection,
            reason="campaign_wall_cash_watchdog",
        )

    def _raise_campaign_watchdog_stop(
        self,
        projection: Mapping[str, Any],
        *,
        reason: str,
    ) -> None:
        policy = self.plan.cost_policy
        path = self.plan.state_root / "campaign-wall-budget-stop.json"
        if not path.exists():
            _write_create_only(
                path,
                _record(
                    "budget_stop",
                    {
                        "campaign_id": self.plan.campaign_id,
                        "plan_sha256": self.plan.plan_sha256,
                        "reason": reason,
                        **dict(projection),
                        "protected_launch_limit_usd": (policy.initial_launch_limit_usd),
                        "hard_limit_usd": policy.hard_limit_usd,
                        "stopped_at": _timestamp(),
                    },
                ),
            )
        raise Experiment5BudgetStop(
            "campaign wall/cash watchdog reached its protected $45 limit"
        )

    def _start_watchdog(self) -> None:
        self._check_campaign_watchdog()
        self._watchdog_shutdown.clear()
        with self._watchdog_lock:
            self._watchdog_error = None

        def watch() -> None:
            while not self._watchdog_shutdown.wait(WATCHDOG_POLL_SECONDS):
                try:
                    self._check_campaign_watchdog()
                except Exception as caught:
                    error = (
                        caught
                        if isinstance(caught, Experiment5CampaignStateError)
                        else Experiment5CampaignStateError(
                            "campaign watchdog failed: "
                            f"{type(caught).__name__}: {caught}"
                        )
                    )
                    with self._watchdog_lock:
                        self._watchdog_error = error
                    self.stop_event.set()
                    return

        self._watchdog_thread = threading.Thread(
            target=watch,
            name="experiment5-cash-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _stop_watchdog(self) -> None:
        self._watchdog_shutdown.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=5)
            self._watchdog_thread = None

    def _observed_watchdog_error(
        self,
    ) -> Experiment5CampaignStateError | None:
        with self._watchdog_lock:
            return self._watchdog_error

    def _probe_resource(
        self,
        resource: CampaignResource,
    ) -> ResourceReadinessObservation:
        self._check_campaign_watchdog()
        observation = self.resource_readiness_probe(resource)
        if not isinstance(observation, ResourceReadinessObservation):
            raise Experiment5CampaignStateError(
                f"{resource.resource_id} readiness probe did not return "
                "ResourceReadinessObservation"
            )
        return observation

    def _record_resource_readiness_event(
        self,
        resource: CampaignResource,
        *,
        task: CampaignTask,
        kind: str,
        stage: str,
        observation: ResourceReadinessObservation,
    ) -> None:
        path = (
            self.plan.state_root
            / "resource-readiness"
            / resource.resource_id
            / f"{stage}-{uuid.uuid4().hex}.json"
        )
        _write_create_only(
            path,
            _record(
                "resource_readiness",
                {
                    "campaign_id": self.plan.campaign_id,
                    "plan_sha256": self.plan.plan_sha256,
                    "resource_id": resource.resource_id,
                    "task_id": task.task_id,
                    "kind": kind,
                    "stage": stage,
                    "observation": observation.to_dict(),
                    "observed_at": _timestamp(),
                },
            ),
        )

    def _require_schedulable_observation(
        self,
        resource: CampaignResource,
        *,
        task: CampaignTask,
        kind: str,
        stage: str,
        observation: ResourceReadinessObservation,
    ) -> None:
        if (
            observation.status
            is not ResourceReadinessStatus.TERMINAL_REPLACEMENT_REQUIRED
        ):
            return
        self._record_resource_readiness_event(
            resource,
            task=task,
            kind=kind,
            stage=stage,
            observation=observation,
        )
        raise Experiment5CampaignStateError(
            f"{resource.resource_id} requires replacement before "
            f"{task.task_id}: {observation.detail}"
        )

    def _acquire_lock(self) -> None:
        self.plan.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = (self.plan.state_root / "controller.lock").open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise Experiment5CampaignStateError(
                "another Experiment-5 controller owns the state root"
            ) from error
        self._lock_handle = handle

    def _release_lock(self) -> None:
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None

    def _bind_plan(self) -> None:
        binding_path = self.plan.state_root / "plan-binding.json"
        expected = _record(
            "plan_binding",
            {
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "plan": self.plan.source,
            },
        )
        if binding_path.exists():
            if _read_record(binding_path, record_type="plan_binding") != expected:
                raise Experiment5CampaignStateError(
                    "state root is bound to another campaign plan"
                )
        else:
            _write_create_only(binding_path, expected)
        ownership = {
            resource.resource_id: {
                "project": resource.project,
                "zone": resource.zone,
                "tpu_name": resource.tpu_name,
                "accelerator_type": resource.accelerator_type,
                "cache_profile": resource.cache_profile,
                "eligible_tasks": sorted(
                    task.task_id
                    for task in (*self.plan.trajectories, *self.plan.evaluations)
                    if resource.resource_id in task.eligible_resource_ids
                ),
            }
            for resource in self.plan.resources
        }
        ownership_path = self.plan.state_root / "resource-ownership.json"
        expected_ownership = _record(
            "resource_ownership",
            {
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "policy": "one-resource-worker-one-attempt-at-a-time",
                "ownership": ownership,
            },
        )
        if ownership_path.exists():
            if (
                _read_record(ownership_path, record_type="resource_ownership")
                != expected_ownership
            ):
                raise Experiment5CampaignStateError("resource ownership ledger changed")
        else:
            _write_create_only(ownership_path, expected_ownership)

    def _task_root(self, task_id: str, *, kind: str) -> Path:
        return self.plan.state_root / kind / task_id

    def _inspector(self, task: CampaignTask) -> CompletionFingerprint | None:
        if isinstance(task, TrainingTask):
            return inspect_training_completion(task)
        return inspect_evaluation_completion(self.plan, task)

    def _completion_witness(
        self,
        task: CampaignTask,
        *,
        kind: str,
        fingerprint: CompletionFingerprint,
    ) -> None:
        path = self._task_root(task.task_id, kind=kind) / "completion.json"
        expected = _record(
            "task_completion",
            {
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "task_id": task.task_id,
                "kind": kind,
                "completion": fingerprint.to_dict(),
            },
        )
        if path.exists():
            if _read_record(path, record_type="task_completion") != expected:
                raise Experiment5CampaignStateError(
                    f"completion witness changed for {task.task_id}"
                )
        else:
            _write_create_only(path, expected)

    def _prior_state(self, task: CampaignTask, *, kind: str) -> tuple[int, int]:
        root = self._task_root(task.task_id, kind=kind)
        failure = root / "failure.json"
        if failure.exists():
            record = _read_record(failure, record_type="task_failure")
            raise Experiment5CampaignStateError(
                f"{task.task_id} has terminal failure "
                f"{record.get('classification')}"
            )
        attempts = root / "attempts"
        maximum = 0
        transient_count = 0
        if not attempts.exists():
            return maximum, transient_count
        for attempt in sorted(attempts.iterdir()):
            if not attempt.is_dir() or not attempt.name.startswith("attempt-"):
                raise Experiment5CampaignStateError(
                    f"unexpected attempt state under {attempts}"
                )
            try:
                number = int(attempt.name.removeprefix("attempt-"))
            except ValueError as error:
                raise Experiment5CampaignStateError(
                    f"invalid attempt directory {attempt}"
                ) from error
            maximum = max(maximum, number)
            intent = attempt / "intent.json"
            result = attempt / "result.json"
            if not intent.is_file():
                raise Experiment5CampaignStateError(
                    f"attempt lacks immutable intent: {attempt}"
                )
            _read_record(intent, record_type="attempt_intent")
            if not result.exists():
                raise Experiment5CampaignStateError(
                    f"{task.task_id} has an unobserved attempt {attempt.name}; "
                    "reconcile its remote owner before retrying"
                )
            record = _read_record(result, record_type="attempt_result")
            classification = record.get("classification")
            if classification in RESUMABLE_FAILURES:
                if classification in TRANSIENT_FAILURES:
                    transient_count += 1
            elif classification != "completed":
                raise Experiment5CampaignStateError(
                    f"{task.task_id} has prior terminal attempt "
                    f"{attempt.name}: {classification}"
                )
        return maximum, transient_count

    def _reservation_records(self) -> list[dict[str, Any]]:
        root = self.plan.state_root / "budget" / "attempt-reservations"
        if not root.exists():
            return []
        records: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*.json")):
            records.append(_read_record(path, record_type="cost_reservation"))
        return records

    def cost_projection(
        self,
        *,
        campaign_elapsed_seconds: float | None = None,
    ) -> dict[str, Any]:
        policy = self.plan.cost_policy
        reservations = self._reservation_records()
        attempt_total = math.fsum(
            float(record["amount_usd"]) for record in reservations
        )
        if campaign_elapsed_seconds is None:
            elapsed = (
                self._campaign_elapsed_seconds() if self._campaign_clock_bound else 0.0
            )
        else:
            if (
                isinstance(campaign_elapsed_seconds, bool)
                or not isinstance(campaign_elapsed_seconds, (int, float))
                or not math.isfinite(float(campaign_elapsed_seconds))
                or campaign_elapsed_seconds < 0
            ):
                raise Experiment5CampaignStateError(
                    "cost projection elapsed time is invalid"
                )
            elapsed = float(campaign_elapsed_seconds)
        accrued_infrastructure = (
            elapsed / 3_600 * CONSERVATIVE_INFRASTRUCTURE_BURN_USD_PER_HOUR
        )
        accounted_infrastructure = max(
            policy.fixed_cost_reserve_usd,
            accrued_infrastructure,
        )
        committed = policy.already_spent_usd + accounted_infrastructure + attempt_total
        return {
            "currency": "USD",
            "hard_limit_usd": policy.hard_limit_usd,
            "contingency_usd": policy.contingency_usd,
            "already_spent_usd": policy.already_spent_usd,
            "fixed_cost_reserve_usd": policy.fixed_cost_reserve_usd,
            "infrastructure_burn_usd_per_hour": (
                CONSERVATIVE_INFRASTRUCTURE_BURN_USD_PER_HOUR
            ),
            "campaign_elapsed_seconds": elapsed,
            "maximum_campaign_wall_seconds": (self._maximum_campaign_wall_seconds()),
            "accrued_infrastructure_usd": accrued_infrastructure,
            "accounted_infrastructure_usd": accounted_infrastructure,
            "attempt_reservations_usd": attempt_total,
            "projected_committed_usd": committed,
            "remaining_to_hard_limit_usd": policy.hard_limit_usd - committed,
            "attempt_reservations": reservations,
        }

    def _reserve_attempt_cost(
        self,
        task: CampaignTask,
        *,
        kind: str,
        attempt_number: int,
        resource: CampaignResource,
    ) -> dict[str, Any]:
        self._check_campaign_watchdog()
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
                observed = _read_record(path, record_type="cost_reservation")
                if (
                    observed.get("task_id") != task.task_id
                    or observed.get("kind") != kind
                    or observed.get("attempt_number") != attempt_number
                    or observed.get("eligible_resource_ids")
                    != list(task.eligible_resource_ids)
                ):
                    raise Experiment5CampaignStateError(
                        f"cost reservation changed for {task.task_id}"
                    )
                return observed
            amount = self.plan.cost_policy.attempt_reserve(kind)
            projection = self.cost_projection()
            before = projection["projected_committed_usd"]
            after = float(before) + amount
            if self._campaign_clock_bound:
                prospective_attempts = (
                    float(projection["attempt_reservations_usd"]) + amount
                )
                protected_attempts = max(
                    prospective_attempts,
                    self._baseline_attempt_reserve_usd(),
                )
                protected_after = (
                    self.plan.cost_policy.already_spent_usd
                    + float(projection["accounted_infrastructure_usd"])
                    + protected_attempts
                )
                if (
                    protected_after
                    > self.plan.cost_policy.initial_launch_limit_usd + 1e-9
                ):
                    self._raise_campaign_watchdog_stop(
                        {
                            "elapsed_seconds": projection["campaign_elapsed_seconds"],
                            "maximum_campaign_wall_seconds": projection[
                                "maximum_campaign_wall_seconds"
                            ],
                            "actual_attempt_reservations_usd": (prospective_attempts),
                            "protected_attempt_reservations_usd": (protected_attempts),
                            "accrued_infrastructure_usd": projection[
                                "accrued_infrastructure_usd"
                            ],
                            "accounted_infrastructure_usd": projection[
                                "accounted_infrastructure_usd"
                            ],
                            "projected_committed_usd": protected_after,
                        },
                        reason="prospective_attempt_crosses_protected_limit",
                    )
            if after > self.plan.cost_policy.hard_limit_usd + 1e-9:
                stop_path = self.plan.state_root / "budget-stop.json"
                stop = _record(
                    "budget_stop",
                    {
                        "campaign_id": self.plan.campaign_id,
                        "plan_sha256": self.plan.plan_sha256,
                        "task_id": task.task_id,
                        "kind": kind,
                        "attempt_number": attempt_number,
                        "resource_id": resource.resource_id,
                        "projected_before_usd": before,
                        "requested_reserve_usd": amount,
                        "projected_after_usd": after,
                        "hard_limit_usd": self.plan.cost_policy.hard_limit_usd,
                        "stopped_at": _timestamp(),
                    },
                )
                if not stop_path.exists():
                    _write_create_only(stop_path, stop)
                raise Experiment5BudgetStop(
                    f"{task.task_id} would project ${after:.2f}, above $50.00"
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
                    "eligible_resource_ids": list(task.eligible_resource_ids),
                    "amount_usd": amount,
                    "projected_committed_after_usd": after,
                    "reserved_at": _timestamp(),
                },
            )
            _write_create_only(path, reservation)
            return reservation

    def _new_attempt(
        self,
        task: CampaignTask,
        *,
        kind: str,
        number: int,
        attempt_id: str,
        command: Sequence[str],
        audited_environment: Mapping[str, str],
        resource: CampaignResource,
        pre_attempt_readiness: ResourceReadinessObservation,
    ) -> Path:
        root = (
            self._task_root(task.task_id, kind=kind)
            / "attempts"
            / f"attempt-{number:06d}"
        )
        intent = _record(
            "attempt_intent",
            {
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "task_id": task.task_id,
                "kind": kind,
                "attempt_number": number,
                "attempt_id": attempt_id,
                "resource_id": resource.resource_id,
                "resource_identity": {
                    "project": resource.project,
                    "zone": resource.zone,
                    "tpu_name": resource.tpu_name,
                    "accelerator_type": resource.accelerator_type,
                    "cache_profile": resource.cache_profile,
                    "deployed_commit": resource.deployed_commit,
                },
                "pre_attempt_readiness": pre_attempt_readiness.to_dict(),
                "command": list(command),
                "command_sha256": canonical_sha256(list(command)),
                "environment_sha256": canonical_sha256(
                    dict(sorted(audited_environment.items()))
                ),
                "started_at": _timestamp(),
            },
        )
        _write_create_only(root / "intent.json", intent)
        return root

    def _terminal_failure(
        self,
        task: CampaignTask,
        *,
        kind: str,
        classification: str,
        attempt_number: int,
        detail: str,
    ) -> None:
        path = self._task_root(task.task_id, kind=kind) / "failure.json"
        if not path.exists():
            _write_create_only(
                path,
                _record(
                    "task_failure",
                    {
                        "campaign_id": self.plan.campaign_id,
                        "plan_sha256": self.plan.plan_sha256,
                        "task_id": task.task_id,
                        "kind": kind,
                        "classification": classification,
                        "attempt_number": attempt_number,
                        "detail": detail[-4_096:],
                        "failed_at": _timestamp(),
                    },
                ),
            )

    def _execute_once(
        self,
        task: CampaignTask,
        *,
        kind: str,
        resource: CampaignResource,
        pre_attempt_readiness: ResourceReadinessObservation,
    ) -> _AttemptOutcome:
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
        environment, audited = _task_environment(
            self.plan,
            resource,
            task,
            provisional / "operator",
        )
        attempt_id = f"e5-a{number:08d}-{uuid.uuid4().hex[:16]}"
        command = (
            build_train_command(self.plan, task, attempt_id)
            if isinstance(task, TrainingTask)
            else build_evaluation_command(self.plan, task)
        )
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
        post_failure_readiness: ResourceReadinessObservation | None = None
        confirmed_preemption = False
        if (
            completion is None
            and classification in TRANSIENT_FAILURES
            and pre_attempt_readiness.status is ResourceReadinessStatus.READY
            and not self.stop_event.is_set()
        ):
            try:
                observed = self.resource_readiness_probe(resource)
                if not isinstance(observed, ResourceReadinessObservation):
                    raise TypeError(
                        "readiness probe did not return " "ResourceReadinessObservation"
                    )
                post_failure_readiness = observed
            except Exception as error:
                # The attempt itself must still receive one terminal result.
                # An unavailable follow-up probe is not evidence of preemption.
                post_failure_readiness = ResourceReadinessObservation(
                    status=(ResourceReadinessStatus.TRANSIENT_CONTROL_PLANE),
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
                post_failure_readiness.status
                is ResourceReadinessStatus.TERMINAL_REPLACEMENT_REQUIRED
                or (
                    post_failure_readiness.status is ResourceReadinessStatus.FORMING
                    and post_failure_readiness.absent
                )
            )
            if pre_attempt_readiness.spot is True and confirmed_loss:
                classification = "spot_preemption"
                confirmed_preemption = True
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
                "classification": classification,
                "pre_attempt_readiness": pre_attempt_readiness.to_dict(),
                "post_failure_readiness": (
                    post_failure_readiness.to_dict()
                    if post_failure_readiness is not None
                    else None
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
        if post_failure_readiness is not None and (
            confirmed_preemption
            or post_failure_readiness.status
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
                observation=post_failure_readiness,
            )
            raise Experiment5CampaignStateError(
                f"{resource.resource_id} requires replacement after "
                f"{task.task_id}: {post_failure_readiness.detail}"
            )
        if completion is not None:
            self._completion_witness(task, kind=kind, fingerprint=completion)
            return _AttemptOutcome(completion)
        if classification == "controller_stop":
            raise Experiment5CampaignStateError(
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
            raise Experiment5CampaignStateError(
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
                raise Experiment5CampaignStateError(
                    f"{task.task_id} exhausted transient retries"
                )
        delay = min(
            self.plan.retry_policy.maximum_backoff_seconds,
            self.plan.retry_policy.initial_backoff_seconds
            * (2 ** max(0, transient_count - 1)),
        )
        return _AttemptOutcome(None, retry_delay_seconds=delay)

    def _run_phase(
        self,
        tasks: Sequence[CampaignTask],
        *,
        kind: str,
        max_parallel: int,
    ) -> dict[str, CompletionFingerprint]:
        completed: dict[str, CompletionFingerprint] = {}
        pending: dict[str, tuple[CampaignTask, float]] = {}
        for task in tasks:
            existing = self._inspector(task)
            if existing is None:
                pending[task.task_id] = (task, 0.0)
            else:
                self._completion_witness(task, kind=kind, fingerprint=existing)
                completed[task.task_id] = existing
        if not pending:
            return completed

        condition = threading.Condition()
        running: set[str] = set()
        failures: list[str] = []
        abort = False
        semaphore = threading.Semaphore(max_parallel)
        active_resource_ids = {
            resource_id for task in tasks for resource_id in task.eligible_resource_ids
        }
        active_resources = tuple(
            resource
            for resource in self.plan.resources
            if resource.resource_id in active_resource_ids
        )
        if not active_resources:
            raise Experiment5CampaignStateError(f"{kind} phase has no active resources")

        def resource_worker(resource: CampaignResource) -> None:
            nonlocal abort
            while True:
                with condition:
                    while True:
                        if abort or self.stop_event.is_set():
                            return
                        if not pending and not running:
                            return
                        now = time.monotonic()
                        choices = [
                            (task, ready)
                            for task, ready in pending.values()
                            if resource.resource_id in task.eligible_resource_ids
                            and ready <= now
                        ]
                        if choices:
                            task = min(
                                (item[0] for item in choices),
                                key=lambda item: (
                                    len(item.eligible_resource_ids),
                                    item.task_id,
                                ),
                            )
                            del pending[task.task_id]
                            running.add(task.task_id)
                            break
                        eligible_ready = [
                            ready
                            for task, ready in pending.values()
                            if resource.resource_id in task.eligible_resource_ids
                        ]
                        timeout = (
                            max(0.01, min(eligible_ready) - now)
                            if eligible_ready
                            else 1.0
                        )
                        condition.wait(timeout=min(timeout, 1.0))
                outcome: _AttemptOutcome | None = None
                error: Exception | None = None
                observation: ResourceReadinessObservation | None = None
                cooldown = False
                try:
                    observation = self._probe_resource(resource)
                    self._require_schedulable_observation(
                        resource,
                        task=task,
                        kind=kind,
                        stage="before-semaphore",
                        observation=observation,
                    )
                    if observation.status is ResourceReadinessStatus.READY:
                        with semaphore:
                            # Readiness can change while this worker waits for
                            # the campaign-wide parallelism slot.
                            observation = self._probe_resource(resource)
                            self._require_schedulable_observation(
                                resource,
                                task=task,
                                kind=kind,
                                stage="inside-semaphore",
                                observation=observation,
                            )
                            if observation.status is ResourceReadinessStatus.READY:
                                outcome = self._execute_once(
                                    task,
                                    kind=kind,
                                    resource=resource,
                                    pre_attempt_readiness=observation,
                                )
                except Exception as caught:
                    if isinstance(caught, Experiment5CampaignStateError):
                        error = caught
                    else:
                        error = Experiment5CampaignStateError(
                            f"{task.task_id} resource worker failed: "
                            f"{type(caught).__name__}: {caught}"
                        )
                finally:
                    with condition:
                        running.discard(task.task_id)
                        if error is not None:
                            failures.append(f"{task.task_id}: {error}")
                            abort = True
                        elif (
                            observation is None
                            or observation.status is not ResourceReadinessStatus.READY
                        ):
                            # Formation and temporary control-plane loss are
                            # scheduling states. They do not create an attempt
                            # or reserve attempt cost.
                            pending[task.task_id] = (task, time.monotonic())
                            cooldown = True
                        elif outcome is not None and outcome.completion is not None:
                            completed[task.task_id] = outcome.completion
                        elif outcome is not None:
                            pending[task.task_id] = (
                                task,
                                time.monotonic() + outcome.retry_delay_seconds,
                            )
                        else:
                            error = Experiment5CampaignStateError(
                                f"{task.task_id} worker produced no outcome"
                            )
                            failures.append(f"{task.task_id}: {error}")
                            abort = True
                        condition.notify_all()
                if error is not None:
                    return
                if cooldown and self.stop_event.wait(
                    self.resource_readiness_poll_seconds
                ):
                    return

        with ThreadPoolExecutor(
            max_workers=len(active_resources),
            thread_name_prefix=f"experiment5-{kind}",
        ) as pool:
            futures = [
                pool.submit(resource_worker, resource) for resource in active_resources
            ]
            for future in futures:
                future.result()
        watchdog_error = self._observed_watchdog_error()
        if watchdog_error is not None:
            raise watchdog_error
        if failures:
            raise Experiment5CampaignStateError(
                f"{kind} campaign stopped: " + "; ".join(sorted(failures))
            )
        if self.stop_event.is_set() or len(completed) != len(tasks):
            raise Experiment5CampaignStateError(
                f"{kind} campaign stopped before all immutable completions"
            )
        return completed

    def _seal_phase(
        self,
        *,
        path: Path,
        record_type: str,
        status: str,
        field: str,
        completions: Mapping[str, CompletionFingerprint],
    ) -> dict[str, Any]:
        expected = _record(
            record_type,
            {
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "status": status,
                field: {key: completions[key].to_dict() for key in sorted(completions)},
                "completed_at": _timestamp(),
            },
        )
        if path.exists():
            observed = _read_record(path, record_type=record_type)
            for key in ("campaign_id", "plan_sha256", "status", field):
                if observed.get(key) != expected.get(key):
                    raise Experiment5CampaignStateError(f"{record_type} record changed")
            return observed
        _write_create_only(path, expected)
        return expected

    def _seal_scientific_report(
        self,
        *,
        evaluation_barrier: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Seal the complete budget ledger and scientific report idempotently."""

        barrier_rows = evaluation_barrier.get("evaluations")
        expected_task_ids = {task.task_id for task in self.plan.evaluations}
        if (
            evaluation_barrier.get("campaign_id") != self.plan.campaign_id
            or evaluation_barrier.get("plan_sha256") != self.plan.plan_sha256
            or evaluation_barrier.get("status") != "evaluation_completed"
            or not isinstance(barrier_rows, Mapping)
            or set(barrier_rows) != expected_task_ids
            or len(expected_task_ids) != EVALUATION_TASK_COUNT
        ):
            raise Experiment5CampaignStateError(
                "evaluation barrier is not the exact 18-task report matrix"
            )

        ordered_tasks = sorted(
            self.plan.evaluations,
            key=lambda task: (task.panel, task.seed, task.milestone_tokens),
        )
        result_paths: list[Path] = []
        paired_paths: list[Path] = []
        validated_results = []
        try:
            for task in ordered_tasks:
                current = inspect_evaluation_completion(self.plan, task)
                if current is None or current.to_dict() != barrier_rows.get(
                    task.task_id
                ):
                    raise Experiment5CampaignStateError(
                        f"evaluation changed after phase seal: {task.task_id}"
                    )
                value, _payload = _read_self_hashed_json(
                    task.completion_path_controller,
                    hash_field="result_sha256",
                )
                result_paths.append(task.completion_path_controller)
                paired_paths.append(task.paired_scores_path_controller)
                validated_results.append(validate_experiment5_result(value))
        except Experiment5ReportError as error:
            raise Experiment5CampaignStateError(
                f"Experiment-5 evaluation cannot be reported: {error}"
            ) from error

        expected_final_identities = {
            (panel, seed, TRAIN_TOKENS) for panel in PANELS for seed in SEEDS
        }
        final_rows = {
            (row.panel, row.seed, row.milestone): row
            for row in validated_results
            if row.milestone == TRAIN_TOKENS
        }
        if set(final_rows) != expected_final_identities or len(final_rows) != len(
            expected_final_identities
        ):
            raise Experiment5CampaignStateError(
                "quota accounting lacks the exact six 10M final results"
            )
        quota_funded_training_chip_seconds = math.fsum(
            final_rows[(panel, seed, TRAIN_TOKENS)].costs[condition][
                "marginal_chip_seconds"
            ]
            for panel in PANELS
            for seed in SEEDS
            for condition in CONDITIONS
        )

        clock = _read_record(
            self.plan.state_root / "campaign-clock.json",
            record_type="campaign_clock",
        )
        started_at = clock.get("started_at")
        completed_at = evaluation_barrier.get("completed_at")
        if not isinstance(started_at, str) or not isinstance(completed_at, str):
            raise Experiment5CampaignStateError(
                "report accounting timestamps are absent"
            )
        try:
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            completed = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise Experiment5CampaignStateError(
                "report accounting timestamps are invalid"
            ) from error
        if started.tzinfo is None or completed.tzinfo is None:
            raise Experiment5CampaignStateError(
                "report accounting timestamps are not timezone-aware"
            )
        elapsed_seconds = (completed - started).total_seconds()
        if elapsed_seconds < 0:
            raise Experiment5CampaignStateError(
                "evaluation completed before the durable campaign clock started"
            )
        infrastructure_usd = max(
            self.plan.cost_policy.fixed_cost_reserve_usd,
            elapsed_seconds / 3_600 * CONSERVATIVE_INFRASTRUCTURE_BURN_USD_PER_HOUR,
        )
        try:
            ledger = build_experiment5_budget_ledger(
                line_items=[
                    {
                        "category": "already_spent_before_durable_campaign",
                        "amount_usd": self.plan.cost_policy.already_spent_usd,
                    },
                    {
                        "category": (
                            "durable_campaign_infrastructure_through_"
                            "evaluation_completion"
                        ),
                        "amount_usd": infrastructure_usd,
                    },
                ],
                quota_funded_tpu_chip_seconds=(quota_funded_training_chip_seconds),
            )
            validate_experiment5_budget_ledger(ledger)
        except Experiment5ReportError as error:
            raise Experiment5CampaignStateError(
                f"Experiment-5 budget ledger is invalid: {error}"
            ) from error

        report_root = self.plan.state_root / "report"
        ledger_path = report_root / "budget-ledger.json"
        ledger_payload = _canonical_json_bytes(ledger) + b"\n"
        ledger_file_sha256 = _seal_exact_bytes(
            ledger_path,
            ledger_payload,
            name="Experiment-5 budget ledger",
        )

        try:
            report = build_experiment5_report(
                results=result_paths,
                paired_scores=paired_paths,
                budget_ledger=ledger_path,
            )
            validate_experiment5_report(report)
            markdown = render_experiment5_report(report)
        except Experiment5ReportError as error:
            raise Experiment5CampaignStateError(
                f"Experiment-5 report is invalid: {error}"
            ) from error

        report_path = report_root / "experiment5-result.json"
        report_file_sha256 = _seal_exact_bytes(
            report_path,
            _canonical_json_bytes(report) + b"\n",
            name="Experiment-5 JSON report",
        )
        markdown_path = report_root / "EXPERIMENT_5.md"
        markdown_file_sha256 = _seal_exact_bytes(
            markdown_path,
            markdown.encode("utf-8"),
            name="Experiment-5 Markdown report",
        )
        return {
            "budget_ledger_path": str(ledger_path),
            "budget_ledger_sha256": ledger["ledger_sha256"],
            "budget_ledger_file_sha256": ledger_file_sha256,
            "report_path": str(report_path),
            "report_sha256": report["report_sha256"],
            "report_file_sha256": report_file_sha256,
            "markdown_path": str(markdown_path),
            "markdown_file_sha256": markdown_file_sha256,
            "evaluation_completion_record_sha256": evaluation_barrier["record_sha256"],
            "quota_funded_tpu_accounting": {
                "training_chip_seconds": quota_funded_training_chip_seconds,
                "training_scope": (
                    "10M cumulative marginal training time; both panels, "
                    "three seeds, and H/R/B/F exactly once"
                ),
                "evaluation_chip_seconds": None,
                "evaluation_scope": (
                    "absent from the Experiment-5 scientific-result contract"
                ),
            },
        }

    def _seal_cost_projection(self, *, completed_at: str) -> dict[str, Any]:
        path = self.plan.state_root / "cost-projection.json"
        try:
            completed = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as error:
            raise Experiment5CampaignStateError(
                "cost projection completion timestamp is invalid"
            ) from error
        if completed.tzinfo is None:
            raise Experiment5CampaignStateError(
                "cost projection completion timestamp is not timezone-aware"
            )
        elapsed = (
            completed.astimezone(UTC) - self._campaign_started_at
        ).total_seconds()
        if elapsed < 0:
            raise Experiment5CampaignStateError(
                "cost projection completion predates the campaign clock"
            )
        projection = self.cost_projection(campaign_elapsed_seconds=elapsed)
        expected = _record(
            "cost_projection",
            {
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                **projection,
                "sealed_at": completed_at,
            },
        )
        if path.exists():
            observed = _read_record(path, record_type="cost_projection")
            for key in (
                "campaign_id",
                "plan_sha256",
                "currency",
                "hard_limit_usd",
                "already_spent_usd",
                "fixed_cost_reserve_usd",
                "infrastructure_burn_usd_per_hour",
                "maximum_campaign_wall_seconds",
                "attempt_reservations_usd",
                "attempt_reservations",
            ):
                if observed.get(key) != expected.get(key):
                    raise Experiment5CampaignStateError(
                        "sealed cost projection changed"
                    )
            return observed
        _write_create_only(path, expected)
        return expected

    def run(self, *, stop_after_training: bool = False) -> dict[str, Any]:
        _preflight(self.plan, verify_checkout=self.verify_checkout)
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
            evaluations = self._run_phase(
                self.plan.evaluations,
                kind="evaluation",
                max_parallel=self.plan.max_parallel_eval,
            )
            evaluation_barrier = self._seal_phase(
                path=self.plan.state_root / "evaluation-complete.json",
                record_type="evaluation_phase_completion",
                status="evaluation_completed",
                field="evaluations",
                completions=evaluations,
            )
            report_identity = self._seal_scientific_report(
                evaluation_barrier=evaluation_barrier,
            )
            cost = self._seal_cost_projection(
                completed_at=evaluation_barrier["completed_at"],
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
                        key: evaluations[key].to_dict() for key in sorted(evaluations)
                    },
                    "report": report_identity,
                    "cost_projection_record_sha256": cost["record_sha256"],
                    "completed_at": evaluation_barrier["completed_at"],
                },
            )
            path = self.plan.state_root / "final.json"
            if path.exists():
                observed = _read_record(path, record_type="campaign_completion")
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
                        raise Experiment5CampaignStateError(
                            "final campaign record changed"
                        )
                return observed
            _write_create_only(path, final)
            return final
        finally:
            self._stop_watchdog()
            self._release_lock()


def experiment5_campaign_status(
    plan: Experiment5CampaignPlan,
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task in (*plan.trajectories, *plan.evaluations):
        kind = "training" if isinstance(task, TrainingTask) else "evaluation"
        completion = (
            inspect_training_completion(task)
            if isinstance(task, TrainingTask)
            else inspect_evaluation_completion(plan, task)
        )
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
            "eligible_resources": list(task.eligible_resource_ids),
            "completion": completion.to_dict() if completion else None,
        }
    controller = Experiment5Campaign(plan, verify_checkout=False)
    clock_path = plan.state_root / "campaign-clock.json"
    campaign_clock = (
        _read_record(clock_path, record_type="campaign_clock")
        if clock_path.exists()
        else None
    )
    if campaign_clock is not None:
        controller._activate_campaign_clock(campaign_clock)
    wall_stop_path = plan.state_root / "campaign-wall-budget-stop.json"
    campaign_wall_budget_stop = (
        _read_record(wall_stop_path, record_type="budget_stop")
        if wall_stop_path.exists()
        else None
    )
    if campaign_wall_budget_stop is not None and campaign_clock is None:
        raise Experiment5CampaignStateError(
            "campaign wall budget stop exists without its durable clock"
        )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.plan_sha256,
        "tasks": tasks,
        "campaign_clock": campaign_clock,
        "cost_projection": controller.cost_projection(),
        "watchdog_projection": (
            controller._watchdog_projection() if campaign_clock is not None else None
        ),
        "budget_stop": (
            _read_record(
                plan.state_root / "budget-stop.json", record_type="budget_stop"
            )
            if (plan.state_root / "budget-stop.json").exists()
            else None
        ),
        "campaign_wall_budget_stop": campaign_wall_budget_stop,
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
            _read_record(
                plan.state_root / "final.json",
                record_type="campaign_completion",
            )
            if (plan.state_root / "final.json").exists()
            else None
        ),
    }


__all__ = [
    "CONDITIONS",
    "CONTINUOUS_RUNTIME_SECONDS",
    "EVALUATION_TASK_COUNT",
    "MILESTONES",
    "PANELS",
    "PLAN_SCHEMA_VERSION",
    "RESOURCE_COUNT",
    "SEEDS",
    "TRAINING_TASK_COUNT",
    "AttemptExecution",
    "CampaignResource",
    "CompletionFingerprint",
    "CostPolicy",
    "EvaluationTask",
    "Experiment5BudgetStop",
    "Experiment5Campaign",
    "Experiment5CampaignPlan",
    "Experiment5CampaignPlanError",
    "Experiment5CampaignStateError",
    "TrainingTask",
    "build_evaluation_command",
    "build_train_command",
    "experiment5_campaign_status",
    "inspect_evaluation_completion",
    "inspect_training_completion",
    "load_experiment5_campaign_plan",
    "parse_experiment5_campaign_plan",
]
