"""Durable controller for the fixed Gemma learning-horizon campaign.

The numerical workers and TPU operators remain separate programs.  This
module supplies only the cloud-resident control loop around them:

* validate and bind one explicit launch DAG;
* reserve globally increasing, create-only attempt identities;
* recognize terminal work only through immutable completion files;
* run each trajectory continuously, resuming its rolling checkpoints after
  Spot preemption or a genuine twelve-hour crash boundary;
* retry only classified Spot/resource/SSH transport losses; and
* reconcile private regional 16-chip Spot TPU resources without list polling.

The controller is intentionally POSIX-local.  Its state root must be on the
small strongly-consistent control filesystem, while captures, mirrors, and
checkpoint payloads stay in the bulk object-store data plane.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from moe_sae.contribution_artifacts import canonical_json_bytes, canonical_sha256
from moe_sae.runtime import REGIONAL_ACCELERATOR_TYPES

PLAN_SCHEMA_VERSION = "gemma4-horizon-launch-plan-v1"
ATTEMPT_INTENT_SCHEMA_VERSION = "gemma4-horizon-attempt-intent-v1"
ATTEMPT_RESULT_SCHEMA_VERSION = "gemma4-horizon-attempt-result-v1"
COMPLETION_WITNESS_SCHEMA_VERSION = "gemma4-horizon-completion-witness-v1"
TASK_FAILURE_SCHEMA_VERSION = "gemma4-horizon-task-failure-v1"
RESOURCE_BINDING_SCHEMA_VERSION = "gemma4-horizon-resource-binding-v1"
RESOURCE_RELEASE_SCHEMA_VERSION = "gemma4-horizon-resource-release-v1"
FINAL_SCHEMA_VERSION = "gemma4-horizon-final-v1"

CONDITIONS = ("A", "R", "B", "F")
SCIENTIFIC_MODELS = ("sae", "transcoder")
MILESTONES = (2_000_000, 5_000_000, 10_000_000)
TASK_KINDS = ("capture", "overlay", "mirror", "train", "eval", "report")
ALLOCATION_MODES = ("queued_spot", "direct_spot")
DEFAULT_TASK_TIMEOUT_SECONDS = 3_600
MAX_TRAIN_INVOCATION_SECONDS = 43_200

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_SAFE_CLOUD_NAME = re.compile(r"[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?")
_PROJECT = re.compile(r"[a-z][a-z0-9-]{4,61}[a-z0-9]")
_ZONE = re.compile(r"[a-z]+-[a-z0-9]+[0-9]-[a-z]")
_EMAIL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9.-]+")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ATTEMPT_DIRECTORY = re.compile(r"attempt-(?P<number>[0-9]{8})-(?P<task>.+)")
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
_SECRET_NAME = re.compile(r"(?:TOKEN|PASSWORD|SECRET|CREDENTIAL|PRIVATE_KEY)", re.I)

_ALLOWED_PLACEHOLDERS = frozenset(
    {
        "attempt_id",
        "attempt_number",
        "campaign_id",
        "plan_sha256",
        "queued_resource",
        "resource_id",
        "segment_seconds",
        "task_id",
        "tpu_name",
        "tpu_zone",
    }
)
_RESOURCE_PLACEHOLDERS = frozenset(
    {"queued_resource", "resource_id", "tpu_name", "tpu_zone"}
)

_NUMERICAL_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"\b(?:nan|inf)\b.*(?:loss|gradient|parameter|activation)",
        r"(?:loss|gradient|parameter|activation).*\b(?:nan|inf)\b",
        r"non[- ]finite",
        r"numerical (?:failure|error)",
        r"fewer than .* positive",
        r"exact l0",
    )
)
_CORRUPTION_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"(?:sha-?256|hash|crc32c).*(?:mismatch|changed|invalid)",
        r"(?:corrupt|corruption|schema mismatch)",
        r"(?:duplicate|missing) (?:range|token|shard)",
        r"(?:token|routed).*(?:drop|overflow)",
        r"manifest.*(?:invalid|changed|mismatch)",
        r"identity.*(?:changed|mismatch)",
    )
)
_SPOT_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"\bpreempt(?:ed|ion)?\b",
        r"\bspot\b.*\b(?:suspend|terminate|preempt|evict)",
        r"\b(?:suspending|suspended|terminating|terminated)\b.*\btpu\b",
        r"\bvm_terminated\b",
    )
)
_RESOURCE_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"zone_resource_pool_exhausted",
        r"resource[_ ]availability",
        r"does not have enough resources available",
        r"currently unavailable in the .* zone",
        r"queued resource.*(?:failed|suspend|cancel)",
        r"tpu.*(?:not found|was not found)",
    )
)
_SSH_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"connection (?:timed out|refused|reset|closed)",
        r"no route to host",
        r"could not resolve hostname",
        r"failed to connect to (?:backend|instance|tpu)",
        r"ssh_exchange_identification",
        r"kex_exchange_identification",
        r"broken pipe",
        r"serveralivecountmax",
        r"transport endpoint",
    )
)


class HorizonPlanError(ValueError):
    """The launch plan is ambiguous, unsafe, or scientifically incomplete."""


class HorizonStateError(RuntimeError):
    """Durable state is missing, inconsistent, or no longer immutable."""


class HorizonStopped(RuntimeError):
    """The controller received a local stop request."""


class ResourceUnavailable(RuntimeError):
    """One exact TPU incarnation is absent or terminal."""

    def __init__(self, message: str, *, classification: str) -> None:
        if classification not in {"spot_preemption", "resource_unavailable", "ssh_transport"}:
            raise ValueError("resource failure classification is not retryable")
        self.classification = classification
        super().__init__(message)


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HorizonPlanError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise HorizonPlanError(f"{name} keys must be strings")
    return value


def _exact_fields(
    value: Any,
    name: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> Mapping[str, Any]:
    selected = _mapping(value, name)
    missing = sorted(required.difference(selected))
    extra = sorted(set(selected).difference(required | optional))
    if missing or extra:
        raise HorizonPlanError(
            f"{name} fields are not exact; missing={missing}, extra={extra}"
        )
    return selected


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise HorizonPlanError(f"{name} must be nonempty NUL-free text")
    return value


def _safe_id(value: Any, name: str) -> str:
    selected = _text(value, name)
    if _SAFE_ID.fullmatch(selected) is None:
        raise HorizonPlanError(f"{name} is not a safe identifier")
    return selected


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HorizonPlanError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise HorizonPlanError(f"{name} must not exceed {maximum}")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise HorizonPlanError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise HorizonPlanError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0.0:
        raise HorizonPlanError(f"{name} must be finite and positive")
    return result


def _absolute_path(value: Any, name: str) -> Path:
    selected = Path(_text(value, name)).expanduser()
    if not selected.is_absolute() or ".." in selected.parts:
        raise HorizonPlanError(f"{name} must be an absolute path without '..'")
    return selected


def _json_primitive(value: Any, name: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise HorizonPlanError(f"{name} must be a finite JSON primitive")


def _argv(value: Any, name: str, *, resource_bound: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise HorizonPlanError(f"{name} must be a nonempty argv array")
    selected = tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))
    placeholders = {
        match.group(1) for argument in selected for match in _PLACEHOLDER.finditer(argument)
    }
    unknown = sorted(placeholders.difference(_ALLOWED_PLACEHOLDERS))
    if unknown:
        raise HorizonPlanError(f"{name} has unsupported placeholders: {unknown}")
    if not resource_bound and placeholders.intersection(_RESOURCE_PLACEHOLDERS):
        raise HorizonPlanError(f"{name} uses resource placeholders without a resource")
    return selected


def _environment(value: Any, name: str) -> dict[str, str]:
    selected = _mapping(value, name)
    result: dict[str, str] = {}
    for key, raw in selected.items():
        if _ENV_NAME.fullmatch(key) is None:
            raise HorizonPlanError(f"{name} contains an invalid environment name")
        if _SECRET_NAME.search(key):
            raise HorizonPlanError(
                f"{name} must not persist secret-like environment variable {key}"
            )
        result[key] = _text(raw, f"{name}.{key}")
    return dict(sorted(result.items()))


@dataclass(frozen=True, slots=True)
class CompletionSpec:
    path: Path
    format: str
    expected: tuple[tuple[str, Any], ...]
    self_hash_field: str | None
    expected_sha256: str | None

    @classmethod
    def parse(cls, value: Any, name: str) -> "CompletionSpec":
        selected = _exact_fields(
            value,
            name,
            required={"path", "format"},
            optional={"expected", "self_hash_field", "expected_sha256"},
        )
        format_name = _text(selected["format"], f"{name}.format")
        if format_name not in {"json", "file"}:
            raise HorizonPlanError(f"{name}.format must be json or file")
        raw_expected = _mapping(selected.get("expected", {}), f"{name}.expected")
        expected = tuple(
            sorted(
                (
                    _text(key, f"{name}.expected key"),
                    _json_primitive(item, f"{name}.expected.{key}"),
                )
                for key, item in raw_expected.items()
            )
        )
        self_hash = selected.get("self_hash_field")
        if self_hash is not None:
            self_hash = _text(self_hash, f"{name}.self_hash_field")
            if format_name != "json" or "." in self_hash:
                raise HorizonPlanError(
                    f"{name}.self_hash_field must be one top-level JSON field"
                )
        expected_sha = selected.get("expected_sha256")
        if expected_sha is not None and (
            not isinstance(expected_sha, str)
            or _SHA256.fullmatch(expected_sha) is None
        ):
            raise HorizonPlanError(f"{name}.expected_sha256 must be a SHA-256")
        if format_name == "file" and (expected or self_hash is not None):
            raise HorizonPlanError(
                f"{name} cannot apply JSON checks to a raw completion file"
            )
        return cls(
            path=_absolute_path(selected["path"], f"{name}.path"),
            format=format_name,
            expected=expected,
            self_hash_field=self_hash,
            expected_sha256=expected_sha,
        )

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": str(self.path),
            "format": self.format,
            "expected": dict(self.expected),
        }
        if self.self_hash_field is not None:
            result["self_hash_field"] = self.self_hash_field
        if self.expected_sha256 is not None:
            result["expected_sha256"] = self.expected_sha256
        return result


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    seconds: int
    max_segments: int
    checkpoint: CompletionSpec
    boundary_exit_codes: tuple[int, ...]

    @classmethod
    def parse(cls, value: Any, name: str) -> "SegmentSpec":
        selected = _exact_fields(
            value,
            name,
            required={"seconds", "max_segments", "checkpoint"},
            optional={"boundary_exit_codes"},
        )
        raw_codes = selected.get("boundary_exit_codes", [0, 124, 137, 143])
        if (
            not isinstance(raw_codes, list)
            or not raw_codes
            or any(isinstance(code, bool) or not isinstance(code, int) for code in raw_codes)
            or len(set(raw_codes)) != len(raw_codes)
        ):
            raise HorizonPlanError(
                f"{name}.boundary_exit_codes must be unique integers"
            )
        return cls(
            seconds=_positive_int(
                selected["seconds"],
                f"{name}.seconds",
                maximum=MAX_TRAIN_INVOCATION_SECONDS,
            ),
            max_segments=_positive_int(
                selected["max_segments"], f"{name}.max_segments", maximum=256
            ),
            checkpoint=CompletionSpec.parse(
                selected["checkpoint"], f"{name}.checkpoint"
            ),
            boundary_exit_codes=tuple(sorted(raw_codes)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "seconds": self.seconds,
            "max_segments": self.max_segments,
            "checkpoint": self.checkpoint.as_dict(),
            "boundary_exit_codes": list(self.boundary_exit_codes),
        }


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    resource_id: str
    allocation: str
    project: str
    zone: str
    node_name_prefix: str
    queued_resource_prefix: str | None
    accelerator_type: str
    runtime_version: str
    service_account: str
    network: str
    subnetwork: str
    process_count: int
    labels: tuple[tuple[str, str], ...]
    formation_timeout_seconds: int
    poll_seconds: float
    cleanup_on_finish: bool

    @classmethod
    def parse(cls, value: Any, name: str) -> "ResourceSpec":
        selected = _exact_fields(
            value,
            name,
            required={
                "id",
                "allocation",
                "project",
                "zone",
                "node_name_prefix",
                "accelerator_type",
                "runtime_version",
                "service_account",
                "network",
                "subnetwork",
                "process_count",
                "labels",
                "formation_timeout_seconds",
                "poll_seconds",
                "cleanup_on_finish",
            },
            optional={"queued_resource_prefix"},
        )
        allocation = _text(selected["allocation"], f"{name}.allocation")
        if allocation not in ALLOCATION_MODES:
            raise HorizonPlanError(
                f"{name}.allocation must be one of {ALLOCATION_MODES}"
            )
        project = _text(selected["project"], f"{name}.project")
        zone = _text(selected["zone"], f"{name}.zone")
        node_prefix = _text(
            selected["node_name_prefix"], f"{name}.node_name_prefix"
        )
        queue_prefix = selected.get("queued_resource_prefix")
        if queue_prefix is not None:
            queue_prefix = _text(queue_prefix, f"{name}.queued_resource_prefix")
        if _PROJECT.fullmatch(project) is None or _ZONE.fullmatch(zone) is None:
            raise HorizonPlanError(f"{name} project or zone is invalid")
        for field_name, field_value in (
            ("node_name_prefix", node_prefix),
            ("queued_resource_prefix", queue_prefix),
        ):
            if field_value is not None and (
                _SAFE_CLOUD_NAME.fullmatch(field_value) is None
                or len(field_value) > 48
            ):
                raise HorizonPlanError(f"{name}.{field_name} is not a safe short name")
        if allocation == "queued_spot" and queue_prefix is None:
            raise HorizonPlanError(
                f"{name}.queued_resource_prefix is required for queued Spot"
            )
        if allocation == "direct_spot" and queue_prefix is not None:
            raise HorizonPlanError(
                f"{name}.queued_resource_prefix is only valid for queued Spot"
            )
        accelerator_type = _text(
            selected["accelerator_type"], f"{name}.accelerator_type"
        )
        if accelerator_type not in REGIONAL_ACCELERATOR_TYPES:
            raise HorizonPlanError(
                f"{name}.accelerator_type must be one of "
                f"{tuple(sorted(REGIONAL_ACCELERATOR_TYPES))}"
            )
        email = _text(selected["service_account"], f"{name}.service_account")
        if _EMAIL.fullmatch(email) is None:
            raise HorizonPlanError(f"{name}.service_account is invalid")
        if selected["process_count"] != 4:
            raise HorizonPlanError(f"{name}.process_count must be exactly 4")
        raw_labels = _mapping(selected["labels"], f"{name}.labels")
        labels: list[tuple[str, str]] = []
        for key, raw in raw_labels.items():
            label_key = _text(key, f"{name}.labels key")
            label_value = _text(raw, f"{name}.labels.{key}")
            if (
                re.fullmatch(r"[a-z][a-z0-9_-]{0,62}", label_key) is None
                or re.fullmatch(r"[a-z0-9_-]{1,63}", label_value) is None
            ):
                raise HorizonPlanError(f"{name}.labels contains an invalid label")
            labels.append((label_key, label_value))
        cleanup = selected["cleanup_on_finish"]
        if type(cleanup) is not bool:
            raise HorizonPlanError(f"{name}.cleanup_on_finish must be boolean")
        return cls(
            resource_id=_safe_id(selected["id"], f"{name}.id"),
            allocation=allocation,
            project=project,
            zone=zone,
            node_name_prefix=node_prefix,
            queued_resource_prefix=queue_prefix,
            accelerator_type=accelerator_type,
            runtime_version=_text(
                selected["runtime_version"], f"{name}.runtime_version"
            ),
            service_account=email,
            network=_text(selected["network"], f"{name}.network"),
            subnetwork=_text(selected["subnetwork"], f"{name}.subnetwork"),
            process_count=4,
            labels=tuple(sorted(labels)),
            formation_timeout_seconds=_positive_int(
                selected["formation_timeout_seconds"],
                f"{name}.formation_timeout_seconds",
                maximum=86_400,
            ),
            poll_seconds=_positive_float(
                selected["poll_seconds"], f"{name}.poll_seconds"
            ),
            cleanup_on_finish=cleanup,
        )

    def as_dict(self) -> dict[str, Any]:
        result = {
            "id": self.resource_id,
            "allocation": self.allocation,
            "project": self.project,
            "zone": self.zone,
            "node_name_prefix": self.node_name_prefix,
            "accelerator_type": self.accelerator_type,
            "runtime_version": self.runtime_version,
            "service_account": self.service_account,
            "network": self.network,
            "subnetwork": self.subnetwork,
            "process_count": self.process_count,
            "labels": dict(self.labels),
            "formation_timeout_seconds": self.formation_timeout_seconds,
            "poll_seconds": self.poll_seconds,
            "cleanup_on_finish": self.cleanup_on_finish,
        }
        if self.queued_resource_prefix is not None:
            result["queued_resource_prefix"] = self.queued_resource_prefix
        return result


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_transient_attempts: int
    initial_backoff_seconds: float
    maximum_backoff_seconds: float
    transport_grace_seconds: int

    @classmethod
    def parse(cls, value: Any) -> "RetryPolicy":
        selected = _exact_fields(
            value,
            "retry_policy",
            required={
                "max_transient_attempts",
                "initial_backoff_seconds",
                "maximum_backoff_seconds",
                "transport_grace_seconds",
            },
        )
        initial = _positive_float(
            selected["initial_backoff_seconds"],
            "retry_policy.initial_backoff_seconds",
        )
        maximum = _positive_float(
            selected["maximum_backoff_seconds"],
            "retry_policy.maximum_backoff_seconds",
        )
        if maximum < initial:
            raise HorizonPlanError(
                "retry_policy.maximum_backoff_seconds is below its initial value"
            )
        return cls(
            max_transient_attempts=_positive_int(
                selected["max_transient_attempts"],
                "retry_policy.max_transient_attempts",
                maximum=1_000,
            ),
            initial_backoff_seconds=initial,
            maximum_backoff_seconds=maximum,
            transport_grace_seconds=_positive_int(
                selected["transport_grace_seconds"],
                "retry_policy.transport_grace_seconds",
                maximum=900,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_transient_attempts": self.max_transient_attempts,
            "initial_backoff_seconds": self.initial_backoff_seconds,
            "maximum_backoff_seconds": self.maximum_backoff_seconds,
            "transport_grace_seconds": self.transport_grace_seconds,
        }


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    kind: str
    depends_on: tuple[str, ...]
    command: tuple[str, ...]
    prepare_commands: tuple[tuple[str, ...], ...]
    environment: tuple[tuple[str, str], ...]
    completion: CompletionSpec
    role: tuple[tuple[str, Any], ...]
    resource_id: str | None
    segment: SegmentSpec | None

    @classmethod
    def parse(cls, value: Any, name: str) -> "TaskSpec":
        selected = _exact_fields(
            value,
            name,
            required={
                "id",
                "kind",
                "depends_on",
                "command",
                "completion",
                "role",
            },
            optional={"prepare_commands", "environment", "resource_id", "segment"},
        )
        task_id = _safe_id(selected["id"], f"{name}.id")
        kind = _text(selected["kind"], f"{name}.kind")
        if kind not in TASK_KINDS:
            raise HorizonPlanError(f"{name}.kind must be one of {TASK_KINDS}")
        raw_dependencies = selected["depends_on"]
        if (
            not isinstance(raw_dependencies, list)
            or any(not isinstance(item, str) for item in raw_dependencies)
            or len(set(raw_dependencies)) != len(raw_dependencies)
        ):
            raise HorizonPlanError(f"{name}.depends_on must be unique task IDs")
        dependencies = tuple(_safe_id(item, f"{name}.depends_on") for item in raw_dependencies)
        resource_id = selected.get("resource_id")
        if resource_id is not None:
            resource_id = _safe_id(resource_id, f"{name}.resource_id")
        command = _argv(
            selected["command"], f"{name}.command", resource_bound=resource_id is not None
        )
        raw_prepare = selected.get("prepare_commands", [])
        if not isinstance(raw_prepare, list):
            raise HorizonPlanError(f"{name}.prepare_commands must be an array")
        prepare = tuple(
            _argv(item, f"{name}.prepare_commands[{index}]", resource_bound=resource_id is not None)
            for index, item in enumerate(raw_prepare)
        )
        environment = _environment(selected.get("environment", {}), f"{name}.environment")
        segment = (
            SegmentSpec.parse(selected["segment"], f"{name}.segment")
            if selected.get("segment") is not None
            else None
        )
        if kind == "train":
            if segment is None:
                raise HorizonPlanError(
                    f"{name} training must declare its invocation boundary"
                )
            placeholders = {
                match.group(1)
                for argument in command
                for match in _PLACEHOLDER.finditer(argument)
            }
            if not {"attempt_id", "segment_seconds"}.issubset(placeholders):
                raise HorizonPlanError(
                    f"{name} training command must bind attempt_id and segment_seconds"
                )
        elif segment is not None:
            raise HorizonPlanError(f"{name}.segment is only valid for training")
        return cls(
            task_id=task_id,
            kind=kind,
            depends_on=dependencies,
            command=command,
            prepare_commands=prepare,
            environment=tuple(environment.items()),
            completion=CompletionSpec.parse(
                selected["completion"], f"{name}.completion"
            ),
            role=tuple(
                sorted(_parse_role(selected["role"], kind=kind, name=f"{name}.role").items())
            ),
            resource_id=resource_id,
            segment=segment,
        )

    def role_dict(self) -> dict[str, Any]:
        return dict(self.role)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.task_id,
            "kind": self.kind,
            "depends_on": list(self.depends_on),
            "command": list(self.command),
            "prepare_commands": [list(item) for item in self.prepare_commands],
            "environment": dict(self.environment),
            "completion": self.completion.as_dict(),
            "role": dict(self.role),
        }
        if self.resource_id is not None:
            result["resource_id"] = self.resource_id
        if self.segment is not None:
            result["segment"] = self.segment.as_dict()
        return result


def _parse_role(value: Any, *, kind: str, name: str) -> dict[str, Any]:
    if kind in {"capture", "overlay"}:
        selected = _exact_fields(value, name, required={"split"})
        split = _text(selected["split"], f"{name}.split")
        if split not in {"train", "eval"}:
            raise HorizonPlanError(f"{name}.split must be train or eval")
        return {"split": split}
    if kind == "mirror":
        selected = _exact_fields(value, name, required={"region"})
        region = _text(selected["region"], f"{name}.region")
        if re.fullmatch(r"[a-z]+-[a-z0-9]+[0-9]", region) is None:
            raise HorizonPlanError(f"{name}.region is invalid")
        return {"region": region}
    if kind == "train":
        selected = _exact_fields(
            value, name, required={"scientific_model", "condition", "seed"}
        )
        return _scientific_role(selected, name=name, milestone=False)
    if kind == "eval":
        selected = _exact_fields(
            value,
            name,
            required={
                "scientific_model",
                "condition",
                "seed",
                "milestone_tokens",
            },
        )
        return _scientific_role(selected, name=name, milestone=True)
    selected = _exact_fields(value, name, required={"scientific_model"})
    model = _text(selected["scientific_model"], f"{name}.scientific_model")
    if model not in SCIENTIFIC_MODELS:
        raise HorizonPlanError(f"{name}.scientific_model is invalid")
    return {"scientific_model": model}


def _scientific_role(
    value: Mapping[str, Any], *, name: str, milestone: bool
) -> dict[str, Any]:
    model = _text(value["scientific_model"], f"{name}.scientific_model")
    condition = _text(value["condition"], f"{name}.condition")
    if model not in SCIENTIFIC_MODELS or condition not in CONDITIONS:
        raise HorizonPlanError(f"{name} model or condition is invalid")
    seed = _positive_int(value["seed"], f"{name}.seed")
    result: dict[str, Any] = {
        "scientific_model": model,
        "condition": condition,
        "seed": seed,
    }
    if milestone:
        milestone_tokens = _positive_int(
            value["milestone_tokens"], f"{name}.milestone_tokens"
        )
        if milestone_tokens not in MILESTONES:
            raise HorizonPlanError(
                f"{name}.milestone_tokens must be one of {MILESTONES}"
            )
        result["milestone_tokens"] = milestone_tokens
    return result


@dataclass(frozen=True, slots=True)
class HorizonPlan:
    campaign_id: str
    state_root: Path
    working_directory: Path
    inherit_environment: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    poll_seconds: float
    max_parallel_tasks: int
    retry_policy: RetryPolicy
    resources: tuple[ResourceSpec, ...]
    tasks: tuple[TaskSpec, ...]
    plan_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "state_root": str(self.state_root),
            "working_directory": str(self.working_directory),
            "inherit_environment": list(self.inherit_environment),
            "environment": dict(self.environment),
            "poll_seconds": self.poll_seconds,
            "max_parallel_tasks": self.max_parallel_tasks,
            "retry_policy": self.retry_policy.as_dict(),
            "resources": [item.as_dict() for item in self.resources],
            "tasks": [item.as_dict() for item in self.tasks],
        }

    @property
    def task_by_id(self) -> dict[str, TaskSpec]:
        return {item.task_id: item for item in self.tasks}

    @property
    def resource_by_id(self) -> dict[str, ResourceSpec]:
        return {item.resource_id: item for item in self.resources}


def parse_horizon_plan(value: Any) -> HorizonPlan:
    selected = _exact_fields(
        value,
        "launch plan",
        required={
            "schema_version",
            "campaign_id",
            "state_root",
            "working_directory",
            "inherit_environment",
            "environment",
            "poll_seconds",
            "max_parallel_tasks",
            "retry_policy",
            "resources",
            "tasks",
        },
    )
    if selected["schema_version"] != PLAN_SCHEMA_VERSION:
        raise HorizonPlanError("launch plan schema_version is unsupported")
    raw_inherit = selected["inherit_environment"]
    if (
        not isinstance(raw_inherit, list)
        or any(not isinstance(item, str) for item in raw_inherit)
        or len(set(raw_inherit)) != len(raw_inherit)
    ):
        raise HorizonPlanError("inherit_environment must contain unique names")
    inherited: list[str] = []
    for item in raw_inherit:
        if _ENV_NAME.fullmatch(item) is None:
            raise HorizonPlanError("inherit_environment contains an invalid name")
        if _SECRET_NAME.search(item):
            raise HorizonPlanError(
                "secrets must come from the controller service account, not the plan"
            )
        inherited.append(item)
    raw_resources = selected["resources"]
    raw_tasks = selected["tasks"]
    if not isinstance(raw_resources, list) or not isinstance(raw_tasks, list):
        raise HorizonPlanError("resources and tasks must be arrays")
    resources = tuple(
        ResourceSpec.parse(item, f"resources[{index}]")
        for index, item in enumerate(raw_resources)
    )
    tasks = tuple(
        TaskSpec.parse(item, f"tasks[{index}]") for index, item in enumerate(raw_tasks)
    )
    plan = HorizonPlan(
        campaign_id=_safe_id(selected["campaign_id"], "campaign_id"),
        state_root=_absolute_path(selected["state_root"], "state_root"),
        working_directory=_absolute_path(
            selected["working_directory"], "working_directory"
        ),
        inherit_environment=tuple(sorted(inherited)),
        environment=tuple(
            _environment(selected["environment"], "environment").items()
        ),
        poll_seconds=_positive_float(selected["poll_seconds"], "poll_seconds"),
        max_parallel_tasks=_positive_int(
            selected["max_parallel_tasks"], "max_parallel_tasks", maximum=64
        ),
        retry_policy=RetryPolicy.parse(selected["retry_policy"]),
        resources=resources,
        tasks=tasks,
        plan_sha256="",
    )
    _validate_plan_graph(plan)
    canonical = plan.as_dict()
    return HorizonPlan(
        campaign_id=plan.campaign_id,
        state_root=plan.state_root,
        working_directory=plan.working_directory,
        inherit_environment=plan.inherit_environment,
        environment=plan.environment,
        poll_seconds=plan.poll_seconds,
        max_parallel_tasks=plan.max_parallel_tasks,
        retry_policy=plan.retry_policy,
        resources=plan.resources,
        tasks=plan.tasks,
        plan_sha256=canonical_sha256(canonical),
    )


def load_horizon_plan(path: str | Path) -> HorizonPlan:
    selected = Path(path).expanduser().resolve()
    try:
        raw = selected.read_text(encoding="utf-8")
    except OSError as error:
        raise HorizonPlanError(f"cannot read launch plan: {selected}") from error
    try:
        decoded = json.loads(raw) if selected.suffix.lower() == ".json" else yaml.safe_load(raw)
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise HorizonPlanError(f"launch plan is not valid JSON/YAML: {selected}") from error
    return parse_horizon_plan(decoded)


def _validate_plan_graph(plan: HorizonPlan) -> None:
    if not plan.tasks:
        raise HorizonPlanError("launch plan has no tasks")
    resource_ids = [item.resource_id for item in plan.resources]
    task_ids = [item.task_id for item in plan.tasks]
    if len(resource_ids) != len(set(resource_ids)):
        raise HorizonPlanError("resource IDs must be unique")
    if len(task_ids) != len(set(task_ids)):
        raise HorizonPlanError("task IDs must be unique")
    tasks = {item.task_id: item for item in plan.tasks}
    resources = set(resource_ids)
    for task in plan.tasks:
        unknown = sorted(set(task.depends_on).difference(tasks))
        if unknown or task.task_id in task.depends_on:
            raise HorizonPlanError(
                f"task {task.task_id} has invalid dependencies: {unknown}"
            )
        if task.resource_id is not None and task.resource_id not in resources:
            raise HorizonPlanError(
                f"task {task.task_id} references an unknown resource"
            )
        if task.kind in {"train", "eval"} and task.resource_id is None:
            raise HorizonPlanError(
                f"{task.kind} task {task.task_id} must bind one accelerator resource"
            )
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise HorizonPlanError("task dependencies contain a cycle")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in tasks[task_id].depends_on:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in task_ids:
        visit(task_id)

    captures = {task.role_dict()["split"] for task in plan.tasks if task.kind == "capture"}
    overlays = {task.role_dict()["split"] for task in plan.tasks if task.kind == "overlay"}
    mirrors = {task.role_dict()["region"] for task in plan.tasks if task.kind == "mirror"}
    if captures != {"train", "eval"} or overlays != {"train", "eval"}:
        raise HorizonPlanError("plan must contain exact train/eval capture and overlay tasks")
    if len(mirrors) < 2:
        raise HorizonPlanError("plan must mirror the sealed data into at least two regions")

    training = [
        (
            task.role_dict()["scientific_model"],
            task.role_dict()["condition"],
            task.role_dict()["seed"],
        )
        for task in plan.tasks
        if task.kind == "train"
    ]
    expected_pairs = {
        (model, condition)
        for model in SCIENTIFIC_MODELS
        for condition in CONDITIONS
    }
    if len(training) != 8 or {(model, condition) for model, condition, _ in training} != expected_pairs:
        raise HorizonPlanError(
            "plan must contain exactly A/R/B/F train tasks for SAE and transcoder"
        )
    if len({(model, condition, seed) for model, condition, seed in training}) != 8:
        raise HorizonPlanError("training scientific roles must be unique")
    seed_by_pair = {(model, condition): seed for model, condition, seed in training}
    evaluations = {
        (
            task.role_dict()["scientific_model"],
            task.role_dict()["condition"],
            task.role_dict()["seed"],
            task.role_dict()["milestone_tokens"],
        )
        for task in plan.tasks
        if task.kind == "eval"
    }
    expected_evaluations = {
        (model, condition, seed_by_pair[(model, condition)], milestone)
        for model, condition in expected_pairs
        for milestone in MILESTONES
    }
    if evaluations != expected_evaluations:
        raise HorizonPlanError(
            "plan must evaluate every SAE/transcoder A/R/B/F trajectory at 2M/5M/10M"
        )
    reports = [
        task.role_dict()["scientific_model"]
        for task in plan.tasks
        if task.kind == "report"
    ]
    if sorted(reports) != sorted(SCIENTIFIC_MODELS):
        raise HorizonPlanError("plan must contain one SAE and one transcoder report")

    ancestors: dict[str, set[str]] = {}

    def task_ancestors(task_id: str) -> set[str]:
        prior = ancestors.get(task_id)
        if prior is not None:
            return prior
        result: set[str] = set()
        for dependency in tasks[task_id].depends_on:
            result.add(dependency)
            result.update(task_ancestors(dependency))
        ancestors[task_id] = result
        return result

    train_by_role = {
        (
            task.role_dict()["scientific_model"],
            task.role_dict()["condition"],
            task.role_dict()["seed"],
        ): task.task_id
        for task in plan.tasks
        if task.kind == "train"
    }
    for task in plan.tasks:
        ancestor_tasks = [tasks[item] for item in task_ancestors(task.task_id)]
        if task.kind == "overlay" and not any(item.kind == "capture" for item in ancestor_tasks):
            raise HorizonPlanError(f"overlay {task.task_id} is not downstream of capture")
        if task.kind == "train":
            ancestor_kinds = {item.kind for item in ancestor_tasks}
            if not {"overlay", "mirror"}.issubset(ancestor_kinds):
                raise HorizonPlanError(
                    f"training {task.task_id} is not downstream of overlay and mirror"
                )
        if task.kind == "eval":
            role = task.role_dict()
            train_id = train_by_role[
                (role["scientific_model"], role["condition"], role["seed"])
            ]
            if train_id not in task_ancestors(task.task_id):
                raise HorizonPlanError(
                    f"evaluation {task.task_id} is not downstream of its train task"
                )
        if task.kind == "report":
            model = task.role_dict()["scientific_model"]
            model_evals = {
                item.task_id
                for item in plan.tasks
                if item.kind == "eval"
                and item.role_dict()["scientific_model"] == model
            }
            if not model_evals.issubset(task_ancestors(task.task_id)):
                raise HorizonPlanError(
                    f"report {task.task_id} is not downstream of all {model} evaluations"
                )


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _lookup_json(value: Any, dotted_path: str) -> Any:
    selected = value
    for component in dotted_path.split("."):
        if not isinstance(selected, Mapping) or component not in selected:
            raise HorizonStateError(
                f"completion JSON lacks expected path {dotted_path}"
            )
        selected = selected[component]
    return selected


def inspect_completion(spec: CompletionSpec) -> FileFingerprint | None:
    path = spec.path
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise HorizonStateError(f"completion is not one regular file: {path}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise HorizonStateError(f"cannot read completion: {path}") from error
    if not payload:
        raise HorizonStateError(f"completion file is empty: {path}")
    digest = hashlib.sha256(payload).hexdigest()
    if spec.expected_sha256 is not None and digest != spec.expected_sha256:
        raise HorizonStateError(f"completion SHA-256 mismatch: {path}")
    if spec.format == "json":
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HorizonStateError(f"completion JSON is invalid: {path}") from error
        for dotted_path, expected in spec.expected:
            if _lookup_json(decoded, dotted_path) != expected:
                raise HorizonStateError(
                    f"completion field {dotted_path} changed: {path}"
                )
        if spec.self_hash_field is not None:
            if not isinstance(decoded, Mapping):
                raise HorizonStateError(f"self-hashed completion is not an object: {path}")
            unsigned = dict(decoded)
            claimed = unsigned.pop(spec.self_hash_field, None)
            if (
                not isinstance(claimed, str)
                or _SHA256.fullmatch(claimed) is None
                or canonical_sha256(unsigned) != claimed
            ):
                raise HorizonStateError(f"completion self-hash is invalid: {path}")
    return FileFingerprint(str(path), len(payload), digest)


def classify_failure(
    *,
    returncode: int | None,
    output: str,
    timed_out: bool,
    completion_present: bool,
    checkpoint_before: FileFingerprint | None,
    checkpoint_after: FileFingerprint | None,
    segment: SegmentSpec | None,
) -> str:
    """Classify one attempt, giving integrity/numerics precedence over transport."""

    if completion_present:
        return "completed"
    for pattern in _NUMERICAL_PATTERNS:
        if pattern.search(output):
            return "numerical_failure"
    for pattern in _CORRUPTION_PATTERNS:
        if pattern.search(output):
            return "scientific_corruption"
    for pattern in _SPOT_PATTERNS:
        if pattern.search(output):
            return "spot_preemption"
    for pattern in _RESOURCE_PATTERNS:
        if pattern.search(output):
            return "resource_unavailable"
    for pattern in _SSH_PATTERNS:
        if pattern.search(output):
            return "ssh_transport"
    if returncode == 255:
        return "ssh_transport"
    checkpoint_advanced = (
        checkpoint_after is not None and checkpoint_after != checkpoint_before
    )
    if (
        segment is not None
        and checkpoint_advanced
        and (timed_out or returncode in segment.boundary_exit_codes)
    ):
        return "segment_boundary"
    if timed_out:
        return "unclassified_timeout"
    if returncode == 0:
        return "missing_completion"
    return "unclassified_failure"


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = canonical_json_bytes(dict(value)) + b"\n"
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise HorizonStateError(f"create-only evidence already exists: {path}") from error
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
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _sealed_record(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    unsigned = dict(value)
    if field in unsigned:
        raise HorizonStateError(f"record already contains seal field {field}")
    return {**unsigned, field: canonical_sha256(unsigned)}


def _read_json(path: Path, *, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HorizonStateError(f"{name} is not one regular file: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HorizonStateError(f"cannot read {name}: {path}") from error
    if not isinstance(value, Mapping):
        raise HorizonStateError(f"{name} is not a JSON object: {path}")
    return value


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    returncode: int | None
    timed_out: bool
    stopped: bool
    elapsed_seconds: float
    stdout: FileFingerprint
    stderr: FileFingerprint
    output_tail: str


class RunCommand(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
        working_directory: Path,
        timeout_seconds: float,
        stdout_path: Path,
        stderr_path: Path,
        stop_event: threading.Event,
    ) -> CommandOutcome: ...


def _fingerprint_regular_file(path: Path) -> FileFingerprint:
    payload = path.read_bytes()
    return FileFingerprint(str(path), len(payload), hashlib.sha256(payload).hexdigest())


def run_command(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    working_directory: Path,
    timeout_seconds: float,
    stdout_path: Path,
    stderr_path: Path,
    stop_event: threading.Event,
) -> CommandOutcome:
    """Run one argv without a shell and reap its entire local process group."""

    stdout_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.monotonic()
    timed_out = False
    stopped = False
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            list(command),
            cwd=working_directory,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            start_new_session=True,
        )
        deadline = started + timeout_seconds
        while process.poll() is None:
            now = time.monotonic()
            if stop_event.is_set() or now >= deadline:
                stopped = stop_event.is_set()
                timed_out = not stopped
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=30.0)
                break
            time.sleep(min(0.25, max(0.01, deadline - now)))
        returncode = process.wait()
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    out_fingerprint = _fingerprint_regular_file(stdout_path)
    err_fingerprint = _fingerprint_regular_file(stderr_path)
    tail_bytes = 512 * 1024
    output = (stdout_path.read_bytes()[-tail_bytes:] + b"\n" + stderr_path.read_bytes()[-tail_bytes:])
    return CommandOutcome(
        returncode=returncode,
        timed_out=timed_out,
        stopped=stopped,
        elapsed_seconds=time.monotonic() - started,
        stdout=out_fingerprint,
        stderr=err_fingerprint,
        output_tail=output.decode("utf-8", errors="replace"),
    )


@dataclass(frozen=True, slots=True)
class ResourceBinding:
    resource_id: str
    attempt_id: str
    allocation: str
    project: str
    zone: str
    tpu_name: str
    queued_resource: str | None
    binding_path: Path

    def placeholders(self) -> dict[str, str]:
        return {
            "resource_id": self.resource_id,
            "tpu_name": self.tpu_name,
            "tpu_zone": self.zone,
            "queued_resource": self.queued_resource or "",
        }


class ResourceManager(Protocol):
    def ensure(
        self,
        spec: ResourceSpec,
        *,
        attempt_id: str,
        stop_event: threading.Event,
    ) -> ResourceBinding: ...

    def invalidate(self, binding: ResourceBinding, *, reason: str) -> None: ...

    def cleanup(self) -> None: ...


class GCloudResourceManager:
    """Exact ``describe``/create/delete adapter for private regional Spot nodes."""

    def __init__(
        self,
        state_root: Path,
        *,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._root = state_root / "resources"
        self._runner = command_runner
        self._sleep = sleeper
        self._lock = threading.Lock()
        self._active: dict[str, ResourceBinding] = {}

    def _load_existing(self, spec: ResourceSpec) -> ResourceBinding | None:
        root = self._root / spec.resource_id / "incarnations"
        if not root.exists():
            return None
        active: list[ResourceBinding] = []
        for incarnation in sorted(root.iterdir()):
            binding_path = incarnation / "binding.json"
            if (
                incarnation.is_symlink()
                or not incarnation.is_dir()
                or not binding_path.is_file()
            ):
                raise HorizonStateError(
                    f"invalid resource incarnation entry: {incarnation}"
                )
            if (incarnation / "release.json").exists():
                continue
            value = _read_json(binding_path, name="resource binding")
            claimed = value.get("binding_sha256")
            unsigned = dict(value)
            unsigned.pop("binding_sha256", None)
            if (
                claimed != canonical_sha256(unsigned)
                or value.get("schema_version") != RESOURCE_BINDING_SCHEMA_VERSION
                or value.get("resource_id") != spec.resource_id
                or value.get("resource_spec_sha256")
                != canonical_sha256(spec.as_dict())
                or value.get("allocation") != spec.allocation
                or value.get("project") != spec.project
                or value.get("zone") != spec.zone
                or value.get("internal_ips") is not True
                or value.get("accelerator_type") != spec.accelerator_type
            ):
                raise HorizonStateError("persisted resource binding changed")
            active.append(
                ResourceBinding(
                    resource_id=spec.resource_id,
                    attempt_id=str(value["attempt_id"]),
                    allocation=spec.allocation,
                    project=spec.project,
                    zone=spec.zone,
                    tpu_name=str(value["tpu_name"]),
                    queued_resource=(
                        None
                        if value.get("queued_resource") is None
                        else str(value["queued_resource"])
                    ),
                    binding_path=binding_path,
                )
            )
        if len(active) > 1:
            raise HorizonStateError(
                f"resource {spec.resource_id} has multiple unreleased incarnations"
            )
        return active[0] if active else None

    @staticmethod
    def _names(spec: ResourceSpec, attempt_id: str) -> tuple[str, str | None]:
        suffix = attempt_id.removeprefix("a")
        node = f"{spec.node_name_prefix}-a{suffix}"
        queue = (
            f"{spec.queued_resource_prefix}-a{suffix}"
            if spec.queued_resource_prefix is not None
            else None
        )
        if len(node) > 63 or (queue is not None and len(queue) > 63):
            raise HorizonStateError("derived TPU resource name exceeds 63 characters")
        return node, queue

    def _invoke(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return self._runner(
            list(command),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )

    @staticmethod
    def _not_found(result: subprocess.CompletedProcess[str]) -> bool:
        text = f"{result.stdout}\n{result.stderr}"
        return result.returncode != 0 and re.search(
            r"(?:\bNOT_FOUND\b|was not found|could not fetch resource.*404)",
            text,
            re.I | re.S,
        ) is not None

    def _describe_queue(self, binding: ResourceBinding) -> Mapping[str, Any] | None:
        assert binding.queued_resource is not None
        result = self._invoke(
            (
                "gcloud",
                "compute",
                "tpus",
                "queued-resources",
                "describe",
                binding.queued_resource,
                f"--project={binding.project}",
                f"--zone={binding.zone}",
                "--format=json",
            )
        )
        if self._not_found(result):
            return None
        if result.returncode != 0:
            classification = classify_failure(
                returncode=result.returncode,
                output=f"{result.stdout}\n{result.stderr}",
                timed_out=False,
                completion_present=False,
                checkpoint_before=None,
                checkpoint_after=None,
                segment=None,
            )
            if classification not in {
                "spot_preemption",
                "resource_unavailable",
                "ssh_transport",
            }:
                classification = "ssh_transport"
            raise ResourceUnavailable(result.stderr.strip(), classification=classification)
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise HorizonStateError("queued-resource describe returned invalid JSON") from error
        return _mapping(value, "queued-resource observation")

    def _describe_node(self, binding: ResourceBinding) -> Mapping[str, Any] | None:
        result = self._invoke(
            (
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "describe",
                binding.tpu_name,
                f"--project={binding.project}",
                f"--zone={binding.zone}",
                "--format=json",
            )
        )
        if self._not_found(result):
            return None
        if result.returncode != 0:
            raise ResourceUnavailable(
                result.stderr.strip(), classification="ssh_transport"
            )
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise HorizonStateError("TPU node describe returned invalid JSON") from error
        return _mapping(value, "TPU node observation")

    @staticmethod
    def _last_component(value: Any) -> str:
        return str(value).rstrip("/").rsplit("/", 1)[-1]

    def _validate_queue(self, value: Mapping[str, Any], binding: ResourceBinding, spec: ResourceSpec) -> str:
        node_specs = _lookup_json(value, "tpu.nodeSpec")
        if not isinstance(node_specs, list) or len(node_specs) != 1:
            raise HorizonStateError("queued resource nodeSpec changed")
        node = _mapping(node_specs[0], "queued node specification").get("node")
        node = _mapping(node, "queued TPU node")
        network = _mapping(node.get("networkConfig"), "queued network config")
        if (
            self._last_component(value.get("name")) != binding.queued_resource
            or not isinstance(value.get("spot"), Mapping)
            or node_specs[0].get("nodeId") != binding.tpu_name
            or node.get("acceleratorType") != spec.accelerator_type
            or node.get("runtimeVersion") != spec.runtime_version
            or bool(network.get("enableExternalIps", False))
        ):
            raise HorizonStateError("queued Spot TPU identity changed")
        state = str(value.get("state", value.get("queueingPolicy", "UNKNOWN"))).upper()
        return state

    def _validate_ready_node(
        self, value: Mapping[str, Any], binding: ResourceBinding, spec: ResourceSpec
    ) -> str:
        network = _mapping(value.get("networkConfig", {}), "TPU network config")
        endpoints = value.get("networkEndpoints", [])
        if (
            self._last_component(value.get("name")) != binding.tpu_name
            or value.get("acceleratorType") != spec.accelerator_type
            or value.get("runtimeVersion") != spec.runtime_version
            or bool(network.get("enableExternalIps", False))
        ):
            raise HorizonStateError("TPU node identity changed")
        state = str(value.get("state", "UNKNOWN")).upper()
        if state == "READY":
            if (
                not isinstance(endpoints, list)
                or len(endpoints) != spec.process_count
                or len({item.get("ipAddress") for item in endpoints if isinstance(item, Mapping)})
                != spec.process_count
                or any(
                    bool(item.get("accessConfig"))
                    for item in endpoints
                    if isinstance(item, Mapping)
                )
            ):
                raise HorizonStateError(
                    f"private {spec.accelerator_type} endpoint topology changed"
                )
        return state

    def _create_binding(
        self, spec: ResourceSpec, *, attempt_id: str
    ) -> ResourceBinding:
        node, queue = self._names(spec, attempt_id)
        root = self._root / spec.resource_id / "incarnations" / attempt_id
        binding = ResourceBinding(
            resource_id=spec.resource_id,
            attempt_id=attempt_id,
            allocation=spec.allocation,
            project=spec.project,
            zone=spec.zone,
            tpu_name=node,
            queued_resource=queue,
            binding_path=root / "binding.json",
        )
        record = _sealed_record(
            {
                "schema_version": RESOURCE_BINDING_SCHEMA_VERSION,
                "resource_id": spec.resource_id,
                "resource_spec_sha256": canonical_sha256(spec.as_dict()),
                "attempt_id": attempt_id,
                "allocation": spec.allocation,
                "project": spec.project,
                "zone": spec.zone,
                "tpu_name": node,
                "queued_resource": queue,
                "internal_ips": True,
                "accelerator_type": spec.accelerator_type,
                "created_at": _utc_now(),
            },
            "binding_sha256",
        )
        _write_create_only(binding.binding_path, record)
        labels = ",".join(f"{key}={value}" for key, value in spec.labels)
        common = [
            f"--project={spec.project}",
            f"--zone={spec.zone}",
            f"--accelerator-type={spec.accelerator_type}",
            "--spot",
            "--internal-ips",
            f"--network={spec.network}",
            f"--subnetwork={spec.subnetwork}",
            f"--service-account={spec.service_account}",
            "--scopes=https://www.googleapis.com/auth/cloud-platform",
            "--quiet",
        ]
        if labels:
            common.append(f"--labels={labels}")
        if spec.allocation == "queued_spot":
            assert queue is not None
            command = [
                "gcloud",
                "compute",
                "tpus",
                "queued-resources",
                "create",
                queue,
                f"--node-id={node}",
                f"--runtime-version={spec.runtime_version}",
                *common,
            ]
        else:
            command = [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "create",
                node,
                f"--version={spec.runtime_version}",
                *common,
            ]
        result = self._invoke(command)
        if result.returncode != 0 and not re.search(
            r"already exists", f"{result.stdout}\n{result.stderr}", re.I
        ):
            classification = classify_failure(
                returncode=result.returncode,
                output=f"{result.stdout}\n{result.stderr}",
                timed_out=False,
                completion_present=False,
                checkpoint_before=None,
                checkpoint_after=None,
                segment=None,
            )
            if classification not in {
                "spot_preemption",
                "resource_unavailable",
                "ssh_transport",
            }:
                raise HorizonStateError(
                    f"unclassified TPU create failure: {result.stderr.strip()}"
                )
            raise ResourceUnavailable(
                result.stderr.strip(), classification=classification
            )
        return binding

    def ensure(
        self,
        spec: ResourceSpec,
        *,
        attempt_id: str,
        stop_event: threading.Event,
    ) -> ResourceBinding:
        with self._lock:
            binding = self._active.get(spec.resource_id)
            if binding is None:
                binding = self._load_existing(spec)
            if binding is None:
                binding = self._create_binding(spec, attempt_id=attempt_id)
                self._active[spec.resource_id] = binding
            else:
                self._active[spec.resource_id] = binding
        deadline = time.monotonic() + spec.formation_timeout_seconds
        terminal_states = {
            "FAILED",
            "SUSPENDED",
            "SUSPENDING",
            "CANCELLED",
            "DELETING",
            "TERMINATED",
        }
        while not stop_event.is_set() and time.monotonic() < deadline:
            if spec.allocation == "queued_spot":
                queue = self._describe_queue(binding)
                if queue is None:
                    raise ResourceUnavailable(
                        "queued Spot resource disappeared",
                        classification="spot_preemption",
                    )
                queue_state = self._validate_queue(queue, binding, spec)
                if queue_state in terminal_states:
                    raise ResourceUnavailable(
                        f"queued Spot resource entered {queue_state}",
                        classification="spot_preemption",
                    )
            node = self._describe_node(binding)
            if node is not None:
                node_state = self._validate_ready_node(node, binding, spec)
                if node_state == "READY":
                    return binding
                if node_state in terminal_states:
                    raise ResourceUnavailable(
                        f"Spot TPU node entered {node_state}",
                        classification="spot_preemption",
                    )
            self._sleep(spec.poll_seconds)
        if stop_event.is_set():
            raise HorizonStopped("controller stop requested during TPU formation")
        raise ResourceUnavailable(
            "TPU formation timeout expired", classification="resource_unavailable"
        )

    def invalidate(self, binding: ResourceBinding, *, reason: str) -> None:
        with self._lock:
            if self._active.get(binding.resource_id) != binding:
                return
            if binding.allocation == "queued_spot":
                assert binding.queued_resource is not None
                command = (
                    "gcloud",
                    "compute",
                    "tpus",
                    "queued-resources",
                    "delete",
                    binding.queued_resource,
                    f"--project={binding.project}",
                    f"--zone={binding.zone}",
                    "--async",
                    "--quiet",
                )
            else:
                command = (
                    "gcloud",
                    "compute",
                    "tpus",
                    "tpu-vm",
                    "delete",
                    binding.tpu_name,
                    f"--project={binding.project}",
                    f"--zone={binding.zone}",
                    "--async",
                    "--quiet",
                )
            result = self._invoke(command)
            if result.returncode != 0 and not self._not_found(result):
                raise HorizonStateError(
                    f"failed to release TPU resource: {result.stderr.strip()}"
                )
            record = _sealed_record(
                {
                    "schema_version": RESOURCE_RELEASE_SCHEMA_VERSION,
                    "resource_id": binding.resource_id,
                    "attempt_id": binding.attempt_id,
                    "reason": reason,
                    "released_at": _utc_now(),
                },
                "release_sha256",
            )
            _write_create_only(binding.binding_path.parent / "release.json", record)
            del self._active[binding.resource_id]

    def cleanup(self) -> None:
        failures: list[str] = []
        for binding in list(self._active.values()):
            try:
                self.invalidate(binding, reason="campaign-terminal-cleanup")
            except Exception as error:  # cleanup must attempt every exact resource
                failures.append(f"{binding.resource_id}: {error}")
        if failures:
            raise HorizonStateError(
                "resource cleanup was incomplete: " + "; ".join(failures)
            )


@dataclass(frozen=True, slots=True)
class AttemptIdentity:
    number: int
    attempt_id: str
    root: Path


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    task_id: str
    status: str
    classification: str
    attempt_id: str | None
    detail: str


class HorizonSupervisor:
    """Run one exact horizon DAG to immutable success or first fatal failure."""

    def __init__(
        self,
        plan: HorizonPlan,
        *,
        command_runner: RunCommand = run_command,
        resource_manager: ResourceManager | None = None,
        stop_event: threading.Event | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.plan = plan
        self.root = plan.state_root
        self._run_command = command_runner
        self._resources = resource_manager or GCloudResourceManager(self.root)
        self._stop = stop_event or threading.Event()
        self._sleeper = sleeper
        self._state_lock = threading.Lock()
        self._lock_handle: Any | None = None
        self._next_attempt = 1
        self._completed: set[str] = set()
        self._running: set[str] = set()

    def request_stop(self) -> None:
        self._stop.set()

    def _acquire_controller_lock(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = self.root / "supervisor.lock"
        self._lock_handle = lock_path.open("a+b")
        try:
            fcntl.flock(
                self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as error:
            self._lock_handle.close()
            self._lock_handle = None
            raise HorizonStateError(
                f"another horizon supervisor owns {lock_path}"
            ) from error

    def _release_controller_lock(self) -> None:
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None

    def _bind_plan(self) -> None:
        path = self.root / "bound-plan.json"
        record = _sealed_record(
            {
                "schema_version": PLAN_SCHEMA_VERSION,
                "plan_sha256": self.plan.plan_sha256,
                "plan": self.plan.as_dict(),
            },
            "binding_sha256",
        )
        if path.exists():
            existing = _read_json(path, name="bound horizon plan")
            claimed = existing.get("binding_sha256")
            unsigned = dict(existing)
            unsigned.pop("binding_sha256", None)
            if (
                claimed != canonical_sha256(unsigned)
                or existing.get("plan_sha256") != self.plan.plan_sha256
                or existing.get("plan") != self.plan.as_dict()
            ):
                raise HorizonStateError(
                    "state root is already bound to a different launch plan"
                )
        else:
            _write_create_only(path, record)
        maximum = 0
        attempts_root = self.root / "attempts"
        if attempts_root.exists():
            for child in attempts_root.iterdir():
                match = _ATTEMPT_DIRECTORY.fullmatch(child.name)
                if match is None or child.is_symlink() or not child.is_dir():
                    raise HorizonStateError(
                        f"attempt inventory contains an invalid entry: {child}"
                    )
                maximum = max(maximum, int(match.group("number")))
        self._next_attempt = maximum + 1
        self._reconcile_interrupted_attempts()
        self._recover_terminal_failures()

    def _reconcile_interrupted_attempts(self) -> None:
        """Fence attempts whose controller/SSH observation was interrupted.

        A create-only intent without a result means this controller lost its
        transport before it could classify the child.  The old argv is never
        reattached.  A fresh monotonic attempt may restore only through the
        worker's immutable checkpoint/control state.
        """

        attempts_root = self.root / "attempts"
        if not attempts_root.exists():
            return
        tasks = self.plan.task_by_id
        for child in sorted(attempts_root.iterdir()):
            match = _ATTEMPT_DIRECTORY.fullmatch(child.name)
            assert match is not None  # validated by _bind_plan inventory scan
            result_path = child / "result.json"
            if result_path.exists():
                continue
            task_id = match.group("task")
            task = tasks.get(task_id)
            if task is None:
                raise HorizonStateError(
                    f"interrupted attempt references unknown task {task_id}"
                )
            number = int(match.group("number"))
            attempt = AttemptIdentity(number, f"a{number:08d}", child)
            segment_number = 0
            intent_path = child / "intent.json"
            if intent_path.exists():
                intent = _read_json(intent_path, name="interrupted attempt intent")
                if (
                    intent.get("plan_sha256") != self.plan.plan_sha256
                    or intent.get("task_id") != task_id
                    or intent.get("attempt_id") != attempt.attempt_id
                    or intent.get("attempt_number") != number
                ):
                    raise HorizonStateError("interrupted attempt intent changed")
                segment_number = int(intent.get("segment_number", 0))
            self._record_attempt_result(
                task,
                attempt,
                segment_number=segment_number,
                classification="ssh_transport",
                outcome=None,
                completion=inspect_completion(task.completion),
                checkpoint_before=None,
                checkpoint_after=(
                    inspect_completion(task.segment.checkpoint)
                    if task.segment is not None
                    else None
                ),
                detail=(
                    "controller restarted before terminal SSH/process observation; "
                    "fresh attempt must resume only from immutable control state"
                ),
            )

    def _prior_attempt_counts(self, task: TaskSpec) -> tuple[int, int]:
        transient = 0
        segments = 0
        attempts_root = self.root / "attempts"
        if not attempts_root.exists():
            return transient, segments
        for child in attempts_root.iterdir():
            match = _ATTEMPT_DIRECTORY.fullmatch(child.name)
            if match is None or match.group("task") != task.task_id:
                continue
            result_path = child / "result.json"
            if not result_path.exists():
                continue
            result = _read_json(result_path, name="attempt result")
            classification = result.get("classification")
            if classification in {
                "spot_preemption",
                "resource_unavailable",
                "ssh_transport",
            }:
                transient += 1
            if classification == "segment_boundary":
                segments += 1
        return transient, segments

    def _recover_terminal_failures(self) -> None:
        retryable = {
            "completed",
            "segment_boundary",
            "spot_preemption",
            "resource_unavailable",
            "ssh_transport",
        }
        attempts_root = self.root / "attempts"
        if not attempts_root.exists():
            return
        latest: dict[str, tuple[int, Mapping[str, Any]]] = {}
        for child in attempts_root.iterdir():
            match = _ATTEMPT_DIRECTORY.fullmatch(child.name)
            if match is None or not (child / "result.json").exists():
                continue
            result = _read_json(child / "result.json", name="attempt result")
            task_id = match.group("task")
            number = int(match.group("number"))
            if number > latest.get(task_id, (-1, {}))[0]:
                latest[task_id] = (number, result)
        for task_id, (_number, result) in latest.items():
            classification = str(result.get("classification"))
            if classification in retryable:
                continue
            self._record_failure(
                TaskOutcome(
                    task_id=task_id,
                    status="failed",
                    classification=classification,
                    attempt_id=(
                        str(result.get("attempt_id"))
                        if result.get("attempt_id") is not None
                        else None
                    ),
                    detail=str(result.get("detail", "recovered terminal attempt")),
                )
            )

    def _completion_witness_path(self, task_id: str) -> Path:
        return self.root / "task-completions" / f"{task_id}.json"

    def _failure_path(self, task_id: str) -> Path:
        return self.root / "task-failures" / f"{task_id}.json"

    def _record_completion(
        self, task: TaskSpec, fingerprint: FileFingerprint
    ) -> None:
        path = self._completion_witness_path(task.task_id)
        record = _sealed_record(
            {
                "schema_version": COMPLETION_WITNESS_SCHEMA_VERSION,
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "task_id": task.task_id,
                "completion_spec_sha256": canonical_sha256(
                    task.completion.as_dict()
                ),
                "completion": fingerprint.as_dict(),
                "observed_at": _utc_now(),
            },
            "witness_sha256",
        )
        with self._state_lock:
            if not path.exists():
                _write_create_only(path, record)
            self._validate_completion_witness(task)
            self._completed.add(task.task_id)

    def _validate_completion_witness(self, task: TaskSpec) -> FileFingerprint:
        path = self._completion_witness_path(task.task_id)
        value = _read_json(path, name=f"{task.task_id} completion witness")
        claimed = value.get("witness_sha256")
        unsigned = dict(value)
        unsigned.pop("witness_sha256", None)
        if (
            claimed != canonical_sha256(unsigned)
            or value.get("schema_version") != COMPLETION_WITNESS_SCHEMA_VERSION
            or value.get("campaign_id") != self.plan.campaign_id
            or value.get("plan_sha256") != self.plan.plan_sha256
            or value.get("task_id") != task.task_id
            or value.get("completion_spec_sha256")
            != canonical_sha256(task.completion.as_dict())
        ):
            raise HorizonStateError(
                f"completion witness changed for task {task.task_id}"
            )
        observed = inspect_completion(task.completion)
        if observed is None or value.get("completion") != observed.as_dict():
            raise HorizonStateError(
                f"immutable completion changed after witnessing: {task.task_id}"
            )
        return observed

    def _discover_completions(self) -> None:
        for task in self.plan.tasks:
            witness = self._completion_witness_path(task.task_id)
            failure = self._failure_path(task.task_id)
            if witness.exists() and failure.exists():
                raise HorizonStateError(
                    f"task {task.task_id} has both completion and failure evidence"
                )
            if witness.exists():
                self._validate_completion_witness(task)
                self._completed.add(task.task_id)
                continue
            observed = inspect_completion(task.completion)
            if observed is not None:
                self._record_completion(task, observed)

    def _reserve_attempt(self, task: TaskSpec, *, segment_number: int) -> AttemptIdentity:
        with self._state_lock:
            number = self._next_attempt
            self._next_attempt += 1
            attempt_id = f"a{number:08d}"
            root = (
                self.root
                / "attempts"
                / f"attempt-{number:08d}-{task.task_id}"
            )
            root.mkdir(parents=True, exist_ok=False, mode=0o700)
            return AttemptIdentity(number, attempt_id, root)

    def _base_environment(self, task: TaskSpec) -> dict[str, str]:
        safe_defaults = (
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "LANG",
            "LC_ALL",
            "CLOUDSDK_CONFIG",
            "SSH_AUTH_SOCK",
            "TPU_SSH_USER",
        )
        result = {
            key: os.environ[key]
            for key in safe_defaults
            if key in os.environ
        }
        for key in self.plan.inherit_environment:
            if key not in os.environ:
                raise HorizonStateError(
                    f"required inherited environment variable is absent: {key}"
                )
            result[key] = os.environ[key]
        result.update(dict(self.plan.environment))
        result.update(dict(task.environment))
        return result

    def _render(
        self,
        argv: Sequence[str],
        *,
        task: TaskSpec,
        attempt: AttemptIdentity,
        binding: ResourceBinding | None,
    ) -> tuple[str, ...]:
        placeholders = {
            "attempt_id": attempt.attempt_id,
            "attempt_number": str(attempt.number),
            "campaign_id": self.plan.campaign_id,
            "plan_sha256": self.plan.plan_sha256,
            "segment_seconds": str(
                task.segment.seconds
                if task.segment is not None
                else DEFAULT_TASK_TIMEOUT_SECONDS
            ),
            "task_id": task.task_id,
            "resource_id": "",
            "tpu_name": "",
            "tpu_zone": "",
            "queued_resource": "",
        }
        if binding is not None:
            placeholders.update(binding.placeholders())

        def replace(match: re.Match[str]) -> str:
            return placeholders[match.group(1)]

        return tuple(_PLACEHOLDER.sub(replace, item) for item in argv)

    def _attempt_intent(
        self,
        task: TaskSpec,
        attempt: AttemptIdentity,
        *,
        segment_number: int,
        binding: ResourceBinding | None,
        environment: Mapping[str, str],
        command: Sequence[str],
        prepare: Sequence[Sequence[str]],
    ) -> None:
        record = _sealed_record(
            {
                "schema_version": ATTEMPT_INTENT_SCHEMA_VERSION,
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "task_id": task.task_id,
                "task_spec_sha256": canonical_sha256(task.as_dict()),
                "attempt_number": attempt.number,
                "attempt_id": attempt.attempt_id,
                "segment_number": segment_number,
                "resource_binding": (
                    None
                    if binding is None
                    else {
                        "resource_id": binding.resource_id,
                        "attempt_id": binding.attempt_id,
                        "tpu_name": binding.tpu_name,
                        "queued_resource": binding.queued_resource,
                        "zone": binding.zone,
                    }
                ),
                "command": list(command),
                "prepare_commands": [list(item) for item in prepare],
                "environment_keys": sorted(environment),
                "environment_sha256": canonical_sha256(dict(environment)),
                "environment_recording": "values hashed but not persisted",
                "created_at": _utc_now(),
            },
            "intent_sha256",
        )
        _write_create_only(attempt.root / "intent.json", record)

    def _run_one_command(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
        timeout_seconds: float,
        root: Path,
        stem: str,
    ) -> CommandOutcome:
        return self._run_command(
            command,
            environment=environment,
            working_directory=self.plan.working_directory,
            timeout_seconds=timeout_seconds,
            stdout_path=root / f"{stem}.stdout.log",
            stderr_path=root / f"{stem}.stderr.log",
            stop_event=self._stop,
        )

    def _record_attempt_result(
        self,
        task: TaskSpec,
        attempt: AttemptIdentity,
        *,
        segment_number: int,
        classification: str,
        outcome: CommandOutcome | None,
        completion: FileFingerprint | None,
        checkpoint_before: FileFingerprint | None,
        checkpoint_after: FileFingerprint | None,
        detail: str,
    ) -> None:
        record = _sealed_record(
            {
                "schema_version": ATTEMPT_RESULT_SCHEMA_VERSION,
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "task_id": task.task_id,
                "attempt_number": attempt.number,
                "attempt_id": attempt.attempt_id,
                "segment_number": segment_number,
                "classification": classification,
                "retryable": classification
                in {
                    "spot_preemption",
                    "resource_unavailable",
                    "ssh_transport",
                },
                "controlled_continuation": classification == "segment_boundary",
                "returncode": None if outcome is None else outcome.returncode,
                "timed_out": False if outcome is None else outcome.timed_out,
                "stopped": False if outcome is None else outcome.stopped,
                "elapsed_seconds": 0.0
                if outcome is None
                else outcome.elapsed_seconds,
                "stdout": None if outcome is None else outcome.stdout.as_dict(),
                "stderr": None if outcome is None else outcome.stderr.as_dict(),
                "completion": None if completion is None else completion.as_dict(),
                "checkpoint_before": (
                    None if checkpoint_before is None else checkpoint_before.as_dict()
                ),
                "checkpoint_after": (
                    None if checkpoint_after is None else checkpoint_after.as_dict()
                ),
                "detail": detail[:4096],
                "finished_at": _utc_now(),
            },
            "result_sha256",
        )
        _write_create_only(attempt.root / "result.json", record)

    def _backoff(self, transient_attempts: int) -> None:
        policy = self.plan.retry_policy
        seconds = min(
            policy.maximum_backoff_seconds,
            policy.initial_backoff_seconds * (2 ** max(0, transient_attempts - 1)),
        )
        deadline = time.monotonic() + seconds
        while not self._stop.is_set() and time.monotonic() < deadline:
            self._sleeper(min(0.25, max(0.0, deadline - time.monotonic())))

    def _run_task(self, task: TaskSpec) -> TaskOutcome:
        transient_attempts, completed_segments = self._prior_attempt_counts(task)
        segment_number = completed_segments
        binding: ResourceBinding | None = None
        while not self._stop.is_set():
            segment_number += 1
            if task.segment is not None and segment_number > task.segment.max_segments:
                return TaskOutcome(
                    task.task_id,
                    "failed",
                    "segment_limit_exceeded",
                    None,
                    "training exceeded its explicit segment limit",
                )
            attempt = self._reserve_attempt(task, segment_number=segment_number)
            try:
                resource_spec: ResourceSpec | None = None
                if task.resource_id is not None:
                    resource_spec = self.plan.resource_by_id[task.resource_id]
                    binding = self._resources.ensure(
                        resource_spec,
                        attempt_id=attempt.attempt_id,
                        stop_event=self._stop,
                    )
                environment = self._base_environment(task)
                if resource_spec is not None:
                    assert binding is not None
                    # The current incarnation is authoritative. Do not leave
                    # preemption-created names or accelerator generations to a
                    # stale literal in the launch plan.
                    environment.update(
                        {
                            "GCP_PROJECT": resource_spec.project,
                            "TPU_NAME": binding.tpu_name,
                            "TPU_ZONE": resource_spec.zone,
                            "TPU_ACCELERATOR_TYPE": (
                                resource_spec.accelerator_type
                            ),
                            "TPU_USE_INTERNAL_IP": "1",
                        }
                    )
                command = self._render(
                    task.command, task=task, attempt=attempt, binding=binding
                )
                prepare = tuple(
                    self._render(item, task=task, attempt=attempt, binding=binding)
                    for item in task.prepare_commands
                )
                self._attempt_intent(
                    task,
                    attempt,
                    segment_number=segment_number,
                    binding=binding,
                    environment=environment,
                    command=command,
                    prepare=prepare,
                )
                checkpoint_before = (
                    inspect_completion(task.segment.checkpoint)
                    if task.segment is not None
                    else None
                )
                outcome: CommandOutcome | None = None
                for index, prepare_command in enumerate(prepare):
                    outcome = self._run_one_command(
                        prepare_command,
                        environment=environment,
                        timeout_seconds=DEFAULT_TASK_TIMEOUT_SECONDS,
                        root=attempt.root,
                        stem=f"prepare-{index:02d}",
                    )
                    if outcome.returncode != 0 or outcome.timed_out or outcome.stopped:
                        break
                else:
                    outcome = self._run_one_command(
                        command,
                        environment=environment,
                        timeout_seconds=(
                            task.segment.seconds
                            if task.segment is not None
                            else DEFAULT_TASK_TIMEOUT_SECONDS
                        )
                        + self.plan.retry_policy.transport_grace_seconds,
                        root=attempt.root,
                        stem="command",
                    )
                completion = inspect_completion(task.completion)
                checkpoint_after = (
                    inspect_completion(task.segment.checkpoint)
                    if task.segment is not None
                    else None
                )
                assert outcome is not None
                classification = classify_failure(
                    returncode=outcome.returncode,
                    output=outcome.output_tail,
                    timed_out=outcome.timed_out,
                    completion_present=completion is not None,
                    checkpoint_before=checkpoint_before,
                    checkpoint_after=checkpoint_after,
                    segment=task.segment,
                )
                detail = outcome.output_tail[-4096:]
                self._record_attempt_result(
                    task,
                    attempt,
                    segment_number=segment_number,
                    classification=classification,
                    outcome=outcome,
                    completion=completion,
                    checkpoint_before=checkpoint_before,
                    checkpoint_after=checkpoint_after,
                    detail=detail,
                )
            except ResourceUnavailable as error:
                classification = error.classification
                self._record_attempt_result(
                    task,
                    attempt,
                    segment_number=segment_number,
                    classification=classification,
                    outcome=None,
                    completion=None,
                    checkpoint_before=None,
                    checkpoint_after=None,
                    detail=str(error),
                )
            except HorizonStopped:
                return TaskOutcome(
                    task.task_id,
                    "stopped",
                    "controller_stop",
                    attempt.attempt_id,
                    "controller stop requested",
                )
            except Exception as error:
                return TaskOutcome(
                    task.task_id,
                    "failed",
                    "controller_or_plan_failure",
                    attempt.attempt_id,
                    f"{type(error).__name__}: {error}",
                )

            if classification == "completed":
                assert completion is not None
                self._record_completion(task, completion)
                return TaskOutcome(
                    task.task_id,
                    "completed",
                    classification,
                    attempt.attempt_id,
                    "immutable completion witnessed",
                )
            if classification == "segment_boundary":
                continue
            if classification in {
                "spot_preemption",
                "resource_unavailable",
                "ssh_transport",
            }:
                transient_attempts += 1
                if binding is not None and classification in {
                    "spot_preemption",
                    "resource_unavailable",
                }:
                    try:
                        self._resources.invalidate(
                            binding, reason=classification
                        )
                    finally:
                        binding = None
                if transient_attempts >= self.plan.retry_policy.max_transient_attempts:
                    return TaskOutcome(
                        task.task_id,
                        "failed",
                        "transient_retry_limit_exceeded",
                        attempt.attempt_id,
                        f"{classification} repeated {transient_attempts} times",
                    )
                self._backoff(transient_attempts)
                continue
            return TaskOutcome(
                task.task_id,
                "failed",
                classification,
                attempt.attempt_id,
                detail,
            )
        return TaskOutcome(
            task.task_id,
            "stopped",
            "controller_stop",
            None,
            "controller stop requested",
        )

    def _record_failure(self, outcome: TaskOutcome) -> None:
        record = _sealed_record(
            {
                "schema_version": TASK_FAILURE_SCHEMA_VERSION,
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "task_id": outcome.task_id,
                "attempt_id": outcome.attempt_id,
                "classification": outcome.classification,
                "detail": outcome.detail[:4096],
                "failed_at": _utc_now(),
            },
            "failure_sha256",
        )
        path = self._failure_path(outcome.task_id)
        if not path.exists():
            _write_create_only(path, record)

    def _record_final(self, *, status: str, detail: str) -> Mapping[str, Any]:
        path = self.root / "final.json"
        record = _sealed_record(
            {
                "schema_version": FINAL_SCHEMA_VERSION,
                "campaign_id": self.plan.campaign_id,
                "plan_sha256": self.plan.plan_sha256,
                "status": status,
                "completed_tasks": sorted(self._completed),
                "task_count": len(self.plan.tasks),
                "detail": detail[:4096],
                "finished_at": _utc_now(),
            },
            "final_sha256",
        )
        if not path.exists():
            _write_create_only(path, record)
        return _read_json(path, name="horizon final record")

    def run(self) -> Mapping[str, Any]:
        self._acquire_controller_lock()
        futures: dict[Future[TaskOutcome], str] = {}
        terminal_status = "failed"
        terminal_detail = "controller exited before completion"
        try:
            self._bind_plan()
            self._discover_completions()
            for task in self.plan.tasks:
                if self._failure_path(task.task_id).exists():
                    failure = _read_json(
                        self._failure_path(task.task_id),
                        name=f"{task.task_id} failure",
                    )
                    return self._record_final(
                        status="failed",
                        detail=(
                            f"prior terminal failure in {task.task_id}: "
                            f"{failure.get('classification')}"
                        ),
                    )
            with ThreadPoolExecutor(
                max_workers=self.plan.max_parallel_tasks,
                thread_name_prefix="horizon",
            ) as executor:
                while not self._stop.is_set():
                    self._discover_completions()
                    if len(self._completed) == len(self.plan.tasks):
                        terminal_status = "completed"
                        terminal_detail = "all immutable completions witnessed"
                        break
                    for task in self.plan.tasks:
                        if (
                            task.task_id not in self._completed
                            and task.task_id not in self._running
                            and all(item in self._completed for item in task.depends_on)
                            and len(futures) < self.plan.max_parallel_tasks
                        ):
                            future = executor.submit(self._run_task, task)
                            futures[future] = task.task_id
                            self._running.add(task.task_id)
                    if not futures:
                        terminal_detail = "no runnable tasks remain before completion"
                        break
                    done, _ = wait(
                        tuple(futures),
                        timeout=self.plan.poll_seconds,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        task_id = futures.pop(future)
                        self._running.remove(task_id)
                        try:
                            outcome = future.result()
                        except Exception as error:
                            outcome = TaskOutcome(
                                task_id,
                                "failed",
                                "controller_or_plan_failure",
                                None,
                                f"{type(error).__name__}: {error}",
                            )
                        if outcome.status == "completed":
                            self._completed.add(task_id)
                        elif outcome.status == "failed":
                            self._record_failure(outcome)
                            self._stop.set()
                            terminal_detail = (
                                f"{task_id} failed: {outcome.classification}: "
                                f"{outcome.detail}"
                            )
                            break
                if self._stop.is_set() and terminal_status != "completed":
                    for future in futures:
                        try:
                            future.result()
                        except Exception:
                            pass
                    if terminal_detail == "controller exited before completion":
                        terminal_status = "stopped"
                        terminal_detail = "controller stop requested"
            if terminal_status in {"completed", "failed"}:
                self._resources.cleanup()
            if terminal_status == "stopped":
                # A local controller/service interruption is not a scientific
                # terminal state.  In particular, systemd must be able to
                # restart against the same bound plan and checkpoint pointers
                # without an immutable "stopped" record masking later success.
                return {
                    "schema_version": FINAL_SCHEMA_VERSION,
                    "campaign_id": self.plan.campaign_id,
                    "plan_sha256": self.plan.plan_sha256,
                    "status": "stopped",
                    "completed_tasks": sorted(self._completed),
                    "task_count": len(self.plan.tasks),
                    "detail": terminal_detail,
                    "durable_terminal_record": False,
                }
            return self._record_final(
                status=terminal_status,
                detail=terminal_detail,
            )
        finally:
            self._release_controller_lock()


def horizon_status(plan: HorizonPlan) -> dict[str, Any]:
    """Return a read-only status snapshot without contacting cloud APIs."""

    completed: list[str] = []
    failed: list[dict[str, Any]] = []
    for task in plan.tasks:
        witness = plan.state_root / "task-completions" / f"{task.task_id}.json"
        failure = plan.state_root / "task-failures" / f"{task.task_id}.json"
        if witness.exists():
            completed.append(task.task_id)
        if failure.exists():
            selected = _read_json(failure, name=f"{task.task_id} failure")
            failed.append(
                {
                    "task_id": task.task_id,
                    "classification": selected.get("classification"),
                    "attempt_id": selected.get("attempt_id"),
                }
            )
    attempts = 0
    attempts_root = plan.state_root / "attempts"
    if attempts_root.exists():
        attempts = sum(1 for item in attempts_root.iterdir() if item.is_dir())
    final_path = plan.state_root / "final.json"
    final = (
        dict(_read_json(final_path, name="horizon final record"))
        if final_path.exists()
        else None
    )
    return {
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.plan_sha256,
        "completed_tasks": completed,
        "task_count": len(plan.tasks),
        "failed_tasks": failed,
        "attempt_count": attempts,
        "final": final,
    }
