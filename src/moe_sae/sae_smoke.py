"""JAX-free transaction protocol for full-shape SAE smoke attempts.

The production trainer, its capacity scans, and evaluation artifacts must not
be touched merely to prove that one full-size optimizer update can be saved and
restored.  This module therefore owns a deliberately separate ``sae-smoke``
namespace and delegates every numerical or Orbax operation to injected
callbacks.

One control-plane coordinator calls :meth:`SAESmokeRunner.run_attempt`.  The
callbacks may themselves launch or synchronize all JAX processes.  A callback
which computes ``state_sha256`` must cover the complete training state (model
parameters, optimizer state, progress counters, and any numerical state needed
to resume); a checkpoint digest must cover the externally finalized Orbax
payload, not a temporary writer directory.

The protocol is fail closed:

* conditions may pass only in the order F, A, B;
* each attempt has a caller-supplied immutable identity and unique paths;
* exactly one update advances step 0 to 1 and consumes one effective batch;
* an independent inspector attests the finalized external checkpoint digest;
* the restored full-state digest and progress must exactly match the save; and
* every caught failure is retained beside the immutable attempt record.

No function in this module imports JAX, Orbax, the distributed trainer, or the
production orchestration modules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
from typing import Any, Literal


__all__ = [
    "FinalizedCheckpoint",
    "SAEFullShape",
    "SAESmokeAttemptExists",
    "SAESmokeAttemptFailed",
    "SAESmokeError",
    "SAESmokeIntegrityError",
    "SAESmokePathError",
    "SAESmokeRunner",
    "SAESmokeSequenceError",
    "SmokeCampaignIdentity",
    "SmokeCheckpointExpectation",
    "SmokeNumericalCallbacks",
    "SmokeOutcome",
    "SmokeRestoreResult",
    "SmokeSuiteAttestation",
    "SmokeSuiteConditionAttestation",
    "SmokeUpdateResult",
    "new_smoke_attempt_id",
]


SmokeCondition = Literal["F", "A", "B"]

SMOKE_CONDITIONS: tuple[SmokeCondition, ...] = ("F", "A", "B")
SMOKE_NAMESPACE = "sae-smoke"
SMOKE_PROTOCOL_SCHEMA = "gemma4-sae-full-shape-smoke-v1"
ATTEMPT_SCHEMA = "gemma4-sae-smoke-attempt-v1"
RESULT_SCHEMA = "gemma4-sae-smoke-result-v1"
FAILURE_SCHEMA = "gemma4-sae-smoke-failure-v1"
SUITE_ATTESTATION_SCHEMA = "gemma4-sae-smoke-suite-attestation-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ATTEMPT_ID_RE = re.compile(r"[0-9a-f]{32}")
_CONDITION_DIRECTORY = {"F": "00-F", "A": "01-A", "B": "02-B"}
_FORBIDDEN_PRODUCTION_COMPONENTS = frozenset(
    {
        "training",
        "run",
        "runs",
        "capacity",
        "capacities",
        "eval",
        "evaluation",
        "evaluations",
    }
)


class SAESmokeError(RuntimeError):
    """The smoke protocol cannot safely continue."""


class SAESmokePathError(SAESmokeError):
    """A smoke root overlaps a production namespace or is otherwise unsafe."""


class SAESmokeIntegrityError(SAESmokeError):
    """An immutable smoke record or numerical attestation is invalid."""


class SAESmokeSequenceError(SAESmokeError):
    """A condition was attempted outside the mandatory F, A, B order."""


class SAESmokeAttemptExists(SAESmokeError):
    """The supplied immutable attempt identity has already been reserved."""


class SAESmokeAttemptFailed(SAESmokeError):
    """A reserved attempt failed and has a durable failure record."""

    def __init__(
        self,
        message: str,
        *,
        attempt: SmokeAttemptIdentity,
        failure_path: Path,
    ) -> None:
        super().__init__(message)
        self.attempt = attempt
        self.failure_path = failure_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("smoke identity must be canonical JSON") from error
    return encoded.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-native value using the smoke protocol's encoding."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _without(payload: Mapping[str, Any], *fields: str) -> dict[str, Any]:
    excluded = set(fields)
    return {key: value for key, value in payload.items() if key not in excluded}


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_condition(value: Any) -> SmokeCondition:
    if value not in SMOKE_CONDITIONS:
        raise ValueError("smoke condition must be one of F, A, B")
    return value


def new_smoke_attempt_id() -> str:
    """Return a 128-bit lowercase hexadecimal attempt identifier."""

    return secrets.token_hex(16)


@dataclass(frozen=True, slots=True)
class SAEFullShape:
    """All dimensions which distinguish a real A/B/F SAE allocation."""

    hidden_dim: int
    num_experts: int
    bank_width: int
    top_routed: int
    topk_features: int
    effective_batch_tokens: int

    def __post_init__(self) -> None:
        for name in (
            "hidden_dim",
            "num_experts",
            "bank_width",
            "top_routed",
            "topk_features",
            "effective_batch_tokens",
        ):
            _require_int(getattr(self, name), name, minimum=1)
        if self.top_routed > self.num_experts:
            raise ValueError("top_routed cannot exceed num_experts")
        if self.topk_features > self.top_routed * self.bank_width:
            raise ValueError("topk_features exceeds A/B candidates per token")

    @classmethod
    def gemma4_26b_a4b(cls) -> SAEFullShape:
        """Return the production Gemma 4 experiment dimensions."""

        return cls(
            hidden_dim=2_816,
            num_experts=128,
            bank_width=1_024,
            top_routed=8,
            topk_features=64,
            effective_batch_tokens=16_384,
        )

    def to_mapping(self) -> dict[str, int | str]:
        return {
            "schema_version": "gemma4-sae-full-shape-v1",
            **asdict(self),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_mapping())

    def for_condition(self, condition: SmokeCondition) -> ConditionShape:
        selected = _require_condition(condition)
        if selected == "A":
            dictionary_layout = "router-conditioned-banks"
            bank_count = self.num_experts
            features_per_bank = self.bank_width
            total_features = self.num_experts * self.bank_width
            candidates = self.top_routed * self.bank_width
        elif selected == "B":
            dictionary_layout = "global"
            bank_count = 1
            features_per_bank = self.top_routed * self.bank_width
            total_features = features_per_bank
            candidates = total_features
        else:
            dictionary_layout = "global"
            bank_count = 1
            features_per_bank = self.num_experts * self.bank_width
            total_features = features_per_bank
            candidates = total_features
        return ConditionShape(
            condition=selected,
            dictionary_layout=dictionary_layout,
            hidden_dim=self.hidden_dim,
            bank_count=bank_count,
            features_per_bank=features_per_bank,
            total_features=total_features,
            candidates_per_token=candidates,
            topk_features=self.topk_features,
            effective_batch_tokens=self.effective_batch_tokens,
            parameter_count=total_features * (2 * self.hidden_dim + 1),
        )


@dataclass(frozen=True, slots=True)
class ConditionShape:
    """Condition-specific physical shape sealed into every attempt."""

    condition: SmokeCondition
    dictionary_layout: str
    hidden_dim: int
    bank_count: int
    features_per_bank: int
    total_features: int
    candidates_per_token: int
    topk_features: int
    effective_batch_tokens: int
    parameter_count: int

    def __post_init__(self) -> None:
        _require_condition(self.condition)
        if self.dictionary_layout not in {"router-conditioned-banks", "global"}:
            raise ValueError("unsupported dictionary layout")
        for name in (
            "hidden_dim",
            "bank_count",
            "features_per_bank",
            "total_features",
            "candidates_per_token",
            "topk_features",
            "effective_batch_tokens",
            "parameter_count",
        ):
            _require_int(getattr(self, name), name, minimum=1)
        if self.bank_count * self.features_per_bank != self.total_features:
            raise ValueError("condition bank shape does not equal total_features")
        if self.topk_features > self.candidates_per_token:
            raise ValueError("topk_features exceeds condition candidates")
        if self.parameter_count != self.total_features * (2 * self.hidden_dim + 1):
            raise ValueError("condition parameter_count is inconsistent")

    def to_mapping(self) -> dict[str, int | str]:
        return {
            "schema_version": "gemma4-sae-condition-full-shape-v1",
            **asdict(self),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_mapping())


def _root_components(value: str) -> tuple[str, ...]:
    if value.startswith("gs://"):
        remainder = value[5:]
        _bucket, separator, object_name = remainder.partition("/")
        if not separator:
            return ()
        return tuple(component for component in object_name.split("/") if component)
    return Path(value).parts


def _reject_production_components(value: str, label: str) -> None:
    overlap = {
        component.lower()
        for component in _root_components(value)
        if component.lower() in _FORBIDDEN_PRODUCTION_COMPONENTS
    }
    if overlap:
        joined = ", ".join(sorted(overlap))
        raise SAESmokePathError(
            f"{label} overlaps forbidden production path component(s): {joined}"
        )


def normalize_checkpoint_payload_root(root: str | Path) -> str:
    """Normalize a local/GCS root and reject production artifact namespaces."""

    value = str(root).strip()
    if not value:
        raise SAESmokePathError("checkpoint payload root cannot be empty")
    if "://" in value and not value.startswith("gs://"):
        raise SAESmokePathError("checkpoint payload root must be local or gs://")
    if value.startswith("gs://"):
        remainder = value[5:]
        bucket, separator, object_name = remainder.partition("/")
        if not bucket or bucket in {".", ".."}:
            raise SAESmokePathError("GCS payload root must include a bucket")
        raw_components = object_name.split("/") if separator else []
        if any(component in {".", ".."} for component in raw_components):
            raise SAESmokePathError("GCS payload root cannot contain dot segments")
        components = tuple(component for component in raw_components if component)
        normalized = f"gs://{bucket}"
        if components:
            normalized += "/" + "/".join(components)
    else:
        normalized = str(Path(value).expanduser().resolve(strict=False))
    _reject_production_components(normalized, "checkpoint payload root")
    return normalized


def normalize_control_root(root: str | Path) -> Path:
    """Normalize a POSIX control root without creating it."""

    value = str(root).strip()
    if not value or "://" in value:
        raise SAESmokePathError("smoke control root must be a POSIX path")
    normalized = Path(value).expanduser().resolve(strict=False)
    _reject_production_components(str(normalized), "smoke control root")
    return normalized


def _join_payload(root: str, *parts: str) -> str:
    if root.startswith("gs://"):
        return "/".join((root.rstrip("/"), *(part.strip("/") for part in parts)))
    return str(Path(root).joinpath(*parts))


@dataclass(frozen=True, slots=True)
class SmokeCampaignIdentity:
    """Pre-attestation inputs, full shape, and external-root identity.

    Smoke execution deliberately precedes SAE streaming-cache qualification,
    so this identity binds its actual capture/environment prerequisites
    directly.  It does not contain an SAE-storage attestation or depend on a
    later production-operation digest which contains one.
    """

    config_sha256: str
    repository_content_sha256: str
    train_capture_manifest_sha256: str
    eval_capture_manifest_sha256: str
    consumer_environment_sha256: str
    capture_compatibility_sha256: str
    full_shape: SAEFullShape
    checkpoint_payload_root: str | Path

    def __post_init__(self) -> None:
        for name in (
            "config_sha256",
            "repository_content_sha256",
            "train_capture_manifest_sha256",
            "eval_capture_manifest_sha256",
            "consumer_environment_sha256",
            "capture_compatibility_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.full_shape, SAEFullShape):
            raise TypeError("full_shape must be an SAEFullShape")
        object.__setattr__(
            self,
            "checkpoint_payload_root",
            normalize_checkpoint_payload_root(self.checkpoint_payload_root),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SMOKE_PROTOCOL_SCHEMA,
            "config_sha256": self.config_sha256,
            "repository_content_sha256": self.repository_content_sha256,
            "train_capture_manifest_sha256": self.train_capture_manifest_sha256,
            "eval_capture_manifest_sha256": self.eval_capture_manifest_sha256,
            "consumer_environment_sha256": self.consumer_environment_sha256,
            "capture_compatibility_sha256": self.capture_compatibility_sha256,
            "full_shape": self.full_shape.to_mapping(),
            "full_shape_sha256": self.full_shape.sha256,
            "checkpoint_payload_root": self.checkpoint_payload_root,
            "condition_order": list(SMOKE_CONDITIONS),
            "updates_per_attempt": 1,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_mapping())


@dataclass(frozen=True, slots=True)
class SmokeAttemptIdentity:
    """Immutable identity supplied to every numerical callback."""

    campaign_sha256: str
    condition: SmokeCondition
    condition_index: int
    condition_attempt_number: int
    attempt_id: str
    full_shape_sha256: str
    condition_shape: ConditionShape
    cache_path: str
    checkpoint_uri: str
    expected_initial_step: int
    expected_final_step: int
    expected_initial_tokens_seen: int
    expected_final_tokens_seen: int

    def __post_init__(self) -> None:
        _require_sha256(self.campaign_sha256, "campaign_sha256")
        _require_sha256(self.full_shape_sha256, "full_shape_sha256")
        selected = _require_condition(self.condition)
        if self.condition_index != SMOKE_CONDITIONS.index(selected):
            raise ValueError("condition_index does not match the fixed order")
        _require_int(
            self.condition_attempt_number,
            "condition_attempt_number",
            minimum=1,
        )
        if not isinstance(self.attempt_id, str) or (
            _ATTEMPT_ID_RE.fullmatch(self.attempt_id) is None
        ):
            raise ValueError("attempt_id must be 32 lowercase hexadecimal digits")
        if not isinstance(self.condition_shape, ConditionShape):
            raise TypeError("condition_shape must be a ConditionShape")
        if self.condition_shape.condition != selected:
            raise ValueError("condition shape belongs to another condition")
        if not isinstance(self.cache_path, str) or not self.cache_path:
            raise ValueError("cache_path cannot be empty")
        if not isinstance(self.checkpoint_uri, str) or not self.checkpoint_uri:
            raise ValueError("checkpoint_uri cannot be empty")
        expected = (0, 1, 0, self.condition_shape.effective_batch_tokens)
        observed = (
            self.expected_initial_step,
            self.expected_final_step,
            self.expected_initial_tokens_seen,
            self.expected_final_tokens_seen,
        )
        if observed != expected:
            raise ValueError("smoke progress must be step 0->1 over one full batch")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": ATTEMPT_SCHEMA,
            "campaign_sha256": self.campaign_sha256,
            "condition": self.condition,
            "condition_index": self.condition_index,
            "condition_attempt_number": self.condition_attempt_number,
            "attempt_id": self.attempt_id,
            "full_shape_sha256": self.full_shape_sha256,
            "condition_shape": self.condition_shape.to_mapping(),
            "condition_shape_sha256": self.condition_shape.sha256,
            "cache_path": self.cache_path,
            "checkpoint_uri": self.checkpoint_uri,
            "expected_initial_step": self.expected_initial_step,
            "expected_final_step": self.expected_final_step,
            "expected_initial_tokens_seen": self.expected_initial_tokens_seen,
            "expected_final_tokens_seen": self.expected_final_tokens_seen,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_mapping())


@dataclass(frozen=True, slots=True)
class SmokeAttemptPaths:
    """All local/external paths reserved for one immutable attempt."""

    namespace_root: Path
    campaign_root: Path
    condition_root: Path
    attempt_root: Path
    attempt_record: Path
    result_record: Path
    failure_record: Path
    cache_path: Path
    checkpoint_uri: str


@dataclass(frozen=True, slots=True)
class SmokeUpdateResult:
    """Attestation returned after constructing state and applying one update."""

    state: object
    initial_step: int
    final_step: int
    initial_tokens_seen: int
    final_tokens_seen: int
    optimizer_updates: int
    attempt_sha256: str
    condition_shape_sha256: str
    state_sha256: str


@dataclass(frozen=True, slots=True)
class SmokeCheckpointExpectation:
    """Exact metadata that a full Orbax save and restore must preserve."""

    attempt_sha256: str
    condition_shape_sha256: str
    checkpoint_uri: str
    step: int
    tokens_seen: int
    optimizer_updates: int
    state_sha256: str


@dataclass(frozen=True, slots=True)
class FinalizedCheckpoint:
    """Independent external inspection of a finalized Orbax payload."""

    checkpoint_uri: str
    finalized: bool
    step: int
    tokens_seen: int
    optimizer_updates: int
    attempt_sha256: str
    condition_shape_sha256: str
    state_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class SmokeRestoreResult:
    """Attestation produced from the state restored from Orbax."""

    state: object
    step: int
    tokens_seen: int
    optimizer_updates: int
    attempt_sha256: str
    condition_shape_sha256: str
    state_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class SmokeNumericalCallbacks:
    """Injected boundary between this control protocol and numerical code.

    ``save_orbax`` must synchronously finalize the complete state at the exact
    expectation URI.  ``inspect_finalized_checkpoint`` must inspect that
    external payload independently of any value returned by the saver.
    """

    update_once: Callable[[SmokeAttemptIdentity], SmokeUpdateResult]
    save_orbax: Callable[[object, SmokeCheckpointExpectation], None]
    inspect_finalized_checkpoint: Callable[[str], FinalizedCheckpoint]
    restore_orbax: Callable[[str, SmokeCheckpointExpectation], SmokeRestoreResult]


@dataclass(frozen=True, slots=True)
class SmokeOutcome:
    """Validated success; ``outcome_sha256`` is safe for operator consensus."""

    attempt: SmokeAttemptIdentity
    paths: SmokeAttemptPaths
    checkpoint_sha256: str
    state_sha256: str
    outcome_sha256: str
    record_sha256: str


@dataclass(frozen=True, slots=True)
class SmokeSuiteConditionAttestation:
    """The exact successful attempt sealed for one suite condition."""

    condition: SmokeCondition
    attempt_id: str
    condition_attempt_number: int
    attempt_sha256: str
    attempt_record_sha256: str
    outcome_sha256: str
    result_record_sha256: str
    checkpoint_sha256: str
    state_sha256: str

    def __post_init__(self) -> None:
        _require_condition(self.condition)
        if not isinstance(self.attempt_id, str) or (
            _ATTEMPT_ID_RE.fullmatch(self.attempt_id) is None
        ):
            raise ValueError("attempt_id must be 32 lowercase hexadecimal digits")
        _require_int(
            self.condition_attempt_number,
            "condition_attempt_number",
            minimum=1,
        )
        for name in (
            "attempt_sha256",
            "attempt_record_sha256",
            "outcome_sha256",
            "result_record_sha256",
            "checkpoint_sha256",
            "state_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SmokeSuiteAttestation:
    """Canonical F -> A -> B handoff into a production SAE operation."""

    campaign_sha256: str
    conditions: tuple[SmokeSuiteConditionAttestation, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.campaign_sha256, "campaign_sha256")
        if not isinstance(self.conditions, tuple) or any(
            not isinstance(item, SmokeSuiteConditionAttestation)
            for item in self.conditions
        ):
            raise TypeError(
                "conditions must be a tuple of SmokeSuiteConditionAttestation"
            )
        observed = tuple(item.condition for item in self.conditions)
        if observed != SMOKE_CONDITIONS:
            raise ValueError("smoke suite conditions must be exactly F, A, B in order")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SUITE_ATTESTATION_SCHEMA,
            "campaign_sha256": self.campaign_sha256,
            "condition_order": list(SMOKE_CONDITIONS),
            "conditions": [item.to_mapping() for item in self.conditions],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_mapping())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish one immutable JSON object without replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(
                    dict(payload),
                    sort_keys=True,
                    indent=2,
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            )
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SAESmokeIntegrityError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise SAESmokeIntegrityError(f"{label} is not a JSON object: {path}")
    return value


def _parse_utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise SAESmokeIntegrityError(f"{label} is not a timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise SAESmokeIntegrityError(f"{label} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SAESmokeIntegrityError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


@contextmanager
def _exclusive_campaign_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b", buffering=0) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _attempt_from_mapping(value: Mapping[str, Any]) -> SmokeAttemptIdentity:
    try:
        shape_value = value["condition_shape"]
        if not isinstance(shape_value, Mapping):
            raise TypeError("condition_shape is not an object")
        shape_fields = {
            key: shape_value[key]
            for key in (
                "condition",
                "dictionary_layout",
                "hidden_dim",
                "bank_count",
                "features_per_bank",
                "total_features",
                "candidates_per_token",
                "topk_features",
                "effective_batch_tokens",
                "parameter_count",
            )
        }
        condition_shape = ConditionShape(**shape_fields)  # type: ignore[arg-type]
        identity = SmokeAttemptIdentity(
            campaign_sha256=value["campaign_sha256"],
            condition=value["condition"],
            condition_index=value["condition_index"],
            condition_attempt_number=value["condition_attempt_number"],
            attempt_id=value["attempt_id"],
            full_shape_sha256=value["full_shape_sha256"],
            condition_shape=condition_shape,
            cache_path=value["cache_path"],
            checkpoint_uri=value["checkpoint_uri"],
            expected_initial_step=value["expected_initial_step"],
            expected_final_step=value["expected_final_step"],
            expected_initial_tokens_seen=value["expected_initial_tokens_seen"],
            expected_final_tokens_seen=value["expected_final_tokens_seen"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SAESmokeIntegrityError("attempt identity is invalid") from error
    if dict(value) != identity.to_mapping():
        raise SAESmokeIntegrityError("attempt identity fields do not match schema")
    return identity


class SAESmokeRunner:
    """Durable coordinator for the three full-shape smoke attempts."""

    def __init__(
        self,
        control_root: str | Path,
        campaign: SmokeCampaignIdentity,
        *,
        smoke_cache_root: str | Path,
    ) -> None:
        if not isinstance(campaign, SmokeCampaignIdentity):
            raise TypeError("campaign must be a SmokeCampaignIdentity")
        self.control_root = normalize_control_root(control_root)
        self.smoke_cache_root = normalize_control_root(smoke_cache_root)
        self.campaign = campaign
        self.namespace_root = self.control_root / SMOKE_NAMESPACE
        self.campaign_root = (
            self.namespace_root / "campaigns" / self.campaign.sha256
        )
        try:
            self.campaign_root.relative_to(self.control_root)
        except ValueError as error:
            raise SAESmokePathError("smoke campaign path escapes control root") from error

    def paths(
        self,
        condition: SmokeCondition,
        attempt_id: str,
        *,
        condition_attempt_number: int,
    ) -> SmokeAttemptPaths:
        """Return paths without creating them."""

        selected = _require_condition(condition)
        if not isinstance(attempt_id, str) or (
            _ATTEMPT_ID_RE.fullmatch(attempt_id) is None
        ):
            raise ValueError("attempt_id must be 32 lowercase hexadecimal digits")
        _require_int(
            condition_attempt_number,
            "condition_attempt_number",
            minimum=1,
        )
        condition_root = self.campaign_root / "conditions" / _CONDITION_DIRECTORY[selected]
        attempt_root = condition_root / "attempts" / attempt_id
        try:
            attempt_root.relative_to(self.namespace_root)
        except ValueError as error:
            raise SAESmokePathError("smoke attempt path escapes its namespace") from error
        checkpoint_name = (
            "checkpoint-step-00000001-tokens-"
            f"{self.campaign.full_shape.effective_batch_tokens:012d}"
        )
        checkpoint_uri = _join_payload(
            str(self.campaign.checkpoint_payload_root),
            SMOKE_NAMESPACE,
            "campaigns",
            self.campaign.sha256,
            "conditions",
            _CONDITION_DIRECTORY[selected],
            "attempts",
            attempt_id,
            checkpoint_name,
        )
        cache_path = (
            self.smoke_cache_root
            / SMOKE_NAMESPACE
            / "campaigns"
            / self.campaign.sha256
            / "conditions"
            / _CONDITION_DIRECTORY[selected]
            / "attempts"
            / attempt_id
            / "cache"
        )
        try:
            cache_path.relative_to(self.smoke_cache_root)
        except ValueError as error:
            raise SAESmokePathError("smoke cache path escapes its namespace") from error
        return SmokeAttemptPaths(
            namespace_root=self.namespace_root,
            campaign_root=self.campaign_root,
            condition_root=condition_root,
            attempt_root=attempt_root,
            attempt_record=attempt_root / "attempt.json",
            result_record=attempt_root / "result.json",
            failure_record=attempt_root / "failure.json",
            cache_path=cache_path,
            checkpoint_uri=checkpoint_uri,
        )

    def _attempt_count(self, condition: SmokeCondition) -> int:
        root = (
            self.campaign_root
            / "conditions"
            / _CONDITION_DIRECTORY[condition]
            / "attempts"
        )
        if not root.exists():
            return 0
        try:
            return sum(1 for item in root.iterdir() if item.is_dir())
        except OSError as error:
            raise SAESmokeIntegrityError("cannot enumerate smoke attempts") from error

    def _build_attempt(
        self,
        condition: SmokeCondition,
        attempt_id: str,
        condition_attempt_number: int,
        paths: SmokeAttemptPaths,
    ) -> SmokeAttemptIdentity:
        shape = self.campaign.full_shape.for_condition(condition)
        return SmokeAttemptIdentity(
            campaign_sha256=self.campaign.sha256,
            condition=condition,
            condition_index=SMOKE_CONDITIONS.index(condition),
            condition_attempt_number=condition_attempt_number,
            attempt_id=attempt_id,
            full_shape_sha256=self.campaign.full_shape.sha256,
            condition_shape=shape,
            cache_path=str(paths.cache_path),
            checkpoint_uri=paths.checkpoint_uri,
            expected_initial_step=0,
            expected_final_step=1,
            expected_initial_tokens_seen=0,
            expected_final_tokens_seen=shape.effective_batch_tokens,
        )

    def _reserve_attempt(
        self, condition: SmokeCondition, attempt_id: str
    ) -> tuple[SmokeAttemptIdentity, SmokeAttemptPaths]:
        number = self._attempt_count(condition) + 1
        paths = self.paths(
            condition,
            attempt_id,
            condition_attempt_number=number,
        )
        if paths.attempt_root.exists():
            raise SAESmokeAttemptExists(
                f"smoke attempt {attempt_id} has already been reserved"
            )
        attempt = self._build_attempt(condition, attempt_id, number, paths)
        record: dict[str, Any] = {
            "schema_version": ATTEMPT_SCHEMA,
            "status": "reserved",
            "reserved_at": _utc_now(),
            "identity": attempt.to_mapping(),
            "attempt_sha256": attempt.sha256,
        }
        record["record_sha256"] = canonical_sha256(record)
        try:
            _publish_json_exclusive(paths.attempt_record, record)
        except FileExistsError as error:
            raise SAESmokeAttemptExists(
                f"smoke attempt {attempt_id} has already been reserved"
            ) from error
        return attempt, paths

    def _validate_attempt_record(
        self, path: Path, *, expected_attempt_id: str | None = None
    ) -> tuple[SmokeAttemptIdentity, dict[str, Any]]:
        record = _read_json_object(path, "smoke attempt")
        expected_keys = {
            "schema_version",
            "status",
            "reserved_at",
            "identity",
            "attempt_sha256",
            "record_sha256",
        }
        if set(record) != expected_keys or record.get("schema_version") != ATTEMPT_SCHEMA:
            raise SAESmokeIntegrityError("smoke attempt record fields do not match schema")
        if record.get("status") != "reserved":
            raise SAESmokeIntegrityError("smoke attempt status is invalid")
        supplied_record_hash = record.get("record_sha256")
        if (
            not isinstance(supplied_record_hash, str)
            or _SHA256_RE.fullmatch(supplied_record_hash) is None
            or canonical_sha256(_without(record, "record_sha256"))
            != supplied_record_hash
        ):
            raise SAESmokeIntegrityError("smoke attempt record self-hash mismatch")
        identity_value = record.get("identity")
        if not isinstance(identity_value, Mapping):
            raise SAESmokeIntegrityError("smoke attempt identity is not an object")
        identity = _attempt_from_mapping(identity_value)
        if identity.sha256 != record.get("attempt_sha256"):
            raise SAESmokeIntegrityError("smoke attempt identity hash mismatch")
        if identity.campaign_sha256 != self.campaign.sha256:
            raise SAESmokeIntegrityError("smoke attempt belongs to another campaign")
        expected_shape = self.campaign.full_shape.for_condition(identity.condition)
        if identity.full_shape_sha256 != self.campaign.full_shape.sha256 or (
            identity.condition_shape != expected_shape
        ):
            raise SAESmokeIntegrityError("smoke attempt full shape mismatch")
        expected_paths = self.paths(
            identity.condition,
            identity.attempt_id,
            condition_attempt_number=identity.condition_attempt_number,
        )
        if (
            path != expected_paths.attempt_record
            or identity.cache_path != str(expected_paths.cache_path)
            or identity.checkpoint_uri != expected_paths.checkpoint_uri
        ):
            raise SAESmokeIntegrityError("smoke attempt path identity mismatch")
        if expected_attempt_id is not None and identity.attempt_id != expected_attempt_id:
            raise SAESmokeIntegrityError("smoke attempt directory name mismatch")
        return identity, record

    def _validate_result_record(
        self, path: Path, attempt: SmokeAttemptIdentity
    ) -> dict[str, Any]:
        record = _read_json_object(path, "smoke result")
        expected_keys = {
            "schema_version",
            "status",
            "completed_at",
            "attempt_sha256",
            "campaign_sha256",
            "condition",
            "condition_shape_sha256",
            "checkpoint_uri",
            "checkpoint_sha256",
            "state_sha256",
            "step",
            "tokens_seen",
            "optimizer_updates",
            "outcome_sha256",
            "record_sha256",
        }
        if set(record) != expected_keys or record.get("schema_version") != RESULT_SCHEMA:
            raise SAESmokeIntegrityError("smoke result fields do not match schema")
        supplied_record_hash = record.get("record_sha256")
        if (
            not isinstance(supplied_record_hash, str)
            or _SHA256_RE.fullmatch(supplied_record_hash) is None
            or canonical_sha256(_without(record, "record_sha256"))
            != supplied_record_hash
        ):
            raise SAESmokeIntegrityError("smoke result self-hash mismatch")
        deterministic = _without(
            record,
            "schema_version",
            "completed_at",
            "outcome_sha256",
            "record_sha256",
        )
        if canonical_sha256(deterministic) != record.get("outcome_sha256"):
            raise SAESmokeIntegrityError("smoke deterministic outcome hash mismatch")
        expected = {
            "status": "passed",
            "attempt_sha256": attempt.sha256,
            "campaign_sha256": attempt.campaign_sha256,
            "condition": attempt.condition,
            "condition_shape_sha256": attempt.condition_shape.sha256,
            "checkpoint_uri": attempt.checkpoint_uri,
            "step": attempt.expected_final_step,
            "tokens_seen": attempt.expected_final_tokens_seen,
            "optimizer_updates": 1,
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise SAESmokeIntegrityError(f"smoke result {field} mismatch")
        for field in ("checkpoint_sha256", "state_sha256", "outcome_sha256"):
            try:
                _require_sha256(record.get(field), field)
            except ValueError as error:
                raise SAESmokeIntegrityError(str(error)) from error
        return record

    def _validate_failure_record(
        self, path: Path, attempt: SmokeAttemptIdentity
    ) -> dict[str, Any]:
        record = _read_json_object(path, "smoke failure")
        expected_keys = {
            "schema_version",
            "status",
            "failed_at",
            "attempt_sha256",
            "campaign_sha256",
            "condition",
            "stage",
            "error_type",
            "error_message",
            "record_sha256",
        }
        if set(record) != expected_keys or record.get("schema_version") != FAILURE_SCHEMA:
            raise SAESmokeIntegrityError("smoke failure fields do not match schema")
        supplied_hash = record.get("record_sha256")
        if (
            not isinstance(supplied_hash, str)
            or _SHA256_RE.fullmatch(supplied_hash) is None
            or canonical_sha256(_without(record, "record_sha256")) != supplied_hash
        ):
            raise SAESmokeIntegrityError("smoke failure self-hash mismatch")
        expected = {
            "status": "failed",
            "attempt_sha256": attempt.sha256,
            "campaign_sha256": attempt.campaign_sha256,
            "condition": attempt.condition,
        }
        if any(record.get(field) != value for field, value in expected.items()):
            raise SAESmokeIntegrityError("smoke failure identity mismatch")
        if not all(
            isinstance(record.get(field), str) and record[field]
            for field in ("stage", "error_type", "error_message")
        ):
            raise SAESmokeIntegrityError("smoke failure detail is invalid")
        return record

    def _successful_attempts(self) -> dict[SmokeCondition, list[SmokeAttemptIdentity]]:
        successes: dict[SmokeCondition, list[SmokeAttemptIdentity]] = {
            condition: [] for condition in SMOKE_CONDITIONS
        }
        for condition in SMOKE_CONDITIONS:
            attempts_root = (
                self.campaign_root
                / "conditions"
                / _CONDITION_DIRECTORY[condition]
                / "attempts"
            )
            if not attempts_root.exists():
                continue
            for attempt_root in sorted(attempts_root.iterdir()):
                if not attempt_root.is_dir():
                    raise SAESmokeIntegrityError(
                        f"unexpected entry in smoke attempts: {attempt_root}"
                    )
                attempt, _attempt_record = self._validate_attempt_record(
                    attempt_root / "attempt.json",
                    expected_attempt_id=attempt_root.name,
                )
                if attempt.condition != condition:
                    raise SAESmokeIntegrityError("attempt stored under wrong condition")
                result_exists = (attempt_root / "result.json").exists()
                failure_exists = (attempt_root / "failure.json").exists()
                if result_exists and failure_exists:
                    raise SAESmokeIntegrityError(
                        "smoke attempt contains both success and failure"
                    )
                if result_exists:
                    self._validate_result_record(attempt_root / "result.json", attempt)
                    successes[condition].append(attempt)
                elif failure_exists:
                    self._validate_failure_record(
                        attempt_root / "failure.json", attempt
                    )
        return successes

    def validate_passed_suite(self) -> SmokeSuiteAttestation:
        """Revalidate and canonically bind one clean, complete F -> A -> B suite.

        This is intentionally a fresh filesystem walk, not a summary of values
        returned by :meth:`run_attempt`.  It validates every immutable attempt
        and terminal record in the campaign, requires exactly one success per
        condition, and binds the successful attempt/result/checkpoint/state
        hashes into the value consumed by the production operation identity.
        """

        lock_path = self.campaign_root / ".coordinator.lock"
        with _exclusive_campaign_lock(lock_path):
            if self.campaign_root.is_symlink():
                raise SAESmokeIntegrityError("smoke campaign root cannot be a symlink")
            if self.campaign_root.name != self.campaign.sha256:
                raise SAESmokeIntegrityError("smoke campaign path identity mismatch")

            passed: list[
                tuple[
                    SmokeSuiteConditionAttestation,
                    datetime,
                    datetime,
                ]
            ] = []
            for condition in SMOKE_CONDITIONS:
                attempts_root = (
                    self.campaign_root
                    / "conditions"
                    / _CONDITION_DIRECTORY[condition]
                    / "attempts"
                )
                if attempts_root.is_symlink() or not attempts_root.is_dir():
                    raise SAESmokeIntegrityError(
                        f"condition {condition} has no valid smoke attempts directory"
                    )

                attempts: list[
                    tuple[
                        SmokeAttemptIdentity,
                        dict[str, Any],
                        dict[str, Any] | None,
                    ]
                ] = []
                try:
                    attempt_roots = sorted(attempts_root.iterdir())
                except OSError as error:
                    raise SAESmokeIntegrityError(
                        f"cannot enumerate condition {condition} smoke attempts"
                    ) from error
                for attempt_root in attempt_roots:
                    if attempt_root.is_symlink() or not attempt_root.is_dir():
                        raise SAESmokeIntegrityError(
                            f"unexpected entry in smoke attempts: {attempt_root}"
                        )
                    attempt_path = attempt_root / "attempt.json"
                    if attempt_path.is_symlink() or not attempt_path.is_file():
                        raise SAESmokeIntegrityError(
                            f"smoke attempt record path is invalid: {attempt_path}"
                        )
                    attempt, attempt_record = self._validate_attempt_record(
                        attempt_path,
                        expected_attempt_id=attempt_root.name,
                    )
                    if attempt.condition != condition:
                        raise SAESmokeIntegrityError(
                            "smoke attempt is stored under the wrong condition"
                        )

                    expected_entries = {"attempt.json"}
                    result_path = attempt_root / "result.json"
                    failure_path = attempt_root / "failure.json"
                    result_exists = result_path.exists() or result_path.is_symlink()
                    failure_exists = failure_path.exists() or failure_path.is_symlink()
                    if result_exists == failure_exists:
                        raise SAESmokeIntegrityError(
                            "smoke attempt must contain exactly one terminal record"
                        )

                    terminal_record: dict[str, Any]
                    result_record: dict[str, Any] | None
                    if result_exists:
                        if result_path.is_symlink() or not result_path.is_file():
                            raise SAESmokeIntegrityError(
                                f"smoke result record path is invalid: {result_path}"
                            )
                        terminal_record = self._validate_result_record(
                            result_path, attempt
                        )
                        result_record = terminal_record
                        expected_entries.add("result.json")
                        terminal_at = _parse_utc_timestamp(
                            terminal_record.get("completed_at"),
                            "smoke result completed_at",
                        )
                    else:
                        if failure_path.is_symlink() or not failure_path.is_file():
                            raise SAESmokeIntegrityError(
                                f"smoke failure record path is invalid: {failure_path}"
                            )
                        terminal_record = self._validate_failure_record(
                            failure_path, attempt
                        )
                        result_record = None
                        expected_entries.add("failure.json")
                        terminal_at = _parse_utc_timestamp(
                            terminal_record.get("failed_at"),
                            "smoke failure failed_at",
                        )

                    try:
                        observed_entries = {item.name for item in attempt_root.iterdir()}
                    except OSError as error:
                        raise SAESmokeIntegrityError(
                            f"cannot enumerate smoke attempt: {attempt_root}"
                        ) from error
                    if observed_entries != expected_entries:
                        raise SAESmokeIntegrityError(
                            f"unexpected entries in smoke attempt: {attempt_root}"
                        )

                    reserved_at = _parse_utc_timestamp(
                        attempt_record.get("reserved_at"),
                        "smoke attempt reserved_at",
                    )
                    if terminal_at < reserved_at:
                        raise SAESmokeIntegrityError(
                            "smoke terminal record predates its attempt reservation"
                        )
                    attempts.append((attempt, attempt_record, result_record))

                ordered_attempts = sorted(
                    attempts, key=lambda item: item[0].condition_attempt_number
                )
                observed_numbers = tuple(
                    item[0].condition_attempt_number for item in ordered_attempts
                )
                expected_numbers = tuple(range(1, len(ordered_attempts) + 1))
                if observed_numbers != expected_numbers:
                    raise SAESmokeIntegrityError(
                        f"condition {condition} smoke attempt numbers are not contiguous"
                    )

                successes = [item for item in ordered_attempts if item[2] is not None]
                if len(successes) != 1:
                    raise SAESmokeIntegrityError(
                        f"condition {condition} must have exactly one passed smoke attempt"
                    )
                attempt, attempt_record, result_record_or_none = successes[0]
                assert result_record_or_none is not None
                result_record = result_record_or_none
                reserved_at = _parse_utc_timestamp(
                    attempt_record.get("reserved_at"),
                    "passed smoke attempt reserved_at",
                )
                completed_at = _parse_utc_timestamp(
                    result_record.get("completed_at"),
                    "passed smoke result completed_at",
                )
                passed.append(
                    (
                        SmokeSuiteConditionAttestation(
                            condition=condition,
                            attempt_id=attempt.attempt_id,
                            condition_attempt_number=attempt.condition_attempt_number,
                            attempt_sha256=attempt.sha256,
                            attempt_record_sha256=attempt_record["record_sha256"],
                            outcome_sha256=result_record["outcome_sha256"],
                            result_record_sha256=result_record["record_sha256"],
                            checkpoint_sha256=result_record["checkpoint_sha256"],
                            state_sha256=result_record["state_sha256"],
                        ),
                        reserved_at,
                        completed_at,
                    )
                )

            for previous, current in zip(passed, passed[1:], strict=False):
                if current[1] < previous[2]:
                    raise SAESmokeIntegrityError(
                        "passed smoke attempts do not follow mandatory F, A, B order"
                    )
            return SmokeSuiteAttestation(
                campaign_sha256=self.campaign.sha256,
                conditions=tuple(item[0] for item in passed),
            )

    def _validate_sequence(self, condition: SmokeCondition) -> None:
        successes = self._successful_attempts()
        index = SMOKE_CONDITIONS.index(condition)
        for earlier in SMOKE_CONDITIONS[:index]:
            if len(successes[earlier]) != 1:
                raise SAESmokeSequenceError(
                    f"condition {condition} requires exactly one passed {earlier} smoke"
                )
        if successes[condition]:
            raise SAESmokeSequenceError(
                f"condition {condition} already has a passed smoke attempt"
            )
        for later in SMOKE_CONDITIONS[index + 1 :]:
            if successes[later]:
                raise SAESmokeSequenceError(
                    f"condition {later} passed before {condition}; campaign is invalid"
                )
        if any(len(items) > 1 for items in successes.values()):
            raise SAESmokeIntegrityError(
                "a condition has multiple passed smoke attempts"
            )

    @staticmethod
    def _validate_update(
        update: SmokeUpdateResult, attempt: SmokeAttemptIdentity
    ) -> SmokeCheckpointExpectation:
        if not isinstance(update, SmokeUpdateResult):
            raise SAESmokeIntegrityError(
                "update_once must return SmokeUpdateResult"
            )
        expected_progress = (
            attempt.expected_initial_step,
            attempt.expected_final_step,
            attempt.expected_initial_tokens_seen,
            attempt.expected_final_tokens_seen,
            1,
        )
        observed_progress = (
            update.initial_step,
            update.final_step,
            update.initial_tokens_seen,
            update.final_tokens_seen,
            update.optimizer_updates,
        )
        if observed_progress != expected_progress:
            raise SAESmokeIntegrityError(
                "optimizer update did not advance exactly step 0->1 over one full batch"
            )
        if update.attempt_sha256 != attempt.sha256:
            raise SAESmokeIntegrityError("optimizer update attempt identity mismatch")
        if update.condition_shape_sha256 != attempt.condition_shape.sha256:
            raise SAESmokeIntegrityError("optimizer update full-shape mismatch")
        try:
            _require_sha256(update.state_sha256, "state_sha256")
        except ValueError as error:
            raise SAESmokeIntegrityError(str(error)) from error
        return SmokeCheckpointExpectation(
            attempt_sha256=attempt.sha256,
            condition_shape_sha256=attempt.condition_shape.sha256,
            checkpoint_uri=attempt.checkpoint_uri,
            step=attempt.expected_final_step,
            tokens_seen=attempt.expected_final_tokens_seen,
            optimizer_updates=1,
            state_sha256=update.state_sha256,
        )

    @staticmethod
    def _validate_finalized_checkpoint(
        checkpoint: FinalizedCheckpoint,
        expectation: SmokeCheckpointExpectation,
    ) -> None:
        if not isinstance(checkpoint, FinalizedCheckpoint):
            raise SAESmokeIntegrityError(
                "checkpoint inspector must return FinalizedCheckpoint"
            )
        if checkpoint.finalized is not True:
            raise SAESmokeIntegrityError("external checkpoint is not finalized")
        expected = {
            "checkpoint_uri": expectation.checkpoint_uri,
            "step": expectation.step,
            "tokens_seen": expectation.tokens_seen,
            "optimizer_updates": expectation.optimizer_updates,
            "attempt_sha256": expectation.attempt_sha256,
            "condition_shape_sha256": expectation.condition_shape_sha256,
            "state_sha256": expectation.state_sha256,
        }
        for field, value in expected.items():
            if getattr(checkpoint, field) != value:
                raise SAESmokeIntegrityError(
                    f"external finalized checkpoint {field} mismatch"
                )
        try:
            _require_sha256(checkpoint.checkpoint_sha256, "checkpoint_sha256")
        except ValueError as error:
            raise SAESmokeIntegrityError(str(error)) from error

    @staticmethod
    def _validate_restore(
        restored: SmokeRestoreResult,
        checkpoint: FinalizedCheckpoint,
        expectation: SmokeCheckpointExpectation,
    ) -> None:
        if not isinstance(restored, SmokeRestoreResult):
            raise SAESmokeIntegrityError(
                "restore_orbax must return SmokeRestoreResult"
            )
        expected = {
            "step": expectation.step,
            "tokens_seen": expectation.tokens_seen,
            "optimizer_updates": expectation.optimizer_updates,
            "attempt_sha256": expectation.attempt_sha256,
            "condition_shape_sha256": expectation.condition_shape_sha256,
            "state_sha256": expectation.state_sha256,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
        }
        for field, value in expected.items():
            if getattr(restored, field) != value:
                raise SAESmokeIntegrityError(f"restored checkpoint {field} mismatch")

    @staticmethod
    def _publish_failure(
        paths: SmokeAttemptPaths,
        attempt: SmokeAttemptIdentity,
        *,
        stage: str,
        error: Exception,
    ) -> None:
        record: dict[str, Any] = {
            "schema_version": FAILURE_SCHEMA,
            "status": "failed",
            "failed_at": _utc_now(),
            "attempt_sha256": attempt.sha256,
            "campaign_sha256": attempt.campaign_sha256,
            "condition": attempt.condition,
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error) or type(error).__name__,
        }
        record["record_sha256"] = canonical_sha256(record)
        try:
            _publish_json_exclusive(paths.failure_record, record)
        except FileExistsError as publication_error:
            raise SAESmokeIntegrityError(
                "smoke failure record already exists"
            ) from publication_error

    @staticmethod
    def _publish_success(
        paths: SmokeAttemptPaths,
        attempt: SmokeAttemptIdentity,
        checkpoint: FinalizedCheckpoint,
    ) -> SmokeOutcome:
        deterministic: dict[str, Any] = {
            "status": "passed",
            "attempt_sha256": attempt.sha256,
            "campaign_sha256": attempt.campaign_sha256,
            "condition": attempt.condition,
            "condition_shape_sha256": attempt.condition_shape.sha256,
            "checkpoint_uri": checkpoint.checkpoint_uri,
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "state_sha256": checkpoint.state_sha256,
            "step": checkpoint.step,
            "tokens_seen": checkpoint.tokens_seen,
            "optimizer_updates": checkpoint.optimizer_updates,
        }
        record: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA,
            **deterministic,
            "completed_at": _utc_now(),
            "outcome_sha256": canonical_sha256(deterministic),
        }
        record["record_sha256"] = canonical_sha256(record)
        try:
            _publish_json_exclusive(paths.result_record, record)
        except FileExistsError as error:
            raise SAESmokeIntegrityError(
                "smoke result record already exists"
            ) from error
        return SmokeOutcome(
            attempt=attempt,
            paths=paths,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            state_sha256=checkpoint.state_sha256,
            outcome_sha256=record["outcome_sha256"],
            record_sha256=record["record_sha256"],
        )

    def run_attempt(
        self,
        condition: SmokeCondition,
        *,
        attempt_id: str,
        callbacks: SmokeNumericalCallbacks,
    ) -> SmokeOutcome:
        """Run, save, externally inspect, restore, and seal one attempt.

        The campaign lock is held across callbacks so two coordinators cannot
        concurrently pass the same condition.  Multi-host numerical processes
        should therefore be launched/synchronized *inside* the callbacks, not
        by calling this coordinator once per JAX process.  Numerical callbacks
        must use ``attempt.cache_path`` for ephemeral input/state caching; this
        helper never creates or populates that host-local directory.
        """

        selected = _require_condition(condition)
        if not isinstance(callbacks, SmokeNumericalCallbacks):
            raise TypeError("callbacks must be SmokeNumericalCallbacks")
        lock_path = self.campaign_root / ".coordinator.lock"
        with _exclusive_campaign_lock(lock_path):
            attempt, paths = self._reserve_attempt(selected, attempt_id)
            stage = "sequence-validation"
            try:
                self._validate_sequence(selected)
                stage = "optimizer-update"
                update = callbacks.update_once(attempt)
                expectation = self._validate_update(update, attempt)
                stage = "orbax-save"
                callbacks.save_orbax(update.state, expectation)
                stage = "external-checkpoint-inspection"
                checkpoint = callbacks.inspect_finalized_checkpoint(
                    expectation.checkpoint_uri
                )
                self._validate_finalized_checkpoint(checkpoint, expectation)
                stage = "orbax-restore"
                restored = callbacks.restore_orbax(
                    expectation.checkpoint_uri, expectation
                )
                self._validate_restore(restored, checkpoint, expectation)
                stage = "result-publication"
                return self._publish_success(paths, attempt, checkpoint)
            except Exception as error:
                self._publish_failure(
                    paths,
                    attempt,
                    stage=stage,
                    error=error,
                )
                raise SAESmokeAttemptFailed(
                    f"smoke {selected} attempt {attempt_id} failed during {stage}: "
                    f"{error}",
                    attempt=attempt,
                    failure_path=paths.failure_record,
                ) from error
