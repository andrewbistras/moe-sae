"""Pure-Python preemption evidence and the executor's one-shot drain gate.

The metadata server is a trust boundary.  A normal observation exists only
when the response is exactly HTTP 200, carries ``Metadata-Flavor: Google``,
and has the byte body ``TRUE`` or ``FALSE``.  Anything else is ambiguous and
closes the step gate without attempting an emergency checkpoint.

``SlicePreemptionGuard`` is deliberately not a poller or a shutdown daemon.
The caller owns polling and must call :meth:`authorize_new_step` immediately
before every accelerator step.  Both operations linearize through one lock,
so a step authorized before a TRUE observation may finish, while no step can
be authorized after it.  This gate is per-process: a production incident sink
must propagate the immutable decision to the other 15 TPU VMs and make their
local gates drain.  That slice-wide fanout is intentionally not implemented
here.

The first terminal condition is immutable.  Ordinary polling never retries a
failed publication.  The supervisor may explicitly call
:meth:`retry_incident_publication`, which republishes the exact same evidence
without reading the clock or reopening the gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading
from typing import Any, Callable, ClassVar, Mapping, Self, TypeAlias

from .protocol import (
    AttemptLeaseV1,
    CanonicalRecord,
    ProtocolError,
    _IDENTIFIER,
    _SHA256,
    _boolean,
    _integer,
    _strict_object,
    _string,
)


CONSERVATIVE_DRAIN_BUDGET_MS = 75_000
MIN_CHECKPOINT_TIMING_SAMPLES = 200
PREEMPTED_METADATA_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
)

_EMERGENCY_CHECKPOINT = "EMERGENCY_CHECKPOINT"
_PRESERVE_LAST_SEALED = "PRESERVE_LAST_SEALED_CHECKPOINT"
_FITS_REASON = "MEASURED_P99_PLUS_MARGIN_FITS_75S_BUDGET"
_DOES_NOT_FIT_REASON = "MEASURED_P99_PLUS_MARGIN_EXCEEDS_75S_BUDGET"
_FAULT_CODES = frozenset(
    {
        "CHECKPOINT_REFERENCE_INVALID",
        "CLOCK_BEFORE_ATTEMPT",
        "CLOCK_READ_FAILED",
        "CLOCK_REGRESSED",
        "CLOCK_VALUE_INVALID",
        "LEASE_EXPIRED",
        "METADATA_BODY_TYPE_AMBIGUOUS",
        "METADATA_FLAVOR_AMBIGUOUS",
        "METADATA_HTTP_STATUS_AMBIGUOUS",
        "METADATA_VALUE_AMBIGUOUS",
        "OBSERVER_ID_INVALID",
    }
)


class PreemptionError(RuntimeError):
    """Base class for executor preemption-control failures."""


class AttemptBindingMismatch(PreemptionError):
    """A caller supplied authority for a different attempt or fence."""


class PreemptionFailClosed(PreemptionError):
    """The step gate closed because a preemption input was not trustworthy."""

    def __init__(self, evidence: PreemptionFailClosedEvidenceV1) -> None:
        self.evidence = evidence
        super().__init__(
            f"preemption guard failed closed ({evidence.fault_code}); "
            f"incident={evidence.digest}"
        )


class AmbiguousMetadata(PreemptionFailClosed):
    """Metadata was not an authenticated, exact TRUE/FALSE response."""


class PreemptionClockError(PreemptionFailClosed):
    """The injected clock failed, was malformed, or moved backwards."""


class LeaseExpiredDuringDrain(PreemptionFailClosed):
    """The supplied attempt lease was expired at the sampled time."""


class NewStepsForbidden(PreemptionError):
    """A caller tried to begin a step after the drain gate closed."""

    def __init__(self, evidence: PreemptionIncident) -> None:
        self.evidence = evidence
        super().__init__(
            f"new accelerator steps are forbidden; incident={evidence.digest}"
        )


class IncidentSinkError(PreemptionError):
    """The one-shot incident sink failed after the gate was safely closed."""

    def __init__(self, evidence: PreemptionIncident, cause: BaseException) -> None:
        self.evidence = evidence
        self.cause = cause
        super().__init__(
            f"incident sink failed after drain latch; incident={evidence.digest}"
        )


def _optional_sha256(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name, pattern=_SHA256)


def _optional_exact_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise ProtocolError(f"{name} must be an integer or null")
    return value


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, name, minimum=0)


def _validate_chain(index: int, previous: str | None) -> None:
    _integer(index, "observation_index", minimum=1)
    _optional_sha256(previous, "previous_observation_sha256")
    if index == 1 and previous is not None:
        raise ProtocolError(
            "the first observation must not name a previous observation"
        )
    if index > 1 and previous is None:
        raise ProtocolError(
            "observations after the first must name the previous observation"
        )


def _fingerprint_bytes(value: object) -> tuple[str | None, int | None]:
    if type(value) is not bytes:
        return None, None
    return hashlib.sha256(value).hexdigest(), len(value)


def _fingerprint_text(value: object) -> tuple[str | None, int | None]:
    if type(value) is not str:
        return None, None
    encoded = value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


@dataclass(frozen=True, slots=True)
class AttemptBindingV1(CanonicalRecord):
    """The stable portion of one lease, unchanged by valid lease renewal."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.preemption-attempt-binding.v1"

    campaign_id: str
    work_id: str
    attempt_id: str
    fence: int
    executor_id: str
    deployment_identity_sha256: str
    acquired_at_unix_ms: int

    def __post_init__(self) -> None:
        _string(self.campaign_id, "campaign_id", pattern=_IDENTIFIER)
        _string(self.work_id, "work_id", pattern=_IDENTIFIER)
        _string(self.attempt_id, "attempt_id", pattern=_IDENTIFIER)
        _integer(self.fence, "fence", minimum=1)
        _string(self.executor_id, "executor_id", pattern=_IDENTIFIER)
        _string(
            self.deployment_identity_sha256,
            "deployment_identity_sha256",
            pattern=_SHA256,
        )
        _integer(self.acquired_at_unix_ms, "acquired_at_unix_ms", minimum=0)

    @classmethod
    def from_lease(cls, lease: AttemptLeaseV1) -> Self:
        if not isinstance(lease, AttemptLeaseV1):
            raise ProtocolError("lease must be AttemptLeaseV1")
        return cls(
            campaign_id=lease.campaign_id,
            work_id=lease.work_id,
            attempt_id=lease.attempt_id,
            fence=lease.fence,
            executor_id=lease.executor_id,
            deployment_identity_sha256=lease.deployment_identity_sha256,
            acquired_at_unix_ms=lease.acquired_at_unix_ms,
        )

    def matches(self, lease: AttemptLeaseV1) -> bool:
        return isinstance(lease, AttemptLeaseV1) and self == type(self).from_lease(
            lease
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "campaign_id": self.campaign_id,
            "work_id": self.work_id,
            "attempt_id": self.attempt_id,
            "fence": self.fence,
            "executor_id": self.executor_id,
            "deployment_identity_sha256": self.deployment_identity_sha256,
            "acquired_at_unix_ms": self.acquired_at_unix_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _strict_object(
            value,
            schema=cls.SCHEMA,
            fields=(
                "campaign_id",
                "work_id",
                "attempt_id",
                "fence",
                "executor_id",
                "deployment_identity_sha256",
                "acquired_at_unix_ms",
            ),
        )
        return cls(
            campaign_id=_string(obj["campaign_id"], "campaign_id"),
            work_id=_string(obj["work_id"], "work_id"),
            attempt_id=_string(obj["attempt_id"], "attempt_id"),
            fence=_integer(obj["fence"], "fence", minimum=1),
            executor_id=_string(obj["executor_id"], "executor_id"),
            deployment_identity_sha256=_string(
                obj["deployment_identity_sha256"],
                "deployment_identity_sha256",
                pattern=_SHA256,
            ),
            acquired_at_unix_ms=_integer(
                obj["acquired_at_unix_ms"], "acquired_at_unix_ms", minimum=0
            ),
        )


@dataclass(frozen=True, slots=True)
class CheckpointTimingEvidenceV1(CanonicalRecord):
    """Independent measured timing used for the emergency-save decision."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.checkpoint-drain-timing.v1"

    measurement_evidence_sha256: str
    measurement_sample_count: int
    full_save_finalize_upload_cas_p99_ms: int
    explicit_safety_margin_ms: int

    def __post_init__(self) -> None:
        _string(
            self.measurement_evidence_sha256,
            "measurement_evidence_sha256",
            pattern=_SHA256,
        )
        _integer(
            self.measurement_sample_count,
            "measurement_sample_count",
            minimum=MIN_CHECKPOINT_TIMING_SAMPLES,
        )
        _integer(
            self.full_save_finalize_upload_cas_p99_ms,
            "full_save_finalize_upload_cas_p99_ms",
            minimum=1,
        )
        _integer(
            self.explicit_safety_margin_ms,
            "explicit_safety_margin_ms",
            minimum=1,
        )

    @property
    def required_drain_ms(self) -> int:
        return (
            self.full_save_finalize_upload_cas_p99_ms + self.explicit_safety_margin_ms
        )

    @property
    def fits_conservative_budget(self) -> bool:
        return self.required_drain_ms <= CONSERVATIVE_DRAIN_BUDGET_MS

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "measurement_evidence_sha256": self.measurement_evidence_sha256,
            "measurement_sample_count": self.measurement_sample_count,
            "full_save_finalize_upload_cas_p99_ms": (
                self.full_save_finalize_upload_cas_p99_ms
            ),
            "explicit_safety_margin_ms": self.explicit_safety_margin_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _strict_object(
            value,
            schema=cls.SCHEMA,
            fields=(
                "measurement_evidence_sha256",
                "measurement_sample_count",
                "full_save_finalize_upload_cas_p99_ms",
                "explicit_safety_margin_ms",
            ),
        )
        return cls(
            measurement_evidence_sha256=_string(
                obj["measurement_evidence_sha256"],
                "measurement_evidence_sha256",
                pattern=_SHA256,
            ),
            measurement_sample_count=_integer(
                obj["measurement_sample_count"],
                "measurement_sample_count",
                minimum=MIN_CHECKPOINT_TIMING_SAMPLES,
            ),
            full_save_finalize_upload_cas_p99_ms=_integer(
                obj["full_save_finalize_upload_cas_p99_ms"],
                "full_save_finalize_upload_cas_p99_ms",
                minimum=1,
            ),
            explicit_safety_margin_ms=_integer(
                obj["explicit_safety_margin_ms"],
                "explicit_safety_margin_ms",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class MetadataPreemptionObservationV1(CanonicalRecord):
    """One authenticated, unambiguous metadata response."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.metadata-preemption-observation.v1"

    attempt: AttemptBindingV1
    observer_id: str
    observation_index: int
    previous_observation_sha256: str | None
    observed_at_unix_ms: int
    lease_expires_at_unix_ms: int
    metadata_url: str
    http_status: int
    metadata_flavor: str
    raw_value: str
    preempted: bool

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, AttemptBindingV1):
            raise ProtocolError("attempt must be AttemptBindingV1")
        _string(self.observer_id, "observer_id", pattern=_IDENTIFIER)
        _validate_chain(self.observation_index, self.previous_observation_sha256)
        _integer(self.observed_at_unix_ms, "observed_at_unix_ms", minimum=0)
        _integer(
            self.lease_expires_at_unix_ms,
            "lease_expires_at_unix_ms",
            minimum=1,
        )
        if self.observed_at_unix_ms < self.attempt.acquired_at_unix_ms:
            raise ProtocolError("observation cannot precede attempt acquisition")
        if self.observed_at_unix_ms >= self.lease_expires_at_unix_ms:
            raise ProtocolError("observation requires an unexpired lease")
        if self.metadata_url != PREEMPTED_METADATA_URL:
            raise ProtocolError("metadata_url is not the preemption metadata URL")
        if _integer(self.http_status, "http_status") != 200:
            raise ProtocolError("http_status must be exactly 200")
        if self.metadata_flavor != "Google":
            raise ProtocolError("metadata_flavor must be exactly 'Google'")
        if self.raw_value not in {"FALSE", "TRUE"}:
            raise ProtocolError("raw_value must be exactly TRUE or FALSE")
        _boolean(self.preempted, "preempted")
        if self.preempted != (self.raw_value == "TRUE"):
            raise ProtocolError("preempted does not agree with raw_value")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "attempt": self.attempt.as_dict(),
            "observer_id": self.observer_id,
            "observation_index": self.observation_index,
            "previous_observation_sha256": self.previous_observation_sha256,
            "observed_at_unix_ms": self.observed_at_unix_ms,
            "lease_expires_at_unix_ms": self.lease_expires_at_unix_ms,
            "metadata_url": self.metadata_url,
            "http_status": self.http_status,
            "metadata_flavor": self.metadata_flavor,
            "raw_value": self.raw_value,
            "preempted": self.preempted,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _strict_object(
            value,
            schema=cls.SCHEMA,
            fields=(
                "attempt",
                "observer_id",
                "observation_index",
                "previous_observation_sha256",
                "observed_at_unix_ms",
                "lease_expires_at_unix_ms",
                "metadata_url",
                "http_status",
                "metadata_flavor",
                "raw_value",
                "preempted",
            ),
        )
        return cls(
            attempt=AttemptBindingV1.from_dict(obj["attempt"]),
            observer_id=_string(obj["observer_id"], "observer_id"),
            observation_index=_integer(
                obj["observation_index"], "observation_index", minimum=1
            ),
            previous_observation_sha256=_optional_sha256(
                obj["previous_observation_sha256"],
                "previous_observation_sha256",
            ),
            observed_at_unix_ms=_integer(
                obj["observed_at_unix_ms"], "observed_at_unix_ms", minimum=0
            ),
            lease_expires_at_unix_ms=_integer(
                obj["lease_expires_at_unix_ms"],
                "lease_expires_at_unix_ms",
                minimum=1,
            ),
            metadata_url=_string(obj["metadata_url"], "metadata_url"),
            http_status=_integer(obj["http_status"], "http_status"),
            metadata_flavor=_string(obj["metadata_flavor"], "metadata_flavor"),
            raw_value=_string(obj["raw_value"], "raw_value"),
            preempted=_boolean(obj["preempted"], "preempted"),
        )


@dataclass(frozen=True, slots=True)
class SliceDrainDecisionV1(CanonicalRecord):
    """The unique drain action selected by the first TRUE observation."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.slice-drain-decision.v1"

    attempt: AttemptBindingV1
    trigger: MetadataPreemptionObservationV1
    checkpoint_timing: CheckpointTimingEvidenceV1
    decision_at_unix_ms: int
    drain_budget_ms: int
    estimated_checkpoint_completion_ms: int
    action: str
    reason: str
    emergency_checkpoint_deadline_unix_ms: int | None
    last_sealed_checkpoint_manifest_sha256: str | None
    new_steps_forbidden: bool

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, AttemptBindingV1):
            raise ProtocolError("attempt must be AttemptBindingV1")
        if not isinstance(self.trigger, MetadataPreemptionObservationV1):
            raise ProtocolError("trigger must be MetadataPreemptionObservationV1")
        if self.trigger.attempt != self.attempt:
            raise ProtocolError("trigger belongs to a different attempt")
        if not self.trigger.preempted:
            raise ProtocolError("a drain decision requires a TRUE observation")
        if not isinstance(self.checkpoint_timing, CheckpointTimingEvidenceV1):
            raise ProtocolError("checkpoint_timing must be CheckpointTimingEvidenceV1")
        _integer(self.decision_at_unix_ms, "decision_at_unix_ms", minimum=0)
        if self.decision_at_unix_ms != self.trigger.observed_at_unix_ms:
            raise ProtocolError("decision time must equal the trigger observation time")
        if self.drain_budget_ms != CONSERVATIVE_DRAIN_BUDGET_MS:
            raise ProtocolError("drain_budget_ms must be the fixed 75 second budget")
        if (
            self.estimated_checkpoint_completion_ms
            != self.checkpoint_timing.required_drain_ms
        ):
            raise ProtocolError(
                "estimated checkpoint completion must equal p99 plus margin"
            )
        expected_action = (
            _EMERGENCY_CHECKPOINT
            if self.checkpoint_timing.fits_conservative_budget
            else _PRESERVE_LAST_SEALED
        )
        expected_reason = (
            _FITS_REASON
            if self.checkpoint_timing.fits_conservative_budget
            else _DOES_NOT_FIT_REASON
        )
        if self.action != expected_action or self.reason != expected_reason:
            raise ProtocolError("drain action does not follow the fixed timing rule")
        deadline = _optional_nonnegative_int(
            self.emergency_checkpoint_deadline_unix_ms,
            "emergency_checkpoint_deadline_unix_ms",
        )
        expected_deadline = (
            self.decision_at_unix_ms + self.drain_budget_ms
            if expected_action == _EMERGENCY_CHECKPOINT
            else None
        )
        if deadline != expected_deadline:
            raise ProtocolError("emergency checkpoint deadline is inconsistent")
        _optional_sha256(
            self.last_sealed_checkpoint_manifest_sha256,
            "last_sealed_checkpoint_manifest_sha256",
        )
        if _boolean(self.new_steps_forbidden, "new_steps_forbidden") is not True:
            raise ProtocolError("new_steps_forbidden must be true")

    @classmethod
    def build(
        cls,
        *,
        trigger: MetadataPreemptionObservationV1,
        checkpoint_timing: CheckpointTimingEvidenceV1,
        last_sealed_checkpoint_manifest_sha256: str | None,
    ) -> Self:
        if not isinstance(trigger, MetadataPreemptionObservationV1):
            raise ProtocolError("trigger must be MetadataPreemptionObservationV1")
        if not isinstance(checkpoint_timing, CheckpointTimingEvidenceV1):
            raise ProtocolError("checkpoint_timing must be CheckpointTimingEvidenceV1")
        fits = checkpoint_timing.fits_conservative_budget
        return cls(
            attempt=trigger.attempt,
            trigger=trigger,
            checkpoint_timing=checkpoint_timing,
            decision_at_unix_ms=trigger.observed_at_unix_ms,
            drain_budget_ms=CONSERVATIVE_DRAIN_BUDGET_MS,
            estimated_checkpoint_completion_ms=checkpoint_timing.required_drain_ms,
            action=_EMERGENCY_CHECKPOINT if fits else _PRESERVE_LAST_SEALED,
            reason=_FITS_REASON if fits else _DOES_NOT_FIT_REASON,
            emergency_checkpoint_deadline_unix_ms=(
                trigger.observed_at_unix_ms + CONSERVATIVE_DRAIN_BUDGET_MS
                if fits
                else None
            ),
            last_sealed_checkpoint_manifest_sha256=(
                last_sealed_checkpoint_manifest_sha256
            ),
            new_steps_forbidden=True,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "attempt": self.attempt.as_dict(),
            "trigger": self.trigger.as_dict(),
            "checkpoint_timing": self.checkpoint_timing.as_dict(),
            "decision_at_unix_ms": self.decision_at_unix_ms,
            "drain_budget_ms": self.drain_budget_ms,
            "estimated_checkpoint_completion_ms": (
                self.estimated_checkpoint_completion_ms
            ),
            "action": self.action,
            "reason": self.reason,
            "emergency_checkpoint_deadline_unix_ms": (
                self.emergency_checkpoint_deadline_unix_ms
            ),
            "last_sealed_checkpoint_manifest_sha256": (
                self.last_sealed_checkpoint_manifest_sha256
            ),
            "new_steps_forbidden": self.new_steps_forbidden,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _strict_object(
            value,
            schema=cls.SCHEMA,
            fields=(
                "attempt",
                "trigger",
                "checkpoint_timing",
                "decision_at_unix_ms",
                "drain_budget_ms",
                "estimated_checkpoint_completion_ms",
                "action",
                "reason",
                "emergency_checkpoint_deadline_unix_ms",
                "last_sealed_checkpoint_manifest_sha256",
                "new_steps_forbidden",
            ),
        )
        return cls(
            attempt=AttemptBindingV1.from_dict(obj["attempt"]),
            trigger=MetadataPreemptionObservationV1.from_dict(obj["trigger"]),
            checkpoint_timing=CheckpointTimingEvidenceV1.from_dict(
                obj["checkpoint_timing"]
            ),
            decision_at_unix_ms=_integer(
                obj["decision_at_unix_ms"], "decision_at_unix_ms", minimum=0
            ),
            drain_budget_ms=_integer(
                obj["drain_budget_ms"], "drain_budget_ms", minimum=1
            ),
            estimated_checkpoint_completion_ms=_integer(
                obj["estimated_checkpoint_completion_ms"],
                "estimated_checkpoint_completion_ms",
                minimum=1,
            ),
            action=_string(obj["action"], "action"),
            reason=_string(obj["reason"], "reason"),
            emergency_checkpoint_deadline_unix_ms=_optional_nonnegative_int(
                obj["emergency_checkpoint_deadline_unix_ms"],
                "emergency_checkpoint_deadline_unix_ms",
            ),
            last_sealed_checkpoint_manifest_sha256=_optional_sha256(
                obj["last_sealed_checkpoint_manifest_sha256"],
                "last_sealed_checkpoint_manifest_sha256",
            ),
            new_steps_forbidden=_boolean(
                obj["new_steps_forbidden"], "new_steps_forbidden"
            ),
        )


@dataclass(frozen=True, slots=True)
class PreemptionFailClosedEvidenceV1(CanonicalRecord):
    """Canonical evidence for a conservative drain caused by uncertain input."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.preemption-fail-closed.v1"

    attempt: AttemptBindingV1
    fault_code: str
    trusted_at_or_before_unix_ms: int | None
    rejected_clock_sample_unix_ms: int | None
    lease_expires_at_unix_ms: int | None
    observer_id: str | None
    observation_index: int | None
    previous_observation_sha256: str | None
    metadata_http_status: int | None
    metadata_flavor_sha256: str | None
    metadata_flavor_size_bytes: int | None
    metadata_body_sha256: str | None
    metadata_body_size_bytes: int | None
    checkpoint_timing: CheckpointTimingEvidenceV1
    last_sealed_checkpoint_manifest_sha256: str | None
    drain_budget_ms: int
    action: str
    new_steps_forbidden: bool

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, AttemptBindingV1):
            raise ProtocolError("attempt must be AttemptBindingV1")
        if self.fault_code not in _FAULT_CODES:
            raise ProtocolError("fault_code is not supported")
        trusted = _optional_nonnegative_int(
            self.trusted_at_or_before_unix_ms,
            "trusted_at_or_before_unix_ms",
        )
        rejected = _optional_exact_int(
            self.rejected_clock_sample_unix_ms,
            "rejected_clock_sample_unix_ms",
        )
        expiry = _optional_nonnegative_int(
            self.lease_expires_at_unix_ms,
            "lease_expires_at_unix_ms",
        )
        if self.observer_id is not None:
            _string(self.observer_id, "observer_id", pattern=_IDENTIFIER)
        index = _optional_nonnegative_int(self.observation_index, "observation_index")
        if index == 0:
            raise ProtocolError("observation_index must be positive or null")
        previous = _optional_sha256(
            self.previous_observation_sha256,
            "previous_observation_sha256",
        )
        if index is None and previous is not None:
            raise ProtocolError("previous observation requires an observation index")
        if index is not None:
            _validate_chain(index, previous)
        _optional_exact_int(self.metadata_http_status, "metadata_http_status")
        for prefix, digest, size in (
            (
                "metadata_flavor",
                self.metadata_flavor_sha256,
                self.metadata_flavor_size_bytes,
            ),
            (
                "metadata_body",
                self.metadata_body_sha256,
                self.metadata_body_size_bytes,
            ),
        ):
            if (digest is None) != (size is None):
                raise ProtocolError(f"{prefix} digest and size must appear together")
            if digest is not None:
                _string(digest, f"{prefix}_sha256", pattern=_SHA256)
                _integer(size, f"{prefix}_size_bytes", minimum=0)
        if not isinstance(self.checkpoint_timing, CheckpointTimingEvidenceV1):
            raise ProtocolError("checkpoint_timing must be CheckpointTimingEvidenceV1")
        _optional_sha256(
            self.last_sealed_checkpoint_manifest_sha256,
            "last_sealed_checkpoint_manifest_sha256",
        )
        if self.drain_budget_ms != CONSERVATIVE_DRAIN_BUDGET_MS:
            raise ProtocolError("drain_budget_ms must be the fixed 75 second budget")
        if self.action != _PRESERVE_LAST_SEALED:
            raise ProtocolError("fail-closed action must preserve the last seal")
        if _boolean(self.new_steps_forbidden, "new_steps_forbidden") is not True:
            raise ProtocolError("new_steps_forbidden must be true")
        if self.fault_code == "CLOCK_REGRESSED":
            if trusted is None or rejected is None or rejected >= trusted:
                raise ProtocolError("CLOCK_REGRESSED evidence is inconsistent")
        if self.fault_code == "CLOCK_BEFORE_ATTEMPT":
            if rejected is None or rejected >= self.attempt.acquired_at_unix_ms:
                raise ProtocolError("CLOCK_BEFORE_ATTEMPT evidence is inconsistent")
        if self.fault_code == "LEASE_EXPIRED":
            if trusted is None or expiry is None or trusted < expiry:
                raise ProtocolError("LEASE_EXPIRED evidence is inconsistent")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "attempt": self.attempt.as_dict(),
            "fault_code": self.fault_code,
            "trusted_at_or_before_unix_ms": self.trusted_at_or_before_unix_ms,
            "rejected_clock_sample_unix_ms": self.rejected_clock_sample_unix_ms,
            "lease_expires_at_unix_ms": self.lease_expires_at_unix_ms,
            "observer_id": self.observer_id,
            "observation_index": self.observation_index,
            "previous_observation_sha256": self.previous_observation_sha256,
            "metadata_http_status": self.metadata_http_status,
            "metadata_flavor_sha256": self.metadata_flavor_sha256,
            "metadata_flavor_size_bytes": self.metadata_flavor_size_bytes,
            "metadata_body_sha256": self.metadata_body_sha256,
            "metadata_body_size_bytes": self.metadata_body_size_bytes,
            "checkpoint_timing": self.checkpoint_timing.as_dict(),
            "last_sealed_checkpoint_manifest_sha256": (
                self.last_sealed_checkpoint_manifest_sha256
            ),
            "drain_budget_ms": self.drain_budget_ms,
            "action": self.action,
            "new_steps_forbidden": self.new_steps_forbidden,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _strict_object(
            value,
            schema=cls.SCHEMA,
            fields=(
                "attempt",
                "fault_code",
                "trusted_at_or_before_unix_ms",
                "rejected_clock_sample_unix_ms",
                "lease_expires_at_unix_ms",
                "observer_id",
                "observation_index",
                "previous_observation_sha256",
                "metadata_http_status",
                "metadata_flavor_sha256",
                "metadata_flavor_size_bytes",
                "metadata_body_sha256",
                "metadata_body_size_bytes",
                "checkpoint_timing",
                "last_sealed_checkpoint_manifest_sha256",
                "drain_budget_ms",
                "action",
                "new_steps_forbidden",
            ),
        )
        observer = obj["observer_id"]
        if observer is not None:
            observer = _string(observer, "observer_id")
        return cls(
            attempt=AttemptBindingV1.from_dict(obj["attempt"]),
            fault_code=_string(obj["fault_code"], "fault_code"),
            trusted_at_or_before_unix_ms=_optional_nonnegative_int(
                obj["trusted_at_or_before_unix_ms"],
                "trusted_at_or_before_unix_ms",
            ),
            rejected_clock_sample_unix_ms=_optional_exact_int(
                obj["rejected_clock_sample_unix_ms"],
                "rejected_clock_sample_unix_ms",
            ),
            lease_expires_at_unix_ms=_optional_nonnegative_int(
                obj["lease_expires_at_unix_ms"],
                "lease_expires_at_unix_ms",
            ),
            observer_id=observer,
            observation_index=_optional_nonnegative_int(
                obj["observation_index"], "observation_index"
            ),
            previous_observation_sha256=_optional_sha256(
                obj["previous_observation_sha256"],
                "previous_observation_sha256",
            ),
            metadata_http_status=_optional_exact_int(
                obj["metadata_http_status"], "metadata_http_status"
            ),
            metadata_flavor_sha256=_optional_sha256(
                obj["metadata_flavor_sha256"], "metadata_flavor_sha256"
            ),
            metadata_flavor_size_bytes=_optional_nonnegative_int(
                obj["metadata_flavor_size_bytes"],
                "metadata_flavor_size_bytes",
            ),
            metadata_body_sha256=_optional_sha256(
                obj["metadata_body_sha256"], "metadata_body_sha256"
            ),
            metadata_body_size_bytes=_optional_nonnegative_int(
                obj["metadata_body_size_bytes"], "metadata_body_size_bytes"
            ),
            checkpoint_timing=CheckpointTimingEvidenceV1.from_dict(
                obj["checkpoint_timing"]
            ),
            last_sealed_checkpoint_manifest_sha256=_optional_sha256(
                obj["last_sealed_checkpoint_manifest_sha256"],
                "last_sealed_checkpoint_manifest_sha256",
            ),
            drain_budget_ms=_integer(
                obj["drain_budget_ms"], "drain_budget_ms", minimum=1
            ),
            action=_string(obj["action"], "action"),
            new_steps_forbidden=_boolean(
                obj["new_steps_forbidden"], "new_steps_forbidden"
            ),
        )


PreemptionIncident: TypeAlias = SliceDrainDecisionV1 | PreemptionFailClosedEvidenceV1
ClockMS: TypeAlias = Callable[[], int]
IncidentSink: TypeAlias = Callable[[PreemptionIncident], object]


class SlicePreemptionGuard:
    """Per-process linearization point for steps and preemption notices.

    The injected production sink is responsible for slice-wide propagation;
    this class does not fan the decision out across the 16 TPU VMs.
    """

    def __init__(
        self,
        lease: AttemptLeaseV1,
        *,
        checkpoint_timing: CheckpointTimingEvidenceV1,
        clock_ms: ClockMS,
        incident_sink: IncidentSink,
    ) -> None:
        self._attempt = AttemptBindingV1.from_lease(lease)
        if not isinstance(checkpoint_timing, CheckpointTimingEvidenceV1):
            raise ProtocolError("checkpoint_timing must be CheckpointTimingEvidenceV1")
        if not callable(clock_ms):
            raise TypeError("clock_ms must be callable")
        if not callable(incident_sink):
            raise TypeError("incident_sink must be callable")
        self._checkpoint_timing = checkpoint_timing
        self._clock_ms = clock_ms
        self._incident_sink = incident_sink
        self._lock = threading.RLock()
        self._last_clock_ms: int | None = None
        self._last_observation: MetadataPreemptionObservationV1 | None = None
        self._incident: PreemptionIncident | None = None
        self._sink_invoked = False
        self._sink_succeeded = False
        self._sink_error: BaseException | None = None

    @property
    def attempt(self) -> AttemptBindingV1:
        return self._attempt

    @property
    def new_steps_allowed(self) -> bool:
        with self._lock:
            return self._incident is None

    @property
    def last_observation(self) -> MetadataPreemptionObservationV1 | None:
        with self._lock:
            return self._last_observation

    @property
    def incident(self) -> PreemptionIncident | None:
        with self._lock:
            return self._incident

    @property
    def incident_published(self) -> bool:
        with self._lock:
            return self._sink_succeeded

    def authorize_new_step(
        self,
        lease: AttemptLeaseV1,
        *,
        last_sealed_checkpoint_manifest_sha256: str | None,
    ) -> int:
        """Linearize the start of one new step and return its trusted time."""

        with self._lock:
            self._require_binding(lease)
            if self._incident is not None:
                self._raise_if_sink_failed_locked()
                raise NewStepsForbidden(self._incident)
            checkpoint = self._checkpoint_or_fail_locked(
                last_sealed_checkpoint_manifest_sha256
            )
            now = self._sample_clock_locked(checkpoint)
            self._require_unexpired_locked(lease, now, checkpoint)
            return now

    def observe_metadata(
        self,
        lease: AttemptLeaseV1,
        *,
        observer_id: str,
        http_status: object,
        metadata_flavor: object,
        body: object,
        last_sealed_checkpoint_manifest_sha256: str | None,
    ) -> SliceDrainDecisionV1 | None:
        """Accept one raw metadata response or fail closed.

        A repeated call after a TRUE observation returns the identical decision
        without sampling the clock or invoking the sink again.  In particular,
        polling never retries a failed sink; only
        :meth:`retry_incident_publication` may do that.
        """

        with self._lock:
            self._require_binding(lease)
            if self._incident is not None:
                return self._terminal_poll_result_locked()
            checkpoint = self._checkpoint_or_fail_locked(
                last_sealed_checkpoint_manifest_sha256
            )
            try:
                checked_observer = _string(
                    observer_id, "observer_id", pattern=_IDENTIFIER
                )
            except ProtocolError:
                self._latch_fault_locked(
                    fault_code="OBSERVER_ID_INVALID",
                    checkpoint=checkpoint,
                )
                raise AssertionError("unreachable")
            now = self._sample_clock_locked(checkpoint)
            self._require_unexpired_locked(lease, now, checkpoint)
            index = (
                1
                if self._last_observation is None
                else (self._last_observation.observation_index + 1)
            )
            previous = (
                None
                if self._last_observation is None
                else self._last_observation.digest
            )
            flavor_sha, flavor_size = _fingerprint_text(metadata_flavor)
            body_sha, body_size = _fingerprint_bytes(body)
            status_evidence = http_status if type(http_status) is int else None
            common_fault = {
                "trusted_at_or_before_unix_ms": now,
                "lease_expires_at_unix_ms": lease.expires_at_unix_ms,
                "observer_id": checked_observer,
                "observation_index": index,
                "previous_observation_sha256": previous,
                "metadata_http_status": status_evidence,
                "metadata_flavor_sha256": flavor_sha,
                "metadata_flavor_size_bytes": flavor_size,
                "metadata_body_sha256": body_sha,
                "metadata_body_size_bytes": body_size,
            }
            if type(http_status) is not int or http_status != 200:
                self._latch_fault_locked(
                    fault_code="METADATA_HTTP_STATUS_AMBIGUOUS",
                    checkpoint=checkpoint,
                    **common_fault,
                )
                raise AssertionError("unreachable")
            if type(metadata_flavor) is not str or metadata_flavor != "Google":
                self._latch_fault_locked(
                    fault_code="METADATA_FLAVOR_AMBIGUOUS",
                    checkpoint=checkpoint,
                    **common_fault,
                )
                raise AssertionError("unreachable")
            if type(body) is not bytes:
                self._latch_fault_locked(
                    fault_code="METADATA_BODY_TYPE_AMBIGUOUS",
                    checkpoint=checkpoint,
                    **common_fault,
                )
                raise AssertionError("unreachable")
            if body not in {b"FALSE", b"TRUE"}:
                self._latch_fault_locked(
                    fault_code="METADATA_VALUE_AMBIGUOUS",
                    checkpoint=checkpoint,
                    **common_fault,
                )
                raise AssertionError("unreachable")
            value = body.decode("ascii")
            observation = MetadataPreemptionObservationV1(
                attempt=self._attempt,
                observer_id=checked_observer,
                observation_index=index,
                previous_observation_sha256=previous,
                observed_at_unix_ms=now,
                lease_expires_at_unix_ms=lease.expires_at_unix_ms,
                metadata_url=PREEMPTED_METADATA_URL,
                http_status=http_status,
                metadata_flavor=metadata_flavor,
                raw_value=value,
                preempted=value == "TRUE",
            )
            self._last_observation = observation
            if not observation.preempted:
                return None
            decision = SliceDrainDecisionV1.build(
                trigger=observation,
                checkpoint_timing=self._checkpoint_timing,
                last_sealed_checkpoint_manifest_sha256=checkpoint,
            )
            self._publish_incident_locked(decision)
            self._raise_if_sink_failed_locked()
            return decision

    def retry_incident_publication(self) -> bool:
        """Explicitly retry a failed sink with the exact terminal evidence.

        Returns ``True`` when this call publishes a previously unpublished
        incident and ``False`` when publication had already succeeded.  It is
        an error to call this before a terminal incident exists.  This method
        samples no clock, never changes the incident, and never reopens the
        step gate.
        """

        with self._lock:
            if self._incident is None:
                raise PreemptionError(
                    "cannot retry incident publication before a terminal incident"
                )
            if self._sink_succeeded:
                return False
            incident = self._incident
            self._invoke_incident_sink_locked(incident)
            if self._incident is not incident:
                raise AssertionError("terminal preemption evidence changed")
            self._raise_if_sink_failed_locked()
            return True

    def _require_binding(self, lease: AttemptLeaseV1) -> None:
        if not self._attempt.matches(lease):
            raise AttemptBindingMismatch(
                "lease does not match the guard's campaign/work/attempt/fence binding"
            )

    def _checkpoint_or_fail_locked(self, value: object) -> str | None:
        if value is None:
            return None
        if type(value) is str and _SHA256.fullmatch(value) is not None:
            return value
        self._latch_fault_locked(
            fault_code="CHECKPOINT_REFERENCE_INVALID",
            checkpoint=None,
        )
        raise AssertionError("unreachable")

    def _sample_clock_locked(self, checkpoint: str | None) -> int:
        try:
            sample = self._clock_ms()
        except Exception:
            self._latch_fault_locked(
                fault_code="CLOCK_READ_FAILED",
                checkpoint=checkpoint,
                trusted_at_or_before_unix_ms=self._last_clock_ms,
            )
            raise AssertionError("unreachable")
        if type(sample) is not int or sample < 0:
            self._latch_fault_locked(
                fault_code="CLOCK_VALUE_INVALID",
                checkpoint=checkpoint,
                trusted_at_or_before_unix_ms=self._last_clock_ms,
                rejected_clock_sample_unix_ms=(sample if type(sample) is int else None),
            )
            raise AssertionError("unreachable")
        if sample < self._attempt.acquired_at_unix_ms:
            self._latch_fault_locked(
                fault_code="CLOCK_BEFORE_ATTEMPT",
                checkpoint=checkpoint,
                trusted_at_or_before_unix_ms=self._last_clock_ms,
                rejected_clock_sample_unix_ms=sample,
            )
            raise AssertionError("unreachable")
        if self._last_clock_ms is not None and sample < self._last_clock_ms:
            self._latch_fault_locked(
                fault_code="CLOCK_REGRESSED",
                checkpoint=checkpoint,
                trusted_at_or_before_unix_ms=self._last_clock_ms,
                rejected_clock_sample_unix_ms=sample,
            )
            raise AssertionError("unreachable")
        self._last_clock_ms = sample
        return sample

    def _require_unexpired_locked(
        self,
        lease: AttemptLeaseV1,
        now: int,
        checkpoint: str | None,
    ) -> None:
        if now >= lease.expires_at_unix_ms:
            self._latch_fault_locked(
                fault_code="LEASE_EXPIRED",
                checkpoint=checkpoint,
                trusted_at_or_before_unix_ms=now,
                lease_expires_at_unix_ms=lease.expires_at_unix_ms,
            )

    def _latch_fault_locked(
        self,
        *,
        fault_code: str,
        checkpoint: str | None,
        trusted_at_or_before_unix_ms: int | None = None,
        rejected_clock_sample_unix_ms: int | None = None,
        lease_expires_at_unix_ms: int | None = None,
        observer_id: str | None = None,
        observation_index: int | None = None,
        previous_observation_sha256: str | None = None,
        metadata_http_status: int | None = None,
        metadata_flavor_sha256: str | None = None,
        metadata_flavor_size_bytes: int | None = None,
        metadata_body_sha256: str | None = None,
        metadata_body_size_bytes: int | None = None,
    ) -> None:
        evidence = PreemptionFailClosedEvidenceV1(
            attempt=self._attempt,
            fault_code=fault_code,
            trusted_at_or_before_unix_ms=trusted_at_or_before_unix_ms,
            rejected_clock_sample_unix_ms=rejected_clock_sample_unix_ms,
            lease_expires_at_unix_ms=lease_expires_at_unix_ms,
            observer_id=observer_id,
            observation_index=observation_index,
            previous_observation_sha256=previous_observation_sha256,
            metadata_http_status=metadata_http_status,
            metadata_flavor_sha256=metadata_flavor_sha256,
            metadata_flavor_size_bytes=metadata_flavor_size_bytes,
            metadata_body_sha256=metadata_body_sha256,
            metadata_body_size_bytes=metadata_body_size_bytes,
            checkpoint_timing=self._checkpoint_timing,
            last_sealed_checkpoint_manifest_sha256=checkpoint,
            drain_budget_ms=CONSERVATIVE_DRAIN_BUDGET_MS,
            action=_PRESERVE_LAST_SEALED,
            new_steps_forbidden=True,
        )
        self._publish_incident_locked(evidence)
        self._raise_if_sink_failed_locked()
        self._raise_fault(evidence)

    def _publish_incident_locked(self, evidence: PreemptionIncident) -> None:
        if self._incident is not None:
            if self._incident != evidence:
                raise AssertionError("terminal preemption evidence changed")
            return
        self._incident = evidence
        self._invoke_incident_sink_locked(evidence)

    def _invoke_incident_sink_locked(self, evidence: PreemptionIncident) -> None:
        if self._incident is not evidence:
            raise AssertionError("incident sink received non-terminal evidence")
        self._sink_invoked = True
        try:
            self._incident_sink(evidence)
        except Exception as error:
            self._sink_error = error
            return
        self._sink_error = None
        self._sink_succeeded = True

    def _raise_if_sink_failed_locked(self) -> None:
        if self._sink_error is not None:
            assert self._incident is not None
            raise IncidentSinkError(
                self._incident, self._sink_error
            ) from self._sink_error

    def _terminal_poll_result_locked(self) -> SliceDrainDecisionV1:
        self._raise_if_sink_failed_locked()
        assert self._incident is not None
        if isinstance(self._incident, SliceDrainDecisionV1):
            return self._incident
        self._raise_fault(self._incident)
        raise AssertionError("unreachable")

    @staticmethod
    def _raise_fault(evidence: PreemptionFailClosedEvidenceV1) -> None:
        if evidence.fault_code.startswith("METADATA_") or evidence.fault_code == (
            "OBSERVER_ID_INVALID"
        ):
            raise AmbiguousMetadata(evidence)
        if evidence.fault_code.startswith("CLOCK_"):
            raise PreemptionClockError(evidence)
        if evidence.fault_code == "LEASE_EXPIRED":
            raise LeaseExpiredDuringDrain(evidence)
        raise PreemptionFailClosed(evidence)


__all__ = [
    "AmbiguousMetadata",
    "AttemptBindingMismatch",
    "AttemptBindingV1",
    "CONSERVATIVE_DRAIN_BUDGET_MS",
    "CheckpointTimingEvidenceV1",
    "IncidentSinkError",
    "LeaseExpiredDuringDrain",
    "MIN_CHECKPOINT_TIMING_SAMPLES",
    "MetadataPreemptionObservationV1",
    "NewStepsForbidden",
    "PREEMPTED_METADATA_URL",
    "PreemptionClockError",
    "PreemptionError",
    "PreemptionFailClosed",
    "PreemptionFailClosedEvidenceV1",
    "PreemptionIncident",
    "SliceDrainDecisionV1",
    "SlicePreemptionGuard",
]
