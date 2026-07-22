"""JAX-free durable control plane for the four-controller SAE campaign.

This module deliberately contains no numerical or cloud code.  It sequences a
fixed SAE experiment plan, proves that exactly four controller supervisors
participate in each attempt, and records only small self-hashed POSIX control
objects.  Numerical children remain responsible for validating and publishing
their scientific artifacts.

The safety model is intentionally conservative:

* every deployment/scientific identity gets its own operation directory;
* a controller holds a kernel ``flock`` for its entire supervisor lifetime;
* phase attempts are bound to the exact four ``(lane_id, lease_id)`` members;
* successful lanes must report the same externally validated result digest;
* passed phases may be skipped only after the caller supplies that digest;
* aborts are durable and boundary-latched, never process signals; and
* stale locked lanes, vanished live lanes, tampered records, and split cohorts
  fail closed before another numerical child may start.

The host-local launcher and CLI integration are intentionally outside this
module so status and abort remain safe to invoke without importing JAX.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Any, Literal


def _condition_seed_phases(prefix: str) -> tuple[str, ...]:
    return tuple(
        f"{prefix}-{condition}-seed-{seed}"
        for seed in (1, 2, 3)
        for condition in ("A", "B", "F")
    )


SAE_PHASES = (
    "validate-prerequisites",
    "smoke-F",
    "smoke-A",
    "smoke-B",
    "gate-A-seed-1",
    "gate-B-seed-1",
    "gate-F-seed-1",
    *_condition_seed_phases("full"),
    *_condition_seed_phases("eval"),
    "report",
)

SPEC_SCHEMA = "gemma4-sae-operation-spec-v2"
STATE_SCHEMA = "gemma4-sae-operator-state-v1"
LANE_SCHEMA = "gemma4-sae-operator-lane-v1"
RELEASE_SCHEMA = "gemma4-sae-operator-release-v1"
ABORT_SCHEMA = "gemma4-sae-operator-abort-v1"
STATUS_SCHEMA = "gemma4-sae-operator-status-v1"

EXPECTED_LANES = 4
LANE_FRESH_SECONDS = 60.0
HEARTBEAT_INTERVAL_SECONDS = 10.0
RENDEZVOUS_TIMEOUT_SECONDS = 300.0
LANE_STATUS_PUBLICATION_GRACE_SECONDS = 10.0
LANE_STATUS_PUBLICATION_POLL_SECONDS = 0.05

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ATTEMPT_RE = re.compile(r"[0-9a-f]{32}")
_LANE_RE = re.compile(r"[0-9a-f]{24}")
_ACTIVE_LANE_STATES = {
    "starting",
    "ready-to-begin",
    "ready",
    "running",
    "succeeded",
    "failed",
    "aborted",
}
_TERMINAL_LANE_STATES = {"succeeded", "failed", "aborted"}
_PHASE_STATUSES = {"pending", "running", "passed", "failed", "aborted"}


class SAEOperatorError(RuntimeError):
    """The SAE control plane is invalid or cannot safely advance."""


class SAETamperError(SAEOperatorError):
    """A self-hashed control object failed validation."""


class SAEOperationMismatch(SAEOperatorError):
    """An artifact root or control object belongs to another operation."""


class SAELaneLeaseHeld(SAEOperatorError):
    """Another supervisor or inherited child still owns a controller lane."""


class SAEOrphanedLane(SAEOperatorError):
    """A kernel-held controller lane no longer has a fresh supervisor."""


class SAEAbortRequested(SAEOperatorError):
    """A durable boundary abort prevents the next phase from starting."""


class SAEPhaseAlreadyPassed(SAEOperatorError):
    """A phase can be skipped only after its durable result is revalidated."""


class SAEPeerFailure(SAEOperatorError):
    """A member of the active four-controller attempt failed or disappeared."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    """Hash one JSON-native value with the control plane's canonical encoding."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("control value must be canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_sha256(
        {name: value for name, value in payload.items() if name != field}
    )


def _require_sha256(value: Any, label: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SAEOperatorError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise SAEOperatorError(f"{label} is not a JSON object: {path}")
    return value


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b", buffering=0) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lock_is_held(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b", buffering=0) as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return False


@contextmanager
def _bounded_exclusive_lock(
    path: Path, *, timeout_seconds: float
) -> Iterator[None]:
    """Acquire a short control lock without letting status inspection hang."""

    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    with path.open("a+b", buffering=0) as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise SAEOrphanedLane(
                        "SAE lane startup did not publish within "
                        f"{timeout_seconds:g}s: {path}"
                    )
                time.sleep(min(LANE_STATUS_PUBLICATION_POLL_SECONDS, remaining))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class SAEOperationSpec:
    """Immutable scientific/deployment identity for one SAE operation."""

    config_sha256: str
    repository_content_sha256: str
    consumer_environment_sha256: str
    train_capture_manifest_sha256: str
    eval_capture_manifest_sha256: str
    activation_storage_attestation_sha256: str
    sae_storage_attestation_sha256: str
    capture_compatibility_sha256: str
    smoke_campaign_sha256: str
    smoke_suite_sha256: str
    checkpoint_payload_root: str
    canary_waiver_sha256: str | None = None
    sealed_recovery_waiver_sha256: str | None = None
    performance_waiver_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "config_sha256",
            "repository_content_sha256",
            "consumer_environment_sha256",
            "train_capture_manifest_sha256",
            "eval_capture_manifest_sha256",
            "activation_storage_attestation_sha256",
            "sae_storage_attestation_sha256",
            "capture_compatibility_sha256",
            "smoke_campaign_sha256",
            "smoke_suite_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_sha256(
            self.canary_waiver_sha256, "canary_waiver_sha256", optional=True
        )
        _require_sha256(
            self.sealed_recovery_waiver_sha256,
            "sealed_recovery_waiver_sha256",
            optional=True,
        )
        _require_sha256(
            self.performance_waiver_sha256,
            "performance_waiver_sha256",
            optional=True,
        )
        if not isinstance(self.checkpoint_payload_root, str) or not (
            self.checkpoint_payload_root.strip()
        ):
            raise ValueError("checkpoint_payload_root cannot be empty")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SPEC_SCHEMA,
            **asdict(self),
            "phase_plan": list(SAE_PHASES),
            "expected_lanes": EXPECTED_LANES,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_mapping())


@dataclass(frozen=True, slots=True)
class SAELaneLease:
    descriptor: int
    lane_id: str
    lease_id: str
    hostname: str
    operation_sha256: str
    lock_path: Path
    status_path: Path


@dataclass(frozen=True, slots=True)
class SAEPhaseAttempt:
    phase: str
    attempt_id: str
    attempt_number: int
    cohort_sha256: str
    members: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class SAEAttemptOutcome:
    phase: str
    attempt_id: str
    outcome: Literal["passed", "failed", "aborted"]
    result_sha256: str | None
    completion_sha256: str


def _empty_phase() -> dict[str, Any]:
    return {
        "status": "pending",
        "attempts": 0,
        "attempt_id": None,
        "cohort_sha256": None,
        "members": None,
        "started_at": None,
        "completed_at": None,
        "result": None,
        "history": [],
    }


class SAEControl:
    """Concurrency-safe control records for one operation under an artifact root."""

    def __init__(
        self,
        artifact_root: str | Path,
        operation_sha256: str | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.base = self.artifact_root / "operator" / "sae" / "operations"
        self._operation_sha256: str | None = None
        self._heartbeat_cache: dict[
            str, tuple[tuple[Any, ...], float, dict[str, Any]]
        ] = {}
        if operation_sha256 is not None:
            self.attach(operation_sha256)

    @property
    def operation_sha256(self) -> str:
        if self._operation_sha256 is None:
            raise SAEOperatorError("SAE control is not attached to an operation")
        return self._operation_sha256

    @property
    def root(self) -> Path:
        return self.base / self.operation_sha256

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def abort_path(self) -> Path:
        return self.root / "abort.json"

    @property
    def lock_path(self) -> Path:
        return self.root / ".control.lock"

    def attach(self, operation_sha256: str) -> None:
        _require_sha256(operation_sha256, "operation_sha256")
        if (
            self._operation_sha256 is not None
            and self._operation_sha256 != operation_sha256
        ):
            raise SAEOperationMismatch("control is already attached to another operation")
        self._operation_sha256 = operation_sha256

    def begin(self, spec: SAEOperationSpec) -> dict[str, Any]:
        """Create or validate the immutable operation state."""

        self.attach(spec.sha256)
        with _exclusive_lock(self.lock_path):
            if self.state_path.exists():
                state = self._load_state_unlocked()
                if state["spec"] != spec.to_mapping():
                    raise SAEOperationMismatch(
                        "SAE operation state belongs to another immutable spec"
                    )
                return state
            now = _utc_now()
            state: dict[str, Any] = {
                "schema_version": STATE_SCHEMA,
                "operation_sha256": spec.sha256,
                "spec": spec.to_mapping(),
                "phase_plan": list(SAE_PHASES),
                "created_at": now,
                "updated_at": now,
                "phases": {phase: _empty_phase() for phase in SAE_PHASES},
                "last_abort_sha256": None,
                "abort_acknowledged_at": None,
                "abort_boundary_phase": None,
            }
            self._write_state_unlocked(state)
            return self._load_state_unlocked()

    def load_state(self) -> dict[str, Any]:
        with _exclusive_lock(self.lock_path):
            return self._load_state_unlocked()

    @contextmanager
    def lane_lease(self, lane: str | None = None) -> Iterator[SAELaneLease]:
        """Hold one supervisor lane until every inherited descriptor is closed."""

        self.load_state()
        hostname = platform.node()
        identity = str(lane or hostname).strip()
        if not identity or len(identity) > 512:
            raise ValueError("lane identity must contain 1..512 characters")
        lane_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        lock_path = self.root / "lanes" / f"lane-{lane_id}.lock"
        status_path = self.root / "lanes" / f"lane-{lane_id}.json"
        startup_path = self.root / "lanes" / ".startup" / f"lane-{lane_id}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b", buffering=0)
        try:
            # Correlate kernel ownership with the first status atomically from
            # an observer's perspective.  In particular, a restarted owner may
            # otherwise hold this reused lane lock while the JSON file still
            # names the previous lease.
            with _bounded_exclusive_lock(
                startup_path,
                timeout_seconds=LANE_STATUS_PUBLICATION_GRACE_SECONDS,
            ):
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise SAELaneLeaseHeld(
                        f"controller lane {lane_id} is held by a supervisor or orphan"
                    ) from error
                lease = SAELaneLease(
                    descriptor=handle.fileno(),
                    lane_id=lane_id,
                    lease_id=os.urandom(16).hex(),
                    hostname=hostname,
                    operation_sha256=self.operation_sha256,
                    lock_path=lock_path,
                    status_path=status_path,
                )
                self.heartbeat(lease, "starting")
            try:
                yield lease
            except BaseException as error:
                current = self._read_lane_if_valid(lease.status_path)
                preserve_terminal = False
                if (
                    current is not None
                    and current.get("state") in _TERMINAL_LANE_STATES
                    and current.get("phase") in SAE_PHASES
                    and isinstance(current.get("attempt_id"), str)
                ):
                    state = self.load_state()
                    record = state["phases"][str(current["phase"])]
                    result = record.get("result")
                    preserve_terminal = (
                        record.get("attempt_id") == current["attempt_id"]
                        and record.get("status") in {"passed", "failed", "aborted"}
                        and isinstance(result, Mapping)
                        and isinstance(
                            result.get("completion_release_sha256"), str
                        )
                    )
                if preserve_terminal:
                    assert current is not None
                    self.heartbeat(
                        lease,
                        str(current["state"]),
                        phase=str(current["phase"]),
                        attempt_id=str(current["attempt_id"]),
                        result_sha256=current.get("result_sha256"),
                        detail=current.get("detail"),
                        force=True,
                    )
                else:
                    self.heartbeat(
                        lease,
                        "failed",
                        phase=current.get("phase") if current else None,
                        attempt_id=current.get("attempt_id") if current else None,
                        detail={
                            "error_type": type(error).__name__,
                            "error": str(error)[:2_000],
                        },
                        force=True,
                    )
                raise
            else:
                current = self._read_lane_if_valid(lease.status_path)
                if current and current.get("state") in {
                    "ready-to-begin",
                    "ready",
                    "running",
                }:
                    self.heartbeat(
                        lease,
                        "failed",
                        phase=current.get("phase"),
                        attempt_id=current.get("attempt_id"),
                        detail={"error": "supervisor exited with an active phase"},
                        force=True,
                    )
                else:
                    self.heartbeat(lease, "exited", force=True)
        finally:
            handle.close()

    def heartbeat(
        self,
        lease: SAELaneLease,
        state: str,
        *,
        phase: str | None = None,
        attempt_id: str | None = None,
        result_sha256: str | None = None,
        detail: Mapping[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Publish one self-hashed lane heartbeat."""

        if lease.operation_sha256 != self.operation_sha256:
            raise SAEOperationMismatch("lane lease belongs to another operation")
        allowed_states = _ACTIVE_LANE_STATES | {"exited"}
        if state not in allowed_states:
            raise ValueError(f"invalid SAE lane state: {state!r}")
        if phase is not None and phase not in SAE_PHASES:
            raise ValueError(f"unknown SAE phase: {phase!r}")
        if attempt_id is not None and _ATTEMPT_RE.fullmatch(attempt_id) is None:
            raise ValueError("attempt_id must be 32 lowercase hex digits")
        _require_sha256(result_sha256, "result_sha256", optional=True)
        if state in {"ready", "running", "succeeded"}:
            if phase is None or attempt_id is None:
                raise ValueError(f"lane state {state!r} requires a phase and attempt")
        if state in {"failed", "aborted"} and attempt_id is not None and phase is None:
            raise ValueError(f"lane state {state!r} has an attempt without a phase")
        if state == "ready-to-begin" and (phase is None or attempt_id is not None):
            raise ValueError("ready-to-begin requires a phase and no attempt")
        if state == "succeeded" and result_sha256 is None:
            raise ValueError("a successful lane must bind a result SHA-256")
        if state != "succeeded" and result_sha256 is not None:
            raise ValueError("only a successful lane may bind a result SHA-256")
        detail_value = None if detail is None else dict(detail)
        if detail_value is not None:
            encoded_size = len(
                json.dumps(detail_value, allow_nan=False, ensure_ascii=True).encode(
                    "utf-8"
                )
            )
            if encoded_size > 16_384:
                raise ValueError("lane detail exceeds 16 KiB")
        cache_key = (
            state,
            phase,
            attempt_id,
            result_sha256,
            canonical_sha256(detail_value),
        )
        now = time.time()
        cached = self._heartbeat_cache.get(lease.lease_id)
        if (
            not force
            and cached is not None
            and cached[0] == cache_key
            and now - cached[1] < HEARTBEAT_INTERVAL_SECONDS
        ):
            return cached[2]
        payload: dict[str, Any] = {
            "schema_version": LANE_SCHEMA,
            "operation_sha256": self.operation_sha256,
            "lane_id": lease.lane_id,
            "lease_id": lease.lease_id,
            "hostname": lease.hostname,
            "pid": os.getpid(),
            "state": state,
            "phase": phase,
            "attempt_id": attempt_id,
            "result_sha256": result_sha256,
            "detail": detail_value,
            "heartbeat_unix": now,
            "heartbeat_at": _utc_now(),
        }
        payload["lane_status_sha256"] = _self_hash(
            payload, "lane_status_sha256"
        )
        _atomic_json(lease.status_path, payload)
        self._heartbeat_cache[lease.lease_id] = (cache_key, now, payload)
        return payload

    def begin_phase(
        self,
        lease: SAELaneLease,
        phase: str,
        *,
        timeout_seconds: float = RENDEZVOUS_TIMEOUT_SECONDS,
    ) -> SAEPhaseAttempt:
        """Form a fresh four-lane cohort and select/create its phase attempt."""

        self._validate_phase(phase)
        abort = self.pending_abort()
        if abort is not None:
            raise SAEAbortRequested(
                f"SAE operation is stopped at a phase boundary: {abort['reason']}"
            )
        deadline = time.time() + timeout_seconds
        ready: list[dict[str, Any]] = []
        while True:
            self.heartbeat(lease, "ready-to-begin", phase=phase)
            ready = self._ready_begin_cohort(phase)
            if len(ready) == EXPECTED_LANES:
                break
            if time.time() >= deadline:
                raise SAEOperatorError(
                    f"phase-begin rendezvous timed out with "
                    f"{len(ready)}/{EXPECTED_LANES} lanes for {phase}"
                )
            time.sleep(0.1)

        members = self._members(ready)
        cohort_sha256 = canonical_sha256(members)
        # Selection is idempotent under the operation lock.  Let every lane
        # execute it so a coordinator-side validation error is observed by all
        # four callers rather than making peers wait for a record that can
        # never be published.
        self._select_or_create_attempt(phase, cohort_sha256, members)

        attempt: SAEPhaseAttempt | None = None
        while True:
            state = self.load_state()
            record = state["phases"][phase]
            if (
                record["status"] == "running"
                and record["cohort_sha256"] == cohort_sha256
                and record["members"] == list(members)
            ):
                attempt = SAEPhaseAttempt(
                    phase=phase,
                    attempt_id=str(record["attempt_id"]),
                    attempt_number=int(record["attempts"]),
                    cohort_sha256=cohort_sha256,
                    members=members,
                )
                break
            if record["status"] == "passed":
                raise SAEPhaseAlreadyPassed(
                    f"phase {phase} already passed; validate its result before skipping"
                )
            if time.time() >= deadline:
                raise SAEOperatorError(
                    f"coordinator did not publish a matching attempt for {phase}"
                )
            time.sleep(0.1)
        assert attempt is not None
        self.heartbeat(
            lease,
            "ready",
            phase=phase,
            attempt_id=attempt.attempt_id,
            force=True,
        )
        self.rendezvous_attempt(lease, attempt, timeout_seconds=timeout_seconds)
        return attempt

    def rendezvous_attempt(
        self,
        lease: SAELaneLease,
        attempt: SAEPhaseAttempt,
        *,
        timeout_seconds: float = RENDEZVOUS_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Release only the exact attempt cohort into its numerical child."""

        deadline = time.time() + timeout_seconds
        while True:
            self.heartbeat(
                lease,
                "ready",
                phase=attempt.phase,
                attempt_id=attempt.attempt_id,
            )
            records = self._attempt_lanes(attempt)
            if len(records) == EXPECTED_LANES and all(
                item["state"] in {"ready", "running"} for item in records
            ):
                release = self._write_release(
                    attempt,
                    kind="start",
                    outcome="start",
                    members=records,
                    result_sha256=None,
                )
                self.heartbeat(
                    lease,
                    "running",
                    phase=attempt.phase,
                    attempt_id=attempt.attempt_id,
                    force=True,
                )
                return release
            failed = [item for item in records if item["state"] in {"failed", "aborted"}]
            if failed:
                raise SAEPeerFailure(
                    f"a controller failed before {attempt.phase} started"
                )
            if time.time() >= deadline:
                raise SAEOperatorError(
                    f"attempt rendezvous timed out with "
                    f"{len(records)}/{EXPECTED_LANES} lanes for {attempt.phase}"
                )
            time.sleep(0.1)

    def require_attempt_peers_healthy(
        self, lease: SAELaneLease, attempt: SAEPhaseAttempt
    ) -> tuple[dict[str, Any], ...]:
        """Fail if a running collective loses or splits a controller lane."""

        del lease  # membership is validated against the immutable attempt cohort
        records = self._attempt_lanes(attempt)
        if len(records) != EXPECTED_LANES:
            raise SAEPeerFailure(
                f"attempt {attempt.attempt_id} has "
                f"{len(records)}/{EXPECTED_LANES} live lanes"
            )
        failures = [
            item for item in records if item["state"] in {"failed", "aborted"}
        ]
        if failures:
            raise SAEPeerFailure(
                f"attempt {attempt.attempt_id} has a failed controller lane"
            )
        return tuple(records)

    def publish_attempt_result(
        self,
        lease: SAELaneLease,
        attempt: SAEPhaseAttempt,
        *,
        outcome: Literal["passed", "failed", "aborted"],
        result_sha256: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish one lane's terminal child verdict without advancing state."""

        if outcome not in {"passed", "failed", "aborted"}:
            raise ValueError("invalid attempt outcome")
        if outcome == "passed":
            _require_sha256(result_sha256, "result_sha256")
            lane_state = "succeeded"
        else:
            if result_sha256 is not None:
                raise ValueError("failed/aborted results cannot bind a success digest")
            lane_state = outcome
        return self.heartbeat(
            lease,
            lane_state,
            phase=attempt.phase,
            attempt_id=attempt.attempt_id,
            result_sha256=result_sha256,
            detail=detail,
            force=True,
        )

    def finalize_attempt(
        self,
        attempt: SAEPhaseAttempt,
        *,
        timeout_seconds: float = RENDEZVOUS_TIMEOUT_SECONDS,
    ) -> SAEAttemptOutcome:
        """Commit the exact four-lane verdict and advance/fail the phase."""

        deadline = time.time() + timeout_seconds
        while True:
            records = self._attempt_lanes(attempt)
            terminal = [
                item for item in records if item["state"] in _TERMINAL_LANE_STATES
            ]
            if len(records) == EXPECTED_LANES and len(terminal) == EXPECTED_LANES:
                break
            if time.time() >= deadline:
                raise SAEOperatorError(
                    f"attempt completion timed out with "
                    f"{len(terminal)}/{EXPECTED_LANES} verdicts for {attempt.phase}"
                )
            time.sleep(0.1)

        successful = [item for item in records if item["state"] == "succeeded"]
        result_sha256: str | None = None
        completion_error: str | None = None
        if len(successful) == EXPECTED_LANES:
            results = {str(item["result_sha256"]) for item in successful}
            if len(results) == 1 and _SHA256_RE.fullmatch(next(iter(results))):
                outcome: Literal["passed", "failed", "aborted"] = "passed"
                result_sha256 = next(iter(results))
            else:
                outcome = "failed"
                completion_error = "successful lanes reported inconsistent results"
        elif any(item["state"] == "failed" for item in records):
            outcome = "failed"
        else:
            outcome = "aborted"
        release = self._write_release(
            attempt,
            kind="complete",
            outcome=outcome,
            members=records,
            result_sha256=result_sha256,
            error=completion_error,
        )
        self._commit_attempt(attempt, release)
        if completion_error is not None:
            raise SAEPeerFailure(completion_error)
        return SAEAttemptOutcome(
            phase=attempt.phase,
            attempt_id=attempt.attempt_id,
            outcome=outcome,
            result_sha256=result_sha256,
            completion_sha256=str(release["release_sha256"]),
        )

    def validate_passed_phase(self, phase: str, result_sha256: str) -> dict[str, Any]:
        """Authorize a restart skip only for the exact revalidated result."""

        self._validate_phase(phase)
        _require_sha256(result_sha256, "result_sha256")
        state = self.load_state()
        record = state["phases"][phase]
        result = record.get("result")
        if (
            record["status"] != "passed"
            or not isinstance(result, Mapping)
            or result.get("result_sha256") != result_sha256
        ):
            raise SAEPhaseAlreadyPassed(
                f"phase {phase} cannot be skipped with result {result_sha256}"
            )
        return dict(record)

    def next_phase(self) -> str | None:
        state = self.load_state()
        for phase in SAE_PHASES:
            if state["phases"][phase]["status"] != "passed":
                return phase
        return None

    def request_abort(
        self, reason: str, *, requested_by: str = "operator"
    ) -> dict[str, Any]:
        """Latch a stop request; numerical children are never signalled here."""

        reason = str(reason).strip()
        requested_by = str(requested_by).strip()
        if not reason or len(reason) > 500:
            raise ValueError("abort reason must contain 1..500 characters")
        if not requested_by or len(requested_by) > 200:
            raise ValueError("requested_by must contain 1..200 characters")
        with _exclusive_lock(self.lock_path):
            self._load_state_unlocked()
            if self.abort_path.exists():
                return self._load_abort_unlocked()
            payload: dict[str, Any] = {
                "schema_version": ABORT_SCHEMA,
                "operation_sha256": self.operation_sha256,
                "reason": reason,
                "requested_by": requested_by,
                "requested_at": _utc_now(),
            }
            payload["abort_sha256"] = _self_hash(payload, "abort_sha256")
            _atomic_json(self.abort_path, payload)
            return payload

    def pending_abort(self) -> dict[str, Any] | None:
        with _exclusive_lock(self.lock_path):
            self._load_state_unlocked()
            return self._load_abort_unlocked() if self.abort_path.exists() else None

    def acknowledge_abort(self, boundary_phase: str | None) -> dict[str, Any]:
        """Record that every child returned and the request reached a boundary."""

        if boundary_phase is not None:
            self._validate_phase(boundary_phase)
        with _exclusive_lock(self.lock_path):
            state = self._load_state_unlocked()
            abort = self._load_abort_unlocked()
            if any(
                record["status"] == "running"
                for record in state["phases"].values()
            ):
                raise SAEOperatorError(
                    "cannot acknowledge a boundary abort while a phase is running"
                )
            state["last_abort_sha256"] = abort["abort_sha256"]
            state["abort_acknowledged_at"] = _utc_now()
            state["abort_boundary_phase"] = boundary_phase
            self._write_state_unlocked(state)
            return self._load_state_unlocked()

    def clear_abort(self) -> bool:
        """Clear only after every lane and inherited child has released its lock."""

        with _exclusive_lock(self.lock_path):
            state = self._load_state_unlocked()
            if not self.abort_path.exists():
                return False
            if any(
                record["status"] == "running"
                for record in state["phases"].values()
            ):
                raise SAEOperatorError("cannot clear abort while a phase is running")
            held = [path for path in self._lane_lock_paths() if _lock_is_held(path)]
            if held:
                raise SAELaneLeaseHeld(
                    "cannot clear abort while a supervisor or inherited child holds a lane"
                )
            request = self._load_abort_unlocked()
            history = self.root / "abort-history"
            history.mkdir(parents=True, exist_ok=True)
            destination = history / f"{request['abort_sha256']}.json"
            if destination.exists():
                existing = _read_object(destination, "abort history")
                if existing != request:
                    raise SAETamperError("abort-history content collision")
            else:
                _atomic_json(destination, request)
            self.abort_path.unlink()
            _fsync_directory(self.abort_path.parent)
            return True

    def status(self) -> dict[str, Any]:
        """Return a JAX-free aggregate of state, abort, and kernel lane leases."""

        state = self.load_state()
        abort = self.pending_abort()
        lanes = self._observe_lanes()
        held = [item for item in lanes if item["lease_held"]]
        orphaned = [item for item in held if item["stale"]]
        vanished = [
            item
            for item in lanes
            if not item["lease_held"]
            and not item["stale"]
            and item["state"] in _ACTIVE_LANE_STATES
        ]
        active_keys = {
            (item.get("phase"), item.get("attempt_id"))
            for item in held
            if item["state"] not in {"starting", "ready-to-begin", "exited"}
        }
        inconsistent = (
            bool(held)
            and len(held) != EXPECTED_LANES
            or len(active_keys) > 1
            or bool(vanished)
        )
        phase_statuses = [state["phases"][phase]["status"] for phase in SAE_PHASES]
        if abort is not None:
            campaign_status = "abort-requested"
        elif orphaned:
            campaign_status = "orphaned"
        elif inconsistent:
            campaign_status = "inconsistent-lanes"
        elif phase_statuses[-1] == "passed":
            campaign_status = "complete"
        elif "running" in phase_statuses or held:
            campaign_status = "running"
        elif "failed" in phase_statuses:
            campaign_status = "failed"
        elif "aborted" in phase_statuses:
            campaign_status = "aborted"
        elif "passed" in phase_statuses:
            campaign_status = "partial"
        else:
            campaign_status = "ready"
        return {
            "schema_version": STATUS_SCHEMA,
            "operation_sha256": self.operation_sha256,
            "campaign_status": campaign_status,
            "next_phase": next(
                (
                    phase
                    for phase in SAE_PHASES
                    if state["phases"][phase]["status"] != "passed"
                ),
                None,
            ),
            "abort_request": abort,
            "state": state,
            "lanes": lanes,
            "orphaned_lanes": [item["lane_id"] for item in orphaned],
            "vanished_lanes": [item["lane_id"] for item in vanished],
        }

    def assert_no_orphaned_lanes(self) -> None:
        status = self.status()
        if status["orphaned_lanes"]:
            raise SAEOrphanedLane("a stale supervisor or inherited child holds a lane")
        if status["vanished_lanes"]:
            raise SAEPeerFailure("a fresh active lane status has lost its kernel lease")

    def _select_or_create_attempt(
        self,
        phase: str,
        cohort_sha256: str,
        members: tuple[dict[str, str], ...],
    ) -> None:
        with _exclusive_lock(self.lock_path):
            state = self._load_state_unlocked()
            if self.abort_path.exists():
                request = self._load_abort_unlocked()
                raise SAEAbortRequested(
                    f"SAE operation is stopped at a phase boundary: {request['reason']}"
                )
            expected = self._next_phase_unlocked(state)
            if expected != phase:
                if state["phases"][phase]["status"] == "passed":
                    raise SAEPhaseAlreadyPassed(
                        f"phase {phase} already passed; validate it before skipping"
                    )
                raise SAEOperatorError(
                    f"phase {phase} cannot start; next fixed phase is {expected}"
                )
            record = state["phases"][phase]
            if (
                record["status"] == "running"
                and record["cohort_sha256"] == cohort_sha256
                and record["members"] == list(members)
            ):
                return
            if record["status"] == "passed":
                raise SAEPhaseAlreadyPassed(
                    f"phase {phase} already passed; validate it before skipping"
                )
            if record["attempts"]:
                record["history"].append(
                    {
                        "status": (
                            "interrupted"
                            if record["status"] == "running"
                            else record["status"]
                        ),
                        "attempt_id": record["attempt_id"],
                        "cohort_sha256": record["cohort_sha256"],
                        "members": record["members"],
                        "started_at": record["started_at"],
                        "completed_at": record["completed_at"],
                        "result": record["result"],
                    }
                )
            record.update(
                status="running",
                attempts=int(record["attempts"]) + 1,
                attempt_id=os.urandom(16).hex(),
                cohort_sha256=cohort_sha256,
                members=list(members),
                started_at=_utc_now(),
                completed_at=None,
                result=None,
            )
            self._write_state_unlocked(state)

    def _commit_attempt(
        self, attempt: SAEPhaseAttempt, release: Mapping[str, Any]
    ) -> None:
        with _exclusive_lock(self.lock_path):
            state = self._load_state_unlocked()
            record = state["phases"][attempt.phase]
            outcome = str(release["outcome"])
            target_status = {
                "passed": "passed",
                "failed": "failed",
                "aborted": "aborted",
            }[outcome]
            result = {
                "result_sha256": release.get("result_sha256"),
                "completion_release_sha256": release["release_sha256"],
                "members": release["members"],
                "error": release.get("error"),
            }
            if record["status"] in {"passed", "failed", "aborted"}:
                if (
                    record["attempt_id"] == attempt.attempt_id
                    and record["status"] == target_status
                    and record["result"] == result
                ):
                    return
                raise SAEOperatorError("phase was already committed by another attempt")
            if (
                record["status"] != "running"
                or record["attempt_id"] != attempt.attempt_id
                or record["cohort_sha256"] != attempt.cohort_sha256
                or record["members"] != list(attempt.members)
            ):
                raise SAEOperationMismatch(
                    "attempt completion does not match the active phase state"
                )
            record.update(
                status=target_status,
                completed_at=_utc_now(),
                result=result,
            )
            self._write_state_unlocked(state)

    def _ready_begin_cohort(self, phase: str) -> list[dict[str, Any]]:
        observed = self._observe_lanes()
        held = [item for item in observed if item["lease_held"]]
        stale = [item for item in held if item["stale"]]
        if stale:
            raise SAEOrphanedLane("a stale lane lock blocks phase rendezvous")
        vanished = [
            item
            for item in observed
            if not item["lease_held"]
            and not item["stale"]
            and item["state"] in _ACTIVE_LANE_STATES
        ]
        if vanished:
            raise SAEPeerFailure("a live lane status has no kernel lease")
        if len(held) > EXPECTED_LANES:
            raise SAEOperatorError(
                f"operation has more than {EXPECTED_LANES} held controller lanes"
            )
        phase_record = self.load_state()["phases"][phase]

        def compatible(item: Mapping[str, Any]) -> bool:
            if item["state"] in {"starting", "ready-to-begin"}:
                return item["state"] == "starting" or (
                    item["phase"] == phase and item["attempt_id"] is None
                )
            # A faster lane may have selected the shared attempt while a peer
            # is still observing the pre-attempt cohort.  This is coherent
            # only when state binds the exact same phase and attempt.
            return (
                phase_record["status"] == "running"
                and item["state"] in {"ready", "running"}
                and item["phase"] == phase
                and item["attempt_id"] == phase_record["attempt_id"]
            )

        incompatible = [
            item
            for item in held
            if not compatible(item)
        ]
        if incompatible:
            raise SAEOperatorError("controller lanes are split across phase cohorts")
        return [
            item
            for item in held
            if (
                item["state"] == "ready-to-begin"
                and item["phase"] == phase
            )
            or (
                phase_record["status"] == "running"
                and item["state"] in {"ready", "running"}
                and item["phase"] == phase
                and item["attempt_id"] == phase_record["attempt_id"]
            )
        ]

    def _attempt_lanes(self, attempt: SAEPhaseAttempt) -> list[dict[str, Any]]:
        observed = self._observe_lanes()
        by_member = {
            (item["lane_id"], item["lease_id"]): item
            for item in observed
            if item["lease_held"]
        }
        stale = [item for item in by_member.values() if item["stale"]]
        if stale:
            raise SAEOrphanedLane("attempt contains a stale kernel-held lane")
        expected = {(item["lane_id"], item["lease_id"]) for item in attempt.members}
        extra = set(by_member) - expected
        if extra:
            raise SAEOperatorError("another held lane is outside the active attempt")
        records: list[dict[str, Any]] = []
        for member in attempt.members:
            key = (member["lane_id"], member["lease_id"])
            item = by_member.get(key)
            if item is None:
                continue
            if item.get("phase") != attempt.phase or item.get(
                "attempt_id"
            ) != attempt.attempt_id:
                # Another cohort member can still be crossing from the
                # pre-attempt rendezvous.  Omit it until it publishes the
                # selected attempt; any different phase/terminal state is a
                # genuine split and remains fatal.
                if item.get("state") in {"starting", "ready-to-begin"} and (
                    item.get("phase") in {None, attempt.phase}
                    and item.get("attempt_id") is None
                ):
                    continue
                raise SAEOperatorError("attempt lanes disagree on phase or attempt ID")
            records.append(item)
        return sorted(records, key=lambda item: item["lane_id"])

    @staticmethod
    def _members(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], ...]:
        members = tuple(
            sorted(
                (
                    {
                        "lane_id": str(item["lane_id"]),
                        "lease_id": str(item["lease_id"]),
                        "hostname": str(item["hostname"]),
                    }
                    for item in records
                ),
                key=lambda item: item["lane_id"],
            )
        )
        if len(members) != EXPECTED_LANES:
            raise SAEOperatorError("attempt membership must contain exactly four lanes")
        if len({item["lane_id"] for item in members}) != EXPECTED_LANES:
            raise SAEOperatorError("attempt membership contains duplicate lanes")
        return members

    def _release_path(self, attempt: SAEPhaseAttempt, kind: str) -> Path:
        return self.root / "attempts" / attempt.attempt_id / f"{kind}.json"

    def _write_release(
        self,
        attempt: SAEPhaseAttempt,
        *,
        kind: Literal["start", "complete"],
        outcome: str,
        members: Sequence[Mapping[str, Any]],
        result_sha256: str | None,
        error: str | None = None,
    ) -> dict[str, Any]:
        member_values = []
        for item in sorted(members, key=lambda value: str(value["lane_id"])):
            member = {
                "lane_id": str(item["lane_id"]),
                "lease_id": str(item["lease_id"]),
                "hostname": str(item["hostname"]),
                # A lane can move from ready to running immediately after the
                # start release is visible.  Normalize that transient state so
                # concurrent writers propose identical release content.
                "state": "ready" if kind == "start" else str(item["state"]),
                "result_sha256": (
                    None if kind == "start" else item.get("result_sha256")
                ),
            }
            member_values.append(member)
        payload: dict[str, Any] = {
            "schema_version": RELEASE_SCHEMA,
            "operation_sha256": self.operation_sha256,
            "kind": kind,
            "phase": attempt.phase,
            "attempt_id": attempt.attempt_id,
            "attempt_number": attempt.attempt_number,
            "cohort_sha256": attempt.cohort_sha256,
            "outcome": outcome,
            "result_sha256": result_sha256,
            "error": error,
            "members": member_values,
            "created_at": _utc_now(),
        }
        payload["release_sha256"] = _self_hash(payload, "release_sha256")
        path = self._release_path(attempt, kind)
        with _exclusive_lock(self.lock_path):
            if path.exists():
                existing = self._load_release(path, attempt, kind)
                comparable_existing = dict(existing)
                comparable_payload = dict(payload)
                # Concurrent lanes can propose the same release milliseconds
                # apart.  ``created_at`` is not part of semantic equality.
                comparable_existing.pop("created_at")
                comparable_existing.pop("release_sha256")
                comparable_payload.pop("created_at")
                comparable_payload.pop("release_sha256")
                if comparable_existing != comparable_payload:
                    raise SAEOperatorError("attempt release content collision")
                return existing
            _atomic_json(path, payload)
        return payload

    def _load_release(
        self, path: Path, attempt: SAEPhaseAttempt, kind: str
    ) -> dict[str, Any]:
        value = _read_object(path, "SAE attempt release")
        claimed = value.get("release_sha256")
        if (
            value.get("schema_version") != RELEASE_SCHEMA
            or value.get("operation_sha256") != self.operation_sha256
            or value.get("kind") != kind
            or value.get("phase") != attempt.phase
            or value.get("attempt_id") != attempt.attempt_id
            or value.get("attempt_number") != attempt.attempt_number
            or value.get("cohort_sha256") != attempt.cohort_sha256
            or not isinstance(claimed, str)
            or claimed != _self_hash(value, "release_sha256")
        ):
            raise SAETamperError(f"SAE attempt release failed validation: {path}")
        return value

    def _load_state_unlocked(self) -> dict[str, Any]:
        if not self.state_path.exists():
            raise SAEOperatorError(f"SAE operation state is absent: {self.state_path}")
        state = _read_object(self.state_path, "SAE operation state")
        claimed = state.get("state_sha256")
        if (
            state.get("schema_version") != STATE_SCHEMA
            or state.get("operation_sha256") != self.operation_sha256
            or state.get("phase_plan") != list(SAE_PHASES)
            or not isinstance(state.get("spec"), Mapping)
            or canonical_sha256(state["spec"]) != self.operation_sha256
            or not isinstance(claimed, str)
            or claimed != _self_hash(state, "state_sha256")
        ):
            raise SAETamperError("SAE operation state failed identity/self-hash validation")
        phases = state.get("phases")
        if not isinstance(phases, Mapping) or set(phases) != set(SAE_PHASES):
            raise SAETamperError("SAE operation state has an invalid phase map")
        first_nonpassed: str | None = None
        running = 0
        for phase in SAE_PHASES:
            record = phases[phase]
            if not isinstance(record, Mapping) or set(record) != set(_empty_phase()):
                raise SAETamperError(f"SAE phase record has invalid fields: {phase}")
            status = record.get("status")
            attempts = record.get("attempts")
            history = record.get("history")
            if (
                status not in _PHASE_STATUSES
                or isinstance(attempts, bool)
                or not isinstance(attempts, int)
                or attempts < 0
                or not isinstance(history, list)
            ):
                raise SAETamperError(f"SAE phase record is invalid: {phase}")
            if status != "passed" and first_nonpassed is None:
                first_nonpassed = phase
            elif first_nonpassed is not None and status != "pending":
                raise SAETamperError(
                    "SAE phases are not a passed prefix followed by one active boundary"
                )
            if status == "pending":
                if attempts != 0 or any(
                    record.get(name) is not None
                    for name in (
                        "attempt_id",
                        "cohort_sha256",
                        "members",
                        "started_at",
                        "completed_at",
                        "result",
                    )
                ) or history:
                    raise SAETamperError(f"pending SAE phase contains attempt state: {phase}")
                continue
            attempt_id = record.get("attempt_id")
            cohort = record.get("cohort_sha256")
            members = record.get("members")
            if (
                attempts <= 0
                or not isinstance(attempt_id, str)
                or _ATTEMPT_RE.fullmatch(attempt_id) is None
                or not isinstance(cohort, str)
                or _SHA256_RE.fullmatch(cohort) is None
                or not isinstance(members, list)
                or len(members) != EXPECTED_LANES
                or canonical_sha256(tuple(members)) != cohort
                or record.get("started_at") is None
            ):
                raise SAETamperError(f"SAE phase attempt identity is invalid: {phase}")
            if status == "running":
                running += 1
                if record.get("completed_at") is not None or record.get("result") is not None:
                    raise SAETamperError(f"running SAE phase is already completed: {phase}")
            else:
                if record.get("completed_at") is None or not isinstance(
                    record.get("result"), Mapping
                ):
                    raise SAETamperError(f"terminal SAE phase lacks its result: {phase}")
                result = record["result"]
                if status == "passed":
                    _require_sha256(result.get("result_sha256"), "result_sha256")
                elif result.get("result_sha256") is not None:
                    raise SAETamperError("failed/aborted phase binds a success digest")
        if running > 1:
            raise SAETamperError("SAE operation has multiple running phases")
        return state

    def _write_state_unlocked(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        payload["updated_at"] = _utc_now()
        payload.pop("state_sha256", None)
        payload["state_sha256"] = _self_hash(payload, "state_sha256")
        _atomic_json(self.state_path, payload)

    def _load_abort_unlocked(self) -> dict[str, Any]:
        if not self.abort_path.exists():
            raise SAEOperatorError("SAE abort request is absent")
        request = _read_object(self.abort_path, "SAE abort request")
        claimed = request.get("abort_sha256")
        if (
            request.get("schema_version") != ABORT_SCHEMA
            or request.get("operation_sha256") != self.operation_sha256
            or not isinstance(request.get("reason"), str)
            or not request["reason"]
            or not isinstance(claimed, str)
            or claimed != _self_hash(request, "abort_sha256")
        ):
            raise SAETamperError("SAE abort request failed validation")
        return request

    def _read_lane_if_valid(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        value = _read_object(path, "SAE lane status")
        claimed = value.get("lane_status_sha256")
        expected_lane = path.stem.removeprefix("lane-")
        if (
            value.get("schema_version") != LANE_SCHEMA
            or value.get("operation_sha256") != self.operation_sha256
            or value.get("lane_id") != expected_lane
            or _LANE_RE.fullmatch(str(value.get("lane_id", ""))) is None
            or _ATTEMPT_RE.fullmatch(str(value.get("lease_id", ""))) is None
            or value.get("state") not in (_ACTIVE_LANE_STATES | {"exited"})
            or not isinstance(value.get("heartbeat_unix"), (int, float))
            or isinstance(value.get("heartbeat_unix"), bool)
            or not isinstance(claimed, str)
            or claimed != _self_hash(value, "lane_status_sha256")
        ):
            raise SAETamperError(f"SAE lane status failed validation: {path}")
        phase = value.get("phase")
        attempt = value.get("attempt_id")
        state = value["state"]
        if phase is not None and phase not in SAE_PHASES:
            raise SAETamperError(f"SAE lane names an invalid phase: {path}")
        if attempt is not None and _ATTEMPT_RE.fullmatch(str(attempt)) is None:
            raise SAETamperError(f"SAE lane has an invalid attempt ID: {path}")
        if state == "ready-to-begin" and (phase is None or attempt is not None):
            raise SAETamperError(f"SAE ready-to-begin lane is invalid: {path}")
        if state in {"ready", "running", "succeeded"} and (
            phase is None or attempt is None
        ):
            raise SAETamperError(f"SAE active lane lacks attempt identity: {path}")
        if state in {"failed", "aborted"} and attempt is not None and phase is None:
            raise SAETamperError(f"SAE failed lane has an attempt without a phase: {path}")
        if state == "succeeded":
            _require_sha256(value.get("result_sha256"), "result_sha256")
        elif value.get("result_sha256") is not None:
            raise SAETamperError(f"non-successful lane binds a result: {path}")
        return value

    def _observe_lanes(self) -> list[dict[str, Any]]:
        lanes_dir = self.root / "lanes"
        lanes_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        observed: list[dict[str, Any]] = []
        status_by_lane: dict[str, Path] = {}
        for path in sorted(lanes_dir.glob("lane-*.json")):
            lane_id = path.stem.removeprefix("lane-")
            if _LANE_RE.fullmatch(lane_id) is None:
                raise SAETamperError(f"invalid SAE lane status name: {path}")
            if lane_id in status_by_lane:
                raise SAETamperError("duplicate SAE lane status")
            status_by_lane[lane_id] = path
        lock_by_lane = {
            path.stem.removeprefix("lane-"): path
            for path in sorted(lanes_dir.glob("lane-*.lock"))
        }
        for lane_id in sorted(set(status_by_lane) | set(lock_by_lane)):
            if _LANE_RE.fullmatch(lane_id) is None:
                raise SAETamperError(f"invalid SAE lane lock name: {lane_id}")
            status_path = status_by_lane.get(
                lane_id, lanes_dir / f"lane-{lane_id}.json"
            )
            lock = lock_by_lane.get(lane_id)
            startup_path = lanes_dir / ".startup" / f"lane-{lane_id}.lock"
            # A new owner takes this short gate before the durable lane lock and
            # releases it only after publishing its lease ID.  Waiting here
            # therefore prevents both the no-status first launch race and the
            # more subtle reused-lock/old-status restart race.
            with _bounded_exclusive_lock(
                startup_path,
                timeout_seconds=LANE_STATUS_PUBLICATION_GRACE_SECONDS,
            ):
                status = self._read_lane_if_valid(status_path)
                held = False if lock is None else _lock_is_held(lock)
            if status is None and held:
                raise SAEOrphanedLane(
                    "kernel-held SAE lane did not publish its initial status: "
                    f"{lane_id}"
                )
            if status is None:
                continue
            age = max(0.0, now - float(status["heartbeat_unix"]))
            item = dict(status)
            item.update(
                lease_held=held,
                heartbeat_age_seconds=age,
                stale=age > LANE_FRESH_SECONDS,
            )
            observed.append(item)
        return observed

    def _lane_lock_paths(self) -> tuple[Path, ...]:
        lanes_dir = self.root / "lanes"
        return tuple(sorted(lanes_dir.glob("lane-*.lock"))) if lanes_dir.exists() else ()

    @staticmethod
    def _validate_phase(phase: str) -> None:
        if phase not in SAE_PHASES:
            raise ValueError(f"unknown SAE phase: {phase!r}")

    @staticmethod
    def _next_phase_unlocked(state: Mapping[str, Any]) -> str | None:
        phases = state["phases"]
        return next(
            (phase for phase in SAE_PHASES if phases[phase]["status"] != "passed"),
            None,
        )


__all__ = [
    "ABORT_SCHEMA",
    "EXPECTED_LANES",
    "LANE_FRESH_SECONDS",
    "LANE_STATUS_PUBLICATION_GRACE_SECONDS",
    "SAEAbortRequested",
    "SAEAttemptOutcome",
    "SAEControl",
    "SAELaneLease",
    "SAELaneLeaseHeld",
    "SAEOperationMismatch",
    "SAEOperationSpec",
    "SAEOperatorError",
    "SAEOrphanedLane",
    "SAEPeerFailure",
    "SAEPhaseAlreadyPassed",
    "SAEPhaseAttempt",
    "SAE_PHASES",
    "SAETamperError",
    "canonical_sha256",
]
