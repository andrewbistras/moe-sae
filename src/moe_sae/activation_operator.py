"""Durable operator control for the activation-only TPU campaign.

The numerical work remains in :mod:`moe_sae.harvest`.  This module owns only
the small shared-POSIX control record used to sequence its four coarse phases:
benchmark, canary, train capture, and eval capture.  It intentionally has no
JAX or cloud dependency, so status and abort requests are safe from an ordinary
controller shell and cannot accidentally initialize a TPU runtime.

Abort is cooperative and phase-bounded.  A request is durable immediately and
remains active until an operator explicitly clears it after every controller
has returned from the current collective phase.  We never signal or kill one
member of a live JAX collective.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Any, Iterator


ACTIVATION_PHASES = ("benchmark", "canary", "train", "eval")
STATE_SCHEMA = "gemma4-activation-operator-v1"
ABORT_SCHEMA = "gemma4-activation-abort-v1"
SUPERVISOR_SCHEMA = "gemma4-activation-supervisor-v1"
SUPERVISOR_PROCESS_COUNT = 4
SUPERVISOR_FRESH_SECONDS = 60.0
SUPERVISOR_HEARTBEAT_INTERVAL_SECONDS = 10.0
SUPERVISOR_RENDEZVOUS_SECONDS = 300.0
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RUN_ID_RE = re.compile(r"[0-9a-f]{32}")


class ActivationOperatorError(RuntimeError):
    """The shared activation operator state is invalid or incompatible."""


class ActivationAbortRequested(ActivationOperatorError):
    """A durable cooperative abort was acknowledged at a phase boundary."""


class ActivationSupervisorLeaseHeld(ActivationOperatorError):
    """Another activation supervisor already owns this controller lane."""


@dataclass(frozen=True, slots=True)
class ActivationSupervisorLease:
    descriptor: int
    lane_id: str
    lease_id: str
    operation_sha256: str
    run_id: str
    status_path: Path


@dataclass(frozen=True, slots=True)
class ActivationRunSpec:
    """Immutable identity of one activation campaign."""

    config_sha256: str
    checkpoint: str
    checkpoint_sha256: str
    grain_train: str
    grain_eval: str
    dataset_train_manifest: str
    dataset_eval_manifest: str

    def __post_init__(self) -> None:
        if not _SHA256_RE.fullmatch(self.config_sha256):
            raise ValueError("config_sha256 must be a lowercase SHA-256")
        if not _SHA256_RE.fullmatch(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256")
        for name in (
            "checkpoint",
            "grain_train",
            "grain_eval",
            "dataset_train_manifest",
            "dataset_eval_manifest",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")

    @property
    def sha256(self) -> str:
        return _sha256_mapping(asdict(self))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = {name: value for name, value in payload.items() if name != field}
    return _sha256_mapping(unsigned)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace one JSON record durably, including the parent-directory entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ActivationOperatorError(f"cannot read activation control record: {path}") from error
    if not isinstance(payload, dict):
        raise ActivationOperatorError(f"activation control record is not an object: {path}")
    return payload


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b", buffering=0) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _empty_phases() -> dict[str, dict[str, Any]]:
    return {
        phase: {
            "status": "pending",
            "attempts": 0,
            "attempt_id": None,
            "started_at": None,
            "completed_at": None,
            "detail": None,
        }
        for phase in ACTIVATION_PHASES
    }


class ActivationControl:
    """Concurrency-safe shared control record for one activation campaign."""

    def __init__(self, artifact_root: str | Path):
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.root = self.artifact_root / "operator" / "activation"
        self.state_path = self.root / "state.json"
        self.abort_path = self.root / "abort.json"
        self.abort_history = self.root / "abort-history"
        self.lock_path = self.root / ".lock"
        self._supervisor_heartbeat_cache: dict[
            str, tuple[str, str | None, float, dict[str, Any]]
        ] = {}

    @contextmanager
    def supervisor_lease(
        self,
        lane: str | None = None,
        *,
        operation_sha256: str | None = None,
    ) -> Iterator[ActivationSupervisorLease]:
        """Hold one lifetime lease and yield its child-inheritable descriptor."""

        identity = str(lane or platform.node()).strip()
        if not identity:
            raise ActivationOperatorError("cannot identify this controller lane")
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        operation = operation_sha256 or ("0" * 64)
        if not _SHA256_RE.fullmatch(operation):
            raise ValueError("supervisor operation_sha256 must be a SHA-256")
        path = self.root / "leases" / f"controller-{digest}.lock"
        status_path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b", buffering=0)
        try:
            with _exclusive_lock(self.lock_path):
                state = self._require_state_unlocked()
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ActivationSupervisorLeaseHeld(
                        "another activation-farm process owns this controller lane"
                    ) from error
            lease = ActivationSupervisorLease(
                descriptor=handle.fileno(),
                lane_id=digest,
                lease_id=os.urandom(16).hex(),
                operation_sha256=operation,
                run_id=str(state["run_id"]),
                status_path=status_path,
            )
            marker: dict[str, Any] = {
                "schema_version": SUPERVISOR_SCHEMA,
                "run_id": lease.run_id,
                "operation_sha256": lease.operation_sha256,
                "lane_id": lease.lane_id,
                "lease_id": lease.lease_id,
                "pid": os.getpid(),
                "hostname": platform.node(),
                "acquired_at": _utc_now(),
            }
            marker["lease_sha256"] = _self_hash(marker, "lease_sha256")
            _atomic_json(
                self.root
                / "leases"
                / "markers"
                / f"controller-{lease.lane_id}.json",
                marker,
            )
            self.supervisor_heartbeat(lease, "starting", step=None)
            try:
                yield lease
            except BaseException as error:
                final_state = (
                    "aborted"
                    if isinstance(error, ActivationAbortRequested)
                    else "failed"
                )
                active_step = None
                if lease.status_path.exists():
                    active_step = _read_json(lease.status_path).get("step")
                self.supervisor_heartbeat(
                    lease, final_state, step=active_step, error=error
                )
                raise
            else:
                self.supervisor_heartbeat(lease, "exited", step=None)
        finally:
            # Closing, rather than LOCK_UN, is intentional.  A phase child
            # inherits this open file description; if the parent dies, the
            # kernel keeps the lane locked until the orphan child also exits.
            handle.close()

    def supervisor_heartbeat(
        self,
        lease: ActivationSupervisorLease,
        state: str,
        *,
        step: str | None,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        if not str(state).strip() or len(str(state)) > 64:
            raise ValueError("supervisor state must contain 1..64 characters")
        if step is not None and (not str(step).strip() or len(str(step)) > 160):
            raise ValueError("supervisor step must contain 1..160 characters")
        now = time.time()
        cached = self._supervisor_heartbeat_cache.get(lease.lease_id)
        if (
            error is None
            and cached is not None
            and cached[0] == str(state)
            and cached[1] == step
            and now - cached[2] < SUPERVISOR_HEARTBEAT_INTERVAL_SECONDS
        ):
            return cached[3]
        payload: dict[str, Any] = {
            "schema_version": SUPERVISOR_SCHEMA,
            "run_id": lease.run_id,
            "operation_sha256": lease.operation_sha256,
            "lane_id": lease.lane_id,
            "lease_id": lease.lease_id,
            "pid": os.getpid(),
            "hostname": platform.node(),
            "state": str(state),
            "step": step,
            "heartbeat_unix": now,
        }
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error"] = str(error)[:2_000]
        payload["status_sha256"] = _self_hash(payload, "status_sha256")
        _atomic_json(lease.status_path, payload)
        self._supervisor_heartbeat_cache[lease.lease_id] = (
            str(state),
            step,
            now,
            payload,
        )
        return payload

    def supervisor_rendezvous(
        self,
        lease: ActivationSupervisorLease,
        step: str,
        *,
        expected: int = SUPERVISOR_PROCESS_COUNT,
        timeout_seconds: float = SUPERVISOR_RENDEZVOUS_SECONDS,
    ) -> list[dict[str, Any]]:
        """Wait until exactly four fresh, matching controller lanes are ready."""

        deadline = time.time() + timeout_seconds
        while True:
            self.supervisor_heartbeat(lease, "ready", step=step)
            records = self._matching_supervisors(lease, step=step)
            active_records = self._matching_supervisors(lease, step=None)
            released = self._load_supervisor_release(
                lease, step=step, kind="start"
            )
            if released is not None and self._release_matches(
                released, lease, active_records, expected=expected
            ):
                return records
            failures = [
                item for item in active_records if item["state"] == "failed"
            ]
            if failures:
                raise ActivationOperatorError(
                    f"activation supervisor peer failed before {step}"
                )
            ready = [item for item in records if item["state"] == "ready"]
            if len(ready) == expected:
                self._write_supervisor_release(
                    lease,
                    step=step,
                    kind="start",
                    members=ready,
                    outcome="start",
                )
                return ready
            if len(ready) > expected:
                raise ActivationOperatorError(
                    f"activation rendezvous found more than {expected} controller lanes"
                )
            if time.time() >= deadline:
                raise ActivationOperatorError(
                    f"activation rendezvous timed out with {len(ready)}/{expected} "
                    f"fresh controller lanes at {step}"
                )
            time.sleep(0.5)

    def supervisor_wait_completion(
        self,
        lease: ActivationSupervisorLease,
        step: str,
        *,
        local_state: str,
        expected: int = SUPERVISOR_PROCESS_COUNT,
        timeout_seconds: float = SUPERVISOR_RENDEZVOUS_SECONDS,
    ) -> str:
        """Publish one child verdict and wait for the four-host verdict."""

        if local_state not in {"succeeded", "aborted", "failed"}:
            raise ValueError("invalid local supervisor completion state")
        deadline = time.time() + timeout_seconds
        while True:
            self.supervisor_heartbeat(lease, local_state, step=step)
            released = self._load_supervisor_release(
                lease, step=step, kind="complete"
            )
            active_records = self._matching_supervisors(lease, step=None)
            if released is not None and self._release_matches(
                released, lease, active_records, expected=expected
            ):
                return str(released["outcome"])
            records = self._matching_supervisors(lease, step=step)
            failures = [item for item in records if item["state"] == "failed"]
            if failures:
                raise ActivationOperatorError(
                    f"activation subprocess failed on a controller at {step}"
                )
            terminal = [
                item
                for item in records
                if item["state"] in {"succeeded", "aborted"}
            ]
            if len(terminal) == expected:
                outcome = (
                    "aborted"
                    if any(item["state"] == "aborted" for item in terminal)
                    else "succeeded"
                )
                self._write_supervisor_release(
                    lease,
                    step=step,
                    kind="complete",
                    members=terminal,
                    outcome=outcome,
                )
                return outcome
            if time.time() >= deadline:
                raise ActivationOperatorError(
                    f"activation completion timed out with {len(terminal)}/{expected} "
                    f"controller verdicts at {step}"
                )
            time.sleep(0.5)

    def require_supervisor_peers_healthy(
        self,
        lease: ActivationSupervisorLease,
        step: str,
        *,
        expected: int = SUPERVISOR_PROCESS_COUNT,
        fresh_seconds: float = SUPERVISOR_FRESH_SECONDS,
    ) -> list[dict[str, Any]]:
        records = self._matching_supervisors(lease, step=step)
        if len(records) != expected:
            raise ActivationOperatorError(
                f"activation step {step} lost a controller lane: "
                f"{len(records)}/{expected} visible"
            )
        now = time.time()
        stale = [
            item for item in records
            if now - float(item["heartbeat_unix"]) > fresh_seconds
        ]
        failed = [item for item in records if item["state"] == "failed"]
        if stale:
            raise ActivationOperatorError(
                f"activation step {step} has a stale controller heartbeat"
            )
        if failed:
            raise ActivationOperatorError(
                f"activation step {step} has a failed controller peer"
            )
        return records

    def _matching_supervisors(
        self, lease: ActivationSupervisorLease, *, step: str | None
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        now = time.time()
        for path in sorted((self.root / "leases").glob("controller-*.json")):
            value = _read_json(path)
            claimed = value.get("status_sha256")
            if (
                value.get("schema_version") != SUPERVISOR_SCHEMA
                or not _SHA256_RE.fullmatch(str(claimed or ""))
                or claimed != _self_hash(value, "status_sha256")
            ):
                raise ActivationOperatorError(
                    f"activation supervisor status failed validation: {path}"
                )
            if (
                value.get("run_id") == lease.run_id
                and value.get("operation_sha256") == lease.operation_sha256
                and (step is None or value.get("step") == step)
            ):
                try:
                    heartbeat_age = now - float(value.get("heartbeat_unix"))
                except (TypeError, ValueError):
                    heartbeat_age = math.inf
                if heartbeat_age < 0.0 or heartbeat_age > SUPERVISOR_FRESH_SECONDS:
                    continue
                lane_id = str(value.get("lane_id", ""))
                lease_id = str(value.get("lease_id", ""))
                if (
                    not re.fullmatch(r"[0-9a-f]{24}", lane_id)
                    or not _RUN_ID_RE.fullmatch(lease_id)
                ):
                    raise ActivationOperatorError(
                        f"activation supervisor identity is invalid: {path}"
                    )
                lock_path = self.root / "leases" / f"controller-{lane_id}.lock"
                marker = _read_json(
                    self.root
                    / "leases"
                    / "markers"
                    / f"controller-{lane_id}.json"
                )
                marker_sha256 = marker.get("lease_sha256")
                if (
                    marker.get("schema_version") != SUPERVISOR_SCHEMA
                    or marker.get("run_id") != lease.run_id
                    or marker.get("operation_sha256") != lease.operation_sha256
                    or marker.get("lane_id") != lane_id
                    or marker.get("lease_id") != lease_id
                    or not _SHA256_RE.fullmatch(str(marker_sha256 or ""))
                    or marker_sha256 != _self_hash(marker, "lease_sha256")
                ):
                    continue
                if lane_id == lease.lane_id and lease_id == lease.lease_id:
                    result.append(value)
                    continue
                handle = lock_path.open("a+b", buffering=0)
                try:
                    try:
                        fcntl.flock(
                            handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                    except BlockingIOError:
                        result.append(value)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()
        return result

    def _supervisor_release_path(
        self, lease: ActivationSupervisorLease, *, step: str, kind: str
    ) -> Path:
        step_sha256 = hashlib.sha256(step.encode("utf-8")).hexdigest()
        return (
            self.root
            / "rendezvous"
            / lease.operation_sha256
            / f"{kind}-{step_sha256}.json"
        )

    def _load_supervisor_release(
        self, lease: ActivationSupervisorLease, *, step: str, kind: str
    ) -> dict[str, Any] | None:
        path = self._supervisor_release_path(lease, step=step, kind=kind)
        if not path.exists():
            return None
        value = _read_json(path)
        claimed = value.get("release_sha256")
        if (
            value.get("schema_version") != SUPERVISOR_SCHEMA
            or value.get("run_id") != lease.run_id
            or value.get("operation_sha256") != lease.operation_sha256
            or value.get("step") != step
            or value.get("kind") != kind
            or not isinstance(value.get("members"), list)
            or not _SHA256_RE.fullmatch(str(claimed or ""))
            or claimed != _self_hash(value, "release_sha256")
        ):
            raise ActivationOperatorError(
                f"activation supervisor release failed validation: {path}"
            )
        return value

    @staticmethod
    def _release_matches(
        release: Mapping[str, Any],
        lease: ActivationSupervisorLease,
        records: list[dict[str, Any]],
        *,
        expected: int,
    ) -> bool:
        members = release.get("members")
        if not isinstance(members, list) or len(members) != expected:
            return False
        identities = {
            (str(item.get("lane_id")), str(item.get("lease_id")))
            for item in members
            if isinstance(item, Mapping)
        }
        active = {
            (str(item.get("lane_id")), str(item.get("lease_id")))
            for item in records
        }
        return (
            len(identities) == expected
            and identities == active
            and (lease.lane_id, lease.lease_id) in identities
        )

    def _write_supervisor_release(
        self,
        lease: ActivationSupervisorLease,
        *,
        step: str,
        kind: str,
        members: list[dict[str, Any]],
        outcome: str,
    ) -> None:
        payload: dict[str, Any] = {
            "schema_version": SUPERVISOR_SCHEMA,
            "run_id": lease.run_id,
            "operation_sha256": lease.operation_sha256,
            "step": step,
            "kind": kind,
            "outcome": outcome,
            "members": sorted(
                [
                    {
                        "lane_id": str(item["lane_id"]),
                        "lease_id": str(item["lease_id"]),
                    }
                    for item in members
                ],
                key=lambda item: item["lane_id"],
            ),
            "released_at": _utc_now(),
        }
        payload["release_sha256"] = _self_hash(payload, "release_sha256")
        _atomic_json(
            self._supervisor_release_path(lease, step=step, kind=kind), payload
        )

    def begin(self, spec: ActivationRunSpec) -> dict[str, Any]:
        """Create or join the one provenance-matching campaign."""

        with _exclusive_lock(self.lock_path):
            if self.state_path.exists():
                state = self._load_state_unlocked()
                if state["spec"] != asdict(spec) or state["spec_sha256"] != spec.sha256:
                    raise ActivationOperatorError(
                        "activation control belongs to a different checkpoint/dataset/config; "
                        "use a fresh artifact root"
                    )
                return state
            now = _utc_now()
            state: dict[str, Any] = {
                "schema_version": STATE_SCHEMA,
                "run_id": os.urandom(16).hex(),
                "spec": asdict(spec),
                "spec_sha256": spec.sha256,
                "created_at": now,
                "updated_at": now,
                "phases": _empty_phases(),
                "last_abort_sha256": None,
            }
            self._write_state_unlocked(state)
            return state

    def load_state(self) -> dict[str, Any] | None:
        with _exclusive_lock(self.lock_path):
            return self._load_state_unlocked() if self.state_path.exists() else None

    def mark_running(self, phase: str) -> dict[str, Any]:
        self._validate_phase(phase)
        with _exclusive_lock(self.lock_path):
            state = self._require_state_unlocked()
            record = state["phases"][phase]
            if record["status"] == "passed":
                return state
            if record["status"] != "running":
                record.update(
                    status="running",
                    attempts=int(record["attempts"]) + 1,
                    attempt_id=os.urandom(16).hex(),
                    started_at=_utc_now(),
                    completed_at=None,
                    detail=None,
                )
                self._write_state_unlocked(state)
            return state

    def mark_passed(
        self,
        phase: str,
        detail: Mapping[str, Any] | None = None,
        *,
        attempt_id: str | None = None,
    ) -> dict[str, Any]:
        return self._finish_phase(
            phase, "passed", detail, attempt_id=attempt_id
        )

    def mark_failed(
        self,
        phase: str,
        error: BaseException,
        *,
        attempt_id: str | None = None,
    ) -> dict[str, Any]:
        return self._finish_phase(
            phase,
            "failed",
            {"error_type": type(error).__name__, "error": str(error)},
            attempt_id=attempt_id,
        )

    def request_abort(self, reason: str) -> dict[str, Any]:
        """Durably request a cooperative stop at the next phase boundary."""

        reason = str(reason).strip()
        if not reason:
            raise ValueError("abort reason cannot be empty")
        if len(reason) > 500:
            raise ValueError("abort reason cannot exceed 500 characters")
        with _exclusive_lock(self.lock_path):
            state = self._require_state_unlocked()
            if self.abort_path.exists():
                request = self._load_abort_unlocked()
                if request["run_id"] != state["run_id"]:
                    raise ActivationOperatorError("abort request belongs to another run")
                return request
            request: dict[str, Any] = {
                "schema_version": ABORT_SCHEMA,
                "request_id": os.urandom(16).hex(),
                "run_id": state["run_id"],
                "spec_sha256": state["spec_sha256"],
                "reason": reason,
                "requested_at": _utc_now(),
                "semantics": "cooperative-next-phase-boundary",
            }
            request["abort_sha256"] = _self_hash(request, "abort_sha256")
            _atomic_json(self.abort_path, request)
            return request

    def pending_abort(self) -> dict[str, Any] | None:
        with _exclusive_lock(self.lock_path):
            return self._load_abort_unlocked() if self.abort_path.exists() else None

    def acknowledge_abort(self, next_phase: str) -> dict[str, Any]:
        """Record an active request at a boundary without consuming it.

        The marker deliberately remains present so four independent controller
        supervisors cannot race such that the first acknowledgement hides the
        request from the other three.
        """

        self._validate_phase(next_phase)
        with _exclusive_lock(self.lock_path):
            state = self._require_state_unlocked()
            if not self.abort_path.exists():
                return state
            request = self._load_abort_unlocked()
            if (
                request["run_id"] != state["run_id"]
                or request["spec_sha256"] != state["spec_sha256"]
            ):
                raise ActivationOperatorError(
                    "abort request identity does not match the active run"
                )
            record = state["phases"][next_phase]
            if record["status"] != "passed":
                record.update(
                    status="aborted",
                    completed_at=_utc_now(),
                    detail={
                        "abort_sha256": request["abort_sha256"],
                        "reason": request["reason"],
                    },
                )
            state["last_abort_sha256"] = request["abort_sha256"]
            self._archive_abort_unlocked(request)
            self._write_state_unlocked(state)
            return state

    def clear_abort(self) -> bool:
        """Clear a boundary request before a deliberate all-controller resume."""

        with _exclusive_lock(self.lock_path):
            state = self._require_state_unlocked()
            if any(
                state["phases"][phase]["status"] == "running"
                for phase in ACTIVATION_PHASES
            ):
                raise ActivationOperatorError(
                    "cannot clear an activation abort while a phase is running"
                )
            lease_handles = self._probe_supervisor_leases_unlocked()
            try:
                if not self.abort_path.exists():
                    return False
                request = self._load_abort_unlocked()
                if (
                    request["run_id"] != state["run_id"]
                    or request["spec_sha256"] != state["spec_sha256"]
                ):
                    raise ActivationOperatorError(
                        "abort request identity does not match the active run"
                    )
                self._archive_abort_unlocked(request)
                self.abort_path.unlink()
                _fsync_directory(self.abort_path.parent)
                return True
            finally:
                for handle in lease_handles:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    finally:
                        handle.close()

    def _probe_supervisor_leases_unlocked(self) -> list[Any]:
        """Hold every free lane lock or fail if any supervisor/orphan owns one."""

        handles: list[Any] = []
        lease_root = self.root / "leases"
        try:
            for path in sorted(lease_root.glob("controller-*.lock")):
                handle = path.open("a+b", buffering=0)
                try:
                    fcntl.flock(
                        handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                except BlockingIOError as error:
                    handle.close()
                    raise ActivationOperatorError(
                        "cannot clear an activation abort while a supervisor "
                        "or inherited phase child still holds a lifetime lease"
                    ) from error
                handles.append(handle)
            return handles
        except Exception:
            for handle in handles:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()
            raise

    def status(self) -> dict[str, Any]:
        """Return control state plus non-network observations of sealed outputs."""

        state = self.load_state()
        abort = self.pending_abort()
        supervisors = self._observe_supervisors()
        observed = {
            "benchmark": self._observe_benchmark(),
            "canary": self._observe_campaign("train-canary.json", 2_000_000, canary=True),
            "train": self._observe_capture("train", 50_000_000),
            "eval": self._observe_capture("eval", 2_000_000),
        }
        if abort is not None:
            campaign_status = "abort-requested"
        elif state is None:
            campaign_status = "not-started"
        elif any(item.get("active") is True for item in supervisors):
            campaign_status = "running"
        elif any(item.get("state") == "failed" for item in supervisors):
            campaign_status = "failed"
        else:
            phase_statuses = [state["phases"][phase]["status"] for phase in ACTIVATION_PHASES]
            if phase_statuses[-1] == "passed":
                campaign_status = "complete"
            elif "failed" in phase_statuses:
                campaign_status = "failed"
            elif "running" in phase_statuses:
                campaign_status = "running"
            elif "aborted" in phase_statuses:
                campaign_status = "aborted"
            elif "passed" in phase_statuses:
                campaign_status = "partial"
            else:
                campaign_status = "ready"
        return {
            "schema_version": STATE_SCHEMA,
            "campaign_status": campaign_status,
            "state": state,
            "abort_request": abort,
            "observed": observed,
            "supervisors": supervisors,
            "harvest_controls": self._observe_harvest_controls(),
        }

    def _archive_abort_unlocked(self, request: Mapping[str, Any]) -> None:
        self.abort_history.mkdir(parents=True, exist_ok=True)
        history = self.abort_history / f"{request['abort_sha256']}.json"
        if history.exists():
            if _read_json(history) != dict(request):
                raise ActivationOperatorError("abort-history content collision")
        else:
            _atomic_json(history, request)

    def _finish_phase(
        self,
        phase: str,
        status: str,
        detail: Mapping[str, Any] | None,
        *,
        attempt_id: str | None,
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        with _exclusive_lock(self.lock_path):
            state = self._require_state_unlocked()
            record = state["phases"][phase]
            if record["status"] == "passed":
                return state
            if record["status"] != "running":
                return state
            current_attempt = record.get("attempt_id")
            if attempt_id is not None and attempt_id != current_attempt:
                raise ActivationOperatorError(
                    f"stale completion for activation phase {phase}"
                )
            record.update(
                status=status,
                completed_at=_utc_now(),
                detail=None if detail is None else dict(detail),
            )
            self._write_state_unlocked(state)
            return state

    def _load_state_unlocked(self) -> dict[str, Any]:
        state = _read_json(self.state_path)
        claimed = state.get("state_sha256")
        phases = state.get("phases")
        if (
            state.get("schema_version") != STATE_SCHEMA
            or not _RUN_ID_RE.fullmatch(str(state.get("run_id", "")))
            or not _SHA256_RE.fullmatch(str(state.get("spec_sha256", "")))
            or not isinstance(state.get("spec"), dict)
            or state.get("spec_sha256") != _sha256_mapping(state["spec"])
            or not isinstance(phases, dict)
            or set(phases) != set(ACTIVATION_PHASES)
            or not _SHA256_RE.fullmatch(str(claimed or ""))
            or claimed != _self_hash(state, "state_sha256")
        ):
            raise ActivationOperatorError("activation state failed schema/self-hash validation")
        allowed = {"pending", "running", "passed", "failed", "aborted"}
        for phase in ACTIVATION_PHASES:
            record = phases[phase]
            if (
                not isinstance(record, dict)
                or record.get("status") not in allowed
                or isinstance(record.get("attempts"), bool)
                or not isinstance(record.get("attempts"), int)
                or record["attempts"] < 0
                or (
                    record.get("status") == "pending"
                    and record.get("attempt_id") is not None
                )
                or (
                    record.get("status") in {"running", "passed", "failed"}
                    and not _RUN_ID_RE.fullmatch(
                        str(record.get("attempt_id", ""))
                    )
                )
                or (
                    record.get("status") == "aborted"
                    and record.get("attempt_id") is not None
                    and not _RUN_ID_RE.fullmatch(str(record.get("attempt_id")))
                )
            ):
                raise ActivationOperatorError(f"activation phase record is invalid: {phase}")
        return state

    def _load_abort_unlocked(self) -> dict[str, Any]:
        request = _read_json(self.abort_path)
        claimed = request.get("abort_sha256")
        if (
            request.get("schema_version") != ABORT_SCHEMA
            or not _RUN_ID_RE.fullmatch(str(request.get("run_id", "")))
            or not _RUN_ID_RE.fullmatch(str(request.get("request_id", "")))
            or not _SHA256_RE.fullmatch(str(request.get("spec_sha256", "")))
            or request.get("semantics") != "cooperative-next-phase-boundary"
            or not _SHA256_RE.fullmatch(str(claimed or ""))
            or claimed != _self_hash(request, "abort_sha256")
        ):
            raise ActivationOperatorError("activation abort request failed validation")
        return request

    def _require_state_unlocked(self) -> dict[str, Any]:
        if not self.state_path.exists():
            raise ActivationOperatorError("no activation campaign exists at this artifact root")
        return self._load_state_unlocked()

    def _write_state_unlocked(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _utc_now()
        state["state_sha256"] = _self_hash(state, "state_sha256")
        _atomic_json(self.state_path, state)

    @staticmethod
    def _validate_phase(phase: str) -> None:
        if phase not in ACTIVATION_PHASES:
            raise ValueError(f"unknown activation phase {phase!r}")

    def _observe_harvest_controls(self) -> list[dict[str, Any]]:
        """Summarize live/resumable per-host capture control without JAX."""

        from moe_sae.harvest_control import load_campaign_control

        observed: list[dict[str, Any]] = []
        for split in ("train", "eval"):
            root = (
                self.artifact_root
                / "captures"
                / split
                / "telemetry"
                / "campaign-control"
            )
            if not root.exists():
                continue
            for directory in sorted(root.iterdir()):
                if not directory.is_dir() or not _SHA256_RE.fullmatch(directory.name):
                    continue
                try:
                    control = load_campaign_control(
                        self.artifact_root / "captures" / split,
                        directory.name,
                    )
                    hosts = control["hosts"]
                    stop_after_tokens = (
                        hosts[0].get("stop_after_tokens") if hosts else None
                    )
                    total_tokens = hosts[0].get("total_tokens") if hosts else None
                    phase = (
                        "canary"
                        if split == "train"
                        and isinstance(stop_after_tokens, int)
                        and isinstance(total_tokens, int)
                        and stop_after_tokens < total_tokens
                        else split
                    )
                    observed.append(
                        {
                            "valid": True,
                            "campaign_id": directory.name,
                            "phase": phase,
                            "split": split,
                            "stop_after_tokens": stop_after_tokens,
                            "total_tokens": total_tokens,
                            "abort_request": control["abort"],
                            "hosts": [
                                {
                                    "process_index": host.get("process_index"),
                                    "state": host.get("state"),
                                    "heartbeat_unix": host.get("heartbeat_unix"),
                                    "progress": host.get("progress"),
                                }
                                for host in hosts
                            ],
                        }
                    )
                except Exception as error:  # status surfaces corruption verbatim
                    observed.append(
                        {
                            "valid": False,
                            "campaign_id": directory.name,
                            "split": split,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
        return observed

    def _observe_supervisors(self) -> list[dict[str, Any]]:
        observed: list[dict[str, Any]] = []
        now = time.time()
        for path in sorted((self.root / "leases").glob("controller-*.json")):
            try:
                value = _read_json(path)
                claimed = value.get("status_sha256")
                if (
                    value.get("schema_version") != SUPERVISOR_SCHEMA
                    or claimed != _self_hash(value, "status_sha256")
                ):
                    raise ActivationOperatorError("invalid supervisor self-hash")
                lane_id = str(value.get("lane_id", ""))
                lock_path = self.root / "leases" / f"controller-{lane_id}.lock"
                marker_path = (
                    self.root
                    / "leases"
                    / "markers"
                    / f"controller-{lane_id}.json"
                )
                marker = _read_json(marker_path)
                marker_sha256 = marker.get("lease_sha256")
                marker_matches = bool(
                    marker.get("schema_version") == SUPERVISOR_SCHEMA
                    and marker.get("run_id") == value.get("run_id")
                    and marker.get("operation_sha256")
                    == value.get("operation_sha256")
                    and marker.get("lane_id") == lane_id
                    and marker.get("lease_id") == value.get("lease_id")
                    and marker_sha256 == _self_hash(marker, "lease_sha256")
                )
                handle = lock_path.open("a+b", buffering=0)
                try:
                    try:
                        fcntl.flock(
                            handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                    except BlockingIOError:
                        lock_held = True
                    else:
                        lock_held = False
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()
                heartbeat_age = now - float(value["heartbeat_unix"])
                observed.append(
                    {
                        **value,
                        "status_path": str(path),
                        "lock_held": lock_held,
                        "lease_marker_matches": marker_matches,
                        "heartbeat_age_seconds": heartbeat_age,
                        "active": bool(
                            lock_held
                            and marker_matches
                            and 0.0 <= heartbeat_age <= SUPERVISOR_FRESH_SECONDS
                        ),
                    }
                )
            except Exception as error:
                observed.append(
                    {
                        "status_path": str(path),
                        "active": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        return observed

    def _observe_benchmark(self) -> dict[str, Any]:
        path = self.artifact_root / "benchmark" / "selected.json"
        if not path.exists():
            return {"present": False, "valid": False, "path": str(path)}
        try:
            payload = _read_json(path)
            selected = payload.get("selected")
            valid = isinstance(selected, dict) and selected.get("passes_hard_gate") is True
            return {
                "present": True,
                "valid": valid,
                "path": str(path),
                "mesh": selected.get("mesh") if isinstance(selected, dict) else None,
                "per_device_sequence_batch": (
                    selected.get("per_device_sequence_batch")
                    if isinstance(selected, dict)
                    else None
                ),
                "useful_mfu": selected.get("useful_mfu") if isinstance(selected, dict) else None,
                "persisted_tokens_per_second": (
                    selected.get("persisted_tokens_per_second")
                    if isinstance(selected, dict)
                    else None
                ),
            }
        except ActivationOperatorError as error:
            return {"present": True, "valid": False, "path": str(path), "error": str(error)}

    def _observe_campaign(self, name: str, tokens: int, *, canary: bool) -> dict[str, Any]:
        path = self.artifact_root / "harvest" / name
        if not path.exists():
            return {"present": False, "valid": False, "path": str(path)}
        try:
            payload = _read_json(path)
            valid = (
                payload.get("schema_version") == "gemma4-harvest-campaign-v2"
                and payload.get("captured_tokens") == tokens
                and (not canary or payload.get("canary_passed") is True)
            )
            return {
                "present": True,
                "valid": valid,
                "path": str(path),
                "captured_tokens": payload.get("captured_tokens"),
                "canary_passed": payload.get("canary_passed"),
                "real_goodput": payload.get("real_goodput"),
                "input_wait_fraction": payload.get("input_wait_fraction"),
                "max_writer_backlog": payload.get("max_writer_backlog"),
            }
        except ActivationOperatorError as error:
            return {"present": True, "valid": False, "path": str(path), "error": str(error)}

    def _observe_capture(self, split: str, tokens: int) -> dict[str, Any]:
        campaign = self._observe_campaign(f"{split}.json", tokens, canary=False)
        manifest_path = self.artifact_root / "captures" / split / "manifest.json"
        manifest_present = manifest_path.exists()
        manifest_valid = False
        manifest_sha256 = None
        error = None
        if manifest_present:
            try:
                from moe_sae.artifacts import load_manifest

                manifest = load_manifest(manifest_path.parent)
                manifest_valid = (
                    manifest.get("sealed") is True
                    and manifest.get("total_tokens") == tokens
                    and manifest.get("storage", {}).get("kind") == "gcs-immutable"
                )
                manifest_sha256 = manifest.get("manifest_sha256")
            except Exception as caught:  # status reports corruption; it does not hide it
                error = f"{type(caught).__name__}: {caught}"
        return {
            "present": bool(campaign["present"] or manifest_present),
            "valid": bool(campaign["valid"] and manifest_valid),
            "campaign": campaign,
            "manifest_path": str(manifest_path),
            "manifest_present": manifest_present,
            "manifest_valid": manifest_valid,
            "manifest_sha256": manifest_sha256,
            "error": error,
        }


__all__ = [
    "ACTIVATION_PHASES",
    "ActivationAbortRequested",
    "ActivationControl",
    "ActivationOperatorError",
    "ActivationRunSpec",
    "ActivationSupervisorLease",
    "ActivationSupervisorLeaseHeld",
]
