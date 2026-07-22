"""JAX-free four-controller supervisor for the fixed SAE phase plan.

Numerical execution and artifact interpretation are deliberately injected.  A
farm supervisor owns one durable lane for its entire lifetime, starts one fresh
child per non-passed phase, and lets :mod:`moe_sae.sae_operator` provide the
cross-host cohort and commit protocol.

The child-termination rule is intentionally strict.  A live local child is
never terminated merely because this supervisor noticed a peer failure.  The
supervisor first publishes its own terminal acknowledgement and waits for
``finalize_attempt`` to commit the four-lane failed/aborted release.  Only that
durable collective decision authorizes the injected executor to terminate and
reap its local child.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Literal, Protocol

from moe_sae.sae_operator import (
    CONTROL_POLL_SECONDS,
    EXPECTED_LANES,
    HEARTBEAT_INTERVAL_SECONDS,
    RENDEZVOUS_TIMEOUT_SECONDS,
    SAEAttemptOutcome,
    SAEControl,
    SAELaneLease,
    SAEPeerFailure,
    SAEPhaseAlreadyPassed,
    SAEPhaseAttempt,
    SAE_PHASES,
    canonical_sha256,
)
from moe_sae.sae_paths import SAEOperationPaths


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_LAUNCH_SCHEMA = "gemma4-sae-supervisor-launch-v1"
_COMPLETION_SCHEMA = "gemma4-sae-supervisor-completion-v1"
_SKIP_SCHEMA = "gemma4-sae-supervisor-skip-v1"


class SAEFarmError(RuntimeError):
    """The supervised SAE campaign could not safely continue."""


class SAEArtifactValidationError(SAEFarmError):
    """A phase artifact did not revalidate to a lowercase SHA-256."""


class SAEFarmPhaseFailed(SAEFarmError):
    """A collectively committed phase attempt failed or was aborted."""


@dataclass(frozen=True, slots=True)
class SAEChildExit:
    """Terminal status returned by a fresh numerical child handle."""

    outcome: Literal["passed", "failed", "aborted"]
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in {"passed", "failed", "aborted"}:
            raise ValueError(f"invalid SAE child outcome: {self.outcome!r}")
        if not isinstance(self.detail, Mapping):
            raise TypeError("SAE child detail must be a mapping")


@dataclass(frozen=True, slots=True)
class SAEPhaseInvocation:
    """Exact child launch identity supplied to the injected executor."""

    phase: str
    attempt_id: str
    attempt_number: int
    operation_sha256: str
    lane_id: str
    lease_id: str
    lease_fd: int
    paths: SAEOperationPaths

    @property
    def pass_fds(self) -> tuple[int, ...]:
        """Descriptors that the subprocess adapter must preserve."""

        return (self.lease_fd,)


class SAEChildHandle(Protocol):
    """One fresh child process owned by a single controller supervisor."""

    def poll(self) -> SAEChildExit | None:
        """Return ``None`` while live, otherwise the reaped terminal result."""

    def terminate_after_collective_failure(self, reason: str) -> None:
        """Terminate and reap after the farm has committed collective failure."""


class SAEFreshChildExecutor(Protocol):
    """Non-blocking factory that starts one numerical child per invocation."""

    def __call__(self, invocation: SAEPhaseInvocation) -> SAEChildHandle:
        """Start promptly, preserve ``pass_fds``, and return the live handle."""


class SAEArtifactRevalidator(Protocol):
    """Recompute the immutable result digest for one completed phase."""

    def __call__(self, phase: str, paths: SAEOperationPaths) -> str:
        """Return the exact lowercase SHA-256 for the phase's final artifact."""


@dataclass(frozen=True, slots=True)
class SAEFarmResult:
    """Small integration result returned by :func:`run_farm`."""

    operation_sha256: str
    status: Literal["complete", "aborted"]
    executed_phases: tuple[str, ...]
    skipped_phases: tuple[str, ...]
    boundary_phase: str | None
    abort_sha256: str | None


def _detail(error: BaseException, **extra: Any) -> dict[str, Any]:
    return {
        "error_type": type(error).__name__,
        "error": str(error)[:2_000],
        **extra,
    }


def _artifact_digest(
    revalidator: SAEArtifactRevalidator,
    phase: str,
    paths: SAEOperationPaths,
) -> str:
    try:
        digest = revalidator(phase, paths)
    except BaseException as error:
        raise SAEArtifactValidationError(
            f"artifact revalidation failed for {phase}: {error}"
        ) from error
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise SAEArtifactValidationError(
            f"artifact revalidator returned an invalid SHA-256 for {phase}"
        )
    return digest


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a create-once object without replacing possible tampering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        else:
            _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _launch_payload(
    control: SAEControl, attempt: SAEPhaseAttempt
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": _LAUNCH_SCHEMA,
        "operation_sha256": control.operation_sha256,
        "phase": attempt.phase,
        "attempt_id": attempt.attempt_id,
        "cohort_sha256": attempt.cohort_sha256,
        "members": list(attempt.members),
    }
    payload["release_sha256"] = canonical_sha256(payload)
    return payload


def _validate_supervisor_release(
    path: Path, expected: Mapping[str, Any]
) -> bool:
    if not path.exists():
        return False
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SAEFarmError(f"cannot read supervisor release: {path}") from error
    if observed != dict(expected):
        raise SAEFarmError(f"supervisor release has unexpected content: {path}")
    return True


def _rendezvous_numerical_launch(
    control: SAEControl,
    lease: SAELaneLease,
    attempt: SAEPhaseAttempt,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> None:
    """Prove all four begin calls returned before any child can finish.

    The operator's start release lets a fast lane leave ``begin_phase`` while
    a peer is still inside its rendezvous.  This second, durable acknowledgement
    barrier prevents a very short child from publishing a terminal heartbeat
    early enough to strand that peer in the first rendezvous.
    """

    release_path = (
        control.root
        / "attempts"
        / attempt.attempt_id
        / "supervisor-launch.json"
    )
    expected = _launch_payload(control, attempt)
    ready_detail = {
        "supervisor_stage": "launch-ready",
        "cohort_sha256": attempt.cohort_sha256,
    }
    deadline = time.monotonic() + timeout_seconds
    while True:
        if _validate_supervisor_release(release_path, expected):
            return
        control.heartbeat(
            lease,
            "running",
            phase=attempt.phase,
            attempt_id=attempt.attempt_id,
            detail=ready_detail,
        )
        try:
            records = control.require_attempt_peers_healthy(lease, attempt)
        except SAEPeerFailure:
            # Another lane may finish a very short child after publishing the
            # immutable launch release but before this lane's next loop-top
            # read.  The release proves all four supervisors acknowledged the
            # launch cohort; do not relabel that authorized launch as a
            # pre-launch failure merely because its peer is already terminal.
            if _validate_supervisor_release(release_path, expected):
                return
            raise
        all_ready = all(
            item["state"] == "running"
            and item.get("detail") == ready_detail
            for item in records
        )
        if len(records) == EXPECTED_LANES and all_ready:
            _create_immutable_json(release_path, expected)
            if _validate_supervisor_release(release_path, expected):
                return
        if time.monotonic() >= deadline:
            raise SAEFarmError(
                f"numerical launch rendezvous timed out for {attempt.phase}/"
                f"{attempt.attempt_id}"
            )
        time.sleep(poll_seconds)


def _completion_payload(
    control: SAEControl,
    attempt: SAEPhaseAttempt,
    phase_record: Mapping[str, Any],
) -> dict[str, Any]:
    result = phase_record.get("result")
    if not isinstance(result, Mapping):
        raise SAEFarmError("committed phase has no completion result")
    completion_sha256 = result.get("completion_release_sha256")
    if (
        phase_record.get("attempt_id") != attempt.attempt_id
        or phase_record.get("status") not in {"passed", "failed", "aborted"}
        or not isinstance(completion_sha256, str)
        or _SHA256_RE.fullmatch(completion_sha256) is None
    ):
        raise SAEFarmError("phase does not contain this attempt's completion release")
    payload: dict[str, Any] = {
        "schema_version": _COMPLETION_SCHEMA,
        "operation_sha256": control.operation_sha256,
        "phase": attempt.phase,
        "attempt_id": attempt.attempt_id,
        "cohort_sha256": attempt.cohort_sha256,
        "members": list(attempt.members),
        "outcome": phase_record["status"],
        "result_sha256": result.get("result_sha256"),
        "operator_completion_release_sha256": completion_sha256,
    }
    payload["release_sha256"] = canonical_sha256(payload)
    return payload


def _rendezvous_completion_observed(
    control: SAEControl,
    lease: SAELaneLease,
    attempt: SAEPhaseAttempt,
    *,
    local_outcome: Literal["passed", "failed", "aborted"],
    local_result_sha256: str | None,
    local_detail: Mapping[str, Any] | None,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    """Keep terminal state stable until every supervisor observes the commit."""

    phase_record = control.load_state()["phases"][attempt.phase]
    expected = _completion_payload(control, attempt, phase_record)
    operator_release = str(expected["operator_completion_release_sha256"])
    release_path = (
        control.root
        / "attempts"
        / attempt.attempt_id
        / "supervisor-complete.json"
    )
    ready_detail = {
        "supervisor_stage": "completion-observed",
        "completion_release_sha256": operator_release,
        "terminal_stage": (
            local_detail.get("stage")
            if isinstance(local_detail, Mapping)
            else None
        ),
        "peer_observation": (
            local_detail.get("peer_observation")
            if isinstance(local_detail, Mapping)
            else None
        ),
    }
    lane_state = "succeeded" if local_outcome == "passed" else local_outcome
    expected_members = {
        (str(member["lane_id"]), str(member["lease_id"]))
        for member in attempt.members
    }
    deadline = time.monotonic() + timeout_seconds
    while True:
        if _validate_supervisor_release(release_path, expected):
            return expected
        control.heartbeat(
            lease,
            lane_state,
            phase=attempt.phase,
            attempt_id=attempt.attempt_id,
            result_sha256=local_result_sha256,
            detail=ready_detail,
        )
        status = control.status()
        if status["orphaned_lanes"]:
            raise SAEFarmError(
                f"completion cohort is orphaned for {attempt.phase}"
            )
        held = [item for item in status["lanes"] if item["lease_held"]]
        held_members = {
            (str(item["lane_id"]), str(item["lease_id"])) for item in held
        }
        if held_members - expected_members:
            raise SAEFarmError("completion cohort contains an unexpected lane")
        matching = [
            item
            for item in held
            if (str(item["lane_id"]), str(item["lease_id"]))
            in expected_members
            and item.get("phase") == attempt.phase
            and item.get("attempt_id") == attempt.attempt_id
        ]
        all_observed = all(
            item["state"] in {"succeeded", "failed", "aborted"}
            and isinstance(item.get("detail"), Mapping)
            and item["detail"].get("supervisor_stage")
            == "completion-observed"
            and item["detail"].get("completion_release_sha256")
            == operator_release
            for item in matching
        )
        if len(matching) == EXPECTED_LANES and all_observed:
            _create_immutable_json(release_path, expected)
            if _validate_supervisor_release(release_path, expected):
                return expected
        if time.monotonic() >= deadline:
            raise SAEFarmError(
                f"completion-observed rendezvous timed out for {attempt.phase}/"
                f"{attempt.attempt_id}"
            )
        time.sleep(poll_seconds)


@contextmanager
def _terminal_heartbeats(
    control: SAEControl,
    lease: SAELaneLease,
    attempt: SAEPhaseAttempt,
    *,
    outcome: Literal["passed", "failed", "aborted"],
    result_sha256: str | None,
    detail: Mapping[str, Any] | None,
    interval_seconds: float,
):
    """Keep a terminal acknowledgement fresh while peers finish validation."""

    state = "succeeded" if outcome == "passed" else outcome
    stopped = threading.Event()

    def publish() -> None:
        while not stopped.wait(interval_seconds):
            try:
                control.heartbeat(
                    lease,
                    state,
                    phase=attempt.phase,
                    attempt_id=attempt.attempt_id,
                    result_sha256=result_sha256,
                    detail=detail,
                    force=True,
                )
            except BaseException:
                # The synchronous finalize path will observe the same corrupt,
                # vanished, or unavailable control plane and fail closed.  A
                # daemon heartbeat must never mask that primary exception.
                return

    thread = threading.Thread(
        target=publish,
        name=f"sae-terminal-heartbeat-{lease.lane_id}",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=max(1.0, interval_seconds * 2.0))


@contextmanager
def _running_heartbeats(
    control: SAEControl,
    lease: SAELaneLease,
    attempt: SAEPhaseAttempt,
    *,
    interval_seconds: float,
):
    """Keep a launched child or blocking artifact validation visibly live."""

    stopped = threading.Event()

    def publish() -> None:
        while not stopped.wait(interval_seconds):
            try:
                control.heartbeat(
                    lease,
                    "running",
                    phase=attempt.phase,
                    attempt_id=attempt.attempt_id,
                    force=True,
                )
            except BaseException:
                # Foreground watchdog/finalization operations fail closed.  Do
                # not let an auxiliary daemon hide their more useful error.
                return

    control.heartbeat(
        lease,
        "running",
        phase=attempt.phase,
        attempt_id=attempt.attempt_id,
        force=True,
    )
    thread = threading.Thread(
        target=publish,
        name=f"sae-running-heartbeat-{lease.lane_id}",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=max(1.0, interval_seconds * 2.0))


@contextmanager
def _idle_heartbeats(
    control: SAEControl,
    lease: SAELaneLease,
    *,
    interval_seconds: float,
):
    """Keep a restart lane fresh while a passed artifact is revalidated."""

    stopped = threading.Event()

    def publish() -> None:
        while not stopped.wait(interval_seconds):
            try:
                control.heartbeat(lease, "starting", force=True)
            except BaseException:
                return

    control.heartbeat(lease, "starting", force=True)
    thread = threading.Thread(
        target=publish,
        name=f"sae-idle-heartbeat-{lease.lane_id}",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=max(1.0, interval_seconds * 2.0))


def _peer_terminal_outcome(
    control: SAEControl,
    attempt: SAEPhaseAttempt,
    *,
    observation_timeout_seconds: float,
    poll_seconds: float,
) -> tuple[Literal["failed", "aborted"], dict[str, Any]]:
    """Classify a peer event from a stable exact-cohort terminal observation.

    The watchdog and this classifier necessarily read separate atomic status
    snapshots.  A lane may be between heartbeat replacements in that interval,
    so an empty second snapshot is not evidence of numerical failure.  Retry
    briefly; an explicit failure remains dominant, while an exact-cohort abort
    can then be mirrored without nondeterministically converting it to failure.
    """

    expected = {
        (str(member["lane_id"]), str(member["lease_id"]))
        for member in attempt.members
    }
    deadline = time.monotonic() + observation_timeout_seconds
    while True:
        lanes = control.status()["lanes"]
        matching = [
            lane
            for lane in lanes
            if (str(lane.get("lane_id")), str(lane.get("lease_id"))) in expected
            and lane.get("phase") == attempt.phase
            and lane.get("attempt_id") == attempt.attempt_id
        ]
        observation = {
            "matching": [
                {
                    "lane_id": str(lane.get("lane_id")),
                    "state": str(lane.get("state")),
                    "lease_held": bool(lane.get("lease_held")),
                }
                for lane in matching
            ]
        }
        if any(lane.get("state") == "failed" for lane in matching):
            return "failed", {"decision": "explicit-failure", **observation}
        if any(lane.get("state") == "aborted" for lane in matching):
            return "aborted", {"decision": "explicit-abort", **observation}
        if time.monotonic() >= deadline:
            return "failed", {"decision": "observation-timeout", **observation}
        time.sleep(poll_seconds)


def _collective_finish(
    control: SAEControl,
    lease: SAELaneLease,
    attempt: SAEPhaseAttempt,
    *,
    outcome: Literal["passed", "failed", "aborted"],
    result_sha256: str | None,
    detail: Mapping[str, Any] | None,
    child: SAEChildHandle | None,
    child_is_live: bool,
    completion_timeout_seconds: float,
    heartbeat_interval_seconds: float,
    poll_seconds: float,
) -> SAEAttemptOutcome:
    """Publish locally, commit four-lane verdict, then optionally terminate."""

    control.publish_attempt_result(
        lease,
        attempt,
        outcome=outcome,
        result_sha256=result_sha256,
        detail=detail,
    )
    finalized: SAEAttemptOutcome | None = None
    finalize_error: BaseException | None = None
    with _terminal_heartbeats(
        control,
        lease,
        attempt,
        outcome=outcome,
        result_sha256=result_sha256,
        detail=detail,
        interval_seconds=heartbeat_interval_seconds,
    ):
        try:
            finalized = control.finalize_attempt(
                attempt,
                timeout_seconds=completion_timeout_seconds,
                poll_seconds=poll_seconds,
            )
        except BaseException as error:
            finalize_error = error

    # Keep the terminal heartbeat in place until all four supervisors have
    # durably acknowledged the operator commit.  This both prevents a fast
    # lane from stranding a peer in ``finalize_attempt`` and is the sole
    # authorization to terminate a still-live numerical child.
    completion_error: BaseException | None = None
    collectively_observed = False
    state = control.load_state()["phases"][attempt.phase]
    if (
        state.get("attempt_id") == attempt.attempt_id
        and state.get("status") in {"passed", "failed", "aborted"}
        and isinstance(state.get("result"), Mapping)
        and state["result"].get("completion_release_sha256") is not None
    ):
        try:
            completion = _rendezvous_completion_observed(
                control,
                lease,
                attempt,
                local_outcome=outcome,
                local_result_sha256=result_sha256,
                local_detail=detail,
                timeout_seconds=completion_timeout_seconds,
                poll_seconds=poll_seconds,
            )
        except BaseException as error:
            completion_error = error
        else:
            collectively_observed = completion["outcome"] in {
                "failed",
                "aborted",
            }
    if child_is_live and child is not None and collectively_observed:
        child.terminate_after_collective_failure(
            f"collective {state['status']} decision for {attempt.phase}/"
            f"{attempt.attempt_id}"
        )
    if completion_error is not None:
        raise SAEFarmPhaseFailed(
            f"phase {attempt.phase} completion was not observed by all supervisors"
        ) from completion_error
    if finalize_error is not None:
        raise SAEFarmPhaseFailed(
            f"phase {attempt.phase} could not commit a passing result"
        ) from finalize_error
    assert finalized is not None
    return finalized


def _run_phase(
    control: SAEControl,
    lease: SAELaneLease,
    attempt: SAEPhaseAttempt,
    paths: SAEOperationPaths,
    executor: SAEFreshChildExecutor,
    revalidator: SAEArtifactRevalidator,
    *,
    poll_seconds: float,
    launch_timeout_seconds: float,
    completion_timeout_seconds: float,
    heartbeat_interval_seconds: float,
) -> SAEAttemptOutcome:
    invocation = SAEPhaseInvocation(
        phase=attempt.phase,
        attempt_id=attempt.attempt_id,
        attempt_number=attempt.attempt_number,
        operation_sha256=control.operation_sha256,
        lane_id=lease.lane_id,
        lease_id=lease.lease_id,
        lease_fd=lease.descriptor,
        paths=paths,
    )
    child: SAEChildHandle | None = None
    outcome: Literal["passed", "failed", "aborted"] = "failed"
    result_sha256: str | None = None
    detail: Mapping[str, Any] | None = None
    child_is_live = False

    try:
        _rendezvous_numerical_launch(
            control,
            lease,
            attempt,
            timeout_seconds=launch_timeout_seconds,
            poll_seconds=poll_seconds,
        )
    except BaseException as error:
        return _collective_finish(
            control,
            lease,
            attempt,
            outcome="failed",
            result_sha256=None,
            detail=_detail(error, stage="numerical-launch-rendezvous"),
            child=None,
            child_is_live=False,
            completion_timeout_seconds=completion_timeout_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            poll_seconds=poll_seconds,
        )

    with _running_heartbeats(
        control,
        lease,
        attempt,
        interval_seconds=heartbeat_interval_seconds,
    ):
        try:
            child = executor(invocation)
        except BaseException as error:
            detail = _detail(error, stage="child-start")
        else:
            while True:
                try:
                    control.require_attempt_peers_healthy(lease, attempt)
                except SAEPeerFailure as error:
                    outcome, peer_observation = _peer_terminal_outcome(
                        control,
                        attempt,
                        observation_timeout_seconds=2.0 * poll_seconds,
                        poll_seconds=poll_seconds,
                    )
                    detail = _detail(
                        error,
                        stage="peer-watchdog",
                        peer_observation=peer_observation,
                    )
                    child_is_live = True
                    break

                try:
                    child_exit = child.poll()
                except BaseException as error:
                    detail = _detail(error, stage="child-poll")
                    child_is_live = True
                    break
                if child_exit is None:
                    time.sleep(poll_seconds)
                    continue
                if not isinstance(child_exit, SAEChildExit):
                    error = TypeError(
                        "child poll returned an invalid terminal result"
                    )
                    detail = _detail(error, stage="child-poll")
                    child_is_live = True
                    break
                if child_exit.outcome != "passed":
                    outcome = child_exit.outcome
                    detail = dict(child_exit.detail)
                    break
                try:
                    result_sha256 = _artifact_digest(
                        revalidator, attempt.phase, paths
                    )
                except SAEArtifactValidationError as error:
                    detail = _detail(error, stage="artifact-revalidation")
                    break
                outcome = "passed"
                detail = dict(child_exit.detail)
                break

    return _collective_finish(
        control,
        lease,
        attempt,
        outcome=outcome,
        result_sha256=result_sha256,
        detail=detail,
        child=child,
        child_is_live=child_is_live,
        completion_timeout_seconds=completion_timeout_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        poll_seconds=poll_seconds,
    )


def _boundary_result(
    control: SAEControl,
    phase: str | None,
    *,
    executed: list[str],
    skipped: list[str],
) -> SAEFarmResult | None:
    request = control.pending_abort()
    if request is None:
        return None
    control.acknowledge_abort(phase)
    return SAEFarmResult(
        operation_sha256=control.operation_sha256,
        status="aborted",
        executed_phases=tuple(executed),
        skipped_phases=tuple(skipped),
        boundary_phase=phase,
        abort_sha256=str(request["abort_sha256"]),
    )


def _rendezvous_phase_boundary(
    control: SAEControl,
    lease: SAELaneLease,
    phase: str,
    *,
    timeout_seconds: float,
    expected_members: tuple[dict[str, str], ...] | None,
    poll_seconds: float,
) -> tuple[dict[str, str], ...]:
    """Move all four lanes off the prior terminal state before phase begin.

    ``SAEControl.begin_phase`` accepts peer lanes in ``starting`` while the
    first lane crosses into ``ready-to-begin``.  Waiting for this compatible
    set prevents a fast supervisor from treating slower lanes' terminal
    heartbeat from the previous phase as a split cohort.
    """

    deadline = time.monotonic() + timeout_seconds
    while True:
        control.heartbeat(lease, "starting")
        status = control.status()
        if status["orphaned_lanes"]:
            raise SAEFarmError(
                f"phase-boundary cohort is unhealthy before {phase}"
            )
        held = [item for item in status["lanes"] if item["lease_held"]]
        compatible = [
            item
            for item in held
            if item["state"] == "starting"
            or (
                item["state"] == "ready-to-begin"
                and item["phase"] == phase
                and item["attempt_id"] is None
            )
        ]
        if len(held) == EXPECTED_LANES and len(compatible) == EXPECTED_LANES:
            if status["vanished_lanes"]:
                raise SAEFarmError(
                    f"phase boundary has vanished lanes before {phase}"
                )
            members = tuple(
                sorted(
                    (
                        {
                            "lane_id": str(item["lane_id"]),
                            "lease_id": str(item["lease_id"]),
                            "hostname": str(item["hostname"]),
                        }
                        for item in held
                    ),
                    key=lambda item: item["lane_id"],
                )
            )
            if expected_members is not None and members != expected_members:
                raise SAEFarmError("lifetime supervisor cohort changed at a boundary")
            return members
        if len(held) > EXPECTED_LANES:
            raise SAEFarmError(
                f"phase boundary has more than {EXPECTED_LANES} held lanes"
            )
        if time.monotonic() >= deadline:
            raise SAEFarmError(
                f"phase-boundary rendezvous timed out with "
                f"{len(compatible)}/{EXPECTED_LANES} compatible lanes for {phase}"
            )
        time.sleep(poll_seconds)


def _skip_payload(
    control: SAEControl,
    phase: str,
    result_sha256: str,
    members: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": _SKIP_SCHEMA,
        "operation_sha256": control.operation_sha256,
        "phase": phase,
        "result_sha256": result_sha256,
        "members": list(members),
    }
    payload["release_sha256"] = canonical_sha256(payload)
    return payload


def _rendezvous_skip_observed(
    control: SAEControl,
    lease: SAELaneLease,
    phase: str,
    result_sha256: str,
    members: tuple[dict[str, str], ...],
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> None:
    """Keep a resume cohort together after exact passed-artifact validation."""

    cohort_sha256 = canonical_sha256(list(members))
    expected = _skip_payload(control, phase, result_sha256, members)
    release_path = (
        control.root
        / "supervisor-sessions"
        / cohort_sha256
        / "skips"
        / f"{phase}.json"
    )
    ready_detail = {
        "supervisor_stage": "skip-observed",
        "phase": phase,
        "result_sha256": result_sha256,
        "cohort_sha256": cohort_sha256,
    }
    expected_keys = {
        (str(member["lane_id"]), str(member["lease_id"])) for member in members
    }
    deadline = time.monotonic() + timeout_seconds
    while True:
        if _validate_supervisor_release(release_path, expected):
            return
        control.heartbeat(lease, "starting", detail=ready_detail)
        status = control.status()
        if status["orphaned_lanes"]:
            raise SAEFarmError(f"skip cohort is orphaned at {phase}")
        held = [item for item in status["lanes"] if item["lease_held"]]
        held_keys = {
            (str(item["lane_id"]), str(item["lease_id"])) for item in held
        }
        if held_keys - expected_keys:
            raise SAEFarmError("skip cohort contains an unexpected lane")
        matching = [
            item
            for item in held
            if (str(item["lane_id"]), str(item["lease_id"])) in expected_keys
        ]
        all_observed = all(
            item["state"] == "starting" and item.get("detail") == ready_detail
            for item in matching
        )
        if len(matching) == EXPECTED_LANES and all_observed:
            _create_immutable_json(release_path, expected)
            if _validate_supervisor_release(release_path, expected):
                return
        if time.monotonic() >= deadline:
            raise SAEFarmError(
                f"skip-observed rendezvous timed out for {phase}"
            )
        time.sleep(poll_seconds)


def _validate_timing(name: str, value: float, *, allow_zero: bool = False) -> float:
    minimum_ok = value >= 0.0 if allow_zero else value > 0.0
    if isinstance(value, bool) or not math.isfinite(value) or not minimum_ok:
        relation = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}")
    return float(value)


def run_farm(
    control: SAEControl,
    paths: SAEOperationPaths,
    executor: SAEFreshChildExecutor,
    revalidator: SAEArtifactRevalidator,
    *,
    lane: str | None = None,
    poll_seconds: float = CONTROL_POLL_SECONDS,
    rendezvous_timeout_seconds: float = RENDEZVOUS_TIMEOUT_SECONDS,
    completion_timeout_seconds: float = RENDEZVOUS_TIMEOUT_SECONDS,
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
) -> SAEFarmResult:
    """Run or exactly resume the fixed SAE campaign on one controller.

    Four copies of this function run concurrently, one per TPU controller.  The
    supplied ``control`` must already be initialized with ``SAEControl.begin``.
    Status and abort APIs intentionally remain methods on that JAX-free object.
    """

    if not isinstance(control, SAEControl):
        raise TypeError("control must be an SAEControl")
    if not isinstance(paths, SAEOperationPaths):
        raise TypeError("paths must be an SAEOperationPaths")
    if control.operation_sha256 != paths.operation_sha256:
        raise SAEFarmError("control and paths name different SAE operations")
    if control.artifact_root != paths.artifact_root:
        raise SAEFarmError("control and paths use different artifact roots")
    poll_seconds = _validate_timing("poll_seconds", poll_seconds)
    rendezvous_timeout_seconds = _validate_timing(
        "rendezvous_timeout_seconds", rendezvous_timeout_seconds
    )
    completion_timeout_seconds = _validate_timing(
        "completion_timeout_seconds", completion_timeout_seconds
    )
    heartbeat_interval_seconds = _validate_timing(
        "heartbeat_interval_seconds", heartbeat_interval_seconds
    )

    executed: list[str] = []
    skipped: list[str] = []
    session_members: tuple[dict[str, str], ...] | None = None
    with control.lane_lease(lane) as lease:
        for phase in SAE_PHASES:
            boundary = _boundary_result(
                control, phase, executed=executed, skipped=skipped
            )
            if boundary is not None:
                return boundary
            session_members = _rendezvous_phase_boundary(
                control,
                lease,
                phase,
                timeout_seconds=rendezvous_timeout_seconds,
                expected_members=session_members,
                poll_seconds=poll_seconds,
            )

            while True:
                record = control.load_state()["phases"][phase]
                if record["status"] == "passed":
                    with _idle_heartbeats(
                        control,
                        lease,
                        interval_seconds=heartbeat_interval_seconds,
                    ):
                        digest = _artifact_digest(revalidator, phase, paths)
                    control.validate_passed_phase(phase, digest)
                    _rendezvous_skip_observed(
                        control,
                        lease,
                        phase,
                        digest,
                        session_members,
                        timeout_seconds=rendezvous_timeout_seconds,
                        poll_seconds=poll_seconds,
                    )
                    skipped.append(phase)
                    break
                try:
                    attempt = control.begin_phase(
                        lease,
                        phase,
                        timeout_seconds=rendezvous_timeout_seconds,
                        poll_seconds=poll_seconds,
                    )
                except SAEPhaseAlreadyPassed:
                    # Another lane may commit while this one crosses the state
                    # read.  Re-enter only through exact artifact revalidation.
                    continue
                outcome = _run_phase(
                    control,
                    lease,
                    attempt,
                    paths,
                    executor,
                    revalidator,
                    poll_seconds=poll_seconds,
                    launch_timeout_seconds=rendezvous_timeout_seconds,
                    completion_timeout_seconds=completion_timeout_seconds,
                    heartbeat_interval_seconds=heartbeat_interval_seconds,
                )
                if outcome.outcome != "passed":
                    raise SAEFarmPhaseFailed(
                        f"phase {phase} attempt {attempt.attempt_id} "
                        f"committed {outcome.outcome}"
                    )
                executed.append(phase)
                break

        boundary = _boundary_result(
            control, None, executed=executed, skipped=skipped
        )
        if boundary is not None:
            return boundary
        return SAEFarmResult(
            operation_sha256=control.operation_sha256,
            status="complete",
            executed_phases=tuple(executed),
            skipped_phases=tuple(skipped),
            boundary_phase=None,
            abort_sha256=None,
        )


__all__ = [
    "SAEArtifactRevalidator",
    "SAEArtifactValidationError",
    "SAEChildExit",
    "SAEChildHandle",
    "SAEFarmError",
    "SAEFarmPhaseFailed",
    "SAEFarmResult",
    "SAEFreshChildExecutor",
    "SAEPhaseInvocation",
    "run_farm",
]
