from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import fcntl
import hashlib
import inspect
from pathlib import Path
import threading
import time
from typing import Any

import pytest

import moe_sae.sae_operator as sae_operator_module
from moe_sae.sae_farm import (
    SAEChildExit,
    SAEFarmError,
    SAEFarmPhaseFailed,
    SAEPhaseInvocation,
    run_farm,
)
from moe_sae.sae_operator import (
    CONTROL_POLL_SECONDS,
    EXPECTED_LANES,
    HEARTBEAT_INTERVAL_SECONDS,
    RENDEZVOUS_TIMEOUT_SECONDS,
    SAEControl,
    SAEOperationSpec,
    SAE_PHASES,
)
from moe_sae.sae_paths import SAEOperationPaths


@pytest.fixture(autouse=True)
def _fast_local_startup_lock_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local tmp_path locks need no NFSv3 recovery backoff."""

    monkeypatch.setattr(
        sae_operator_module, "LANE_STATUS_PUBLICATION_POLL_SECONDS", 0.001
    )


def digest(character: str) -> str:
    return character * 64


def phase_digest(
    phase: str, paths: SAEOperationPaths | None = None
) -> str:
    del paths
    return hashlib.sha256(phase.encode("utf-8")).hexdigest()


def operation_spec() -> SAEOperationSpec:
    return SAEOperationSpec(
        config_sha256=digest("1"),
        repository_content_sha256=digest("2"),
        consumer_environment_sha256=digest("3"),
        train_capture_manifest_sha256=digest("4"),
        eval_capture_manifest_sha256=digest("5"),
        activation_storage_attestation_sha256=digest("6"),
        sae_storage_attestation_sha256=digest("7"),
        capture_compatibility_sha256=digest("8"),
        smoke_campaign_sha256=digest("e"),
        smoke_suite_sha256=digest("b"),
        checkpoint_payload_root="gs://test-bucket/checkpoints",
        canary_waiver_sha256=digest("9"),
        sealed_recovery_waiver_sha256=digest("c"),
        performance_waiver_sha256=digest("a"),
    )


def operation(tmp_path: Path) -> tuple[list[SAEControl], SAEOperationPaths]:
    spec = operation_spec()
    creator = SAEControl(tmp_path)
    creator.begin(spec)
    controls = [
        SAEControl(tmp_path, spec.sha256) for _ in range(EXPECTED_LANES)
    ]
    return controls, SAEOperationPaths(tmp_path, spec.sha256)


def test_farm_defaults_use_the_nfs_safe_control_cadence() -> None:
    parameters = inspect.signature(run_farm).parameters
    assert parameters["poll_seconds"].default == CONTROL_POLL_SECONDS
    assert (
        parameters["heartbeat_interval_seconds"].default
        == HEARTBEAT_INTERVAL_SECONDS
    )
    assert (
        parameters["rendezvous_timeout_seconds"].default
        == RENDEZVOUS_TIMEOUT_SECONDS
    )
    assert (
        parameters["completion_timeout_seconds"].default
        == RENDEZVOUS_TIMEOUT_SECONDS
    )


@dataclass
class ImmediateHandle:
    terminal: SAEChildExit
    terminated: bool = False

    def poll(self) -> SAEChildExit:
        return self.terminal

    def terminate_after_collective_failure(self, reason: str) -> None:
        del reason
        self.terminated = True


class RecordingExecutor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.invocations: list[SAEPhaseInvocation] = []
        self.handles: list[ImmediateHandle] = []

    def __call__(self, invocation: SAEPhaseInvocation) -> ImmediateHandle:
        # The descriptor must still be held by the supervisor at child launch;
        # a subprocess adapter will preserve it with Popen(pass_fds=...).
        fcntl.fcntl(invocation.lease_fd, fcntl.F_GETFD)
        assert invocation.pass_fds == (invocation.lease_fd,)
        handle = ImmediateHandle(
            SAEChildExit("passed", {"phase": invocation.phase})
        )
        with self._lock:
            self.invocations.append(invocation)
            self.handles.append(handle)
        return handle


def run_all(
    controls: list[SAEControl],
    paths: SAEOperationPaths,
    executors: list[Any] | Any,
    revalidators: list[Any] | Any,
    *,
    timeout: float = 30.0,
) -> list[Any]:
    selected_executors = (
        executors
        if isinstance(executors, list)
        else [executors] * EXPECTED_LANES
    )
    selected_revalidators = (
        revalidators
        if isinstance(revalidators, list)
        else [revalidators] * EXPECTED_LANES
    )
    with ThreadPoolExecutor(max_workers=EXPECTED_LANES) as pool:
        futures: list[Future[Any]] = [
            pool.submit(
                run_farm,
                controls[index],
                paths,
                selected_executors[index],
                selected_revalidators[index],
                lane=f"controller-{index}",
                poll_seconds=0.001,
                rendezvous_timeout_seconds=5.0,
                completion_timeout_seconds=5.0,
                heartbeat_interval_seconds=0.01,
            )
            for index in range(EXPECTED_LANES)
        ]
        return [future.result(timeout=timeout) for future in futures]


def test_farm_walks_exact_plan_with_fresh_children_and_lifetime_lease(
    tmp_path: Path,
) -> None:
    controls, paths = operation(tmp_path)
    executor = RecordingExecutor()

    results = run_all(controls, paths, executor, phase_digest)

    assert all(result.status == "complete" for result in results)
    assert all(result.executed_phases == SAE_PHASES for result in results)
    assert all(result.skipped_phases == () for result in results)
    assert len(executor.invocations) == EXPECTED_LANES * len(SAE_PHASES)
    assert len({id(handle) for handle in executor.handles}) == len(
        executor.handles
    )
    assert not any(handle.terminated for handle in executor.handles)

    by_lane: dict[str, list[SAEPhaseInvocation]] = {}
    for invocation in executor.invocations:
        by_lane.setdefault(invocation.lane_id, []).append(invocation)
        assert invocation.paths is paths
        assert invocation.operation_sha256 == paths.operation_sha256
    assert len(by_lane) == EXPECTED_LANES
    for invocations in by_lane.values():
        assert [item.phase for item in invocations] == list(SAE_PHASES)
        assert len({item.lease_id for item in invocations}) == 1
        assert len({item.lease_fd for item in invocations}) == 1

    state = controls[0].load_state()
    for phase in SAE_PHASES:
        record = state["phases"][phase]
        assert record["status"] == "passed"
        assert record["result"]["result_sha256"] == phase_digest(phase)
        assert record["attempts"] == 1
        attempt_root = controls[0].root / "attempts" / record["attempt_id"]
        assert (attempt_root / "supervisor-launch.json").is_file()
        assert (attempt_root / "supervisor-complete.json").is_file()


def test_resume_skips_only_after_exact_artifact_revalidation(
    tmp_path: Path,
) -> None:
    controls, paths = operation(tmp_path)
    run_all(controls, paths, RecordingExecutor(), phase_digest)
    revalidated: list[str] = []
    lock = threading.Lock()

    def revalidate(phase: str, selected_paths: SAEOperationPaths) -> str:
        assert selected_paths is paths
        with lock:
            revalidated.append(phase)
        return phase_digest(phase)

    def forbidden_executor(invocation: SAEPhaseInvocation) -> ImmediateHandle:
        raise AssertionError(f"passed phase relaunched: {invocation.phase}")

    results = run_all(controls, paths, forbidden_executor, revalidate)

    assert all(result.status == "complete" for result in results)
    assert all(result.executed_phases == () for result in results)
    assert all(result.skipped_phases == SAE_PHASES for result in results)
    assert sorted(revalidated) == sorted(SAE_PHASES * EXPECTED_LANES)
    skip_releases = list(
        controls[0].root.glob("supervisor-sessions/*/skips/*.json")
    )
    assert len(skip_releases) == len(SAE_PHASES)


def test_inconsistent_revalidated_result_fails_closed(tmp_path: Path) -> None:
    controls, paths = operation(tmp_path)
    executor = RecordingExecutor()
    validators = [
        (lambda phase, selected_paths: phase_digest(phase))
        for _ in range(EXPECTED_LANES)
    ]
    validators[-1] = lambda phase, selected_paths: digest("f")

    with pytest.raises(SAEFarmPhaseFailed):
        run_all(controls, paths, executor, validators)

    record = controls[0].load_state()["phases"][SAE_PHASES[0]]
    assert record["status"] == "failed"
    assert record["result"]["result_sha256"] is None
    assert not any(handle.terminated for handle in executor.handles)


class BlockingHandle:
    def __init__(
        self,
        control: SAEControl,
        invocation: SAEPhaseInvocation,
        terminal: SAEChildExit | None,
        termination_pause: TerminationPause | None = None,
    ) -> None:
        self.control = control
        self.invocation = invocation
        self.terminal = terminal
        self.termination_pause = termination_pause
        self.terminated = threading.Event()

    def poll(self) -> SAEChildExit | None:
        return self.terminal

    def terminate_after_collective_failure(self, reason: str) -> None:
        assert "collective" in reason
        record = self.control.load_state()["phases"][self.invocation.phase]
        assert record["status"] in {"failed", "aborted"}
        assert record["attempt_id"] == self.invocation.attempt_id
        assert record["result"]["completion_release_sha256"]
        supervisor_release = (
            self.control.root
            / "attempts"
            / self.invocation.attempt_id
            / "supervisor-complete.json"
        )
        assert supervisor_release.is_file()
        if self.termination_pause is not None:
            self.termination_pause.wait()
        self.terminated.set()


class TerminationPause:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entered = 0
        self.all_entered = threading.Event()
        self.release = threading.Event()

    def wait(self) -> None:
        with self._lock:
            self._entered += 1
            if self._entered == EXPECTED_LANES - 1:
                self.all_entered.set()
        if not self.release.wait(5.0):
            raise TimeoutError("test did not release peer termination pause")


class OneTerminalExecutor:
    def __init__(
        self,
        control: SAEControl,
        outcome: str,
        *,
        termination_pause: TerminationPause | None = None,
    ) -> None:
        self.control = control
        self.outcome = outcome
        self.termination_pause = termination_pause
        self.failed_lane = hashlib.sha256(b"controller-0").hexdigest()[:24]
        self._lock = threading.Lock()
        self.handles: list[BlockingHandle] = []

    def __call__(self, invocation: SAEPhaseInvocation) -> BlockingHandle:
        terminal = (
            SAEChildExit(self.outcome)  # type: ignore[arg-type]
            if invocation.lane_id == self.failed_lane
            else None
        )
        handle = BlockingHandle(
            self.control,
            invocation,
            terminal,
            self.termination_pause,
        )
        with self._lock:
            self.handles.append(handle)
        return handle


@pytest.mark.parametrize("terminal_outcome", ("failed", "aborted"))
def test_live_peers_terminate_only_after_collective_release_and_history_survives(
    tmp_path: Path,
    terminal_outcome: str,
) -> None:
    controls, paths = operation(tmp_path)
    executor = OneTerminalExecutor(controls[0], terminal_outcome)

    with pytest.raises(SAEFarmPhaseFailed):
        run_all(controls, paths, executor, phase_digest)

    record = controls[0].load_state()["phases"][SAE_PHASES[0]]
    assert record["status"] == terminal_outcome
    live_peers = [handle for handle in executor.handles if handle.terminal is None]
    failed_child = [
        handle for handle in executor.handles if handle.terminal is not None
    ]
    assert len(live_peers) == EXPECTED_LANES - 1
    assert all(handle.terminated.is_set() for handle in live_peers)
    assert len(failed_child) == 1
    assert not failed_child[0].terminated.is_set()

    abort_control = SAEControl(tmp_path, paths.operation_sha256)

    def revalidate_then_stop(
        phase: str, selected_paths: SAEOperationPaths
    ) -> str:
        if phase == SAE_PHASES[0]:
            abort_control.request_abort("test stop after retry")
        return phase_digest(phase)

    retry = run_all(
        controls,
        paths,
        RecordingExecutor(),
        revalidate_then_stop,
    )
    assert all(result.status == "aborted" for result in retry)
    retried = controls[0].load_state()["phases"][SAE_PHASES[0]]
    assert retried["status"] == "passed"
    assert retried["attempts"] == 2
    assert retried["history"][0]["status"] == terminal_outcome
    assert abort_control.clear_abort()


def test_fast_aborted_lane_preserves_verdict_while_peers_reap(
    tmp_path: Path,
) -> None:
    controls, paths = operation(tmp_path)
    pause = TerminationPause()
    executor = OneTerminalExecutor(
        controls[0],
        "aborted",
        termination_pause=pause,
    )

    with ThreadPoolExecutor(max_workers=EXPECTED_LANES) as pool:
        futures = [
            pool.submit(
                run_farm,
                controls[index],
                paths,
                executor,
                phase_digest,
                lane=f"controller-{index}",
                poll_seconds=0.001,
                rendezvous_timeout_seconds=5.0,
                completion_timeout_seconds=5.0,
                heartbeat_interval_seconds=0.01,
            )
            for index in range(EXPECTED_LANES)
        ]
        try:
            assert pause.all_entered.wait(5.0)
            with pytest.raises(SAEFarmPhaseFailed):
                futures[0].result(timeout=3.0)
            fast_lane = hashlib.sha256(b"controller-0").hexdigest()[:24]
            observed = {
                lane["lane_id"]: lane for lane in controls[0].status()["lanes"]
            }
            assert observed[fast_lane]["state"] == "aborted"
            assert all(not future.done() for future in futures[1:])
        finally:
            pause.release.set()
        for future in futures[1:]:
            with pytest.raises(SAEFarmPhaseFailed):
                future.result(timeout=5.0)

    record = controls[0].load_state()["phases"][SAE_PHASES[0]]
    assert record["status"] == "aborted"
    assert {
        lane["state"] for lane in controls[0].status()["lanes"]
    } == {"aborted"}


class ReleasedHandle:
    def __init__(self, release: threading.Event) -> None:
        self.release = release
        self.terminated = False

    def poll(self) -> SAEChildExit | None:
        return SAEChildExit("passed") if self.release.is_set() else None

    def terminate_after_collective_failure(self, reason: str) -> None:
        del reason
        self.terminated = True


class GateExecutor:
    def __init__(self) -> None:
        self.release = threading.Event()
        self.all_started = threading.Event()
        self._lock = threading.Lock()
        self.handles: list[ReleasedHandle] = []

    def __call__(self, invocation: SAEPhaseInvocation) -> ReleasedHandle:
        assert invocation.phase == SAE_PHASES[0]
        handle = ReleasedHandle(self.release)
        with self._lock:
            self.handles.append(handle)
            if len(self.handles) == EXPECTED_LANES:
                self.all_started.set()
        return handle


def test_abort_is_latched_until_the_running_phase_reaches_a_boundary(
    tmp_path: Path,
) -> None:
    controls, paths = operation(tmp_path)
    executor = GateExecutor()
    abort_control = SAEControl(tmp_path, paths.operation_sha256)

    with ThreadPoolExecutor(max_workers=EXPECTED_LANES) as pool:
        futures = [
            pool.submit(
                run_farm,
                controls[index],
                paths,
                executor,
                phase_digest,
                lane=f"controller-{index}",
                poll_seconds=0.001,
                rendezvous_timeout_seconds=5.0,
                completion_timeout_seconds=5.0,
                heartbeat_interval_seconds=0.01,
            )
            for index in range(EXPECTED_LANES)
        ]
        assert executor.all_started.wait(5.0)
        request = abort_control.request_abort("stop after this phase")
        time.sleep(0.02)
        assert not any(handle.terminated for handle in executor.handles)
        assert not any(future.done() for future in futures)
        executor.release.set()
        results = [future.result(timeout=10.0) for future in futures]

    assert all(result.status == "aborted" for result in results)
    assert all(result.executed_phases == (SAE_PHASES[0],) for result in results)
    assert all(result.boundary_phase == SAE_PHASES[1] for result in results)
    assert all(result.abort_sha256 == request["abort_sha256"] for result in results)
    assert controls[0].load_state()["phases"][SAE_PHASES[0]]["status"] == "passed"
    assert controls[0].load_state()["phases"][SAE_PHASES[1]]["status"] == "pending"
    assert not any(handle.terminated for handle in executor.handles)
    assert abort_control.clear_abort()


def test_farm_rejects_cross_operation_paths_before_acquiring_a_lane(
    tmp_path: Path,
) -> None:
    controls, _ = operation(tmp_path)
    wrong_paths = SAEOperationPaths(tmp_path, digest("d"))

    with pytest.raises(SAEFarmError, match="different SAE operations"):
        run_farm(controls[0], wrong_paths, RecordingExecutor(), phase_digest)
