from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import signal
import subprocess
from typing import Any

import pytest

from moe_sae.sae_farm import SAEPhaseInvocation
from moe_sae.sae_farm_runtime import (
    SAEChildLaunchError,
    SAELocalSubprocessExecutor,
    SAE_LEASE_FD_ENV,
    revalidate_phase_artifact,
)
from moe_sae.sae_operator import SAE_PHASES
from moe_sae.sae_paths import SAEOperationPaths
from moe_sae.sae_phase_results import publish_phase_result


OPERATION_SHA256 = "a" * 64
ATTEMPT_ID = "b" * 32
LEASE_ID = "c" * 32
LANE_ID = "d" * 24


class FakeProcess:
    def __init__(
        self,
        *,
        pid: int = 12_345,
        returncode: int | None = 0,
        wait_results: list[int | BaseException] | None = None,
    ) -> None:
        self.pid = pid
        self.returncode = returncode
        self.wait_results = list(wait_results or [])
        self.wait_timeouts: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self.wait_results:
            result = self.wait_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            self.returncode = result
            return result
        if self.returncode is None:
            raise AssertionError("test process has no configured wait result")
        return self.returncode


def setup_invocation(
    tmp_path: Path,
    *,
    lane_id: str = LANE_ID,
    attempt_id: str = ATTEMPT_ID,
    attempt_number: int = 1,
) -> tuple[SAEPhaseInvocation, int, Path]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    paths = SAEOperationPaths(artifact_root, OPERATION_SHA256)
    lease_path = tmp_path / f"lease-{lane_id}-{attempt_id}"
    lease_fd = os.open(lease_path, os.O_CREAT | os.O_RDWR, 0o600)
    invocation = SAEPhaseInvocation(
        phase=SAE_PHASES[0],
        attempt_id=attempt_id,
        attempt_number=attempt_number,
        operation_sha256=OPERATION_SHA256,
        lane_id=lane_id,
        lease_id=LEASE_ID,
        lease_fd=lease_fd,
        paths=paths,
    )
    config = tmp_path / "config.yaml"
    config.write_text("model: test\n", encoding="utf-8")
    return invocation, lease_fd, config


def test_executor_passes_exact_argv_environment_descriptor_and_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocation, lease_fd, config = setup_invocation(tmp_path)
    process = FakeProcess(returncode=0)
    captured: dict[str, Any] = {}

    def popen(command: tuple[str, ...], **kwargs: Any) -> FakeProcess:
        captured["command"] = command
        captured.update(kwargs)
        kwargs["stdout"].write(b"phase stdout\n")
        kwargs["stderr"].write(b"phase stderr\n")
        return process

    monkeypatch.setenv("MOE_SAE_PARENT_SENTINEL", "inherited")
    monkeypatch.setattr(
        "moe_sae.sae_farm_runtime.subprocess.Popen",
        popen,
    )
    try:
        executor = SAELocalSubprocessExecutor(
            config,
            invocation.paths.artifact_root,
            environment_overrides={"MOE_SAE_CHILD_SENTINEL": "added"},
        )
        handle = executor(invocation)

        assert captured["command"] == (
            str(Path(os.sys.executable).resolve()),
            "-m",
            "moe_sae.cli",
            "sae-phase",
            "--config",
            str(config.resolve()),
            "--artifact-root",
            str(invocation.paths.artifact_root),
            "--operation-sha256",
            OPERATION_SHA256,
            "--phase",
            SAE_PHASES[0],
            "--attempt-id",
            ATTEMPT_ID,
            "--attempt-number",
            "1",
        )
        assert captured["pass_fds"] == (lease_fd,)
        assert captured["env"][SAE_LEASE_FD_ENV] == str(lease_fd)
        assert captured["env"]["MOE_SAE_PARENT_SENTINEL"] == "inherited"
        assert captured["env"]["MOE_SAE_CHILD_SENTINEL"] == "added"
        assert captured["start_new_session"] is True
        assert captured["shell"] is False
        assert captured["stdin"] is subprocess.DEVNULL
        assert not captured["stdout"].closed
        assert not captured["stderr"].closed

        result = handle.poll()
        assert result is not None
        assert result.outcome == "passed"
        assert result.detail["returncode"] == 0
        assert result.detail["argv"] == list(captured["command"])
        assert captured["stdout"].closed
        assert captured["stderr"].closed
        assert handle.stdout_path.read_bytes() == b"phase stdout\n"
        assert handle.stderr_path.read_bytes() == b"phase stderr\n"
        assert handle.stdout_path != handle.stderr_path
        assert invocation.paths.operation_root in handle.stdout_path.parents
    finally:
        os.close(lease_fd)


@pytest.mark.parametrize(
    ("returncode", "outcome"),
    ((0, "passed"), (17, "failed"), (-9, "failed")),
)
def test_exit_code_mapping_and_idempotent_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    outcome: str,
) -> None:
    invocation, lease_fd, config = setup_invocation(tmp_path)
    process = FakeProcess(returncode=returncode)
    monkeypatch.setattr(
        "moe_sae.sae_farm_runtime.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    try:
        handle = SAELocalSubprocessExecutor(
            config, invocation.paths.artifact_root
        )(invocation)
        first = handle.poll()
        second = handle.poll()
        assert first is second
        assert first is not None
        assert first.outcome == outcome
        assert first.detail["returncode"] == returncode
    finally:
        os.close(lease_fd)


def test_logs_are_isolated_by_operation_phase_attempt_and_lane_and_never_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, first_fd, config = setup_invocation(tmp_path)
    second_lease_path = tmp_path / "second-lease"
    second_fd = os.open(second_lease_path, os.O_CREAT | os.O_RDWR, 0o600)
    second = replace(first, lane_id="e" * 24, lease_id="f" * 32, lease_fd=second_fd)
    third = replace(
        first,
        attempt_id="1" * 32,
        attempt_number=2,
    )
    processes: list[FakeProcess] = []

    def popen(*args: Any, **kwargs: Any) -> FakeProcess:
        process = FakeProcess(pid=20_000 + len(processes), returncode=0)
        processes.append(process)
        return process

    monkeypatch.setattr("moe_sae.sae_farm_runtime.subprocess.Popen", popen)
    try:
        executor = SAELocalSubprocessExecutor(
            config, first.paths.artifact_root
        )
        first_handle = executor(first)
        second_handle = executor(second)
        third_handle = executor(third)
        assert len(
            {
                first_handle.stdout_path,
                second_handle.stdout_path,
                third_handle.stdout_path,
            }
        ) == 3
        assert first.attempt_id in str(first_handle.stdout_path)
        assert second.lane_id in str(second_handle.stdout_path)
        assert third.attempt_id in str(third_handle.stdout_path)
        assert first_handle.poll() is not None
        assert second_handle.poll() is not None
        assert third_handle.poll() is not None

        with pytest.raises(SAEChildLaunchError, match="already exists"):
            executor(first)
        assert len(processes) == 3
    finally:
        os.close(first_fd)
        os.close(second_fd)


def test_collective_termination_stops_process_group_with_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocation, lease_fd, config = setup_invocation(tmp_path)
    process = FakeProcess(returncode=None, wait_results=[-signal.SIGTERM])
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        "moe_sae.sae_farm_runtime.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        "moe_sae.sae_farm_runtime.os.killpg",
        lambda pid, selected_signal: signals.append((pid, selected_signal)),
    )
    try:
        handle = SAELocalSubprocessExecutor(
            config,
            invocation.paths.artifact_root,
            termination_grace_seconds=2.0,
            kill_wait_seconds=1.0,
        )(invocation)
        handle.terminate_after_collective_failure("collective phase failure")

        assert signals == [(process.pid, signal.SIGTERM)]
        assert process.wait_timeouts == [2.0]
        terminal = handle.poll()
        assert terminal is not None
        assert terminal.outcome == "aborted"
        assert terminal.detail["returncode"] == -signal.SIGTERM
        assert terminal.detail["collective_termination_reason"] == (
            "collective phase failure"
        )
    finally:
        os.close(lease_fd)


def test_collective_termination_escalates_to_sigkill_and_reaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocation, lease_fd, config = setup_invocation(tmp_path)
    timed_out = subprocess.TimeoutExpired(("child",), 0.25)
    process = FakeProcess(returncode=None, wait_results=[timed_out, -signal.SIGKILL])
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        "moe_sae.sae_farm_runtime.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        "moe_sae.sae_farm_runtime.os.killpg",
        lambda pid, selected_signal: signals.append((pid, selected_signal)),
    )
    try:
        handle = SAELocalSubprocessExecutor(
            config,
            invocation.paths.artifact_root,
            termination_grace_seconds=0.25,
            kill_wait_seconds=0.5,
        )(invocation)
        handle.terminate_after_collective_failure("peer failed")

        assert signals == [
            (process.pid, signal.SIGTERM),
            (process.pid, signal.SIGKILL),
        ]
        assert process.wait_timeouts == [0.25, 0.5]
        assert handle.poll() is not None
        assert handle.poll().outcome == "aborted"  # type: ignore[union-attr]
    finally:
        os.close(lease_fd)


def test_popen_failure_closes_logs_and_same_identity_cannot_be_relaunched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocation, lease_fd, config = setup_invocation(tmp_path)
    captured: dict[str, Any] = {}

    def fail_popen(*args: Any, **kwargs: Any) -> FakeProcess:
        captured.update(kwargs)
        raise OSError("exec failed")

    monkeypatch.setattr(
        "moe_sae.sae_farm_runtime.subprocess.Popen", fail_popen
    )
    try:
        executor = SAELocalSubprocessExecutor(
            config, invocation.paths.artifact_root
        )
        with pytest.raises(SAEChildLaunchError, match="could not launch"):
            executor(invocation)
        assert captured["stdout"].closed
        assert captured["stderr"].closed
        with pytest.raises(SAEChildLaunchError, match="already exists"):
            executor(invocation)
    finally:
        os.close(lease_fd)


def test_invocation_validation_precedes_log_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocation, lease_fd, config = setup_invocation(tmp_path)
    called = False

    def popen(*args: Any, **kwargs: Any) -> FakeProcess:
        nonlocal called
        called = True
        return FakeProcess()

    monkeypatch.setattr("moe_sae.sae_farm_runtime.subprocess.Popen", popen)
    executor = SAELocalSubprocessExecutor(config, invocation.paths.artifact_root)
    try:
        with pytest.raises(ValueError, match="unknown SAE phase"):
            executor(replace(invocation, phase="not-a-phase"))
        os.close(lease_fd)
        with pytest.raises(ValueError, match="not open"):
            executor(invocation)
        assert not called
        assert not invocation.paths.operation_root.exists()
    finally:
        try:
            os.close(lease_fd)
        except OSError:
            pass


def test_revalidator_adapter_calls_immutable_phase_result_validation(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    paths = SAEOperationPaths(artifact_root, OPERATION_SHA256)
    expected = publish_phase_result(
        paths,
        SAE_PHASES[0],
        {"passed": True, "tokens": 5_000_000},
        created_at_utc="2026-07-22T18:00:00+00:00",
    )
    assert revalidate_phase_artifact(SAE_PHASES[0], paths) == expected
