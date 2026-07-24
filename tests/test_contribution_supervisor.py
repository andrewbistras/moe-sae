from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.contribution_performance import (
    ExactPerformanceIdentity,
    authorize_accelerator_run,
)
from moe_sae.contribution_supervisor import (
    ContributionSupervisorError,
    build_execution_record,
    supervise_authorized_process,
    validate_execution_record,
    validate_supervision_result,
)


def identity() -> ExactPerformanceIdentity:
    return ExactPerformanceIdentity(
        graph_sha256=canonical_sha256({"graph": "test"}),
        hardware_sha256=canonical_sha256({"hardware": "test"}),
        config_sha256=canonical_sha256({"config": "test"}),
    )


def authorization(*, seconds: int = 10):
    return authorize_accelerator_run(
        identity=identity(),
        maximum_runtime_seconds=seconds,
        purpose="qualification",
    )


def paths(root: Path) -> dict[str, Path]:
    return {
        "log_path": root / "worker.log",
        "execution_record_path": root / "execution.json",
        "result_path": root / "result.json",
    }


class FakeClock:
    def __init__(self) -> None:
        self.value = 1_000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeProcess:
    def __init__(
        self,
        clock: FakeClock,
        *,
        exit_after: float | None,
        exit_on_term: bool = False,
        returncode: int = 0,
    ) -> None:
        self.clock = clock
        self.exit_after = exit_after
        self.exit_on_term = exit_on_term
        self.final_returncode = returncode
        self.pid = 4242
        self.returncode: int | None = None
        self.killed = False
        self.terminated = False

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is not None:
            return self.returncode
        if self.killed:
            self.returncode = -int(signal.SIGKILL)
            return self.returncode
        if self.terminated and self.exit_on_term:
            self.clock.advance(min(0.05, timeout or 0.05))
            self.returncode = -int(signal.SIGTERM)
            return self.returncode
        if timeout is None:
            raise AssertionError("fake process received an unbounded wait")
        if self.exit_after is not None and self.exit_after <= timeout:
            self.clock.advance(self.exit_after)
            self.exit_after = 0.0
            self.returncode = self.final_returncode
            return self.returncode
        if self.exit_after is not None:
            self.exit_after -= timeout
        self.clock.advance(timeout)
        raise subprocess.TimeoutExpired(("fake-worker",), timeout)

    def poll(self) -> int | None:
        return self.returncode


class FakeRuntime:
    def __init__(self, process: FakeProcess) -> None:
        self.process = process
        self.factory_calls = []
        self.signals: list[int] = []

    def factory(self, command, **kwargs):
        self.factory_calls.append((command, kwargs))
        kwargs["stdout"].write(b"fake worker output\n")
        return self.process

    def signal_group(self, pid: int, selected_signal: int) -> None:
        assert pid == self.process.pid
        self.signals.append(selected_signal)
        if selected_signal == int(signal.SIGTERM):
            self.process.terminated = True
        elif selected_signal == int(signal.SIGKILL):
            self.process.killed = True


class BrokenWaitProcess(FakeProcess):
    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock, exit_after=None)
        self.broken = True

    def wait(self, timeout: float | None = None) -> int:
        if self.broken:
            self.broken = False
            raise RuntimeError("wait backend failed")
        return super().wait(timeout)


def supervise_fake(
    tmp_path: Path,
    process: FakeProcess,
    clock: FakeClock,
    *,
    seconds: int = 10,
    grace: float = 2.0,
):
    runtime = FakeRuntime(process)
    destinations = paths(tmp_path)
    result = supervise_authorized_process(
        authorization=authorization(seconds=seconds),
        command=("fake-worker", "--fixed"),
        environment={"LANG": "C", "SECRET": "not-persisted"},
        working_directory=tmp_path,
        termination_grace_seconds=grace,
        clock=clock,
        process_factory=runtime.factory,
        signal_process_group=runtime.signal_group,
        **destinations,
    )
    return result, runtime, destinations


def test_execution_record_binds_command_environment_paths_and_deadlines(tmp_path):
    destinations = paths(tmp_path)
    record = build_execution_record(
        authorization=authorization(),
        command=("worker", "--mode", "capture"),
        environment={"TOKEN": "secret", "LANG": "C"},
        working_directory=tmp_path,
        termination_grace_seconds=2.0,
        **destinations,
    )
    checked = validate_execution_record(record)

    assert checked["command"] == ["worker", "--mode", "capture"]
    assert checked["environment_keys"] == ["LANG", "TOKEN"]
    assert "secret" not in json.dumps(checked)
    assert checked["environment_sha256"] == canonical_sha256(
        {"LANG": "C", "TOKEN": "secret"}
    )
    assert checked["soft_termination_seconds"] == 8.0
    assert checked["hard_deadline_seconds"] == 10
    assert checked["execution_sha256"] == canonical_sha256(
        {key: value for key, value in checked.items() if key != "execution_sha256"}
    )

    changed = build_execution_record(
        authorization=authorization(),
        command=("worker", "--mode", "capture"),
        environment={"TOKEN": "different", "LANG": "C"},
        working_directory=tmp_path,
        termination_grace_seconds=2.0,
        **destinations,
    )
    assert changed["execution_sha256"] != record["execution_sha256"]


def test_completed_fake_process_is_reaped_and_records_log(tmp_path):
    clock = FakeClock()
    process = FakeProcess(clock, exit_after=1.25)
    result, runtime, destinations = supervise_fake(tmp_path, process, clock)

    assert result["status"] == "completed"
    assert result["returncode"] == 0
    assert result["elapsed_seconds"] == 1.25
    assert result["process_group_reaped"] is True
    assert not result["term_sent"]
    assert not result["kill_sent"]
    assert runtime.signals == []
    assert runtime.factory_calls[0][1]["start_new_session"] is True
    assert runtime.factory_calls[0][1]["shell"] is False
    assert destinations["log_path"].read_bytes() == b"fake worker output\n"
    assert json.loads(destinations["result_path"].read_text()) == result
    assert validate_supervision_result(result) == result


def test_term_then_kill_occurs_no_later_than_qualification_hard_deadline(
    tmp_path,
):
    clock = FakeClock()
    process = FakeProcess(clock, exit_after=None)
    result, runtime, _ = supervise_fake(
        tmp_path,
        process,
        clock,
        seconds=10,
        grace=2.0,
    )

    assert result["status"] == "killed_at_deadline"
    assert result["term_requested_elapsed_seconds"] == 8.0
    assert result["kill_requested_elapsed_seconds"] == 10.0
    assert result["elapsed_seconds"] == 10.0
    assert result["process_group_reaped"] is True
    assert runtime.signals == [int(signal.SIGTERM), int(signal.SIGKILL)]


def test_full_600_second_qualification_kills_at_exact_fake_deadline(tmp_path):
    clock = FakeClock()
    process = FakeProcess(clock, exit_after=None)
    result, _, _ = supervise_fake(
        tmp_path,
        process,
        clock,
        seconds=600,
        grace=10.0,
    )

    assert result["term_requested_elapsed_seconds"] == 590.0
    assert result["kill_requested_elapsed_seconds"] == 600.0
    assert result["elapsed_seconds"] == 600.0


def test_process_exiting_during_grace_is_terminated_and_reaped(tmp_path):
    clock = FakeClock()
    process = FakeProcess(clock, exit_after=None, exit_on_term=True)
    result, runtime, _ = supervise_fake(tmp_path, process, clock)

    assert result["status"] == "terminated_at_deadline"
    assert result["term_sent"] is True
    assert result["kill_sent"] is False
    assert result["returncode"] == -int(signal.SIGTERM)
    assert runtime.signals == [int(signal.SIGTERM)]


def test_nonzero_early_exit_is_a_sealed_failed_result(tmp_path):
    clock = FakeClock()
    process = FakeProcess(clock, exit_after=0.5, returncode=7)
    result, _, _ = supervise_fake(tmp_path, process, clock)

    assert result["status"] == "failed"
    assert result["returncode"] == 7
    assert result["result_sha256"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )


def test_launch_failure_is_logged_and_sealed(tmp_path):
    def fail_factory(*_args, **_kwargs):
        raise FileNotFoundError("no worker")

    result = supervise_authorized_process(
        authorization=authorization(),
        command=("absent-worker",),
        environment={},
        working_directory=tmp_path,
        termination_grace_seconds=1.0,
        clock=FakeClock(),
        process_factory=fail_factory,
        signal_process_group=lambda _pid, _signal: None,
        **paths(tmp_path),
    )

    assert result["status"] == "launch_failed"
    assert result["pid"] is None
    assert result["process_group_reaped"] is False
    assert result["error_type"] == "FileNotFoundError"
    assert "no worker" in result["error_message"]


def test_supervisor_error_kills_reaps_persists_then_raises(tmp_path):
    clock = FakeClock()
    process = BrokenWaitProcess(clock)
    runtime = FakeRuntime(process)
    destinations = paths(tmp_path)

    with pytest.raises(ContributionSupervisorError, match="failed closed"):
        supervise_authorized_process(
            authorization=authorization(),
            command=("fake-worker",),
            environment={},
            working_directory=tmp_path,
            termination_grace_seconds=2.0,
            clock=clock,
            process_factory=runtime.factory,
            signal_process_group=runtime.signal_group,
            **destinations,
        )

    result = json.loads(destinations["result_path"].read_text())
    validated = validate_supervision_result(result)
    assert validated["status"] == "supervisor_error"
    assert validated["process_group_reaped"] is True
    assert validated["term_sent"] is True
    assert validated["kill_sent"] is True
    assert runtime.signals == [int(signal.SIGTERM), int(signal.SIGKILL)]


def test_create_only_intent_log_and_result_refuse_reuse(tmp_path):
    clock = FakeClock()
    process = FakeProcess(clock, exit_after=0.1)
    supervise_fake(tmp_path, process, clock)

    with pytest.raises(ContributionSupervisorError, match="already exists"):
        supervise_fake(
            tmp_path,
            FakeProcess(FakeClock(), exit_after=0.1),
            FakeClock(),
        )


def test_tampered_execution_and_result_self_hashes_are_rejected(tmp_path):
    clock = FakeClock()
    result, _, destinations = supervise_fake(
        tmp_path,
        FakeProcess(clock, exit_after=0.1),
        clock,
    )
    execution = json.loads(destinations["execution_record_path"].read_text())
    execution["command"].append("--tampered")
    with pytest.raises(ContributionSupervisorError, match="self-hash"):
        validate_execution_record(execution)

    result["elapsed_seconds"] = 999.0
    with pytest.raises(ContributionSupervisorError, match="self-hash"):
        validate_supervision_result(result)


def test_manually_forged_overlong_qualification_authorization_is_rejected(
    tmp_path,
):
    valid = authorization()
    forged = type(valid)(
        purpose="qualification",
        maximum_runtime_seconds=601,
        identity_sha256=valid.identity_sha256,
        basis="bounded_qualification",
        qualification_receipt_sha256=None,
    )
    with pytest.raises(ContributionSupervisorError, match="exceeds 600"):
        build_execution_record(
            authorization=forged,
            command=("worker",),
            environment={},
            working_directory=tmp_path,
            termination_grace_seconds=1.0,
            **paths(tmp_path),
        )


def test_qualified_authorization_requires_full_receipt_revalidation(tmp_path):
    valid = authorization()
    forged = type(valid)(
        purpose="workload",
        maximum_runtime_seconds=601,
        identity_sha256=valid.identity_sha256,
        basis="qualified_exact_identity",
        qualification_receipt_sha256=canonical_sha256({"forged": True}),
    )
    with pytest.raises(ContributionSupervisorError, match="full qualification"):
        build_execution_record(
            authorization=forged,
            command=("worker",),
            environment={},
            working_directory=tmp_path,
            termination_grace_seconds=1.0,
            **paths(tmp_path),
        )


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="requires POSIX process groups")
def test_real_subprocess_ignoring_term_is_killed_and_reaped(tmp_path):
    destinations = paths(tmp_path)
    result = supervise_authorized_process(
        authorization=authorization(seconds=1),
        command=(
            sys.executable,
            "-c",
            (
                "import signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ready', flush=True); "
                "time.sleep(30)"
            ),
        ),
        environment=dict(os.environ),
        working_directory=tmp_path,
        termination_grace_seconds=0.2,
        **destinations,
    )

    assert result["status"] == "killed_at_deadline"
    assert result["term_sent"] is True
    assert result["kill_sent"] is True
    assert result["process_group_reaped"] is True
    assert result["returncode"] == -int(signal.SIGKILL)
    assert result["elapsed_seconds"] < 3.0
    with pytest.raises(ProcessLookupError):
        os.kill(result["pid"], 0)
    assert b"ready" in destinations["log_path"].read_bytes()
