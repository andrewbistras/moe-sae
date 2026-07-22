"""Local subprocess adapters for the JAX-free SAE farm supervisor.

Every numerical phase runs in a fresh interpreter.  The parent supervisor
keeps ownership of the durable lane lease and passes that descriptor to the
child only as a liveness/fencing witness.  A child cannot claim success by
exit status alone: :func:`revalidate_phase_artifact` independently validates
the immutable phase-result envelope consumed by :func:`moe_sae.sae_farm.run_farm`.

This module intentionally imports neither JAX nor any numerical training
module.  It is safe to import in the long-lived control-plane process.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
from typing import Any, BinaryIO

from moe_sae.sae_farm import SAEChildExit, SAEPhaseInvocation
from moe_sae.sae_operator import SAE_PHASES
from moe_sae.sae_paths import SAEOperationPaths
from moe_sae.sae_phase_results import validate_phase_result


SAE_LEASE_FD_ENV = "MOE_SAE_SAE_LEASE_FD"

_ATTEMPT_RE = re.compile(r"[0-9a-f]{32}")
_LANE_RE = re.compile(r"[0-9a-f]{24}")
_LEASE_RE = re.compile(r"[0-9a-f]{32}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class SAEFarmRuntimeError(RuntimeError):
    """A numerical child could not be launched, observed, or terminated."""


class SAEChildLaunchError(SAEFarmRuntimeError):
    """A fresh phase child could not be launched without ambiguity."""


class SAEChildTerminationError(SAEFarmRuntimeError):
    """A collectively authorized child shutdown could not be completed."""


def _positive_seconds(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _resolved_file(path: str | Path, name: str, *, executable: bool = False) -> Path:
    selected = Path(path).expanduser().resolve()
    if not selected.is_file():
        raise ValueError(f"{name} must name an existing regular file: {selected}")
    if executable and not os.access(selected, os.X_OK):
        raise ValueError(f"{name} is not executable: {selected}")
    return selected


def _python_launcher(path: str | Path) -> Path:
    """Return an absolute Python launcher path without resolving symlinks.

    A virtual environment's ``bin/python`` is normally a symlink to its base
    interpreter.  CPython uses the launcher location to discover
    ``pyvenv.cfg``; resolving that symlink therefore escapes the environment
    and can make the installed ``moe_sae`` package unavailable to a child.
    """

    selected = Path(os.path.abspath(Path(path).expanduser()))
    if not selected.is_file():
        raise ValueError(
            f"python_executable must name an existing regular file: {selected}"
        )
    if not os.access(selected, os.X_OK):
        raise ValueError(f"python_executable is not executable: {selected}")
    return selected


def _environment(overrides: Mapping[str, str] | None) -> dict[str, str]:
    result = dict(os.environ)
    if overrides is None:
        return result
    if not isinstance(overrides, Mapping):
        raise TypeError("environment_overrides must be a mapping")
    for name, value in overrides.items():
        if (
            not isinstance(name, str)
            or not name
            or "=" in name
            or "\x00" in name
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise ValueError("environment overrides must contain valid strings")
        result[name] = value
    return result


def _validate_invocation(
    invocation: SAEPhaseInvocation,
    *,
    artifact_root: Path,
) -> None:
    if not isinstance(invocation, SAEPhaseInvocation):
        raise TypeError("invocation must be an SAEPhaseInvocation")
    if invocation.phase not in SAE_PHASES:
        raise ValueError(f"unknown SAE phase: {invocation.phase!r}")
    if _ATTEMPT_RE.fullmatch(invocation.attempt_id) is None:
        raise ValueError("attempt_id must be 32 lowercase hexadecimal characters")
    if (
        isinstance(invocation.attempt_number, bool)
        or not isinstance(invocation.attempt_number, int)
        or invocation.attempt_number <= 0
    ):
        raise ValueError("attempt_number must be a positive integer")
    if _SHA256_RE.fullmatch(invocation.operation_sha256) is None:
        raise ValueError(
            "operation_sha256 must be 64 lowercase hexadecimal characters"
        )
    if _LANE_RE.fullmatch(invocation.lane_id) is None:
        raise ValueError("lane_id must be 24 lowercase hexadecimal characters")
    if _LEASE_RE.fullmatch(invocation.lease_id) is None:
        raise ValueError("lease_id must be 32 lowercase hexadecimal characters")
    if not isinstance(invocation.paths, SAEOperationPaths):
        raise TypeError("invocation paths must be SAEOperationPaths")
    if invocation.paths.operation_sha256 != invocation.operation_sha256:
        raise ValueError("invocation and paths name different SAE operations")
    if invocation.paths.artifact_root != artifact_root:
        raise ValueError("invocation and executor use different artifact roots")
    if (
        isinstance(invocation.lease_fd, bool)
        or not isinstance(invocation.lease_fd, int)
        or invocation.lease_fd < 0
    ):
        raise ValueError("lease_fd must be a nonnegative integer descriptor")
    if invocation.pass_fds != (invocation.lease_fd,):
        raise ValueError("invocation pass_fds does not contain exactly the lease FD")
    try:
        os.fstat(invocation.lease_fd)
    except OSError as error:
        raise ValueError("lease_fd is not open in the supervisor") from error


def _within_operation(paths: SAEOperationPaths, candidate: Path) -> Path:
    operation_root = paths.operation_root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(operation_root)
    except ValueError as error:
        raise SAEChildLaunchError(
            f"child log path escapes the SAE operation: {candidate}"
        ) from error
    return resolved


def _open_unique_logs(
    invocation: SAEPhaseInvocation,
) -> tuple[Path, Path, BinaryIO, BinaryIO]:
    """Create a collision-proof lane directory and its two log files."""

    parent = _within_operation(
        invocation.paths,
        invocation.paths.operation_root
        / "logs"
        / "sae-farm"
        / invocation.phase
        / f"attempt-{invocation.attempt_number:04d}-{invocation.attempt_id}",
    )
    parent.mkdir(parents=True, exist_ok=True)
    # Resolve again after creation so a pre-existing symlinked parent cannot
    # redirect a supposedly operation-scoped log.
    parent = _within_operation(invocation.paths, parent)
    lane_root = _within_operation(
        invocation.paths, parent / f"lane-{invocation.lane_id}"
    )
    try:
        lane_root.mkdir(mode=0o750, exist_ok=False)
    except FileExistsError as error:
        raise SAEChildLaunchError(
            "a child log directory already exists for this exact "
            f"phase/attempt/lane: {lane_root}"
        ) from error
    stdout_path = _within_operation(invocation.paths, lane_root / "stdout.log")
    stderr_path = _within_operation(invocation.paths, lane_root / "stderr.log")
    stdout: BinaryIO | None = None
    stderr: BinaryIO | None = None
    try:
        stdout_descriptor = os.open(
            stdout_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o640,
        )
        stdout = os.fdopen(stdout_descriptor, "wb", buffering=0)
        stderr_descriptor = os.open(
            stderr_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o640,
        )
        stderr = os.fdopen(stderr_descriptor, "wb", buffering=0)
    except BaseException:
        if stderr is not None:
            stderr.close()
        if stdout is not None:
            stdout.close()
        raise
    return stdout_path, stderr_path, stdout, stderr


class SAESubprocessHandle:
    """Non-blocking child handle with process-group-aware bounded shutdown."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        command: tuple[str, ...],
        stdout_path: Path,
        stderr_path: Path,
        stdout: BinaryIO,
        stderr: BinaryIO,
        termination_grace_seconds: float,
        kill_wait_seconds: float,
    ) -> None:
        self._process = process
        self._command = command
        self._stdout_path = stdout_path
        self._stderr_path = stderr_path
        self._stdout = stdout
        self._stderr = stderr
        self._termination_grace_seconds = termination_grace_seconds
        self._kill_wait_seconds = kill_wait_seconds
        self._terminal: SAEChildExit | None = None
        self._lock = threading.Lock()

    @property
    def pid(self) -> int:
        return int(self._process.pid)

    @property
    def stdout_path(self) -> Path:
        return self._stdout_path

    @property
    def stderr_path(self) -> Path:
        return self._stderr_path

    def _close_logs(self) -> None:
        for handle in (self._stdout, self._stderr):
            if not handle.closed:
                handle.close()

    def _exit(
        self,
        returncode: int,
        *,
        collectively_terminated: bool,
        reason: str | None = None,
    ) -> SAEChildExit:
        if self._terminal is None:
            outcome = (
                "aborted"
                if collectively_terminated
                else ("passed" if returncode == 0 else "failed")
            )
            detail: dict[str, Any] = {
                "pid": self.pid,
                "returncode": int(returncode),
                "argv": list(self._command),
                "stdout_path": str(self._stdout_path),
                "stderr_path": str(self._stderr_path),
            }
            if reason is not None:
                detail["collective_termination_reason"] = reason[:2_000]
            self._terminal = SAEChildExit(outcome, detail)
        self._close_logs()
        return self._terminal

    def poll(self) -> SAEChildExit | None:
        with self._lock:
            if self._terminal is not None:
                return self._terminal
            try:
                returncode = self._process.poll()
            except BaseException:
                self._close_logs()
                raise
            if returncode is None:
                return None
            # subprocess.Popen.poll() performs waitpid(WNOHANG), so observing
            # a return code here also reaps the direct child.
            return self._exit(int(returncode), collectively_terminated=False)

    def terminate_after_collective_failure(self, reason: str) -> None:
        """Terminate the whole fresh session only after the farm authorizes it."""

        if not isinstance(reason, str) or not reason:
            raise ValueError("collective termination reason cannot be empty")
        with self._lock:
            if self._terminal is not None:
                return
            returncode = self._process.poll()
            if returncode is not None:
                self._exit(int(returncode), collectively_terminated=False)
                return

            try:
                try:
                    os.killpg(self.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    returncode = self._process.wait(
                        timeout=self._termination_grace_seconds
                    )
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        returncode = self._process.wait(
                            timeout=self._kill_wait_seconds
                        )
                    except subprocess.TimeoutExpired:
                        # SIGKILL has already been delivered.  A final wait is
                        # required to reap the direct child and prevent a
                        # zombie; under normal kernels it returns immediately.
                        returncode = self._process.wait()
            except BaseException as error:
                raise SAEChildTerminationError(
                    f"could not terminate/reap SAE child process group {self.pid}"
                ) from error
            finally:
                # Even an exceptional OS-level termination path must not leak
                # the parent's copies of the diagnostic log descriptors.
                self._close_logs()
            self._exit(
                int(returncode),
                collectively_terminated=True,
                reason=reason,
            )


class SAELocalSubprocessExecutor:
    """Start exactly one fresh ``python -m moe_sae.cli sae-phase`` child."""

    def __init__(
        self,
        config_path: str | Path,
        artifact_root: str | Path,
        *,
        python_executable: str | Path | None = None,
        environment_overrides: Mapping[str, str] | None = None,
        termination_grace_seconds: float = 30.0,
        kill_wait_seconds: float = 10.0,
    ) -> None:
        self.config_path = _resolved_file(config_path, "config_path")
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        if not self.artifact_root.is_dir():
            raise ValueError(
                f"artifact_root must name an existing directory: {self.artifact_root}"
            )
        # Preserve the environment launcher rather than resolving it to the
        # uv-managed base interpreter.  Phase children must use the already
        # synchronized project environment directly; they never invoke uv or
        # perform dependency resolution themselves.
        self.python_executable = _python_launcher(
            python_executable or sys.executable
        )
        self._base_environment = _environment(environment_overrides)
        self._termination_grace_seconds = _positive_seconds(
            "termination_grace_seconds", termination_grace_seconds
        )
        self._kill_wait_seconds = _positive_seconds(
            "kill_wait_seconds", kill_wait_seconds
        )

    def __call__(self, invocation: SAEPhaseInvocation) -> SAESubprocessHandle:
        _validate_invocation(invocation, artifact_root=self.artifact_root)
        command = (
            str(self.python_executable),
            "-m",
            "moe_sae.cli",
            "sae-phase",
            "--config",
            str(self.config_path),
            "--artifact-root",
            str(self.artifact_root),
            "--operation-sha256",
            invocation.operation_sha256,
            "--phase",
            invocation.phase,
            "--attempt-id",
            invocation.attempt_id,
            "--attempt-number",
            str(invocation.attempt_number),
        )
        environment = dict(self._base_environment)
        environment[SAE_LEASE_FD_ENV] = str(invocation.lease_fd)
        stdout_path, stderr_path, stdout, stderr = _open_unique_logs(invocation)
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                pass_fds=invocation.pass_fds,
                start_new_session=True,
                shell=False,
            )
        except BaseException as error:
            stdout.close()
            stderr.close()
            raise SAEChildLaunchError(
                f"could not launch SAE phase child for {invocation.phase}/"
                f"{invocation.attempt_id}"
            ) from error
        return SAESubprocessHandle(
            process,
            command=command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdout=stdout,
            stderr=stderr,
            termination_grace_seconds=self._termination_grace_seconds,
            kill_wait_seconds=self._kill_wait_seconds,
        )


def revalidate_phase_artifact(phase: str, paths: SAEOperationPaths) -> str:
    """Concrete :class:`SAEArtifactRevalidator` for supervised phase results."""

    return validate_phase_result(paths, phase)


__all__ = [
    "SAEChildLaunchError",
    "SAEChildTerminationError",
    "SAEFarmRuntimeError",
    "SAELocalSubprocessExecutor",
    "SAESubprocessHandle",
    "SAE_LEASE_FD_ENV",
    "revalidate_phase_artifact",
]
