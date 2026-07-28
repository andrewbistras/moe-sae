"""Hard-deadline process-group supervisor for Experiment-2 accelerator work.

The supervisor consumes an :class:`AcceleratorRunAuthorization`, publishes a
create-only execution intent, starts one new POSIX process group, and reserves
the end of the authorized interval for graceful termination.  SIGKILL is sent
no later than the hard deadline when SIGTERM is ignored, after which the child
is reaped before a self-hashed result becomes visible.

Environment values are included in the execution identity hash but omitted
from the persisted record to avoid copying credentials into an audit artifact.
The immutable record retains the sorted key inventory and an explicit
redaction policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import errno
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
from typing import Any, Protocol

from moe_sae.contribution_artifacts import (
    ContributionSchemaError,
    canonical_json_bytes,
    canonical_sha256,
)
from moe_sae.contribution_performance import (
    MAX_LEARNING_HORIZON_ACCELERATOR_SECONDS,
    MAX_QUALIFICATION_SECONDS,
    MAX_UNGATED_ACCELERATOR_SECONDS,
    AcceleratorRunAuthorization,
    validate_qualification_receipt,
)

EXECUTION_RECORD_SCHEMA_VERSION = "gemma4-contribution-supervised-execution-v1"
SUPERVISION_RESULT_SCHEMA_VERSION = "gemma4-contribution-supervision-result-v1"
DEFAULT_TERMINATION_GRACE_SECONDS = 10.0
KILL_REAP_TIMEOUT_SECONDS = 30.0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_AUTHORIZATION_FIELDS = {
    "purpose",
    "maximum_runtime_seconds",
    "identity_sha256",
    "basis",
    "qualification_receipt_sha256",
}
_EXECUTION_FIELDS = {
    "schema_version",
    "authorization",
    "command",
    "environment_sha256",
    "environment_keys",
    "environment_recording",
    "working_directory",
    "log_path",
    "execution_record_path",
    "result_path",
    "termination_grace_seconds",
    "soft_termination_seconds",
    "hard_deadline_seconds",
    "execution_sha256",
}
_RESULT_FIELDS = {
    "schema_version",
    "execution_sha256",
    "status",
    "pid",
    "returncode",
    "term_sent",
    "kill_sent",
    "process_group_reaped",
    "elapsed_seconds",
    "term_requested_elapsed_seconds",
    "kill_requested_elapsed_seconds",
    "log_size_bytes",
    "log_sha256",
    "error_type",
    "error_message",
    "result_sha256",
}


class ContributionSupervisorError(RuntimeError):
    """A launch, deadline, persistence, or result invariant failed."""


class _Process(Protocol):
    pid: int
    returncode: int | None

    def wait(self, timeout: float | None = None) -> int: ...

    def poll(self) -> int | None: ...


ProcessFactory = Callable[..., _Process]
SignalProcessGroup = Callable[[int, int], None]
Clock = Callable[[], float]


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContributionSupervisorError(f"{name} must be a lowercase SHA-256")
    return value


def _strict_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ContributionSupervisorError(f"{name} must be boolean")
    return value


def _nonnegative_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ContributionSupervisorError(f"{name} must be numeric")
    try:
        selected = float(value)
    except (TypeError, ValueError) as error:
        raise ContributionSupervisorError(f"{name} must be numeric") from error
    if not math.isfinite(selected) or selected < 0.0:
        raise ContributionSupervisorError(f"{name} must be finite and nonnegative")
    return selected


def _positive_float(name: str, value: Any) -> float:
    selected = _nonnegative_float(name, value)
    if selected == 0.0:
        raise ContributionSupervisorError(f"{name} must be strictly positive")
    return selected


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContributionSupervisorError(f"{name} must be a positive integer")
    return value


def _optional_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContributionSupervisorError(f"{name} must be an integer or null")
    return value


def _exact_mapping(
    name: str,
    value: Any,
    fields: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContributionSupervisorError(f"{name} must be a mapping")
    if set(value) != fields:
        missing = sorted(fields.difference(value))
        extra = sorted(set(value).difference(fields))
        raise ContributionSupervisorError(
            f"{name} fields are not exact; missing={missing}, extra={extra}"
        )
    return value


def _canonical_sha256(value: Any, *, name: str) -> str:
    try:
        return canonical_sha256(value)
    except ContributionSchemaError as error:
        raise ContributionSupervisorError(
            f"{name} is not finite canonical JSON"
        ) from error


def _validate_authorization(
    value: AcceleratorRunAuthorization | Mapping[str, Any],
    *,
    qualification_receipt: Mapping[str, Any] | None = None,
    require_full_receipt: bool = False,
) -> dict[str, Any]:
    if type(value) is AcceleratorRunAuthorization:
        raw = asdict(value)
    else:
        raw = dict(_exact_mapping("authorization", value, _AUTHORIZATION_FIELDS))
    selected = dict(_exact_mapping("authorization", raw, _AUTHORIZATION_FIELDS))
    purpose = selected["purpose"]
    maximum = _positive_int(
        "authorization.maximum_runtime_seconds",
        selected["maximum_runtime_seconds"],
    )
    identity_sha256 = _sha256(
        "authorization.identity_sha256",
        selected["identity_sha256"],
    )
    receipt = selected["qualification_receipt_sha256"]
    if receipt is not None:
        receipt = _sha256(
            "authorization.qualification_receipt_sha256",
            receipt,
        )
    basis = selected["basis"]
    if purpose == "qualification":
        if (
            maximum > MAX_QUALIFICATION_SECONDS
            or basis != "bounded_qualification"
            or receipt is not None
        ):
            raise ContributionSupervisorError(
                "qualification authorization is inconsistent or exceeds 600 seconds"
            )
    elif purpose == "workload":
        if basis == "bounded_ungated_workload":
            if maximum > MAX_UNGATED_ACCELERATOR_SECONDS or receipt is not None:
                raise ContributionSupervisorError(
                    "bounded workload authorization is inconsistent"
                )
        elif basis == "bounded_learning_horizon_workload":
            if (
                maximum > MAX_LEARNING_HORIZON_ACCELERATOR_SECONDS
                or receipt is not None
            ):
                raise ContributionSupervisorError(
                    "learning-horizon workload authorization is inconsistent"
                )
        elif basis == "qualified_exact_identity":
            if receipt is None:
                raise ContributionSupervisorError(
                    "qualified workload lacks its qualification receipt"
                )
            if qualification_receipt is None and require_full_receipt:
                raise ContributionSupervisorError(
                    "qualified workload must provide the full qualification "
                    "receipt at launch"
                )
            if qualification_receipt is not None:
                try:
                    checked_receipt = validate_qualification_receipt(
                        qualification_receipt
                    )
                except ValueError as error:
                    raise ContributionSupervisorError(
                        "qualified workload receipt is invalid"
                    ) from error
                if (
                    checked_receipt["receipt_sha256"] != receipt
                    or checked_receipt["identity_sha256"] != identity_sha256
                ):
                    raise ContributionSupervisorError(
                        "qualified workload receipt does not match its authorization"
                    )
        else:
            raise ContributionSupervisorError("workload authorization basis is invalid")
    else:
        raise ContributionSupervisorError(
            "authorization purpose must be qualification or workload"
        )
    if basis != "qualified_exact_identity" and qualification_receipt is not None:
        raise ContributionSupervisorError(
            "an unqualified bounded authorization must not carry a receipt"
        )
    return {
        "purpose": purpose,
        "maximum_runtime_seconds": maximum,
        "identity_sha256": identity_sha256,
        "basis": basis,
        "qualification_receipt_sha256": receipt,
    }


def _normalize_command(command: Sequence[str]) -> tuple[str, ...]:
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise ContributionSupervisorError(
            "command must be a nonempty sequence of strings"
        )
    selected = tuple(command)
    if not selected:
        raise ContributionSupervisorError("command must not be empty")
    if any(
        not isinstance(item, str) or not item or "\x00" in item for item in selected
    ):
        raise ContributionSupervisorError(
            "every command argument must be a nonempty NUL-free string"
        )
    return selected


def _normalize_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(environment, Mapping):
        raise ContributionSupervisorError("environment must be a mapping")
    selected: dict[str, str] = {}
    for key, value in environment.items():
        if (
            not isinstance(key, str)
            or _ENVIRONMENT_NAME_RE.fullmatch(key) is None
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise ContributionSupervisorError(
                "environment keys and values must be valid NUL-free strings"
            )
        selected[key] = value
    return dict(sorted(selected.items()))


def _absolute_file_path(value: str | Path, *, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ContributionSupervisorError(f"{name} must be a filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ContributionSupervisorError(f"{name} must be absolute")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise ContributionSupervisorError(
            f"{name} parent directory is unavailable"
        ) from error
    if not parent.is_dir() or parent.is_symlink():
        raise ContributionSupervisorError(f"{name} parent must be one real directory")
    if path.name in ("", ".", ".."):
        raise ContributionSupervisorError(f"{name} filename is invalid")
    return parent / path.name


def _working_directory(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise ContributionSupervisorError("working_directory must be a filesystem path")
    try:
        selected = Path(value).expanduser().resolve(strict=True)
    except OSError as error:
        raise ContributionSupervisorError("working_directory is unavailable") from error
    if not selected.is_dir():
        raise ContributionSupervisorError(
            "working_directory must be one existing directory"
        )
    return selected


def _require_distinct_paths(paths: Mapping[str, Path]) -> None:
    inverse: dict[Path, str] = {}
    for name, path in paths.items():
        previous = inverse.get(path)
        if previous is not None:
            raise ContributionSupervisorError(
                f"{name} and {previous} must use distinct paths"
            )
        inverse[path] = name


def _require_absent(path: Path, *, name: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise ContributionSupervisorError(f"cannot inspect {name}") from error
    raise ContributionSupervisorError(f"{name} already exists")


def build_execution_record(
    *,
    authorization: AcceleratorRunAuthorization,
    command: Sequence[str],
    environment: Mapping[str, str],
    working_directory: str | Path,
    log_path: str | Path,
    execution_record_path: str | Path,
    result_path: str | Path,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
    qualification_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable, credential-redacted launch identity."""

    auth = _validate_authorization(
        authorization,
        qualification_receipt=qualification_receipt,
        require_full_receipt=True,
    )
    argv = _normalize_command(command)
    env = _normalize_environment(environment)
    cwd = _working_directory(working_directory)
    paths = {
        "log_path": _absolute_file_path(log_path, name="log_path"),
        "execution_record_path": _absolute_file_path(
            execution_record_path,
            name="execution_record_path",
        ),
        "result_path": _absolute_file_path(result_path, name="result_path"),
    }
    _require_distinct_paths(paths)
    grace = _positive_float(
        "termination_grace_seconds",
        termination_grace_seconds,
    )
    maximum = auth["maximum_runtime_seconds"]
    if grace >= maximum:
        raise ContributionSupervisorError(
            "termination grace must be shorter than the authorized runtime"
        )
    unsigned = {
        "schema_version": EXECUTION_RECORD_SCHEMA_VERSION,
        "authorization": auth,
        "command": list(argv),
        "environment_sha256": _canonical_sha256(
            env,
            name="execution environment",
        ),
        "environment_keys": list(env),
        "environment_recording": "values-redacted-hash-binds-complete-mapping",
        "working_directory": str(cwd),
        "log_path": str(paths["log_path"]),
        "execution_record_path": str(paths["execution_record_path"]),
        "result_path": str(paths["result_path"]),
        "termination_grace_seconds": grace,
        "soft_termination_seconds": maximum - grace,
        "hard_deadline_seconds": maximum,
    }
    return {
        **unsigned,
        "execution_sha256": _canonical_sha256(
            unsigned,
            name="execution record",
        ),
    }


def validate_execution_record(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a persisted execution record and re-derive its deadline."""

    selected = dict(_exact_mapping("execution record", value, _EXECUTION_FIELDS))
    claimed = _sha256("execution_sha256", selected.pop("execution_sha256"))
    if selected.get("schema_version") != EXECUTION_RECORD_SCHEMA_VERSION:
        raise ContributionSupervisorError("execution record schema is incompatible")
    if _canonical_sha256(selected, name="execution record") != claimed:
        raise ContributionSupervisorError("execution record self-hash is invalid")
    auth = _validate_authorization(selected["authorization"])
    _normalize_command(selected["command"])
    _sha256("environment_sha256", selected["environment_sha256"])
    keys = selected["environment_keys"]
    if (
        not isinstance(keys, list)
        or keys != sorted(keys)
        or len(keys) != len(set(keys))
        or any(
            not isinstance(key, str) or _ENVIRONMENT_NAME_RE.fullmatch(key) is None
            for key in keys
        )
    ):
        raise ContributionSupervisorError(
            "execution environment key inventory is invalid"
        )
    if (
        selected["environment_recording"]
        != "values-redacted-hash-binds-complete-mapping"
    ):
        raise ContributionSupervisorError(
            "execution environment redaction policy changed"
        )
    cwd = _working_directory(selected["working_directory"])
    paths = {
        name: _absolute_file_path(selected[name], name=name)
        for name in ("log_path", "execution_record_path", "result_path")
    }
    _require_distinct_paths(paths)
    grace = _positive_float(
        "termination_grace_seconds",
        selected["termination_grace_seconds"],
    )
    soft = _positive_float(
        "soft_termination_seconds",
        selected["soft_termination_seconds"],
    )
    hard = _positive_float(
        "hard_deadline_seconds",
        selected["hard_deadline_seconds"],
    )
    if (
        not _same_deadline(hard, float(auth["maximum_runtime_seconds"]))
        or not _same_deadline(soft, hard - grace)
        or grace >= hard
    ):
        raise ContributionSupervisorError(
            "execution deadlines do not derive from the authorization"
        )
    selected["authorization"] = auth
    selected["working_directory"] = str(cwd)
    for name, path in paths.items():
        selected[name] = str(path)
    selected["execution_sha256"] = claimed
    return selected


def _same_deadline(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise ContributionSupervisorError("atomic record write made no progress")
        offset += written


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish canonical bytes atomically without replacing an existing path."""

    try:
        payload = canonical_json_bytes(dict(value))
    except ContributionSchemaError as error:
        raise ContributionSupervisorError(
            "atomic record is not finite canonical JSON"
        ) from error
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise ContributionSupervisorError(
                f"immutable record already exists: {path}"
            ) from error
        directory = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _open_log(path: Path) -> Any:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise ContributionSupervisorError(
            f"immutable log already exists: {path}"
        ) from error
    except OSError as error:
        raise ContributionSupervisorError(f"cannot create log: {path}") from error
    return os.fdopen(descriptor, "w+b", buffering=0)


def _hash_open_log(handle: Any, path: Path) -> tuple[int, str]:
    try:
        handle.flush()
        os.fsync(handle.fileno())
        before = os.fstat(handle.fileno())
        observed = path.stat(follow_symlinks=False)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ) != (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
        ):
            raise ContributionSupervisorError(
                "log destination changed while the child was running"
            )
        handle.seek(0)
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(handle.fileno())
    except OSError as error:
        raise ContributionSupervisorError("cannot finalize process log") from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ContributionSupervisorError("process log changed while hashing")
    if size != before.st_size:
        raise ContributionSupervisorError("process log byte count changed")
    return size, digest.hexdigest()


def _clock_value(clock: Clock, *, name: str) -> float:
    return _nonnegative_float(name, clock())


def _wait_until(
    process: _Process,
    *,
    deadline: float,
    clock: Clock,
) -> int | None:
    remaining = max(0.0, deadline - _clock_value(clock, name="monotonic clock"))
    if remaining == 0.0:
        return process.poll()
    try:
        return process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        return None


def _signal_group(
    process: _Process,
    selected_signal: signal.Signals,
    *,
    signal_process_group: SignalProcessGroup,
) -> None:
    pid = _positive_int("child pid", process.pid)
    try:
        signal_process_group(pid, int(selected_signal))
    except ProcessLookupError:
        return
    except OSError as error:
        if error.errno == errno.ESRCH:
            return
        raise ContributionSupervisorError(
            f"cannot signal child process group with {selected_signal.name}"
        ) from error


def _reap_after_kill(
    process: _Process,
) -> int:
    try:
        return process.wait(timeout=KILL_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise ContributionSupervisorError(
            "SIGKILL was sent but the process-group leader was not reaped"
        ) from error


def _result_record(
    *,
    execution_sha256: str,
    status: str,
    pid: int | None,
    returncode: int | None,
    term_sent: bool,
    kill_sent: bool,
    process_group_reaped: bool,
    elapsed_seconds: float,
    term_requested_elapsed_seconds: float | None,
    kill_requested_elapsed_seconds: float | None,
    log_size_bytes: int,
    log_sha256: str,
    error: BaseException | None,
) -> dict[str, Any]:
    unsigned = {
        "schema_version": SUPERVISION_RESULT_SCHEMA_VERSION,
        "execution_sha256": _sha256("execution_sha256", execution_sha256),
        "status": status,
        "pid": pid,
        "returncode": returncode,
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "process_group_reaped": process_group_reaped,
        "elapsed_seconds": elapsed_seconds,
        "term_requested_elapsed_seconds": term_requested_elapsed_seconds,
        "kill_requested_elapsed_seconds": kill_requested_elapsed_seconds,
        "log_size_bytes": log_size_bytes,
        "log_sha256": _sha256("log_sha256", log_sha256),
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
    }
    return {
        **unsigned,
        "result_sha256": _canonical_sha256(
            unsigned,
            name="supervision result",
        ),
    }


def validate_supervision_result(
    value: Mapping[str, Any],
    *,
    expected_execution_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a result record and its terminal-state relationships."""

    selected = dict(_exact_mapping("supervision result", value, _RESULT_FIELDS))
    claimed = _sha256("result_sha256", selected.pop("result_sha256"))
    if selected.get("schema_version") != SUPERVISION_RESULT_SCHEMA_VERSION:
        raise ContributionSupervisorError("supervision result schema is incompatible")
    if _canonical_sha256(selected, name="supervision result") != claimed:
        raise ContributionSupervisorError("supervision result self-hash is invalid")
    execution_sha256 = _sha256(
        "result.execution_sha256",
        selected["execution_sha256"],
    )
    if expected_execution_sha256 is not None and execution_sha256 != _sha256(
        "expected_execution_sha256",
        expected_execution_sha256,
    ):
        raise ContributionSupervisorError(
            "supervision result belongs to another execution"
        )
    status = selected["status"]
    pid = _optional_int("result.pid", selected["pid"])
    returncode = _optional_int("result.returncode", selected["returncode"])
    term_sent = _strict_bool("result.term_sent", selected["term_sent"])
    kill_sent = _strict_bool("result.kill_sent", selected["kill_sent"])
    reaped = _strict_bool(
        "result.process_group_reaped",
        selected["process_group_reaped"],
    )
    elapsed = _nonnegative_float(
        "result.elapsed_seconds",
        selected["elapsed_seconds"],
    )
    term_elapsed = selected["term_requested_elapsed_seconds"]
    kill_elapsed = selected["kill_requested_elapsed_seconds"]
    if term_elapsed is not None:
        term_elapsed = _nonnegative_float(
            "result.term_requested_elapsed_seconds",
            term_elapsed,
        )
    if kill_elapsed is not None:
        kill_elapsed = _nonnegative_float(
            "result.kill_requested_elapsed_seconds",
            kill_elapsed,
        )
    log_size = selected["log_size_bytes"]
    if isinstance(log_size, bool) or not isinstance(log_size, int) or log_size < 0:
        raise ContributionSupervisorError(
            "result.log_size_bytes must be a nonnegative integer"
        )
    _sha256("result.log_sha256", selected["log_sha256"])
    error_type = selected["error_type"]
    error_message = selected["error_message"]
    if (error_type is None) != (error_message is None) or (
        error_type is not None
        and (
            not isinstance(error_type, str)
            or not error_type
            or not isinstance(error_message, str)
        )
    ):
        raise ContributionSupervisorError("supervision error fields are invalid")

    if status == "completed":
        valid = (
            pid is not None
            and returncode == 0
            and not term_sent
            and not kill_sent
            and reaped
            and error_type is None
        )
    elif status == "failed":
        valid = (
            pid is not None
            and returncode not in (None, 0)
            and not term_sent
            and not kill_sent
            and reaped
            and error_type is None
        )
    elif status == "terminated_at_deadline":
        valid = (
            pid is not None
            and returncode is not None
            and term_sent
            and not kill_sent
            and reaped
            and term_elapsed is not None
            and kill_elapsed is None
            and error_type is None
        )
    elif status == "killed_at_deadline":
        valid = (
            pid is not None
            and returncode is not None
            and term_sent
            and kill_sent
            and reaped
            and term_elapsed is not None
            and kill_elapsed is not None
            and error_type is None
        )
    elif status == "launch_failed":
        valid = (
            pid is None
            and returncode is None
            and not term_sent
            and not kill_sent
            and not reaped
            and error_type is not None
        )
    elif status == "supervisor_error":
        valid = (
            pid is not None
            and error_type is not None
            and (not reaped or returncode is not None)
        )
    else:
        raise ContributionSupervisorError("supervision result status is invalid")
    if not valid or (
        term_sent != (term_elapsed is not None)
        or kill_sent != (kill_elapsed is not None)
        or (kill_sent and not term_sent)
        or (term_elapsed is not None and term_elapsed > elapsed)
        or (kill_elapsed is not None and kill_elapsed > elapsed)
        or (
            term_elapsed is not None
            and kill_elapsed is not None
            and kill_elapsed < term_elapsed
        )
    ):
        raise ContributionSupervisorError(
            "supervision result terminal fields are inconsistent"
        )
    selected["result_sha256"] = claimed
    return selected


def supervise_authorized_process(
    *,
    authorization: AcceleratorRunAuthorization,
    command: Sequence[str],
    environment: Mapping[str, str],
    working_directory: str | Path,
    log_path: str | Path,
    execution_record_path: str | Path,
    result_path: str | Path,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
    qualification_receipt: Mapping[str, Any] | None = None,
    clock: Clock = time.monotonic,
    process_factory: ProcessFactory = subprocess.Popen,
    signal_process_group: SignalProcessGroup = os.killpg,
) -> dict[str, Any]:
    """Launch, terminate by the hard deadline, reap, and seal the result."""

    if (
        not callable(clock)
        or not callable(process_factory)
        or not callable(signal_process_group)
    ):
        raise ContributionSupervisorError(
            "clock, process_factory, and signal_process_group must be callable"
        )
    argv = _normalize_command(command)
    env = _normalize_environment(environment)
    execution = build_execution_record(
        authorization=authorization,
        command=argv,
        environment=env,
        working_directory=working_directory,
        log_path=log_path,
        execution_record_path=execution_record_path,
        result_path=result_path,
        termination_grace_seconds=termination_grace_seconds,
        qualification_receipt=qualification_receipt,
    )
    checked = validate_execution_record(execution)
    log = Path(checked["log_path"])
    intent = Path(checked["execution_record_path"])
    result_destination = Path(checked["result_path"])
    for name, path in (
        ("log_path", log),
        ("execution_record_path", intent),
        ("result_path", result_destination),
    ):
        _require_absent(path, name=name)
    _atomic_create_json(intent, execution)

    log_handle = _open_log(log)
    process: _Process | None = None
    started = _clock_value(clock, name="monotonic start")
    hard_deadline = started + float(checked["hard_deadline_seconds"])
    soft_deadline = started + float(checked["soft_termination_seconds"])
    term_sent = False
    kill_sent = False
    reaped = False
    returncode: int | None = None
    term_elapsed: float | None = None
    kill_elapsed: float | None = None
    launch_error: BaseException | None = None
    supervision_error: BaseException | None = None
    try:
        try:
            process = process_factory(
                argv,
                cwd=checked["working_directory"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
                close_fds=True,
            )
            _positive_int("child pid", process.pid)
        except BaseException as error:
            launch_error = error

        if process is not None:
            returncode = _wait_until(
                process,
                deadline=soft_deadline,
                clock=clock,
            )
            if returncode is None:
                requested_elapsed = max(
                    0.0,
                    _clock_value(clock, name="TERM clock") - started,
                )
                term_sent = True
                term_elapsed = requested_elapsed
                _signal_group(
                    process,
                    signal.SIGTERM,
                    signal_process_group=signal_process_group,
                )
                returncode = _wait_until(
                    process,
                    deadline=hard_deadline,
                    clock=clock,
                )
            if returncode is None:
                requested_elapsed = max(
                    0.0,
                    _clock_value(clock, name="KILL clock") - started,
                )
                kill_sent = True
                kill_elapsed = requested_elapsed
                _signal_group(
                    process,
                    signal.SIGKILL,
                    signal_process_group=signal_process_group,
                )
                returncode = _reap_after_kill(process)
            reaped = True
    except BaseException as error:
        supervision_error = error
    finally:
        if process is not None and not reaped:
            cleanup_errors: list[BaseException] = []
            if not term_sent:
                try:
                    requested_elapsed = max(
                        0.0,
                        _clock_value(
                            clock,
                            name="cleanup TERM clock",
                        )
                        - started,
                    )
                    term_sent = True
                    term_elapsed = requested_elapsed
                    _signal_group(
                        process,
                        signal.SIGTERM,
                        signal_process_group=signal_process_group,
                    )
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            try:
                requested_elapsed = max(
                    0.0,
                    _clock_value(clock, name="cleanup KILL clock") - started,
                )
                kill_sent = True
                kill_elapsed = requested_elapsed
                _signal_group(
                    process,
                    signal.SIGKILL,
                    signal_process_group=signal_process_group,
                )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                returncode = _reap_after_kill(process)
                reaped = True
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            if cleanup_errors and supervision_error is None:
                supervision_error = cleanup_errors[0]
            elif cleanup_errors and supervision_error is not None:
                try:
                    supervision_error.add_note(
                        "cleanup errors: "
                        + "; ".join(
                            f"{type(error).__name__}: {error}"
                            for error in cleanup_errors
                        )
                    )
                except AttributeError:  # pragma: no cover - Python >=3.11
                    pass
        stopped = _clock_value(clock, name="monotonic stop")
        try:
            log_size, log_sha256 = _hash_open_log(log_handle, log)
        finally:
            log_handle.close()

    elapsed = max(0.0, stopped - started)
    if process is None:
        status = "launch_failed"
        result_error = launch_error
    elif supervision_error is not None:
        status = "supervisor_error"
        result_error = supervision_error
    elif kill_sent:
        status = "killed_at_deadline"
        result_error = None
    elif term_sent:
        status = "terminated_at_deadline"
        result_error = None
    elif returncode == 0:
        status = "completed"
        result_error = None
    else:
        status = "failed"
        result_error = None
    result = _result_record(
        execution_sha256=execution["execution_sha256"],
        status=status,
        pid=process.pid if process is not None else None,
        returncode=returncode,
        term_sent=term_sent,
        kill_sent=kill_sent,
        process_group_reaped=reaped,
        elapsed_seconds=elapsed,
        term_requested_elapsed_seconds=term_elapsed,
        kill_requested_elapsed_seconds=kill_elapsed,
        log_size_bytes=log_size,
        log_sha256=log_sha256,
        error=result_error,
    )
    validate_supervision_result(
        result,
        expected_execution_sha256=execution["execution_sha256"],
    )
    _atomic_create_json(result_destination, result)
    if supervision_error is not None:
        raise ContributionSupervisorError(
            "supervisor failed closed; result "
            f"{result['result_sha256']} records the terminal state"
        ) from supervision_error
    return result


__all__ = [
    "DEFAULT_TERMINATION_GRACE_SECONDS",
    "EXECUTION_RECORD_SCHEMA_VERSION",
    "KILL_REAP_TIMEOUT_SECONDS",
    "SUPERVISION_RESULT_SCHEMA_VERSION",
    "ContributionSupervisorError",
    "build_execution_record",
    "supervise_authorized_process",
    "validate_execution_record",
    "validate_supervision_result",
]
