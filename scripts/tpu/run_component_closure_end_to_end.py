#!/usr/bin/env python3
"""Durably run component closure from training through compact GCS archival.

This controller-side driver deliberately owns no cloud-resource lifecycle.  It
only invokes the already-built five-resource training plan, constructs and runs
the six resident evaluation bundles, seals the scientific outputs, and uploads
one compact evidence archive plus its manifest.  Capture shards and checkpoints
are never archive inputs.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from moe_sae.component_closure_campaign import (
    inspect_training_completion,
    load_training_plan,
    parse_training_campaign,
)
from moe_sae.component_closure_evaluation import (
    build_evaluation_plan,
    collect_score_grid,
    collect_training_costs,
    load_evaluation_plan,
    validate_evaluation_plan,
)
from moe_sae.component_closure_references import load_a_reference_manifest
from moe_sae.component_closure_report import (
    COMMON_SST_POLICY,
    REPORT_SCHEMA_VERSION,
    SST_AGGREGATE_RELATIVE_TOLERANCE,
    SST_NORMALIZED_L1_TOLERANCE,
    build_closure_report,
    load_score_grid,
    render_markdown,
)
from moe_sae.contribution_artifacts import canonical_sha256


SCHEMA_VERSION = "gemma4-a512-component-closure-end-to-end-v1"
PUBLICATION_RECOVERY_SCHEMA_VERSION = (
    "gemma4-component-closure-publication-recovery-v1"
)
REPORT_RECOVERY_SCHEMA_VERSION = (
    "gemma4-a512-component-closure-report-recovery-v1"
)
REPORT_RECOVERY_REASON = (
    "condition_graph_and_accelerator_fp32_sst_reduction_order"
)
PUBLICATION_RECOVERY_REASON = (
    "fresh gcloud TPU SSH subprocesses returned transport code 1 "
    "after emitting exact per-worker terminal status 0"
)
PUBLICATION_RECOVERY_SCRIPT_SHA256 = (
    "53bf897bdce644c4d0bffe51eaccc2579cfd850c739242b7f883c9d5e10f72a1"
)
MAXIMUM_WALL_SECONDS = 43_200
MAXIMUM_EVALUATION_ATTEMPTS = 4
MAXIMUM_COMPACT_ARCHIVE_BYTES = 256 * 1024 * 1024
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_GCS_RE = re.compile(r"^gs://[a-z0-9][a-z0-9._-]{1,221}[a-z0-9](?:/[^?#\s]*)?$")


class ComponentClosureEndToEndError(RuntimeError):
    """The durable closure handoff is unsafe, incomplete, or inconsistent."""


@dataclass(frozen=True, slots=True)
class DriverPaths:
    output_root: Path
    state_root: Path

    @property
    def evaluation_plan(self) -> Path:
        return self.output_root / "evaluation-plan.json"

    @property
    def training_costs(self) -> Path:
        return self.output_root / "training-costs.json"

    @property
    def score_grid(self) -> Path:
        return self.output_root / "score-grid.json"

    @property
    def report_json(self) -> Path:
        return self.output_root / "report.json"

    @property
    def report_markdown(self) -> Path:
        return self.output_root / "report.md"

    @property
    def archive_index(self) -> Path:
        return self.output_root / "archive-index.json"

    @property
    def upload_manifest(self) -> Path:
        return self.output_root / "upload-manifest.json"

    @property
    def clock(self) -> Path:
        return self.state_root / "run-clock.json"

    @property
    def terminal_completion(self) -> Path:
        return self.state_root / "terminal-completion.json"

    @property
    def report_recovery_receipt(self) -> Path:
        return self.state_root / "report-recovery" / "receipt.json"

    @property
    def evaluation_state(self) -> Path:
        return self.state_root / "evaluation"

    def stage(self, name: str) -> Path:
        return self.state_root / "stages" / f"{name}.json"

    def log_attempt(self, stage: str, attempt: int) -> Path:
        nonce = uuid.uuid4().hex[:12]
        return self.state_root / "logs" / stage / f"attempt-{attempt:02d}-{nonce}.log"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", required=True, type=Path)
    parser.add_argument("--a-references", required=True, type=Path)
    parser.add_argument("--evaluation-runtime", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument(
        "--report-recovery-commit",
        help=(
            "clean report-generator revision for a report-only recovery; "
            "requires already-sealed training and evaluation stages"
        ),
    )
    parser.add_argument(
        "--report-recovery-script",
        type=Path,
        help=(
            "exact controller publication-recovery script; required with "
            "--report-recovery-commit"
        ),
    )
    parser.add_argument("--archive-gcs-prefix", required=True)
    parser.add_argument(
        "--maximum-wall-seconds",
        type=int,
        default=MAXIMUM_WALL_SECONDS,
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--gcloud-executable", default="gcloud")
    return parser


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ComponentClosureEndToEndError("record is not canonical JSON") from error


def _sha256_path(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ComponentClosureEndToEndError(
            f"expected immutable regular file is unavailable: {path}"
        )
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise ComponentClosureEndToEndError(f"cannot hash {path}") from error
    return digest.hexdigest(), size


def _seal_exact_bytes(path: Path, payload: bytes, *, label: str) -> tuple[str, int]:
    """Publish exact bytes without ever replacing an existing final artifact."""

    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        observed, size = _sha256_path(path)
        if observed != digest or size != len(payload):
            raise ComponentClosureEndToEndError(f"existing {label} changed: {path}")
        return digest, size
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return _seal_exact_bytes(path, payload, label=label)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return digest, len(payload)


def _seal_json(path: Path, value: Mapping[str, Any], *, label: str) -> tuple[str, int]:
    return _seal_exact_bytes(path, _canonical_json_bytes(value), label=label)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComponentClosureEndToEndError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ComponentClosureEndToEndError(f"{label} is not an object: {path}")
    return value


def _validate_self_hash(
    value: Mapping[str, Any],
    *,
    hash_field: str,
    label: str,
) -> dict[str, Any]:
    selected = copy.deepcopy(dict(value))
    claimed = selected.pop(hash_field, None)
    if (
        not isinstance(claimed, str)
        or _SHA_RE.fullmatch(claimed) is None
        or canonical_sha256(selected) != claimed
    ):
        raise ComponentClosureEndToEndError(f"{label} self-hash is invalid")
    return {**selected, hash_field: claimed}


def _artifact_row(path: Path) -> dict[str, Any]:
    digest, size = _sha256_path(path)
    return {"path": str(path), "sha256": digest, "size_bytes": size}


def _validate_artifact_rows(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ComponentClosureEndToEndError(f"{label} has no output inventory")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ComponentClosureEndToEndError(
                f"{label} output {index} fields changed"
            )
        path = Path(str(raw["path"]))
        observed_sha, observed_size = _sha256_path(path)
        if observed_sha != raw["sha256"] or observed_size != raw["size_bytes"]:
            raise ComponentClosureEndToEndError(
                f"{label} output {index} changed: {path}"
            )
        rows.append(dict(raw))
    return rows


def _load_stage(
    path: Path,
    *,
    name: str,
    project_commit: str,
    input_sha256: Mapping[str, str],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = _validate_self_hash(
        _load_json(path, label=f"{name} stage"),
        hash_field="stage_sha256",
        label=f"{name} stage",
    )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("record_type") != "stage_completion"
        or value.get("stage") != name
        or value.get("status") != "completed"
        or value.get("project_commit") != project_commit
        or value.get("input_sha256") != dict(input_sha256)
    ):
        raise ComponentClosureEndToEndError(f"{name} stage identity changed")
    _validate_artifact_rows(value.get("outputs"), label=f"{name} stage")
    if not isinstance(value.get("metadata"), Mapping):
        raise ComponentClosureEndToEndError(f"{name} stage metadata changed")
    return value


def _seal_stage(
    path: Path,
    *,
    name: str,
    project_commit: str,
    input_sha256: Mapping[str, str],
    outputs: Sequence[Path],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "stage_completion",
        "stage": name,
        "status": "completed",
        "project_commit": project_commit,
        "input_sha256": dict(input_sha256),
        "outputs": [_artifact_row(path) for path in outputs],
        "metadata": copy.deepcopy(dict(metadata or {})),
        "completed_at": _timestamp(),
    }
    value = {**body, "stage_sha256": canonical_sha256(body)}
    if path.exists():
        observed = _load_stage(
            path,
            name=name,
            project_commit=project_commit,
            input_sha256=input_sha256,
        )
        assert observed is not None
        expected_outputs = body["outputs"]
        if (
            observed["outputs"] != expected_outputs
            or observed["metadata"] != body["metadata"]
        ):
            raise ComponentClosureEndToEndError(
                f"{name} stage completion differs from current outputs"
            )
        return observed
    _seal_json(path, value, label=f"{name} stage completion")
    return value


def _validate_roots(paths: DriverPaths, *, checkout: Path) -> None:
    for label, root in (
        ("output root", paths.output_root),
        ("state root", paths.state_root),
    ):
        if not root.is_absolute():
            raise ComponentClosureEndToEndError(f"{label} must be absolute")
        if root.exists() and (root.is_symlink() or not root.is_dir()):
            raise ComponentClosureEndToEndError(
                f"{label} is not a regular directory: {root}"
            )
        try:
            root.relative_to(checkout)
        except ValueError:
            pass
        else:
            raise ComponentClosureEndToEndError(
                f"{label} must be outside the immutable Git checkout"
            )
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if paths.output_root == paths.state_root:
        raise ComponentClosureEndToEndError("output and state roots must be distinct")


def _validate_checkout(training: Any) -> None:
    if (
        training.working_directory.is_symlink()
        or not training.working_directory.is_dir()
        or training.python_executable.is_symlink()
        or not training.python_executable.is_file()
        or not os.access(training.python_executable, os.X_OK)
    ):
        raise ComponentClosureEndToEndError(
            "controller checkout or Python executable is invalid"
        )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=training.working_directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            cwd=training.working_directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ComponentClosureEndToEndError(
            "cannot validate the controller Git checkout"
        ) from error
    if head != training.project_commit or dirty:
        raise ComponentClosureEndToEndError(
            "controller checkout must be clean at the training-plan commit"
        )


def _validate_report_recovery_checkout(
    training: Any,
    *,
    report_recovery_commit: str,
) -> None:
    if (
        _GIT_RE.fullmatch(report_recovery_commit) is None
        or report_recovery_commit == training.project_commit
    ):
        raise ComponentClosureEndToEndError(
            "report recovery requires one distinct full lowercase Git SHA"
        )
    if (
        training.working_directory.is_symlink()
        or not training.working_directory.is_dir()
        or training.python_executable.is_symlink()
        or not training.python_executable.is_file()
        or not os.access(training.python_executable, os.X_OK)
    ):
        raise ComponentClosureEndToEndError(
            "controller checkout or Python executable is invalid"
        )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=training.working_directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            cwd=training.working_directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        ancestry = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                training.project_commit,
                report_recovery_commit,
            ],
            cwd=training.working_directory,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ComponentClosureEndToEndError(
            "cannot validate the report-recovery Git checkout"
        ) from error
    if head != report_recovery_commit or dirty or ancestry.returncode != 0:
        raise ComponentClosureEndToEndError(
            "report-recovery checkout must be a clean descendant of the "
            "scientific commit"
        )


@contextmanager
def _exclusive_driver_lock(state_root: Path):
    path = state_root / "driver.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ComponentClosureEndToEndError(
                "another end-to-end driver owns this state root"
            ) from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _bind_clock(
    path: Path,
    *,
    project_commit: str,
    maximum_wall_seconds: int,
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if maximum_wall_seconds <= 0 or maximum_wall_seconds > MAXIMUM_WALL_SECONDS:
        raise ComponentClosureEndToEndError(
            "maximum wall must be in (0, 43200] seconds"
        )
    if path.exists():
        value = _validate_self_hash(
            _load_json(path, label="run clock"),
            hash_field="clock_sha256",
            label="run clock",
        )
        if (
            value.get("schema_version") != SCHEMA_VERSION
            or value.get("record_type") != "run_clock"
            or value.get("project_commit") != project_commit
            or value.get("maximum_wall_seconds") != maximum_wall_seconds
            or value.get("input_sha256") != dict(input_sha256)
            or not isinstance(value.get("started_at_unix_seconds"), (int, float))
        ):
            raise ComponentClosureEndToEndError("run clock identity changed")
        return value
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "run_clock",
        "project_commit": project_commit,
        "maximum_wall_seconds": maximum_wall_seconds,
        "input_sha256": dict(input_sha256),
        "started_at": _timestamp(),
        "started_at_unix_seconds": time.time(),
    }
    value = {**body, "clock_sha256": canonical_sha256(body)}
    _seal_json(path, value, label="run clock")
    return value


def _remaining_seconds(clock: Mapping[str, Any]) -> int:
    deadline = float(clock["started_at_unix_seconds"]) + int(
        clock["maximum_wall_seconds"]
    )
    remaining = int(deadline - time.time())
    if remaining <= 0:
        raise ComponentClosureEndToEndError("end-to-end campaign reached its wall")
    return min(remaining, MAXIMUM_WALL_SECONDS)


def _run_logged_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
) -> int:
    if timeout_seconds <= 0 or timeout_seconds > MAXIMUM_WALL_SECONDS:
        raise ComponentClosureEndToEndError("subprocess timeout is outside the wall")
    log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(
            log_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise ComponentClosureEndToEndError(
            f"attempt log already exists: {log_path}"
        ) from error
    with os.fdopen(descriptor, "wb", closefd=True) as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, 15)
                process.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, 9)
                except ProcessLookupError:
                    pass
                process.wait()
            return 124


def _validate_training_barrier(training: Any) -> Path:
    fingerprints = []
    for task in training.trajectories:
        fingerprint = inspect_training_completion(training, task)
        if fingerprint is None:
            raise ComponentClosureEndToEndError(
                f"training is incomplete: {task.task_id}"
            )
        fingerprints.append(fingerprint)
    if len(fingerprints) != 18:
        raise ComponentClosureEndToEndError("training matrix is not 18 trajectories")
    path = training.state_root / "training-complete.json"
    value = _validate_self_hash(
        _load_json(path, label="training barrier"),
        hash_field="record_sha256",
        label="training barrier",
    )
    if (
        value.get("record_type") != "training_phase_completion"
        or value.get("status") != "training_completed"
        or value.get("plan_sha256") != training.plan_sha256
        or not isinstance(value.get("training"), Mapping)
        or set(value["training"]) != {task.task_id for task in training.trajectories}
    ):
        raise ComponentClosureEndToEndError("training barrier identity changed")
    return path


def _run_training(
    *,
    training_plan_path: Path,
    training: Any,
    paths: DriverPaths,
    clock: Mapping[str, Any],
    input_sha256: Mapping[str, str],
) -> None:
    stage_name = "training"
    if (
        _load_stage(
            paths.stage(stage_name),
            name=stage_name,
            project_commit=training.project_commit,
            input_sha256=input_sha256,
        )
        is not None
    ):
        _validate_training_barrier(training)
        return
    existing_barrier = training.state_root / "training-complete.json"
    if existing_barrier.exists():
        barrier = _validate_training_barrier(training)
        _seal_stage(
            paths.stage(stage_name),
            name=stage_name,
            project_commit=training.project_commit,
            input_sha256=input_sha256,
            outputs=[barrier],
            metadata={
                "controller_log": None,
                "recovered_existing_barrier": True,
            },
        )
        return
    command = [
        str(training.python_executable),
        str(
            training.working_directory / "scripts/tpu/run_component_closure_campaign.py"
        ),
        "run",
        "--plan",
        str(training_plan_path),
    ]
    log_path = paths.log_attempt(stage_name, 1)
    returncode = _run_logged_command(
        command,
        cwd=training.working_directory,
        log_path=log_path,
        timeout_seconds=_remaining_seconds(clock),
    )
    if returncode != 0:
        raise ComponentClosureEndToEndError(
            f"training controller failed with {returncode}; see {log_path}"
        )
    barrier = _validate_training_barrier(training)
    _seal_stage(
        paths.stage(stage_name),
        name=stage_name,
        project_commit=training.project_commit,
        input_sha256=input_sha256,
        outputs=[barrier],
        metadata={"controller_log": str(log_path)},
    )


def _build_eval_plan(
    *,
    training_plan: Mapping[str, Any],
    a_references: Mapping[str, Any],
    runtime: Mapping[str, Any],
    project_commit: str,
    paths: DriverPaths,
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    stage_name = "evaluation-plan"
    existing = _load_stage(
        paths.stage(stage_name),
        name=stage_name,
        project_commit=project_commit,
        input_sha256=input_sha256,
    )
    if existing is not None:
        return load_evaluation_plan(paths.evaluation_plan)
    value = build_evaluation_plan(
        training_plan,
        a_references,
        runtime,
        evaluation_source_revision=project_commit,
        state_root=paths.evaluation_state,
    )
    value = validate_evaluation_plan(value)
    _seal_json(paths.evaluation_plan, value, label="evaluation plan")
    checked = load_evaluation_plan(paths.evaluation_plan)
    _seal_stage(
        paths.stage(stage_name),
        name=stage_name,
        project_commit=project_commit,
        input_sha256=input_sha256,
        outputs=[paths.evaluation_plan],
        metadata={"evaluation_plan_sha256": checked["plan_sha256"]},
    )
    return checked


def _validate_evaluation_completions(evaluation: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for bundle in evaluation["bundles"]:
        path = (
            Path(evaluation["state_root"])
            / "bundles"
            / f"{bundle['panel']}-seed{bundle['seed']}"
            / "completion.json"
        )
        value = _validate_self_hash(
            _load_json(path, label="evaluation bundle completion"),
            hash_field="completion_sha256",
            label="evaluation bundle completion",
        )
        if (
            value.get("status") != "bundle_published"
            or value.get("bundle_sha256") != bundle["bundle_sha256"]
            or value.get("resource_id") != bundle["resource_id"]
            or not isinstance(value.get("cells"), list)
            or len(value["cells"]) != 12
        ):
            raise ComponentClosureEndToEndError(
                f"evaluation bundle completion changed: {path}"
            )
        paths.append(path)
    if len(paths) != 6:
        raise ComponentClosureEndToEndError("evaluation requires six completions")
    return paths


def _require_stage_outputs(
    stage: Mapping[str, Any],
    *,
    expected: Sequence[Path],
    label: str,
) -> None:
    if stage.get("outputs") != [_artifact_row(path) for path in expected]:
        raise ComponentClosureEndToEndError(f"{label} stage outputs changed")


def _publication_log_terminal_status(
    path: Path,
    *,
    controller: int,
    process: int,
) -> int:
    prefix = (
        "terminal action=component-closure-evaluate-bundle "
        f"controller={controller} process={process} status="
    ).encode("ascii")
    rows = [
        line
        for line in path.read_bytes().splitlines()
        if line.startswith(prefix)
    ]
    if (
        len(rows) != 1
        or re.fullmatch(
            rb"(?:0|[1-9][0-9]{0,2})",
            rows[0][len(prefix) :],
        )
        is None
    ):
        raise ComponentClosureEndToEndError(
            f"publication log lacks one exact terminal marker: {path}"
        )
    return int(rows[0][len(prefix) :])


def _validate_publication_recovery_receipts(
    evaluation: Mapping[str, Any],
    *,
    recovery_script: Path,
) -> list[dict[str, Any]]:
    script_sha, script_size = _sha256_path(recovery_script)
    if script_sha != PUBLICATION_RECOVERY_SCRIPT_SHA256:
        raise ComponentClosureEndToEndError(
            "publication-recovery script identity changed"
        )
    state_root = Path(str(evaluation["state_root"]))
    receipt_root = state_root / "publication-recovery"
    completions = _validate_evaluation_completions(evaluation)
    expected_names = {
        f"{bundle['panel']}-seed{bundle['seed']}.json"
        for bundle in evaluation["bundles"]
    }
    observed_names = (
        {path.name for path in receipt_root.glob("*.json")}
        if receipt_root.is_dir() and not receipt_root.is_symlink()
        else set()
    )
    if len(expected_names) != 6 or observed_names != expected_names:
        raise ComponentClosureEndToEndError(
            "report recovery requires exactly the six derived publication receipts"
        )

    validated: list[dict[str, Any]] = []
    for bundle_index, (bundle, completion_path) in enumerate(
        zip(evaluation["bundles"], completions, strict=True)
    ):
        panel = str(bundle["panel"])
        seed = int(bundle["seed"])
        recovery_id = f"controller-publish-recovery-{panel}-seed{seed}"
        completion = _validate_self_hash(
            _load_json(
                completion_path,
                label=f"{panel} seed {seed} evaluation completion",
            ),
            hash_field="completion_sha256",
            label=f"{panel} seed {seed} evaluation completion",
        )
        if (
            set(completion)
            != {
                "schema_version",
                "status",
                "plan_sha256",
                "bundle_sha256",
                "resource_id",
                "panel",
                "seed",
                "attempt_id",
                "cells",
                "completion_sha256",
            }
            or completion["schema_version"]
            != "gemma4-component-closure-evaluation-operator-v1"
            or completion["status"] != "bundle_published"
            or completion["plan_sha256"] != evaluation["plan_sha256"]
            or completion["bundle_sha256"] != bundle["bundle_sha256"]
            or completion["resource_id"] != bundle["resource_id"]
            or completion["panel"] != panel
            or completion["seed"] != seed
            or completion["attempt_id"] != recovery_id
            or not isinstance(completion["cells"], list)
            or len(completion["cells"]) != 12
        ):
            raise ComponentClosureEndToEndError(
                f"{panel} seed {seed} completion is not publication-recovered"
            )

        path = receipt_root / f"{panel}-seed{seed}.json"
        if path.is_symlink() or not path.is_file():
            raise ComponentClosureEndToEndError(
                f"publication-recovery receipt is unavailable: {path}"
            )
        value = _validate_self_hash(
            _load_json(path, label="publication-recovery receipt"),
            hash_field="receipt_sha256",
            label="publication-recovery receipt",
        )
        if set(value) != {
            "schema_version",
            "status",
            "project_commit",
            "plan_sha256",
            "bundle_sha256",
            "bundle_index",
            "panel",
            "seed",
            "resource_id",
            "recovery_id",
            "source_log_root",
            "source_logs",
            "completion_path",
            "completion_sha256",
            "recovery_script_file_sha256",
            "reason",
            "receipt_sha256",
        }:
            raise ComponentClosureEndToEndError(
                f"publication-recovery receipt fields changed: {path}"
            )
        if (
            value["schema_version"] != PUBLICATION_RECOVERY_SCHEMA_VERSION
            or value["status"]
            != "bundle_published_after_conclusive_terminal_success"
            or value["project_commit"] != evaluation["evaluation_source_revision"]
            or value["plan_sha256"] != evaluation["plan_sha256"]
            or value["bundle_sha256"] != bundle["bundle_sha256"]
            or value["bundle_index"] != bundle_index
            or value["panel"] != panel
            or value["seed"] != seed
            or value["resource_id"] != bundle["resource_id"]
            or value["recovery_id"] != recovery_id
            or value["completion_path"] != str(completion_path)
            or value["completion_sha256"] != completion["completion_sha256"]
            or value["reason"] != PUBLICATION_RECOVERY_REASON
            or value["recovery_script_file_sha256"] != script_sha
        ):
            raise ComponentClosureEndToEndError(
                f"publication-recovery receipt identity changed: {path}"
            )

        expected_parent = (
            state_root / "operator-logs" / f"{panel}-seed{seed}"
        ).resolve()
        source_root = Path(str(value["source_log_root"]))
        if (
            not source_root.is_absolute()
            or source_root.is_symlink()
            or not source_root.is_dir()
            or source_root != source_root.resolve()
            or source_root.parent != expected_parent
        ):
            raise ComponentClosureEndToEndError(
                f"publication-recovery log root escaped its bundle: {path}"
            )
        source_logs = value["source_logs"]
        if not isinstance(source_logs, list) or len(source_logs) != 4:
            raise ComponentClosureEndToEndError(
                f"publication-recovery log inventory changed: {path}"
            )
        controllers: set[int] = set()
        processes: set[int] = set()
        validated_logs: list[dict[str, Any]] = []
        for row in source_logs:
            if not isinstance(row, Mapping) or set(row) != {
                "controller",
                "process",
                "path",
                "file_sha256",
                "terminal_status",
            }:
                raise ComponentClosureEndToEndError(
                    f"publication-recovery log fields changed: {path}"
                )
            controller = row["controller"]
            process = row["process"]
            if (
                isinstance(controller, bool)
                or not isinstance(controller, int)
                or isinstance(process, bool)
                or not isinstance(process, int)
                or controller not in range(4)
                or process not in range(4)
                or row["terminal_status"] != 0
            ):
                raise ComponentClosureEndToEndError(
                    f"publication-recovery terminal evidence changed: {path}"
                )
            log_path = source_root / (
                f"bundle-c{controller:05d}-p{process:05d}.log"
            )
            if (
                row["path"] != str(log_path)
                or not isinstance(row["file_sha256"], str)
                or _SHA_RE.fullmatch(row["file_sha256"]) is None
                or _sha256_path(log_path)[0] != row["file_sha256"]
                or _publication_log_terminal_status(
                    log_path,
                    controller=controller,
                    process=process,
                )
                != 0
            ):
                raise ComponentClosureEndToEndError(
                    f"publication-recovery source log changed: {path}"
                )
            controllers.add(controller)
            processes.add(process)
            validated_logs.append(
                {
                    "archive_name": (
                        "state/evaluation/publication-recovery/logs/"
                        f"{panel}-seed{seed}/"
                        f"bundle-c{controller:05d}-p{process:05d}.log"
                    ),
                    "path": log_path,
                    "controller": controller,
                    "process": process,
                    "sha256": row["file_sha256"],
                    "size_bytes": _sha256_path(log_path)[1],
                    "terminal_status": 0,
                }
            )
        if controllers != set(range(4)) or processes != set(range(4)):
            raise ComponentClosureEndToEndError(
                f"publication-recovery worker mapping changed: {path}"
            )
        file_sha, size = _sha256_path(path)
        validated.append(
            {
                "archive_name": (
                    "state/evaluation/publication-recovery/"
                    f"{panel}-seed{seed}.json"
                ),
                "path": path,
                "panel": panel,
                "seed": seed,
                "receipt_sha256": value["receipt_sha256"],
                "completion_sha256": completion["completion_sha256"],
                "file_sha256": file_sha,
                "size_bytes": size,
                "terminal_logs": sorted(
                    validated_logs,
                    key=lambda row: int(row["controller"]),
                ),
                "recovery_script": {
                    "path": recovery_script,
                    "sha256": script_sha,
                    "size_bytes": script_size,
                },
            }
        )
    return validated


def _validated_nullable_sha256(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ComponentClosureEndToEndError(
            f"{label} must be one lowercase SHA-256 or null"
        )
    return value


def _validate_evaluation_metadata_artifacts(
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    roles: set[tuple[str, int, str, int]] = set()
    completions = _validate_evaluation_completions(evaluation)
    expected_centering_fields = {
        "schema_version",
        "protocol_id",
        "panel",
        "config_sha256",
        "target_manifest_sha256",
        "source_manifest_sha256",
        "recomposition_sidecar_sha256",
        "target_mean_sha256",
        "source_mean_sha256",
        "routed_mean_sha256",
        "oracle_shared_mean_sha256",
        "endpoint_receipt_sha256",
        "endpoint_cache_key_sha256",
        "receipt_sha256",
    }
    for bundle, completion_path in zip(
        evaluation["bundles"],
        completions,
        strict=True,
    ):
        completion = _validate_self_hash(
            _load_json(
                completion_path,
                label="evaluation bundle completion",
            ),
            hash_field="completion_sha256",
            label="evaluation bundle completion",
        )
        completion_cells = {
            (
                row.get("logical_condition"),
                row.get("checkpoint_tokens"),
            ): row
            for row in completion["cells"]
            if isinstance(row, Mapping)
        }
        if len(completion_cells) != 12:
            raise ComponentClosureEndToEndError(
                "evaluation completion cell inventory changed"
            )
        for cell in bundle["cells"]:
            panel = str(bundle["panel"])
            seed = int(bundle["seed"])
            condition = str(cell["logical_condition"])
            checkpoint = int(cell["checkpoint_tokens"])
            role = (panel, seed, condition, checkpoint)
            if role in roles:
                raise ComponentClosureEndToEndError(
                    f"evaluation metadata role repeated: {role}"
                )
            roles.add(role)
            completion_cell = completion_cells.get((condition, checkpoint))
            root = Path(str(cell["result_root_controller"]))
            result_path = root / "evaluation-result.json"
            centering_path = root / "centering-receipt.json"
            result = _validate_self_hash(
                _load_json(result_path, label="evaluation result"),
                hash_field="result_sha256",
                label="evaluation result",
            )
            centering = _validate_self_hash(
                _load_json(centering_path, label="centering receipt"),
                hash_field="receipt_sha256",
                label="centering receipt",
            )
            result_file_sha, result_size = _sha256_path(result_path)
            centering_file_sha, centering_size = _sha256_path(centering_path)
            result_source_manifest = _validated_nullable_sha256(
                result.get("source_manifest_sha256"),
                label="evaluation source manifest",
            )
            centering_source_manifest = _validated_nullable_sha256(
                centering.get("source_manifest_sha256"),
                label="centering source manifest",
            )
            if (
                not isinstance(completion_cell, Mapping)
                or completion_cell.get("result_sha256")
                != result["result_sha256"]
                or completion_cell.get("result_file_sha256")
                != result_file_sha
                or result.get("status") != "cell_evaluated"
                or result.get("protocol_id") != cell["source_protocol_id"]
                or result.get("panel") != panel
                or result.get("condition") != cell["source_condition"]
                or result.get("seed") != seed
                or result.get("checkpoint_tokens") != checkpoint
                or result.get("source_revision") != bundle["source_revision"]
                or result.get("parent_training_result_sha256")
                != cell["parent_training_result_sha256"]
                or result.get("parent_checkpoint_pointer_sha256")
                != cell["parent_checkpoint_pointer_sha256"]
                or set(centering) != expected_centering_fields
                or centering.get("receipt_sha256")
                != result.get("centering_receipt_sha256")
                or centering.get("protocol_id") != result.get("protocol_id")
                or centering.get("panel") != panel
                or centering.get("recomposition_sidecar_sha256")
                != result.get("recomposition_sidecar_sha256")
            ):
                raise ComponentClosureEndToEndError(
                    f"evaluation metadata identity changed: {role}"
                )
            prefix = (
                "state/evaluation/cells/"
                f"{panel}/seed{seed}/{condition}/tokens-{checkpoint:08d}"
            )
            artifacts.extend(
                (
                    {
                        "archive_name": f"{prefix}/evaluation-result.json",
                        "path": result_path,
                        "sha256": result_file_sha,
                        "size_bytes": result_size,
                        "semantic_sha256": result["result_sha256"],
                        "source_manifest_sha256": result_source_manifest,
                        "role": role,
                    },
                    {
                        "archive_name": f"{prefix}/centering-receipt.json",
                        "path": centering_path,
                        "sha256": centering_file_sha,
                        "size_bytes": centering_size,
                        "semantic_sha256": centering["receipt_sha256"],
                        "source_manifest_sha256": centering_source_manifest,
                        "role": role,
                    },
                )
            )
    if len(roles) != 72 or len(artifacts) != 144:
        raise ComponentClosureEndToEndError(
            "report recovery requires 72 exact evaluation metadata pairs"
        )
    return artifacts


def _load_sealed_report_recovery_inputs(
    *,
    training: Any,
    paths: DriverPaths,
    project_commit: str,
    input_sha256: Mapping[str, str],
    recovery_script: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    barrier = _validate_training_barrier(training)
    training_stage = _load_stage(
        paths.stage("training"),
        name="training",
        project_commit=project_commit,
        input_sha256=input_sha256,
    )
    if training_stage is None:
        raise ComponentClosureEndToEndError(
            "report recovery requires a sealed training stage"
        )
    _require_stage_outputs(
        training_stage,
        expected=[barrier],
        label="training",
    )

    evaluation_plan_stage = _load_stage(
        paths.stage("evaluation-plan"),
        name="evaluation-plan",
        project_commit=project_commit,
        input_sha256=input_sha256,
    )
    if evaluation_plan_stage is None:
        raise ComponentClosureEndToEndError(
            "report recovery requires a sealed evaluation-plan stage"
        )
    _require_stage_outputs(
        evaluation_plan_stage,
        expected=[paths.evaluation_plan],
        label="evaluation-plan",
    )
    evaluation = load_evaluation_plan(paths.evaluation_plan)
    if (
        evaluation["evaluation_source_revision"] != project_commit
        or evaluation["training_plan_sha256"] != training.plan_sha256
        or Path(str(evaluation["state_root"])).resolve()
        != paths.evaluation_state.resolve()
        or evaluation_plan_stage["metadata"].get("evaluation_plan_sha256")
        != evaluation["plan_sha256"]
    ):
        raise ComponentClosureEndToEndError(
            "sealed evaluation plan does not belong to this recovery"
        )

    completions = _validate_evaluation_completions(evaluation)
    evaluation_stage = _load_stage(
        paths.stage("evaluation"),
        name="evaluation",
        project_commit=project_commit,
        input_sha256=input_sha256,
    )
    if evaluation_stage is None:
        raise ComponentClosureEndToEndError(
            "report recovery requires a sealed evaluation stage"
        )
    _require_stage_outputs(
        evaluation_stage,
        expected=completions,
        label="evaluation",
    )
    receipts = _validate_publication_recovery_receipts(
        evaluation,
        recovery_script=recovery_script,
    )
    evaluation_artifacts = _validate_evaluation_metadata_artifacts(evaluation)
    return (
        evaluation,
        {
            "training": training_stage,
            "evaluation-plan": evaluation_plan_stage,
            "evaluation": evaluation_stage,
        },
        receipts,
        evaluation_artifacts,
    )


def _run_evaluation(
    *,
    training_plan_path: Path,
    evaluation_plan_path: Path,
    evaluation: Mapping[str, Any],
    training: Any,
    paths: DriverPaths,
    clock: Mapping[str, Any],
    input_sha256: Mapping[str, str],
) -> None:
    stage_name = "evaluation"
    if (
        _load_stage(
            paths.stage(stage_name),
            name=stage_name,
            project_commit=training.project_commit,
            input_sha256=input_sha256,
        )
        is not None
    ):
        _validate_evaluation_completions(evaluation)
        return
    expected_completions = [
        Path(evaluation["state_root"])
        / "bundles"
        / f"{bundle['panel']}-seed{bundle['seed']}"
        / "completion.json"
        for bundle in evaluation["bundles"]
    ]
    if all(path.exists() for path in expected_completions):
        completions = _validate_evaluation_completions(evaluation)
        _seal_stage(
            paths.stage(stage_name),
            name=stage_name,
            project_commit=training.project_commit,
            input_sha256=input_sha256,
            outputs=completions,
            metadata={
                "attempt_count": 0,
                "failed_returncodes": [],
                "controller_logs": [],
                "resume_policy": "recovered_existing_immutable_bundle_completions",
            },
        )
        return
    logs: list[Path] = []
    failures: list[int] = []
    for attempt in range(1, MAXIMUM_EVALUATION_ATTEMPTS + 1):
        remaining = _remaining_seconds(clock)
        log_path = paths.log_attempt(stage_name, attempt)
        logs.append(log_path)
        command = [
            str(training.python_executable),
            str(
                training.working_directory / "scripts/tpu/evaluate_component_closure.py"
            ),
            "run-all-operators",
            "--plan",
            str(evaluation_plan_path),
            "--training-plan",
            str(training_plan_path),
            "--maximum-wall-seconds",
            str(remaining),
        ]
        returncode = _run_logged_command(
            command,
            cwd=training.working_directory,
            log_path=log_path,
            timeout_seconds=remaining,
        )
        if returncode == 0:
            break
        failures.append(returncode)
        if attempt == MAXIMUM_EVALUATION_ATTEMPTS:
            raise ComponentClosureEndToEndError(
                "evaluation exhausted four bounded attempts; "
                f"returncodes={failures}, logs={[str(path) for path in logs]}"
            )
        time.sleep(min(30 * attempt, _remaining_seconds(clock)))
    completions = _validate_evaluation_completions(evaluation)
    _seal_stage(
        paths.stage(stage_name),
        name=stage_name,
        project_commit=training.project_commit,
        input_sha256=input_sha256,
        outputs=completions,
        metadata={
            "attempt_count": len(logs),
            "failed_returncodes": failures,
            "controller_logs": [str(path) for path in logs],
            "resume_policy": "immutable_completed_cells_are_reused",
        },
    )


def _expected_report_recovery_receipt(
    *,
    evaluation: Mapping[str, Any],
    project_commit: str,
    report_recovery_commit: str,
    paths: DriverPaths,
    input_sha256: Mapping[str, str],
    bootstrap_replicates: int,
    recovery_stages: Mapping[str, Mapping[str, Any]],
    publication_receipts: Sequence[Mapping[str, Any]],
    recovery_script: Path,
    evaluation_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stage_rows = []
    for name in ("training", "evaluation-plan", "evaluation"):
        stage = recovery_stages.get(name)
        if not isinstance(stage, Mapping):
            raise ComponentClosureEndToEndError(
                f"report recovery lacks its sealed {name} stage"
            )
        stage_rows.append(
            {
                "stage": name,
                "stage_sha256": stage["stage_sha256"],
                "stage_file": _artifact_row(paths.stage(name)),
                "outputs": copy.deepcopy(stage["outputs"]),
            }
        )
    publication_rows = [
        {
            "panel": row["panel"],
            "seed": row["seed"],
            "archive_name": row["archive_name"],
            "receipt_sha256": row["receipt_sha256"],
            "completion_sha256": row["completion_sha256"],
            "receipt_file": _artifact_row(Path(row["path"])),
            "terminal_logs": [
                {
                    "archive_name": log["archive_name"],
                    "controller": log["controller"],
                    "process": log["process"],
                    "terminal_status": log["terminal_status"],
                    "file": _artifact_row(Path(log["path"])),
                }
                for log in row["terminal_logs"]
            ],
        }
        for row in publication_receipts
    ]
    if any(
        row["receipt_file"]["sha256"] != source["file_sha256"]
        or row["receipt_file"]["size_bytes"] != source["size_bytes"]
        for row, source in zip(
            publication_rows,
            publication_receipts,
            strict=True,
        )
    ):
        raise ComponentClosureEndToEndError(
            "publication-recovery receipt changed during report recovery"
        )
    if len(publication_rows) != 6:
        raise ComponentClosureEndToEndError(
            "report recovery receipt requires six publication receipts"
        )
    if sum(len(row["terminal_logs"]) for row in publication_rows) != 24:
        raise ComponentClosureEndToEndError(
            "report recovery receipt requires 24 terminal logs"
        )
    recovery_script_row = _artifact_row(recovery_script)
    if recovery_script_row["sha256"] != PUBLICATION_RECOVERY_SCRIPT_SHA256:
        raise ComponentClosureEndToEndError(
            "report recovery script changed before receipt sealing"
        )
    evaluation_rows = [
        {
            "archive_name": row["archive_name"],
            "role": list(row["role"]),
            "semantic_sha256": row["semantic_sha256"],
            "file": _artifact_row(Path(row["path"])),
        }
        for row in evaluation_artifacts
    ]
    if (
        len(evaluation_rows) != 144
        or any(
            observed["file"]["sha256"] != source["sha256"]
            or observed["file"]["size_bytes"] != source["size_bytes"]
            for observed, source in zip(
                evaluation_rows,
                evaluation_artifacts,
                strict=True,
            )
        )
    ):
        raise ComponentClosureEndToEndError(
            "evaluation metadata changed during report recovery"
        )
    grid = load_score_grid(paths.score_grid)
    report = _validate_self_hash(
        _load_json(paths.report_json, label="closure report"),
        hash_field="report_sha256",
        label="closure report",
    )
    body = {
        "schema_version": REPORT_RECOVERY_SCHEMA_VERSION,
        "record_type": "report_recovery_receipt",
        "status": "report_recovered_from_sealed_training_and_evaluation",
        "scientific_project_commit": project_commit,
        "report_generator_commit": report_recovery_commit,
        "commit_ancestry": {
            "ancestor": project_commit,
            "descendant": report_recovery_commit,
            "command": "git merge-base --is-ancestor",
            "verified": True,
        },
        "reason": REPORT_RECOVERY_REASON,
        "input_sha256": dict(input_sha256),
        "evaluation_plan_sha256": evaluation["plan_sha256"],
        "report_policy": {
            "policy": COMMON_SST_POLICY,
            "reference_condition": "A",
            "aggregate_relative_tolerance": SST_AGGREGATE_RELATIVE_TOLERANCE,
            "normalized_l1_tolerance": SST_NORMALIZED_L1_TOLERANCE,
            "report_schema_version": REPORT_SCHEMA_VERSION,
        },
        "bootstrap_replicates": bootstrap_replicates,
        "sealed_stages": stage_rows,
        "publication_recovery_receipts": publication_rows,
        "publication_recovery_script": recovery_script_row,
        "evaluation_metadata_artifacts": evaluation_rows,
        "score_grid_evidence_sha256": grid["evidence_sha256"],
        "report_sha256": report["report_sha256"],
        "outputs": [
            _artifact_row(path)
            for path in (
                paths.training_costs,
                paths.score_grid,
                paths.report_json,
                paths.report_markdown,
            )
        ],
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _seal_report_recovery_receipt(
    *,
    evaluation: Mapping[str, Any],
    project_commit: str,
    report_recovery_commit: str,
    paths: DriverPaths,
    input_sha256: Mapping[str, str],
    bootstrap_replicates: int,
    recovery_stages: Mapping[str, Mapping[str, Any]],
    publication_receipts: Sequence[Mapping[str, Any]],
    recovery_script: Path,
    evaluation_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = _expected_report_recovery_receipt(
        evaluation=evaluation,
        project_commit=project_commit,
        report_recovery_commit=report_recovery_commit,
        paths=paths,
        input_sha256=input_sha256,
        bootstrap_replicates=bootstrap_replicates,
        recovery_stages=recovery_stages,
        publication_receipts=publication_receipts,
        recovery_script=recovery_script,
        evaluation_artifacts=evaluation_artifacts,
    )
    if paths.report_recovery_receipt.exists():
        observed = _validate_self_hash(
            _load_json(
                paths.report_recovery_receipt,
                label="report-recovery receipt",
            ),
            hash_field="receipt_sha256",
            label="report-recovery receipt",
        )
        if observed != expected:
            raise ComponentClosureEndToEndError(
                "report-recovery receipt differs from sealed evidence"
            )
        return observed
    _seal_json(
        paths.report_recovery_receipt,
        expected,
        label="report-recovery receipt",
    )
    return expected


def _collect_and_report(
    *,
    training_plan: Mapping[str, Any],
    a_references: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    project_commit: str,
    paths: DriverPaths,
    input_sha256: Mapping[str, str],
    bootstrap_replicates: int,
    report_recovery_commit: str | None = None,
    recovery_stages: Mapping[str, Mapping[str, Any]] | None = None,
    publication_receipts: Sequence[Mapping[str, Any]] = (),
    recovery_script: Path | None = None,
    evaluation_artifacts: Sequence[Mapping[str, Any]] = (),
) -> None:
    stage_name = "scientific-report"
    existing = _load_stage(
        paths.stage(stage_name),
        name=stage_name,
        project_commit=project_commit,
        input_sha256=input_sha256,
    )
    if existing is not None:
        _verify_scientific_outputs(paths, bootstrap_replicates=bootstrap_replicates)
        if report_recovery_commit is not None:
            if (
                recovery_stages is None
                or recovery_script is None
                or not paths.report_recovery_receipt.exists()
            ):
                raise ComponentClosureEndToEndError(
                    "sealed recovered report lacks its recovery receipt"
                )
            receipt = _seal_report_recovery_receipt(
                evaluation=evaluation,
                project_commit=project_commit,
                report_recovery_commit=report_recovery_commit,
                paths=paths,
                input_sha256=input_sha256,
                bootstrap_replicates=bootstrap_replicates,
                recovery_stages=recovery_stages,
                publication_receipts=publication_receipts,
                recovery_script=recovery_script,
                evaluation_artifacts=evaluation_artifacts,
            )
            expected_outputs = [
                paths.training_costs,
                paths.score_grid,
                paths.report_json,
                paths.report_markdown,
                paths.report_recovery_receipt,
            ]
            _require_stage_outputs(
                existing,
                expected=expected_outputs,
                label=stage_name,
            )
            if existing["metadata"] != {
                "bootstrap_replicates": bootstrap_replicates,
                "report_sha256": _load_json(
                    paths.report_json,
                    label="closure report",
                )["report_sha256"],
                "report_generator_commit": report_recovery_commit,
                "report_policy": COMMON_SST_POLICY,
                "report_recovery_receipt_sha256": receipt["receipt_sha256"],
            }:
                raise ComponentClosureEndToEndError(
                    "scientific-report recovery metadata changed"
                )
        return
    costs = collect_training_costs(training_plan, a_references)
    _seal_json(paths.training_costs, {"costs": costs}, label="training costs")
    costs_value = _load_json(paths.training_costs, label="training costs")
    if set(costs_value) != {"costs"} or costs_value["costs"] != costs:
        raise ComponentClosureEndToEndError("training costs changed after publication")
    grid = collect_score_grid(evaluation, costs)
    _seal_json(paths.score_grid, grid, label="score grid")
    checked_grid = load_score_grid(paths.score_grid)
    report = build_closure_report(
        checked_grid,
        bootstrap_replicates=bootstrap_replicates,
    )
    _seal_json(paths.report_json, report, label="closure report")
    markdown = (render_markdown(report) + "\n").encode("utf-8")
    _seal_exact_bytes(paths.report_markdown, markdown, label="closure Markdown")
    _verify_scientific_outputs(paths, bootstrap_replicates=bootstrap_replicates)
    outputs = [
        paths.training_costs,
        paths.score_grid,
        paths.report_json,
        paths.report_markdown,
    ]
    metadata = {
        "bootstrap_replicates": bootstrap_replicates,
        "report_sha256": report["report_sha256"],
    }
    if report_recovery_commit is not None:
        if recovery_stages is None or recovery_script is None:
            raise ComponentClosureEndToEndError(
                "report recovery lacks its sealed-stage inventory"
            )
        receipt = _seal_report_recovery_receipt(
            evaluation=evaluation,
            project_commit=project_commit,
            report_recovery_commit=report_recovery_commit,
            paths=paths,
            input_sha256=input_sha256,
            bootstrap_replicates=bootstrap_replicates,
            recovery_stages=recovery_stages,
            publication_receipts=publication_receipts,
            recovery_script=recovery_script,
            evaluation_artifacts=evaluation_artifacts,
        )
        outputs.append(paths.report_recovery_receipt)
        metadata.update(
            {
                "report_generator_commit": report_recovery_commit,
                "report_policy": COMMON_SST_POLICY,
                "report_recovery_receipt_sha256": receipt["receipt_sha256"],
            }
        )
    _seal_stage(
        paths.stage(stage_name),
        name=stage_name,
        project_commit=project_commit,
        input_sha256=input_sha256,
        outputs=outputs,
        metadata=metadata,
    )


def _verify_scientific_outputs(
    paths: DriverPaths,
    *,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    costs = _load_json(paths.training_costs, label="training costs")
    if set(costs) != {"costs"} or not isinstance(costs["costs"], list):
        raise ComponentClosureEndToEndError("training-cost fields changed")
    grid = load_score_grid(paths.score_grid)
    if grid["costs"] != costs["costs"]:
        raise ComponentClosureEndToEndError("score grid and cost file differ")
    expected = build_closure_report(
        grid,
        bootstrap_replicates=bootstrap_replicates,
    )
    observed = _validate_self_hash(
        _load_json(paths.report_json, label="closure report"),
        hash_field="report_sha256",
        label="closure report",
    )
    if observed != expected:
        raise ComponentClosureEndToEndError("closure report is not reproducible")
    expected_markdown = (render_markdown(expected) + "\n").encode("utf-8")
    if paths.report_markdown.read_bytes() != expected_markdown:
        raise ComponentClosureEndToEndError("closure Markdown changed")
    return observed


def _archive_sources(
    *,
    training_plan_path: Path,
    a_references_path: Path,
    evaluation_runtime_path: Path,
    training_state_root: Path,
    evaluation: Mapping[str, Any],
    paths: DriverPaths,
    publication_receipts: Sequence[Mapping[str, Any]] = (),
    report_recovery_receipt: Path | None = None,
    recovery_script: Path | None = None,
    evaluation_artifacts: Sequence[Mapping[str, Any]] = (),
) -> list[tuple[str, Path]]:
    sources = [
        ("inputs/training-plan" + training_plan_path.suffix, training_plan_path),
        ("inputs/a-references" + a_references_path.suffix, a_references_path),
        (
            "inputs/evaluation-runtime" + evaluation_runtime_path.suffix,
            evaluation_runtime_path,
        ),
        ("outputs/evaluation-plan.json", paths.evaluation_plan),
        ("outputs/training-costs.json", paths.training_costs),
        ("outputs/score-grid.json", paths.score_grid),
        ("outputs/report.json", paths.report_json),
        ("outputs/report.md", paths.report_markdown),
        (
            "state/training-complete.json",
            training_state_root / "training-complete.json",
        ),
        ("state/run-clock.json", paths.clock),
    ]
    for bundle in evaluation["bundles"]:
        completion = (
            Path(evaluation["state_root"])
            / "bundles"
            / f"{bundle['panel']}-seed{bundle['seed']}"
            / "completion.json"
        )
        sources.append(
            (
                "state/evaluation/"
                f"{bundle['panel']}-seed{bundle['seed']}-completion.json",
                completion,
            )
        )
    for receipt in publication_receipts:
        sources.append((str(receipt["archive_name"]), Path(receipt["path"])))
        for log in receipt["terminal_logs"]:
            sources.append((str(log["archive_name"]), Path(log["path"])))
    if recovery_script is not None:
        sources.append(
            (
                "state/evaluation/publication-recovery/recovery-script.py",
                recovery_script,
            )
        )
    for artifact in evaluation_artifacts:
        sources.append(
            (
                str(artifact["archive_name"]),
                Path(artifact["path"]),
            )
        )
    if report_recovery_receipt is not None:
        sources.append(
            (
                "state/report-recovery/receipt.json",
                report_recovery_receipt,
            )
        )
    for stage in (
        "training",
        "evaluation-plan",
        "evaluation",
        "scientific-report",
    ):
        sources.append((f"state/stages/{stage}.json", paths.stage(stage)))
    return sources


def _tar_add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    from io import BytesIO

    archive.addfile(info, BytesIO(payload))


def _build_compact_archive(
    *,
    sources: Sequence[tuple[str, Path]],
    project_commit: str,
    paths: DriverPaths,
) -> tuple[Path, dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    payloads: list[tuple[str, bytes]] = []
    total_bytes = 0
    names: set[str] = set()
    for name, path in sources:
        if (
            name.startswith("/")
            or ".." in Path(name).parts
            or name in names
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ComponentClosureEndToEndError(
                f"unsafe compact-archive member: {name}"
            )
        payload = path.read_bytes()
        total_bytes += len(payload)
        if total_bytes > MAXIMUM_COMPACT_ARCHIVE_BYTES:
            raise ComponentClosureEndToEndError(
                "compact evidence exceeds 256 MiB; bulk data is not archiveable"
            )
        names.add(name)
        payloads.append((name, payload))
        entries.append(
            {
                "name": name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "compact_archive_index",
        "project_commit": project_commit,
        "bulk_capture_or_checkpoint_data_included": False,
        "entries": entries,
        "total_uncompressed_bytes": total_bytes,
    }
    index = {**body, "index_sha256": canonical_sha256(body)}
    index_payload = _canonical_json_bytes(index)
    _seal_exact_bytes(paths.archive_index, index_payload, label="archive index")

    temporary = (
        paths.output_root / f".component-closure-archive.{uuid.uuid4().hex}.tar.gz.tmp"
    )
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with tarfile.open(
                    mode="w", fileobj=zipped, format=tarfile.PAX_FORMAT
                ) as tar:
                    for name, payload in payloads:
                        _tar_add_bytes(tar, name, payload)
                    _tar_add_bytes(tar, "archive-index.json", index_payload)
            raw.flush()
            os.fsync(raw.fileno())
        archive_sha, archive_size = _sha256_path(temporary)
        if archive_size > MAXIMUM_COMPACT_ARCHIVE_BYTES:
            raise ComponentClosureEndToEndError(
                "compressed archive exceeds the 256 MiB compact boundary"
            )
        destination = paths.output_root / f"component-closure-{archive_sha}.tar.gz"
        if destination.exists():
            observed_sha, observed_size = _sha256_path(destination)
            if observed_sha != archive_sha or observed_size != archive_size:
                raise ComponentClosureEndToEndError(
                    "content-addressed archive destination changed"
                )
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                observed_sha, observed_size = _sha256_path(destination)
                if observed_sha != archive_sha or observed_size != archive_size:
                    raise ComponentClosureEndToEndError(
                        "raced archive destination changed"
                    )
        return destination, index
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def _capture_command(
    command: Sequence[str],
    *,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ComponentClosureEndToEndError(
            f"command failed before completion: {command[:3]}"
        ) from error


def _gcs_describe(
    *,
    executable: str,
    project: str,
    uri: str,
    timeout_seconds: int,
) -> dict[str, Any] | None:
    result = _capture_command(
        [
            executable,
            "storage",
            "objects",
            "describe",
            uri,
            f"--project={project}",
            "--format=json",
        ],
        timeout_seconds=timeout_seconds,
    )
    if result.returncode != 0:
        return None
    try:
        value = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComponentClosureEndToEndError(
            f"GCS describe response is invalid: {uri}"
        ) from error
    if not isinstance(value, dict):
        raise ComponentClosureEndToEndError(
            f"GCS describe response is not an object: {uri}"
        )
    return value


def _validate_gcs_identity(
    value: Mapping[str, Any],
    *,
    uri: str,
    sha256: str,
    size_bytes: int,
) -> dict[str, Any]:
    raw_size = value.get("size", value.get("size_bytes"))
    metadata = value.get("metadata", value.get("custom_fields"))
    if (
        isinstance(raw_size, bool)
        or not isinstance(raw_size, (str, int))
        or int(raw_size) != size_bytes
        or not isinstance(metadata, Mapping)
        or metadata.get("sha256") != sha256
        or not isinstance(value.get("generation"), (str, int))
    ):
        raise ComponentClosureEndToEndError(f"GCS object identity changed: {uri}")
    return {
        "uri": uri,
        "generation": int(value["generation"]),
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _upload_gcs_object(
    *,
    path: Path,
    uri: str,
    executable: str,
    project: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds

    def remaining() -> int:
        value = int(deadline - time.monotonic())
        if value <= 0:
            raise ComponentClosureEndToEndError(
                f"compact GCS upload reached its wall: {uri}"
            )
        return value

    digest, size = _sha256_path(path)
    observed = _gcs_describe(
        executable=executable,
        project=project,
        uri=uri,
        timeout_seconds=remaining(),
    )
    if observed is not None:
        return _validate_gcs_identity(
            observed,
            uri=uri,
            sha256=digest,
            size_bytes=size,
        )
    copy_command = [
        executable,
        "storage",
        "cp",
        str(path),
        uri,
        f"--project={project}",
        "--if-generation-match=0",
        f"--custom-metadata=sha256={digest}",
        "--quiet",
    ]
    copy_error: ComponentClosureEndToEndError | None = None
    try:
        result = _capture_command(
            copy_command,
            timeout_seconds=remaining(),
        )
    except ComponentClosureEndToEndError as error:
        copy_error = error
        result = None
    if result is not None and result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-2000:]
        copy_error = ComponentClosureEndToEndError(
            f"compact GCS upload failed for {uri}: {detail}"
        )
    observed = _gcs_describe(
        executable=executable,
        project=project,
        uri=uri,
        timeout_seconds=remaining(),
    )
    if observed is None:
        if copy_error is not None:
            raise copy_error
        raise ComponentClosureEndToEndError(
            f"uploaded GCS object cannot be verified: {uri}"
        )
    return _validate_gcs_identity(
        observed,
        uri=uri,
        sha256=digest,
        size_bytes=size,
    )


def _compact_source_inventory(
    sources: Sequence[tuple[str, Path]],
) -> tuple[list[dict[str, Any]], str]:
    rows = []
    names: set[str] = set()
    for name, path in sources:
        if (
            name.startswith("/")
            or ".." in Path(name).parts
            or name in names
        ):
            raise ComponentClosureEndToEndError(
                f"unsafe compact source inventory member: {name}"
            )
        names.add(name)
        digest, size = _sha256_path(path)
        rows.append(
            {
                "name": name,
                "size_bytes": size,
                "sha256": digest,
            }
        )
    return rows, canonical_sha256({"entries": rows})


def _validate_compact_archive_against_inventory(
    *,
    archive: Path,
    index_path: Path,
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    index = _validate_self_hash(
        _load_json(index_path, label="compact archive index"),
        hash_field="index_sha256",
        label="compact archive index",
    )
    if index.get("entries") != list(inventory):
        raise ComponentClosureEndToEndError(
            "compact archive index differs from current source inventory"
        )
    expected_names = [str(row["name"]) for row in inventory]
    expected = {str(row["name"]): row for row in inventory}
    try:
        with tarfile.open(archive, "r:gz") as handle:
            if handle.getnames() != [*expected_names, "archive-index.json"]:
                raise ComponentClosureEndToEndError(
                    "compact archive member inventory changed"
                )
            for name in expected_names:
                member = handle.extractfile(name)
                if member is None:
                    raise ComponentClosureEndToEndError(
                        f"compact archive member is unavailable: {name}"
                    )
                payload = member.read()
                row = expected[name]
                if (
                    len(payload) != row["size_bytes"]
                    or hashlib.sha256(payload).hexdigest() != row["sha256"]
                ):
                    raise ComponentClosureEndToEndError(
                        f"compact archive member changed: {name}"
                    )
            index_member = handle.extractfile("archive-index.json")
            if (
                index_member is None
                or index_member.read() != index_path.read_bytes()
            ):
                raise ComponentClosureEndToEndError(
                    "embedded compact archive index changed"
                )
    except (OSError, tarfile.TarError) as error:
        raise ComponentClosureEndToEndError(
            "cannot validate compact archive contents"
        ) from error
    return index


def _archive_and_upload(
    *,
    sources: Sequence[tuple[str, Path]],
    project_commit: str,
    project: str,
    archive_gcs_prefix: str,
    gcloud_executable: str,
    paths: DriverPaths,
    clock: Mapping[str, Any],
    input_sha256: Mapping[str, str],
    report_recovery_commit: str | None = None,
    report_recovery_receipt: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    stage_name = "compact-archive"
    source_inventory, source_inventory_sha256 = _compact_source_inventory(
        sources
    )
    recovery_receipt = None
    if report_recovery_commit is not None:
        if report_recovery_receipt is None:
            raise ComponentClosureEndToEndError(
                "compact recovery lacks its report-recovery receipt"
            )
        recovery_receipt = _validate_self_hash(
            _load_json(
                report_recovery_receipt,
                label="report-recovery receipt",
            ),
            hash_field="receipt_sha256",
            label="report-recovery receipt",
        )
    existing = _load_stage(
        paths.stage(stage_name),
        name=stage_name,
        project_commit=project_commit,
        input_sha256=input_sha256,
    )
    if existing is not None:
        manifest = _validate_self_hash(
            _load_json(paths.upload_manifest, label="upload manifest"),
            hash_field="manifest_sha256",
            label="upload manifest",
        )
        archive = Path(manifest["archive"]["local_path"])
        archive_sha, archive_size = _sha256_path(archive)
        if archive_sha != manifest["archive"].get("sha256") or archive_size != manifest[
            "archive"
        ].get("size_bytes"):
            raise ComponentClosureEndToEndError(
                "local archive differs from its upload manifest"
            )
        index = _validate_compact_archive_against_inventory(
            archive=archive,
            index_path=paths.archive_index,
            inventory=source_inventory,
        )
        metadata = existing["metadata"]
        if report_recovery_commit is not None:
            assert recovery_receipt is not None
            if (
                metadata.get("source_inventory_sha256")
                != source_inventory_sha256
                or metadata.get("archive_index_sha256")
                != index["index_sha256"]
                or metadata.get("report_generator_commit")
                != report_recovery_commit
                or metadata.get("report_recovery_receipt_sha256")
                != recovery_receipt["receipt_sha256"]
                or manifest.get("source_inventory_sha256")
                != source_inventory_sha256
                or manifest.get("report_generator_commit")
                != report_recovery_commit
                or manifest.get("report_recovery_receipt_sha256")
                != recovery_receipt["receipt_sha256"]
            ):
                raise ComponentClosureEndToEndError(
                    "sealed compact recovery identity changed"
                )
        for name, path in (
            ("archive_object", archive),
            ("manifest_object", paths.upload_manifest),
        ):
            identity = metadata.get(name)
            if not isinstance(identity, Mapping):
                raise ComponentClosureEndToEndError(
                    f"compact archive stage lacks {name}"
                )
            observed = _gcs_describe(
                executable=gcloud_executable,
                project=project,
                uri=str(identity.get("uri")),
                timeout_seconds=_remaining_seconds(clock),
            )
            if observed is None:
                raise ComponentClosureEndToEndError(
                    f"sealed compact GCS object is unavailable: {identity.get('uri')}"
                )
            digest, size = _sha256_path(path)
            if (
                _validate_gcs_identity(
                    observed,
                    uri=str(identity.get("uri")),
                    sha256=digest,
                    size_bytes=size,
                )
                != identity
            ):
                raise ComponentClosureEndToEndError(
                    f"sealed compact GCS identity changed: {identity.get('uri')}"
                )
        _validate_artifact_rows(
            existing["outputs"],
            label="compact archive stage",
        )
        return manifest, archive

    archive, index = _build_compact_archive(
        sources=sources,
        project_commit=project_commit,
        paths=paths,
    )
    if index["entries"] != source_inventory:
        raise ComponentClosureEndToEndError(
            "new compact archive differs from current source inventory"
        )
    archive_sha, archive_size = _sha256_path(archive)
    prefix = archive_gcs_prefix.rstrip("/")
    archive_uri = f"{prefix}/component-closure-{archive_sha}.tar.gz"
    archive_object = _upload_gcs_object(
        path=archive,
        uri=archive_uri,
        executable=gcloud_executable,
        project=project,
        timeout_seconds=_remaining_seconds(clock),
    )
    report = _load_json(paths.report_json, label="closure report")
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "compact_archive_upload",
        "project_commit": project_commit,
        "archive_scope": "compact_control_and_scientific_evidence_only",
        "bulk_capture_or_checkpoint_data_uploaded": False,
        "archive": {
            **archive_object,
            "local_path": str(archive),
        },
        "archive_index_sha256": index["index_sha256"],
        "report_sha256": report["report_sha256"],
    }
    if report_recovery_commit is not None:
        assert recovery_receipt is not None
        body.update(
            {
                "source_inventory_sha256": source_inventory_sha256,
                "report_generator_commit": report_recovery_commit,
                "report_recovery_receipt_sha256": (
                    recovery_receipt["receipt_sha256"]
                ),
            }
        )
    manifest = {**body, "manifest_sha256": canonical_sha256(body)}
    _seal_json(paths.upload_manifest, manifest, label="upload manifest")
    manifest_sha, _ = _sha256_path(paths.upload_manifest)
    manifest_uri = f"{prefix}/component-closure-{archive_sha}.manifest.json"
    manifest_object = _upload_gcs_object(
        path=paths.upload_manifest,
        uri=manifest_uri,
        executable=gcloud_executable,
        project=project,
        timeout_seconds=_remaining_seconds(clock),
    )
    metadata = {
        "archive_object": archive_object,
        "manifest_object": manifest_object,
        "manifest_file_sha256": manifest_sha,
        "uploaded_bulk_data": False,
    }
    if report_recovery_commit is not None:
        assert recovery_receipt is not None
        metadata.update(
            {
                "source_inventory_sha256": source_inventory_sha256,
                "archive_index_sha256": index["index_sha256"],
                "report_generator_commit": report_recovery_commit,
                "report_recovery_receipt_sha256": (
                    recovery_receipt["receipt_sha256"]
                ),
            }
        )
    _seal_stage(
        paths.stage(stage_name),
        name=stage_name,
        project_commit=project_commit,
        input_sha256=input_sha256,
        outputs=[archive, paths.archive_index, paths.upload_manifest],
        metadata=metadata,
    )
    return manifest, archive


def _validate_terminal(
    path: Path,
    *,
    paths: DriverPaths,
    project_commit: str,
    input_sha256: Mapping[str, str],
    report_recovery_commit: str | None = None,
    report_recovery_receipt: Path | None = None,
) -> dict[str, Any]:
    value = _validate_self_hash(
        _load_json(path, label="terminal completion"),
        hash_field="completion_sha256",
        label="terminal completion",
    )
    normal_fields = {
        "schema_version",
        "record_type",
        "status",
        "project_commit",
        "input_sha256",
        "clock_sha256",
        "maximum_wall_seconds",
        "elapsed_wall_seconds",
        "report_sha256",
        "archive_manifest_sha256",
        "archive_object",
        "manifest_object",
        "bulk_capture_or_checkpoint_data_uploaded",
        "outputs",
        "completed_at",
        "completion_sha256",
    }
    recovery_fields = {
        *normal_fields,
        "report_generator_commit",
        "report_recovery_receipt_sha256",
    }
    expected_fields = (
        recovery_fields
        if report_recovery_commit is not None
        else normal_fields
    )
    if (
        set(value) != expected_fields
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("record_type") != "terminal_completion"
        or value.get("status") != "completed"
        or value.get("project_commit") != project_commit
        or value.get("input_sha256") != dict(input_sha256)
        or value.get("bulk_capture_or_checkpoint_data_uploaded") is not False
    ):
        raise ComponentClosureEndToEndError("terminal completion identity changed")
    report = _validate_self_hash(
        _load_json(paths.report_json, label="closure report"),
        hash_field="report_sha256",
        label="closure report",
    )
    manifest = _validate_self_hash(
        _load_json(paths.upload_manifest, label="upload manifest"),
        hash_field="manifest_sha256",
        label="upload manifest",
    )
    clock = _validate_self_hash(
        _load_json(paths.clock, label="run clock"),
        hash_field="clock_sha256",
        label="run clock",
    )
    archive_stage = _load_stage(
        paths.stage("compact-archive"),
        name="compact-archive",
        project_commit=project_commit,
        input_sha256=input_sha256,
    )
    if archive_stage is None:
        raise ComponentClosureEndToEndError(
            "terminal completion lacks a compact-archive stage"
        )
    archive = Path(str(manifest.get("archive", {}).get("local_path")))
    _require_stage_outputs(
        archive_stage,
        expected=[archive, paths.archive_index, paths.upload_manifest],
        label="compact-archive",
    )
    metadata = archive_stage["metadata"]
    expected_outputs = [
        _artifact_row(output)
        for output in (
            paths.evaluation_plan,
            paths.training_costs,
            paths.score_grid,
            paths.report_json,
            paths.report_markdown,
            paths.archive_index,
            archive,
            paths.upload_manifest,
        )
    ]
    if report_recovery_commit is not None:
        if report_recovery_receipt is None:
            raise ComponentClosureEndToEndError(
                "terminal recovery lacks its receipt path"
            )
        receipt = _validate_self_hash(
            _load_json(
                report_recovery_receipt,
                label="report-recovery receipt",
            ),
            hash_field="receipt_sha256",
            label="report-recovery receipt",
        )
        expected_outputs.append(_artifact_row(report_recovery_receipt))
        if (
            value.get("report_generator_commit")
            != report_recovery_commit
            or value.get("report_recovery_receipt_sha256")
            != receipt["receipt_sha256"]
            or receipt.get("report_generator_commit")
            != report_recovery_commit
            or receipt.get("report_sha256") != report["report_sha256"]
            or metadata.get("report_generator_commit")
            != report_recovery_commit
            or metadata.get("report_recovery_receipt_sha256")
            != receipt["receipt_sha256"]
            or manifest.get("report_generator_commit")
            != report_recovery_commit
            or manifest.get("report_recovery_receipt_sha256")
            != receipt["receipt_sha256"]
        ):
            raise ComponentClosureEndToEndError(
                "terminal completion lacks its report-recovery receipt"
            )
    if (
        value["clock_sha256"] != clock["clock_sha256"]
        or value["maximum_wall_seconds"] != clock["maximum_wall_seconds"]
        or value["report_sha256"] != report["report_sha256"]
        or value["archive_manifest_sha256"] != manifest["manifest_sha256"]
        or manifest.get("report_sha256") != report["report_sha256"]
        or value["archive_object"] != metadata.get("archive_object")
        or value["manifest_object"] != metadata.get("manifest_object")
        or value["outputs"] != expected_outputs
        or not isinstance(value["elapsed_wall_seconds"], (int, float))
        or isinstance(value["elapsed_wall_seconds"], bool)
        or not math.isfinite(float(value["elapsed_wall_seconds"]))
        or value["elapsed_wall_seconds"] < 0
        or not isinstance(value["completed_at"], str)
    ):
        raise ComponentClosureEndToEndError(
            "terminal semantic linkage changed"
        )
    _validate_artifact_rows(value["outputs"], label="terminal completion")
    return value


def _seal_terminal(
    *,
    paths: DriverPaths,
    project_commit: str,
    input_sha256: Mapping[str, str],
    clock: Mapping[str, Any],
    archive_manifest: Mapping[str, Any],
    archive: Path,
    archive_stage: Mapping[str, Any],
    report_recovery_commit: str | None = None,
    report_recovery_receipt: Path | None = None,
) -> dict[str, Any]:
    metadata = archive_stage["metadata"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "terminal_completion",
        "status": "completed",
        "project_commit": project_commit,
        "input_sha256": dict(input_sha256),
        "clock_sha256": clock["clock_sha256"],
        "maximum_wall_seconds": clock["maximum_wall_seconds"],
        "elapsed_wall_seconds": max(
            0.0, time.time() - float(clock["started_at_unix_seconds"])
        ),
        "report_sha256": _load_json(paths.report_json, label="closure report")[
            "report_sha256"
        ],
        "archive_manifest_sha256": archive_manifest["manifest_sha256"],
        "archive_object": metadata["archive_object"],
        "manifest_object": metadata["manifest_object"],
        "bulk_capture_or_checkpoint_data_uploaded": False,
        "outputs": [
            _artifact_row(path)
            for path in (
                paths.evaluation_plan,
                paths.training_costs,
                paths.score_grid,
                paths.report_json,
                paths.report_markdown,
                paths.archive_index,
                archive,
                paths.upload_manifest,
            )
        ],
        "completed_at": _timestamp(),
    }
    if report_recovery_commit is not None:
        if report_recovery_receipt is None:
            raise ComponentClosureEndToEndError(
                "terminal recovery lacks its receipt"
            )
        receipt = _validate_self_hash(
            _load_json(
                report_recovery_receipt,
                label="report-recovery receipt",
            ),
            hash_field="receipt_sha256",
            label="report-recovery receipt",
        )
        body["report_generator_commit"] = report_recovery_commit
        body["report_recovery_receipt_sha256"] = receipt["receipt_sha256"]
        body["outputs"].append(_artifact_row(report_recovery_receipt))
    value = {**body, "completion_sha256": canonical_sha256(body)}
    _seal_json(paths.terminal_completion, value, label="terminal completion")
    return _validate_terminal(
        paths.terminal_completion,
        paths=paths,
        project_commit=project_commit,
        input_sha256=input_sha256,
        report_recovery_commit=report_recovery_commit,
        report_recovery_receipt=report_recovery_receipt,
    )


def _input_hashes(
    *,
    training_plan: Path,
    a_references: Path,
    evaluation_runtime: Path,
    archive_gcs_prefix: str,
    bootstrap_replicates: int,
) -> dict[str, str]:
    return {
        "training_plan_file": _sha256_path(training_plan)[0],
        "a_references_file": _sha256_path(a_references)[0],
        "evaluation_runtime_file": _sha256_path(evaluation_runtime)[0],
        "archive_gcs_prefix": hashlib.sha256(
            archive_gcs_prefix.encode("utf-8")
        ).hexdigest(),
        "bootstrap_replicates": hashlib.sha256(
            str(bootstrap_replicates).encode("ascii")
        ).hexdigest(),
    }


def _validate_cli_identity(
    args: argparse.Namespace,
    *,
    training: Any,
) -> tuple[str, str]:
    if _GIT_RE.fullmatch(args.project_commit) is None:
        raise ComponentClosureEndToEndError(
            "project commit must be one full lowercase Git SHA"
        )
    if training.project_commit != args.project_commit:
        raise ComponentClosureEndToEndError(
            "training plan belongs to another project commit"
        )
    if (
        not isinstance(args.archive_gcs_prefix, str)
        or _GCS_RE.fullmatch(args.archive_gcs_prefix.rstrip("/")) is None
    ):
        raise ComponentClosureEndToEndError(
            "archive prefix must be one safe gs:// path"
        )
    projects = {resource.project for resource in training.resources}
    if len(projects) != 1:
        raise ComponentClosureEndToEndError(
            "all five resources must use one GCP project"
        )
    if isinstance(args.bootstrap_replicates, bool) or args.bootstrap_replicates < 100:
        raise ComponentClosureEndToEndError(
            "bootstrap requires at least 100 replicates"
        )
    return projects.pop(), args.archive_gcs_prefix.rstrip("/")


def run(args: argparse.Namespace) -> dict[str, Any]:
    training_plan_path = args.training_plan.resolve()
    a_references_path = args.a_references.resolve()
    evaluation_runtime_path = args.evaluation_runtime.resolve()
    training_plan = load_training_plan(training_plan_path)
    training = parse_training_campaign(training_plan)
    a_references = load_a_reference_manifest(a_references_path)
    project, archive_prefix = _validate_cli_identity(args, training=training)
    report_recovery_commit = getattr(args, "report_recovery_commit", None)
    raw_recovery_script = getattr(args, "report_recovery_script", None)
    if (report_recovery_commit is None) is not (raw_recovery_script is None):
        raise ComponentClosureEndToEndError(
            "report recovery commit and script must be supplied together"
        )
    report_recovery_script = (
        raw_recovery_script.expanduser().resolve()
        if raw_recovery_script is not None
        else None
    )
    if report_recovery_commit is None:
        _validate_checkout(training)
    else:
        _validate_report_recovery_checkout(
            training,
            report_recovery_commit=report_recovery_commit,
        )
    try:
        runtime = yaml.safe_load(evaluation_runtime_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ComponentClosureEndToEndError("cannot read evaluation runtime") from error
    if not isinstance(runtime, Mapping):
        raise ComponentClosureEndToEndError("evaluation runtime is not a mapping")
    paths = DriverPaths(
        output_root=args.output_root.resolve(),
        state_root=args.state_root.resolve(),
    )
    _validate_roots(paths, checkout=training.working_directory)
    input_sha256 = _input_hashes(
        training_plan=training_plan_path,
        a_references=a_references_path,
        evaluation_runtime=evaluation_runtime_path,
        archive_gcs_prefix=archive_prefix,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    with _exclusive_driver_lock(paths.state_root):
        clock = _bind_clock(
            paths.clock,
            project_commit=args.project_commit,
            maximum_wall_seconds=args.maximum_wall_seconds,
            input_sha256=input_sha256,
        )
        recovery_stages: dict[str, dict[str, Any]] | None = None
        publication_receipts: list[dict[str, Any]] = []
        evaluation_artifacts: list[dict[str, Any]] = []
        if report_recovery_commit is None:
            if paths.terminal_completion.exists():
                return _validate_terminal(
                    paths.terminal_completion,
                    paths=paths,
                    project_commit=args.project_commit,
                    input_sha256=input_sha256,
                )
            _remaining_seconds(clock)
            _run_training(
                training_plan_path=training_plan_path,
                training=training,
                paths=paths,
                clock=clock,
                input_sha256=input_sha256,
            )
            evaluation = _build_eval_plan(
                training_plan=training_plan,
                a_references=a_references,
                runtime=runtime,
                project_commit=args.project_commit,
                paths=paths,
                input_sha256=input_sha256,
            )
            _run_evaluation(
                training_plan_path=training_plan_path,
                evaluation_plan_path=paths.evaluation_plan,
                evaluation=evaluation,
                training=training,
                paths=paths,
                clock=clock,
                input_sha256=input_sha256,
            )
        else:
            (
                evaluation,
                recovery_stages,
                publication_receipts,
                evaluation_artifacts,
            ) = _load_sealed_report_recovery_inputs(
                training=training,
                paths=paths,
                project_commit=args.project_commit,
                input_sha256=input_sha256,
                recovery_script=report_recovery_script,
            )
            if paths.terminal_completion.exists():
                _collect_and_report(
                    training_plan=training_plan,
                    a_references=a_references,
                    evaluation=evaluation,
                    project_commit=args.project_commit,
                    paths=paths,
                    input_sha256=input_sha256,
                    bootstrap_replicates=args.bootstrap_replicates,
                    report_recovery_commit=report_recovery_commit,
                    recovery_stages=recovery_stages,
                    publication_receipts=publication_receipts,
                    recovery_script=report_recovery_script,
                    evaluation_artifacts=evaluation_artifacts,
                )
                return _validate_terminal(
                    paths.terminal_completion,
                    paths=paths,
                    project_commit=args.project_commit,
                    input_sha256=input_sha256,
                    report_recovery_commit=report_recovery_commit,
                    report_recovery_receipt=paths.report_recovery_receipt,
                )
            _remaining_seconds(clock)
        _collect_and_report(
            training_plan=training_plan,
            a_references=a_references,
            evaluation=evaluation,
            project_commit=args.project_commit,
            paths=paths,
            input_sha256=input_sha256,
            bootstrap_replicates=args.bootstrap_replicates,
            report_recovery_commit=report_recovery_commit,
            recovery_stages=recovery_stages,
            publication_receipts=publication_receipts,
            recovery_script=report_recovery_script,
            evaluation_artifacts=evaluation_artifacts,
        )
        sources = _archive_sources(
            training_plan_path=training_plan_path,
            a_references_path=a_references_path,
            evaluation_runtime_path=evaluation_runtime_path,
            training_state_root=training.state_root,
            evaluation=evaluation,
            paths=paths,
            publication_receipts=publication_receipts,
            report_recovery_receipt=(
                paths.report_recovery_receipt
                if report_recovery_commit is not None
                else None
            ),
            recovery_script=report_recovery_script,
            evaluation_artifacts=evaluation_artifacts,
        )
        archive_manifest, archive = _archive_and_upload(
            sources=sources,
            project_commit=args.project_commit,
            project=project,
            archive_gcs_prefix=archive_prefix,
            gcloud_executable=args.gcloud_executable,
            paths=paths,
            clock=clock,
            input_sha256=input_sha256,
            report_recovery_commit=report_recovery_commit,
            report_recovery_receipt=(
                paths.report_recovery_receipt
                if report_recovery_commit is not None
                else None
            ),
        )
        archive_stage = _load_stage(
            paths.stage("compact-archive"),
            name="compact-archive",
            project_commit=args.project_commit,
            input_sha256=input_sha256,
        )
        assert archive_stage is not None
        _remaining_seconds(clock)
        return _seal_terminal(
            paths=paths,
            project_commit=args.project_commit,
            input_sha256=input_sha256,
            clock=clock,
            archive_manifest=archive_manifest,
            archive=archive,
            archive_stage=archive_stage,
            report_recovery_commit=report_recovery_commit,
            report_recovery_receipt=(
                paths.report_recovery_receipt
                if report_recovery_commit is not None
                else None
            ),
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        completion = run(args)
    except ComponentClosureEndToEndError as error:
        print(f"component-closure-end-to-end: {error}", file=sys.stderr, flush=True)
        return 1
    print(_canonical_json_bytes(completion).decode("ascii"), end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
