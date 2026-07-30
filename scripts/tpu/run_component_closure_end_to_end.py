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
    build_closure_report,
    load_score_grid,
    render_markdown,
)
from moe_sae.contribution_artifacts import canonical_sha256


SCHEMA_VERSION = "gemma4-a512-component-closure-end-to-end-v1"
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


def _collect_and_report(
    *,
    training_plan: Mapping[str, Any],
    a_references: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    project_commit: str,
    paths: DriverPaths,
    input_sha256: Mapping[str, str],
    bootstrap_replicates: int,
) -> None:
    stage_name = "scientific-report"
    if (
        _load_stage(
            paths.stage(stage_name),
            name=stage_name,
            project_commit=project_commit,
            input_sha256=input_sha256,
        )
        is not None
    ):
        _verify_scientific_outputs(paths, bootstrap_replicates=bootstrap_replicates)
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
    _seal_stage(
        paths.stage(stage_name),
        name=stage_name,
        project_commit=project_commit,
        input_sha256=input_sha256,
        outputs=[
            paths.training_costs,
            paths.score_grid,
            paths.report_json,
            paths.report_markdown,
        ],
        metadata={
            "bootstrap_replicates": bootstrap_replicates,
            "report_sha256": report["report_sha256"],
        },
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
) -> tuple[dict[str, Any], Path]:
    stage_name = "compact-archive"
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
        metadata = existing["metadata"]
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
    _seal_stage(
        paths.stage(stage_name),
        name=stage_name,
        project_commit=project_commit,
        input_sha256=input_sha256,
        outputs=[archive, paths.archive_index, paths.upload_manifest],
        metadata={
            "archive_object": archive_object,
            "manifest_object": manifest_object,
            "manifest_file_sha256": manifest_sha,
            "uploaded_bulk_data": False,
        },
    )
    return manifest, archive


def _validate_terminal(
    path: Path,
    *,
    project_commit: str,
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    value = _validate_self_hash(
        _load_json(path, label="terminal completion"),
        hash_field="completion_sha256",
        label="terminal completion",
    )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("record_type") != "terminal_completion"
        or value.get("status") != "completed"
        or value.get("project_commit") != project_commit
        or value.get("input_sha256") != dict(input_sha256)
        or value.get("bulk_capture_or_checkpoint_data_uploaded") is not False
    ):
        raise ComponentClosureEndToEndError("terminal completion identity changed")
    _validate_artifact_rows(value.get("outputs"), label="terminal completion")
    if not isinstance(value.get("archive_object"), Mapping) or not isinstance(
        value.get("manifest_object"), Mapping
    ):
        raise ComponentClosureEndToEndError("terminal GCS evidence changed")
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
    value = {**body, "completion_sha256": canonical_sha256(body)}
    _seal_json(paths.terminal_completion, value, label="terminal completion")
    return _validate_terminal(
        paths.terminal_completion,
        project_commit=project_commit,
        input_sha256=input_sha256,
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
    _validate_checkout(training)
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
        if paths.terminal_completion.exists():
            return _validate_terminal(
                paths.terminal_completion,
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
        _collect_and_report(
            training_plan=training_plan,
            a_references=a_references,
            evaluation=evaluation,
            project_commit=args.project_commit,
            paths=paths,
            input_sha256=input_sha256,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        sources = _archive_sources(
            training_plan_path=training_plan_path,
            a_references_path=a_references_path,
            evaluation_runtime_path=evaluation_runtime_path,
            training_state_root=training.state_root,
            evaluation=evaluation,
            paths=paths,
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
