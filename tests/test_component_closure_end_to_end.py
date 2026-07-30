from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from moe_sae.contribution_artifacts import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tpu" / "run_component_closure_end_to_end.py"
REVISION = "a" * 40


@pytest.fixture()
def driver():
    name = "_test_component_closure_end_to_end"
    specification = importlib.util.spec_from_file_location(name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _completion(driver, bundle: dict) -> dict:
    body = {
        "schema_version": "test-evaluation-completion-v1",
        "status": "bundle_published",
        "bundle_sha256": bundle["bundle_sha256"],
        "resource_id": bundle["resource_id"],
        "cells": [{"index": index} for index in range(12)],
    }
    return {**body, "completion_sha256": canonical_sha256(body)}


def _evaluation(driver, state_root: Path) -> dict:
    bundles = []
    for panel in ("sae", "transcoder"):
        for seed in (0, 1, 2):
            body = {
                "resource_id": f"{panel}-{seed}",
                "panel": panel,
                "seed": seed,
            }
            bundles.append({**body, "bundle_sha256": canonical_sha256(body)})
    return {
        "state_root": str(state_root / "evaluation"),
        "bundles": bundles,
    }


def _write_completions(driver, evaluation: dict) -> None:
    for bundle in evaluation["bundles"]:
        path = (
            Path(evaluation["state_root"])
            / "bundles"
            / f"{bundle['panel']}-seed{bundle['seed']}"
            / "completion.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_completion(driver, bundle), sort_keys=True),
            encoding="ascii",
        )


def test_clock_is_durable_and_enforces_one_twelve_hour_wall(
    driver, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1000.0]
    monkeypatch.setattr(driver.time, "time", lambda: now[0])
    path = tmp_path / "state" / "clock.json"
    inputs = {"plan": "b" * 64}
    clock = driver._bind_clock(
        path,
        project_commit=REVISION,
        maximum_wall_seconds=43_200,
        input_sha256=inputs,
    )
    assert driver._remaining_seconds(clock) == 43_200

    now[0] += 120
    resumed = driver._bind_clock(
        path,
        project_commit=REVISION,
        maximum_wall_seconds=43_200,
        input_sha256=inputs,
    )
    assert resumed == clock
    assert driver._remaining_seconds(resumed) == 43_080

    now[0] += 43_081
    with pytest.raises(driver.ComponentClosureEndToEndError, match="reached its wall"):
        driver._remaining_seconds(resumed)
    with pytest.raises(driver.ComponentClosureEndToEndError, match="43200"):
        driver._bind_clock(
            tmp_path / "other-clock.json",
            project_commit=REVISION,
            maximum_wall_seconds=43_201,
            input_sha256=inputs,
        )


def test_stage_resume_is_create_only_and_detects_output_mutation(
    driver, tmp_path: Path
) -> None:
    output = tmp_path / "result.json"
    output.write_text('{"ok":true}\n', encoding="ascii")
    stage = tmp_path / "state" / "stage.json"
    inputs = {"plan": "b" * 64}
    sealed = driver._seal_stage(
        stage,
        name="test",
        project_commit=REVISION,
        input_sha256=inputs,
        outputs=[output],
        metadata={"attempts": 1},
    )
    assert (
        driver._load_stage(
            stage,
            name="test",
            project_commit=REVISION,
            input_sha256=inputs,
        )
        == sealed
    )
    output.write_text('{"ok":false}\n', encoding="ascii")
    with pytest.raises(driver.ComponentClosureEndToEndError, match="changed"):
        driver._load_stage(
            stage,
            name="test",
            project_commit=REVISION,
            input_sha256=inputs,
        )


def test_checkout_must_be_clean_and_exact(
    driver, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "python"
    executable.write_text("#!/bin/sh\n", encoding="ascii")
    executable.chmod(0o700)
    training = SimpleNamespace(
        working_directory=tmp_path,
        python_executable=executable,
        project_commit=REVISION,
    )
    outputs = [
        subprocess.CompletedProcess([], 0, REVISION + "\n", ""),
        subprocess.CompletedProcess([], 0, "", ""),
    ]
    monkeypatch.setattr(
        driver.subprocess, "run", lambda *args, **kwargs: outputs.pop(0)
    )
    driver._validate_checkout(training)

    outputs.extend(
        [
            subprocess.CompletedProcess([], 0, REVISION + "\n", ""),
            subprocess.CompletedProcess([], 0, "?? unexpected\n", ""),
        ]
    )
    with pytest.raises(driver.ComponentClosureEndToEndError, match="must be clean"):
        driver._validate_checkout(training)


def test_evaluation_retries_with_fresh_logs_and_reuses_partial_cells(
    driver, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = driver.DriverPaths(
        output_root=tmp_path / "outputs",
        state_root=tmp_path / "state",
    )
    paths.output_root.mkdir()
    paths.state_root.mkdir()
    evaluation = _evaluation(driver, paths.state_root)
    calls: list[tuple[list[str], Path]] = []

    def run_logged(command, *, cwd, log_path, timeout_seconds):
        calls.append((list(command), log_path))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"attempt={len(calls)}\n", encoding="ascii")
        if len(calls) == 3:
            _write_completions(driver, evaluation)
            return 0
        return 17

    monkeypatch.setattr(driver, "_run_logged_command", run_logged)
    monkeypatch.setattr(driver.time, "sleep", lambda _seconds: None)
    training = SimpleNamespace(
        project_commit=REVISION,
        python_executable=Path("/python"),
        working_directory=tmp_path,
    )
    clock = {
        "started_at_unix_seconds": time.time(),
        "maximum_wall_seconds": 1000,
    }
    driver._run_evaluation(
        training_plan_path=tmp_path / "training.yaml",
        evaluation_plan_path=tmp_path / "evaluation.json",
        evaluation=evaluation,
        training=training,
        paths=paths,
        clock=clock,
        input_sha256={"plan": "b" * 64},
    )
    assert len(calls) == 3
    assert len({str(row[1]) for row in calls}) == 3
    assert all("run-all-operators" in command for command, _ in calls)
    stage = driver._load_stage(
        paths.stage("evaluation"),
        name="evaluation",
        project_commit=REVISION,
        input_sha256={"plan": "b" * 64},
    )
    assert stage["metadata"]["attempt_count"] == 3
    assert stage["metadata"]["failed_returncodes"] == [17, 17]
    assert stage["metadata"]["resume_policy"] == (
        "immutable_completed_cells_are_reused"
    )

    calls.clear()
    driver._run_evaluation(
        training_plan_path=tmp_path / "training.yaml",
        evaluation_plan_path=tmp_path / "evaluation.json",
        evaluation=evaluation,
        training=training,
        paths=paths,
        clock=clock,
        input_sha256={"plan": "b" * 64},
    )
    assert calls == []


def test_evaluation_stops_after_four_bounded_attempts(
    driver, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = driver.DriverPaths(tmp_path / "outputs", tmp_path / "state")
    paths.output_root.mkdir()
    paths.state_root.mkdir()
    attempts = []

    def fail(command, *, cwd, log_path, timeout_seconds):
        attempts.append((command, log_path, timeout_seconds))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("failed\n", encoding="ascii")
        return 9

    monkeypatch.setattr(driver, "_run_logged_command", fail)
    monkeypatch.setattr(driver.time, "sleep", lambda _seconds: None)
    training = SimpleNamespace(
        project_commit=REVISION,
        python_executable=Path("/python"),
        working_directory=tmp_path,
    )
    with pytest.raises(driver.ComponentClosureEndToEndError, match="four"):
        driver._run_evaluation(
            training_plan_path=tmp_path / "training.yaml",
            evaluation_plan_path=tmp_path / "evaluation.json",
            evaluation=_evaluation(driver, paths.state_root),
            training=training,
            paths=paths,
            clock={
                "started_at_unix_seconds": time.time(),
                "maximum_wall_seconds": 1000,
            },
            input_sha256={"plan": "b" * 64},
        )
    assert len(attempts) == 4
    assert all(0 < timeout <= 1000 for _, _, timeout in attempts)


def test_evaluation_recovers_complete_bundles_without_relaunch(
    driver, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = driver.DriverPaths(tmp_path / "outputs", tmp_path / "state")
    paths.output_root.mkdir()
    paths.state_root.mkdir()
    evaluation = _evaluation(driver, paths.state_root)
    _write_completions(driver, evaluation)
    monkeypatch.setattr(
        driver,
        "_run_logged_command",
        lambda *args, **kwargs: pytest.fail("completed bundles must not relaunch"),
    )
    training = SimpleNamespace(
        project_commit=REVISION,
        python_executable=Path("/python"),
        working_directory=tmp_path,
    )
    driver._run_evaluation(
        training_plan_path=tmp_path / "training.yaml",
        evaluation_plan_path=tmp_path / "evaluation.json",
        evaluation=evaluation,
        training=training,
        paths=paths,
        clock={
            "started_at_unix_seconds": time.time(),
            "maximum_wall_seconds": 1000,
        },
        input_sha256={"plan": "b" * 64},
    )
    stage = driver._load_stage(
        paths.stage("evaluation"),
        name="evaluation",
        project_commit=REVISION,
        input_sha256={"plan": "b" * 64},
    )
    assert stage["metadata"]["attempt_count"] == 0


def test_compact_archive_is_deterministic_and_contains_no_bulk_paths(
    driver, tmp_path: Path
) -> None:
    source_a = tmp_path / "plan.yaml"
    source_b = tmp_path / "report.json"
    source_a.write_text("plan: exact\n", encoding="utf-8")
    source_b.write_text('{"report":true}\n', encoding="ascii")
    sources = [
        ("inputs/training-plan.yaml", source_a),
        ("outputs/report.json", source_b),
    ]
    first_paths = driver.DriverPaths(tmp_path / "one", tmp_path / "state-one")
    second_paths = driver.DriverPaths(tmp_path / "two", tmp_path / "state-two")
    first_paths.output_root.mkdir()
    second_paths.output_root.mkdir()
    first, first_index = driver._build_compact_archive(
        sources=sources,
        project_commit=REVISION,
        paths=first_paths,
    )
    second, second_index = driver._build_compact_archive(
        sources=sources,
        project_commit=REVISION,
        paths=second_paths,
    )
    assert first.read_bytes() == second.read_bytes()
    assert first_index == second_index
    assert first_index["bulk_capture_or_checkpoint_data_included"] is False
    with tarfile.open(first, "r:gz") as archive:
        assert archive.getnames() == [
            "inputs/training-plan.yaml",
            "outputs/report.json",
            "archive-index.json",
        ]
        assert all(
            "checkpoint" not in name and "capture" not in name
            for name in archive.getnames()
        )


def test_compact_archive_refuses_large_evidence(
    driver, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "too-large"
    source.write_bytes(b"12345")
    paths = driver.DriverPaths(tmp_path / "output", tmp_path / "state")
    paths.output_root.mkdir()
    monkeypatch.setattr(driver, "MAXIMUM_COMPACT_ARCHIVE_BYTES", 4)
    with pytest.raises(driver.ComponentClosureEndToEndError, match="bulk data"):
        driver._build_compact_archive(
            sources=[("output/evidence", source)],
            project_commit=REVISION,
            paths=paths,
        )


def test_gcs_upload_is_create_only_content_verified_and_compact(
    driver, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "archive.tar.gz"
    source.write_bytes(b"compact archive")
    digest = driver._sha256_path(source)[0]
    calls: list[list[str]] = []

    def capture(command, *, timeout_seconds):
        calls.append(list(command))
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 1, b"", b"not found")
        if len(calls) == 2:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "generation": "123",
                    "size": len(source.read_bytes()),
                    "custom_fields": {"sha256": digest},
                }
            ).encode("ascii"),
            b"",
        )

    monkeypatch.setattr(driver, "_capture_command", capture)
    identity = driver._upload_gcs_object(
        path=source,
        uri=f"gs://bucket/closure/{digest}.tar.gz",
        executable="gcloud",
        project="neural-dragnet",
        timeout_seconds=60,
    )
    assert identity["generation"] == 123
    copy = calls[1]
    assert "--if-generation-match=0" in copy
    assert f"--custom-metadata=sha256={digest}" in copy
    assert str(source) in copy
    assert all("--recursive" not in argument for argument in copy)


def test_gcs_upload_accepts_exact_object_after_create_only_race(
    driver, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "manifest.json"
    source.write_bytes(b"exact")
    digest = driver._sha256_path(source)[0]
    calls = []

    def capture(command, *, timeout_seconds):
        calls.append(list(command))
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 1, b"", b"not found")
        if len(calls) == 2:
            return subprocess.CompletedProcess(command, 1, b"", b"precondition")
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "generation": 9,
                    "size_bytes": 5,
                    "metadata": {"sha256": digest},
                }
            ).encode("ascii"),
            b"",
        )

    monkeypatch.setattr(driver, "_capture_command", capture)
    identity = driver._upload_gcs_object(
        path=source,
        uri="gs://bucket/closure/exact.json",
        executable="gcloud",
        project="neural-dragnet",
        timeout_seconds=60,
    )
    assert identity["generation"] == 9
    assert len(calls) == 3


def test_terminal_completion_is_self_hashed_and_fails_closed_on_tamper(
    driver, tmp_path: Path
) -> None:
    output = tmp_path / "report.json"
    output.write_text('{"report":true}\n', encoding="ascii")
    body = {
        "schema_version": driver.SCHEMA_VERSION,
        "record_type": "terminal_completion",
        "status": "completed",
        "project_commit": REVISION,
        "input_sha256": {"plan": "b" * 64},
        "bulk_capture_or_checkpoint_data_uploaded": False,
        "archive_object": {"uri": "gs://bucket/archive"},
        "manifest_object": {"uri": "gs://bucket/manifest"},
        "outputs": [driver._artifact_row(output)],
    }
    value = {**body, "completion_sha256": canonical_sha256(body)}
    path = tmp_path / "completion.json"
    path.write_text(json.dumps(value), encoding="ascii")
    assert (
        driver._validate_terminal(
            path,
            project_commit=REVISION,
            input_sha256={"plan": "b" * 64},
        )["status"]
        == "completed"
    )
    output.write_text('{"report":false}\n', encoding="ascii")
    with pytest.raises(driver.ComponentClosureEndToEndError, match="changed"):
        driver._validate_terminal(
            path,
            project_commit=REVISION,
            input_sha256={"plan": "b" * 64},
        )


def test_help_is_offline() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--archive-gcs-prefix" in completed.stdout
    assert "--maximum-wall-seconds" in completed.stdout
