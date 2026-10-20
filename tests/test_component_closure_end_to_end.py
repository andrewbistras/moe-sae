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


def _write_publication_recovery_receipts(
    driver,
    evaluation: dict,
) -> tuple[list[dict], Path]:
    evaluation["evaluation_source_revision"] = REVISION
    evaluation["plan_sha256"] = "c" * 64
    recovery_script = Path(evaluation["state_root"]).parent / "recover.py"
    recovery_script.parent.mkdir(parents=True, exist_ok=True)
    recovery_script.write_text("# exact recovery\n", encoding="ascii")
    script_sha = driver._sha256_path(recovery_script)[0]
    driver.PUBLICATION_RECOVERY_SCRIPT_SHA256 = script_sha
    for bundle_index, bundle in enumerate(evaluation["bundles"]):
        panel = bundle["panel"]
        seed = bundle["seed"]
        recovery_id = f"controller-publish-recovery-{panel}-seed{seed}"
        completion_body = {
            "schema_version": "gemma4-component-closure-evaluation-operator-v1",
            "status": "bundle_published",
            "plan_sha256": evaluation["plan_sha256"],
            "bundle_sha256": bundle["bundle_sha256"],
            "resource_id": bundle["resource_id"],
            "panel": panel,
            "seed": seed,
            "attempt_id": recovery_id,
            "cells": [{"index": index} for index in range(12)],
        }
        completion = {
            **completion_body,
            "completion_sha256": canonical_sha256(completion_body),
        }
        completion_path = (
            Path(evaluation["state_root"])
            / "bundles"
            / f"{panel}-seed{seed}"
            / "completion.json"
        )
        completion_path.parent.mkdir(parents=True, exist_ok=True)
        completion_path.write_text(
            json.dumps(completion, sort_keys=True),
            encoding="ascii",
        )

        source_root = (
            Path(evaluation["state_root"])
            / "operator-logs"
            / f"{panel}-seed{seed}"
            / f"published-{panel}-seed{seed}"
        )
        source_root.mkdir(parents=True)
        source_logs = []
        for controller in range(4):
            process = (controller + seed) % 4
            log_path = source_root / (
                f"bundle-c{controller:05d}-p{process:05d}.log"
            )
            log_path.write_text(
                "terminal action=component-closure-evaluate-bundle "
                f"controller={controller} process={process} status=0\n",
                encoding="ascii",
            )
            source_logs.append(
                {
                    "controller": controller,
                    "process": process,
                    "path": str(log_path),
                    "file_sha256": driver._sha256_path(log_path)[0],
                    "terminal_status": 0,
                }
            )
        receipt_body = {
            "schema_version": driver.PUBLICATION_RECOVERY_SCHEMA_VERSION,
            "status": "bundle_published_after_conclusive_terminal_success",
            "project_commit": REVISION,
            "plan_sha256": evaluation["plan_sha256"],
            "bundle_sha256": bundle["bundle_sha256"],
            "bundle_index": bundle_index,
            "panel": panel,
            "seed": seed,
            "resource_id": bundle["resource_id"],
            "recovery_id": recovery_id,
            "source_log_root": str(source_root),
            "source_logs": source_logs,
            "completion_path": str(completion_path),
            "completion_sha256": completion["completion_sha256"],
            "recovery_script_file_sha256": script_sha,
            "reason": driver.PUBLICATION_RECOVERY_REASON,
        }
        receipt = {
            **receipt_body,
            "receipt_sha256": canonical_sha256(receipt_body),
        }
        receipt_path = (
            Path(evaluation["state_root"])
            / "publication-recovery"
            / f"{panel}-seed{seed}.json"
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True),
            encoding="ascii",
        )
    return (
        driver._validate_publication_recovery_receipts(
            evaluation,
            recovery_script=recovery_script,
        ),
        recovery_script,
    )


def _write_evaluation_metadata_inventory(
    driver,
    root: Path,
) -> list[dict]:
    rows = []
    for panel in ("sae", "transcoder"):
        for seed in (0, 1, 2):
            for condition in ("A", "R", "B", "F"):
                for checkpoint in (2_000_000, 5_000_000, 10_000_000):
                    role = (panel, seed, condition, checkpoint)
                    for name in (
                        "evaluation-result.json",
                        "centering-receipt.json",
                    ):
                        path = (
                            root
                            / panel
                            / f"seed{seed}"
                            / condition
                            / str(checkpoint)
                            / name
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(
                            json.dumps({"role": role, "name": name}),
                            encoding="ascii",
                        )
                        digest, size = driver._sha256_path(path)
                        rows.append(
                            {
                                "archive_name": (
                                    "state/evaluation/cells/"
                                    f"{panel}/seed{seed}/{condition}/"
                                    f"tokens-{checkpoint:08d}/{name}"
                                ),
                                "path": path,
                                "sha256": digest,
                                "size_bytes": size,
                                "semantic_sha256": canonical_sha256(
                                    {"role": role, "name": name}
                                ),
                                "role": role,
                            }
                        )
    return rows


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


def test_report_recovery_checkout_requires_clean_descendant(
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
    recovery_commit = "9" * 40
    outputs = [
        subprocess.CompletedProcess([], 0, recovery_commit + "\n", ""),
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 0, "", ""),
    ]
    monkeypatch.setattr(
        driver.subprocess,
        "run",
        lambda *args, **kwargs: outputs.pop(0),
    )
    driver._validate_report_recovery_checkout(
        training,
        report_recovery_commit=recovery_commit,
    )

    outputs.extend(
        [
            subprocess.CompletedProcess([], 0, recovery_commit + "\n", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 1, "", ""),
        ]
    )
    with pytest.raises(
        driver.ComponentClosureEndToEndError,
        match="clean descendant",
    ):
        driver._validate_report_recovery_checkout(
            training,
            report_recovery_commit=recovery_commit,
        )


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


def test_publication_recovery_requires_exact_six_and_binds_terminal_logs(
    driver, tmp_path: Path
) -> None:
    evaluation = _evaluation(driver, tmp_path / "state")
    receipts, script = _write_publication_recovery_receipts(
        driver,
        evaluation,
    )
    assert [
        (row["panel"], row["seed"]) for row in receipts
    ] == [
        ("sae", 0),
        ("sae", 1),
        ("sae", 2),
        ("transcoder", 0),
        ("transcoder", 1),
        ("transcoder", 2),
    ]
    assert all(row["archive_name"].endswith(".json") for row in receipts)

    extra = (
        Path(evaluation["state_root"])
        / "publication-recovery"
        / "unexpected.json"
    )
    extra.write_text("{}\n", encoding="ascii")
    with pytest.raises(
        driver.ComponentClosureEndToEndError,
        match="exactly the six",
    ):
        driver._validate_publication_recovery_receipts(
            evaluation,
            recovery_script=script,
        )


def test_publication_recovery_rejects_resealed_nonzero_terminal_status(
    driver, tmp_path: Path
) -> None:
    evaluation = _evaluation(driver, tmp_path / "state")
    _receipts, script = _write_publication_recovery_receipts(
        driver,
        evaluation,
    )
    path = (
        Path(evaluation["state_root"])
        / "publication-recovery"
        / "sae-seed0.json"
    )
    value = json.loads(path.read_text(encoding="ascii"))
    value["source_logs"][0]["terminal_status"] = 9
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    value["receipt_sha256"] = canonical_sha256(body)
    path.write_text(json.dumps(value, sort_keys=True), encoding="ascii")
    with pytest.raises(
        driver.ComponentClosureEndToEndError,
        match="terminal evidence",
    ):
        driver._validate_publication_recovery_receipts(
            evaluation,
            recovery_script=script,
        )


def test_publication_recovery_independently_rejects_duplicate_log_marker(
    driver, tmp_path: Path
) -> None:
    evaluation = _evaluation(driver, tmp_path / "state")
    _receipts, script = _write_publication_recovery_receipts(
        driver,
        evaluation,
    )
    receipt_path = (
        Path(evaluation["state_root"])
        / "publication-recovery"
        / "sae-seed0.json"
    )
    value = json.loads(receipt_path.read_text(encoding="ascii"))
    log_path = Path(value["source_logs"][0]["path"])
    marker = log_path.read_text(encoding="ascii")
    log_path.write_text(marker + marker, encoding="ascii")
    value["source_logs"][0]["file_sha256"] = driver._sha256_path(log_path)[0]
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    value["receipt_sha256"] = canonical_sha256(body)
    receipt_path.write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(
        driver.ComponentClosureEndToEndError,
        match="one exact terminal marker",
    ):
        driver._validate_publication_recovery_receipts(
            evaluation,
            recovery_script=script,
        )


def test_distinct_evaluation_and_centering_source_manifests_are_valid(
    driver,
) -> None:
    evaluation_source = driver._validated_nullable_sha256(
        "f" * 64,
        label="evaluation source",
    )
    centering_source = driver._validated_nullable_sha256(
        "8" * 64,
        label="centering source",
    )
    assert evaluation_source != centering_source
    assert (
        driver._validated_nullable_sha256(
            None,
            label="optional source",
        )
        is None
    )
    with pytest.raises(
        driver.ComponentClosureEndToEndError,
        match="lowercase SHA-256 or null",
    ):
        driver._validated_nullable_sha256(
            "F" * 64,
            label="evaluation source",
        )


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


def test_recovery_archive_contains_exact_receipt_inventory(
    driver, tmp_path: Path
) -> None:
    paths = driver.DriverPaths(tmp_path / "outputs", tmp_path / "state")
    paths.output_root.mkdir()
    paths.state_root.mkdir()
    evaluation = _evaluation(driver, paths.state_root)
    receipts, recovery_script = _write_publication_recovery_receipts(
        driver,
        evaluation,
    )
    evaluation_artifacts = _write_evaluation_metadata_inventory(
        driver,
        tmp_path / "evaluation-metadata",
    )
    training_plan = tmp_path / "training.yaml"
    a_references = tmp_path / "a-references.json"
    runtime = tmp_path / "runtime.yaml"
    for path in (training_plan, a_references, runtime):
        path.write_text("{}\n", encoding="ascii")
    training_state = tmp_path / "training-state"
    training_state.mkdir()
    required = [
        paths.evaluation_plan,
        paths.training_costs,
        paths.score_grid,
        paths.report_json,
        paths.report_markdown,
        paths.clock,
        training_state / "training-complete.json",
        *(
            paths.stage(stage)
            for stage in (
                "training",
                "evaluation-plan",
                "evaluation",
                "scientific-report",
            )
        ),
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{path.name}\n", encoding="ascii")
    paths.report_recovery_receipt.parent.mkdir(parents=True)
    paths.report_recovery_receipt.write_text(
        '{"receipt":"bound"}\n',
        encoding="ascii",
    )
    sources = driver._archive_sources(
        training_plan_path=training_plan,
        a_references_path=a_references,
        evaluation_runtime_path=runtime,
        training_state_root=training_state,
        evaluation=evaluation,
        paths=paths,
        publication_receipts=receipts,
        report_recovery_receipt=paths.report_recovery_receipt,
        recovery_script=recovery_script,
        evaluation_artifacts=evaluation_artifacts,
    )
    archive, index = driver._build_compact_archive(
        sources=sources,
        project_commit=REVISION,
        paths=paths,
    )
    receipt_names = [
        f"state/evaluation/publication-recovery/{panel}-seed{seed}.json"
        for panel in ("sae", "transcoder")
        for seed in (0, 1, 2)
    ]
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
        assert all(name in names for name in receipt_names)
        assert "state/report-recovery/receipt.json" in names
        assert (
            "state/evaluation/publication-recovery/recovery-script.py"
            in names
        )
        assert sum(
            name.endswith("/evaluation-result.json") for name in names
        ) == 72
        assert sum(
            name.endswith("/centering-receipt.json") for name in names
        ) == 72
        assert sum(
            "/publication-recovery/logs/" in name for name in names
        ) == 24
    indexed = {row["name"]: row["sha256"] for row in index["entries"]}
    assert set(receipt_names) < set(indexed)
    assert indexed["state/report-recovery/receipt.json"] == (
        driver._sha256_path(paths.report_recovery_receipt)[0]
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


def test_report_recovery_receipt_binds_generator_policy_stages_and_outputs(
    driver, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = driver.DriverPaths(tmp_path / "outputs", tmp_path / "state")
    paths.output_root.mkdir()
    paths.state_root.mkdir()
    evaluation = _evaluation(driver, paths.state_root)
    publication_receipts, recovery_script = _write_publication_recovery_receipts(
        driver,
        evaluation,
    )
    evaluation_artifacts = _write_evaluation_metadata_inventory(
        driver,
        tmp_path / "evaluation-metadata",
    )
    paths.training_costs.write_text('{"costs":[]}\n', encoding="ascii")
    paths.score_grid.write_text('{"grid":"sealed"}\n', encoding="ascii")
    report_body = {"schema_version": driver.REPORT_SCHEMA_VERSION}
    report = {
        **report_body,
        "report_sha256": canonical_sha256(report_body),
    }
    paths.report_json.write_text(
        json.dumps(report, sort_keys=True),
        encoding="ascii",
    )
    paths.report_markdown.write_text("# sealed\n", encoding="utf-8")
    monkeypatch.setattr(
        driver,
        "load_score_grid",
        lambda _path: {"evidence_sha256": "e" * 64},
    )

    recovery_stages = {}
    for stage_name in ("training", "evaluation-plan", "evaluation"):
        output = tmp_path / f"{stage_name}-output.json"
        output.write_text(f'{{"stage":"{stage_name}"}}\n', encoding="ascii")
        recovery_stages[stage_name] = driver._seal_stage(
            paths.stage(stage_name),
            name=stage_name,
            project_commit=REVISION,
            input_sha256={"plan": "f" * 64},
            outputs=[output],
        )
    generator_commit = "9" * 40
    receipt = driver._seal_report_recovery_receipt(
        evaluation=evaluation,
        project_commit=REVISION,
        report_recovery_commit=generator_commit,
        paths=paths,
        input_sha256={"plan": "f" * 64},
        bootstrap_replicates=20_000,
        recovery_stages=recovery_stages,
        publication_receipts=publication_receipts,
        recovery_script=recovery_script,
        evaluation_artifacts=evaluation_artifacts,
    )
    assert receipt["report_generator_commit"] == generator_commit
    assert receipt["commit_ancestry"] == {
        "ancestor": REVISION,
        "descendant": generator_commit,
        "command": "git merge-base --is-ancestor",
        "verified": True,
    }
    assert receipt["report_policy"]["policy"] == driver.COMMON_SST_POLICY
    assert receipt["report_policy"]["reference_condition"] == "A"
    assert len(receipt["publication_recovery_receipts"]) == 6
    assert [row["stage"] for row in receipt["sealed_stages"]] == [
        "training",
        "evaluation-plan",
        "evaluation",
    ]
    assert (
        driver._seal_report_recovery_receipt(
            evaluation=evaluation,
            project_commit=REVISION,
            report_recovery_commit=generator_commit,
            paths=paths,
            input_sha256={"plan": "f" * 64},
            bootstrap_replicates=20_000,
            recovery_stages=recovery_stages,
            publication_receipts=publication_receipts,
            recovery_script=recovery_script,
            evaluation_artifacts=evaluation_artifacts,
        )
        == receipt
    )

    paths.report_markdown.write_text("# changed\n", encoding="utf-8")
    with pytest.raises(
        driver.ComponentClosureEndToEndError,
        match="differs from sealed evidence",
    ):
        driver._seal_report_recovery_receipt(
            evaluation=evaluation,
            project_commit=REVISION,
            report_recovery_commit=generator_commit,
            paths=paths,
            input_sha256={"plan": "f" * 64},
            bootstrap_replicates=20_000,
            recovery_stages=recovery_stages,
            publication_receipts=publication_receipts,
            recovery_script=recovery_script,
            evaluation_artifacts=evaluation_artifacts,
        )


def _seal_terminal_fixture(
    driver,
    root: Path,
    *,
    recovery: bool,
) -> tuple[object, dict, Path | None]:
    paths = driver.DriverPaths(root / "outputs", root / "state")
    paths.output_root.mkdir(parents=True)
    paths.state_root.mkdir(parents=True)
    for path in (
        paths.evaluation_plan,
        paths.training_costs,
        paths.score_grid,
        paths.report_markdown,
        paths.archive_index,
    ):
        path.write_text(f"{path.name}\n", encoding="ascii")
    report_body = {"schema_version": "test-report-v1"}
    report = {
        **report_body,
        "report_sha256": canonical_sha256(report_body),
    }
    paths.report_json.write_text(json.dumps(report), encoding="ascii")
    archive = paths.output_root / "archive.tar.gz"
    archive.write_bytes(b"archive")
    clock_body = {
        "schema_version": driver.SCHEMA_VERSION,
        "record_type": "run_clock",
        "project_commit": REVISION,
        "maximum_wall_seconds": 1000,
        "input_sha256": {"plan": "b" * 64},
        "started_at": "test",
        "started_at_unix_seconds": time.time() - 10,
    }
    clock = {**clock_body, "clock_sha256": canonical_sha256(clock_body)}
    paths.clock.write_text(json.dumps(clock), encoding="ascii")
    receipt_path = None
    generator_commit = "9" * 40 if recovery else None
    receipt = None
    if recovery:
        receipt_body = {
            "schema_version": driver.REPORT_RECOVERY_SCHEMA_VERSION,
            "report_generator_commit": generator_commit,
            "report_sha256": report["report_sha256"],
        }
        receipt = {
            **receipt_body,
            "receipt_sha256": canonical_sha256(receipt_body),
        }
        receipt_path = paths.report_recovery_receipt
        receipt_path.parent.mkdir(parents=True)
        receipt_path.write_text(json.dumps(receipt), encoding="ascii")
    manifest_body = {
        "archive": {"local_path": str(archive)},
        "report_sha256": report["report_sha256"],
    }
    if recovery:
        assert receipt is not None
        manifest_body.update(
            {
                "report_generator_commit": generator_commit,
                "report_recovery_receipt_sha256": receipt["receipt_sha256"],
            }
        )
    manifest = {
        **manifest_body,
        "manifest_sha256": canonical_sha256(manifest_body),
    }
    paths.upload_manifest.write_text(json.dumps(manifest), encoding="ascii")
    archive_object = {"uri": "gs://bucket/archive"}
    manifest_object = {"uri": "gs://bucket/manifest"}
    metadata = {
        "archive_object": archive_object,
        "manifest_object": manifest_object,
    }
    if recovery:
        assert receipt is not None
        metadata.update(
            {
                "report_generator_commit": generator_commit,
                "report_recovery_receipt_sha256": receipt["receipt_sha256"],
            }
        )
    stage = driver._seal_stage(
        paths.stage("compact-archive"),
        name="compact-archive",
        project_commit=REVISION,
        input_sha256={"plan": "b" * 64},
        outputs=[archive, paths.archive_index, paths.upload_manifest],
        metadata=metadata,
    )
    terminal = driver._seal_terminal(
        paths=paths,
        project_commit=REVISION,
        input_sha256={"plan": "b" * 64},
        clock=clock,
        archive_manifest=manifest,
        archive=archive,
        archive_stage=stage,
        report_recovery_commit=generator_commit,
        report_recovery_receipt=receipt_path,
    )
    return paths, terminal, receipt_path


def test_terminal_completion_enforces_normal_semantic_linkage(
    driver, tmp_path: Path
) -> None:
    paths, terminal, _receipt = _seal_terminal_fixture(
        driver,
        tmp_path,
        recovery=False,
    )
    assert terminal["status"] == "completed"
    paths.report_markdown.write_text("changed\n", encoding="ascii")
    with pytest.raises(
        driver.ComponentClosureEndToEndError,
        match="changed",
    ):
        driver._validate_terminal(
            paths.terminal_completion,
            paths=paths,
            project_commit=REVISION,
            input_sha256={"plan": "b" * 64},
        )


def test_terminal_completion_binds_recovery_generator_and_receipt(
    driver, tmp_path: Path
) -> None:
    paths, terminal, receipt_path = _seal_terminal_fixture(
        driver,
        tmp_path,
        recovery=True,
    )
    assert terminal["report_generator_commit"] == "9" * 40
    assert receipt_path is not None
    assert (
        driver._validate_terminal(
            paths.terminal_completion,
            paths=paths,
            project_commit=REVISION,
            input_sha256={"plan": "b" * 64},
            report_recovery_commit="9" * 40,
            report_recovery_receipt=receipt_path,
        )
        == terminal
    )


def test_report_recovery_run_skips_all_accelerator_work(
    driver, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("training.yaml", "a.json", "runtime.yaml"):
        (tmp_path / name).write_text("{}\n", encoding="ascii")
    output_root = tmp_path / "outputs"
    state_root = tmp_path / "state"
    output_root.mkdir()
    state_root.mkdir()
    executable = tmp_path / "python"
    executable.write_text("#!/bin/sh\n", encoding="ascii")
    executable.chmod(0o700)
    recovery_script = tmp_path / "report-recovery.py"
    recovery_script.write_text("# recover\n", encoding="ascii")
    training = SimpleNamespace(
        project_commit=REVISION,
        working_directory=tmp_path,
        python_executable=executable,
        plan_sha256="1" * 64,
        state_root=tmp_path / "training-state",
    )
    args = SimpleNamespace(
        training_plan=tmp_path / "training.yaml",
        a_references=tmp_path / "a.json",
        evaluation_runtime=tmp_path / "runtime.yaml",
        output_root=output_root,
        state_root=state_root,
        project_commit=REVISION,
        report_recovery_commit="9" * 40,
        report_recovery_script=recovery_script,
        archive_gcs_prefix="gs://bucket/recovery",
        maximum_wall_seconds=1000,
        bootstrap_replicates=100,
        gcloud_executable="gcloud",
    )
    monkeypatch.setattr(driver, "load_training_plan", lambda _path: {})
    monkeypatch.setattr(driver, "parse_training_campaign", lambda _value: training)
    monkeypatch.setattr(driver, "load_a_reference_manifest", lambda _path: {})
    monkeypatch.setattr(
        driver,
        "_validate_cli_identity",
        lambda _args, *, training: ("project", "gs://bucket/recovery"),
    )
    monkeypatch.setattr(
        driver,
        "_validate_report_recovery_checkout",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(driver, "_validate_roots", lambda *a, **k: None)
    monkeypatch.setattr(
        driver,
        "_input_hashes",
        lambda **kwargs: {"plan": "2" * 64},
    )
    monkeypatch.setattr(
        driver,
        "_bind_clock",
        lambda *a, **k: {
            "clock_sha256": "3" * 64,
            "started_at_unix_seconds": time.time(),
            "maximum_wall_seconds": 1000,
        },
    )
    monkeypatch.setattr(driver, "_remaining_seconds", lambda _clock: 900)
    evaluation = {"bundles": [], "state_root": str(state_root / "evaluation")}
    sealed = {
        name: {"stage_sha256": str(index) * 64, "outputs": []}
        for index, name in enumerate(
            ("training", "evaluation-plan", "evaluation"),
            start=4,
        )
    }
    monkeypatch.setattr(
        driver,
        "_load_sealed_report_recovery_inputs",
        lambda **kwargs: (evaluation, sealed, [], []),
    )
    for forbidden in ("_run_training", "_build_eval_plan", "_run_evaluation"):
        monkeypatch.setattr(
            driver,
            forbidden,
            lambda *args, _name=forbidden, **kwargs: pytest.fail(
                f"report recovery called {_name}"
            ),
        )
    observed = {}

    def collect(**kwargs):
        observed["collect"] = kwargs

    monkeypatch.setattr(driver, "_collect_and_report", collect)
    monkeypatch.setattr(driver, "_archive_sources", lambda **kwargs: [])
    archive = output_root / "archive.tar.gz"
    monkeypatch.setattr(
        driver,
        "_archive_and_upload",
        lambda **kwargs: ({"manifest_sha256": "7" * 64}, archive),
    )
    monkeypatch.setattr(
        driver,
        "_load_stage",
        lambda *args, **kwargs: {"metadata": {}},
    )
    monkeypatch.setattr(
        driver,
        "_seal_terminal",
        lambda **kwargs: {"status": "completed"},
    )

    assert driver.run(args) == {"status": "completed"}
    assert observed["collect"]["report_recovery_commit"] == "9" * 40
    assert observed["collect"]["recovery_stages"] == sealed


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
