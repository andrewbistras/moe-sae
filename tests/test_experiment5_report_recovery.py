from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

import moe_sae.experiment5_report_recovery as report_recovery
from moe_sae.contribution_artifacts import canonical_json_bytes, canonical_sha256
from moe_sae.experiment5_campaign import (
    _record,
    _write_create_only,
    inspect_evaluation_completion,
)
from moe_sae.experiment5_evaluation_recovery import (
    build_evaluation_recovery_plan,
    parse_evaluation_recovery_plan,
)
from moe_sae.experiment5_report_recovery import (
    BASIS_AUDIT_SCHEMA_VERSION,
    Experiment5ReportRecoveryError,
    build_report_recovery_plan,
    load_report_recovery_plan,
    parse_report_recovery_plan,
    report_recovery_status,
    seal_report_recovery,
    validate_report_recovery_handoff,
    verify_report_recovery,
    write_report_recovery_plan,
)

from test_experiment5_evaluation_recovery import (
    _core_with_training_barrier,
    _recovery_resources,
)
from test_experiment5_report import SOURCE_REVISION, _budget, _grid


REPORTING_REVISION_MISMATCH = "a" * 40


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _clean_git_checkout(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path,
        check=True,
    )
    (path / "README.md").write_text("report checkout\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "report recovery test"],
        cwd=path,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _rehash_pair(result: dict, paired: dict) -> tuple[dict, dict]:
    paired_body = dict(paired)
    paired_body.pop("artifact_sha256", None)
    paired = {
        **paired_body,
        "artifact_sha256": canonical_sha256(paired_body),
    }
    result_body = dict(result)
    result_body.pop("result_sha256", None)
    result_body["predictions_artifact_sha256"] = paired["artifact_sha256"]
    result = {
        **result_body,
        "result_sha256": canonical_sha256(result_body),
    }
    return result, paired


def _sealed_report_plan(
    tmp_path: Path,
    *,
    mutate: Callable[[dict, dict], None] | None = None,
    reporting_revision: str | None = None,
) -> tuple[dict, Path, Path]:
    core_path, raw_core = _core_with_training_barrier(tmp_path / "parents")
    resources = _recovery_resources(raw_core)
    for resource in resources:
        resource["deployed_commit"] = SOURCE_REVISION
    recovery_raw = build_evaluation_recovery_plan(
        core_plan_path=core_path,
        evaluation_source_revision=SOURCE_REVISION,
        resources=resources,
        state_root=tmp_path / "evaluation-recovery" / "state",
        result_roots_controller={
            "us-test1": tmp_path / "evaluation-recovery" / "results"
        },
        result_root_remote="/remote/evaluation-recovery",
        already_spent_usd=31.64,
    )
    evaluation_plan_path = tmp_path / "evaluation-recovery" / "evaluation-plan.yaml"
    evaluation_plan_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_plan_path.write_text(
        yaml.safe_dump(recovery_raw, sort_keys=False),
        encoding="utf-8",
    )
    evaluation_plan = parse_evaluation_recovery_plan(recovery_raw)

    results, paired_scores = _grid()
    result_grid = {
        (
            row["panel"],
            row["seed"],
            row["checkpoint_target_tokens"],
        ): copy.deepcopy(row)
        for row in results
    }
    paired_grid = {
        (
            row["panel"],
            row["seed"],
            row["checkpoint_target_tokens"],
        ): copy.deepcopy(row)
        for row in paired_scores
    }
    completions = {}
    for task in evaluation_plan.evaluations:
        identity = (task.panel, task.seed, task.milestone_tokens)
        result = result_grid[identity]
        paired = paired_grid[identity]
        if mutate is not None:
            mutate(result, paired)
        result, paired = _rehash_pair(result, paired)
        _write_json(task.completion_path_controller, result)
        _write_json(task.paired_scores_path_controller, paired)
        completion = inspect_evaluation_completion(evaluation_plan, task)
        assert completion is not None
        completions[task.task_id] = completion.to_dict()
    barrier = _record(
        "evaluation_phase_completion",
        {
            "campaign_id": evaluation_plan.campaign_id,
            "plan_sha256": evaluation_plan.plan_sha256,
            "status": "evaluation_completed",
            "evaluations": {key: completions[key] for key in sorted(completions)},
            "completed_at": "2026-07-29T14:00:00Z",
        },
    )
    _write_create_only(
        evaluation_plan.state_root / "evaluation-complete.json",
        barrier,
    )
    ledger_path = evaluation_plan.state_root / "report" / "budget-ledger.json"
    _write_json(ledger_path, _budget(spent=5.0))

    checkout = tmp_path / "reporting-checkout"
    actual_revision = _clean_git_checkout(checkout)
    raw = build_report_recovery_plan(
        evaluation_plan_path=evaluation_plan_path,
        budget_ledger_path=ledger_path,
        report_root=tmp_path / "report-recovery",
        working_directory=checkout,
        reporting_source_revision=(
            actual_revision if reporting_revision is None else reporting_revision
        ),
    )
    return raw, evaluation_plan.state_root, actual_revision


def test_complete_seal_is_self_hashed_append_only_and_idempotent(
    tmp_path: Path,
) -> None:
    raw, _evaluation_state, actual_revision = _sealed_report_plan(tmp_path)
    plan = parse_report_recovery_plan(raw)

    first = seal_report_recovery(plan)
    before = {
        path.name: path.read_bytes()
        for path in plan.report_root.iterdir()
        if path.is_file()
    }
    second = seal_report_recovery(plan)
    after = {
        path.name: path.read_bytes()
        for path in plan.report_root.iterdir()
        if path.is_file()
    }

    assert second == first
    assert after == before
    assert first["status"] == "recovery_report_sealed"
    assert first["reporting_source_revision"] == actual_revision
    assert first["evaluation_source_revision"] == SOURCE_REVISION
    assert first["within_limit"] is True
    assert first["incremental_cash_spent_usd"] == 5.0
    assert {
        "heldout-basis-audit.json",
        "experiment5-result.json",
        "EXPERIMENT_5.md",
        "recovery-envelope.json",
        "final.json",
        "recovery-handoff.json",
    } == set(before)
    audit = json.loads(
        (plan.report_root / "heldout-basis-audit.json").read_text(encoding="ascii")
    )
    audit_body = dict(audit)
    claimed = audit_body.pop("audit_sha256")
    assert claimed == canonical_sha256(audit_body)
    assert audit["schema_version"] == BASIS_AUDIT_SCHEMA_VERSION
    assert len(audit["cells"]) == 18
    assert audit["basis"]["valid_tokens"] == 250_000
    assert report_recovery_status(plan)["status"] == "sealed"


def test_parent_barrier_tamper_fails_before_writing_outputs(
    tmp_path: Path,
) -> None:
    raw, evaluation_state, _actual_revision = _sealed_report_plan(tmp_path)
    plan = parse_report_recovery_plan(raw)
    barrier = evaluation_state / "evaluation-complete.json"
    original = barrier.read_bytes()
    barrier.write_bytes(original + b" ")

    with pytest.raises(
        Experiment5ReportRecoveryError,
        match="barrier file changed",
    ):
        seal_report_recovery(plan)
    assert not plan.report_root.exists()


def test_exact_sequence_identity_is_required_even_with_valid_pair_hashes(
    tmp_path: Path,
) -> None:
    def mutate(result: dict, paired: dict) -> None:
        if (
            result["panel"] == "sae"
            and result["seed"] == 1
            and result["checkpoint_target_tokens"] == 2_000_000
        ):
            paired["packed_sequence_ids"] = [1001, 1003]

    raw, _evaluation_state, _actual_revision = _sealed_report_plan(
        tmp_path,
        mutate=mutate,
    )
    plan = parse_report_recovery_plan(raw)

    with pytest.raises(
        Experiment5ReportRecoveryError,
        match="held-out sequence basis changed",
    ):
        seal_report_recovery(plan)
    assert not plan.report_root.exists()


def test_per_sequence_sst_tolerance_breach_fails_closed(
    tmp_path: Path,
) -> None:
    def mutate(result: dict, paired: dict) -> None:
        if (
            result["panel"] == "sae"
            and result["seed"] == 1
            and result["checkpoint_target_tokens"] == 2_000_000
        ):
            paired["sequence_sst"] = [500.01, 499.99]

    raw, _evaluation_state, _actual_revision = _sealed_report_plan(
        tmp_path,
        mutate=mutate,
    )
    plan = parse_report_recovery_plan(raw)

    with pytest.raises(
        Experiment5ReportRecoveryError,
        match="per-sequence SST tolerance breached",
    ):
        seal_report_recovery(plan)
    assert not plan.report_root.exists()


@pytest.mark.parametrize(
    ("changed_panel", "changed_milestone"),
    [
        ("transcoder", None),
        (None, 5_000_000),
    ],
)
def test_sst_basis_is_global_across_panels_and_milestones(
    tmp_path: Path,
    changed_panel: str | None,
    changed_milestone: int | None,
) -> None:
    def mutate(result: dict, paired: dict) -> None:
        if (
            (changed_panel is None or result["panel"] == changed_panel)
            and (
                changed_milestone is None
                or result["checkpoint_target_tokens"] == changed_milestone
            )
        ):
            # Keep the aggregate SST unchanged so the paired result remains
            # internally valid.  Applying the same drift to an entire panel or
            # milestone proves the audit uses one matrix-wide reference rather
            # than a separate reference inside each subgroup.
            paired["sequence_sst"] = [500.01, 499.99]

    raw, _evaluation_state, _actual_revision = _sealed_report_plan(
        tmp_path,
        mutate=mutate,
    )
    plan = parse_report_recovery_plan(raw)

    with pytest.raises(
        Experiment5ReportRecoveryError,
        match="per-sequence SST tolerance breached",
    ):
        seal_report_recovery(plan)
    assert not plan.report_root.exists()


def test_reporting_checkout_revision_must_match_plan(
    tmp_path: Path,
) -> None:
    raw, _evaluation_state, actual_revision = _sealed_report_plan(
        tmp_path,
        reporting_revision=REPORTING_REVISION_MISMATCH,
    )
    assert actual_revision != REPORTING_REVISION_MISMATCH
    plan = parse_report_recovery_plan(raw)

    with pytest.raises(
        Experiment5ReportRecoveryError,
        match="checkout revision differs",
    ):
        seal_report_recovery(plan)
    assert not plan.report_root.exists()


def test_existing_output_byte_mismatch_is_terminal(tmp_path: Path) -> None:
    raw, _evaluation_state, _actual_revision = _sealed_report_plan(tmp_path)
    plan = parse_report_recovery_plan(raw)
    plan.report_root.mkdir(parents=True)
    conflicting = plan.report_root / "heldout-basis-audit.json"
    conflicting.write_text("{}\n", encoding="ascii")

    with pytest.raises(
        Experiment5ReportRecoveryError,
        match="existing held-out basis audit changed",
    ):
        seal_report_recovery(plan)
    assert (
        hashlib.sha256(conflicting.read_bytes()).hexdigest()
        == hashlib.sha256(b"{}\n").hexdigest()
    )


def test_partial_seal_resumes_without_replacing_published_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, _evaluation_state, _actual_revision = _sealed_report_plan(tmp_path)
    plan = parse_report_recovery_plan(raw)
    original_seal = report_recovery._seal_exact_bytes
    published = 0

    def interrupt_after_three(
        path: Path,
        payload: bytes,
        *,
        name: str,
    ) -> str:
        nonlocal published
        digest = original_seal(path, payload, name=name)
        published += 1
        if published == 3:
            raise RuntimeError("simulated controller interruption")
        return digest

    monkeypatch.setattr(
        report_recovery,
        "_seal_exact_bytes",
        interrupt_after_three,
    )
    with pytest.raises(RuntimeError, match="simulated controller interruption"):
        seal_report_recovery(plan)
    paths_before = {
        path.name: path.read_bytes()
        for path in plan.report_root.iterdir()
        if path.is_file()
    }
    assert set(paths_before) == {
        "heldout-basis-audit.json",
        "experiment5-result.json",
        "EXPERIMENT_5.md",
    }

    monkeypatch.setattr(report_recovery, "_seal_exact_bytes", original_seal)
    seal_report_recovery(plan)

    for name, payload in paths_before.items():
        assert (plan.report_root / name).read_bytes() == payload
    assert verify_report_recovery(plan)["status"] == "recovery_report_sealed"


def test_status_reopens_and_verifies_every_sealed_artifact(
    tmp_path: Path,
) -> None:
    raw, _evaluation_state, _actual_revision = _sealed_report_plan(tmp_path)
    plan = parse_report_recovery_plan(raw)
    seal_report_recovery(plan)

    for name in (
        "heldout-basis-audit.json",
        "experiment5-result.json",
        "EXPERIMENT_5.md",
        "recovery-envelope.json",
        "final.json",
        "recovery-handoff.json",
    ):
        path = plan.report_root / name
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with pytest.raises(
            Experiment5ReportRecoveryError,
            match="sealed .* changed",
        ):
            report_recovery_status(plan)
        path.write_bytes(original)

    assert report_recovery_status(plan)["status"] == "sealed"


@pytest.mark.parametrize(
    "field",
    [
        "evaluation_plan_file_sha256",
        "evaluation_barrier_file_sha256",
        "budget_ledger_file_sha256",
    ],
)
def test_handoff_parent_file_hashes_are_plan_bound(
    tmp_path: Path,
    field: str,
) -> None:
    raw, _evaluation_state, _actual_revision = _sealed_report_plan(tmp_path)
    plan = parse_report_recovery_plan(raw)
    changed = copy.deepcopy(seal_report_recovery(plan))
    changed[field] = "0" * 64 if changed[field] != "0" * 64 else "1" * 64
    changed_body = dict(changed)
    changed_body.pop("handoff_sha256")
    changed["handoff_sha256"] = canonical_sha256(changed_body)

    with pytest.raises(
        Experiment5ReportRecoveryError,
        match="handoff differs from its plan",
    ):
        validate_report_recovery_handoff(changed, plan=plan)


def test_handoff_final_path_is_plan_bound(tmp_path: Path) -> None:
    raw, _evaluation_state, _actual_revision = _sealed_report_plan(tmp_path)
    plan = parse_report_recovery_plan(raw)
    changed = copy.deepcopy(seal_report_recovery(plan))
    changed["final"]["path"] = str(tmp_path / "other-final.json")
    changed_body = dict(changed)
    changed_body.pop("handoff_sha256")
    changed["handoff_sha256"] = canonical_sha256(changed_body)

    with pytest.raises(
        Experiment5ReportRecoveryError,
        match="handoff differs from its plan",
    ):
        validate_report_recovery_handoff(changed, plan=plan)


def test_plan_yaml_round_trips_and_is_byte_immutable(tmp_path: Path) -> None:
    raw, _evaluation_state, _actual_revision = _sealed_report_plan(tmp_path)
    output = tmp_path / "report-plan.yaml"

    first = write_report_recovery_plan(output, raw)
    before = output.read_bytes()
    second = write_report_recovery_plan(output, raw)
    assert second.plan_sha256 == first.plan_sha256
    assert output.read_bytes() == before
    assert load_report_recovery_plan(output).plan_sha256 == first.plan_sha256

    output.write_bytes(before + b"\n")
    with pytest.raises(
        Experiment5ReportRecoveryError,
        match="existing Experiment-5 report-recovery plan changed",
    ):
        write_report_recovery_plan(output, raw)


def test_plan_semantic_tamper_without_rehash_is_rejected(tmp_path: Path) -> None:
    raw, _evaluation_state, _actual_revision = _sealed_report_plan(tmp_path)
    changed = copy.deepcopy(raw)
    changed["recovery_id"] = "different-recovery"

    with pytest.raises(
        Experiment5ReportRecoveryError,
        match="plan self-hash is invalid",
    ):
        parse_report_recovery_plan(changed)
