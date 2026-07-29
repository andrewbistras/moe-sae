from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import moe_sae.experiment5_campaign as experiment5_campaign
from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_campaign import (
    CONDITIONS,
    CONTINUOUS_RUNTIME_SECONDS,
    MILESTONES,
    PANELS,
    SEEDS,
    AttemptExecution,
    Experiment5BudgetStop,
    Experiment5Campaign,
    Experiment5CampaignPlanError,
    Experiment5CampaignStateError,
    ResourceReadinessObservation,
    ResourceReadinessStatus,
    _default_resource_readiness_probe,
    build_evaluation_command,
    build_train_command,
    experiment5_campaign_status,
    parse_experiment5_campaign_plan,
)


def _readiness(
    status: ResourceReadinessStatus = ResourceReadinessStatus.READY,
    *,
    state: str | None = None,
    health: str | None = None,
    create_time: str | None = "2026-07-29T10:11:12.123Z",
    spot: bool | None = True,
    absent: bool = False,
    detail: str = "test observation",
) -> ResourceReadinessObservation:
    if state is None and status is ResourceReadinessStatus.READY:
        state = "READY"
    if health is None and status is ResourceReadinessStatus.READY:
        health = "HEALTHY"
    return ResourceReadinessObservation(
        status=status,
        state=state,
        health=health,
        create_time=create_time,
        spot=spot,
        absent=absent,
        detail=detail,
    )


def _sealed(path: Path, payload: bytes) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "controller_path": str(path),
        "remote_path": f"/remote/evidence/{path.parent.name}/{path.name}",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _raw_plan(tmp_path: Path) -> dict:
    commit = "0" * 40
    resources = []
    resource_ids = []
    for index in range(7):
        resource_id = f"slice-{index}"
        resource_ids.append(resource_id)
        evidence = tmp_path / "evidence" / resource_id
        resources.append(
            {
                "id": resource_id,
                "project": "test-project",
                "tpu_name": f"experiment5-{index}",
                "zone": "us-test1-a",
                "region": "us-test1",
                "accelerator_type": "v6e-16",
                "internal_ip": True,
                "ssh_user": "researcher",
                "cache_profile": "paired",
                "deployed_commit": commit,
                "evidence": {
                    "controller_mapping": _sealed(
                        evidence / "mapping.json",
                        f"mapping-{index}".encode(),
                    ),
                    "cache_inventory": _sealed(
                        evidence / "cache.json",
                        f"cache-{index}".encode(),
                    ),
                    "mirror_seal": _sealed(
                        evidence / "mirror.json",
                        f"mirror-{index}".encode(),
                    ),
                },
                "environment": {},
            }
        )

    trajectories = []
    evaluations = []
    for panel in PANELS:
        for seed in SEEDS:
            control = f"/remote/control/{panel}/seed{seed}"
            checkpoint = f"gs://test-bucket/checkpoints/{panel}/seed{seed}"
            config = f"/remote/configs/gemma4_hybrid_{panel}_v1.yaml"
            train_source = (
                f"/remote/overlay/train/{seed}" if panel == "transcoder" else None
            )
            train_source_objects = (
                f"gs://test-bucket/overlay/train/{seed}"
                if panel == "transcoder"
                else None
            )
            for condition in CONDITIONS:
                run_id = f"experiment5-{panel}-{condition.lower()}-seed{seed}"
                trajectories.append(
                    {
                        "id": f"train-{panel}-{condition.lower()}-seed{seed}",
                        "panel": panel,
                        "condition": condition,
                        "seed": seed,
                        "eligible_resource_ids": list(resource_ids),
                        "run_id": run_id,
                        "config_remote": config,
                        "phase": "routed_shared_hybrid_10m",
                        "control_root_remote": control,
                        "checkpoint_root_gcs": checkpoint,
                        "source_overlay_control_root_remote": train_source,
                        "source_overlay_object_prefix_gcs": train_source_objects,
                        "completion_path_controller": str(
                            tmp_path
                            / "results"
                            / "runs"
                            / run_id
                            / "training-result.json"
                        ),
                        "environment": {},
                    }
                )
            for milestone in MILESTONES:
                result_root = (
                    tmp_path
                    / "results"
                    / "evaluation"
                    / panel
                    / f"seed{seed}"
                    / str(milestone)
                )
                evaluations.append(
                    {
                        "id": f"eval-{panel}-seed{seed}-{milestone}",
                        "panel": panel,
                        "seed": seed,
                        "milestone_tokens": milestone,
                        "eligible_resource_ids": list(resource_ids),
                        "config_remote": config,
                        "phase": "routed_shared_hybrid_10m",
                        "control_root_remote": control,
                        "checkpoint_root_gcs": checkpoint,
                        "train_source_root_remote": train_source,
                        "eval_source_root_remote": (
                            "/remote/overlay/eval" if panel == "transcoder" else None
                        ),
                        "train_source_object_prefix_gcs": train_source_objects,
                        "eval_source_object_prefix_gcs": (
                            "gs://test-bucket/overlay/eval"
                            if panel == "transcoder"
                            else None
                        ),
                        "result_root_remote": (
                            f"/remote/evaluation/{panel}/seed{seed}/{milestone}"
                        ),
                        "completion_path_controller": str(
                            result_root / "scientific-result.json"
                        ),
                        "paired_scores_path_controller": str(
                            result_root / "paired-scores.json"
                        ),
                        "run_roots_remote": {
                            condition: (
                                f"{control}/runs/experiment5-{panel}-"
                                f"{condition.lower()}-seed{seed}"
                            )
                            for condition in CONDITIONS
                        },
                        "environment": {},
                    }
                )
    return {
        "schema_version": "gemma4-experiment5-campaign-plan-v1",
        "campaign_id": "experiment5-test",
        "state_root": str(tmp_path / "state"),
        "working_directory": str(tmp_path / "repo"),
        "python_executable": sys.executable,
        "remote_project_root": "/remote/repo",
        "remote_uv": "/remote/bin/uv",
        "project_commit": commit,
        "max_parallel_train": 7,
        "max_parallel_eval": 7,
        "retry_policy": {
            "max_transient_attempts": 3,
            "initial_backoff_seconds": 0,
            "maximum_backoff_seconds": 0,
        },
        "cost_policy": {
            "hard_limit_usd": 50,
            "contingency_usd": 5,
            "already_spent_usd": 0,
            "fixed_cost_reserve_usd": 1,
            "training_attempt_reserve_usd": 0.1,
            "evaluation_attempt_reserve_usd": 0.1,
        },
        "environment": {
            "CONTRIBUTION_SAE_CACHE_MAX_BYTES": "171798691840",
        },
        "resources": resources,
        "trajectories": trajectories,
        "evaluations": evaluations,
    }


def _self_hashed(body: dict, field: str) -> dict:
    return {**body, field: canonical_sha256(body)}


def _write_training_completion(
    plan,
    task_id: str,
    *,
    source_revision: str | None = None,
) -> None:
    task = next(item for item in plan.trajectories if item.task_id == task_id)
    body = {
        "schema_version": "gemma4-contribution-sae-training-result-v1",
        "status": "completed",
        "tokens": 10_000_000,
        "source_revision": (
            plan.project_commit if source_revision is None else source_revision
        ),
        "run_identity": {
            "plan": {
                "phase_name": "routed_shared_hybrid_10m",
                "condition": task.condition,
                "seed": task.seed,
                "total_tokens": 10_000_000,
            }
        },
    }
    task.completion_path_controller.parent.mkdir(parents=True, exist_ok=True)
    task.completion_path_controller.write_text(
        json.dumps(_self_hashed(body, "result_sha256"), sort_keys=True),
        encoding="ascii",
    )


def test_training_completion_rejects_wrong_source_revision(tmp_path: Path) -> None:
    plan = parse_experiment5_campaign_plan(_raw_plan(tmp_path))
    task = plan.trajectories[0]
    _write_training_completion(
        plan,
        task.task_id,
        source_revision="1" * 40,
    )

    with pytest.raises(
        Experiment5CampaignStateError,
        match="training completion role changed",
    ):
        experiment5_campaign.inspect_training_completion(plan, task)


def _write_evaluation_completion(plan, task_id: str) -> None:
    task = next(item for item in plan.evaluations if item.task_id == task_id)
    paired_body = {
        "schema_version": "gemma4-routed-shared-hybrid-paired-scores-v1",
        "protocol_id": "gemma4-routed-shared-hybrid-v1-20260728",
        "panel": task.panel,
        "seed": task.seed,
        "checkpoint_target_tokens": task.milestone_tokens,
    }
    paired = _self_hashed(paired_body, "artifact_sha256")
    result_body = {
        "schema_version": "gemma4-routed-shared-hybrid-scientific-result-v1",
        "protocol_id": "gemma4-routed-shared-hybrid-v1-20260728",
        "status": "curve_point_measured",
        "panel": task.panel,
        "seed": task.seed,
        "checkpoint_target_tokens": task.milestone_tokens,
        "source_revision": plan.project_commit,
        "predictions_artifact_sha256": paired["artifact_sha256"],
        "conditions": {
            condition: {"training": {"marginal_chip_seconds": 1.0}}
            for condition in CONDITIONS
        },
    }
    task.completion_path_controller.parent.mkdir(parents=True, exist_ok=True)
    task.paired_scores_path_controller.write_text(
        json.dumps(paired, sort_keys=True),
        encoding="ascii",
    )
    task.completion_path_controller.write_text(
        json.dumps(_self_hashed(result_body, "result_sha256"), sort_keys=True),
        encoding="ascii",
    )


def _install_report_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep campaign tests small; report.py has its own full-contract suite."""

    class Validated:
        def __init__(self, value: dict) -> None:
            self.panel = value["panel"]
            self.seed = value["seed"]
            self.milestone = value["checkpoint_target_tokens"]
            self.costs = {
                condition: {
                    "marginal_chip_seconds": value["conditions"][condition]["training"][
                        "marginal_chip_seconds"
                    ]
                }
                for condition in CONDITIONS
            }

    def validate_result(value):
        return Validated(dict(value))

    def build_report(*, results, paired_scores, budget_ledger):
        identities = set()
        for path in results:
            value = json.loads(Path(path).read_text(encoding="ascii"))
            identities.add(
                (
                    value["panel"],
                    value["seed"],
                    value["checkpoint_target_tokens"],
                )
            )
        assert identities == {
            (panel, seed, milestone)
            for panel in PANELS
            for seed in SEEDS
            for milestone in MILESTONES
        }
        assert len(paired_scores) == 18
        ledger = json.loads(Path(budget_ledger).read_text(encoding="ascii"))
        body = {
            "kind": "test-experiment5-report",
            "evaluation_identities": sorted(
                [list(identity) for identity in identities]
            ),
            "ledger_sha256": ledger["ledger_sha256"],
        }
        return _self_hashed(body, "report_sha256")

    def validate_report(value):
        selected = dict(value)
        claimed = selected.pop("report_sha256")
        assert claimed == canonical_sha256(selected)
        return dict(value)

    monkeypatch.setattr(
        experiment5_campaign,
        "validate_experiment5_result",
        validate_result,
    )
    monkeypatch.setattr(
        experiment5_campaign,
        "build_experiment5_report",
        build_report,
    )
    monkeypatch.setattr(
        experiment5_campaign,
        "validate_experiment5_report",
        validate_report,
    )
    monkeypatch.setattr(
        experiment5_campaign,
        "render_experiment5_report",
        lambda value: (
            "# Test Experiment 5\n\n" f"Report: `{value['report_sha256']}`\n"
        ),
    )


def test_exact_matrix_and_commands(tmp_path: Path) -> None:
    plan = parse_experiment5_campaign_plan(_raw_plan(tmp_path))

    assert len(plan.resources) == 7
    assert len(plan.trajectories) == 24
    assert len(plan.evaluations) == 18
    assert {(task.panel, task.condition, task.seed) for task in plan.trajectories} == {
        (panel, condition, seed)
        for panel in PANELS
        for condition in CONDITIONS
        for seed in SEEDS
    }
    train = build_train_command(plan, plan.trajectories[0], "attempt-1")
    assert train[train.index("--maximum-runtime-seconds") + 1] == str(
        CONTINUOUS_RUNTIME_SECONDS
    )
    assert train[train.index("--config") + 1].startswith("/remote/")
    assert train[train.index("--controller-config") + 1] == str(
        plan.working_directory / "configs" / "gemma4_hybrid_sae_v1.yaml"
    )
    evaluation = build_evaluation_command(plan, plan.evaluations[0])
    assert "evaluate_experiment5.py" in evaluation[1]
    assert "--controller-config" not in evaluation
    for condition in CONDITIONS:
        assert f"--run-root-{condition.lower()}" in evaluation


def test_transcoder_requires_paired_cache_and_overlay(tmp_path: Path) -> None:
    raw = _raw_plan(tmp_path)
    raw["resources"][0]["cache_profile"] = "target"
    with pytest.raises(Experiment5CampaignPlanError, match="target-only cache"):
        parse_experiment5_campaign_plan(raw)

    raw = _raw_plan(tmp_path)
    transcoder = next(
        task for task in raw["trajectories"] if task["panel"] == "transcoder"
    )
    transcoder["source_overlay_control_root_remote"] = None
    with pytest.raises(
        Experiment5CampaignPlanError,
        match="required for the transcoder panel",
    ):
        parse_experiment5_campaign_plan(raw)


def test_sae_prohibits_transcoder_overlay(tmp_path: Path) -> None:
    raw = _raw_plan(tmp_path)
    sae = next(task for task in raw["evaluations"] if task["panel"] == "sae")
    sae["eval_source_root_remote"] = "/wrong/overlay"
    with pytest.raises(Experiment5CampaignPlanError, match="null for the SAE panel"):
        parse_experiment5_campaign_plan(raw)


def test_h_and_f_must_share_accelerator_generation(tmp_path: Path) -> None:
    raw = _raw_plan(tmp_path)
    raw["resources"][0]["accelerator_type"] = "v4-32"
    raw["resources"][0]["internal_ip"] = False
    raw["resources"][0]["evidence"]["mirror_seal"] = None
    for task in (*raw["trajectories"], *raw["evaluations"]):
        task["eligible_resource_ids"] = ["slice-1"]
    h = next(
        task
        for task in raw["trajectories"]
        if task["panel"] == "sae" and task["seed"] == 0 and task["condition"] == "H"
    )
    f = next(
        task
        for task in raw["trajectories"]
        if task["panel"] == "sae" and task["seed"] == 0 and task["condition"] == "F"
    )
    h["eligible_resource_ids"] = ["slice-0"]
    f["eligible_resource_ids"] = ["slice-1"]
    with pytest.raises(Experiment5CampaignPlanError, match="H/F"):
        parse_experiment5_campaign_plan(raw)


def test_campaign_serializes_each_resource_retries_transient_and_seals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_report_stubs(monkeypatch)
    plan = parse_experiment5_campaign_plan(_raw_plan(tmp_path))
    active: set[str] = set()
    calls: dict[str, int] = {}
    maximum_active = 0
    lock = threading.Lock()
    observed_environments: dict[str, dict[str, str]] = {}

    def runner(task_id, _command, environment, _cwd, stdout, _stop):
        nonlocal maximum_active
        resource = environment["TPU_NAME"]
        with lock:
            assert resource not in active
            active.add(resource)
            maximum_active = max(maximum_active, len(active))
            calls[task_id] = calls.get(task_id, 0) + 1
            attempt = calls[task_id]
        try:
            stdout.parent.mkdir(parents=True, exist_ok=True)
            if task_id == "train-sae-h-seed0" and attempt == 1:
                stdout.write_text(
                    "ssh: connect to host failed: Connection timed out\n",
                    encoding="ascii",
                )
                return AttemptExecution(255)
            time.sleep(0.01)
            stdout.write_text("ok\n", encoding="ascii")
            observed_environments[task_id] = dict(environment)
            if task_id.startswith("train-"):
                _write_training_completion(plan, task_id)
            else:
                _write_evaluation_completion(plan, task_id)
            return AttemptExecution(0)
        finally:
            with lock:
                active.remove(resource)

    result = Experiment5Campaign(
        plan,
        attempt_runner=runner,
        resource_readiness_probe=lambda _resource: _readiness(),
        verify_checkout=False,
    ).run()

    assert result["status"] == "completed"
    assert len(result["training"]) == 24
    assert len(result["evaluations"]) == 18
    assert result["report"]["quota_funded_tpu_accounting"] == {
        "training_chip_seconds": 24.0,
        "training_scope": (
            "10M cumulative marginal training time; both panels, "
            "three seeds, and H/R/B/F exactly once"
        ),
        "evaluation_chip_seconds": None,
        "evaluation_scope": ("absent from the Experiment-5 scientific-result contract"),
    }
    report_root = plan.state_root / "report"
    ledger = json.loads(
        (report_root / "budget-ledger.json").read_text(encoding="ascii")
    )
    assert ledger["quota_funded_tpu_chip_seconds"] == 24.0
    assert [row["category"] for row in ledger["line_items"]] == [
        "already_spent_before_durable_campaign",
        "durable_campaign_infrastructure_through_evaluation_completion",
    ]
    assert all("attempt" not in row["category"] for row in ledger["line_items"])
    assert (report_root / "experiment5-result.json").is_file()
    assert (report_root / "EXPERIMENT_5.md").is_file()
    assert calls["train-sae-h-seed0"] == 2
    assert maximum_active > 1
    assert (
        not {
            "CONTRIBUTION_SOURCE_OVERLAY_ROOT",
            "CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT",
            "CONTRIBUTION_SOURCE_OVERLAY_PREFIX",
            "CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX",
        }
        & observed_environments["eval-sae-seed0-2000000"].keys()
    )
    transcoder = observed_environments["eval-transcoder-seed0-2000000"]
    assert transcoder["CONTRIBUTION_SOURCE_OVERLAY_ROOT"].startswith("/remote/")
    assert transcoder["CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX"].startswith("gs://")
    selected_resource = next(
        resource
        for resource in plan.resources
        if resource.tpu_name == transcoder["TPU_NAME"]
    )
    assert transcoder["CONTRIBUTION_FRESH_CONTROLLER_MAPPING"] == str(
        selected_resource.controller_mapping.controller_path
    )


def _report_ready_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Experiment5Campaign, dict]:
    _install_report_stubs(monkeypatch)
    raw = _raw_plan(tmp_path)
    raw["cost_policy"]["already_spent_usd"] = 2.0
    plan = parse_experiment5_campaign_plan(raw)
    campaign = Experiment5Campaign(plan, verify_checkout=False)
    campaign._bind_campaign_clock()
    clock = json.loads(
        (plan.state_root / "campaign-clock.json").read_text(encoding="ascii")
    )
    started = datetime.fromisoformat(clock["started_at"].replace("Z", "+00:00"))
    completed = (started + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    completions = {}
    for task in plan.evaluations:
        _write_evaluation_completion(plan, task.task_id)
        fingerprint = experiment5_campaign.inspect_evaluation_completion(plan, task)
        assert fingerprint is not None
        completions[task.task_id] = fingerprint.to_dict()
    barrier = experiment5_campaign._record(
        "evaluation_phase_completion",
        {
            "campaign_id": plan.campaign_id,
            "plan_sha256": plan.plan_sha256,
            "status": "evaluation_completed",
            "evaluations": {key: completions[key] for key in sorted(completions)},
            "completed_at": completed,
        },
    )
    return campaign, barrier


def test_auto_report_resume_is_exact_and_does_not_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, barrier = _report_ready_campaign(tmp_path, monkeypatch)
    seal_exact_bytes = experiment5_campaign._seal_exact_bytes

    def interrupt_before_json(path, payload, *, name):
        if path.name == "experiment5-result.json":
            raise RuntimeError("simulated controller interruption")
        return seal_exact_bytes(path, payload, name=name)

    monkeypatch.setattr(
        experiment5_campaign,
        "_seal_exact_bytes",
        interrupt_before_json,
    )
    with pytest.raises(RuntimeError, match="simulated controller interruption"):
        campaign._seal_scientific_report(evaluation_barrier=barrier)
    assert (campaign.plan.state_root / "report" / "budget-ledger.json").is_file()
    assert not (
        campaign.plan.state_root / "report" / "experiment5-result.json"
    ).exists()

    monkeypatch.setattr(
        experiment5_campaign,
        "_seal_exact_bytes",
        seal_exact_bytes,
    )
    first = campaign._seal_scientific_report(evaluation_barrier=barrier)
    paths = [
        Path(first["budget_ledger_path"]),
        Path(first["report_path"]),
        Path(first["markdown_path"]),
    ]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}

    second = campaign._seal_scientific_report(evaluation_barrier=barrier)

    assert second == first
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths
    } == before
    ledger = json.loads(Path(first["budget_ledger_path"]).read_text(encoding="ascii"))
    assert ledger["incremental_cash_spent"] == pytest.approx(3.25)
    assert ledger["line_items"][0]["amount_usd"] == 2.0
    assert ledger["line_items"][1]["amount_usd"] == pytest.approx(1.25)


@pytest.mark.parametrize(
    "artifact_name",
    ["budget-ledger.json", "experiment5-result.json"],
)
def test_auto_report_fails_closed_on_wrong_existing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    campaign, barrier = _report_ready_campaign(tmp_path, monkeypatch)
    identity = campaign._seal_scientific_report(evaluation_barrier=barrier)
    artifact = Path(identity["report_path"]).parent / artifact_name
    artifact.write_bytes(artifact.read_bytes() + b" ")

    with pytest.raises(Experiment5CampaignStateError, match="changed"):
        campaign._seal_scientific_report(evaluation_barrier=barrier)


def test_auto_report_requires_exact_eighteen_evaluation_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, barrier = _report_ready_campaign(tmp_path, monkeypatch)
    assert len(campaign.plan.trajectories) == 24
    assert len(barrier["evaluations"]) == 18
    incomplete = copy.deepcopy(barrier)
    incomplete["evaluations"].pop(next(iter(incomplete["evaluations"])))

    with pytest.raises(
        Experiment5CampaignStateError,
        match="exact 18-task report matrix",
    ):
        campaign._seal_scientific_report(evaluation_barrier=incomplete)


def test_preflight_ignores_missing_evidence_for_inactive_placeholder(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path)
    inactive_id = "slice-6"
    for task in (*raw["trajectories"], *raw["evaluations"]):
        task["eligible_resource_ids"].remove(inactive_id)
    plan = parse_experiment5_campaign_plan(raw)
    inactive = plan.resource(inactive_id)
    inactive.controller_mapping.controller_path.unlink()
    inactive.cache_inventory.controller_path.unlink()
    assert inactive.mirror_seal is not None
    inactive.mirror_seal.controller_path.unlink()

    experiment5_campaign._preflight(plan, verify_checkout=False)


def test_preflight_rejects_missing_active_resource_evidence(
    tmp_path: Path,
) -> None:
    plan = parse_experiment5_campaign_plan(_raw_plan(tmp_path))
    active = plan.resource("slice-0")
    active.controller_mapping.controller_path.unlink()

    with pytest.raises(
        Experiment5CampaignStateError,
        match="slice-0 controller mapping is not a regular file",
    ):
        experiment5_campaign._preflight(plan, verify_checkout=False)


def test_inactive_placeholder_is_never_probed_or_run(tmp_path: Path) -> None:
    raw = _raw_plan(tmp_path)
    inactive_id = "slice-6"
    for item in (*raw["trajectories"], *raw["evaluations"]):
        item["eligible_resource_ids"].remove(inactive_id)
    plan = parse_experiment5_campaign_plan(raw)
    task = plan.trajectories[0]
    probed: list[str] = []
    executed_on: list[str] = []

    def readiness(resource):
        assert resource.resource_id != inactive_id
        probed.append(resource.resource_id)
        return _readiness()

    def runner(task_id, _command, environment, _cwd, stdout, _stop):
        executed_on.append(environment["TPU_NAME"])
        assert environment["TPU_NAME"] != plan.resource(inactive_id).tpu_name
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text("ok\n", encoding="ascii")
        _write_training_completion(plan, task_id)
        return AttemptExecution(0)

    result = Experiment5Campaign(
        plan,
        attempt_runner=runner,
        resource_readiness_probe=readiness,
        verify_checkout=False,
    )._run_phase([task], kind="training", max_parallel=1)

    assert set(result) == {task.task_id}
    assert probed
    assert executed_on
    assert inactive_id not in probed


def test_unready_resource_is_idle_without_attempt_or_cost_reservation(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path)
    for task in (*raw["trajectories"], *raw["evaluations"]):
        task["eligible_resource_ids"] = ["slice-0"]
    plan = parse_experiment5_campaign_plan(raw)
    task = plan.trajectories[0]
    probe_calls = 0
    runner_calls = 0

    def readiness(_resource):
        nonlocal probe_calls
        probe_calls += 1
        if probe_calls == 2:
            assert not (
                plan.state_root / "training" / task.task_id / "attempts"
            ).exists()
            assert not (plan.state_root / "budget" / "attempt-reservations").exists()
            return _readiness(
                ResourceReadinessStatus.FORMING,
                state="CREATING",
                health=None,
            )
        return _readiness()

    def runner(task_id, _command, _environment, _cwd, stdout, _stop):
        nonlocal runner_calls
        runner_calls += 1
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text("ok\n", encoding="ascii")
        _write_training_completion(plan, task_id)
        return AttemptExecution(0)

    campaign = Experiment5Campaign(
        plan,
        attempt_runner=runner,
        resource_readiness_probe=readiness,
        resource_readiness_poll_seconds=0.001,
        verify_checkout=False,
    )
    result = campaign._run_phase([task], kind="training", max_parallel=1)

    assert set(result) == {task.task_id}
    assert probe_calls == 4
    assert runner_calls == 1
    assert len(campaign._reservation_records()) == 1
    attempts = (plan.state_root / "training" / task.task_id / "attempts").iterdir()
    assert len(list(attempts)) == 1


def test_terminal_resource_state_fails_promptly_and_is_audited(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path)
    for task in (*raw["trajectories"], *raw["evaluations"]):
        task["eligible_resource_ids"] = ["slice-0"]
    plan = parse_experiment5_campaign_plan(raw)
    task = plan.trajectories[0]
    campaign = Experiment5Campaign(
        plan,
        resource_readiness_probe=lambda _resource: _readiness(
            ResourceReadinessStatus.TERMINAL_REPLACEMENT_REQUIRED,
            state="PREEMPTED",
            health=None,
            detail="TPU entered terminal state PREEMPTED",
        ),
        verify_checkout=False,
    )

    started = time.monotonic()
    with pytest.raises(
        Experiment5CampaignStateError,
        match="requires replacement",
    ):
        campaign._run_phase([task], kind="training", max_parallel=1)
    assert time.monotonic() - started < 2
    events = list(
        (plan.state_root / "resource-readiness" / "slice-0").glob(
            "before-semaphore-*.json"
        )
    )
    assert len(events) == 1
    event = json.loads(events[0].read_text(encoding="ascii"))
    assert event["observation"]["status"] == "terminal_replacement_required"
    assert not (plan.state_root / "training" / task.task_id / "attempts").exists()


def test_unexpected_readiness_probe_exception_aborts_without_deadlock(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path)
    for task in (*raw["trajectories"], *raw["evaluations"]):
        task["eligible_resource_ids"] = ["slice-0"]
    plan = parse_experiment5_campaign_plan(raw)
    task = plan.trajectories[0]

    def exploding_probe(_resource):
        raise RuntimeError("probe exploded")

    campaign = Experiment5Campaign(
        plan,
        resource_readiness_probe=exploding_probe,
        verify_checkout=False,
    )
    started = time.monotonic()
    with pytest.raises(
        Experiment5CampaignStateError,
        match="RuntimeError: probe exploded",
    ):
        campaign._run_phase([task], kind="training", max_parallel=1)
    assert time.monotonic() - started < 2


def test_inflight_spot_preemption_is_resumable_and_not_transient_exhaustion(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path)
    raw["retry_policy"].update(
        {
            "max_transient_attempts": 1,
            "initial_backoff_seconds": 0,
            "maximum_backoff_seconds": 0,
        }
    )
    for task in (*raw["trajectories"], *raw["evaluations"]):
        task["eligible_resource_ids"] = ["slice-0"]
    plan = parse_experiment5_campaign_plan(raw)
    task = plan.trajectories[0]
    runner_calls = 0

    def runner(task_id, _command, _environment, _cwd, stdout, _stop):
        nonlocal runner_calls
        runner_calls += 1
        stdout.parent.mkdir(parents=True, exist_ok=True)
        if runner_calls == 1:
            stdout.write_text("TPU VM was preempted\n", encoding="ascii")
            return AttemptExecution(1)
        stdout.write_text("ok\n", encoding="ascii")
        _write_training_completion(plan, task_id)
        return AttemptExecution(0)

    campaign = Experiment5Campaign(
        plan,
        attempt_runner=runner,
        resource_readiness_probe=lambda _resource: _readiness(),
        verify_checkout=False,
    )
    result = campaign._run_phase([task], kind="training", max_parallel=1)

    assert set(result) == {task.task_id}
    assert runner_calls == 2
    assert len(campaign._reservation_records()) == 2
    first_result = (
        plan.state_root
        / "training"
        / task.task_id
        / "attempts"
        / "attempt-000001"
        / "result.json"
    )
    assert (
        json.loads(first_result.read_text(encoding="ascii"))["classification"]
        == "spot_preemption"
    )
    assert not (plan.state_root / "training" / task.task_id / "failure.json").exists()


def test_transport_failure_is_reclassified_only_by_confirmed_spot_loss(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path)
    raw["retry_policy"]["max_transient_attempts"] = 1
    for task in (*raw["trajectories"], *raw["evaluations"]):
        task["eligible_resource_ids"] = ["slice-0"]
    plan = parse_experiment5_campaign_plan(raw)
    task = plan.trajectories[0]

    def runner(_task_id, _command, _environment, _cwd, stdout, _stop):
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text("ssh: Connection closed by remote host\n", encoding="ascii")
        return AttemptExecution(255)

    campaign = Experiment5Campaign(
        plan,
        attempt_runner=runner,
        resource_readiness_probe=lambda _resource: _readiness(
            ResourceReadinessStatus.TERMINAL_REPLACEMENT_REQUIRED,
            state="PREEMPTED",
            health=None,
            detail="confirmed terminal spot state",
        ),
        verify_checkout=False,
    )
    with pytest.raises(
        Experiment5CampaignStateError,
        match="requires replacement after",
    ):
        campaign._execute_once(
            task,
            kind="training",
            resource=plan.resources[0],
            pre_attempt_readiness=_readiness(spot=True),
        )
    result_path = (
        plan.state_root
        / "training"
        / task.task_id
        / "attempts"
        / "attempt-000001"
        / "result.json"
    )
    result = json.loads(result_path.read_text(encoding="ascii"))
    assert result["classification"] == "spot_preemption"
    assert result["confirmed_preemption"] is True
    assert result["post_failure_readiness"]["state"] == "PREEMPTED"
    assert campaign._prior_state(task, kind="training") == (1, 0)
    assert not (plan.state_root / "training" / task.task_id / "failure.json").exists()


def test_unconfirmed_transport_failure_remains_bounded(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path)
    raw["retry_policy"]["max_transient_attempts"] = 1
    for task in (*raw["trajectories"], *raw["evaluations"]):
        task["eligible_resource_ids"] = ["slice-0"]
    plan = parse_experiment5_campaign_plan(raw)
    task = plan.trajectories[0]

    def runner(_task_id, _command, _environment, _cwd, stdout, _stop):
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text("ssh: Connection reset by peer\n", encoding="ascii")
        return AttemptExecution(255)

    campaign = Experiment5Campaign(
        plan,
        attempt_runner=runner,
        resource_readiness_probe=lambda _resource: _readiness(
            ResourceReadinessStatus.TRANSIENT_CONTROL_PLANE,
            state=None,
            health=None,
            spot=None,
            detail="describe API unavailable",
        ),
        verify_checkout=False,
    )
    with pytest.raises(
        Experiment5CampaignStateError,
        match="exhausted transient retries",
    ):
        campaign._execute_once(
            task,
            kind="training",
            resource=plan.resources[0],
            pre_attempt_readiness=_readiness(spot=True),
        )
    result_path = (
        plan.state_root
        / "training"
        / task.task_id
        / "attempts"
        / "attempt-000001"
        / "result.json"
    )
    result = json.loads(result_path.read_text(encoding="ascii"))
    assert result["classification"] == "ssh_transport"
    assert result["confirmed_preemption"] is False


def test_default_readiness_probe_requires_exact_ready_incarnation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = parse_experiment5_campaign_plan(_raw_plan(tmp_path))
    resource = plan.resources[0]
    resource.controller_mapping.controller_path.write_text("{}", encoding="ascii")
    create_time = "2026-07-29T10:11:12.123Z"
    monkeypatch.setattr(
        experiment5_campaign,
        "validate_controller_mapping_receipt",
        lambda _value: {
            "tpu_name": resource.tpu_name,
            "zone": resource.zone,
            "accelerator_type": resource.accelerator_type,
            "resource_create_time": create_time,
        },
    )
    description = {
        "name": (
            f"projects/{resource.project}/locations/{resource.zone}/nodes/"
            f"{resource.tpu_name}"
        ),
        "acceleratorType": resource.accelerator_type,
        "createTime": create_time,
        "state": "CREATING",
        "health": "HEALTHY",
        "schedulingConfig": {"spot": True},
    }

    def completed(value, *, returncode=0, stderr=""):
        return subprocess.CompletedProcess(
            args=[],
            returncode=returncode,
            stdout=json.dumps(value),
            stderr=stderr,
        )

    monkeypatch.setattr(
        experiment5_campaign.subprocess,
        "run",
        lambda *_args, **_kwargs: completed(
            {},
            returncode=1,
            stderr="NOT_FOUND: TPU node was not found",
        ),
    )
    absent = _default_resource_readiness_probe(resource)
    assert absent.status is ResourceReadinessStatus.FORMING
    assert absent.absent is True

    monkeypatch.setattr(
        experiment5_campaign.subprocess,
        "run",
        lambda *_args, **_kwargs: completed(description),
    )
    forming = _default_resource_readiness_probe(resource)
    assert forming.status is ResourceReadinessStatus.FORMING
    assert forming.state == "CREATING"

    description["state"] = "READY"
    ready = _default_resource_readiness_probe(resource)
    assert ready.status is ResourceReadinessStatus.READY
    assert ready.health == "HEALTHY"
    assert ready.spot is True

    description["health"] = "UNHEALTHY_MAINTENANCE"
    unhealthy = _default_resource_readiness_probe(resource)
    assert unhealthy.status is ResourceReadinessStatus.FORMING
    assert unhealthy.health == "UNHEALTHY_MAINTENANCE"
    description["health"] = "HEALTHY"

    description["createTime"] = "2026-07-29T10:11:13.123Z"
    replaced = _default_resource_readiness_probe(resource)
    assert replaced.status is ResourceReadinessStatus.TERMINAL_REPLACEMENT_REQUIRED
    description["createTime"] = create_time
    description["state"] = "PREEMPTED"
    terminal = _default_resource_readiness_probe(resource)
    assert terminal.status is ResourceReadinessStatus.TERMINAL_REPLACEMENT_REQUIRED
    description["state"] = "READY"
    del description["health"]
    with pytest.raises(Experiment5CampaignStateError, match="lacks.*health"):
        _default_resource_readiness_probe(resource)
    description["health"] = "HEALTHY"

    exact_name = description["name"]
    description["name"] = exact_name + "-other"
    with pytest.raises(Experiment5CampaignStateError, match="identity or state"):
        _default_resource_readiness_probe(resource)

    description["name"] = exact_name
    description["acceleratorType"] = "v4-32"
    with pytest.raises(Experiment5CampaignStateError, match="identity or state"):
        _default_resource_readiness_probe(resource)


def test_default_readiness_probe_fails_closed_on_malformed_or_unauthorized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = parse_experiment5_campaign_plan(_raw_plan(tmp_path))
    resource = plan.resources[0]
    resource.controller_mapping.controller_path.write_text("{}", encoding="ascii")
    monkeypatch.setattr(
        experiment5_campaign,
        "validate_controller_mapping_receipt",
        lambda _value: {
            "tpu_name": resource.tpu_name,
            "zone": resource.zone,
            "accelerator_type": resource.accelerator_type,
            "resource_create_time": "2026-07-29T10:11:12.123Z",
        },
    )
    monkeypatch.setattr(
        experiment5_campaign.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="{broken",
            stderr="",
        ),
    )
    with pytest.raises(Experiment5CampaignStateError, match="malformed JSON"):
        _default_resource_readiness_probe(resource)

    def timed_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["gcloud"], timeout=120)

    monkeypatch.setattr(experiment5_campaign.subprocess, "run", timed_out)
    transient = _default_resource_readiness_probe(resource)
    assert transient.status is ResourceReadinessStatus.TRANSIENT_CONTROL_PLANE

    monkeypatch.setattr(
        experiment5_campaign.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="PERMISSION_DENIED: caller does not have permission",
        ),
    )
    with pytest.raises(Experiment5CampaignStateError, match="unauthorized"):
        _default_resource_readiness_probe(resource)


def test_budget_reservations_stop_before_projecting_above_fifty(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path)
    raw["cost_policy"].update(
        {
            "fixed_cost_reserve_usd": 0,
            "training_attempt_reserve_usd": 1,
            "evaluation_attempt_reserve_usd": 0,
        }
    )
    plan = parse_experiment5_campaign_plan(raw)
    campaign = Experiment5Campaign(plan, verify_checkout=False)
    task = plan.trajectories[0]
    resource = plan.resources[0]

    for attempt in range(1, 51):
        campaign._reserve_attempt_cost(
            task,
            kind="training",
            attempt_number=attempt,
            resource=resource,
        )
    with pytest.raises(Experiment5BudgetStop, match=r"above \$50"):
        campaign._reserve_attempt_cost(
            task,
            kind="training",
            attempt_number=51,
            resource=resource,
        )
    assert (plan.state_root / "budget-stop.json").is_file()


def test_campaign_wall_cash_watchdog_is_finite_and_audited(
    tmp_path: Path,
) -> None:
    plan = parse_experiment5_campaign_plan(_raw_plan(tmp_path))
    campaign = Experiment5Campaign(plan, verify_checkout=False)
    campaign._bind_campaign_clock()
    maximum = campaign._maximum_campaign_wall_seconds()
    assert 0 < maximum < 48 * 3_600
    clock_path = plan.state_root / "campaign-clock.json"
    clock = json.loads(clock_path.read_text(encoding="ascii"))
    unsigned_clock = dict(clock)
    unsigned_clock.pop("record_sha256")
    unsigned_clock["started_at"] = (
        (datetime.now(UTC) - timedelta(seconds=maximum + 1))
        .isoformat()
        .replace("+00:00", "Z")
    )
    clock = {
        **unsigned_clock,
        "record_sha256": canonical_sha256(unsigned_clock),
    }
    clock_path.write_text(
        json.dumps(clock, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    campaign._activate_campaign_clock(clock)

    with pytest.raises(Experiment5BudgetStop, match=r"protected \$45"):
        campaign._check_campaign_watchdog()
    stop = plan.state_root / "campaign-wall-budget-stop.json"
    record = json.loads(stop.read_text(encoding="ascii"))
    assert record["reason"] == "campaign_wall_cash_watchdog"
    assert (
        record["projected_committed_usd"] >= plan.cost_policy.initial_launch_limit_usd
    )
    assert record["hard_limit_usd"] == 50
    status = experiment5_campaign_status(plan)
    assert status["campaign_clock"] == clock
    assert status["budget_stop"] is None
    assert status["campaign_wall_budget_stop"] == record
    assert (
        status["cost_projection"]["campaign_elapsed_seconds"]
        > status["cost_projection"]["maximum_campaign_wall_seconds"]
    )
    assert status["cost_projection"]["accrued_infrastructure_usd"] > 0
    assert (
        status["watchdog_projection"]["projected_committed_usd"]
        >= plan.cost_policy.initial_launch_limit_usd
    )


def test_plan_is_immutable_under_field_addition(tmp_path: Path) -> None:
    raw = copy.deepcopy(_raw_plan(tmp_path))
    raw["resources"][0]["unexpected"] = True
    with pytest.raises(Experiment5CampaignPlanError, match="strict schema"):
        parse_experiment5_campaign_plan(raw)
