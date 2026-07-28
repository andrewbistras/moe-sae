from __future__ import annotations

import copy
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import pytest

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
    build_evaluation_command,
    build_train_command,
    parse_experiment5_campaign_plan,
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


def _write_training_completion(plan, task_id: str) -> None:
    task = next(item for item in plan.trajectories if item.task_id == task_id)
    body = {
        "schema_version": "gemma4-contribution-sae-training-result-v1",
        "status": "completed",
        "tokens": 10_000_000,
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
    evaluation = build_evaluation_command(plan, plan.evaluations[0])
    assert "evaluate_experiment5.py" in evaluation[1]
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
) -> None:
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
            time.sleep(0.001)
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
        verify_checkout=False,
    ).run()

    assert result["status"] == "completed"
    assert len(result["training"]) == 24
    assert len(result["evaluations"]) == 18
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


def test_plan_is_immutable_under_field_addition(tmp_path: Path) -> None:
    raw = copy.deepcopy(_raw_plan(tmp_path))
    raw["resources"][0]["unexpected"] = True
    with pytest.raises(Experiment5CampaignPlanError, match="strict schema"):
        parse_experiment5_campaign_plan(raw)
