from __future__ import annotations

import copy
import hashlib
import json
import sys
import threading
from pathlib import Path

import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.direct_horizon_campaign import (
    CONDITIONS,
    CONTINUOUS_RUNTIME_SECONDS,
    MILESTONES,
    MODELS,
    AttemptExecution,
    DirectCampaign,
    DirectCampaignPlanError,
    DirectCampaignStateError,
    build_evaluation_command,
    build_train_command,
    direct_campaign_status,
    parse_direct_campaign_plan,
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
    resources = []
    trajectories = []
    evaluations = []
    roles = [(model, condition) for model in MODELS for condition in CONDITIONS]
    for index, (model, condition) in enumerate(roles):
        resource_id = f"slice-{model}-{condition.lower()}"
        evidence_root = tmp_path / "evidence" / resource_id
        resources.append(
            {
                "id": resource_id,
                "project": "test-project",
                "tpu_name": f"test-{model}-{condition.lower()}",
                "zone": "us-test1-a",
                "region": "us-test1",
                "accelerator_type": "v6e-16",
                "internal_ip": True,
                "ssh_user": "researcher",
                "evidence": {
                    "controller_mapping": _sealed(
                        evidence_root / "mapping.json",
                        f"mapping-{index}".encode(),
                    ),
                    "cache_inventory": _sealed(
                        evidence_root / "cache.json",
                        f"cache-{index}".encode(),
                    ),
                    "mirror_seal": _sealed(
                        evidence_root / "mirror.json",
                        b"one-region-mirror",
                    ),
                },
                "environment": {},
            }
        )
        run_id = f"horizon10m-{model}-{condition.lower()}-seed0"
        trajectories.append(
            {
                "id": f"train-{model}-{condition.lower()}",
                "scientific_model": model,
                "condition": condition,
                "seed": 0,
                "resource_id": resource_id,
                "run_id": run_id,
                "config_remote": f"/remote/contracts/{model}.yaml",
                "phase": "learning_horizon_10m",
                "control_root_remote": f"/remote/runs/{model}",
                "checkpoint_root_gcs": f"gs://test-bucket/checkpoints/{model}",
                "completion_path_controller": str(
                    tmp_path / "results" / "runs" / run_id / "training-result.json"
                ),
                "environment": {},
            }
        )
    for model in MODELS:
        resource_id = f"slice-{model}-a"
        for milestone in MILESTONES:
            evaluations.append(
                {
                    "id": f"eval-{model}-{milestone}",
                    "scientific_model": model,
                    "milestone_tokens": milestone,
                    "seed": 0,
                    "resource_id": resource_id,
                    "config_remote": f"/remote/contracts/{model}.yaml",
                    "phase": "learning_horizon_10m",
                    "control_root_remote": f"/remote/runs/{model}",
                    "checkpoint_root_gcs": (f"gs://test-bucket/checkpoints/{model}"),
                    "result_root_remote": (f"/remote/evaluation/{model}/{milestone}"),
                    "completion_path_controller": str(
                        tmp_path
                        / "results"
                        / "evaluation"
                        / model
                        / str(milestone)
                        / "scientific-result.json"
                    ),
                    "run_roots_remote": {
                        condition: (
                            f"/remote/runs/{model}/runs/"
                            f"horizon10m-{model}-{condition.lower()}-seed0"
                        )
                        for condition in CONDITIONS
                    },
                    "environment": {},
                }
            )
    return {
        "schema_version": "gemma4-horizon-direct-campaign-plan-v1",
        "campaign_id": "direct-horizon-test",
        "state_root": str(tmp_path / "state"),
        "working_directory": str(tmp_path / "repo"),
        "python_executable": sys.executable,
        "remote_project_root": "/remote/repo",
        "remote_uv": "/remote/bin/uv",
        "project_commit": "0" * 40,
        "max_parallel_train": 8,
        "max_parallel_eval": 2,
        "retry_policy": {
            "max_transient_attempts": 3,
            "initial_backoff_seconds": 0,
            "maximum_backoff_seconds": 0,
        },
        "environment": {
            "ARTIFACT_ROOT": "/remote/artifacts",
            "CONTRIBUTION_SAE_CACHE_MAX_BYTES": "171798691840",
        },
        "resources": resources,
        "trajectories": trajectories,
        "evaluations": evaluations,
    }


def _self_hashed(body: dict) -> dict:
    return {**body, "result_sha256": canonical_sha256(body)}


def _write_training_completion(plan, task_id: str) -> None:
    task = next(item for item in plan.trajectories if item.task_id == task_id)
    body = {
        "schema_version": "gemma4-contribution-sae-training-result-v1",
        "status": "completed",
        "tokens": 10_000_000,
        "run_identity": {
            "plan": {
                "phase_name": "learning_horizon_10m",
                "condition": task.condition,
                "seed": 0,
                "total_tokens": 10_000_000,
            }
        },
    }
    task.completion_path_controller.parent.mkdir(parents=True, exist_ok=True)
    task.completion_path_controller.write_text(
        json.dumps(_self_hashed(body), sort_keys=True),
        encoding="ascii",
    )


def _write_evaluation_completion(plan, task_id: str) -> None:
    task = next(item for item in plan.evaluations if item.task_id == task_id)
    body = {
        "schema_version": (
            "gemma4-contribution-learning-horizon-curve-point-v1"
            if task.scientific_model == "sae"
            else "gemma4-contribution-transcoder-learning-horizon-result-v1"
        ),
        "protocol_id": "gemma4-pilot-learning-horizon-v1-20260728",
        "status": "curve_point_measured",
        "seed": 0,
        **(
            {"checkpoint_target_tokens": task.milestone_tokens}
            if task.scientific_model == "sae"
            else {"learning_curve_checkpoint": {"target_tokens": task.milestone_tokens}}
        ),
    }
    task.completion_path_controller.parent.mkdir(parents=True, exist_ok=True)
    task.completion_path_controller.write_text(
        json.dumps(_self_hashed(body), sort_keys=True),
        encoding="ascii",
    )


def test_plan_is_exactly_eight_unique_trajectories_and_six_joint_evaluations(
    tmp_path: Path,
) -> None:
    plan = parse_direct_campaign_plan(_raw_plan(tmp_path))

    assert len(plan.resources) == 8
    assert len(plan.trajectories) == 8
    assert len(plan.evaluations) == 6
    assert {(item.scientific_model, item.condition) for item in plan.trajectories} == {
        (model, condition) for model in MODELS for condition in CONDITIONS
    }
    assert {
        (item.scientific_model, item.milestone_tokens) for item in plan.evaluations
    } == {(model, milestone) for model in MODELS for milestone in MILESTONES}


def test_train_commands_are_continuous_and_evaluators_are_joint(
    tmp_path: Path,
) -> None:
    plan = parse_direct_campaign_plan(_raw_plan(tmp_path))
    train = build_train_command(plan, plan.trajectories[0], "attempt-1")
    assert train[train.index("--maximum-runtime-seconds") + 1] == str(
        CONTINUOUS_RUNTIME_SECONDS
    )
    assert train[train.index("--action") + 1] == "train"

    evaluation = build_evaluation_command(plan, plan.evaluations[0])
    for condition in CONDITIONS:
        assert f"--run-root-{condition.lower()}" in evaluation
    assert "--condition" not in evaluation


def test_plan_rejects_duplicate_resource_ownership(tmp_path: Path) -> None:
    raw = _raw_plan(tmp_path)
    raw["trajectories"][1]["resource_id"] = raw["trajectories"][0]["resource_id"]

    with pytest.raises(
        DirectCampaignPlanError, match="exactly one training trajectory"
    ):
        parse_direct_campaign_plan(raw)


def test_plan_rejects_24_style_or_missing_joint_evaluations(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path)
    raw["evaluations"].append(copy.deepcopy(raw["evaluations"][0]))

    with pytest.raises(
        DirectCampaignPlanError, match="exactly six physical joint evaluations"
    ):
        parse_direct_campaign_plan(raw)


def test_plan_rejects_controller_owned_environment_override(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path)
    raw["environment"]["TPU_NAME"] = "wrong-owner"

    with pytest.raises(DirectCampaignPlanError, match="controller-owned variable"):
        parse_direct_campaign_plan(raw)


def test_campaign_runs_eight_trains_then_exactly_six_evaluations(
    tmp_path: Path,
) -> None:
    plan = parse_direct_campaign_plan(_raw_plan(tmp_path))
    calls: list[str] = []
    lock = threading.Lock()

    def runner(task_id, _command, _environment, _cwd, stdout, _stop):
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text("ok\n", encoding="ascii")
        with lock:
            calls.append(task_id)
        if task_id.startswith("train-"):
            _write_training_completion(plan, task_id)
        else:
            _write_evaluation_completion(plan, task_id)
        return AttemptExecution(0)

    result = DirectCampaign(
        plan,
        attempt_runner=runner,
        verify_checkout=False,
    ).run()

    assert result["status"] == "completed"
    assert len([item for item in calls if item.startswith("train-")]) == 8
    assert len([item for item in calls if item.startswith("eval-")]) == 6
    first_eval = min(
        index for index, task_id in enumerate(calls) if task_id.startswith("eval-")
    )
    assert all(task_id.startswith("train-") for task_id in calls[:first_eval])
    assert len(result["training"]) == 8
    assert len(result["evaluations"]) == 6


def test_stop_after_training_is_a_durable_reconciliation_barrier(
    tmp_path: Path,
) -> None:
    plan = parse_direct_campaign_plan(_raw_plan(tmp_path))
    first_calls: list[str] = []

    def train_only(task_id, _command, _environment, _cwd, stdout, _stop):
        assert task_id.startswith("train-")
        first_calls.append(task_id)
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text("ok\n", encoding="ascii")
        _write_training_completion(plan, task_id)
        return AttemptExecution(0)

    barrier = DirectCampaign(
        plan,
        attempt_runner=train_only,
        verify_checkout=False,
    ).run(stop_after_training=True)

    assert barrier["status"] == "training_completed"
    assert len(first_calls) == 8
    assert not (plan.state_root / "final.json").exists()

    second_calls: list[str] = []

    def eval_only(task_id, _command, _environment, _cwd, stdout, _stop):
        assert task_id.startswith("eval-")
        second_calls.append(task_id)
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text("ok\n", encoding="ascii")
        _write_evaluation_completion(plan, task_id)
        return AttemptExecution(0)

    final = DirectCampaign(
        plan,
        attempt_runner=eval_only,
        verify_checkout=False,
    ).run()

    assert final["status"] == "completed"
    assert len(second_calls) == 6


def test_v4_resource_may_omit_regional_mirror_evidence_and_environment(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path)
    resource = raw["resources"][0]
    resource["accelerator_type"] = "v4-32"
    resource["internal_ip"] = False
    resource["evidence"]["mirror_seal"] = None
    plan = parse_direct_campaign_plan(raw)
    observed: dict[str, str] = {}

    def runner(task_id, _command, environment, _cwd, stdout, _stop):
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text("ok\n", encoding="ascii")
        if task_id == "train-sae-a":
            observed.update(environment)
        if task_id.startswith("train-"):
            _write_training_completion(plan, task_id)
        else:
            _write_evaluation_completion(plan, task_id)
        return AttemptExecution(0)

    DirectCampaign(
        plan,
        attempt_runner=runner,
        verify_checkout=False,
    ).run()

    assert "CONTRIBUTION_REGIONAL_MIRROR_SEAL" not in observed
    assert observed["TPU_ACCELERATOR_TYPE"] == "v4-32"
    assert observed["TPU_USE_INTERNAL_IP"] == "0"


def test_regional_resource_cannot_omit_mirror_evidence(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path)
    raw["resources"][0]["evidence"]["mirror_seal"] = None

    with pytest.raises(DirectCampaignPlanError, match="requires mirror-seal evidence"):
        parse_direct_campaign_plan(raw)


def test_transient_ssh_failure_retries_with_new_attempt_without_duplication(
    tmp_path: Path,
) -> None:
    plan = parse_direct_campaign_plan(_raw_plan(tmp_path))
    counts: dict[str, int] = {}
    lock = threading.Lock()

    def runner(task_id, _command, _environment, _cwd, stdout, _stop):
        with lock:
            counts[task_id] = counts.get(task_id, 0) + 1
            number = counts[task_id]
        stdout.parent.mkdir(parents=True, exist_ok=True)
        if task_id == "train-sae-a" and number == 1:
            stdout.write_text(
                "ssh: Connection reset by peer\n",
                encoding="ascii",
            )
            return AttemptExecution(255)
        stdout.write_text("ok\n", encoding="ascii")
        if task_id.startswith("train-"):
            _write_training_completion(plan, task_id)
        else:
            _write_evaluation_completion(plan, task_id)
        return AttemptExecution(0)

    DirectCampaign(
        plan,
        attempt_runner=runner,
        verify_checkout=False,
    ).run()

    assert counts["train-sae-a"] == 2
    attempts = plan.state_root / "training" / "train-sae-a" / "attempts"
    assert sorted(item.name for item in attempts.iterdir()) == [
        "attempt-000001",
        "attempt-000002",
    ]
    first = json.loads(
        (attempts / "attempt-000001" / "result.json").read_text(encoding="ascii")
    )
    assert first["classification"] == "ssh_transport"


def test_scientific_failure_is_terminal_and_is_not_retried(
    tmp_path: Path,
) -> None:
    plan = parse_direct_campaign_plan(_raw_plan(tmp_path))
    counts: dict[str, int] = {}
    lock = threading.Lock()

    def runner(task_id, _command, _environment, _cwd, stdout, _stop):
        with lock:
            counts[task_id] = counts.get(task_id, 0) + 1
        stdout.parent.mkdir(parents=True, exist_ok=True)
        if task_id == "train-sae-a":
            stdout.write_text("non-finite loss\n", encoding="ascii")
            return AttemptExecution(1)
        stdout.write_text("ok\n", encoding="ascii")
        _write_training_completion(plan, task_id)
        return AttemptExecution(0)

    with pytest.raises(DirectCampaignStateError, match="terminal failures"):
        DirectCampaign(
            plan,
            attempt_runner=runner,
            verify_checkout=False,
        ).run()

    assert counts["train-sae-a"] == 1
    failure = json.loads(
        (plan.state_root / "training" / "train-sae-a" / "failure.json").read_text(
            encoding="ascii"
        )
    )
    assert failure["classification"] == "numerical_failure"


def test_existing_completions_are_recognized_without_operator_calls(
    tmp_path: Path,
) -> None:
    plan = parse_direct_campaign_plan(_raw_plan(tmp_path))
    for task in plan.trajectories:
        _write_training_completion(plan, task.task_id)
    for task in plan.evaluations:
        _write_evaluation_completion(plan, task.task_id)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("completed tasks must not launch an operator")

    result = DirectCampaign(
        plan,
        attempt_runner=forbidden,
        verify_checkout=False,
    ).run()
    status = direct_campaign_status(plan)

    assert result["status"] == "completed"
    assert {item["state"] for item in status["tasks"].values()} == {"completed"}


def test_invalid_training_completion_fails_closed(tmp_path: Path) -> None:
    plan = parse_direct_campaign_plan(_raw_plan(tmp_path))
    task = plan.trajectories[0]
    _write_training_completion(plan, task.task_id)
    payload = json.loads(task.completion_path_controller.read_text(encoding="ascii"))
    payload["tokens"] = 9_999_999
    unsigned = dict(payload)
    unsigned.pop("result_sha256")
    payload["result_sha256"] = canonical_sha256(unsigned)
    task.completion_path_controller.write_text(
        json.dumps(payload),
        encoding="ascii",
    )

    with pytest.raises(
        DirectCampaignStateError, match="training completion role changed"
    ):
        direct_campaign_status(plan)
