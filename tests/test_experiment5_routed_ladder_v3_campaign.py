from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

import scripts.tpu.build_experiment5_routed_ladder_v3_training_plan as build_cli
import scripts.tpu.run_experiment5_routed_ladder_v3_training_campaign as run_cli
from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_routed_ladder_resources import (
    COHORT_KEYS,
    seal_routed_ladder_resource_pool,
)
from moe_sae.experiment5_routed_ladder_v3_campaign import (
    EXTENSION_NAMESPACE,
    TRAIN_CONDITION,
    TRAINING_ROLES,
    TRAINING_TASK_COUNT,
    build_routed_ladder_v3_training_plan,
    build_v3_train_command,
    load_routed_ladder_v3_training_plan,
    parse_routed_ladder_v3_training_plan,
)
from moe_sae.experiment5_routed_ladder_v3_config import (
    PHASE_NAME,
    PROTOCOL_ID,
)

from test_experiment5_routed_ladder_campaign import (
    COMMIT,
    _evidence,
    _raw_plan as _raw_v2_plan,
)


def _source_and_six_resource_pool(
    tmp_path: Path,
) -> tuple[dict, dict]:
    """Reuse the sealed v2 fixtures, then admit six distinct v3 owners."""

    source = _raw_v2_plan(
        tmp_path / "source-fixture",
        extra_v5_resources=2,
    )
    source["environment"].update(
        {
            "CONTRIBUTION_SOURCE_OVERLAY_ROOT": (
                "/captures/overlay/train"
            ),
            "CONTRIBUTION_SOURCE_OVERLAY_PREFIX": (
                "gs://source/overlay/train"
            ),
        }
    )

    pool = copy.deepcopy(source["resource_pool"])
    template = copy.deepcopy(pool["resources"][-1])
    old_id = template["id"]
    new_id = "slice-hot-v5-2"
    template["id"] = new_id
    template["tpu_name"] = "gemma4-hot-v5-2"
    plane = copy.deepcopy(pool["planes"][old_id])
    controller_root = Path(plane["controller_artifact_root"])
    remote_root = plane["remote_artifact_root"]
    template["evidence"] = {
        name: _evidence(
            controller_root / "hot" / new_id / f"{name}.json",
            f"{remote_root}/hot/{new_id}/{name}.json",
        )
        for name in (
            "controller_mapping",
            "cache_inventory",
            "mirror_seal",
        )
    }
    pool["resources"].append(template)
    pool["planes"][new_id] = plane

    resource_ids = [row["id"] for row in pool["resources"]]
    pool["cohorts"] = {
        key: [resource_id]
        for key, resource_id in zip(
            COHORT_KEYS,
            resource_ids,
            strict=True,
        )
    }
    pool.pop("pool_sha256")
    return source, seal_routed_ladder_resource_pool(pool)


def _raw_v3_plan(tmp_path: Path) -> dict:
    source, resource_pool = _source_and_six_resource_pool(tmp_path)
    value = build_routed_ladder_v3_training_plan(
        source,
        project_commit=COMMIT,
        resource_pool=resource_pool,
        already_spent_usd=7.25,
        campaign_id="focused-v3-campaign",
    )
    value["state_root"] = str(
        tmp_path
        / "experiment5"
        / "extensions"
        / "routed-activation-ladder-v3"
        / "controller"
        / COMMIT[:12]
    )
    return value


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )


def _write_training_completion(
    plan,
    task,
    *,
    condition: str | None = None,
) -> None:
    body = {
        "schema_version": "gemma4-contribution-sae-training-result-v1",
        "status": "completed",
        "tokens": 10_000_000,
        "source_revision": plan.project_commit,
        "run_identity": {
            "plan": {
                "phase_name": PHASE_NAME,
                "condition": condition or task.condition,
                "seed": task.seed,
                "total_tokens": 10_000_000,
            }
        },
    }
    value = {**body, "result_sha256": canonical_sha256(body)}
    task.completion_path_controller.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    task.completion_path_controller.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="ascii",
    )


def test_build_cli_writes_exact_create_only_v3_training_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, resource_pool = _source_and_six_resource_pool(tmp_path)
    source_path = tmp_path / "r8-source.yaml"
    pool_path = tmp_path / "resource-pool.yaml"
    output_path = tmp_path / "v3-training-plan.yaml"
    _write_yaml(source_path, source)
    _write_yaml(pool_path, resource_pool)

    result = build_cli.main(
        [
            "--r8-plan",
            str(source_path),
            "--project-commit",
            COMMIT,
            "--resource-pool",
            str(pool_path),
            "--already-spent-usd",
            "7.25",
            "--campaign-id",
            "focused-v3-campaign",
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    assert "resources=6 trajectories=6" in capsys.readouterr().out
    plan = load_routed_ladder_v3_training_plan(output_path)
    assert plan.campaign_id == "focused-v3-campaign"
    assert plan.cost_policy.already_spent_usd == 7.25
    assert plan.evaluations == ()
    assert len(plan.resources) == TRAINING_TASK_COUNT
    assert len(plan.trajectories) == TRAINING_TASK_COUNT
    assert {
        (task.panel, task.condition, task.seed)
        for task in plan.trajectories
    } == TRAINING_ROLES
    assert {
        task.eligible_resource_ids[0]
        for task in plan.trajectories
    } == {resource.resource_id for resource in plan.resources}
    assert all(
        f"/{EXTENSION_NAMESPACE}/{COMMIT[:12]}/"
        in task.control_root_remote
        for task in plan.trajectories
    )
    assert all(
        task.source_overlay_control_root_remote is None
        for task in plan.trajectories
        if task.panel == "sae"
    )
    assert all(
        task.source_overlay_control_root_remote
        == "/captures/overlay/train"
        for task in plan.trajectories
        if task.panel == "transcoder"
    )

    task = next(
        item
        for item in plan.trajectories
        if item.panel == "transcoder" and item.seed == 2
    )
    command = build_v3_train_command(plan, task, "v3-attempt")
    assert command[command.index("--condition") + 1] == TRAIN_CONDITION
    assert command[command.index("--attempt-id") + 1] == "v3-attempt"
    assert (
        command[command.index("--maximum-runtime-seconds") + 1]
        == "43200"
    )
    assert "--controller-config" in command

    assert (
        build_cli.main(
            [
                "--r8-plan",
                str(source_path),
                "--project-commit",
                COMMIT,
                "--resource-pool",
                str(pool_path),
                "--already-spent-usd",
                "7.25",
                "--output",
                str(output_path),
            ]
        )
        == 2
    )
    assert "refusing to overwrite" in capsys.readouterr().err


def test_validate_cli_reports_identity_and_rejects_throttled_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    value = _raw_v3_plan(tmp_path)
    plan_path = tmp_path / "plan.yaml"
    _write_yaml(plan_path, value)

    assert run_cli.main(["validate", "--plan", str(plan_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "campaign_id": "focused-v3-campaign",
        "cost_hard_limit_usd": 50.0,
        "plan_sha256": parse_routed_ladder_v3_training_plan(
            value
        ).plan_sha256,
        "resources": 6,
        "trajectories": 6,
    }

    changed = copy.deepcopy(value)
    changed["max_parallel_train"] = 5
    changed_path = tmp_path / "throttled.yaml"
    _write_yaml(changed_path, changed)
    assert run_cli.main(["validate", "--plan", str(changed_path)]) == 2
    assert "one distinct resource owner" in capsys.readouterr().err


def test_status_cli_classifies_completion_retry_and_inflight_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    value = _raw_v3_plan(tmp_path)
    plan = parse_routed_ladder_v3_training_plan(value)
    plan_path = tmp_path / "plan.yaml"
    _write_yaml(plan_path, value)
    completed, retryable, inflight, *_ = plan.trajectories
    _write_training_completion(plan, completed)

    retry_root = (
        plan.state_root
        / "training"
        / retryable.task_id
        / "attempts"
        / "attempt-000001"
    )
    retry_root.mkdir(parents=True)
    (retry_root / "result.json").write_text("{}\n", encoding="ascii")
    inflight_root = (
        plan.state_root
        / "training"
        / inflight.task_id
        / "attempts"
        / "attempt-000001"
    )
    inflight_root.mkdir(parents=True)

    assert run_cli.main(["status", "--plan", str(plan_path)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["schema_version"] == value["schema_version"]
    assert status["protocol_id"] == PROTOCOL_ID
    assert status["training_barrier"] is None
    assert status["tasks"][completed.task_id]["state"] == "completed"
    assert status["tasks"][retryable.task_id] == {
        "attempts": 1,
        "completion": None,
        "eligible_resources": list(retryable.eligible_resource_ids),
        "state": "retryable",
    }
    assert status["tasks"][inflight.task_id]["state"] == (
        "ambiguous_inflight"
    )
    assert {
        row["state"] for row in status["tasks"].values()
    } == {
        "completed",
        "retryable",
        "ambiguous_inflight",
        "pending",
    }

    _write_training_completion(
        plan,
        completed,
        condition="A512Softplus",
    )
    assert run_cli.main(["status", "--plan", str(plan_path)]) == 2
    assert "training completion role changed" in capsys.readouterr().err


def test_run_cli_installs_stop_handlers_and_dispatches_training_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    value = _raw_v3_plan(tmp_path)
    plan_path = tmp_path / "plan.yaml"
    _write_yaml(plan_path, value)
    expected = parse_routed_ladder_v3_training_plan(value)
    observed: dict[str, object] = {}

    class FakeCampaign:
        def __init__(self, plan, *, stop_event):
            observed["plan"] = plan
            observed["stop_event"] = stop_event

        def run(self):
            return {
                "status": "training_completed",
                "training": {"count": TRAINING_TASK_COUNT},
            }

    handlers: list[tuple[int, object]] = []
    monkeypatch.setattr(
        run_cli,
        "RoutedLadderV3TrainingCampaign",
        FakeCampaign,
    )
    monkeypatch.setattr(
        run_cli.signal,
        "signal",
        lambda selected, handler: handlers.append((selected, handler)),
    )

    assert run_cli.main(["run", "--plan", str(plan_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "status": "training_completed",
        "training": {"count": TRAINING_TASK_COUNT},
    }
    assert observed["plan"].plan_sha256 == expected.plan_sha256
    assert not observed["stop_event"].is_set()
    assert {selected for selected, _handler in handlers} == {
        run_cli.signal.SIGINT,
        run_cli.signal.SIGTERM,
    }
