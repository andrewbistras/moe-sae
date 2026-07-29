from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import yaml

import moe_sae.experiment5_evaluation_recovery as recovery_module
from moe_sae.checkpointing import (
    checkpoint_payload_uri,
    make_pointer,
    milestone_pointer_path,
)
from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.direct_horizon_campaign import AttemptExecution
from moe_sae.experiment5_campaign import (
    MILESTONES,
    AttemptExecution as CampaignAttemptExecution,
    ResourceReadinessObservation,
    ResourceReadinessStatus,
    _record,
    _write_create_only,
    inspect_training_completion,
    parse_experiment5_campaign_plan,
)
from moe_sae.experiment5_evaluation_recovery import (
    CHECKPOINT_POINTER_COUNT,
    FROZEN_TRAINING_REVISION,
    HANDOFF_SCHEMA_VERSION,
    EvaluationRecoveryCampaign,
    EvaluationRecoveryPlanError,
    EvaluationRecoveryStateError,
    _verify_parent,
    build_evaluation_recovery_plan,
    parse_evaluation_recovery_plan,
    seal_recovery_handoff,
    validate_recovery_handoff,
)
from scripts.tpu.build_experiment5_evaluation_recovery import _region_roots

# Reusing the frozen campaign's own test-plan builder keeps this suite focused
# on the recovery boundary instead of maintaining a second 24+18 core fixture.
from test_experiment5_campaign import (
    _install_report_stubs,
    _raw_plan,
    _write_evaluation_completion,
    _write_training_completion,
)


EVALUATION_REVISION = "9" * 40


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )


def _core_with_training_barrier(
    tmp_path: Path,
    *,
    two_regions: bool = False,
) -> tuple[Path, dict]:
    raw = _raw_plan(tmp_path / "core")
    raw["project_commit"] = FROZEN_TRAINING_REVISION
    for resource in raw["resources"]:
        resource["deployed_commit"] = FROZEN_TRAINING_REVISION
    if two_regions:
        us_ids = [row["id"] for row in raw["resources"][:3]]
        us2_ids = [row["id"] for row in raw["resources"][3:]]
        for resource in raw["resources"][3:]:
            resource["region"] = "us-test2"
            resource["zone"] = "us-test2-a"
        raw["state_root"] = str(tmp_path / "mount-control" / "core-state")
        for task in raw["trajectories"]:
            task["eligible_resource_ids"] = (
                us_ids if task["seed"] == 0 else us2_ids
            )
            mount = (
                tmp_path / "mount-usc"
                if task["seed"] == 0
                else tmp_path / "mount-control"
            )
            task["completion_path_controller"] = str(
                mount
                / "core-results"
                / "runs"
                / task["run_id"]
                / "training-result.json"
            )
        for task in raw["evaluations"]:
            task["eligible_resource_ids"] = (
                us_ids if task["seed"] == 0 else us2_ids
            )
            mount = (
                tmp_path / "mount-usc"
                if task["seed"] == 0
                else tmp_path / "mount-control"
            )
            destination = (
                mount
                / "core-results"
                / "evaluation"
                / task["panel"]
                / f"seed{task['seed']}"
                / str(task["milestone_tokens"])
            )
            task["completion_path_controller"] = str(
                destination / "scientific-result.json"
            )
            task["paired_scores_path_controller"] = str(
                destination / "paired-scores.json"
            )
    core_path = tmp_path / "contracts" / "core.yaml"
    core_path.parent.mkdir(parents=True)
    core_path.write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )
    core = parse_experiment5_campaign_plan(raw)
    completions = {}
    for task in core.trajectories:
        _write_training_completion(core, task.task_id)
        training = json.loads(
            task.completion_path_controller.read_text(encoding="ascii")
        )
        training_body = dict(training)
        training_body.pop("result_sha256")
        training_body["run_identity"]["plan"].update(
            {
                "global_microbatch_tokens": 4_096,
                "microbatches_per_update": 4,
                "planned_updates": math.ceil(10_000_000 / 16_384),
            }
        )
        training = {
            **training_body,
            "result_sha256": canonical_sha256(training_body),
        }
        _write_json(task.completion_path_controller, training)
        identity = training["run_identity"]
        checkpoint_root = task.completion_path_controller.parent / "checkpoints"
        for index, milestone in enumerate(MILESTONES, start=1):
            attempt_id = f"{index:032x}"
            step = math.ceil(milestone / 16_384)
            tokens_seen = min(step * 16_384, 10_000_000)
            pointer = make_pointer(
                kind="latest",
                payload_uri=checkpoint_payload_uri(
                    task.checkpoint_root_gcs,
                    identity,
                    step=step,
                    tokens_seen=tokens_seen,
                    attempt_id=attempt_id,
                ),
                step=step,
                tokens_seen=tokens_seen,
                attempt_id=attempt_id,
                run_identity=identity,
            )
            _write_json(
                milestone_pointer_path(checkpoint_root, milestone),
                pointer,
            )
        completion = inspect_training_completion(core, task)
        assert completion is not None
        completions[task.task_id] = completion.to_dict()
    barrier = _record(
        "training_phase_completion",
        {
            "campaign_id": core.campaign_id,
            "plan_sha256": core.plan_sha256,
            "status": "training_completed",
            "training": {
                key: completions[key] for key in sorted(completions)
            },
            "completed_at": "2026-07-29T12:00:00Z",
        },
    )
    _write_create_only(core.state_root / "training-complete.json", barrier)
    return core_path, raw


def _recovery_resources(
    raw_core: dict,
    count: int | None = 3,
) -> list[dict]:
    selected = (
        raw_core["resources"]
        if count is None
        else raw_core["resources"][:count]
    )
    resources = copy.deepcopy(selected)
    # A recovered cell must retain the accelerator generation and regional
    # artifact plane recorded by its frozen training receipts.
    for resource in resources:
        resource["cache_profile"] = "paired"
        resource["deployed_commit"] = EVALUATION_REVISION
    return resources


@pytest.fixture
def recovery_raw(tmp_path: Path) -> dict:
    core_path, raw_core = _core_with_training_barrier(tmp_path)
    return build_evaluation_recovery_plan(
        core_plan_path=core_path,
        evaluation_source_revision=EVALUATION_REVISION,
        resources=_recovery_resources(raw_core),
        state_root=tmp_path / "recovery" / "state",
        result_roots_controller={
            "us-test1": tmp_path / "recovery" / "results"
        },
        result_root_remote="/remote/recovery/evaluation",
        already_spent_usd=31.64,
    )


def _two_region_recovery_raw(tmp_path: Path) -> dict:
    core_path, raw_core = _core_with_training_barrier(
        tmp_path,
        two_regions=True,
    )
    return build_evaluation_recovery_plan(
        core_plan_path=core_path,
        evaluation_source_revision=EVALUATION_REVISION,
        resources=_recovery_resources(raw_core, count=None),
        state_root=tmp_path / "recovery" / "state",
        result_roots_controller={
            "us-test1": tmp_path / "recovery-usc" / "results",
            "us-test2": tmp_path / "recovery-control" / "results",
        },
        result_root_remote="/remote/recovery/evaluation",
        already_spent_usd=31.64,
    )


def _ready(
    *,
    status: ResourceReadinessStatus = ResourceReadinessStatus.READY,
    spot: bool = True,
    absent: bool = False,
) -> ResourceReadinessObservation:
    return ResourceReadinessObservation(
        status=status,
        state=("READY" if status is ResourceReadinessStatus.READY else "PREEMPTED"),
        health=("HEALTHY" if status is ResourceReadinessStatus.READY else None),
        create_time="2026-07-29T10:11:12.123Z",
        spot=spot,
        absent=absent,
        detail="test readiness",
    )


def test_builder_is_evaluation_only_and_hash_pins_all_training(
    recovery_raw: dict,
) -> None:
    plan = parse_evaluation_recovery_plan(recovery_raw)

    assert plan.trajectories == ()
    assert len(plan.evaluations) == 18
    assert len(plan.parent.training_cells) == 24
    assert (
        sum(
            len(row.checkpoint_pointers)
            for row in plan.parent.training_cells
        )
        == CHECKPOINT_POINTER_COUNT
        == 72
    )
    assert plan.project_commit == EVALUATION_REVISION
    assert plan.parent.training_source_revision == FROZEN_TRAINING_REVISION
    assert not plan.state_root.is_relative_to(plan.core_plan.state_root)
    result_roots = plan.result_roots_controller_dict()
    assert set(result_roots) == {"us-test1"}
    assert not result_roots["us-test1"].is_relative_to(plan.core_plan.state_root)
    assert all(
        task.completion_path_controller.is_relative_to(
            result_roots["us-test1"]
        )
        for task in plan.evaluations
    )


def test_two_region_builder_and_parser_pin_each_controller_plane(
    tmp_path: Path,
) -> None:
    raw = _two_region_recovery_raw(tmp_path)
    plan = parse_evaluation_recovery_plan(raw)
    roots = plan.result_roots_controller_dict()
    resource_by_id = {
        resource.resource_id: resource for resource in plan.resources
    }
    counts = {"us-test1": 0, "us-test2": 0}
    for task in plan.evaluations:
        regions = {
            resource_by_id[resource_id].region
            for resource_id in task.eligible_resource_ids
        }
        assert len(regions) == 1
        region = next(iter(regions))
        counts[region] += 1
        assert task.completion_path_controller.is_relative_to(roots[region])
        assert task.paired_scores_path_controller.is_relative_to(roots[region])
        assert task.result_root_remote.startswith(
            "/remote/recovery/evaluation/"
        )
    assert counts == {"us-test1": 6, "us-test2": 12}

    missing_plane = copy.deepcopy(raw)
    del missing_plane["result_roots_controller"]["us-test2"]
    with pytest.raises(
        EvaluationRecoveryPlanError,
        match="do not exactly cover frozen regions",
    ):
        parse_evaluation_recovery_plan(missing_plane)


def test_cli_parses_repeated_regional_controller_roots() -> None:
    assert _region_roots(
        [
            "us-central1=/mnt/moe-sae-usc1/recovery/results",
            "us-central2=/mnt/moe-sae-control/recovery/results",
        ]
    ) == {
        "us-central1": Path("/mnt/moe-sae-usc1/recovery/results"),
        "us-central2": Path("/mnt/moe-sae-control/recovery/results"),
    }
    with pytest.raises(
        EvaluationRecoveryPlanError,
        match="unique REGION=/absolute/path",
    ):
        _region_roots(
            [
                "us-central1=/mnt/a",
                "us-central1=/mnt/b",
            ]
        )
    with pytest.raises(
        EvaluationRecoveryPlanError,
        match="path must be absolute",
    ):
        _region_roots(["us-central1=relative/results"])


def test_parent_preflight_reproves_barrier_results_and_pointers(
    recovery_raw: dict,
) -> None:
    plan = parse_evaluation_recovery_plan(recovery_raw)
    barrier = _verify_parent(plan)
    assert barrier["record_sha256"] == plan.parent.training_barrier_record_sha256

    pointer = plan.parent.training_cells[0].checkpoint_pointers[0].artifact
    pointer.controller_path.write_text("{}\n", encoding="ascii")
    with pytest.raises(
        EvaluationRecoveryStateError,
        match="checkpoint pointer",
    ):
        _verify_parent(plan)


def test_parser_rejects_old_or_overlapping_namespaces(
    recovery_raw: dict,
) -> None:
    core = parse_evaluation_recovery_plan(recovery_raw).core_plan
    overlapping_state = copy.deepcopy(recovery_raw)
    overlapping_state["state_root"] = str(core.state_root / "repair")
    overlapping_state["handoff_path"] = str(
        core.state_root / "repair" / "recovery-handoff.json"
    )
    with pytest.raises(
        EvaluationRecoveryPlanError,
        match="overlaps a frozen namespace",
    ):
        parse_evaluation_recovery_plan(overlapping_state)

    overlapping_result = copy.deepcopy(recovery_raw)
    old = core.evaluations[0].completion_path_controller.parent
    overlapping_result["result_roots_controller"] = {"us-test1": str(old)}
    for task in overlapping_result["evaluations"]:
        destination = (
            old
            / task["panel"]
            / f"seed{task['seed']}"
            / str(task["milestone_tokens"])
        )
        task["completion_path_controller"] = str(
            destination / "scientific-result.json"
        )
        task["paired_scores_path_controller"] = str(
            destination / "paired-scores.json"
        )
    with pytest.raises(
        EvaluationRecoveryPlanError,
        match="overlaps a frozen namespace",
    ):
        parse_evaluation_recovery_plan(overlapping_result)


def test_parser_rejects_missing_training_cell(recovery_raw: dict) -> None:
    changed = copy.deepcopy(recovery_raw)
    changed["parent"]["training_cells"].pop()
    with pytest.raises(
        EvaluationRecoveryPlanError,
        match="exactly 24 training cells",
    ):
        parse_evaluation_recovery_plan(changed)


def test_parser_rejects_cross_generation_evaluation_resource(
    recovery_raw: dict,
) -> None:
    changed = copy.deepcopy(recovery_raw)
    changed["resources"][0]["accelerator_type"] = "v4-32"
    with pytest.raises(
        EvaluationRecoveryPlanError,
        match="changes its frozen accelerator or regional artifact plane",
    ):
        parse_evaluation_recovery_plan(changed)


def test_builder_filters_incompatible_resource_from_every_cell(
    tmp_path: Path,
) -> None:
    core_path, raw_core = _core_with_training_barrier(tmp_path)
    resources = _recovery_resources(raw_core)
    incompatible = copy.deepcopy(resources[0])
    incompatible.update(
        {
            "id": "slice-incompatible-v4",
            "tpu_name": "experiment5-incompatible-v4",
            "accelerator_type": "v4-32",
        }
    )
    resources.append(incompatible)
    raw = build_evaluation_recovery_plan(
        core_plan_path=core_path,
        evaluation_source_revision=EVALUATION_REVISION,
        resources=resources,
        state_root=tmp_path / "recovery" / "state",
        result_roots_controller={
            "us-test1": tmp_path / "recovery" / "results"
        },
        result_root_remote="/remote/recovery/evaluation",
        already_spent_usd=31.64,
    )
    plan = parse_evaluation_recovery_plan(raw)

    assert len(plan.resources) == 4
    assert all(
        "slice-incompatible-v4" not in task.eligible_resource_ids
        for task in plan.evaluations
    )


def test_uncorroborated_spot_label_becomes_workload_failure(
    recovery_raw: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = parse_evaluation_recovery_plan(recovery_raw)
    task = plan.evaluations[0]
    resource = plan.resource(task.eligible_resource_ids[0])

    def runner(
        _task_id,
        _command,
        _environment,
        _working_directory,
        log_path,
        _stop_event,
    ):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("worker aborted\n", encoding="utf-8")
        return AttemptExecution(134, detail="worker aborted")

    monkeypatch.setattr(
        recovery_module,
        "classify_failure",
        lambda **_kwargs: "spot_preemption",
    )
    campaign = EvaluationRecoveryCampaign(
        plan,
        verify_checkout=False,
        attempt_runner=runner,
        resource_readiness_probe=lambda _resource: _ready(),
    )
    with pytest.raises(
        EvaluationRecoveryStateError,
        match="workload_failure",
    ):
        campaign._execute_once(
            task,
            kind="evaluation",
            resource=resource,
            pre_attempt_readiness=_ready(),
        )
    result_path = (
        plan.state_root
        / "evaluation"
        / task.task_id
        / "attempts"
        / "attempt-000001"
        / "result.json"
    )
    result = json.loads(result_path.read_text(encoding="ascii"))
    assert result["raw_classification"] == "spot_preemption"
    assert result["classification"] == "workload_failure"
    assert result["confirmed_preemption"] is False
    assert result["post_failure_readiness"]["status"] == "ready"


def test_confirmed_spot_preemption_retires_slice_and_keeps_alternatives(
    recovery_raw: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = copy.deepcopy(recovery_raw)
    changed["state_root"] = str(
        Path(changed["state_root"]).parent / "preempted-state"
    )
    changed["handoff_path"] = str(
        Path(changed["state_root"]) / "recovery-handoff.json"
    )
    plan = parse_evaluation_recovery_plan(changed)
    task = plan.evaluations[0]
    resource = plan.resource(task.eligible_resource_ids[0])

    monkeypatch.setattr(
        recovery_module,
        "classify_failure",
        lambda **_kwargs: "spot_preemption",
    )
    campaign = EvaluationRecoveryCampaign(
        plan,
        verify_checkout=False,
        attempt_runner=lambda *_args: CampaignAttemptExecution(255),
        resource_readiness_probe=lambda _resource: _ready(
            status=ResourceReadinessStatus.TERMINAL_REPLACEMENT_REQUIRED
        ),
    )
    outcome = campaign._execute_once(
        task,
        kind="evaluation",
        resource=resource,
        pre_attempt_readiness=_ready(),
    )
    assert outcome.completion is None
    assert resource.resource_id in campaign._retired_resources
    assert (
        campaign._probe_resource(resource).status
        is ResourceReadinessStatus.FORMING
    )
    result_path = (
        plan.state_root
        / "evaluation"
        / task.task_id
        / "attempts"
        / "attempt-000001"
        / "result.json"
    )
    result = json.loads(result_path.read_text(encoding="ascii"))
    assert result["classification"] == "spot_preemption"
    assert result["confirmed_preemption"] is True
    assert (
        result["post_failure_readiness"]["status"]
        == "terminal_replacement_required"
    )


def test_recovery_handoff_is_self_hashed_and_plan_bound(
    recovery_raw: dict,
) -> None:
    plan = parse_evaluation_recovery_plan(recovery_raw)
    body = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "protocol_id": recovery_module.PROTOCOL_ID,
        "status": "recovery_report_sealed",
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.plan_sha256,
        "evaluation_source_revision": plan.project_commit,
        "training_source_revision": plan.parent.training_source_revision,
        "core_plan_sha256": plan.parent.core_plan_sha256,
        "training_barrier_record_sha256": (
            plan.parent.training_barrier_record_sha256
        ),
        "recovery_final_record_sha256": "1" * 64,
        "evaluation_barrier_record_sha256": "2" * 64,
        "report_sha256": "3" * 64,
        "report_file_sha256": "4" * 64,
        "budget_ledger_sha256": "5" * 64,
        "budget_ledger_file_sha256": "6" * 64,
        "recovery_envelope_sha256": "7" * 64,
        "recovery_envelope_file_sha256": "8" * 64,
        "incremental_cash_spent_usd": 31.64,
    }
    handoff = {**body, "handoff_sha256": canonical_sha256(body)}
    assert validate_recovery_handoff(handoff, plan=plan) == handoff

    changed = dict(handoff)
    changed["report_sha256"] = "0" * 64
    with pytest.raises(
        EvaluationRecoveryStateError, match="handoff is invalid"
    ):
        validate_recovery_handoff(changed, plan=plan)


def test_recovery_runs_only_evaluations_and_seals_handoff(
    recovery_raw: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = parse_evaluation_recovery_plan(recovery_raw)
    _install_report_stubs(monkeypatch)
    monkeypatch.setattr(
        recovery_module,
        "validate_experiment5_report",
        lambda value: dict(value),
    )

    def runner(
        task_id,
        _command,
        _environment,
        _working_directory,
        log_path,
        _stop_event,
    ):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("recovery evaluation complete\n", encoding="utf-8")
        _write_evaluation_completion(plan, task_id)
        return AttemptExecution(0)

    final = EvaluationRecoveryCampaign(
        plan,
        verify_checkout=False,
        attempt_runner=runner,
        resource_readiness_probe=lambda _resource: _ready(),
    ).run()

    assert final["status"] == "completed"
    assert len(final["evaluations"]) == 18
    assert not (plan.state_root / "training-complete.json").exists()
    assert final["evaluation_source_revision"] == EVALUATION_REVISION
    parent_binding = json.loads(
        (plan.state_root / "parent-training-bound.json").read_text(
            encoding="ascii"
        )
    )
    assert (
        final["parent_binding_record_sha256"]
        == parent_binding["record_sha256"]
    )
    handoff = seal_recovery_handoff(plan)
    assert handoff["status"] == "recovery_report_sealed"
    assert handoff["training_source_revision"] == FROZEN_TRAINING_REVISION
    assert handoff["evaluation_source_revision"] == EVALUATION_REVISION
    assert validate_recovery_handoff(handoff, plan=plan) == handoff


def test_two_region_recovery_runs_and_seals_outputs_on_both_planes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = parse_evaluation_recovery_plan(
        _two_region_recovery_raw(tmp_path)
    )
    _install_report_stubs(monkeypatch)
    monkeypatch.setattr(
        recovery_module,
        "validate_experiment5_report",
        lambda value: dict(value),
    )

    def runner(
        task_id,
        _command,
        _environment,
        _working_directory,
        log_path,
        _stop_event,
    ):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("two-region evaluation complete\n", encoding="utf-8")
        _write_evaluation_completion(plan, task_id)
        return AttemptExecution(0)

    final = EvaluationRecoveryCampaign(
        plan,
        verify_checkout=False,
        attempt_runner=runner,
        resource_readiness_probe=lambda _resource: _ready(),
    ).run()

    assert final["status"] == "completed"
    roots = plan.result_roots_controller_dict()
    completed_by_region = {
        region: sum(
            task.completion_path_controller.is_file()
            and task.completion_path_controller.is_relative_to(root)
            for task in plan.evaluations
        )
        for region, root in roots.items()
    }
    assert completed_by_region == {"us-test1": 6, "us-test2": 12}
    handoff = seal_recovery_handoff(plan)
    assert validate_recovery_handoff(handoff, plan=plan) == handoff
