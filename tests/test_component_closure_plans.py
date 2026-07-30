from __future__ import annotations

import copy
from pathlib import Path

import pytest

from moe_sae.component_closure_campaign import (
    ComponentClosureCampaignStateError,
    ComponentClosureTrainingCampaign,
    ComponentClosureCampaignError,
    build_train_command,
    build_training_plan,
    inspect_training_completion,
    parse_training_campaign,
    validate_training_plan,
)
from moe_sae.component_closure_config import (
    CHECKPOINT_TOKENS,
    PANELS,
    PHASE_NAME,
    PROTOCOL_ID,
    SEEDS,
    SOURCE_CONDITION,
    SOURCE_CONFIG_FILE_SHA256,
    SOURCE_CONFIG_SHA256,
    SOURCE_PHASE_NAME,
    SOURCE_PROTOCOL_ID,
)
from moe_sae.component_closure_evaluation import (
    BUNDLE_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION as EVAL_PLAN_SCHEMA_VERSION,
    ComponentClosureEvaluationError,
    build_evaluation_plan,
    validate_evaluation_plan,
)
from moe_sae.component_closure_profile import build_profile_plan
from moe_sae.component_closure_references import (
    SCHEMA_VERSION as A_SCHEMA_VERSION,
    validate_a_reference_manifest,
)
from moe_sae.contribution_artifacts import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64
REVISION = "b" * 40


def _campaign_source(tmp_path: Path) -> dict:
    identities = (
        ("sae-v5-a", "v5litepod-16", "us-central1-a", "us-central1"),
        ("sae-v5-b", "v5litepod-16", "us-central1-a", "us-central1"),
        ("transcoder-v5-seed0", "v5litepod-16", "us-central1-a", "us-central1"),
        ("transcoder-v4-od-seed1", "v4-32", "us-central2-b", "us-central2"),
        ("transcoder-v4-spot-seed2", "v4-32", "us-central2-b", "us-central2"),
    )
    resources = []
    planes = {}
    for resource_id, accelerator, zone, region in identities:
        sae = resource_id.startswith("sae-")
        controller_root = (
            tmp_path / "planes" / "sae-shared"
            if sae
            else tmp_path / "planes" / resource_id
        )
        remote_root = (
            "/remote/planes/sae-shared" if sae else f"/remote/planes/{resource_id}"
        )
        runtime_root = f"/remote/resources/{resource_id}"
        v5 = accelerator == "v5litepod-16"
        environment = {
            "CONTRIBUTION_QUALIFICATION_BUNDLE": (
                f"{runtime_root}/evidence/qualification.json"
            ),
            "CONTRIBUTION_TRAIN_CAPTURE_ROOT": (f"{runtime_root}/capture/train-target"),
            "CONTRIBUTION_OBJECT_PREFIX": (f"gs://bucket/{resource_id}/train-target"),
            "CONTRIBUTION_RECOMPOSITION_SIDECAR": (
                f"{runtime_root}/evidence/recomposition.json"
            ),
            "CONTRIBUTION_SAE_CACHE_ROOT": ("/dev/shm/moe-sae-local/horizon-10m/cache"),
            "CONTRIBUTION_SAE_CACHE_MAX_BYTES": "171798691840",
        }
        if v5:
            environment["MOE_SAE_V5LITEPOD_DEVICE_KIND"] = "TPU v5"
        mirror = (
            {
                "controller_path": str(
                    controller_root
                    / "evidence"
                    / ("mirror.json" if sae else f"{resource_id}-mirror.json")
                ),
                "remote_path": (
                    f"{remote_root}/evidence/mirror.json"
                    if sae
                    else f"{remote_root}/evidence/{resource_id}-mirror.json"
                ),
                "sha256": SHA,
            }
            if v5
            else None
        )
        resources.append(
            {
                "id": resource_id,
                "project": "project",
                "tpu_name": f"tpu-{resource_id}",
                "zone": zone,
                "region": region,
                "accelerator_type": accelerator,
                "internal_ip": True,
                "ssh_user": None,
                "cache_profile": "paired",
                "deployed_commit": REVISION,
                "evidence": {
                    "controller_mapping": {
                        "controller_path": str(
                            controller_root / "evidence" / f"{resource_id}-mapping.json"
                        ),
                        "remote_path": (
                            f"{remote_root}/evidence/{resource_id}-mapping.json"
                        ),
                        "sha256": SHA,
                    },
                    "cache_inventory": {
                        "controller_path": str(
                            controller_root / "evidence" / f"{resource_id}-cache.json"
                        ),
                        "remote_path": (
                            f"{remote_root}/evidence/{resource_id}-cache.json"
                        ),
                        "sha256": SHA,
                    },
                    "mirror_seal": mirror,
                },
                "environment": environment,
            }
        )
        transcoder = resource_id.startswith("transcoder")
        planes[resource_id] = {
            "controller_artifact_root": str(controller_root),
            "remote_artifact_root": remote_root,
            "bulk_gcs_root": (
                "gs://bucket/sae-shared" if sae else f"gs://bucket/{resource_id}"
            ),
            "source_overlay_control_root_remote": (
                f"{remote_root}/capture/train-source" if transcoder else None
            ),
            "source_overlay_object_prefix_gcs": (
                f"gs://bucket/{resource_id}/train-source" if transcoder else None
            ),
        }
    return {
        "campaign_id": "closure-test",
        "working_directory": str(ROOT),
        "python_executable": "/usr/bin/python3",
        "remote_project_root": "/opt/moe-sae",
        "remote_uv": "/opt/moe-sae/.venv/bin/uv",
        "state_root": str(tmp_path / "state"),
        "resources": resources,
        "planes": planes,
        "cohorts": {
            "sae.seed0": ["sae-v5-a", "sae-v5-b"],
            "sae.seed1": ["sae-v5-a", "sae-v5-b"],
            "sae.seed2": ["sae-v5-a", "sae-v5-b"],
            "transcoder.seed0": ["transcoder-v5-seed0"],
            "transcoder.seed1": ["transcoder-v4-od-seed1"],
            "transcoder.seed2": ["transcoder-v4-spot-seed2"],
        },
    }


def _evaluation_runtime(
    tmp_path: Path,
    *,
    panel: str,
    resource_id: str,
    accelerator_type: str,
) -> dict:
    transcoder = panel == "transcoder"
    runtime_root = f"/remote/resources/{resource_id}"
    plane_root = (
        "/remote/planes/sae-shared"
        if panel == "sae"
        else f"/remote/planes/{resource_id}"
    )
    controller_plane = (
        tmp_path / "planes" / "sae-shared"
        if panel == "sae"
        else tmp_path / "planes" / resource_id
    )
    return {
        "resource_id": resource_id,
        "a_config_remote": f"/configs/a-{panel}.yaml",
        "result_root_remote": f"{plane_root}/evaluation",
        "result_root_controller": str(controller_plane / "evaluation"),
        "accelerator_type": accelerator_type,
        "peak_flops_per_second": (
            3_152_000_000_000_000
            if accelerator_type == "v5litepod-16"
            else 4_400_000_000_000_000
        ),
        "train_target_root": f"{runtime_root}/capture/train-target",
        "eval_target_root": f"{runtime_root}/capture/eval-target",
        "train_target_object_prefix": f"gs://bucket/{resource_id}/train-target",
        "eval_target_object_prefix": f"gs://bucket/{resource_id}/eval-target",
        "train_source_root": (
            f"{plane_root}/capture/train-source" if transcoder else None
        ),
        "eval_source_root": (
            f"{runtime_root}/capture/eval-source" if transcoder else None
        ),
        "train_source_object_prefix": (
            f"gs://bucket/{resource_id}/train-source" if transcoder else None
        ),
        "eval_source_object_prefix": (
            f"gs://bucket/{resource_id}/eval-source" if transcoder else None
        ),
        "cache_root": "/dev/shm/moe-sae-local/horizon-10m/cache",
        "cache_max_bytes": 171_798_691_840,
        "regional_mirror_seal": (
            (
                f"{plane_root}/evidence/mirror.json"
                if panel == "sae"
                else f"{plane_root}/evidence/{resource_id}-mirror.json"
            )
            if accelerator_type == "v5litepod-16"
            else None
        ),
        "recomposition_sidecar": f"{runtime_root}/evidence/recomposition.json",
        "expected_token_identity_sha256": None,
    }


def _a_manifest() -> dict:
    cells = []
    for panel in PANELS:
        for seed in SEEDS:
            for checkpoint in CHECKPOINT_TOKENS:
                cells.append(
                    {
                        "panel": panel,
                        "target_condition": "A",
                        "source_condition": SOURCE_CONDITION,
                        "seed": seed,
                        "checkpoint_tokens": checkpoint,
                        "source_phase_name": SOURCE_PHASE_NAME,
                        "source_config_sha256": SOURCE_CONFIG_SHA256[panel],
                        "source_config_file_sha256": (SOURCE_CONFIG_FILE_SHA256[panel]),
                        "training_source_revision": REVISION,
                        "training_result_controller_path": (
                            f"/controller/{panel}/{seed}/training-result.json"
                        ),
                        "run_root_remote": f"/runs/{panel}/{seed}",
                        "checkpoint_payload_root_gcs": f"gs://bucket/{panel}/{seed}",
                        "parent_training_result_sha256": SHA,
                        "parent_training_result_file_sha256": SHA,
                        "parent_run_contract_sha256": SHA,
                        "parent_run_contract_file_sha256": SHA,
                        "parent_checkpoint_pointer_sha256": SHA,
                        "parent_checkpoint_pointer_file_sha256": SHA,
                        "resource_id": f"{panel}-{seed}",
                        "accelerator_type": "v5litepod-16",
                    }
                )
    body = {
        "schema_version": A_SCHEMA_VERSION,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "target_protocol_id": PROTOCOL_ID,
        "source_plan_sha256": SHA,
        "import_policy": "immutable_read_only_no_copy_no_relabel",
        "cells": cells,
    }
    return validate_a_reference_manifest(
        {**body, "manifest_sha256": canonical_sha256(body)}
    )


def test_training_plan_has_exact_18_fresh_roles(tmp_path: Path) -> None:
    plan = build_training_plan(
        _campaign_source(tmp_path),
        project_commit=REVISION,
        a_reference_manifest=_a_manifest(),
    )
    assert len(plan["tasks"]) == 18
    assert len(plan["resources"]) == 5
    assert {row["condition"] for row in plan["tasks"]} == {"R", "B", "F"}
    parsed = parse_training_campaign(plan)
    assert [row.condition for row in parsed.trajectories[:9]] == [
        "R",
        "R",
        "R",
        "B",
        "B",
        "B",
        "F",
        "F",
        "F",
    ]
    assert all(
        row.eligible_resource_ids == ("sae-v5-a", "sae-v5-b")
        for row in parsed.trajectories
        if row.panel == "sae"
    )
    assert plan["planes"]["sae-v5-a"] == plan["planes"]["sae-v5-b"]
    campaign = ComponentClosureTrainingCampaign(parsed, verify_checkout=False)
    assert [row.condition for row in campaign._prioritized_trajectories()[:6]] == [
        "F"
    ] * 6
    assert all(
        row.task_id.split("-")[2] == "p0"
        for row in parsed.trajectories
        if row.condition == "F"
    )
    changed = copy.deepcopy(plan)
    changed["tasks"][0]["condition"] = "A"
    changed_body = {
        key: value for key, value in changed.items() if key != "plan_sha256"
    }
    changed["plan_sha256"] = canonical_sha256(changed_body)
    with pytest.raises(ComponentClosureCampaignError):
        validate_training_plan(changed)


def _shared_result_base_evaluation_plan(tmp_path: Path) -> dict:
    bundles = []
    shared_remote_root = "/remote/shared/evaluation"
    shared_controller_root = tmp_path / "shared" / "evaluation"
    for panel in PANELS:
        for seed in SEEDS:
            cells = [
                {
                    "logical_condition": condition,
                    "checkpoint_tokens": checkpoint,
                    "result_root": str(
                        Path(shared_remote_root)
                        / panel
                        / f"seed{seed}"
                        / condition.lower()
                        / str(checkpoint)
                    ),
                    "result_root_controller": str(
                        shared_controller_root
                        / panel
                        / f"seed{seed}"
                        / condition.lower()
                        / str(checkpoint)
                    ),
                }
                for condition in ("A", "R", "B", "F")
                for checkpoint in CHECKPOINT_TOKENS
            ]
            runtime = _evaluation_runtime(
                tmp_path,
                panel=panel,
                resource_id=f"{panel}-{seed}",
                accelerator_type="v5litepod-16",
            )
            runtime["result_root_remote"] = shared_remote_root
            runtime["result_root_controller"] = str(shared_controller_root)
            body = {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "resource_id": f"{panel}-{seed}",
                "panel": panel,
                "seed": seed,
                "source_revision": REVISION,
                "runtime": runtime,
                "cells": cells,
            }
            bundles.append({**body, "bundle_sha256": canonical_sha256(body)})
    body = {
        "schema_version": EVAL_PLAN_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "evaluation_source_revision": REVISION,
        "training_plan_sha256": SHA,
        "a_reference_manifest_sha256": SHA,
        "state_root": str(tmp_path),
        "bundles": bundles,
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def test_evaluation_validator_panel_qualifies_shared_result_bases(
    tmp_path: Path,
) -> None:
    plan = _shared_result_base_evaluation_plan(tmp_path)
    checked = validate_evaluation_plan(plan)
    remote_roots = [
        cell["result_root"] for bundle in checked["bundles"] for cell in bundle["cells"]
    ]
    controller_roots = [
        cell["result_root_controller"]
        for bundle in checked["bundles"]
        for cell in bundle["cells"]
    ]
    assert len(checked["bundles"]) == 6
    assert len(set(remote_roots)) == 72
    assert len(set(controller_roots)) == 72
    assert all(
        f"/{bundle['panel']}/seed{bundle['seed']}/" in cell["result_root"]
        for bundle in checked["bundles"]
        for cell in bundle["cells"]
    )


def test_evaluation_validator_rejects_rehashed_duplicate_result_roots(
    tmp_path: Path,
) -> None:
    plan = _shared_result_base_evaluation_plan(tmp_path)
    duplicate = copy.deepcopy(plan)
    duplicate["bundles"][3]["cells"][0]["result_root"] = duplicate["bundles"][0][
        "cells"
    ][0]["result_root"]
    duplicate["bundles"][3]["cells"][0]["result_root_controller"] = duplicate[
        "bundles"
    ][0]["cells"][0]["result_root_controller"]
    duplicate_bundle = duplicate["bundles"][3]
    duplicate_bundle_body = {
        key: value for key, value in duplicate_bundle.items() if key != "bundle_sha256"
    }
    duplicate_bundle["bundle_sha256"] = canonical_sha256(duplicate_bundle_body)
    duplicate_body = {
        key: value for key, value in duplicate.items() if key != "plan_sha256"
    }
    duplicate["plan_sha256"] = canonical_sha256(duplicate_body)
    with pytest.raises(
        ComponentClosureEvaluationError,
        match="result roots collide",
    ):
        validate_evaluation_plan(duplicate)


def test_evaluation_bundles_follow_resource_specific_v5_v4_planes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = build_training_plan(
        _campaign_source(tmp_path),
        project_commit=REVISION,
        a_reference_manifest=_a_manifest(),
    )
    fresh = {}
    for task in training["tasks"]:
        for checkpoint in CHECKPOINT_TOKENS:
            fresh[
                (
                    task["panel"],
                    task["seed"],
                    task["condition"],
                    checkpoint,
                )
            ] = {
                "logical_condition": task["condition"],
                "source_condition": task["condition"],
                "source_protocol_id": PROTOCOL_ID,
                "source_phase_name": PHASE_NAME,
                "source_config": task["config_remote"],
                "training_source_revision": REVISION,
                "run_root": f"/runs/{task['task_id']}",
                "checkpoint_payload_root": task["checkpoint_root_gcs"],
                "parent_training_result_sha256": SHA,
                "parent_training_result_file_sha256": SHA,
                "parent_run_contract_sha256": SHA,
                "parent_run_contract_file_sha256": SHA,
                "parent_checkpoint_pointer_sha256": SHA,
                "parent_checkpoint_pointer_file_sha256": SHA,
                "checkpoint_tokens": checkpoint,
                "reference_kind": "none",
            }
    monkeypatch.setattr(
        "moe_sae.component_closure_evaluation._fresh_cells",
        lambda _training: fresh,
    )
    assignments = {
        "sae.seed0": "sae-v5-a",
        "sae.seed1": "sae-v5-b",
        "sae.seed2": "sae-v5-a",
        "transcoder.seed0": "transcoder-v5-seed0",
        "transcoder.seed1": "transcoder-v4-od-seed1",
        "transcoder.seed2": "transcoder-v4-spot-seed2",
    }
    resources = {}
    accelerator_by_id = {
        row["id"]: row["accelerator_type"] for row in training["resources"]
    }
    for key, resource_id in assignments.items():
        panel = key.split(".", 1)[0]
        resources[resource_id] = _evaluation_runtime(
            tmp_path,
            panel=panel,
            resource_id=resource_id,
            accelerator_type=accelerator_by_id[resource_id],
        )
    plan = build_evaluation_plan(
        training,
        _a_manifest(),
        {"assignments": assignments, "resources": resources},
        evaluation_source_revision=REVISION,
        state_root=tmp_path / "evaluation-state",
    )
    observed = {
        (bundle["panel"], bundle["seed"]): (
            bundle["resource_id"],
            bundle["runtime"]["accelerator_type"],
        )
        for bundle in plan["bundles"]
    }
    assert observed[("transcoder", 0)][1] == "v5litepod-16"
    assert observed[("transcoder", 1)][1] == "v4-32"
    assert observed[("transcoder", 2)][1] == "v4-32"
    assert all(observed[("sae", seed)][1] == "v5litepod-16" for seed in SEEDS)
    remote_roots = [
        cell["result_root"] for bundle in plan["bundles"] for cell in bundle["cells"]
    ]
    controller_roots = [
        cell["result_root_controller"]
        for bundle in plan["bundles"]
        for cell in bundle["cells"]
    ]
    assert len(set(remote_roots)) == 72
    assert len(set(controller_roots)) == 72
    assert all(
        cell["result_root"].startswith(
            f"{bundle['runtime']['result_root_remote']}/{bundle['panel']}/"
        )
        and cell["result_root_controller"].startswith(
            f"{bundle['runtime']['result_root_controller']}/{bundle['panel']}/"
        )
        for bundle in plan["bundles"]
        for cell in bundle["cells"]
    )


def test_training_scheduler_binds_command_environment_and_planes(
    tmp_path: Path,
) -> None:
    value = build_training_plan(
        _campaign_source(tmp_path),
        project_commit=REVISION,
        a_reference_manifest=_a_manifest(),
    )
    plan = parse_training_campaign(value)
    task = next(
        row
        for row in plan.trajectories
        if row.panel == "transcoder" and row.seed == 1 and row.condition == "F"
    )
    resource = plan.resource(task.eligible_resource_ids[0])
    campaign = ComponentClosureTrainingCampaign(
        plan,
        verify_checkout=False,
    )
    _executable, audited = campaign._attempt_environment(
        resource,
        task,
        tmp_path / "operator",
    )
    command = build_train_command(plan, task, "closure-a00000001-test")
    assert task.control_root_remote.startswith(
        plan.plane(resource.resource_id).remote_artifact_root + "/"
    )
    assert task.checkpoint_root_gcs.startswith(
        plan.plane(resource.resource_id).bulk_gcs_root + "/"
    )
    assert audited["TPU_NAME"] == resource.tpu_name
    assert audited["TPU_ACCELERATOR_TYPE"] == "v4-32"
    assert (
        audited["CONTRIBUTION_SOURCE_OVERLAY_ROOT"]
        == plan.plane(resource.resource_id).source_overlay_control_root_remote
    )
    assert audited["CONTRIBUTION_SAE_CONTROL_ROOT"] == task.control_root_remote
    assert command[command.index("--attempt-id") + 1] == "closure-a00000001-test"
    assert command[command.index("--maximum-runtime-seconds") + 1] == "43200"


def test_completion_inspector_uses_adjacent_run_contract(
    tmp_path: Path,
) -> None:
    value = build_training_plan(
        _campaign_source(tmp_path),
        project_commit=REVISION,
        a_reference_manifest=_a_manifest(),
    )
    plan = parse_training_campaign(value)
    task = plan.trajectories[0]
    task.completion_path_controller.parent.mkdir(parents=True)
    identity_plan = {
        "phase_name": PHASE_NAME,
        "condition": task.condition,
        "seed": task.seed,
        "total_tokens": 10_000_000,
    }
    result_body = {
        "schema_version": "gemma4-contribution-sae-training-result-v1",
        "status": "completed",
        "tokens": 10_000_000,
        "run_identity": {"plan": identity_plan},
    }
    result = {
        **result_body,
        "result_sha256": canonical_sha256(result_body),
    }
    task.completion_path_controller.write_text(
        __import__("json").dumps(result),
        encoding="ascii",
    )
    contract_body = {
        "schema_version": "gemma4-contribution-sae-operator-run-v1",
        "run_id": task.run_id,
        "project_commit": REVISION,
        "phase": PHASE_NAME,
        "condition": task.condition,
        "seed": task.seed,
        "checkpoint_payload_root": task.checkpoint_root_gcs,
    }
    contract = {
        **contract_body,
        "run_contract_sha256": canonical_sha256(contract_body),
    }
    contract_path = task.completion_path_controller.parent / "run-contract.json"
    contract_path.write_text(__import__("json").dumps(contract), encoding="ascii")
    assert inspect_training_completion(plan, task) is not None
    changed = dict(contract_body)
    changed["project_commit"] = "c" * 40
    contract_path.write_text(
        __import__("json").dumps(
            {
                **changed,
                "run_contract_sha256": canonical_sha256(changed),
            }
        ),
        encoding="ascii",
    )
    with pytest.raises(ComponentClosureCampaignStateError):
        inspect_training_completion(plan, task)


def test_profile_is_bounded_and_same_resource_per_panel() -> None:
    plan = build_profile_plan(
        {
            "working_directory": str(ROOT),
            "remote_project_root": "/opt/moe-sae",
            "resources": {
                "sae": {
                    "resource_id": "sae-profile",
                    "accelerator_type": "v5litepod-16",
                },
                "transcoder": {
                    "resource_id": "transcoder-profile",
                    "accelerator_type": "v5litepod-16",
                },
            },
        },
        source_revision=REVISION,
    )
    assert len(plan["tasks"]) == 4
    assert plan["bounds"]["maximum_seconds_per_task"] == 600
    assert not plan["bounds"]["training_or_checkpoint_publication_allowed"]
