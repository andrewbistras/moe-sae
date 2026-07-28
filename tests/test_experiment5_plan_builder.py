from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from moe_sae.experiment5_campaign import (
    CONDITIONS,
    MILESTONES,
    PANELS,
    SEEDS,
    parse_experiment5_campaign_plan,
)
from moe_sae.experiment5_plan_builder import (
    EXPECTED_SURVIVING_TPUS,
    PREEMPTED_TPU_NAME,
    Experiment5PlanBuildError,
    build_experiment5_plan,
    main,
)

COMMIT = "a" * 40


def _evidence(resource_id: str, region: str, *, mirror: bool) -> dict:
    def item(kind: str) -> dict[str, str]:
        payload = f"{resource_id}-{kind}".encode()
        return {
            "controller_path": (
                f"/mnt/{region}/horizon-10m/evidence/{resource_id}/{kind}.json"
            ),
            "remote_path": (
                f"/home/worker/moe-sae/artifacts/gemma4_v1/production/"
                f"horizon-10m/evidence/{resource_id}/{kind}.json"
            ),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    return {
        "controller_mapping": item("mapping"),
        "cache_inventory": item("inventory"),
        "mirror_seal": item("mirror") if mirror else None,
    }


def _source() -> dict:
    physical = (
        ("slice-sae-a", "gemma4-horizon-v5e-usc-a", "us-central1-a", "sae"),
        ("slice-sae-r", "gemma4-horizon-v5e-euw2-f", "europe-west4-b", "sae"),
        ("slice-sae-b", "gemma4-horizon-harvest-0728", "us-central2-b", "sae"),
        ("slice-sae-f", PREEMPTED_TPU_NAME, "us-central1-a", "sae"),
        (
            "slice-transcoder-a",
            "gemma4-horizon-v5e-euw-a",
            "europe-west4-b",
            "transcoder",
        ),
        (
            "slice-transcoder-r",
            "gemma4-horizon-v5e-euw-r",
            "europe-west4-b",
            "transcoder",
        ),
        (
            "slice-transcoder-b",
            "gemma4-horizon-v4-spot-trans-b",
            "us-central2-b",
            "transcoder",
        ),
        (
            "slice-transcoder-f",
            "gemma4-horizon-v5e-euw2-b",
            "europe-west4-b",
            "transcoder",
        ),
    )
    resources = []
    trajectories = []
    for resource_id, tpu_name, zone, model in physical:
        region = zone.rsplit("-", 1)[0]
        accelerator = (
            "v4-32"
            if "v4-" in tpu_name or tpu_name == "gemma4-horizon-harvest-0728"
            else "v5litepod-16"
        )
        regional_root = {
            "us-central1": "gs://usc-bucket/horizon-10m",
            "europe-west4": "gs://euw-bucket/horizon-10m",
        }.get(region)
        environment = {}
        if regional_root is not None:
            environment = {
                "BULK_GCS_ROOT": regional_root,
                "MOE_SAE_V5LITEPOD_DEVICE_KIND": "TPU v5 lite",
            }
        resources.append(
            {
                "id": resource_id,
                "project": "test-project",
                "tpu_name": tpu_name,
                "zone": zone,
                "accelerator_type": accelerator,
                "internal_ip": True,
                "ssh_user": "worker",
                "evidence": _evidence(
                    resource_id,
                    {
                        "us-central1": "moe-sae-usc1",
                        "europe-west4": "moe-sae-ew4",
                        "us-central2": "moe-sae-control",
                    }[region],
                    mirror=accelerator != "v4-32",
                ),
                "environment": environment,
            }
        )
        mount = {
            "us-central1": "/mnt/moe-sae-usc1",
            "europe-west4": "/mnt/moe-sae-ew4",
            "us-central2": "/mnt/moe-sae-control",
        }[region]
        trajectories.append(
            {
                "id": f"old-{resource_id}",
                "scientific_model": model,
                "resource_id": resource_id,
                "completion_path_controller": (
                    f"{mount}/horizon-10m/runs/{model}/runs/old-{resource_id}/"
                    "training-result.json"
                ),
            }
        )
    return {
        "schema_version": "gemma4-horizon-direct-campaign-plan-v1",
        "campaign_id": "old-r8",
        "working_directory": "/home/controller/moe-sae",
        "python_executable": "/home/controller/moe-sae/.venv/bin/python",
        "remote_project_root": "/home/worker/moe-sae",
        "remote_uv": "/home/worker/.local/bin/uv",
        "environment": {
            "ARTIFACT_ROOT": (
                "/home/worker/moe-sae/artifacts/gemma4_v1/production"
            ),
            "BULK_GCS_ROOT": "gs://central-bucket/moe-sae/gemma4_v1",
            "CONTRIBUTION_TRAIN_CAPTURE_ROOT": "/captures/target/train",
            "CONTRIBUTION_EVAL_CAPTURE_ROOT": "/captures/target/eval",
            "CONTRIBUTION_OBJECT_PREFIX": "gs://source/target/train",
            "CONTRIBUTION_EVAL_OBJECT_PREFIX": "gs://source/target/eval",
            "CONTRIBUTION_SOURCE_OVERLAY_ROOT": "/captures/overlay/train",
            "CONTRIBUTION_EVAL_SOURCE_OVERLAY_ROOT": "/captures/overlay/eval",
            "CONTRIBUTION_SOURCE_OVERLAY_PREFIX": "gs://source/overlay/train",
            "CONTRIBUTION_EVAL_SOURCE_OVERLAY_PREFIX": "gs://source/overlay/eval",
            "CONTRIBUTION_RECOMPOSITION_SIDECAR": "/captures/sidecar.json",
            "CONTRIBUTION_SAE_CACHE_ROOT": "/dev/shm/moe-sae/cache",
            "CONTRIBUTION_SAE_CACHE_MAX_BYTES": "171798691840",
            "TRAINING_PROJECT_COMMIT": "b" * 40,
            "TRANSCODER_HARVEST_CHIP_SECONDS": "8987.674096607989",
        },
        "resources": resources,
        "trajectories": trajectories,
    }


def test_builds_exact_safe_matrix_and_filters_preempted_resource() -> None:
    raw = build_experiment5_plan(_source(), project_commit=COMMIT)
    plan = parse_experiment5_campaign_plan(raw)
    short_commit = COMMIT[:12]
    assert all(
        f"/experiment5/{short_commit}/" in task.control_root_remote
        for task in plan.trajectories
    )
    assert all(
        f"/experiment5/{short_commit}/" in task.checkpoint_root_gcs
        for task in plan.trajectories
    )
    assert all(
        f"/experiment5/{short_commit}/" in str(task.completion_path_controller)
        for task in plan.trajectories
    )
    assert all(
        f"/experiment5/{short_commit}/" in task.result_root_remote
        and f"/experiment5/{short_commit}/"
        in str(task.completion_path_controller)
        and f"/experiment5/{short_commit}/"
        in str(task.paired_scores_path_controller)
        for task in plan.evaluations
    )

    assert len(plan.resources) == 7
    assert len(plan.trajectories) == 24
    assert len(plan.evaluations) == 18
    assert {resource.tpu_name for resource in plan.resources} == set(
        EXPECTED_SURVIVING_TPUS
    )
    assert PREEMPTED_TPU_NAME not in {
        resource.tpu_name for resource in plan.resources
    }
    assert {
        (task.panel, task.condition, task.seed) for task in plan.trajectories
    } == {
        (panel, condition, seed)
        for panel in PANELS
        for condition in CONDITIONS
        for seed in SEEDS
    }
    assert {
        (task.panel, task.seed, task.milestone_tokens)
        for task in plan.evaluations
    } == {
        (panel, seed, milestone)
        for panel in PANELS
        for seed in SEEDS
        for milestone in MILESTONES
    }
    assert all(resource.deployed_commit == COMMIT for resource in plan.resources)


def test_planes_profiles_overlays_and_regional_storage_are_exact() -> None:
    source = _source()
    plan = parse_experiment5_campaign_plan(
        build_experiment5_plan(source, project_commit=COMMIT)
    )
    resources = {resource.resource_id: resource for resource in plan.resources}
    paired_names = {
        "gemma4-horizon-v5e-euw-a",
        "gemma4-horizon-v5e-euw-r",
        "gemma4-horizon-v5e-euw2-b",
        "gemma4-horizon-v4-spot-trans-b",
    }
    assert {
        resource.tpu_name
        for resource in plan.resources
        if resource.cache_profile == "paired"
    } == paired_names

    for panel in PANELS:
        for seed in SEEDS:
            group = [
                task
                for task in plan.trajectories
                if task.panel == panel and task.seed == seed
            ]
            eligible = {
                resources[resource_id]
                for task in group
                for resource_id in task.eligible_resource_ids
            }
            assert len({resource.region for resource in eligible}) == 1
            assert len({resource.accelerator_type for resource in eligible}) == 1
            if panel == "transcoder":
                assert {resource.cache_profile for resource in eligible} == {"paired"}
                assert {
                    task.source_overlay_control_root_remote for task in group
                } == {"/captures/overlay/train"}
                assert {
                    task.source_overlay_object_prefix_gcs for task in group
                } == {"gs://source/overlay/train"}
            else:
                assert {
                    task.source_overlay_control_root_remote for task in group
                } == {None}

            h = next(task for task in group if task.condition == "H")
            f = next(task for task in group if task.condition == "F")
            assert {
                resources[item].accelerator_type for item in h.eligible_resource_ids
            } == {
                resources[item].accelerator_type for item in f.eligible_resource_ids
            }
            if {resource.region for resource in eligible} == {"europe-west4"}:
                assert h.checkpoint_root_gcs.startswith("gs://euw-bucket/experiment5/")
                assert str(h.completion_path_controller).startswith(
                    "/mnt/moe-sae-ew4/experiment5/"
                )
            elif {resource.region for resource in eligible} == {"us-central1"}:
                assert h.checkpoint_root_gcs.startswith("gs://usc-bucket/experiment5/")
                assert str(h.completion_path_controller).startswith(
                    "/mnt/moe-sae-usc1/experiment5/"
                )
            else:
                assert h.checkpoint_root_gcs.startswith(
                    "gs://central-bucket/moe-sae/gemma4_v1/experiment5/"
                )
                assert str(h.completion_path_controller).startswith(
                    "/mnt/moe-sae-control/experiment5/"
                )

    transcoder_evals = [
        task for task in plan.evaluations if task.panel == "transcoder"
    ]
    assert {task.eval_source_root_remote for task in transcoder_evals} == {
        "/captures/overlay/eval"
    }
    assert {task.eval_source_object_prefix_gcs for task in transcoder_evals} == {
        "gs://source/overlay/eval"
    }
    assert "CONTRIBUTION_SOURCE_OVERLAY_ROOT" not in plan.environment_dict()
    assert plan.environment_dict()["CONTRIBUTION_TRAIN_CAPTURE_ROOT"] == (
        "/captures/target/train"
    )


def test_rejects_inventory_or_cache_proof_drift() -> None:
    changed = _source()
    changed["resources"][0]["tpu_name"] = "unexpected-slice"
    with pytest.raises(Experiment5PlanBuildError, match="physical inventory changed"):
        build_experiment5_plan(changed, project_commit=COMMIT)

    missing = _source()
    missing["trajectories"] = [
        task
        for task in missing["trajectories"]
        if task["resource_id"] != "slice-transcoder-a"
    ]
    with pytest.raises(Experiment5PlanBuildError, match="no prior cache assignment"):
        build_experiment5_plan(missing, project_commit=COMMIT)

    incomplete = _source()
    incomplete["resources"][0]["evidence"]["cache_inventory"].pop("sha256")
    with pytest.raises(Experiment5PlanBuildError, match="witness is incomplete"):
        build_experiment5_plan(incomplete, project_commit=COMMIT)


def test_cli_is_create_only_and_deterministic(tmp_path: Path) -> None:
    source_path = tmp_path / "r8.yaml"
    output_path = tmp_path / "experiment5.yaml"
    source_path.write_text(yaml.safe_dump(_source(), sort_keys=False), encoding="utf-8")

    assert (
        main(
            [
                "--r8-plan",
                str(source_path),
                "--project-commit",
                COMMIT,
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    emitted = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    expected = build_experiment5_plan(_source(), project_commit=COMMIT)
    assert emitted == expected
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(
            [
                "--r8-plan",
                str(source_path),
                "--project-commit",
                COMMIT,
                "--output",
                str(output_path),
            ]
        )


def test_conflicting_prior_assignment_is_rejected() -> None:
    source = _source()
    duplicate = copy.deepcopy(source["trajectories"][0])
    duplicate["scientific_model"] = "transcoder"
    source["trajectories"].append(duplicate)
    with pytest.raises(Experiment5PlanBuildError, match="conflicting"):
        build_experiment5_plan(source, project_commit=COMMIT)
