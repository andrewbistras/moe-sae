from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

import pytest
import yaml

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_campaign import (
    CONDITIONS,
    MILESTONES,
    PANELS,
    SEEDS,
    parse_experiment5_campaign_plan,
)
from moe_sae.experiment5_plan_builder import (
    EXECUTION_PROFILE,
    EXPECTED_RESOURCE_EVIDENCE_OVERRIDES,
    EXPECTED_SURVIVING_TPUS,
    PREEMPTED_TPU_NAME,
    RESOURCE_EVIDENCE_OVERRIDE_SCHEMA_VERSION,
    Experiment5PlanBuildError,
    build_resource_evidence_overrides,
    build_experiment5_plan,
    evidence_override_main,
    main,
)
from moe_sae.regional_execution import (
    assert_spot_tpu_description,
    build_accelerator_contract,
    build_controller_observation,
    build_local_cache_observation,
    seal_controller_mapping,
    seal_local_cache_inventory,
)

COMMIT = "a" * 40
GIB = 1024**3


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
            "ARTIFACT_ROOT": ("/home/worker/moe-sae/artifacts/gemma4_v1/production"),
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


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _reseal(value: dict, hash_field: str) -> None:
    body = dict(value)
    body.pop(hash_field, None)
    value[hash_field] = canonical_sha256(body)


def _rewrite_override_evidence(
    evidence: dict,
    value: dict,
) -> None:
    path = Path(evidence["controller_path"])
    evidence["sha256"] = _write_json(path, value)


def _fresh_evidence_overrides(
    tmp_path: Path,
    source: dict,
    *,
    project_commit: str = COMMIT,
    second_device_kind: bool = False,
) -> dict:
    controller_mount = tmp_path / "controller-mount"
    for trajectory in source["trajectories"]:
        if trajectory["resource_id"] in EXPECTED_RESOURCE_EVIDENCE_OVERRIDES:
            _, suffix = trajectory["completion_path_controller"].split(
                "/horizon-10m/", 1
            )
            trajectory["completion_path_controller"] = str(
                controller_mount / "horizon-10m" / suffix
            )
    resources_by_id = {resource["id"]: resource for resource in source["resources"]}
    cache_root = source["environment"]["CONTRIBUTION_SAE_CACHE_ROOT"]
    records: dict[str, dict] = {}
    for resource_number, (
        resource_id,
        expected_tpu_name,
    ) in enumerate(sorted(EXPECTED_RESOURCE_EVIDENCE_OVERRIDES.items())):
        resource = resources_by_id[resource_id]
        assert resource["tpu_name"] == expected_tpu_name
        create_time = f"2026-07-{20 + resource_number:02d}T12:00:00Z"
        contract = build_accelerator_contract(
            run_id=(
                f"gemma4-experiment5-{EXECUTION_PROFILE}-" f"{project_commit[:12]}"
            ),
            scientific_model="hybrid",
            condition="all",
            project=resource["project"],
            tpu_name=expected_tpu_name,
            resource_create_time=create_time,
            region=resource["zone"].rsplit("-", 1)[0],
            zone=resource["zone"],
            accelerator_type=resource["accelerator_type"],
            provisioning_model=(
                "ON_DEMAND" if resource_id == "slice-sae-b" else "SPOT"
            ),
        )
        description = {
            "name": (
                f"projects/{resource['project']}/locations/{resource['zone']}/"
                f"nodes/{expected_tpu_name}"
            ),
            "acceleratorType": resource["accelerator_type"],
            "state": "READY",
            "createTime": create_time,
            "schedulingConfig": (
                {}
                if resource_id == "slice-sae-b"
                else {"spot": True, "preemptible": True}
            ),
        }
        attestation = assert_spot_tpu_description(description, contract)
        probe_id = f"probe-exp5-{resource_number}"
        observations = []
        for controller in range(4):
            device_kind = (
                "unexpected second kind"
                if second_device_kind and controller == 3
                else (
                    f"TPU v5 lite fresh {resource_number}"
                    if resource["accelerator_type"] == "v5litepod-16"
                    else "TPU v4"
                )
            )
            observations.append(
                build_controller_observation(
                    contract=contract,
                    probe_id=probe_id,
                    controller_index=controller,
                    process_index=(controller + resource_number) % 4,
                    process_count=4,
                    global_device_count=16,
                    local_device_count=4,
                    platforms=["tpu"],
                    device_kinds=[device_kind],
                    boot_id=(
                        f"{resource_number + 1:08x}-"
                        f"0000-0000-0000-{controller + 1:012x}"
                    ),
                    hostname=f"host-{resource_number}-{controller}",
                )
            )
        mapping = seal_controller_mapping(
            contract=contract,
            probe_id=probe_id,
            observations=observations,
        )
        cache_observations = [
            build_local_cache_observation(
                contract=contract,
                controller_mapping=mapping,
                controller_index=controller,
                cache_root=cache_root,
                required_lane_bytes=159_000_000_000,
                required_free_bytes=176 * GIB,
                available_bytes=180 * GIB,
                mem_available_bytes=180 * GIB,
                mount_target="/dev/shm",
                mount_source="tmpfs",
                filesystem_type="tmpfs",
                mount_options=["rw", "nosuid", "nodev", "noexec"],
                boot_id=(
                    f"{resource_number + 1:08x}-"
                    f"0000-0000-0000-{controller + 1:012x}"
                ),
            )
            for controller in range(4)
        ]
        inventory = seal_local_cache_inventory(
            contract=contract,
            controller_mapping=mapping,
            observations=cache_observations,
        )
        controller_root = (
            controller_mount
            / "horizon-10m"
            / "topology"
            / expected_tpu_name
            / "assignments"
            / "fresh"
            / "probes"
            / probe_id
        )
        remote_root = (
            "/home/worker/moe-sae/artifacts/gemma4_v1/production/"
            f"horizon-10m/topology/{expected_tpu_name}/assignments/fresh/"
            f"probes/{probe_id}"
        )
        mapping_path = controller_root / "controller-mapping.json"
        inventory_path = controller_root / "cache-inventory.json"
        assignment_root = controller_root.parent.parent
        remote_assignment_root = str(PurePosixPath(remote_root).parent.parent)
        records[resource_id] = {
            "tpu_name": expected_tpu_name,
            "tpu_description": {
                "controller_path": str(assignment_root / "tpu-description.json"),
                "remote_path": f"{remote_assignment_root}/tpu-description.json",
                "sha256": _write_json(
                    assignment_root / "tpu-description.json",
                    description,
                ),
            },
            "accelerator_contract": {
                "controller_path": str(assignment_root / "accelerator-contract.json"),
                "remote_path": (f"{remote_assignment_root}/accelerator-contract.json"),
                "sha256": _write_json(
                    assignment_root / "accelerator-contract.json",
                    contract,
                ),
            },
            "spot_attestation": {
                "controller_path": str(assignment_root / "spot-attestation.json"),
                "remote_path": f"{remote_assignment_root}/spot-attestation.json",
                "sha256": _write_json(
                    assignment_root / "spot-attestation.json",
                    attestation,
                ),
            },
            "controller_mapping": {
                "controller_path": str(mapping_path),
                "remote_path": f"{remote_root}/controller-mapping.json",
                "sha256": _write_json(mapping_path, mapping),
            },
            "cache_inventory": {
                "controller_path": str(inventory_path),
                "remote_path": f"{remote_root}/cache-inventory.json",
                "sha256": _write_json(inventory_path, inventory),
            },
        }
    return {
        "schema_version": RESOURCE_EVIDENCE_OVERRIDE_SCHEMA_VERSION,
        "execution_profile": EXECUTION_PROFILE,
        "project_commit": project_commit,
        "resources": records,
    }


def _rewrite_authority_chain(
    record: dict,
    *,
    contract: dict,
    attestation: dict,
    mapping: dict,
    inventory: dict,
) -> None:
    _reseal(contract, "contract_sha256")
    attestation["contract_sha256"] = contract["contract_sha256"]
    _reseal(attestation, "attestation_sha256")
    mapping["contract_sha256"] = contract["contract_sha256"]
    _reseal(mapping, "mapping_sha256")
    inventory["contract_sha256"] = contract["contract_sha256"]
    inventory["controller_mapping_sha256"] = mapping["mapping_sha256"]
    _reseal(inventory, "inventory_sha256")
    for name, value in (
        ("accelerator_contract", contract),
        ("spot_attestation", attestation),
        ("controller_mapping", mapping),
        ("cache_inventory", inventory),
    ):
        _rewrite_override_evidence(record[name], value)


def _build_ready(source: dict, tmp_path: Path) -> dict:
    overrides = _fresh_evidence_overrides(tmp_path, source)
    return build_experiment5_plan(
        source,
        project_commit=COMMIT,
        execution_profile=EXECUTION_PROFILE,
        resource_evidence_overrides=overrides,
    )


def test_builds_exact_safe_matrix_and_filters_preempted_resource(
    tmp_path: Path,
) -> None:
    raw = _build_ready(_source(), tmp_path)
    plan = parse_experiment5_campaign_plan(raw)
    short_commit = COMMIT[:12]
    artifact_marker = f"/experiment5/{EXECUTION_PROFILE}/{short_commit}/"
    assert all(
        artifact_marker in task.control_root_remote for task in plan.trajectories
    )
    assert all(
        artifact_marker in task.checkpoint_root_gcs for task in plan.trajectories
    )
    assert all(
        artifact_marker in str(task.completion_path_controller)
        for task in plan.trajectories
    )
    assert all(
        artifact_marker in task.result_root_remote
        and artifact_marker in str(task.completion_path_controller)
        and artifact_marker in str(task.paired_scores_path_controller)
        for task in plan.evaluations
    )

    assert len(plan.resources) == 7
    assert len(plan.trajectories) == 24
    assert len(plan.evaluations) == 18
    assert plan.campaign_id == (
        f"gemma4-experiment5-{EXECUTION_PROFILE}-{short_commit}"
    )
    assert str(plan.state_root).endswith(
        f"/experiment5/controller/{EXECUTION_PROFILE}/{short_commit}"
    )
    assert {resource.tpu_name for resource in plan.resources} == set(
        EXPECTED_SURVIVING_TPUS
    )
    assert PREEMPTED_TPU_NAME not in {resource.tpu_name for resource in plan.resources}
    assert {(task.panel, task.condition, task.seed) for task in plan.trajectories} == {
        (panel, condition, seed)
        for panel in PANELS
        for condition in CONDITIONS
        for seed in SEEDS
    }
    assert {
        (task.panel, task.seed, task.milestone_tokens) for task in plan.evaluations
    } == {
        (panel, seed, milestone)
        for panel in PANELS
        for seed in SEEDS
        for milestone in MILESTONES
    }
    assert all(resource.deployed_commit == COMMIT for resource in plan.resources)
    assert plan.max_parallel_train == 3
    assert plan.max_parallel_eval == 3


def test_planes_profiles_overlays_and_regional_storage_are_exact(
    tmp_path: Path,
) -> None:
    source = _source()
    plan = parse_experiment5_campaign_plan(_build_ready(source, tmp_path))
    resources = {resource.resource_id: resource for resource in plan.resources}
    active_paired_names = {
        "gemma4-horizon-v5e-usc-a",
        "gemma4-horizon-harvest-0728",
        "gemma4-horizon-v4-spot-trans-b",
    }
    assert active_paired_names <= {
        resource.tpu_name
        for resource in plan.resources
        if resource.cache_profile == "paired"
    }
    active_ids = {
        resource_id
        for task in (*plan.trajectories, *plan.evaluations)
        for resource_id in task.eligible_resource_ids
    }
    assert active_ids == {
        "slice-sae-a",
        "slice-sae-b",
        "slice-transcoder-b",
    }
    assert {resource.resource_id for resource in plan.resources} - active_ids == {
        "slice-sae-r",
        "slice-transcoder-a",
        "slice-transcoder-r",
        "slice-transcoder-f",
    }

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
                assert {task.source_overlay_control_root_remote for task in group} == {
                    "/captures/overlay/train"
                }
                assert {task.source_overlay_object_prefix_gcs for task in group} == {
                    "gs://source/overlay/train"
                }
            else:
                assert {task.source_overlay_control_root_remote for task in group} == {
                    None
                }

            h = next(task for task in group if task.condition == "H")
            f = next(task for task in group if task.condition == "F")
            expected_ids = (
                ("slice-sae-a",) if seed == 0 else ("slice-sae-b", "slice-transcoder-b")
            )
            assert {task.eligible_resource_ids for task in group} == {expected_ids}
            assert {
                resources[item].accelerator_type for item in h.eligible_resource_ids
            } == {resources[item].accelerator_type for item in f.eligible_resource_ids}
            if {resource.region for resource in eligible} == {"us-central1"}:
                assert h.checkpoint_root_gcs.startswith(
                    f"gs://usc-bucket/experiment5/{EXECUTION_PROFILE}/"
                )
            else:
                assert h.checkpoint_root_gcs.startswith(
                    "gs://central-bucket/moe-sae/gemma4_v1/experiment5/"
                    f"{EXECUTION_PROFILE}/"
                )
            assert f"/experiment5/{EXECUTION_PROFILE}/{COMMIT[:12]}/" in str(
                h.completion_path_controller
            )

    transcoder_evals = [task for task in plan.evaluations if task.panel == "transcoder"]
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


def test_rejects_inventory_or_cache_proof_drift(tmp_path: Path) -> None:
    changed = _source()
    changed_overrides = _fresh_evidence_overrides(tmp_path / "changed", changed)
    changed["resources"][0]["tpu_name"] = "unexpected-slice"
    with pytest.raises(Experiment5PlanBuildError, match="physical inventory changed"):
        build_experiment5_plan(
            changed,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
            resource_evidence_overrides=changed_overrides,
        )

    missing = _source()
    missing_overrides = _fresh_evidence_overrides(tmp_path / "missing", missing)
    missing["trajectories"] = [
        task
        for task in missing["trajectories"]
        if task["resource_id"] != "slice-transcoder-a"
    ]
    with pytest.raises(Experiment5PlanBuildError, match="no prior cache assignment"):
        build_experiment5_plan(
            missing,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
            resource_evidence_overrides=missing_overrides,
        )

    incomplete = _source()
    incomplete_overrides = _fresh_evidence_overrides(
        tmp_path / "incomplete", incomplete
    )
    incomplete["resources"][0]["evidence"]["cache_inventory"].pop("sha256")
    with pytest.raises(Experiment5PlanBuildError, match="witness is incomplete"):
        build_experiment5_plan(
            incomplete,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
            resource_evidence_overrides=incomplete_overrides,
        )


def test_cli_is_create_only_and_deterministic(tmp_path: Path) -> None:
    source = _source()
    overrides = _fresh_evidence_overrides(tmp_path / "evidence", source)
    source_path = tmp_path / "r8.yaml"
    override_path = tmp_path / "overrides.yaml"
    output_path = tmp_path / "experiment5.yaml"
    source_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    override_path.write_text(
        yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8"
    )

    common = [
        "--r8-plan",
        str(source_path),
        "--project-commit",
        COMMIT,
        "--execution-profile",
        EXECUTION_PROFILE,
        "--resource-evidence-overrides",
        str(override_path),
        "--output",
        str(output_path),
    ]
    assert main(common) == 0
    emitted = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    expected = build_experiment5_plan(
        source,
        project_commit=COMMIT,
        execution_profile=EXECUTION_PROFILE,
        resource_evidence_overrides=overrides,
    )
    assert emitted == expected
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(common)


def test_conflicting_prior_assignment_is_rejected(tmp_path: Path) -> None:
    source = _source()
    overrides = _fresh_evidence_overrides(tmp_path, source)
    duplicate = copy.deepcopy(source["trajectories"][0])
    duplicate["scientific_model"] = "transcoder"
    source["trajectories"].append(duplicate)
    with pytest.raises(Experiment5PlanBuildError, match="conflicting"):
        build_experiment5_plan(
            source,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
            resource_evidence_overrides=overrides,
        )


def test_fresh_active_evidence_override_is_exact_and_derives_device_kind(
    tmp_path: Path,
) -> None:
    source = _source()
    old_resources = {
        resource["id"]: copy.deepcopy(resource) for resource in source["resources"]
    }
    overrides = _fresh_evidence_overrides(tmp_path, source)

    raw = build_experiment5_plan(
        source,
        project_commit=COMMIT,
        execution_profile=EXECUTION_PROFILE,
        resource_evidence_overrides=overrides,
    )
    resources = {resource["id"]: resource for resource in raw["resources"]}
    for resource_number, resource_id in enumerate(
        sorted(EXPECTED_RESOURCE_EVIDENCE_OVERRIDES)
    ):
        emitted = resources[resource_id]
        replacement = overrides["resources"][resource_id]
        assert (
            emitted["evidence"]["controller_mapping"]
            == replacement["controller_mapping"]
        )
        assert emitted["evidence"]["cache_inventory"] == replacement["cache_inventory"]
        assert (
            emitted["evidence"]["mirror_seal"]
            == old_resources[resource_id]["evidence"]["mirror_seal"]
        )
        if emitted["accelerator_type"] == "v5litepod-16":
            assert emitted["environment"]["MOE_SAE_V5LITEPOD_DEVICE_KIND"] == (
                f"TPU v5 lite fresh {resource_number}"
            )
        else:
            assert "MOE_SAE_V5LITEPOD_DEVICE_KIND" not in emitted["environment"]

    # The inactive Europe rows retain historical witnesses byte-for-byte.
    for resource_id in {
        "slice-sae-r",
        "slice-transcoder-a",
        "slice-transcoder-r",
        "slice-transcoder-f",
    }:
        assert (
            resources[resource_id]["evidence"] == old_resources[resource_id]["evidence"]
        )


def test_override_rejects_partial_unknown_and_patch_style_fields(
    tmp_path: Path,
) -> None:
    source = _source()
    overrides = _fresh_evidence_overrides(tmp_path, source)

    with pytest.raises(Experiment5PlanBuildError, match="requires fresh"):
        build_experiment5_plan(
            source,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
        )

    wrong_profile = copy.deepcopy(overrides)
    wrong_profile["execution_profile"] = "other-profile"
    with pytest.raises(Experiment5PlanBuildError, match="another execution profile"):
        build_experiment5_plan(
            source,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
            resource_evidence_overrides=wrong_profile,
        )

    partial = copy.deepcopy(overrides)
    partial["resources"].pop("slice-sae-a")
    with pytest.raises(Experiment5PlanBuildError, match="override IDs changed"):
        build_experiment5_plan(
            source,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
            resource_evidence_overrides=partial,
        )

    unknown = copy.deepcopy(overrides)
    unknown["resources"]["unexpected"] = unknown["resources"].pop("slice-sae-a")
    with pytest.raises(Experiment5PlanBuildError, match="override IDs changed"):
        build_experiment5_plan(
            source,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
            resource_evidence_overrides=unknown,
        )

    patch_style = copy.deepcopy(overrides)
    patch_style["resources"]["slice-sae-a"]["mirror_seal"] = {}
    with pytest.raises(Experiment5PlanBuildError, match="fields changed"):
        build_experiment5_plan(
            source,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
            resource_evidence_overrides=patch_style,
        )


def test_override_rejects_wrong_commit_stale_witness_and_nonunique_device_kind(
    tmp_path: Path,
) -> None:
    source = _source()
    wrong_commit = _fresh_evidence_overrides(tmp_path / "commit", source)
    wrong_commit["project_commit"] = "b" * 40
    with pytest.raises(Experiment5PlanBuildError, match="another project commit"):
        build_experiment5_plan(
            source,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
            resource_evidence_overrides=wrong_commit,
        )

    stale = _fresh_evidence_overrides(tmp_path / "stale", source)
    source_by_id = {resource["id"]: resource for resource in source["resources"]}
    stale_record = stale["resources"]["slice-sae-a"]
    stale_record["controller_mapping"] = copy.deepcopy(
        source_by_id["slice-sae-a"]["evidence"]["controller_mapping"]
    )
    with pytest.raises(Experiment5PlanBuildError, match="stale witnesses"):
        build_experiment5_plan(
            source,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
            resource_evidence_overrides=stale,
        )

    mixed_devices = _fresh_evidence_overrides(
        tmp_path / "devices",
        source,
        second_device_kind=True,
    )
    with pytest.raises(Experiment5PlanBuildError, match="exact live role"):
        build_experiment5_plan(
            source,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
            resource_evidence_overrides=mixed_devices,
        )


def test_override_rejects_byte_hash_and_cross_probe_inventory(
    tmp_path: Path,
) -> None:
    source = _source()
    overrides = _fresh_evidence_overrides(tmp_path, source)

    bad_hash = copy.deepcopy(overrides)
    bad_hash["resources"]["slice-sae-a"]["controller_mapping"]["sha256"] = "0" * 64
    with pytest.raises(Experiment5PlanBuildError, match="byte SHA-256 differs"):
        build_experiment5_plan(
            source,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
            resource_evidence_overrides=bad_hash,
        )

    crossed = copy.deepcopy(overrides)
    left = crossed["resources"]["slice-sae-a"]
    right = crossed["resources"]["slice-sae-b"]
    left["cache_inventory"] = copy.deepcopy(right["cache_inventory"])
    # Keep the common-root path contract while presenting another mapping's
    # signed inventory; semantic authority validation must still reject it.
    source_path = Path(left["cache_inventory"]["controller_path"])
    destination = (
        Path(left["controller_mapping"]["controller_path"]).parent
        / "cache-inventory.json"
    )
    destination.write_bytes(source_path.read_bytes())
    left["cache_inventory"] = {
        **left["cache_inventory"],
        "controller_path": str(destination),
        "remote_path": str(
            PurePosixPath(left["controller_mapping"]["remote_path"]).parent
            / "cache-inventory.json"
        ),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    with pytest.raises(
        Experiment5PlanBuildError, match="fresh incarnation evidence is invalid"
    ):
        build_experiment5_plan(
            source,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
            resource_evidence_overrides=crossed,
        )


def test_cli_accepts_strict_resource_evidence_override(tmp_path: Path) -> None:
    source = _source()
    source_path = tmp_path / "r8.yaml"
    override_path = tmp_path / "fresh-evidence.yaml"
    output_path = tmp_path / "experiment5.yaml"
    overrides = _fresh_evidence_overrides(tmp_path / "evidence", source)
    source_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    override_path.write_text(
        yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8"
    )

    assert (
        main(
            [
                "--r8-plan",
                str(source_path),
                "--project-commit",
                COMMIT,
                "--execution-profile",
                EXECUTION_PROFILE,
                "--resource-evidence-overrides",
                str(override_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    emitted = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    emitted_by_id = {item["id"]: item for item in emitted["resources"]}
    assert (
        emitted_by_id["slice-sae-a"]["evidence"]["controller_mapping"]
        == overrides["resources"]["slice-sae-a"]["controller_mapping"]
    )


def test_override_binds_commit_run_id_and_contract_project(
    tmp_path: Path,
) -> None:
    for mutation, expected in (
        ("run_id", "exact live role"),
        ("project", "fresh incarnation evidence is invalid"),
        ("provisioning", "fresh incarnation evidence is invalid"),
    ):
        source = _source()
        overrides = _fresh_evidence_overrides(tmp_path / mutation, source)
        record = overrides["resources"]["slice-sae-a"]
        contract = json.loads(
            Path(record["accelerator_contract"]["controller_path"]).read_text(
                encoding="ascii"
            )
        )
        attestation = json.loads(
            Path(record["spot_attestation"]["controller_path"]).read_text(
                encoding="ascii"
            )
        )
        mapping = json.loads(
            Path(record["controller_mapping"]["controller_path"]).read_text(
                encoding="ascii"
            )
        )
        inventory = json.loads(
            Path(record["cache_inventory"]["controller_path"]).read_text(
                encoding="ascii"
            )
        )
        if mutation == "run_id":
            contract["run_id"] = "gemma4-experiment5-oldcommit000"
            mapping["run_id"] = contract["run_id"]
        elif mutation == "project":
            contract["project"] = "other-project"
        else:
            contract["provisioning_model"] = "ON_DEMAND"
        _rewrite_authority_chain(
            record,
            contract=contract,
            attestation=attestation,
            mapping=mapping,
            inventory=inventory,
        )
        with pytest.raises(Experiment5PlanBuildError, match=expected):
            build_experiment5_plan(
                source,
                project_commit=COMMIT,
                execution_profile=EXECUTION_PROFILE,
                resource_evidence_overrides=overrides,
            )


def test_override_binds_each_active_resource_provisioning_model(
    tmp_path: Path,
) -> None:
    source = _source()
    overrides = _fresh_evidence_overrides(tmp_path, source)
    record = overrides["resources"]["slice-sae-b"]
    description = json.loads(
        Path(record["tpu_description"]["controller_path"]).read_text(encoding="ascii")
    )
    contract = json.loads(
        Path(record["accelerator_contract"]["controller_path"]).read_text(
            encoding="ascii"
        )
    )
    mapping = json.loads(
        Path(record["controller_mapping"]["controller_path"]).read_text(
            encoding="ascii"
        )
    )
    inventory = json.loads(
        Path(record["cache_inventory"]["controller_path"]).read_text(encoding="ascii")
    )
    contract["provisioning_model"] = "SPOT"
    description["schedulingConfig"] = {"spot": True, "preemptible": True}
    _reseal(contract, "contract_sha256")
    attestation = assert_spot_tpu_description(description, contract)
    _rewrite_override_evidence(record["tpu_description"], description)
    _rewrite_authority_chain(
        record,
        contract=contract,
        attestation=attestation,
        mapping=mapping,
        inventory=inventory,
    )
    with pytest.raises(Experiment5PlanBuildError, match="exact live role"):
        build_experiment5_plan(
            source,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
            resource_evidence_overrides=overrides,
        )


def test_override_cross_checks_inventory_count_and_mapping_pairs(
    tmp_path: Path,
) -> None:
    for mutation in ("count", "permutation", "paired-capacity"):
        source = _source()
        overrides = _fresh_evidence_overrides(tmp_path / mutation, source)
        record = overrides["resources"]["slice-sae-a"]
        inventory_path = Path(record["cache_inventory"]["controller_path"])
        inventory = json.loads(inventory_path.read_text(encoding="ascii"))
        if mutation == "count":
            extra = dict(inventory["workers"][-1])
            extra.update(
                {
                    "controller_index": 4,
                    "process_index": 4,
                    "observation_sha256": "f" * 64,
                }
            )
            inventory["process_count"] = 5
            inventory["workers"].append(extra)
        elif mutation == "permutation":
            inventory["workers"][0]["process_index"] = 1
            inventory["workers"][1]["process_index"] = 0
        else:
            inventory["workers"][0]["required_lane_bytes"] -= 1
        _reseal(inventory, "inventory_sha256")
        _rewrite_override_evidence(record["cache_inventory"], inventory)
        with pytest.raises(Experiment5PlanBuildError, match="exact live role"):
            build_experiment5_plan(
                source,
                project_commit=COMMIT,
                execution_profile=EXECUTION_PROFILE,
                resource_evidence_overrides=overrides,
            )


def test_create_only_generator_derives_all_paths_and_hashes(
    tmp_path: Path,
) -> None:
    source = _source()
    expected = _fresh_evidence_overrides(tmp_path / "evidence", source)
    roots = {
        resource_id: Path(record["controller_mapping"]["controller_path"]).parent
        for resource_id, record in expected["resources"].items()
    }

    generated = build_resource_evidence_overrides(
        source,
        project_commit=COMMIT,
        execution_profile=EXECUTION_PROFILE,
        controller_probe_roots=roots,
    )
    assert generated == expected

    source_path = tmp_path / "r8.yaml"
    output_path = tmp_path / "fresh-overrides.yaml"
    source_path.write_text(
        yaml.safe_dump(source, sort_keys=False),
        encoding="utf-8",
    )
    arguments = [
        "--r8-plan",
        str(source_path),
        "--project-commit",
        COMMIT,
        "--execution-profile",
        EXECUTION_PROFILE,
    ]
    for resource_id, root in sorted(roots.items()):
        arguments.extend(("--probe-root", f"{resource_id}={root}"))
    arguments.extend(("--output", str(output_path)))
    assert evidence_override_main(arguments) == 0
    assert yaml.safe_load(output_path.read_text(encoding="utf-8")) == expected
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        evidence_override_main(arguments)
