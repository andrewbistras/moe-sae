from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.tpu.build_experiment5_routed_ladder_plan import (
    main as routed_ladder_plan_main,
)
from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.checkpointing import (
    checkpoint_payload_uri,
    make_pointer,
    milestone_pointer_path,
)
from moe_sae.experiment5_campaign import parse_experiment5_campaign_plan
from moe_sae.direct_horizon_campaign import parse_direct_campaign_plan
from moe_sae.experiment5_plan_builder import (
    EXECUTION_PROFILE,
    build_experiment5_plan,
)
from moe_sae.experiment5_routed_ladder import PROTOCOL_ID
from moe_sae.experiment5_routed_ladder_campaign import (
    A256_REFERENCE_MANIFEST_SCHEMA_VERSION,
    EVALUATION_BUNDLE_COUNT,
    EVALUATION_TASK_COUNT,
    H_REFERENCE_MANIFEST_SCHEMA_VERSION,
    MILESTONES,
    OPERATOR_BUNDLE_MANIFEST_SCHEMA_VERSION,
    PANELS,
    SEEDS,
    TRAINING_TASK_COUNT,
    RoutedLadderCampaign,
    RoutedLadderCampaignPlanError,
    RoutedLadderCampaignStateError,
    _bundle_cell_completion,
    _evaluation_bundles,
    _evaluation_parent_pins,
    _operator_bundle_manifest,
    build_evaluation_bundle_command,
    build_evaluation_command,
    build_h_reference_manifest,
    build_routed_ladder_campaign_plan,
    build_train_command,
    inspect_evaluation_bundle_completion,
    inspect_evaluation_completion,
    parse_routed_ladder_campaign_plan,
)
from moe_sae.experiment5_routed_ladder_config import H_CONFIG_SHA256
from moe_sae.experiment5_routed_ladder_evaluation_worker import (
    BUNDLE_CONDITIONS,
    BUNDLE_RESULT_SCHEMA_VERSION,
    load_process_evidence,
    merge_process_evidence,
    seal_merged_result,
    seal_process_evidence,
)
from moe_sae.experiment5_routed_ladder_resources import (
    COHORT_KEYS,
    RESOURCE_POOL_SCHEMA_VERSION,
    seal_routed_ladder_resource_pool,
)

from test_experiment5_plan_builder import (
    _fresh_evidence_overrides,
    _source,
)


COMMIT = "a" * 40
H_TRAINING_COMMIT = "b" * 40
REPOSITORY = Path(__file__).resolve().parents[1]


def _evidence(path: Path, remote: str) -> dict[str, str]:
    payload = b"sealed-evidence\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "controller_path": str(path),
        "remote_path": remote,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _h_reference_manifest(tmp_path: Path) -> dict:
    cells = []
    for panel in PANELS:
        for seed in SEEDS:
            run_root = f"/immutable/core/{panel}/seed{seed}/runs/h"
            for milestone in MILESTONES:
                training = tmp_path / "h" / panel / f"seed{seed}" / "training.json"
                pointer = (
                    tmp_path
                    / "h"
                    / panel
                    / f"seed{seed}"
                    / "milestones"
                    / f"{milestone}.json"
                )
                cells.append(
                    {
                        "panel": panel,
                        "seed": seed,
                        "checkpoint_tokens": milestone,
                        "training_source_revision": H_TRAINING_COMMIT,
                        "h_config_remote": (
                            f"/remote/configs/gemma4_hybrid_{panel}_v1.yaml"
                        ),
                        "h_config_sha256": H_CONFIG_SHA256[panel],
                        "run_root_remote": run_root,
                        "checkpoint_payload_root_gcs": (
                            f"gs://immutable/core/{panel}/seed{seed}"
                        ),
                        "training_result": _evidence(
                            training,
                            f"{run_root}/training-result.json",
                        ),
                        "training_result_sha256": f"{seed + 1:064x}",
                        "checkpoint_pointer": _evidence(
                            pointer,
                            str(
                                PurePosixPath(run_root)
                                / "checkpoints"
                                / "milestones"
                                / f"target-tokens-{milestone:012d}.json"
                            ),
                        ),
                        "checkpoint_pointer_sha256": (
                            f"{milestone + seed + 10:064x}"
                        ),
                    }
                )
    body = {
        "schema_version": H_REFERENCE_MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "cells": cells,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _a256_reference_manifest(tmp_path: Path, source: dict) -> dict:
    plan = parse_direct_campaign_plan(source)
    task = next(
        task
        for task in plan.trajectories
        if task.scientific_model == "sae" and task.condition == "A"
    )
    resource = plan.resource(task.resource_id)
    training_revision = plan.environment_dict()["TRAINING_PROJECT_COMMIT"]
    run_root = str(
        PurePosixPath(task.control_root_remote) / "runs" / task.run_id
    )
    config = _evidence(
        tmp_path / "a256" / "sae.yaml",
        task.config_remote,
    )
    run_contract = _evidence(
        task.completion_path_controller.parent / "run-contract.json",
        f"{run_root}/run-contract.json",
    )
    training = _evidence(
        task.completion_path_controller,
        f"{run_root}/training-result.json",
    )
    cells = []
    for milestone in MILESTONES:
        cells.append(
            {
                "panel": "sae",
                "condition": "A256ReLU",
                "seed": 0,
                "checkpoint_tokens": milestone,
                "training_source_revision": training_revision,
                "a256_config": config,
                "a256_config_sha256": (
                    "d2bc2a5f77ac303398b4570bd174ebb1afc4d69b9f94d1d5f"
                    "601156b01ea8f50"
                ),
                "run_root_remote": run_root,
                "checkpoint_payload_root_gcs": task.checkpoint_root_gcs,
                "run_contract": run_contract,
                "run_contract_sha256": "9" * 64,
                "training_result": training,
                "training_result_sha256": "8" * 64,
                "checkpoint_pointer": _evidence(
                    (
                        task.completion_path_controller.parent
                        / "checkpoints"
                        / "milestones"
                        / f"target-tokens-{milestone:012d}.json"
                    ),
                    (
                        f"{run_root}/checkpoints/milestones/"
                        f"target-tokens-{milestone:012d}.json"
                    ),
                ),
                "checkpoint_pointer_sha256": f"{milestone + 20:064x}",
            }
        )
    body = {
        "schema_version": A256_REFERENCE_MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": (
            "gemma4-pilot-learning-horizon-v1-20260728"
        ),
        "source_plan_sha256": plan.plan_sha256,
        "historical_training_cost": {
            "resource_id": resource.resource_id,
            "accelerator_type": resource.accelerator_type,
            "chip_count": 16,
            "marginal_wall_seconds": 10.0,
            "chip_seconds": 160.0,
            "chip_seconds_scope": "training",
            "chip_seconds_excludes": ["compile", "checkpoint"],
        },
        "cells": cells,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _raw_plan(
    tmp_path: Path,
    *,
    extra_v5_resources: int = 0,
    wrong_a256_source_plan: bool = False,
    already_spent_usd: float = 0.0,
) -> dict:
    source = _source()
    source["working_directory"] = str(REPOSITORY)
    source["python_executable"] = sys.executable
    source.update(
        {
            "state_root": str(tmp_path / "direct-state"),
            "project_commit": "c" * 40,
            "max_parallel_train": 8,
            "max_parallel_eval": 2,
            "retry_policy": {
                "max_transient_attempts": 3,
                "initial_backoff_seconds": 0,
                "maximum_backoff_seconds": 0,
            },
        }
    )
    for resource in source["resources"]:
        resource["region"] = resource["zone"].rsplit("-", 1)[0]
    strict_trajectories = []
    for raw in source["trajectories"]:
        resource_id = raw["resource_id"]
        model = raw["scientific_model"]
        condition = resource_id.rsplit("-", 1)[1].upper()
        run_id = f"horizon10m-{model}-{condition.lower()}-seed0"
        completion = (
            tmp_path
            / "direct-results"
            / "horizon-10m"
            / model
            / "runs"
            / run_id
            / "training-result.json"
        )
        strict_trajectories.append(
            {
                "id": f"train-{model}-{condition.lower()}",
                "scientific_model": model,
                "condition": condition,
                "seed": 0,
                "resource_id": resource_id,
                "run_id": run_id,
                "config_remote": f"/remote/contracts/{model}.yaml",
                "phase": "learning_horizon_10m",
                "control_root_remote": f"/remote/horizon-10m/runs/{model}",
                "checkpoint_root_gcs": (
                    f"gs://historical/checkpoints/{model}"
                ),
                "completion_path_controller": str(completion),
                "environment": {},
            }
        )
    source["trajectories"] = strict_trajectories
    source["evaluations"] = [
        {
            "id": f"eval-{model}-{milestone}",
            "scientific_model": model,
            "milestone_tokens": milestone,
            "seed": 0,
            "resource_id": f"slice-{model}-a",
            "config_remote": f"/remote/contracts/{model}.yaml",
            "phase": "learning_horizon_10m",
            "control_root_remote": f"/remote/horizon-10m/runs/{model}",
            "checkpoint_root_gcs": (
                f"gs://historical/checkpoints/{model}"
            ),
            "result_root_remote": (
                f"/remote/horizon-10m/evaluation/{model}/{milestone}"
            ),
            "completion_path_controller": str(
                tmp_path
                / "direct-results"
                / "evaluation"
                / model
                / str(milestone)
                / "scientific-result.json"
            ),
            "run_roots_remote": {
                condition: (
                    f"/remote/horizon-10m/runs/{model}/runs/"
                    f"horizon10m-{model}-{condition.lower()}-seed0"
                )
                for condition in ("A", "R", "B", "F")
            },
            "environment": {},
        }
        for model in ("sae", "transcoder")
        for milestone in MILESTONES
    ]
    overrides = _fresh_evidence_overrides(tmp_path, source, project_commit=COMMIT)
    core = build_experiment5_plan(
        source,
        project_commit=COMMIT,
        execution_profile=EXECUTION_PROFILE,
        resource_evidence_overrides=overrides,
    )
    active_ids = {
        resource_id
        for task in (*core["trajectories"], *core["evaluations"])
        for resource_id in task["eligible_resource_ids"]
    }
    resources = [
        copy.deepcopy(row)
        for row in core["resources"]
        if row["id"] in active_ids
    ]
    controller_root = tmp_path / "controller-mount"
    remote_root = "/home/worker/moe-sae/artifacts/gemma4_v1/production"
    for row in resources:
        if row["evidence"]["mirror_seal"] is not None:
            row["evidence"]["mirror_seal"] = _evidence(
                controller_root / "mirrors" / row["id"] / "mirror-seal.json",
                f"{remote_root}/mirrors/{row['id']}/mirror-seal.json",
            )
    v5_template = next(
        row for row in resources if row["accelerator_type"] == "v5litepod-16"
    )
    for index in range(extra_v5_resources):
        clone = copy.deepcopy(v5_template)
        clone["id"] = f"slice-hot-v5-{index}"
        clone["tpu_name"] = f"gemma4-hot-v5-{index}"
        clone["evidence"] = {
            "controller_mapping": _evidence(
                controller_root
                / "hot"
                / clone["id"]
                / "controller-mapping.json",
                (
                    f"{remote_root}/hot/{clone['id']}/"
                    "controller-mapping.json"
                ),
            ),
            "cache_inventory": _evidence(
                controller_root
                / "hot"
                / clone["id"]
                / "cache-inventory.json",
                f"{remote_root}/hot/{clone['id']}/cache-inventory.json",
            ),
            "mirror_seal": _evidence(
                controller_root
                / "hot"
                / clone["id"]
                / "mirror-seal.json",
                f"{remote_root}/hot/{clone['id']}/mirror-seal.json",
            ),
        }
        resources.append(clone)
    by_id = {row["id"]: row for row in resources}
    v5 = next(
        resource_id
        for resource_id, row in by_id.items()
        if row["accelerator_type"] == "v5litepod-16"
    )
    v4 = sorted(
        resource_id
        for resource_id, row in by_id.items()
        if row["accelerator_type"] == "v4-32"
    )
    v5_cohort = [v5, *[f"slice-hot-v5-{index}" for index in range(extra_v5_resources)]]
    cohorts = {
        key: (v5_cohort if key.endswith("seed0") else v4)
        for key in COHORT_KEYS
    }
    planes = {}
    for resource_id, row in by_id.items():
        declared = row["environment"].get(
            "BULK_GCS_ROOT",
            source["environment"]["BULK_GCS_ROOT"],
        )
        planes[resource_id] = {
            "controller_artifact_root": str(controller_root),
            "remote_artifact_root": remote_root,
            "bulk_gcs_root": declared.removesuffix("/horizon-10m"),
        }
    pool = seal_routed_ladder_resource_pool(
        {
            "schema_version": RESOURCE_POOL_SCHEMA_VERSION,
            "project_commit": COMMIT,
            "resources": resources,
            "planes": planes,
            "cohorts": cohorts,
        }
    )
    a256_manifest = _a256_reference_manifest(tmp_path, source)
    if wrong_a256_source_plan:
        a256_manifest["source_plan_sha256"] = "f" * 64
        a256_manifest["manifest_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in a256_manifest.items()
                if key != "manifest_sha256"
            }
        )
    plan = build_routed_ladder_campaign_plan(
        source,
        project_commit=COMMIT,
        h_reference_manifest=_h_reference_manifest(tmp_path),
        a256_reference_manifest=a256_manifest,
        already_spent_usd=already_spent_usd,
        resource_pool=pool,
    )
    plan["state_root"] = str(
        tmp_path
        / "experiment5"
        / "extensions"
        / "routed-activation-ladder-v2"
        / "controller"
        / COMMIT[:12]
    )
    return plan


def _core_plan(tmp_path: Path):
    source = _source()
    source["working_directory"] = str(REPOSITORY)
    source["python_executable"] = sys.executable
    overrides = _fresh_evidence_overrides(tmp_path, source, project_commit=COMMIT)
    return parse_experiment5_campaign_plan(
        build_experiment5_plan(
            source,
            project_commit=COMMIT,
            execution_profile=EXECUTION_PROFILE,
            resource_evidence_overrides=overrides,
        )
    )


def _write_self_hashed(path: Path, body: dict, field: str) -> dict:
    value = {**body, field: canonical_sha256(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    return value


def test_h_inventory_builder_reads_and_hash_pins_exact_core_checkpoints(
    tmp_path: Path,
) -> None:
    core = _core_plan(tmp_path)
    for task in (
        task for task in core.trajectories if task.condition == "H"
    ):
        worker_plan = {
            "phase_name": "routed_shared_hybrid_10m",
            "condition": "H",
            "seed": task.seed,
            "total_tokens": 10_000_000,
            "planned_updates": 100,
            "warmup_updates": 1,
            "global_microbatch_tokens": 100_000,
            "microbatches_per_update": 1,
            "checkpoint_tokens": list(MILESTONES),
            "randomization_key": task.seed,
        }
        run_identity = {
            "plan": worker_plan,
            "run_identity_sha256": "d" * 64,
        }
        _write_self_hashed(
            task.completion_path_controller,
            {
                "schema_version": (
                    "gemma4-contribution-sae-training-result-v1"
                ),
                "status": "completed",
                "source_revision": core.project_commit,
                "tokens": 10_000_000,
                "run_identity": run_identity,
            },
            "result_sha256",
        )
        checkpoint_root = (
            task.completion_path_controller.parent / "checkpoints"
        )
        for milestone in MILESTONES:
            step = milestone // 100_000
            attempt = f"{milestone + task.seed:032x}"
            pointer = make_pointer(
                kind="latest",
                payload_uri=checkpoint_payload_uri(
                    task.checkpoint_root_gcs,
                    run_identity,
                    step=step,
                    tokens_seen=milestone,
                    attempt_id=attempt,
                ),
                step=step,
                tokens_seen=milestone,
                attempt_id=attempt,
                run_identity=run_identity,
            )
            pointer_path = milestone_pointer_path(checkpoint_root, milestone)
            pointer_path.parent.mkdir(parents=True, exist_ok=True)
            pointer_path.write_text(
                json.dumps(pointer, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="ascii",
            )

    manifest = build_h_reference_manifest(core)

    assert len(manifest["cells"]) == 18
    assert manifest["manifest_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_sha256"
        }
    )
    assert {
        (row["panel"], row["seed"], row["checkpoint_tokens"])
        for row in manifest["cells"]
    } == {
        (panel, seed, milestone)
        for panel in PANELS
        for seed in SEEDS
        for milestone in MILESTONES
    }


def test_exact_training_and_evaluation_matrix_isolated_from_core(
    tmp_path: Path,
) -> None:
    plan = parse_routed_ladder_campaign_plan(_raw_plan(tmp_path))

    assert len(plan.trajectories) == TRAINING_TASK_COUNT == 17
    assert len(plan.evaluations) == EVALUATION_TASK_COUNT == 72
    assert len(plan.h_references) == 18
    assert len(plan.a256_references) == 3
    assert {
        (task.panel, task.condition, task.seed) for task in plan.trajectories
    } == {
        (panel, condition, seed)
        for panel in PANELS
        for condition in ("A256ReLU", "A512ReLU", "A512Softplus")
        for seed in SEEDS
        if not (
            panel == "sae"
            and condition == "A256ReLU"
            and seed == 0
        )
    }
    assert all(task.condition != "H" for task in plan.trajectories)
    assert {task.task_id for task in plan.trajectories} == {
        f"train-{panel}-{slug}-seed{seed}"
        for panel in PANELS
        for slug in ("a256-relu", "a512-relu", "a512-softplus")
        for seed in SEEDS
        if not (panel == "sae" and slug == "a256-relu" and seed == 0)
    }
    assert {task.run_id for task in plan.trajectories} == {
        f"experiment5-activation-ladder-{panel}-{slug}-seed{seed}"
        for panel in PANELS
        for slug in ("a256-relu", "a512-relu", "a512-softplus")
        for seed in SEEDS
        if not (panel == "sae" and slug == "a256-relu" and seed == 0)
    }
    assert {
        (task.panel, task.condition, task.seed, task.milestone_tokens)
        for task in plan.evaluations
    } == {
        (panel, condition, seed, milestone)
        for panel in PANELS
        for condition in (
            "A256ReLU",
            "A512ReLU",
            "A512Softplus",
            "H",
        )
        for seed in SEEDS
        for milestone in MILESTONES
    }
    assert {task.task_id for task in plan.evaluations} == {
        f"eval-{panel}-{slug}-seed{seed}-{milestone}"
        for panel in PANELS
        for slug in ("a256-relu", "a512-relu", "a512-softplus", "h")
        for seed in SEEDS
        for milestone in MILESTONES
    }
    assert all(
        "/experiment5/extensions/routed-activation-ladder-v2/"
        in task.result_root_remote
        for task in plan.evaluations
    )


def test_six_resident_bundles_preserve_exact_seventy_two_cell_plan(
    tmp_path: Path,
) -> None:
    plan = parse_routed_ladder_campaign_plan(_raw_plan(tmp_path))
    bundles = _evaluation_bundles(plan)

    assert len(bundles) == EVALUATION_BUNDLE_COUNT == 6
    assert {(bundle.panel, bundle.seed) for bundle in bundles} == {
        (panel, seed) for panel in PANELS for seed in SEEDS
    }
    for bundle in bundles:
        assert len(bundle.cells) == 12
        assert [
            (cell.condition, cell.milestone_tokens)
            for cell in bundle.cells
        ] == [
            (condition, milestone)
            for condition in BUNDLE_CONDITIONS
            for milestone in MILESTONES
        ]
        assert {
            cell.eligible_resource_ids for cell in bundle.cells
        } == {bundle.eligible_resource_ids}
        assert bundle.completion_path_controller == (
            bundle.result_root_controller / "bundle-result.json"
        )

    for task in plan.trajectories:
        _seal_training_completion(plan, task)
    manifest = _operator_bundle_manifest(plan, bundles[0])
    unsigned = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    assert set(manifest) == {
        "schema_version",
        "protocol_id",
        "panel",
        "seed",
        "source_revision",
        "config_remote",
        "bundle_result_root_remote",
        "bundle_result_root_controller",
        "cells",
        "manifest_sha256",
    }
    assert (
        manifest["schema_version"]
        == OPERATOR_BUNDLE_MANIFEST_SCHEMA_VERSION
    )
    assert manifest["manifest_sha256"] == canonical_sha256(unsigned)
    assert len(manifest["cells"]) == 12
    a256 = manifest["cells"][0]
    assert a256["reference_kind"] == "a256"
    assert a256["reference_config_remote"] is not None
    assert a256["reference_run_contract_sha256"] == "9" * 64
    assert a256["reference_run_contract_file_sha256"] is not None
    h_rows = [
        row for row in manifest["cells"] if row["reference_kind"] == "h"
    ]
    assert len(h_rows) == 3
    assert all(row["reference_config_remote"] is not None for row in h_rows)
    assert all(
        row["reference_run_contract_sha256"] is None
        and row["reference_run_contract_file_sha256"] is None
        for row in h_rows
    )


def _seal_training_completion(plan, task) -> None:
    worker_plan = {
        "phase_name": (
            "experiment5_routed_activation_ladder_v2_10m"
        ),
        "condition": task.condition,
        "seed": task.seed,
        "total_tokens": 10_000_000,
        "planned_updates": 611,
        "global_microbatch_tokens": 4096,
        "microbatches_per_update": 4,
    }
    run_identity_body = {"plan": worker_plan}
    run_identity = {
        **run_identity_body,
        "run_identity_sha256": canonical_sha256(run_identity_body),
    }
    _write_self_hashed(
        task.completion_path_controller,
        {
            "schema_version": (
                "gemma4-contribution-sae-training-result-v1"
            ),
            "status": "completed",
            "source_revision": plan.project_commit,
            "tokens": 10_000_000,
            "run_identity": run_identity,
            "marginal_wall_seconds": 10.0,
            "chip_seconds": 160.0,
            "chip_seconds_scope": "training",
            "chip_seconds_excludes": ["compile", "checkpoint"],
        },
        "result_sha256",
    )
    config_path = (
        plan.working_directory
        / "configs"
        / (
            "gemma4_experiment5_routed_ladder_sae_v2.yaml"
            if task.panel == "sae"
            else "gemma4_experiment5_routed_ladder_transcoder_v2.yaml"
        )
    )
    _write_self_hashed(
        task.completion_path_controller.parent / "run-contract.json",
        {
            "schema_version": "test-run-contract-v1",
            "project_commit": plan.project_commit,
            "config_file_sha256": hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
            "phase": (
                "experiment5_routed_activation_ladder_v2_10m"
            ),
            "scientific_model": task.panel,
            "condition": task.condition,
            "seed": task.seed,
            "checkpoint_payload_root": task.checkpoint_root_gcs,
        },
        "run_contract_sha256",
    )
    for milestone in MILESTONES:
        step = (milestone + 16_383) // 16_384
        tokens = min(step * 16_384, 10_000_000)
        pointer = make_pointer(
            kind="latest",
            payload_uri=checkpoint_payload_uri(
                task.checkpoint_root_gcs,
                run_identity,
                step=step,
                tokens_seen=tokens,
                attempt_id=f"{milestone:032x}",
            ),
            step=step,
            tokens_seen=tokens,
            attempt_id=f"{milestone:032x}",
            run_identity=run_identity,
        )
        pointer_path = milestone_pointer_path(
            task.completion_path_controller.parent / "checkpoints",
            milestone,
        )
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        pointer_path.write_text(
            json.dumps(pointer, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="ascii",
        )


def test_bundle_launch_waits_for_all_current_v2_parents(
    tmp_path: Path,
) -> None:
    plan = parse_routed_ladder_campaign_plan(_raw_plan(tmp_path))
    bundle = next(
        bundle
        for bundle in _evaluation_bundles(plan)
        if bundle.panel == "sae" and bundle.seed == 0
    )
    with pytest.raises(
        RoutedLadderCampaignStateError,
        match="not runnable",
    ):
        build_evaluation_bundle_command(plan, bundle, "e5rl-attempt")

    for task in plan.trajectories:
        if task.panel == bundle.panel and task.seed == bundle.seed:
            _seal_training_completion(plan, task)
    command = build_evaluation_bundle_command(
        plan,
        bundle,
        "e5rl-attempt",
    )
    assert "--bundle-manifest" in command
    assert "--attempt-id" in command
    manifest_path = Path(
        command[command.index("--bundle-manifest") + 1]
    )
    assert manifest_path.is_file()
    assert json.loads(manifest_path.read_text()) == _operator_bundle_manifest(
        plan,
        bundle,
    )
    campaign = RoutedLadderCampaign(plan, verify_checkout=False)
    policy = plan.cost_policy
    assert campaign._baseline_attempt_reserve_usd() == pytest.approx(
        TRAINING_TASK_COUNT * policy.training_attempt_reserve_usd
        + EVALUATION_BUNDLE_COUNT * policy.evaluation_attempt_reserve_usd
    )


def test_five_resource_pool_is_consumed_with_one_owner_per_task(
    tmp_path: Path,
) -> None:
    plan = parse_routed_ladder_campaign_plan(
        _raw_plan(tmp_path, extra_v5_resources=2)
    )

    assert len(plan.resources) == 5
    assert plan.max_parallel_train == 5
    assert plan.max_parallel_eval == len(
        {
            task.eligible_resource_ids[0]
            for task in plan.evaluations
        }
    )
    assert all(
        len(task.eligible_resource_ids) == 1
        for task in (*plan.trajectories, *plan.evaluations)
    )
    assert {
        owner
        for task in (*plan.trajectories, *plan.evaluations)
        for owner in task.eligible_resource_ids
    } == {resource.resource_id for resource in plan.resources}
    seed_zero_owners = {
        task.eligible_resource_ids[0]
        for task in (*plan.trajectories, *plan.evaluations)
        if task.seed == 0
    }
    assert seed_zero_owners == {
        "slice-sae-a",
        "slice-hot-v5-0",
        "slice-hot-v5-1",
    }


def test_historical_spend_is_preserved_but_never_blocks_v2(
    tmp_path: Path,
) -> None:
    plan = parse_routed_ladder_campaign_plan(
        _raw_plan(tmp_path, already_spent_usd=40.12)
    )
    campaign = RoutedLadderCampaign(plan, verify_checkout=False)
    campaign._bind_campaign_clock()
    task = plan.trajectories[0]
    resource = plan.resource(task.eligible_resource_ids[0])

    reservation = campaign._reserve_attempt_cost(
        task,
        kind="training",
        attempt_number=1,
        resource=resource,
    )

    assert plan.cost_policy.already_spent_usd == 40.12
    assert (
        reservation["enforcement"]
        == "measurement_only_grant_funded_v2"
    )
    assert campaign.cost_projection()["projected_committed_usd"] > 45.0
    campaign._check_campaign_watchdog()


def test_resource_pool_hash_and_cross_plane_cohort_fail_closed(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path)
    changed = copy.deepcopy(raw)
    changed["resource_pool"]["cohorts"]["sae.seed0"].append(
        "slice-sae-b"
    )
    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="self-hash",
    ):
        parse_routed_ladder_campaign_plan(changed)

    changed["resource_pool"]["pool_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed["resource_pool"].items()
            if key != "pool_sha256"
        }
    )
    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="crosses an artifact or accelerator plane",
    ):
        parse_routed_ladder_campaign_plan(changed)


def test_reference_trajectory_coherence_and_source_plan_fail_closed(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path)
    changed = copy.deepcopy(raw)
    first = changed["h_reference_manifest"]["cells"][0]
    second = changed["h_reference_manifest"]["cells"][1]
    second["training_source_revision"] = "e" * 40
    changed["h_reference_manifest"]["manifest_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed["h_reference_manifest"].items()
            if key != "manifest_sha256"
        }
    )
    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="do not share one immutable training trajectory",
    ):
        parse_routed_ladder_campaign_plan(changed)

    changed = copy.deepcopy(raw)
    first = changed["h_reference_manifest"]["cells"][0]
    second = changed["h_reference_manifest"]["cells"][1]
    second["checkpoint_pointer"] = copy.deepcopy(
        first["checkpoint_pointer"]
    )
    second["checkpoint_pointer_sha256"] = first[
        "checkpoint_pointer_sha256"
    ]
    changed["h_reference_manifest"]["manifest_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed["h_reference_manifest"].items()
            if key != "manifest_sha256"
        }
    )
    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="milestone pointers are not unique",
    ):
        parse_routed_ladder_campaign_plan(changed)

    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="different Experiment-4 plan",
    ):
        _raw_plan(tmp_path / "wrong-source", wrong_a256_source_plan=True)


def test_single_owner_and_work_conserving_parallelism_are_mandatory(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path, extra_v5_resources=2)
    multiple_owners = copy.deepcopy(raw)
    task = next(
        row
        for row in multiple_owners["trajectories"]
        if row["seed"] == 0
    )
    second = next(
        resource_id
        for resource_id in multiple_owners["resource_pool"]["cohorts"][
            "sae.seed0"
        ]
        if resource_id not in task["eligible_resource_ids"]
    )
    task["eligible_resource_ids"].append(second)
    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="one immutable resource owner",
    ):
        parse_routed_ladder_campaign_plan(multiple_owners)

    throttled = copy.deepcopy(raw)
    throttled["max_parallel_train"] = 3
    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="consume every training owner",
    ):
        parse_routed_ladder_campaign_plan(throttled)


def test_legacy_ready_three_plan_cli_fails_with_migration_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        routed_ladder_plan_main(
            [
                "--r8-plan",
                "/tmp/r8.yaml",
                "--project-commit",
                COMMIT,
                "--execution-profile",
                "ready-three-v1",
                "--resource-evidence-overrides",
                "/tmp/legacy.yaml",
                "--h-reference-manifest",
                "/tmp/h.yaml",
                "--already-spent-usd",
                "0",
                "--output",
                "/tmp/plan.yaml",
            ]
        )

    assert raised.value.code == 2
    stderr = capsys.readouterr().err
    assert "ready-three" in stderr
    assert "--resource-pool is required" in stderr


def test_extension_namespace_and_source_revision_fail_closed(
    tmp_path: Path,
) -> None:
    raw = _raw_plan(tmp_path)
    escaped = copy.deepcopy(raw)
    escaped["evaluations"][0]["result_root_remote"] = "/tmp/escaped"
    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="namespace",
    ):
        parse_routed_ladder_campaign_plan(escaped)

    substituted = copy.deepcopy(raw)
    h_task = next(
        row for row in substituted["evaluations"] if row["condition"] == "H"
    )
    h_task["training_source_revision"] = COMMIT
    with pytest.raises(
        RoutedLadderCampaignPlanError,
        match="immutable H checkpoint",
    ):
        parse_routed_ladder_campaign_plan(substituted)


def test_commands_distinguish_new_a_training_from_h_rescoring(
    tmp_path: Path,
) -> None:
    plan = parse_routed_ladder_campaign_plan(_raw_plan(tmp_path))
    training = next(
        task
        for task in plan.trajectories
        if task.panel == "sae"
        and task.condition == "A512ReLU"
        and task.seed == 0
    )
    train_command = build_train_command(plan, training, "e5rl-attempt")
    assert "--condition" in train_command
    assert (
        train_command[train_command.index("--condition") + 1]
        == "A512ReLU"
    )
    assert "--attempt-id" in train_command

    a_eval = next(
        task
        for task in plan.evaluations
        if task.panel == "sae"
        and task.condition == "A512Softplus"
        and task.seed == 1
        and task.milestone_tokens == 5_000_000
    )
    a_command = build_evaluation_command(plan, a_eval)
    assert "--h-config" not in a_command
    assert (
        a_command[a_command.index("--training-source-revision") + 1]
        == COMMIT
    )

    h_eval = next(
        task
        for task in plan.evaluations
        if task.panel == "transcoder"
        and task.condition == "H"
        and task.seed == 2
        and task.milestone_tokens == 10_000_000
    )
    h_command = build_evaluation_command(plan, h_eval)
    assert h_command[h_command.index("--h-config") + 1].endswith(
        "gemma4_hybrid_transcoder_v1.yaml"
    )
    assert (
        h_command[h_command.index("--training-source-revision") + 1]
        == H_TRAINING_COMMIT
    )


def _seal_a_completion(plan, task) -> None:
    parent_pins = _evaluation_parent_pins(plan, task)
    rows = []
    for process in range(2):
        lane = task.result_root_controller / f"lane-{process}"
        values = np.asarray([2.0, 3.0], dtype=np.float64)
        seal_process_evidence(
            lane,
            panel=task.panel,
            condition=task.condition,
            seed=task.seed,
            checkpoint_tokens=task.milestone_tokens,
            checkpoint_actual_tokens=task.milestone_tokens,
            process_index=process,
            process_count=2,
            source_revision=plan.project_commit,
            training_source_revision=task.training_source_revision,
            config_sha256=task.config_sha256,
            config_file_sha256="1" * 64,
            h_config_sha256=None,
            h_config_file_sha256=None,
            training_result_sha256=(
                parent_pins["parent_training_result_sha256"]
            ),
            training_contract_sha256=(
                parent_pins["parent_run_contract_sha256"] or "3" * 64
            ),
            checkpoint_sha256=(
                parent_pins["parent_checkpoint_pointer_sha256"]
            ),
            parent_training_result_sha256=(
                parent_pins["parent_training_result_sha256"]
            ),
            parent_training_result_file_sha256=(
                parent_pins["parent_training_result_file_sha256"]
            ),
            parent_run_contract_sha256=(
                parent_pins["parent_run_contract_sha256"]
            ),
            parent_run_contract_file_sha256=(
                parent_pins["parent_run_contract_file_sha256"]
            ),
            parent_checkpoint_pointer_sha256=(
                parent_pins["parent_checkpoint_pointer_sha256"]
            ),
            parent_checkpoint_pointer_file_sha256=(
                parent_pins["parent_checkpoint_pointer_file_sha256"]
            ),
            target_manifest_sha256="5" * 64,
            source_manifest_sha256=None,
            recomposition_sidecar_sha256="6" * 64,
            centering_receipt_sha256="7" * 64,
            input_receipt_sha256=f"{process + 8:064x}",
            learned_input_receipt_sha256=None,
            evaluation_wall_seconds=1.0,
            sequence_id=np.asarray([process, process], dtype=np.int64),
            position=np.asarray([0, 1], dtype=np.uint16),
            scores=type(
                "Scores",
                (),
                {
                    "valid_mask": np.ones((2,), dtype=np.bool_),
                    "component_sse": values,
                    "component_sst": np.asarray([10.0, 12.0]),
                    "routed_sse": values - 0.5,
                    "routed_sst": np.asarray([9.0, 11.0]),
                    "oracle_shared_sse": values - 1.0,
                    "oracle_shared_sst": np.asarray([8.0, 10.0]),
                },
            )(),
        )
        rows.append(load_process_evidence(lane / "process-evidence.json"))
    summary, tensors = merge_process_evidence(rows)
    seal_merged_result(
        task.result_root_controller,
        summary=summary,
        tensors=tensors,
        expected_tokens=4,
    )


def test_evaluation_completion_binds_result_and_score_object(
    tmp_path: Path,
) -> None:
    plan = parse_routed_ladder_campaign_plan(_raw_plan(tmp_path))
    task = next(
        task
        for task in plan.evaluations
        if task.panel == "sae"
        and task.condition == "A512ReLU"
        and task.seed == 0
        and task.milestone_tokens == 2_000_000
    )
    parent = next(
        row
        for row in plan.trajectories
        if (row.panel, row.condition, row.seed)
        == (task.panel, task.condition, task.seed)
    )
    _seal_training_completion(plan, parent)
    _seal_a_completion(plan, task)

    completion = inspect_evaluation_completion(plan, task)
    assert completion is not None
    assert completion.path == str(task.completion_path_controller)
    assert completion.companion_path is not None

    score_path = Path(completion.companion_path)
    payload = bytearray(score_path.read_bytes())
    payload[-1] ^= 1
    score_path.write_bytes(payload)
    with pytest.raises(
        Exception,
        match="role changed",
    ):
        inspect_evaluation_completion(plan, task)


def _seal_report_cell(plan, task) -> None:
    identity = (task.panel, task.seed, task.milestone_tokens)
    h_parent = (
        {row.identity: row for row in plan.h_references}[identity]
        if task.condition == "H"
        else None
    )
    a256_parent = (
        {
            row.checkpoint_tokens: row for row in plan.a256_references
        }[task.milestone_tokens]
        if (
            task.panel == "sae"
            and task.condition == "A256ReLU"
            and task.seed == 0
        )
        else None
    )
    scale = {
        "A256ReLU": 1.2 + 0.05 * task.seed,
        "A512ReLU": 1.0 + 0.05 * task.seed,
        "A512Softplus": 0.8 + 0.05 * task.seed,
        "H": 0.7 + 0.05 * task.seed,
    }[task.condition]
    parent_pins = _evaluation_parent_pins(plan, task)
    rows = []
    for process in range(2):
        lane = task.result_root_controller / ".lanes" / f"process-{process}"
        learned = task.condition == "H"
        scores = SimpleNamespace(
            valid_mask=np.ones((2,), dtype=np.bool_),
            component_sse=np.asarray([2.0, 3.0]) * scale,
            component_sst=np.asarray([10.0, 12.0]),
            routed_sse=np.asarray([1.5, 2.5]) * scale,
            routed_sst=np.asarray([9.0, 11.0]),
            oracle_shared_sse=np.asarray([1.0, 2.0]) * scale,
            oracle_shared_sst=np.asarray([8.0, 10.0]),
            learned_full_ffn_sse=(
                np.asarray([1.1, 2.1]) * scale if learned else None
            ),
            learned_full_ffn_sst=(
                np.asarray([8.0, 10.0]) if learned else None
            ),
        )
        seal_process_evidence(
            lane,
            panel=task.panel,
            condition=task.condition,
            seed=task.seed,
            checkpoint_tokens=task.milestone_tokens,
            checkpoint_actual_tokens=task.milestone_tokens,
            process_index=process,
            process_count=2,
            source_revision=plan.project_commit,
            training_source_revision=task.training_source_revision,
            config_sha256=task.config_sha256,
            config_file_sha256="1" * 64,
            h_config_sha256=task.h_config_sha256,
            h_config_file_sha256=("2" * 64 if learned else None),
            training_result_sha256=(
                parent_pins["parent_training_result_sha256"]
            ),
            training_contract_sha256=(
                parent_pins["parent_run_contract_sha256"]
                or "4" * 64
            ),
            checkpoint_sha256=(
                parent_pins["parent_checkpoint_pointer_sha256"]
            ),
            parent_training_result_sha256=(
                parent_pins["parent_training_result_sha256"]
            ),
            parent_training_result_file_sha256=(
                parent_pins["parent_training_result_file_sha256"]
            ),
            parent_run_contract_sha256=(
                parent_pins["parent_run_contract_sha256"]
            ),
            parent_run_contract_file_sha256=(
                parent_pins["parent_run_contract_file_sha256"]
            ),
            parent_checkpoint_pointer_sha256=(
                parent_pins["parent_checkpoint_pointer_sha256"]
            ),
            parent_checkpoint_pointer_file_sha256=(
                parent_pins["parent_checkpoint_pointer_file_sha256"]
            ),
            target_manifest_sha256=(
                "6" * 64 if task.panel == "sae" else "7" * 64
            ),
            source_manifest_sha256=(
                None if task.panel == "sae" else "8" * 64
            ),
            recomposition_sidecar_sha256=(
                "9" * 64 if task.panel == "sae" else "a" * 64
            ),
            centering_receipt_sha256="b" * 64,
            input_receipt_sha256=f"{process + 12:064x}",
            learned_input_receipt_sha256=(
                f"{process + 14:064x}" if learned else None
            ),
            evaluation_wall_seconds=1.0,
            sequence_id=np.asarray([process, process], dtype=np.int64),
            position=np.asarray([0, 1], dtype=np.uint16),
            scores=scores,
        )
        rows.append(load_process_evidence(lane / "process-evidence.json"))
    summary, tensors = merge_process_evidence(rows)
    seal_merged_result(
        task.result_root_controller,
        summary=summary,
        tensors=tensors,
        expected_tokens=4,
    )


def test_resident_bundle_completion_revalidates_all_twelve_cells(
    tmp_path: Path,
) -> None:
    plan = parse_routed_ladder_campaign_plan(_raw_plan(tmp_path))
    bundle = next(
        bundle
        for bundle in _evaluation_bundles(plan)
        if bundle.panel == "transcoder" and bundle.seed == 1
    )
    for task in plan.trajectories:
        if task.panel == bundle.panel and task.seed == bundle.seed:
            _seal_training_completion(plan, task)
    for task in bundle.cells:
        _seal_report_cell(plan, task)
    manifest = _operator_bundle_manifest(plan, bundle)
    _write_self_hashed(
        bundle.completion_path_controller,
        {
            "schema_version": BUNDLE_RESULT_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "bundle_evaluated",
            "panel": bundle.panel,
            "seed": bundle.seed,
            "source_revision": plan.project_commit,
            "operator_manifest_sha256": manifest["manifest_sha256"],
            "bundle_spec_sha256": "e" * 64,
            "timing": {
                "accelerator_type": plan.resource(
                    bundle.eligible_resource_ids[0]
                ).accelerator_type,
                "chip_count": 16,
                "successful_attempt_wall_seconds": 2.0,
                "successful_attempt_chip_seconds": 32.0,
                "timing_scope": (
                    "successful_resident_worker_after_cli_parse_through_"
                    "cell_completion"
                ),
            },
            "cell_results": [
                _bundle_cell_completion(plan, task)
                for task in bundle.cells
            ],
        },
        "result_sha256",
    )

    completion = inspect_evaluation_bundle_completion(plan, bundle)
    assert completion is not None
    assert completion.path == str(bundle.completion_path_controller)
    assert len(
        [
            inspect_evaluation_completion(plan, task)
            for task in bundle.cells
        ]
    ) == 12

    result = json.loads(bundle.completion_path_controller.read_text())
    result["cell_results"] = list(reversed(result["cell_results"]))
    unsigned = {
        key: value
        for key, value in result.items()
        if key != "result_sha256"
    }
    result["result_sha256"] = canonical_sha256(unsigned)
    bundle.completion_path_controller.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(
        RoutedLadderCampaignStateError,
        match="completion changed",
    ):
        inspect_evaluation_bundle_completion(plan, bundle)


def test_report_seal_is_actual_validated_json_and_markdown(
    tmp_path: Path,
) -> None:
    plan = parse_routed_ladder_campaign_plan(_raw_plan(tmp_path))
    for task in plan.trajectories:
        _seal_training_completion(plan, task)
    completions = {}
    for task in plan.evaluations:
        _seal_report_cell(plan, task)
        completion = inspect_evaluation_completion(plan, task)
        assert completion is not None
        completions[task.task_id] = completion
    for bundle in _evaluation_bundles(plan):
        manifest = _operator_bundle_manifest(plan, bundle)
        resource = plan.resource(bundle.eligible_resource_ids[0])
        _write_self_hashed(
            bundle.completion_path_controller,
            {
                "schema_version": BUNDLE_RESULT_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": "bundle_evaluated",
                "panel": bundle.panel,
                "seed": bundle.seed,
                "source_revision": plan.project_commit,
                "operator_manifest_sha256": manifest["manifest_sha256"],
                "bundle_spec_sha256": "e" * 64,
                "timing": {
                    "accelerator_type": resource.accelerator_type,
                    "chip_count": 16,
                    "successful_attempt_wall_seconds": 2.0,
                    "successful_attempt_chip_seconds": 32.0,
                    "timing_scope": (
                        "successful_resident_worker_after_cli_parse_through_"
                        "cell_completion"
                    ),
                },
                "cell_results": [
                    _bundle_cell_completion(plan, task)
                    for task in bundle.cells
                ],
            },
            "result_sha256",
        )
    barrier_body = {
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.plan_sha256,
        "status": "evaluation_completed",
        "evaluations": {
            task_id: completions[task_id].to_dict()
            for task_id in sorted(completions)
        },
        "record_sha256": "f" * 64,
    }

    campaign = RoutedLadderCampaign(
        plan,
        verify_checkout=False,
    )
    campaign._bind_campaign_clock()
    clock = json.loads(
        (plan.state_root / "campaign-clock.json").read_text()
    )
    cost = campaign._seal_cost_projection(
        completed_at=clock["started_at"],
    )
    sealed = campaign._seal_scientific_report(
        evaluation_barrier=barrier_body,
        cost_projection=cost,
    )

    assert sealed["status"] == "report_sealed"
    report = json.loads(Path(sealed["report_path"]).read_text())
    assert report["report_sha256"] == sealed["report_sha256"]
    assert len(report["cohorts"]) == 18
    assert (
        len(report["across_seed_a512_softplus_minus_a512_relu"])
        == 6
    )
    markdown = Path(sealed["markdown_path"]).read_text()
    assert markdown.startswith("# Experiment 5 routed-activation ladder")
    assert "## Across-seed verdict" in markdown
    assert "A512Softplus−A512ReLU" in markdown
    assert "positive seeds" in markdown
    h_task = next(task for task in plan.evaluations if task.condition == "H")
    h_completion = completions[h_task.task_id]
    h_cohort = next(
        row
        for row in report["cohorts"]
        if row["identity"]
        == {
            "panel": h_task.panel,
            "seed": h_task.seed,
            "checkpoint_tokens": h_task.milestone_tokens,
        }
    )
    assert (
        h_cohort["conditions"]["H"]["scientific_result"]["sha256"]
        == h_completion.sha256
    )
