from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.checkpointing import (
    checkpoint_payload_uri,
    make_pointer,
    milestone_pointer_path,
)
from moe_sae.experiment5_campaign import parse_experiment5_campaign_plan
from moe_sae.experiment5_plan_builder import (
    EXECUTION_PROFILE,
    build_experiment5_plan,
)
from moe_sae.experiment5_routed_ladder import PROTOCOL_ID
from moe_sae.experiment5_routed_ladder_campaign import (
    EVALUATION_TASK_COUNT,
    H_REFERENCE_MANIFEST_SCHEMA_VERSION,
    MILESTONES,
    PANELS,
    SEEDS,
    TRAINING_TASK_COUNT,
    RoutedLadderCampaign,
    RoutedLadderCampaignPlanError,
    build_evaluation_command,
    build_h_reference_manifest,
    build_routed_ladder_campaign_plan,
    build_train_command,
    inspect_evaluation_completion,
    parse_routed_ladder_campaign_plan,
)
from moe_sae.experiment5_routed_ladder_config import H_CONFIG_SHA256
from moe_sae.experiment5_routed_ladder_evaluation_worker import (
    load_process_evidence,
    merge_process_evidence,
    seal_merged_result,
    seal_process_evidence,
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


def _raw_plan(tmp_path: Path) -> dict:
    source = _source()
    source["working_directory"] = str(REPOSITORY)
    source["python_executable"] = sys.executable
    overrides = _fresh_evidence_overrides(tmp_path, source, project_commit=COMMIT)
    plan = build_routed_ladder_campaign_plan(
        source,
        project_commit=COMMIT,
        execution_profile=EXECUTION_PROFILE,
        h_reference_manifest=_h_reference_manifest(tmp_path),
        already_spent_usd=0.0,
        resource_evidence_overrides=overrides,
    )
    plan["state_root"] = str(
        tmp_path
        / "experiment5"
        / "extensions"
        / "routed-activation-ladder-v1"
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

    assert len(plan.trajectories) == TRAINING_TASK_COUNT == 12
    assert len(plan.evaluations) == EVALUATION_TASK_COUNT == 54
    assert len(plan.h_references) == 18
    assert {
        (task.panel, task.condition, task.seed) for task in plan.trajectories
    } == {
        (panel, condition, seed)
        for panel in PANELS
        for condition in ("A512ReLU", "A512Softplus")
        for seed in SEEDS
    }
    assert all(task.condition != "H" for task in plan.trajectories)
    assert {task.task_id for task in plan.trajectories} == {
        f"train-{panel}-{slug}-seed{seed}"
        for panel in PANELS
        for slug in ("a512-relu", "a512-softplus")
        for seed in SEEDS
    }
    assert {task.run_id for task in plan.trajectories} == {
        f"experiment5-activation-ladder-{panel}-{slug}-seed{seed}"
        for panel in PANELS
        for slug in ("a512-relu", "a512-softplus")
        for seed in SEEDS
    }
    assert {
        (task.panel, task.condition, task.seed, task.milestone_tokens)
        for task in plan.evaluations
    } == {
        (panel, condition, seed, milestone)
        for panel in PANELS
        for condition in ("A512ReLU", "A512Softplus", "H")
        for seed in SEEDS
        for milestone in MILESTONES
    }
    assert {task.task_id for task in plan.evaluations} == {
        f"eval-{panel}-{slug}-seed{seed}-{milestone}"
        for panel in PANELS
        for slug in ("a512-relu", "a512-softplus", "h")
        for seed in SEEDS
        for milestone in MILESTONES
    }
    assert all(
        "/experiment5/extensions/routed-activation-ladder-v1/"
        in task.result_root_remote
        for task in plan.evaluations
    )


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
            training_result_sha256="2" * 64,
            training_contract_sha256="3" * 64,
            checkpoint_sha256="4" * 64,
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
    scale = {
        "A512ReLU": 1.0 + 0.05 * task.seed,
        "A512Softplus": 0.8 + 0.05 * task.seed,
        "H": 0.7 + 0.05 * task.seed,
    }[task.condition]
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
                h_parent.training_result_sha256 if h_parent else "3" * 64
            ),
            training_contract_sha256="4" * 64,
            checkpoint_sha256=(
                h_parent.checkpoint_pointer_sha256 if h_parent else "5" * 64
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


def test_report_seal_is_actual_validated_json_and_markdown(
    tmp_path: Path,
) -> None:
    plan = parse_routed_ladder_campaign_plan(_raw_plan(tmp_path))
    completions = {}
    for task in plan.evaluations:
        _seal_report_cell(plan, task)
        completion = inspect_evaluation_completion(plan, task)
        assert completion is not None
        completions[task.task_id] = completion
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

    sealed = RoutedLadderCampaign(
        plan,
        verify_checkout=False,
    )._seal_scientific_report(evaluation_barrier=barrier_body)

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
