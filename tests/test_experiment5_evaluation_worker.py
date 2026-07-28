from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.contribution_training_worker import TrainingWorkerPlan
from moe_sae.experiment5_config import PHASE_NAME, load_experiment5_config
from moe_sae.experiment5_distributed import experiment5_physical_layout
from moe_sae.experiment5_evaluation_worker import (
    Experiment5EvaluationWorkerError,
    _existing_completion,
    _gather_physical_firing_mask,
    _mean_sha256,
    _milestone_routed_counts,
    _parser,
    _require_source_arguments,
    _training_receipt,
    _validate_training_context,
)
from moe_sae.experiment5_hybrid import production_experiment5_layout


def _config(panel: str = "sae"):
    return load_experiment5_config(
        f"configs/gemma4_hybrid_{panel}_v1.yaml",
        expected_panel=panel,
    )


def _plan(condition: str = "H"):
    config = _config()
    return TrainingWorkerPlan.from_config(
        config,
        phase=PHASE_NAME,
        condition=condition,
        seed=0,
    )


def _base_argv(panel: str = "sae") -> list[str]:
    return [
        "--panel",
        panel,
        "--config",
        f"/tmp/{panel}.yaml",
        "--accelerator-type",
        "v4-32",
        "--phase",
        PHASE_NAME,
        "--seed",
        "0",
        "--checkpoint-tokens",
        "2000000",
        "--source-revision",
        "a" * 40,
        "--train-target-root",
        "/tmp/train-target",
        "--eval-target-root",
        "/tmp/eval-target",
        "--train-target-object-prefix",
        "gs://bucket/train",
        "--eval-target-object-prefix",
        "gs://bucket/eval",
        "--cache-root",
        "/tmp/cache",
        "--cache-max-bytes",
        "1",
        "--sae-control-root",
        "/tmp/control",
        "--checkpoint-payload-root",
        "gs://bucket/checkpoints",
        "--recomposition-sidecar",
        "/tmp/sidecar.json",
        "--result-root",
        "/tmp/result",
        "--process-index",
        "0",
        "--peak-flops-per-second",
        "4400000000000000",
        "--run-root-h",
        "/tmp/control/runs/h",
        "--run-root-r",
        "/tmp/control/runs/r",
        "--run-root-b",
        "/tmp/control/runs/b",
        "--run-root-f",
        "/tmp/control/runs/f",
    ]


def test_cli_requires_exactly_four_overlay_arguments_for_transcoder() -> None:
    sae = _parser().parse_args(_base_argv())
    _require_source_arguments(sae)
    transcoder = _parser().parse_args(_base_argv("transcoder"))
    with pytest.raises(
        Experiment5EvaluationWorkerError,
        match="all four",
    ):
        _require_source_arguments(transcoder)
    transcoder = _parser().parse_args(
        [
            *_base_argv("transcoder"),
            "--train-source-root",
            "/tmp/train-source",
            "--eval-source-root",
            "/tmp/eval-source",
            "--train-source-object-prefix",
            "gs://bucket/train-source",
            "--eval-source-object-prefix",
            "gs://bucket/eval-source",
        ]
    )
    _require_source_arguments(transcoder)
    sae_with_source = _parser().parse_args(
        [
            *_base_argv(),
            "--train-source-root",
            "/tmp/train-source",
            "--eval-source-root",
            "/tmp/eval-source",
            "--train-source-object-prefix",
            "gs://bucket/train-source",
            "--eval-source-object-prefix",
            "gs://bucket/eval-source",
        ]
    )
    with pytest.raises(Experiment5EvaluationWorkerError, match="forbids"):
        _require_source_arguments(sae_with_source)


def test_orphaned_paired_sidecar_is_replayable_but_result_only_is_not(
    tmp_path,
) -> None:
    config = _config()
    args = _parser().parse_args(_base_argv())
    root = tmp_path / "result"
    root.mkdir()
    body = {
        "schema_version": "gemma4-routed-shared-hybrid-paired-scores-v1",
        "protocol_id": config.protocol_id,
        "panel": "sae",
        "seed": 0,
        "checkpoint_target_tokens": 2_000_000,
        "packed_sequence_ids": [],
        "valid_tokens": [],
        "sequence_sst": [],
        "sequence_sse": {"H": [], "R": [], "B": [], "F": []},
    }
    paired = {**body, "artifact_sha256": canonical_sha256(body)}
    (root / "paired-scores.json").write_text(
        json.dumps(paired),
        encoding="ascii",
    )
    assert not _existing_completion(root, args=args, config=config)

    (root / "paired-scores.json").unlink()
    (root / "scientific-result.json").write_text("{}", encoding="ascii")
    with pytest.raises(
        Experiment5EvaluationWorkerError,
        match="without its paired evidence",
    ):
        _existing_completion(root, args=args, config=config)


def test_milestone_route_counts_are_selected_by_actual_optimizer_tokens() -> None:
    layout = production_experiment5_layout("H", randomization_key=0)
    actual = 2_015_232
    counts = [actual // 16] * 128
    context = {
        "routed_bank_component_counts": [625_000] * 128,
        "routed_bank_component_counts_by_training_tokens": [
            [actual, counts],
            [5_013_504, [313_344] * 128],
            [10_000_000, [625_000] * 128],
        ],
    }
    assert (
        _milestone_routed_counts(
            context,
            layout=layout,
            actual_tokens=actual,
        )
        == counts
    )
    with pytest.raises(
        Experiment5EvaluationWorkerError,
        match="lacks the selected milestone",
    ):
        _milestone_routed_counts(
            context,
            layout=layout,
            actual_tokens=2_000_000,
        )


def test_addressable_checkpoint_mask_requires_one_exact_physical_cover() -> None:
    class Array:
        is_fully_addressable = False

        def __init__(self):
            self.addressable_shards = [
                SimpleNamespace(
                    index=(slice(2, 4),),
                    data=np.asarray([False, True], dtype=np.bool_),
                ),
                SimpleNamespace(
                    index=(slice(0, 2),),
                    data=np.asarray([True, False], dtype=np.bool_),
                ),
            ]

    assert np.array_equal(
        _gather_physical_firing_mask(Array(), total_features=4),
        np.asarray([True, False, False, True]),
    )
    broken = Array()
    broken.addressable_shards = broken.addressable_shards[1:]
    with pytest.raises(
        Experiment5EvaluationWorkerError,
        match="incomplete",
    ):
        _gather_physical_firing_mask(broken, total_features=4)


def test_checkpoint_timing_builds_milestone_specific_mfu_receipt() -> None:
    config = _config()
    plan = _plan()
    layout = production_experiment5_layout("H", randomization_key=0)
    actual = plan.expected_tokens_after(
        int(np.ceil(2_000_000 / plan.effective_batch_tokens))
    )
    counts = [actual // 16] * 128
    receipt = _training_receipt(
        config=config,
        plan=plan,
        layout=layout,
        result={"result_sha256": "1" * 64},
        pointer={
            "step": 123,
            "tokens_seen": actual,
            "pointer_sha256": "2" * 64,
        },
        metadata={
            "timing": {
                "compile_seconds": 10.0,
                "lowering_seconds": 1.0,
                "preprocessing_wall_seconds": 20.0,
                "optimizer_wall_seconds": 100.0,
                "checkpoint_seconds_before_save": 5.0,
                "global_chip_count": 16,
            }
        },
        context={
            "routed_bank_component_counts": [625_000] * 128,
            "routed_bank_component_counts_by_training_tokens": [
                [actual, counts]
            ],
        },
        execution={
            "accelerator_type": "v4-32",
            "peak_flops_per_second": 4_400_000_000_000_000,
            "global_chip_count": 16,
        },
        checkpoint_target_tokens=2_000_000,
    )
    assert receipt.actual_training_tokens == actual
    assert receipt.marginal_wall_seconds == 120.0
    assert receipt.marginal_chip_seconds == 1_920.0
    assert sum(receipt.routed_bank_component_counts or ()) == actual * 8
    assert receipt.useful_mfu is not None and 0 < receipt.useful_mfu < 1


def test_training_context_binds_physical_layout_and_manifest_centering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    plan = _plan()
    layout = production_experiment5_layout("H", randomization_key=0)
    routed = np.full((2_816,), 0.25, dtype=np.float32)
    shared = np.full((2_816,), -0.125, dtype=np.float32)
    full = np.linspace(-1, 1, 2_816, dtype=np.float32)
    physical = experiment5_physical_layout(layout, 4).to_dict()
    rows = [{"process_index": index} for index in range(4)]
    context = {
        "experiment": 5,
        "panel": "sae",
        "target_manifest_sha256": config.parents[
            "train_target_manifest_sha256"
        ],
        "source_manifest_sha256": None,
        "input_receipts": rows,
        "input_receipt_set_sha256": canonical_sha256(rows),
        "prefetch_receipts": rows,
        "prefetch_receipt_set_sha256": canonical_sha256(rows),
        "preprocessing_receipts": rows,
        "preprocessing_receipt_set_sha256": canonical_sha256(rows),
        "centering": {
            "source_mean_sha256": _mean_sha256(
                b"gemma4-experiment5-full-target-mean-v1",
                full,
            ),
            "routed_target_mean_sha256": _mean_sha256(
                b"gemma4-contribution-centering-mean-v1",
                routed,
            ),
            "shared_target_mean_sha256": _mean_sha256(
                b"gemma4-shared-q-centering-mean-v1",
                shared,
            ),
            "full_target_mean_sha256": _mean_sha256(
                b"gemma4-experiment5-full-target-mean-v1",
                full,
            ),
            "full_target_sst": 1.0,
            "recomposition_sidecar_sha256": (
                config.recomposition_sidecar_sha256
            ),
        },
        "centering_values": {
            "source_mean": full.tolist(),
            "routed_target_mean": routed.tolist(),
            "shared_target_mean": shared.tolist(),
            "full_target_mean": full.tolist(),
        },
        "bank_capacity": 1,
        "routed_bank_component_counts": [625_000] * 128,
        "routed_bank_component_counts_by_training_tokens": [],
        "physical_feature_layout": (
            "balanced_routed_banks_then_shared_slice_per_feature_shard_v1"
        ),
        "physical_feature_axis_size": 4,
        "physical_feature_layout_identity": physical,
        "resident_cache_policy": "resident_only_no_refill",
    }
    execution = {
        "accelerator_type": "v4-32",
        "peak_flops_per_second": 4_400_000_000_000_000,
        "global_chip_count": 16,
        "physical_feature_layout_identity": physical,
    }
    monkeypatch.setattr(
        "moe_sae.experiment5_evaluation_worker._pooled_routed_mean",
        lambda _dataset: (
            routed,
            context["centering"]["routed_target_mean_sha256"],
        ),
    )
    monkeypatch.setattr(
        "moe_sae.contribution_shared.build_shared_q_centering",
        lambda _dataset, _sidecar: SimpleNamespace(
            mean=shared,
            mean_sha256=context["centering"]["shared_target_mean_sha256"],
        ),
    )
    selected_context, selected_execution, centering, _identity = (
        _validate_training_context(
            {
                "experiment5_context": context,
                "experiment5_execution": execution,
            },
            config=config,
            plan=plan,
            layout=layout,
            feature_axis_size=4,
            global_chip_count=16,
            accelerator_type="v4-32",
            training_target=object(),
            training_source=None,
            sidecar={},
        )
    )
    assert selected_context == context
    assert selected_execution == execution
    assert np.array_equal(centering.full_target_mean, full)
