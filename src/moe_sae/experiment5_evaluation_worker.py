"""Multi-host production evaluator for Experiment 5.

Every TPU process restores the same physical-feature-sharded checkpoint,
scores only its contiguous held-out lane, and publishes one small self-hashed
partial. Process zero merges exact token-position bitmaps and additive
statistics, then seals ``paired-scores.json`` before the commit-marker
``scientific-result.json``. No parameter tree or activation reconstruction is
materialized on a host.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from moe_sae.checkpointing import milestone_pointer_path
from moe_sae.contribution_artifacts import (
    canonical_json_bytes,
    canonical_sha256,
)
from moe_sae.contribution_evaluation_worker import (
    _load_run,
)
from moe_sae.contribution_training import ContributionOptimizerConfig
from moe_sae.contribution_training_performance import (
    SAEArithmetic,
    sae_training_useful_flops,
)
from moe_sae.contribution_training_worker import (
    TrainingWorkerPlan,
    _collective_barrier,
    _collective_records,
    _pooled_routed_mean,
    _verify_repository_revision,
)
from moe_sae.experiment5_config import (
    CONDITIONS,
    PHASE_NAME,
    Experiment5Config,
    load_experiment5_config,
)
from moe_sae.experiment5_device_evaluation import (
    Experiment5CanonicalTargetEvaluator,
    Experiment5DistributedEvaluator,
    process_local_experiment5_device_scores,
)
from moe_sae.experiment5_distributed import (
    experiment5_physical_layout,
    init_experiment5_distributed_state,
)
from moe_sae.experiment5_evaluation import (
    Experiment5EvaluationAccumulator,
    Experiment5EvaluationError,
    build_experiment5_evaluation_result,
    build_experiment5_training_receipt,
    logicalize_experiment5_checkpoint_firing_mask,
    pack_experiment5_device_rows,
)
from moe_sae.experiment5_hybrid import (
    Experiment5Centering,
    Experiment5Layout,
    Experiment5Recomposition,
    experiment5_optimizer_sae_config,
    production_experiment5_layout,
)
from moe_sae.experiment5_input import Experiment5ProcessInput
from moe_sae.experiment5_report import (
    EVAL_TOKENS,
    validate_experiment5_paired_scores,
    validate_experiment5_result,
)
from moe_sae.pipeline import TrainingConfig

_CONTEXT_FIELDS = {
    "experiment",
    "panel",
    "target_manifest_sha256",
    "source_manifest_sha256",
    "input_receipts",
    "input_receipt_set_sha256",
    "prefetch_receipts",
    "prefetch_receipt_set_sha256",
    "preprocessing_receipts",
    "preprocessing_receipt_set_sha256",
    "centering",
    "centering_values",
    "bank_capacity",
    "routed_bank_component_counts",
    "routed_bank_component_counts_by_training_tokens",
    "physical_feature_layout",
    "physical_feature_axis_size",
    "physical_feature_layout_identity",
    "resident_cache_policy",
}
_CENTERING_FIELDS = {
    "source_mean_sha256",
    "routed_target_mean_sha256",
    "shared_target_mean_sha256",
    "full_target_mean_sha256",
    "full_target_sst",
    "recomposition_sidecar_sha256",
}
_CENTERING_VALUE_FIELDS = {
    "source_mean",
    "routed_target_mean",
    "shared_target_mean",
    "full_target_mean",
}
_EXECUTION_FIELDS = {
    "accelerator_type",
    "peak_flops_per_second",
    "global_chip_count",
    "physical_feature_layout_identity",
}
_TIMING_FIELDS = {
    "compile_seconds",
    "lowering_seconds",
    "preprocessing_wall_seconds",
    "optimizer_wall_seconds",
    "checkpoint_seconds_before_save",
    "global_chip_count",
}
_ACCELERATOR_GENERATION = {
    "v4-32": "v4",
    "v5litepod-16": "v5e",
    "v6e-16": "v6e",
}


class Experiment5EvaluationWorkerError(RuntimeError):
    """Production evaluation evidence is incomplete or incompatible."""


def _mapping(value: Any, *, name: str, fields: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Experiment5EvaluationWorkerError(f"{name} must be a mapping")
    selected = dict(value)
    if fields is not None and set(selected) != fields:
        raise Experiment5EvaluationWorkerError(f"{name} fields changed")
    return selected


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Experiment5EvaluationWorkerError(f"{name} must be positive")
    return value


def _nonnegative(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Experiment5EvaluationWorkerError(f"{name} must be numeric")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0 or (positive and selected <= 0):
        raise Experiment5EvaluationWorkerError(f"{name} is invalid")
    return selected


def _read_json(path: str | Path, *, name: str) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        value = json.loads(selected.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Experiment5EvaluationWorkerError(
            f"cannot read {name}: {selected}"
        ) from error
    if not isinstance(value, Mapping):
        raise Experiment5EvaluationWorkerError(f"{name} must be an object")
    return dict(value)


def _read_sidecar(path: str | Path, config: Experiment5Config) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        payload = selected.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Experiment5EvaluationWorkerError(
            f"cannot read recomposition sidecar: {selected}"
        ) from error
    expected_file = config.raw["artifacts"][
        "recomposition_sidecar_file_sha256"
    ]
    if hashlib.sha256(payload).hexdigest() != expected_file:
        raise Experiment5EvaluationWorkerError(
            "recomposition sidecar file-byte identity changed"
        )
    if not isinstance(value, Mapping):
        raise Experiment5EvaluationWorkerError(
            "recomposition sidecar must be an object"
        )
    from moe_sae.contribution_recomposition import decode_recomposition_sidecar

    decoded = decode_recomposition_sidecar(value)
    if decoded.sidecar_sha256 != config.recomposition_sidecar_sha256:
        raise Experiment5EvaluationWorkerError(
            "recomposition sidecar content identity changed"
        )
    return dict(value)


def _create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise Experiment5EvaluationWorkerError(
                f"immutable evaluation artifact changed on replay: {path}"
            )
        return
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _mean_sha256(prefix: bytes, value: np.ndarray) -> str:
    selected = np.ascontiguousarray(value, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(prefix + b"\0")
    digest.update(selected.tobytes(order="C"))
    return digest.hexdigest()


def _centering_from_context(
    context: Mapping[str, Any],
    *,
    panel: str,
    training_target: Any,
    training_source: Any | None,
    sidecar: Mapping[str, Any],
) -> tuple[Experiment5Centering, dict[str, Any]]:
    identity = _mapping(
        context.get("centering"),
        name="experiment5_context.centering",
        fields=_CENTERING_FIELDS,
    )
    values = _mapping(
        context.get("centering_values"),
        name="experiment5_context.centering_values",
        fields=_CENTERING_VALUE_FIELDS,
    )
    arrays: dict[str, np.ndarray] = {}
    for name in _CENTERING_VALUE_FIELDS:
        selected = np.asarray(values[name])
        if (
            selected.shape != (2_816,)
            or not np.issubdtype(selected.dtype, np.floating)
            or not np.isfinite(selected).all()
        ):
            raise Experiment5EvaluationWorkerError(
                f"training {name} is not finite FP32[2816]"
            )
        arrays[name] = np.ascontiguousarray(selected, dtype=np.float32)
    expected_hashes = {
        "routed_target_mean_sha256": _mean_sha256(
            b"gemma4-contribution-centering-mean-v1",
            arrays["routed_target_mean"],
        ),
        "shared_target_mean_sha256": _mean_sha256(
            b"gemma4-shared-q-centering-mean-v1",
            arrays["shared_target_mean"],
        ),
        "full_target_mean_sha256": _mean_sha256(
            b"gemma4-experiment5-full-target-mean-v1",
            arrays["full_target_mean"],
        ),
    }
    expected_hashes["source_mean_sha256"] = (
        expected_hashes["full_target_mean_sha256"]
        if panel == "sae"
        else _mean_sha256(
            b"gemma4-routed-input-centering-mean-v1",
            arrays["source_mean"],
        )
    )
    if any(identity[name] != value for name, value in expected_hashes.items()):
        raise Experiment5EvaluationWorkerError(
            "training centering values differ from their hashes"
        )
    if not _nonnegative(
        identity["full_target_sst"],
        name="training full_target_sst",
        positive=True,
    ):
        raise Experiment5EvaluationWorkerError("training full-target SST is invalid")

    routed_mean, routed_sha = _pooled_routed_mean(training_target)
    if (
        routed_sha != identity["routed_target_mean_sha256"]
        or not np.array_equal(routed_mean, arrays["routed_target_mean"])
    ):
        raise Experiment5EvaluationWorkerError(
            "training routed centering differs from the sealed target manifest"
        )
    from moe_sae.contribution_shared import build_shared_q_centering

    shared = build_shared_q_centering(training_target, sidecar)
    if (
        shared.mean_sha256 != identity["shared_target_mean_sha256"]
        or not np.array_equal(shared.mean, arrays["shared_target_mean"])
    ):
        raise Experiment5EvaluationWorkerError(
            "training shared centering differs from the sealed target manifest"
        )
    if panel == "transcoder":
        if training_source is None:
            raise Experiment5EvaluationWorkerError(
                "transcoder training source is unavailable"
            )
        from moe_sae.contribution_transcoder_input import routed_input_mean

        source_mean, source_sha = routed_input_mean(training_source)
        if (
            source_sha != identity["source_mean_sha256"]
            or not np.array_equal(source_mean, arrays["source_mean"])
        ):
            raise Experiment5EvaluationWorkerError(
                "training source centering differs from the sealed overlay"
            )
    elif not np.array_equal(
        arrays["source_mean"],
        arrays["full_target_mean"],
    ):
        raise Experiment5EvaluationWorkerError(
            "SAE source mean must equal its full-target mean placeholder"
        )
    return (
        Experiment5Centering(
            source_mean=arrays["source_mean"],
            routed_target_mean=arrays["routed_target_mean"],
            shared_target_mean=arrays["shared_target_mean"],
            full_target_mean=arrays["full_target_mean"],
        ),
        {
            "identity": identity,
            "values_sha256": canonical_sha256(
                {
                    name: arrays[name].tolist()
                    for name in sorted(arrays)
                }
            ),
        },
    )


def _validate_training_context(
    result: Mapping[str, Any],
    *,
    config: Experiment5Config,
    plan: TrainingWorkerPlan,
    layout: Experiment5Layout,
    feature_axis_size: int,
    global_chip_count: int,
    accelerator_type: str,
    training_target: Any,
    training_source: Any | None,
    sidecar: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Experiment5Centering, dict[str, Any]]:
    context = _mapping(
        result.get("experiment5_context"),
        name="experiment5_context",
        fields=_CONTEXT_FIELDS,
    )
    execution = _mapping(
        result.get("experiment5_execution"),
        name="experiment5_execution",
        fields=_EXECUTION_FIELDS,
    )
    expected_source_sha = (
        config.parents["train_source_overlay_manifest_sha256"]
        if config.panel == "transcoder"
        else None
    )
    if (
        context["experiment"] != 5
        or context["panel"] != config.panel
        or context["target_manifest_sha256"]
        != config.parents["train_target_manifest_sha256"]
        or context["source_manifest_sha256"] != expected_source_sha
        or context["resident_cache_policy"] != "resident_only_no_refill"
        or context["physical_feature_axis_size"] != feature_axis_size
        or _mapping(
            context["centering"],
            name="experiment5_context.centering",
            fields=_CENTERING_FIELDS,
        )["recomposition_sidecar_sha256"]
        != config.recomposition_sidecar_sha256
    ):
        raise Experiment5EvaluationWorkerError(
            "training Experiment-5 context identity changed"
        )
    for rows_name, hash_name in (
        ("input_receipts", "input_receipt_set_sha256"),
        ("prefetch_receipts", "prefetch_receipt_set_sha256"),
        ("preprocessing_receipts", "preprocessing_receipt_set_sha256"),
    ):
        rows = context[rows_name]
        if (
            not isinstance(rows, list)
            or len(rows) != plan.microbatches_per_update
            or canonical_sha256(rows) != context[hash_name]
        ):
            raise Experiment5EvaluationWorkerError(
                f"training {rows_name} evidence changed"
            )
    physical = experiment5_physical_layout(layout, feature_axis_size).to_dict()
    expected_physical_name = (
        "balanced_routed_banks_then_shared_slice_per_feature_shard_v1"
        if layout.is_hybrid
        else "contiguous_global_features_v1"
    )
    if (
        context["physical_feature_layout"] != expected_physical_name
        or context["physical_feature_layout_identity"] != physical
        or execution["physical_feature_layout_identity"] != physical
    ):
        raise Experiment5EvaluationWorkerError(
            "training physical feature layout differs from evaluation"
        )
    accelerator = execution["accelerator_type"]
    if (
        accelerator not in _ACCELERATOR_GENERATION
        or accelerator != accelerator_type
    ):
        raise Experiment5EvaluationWorkerError(
            "training and evaluation accelerator types differ"
        )
    if (
        _positive_int(
            execution["peak_flops_per_second"],
            name="training peak_flops_per_second",
        )
        <= 0
        or execution["global_chip_count"] != global_chip_count
    ):
        raise Experiment5EvaluationWorkerError(
            "training execution topology differs from evaluation"
        )
    centering, centering_identity = _centering_from_context(
        context,
        panel=config.panel,
        training_target=training_target,
        training_source=training_source,
        sidecar=sidecar,
    )
    return context, execution, centering, centering_identity


def _milestone_routed_counts(
    context: Mapping[str, Any],
    *,
    layout: Experiment5Layout,
    actual_tokens: int,
) -> Sequence[int] | None:
    final_counts = context["routed_bank_component_counts"]
    snapshots = context["routed_bank_component_counts_by_training_tokens"]
    if not layout.is_hybrid:
        if final_counts is not None or snapshots is not None:
            raise Experiment5EvaluationWorkerError(
                "global training context has routed exposure"
            )
        return None
    if (
        not isinstance(final_counts, list)
        or len(final_counts) != 128
        or sum(final_counts) != 10_000_000 * 8
        or not isinstance(snapshots, list)
    ):
        raise Experiment5EvaluationWorkerError(
            "hybrid training route exposure is invalid"
        )
    selected: list[int] | None = None
    observed_tokens: list[int] = []
    for row in snapshots:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or isinstance(row[0], bool)
            or not isinstance(row[0], int)
            or not isinstance(row[1], list)
            or len(row[1]) != 128
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in row[1]
            )
            or sum(row[1]) != row[0] * 8
        ):
            raise Experiment5EvaluationWorkerError(
                "hybrid milestone route exposure is invalid"
            )
        observed_tokens.append(row[0])
        if row[0] == actual_tokens:
            selected = row[1]
    if observed_tokens != sorted(set(observed_tokens)) or selected is None:
        raise Experiment5EvaluationWorkerError(
            "training context lacks the selected milestone route counts"
        )
    return selected


def _optimizer(config: Experiment5Config, plan: TrainingWorkerPlan) -> ContributionOptimizerConfig:
    return ContributionOptimizerConfig(
        parameter_dtype=config.optimizer.parameter_dtype,
        adam_beta1=config.optimizer.adam_beta1,
        adam_beta2=config.optimizer.adam_beta2,
        adam_eps=config.optimizer.adam_epsilon,
        grad_clip=config.optimizer.global_gradient_clip,
        peak_lr=config.optimizer.peak_learning_rate,
        min_lr=config.optimizer.minimum_learning_rate,
        warmup_steps=plan.warmup_updates,
        total_steps=plan.planned_updates,
    )


def _checkpoint_training_config(
    config: Experiment5Config,
    plan: TrainingWorkerPlan,
    layout: Experiment5Layout,
    bank_capacity: int | None,
) -> TrainingConfig:
    return TrainingConfig(
        sae=experiment5_optimizer_sae_config(layout, _optimizer(config, plan)),
        seed=plan.seed,
        total_tokens=plan.total_tokens,
        effective_batch_tokens=plan.effective_batch_tokens,
        microbatch_tokens=plan.global_microbatch_tokens,
        checkpoint_every_tokens=plan.total_tokens,
        health_gate_tokens=plan.total_tokens,
        a_capacity=bank_capacity,
    )


def _training_receipt(
    *,
    config: Experiment5Config,
    plan: TrainingWorkerPlan,
    layout: Experiment5Layout,
    result: Mapping[str, Any],
    pointer: Mapping[str, Any],
    metadata: Mapping[str, Any],
    context: Mapping[str, Any],
    execution: Mapping[str, Any],
    checkpoint_target_tokens: int,
) -> Any:
    timing = _mapping(metadata.get("timing"), name="checkpoint timing")
    if not _TIMING_FIELDS <= set(timing):
        raise Experiment5EvaluationWorkerError(
            "checkpoint timing fields are incomplete"
        )
    actual = int(pointer["tokens_seen"])
    expected_step = math.ceil(
        checkpoint_target_tokens / plan.effective_batch_tokens
    )
    if (
        pointer["step"] != expected_step
        or actual != plan.expected_tokens_after(expected_step)
    ):
        raise Experiment5EvaluationWorkerError(
            "checkpoint progress differs from the requested milestone"
        )
    preprocessing = _nonnegative(
        timing["preprocessing_wall_seconds"],
        name="preprocessing_wall_seconds",
    )
    optimizer_seconds = _nonnegative(
        timing["optimizer_wall_seconds"],
        name="optimizer_wall_seconds",
        positive=True,
    )
    chip_count = _positive_int(
        timing["global_chip_count"],
        name="global_chip_count",
    )
    if chip_count != execution["global_chip_count"]:
        raise Experiment5EvaluationWorkerError(
            "checkpoint and training-result chip counts differ"
        )
    marginal = preprocessing + optimizer_seconds
    peak = _positive_int(
        execution["peak_flops_per_second"],
        name="peak_flops_per_second",
    )
    useful_flops = sae_training_useful_flops(
        SAEArithmetic(
            workload="routed" if layout.is_hybrid else "shared_q",
            condition=layout.condition,
            hidden_dim=layout.hidden_dim,
            candidates_per_token=layout.candidates_per_token,
            selected_features_per_token=layout.total_l0,
        ),
        scientific_tokens=actual,
    )
    useful_mfu = useful_flops / (peak * optimizer_seconds)
    return build_experiment5_training_receipt(
        panel=config.panel,
        condition=layout.condition,
        seed=plan.seed,
        checkpoint_target_tokens=checkpoint_target_tokens,
        actual_training_tokens=actual,
        training_result_sha256=result["result_sha256"],
        checkpoint_pointer_sha256=pointer["pointer_sha256"],
        accelerator_type=execution["accelerator_type"],
        accelerator_generation=_ACCELERATOR_GENERATION[
            execution["accelerator_type"]
        ],
        chip_count=chip_count,
        marginal_wall_seconds=marginal,
        marginal_chip_seconds=marginal * chip_count,
        tokens_per_second=actual / marginal,
        useful_mfu=useful_mfu,
        routed_bank_component_counts=_milestone_routed_counts(
            context,
            layout=layout,
            actual_tokens=actual,
        ),
    )


def _gather_physical_firing_mask(value: Any, *, total_features: int) -> np.ndarray:
    if getattr(value, "is_fully_addressable", False):
        selected = np.asarray(value)
    else:
        pieces: dict[tuple[int, int], np.ndarray] = {}
        for shard in value.addressable_shards:
            index = shard.index[0]
            if not isinstance(index, slice):
                raise Experiment5EvaluationWorkerError(
                    "checkpoint firing shard index is not a slice"
                )
            start = int(index.start or 0)
            stop = int(index.stop or total_features)
            pieces.setdefault((start, stop), np.asarray(shard.data).reshape(-1))
        cursor = 0
        ordered: list[np.ndarray] = []
        for (start, stop), piece in sorted(pieces.items()):
            if start != cursor or piece.shape != (stop - start,):
                raise Experiment5EvaluationWorkerError(
                    "checkpoint firing shards do not cover physical features"
                )
            ordered.append(piece)
            cursor = stop
        if cursor != total_features:
            raise Experiment5EvaluationWorkerError(
                "checkpoint firing shards are incomplete"
            )
        selected = np.concatenate(ordered)
    if selected.shape != (total_features,) or selected.dtype != np.bool_:
        raise Experiment5EvaluationWorkerError(
            "checkpoint firing mask shape or dtype changed"
        )
    return np.ascontiguousarray(selected)


def _require_source_arguments(args: argparse.Namespace) -> None:
    values = (
        args.train_source_root,
        args.eval_source_root,
        args.train_source_object_prefix,
        args.eval_source_object_prefix,
    )
    if args.panel == "transcoder" and any(value is None for value in values):
        raise Experiment5EvaluationWorkerError(
            "transcoder evaluation requires all four train/eval source arguments"
        )
    if args.panel == "sae" and any(value is not None for value in values):
        raise Experiment5EvaluationWorkerError(
            "SAE evaluation forbids transcoder source arguments"
        )


def _existing_completion(
    root: Path,
    *,
    args: argparse.Namespace,
    config: Experiment5Config,
) -> bool:
    result_path = root / "scientific-result.json"
    paired_path = root / "paired-scores.json"
    if result_path.exists() and not paired_path.exists():
        raise Experiment5EvaluationWorkerError(
            "evaluation completion marker exists without its paired evidence"
        )
    # Paired evidence is intentionally published first. Losing a worker in
    # that narrow interval is replay-safe because this artifact is
    # deterministic and create-only publication compares the recomputed
    # bytes before writing the scientific completion marker.
    if paired_path.exists() and not result_path.exists():
        paired = _read_json(paired_path, name="orphaned paired scores")
        unsigned = dict(paired)
        claimed = unsigned.pop("artifact_sha256", None)
        if (
            not isinstance(claimed, str)
            or canonical_sha256(unsigned) != claimed
            or paired.get("panel") != args.panel
            or paired.get("seed") != args.seed
            or paired.get("checkpoint_target_tokens")
            != args.checkpoint_tokens
        ):
            raise Experiment5EvaluationWorkerError(
                "orphaned paired evidence is invalid or belongs elsewhere"
            )
        return False
    if not result_path.exists():
        return False
    checked = validate_experiment5_result(
        _read_json(result_path, name="existing scientific result")
    )
    paired = _read_json(paired_path, name="existing paired scores")
    validate_experiment5_paired_scores(paired, result=checked)
    if (
        checked.panel != args.panel
        or checked.seed != args.seed
        or checked.milestone != args.checkpoint_tokens
        or checked.body["source_revision"] != args.source_revision
        or checked.body["config_sha256"] != config.config_hash
    ):
        raise Experiment5EvaluationWorkerError(
            "existing completion belongs to another evaluation"
        )
    return True


def _load_datasets(
    args: argparse.Namespace,
    config: Experiment5Config,
    sidecar: Mapping[str, Any],
) -> tuple[Any, Any, Any | None, Any | None]:
    from moe_sae.contribution_storage import ContributionDataset
    from moe_sae.object_store import GCSObjectStore, ShardCache

    object_store: Any = GCSObjectStore()
    if args.regional_mirror_seal is not None:
        from moe_sae.regional_execution import RegionalMirrorStore

        object_store = RegionalMirrorStore(
            _read_json(args.regional_mirror_seal, name="regional mirror seal"),
            object_store,
        )
    cache = ShardCache(
        args.cache_root,
        object_store,
        max_bytes=args.cache_max_bytes,
    )
    train_target = ContributionDataset(
        args.train_target_root,
        object_prefix=args.train_target_object_prefix,
        cache=cache,
        recomposition_sidecar=sidecar,
    )
    eval_target = ContributionDataset(
        args.eval_target_root,
        object_prefix=args.eval_target_object_prefix,
        cache=cache,
        recomposition_sidecar=sidecar,
    )
    if (
        train_target.manifest_sha256
        != config.parents["train_target_manifest_sha256"]
        or eval_target.manifest_sha256
        != config.parents["eval_target_manifest_sha256"]
        or int(train_target.total_tokens) != 10_000_000
        or int(eval_target.total_tokens) != EVAL_TOKENS
    ):
        raise Experiment5EvaluationWorkerError(
            "target datasets differ from the pinned Experiment-5 parents"
        )
    if args.panel == "sae":
        return train_target, eval_target, None, None
    from moe_sae.contribution_transcoder_input import (
        require_overlay_capture_contract,
    )
    from moe_sae.contribution_transcoder_storage import RoutedInputDataset

    train_source = RoutedInputDataset(
        args.train_source_root,
        object_prefix=args.train_source_object_prefix,
        cache=cache,
        target_manifest=train_target.manifest,
    )
    eval_source = RoutedInputDataset(
        args.eval_source_root,
        object_prefix=args.eval_source_object_prefix,
        cache=cache,
        target_manifest=eval_target.manifest,
    )
    for source in (train_source, eval_source):
        require_overlay_capture_contract(
            source,
            config.parents["source_overlay_capture_contract_sha256"],
        )
    if (
        train_source.manifest_sha256
        != config.parents["train_source_overlay_manifest_sha256"]
        or eval_source.manifest_sha256
        != config.parents["eval_source_overlay_manifest_sha256"]
        or int(train_source.total_tokens) != 10_000_000
        or int(eval_source.total_tokens) != EVAL_TOKENS
    ):
        raise Experiment5EvaluationWorkerError(
            "source overlays differ from the pinned Experiment-5 parents"
        )
    return train_target, eval_target, train_source, eval_source


def _validate_run_roots(args: argparse.Namespace) -> dict[str, Path]:
    control = Path(args.sae_control_root).expanduser().resolve()
    roots = {
        condition: Path(getattr(args, f"run_root_{condition.lower()}"))
        .expanduser()
        .resolve()
        for condition in CONDITIONS
    }
    if len(set(roots.values())) != len(CONDITIONS):
        raise Experiment5EvaluationWorkerError("condition run roots must be unique")
    for condition, root in roots.items():
        if control not in root.parents:
            raise Experiment5EvaluationWorkerError(
                f"{condition} run root is outside --sae-control-root"
            )
    return roots


def run(args: argparse.Namespace) -> int:
    import jax

    _require_source_arguments(args)
    _verify_repository_revision(args.source_revision)
    config = load_experiment5_config(args.config, expected_panel=args.panel)
    if args.phase != PHASE_NAME or args.checkpoint_tokens not in config.phase(
        args.phase
    ).evaluation_checkpoint_tokens:
        raise Experiment5EvaluationWorkerError(
            "phase or checkpoint milestone differs from Experiment 5"
        )
    sidecar = _read_sidecar(args.recomposition_sidecar, config)
    run_roots = _validate_run_roots(args)

    from moe_sae.runtime import (
        initialize_jax_distributed,
        require_accelerator_runtime,
    )

    initialize_jax_distributed()
    require_accelerator_runtime(
        args.accelerator_type,
        peak_flops_per_second=args.peak_flops_per_second,
    )
    if int(jax.process_index()) != args.process_index:
        raise Experiment5EvaluationWorkerError(
            "physical controller/JAX process mapping changed"
        )
    from moe_sae.distributed import (
        DistributedConfig,
        create_mesh,
        restore_distributed_checkpoint,
    )

    distributed = DistributedConfig(
        data_axis_size=int(jax.process_count()),
        feature_axis_size=int(jax.local_device_count()),
        microbatches_per_update=4,
    )
    mesh = create_mesh(distributed)
    global_chip_count = int(np.asarray(mesh.devices, dtype=object).size)
    output = Path(args.result_root).expanduser().resolve()
    statuses = _collective_records(
        {
            "process_index": int(jax.process_index()),
            "complete": _existing_completion(output, args=args, config=config),
        }
    )
    if len({bool(row["complete"]) for row in statuses}) != 1:
        raise Experiment5EvaluationWorkerError(
            "evaluation completion visibility differs across processes"
        )
    if statuses[0]["complete"]:
        if jax.process_index() == 0:
            print(
                json.dumps(
                    {
                        "status": "already_complete",
                        "result_root": str(output),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        _collective_barrier(
            f"experiment5-eval-existing-{args.panel}-{args.seed}-"
            f"{args.checkpoint_tokens}"
        )
        return 0

    train_target, eval_target, train_source, eval_source = _load_datasets(
        args,
        config,
        sidecar,
    )
    from moe_sae.contribution_artifacts import bf16_bits_to_float32
    from moe_sae.contribution_recomposition import decode_recomposition_sidecar

    decoded_sidecar = decode_recomposition_sidecar(sidecar)
    recomposition = Experiment5Recomposition(
        routed_scale=bf16_bits_to_float32(
            decoded_sidecar.routed_rmsnorm_scale_bf16_bits
        ),
        outer_scale=bf16_bits_to_float32(
            decoded_sidecar.outer_ffn_rmsnorm_scale_bf16_bits
        ),
        epsilon=float(decoded_sidecar.epsilon),
    )
    canonical_target_evaluator = Experiment5CanonicalTargetEvaluator(
        mesh,
        distributed,
        hidden_dim=2_816,
        top_routed=8,
        rmsnorm_epsilon=float(recomposition.epsilon),
    )

    evidence: dict[str, dict[str, Any]] = {}
    common_centering_sha: str | None = None
    for condition in CONDITIONS:
        plan = TrainingWorkerPlan.from_config(
            config,
            phase=args.phase,
            condition=condition,
            seed=args.seed,
        )
        result, pointer = _load_run(
            run_roots[condition],
            plan,
            args.checkpoint_payload_root,
            checkpoint_target_tokens=args.checkpoint_tokens,
        )
        layout = production_experiment5_layout(
            condition,
            randomization_key=plan.randomization_key,
        )
        context, execution, centering, centering_identity = (
            _validate_training_context(
                result,
                config=config,
                plan=plan,
                layout=layout,
                feature_axis_size=int(jax.local_device_count()),
                global_chip_count=global_chip_count,
                accelerator_type=args.accelerator_type,
                training_target=train_target,
                training_source=train_source,
                sidecar=sidecar,
            )
        )
        selected_centering_sha = canonical_sha256(centering_identity)
        if common_centering_sha is None:
            common_centering_sha = selected_centering_sha
        elif selected_centering_sha != common_centering_sha:
            raise Experiment5EvaluationWorkerError(
                "H/R/B/F training centering evidence differs"
            )
        evidence[condition] = {
            "plan": plan,
            "result": result,
            "pointer": pointer,
            "layout": layout,
            "context": context,
            "execution": execution,
            "centering": centering,
        }

    global_accumulators: dict[str, Experiment5EvaluationAccumulator] = {}
    for condition in CONDITIONS:
        selected = evidence[condition]
        plan = selected["plan"]
        layout = selected["layout"]
        context = selected["context"]
        bank_capacity = context["bank_capacity"]
        if layout.is_hybrid:
            bank_capacity = _positive_int(
                bank_capacity,
                name=f"{condition} bank_capacity",
            )
        elif bank_capacity is not None:
            raise Experiment5EvaluationWorkerError(
                f"{condition} global layout has routed capacity"
            )
        optimizer = _optimizer(config, plan)
        template = init_experiment5_distributed_state(
            jax.random.key(args.seed),
            mesh,
            layout,
            optimizer,
            distributed,
        )
        checkpoint_path = milestone_pointer_path(
            run_roots[condition] / "checkpoints",
            args.checkpoint_tokens,
        )
        state, metadata = restore_distributed_checkpoint(
            checkpoint_path,
            template,
            mesh,
            _checkpoint_training_config(
                config,
                plan,
                layout,
                bank_capacity,
            ),
            checkpoint_payload_root=args.checkpoint_payload_root,
            checkpoint_identity=selected["result"]["run_identity"],
            checkpoint_metadata_identity={
                "run_identity_sha256": selected["result"]["run_identity"][
                    "run_identity_sha256"
                ]
            },
        )
        if not isinstance(metadata, Mapping):
            raise Experiment5EvaluationWorkerError(
                f"{condition} checkpoint lacks metadata"
            )
        physical_mask = _gather_physical_firing_mask(
            state.fired_mask,
            total_features=layout.total_features,
        )
        logical_shard = logicalize_experiment5_checkpoint_firing_mask(
            physical_mask,
            layout=layout,
            feature_axis_size=int(jax.local_device_count()),
            expected_physical_layout_sha256=context[
                "physical_feature_layout_identity"
            ]["identity_sha256"],
        )
        receipt = _training_receipt(
            config=config,
            plan=plan,
            layout=layout,
            result=selected["result"],
            pointer=selected["pointer"],
            metadata=metadata,
            context=context,
            execution=selected["execution"],
            checkpoint_target_tokens=args.checkpoint_tokens,
        )
        mask_records = _collective_records(
            {
                "process_index": int(jax.process_index()),
                "training_receipt_sha256": receipt.receipt_sha256,
                "fired_count": int(np.count_nonzero(logical_shard.fired)),
                "fired_mask_sha256": hashlib.sha256(
                    np.packbits(
                        logical_shard.fired,
                        bitorder="little",
                    ).tobytes(order="C")
                ).hexdigest(),
            }
        )
        if len(
            {
                (
                    row["training_receipt_sha256"],
                    row["fired_count"],
                    row["fired_mask_sha256"],
                )
                for row in mask_records
            }
        ) != 1:
            raise Experiment5EvaluationWorkerError(
                f"{condition} restored checkpoint differs across processes"
            )

        source = Experiment5ProcessInput(
            eval_target,
            sidecar,
            layout,
            panel=args.panel,
            total_tokens=EVAL_TOKENS,
            source_dataset=eval_source,
            global_microbatch_tokens=plan.global_microbatch_tokens,
            microbatches_per_update=plan.microbatches_per_update,
            process_count=int(jax.process_count()),
            process_index=int(jax.process_index()),
        )
        accumulator = Experiment5EvaluationAccumulator(
            panel=args.panel,
            layout=layout,
            seed=args.seed,
            milestone=args.checkpoint_tokens,
            training_receipt=receipt,
            checkpoint_firing_shards=[logical_shard],
        )
        evaluator = Experiment5DistributedEvaluator(
            mesh,
            layout,
            distributed,
            encoder_kind=args.panel,
            bank_capacity=bank_capacity,
            rmsnorm_epsilon=float(recomposition.epsilon),
            expected_physical_layout_sha256=context[
                "physical_feature_layout_identity"
            ]["identity_sha256"],
            canonical_target_evaluator=canonical_target_evaluator,
        )
        iterator = iter(source.iter_updates())
        try:
            first = next(iterator)
        except StopIteration as error:
            raise Experiment5EvaluationWorkerError(
                "held-out input produced no updates"
            ) from error
        compile_seconds = evaluator.compile(
            state,
            first,
            selected["centering"],
            recomposition,
            process_local=True,
        )
        import time

        scoring_started = time.perf_counter()
        for update, microbatches in enumerate(
            itertools.chain((first,), iterator)
        ):
            device_scores = evaluator.score(
                state,
                microbatches,
                selected["centering"],
                recomposition,
                process_local=True,
            )
            local_scores = process_local_experiment5_device_scores(
                device_scores
            )
            sequence_id = np.stack(
                [np.asarray(batch.sequence_id) for batch in microbatches]
            ).reshape(-1)
            position = np.stack(
                [np.asarray(batch.position) for batch in microbatches]
            ).reshape(-1)
            valid_mask = np.stack(
                [np.asarray(batch.valid_mask) for batch in microbatches]
            ).reshape(-1)
            accumulator.add(
                pack_experiment5_device_rows(
                    chunk_id=(
                        f"{condition}-p{jax.process_index():05d}-"
                        f"u{update:06d}"
                    ),
                    panel=args.panel,
                    condition=condition,
                    seed=args.seed,
                    checkpoint_target_tokens=args.checkpoint_tokens,
                    sequence_id=sequence_id,
                    position=position,
                    valid_mask=valid_mask,
                    device_scores=local_scores,
                )
            )
        scoring_seconds = time.perf_counter() - scoring_started
        partial = accumulator.export_partial(
            partial_id=f"process-{jax.process_index():05d}"
        )
        partial_path = (
            output
            / "partials"
            / condition
            / f"process-p{jax.process_index():05d}.json"
        )
        _create_only_json(partial_path, partial.to_dict())
        _collective_barrier(
            f"experiment5-eval-partial-{args.panel}-{args.seed}-"
            f"{args.checkpoint_tokens}-{condition}"
        )
        if jax.process_index() == 0:
            merged = Experiment5EvaluationAccumulator(
                panel=args.panel,
                layout=layout,
                seed=args.seed,
                milestone=args.checkpoint_tokens,
                training_receipt=receipt,
                checkpoint_firing_shards=[logical_shard],
            )
            for process_index in range(jax.process_count()):
                merged.merge_partial(
                    _read_json(
                        output
                        / "partials"
                        / condition
                        / f"process-p{process_index:05d}.json",
                        name=f"{condition} process partial",
                    )
                )
            global_accumulators[condition] = merged
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "checkpoint_tokens": args.checkpoint_tokens,
                        "compile_seconds": compile_seconds,
                        "scoring_seconds": scoring_seconds,
                        "partial_sha256": partial.partial_sha256,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del evaluator, state, template, accumulator
        jax.clear_caches()
        _collective_barrier(
            f"experiment5-eval-condition-{args.panel}-{args.seed}-"
            f"{args.checkpoint_tokens}-{condition}"
        )

    if jax.process_index() == 0:
        result, paired = build_experiment5_evaluation_result(
            accumulators=global_accumulators,
            source_revision=args.source_revision,
            config_sha256=config.config_hash,
            completed_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        # Publish the evidence sidecar first; the result is the completion
        # marker and already binds the sidecar's artifact SHA.
        _create_only_json(output / "paired-scores.json", paired)
        _create_only_json(output / "scientific-result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    _collective_barrier(
        f"experiment5-eval-final-{args.panel}-{args.seed}-"
        f"{args.checkpoint_tokens}"
    )
    return 0


def _absolute_path(value: str) -> str:
    selected = Path(value).expanduser()
    if not selected.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return str(selected.resolve(strict=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("sae", "transcoder"), required=True)
    parser.add_argument("--config", required=True, type=_absolute_path)
    parser.add_argument("--accelerator-type", required=True, choices=tuple(_ACCELERATOR_GENERATION))
    parser.add_argument("--phase", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument(
        "--checkpoint-tokens",
        required=True,
        type=int,
        choices=(2_000_000, 5_000_000, 10_000_000),
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--train-target-root", required=True, type=_absolute_path)
    parser.add_argument("--eval-target-root", required=True, type=_absolute_path)
    parser.add_argument("--train-target-object-prefix", required=True)
    parser.add_argument("--eval-target-object-prefix", required=True)
    parser.add_argument("--train-source-root", type=_absolute_path)
    parser.add_argument("--eval-source-root", type=_absolute_path)
    parser.add_argument("--train-source-object-prefix")
    parser.add_argument("--eval-source-object-prefix")
    parser.add_argument("--cache-root", required=True, type=_absolute_path)
    parser.add_argument("--cache-max-bytes", required=True, type=int)
    parser.add_argument("--sae-control-root", required=True, type=_absolute_path)
    parser.add_argument("--checkpoint-payload-root", required=True)
    parser.add_argument("--recomposition-sidecar", required=True, type=_absolute_path)
    parser.add_argument("--result-root", required=True, type=_absolute_path)
    parser.add_argument("--process-index", required=True, type=int)
    parser.add_argument("--peak-flops-per-second", required=True, type=int)
    parser.add_argument("--regional-mirror-seal", type=_absolute_path)
    for condition in CONDITIONS:
        parser.add_argument(
            f"--run-root-{condition.lower()}",
            required=True,
            type=_absolute_path,
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(_parser().parse_args(argv))
    except Experiment5EvaluationError as error:
        raise Experiment5EvaluationWorkerError(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
