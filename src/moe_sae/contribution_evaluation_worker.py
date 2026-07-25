"""Multi-host final evaluator for the Experiment-2 mechanism pilot.

This is intentionally a practical execution boundary rather than another
accelerator admission layer.  Four already-provisioned TPU controllers invoke
the command concurrently.  It restores only validated ``final.json`` pointers,
evaluates A/R/B/F and Q on the same complete held-out contribution capture,
retains content-addressed per-sequence score rows, and lets process zero seal a
paired scientific result.

The result is sufficient for the Phase-2 decision:

* pooled weighted-contribution and exact recomposed-routed EV;
* paired packed-sequence intervals for A-R, A-B, and F-B;
* G2, C2, and the route-key effect;
* exact L0, feature coverage, checkpoint identity, and Q diagnostics.

The slower full-FFN/Q recomposition artifacts remain a downstream diagnostic;
they are not substituted for either primary endpoint here.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from moe_sae.checkpointing import pointer_path, validate_pointer
from moe_sae.contribution_artifacts import (
    bf16_bits_to_float32,
    canonical_json_bytes,
    canonical_sha256,
    float32_to_bf16_bits,
)
from moe_sae.contribution_config import load_contribution_config
from moe_sae.contribution_evaluation import (
    ContributionEvaluationEvidenceChunk,
    evaluate_contribution_input,
    routed_recomposition_token_scores,
)
from moe_sae.contribution_input import (
    CapacityReceipt,
    ContributionProcessInput,
    contiguous_process_range,
)
from moe_sae.contribution_recomposition import (
    decode_recomposition_sidecar,
    maxtext_rmsnorm_bf16,
)
from moe_sae.contribution_report import (
    PairedSequenceScores,
    paired_sequence_bootstrap,
)
from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_score_artifacts import (
    ScoreArtifactRef,
    resolve_score_manifest_matrix,
    seal_score_manifest,
    seal_score_row_shard,
)
from moe_sae.contribution_shared import (
    SharedQLayout,
    SharedQProcessInput,
    build_shared_q_centering,
    evaluate_shared_q_input,
    init_shared_q_state,
    shared_q_sae_config,
)
from moe_sae.contribution_storage import ContributionDataset
from moe_sae.contribution_training import (
    ContributionOptimizerConfig,
    optimizer_sae_config,
)
from moe_sae.contribution_training_worker import (
    TRAINING_RESULT_SCHEMA_VERSION,
    TrainingWorkerPlan,
    _collective_barrier,
    _collective_records,
    _pooled_routed_mean,
)
from moe_sae.distributed import (
    DistributedConfig,
    create_mesh,
    restore_distributed_checkpoint,
)
from moe_sae.object_store import GCSObjectStore, ShardCache
from moe_sae.pipeline import TrainingConfig
from moe_sae.runtime import initialize_jax_distributed, require_v4_32_runtime

CONDITIONS = ("A", "R", "B", "F")
ALL_CONDITIONS = (*CONDITIONS, "Q")
RESULT_SCHEMA_VERSION = "gemma4-contribution-phase2-scientific-result-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Experiment2EvaluationError(RuntimeError):
    """The final checkpoint/evaluation evidence is incomplete or inconsistent."""


def _read_json(path: str | Path, *, name: str) -> dict[str, Any]:
    selected = Path(path)
    try:
        value = json.loads(selected.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Experiment2EvaluationError(f"cannot read {name}: {selected}") from error
    if not isinstance(value, dict):
        raise Experiment2EvaluationError(f"{name} must be a JSON object")
    return value


def _create_only(path: str | Path, payload: bytes) -> Path:
    selected = Path(path)
    selected.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(selected, flags, 0o640)
    except FileExistsError:
        if selected.read_bytes() != payload:
            raise Experiment2EvaluationError(
                f"immutable evaluation artifact changed on replay: {selected}"
            )
        return selected
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        selected.unlink(missing_ok=True)
        raise
    return selected


def _write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    return _create_only(path, canonical_json_bytes(dict(value)))


class _ContentStore:
    """Small shared-POSIX content store for paired score evidence."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def retain(self, reference: Any, blob: bytes) -> None:
        digest = str(reference.sha256)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise Experiment2EvaluationError("score blob differs from its digest")
        _create_only(self.root / digest, blob)

    def __call__(self, reference: Any) -> bytes:
        path = self.root / str(reference.sha256)
        try:
            blob = path.read_bytes()
        except OSError as error:
            raise Experiment2EvaluationError(
                f"missing retained score artifact {reference.sha256}"
            ) from error
        if len(blob) != int(reference.size_bytes):
            raise Experiment2EvaluationError("retained score artifact size changed")
        if hashlib.sha256(blob).hexdigest() != reference.sha256:
            raise Experiment2EvaluationError("retained score artifact hash changed")
        return blob


def _validate_training_result(
    value: Mapping[str, Any],
    plan: TrainingWorkerPlan,
) -> dict[str, Any]:
    selected = dict(value)
    claimed = selected.pop("result_sha256", None)
    if not isinstance(claimed, str) or _SHA256_RE.fullmatch(claimed) is None:
        raise Experiment2EvaluationError("training result lacks a SHA-256")
    if canonical_sha256(selected) != claimed:
        raise Experiment2EvaluationError("training result self-hash is invalid")
    if selected.get("schema_version") != TRAINING_RESULT_SCHEMA_VERSION:
        raise Experiment2EvaluationError("training result schema is incompatible")
    if selected.get("status") not in {"completed", "already_complete"}:
        raise Experiment2EvaluationError("training run is not complete")
    if selected.get("steps") != plan.planned_updates:
        raise Experiment2EvaluationError("training result update count is not final")
    if selected.get("tokens") != plan.total_tokens:
        raise Experiment2EvaluationError("training result token count is not final")
    identity = selected.get("run_identity")
    if not isinstance(identity, Mapping):
        raise Experiment2EvaluationError("training result lacks its run identity")
    identity_body = dict(identity)
    identity_sha = identity_body.pop("run_identity_sha256", None)
    if (
        not isinstance(identity_sha, str)
        or canonical_sha256(identity_body) != identity_sha
    ):
        raise Experiment2EvaluationError("training run identity self-hash is invalid")
    expected_plan = {
        **dataclasses.asdict(plan),
        "checkpoint_tokens": list(plan.checkpoint_tokens),
    }
    if identity.get("plan") != expected_plan:
        raise Experiment2EvaluationError("training result belongs to another plan")
    return dict(value)


def _load_run(
    run_root: Path,
    plan: TrainingWorkerPlan,
    checkpoint_payload_root: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _validate_training_result(
        _read_json(run_root / "training-result.json", name="training result"),
        plan,
    )
    final_path = pointer_path(run_root / "checkpoints", "final")
    pointer = validate_pointer(
        _read_json(final_path, name="final checkpoint pointer"),
        expected_kind="final",
        payload_root=checkpoint_payload_root,
        run_identity=result["run_identity"],
    )
    if (
        pointer["step"] != plan.planned_updates
        or pointer["tokens_seen"] != plan.total_tokens
    ):
        raise Experiment2EvaluationError("final pointer is not at the phase boundary")
    return result, pointer


def _optimizer(config: Any, plan: TrainingWorkerPlan) -> ContributionOptimizerConfig:
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


def _layout(config: Any, plan: TrainingWorkerPlan) -> ContributionSAEConfig:
    matches = [
        item
        for item in config.layouts_for_phase(plan.phase_name)
        if item.condition == plan.condition
    ]
    if len(matches) != 1:
        raise Experiment2EvaluationError("condition has no unique phase layout")
    selected = matches[0]
    return ContributionSAEConfig(
        condition=plan.condition,
        hidden_dim=config.model.hidden_size,
        num_experts=config.model.num_routed_experts,
        top_routed=config.model.experts_per_token,
        bank_width=selected.bank_width,
        k_per_contribution=selected.topk_per_component,
        randomization_key=plan.randomization_key,
    )


def _training_config(
    plan: TrainingWorkerPlan,
    sae: Any,
    *,
    bank_capacity: int | None,
) -> TrainingConfig:
    return TrainingConfig(
        sae=sae,
        seed=plan.seed,
        total_tokens=plan.total_tokens,
        effective_batch_tokens=plan.effective_batch_tokens,
        microbatch_tokens=plan.global_microbatch_tokens,
        checkpoint_every_tokens=plan.total_tokens,
        health_gate_tokens=plan.total_tokens,
        a_capacity=bank_capacity,
    )


def _checkpoint_a_capacity(pointer: Mapping[str, Any]) -> int | None:
    """Read the immutable training capacity needed to restore an A/R state."""

    try:
        import orbax.checkpoint as ocp
    except ImportError as error:  # pragma: no cover - TPU-only dependency
        raise Experiment2EvaluationError(
            "checkpoint metadata inspection requires the TPU extra"
        ) from error
    checkpointer = ocp.StandardCheckpointer()
    try:
        custom = checkpointer.metadata(pointer["payload_uri"]).custom_metadata or {}
    finally:
        checkpointer.close()
    training = custom.get("training_config")
    if not isinstance(training, Mapping):
        raise Experiment2EvaluationError(
            "checkpoint lacks its immutable training configuration"
        )
    value = training.get("a_capacity")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Experiment2EvaluationError("checkpoint A/R capacity is invalid")
    return value


def _score_payload(result: Any) -> dict[str, Any]:
    return {
        **dataclasses.asdict(result),
        "component_nmse": result.component_nmse,
        "component_ev": result.component_ev,
        "receipt_sha256": result.receipt_sha256,
    }


def _q_payload(result: Any) -> dict[str, Any]:
    return {
        **dataclasses.asdict(result),
        "nmse": result.nmse,
        "ev": result.ev,
        "receipt_sha256": result.receipt_sha256,
    }


def _effect_payload(value: Any) -> dict[str, Any]:
    return dataclasses.asdict(value)


def _training_routed_mean(
    dataset: ContributionDataset,
    sidecar: Mapping[str, Any],
) -> tuple[np.ndarray, str]:
    """Compute the exact post-routed-RMSNorm training mean in parallel."""

    import jax
    from jax.experimental import multihost_utils

    start, end = contiguous_process_range(
        int(dataset.total_tokens),
        process_count=int(jax.process_count()),
        process_index=int(jax.process_index()),
    )
    local_sum = np.zeros((2_816,), dtype=np.float64)
    local_count = 0
    decoded_sidecar = decode_recomposition_sidecar(sidecar)
    for batch in dataset.iter_token_range(start, end):
        expert = bf16_bits_to_float32(batch.expert_output)
        weights = bf16_bits_to_float32(batch.router_weights)
        raw = np.einsum(
            "nkh,nk->nh",
            expert,
            weights,
            dtype=np.float32,
            optimize=False,
        )
        _, post_bits = maxtext_rmsnorm_bf16(
            float32_to_bf16_bits(raw),
            decoded_sidecar.routed_rmsnorm_scale_bf16_bits,
            epsilon=float(decoded_sidecar.epsilon),
        )
        target = bf16_bits_to_float32(post_bits).astype(np.float64)
        local_sum += target.sum(axis=0, dtype=np.float64)
        local_count += int(target.shape[0])
    if local_count != end - start:
        raise Experiment2EvaluationError(
            "routed centering did not consume its exact training lane"
        )
    # JAX disables x64 by default on TPU.  Preserve the FP64 host accumulation
    # across the multihost collective as an exact high/residual-low FP32 pair
    # instead of silently downcasting the sum at the device boundary.
    local_sum_high = local_sum.astype(np.float32)
    local_sum_low = (local_sum - local_sum_high.astype(np.float64)).astype(np.float32)
    gathered_sum_high = np.asarray(
        multihost_utils.process_allgather(local_sum_high),
        dtype=np.float32,
    ).reshape(-1, 2_816)
    gathered_sum_low = np.asarray(
        multihost_utils.process_allgather(local_sum_low),
        dtype=np.float32,
    ).reshape(-1, 2_816)
    gathered_sum = gathered_sum_high.astype(np.float64) + gathered_sum_low.astype(
        np.float64
    )
    gathered_count = np.asarray(
        multihost_utils.process_allgather(np.asarray(local_count, np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    if int(gathered_count.sum()) != int(dataset.total_tokens):
        raise Experiment2EvaluationError(
            "routed centering token count differs from the training capture"
        )
    mean = np.ascontiguousarray(
        (gathered_sum.sum(axis=0, dtype=np.float64) / int(dataset.total_tokens))
        .astype(np.float32)
    )
    digest = hashlib.sha256()
    digest.update(b"gemma4-routed-post-norm-centering-mean-v1\0")
    digest.update(mean.astype("<f4", copy=False).tobytes(order="C"))
    return mean, digest.hexdigest()


def _evaluate_condition_endpoints(
    params: Any,
    source: ContributionProcessInput,
    component_mean: Any,
    routed_mean: Any,
    mesh: Any,
    *,
    sidecar: Mapping[str, Any],
    checkpoint_sha256: str,
    capacity_receipts: Sequence[CapacityReceipt],
    distributed: DistributedConfig,
    paired_row_sink: Callable[[Any, Any], None],
) -> Any:
    """Keep the two scientifically distinct centering means at one seam."""

    def score_rows(evidence: ContributionEvaluationEvidenceChunk) -> None:
        routed = routed_recomposition_token_scores(
            evidence.scores,
            sidecar,
            routed_mean,
        )
        paired_row_sink(evidence, routed)

    return evaluate_contribution_input(
        params,
        source,
        component_mean,
        mesh,
        checkpoint_sha256=checkpoint_sha256,
        capacity_receipts=capacity_receipts,
        distributed=distributed,
        row_sink=score_rows,
    )


def _paired_scores(
    matrix: Mapping[tuple[int, str], Any],
    *,
    seed: int,
    endpoint: str,
) -> PairedSequenceScores:
    first = matrix[(seed, "A")]
    sst_name = f"{endpoint}_sst"
    sse_name = f"{endpoint}_sse"
    return PairedSequenceScores(
        seed=seed,
        endpoint=endpoint,  # type: ignore[arg-type]
        capture_manifest_sha256=first.evaluation_capture_manifest_sha256,
        token_identity_sha256=first.token_identity_sha256,
        sequence_id=first.sequence_id,
        token_count=first.token_count,
        common_sst=getattr(first, sst_name),
        condition_sse={
            condition: getattr(matrix[(seed, condition)], sse_name)
            for condition in CONDITIONS
        },
    )


def run(args: argparse.Namespace) -> int:
    import jax

    config = load_contribution_config(args.config)
    plans = {
        condition: TrainingWorkerPlan.from_config(
            config,
            phase=args.phase,
            condition=condition,
            seed=args.seed,
        )
        for condition in ALL_CONDITIONS
    }
    initialize_jax_distributed()
    require_v4_32_runtime()
    if jax.process_index() != args.process_index:
        raise Experiment2EvaluationError(
            "physical controller/JAX process mapping changed"
        )
    distributed = DistributedConfig(
        data_axis_size=int(jax.process_count()),
        feature_axis_size=int(jax.local_device_count()),
        microbatches_per_update=(
            config.runtime.effective_batch_tokens
            // config.runtime.microbatch_tokens
        ),
    )
    mesh = create_mesh(distributed)
    sidecar = _read_json(args.recomposition_sidecar, name="recomposition sidecar")
    sidecar_identity = decode_recomposition_sidecar(sidecar)
    cache = ShardCache(
        Path(args.cache_root) / f"process-p{jax.process_index():05d}",
        GCSObjectStore(),
        max_bytes=args.cache_max_bytes,
    )
    train_dataset = ContributionDataset(
        args.train_capture_root,
        object_prefix=args.train_object_prefix,
        cache=cache,
        recomposition_sidecar=sidecar,
    )
    eval_dataset = ContributionDataset(
        args.eval_capture_root,
        object_prefix=args.eval_object_prefix,
        cache=cache,
        recomposition_sidecar=sidecar,
    )
    phase = config.phase(args.phase)
    if (
        int(train_dataset.total_tokens) != phase.train_tokens
        or int(eval_dataset.total_tokens) != phase.eval_tokens
    ):
        raise Experiment2EvaluationError("capture token counts differ from the phase")
    output = Path(args.result_root).expanduser().resolve()
    store = _ContentStore(output / "objects")
    runs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for condition, plan in plans.items():
        run_root = (
            Path(args.sae_control_root)
            / "runs"
            / f"phase2-{condition.lower()}-seed{args.seed}"
        )
        runs[condition] = _load_run(
            run_root,
            plan,
            args.checkpoint_payload_root,
        )

    # Q is evaluated first.  It is common diagnostic evidence and does not
    # enter G2/C2.
    q_plan = plans["Q"]
    q_centering = build_shared_q_centering(train_dataset, sidecar)
    q_source = SharedQProcessInput(
        eval_dataset,
        sidecar,
        q_centering,
        role="evaluation",
        total_tokens=phase.eval_tokens,
        global_microbatch_tokens=q_plan.global_microbatch_tokens,
        microbatches_per_update=q_plan.microbatches_per_update,
        process_count=int(jax.process_count()),
        process_index=int(jax.process_index()),
    )
    q_layout = next(
        item
        for item in config.layouts_for_phase(args.phase)
        if item.condition == "Q"
    )
    q_config = shared_q_sae_config(
        SharedQLayout(
            hidden_dim=config.model.hidden_size,
            total_features=q_layout.bank_width,
            k=q_layout.topk_per_component,
        ),
        parameter_dtype=config.optimizer.parameter_dtype,
        adam_beta1=config.optimizer.adam_beta1,
        adam_beta2=config.optimizer.adam_beta2,
        adam_eps=config.optimizer.adam_epsilon,
        grad_clip=config.optimizer.global_gradient_clip,
        peak_lr=config.optimizer.peak_learning_rate,
        min_lr=config.optimizer.minimum_learning_rate,
        warmup_steps=q_plan.warmup_updates,
        total_steps=q_plan.planned_updates,
    )
    q_training_config = _training_config(q_plan, q_config, bank_capacity=None)
    q_state = init_shared_q_state(
        jax.random.key(args.seed),
        mesh,
        q_config,
        distributed,
    )
    q_pointer = runs["Q"][1]
    q_state, _ = restore_distributed_checkpoint(
        pointer_path(
            Path(args.sae_control_root)
            / "runs"
            / f"phase2-q-seed{args.seed}"
            / "checkpoints",
            "final",
        ),
        q_state,
        mesh,
        q_training_config,
        checkpoint_payload_root=args.checkpoint_payload_root,
        checkpoint_identity=runs["Q"][0]["run_identity"],
        checkpoint_metadata_identity={
            "run_identity_sha256": runs["Q"][0]["run_identity"][
                "run_identity_sha256"
            ]
        },
    )
    q_result = evaluate_shared_q_input(
        q_state.params,
        q_source,
        mesh,
        q_config,
        checkpoint_sha256=q_pointer["pointer_sha256"],
        distributed=distributed,
    )
    lane_root = output / f"process-p{jax.process_index():05d}"
    _write_json(lane_root / "q-result.json", _q_payload(q_result))
    _write_json(lane_root / "q-input-receipt.json", dataclasses.asdict(q_source.receipt))
    del q_state
    jax.clear_caches()
    _collective_barrier(f"experiment2-eval-q-{args.seed}")

    component_mean, component_mean_sha = _pooled_routed_mean(train_dataset)
    routed_mean, routed_mean_sha = _training_routed_mean(train_dataset, sidecar)
    endpoint_centering_sha = canonical_sha256(
        {
            "schema_version": "gemma4-contribution-endpoint-centering-v1",
            "component_mean_sha256": component_mean_sha,
            "routed_mean_sha256": routed_mean_sha,
        }
    )
    local_results: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        plan = plans[condition]
        layout = _layout(config, plan)
        source = ContributionProcessInput(
            eval_dataset,
            layout,
            total_tokens=phase.eval_tokens,
            global_microbatch_tokens=plan.global_microbatch_tokens,
            microbatches_per_update=plan.microbatches_per_update,
            process_count=int(jax.process_count()),
            process_index=int(jax.process_index()),
        )
        capacity_receipts: tuple[CapacityReceipt, ...] = ()
        if condition in {"A", "R"}:
            local_capacity = source.scan_local_capacity(
                margin_fraction=config.runtime.capacity_margin_fraction
            )
            capacity_receipts = tuple(
                CapacityReceipt(**item)
                for item in _collective_records(dataclasses.asdict(local_capacity))
            )
        optimizer = _optimizer(config, plan)
        training_source = ContributionProcessInput(
            train_dataset,
            layout,
            total_tokens=phase.train_tokens,
            global_microbatch_tokens=plan.global_microbatch_tokens,
            microbatches_per_update=plan.microbatches_per_update,
            process_count=int(jax.process_count()),
            process_index=int(jax.process_index()),
        )
        training_receipts = _collective_records(
            dataclasses.asdict(training_source.receipt)
        )
        pointer = runs[condition][1]
        training_capacity = _checkpoint_a_capacity(pointer)
        if (condition in {"A", "R"}) != (training_capacity is not None):
            raise Experiment2EvaluationError(
                f"{condition} checkpoint capacity differs from its layout"
            )
        train_config = _training_config(
            plan,
            optimizer_sae_config(layout, optimizer),
            bank_capacity=training_capacity,
        )
        from moe_sae.contribution_distributed import (
            init_contribution_distributed_state,
        )

        state = init_contribution_distributed_state(
            jax.random.key(args.seed),
            mesh,
            layout,
            optimizer,
            distributed,
        )
        run_root = (
            Path(args.sae_control_root)
            / "runs"
            / f"phase2-{condition.lower()}-seed{args.seed}"
        )
        state, _ = restore_distributed_checkpoint(
            pointer_path(run_root / "checkpoints", "final"),
            state,
            mesh,
            train_config,
            checkpoint_payload_root=args.checkpoint_payload_root,
            checkpoint_identity=runs[condition][0]["run_identity"],
            checkpoint_metadata_identity={
                "run_identity_sha256": runs[condition][0]["run_identity"][
                    "run_identity_sha256"
                ]
            },
        )
        row_refs: list[ScoreArtifactRef] = []

        def retain_row(
            evidence: ContributionEvaluationEvidenceChunk,
            routed: Any,
            *,
            selected_condition: str = condition,
            selected_pointer: Mapping[str, Any] = pointer,
            selected_source: ContributionProcessInput = source,
            selected_refs: list[ScoreArtifactRef] = row_refs,
        ) -> None:
            reference, blob = seal_score_row_shard(
                evidence,
                routed,
                condition=selected_condition,
                seed=args.seed,
                process_index=int(jax.process_index()),
                evaluation_capture_manifest_sha256=eval_dataset.manifest_sha256,
                checkpoint_sha256=selected_pointer["pointer_sha256"],
                recomposition_sidecar_sha256=(
                    routed.recomposition_sidecar_sha256
                ),
                centering_mean_sha256=endpoint_centering_sha,
                evaluation_input_receipt_sha256=(
                    selected_source.receipt.receipt_sha256
                ),
            )
            store.retain(reference, blob)
            selected_refs.append(reference)

        result = _evaluate_condition_endpoints(
            state.params,
            source,
            component_mean,
            routed_mean,
            mesh,
            sidecar=sidecar,
            checkpoint_sha256=pointer["pointer_sha256"],
            capacity_receipts=capacity_receipts,
            distributed=distributed,
            paired_row_sink=retain_row,
        )
        local_results[condition] = _score_payload(result)
        _write_json(lane_root / f"{condition}-result.json", local_results[condition])
        _write_json(
            lane_root / f"{condition}-input-receipt.json",
            dataclasses.asdict(source.receipt),
        )
        _write_json(
            lane_root / f"{condition}-row-refs.json",
            {"references": [item.to_dict() for item in row_refs]},
        )
        del state
        jax.clear_caches()
        _collective_barrier(f"experiment2-eval-{condition}-{args.seed}")

        if jax.process_index() == 0:
            all_refs: list[ScoreArtifactRef] = []
            eval_receipts: list[dict[str, Any]] = []
            for process_index in range(jax.process_count()):
                lane = output / f"process-p{process_index:05d}"
                rows = _read_json(
                    lane / f"{condition}-row-refs.json",
                    name=f"{condition} row references",
                )
                all_refs.extend(
                    ScoreArtifactRef.from_value(item)
                    for item in rows["references"]
                )
                eval_receipts.append(
                    _read_json(
                        lane / f"{condition}-input-receipt.json",
                        name=f"{condition} evaluation input receipt",
                    )
                )
            manifest_ref, manifest_blob = seal_score_manifest(
                all_refs,
                store,
                phase=args.phase,
                protocol_sha256=config.config_hash,
                condition=condition,
                seed=args.seed,
                randomization_key=plan.randomization_key,
                training_capture_manifest_sha256=train_dataset.manifest_sha256,
                evaluation_capture_manifest_sha256=eval_dataset.manifest_sha256,
                recomposition_sidecar_sha256=sidecar_identity.sidecar_sha256,
                run_manifest_sha256=runs[condition][0]["result_sha256"],
                config_sha256=config.config_hash,
                checkpoint_sha256=pointer["pointer_sha256"],
                layout_sha256=canonical_sha256(dataclasses.asdict(layout)),
                centering_mean_sha256=endpoint_centering_sha,
                training_input_receipt_sha256s=[
                    item["receipt_sha256"] for item in training_receipts
                ],
                evaluation_input_receipts=eval_receipts,
                expected_tokens=phase.eval_tokens,
            )
            store.retain(manifest_ref, manifest_blob)
            _write_json(
                output / f"{condition}-score-manifest-ref.json",
                manifest_ref.to_dict(),
            )
        _collective_barrier(f"experiment2-eval-manifest-{condition}-{args.seed}")

    if jax.process_index() == 0:
        references = [
            ScoreArtifactRef.from_value(
                _read_json(
                    output / f"{condition}-score-manifest-ref.json",
                    name=f"{condition} score manifest reference",
                )
            )
            for condition in CONDITIONS
        ]
        matrix = resolve_score_manifest_matrix(
            references,
            store,
            expected_seeds=[args.seed],
        )
        paired = {
            endpoint: _paired_scores(matrix, seed=args.seed, endpoint=endpoint)
            for endpoint in ("component", "routed")
        }
        effects = {
            endpoint: {
                f"{left}-{right}": _effect_payload(
                    paired_sequence_bootstrap(
                        paired[endpoint],
                        left=left,
                        right=right,
                        replicates=args.bootstrap_replicates,
                    )
                )
                for left, right in (("A", "R"), ("A", "B"), ("F", "B"))
            }
            for endpoint in ("component", "routed")
        }
        endpoints: dict[str, dict[str, Any]] = {}
        for endpoint in ("component", "routed"):
            selected = paired[endpoint]
            total_sst = float(selected.common_sst.sum(dtype=np.float64))
            endpoints[endpoint] = {
                condition: {
                    "sse": float(
                        selected.condition_sse[condition].sum(dtype=np.float64)
                    ),
                    "sst": total_sst,
                    "nmse": float(
                        selected.condition_sse[condition].sum(dtype=np.float64)
                        / total_sst
                    ),
                    "ev": float(
                        1.0
                        - selected.condition_sse[condition].sum(dtype=np.float64)
                        / total_sst
                    ),
                }
                for condition in CONDITIONS
            }
        component = endpoints["component"]
        denominator = component["F"]["ev"] - component["B"]["ev"]
        denominator_effect = effects["component"]["F-B"]
        denominator_interval_includes_zero = (
            float(denominator_effect["ci95_low"])
            <= 0.0
            <= float(denominator_effect["ci95_high"])
        )
        g2 = (
            None
            if denominator <= 0
            else 100.0
            * (component["A"]["ev"] - component["B"]["ev"])
            / denominator
        )
        costs = {
            condition: float(runs[condition][0]["chip_seconds"])
            for condition in CONDITIONS
        }
        c2 = 100.0 * costs["A"] / costs["F"]
        q_rows = [
            _read_json(
                output / f"process-p{process_index:05d}" / "q-result.json",
                name="Q evaluation result",
            )
            for process_index in range(jax.process_count())
        ]
        q_summary = dict(q_rows[0])
        q_summary["input_receipt_sha256s"] = sorted(
            row["input_receipt_sha256"] for row in q_rows
        )
        q_summary["local_sufficient_rows_total"] = sum(
            int(row["local_sufficient_rows"]) for row in q_rows
        )
        for name in (
            "input_receipt_sha256",
            "local_sufficient_rows",
            "receipt_sha256",
        ):
            q_summary.pop(name)
        for row in q_rows[1:]:
            comparable = dict(row)
            for name in (
                "input_receipt_sha256",
                "local_sufficient_rows",
                "receipt_sha256",
            ):
                comparable.pop(name)
            if comparable != {
                key: value
                for key, value in q_summary.items()
                if key
                not in {
                    "input_receipt_sha256s",
                    "local_sufficient_rows_total",
                }
            }:
                raise Experiment2EvaluationError(
                    "global Q metrics differ across processes"
                )
        report_body = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "phase": args.phase,
            "seed": args.seed,
            "training_capture_manifest_sha256": train_dataset.manifest_sha256,
            "evaluation_capture_manifest_sha256": eval_dataset.manifest_sha256,
            "evaluation_tokens": phase.eval_tokens,
            "endpoint_centering": {
                "component_mean_sha256": component_mean_sha,
                "routed_mean_sha256": routed_mean_sha,
                "contract_sha256": endpoint_centering_sha,
            },
            "checkpoint_sha256s": {
                condition: runs[condition][1]["pointer_sha256"]
                for condition in ALL_CONDITIONS
            },
            "score_manifest_references": {
                condition: reference.to_dict()
                for condition, reference in zip(CONDITIONS, references, strict=True)
            },
            "endpoints": endpoints,
            "paired_sequence_effects": effects,
            "primary": {
                "g2_percent": g2,
                "g2_defined": g2 is not None,
                "g2_stable": (
                    g2 is not None
                    and not denominator_interval_includes_zero
                ),
                "f_minus_b_interval_includes_zero": (
                    denominator_interval_includes_zero
                ),
                "c2_percent": c2,
                "delta_route_ev": (
                    component["A"]["ev"] - component["R"]["ev"]
                ),
                "component_a_minus_b_positive": (
                    component["A"]["ev"] > component["B"]["ev"]
                ),
                "routed_a_minus_b_positive": (
                    endpoints["routed"]["A"]["ev"]
                    > endpoints["routed"]["B"]["ev"]
                ),
            },
            "training_chip_seconds": costs,
            "condition_evaluations": local_results,
            "shared_q": q_summary,
            "bootstrap_replicates": args.bootstrap_replicates,
        }
        report = {
            **report_body,
            "result_sha256": canonical_sha256(report_body),
        }
        _write_json(output / "scientific-result.json", report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    _collective_barrier(f"experiment2-eval-final-{args.seed}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", default="phase2_mechanism_pilot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-capture-root", required=True)
    parser.add_argument("--eval-capture-root", required=True)
    parser.add_argument("--train-object-prefix", required=True)
    parser.add_argument("--eval-object-prefix", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cache-max-bytes", type=int, required=True)
    parser.add_argument("--sae-control-root", required=True)
    parser.add_argument("--checkpoint-payload-root", required=True)
    parser.add_argument("--recomposition-sidecar", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
