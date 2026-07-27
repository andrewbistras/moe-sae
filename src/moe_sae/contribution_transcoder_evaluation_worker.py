"""Multi-host final evaluator for the Experiment-3 transcoder pilot.

Four TPU controllers invoke this module concurrently.  Each controller
restores the same feature-sharded A/R/B/F checkpoint, scores its contiguous
held-out source/target lane, and writes one immutable sufficient-row artifact.
Process zero then joins the four conditions by exact token key, recomputes all
paired intervals, and seals the scientific result.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load as load_safetensors
from safetensors.numpy import save as save_safetensors

from moe_sae.checkpointing import pointer_path
from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    TOP_K_EXPERTS,
    bf16_bits_to_float32,
    canonical_json_bytes,
    canonical_sha256,
)
from moe_sae.contribution_evaluation_worker import (
    _checkpoint_a_capacity,
    _layout,
    _load_run,
    _optimizer,
    _training_config,
)
from moe_sae.contribution_evaluation import _validate_capacity_receipt
from moe_sae.contribution_input import (
    CapacityReceipt,
    contiguous_process_range,
    merge_capacity_receipts,
)
from moe_sae.contribution_recomposition import (
    decode_recomposition_sidecar,
    recompose_gemma4,
)
from moe_sae.contribution_storage import ContributionDataset
from moe_sae.contribution_training_performance import (
    SAEArithmetic,
    sae_training_useful_flops,
)
from moe_sae.contribution_training import optimizer_sae_config
from moe_sae.contribution_training_worker import (
    TrainingWorkerPlan,
    _collective_barrier,
    _collective_records,
    _pooled_routed_mean,
)
from moe_sae.contribution_transcoder_config import (
    load_contribution_transcoder_config,
)
from moe_sae.contribution_transcoder_distributed import (
    init_contribution_transcoder_distributed_state,
)
from moe_sae.contribution_transcoder_evaluation import (
    CONDITIONS,
    ContributionTranscoderDistributedEvaluator,
    ContributionTranscoderTokenScores,
    TranscoderPairedSequenceScores,
    aggregate_paired_sequence_scores,
    expert_rank_diagnostics,
    score_transcoder_sufficient_rows,
)
from moe_sae.contribution_transcoder_input import (
    ContributionTranscoderProcessInput,
    routed_input_mean,
)
from moe_sae.contribution_transcoder_report import (
    BOOTSTRAP_REPLICATES,
    TranscoderTrainingLedger,
    build_condition_summary,
    build_transcoder_scientific_result,
    validate_transcoder_scientific_result,
    write_transcoder_result,
)
from moe_sae.contribution_transcoder_storage import RoutedInputDataset
from moe_sae.distributed import (
    DistributedConfig,
    create_mesh,
    restore_distributed_checkpoint,
)
from moe_sae.object_store import GCSObjectStore, ShardCache
from moe_sae.runtime import initialize_jax_distributed, require_v4_32_runtime

EVIDENCE_SCHEMA_VERSION = (
    "gemma4-contribution-transcoder-process-evaluation-v1"
)
ENDPOINT_CENTERING_SCHEMA_VERSION = (
    "gemma4-contribution-transcoder-endpoint-centering-v1"
)
COMMON_SST_SCHEMA_VERSION = (
    "gemma4-contribution-transcoder-common-sst-v1"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_EVIDENCE_TENSORS = {
    "sequence_id",
    "position",
    "expert_ids",
    "component_sse",
    "component_sst",
    "routed_sse",
    "routed_sst",
    "oracle_shared_sse",
    "oracle_shared_sst",
    "route_component_sse",
    "route_component_sst",
}


class Experiment3EvaluationWorkerError(RuntimeError):
    """Final Experiment-3 evaluation cannot produce authoritative evidence."""


@dataclasses.dataclass(frozen=True, slots=True)
class _EvaluationCapacityRoles:
    """Keep checkpoint identity separate from held-out execution shape."""

    checkpoint: int | None
    execution: int | None


def _evaluation_capacity_roles(
    condition: str,
    *,
    training_capacity: int | None,
    evaluation_capacity: int | None,
) -> _EvaluationCapacityRoles:
    """Return immutable-restore and execution-only capacities.

    A/R checkpoint metadata hashes the capacity used during training. Held-out
    data may require a larger grouped-GEMM shape, but that execution capacity
    is not part of checkpoint identity and must never be supplied to restore.
    """

    if condition not in CONDITIONS:
        raise ValueError(f"unsupported transcoder condition: {condition!r}")
    if condition not in {"A", "R"}:
        if training_capacity is not None or evaluation_capacity is not None:
            raise ValueError("B/F capacity roles must both be absent")
        return _EvaluationCapacityRoles(checkpoint=None, execution=None)
    for name, value in (
        ("training_capacity", training_capacity),
        ("evaluation_capacity", evaluation_capacity),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer for A/R")
    assert training_capacity is not None and evaluation_capacity is not None
    return _EvaluationCapacityRoles(
        checkpoint=training_capacity,
        execution=max(training_capacity, evaluation_capacity),
    )


def _read_json(path: str | Path, *, name: str) -> dict[str, Any]:
    selected = Path(path)
    try:
        value = json.loads(selected.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Experiment3EvaluationWorkerError(
            f"cannot read {name}: {selected}"
        ) from error
    if not isinstance(value, dict):
        raise Experiment3EvaluationWorkerError(f"{name} must be a JSON object")
    return value


def _create_only(path: str | Path, payload: bytes) -> Path:
    selected = Path(path)
    selected.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(
            selected,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o640,
        )
    except FileExistsError:
        try:
            observed = selected.read_bytes()
        except OSError as error:
            raise Experiment3EvaluationWorkerError(
                f"cannot replay immutable artifact: {selected}"
            ) from error
        if observed != payload:
            raise Experiment3EvaluationWorkerError(
                f"immutable artifact differs on replay: {selected}"
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
    return _create_only(
        path,
        canonical_json_bytes(dict(value)) + b"\n",
    )


def _self_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return {**unsigned, field: canonical_sha256(unsigned)}


def _condition_run_root(args: argparse.Namespace, condition: str) -> Path:
    explicit = getattr(args, f"run_root_{condition.lower()}")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (
        Path(args.sae_control_root).expanduser().resolve()
        / "runs"
        / f"experiment3-{condition.lower()}-seed{args.seed}"
    )


def _validate_run_contract(
    run_root: Path,
    *,
    condition: str,
    source_manifest_sha256: str,
    target_manifest_sha256: str,
    training_source_revision: str,
) -> dict[str, Any]:
    contract = _read_json(run_root / "run-contract.json", name="run contract")
    claimed = contract.get("run_contract_sha256")
    if (
        not isinstance(claimed, str)
        or _SHA256_RE.fullmatch(claimed) is None
        or canonical_sha256(
            {
                name: value
                for name, value in contract.items()
                if name != "run_contract_sha256"
            }
        )
        != claimed
    ):
        raise Experiment3EvaluationWorkerError(
            "training run contract self-hash is invalid"
        )
    if (
        contract.get("scientific_model") != "transcoder"
        or contract.get("condition") != condition
        or contract.get("seed") != 0
        or contract.get("project_commit") != training_source_revision
        or contract.get("source_overlay_manifest_sha256")
        != source_manifest_sha256
        or contract.get("capture_manifest_sha256")
        != target_manifest_sha256
    ):
        raise Experiment3EvaluationWorkerError(
            "training run contract does not prove the selected source-only run"
        )
    return contract


def _endpoint_training_means(
    dataset: ContributionDataset,
    sidecar: Mapping[str, Any],
) -> tuple[np.ndarray, str, np.ndarray, str, dict[str, Any]]:
    """Compute routed and oracle-shared training means in one exact pass."""

    import jax
    from jax.experimental import multihost_utils

    start, end = contiguous_process_range(
        int(dataset.total_tokens),
        process_count=int(jax.process_count()),
        process_index=int(jax.process_index()),
    )
    routed_sum = np.zeros((HIDDEN_SIZE,), dtype=np.float64)
    oracle_sum = np.zeros((HIDDEN_SIZE,), dtype=np.float64)
    local_count = 0
    for batch in dataset.iter_token_range(start, end):
        # Keep the host oracle's largest FP32 intermediate below roughly
        # 370 MB instead of expanding an entire 16,384-token target shard.
        for offset in range(0, int(batch.tokens), 4_096):
            stop = min(offset + 4_096, int(batch.tokens))
            selected = dataclasses.replace(
                batch,
                **{
                    field.name: np.ascontiguousarray(
                        np.asarray(getattr(batch, field.name))[offset:stop]
                    )
                    for field in dataclasses.fields(batch)
                },
            )
            stages = recompose_gemma4(selected, sidecar)
            routed = bf16_bits_to_float32(
                stages.routed_post_norm_bf16_bits
            ).astype(np.float64)
            oracle = bf16_bits_to_float32(
                stages.outer_ffn_post_norm_bf16_bits
            ).astype(np.float64)
            routed_sum += routed.sum(axis=0, dtype=np.float64)
            oracle_sum += oracle.sum(axis=0, dtype=np.float64)
            local_count += int(selected.tokens)
    if local_count != end - start:
        raise Experiment3EvaluationWorkerError(
            "endpoint centering did not consume its complete process lane"
        )

    def gather_sum(value: np.ndarray) -> np.ndarray:
        high = value.astype(np.float32)
        low = (value - high.astype(np.float64)).astype(np.float32)
        gathered_high = np.asarray(
            multihost_utils.process_allgather(high),
            dtype=np.float32,
        ).reshape(-1, HIDDEN_SIZE)
        gathered_low = np.asarray(
            multihost_utils.process_allgather(low),
            dtype=np.float32,
        ).reshape(-1, HIDDEN_SIZE)
        return (gathered_high.astype(np.float64) + gathered_low.astype(np.float64)).sum(
            axis=0,
            dtype=np.float64,
        )

    count = int(
        np.asarray(
            multihost_utils.process_allgather(
                np.asarray(local_count, dtype=np.int32)
            ),
            dtype=np.int32,
        ).sum(dtype=np.int64)
    )
    if count != int(dataset.total_tokens):
        raise Experiment3EvaluationWorkerError(
            "endpoint centering collective token count changed"
        )
    routed_mean = np.ascontiguousarray(
        (gather_sum(routed_sum) / count).astype(np.float32)
    )
    oracle_mean = np.ascontiguousarray(
        (gather_sum(oracle_sum) / count).astype(np.float32)
    )

    def digest(domain: bytes, value: np.ndarray) -> str:
        selected = hashlib.sha256()
        selected.update(domain + b"\0")
        selected.update(value.astype("<f4", copy=False).tobytes(order="C"))
        return selected.hexdigest()

    routed_sha = digest(
        b"gemma4-transcoder-routed-endpoint-mean-v1",
        routed_mean,
    )
    oracle_sha = digest(
        b"gemma4-transcoder-oracle-shared-endpoint-mean-v1",
        oracle_mean,
    )
    unsigned = {
        "schema_version": ENDPOINT_CENTERING_SCHEMA_VERSION,
        "training_target_manifest_sha256": dataset.manifest_sha256,
        "recomposition_sidecar_sha256": decode_recomposition_sidecar(
            sidecar
        ).sidecar_sha256,
        "tokens": count,
        "routed_mean_sha256": routed_sha,
        "oracle_shared_mean_sha256": oracle_sha,
        "reduction_dtype": "float64_high_residual_low_float32_collective",
    }
    return (
        routed_mean,
        routed_sha,
        oracle_mean,
        oracle_sha,
        {**unsigned, "receipt_sha256": canonical_sha256(unsigned)},
    )


def _target_auxiliary(
    source: ContributionTranscoderProcessInput,
    *,
    update_index: int,
    microbatches: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Load only shared output and true IDs for the process-local score rows."""

    shared = np.zeros(
        (
            len(microbatches),
            source.local_rows,
            HIDDEN_SIZE,
        ),
        dtype=np.uint16,
    )
    expert_ids = np.zeros(
        (
            len(microbatches),
            source.local_rows,
            TOP_K_EXPERTS,
        ),
        dtype=np.uint8,
    )
    for local_index, microbatch in enumerate(microbatches):
        valid = np.asarray(microbatch.valid_mask)
        count = int(np.count_nonzero(valid))
        if count == 0:
            continue
        microbatch_index = (
            update_index * source.microbatches_per_update + local_index
        )
        start = source.process_range_start + microbatch_index * source.local_rows
        end = start + count
        try:
            shared_pieces = tuple(
                source.target_dataset.iter_shared_token_range(start, end)
            )
            id_pieces = tuple(
                source.target_dataset.iter_capacity_token_range(
                    start,
                    end,
                    include_sequence_id=False,
                )
            )
        except Exception as error:
            raise Experiment3EvaluationWorkerError(
                f"cannot read held-out target auxiliaries [{start},{end})"
            ) from error
        selected_shared = np.ascontiguousarray(
            np.concatenate(
                [np.asarray(piece.shared_output) for piece in shared_pieces],
                axis=0,
            )
        )
        sequence = np.ascontiguousarray(
            np.concatenate(
                [np.asarray(piece.sequence_id) for piece in shared_pieces],
                axis=0,
            )
        )
        position = np.ascontiguousarray(
            np.concatenate(
                [np.asarray(piece.position) for piece in shared_pieces],
                axis=0,
            )
        )
        selected_ids = np.ascontiguousarray(
            np.concatenate(
                [np.asarray(piece.expert_ids) for piece in id_pieces],
                axis=0,
            )
        )
        if (
            selected_shared.dtype != np.uint16
            or selected_shared.shape != (count, HIDDEN_SIZE)
            or selected_ids.dtype != np.uint8
            or selected_ids.shape != (count, TOP_K_EXPERTS)
            or not np.array_equal(sequence, microbatch.sequence_id[:count])
            or not np.array_equal(position, microbatch.position[:count])
        ):
            raise Experiment3EvaluationWorkerError(
                "target auxiliary token identity differs from paired input"
            )
        shared[local_index, :count] = selected_shared
        expert_ids[local_index, :count] = selected_ids
    return shared, expert_ids


def _pack_feature_mask(value: np.ndarray) -> str:
    selected = np.asarray(value)
    if selected.ndim != 1 or selected.dtype != np.bool_:
        raise Experiment3EvaluationWorkerError(
            "feature mask must be boolean [features]"
        )
    return np.packbits(selected, bitorder="little").tobytes().hex()


def _unpack_feature_mask(value: str, *, features: int) -> np.ndarray:
    try:
        raw = bytes.fromhex(value)
    except (TypeError, ValueError) as error:
        raise Experiment3EvaluationWorkerError(
            "packed feature mask is not hexadecimal"
        ) from error
    unpacked = np.unpackbits(
        np.frombuffer(raw, dtype=np.uint8),
        bitorder="little",
    )
    if unpacked.size < features or np.any(unpacked[features:] != 0):
        raise Experiment3EvaluationWorkerError(
            "packed feature mask length is incompatible"
        )
    return np.ascontiguousarray(unpacked[:features].astype(np.bool_))


def _seal_process_evidence(
    root: Path,
    *,
    condition: str,
    process_index: int,
    checkpoint_sha256: str,
    source_manifest_sha256: str,
    target_manifest_sha256: str,
    input_receipt_sha256: str,
    centering_receipt_sha256: str,
    capacity_contract_sha256: str,
    evaluation_wall_seconds: float,
    fired_mask: np.ndarray,
    sequence_id: np.ndarray,
    position: np.ndarray,
    expert_ids: np.ndarray,
    scores: ContributionTranscoderTokenScores,
) -> dict[str, Any]:
    tensors = {
        "sequence_id": np.asarray(sequence_id, dtype=np.int64),
        "position": np.asarray(position, dtype=np.uint16),
        "expert_ids": np.asarray(expert_ids, dtype=np.uint8),
        "component_sse": np.asarray(scores.component_sse, dtype=np.float64),
        "component_sst": np.asarray(scores.component_sst, dtype=np.float64),
        "routed_sse": np.asarray(scores.routed_sse, dtype=np.float64),
        "routed_sst": np.asarray(scores.routed_sst, dtype=np.float64),
        "oracle_shared_sse": np.asarray(
            scores.oracle_shared_sse,
            dtype=np.float64,
        ),
        "oracle_shared_sst": np.asarray(
            scores.oracle_shared_sst,
            dtype=np.float64,
        ),
        "route_component_sse": np.asarray(
            scores.route_component_sse,
            dtype=np.float64,
        ),
        "route_component_sst": np.asarray(
            scores.route_component_sst,
            dtype=np.float64,
        ),
    }
    blob = save_safetensors(tensors)
    blob_sha = hashlib.sha256(blob).hexdigest()
    data_path = root / f"{condition}-rows-{blob_sha}.safetensors"
    _create_only(data_path, blob)
    unsigned = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "condition": condition,
        "process_index": process_index,
        "checkpoint_sha256": checkpoint_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "input_receipt_sha256": input_receipt_sha256,
        "centering_receipt_sha256": centering_receipt_sha256,
        "capacity_contract_sha256": capacity_contract_sha256,
        "recomposition_sidecar_sha256": (
            scores.recomposition_sidecar_sha256
        ),
        "tokens": scores.tokens,
        "evaluation_wall_seconds": float(evaluation_wall_seconds),
        "total_features": int(fired_mask.size),
        "fired_mask_hex": _pack_feature_mask(fired_mask),
        "data_path": str(data_path),
        "data_sha256": blob_sha,
        "data_bytes": len(blob),
    }
    metadata = {**unsigned, "evidence_sha256": canonical_sha256(unsigned)}
    _write_json(root / f"{condition}-evidence.json", metadata)
    return metadata


def _load_process_evidence(
    path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], ContributionTranscoderTokenScores]:
    metadata = _read_json(path, name="process evaluation evidence")
    claimed = metadata.get("evidence_sha256")
    if (
        not isinstance(claimed, str)
        or _SHA256_RE.fullmatch(claimed) is None
        or canonical_sha256(
            {
                name: value
                for name, value in metadata.items()
                if name != "evidence_sha256"
            }
        )
        != claimed
        or metadata.get("schema_version") != EVIDENCE_SCHEMA_VERSION
    ):
        raise Experiment3EvaluationWorkerError(
            "process evaluation evidence self-hash is invalid"
        )
    data_path = Path(metadata["data_path"])
    try:
        blob = data_path.read_bytes()
    except OSError as error:
        raise Experiment3EvaluationWorkerError(
            f"cannot read score rows: {data_path}"
        ) from error
    if (
        len(blob) != metadata["data_bytes"]
        or hashlib.sha256(blob).hexdigest() != metadata["data_sha256"]
    ):
        raise Experiment3EvaluationWorkerError(
            "process score rows differ from retained content identity"
        )
    try:
        tensors = {
            name: np.asarray(value)
            for name, value in load_safetensors(blob).items()
        }
    except Exception as error:
        raise Experiment3EvaluationWorkerError(
            "process score rows cannot be decoded"
        ) from error
    if set(tensors) != _EVIDENCE_TENSORS:
        raise Experiment3EvaluationWorkerError(
            "process score tensor inventory changed"
        )
    tokens = int(metadata["tokens"])
    valid = np.ones((tokens,), dtype=np.bool_)
    scores = ContributionTranscoderTokenScores(
        component_sse=tensors["component_sse"],
        component_sst=tensors["component_sst"],
        routed_sse=tensors["routed_sse"],
        routed_sst=tensors["routed_sst"],
        oracle_shared_sse=tensors["oracle_shared_sse"],
        oracle_shared_sst=tensors["oracle_shared_sst"],
        route_component_sse=tensors["route_component_sse"],
        route_component_sst=tensors["route_component_sst"],
        valid_mask=valid,
        recomposition_sidecar_sha256=metadata[
            "recomposition_sidecar_sha256"
        ],
    )
    if (
        scores.tokens != tokens
        or tensors["sequence_id"].dtype != np.int64
        or tensors["sequence_id"].shape != (tokens,)
        or tensors["position"].dtype != np.uint16
        or tensors["position"].shape != (tokens,)
        or tensors["expert_ids"].dtype != np.uint8
        or tensors["expert_ids"].shape != (tokens, TOP_K_EXPERTS)
    ):
        raise Experiment3EvaluationWorkerError(
            "process score key arrays are incompatible"
        )
    return metadata, tensors, scores


def _concatenate_condition_evidence(
    rows: Sequence[tuple[dict[str, Any], dict[str, np.ndarray], ContributionTranscoderTokenScores]],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, ContributionTranscoderTokenScores]:
    ordered = sorted(rows, key=lambda item: int(item[0]["process_index"]))
    if [int(item[0]["process_index"]) for item in ordered] != list(
        range(len(ordered))
    ):
        raise Experiment3EvaluationWorkerError(
            "condition evidence process lanes are incomplete"
        )
    fixed_names = (
        "condition",
        "checkpoint_sha256",
        "source_manifest_sha256",
        "target_manifest_sha256",
        "centering_receipt_sha256",
        "capacity_contract_sha256",
        "recomposition_sidecar_sha256",
        "total_features",
    )
    for name in fixed_names:
        if len({item[0][name] for item in ordered}) != 1:
            raise Experiment3EvaluationWorkerError(
                f"condition evidence {name} differs across processes"
            )
    sequence = np.concatenate(
        [item[1]["sequence_id"] for item in ordered],
        axis=0,
    )
    position = np.concatenate(
        [item[1]["position"] for item in ordered],
        axis=0,
    )
    expert_ids = np.concatenate(
        [item[1]["expert_ids"] for item in ordered],
        axis=0,
    )
    valid = np.ones(sequence.shape, dtype=np.bool_)

    def joined(name: str) -> np.ndarray:
        return np.concatenate(
            [getattr(item[2], name) for item in ordered],
            axis=0,
        )

    scores = ContributionTranscoderTokenScores(
        component_sse=joined("component_sse"),
        component_sst=joined("component_sst"),
        routed_sse=joined("routed_sse"),
        routed_sst=joined("routed_sst"),
        oracle_shared_sse=joined("oracle_shared_sse"),
        oracle_shared_sst=joined("oracle_shared_sst"),
        route_component_sse=joined("route_component_sse"),
        route_component_sst=joined("route_component_sst"),
        valid_mask=valid,
        recomposition_sidecar_sha256=ordered[0][0][
            "recomposition_sidecar_sha256"
        ],
    )
    features = int(ordered[0][0]["total_features"])
    fired = np.zeros((features,), dtype=np.bool_)
    for metadata, _tensors, _scores in ordered:
        fired |= _unpack_feature_mask(
            metadata["fired_mask_hex"],
            features=features,
        )
    summary = {
        **{
            name: ordered[0][0][name]
            for name in fixed_names
        },
        "tokens": scores.tokens,
        "evaluation_wall_seconds": max(
            float(item[0]["evaluation_wall_seconds"]) for item in ordered
        ),
        "fired_mask": fired,
        "evaluation_receipt_sha256": canonical_sha256(
            sorted(item[0]["evidence_sha256"] for item in ordered)
        ),
    }
    return summary, sequence, position, expert_ids, scores


def _training_ledger(
    training_result: Mapping[str, Any],
    *,
    layout: Any,
    evaluation_wall_seconds: float,
    peak_flops_per_second: int,
) -> TranscoderTrainingLedger:
    marginal = float(training_result["marginal_wall_seconds"])
    chip_seconds = float(training_result["chip_seconds"])
    if marginal <= 0 or chip_seconds <= 0:
        raise Experiment3EvaluationWorkerError(
            "training result has non-positive marginal timing"
        )
    chip_count_float = chip_seconds / marginal
    chip_count = int(round(chip_count_float))
    if not math.isclose(chip_count_float, chip_count, abs_tol=1e-9):
        raise Experiment3EvaluationWorkerError(
            "training result chip count is not integral"
        )
    useful_flops = sae_training_useful_flops(
        SAEArithmetic.from_contribution_layout(layout),
        scientific_tokens=int(training_result["tokens"]),
    )
    optimizer_seconds = float(training_result["optimizer_wall_seconds"])
    useful_mfu = useful_flops / (
        int(peak_flops_per_second) * optimizer_seconds
    )
    timing_unsigned = {
        name: training_result[name]
        for name in (
            "compile_seconds",
            "lowering_seconds",
            "preprocessing_wall_seconds",
            "optimizer_wall_seconds",
            "marginal_wall_seconds",
            "checkpoint_seconds",
            "chip_seconds",
            "chip_seconds_scope",
            "chip_seconds_excludes",
            "tokens_per_second",
        )
    }
    return TranscoderTrainingLedger(
        training_result_sha256=training_result["result_sha256"],
        timing_receipt_sha256=canonical_sha256(timing_unsigned),
        chip_count=chip_count,
        preprocessing_wall_seconds=float(
            training_result["preprocessing_wall_seconds"]
        ),
        optimizer_wall_seconds=optimizer_seconds,
        marginal_wall_seconds=marginal,
        marginal_chip_seconds=chip_seconds,
        compilation_wall_seconds=float(
            training_result["compile_seconds"]
        )
        + float(training_result["lowering_seconds"]),
        checkpoint_serialization_wall_seconds=float(
            training_result["checkpoint_seconds"]
        ),
        evaluation_wall_seconds=float(evaluation_wall_seconds),
        total_wall_seconds=(
            marginal
            + float(training_result["compile_seconds"])
            + float(training_result["lowering_seconds"])
            + float(training_result["checkpoint_seconds"])
            + float(evaluation_wall_seconds)
        ),
        tokens_per_second=float(training_result["tokens_per_second"]),
        useful_mfu=float(useful_mfu),
    )


def _validate_phase_datasets(
    config: Any,
    phase: Any,
    *,
    train_target: Any,
    train_source: Any,
    eval_target: Any,
    eval_source: Any,
) -> None:
    """Require the exact four frozen target/overlay manifests."""

    if (
        int(train_target.total_tokens) != phase.train_tokens
        or int(train_source.total_tokens) != phase.train_tokens
        or int(eval_target.total_tokens) != phase.eval_tokens
        or int(eval_source.total_tokens) != phase.eval_tokens
        or train_target.manifest_sha256
        != config.parents["train_target_manifest_sha256"]
        or eval_target.manifest_sha256
        != config.parents["eval_target_manifest_sha256"]
        or train_source.manifest_sha256
        != config.parents["train_source_overlay_manifest_sha256"]
        or eval_source.manifest_sha256
        != config.parents["eval_source_overlay_manifest_sha256"]
    ):
        raise Experiment3EvaluationWorkerError(
            "source/target manifests differ from the frozen phase"
        )


def run(args: argparse.Namespace) -> int:
    import jax

    config = load_contribution_transcoder_config(args.config)
    if args.seed != 0 or args.phase != "mechanism_pilot":
        raise Experiment3EvaluationWorkerError(
            "Experiment 3 evaluator is frozen to mechanism_pilot seed 0"
        )
    if (
        not isinstance(args.source_revision, str)
        or _GIT_RE.fullmatch(args.source_revision) is None
    ):
        raise Experiment3EvaluationWorkerError(
            "source revision must be a full lowercase Git SHA"
        )
    if (
        not isinstance(args.training_source_revision, str)
        or _GIT_RE.fullmatch(args.training_source_revision) is None
    ):
        raise Experiment3EvaluationWorkerError(
            "training source revision must be a full lowercase Git SHA"
        )
    initialize_jax_distributed()
    require_v4_32_runtime()
    if int(jax.process_index()) != args.process_index:
        raise Experiment3EvaluationWorkerError(
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
    decoded_sidecar = decode_recomposition_sidecar(sidecar)
    cache = ShardCache(
        Path(args.cache_root) / f"process-p{jax.process_index():05d}",
        GCSObjectStore(),
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
    phase = config.phase(args.phase)
    _validate_phase_datasets(
        config,
        phase,
        train_target=train_target,
        train_source=train_source,
        eval_target=eval_target,
        eval_source=eval_source,
    )
    output = Path(args.result_root).expanduser().resolve()
    lane_root = output / f"process-p{jax.process_index():05d}"
    lane_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    target_mean, target_mean_sha = _pooled_routed_mean(train_target)
    source_mean, source_mean_sha = routed_input_mean(train_source)
    (
        routed_mean,
        routed_mean_sha,
        oracle_mean,
        oracle_mean_sha,
        endpoint_receipt,
    ) = _endpoint_training_means(train_target, sidecar)
    centering_unsigned = {
        "schema_version": (
            "gemma4-contribution-transcoder-all-centering-v1"
        ),
        "source_mean_sha256": source_mean_sha,
        "target_mean_sha256": target_mean_sha,
        "routed_mean_sha256": routed_mean_sha,
        "oracle_shared_mean_sha256": oracle_mean_sha,
        "endpoint_receipt_sha256": endpoint_receipt["receipt_sha256"],
    }
    centering_receipt = {
        **centering_unsigned,
        "receipt_sha256": canonical_sha256(centering_unsigned),
    }
    _write_json(output / "centering-receipt.json", centering_receipt)
    _collective_barrier("experiment3-endpoint-centering")

    plans = {
        condition: TrainingWorkerPlan.from_config(
            config,
            phase=args.phase,
            condition=condition,
            seed=args.seed,
        )
        for condition in CONDITIONS
    }
    runs: dict[str, tuple[dict[str, Any], dict[str, Any], Path]] = {}
    run_contract_sha256: dict[str, str] = {}
    for condition in CONDITIONS:
        run_root = _condition_run_root(args, condition)
        result, pointer = _load_run(
            run_root,
            plans[condition],
            args.checkpoint_payload_root,
        )
        contract = _validate_run_contract(
            run_root,
            condition=condition,
            source_manifest_sha256=train_source.manifest_sha256,
            target_manifest_sha256=train_target.manifest_sha256,
            training_source_revision=args.training_source_revision,
        )
        run_contract_sha256[condition] = str(
            contract["run_contract_sha256"]
        )
        runs[condition] = (result, pointer, run_root)

    for condition in CONDITIONS:
        plan = plans[condition]
        layout = _layout(config, plan)
        source = ContributionTranscoderProcessInput(
            eval_source,
            eval_target,
            layout,
            total_tokens=phase.eval_tokens,
            global_microbatch_tokens=plan.global_microbatch_tokens,
            microbatches_per_update=plan.microbatches_per_update,
            process_count=int(jax.process_count()),
            process_index=int(jax.process_index()),
        )
        result, pointer, run_root = runs[condition]
        training_capacity = (
            _checkpoint_a_capacity(pointer)
            if condition in {"A", "R"}
            else None
        )
        started = time.perf_counter()
        if condition in {"A", "R"}:
            local_capacity = source.scan_local_capacity(
                margin_fraction=config.runtime.capacity_margin_fraction
            )
            capacity_rows = _collective_records(
                dataclasses.asdict(local_capacity)
            )
            try:
                capacity_receipts = tuple(
                    CapacityReceipt(**row) for row in capacity_rows
                )
                for receipt in capacity_receipts:
                    _validate_capacity_receipt(receipt, layout)
                evaluation_capacity = merge_capacity_receipts(capacity_receipts)
            except Exception as error:
                raise Experiment3EvaluationWorkerError(
                    "cannot merge complete held-out A/R capacity evidence"
                ) from error
            try:
                capacity_roles = _evaluation_capacity_roles(
                    condition,
                    training_capacity=training_capacity,
                    evaluation_capacity=int(evaluation_capacity.capacity),
                )
            except ValueError as error:
                raise Experiment3EvaluationWorkerError(
                    "A/R checkpoint or held-out capacity is invalid"
                ) from error
            capacity = capacity_roles.execution
            capacity_unsigned = {
                "schema_version": (
                    "gemma4-transcoder-evaluation-capacity-contract-v1"
                ),
                "condition": condition,
                "training_capacity": int(training_capacity),
                "evaluation_capacity": int(evaluation_capacity.capacity),
                "evaluation_capacity_receipt_sha256": (
                    evaluation_capacity.receipt_sha256
                ),
                "selected_capacity": capacity,
            }
        else:
            capacity_roles = _evaluation_capacity_roles(
                condition,
                training_capacity=None,
                evaluation_capacity=None,
            )
            capacity = capacity_roles.execution
            capacity_unsigned = {
                "schema_version": (
                    "gemma4-transcoder-evaluation-capacity-contract-v1"
                ),
                "condition": condition,
                "training_capacity": None,
                "evaluation_capacity": None,
                "evaluation_capacity_receipt_sha256": None,
                "selected_capacity": None,
            }
        capacity_contract = {
            **capacity_unsigned,
            "capacity_contract_sha256": canonical_sha256(capacity_unsigned),
        }
        optimizer = _optimizer(config, plan)
        state = init_contribution_transcoder_distributed_state(
            jax.random.key(args.seed),
            mesh,
            layout,
            optimizer,
            distributed,
        )
        state, _ = restore_distributed_checkpoint(
            pointer_path(run_root / "checkpoints", "final"),
            state,
            mesh,
            _training_config(
                plan,
                # The checkpoint serializer uses the shared SAE-shaped
                # training config while the restored target remains explicitly
                # transcoder-typed.
                optimizer_sae_config(layout, optimizer),
                bank_capacity=capacity_roles.checkpoint,
            ),
            checkpoint_payload_root=args.checkpoint_payload_root,
            checkpoint_identity=result["run_identity"],
            checkpoint_metadata_identity={
                "run_identity_sha256": result["run_identity"][
                    "run_identity_sha256"
                ]
            },
        )
        evaluator = ContributionTranscoderDistributedEvaluator(
            mesh,
            layout,
            distributed,
            bank_capacity=capacity,
        )
        pieces: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "sequence_id",
                "position",
                "expert_ids",
                "component_sse",
                "component_sst",
                "routed_sse",
                "routed_sst",
                "oracle_shared_sse",
                "oracle_shared_sst",
                "route_component_sse",
                "route_component_sst",
            )
        }
        fired = np.zeros((layout.total_features,), dtype=np.bool_)
        for update_index, microbatches in enumerate(source.iter_updates()):
            aux, chunk = evaluator.step(
                state.params,
                microbatches,
                source_mean,
                target_mean,
                process_local=int(jax.process_count()) > 1,
            )
            fired |= evaluator.complete_feature_mask(aux)
            shared_bits, expert_ids = _target_auxiliary(
                source,
                update_index=update_index,
                microbatches=microbatches,
            )
            flat_valid = np.asarray(chunk.valid_mask).reshape(-1)
            scored = score_transcoder_sufficient_rows(
                component_sse=np.asarray(
                    chunk.component_token_sse
                ).reshape(-1),
                component_sst=np.asarray(
                    chunk.component_token_sst
                ).reshape(-1),
                route_component_sse=np.asarray(
                    chunk.component_route_sse
                ).reshape(-1, TOP_K_EXPERTS),
                route_component_sst=np.asarray(
                    chunk.component_route_sst
                ).reshape(-1, TOP_K_EXPERTS),
                raw_routed_target=np.asarray(
                    chunk.raw_routed_target
                ).reshape(-1, HIDDEN_SIZE),
                raw_routed_reconstruction=np.asarray(
                    chunk.raw_routed_reconstruction
                ).reshape(-1, HIDDEN_SIZE),
                shared_output_bf16_bits=shared_bits.reshape(
                    -1,
                    HIDDEN_SIZE,
                ),
                routed_mean=routed_mean,
                oracle_shared_mean=oracle_mean,
                valid_mask=flat_valid,
                recomposition_sidecar=sidecar,
            )
            sequence = np.stack(
                [np.asarray(item.sequence_id) for item in microbatches],
                axis=0,
            ).reshape(-1)
            position = np.stack(
                [np.asarray(item.position) for item in microbatches],
                axis=0,
            ).reshape(-1)
            pieces["sequence_id"].append(sequence[flat_valid])
            pieces["position"].append(position[flat_valid])
            pieces["expert_ids"].append(
                expert_ids.reshape(-1, TOP_K_EXPERTS)[flat_valid]
            )
            for name in (
                "component_sse",
                "component_sst",
                "routed_sse",
                "routed_sst",
                "oracle_shared_sse",
                "oracle_shared_sst",
                "route_component_sse",
                "route_component_sst",
            ):
                pieces[name].append(np.asarray(getattr(scored, name))[flat_valid])
        local_elapsed = time.perf_counter() - started
        timing_rows = _collective_records(
            {
                "process_index": int(jax.process_index()),
                "condition": condition,
                "evaluation_wall_seconds": float(local_elapsed),
            }
        )
        evaluation_seconds = max(
            float(row["evaluation_wall_seconds"]) for row in timing_rows
        )
        combined = {
            name: np.ascontiguousarray(np.concatenate(values, axis=0))
            for name, values in pieces.items()
        }
        valid = np.ones(combined["sequence_id"].shape, dtype=np.bool_)
        scores = ContributionTranscoderTokenScores(
            component_sse=combined["component_sse"],
            component_sst=combined["component_sst"],
            routed_sse=combined["routed_sse"],
            routed_sst=combined["routed_sst"],
            oracle_shared_sse=combined["oracle_shared_sse"],
            oracle_shared_sst=combined["oracle_shared_sst"],
            route_component_sse=combined["route_component_sse"],
            route_component_sst=combined["route_component_sst"],
            valid_mask=valid,
            recomposition_sidecar_sha256=decoded_sidecar.sidecar_sha256,
        )
        expected_local = source.process_range_end - source.process_range_start
        if scores.tokens != expected_local:
            raise Experiment3EvaluationWorkerError(
                "condition evaluator did not retain its exact process lane"
            )
        _seal_process_evidence(
            lane_root,
            condition=condition,
            process_index=int(jax.process_index()),
            checkpoint_sha256=pointer["pointer_sha256"],
            source_manifest_sha256=eval_source.manifest_sha256,
            target_manifest_sha256=eval_target.manifest_sha256,
            input_receipt_sha256=source.receipt.receipt_sha256,
            centering_receipt_sha256=centering_receipt["receipt_sha256"],
            capacity_contract_sha256=capacity_contract[
                "capacity_contract_sha256"
            ],
            evaluation_wall_seconds=evaluation_seconds,
            fired_mask=fired,
            sequence_id=combined["sequence_id"],
            position=combined["position"],
            expert_ids=combined["expert_ids"],
            scores=scores,
        )
        del state
        jax.clear_caches()
        _collective_barrier(f"experiment3-evaluation-{condition}")

    if int(jax.process_index()) == 0:
        condition_data: dict[str, Any] = {}
        token_scores: dict[str, ContributionTranscoderTokenScores] = {}
        canonical_sequence: np.ndarray | None = None
        canonical_position: np.ndarray | None = None
        for condition in CONDITIONS:
            process_rows = [
                _load_process_evidence(
                    output
                    / f"process-p{process_index:05d}"
                    / f"{condition}-evidence.json"
                )
                for process_index in range(int(jax.process_count()))
            ]
            summary, sequence, position, expert_ids, selected_scores = (
                _concatenate_condition_evidence(process_rows)
            )
            if canonical_sequence is None:
                canonical_sequence = sequence
                canonical_position = position
            elif not np.array_equal(
                sequence,
                canonical_sequence,
            ) or not np.array_equal(position, canonical_position):
                raise Experiment3EvaluationWorkerError(
                    "condition score rows are not exactly paired by token key"
                )
            condition_data[condition] = (
                summary,
                expert_ids,
                selected_scores,
            )
            token_scores[condition] = selected_scores
        assert canonical_sequence is not None
        assert canonical_position is not None
        paired = aggregate_paired_sequence_scores(
            scores=token_scores,
            sequence_id=canonical_sequence,
            position=canonical_position,
            source_manifest_sha256=eval_source.manifest_sha256,
            target_manifest_sha256=eval_target.manifest_sha256,
        )
        _write_json(output / "paired-sequence-scores.json", paired.to_dict())
        common_sst_unsigned = {
            "schema_version": COMMON_SST_SCHEMA_VERSION,
            "score_artifact_sha256": paired.score_artifact_sha256,
            "target_manifest_sha256": paired.target_manifest_sha256,
            "token_identity_sha256": paired.token_identity_sha256,
            "endpoint_sst_sha256": {
                endpoint: hashlib.sha256(
                    paired.common_sst[endpoint]
                    .astype("<f8", copy=False)
                    .tobytes(order="C")
                ).hexdigest()
                for endpoint in paired.common_sst
            },
        }
        common_sst_receipt = {
            **common_sst_unsigned,
            "receipt_sha256": canonical_sha256(common_sst_unsigned),
        }
        _write_json(output / "common-sst-receipt.json", common_sst_receipt)
        summaries = {}
        ledgers = {}
        for condition in CONDITIONS:
            summary, expert_ids, selected_scores = condition_data[condition]
            diagnostics = expert_rank_diagnostics(
                selected_scores,
                expert_ids,
            )
            _write_json(
                output / f"{condition}-diagnostics.json",
                diagnostics,
            )
            fired = np.asarray(summary["fired_mask"])
            summaries[condition] = build_condition_summary(
                paired,
                condition=condition,
                checkpoint_sha256=summary["checkpoint_sha256"],
                evaluation_receipt_sha256=summary[
                    "evaluation_receipt_sha256"
                ],
                fired_feature_fraction=float(np.mean(fired)),
                macro_expert_ev=float(diagnostics["macro_expert_ev"]),
                supported_experts=int(diagnostics["supported_experts"]),
                diagnostics_sha256=diagnostics["diagnostics_sha256"],
            )
            training_result = runs[condition][0]
            ledgers[condition] = _training_ledger(
                training_result,
                layout=_layout(config, plans[condition]),
                evaluation_wall_seconds=float(
                    summary["evaluation_wall_seconds"]
                ),
                peak_flops_per_second=args.peak_flops_per_second,
            )
        harvest_receipt_sha = args.shared_harvest_receipt_sha256
        if harvest_receipt_sha is None:
            harvest_receipt_sha = canonical_sha256(
                {
                    "schema_version": (
                        "gemma4-transcoder-shared-harvest-derived-receipt-v1"
                    ),
                    "train_source_manifest_sha256": (
                        train_source.manifest_sha256
                    ),
                    "eval_source_manifest_sha256": (
                        eval_source.manifest_sha256
                    ),
                }
            )
        result = build_transcoder_scientific_result(
            scores=paired,
            summaries=summaries,
            ledgers=ledgers,
            source_revision=args.source_revision,
            training_source_revision=args.training_source_revision,
            run_contract_sha256=run_contract_sha256,
            config_sha256=config.config_hash,
            train_source_manifest_sha256=train_source.manifest_sha256,
            train_target_manifest_sha256=train_target.manifest_sha256,
            common_sst_receipt_sha256=common_sst_receipt[
                "receipt_sha256"
            ],
            shared_harvest_receipt_sha256=harvest_receipt_sha,
            shared_harvest_chip_seconds=args.shared_harvest_chip_seconds,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        validate_transcoder_scientific_result(result, scores=paired)
        write_transcoder_result(
            output / "scientific-result.json",
            output / "REPORT.md",
            result,
            scores=paired,
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    _collective_barrier("experiment3-evaluation-final")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", default="mechanism_pilot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--training-source-revision", required=True)
    parser.add_argument("--train-target-root", required=True)
    parser.add_argument("--eval-target-root", required=True)
    parser.add_argument("--train-target-object-prefix", required=True)
    parser.add_argument("--eval-target-object-prefix", required=True)
    parser.add_argument("--train-source-root", required=True)
    parser.add_argument("--eval-source-root", required=True)
    parser.add_argument("--train-source-object-prefix", required=True)
    parser.add_argument("--eval-source-object-prefix", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cache-max-bytes", type=int, required=True)
    parser.add_argument("--sae-control-root", required=True)
    parser.add_argument("--checkpoint-payload-root", required=True)
    parser.add_argument("--recomposition-sidecar", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--run-root-a")
    parser.add_argument("--run-root-r")
    parser.add_argument("--run-root-b")
    parser.add_argument("--run-root-f")
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--peak-flops-per-second",
        type=int,
        default=4_400_000_000_000_000,
    )
    parser.add_argument("--shared-harvest-receipt-sha256")
    parser.add_argument(
        "--shared-harvest-chip-seconds",
        type=float,
        default=0.0,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
