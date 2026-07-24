from __future__ import annotations

import copy
import dataclasses
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    LEGACY_AGGREGATE_SCHEMA_VERSION,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    TOP_K_EXPERTS,
    ContributionBatch,
    ContributionMetadata,
    bf16_bits_to_float32,
    build_recomposition_sidecar,
    canonical_sha256,
    compute_contribution_statistics,
    contribution_batch_token_identity_sha256,
    deterministic_range_id,
    float32_to_bf16_bits,
    seal_contribution_manifest,
)
from moe_sae.contribution_input import ContributionInputReceipt
from moe_sae.contribution_recomposition import (
    SUPPORTED_ACCUMULATION_ORDER,
    maxtext_rmsnorm_bf16,
    recompose_gemma4,
)
from moe_sae.contribution_score_artifacts import (
    seal_score_manifest,
    seal_score_row_shard,
)
from moe_sae.contribution_shared import (
    FullFFNRecompositionRows,
    SharedQEvaluationEvidenceChunk,
    SharedQEvaluationRows,
    SharedQInputReceipt,
)
from moe_sae.contribution_shared_artifacts import (
    FullFFNCenteringAccumulator,
    FullFFNScoreRowSink,
    SharedEvidenceArtifactError,
    SharedEvidenceRef,
    SharedQScoreRowSink,
    resolve_full_ffn_centering,
    resolve_full_ffn_score_manifest,
    resolve_shared_q_score_manifest,
    seal_full_ffn_centering_from_dataset,
    seal_full_ffn_score_manifest,
    seal_shared_q_score_manifest,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def sidecar() -> dict:
    ones = float32_to_bf16_bits(np.ones((HIDDEN_SIZE,), np.float32))
    return build_recomposition_sidecar(
        checkpoint_sha256=digest("checkpoint"),
        maxtext_fork_commit="1" * 40,
        source_fingerprints={
            "checkpoint": digest("checkpoint"),
            "fork": digest("fork"),
        },
        routed_rmsnorm_scale=ones,
        shared_rmsnorm_scale=ones,
        outer_ffn_rmsnorm_scale=ones,
        layer_scalar=float32_to_bf16_bits(np.ones((1,), np.float32)),
        accumulation_order=SUPPORTED_ACCUMULATION_ORDER,
    )


def wire(tokens: int, *, offset: int = 0) -> ContributionBatch:
    feature = np.linspace(0.2, 1.2, HIDDEN_SIZE, dtype=np.float32)
    expert = np.stack(
        [
            np.broadcast_to(
                feature[None, :] * np.float32(1.0 + 0.02 * (row + offset)),
                (TOP_K_EXPERTS, HIDDEN_SIZE),
            )
            for row in range(tokens)
        ]
    )
    shared = np.stack(
        [
            feature * np.float32(0.4 + 0.05 * (row + offset))
            + np.float32((row + offset) % 2) * 0.1
            for row in range(tokens)
        ]
    )
    weights = np.asarray(
        [0.25, 0.1875, 0.15625, 0.125, 0.09375, 0.078125, 0.0625, 0.046875],
        np.float32,
    )
    return ContributionBatch(
        expert_output=float32_to_bf16_bits(expert),
        expert_ids=np.broadcast_to(
            np.arange(TOP_K_EXPERTS, dtype=np.uint8),
            (tokens, TOP_K_EXPERTS),
        ).copy(),
        router_weights=float32_to_bf16_bits(
            np.broadcast_to(weights, (tokens, TOP_K_EXPERTS))
        ),
        shared_output=float32_to_bf16_bits(shared),
        sequence_id=np.asarray(
            [100 + (row + offset) // 2 for row in range(tokens)],
            np.int64,
        ),
        position=np.arange(offset, offset + tokens, dtype=np.uint16),
    )


def manifest(value: ContributionBatch, selected_sidecar: dict, *, split: str) -> dict:
    _, normalized = maxtext_rmsnorm_bf16(
        value.shared_output,
        np.asarray(
            selected_sidecar["tensors"]["shared_rmsnorm_scale"]["data"],
            np.uint16,
        ),
    )
    metadata = ContributionMetadata(
        checkpoint_sha256=digest("checkpoint"),
        maxtext_fork_commit="1" * 40,
        dataset_fingerprint=digest("dataset"),
        split=split,
        config_sha256=digest("config"),
        environment_sha256=digest("environment"),
        source_v1_manifest_sha256=digest("source"),
        recomposition_sidecar_sha256=selected_sidecar["sidecar_sha256"],
    ).to_dict()
    metadata_sha = canonical_sha256(metadata)
    token_sha = contribution_batch_token_identity_sha256(value)
    object_sha = digest(f"{split} object")
    entry = {
        "range_id": deterministic_range_id(
            split=split,
            range_start=0,
            range_end=value.tokens,
            source_v1_range_start=0,
            source_v1_range_end=value.tokens,
            source_v1_manifest_sha256=digest("source"),
            token_identity_sha256=token_sha,
        ),
        "range_start": 0,
        "range_end": value.tokens,
        "source_v1_range_start": 0,
        "source_v1_range_end": value.tokens,
        "tokens": value.tokens,
        "host_id": "worker-0",
        "path": f"shards/host-worker-0/{object_sha}.safetensors",
        "sha256": object_sha,
        "token_identity_sha256": token_sha,
        "metadata_sha256": metadata_sha,
        "source_v1_manifest_sha256": digest("source"),
        "recomposition_sidecar_sha256": selected_sidecar["sidecar_sha256"],
        "statistics": compute_contribution_statistics(
            value,
            normalized_shared_output=normalized,
        ),
    }
    return seal_contribution_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "sealed": True,
            "metadata": metadata,
            "metadata_sha256": metadata_sha,
            "source_v1_manifest": {
                "schema_version": LEGACY_AGGREGATE_SCHEMA_VERSION,
                "manifest_sha256": digest("source"),
                "split": split,
                "total_tokens": value.tokens,
            },
            "recomposition_sidecar_sha256": selected_sidecar["sidecar_sha256"],
            "total_tokens": value.tokens,
            "shard_count": 1,
            "statistics": entry["statistics"],
            "shards": [entry],
        }
    )


class Dataset:
    def __init__(self, value: ContributionBatch, selected_sidecar: dict, *, split: str):
        self.value = value
        self.manifest = manifest(value, selected_sidecar, split=split)
        self.total_tokens = value.tokens

    @property
    def manifest_sha256(self) -> str:
        return self.manifest["manifest_sha256"]

    def iter_token_range(self, start: int, end: int):
        boundary = min(start + 2, end)
        for left, right in ((start, boundary), (boundary, end)):
            if left == right:
                continue
            yield ContributionBatch(
                **{
                    name: np.ascontiguousarray(
                        np.asarray(getattr(self.value, name))[left:right]
                    )
                    for name in ContributionBatch.__dataclass_fields__
                }
            )


class MemoryStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def retain(self, reference, blob: bytes) -> None:
        assert hashlib.sha256(blob).hexdigest() == reference.sha256
        previous = self.objects.setdefault(reference.sha256, blob)
        assert previous == blob

    def __call__(self, reference) -> bytes:
        return self.objects[reference.sha256]


def q_receipt(
    *,
    role: str,
    process_index: int = 0,
    total_tokens: int,
    capture: str,
    centering: str,
    selected_sidecar: str,
) -> SharedQInputReceipt:
    global_microbatch = 2
    updates = (total_tokens + global_microbatch - 1) // global_microbatch
    unsigned = {
        "schema_version": "gemma4-shared-q-input-v1",
        "role": role,
        "capture_manifest_sha256": capture,
        "centering_contract_sha256": centering,
        "centering_parent_manifest_sha256": None,
        "recomposition_sidecar_sha256": selected_sidecar,
        "total_tokens": total_tokens,
        "sealed_sst": 10.0,
        "global_microbatch_tokens": global_microbatch,
        "microbatches_per_update": 1,
        "planned_updates": updates,
        "process_count": 1,
        "process_index": process_index,
        "local_rows_per_microbatch": global_microbatch,
    }
    return SharedQInputReceipt(
        **unsigned,
        receipt_sha256=canonical_sha256(unsigned),
    )


def q_evidence(
    update: int,
    *,
    capture: str,
    centering: str,
    checkpoint: str,
    sidecar_sha: str,
) -> SharedQEvaluationEvidenceChunk:
    if update == 0:
        sequence = np.asarray([[200, 200]], np.int64)
        position = np.asarray([[0, 1]], np.uint16)
        valid = np.asarray([[True, True]])
    else:
        sequence = np.asarray([[201, -1]], np.int64)
        position = np.asarray([[2, 65_535]], np.uint16)
        valid = np.asarray([[True, False]])
    target = np.zeros((1, 2, HIDDEN_SIZE), np.float32)
    target[0, 0] = np.linspace(0.1, 0.5, HIDDEN_SIZE, dtype=np.float32)
    if update == 0:
        target[0, 1] = target[0, 0] + 0.1
    reconstruction = np.where(valid[..., None], target * 0.9, 0.0)
    token_sse = np.sum(
        np.square(target - reconstruction, dtype=np.float32),
        axis=2,
        dtype=np.float64,
    )
    token_sst = np.where(valid, np.asarray([[2.0, 3.0]]), 0.0)
    rows = SharedQEvaluationRows(
        token_sse=np.where(valid, token_sse, 0.0),
        token_sst=token_sst,
        target=target,
        reconstruction=reconstruction,
        l0=np.where(valid, 64, 0).astype(np.int32),
        valid_mask=valid,
    )
    return SharedQEvaluationEvidenceChunk(
        update_index=update,
        capture_manifest_sha256=capture,
        centering_contract_sha256=centering,
        checkpoint_sha256=checkpoint,
        recomposition_sidecar_sha256=sidecar_sha,
        sequence_id=sequence,
        position=position,
        rows=rows,
    )


def test_streaming_full_ffn_centering_is_sealed_and_exact() -> None:
    selected_sidecar = sidecar()
    value = wire(3)
    dataset = Dataset(value, selected_sidecar, split="train")
    reference, blob = seal_full_ffn_centering_from_dataset(
        dataset,
        selected_sidecar,
        recomposition_chunk_tokens=1,
    )
    store = MemoryStore()
    store.retain(reference, blob)

    resolved = resolve_full_ffn_centering(reference, store)
    target = bf16_bits_to_float32(
        recompose_gemma4(value, selected_sidecar).outer_ffn_post_norm_bf16_bits
    )
    expected_sum = target.astype(np.float64).sum(axis=0, dtype=np.float64)
    expected_mean = (expected_sum / value.tokens).astype(np.float32)
    expected_sst = np.sum(
        np.square(
            target.astype(np.float64) - expected_mean.astype(np.float64),
            dtype=np.float64,
        ),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(resolved.target_sum, expected_sum)
    np.testing.assert_array_equal(resolved.mean, expected_mean)
    assert resolved.sst == pytest.approx(expected_sst, rel=1e-10, abs=1e-10)
    assert resolved.token_identity_sha256 == contribution_batch_token_identity_sha256(
        value
    )

    accumulator = FullFFNCenteringAccumulator(
        training_capture_manifest_sha256=dataset.manifest_sha256,
        recomposition_sidecar=selected_sidecar,
        expected_tokens=3,
    )
    with pytest.raises(SharedEvidenceArtifactError, match="contiguous"):
        accumulator.add(value, range_start=1)

    corrupt = MemoryStore()
    corrupt.objects[reference.sha256] = blob[:-1] + bytes([blob[-1] ^ 1])
    with pytest.raises(SharedEvidenceArtifactError, match="hash differs"):
        resolve_full_ffn_centering(reference, corrupt)


def retain_q_manifest(
    store: MemoryStore,
    *,
    selected_sidecar: dict,
    eval_capture: str,
    train_capture: str,
    centering: str,
    q_checkpoint: str,
) -> tuple[SharedQScoreRowSink, SharedEvidenceRef]:
    evaluation = q_receipt(
        role="evaluation",
        total_tokens=3,
        capture=eval_capture,
        centering=centering,
        selected_sidecar=selected_sidecar["sidecar_sha256"],
    )
    training = q_receipt(
        role="train",
        total_tokens=3,
        capture=train_capture,
        centering=centering,
        selected_sidecar=selected_sidecar["sidecar_sha256"],
    )
    sink = SharedQScoreRowSink(
        seed=0,
        process_index=0,
        evaluation_input_receipt=evaluation,
        retain=store.retain,
        resolve=store,
    )
    for update in range(2):
        sink(
            q_evidence(
                update,
                capture=eval_capture,
                centering=centering,
                checkpoint=q_checkpoint,
                sidecar_sha=selected_sidecar["sidecar_sha256"],
            )
        )
    reference, blob = seal_shared_q_score_manifest(
        sink.row_references,
        store,
        phase="phase2",
        protocol_sha256=digest("protocol"),
        seed=0,
        training_capture_manifest_sha256=train_capture,
        evaluation_capture_manifest_sha256=eval_capture,
        recomposition_sidecar_sha256=selected_sidecar["sidecar_sha256"],
        run_manifest_sha256=digest("Q run"),
        config_sha256=digest("Q config"),
        checkpoint_sha256=q_checkpoint,
        centering_contract_sha256=centering,
        training_input_receipts=[training],
        evaluation_input_receipts=[evaluation],
        expected_tokens=3,
    )
    store.retain(reference, blob)
    return sink, reference


def routed_input_receipt(eval_capture: str) -> ContributionInputReceipt:
    unsigned = {
        "schema_version": "gemma4-contribution-training-input-v1",
        "capture_manifest_sha256": eval_capture,
        "condition": "A",
        "randomization_key": 0,
        "layout_sha256": digest("A layout"),
        "total_tokens": 3,
        "global_microbatch_tokens": 2,
        "microbatches_per_update": 1,
        "planned_updates": 2,
        "process_count": 1,
        "process_index": 0,
        "local_rows_per_microbatch": 2,
    }
    return ContributionInputReceipt(
        **unsigned,
        receipt_sha256=canonical_sha256(unsigned),
    )


def retain_routed_manifest(
    store: MemoryStore,
    *,
    selected_sidecar: dict,
    eval_capture: str,
    train_capture: str,
    checkpoint: str,
) -> tuple[dict[int, object], object]:
    receipt = routed_input_receipt(eval_capture)
    references = {}
    for update in range(2):
        q = q_evidence(
            update,
            capture=eval_capture,
            centering=digest("Q centering"),
            checkpoint=digest("Q checkpoint"),
            sidecar_sha=selected_sidecar["sidecar_sha256"],
        )
        valid = q.rows.valid_mask
        score = SimpleNamespace(
            valid_mask=valid,
            component_token_sse=np.where(valid, [[0.1, 0.2]], 0.0),
            component_token_sst=np.where(valid, [[1.0, 1.5]], 0.0),
        )
        routed = SimpleNamespace(
            valid_mask=valid.copy(),
            token_sse=np.where(valid, [[0.2, 0.3]], 0.0),
            token_sst=np.where(valid, [[2.0, 2.5]], 0.0),
            recomposition_sidecar_sha256=selected_sidecar["sidecar_sha256"],
        )
        evidence = SimpleNamespace(
            update_index=update,
            sequence_id=q.sequence_id,
            position=q.position,
            scores=score,
        )
        reference, blob = seal_score_row_shard(
            evidence,
            routed,
            condition="A",
            seed=0,
            process_index=0,
            evaluation_capture_manifest_sha256=eval_capture,
            checkpoint_sha256=checkpoint,
            recomposition_sidecar_sha256=selected_sidecar["sidecar_sha256"],
            centering_mean_sha256=digest("routed mean"),
            evaluation_input_receipt_sha256=receipt.receipt_sha256,
        )
        store.retain(reference, blob)
        references[update] = reference
    manifest_reference, manifest_blob = seal_score_manifest(
        list(references.values()),
        store,
        phase="phase2",
        protocol_sha256=digest("protocol"),
        condition="A",
        seed=0,
        randomization_key=0,
        training_capture_manifest_sha256=train_capture,
        evaluation_capture_manifest_sha256=eval_capture,
        recomposition_sidecar_sha256=selected_sidecar["sidecar_sha256"],
        run_manifest_sha256=digest("A run"),
        config_sha256=digest("A config"),
        checkpoint_sha256=checkpoint,
        layout_sha256=digest("A layout"),
        centering_mean_sha256=digest("routed mean"),
        training_input_receipt_sha256s=[digest("A train receipt")],
        evaluation_input_receipts=[receipt],
        expected_tokens=3,
    )
    store.retain(manifest_reference, manifest_blob)
    return references, manifest_reference


def full_rows(
    update: int,
    *,
    selected_sidecar: dict,
    q_checkpoint: str,
) -> FullFFNRecompositionRows:
    q = q_evidence(
        update,
        capture=digest("eval capture"),
        centering=digest("Q centering"),
        checkpoint=q_checkpoint,
        sidecar_sha=selected_sidecar["sidecar_sha256"],
    )
    valid = q.rows.valid_mask
    target = np.where(valid[..., None], q.rows.target + 0.2, 0.0)
    oracle = np.where(valid[..., None], target * 0.95, 0.0)
    learned = np.where(valid[..., None], target * 0.9, 0.0)
    return FullFFNRecompositionRows(
        sequence_id=q.sequence_id,
        position=q.position,
        valid_mask=valid,
        target=target,
        oracle_shared_reconstruction=oracle,
        learned_shared_reconstruction=learned,
        recomposition_sidecar_sha256=selected_sidecar["sidecar_sha256"],
        shared_q_capture_manifest_sha256=digest("eval capture"),
        shared_q_centering_contract_sha256=digest("Q centering"),
        shared_q_checkpoint_sha256=q_checkpoint,
    )


def test_q_and_full_ffn_sinks_seal_complete_identity_bound_manifests() -> None:
    selected_sidecar = sidecar()
    store = MemoryStore()
    eval_capture = digest("eval capture")
    centering_contract = digest("Q centering")
    q_checkpoint = digest("Q checkpoint")
    training = Dataset(wire(3), selected_sidecar, split="train")
    q_sink, q_manifest_ref = retain_q_manifest(
        store,
        selected_sidecar=selected_sidecar,
        eval_capture=eval_capture,
        train_capture=training.manifest_sha256,
        centering=centering_contract,
        q_checkpoint=q_checkpoint,
    )
    q_manifest = resolve_shared_q_score_manifest(q_manifest_ref, store)
    assert q_manifest.tokens == 3
    assert q_manifest.q_l0_sum.sum() == 3 * 64

    # Exact replay is idempotent; a changed replay of update zero fails closed.
    q_sink(
        q_evidence(
            0,
            capture=eval_capture,
            centering=centering_contract,
            checkpoint=q_checkpoint,
            sidecar_sha=selected_sidecar["sidecar_sha256"],
        )
    )
    changed = q_evidence(
        0,
        capture=eval_capture,
        centering=centering_contract,
        checkpoint=q_checkpoint,
        sidecar_sha=selected_sidecar["sidecar_sha256"],
    )
    changed_rows = dataclasses.replace(
        changed.rows,
        token_sse=changed.rows.token_sse + 1.0,
    )
    with pytest.raises(SharedEvidenceArtifactError, match="replay changed"):
        q_sink(dataclasses.replace(changed, rows=changed_rows))

    centering_ref, centering_blob = seal_full_ffn_centering_from_dataset(
        training,
        selected_sidecar,
        recomposition_chunk_tokens=2,
    )
    store.retain(centering_ref, centering_blob)
    full_centering = resolve_full_ffn_centering(centering_ref, store)
    routed_rows, routed_manifest_ref = retain_routed_manifest(
        store,
        selected_sidecar=selected_sidecar,
        eval_capture=eval_capture,
        train_capture=training.manifest_sha256,
        checkpoint=digest("A checkpoint"),
    )
    full_sink = FullFFNScoreRowSink(
        condition="A",
        seed=0,
        process_index=0,
        evaluation_capture_manifest_sha256=eval_capture,
        routed_checkpoint_sha256=digest("A checkpoint"),
        centering=full_centering,
        dependency_rows={
            update: (
                routed_rows[update].sha256,
                q_sink.row_references[update].sha256,
            )
            for update in range(2)
        },
        retain=store.retain,
        resolve=store,
    )
    for update in range(2):
        full_sink.retain_rows(
            update,
            full_rows(
                update,
                selected_sidecar=selected_sidecar,
                q_checkpoint=q_checkpoint,
            ),
        )
    full_manifest_ref, full_manifest_blob = seal_full_ffn_score_manifest(
        full_sink.row_references,
        store,
        phase="phase2",
        protocol_sha256=digest("protocol"),
        condition="A",
        seed=0,
        evaluation_capture_manifest_sha256=eval_capture,
        routed_checkpoint_sha256=digest("A checkpoint"),
        routed_score_manifest_reference=routed_manifest_ref,
        shared_q_score_manifest=q_manifest,
        full_ffn_centering=full_centering,
        expected_process_count=1,
        expected_updates=2,
        expected_tokens=3,
    )
    store.retain(full_manifest_ref, full_manifest_blob)

    resolved = resolve_full_ffn_score_manifest(full_manifest_ref, store)

    assert resolved.tokens == 3
    assert resolved.condition == "A"
    assert resolved.shared_q_score_manifest_reference == q_manifest_ref
    assert resolved.full_ffn_centering_reference == centering_ref
    assert resolved.full_ffn_sst.sum() > 0

    missing = list(full_sink.row_references[:-1])
    with pytest.raises(SharedEvidenceArtifactError, match="coverage"):
        seal_full_ffn_score_manifest(
            missing,
            store,
            phase="phase2",
            protocol_sha256=digest("protocol"),
            condition="A",
            seed=0,
            evaluation_capture_manifest_sha256=eval_capture,
            routed_checkpoint_sha256=digest("A checkpoint"),
            routed_score_manifest_reference=routed_manifest_ref,
            shared_q_score_manifest=q_manifest,
            full_ffn_centering=full_centering,
            expected_process_count=1,
            expected_updates=2,
            expected_tokens=3,
        )


def test_centering_rejects_wrong_split_and_sidecar() -> None:
    selected_sidecar = sidecar()
    evaluation = Dataset(wire(2), selected_sidecar, split="validation")
    with pytest.raises(SharedEvidenceArtifactError, match="training split"):
        seal_full_ffn_centering_from_dataset(evaluation, selected_sidecar)

    training = Dataset(wire(2), selected_sidecar, split="train")
    changed = copy.deepcopy(selected_sidecar)
    changed["sidecar_sha256"] = digest("changed")
    with pytest.raises(Exception):
        seal_full_ffn_centering_from_dataset(training, changed)
