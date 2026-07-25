from __future__ import annotations

import dataclasses
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from moe_sae.contribution_artifacts import canonical_json_bytes, canonical_sha256
from moe_sae.contribution_input import ContributionInputReceipt
from moe_sae.contribution_sae import CONTRIBUTION_CONDITIONS
from moe_sae.contribution_score_artifacts import (
    SCORE_MANIFEST_MEDIA_TYPE,
    ContributionScoreArtifactError,
    ScoreArtifactRef,
    resolve_score_manifest,
    resolve_score_manifest_matrix,
    resolve_score_manifest_matrix_with_receipt,
    resolve_score_row_shard,
    seal_score_manifest,
    seal_score_row_shard,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class MemoryResolver:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def retain(self, reference: ScoreArtifactRef, blob: bytes) -> None:
        assert hashlib.sha256(blob).hexdigest() == reference.sha256
        self.objects[reference.sha256] = blob

    def __call__(self, reference: ScoreArtifactRef) -> bytes:
        return self.objects[reference.sha256]


def input_receipt(
    condition: str,
    process_index: int,
    *,
    randomization_key: int = 0,
    capture: str | None = None,
) -> ContributionInputReceipt:
    unsigned = {
        "schema_version": "gemma4-contribution-training-input-v1",
        "capture_manifest_sha256": capture or digest("eval capture"),
        "condition": condition,
        "randomization_key": randomization_key,
        "layout_sha256": digest(f"{condition} layout {randomization_key}"),
        "total_tokens": 8,
        "global_microbatch_tokens": 4,
        "microbatches_per_update": 1,
        "planned_updates": 2,
        "process_count": 2,
        "process_index": process_index,
        "local_rows_per_microbatch": 2,
    }
    return ContributionInputReceipt(
        **unsigned,
        receipt_sha256=canonical_sha256(unsigned),
    )


def evidence(
    process_index: int,
    update_index: int,
    *,
    sse_offset: float,
    duplicate_first_key: bool = False,
):
    start = 4 * process_index + 2 * update_index
    sequence_id = np.asarray([[start, start + 1]], dtype=np.int64)
    position = np.asarray([[0, 1]], dtype=np.uint16)
    if duplicate_first_key:
        sequence_id[0, 0] = 0
        position[0, 0] = 0
    valid = np.ones((1, 2), dtype=np.bool_)
    component_sst = np.asarray([[2.0, 3.0]], dtype=np.float64)
    routed_sst = np.asarray([[5.0, 7.0]], dtype=np.float64)
    scores = SimpleNamespace(
        valid_mask=valid,
        component_token_sse=np.asarray(
            [[0.1 + sse_offset, 0.2 + sse_offset]],
            dtype=np.float32,
        ),
        component_token_sst=component_sst,
    )
    routed = SimpleNamespace(
        valid_mask=valid.copy(),
        token_sse=np.asarray(
            [[0.3 + sse_offset, 0.4 + sse_offset]],
            dtype=np.float64,
        ),
        token_sst=routed_sst,
        recomposition_sidecar_sha256=digest("sidecar"),
    )
    return (
        SimpleNamespace(
            update_index=update_index,
            sequence_id=sequence_id,
            position=position,
            scores=scores,
        ),
        routed,
    )


def retain_condition(
    resolver: MemoryResolver,
    condition: str,
    *,
    seed: int = 0,
    token_mutation: tuple[int, int] | None = None,
    sst_mutation: tuple[int, int] | None = None,
    sst_delta: float = 1.0,
):
    randomization_key = 7 if condition == "R" else 0
    receipts = tuple(
        input_receipt(
            condition,
            process_index,
            randomization_key=randomization_key,
        )
        for process_index in range(2)
    )
    row_refs = []
    for process_index in range(2):
        for update_index in range(2):
            score_evidence, routed = evidence(
                process_index,
                update_index,
                sse_offset=0.01 * CONTRIBUTION_CONDITIONS.index(condition),
            )
            if token_mutation == (process_index, update_index):
                score_evidence.sequence_id[0, 0] += 10_000
            if sst_mutation == (process_index, update_index):
                score_evidence.scores.component_token_sst[0, 0] += sst_delta
            reference, blob = seal_score_row_shard(
                score_evidence,
                routed,
                condition=condition,
                seed=seed,
                process_index=process_index,
                evaluation_capture_manifest_sha256=digest("eval capture"),
                checkpoint_sha256=digest(f"{condition} checkpoint"),
                recomposition_sidecar_sha256=digest("sidecar"),
                centering_mean_sha256=digest("centering mean"),
                evaluation_input_receipt_sha256=receipts[process_index].receipt_sha256,
            )
            resolver.retain(reference, blob)
            row_refs.append(reference)
    manifest_ref, manifest_blob = seal_score_manifest(
        row_refs,
        resolver,
        phase="phase2",
        protocol_sha256=digest("protocol"),
        condition=condition,
        seed=seed,
        randomization_key=randomization_key,
        training_capture_manifest_sha256=digest("train capture"),
        evaluation_capture_manifest_sha256=digest("eval capture"),
        recomposition_sidecar_sha256=digest("sidecar"),
        run_manifest_sha256=digest(f"{condition} run"),
        config_sha256=digest(f"{condition} config"),
        checkpoint_sha256=digest(f"{condition} checkpoint"),
        layout_sha256=digest(f"{condition} layout {randomization_key}"),
        centering_mean_sha256=digest("centering mean"),
        training_input_receipt_sha256s=(
            digest(f"{condition} train input 0"),
            digest(f"{condition} train input 1"),
        ),
        evaluation_input_receipts=receipts,
        expected_tokens=8,
    )
    resolver.retain(manifest_ref, manifest_blob)
    return manifest_ref


def test_score_row_and_manifest_round_trip_are_content_addressed():
    resolver = MemoryResolver()
    reference = retain_condition(resolver, "A")
    resolved = resolve_score_manifest(reference, resolver)

    assert reference.media_type == SCORE_MANIFEST_MEDIA_TYPE
    assert resolved.condition == "A"
    assert resolved.tokens == 8
    assert resolved.sequence_id.tolist() == list(range(8))
    assert resolved.token_count.tolist() == [1] * 8
    assert resolved.position_bitmap[:, 0].tolist() == [1, 2, 1, 2, 1, 2, 1, 2]
    assert len(resolved.row_references) == 4
    assert len(resolved.evaluation_input_receipts) == 2

    # Sealing the same scientific evidence produces the same object identities.
    second = MemoryResolver()
    assert retain_condition(second, "A") == reference
    assert second.objects == resolver.objects


def test_resolver_rejects_corrupt_or_wrong_sized_retained_object():
    resolver = MemoryResolver()
    reference = retain_condition(resolver, "A")
    bad = dataclasses.replace(reference, size_bytes=reference.size_bytes + 1)
    with pytest.raises(ContributionScoreArtifactError, match="size differs"):
        resolve_score_manifest(bad, resolver)

    blob = resolver.objects[reference.sha256]
    forged_blob = blob[:-1] + bytes([blob[-1] ^ 1])
    forged = dataclasses.replace(
        reference,
        sha256=hashlib.sha256(forged_blob).hexdigest(),
    )
    resolver.objects[forged.sha256] = forged_blob
    with pytest.raises(ContributionScoreArtifactError):
        resolve_score_manifest(forged, resolver)


def test_manifest_rejects_duplicate_keys_across_row_artifacts():
    resolver = MemoryResolver()
    condition = "A"
    receipts = (input_receipt("A", 0), input_receipt("A", 1))
    row_refs = []
    for process_index in range(2):
        for update_index in range(2):
            score_evidence, routed = evidence(
                process_index,
                update_index,
                sse_offset=0.0,
                duplicate_first_key=(process_index, update_index) == (1, 1),
            )
            ref, blob = seal_score_row_shard(
                score_evidence,
                routed,
                condition=condition,
                seed=0,
                process_index=process_index,
                evaluation_capture_manifest_sha256=digest("eval capture"),
                checkpoint_sha256=digest("A checkpoint"),
                recomposition_sidecar_sha256=digest("sidecar"),
                centering_mean_sha256=digest("centering mean"),
                evaluation_input_receipt_sha256=receipts[process_index].receipt_sha256,
            )
            resolver.retain(ref, blob)
            row_refs.append(ref)
    with pytest.raises(ContributionScoreArtifactError, match="duplicate"):
        seal_score_manifest(
            row_refs,
            resolver,
            phase="phase2",
            protocol_sha256=digest("protocol"),
            condition="A",
            seed=0,
            randomization_key=0,
            training_capture_manifest_sha256=digest("train capture"),
            evaluation_capture_manifest_sha256=digest("eval capture"),
            recomposition_sidecar_sha256=digest("sidecar"),
            run_manifest_sha256=digest("A run"),
            config_sha256=digest("A config"),
            checkpoint_sha256=digest("A checkpoint"),
            layout_sha256=digest("A layout 0"),
            centering_mean_sha256=digest("centering mean"),
            training_input_receipt_sha256s=(digest("train input"),),
            evaluation_input_receipts=receipts,
            expected_tokens=8,
        )


def test_manifest_requires_complete_process_update_grid():
    resolver = MemoryResolver()
    reference = retain_condition(resolver, "A")
    raw = json.loads(resolver.objects[reference.sha256])
    missing = raw["row_artifacts"][:-1]
    receipts = tuple(
        ContributionInputReceipt(**item) for item in raw["evaluation_input_receipts"]
    )
    with pytest.raises(
        ContributionScoreArtifactError,
        match="process/update coverage",
    ):
        seal_score_manifest(
            missing,
            resolver,
            phase="phase2",
            protocol_sha256=digest("protocol"),
            condition="A",
            seed=0,
            randomization_key=0,
            training_capture_manifest_sha256=digest("train capture"),
            evaluation_capture_manifest_sha256=digest("eval capture"),
            recomposition_sidecar_sha256=digest("sidecar"),
            run_manifest_sha256=digest("A run"),
            config_sha256=digest("A config"),
            checkpoint_sha256=digest("A checkpoint"),
            layout_sha256=digest("A layout 0"),
            centering_mean_sha256=digest("centering mean"),
            training_input_receipt_sha256s=(digest("train input"),),
            evaluation_input_receipts=receipts,
            expected_tokens=8,
        )


def test_manifest_rejects_rehashed_aggregate_tampering():
    resolver = MemoryResolver()
    reference = retain_condition(resolver, "A")
    raw = json.loads(resolver.objects[reference.sha256])
    raw["component_sse"] += 10.0
    tampered = canonical_json_bytes(raw)
    tampered_ref = ScoreArtifactRef(
        sha256=hashlib.sha256(tampered).hexdigest(),
        size_bytes=len(tampered),
        media_type=SCORE_MANIFEST_MEDIA_TYPE,
    )
    resolver.retain(tampered_ref, tampered)
    with pytest.raises(ContributionScoreArtifactError, match="differs from retained"):
        resolve_score_manifest(tampered_ref, resolver)


def test_matrix_requires_exact_common_keys_sst_and_centering():
    resolver = MemoryResolver()
    refs = [
        retain_condition(resolver, condition) for condition in CONTRIBUTION_CONDITIONS
    ]
    matrix = resolve_score_manifest_matrix(refs, resolver, expected_seeds=(0,))
    assert set(matrix) == {(0, condition) for condition in CONTRIBUTION_CONDITIONS}

    token_resolver = MemoryResolver()
    token_refs = [
        retain_condition(
            token_resolver,
            condition,
            token_mutation=(0, 0) if condition == "F" else None,
        )
        for condition in CONTRIBUTION_CONDITIONS
    ]
    with pytest.raises(ContributionScoreArtifactError, match="token inventories"):
        resolve_score_manifest_matrix(
            token_refs,
            token_resolver,
            expected_seeds=(0,),
        )

    sst_resolver = MemoryResolver()
    sst_refs = [
        retain_condition(
            sst_resolver,
            condition,
            sst_mutation=(0, 0) if condition == "F" else None,
        )
        for condition in CONTRIBUTION_CONDITIONS
    ]
    with pytest.raises(
        ContributionScoreArtifactError,
        match="materially different component SST",
    ):
        resolve_score_manifest_matrix(
            sst_refs,
            sst_resolver,
            expected_seeds=(0,),
        )


def test_matrix_canonicalizes_bounded_binary32_sst_drift_with_receipt():
    resolver = MemoryResolver()
    refs = [
        retain_condition(
            resolver,
            condition,
            sst_mutation=(0, 0) if condition == "F" else None,
            sst_delta=1e-7,
        )
        for condition in CONTRIBUTION_CONDITIONS
    ]
    matrix, receipt = resolve_score_manifest_matrix_with_receipt(
        refs,
        resolver,
        expected_seeds=(0,),
    )
    canonical = matrix[(0, "A")].component_sst
    assert all(
        np.array_equal(matrix[(0, condition)].component_sst, canonical)
        for condition in CONTRIBUTION_CONDITIONS
    )
    assert receipt.canonical_condition == "A"
    assert receipt.canonical_seed == 0
    assert receipt.component_nonexact_values == 1
    assert receipt.routed_nonexact_values == 0
    assert receipt.component_max_absolute_difference == pytest.approx(1e-7)
    assert receipt.component_max_relative_difference < receipt.relative_tolerance
    body = receipt.to_dict()
    claimed = body.pop("receipt_sha256")
    assert claimed == canonical_sha256(body)


def test_matrix_requires_nonempty_unique_expected_seeds():
    resolver = MemoryResolver()
    with pytest.raises(ContributionScoreArtifactError, match="nonempty and unique"):
        resolve_score_manifest_matrix([], resolver, expected_seeds=())
    with pytest.raises(ContributionScoreArtifactError, match="nonempty and unique"):
        resolve_score_manifest_matrix([], resolver, expected_seeds=(0, 0))


def test_row_resolver_rejects_reference_media_type_confusion():
    resolver = MemoryResolver()
    manifest = retain_condition(resolver, "A")
    with pytest.raises(ContributionScoreArtifactError, match="wrong media"):
        resolve_score_row_shard(manifest, resolver)
