from __future__ import annotations

import copy

import numpy as np
import pytest

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    LEGACY_AGGREGATE_SCHEMA_VERSION,
    MAX_SHARD_TOKENS,
    PROTOCOL_ID,
    ROUTER_WEIGHT_SUM_ATOL,
    SCHEMA_VERSION,
    TOP_K_EXPERTS,
    ContributionBatch,
    ContributionIntegrityError,
    ContributionLegacySchemaError,
    ContributionMetadata,
    ContributionRangeError,
    ContributionSchemaError,
    build_recomposition_sidecar,
    canonical_sha256,
    compute_contribution_statistics,
    contribution_batch_token_identity_sha256,
    deterministic_range_id,
    float32_to_bf16_bits,
    merge_contribution_statistics,
    seal_contribution_manifest,
    validate_contribution_batch,
    validate_contribution_manifest,
    validate_contribution_shard_payload,
    validate_contribution_statistics,
    validate_recomposition_sidecar,
    validate_source_v1_token_keys,
)

CHECKPOINT_SHA = "1" * 64
CONFIG_SHA = "2" * 64
ENVIRONMENT_SHA = "3" * 64
DATASET_SHA = "4" * 64
SOURCE_V1_SHA = "5" * 64
FORK_COMMIT = "6" * 40


def batch(tokens: int = 2, *, offset: float = 0.0) -> ContributionBatch:
    expert = np.arange(tokens * TOP_K_EXPERTS * HIDDEN_SIZE, dtype=np.float32).reshape(
        tokens, TOP_K_EXPERTS, HIDDEN_SIZE
    )
    expert = expert / float(HIDDEN_SIZE * TOP_K_EXPERTS) + offset
    shared = np.arange(tokens * HIDDEN_SIZE, dtype=np.float32).reshape(
        tokens, HIDDEN_SIZE
    )
    shared = shared / float(HIDDEN_SIZE) - offset
    # Strict descending order survives BF16 conversion and sums to one.
    weights = np.asarray(
        [0.25, 0.1875, 0.15625, 0.125, 0.09375, 0.078125, 0.0625, 0.046875],
        dtype=np.float32,
    )
    return ContributionBatch(
        expert_output=float32_to_bf16_bits(expert),
        expert_ids=np.tile(np.arange(TOP_K_EXPERTS, dtype=np.uint8), (tokens, 1)),
        router_weights=float32_to_bf16_bits(np.tile(weights, (tokens, 1))),
        shared_output=float32_to_bf16_bits(shared),
        sequence_id=np.arange(100, 100 + tokens, dtype=np.int64),
        position=np.arange(tokens, dtype=np.uint16),
    )


def sidecar() -> dict:
    scale = float32_to_bf16_bits(np.ones(HIDDEN_SIZE, dtype=np.float32))
    scalar = float32_to_bf16_bits(np.asarray([1.0], dtype=np.float32))
    return build_recomposition_sidecar(
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_fork_commit=FORK_COMMIT,
        source_fingerprints={
            "checkpoint": CHECKPOINT_SHA,
            "config": CONFIG_SHA,
            "environment": ENVIRONMENT_SHA,
            "fork_tree": "7" * 64,
        },
        routed_rmsnorm_scale=scale,
        shared_rmsnorm_scale=scale,
        outer_ffn_rmsnorm_scale=scale,
        layer_scalar=scalar,
        accumulation_order="jax_reduce_sum_slot_axis_in_pinned_fork",
    )


def metadata(sidecar_sha: str) -> dict:
    return ContributionMetadata(
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_fork_commit=FORK_COMMIT,
        dataset_fingerprint=DATASET_SHA,
        split="train",
        config_sha256=CONFIG_SHA,
        environment_sha256=ENVIRONMENT_SHA,
        source_v1_manifest_sha256=SOURCE_V1_SHA,
        recomposition_sidecar_sha256=sidecar_sha,
    ).to_dict()


def shard_entry(
    start: int,
    value: ContributionBatch,
    *,
    digest_character: str,
    metadata_sha: str,
    sidecar_sha: str,
    source_start: int | None = None,
) -> dict:
    end = start + value.tokens
    selected_source_start = start if source_start is None else source_start
    source_end = selected_source_start + value.tokens
    digest = digest_character * 64
    token_digest = contribution_batch_token_identity_sha256(value)
    return {
        "range_id": deterministic_range_id(
            split="train",
            range_start=start,
            range_end=end,
            source_v1_range_start=selected_source_start,
            source_v1_range_end=source_end,
            source_v1_manifest_sha256=SOURCE_V1_SHA,
            token_identity_sha256=token_digest,
        ),
        "range_start": start,
        "range_end": end,
        "source_v1_range_start": selected_source_start,
        "source_v1_range_end": source_end,
        "tokens": value.tokens,
        "host_id": "worker-0",
        "path": f"shards/host-worker-0/{digest}.safetensors",
        "sha256": digest,
        "token_identity_sha256": token_digest,
        "metadata_sha256": metadata_sha,
        "source_v1_manifest_sha256": SOURCE_V1_SHA,
        "recomposition_sidecar_sha256": sidecar_sha,
        "statistics": compute_contribution_statistics(
            value,
            normalized_shared_output=value.shared_output,
        ),
    }


def manifest() -> tuple[dict, dict]:
    recomposition = sidecar()
    meta = metadata(recomposition["sidecar_sha256"])
    meta_sha = canonical_sha256(meta)
    first_batch = batch(2)
    second_batch = batch(3, offset=1.0)
    shards = [
        shard_entry(
            0,
            first_batch,
            digest_character="8",
            metadata_sha=meta_sha,
            sidecar_sha=recomposition["sidecar_sha256"],
        ),
        shard_entry(
            2,
            second_batch,
            digest_character="9",
            metadata_sha=meta_sha,
            sidecar_sha=recomposition["sidecar_sha256"],
        ),
    ]
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "sealed": True,
        "metadata": meta,
        "metadata_sha256": meta_sha,
        "source_v1_manifest": {
            "schema_version": LEGACY_AGGREGATE_SCHEMA_VERSION,
            "manifest_sha256": SOURCE_V1_SHA,
            "split": "train",
            "total_tokens": 5,
        },
        "recomposition_sidecar_sha256": recomposition["sidecar_sha256"],
        "total_tokens": 5,
        "shard_count": 2,
        "statistics": merge_contribution_statistics(
            [entry["statistics"] for entry in shards]
        ),
        "shards": shards,
    }
    return seal_contribution_manifest(unsigned), recomposition


def reseal(value: dict) -> dict:
    unsigned = copy.deepcopy(value)
    unsigned.pop("manifest_sha256", None)
    return seal_contribution_manifest(unsigned)


def test_exact_contribution_schema_accepts_wire_batch():
    original = batch()
    checked = validate_contribution_batch(original)

    assert checked.tokens == 2
    assert checked.expert_output.shape == (2, 8, 2816)
    assert checked.expert_output.dtype == np.uint16
    assert checked.expert_ids.dtype == np.uint8
    assert checked.router_weights.dtype == np.uint16
    assert checked.shared_output.dtype == np.uint16
    assert checked.sequence_id.dtype == np.int64
    assert checked.position.dtype == np.uint16
    assert all(
        np.asarray(getattr(checked, field)).flags.c_contiguous
        for field in checked.__dataclass_fields__
    )


def test_batch_rejects_old_schema_wrong_wire_dtype_and_shape():
    original = batch()
    with pytest.raises(ContributionLegacySchemaError, match="aggregate"):
        validate_contribution_batch(
            {
                "activation": np.zeros((2, HIDDEN_SIZE), dtype=np.uint16),
                "expert_ids": original.expert_ids,
                "router_weights": original.router_weights,
                "sequence_id": original.sequence_id,
                "position": original.position,
            }
        )
    wrong_dtype = dataclass_replace(
        original, expert_output=original.expert_output.astype(np.float32)
    )
    with pytest.raises(ContributionSchemaError, match="wire dtype uint16"):
        validate_contribution_batch(wrong_dtype)
    wrong_shape = dataclass_replace(
        original, shared_output=original.shared_output[:, :-1]
    )
    with pytest.raises(ContributionSchemaError, match="shared_output must have shape"):
        validate_contribution_batch(wrong_shape)


def dataclass_replace(
    value: ContributionBatch, **changes: np.ndarray
) -> ContributionBatch:
    fields = {name: getattr(value, name) for name in value.__dataclass_fields__}
    fields.update(changes)
    return ContributionBatch(**fields)


def test_batch_enforces_shard_limit_route_and_token_invariants():
    original = batch(2)
    with pytest.raises(ContributionSchemaError, match="token count"):
        validate_contribution_batch(original, max_tokens=1)

    duplicate_ids = original.expert_ids.copy()
    duplicate_ids[:, 1] = duplicate_ids[:, 0]
    with pytest.raises(ContributionSchemaError, match="unique"):
        validate_contribution_batch(
            dataclass_replace(original, expert_ids=duplicate_ids)
        )

    out_of_range_ids = original.expert_ids.copy()
    out_of_range_ids[:, -1] = 128
    with pytest.raises(ContributionSchemaError, match=r"\[0,127\]"):
        validate_contribution_batch(
            dataclass_replace(original, expert_ids=out_of_range_ids)
        )

    wrong_order = original.router_weights.copy()
    wrong_order[:, [0, 1]] = wrong_order[:, [1, 0]]
    with pytest.raises(ContributionSchemaError, match="descending weight"):
        validate_contribution_batch(
            dataclass_replace(original, router_weights=wrong_order)
        )

    bad_sum = float32_to_bf16_bits(np.full((2, TOP_K_EXPERTS), 0.124, dtype=np.float32))
    assert (
        abs(
            float(
                (bad_sum.astype(np.uint32) << np.uint32(16)).view(np.float32)[0].sum()
            )
            - 1.0
        )
        > ROUTER_WEIGHT_SUM_ATOL
    )
    with pytest.raises(ContributionSchemaError, match="sum to one"):
        validate_contribution_batch(dataclass_replace(original, router_weights=bad_sum))

    nonfinite_expert = original.expert_output.copy()
    nonfinite_expert[0, 0, 0] = np.uint16(0x7F80)
    with pytest.raises(ContributionSchemaError, match="expert_output.*non-finite"):
        validate_contribution_batch(
            dataclass_replace(original, expert_output=nonfinite_expert)
        )

    nonfinite_shared = original.shared_output.copy()
    nonfinite_shared[0, 0] = np.uint16(0x7FC0)
    with pytest.raises(ContributionSchemaError, match="shared_output.*non-finite"):
        validate_contribution_batch(
            dataclass_replace(original, shared_output=nonfinite_shared)
        )

    duplicate_tokens = dataclass_replace(
        original,
        sequence_id=np.asarray([100, 100], dtype=np.int64),
        position=np.asarray([0, 0], dtype=np.uint16),
    )
    with pytest.raises(ContributionSchemaError, match="duplicate"):
        validate_contribution_batch(duplicate_tokens)

    assert MAX_SHARD_TOKENS == 16_384


def test_equal_weight_slots_use_ascending_expert_id_tiebreak():
    original = batch(1)
    tied_weights = np.full((1, TOP_K_EXPERTS), 1 / TOP_K_EXPERTS, dtype=np.float32)
    ascending = dataclass_replace(
        original, router_weights=float32_to_bf16_bits(tied_weights)
    )
    validate_contribution_batch(ascending)

    descending_ids = ascending.expert_ids.copy()
    descending_ids[:, [0, 1]] = descending_ids[:, [1, 0]]
    with pytest.raises(ContributionSchemaError, match="tie-breaker"):
        validate_contribution_batch(
            dataclass_replace(ascending, expert_ids=descending_ids)
        )


def test_statistics_encode_common_mean_and_weight_squared_centering():
    original = batch(2)
    stats = compute_contribution_statistics(
        original, normalized_shared_output=original.shared_output
    )
    checked = validate_contribution_statistics(stats, expected_tokens=2)

    assert checked["component_count"] == 16
    assert checked["pooled_expert_output"]["count"] == 16
    assert checked["shared_raw"]["count"] == 2
    assert checked["router_rank"]["count"] == [2] * 8
    assert sum(checked["expert_id"]["count"]) == 16

    expert = (original.expert_output.astype(np.uint32) << np.uint32(16)).view(
        np.float32
    )
    weights = (original.router_weights.astype(np.uint32) << np.uint32(16)).view(
        np.float32
    )
    expected = np.einsum(
        "ns,nsh->h",
        np.square(weights, dtype=np.float32).astype(np.float64),
        expert.astype(np.float64),
    )
    assert np.allclose(
        checked["weighted_contribution"]["weight_squared_output_sum"],
        expected,
        rtol=0,
        atol=0,
    )


def test_statistics_fail_closed_on_count_energy_and_finiteness():
    original = batch(1)
    stats = compute_contribution_statistics(
        original, normalized_shared_output=original.shared_output
    )
    bad_count = copy.deepcopy(stats)
    bad_count["component_count"] = 7
    with pytest.raises(ContributionIntegrityError, match="top_k"):
        validate_contribution_statistics(bad_count)

    bad_energy = copy.deepcopy(stats)
    bad_energy["expert_id"]["energy"][0] += 1.0
    with pytest.raises(ContributionIntegrityError, match="conserve"):
        validate_contribution_statistics(bad_energy)

    bad_finite = copy.deepcopy(stats)
    bad_finite["shared_raw"]["sum"][0] = float("nan")
    with pytest.raises(ContributionIntegrityError, match="finite"):
        validate_contribution_statistics(bad_finite)


def test_statistics_merge_is_deterministic_and_conserves_counts():
    first = batch(2)
    second = batch(3, offset=4)
    pieces = [
        compute_contribution_statistics(
            item, normalized_shared_output=item.shared_output
        )
        for item in (first, second)
    ]
    merged = merge_contribution_statistics(pieces)

    assert merged["token_count"] == 5
    assert merged["component_count"] == 40
    assert merged["router_rank"]["count"] == [5] * 8
    assert sum(merged["expert_id"]["count"]) == 40
    assert merge_contribution_statistics(pieces) == merged


def test_recomposition_sidecar_is_canonical_content_addressed_and_exact():
    first = sidecar()
    second = sidecar()

    assert first == second
    assert validate_recomposition_sidecar(first) == first
    assert first["checkpoint_sha256"] == CHECKPOINT_SHA
    assert first["maxtext_fork_commit"] == FORK_COMMIT
    assert first["tensors"]["routed_rmsnorm_scale"]["shape"] == [HIDDEN_SIZE]
    assert first["tensors"]["layer_scalar"]["shape"] == [1]

    tampered = copy.deepcopy(first)
    tampered["tensors"]["layer_scalar"]["data"][0] ^= 1
    with pytest.raises(ContributionIntegrityError, match="SHA-256 mismatch"):
        validate_recomposition_sidecar(tampered)


def test_sidecar_rejects_missing_fork_policy_and_nonfinite_scale():
    scale = float32_to_bf16_bits(np.ones(HIDDEN_SIZE, dtype=np.float32))
    with pytest.raises(ContributionSchemaError, match="Git commit"):
        build_recomposition_sidecar(
            checkpoint_sha256=CHECKPOINT_SHA,
            maxtext_fork_commit="not-a-commit",
            source_fingerprints={"checkpoint": CHECKPOINT_SHA},
            routed_rmsnorm_scale=scale,
            shared_rmsnorm_scale=scale,
            outer_ffn_rmsnorm_scale=scale,
            layer_scalar=float32_to_bf16_bits([1.0]),
            accumulation_order="ordered",
        )
    nonfinite = scale.copy()
    nonfinite[0] = np.uint16(0x7F80)
    with pytest.raises(ContributionSchemaError, match="non-finite"):
        build_recomposition_sidecar(
            checkpoint_sha256=CHECKPOINT_SHA,
            maxtext_fork_commit=FORK_COMMIT,
            source_fingerprints={"checkpoint": CHECKPOINT_SHA},
            routed_rmsnorm_scale=nonfinite,
            shared_rmsnorm_scale=scale,
            outer_ffn_rmsnorm_scale=scale,
            layer_scalar=float32_to_bf16_bits([1.0]),
            accumulation_order="ordered",
        )


def test_deterministic_range_id_binds_source_split_and_bounds():
    token_digest = contribution_batch_token_identity_sha256(batch(2))
    identity = deterministic_range_id(
        split="train",
        range_start=0,
        range_end=16_384,
        source_v1_range_start=32_768,
        source_v1_range_end=49_152,
        source_v1_manifest_sha256=SOURCE_V1_SHA,
        token_identity_sha256=token_digest,
    )
    assert (
        deterministic_range_id(
            split="train",
            range_start=0,
            range_end=16_384,
            source_v1_range_start=32_768,
            source_v1_range_end=49_152,
            source_v1_manifest_sha256=SOURCE_V1_SHA,
            token_identity_sha256=token_digest,
        )
        == identity
    )
    assert (
        deterministic_range_id(
            split="eval",
            range_start=0,
            range_end=16_384,
            source_v1_range_start=32_768,
            source_v1_range_end=49_152,
            source_v1_manifest_sha256=SOURCE_V1_SHA,
            token_identity_sha256=token_digest,
        )
        != identity
    )
    assert (
        deterministic_range_id(
            split="train",
            range_start=0,
            range_end=16_384,
            source_v1_range_start=32_768,
            source_v1_range_end=49_152,
            source_v1_manifest_sha256="a" * 64,
            token_identity_sha256=token_digest,
        )
        != identity
    )
    with pytest.raises(ContributionRangeError, match="exceeds"):
        deterministic_range_id(
            split="train",
            range_start=0,
            range_end=16_385,
            source_v1_range_start=32_768,
            source_v1_range_end=49_153,
            source_v1_manifest_sha256=SOURCE_V1_SHA,
            token_identity_sha256=token_digest,
        )

    assert (
        deterministic_range_id(
            split="train",
            range_start=0,
            range_end=16_384,
            source_v1_range_start=32_768,
            source_v1_range_end=49_152,
            source_v1_manifest_sha256=SOURCE_V1_SHA,
            token_identity_sha256="b" * 64,
        )
        != identity
    )
    assert (
        deterministic_range_id(
            split="train",
            range_start=0,
            range_end=16_384,
            source_v1_range_start=49_152,
            source_v1_range_end=65_536,
            source_v1_manifest_sha256=SOURCE_V1_SHA,
            token_identity_sha256=token_digest,
        )
        != identity
    )
    with pytest.raises(ContributionRangeError, match="same token count"):
        deterministic_range_id(
            split="train",
            range_start=0,
            range_end=16_384,
            source_v1_range_start=32_768,
            source_v1_range_end=40_000,
            source_v1_manifest_sha256=SOURCE_V1_SHA,
            token_identity_sha256=token_digest,
        )


def test_shard_payload_is_bound_to_exact_v1_token_key_sequence():
    value = batch(2)
    recomposition = sidecar()
    meta = metadata(recomposition["sidecar_sha256"])
    entry = shard_entry(
        0,
        value,
        digest_character="8",
        metadata_sha=canonical_sha256(meta),
        sidecar_sha=recomposition["sidecar_sha256"],
    )
    validate_contribution_shard_payload(value, entry)
    validate_source_v1_token_keys(value.sequence_id, value.position, entry)

    # Same shapes, routes, and values, but a payload harvested from another
    # source slice cannot claim this v1 range.
    wrong_range = dataclass_replace(
        value,
        sequence_id=value.sequence_id + 10_000,
    )
    with pytest.raises(ContributionIntegrityError, match="token-key sequence"):
        validate_contribution_shard_payload(wrong_range, entry)
    with pytest.raises(ContributionIntegrityError, match="source v1 token-key"):
        validate_source_v1_token_keys(
            wrong_range.sequence_id, wrong_range.position, entry
        )


def test_sealed_manifest_roundtrip_binds_sidecar_v1_and_statistics():
    value, recomposition = manifest()
    checked = validate_contribution_manifest(value, recomposition_sidecar=recomposition)

    assert checked["total_tokens"] == 5
    assert checked["shard_count"] == 2
    assert checked["metadata"]["source_v1_manifest_sha256"] == SOURCE_V1_SHA
    assert (
        checked["metadata"]["recomposition_sidecar_sha256"]
        == recomposition["sidecar_sha256"]
    )
    assert checked["statistics"]["token_count"] == 5


def test_manifest_rejects_old_schema_and_canonical_hash_tampering():
    with pytest.raises(ContributionLegacySchemaError, match="aggregate"):
        validate_contribution_manifest(
            {"schema_version": LEGACY_AGGREGATE_SCHEMA_VERSION}
        )
    value, _ = manifest()
    value["total_tokens"] = 6
    with pytest.raises(ContributionIntegrityError, match="canonical SHA-256"):
        validate_contribution_manifest(value)


def test_manifest_rejects_range_gaps_nondeterminism_and_oversized_source():
    value, _ = manifest()
    gap = copy.deepcopy(value)
    gap["shards"][1]["range_start"] = 3
    gap["shards"][1]["range_end"] = 6
    gap["shards"][1]["range_id"] = deterministic_range_id(
        split="train",
        range_start=3,
        range_end=6,
        source_v1_range_start=gap["shards"][1]["source_v1_range_start"],
        source_v1_range_end=gap["shards"][1]["source_v1_range_end"],
        source_v1_manifest_sha256=SOURCE_V1_SHA,
        token_identity_sha256=gap["shards"][1]["token_identity_sha256"],
    )
    gap["total_tokens"] = 6
    gap["source_v1_manifest"]["total_tokens"] = 6
    with pytest.raises(ContributionRangeError, match="gap or overlap"):
        validate_contribution_manifest(reseal(gap))

    wrong_id = copy.deepcopy(value)
    wrong_id["shards"][0]["range_id"] = "caller-selected-id"
    with pytest.raises(ContributionRangeError, match="deterministic"):
        validate_contribution_manifest(reseal(wrong_id))

    beyond_source = copy.deepcopy(value)
    beyond_source["source_v1_manifest"]["total_tokens"] = 4
    with pytest.raises(ContributionIntegrityError, match="exceeds"):
        validate_contribution_manifest(reseal(beyond_source))

    overlapping_source = copy.deepcopy(value)
    second = overlapping_source["shards"][1]
    second["source_v1_range_start"] = 1
    second["source_v1_range_end"] = 4
    second["range_id"] = deterministic_range_id(
        split="train",
        range_start=second["range_start"],
        range_end=second["range_end"],
        source_v1_range_start=1,
        source_v1_range_end=4,
        source_v1_manifest_sha256=SOURCE_V1_SHA,
        token_identity_sha256=second["token_identity_sha256"],
    )
    with pytest.raises(ContributionRangeError, match="overlap or duplicate"):
        validate_contribution_manifest(reseal(overlapping_source))


def test_manifest_rejects_metadata_sidecar_and_statistic_identity_changes():
    value, recomposition = manifest()

    changed_sidecar = copy.deepcopy(recomposition)
    changed_sidecar["accumulation_order"] = "different-order"
    unsigned = {
        key: val for key, val in changed_sidecar.items() if key != "sidecar_sha256"
    }
    changed_sidecar["sidecar_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ContributionIntegrityError, match="differs"):
        validate_contribution_manifest(value, recomposition_sidecar=changed_sidecar)

    bad_stats = copy.deepcopy(value)
    bad_stats["statistics"]["shared_raw"]["squared_norm"] += 1.0
    with pytest.raises(ContributionIntegrityError, match="deterministic shard merge"):
        validate_contribution_manifest(reseal(bad_stats))

    bad_metadata = copy.deepcopy(value)
    bad_metadata["metadata"]["checkpoint_sha256"] = "a" * 64
    with pytest.raises(ContributionIntegrityError, match="metadata SHA-256"):
        validate_contribution_manifest(reseal(bad_metadata))
