from __future__ import annotations

import copy

import numpy as np
import pytest

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    canonical_sha256,
    float32_to_bf16_bits,
)
from moe_sae.contribution_transcoder_artifacts import (
    PROTOCOL_ID,
    SCHEMA_VERSION,
    RoutedInputBatch,
    RoutedInputIntegrityError,
    RoutedInputMetadata,
    RoutedInputRangeError,
    RoutedInputSchemaError,
    compute_routed_input_statistics,
    deterministic_overlay_range_id,
    merge_routed_input_statistics,
    routed_input_token_identity_sha256,
    seal_routed_input_manifest,
    target_manifest_identity,
    validate_routed_input_batch,
    validate_routed_input_manifest,
)
from tests.test_contribution_artifacts import manifest as contribution_manifest


def routed_batch(tokens: int, *, offset: int = 0) -> RoutedInputBatch:
    values = np.arange(tokens * HIDDEN_SIZE, dtype=np.float32).reshape(
        tokens, HIDDEN_SIZE
    )
    values = values / HIDDEN_SIZE + offset
    return RoutedInputBatch(
        routed_input=float32_to_bf16_bits(values),
        sequence_id=np.arange(100, 100 + tokens, dtype=np.int64),
        position=np.arange(tokens, dtype=np.uint16),
    )


def metadata(target: dict) -> dict:
    parent = target["metadata"]
    return RoutedInputMetadata(
        checkpoint_sha256=parent["checkpoint_sha256"],
        maxtext_fork_commit=parent["maxtext_fork_commit"],
        dataset_fingerprint=parent["dataset_fingerprint"],
        split=parent["split"],
        config_sha256="a" * 64,
        environment_sha256="b" * 64,
        target_manifest_sha256=target["manifest_sha256"],
    ).to_dict()


def overlay_manifest(target: dict) -> dict:
    meta = metadata(target)
    meta_sha = canonical_sha256(meta)
    batches = (routed_batch(2), routed_batch(3))
    shards = []
    for index, (parent, batch) in enumerate(
        zip(target["shards"], batches, strict=True)
    ):
        # Match the immutable target token keys exactly.
        batch = RoutedInputBatch(
            routed_input=batch.routed_input,
            sequence_id=np.arange(100, 100 + batch.tokens, dtype=np.int64),
            position=np.arange(batch.tokens, dtype=np.uint16),
        )
        token_sha = routed_input_token_identity_sha256(batch)
        assert token_sha == parent["token_identity_sha256"]
        digest = str(index + 1) * 64
        range_id = deterministic_overlay_range_id(
            split=meta["split"],
            range_start=parent["range_start"],
            range_end=parent["range_end"],
            target_manifest_sha256=target["manifest_sha256"],
            target_range_id=parent["range_id"],
            token_identity_sha256=token_sha,
        )
        shards.append(
            {
                "range_id": range_id,
                "target_range_id": parent["range_id"],
                "range_start": parent["range_start"],
                "range_end": parent["range_end"],
                "tokens": parent["tokens"],
                "host_id": "worker-0",
                "path": f"shards/host-worker-0/{digest}.safetensors",
                "sha256": digest,
                "token_identity_sha256": token_sha,
                "metadata_sha256": meta_sha,
                "target_manifest_sha256": target["manifest_sha256"],
                "statistics": compute_routed_input_statistics(batch),
                "object": {
                    "uri": (
                        "gs://test/overlay/shards/host-worker-0/"
                        f"{digest}.safetensors"
                    ),
                    "generation": index + 1,
                    "size_bytes": 123,
                    "crc32c": "AAAAAA==",
                    "sha256": digest,
                },
            }
        )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "sealed": True,
        "metadata": meta,
        "metadata_sha256": meta_sha,
        "target_manifest": target_manifest_identity(target),
        "total_tokens": target["total_tokens"],
        "shard_count": target["shard_count"],
        "statistics": merge_routed_input_statistics(
            [entry["statistics"] for entry in shards]
        ),
        "shards": shards,
    }
    return seal_routed_input_manifest(unsigned)


def test_batch_has_exact_three_field_wire_schema_and_finite_values():
    original = routed_batch(2)
    checked = validate_routed_input_batch(original)

    assert checked.routed_input.shape == (2, HIDDEN_SIZE)
    assert checked.routed_input.dtype == np.uint16
    assert checked.sequence_id.dtype == np.int64
    assert checked.position.dtype == np.uint16

    with pytest.raises(RoutedInputSchemaError, match="exactly"):
        validate_routed_input_batch(
            {
                "routed_input": original.routed_input,
                "sequence_id": original.sequence_id,
                "position": original.position,
                "expert_ids": np.zeros((2, 8), dtype=np.uint8),
            }
        )
    nonfinite = original.routed_input.copy()
    nonfinite[0, 0] = np.uint16(0x7F80)
    with pytest.raises(RoutedInputSchemaError, match="non-finite"):
        validate_routed_input_batch(
            RoutedInputBatch(nonfinite, original.sequence_id, original.position)
        )


def test_statistics_are_fp64_sufficient_and_merge_deterministically():
    first = routed_batch(2)
    second = routed_batch(3, offset=2)
    first_stats = compute_routed_input_statistics(first)
    second_stats = compute_routed_input_statistics(second)
    merged = merge_routed_input_statistics([first_stats, second_stats])

    decoded = np.concatenate(
        [
            (batch.routed_input.astype(np.uint32) << np.uint32(16)).view(
                np.float32
            )
            for batch in (first, second)
        ]
    ).astype(np.float64)
    assert merged["count"] == 5
    np.testing.assert_array_equal(merged["sum"], decoded.sum(axis=0))
    assert merged["squared_norm"] == float(np.square(decoded).sum())


def test_manifest_binds_every_exact_target_range_and_token_sequence():
    target, _sidecar = contribution_manifest()
    value = overlay_manifest(target)

    checked = validate_routed_input_manifest(value, target_manifest=target)

    assert checked["target_manifest"]["manifest_sha256"] == target["manifest_sha256"]
    assert [entry["target_range_id"] for entry in checked["shards"]] == [
        entry["range_id"] for entry in target["shards"]
    ]

    tampered = copy.deepcopy(value)
    tampered["shards"][0]["target_range_id"] = target["shards"][1]["range_id"]
    tampered.pop("manifest_sha256")
    tampered = seal_routed_input_manifest(tampered)
    with pytest.raises(
        (RoutedInputIntegrityError, RoutedInputRangeError), match="range"
    ):
        validate_routed_input_manifest(tampered, target_manifest=target)

    reordered = copy.deepcopy(value)
    reordered["shards"].reverse()
    reordered.pop("manifest_sha256")
    reordered = seal_routed_input_manifest(reordered)
    with pytest.raises(RoutedInputRangeError, match="order"):
        validate_routed_input_manifest(reordered, target_manifest=target)
