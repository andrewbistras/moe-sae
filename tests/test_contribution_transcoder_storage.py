from __future__ import annotations

import copy

import numpy as np
import pytest
from safetensors import safe_open

from moe_sae.contribution_artifacts import HIDDEN_SIZE, float32_to_bf16_bits
from moe_sae.contribution_transcoder_artifacts import (
    RoutedInputBatch,
    RoutedInputMetadata,
    RoutedInputRangeError,
)
from moe_sae.contribution_transcoder_storage import (
    RoutedInputDataset,
    RoutedInputShardWriter,
    load_routed_input_manifest,
    publish_routed_input_manifest,
)
from moe_sae.object_store import LocalObjectStore, ShardCache
from tests.test_contribution_artifacts import manifest as contribution_manifest


OBJECT_PREFIX = "gs://test-bucket/experiment-3/overlay"


def metadata(target: dict) -> RoutedInputMetadata:
    parent = target["metadata"]
    return RoutedInputMetadata(
        checkpoint_sha256=parent["checkpoint_sha256"],
        maxtext_fork_commit=parent["maxtext_fork_commit"],
        dataset_fingerprint=parent["dataset_fingerprint"],
        split=parent["split"],
        config_sha256="a" * 64,
        environment_sha256="b" * 64,
        target_manifest_sha256=target["manifest_sha256"],
    )


def batch_for_target(target_entry: dict, *, offset: float) -> RoutedInputBatch:
    tokens = int(target_entry["tokens"])
    values = np.arange(tokens * HIDDEN_SIZE, dtype=np.float32).reshape(
        tokens, HIDDEN_SIZE
    )
    values = values / HIDDEN_SIZE + offset
    # The compact fixture's immutable target key sequence is deterministic.
    return RoutedInputBatch(
        routed_input=float32_to_bf16_bits(values),
        sequence_id=np.arange(100, 100 + tokens, dtype=np.int64),
        position=np.arange(tokens, dtype=np.uint16),
    )


def test_content_addressed_round_trip_is_exactly_parent_bound(tmp_path):
    target, _sidecar = contribution_manifest()
    store = LocalObjectStore(tmp_path / "objects")
    control = tmp_path / "control"
    writer = RoutedInputShardWriter(
        control,
        "worker-0",
        metadata(target),
        target_manifest=target,
        local_buffer_root=tmp_path / "local",
        object_prefix=OBJECT_PREFIX,
        object_store=store,
    )
    expected = []
    for index, target_entry in enumerate(target["shards"]):
        batch = batch_for_target(target_entry, offset=float(index))
        expected.append(batch)
        entry = writer.write_range(target_entry, batch)
        assert entry["target_range_id"] == target_entry["range_id"]
        assert writer.has_target_range(target_entry)
    writer.finalize_host()
    publish_routed_input_manifest(
        control,
        metadata(target),
        target_manifest=target,
        object_prefix=OBJECT_PREFIX,
        object_store=store,
        expected_hosts=1,
    )

    cache = ShardCache(tmp_path / "cache", store, max_bytes=64 << 20)
    dataset = RoutedInputDataset(
        control,
        object_prefix=OBJECT_PREFIX,
        cache=cache,
        target_manifest=target,
    )
    shards = list(dataset.iter_shards())

    assert dataset.total_tokens == target["total_tokens"] == 5
    assert len(shards) == 2
    assert len(dataset.token_range_object_identities(1, 4)) == 2
    assert dataset.prefetch_token_range(0, 5)["objects"] == 2
    assert dataset.require_resident_token_range(0, 5)["objects"] == 2
    for shard, original, parent in zip(
        shards, expected, target["shards"], strict=True
    ):
        np.testing.assert_array_equal(shard.batch.routed_input, original.routed_input)
        np.testing.assert_array_equal(shard.batch.sequence_id, original.sequence_id)
        np.testing.assert_array_equal(shard.batch.position, original.position)
        assert shard.target_range_id == parent["range_id"]
        cached = cache.root / f"{shard.sha256}.safetensors"
        with safe_open(cached, framework="numpy") as handle:
            assert set(handle.keys()) == {
                "routed_input",
                "sequence_id",
                "position",
                "__stats_count",
                "__stats_sum",
                "__stats_squared_norm",
            }
            assert handle.get_tensor("__stats_sum").dtype == np.float64

    pieces = list(dataset.iter_token_range(1, 4))
    assert [piece.tokens for piece in pieces] == [1, 2]
    assert sum(piece.tokens for piece in pieces) == 3
    assert not list((tmp_path / "local").rglob("*.safetensors"))


def test_writer_rejects_equal_length_but_wrong_token_alignment(tmp_path):
    target, _sidecar = contribution_manifest()
    writer = RoutedInputShardWriter(
        tmp_path / "control",
        "worker-0",
        metadata(target),
        target_manifest=target,
        local_buffer_root=tmp_path / "local",
        object_prefix=OBJECT_PREFIX,
        object_store=LocalObjectStore(tmp_path / "objects"),
    )
    target_entry = target["shards"][0]
    original = batch_for_target(target_entry, offset=0.0)
    wrong = RoutedInputBatch(
        routed_input=original.routed_input,
        sequence_id=original.sequence_id + 1_000,
        position=original.position,
    )
    with pytest.raises(RoutedInputRangeError, match="token-key sequence"):
        writer.write_range(target_entry, wrong)

    forged_target = copy.deepcopy(target_entry)
    forged_target["range_start"] += 1
    with pytest.raises(RoutedInputRangeError, match="exact shard"):
        writer.write_range(forged_target, original)


def test_manifest_load_requires_the_same_parent_manifest(tmp_path):
    target, _sidecar = contribution_manifest()
    store = LocalObjectStore(tmp_path / "objects")
    writer = RoutedInputShardWriter(
        tmp_path / "control",
        "worker-0",
        metadata(target),
        target_manifest=target,
        local_buffer_root=tmp_path / "local",
        object_prefix=OBJECT_PREFIX,
        object_store=store,
    )
    for index, target_entry in enumerate(target["shards"]):
        writer.write_range(
            target_entry, batch_for_target(target_entry, offset=float(index))
        )
    writer.finalize_host()
    publish_routed_input_manifest(
        tmp_path / "control",
        metadata(target),
        target_manifest=target,
        object_prefix=OBJECT_PREFIX,
        object_store=store,
        expected_hosts=1,
    )

    wrong_parent = copy.deepcopy(target)
    wrong_parent["metadata"]["dataset_fingerprint"] = "c" * 64
    # A changed parent cannot retain a valid canonical manifest hash.
    with pytest.raises(Exception):
        load_routed_input_manifest(
            tmp_path / "control", target_manifest=wrong_parent
        )
