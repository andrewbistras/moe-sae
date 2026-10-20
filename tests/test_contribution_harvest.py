from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from moe_sae.artifacts import (
    ArtifactDataset,
    ArtifactWriter,
    CaptureBatch,
    CaptureMetadata,
    publish_manifest,
)
from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    TOP_K_EXPERTS,
    ContributionMetadata,
    build_recomposition_sidecar,
    float32_to_bf16_bits,
)
from moe_sae.contribution_harvest import (
    ContributionHarvestError,
    contribution_capture_to_wire_chunks,
    harvest_contribution_host,
)
from moe_sae.contribution_ranges import build_contribution_range_plan
from moe_sae.contribution_recomposition import SUPPORTED_ACCUMULATION_ORDER
from moe_sae.contribution_storage import (
    ContributionDataset,
    ContributionShardWriter,
    load_contribution_manifest,
    publish_contribution_manifest,
)
from moe_sae.harvest import make_sequence_ids
from moe_sae.object_store import LocalObjectStore, ShardCache
from moe_sae.runtime import scientific_token_mask

CHECKPOINT_SHA = "1" * 64
DATASET_SHA = "2" * 64
CONFIG_SHA = "3" * 64
ENVIRONMENT_SHA = "4" * 64
FORK_COMMIT = "5" * 40
BASE_COMMIT = "48c7da3e28e8f57fe36ddf09d00eec53072a2ea0"
OBJECT_PREFIX = "gs://test-bucket/gemma4-v2/train"
SEQUENCE_LENGTH = 1_024
WEIGHTS = np.asarray(
    [0.25, 0.1875, 0.15625, 0.125, 0.09375, 0.078125, 0.0625, 0.046875],
    dtype=np.float32,
)


def test_wire_conversion_bitcasts_ml_dtypes_bfloat16_and_chunks() -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    rows = 3
    expert = np.zeros(
        (rows, TOP_K_EXPERTS, HIDDEN_SIZE),
        dtype=ml_dtypes.bfloat16,
    )
    shared = np.zeros(
        (rows, HIDDEN_SIZE),
        dtype=ml_dtypes.bfloat16,
    )
    weights = np.broadcast_to(
        WEIGHTS.astype(ml_dtypes.bfloat16),
        (rows, TOP_K_EXPERTS),
    ).copy()
    chunks = contribution_capture_to_wire_chunks(
        expert_output=expert,
        expert_ids=np.broadcast_to(
            np.arange(TOP_K_EXPERTS, dtype=np.uint8),
            (rows, TOP_K_EXPERTS),
        ),
        router_weights=weights,
        shared_output=shared,
        sequence_id=np.arange(rows, dtype=np.int64),
        position=np.arange(rows, dtype=np.uint16),
        max_tokens=2,
    )

    assert [chunk.tokens for chunk in chunks] == [2, 1]
    assert all(chunk.expert_output.dtype == np.uint16 for chunk in chunks)
    assert np.array_equal(
        chunks[0].router_weights,
        weights[:2].view(np.uint16),
    )


def raw_batch(scientific_tokens: int) -> dict[str, np.ndarray]:
    tokens = np.zeros((1, SEQUENCE_LENGTH), dtype=np.int32)
    tokens[0, :scientific_tokens] = np.arange(
        3,
        3 + scientific_tokens,
        dtype=np.int32,
    )
    positions = np.broadcast_to(
        np.arange(SEQUENCE_LENGTH, dtype=np.int32),
        tokens.shape,
    ).copy()
    segments = np.zeros_like(tokens)
    segments[0, :scientific_tokens] = 1
    return {
        "inputs": tokens,
        "inputs_position": positions,
        "inputs_segmentation": segments,
    }


def source_capture(raw: dict[str, np.ndarray], process: int, batch_index: int):
    mask = scientific_token_mask(
        raw["inputs"],
        raw["inputs_segmentation"],
        excluded_token_ids=(0, 1, 2),
    )
    indices = np.flatnonzero(mask.reshape(-1))
    rows = int(indices.size)
    sequence = make_sequence_ids(
        raw["inputs_segmentation"],
        process_index=process,
        batch_index=batch_index,
    ).reshape(-1)[indices]
    position = raw["inputs_position"].reshape(-1)[indices]
    return CaptureBatch(
        activation=float32_to_bf16_bits(
            np.zeros((rows, HIDDEN_SIZE), dtype=np.float32)
        ),
        expert_ids=np.broadcast_to(
            np.arange(TOP_K_EXPERTS, dtype=np.uint8),
            (rows, TOP_K_EXPERTS),
        ).copy(),
        router_weights=float32_to_bf16_bits(
            np.broadcast_to(WEIGHTS, (rows, TOP_K_EXPERTS))
        ),
        sequence_id=sequence.astype(np.int64),
        position=position.astype(np.uint16),
    )


def source_metadata() -> CaptureMetadata:
    return CaptureMetadata(
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_commit=BASE_COMMIT,
        dataset_fingerprint=DATASET_SHA,
        split="train",
        config_sha256="6" * 64,
        environment_sha256="7" * 64,
        conversion_attestation_sha256="8" * 64,
    )


def build_source(
    root: Path,
    streams: tuple[tuple[dict[str, np.ndarray], ...], ...],
) -> ArtifactDataset:
    metadata = source_metadata()
    host_totals = [
        sum(
            source_capture(raw, process, index).activation.shape[0]
            for index, raw in enumerate(stream)
        )
        for process, stream in enumerate(streams)
    ]
    assert len(set(host_totals)) == 1
    lane_tokens = host_totals[0]
    for process, stream in enumerate(streams):
        writer = ArtifactWriter(root, f"p{process:05d}", metadata)
        cursor = process * lane_tokens
        for batch_index, raw in enumerate(stream):
            value = source_capture(raw, process, batch_index)
            end = cursor + value.activation.shape[0]
            writer.write_range(
                f"source-p{process}-b{batch_index}",
                cursor,
                end,
                value,
            )
            cursor = end
        writer.finalize_host()
    publish_manifest(
        root,
        metadata,
        expected_hosts=len(streams),
        expected_tokens=sum(host_totals),
    )
    return ArtifactDataset(root, decode_bf16=False)


def sidecar() -> dict:
    scale = float32_to_bf16_bits(np.ones(HIDDEN_SIZE, dtype=np.float32))
    return build_recomposition_sidecar(
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_fork_commit=FORK_COMMIT,
        source_fingerprints={
            "checkpoint": CHECKPOINT_SHA,
            "config": CONFIG_SHA,
            "environment": ENVIRONMENT_SHA,
            "fork_tree": "9" * 64,
        },
        routed_rmsnorm_scale=scale,
        shared_rmsnorm_scale=scale,
        outer_ffn_rmsnorm_scale=scale,
        layer_scalar=float32_to_bf16_bits(np.asarray([1.0], dtype=np.float32)),
        accumulation_order=SUPPORTED_ACCUMULATION_ORDER,
    )


def contribution_metadata(source: ArtifactDataset, value: dict) -> dict:
    return ContributionMetadata(
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_fork_commit=FORK_COMMIT,
        dataset_fingerprint=DATASET_SHA,
        split="train",
        config_sha256=CONFIG_SHA,
        environment_sha256=ENVIRONMENT_SHA,
        source_v1_manifest_sha256=source.manifest_sha256,
        recomposition_sidecar_sha256=value["sidecar_sha256"],
    ).to_dict()


class FakeAdapter:
    def __init__(self, *, fail_call: int | None = None):
        self.calls = 0
        self.fail_call = fail_call
        base = np.arange(
            TOP_K_EXPERTS * HIDDEN_SIZE,
            dtype=np.float32,
        ).reshape(1, TOP_K_EXPERTS, HIDDEN_SIZE)
        self.expert = base / np.float32(TOP_K_EXPERTS * HIDDEN_SIZE)
        self.shared = np.zeros((1, HIDDEN_SIZE), dtype=np.float32)

    def __call__(self, tokens, positions, segments, sequence_ids):
        del positions, segments, sequence_ids
        self.calls += 1
        if self.fail_call == self.calls:
            raise RuntimeError("simulated adapter interruption")
        rows = int(np.prod(tokens.shape))
        return SimpleNamespace(
            expert_output=np.broadcast_to(
                self.expert,
                (rows, TOP_K_EXPERTS, HIDDEN_SIZE),
            ),
            expert_ids=np.broadcast_to(
                np.arange(TOP_K_EXPERTS, dtype=np.uint8),
                (rows, TOP_K_EXPERTS),
            ),
            router_weights=np.broadcast_to(
                WEIGHTS,
                (rows, TOP_K_EXPERTS),
            ),
            shared_output=np.broadcast_to(
                self.shared,
                (rows, HIDDEN_SIZE),
            ),
        )


def writer_for(
    control: Path,
    local: Path,
    objects: Path,
    host: str,
    metadata: dict,
    *,
    shard_tokens: int,
) -> tuple[ContributionShardWriter, LocalObjectStore]:
    store = LocalObjectStore(objects)
    return (
        ContributionShardWriter(
            control,
            host,
            metadata,
            local_buffer_root=local,
            object_prefix=OBJECT_PREFIX,
            object_store=store,
            max_tokens=shard_tokens,
        ),
        store,
    )


def test_single_host_capture_seals_exact_v2_and_shared_statistics(tmp_path):
    raw = raw_batch(4)
    source = build_source(tmp_path / "source", ((raw,),))
    recomposition = sidecar()
    metadata = contribution_metadata(source, recomposition)
    plan = build_contribution_range_plan(
        source.manifest,
        (4,),
        process_count=1,
        max_shard_tokens=2,
    )
    writer, store = writer_for(
        tmp_path / "control",
        tmp_path / "local",
        tmp_path / "objects",
        "p00000",
        metadata,
        shard_tokens=2,
    )

    result = harvest_contribution_host(
        (raw,),
        FakeAdapter(),
        writer=writer,
        source_v1_dataset=source,
        range_plan=plan,
        phase_total_tokens=4,
        process_count=1,
        process_index=0,
        recomposition_sidecar=recomposition,
    )
    publish_contribution_manifest(
        tmp_path / "control",
        metadata,
        source_v1_dataset=source,
        object_prefix=OBJECT_PREFIX,
        object_store=store,
        expected_hosts=1,
        expected_tokens=4,
        recomposition_sidecar=recomposition,
    )
    dataset = ContributionDataset(
        tmp_path / "control",
        object_prefix=OBJECT_PREFIX,
        cache=ShardCache(tmp_path / "cache", store, max_bytes=64 << 20),
        recomposition_sidecar=recomposition,
    )

    assert result.new_tokens == 4
    assert result.resumed_tokens == 0
    assert result.max_writer_backlog_batches == 1
    assert result.batches_executed == 1
    assert dataset.total_tokens == 4
    assert len(list(dataset.iter_shards())) == 2
    assert (
        load_contribution_manifest(tmp_path / "control")["statistics"][
            "shared_normalized"
        ]["count"]
        == 4
    )


def test_interrupted_host_resumes_by_immutable_range_without_reexecution(tmp_path):
    first = raw_batch(4)
    second = raw_batch(4)
    source = build_source(tmp_path / "source", ((first, second),))
    recomposition = sidecar()
    metadata = contribution_metadata(source, recomposition)
    plan = build_contribution_range_plan(
        source.manifest,
        (8,),
        process_count=1,
        max_shard_tokens=4,
    )
    control = tmp_path / "control"
    local = tmp_path / "local"
    objects = tmp_path / "objects"
    writer, _store = writer_for(
        control,
        local,
        objects,
        "p00000",
        metadata,
        shard_tokens=4,
    )

    with pytest.raises(RuntimeError, match="simulated adapter interruption"):
        harvest_contribution_host(
            (first, second),
            FakeAdapter(fail_call=2),
            writer=writer,
            source_v1_dataset=source,
            range_plan=plan,
            phase_total_tokens=8,
            process_count=1,
            process_index=0,
            recomposition_sidecar=recomposition,
        )
    durable_before = tuple(objects.rglob("*.safetensors"))
    assert len(durable_before) == 1

    resumed_writer, _ = writer_for(
        control,
        local,
        objects,
        "p00000",
        metadata,
        shard_tokens=4,
    )
    adapter = FakeAdapter()
    result = harvest_contribution_host(
        (first, second),
        adapter,
        writer=resumed_writer,
        source_v1_dataset=source,
        range_plan=plan,
        phase_total_tokens=8,
        process_count=1,
        process_index=0,
        recomposition_sidecar=recomposition,
    )

    assert result.resumed_tokens == 4
    assert result.new_tokens == 4
    assert result.batches_executed == 1
    assert adapter.calls == 1
    assert len(tuple(objects.rglob("*.safetensors"))) == 2


def test_exact_source_alignment_survives_larger_static_rebatching(tmp_path):
    """V1 batch/row ID bits must not pin Experiment 2 to the old batch size."""

    first = raw_batch(4)
    second = raw_batch(4)
    source = build_source(tmp_path / "source", ((first, second),))
    combined = {
        name: np.concatenate((first[name], second[name]), axis=0) for name in first
    }
    recomposition = sidecar()
    metadata = contribution_metadata(source, recomposition)
    plan = build_contribution_range_plan(
        source.manifest,
        (8,),
        process_count=1,
        max_shard_tokens=4,
    )
    control = tmp_path / "control"
    writer, store = writer_for(
        control,
        tmp_path / "local",
        tmp_path / "objects",
        "p00000",
        metadata,
        shard_tokens=4,
    )

    harvest_contribution_host(
        (combined,),
        FakeAdapter(),
        writer=writer,
        source_v1_dataset=source,
        range_plan=plan,
        phase_total_tokens=8,
        process_count=1,
        process_index=0,
        recomposition_sidecar=recomposition,
    )
    publish_contribution_manifest(
        control,
        metadata,
        source_v1_dataset=source,
        object_prefix=OBJECT_PREFIX,
        object_store=store,
        expected_hosts=1,
        expected_tokens=8,
        recomposition_sidecar=recomposition,
    )
    dataset = ContributionDataset(
        control,
        object_prefix=OBJECT_PREFIX,
        cache=ShardCache(tmp_path / "cache", store, max_bytes=64 << 20),
        recomposition_sidecar=recomposition,
    )
    source_keys = tuple(source.iter_token_keys_range(0, 8))
    expected_sequence = np.concatenate([item[0] for item in source_keys])
    expected_position = np.concatenate([item[1] for item in source_keys])
    captured = tuple(dataset.iter_shards())
    actual_sequence = np.concatenate([shard.batch.sequence_id for shard in captured])
    actual_position = np.concatenate([shard.batch.position for shard in captured])

    np.testing.assert_array_equal(actual_sequence, expected_sequence)
    np.testing.assert_array_equal(actual_position, expected_position)
    # The second source row was v1 batch 1/row 0, not the live batch 0/row 1.
    assert int(actual_sequence[4]) != int(
        make_sequence_ids(
            combined["inputs_segmentation"],
            process_index=0,
            batch_index=0,
        )[1, 0]
    )


def test_two_hosts_interleave_logical_ranges_but_bind_native_source_lanes(tmp_path):
    host0 = raw_batch(4)
    host1 = raw_batch(4)
    source = build_source(tmp_path / "source", ((host0,), (host1,)))
    recomposition = sidecar()
    metadata = contribution_metadata(source, recomposition)
    plan = build_contribution_range_plan(
        source.manifest,
        (8,),
        process_count=2,
        max_shard_tokens=2,
    )
    control = tmp_path / "control"
    objects = tmp_path / "objects"
    store = LocalObjectStore(objects)
    for process, raw in enumerate((host0, host1)):
        writer = ContributionShardWriter(
            control,
            f"p{process:05d}",
            metadata,
            local_buffer_root=tmp_path / f"local-{process}",
            object_prefix=OBJECT_PREFIX,
            object_store=store,
            max_tokens=2,
        )
        harvest_contribution_host(
            (raw,),
            FakeAdapter(),
            writer=writer,
            source_v1_dataset=source,
            range_plan=plan,
            phase_total_tokens=8,
            process_count=2,
            process_index=process,
            recomposition_sidecar=recomposition,
        )
    publish_contribution_manifest(
        control,
        metadata,
        source_v1_dataset=source,
        object_prefix=OBJECT_PREFIX,
        object_store=store,
        expected_hosts=2,
        expected_tokens=8,
        recomposition_sidecar=recomposition,
    )

    entries = load_contribution_manifest(control)["shards"]
    assert [entry["range_start"] for entry in entries] == [0, 2, 4, 6]
    assert [entry["host_id"] for entry in entries] == [
        "p00000",
        "p00001",
        "p00000",
        "p00001",
    ]
    assert [
        (entry["source_v1_range_start"], entry["source_v1_range_end"])
        for entry in entries
    ] == [(0, 2), (4, 6), (2, 4), (6, 8)]


def test_staged_input_identity_mismatch_fails_before_persistence(tmp_path):
    raw = raw_batch(4)
    source = build_source(tmp_path / "source", ((raw,),))
    recomposition = sidecar()
    metadata = contribution_metadata(source, recomposition)
    plan = build_contribution_range_plan(
        source.manifest,
        (4,),
        process_count=1,
        max_shard_tokens=4,
    )
    changed = {name: value.copy() for name, value in raw.items()}
    changed["inputs_segmentation"][0, :4] = 2
    writer, _store = writer_for(
        tmp_path / "control",
        tmp_path / "local",
        tmp_path / "objects",
        "p00000",
        metadata,
        shard_tokens=4,
    )

    with pytest.raises(
        ContributionHarvestError,
        match="token keys differ",
    ):
        harvest_contribution_host(
            (changed,),
            FakeAdapter(),
            writer=writer,
            source_v1_dataset=source,
            range_plan=plan,
            phase_total_tokens=4,
            process_count=1,
            process_index=0,
            recomposition_sidecar=recomposition,
        )
    assert not tuple((tmp_path / "objects").rglob("*.safetensors"))
