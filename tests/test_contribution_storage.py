from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors import safe_open

import moe_sae.contribution_storage as contribution_storage_module
from moe_sae.artifacts import (
    ArtifactDataset as AggregateDataset,
    ArtifactWriter as AggregateWriter,
    CaptureBatch as AggregateBatch,
    CaptureMetadata as AggregateMetadata,
    publish_manifest as publish_aggregate_manifest,
)
from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    LEGACY_AGGREGATE_SCHEMA_VERSION,
    MAXTEXT_BASE_COMMIT,
    TOP_K_EXPERTS,
    ContributionBatch,
    ContributionIntegrityError,
    ContributionLegacySchemaError,
    ContributionMetadata,
    ContributionRangeError,
    build_recomposition_sidecar,
    float32_to_bf16_bits,
)
from moe_sae.contribution_storage import (
    ContributionAdoptionResult,
    ContributionCapacityBatch,
    ContributionDataset,
    ContributionShardWriter,
    ContributionStorageError,
    adopt_contribution_phase,
    load_contribution_manifest,
    publish_contribution_manifest,
)
from moe_sae.object_store import (
    ImmutableObjectStore,
    LocalObjectStore,
    ObjectIdentity,
    ObjectStoreError,
    ShardCache,
    SourceFingerprint,
)


CHECKPOINT_SHA = "1" * 64
CONFIG_SHA = "2" * 64
ENVIRONMENT_SHA = "3" * 64
DATASET_SHA = "4" * 64
SOURCE_V1_SHA = "5" * 64
FORK_COMMIT = "6" * 40
OBJECT_PREFIX = "gs://test-bucket/experiment-2/run-001"


def contribution_batch(tokens: int, *, offset: int = 0) -> ContributionBatch:
    expert = np.arange(tokens * TOP_K_EXPERTS * HIDDEN_SIZE, dtype=np.float32).reshape(
        tokens, TOP_K_EXPERTS, HIDDEN_SIZE
    )
    expert = expert / float(TOP_K_EXPERTS * HIDDEN_SIZE) + float(offset)
    shared = np.arange(tokens * HIDDEN_SIZE, dtype=np.float32).reshape(
        tokens, HIDDEN_SIZE
    )
    shared = shared / float(HIDDEN_SIZE) - float(offset)
    weights = np.asarray(
        [
            0.25,
            0.1875,
            0.15625,
            0.125,
            0.09375,
            0.078125,
            0.0625,
            0.046875,
        ],
        dtype=np.float32,
    )
    return ContributionBatch(
        expert_output=float32_to_bf16_bits(expert),
        expert_ids=np.tile(np.arange(TOP_K_EXPERTS, dtype=np.uint8), (tokens, 1)),
        router_weights=float32_to_bf16_bits(np.tile(weights, (tokens, 1))),
        shared_output=float32_to_bf16_bits(shared),
        sequence_id=np.arange(100 + offset, 100 + offset + tokens, dtype=np.int64),
        position=np.arange(offset, offset + tokens, dtype=np.uint16),
    )


def recomposition_sidecar() -> dict[str, Any]:
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


def contribution_metadata(
    sidecar_sha256: str,
    source_v1_sha256: str = SOURCE_V1_SHA,
) -> dict[str, Any]:
    return ContributionMetadata(
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_fork_commit=FORK_COMMIT,
        dataset_fingerprint=DATASET_SHA,
        split="train",
        config_sha256=CONFIG_SHA,
        environment_sha256=ENVIRONMENT_SHA,
        source_v1_manifest_sha256=source_v1_sha256,
        recomposition_sidecar_sha256=sidecar_sha256,
    ).to_dict()


def sealed_source_v1(
    root: Path,
    values: tuple[ContributionBatch, ...],
) -> AggregateDataset:
    metadata = AggregateMetadata(
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_commit=MAXTEXT_BASE_COMMIT,
        dataset_fingerprint=DATASET_SHA,
        split="train",
        config_sha256="8" * 64,
        environment_sha256="9" * 64,
        conversion_attestation_sha256="a" * 64,
    )
    writer = AggregateWriter(root, "source-host", metadata)
    cursor = 0
    for index, value in enumerate(values):
        end = cursor + value.tokens
        writer.write_range(
            f"source-{index}",
            cursor,
            end,
            AggregateBatch(
                activation=value.shared_output,
                expert_ids=value.expert_ids,
                router_weights=value.router_weights,
                sequence_id=value.sequence_id,
                position=value.position,
            ),
        )
        cursor = end
    writer.finalize_host()
    publish_aggregate_manifest(
        root,
        metadata,
        expected_hosts=1,
        expected_tokens=cursor,
    )
    return AggregateDataset(root, decode_bf16=False)


def write_one(
    writer: ContributionShardWriter,
    value: ContributionBatch,
    *,
    start: int,
    source_start: int | None = None,
) -> dict[str, Any]:
    end = start + value.tokens
    selected_source_start = start if source_start is None else source_start
    source_end = selected_source_start + value.tokens
    range_id = writer.range_id(
        start,
        end,
        value.sequence_id,
        value.position,
        source_v1_range_start=selected_source_start,
        source_v1_range_end=source_end,
    )
    return writer.write_range(
        range_id,
        start,
        end,
        value,
        source_v1_range_start=selected_source_start,
        source_v1_range_end=source_end,
        normalized_shared_output=value.shared_output,
        source_sequence_id=value.sequence_id,
        source_position=value.position,
    )


def sealed_dataset(
    tmp_path: Path,
) -> tuple[
    ContributionDataset,
    LocalObjectStore,
    Path,
    tuple[ContributionBatch, ContributionBatch],
]:
    sidecar = recomposition_sidecar()
    control = tmp_path / "control"
    local = tmp_path / "local"
    objects = tmp_path / "objects"
    store = LocalObjectStore(objects)
    first = contribution_batch(2, offset=0)
    second = contribution_batch(3, offset=2)
    source = sealed_source_v1(tmp_path / "source-v1", (first, second))
    metadata = contribution_metadata(
        sidecar["sidecar_sha256"], source.manifest_sha256
    )
    first_writer = ContributionShardWriter(
        control,
        "worker-0",
        metadata,
        local_buffer_root=local / "worker-0",
        object_prefix=OBJECT_PREFIX,
        object_store=store,
    )
    second_writer = ContributionShardWriter(
        control,
        "worker-1",
        metadata,
        local_buffer_root=local / "worker-1",
        object_prefix=OBJECT_PREFIX,
        object_store=store,
    )
    write_one(first_writer, first, start=0)
    write_one(second_writer, second, start=2)
    first_writer.finalize_host()
    second_writer.finalize_host()
    publish_contribution_manifest(
        control,
        metadata,
        source_v1_dataset=source,
        object_prefix=OBJECT_PREFIX,
        object_store=store,
        expected_hosts=2,
        expected_tokens=5,
        recomposition_sidecar=sidecar,
    )
    cache = ShardCache(tmp_path / "cache", store, max_bytes=64 << 20)
    return (
        ContributionDataset(
            control,
            object_prefix=OBJECT_PREFIX,
            cache=cache,
            recomposition_sidecar=sidecar,
        ),
        store,
        control,
        (first, second),
    )


def test_round_trip_uses_content_addressed_objects_and_fp64_statistics(
    tmp_path: Path,
) -> None:
    dataset, _store, control, expected = sealed_dataset(tmp_path)

    shards = list(dataset.iter_shards())

    assert dataset.total_tokens == 5
    assert len(shards) == 2
    assert [shard.range_start for shard in shards] == [0, 2]
    assert [shard.range_end for shard in shards] == [2, 5]
    for shard, original in zip(shards, expected, strict=True):
        for field in original.__dataclass_fields__:
            assert np.array_equal(
                np.asarray(getattr(shard.batch, field)),
                np.asarray(getattr(original, field)),
            )
        assert shard.object.uri == (
            f"{OBJECT_PREFIX}/shards/host-{shard.host_id}/{shard.sha256}.safetensors"
        )
        cached = dataset.cache.root / f"{shard.sha256}.safetensors"
        with safe_open(cached, framework="numpy") as handle:
            assert handle.get_tensor("__stats_pooled_sum").dtype == np.float64
            assert handle.get_tensor("__stats_token_count").dtype == np.int64

    manifest = load_contribution_manifest(control)
    assert manifest["statistics"]["token_count"] == 5
    selected = list(dataset.iter_token_range(1, 4))
    assert [value.tokens for value in selected] == [1, 2]
    assert sum(value.tokens for value in selected) == 3
    assert not list((tmp_path / "local").rglob("*.safetensors"))


def test_shared_reader_does_not_load_expert_payload_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _store, _control, expected = sealed_dataset(tmp_path)

    def reject_full_load(_path):
        raise AssertionError("Q-only reads must not call safetensors.load_file")

    monkeypatch.setattr(contribution_storage_module, "load_file", reject_full_load)
    pieces = tuple(dataset.iter_shared_token_range(1, 4))

    assert [piece.tokens for piece in pieces] == [1, 2]
    np.testing.assert_array_equal(
        np.concatenate([piece.shared_output for piece in pieces]),
        np.concatenate(
            [expected[0].shared_output[1:], expected[1].shared_output[:2]]
        ),
    )
    np.testing.assert_array_equal(
        np.concatenate([piece.sequence_id for piece in pieces]),
        np.concatenate(
            [expected[0].sequence_id[1:], expected[1].sequence_id[:2]]
        ),
    )
    np.testing.assert_array_equal(
        np.concatenate([piece.position for piece in pieces]),
        np.concatenate([expected[0].position[1:], expected[1].position[:2]]),
    )


def test_training_range_reader_slices_payload_and_reuses_local_hash_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _store, _control, expected = sealed_dataset(tmp_path)

    def reject_full_load(_path):
        raise AssertionError("training ranges must not load whole shards")

    real_safe_open = contribution_storage_module.safe_open
    sliced_fields: list[str] = []

    @contextmanager
    def recording_safe_open(*args, **kwargs):
        with real_safe_open(*args, **kwargs) as handle:

            class RecordingHandle:
                def keys(self):
                    return handle.keys()

                def metadata(self):
                    return handle.metadata()

                def get_slice(self, name):
                    sliced_fields.append(name)
                    return handle.get_slice(name)

            yield RecordingHandle()

    monkeypatch.setattr(contribution_storage_module, "load_file", reject_full_load)
    monkeypatch.setattr(
        contribution_storage_module,
        "safe_open",
        recording_safe_open,
    )
    materializations = 0
    real_materialize = dataset.cache.materialize

    def counted_materialize(identity):
        nonlocal materializations
        materializations += 1
        return real_materialize(identity)

    monkeypatch.setattr(dataset.cache, "materialize", counted_materialize)
    first = tuple(dataset.iter_token_range(0, 1))
    second = tuple(dataset.iter_token_range(1, 2))

    assert materializations == 1
    assert sliced_fields == list(ContributionBatch.__dataclass_fields__) * 2
    for observed, start, end in ((first[0], 0, 1), (second[0], 1, 2)):
        for field in ContributionBatch.__dataclass_fields__:
            np.testing.assert_array_equal(
                getattr(observed, field),
                getattr(expected[0], field)[start:end],
            )


def test_capacity_reader_loads_only_route_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _store, _control, expected = sealed_dataset(tmp_path)

    def reject_full_load(_path):
        raise AssertionError("capacity reads must not call safetensors.load_file")

    real_safe_open = contribution_storage_module.safe_open
    sliced_fields: list[str] = []

    @contextmanager
    def recording_safe_open(*args, **kwargs):
        with real_safe_open(*args, **kwargs) as handle:

            class RecordingHandle:
                def keys(self):
                    return handle.keys()

                def metadata(self):
                    return handle.metadata()

                def get_slice(self, name):
                    sliced_fields.append(name)
                    return handle.get_slice(name)

            yield RecordingHandle()

    monkeypatch.setattr(contribution_storage_module, "load_file", reject_full_load)
    monkeypatch.setattr(
        contribution_storage_module,
        "safe_open",
        recording_safe_open,
    )

    a_pieces = tuple(
        dataset.iter_capacity_token_range(
            1,
            4,
            include_sequence_id=False,
        )
    )
    r_pieces = tuple(
        dataset.iter_capacity_token_range(
            1,
            4,
            include_sequence_id=True,
        )
    )

    assert all(isinstance(piece, ContributionCapacityBatch) for piece in a_pieces)
    assert [piece.tokens for piece in a_pieces] == [1, 2]
    assert all(piece.sequence_id is None for piece in a_pieces)
    np.testing.assert_array_equal(
        np.concatenate([piece.expert_ids for piece in a_pieces]),
        np.concatenate([expected[0].expert_ids[1:], expected[1].expert_ids[:2]]),
    )
    np.testing.assert_array_equal(
        np.concatenate(
            [piece.sequence_id for piece in r_pieces if piece.sequence_id is not None]
        ),
        np.concatenate([expected[0].sequence_id[1:], expected[1].sequence_id[:2]]),
    )
    assert sliced_fields.count("expert_ids") == 4
    assert sliced_fields.count("sequence_id") == 2
    assert not {
        "expert_output",
        "router_weights",
        "shared_output",
        "position",
    }.intersection(sliced_fields)


def test_publication_derives_source_identity_from_a_sealed_v1_root(
    tmp_path: Path,
) -> None:
    sidecar = recomposition_sidecar()
    value = contribution_batch(1)
    source = sealed_source_v1(tmp_path / "source-v1", (value,))
    forged_metadata = contribution_metadata(
        sidecar["sidecar_sha256"],
        SOURCE_V1_SHA,
    )
    control = tmp_path / "control"
    store = LocalObjectStore(tmp_path / "objects")
    writer = ContributionShardWriter(
        control,
        "worker-0",
        forged_metadata,
        local_buffer_root=tmp_path / "local",
        object_prefix=OBJECT_PREFIX,
        object_store=store,
    )
    write_one(writer, value, start=0)
    writer.finalize_host()

    with pytest.raises(
        ContributionStorageError,
        match="manifest SHA-256 differs",
    ):
        publish_contribution_manifest(
            control,
            forged_metadata,
            source_v1_dataset=source,
            object_prefix=OBJECT_PREFIX,
            object_store=store,
            expected_hosts=1,
            expected_tokens=1,
            recomposition_sidecar=sidecar,
        )


def test_publication_rejects_a_forged_in_memory_source_manifest(
    tmp_path: Path,
) -> None:
    sidecar = recomposition_sidecar()
    value = contribution_batch(1)
    source = sealed_source_v1(tmp_path / "source-v1", (value,))
    metadata = contribution_metadata(
        sidecar["sidecar_sha256"],
        source.manifest_sha256,
    )
    control = tmp_path / "control"
    store = LocalObjectStore(tmp_path / "objects")
    writer = ContributionShardWriter(
        control,
        "worker-0",
        metadata,
        local_buffer_root=tmp_path / "local",
        object_prefix=OBJECT_PREFIX,
        object_store=store,
    )
    write_one(writer, value, start=0)
    writer.finalize_host()
    source.manifest["manifest_sha256"] = "0" * 64

    with pytest.raises(
        ContributionStorageError,
        match="differs from its sealed root",
    ):
        publish_contribution_manifest(
            control,
            metadata,
            source_v1_dataset=source,
            object_prefix=OBJECT_PREFIX,
            object_store=store,
            expected_hosts=1,
            expected_tokens=1,
            recomposition_sidecar=sidecar,
        )


def test_logical_v2_range_can_bind_a_distinct_nonoverlapping_v1_slice(
    tmp_path: Path,
) -> None:
    sidecar = recomposition_sidecar()
    skipped = contribution_batch(2, offset=0)
    selected = contribution_batch(3, offset=2)
    source = sealed_source_v1(tmp_path / "source-v1", (skipped, selected))
    metadata = contribution_metadata(
        sidecar["sidecar_sha256"],
        source.manifest_sha256,
    )
    control = tmp_path / "control"
    store = LocalObjectStore(tmp_path / "objects")
    writer = ContributionShardWriter(
        control,
        "worker-0",
        metadata,
        local_buffer_root=tmp_path / "local",
        object_prefix=OBJECT_PREFIX,
        object_store=store,
    )
    entry = write_one(writer, selected, start=0, source_start=2)
    writer.finalize_host()

    publish_contribution_manifest(
        control,
        metadata,
        source_v1_dataset=source,
        object_prefix=OBJECT_PREFIX,
        object_store=store,
        expected_hosts=1,
        expected_tokens=3,
        recomposition_sidecar=sidecar,
    )

    assert entry["range_start"] == 0
    assert entry["range_end"] == 3
    assert entry["source_v1_range_start"] == 2
    assert entry["source_v1_range_end"] == 5
    assert load_contribution_manifest(control)["source_v1_manifest"][
        "total_tokens"
    ] == 5


def test_later_phase_adopts_sealed_receipts_without_reuploading_bulk_objects(
    tmp_path: Path,
) -> None:
    sidecar = recomposition_sidecar()
    first = contribution_batch(2, offset=0)
    second = contribution_batch(2, offset=2)
    source = sealed_source_v1(tmp_path / "source-v1", (first, second))
    metadata = contribution_metadata(
        sidecar["sidecar_sha256"],
        source.manifest_sha256,
    )
    objects = tmp_path / "objects"
    store = LocalObjectStore(objects)

    phase_1_root = tmp_path / "phase-1-control"
    phase_1_writer = ContributionShardWriter(
        phase_1_root,
        "worker-0",
        metadata,
        local_buffer_root=tmp_path / "phase-1-local",
        object_prefix=OBJECT_PREFIX,
        object_store=store,
    )
    first_entry = write_one(phase_1_writer, first, start=0, source_start=0)
    phase_1_writer.finalize_host()
    publish_contribution_manifest(
        phase_1_root,
        metadata,
        source_v1_dataset=source,
        object_prefix=OBJECT_PREFIX,
        object_store=store,
        expected_hosts=1,
        expected_tokens=2,
        recomposition_sidecar=sidecar,
    )
    phase_1_manifest = load_contribution_manifest(
        phase_1_root,
        recomposition_sidecar=sidecar,
    )
    assert len(list(objects.rglob("*.safetensors"))) == 1

    phase_2_root = tmp_path / "phase-2-control"
    phase_2_writer = ContributionShardWriter(
        phase_2_root,
        "worker-0",
        metadata,
        local_buffer_root=tmp_path / "phase-2-local",
        object_prefix=OBJECT_PREFIX,
        object_store=store,
    )
    adoption = adopt_contribution_phase(
        phase_2_writer,
        phase_1_root,
        recomposition_sidecar=sidecar,
    )

    assert isinstance(adoption, ContributionAdoptionResult)
    assert adoption.prior_manifest_sha256 == phase_1_manifest["manifest_sha256"]
    assert adoption.host_id == "worker-0"
    assert adoption.adopted_shards == 1
    assert adoption.adopted_tokens == 2
    assert phase_2_writer.has_range(
        first_entry["range_id"],
        first_entry["range_start"],
        first_entry["range_end"],
    )
    assert len(list(objects.rglob("*.safetensors"))) == 1

    second_entry = write_one(phase_2_writer, second, start=2, source_start=2)
    phase_2_writer.finalize_host()
    publish_contribution_manifest(
        phase_2_root,
        metadata,
        source_v1_dataset=source,
        object_prefix=OBJECT_PREFIX,
        object_store=store,
        expected_hosts=1,
        expected_tokens=4,
        recomposition_sidecar=sidecar,
    )
    phase_2_manifest = load_contribution_manifest(
        phase_2_root,
        recomposition_sidecar=sidecar,
    )

    assert phase_2_manifest["shards"] == [first_entry, second_entry]
    assert phase_2_manifest["total_tokens"] == 4
    assert len(list(objects.rglob("*.safetensors"))) == 2


class PublishThenFailStore:
    """Simulate loss of the worker after GCS accepts an immutable object."""

    def __init__(self, inner: ImmutableObjectStore) -> None:
        self.inner = inner
        self.fail_once = True
        self.put_calls = 0

    def put_if_absent(
        self,
        source: Path,
        uri: str,
        *,
        sha256: str,
        source_fingerprint: SourceFingerprint | None = None,
    ) -> ObjectIdentity:
        self.put_calls += 1
        identity = self.inner.put_if_absent(
            source,
            uri,
            sha256=sha256,
            source_fingerprint=source_fingerprint,
        )
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("simulated process loss after immutable upload")
        return identity

    def download(self, identity: ObjectIdentity, destination: Path) -> None:
        self.inner.download(identity, destination)

    def verify(self, identity: ObjectIdentity) -> None:
        self.inner.verify(identity)

    def delete(self, identity: ObjectIdentity) -> None:
        self.inner.delete(identity)


def test_upload_completed_before_receipt_is_adopted_on_resume(
    tmp_path: Path,
) -> None:
    sidecar = recomposition_sidecar()
    metadata = contribution_metadata(sidecar["sidecar_sha256"])
    control = tmp_path / "control"
    local = tmp_path / "local"
    objects = tmp_path / "objects"
    inner = LocalObjectStore(objects)
    store = PublishThenFailStore(inner)
    writer = ContributionShardWriter(
        control,
        "worker-0",
        metadata,
        local_buffer_root=local,
        object_prefix=OBJECT_PREFIX,
        object_store=store,
    )
    value = contribution_batch(2)
    range_id = writer.range_id(
        0,
        value.tokens,
        value.sequence_id,
        value.position,
        source_v1_range_start=0,
        source_v1_range_end=value.tokens,
    )

    with pytest.raises(RuntimeError, match="after immutable upload"):
        write_one(writer, value, start=0)

    assert len(list(objects.rglob("*.safetensors"))) == 1
    assert len(list(local.rglob("*.safetensors"))) == 1
    assert len(list(local.rglob("*.json"))) == 1
    assert writer.has_range(range_id, 0, value.tokens)
    assert writer.has_range(range_id, 0, value.tokens)
    assert store.put_calls == 2
    assert len(list(objects.rglob("*.safetensors"))) == 1
    assert not list(local.rglob("*.safetensors"))
    assert not list(local.rglob("*.json"))
    assert len(list((control / "receipts").rglob("*.json"))) == 1


def test_resume_is_source_bound_and_enforces_the_writer_shard_limit(
    tmp_path: Path,
) -> None:
    sidecar = recomposition_sidecar()
    metadata = contribution_metadata(sidecar["sidecar_sha256"])
    writer = ContributionShardWriter(
        tmp_path / "control",
        "worker-0",
        metadata,
        local_buffer_root=tmp_path / "local",
        object_prefix=OBJECT_PREFIX,
        object_store=LocalObjectStore(tmp_path / "objects"),
        max_tokens=1,
    )
    value = contribution_batch(2)

    with pytest.raises(ContributionRangeError, match="at most 1"):
        writer.range_id(
            0,
            2,
            value.sequence_id,
            value.position,
            source_v1_range_start=0,
            source_v1_range_end=2,
        )

    one = contribution_batch(1)
    range_id = writer.range_id(
        0,
        1,
        one.sequence_id,
        one.position,
        source_v1_range_start=0,
        source_v1_range_end=1,
    )
    wrong_source_position = one.position.copy()
    wrong_source_position[0] += 1
    with pytest.raises(ContributionIntegrityError, match="token-key sequence"):
        writer.write_range(
            range_id,
            0,
            1,
            one,
            source_v1_range_start=0,
            source_v1_range_end=1,
            normalized_shared_output=one.shared_output,
            source_sequence_id=one.sequence_id,
            source_position=wrong_source_position,
        )
    changed_sequence = one.sequence_id.copy()
    changed_sequence[0] += 1
    assert (
        writer.range_id(
            0,
            1,
            changed_sequence,
            one.position,
            source_v1_range_start=0,
            source_v1_range_end=1,
        )
        != range_id
    )


def test_coordinator_rejects_wrong_source_keys_and_changed_object(
    tmp_path: Path,
) -> None:
    sidecar = recomposition_sidecar()
    value = contribution_batch(2)
    wrong_source_value = ContributionBatch(
        expert_output=value.expert_output,
        expert_ids=value.expert_ids,
        router_weights=value.router_weights,
        shared_output=value.shared_output,
        sequence_id=value.sequence_id,
        position=value.position[::-1].copy(),
    )
    source = sealed_source_v1(tmp_path / "wrong-source-v1", (wrong_source_value,))
    metadata = contribution_metadata(
        sidecar["sidecar_sha256"], source.manifest_sha256
    )
    control = tmp_path / "control"
    objects = tmp_path / "objects"
    store = LocalObjectStore(objects)
    writer = ContributionShardWriter(
        control,
        "worker-0",
        metadata,
        local_buffer_root=tmp_path / "local",
        object_prefix=OBJECT_PREFIX,
        object_store=store,
    )
    write_one(writer, value, start=0)
    writer.finalize_host()

    with pytest.raises(ContributionIntegrityError, match="token-key sequence"):
        publish_contribution_manifest(
            control,
            metadata,
            source_v1_dataset=source,
            object_prefix=OBJECT_PREFIX,
            object_store=store,
            expected_hosts=1,
            expected_tokens=2,
            recomposition_sidecar=sidecar,
        )

    object_path = next(objects.rglob("*.safetensors"))
    with object_path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ObjectStoreError, match="changed"):
        publish_contribution_manifest(
            control,
            metadata,
            source_v1_dataset=source,
            object_prefix=OBJECT_PREFIX,
            object_store=store,
            expected_hosts=1,
            expected_tokens=2,
            recomposition_sidecar=sidecar,
        )


def test_reader_repairs_corrupt_cache_and_rejects_tampered_receipt(
    tmp_path: Path,
) -> None:
    dataset, _store, control, expected = sealed_dataset(tmp_path)
    first = next(dataset.iter_shards())
    cached = dataset.cache.root / f"{first.sha256}.safetensors"
    cached.write_bytes(b"corrupt-cache")

    recovered = next(dataset.iter_shards())

    assert np.array_equal(recovered.batch.expert_output, expected[0].expert_output)
    receipt_path = next((control / "receipts").rglob("*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    receipt["object"]["size_bytes"] += 1
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ContributionStorageError, match="self-hash mismatch"):
        next(dataset.iter_shards())


def test_aggregate_v1_manifest_is_never_auto_detected_as_v2(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": LEGACY_AGGREGATE_SCHEMA_VERSION,
                "manifest_sha256": SOURCE_V1_SHA,
            }
        ),
        encoding="ascii",
    )

    with pytest.raises(ContributionLegacySchemaError, match="aggregate"):
        load_contribution_manifest(root)
