from __future__ import annotations

import json
import hashlib
from dataclasses import replace

import numpy as np
import pytest
import moe_sae.artifacts as artifacts_module

from moe_sae.artifacts import (
    ArtifactDataset,
    ArtifactError,
    ArtifactWriter,
    CaptureBatch,
    CaptureMetadata,
    HIDDEN_SIZE,
    IntegrityError,
    LegacySchemaError,
    RangeConflictError,
    SCHEMA_VERSION,
    SchemaError,
    bf16_bits_to_float32,
    float32_to_bf16_bits,
    load_manifest,
    publish_manifest,
    verify_staged_dataset_manifest,
)
from moe_sae.model import CaptureBatch as ModelCaptureBatch


def metadata(split: str = "train") -> CaptureMetadata:
    return CaptureMetadata(
        checkpoint_sha256="1" * 64,
        maxtext_commit="48c7da3e28e8f57fe36ddf09d00eec53072a2ea0",
        dataset_fingerprint="1" * 64,
        split=split,
        config_sha256="2" * 64,
        environment_sha256="3" * 64,
        conversion_attestation_sha256="9" * 64,
    )


def batch(n: int, offset: float = 0.0) -> CaptureBatch:
    values = np.arange(n * HIDDEN_SIZE, dtype=np.float32).reshape(n, HIDDEN_SIZE)
    values = values / HIDDEN_SIZE + offset
    return CaptureBatch(
        activation=float32_to_bf16_bits(values),
        expert_ids=np.tile(np.arange(8, dtype=np.uint8), (n, 1)),
        router_weights=float32_to_bf16_bits(np.full((n, 8), 1 / 8, dtype=np.float32)),
        sequence_id=np.arange(100, 100 + n, dtype=np.int64),
        position=np.arange(n, dtype=np.uint16),
    )


def test_content_addressed_roundtrip_and_integrated_statistics(tmp_path):
    assert CaptureBatch is ModelCaptureBatch
    root = tmp_path / "capture"
    writer = ArtifactWriter(root, "worker-0", metadata())
    first = writer.write_range("c4-train-000", 0, 2, batch(2))
    second = writer.write_range("c4-train-001", 2, 5, batch(3, 2.0))
    writer.finalize_host()
    manifest_path = publish_manifest(root, metadata())

    assert manifest_path.name == "manifest.json"
    manifest = load_manifest(root)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["sealed"] is True
    assert manifest["total_tokens"] == 5
    assert manifest["shard_count"] == 2
    for entry in (first, second):
        path = root / entry["path"]
        assert path.parent.name == "host-worker-0"
        assert path.name == f"{entry['sha256']}.safetensors"

    dataset = ArtifactDataset(root)
    shards = list(dataset.iter_shards())
    assert [shard.range_id for shard in shards] == ["c4-train-000", "c4-train-001"]
    expected = np.concatenate(
        [
            bf16_bits_to_float32(batch(2).activation),
            bf16_bits_to_float32(batch(3, 2).activation),
        ]
    )
    assert np.array_equal(dataset.mean, expected.astype(np.float64).mean(axis=0))
    assert dataset.squared_norm == pytest.approx(
        np.square(expected, dtype=np.float64).sum(dtype=np.float64)
    )
    assert dataset.centered_sum_of_squares == pytest.approx(
        np.square(
            expected.astype(np.float64) - expected.astype(np.float64).mean(axis=0)
        ).sum()
    )
    train_mean = expected.astype(np.float64).mean(axis=0) + 0.25
    assert dataset.sst_about(train_mean) == pytest.approx(
        np.square(expected.astype(np.float64) - train_mean).sum()
    )
    assert all(shard.batch.activation.dtype == np.float32 for shard in shards)
    wire_shards = list(ArtifactDataset(root, decode_bf16=False).iter_shards())
    assert all(shard.batch.activation.dtype == np.uint16 for shard in wire_shards)


def test_multiple_hosts_publish_in_input_range_order(tmp_path):
    root = tmp_path / "capture"
    high = ArtifactWriter(root, "host-b", metadata())
    low = ArtifactWriter(root, "host-a", metadata())
    high.write_range("high", 1, 2, batch(1, 2))
    low.write_range("low", 0, 1, batch(1, 1))
    publish_manifest(root, metadata())
    manifest = load_manifest(root)
    assert [entry["range_id"] for entry in manifest["shards"]] == ["low", "high"]
    assert {entry["host_id"] for entry in manifest["shards"]} == {"host-a", "host-b"}


def test_token_range_partitions_concatenate_to_full_manifest_order(tmp_path):
    writer = ArtifactWriter(tmp_path, "host-0", metadata())
    writer.write_range("first", 0, 2, batch(2, 0.0))
    writer.write_range("second", 2, 5, batch(3, 10.0))
    writer.write_range("third", 5, 9, batch(4, 20.0))
    writer.finalize_host()
    publish_manifest(tmp_path, metadata(), expected_hosts=1, expected_tokens=9)
    dataset = ArtifactDataset(tmp_path)

    full = list(dataset.iter_batches())
    partitioned = [
        piece
        for start, end in ((0, 3), (3, 6), (6, 9))
        for piece in dataset.iter_token_range(start, end)
    ]
    for field in CaptureBatch._fields:
        expected = np.concatenate([np.asarray(getattr(item, field)) for item in full])
        actual = np.concatenate(
            [np.asarray(getattr(item, field)) for item in partitioned]
        )
        assert np.array_equal(actual, expected)
    assert list(dataset.iter_token_range(4, 4)) == []


def test_token_range_skips_irrelevant_shards(tmp_path, monkeypatch):
    import moe_sae.artifacts as artifacts

    writer = ArtifactWriter(tmp_path, "host-0", metadata())
    writer.write_range("first", 0, 2, batch(2, 0.0))
    writer.write_range("second", 2, 5, batch(3, 10.0))
    writer.write_range("third", 5, 9, batch(4, 20.0))
    writer.finalize_host()
    publish_manifest(tmp_path, metadata(), expected_hosts=1, expected_tokens=9)
    dataset = ArtifactDataset(tmp_path)

    called: list[str] = []
    real_load = artifacts._load_shard

    def recording_load(root, entry, verify_hash, **kwargs):
        called.append(entry["range_id"])
        return real_load(root, entry, verify_hash, **kwargs)

    monkeypatch.setattr(artifacts, "_load_shard", recording_load)
    pieces = list(dataset.iter_token_range(2, 5))
    assert called == ["second"]
    assert sum(piece.rows for piece in pieces) == 3

    for invalid in ((-1, 1), (2, 1), (0, 10), (True, 1)):
        with pytest.raises(ValueError):
            list(dataset.iter_token_range(*invalid))


def test_production_publication_requires_exact_completed_hosts_and_tokens(tmp_path):
    root = tmp_path / "capture"
    first = ArtifactWriter(root, "host-a", metadata())
    second = ArtifactWriter(root, "host-b", metadata())
    first.write_range("low", 0, 2, batch(2))
    second.write_range("high", 2, 5, batch(3))
    first.finalize_host()
    with pytest.raises(ArtifactError, match="1 unique completed hosts"):
        publish_manifest(root, metadata(), expected_hosts=2, expected_tokens=5)

    second.finalize_host()
    with pytest.raises(ArtifactError, match="captured 5 tokens, expected 6"):
        publish_manifest(root, metadata(), expected_hosts=2, expected_tokens=6)
    publish_manifest(root, metadata(), expected_hosts=2, expected_tokens=5)
    manifest = load_manifest(root)
    assert manifest["host_count"] == manifest["completed_host_count"] == 2


def test_finalized_host_rejects_new_ranges_but_exact_resume_remains_idempotent(
    tmp_path,
):
    writer = ArtifactWriter(tmp_path, "host-0", metadata())
    entry = writer.write_range("range-0", 0, 10, batch(1))
    writer.finalize_host()
    assert writer.write_range("range-0", 0, 10, batch(1, 99)) == entry
    with pytest.raises(ArtifactError, match="finalized"):
        writer.write_range("range-1", 10, 20, batch(1))


def test_resume_is_by_immutable_range_identity(tmp_path):
    writer = ArtifactWriter(tmp_path, "host-0", metadata())
    entry = writer.write_range("range-0", 10, 20, batch(2))
    assert writer.has_range("range-0", 10, 20)
    # Exact identity is a no-op even if a caller accidentally supplies another payload.
    assert writer.write_range("range-0", 10, 20, batch(1, 99)) == entry

    with pytest.raises(RangeConflictError, match="already bound"):
        writer.has_range("range-0", 10, 21)
    with pytest.raises(RangeConflictError, match="overlaps"):
        writer.write_range("another-id", 19, 30, batch(1))
    # Adjacent half-open ranges do not overlap.
    writer.write_range("range-1", 20, 30, batch(1))
    assert len(writer.finalize_host()["shards"]) == 2


def test_schema_shape_dtype_and_router_invariants_fail_closed(tmp_path):
    writer = ArtifactWriter(tmp_path, "host-0", metadata(), max_tokens=2)
    with pytest.raises(SchemaError, match="token count"):
        writer.write_range("large", 0, 1, batch(3))

    good = batch(2)
    with pytest.raises(SchemaError, match="activation must have shape"):
        writer.write_range(
            "bad-shape",
            0,
            1,
            CaptureBatch(
                good.activation[:, :-1],
                good.expert_ids,
                good.router_weights,
                good.sequence_id,
                good.position,
            ),
        )
    with pytest.raises(SchemaError, match=r"expert_ids must be in \[0,127\]"):
        writer.write_range(
            "bad-dtype",
            0,
            1,
            CaptureBatch(
                good.activation,
                good.expert_ids.astype(np.int16) + 128,
                good.router_weights,
                good.sequence_id,
                good.position,
            ),
        )
    with pytest.raises(SchemaError, match="unique"):
        writer.write_range(
            "duplicate-router",
            0,
            1,
            CaptureBatch(
                good.activation,
                np.zeros_like(good.expert_ids),
                good.router_weights,
                good.sequence_id,
                good.position,
            ),
        )
    with pytest.raises(SchemaError, match="sum to one"):
        writer.write_range(
            "bad-weights",
            0,
            1,
            CaptureBatch(
                good.activation,
                good.expert_ids,
                float32_to_bf16_bits(np.full((2, 8), 1 / 16, dtype=np.float32)),
                good.sequence_id,
                good.position,
            ),
        )


def test_float32_payload_is_explicitly_quantized_at_storage_boundary(tmp_path):
    writer = ArtifactWriter(tmp_path, "host-0", metadata())
    good = batch(1)
    writer.write_range(
        "float32",
        0,
        1,
        CaptureBatch(
            np.full((1, HIDDEN_SIZE), 1.125, dtype=np.float32),
            good.expert_ids.astype(np.int32),
            bf16_bits_to_float32(good.router_weights),
            good.sequence_id.astype(np.int32),
            good.position.astype(np.int32),
        ),
    )
    publish_manifest(tmp_path, metadata())
    stored = next(ArtifactDataset(tmp_path, decode_bf16=False).iter_batches())
    assert stored.activation.dtype == np.uint16
    decoded = next(ArtifactDataset(tmp_path).iter_batches())
    assert decoded.activation.dtype == np.float32
    assert np.all(decoded.activation == 1.125)
    assert decoded.expert_ids.dtype == np.uint8
    assert decoded.sequence_id.dtype == np.int64
    assert decoded.position.dtype == np.uint16


def test_integer_narrowing_rejects_overflow(tmp_path):
    writer = ArtifactWriter(tmp_path, "host-0", metadata())
    good = batch(1)
    with pytest.raises(SchemaError, match="position must be in"):
        writer.write_range(
            "position-overflow",
            0,
            1,
            CaptureBatch(
                good.activation,
                good.expert_ids,
                good.router_weights,
                good.sequence_id,
                np.array([70_000], dtype=np.int32),
            ),
        )


def test_publication_seals_store(tmp_path):
    writer = ArtifactWriter(tmp_path, "host-0", metadata())
    writer.write_range("range", 0, 1, batch(1))
    first = publish_manifest(tmp_path, metadata())
    assert publish_manifest(tmp_path, metadata()) == first  # idempotent publication
    with pytest.raises(ArtifactError, match="sealed"):
        ArtifactWriter(tmp_path, "host-1", metadata())
    with pytest.raises(ArtifactError, match="sealed"):
        writer.write_range("late", 1, 2, batch(1))


def test_legacy_and_missing_schema_are_rejected(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    # The old pipeline's top-level metadata did not carry this schema.
    (legacy / "manifest.json").write_text(json.dumps({"schema_version": "v0.1.2"}))
    with pytest.raises(LegacySchemaError):
        ArtifactDataset(legacy)

    no_manifest = tmp_path / "unpublished"
    writer = ArtifactWriter(no_manifest, "host-0", metadata())
    writer.write_range("range", 0, 1, batch(1))
    with pytest.raises(LegacySchemaError, match="no .* final manifest"):
        ArtifactDataset(no_manifest)


def test_hash_tampering_is_detected_on_read(tmp_path):
    writer = ArtifactWriter(tmp_path, "host-0", metadata())
    entry = writer.write_range("range", 0, 1, batch(1))
    publish_manifest(tmp_path, metadata())
    path = tmp_path / entry["path"]
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(IntegrityError, match="SHA-256"):
        list(ArtifactDataset(tmp_path).iter_shards())


def test_manifest_range_order_tampering_is_detected_on_open(tmp_path):
    writer = ArtifactWriter(tmp_path, "host-0", metadata())
    writer.write_range("first", 0, 1, batch(1))
    writer.write_range("second", 1, 2, batch(1, 2.0))
    publish_manifest(tmp_path, metadata())
    path = tmp_path / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["shards"].reverse()
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(IntegrityError, match="manifest SHA-256"):
        ArtifactDataset(tmp_path)


def test_metadata_mismatch_cannot_resume_or_publish(tmp_path):
    writer = ArtifactWriter(tmp_path, "host-0", metadata("train"))
    writer.write_range("range", 0, 1, batch(1))
    with pytest.raises(SchemaError, match="metadata"):
        ArtifactWriter(tmp_path, "host-0", metadata("validation"))
    with pytest.raises(SchemaError, match="metadata"):
        publish_manifest(tmp_path, metadata("validation"))


def test_provenance_hashes_are_required_lowercase_sha256(tmp_path):
    with pytest.raises(SchemaError, match="config_sha256"):
        ArtifactWriter(
            tmp_path / "bad-config", "host-0", replace(metadata(), config_sha256="x")
        )
    with pytest.raises(SchemaError, match="environment_sha256"):
        ArtifactWriter(
            tmp_path / "bad-environment",
            "host-0",
            replace(metadata(), environment_sha256="A" * 64),
        )


def _staging_manifest(tmp_path):
    first = tmp_path / "part-000.array_record"
    second = tmp_path / "part-001.array_record"
    first.write_bytes(b"first-packed-shard")
    second.write_bytes(b"second-packed-shard")
    entries = []
    for index, path in enumerate((first, second)):
        entries.append(
            {
                "uri": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
                "record_count": 10 + index,
                "scientific_tokens": 60 + index,
            }
        )
    manifest = tmp_path / "train.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "gemma4-c4-arrayrecord-v1",
                "dataset": {
                    "name": "c4",
                    "config": "en",
                    "version": "3.1.0",
                    "split": "train",
                },
                "tokenizer_id": "google/gemma-4-26B-A4B",
                "sequence_length": 1024,
                "packing": {
                    "strategy": "packed_fixed_length",
                    "exclude_from_scientific_count": ["padding", "bos", "eos"],
                    "special_token_ids": {"padding": 0, "eos": 1, "bos": 2},
                },
                "files": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest, first, second


def test_staged_dataset_fingerprint_is_derived_from_verified_inventory(tmp_path):
    manifest, first, second = _staging_manifest(tmp_path)
    result = verify_staged_dataset_manifest(
        str(manifest),
        str(tmp_path / "part-*.array_record"),
        expected_split="train",
        expected_tokens=121,
        tokenizer_id="google/gemma-4-26B-A4B",
        sequence_length=1024,
    )
    assert len(result.fingerprint_sha256) == 64
    assert result.files == (str(first.resolve()), str(second.resolve()))
    assert result.scientific_tokens == 121


def test_production_staging_check_recomputes_arrayrecord_count(tmp_path, monkeypatch):
    manifest, _, _ = _staging_manifest(tmp_path)

    class ArrayRecordDataSource:
        def __init__(self, files):
            assert len(files) == 2

        def __len__(self):
            return 20

    monkeypatch.setattr(
        artifacts_module,
        "import_module",
        lambda name: type("Grain", (), {"ArrayRecordDataSource": ArrayRecordDataSource}),
    )
    with pytest.raises(IntegrityError, match="count differs"):
        verify_staged_dataset_manifest(
            str(manifest),
            str(tmp_path / "part-*.array_record"),
            expected_split="train",
            expected_tokens=121,
            tokenizer_id="google/gemma-4-26B-A4B",
            sequence_length=1024,
            verify_arrayrecord_count=True,
        )


def test_staged_dataset_verification_rejects_tampering_and_inventory_drift(tmp_path):
    manifest, first, second = _staging_manifest(tmp_path)
    first.write_bytes(b"tampered")
    with pytest.raises(IntegrityError, match="hash/size mismatch"):
        verify_staged_dataset_manifest(
            str(manifest),
            str(tmp_path / "part-*.array_record"),
            expected_split="train",
            expected_tokens=121,
            tokenizer_id="google/gemma-4-26B-A4B",
            sequence_length=1024,
        )
    first.unlink()
    with pytest.raises(IntegrityError, match="inventory differs"):
        verify_staged_dataset_manifest(
            str(manifest),
            str(second),
            expected_split="train",
            expected_tokens=121,
            tokenizer_id="google/gemma-4-26B-A4B",
            sequence_length=1024,
            verify_file_hashes=False,
        )
