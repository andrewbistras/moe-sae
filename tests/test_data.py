import json
from pathlib import Path
import struct

import numpy as np
import pytest

from moe_sae.data import (
    BOS_ID,
    EOS_ID,
    PACKED_DATASET_SCHEMA,
    PACKED_RECORD_FORMAT,
    OrderedPacker,
    PackedBatchIterable,
    StagingError,
    decode_packed_record,
    encode_packed_record,
    publish_staged_dataset,
    stage_c4_split,
    verify_packed_source,
)
from moe_sae.object_store import LocalObjectStore, parse_gcs_uri


TOKENIZER_ID = "google/gemma-4-26B-A4B"
TOKENIZER_REVISION = "a" * 40
SOURCE_INFO_SHA256 = "b" * 64


def _valid_arrays(length: int = 10):
    token_ids = np.asarray(
        [BOS_ID, 10, 11, EOS_ID, BOS_ID, 12, EOS_ID, 0, 0, 0],
        dtype=np.int32,
    )
    positions = np.asarray([0, 1, 2, 3, 0, 1, 2, 0, 0, 0], dtype=np.uint16)
    segment_ids = np.asarray([1, 1, 1, 1, 2, 2, 2, 0, 0, 0], dtype=np.uint16)
    assert length == 10
    return token_ids, positions, segment_ids


def _require_arrayrecord() -> None:
    pytest.importorskip("array_record")
    pytest.importorskip("grain")


def _manifest(result) -> dict:
    return json.loads(Path(result.manifest_location).read_text(encoding="utf-8"))


def _files(result) -> list[str]:
    manifest_path = Path(result.manifest_location)
    return [
        str(manifest_path.parent / entry["uri"])
        for entry in _manifest(result)["files"]
    ]


def _tokenize_full_record(text: str, *, sequence_length: int = 16) -> list[int]:
    return [10 + int(text)] * (sequence_length - 2)


def _stage(
    output: Path,
    *,
    target: int,
    document_count: int = 32,
    tokenize=_tokenize_full_record,
    records_per_shard: int = 4,
):
    _require_arrayrecord()
    return stage_c4_split(
        split="train",
        target_scientific_tokens=target,
        output_directory=output,
        tfds_data_dir=None,
        tokenizer_id=TOKENIZER_ID,
        tokenizer_revision=TOKENIZER_REVISION,
        records_per_shard=records_per_shard,
        sequence_length=16,
        texts=[str(index) for index in range(document_count)],
        tokenize=tokenize,
        source_info_sha256=SOURCE_INFO_SHA256,
    )


def test_packed_record_round_trip_preserves_exact_schema():
    arrays = _valid_arrays()
    payload = encode_packed_record(7, *arrays, sequence_length=10)
    record = decode_packed_record(
        payload, expected_index=7, expected_sequence_length=10
    )

    assert record.record_index == 7
    assert record.scientific_tokens == 3
    assert record.token_ids.dtype == np.dtype("int32")
    assert record.positions.dtype == np.dtype("uint16")
    assert record.segment_ids.dtype == np.dtype("uint16")
    for actual, expected in zip(
        (record.token_ids, record.positions, record.segment_ids), arrays, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("array_index", "replacement", "message"),
    [
        (2, np.asarray([3, 3, 3, 3, 2, 2, 2, 0, 0, 0]), "one-indexed"),
        (2, np.asarray([2, 2, 2, 2, 1, 1, 1, 0, 0, 0]), "increasing order"),
        (2, np.asarray([1, 1, 1, 1, 0, 2, 2, 0, 0, 0]), "padding"),
        (1, np.asarray([0, 1, 2, 3, 0, 2, 1, 0, 0, 0]), "restart"),
        (2, np.asarray([1, 1, 1, 1, 2, 2, 2, 0, 3, 0]), "record tail"),
    ],
)
def test_encoder_rejects_noncanonical_segment_layout(
    array_index: int, replacement: np.ndarray, message: str
):
    arrays = list(_valid_arrays())
    arrays[array_index] = replacement
    with pytest.raises(StagingError, match=message):
        encode_packed_record(0, *arrays, sequence_length=10)


def test_decoder_rejects_header_count_payload_and_index_corruption():
    payload = encode_packed_record(7, *_valid_arrays(), sequence_length=10)

    with pytest.raises(StagingError, match="byte length"):
        decode_packed_record(payload[:-1], expected_sequence_length=10)

    bad_version = bytearray(payload)
    struct.pack_into("<H", bad_version, 8, 2)
    with pytest.raises(StagingError, match="header"):
        decode_packed_record(bytes(bad_version), expected_sequence_length=10)

    bad_count = bytearray(payload)
    struct.pack_into("<I", bad_count, 12, 4)
    with pytest.raises(StagingError, match="scientific-token count"):
        decode_packed_record(bytes(bad_count), expected_sequence_length=10)

    bad_padding = bytearray(payload)
    # Header plus the final int32 token ID, whose segment remains padding.
    struct.pack_into("<i", bad_padding, 20 + 9 * 4, 99)
    with pytest.raises(StagingError, match="padding"):
        decode_packed_record(bytes(bad_padding), expected_sequence_length=10)

    with pytest.raises(StagingError, match="index changed"):
        decode_packed_record(payload, expected_index=8, expected_sequence_length=10)


def test_ordered_packer_chunks_long_documents_without_dropping_tokens():
    packer = OrderedPacker(sequence_length=8)
    content = list(range(10, 20))
    emitted = packer.add_content(content)
    emitted.extend(packer.flush())
    assert len(emitted) == 2

    decoded = [
        decode_packed_record(
            encode_packed_record(index, *arrays, sequence_length=8),
            expected_index=index,
            expected_sequence_length=8,
        )
        for index, arrays in enumerate(emitted)
    ]
    recovered = []
    for record in decoded:
        active = record.segment_ids != 0
        recovered.extend(record.token_ids[active][1:-1].tolist())
    assert recovered == content
    assert [record.scientific_tokens for record in decoded] == [6, 4]


def test_interrupted_staging_resumes_at_a_sealed_document_boundary(tmp_path):
    interrupted = tmp_path / "interrupted"
    uninterrupted = tmp_path / "uninterrupted"

    def fail_after_first_shard(text: str) -> list[int]:
        if text == "1":
            raise RuntimeError("simulated tokenizer interruption")
        return _tokenize_full_record(text)

    with pytest.raises(RuntimeError, match="simulated"):
        _stage(
            interrupted,
            target=40,
            document_count=8,
            tokenize=fail_after_first_shard,
            records_per_shard=1,
        )
    assert (interrupted / ".train.staging.json").is_file()

    resumed = _stage(
        interrupted, target=40, document_count=8, records_per_shard=1
    )
    reference = _stage(
        uninterrupted, target=40, document_count=8, records_per_shard=1
    )

    assert resumed.resumed is True
    assert Path(resumed.manifest_location).read_bytes() == Path(
        reference.manifest_location
    ).read_bytes()
    assert [Path(path).read_bytes() for path in _files(resumed)] == [
        Path(path).read_bytes() for path in _files(reference)
    ]


def test_stager_meets_every_lane_quota_and_pads_all_batch_candidates(tmp_path):
    result = _stage(tmp_path, target=220)
    manifest = _manifest(result)
    records, scientific, lanes = verify_packed_source(
        _files(result), sequence_length=16
    )

    assert records == result.record_count == manifest["record_count"] == 64
    assert records % (4 * 4 * 4) == 0
    assert scientific == result.scientific_tokens == 224
    assert lanes == tuple(manifest["lane_scientific_tokens"]) == (56, 56, 56, 56)
    assert all(count >= 55 for count in lanes)


def test_manifest_binds_exact_c4_and_revision_pinned_tokenizer(tmp_path):
    result = _stage(tmp_path, target=40, records_per_shard=1)
    manifest = _manifest(result)

    assert manifest["schema_version"] == PACKED_DATASET_SCHEMA
    assert manifest["dataset"] == {
        "name": "c4",
        "config": "en",
        "version": "3.1.0",
        "split": "train",
    }
    assert manifest["source"] == {
        "provider": "tensorflow_datasets",
        "full_name": "c4/en/3.1.0",
        "dataset_info_sha256": SOURCE_INFO_SHA256,
    }
    assert manifest["tokenizer"] == {
        "id": TOKENIZER_ID,
        "revision": TOKENIZER_REVISION,
        "padding_id": 0,
        "eos_id": 1,
        "bos_id": 2,
    }
    assert manifest["record_format"] == {
        "name": PACKED_RECORD_FORMAT,
        "arrayrecord_options": "group_size:1",
        "token_ids": "int32[16]",
        "positions": "uint16[16]",
        "segment_ids": "uint16[16]",
    }

    with pytest.raises(StagingError, match="another input"):
        stage_c4_split(
            split="train",
            target_scientific_tokens=40,
            output_directory=tmp_path,
            tfds_data_dir=None,
            tokenizer_id=TOKENIZER_ID,
            tokenizer_revision="c" * 40,
            records_per_shard=1,
            sequence_length=16,
            texts=[str(index) for index in range(8)],
            tokenize=_tokenize_full_record,
            source_info_sha256=SOURCE_INFO_SHA256,
        )


def test_sealed_stage_rejects_aggregate_or_shard_corruption(tmp_path):
    result = _stage(tmp_path, target=40, records_per_shard=1)
    manifest_path = Path(result.manifest_location)
    manifest = _manifest(result)
    manifest["lane_scientific_tokens"][0] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(StagingError, match="aggregate counters"):
        _stage(tmp_path, target=40, records_per_shard=1)

    manifest["lane_scientific_tokens"][0] -= 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    shard = Path(_files(result)[0])
    payload = bytearray(shard.read_bytes())
    payload[len(payload) // 2] ^= 1
    shard.write_bytes(payload)
    with pytest.raises(StagingError, match="shard changed"):
        _stage(tmp_path, target=40, records_per_shard=1)


def test_fixed_host_reader_order_is_independent_of_candidate_batch_size(tmp_path):
    result = _stage(tmp_path, target=220)
    files = _files(result)
    leading_rows = []
    for global_batch_size in (16, 32, 64):
        batch = next(
            iter(
                PackedBatchIterable(
                    files,
                    global_batch_size=global_batch_size,
                    process_index=2,
                    process_count=4,
                    sequence_length=16,
                )
            )
        )
        assert batch.keys() == {
            "inputs",
            "inputs_position",
            "inputs_segmentation",
        }
        leading_rows.append(batch["inputs"][:4])
    for rows in leading_rows[1:]:
        np.testing.assert_array_equal(rows, leading_rows[0])
    assert leading_rows[0][:, 1].tolist() == [12, 16, 20, 24]

    with pytest.raises(ValueError, match="qualified per-device"):
        PackedBatchIterable(
            files,
            global_batch_size=8,
            process_index=0,
            process_count=4,
            sequence_length=16,
        )


def test_staged_publication_is_conditional_content_addressed_and_manifest_last(
    tmp_path,
):
    result = _stage(tmp_path / "stage", target=40, records_per_shard=1)
    object_root = tmp_path / "objects"
    store = LocalObjectStore(object_root)
    published = publish_staged_dataset(
        result.manifest_location,
        "gs://staging-test/c4/train",
        object_store=store,
    )

    bucket, manifest_name = parse_gcs_uri(published.manifest_location)
    remote_manifest = json.loads(
        (object_root / bucket / manifest_name).read_text(encoding="utf-8")
    )
    uris = [entry["uri"] for entry in remote_manifest["files"]]
    assert uris == sorted(uris)
    assert all(entry["sha256"] in entry["uri"] for entry in remote_manifest["files"])
    assert all(
        entry["object"]["uri"] == entry["uri"]
        and entry["object"]["sha256"] == entry["sha256"]
        and entry["object"]["size_bytes"] == entry["size_bytes"]
        and entry["object"]["generation"] > 0
        for entry in remote_manifest["files"]
    )
    assert published.grain_files == "gs://staging-test/c4/train/shards/*.array_record"
    # Idempotent replay validates existing generations rather than overwriting.
    replay = publish_staged_dataset(
        result.manifest_location,
        "gs://staging-test/c4/train",
        object_store=store,
    )
    assert replay.manifest_location == published.manifest_location
