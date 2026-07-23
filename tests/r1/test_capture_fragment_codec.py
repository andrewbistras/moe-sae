from __future__ import annotations

from dataclasses import replace
import hashlib

import ml_dtypes
import numpy as np
import pytest

from moe_sae.r1.capture_data_plane_contract import (
    PINNED_CAPTURE_DATA_PLANE_CONTRACT,
    PINNED_CAPTURE_ROW_BYTES,
    build_capture_work_plan,
)
from moe_sae.r1.capture_fragment_codec import (
    AddressableCaptureShardV1,
    CaptureFragmentCodecError,
    PackedCaptureFragmentV1,
    decode_capture_fragment,
    pack_capture_fragment,
)


_BF16 = np.dtype(ml_dtypes.bfloat16)


@pytest.fixture(scope="module")
def addressable_shard() -> AddressableCaptureShardV1:
    activation = np.zeros((2, 1_024, 7_168), dtype=_BF16)
    activation.reshape(-1, 7_168)[:3, 0] = np.asarray(
        [1.0, 2.0, 3.0],
        dtype=_BF16,
    )
    descending_ids = np.arange(7, -1, -1, dtype=np.int32)
    expert_ids = np.broadcast_to(descending_ids, (2, 1_024, 8)).copy()
    ascending_weights = (
        np.arange(1, 9, dtype=np.float32) / np.float32(36.0) * np.float32(2.5)
    )
    router_weights = np.broadcast_to(
        ascending_weights[::-1],
        (2, 1_024, 8),
    ).copy()
    sequence_id = np.broadcast_to(
        np.asarray([101, 102], dtype=np.int64)[:, None],
        (2, 1_024),
    ).copy()
    position = np.broadcast_to(
        np.arange(1_024, dtype=np.uint16),
        (2, 1_024),
    ).copy()
    return AddressableCaptureShardV1(
        routed_activation=activation,
        global_expert_ids=expert_ids,
        scaled_router_weights=router_weights,
        sequence_id=sequence_id,
        position=position,
    )


@pytest.fixture(scope="module")
def packed_full(
    addressable_shard: AddressableCaptureShardV1,
) -> PackedCaptureFragmentV1:
    work = build_capture_work_plan(
        work_id="codec-full-fragment",
        input_range_sha256="a" * 64,
        start_token_ordinal=9,
        token_count=2_048,
    )
    plan = next(work.iter_fragments())
    return pack_capture_fragment(plan, addressable_shard)


def test_full_fragment_codec_matches_exact_layout_and_hashes(
    packed_full: PackedCaptureFragmentV1,
) -> None:
    assert len(packed_full.payload) == 2_048 * PINNED_CAPTURE_ROW_BYTES
    assert packed_full.sha256 == hashlib.sha256(packed_full.payload).hexdigest()
    assert packed_full.object_id.endswith(f"/{packed_full.sha256}.bin")
    assert packed_full.field_byte_ranges == (
        PINNED_CAPTURE_DATA_PLANE_CONTRACT.field_byte_ranges(2_048)
    )
    assert packed_full.persisted is False
    assert packed_full.readback_verified is False
    assert packed_full.production_sealable is False


def test_codec_canonicalizes_router_pairs_and_preserves_alignment(
    packed_full: PackedCaptureFragmentV1,
) -> None:
    decoded = decode_capture_fragment(
        packed_full.plan,
        packed_full.payload,
        expected_sha256=packed_full.sha256,
    )
    ids = decoded.fields["global_expert_ids"]
    weights = decoded.fields["scaled_router_weights"].astype(np.float32)
    assert np.array_equal(ids[0], np.arange(8, dtype=np.uint16))
    expected = (
        (np.arange(1, 9, dtype=np.float32) / np.float32(36.0) * np.float32(2.5))
        .astype(_BF16)
        .astype(np.float32)
    )
    np.testing.assert_array_equal(weights[0], expected)
    assert decoded.fields["sequence_id"][0] == 101
    assert decoded.fields["sequence_id"][-1] == 102
    assert decoded.fields["position"][0] == 0
    assert decoded.fields["position"][1_023] == 1_023
    assert decoded.fields["position"][1_024] == 0
    assert all(not value.flags.writeable for value in decoded.fields.values())


def test_partial_fragment_compacts_only_the_retained_prefix(
    addressable_shard: AddressableCaptureShardV1,
) -> None:
    work = build_capture_work_plan(
        work_id="codec-tail-fragment",
        input_range_sha256="b" * 64,
        start_token_ordinal=1_000,
        token_count=3,
    )
    plan = next(work.iter_fragments())
    packed = pack_capture_fragment(plan, addressable_shard)
    assert len(packed.payload) == 3 * PINNED_CAPTURE_ROW_BYTES
    decoded = decode_capture_fragment(
        plan,
        packed.payload,
        expected_sha256=packed.sha256,
    )
    actual = decoded.fields["routed_activation"][:, 0].astype(np.float32)
    np.testing.assert_array_equal(actual, np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(
        decoded.fields["position"],
        np.arange(3, dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        decoded.fields["sequence_id"],
        np.full(3, 101, dtype=np.int64),
    )


def test_invalid_router_or_metadata_never_produces_payload(
    addressable_shard: AddressableCaptureShardV1,
) -> None:
    work = build_capture_work_plan(
        work_id="codec-invalid-fragment",
        input_range_sha256="c" * 64,
        start_token_ordinal=0,
        token_count=3,
    )
    plan = next(work.iter_fragments())
    duplicate_ids = addressable_shard.global_expert_ids.copy()
    duplicate_ids.reshape(-1, 8)[0] = 1
    bad_router = replace(addressable_shard, global_expert_ids=duplicate_ids)
    with pytest.raises(CaptureFragmentCodecError, match="router contract"):
        pack_capture_fragment(plan, bad_router)

    bad_positions = addressable_shard.position.copy()
    bad_positions.reshape(-1)[1] = 7
    bad_metadata = replace(addressable_shard, position=bad_positions)
    with pytest.raises(CaptureFragmentCodecError, match="positions"):
        pack_capture_fragment(plan, bad_metadata)


def test_wrong_host_dtype_and_public_packed_construction_fail_closed(
    addressable_shard: AddressableCaptureShardV1,
    packed_full: PackedCaptureFragmentV1,
) -> None:
    with pytest.raises(CaptureFragmentCodecError, match="dtype"):
        replace(
            addressable_shard,
            global_expert_ids=addressable_shard.global_expert_ids.astype(np.int64),
        )
    with pytest.raises(CaptureFragmentCodecError, match="only through"):
        PackedCaptureFragmentV1(
            packed_full.plan,
            packed_full.payload,
        )


def test_decode_rejects_size_hash_and_durable_pair_corruption(
    packed_full: PackedCaptureFragmentV1,
) -> None:
    class _EqualString(str):
        def __eq__(self, other: object) -> bool:
            return True

    with pytest.raises(CaptureFragmentCodecError, match="exact SHA-256"):
        decode_capture_fragment(
            packed_full.plan,
            packed_full.payload,
            expected_sha256=_EqualString(packed_full.sha256),
        )
    with pytest.raises(CaptureFragmentCodecError, match="size"):
        decode_capture_fragment(
            packed_full.plan,
            packed_full.payload[:-1],
            expected_sha256=hashlib.sha256(packed_full.payload[:-1]).hexdigest(),
        )
    corrupted = bytearray(packed_full.payload)
    corrupted[0] ^= 1
    with pytest.raises(CaptureFragmentCodecError, match="SHA-256"):
        decode_capture_fragment(
            packed_full.plan,
            bytes(corrupted),
            expected_sha256=packed_full.sha256,
        )

    ids_range = next(
        item for item in packed_full.field_byte_ranges if item[0] == "global_expert_ids"
    )
    corrupted = bytearray(packed_full.payload)
    corrupted[ids_range[1] : ids_range[1] + 4] = b"\x01\x00\x01\x00"
    payload = bytes(corrupted)
    with pytest.raises(CaptureFragmentCodecError, match="unique ascending"):
        decode_capture_fragment(
            packed_full.plan,
            payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
