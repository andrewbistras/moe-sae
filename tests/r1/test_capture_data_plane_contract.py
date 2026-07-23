from __future__ import annotations

from dataclasses import replace
import random

import pytest

from moe_sae.r1.capture_data_plane_contract import (
    CaptureDataPlaneContractError,
    PINNED_CAPTURE_DATA_PLANE_CONTRACT,
    PINNED_CAPTURE_DATA_PLANE_CONTRACT_SHA256,
    PINNED_CAPTURE_ROW_BYTES,
    PINNED_FRAGMENT_TOKEN_CAPACITY,
    PINNED_FULL_FRAGMENT_BYTES,
    PINNED_FULL_HOST_STEP_BYTES,
    PINNED_FULL_STEP_BYTES,
    PINNED_FRAGMENTS_PER_STEP,
    PINNED_PLANNING_TOKENS_PER_SECOND,
    build_capture_work_plan,
    fragment_content_object_id,
)
from moe_sae.r1.gcs_immutable_byte_store import MAX_IMMUTABLE_SNAPSHOT_BYTES
from moe_sae.r1.prefix_batchsplit_contract import (
    PINNED_GLOBAL_BATCH_SIZE,
    PINNED_PREFIX_BATCHSPLIT_CONTRACT_SHA256,
)


def test_contract_freezes_exact_byte_and_throughput_arithmetic() -> None:
    contract = PINNED_CAPTURE_DATA_PLANE_CONTRACT
    assert contract.digest == PINNED_CAPTURE_DATA_PLANE_CONTRACT_SHA256
    assert contract.batchsplit_contract_sha256 == (
        PINNED_PREFIX_BATCHSPLIT_CONTRACT_SHA256
    )
    assert contract.row_bytes == PINNED_CAPTURE_ROW_BYTES == 14_378
    assert contract.full_fragment_bytes == PINNED_FULL_FRAGMENT_BYTES == 29_446_144
    assert contract.full_host_step_bytes == (PINNED_FULL_HOST_STEP_BYTES) == 117_784_576
    assert contract.full_step_bytes == PINNED_FULL_STEP_BYTES == 1_884_553_216
    assert contract.full_fragment_bytes < MAX_IMMUTABLE_SNAPSHOT_BYTES
    assert (
        contract.planning_tokens_per_second
        == (PINNED_PLANNING_TOKENS_PER_SECOND)
        == 801_000
    )
    assert contract.planning_bytes_per_second == 11_516_778_000
    assert contract.planning_bytes_per_second_per_host == 719_798_625
    assert (
        contract.planning_step_duration_ns_numerator
        / contract.planning_step_duration_ns_denominator
        == pytest.approx(163_635_455.6803995)
    )
    assert contract.writer_slot_count == 2
    assert contract.maximum_queued_complete_steps == 1
    assert contract.upload_streams_per_host == 4
    assert contract.router_pair_order == "ascending-global-expert-id"
    assert contract.synchronized_path_may_wait_for_writer is False
    assert contract.create_ack_is_durable_qualification is False
    assert contract.production_sealable is False
    assert contract.production_admission_enabled is False


def test_columnar_payload_intervals_are_exact_and_contiguous() -> None:
    contract = PINNED_CAPTURE_DATA_PLANE_CONTRACT
    ranges = contract.field_byte_ranges(PINNED_FRAGMENT_TOKEN_CAPACITY)
    assert tuple(item[0] for item in ranges) == (
        "routed_activation",
        "global_expert_ids",
        "scaled_router_weights",
        "sequence_id",
        "position",
    )
    assert ranges[0][1] == 0
    assert all(left[2] == right[1] for left, right in zip(ranges, ranges[1:]))
    assert ranges[-1][2] == PINNED_FULL_FRAGMENT_BYTES
    with pytest.raises(CaptureDataPlaneContractError):
        contract.field_byte_ranges(0)
    with pytest.raises(CaptureDataPlaneContractError):
        contract.field_byte_ranges(PINNED_FRAGMENT_TOKEN_CAPACITY + 1)


@pytest.mark.parametrize(
    ("token_count", "step_count", "fragment_count", "tail_tokens"),
    (
        (50_000_000, 382, 24_415, 128),
        (2_000_000, 16, 977, 1_152),
        (131_072, 1, 64, 2_048),
        (1, 1, 1, 1),
    ),
)
def test_work_plans_are_compact_exact_and_tail_aware(
    token_count: int,
    step_count: int,
    fragment_count: int,
    tail_tokens: int,
) -> None:
    plan = build_capture_work_plan(
        work_id="train-range-0001",
        input_range_sha256="a" * 64,
        start_token_ordinal=17,
        token_count=token_count,
    )
    assert plan.step_count == step_count
    assert plan.fragment_count == fragment_count
    assert plan.payload_bytes == token_count * PINNED_CAPTURE_ROW_BYTES
    fragments = list(plan.iter_fragments())
    assert len(fragments) == fragment_count
    assert fragments[0].token_ordinal_start == 17
    assert fragments[-1].valid_tokens == tail_tokens
    assert fragments[-1].payload_bytes == tail_tokens * PINNED_CAPTURE_ROW_BYTES
    assert sum(item.valid_tokens for item in fragments) == token_count
    assert sum(item.payload_bytes for item in fragments) == plan.payload_bytes
    assert all(
        left.token_ordinal_start + left.valid_tokens == right.token_ordinal_start
        for left, right in zip(fragments, fragments[1:])
    )


def test_addressable_shard_mapping_is_independent_of_observed_device_order() -> None:
    plan = build_capture_work_plan(
        work_id="topology-neutral-range",
        input_range_sha256="b" * 64,
        start_token_ordinal=0,
        token_count=131_072,
    )
    slices = [(start, start + 2) for start in range(0, PINNED_GLOBAL_BATCH_SIZE, 2)]
    expected = [
        plan.fragment_for_global_batch_slice(0, start, stop) for start, stop in slices
    ]
    shuffled = list(slices)
    random.Random(7).shuffle(shuffled)
    observed = [
        plan.fragment_for_global_batch_slice(0, start, stop) for start, stop in shuffled
    ]
    assert {item.digest for item in expected if item is not None} == {
        item.digest for item in observed if item is not None
    }
    assert len(expected) == PINNED_FRAGMENTS_PER_STEP
    assert all(item is not None for item in expected)
    assert len({item.digest for item in expected if item is not None}) == 64


def test_padding_only_last_step_slices_are_not_published() -> None:
    plan = build_capture_work_plan(
        work_id="partial-final-step",
        input_range_sha256="c" * 64,
        start_token_ordinal=0,
        token_count=50_000_000,
    )
    final_fragments = [
        plan.fragment_for_global_batch_slice(
            plan.step_count - 1,
            start,
            start + 2,
        )
        for start in range(0, PINNED_GLOBAL_BATCH_SIZE, 2)
    ]
    retained = [item for item in final_fragments if item is not None]
    assert len(retained) == 31
    assert retained[-1].valid_tokens == 128
    assert all(item is None for item in final_fragments[31:])


def test_fragment_identity_rejects_unaligned_or_wider_device_slices() -> None:
    plan = build_capture_work_plan(
        work_id="slice-validation",
        input_range_sha256="d" * 64,
        start_token_ordinal=0,
        token_count=131_072,
    )
    with pytest.raises(CaptureDataPlaneContractError, match="two-sequence"):
        plan.fragment_for_global_batch_slice(0, 1, 3)
    with pytest.raises(CaptureDataPlaneContractError, match="two-sequence"):
        plan.fragment_for_global_batch_slice(0, 0, 4)
    with pytest.raises(CaptureDataPlaneContractError):
        plan.fragment_for_global_batch_slice(1, 0, 2)


def test_fragment_record_cross_checks_its_complete_work_interval() -> None:
    plan = build_capture_work_plan(
        work_id="fragment-cross-check",
        input_range_sha256="9" * 64,
        start_token_ordinal=41,
        token_count=2_049,
    )
    fragment = list(plan.iter_fragments())[-1]
    assert fragment.valid_tokens == 1
    with pytest.raises(CaptureDataPlaneContractError, match="exact work"):
        replace(fragment, token_ordinal_start=fragment.token_ordinal_start + 1)
    with pytest.raises(CaptureDataPlaneContractError, match="exact work"):
        replace(fragment, valid_tokens=2, payload_bytes=2 * PINNED_CAPTURE_ROW_BYTES)
    with pytest.raises(CaptureDataPlaneContractError, match="padding-only"):
        replace(
            fragment,
            step_index=1,
            global_batch_start=0,
            global_batch_stop=2,
        )


def test_contract_cannot_claim_qualification_or_weaken_backpressure() -> None:
    contract = PINNED_CAPTURE_DATA_PLANE_CONTRACT
    with pytest.raises(CaptureDataPlaneContractError, match="exact false"):
        replace(contract, create_ack_is_durable_qualification=True)
    with pytest.raises(CaptureDataPlaneContractError, match="exact false"):
        replace(contract, production_sealable=True)
    with pytest.raises(CaptureDataPlaneContractError, match="queued_complete_steps"):
        replace(contract, maximum_queued_complete_steps=2)
    with pytest.raises(CaptureDataPlaneContractError, match="upload_streams_per_host"):
        replace(contract, upload_streams_per_host=3)


def test_exact_builtin_identity_types_and_content_paths_are_required() -> None:
    class _EqualString(str):
        def __eq__(self, other: object) -> bool:
            return True

        __hash__ = str.__hash__

    with pytest.raises(CaptureDataPlaneContractError, match="exact string"):
        build_capture_work_plan(
            work_id=_EqualString("forged"),
            input_range_sha256="e" * 64,
            start_token_ordinal=0,
            token_count=1,
        )
    with pytest.raises(CaptureDataPlaneContractError, match="exact string"):
        fragment_content_object_id(_EqualString("f" * 64))
    with pytest.raises(CaptureDataPlaneContractError, match="exact string"):
        replace(
            PINNED_CAPTURE_DATA_PLANE_CONTRACT,
            batchsplit_contract_sha256=_EqualString(
                PINNED_PREFIX_BATCHSPLIT_CONTRACT_SHA256
            ),
        )
    plan = build_capture_work_plan(
        work_id="integer-type-check",
        input_range_sha256="f" * 64,
        start_token_ordinal=0,
        token_count=1,
    )
    with pytest.raises(CaptureDataPlaneContractError, match="step_count"):
        replace(plan, step_count=True)
    assert fragment_content_object_id("f" * 64) == (f"objects/sha256/ff/{'f' * 64}.bin")
