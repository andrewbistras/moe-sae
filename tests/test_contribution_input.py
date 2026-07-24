from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    SCHEMA_VERSION,
    TOP_K_EXPERTS,
    ContributionBatch,
    bf16_bits_to_float32,
    float32_to_bf16_bits,
)
from moe_sae.contribution_input import (
    CONTRIBUTION_INPUT_SCHEMA_VERSION,
    PROCESS_PARTITION_STRATEGY,
    ContributionInputError,
    ContributionProcessInput,
    PADDING_POSITION,
    PADDING_SEQUENCE_ID,
    merge_capacity_receipts,
)
from moe_sae.contribution_sae import ContributionSAEConfig


def wire_batch(tokens: int) -> ContributionBatch:
    expert = np.arange(
        tokens * TOP_K_EXPERTS * HIDDEN_SIZE,
        dtype=np.float32,
    ).reshape(tokens, TOP_K_EXPERTS, HIDDEN_SIZE)
    expert = expert / np.float32(TOP_K_EXPERTS * HIDDEN_SIZE)
    weights = np.asarray(
        [0.25, 0.1875, 0.15625, 0.125, 0.09375, 0.078125, 0.0625, 0.046875],
        dtype=np.float32,
    )
    expert_ids = np.stack(
        [
            (np.arange(TOP_K_EXPERTS, dtype=np.uint8) + row * 4) % 128
            for row in range(tokens)
        ]
    )
    sequence_ids = np.asarray(
        [100 + row // 2 for row in range(tokens)],
        dtype=np.int64,
    )
    return ContributionBatch(
        expert_output=float32_to_bf16_bits(expert),
        expert_ids=expert_ids,
        router_weights=float32_to_bf16_bits(
            np.broadcast_to(weights, (tokens, TOP_K_EXPERTS))
        ),
        shared_output=float32_to_bf16_bits(
            np.zeros((tokens, HIDDEN_SIZE), dtype=np.float32)
        ),
        sequence_id=sequence_ids,
        position=np.arange(tokens, dtype=np.uint16),
    )


class FakeDataset:
    def __init__(self, value: ContributionBatch):
        self.value = value
        self.total_tokens = value.tokens
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "manifest_sha256": "a" * 64,
        }
        self.reads: list[tuple[int, int]] = []

    def iter_token_range(self, start: int, end: int):
        self.reads.append((start, end))
        yield ContributionBatch(
            **{
                name: np.ascontiguousarray(
                    np.asarray(getattr(self.value, name))[start:end]
                )
                for name in ContributionBatch.__dataclass_fields__
            }
        )


def input_for(
    condition: str,
    *,
    tokens: int = 10,
    process_count: int = 1,
    process_index: int = 0,
    randomization_key: int = 17,
) -> tuple[FakeDataset, ContributionProcessInput]:
    dataset = FakeDataset(wire_batch(tokens))
    layout = ContributionSAEConfig(
        condition=condition,
        bank_width=16,
        k_per_contribution=2,
        randomization_key=randomization_key,
    )
    return dataset, ContributionProcessInput(
        dataset,
        layout,
        total_tokens=tokens,
        global_microbatch_tokens=8,
        microbatches_per_update=2,
        process_count=process_count,
        process_index=process_index,
    )


def test_adapter_decodes_wire_bf16_and_constructs_a_and_r_internally():
    dataset_a, source_a = input_for("A", tokens=8)
    dataset_r, source_r = input_for("R", tokens=8)

    batch_a = next(source_a.iter_updates())[0]
    batch_r = next(source_r.iter_updates())[0]

    assert batch_a.expert_output.dtype == np.float32
    assert batch_a.router_weights.dtype == np.float32
    assert np.array_equal(
        batch_a.expert_output,
        bf16_bits_to_float32(dataset_a.value.expert_output),
    )
    assert np.array_equal(batch_a.bank_ids, dataset_a.value.expert_ids)
    assert not np.array_equal(batch_r.bank_ids, batch_a.bank_ids)
    # Two tokens from the same packed sequence use one stable bijection.
    first_ids = dataset_r.value.expert_ids[:2]
    first_banks = batch_r.bank_ids[:2]
    shared_experts = np.intersect1d(first_ids[0], first_ids[1])
    for expert in shared_experts:
        left = first_banks[0, np.flatnonzero(first_ids[0] == expert)[0]]
        right = first_banks[1, np.flatnonzero(first_ids[1] == expert)[0]]
        assert left == right
    assert source_a.receipt.layout_sha256 != source_r.receipt.layout_sha256


@pytest.mark.parametrize("condition", ["B", "F"])
def test_rank_conditions_ignore_expert_identity(condition):
    _dataset, source = input_for(condition, tokens=8)

    microbatch = next(source.iter_updates())[0]

    expected = np.broadcast_to(
        np.arange(TOP_K_EXPERTS, dtype=np.int32),
        microbatch.bank_ids.shape,
    )
    assert np.array_equal(microbatch.bank_ids, expected)


def test_process_partition_and_tail_masks_cover_each_token_once():
    dataset0, source0 = input_for(
        "A",
        tokens=10,
        process_count=2,
        process_index=0,
    )
    dataset1, source1 = input_for(
        "A",
        tokens=10,
        process_count=2,
        process_index=1,
    )

    update0 = next(source0.iter_updates())
    update1 = next(source1.iter_updates())

    assert [int(item.valid_mask.sum()) for item in update0] == [4, 1]
    assert [int(item.valid_mask.sum()) for item in update1] == [4, 1]
    assert dataset0.reads == [(0, 4), (4, 5)]
    assert dataset1.reads == [(5, 9), (9, 10)]
    assert source0.process_partition_strategy == PROCESS_PARTITION_STRATEGY
    assert source0.receipt.schema_version == CONTRIBUTION_INPUT_SCHEMA_VERSION
    assert (source0.process_range_start, source0.process_range_end) == (0, 5)
    assert (source1.process_range_start, source1.process_range_end) == (5, 10)
    assert all(item.expert_output.shape == (4, 8, HIDDEN_SIZE) for item in update0)
    np.testing.assert_array_equal(update0[0].sequence_id, [100, 100, 101, 101])
    np.testing.assert_array_equal(update1[0].sequence_id, [102, 103, 103, 104])
    np.testing.assert_array_equal(update0[0].position, [0, 1, 2, 3])
    np.testing.assert_array_equal(update1[0].position, [5, 6, 7, 8])
    np.testing.assert_array_equal(
        update0[1].sequence_id,
        [102, PADDING_SEQUENCE_ID, PADDING_SEQUENCE_ID, PADDING_SEQUENCE_ID],
    )
    np.testing.assert_array_equal(
        update1[1].position,
        [9, PADDING_POSITION, PADDING_POSITION, PADDING_POSITION],
    )
    assert np.count_nonzero(update0[1].expert_output[1:]) == 0
    assert np.count_nonzero(update1[1].router_weights[1:]) == 0
    observed = [
        int(position)
        for update in (update0, update1)
        for batch in update
        for position in batch.position[batch.valid_mask]
    ]
    assert sorted(observed) == list(range(10))
    assert sum(batch.valid_mask.size for update in (update0, update1) for batch in update) - len(observed) == 6


def test_update_schedule_matches_frozen_effective_batch_tail():
    dataset = FakeDataset(wire_batch(1))
    dataset.total_tokens = 2_000_000
    source = ContributionProcessInput(
        dataset,
        ContributionSAEConfig(
            condition="A",
            bank_width=16,
            k_per_contribution=2,
        ),
        total_tokens=2_000_000,
        global_microbatch_tokens=4_096,
        microbatches_per_update=4,
        process_count=4,
        process_index=0,
    )

    assert source.planned_updates == 123
    assert 123 * 16_384 - 2_000_000 == 15_232
    assert source.receipt.planned_updates == 123


def test_capacity_scan_binds_condition_key_partition_and_margin():
    _dataset_a, source_a = input_for("A", tokens=10)
    _dataset_r, source_r = input_for("R", tokens=10)

    receipt_a = source_a.scan_local_capacity(margin_fraction=0.10)
    receipt_r = source_r.scan_local_capacity(margin_fraction=0.10)
    merged = merge_capacity_receipts((receipt_a, replace(receipt_a)))

    assert receipt_a.observed_maximum > 0
    assert receipt_a.capacity == np.ceil(receipt_a.observed_maximum * 1.1)
    assert receipt_a.receipt_sha256 != receipt_r.receipt_sha256
    assert merged.observed_maximum == receipt_a.observed_maximum
    assert merged.capacity == receipt_a.capacity
    with pytest.raises(ContributionInputError, match="different conditions"):
        merge_capacity_receipts((receipt_a, receipt_r))


def test_input_fails_closed_on_schema_range_and_partition_mismatch():
    dataset = FakeDataset(wire_batch(2))
    layout = ContributionSAEConfig(
        condition="A",
        bank_width=16,
        k_per_contribution=2,
    )
    dataset.manifest["schema_version"] = "gemma4-router-aggregate-v1"
    with pytest.raises(ContributionInputError, match="exact contribution-v2"):
        ContributionProcessInput(dataset, layout, total_tokens=2)

    dataset.manifest["schema_version"] = SCHEMA_VERSION
    with pytest.raises(ContributionInputError, match="exceeds"):
        ContributionProcessInput(dataset, layout, total_tokens=3)
    with pytest.raises(ContributionInputError, match="divide"):
        ContributionProcessInput(
            dataset,
            layout,
            total_tokens=2,
            global_microbatch_tokens=7,
            process_count=2,
        )
