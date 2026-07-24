from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from moe_sae.diagnostics import (
    PAIRED_BOOTSTRAP_REPLICATES,
    PAIRED_BOOTSTRAP_SCHEMA_VERSION,
    PAIRED_SSE_SCHEMA_VERSION,
    ROUTE_PERMUTATION_DOMAIN,
    SEQUENCE_SSE_SCHEMA_VERSION,
    DiagnosticError,
    PairedSequenceSSEAccumulator,
    SequenceSSEAccumulator,
    apply_route_mode,
    merge_and_pair_sequence_sse_summaries,
    merge_sequence_sse_summaries,
    pair_sequence_sse_summaries,
    paired_sequence_cluster_bootstrap_ci,
    sequence_expert_permutation,
    validate_paired_sequence_bootstrap_ci,
    validate_paired_sequence_sse_summary,
    validate_sequence_sse_summary,
)


def test_sequence_permutation_is_stable_bijective_and_keyed() -> None:
    first = sequence_expert_permutation(42, seed=7, num_experts=128)
    assert np.array_equal(
        first, sequence_expert_permutation(42, seed=7, num_experts=128)
    )
    assert sorted(first.tolist()) == list(range(128))
    assert not np.array_equal(
        first, sequence_expert_permutation(43, seed=7, num_experts=128)
    )
    assert not np.array_equal(
        first, sequence_expert_permutation(42, seed=8, num_experts=128)
    )


def test_route_modes_preserve_shape_dtype_range_and_token_uniqueness() -> None:
    expert_ids = np.array(
        [
            [0, 3, 5, 7],
            [0, 3, 5, 7],
            [1, 2, 4, 6],
            [0, 3, 5, 7],
        ],
        dtype=np.uint8,
    )
    sequence_ids = np.array([10, 10, 11, 10], dtype=np.int64)

    true = apply_route_mode(
        expert_ids, sequence_ids, mode="true", seed=123, num_experts=8
    )
    shuffled = apply_route_mode(
        expert_ids, sequence_ids, mode="shuffled", seed=123, num_experts=8
    )
    repeated = apply_route_mode(
        expert_ids, sequence_ids, mode="shuffled", seed=123, num_experts=8
    )

    assert np.array_equal(true, expert_ids)
    assert true is not expert_ids
    assert shuffled.shape == expert_ids.shape
    assert shuffled.dtype == expert_ids.dtype
    assert np.array_equal(shuffled, repeated)
    assert shuffled.min() >= 0 and shuffled.max() < 8
    assert all(len(set(row.tolist())) == row.size for row in shuffled)
    # The same original labels use exactly the same mapping everywhere in
    # sequence 10, including non-contiguous batch rows.
    assert np.array_equal(shuffled[0], shuffled[1])
    assert np.array_equal(shuffled[0], shuffled[3])
    expected = sequence_expert_permutation(10, seed=123, num_experts=8)
    assert np.array_equal(shuffled[0], expected[expert_ids[0]])


@pytest.mark.parametrize(
    ("ids", "sequences", "num_experts", "message"),
    [
        (np.array([0, 1]), np.array([1]), 8, "shape"),
        (np.array([[0, 1]]), np.array([[1]]), 8, "shape"),
        (np.array([[0.0, 1.0]]), np.array([1]), 8, "integer"),
        (np.array([[0, 1]]), np.array([1], dtype=np.uint64), 8, "signed"),
        (np.array([[0, 0]]), np.array([1]), 8, "unique"),
        (np.array([[0, 8]]), np.array([1]), 8, "outside"),
        (
            np.array([[0, 1]], dtype=np.uint8),
            np.array([1]),
            300,
            "cannot represent",
        ),
    ],
)
def test_route_modes_reject_invalid_routes(
    ids, sequences, num_experts, message
) -> None:
    with pytest.raises(DiagnosticError, match=message):
        apply_route_mode(
            ids,
            sequences,
            mode="shuffled",
            seed=0,
            num_experts=num_experts,
        )


def test_route_modes_reject_implicit_or_invalid_randomization_identity() -> None:
    ids = np.array([[0, 1]], dtype=np.uint8)
    sequences = np.array([1], dtype=np.int64)
    with pytest.raises(DiagnosticError, match="mode"):
        apply_route_mode(ids, sequences, mode="random", seed=0)  # type: ignore[arg-type]
    with pytest.raises(DiagnosticError, match="seed"):
        apply_route_mode(ids, sequences, mode="true", seed=-1)
    with pytest.raises(DiagnosticError, match="sequence_id"):
        sequence_expert_permutation(2**63, seed=0)
    with pytest.raises(DiagnosticError, match="domain"):
        apply_route_mode(
            ids,
            sequences,
            mode="shuffled",
            seed=0,
            domain="wrong-domain",
        )
    assert ROUTE_PERMUTATION_DOMAIN == "gemma4-gate1-router-permutation-v1"


def _paired_summary() -> dict:
    accumulator = PairedSequenceSSEAccumulator(
        np.array([0.0, 0.0]),
        baseline_label="A:true",
        candidate_label="A:shuffled",
    )
    # Sequence 7 deliberately spans calls.  The final rows must merge it into
    # one cluster rather than treating each batch fragment as independent.
    accumulator.update(
        np.array([7, 8], dtype=np.int64),
        np.array([[1.0, 0.0], [0.0, 2.0]]),
        np.array([[0.5, 0.0], [0.0, 1.0]]),
        np.array([[0.0, 0.0], [0.0, 1.5]]),
    )
    accumulator.update(
        np.array([7], dtype=np.int64),
        np.array([[1.0, 1.0]]),
        np.array([[0.5, 0.5]]),
        np.array([[0.75, 0.75]]),
    )
    return accumulator.finalize()


def test_paired_accumulator_merges_sequences_and_preserves_bootstrap_clusters() -> None:
    result = _paired_summary()
    assert result["schema_version"] == PAIRED_SSE_SCHEMA_VERSION
    assert result["tokens"] == 3
    assert result["sequence_count"] == 2
    assert [row["sequence_id"] for row in result["clusters"]] == [7, 8]
    assert [row["tokens"] for row in result["clusters"]] == [2, 1]
    assert result["sst"] == pytest.approx(7.0)
    assert result["baseline_sse"] == pytest.approx(1.75)
    assert result["candidate_sse"] == pytest.approx(1.375)
    assert result["baseline_nmse"] == pytest.approx(0.25)
    assert result["candidate_nmse"] == pytest.approx(1.375 / 7.0)
    assert result["candidate_minus_baseline_ev"] == pytest.approx(0.375 / 7.0)
    assert sum(
        row["candidate_minus_baseline_sse"] for row in result["clusters"]
    ) == pytest.approx(result["candidate_minus_baseline_sse"])
    # It can be serialized without NumPy scalars or non-standard JSON values.
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_paired_accumulator_is_batch_partition_invariant() -> None:
    sequence_ids = np.array([2, 1, 2, 3], dtype=np.int64)
    target = np.arange(8, dtype=np.float64).reshape(4, 2)
    baseline = target * 0.5
    candidate = target * 0.75

    whole = PairedSequenceSSEAccumulator(
        np.ones(2), baseline_label="B", candidate_label="A"
    )
    whole.update(sequence_ids, target, baseline, candidate)
    split = PairedSequenceSSEAccumulator(
        np.ones(2), baseline_label="B", candidate_label="A"
    )
    split.update(sequence_ids[:2], target[:2], baseline[:2], candidate[:2])
    split.update(sequence_ids[2:], target[2:], baseline[2:], candidate[2:])
    assert split.finalize() == whole.finalize()


def test_paired_accumulator_rejects_invalid_batches_and_zero_sst() -> None:
    accumulator = PairedSequenceSSEAccumulator(
        np.zeros(2), baseline_label="true", candidate_label="shuffled"
    )
    with pytest.raises(DiagnosticError, match="identical"):
        accumulator.update(
            np.array([1], dtype=np.int64),
            np.zeros((1, 2)),
            np.zeros((1, 1)),
            np.zeros((1, 2)),
        )
    with pytest.raises(DiagnosticError, match="signed"):
        accumulator.update(
            np.array([1], dtype=np.uint64),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
        )
    with pytest.raises(DiagnosticError, match="finite"):
        accumulator.update(
            np.array([1], dtype=np.int64),
            np.array([[np.nan, 0.0]]),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
        )
    accumulator.update(
        np.array([1], dtype=np.int64),
        np.zeros((1, 2)),
        np.zeros((1, 2)),
        np.zeros((1, 2)),
    )
    with pytest.raises(DiagnosticError, match="SST"):
        accumulator.finalize()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(schema_version="old"),
        lambda value: value.update(tokens=value["tokens"] + 1),
        lambda value: value.update(candidate_ev=value["candidate_ev"] + 0.1),
        lambda value: value["clusters"][0].update(candidate_minus_baseline_sse=999.0),
        lambda value: value["clusters"].reverse(),
        lambda value: value.update(unexpected=True),
    ],
)
def test_paired_summary_validator_fails_closed(mutation) -> None:
    value = copy.deepcopy(_paired_summary())
    mutation(value)
    with pytest.raises(DiagnosticError):
        validate_paired_sequence_sse_summary(value)


def _single_summary(label: str, scale: float) -> dict:
    accumulator = SequenceSSEAccumulator(label=label)
    accumulator.update(
        np.array([7, 8, 7], dtype=np.int64),
        np.array([1.0, 2.0, 3.0]) * scale,
        np.array([4.0, 5.0, 6.0]),
    )
    return accumulator.finalize()


def test_device_sse_accumulator_and_pairing_preserve_clusters() -> None:
    baseline = _single_summary("A:true", 1.0)
    candidate = _single_summary("A:shuffled", 1.5)
    assert baseline["schema_version"] == SEQUENCE_SSE_SCHEMA_VERSION
    assert baseline["tokens"] == 3
    assert baseline["sequence_count"] == 2
    assert baseline["sst"] == pytest.approx(15.0)
    assert baseline["sse"] == pytest.approx(6.0)
    assert baseline["nmse"] == pytest.approx(0.4)
    paired = pair_sequence_sse_summaries(baseline, candidate)
    assert paired["baseline_sse"] == pytest.approx(6.0)
    assert paired["candidate_sse"] == pytest.approx(9.0)
    assert paired["candidate_minus_baseline_ev"] == pytest.approx(-0.2)
    assert [row["sequence_id"] for row in paired["clusters"]] == [7, 8]


def test_device_sse_accumulator_masks_padding_and_rejects_tampering() -> None:
    accumulator = SequenceSSEAccumulator(label="A")
    accumulator.update(
        np.array([1, -1], dtype=np.int64),
        np.array([2.0, 999.0]),
        np.array([4.0, 999.0]),
        valid_mask=np.array([True, False]),
    )
    summary = accumulator.finalize()
    assert summary["tokens"] == 1
    assert summary["sse"] == pytest.approx(2.0)
    tampered = copy.deepcopy(summary)
    tampered["clusters"][0]["sse"] += 1.0
    with pytest.raises(DiagnosticError):
        validate_sequence_sse_summary(tampered)


def test_pairing_rejects_different_sequence_measure() -> None:
    baseline = _single_summary("A", 1.0)
    candidate = _single_summary("B", 1.0)
    candidate["clusters"][0]["sequence_id"] = 6
    with pytest.raises(DiagnosticError):
        pair_sequence_sse_summaries(baseline, candidate)


def _summary_fragment(
    label: str,
    sequence_ids: list[int],
    sse: list[float],
    sst: list[float],
) -> dict:
    accumulator = SequenceSSEAccumulator(label=label)
    accumulator.update(
        np.asarray(sequence_ids, dtype=np.int64),
        np.asarray(sse, dtype=np.float64),
        np.asarray(sst, dtype=np.float64),
    )
    return accumulator.finalize()


def _fragmented_pair() -> tuple[list[dict], list[dict]]:
    baseline = [
        _summary_fragment("B", [7, 8], [1.0, 2.0], [4.0, 5.0]),
        _summary_fragment("B", [9, 7], [4.0, 3.0], [7.0, 6.0]),
    ]
    # Deliberately partition the same sequence/token/SST measure differently.
    candidate = [
        _summary_fragment("A", [7], [1.5], [4.0]),
        _summary_fragment("A", [8, 7, 9], [1.0, 2.5, 5.0], [5.0, 6.0, 7.0]),
    ]
    return baseline, candidate


def test_merge_sums_partial_clusters_and_is_input_order_invariant() -> None:
    baseline_parts, _ = _fragmented_pair()

    merged = merge_sequence_sse_summaries(baseline_parts)
    reversed_merge = merge_sequence_sse_summaries(reversed(baseline_parts))

    assert merged == reversed_merge
    assert merged["label"] == "B"
    assert merged["tokens"] == 4
    assert merged["sequence_count"] == 3
    assert [row["sequence_id"] for row in merged["clusters"]] == [7, 8, 9]
    assert [row["tokens"] for row in merged["clusters"]] == [2, 1, 1]
    assert merged["clusters"][0]["sst"] == pytest.approx(10.0)
    assert merged["clusters"][0]["sse"] == pytest.approx(4.0)
    assert merged["sst"] == pytest.approx(22.0)
    assert merged["sse"] == pytest.approx(10.0)
    assert json.loads(json.dumps(merged, allow_nan=False)) == merged


def test_merge_rejects_empty_mixed_label_and_invalid_fragments() -> None:
    baseline_parts, candidate_parts = _fragmented_pair()
    with pytest.raises(DiagnosticError, match="empty"):
        merge_sequence_sse_summaries([])
    with pytest.raises(DiagnosticError, match="same label"):
        merge_sequence_sse_summaries([baseline_parts[0], candidate_parts[0]])
    tampered = copy.deepcopy(baseline_parts[0])
    tampered["clusters"][0]["sse"] += 1.0
    with pytest.raises(DiagnosticError):
        merge_sequence_sse_summaries([tampered])


def test_merge_then_pair_requires_exact_sequence_tokens_and_sst() -> None:
    baseline_parts, candidate_parts = _fragmented_pair()
    paired = merge_and_pair_sequence_sse_summaries(baseline_parts, candidate_parts)
    assert paired["tokens"] == 4
    assert paired["sequence_count"] == 3
    assert paired["baseline_sse"] == pytest.approx(10.0)
    assert paired["candidate_sse"] == pytest.approx(10.0)
    assert paired["candidate_minus_baseline_ev"] == pytest.approx(0.0)

    candidate = SequenceSSEAccumulator(label="A")
    candidate.update(
        np.array([7, 8, 7], dtype=np.int64),
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0 + 5e-10, 5.0, 6.0]),
    )
    with pytest.raises(DiagnosticError, match="exact SST"):
        pair_sequence_sse_summaries(_single_summary("B", 1.0), candidate.finalize())


def test_paired_sequence_bootstrap_is_deterministic_json_safe_and_bound() -> None:
    baseline_parts, candidate_parts = _fragmented_pair()
    paired = merge_and_pair_sequence_sse_summaries(baseline_parts, candidate_parts)
    first = paired_sequence_cluster_bootstrap_ci(paired, seed=2**64 - 1)
    second = paired_sequence_cluster_bootstrap_ci(paired, seed=2**64 - 1)

    assert first == second
    assert first["schema_version"] == PAIRED_BOOTSTRAP_SCHEMA_VERSION
    assert first["replicates"] == PAIRED_BOOTSTRAP_REPLICATES
    assert first["seed"] == 2**64 - 1
    assert first["point_candidate_minus_baseline_ev"] == pytest.approx(0.0)
    assert first["ci_lower"] <= first["ci_upper"]
    assert validate_paired_sequence_bootstrap_ci(first, paired) == first
    assert json.loads(json.dumps(first, allow_nan=False)) == first

    for invalid_seed in (True, -1, 2**64):
        with pytest.raises(DiagnosticError, match="seed"):
            paired_sequence_cluster_bootstrap_ci(paired, seed=invalid_seed)


def test_paired_bootstrap_identical_errors_have_exact_zero_interval() -> None:
    baseline = _single_summary("B", 1.0)
    candidate = copy.deepcopy(baseline)
    candidate["label"] = "A"
    paired = pair_sequence_sse_summaries(baseline, candidate)

    result = paired_sequence_cluster_bootstrap_ci(paired, seed=7)

    assert result["point_candidate_minus_baseline_ev"] == 0.0
    assert result["bootstrap_mean_candidate_minus_baseline_ev"] == 0.0
    assert result["bootstrap_standard_error"] == 0.0
    assert result["ci_lower"] == 0.0
    assert result["ci_upper"] == 0.0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(replicates=19_999),
        lambda value: value.update(paired_summary_sha256="0" * 64),
        lambda value: value.update(
            bootstrap_mean_candidate_minus_baseline_ev=(
                value["bootstrap_mean_candidate_minus_baseline_ev"] + 1e-4
            )
        ),
        lambda value: value.update(ci_lower=value["ci_upper"] + 1.0),
        lambda value: value.update(unexpected=True),
    ],
)
def test_paired_bootstrap_validator_fails_closed(mutation) -> None:
    baseline_parts, candidate_parts = _fragmented_pair()
    paired = merge_and_pair_sequence_sse_summaries(baseline_parts, candidate_parts)
    result = paired_sequence_cluster_bootstrap_ci(paired, seed=11)
    mutation(result)
    with pytest.raises(DiagnosticError):
        validate_paired_sequence_bootstrap_ci(result, paired)


def test_paired_bootstrap_fails_if_a_resample_can_have_zero_sst() -> None:
    baseline = _summary_fragment("B", [1, 2], [0.0, 1.0], [0.0, 1.0])
    candidate = _summary_fragment("A", [1, 2], [0.0, 0.5], [0.0, 1.0])
    paired = pair_sequence_sse_summaries(baseline, candidate)

    with pytest.raises(DiagnosticError, match="zero-SST"):
        paired_sequence_cluster_bootstrap_ci(paired, seed=0)
