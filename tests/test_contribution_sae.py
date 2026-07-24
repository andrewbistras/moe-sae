"""Scientific contract tests for Gemma Experiment 2."""

from __future__ import annotations

import numpy as np
import pytest

from moe_sae.contribution_sae import (
    ContributionSAEConfig,
    ContributionSAEError,
    canonical_router_order,
    condition_bank_ids,
    gain_recovery,
    pilot_contribution_config,
    production_contribution_config,
    sequence_bank_permutation,
    validate_contribution_l0,
    weighted_contribution_token_scores,
    weighted_expert_sum,
)


def test_production_matrix_preserves_candidate_and_parameter_matches() -> None:
    configs = {
        condition: production_contribution_config(
            condition,
            randomization_key=17,
        )
        for condition in ("A", "R", "B", "F")
    }

    assert configs["A"].candidates_per_token == 8_192
    assert configs["R"].candidates_per_token == 8_192
    assert configs["B"].candidates_per_token == 8_192
    assert configs["F"].candidates_per_token == 131_072
    assert configs["A"].total_features == 131_072
    assert configs["A"].total_features == configs["F"].total_features
    assert configs["A"].parameter_count == 738_328_576
    assert configs["A"].parameter_count == configs["F"].parameter_count
    assert configs["A"].total_l0 == configs["F"].total_l0 == 64
    assert configs["A"].candidates_per_token / configs["A"].total_features == 0.0625


def test_pilot_matrix_is_a_scaled_version_of_the_same_comparisons() -> None:
    configs = {
        condition: pilot_contribution_config(condition, randomization_key=1)
        for condition in ("A", "R", "B", "F")
    }

    assert configs["A"].candidates_per_token == 2_048
    assert configs["A"].candidates_per_token == configs["B"].candidates_per_token
    assert configs["A"].total_features == 32_768
    assert configs["A"].total_features == configs["F"].total_features
    assert configs["A"].parameter_count == configs["F"].parameter_count
    assert configs["A"].total_l0 == configs["F"].total_l0 == 16


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"condition": "Z"}, "condition must be one of"),
        ({"condition": "A", "top_routed": 129}, "cannot exceed"),
        (
            {"condition": "A", "bank_width": 4, "k_per_contribution": 5},
            "cannot exceed",
        ),
        ({"condition": "A", "randomization_key": -1}, "randomization_key"),
    ],
)
def test_config_rejects_invalid_layouts(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ContributionSAEError, match=message):
        ContributionSAEConfig(**kwargs)  # type: ignore[arg-type]


def test_canonical_router_order_has_a_total_deterministic_tie_break() -> None:
    expert_ids = np.asarray(
        [
            [9, 4, 7, 2, 6, 5, 1, 3],
            [17, 12, 11, 16, 14, 10, 15, 13],
        ],
        dtype=np.uint8,
    )
    weights = np.asarray(
        [
            [0.10, 0.20, 0.15, 0.20, 0.10, 0.05, 0.15, 0.05],
            [0.125] * 8,
        ],
        dtype=np.float32,
    )

    ordered_ids, ordered_weights, slots = canonical_router_order(
        expert_ids,
        weights,
    )

    np.testing.assert_array_equal(
        ordered_ids[0],
        np.asarray([2, 4, 1, 7, 6, 9, 3, 5], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        ordered_ids[1],
        np.asarray([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.take_along_axis(expert_ids, slots, axis=1),
        ordered_ids,
    )
    np.testing.assert_array_equal(
        np.take_along_axis(weights, slots, axis=1),
        ordered_weights,
    )


@pytest.mark.parametrize(
    ("ids", "weights", "message"),
    [
        (
            np.arange(7, dtype=np.uint8)[None, :],
            np.full((1, 7), 1 / 7, dtype=np.float32),
            r"shape \[N, 8\]",
        ),
        (
            np.asarray([[0, 0, 1, 2, 3, 4, 5, 6]], dtype=np.uint8),
            np.full((1, 8), 1 / 8, dtype=np.float32),
            "unique",
        ),
        (
            np.asarray([[0, 1, 2, 3, 4, 5, 6, 128]], dtype=np.int16),
            np.full((1, 8), 1 / 8, dtype=np.float32),
            r"outside \[0, 127\]",
        ),
        (
            np.arange(8, dtype=np.uint8)[None, :],
            np.full((1, 8), 0.1, dtype=np.float32),
            "sum to one",
        ),
        (
            np.arange(8, dtype=np.uint8)[None, :],
            np.asarray([[np.nan] + [1 / 7] * 7], dtype=np.float32),
            "finite",
        ),
    ],
)
def test_canonical_router_order_rejects_invalid_routes(
    ids: np.ndarray,
    weights: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ContributionSAEError, match=message):
        canonical_router_order(ids, weights)


def test_sequence_randomization_is_stable_keyed_and_bijective() -> None:
    first = sequence_bank_permutation(81, key=9, num_experts=128)
    repeat = sequence_bank_permutation(81, key=9, num_experts=128)
    other_sequence = sequence_bank_permutation(82, key=9, num_experts=128)
    other_key = sequence_bank_permutation(81, key=10, num_experts=128)

    np.testing.assert_array_equal(first, repeat)
    np.testing.assert_array_equal(np.sort(first), np.arange(128))
    assert not np.array_equal(first, other_sequence)
    assert not np.array_equal(first, other_key)


def test_condition_bank_ids_implements_true_randomized_and_rank_controls() -> None:
    ids = np.asarray(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ],
        dtype=np.uint8,
    )
    sequences = np.asarray([100, 100, 101], dtype=np.int64)

    a = condition_bank_ids(
        ids,
        sequences,
        production_contribution_config("A", randomization_key=5),
    )
    randomized = condition_bank_ids(
        ids,
        sequences,
        production_contribution_config("R", randomization_key=5),
    )
    b = condition_bank_ids(
        ids,
        sequences,
        production_contribution_config("B", randomization_key=5),
    )
    f = condition_bank_ids(
        ids,
        sequences,
        production_contribution_config("F", randomization_key=5),
    )

    np.testing.assert_array_equal(a, ids)
    np.testing.assert_array_equal(randomized[0], randomized[1])
    assert not np.array_equal(randomized[0], randomized[2])
    assert np.unique(randomized, axis=1).shape == randomized.shape
    ranks = np.broadcast_to(np.arange(8, dtype=np.int32), ids.shape)
    np.testing.assert_array_equal(b, ranks)
    np.testing.assert_array_equal(f, ranks)


def test_condition_bank_ids_rejects_non_vector_sequence_ids() -> None:
    config = production_contribution_config("A", randomization_key=0)
    with pytest.raises(ContributionSAEError, match=r"shape \[N\]"):
        condition_bank_ids(
            np.arange(8, dtype=np.uint8)[None, :],
            np.asarray(1, dtype=np.int64),
            config,
        )


def test_weighted_scores_use_squared_router_weights_and_mask_padding() -> None:
    target = np.ones((2, 8, 2), dtype=np.float32)
    reconstruction = np.full_like(target, 0.5)
    weights = np.full((2, 8), 1 / 8, dtype=np.float32)
    mean = np.zeros((2,), dtype=np.float32)

    scores = weighted_contribution_token_scores(
        target,
        reconstruction,
        weights,
        mean,
        valid_mask=np.asarray([True, False]),
    )

    np.testing.assert_allclose(scores.token_sse, [0.0625, 0.0])
    np.testing.assert_allclose(scores.token_sst, [0.25, 0.0])
    assert scores.tokens == 1
    assert scores.sse == pytest.approx(0.0625)
    assert scores.sst == pytest.approx(0.25)
    assert scores.nmse == pytest.approx(0.25)
    assert scores.ev == pytest.approx(0.75)


def test_weighted_scores_fail_closed_on_invalid_inputs() -> None:
    target = np.ones((1, 8, 2), dtype=np.float32)
    weights = np.full((1, 8), 1 / 8, dtype=np.float32)
    mean = np.zeros((2,), dtype=np.float32)

    with pytest.raises(ContributionSAEError, match="reconstruction"):
        weighted_contribution_token_scores(
            target,
            np.ones((1, 8, 3), dtype=np.float32),
            weights,
            mean,
        )
    with pytest.raises(ContributionSAEError, match="nonnegative"):
        weighted_contribution_token_scores(
            target,
            target,
            np.asarray([[-0.1] + [1.1 / 7] * 7], dtype=np.float32),
            mean,
        )
    with pytest.raises(ContributionSAEError, match="valid_mask"):
        weighted_contribution_token_scores(
            target,
            target,
            weights,
            mean,
            valid_mask=np.asarray([1], dtype=np.uint8),
        )


def test_zero_sst_does_not_produce_a_fabricated_ev() -> None:
    zeros = np.zeros((1, 8, 2), dtype=np.float32)
    scores = weighted_contribution_token_scores(
        zeros,
        zeros,
        np.full((1, 8), 1 / 8, dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
    )

    with pytest.raises(ContributionSAEError, match="SST must be positive"):
        _ = scores.ev


def test_weighted_expert_sum_matches_the_fp32_recomposition_oracle() -> None:
    expert_output = np.broadcast_to(
        np.arange(1, 9, dtype=np.float32)[None, :, None],
        (2, 8, 3),
    )
    weights = np.full((2, 8), 1 / 8, dtype=np.float32)

    routed_sum = weighted_expert_sum(expert_output, weights)

    np.testing.assert_allclose(routed_sum, np.full((2, 3), 4.5))
    assert routed_sum.dtype == np.float32


def test_exact_contribution_l0_includes_padding_contract() -> None:
    config = production_contribution_config("A", randomization_key=0)
    l0 = np.asarray([[8] * 8, [0] * 8], dtype=np.int16)
    valid = np.asarray([True, False])

    validate_contribution_l0(l0, valid, config)

    invalid = l0.copy()
    invalid[0, 3] = 7
    with pytest.raises(ContributionSAEError, match="exact per-contribution"):
        validate_contribution_l0(invalid, valid, config)
    with pytest.raises(ContributionSAEError, match=r"shape \[N\]"):
        validate_contribution_l0(l0[:1], np.asarray(True), config)


@pytest.mark.parametrize(
    ("ev_a", "ev_b", "ev_f", "expected"),
    [
        (0.6, 0.5, 0.7, 50.0),
        (0.8, 0.5, 0.7, 150.0),
        (0.4, 0.5, 0.7, -50.0),
        (0.6, 0.5, 0.5, None),
        (0.6, 0.5, 0.4, None),
    ],
)
def test_gain_recovery_is_unclipped_and_requires_positive_f_over_b(
    ev_a: float,
    ev_b: float,
    ev_f: float,
    expected: float | None,
) -> None:
    result = gain_recovery(ev_a, ev_b, ev_f)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_gain_recovery_rejects_nonfinite_ev() -> None:
    with pytest.raises(ContributionSAEError, match="finite"):
        gain_recovery(float("nan"), 0.5, 0.7)
