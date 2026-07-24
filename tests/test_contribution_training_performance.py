from __future__ import annotations

from dataclasses import replace

import pytest

from moe_sae.contribution_performance import (
    ContributionPerformanceError,
    ExactPerformanceIdentity,
)
from moe_sae.contribution_sae import (
    pilot_contribution_config,
    production_contribution_config,
)
from moe_sae.contribution_training_performance import (
    SAEArithmetic,
    SAEQualificationObservation,
    build_sae_qualification_receipt,
    sae_training_useful_flops,
    sae_useful_flops_ledger,
)


_SHA = "a" * 64


def _identity() -> ExactPerformanceIdentity:
    return ExactPerformanceIdentity(_SHA, "b" * 64, "c" * 64)


def test_sae_flop_ledger_matches_candidate_and_sparse_decode_geometry() -> None:
    a = SAEArithmetic.from_contribution_layout(
        production_contribution_config("A", randomization_key=1)
    )
    f = SAEArithmetic.from_contribution_layout(
        production_contribution_config("F", randomization_key=1)
    )
    q = SAEArithmetic.shared_q()
    assert a.candidates_per_token == q.candidates_per_token == 8_192
    assert f.candidates_per_token == 131_072
    assert a.selected_features_per_token == f.selected_features_per_token == 64

    expected = 6 * 100 * 2_816 * (8_192 + 64)
    assert sae_training_useful_flops(a, scientific_tokens=100) == expected
    ledger = sae_useful_flops_ledger(
        a,
        scientific_tokens=1_000,
        measured_steps=10,
    )
    assert ledger["useful_flops"] == 10 * expected
    assert "capacity_padding" in ledger["excluded_from_useful_flops"]
    assert "rematerialized_forward_work" in ledger["excluded_from_useful_flops"]


def _observation(*, device_seconds: float = 1.0) -> SAEQualificationObservation:
    return SAEQualificationObservation(
        identity=_identity(),
        qualification_wall_seconds=max(2.0, device_seconds),
        measured_device_seconds=device_seconds,
        scientific_tokens=16_384,
        warmup_steps=2,
        measured_steps=2,
        compile_count=1,
        recompilations_after_warmup=0,
        oom_events=0,
        nonfinite_events=0,
        capacity_overflow_events=0,
        exact_l0=True,
        state_progress_exact=True,
        parameters_resident_before_timing=True,
        inputs_resident_before_timing=True,
        synchronized_device_timing=True,
        transition_output_sha256=_SHA,
        replay_output_sha256=_SHA,
    )


def test_sae_receipt_applies_the_same_ten_percent_gate() -> None:
    arithmetic = SAEArithmetic.from_contribution_layout(
        pilot_contribution_config("A", randomization_key=0)
    )
    evidence, bundle = build_sae_qualification_receipt(
        _observation(),
        arithmetic,
        peak_flops_per_second=3_000_000_000_000,
    )
    assert evidence.useful_prefix_mfu >= 0.10
    assert bundle["qualification_receipt"]["status"] == "passed"
    assert (
        bundle["qualification_receipt"]["identity_sha256"]
        == _identity().identity_sha256
    )

    with pytest.raises(ContributionPerformanceError, match="10%"):
        build_sae_qualification_receipt(
            _observation(device_seconds=600.0),
            arithmetic,
            peak_flops_per_second=3_000_000_000_000,
        )


def test_sae_receipt_rejects_nonreplayable_or_inexact_transition() -> None:
    arithmetic = SAEArithmetic.shared_q()
    with pytest.raises(ValueError, match="deterministic"):
        build_sae_qualification_receipt(
            replace(_observation(), replay_output_sha256="d" * 64),
            arithmetic,
            peak_flops_per_second=3_000_000_000_000,
        )
    with pytest.raises(ValueError, match="exact L0"):
        build_sae_qualification_receipt(
            replace(_observation(), exact_l0=False),
            arithmetic,
            peak_flops_per_second=3_000_000_000_000,
        )
