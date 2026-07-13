"""Spec §15 enrichment R_M(e) and the §16.3 load floor: exact values against a
naive reference, sign correctness on planted structure, floor semantics, and
the stats artifact loader."""

import math

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from moe_sae.analysis.enrichment import (
    EPSILON,
    RouterStats,
    enrichment,
    enrichment_analysis,
    load_floor_mask,
    load_stats,
)

N_BANKS = 64


def uniform_stats(starved=(), count=10_000_000):
    uni = np.full(N_BANKS, 1.0 / N_BANKS)
    counts = np.full(N_BANKS, count, np.int64)
    for b in starved:
        counts[b] = 100
    return RouterStats(
        p_route={"image": uni.copy(), "text": uni.copy()},
        p_gate={"image": uni.copy(), "text": uni.copy()},
        bank_counts=counts,
    )


def planted_q(banks, weights=None):
    q = np.zeros(N_BANKS)
    banks = list(banks)
    w = weights if weights is not None else [1.0 / len(banks)] * len(banks)
    for b, wi in zip(banks, w):
        q[b] = wi
    return q


# --------------------------------------------------------------------------- #
# Exact values                                                                 #
# --------------------------------------------------------------------------- #


def test_enrichment_exact_against_naive():
    rng = np.random.default_rng(0)
    q = rng.dirichlet(np.ones(N_BANKS))
    p = rng.dirichlet(np.ones(N_BANKS))
    r = enrichment(q, p)
    for e in range(N_BANKS):
        expected = math.log((q[e] + EPSILON) / (p[e] + EPSILON))
        assert r[e] == pytest.approx(expected, rel=1e-12)


def test_enrichment_zero_bank_uses_epsilon():
    q = planted_q([0], [1.0])  # everything on bank 0
    p = np.full(N_BANKS, 1.0 / N_BANKS)
    r = enrichment(q, p)
    assert r[1] == pytest.approx(math.log(EPSILON / (1.0 / N_BANKS + EPSILON)))
    assert r[0] == pytest.approx(math.log((1.0 + EPSILON) / (1.0 / N_BANKS + EPSILON)))


def test_enrichment_sign_correct_on_planted_structure():
    # visual mass concentrated on banks 1,2,3 against a uniform gate baseline:
    # enriched banks positive, all-but-noise banks negative.
    q_img = planted_q([1, 2, 3])
    q_img[10] = 1e-5  # cross-modality noise
    q_img = q_img / q_img.sum()
    stats = uniform_stats()
    r = enrichment(q_img, stats.p_gate["image"])
    assert all(r[b] > 0 for b in (1, 2, 3))
    assert r[10] < 0
    assert r[40] < 0  # never-fired bank


# --------------------------------------------------------------------------- #
# Load floor (spec §16.3)                                                      #
# --------------------------------------------------------------------------- #


def test_load_floor_mask_boundary():
    counts = np.array([499_999, 500_000, 500_001] + [10**6] * (N_BANKS - 3))
    mask = load_floor_mask(counts, 500_000)
    # "fewer than 500,000 ... are excluded": strict-less excluded, equal kept.
    assert not mask[0] and mask[1] and mask[2]
    assert mask.sum() == N_BANKS - 1


def test_enrichment_analysis_reports_all_counts_and_masks_floor():
    stats = uniform_stats(starved=(2, 40))
    q = {"image": planted_q([1, 2, 3]), "text": planted_q([10, 11])}
    report = enrichment_analysis(q, stats, load_floor=500_000)
    assert len(report["bank_counts"]) == N_BANKS  # reported for all 64 banks
    assert report["n_banks_above_floor"] == N_BANKS - 2
    assert report["floor_mask"][2] is False and report["floor_mask"][1] is True
    r_img = report["R"]["image"]["vs_gate"]
    assert r_img[2] is None and r_img[40] is None  # excluded below the floor
    assert r_img[1] is not None and r_img[1] > 0
    assert r_img[10] is not None and r_img[10] < 0
    # vs_route also present and masked identically
    assert report["R"]["image"]["vs_route"][2] is None
    # missing modality (global condition) -> null
    report2 = enrichment_analysis({"image": None, "text": None}, stats, 500_000)
    assert report2["R"]["image"] is None


def test_enrichment_analysis_split_counts_override_a13():
    """A13: when the analysis dump's per-split counts are given, THEY carry
    the floor — a bank starved in the split is excluded even though the
    stats-pass loads are healthy; counts are reported for all 64 banks."""
    stats = uniform_stats()  # healthy stats-pass loads for every bank
    split_counts = np.full(N_BANKS, 100, dtype=np.int64)
    split_counts[2] = 0  # starved IN THE SPLIT UNDER ANALYSIS
    q = {"image": planted_q([1, 2, 3]), "text": planted_q([10, 11])}
    report = enrichment_analysis(q, stats, load_floor=1.0, bank_counts=split_counts)
    assert report["bank_counts"] == [int(c) for c in split_counts]
    assert report["floor_mask"][2] is False and report["floor_mask"][1] is True
    assert report["R"]["image"]["vs_gate"][2] is None
    assert report["R"]["image"]["vs_gate"][1] is not None
    assert "split" in report["load_floor_source"]


def test_enrichment_analysis_exact_values():
    stats = uniform_stats()
    q_img = planted_q([1, 2, 3])
    report = enrichment_analysis({"image": q_img, "text": planted_q([10, 11])}, stats, 1)
    expected = math.log((1 / 3 + EPSILON) / (1 / 64 + EPSILON))
    assert report["R"]["image"]["vs_gate"][1] == pytest.approx(expected, rel=1e-12)
    assert report["R"]["image"]["vs_route"][1] == pytest.approx(expected, rel=1e-12)
    assert report["epsilon"] == EPSILON and report["log_base"] == "e"


# --------------------------------------------------------------------------- #
# Stats artifact loader                                                        #
# --------------------------------------------------------------------------- #


def test_load_stats_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    tensors = {}
    expect = {}
    # Producer naming (train/stats_pass.py): the image modality is "visual".
    for key in ("p_route_visual", "p_route_text", "p_gate_visual", "p_gate_text"):
        v = rng.dirichlet(np.ones(N_BANKS)).astype(np.float32)
        tensors[key] = torch.from_numpy(v)
        expect[key] = v
    counts = rng.integers(0, 10**7, N_BANKS)
    tensors["bank_counts"] = torch.from_numpy(counts)
    save_file(tensors, str(tmp_path / "stats.safetensors"))

    # loadable from the file path and from the containing directory
    for target in (tmp_path / "stats.safetensors", tmp_path):
        stats = load_stats(target)
        np.testing.assert_allclose(stats.p_gate["image"], expect["p_gate_visual"], rtol=1e-6)
        np.testing.assert_allclose(stats.p_route["text"], expect["p_route_text"], rtol=1e-6)
        np.testing.assert_array_equal(stats.bank_counts, counts)
        assert stats.bank_counts.dtype == np.int64
