"""Spec §16.1–§16.3: JSD / entropy exact-value checks against naive references,
shuffled-label null and bootstrap behavior on planted structure, load-floor
exclusion, and top-6 stability (Jaccard / Spearman / selection frequency)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from moe_sae.analysis.enrichment import RouterStats
from moe_sae.analysis.mass import load_dump, per_example_masses, token_bank_masses
from moe_sae.analysis.stability import (
    bootstrap_analysis,
    entropy_stats,
    jaccard,
    jsd,
    observed_q,
    shuffled_label_jsds,
    spearman,
    stability_analysis,
    top_enriched,
)

N_BANKS = 64
TOP_K = 6
VIS_BANKS = (1, 2, 3)
TXT_BANKS = (10, 11)


# --------------------------------------------------------------------------- #
# Contract-format synthetic dump (same planted structure as test_analysis_mass)#
# --------------------------------------------------------------------------- #


def write_synth_dump(
    root, condition="A", f=16, n_examples=120, tokens_per_side=6, seed=0,
    signal=5.0, noise=0.05, vis_banks=VIS_BANKS, txt_banks=TXT_BANKS,
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pt = {k: [] for k in ("experts", "gates", "token_type", "example_id", "domain", "l0")}
    ac = {k: [] for k in ("act_token", "act_slot", "act_latent", "act_value")}
    t = 0
    for ex in range(n_examples):
        for tt in (1, 0):
            banks = vis_banks if tt else txt_banks
            for _ in range(tokens_per_side):
                others = np.setdiff1d(np.arange(N_BANKS), np.asarray(banks))
                e = np.concatenate(
                    [np.asarray(banks), rng.choice(others, TOP_K - len(banks), replace=False)]
                )
                rng.shuffle(e)
                n_acts = 0
                for b in banks:
                    ac["act_token"].append(t)
                    ac["act_slot"].append(int(np.argmax(e == b)))
                    ac["act_latent"].append(int(rng.integers(f)))
                    ac["act_value"].append(float(signal * (0.8 + 0.4 * rng.random())))
                    n_acts += 1
                ac["act_token"].append(t)
                ac["act_slot"].append(int(rng.integers(TOP_K)))
                ac["act_latent"].append(int(rng.integers(f)))
                ac["act_value"].append(float(noise * (0.5 + rng.random())))
                n_acts += 1
                pt["experts"].append(e)
                pt["gates"].append(rng.uniform(0.05, 0.3, TOP_K))
                pt["token_type"].append(tt)
                pt["example_id"].append(ex)
                pt["domain"].append(ex % 4)
                pt["l0"].append(n_acts)
                t += 1
    shard = {
        "experts": torch.from_numpy(np.asarray(pt["experts"], np.uint8)),
        "gates": torch.from_numpy(np.asarray(pt["gates"], np.float32)),
        "token_type": torch.from_numpy(np.asarray(pt["token_type"], np.uint8)),
        "example_id": torch.from_numpy(np.asarray(pt["example_id"], np.int32)),
        "domain": torch.from_numpy(np.asarray(pt["domain"], np.uint8)),
        "l0": torch.from_numpy(np.asarray(pt["l0"], np.int32)),
        "act_token": torch.from_numpy(np.asarray(ac["act_token"], np.int64)),
        "act_slot": torch.from_numpy(np.asarray(ac["act_slot"], np.uint8)),
        "act_latent": torch.from_numpy(np.asarray(ac["act_latent"], np.int32)),
        "act_value": torch.from_numpy(np.asarray(ac["act_value"], np.float32)),
    }
    save_file(shard, str(root / "shard_00000.safetensors"))
    (root / "meta.json").write_text(json.dumps({"condition": condition, "f": f}))
    return root


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


def load_planted(tmp_path, **kw):
    dump = load_dump(write_synth_dump(tmp_path / "dump", **kw))
    masses = token_bank_masses(dump)
    return dump, masses


# --------------------------------------------------------------------------- #
# JSD / entropy exact values                                                   #
# --------------------------------------------------------------------------- #


def jsd_ref(p, q):
    """Naive base-2 JSD."""
    p = np.asarray(p, float) / np.sum(p)
    q = np.asarray(q, float) / np.sum(q)
    m = 0.5 * (p + q)
    s = 0.0
    for a in (p, q):
        for ai, mi in zip(a, m):
            if ai > 0:
                s += 0.5 * ai * math.log2(ai / mi)
    return s


def test_jsd_exact_cases():
    assert jsd([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)  # disjoint -> 1 (base 2)
    assert jsd([0.3, 0.7], [0.3, 0.7]) == pytest.approx(0.0, abs=1e-12)
    # hand case: JSD([1,0],[1/2,1/2]) = 3/2 - log2(3)/... verified via reference
    assert jsd([1.0, 0.0], [0.5, 0.5]) == pytest.approx(jsd_ref([1, 0], [0.5, 0.5]))
    assert jsd([1.0, 0.0], [0.5, 0.5]) == pytest.approx(0.31127812, abs=1e-7)


def test_jsd_matches_naive_and_is_symmetric():
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = rng.dirichlet(np.ones(N_BANKS))
        q = rng.dirichlet(np.ones(N_BANKS))
        assert jsd(p, q) == pytest.approx(jsd_ref(p, q), rel=1e-10)
        assert jsd(p, q) == pytest.approx(jsd(q, p), rel=1e-12)
        assert 0.0 <= jsd(p, q) <= 1.0


def test_entropy_stats_exact():
    uni = np.full(N_BANKS, 1.0 / N_BANKS)
    s = entropy_stats(uni)
    assert s["H_norm"] == pytest.approx(1.0)
    assert s["effective_experts"] == pytest.approx(64.0)
    assert s["top1_mass"] == pytest.approx(1 / 64)
    assert s["top6_mass"] == pytest.approx(6 / 64)

    onehot = np.zeros(N_BANKS)
    onehot[5] = 1.0
    s = entropy_stats(onehot)
    assert s["H_norm"] == pytest.approx(0.0)
    assert s["effective_experts"] == pytest.approx(1.0)
    assert s["top1_mass"] == pytest.approx(1.0)

    two = np.zeros(N_BANKS)
    two[3] = two[7] = 0.5
    s = entropy_stats(two)
    assert s["H_norm"] == pytest.approx(math.log(2) / math.log(64))  # = 1/6
    assert s["effective_experts"] == pytest.approx(2.0)
    assert s["top1_mass"] == pytest.approx(0.5)
    assert s["top3_mass"] == pytest.approx(1.0)


def test_jaccard_and_spearman():
    assert jaccard({1, 2, 3}, {1, 2, 3}) == pytest.approx(1.0)
    assert jaccard({1, 2, 3}, {4, 5, 6}) == pytest.approx(0.0)
    assert jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)
    assert jaccard(set(), set()) == pytest.approx(1.0)
    x = np.array([1.0, 5.0, 3.0, 2.0])
    assert spearman(x, x) == pytest.approx(1.0)
    assert spearman(x, np.exp(x)) == pytest.approx(1.0)  # rank-invariant
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_top_enriched_respects_floor():
    q = np.zeros(N_BANKS)
    q[1], q[2], q[3] = 0.5, 0.3, 0.2
    baseline = np.full(N_BANKS, 1.0 / N_BANKS)
    mask = np.ones(N_BANKS, bool)
    top = top_enriched(q, baseline, mask, k=3)
    assert top == [1, 2, 3]  # descending R order
    mask[2] = False  # starved bank excluded even though maximally enriched
    top = top_enriched(q, baseline, mask, k=3)
    assert 2 not in top and top[:2] == [1, 3]


# --------------------------------------------------------------------------- #
# Planted structure: null, bootstrap, full report                              #
# --------------------------------------------------------------------------- #


def test_observed_jsd_beats_shuffled_null(tmp_path):
    dump, masses = load_planted(tmp_path)
    ex = per_example_masses(masses.m_act, dump)
    qi, qt = observed_q(ex)
    obs = jsd(qi, qt)
    null = shuffled_label_jsds(ex, n_shuffles=200, seed=0)
    assert obs > 0.9  # nearly disjoint supports
    assert obs > np.nanmax(null)
    assert np.nanmean(null) < 0.1


def test_bootstrap_deterministic_and_ci_brackets_observed(tmp_path):
    dump, masses = load_planted(tmp_path)
    ex = per_example_masses(masses.m_act, dump)
    stats = uniform_stats()
    mask = np.ones(N_BANKS, bool)
    a = bootstrap_analysis(ex, stats.p_gate["image"], mask, n_boot=100, seed=5)
    b = bootstrap_analysis(ex, stats.p_gate["image"], mask, n_boot=100, seed=5)
    assert a == b  # same seed, same replicates
    assert a["jsd_ci_low"] <= a["jsd_observed"] <= a["jsd_ci_high"]
    freq = np.asarray(a["selection_frequency"])
    assert all(freq[list(VIS_BANKS)] == 1.0)  # planted banks always selected


def test_stability_analysis_full_report(tmp_path):
    dump, masses = load_planted(tmp_path)
    stats = uniform_stats()
    qi, _ = observed_q(per_example_masses(masses.m_act, dump))
    rep = stability_analysis(
        dump, masses, stats, load_floor=500_000, seed=0,
        n_shuffles=200, n_boot=200,
        compare_example_masses=per_example_masses(masses.m_act, dump),
        seed_q_image={"seed2": qi},
    )
    sep = rep["separation"]
    assert sep["jsd_observed"] > sep["shuffled_null_p95"]
    assert sep["shuffled_null_p_value"] < 0.01
    assert sep["jsd_ci_low"] <= sep["jsd_observed"] <= sep["jsd_ci_high"]
    assert sep["gate_mass_jsd"] == pytest.approx(0.0)  # uniform baselines
    # compare hook fed with the same dump reproduces the observed JSD
    assert sep["compare_dump"]["jsd_observed"] == pytest.approx(sep["jsd_observed"])

    # §16.2 concentration: image mass concentrated on 3 banks
    conc = rep["concentration"]["image"]
    assert conc["top3_mass"] > 0.95
    assert conc["effective_experts"] < 4.0
    assert 0.0 <= conc["H_norm"] <= 1.0

    # §16.3 top-6 recovery: planted banks in every set, ranked first
    sets = rep["top_enriched"]
    for name in ["all", "half:0", "half:1", "seed:seed2"] + [
        k for k in sets if k.startswith("domain:")
    ]:
        assert set(VIS_BANKS) <= set(sets[name]), name
        assert set(sets[name][:3]) == set(VIS_BANKS), name
    assert len([k for k in sets if k.startswith("domain:")]) == 4
    # identical enrichment vectors -> Jaccard 1 and Spearman 1 for all|seed:seed2
    assert rep["jaccard"]["all|seed:seed2"] == pytest.approx(1.0)
    assert rep["spearman"]["all|seed:seed2"] == pytest.approx(1.0)
    # halves share at least the 3 planted banks of their top-6
    assert rep["jaccard"]["half:0|half:1"] >= 1 / 3
    for v in rep["spearman"].values():
        assert np.isfinite(v) and -1.0 <= v <= 1.0
    freq = np.asarray(rep["selection_frequency"])
    assert all(freq[list(VIS_BANKS)] == 1.0)


def test_stability_analysis_load_floor_excludes_starved_bank(tmp_path):
    dump, masses = load_planted(tmp_path)
    stats = uniform_stats(starved=(2,))  # starve a planted visual bank
    rep = stability_analysis(
        dump, masses, stats, load_floor=500_000, seed=0, n_shuffles=20, n_boot=50
    )
    assert rep["floor_mask"][2] is False
    for name, banks in rep["top_enriched"].items():
        assert 2 not in banks, name
    assert rep["selection_frequency"][2] == 0.0
    # the remaining planted banks are still recovered
    assert {1, 3} <= set(rep["top_enriched"]["all"])
