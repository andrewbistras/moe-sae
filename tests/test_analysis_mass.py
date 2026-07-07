"""Spec §14 masses over FeatureDump artifacts: exact-value checks against a
naive reference, planted-structure separation, shard-offset handling, global
conditions, example-level aggregation, and the run_analysis CLI end-to-end."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from moe_sae.analysis.mass import (
    FeatureDump,
    load_dump,
    per_example_masses,
    q_by_domain,
    q_by_modality,
    q_from_mass,
    token_bank_masses,
)

N_BANKS = 64
TOP_K = 6
VIS_BANKS = (1, 2, 3)
TXT_BANKS = (10, 11)


# --------------------------------------------------------------------------- #
# Contract-format synthetic dumps (docs/INTERFACES.md §FeatureDump)            #
# --------------------------------------------------------------------------- #


def write_synth_dump(
    root,
    condition="A",
    f=16,
    n_examples=120,
    tokens_per_side=6,
    seed=0,
    n_shards=2,
    signal=5.0,
    noise=0.05,
    vis_banks=VIS_BANKS,
    txt_banks=TXT_BANKS,
):
    """Planted structure: visual tokens fire high-mass latents in vis_banks,
    text tokens in txt_banks, plus one tiny noise activation per token."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pt = {k: [] for k in ("experts", "gates", "token_type", "example_id", "domain", "l0", "energy")}
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
                en = np.zeros(TOP_K)
                n_acts = 0
                for b in banks:
                    slot = int(np.argmax(e == b))
                    val = float(signal * (0.8 + 0.4 * rng.random()))
                    ac["act_token"].append(t)
                    ac["act_slot"].append(slot)
                    ac["act_latent"].append(int(rng.integers(f)))
                    ac["act_value"].append(val)
                    en[slot] += val * val
                    n_acts += 1
                slot = int(rng.integers(TOP_K))
                val = float(noise * (0.5 + rng.random()))
                ac["act_token"].append(t)
                ac["act_slot"].append(slot)
                ac["act_latent"].append(int(rng.integers(f)))
                ac["act_value"].append(val)
                en[slot] += val * val
                n_acts += 1
                pt["experts"].append(e)
                pt["gates"].append(rng.uniform(0.05, 0.3, TOP_K))
                pt["token_type"].append(tt)
                pt["example_id"].append(ex)
                pt["domain"].append(ex % 4)
                pt["l0"].append(n_acts)
                pt["energy"].append(en)
                t += 1
    _write_shards(root, condition, f, pt, ac, t, n_shards)
    return root


def write_synth_global_dump(
    root, condition="B", f=16, n_examples=120, tokens_per_side=6, seed=0, n_shards=2
):
    """Global-condition dump: single dictionary, act_slot == 0; visual tokens
    fire latents [0, 8), text tokens fire latents [F-8, F)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    f_total = TOP_K * f if condition == "B" else N_BANKS * f
    rng = np.random.default_rng(seed)
    pt = {k: [] for k in ("experts", "gates", "token_type", "example_id", "domain", "l0")}
    ac = {k: [] for k in ("act_token", "act_slot", "act_latent", "act_value")}
    t = 0
    for ex in range(n_examples):
        for tt in (1, 0):
            lat_lo = 0 if tt else f_total - 8
            e_lo = 0 if tt else 8
            for _ in range(tokens_per_side):
                for _ in range(3):
                    ac["act_token"].append(t)
                    ac["act_slot"].append(0)
                    ac["act_latent"].append(int(lat_lo + rng.integers(8)))
                    ac["act_value"].append(float(4.0 + rng.random()))
                pt["experts"].append(e_lo + rng.choice(8, TOP_K, replace=False))
                pt["gates"].append(rng.uniform(0.05, 0.3, TOP_K))
                pt["token_type"].append(tt)
                pt["example_id"].append(ex)
                pt["domain"].append(ex % 4)
                pt["l0"].append(3)
                t += 1
    _write_shards(root, condition, f, pt, ac, t, n_shards)
    return root


def _write_shards(root, condition, f, pt, ac, n_tokens, n_shards):
    experts = np.asarray(pt["experts"], np.uint8)
    gates = np.asarray(pt["gates"], np.float32)
    token_type = np.asarray(pt["token_type"], np.uint8)
    example_id = np.asarray(pt["example_id"], np.int32)
    domain = np.asarray(pt["domain"], np.uint8)
    l0 = np.asarray(pt["l0"], np.int32)
    energy = np.asarray(pt["energy"], np.float32) if "energy" in pt else None
    act_token = np.asarray(ac["act_token"], np.int64)
    act_slot = np.asarray(ac["act_slot"], np.uint8)
    act_latent = np.asarray(ac["act_latent"], np.int32)
    act_value = np.asarray(ac["act_value"], np.float32)
    bounds = np.linspace(0, n_tokens, n_shards + 1).astype(int)
    for i in range(n_shards):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        sel = (act_token >= lo) & (act_token < hi)
        shard = {
            "experts": torch.from_numpy(experts[lo:hi]),
            "gates": torch.from_numpy(gates[lo:hi]),
            "token_type": torch.from_numpy(token_type[lo:hi]),
            "example_id": torch.from_numpy(example_id[lo:hi]),
            "domain": torch.from_numpy(domain[lo:hi]),
            "l0": torch.from_numpy(l0[lo:hi]),
            "act_token": torch.from_numpy(act_token[sel] - lo),
            "act_slot": torch.from_numpy(act_slot[sel]),
            "act_latent": torch.from_numpy(act_latent[sel]),
            "act_value": torch.from_numpy(act_value[sel]),
        }
        if energy is not None:
            shard["energy"] = torch.from_numpy(energy[lo:hi])
        save_file(shard, str(root / f"shard_{i:05d}.safetensors"))
    (root / "meta.json").write_text(json.dumps({"condition": condition, "f": f}))


def write_stats(dir_path, starved=(), count=10_000_000):
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    uni = torch.full((N_BANKS,), 1.0 / N_BANKS, dtype=torch.float32)
    counts = torch.full((N_BANKS,), count, dtype=torch.int64)
    for b in starved:
        counts[b] = 100
    # Producer naming (train/stats_pass.py): the image modality is "visual".
    save_file(
        {
            "p_route_visual": uni.clone(),
            "p_route_text": uni.clone(),
            "p_gate_visual": uni.clone(),
            "p_gate_text": uni.clone(),
            "bank_counts": counts,
        },
        str(d / "stats.safetensors"),
    )
    return d


# --------------------------------------------------------------------------- #
# Exact-value checks against a naive reference                                 #
# --------------------------------------------------------------------------- #


def make_tiny_dump():
    """Two tokens with fully hand-specified activations."""
    return FeatureDump(
        condition="A",
        f=16,
        experts=np.array([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]], np.int64),
        gates=np.full((2, 6), 0.1, np.float32),
        token_type=np.array([1, 0], np.uint8),
        example_id=np.array([0, 0], np.int64),
        domain=np.array([0, 0], np.uint8),
        l0=np.array([3, 1], np.int64),
        act_token=np.array([0, 0, 0, 1], np.int64),
        act_slot=np.array([0, 0, 2, 1], np.int64),
        act_latent=np.array([3, 5, 0, 7], np.int64),
        act_value=np.array([2.0, 1.0, 0.5, 4.0], np.float64),
        # ‖D_e z‖² per slot: token 0 slot 0 (banks share latents 3,5), slot 2;
        # token 1 slot 1. Values hand-fixed.
        energy=np.array(
            [[5.0, 0, 0.25, 0, 0, 0], [0, 16.0, 0, 0, 0, 0]], np.float64
        ),
    )


def naive_masses(dump):
    n = dump.n_tokens
    mc = np.zeros((n, N_BANKS))
    ma = np.zeros((n, N_BANKS))
    for tk, sl, val in zip(dump.act_token, dump.act_slot, dump.act_value):
        b = dump.experts[tk, sl]
        ma[tk, b] += val
        if val > 0:
            mc[tk, b] += 1
    me = None
    if dump.energy is not None:
        me = np.zeros((n, N_BANKS))
        for tk in range(n):
            for sl in range(TOP_K):
                me[tk, dump.experts[tk, sl]] += dump.energy[tk, sl]
    return mc, ma, me


def test_masses_exact_tiny():
    dump = make_tiny_dump()
    m = token_bank_masses(dump)
    assert m.m_act[0, 0] == pytest.approx(3.0)
    assert m.m_act[0, 2] == pytest.approx(0.5)
    assert m.m_act[1, 11] == pytest.approx(4.0)
    assert m.m_act.sum() == pytest.approx(7.5)
    assert m.m_count[0, 0] == 2
    assert m.m_count[0, 2] == 1
    assert m.m_count[1, 11] == 1
    assert m.m_count.sum() == 4
    assert m.m_energy[0, 0] == pytest.approx(5.0)
    assert m.m_energy[0, 2] == pytest.approx(0.25)
    assert m.m_energy[1, 11] == pytest.approx(16.0)
    q = q_from_mass(m.m_act)
    assert q[0] == pytest.approx(3.0 / 7.5)
    assert q[2] == pytest.approx(0.5 / 7.5)
    assert q[11] == pytest.approx(4.0 / 7.5)
    assert q.sum() == pytest.approx(1.0)


def test_masses_match_naive_on_synth(tmp_path):
    dump = load_dump(write_synth_dump(tmp_path / "d", n_examples=30, seed=3))
    m = token_bank_masses(dump)
    mc, ma, me = naive_masses(dump)
    np.testing.assert_allclose(m.m_count, mc, rtol=0, atol=0)
    np.testing.assert_allclose(m.m_act, ma, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(m.m_energy, me, rtol=1e-5, atol=1e-6)


def test_q_from_mass_empty_mask_is_none():
    dump = make_tiny_dump()
    m = token_bank_masses(dump)
    assert q_from_mass(m.m_act, np.zeros(2, dtype=bool)) is None


# --------------------------------------------------------------------------- #
# Loading: shard offsets, meta                                                 #
# --------------------------------------------------------------------------- #


def test_multi_shard_equals_single_shard(tmp_path):
    d1 = load_dump(write_synth_dump(tmp_path / "one", n_shards=1, seed=7))
    d3 = load_dump(write_synth_dump(tmp_path / "three", n_shards=3, seed=7))
    assert d1.n_tokens == d3.n_tokens
    np.testing.assert_array_equal(d1.experts, d3.experts)
    np.testing.assert_array_equal(d1.act_token, d3.act_token)
    np.testing.assert_array_equal(d1.act_latent, d3.act_latent)
    np.testing.assert_allclose(d1.act_value, d3.act_value)
    assert d3.act_token.max() < d3.n_tokens
    assert d1.condition == "A" and d1.f == 16


def test_load_dump_requires_condition(tmp_path):
    root = write_synth_dump(tmp_path / "d")
    (root / "meta.json").unlink()
    with pytest.raises(ValueError):
        load_dump(root)
    assert load_dump(root, condition="A", f=16).condition == "A"


# --------------------------------------------------------------------------- #
# Planted structure and global conditions                                      #
# --------------------------------------------------------------------------- #


def test_q_planted_separation(tmp_path):
    dump = load_dump(write_synth_dump(tmp_path / "d"))
    m = token_bank_masses(dump)
    q = q_by_modality(m, dump, "act")
    top3_img = set(np.argsort(-q["image"])[:3].tolist())
    top2_txt = set(np.argsort(-q["text"])[:2].tolist())
    assert top3_img == set(VIS_BANKS)
    assert top2_txt == set(TXT_BANKS)
    assert q["image"][list(VIS_BANKS)].sum() > 0.95
    assert q["text"][list(TXT_BANKS)].sum() > 0.95
    # same structure under count and energy masses (robustness kinds, §14)
    for kind in ("count", "energy"):
        qk = q_by_modality(m, dump, kind)
        assert set(np.argsort(-qk["image"])[:3].tolist()) == set(VIS_BANKS)


def test_q_by_domain(tmp_path):
    dump = load_dump(write_synth_dump(tmp_path / "d"))
    m = token_bank_masses(dump)
    per_dom = q_by_domain(m, dump, "act")
    assert set(per_dom) == {"coco", "textvqa", "chartqa", "ai2d"}
    for dom in per_dom.values():
        assert set(np.argsort(-dom["image"])[:3].tolist()) == set(VIS_BANKS)


def test_global_condition_has_no_bank_masses(tmp_path):
    dump = load_dump(write_synth_global_dump(tmp_path / "g", condition="B"))
    assert not dump.is_banked
    assert token_bank_masses(dump) is None


def test_example_aggregation_consistency(tmp_path):
    dump = load_dump(write_synth_dump(tmp_path / "d", n_examples=40))
    m = token_bank_masses(dump)
    ex = per_example_masses(m.m_act, dump)
    assert ex.example_ids.shape[0] == 40
    vis = dump.token_type == 1
    np.testing.assert_allclose(
        ex.image.sum(axis=0), m.m_act[vis].sum(axis=0, dtype=np.float64), rtol=1e-6
    )
    np.testing.assert_allclose(
        ex.text.sum(axis=0), m.m_act[~vis].sum(axis=0, dtype=np.float64), rtol=1e-6
    )
    # q from the example table equals q over tokens
    q_tok = q_from_mass(m.m_act, vis)
    q_ex = ex.image.sum(axis=0) / ex.image.sum()
    np.testing.assert_allclose(q_ex, q_tok, rtol=1e-6)


# --------------------------------------------------------------------------- #
# CLI end-to-end                                                               #
# --------------------------------------------------------------------------- #


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_analysis.py"
    spec = importlib.util.spec_from_file_location("run_analysis_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OUTPUT_FILES = ("mass.json", "enrichment.json", "stability.json", "probes.json", "summary.md")


def test_run_analysis_cli_banked(tmp_path):
    dump = write_synth_dump(tmp_path / "dumps" / "runA" / "train", n_examples=110)
    stats = write_stats(tmp_path / "stats")
    out = tmp_path / "reports"
    _load_script().main(
        [
            "--dump", str(dump), "--stats", str(stats), "--out", str(out),
            "--load-floor", "500000", "--n-shuffles", "40", "--n-boot", "40",
            "--compare-dump", str(dump), "--seed-dumps", str(dump),
        ]
    )
    for name in OUTPUT_FILES:
        assert (out / name).exists(), name
    mass = json.loads((out / "mass.json").read_text())
    assert mass["condition"] == "A" and mass["q"]["act"]["image"] is not None
    stab = json.loads((out / "stability.json").read_text())
    sep = stab["separation"]
    assert sep["jsd_observed"] > sep["shuffled_null_p95"]
    assert sep["compare_dump"] is not None
    assert set(VIS_BANKS) <= set(stab["top_enriched"]["all"])
    assert any(k.startswith("seed:") for k in stab["top_enriched"])
    prob = json.loads((out / "probes.json").read_text())
    assert prob["probes"]["feature_identity"]["heldout"]["accuracy"] > 0.9


def test_run_analysis_cli_global(tmp_path):
    dump = write_synth_global_dump(tmp_path / "dumps" / "runB" / "train", n_examples=110)
    stats = write_stats(tmp_path / "stats")
    out = tmp_path / "reports"
    _load_script().main(
        ["--dump", str(dump), "--stats", str(stats), "--out", str(out),
         "--n-shuffles", "10", "--n-boot", "10"]
    )
    for name in OUTPUT_FILES:
        assert (out / name).exists(), name
    mass = json.loads((out / "mass.json").read_text())
    assert mass["condition"] == "B" and mass["q"] is None
    stab = json.loads((out / "stability.json").read_text())
    assert "note" in stab  # §14–§16.3 undefined for global conditions
    prob = json.loads((out / "probes.json").read_text())
    assert "skipped" in prob["probes"]["sae_bank_mass"]
    assert prob["probes"]["feature_identity"]["heldout"]["accuracy"] > 0.9
