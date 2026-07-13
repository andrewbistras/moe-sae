"""Spec §16.4 incremental-modality-information probes: frozen example-level
splits, exact feature construction (incl. condition-C bank permutation and
f_dict widths), the train-dump/score-dump flow with the eval split untouched
by default, planted-structure probe performance, and global-condition
handling."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from split_test_utils import ids_in_split, ids_in_split_domain_mod4

from moe_sae.analysis.mass import FeatureDump, load_dump, token_bank_masses
from moe_sae.analysis.probes import (
    SPLIT_DEV,
    SPLIT_EVAL,
    SPLIT_TRAIN,
    example_split,
    gate_features,
    identity_features,
    run_probes,
    total_features,
)
from moe_sae.data.splits import split_bucket

N_BANKS = 64
TOP_K = 6
VIS_BANKS = (1, 2, 3)
TXT_BANKS = (10, 11)

# Default mixed-split example ids (fixture convention: domain = ex % 4):
# 100 train + 10 dev + 10 eval under the A3 keyed hash.
DEFAULT_EX_IDS = (
    ids_in_split_domain_mod4("train", 100)
    + ids_in_split_domain_mod4("dev", 10)
    + ids_in_split_domain_mod4("eval", 10)
)


# --------------------------------------------------------------------------- #
# Contract-format synthetic dumps                                              #
# --------------------------------------------------------------------------- #


def write_synth_dump(
    root, condition="A", f=16, tokens_per_side=6, seed=0,
    signal=5.0, noise=0.05, ex_ids=None,
):
    """Planted banked dump (visual -> banks 1,2,3; text -> banks 10,11).
    `ex_ids` restricts example ids (default DEFAULT_EX_IDS: 100 train + 10 dev
    + 10 eval under the A3 hash, domain = ex % 4) — used to build single-split
    dumps matching the run_eval per-split layout."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pt = {k: [] for k in ("experts", "gates", "token_type", "example_id", "domain", "l0")}
    ac = {k: [] for k in ("act_token", "act_slot", "act_latent", "act_value")}
    t = 0
    for ex in ex_ids if ex_ids is not None else DEFAULT_EX_IDS:
        for tt in (1, 0):
            banks = VIS_BANKS if tt else TXT_BANKS
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
    _save(root, condition, f, pt, ac)
    return root


def write_synth_global_dump(root, condition="B", f=16, tokens_per_side=6, seed=0):
    """Global dump: act_slot == 0; visual latents [0,8), text latents [F-8,F)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    f_total = TOP_K * f if condition == "B" else N_BANKS * f
    rng = np.random.default_rng(seed)
    pt = {k: [] for k in ("experts", "gates", "token_type", "example_id", "domain", "l0")}
    ac = {k: [] for k in ("act_token", "act_slot", "act_latent", "act_value")}
    t = 0
    for ex in DEFAULT_EX_IDS:
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
    _save(root, condition, f, pt, ac)
    return root


def _save(root, condition, f, pt, ac):
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
    # Producer meta format (run_eval): condition/f/f_dict/num_banks/split/theta_star.
    banked = condition in ("A", "C", "D", "E")
    f_dict = N_BANKS * f if banked or condition == "F" else TOP_K * f
    (root / "meta.json").write_text(
        json.dumps(
            {
                "condition": condition,
                "f": f,
                "f_dict": f_dict,
                "num_banks": N_BANKS if banked else 1,
                "split": "train",
                "theta_star": 0.1,
            }
        )
    )


# --------------------------------------------------------------------------- #
# Frozen split (contract §Splits)                                              #
# --------------------------------------------------------------------------- #


def test_example_split_exact():
    """Split codes follow the frozen A3 bucket thresholds per (domain, id)."""
    ids = np.array(
        ids_in_split(0, "train", 2)
        + ids_in_split(1, "dev", 2)
        + ids_in_split(2, "eval", 2),
        dtype=np.int64,
    )
    doms = np.array([0, 0, 1, 1, 2, 2], dtype=np.uint8)
    expect = np.array(
        [SPLIT_TRAIN, SPLIT_TRAIN, SPLIT_DEV, SPLIT_DEV, SPLIT_EVAL, SPLIT_EVAL]
    )
    np.testing.assert_array_equal(example_split(doms, ids), expect)
    # and against the raw bucket function
    for d, e, s in zip(doms, ids, expect):
        b = split_bucket(int(d), int(e))
        assert (b < 80) == (s == SPLIT_TRAIN)
        assert (80 <= b < 90) == (s == SPLIT_DEV)
        assert (b >= 90) == (s == SPLIT_EVAL)


def test_split_is_example_level(tmp_path):
    dump = load_dump(write_synth_dump(tmp_path / "d"))
    split = example_split(dump.domain, dump.example_id)
    # every example's tokens land in exactly one split
    for ex in np.unique(dump.example_id):
        assert np.unique(split[dump.example_id == ex]).size == 1


# --------------------------------------------------------------------------- #
# Feature construction: exact tiny cases                                       #
# --------------------------------------------------------------------------- #


def make_tiny_dump(condition="A", f=16):
    return FeatureDump(
        condition=condition,
        f=f,
        experts=np.array([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]], np.int64),
        gates=np.array(
            [[0.4, 0.2, 0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]], np.float32
        ),
        token_type=np.array([1, 0], np.uint8),
        example_id=np.array([0, 1], np.int64),
        domain=np.array([0, 1], np.uint8),
        l0=np.array([2, 1], np.int64),
        act_token=np.array([0, 0, 1], np.int64),
        act_slot=np.array([0, 2, 1], np.int64),
        act_latent=np.array([3, 15, 7], np.int64),
        act_value=np.array([2.0, 1.0, 4.0], np.float64),
        energy=None,
    )


def test_gate_features_exact():
    x = gate_features(make_tiny_dump())
    assert x.shape == (2, 64)
    assert x[0, 0] == pytest.approx(0.4)
    assert x[0, 5] == pytest.approx(0.1)
    assert x[0, 10] == 0.0
    assert x[1, 10] == pytest.approx(0.3)
    assert x[0].sum() == pytest.approx(1.0)


def test_identity_features_exact_banked():
    dump = make_tiny_dump()
    x = identity_features(dump)
    assert x.shape == (2, 64 * 16)
    dense = np.asarray(x.todense())
    # (token 0, slot 0 -> bank 0, latent 3) -> col 0*16+3
    assert dense[0, 3] == 1.0
    # (token 0, slot 2 -> bank 2, latent 15) -> col 2*16+15
    assert dense[0, 2 * 16 + 15] == 1.0
    # (token 1, slot 1 -> bank 11, latent 7) -> col 11*16+7
    assert dense[1, 11 * 16 + 7] == 1.0
    assert dense.sum() == 3.0  # binary, nothing else set


def test_identity_features_exact_global():
    dump = make_tiny_dump(condition="B")
    x = identity_features(dump)
    assert x.shape == (2, 6 * 16)  # F_global = 6f for condition B
    dense = np.asarray(x.todense())
    assert dense[0, 3] == 1.0 and dense[0, 15] == 1.0 and dense[1, 7] == 1.0
    assert dense.sum() == 3.0


def test_identity_features_condition_c_uses_act_bank():
    """FINDING regression: condition C's bank is a per-example permutation
    carried in act_bank — two tokens from different examples firing the SAME
    (bank, latent) must share one identity column, even though their host
    experts (experts[token, slot]) differ."""
    f = 16
    dump = FeatureDump(
        condition="C",
        f=f,
        experts=np.array([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]], np.int64),
        gates=np.full((2, 6), 0.1, np.float32),
        token_type=np.array([1, 1], np.uint8),
        example_id=np.array([0, 1], np.int64),
        domain=np.zeros(2, np.uint8),
        l0=np.array([1, 1], np.int64),
        act_token=np.array([0, 1], np.int64),
        act_slot=np.array([0, 3], np.int64),  # host experts 0 and 13 — differ
        act_latent=np.array([7, 7], np.int64),
        act_value=np.array([1.0, 1.0], np.float64),
        energy=None,
        act_bank=np.array([42, 42], np.int64),  # planted permutation: same bank
    )
    dense = np.asarray(identity_features(dump).todense())
    col = 42 * f + 7
    assert dense[0, col] == 1.0 and dense[1, col] == 1.0  # same column
    assert dense.sum() == 2.0
    # the host-expert-derived columns must NOT be set
    assert dense[0, 0 * f + 7] == 0.0
    assert dense[1, 13 * f + 7] == 0.0


def test_identity_features_banked_falls_back_to_host_expert():
    dump = make_tiny_dump()  # act_bank is None -> experts[token, slot]
    assert dump.act_bank is None
    dense = np.asarray(identity_features(dump).todense())
    assert dense[0, 0 * 16 + 3] == 1.0 and dense[1, 11 * 16 + 7] == 1.0


def test_identity_features_global_uses_f_dict():
    dump = make_tiny_dump(condition="B")
    dump.f = None
    dump.f_dict = 6 * 16  # total dictionary width from meta.json
    x = identity_features(dump)
    assert x.shape == (2, 6 * 16)


def test_identity_features_banked_derives_f_from_f_dict():
    dump = make_tiny_dump()
    dump.f = None
    dump.f_dict = 64 * 16  # total width -> per-bank f = 16
    dense = np.asarray(identity_features(dump).todense())
    assert dense.shape == (2, 64 * 16)
    assert dense[1, 11 * 16 + 7] == 1.0


def test_total_features_per_condition():
    assert total_features("A", 4096) == 64 * 4096
    assert total_features("C", 1024) == 64 * 1024
    assert total_features("B", 4096) == 6 * 4096
    assert total_features("F", 4096) == 64 * 4096


def test_identity_features_requires_f():
    dump = make_tiny_dump()
    dump.f = None
    with pytest.raises(ValueError):
        identity_features(dump)


# --------------------------------------------------------------------------- #
# Probes on planted structure                                                  #
# --------------------------------------------------------------------------- #

ALL_PROBES = (
    "router_gates",
    "sae_bank_mass",
    "feature_identity",
    "gates_plus_bank_mass",
    "gates_plus_identity",
)


def test_probes_beat_chance_on_planted(tmp_path):
    dump = load_dump(write_synth_dump(tmp_path / "d"))
    masses = token_bank_masses(dump)
    rep = run_probes(dump, masses, dump, masses, c=1.0, seed=0)

    assert rep["condition"] == "A"
    assert rep["include_eval_split"] is False
    for split in ("train", "dev"):
        assert rep["splits"][split]["n_tokens"] > 0
        assert rep["splits"][split]["frac_visual"] == pytest.approx(0.5, abs=0.05)
    # example-level counts: ids 0..119 -> 100 train, 10 dev examples
    assert rep["splits"]["train"]["n_examples"] == 100
    assert rep["splits"]["dev"]["n_examples"] == 10
    # eval split untouched pre-intervention: no balance report, no scores
    assert "eval" not in rep["splits"]

    for name in ALL_PROBES:
        res = rep["probes"][name]
        assert "skipped" not in res, name
        assert res["dev"]["accuracy"] > 0.95, name
        assert res["dev"]["auc"] > 0.99, name
        assert "skipped" in res["eval"], name
    # L1 identity probe: sparse, nonzero coefficients reported
    ident = rep["probes"]["feature_identity"]
    assert ident["penalty"] == "l1"
    assert 0 < ident["n_nonzero_coef"] <= ident["n_features"]
    assert ident["n_features"] == 64 * 16
    assert rep["probes"]["gates_plus_identity"]["n_features"] == 64 + 64 * 16


def test_probes_separate_train_and_dev_dumps(tmp_path):
    """FINDING regression: run_eval produces single-split dumps — probes must
    fit on a TRAIN-split dump and score on a separate DEV-split dump."""
    train = load_dump(
        write_synth_dump(tmp_path / "train", ex_ids=ids_in_split_domain_mod4("train", 80))
    )
    dev_ids = ids_in_split_domain_mod4("dev", 30)
    dev = load_dump(write_synth_dump(tmp_path / "dev", ex_ids=dev_ids, seed=1))
    rep = run_probes(train, token_bank_masses(train), dev, token_bank_masses(dev))
    assert rep["splits"]["train"]["n_examples"] == 80
    assert rep["splits"]["dev"]["n_examples"] == 30
    for name in ALL_PROBES:
        assert rep["probes"][name]["dev"]["accuracy"] > 0.95, name
    # swapped dumps fail loudly: the dev dump has no train-split tokens
    with pytest.raises(ValueError, match="train-split"):
        run_probes(dev, token_bank_masses(dev), train, token_bank_masses(train))


def test_probes_eval_split_requires_explicit_opt_in(tmp_path):
    """FINDING regression: eval-split scoring (spec §17/§20 post-intervention
    only) is off by default and enabled only via include_eval_split."""
    dump = load_dump(write_synth_dump(tmp_path / "d"))
    masses = token_bank_masses(dump)
    rep = run_probes(dump, masses, dump, masses, include_eval_split=True)
    assert rep["include_eval_split"] is True
    assert rep["splits"]["eval"]["n_examples"] == 10
    for name in ALL_PROBES:
        res = rep["probes"][name]
        assert "skipped" not in res["eval"], name
        assert res["eval"]["accuracy"] > 0.95, name


def test_probes_reject_dev_dump_without_dev_tokens(tmp_path):
    train = load_dump(
        write_synth_dump(tmp_path / "train", ex_ids=ids_in_split_domain_mod4("train", 80))
    )
    m = token_bank_masses(train)
    with pytest.raises(ValueError, match="dev-split"):
        run_probes(train, m, train, m)


def test_probes_reject_condition_mismatch(tmp_path):
    a = load_dump(write_synth_dump(tmp_path / "a"))
    b = load_dump(write_synth_global_dump(tmp_path / "b", condition="B"))
    with pytest.raises(ValueError, match="condition"):
        run_probes(a, token_bank_masses(a), b, None)


def test_probes_deterministic(tmp_path):
    dump = load_dump(write_synth_dump(tmp_path / "d", ex_ids=DEFAULT_EX_IDS[:110]))
    masses = token_bank_masses(dump)
    a = run_probes(dump, masses, dump, masses, seed=3)
    b = run_probes(dump, masses, dump, masses, seed=3)
    assert a == b


def test_probes_global_condition_skips_bank_mass(tmp_path):
    dump = load_dump(write_synth_global_dump(tmp_path / "g", condition="B"))
    assert token_bank_masses(dump) is None
    rep = run_probes(dump, None, dump, None, seed=0)
    assert "skipped" in rep["probes"]["sae_bank_mass"]
    assert "skipped" in rep["probes"]["gates_plus_bank_mass"]
    # probes 1 and 3 apply to all conditions and separate the planted structure
    assert rep["probes"]["router_gates"]["dev"]["accuracy"] > 0.95
    assert rep["probes"]["feature_identity"]["dev"]["accuracy"] > 0.95
    assert rep["probes"]["feature_identity"]["n_features"] == 6 * 16


def test_probes_raise_on_single_class_train():
    dump = make_tiny_dump()
    # no train examples: dev-bucket ids for the dump's domains (0 and 1)
    dump.example_id = np.array(
        [ids_in_split(0, "dev", 1)[0], ids_in_split(1, "dev", 1)[0]], np.int64
    )
    with pytest.raises(ValueError):
        run_probes(dump, token_bank_masses(dump), dump, token_bank_masses(dump))
