"""Reconstruction evaluation (spec §13): metrics against a naive per-token
reference, FeatureDump contents against the contract, split filtering, global
conditions, real tiny SAECondition end-to-end runs (regression: dictionary
sizing must use f_dict, not the per-bank-equivalent f), dump meta.json, script
wiring (run_eval / run_intervention), and the A-vs-C sanity gate (spec §20
step 5)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from moe_sae.capture.writer import ShardWriter, iter_shards
from moe_sae.eval.reconstruction import (
    ThresholdedEncoding,
    evaluate_reconstruction,
    iter_dump_shards,
    split_mask,
)
from moe_sae.eval.sanity_gate import compare_sanity

REPO = Path(__file__).resolve().parent.parent

BANKS = 8
F = 8
D = 4
K = 6
THETA = 0.8


class FakeRoutedSAE:
    """Tiny router-tied SAE implementing the ThresholdedSAE protocol with
    explicit per-token loops (the trusted-by-inspection reference encoder)."""

    condition = "A"

    def __init__(self, num_banks=BANKS, f=F, d=D, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.num_banks, self.f, self.d = num_banks, f, d
        self.f_dict = f  # router-tied: per-dictionary-bank width == f
        self.W = torch.randn(num_banks, f, d, generator=g)
        self.b = torch.randn(num_banks, f, generator=g) * 0.1
        Dm = torch.randn(num_banks, d, f, generator=g)
        self.D = Dm / Dm.norm(dim=1, keepdim=True)
        self.mu = torch.randn(num_banks, d, generator=g) * 0.5

    def encode_thresholded(self, c, experts, example_id, theta_star):
        c = c.to(torch.float32)
        n, k, d = c.shape
        toks, slots, banks, lats, vals = [], [], [], [], []
        recon = torch.zeros(n, k, d)
        mus = torch.zeros(n, k, d)
        for t in range(n):
            for s in range(k):
                e = int(experts[t, s])
                a = torch.relu(self.W[e] @ (c[t, s] - self.mu[e]) + self.b[e])
                z = a * (a > theta_star)
                recon[t, s] = self.mu[e] + self.D[e] @ z
                mus[t, s] = self.mu[e]
                for j in z.nonzero(as_tuple=True)[0].tolist():
                    toks.append(t)
                    slots.append(s)
                    banks.append(e)
                    lats.append(j)
                    vals.append(float(z[j]))
        return ThresholdedEncoding(
            act_token=torch.tensor(toks, dtype=torch.int64),
            act_slot=torch.tensor(slots, dtype=torch.int64),
            act_bank=torch.tensor(banks, dtype=torch.int64),
            act_latent=torch.tensor(lats, dtype=torch.int64),
            act_value=torch.tensor(vals, dtype=torch.float32),
            recon_slots=recon,
            mu_slots=mus,
        )

    def decoder_cols(self, bank, latent):
        return self.D[bank, :, latent]


class FakeGlobalSAE:
    """Tiny global SAE (condition B shape): encodes c_route, num_banks=1.

    Honest widths: `f` is the per-bank-equivalent width (as on SAECondition)
    while the actual dictionary is `f_dict = 6f` — latent ids run up to
    f_dict, so anything sized by num_banks·f under-allocates (the original
    finding; the old fake set f = full width, masking the bug)."""

    condition = "B"

    def __init__(self, f=4, d=D, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.num_banks, self.f, self.d = 1, f, d
        self.f_dict = 6 * f  # condition B: compute-matched global width
        self.W = torch.randn(self.f_dict, d, generator=g)
        Dm = torch.randn(d, self.f_dict, generator=g)
        self.D = Dm / Dm.norm(dim=0, keepdim=True)
        self.mu = torch.randn(d, generator=g) * 0.5

    def encode_thresholded(self, c, experts, example_id, theta_star):
        route = c.to(torch.float32).sum(dim=1)  # [n, d]
        a = torch.relu((route - self.mu) @ self.W.T)
        z = a * (a > theta_star)
        tok, lat = z.nonzero(as_tuple=True)
        recon = self.mu + z @ self.D.T
        n = route.shape[0]
        return ThresholdedEncoding(
            act_token=tok.to(torch.int64),
            act_slot=torch.zeros_like(tok),
            act_bank=torch.zeros_like(tok),
            act_latent=lat.to(torch.int64),
            act_value=z[tok, lat].to(torch.float32),
            recon_slots=recon.unsqueeze(1),
            mu_slots=self.mu.expand(n, 1, self.d),
        )

    def decoder_cols(self, bank, latent):
        return self.D[:, latent].T


def build_capture(root, n_per_pool=12, seed=0):
    """Example ids alternate across train/dev/eval/dev split classes under the
    A3 keyed hash (row i has domain i % 4; its id is searched to land in the
    wanted bucket range for that domain)."""
    from split_test_utils import ids_in_split

    g = torch.Generator().manual_seed(seed)
    split_cycle = ("train", "dev", "eval", "dev")
    for pool, base in (("visual", 0), ("text", 1000)):
        w = ShardWriter(root, pool, shard_size=5)
        n = n_per_pool
        experts = torch.stack([torch.randperm(BANKS, generator=g)[:K] for _ in range(n)])
        c = torch.randn(n, K, D, generator=g)
        gates = torch.rand(n, K, generator=g)
        raw_ids, cursor = [], base
        for i in range(n):
            eid = ids_in_split(i % 4, split_cycle[i % 4], 1, start=cursor)[0]
            raw_ids.append(eid)
            cursor = eid + 1
        ids = torch.tensor(raw_ids, dtype=torch.int32)
        w.add(c, experts, gates, ids, (torch.arange(n) % 4).to(torch.uint8))
        w.close()


def naive_metrics(root, split, sae, theta):
    """Plain-loop float64 reference over the stored shards."""
    rows = []  # (pool, c f32 [K,D], experts [K])
    for pool in ("visual", "text"):
        for shard in iter_shards(root, pool):
            keep = split_mask(shard["domain"], shard["example_id"], split)
            for t in keep.nonzero(as_tuple=True)[0].tolist():
                rows.append((pool, shard["c"][t], shard["experts"][t].to(torch.int64)))
    n_total = len(rows)
    route_mean = torch.stack(
        [c.to(torch.float64).sum(dim=0) for _, c, _ in rows]
    ).mean(dim=0)

    num_l = den_l = num_a = den_a = 0.0
    num_e = torch.zeros(64, dtype=torch.float64)
    den_e = torch.zeros(64, dtype=torch.float64)
    l0_sum = {"visual": 0, "text": 0}
    n_pool = {"visual": 0, "text": 0}
    fired = set()
    for pool, c, experts in rows:
        enc = sae.encode_thresholded(
            c.unsqueeze(0), experts.unsqueeze(0), torch.zeros(1, dtype=torch.int64), theta
        )
        c64 = c.to(torch.float64)
        recon = enc.recon_slots[0].to(torch.float64)
        mu = enc.mu_slots[0].to(torch.float64)
        target = c64 if sae.num_banks > 1 else c64.sum(dim=0, keepdim=True)
        for s in range(target.shape[0]):
            err = float(((target[s] - recon[s]) ** 2).sum())
            base = float(((target[s] - mu[s]) ** 2).sum())
            num_l += err
            den_l += base
            if sae.num_banks > 1:
                e = int(experts[s])
                num_e[e] += err
                den_e[e] += base
        route = c64.sum(dim=0)
        num_a += float(((route - recon.sum(dim=0)) ** 2).sum())
        den_a += float(((route - route_mean) ** 2).sum())
        l0_sum[pool] += enc.act_token.shape[0]
        n_pool[pool] += 1
        for b, l in zip(enc.act_bank.tolist(), enc.act_latent.tolist()):
            fired.add(b * sae.f_dict + l)

    covered = [e for e in range(64) if den_e[e] > 0]
    return {
        "local_micro_nmse": num_l / den_l,
        "local_macro_nmse": (
            sum(float(num_e[e] / den_e[e]) for e in covered) / len(covered)
            if sae.num_banks > 1
            else None
        ),
        "aggregate_nmse": num_a / den_a,
        "mean_l0": (l0_sum["visual"] + l0_sum["text"]) / n_total,
        "mean_l0_visual": l0_sum["visual"] / max(n_pool["visual"], 1),
        "dead_pct": 100.0
        * (sae.num_banks * sae.f_dict - len(fired))
        / (sae.num_banks * sae.f_dict),
        "n_total": n_total,
        "n_visual": n_pool["visual"],
    }


@pytest.mark.parametrize("chunk_tokens", [3, 1024])
def test_metrics_match_naive_reference(tmp_path, chunk_tokens):
    build_capture(tmp_path / "cap")
    sae = FakeRoutedSAE()
    metrics = evaluate_reconstruction(
        sae, THETA, tmp_path / "cap", "dev", tmp_path / "dump", chunk_tokens=chunk_tokens
    )
    ref = naive_metrics(tmp_path / "cap", "dev", sae, THETA)

    assert metrics["n_tokens"]["total"] == ref["n_total"]
    assert metrics["n_tokens"]["visual"] == ref["n_visual"]
    assert metrics["local_micro_nmse"] == pytest.approx(ref["local_micro_nmse"], rel=1e-6)
    assert metrics["local_macro_nmse"] == pytest.approx(ref["local_macro_nmse"], rel=1e-6)
    assert metrics["aggregate_nmse"] == pytest.approx(ref["aggregate_nmse"], rel=1e-6)
    assert metrics["explained_variance"] == pytest.approx(1 - ref["aggregate_nmse"], rel=1e-6)
    assert metrics["mean_l0"]["overall"] == pytest.approx(ref["mean_l0"])
    assert metrics["mean_l0"]["visual"] == pytest.approx(ref["mean_l0_visual"])
    assert metrics["dead_feature_pct"] == pytest.approx(ref["dead_pct"])
    # L0 histogram integrates back to the token count and the L0 sum
    for pool in ("visual", "text"):
        hist = metrics["l0_hist"][pool]
        assert sum(hist) == metrics["n_tokens"][pool]
    # firing-frequency histogram covers every latent that ever fired
    fh = metrics["firing_freq_hist"]
    n_fired = metrics["n_latents_total"] - fh["never_fired"]
    assert sum(fh["counts"]) == n_fired


def test_dump_contents_match_encoding(tmp_path):
    build_capture(tmp_path / "cap")
    sae = FakeRoutedSAE()
    evaluate_reconstruction(sae, THETA, tmp_path / "cap", "dev", tmp_path / "dump")

    shards = list(iter_dump_shards(tmp_path / "dump"))
    assert shards, "no dump shards written"
    # capture shard order: visual shards then text shards, dev rows only
    kept = []
    for pool in ("visual", "text"):
        for shard in iter_shards(tmp_path / "cap", pool):
            keep = split_mask(shard["domain"], shard["example_id"], "dev")
            if keep.any():
                kept.append((pool, {k: v[keep] for k, v in shard.items()}))
    assert len(shards) == len(kept)

    for dump, (pool, src) in zip(shards, kept):
        # contract dtypes
        assert dump["act_token"].dtype == torch.int64
        assert dump["act_slot"].dtype == torch.uint8
        assert dump["act_latent"].dtype == torch.int32
        assert dump["act_value"].dtype == torch.float32
        assert dump["l0"].dtype == torch.int32
        assert dump["energy"].dtype == torch.float32 and dump["energy"].shape[1] == 6
        assert torch.equal(dump["experts"], src["experts"])
        assert torch.equal(dump["example_id"], src["example_id"])
        assert torch.equal(dump["domain"], src["domain"])
        assert (dump["token_type"] == (1 if pool == "visual" else 0)).all()

        enc = sae.encode_thresholded(
            src["c"],
            src["experts"].to(torch.int64),
            src["example_id"].to(torch.int64),
            THETA,
        )
        assert torch.equal(dump["act_token"], enc.act_token)
        assert torch.equal(dump["act_slot"].to(torch.int64), enc.act_slot)
        assert torch.equal(dump["act_latent"].to(torch.int64), enc.act_latent)
        assert torch.equal(dump["act_bank"].to(torch.int64), enc.act_bank)
        assert torch.allclose(dump["act_value"], enc.act_value)
        # bank identity: for condition A, bank == experts[token, slot]
        banks_via_experts = src["experts"].to(torch.int64)[enc.act_token, enc.act_slot]
        assert torch.equal(enc.act_bank, banks_via_experts)
        # l0 = per-token activation count
        n = src["c"].shape[0]
        assert torch.equal(
            dump["l0"].to(torch.int64), torch.bincount(enc.act_token, minlength=n)
        )
        # energy = ‖D_e z_te‖² = ‖ĉ_te − μ_e‖² per slot
        expected_energy = ((enc.recon_slots - enc.mu_slots) ** 2).sum(dim=-1)
        assert torch.allclose(dump["energy"], expected_energy, atol=1e-5)


def test_split_filtering_is_example_level(tmp_path):
    from moe_sae.data.splits import split_buckets

    build_capture(tmp_path / "cap")
    sae = FakeRoutedSAE()
    for split, (lo, hi) in (("dev", (80, 90)), ("eval", (90, 100))):
        out = tmp_path / f"dump_{split}"
        evaluate_reconstruction(sae, THETA, tmp_path / "cap", split, out)
        for shard in iter_dump_shards(out):
            buckets = split_buckets(shard["domain"], shard["example_id"])
            assert ((buckets >= lo) & (buckets < hi)).all()


def test_global_condition(tmp_path):
    build_capture(tmp_path / "cap")
    sae = FakeGlobalSAE()
    metrics = evaluate_reconstruction(sae, 0.5, tmp_path / "cap", "dev", tmp_path / "dump")
    ref = naive_metrics(tmp_path / "cap", "dev", sae, 0.5)

    assert metrics["local_macro_nmse"] is None
    assert metrics["local_micro_nmse"] == pytest.approx(ref["local_micro_nmse"], rel=1e-6)
    assert metrics["aggregate_nmse"] == pytest.approx(ref["aggregate_nmse"], rel=1e-6)
    # regression: the dictionary is f_dict (6f) wide, not the per-bank f
    assert metrics["n_latents_total"] == sae.f_dict
    assert metrics["dead_feature_pct"] == pytest.approx(ref["dead_pct"])

    saw_wide_latent = False
    for dump in iter_dump_shards(tmp_path / "dump"):
        assert (dump["act_slot"] == 0).all()
        # global conditions: slot 0 carries ‖D z‖²; slots 1..5 are zero
        assert (dump["energy"][:, 1:] == 0).all()
        if dump["act_latent"].numel():
            assert int(dump["act_latent"].max()) < sae.f_dict
            saw_wide_latent |= int(dump["act_latent"].max()) >= sae.f
    # the fixture must actually exercise latent ids beyond the per-bank f,
    # otherwise this test cannot catch num_banks·f sizing again
    assert saw_wide_latent


# --------------------------------------------------------------------------- #
# Real tiny SAEConditions end to end (regression: num_banks·f sizing crashed   #
# global conditions whose dictionary is f_dict = 6f/64f wide)                  #
# --------------------------------------------------------------------------- #


def tiny_real_condition(condition, f=4, k_bar=6, seed=1):
    """A real SAECondition over the tiny capture dims (BANKS banks, d=D)."""
    from moe_sae.sae import build_condition
    from moe_sae.train.config import RunConfig

    g = torch.Generator().manual_seed(seed)
    mu_e = torch.randn(BANKS, D, generator=g) * 0.1
    mu_route = torch.randn(D, generator=g) * 0.1
    cfg = RunConfig(
        run_name=f"tiny_{condition}",
        condition=condition,
        seed=seed,
        f=f,
        k_bar=k_bar,
        total_tokens=0,
        capture_root="unused",
        stats_path="unused",
        pipeline=True,
        compute_dtype="float32",
    )
    return build_condition(condition, cfg, mu_e, mu_route, seed)


@pytest.mark.parametrize("condition", ["A", "B", "F"])
def test_real_tiny_condition_end_to_end(tmp_path, condition):
    """Reviewer repro: condition B with f=4 has f_dict=24; sizing the
    dictionary as num_banks·f made the fire-count bincount crash. Runs a real
    build_condition SAE through evaluate_reconstruction on a tiny capture."""
    build_capture(tmp_path / "cap")
    cond = tiny_real_condition(condition)
    theta = 0.5
    metrics = evaluate_reconstruction(
        cond, theta, tmp_path / "cap", "dev", tmp_path / "dump"
    )
    assert metrics["n_latents_total"] == cond.num_banks * cond.f_dict

    ref = naive_metrics(tmp_path / "cap", "dev", cond, theta)
    assert metrics["local_micro_nmse"] == pytest.approx(ref["local_micro_nmse"], rel=1e-5)
    assert metrics["aggregate_nmse"] == pytest.approx(ref["aggregate_nmse"], rel=1e-5)
    assert metrics["dead_feature_pct"] == pytest.approx(ref["dead_pct"])
    fh = metrics["firing_freq_hist"]
    assert sum(fh["counts"]) == metrics["n_latents_total"] - fh["never_fired"]

    max_latent = -1
    for dump in iter_dump_shards(tmp_path / "dump"):
        if dump["act_latent"].numel():
            max_latent = max(max_latent, int(dump["act_latent"].max()))
    assert max_latent < cond.f_dict
    if condition in ("B", "F"):
        # the fixture must fire latents beyond the per-bank-equivalent f,
        # otherwise it could not catch num_banks·f sizing again
        assert max_latent >= cond.f


def test_dump_meta_json_and_load_dump_roundtrip(tmp_path):
    """FeatureDump producers write meta.json so analysis.load_dump can resolve
    condition/f without explicit overrides (regression: never written)."""
    from moe_sae.analysis.mass import load_dump

    build_capture(tmp_path / "cap")
    sae = FakeGlobalSAE()
    evaluate_reconstruction(sae, 0.5, tmp_path / "cap", "dev", tmp_path / "dump")

    meta = json.loads((tmp_path / "dump" / "meta.json").read_text())
    assert meta == {
        "condition": "B",
        "f": sae.f,
        "f_dict": sae.f_dict,
        "num_banks": 1,
        "split": "dev",
        "theta_star": 0.5,
    }
    dump = load_dump(tmp_path / "dump")  # no condition= override needed now
    assert dump.condition == "B"
    assert dump.f == sae.f


# --------------------------------------------------------------------------- #
# Script wiring (scripts/run_eval.py, scripts/run_intervention.py)             #
# --------------------------------------------------------------------------- #


def _import_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_eval_script_end_to_end(tmp_path):
    """Regression: run_eval called a nonexistent conditions.load_checkpoint;
    also pins --split train (§16.4 probes fit on train) and dump meta.json."""
    build_capture(tmp_path / "cap")
    cond = tiny_real_condition("A")
    run_dir = tmp_path / "run"
    cond.save_checkpoint(
        run_dir / "final" / "checkpoint.safetensors", cond.theta_ema, cond.dead_tracker, step=3
    )
    (run_dir / "final" / "theta_star.json").write_text(json.dumps({"theta_star": 0.5}))

    res = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_eval.py"),
            "--run", str(run_dir),
            "--capture", str(tmp_path / "cap"),
            "--split", "train",
            "--dump", str(tmp_path / "dump"),
            "--device", "cpu",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert res.returncode == 0, res.stderr
    meta = json.loads((tmp_path / "dump" / "meta.json").read_text())
    assert meta["condition"] == "A" and meta["split"] == "train"
    assert meta["theta_star"] == 0.5
    metrics = json.loads((run_dir / "eval_train" / "metrics.json").read_text())
    assert metrics["split"] == "train"
    assert metrics["n_latents_total"] == cond.num_banks * cond.f_dict


def _write_intervention_dump(dump_dir):
    """Minimal FeatureDump: banks 2 and 3 carry visual-enriched latents."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    acts, token_type = [], []
    n = 40
    for t in range(n):
        if t < 20:  # visual tokens
            token_type.append(1)
            acts.append((t, 0, 2, 1, 3.0))
            acts.append((t, 1, 3, 2, 2.0))
        else:  # text tokens
            token_type.append(0)
            acts.append((t, 0, 0, 3, 3.0))
    tok, slot, bank, lat, val = zip(*acts)
    save_file(
        {
            "act_token": torch.tensor(tok, dtype=torch.int64),
            "act_slot": torch.tensor(slot, dtype=torch.uint8),
            "act_bank": torch.tensor(bank, dtype=torch.uint8),
            "act_latent": torch.tensor(lat, dtype=torch.int32),
            "act_value": torch.tensor(val, dtype=torch.float32),
            "token_type": torch.tensor(token_type, dtype=torch.uint8),
            "experts": torch.zeros(n, 6, dtype=torch.uint8),
            "gates": torch.zeros(n, 6, dtype=torch.float32),
            "example_id": torch.arange(n, dtype=torch.int32),
            "domain": torch.zeros(n, dtype=torch.uint8),
            "l0": torch.zeros(n, dtype=torch.int32),
            "energy": torch.zeros(n, 6, dtype=torch.float32),
        },
        str(dump_dir / "shard_00000.safetensors"),
    )


def test_intervention_load_floor_uses_training_loads(tmp_path):
    """Regression: the §16.3 floor must see runs/<run>/final/bank_loads.npy
    (training loads), not the stats-pass counts, when the artifact exists."""
    ri = _import_script("run_intervention")
    _write_intervention_dump(tmp_path / "dump")
    stats = SimpleNamespace(
        p_gate_visual=torch.full((64,), 1.0 / 64.0),
        bank_counts=torch.full((64,), 10**6, dtype=torch.int64),  # all pass
    )
    run_dir = tmp_path / "run"
    (run_dir / "final").mkdir(parents=True)
    loads = np.full(64, 10**6, dtype=np.int64)
    loads[3] = 0  # bank 3 starved in TRAINING, healthy in the stats pass
    np.save(run_dir / "final" / "bank_loads.npy", loads)

    counts = ri.load_training_bank_counts(run_dir, stats)
    assert int(counts[3]) == 0 and int(counts[2]) == 10**6

    kwargs = dict(f=4, top_n_experts=2, quantile=0.5, load_floor=500_000)
    J = ri.select_J_with_training_loads(tmp_path / "dump", stats, counts, **kwargs)
    assert (2, 1) in J
    assert all(b != 3 for b, _ in J)  # floor came from bank_loads.npy

    # sanity: with the training loads all healthy, bank 3 is selected — the
    # array we pass is really what reaches the floor comparison
    healthy = torch.full((64,), 10**6, dtype=torch.int64)
    J2 = ri.select_J_with_training_loads(tmp_path / "dump", stats, healthy, **kwargs)
    assert (3, 2) in J2

    # the adapter must not mutate the caller's stats object
    assert int(stats.bank_counts[3]) == 10**6


def test_intervention_bank_counts_fallback_warns(tmp_path, capsys):
    ri = _import_script("run_intervention")
    stats = SimpleNamespace(
        p_gate_visual=torch.full((64,), 1.0 / 64.0),
        bank_counts=torch.arange(64, dtype=torch.int64),
    )
    counts = ri.load_training_bank_counts(tmp_path / "missing_run", stats)
    assert torch.equal(counts, stats.bank_counts)
    assert "bank_loads.npy" in capsys.readouterr().out


def test_sanity_gate(tmp_path):
    def write_run(name, nmse):
        d = tmp_path / name / "eval"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"local_micro_nmse": nmse}))
        return tmp_path / name

    a = write_run("run_A", 0.08)
    c = write_run("run_C", 0.10)
    out = tmp_path / "reports" / "sanity_gate.json"
    verdict = compare_sanity(a, c, out)
    assert verdict["passed"] is True
    assert json.loads(out.read_text())["passed"] is True

    # boundary: exactly 0.9·NMSE_C passes; above it fails
    a2 = write_run("run_A2", 0.09)
    assert compare_sanity(a2, c, out)["passed"] is True
    a3 = write_run("run_A3", 0.091)
    verdict = compare_sanity(a3, c, out)
    assert verdict["passed"] is False
    assert "implementation audit" in verdict["interpretation"]
    assert "not a hypothesis test" in verdict["interpretation"]
