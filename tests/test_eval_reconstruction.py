"""Reconstruction evaluation (spec §13): metrics against a naive per-token
reference, FeatureDump contents against the contract, split filtering, global
conditions, and the A-vs-C sanity gate (spec §20 step 5)."""

import json

import pytest
import torch

from moe_sae.capture.writer import ShardWriter, iter_shards
from moe_sae.eval.reconstruction import (
    ThresholdedEncoding,
    evaluate_reconstruction,
    iter_dump_shards,
    split_mask,
)
from moe_sae.eval.sanity_gate import compare_sanity

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
    """Tiny global SAE (condition B/F shape): encodes c_route, num_banks=1."""

    condition = "B"

    def __init__(self, f=16, d=D, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.num_banks, self.f, self.d = 1, f, d
        self.W = torch.randn(f, d, generator=g)
        Dm = torch.randn(d, f, generator=g)
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
    """Example ids alternate across train/dev/eval split residues."""
    g = torch.Generator().manual_seed(seed)
    residues = [5, 82, 91, 88]  # train, dev, eval, dev
    for pool, base in (("visual", 0), ("text", 1000)):
        w = ShardWriter(root, pool, shard_size=5)
        n = n_per_pool
        experts = torch.stack([torch.randperm(BANKS, generator=g)[:K] for _ in range(n)])
        c = torch.randn(n, K, D, generator=g)
        gates = torch.rand(n, K, generator=g)
        ids = torch.tensor(
            [base + 100 * i + residues[i % 4] for i in range(n)], dtype=torch.int32
        )
        w.add(c, experts, gates, ids, (torch.arange(n) % 4).to(torch.uint8))
        w.close()


def naive_metrics(root, split, sae, theta):
    """Plain-loop float64 reference over the stored shards."""
    rows = []  # (pool, c f32 [K,D], experts [K])
    for pool in ("visual", "text"):
        for shard in iter_shards(root, pool):
            keep = split_mask(shard["example_id"], split)
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
            fired.add(b * sae.f + l)

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
        "dead_pct": 100.0 * (sae.num_banks * sae.f - len(fired)) / (sae.num_banks * sae.f),
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
            keep = split_mask(shard["example_id"], "dev")
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
    build_capture(tmp_path / "cap")
    sae = FakeRoutedSAE()
    for split, (lo, hi) in (("dev", (80, 90)), ("eval", (90, 100))):
        out = tmp_path / f"dump_{split}"
        evaluate_reconstruction(sae, THETA, tmp_path / "cap", split, out)
        for shard in iter_dump_shards(out):
            residues = shard["example_id"].to(torch.int64) % 100
            assert ((residues >= lo) & (residues < hi)).all()


def test_global_condition(tmp_path):
    build_capture(tmp_path / "cap")
    sae = FakeGlobalSAE()
    metrics = evaluate_reconstruction(sae, 0.5, tmp_path / "cap", "dev", tmp_path / "dump")
    ref = naive_metrics(tmp_path / "cap", "dev", sae, 0.5)

    assert metrics["local_macro_nmse"] is None
    assert metrics["local_micro_nmse"] == pytest.approx(ref["local_micro_nmse"], rel=1e-6)
    assert metrics["aggregate_nmse"] == pytest.approx(ref["aggregate_nmse"], rel=1e-6)
    assert metrics["n_latents_total"] == sae.f

    for dump in iter_dump_shards(tmp_path / "dump"):
        assert (dump["act_slot"] == 0).all()
        # global conditions: slot 0 carries ‖D z‖²; slots 1..5 are zero
        assert (dump["energy"][:, 1:] == 0).all()


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
