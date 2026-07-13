"""Stats pass (spec §20 step 2): exact agreement with a naive float64 reference
on synthetic capture shards, and save/load round-trip."""

import torch

from split_test_utils import ids_in_split

from moe_sae.capture.writer import ShardWriter, iter_shards
from moe_sae.data.splits import split_bucket
from moe_sae.train.stats_pass import compute_stats, load_stats, save_stats

E = 8  # experts (subset of expert-id space; arrays still sized to E here)
D = 4
K = 6  # slots per token (frozen top-6 shape)


def train_ids_mod4(n: int, start: int) -> list[int]:
    """n TRAIN-bucket ids under domain = t % 4 (fixture convention), unique."""
    out, cur = [], start
    for t in range(n):
        while split_bucket(t % 4, cur) >= 80:
            cur += 1
        out.append(cur)
        cur += 1
    return out


def build_capture(root, n_visual=11, n_text=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    for pool, n, id_base in (("visual", n_visual, 0), ("text", n_text, 500)):
        w = ShardWriter(root, pool, shard_size=4)  # force multiple shards
        experts = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(n)])
        c = torch.randn(n, K, D, generator=g)
        gates = torch.rand(n, K, generator=g)
        # TRAIN-bucket ids under the A3 hash (compute_stats filters by default)
        example_id = torch.tensor(train_ids_mod4(n, id_base), dtype=torch.int32)
        domain = (torch.arange(n) % 4).to(torch.uint8)
        w.add(c, experts, gates, example_id, domain)
        w.close()


def naive_reference(root):
    """Plain-loop float64 reference over the *stored* (bf16-rounded) shards,
    under the A11 modality-balanced (50/50) measure: per-pool means are
    combined 0.5/0.5; single-modality banks use the available modality's
    mean."""
    mu_sum = {p: torch.zeros(E, D, dtype=torch.float64) for p in ("visual", "text")}
    counts = {p: torch.zeros(E, dtype=torch.int64) for p in ("visual", "text")}
    gate_sum = {p: torch.zeros(E, dtype=torch.float64) for p in ("visual", "text")}
    route_sum = {p: torch.zeros(D, dtype=torch.float64) for p in ("visual", "text")}
    n_tok = {"visual": 0, "text": 0}
    for pool in ("visual", "text"):
        for shard in iter_shards(root, pool):
            c = shard["c"].to(torch.float64)
            for t in range(c.shape[0]):
                n_tok[pool] += 1
                route_sum[pool] += c[t].sum(dim=0)
                for s in range(K):
                    e = int(shard["experts"][t, s])
                    mu_sum[pool][e] += c[t, s]
                    counts[pool][e] += 1
                    gate_sum[pool][e] += float(shard["gates"][t, s])
    total = counts["visual"] + counts["text"]
    mu_e = torch.zeros(E, D, dtype=torch.float64)
    for e in range(E):
        means = [
            mu_sum[p][e] / int(counts[p][e])
            for p in ("visual", "text")
            if counts[p][e] > 0
        ]
        if means:
            mu_e[e] = sum(means) / len(means)
    route_means = [
        route_sum[p] / n_tok[p] for p in ("visual", "text") if n_tok[p] > 0
    ]
    return {
        "mu_e": mu_e,
        "mu_route": sum(route_means) / len(route_means),
        "counts": counts,
        "total": total,
        "p_route": {p: counts[p].to(torch.float64) / (K * n_tok[p]) for p in counts},
        "p_gate": {p: gate_sum[p] / gate_sum[p].sum() for p in gate_sum},
        "n_tok": n_tok,
    }


def test_stats_match_naive_reference(tmp_path):
    build_capture(tmp_path)
    stats = compute_stats(tmp_path, num_experts=E, hidden_size=D)
    ref = naive_reference(tmp_path)

    assert torch.allclose(stats.mu_e.to(torch.float64), ref["mu_e"], atol=1e-6)
    assert torch.allclose(stats.mu_route.to(torch.float64), ref["mu_route"], atol=1e-6)
    assert torch.equal(stats.bank_counts, ref["total"])
    assert torch.equal(stats.bank_counts_visual, ref["counts"]["visual"])
    assert torch.equal(stats.bank_counts_text, ref["counts"]["text"])
    for pool, p_route, p_gate, n in (
        ("visual", stats.p_route_visual, stats.p_gate_visual, stats.tokens_visual),
        ("text", stats.p_route_text, stats.p_gate_text, stats.tokens_text),
    ):
        assert n == ref["n_tok"][pool]
        assert torch.allclose(p_route.to(torch.float64), ref["p_route"][pool], atol=1e-7)
        assert torch.allclose(p_gate.to(torch.float64), ref["p_gate"][pool], atol=1e-7)
        assert abs(float(p_route.sum()) - 1.0) < 1e-5
        assert abs(float(p_gate.sum()) - 1.0) < 1e-5


def test_never_routed_bank_is_zero(tmp_path):
    # All tokens route to experts 0..5 only; experts 6, 7 are never routed.
    w = ShardWriter(tmp_path, "visual", shard_size=8)
    n = 5
    experts = torch.arange(K).unsqueeze(0).repeat(n, 1)
    w.add(
        torch.randn(n, K, D),
        experts,
        torch.rand(n, K),
        torch.tensor(ids_in_split(0, "train", n), dtype=torch.int32),
        torch.zeros(n, dtype=torch.uint8),
    )
    w.close()
    w2 = ShardWriter(tmp_path, "text", shard_size=8)
    w2.add(
        torch.randn(2, K, D),
        experts[:2],
        torch.rand(2, K),
        torch.tensor(ids_in_split(0, "train", 2, start=1000), dtype=torch.int32),
        torch.zeros(2, dtype=torch.uint8),
    )
    w2.close()

    stats = compute_stats(tmp_path, num_experts=E, hidden_size=D)
    assert int(stats.bank_counts[6]) == 0 and int(stats.bank_counts[7]) == 0
    assert torch.equal(stats.mu_e[6], torch.zeros(D))
    assert float(stats.p_route_visual[7]) == 0.0
    assert float(stats.p_gate_visual[7]) == 0.0


def test_balanced_measure_exact_hand_computed(tmp_path):
    """A11: exact hand-computed balanced means. Visual: two tokens with c ≡ 1
    and c ≡ 5 (per-slot); text: one token with c ≡ 2. Balanced bank mean for
    a bank seen in both pools = 0.5·3 + 0.5·2 = 2.5, NOT the raw-capture
    (1+5+2)/3 ≈ 2.667; single-modality banks use their own pool's mean."""
    ids = ids_in_split(0, "train", 3)
    wv = ShardWriter(tmp_path, "visual", shard_size=8)
    experts_v = torch.tensor([[0, 1, 2, 3, 4, 6], [0, 1, 2, 3, 4, 5]], dtype=torch.uint8)
    c_v = torch.stack([torch.full((K, D), 1.0), torch.full((K, D), 5.0)])
    wv.add(c_v, experts_v, torch.full((2, K), 0.1),
           torch.tensor(ids[:2], dtype=torch.int32), torch.zeros(2, dtype=torch.uint8))
    wv.close()
    wt = ShardWriter(tmp_path, "text", shard_size=8)
    experts_t = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.uint8)
    wt.add(torch.full((1, K, D), 2.0), experts_t, torch.full((1, K), 0.1),
           torch.tensor(ids[2:], dtype=torch.int32), torch.zeros(1, dtype=torch.uint8))
    wt.close()

    stats = compute_stats(tmp_path, num_experts=E, hidden_size=D)
    assert stats.measure == "balanced_train_split"
    # banks 0-4: both pools -> 0.5·((1+5)/2) + 0.5·2 = 2.5
    for e in range(5):
        assert torch.allclose(stats.mu_e[e], torch.full((D,), 2.5)), e
    # bank 5: visual mean 5 (one token), text mean 2 -> 3.5
    assert torch.allclose(stats.mu_e[5], torch.full((D,), 3.5))
    # bank 6: visual-only -> that modality's mean (1.0)
    assert torch.allclose(stats.mu_e[6], torch.full((D,), 1.0))
    # bank 7: never routed -> 0
    assert torch.equal(stats.mu_e[7], torch.zeros(D))
    # μ_route: visual token routes sum to 6·1=6 and 6·5=30 (mean 18); text 12
    # -> balanced 0.5·18 + 0.5·12 = 15 (raw-capture mean would be 16)
    assert torch.allclose(stats.mu_route, torch.full((D,), 15.0))


def test_domain_counts(tmp_path):
    build_capture(tmp_path, n_visual=11, n_text=7)
    stats = compute_stats(tmp_path, num_experts=E, hidden_size=D)
    # domains cycle 0,1,2,3 over 11 visual tokens: 3,3,3,2
    assert stats.domain_counts["visual"] == {"coco": 3, "textvqa": 3, "chartqa": 3, "ai2d": 2}
    assert stats.domain_counts["text"] == {"coco": 2, "textvqa": 2, "chartqa": 2, "ai2d": 1}


def mixed_ids(start: int) -> list[int]:
    """4 ids for the fixture's domain = t % 4 convention: domains 0/1 land in
    train, domain 2 in dev, domain 3 in eval (under the A3 keyed hash)."""
    return [
        ids_in_split(0, "train", 1, start=start)[0],
        ids_in_split(1, "train", 1, start=start)[0],
        ids_in_split(2, "dev", 1, start=start)[0],
        ids_in_split(3, "eval", 1, start=start)[0],
    ]


def build_mixed_capture(root, seed=3):
    """Both pools contain train, dev, and eval buckets under the A3 hash."""
    g = torch.Generator().manual_seed(seed)
    ids = {
        "visual": mixed_ids(250_000_000),
        "text": mixed_ids(300_000_000),
    }
    rows = {}
    for pool, n in (("visual", 12), ("text", 8)):
        w = ShardWriter(root, pool, shard_size=4)
        experts = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(n)])
        c = torch.randn(n, K, D, generator=g)
        gates = torch.rand(n, K, generator=g)
        eid = torch.tensor([ids[pool][t % 4] for t in range(n)], dtype=torch.int32)
        domain = (torch.arange(n) % 4).to(torch.uint8)
        w.add(c, experts, gates, eid, domain)
        w.close()
        rows[pool] = eid
    return rows


def test_split_filter_matches_train_only_reference(tmp_path):
    # Regression (blocker): μ_e / baselines must be train-bucket statistics.
    build_mixed_capture(tmp_path / "mixed")
    # reference capture holding ONLY the train-bucket rows
    g = torch.Generator().manual_seed(3)
    ids = {
        "visual": mixed_ids(250_000_000),
        "text": mixed_ids(300_000_000),
    }
    for pool, n in (("visual", 12), ("text", 8)):
        w = ShardWriter(tmp_path / "trainonly", pool, shard_size=4)
        experts = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(n)])
        c = torch.randn(n, K, D, generator=g)
        gates = torch.rand(n, K, generator=g)
        eid = torch.tensor([ids[pool][t % 4] for t in range(n)], dtype=torch.int32)
        domain = (torch.arange(n) % 4).to(torch.uint8)
        keep = torch.tensor(
            [split_bucket(int(d), int(e)) < 80 for d, e in zip(domain, eid)]
        )
        w.add(c[keep], experts[keep], gates[keep], eid[keep], domain[keep])
        w.close()

    stats = compute_stats(tmp_path / "mixed", num_experts=E, hidden_size=D)
    ref = compute_stats(
        tmp_path / "trainonly", num_experts=E, hidden_size=D, split=None
    )
    assert torch.allclose(stats.mu_e, ref.mu_e)
    assert torch.allclose(stats.mu_route, ref.mu_route)
    assert torch.equal(stats.bank_counts, ref.bank_counts)
    assert torch.allclose(stats.p_route_visual, ref.p_route_visual)
    assert torch.allclose(stats.p_gate_text, ref.p_gate_text)
    # kept/excluded accounting: 12 visual rows -> 6 train / 6 dev+eval, etc.
    assert stats.tokens_visual == 6 and stats.excluded_visual == 6
    assert stats.tokens_text == 4 and stats.excluded_text == 4
    assert stats.split == "train"


def test_split_none_keeps_all_rows(tmp_path):
    build_mixed_capture(tmp_path)
    stats = compute_stats(tmp_path, num_experts=E, hidden_size=D, split=None)
    ref = naive_reference(tmp_path)  # naive loop has no filter
    assert torch.allclose(stats.mu_e.to(torch.float64), ref["mu_e"], atol=1e-6)
    assert stats.tokens_visual == 12 and stats.tokens_text == 8
    assert stats.excluded_visual == 0 and stats.excluded_text == 0
    assert stats.split is None


def test_split_counts_in_summary_json(tmp_path):
    import json

    build_mixed_capture(tmp_path / "cap")
    stats = compute_stats(tmp_path / "cap", num_experts=E, hidden_size=D)
    save_stats(stats, tmp_path / "stats")
    summary = json.loads((tmp_path / "stats" / "stats.json").read_text())
    assert summary["split"] == "train"
    assert summary["measure"] == "balanced_train_split"  # A11 record
    assert summary["tokens"]["visual"] == 6
    assert summary["tokens"]["excluded_visual"] == 6
    assert summary["tokens"]["excluded_text"] == 4
    loaded = load_stats(tmp_path / "stats")
    assert loaded.split == "train"
    assert loaded.excluded_visual == 6 and loaded.excluded_text == 4


def test_save_load_roundtrip(tmp_path):
    build_capture(tmp_path / "cap")
    stats = compute_stats(tmp_path / "cap", num_experts=E, hidden_size=D)
    save_stats(stats, tmp_path / "stats")
    assert (tmp_path / "stats" / "stats.safetensors").exists()
    assert (tmp_path / "stats" / "stats.json").exists()

    loaded = load_stats(tmp_path / "stats")
    assert torch.equal(loaded.mu_e, stats.mu_e)
    assert torch.equal(loaded.mu_route, stats.mu_route)
    assert torch.equal(loaded.bank_counts, stats.bank_counts)
    assert torch.equal(loaded.p_gate_visual, stats.p_gate_visual)
    assert torch.equal(loaded.p_route_text, stats.p_route_text)
    assert loaded.tokens_visual == stats.tokens_visual
    assert loaded.domain_counts == stats.domain_counts
