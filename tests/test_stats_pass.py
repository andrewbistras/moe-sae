"""Stats pass (spec §20 step 2): exact agreement with a naive float64 reference
on synthetic capture shards, and save/load round-trip."""

import torch

from moe_sae.capture.writer import ShardWriter, iter_shards
from moe_sae.train.stats_pass import compute_stats, load_stats, save_stats

E = 8  # experts (subset of expert-id space; arrays still sized to E here)
D = 4
K = 6  # slots per token (frozen top-6 shape)


def build_capture(root, n_visual=11, n_text=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    for pool, n, id_base in (("visual", n_visual, 0), ("text", n_text, 500)):
        w = ShardWriter(root, pool, shard_size=4)  # force multiple shards
        experts = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(n)])
        c = torch.randn(n, K, D, generator=g)
        gates = torch.rand(n, K, generator=g)
        example_id = torch.arange(n, dtype=torch.int32) + id_base
        domain = (torch.arange(n) % 4).to(torch.uint8)
        w.add(c, experts, gates, example_id, domain)
        w.close()


def naive_reference(root):
    """Plain-loop float64 reference over the *stored* (bf16-rounded) shards."""
    mu_sum = torch.zeros(E, D, dtype=torch.float64)
    counts = {p: torch.zeros(E, dtype=torch.int64) for p in ("visual", "text")}
    gate_sum = {p: torch.zeros(E, dtype=torch.float64) for p in ("visual", "text")}
    route_sum = torch.zeros(D, dtype=torch.float64)
    n_tok = {"visual": 0, "text": 0}
    for pool in ("visual", "text"):
        for shard in iter_shards(root, pool):
            c = shard["c"].to(torch.float64)
            for t in range(c.shape[0]):
                n_tok[pool] += 1
                route_sum += c[t].sum(dim=0)
                for s in range(K):
                    e = int(shard["experts"][t, s])
                    mu_sum[e] += c[t, s]
                    counts[pool][e] += 1
                    gate_sum[pool][e] += float(shard["gates"][t, s])
    total = counts["visual"] + counts["text"]
    mu_e = torch.zeros(E, D, dtype=torch.float64)
    for e in range(E):
        if total[e] > 0:
            mu_e[e] = mu_sum[e] / int(total[e])
    return {
        "mu_e": mu_e,
        "mu_route": route_sum / sum(n_tok.values()),
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
        torch.arange(n, dtype=torch.int32),
        torch.zeros(n, dtype=torch.uint8),
    )
    w.close()
    w2 = ShardWriter(tmp_path, "text", shard_size=8)
    w2.add(
        torch.randn(2, K, D),
        experts[:2],
        torch.rand(2, K),
        torch.arange(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.uint8),
    )
    w2.close()

    stats = compute_stats(tmp_path, num_experts=E, hidden_size=D)
    assert int(stats.bank_counts[6]) == 0 and int(stats.bank_counts[7]) == 0
    assert torch.equal(stats.mu_e[6], torch.zeros(D))
    assert float(stats.p_route_visual[7]) == 0.0
    assert float(stats.p_gate_visual[7]) == 0.0


def test_domain_counts(tmp_path):
    build_capture(tmp_path, n_visual=11, n_text=7)
    stats = compute_stats(tmp_path, num_experts=E, hidden_size=D)
    # domains cycle 0,1,2,3 over 11 visual tokens: 3,3,3,2
    assert stats.domain_counts["visual"] == {"coco": 3, "textvqa": 3, "chartqa": 3, "ai2d": 2}
    assert stats.domain_counts["text"] == {"coco": 2, "textvqa": 2, "chartqa": 2, "ai2d": 1}


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
