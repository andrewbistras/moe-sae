"""Statistics pass over capture shards (spec §20 step 2).

Computes, in one streaming pass with float64 accumulators:

- per-expert means μ_e = E[c_te | e ∈ S_t] over both pools (spec §5 centering,
  frozen once computed — the trainer must load them from here, never re-estimate);
- routed-aggregate mean μ_route = E_t[Σ_slots c_te] (denominator of the
  aggregate diagnostic loss, spec §8);
- per-bank routed-token counts (total and per pool) — the §16.3 load-floor
  inputs;
- router baselines per modality (spec §15): routing frequency
  p_route(e|M) = #{(t,slot): t∈M, S_t[slot]=e} / (6·N_M) and gate mass
  p_gate(e|M) = Σ_{t∈M, e∈S_t} g_te / Σ_{t∈M} Σ_e' g_te';
- per-domain token counts per pool.

Outputs `stats.safetensors` (tensors, float32/int64) plus a human-readable
`stats.json` summary. Banks never routed to have μ_e = 0 and count 0.

Decisions:
- Split filtering (docs/INTERFACES.md §Splits): μ_e and the router baselines
  are training-distribution statistics (spec §5 "initial statistics pass"), so
  `compute_stats` consumes ONLY the train buckets (`example_id % 100` in 0–79)
  by default (`split='train'`; `split=None` disables filtering). stats.json
  reports kept and excluded token counts per pool plus the split used. The
  modulo applies to the raw int32 domain-offset example_id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from moe_sae.capture.writer import DOMAIN_NAMES, POOLS, iter_shards
from moe_sae.data.batches import SPLIT_RANGES, split_mask
from moe_sae.model_loading import HIDDEN_SIZE, NUM_ROUTED_EXPERTS, TOP_K_ROUTED

STATS_FILE = "stats.safetensors"
SUMMARY_FILE = "stats.json"


@dataclass
class ActivationStats:
    """Frozen statistics consumed by training (μ_e) and analysis (§15, §16.3)."""

    mu_e: torch.Tensor  # [64, 1280] float32
    mu_route: torch.Tensor  # [1280] float32
    bank_counts: torch.Tensor  # [64] int64 — routed tokens per bank, both pools
    bank_counts_visual: torch.Tensor  # [64] int64
    bank_counts_text: torch.Tensor  # [64] int64
    p_route_visual: torch.Tensor  # [64] float32, sums to 1
    p_route_text: torch.Tensor  # [64] float32, sums to 1
    p_gate_visual: torch.Tensor  # [64] float32, sums to 1
    p_gate_text: torch.Tensor  # [64] float32, sums to 1
    tokens_visual: int
    tokens_text: int
    domain_counts: dict  # {pool: {domain_name: n_tokens}}
    split: str | None = "train"  # split the pass was restricted to (None = all)
    excluded_visual: int = 0  # tokens dropped by the split filter per pool
    excluded_text: int = 0

    @property
    def tokens_total(self) -> int:
        return self.tokens_visual + self.tokens_text


@torch.no_grad()
def compute_stats(
    capture_root: str | Path,
    num_experts: int = NUM_ROUTED_EXPERTS,
    hidden_size: int = HIDDEN_SIZE,
    top_k: int = TOP_K_ROUTED,
    split: str | None = "train",
) -> ActivationStats:
    """Single streaming pass over both pools; all accumulators float64/int64.

    Only `split` rows (default: train buckets, example_id % 100 in 0–79) are
    accumulated; `split=None` disables the filter (debugging only).
    """
    assert split is None or split in SPLIT_RANGES, split
    root = Path(capture_root)
    d = hidden_size

    mu_sum = torch.zeros(num_experts, d, dtype=torch.float64)
    route_sum = torch.zeros(d, dtype=torch.float64)
    counts = {p: torch.zeros(num_experts, dtype=torch.int64) for p in POOLS}
    gate_sums = {p: torch.zeros(num_experts, dtype=torch.float64) for p in POOLS}
    n_tokens = {p: 0 for p in POOLS}
    n_excluded = {p: 0 for p in POOLS}
    domain_counts: dict[str, dict[str, int]] = {p: {} for p in POOLS}

    for pool in POOLS:
        for shard in iter_shards(root, pool):
            if split is not None:
                keep = split_mask(shard["example_id"], split)
                n_excluded[pool] += int((~keep).sum())
                shard = {k: v[keep] for k, v in shard.items()}
                if not bool(keep.any()):
                    continue
            c = shard["c"].to(torch.float64)  # [N, 6, d]
            experts = shard["experts"].to(torch.int64)  # [N, 6]
            gates = shard["gates"].to(torch.float64)  # [N, 6]
            n, k, _ = c.shape
            assert k == top_k, f"expected {top_k} slots, got {k}"

            flat_e = experts.reshape(-1)
            mu_sum.index_add_(0, flat_e, c.reshape(-1, d))
            counts[pool] += torch.bincount(flat_e, minlength=num_experts)
            gate_sums[pool].index_add_(0, flat_e, gates.reshape(-1))
            route_sum += c.sum(dim=1).sum(dim=0)
            n_tokens[pool] += n

            for dom, cnt in zip(*torch.unique(shard["domain"], return_counts=True)):
                name = DOMAIN_NAMES.get(int(dom), str(int(dom)))
                domain_counts[pool][name] = domain_counts[pool].get(name, 0) + int(cnt)

    total_tokens = sum(n_tokens.values())
    assert total_tokens > 0, f"no captured tokens under {root}"
    bank_counts = counts["visual"] + counts["text"]

    # μ_e = Σ c_te / count_e, guarding never-routed banks (μ_e := 0).
    mu_e = torch.where(
        bank_counts.unsqueeze(1) > 0,
        mu_sum / bank_counts.clamp_min(1).unsqueeze(1).to(torch.float64),
        torch.zeros_like(mu_sum),
    )
    mu_route = route_sum / total_tokens

    def p_route(pool: str) -> torch.Tensor:
        denom = max(top_k * n_tokens[pool], 1)
        return (counts[pool].to(torch.float64) / denom).to(torch.float32)

    def p_gate(pool: str) -> torch.Tensor:
        total = gate_sums[pool].sum().clamp_min(1e-300)
        return (gate_sums[pool] / total).to(torch.float32)

    return ActivationStats(
        mu_e=mu_e.to(torch.float32),
        mu_route=mu_route.to(torch.float32),
        bank_counts=bank_counts,
        bank_counts_visual=counts["visual"],
        bank_counts_text=counts["text"],
        p_route_visual=p_route("visual"),
        p_route_text=p_route("text"),
        p_gate_visual=p_gate("visual"),
        p_gate_text=p_gate("text"),
        tokens_visual=n_tokens["visual"],
        tokens_text=n_tokens["text"],
        domain_counts=domain_counts,
        split=split,
        excluded_visual=n_excluded["visual"],
        excluded_text=n_excluded["text"],
    )


def save_stats(stats: ActivationStats, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {
        "mu_e": stats.mu_e.contiguous(),
        "mu_route": stats.mu_route.contiguous(),
        "bank_counts": stats.bank_counts.contiguous(),
        "bank_counts_visual": stats.bank_counts_visual.contiguous(),
        "bank_counts_text": stats.bank_counts_text.contiguous(),
        "p_route_visual": stats.p_route_visual.contiguous(),
        "p_route_text": stats.p_route_text.contiguous(),
        "p_gate_visual": stats.p_gate_visual.contiguous(),
        "p_gate_text": stats.p_gate_text.contiguous(),
    }
    save_file(tensors, str(out / STATS_FILE))

    counts = stats.bank_counts
    summary = {
        "split": stats.split,
        "tokens": {
            "visual": stats.tokens_visual,
            "text": stats.tokens_text,
            "total": stats.tokens_total,
            "excluded_visual": stats.excluded_visual,
            "excluded_text": stats.excluded_text,
        },
        "domain_counts": stats.domain_counts,
        "bank_counts": counts.tolist(),
        "bank_load": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.to(torch.float64).mean()),
        },
        "mu_e_norm_mean": float(stats.mu_e.norm(dim=1).mean()),
        "mu_route_norm": float(stats.mu_route.norm()),
        "top6_p_gate_visual": torch.topk(stats.p_gate_visual, 6).indices.tolist(),
        "top6_p_gate_text": torch.topk(stats.p_gate_text, 6).indices.tolist(),
    }
    (out / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    return out / STATS_FILE


def load_stats(stats_dir: str | Path) -> ActivationStats:
    """Load a saved stats pass. `stats_dir` is the directory (or the .safetensors
    file itself); the json summary supplies the scalar counts."""
    p = Path(stats_dir)
    if p.is_dir():
        tens_path, json_path = p / STATS_FILE, p / SUMMARY_FILE
    else:
        tens_path, json_path = p, p.parent / SUMMARY_FILE
    t = load_file(str(tens_path))
    summary = json.loads(json_path.read_text())
    return ActivationStats(
        mu_e=t["mu_e"],
        mu_route=t["mu_route"],
        bank_counts=t["bank_counts"],
        bank_counts_visual=t["bank_counts_visual"],
        bank_counts_text=t["bank_counts_text"],
        p_route_visual=t["p_route_visual"],
        p_route_text=t["p_route_text"],
        p_gate_visual=t["p_gate_visual"],
        p_gate_text=t["p_gate_text"],
        tokens_visual=summary["tokens"]["visual"],
        tokens_text=summary["tokens"]["text"],
        domain_counts=summary["domain_counts"],
        split=summary.get("split", "train"),
        excluded_visual=summary["tokens"].get("excluded_visual", 0),
        excluded_text=summary["tokens"].get("excluded_text", 0),
    )
