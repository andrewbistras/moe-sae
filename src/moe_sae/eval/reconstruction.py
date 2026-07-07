"""Reconstruction evaluation under frozen-θ* inference (spec §13, §20 step 6)
and FeatureDump production (docs/INTERFACES.md §FeatureDump).

Everything here runs on held-out capture shards: rows are filtered by the
frozen split convention `example_id % 100` (0–79 train, 80–89 dev, 90–99 eval;
docs/INTERFACES.md §Splits).

Condition interface
-------------------
This module consumes any object satisfying `ThresholdedSAE` below. The concrete
implementations live in `moe_sae.sae.conditions` (built in parallel); this
Protocol pins the exact surface the evaluation needs:

- `encode_thresholded(c, experts, theta_star) -> ThresholdedEncoding` runs
  θ*-thresholded inference (spec §6): z = a·1[a > θ*], ĉ = μ + D z. For
  router-tied conditions (A/C/D/E) `recon_slots` is per-slot [N, 6, d]; for
  global conditions (B/F, num_banks == 1) the implementation sums the slots
  itself and returns [N, 1, d].
- `decoder_cols(bank, latent) -> [M, d]` unit-norm decoder columns D_{e,j}
  (needed by the §17 subtraction intervention).

Deadness note: the "dead-feature %" reported here is *eval-set deadness* —
latents that never fired under θ*-inference on this split. It is distinct from
the training-time deadness definition (spec §8: bank-relative 250k-token
window), which drives the aux loss and is reported from training artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from safetensors.torch import load_file, save_file

from moe_sae.capture.writer import POOLS, iter_shards
from moe_sae.model_loading import NUM_ROUTED_EXPERTS

# Frozen split convention (docs/INTERFACES.md): example_id % 100.
SPLIT_RANGES = {"train": (0, 80), "dev": (80, 90), "eval": (90, 100)}

# Firing-frequency histogram bins (log10 of per-token firing rate).
FREQ_HIST_EDGES = [-8.0 + 0.2 * i for i in range(41)]  # 40 bins, 1e-8 .. 1

TOKEN_TYPE = {"text": 0, "visual": 1}  # matches TrainBatch contract


@dataclass
class ThresholdedEncoding:
    """Sparse θ*-inference result for N tokens (see module docstring)."""

    act_token: torch.Tensor  # [M] int64 — token row index (within the input)
    act_slot: torch.Tensor  # [M] int64 — slot 0..5; 0 for global conditions
    act_bank: torch.Tensor  # [M] int64 — bank the latent lives in; 0 for global
    act_latent: torch.Tensor  # [M] int64 — latent index within the bank
    act_value: torch.Tensor  # [M] float32 — z value (post θ* threshold)
    recon_slots: torch.Tensor  # [N, S, d] float32 — ĉ per slot (S=6) / [N,1,d] global
    mu_slots: torch.Tensor  # [N, S, d] float32 — μ_e per slot / μ broadcast


@runtime_checkable
class ThresholdedSAE(Protocol):
    condition: str  # "A".."F"
    f: int  # per-bank dictionary width
    num_banks: int  # 64 for router-tied conditions, 1 for global (B/F)

    def encode_thresholded(
        self,
        c: torch.Tensor,
        experts: torch.Tensor,
        example_id: torch.Tensor,
        theta_star: float,
    ) -> ThresholdedEncoding: ...

    def decoder_cols(self, bank: torch.Tensor, latent: torch.Tensor) -> torch.Tensor: ...


def split_mask(example_id: torch.Tensor, split: str) -> torch.Tensor:
    """Frozen example-level split membership (never position-level; spec §16.4)."""
    lo, hi = SPLIT_RANGES[split]
    m = example_id.to(torch.int64) % 100
    return (m >= lo) & (m < hi)


def _encode_chunked(
    condition: ThresholdedSAE,
    c: torch.Tensor,
    experts: torch.Tensor,
    example_id: torch.Tensor,
    theta_star: float,
    device: str,
    chunk_tokens: int,
) -> ThresholdedEncoding:
    """Chunk over tokens (24.6k candidates/token at discovery scale) and stitch
    the sparse encodings back together with corrected token offsets.

    `example_id` travels with each chunk: condition C's bank assignment is a
    per-example permutation even at inference."""
    parts: list[ThresholdedEncoding] = []
    n = c.shape[0]
    for start in range(0, n, chunk_tokens):
        enc = condition.encode_thresholded(
            c[start : start + chunk_tokens].to(device),
            experts[start : start + chunk_tokens].to(device),
            example_id[start : start + chunk_tokens].to(device),
            theta_star,
        )
        parts.append(
            ThresholdedEncoding(
                act_token=enc.act_token.to("cpu", torch.int64) + start,
                act_slot=enc.act_slot.to("cpu", torch.int64),
                act_bank=enc.act_bank.to("cpu", torch.int64),
                act_latent=enc.act_latent.to("cpu", torch.int64),
                act_value=enc.act_value.to("cpu", torch.float32),
                recon_slots=enc.recon_slots.to("cpu", torch.float32),
                mu_slots=enc.mu_slots.to("cpu", torch.float32),
            )
        )
    return ThresholdedEncoding(
        act_token=torch.cat([p.act_token for p in parts]),
        act_slot=torch.cat([p.act_slot for p in parts]),
        act_bank=torch.cat([p.act_bank for p in parts]),
        act_latent=torch.cat([p.act_latent for p in parts]),
        act_value=torch.cat([p.act_value for p in parts]),
        recon_slots=torch.cat([p.recon_slots for p in parts]),
        mu_slots=torch.cat([p.mu_slots for p in parts]),
    )


def _extend_bincount(acc: list[int], values: torch.Tensor) -> None:
    bc = torch.bincount(values)
    if len(acc) < bc.shape[0]:
        acc.extend([0] * (bc.shape[0] - len(acc)))
    for i, v in enumerate(bc.tolist()):
        acc[i] += v


@torch.no_grad()
def evaluate_reconstruction(
    condition: ThresholdedSAE,
    theta_star: float,
    capture_root: str | Path,
    split: str,
    out_dump_dir: str | Path | None,
    device: str = "cpu",
    chunk_tokens: int = 1024,
) -> dict:
    """Spec §13 reconstruction metrics + FeatureDump shards over one split.

    Two streaming passes: pass 1 computes the split's routed-aggregate mean
    (denominator of the aggregate NMSE / explained variance, spec §8); pass 2
    encodes under θ*, accumulates all metrics in float64, and writes one
    FeatureDump shard per capture shard that has ≥1 row in the split.

    Local NMSE denominators use the condition's *frozen* means μ (spec §5),
    carried in `mu_slots`; for global conditions (num_banks == 1) the "local"
    metric is the same objective computed on c_route with the fixed-μ
    denominator, and the macro metric is undefined (null).
    """
    root = Path(capture_root)
    assert split in SPLIT_RANGES, split
    is_global = condition.num_banks == 1
    n_latents_total = condition.num_banks * condition.f

    # ---- pass 1: split mean of c_route (float64) ----
    route_sum = torch.zeros(0, dtype=torch.float64)
    n_total = 0
    for pool in POOLS:
        for shard in iter_shards(root, pool):
            keep = split_mask(shard["example_id"], split)
            if not keep.any():
                continue
            c = shard["c"][keep].to(torch.float64)
            if route_sum.numel() == 0:
                route_sum = torch.zeros(c.shape[-1], dtype=torch.float64)
            route_sum += c.sum(dim=1).sum(dim=0)
            n_total += int(keep.sum())
    assert n_total > 0, f"split '{split}' has no tokens under {root}"
    route_mean = route_sum / n_total

    # ---- pass 2: encode, accumulate, dump ----
    dump_dir = Path(out_dump_dir) if out_dump_dir is not None else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)

    num_local = den_local = 0.0
    num_agg = den_agg = 0.0
    num_e = torch.zeros(NUM_ROUTED_EXPERTS, dtype=torch.float64)
    den_e = torch.zeros(NUM_ROUTED_EXPERTS, dtype=torch.float64)
    fire_counts = torch.zeros(n_latents_total, dtype=torch.int64)
    l0_hist = {p: [] for p in POOLS}
    l0_sum = {p: 0 for p in POOLS}
    n_pool = {p: 0 for p in POOLS}
    dump_idx = 0

    for pool in POOLS:
        for shard in iter_shards(root, pool):
            keep = split_mask(shard["example_id"], split)
            if not keep.any():
                continue
            c = shard["c"][keep]  # [n, 6, d] bf16
            experts = shard["experts"][keep].to(torch.int64)  # [n, 6]
            ex_ids = shard["example_id"][keep].to(torch.int64)
            n = c.shape[0]
            enc = _encode_chunked(
                condition, c, experts, ex_ids, theta_star, device, chunk_tokens
            )

            c64 = c.to(torch.float64)
            recon = enc.recon_slots.to(torch.float64)  # [n, S, d]
            mu = enc.mu_slots.to(torch.float64)
            # Target the reconstruction approximates: per-slot c_te for
            # router-tied conditions, c_route for global conditions.
            target = c64 if not is_global else c64.sum(dim=1, keepdim=True)

            err = ((target - recon) ** 2).sum(dim=-1)  # [n, S]
            base = ((target - mu) ** 2).sum(dim=-1)  # [n, S]
            num_local += float(err.sum())
            den_local += float(base.sum())

            route = c64.sum(dim=1)
            recon_route = recon.sum(dim=1)
            num_agg += float(((route - recon_route) ** 2).sum())
            den_agg += float(((route - route_mean) ** 2).sum())

            if not is_global:
                flat_e = experts.reshape(-1)
                num_e.index_add_(0, flat_e, err.reshape(-1))
                den_e.index_add_(0, flat_e, base.reshape(-1))

            flat_ids = enc.act_bank * condition.f + enc.act_latent
            fire_counts += torch.bincount(flat_ids, minlength=n_latents_total)

            l0 = torch.bincount(enc.act_token, minlength=n).to(torch.int32)
            _extend_bincount(l0_hist[pool], l0.to(torch.int64))
            l0_sum[pool] += int(l0.sum())
            n_pool[pool] += n

            if dump_dir is not None:
                energy = torch.zeros(n, 6, dtype=torch.float32)
                slot_energy = ((recon - mu) ** 2).sum(dim=-1).to(torch.float32)
                energy[:, : slot_energy.shape[1]] = slot_energy  # global → slot 0
                dump = {
                    # sparse activations (contract dtypes)
                    "act_token": enc.act_token.contiguous(),
                    "act_slot": enc.act_slot.to(torch.uint8).contiguous(),
                    "act_latent": enc.act_latent.to(torch.int32).contiguous(),
                    "act_value": enc.act_value.contiguous(),
                    # extra: true bank identity (== experts[token, slot] for A;
                    # the permuted bank for condition C — required so the §17
                    # intervention subtracts bank-correct decoder columns).
                    "act_bank": enc.act_bank.to(torch.uint8).contiguous(),
                    # per-token fields
                    "experts": shard["experts"][keep].contiguous(),
                    "gates": shard["gates"][keep].contiguous(),
                    "token_type": torch.full(
                        (n,), TOKEN_TYPE[pool], dtype=torch.uint8
                    ),
                    "example_id": shard["example_id"][keep].contiguous(),
                    "domain": shard["domain"][keep].contiguous(),
                    "l0": l0.contiguous(),
                    # extra: per-slot reconstructed-vector energy ‖D_e z_te‖²
                    "energy": energy.contiguous(),
                }
                save_file(dump, str(dump_dir / f"shard_{dump_idx:05d}.safetensors"))
                dump_idx += 1

    # ---- assemble metrics ----
    covered = (den_e > 0).nonzero(as_tuple=True)[0]
    macro = (
        float((num_e[covered] / den_e[covered]).mean()) if not is_global and len(covered) else None
    )
    fired = fire_counts > 0
    freqs = fire_counts[fired].to(torch.float64) / n_total
    log_f = torch.log10(freqs.clamp_min(1e-30))
    edges = torch.tensor(FREQ_HIST_EDGES, dtype=torch.float64)
    freq_hist = torch.histogram(log_f.to(torch.float32), bins=edges.to(torch.float32))[0]

    metrics = {
        "condition": getattr(condition, "condition", "?"),
        "split": split,
        "theta_star": float(theta_star),
        "n_tokens": {"visual": n_pool["visual"], "text": n_pool["text"], "total": n_total},
        "local_micro_nmse": (num_local / den_local) if den_local > 0 else None,
        "local_macro_nmse": macro,
        "local_experts_covered": int(len(covered)) if not is_global else None,
        "aggregate_nmse": (num_agg / den_agg) if den_agg > 0 else None,
        "explained_variance": (1.0 - num_agg / den_agg) if den_agg > 0 else None,
        "mean_l0": {
            "overall": (l0_sum["visual"] + l0_sum["text"]) / n_total,
            "visual": l0_sum["visual"] / max(n_pool["visual"], 1),
            "text": l0_sum["text"] / max(n_pool["text"], 1),
        },
        "l0_hist": {p: l0_hist[p] for p in POOLS},
        "n_latents_total": n_latents_total,
        "dead_feature_pct": 100.0 * float((~fired).sum()) / n_latents_total,
        "firing_freq_hist": {
            "log10_bin_edges": FREQ_HIST_EDGES,
            "counts": freq_hist.to(torch.int64).tolist(),
            "never_fired": int((~fired).sum()),
        },
        "notes": (
            "dead_feature_pct is eval-set deadness (never fired under θ*-inference "
            "on this split), distinct from the spec §8 training deadness window; "
            "aggregate denominator uses this split's own c_route mean, local "
            "denominators use the condition's frozen μ (spec §5)."
        ),
    }
    if dump_dir is not None:
        (dump_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def dump_features(
    condition: ThresholdedSAE,
    theta_star: float,
    capture_root: str | Path,
    split: str,
    out_dir: str | Path,
    device: str = "cpu",
    chunk_tokens: int = 1024,
) -> dict:
    """Contract entry point (docs/INTERFACES.md §FeatureDump): produce the
    θ*-inference dump for a split. Metrics come along for free (same pass)."""
    return evaluate_reconstruction(
        condition, theta_star, capture_root, split, out_dir, device, chunk_tokens
    )


def iter_dump_shards(dump_dir: str | Path):
    """Iterate FeatureDump shards in shard order."""
    for p in sorted(Path(dump_dir).glob("shard_*.safetensors")):
        yield load_file(str(p))
