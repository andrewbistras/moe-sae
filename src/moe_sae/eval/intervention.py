"""Feature-subtraction intervention mechanics (spec §17, changelog #2).

Convention (frozen): for each held-out token, encode the TRUE routed
contributions under frozen θ*, intersect the frozen ablation set J with the
latents that actually fired, and subtract exactly those latents' decoder
contributions from the model's own activation:

    c_route'(t) = c_route(t) − Σ_{(e,j) ∈ J_t} z_{t,e,j} · D_{e,j},
    J_t = J ∩ fired(t).

No reconstruction error is injected anywhere: positions where J_t is empty are
bit-identical to the clean forward (unit-tested), and modified positions differ
from the true activation only by the named features' contributions. Single-shot:
the modified vector is never re-encoded.

The enriched-set definition (`select_enriched_latents`) runs on the DEV split
dump only (spec §20 step 9); `run_intervention` then runs on untouched
evaluation data. The identical pipeline is run on condition C as the negative
control (changelog #10) — the FeatureDump's `act_bank` field carries the true
bank identity so the subtraction uses bank-correct decoder columns even when
banks are permuted (condition C).

Decisions:
- Spec §17 fifth matching criterion (removed-vector norm
  ‖Σ_{(e,j)∈J_t} z·D_{e,j}‖₂): exact pre-hoc token-level norm matching is
  impossible — the norm depends on which set members co-fire at each token and
  on cross-latent decoder-column geometry, both token-dependent. It is
  implemented as the per-latent coefficient-mass proxy (exact per latent under
  unit decoder columns) at construction time, plus POST-HOC measurement:
  `removed_norm_report` computes per-token removed-vector norms for J and each
  control over a provided sample of encoded tokens (via the same
  `decoder_cols` access `run_intervention` uses) and flags any control set
  whose mean removed norm deviates from J's by more than `norm_tolerance`
  (default 20%, configurable). `matched_control_sets` embeds this under
  `removed_norms` in its match-quality report when given `decoder_cols` +
  `encoded_samples`.
- §16.3 load floor (A13, v0.1.2 addendum — supersedes the v0.1.1 absolute
  500k floor and the earlier training-loads source): a bank enters enrichment
  rankings iff its routed token count IN THE SPLIT UNDER ANALYSIS is at least
  10% of the split's uniform-load expectation E_uniform = (split_tokens×6)/64.
  Both the counts (from the dump's host-expert routing, `dump.experts`) and
  the default floor are computed from the analysis dump itself
  (`dump_split_bank_counts` / `split_proportional_floor`); an absolute
  `load_floor` override remains available (scripts keep an --load-floor
  flag). Per-bank counts are still reported for all 64 banks. Matched-control
  LOAD matching keeps the stats-pass counts (the floor is the only
  comparison A13 re-sources).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Iterable

import torch

from moe_sae.capture.hooks import instrumented_capture
from moe_sae.eval.fidelity import (
    _aggregate,
    _aggregate_by_domain,
    add_to_moe_output,
    answer_ce_and_acc,
    forward_logits,
)
from moe_sae.eval.reconstruction import (
    ThresholdedEncoding,
    ThresholdedSAE,
    iter_dump_shards,
    theta_star_json,
)

# A13 (v0.1.2): the load floor is split-proportional — 10% of the split's
# uniform-load expectation E_uniform = (split_tokens × TOP_K) / N_BANKS.
# (The v0.1.1 absolute 500k floor is superseded; see module Decisions.)
FLOOR_FRACTION = 0.10

# Matching tolerances in log10 units (documented free parameters of the
# §17 matched-control construction; achieved quality is always recorded).
DEFAULT_TOLERANCES = {"freq": 0.3, "mass": 0.3, "load": 0.3}

# Post-hoc removed-vector-norm check (spec §17 fifth criterion, see module
# Decisions): flag a control set whose mean per-token removed norm deviates
# from J's by more than this relative tolerance.
DEFAULT_NORM_TOLERANCE = 0.2

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Per-latent statistics from a FeatureDump                                     #
# --------------------------------------------------------------------------- #


def _activation_banks(shard: dict) -> torch.Tensor:
    """True bank per activation: the dump's act_bank field, falling back to
    experts[token, slot] for contract-minimal dumps (exact for condition A)."""
    if "act_bank" in shard:
        return shard["act_bank"].to(torch.int64)
    tok = shard["act_token"].to(torch.int64)
    slot = shard["act_slot"].to(torch.int64)
    return shard["experts"].to(torch.int64)[tok, slot]


def latent_stats_from_dump(dump_dir: str | Path, f: int) -> dict:
    """Aggregate per-latent statistics over a FeatureDump.

    Returns flat-id keyed tensors (flat id = bank·f + latent, deterministic):
    ids [L] i64 (sorted), fires_visual/fires_text [L] i64, mass [L] f64,
    plus per-bank coefficient-mass totals by modality (for R_M(e)) and token
    counts. Only latents that fired at least once appear.
    """
    acc_fires_v: dict[int, int] = {}
    acc_fires_t: dict[int, int] = {}
    acc_mass: dict[int, float] = {}
    bank_mass_v = torch.zeros(64, dtype=torch.float64)
    bank_mass_t = torch.zeros(64, dtype=torch.float64)
    n_vis = n_txt = 0

    for shard in iter_dump_shards(dump_dir):
        bank = _activation_banks(shard)
        latent = shard["act_latent"].to(torch.int64)
        value = shard["act_value"].to(torch.float64)
        tok_type = shard["token_type"].to(torch.int64)
        act_vis = tok_type[shard["act_token"].to(torch.int64)] == 1
        flat = bank * f + latent

        bank_mass_v.index_add_(0, bank[act_vis], value[act_vis])
        bank_mass_t.index_add_(0, bank[~act_vis], value[~act_vis])
        n_vis += int((tok_type == 1).sum())
        n_txt += int((tok_type == 0).sum())

        for vis in (True, False):
            sel = act_vis if vis else ~act_vis
            ids, inverse = torch.unique(flat[sel], return_inverse=True)
            counts = torch.bincount(inverse, minlength=ids.shape[0])
            masses = torch.zeros(ids.shape[0], dtype=torch.float64)
            masses.index_add_(0, inverse, value[sel])
            acc = acc_fires_v if vis else acc_fires_t
            for i, cnt, m in zip(ids.tolist(), counts.tolist(), masses.tolist()):
                acc[i] = acc.get(i, 0) + cnt
                acc_mass[i] = acc_mass.get(i, 0.0) + m

    ids = sorted(set(acc_fires_v) | set(acc_fires_t))
    return {
        "ids": torch.tensor(ids, dtype=torch.int64),
        "fires_visual": torch.tensor([acc_fires_v.get(i, 0) for i in ids], dtype=torch.int64),
        "fires_text": torch.tensor([acc_fires_t.get(i, 0) for i in ids], dtype=torch.int64),
        "mass": torch.tensor([acc_mass[i] for i in ids], dtype=torch.float64),
        "bank_mass_visual": bank_mass_v,
        "bank_mass_text": bank_mass_t,
        "n_tokens_visual": n_vis,
        "n_tokens_text": n_txt,
    }


# --------------------------------------------------------------------------- #
# Enriched-set definition (DEV split only; spec §20 step 9)                    #
# --------------------------------------------------------------------------- #


def dump_split_bank_counts(dump_dir: str | Path) -> tuple[torch.Tensor, int]:
    """A13 floor inputs: per-bank routed token counts in the split under
    analysis, from the dump's host-expert routing (`experts` field), plus the
    split's token count. Returns ([64] int64 counts, n_split_tokens)."""
    counts = torch.zeros(64, dtype=torch.int64)
    n_tokens = 0
    for shard in iter_dump_shards(dump_dir):
        experts = shard["experts"].to(torch.int64)
        counts += torch.bincount(experts.reshape(-1), minlength=64)
        n_tokens += experts.shape[0]
    return counts, n_tokens


def split_proportional_floor(
    n_split_tokens: int, top_k: int = 6, n_banks: int = 64
) -> float:
    """A13 frozen floor: 10% of the split's uniform-load expectation
    E_uniform = (split_tokens × top_k) / n_banks."""
    return FLOOR_FRACTION * (n_split_tokens * top_k) / n_banks


def select_enriched_latents(
    dev_dump_dir: str | Path,
    stats,
    f: int,
    top_n_experts: int = 6,
    quantile: float = 0.9,
    load_floor: float | None = None,
    min_visual_fires: int = 1,
    eps: float = 1e-9,
    bank_counts=None,
) -> list[tuple[int, int]]:
    """Deterministic frozen ablation set J from the DEV-split FeatureDump.

    1. Image-enriched experts: rank banks by R_visual(e) = log((q_vis(e)+ε) /
       (p_gate_visual(e)+ε)) — SAE coefficient mass from the dump itself vs the
       stats-pass gate-mass baseline (spec §15) — over banks meeting the
       §16.3 load floor; take the top `top_n_experts`.
    2. Within those banks: latents with ≥ `min_visual_fires` visual fires whose
       visual firing-rate enrichment (rate_vis+ε)/(rate_text+ε) is at or above
       the given `quantile` of that distribution.

    Load floor (A13): a bank qualifies iff its routed token count IN THE
    SPLIT UNDER ANALYSIS ≥ 0.10 × (split_tokens×6)/64. `bank_counts` defaults
    to the dump's own host-expert routing (`dump_split_bank_counts`);
    `load_floor=None` (default) computes the split-proportional floor, while
    an explicit value is an absolute override (kept for audits/tests).
    `stats` is a `moe_sae.train.stats_pass.ActivationStats` (duck-typed:
    needs `.p_gate_visual` [64]). Returns a sorted list of (bank, latent)
    pairs.
    """
    if bank_counts is None:
        bank_counts, n_split_tokens = dump_split_bank_counts(dev_dump_dir)
    else:
        bank_counts = torch.as_tensor(bank_counts, dtype=torch.int64)
        # each token contributes exactly TOP_K routed slots by construction
        n_split_tokens = int(bank_counts.sum().item()) // 6
    bank_counts = torch.as_tensor(bank_counts, dtype=torch.int64)
    assert bank_counts.shape == (64,), bank_counts.shape
    if load_floor is None:
        load_floor = split_proportional_floor(n_split_tokens)
        logger.info(
            "select_enriched_latents: A13 split-proportional load floor = %.1f "
            "(10%% of uniform expectation over %d split tokens)",
            load_floor, n_split_tokens,
        )

    ls = latent_stats_from_dump(dev_dump_dir, f)
    q_vis = ls["bank_mass_visual"] / ls["bank_mass_visual"].sum().clamp_min(1e-300)
    r_vis = torch.log(
        (q_vis + eps) / (stats.p_gate_visual.to(torch.float64) + eps)
    )
    eligible = bank_counts >= load_floor
    assert int(eligible.sum()) >= top_n_experts, (
        f"only {int(eligible.sum())} banks meet the load floor {load_floor}"
    )
    r_masked = torch.where(eligible, r_vis, torch.full_like(r_vis, -torch.inf))
    enriched_banks = torch.topk(r_masked, top_n_experts).indices
    enriched_set = set(enriched_banks.tolist())

    n_vis = max(ls["n_tokens_visual"], 1)
    n_txt = max(ls["n_tokens_text"], 1)
    banks = torch.div(ls["ids"], f, rounding_mode="floor")
    in_bank = torch.tensor([int(b) in enriched_set for b in banks.tolist()], dtype=torch.bool)
    candidate = in_bank & (ls["fires_visual"] >= min_visual_fires)
    if not candidate.any():
        return []

    rate_v = ls["fires_visual"].to(torch.float64) / n_vis
    rate_t = ls["fires_text"].to(torch.float64) / n_txt
    enrichment = (rate_v + eps) / (rate_t + eps)
    scores = enrichment[candidate]
    cutoff = torch.quantile(scores, quantile)
    keep_ids = ls["ids"][candidate][scores >= cutoff]
    return sorted((int(i) // f, int(i) % f) for i in keep_ids.tolist())


def matched_control_sets(
    J: list[tuple[int, int]],
    dev_dump_dir: str | Path,
    stats,
    f: int,
    n_controls: int = 10,
    seed: int = 0,
    tolerances: dict | None = None,
    decoder_cols: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    encoded_samples: Iterable[ThresholdedEncoding] | None = None,
    d: int | None = None,
    norm_tolerance: float = DEFAULT_NORM_TOLERANCE,
) -> tuple[list[list[tuple[int, int]]], dict]:
    """Control ablation sets matched to J (spec §17): same #latents, and — via
    greedy per-latent nearest matching in log10 space — matched total
    coefficient mass, activation frequency, and expert load.

    Removed-vector norm: with unit-norm decoder columns, each latent's removed
    contribution has norm exactly z, so total coefficient mass is the
    per-latent removed-norm proxy at construction time; the token-level norm
    (spec §17's fifth criterion, not matchable pre hoc — see module Decisions)
    is measured post hoc when `decoder_cols`, `encoded_samples` (a sample of
    `ThresholdedEncoding`s), and `d` are given: `removed_norm_report` output
    is embedded under `removed_norms` in the returned match-quality report,
    flagging sets whose mean removed norm deviates from J's by more than
    `norm_tolerance`.

    Per latent in J (deterministic sorted order) a control latent is drawn
    uniformly (seeded) from fired latents outside J and unused within the set
    whose |Δlog10| in every matched dimension is within `tolerances`; if none
    qualifies the nearest is taken and counted as out-of-tolerance. Achieved
    match quality is recorded in the returned report.
    """
    tol = dict(DEFAULT_TOLERANCES, **(tolerances or {}))
    ls = latent_stats_from_dump(dev_dump_dir, f)
    ids = ls["ids"]
    n_total = max(ls["n_tokens_visual"] + ls["n_tokens_text"], 1)
    fires = (ls["fires_visual"] + ls["fires_text"]).to(torch.float64)

    log_freq = torch.log10(fires.clamp_min(1) / n_total)
    log_mass = torch.log10(ls["mass"].clamp_min(1e-30))
    bank_of = torch.div(ids, f, rounding_mode="floor")
    load = stats.bank_counts.to(torch.float64)[bank_of]
    log_load = torch.log10(load.clamp_min(1))
    feats = torch.stack([log_freq, log_mass, log_load], dim=1)  # [L, 3]
    tol_vec = torch.tensor([tol["freq"], tol["mass"], tol["load"]], dtype=torch.float64)

    id_to_row = {int(i): r for r, i in enumerate(ids.tolist())}
    j_rows = []
    for b, l in sorted(J):
        flat = b * f + l
        assert flat in id_to_row, f"J latent ({b},{l}) never fired in the dev dump"
        j_rows.append(id_to_row[flat])
    j_row_set = set(j_rows)

    gen = torch.Generator().manual_seed(seed)
    controls: list[list[tuple[int, int]]] = []
    set_reports = []
    for _ in range(n_controls):
        used: set[int] = set()
        rows: list[int] = []
        n_out = 0
        for jr in j_rows:
            diff = (feats - feats[jr]).abs()  # [L, 3]
            blocked = torch.zeros(ids.shape[0], dtype=torch.bool)
            for r in j_row_set | used:
                blocked[r] = True
            within = (diff <= tol_vec).all(dim=1) & ~blocked
            if within.any():
                cand = within.nonzero(as_tuple=True)[0]
                pick = int(cand[torch.randint(cand.shape[0], (1,), generator=gen)])
            else:
                dist = (diff / tol_vec).norm(dim=1)
                dist[blocked] = torch.inf
                pick = int(dist.argmin())
                n_out += 1
            used.add(pick)
            rows.append(pick)
        controls.append(sorted((int(ids[r]) // f, int(ids[r]) % f) for r in rows))
        rt = torch.tensor(rows)
        jt = torch.tensor(j_rows)
        set_reports.append(
            {
                "n_latents": len(rows),
                "n_out_of_tolerance": n_out,
                "total_mass_ratio": float(ls["mass"][rt].sum() / ls["mass"][jt].sum()),
                "total_fires_ratio": float(fires[rt].sum() / fires[jt].sum()),
                "mean_abs_dlog10_freq": float((log_freq[rt] - log_freq[jt]).abs().mean()),
                "mean_abs_dlog10_load": float((log_load[rt] - log_load[jt]).abs().mean()),
            }
        )

    report = {
        "tolerances_log10": tol,
        "seed": seed,
        "n_controls": n_controls,
        "j_total_mass": float(ls["mass"][torch.tensor(j_rows)].sum()),
        "removed_norm_proxy": "total coefficient mass (unit-norm decoder columns)",
        "sets": set_reports,
    }
    if decoder_cols is not None and encoded_samples is not None:
        assert d is not None, "d (model dim) required for removed-norm measurement"
        report["removed_norms"] = removed_norm_report(
            J, controls, encoded_samples, f, decoder_cols, d, norm_tolerance
        )
    return controls, report


def removed_norm_report(
    J: list[tuple[int, int]],
    controls: list[list[tuple[int, int]]],
    encoded_samples: Iterable[ThresholdedEncoding],
    f: int,
    decoder_cols: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    d: int,
    norm_tolerance: float = DEFAULT_NORM_TOLERANCE,
) -> dict:
    """Post-hoc spec §17 fifth-criterion measurement (see module Decisions).

    Over a sample of encoded tokens, computes the per-token removed-vector
    norm ‖Σ_{(e,j)∈J_t} z·D_{e,j}‖₂ (at positions with nonempty J_t) for J and
    for each control set, and reports the distributions (n_positions, mean,
    median, p90) plus each control's mean ratio vs J. A control whose mean
    deviates from J's by more than `norm_tolerance` relative (or that cannot
    be compared: no ablated positions) gets `flag_removed_norm` True.
    """
    named = [("J", J)] + [(f"control_{i:02d}", c) for i, c in enumerate(controls)]
    norms: dict[str, list[torch.Tensor]] = {name: [] for name, _ in named}
    for enc in encoded_samples:
        if enc.act_token.numel() == 0:
            continue
        n_tok = int(enc.act_token.max()) + 1
        for name, latents in named:
            deltas, positions = compute_ablation_deltas(
                enc, latents, f, decoder_cols, n_tok, d
            )
            if positions.any():
                norms[name].append(deltas[positions].norm(dim=1))

    def _dist(chunks: list[torch.Tensor]) -> dict:
        if not chunks:
            return {"n_positions": 0, "mean": None, "median": None, "p90": None}
        v = torch.cat(chunks).to(torch.float64)
        return {
            "n_positions": int(v.numel()),
            "mean": float(v.mean()),
            "median": float(v.median()),
            "p90": float(torch.quantile(v, 0.9)),
        }

    j_dist = _dist(norms["J"])
    sets = []
    for i in range(len(controls)):
        dist = _dist(norms[f"control_{i:02d}"])
        comparable = dist["mean"] is not None and (j_dist["mean"] or 0) > 0
        ratio = dist["mean"] / j_dist["mean"] if comparable else None
        dist["mean_ratio_vs_J"] = ratio
        dist["flag_removed_norm"] = bool(
            ratio is None or abs(ratio - 1.0) > norm_tolerance
        )
        sets.append(dist)
    return {"norm_tolerance": norm_tolerance, "J": j_dist, "sets": sets}


# --------------------------------------------------------------------------- #
# Subtraction mechanics (spec §17)                                             #
# --------------------------------------------------------------------------- #


def compute_ablation_deltas(
    enc: ThresholdedEncoding,
    J: list[tuple[int, int]],
    f: int,
    decoder_cols: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n_tokens: int,
    d: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """delta[t] = Σ_{(e,j) ∈ J ∩ fired(t)} z_{t,e,j} · D_{e,j}  (spec §17).

    Returns (deltas [n_tokens, d] float32, positions [n_tokens] bool) where
    positions marks tokens with nonempty J_t. Tokens with empty J_t have an
    exactly-zero delta row and positions False.
    """
    deltas = torch.zeros(n_tokens, d, dtype=torch.float32)
    positions = torch.zeros(n_tokens, dtype=torch.bool)
    if not J:
        return deltas, positions
    j_flat = torch.tensor([b * f + l for b, l in J], dtype=torch.int64)
    act_flat = enc.act_bank.to(torch.int64) * f + enc.act_latent.to(torch.int64)
    mask = torch.isin(act_flat, j_flat)
    if mask.any():
        cols = decoder_cols(enc.act_bank[mask], enc.act_latent[mask]).to(torch.float32).cpu()
        contrib = enc.act_value[mask].to(torch.float32).cpu().unsqueeze(1) * cols
        tok = enc.act_token[mask].to(torch.int64).cpu()
        deltas.index_add_(0, tok, contrib)
        positions[tok] = True
    return deltas, positions


@torch.no_grad()
def run_intervention(
    model: torch.nn.Module,
    condition: ThresholdedSAE,
    theta_star: float,
    J: list[tuple[int, int]],
    eval_examples: Iterable,
    device: str = "cpu",
    moe: torch.nn.Module | None = None,
) -> dict:
    """Spec §17 subtraction intervention on held-out examples (batch 1).

    Per example: one clean instrumented forward (baseline logits + true
    contributions) → θ*-encode the TRUE activations → J_t = J ∩ fired(t) →
    second forward with c_route'(t) = c_route(t) − Σ_{J_t} z·D at exactly the
    tokens where J_t is nonempty. All other positions (and all positions when
    J_t is empty) are bit-identical to the clean forward — no reconstruction
    error is injected, and the modified vector is never re-encoded.

    The pipeline signature is condition-agnostic so the identical run serves
    as the condition-C negative control (spec §17, changelog #10).
    """
    if moe is None:
        from moe_sae.model_loading import get_target_moe

        moe = get_target_moe(model)

    records = []
    for ex in eval_examples:
        with instrumented_capture(moe, store_dtype=torch.float32, device="cpu") as sink:
            logits_clean = forward_logits(model, ex, device).to(torch.float32).cpu()
            chunks = sink.drain()
        assert len(chunks) == 1, "expected exactly one MoE forward per example"
        ch = chunks[0]
        c = ch.per_slot_contrib.to(torch.float32)  # [T, k, d]
        n_tok, _, d = c.shape

        ex_ids = torch.full((n_tok,), int(ex.example_id), dtype=torch.int64)
        enc = condition.encode_thresholded(
            c.to(device), ch.topk_idx.to(device), ex_ids.to(device), theta_star
        )
        enc = ThresholdedEncoding(  # normalize to CPU for the delta assembly
            act_token=enc.act_token.cpu(),
            act_slot=enc.act_slot.cpu(),
            act_bank=enc.act_bank.cpu(),
            act_latent=enc.act_latent.cpu(),
            act_value=enc.act_value.cpu(),
            recon_slots=enc.recon_slots.cpu(),
            mu_slots=enc.mu_slots.cpu(),
        )
        deltas, positions = compute_ablation_deltas(
            enc, J, condition.f, condition.decoder_cols, n_tok, d
        )

        with add_to_moe_output(moe, -deltas.to(device), positions.to(device)):
            logits_abl = forward_logits(model, ex, device).to(torch.float32).cpu()

        ce_c, acc_c = answer_ce_and_acc(logits_clean, ex.input_ids, ex.answer_mask)
        ce_a, acc_a = answer_ce_and_acc(logits_abl, ex.input_ids, ex.answer_mask)
        records.append(
            {
                "example_id": int(ex.example_id),
                "domain": ex.domain,
                "n_ablated_positions": int(positions.sum()),
                "mean_removed_norm": float(deltas[positions].norm(dim=1).mean())
                if positions.any()
                else 0.0,
                "ce_clean": ce_c,
                "ce_ablated": ce_a,
                "ce_delta": ce_a - ce_c,
                "acc_clean": acc_c,
                "acc_ablated": acc_a,
                "acc_delta": acc_a - acc_c,
            }
        )

    # _aggregate expects ce_replaced/acc_replaced keys; map ablated names.
    mapped = [
        {
            **r,
            "ce_replaced": r["ce_ablated"],
            "acc_replaced": r["acc_ablated"],
        }
        for r in records
    ]
    return {
        "condition": getattr(condition, "condition", "?"),
        "theta_star": theta_star_json(theta_star),
        "n_ablation_latents": len(J),
        "per_domain": _aggregate_by_domain(mapped),
        "overall": _aggregate(mapped),
        "examples": records,
    }


def save_ablation_set(J: list[tuple[int, int]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"latents": [[b, l] for b, l in J]}, indent=2))


def load_ablation_set(path: str | Path) -> list[tuple[int, int]]:
    return [(int(b), int(l)) for b, l in json.loads(Path(path).read_text())["latents"]]
