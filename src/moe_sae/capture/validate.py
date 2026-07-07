"""Spec §3 hook-validation gate (§20 step 1, HARD GATE) + BF16 production report.

Two phases, run sequentially in one process with explicit memory release:

1. **FP32 GATE** — a memory-streamed pure-FP32 forward through the target layer
   (capture/streamed.py). At layer 5's MoE, `reference_capture` records the
   module input x, the gate outputs (S_t, g_te), and the model's own routed
   aggregate; `recompute_contributions` then *independently* recomputes
   c_te = g_te·expert_e(x_t). Per token,
   rel_err = ‖Σ_slots c_te − aggregate_t‖₂ / ‖aggregate_t‖₂ must be < 1e-5
   for EVERY token (zero-norm aggregates are reported separately, never
   divided). The production capture path (`instrumented_capture`) is
   cross-checked against the reference recompute on the same layer-5 input to
   ≤ 1e-5 rel per token, so the shard-writing path inherits the validation.

2. **BF16 REPORT** — the full model is loaded normally (BF16, production
   path) and the same inputs are re-run with `reference_capture` at layer 5.
   The same per-token rel_err, computed in FP32 arithmetic, is *reported*
   (max/median/p99), NOT gated: fused/grouped expert kernels reduce in
   implementation-defined orders (spec §3). Systematic structure is probed via
   Spearman correlations of the per-token error with max |gate|, token
   position, and visual-vs-text type; |ρ| > 0.3 is flagged for investigation.

Validation inputs: `examples_{i:04d}.pt` dicts produced by
`moe_sae.data.evalsets.export_validation_inputs` (docs/INTERFACES.md).

Decisions:
- Token budget: the §3 gate is requested over `n_tokens` tokens; a PASS on
  fewer accumulated tokens (inputs dir too small) is NOT a PASS. In that case
  the overall `gate` is reported as "INSUFFICIENT_TOKENS" (never "PASS"), with
  requested vs actual counts in `token_budget`. A FAIL on insufficient tokens
  stays "FAIL" (more severe); the budget shortfall is still recorded.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch

from moe_sae.capture.hooks import (
    instrumented_capture,
    recompute_contributions,
    reference_capture,
)
from moe_sae.capture.streamed import (
    CheckpointStreamer,
    build_skeleton,
    fp32_prefix_forward,
    free_memory,
)
from moe_sae.model_loading import (
    MODEL_ID,
    TARGET_LAYER_ZERO_INDEXED,
    assert_frozen_architecture,
    get_target_moe,
    load_model,
    pick_device,
)

logger = logging.getLogger(__name__)

# FROZEN (spec §3): pure-FP32 gate threshold, per token, strict "<".
GATE_REL_TOL = 1e-5
# Contract (docs/INTERFACES.md): production capture must agree with the
# independent reference recompute to ≤ 1e-5 rel per token in FP32.
CROSSCHECK_REL_TOL = 1e-5
# BF16 report: |Spearman ρ| above this is flagged as systematic structure.
CORR_FLAG_THRESHOLD = 0.3
CORR_FLAG_MESSAGE = "systematic structure — investigate"

N_WORST_OFFENDERS = 20


# --------------------------------------------------------------------------- #
# Pure math helpers (unit-tested against naive references)                     #
# --------------------------------------------------------------------------- #


def per_token_rel_err(
    contrib_sum: torch.Tensor, aggregate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """rel_err_t = ‖Σ_slots c_te − aggregate_t‖₂ / ‖aggregate_t‖₂, FP32.

    Returns (rel [N] float32, zero_norm [N] bool). Tokens whose aggregate has
    exactly zero norm are masked (rel set to 0, zero_norm True) — reported
    separately, never divided (spec §3 guard).
    """
    a = aggregate.to(torch.float32)
    c = contrib_sum.to(torch.float32)
    assert a.shape == c.shape and a.dim() == 2, (a.shape, c.shape)
    agg_norm = a.norm(dim=-1)
    zero_norm = agg_norm == 0
    rel = torch.zeros_like(agg_norm)
    nz = ~zero_norm
    rel[nz] = (c[nz] - a[nz]).norm(dim=-1) / agg_norm[nz]
    return rel, zero_norm


def per_token_slot_rel_err(
    produced: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token relative Frobenius error over the [6, d] slot block, FP32:
    ‖prod_t − ref_t‖_F / ‖ref_t‖_F, with the same zero-norm guard."""
    p = produced.to(torch.float32)
    r = reference.to(torch.float32)
    assert p.shape == r.shape and p.dim() == 3, (p.shape, r.shape)
    ref_norm = r.flatten(1).norm(dim=-1)
    zero_norm = ref_norm == 0
    rel = torch.zeros_like(ref_norm)
    nz = ~zero_norm
    rel[nz] = (p - r).flatten(1).norm(dim=-1)[nz] / ref_norm[nz]
    return rel, zero_norm


def gate_verdict(
    rel: np.ndarray,
    tol: float = GATE_REL_TOL,
    example_id: np.ndarray | None = None,
    token_pos: np.ndarray | None = None,
    n_worst: int = N_WORST_OFFENDERS,
) -> dict:
    """Spec §3 gate: PASS iff every token's rel_err < tol (strict). The worst
    offenders (token index, example, err) are always listed for diagnosis."""
    rel = np.asarray(rel, dtype=np.float64)
    n = int(rel.size)
    violations = int((rel >= tol).sum()) if n else 0
    order = np.argsort(rel)[::-1][: min(n_worst, n)]
    worst = []
    for i in order:
        entry = {"rel_err": float(rel[i])}
        if example_id is not None:
            entry["example_id"] = int(example_id[i])
        if token_pos is not None:
            entry["token"] = int(token_pos[i])
        worst.append(entry)
    status = "PASS" if n > 0 and violations == 0 else "FAIL"
    out = {
        "status": status,
        "tol": tol,
        "n_tokens": n,
        "n_violations": violations,
        "worst": worst,
    }
    if n == 0:
        out["note"] = "no tokens validated — cannot pass an empty gate"
    return out


def token_budget_verdict(gate: str, requested: int, actual: int) -> tuple[str, dict]:
    """Spec §3 token-budget check: a PASS may not stand on fewer accumulated
    tokens than requested. Returns (gate', budget_report). A "PASS" with
    actual < requested is demoted to "INSUFFICIENT_TOKENS"; a "FAIL" stays
    "FAIL" (more severe) with the shortfall still recorded."""
    budget = {
        "requested": int(requested),
        "actual": int(actual),
        "sufficient": bool(actual >= requested),
    }
    if not budget["sufficient"]:
        budget["note"] = "validated fewer tokens than requested — gate cannot PASS"
        if gate == "PASS":
            gate = "INSUFFICIENT_TOKENS"
    return gate, budget


def summarize_rel(rel: np.ndarray) -> dict:
    rel = np.asarray(rel, dtype=np.float64)
    if rel.size == 0:
        return {"n_tokens": 0, "max": None, "median": None, "p99": None}
    return {
        "n_tokens": int(rel.size),
        "max": float(rel.max()),
        "median": float(np.median(rel)),
        "p99": float(np.percentile(rel, 99)),
    }


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rank correlation; None when undefined (constant input, n<2)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert x.shape == y.shape
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    from scipy.stats import spearmanr

    rho = spearmanr(x, y).statistic
    if np.isnan(rho):
        return None
    return float(rho)


def correlation_report(err: np.ndarray, covariates: dict[str, np.ndarray]) -> dict:
    """Spearman ρ of the per-token error against each covariate; |ρ| > 0.3
    flags systematic structure (spec §3: investigated before proceeding)."""
    out = {}
    for name, cov in covariates.items():
        rho = spearman_rho(err, cov)
        entry: dict = {"rho": rho}
        if rho is not None and abs(rho) > CORR_FLAG_THRESHOLD:
            entry["flag"] = CORR_FLAG_MESSAGE
        out[name] = entry
    return out


# --------------------------------------------------------------------------- #
# Input handling                                                               #
# --------------------------------------------------------------------------- #


def _input_files(inputs_dir: str | Path) -> list[Path]:
    files = sorted(Path(inputs_dir).glob("examples_*.pt"))
    if not files:
        raise FileNotFoundError(f"no examples_*.pt under {inputs_dir}")
    return files


def _load_example(path: Path) -> dict:
    ex = torch.load(path, map_location="cpu")
    assert "input_ids" in ex, f"{path} is not a validation-input export"
    return ex


def _domain_name(raw) -> str:
    if isinstance(raw, str):
        return raw
    from moe_sae.capture.writer import DOMAIN_NAMES

    return DOMAIN_NAMES.get(int(raw), str(int(raw)))


def _seq_len(ex: dict) -> int:
    ids = ex["input_ids"]
    return int(ids.shape[-1])


def _visual_mask(ex: dict, n: int) -> np.ndarray:
    mask = ex.get("images_seq_mask")
    if mask is None:
        return np.zeros(n, dtype=np.float64)
    return np.asarray(mask, dtype=np.float64).reshape(-1)[:n]


def _chunks(seq: Sequence, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


class _TokenAccumulator:
    """Flat per-token arrays accumulated across examples."""

    def __init__(self):
        self.fields: dict[str, list[np.ndarray]] = {}
        self.n_tokens = 0

    def add(self, **arrays: np.ndarray) -> None:
        n = None
        for k, v in arrays.items():
            v = np.asarray(v)
            n = v.size if n is None else n
            assert v.size == n, (k, v.size, n)
            self.fields.setdefault(k, []).append(v)
        self.n_tokens += int(n)

    def cat(self, key: str) -> np.ndarray:
        if key not in self.fields:
            return np.array([])
        return np.concatenate(self.fields[key])


# --------------------------------------------------------------------------- #
# Phase 1: FP32 gate (streamed reference forward)                              #
# --------------------------------------------------------------------------- #


def _run_fp32_gate_phase(
    files: list[Path],
    n_tokens: int,
    device: torch.device,
    model_id: str,
    chunk_examples: int,
) -> tuple[_TokenAccumulator, list[Path]]:
    streamer = CheckpointStreamer(model_id)
    skeleton = build_skeleton(model_id)
    assert_frozen_architecture(skeleton)  # spec §2 structure checks (meta-safe)

    acc = _TokenAccumulator()
    used_files: list[Path] = []
    records: list[dict] = []

    @contextmanager
    def _gate_capture(moe):
        """Around one example's layer-5 forward: capture, independently
        recompute, and cross-check the production path — all while the FP32
        MoE is still materialized. Reduces to per-token scalars immediately."""
        with reference_capture(moe) as sink:
            yield sink
        (x, topk_idx, topk_weight, aggregate) = sink.drain()[0]
        per_slot_ref = recompute_contributions(x, topk_idx, topk_weight, moe)
        rel, zero = per_token_rel_err(per_slot_ref.sum(dim=1), aggregate)
        # Production-path cross-check: same layer-5 input through the
        # instrumented moe_infer clone (the shard-writing path).
        with instrumented_capture(moe, store_dtype=torch.float32) as psink:
            moe.moe_infer(x, topk_idx, topk_weight)
            produced = psink.drain()[0].per_slot_contrib.to(per_slot_ref.device)
        xrel, xzero = per_token_slot_rel_err(produced, per_slot_ref)
        records.append(
            {
                "rel": rel.cpu().numpy(),
                "zero": zero.cpu().numpy(),
                "xrel": xrel.cpu().numpy(),
                "xzero": xzero.cpu().numpy(),
                "zero_contrib_norm": per_slot_ref.sum(dim=1)
                .norm(dim=-1)[zero]
                .cpu()
                .numpy(),
            }
        )

    for chunk_paths in _chunks(files, chunk_examples):
        if acc.n_tokens >= n_tokens:
            break
        chunk = [_load_example(p) for p in chunk_paths]
        records.clear()
        fp32_prefix_forward(
            streamer,
            skeleton,
            chunk,
            n_layers=TARGET_LAYER_ZERO_INDEXED + 1,
            device=device,
            on_target_moe=_gate_capture,
        )
        assert len(records) == len(chunk), "one capture record per example expected"
        for ex, rec in zip(chunk, records):
            n = rec["rel"].size
            acc.add(
                rel=rec["rel"],
                zero=rec["zero"],
                xrel=rec["xrel"],
                xzero=rec["xzero"],
                example_id=np.full(n, int(ex.get("example_id", -1)), dtype=np.int64),
                token_pos=np.arange(n, dtype=np.int64),
            )
            acc.fields.setdefault("domain", []).append(
                np.array([_domain_name(ex.get("domain", "unknown"))] * n)
            )
            acc.fields.setdefault("zero_contrib_norm", []).append(
                rec["zero_contrib_norm"]
            )
        used_files.extend(chunk_paths)
        logger.info("FP32 gate: %d/%d tokens", acc.n_tokens, n_tokens)

    del skeleton, streamer
    free_memory(device)
    return acc, used_files


# --------------------------------------------------------------------------- #
# Phase 2: BF16 production-path report                                         #
# --------------------------------------------------------------------------- #


def _run_bf16_report_phase(
    files: list[Path], device: torch.device, model_id: str
) -> _TokenAccumulator:
    model, _info = load_model(model_id, device=str(device))  # BF16 production path
    moe = get_target_moe(model)
    acc = _TokenAccumulator()

    for path in files:
        ex = _load_example(path)
        input_ids = torch.as_tensor(ex["input_ids"], dtype=torch.long)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
        images = ex.get("images")
        images_seq_mask = ex.get("images_seq_mask")
        images_spatial_crop = ex.get("images_spatial_crop")
        if images is not None and images_spatial_crop is not None:
            images = images.to(device=device, dtype=torch.bfloat16)
            if images.dim() == 4:
                images = images.unsqueeze(0)
            images_seq_mask = torch.as_tensor(images_seq_mask, dtype=torch.bool)
            if images_seq_mask.dim() == 1:
                images_seq_mask = images_seq_mask.unsqueeze(0)
            images_seq_mask = images_seq_mask.to(device)
            images_spatial_crop = torch.as_tensor(
                images_spatial_crop, dtype=torch.long
            )
            if images_spatial_crop.dim() == 2:
                images_spatial_crop = images_spatial_crop.unsqueeze(0)
            images_spatial_crop = images_spatial_crop.to(device)

        with reference_capture(moe) as sink:
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    images=images,
                    images_seq_mask=images_seq_mask,
                    images_spatial_crop=images_spatial_crop,
                    use_cache=False,
                )
        (x, topk_idx, topk_weight, aggregate) = sink.drain()[0]
        per_slot = recompute_contributions(x, topk_idx, topk_weight, moe)
        rel, zero = per_token_rel_err(per_slot.sum(dim=1), aggregate)

        n = rel.numel()
        acc.add(
            rel=rel.cpu().numpy(),
            zero=zero.cpu().numpy(),
            max_gate=topk_weight.abs().max(dim=1).values.to(torch.float32).cpu().numpy(),
            token_pos=np.arange(n, dtype=np.int64),
            visual=_visual_mask(ex, n),
            example_id=np.full(n, int(ex.get("example_id", -1)), dtype=np.int64),
        )
        acc.fields.setdefault("domain", []).append(
            np.array([_domain_name(ex.get("domain", "unknown"))] * n)
        )
        del per_slot, x, aggregate
        logger.info("BF16 report: %d tokens", acc.n_tokens)

    del model, moe
    free_memory(device)
    return acc


# --------------------------------------------------------------------------- #
# Report assembly                                                              #
# --------------------------------------------------------------------------- #


def _per_domain(
    domains: np.ndarray, rel: np.ndarray, valid: np.ndarray, prefix: str
) -> dict:
    out: dict[str, dict] = {}
    for d in sorted(set(domains.tolist())):
        m = (domains == d) & valid
        s = summarize_rel(rel[m])
        out[d] = {
            f"{prefix}_n_tokens": s["n_tokens"],
            f"{prefix}_max": s["max"],
            f"{prefix}_median": s["median"],
            f"{prefix}_p99": s["p99"],
        }
    return out


def run_validation(
    inputs_dir: str | Path,
    out_path: str | Path,
    n_tokens: int = 100_000,
    device: str | None = None,
    model_id: str = MODEL_ID,
    chunk_examples: int = 8,
    bf16_report: bool = True,
) -> dict:
    """Run both phases and write the report JSON. Returns the report dict.

    Gate semantics (decision, see docs): overall `gate` is "PASS" iff ALL of
    the spec §3 FP32 criterion (every token rel < 1e-5), the production-path
    cross-check (every token ≤ 1e-5), AND the token budget (accumulated tokens
    ≥ `n_tokens`) hold — a failing cross-check means the shard-writing capture
    path is invalid, and a budget shortfall means the gate never saw the
    requested sample; either must block training. A budget shortfall alone is
    reported as gate "INSUFFICIENT_TOKENS" (see module Decisions).
    """
    dev = pick_device(device)
    files = _input_files(inputs_dir)

    # ---- Phase 1: FP32 gate ------------------------------------------------ #
    logger.info("Phase 1 (FP32 gate) on %s: target %d tokens", dev, n_tokens)
    acc, used_files = _run_fp32_gate_phase(files, n_tokens, dev, model_id, chunk_examples)

    rel = acc.cat("rel")
    zero = acc.cat("zero").astype(bool)
    xrel = acc.cat("xrel")
    xzero = acc.cat("xzero").astype(bool)
    example_id = acc.cat("example_id")
    token_pos = acc.cat("token_pos")
    domains = acc.cat("domain")
    nz = ~zero

    fp32_verdict = gate_verdict(
        rel[nz], GATE_REL_TOL, example_id[nz], token_pos[nz]
    )
    xnz = ~xzero
    x_max = float(xrel[xnz].max()) if xnz.any() else None
    crosscheck_pass = bool(xnz.any() and x_max <= CROSSCHECK_REL_TOL)
    fp32_summary = summarize_rel(rel[nz])
    zero_contrib = acc.cat("zero_contrib_norm")

    report: dict = {
        "spec": "frozen_spec_v0.1.1 §3 / §20 step 1",
        "model_id": model_id,
        "target_layer_zero_indexed": TARGET_LAYER_ZERO_INDEXED,
        "device": str(dev),
        "n_examples": len(used_files),
        "fp32": {
            "tol": GATE_REL_TOL,
            "n_tokens": fp32_summary["n_tokens"],
            "max_rel": fp32_summary["max"],
            "median_rel": fp32_summary["median"],
            "status": fp32_verdict["status"],
            "n_violations": fp32_verdict["n_violations"],
            "worst": fp32_verdict["worst"],
            "zero_norm_aggregates": {
                "n": int(zero.sum()),
                "max_contrib_norm": float(zero_contrib.max())
                if zero_contrib.size
                else None,
            },
        },
        "production_crosscheck": {
            "tol": CROSSCHECK_REL_TOL,
            "n_tokens": int(xnz.sum()),
            "max_rel": x_max,
            "median_rel": float(np.median(xrel[xnz])) if xnz.any() else None,
            "pass": crosscheck_pass,
        },
    }
    report["gate"] = (
        "PASS" if fp32_verdict["status"] == "PASS" and crosscheck_pass else "FAIL"
    )
    # Spec §3 token budget: PASS must not stand on fewer tokens than requested.
    report["gate"], report["token_budget"] = token_budget_verdict(
        report["gate"], n_tokens, int(rel.size)
    )
    per_domain = _per_domain(domains, rel, nz, "fp32")

    # ---- Phase 2: BF16 report (never a gate, spec §3) ----------------------- #
    if bf16_report:
        logger.info("Phase 2 (BF16 report) on %s: %d examples", dev, len(used_files))
        bacc = _run_bf16_report_phase(used_files, dev, model_id)
        brel = bacc.cat("rel")
        bzero = bacc.cat("zero").astype(bool)
        bnz = ~bzero
        bsummary = summarize_rel(brel[bnz])
        correlations = correlation_report(
            brel[bnz],
            {
                "max_gate": bacc.cat("max_gate")[bnz],
                "token_position": bacc.cat("token_pos")[bnz].astype(np.float64),
                "visual_vs_text": bacc.cat("visual")[bnz],
            },
        )
        report["bf16"] = {
            "n_tokens": bsummary["n_tokens"],
            "max": bsummary["max"],
            "median": bsummary["median"],
            "p99": bsummary["p99"],
            "n_zero_norm": int(bzero.sum()),
            "correlations": correlations,
        }
        bdomains = bacc.cat("domain")
        bper = _per_domain(bdomains, brel, bnz, "bf16")
        for d, vals in bper.items():
            per_domain.setdefault(d, {}).update(vals)

    report["per_domain"] = per_domain

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Report written to %s — gate: %s", out_path, report["gate"])
    return report
