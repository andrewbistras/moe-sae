"""Frozen BatchTopK selection engine (spec §6) — pure functions, FP32 math.

All scores are upcast to float32 before comparison so selection is
deterministic across backends (docs/INTERFACES.md §Numerics). Selection
operates on ReLU'd pre-activations; because exactly K candidates are always
taken, zero-valued candidates may be selected when fewer than K are positive —
they contribute z = a = 0 to the reconstruction, which is harmless, and this
keeps the batch threshold θ_b well defined for every batch.

θ_b conventions per condition (docs/INTERFACES.md §Conditions):
A/B/C/F — the k̄B-th largest candidate score (`global_batch_topk`);
D — smallest selected score across the union of per-bank selections;
E — mean of per-token k-th-largest thresholds (log-only convention).
"""

from __future__ import annotations

import torch

from moe_sae.train.config import THETA_EMA_DECAY


def global_batch_topk(scores: torch.Tensor, k_total: int) -> tuple[torch.Tensor, float]:
    """Select exactly k_total candidates over the *entire* flattened set.

    There is deliberately no per-token or per-bank constraint (spec §6: the
    allocation is the phenomenon under investigation).

    scores: [n] (any float dtype; upcast to fp32). Returns (mask [n] bool,
    θ_b = value of the k_total-th largest score).

    Backend note: MPS's topk kernel raises an internal index-out-of-bounds on
    large inputs (observed at n = 25,165,824 — the pipeline-scale candidate
    set, 2026-07-07). The selection is computed on CPU for MPS tensors: same
    fp32 math, ~100 ms at pipeline scale; CUDA/CPU paths are unchanged.
    """
    s = scores.float()
    assert s.ndim == 1 and 0 < k_total <= s.numel(), (s.shape, k_total)
    if s.device.type == "mps":
        s_sel = s.cpu()
        vals, idx = torch.topk(s_sel, k_total, sorted=True)
        mask = torch.zeros_like(s_sel, dtype=torch.bool)
        mask[idx] = True
        return mask.to(scores.device), float(vals[-1].item())
    vals, idx = torch.topk(s, k_total, sorted=True)
    mask = torch.zeros_like(s, dtype=torch.bool)
    mask[idx] = True
    return mask, float(vals[-1].item())


def per_bank_topk(
    scores: torch.Tensor,
    bank_of_candidate: torch.Tensor,
    budgets: torch.Tensor,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Condition D: independent per-bank budgets, no cross-bank competition.

    scores: [n] fp-any; bank_of_candidate: [n] int64; budgets: [E] int64
    (Σ budgets == k̄B, from largest-remainder rounding of this batch's loads).
    Returns (mask [n] bool, θ_b = smallest selected score across the union —
    the log-only scalar convention, kept as-is — and per_bank_theta [E]
    float32: each bank's own selection threshold, the smallest selected score
    within that bank; NaN for banks with zero budget this batch — A4
    committee amendment: D's inference thresholds are per-bank θ*_e).
    """
    s = scores.float()
    assert s.shape == bank_of_candidate.shape
    mask = torch.zeros_like(s, dtype=torch.bool)
    theta = torch.inf
    per_bank_theta = torch.full(
        (budgets.shape[0],), float("nan"), dtype=torch.float32
    )
    for e in torch.unique(bank_of_candidate).tolist():
        k = int(budgets[e].item())
        if k == 0:
            continue
        rows = (bank_of_candidate == e).nonzero(as_tuple=True)[0]
        assert k <= rows.numel(), (
            f"bank {e}: budget {k} exceeds its {rows.numel()} candidates"
        )
        vals, idx = torch.topk(s[rows], k, sorted=True)
        mask[rows[idx]] = True
        per_bank_theta[e] = float(vals[-1].item())
        theta = min(theta, float(vals[-1].item()))
    assert int(mask.sum().item()) == int(budgets.sum().item())
    return mask, theta, per_bank_theta


def per_token_topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Condition E: exactly k selected per token across its visible candidates.

    scores: [B, n_cand_per_token]. Returns (mask [B, n_cand] bool,
    per-token thresholds [B] fp32 — the k-th largest score of each token).
    """
    s = scores.float()
    assert s.ndim == 2 and 0 < k <= s.shape[1], (s.shape, k)
    vals, idx = torch.topk(s, k, dim=1, sorted=True)
    mask = torch.zeros_like(s, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask, vals[:, -1]


def thresholded_inference(
    preacts: torch.Tensor, theta_star: float | torch.Tensor
) -> torch.Tensor:
    """Frozen-θ* inference (spec §6): z = a · 1[a > θ*]. Strict inequality.

    `theta_star` may be a scalar (all conditions but D) or a tensor
    broadcastable against `preacts` (condition D under A4: per-assigned-bank
    thresholds θ*_e). NaN thresholds (banks whose budget was never > 0 during
    training, so θ*_e is undefined) compare False and yield no activations.
    """
    a = preacts.float()
    if isinstance(theta_star, torch.Tensor):
        theta_star = theta_star.to(a.device, torch.float32)
    return a * (a > theta_star)


def largest_remainder_budgets(weights: torch.Tensor, total: int) -> torch.Tensor:
    """Integer budgets ∝ weights summing exactly to `total` (largest-remainder
    rounding; ties broken by lower index — deterministic).

    weights: [E] nonnegative (e.g. this batch's per-bank routed loads)."""
    w = weights.float()
    wsum = float(w.sum().item())
    assert wsum > 0, "no load to allocate against"
    quota = w * (total / wsum)
    base = torch.floor(quota).to(torch.int64)
    short = total - int(base.sum().item())
    assert 0 <= short <= w.numel()
    if short > 0:
        frac = quota - base.to(quota.dtype)
        # stable sort: equal remainders resolve to the lower bank index
        order = torch.argsort(frac, descending=True, stable=True)
        base[order[:short]] += 1
    return base


class ThetaEMA:
    """Frozen inference-threshold estimator (spec §6, changelog #6).

    θ ← decay·θ + (1−decay)·θ_b, decay = 0.999 (frozen), initialized to the
    first batch's θ_b. The final value θ* is frozen at end of training and
    used for all inference and evaluation.
    """

    def __init__(self, decay: float = THETA_EMA_DECAY):
        self.decay = decay
        self.value: float | None = None

    def update(self, theta_b: float) -> None:
        if self.value is None:
            self.value = float(theta_b)
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * float(theta_b)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "value": self.value}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.value = None if state["value"] is None else float(state["value"])


class ThetaEMAVector:
    """Per-bank inference-threshold estimator for condition D (A4, v0.1.2
    committee amendment, Reviewer F).

    One EMA(0.999) per bank over that bank's own per-batch selection
    threshold (`per_bank_topk`'s per-bank θ). Each bank's EMA initializes at
    the FIRST batch that updates that bank (budget > 0); banks never updated
    stay None (stored as NaN). The frozen vector θ*_e is used for all D
    inference: z = a·1[a > θ*_{bank}].

    D trains under per-bank budgets, so per-bank selection scales diverge by
    construction; a single global θ* re-imposed at evaluation would
    systematically mis-threshold low-scale banks (A4 reason).
    """

    def __init__(self, decay: float = THETA_EMA_DECAY, n_banks: int = 64):
        self.decay = decay
        self.n_banks = n_banks
        # NaN encodes "never updated" (None semantics per bank).
        self.value = torch.full((n_banks,), float("nan"), dtype=torch.float64)

    def update(self, per_bank_theta: torch.Tensor) -> None:
        """per_bank_theta: [n_banks] float, NaN for banks with no budget this
        batch (those banks are left untouched)."""
        t = per_bank_theta.to(torch.float64)
        assert t.shape == (self.n_banks,), t.shape
        seen = ~torch.isnan(t)
        init = seen & torch.isnan(self.value)  # per-bank init at first update
        upd = seen & ~torch.isnan(self.value)
        self.value[init] = t[init]
        self.value[upd] = self.decay * self.value[upd] + (1.0 - self.decay) * t[upd]

    def values_list(self) -> list[float | None]:
        """JSON-ready θ* vector (None for never-updated banks)."""
        return [None if torch.isnan(v) else float(v) for v in self.value]

    def state_dict(self) -> dict:
        return {"decay": self.decay, "value": self.value.clone()}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.value = state["value"].to(torch.float64).clone()
        self.n_banks = self.value.shape[0]
