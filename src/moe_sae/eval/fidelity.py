"""Downstream-fidelity replacement evaluation (spec §13, changelog #1).

Replaces only the routed expert aggregate c_route(t) → ĉ_route(t); attention,
shared experts, and every other layer stay untouched.

The swap identity: the DeepseekV2MoE module's forward returns
    output_t = c_route(t) + shared_experts(x)_t
(the residual is added by the caller, the decoder layer). A forward hook that
adds Δ_t = ĉ_route(t) − c_route(t) to the module output at position t therefore
yields ĉ_route(t) + shared_experts(x)_t: the routed part is swapped, the shared
part and the residual are untouched *by construction*.

Modes (changelog #1):
- 'matched' (primary): replacement only at positions matching the SAE training
  distribution — projected visual tokens and prompt-text tokens. Answer-span
  positions and BOS keep their true activations.
- 'full' (secondary): replacement at every position, including answer spans —
  reported with the explicit train/eval distribution-shift flag.

Metrics per example vs a clean baseline forward: teacher-forced answer-token
cross-entropy delta and answer top-1 accuracy delta, aggregated per domain.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F

from moe_sae.capture.hooks import instrumented_capture
from moe_sae.eval.reconstruction import ThresholdedSAE

FIDELITY_MODES = ("matched", "full")


def replacement_mask(answer_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Replaced-position mask [T] (spec §13). matched = NOT answer and NOT BOS
    (position 0); full = every position."""
    assert mode in FIDELITY_MODES, mode
    answer_mask = answer_mask.to(torch.bool)
    if mode == "full":
        return torch.ones_like(answer_mask)
    m = ~answer_mask
    m[0] = False  # BOS keeps its true activation in matched mode
    return m


@contextmanager
def add_to_moe_output(
    moe: torch.nn.Module, delta: torch.Tensor, positions: torch.Tensor
) -> Iterator[None]:
    """Forward hook: out[t] += delta[t] for positions[t] True, batch size 1.

    The output is cloned and only the selected rows are modified, so
    non-selected positions are bit-identical to the unhooked forward (the
    no-injection guarantees of spec §13/§17 are testable at bit level).
    """

    def hook(module, args, output):
        out = output.clone()
        flat = out.view(-1, out.shape[-1])
        assert flat.shape[0] == positions.shape[0], (
            f"MoE output has {flat.shape[0]} positions, mask has "
            f"{positions.shape[0]} — batch size must be 1"
        )
        idx = positions.nonzero(as_tuple=True)[0]
        if idx.numel():
            flat[idx] += delta.to(flat.dtype)[idx]
        return out

    handle = moe.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def forward_logits(model: torch.nn.Module, ex, device: str) -> torch.Tensor:
    """Teacher-forced forward of one EvalExample (batch 1) → logits [T, V]."""
    kwargs = {"input_ids": ex.input_ids.unsqueeze(0).to(device)}
    if getattr(ex, "images", None) is not None:
        kwargs["images"] = ex.images.unsqueeze(0).to(device)
        kwargs["images_seq_mask"] = ex.images_seq_mask.unsqueeze(0).to(device)
        kwargs["images_spatial_crop"] = ex.images_spatial_crop.unsqueeze(0).to(device)
    out = model(**kwargs)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return logits[0]


def answer_ce_and_acc(
    logits: torch.Tensor, input_ids: torch.Tensor, answer_mask: torch.Tensor
) -> tuple[float, float]:
    """Teacher-forced answer metrics: mean cross-entropy of logits[t−1]
    predicting input_ids[t] over answer positions t, and top-1 accuracy."""
    pos = answer_mask.to(torch.bool).nonzero(as_tuple=True)[0]
    pos = pos[pos > 0]  # position 0 has no preceding logit
    assert pos.numel() > 0, "example has no scorable answer positions"
    pred = logits[pos - 1].to(torch.float32)
    targets = input_ids[pos].to(torch.int64)
    ce = F.cross_entropy(pred, targets, reduction="mean")
    acc = (pred.argmax(dim=-1) == targets).to(torch.float32).mean()
    return float(ce), float(acc)


@torch.no_grad()
def run_fidelity(
    model: torch.nn.Module,
    processor,
    condition: ThresholdedSAE,
    theta_star: float,
    eval_examples: Iterable,
    mode: str = "matched",
    device: str = "cpu",
    moe: torch.nn.Module | None = None,
) -> dict:
    """Downstream replacement evaluation (spec §13). Per example (batch 1):

    1. one clean forward with `instrumented_capture` → baseline logits AND the
       per-slot contributions c_te at the frozen site (the instrumented path is
       bit-identical to the original, so the baseline is unperturbed);
    2. θ*-inference: ĉ_route(t) = Σ_slots ĉ_te (or the global recon for B/F);
    3. a second forward with a hook adding (ĉ_route − c_route) at replaced
       positions only (see module docstring for the swap identity).

    `processor` is unused here (EvalExamples arrive pre-processed) but kept in
    the signature for parity with the capture/eval drivers.
    """
    assert mode in FIDELITY_MODES, mode
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
        assert c.shape[0] == ex.input_ids.shape[0], (
            "capture length != sequence length — is kv-cache generation on?"
        )

        ex_ids = torch.full((c.shape[0],), int(ex.example_id), dtype=torch.int64)
        enc = condition.encode_thresholded(
            c.to(device), ch.topk_idx.to(device), ex_ids.to(device), theta_star
        )
        recon_route = enc.recon_slots.to(torch.float32).sum(dim=1).cpu()  # [T, d]
        delta = recon_route - c.sum(dim=1)  # ĉ_route − c_route

        mask = replacement_mask(ex.answer_mask, mode).cpu()
        with add_to_moe_output(moe, delta.to(device), mask.to(device)):
            logits_rep = forward_logits(model, ex, device).to(torch.float32).cpu()

        ce_c, acc_c = answer_ce_and_acc(logits_clean, ex.input_ids, ex.answer_mask)
        ce_r, acc_r = answer_ce_and_acc(logits_rep, ex.input_ids, ex.answer_mask)
        records.append(
            {
                "example_id": int(ex.example_id),
                "domain": ex.domain,
                "n_replaced": int(mask.sum()),
                "ce_clean": ce_c,
                "ce_replaced": ce_r,
                "ce_delta": ce_r - ce_c,
                "acc_clean": acc_c,
                "acc_replaced": acc_r,
                "acc_delta": acc_r - acc_c,
            }
        )

    return {
        "mode": mode,
        "condition": getattr(condition, "condition", "?"),
        "theta_star": float(theta_star),
        "per_domain": _aggregate_by_domain(records),
        "overall": _aggregate(records),
        "examples": records,
        "notes": (
            "full mode contains a designed-in train/eval distribution shift "
            "(the SAE never saw answer-position contributions; spec changelog #1)"
        )
        if mode == "full"
        else "matched-position replacement (primary; spec §13)",
    }


def _aggregate(records: list[dict]) -> dict:
    n = len(records)
    keys = ("ce_clean", "ce_replaced", "ce_delta", "acc_clean", "acc_replaced", "acc_delta")
    out = {"n": n}
    for k in keys:
        out[k] = sum(r[k] for r in records) / n if n else None
    return out


def _aggregate_by_domain(records: list[dict]) -> dict:
    domains = sorted({r["domain"] for r in records})
    return {d: _aggregate([r for r in records if r["domain"] == d]) for d in domains}
