"""Spec §3 activation capture at the frozen MoE site (language_layers[5].mlp).

Two capture paths, by design:

1. **Production path** (`instrumented_capture`): replaces `moe.moe_infer` with a
   line-identical clone that additionally records the per-slot gated
   contribution intermediate `c_te = g_te · y_te` right before the slot-sum.
   Its returned aggregate is bit-identical to the original (unit-tested), so
   the model's computation is unchanged.

2. **Reference path** (`reference_capture` + `recompute_contributions`): captures
   the MoE module *input* x, the gate outputs (S_t, g_te), and the model's own
   `moe_infer` return value (the internal routed aggregate) via a
   non-perturbing wrapper — then *independently* recomputes
   c_te = g_te · expert_e(x_t) with its own expert grouping (no reliance on the
   model's argsort/scatter scheduling). The spec §3 FP32 gate compares
   Σ_slots c_te from this independent path against the model's own aggregate;
   the production path is cross-checked against the reference path in the same
   run and thereby inherits the validation.

Token order: the MoE flattens [batch, seq, d] row-major; all captured tensors
are indexed by flattened token position. The driver owns the mapping back to
(sequence, position) and the visual/text tagging.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch

from moe_sae.vendor.deepseek_vl2.modeling_deepseek import DeepseekV2MoE


# --------------------------------------------------------------------------- #
# Production path: instrumented moe_infer                                      #
# --------------------------------------------------------------------------- #


@dataclass
class CaptureChunk:
    """Per-forward capture at the frozen site (flattened token order)."""

    topk_idx: torch.Tensor  # [N, 6] int64
    topk_weight: torch.Tensor  # [N, 6] float32
    per_slot_contrib: torch.Tensor  # [N, 6, d] (store_dtype)


class CaptureSink:
    """Accumulates one CaptureChunk per MoE forward; drained by the driver."""

    def __init__(self, store_dtype: torch.dtype = torch.bfloat16, device: str = "cpu"):
        self.store_dtype = store_dtype
        self.device = device
        self.chunks: list[CaptureChunk] = []

    def _append(
        self,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
        per_slot: torch.Tensor,
    ) -> None:
        self.chunks.append(
            CaptureChunk(
                topk_idx=topk_idx.detach().to(self.device),
                topk_weight=topk_weight.detach().to(torch.float32).to(self.device),
                per_slot_contrib=per_slot.detach().to(self.store_dtype).to(self.device),
            )
        )

    def drain(self) -> list[CaptureChunk]:
        out, self.chunks = self.chunks, []
        return out


def _moe_infer_instrumented(
    moe: DeepseekV2MoE,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    sink: CaptureSink,
) -> torch.Tensor:
    """Line-identical clone of vendored DeepseekV2MoE.moe_infer (ep_size == 1),
    plus capture of the per-slot gated contributions before the slot-sum.

    Any edit to the vendored moe_infer requires re-verifying bit-identity
    (tests/test_capture_hooks.py).
    """
    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(moe.experts)))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        expert = moe.experts[i]  # ep_rank == 0, asserted at entry
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = expert(tokens_for_this_expert)
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    # Original: .view(...).type(topk_weight.dtype).mul_(topk_weight.unsqueeze(-1))
    #           .sum(dim=1).type(new_x.dtype)
    # Identical values; the [N, 6, d] gated intermediate is kept for capture.
    per_slot = new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype) * topk_weight.unsqueeze(
        dim=-1
    )
    final_out = per_slot.sum(dim=1).type(new_x.dtype)
    sink._append(topk_ids, topk_weight, per_slot)
    return final_out


@contextmanager
def instrumented_capture(
    moe: DeepseekV2MoE,
    store_dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> Iterator[CaptureSink]:
    """Production capture context. While active, every forward through `moe`
    records a CaptureChunk into the yielded sink."""
    assert moe.ep_size == 1, "Expert parallelism must be disabled for capture."
    assert "moe_infer" not in moe.__dict__, "instrumented_capture cannot be nested."
    sink = CaptureSink(store_dtype=store_dtype, device=device)

    def patched(x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor):
        with torch.no_grad():
            return _moe_infer_instrumented(moe, x, topk_ids, topk_weight, sink)

    moe.moe_infer = patched  # instance attribute shadows the class method
    try:
        yield sink
    finally:
        del moe.__dict__["moe_infer"]


# --------------------------------------------------------------------------- #
# Reference path: non-perturbing hooks + independent recomputation             #
# --------------------------------------------------------------------------- #


@dataclass
class ReferenceSink:
    """Per-forward raw material for the independent recomputation."""

    x: list = field(default_factory=list)  # [N, d] module input (flattened)
    topk_idx: list = field(default_factory=list)  # [N, 6] int64
    topk_weight: list = field(default_factory=list)  # [N, 6] gate dtype (fp32)
    aggregate: list = field(default_factory=list)  # [N, d] model's own routed sum

    def drain(self) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        n = len(self.x)
        assert len(self.topk_idx) == len(self.aggregate) == n
        out = list(zip(self.x, self.topk_idx, self.topk_weight, self.aggregate))
        self.x, self.topk_idx, self.topk_weight, self.aggregate = [], [], [], []
        return out


@contextmanager
def reference_capture(moe: DeepseekV2MoE) -> Iterator[ReferenceSink]:
    """Capture (x, S_t, g_te, model aggregate) without altering computation."""
    assert moe.ep_size == 1
    assert "moe_infer" not in moe.__dict__, "cannot nest with instrumented_capture."
    sink = ReferenceSink()

    def pre_hook(module, args):
        hidden = args[0]
        sink.x.append(hidden.detach().reshape(-1, hidden.shape[-1]))

    def gate_hook(module, args, output):
        topk_idx, topk_weight, _aux = output
        sink.topk_idx.append(topk_idx.detach())
        sink.topk_weight.append(topk_weight.detach())

    cls_moe_infer = type(moe).moe_infer

    def wrapped(x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor):
        out = cls_moe_infer(moe, x, topk_ids, topk_weight)
        sink.aggregate.append(out.detach())
        return out

    h_pre = moe.register_forward_pre_hook(pre_hook)
    h_gate = moe.gate.register_forward_hook(gate_hook)
    moe.moe_infer = wrapped
    try:
        yield sink
    finally:
        h_pre.remove()
        h_gate.remove()
        del moe.__dict__["moe_infer"]


@torch.no_grad()
def recompute_contributions(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    moe: DeepseekV2MoE,
) -> torch.Tensor:
    """Independently recompute per-slot gated contributions c_te = g_te·y_te.

    Groups tokens per expert by boolean mask (token-ascending order) — a
    scheduling entirely independent of moe_infer's argsort/scatter path. The
    gate multiply is done in the gate dtype (FP32), matching spec §3's
    definition of c_te.

    Returns [N, 6, d] in topk_weight.dtype (FP32).
    """
    n_tok, k = topk_idx.shape
    d = x.shape[-1]
    per_slot = torch.zeros(n_tok, k, d, dtype=topk_weight.dtype, device=x.device)
    for e in range(len(moe.experts)):
        mask = topk_idx == e  # [N, 6]
        if not mask.any():
            continue
        tok, slot = mask.nonzero(as_tuple=True)
        y = moe.experts[e](x[tok])
        per_slot[tok, slot] = y.to(topk_weight.dtype) * topk_weight[tok, slot].unsqueeze(-1)
    return per_slot
