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

Metrics per example vs a clean baseline (A8, v0.1.2 committee extension):
- answer-token CROSS-ENTROPY is teacher-forced (unchanged);
- TASK ACCURACY is measured under greedy generation: the prompt is prefilled
  WITH the replacement applied at the frozen matched-position mask (the mask
  covers only prompt/visual, non-BOS positions, so generated tokens are
  excluded from replacement by construction), generation runs greedily to
  EOS/limit, and accuracy is exact match of the normalized generated text
  against the normalized answer_text.

Generation mechanics (decision, recorded): implemented as full re-forwards of
prompt+generated-so-far with the FIXED prefill delta applied at the frozen
mask (padded False over generated positions). For a causal model with a fixed
additive delta confined to prompt positions this is mathematically identical
to prefill-with-hook + KV-cached decoding — the prompt positions' states are
unchanged across steps and generated tokens attend to exactly the replaced
prefill states — while keeping ONE code path that also runs on cache-less
test fixtures. `run_fidelity` therefore requires a decode-capable
processor/tokenizer (generation accuracy is text exact-match).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F

from moe_sae.capture.hooks import instrumented_capture
from moe_sae.eval.reconstruction import ThresholdedSAE, theta_star_json

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
    return forward_logits_ids(model, ex, ex.input_ids, device)


def forward_logits_ids(
    model: torch.nn.Module, ex, input_ids: torch.Tensor, device: str
) -> torch.Tensor:
    """Forward `input_ids` (a prefix of ex.input_ids possibly extended with
    generated tokens; batch 1) → logits [T, V]. Visual kwargs are sliced/
    padded to the id length (visual positions live in the prompt)."""
    t = input_ids.shape[0]
    kwargs = {"input_ids": input_ids.unsqueeze(0).to(device)}
    if getattr(ex, "images", None) is not None:
        seq_mask = ex.images_seq_mask
        if seq_mask.shape[0] >= t:
            seq_mask = seq_mask[:t]
        else:
            seq_mask = torch.cat(
                [seq_mask, torch.zeros(t - seq_mask.shape[0], dtype=seq_mask.dtype)]
            )
        kwargs["images"] = ex.images.unsqueeze(0).to(device)
        kwargs["images_seq_mask"] = seq_mask.unsqueeze(0).to(device)
        kwargs["images_spatial_crop"] = ex.images_spatial_crop.unsqueeze(0).to(device)
    out = model(**kwargs)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return logits[0]


def normalize_answer(text: str) -> str:
    """A8 exact-match normalization: lowercase, strip, collapse whitespace."""
    return " ".join(str(text).strip().lower().split())


def prompt_length(answer_mask: torch.Tensor) -> int:
    """Number of prompt positions = first answer-span position (the answer is
    a trailing span by construction; spec §9/§13)."""
    pos = answer_mask.to(torch.bool).nonzero(as_tuple=True)[0]
    assert pos.numel() > 0, "example has no answer span"
    return int(pos[0])


@torch.no_grad()
def greedy_generate_with_replacement(
    model: torch.nn.Module,
    ex,
    delta: torch.Tensor,  # [T, d] — prefill replacement delta (rows ≥ prompt ignored)
    mask: torch.Tensor,  # [T] bool — frozen replacement mask (rows ≥ prompt ignored)
    moe: torch.nn.Module,
    device: str,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> list[int]:
    """A8 greedy generation. Per step the replacement applies ONLY at the
    frozen matched-position mask over the PROMPT (visual + prompt-text,
    non-BOS); generated positions are excluded by construction — their mask
    entries are always False and their delta rows always zero. See module
    docstring for the KV-cache equivalence of the re-forward implementation."""
    plen = prompt_length(ex.answer_mask)
    ids = ex.input_ids[:plen].clone()
    prompt_mask = mask[:plen].clone().cpu()
    prompt_delta = delta[:plen].clone().cpu()
    d = delta.shape[1]
    generated: list[int] = []
    for _ in range(max_new_tokens):
        pad = ids.shape[0] - plen
        full_mask = torch.cat([prompt_mask, torch.zeros(pad, dtype=torch.bool)])
        full_delta = torch.cat(
            [prompt_delta, torch.zeros(pad, d, dtype=prompt_delta.dtype)]
        )
        with add_to_moe_output(moe, full_delta.to(device), full_mask.to(device)):
            logits = forward_logits_ids(model, ex, ids, device)
        nxt = int(logits[-1].argmax())
        generated.append(nxt)
        if eos_token_id is not None and nxt == eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([nxt], dtype=ids.dtype)])
    return generated


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
    max_new_tokens: int | None = None,
) -> dict:
    """Downstream replacement evaluation (spec §13; A8 evaluation pins).
    Per example (batch 1):

    1. one clean forward with `instrumented_capture` → baseline logits AND the
       per-slot contributions c_te at the frozen site (the instrumented path is
       bit-identical to the original, so the baseline is unperturbed);
    2. θ*-inference: ĉ_route(t) = Σ_slots ĉ_te (or the global recon for B/F);
    3. teacher-forced CE: a second forward with a hook adding
       (ĉ_route − c_route) at replaced positions only (swap identity, module
       docstring) — ce_clean / ce_replaced are teacher-forced (A8);
    4. task accuracy (A8): greedy generation — clean (no replacement) and
       replaced (prefill delta at the frozen mask restricted to prompt
       positions; generated tokens never replaced, by construction) — scored
       as exact match of normalized generated text vs normalized answer_text.

    `processor` supplies the tokenizer used to decode generated ids (an
    object with `.tokenizer` or itself exposing `decode`); its
    `eos_token_id`, when present, terminates generation early.
    `max_new_tokens=None` defaults per example to the answer-span token count
    plus 8.
    """
    assert mode in FIDELITY_MODES, mode
    if moe is None:
        from moe_sae.model_loading import get_target_moe

        moe = get_target_moe(model)
    tokenizer = getattr(processor, "tokenizer", processor)
    assert tokenizer is not None and hasattr(tokenizer, "decode"), (
        "run_fidelity needs a decode-capable processor/tokenizer: A8 task "
        "accuracy is exact match on generated text"
    )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

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

        # A8: teacher-forced CE (unchanged pin).
        ce_c, _tf_acc_c = answer_ce_and_acc(logits_clean, ex.input_ids, ex.answer_mask)
        ce_r, _tf_acc_r = answer_ce_and_acc(logits_rep, ex.input_ids, ex.answer_mask)

        # A8: task accuracy under greedy generation, exact-match normalized.
        n_answer = int(ex.answer_mask.to(torch.bool).sum())
        limit = max_new_tokens if max_new_tokens is not None else n_answer + 8
        zero_delta = torch.zeros_like(delta)
        no_mask = torch.zeros_like(mask)
        gen_clean = greedy_generate_with_replacement(
            model, ex, zero_delta, no_mask, moe, device, limit, eos_token_id
        )
        gen_rep = greedy_generate_with_replacement(
            model, ex, delta, mask, moe, device, limit, eos_token_id
        )

        def decode(ids: list[int]) -> str:
            try:
                return tokenizer.decode(ids, skip_special_tokens=True)
            except TypeError:  # decoders without the kwarg
                return tokenizer.decode(ids)

        want = normalize_answer(ex.answer_text)
        text_clean = normalize_answer(decode(gen_clean))
        text_rep = normalize_answer(decode(gen_rep))
        acc_c = float(text_clean == want)
        acc_r = float(text_rep == want)

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
                "generated_clean": text_clean,
                "generated_replaced": text_rep,
            }
        )

    return {
        "mode": mode,
        "condition": getattr(condition, "condition", "?"),
        "theta_star": theta_star_json(theta_star),
        "per_domain": _aggregate_by_domain(records),
        "overall": _aggregate(records),
        "examples": records,
        "notes": (
            (
                "full mode contains a designed-in train/eval distribution shift "
                "(the SAE never saw answer-position contributions; spec changelog #1)"
            )
            if mode == "full"
            else "matched-position replacement (primary; spec §13)"
        )
        + "; A8: CE teacher-forced, accuracy = greedy-generation exact match "
        "(replacement at the frozen mask during prefill only)",
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
