"""Capture driver: examples → prompt-only forward → per-token records (spec §3, §9).

Per example (batch size 1): build the prompt-only conversation, run the
processor, embed via ``model.prepare_inputs_embeds``, and forward
``model.language.model`` under ``instrumented_capture`` on the frozen target
MoE (language layer index 5). A forward hook on that layer raises the private
``StopForward`` exception, which the driver catches — layers 6–11 are skipped.
This is purely a compute saving: the transformer is causal and strictly
feed-forward across layers, so activations at layer 5 are unaffected by
whether later layers run (the hook fires *after* layer 5 completes).

Token bookkeeping: with batch size 1 the MoE flattens [1, T, d] row-major, so
flattened chunk row i == sequence position i (asserted). Position 0 (BOS) is
dropped. Visual rows = ``images_seq_mask`` positions; text rows = the rest.
Visual rows are subsampled to ``visual_cap_per_example`` without replacement,
seeded per example by ``f"{seed}:{example_id}"`` (string seeding is stable
across processes/platforms), and written in ascending position order. Only
prompt tokens exist here — answer-generation tokens are never captured
(spec §9); ``tokenize_capture`` builds generation-ready prompts only.

``text_control=True`` (spec §13 baseline): the same prompts WITHOUT the image
(no <image> token enters the sequence); every non-BOS row goes to the text
pool with domain=255. Text-control captures live in their own capture root and
are never mixed into training captures.

This module is import-safe without the model; model loading happens only in
scripts/run_capture.py.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Iterable

import torch

from moe_sae.capture.hooks import instrumented_capture
from moe_sae.capture.writer import DOMAINS, CaptureMeta, ShardWriter
from moe_sae.data.prompts import tokenize_capture
from moe_sae.data.streams import Example
from moe_sae.model_loading import MODEL_ID, TARGET_LAYER_ZERO_INDEXED

logger = logging.getLogger(__name__)

SPEC_VERSION = "v0.1.1"


class StopForward(Exception):
    """Private early-exit signal raised by the layer-5 forward hook."""


@torch.no_grad()
def capture_examples(
    model,
    processor,
    examples_iter: Iterable[Example],
    out_root: str | Path,
    visual_cap_per_example: int,
    seed: int,
    max_seq_len: int = 2048,
    text_control: bool = False,
    shard_size: int = 16384,
    log_every: int = 50,
) -> CaptureMeta:
    """Run capture over `examples_iter`, writing pools + meta.json to out_root."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    language = model.language.model  # DeepseekV2Model
    target_layer = language.layers[TARGET_LAYER_ZERO_INDEXED]
    moe = target_layer.mlp

    writers = {"text": ShardWriter(out_root, "text", shard_size=shard_size)}
    if not text_control:
        writers["visual"] = ShardWriter(out_root, "visual", shard_size=shard_size)

    def _early_exit(_module, _args, _output):
        raise StopForward

    n_examples = 0
    n_skipped_long = 0
    domain_counts: dict[str, int] = {}
    t0 = time.monotonic()

    hook = target_layer.register_forward_hook(_early_exit)
    try:
        with instrumented_capture(moe, store_dtype=torch.bfloat16, device="cpu") as sink:
            for ex in examples_iter:
                if text_control:
                    assert ex.image is None and ex.domain == "text_control", (
                        "text_control capture requires the image-free control stream"
                    )
                else:
                    assert ex.image is not None, (
                        f"example {ex.example_id} has no image; training captures "
                        "require all four multimodal domains (spec §9)"
                    )

                tok = tokenize_capture(processor, ex)
                T = tok.seq_len
                if T > max_seq_len:
                    n_skipped_long += 1
                    logger.warning(
                        "skipping example %d (%s): %d tokens > max_seq_len=%d",
                        ex.example_id,
                        ex.domain,
                        T,
                        max_seq_len,
                    )
                    continue

                input_ids = tok.input_ids.unsqueeze(0).to(device)
                if tok.images is not None:
                    inputs_embeds = model.prepare_inputs_embeds(
                        input_ids=input_ids,
                        images=tok.images.unsqueeze(0).to(device=device, dtype=model_dtype),
                        images_seq_mask=tok.images_seq_mask.unsqueeze(0).to(device),
                        images_spatial_crop=tok.images_spatial_crop.unsqueeze(0).to(device),
                    )
                else:
                    inputs_embeds = model.language.get_input_embeddings()(input_ids)

                try:
                    language(inputs_embeds=inputs_embeds, use_cache=False)
                except StopForward:
                    pass

                chunks = sink.drain()
                assert len(chunks) == 1, f"expected 1 capture chunk, got {len(chunks)}"
                chunk = chunks[0]
                assert chunk.per_slot_contrib.shape[0] == T, (
                    "flattened chunk rows must equal sequence positions "
                    f"({chunk.per_slot_contrib.shape[0]} vs {T})"
                )

                visual_mask = tok.visual_mask  # [T] bool, cpu
                assert not visual_mask[0], "position 0 (BOS) cannot be visual"
                positions = torch.arange(T)
                if text_control:
                    assert not visual_mask.any(), "text_control sequence has visual tokens"
                    rows = {"text": positions[1:]}  # drop BOS
                    domain_code = DOMAINS["text_control"]
                else:
                    visual_rows = positions[visual_mask]
                    text_rows = positions[(~visual_mask) & (positions != 0)]
                    if visual_rows.shape[0] > visual_cap_per_example:
                        rng = random.Random(f"{seed}:{ex.example_id}")
                        keep = sorted(
                            rng.sample(range(visual_rows.shape[0]), visual_cap_per_example)
                        )
                        visual_rows = visual_rows[torch.tensor(keep, dtype=torch.long)]
                    rows = {"visual": visual_rows, "text": text_rows}
                    domain_code = DOMAINS[ex.domain]

                for pool, idx in rows.items():
                    n = idx.shape[0]
                    if n == 0:
                        continue
                    writers[pool].add(
                        c=chunk.per_slot_contrib[idx],
                        experts=chunk.topk_idx[idx],
                        gates=chunk.topk_weight[idx],
                        example_id=torch.full((n,), ex.example_id, dtype=torch.int32),
                        domain=torch.full((n,), domain_code, dtype=torch.uint8),
                    )

                n_examples += 1
                domain_counts[ex.domain] = domain_counts.get(ex.domain, 0) + 1
                if n_examples % log_every == 0:
                    written = {p: w.total_written + w._buffered for p, w in writers.items()}
                    dt = time.monotonic() - t0
                    logger.info(
                        "%d examples | tokens %s | %.1f tok/s",
                        n_examples,
                        written,
                        sum(written.values()) / max(dt, 1e-9),
                    )
    finally:
        hook.remove()

    totals = {pool: w.close() for pool, w in writers.items()}
    meta = CaptureMeta(
        name=out_root.name,
        model_id=MODEL_ID,
        spec_version=SPEC_VERSION,
        target_layer_zero_indexed=TARGET_LAYER_ZERO_INDEXED,
        tokens={"visual": totals.get("visual", 0), "text": totals.get("text", 0)},
        examples=n_examples,
        config={
            "visual_cap_per_example": visual_cap_per_example,
            "seed": seed,
            "max_seq_len": max_seq_len,
            "text_control": text_control,
            "shard_size": shard_size,
            "skipped_too_long": n_skipped_long,
            "domain_example_counts": domain_counts,
        },
    )
    meta.save(out_root)
    logger.info(
        "capture done: %d examples (%d skipped >%d tokens), tokens=%s → %s",
        n_examples,
        n_skipped_long,
        max_seq_len,
        meta.tokens,
        out_root,
    )
    return meta
