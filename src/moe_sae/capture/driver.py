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

Resilience (interrupt/sleep/crash safety):
- Per-example failures (bad images, processor edge cases) are logged and
  skipped, never fatal — unless the failure rate exceeds 5% of consumed
  examples (systematic problem → abort loudly).
- Progress is checkpointed to ``progress.json`` every ``checkpoint_every``
  consumed examples: writers flush their buffers to (possibly small) shards
  FIRST, then the progress record is written atomically (tmp+rename). A killed
  run therefore never loses more than one checkpoint window, and duplicates
  are impossible: on ``resume=True`` any stray shards beyond the recorded
  count (the un-checkpointed window) are deleted before continuing, and the
  deterministic example stream is fast-forwarded by ``stream_consumed``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import torch

from moe_sae.capture.hooks import instrumented_capture
from moe_sae.capture.writer import DOMAINS, CaptureMeta, ShardWriter, shard_paths
from moe_sae.data.prompts import tokenize_capture
from moe_sae.data.streams import Example
from moe_sae.model_loading import MODEL_ID, TARGET_LAYER_ZERO_INDEXED

logger = logging.getLogger(__name__)

SPEC_VERSION = "v0.1.1"

PROGRESS_FILE = "progress.json"


class StopForward(Exception):
    """Private early-exit signal raised by the layer-5 forward hook."""


@dataclass
class CaptureProgress:
    """Resume state; on disk it exactly matches the flushed shards."""

    stream_consumed: int = 0  # examples pulled from the stream (any outcome)
    captured: int = 0
    skipped_too_long: int = 0
    failed: int = 0
    tokens: dict = field(default_factory=dict)  # pool -> flushed token count
    shards: dict = field(default_factory=dict)  # pool -> shard files on disk
    domain_counts: dict = field(default_factory=dict)
    complete: bool = False
    config: dict = field(default_factory=dict)

    def save(self, root: Path) -> None:
        tmp = root / (PROGRESS_FILE + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.replace(tmp, root / PROGRESS_FILE)

    @staticmethod
    def load(root: Path) -> "CaptureProgress | None":
        p = root / PROGRESS_FILE
        if not p.exists():
            return None
        return CaptureProgress(**json.loads(p.read_text()))


def _resume_state(
    out_root: Path, resume: bool, run_config: dict
) -> CaptureProgress:
    """Validate/prepare the capture root and return the starting progress."""
    progress = CaptureProgress.load(out_root)
    existing = {pool: shard_paths(out_root, pool) for pool in ("visual", "text")}
    has_shards = any(existing.values())

    if not resume:
        if progress is not None or has_shards:
            raise RuntimeError(
                f"{out_root} already contains capture data; pass resume=True "
                "(--resume) to continue it, or choose a fresh --out directory."
            )
        return CaptureProgress(config=run_config)

    if progress is None:
        if has_shards:
            raise RuntimeError(
                f"{out_root} has shards but no {PROGRESS_FILE}; cannot resume "
                "safely — use a fresh --out directory."
            )
        logger.info("resume requested but no prior progress at %s; starting fresh", out_root)
        return CaptureProgress(config=run_config)

    mismatched = {
        k: (progress.config.get(k), run_config[k])
        for k in run_config
        if progress.config.get(k) != run_config[k]
    }
    if mismatched:
        raise RuntimeError(
            f"resume config mismatch at {out_root}: {mismatched} — a capture "
            "must be resumed with identical parameters."
        )

    # Drop shards from the un-checkpointed window (crash between flush and
    # progress write) so on-disk state matches the record exactly.
    for pool, paths in existing.items():
        recorded = progress.shards.get(pool, 0)
        for stray in paths[recorded:]:
            logger.warning("resume: deleting un-checkpointed shard %s", stray)
            stray.unlink()
    return progress


def _fast_forward(examples_iter: Iterator[Example], n: int) -> None:
    """Consume n examples from the deterministic stream (no model work)."""
    t0 = time.monotonic()
    for i in range(n):
        try:
            next(examples_iter)
        except StopIteration:
            raise RuntimeError(
                f"stream exhausted after {i}/{n} fast-forward examples; the "
                "stream is shorter than the recorded progress — dataset "
                "changed upstream?"
            ) from None
    if n:
        logger.info("resume: fast-forwarded %d examples in %.1fs", n, time.monotonic() - t0)


def _capture_one(
    model,
    processor,
    ex: Example,
    sink,
    language,
    visual_cap_per_example: int,
    seed: int,
    max_seq_len: int,
    text_control: bool,
    writers: dict[str, ShardWriter],
) -> str:
    """Capture a single example into the writers. Returns 'ok' or 'too_long'."""
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

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
        logger.warning(
            "skipping example %d (%s): %d tokens > max_seq_len=%d",
            ex.example_id, ex.domain, T, max_seq_len,
        )
        return "too_long"

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
            keep = sorted(rng.sample(range(visual_rows.shape[0]), visual_cap_per_example))
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
    return "ok"


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
    n_examples_target: int | None = None,
    resume: bool = False,
    checkpoint_every: int = 100,
) -> CaptureMeta:
    """Run capture over `examples_iter`, writing pools + meta.json to out_root.

    `n_examples_target` counts CONSUMED stream examples (so a resumed run
    stops at the same stream position a single run would). Interrupt-safe:
    see module docstring (Resilience).
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        "visual_cap_per_example": visual_cap_per_example,
        "seed": seed,
        "max_seq_len": max_seq_len,
        "text_control": text_control,
        "shard_size": shard_size,
    }
    progress = _resume_state(out_root, resume, run_config)
    if progress.complete:
        logger.info("capture at %s already complete; nothing to do", out_root)
        return CaptureMeta.load(out_root)

    examples_iter = iter(examples_iter)
    _fast_forward(examples_iter, progress.stream_consumed)

    language = model.language.model  # DeepseekV2Model
    target_layer = language.layers[TARGET_LAYER_ZERO_INDEXED]
    moe = target_layer.mlp

    pools = ("text",) if text_control else ("visual", "text")
    writers = {p: ShardWriter(out_root, p, shard_size=shard_size) for p in pools}
    base_tokens = {p: progress.tokens.get(p, 0) for p in pools}

    def _early_exit(_module, _args, _output):
        raise StopForward

    def _checkpoint() -> None:
        for w in writers.values():
            w.flush_partial()
        progress.tokens = {p: base_tokens[p] + writers[p].total_written for p in pools}
        progress.shards = {p: writers[p].shard_count for p in pools}
        progress.save(out_root)

    t0 = time.monotonic()
    session_consumed = 0
    hook = target_layer.register_forward_hook(_early_exit)
    try:
        with instrumented_capture(moe, store_dtype=torch.bfloat16, device="cpu") as sink:
            for ex in examples_iter:
                if n_examples_target is not None and progress.stream_consumed >= n_examples_target:
                    break
                try:
                    outcome = _capture_one(
                        model, processor, ex, sink, language,
                        visual_cap_per_example, seed, max_seq_len,
                        text_control, writers,
                    )
                except Exception:
                    sink.drain()  # discard any partial chunk from the failure
                    progress.failed += 1
                    log = logger.exception if progress.failed <= 3 else logger.warning
                    log(
                        "example %d (%s) failed (%d failures so far)%s",
                        ex.example_id, ex.domain, progress.failed,
                        "" if progress.failed <= 3 else f": {traceback.format_exc(limit=1).splitlines()[-1]}",
                    )
                    outcome = "failed"
                progress.stream_consumed += 1
                session_consumed += 1
                if outcome == "ok":
                    progress.captured += 1
                    progress.domain_counts[ex.domain] = progress.domain_counts.get(ex.domain, 0) + 1
                elif outcome == "too_long":
                    progress.skipped_too_long += 1

                if progress.failed > max(20, progress.stream_consumed // 20):
                    raise RuntimeError(
                        f"{progress.failed} failures in {progress.stream_consumed} "
                        "examples (>5%) — systematic problem, aborting."
                    )

                if progress.stream_consumed % checkpoint_every == 0:
                    _checkpoint()
                if session_consumed % log_every == 0:
                    written = {p: base_tokens[p] + w.total_written + w._buffered for p, w in writers.items()}
                    dt = time.monotonic() - t0
                    logger.info(
                        "%d consumed (%d captured, %d failed) | tokens %s | %.1f ex/s",
                        progress.stream_consumed, progress.captured, progress.failed,
                        written, session_consumed / max(dt, 1e-9),
                    )
    finally:
        hook.remove()

    for w in writers.values():
        w.close()
    progress.complete = True
    _checkpoint_final = {p: base_tokens[p] + writers[p].total_written for p in pools}
    progress.tokens = _checkpoint_final
    progress.shards = {p: writers[p].shard_count for p in pools}
    progress.save(out_root)

    meta = CaptureMeta(
        name=out_root.name,
        model_id=MODEL_ID,
        spec_version=SPEC_VERSION,
        target_layer_zero_indexed=TARGET_LAYER_ZERO_INDEXED,
        tokens={
            "visual": _checkpoint_final.get("visual", 0),
            "text": _checkpoint_final.get("text", 0),
        },
        examples=progress.captured,
        config={
            **run_config,
            "skipped_too_long": progress.skipped_too_long,
            "failed": progress.failed,
            "stream_consumed": progress.stream_consumed,
            "domain_example_counts": progress.domain_counts,
        },
    )
    meta.save(out_root)
    logger.info(
        "capture done: %d captured / %d consumed (%d too long, %d failed), tokens=%s → %s",
        progress.captured, progress.stream_consumed,
        progress.skipped_too_long, progress.failed, meta.tokens, out_root,
    )
    return meta
