"""Eval example provider + validation-input export (spec §13; docs/INTERFACES.md).

Split choice per dataset (canonical "validation" requested by default):
coco → jxie/coco_captions "validation"; textvqa → lmms-lab/textvqa
"validation"; chartqa → HuggingFaceM4/ChartQA "val"; ai2d → the
index-carved pseudo-validation slice of its single "test" split (see
moe_sae.data.streams). text_control interleaves the four domains'
validation prompts with images stripped.

``iter_eval_examples`` selection: to give ``seed`` meaning without a full
dataset pass, the target examples are chosen as a seeded sample of n stream
indices from the first ``EVAL_POOL_FACTOR * n`` positions of the split; only
selected examples are materialized (streaming-friendly, deterministic). If the
split is shorter than requested, whatever was collected is yielded with a
warning.

``export_validation_inputs`` writes prompt-only processor outputs (batch dim
kept, i.e. input_ids [1,T], ready for ``model.prepare_inputs_embeds``) as
``examples_{i:04d}.pt``, mixed evenly across the 4 domains at the example
level, until ≥ n_tokens_target total token positions are covered. A
``meta.json`` with counts is written alongside for bookkeeping.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from moe_sae.data.prompts import tokenize_capture, tokenize_eval
from moe_sae.data.streams import (
    TRAIN_DOMAINS,
    Example,
    domain_stream,
    interleave_domains,
    text_control_stream,
)

logger = logging.getLogger(__name__)

EVAL_POOL_FACTOR = 4  # sample n eval examples from the first 4n of the split
EXPORT_MAX_SEQ_LEN = 2048  # match the capture driver's default cap

_PROCESSOR = None


def _get_processor():
    global _PROCESSOR
    if _PROCESSOR is None:
        from moe_sae.model_loading import load_processor  # tokenizer only, no weights

        _PROCESSOR = load_processor()
    return _PROCESSOR


@dataclass
class EvalExample:
    """Contract type (docs/INTERFACES.md): prompt AND teacher-forced answer."""

    input_ids: torch.Tensor  # [T]
    images: torch.Tensor | None
    images_seq_mask: torch.Tensor | None
    images_spatial_crop: torch.Tensor | None
    answer_mask: torch.Tensor  # [T] bool
    visual_mask: torch.Tensor  # [T] bool
    example_id: int
    domain: str  # coco|textvqa|chartqa|ai2d|text_control
    answer_text: str


def _select_stream(
    stream: Iterator[Example], n: int, seed: int, tag: str
) -> Iterator[Example]:
    """Yield a seeded n-subsample of the first EVAL_POOL_FACTOR*n examples,
    in stream order, materializing only selected examples."""
    rng = random.Random(f"evalsets:{seed}:{tag}")
    chosen = set(rng.sample(range(EVAL_POOL_FACTOR * n), n))
    got = 0
    for i, ex in enumerate(stream):
        if i in chosen:
            yield ex
            got += 1
            if got == n:
                return
    logger.warning("stream %s exhausted: %d/%d eval examples collected", tag, got, n)


def iter_eval_examples(
    domain: str, n: int, seed: int, split: str = "validation"
) -> Iterator[EvalExample]:
    """Teacher-forced eval examples for one domain (spec §13)."""
    processor = _get_processor()
    if domain == "text_control":
        stream = text_control_stream(split=split, seed=seed)
    else:
        stream = domain_stream(domain, split=split)
    for ex in _select_stream(stream, n, seed, f"{domain}:{split}"):
        tok = tokenize_eval(processor, ex)
        yield EvalExample(
            input_ids=tok.input_ids,
            images=tok.images,
            images_seq_mask=tok.images_seq_mask,
            images_spatial_crop=tok.images_spatial_crop,
            answer_mask=tok.answer_mask,
            visual_mask=tok.visual_mask,
            example_id=ex.example_id,
            domain=ex.domain,
            answer_text=ex.answer_text,
        )


def export_validation_inputs(
    out_dir: str | Path, n_tokens_target: int = 100_000, seed: int = 7
) -> list[Path]:
    """Write prompt-only processor outputs for the spec §3 FP32 hook-validation
    gate: examples_{i:04d}.pt dicts covering ≥ n_tokens_target token positions,
    mixed evenly (round-robin) across the 4 domains' validation splits."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processor = _get_processor()

    streams = {d: domain_stream(d, split="validation") for d in TRAIN_DOMAINS}
    paths: list[Path] = []
    total_tokens = 0
    domain_counts: dict[str, int] = {}
    for ex in interleave_domains(streams, seed):
        tok = tokenize_capture(processor, ex)
        if tok.seq_len > EXPORT_MAX_SEQ_LEN:
            continue
        payload = {
            "input_ids": tok.input_ids.unsqueeze(0),
            "images": None if tok.images is None else tok.images.unsqueeze(0),
            "images_seq_mask": (
                None if tok.images_seq_mask is None else tok.images_seq_mask.unsqueeze(0)
            ),
            "images_spatial_crop": (
                None
                if tok.images_spatial_crop is None
                else tok.images_spatial_crop.unsqueeze(0)
            ),
            "example_id": ex.example_id,
            "domain": ex.domain,
        }
        path = out_dir / f"examples_{len(paths):04d}.pt"
        torch.save(payload, path)
        paths.append(path)
        total_tokens += tok.seq_len
        domain_counts[ex.domain] = domain_counts.get(ex.domain, 0) + 1
        if total_tokens >= n_tokens_target:
            break

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "n_examples": len(paths),
                "total_tokens": total_tokens,
                "n_tokens_target": n_tokens_target,
                "seed": seed,
                "domain_example_counts": domain_counts,
                "split": "validation",
            },
            indent=2,
        )
    )
    logger.info(
        "exported %d validation inputs (%d tokens, target %d) → %s",
        len(paths),
        total_tokens,
        n_tokens_target,
        out_dir,
    )
    return paths
