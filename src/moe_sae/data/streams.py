"""The four frozen training domains (spec §9), streamed from HuggingFace Hub.

| Domain   | Dataset (verified to stream, 2026-07-06)  | HF splits            |
|----------|-------------------------------------------|----------------------|
| coco     | jxie/coco_captions (parquet COCO-caption) | train/validation/test|
| textvqa  | lmms-lab/textvqa                          | train/validation/test|
| chartqa  | HuggingFaceM4/ChartQA                     | train/val/test       |
| ai2d     | lmms-lab/ai2d                             | test only            |

`HuggingFaceM4/COCO` (the obvious COCO candidate) is a loading-script dataset
and no longer loads under `datasets>=3`; `jxie/coco_captions` is the most
downloaded parquet COCO-captions mirror with images + captions and real
train/validation/test splits, so it is used instead.

AI2D ships a single HF split; disjoint pseudo-splits are carved from it by raw
record index: index % 10 in 0–7 → "train", 8 → "validation", 9 → "test". This
keeps train/eval example sets disjoint despite the single upstream split.

Example ids are frozen as ``DOMAIN_OFFSET + SPLIT_OFFSET + yielded index``
(offsets below). **Determinism assumptions:** HF streaming yields records in a
fixed order for a fixed dataset revision (parquet file order × row order), and
the upstream repos are not rewritten. Given identical stream order the ids are
stable across runs; they feed the `example_id % 100` train/dev/eval convention
(docs/INTERFACES.md "Splits").

Record→Example field mapping is isolated in pure `map_*_record` functions so it
is unit-testable on dict fixtures without the network.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping

import PIL.Image

logger = logging.getLogger(__name__)

TRAIN_DOMAINS = ("coco", "textvqa", "chartqa", "ai2d")

# Frozen example-id offsets. 100M per domain; sub-offsets keep ids from
# different HF splits of the same domain globally unique.
DOMAIN_OFFSETS = {
    "coco": 0,
    "textvqa": 100_000_000,
    "chartqa": 200_000_000,
    "ai2d": 300_000_000,
    "text_control": 400_000_000,
}
SPLIT_OFFSETS = {"train": 0, "validation": 50_000_000, "test": 80_000_000}

# Fixed COCO captioning instruction (spec §9 "caption prompts"): the dataset
# provides no question, so a single frozen instruction is used for every
# example; the caption is the teacher-forced answer.
COCO_CAPTION_INSTRUCTION = "Describe this image."

# AI2D multiple-choice prompt suffix (frozen).
_AI2D_SUFFIX = "Answer with the correct option."

_AI2D_SPLIT_BUCKETS = {
    "train": frozenset(range(8)),
    "validation": frozenset({8}),
    "test": frozenset({9}),
}


@dataclass
class Example:
    """One multimodal training/eval example (image is None only for the
    text-only control stream)."""

    example_id: int
    domain: str
    image: PIL.Image.Image | None
    prompt_text: str
    answer_text: str


# --------------------------------------------------------------------------- #
# Dataset-specific record mapping (pure, network-free, unit-tested)            #
# --------------------------------------------------------------------------- #


def map_coco_record(rec: Mapping[str, Any]) -> tuple[Any, str, str]:
    """jxie/coco_captions: {image, filename, cocoid, caption}."""
    return rec["image"], COCO_CAPTION_INSTRUCTION, str(rec["caption"]).strip()


def map_textvqa_record(rec: Mapping[str, Any]) -> tuple[Any, str, str]:
    """lmms-lab/textvqa: {image, question, answers: list[str] (10 raters), ...}.

    answer_text = the modal (most common) rater answer; Counter.most_common
    breaks ties by first insertion, so the choice is deterministic given
    record order. Test split has no answers → empty answer_text.
    """
    answers = rec.get("answers") or []
    if answers:
        answer = Counter(str(a).strip() for a in answers).most_common(1)[0][0]
    else:
        answer = ""
    return rec["image"], str(rec["question"]).strip(), answer


def map_chartqa_record(rec: Mapping[str, Any]) -> tuple[Any, str, str]:
    """HuggingFaceM4/ChartQA: {image, query, label: list[str], human_or_machine}."""
    label = rec.get("label") or []
    answer = str(label[0]).strip() if label else ""
    return rec["image"], str(rec["query"]).strip(), answer


def map_ai2d_record(rec: Mapping[str, Any]) -> tuple[Any, str, str]:
    """lmms-lab/ai2d: {image, question, options: list[str], answer: str index}.

    The prompt lists the options as (A)/(B)/... choices; answer_text is the
    text of the correct option (not the letter), so teacher forcing scores
    content rather than a single letter token.
    """
    options = [str(o).strip() for o in rec["options"]]
    lines = [str(rec["question"]).strip()]
    lines += [f"({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)]
    lines.append(_AI2D_SUFFIX)
    answer = options[int(rec["answer"])]
    return rec["image"], "\n".join(lines), answer


@dataclass(frozen=True)
class DomainSpec:
    domain: str
    dataset_id: str
    hf_splits: Mapping[str, str]  # canonical split -> HF split name
    map_record: Callable[[Mapping[str, Any]], tuple[Any, str, str]]
    pseudo_split: bool = False  # carve splits from a single HF split by index


DOMAIN_SPECS: dict[str, DomainSpec] = {
    "coco": DomainSpec(
        "coco",
        "jxie/coco_captions",
        {"train": "train", "validation": "validation", "test": "test"},
        map_coco_record,
    ),
    "textvqa": DomainSpec(
        "textvqa",
        "lmms-lab/textvqa",
        {"train": "train", "validation": "validation", "test": "test"},
        map_textvqa_record,
    ),
    "chartqa": DomainSpec(
        "chartqa",
        "HuggingFaceM4/ChartQA",
        {"train": "train", "validation": "val", "test": "test"},
        map_chartqa_record,
    ),
    "ai2d": DomainSpec(
        "ai2d",
        "lmms-lab/ai2d",
        {"train": "test", "validation": "test", "test": "test"},
        map_ai2d_record,
        pseudo_split=True,
    ),
}


# --------------------------------------------------------------------------- #
# Streams                                                                      #
# --------------------------------------------------------------------------- #


def domain_stream(
    domain: str,
    split: str = "train",
    records: Iterable[Mapping[str, Any]] | None = None,
) -> Iterator[Example]:
    """Yield Examples for one domain with frozen, stable example ids.

    `records` injects raw records for tests; by default the HF dataset is
    opened with streaming=True (no full download).
    """
    spec = DOMAIN_SPECS[domain]
    if split not in SPLIT_OFFSETS:
        raise ValueError(f"unknown split {split!r}")
    if records is None:
        from datasets import load_dataset  # deferred: heavy import

        records = load_dataset(spec.dataset_id, split=spec.hf_splits[split], streaming=True)

    base = DOMAIN_OFFSETS[domain] + SPLIT_OFFSETS[split]
    buckets = _AI2D_SPLIT_BUCKETS[split] if spec.pseudo_split else None
    out_index = 0
    for raw_index, rec in enumerate(records):
        if buckets is not None and raw_index % 10 not in buckets:
            continue
        image, prompt_text, answer_text = spec.map_record(rec)
        # The vendored processor pads with a 3-tuple background color and
        # normalizes 3 channels; grayscale/palette/RGBA images (present in
        # COCO/TextVQA train splits) crash ImageOps.pad. RGB is the contract.
        # (getattr guard: tests inject non-PIL sentinel objects as images.)
        if image is not None and getattr(image, "mode", "RGB") != "RGB":
            image = image.convert("RGB")
        yield Example(
            example_id=base + out_index,
            domain=domain,
            image=image,
            prompt_text=prompt_text,
            answer_text=answer_text,
        )
        out_index += 1


def interleave_domains(
    streams: Mapping[str, Iterator[Example]], seed: int
) -> Iterator[Example]:
    """Round-robin evenly at the example level (spec §9): each round yields one
    example per live domain, domain order shuffled per round by `seed`
    (str-seeded RNG → stable across processes/platforms). Exhausted domains
    drop out of the rotation; remaining domains stay even among themselves.
    """
    rng = random.Random(f"interleave:{seed}")
    active: dict[str, Iterator[Example]] = {k: iter(v) for k, v in streams.items()}
    while active:
        order = sorted(active)
        rng.shuffle(order)
        for name in order:
            try:
                yield next(active[name])
            except StopIteration:
                logger.info("domain stream %s exhausted; dropping from rotation", name)
                del active[name]


def text_control_stream(
    split: str = "train",
    seed: int = 0,
    streams: Mapping[str, Iterator[Example]] | None = None,
) -> Iterator[Example]:
    """Text-only control (spec §13): the same four-domain prompts WITHOUT the
    image. Eval-only baseline — never mixed into training captures. Ids get
    the text_control offset with a fresh running index; domain='text_control'.
    """
    if streams is None:
        streams = {d: domain_stream(d, split) for d in TRAIN_DOMAINS}
    base = DOMAIN_OFFSETS["text_control"] + SPLIT_OFFSETS[split]
    for i, ex in enumerate(interleave_domains(streams, seed)):
        yield Example(
            example_id=base + i,
            domain="text_control",
            image=None,
            prompt_text=ex.prompt_text,
            answer_text=ex.answer_text,
        )
