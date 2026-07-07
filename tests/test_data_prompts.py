"""Prompt construction + tokenization masks against the real DeepSeek-VL2-Tiny
tokenizer/processor (small, cached; no model weights are loaded)."""

import re

import PIL.Image
import pytest
import torch

from moe_sae.data.prompts import (
    ASSISTANT_ROLE,
    USER_ROLE,
    capture_conversation,
    eval_conversation,
    tokenize_capture,
    tokenize_eval,
)
from moe_sae.data.streams import Example


@pytest.fixture(scope="module")
def processor():
    from moe_sae.model_loading import load_processor

    return load_processor()


def make_example(with_image=True, prompt="What color is the square?", answer="It is teal."):
    img = PIL.Image.new("RGB", (96, 64), (0, 128, 128)) if with_image else None
    return Example(
        example_id=123,
        domain="coco" if with_image else "text_control",
        image=img,
        prompt_text=prompt,
        answer_text=answer,
    )


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def test_capture_conversation_upstream_format():
    ex = make_example()
    conv, images = capture_conversation(ex)
    assert conv == [
        {"role": USER_ROLE, "content": "<image>\nWhat color is the square?"},
        {"role": ASSISTANT_ROLE, "content": ""},
    ]
    assert images == [ex.image]


def test_capture_conversation_text_control_has_no_image_tag():
    conv, images = capture_conversation(make_example(with_image=False))
    assert "<image>" not in conv[0]["content"]
    assert images == []


def test_eval_conversation_carries_answer():
    conv, _ = eval_conversation(make_example())
    assert conv[-1] == {"role": ASSISTANT_ROLE, "content": "It is teal."}


def test_tokenize_capture_masks(processor):
    tok = tokenize_capture(processor, make_example())
    T = tok.seq_len
    assert tok.input_ids.shape == (T,)
    assert tok.visual_mask.shape == (T,) and tok.answer_mask.shape == (T,)
    assert not tok.answer_mask.any(), "prompt-only: no answer tokens"
    assert not tok.visual_mask[0], "BOS is not visual"
    # visual positions are exactly the <image> token positions
    image_positions = tok.input_ids == processor.image_token_id
    assert torch.equal(tok.visual_mask, image_positions)
    assert tok.visual_mask.sum() > 0
    assert tok.images is not None and tok.images_spatial_crop is not None


def test_tokenize_capture_text_control(processor):
    tok = tokenize_capture(processor, make_example(with_image=False))
    assert not tok.visual_mask.any()
    assert (tok.input_ids == processor.image_token_id).sum() == 0
    assert tok.images is None and tok.images_spatial_crop is None


def test_tokenize_eval_answer_roundtrip(processor):
    ex = make_example()
    tok = tokenize_eval(processor, ex)
    # prompt-only ids are a strict prefix (asserted inside; re-check here)
    tok_p = tokenize_capture(processor, ex)
    n_p = tok_p.seq_len
    assert torch.equal(tok.input_ids[:n_p], tok_p.input_ids)
    assert tok.answer_mask[:n_p].sum() == 0 and tok.answer_mask[n_p:].all()
    # masked span round-trips to the answer text modulo whitespace/specials
    span = tok.input_ids[tok.answer_mask].tolist()
    decoded = processor.tokenizer.decode(span, skip_special_tokens=True)
    assert _norm(decoded) == _norm(ex.answer_text)
    # answer span terminates with EOS (teacher-forced target includes it)
    assert span[-1] == processor.eos_id


def test_tokenize_eval_masks_disjoint_and_visual_stable(processor):
    ex = make_example()
    tok = tokenize_eval(processor, ex)
    assert not (tok.answer_mask & tok.visual_mask).any()
    assert not (tok.answer_mask & (tok.input_ids == processor.image_token_id)).any()
    # visual token count identical to the prompt-only tokenization
    tok_p = tokenize_capture(processor, ex)
    assert int(tok.visual_mask.sum()) == int(tok_p.visual_mask.sum())
    # matched-position mask of spec §13 = NOT answer_mask (and not BOS): the
    # three token classes must partition the sequence
    n_answer = int(tok.answer_mask.sum())
    n_visual = int(tok.visual_mask.sum())
    n_text = tok.seq_len - n_answer - n_visual
    assert n_text > 1  # BOS + at least some prompt text


def test_tokenize_eval_text_control(processor):
    ex = make_example(with_image=False, answer="Just words.")
    tok = tokenize_eval(processor, ex)
    assert not tok.visual_mask.any()
    decoded = processor.tokenizer.decode(
        tok.input_ids[tok.answer_mask].tolist(), skip_special_tokens=True
    )
    assert _norm(decoded) == "Just words."


def test_tokenize_eval_multitoken_answer(processor):
    ex = make_example(answer="The 2015 favorability value was 38 percent, up from 24.")
    tok = tokenize_eval(processor, ex)
    decoded = processor.tokenizer.decode(
        tok.input_ids[tok.answer_mask].tolist(), skip_special_tokens=True
    )
    assert _norm(decoded) == _norm(ex.answer_text)
