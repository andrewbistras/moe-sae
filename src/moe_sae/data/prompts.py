"""Conversation construction + tokenization for DeepseekVLV2Processor.

Conversation format follows the upstream DeepSeek-VL2 inference example
exactly: a "<|User|>" turn whose content is "<image>\\n{prompt}" plus an empty
"<|Assistant|>" turn (capture / generation-ready), or the same with the
ground-truth answer in the Assistant turn (teacher-forced eval, spec §13).

Mask semantics over the token sequence (length T, batch dim stripped):

- ``visual_mask``: the processor's ``images_seq_mask`` — projected visual
  positions (spec §9 "projected visual token positions").
- ``answer_mask``: positions of the teacher-forced answer tokens (incl. the
  trailing EOS). Computed by processing the prompt-only and the full
  conversation and diffing lengths; the processor tokenizes each message
  separately and concatenates (format_messages_v2), so the prompt token ids
  are a strict prefix of the full sequence — asserted, and the masked span is
  verified in tests to decode back to the answer text.

Prompt-only sequences use inference_mode=True (the processor strips the EOS
that terminates the empty Assistant turn — the generation-ready form used for
capture); teacher-forced sequences use inference_mode=False so the answer's
EOS is kept and included in answer_mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import PIL.Image
import torch

from moe_sae.data.streams import Example

USER_ROLE = "<|User|>"
ASSISTANT_ROLE = "<|Assistant|>"


def capture_conversation(
    example: Example,
) -> tuple[list[dict[str, str]], list[PIL.Image.Image]]:
    """Prompt-only conversation (User turn + empty Assistant turn).

    For examples without an image (text_control) the "<image>\\n" prefix is
    omitted and the image list is empty — no <image> token enters the
    sequence.
    """
    if example.image is not None:
        content = f"<image>\n{example.prompt_text}"
        images = [example.image]
    else:
        content = example.prompt_text
        images = []
    conv = [
        {"role": USER_ROLE, "content": content},
        {"role": ASSISTANT_ROLE, "content": ""},
    ]
    return conv, images


def eval_conversation(
    example: Example,
) -> tuple[list[dict[str, str]], list[PIL.Image.Image]]:
    """Teacher-forced conversation: Assistant turn carries the ground truth."""
    conv, images = capture_conversation(example)
    conv[-1] = {"role": ASSISTANT_ROLE, "content": example.answer_text}
    return conv, images


@dataclass
class TokenizedExample:
    """Processor outputs for one example (batch dim stripped) + boolean masks.

    images / images_spatial_crop are None when the example has no image (the
    processor's dummy zero-image placeholder is dropped); consumers unsqueeze
    a batch dim before model.prepare_inputs_embeds.
    """

    input_ids: torch.Tensor  # [T] long
    images: torch.Tensor | None  # [n_tiles, 3, H, W]
    images_seq_mask: torch.Tensor | None  # [T] bool (None iff no image)
    images_spatial_crop: torch.Tensor | None  # [n_images, 2] long
    visual_mask: torch.Tensor  # [T] bool
    answer_mask: torch.Tensor  # [T] bool

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[0])


def _process(processor, conversations, images, inference_mode: bool):
    return processor(
        conversations=conversations,
        images=images,
        force_batchify=True,
        inference_mode=inference_mode,
        system_prompt="",
    )


def tokenize_capture(processor, example: Example) -> TokenizedExample:
    """Tokenize the prompt-only conversation; answer_mask is all False."""
    conv, images = capture_conversation(example)
    out = _process(processor, conv, images, inference_mode=True)
    ids = out.input_ids[0]
    vm = out.images_seq_mask[0].to(torch.bool)
    has_image = len(images) > 0
    return TokenizedExample(
        input_ids=ids,
        images=out.images[0] if has_image else None,
        images_seq_mask=vm if has_image else None,
        images_spatial_crop=out.images_spatial_crop[0] if has_image else None,
        visual_mask=vm.clone(),
        answer_mask=torch.zeros(ids.shape[0], dtype=torch.bool),
    )


def tokenize_eval(processor, example: Example) -> TokenizedExample:
    """Tokenize the teacher-forced conversation with visual + answer masks.

    The prompt-only token ids must be a strict prefix of the full sequence
    (message-wise tokenization guarantees no cross-boundary merging);
    answer_mask marks everything after the prompt (answer tokens + EOS).
    """
    conv_p, images = capture_conversation(example)
    conv_f, _ = eval_conversation(example)
    out_p = _process(processor, conv_p, list(images), inference_mode=True)
    out_f = _process(processor, conv_f, list(images), inference_mode=False)

    ids_p = out_p.input_ids[0]
    ids_f = out_f.input_ids[0]
    n_p, n_f = ids_p.shape[0], ids_f.shape[0]
    assert n_f > n_p and torch.equal(ids_f[:n_p], ids_p), (
        "prompt token ids are not a prefix of the teacher-forced sequence; "
        "answer_mask diffing is invalid for this example"
    )

    vm = out_f.images_seq_mask[0].to(torch.bool)
    answer_mask = torch.zeros(n_f, dtype=torch.bool)
    answer_mask[n_p:] = True
    assert not (answer_mask & vm).any(), "answer and visual masks overlap"

    has_image = len(images) > 0
    return TokenizedExample(
        input_ids=ids_f,
        images=out_f.images[0] if has_image else None,
        images_seq_mask=vm if has_image else None,
        images_spatial_crop=out_f.images_spatial_crop[0] if has_image else None,
        visual_mask=vm.clone(),
        answer_mask=answer_mask,
    )
