"""Memory-bounded pure-FP32 reference forward for the spec §3 hook-validation gate.

The gate (spec §3, §20 step 1) requires a *pure FP32* forward of the language
stack through the target layer (language_layers[5]). The model is 3.37B params
(13.5 GB FP32) and the machine has 16 GB RAM, so a full FP32 load is not
viable. Instead we stream: build the model skeleton on the meta device (no
weights), then materialize one FP32 submodule at a time from the checkpoint —
vision tower + projector + embed_tokens for the embedding phase (~2 GB FP32,
acceptable as one piece), then language layers 0..5 sequentially, freeing each
before loading the next.

`fp32_prefix_forward` replicates `DeepseekV2Model.forward`'s preamble EXACTLY
for the transformers==4.38.2 eager path with no KV cache: position_ids =
arange(0, T).unsqueeze(0), `_prepare_4d_causal_attention_mask(attention_mask,
(b, T), inputs_embeds, 0)`, and the identical decoder-layer call signature.
Multimodal inputs go through the skeleton's own `prepare_inputs_embeds` (the
tile scatter is NOT reimplemented). Unit-tested for exact agreement with a
direct full-model FP32 forward on a tiny random checkpoint
(tests/test_streamed_forward.py).
"""

from __future__ import annotations

import gc
import json
import logging
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, ContextManager, Sequence

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from moe_sae.model_loading import MODEL_ID, TARGET_LAYER_ZERO_INDEXED
from moe_sae.vendor.deepseek_vl2.modeling_deepseek_vl_v2 import (
    DeepseekVLV2ForCausalLM,
)

logger = logging.getLogger(__name__)

_INDEX_FILE = "model.safetensors.index.json"
_SINGLE_FILE = "model.safetensors"


def _resolve_checkpoint_dir(model_id_or_path: str | Path) -> Path:
    """Local directory holding the checkpoint; never downloads (offline gate)."""
    p = Path(model_id_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model_id_or_path), local_files_only=True))


class CheckpointStreamer:
    """Lazy weight loader over a safetensors checkpoint (sharded or single-file).

    Parses model.safetensors.index.json (or the single model.safetensors) into
    a tensor-name -> file map; `load_submodule_state` reads only the tensors
    under one module prefix and upcasts them, so peak host memory is one FP32
    submodule, never the full model.
    """

    def __init__(self, model_id_or_path: str | Path = MODEL_ID):
        self.root = _resolve_checkpoint_dir(model_id_or_path)
        index = self.root / _INDEX_FILE
        single = self.root / _SINGLE_FILE
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            self._weight_map: dict[str, Path] = {
                name: self.root / fname for name, fname in weight_map.items()
            }
        elif single.exists():
            with safe_open(str(single), framework="pt") as f:
                self._weight_map = {name: single for name in f.keys()}
        else:
            raise FileNotFoundError(
                f"No {_INDEX_FILE} or {_SINGLE_FILE} in {self.root}"
            )

    def keys(self) -> list[str]:
        return sorted(self._weight_map)

    def load_tensor(self, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        if name not in self._weight_map:
            raise KeyError(f"tensor {name!r} not in checkpoint {self.root}")
        with safe_open(str(self._weight_map[name]), framework="pt") as f:
            t = f.get_tensor(name)
        if dtype is not None and t.is_floating_point():
            t = t.to(dtype)
        return t

    def load_submodule_state(
        self, prefix: str, dtype: torch.dtype | None = torch.float32
    ) -> dict[str, torch.Tensor]:
        """State dict of every checkpoint tensor under `prefix.`, with keys
        relative to the prefix, floating tensors upcast to `dtype`."""
        dotted = prefix + "."
        by_file: dict[Path, list[str]] = defaultdict(list)
        for name, path in self._weight_map.items():
            if name.startswith(dotted):
                by_file[path].append(name)
        if not by_file:
            raise KeyError(f"no checkpoint tensors under prefix {prefix!r} in {self.root}")
        state: dict[str, torch.Tensor] = {}
        for path, names in by_file.items():
            with safe_open(str(path), framework="pt") as f:
                for name in names:
                    t = f.get_tensor(name)
                    if dtype is not None and t.is_floating_point():
                        t = t.to(dtype)
                    state[name[len(dotted):]] = t
        return state


def build_skeleton(
    model_id_or_path: str | Path = MODEL_ID,
    config=None,
    model_cls: type = DeepseekVLV2ForCausalLM,
) -> torch.nn.Module:
    """Construct the model on the meta device — structure only, no weights.

    Buffers (rotary caches etc.) are created for real on CPU
    (`init_empty_weights(include_buffers=False)`): they are small, absent from
    the checkpoint, and must survive materialize/dematerialize cycles.
    Attention is pinned to eager — the spec §3 reference path and the BF16
    production path (model_loading.load_model) both use it.
    """
    if config is None:
        config = model_cls.config_class.from_pretrained(
            str(_resolve_checkpoint_dir(model_id_or_path))
        )
    config._attn_implementation = "eager"
    language_config = getattr(config, "language_config", None)
    if language_config is not None:
        language_config._attn_implementation = "eager"
    with init_empty_weights():
        model = model_cls(config)
    return model.eval()


def free_memory(device: str | torch.device | None = None) -> None:
    """Aggressively release host/accelerator memory between submodules."""
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def materialize(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
    device: str | torch.device,
    dtype: torch.dtype | None = torch.float32,
) -> torch.nn.Module:
    """Swap real tensors into a meta-skeleton submodule (to_empty semantics via
    load_state_dict(assign=True)); construction-time non-persistent buffers are
    moved to `device` alongside."""
    converted = {}
    for key, t in state.items():
        if dtype is not None and t.is_floating_point():
            t = t.to(dtype)
        converted[key] = t.to(device)
    module.load_state_dict(converted, strict=True, assign=True)
    module.to(device)  # moves the (real, CPU) non-persistent buffers
    for p in module.parameters():
        p.requires_grad_(False)
    for name, t in list(module.named_parameters()) + list(module.named_buffers()):
        assert not t.is_meta, f"{name} still on meta after materialize"
    return module


def dematerialize(module: torch.nn.Module) -> None:
    """Return a materialized submodule to the meta device, freeing its memory.

    Non-persistent buffers (rotary caches — created at construction, absent
    from the checkpoint) are kept real so the module can be re-materialized;
    they are a few MB total.
    """
    for sub in module.modules():
        for name, p in list(sub._parameters.items()):
            if p is not None and not p.is_meta:
                sub._parameters[name] = torch.nn.Parameter(
                    torch.empty_like(p, device="meta"), requires_grad=False
                )
        for name, b in list(sub._buffers.items()):
            if b is None or b.is_meta or name in sub._non_persistent_buffers_set:
                continue
            sub._buffers[name] = torch.empty_like(b, device="meta")
    free_memory()


def _meta_param(module: torch.nn.Module, name: str) -> None:
    setattr(
        module,
        name,
        torch.nn.Parameter(
            torch.empty_like(getattr(module, name), device="meta"), requires_grad=False
        ),
    )


def _resolve_language(skeleton: torch.nn.Module) -> tuple[torch.nn.Module, str]:
    """(DeepseekV2Model, checkpoint prefix of that module). Supports both the
    full DeepseekVLV2ForCausalLM and a bare DeepseekV2ForCausalLM (tests)."""
    if hasattr(skeleton, "language"):
        return skeleton.language.model, "language.model."
    return skeleton.model, "model."


def _standardize_example(
    ex: dict, device: torch.device, dtype: torch.dtype
) -> dict:
    """Normalize one validation-input dict (docs/INTERFACES.md export format)
    to batched-of-one tensors on the target device."""
    input_ids = torch.as_tensor(ex["input_ids"], dtype=torch.long, device=device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    images = ex.get("images")
    images_seq_mask = ex.get("images_seq_mask")
    images_spatial_crop = ex.get("images_spatial_crop")
    if images is not None and images_spatial_crop is not None:
        images = images.to(device=device, dtype=dtype)
        if images.dim() == 4:
            images = images.unsqueeze(0)
        images_spatial_crop = torch.as_tensor(
            images_spatial_crop, dtype=torch.long, device=device
        )
        if images_spatial_crop.dim() == 2:
            images_spatial_crop = images_spatial_crop.unsqueeze(0)
        images_seq_mask = torch.as_tensor(
            images_seq_mask, dtype=torch.bool, device=device
        )
        if images_seq_mask.dim() == 1:
            images_seq_mask = images_seq_mask.unsqueeze(0)
    else:
        images = images_seq_mask = images_spatial_crop = None

    attention_mask = ex.get("attention_mask")
    if attention_mask is not None:
        attention_mask = torch.as_tensor(attention_mask, device=device)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

    return {
        "input_ids": input_ids,
        "images": images,
        "images_seq_mask": images_seq_mask,
        "images_spatial_crop": images_spatial_crop,
        "attention_mask": attention_mask,
    }


@torch.no_grad()
def fp32_prefix_forward(
    streamer: CheckpointStreamer,
    skeleton: torch.nn.Module,
    batch: dict | Sequence[dict],
    n_layers: int = TARGET_LAYER_ZERO_INDEXED + 1,
    device: str | torch.device = "cpu",
    on_target_moe: Callable[[torch.nn.Module], ContextManager] | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | list[torch.Tensor]:
    """FP32 forward of embeddings + language layers 0..n_layers-1, one FP32
    submodule resident at a time.

    `batch` is one example dict or a sequence of them (each per
    docs/INTERFACES.md validation-input export; extra keys ignored). Passing
    several examples amortizes checkpoint IO: each layer is materialized once
    and applied to every example before being freed.

    `on_target_moe(layer.mlp)` must return a context manager; it is entered
    around the target layer's forward for each example, in example order
    (the validator hangs reference_capture there).

    Returns the hidden state after layer n_layers-1 — [1, T, d] per example
    (a single tensor if `batch` was a single dict).
    """
    single = isinstance(batch, dict)
    examples = [batch] if single else list(batch)
    device = torch.device(device)
    skeleton.eval()

    base, lm_prefix = _resolve_language(skeleton)
    assert not base._use_flash_attention_2, (
        "spec §3 reference forward requires eager attention"
    )
    assert 1 <= n_layers <= len(base.layers), n_layers

    exs = [_standardize_example(e, device, dtype) for e in examples]
    needs_vision = any(
        ex["images"] is not None and int(ex["images_spatial_crop"].sum()) > 0
        for ex in exs
    )

    # ---- embedding phase: embed_tokens (+ vision tower & projector) -------- #
    materialize(
        base.embed_tokens,
        streamer.load_submodule_state(lm_prefix + "embed_tokens", dtype),
        device,
        dtype,
    )
    if needs_vision:
        assert hasattr(skeleton, "vision"), "multimodal input needs the VL model"
        materialize(
            skeleton.vision, streamer.load_submodule_state("vision", dtype), device, dtype
        )
        materialize(
            skeleton.projector,
            streamer.load_submodule_state("projector", dtype),
            device,
            dtype,
        )
        for pname in ("image_newline", "view_seperator"):
            skeleton.register_parameter(
                pname,
                torch.nn.Parameter(
                    streamer.load_tensor(pname, dtype).to(device), requires_grad=False
                ),
            )

    hiddens: list[torch.Tensor] = []
    masks: list[torch.Tensor | None] = []
    position_ids_list: list[torch.Tensor] = []
    for ex in exs:
        if hasattr(skeleton, "prepare_inputs_embeds"):
            inputs_embeds = skeleton.prepare_inputs_embeds(
                input_ids=ex["input_ids"],
                images=ex["images"],
                images_seq_mask=ex["images_seq_mask"],
                images_spatial_crop=ex["images_spatial_crop"],
            )
        else:
            inputs_embeds = base.embed_tokens(ex["input_ids"])

        # --- DeepseekV2Model.forward preamble, transformers 4.38.2, eager, ---
        # --- no KV cache (past_key_values_length == 0). ----------------------
        batch_size, seq_length = inputs_embeds.shape[:2]
        position_ids = torch.arange(
            0, seq_length, dtype=torch.long, device=device
        ).unsqueeze(0)
        attention_mask = _prepare_4d_causal_attention_mask(
            ex["attention_mask"], (batch_size, seq_length), inputs_embeds, 0
        )
        hiddens.append(inputs_embeds)
        masks.append(attention_mask)
        position_ids_list.append(position_ids)

    dematerialize(base.embed_tokens)
    if needs_vision:
        dematerialize(skeleton.vision)
        dematerialize(skeleton.projector)
        _meta_param(skeleton, "image_newline")
        _meta_param(skeleton, "view_seperator")
    free_memory(device)

    # ---- layer phase: materialize each decoder layer once, run all examples #
    for layer_idx in range(n_layers):
        decoder_layer = base.layers[layer_idx]
        materialize(
            decoder_layer,
            streamer.load_submodule_state(f"{lm_prefix}layers.{layer_idx}", dtype),
            device,
            dtype,
        )
        is_target = layer_idx == n_layers - 1
        for i in range(len(hiddens)):
            ctx = (
                on_target_moe(decoder_layer.mlp)
                if (is_target and on_target_moe is not None)
                else nullcontext()
            )
            with ctx:
                # Identical call to the DeepseekV2Model.forward layer loop
                # (eager, no cache, no gradient checkpointing).
                layer_outputs = decoder_layer(
                    hiddens[i],
                    attention_mask=masks[i],
                    position_ids=position_ids_list[i],
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
            hiddens[i] = layer_outputs[0]
        dematerialize(decoder_layer)
        free_memory(device)

    return hiddens[0] if single else hiddens
