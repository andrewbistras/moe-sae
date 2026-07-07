"""Frozen model loading + spec §2 startup assertions.

Spec v0.1.1 §2: the SAE activation site is "language transformer block 6,
one-indexed", i.e. zero-indexed entry 5 of the checkpoint's language-model layer
list — the 5th MoE block, since entry 0 is dense (first_k_dense_replace = 1).

Every entry point that touches the model MUST call `assert_frozen_architecture`
before doing anything else; it aborts on any mismatch (spec §2 "Index pinning").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from moe_sae.vendor.deepseek_vl2.modeling_deepseek import (
    DeepseekV2MLP,
    DeepseekV2MoE,
)
from moe_sae.vendor.deepseek_vl2.modeling_deepseek_vl_v2 import (
    DeepseekVLV2ForCausalLM,
)
from moe_sae.vendor.deepseek_vl2.processing_deepseek_vl_v2 import (
    DeepseekVLV2Processor,
)

logger = logging.getLogger(__name__)

MODEL_ID = "deepseek-ai/deepseek-vl2-tiny"

# Frozen constants (spec §2, §5). Do not modify without a spec version bump.
HIDDEN_SIZE = 1280
NUM_LANGUAGE_LAYERS = 12
NUM_ROUTED_EXPERTS = 64
NUM_SHARED_EXPERTS = 2
TOP_K_ROUTED = 6
TARGET_LAYER_ZERO_INDEXED = 5  # "block 6, one-indexed" == language_layers[5]


@dataclass(frozen=True)
class FrozenModelInfo:
    """Facts verified at startup, recorded alongside every artifact."""

    model_id: str
    hidden_size: int
    num_language_layers: int
    num_routed_experts: int
    num_shared_experts: int
    top_k_routed: int
    target_layer_zero_indexed: int
    dtype: str
    device: str


def pick_device(preferred: str | None = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def assert_frozen_architecture(model: DeepseekVLV2ForCausalLM) -> None:
    """Spec §2 index-pinning assertions. Abort on any mismatch."""
    layers = model.language.model.layers

    # 1. Exactly 12 language layers.
    assert len(layers) == NUM_LANGUAGE_LAYERS, (
        f"Expected {NUM_LANGUAGE_LAYERS} language layers, found {len(layers)}. "
        "Wrong checkpoint or wrong module path — aborting (spec §2)."
    )

    # 2. Entry 0's MLP is dense (no router).
    mlp0 = layers[0].mlp
    assert isinstance(mlp0, DeepseekV2MLP) and not isinstance(mlp0, DeepseekV2MoE), (
        f"language_layers[0].mlp is {type(mlp0).__name__}, expected dense "
        "DeepseekV2MLP (first_k_dense_replace=1). Aborting (spec §2)."
    )
    assert not hasattr(mlp0, "gate") or not hasattr(mlp0.gate, "weight") or (
        not isinstance(mlp0, DeepseekV2MoE)
    ), "language_layers[0].mlp unexpectedly has a router gate. Aborting (spec §2)."

    # 3. Entry 5's MLP is an MoE with 64 routed experts, 2 shared experts, top-6.
    mlp5 = layers[TARGET_LAYER_ZERO_INDEXED].mlp
    assert isinstance(mlp5, DeepseekV2MoE), (
        f"language_layers[{TARGET_LAYER_ZERO_INDEXED}].mlp is "
        f"{type(mlp5).__name__}, expected DeepseekV2MoE. Aborting (spec §2)."
    )
    n_experts = len(mlp5.experts)
    assert n_experts == NUM_ROUTED_EXPERTS, (
        f"Expected {NUM_ROUTED_EXPERTS} routed experts at target layer, found "
        f"{n_experts}. Aborting (spec §2)."
    )
    assert mlp5.config.n_shared_experts == NUM_SHARED_EXPERTS, (
        f"Expected {NUM_SHARED_EXPERTS} shared experts, found "
        f"{mlp5.config.n_shared_experts}. Aborting (spec §2)."
    )
    assert mlp5.gate.top_k == TOP_K_ROUTED, (
        f"Expected top-{TOP_K_ROUTED} routing, found top-{mlp5.gate.top_k}. "
        "Aborting (spec §2)."
    )
    assert mlp5.ep_size == 1, "Expert parallelism must be disabled for capture."

    # Hidden width sanity.
    assert model.language.config.hidden_size == HIDDEN_SIZE, (
        f"Expected hidden_size {HIDDEN_SIZE}, found "
        f"{model.language.config.hidden_size}. Aborting (spec §2)."
    )

    logger.info(
        "Frozen architecture verified: %d layers, dense layer 0, MoE layer %d "
        "with %d routed + %d shared experts, top-%d routing, d=%d.",
        NUM_LANGUAGE_LAYERS,
        TARGET_LAYER_ZERO_INDEXED,
        NUM_ROUTED_EXPERTS,
        NUM_SHARED_EXPERTS,
        TOP_K_ROUTED,
        HIDDEN_SIZE,
    )


def get_target_moe(model: DeepseekVLV2ForCausalLM) -> DeepseekV2MoE:
    """The frozen activation-site module (language_layers[5].mlp)."""
    return model.language.model.layers[TARGET_LAYER_ZERO_INDEXED].mlp


def load_processor(model_id: str = MODEL_ID) -> DeepseekVLV2Processor:
    return DeepseekVLV2Processor.from_pretrained(model_id)


def load_model(
    model_id: str = MODEL_ID,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[DeepseekVLV2ForCausalLM, FrozenModelInfo]:
    """Load the frozen model in eval mode and verify spec §2 assertions.

    The production path is BF16 (spec §3). The FP32 validation gate uses the
    layer-streamed reference forward in capture/validate.py instead of loading
    a second full-precision copy (16 GB host RAM constraint).
    """
    dev = pick_device(device)
    logger.info("Loading %s in %s on %s ...", model_id, dtype, dev)
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model = model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)

    assert_frozen_architecture(model)

    info = FrozenModelInfo(
        model_id=model_id,
        hidden_size=HIDDEN_SIZE,
        num_language_layers=NUM_LANGUAGE_LAYERS,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        num_shared_experts=NUM_SHARED_EXPERTS,
        top_k_routed=TOP_K_ROUTED,
        target_layer_zero_indexed=TARGET_LAYER_ZERO_INDEXED,
        dtype=str(dtype),
        device=str(dev),
    )
    return model, info
