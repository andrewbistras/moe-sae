"""Useful-matmul FLOP ledger for the proposed DeepSeek-R1 prefix tap.

This is an analytic cross-check for the pinned R1-0528 geometry. Production
qualification must additionally reconcile the returned terms with the exact
checkpoint tensor inventory and compiled HLO.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .mfu import FlopTermV1, MFUContractError
from .protocol import canonical_sha256


@dataclass(frozen=True, slots=True)
class DeepSeekR1PrefixGeometryV1:
    """Dimensions and executed layer counts for the one-indexed block-31 tap."""

    hidden_dim: int = 7_168
    attention_heads: int = 128
    q_lora_rank: int = 1_536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    value_head_dim: int = 128
    dense_mlp_dim: int = 18_432
    routed_mlp_dim: int = 2_048
    routed_experts: int = 256
    selected_experts: int = 8
    shared_experts: int = 1
    attention_blocks: int = 31
    dense_blocks: int = 3
    completed_moe_blocks: int = 27

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise MFUContractError(f"{name} must be a positive integer")
        if self.dense_blocks + self.completed_moe_blocks + 1 != self.attention_blocks:
            raise MFUContractError(
                "dense blocks, completed MoE blocks, and target must match attention blocks"
            )
        if self.selected_experts > self.routed_experts:
            raise MFUContractError("selected_experts cannot exceed routed_experts")


DEEPSEEK_R1_PREFIX_GEOMETRY_V1 = DeepSeekR1PrefixGeometryV1()
DEEPSEEK_R1_PREFIX_FORMULA_VERSION = "deepseek-r1-prefix-useful-bf16-v1"


def deepseek_r1_prefix_flop_coefficients(
    geometry: DeepSeekR1PrefixGeometryV1 | None = None,
) -> dict[str, int]:
    """Return the sole frozen coefficient ledger for the analytic reducer."""

    g = geometry or DEEPSEEK_R1_PREFIX_GEOMETRY_V1
    d = g.hidden_dim
    h = g.attention_heads
    rq = g.q_lora_rank
    rkv = g.kv_lora_rank
    qn = g.qk_nope_head_dim
    qr = g.qk_rope_head_dim
    v = g.value_head_dim
    expert = 6 * d * g.routed_mlp_dim
    return {
        "attention_products_per_causal_pair": (
            g.attention_blocks * 2 * h * (qn + qr + v)
        ),
        "attention_projections_per_valid_token": (
            g.attention_blocks
            * 2
            * (
                d * rq
                + rq * h * (qn + qr)
                + d * (rkv + qr)
                + rkv * h * (qn + v)
                + h * v * d
            )
        ),
        "completed_moe_blocks_per_valid_token": (
            g.completed_moe_blocks * expert * (g.selected_experts + g.shared_experts)
        ),
        "dense_ffn_blocks_per_valid_token": (g.dense_blocks * 6 * d * g.dense_mlp_dim),
        "target_routed_experts_per_valid_token": (expert * g.selected_experts),
        "excluded_fp32_router_flops_per_valid_token": (
            (g.completed_moe_blocks + 1) * 2 * d * g.routed_experts
        ),
    }


def deepseek_r1_prefix_formula_manifest() -> dict[str, object]:
    """Return the frozen analytic formula identity used by every reducer.

    The implementation/checkpoint/HLO identities remain separate runtime
    evidence.  This digest prevents a second protocol layer from silently
    restating or weakening the analytic geometry.
    """

    return {
        "formula_version": DEEPSEEK_R1_PREFIX_FORMULA_VERSION,
        "geometry": asdict(DEEPSEEK_R1_PREFIX_GEOMETRY_V1),
        "coefficients": deepseek_r1_prefix_flop_coefficients(),
        "counted_dtype": "bfloat16",
        "router_policy": "FP32_ROUTER_EXCLUDED_FROM_BF16_NUMERATOR",
        "terms": [
            "attention_products",
            "attention_projections",
            "completed_moe_blocks",
            "dense_ffn_blocks",
            "target_routed_experts",
        ],
    }


DEEPSEEK_R1_PREFIX_FORMULA_SHA256 = canonical_sha256(
    deepseek_r1_prefix_formula_manifest()
)


def causal_attention_pairs(segment_lengths: Iterable[int]) -> tuple[int, int]:
    """Return valid tokens and exact causal QK/AV pairs for packed segments."""

    total_tokens = 0
    total_pairs = 0
    saw_segment = False
    for length in segment_lengths:
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise MFUContractError("packed segment lengths must be positive integers")
        saw_segment = True
        total_tokens += length
        total_pairs += length * (length + 1) // 2
    if not saw_segment:
        raise MFUContractError("at least one packed segment is required")
    return total_tokens, total_pairs


def deepseek_r1_prefix_flop_terms(
    *,
    valid_tokens: int,
    causal_pairs: int,
    geometry: DeepSeekR1PrefixGeometryV1 | None = None,
) -> tuple[FlopTermV1, ...]:
    """Return sorted useful-FLOP terms through the target routed aggregate.

    ``causal_pairs`` is the sum of ``s * (s + 1) / 2`` over actual packed
    segments. It must never be derived from the rectangular padded shape.
    """

    if (
        isinstance(valid_tokens, bool)
        or not isinstance(valid_tokens, int)
        or valid_tokens <= 0
    ):
        raise MFUContractError("valid_tokens must be a positive integer")
    if (
        isinstance(causal_pairs, bool)
        or not isinstance(causal_pairs, int)
        or causal_pairs <= 0
    ):
        raise MFUContractError("causal_pairs must be a positive integer")
    maximum_pairs = valid_tokens * (valid_tokens + 1) // 2
    if causal_pairs < valid_tokens or causal_pairs > maximum_pairs:
        raise MFUContractError("causal_pairs is incompatible with valid_tokens")

    coefficients = deepseek_r1_prefix_flop_coefficients(geometry)

    return (
        FlopTermV1(
            "attention_products",
            coefficients["attention_products_per_causal_pair"] * causal_pairs,
            "exact causal MLA QK and AV products over packed segment pairs",
        ),
        FlopTermV1(
            "attention_projections",
            coefficients["attention_projections_per_valid_token"] * valid_tokens,
            "MLA projections for every executed attention block",
        ),
        FlopTermV1(
            "completed_moe_blocks",
            coefficients["completed_moe_blocks_per_valid_token"] * valid_tokens,
            "eight routed experts and one shared expert in completed MoE blocks; "
            "FP32 router projections excluded from the BF16 numerator",
        ),
        FlopTermV1(
            "dense_ffn_blocks",
            coefficients["dense_ffn_blocks_per_valid_token"] * valid_tokens,
            "three gated dense FFN matmuls in each initial dense block",
        ),
        FlopTermV1(
            "target_routed_experts",
            coefficients["target_routed_experts_per_valid_token"] * valid_tokens,
            "eight target routed experts; FP32 router projection, shared expert, and suffix excluded",
        ),
    )


def deepseek_r1_fp32_router_flops(
    *,
    valid_tokens: int,
    geometry: DeepSeekR1PrefixGeometryV1 | None = None,
) -> int:
    """Return executed FP32 router matmul FLOPs for separate diagnostics.

    DeepSeek-R1 computes router logits in FP32. These operations execute and
    belong in the profiler/cost ledger, but they are deliberately excluded
    from the useful-BF16 numerator divided by TPU BF16 peak.
    """

    if (
        isinstance(valid_tokens, bool)
        or not isinstance(valid_tokens, int)
        or valid_tokens <= 0
    ):
        raise MFUContractError("valid_tokens must be a positive integer")
    coefficients = deepseek_r1_prefix_flop_coefficients(geometry)
    return valid_tokens * coefficients["excluded_fp32_router_flops_per_valid_token"]


def deepseek_r1_prefix_useful_flops(
    *,
    segment_lengths: Iterable[int],
    geometry: DeepSeekR1PrefixGeometryV1 | None = None,
) -> int:
    """Convenience reduction of the exact packed-segment FLOP terms."""

    valid_tokens, pairs = causal_attention_pairs(segment_lengths)
    return sum(
        term.useful_flops
        for term in deepseek_r1_prefix_flop_terms(
            valid_tokens=valid_tokens,
            causal_pairs=pairs,
            geometry=geometry,
        )
    )


def dense_scope_sae_flops(*, tokens: int, hidden_dim: int, features: int) -> int:
    """Dense encoder+decoder forward/backward convention used for Scope comparison."""

    for name, value in (
        ("tokens", tokens),
        ("hidden_dim", hidden_dim),
        ("features", features),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MFUContractError(f"{name} must be a positive integer")
    return 12 * tokens * hidden_dim * features


def sparse_selected_decode_sae_flops(
    *, tokens: int, hidden_dim: int, features: int, top_k: int
) -> int:
    """Frozen conventional model FLOPs for dense encode and selected-column decode."""

    for name, value in (
        ("tokens", tokens),
        ("hidden_dim", hidden_dim),
        ("features", features),
        ("top_k", top_k),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MFUContractError(f"{name} must be a positive integer")
    if top_k > features:
        raise MFUContractError("top_k cannot exceed features")
    return 6 * tokens * hidden_dim * (features + top_k)


__all__ = [
    "DEEPSEEK_R1_PREFIX_FORMULA_SHA256",
    "DEEPSEEK_R1_PREFIX_FORMULA_VERSION",
    "DEEPSEEK_R1_PREFIX_GEOMETRY_V1",
    "DeepSeekR1PrefixGeometryV1",
    "causal_attention_pairs",
    "dense_scope_sae_flops",
    "deepseek_r1_fp32_router_flops",
    "deepseek_r1_prefix_flop_coefficients",
    "deepseek_r1_prefix_flop_terms",
    "deepseek_r1_prefix_formula_manifest",
    "deepseek_r1_prefix_useful_flops",
    "sparse_selected_decode_sae_flops",
]
