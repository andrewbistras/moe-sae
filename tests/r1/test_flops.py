from __future__ import annotations

from dataclasses import replace

import pytest

from moe_sae.r1.flops import (
    DeepSeekR1PrefixGeometryV1,
    causal_attention_pairs,
    dense_scope_sae_flops,
    deepseek_r1_fp32_router_flops,
    deepseek_r1_prefix_flop_terms,
    deepseek_r1_prefix_useful_flops,
    sparse_selected_decode_sae_flops,
)
from moe_sae.r1.mfu import MFUContractError


def test_full_1024_token_segment_matches_independent_r1_crosscheck() -> None:
    useful = deepseek_r1_prefix_useful_flops(segment_lengths=(1024,))
    assert useful == 38_285_699_186_688
    assert useful / 1024 == pytest.approx(37.388378112e9)
    assert deepseek_r1_fp32_router_flops(valid_tokens=1024) == 105_226_698_752


def test_packing_counts_actual_causal_pairs_not_rectangular_padding() -> None:
    tokens, split_pairs = causal_attention_pairs((512, 512))
    assert tokens == 1024
    assert split_pairs == 2 * (512 * 513 // 2)
    split = deepseek_r1_prefix_useful_flops(segment_lengths=(512, 512))
    unsplit = deepseek_r1_prefix_useful_flops(segment_lengths=(1024,))
    assert split < unsplit
    expected_attention_difference = (
        31 * 2 * 128 * (128 + 64 + 128) * ((1024 * 1025 // 2) - split_pairs)
    )
    assert unsplit - split == expected_attention_difference


def test_terms_are_sorted_and_exclude_target_shared_expert() -> None:
    tokens, pairs = causal_attention_pairs((16, 8))
    terms = deepseek_r1_prefix_flop_terms(valid_tokens=tokens, causal_pairs=pairs)
    assert tuple(term.name for term in terms) == tuple(
        sorted(term.name for term in terms)
    )

    target = next(term for term in terms if term.name == "target_routed_experts")
    d = 7168
    expected_per_token = 6 * d * 2048 * 8
    assert target.useful_flops == tokens * expected_per_token

    descriptions = " ".join(term.derivation for term in terms)
    assert "FP32 router" in descriptions


def test_geometry_and_counts_fail_closed() -> None:
    with pytest.raises(MFUContractError, match="must match"):
        replace(DeepSeekR1PrefixGeometryV1(), completed_moe_blocks=26)
    with pytest.raises(MFUContractError, match="positive integers"):
        causal_attention_pairs((32, True))
    with pytest.raises(MFUContractError, match="incompatible"):
        deepseek_r1_prefix_flop_terms(valid_tokens=10, causal_pairs=1000)


def test_dense_scope_and_sparse_f_conventions_are_not_conflated() -> None:
    dense = dense_scope_sae_flops(tokens=16_384, hidden_dim=7_168, features=131_072)
    sparse = sparse_selected_decode_sae_flops(
        tokens=16_384,
        hidden_dim=7_168,
        features=131_072,
        top_k=64,
    )
    assert dense == 184_717_953_466_368
    assert sparse == 92_404_073_889_792
    assert sparse < dense

    with pytest.raises(MFUContractError, match="cannot exceed"):
        sparse_selected_decode_sae_flops(
            tokens=1, hidden_dim=7_168, features=32, top_k=64
        )
