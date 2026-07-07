"""Capture-hook validity: bit-identity of the instrumented path, agreement of
the independent reference recomputation (spec §3)."""

import pytest
import torch

from moe_sae.capture.hooks import (
    instrumented_capture,
    recompute_contributions,
    reference_capture,
)
from moe_sae.vendor.deepseek_vl2.configuration_deepseek import DeepseekV2Config
from moe_sae.vendor.deepseek_vl2.modeling_deepseek import DeepseekV2MoE


def make_moe(dtype: torch.dtype, device: str = "cpu", seed: int = 0) -> DeepseekV2MoE:
    torch.manual_seed(seed)
    cfg = DeepseekV2Config(
        hidden_size=64,
        moe_intermediate_size=32,
        intermediate_size=128,
        n_routed_experts=16,
        num_experts_per_tok=4,
        n_shared_experts=2,
        topk_method="greedy",
        scoring_func="softmax",
        n_group=1,
        topk_group=1,
        norm_topk_prob=False,
        first_k_dense_replace=0,
        num_hidden_layers=2,
    )
    moe = DeepseekV2MoE(cfg).to(dtype).to(device).eval()
    # Gate weights are used in FP32 internally regardless of module dtype.
    return moe


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_instrumented_bit_identical(dtype):
    moe = make_moe(dtype)
    x = torch.randn(3, 17, 64, dtype=dtype)

    with torch.no_grad():
        out_ref = moe(x)

    with instrumented_capture(moe, store_dtype=torch.float32) as sink:
        with torch.no_grad():
            out_inst = moe(x)
        chunks = sink.drain()

    assert torch.equal(out_ref, out_inst), "instrumented moe_infer altered the output"
    assert len(chunks) == 1
    ch = chunks[0]
    n = 3 * 17
    assert ch.topk_idx.shape == (n, 4)
    assert ch.per_slot_contrib.shape == (n, 4, 64)
    # patch removed after context exit
    assert "moe_infer" not in moe.__dict__
    with torch.no_grad():
        out_after = moe(x)
    assert torch.equal(out_ref, out_after)


def test_per_slot_sum_matches_aggregate_fp32():
    moe = make_moe(torch.float32)
    x = torch.randn(2, 31, 64)
    with instrumented_capture(moe, store_dtype=torch.float32) as sink:
        with torch.no_grad():
            moe(x)
        ch = sink.drain()[0]
    with reference_capture(moe) as ref:
        with torch.no_grad():
            moe(x)
        (rx, ridx, rw, ragg) = ref.drain()[0]
    # instrumented per-slot sum vs the model's own aggregate
    rel = (ch.per_slot_contrib.sum(dim=1) - ragg).norm() / ragg.norm()
    assert rel < 1e-6, f"instrumented sum mismatch: rel={rel:.3e}"


def test_reference_recompute_matches_aggregate_fp32():
    moe = make_moe(torch.float32)
    x = torch.randn(4, 13, 64)
    with reference_capture(moe) as ref:
        with torch.no_grad():
            out = moe(x)
        (rx, ridx, rw, ragg) = ref.drain()[0]

    assert rx.shape == (4 * 13, 64)
    per_slot = recompute_contributions(rx, ridx, rw, moe)
    agg = per_slot.sum(dim=1)
    per_token_rel = (agg - ragg).norm(dim=-1) / ragg.norm(dim=-1).clamp_min(1e-30)
    assert per_token_rel.max() < 1e-5, (
        f"independent recomputation disagrees: max rel={per_token_rel.max():.3e}"
    )
    # reference hooks did not perturb the module output path
    with torch.no_grad():
        out2 = moe(x)
    assert torch.equal(out, out2)


def test_instrumented_matches_reference_recompute():
    """Production capture inherits validation from the independent path."""
    moe = make_moe(torch.float32)
    x = torch.randn(5, 9, 64)
    with instrumented_capture(moe, store_dtype=torch.float32) as sink:
        with torch.no_grad():
            moe(x)
        ch = sink.drain()[0]
    with reference_capture(moe) as ref:
        with torch.no_grad():
            moe(x)
        (rx, ridx, rw, _ragg) = ref.drain()[0]

    per_slot_ref = recompute_contributions(rx, ridx, rw, moe)
    # Slot order must agree (same gate topk call), so compare slot-aligned.
    assert torch.equal(ch.topk_idx, ridx)
    rel = (ch.per_slot_contrib - per_slot_ref).norm() / per_slot_ref.norm()
    assert rel < 1e-6, f"production vs reference mismatch: rel={rel:.3e}"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_instrumented_bit_identical_mps():
    moe = make_moe(torch.bfloat16, device="mps")
    x = torch.randn(2, 25, 64, dtype=torch.bfloat16, device="mps")
    with torch.no_grad():
        out_ref = moe(x)
    with instrumented_capture(moe, store_dtype=torch.float32) as sink:
        with torch.no_grad():
            out_inst = moe(x)
        assert len(sink.drain()) == 1
    assert torch.equal(out_ref.cpu(), out_inst.cpu())
