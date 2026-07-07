"""Losses (spec §8) against naive per-token/per-slot reference loops."""

import torch

from moe_sae.sae.losses import aggregate_nmse, aux_loss, local_nmse


def test_local_nmse_matches_naive_loop():
    torch.manual_seed(0)
    B, S, d = 7, 6, 5
    c = torch.randn(B, S, d)
    chat = torch.randn(B, S, d)
    mu = torch.randn(B, S, d)

    got = local_nmse(c.reshape(-1, d), chat.reshape(-1, d), mu.reshape(-1, d))

    num = 0.0
    den = 0.0
    for t in range(B):
        for s in range(S):
            num += float((c[t, s] - chat[t, s]).pow(2).sum())
            den += float((c[t, s] - mu[t, s]).pow(2).sum())
    assert abs(float(got) - num / den) < 1e-5
    assert got.dtype == torch.float32


def test_local_nmse_differentiable_through_chat_only():
    c = torch.randn(4, 3)
    mu = torch.zeros(4, 3)
    chat = torch.randn(4, 3, requires_grad=True)
    loss = local_nmse(c, chat, mu)
    loss.backward()
    # d/dchat of ||c-chat||²/den = -2(c-chat)/den
    den = (c - mu).pow(2).sum()
    want = -2.0 * (c - chat.detach()) / den
    assert torch.allclose(chat.grad, want, atol=1e-6)


def test_aggregate_nmse_matches_naive_loop():
    torch.manual_seed(1)
    B, d = 11, 5
    c = torch.randn(B, d)
    chat = torch.randn(B, d)
    got = aggregate_nmse(c, chat)

    cbar = c.mean(dim=0)
    num = sum(float((c[t] - chat[t]).pow(2).sum()) for t in range(B))
    den = sum(float((c[t] - cbar).pow(2).sum()) for t in range(B))
    assert abs(float(got) - num / den) < 1e-5


def test_aggregate_nmse_is_no_grad():
    c = torch.randn(4, 3)
    chat = torch.randn(4, 3, requires_grad=True)
    out = aggregate_nmse(c, chat)
    assert not out.requires_grad  # diagnostic only, never optimized (spec §8)


def test_aux_loss_matches_naive_loop():
    torch.manual_seed(2)
    M, d = 9, 4
    r = torch.randn(M, d)
    rhat = torch.randn(M, d)
    got = aux_loss(r, rhat)
    num = sum(float((r[i] - rhat[i]).pow(2).sum()) for i in range(M))
    den = sum(float(r[i].pow(2).sum()) for i in range(M))
    assert abs(float(got) - num / den) < 1e-5


def test_aux_loss_tiny_residual_guard():
    r = torch.zeros(5, 4)
    rhat = torch.randn(5, 4)
    got = aux_loss(r, rhat)
    assert float(got) == 0.0  # Σ‖r‖² < 1e-12 contributes 0 (contract)


def test_losses_fp32_reduction_from_bf16_inputs():
    c = torch.randn(6, 4).to(torch.bfloat16)
    chat = torch.randn(6, 4).to(torch.bfloat16)
    mu = torch.zeros(6, 4).to(torch.bfloat16)
    assert local_nmse(c, chat, mu).dtype == torch.float32
    assert aggregate_nmse(c.float(), chat.float()).dtype == torch.float32
