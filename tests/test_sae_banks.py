"""RouterTiedBanks / GlobalSAE: grouped-GEMM encode/decode vs a naive per-row
loop (exact equality via integer-valued inputs), autograd correctness, init
invariants (spec §5)."""

import pytest
import torch

from moe_sae.sae.banks import GlobalSAE, RouterTiedBanks

E, F_BANK, D = 8, 5, 7


def make_banks(dtype=torch.float32, seed=0) -> RouterTiedBanks:
    return RouterTiedBanks(E, F_BANK, D, dtype=dtype, seed=seed)


def randint_(shape, lo=-3, hi=4, dtype=torch.float32):
    """Small-integer-valued floats: every sum-of-products is exact in fp32,
    so grouped vs per-row accumulation order cannot differ."""
    return torch.randint(lo, hi, shape).to(dtype)


def set_integer_params(banks):
    with torch.no_grad():
        banks.W_enc.copy_(randint_(banks.W_enc.shape))
        banks.b_enc.copy_(randint_(banks.b_enc.shape))
        banks.D.copy_(randint_(banks.D.shape))
        banks.mu.copy_(randint_(banks.mu.shape))


def naive_encode(banks, c, bank_ids):
    B, S, _ = c.shape
    out = torch.zeros(B, S, banks.f, dtype=banks.W_enc.dtype)
    for b in range(B):
        for s in range(S):
            e = int(bank_ids[b, s])
            tilde = c[b, s].to(banks.W_enc.dtype) - banks.mu[e].to(banks.W_enc.dtype)
            out[b, s] = torch.relu(banks.W_enc[e] @ tilde + banks.b_enc[e])
    return out


def naive_decode(banks, z, bank_ids, add_mu):
    B, S, _ = z.shape
    out = torch.zeros(B, S, banks.d, dtype=banks.D.dtype)
    for b in range(B):
        for s in range(S):
            e = int(bank_ids[b, s])
            rec = banks.D[e] @ z[b, s].to(banks.D.dtype)
            if add_mu:
                rec = rec + banks.mu[e].to(banks.D.dtype)
            out[b, s] = rec
    return out


def test_init_invariants():
    torch.manual_seed(0)
    banks = make_banks()
    # decoder columns D[e, :, j] unit norm
    norms = banks.D.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    # W_enc initialized to the per-bank decoder transpose; b_enc zero
    assert torch.equal(banks.W_enc, banks.D.transpose(1, 2))
    assert torch.equal(banks.b_enc, torch.zeros_like(banks.b_enc))
    # mu is a frozen buffer, never a Parameter
    assert "mu" in dict(banks.named_buffers())
    assert "mu" not in dict(banks.named_parameters())
    assert banks.mu.dtype == torch.float32
    # seeded init is deterministic and torch-seed-independent
    torch.manual_seed(1234)
    banks2 = make_banks(seed=0)
    assert torch.equal(banks.D, banks2.D)


def test_bf16_param_dtype():
    banks = make_banks(dtype=torch.bfloat16)
    assert banks.W_enc.dtype == torch.bfloat16
    assert banks.D.dtype == torch.bfloat16
    assert banks.mu.dtype == torch.float32  # buffer stays fp32


def test_encode_matches_naive_exactly():
    torch.manual_seed(0)
    banks = make_banks()
    set_integer_params(banks)
    B, S = 11, 6
    c = randint_((B, S, D))
    bank_ids = torch.randint(0, E, (B, S))
    got = banks.encode(c, bank_ids)
    want = naive_encode(banks, c, bank_ids)
    assert torch.equal(got, want)


def test_encode_handles_absent_banks():
    torch.manual_seed(1)
    banks = make_banks()
    set_integer_params(banks)
    c = randint_((4, 2, D))
    bank_ids = torch.full((4, 2), 3, dtype=torch.int64)  # only bank 3 present
    got = banks.encode(c, bank_ids)
    assert torch.equal(got, naive_encode(banks, c, bank_ids))


def test_decode_matches_naive_exactly():
    torch.manual_seed(2)
    banks = make_banks()
    set_integer_params(banks)
    B, S = 9, 6
    z = randint_((B, S, F_BANK), lo=0, hi=3)
    bank_ids = torch.randint(0, E, (B, S))
    for add_mu in (True, False):
        got = banks.decode(z, bank_ids, add_mu=add_mu)
        assert torch.equal(got, naive_decode(banks, z, bank_ids, add_mu))


def test_encode_decode_autograd_matches_naive():
    """Gradients through the grouped path == gradients through the per-row
    loop (same Parameters, two graphs)."""
    torch.manual_seed(3)
    banks = make_banks()
    set_integer_params(banks)
    B, S = 5, 3
    c = randint_((B, S, D))
    bank_ids = torch.randint(0, E, (B, S))

    loss_grouped = banks.decode(banks.encode(c, bank_ids), bank_ids, add_mu=True).pow(2).sum()
    loss_grouped.backward()
    grads_grouped = {
        n: p.grad.clone() for n, p in banks.named_parameters() if p.grad is not None
    }
    for p in banks.parameters():
        p.grad = None

    loss_naive = naive_decode(
        banks, naive_encode(banks, c, bank_ids), bank_ids, add_mu=True
    ).pow(2).sum()
    loss_naive.backward()
    assert torch.equal(loss_grouped.detach(), loss_naive.detach())
    for n, p in banks.named_parameters():
        assert torch.allclose(grads_grouped[n], p.grad, atol=1e-4), n


def test_set_mu_and_centering():
    banks = make_banks()
    mu = torch.randn(E, D)
    banks.set_mu(mu)
    assert torch.equal(banks.mu, mu)
    # encode of c == mu with zero bias gives ReLU(0) == 0
    with torch.no_grad():
        banks.b_enc.zero_()
    bank_ids = torch.arange(E).reshape(E, 1) % E
    c = mu.reshape(E, 1, D)
    assert torch.equal(banks.encode(c, bank_ids), torch.zeros(E, 1, F_BANK))


def test_global_sae_matches_naive():
    torch.manual_seed(4)
    g = GlobalSAE(13, D, dtype=torch.float32, seed=0)
    with torch.no_grad():
        g.W_enc.copy_(randint_(g.W_enc.shape))
        g.b_enc.copy_(randint_(g.b_enc.shape))
        g.D.copy_(randint_(g.D.shape))
        g.mu.copy_(randint_(g.mu.shape))
    x = randint_((6, D))
    pre = g.encode(x)
    want = torch.stack([torch.relu(g.W_enc @ (x[i] - g.mu) + g.b_enc) for i in range(6)])
    assert torch.equal(pre, want)
    z = randint_((6, 13), lo=0, hi=3)
    rec = g.decode(z, add_mu=True)
    want_rec = torch.stack([g.D @ z[i] + g.mu for i in range(6)])
    assert torch.equal(rec, want_rec)
    assert torch.equal(g.decode(z, add_mu=False), want_rec - g.mu)


def test_global_sae_init_invariants():
    g = GlobalSAE(13, D, dtype=torch.float32, seed=7)
    norms = g.D.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    assert torch.equal(g.W_enc, g.D.T)
    assert "mu" not in dict(g.named_parameters())
    assert g.DECODER_AXIS == 0 and RouterTiedBanks.DECODER_AXIS == 1
