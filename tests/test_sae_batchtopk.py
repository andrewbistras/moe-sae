"""BatchTopK selection engine (spec §6): exact-K budgets, absence of per-token
constraints in the global variant, per-token exactness for E, largest-remainder
budgets for D, θ-EMA mechanics, determinism."""

import torch

from moe_sae.sae.batchtopk import (
    ThetaEMA,
    global_batch_topk,
    largest_remainder_budgets,
    per_bank_topk,
    per_token_topk,
    thresholded_inference,
)


def test_global_exact_k_and_theta():
    torch.manual_seed(0)
    scores = torch.rand(1000)
    k = 137
    mask, theta_b = global_batch_topk(scores, k)
    assert int(mask.sum()) == k
    sorted_desc = torch.sort(scores, descending=True).values
    assert theta_b == sorted_desc[k - 1].item()
    # selected set == top-k by value
    assert scores[mask].min() >= scores[~mask].max()


def test_global_no_per_token_constraint():
    """One token can take most of the budget (spec §6: 'that allocation is
    the phenomenon under investigation')."""
    B, n_per_tok, k = 8, 16, 20
    scores = torch.zeros(B, n_per_tok)
    scores[0] = 100.0 + torch.arange(n_per_tok).float()  # token 0 dominates
    scores[1:, 0] = 1.0
    mask, _ = global_batch_topk(scores.reshape(-1), k)
    per_token = mask.view(B, n_per_tok).sum(dim=1)
    assert per_token[0] == n_per_tok  # all 16 of token 0's candidates selected
    assert int(per_token.sum()) == k
    assert per_token.max() > per_token.min()  # unequal allocation


def test_global_selects_zeros_when_short_of_positives():
    """Exactly K always selected; zero-valued candidates may be included."""
    scores = torch.zeros(50)
    scores[:3] = torch.tensor([5.0, 4.0, 3.0])
    mask, theta_b = global_batch_topk(scores, 10)
    assert int(mask.sum()) == 10
    assert theta_b == 0.0


def test_per_token_exact_k():
    torch.manual_seed(1)
    scores = torch.rand(32, 48)
    k = 7
    mask, thetas = per_token_topk(scores, k)
    assert torch.equal(mask.sum(dim=1), torch.full((32,), k))
    want = torch.sort(scores, dim=1, descending=True).values[:, k - 1]
    assert torch.allclose(thetas, want)


def test_per_bank_topk_budgets_and_theta():
    torch.manual_seed(2)
    n, E = 300, 4
    scores = torch.rand(n)
    bank_of = torch.randint(0, E, (n,))
    budgets = torch.tensor([5, 0, 11, 3])
    mask, theta = per_bank_topk(scores, bank_of, budgets)
    assert int(mask.sum()) == 19
    for e in range(E):
        sel_e = mask & (bank_of == e)
        assert int(sel_e.sum()) == int(budgets[e])
        if budgets[e] > 0:
            # within-bank top-k property
            assert scores[sel_e].min() >= scores[(bank_of == e) & ~mask].max()
    assert theta == scores[mask].min().item()


def test_largest_remainder_budgets_sum_and_proportionality():
    loads = torch.tensor([384.0, 200.0, 0.0, 8.0, 100.0])
    total = 64 * 4
    budgets = largest_remainder_budgets(loads, total)
    assert budgets.sum() == total
    assert budgets[2] == 0  # zero-load bank gets nothing
    quota = loads * total / loads.sum()
    assert ((budgets.float() - quota).abs() < 1.0).all()  # LR rounding property


def test_largest_remainder_deterministic_on_ties():
    loads = torch.tensor([1.0, 1.0, 1.0, 1.0])
    b1 = largest_remainder_budgets(loads, 6)
    b2 = largest_remainder_budgets(loads, 6)
    assert torch.equal(b1, b2)
    assert b1.sum() == 6
    # ties resolve to lower indices (stable sort)
    assert torch.equal(b1, torch.tensor([2, 2, 1, 1]))


def test_thresholded_inference_strict_inequality():
    a = torch.tensor([0.0, 0.5, 1.0, 1.5])
    z = thresholded_inference(a, 1.0)
    assert torch.equal(z, torch.tensor([0.0, 0.0, 0.0, 1.5]))  # a == θ* → 0
    assert z.dtype == torch.float32


def test_selection_determinism():
    torch.manual_seed(3)
    scores = torch.rand(500, dtype=torch.float32)
    m1, t1 = global_batch_topk(scores, 64)
    m2, t2 = global_batch_topk(scores.clone(), 64)
    assert torch.equal(m1, m2) and t1 == t2


def test_theta_ema_init_update_and_state():
    ema = ThetaEMA()  # frozen default decay 0.999
    assert ema.decay == 0.999
    assert ema.value is None
    ema.update(2.0)
    assert ema.value == 2.0  # initialized to first batch's θ_b (spec §6)
    ema.update(1.0)
    assert abs(ema.value - (0.999 * 2.0 + 0.001 * 1.0)) < 1e-12

    state = ema.state_dict()
    ema2 = ThetaEMA()
    ema2.load_state_dict(state)
    assert ema2.value == ema.value and ema2.decay == ema.decay
    ema2.update(0.0)
    assert abs(ema2.value - 0.999 * ema.value) < 1e-12
