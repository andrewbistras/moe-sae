"""SpecAdam (spec §11; contract §Optimizer): step order, decoder tangent-space
projection, unit-norm renormalization, FP32 moments on BF16 params, global
grad clip, plain-Adam agreement on non-decoder params."""

import math

import torch
from torch import nn

from moe_sae.sae.optim import SpecAdam, project_decoder_grad


def make_decoder(d=6, cols=4, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(d, cols, generator=g)
    D = D / D.norm(dim=0, keepdim=True)
    return nn.Parameter(D.to(dtype))


def test_project_removes_parallel_component():
    torch.manual_seed(0)
    d = 8
    col = torch.randn(d, 1)
    col = col / col.norm()
    g = torch.randn(d, 1)
    proj = project_decoder_grad(g, col, axis=0)
    # analytic: g − (g·d)d, and orthogonal to the column
    want = g - (g * col).sum() * col
    assert torch.allclose(proj, want, atol=1e-6)
    assert abs(float((proj * col).sum())) < 1e-6


def test_first_moment_is_projected_gradient():
    """After one step from fresh state, exp_avg == (1−β1)·(clipped, projected)
    gradient — analytic check on a 1-column toy."""
    D = make_decoder(d=5, cols=1)
    opt = SpecAdam([{"params": [D], "decoder_axis": 0}])
    g = torch.randn(5, 1) * 0.01  # below clip threshold → clip_coef == 1
    col = D.detach().clone()
    D.grad = g.clone()
    opt.step(lr=1e-3)
    g_proj = g - (g * col).sum() * col
    m = opt.state[id(D)]["exp_avg"]
    assert torch.allclose(m, 0.1 * g_proj, atol=1e-7)


def test_parallel_only_gradient_leaves_column_unchanged():
    """A gradient exactly parallel to a unit column is fully projected out:
    zero moments, zero update, unit renorm → the column is a fixed point.

    Uses exactly-representable one-hot unit columns so the projection is
    bitwise zero (with generic fp32 columns a ~1e-9 rounding residue remains,
    which Adam's elementwise normalization would amplify — that regime never
    occurs in training, where gradients carry real perpendicular signal)."""
    D = nn.Parameter(torch.eye(7)[:, :3].clone())  # one-hot unit columns
    before = D.detach().clone()
    opt = SpecAdam([{"params": [D], "decoder_axis": 0}])
    D.grad = before * torch.tensor([0.5, -0.2, 0.03])  # parallel per column
    opt.step(lr=0.1)
    assert torch.equal(D.detach(), before)


def test_unit_norm_after_every_step():
    torch.manual_seed(1)
    D = make_decoder(d=6, cols=5)
    W = nn.Parameter(torch.randn(5, 6))
    opt = SpecAdam(
        [
            {"params": [W], "decoder_axis": None},
            {"params": [D], "decoder_axis": 0},
        ]
    )
    for _ in range(5):
        D.grad = torch.randn_like(D)
        W.grad = torch.randn_like(W)
        opt.step(lr=0.05)
        norms = D.detach().norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_bf16_params_fp32_moments():
    D = make_decoder(d=6, cols=4, dtype=torch.bfloat16)
    W = nn.Parameter(torch.randn(4, 6).to(torch.bfloat16))
    opt = SpecAdam(
        [
            {"params": [W], "decoder_axis": None},
            {"params": [D], "decoder_axis": 0},
        ]
    )
    for p in (W, D):
        st = opt.state[id(p)]
        assert st["exp_avg"].dtype == torch.float32
        assert st["exp_avg_sq"].dtype == torch.float32
    W.grad = torch.randn_like(W)
    D.grad = torch.randn_like(D)
    opt.step(lr=1e-3)
    assert W.dtype == torch.bfloat16 and D.dtype == torch.bfloat16
    norms = D.detach().float().norm(dim=0)
    assert (norms - 1.0).abs().max() < 0.02  # unit up to bf16 rounding


def test_global_clip_scale_invariance():
    """Two grad sets differing by a positive scalar, both above the clip
    threshold, produce identical steps — i.e. the clip normalized both to
    global norm 1.0."""
    results = []
    for scale in (10.0, 1000.0):
        torch.manual_seed(2)
        p1 = nn.Parameter(torch.randn(3, 4))
        p2 = nn.Parameter(torch.randn(7))
        opt = SpecAdam([{"params": [p1, p2], "decoder_axis": None}])
        g1, g2 = torch.randn(3, 4), torch.randn(7)
        p1.grad = g1 * scale
        p2.grad = g2 * scale
        pre_norm = opt.step(lr=1e-2)
        assert abs(pre_norm - float((g1.pow(2).sum() + g2.pow(2).sum()).sqrt()) * scale) < 1e-2
        results.append((p1.detach().clone(), p2.detach().clone()))
    assert torch.allclose(results[0][0], results[1][0], atol=1e-6)
    assert torch.allclose(results[0][1], results[1][1], atol=1e-6)


def test_matches_naive_adam_on_non_decoder_param():
    """Full step math (clip → Adam with bias correction, eps=1e-8) against a
    naive reference implementation."""
    torch.manual_seed(3)
    p = nn.Parameter(torch.randn(5, 3))
    ref = p.detach().clone().double()
    opt = SpecAdam([{"params": [p], "decoder_axis": None}])
    b1, b2, eps, clip = 0.9, 0.999, 1e-8, 1.0
    m = torch.zeros_like(ref)
    v = torch.zeros_like(ref)
    for t in range(1, 4):
        g = torch.randn(5, 3)
        p.grad = g.clone()
        opt.step(lr=1e-3)
        # naive reference
        gn = float(g.double().norm())
        gd = g.double() * min(1.0, clip / (gn + 1e-6))
        m = b1 * m + (1 - b1) * gd
        v = b2 * v + (1 - b2) * gd * gd
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        ref = ref - 1e-3 * mhat / (vhat.sqrt() + eps)
        assert torch.allclose(p.detach().double(), ref, atol=1e-5)


def test_step_order_projection_before_adam_renorm_after():
    """Perpendicular gradient: the Adam update uses the projected gradient
    (elementwise sign at step 1), then columns are renormalized."""
    d = 4
    col = torch.zeros(d, 1)
    col[0] = 1.0
    D = nn.Parameter(col.clone())
    opt = SpecAdam([{"params": [D], "decoder_axis": 0}])
    g = torch.zeros(d, 1)
    g[1] = 0.5  # already perpendicular to col, below clip
    D.grad = g.clone()
    lr = 0.01
    opt.step(lr=lr)
    # step 1 Adam with fresh moments: update = lr·ĝ/(√ĝ²+eps) ≈ lr·sign(g)
    raw = col - lr * torch.sign(g)
    want = raw / raw.norm()
    assert torch.allclose(D.detach(), want, atol=1e-5)
    assert abs(float(D.detach().norm()) - 1.0) < 1e-6
