"""SpecAdam (spec §11; contract §Optimizer). Frozen per-step order:

1. global grad-norm clip 1.0 over ALL params (FP32 accumulation);
2. for decoder params: project out the gradient component parallel to each
   unit decoder column (tangent-space projection, changelog #5);
3. Adam moment update + step — moments FP32 even on BF16 params, step math
   FP32, update cast back to parameter dtype;
4. renormalize decoder columns to unit ℓ2 norm.

Constructor takes explicit groups: [{'params': [...], 'decoder_axis': None|int}]
where `decoder_axis` is the axis of the tensor along which each decoder column
lies (RouterTiedBanks.D [E,d,f] → 1; GlobalSAE.D [d,F] → 0). `step(lr)` takes
the externally scheduled LR (the schedule lives in moe_sae.train.loop).

Decisions (recorded): Adam eps = 1e-8 (torch default; spec does not pin it);
bias correction applied (standard Adam); the projection normalizes the stored
column direction first so BF16 rounding of the unit-norm invariant cannot
leave a parallel residue.
"""

from __future__ import annotations

import torch

from moe_sae.train.config import ADAM_BETAS, GRAD_CLIP


def project_decoder_grad(
    grad32: torch.Tensor, param32: torch.Tensor, axis: int
) -> torch.Tensor:
    """g_j ← g_j − (g_j·d̂_j) d̂_j per decoder column (FP32 in, FP32 out)."""
    norm = param32.norm(dim=axis, keepdim=True).clamp_min(1e-12)
    dhat = param32 / norm
    parallel = (grad32 * dhat).sum(dim=axis, keepdim=True)
    return grad32 - parallel * dhat


class SpecAdam:
    """Adam with FP32 moments on (possibly BF16) params + decoder constraint."""

    def __init__(
        self,
        groups: list[dict],
        betas: tuple[float, float] = ADAM_BETAS,
        eps: float = 1e-8,
        clip: float = GRAD_CLIP,
    ):
        self.groups = groups
        self.betas = betas
        self.eps = eps
        self.clip = clip
        self.t = 0
        self.state: dict[int, dict[str, torch.Tensor]] = {}
        for g in groups:
            assert set(g) >= {"params", "decoder_axis"}, g.keys()
            for p in g["params"]:
                self.state[id(p)] = {
                    "exp_avg": torch.zeros_like(p, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(p, dtype=torch.float32),
                }

    def _params(self):
        for g in self.groups:
            for p in g["params"]:
                yield p, g["decoder_axis"]

    def zero_grad(self) -> None:
        for p, _ in self._params():
            p.grad = None

    @torch.no_grad()
    def step(self, lr: float) -> float:
        """One update at the given scheduled LR. Returns the pre-clip global
        grad norm (diagnostic)."""
        self.t += 1
        b1, b2 = self.betas

        # 1. Global grad-norm clip over ALL params, FP32 accumulation.
        sq = torch.zeros((), dtype=torch.float32)
        for p, _ in self._params():
            if p.grad is not None:
                sq += p.grad.float().pow(2).sum().cpu()
        total_norm = float(sq.sqrt().item())
        clip_coef = min(1.0, self.clip / (total_norm + 1e-6))

        for p, decoder_axis in self._params():
            if p.grad is None:
                continue
            g = p.grad.float() * clip_coef
            p32 = p.float()

            # 2. Tangent-space projection for decoder columns.
            if decoder_axis is not None:
                g = project_decoder_grad(g, p32, decoder_axis)

            # 3. Adam in FP32.
            st = self.state[id(p)]
            m, v = st["exp_avg"], st["exp_avg_sq"]
            m.mul_(b1).add_(g, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
            m_hat = m / (1.0 - b1**self.t)
            v_hat = v / (1.0 - b2**self.t)
            p32 = p32 - lr * m_hat / (v_hat.sqrt() + self.eps)

            # 4. Renormalize decoder columns to unit norm.
            if decoder_axis is not None:
                p32 = p32 / p32.norm(dim=decoder_axis, keepdim=True).clamp_min(1e-12)

            p.copy_(p32.to(p.dtype))
        return total_norm
