"""Training losses (spec §8). All reductions in FP32 (contract §Numerics).

L_local (primary, optimized):
    Σ_{t,e∈S_t} ‖c_te − ĉ_te‖² / Σ_{t,e∈S_t} ‖c_te − μ_e‖²
— per-(t,e) sums, so no bank can use its decoder to cancel another bank's
errors. The denominator uses the same batch.

L_aggregate (diagnostic, never optimized — computed under no_grad):
    Σ_t ‖c_route − ĉ_route‖² / Σ_t ‖c_route − c̄_route‖²,
ĉ_route = Σ_slots ĉ_te, c̄_route = batch mean of c_route.

L_aux (dead-latent auxiliary, contract §Aux loss):
    Σ_te ‖r_te − r̂_te‖² / Σ_te ‖r_te‖², residuals r detached by the caller;
batches with Σ‖r‖² < 1e-12 contribute 0. The α = 1/32 coefficient is applied
by the caller.
"""

from __future__ import annotations

import torch

_DEN_EPS = 1e-12


def local_nmse(
    c_rows: torch.Tensor, chat_rows: torch.Tensor, mu_rows: torch.Tensor
) -> torch.Tensor:
    """Spec §8 primary loss over flattened (t, e) rows.

    c_rows, chat_rows, mu_rows: [M, d] (any float dtype; upcast to fp32).
    Differentiable through chat_rows. Returns fp32 scalar.
    """
    c = c_rows.float()
    num = (c - chat_rows.float()).pow(2).sum()
    den = (c - mu_rows.float()).pow(2).sum().clamp_min(_DEN_EPS)
    return num / den


@torch.no_grad()
def aggregate_nmse(c_route: torch.Tensor, chat_route: torch.Tensor) -> torch.Tensor:
    """Spec §8 aggregate diagnostic. c_route, chat_route: [B, d]. no_grad —
    reported as a metric, never optimized (spec §8, §19)."""
    c = c_route.float()
    num = (c - chat_route.float()).pow(2).sum()
    cbar = c.mean(dim=0, keepdim=True)
    den = (c - cbar).pow(2).sum().clamp_min(_DEN_EPS)
    return num / den


def aux_loss(r_rows: torch.Tensor, rhat_rows: torch.Tensor) -> torch.Tensor:
    """Residual-energy-normalized dead-latent loss over flattened (t, e) rows.

    r_rows must already be detached by the caller (contract). Returns fp32
    scalar; 0 when the residual energy is below the guard."""
    r = r_rows.float()
    den = r.pow(2).sum()
    if float(den.item()) < _DEN_EPS:
        return torch.zeros((), dtype=torch.float32, device=r.device)
    num = (r - rhat_rows.float()).pow(2).sum()
    return num / den
