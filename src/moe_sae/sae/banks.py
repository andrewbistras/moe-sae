"""SAE parameter containers (spec §5).

`RouterTiedBanks`: one local SAE bank per routed host expert (conditions
A/C/D/E). For expert e: W_enc_e [f, d], b_enc_e [f], decoder D_e [d, f] with
every decoder column D[e, :, j] constrained to unit ℓ2 norm (spec §5; the
optimization mechanics of the constraint live in `moe_sae.sae.optim`).

`GlobalSAE`: one shared dictionary reconstructing c_route (conditions B/F),
same API with the slot axis collapsed.

Per-expert means μ_e are computed in a prior statistics pass and *frozen*
(spec §5 "Centering") — they are registered buffers, never Parameters.

The grouped encode/decode runs one GEMM per bank present in the batch (never a
[B, S, d, f] gather — memory). It is autograd-correct: unit-tested against a
naive per-row loop for exact equality and matching gradients.

Init (decision, recorded in docs/INTERFACES.md conventions): decoder columns
drawn N(0,1) and normalized to unit norm; W_enc initialized to the transpose of
the decoder (per bank); b_enc zeros.
"""

from __future__ import annotations

import torch
from torch import nn


def _init_decoder(shape_ed_f: tuple[int, ...], seed: int) -> torch.Tensor:
    """Unit-norm decoder columns, N(0,1)-normalized. Columns lie along the
    second-to-last axis (the d axis): D[..., :, j] is column j."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dec = torch.randn(*shape_ed_f, generator=g, dtype=torch.float32)
    dec = dec / dec.norm(dim=-2, keepdim=True).clamp_min(1e-12)
    return dec


class RouterTiedBanks(nn.Module):
    """64 expert-tied SAE banks with shared grouped-GEMM encode/decode."""

    # Axis of `D` along which each decoder column lies (for SpecAdam).
    DECODER_AXIS = 1

    def __init__(
        self,
        n_banks: int,
        f: int,
        d: int,
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 0,
    ):
        super().__init__()
        self.n_banks = n_banks
        self.f = f
        self.d = d
        dec = _init_decoder((n_banks, d, f), seed)  # [E, d, f]
        self.D = nn.Parameter(dec.to(dtype))
        # W_enc_e = D_e^T (per-bank transpose of the decoder), b_enc = 0.
        self.W_enc = nn.Parameter(dec.transpose(1, 2).contiguous().to(dtype))  # [E, f, d]
        self.b_enc = nn.Parameter(torch.zeros(n_banks, f, dtype=dtype))
        # Frozen per-expert means (spec §5) — buffer, never a Parameter.
        self.register_buffer("mu", torch.zeros(n_banks, d, dtype=torch.float32))

    @torch.no_grad()
    def set_mu(self, mu_e: torch.Tensor) -> None:
        assert mu_e.shape == (self.n_banks, self.d), mu_e.shape
        self.mu.copy_(mu_e.to(torch.float32))

    def encode(self, c: torch.Tensor, bank_ids: torch.Tensor) -> torch.Tensor:
        """a_te = ReLU(W_enc_e (c_te − μ_e) + b_enc_e).

        c: [B, S, d] (any float dtype); bank_ids: [B, S] int64.
        Returns [B, S, f] in parameter dtype. One GEMM per bank present.
        """
        B, S, d = c.shape
        flat = c.reshape(B * S, d).to(self.W_enc.dtype)
        ids = bank_ids.reshape(B * S)
        out = torch.zeros(B * S, self.f, dtype=self.W_enc.dtype, device=flat.device)
        for e in torch.unique(ids).tolist():
            rows = (ids == e).nonzero(as_tuple=True)[0]
            tilde = flat[rows] - self.mu[e].to(self.W_enc.dtype)
            pre = tilde @ self.W_enc[e].T + self.b_enc[e]
            out = out.index_put((rows,), torch.relu(pre))
        return out.reshape(B, S, self.f)

    def decode(self, z: torch.Tensor, bank_ids: torch.Tensor, add_mu: bool) -> torch.Tensor:
        """ĉ_te = [μ_e +] D_e z_te.  z: [B, S, f] → [B, S, d]."""
        B, S, f = z.shape
        flat = z.reshape(B * S, f).to(self.D.dtype)
        ids = bank_ids.reshape(B * S)
        out = torch.zeros(B * S, self.d, dtype=self.D.dtype, device=flat.device)
        for e in torch.unique(ids).tolist():
            rows = (ids == e).nonzero(as_tuple=True)[0]
            rec = flat[rows] @ self.D[e].T
            if add_mu:
                rec = rec + self.mu[e].to(self.D.dtype)
            out = out.index_put((rows,), rec)
        return out.reshape(B, S, self.d)


class GlobalSAE(nn.Module):
    """One shared dictionary on c_route (conditions B/F). S=1 semantics."""

    DECODER_AXIS = 0  # D is [d, F]; columns lie along axis 0

    def __init__(
        self,
        F: int,
        d: int,
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 0,
    ):
        super().__init__()
        self.F = F
        self.d = d
        dec = _init_decoder((d, F), seed)  # [d, F]
        self.D = nn.Parameter(dec.to(dtype))
        self.W_enc = nn.Parameter(dec.T.contiguous().to(dtype))  # [F, d]
        self.b_enc = nn.Parameter(torch.zeros(F, dtype=dtype))
        self.register_buffer("mu", torch.zeros(d, dtype=torch.float32))

    @torch.no_grad()
    def set_mu(self, mu_route: torch.Tensor) -> None:
        assert mu_route.shape == (self.d,), mu_route.shape
        self.mu.copy_(mu_route.to(torch.float32))

    def encode(self, c_route: torch.Tensor) -> torch.Tensor:
        """a_t = ReLU(W_enc (c_route − μ) + b_enc).  [B, d] → [B, F]."""
        tilde = c_route.to(self.W_enc.dtype) - self.mu.to(self.W_enc.dtype)
        return torch.relu(tilde @ self.W_enc.T + self.b_enc)

    def decode(self, z: torch.Tensor, add_mu: bool) -> torch.Tensor:
        """ĉ_t = [μ +] D z_t.  [B, F] → [B, d]."""
        rec = z.to(self.D.dtype) @ self.D.T
        if add_mu:
            rec = rec + self.mu.to(self.D.dtype)
        return rec
