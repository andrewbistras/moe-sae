"""The six frozen comparison conditions (spec §12) behind one training API.

`build_condition(condition, cfg, mu_e, mu_route, seed)` returns an
`SAECondition` whose `training_step(batch, step)` performs forward + selection
+ losses (never backward), and whose `encode_thresholded` implements frozen-θ*
inference (spec §6) for evaluation.

Condition → dictionary + selection:
  A  router-tied banks, one global BatchTopK over all visible candidates.
  B  compute-matched global SAE on c_route (F = 6f), global BatchTopK.
  C  = A with bank ids permuted per example by a torch-seed-independent
     numpy PCG64 stream keyed on CONDITION_C_SALT ^ example_id (contract).
  D  router-tied banks, independent per-bank budgets ∝ this batch's routed
     loads (largest-remainder rounding to exactly k̄B total).
  E  router-tied banks, exactly k̄ per token across its 6 visible banks.
  F  capacity-matched global SAE on c_route (F = 64f), global BatchTopK.

θ_b logging conventions (contract): A/B/C/F — k̄B-th largest score;
D — smallest selected score across the union; E — mean of per-token k-th
largest thresholds (log-only convention).

Decisions (recorded):
- The condition owns its ThetaEMA and DeadTracker (attributes created here);
  the training loop reads/updates them via the StepOutput bookkeeping fields.
- For B/F, `bank_route_counts` still reports host-expert routing (for
  bank_loads.npy / §16.3 load floor), while the deadness clock
  (`dict_route_counts`) advances by B per batch on the single dictionary bank.
- For C, the deadness clock uses *assigned* (permuted) bank ids — deadness is
  relative to the dictionary bank actually receiving the tokens.
- Aux candidates are "dead now" per the tracker state entering the step; a
  latent both main-selected and aux-selected in the same batch is possible and
  permitted (contract does not exclude it); aux selection never resets
  deadness.
- l0_visual / l0_text are 0.0 when the batch contains no token of that type.
- `load_condition_from_run(run_dir, device)` is the run-directory loader used
  by the eval/intervention scripts: it prefers final/checkpoint.safetensors,
  falls back to the highest-step checkpoints/{step}.safetensors, and resolves
  θ* from final/theta_star.json (canonical) before the checkpoint's stored EMA
  value. Additive helper — training/inference logic is untouched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from moe_sae.model_loading import HIDDEN_SIZE, NUM_ROUTED_EXPERTS
from moe_sae.sae import TrainBatch
from moe_sae.sae.banks import GlobalSAE, RouterTiedBanks
from moe_sae.sae.batchtopk import (
    ThetaEMA,
    global_batch_topk,
    largest_remainder_budgets,
    per_bank_topk,
    per_token_topk,
    thresholded_inference,
)
from moe_sae.sae.dead import DeadTracker
from moe_sae.sae.losses import aggregate_nmse, aux_loss, local_nmse
from moe_sae.train.config import (
    AUX_ALPHA,
    AUX_TOKEN_BUDGET,
    CONDITION_C_SALT,
    CONDITIONS,
    DEAD_WINDOW_BANK_TOKENS,
    THETA_EMA_DECAY,
    global_width,
)

GLOBAL_CONDITIONS = ("B", "F")
ROUTED_CONDITIONS = ("A", "C", "D", "E")

_NEG_INF = float("-inf")


@dataclass
class ThresholdedEncoding:
    """Sparse frozen-θ* inference result for N tokens (eval contract).

    S = 6 slots for router-tied conditions, S = 1 for global (B/F).
    `act_bank` is the bank the latent lives in — equals experts[token, slot]
    for A/D/E, the *permuted* assignment for C, and 0 for B/F.
    """

    act_token: torch.Tensor  # [M] int64 — token row index within the input
    act_slot: torch.Tensor  # [M] uint8 — slot 0..5; 0 for global conditions
    act_bank: torch.Tensor  # [M] int64 — dictionary bank of the latent
    act_latent: torch.Tensor  # [M] int32 — latent index within the bank
    act_value: torch.Tensor  # [M] float32 — z (post θ* threshold)
    l0: torch.Tensor  # [N] int32 — active latents per token
    recon_slots: torch.Tensor  # [N, S, d] float32 — ĉ = μ + Dz per slot
    mu_slots: torch.Tensor  # [N, S, d] float32 — μ_e per slot
    preacts: torch.Tensor | None = None  # [N, S, f] float32, on request


@dataclass
class StepOutput:
    """Everything the loop needs to backward, log, and update trackers."""

    loss: torch.Tensor  # L_local + α·L_aux (fp32 scalar, graph attached)
    l_local: float
    l_aggregate: float
    l_aux: float
    theta_b: float
    l0_mean: float
    l0_visual: float
    l0_text: float
    n_dead_now: int
    k_selected: int  # total main-TopK selections (== k̄B, unit-tested)
    bank_route_counts: torch.Tensor  # [64] int64 — host-expert routing (loads)
    bank_fire_counts: torch.Tensor  # [n_dict_banks] int64 — fires per bank
    dict_route_counts: torch.Tensor  # [n_dict_banks] int64 — deadness clock
    fired_mask: torch.Tensor  # [n_dict_banks, f_dict] bool — for DeadTracker

    def scalars(self) -> dict:
        """Loggable scalar fields for metrics.jsonl."""
        return {
            "loss": float(self.loss.item()),
            "l_local": self.l_local,
            "l_aggregate": self.l_aggregate,
            "l_aux": self.l_aux,
            "theta_b": self.theta_b,
            "l0_mean": self.l0_mean,
            "l0_visual": self.l0_visual,
            "l0_text": self.l0_text,
            "n_dead_now": self.n_dead_now,
            "k_selected": self.k_selected,
        }


class SAECondition(nn.Module):
    """One frozen comparison condition (spec §12) with the shared engine."""

    def __init__(
        self,
        condition: str,
        f: int,
        k_bar: int,
        d: int = HIDDEN_SIZE,
        n_banks: int = NUM_ROUTED_EXPERTS,
        compute_dtype: torch.dtype = torch.bfloat16,
        aux_alpha: float = AUX_ALPHA,
        aux_token_budget: int = AUX_TOKEN_BUDGET,
        dead_window: int = DEAD_WINDOW_BANK_TOKENS,
        theta_decay: float = THETA_EMA_DECAY,
        seed: int = 0,
    ):
        super().__init__()
        assert condition in CONDITIONS, condition
        self.condition = condition
        self.f = f
        self.k_bar = k_bar
        self.d = d
        self.n_banks = n_banks
        self.compute_dtype = compute_dtype
        self.aux_alpha = aux_alpha
        self.aux_token_budget = aux_token_budget
        self.seed = seed
        self.is_global = condition in GLOBAL_CONDITIONS

        if self.is_global:
            width = global_width(condition, f)  # 6f for B, 64f for F
            self.sae = GlobalSAE(width, d, dtype=compute_dtype, seed=seed)
            self.n_dict_banks, self.f_dict = 1, width
        else:
            self.sae = RouterTiedBanks(n_banks, f, d, dtype=compute_dtype, seed=seed)
            self.n_dict_banks, self.f_dict = n_banks, f

        # Owned by the condition; the loop updates them after each step.
        self.theta_ema = ThetaEMA(theta_decay)
        self.dead_tracker = DeadTracker(self.n_dict_banks, self.f_dict, dead_window)
        self._perm_cache: dict[int, torch.Tensor] = {}  # condition C only

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def set_means(self, mu_e: torch.Tensor, mu_route: torch.Tensor) -> None:
        """Install the frozen statistics-pass means (spec §5 centering)."""
        if self.is_global:
            self.sae.set_mu(mu_route)
        else:
            self.sae.set_mu(mu_e)

    def optim_groups(self) -> list[dict]:
        """Parameter groups for SpecAdam, tagging the decoder column axis."""
        return [
            {"params": [self.sae.W_enc, self.sae.b_enc], "decoder_axis": None},
            {"params": [self.sae.D], "decoder_axis": type(self.sae).DECODER_AXIS},
        ]

    def decoder_columns(self) -> torch.Tensor:
        """Decoder parameter view: [E, d, f] (routed) or [d, F] (global);
        columns lie along the d axis."""
        return self.sae.D

    @property
    def device(self) -> torch.device:
        return self.sae.W_enc.device

    # ------------------------------------------------------------------ #
    # Condition C bank permutation                                       #
    # ------------------------------------------------------------------ #

    def _permuted_bank_ids(
        self, experts: torch.Tensor, example_id: torch.Tensor
    ) -> torch.Tensor:
        """σ_ex(experts): per-example permutation of range(n_banks) from
        numpy PCG64(CONDITION_C_SALT ^ example_id) — independent of torch
        seeds by construction (contract; spec §12.C)."""
        uniq, inverse = torch.unique(example_id.cpu().to(torch.int64), return_inverse=True)
        table = torch.empty(uniq.numel(), self.n_banks, dtype=torch.int64)
        for i, eid in enumerate(uniq.tolist()):
            key = int(eid)
            perm = self._perm_cache.get(key)
            if perm is None:
                # & 0xFFFFFFFF guards hypothetical negative int32 ids (PCG64
                # requires a nonnegative seed); identity for nonnegative ids.
                rng = np.random.Generator(
                    np.random.PCG64(CONDITION_C_SALT ^ (key & 0xFFFFFFFF))
                )
                perm = torch.from_numpy(rng.permutation(self.n_banks)).to(torch.int64)
                if len(self._perm_cache) >= 65536:  # bound memory over 50M tokens
                    self._perm_cache.clear()
                self._perm_cache[key] = perm
            table[i] = perm
        # bank_ids[t, s] = σ_{ex(t)}(experts[t, s])
        rows = table[inverse].to(experts.device)  # [B, n_banks]
        return torch.gather(rows, 1, experts.to(torch.int64))

    def _bank_ids(self, experts: torch.Tensor, example_id: torch.Tensor) -> torch.Tensor:
        experts = experts.to(torch.int64)
        if self.condition == "C":
            return self._permuted_bank_ids(experts, example_id)
        return experts

    # ------------------------------------------------------------------ #
    # Selection engines                                                  #
    # ------------------------------------------------------------------ #

    def _select(
        self, scores: torch.Tensor, bank_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, float]:
        """Main-TopK selection mask + θ_b. scores: [B, S, f] fp32 (routed) or
        [B, F] fp32 (global; treated as S=1)."""
        if scores.ndim == 2:
            scores = scores.unsqueeze(1)  # [B, 1, F]
        B, S, f = scores.shape
        k_total = self.k_bar * B
        if self.condition in ("A", "B", "C", "F"):
            mask, theta_b = global_batch_topk(scores.reshape(-1), k_total)
            return mask.view(B, S, f), theta_b
        if self.condition == "D":
            loads = torch.bincount(bank_ids.reshape(-1), minlength=self.n_banks)
            budgets = largest_remainder_budgets(loads.float(), k_total)
            bank_of_cand = (
                bank_ids.unsqueeze(-1).expand(B, S, f).reshape(-1)
            )
            mask, theta_b = per_bank_topk(scores.reshape(-1), bank_of_cand, budgets)
            return mask.view(B, S, f), theta_b
        if self.condition == "E":
            mask2, per_tok_theta = per_token_topk(scores.reshape(B, S * f), self.k_bar)
            # Log-only convention: batch θ_b = mean of per-token thresholds.
            return mask2.view(B, S, f), float(per_tok_theta.mean().item())
        raise ValueError(self.condition)

    # ------------------------------------------------------------------ #
    # Training step                                                      #
    # ------------------------------------------------------------------ #

    def training_step(self, batch: TrainBatch, step: int) -> StepOutput:
        """Forward + selection + losses. Does NOT call backward."""
        dev = self.device
        experts = batch.experts.to(dev).to(torch.int64)
        host_route_counts = torch.bincount(
            experts.reshape(-1).cpu(), minlength=NUM_ROUTED_EXPERTS
        )
        dead = self.dead_tracker.dead_mask().to(dev)  # [n_dict_banks, f_dict]
        n_dead_now = int(dead.sum().item())

        if self.is_global:
            # Frozen input for B/F: c_route summed in fp32, then compute dtype.
            x = batch.c.to(dev).float().sum(dim=1).to(self.compute_dtype)  # [B, d]
            B = x.shape[0]
            bank_ids = None
            preacts = self.sae.encode(x)  # [B, F]
            scores = preacts.detach().float()
            mask, theta_b = self._select(scores, None)
            mask = mask.view(B, self.f_dict)
            z = preacts * mask.to(preacts.dtype)
            chat = self.sae.decode(z, add_mu=True)  # [B, d]

            x32 = x.float()
            mu_rows = self.sae.mu.unsqueeze(0).expand(B, self.d)
            # For B/F the "local" loss is the direct c_route reconstruction
            # NMSE with the μ_route-centered denominator (spec §12: one shared
            # SAE reconstructs c_route; the §8 (t,e) sum has a single term).
            l_local = local_nmse(x32, chat, mu_rows)
            l_agg = aggregate_nmse(x32, chat.float())

            dead_vis = dead.view(1, self.f_dict).expand(B, self.f_dict)
            l_aux_t = self._aux_global(scores, preacts, dead_vis, x32, chat)

            z_pos = (z.detach().float() > 0)
            fired_mask = torch.zeros(1, self.f_dict, dtype=torch.bool)
            fired_mask[0] = mask.any(dim=0).cpu()
            bank_fire_counts = mask.sum().to(torch.int64).cpu().reshape(1)
            dict_route_counts = torch.tensor([B], dtype=torch.int64)
            l0_tok = z_pos.sum(dim=1).float().cpu()
            k_selected = int(mask.sum().item())
        else:
            c = batch.c.to(dev).to(self.compute_dtype)  # [B, 6, d]
            B, S, _ = c.shape
            bank_ids = self._bank_ids(experts, batch.example_id)
            preacts = self.sae.encode(c, bank_ids)  # [B, 6, f]
            scores = preacts.detach().float()
            mask, theta_b = self._select(scores, bank_ids)
            z = preacts * mask.to(preacts.dtype)
            chat = self.sae.decode(z, bank_ids, add_mu=True)  # [B, 6, d]

            c32 = c.float()
            mu_rows = self.sae.mu[bank_ids]  # [B, 6, d] fp32
            l_local = local_nmse(
                c32.reshape(-1, self.d), chat.reshape(-1, self.d), mu_rows.reshape(-1, self.d)
            )
            l_agg = aggregate_nmse(c32.sum(dim=1), chat.float().sum(dim=1))

            dead_vis = dead[bank_ids]  # [B, 6, f]
            l_aux_t = self._aux_routed(scores, preacts, dead_vis, bank_ids, c32, chat)

            sel_b, sel_s, sel_j = mask.nonzero(as_tuple=True)
            sel_e = bank_ids[sel_b, sel_s].cpu()
            fired_mask = torch.zeros(self.n_dict_banks, self.f_dict, dtype=torch.bool)
            fired_mask[sel_e, sel_j.cpu()] = True
            bank_fire_counts = torch.bincount(sel_e, minlength=self.n_dict_banks)
            dict_route_counts = torch.bincount(
                bank_ids.reshape(-1).cpu(), minlength=self.n_dict_banks
            )
            l0_tok = (z.detach().float() > 0).sum(dim=(1, 2)).float().cpu()
            k_selected = int(mask.sum().item())

        loss = l_local + self.aux_alpha * l_aux_t

        token_type = batch.token_type.cpu()
        vis = token_type == 1
        txt = token_type == 0
        l0_visual = float(l0_tok[vis].mean().item()) if bool(vis.any()) else 0.0
        l0_text = float(l0_tok[txt].mean().item()) if bool(txt.any()) else 0.0

        return StepOutput(
            loss=loss,
            l_local=float(l_local.item()),
            l_aggregate=float(l_agg.item()),
            l_aux=float(l_aux_t.item()),
            theta_b=theta_b,
            l0_mean=float(l0_tok.mean().item()),
            l0_visual=l0_visual,
            l0_text=l0_text,
            n_dead_now=n_dead_now,
            k_selected=k_selected,
            bank_route_counts=host_route_counts,
            bank_fire_counts=bank_fire_counts,
            dict_route_counts=dict_route_counts,
            fired_mask=fired_mask,
        )

    # ------------------------------------------------------------------ #
    # Aux loss (spec §8 revised; contract §Aux loss)                     #
    # ------------------------------------------------------------------ #

    def _aux_select(self, scores2d: torch.Tensor, dead2d: torch.Tensor) -> torch.Tensor:
        """Per-token top min(512, #dead visible) dead candidates by preact —
        token-level competition. scores2d/dead2d: [B, n_cand]."""
        masked = scores2d.masked_fill(~dead2d, _NEG_INF)
        k_aux = min(self.aux_token_budget, masked.shape[1])
        vals, idx = torch.topk(masked, k_aux, dim=1)
        aux_mask = torch.zeros_like(dead2d)
        aux_mask.scatter_(1, idx, vals > _NEG_INF)
        return aux_mask

    def _aux_routed(self, scores, preacts, dead_vis, bank_ids, c32, chat) -> torch.Tensor:
        if not bool(dead_vis.any()):
            return torch.zeros((), dtype=torch.float32, device=preacts.device)
        B, S, f = scores.shape
        aux_mask = self._aux_select(
            scores.reshape(B, S * f), dead_vis.reshape(B, S * f)
        ).view(B, S, f)
        z_aux = preacts * aux_mask.to(preacts.dtype)
        # Detached per-expert residuals reconstructed through own banks'
        # decoders, no μ (contract) — preserves no-cross-bank-cancellation.
        r = (c32 - chat.float()).detach()
        rhat = self.sae.decode(z_aux, bank_ids, add_mu=False)
        return aux_loss(r.reshape(-1, self.d), rhat.reshape(-1, self.d))

    def _aux_global(self, scores, preacts, dead_vis, x32, chat) -> torch.Tensor:
        if not bool(dead_vis.any()):
            return torch.zeros((), dtype=torch.float32, device=preacts.device)
        aux_mask = self._aux_select(scores, dead_vis)
        z_aux = preacts * aux_mask.to(preacts.dtype)
        r = (x32 - chat.float()).detach()
        rhat = self.sae.decode(z_aux, add_mu=False)
        return aux_loss(r, rhat)

    # ------------------------------------------------------------------ #
    # Frozen-θ* inference (spec §6)                                      #
    # ------------------------------------------------------------------ #

    @property
    def num_banks(self) -> int:
        """Dictionary bank count: 64 router-tied, 1 global (eval contract)."""
        return 1 if self.is_global else self.n_banks

    def decoder_cols(self, bank: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Unit-norm decoder columns D_{e,j} for (bank, latent) pairs → [M, d].
        Global conditions ignore `bank` (single dictionary)."""
        bank = bank.to(self.device, torch.int64)
        latent = latent.to(self.device, torch.int64)
        if self.is_global:
            return self.sae.D[:, latent].T.float()  # D is [d, F]
        return self.sae.D[bank, :, latent].float()  # D is [E, d, f] → [M, d]

    @torch.no_grad()
    def encode_thresholded(
        self,
        c: torch.Tensor,
        experts: torch.Tensor,
        example_id: torch.Tensor,
        theta_star: float,
        return_preacts: bool = False,
    ) -> ThresholdedEncoding:
        """z = a·1[a > θ*] over N tokens (spec §6 frozen-θ* inference).

        Returns the sparse activations plus per-slot reconstructions and μ
        (the eval contract). `example_id` is required: condition C's bank
        assignment is a per-example permutation even at inference. For B/F,
        operates on c_route with a single slot (S=1)."""
        dev = self.device
        if self.is_global:
            x = c.to(dev).float().sum(dim=1).to(self.compute_dtype)
            preacts = self.sae.encode(x)  # [N, F]
            z = thresholded_inference(preacts, theta_star)
            tok, lat = (z > 0).nonzero(as_tuple=True)
            slot = torch.zeros_like(tok)
            bank = torch.zeros_like(tok)
            values = z[tok, lat]
            l0 = (z > 0).sum(dim=1)
            recon = self.sae.decode(z, add_mu=True).float().unsqueeze(1)  # [N,1,d]
            mu_slots = self.sae.mu.float().expand(c.shape[0], 1, self.d)
        else:
            xc = c.to(dev).to(self.compute_dtype)
            bank_ids = self._bank_ids(experts.to(dev), example_id)
            preacts = self.sae.encode(xc, bank_ids)  # [N, 6, f]
            z = thresholded_inference(preacts, theta_star)
            tok, slot, lat = (z > 0).nonzero(as_tuple=True)
            bank = bank_ids[tok, slot]
            values = z[tok, slot, lat]
            l0 = (z > 0).sum(dim=(1, 2))
            recon = self.sae.decode(z, bank_ids, add_mu=True).float()  # [N,6,d]
            mu_slots = self.sae.mu[bank_ids].float()  # [N,6,d]
        return ThresholdedEncoding(
            act_token=tok.to(torch.int64).cpu(),
            act_slot=slot.to(torch.uint8).cpu(),
            act_bank=bank.to(torch.int64).cpu(),
            act_latent=lat.to(torch.int32).cpu(),
            act_value=values.float().cpu(),
            l0=l0.to(torch.int32).cpu(),
            recon_slots=recon.cpu(),
            mu_slots=mu_slots.cpu(),
            preacts=preacts.float().cpu() if return_preacts else None,
        )

    # ------------------------------------------------------------------ #
    # Checkpointing                                                      #
    # ------------------------------------------------------------------ #

    def _init_args(self) -> dict:
        return {
            "condition": self.condition,
            "f": self.f,
            "k_bar": self.k_bar,
            "d": self.d,
            "n_banks": self.n_banks,
            "compute_dtype": str(self.compute_dtype).removeprefix("torch."),
            "aux_alpha": self.aux_alpha,
            "aux_token_budget": self.aux_token_budget,
            "dead_window": self.dead_tracker.window,
            "theta_decay": self.theta_ema.decay,
            "seed": self.seed,
        }

    def save_checkpoint(
        self,
        path: str | Path,
        theta_ema: ThetaEMA,
        dead_tracker: DeadTracker,
        step: int,
    ) -> None:
        theta_value = theta_ema.value
        dead_state = dead_tracker.state_dict()
        tensors = {
            "W_enc": self.sae.W_enc.detach().cpu().contiguous(),
            "b_enc": self.sae.b_enc.detach().cpu().contiguous(),
            "D": self.sae.D.detach().cpu().contiguous(),
            "mu": self.sae.mu.detach().cpu().contiguous(),
            "theta_value": torch.tensor(
                float("nan") if theta_value is None else theta_value,
                dtype=torch.float64,
            ),
            "theta_decay": torch.tensor(theta_ema.decay, dtype=torch.float64),
            "dead_tokens_seen": dead_state["tokens_seen"],
            "dead_last_fire": dead_state["last_fire"],
            "dead_window": dead_state["window"],
            "step": torch.tensor(step, dtype=torch.int64),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path), metadata={"config": json.dumps(self._init_args())})

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> tuple["SAECondition", float | None, dict]:
        """Rebuild the condition from a checkpoint. Returns
        (condition, theta_star, extras) with extras = {'step', 'theta_ema',
        'dead_tracker'}; the restored trackers are also installed on the
        returned condition."""
        with safe_open(str(path), framework="pt") as fh:
            args = json.loads(fh.metadata()["config"])
            tensors = {k: fh.get_tensor(k) for k in fh.keys()}
        args["compute_dtype"] = getattr(torch, args["compute_dtype"])
        cond = cls(**args)
        with torch.no_grad():
            cond.sae.W_enc.copy_(tensors["W_enc"].to(cond.sae.W_enc.dtype))
            cond.sae.b_enc.copy_(tensors["b_enc"].to(cond.sae.b_enc.dtype))
            cond.sae.D.copy_(tensors["D"].to(cond.sae.D.dtype))
            cond.sae.mu.copy_(tensors["mu"].to(torch.float32))
        theta_value = float(tensors["theta_value"].item())
        theta_star = None if theta_value != theta_value else theta_value  # nan → None
        cond.theta_ema.load_state_dict(
            {"decay": float(tensors["theta_decay"].item()), "value": theta_star}
        )
        cond.dead_tracker.load_state_dict(
            {
                "tokens_seen": tensors["dead_tokens_seen"],
                "last_fire": tensors["dead_last_fire"],
                "window": tensors["dead_window"],
            }
        )
        extras = {
            "step": int(tensors["step"].item()),
            "theta_ema": cond.theta_ema,
            "dead_tracker": cond.dead_tracker,
        }
        return cond, theta_star, extras


def load_condition_from_run(
    run_dir: str | Path, device: str | torch.device = "cpu"
) -> tuple[SAECondition, float, dict]:
    """Load a trained condition from a run directory (contract:
    docs/INTERFACES.md §"Run artifact layout").

    Checkpoint resolution: prefer run_dir/final/checkpoint.safetensors, else
    the highest-step run_dir/checkpoints/{step}.safetensors (numeric order).
    θ* resolution: canonical source is run_dir/final/theta_star.json
    ({'theta_star': x} or a bare float), falling back to the checkpoint's
    stored θ EMA value; errors if neither exists.

    Returns (condition-on-device, theta_star, extras) with extras as produced
    by `SAECondition.load_checkpoint`.
    """
    run_dir = Path(run_dir)
    ckpt = run_dir / "final" / "checkpoint.safetensors"
    if not ckpt.exists():
        candidates: list[tuple[int, Path]] = []
        for p in (run_dir / "checkpoints").glob("*.safetensors"):
            try:
                candidates.append((int(p.stem), p))
            except ValueError:
                continue  # not a {step}.safetensors file
        if not candidates:
            raise FileNotFoundError(
                f"no checkpoint under {run_dir}: expected "
                "final/checkpoint.safetensors or checkpoints/{step}.safetensors"
            )
        ckpt = max(candidates)[1]

    condition, ckpt_theta, extras = SAECondition.load_checkpoint(ckpt)
    condition = condition.to(torch.device(device))

    theta_path = run_dir / "final" / "theta_star.json"
    if theta_path.exists():
        raw = json.loads(theta_path.read_text())
        theta_star = float(raw["theta_star"]) if isinstance(raw, dict) else float(raw)
    elif ckpt_theta is not None:
        theta_star = float(ckpt_theta)
    else:
        raise ValueError(
            f"cannot resolve theta_star for {run_dir}: no final/theta_star.json "
            f"and checkpoint {ckpt} carries no θ EMA value"
        )
    return condition, theta_star, extras


def build_condition(
    condition: str,
    cfg,
    mu_e: torch.Tensor,
    mu_route: torch.Tensor,
    seed: int,
) -> SAECondition:
    """Construct a condition from a RunConfig + frozen statistics-pass means.

    mu_e: [64, d] fp32; mu_route: [d] fp32. d is inferred from the means so
    tiny-dimension tests exercise the identical code path."""
    n_banks, d = mu_e.shape
    assert mu_route.shape == (d,), (mu_e.shape, mu_route.shape)
    cond = SAECondition(
        condition=condition,
        f=cfg.f,
        k_bar=cfg.k_bar,
        d=d,
        n_banks=n_banks,
        compute_dtype=getattr(torch, cfg.compute_dtype),
        aux_alpha=cfg.aux_alpha,
        aux_token_budget=cfg.aux_token_budget,
        dead_window=cfg.dead_window,
        theta_decay=cfg.theta_ema_decay,
        seed=seed,
    )
    cond.set_means(mu_e, mu_route)
    return cond
