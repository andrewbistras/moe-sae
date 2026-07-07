"""JAX training run wrapping a torch SAECondition (spec §10–§12).

The torch condition (CPU) owns everything cold: initialization (bit-identical
to a torch run at the same seed), the frozen μ, condition-C permutations,
θ-EMA + DeadTracker, and checkpoint serialization (so
`load_condition_from_run` and the whole eval/analysis stack read JAX-trained
runs unchanged). JAX owns the hot per-step math via `moe_sae.tpu.sae_jax`.

Artifacts and metrics.jsonl lines are formatted exactly like
moe_sae.train.loop.train (same keys), so scripts/tpu/compare_runs.py can diff
backends line by line. The batch stream is the torch BatchComposer — identical
batch sequences across backends by construction.

Condition D's per-bank selection runs host-side through the torch engine
(decision recorded in sae_jax.py).
"""

from __future__ import annotations

import json
import logging
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from moe_sae.sae import TrainBatch
from moe_sae.sae.batchtopk import largest_remainder_budgets, per_bank_topk
from moe_sae.sae.conditions import SAECondition
from moe_sae.tpu import sae_jax as sj

logger = logging.getLogger(__name__)

_JNP_DTYPE = {torch.float32: jnp.float32, torch.bfloat16: jnp.bfloat16}


# --------------------------------------------------------------------------- #
# Jitted loss/grad kernels                                                     #
# --------------------------------------------------------------------------- #


def _routed_loss(
    params, c, bank_flat, order, dest, mu, dead_ef, ext_mask,
    *, kind, k_bar, cap, n_banks, f, d, aux_alpha, aux_budget,
):
    B, S, _ = c.shape
    n = B * S
    preacts = sj.encode_routed(
        params["W_enc"], params["b_enc"], mu, c, bank_flat, order, dest,
        cap=cap, n_banks=n_banks, out_f=f,
    )
    scores = jax.lax.stop_gradient(preacts).astype(jnp.float32)
    if kind == "global":
        mask_flat, theta_b = sj.global_topk_mask(scores.reshape(-1), k_bar * B)
        mask = mask_flat.reshape(B, S, f)
    elif kind == "per_token":
        mask2, per_tok = sj.per_token_topk_mask(scores.reshape(B, S * f), k_bar)
        mask = mask2.reshape(B, S, f)
        theta_b = jnp.mean(per_tok)
    else:  # external (condition D): host-computed mask
        mask = ext_mask
        theta_b = jnp.float32(0.0)  # host supplies the real θ_b

    z = preacts * mask.astype(preacts.dtype)
    decode = partial(
        sj.decode_routed, params["D"], mu,
        bank_flat=bank_flat, order=order, dest=dest, cap=cap, n_banks=n_banks,
    )
    chat = decode(z=z, add_mu=True)

    c32 = c.astype(jnp.float32)
    mu_rows = mu[bank_flat].reshape(B, S, d)
    l_local = sj.local_nmse(
        c32.reshape(n, d), chat.reshape(n, d), mu_rows.reshape(n, d)
    )
    l_agg = jax.lax.stop_gradient(
        sj.aggregate_nmse(c32.sum(axis=1), chat.astype(jnp.float32).sum(axis=1))
    )

    dead_vis = dead_ef[bank_flat].reshape(B, S * f)  # [N=B·S rows of banks]
    aux_mask = sj.aux_select_mask(scores.reshape(B, S * f), dead_vis, aux_budget)
    z_aux = preacts * aux_mask.reshape(B, S, f).astype(preacts.dtype)
    r = jax.lax.stop_gradient(c32 - chat.astype(jnp.float32))
    rhat = decode(z=z_aux, add_mu=False)
    l_aux = jnp.where(
        dead_ef.any(),
        sj.aux_loss(r.reshape(n, d), rhat.reshape(n, d)),
        jnp.float32(0.0),
    )

    loss = l_local + aux_alpha * l_aux
    z_pos = jax.lax.stop_gradient(z).astype(jnp.float32) > 0
    l0_tok = z_pos.sum(axis=(1, 2)).astype(jnp.int32)
    mask_rows = mask.reshape(n, f)
    fired = jnp.zeros((n_banks, f), bool).at[bank_flat].max(mask_rows)
    bank_fire = jnp.zeros(n_banks, jnp.int32).at[bank_flat].add(
        mask_rows.sum(axis=1).astype(jnp.int32)
    )
    aux = {
        "l_local": l_local, "l_aggregate": l_agg, "l_aux": l_aux,
        "theta_b": theta_b, "k_selected": mask.sum().astype(jnp.int32),
        "l0_tok": l0_tok, "fired_mask": fired, "bank_fire_counts": bank_fire,
    }
    return loss, aux


def _global_loss(
    params, x, mu, dead_f, *, kind, k_bar, f_dict, d, aux_alpha, aux_budget,
):
    B = x.shape[0]
    preacts = sj.encode_global(params["W_enc"], params["b_enc"], mu, x)
    scores = jax.lax.stop_gradient(preacts).astype(jnp.float32)
    mask_flat, theta_b = sj.global_topk_mask(scores.reshape(-1), k_bar * B)
    mask = mask_flat.reshape(B, f_dict)

    z = preacts * mask.astype(preacts.dtype)
    chat = sj.decode_global(params["D"], mu, z, add_mu=True)

    x32 = x.astype(jnp.float32)
    mu_rows = jnp.broadcast_to(mu, (B, d))
    l_local = sj.local_nmse(x32, chat, mu_rows)
    l_agg = jax.lax.stop_gradient(sj.aggregate_nmse(x32, chat.astype(jnp.float32)))

    dead_vis = jnp.broadcast_to(dead_f.reshape(1, f_dict), (B, f_dict))
    aux_mask = sj.aux_select_mask(scores, dead_vis, aux_budget)
    z_aux = preacts * aux_mask.astype(preacts.dtype)
    r = jax.lax.stop_gradient(x32 - chat.astype(jnp.float32))
    rhat = sj.decode_global(params["D"], mu, z_aux, add_mu=False)
    l_aux = jnp.where(dead_f.any(), sj.aux_loss(r, rhat), jnp.float32(0.0))

    loss = l_local + aux_alpha * l_aux
    z_pos = jax.lax.stop_gradient(z).astype(jnp.float32) > 0
    aux = {
        "l_local": l_local, "l_aggregate": l_agg, "l_aux": l_aux,
        "theta_b": theta_b, "k_selected": mask.sum().astype(jnp.int32),
        "l0_tok": z_pos.sum(axis=1).astype(jnp.int32),
        "fired_mask": mask.any(axis=0).reshape(1, f_dict),
        "bank_fire_counts": mask.sum().astype(jnp.int32).reshape(1),
    }
    return loss, aux


# --------------------------------------------------------------------------- #
# The run wrapper                                                              #
# --------------------------------------------------------------------------- #


class JaxRun:
    """One condition's training state on a JAX device, wrapping the torch
    condition for everything cold (init, trackers, permutations, checkpoints)."""

    def __init__(self, cond: SAECondition, capacity: int | None = None, device=None):
        self.cond = cond
        self.device = device or jax.devices()[0]
        self.is_global = cond.is_global
        self.dtype = _JNP_DTYPE[cond.compute_dtype]
        self.capacity = capacity  # None → chosen from the first batch's shape
        with jax.default_device(self.device):
            self.params = {
                "W_enc": jnp.asarray(
                    cond.sae.W_enc.detach().float().numpy(), self.dtype
                ),
                "b_enc": jnp.asarray(cond.sae.b_enc.detach().float().numpy(), self.dtype),
                "D": jnp.asarray(cond.sae.D.detach().float().numpy(), self.dtype),
            }
            self.mu = jnp.asarray(cond.sae.mu.numpy(), jnp.float32)
        decoder_axis = type(cond.sae).DECODER_AXIS
        self.opt = sj.SpecAdamJax(self.params, decoder_axis)
        self._grad_fn = None

    # ---- helpers ---------------------------------------------------------- #

    def _capacity_for(self, batch_slots: int) -> int:
        if self.capacity is not None:
            return self.capacity
        # default: 2× mean load, rounded up to a multiple of 128 (contract w/
        # sae_jax grouped encode; overflow is a hard error there).
        mean = batch_slots / self.cond.n_banks
        cap = int(np.ceil(2 * mean / 128.0) * 128)
        self.capacity = max(cap, 128)
        return self.capacity

    def _plan_with_adaptive_capacity(self, bank_ids_np, n_banks):
        """Grouping plan that grows capacity on overflow instead of failing.

        Real router loads are example-correlated (all tokens of an image pile
        onto its experts together), so per-batch maxima far exceed iid
        estimates — observed 909 slots on a bank whose aggregate share
        predicts ~500 ± 22. Growth re-jits the step once per resize (rare;
        capacity only ratchets up) and never drops a token (spec §6)."""
        while True:
            try:
                return sj.grouping_plan(bank_ids_np, n_banks, self.capacity)
            except RuntimeError:
                observed = int(np.bincount(bank_ids_np.reshape(-1), minlength=n_banks).max())
                new_cap = int(np.ceil(observed * 1.25 / 128.0) * 128)
                assert new_cap > self.capacity, (observed, self.capacity)
                logger.warning(
                    "grouped-GEMM capacity %d → %d (max bank load %d); re-jitting",
                    self.capacity, new_cap, observed,
                )
                self.capacity = new_cap
                self._grad_fn = None  # rebuilt with the new static capacity

    def _build_grad_fn(self, kind: str):
        c = self.cond
        if self.is_global:
            fn = partial(
                _global_loss, kind=kind, k_bar=c.k_bar, f_dict=c.f_dict, d=c.d,
                aux_alpha=c.aux_alpha, aux_budget=c.aux_token_budget,
            )
        else:
            fn = partial(
                _routed_loss, kind=kind, k_bar=c.k_bar, cap=self.capacity,
                n_banks=c.n_banks, f=c.f, d=c.d,
                aux_alpha=c.aux_alpha, aux_budget=c.aux_token_budget,
            )
        return jax.jit(jax.value_and_grad(fn, has_aux=True))

    # ---- one training step ------------------------------------------------ #

    def training_step(self, batch: TrainBatch, lr: float) -> dict:
        c = self.cond
        experts = batch.experts.to(torch.int64)
        host_route_counts = torch.bincount(experts.reshape(-1), minlength=64)
        dead_t = c.dead_tracker.dead_mask()  # [n_dict_banks, f_dict] torch bool
        dead = jnp.asarray(dead_t.numpy())

        if self.is_global:
            x = jnp.asarray(
                batch.c.float().sum(dim=1).numpy(), self.dtype
            )
            if self._grad_fn is None:
                self._grad_fn = self._build_grad_fn("global")
            (loss, aux), grads = self._grad_fn(self.params, x, self.mu, dead.reshape(-1))
            dict_route_counts = torch.tensor([x.shape[0]], dtype=torch.int64)
            theta_b = float(aux["theta_b"])
        else:
            bank_ids = c._bank_ids(experts, batch.example_id)  # torch (C-aware)
            B, S = bank_ids.shape
            self._capacity_for(B * S)
            plan = self._plan_with_adaptive_capacity(bank_ids.numpy(), c.n_banks)
            cj = jnp.asarray(batch.c.float().numpy(), self.dtype)
            args = (
                jnp.asarray(plan.bank_flat), jnp.asarray(plan.order),
                jnp.asarray(plan.dest),
            )
            if c.condition == "D":
                # Host-side per-bank selection through the torch engine.
                if self._grad_fn is None:
                    self._grad_fn = self._build_grad_fn("external")
                pre = sj.encode_routed(
                    self.params["W_enc"], self.params["b_enc"], self.mu, cj,
                    *args, cap=self.capacity, n_banks=c.n_banks, out_f=c.f,
                )
                scores_t = torch.from_numpy(np.asarray(pre, np.float32))
                loads = torch.bincount(bank_ids.reshape(-1), minlength=c.n_banks)
                budgets = largest_remainder_budgets(loads.float(), c.k_bar * B)
                bank_of_cand = bank_ids.unsqueeze(-1).expand(B, S, c.f).reshape(-1)
                mask_t, theta_b = per_bank_topk(
                    scores_t.reshape(-1), bank_of_cand, budgets
                )
                ext_mask = jnp.asarray(mask_t.view(B, S, c.f).numpy())
                (loss, aux), grads = self._grad_fn(
                    self.params, cj, *args, self.mu, dead, ext_mask
                )
            else:
                kind = "per_token" if c.condition == "E" else "global"
                if self._grad_fn is None:
                    self._grad_fn = self._build_grad_fn(kind)
                (loss, aux), grads = self._grad_fn(
                    self.params, cj, *args, self.mu, dead, None
                )
                theta_b = float(aux["theta_b"])
            dict_route_counts = torch.bincount(
                bank_ids.reshape(-1), minlength=c.n_dict_banks
            )

        self.params, grad_norm = self.opt.step(self.params, grads, lr)

        l0 = np.asarray(aux["l0_tok"], np.float64)
        tt = batch.token_type.numpy()
        out = {
            "loss": float(loss),
            "l_local": float(aux["l_local"]),
            "l_aggregate": float(aux["l_aggregate"]),
            "l_aux": float(aux["l_aux"]),
            "theta_b": theta_b,
            "l0_mean": float(l0.mean()),
            "l0_visual": float(l0[tt == 1].mean()) if (tt == 1).any() else 0.0,
            "l0_text": float(l0[tt == 0].mean()) if (tt == 0).any() else 0.0,
            "n_dead_now": int(dead_t.sum()),
            "k_selected": int(aux["k_selected"]),
            "grad_norm": grad_norm,
        }
        # Tracker updates through the torch classes (identical semantics).
        c.theta_ema.update(out["theta_b"])
        c.dead_tracker.update(
            dict_route_counts,
            torch.from_numpy(np.asarray(aux["fired_mask"])),
        )
        return out, host_route_counts

    # ---- checkpointing through torch (format identical) ------------------- #

    def sync_params_to_torch(self) -> None:
        with torch.no_grad():
            for name in ("W_enc", "b_enc", "D"):
                t = getattr(self.cond.sae, name)
                t.copy_(torch.from_numpy(np.asarray(self.params[name], np.float32)).to(t.dtype))

    def save_checkpoint(self, path, step: int) -> None:
        self.sync_params_to_torch()
        self.cond.save_checkpoint(path, self.cond.theta_ema, self.cond.dead_tracker, step)


# --------------------------------------------------------------------------- #
# The loop (mirrors moe_sae.train.loop.train artifact-for-artifact)            #
# --------------------------------------------------------------------------- #


def train_jax(cfg, cond: SAECondition, batch_iter, out_dir, capacity=None, device=None):
    from moe_sae.train.loop import lr_at_step  # single source for the schedule

    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "final").mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "config.yaml")

    run = JaxRun(cond, capacity=capacity, device=device)
    bank_loads = torch.zeros(64, dtype=torch.int64)
    theta_trajectory: list[float] = []
    metrics_path = out_dir / "metrics.jsonl"
    t_start = time.time()

    n_steps = cfg.n_steps
    for step in range(n_steps):
        batch = next(batch_iter)
        lr = lr_at_step(step, cfg)
        out, host_route_counts = run.training_step(batch, lr)
        theta_trajectory.append(out["theta_b"])
        bank_loads += host_route_counts

        if step % cfg.log_every == 0 or step == n_steps - 1:
            line = {
                "step": step, "lr": lr, "grad_norm": out.pop("grad_norm"),
                "theta_ema": cond.theta_ema.value,
                "wall_time": time.time() - t_start, **out,
            }
            with metrics_path.open("a") as fh:
                fh.write(json.dumps(line) + "\n")
        else:
            out.pop("grad_norm")

        if (step + 1) % cfg.ckpt_every_steps == 0 and (step + 1) < n_steps:
            run.save_checkpoint(out_dir / "checkpoints" / f"{step + 1}.safetensors", step + 1)

    (out_dir / "final" / "theta_star.json").write_text(
        json.dumps(
            {
                "theta_star": cond.theta_ema.value, "decay": cond.theta_ema.decay,
                "n_steps": n_steps, "condition": cond.condition, "seed": cfg.seed,
            },
            indent=2,
        )
    )
    np.save(out_dir / "theta_trajectory.npy", np.asarray(theta_trajectory, np.float64))
    np.save(out_dir / "final" / "bank_loads.npy", bank_loads.numpy())
    np.save(out_dir / "final" / "dead_final.npy", cond.dead_tracker.dead_mask().numpy())
    run.save_checkpoint(out_dir / "final" / "checkpoint.safetensors", n_steps)
