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

Resume / anomaly-halt / ops-metrics semantics are shared with the torch loop
(helpers imported from moe_sae.train.loop; see its module docstring and
Decisions). JAX-specific resume state persisted in the {step}.aux.npz
sidecar: the SpecAdamJax FP32 moments + step count, and the (possibly
ratcheted) grouped-GEMM capacity so a resumed run re-jits with the identical
static capacity (-1 encodes "not yet chosen"/global).
"""

from __future__ import annotations

import json
import logging
import math
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
from moe_sae.train.loop import (
    OPS_METRICS_NAME,
    aux_path_for,
    halt_anomaly,
    load_aux,
    lr_at_step,
    open_sinks,
    resolve_resume,
    restore_condition,
    run_logbook_path,
    save_aux,
    truncate_metrics_jsonl,
)
from moe_sae.util.manifest import logbook_append
from moe_sae.util.metrics import (
    GoodputTracker,
    MFUMeter,
    flops_per_step_estimate,
    load_dispersion_stats,
)

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
    preacts_raw = sj.encode_routed(
        params["W_enc"], params["b_enc"], mu, c, bank_flat, order, dest,
        cap=cap, n_banks=n_banks, out_f=f, raw=True,
    )
    preacts = jax.nn.relu(preacts_raw)  # MAIN path semantics unchanged (A1)
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

    # A1: aux selection ranks dead-masked RAW preacts; aux values are RAW.
    raw_scores = jax.lax.stop_gradient(preacts_raw).astype(jnp.float32)
    dead_vis = dead_ef[bank_flat].reshape(B, S * f)  # [N=B·S rows of banks]
    aux_mask = sj.aux_select_mask(raw_scores.reshape(B, S * f), dead_vis, aux_budget)
    z_aux = preacts_raw * aux_mask.reshape(B, S, f).astype(preacts_raw.dtype)
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
    aux_rows = aux_mask.reshape(n, f)
    aux_bank_counts = jnp.zeros(n_banks, jnp.int32).at[bank_flat].add(
        aux_rows.sum(axis=1).astype(jnp.int32)
    )
    aux = {
        "l_local": l_local, "l_aggregate": l_agg, "l_aux": l_aux,
        "theta_b": theta_b, "k_selected": mask.sum().astype(jnp.int32),
        "l0_tok": l0_tok, "fired_mask": fired, "bank_fire_counts": bank_fire,
        # A1 diagnostics
        "aux_selected": aux_mask.sum().astype(jnp.int32),
        "aux_main_overlap": (aux_mask.reshape(B, S, f) & mask).sum().astype(jnp.int32),
        "aux_bank_counts": aux_bank_counts,
    }
    return loss, aux


def _global_loss(
    params, x, mu, dead_f, *, kind, k_bar, f_dict, d, aux_alpha, aux_budget,
):
    B = x.shape[0]
    preacts_raw = sj.encode_global(params["W_enc"], params["b_enc"], mu, x, raw=True)
    preacts = jax.nn.relu(preacts_raw)  # MAIN path semantics unchanged (A1)
    scores = jax.lax.stop_gradient(preacts).astype(jnp.float32)
    mask_flat, theta_b = sj.global_topk_mask(scores.reshape(-1), k_bar * B)
    mask = mask_flat.reshape(B, f_dict)

    z = preacts * mask.astype(preacts.dtype)
    chat = sj.decode_global(params["D"], mu, z, add_mu=True)

    x32 = x.astype(jnp.float32)
    mu_rows = jnp.broadcast_to(mu, (B, d))
    l_local = sj.local_nmse(x32, chat, mu_rows)
    l_agg = jax.lax.stop_gradient(sj.aggregate_nmse(x32, chat.astype(jnp.float32)))

    # A1: aux selection ranks dead-masked RAW preacts; aux values are RAW.
    raw_scores = jax.lax.stop_gradient(preacts_raw).astype(jnp.float32)
    dead_vis = jnp.broadcast_to(dead_f.reshape(1, f_dict), (B, f_dict))
    aux_mask = sj.aux_select_mask(raw_scores, dead_vis, aux_budget)
    z_aux = preacts_raw * aux_mask.astype(preacts_raw.dtype)
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
        # A1 diagnostics
        "aux_selected": aux_mask.sum().astype(jnp.int32),
        "aux_main_overlap": (aux_mask & mask).sum().astype(jnp.int32),
        "aux_bank_counts": aux_mask.sum().astype(jnp.int32).reshape(1),
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
        self._aux_grad_fn = None
        self._kind = None

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
                self._aux_grad_fn = None

    def _loss_fn(self, kind: str):
        c = self.cond
        if self.is_global:
            return partial(
                _global_loss, kind=kind, k_bar=c.k_bar, f_dict=c.f_dict, d=c.d,
                aux_alpha=c.aux_alpha, aux_budget=c.aux_token_budget,
            )
        return partial(
            _routed_loss, kind=kind, k_bar=c.k_bar, cap=self.capacity,
            n_banks=c.n_banks, f=c.f, d=c.d,
            aux_alpha=c.aux_alpha, aux_budget=c.aux_token_budget,
        )

    def _build_grad_fn(self, kind: str):
        self._kind = kind
        return jax.jit(jax.value_and_grad(self._loss_fn(kind), has_aux=True))

    def _build_aux_grad_fn(self):
        """A1 (ii): gradient of the α-scaled aux term ONLY — a second
        backward, mirroring moe_sae.train.loop.aux_grad_norm_ratio. Built
        lazily and invoked exclusively at log steps (documented ~10% log-step
        cost: one extra forward+backward of the same loss graph)."""
        fn = self._loss_fn(self._kind)
        alpha = self.cond.aux_alpha

        def aux_term(*args):
            _, aux = fn(*args)
            return alpha * aux["l_aux"]

        return jax.jit(jax.grad(aux_term))

    def _aux_grad_ratio(self, call_args, grads) -> float:
        """‖g_aux‖ / ‖g_main‖ with g_main = g_total − g_aux elementwise."""
        if self._aux_grad_fn is None:
            self._aux_grad_fn = self._build_aux_grad_fn()
        g_aux = self._aux_grad_fn(*call_args)
        aux_sq = sum(
            jnp.sum(g.astype(jnp.float32) ** 2) for g in jax.tree.leaves(g_aux)
        )
        main_sq = sum(
            jnp.sum((grads[k].astype(jnp.float32) - g_aux[k].astype(jnp.float32)) ** 2)
            for k in grads
        )
        aux_norm = float(jnp.sqrt(aux_sq))
        main_norm = float(jnp.sqrt(main_sq))
        return aux_norm / max(main_norm, 1e-30)

    # ---- one training step ------------------------------------------------ #

    def training_step(
        self, batch: TrainBatch, lr: float, compute_aux_grad_ratio: bool = False
    ) -> tuple[dict, torch.Tensor]:
        c = self.cond
        experts = batch.experts.to(torch.int64)
        host_route_counts = torch.bincount(experts.reshape(-1), minlength=64)
        dead_t = c.dead_tracker.dead_mask()  # [n_dict_banks, f_dict] torch bool
        dead = jnp.asarray(dead_t.numpy())
        n_tokens = batch.c.shape[0]

        if self.is_global:
            x = jnp.asarray(
                batch.c.float().sum(dim=1).numpy(), self.dtype
            )
            if self._grad_fn is None:
                self._grad_fn = self._build_grad_fn("global")
            call_args = (self.params, x, self.mu, dead.reshape(-1))
            (loss, aux), grads = self._grad_fn(*call_args)
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
                mask_t, theta_b, theta_b_per_bank = per_bank_topk(
                    scores_t.reshape(-1), bank_of_cand, budgets
                )
                if c.theta_ema_vec is not None:
                    c.theta_ema_vec.update(theta_b_per_bank)  # A4 (D)
                ext_mask = jnp.asarray(mask_t.view(B, S, c.f).numpy())
                call_args = (self.params, cj, *args, self.mu, dead, ext_mask)
                (loss, aux), grads = self._grad_fn(*call_args)
            else:
                kind = "per_token" if c.condition == "E" else "global"
                if self._grad_fn is None:
                    self._grad_fn = self._build_grad_fn(kind)
                call_args = (self.params, cj, *args, self.mu, dead, None)
                (loss, aux), grads = self._grad_fn(*call_args)
                theta_b = float(aux["theta_b"])
            dict_route_counts = torch.bincount(
                bank_ids.reshape(-1), minlength=c.n_dict_banks
            )

        # A1 (ii): second backward of the aux term ONLY at log steps.
        aux_grad_ratio = None
        if compute_aux_grad_ratio and int(dead_t.sum()) > 0:
            aux_grad_ratio = self._aux_grad_ratio(call_args, grads)

        self.params, grad_norm = self.opt.step(self.params, grads, lr)

        l0 = np.asarray(aux["l0_tok"], np.float64)
        tt = batch.token_type.numpy()
        n_aux_sel = int(aux["aux_selected"])
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
            # A1 diagnostics (mirroring torch StepOutput.scalars key order)
            "aux_primary_ratio": (
                float(aux["l_aux"]) / float(aux["l_local"])
                if float(aux["l_local"]) > 0 else 0.0
            ),
            "aux_mean_selected_per_token": n_aux_sel / n_tokens if n_tokens else 0.0,
            "aux_frac_also_main": (
                int(aux["aux_main_overlap"]) / n_aux_sel if n_aux_sel else 0.0
            ),
            "aux_grad_norm_ratio": aux_grad_ratio,
            "grad_norm": grad_norm,
            "aux_bank_counts": torch.from_numpy(
                np.asarray(aux["aux_bank_counts"], np.int64)
            ),
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


_ADAM_PARAM_KEYS = ("W_enc", "b_enc", "D")


def _jax_aux_arrays(run: JaxRun) -> dict:
    """SpecAdamJax FP32 moments + step count + static capacity, npz-ready."""
    arrays: dict = {
        "adam_t": np.int64(run.opt.state["t"]),
        "capacity": np.int64(-1 if run.capacity is None else run.capacity),
    }
    for name in _ADAM_PARAM_KEYS:
        arrays[f"adam_m_{name}"] = np.asarray(run.opt.state["m"][name], np.float32)
        arrays[f"adam_v_{name}"] = np.asarray(run.opt.state["v"][name], np.float32)
    return arrays


def _load_jax_adam_state(run: JaxRun, aux) -> None:
    run.opt.state["t"] = int(aux["adam_t"])
    for name in _ADAM_PARAM_KEYS:
        run.opt.state["m"][name] = jnp.asarray(aux[f"adam_m_{name}"], jnp.float32)
        run.opt.state["v"][name] = jnp.asarray(aux[f"adam_v_{name}"], jnp.float32)


def train_jax(
    cfg,
    cond: SAECondition,
    batch_iter,
    out_dir,
    capacity=None,
    device=None,
    *,
    resume: bool = False,
    peak_flops: float | None = None,
    composer=None,
):
    """`composer` (optional): the BatchComposer behind `batch_iter`; when it
    records A15 text draw counts they are persisted as
    final/text_draw_counts.npz (mirrors moe_sae.train.loop.train)."""
    out_dir = Path(out_dir)
    start_step, ckpt_path, complete = resolve_resume(out_dir, resume)
    if complete:
        logbook_append(
            "resume_noop",
            {"out_dir": str(out_dir), "note": "final checkpoint exists; run complete"},
            path=run_logbook_path(out_dir),
        )
        return
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "final").mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "config.yaml")

    bank_loads = torch.zeros(64, dtype=torch.int64)
    theta_trajectory: list[float] = []
    # A1 diagnostic (v): cumulative per-bank aux-selection counts (sidecar).
    aux_bank_counts = torch.zeros(cond.n_dict_banks, dtype=torch.int64)
    gp = GoodputTracker()
    aux = None
    if ckpt_path is not None:
        # Restore the torch condition FIRST: JaxRun uploads its params.
        k = restore_condition(cond, ckpt_path)
        assert k == start_step, (k, start_step)
        aux = load_aux(ckpt_path)
        bank_loads = torch.from_numpy(aux["bank_loads"].copy())
        theta_trajectory = [float(x) for x in aux["theta_trajectory"]]
        if "aux_bank_counts" in aux.files:
            aux_bank_counts = torch.from_numpy(aux["aux_bank_counts"].copy())
        if capacity is None and int(aux["capacity"]) > 0:
            capacity = int(aux["capacity"])  # identical static jit capacity

    run = JaxRun(cond, capacity=capacity, device=device)
    if aux is not None:
        _load_jax_adam_state(run, aux)
        with gp.data_wait():
            # Composer fast-forward: the ONLY stream interaction on resume.
            for _ in range(start_step):
                next(batch_iter)
        logbook_append(
            "resume",
            {"step": start_step, "checkpoint": str(ckpt_path)},
            path=run_logbook_path(out_dir),
        )
    if resume:
        truncate_metrics_jsonl(out_dir / "metrics.jsonl", start_step)
        truncate_metrics_jsonl(out_dir / OPS_METRICS_NAME, start_step)

    meter = None
    if peak_flops is not None:
        cap_arg = run.capacity if cfg.condition in ("A", "C", "D", "E") else None
        meter = MFUMeter(
            flops_per_step_estimate(
                cfg.condition, cfg.batch_tokens, cfg.f, cond.d, cfg.k_bar,
                capacity=cap_arg,
            ),
            peak_flops,
        )
    sink, ops_sink = open_sinks(out_dir)
    t_start = time.time()
    spike = getattr(cfg, "spike_threshold", None)  # ops-only knob, default off

    n_steps = cfg.n_steps
    try:
        for step in range(start_step, n_steps):
            t_iter = time.perf_counter()
            with gp.data_wait():
                batch = next(batch_iter)
            is_log_step = step % cfg.log_every == 0 or step == n_steps - 1
            with gp.compute():
                lr = lr_at_step(step, cfg)
                out, host_route_counts = run.training_step(
                    batch, lr, compute_aux_grad_ratio=is_log_step
                )
                grad_norm = out.pop("grad_norm")
                aux_bank_counts += out.pop("aux_bank_counts")
                if not is_log_step:
                    out.pop("aux_grad_norm_ratio", None)
                if not math.isfinite(out["loss"]) or not math.isfinite(grad_norm):
                    reason = (
                        "non-finite loss" if not math.isfinite(out["loss"])
                        else "non-finite grad_norm"
                    )
                    halt_anomaly(
                        out_dir, step, reason,
                        {"loss": out["loss"], "grad_norm": grad_norm},
                        lambda p: run.save_checkpoint(p, step),
                    )
                if spike is not None and grad_norm > spike:
                    halt_anomaly(
                        out_dir, step, f"grad_norm spike > {spike}",
                        {"loss": out["loss"], "grad_norm": grad_norm},
                        lambda p: run.save_checkpoint(p, step),
                    )
                theta_trajectory.append(out["theta_b"])
                bank_loads += host_route_counts

            dt = max(time.perf_counter() - t_iter, 1e-9)
            ops_stats = meter.update(dt) if meter is not None else {"steps_per_sec": 1.0 / dt}
            if is_log_step:
                sink.log(
                    step,
                    {
                        "lr": lr, "grad_norm": grad_norm,
                        "theta_ema": cond.theta_ema.value,
                        "wall_time": time.time() - t_start, **out,
                    },
                )
                # A19: per-batch load-dispersion diagnostics; condition D also
                # logs its allocated-budget distribution (max/Gini).
                a19 = load_dispersion_stats(host_route_counts.numpy())
                if cond.condition == "D":
                    budgets = largest_remainder_budgets(
                        host_route_counts.float(), cfg.k_bar * batch.c.shape[0]
                    )
                    b19 = load_dispersion_stats(budgets.numpy(), prefix="budget")
                    a19 = {**a19, "budget_max": b19["budget_max"], "budget_gini": b19["budget_gini"]}
                ops_sink.log(
                    step,
                    {
                        **ops_stats,
                        **gp.report(),
                        # A1 diagnostics mirrored into the ops sidecar
                        "aux_primary_ratio": out["aux_primary_ratio"],
                        "aux_grad_norm_ratio": out["aux_grad_norm_ratio"],
                        "aux_mean_selected_per_token": out["aux_mean_selected_per_token"],
                        "aux_frac_also_main": out["aux_frac_also_main"],
                        **a19,
                    },
                )

            if (step + 1) % cfg.ckpt_every_steps == 0 and (step + 1) < n_steps:
                with gp.checkpoint():
                    ckpt = out_dir / "checkpoints" / f"{step + 1}.safetensors"
                    run.save_checkpoint(ckpt, step + 1)
                    save_aux(
                        aux_path_for(ckpt), bank_loads, theta_trajectory,
                        {
                            **_jax_aux_arrays(run),
                            "aux_bank_counts": aux_bank_counts.numpy(),
                        },
                    )
    finally:
        sink.close()
        ops_sink.close()

    # θ* frozen at end of training; condition D freezes the per-bank vector
    # θ*_e (A4), the scalar staying as the log-only convention.
    theta_star_record = {
        "theta_star": cond.theta_ema.value, "decay": cond.theta_ema.decay,
        "n_steps": n_steps, "condition": cond.condition, "seed": cfg.seed,
    }
    if cond.theta_ema_vec is not None:
        theta_star_record = {
            "theta_star_per_bank": cond.theta_ema_vec.values_list(),
            **theta_star_record,
        }
    (out_dir / "final" / "theta_star.json").write_text(
        json.dumps(theta_star_record, indent=2)
    )
    np.save(out_dir / "theta_trajectory.npy", np.asarray(theta_trajectory, np.float64))
    np.save(out_dir / "final" / "bank_loads.npy", bank_loads.numpy())
    np.save(out_dir / "final" / "dead_final.npy", cond.dead_tracker.dead_mask().numpy())
    # A1 diagnostic (v): cumulative per-bank aux pressure over the whole run.
    np.save(out_dir / "final" / "aux_bank_counts.npy", aux_bank_counts.numpy())
    # A15: text draw counts (per (shard_idx, row)) for the robustness check.
    if composer is not None and getattr(composer, "record_draws", False):
        np.savez(
            str(out_dir / "final" / "text_draw_counts.npz"),
            **composer.text_draw_counts_arrays(),
        )
    run.save_checkpoint(out_dir / "final" / "checkpoint.safetensors", n_steps)
