"""Training loop (spec §10–§11): schedule, step, logging, artifacts.

Per step: `condition.training_step` → `loss.backward()` → `SpecAdam.step(lr)`
→ θ-EMA update → DeadTracker update. LR schedule: linear warmup over
`cfg.warmup_steps` then cosine decay lr → lr_min over the remaining steps
(spec §11).

Artifacts (contract §Run artifact layout): config.yaml echo, metrics.jsonl,
theta_trajectory.npy, checkpoints/{step}.safetensors, and final/
{theta_star.json, bank_loads.npy, dead_final.npy, checkpoint.safetensors}.

`train` deliberately does NOT call `cfg.validate()`: validation of frozen
constants belongs to the entry-point script (scripts/run_train.py); tests
drive the loop with tiny non-frozen configs through the identical code path.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from moe_sae.sae import TrainBatch
from moe_sae.sae.conditions import SAECondition
from moe_sae.sae.optim import SpecAdam
from moe_sae.train.config import RunConfig


def lr_at_step(step: int, cfg: RunConfig) -> float:
    """Spec §11: linear warmup to cfg.lr over warmup_steps, then cosine decay
    to cfg.lr_min at cfg.n_steps. `step` is zero-indexed."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    span = max(1, cfg.n_steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / span)
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * progress))


def train(
    cfg: RunConfig,
    condition: SAECondition,
    batch_iter: Iterator[TrainBatch],
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "final").mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "config.yaml")

    torch.manual_seed(cfg.seed)

    opt = SpecAdam(
        condition.optim_groups(),
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        clip=cfg.grad_clip,
    )
    theta_ema = condition.theta_ema
    dead_tracker = condition.dead_tracker

    bank_loads = None  # cumulative host-expert routed tokens, [64] int64
    theta_trajectory: list[float] = []
    metrics_path = out_dir / "metrics.jsonl"
    t_start = time.time()

    n_steps = cfg.n_steps
    for step in range(n_steps):
        batch = next(batch_iter)
        out = condition.training_step(batch, step)
        out.loss.backward()
        lr = lr_at_step(step, cfg)
        grad_norm = opt.step(lr)
        opt.zero_grad()

        theta_ema.update(out.theta_b)
        dead_tracker.update(out.dict_route_counts, out.fired_mask)
        theta_trajectory.append(out.theta_b)
        if bank_loads is None:
            bank_loads = out.bank_route_counts.clone().to(torch.int64)
        else:
            bank_loads += out.bank_route_counts.to(torch.int64)

        if step % cfg.log_every == 0 or step == n_steps - 1:
            line = {
                "step": step,
                "lr": lr,
                "grad_norm": grad_norm,
                "theta_ema": theta_ema.value,
                "wall_time": time.time() - t_start,
                **out.scalars(),
            }
            with metrics_path.open("a") as fh:
                fh.write(json.dumps(line) + "\n")

        if (step + 1) % cfg.ckpt_every_steps == 0 and (step + 1) < n_steps:
            condition.save_checkpoint(
                out_dir / "checkpoints" / f"{step + 1}.safetensors",
                theta_ema,
                dead_tracker,
                step + 1,
            )

    # θ* frozen at end of training (spec §6, changelog #6).
    (out_dir / "final" / "theta_star.json").write_text(
        json.dumps(
            {
                "theta_star": theta_ema.value,
                "decay": theta_ema.decay,
                "n_steps": n_steps,
                "condition": condition.condition,
                "seed": cfg.seed,
            },
            indent=2,
        )
    )
    np.save(out_dir / "theta_trajectory.npy", np.asarray(theta_trajectory, dtype=np.float64))
    np.save(out_dir / "final" / "bank_loads.npy", bank_loads.numpy())
    np.save(out_dir / "final" / "dead_final.npy", dead_tracker.dead_mask().numpy())
    condition.save_checkpoint(
        out_dir / "final" / "checkpoint.safetensors", theta_ema, dead_tracker, n_steps
    )
