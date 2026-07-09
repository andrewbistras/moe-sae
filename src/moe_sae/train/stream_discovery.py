"""Rotating-window streaming discovery: one GPU, no bucket, 14 runs (spec §10).

Per window: compose K frozen batches from live capture → write to disk →
each condition replays the window (one resident on GPU at a time, StateSwapper)
→ delete window. Matched data order across conditions is bitwise (identical
files); capture compute is paid once; peak activation storage = one window.

Per-condition artifacts are IDENTICAL in layout to moe_sae.train.loop.train
(metrics.jsonl, theta_trajectory.npy, final/{theta_star.json, bank_loads.npy,
dead_final.npy, checkpoint.safetensors}) — the eval/analysis stack is unaware
streaming happened.

Resilience posture (user-accepted, 2026-07-08): windows checkpoint the
parameter state periodically and Adam moments are disk-backed, but the
capture stream itself is not resumable — a session death mid-run means
restarting the stream. Colab A100 sessions comfortably cover the ~5-7 h run.

The per-step block below intentionally mirrors loop.train's step body; the
equivalence is enforced by tests/test_streaming.py (windowed+swapped run ==
plain loop.train on the same batches, parameters bit-equal).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from moe_sae.sae.conditions import SAECondition, build_condition
from moe_sae.sae.optim import SpecAdam
from moe_sae.train.config import RunConfig
from moe_sae.train.loop import lr_at_step
from moe_sae.train.streaming import StateSwapper, StreamComposer, WindowStore

logger = logging.getLogger(__name__)


class RunSlot:
    """One (condition, seed) training run inside the co-training rotation."""

    def __init__(self, cfg: RunConfig, mu_e, mu_route, out_root: Path):
        self.cfg = cfg
        self.name = cfg.run_name
        self.cond: SAECondition = build_condition(
            cfg.condition, cfg, mu_e, mu_route, cfg.seed
        )
        self.opt = SpecAdam(
            self.cond.optim_groups(),
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            clip=cfg.grad_clip,
        )
        self.out = out_root / cfg.run_name
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.out / "final").mkdir(parents=True, exist_ok=True)
        cfg.to_yaml(self.out / "config.yaml")
        self.bank_loads = torch.zeros(64, dtype=torch.int64)
        self.theta_trajectory: list[float] = []
        self.step = 0
        torch.manual_seed(cfg.seed)  # init happened in build_condition above

    def replay(self, batches) -> None:
        """Mirrors loop.train's per-step body (equivalence unit-tested)."""
        cond, cfg, opt = self.cond, self.cfg, self.opt
        metrics_path = self.out / "metrics.jsonl"
        t0 = time.time()
        for batch in batches:
            out = cond.training_step(batch, self.step)
            out.loss.backward()
            lr = lr_at_step(self.step, cfg)
            grad_norm = opt.step(lr)
            opt.zero_grad()
            cond.theta_ema.update(out.theta_b)
            cond.dead_tracker.update(out.dict_route_counts, out.fired_mask)
            self.theta_trajectory.append(out.theta_b)
            self.bank_loads += out.bank_route_counts.to(torch.int64)
            if self.step % cfg.log_every == 0 or self.step == cfg.n_steps - 1:
                line = {
                    "step": self.step, "lr": lr, "grad_norm": grad_norm,
                    "theta_ema": cond.theta_ema.value,
                    "wall_time": time.time() - t0, **out.scalars(),
                }
                with metrics_path.open("a") as fh:
                    fh.write(json.dumps(line) + "\n")
            self.step += 1

    def finalize(self) -> None:
        cond, cfg = self.cond, self.cfg
        (self.out / "final" / "theta_star.json").write_text(
            json.dumps(
                {
                    "theta_star": cond.theta_ema.value, "decay": cond.theta_ema.decay,
                    "n_steps": self.step, "condition": cond.condition, "seed": cfg.seed,
                },
                indent=2,
            )
        )
        np.save(self.out / "theta_trajectory.npy",
                np.asarray(self.theta_trajectory, np.float64))
        np.save(self.out / "final" / "bank_loads.npy", self.bank_loads.numpy())
        np.save(self.out / "final" / "dead_final.npy",
                cond.dead_tracker.dead_mask().numpy())
        cond.save_checkpoint(
            self.out / "final" / "checkpoint.safetensors",
            cond.theta_ema, cond.dead_tracker, self.step,
        )


def stream_discovery(
    configs: list[RunConfig],
    example_source,
    mu_e: torch.Tensor,
    mu_route: torch.Tensor,
    out_root: str | Path = "runs",
    window_batches: int = 800,
    device: str = "cuda",
    work_dir: str | Path = "stream_work",
    model=None,  # optional: moved to CPU during replay to free GPU headroom
    ckpt_every_windows: int = 4,
    min_visual_buffer: int = 65536,  # frozen shuffle-buffer depth (contract)
    pipelined: bool = False,  # A100-80: compose window w+1 during replay of w
) -> None:
    n_steps = configs[0].n_steps
    assert all(c.n_steps == n_steps for c in configs), "runs must share n_steps"
    assert all(c.batch_tokens == configs[0].batch_tokens for c in configs)

    out_root = Path(out_root)
    work = Path(work_dir)
    store = WindowStore(work / "windows")
    swapper = StateSwapper(work / "adam_states")
    composer = StreamComposer(
        example_source, configs[0].batch_tokens, min_visual_buffer=min_visual_buffer
    )

    slots = [RunSlot(cfg, mu_e, mu_route, out_root) for cfg in configs]
    for s in slots:  # park everything off-GPU; moments to disk memmaps
        swapper.swap_out(s.name, s.cond, s.opt)

    n_windows = (n_steps + window_batches - 1) // window_batches
    logger.info("streaming %d runs × %d steps in %d windows of %d (pipelined=%s)",
                len(slots), n_steps, n_windows, window_batches, pipelined)

    def compose_window(w: int) -> None:
        k = min(window_batches, n_steps - w * window_batches)
        if not store.exists(w):
            batches = [composer.next_batch() for _ in range(k)]
            store.write(w, batches)
            logger.info("window %d/%d composed (%d batches, %d examples consumed)",
                        w + 1, n_windows, k, composer.examples_consumed)

    producer_state = {"exc": None}
    if pipelined:
        import threading

        events = [threading.Event() for _ in range(n_windows)]

        def producer():
            try:
                for w in range(n_windows):
                    compose_window(w)
                    events[w].set()
            except BaseException as e:  # surfaced by the consumer below
                producer_state["exc"] = e
                for ev in events:
                    ev.set()

        if model is not None:
            model.to(device)  # stays resident: capture overlaps replay (80 GB)
        threading.Thread(target=producer, daemon=True).start()

    for w in range(n_windows):
        if pipelined:
            events[w].wait()
            if producer_state["exc"] is not None:
                raise producer_state["exc"]
        else:
            if model is not None:
                model.to(device)
            compose_window(w)
            if model is not None:
                model.to("cpu")
                if device == "cuda":
                    torch.cuda.empty_cache()

        batches = store.read(w)
        for s in slots:
            swapper.swap_in(s.name, s.cond, s.opt, device)
            s.replay(batches)
            swapper.swap_out(s.name, s.cond, s.opt)
            if device == "cuda":
                torch.cuda.empty_cache()
        store.delete(w)

        if (w + 1) % ckpt_every_windows == 0 and (w + 1) < n_windows:
            for s in slots:
                s.cond.save_checkpoint(
                    s.out / "checkpoints" / f"{s.step}.safetensors",
                    s.cond.theta_ema, s.cond.dead_tracker, s.step,
                )
            logger.info("window %d: checkpoints written", w + 1)

    for s in slots:
        swapper.swap_in(s.name, s.cond, s.opt, "cpu")
        s.finalize()
    logger.info("stream discovery complete: %d runs under %s", len(slots), out_root)

