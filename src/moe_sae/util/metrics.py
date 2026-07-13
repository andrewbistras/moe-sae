"""Ops-only metrics utilities: MetricsSink, MFUMeter, GoodputTracker.

Observability tooling for the training loops (contract §Run artifact layout
in docs/INTERFACES.md; MFU/goodput targets in docs/RUNBOOK.md §TPU v4-32
execution plan). Nothing here touches training math, selection, losses, or
data order — a `MetricsSink` with `jsonl_only=True` produces run artifacts
byte-identical to what `moe_sae.train.loop.train` writes today.

MetricsSink
  Writes the canonical metrics.jsonl line exactly as loop.py formats it
  (`json.dumps({"step": step, **scalars}) + "\\n"`, appended), and
  optionally mirrors numeric scalars to TensorBoard event files under
  `run_dir/tb/` (torch.utils.tensorboard; no wandb, no accounts).

MFUMeter / flops_per_step_estimate
  FLOPs-per-step counting basis (docs/RUNBOOK.md TPU section):
    fwd  = 2 (encoder + decoder GEMMs) × 2 (multiply-accumulate)
           × B × cands × d
    step = fwd + bwd = 3 × fwd            (backward ≈ 2 × forward)
  where cands is the per-token candidate count: 6f for routed conditions
  A/C/D/E (each token sees its 6 visible banks of width f, spec §12) and
  for the compute-matched global B; 64f for the capacity-matched global F
  (`global_width`, moe_sae.train.config). With grouped-GEMM `capacity`
  given (routed conditions only), the B×6 routed token-copies become
  n_banks × capacity padded slots, i.e. a padding factor of
  capacity × 64 / (B × 6) on the whole step (RUNBOOK: Cap=768 vs mean
  load 384 at B=4096 → ~2×). The decoder is counted at candidate width —
  the grouped GEMM runs over the same (padded) candidate slots, not over
  the k̄ post-BatchTopK survivors — so k̄ does not enter the count; it is
  kept in the signature to document the shape being estimated.

GoodputTracker
  Wall-time accounting split into compute / data-wait / checkpoint buckets
  via context managers; `report()` returns fractions of total wall time.

Decisions (recorded): MFU EMA decay = 0.98 (ops-only smoothing, not a
training constant); TensorBoard unavailability degrades MetricsSink to
jsonl-only with a warning instead of raising (observability must never
kill a run); GoodputTracker takes an injectable `clock` for deterministic
tests; capacity passed for a global condition (B/F) is a ValueError —
capacity padding is a routed-condition concept.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Callable, Iterator

from contextlib import contextmanager

import numpy as np

from moe_sae.model_loading import NUM_ROUTED_EXPERTS, TOP_K_ROUTED
from moe_sae.train.config import CONDITIONS, global_width

ROUTED_CONDITIONS = ("A", "C", "D", "E")


def gini_coefficient(x) -> float:
    """Gini coefficient of a nonnegative count vector (0 = perfectly uniform,
    (n−1)/n = all mass on one entry). Zero-total vectors return 0.0."""
    v = np.sort(np.asarray(x, dtype=np.float64))
    n = v.size
    s = float(v.sum())
    if n == 0 or s <= 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * v)) / (n * s) - (n + 1) / n)


def load_dispersion_stats(counts, prefix: str = "load") -> dict:
    """A19 (v0.1.2 addendum) per-batch load-dispersion diagnostics from a
    per-bank count vector (`bank_route_counts` for host loads; condition D's
    allocated budgets for `prefix='budget'`): max, Gini, and the fraction of
    zero-count banks. Ops-only observability — no training-math change."""
    v = np.asarray(counts, dtype=np.int64)
    return {
        f"{prefix}_max": int(v.max()) if v.size else 0,
        f"{prefix}_gini": gini_coefficient(v),
        f"{prefix}_zero_frac": float((v == 0).mean()) if v.size else 0.0,
    }


class MetricsSink:
    """Append canonical metrics.jsonl lines; optionally mirror to TensorBoard.

    `log(step, scalars)` writes `json.dumps({"step": step, **scalars})`
    followed by a newline, appended to `run_dir/metrics.jsonl` — byte-for-byte
    the line `moe_sae.train.loop.train` currently emits when handed the same
    scalars dict in the same insertion order.

    Args:
        run_dir: run artifact directory; metrics.jsonl lives at its root.
        enable_tensorboard: mirror numeric scalars to `run_dir/tb/` event
            files via torch.utils.tensorboard.SummaryWriter.
        jsonl_only: registered-artifact-identical mode — write nothing but
            metrics.jsonl (no tb/ directory is ever created). Overrides
            enable_tensorboard.
    """

    def __init__(
        self,
        run_dir: str | Path,
        enable_tensorboard: bool = True,
        jsonl_only: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._writer = None
        if enable_tensorboard and not jsonl_only:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except ImportError:
                warnings.warn(
                    "tensorboard is not installed; MetricsSink degrading to "
                    "jsonl-only (pip install tensorboard, dev group)",
                    stacklevel=2,
                )

    def log(self, step: int, scalars: dict) -> None:
        """Append one metrics.jsonl line and (optionally) TensorBoard scalars."""
        line = {"step": step, **scalars}
        with self.metrics_path.open("a") as fh:
            fh.write(json.dumps(line) + "\n")
        if self._writer is not None:
            for key, value in scalars.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                self._writer.add_scalar(key, value, global_step=step)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "MetricsSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def flops_per_step_estimate(
    condition: str,
    B: int,
    f: int,
    d: int,
    k_bar: int,
    capacity: int | None = None,
) -> float:
    """FLOPs per training step under the RUNBOOK TPU counting basis.

    fwd = 2 (enc+dec) × 2 (MAC) × B × cands × d; step = 3 × fwd (bwd = 2×fwd).
    cands = 6f for A/C/D/E and B, 64f for F. With `capacity` (routed
    conditions only) the routed token-copies B×6 are padded to
    NUM_ROUTED_EXPERTS × capacity grouped-GEMM slots.

    `k_bar` is documented but unused: encoder and decoder are both counted
    at (padded) candidate width, per the grouped-GEMM execution model.
    """
    assert condition in CONDITIONS, condition
    del k_bar  # see docstring: decoder counted at candidate width
    if condition in ROUTED_CONDITIONS:
        # token-copy × per-bank-width product for the routed grouped GEMM
        if capacity is None:
            copies = B * TOP_K_ROUTED
        else:
            copies = NUM_ROUTED_EXPERTS * capacity
        cands_times_B = copies * f
    else:
        if capacity is not None:
            raise ValueError(
                f"capacity padding applies to routed conditions only, got "
                f"condition={condition!r}"
            )
        cands_times_B = B * global_width(condition, f)  # 6f for B, 64f for F
    return 3.0 * 2.0 * 2.0 * cands_times_B * d


class MFUMeter:
    """Model-FLOPs-utilization meter: per-step timing → steps/s, MFU, EMA-MFU.

    Args:
        flops_per_step: FLOPs one optimizer step costs (see
            `flops_per_step_estimate`).
        peak_flops: hardware peak FLOP/s to normalize against.
        ema_decay: EMA smoothing for `ema_mfu` (ops-only constant, 0.98).
    """

    def __init__(
        self, flops_per_step: float, peak_flops: float, ema_decay: float = 0.98
    ) -> None:
        assert flops_per_step > 0 and peak_flops > 0
        assert 0.0 <= ema_decay < 1.0
        self.flops_per_step = float(flops_per_step)
        self.peak_flops = float(peak_flops)
        self.ema_decay = ema_decay
        self._ema_mfu: float | None = None

    def update(self, step_seconds: float) -> dict:
        """Record one step's wall time; returns steps_per_sec, mfu, ema_mfu."""
        assert step_seconds > 0, step_seconds
        steps_per_sec = 1.0 / step_seconds
        mfu = self.flops_per_step * steps_per_sec / self.peak_flops
        if self._ema_mfu is None:
            self._ema_mfu = mfu
        else:
            self._ema_mfu = self.ema_decay * self._ema_mfu + (1.0 - self.ema_decay) * mfu
        return {"steps_per_sec": steps_per_sec, "mfu": mfu, "ema_mfu": self._ema_mfu}


class GoodputTracker:
    """Wall-time accounting: compute / data-wait / checkpoint buckets.

    Usage::

        gp = GoodputTracker()
        with gp.data_wait():
            batch = next(it)
        with gp.compute():
            ...train step...
        with gp.checkpoint():
            ...save...
        gp.report()  # fractions of wall time since construction

    Args:
        clock: monotonic time source, seconds (injectable for tests).
    """

    BUCKETS = ("compute", "data_wait", "checkpoint")

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._t0 = clock()
        self._seconds = {name: 0.0 for name in self.BUCKETS}

    @contextmanager
    def _track(self, bucket: str) -> Iterator[None]:
        start = self._clock()
        try:
            yield
        finally:
            self._seconds[bucket] += self._clock() - start

    def compute(self):
        return self._track("compute")

    def data_wait(self):
        return self._track("data_wait")

    def checkpoint(self):
        return self._track("checkpoint")

    def report(self) -> dict:
        """Fractions of wall time per bucket, plus untracked `other_frac`
        and `total_seconds`. All-zero fractions when no time has elapsed."""
        total = self._clock() - self._t0
        if total <= 0:
            fracs = {name: 0.0 for name in self.BUCKETS}
            other = 0.0
        else:
            fracs = {name: sec / total for name, sec in self._seconds.items()}
            other = max(0.0, 1.0 - sum(fracs.values()))
        return {
            "compute_frac": fracs["compute"],
            "data_wait_frac": fracs["data_wait"],
            "checkpoint_frac": fracs["checkpoint"],
            "other_frac": other,
            "total_seconds": total,
        }
