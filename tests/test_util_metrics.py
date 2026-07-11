"""Tests for moe_sae.util.metrics (ops tooling; no training math touched).

Covers: metrics.jsonl byte-compatibility against a golden line built exactly
the way moe_sae.train.loop.train builds it; MFU FLOPs counting vs
hand-computed values for the A and F discovery shapes (RUNBOOK TPU counting
basis: fwd = 2 enc+dec × 2 MAC × B × cands × d, step = 3 × fwd, cands = 6f
for A/B/C/D/E and 64f for F, grouped-GEMM capacity padding
capacity×64/(B×6) for routed conditions); goodput fractions with an
injected clock; TensorBoard event files (skipped cleanly if tensorboard is
not installed).

CPU-only, tiny, fast; never loads deepseek-vl2-tiny.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from moe_sae.util.metrics import (
    GoodputTracker,
    MetricsSink,
    MFUMeter,
    flops_per_step_estimate,
)

HAS_TENSORBOARD = importlib.util.find_spec("tensorboard") is not None


# ---------------------------------------------------------------------------
# MetricsSink: jsonl byte-compatibility with train.loop
# ---------------------------------------------------------------------------


def _loop_style_line(step: int, scalars: dict) -> str:
    """Byte-for-byte what loop.py writes: json.dumps({"step": step, **rest})
    + newline (train/loop.py, metrics block)."""
    return json.dumps({"step": step, **scalars}) + "\n"


def _fake_scalars() -> dict:
    """Keys/ordering shaped like loop.py's line: lr, grad_norm, theta_ema,
    wall_time, then out.scalars(). Values chosen with non-trivial float
    reprs to catch any formatting drift."""
    return {
        "lr": 3.0000000000000004e-05,
        "grad_norm": 0.30000000000000004,
        "theta_ema": 0.1 + 0.2,
        "wall_time": 12.345678901234567,
        "loss": 1.2345e-01,
        "nmse": 0.7071067811865476,
        "l0": 64,
        "dead": 0,
        "theta_b": -0.5,
    }


def test_jsonl_byte_compat_single_line(tmp_path):
    scalars = _fake_scalars()
    golden = _loop_style_line(7, scalars)

    sink = MetricsSink(tmp_path, jsonl_only=True)
    sink.log(7, scalars)
    sink.close()

    assert (tmp_path / "metrics.jsonl").read_bytes() == golden.encode()


def test_jsonl_appends_multiple_lines(tmp_path):
    scalars = _fake_scalars()
    golden = _loop_style_line(0, scalars) + _loop_style_line(50, scalars)

    sink = MetricsSink(tmp_path, jsonl_only=True)
    sink.log(0, scalars)
    sink.log(50, scalars)

    assert (tmp_path / "metrics.jsonl").read_bytes() == golden.encode()


def test_jsonl_only_writes_no_tb_dir(tmp_path):
    sink = MetricsSink(tmp_path, enable_tensorboard=True, jsonl_only=True)
    sink.log(0, _fake_scalars())
    sink.close()
    assert not (tmp_path / "tb").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["metrics.jsonl"]


def test_jsonl_identical_with_and_without_tensorboard(tmp_path):
    scalars = _fake_scalars()
    a, b = tmp_path / "a", tmp_path / "b"
    MetricsSink(a, jsonl_only=True).log(3, scalars)
    with MetricsSink(b, enable_tensorboard=True) as sink:
        sink.log(3, scalars)
    assert (a / "metrics.jsonl").read_bytes() == (b / "metrics.jsonl").read_bytes()


@pytest.mark.skipif(not HAS_TENSORBOARD, reason="tensorboard not installed")
def test_tensorboard_event_files_created(tmp_path):
    with MetricsSink(tmp_path, enable_tensorboard=True) as sink:
        sink.log(0, _fake_scalars())
        sink.log(1, {**_fake_scalars(), "note": "non-numeric values skipped"})
        sink.flush()
    events = list((tmp_path / "tb").glob("events.out.tfevents.*"))
    assert events, "no TensorBoard event files under run_dir/tb/"
    assert all(p.stat().st_size > 0 for p in events)


# ---------------------------------------------------------------------------
# flops_per_step_estimate: hand-computed A/F discovery shapes
# ---------------------------------------------------------------------------
#
# Counting basis (docs/RUNBOOK.md TPU section; module docstring):
#   step = 3 × fwd,  fwd = 2 (enc+dec) × 2 (MAC) × B × cands × d
# A (routed, capacity=None): cands = 6f (top-6 of 64 banks, spec §12)
#   → 12 × 4096 × 6·4096 × 1280 = 1.5462e12
# F (global, 64f dictionary):
#   → 12 × 4096 × 64·4096 × 1280 = 1.6493e13
# A at capacity=768: grouped-GEMM slots 64×768 vs routed copies 4096×6
#   → padding factor 768·64/(4096·6) = 2.0 → 3.0924e12

SHAPE = dict(B=4096, f=4096, d=1280, k_bar=64)


def test_flops_condition_A_uncapped():
    got = flops_per_step_estimate("A", **SHAPE, capacity=None)
    exact = 12 * 4096 * 6 * 4096 * 1280
    assert got == pytest.approx(exact)
    assert got == pytest.approx(1.55e12, rel=0.05)


def test_flops_condition_F():
    got = flops_per_step_estimate("F", **SHAPE)
    exact = 12 * 4096 * 64 * 4096 * 1280
    assert got == pytest.approx(exact)
    assert got == pytest.approx(16.6e12, rel=0.05)


def test_flops_condition_A_capacity_768():
    got = flops_per_step_estimate("A", **SHAPE, capacity=768)
    assert got == pytest.approx(3.1e12, rel=0.05)
    # RUNBOOK: Cap=768 vs mean load 384 → exactly 2× the uncapped A step.
    assert got == pytest.approx(2.0 * flops_per_step_estimate("A", **SHAPE))


def test_flops_B_matches_A_active_compute():
    # B is the active-compute-matched global (width 6f, spec §12.B).
    assert flops_per_step_estimate("B", **SHAPE) == pytest.approx(
        flops_per_step_estimate("A", **SHAPE)
    )


def test_flops_capacity_rejected_for_global_conditions():
    with pytest.raises(ValueError):
        flops_per_step_estimate("F", **SHAPE, capacity=768)


# ---------------------------------------------------------------------------
# MFUMeter
# ---------------------------------------------------------------------------


def test_mfu_meter_arithmetic():
    meter = MFUMeter(flops_per_step=1e12, peak_flops=2e12, ema_decay=0.98)
    out = meter.update(1.0)
    assert out["steps_per_sec"] == pytest.approx(1.0)
    assert out["mfu"] == pytest.approx(0.5)
    assert out["ema_mfu"] == pytest.approx(0.5)  # EMA seeded at first value
    out = meter.update(0.5)
    assert out["steps_per_sec"] == pytest.approx(2.0)
    assert out["mfu"] == pytest.approx(1.0)
    assert out["ema_mfu"] == pytest.approx(0.98 * 0.5 + 0.02 * 1.0)


def test_mfu_meter_discovery_shape_end_to_end():
    # A-shape at 0.01 s/step on a 275 TFLOP/s v4 chip pair-equivalent:
    flops = flops_per_step_estimate("A", **SHAPE)
    meter = MFUMeter(flops_per_step=flops, peak_flops=275e12)
    out = meter.update(0.01)
    assert out["mfu"] == pytest.approx(flops / (0.01 * 275e12))


# ---------------------------------------------------------------------------
# GoodputTracker
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_goodput_fractions():
    fc = _FakeClock()
    gp = GoodputTracker(clock=fc)
    with gp.compute():
        fc.t += 6.0
    with gp.data_wait():
        fc.t += 3.0
    with gp.checkpoint():
        fc.t += 1.0
    rep = gp.report()
    assert rep["total_seconds"] == pytest.approx(10.0)
    assert rep["compute_frac"] == pytest.approx(0.6)
    assert rep["data_wait_frac"] == pytest.approx(0.3)
    assert rep["checkpoint_frac"] == pytest.approx(0.1)
    assert rep["other_frac"] == pytest.approx(0.0)


def test_goodput_untracked_time_is_other():
    fc = _FakeClock()
    gp = GoodputTracker(clock=fc)
    with gp.compute():
        fc.t += 5.0
    fc.t += 5.0  # untracked
    rep = gp.report()
    assert rep["compute_frac"] == pytest.approx(0.5)
    assert rep["other_frac"] == pytest.approx(0.5)


def test_goodput_bucket_accumulates_on_exception():
    fc = _FakeClock()
    gp = GoodputTracker(clock=fc)
    with pytest.raises(RuntimeError):
        with gp.data_wait():
            fc.t += 2.0
            raise RuntimeError("stream died")
    fc.t += 2.0
    rep = gp.report()
    assert rep["data_wait_frac"] == pytest.approx(0.5)


def test_goodput_zero_elapsed():
    gp = GoodputTracker(clock=_FakeClock())
    rep = gp.report()
    assert rep["total_seconds"] == 0.0
    assert rep["compute_frac"] == 0.0
    assert rep["other_frac"] == 0.0
