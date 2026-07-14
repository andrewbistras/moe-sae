from __future__ import annotations

import json

import numpy as np
import pytest

from moe_sae.metrics import (
    MetricError,
    MetricsSink,
    ReconstructionAccumulator,
    build_efficiency_report,
    compute_goodput,
    compute_mfu,
    gain_recovery,
    paired_cost_ratio,
    reconstruction_metrics,
    render_headline,
    write_report_json,
)


def row(condition: str, seed: int, ev: float, cost: float) -> dict:
    return {
        "condition": condition,
        "seed": seed,
        "ev": ev,
        "nmse": 1.0 - ev,
        "l0": 64.0,
        "dead_rate": 0.01,
        "chip_seconds": cost,
        "wall_seconds": cost / 16,
        "tokens_per_second": 10_000.0,
    }


def stable_rows() -> list[dict]:
    rows = []
    for seed, delta in enumerate((0.00, 0.01, -0.01), start=1):
        rows.extend(
            [
                row("A", seed, 0.65 + delta, 60 + seed),
                row("B", seed, 0.50 + delta, 55 + seed),
                row("F", seed, 0.80 + delta, 1_000 + 10 * seed),
            ]
        )
    return rows


def test_reconstruction_metrics_common_sst():
    target = np.array([[1.0, 2.0], [3.0, 4.0]])
    reconstruction = target + 1.0
    metrics = reconstruction_metrics(target, reconstruction, mean=np.array([2.0, 3.0]))
    assert metrics == {
        "tokens": 2,
        "sse": 4.0,
        "sst": 4.0,
        "nmse": 1.0,
        "ev": 0.0,
    }


def test_reconstruction_metrics_reject_shape_nonfinite_and_zero_sst():
    with pytest.raises(MetricError, match="shapes differ"):
        reconstruction_metrics(np.zeros((2, 3)), np.zeros((2, 2)), np.zeros(3))
    with pytest.raises(MetricError, match="finite"):
        reconstruction_metrics(np.array([[np.nan, 0.0]]), np.zeros((1, 2)), np.zeros(2))
    with pytest.raises(MetricError, match="SST"):
        reconstruction_metrics(np.ones((2, 2)), np.ones((2, 2)), np.ones(2))


def test_streaming_accumulator_exact_l0_and_dead_rate():
    accumulator = ReconstructionAccumulator(np.array([0.0, 0.0]), feature_count=8)
    accumulator.update(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.5, 0.0], [0.0, 0.5]]),
        active_counts=np.array([64, 64]),
        fired_latent_ids=np.array([0, 1, 2]),
    )
    accumulator.update(
        np.array([[1.0, 1.0]]),
        np.array([[0.5, 0.5]]),
        active_counts=np.array([64]),
        fired_latent_ids=np.array([3, 4, 5, 6, 7]),
    )
    result = accumulator.finalize()
    assert result["tokens"] == 3
    assert result["nmse"] == pytest.approx(0.25)
    assert result["ev"] == pytest.approx(0.75)
    assert result["l0"] == 64
    assert result["exact_l0"] is True
    assert result["dead_rate"] == 0.0


def test_gain_recovery_stable_and_unclipped():
    result = gain_recovery(stable_rows())
    assert result["gain_recovery_percent"] == pytest.approx(50.0)
    assert result["status"] == "stable"
    assert result["headline_eligible"] is True
    assert result["paired_f_minus_b_ci95"] == pytest.approx([0.3, 0.3])

    over = [
        row("A", 1, 0.9, 10),
        row("B", 1, 0.5, 10),
        row("F", 1, 0.7, 100),
    ]
    assert gain_recovery(over)["gain_recovery_percent"] == pytest.approx(200.0)
    below = [
        row("A", 1, 0.4, 10),
        row("B", 1, 0.5, 10),
        row("F", 1, 0.7, 100),
    ]
    assert gain_recovery(below)["gain_recovery_percent"] == pytest.approx(-50.0)


def test_gain_recovery_undefined_and_unstable_policies():
    undefined = [
        row("A", 1, 0.6, 10),
        row("B", 1, 0.5, 10),
        row("F", 1, 0.4, 100),
    ]
    result = gain_recovery(undefined)
    assert result["status"] == "undefined"
    assert result["gain_recovery_percent"] is None
    assert "non-positive" in result["reason"]

    unstable = []
    for seed, (b_ev, f_ev) in enumerate(((0.4, 0.8), (0.6, 0.5), (0.5, 0.8)), 1):
        unstable.extend(
            [
                row("A", seed, 0.6, 10),
                row("B", seed, b_ev, 10),
                row("F", seed, f_ev, 100),
            ]
        )
    result = gain_recovery(unstable)
    assert result["status"] == "unstable"
    assert result["gain_recovery_percent"] is not None
    low, high = result["paired_f_minus_b_ci95"]
    assert low < 0 < high
    assert "includes zero" in result["reason"]


def test_seed_pairing_and_median_cost_policy():
    result = paired_cost_ratio(stable_rows())
    expected = np.median([100 * 61 / 1010, 100 * 62 / 1020, 100 * 63 / 1030])
    assert result["cost_ratio_percent"] == pytest.approx(expected)
    assert set(result["per_seed_cost_ratio_percent"]) == {"1", "2", "3"}

    mismatched = stable_rows()[:-1]
    with pytest.raises(MetricError, match="same seeds"):
        gain_recovery(mismatched)


def test_mfu_uses_custom_flops_and_goodput_uses_persisted_wall_time():
    mfu = compute_mfu(useful_flops=2.2e15, elapsed_seconds=1.0, peak_flops=4.4e15)
    assert mfu["mfu"] == pytest.approx(0.5)
    assert mfu["mfu_percent"] == pytest.approx(50.0)
    assert mfu["passes_50_percent"] is True

    goodput = compute_goodput(900, device_output_seconds=0.9, persisted_seconds=1.0)
    assert goodput["device_output_tokens_per_second"] == pytest.approx(1_000)
    assert goodput["persisted_tokens_per_second"] == pytest.approx(900)
    assert goodput["persisted_to_device_ratio"] == pytest.approx(0.9)
    assert goodput["passes_90_percent"] is True


def test_json_report_headline_and_end_to_end_amortization(tmp_path):
    harvest = {
        **compute_mfu(2.64e15, 1.0, 4.4e15),
        **compute_goodput(1_000, 1.0, 1.05),
        "chip_seconds": 5_000.0,
    }
    report = build_efficiency_report(
        stable_rows(),
        harvester=harvest,
        compile_seconds=123.0,
        checkpoint_seconds=45.0,
        evaluation_seconds=67.0,
    )
    assert report["status"] == "headline-ready"
    assert "recovers 50.0%" in report["headline"]
    assert "6.25% nominal" in report["headline"]
    assert render_headline(report) == report["headline"]
    assert report["compile_seconds_excluded"] == 123.0
    assert report["checkpoint_seconds_excluded"] == 45.0
    assert report["evaluation_seconds_excluded"] == 67.0
    assert set(report["secondary_end_to_end_cost"]) >= {
        "one_seed_ratio_percent",
        "three_seed_ratio_percent",
    }

    path = write_report_json(tmp_path / "report.json", report)
    loaded = json.loads(path.read_text())
    assert loaded == report
    assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()


def test_undefined_report_does_not_emit_a_misleading_gain_headline():
    values = [
        row("A", 1, 0.6, 10),
        row("B", 1, 0.5, 10),
        row("F", 1, 0.4, 100),
    ]
    report = build_efficiency_report(values)
    assert report["status"] == "undefined"
    assert report["primary"]["quality"]["gain_recovery_percent"] is None
    assert report["headline"].startswith("Reconstruction gain recovery is undefined")


def test_metrics_sink_accepts_step_or_event_record(tmp_path):
    sink = MetricsSink(tmp_path)
    sink.log(7, {"loss": np.float32(0.5)})
    sink.log({"event": "checkpoint", "tokens": np.int64(10_000)})
    lines = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert lines == [
        {"step": 7, "loss": 0.5},
        {"event": "checkpoint", "tokens": 10_000},
    ]
    with pytest.raises(MetricError, match="non-finite"):
        sink.log({"loss": float("nan")})
