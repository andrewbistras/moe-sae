"""Reconstruction, efficiency, MFU, and report metrics for Gemma 4 v1.

The primary quality comparison uses one common held-out target and one common
SST for A, B, and F.  The headline quantity is intentionally *not* clipped::

    100 * (mean(EV_A) - mean(EV_B)) / (mean(EV_F) - mean(EV_B))

It is undefined when the mean F-minus-B gain is non-positive and exploratory
when a paired 95% Student-t interval for F-minus-B contains zero.  Cost is the
median of paired, same-seed A/F chip-second ratios.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPORT_SCHEMA_VERSION = "gemma4-router-efficiency-report-v1"
V4_32_PEAK_BF16_FLOPS = 4.4e15
NOMINAL_CANDIDATE_EXPOSURE_PERCENT = 6.25


class MetricError(ValueError):
    """A metric input is incomplete, non-finite, or mathematically invalid."""


def _finite_float(name: str, value: Any, *, nonnegative: bool = False) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(out):
        raise MetricError(f"{name} must be finite, got {out}")
    if nonnegative and out < 0:
        raise MetricError(f"{name} must be nonnegative, got {out}")
    return out


def reconstruction_metrics(
    target: Any,
    reconstruction: Any,
    mean: Any,
) -> dict[str, float | int]:
    """Compute SSE, SST, NMSE, and EV with FP64 accumulation.

    ``mean`` is explicit so all conditions can use exactly the same held-out
    centering convention.  NMSE is ``SSE/SST`` and EV is ``1-NMSE``.  The
    target may have arbitrary leading dimensions; its final dimension must
    match ``mean``.
    """
    target64 = np.asarray(target, dtype=np.float64)
    reconstruction64 = np.asarray(reconstruction, dtype=np.float64)
    mean64 = np.asarray(mean, dtype=np.float64)
    if target64.shape != reconstruction64.shape:
        raise MetricError(
            f"target and reconstruction shapes differ: "
            f"{target64.shape} vs {reconstruction64.shape}"
        )
    if target64.ndim < 2 or target64.shape[-1] == 0:
        raise MetricError(f"target must have shape [...,D], got {target64.shape}")
    if mean64.shape not in {(target64.shape[-1],), (1, target64.shape[-1])}:
        raise MetricError(
            f"mean must have shape [{target64.shape[-1]}], got {mean64.shape}"
        )
    if not (
        np.isfinite(target64).all()
        and np.isfinite(reconstruction64).all()
        and np.isfinite(mean64).all()
    ):
        raise MetricError("reconstruction inputs must all be finite")
    error = target64 - reconstruction64
    centered = target64 - mean64.reshape((1,) * (target64.ndim - 1) + (-1,))
    sse = float(np.square(error).sum(dtype=np.float64))
    sst = float(np.square(centered).sum(dtype=np.float64))
    if sst <= 0:
        raise MetricError("SST must be positive; EV/NMSE are undefined")
    nmse = sse / sst
    return {
        "tokens": int(np.prod(target64.shape[:-1], dtype=np.int64)),
        "sse": sse,
        "sst": sst,
        "nmse": nmse,
        "ev": 1.0 - nmse,
    }


class ReconstructionAccumulator:
    """Streaming, FP64 reconstruction metrics with exact-L0 diagnostics."""

    def __init__(
        self,
        mean: Any,
        *,
        expected_l0: int = 64,
        feature_count: int | None = None,
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        if (
            self.mean.ndim != 1
            or self.mean.size == 0
            or not np.isfinite(self.mean).all()
        ):
            raise MetricError("mean must be a finite nonempty vector")
        if expected_l0 <= 0:
            raise MetricError("expected_l0 must be positive")
        if feature_count is not None and feature_count <= 0:
            raise MetricError("feature_count must be positive")
        self.expected_l0 = int(expected_l0)
        self.feature_count = feature_count
        self.sse = 0.0
        self.sst = 0.0
        self.tokens = 0
        self.l0_sum = 0
        self._l0_tokens = 0
        self.l0_min: int | None = None
        self.l0_max: int | None = None
        self._fired = (
            np.zeros(feature_count, dtype=np.bool_)
            if feature_count is not None
            else None
        )
        self._fire_batches = 0

    def update(
        self,
        target: Any,
        reconstruction: Any,
        *,
        active_counts: Any | None = None,
        fired_latent_ids: Any | None = None,
    ) -> None:
        target64 = np.asarray(target, dtype=np.float64)
        reconstruction64 = np.asarray(reconstruction, dtype=np.float64)
        if target64.shape != reconstruction64.shape:
            raise MetricError("target and reconstruction shapes differ")
        if target64.ndim != 2 or target64.shape[1] != self.mean.size:
            raise MetricError(
                f"streaming batches must have shape [N,{self.mean.size}], got {target64.shape}"
            )
        if not np.isfinite(target64).all() or not np.isfinite(reconstruction64).all():
            raise MetricError("reconstruction inputs must be finite")
        self.sse += float(np.square(target64 - reconstruction64).sum(dtype=np.float64))
        self.sst += float(np.square(target64 - self.mean).sum(dtype=np.float64))
        self.tokens += int(target64.shape[0])

        if active_counts is not None:
            counts = np.asarray(active_counts)
            if counts.shape != (target64.shape[0],) or not np.issubdtype(
                counts.dtype, np.integer
            ):
                raise MetricError("active_counts must be an integer [N] vector")
            if np.any(counts < 0):
                raise MetricError("active_counts must be nonnegative")
            batch_min = int(counts.min()) if counts.size else self.expected_l0
            batch_max = int(counts.max()) if counts.size else self.expected_l0
            self.l0_min = (
                batch_min if self.l0_min is None else min(self.l0_min, batch_min)
            )
            self.l0_max = (
                batch_max if self.l0_max is None else max(self.l0_max, batch_max)
            )
            self.l0_sum += int(counts.sum(dtype=np.int64))
            self._l0_tokens += int(counts.size)
        if fired_latent_ids is not None:
            if self._fired is None:
                raise MetricError("feature_count is required to track dead features")
            ids = np.asarray(fired_latent_ids)
            if not np.issubdtype(ids.dtype, np.integer):
                raise MetricError("fired_latent_ids must be integers")
            if np.any(ids < 0) or np.any(ids >= self._fired.size):
                raise MetricError("fired_latent_ids are outside the dictionary")
            self._fired[ids] = True
            self._fire_batches += 1

    def finalize(self) -> dict[str, float | int | bool | None]:
        if self.tokens == 0 or self.sst <= 0:
            raise MetricError(
                "cannot finalize empty or zero-SST reconstruction metrics"
            )
        nmse = self.sse / self.sst
        l0 = self.l0_sum / self._l0_tokens if self.l0_min is not None else None
        exact_l0 = (
            self.l0_min == self.expected_l0 == self.l0_max
            if self.l0_min is not None
            else None
        )
        dead_rate = (
            float(1.0 - self._fired.mean())
            if self._fired is not None and self._fire_batches > 0
            else None
        )
        return {
            "tokens": self.tokens,
            "sse": self.sse,
            "sst": self.sst,
            "nmse": nmse,
            "ev": 1.0 - nmse,
            "l0": l0,
            "l0_min": self.l0_min,
            "l0_max": self.l0_max,
            "exact_l0": exact_l0,
            "dead_rate": dead_rate,
        }


@dataclasses.dataclass(frozen=True)
class SeedEvaluation:
    """Final held-out and marginal training-cost record for one seed/run."""

    condition: str
    seed: int
    ev: float
    nmse: float
    l0: float
    dead_rate: float
    chip_seconds: float
    wall_seconds: float | None = None
    tokens_per_second: float | None = None

    @classmethod
    def from_value(
        cls, value: "SeedEvaluation | Mapping[str, Any]"
    ) -> "SeedEvaluation":
        if isinstance(value, cls):
            result = value
        else:
            required = {
                "condition",
                "seed",
                "ev",
                "nmse",
                "l0",
                "dead_rate",
                "chip_seconds",
            }
            missing = sorted(required - value.keys())
            if missing:
                raise MetricError(f"evaluation missing fields: {', '.join(missing)}")
            raw_seed = value["seed"]
            if (
                isinstance(raw_seed, bool)
                or not isinstance(raw_seed, (int, np.integer))
                or int(raw_seed) < 0
            ):
                raise MetricError("seed must be a nonnegative integer")
            result = cls(
                condition=str(value["condition"]),
                seed=int(raw_seed),
                ev=float(value["ev"]),
                nmse=float(value["nmse"]),
                l0=float(value["l0"]),
                dead_rate=float(value["dead_rate"]),
                chip_seconds=float(value["chip_seconds"]),
                wall_seconds=(
                    None
                    if value.get("wall_seconds") is None
                    else float(value["wall_seconds"])
                ),
                tokens_per_second=(
                    None
                    if value.get("tokens_per_second") is None
                    else float(value["tokens_per_second"])
                ),
            )
        if result.condition not in {"A", "B", "F"}:
            raise MetricError(f"condition must be A, B, or F, got {result.condition!r}")
        if isinstance(result.seed, bool) or result.seed < 0:
            raise MetricError("seed must be a nonnegative integer")
        for name in ("ev", "nmse", "l0", "dead_rate", "chip_seconds"):
            _finite_float(name, getattr(result, name), nonnegative=name != "ev")
        if result.chip_seconds <= 0:
            raise MetricError("chip_seconds must be positive")
        if result.wall_seconds is not None:
            if (
                _finite_float("wall_seconds", result.wall_seconds, nonnegative=True)
                <= 0
            ):
                raise MetricError("wall_seconds must be positive when supplied")
        if result.tokens_per_second is not None:
            if (
                _finite_float(
                    "tokens_per_second", result.tokens_per_second, nonnegative=True
                )
                <= 0
            ):
                raise MetricError("tokens_per_second must be positive when supplied")
        if not 0 <= result.dead_rate <= 1:
            raise MetricError("dead_rate must be in [0,1]")
        if abs(result.l0 - 64.0) > 1e-9:
            raise MetricError("held-out L0 must be exactly 64 for the v1 report")
        if abs((1.0 - result.nmse) - result.ev) > 1e-6:
            raise MetricError("ev must equal 1 - nmse under the common SST")
        return result


def _normalized_evaluations(
    evaluations: Iterable[SeedEvaluation | Mapping[str, Any]],
) -> list[SeedEvaluation]:
    values = [SeedEvaluation.from_value(value) for value in evaluations]
    if not values:
        raise MetricError("at least one seed evaluation is required")
    keys = [(value.condition, value.seed) for value in values]
    if len(keys) != len(set(keys)):
        raise MetricError("duplicate condition/seed evaluation")
    seeds_by_condition = {
        condition: {value.seed for value in values if value.condition == condition}
        for condition in ("A", "B", "F")
    }
    if any(not seeds for seeds in seeds_by_condition.values()):
        raise MetricError("evaluations must contain A, B, and F")
    if len({frozenset(seeds) for seeds in seeds_by_condition.values()}) != 1:
        raise MetricError("A, B, and F must use exactly the same seeds")
    return values


# Two-sided 95% Student-t critical values, indexed by degrees of freedom.
# Values beyond 30 use the normal limit; production has df=2 (three seeds).
_T95 = (
    math.nan,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def _paired_ci95(values: Sequence[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    df = len(values) - 1
    critical = _T95[df] if df < len(_T95) else 1.96
    half_width = critical * standard_error
    return mean - half_width, mean + half_width


def gain_recovery(
    evaluations: Iterable[SeedEvaluation | Mapping[str, Any]],
) -> dict[str, Any]:
    """Return raw G and its required undefined/unstable policy."""
    values = _normalized_evaluations(evaluations)
    by_key = {(value.condition, value.seed): value for value in values}
    seeds = sorted({value.seed for value in values})
    means = {
        condition: statistics.fmean(by_key[(condition, seed)].ev for seed in seeds)
        for condition in ("A", "B", "F")
    }
    paired_f_minus_b = [
        by_key[("F", seed)].ev - by_key[("B", seed)].ev for seed in seeds
    ]
    denominator = means["F"] - means["B"]
    ci = _paired_ci95(paired_f_minus_b)
    result: dict[str, Any] = {
        "mean_ev": means,
        "mean_f_minus_b": denominator,
        "paired_f_minus_b": paired_f_minus_b,
        "paired_f_minus_b_ci95": list(ci) if ci is not None else None,
        "gain_recovery_percent": None,
        "status": "undefined",
        "headline_eligible": False,
        "reason": None,
    }
    if denominator <= 0:
        result["reason"] = "mean F-minus-B held-out EV is non-positive"
        return result
    result["gain_recovery_percent"] = 100.0 * (means["A"] - means["B"]) / denominator
    if ci is None:
        result.update(
            status="unstable",
            reason="fewer than two paired seeds; a 95% interval is unavailable",
        )
    elif ci[0] <= 0 <= ci[1]:
        result.update(
            status="unstable",
            reason="paired 95% F-minus-B EV interval includes zero",
        )
    else:
        result.update(status="stable", headline_eligible=True, reason=None)
    return result


def paired_cost_ratio(
    evaluations: Iterable[SeedEvaluation | Mapping[str, Any]],
) -> dict[str, Any]:
    """Median paired A/F marginal chip-second ratio, expressed as percent."""
    values = _normalized_evaluations(evaluations)
    by_key = {(value.condition, value.seed): value for value in values}
    seeds = sorted({value.seed for value in values})
    ratios = {
        str(seed): 100.0
        * by_key[("A", seed)].chip_seconds
        / by_key[("F", seed)].chip_seconds
        for seed in seeds
    }
    return {
        "cost_ratio_percent": statistics.median(ratios.values()),
        "per_seed_cost_ratio_percent": ratios,
        "basis": (
            "paired marginal SAE chip-seconds, including A's measured routing-capacity "
            "prepass; compile/checkpoint/eval excluded"
        ),
    }


def compute_mfu(
    useful_flops: float,
    elapsed_seconds: float,
    peak_flops: float = V4_32_PEAK_BF16_FLOPS,
) -> dict[str, float | bool]:
    """Compute MFU from caller-supplied useful-prefix FLOPs and wall time."""
    useful = _finite_float("useful_flops", useful_flops, nonnegative=True)
    elapsed = _finite_float("elapsed_seconds", elapsed_seconds, nonnegative=True)
    peak = _finite_float("peak_flops", peak_flops, nonnegative=True)
    if useful <= 0 or elapsed <= 0 or peak <= 0:
        raise MetricError(
            "useful_flops, elapsed_seconds, and peak_flops must be positive"
        )
    mfu = useful / (elapsed * peak)
    return {
        "useful_flops": useful,
        "elapsed_seconds": elapsed,
        "peak_flops": peak,
        "mfu": mfu,
        "mfu_percent": 100.0 * mfu,
        "passes_50_percent": mfu >= 0.5,
    }


def compute_goodput(
    tokens: int,
    device_output_seconds: float,
    persisted_seconds: float,
) -> dict[str, float | int | bool]:
    """Compare device-output throughput with durable persisted throughput."""
    if isinstance(tokens, bool) or int(tokens) != tokens or tokens <= 0:
        raise MetricError("tokens must be a positive integer")
    device_seconds = _finite_float(
        "device_output_seconds", device_output_seconds, nonnegative=True
    )
    durable_seconds = _finite_float(
        "persisted_seconds", persisted_seconds, nonnegative=True
    )
    if device_seconds <= 0 or durable_seconds <= 0:
        raise MetricError("goodput durations must be positive")
    device_rate = int(tokens) / device_seconds
    persisted_rate = int(tokens) / durable_seconds
    ratio = persisted_rate / device_rate
    return {
        "tokens": int(tokens),
        "device_output_seconds": device_seconds,
        "persisted_seconds": durable_seconds,
        "device_output_tokens_per_second": device_rate,
        "persisted_tokens_per_second": persisted_rate,
        "persisted_to_device_ratio": ratio,
        "passes_90_percent": ratio >= 0.9,
    }


compute_persisted_goodput = compute_goodput


def _condition_summaries(values: list[SeedEvaluation]) -> dict[str, Any]:
    dictionary_features = {"A": 131_072, "B": 8_192, "F": 131_072}
    candidates_per_token = {"A": 8_192, "B": 8_192, "F": 131_072}
    parameter_counts = {"A": 738_328_576, "B": 46_145_536, "F": 738_328_576}
    summaries: dict[str, Any] = {}
    for condition in ("A", "B", "F"):
        rows = sorted(
            (value for value in values if value.condition == condition),
            key=lambda value: value.seed,
        )
        summaries[condition] = {
            "dictionary_features": dictionary_features[condition],
            "parameter_count": parameter_counts[condition],
            "candidates_per_token": candidates_per_token[condition],
            "candidate_exposure_percent_of_f": 100.0
            * candidates_per_token[condition]
            / 131_072,
            "mean_ev": statistics.fmean(row.ev for row in rows),
            "mean_nmse": statistics.fmean(row.nmse for row in rows),
            "mean_l0": statistics.fmean(row.l0 for row in rows),
            "exact_l0": all(row.l0 == 64.0 for row in rows),
            "mean_dead_rate": statistics.fmean(row.dead_rate for row in rows),
            "median_chip_seconds": statistics.median(row.chip_seconds for row in rows),
            "seeds": [
                {
                    **dataclasses.asdict(row),
                    "exact_l0": row.l0 == 64.0,
                    "dictionary_features": dictionary_features[condition],
                    "parameter_count": parameter_counts[condition],
                    "candidates_per_token": candidates_per_token[condition],
                    "candidate_exposure_percent_of_f": 100.0
                    * candidates_per_token[condition]
                    / 131_072,
                }
                for row in rows
            ],
        }
    return summaries


def render_headline(report: Mapping[str, Any]) -> str:
    """Render the canonical human-readable result from a report dictionary."""
    primary = report["primary"]
    quality = primary["quality"]
    cost = float(primary["cost"]["cost_ratio_percent"])
    exposure = float(primary["nominal_candidate_exposure_percent"])
    gain = quality.get("gain_recovery_percent")
    if gain is None:
        return (
            "Reconstruction gain recovery is undefined because mean F-minus-B "
            f"held-out EV is non-positive; A costs {cost:.1f}% of F's measured "
            f"marginal SAE training cost ({exposure:.2f}% nominal candidate exposure)."
        )
    prefix = "" if quality.get("status") == "stable" else "Exploratory: "
    headline = (
        f"{prefix}router-conditioned A recovers {float(gain):.1f}% of the F-over-B "
        f"held-out reconstruction gain at {cost:.1f}% of F's measured marginal "
        f"SAE training cost ({exposure:.2f}% nominal candidate exposure)."
    )
    if quality.get("status") != "stable" and quality.get("reason"):
        headline += f" {quality['reason'].capitalize()}."
    return headline


def build_efficiency_report(
    evaluations: Iterable[SeedEvaluation | Mapping[str, Any]],
    *,
    harvester: Mapping[str, Any] | None = None,
    compile_seconds: float = 0.0,
    checkpoint_seconds: float = 0.0,
    evaluation_seconds: float = 0.0,
    nominal_candidate_exposure_percent: float = NOMINAL_CANDIDATE_EXPOSURE_PERCENT,
) -> dict[str, Any]:
    """Build the complete JSON-safe A/B/F efficiency report.

    If ``harvester`` contains ``chip_seconds``, secondary end-to-end A/F cost
    ratios are included with that shared cost amortized over one and three
    seeds.  The harvester mapping can directly combine :func:`compute_mfu`
    and :func:`compute_goodput` outputs.
    """
    values = _normalized_evaluations(evaluations)
    quality = gain_recovery(values)
    cost = paired_cost_ratio(values)
    compile_value = _finite_float("compile_seconds", compile_seconds, nonnegative=True)
    checkpoint_value = _finite_float(
        "checkpoint_seconds", checkpoint_seconds, nonnegative=True
    )
    evaluation_value = _finite_float(
        "evaluation_seconds", evaluation_seconds, nonnegative=True
    )
    exposure = _finite_float(
        "nominal_candidate_exposure_percent",
        nominal_candidate_exposure_percent,
        nonnegative=True,
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": (
            "headline-ready" if quality["status"] == "stable" else quality["status"]
        ),
        "primary": {
            "quality": quality,
            "cost": cost,
            "nominal_candidate_exposure_percent": exposure,
        },
        "conditions": _condition_summaries(values),
        "harvester": _jsonable(dict(harvester or {})),
        "compile_seconds_excluded": compile_value,
        "checkpoint_seconds_excluded": checkpoint_value,
        "evaluation_seconds_excluded": evaluation_value,
    }
    harvest_chip_seconds = report["harvester"].get("chip_seconds")
    if harvest_chip_seconds is not None:
        harvest_cost = _finite_float(
            "harvester.chip_seconds", harvest_chip_seconds, nonnegative=True
        )
        med_a = report["conditions"]["A"]["median_chip_seconds"]
        med_f = report["conditions"]["F"]["median_chip_seconds"]
        sum_a = sum(row.chip_seconds for row in values if row.condition == "A")
        sum_f = sum(row.chip_seconds for row in values if row.condition == "F")
        report["secondary_end_to_end_cost"] = {
            "one_seed_ratio_percent": 100.0
            * (harvest_cost + med_a)
            / (harvest_cost + med_f),
            "three_seed_ratio_percent": 100.0
            * (harvest_cost + sum_a)
            / (harvest_cost + sum_f),
            "shared_harvest_chip_seconds": harvest_cost,
            "one_seed_basis": "shared harvest plus median measured seed cost",
            "three_seed_basis": "shared harvest plus measured sum across all three seeds",
        }
    report["headline"] = render_headline(report)
    _ensure_json_safe(report)
    return report


def _ensure_json_safe(value: Any, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _ensure_json_safe(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _ensure_json_safe(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise MetricError(f"{path} is non-finite")
    elif isinstance(value, np.ndarray):
        _ensure_json_safe(value.tolist(), path)


def _jsonable(value: Any) -> Any:
    """Convert NumPy/dataclass telemetry to strict JSON-native values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_report_json(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Atomically write a validated report as strict JSON (no NaN/Infinity)."""
    path = Path(path)
    payload = _jsonable(report)
    _ensure_json_safe(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


class MetricsSink:
    """Minimal append-only JSONL sink for training and operational events."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        self.path = path if path.suffix == ".jsonl" else path / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        step_or_record: int | Mapping[str, Any],
        scalars: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(step_or_record, Mapping):
            if scalars is not None:
                raise MetricError("scalars must be omitted when logging a record")
            record = dict(step_or_record)
        else:
            record = {"step": int(step_or_record), **dict(scalars or {})}
        record = _jsonable(record)
        _ensure_json_safe(record, "metrics record")
        encoded = (json.dumps(record, allow_nan=False) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, encoded)
        finally:
            os.close(fd)


__all__ = [
    "MetricError",
    "MetricsSink",
    "NOMINAL_CANDIDATE_EXPOSURE_PERCENT",
    "REPORT_SCHEMA_VERSION",
    "ReconstructionAccumulator",
    "SeedEvaluation",
    "V4_32_PEAK_BF16_FLOPS",
    "build_efficiency_report",
    "compute_goodput",
    "compute_mfu",
    "compute_persisted_goodput",
    "gain_recovery",
    "paired_cost_ratio",
    "reconstruction_metrics",
    "render_headline",
    "write_report_json",
]
