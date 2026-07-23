"""Small executable measurement core for the DeepSeek-R1 B128 prefix.

This module is intentionally not another admission protocol.  It is the
worker-side stopwatch used after an already-compiled prefix callable exists.
Compilation, warmup, evidence serialization, and telemetry reduction are kept
outside measured intervals.  Every measured interval includes the device
completion barrier for that step.

JAX is not imported at module import time.  The benchmark can consequently be
tested on a controller, while the real worker supplies JAX-backed ``step`` and
``synchronize`` callables after runtime admission.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
import math
import time
from typing import Any

from .flops import deepseek_r1_prefix_useful_flops


GLOBAL_SEQUENCE_BATCH = 128
SEQUENCE_LENGTH = 1_024
DEVICE_COUNT = 64

# Google publishes 918 TFLOP/s BF16 per v6e chip.  Use that allocation-sized
# figure directly instead of quartering the rounded 256-chip Pod figure; this
# is both more precise and very slightly more conservative.
V6E_BF16_PEAK_FLOPS_PER_CHIP_SECOND = 918_000_000_000_000
V6E64_BF16_PEAK_FLOPS_PER_SECOND = (
    V6E_BF16_PEAK_FLOPS_PER_CHIP_SECOND * DEVICE_COUNT
)
TARGET_MFU = Fraction(51, 100)
MINIMUM_WARMUP_STEPS = 5
MINIMUM_MEASUREMENT_STEPS = 500


class PrefixBenchmarkError(RuntimeError):
    """The executable measurement contract was violated."""


@dataclass(frozen=True, slots=True)
class PackedStepShape:
    """Exact valid-segment geometry for one static B128×1024 input."""

    segments_by_sequence: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            type(self.segments_by_sequence) is not tuple
            or len(self.segments_by_sequence) != GLOBAL_SEQUENCE_BATCH
        ):
            raise PrefixBenchmarkError(
                "packed geometry must contain exactly 128 sequences"
            )
        for sequence in self.segments_by_sequence:
            if type(sequence) is not tuple:
                raise PrefixBenchmarkError("each sequence geometry must be a tuple")
            if not sequence:
                raise PrefixBenchmarkError(
                    "every measured sequence must contain valid tokens"
                )
            if any(
                type(length) is not int or length <= 0 for length in sequence
            ):
                raise PrefixBenchmarkError(
                    "packed segment lengths must be positive integers"
                )
            if sum(sequence) > SEQUENCE_LENGTH:
                raise PrefixBenchmarkError(
                    "packed sequence exceeds the static length 1024"
                )

    @classmethod
    def full_sequences(cls) -> PackedStepShape:
        return cls(((SEQUENCE_LENGTH,),) * GLOBAL_SEQUENCE_BATCH)

    @property
    def valid_tokens(self) -> int:
        return sum(
            length
            for sequence in self.segments_by_sequence
            for length in sequence
        )

    @property
    def useful_flops(self) -> int:
        return deepseek_r1_prefix_useful_flops(
            segment_lengths=(
                length
                for sequence in self.segments_by_sequence
                for length in sequence
            )
        )


@dataclass(frozen=True, slots=True)
class PrefixBenchmarkResult:
    """Raw synchronized durations and their direct aggregate reduction."""

    warmup_steps: int
    measurement_steps: int
    valid_tokens_per_step: int
    useful_flops_per_step: int
    peak_flops_per_second: int
    durations_ns: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.warmup_steps < MINIMUM_WARMUP_STEPS:
            raise PrefixBenchmarkError("at least five warmup steps are required")
        if self.measurement_steps < MINIMUM_MEASUREMENT_STEPS:
            raise PrefixBenchmarkError(
                "at least 500 synchronized measurement steps are required"
            )
        if len(self.durations_ns) != self.measurement_steps:
            raise PrefixBenchmarkError(
                "duration count differs from measurement step count"
            )
        for name, value in (
            ("valid_tokens_per_step", self.valid_tokens_per_step),
            ("useful_flops_per_step", self.useful_flops_per_step),
            ("peak_flops_per_second", self.peak_flops_per_second),
        ):
            if type(value) is not int or value <= 0:
                raise PrefixBenchmarkError(f"{name} must be a positive integer")
        if any(type(value) is not int or value <= 0 for value in self.durations_ns):
            raise PrefixBenchmarkError(
                "every synchronized step duration must be positive"
            )

    @property
    def total_duration_ns(self) -> int:
        return sum(self.durations_ns)

    @property
    def total_valid_tokens(self) -> int:
        return self.valid_tokens_per_step * self.measurement_steps

    @property
    def total_useful_flops(self) -> int:
        return self.useful_flops_per_step * self.measurement_steps

    @property
    def aggregate_mfu(self) -> Fraction:
        return Fraction(
            self.total_useful_flops * 1_000_000_000,
            self.peak_flops_per_second * self.total_duration_ns,
        )

    @property
    def tokens_per_second(self) -> Fraction:
        return Fraction(
            self.total_valid_tokens * 1_000_000_000,
            self.total_duration_ns,
        )

    @property
    def median_step_ns(self) -> int:
        # statistics.median returns float for an even-sized integer sample.
        ordered = sorted(self.durations_ns)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) // 2

    @property
    def p95_step_ns(self) -> int:
        ordered = sorted(self.durations_ns)
        rank = math.ceil(0.95 * len(ordered))
        return ordered[rank - 1]

    @property
    def passes_51_percent(self) -> bool:
        return self.aggregate_mfu >= TARGET_MFU

    def as_dict(self) -> dict[str, Any]:
        mfu = self.aggregate_mfu
        rate = self.tokens_per_second
        return {
            "schema": "moe-sae.r1.prefix-benchmark-result.v1",
            "warmup_steps": self.warmup_steps,
            "measurement_steps": self.measurement_steps,
            "valid_tokens_per_step": self.valid_tokens_per_step,
            "useful_flops_per_step": self.useful_flops_per_step,
            "peak_flops_per_second": self.peak_flops_per_second,
            "durations_ns": list(self.durations_ns),
            "total_duration_ns": self.total_duration_ns,
            "total_valid_tokens": self.total_valid_tokens,
            "total_useful_flops": self.total_useful_flops,
            "aggregate_mfu": {
                "numerator": mfu.numerator,
                "denominator": mfu.denominator,
            },
            "tokens_per_second": {
                "numerator": rate.numerator,
                "denominator": rate.denominator,
            },
            "median_step_ns": self.median_step_ns,
            "p95_step_ns": self.p95_step_ns,
            "passes_51_percent": self.passes_51_percent,
        }


Step = Callable[[Any], Any]
Synchronize = Callable[[Any], None]
Clock = Callable[[], int]
DurationReducer = Callable[[tuple[int, ...]], tuple[int, ...]]


def _positive_count(value: int, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise PrefixBenchmarkError(f"{name} must be at least {minimum}")
    return value


def run_synchronized_prefix_benchmark(
    step: Step,
    resident_batch: Any,
    *,
    packed_shape: PackedStepShape,
    synchronize: Synchronize,
    warmup_steps: int = MINIMUM_WARMUP_STEPS,
    measurement_steps: int = MINIMUM_MEASUREMENT_STEPS,
    clock_ns: Clock = time.monotonic_ns,
    reduce_max_durations_ns: DurationReducer = lambda durations: durations,
) -> PrefixBenchmarkResult:
    """Measure one already-compiled, resident-input prefix callable.

    ``step`` must already be compiled.  Calling or lowering a compiler belongs
    before this function.  ``synchronize`` must block until every returned
    device output is ready.  In a 16-process job,
    ``reduce_max_durations_ns`` performs one post-window all-gather and returns
    the elementwise slowest process durations.  It runs after all timestamps,
    so host coordination is not charged to the prefix execution window.
    """

    if not callable(step) or not callable(synchronize) or not callable(clock_ns):
        raise PrefixBenchmarkError(
            "step, synchronize, and clock_ns must be callable"
        )
    if not callable(reduce_max_durations_ns):
        raise PrefixBenchmarkError("duration reducer must be callable")
    if type(packed_shape) is not PackedStepShape:
        raise PrefixBenchmarkError("packed_shape has the wrong type")
    warmups = _positive_count(
        warmup_steps, "warmup_steps", MINIMUM_WARMUP_STEPS
    )
    measurements = _positive_count(
        measurement_steps,
        "measurement_steps",
        MINIMUM_MEASUREMENT_STEPS,
    )

    # The first call may trigger lazy runtime initialization even after an
    # explicit compile.  All warmups complete before the first timestamp.
    for _ in range(warmups):
        synchronize(step(resident_batch))

    local_durations: list[int] = []
    for _ in range(measurements):
        started = clock_ns()
        output = step(resident_batch)
        synchronize(output)
        stopped = clock_ns()
        if type(started) is not int or type(stopped) is not int or stopped <= started:
            raise PrefixBenchmarkError("monotonic clock returned an invalid interval")
        local_durations.append(stopped - started)

    reduced = reduce_max_durations_ns(tuple(local_durations))
    if type(reduced) is not tuple or len(reduced) != measurements:
        raise PrefixBenchmarkError(
            "global duration reducer must return one tuple entry per step"
        )
    for local_duration, reduced_duration in zip(
        local_durations, reduced, strict=True
    ):
        if type(reduced_duration) is not int or reduced_duration < local_duration:
            raise PrefixBenchmarkError(
                "global duration must be an integer no smaller than local duration"
            )

    return PrefixBenchmarkResult(
        warmup_steps=warmups,
        measurement_steps=measurements,
        valid_tokens_per_step=packed_shape.valid_tokens,
        useful_flops_per_step=packed_shape.useful_flops,
        peak_flops_per_second=V6E64_BF16_PEAK_FLOPS_PER_SECOND,
        durations_ns=reduced,
    )


def maximum_step_time_for_target_ns(
    packed_shape: PackedStepShape,
    *,
    target_mfu: Fraction = TARGET_MFU,
) -> Fraction:
    """Return the exact per-step duration ceiling for a target MFU."""

    if type(packed_shape) is not PackedStepShape:
        raise PrefixBenchmarkError("packed_shape has the wrong type")
    if not isinstance(target_mfu, Fraction) or not 0 < target_mfu <= 1:
        raise PrefixBenchmarkError("target_mfu must be a Fraction in (0, 1]")
    return Fraction(
        packed_shape.useful_flops * 1_000_000_000,
        V6E64_BF16_PEAK_FLOPS_PER_SECOND,
    ) / target_mfu


def instrumentation_delta_percentage_points(
    baseline: PrefixBenchmarkResult,
    instrumented: PrefixBenchmarkResult,
) -> Fraction:
    """Return baseline minus instrumented aggregate MFU in percentage points."""

    if (
        type(baseline) is not PrefixBenchmarkResult
        or type(instrumented) is not PrefixBenchmarkResult
    ):
        raise PrefixBenchmarkError("instrumentation comparison requires two results")
    identity = (
        "measurement_steps",
        "valid_tokens_per_step",
        "useful_flops_per_step",
        "peak_flops_per_second",
    )
    if any(getattr(baseline, name) != getattr(instrumented, name) for name in identity):
        raise PrefixBenchmarkError(
            "instrumentation runs do not share an identical measurement workload"
        )
    return 100 * (baseline.aggregate_mfu - instrumented.aggregate_mfu)


def reduce_max_durations_across_jax_processes(
    local_durations_ns: tuple[int, ...],
) -> tuple[int, ...]:
    """Reduce measured durations after the timed window, without INT64 on TPU.

    JAX commonly runs with x64 disabled.  Each nanosecond duration is therefore
    split into two UINT32 words before one host collective, reconstructed on
    the host, and reduced elementwise.  No collective executes between a
    measured start/stop pair.
    """

    if (
        type(local_durations_ns) is not tuple
        or not local_durations_ns
        or any(type(value) is not int or value <= 0 for value in local_durations_ns)
    ):
        raise PrefixBenchmarkError(
            "local durations must be a non-empty tuple of positive integers"
        )
    try:
        import jax
        import numpy as np
        from jax.experimental import multihost_utils
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - TPU image
        raise PrefixBenchmarkError("JAX multi-host utilities are unavailable") from error

    words = np.asarray(
        [
            ((duration >> 32) & 0xFFFF_FFFF, duration & 0xFFFF_FFFF)
            for duration in local_durations_ns
        ],
        dtype=np.uint32,
    )
    gathered = np.asarray(
        multihost_utils.process_allgather(words, tiled=False),
        dtype=np.uint32,
    )
    process_count = jax.process_count()
    expected_single = words.shape
    expected_multi = (process_count, *words.shape)
    if process_count == 1 and gathered.shape == expected_single:
        gathered = gathered.reshape(expected_multi)
    if gathered.shape != expected_multi:
        raise PrefixBenchmarkError(
            "JAX duration all-gather returned an unexpected shape"
        )
    result: list[int] = []
    for step in range(len(local_durations_ns)):
        values = [
            (int(gathered[process, step, 0]) << 32)
            | int(gathered[process, step, 1])
            for process in range(process_count)
        ]
        result.append(max(values))
    return tuple(result)


__all__ = [
    "DEVICE_COUNT",
    "GLOBAL_SEQUENCE_BATCH",
    "MINIMUM_MEASUREMENT_STEPS",
    "MINIMUM_WARMUP_STEPS",
    "PackedStepShape",
    "PrefixBenchmarkError",
    "PrefixBenchmarkResult",
    "SEQUENCE_LENGTH",
    "TARGET_MFU",
    "V6E64_BF16_PEAK_FLOPS_PER_SECOND",
    "instrumentation_delta_percentage_points",
    "maximum_step_time_for_target_ns",
    "reduce_max_durations_across_jax_processes",
    "run_synchronized_prefix_benchmark",
]
