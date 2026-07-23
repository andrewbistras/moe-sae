from __future__ import annotations

from fractions import Fraction

import pytest

from moe_sae.r1.prefix_benchmark import (
    MINIMUM_MEASUREMENT_STEPS,
    PackedStepShape,
    PrefixBenchmarkError,
    PrefixBenchmarkResult,
    TARGET_MFU,
    V6E64_BF16_PEAK_FLOPS_PER_SECOND,
    instrumentation_delta_percentage_points,
    maximum_step_time_for_target_ns,
    reduce_max_durations_across_jax_processes,
    run_synchronized_prefix_benchmark,
)


class FakeClock:
    def __init__(self, duration_ns: int) -> None:
        self.duration_ns = duration_ns
        self.now = 1_000_000
        self.start = True

    def __call__(self) -> int:
        if self.start:
            self.start = False
            return self.now
        self.start = True
        self.now += self.duration_ns
        return self.now


def _result(duration_ns: int) -> PrefixBenchmarkResult:
    shape = PackedStepShape.full_sequences()
    return PrefixBenchmarkResult(
        warmup_steps=5,
        measurement_steps=500,
        valid_tokens_per_step=shape.valid_tokens,
        useful_flops_per_step=shape.useful_flops,
        peak_flops_per_second=V6E64_BF16_PEAK_FLOPS_PER_SECOND,
        durations_ns=(duration_ns,) * 500,
    )


def test_full_b128_target_is_exact() -> None:
    shape = PackedStepShape.full_sequences()
    assert shape.valid_tokens == 128 * 1_024
    assert shape.useful_flops == 4_900_569_495_896_064
    ceiling = maximum_step_time_for_target_ns(shape)
    assert ceiling == Fraction(
        shape.useful_flops * 1_000_000_000,
        V6E64_BF16_PEAK_FLOPS_PER_SECOND,
    ) / TARGET_MFU
    assert float(ceiling / 1_000_000_000) == pytest.approx(
        0.1635511947827246
    )


def test_measurement_excludes_warmup_and_includes_device_completion() -> None:
    calls: list[tuple[str, int]] = []

    def step(value: int) -> int:
        calls.append(("step", value))
        return value + 1

    def synchronize(value: int) -> None:
        calls.append(("sync", value))

    result = run_synchronized_prefix_benchmark(
        step,
        9,
        packed_shape=PackedStepShape.full_sequences(),
        synchronize=synchronize,
        warmup_steps=5,
        measurement_steps=MINIMUM_MEASUREMENT_STEPS,
        clock_ns=FakeClock(160_000_000),
    )

    assert len(calls) == 2 * (5 + MINIMUM_MEASUREMENT_STEPS)
    assert calls[:4] == [
        ("step", 9),
        ("sync", 10),
        ("step", 9),
        ("sync", 10),
    ]
    assert result.durations_ns == (160_000_000,) * MINIMUM_MEASUREMENT_STEPS
    assert result.total_duration_ns == 80_000_000_000
    assert result.passes_51_percent is True


def test_slowest_process_duration_is_authoritative() -> None:
    observed: list[tuple[int, ...]] = []

    def reduce_max(local: tuple[int, ...]) -> tuple[int, ...]:
        observed.append(local)
        return tuple(value + 7_000_000 for value in local)

    result = run_synchronized_prefix_benchmark(
        lambda value: value,
        object(),
        packed_shape=PackedStepShape.full_sequences(),
        synchronize=lambda _value: None,
        clock_ns=FakeClock(160_000_000),
        reduce_max_durations_ns=reduce_max,
    )
    assert observed == [(160_000_000,) * 500]
    assert result.durations_ns == (167_000_000,) * 500
    assert result.passes_51_percent is False


def test_packed_segments_reduce_only_exact_attention_work() -> None:
    full = PackedStepShape.full_sequences()
    split = PackedStepShape(((512, 512),) * 128)
    padded = PackedStepShape(((900,),) * 128)
    assert split.valid_tokens == full.valid_tokens
    assert split.useful_flops < full.useful_flops
    assert padded.valid_tokens == 115_200
    assert padded.useful_flops < full.useful_flops


def test_instrumentation_delta_uses_identical_workload() -> None:
    baseline = _result(160_000_000)
    instrumented = _result(165_000_000)
    delta = instrumentation_delta_percentage_points(baseline, instrumented)
    assert delta > 0
    assert float(delta) == pytest.approx(
        100
        * float(baseline.aggregate_mfu - instrumented.aggregate_mfu)
    )

    with pytest.raises(PrefixBenchmarkError, match="identical"):
        instrumentation_delta_percentage_points(
            baseline,
            PrefixBenchmarkResult(
                warmup_steps=5,
                measurement_steps=501,
                valid_tokens_per_step=baseline.valid_tokens_per_step,
                useful_flops_per_step=baseline.useful_flops_per_step,
                peak_flops_per_second=baseline.peak_flops_per_second,
                durations_ns=(165_000_000,) * 501,
            ),
        )


def test_invalid_measurement_inputs_fail_before_work() -> None:
    with pytest.raises(PrefixBenchmarkError, match="128"):
        PackedStepShape(((1_024,),))
    with pytest.raises(PrefixBenchmarkError, match="exceeds"):
        PackedStepShape(((1_025,),) * 128)
    with pytest.raises(PrefixBenchmarkError, match="at least 500"):
        run_synchronized_prefix_benchmark(
            lambda value: value,
            object(),
            packed_shape=PackedStepShape.full_sequences(),
            synchronize=lambda _value: None,
            measurement_steps=499,
        )
    with pytest.raises(PrefixBenchmarkError, match="no smaller"):
        run_synchronized_prefix_benchmark(
            lambda value: value,
            object(),
            packed_shape=PackedStepShape.full_sequences(),
            synchronize=lambda _value: None,
            clock_ns=FakeClock(10),
            reduce_max_durations_ns=lambda durations: (9,) * len(durations),
        )


def test_single_process_duration_reducer_preserves_full_uint64_words() -> None:
    durations = (1, (1 << 32) + 17, (7 << 32) + 5)
    assert reduce_max_durations_across_jax_processes(durations) == durations

    with pytest.raises(PrefixBenchmarkError, match="non-empty"):
        reduce_max_durations_across_jax_processes(())
