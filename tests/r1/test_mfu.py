from __future__ import annotations

from dataclasses import replace

import pytest

from moe_sae.r1.mfu import (
    FlopTermV1,
    MFUContractError,
    MFUWindowV1,
    UsefulFlopInventoryV1,
    V6E64_BF16_PEAK_FLOPS,
    qualify_mfu_windows,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _inventory() -> UsefulFlopInventoryV1:
    return UsefulFlopInventoryV1(
        phase="harvest_prefix",
        model_revision_sha256=SHA_A,
        tensor_inventory_sha256=SHA_B,
        hlo_sha256=SHA_C,
        terms=(
            FlopTermV1("attention", 10, "sum of exact packed-segment MLA matmuls"),
            FlopTermV1(
                "routed_experts", 20, "eight selected experts for every valid token"
            ),
        ),
        excluded_operations=("padding", "post_target_layers", "topk"),
    )


def _window(
    index: int, *, mfu: float = 0.52, phase: str = "harvest_prefix"
) -> MFUWindowV1:
    wall = 60.0
    return MFUWindowV1(
        window_id=f"window-{index}",
        phase=phase,  # type: ignore[arg-type]
        flop_inventory_sha256=SHA_A,
        hlo_sha256=SHA_B,
        profiler_sha256=SHA_C,
        step_start=index * 600,
        step_end=index * 600 + 500,
        useful_flops=int(mfu * V6E64_BF16_PEAK_FLOPS * wall),
        scientific_tokens=500 * 65_536,
        wall_seconds=wall,
        input_wait_seconds=2.4,
        real_tokens_per_second=100.0,
        synthetic_tokens_per_second=105.0,
        device_output_tokens_per_second=100.0,
        persisted_tokens_per_second=95.0 if phase == "harvest_prefix" else None,
        instrumentation_mfu_delta=0.01,
        p50_step_seconds=0.120,
        p95_step_seconds=0.126,
        median_host_step_seconds=0.120,
        slowest_host_step_seconds=0.123,
        routing_capacity_amplification=1.05,
        peak_hbm_bytes_per_chip=27,
        hbm_capacity_bytes_per_chip=32,
        recompilations=0,
        dropped_routed_tokens=0,
        capacity_overflows=0,
        finite_state=True,
    )


def test_inventory_is_content_addressed_and_explicit() -> None:
    inventory = _inventory()
    assert inventory.useful_flops_per_step == 30
    assert len(inventory.sha256) == 64
    assert inventory.sha256 == _inventory().sha256


def test_inventory_rejects_duplicate_or_unsorted_terms() -> None:
    with pytest.raises(MFUContractError, match="sorted"):
        replace(
            _inventory(),
            terms=(
                FlopTermV1("z", 1, "z term"),
                FlopTermV1("a", 1, "a term"),
            ),
        )
    with pytest.raises(MFUContractError, match="unique"):
        replace(
            _inventory(),
            terms=(
                FlopTermV1("same", 1, "first"),
                FlopTermV1("same", 2, "second"),
            ),
        )


def test_three_independent_windows_pass_the_nonwaivable_gate() -> None:
    result = qualify_mfu_windows(
        [_window(2), _window(0), _window(1)], phase="harvest_prefix"
    )
    assert result.aggregate_mfu == pytest.approx(0.52)
    assert result.minimum_window_mfu == pytest.approx(0.52)
    assert result.total_steps == 1_500
    assert result.window_ids == ("window-0", "window-1", "window-2")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"recompilations": 1}, "recompiled"),
        ({"dropped_routed_tokens": 1}, "dropped"),
        ({"capacity_overflows": 1}, "overflowed"),
        ({"finite_state": False}, "non-finite"),
        ({"input_wait_seconds": 3.0}, "input wait"),
        ({"real_tokens_per_second": 89.0}, "real/synthetic"),
        ({"persisted_tokens_per_second": 89.0}, "persistence"),
        ({"peak_hbm_bytes_per_chip": 28}, "HBM"),
        ({"instrumentation_mfu_delta": 0.021}, "instrumentation"),
        ({"p95_step_seconds": 0.133}, "step latency"),
        ({"slowest_host_step_seconds": 0.127}, "host straggler"),
        ({"routing_capacity_amplification": 1.101}, "routing amplification"),
    ],
)
def test_integrity_and_goodput_fail_closed(
    mutation: dict[str, object], message: str
) -> None:
    windows = [_window(0), replace(_window(1), **mutation), _window(2)]
    with pytest.raises(MFUContractError, match=message):
        qualify_mfu_windows(windows, phase="harvest_prefix")


def test_one_window_below_51_percent_fails_even_when_average_passes() -> None:
    windows = [_window(0, mfu=0.60), _window(1, mfu=0.509), _window(2, mfu=0.60)]
    with pytest.raises(MFUContractError, match="below 51.00%"):
        qualify_mfu_windows(windows, phase="harvest_prefix")


def test_more_than_physical_peak_is_rejected_as_an_accounting_error() -> None:
    windows = [_window(0), _window(1, mfu=1.01), _window(2)]
    with pytest.raises(MFUContractError, match="exceeds physical peak"):
        qualify_mfu_windows(windows, phase="harvest_prefix")


def test_threshold_cannot_be_weakened() -> None:
    with pytest.raises(MFUContractError, match="at least 0.51"):
        qualify_mfu_windows(
            [_window(0), _window(1), _window(2)],
            phase="harvest_prefix",
            minimum_mfu=0.50,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"minimum_windows": 2}, "minimum_windows"),
        ({"minimum_steps_per_window": 499}, "minimum_steps_per_window"),
        (
            {"minimum_wall_seconds_per_window": 59.999},
            "minimum_wall_seconds_per_window",
        ),
        ({"maximum_input_wait_fraction": 0.051}, "maximum_input_wait_fraction"),
        (
            {"minimum_real_to_synthetic_ratio": 0.899},
            "minimum_real_to_synthetic_ratio",
        ),
        (
            {"minimum_persisted_to_device_ratio": 0.899},
            "minimum_persisted_to_device_ratio",
        ),
        (
            {"minimum_hbm_headroom_fraction": 0.149},
            "minimum_hbm_headroom_fraction",
        ),
        (
            {"maximum_instrumentation_mfu_delta": 0.021},
            "maximum_instrumentation_mfu_delta",
        ),
        ({"maximum_p95_to_p50_ratio": 1.101}, "maximum_p95_to_p50_ratio"),
        (
            {"maximum_slowest_to_median_host_ratio": 1.051},
            "maximum_slowest_to_median_host_ratio",
        ),
        (
            {"maximum_routing_capacity_amplification": 1.101},
            "maximum_routing_capacity_amplification",
        ),
    ],
)
def test_no_companion_hard_gate_can_be_weakened(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(MFUContractError, match=message):
        qualify_mfu_windows(
            [_window(0), _window(1), _window(2)],
            phase="harvest_prefix",
            **override,  # type: ignore[arg-type]
        )


def test_companion_hard_gates_may_be_made_stricter() -> None:
    windows = [_window(index, mfu=0.53) for index in range(4)]
    result = qualify_mfu_windows(
        windows,
        phase="harvest_prefix",
        minimum_mfu=0.52,
        minimum_windows=4,
        minimum_steps_per_window=500,
        minimum_wall_seconds_per_window=60.0,
        maximum_input_wait_fraction=0.041,
        minimum_real_to_synthetic_ratio=0.95,
        minimum_persisted_to_device_ratio=0.95,
        minimum_hbm_headroom_fraction=0.15,
        maximum_instrumentation_mfu_delta=0.01,
        maximum_p95_to_p50_ratio=1.05,
        maximum_slowest_to_median_host_ratio=1.026,
        maximum_routing_capacity_amplification=1.05,
    )
    assert len(result.window_ids) == 4


def test_windows_must_be_nonoverlapping_and_static() -> None:
    overlap = replace(_window(1), step_start=499, step_end=999)
    with pytest.raises(MFUContractError, match="must not overlap"):
        qualify_mfu_windows([_window(0), overlap, _window(2)], phase="harvest_prefix")

    changed_hlo = replace(_window(1), hlo_sha256="d" * 64)
    with pytest.raises(MFUContractError, match="one static HLO"):
        qualify_mfu_windows(
            [_window(0), changed_hlo, _window(2)], phase="harvest_prefix"
        )


def test_dense_f_does_not_require_persistence_goodput() -> None:
    windows = [_window(index, phase="sae_f") for index in range(3)]
    result = qualify_mfu_windows(windows, phase="sae_f")
    assert result.phase == "sae_f"


def test_actual_dense_scope_calibration_has_its_own_phase() -> None:
    windows = [_window(index, phase="sae_dense_calibration") for index in range(3)]
    result = qualify_mfu_windows(windows, phase="sae_dense_calibration")
    assert result.phase == "sae_dense_calibration"


def test_bool_is_not_accepted_as_an_integer() -> None:
    with pytest.raises(MFUContractError, match="positive integer"):
        FlopTermV1("bad", True, "bool is not an integer")  # type: ignore[arg-type]


def test_nonfinite_evidence_is_rejected() -> None:
    with pytest.raises(MFUContractError, match="finite"):
        replace(_window(0), wall_seconds=float("nan"))
