#!/usr/bin/env python3
"""Strictly validate one identity-bound EP8 or EP16 v6e-64 benchmark result."""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

LAYOUT_PROFILES: dict[str, dict[str, Any]] = {
    "ep8fsdp8": {
        "mesh": "EP8/FSDP8",
        "fsdp": 8,
        "expert": 8,
        "logical_shape": [1, 8, 8, 1],
        "allow_split_physical_axes": False,
    },
    "ep16fsdp4": {
        "mesh": "EP16/FSDP4",
        "fsdp": 4,
        "expert": 16,
        "logical_shape": [1, 4, 16, 1],
        "allow_split_physical_axes": True,
    },
}
EXPECTED_VALID_TOKENS = 128 * 1_024
EXPECTED_USEFUL_FLOPS = 4_900_569_495_896_064
EXPECTED_PEAK_FLOPS = 58_752_000_000_000_000
EXPECTED_MEASUREMENT_STEPS = 500
EXPECTED_WARMUP_STEPS = 5
TARGET_MFU = Fraction(51, 100)
DEVICE_MEMORY_STATS_SCHEMA = "moe-sae.r1.v6e-device-memory-stats.v1"
DEVICE_MEMORY_STATS_SOURCE = "jax.Device.memory_stats"
DEVICE_MEMORY_SAMPLE_PHASE = (
    "after-final-synchronized-measurement-step-before-duration-and-evidence-gathers"
)
V6E_DOCUMENTED_HBM_BYTES = 32_000_000_000
MAXIMUM_HBM_USAGE = Fraction(85, 100)


class ResultValidationError(RuntimeError):
    """The measured result is malformed, mismatched, or below the MFU gate."""


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResultValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ResultValidationError(f"{name} fields changed")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ResultValidationError(f"{name} must be a positive integer")
    return value


def _require_exact_int(value: Any, expected: int, name: str) -> None:
    if type(value) is not int or value != expected:
        raise ResultValidationError(f"{name} changed")


def _fraction_record(value: Any, name: str) -> Fraction:
    record = _exact_keys(value, {"numerator", "denominator"}, name)
    numerator = _positive_int(record["numerator"], f"{name}.numerator")
    denominator = _positive_int(record["denominator"], f"{name}.denominator")
    if math.gcd(numerator, denominator) != 1:
        raise ResultValidationError(f"{name} is not a reduced fraction")
    return Fraction(numerator, denominator)


def _layout_profile(name: str) -> dict[str, Any]:
    if type(name) is not str or name not in LAYOUT_PROFILES:
        raise ResultValidationError("layout profile is not closed")
    return LAYOUT_PROFILES[name]


def _validate_physical_mesh(
    value: Any,
    profile: dict[str, Any],
) -> dict[int, int]:
    mesh = _exact_keys(
        value,
        {
            "axis_names",
            "logical_shape",
            "allow_split_physical_axes",
            "devices",
        },
        "physical_mesh",
    )
    if mesh["axis_names"] != ["data", "fsdp", "expert", "context"]:
        raise ResultValidationError("physical mesh axes changed")
    if mesh["logical_shape"] != profile["logical_shape"]:
        raise ResultValidationError(
            f"physical mesh shape is not {profile['mesh']}"
        )
    if (
        mesh["allow_split_physical_axes"]
        is not profile["allow_split_physical_axes"]
    ):
        raise ResultValidationError(
            f"physical-axis split policy is not {profile['mesh']}"
        )
    devices = mesh["devices"]
    if type(devices) is not list or len(devices) != 64:
        raise ResultValidationError("physical mesh must contain 64 devices")
    logical_indices: set[tuple[int, ...]] = set()
    device_ids: set[int] = set()
    physical_coordinates: set[tuple[int, ...]] = set()
    process_counts = {index: 0 for index in range(16)}
    device_processes: dict[int, int] = {}
    for index, raw in enumerate(devices):
        device = _exact_keys(
            raw,
            {
                "logical_index",
                "device_id",
                "process_index",
                "physical_coordinates",
            },
            f"physical_mesh.devices[{index}]",
        )
        logical = device["logical_index"]
        if (
            type(logical) is not list
            or len(logical) != 4
            or any(type(item) is not int for item in logical)
            or not (
                logical[0] == 0
                and 0 <= logical[1] < profile["fsdp"]
                and 0 <= logical[2] < profile["expert"]
                and logical[3] == 0
            )
        ):
            raise ResultValidationError("logical device index is invalid")
        logical_tuple = tuple(logical)
        if logical_tuple in logical_indices:
            raise ResultValidationError("logical device index is duplicated")
        logical_indices.add(logical_tuple)
        device_id = device["device_id"]
        if type(device_id) is not int or device_id < 0 or device_id in device_ids:
            raise ResultValidationError("physical device ID is invalid or duplicated")
        device_ids.add(device_id)
        process_index = device["process_index"]
        if type(process_index) is not int or process_index not in process_counts:
            raise ResultValidationError("JAX process index is out of range")
        process_counts[process_index] += 1
        device_processes[device_id] = process_index
        coordinates = device["physical_coordinates"]
        if (
            type(coordinates) is not list
            or not coordinates
            or any(type(item) is not int or item < 0 for item in coordinates)
        ):
            raise ResultValidationError("physical coordinates are invalid")
        coordinate_tuple = tuple(coordinates)
        if coordinate_tuple in physical_coordinates:
            raise ResultValidationError("physical coordinates are duplicated")
        physical_coordinates.add(coordinate_tuple)
    if logical_indices != {
        (0, fsdp, expert, 0)
        for fsdp in range(profile["fsdp"])
        for expert in range(profile["expert"])
    }:
        raise ResultValidationError(
            f"physical mesh does not cover {profile['mesh']}"
        )
    if set(process_counts.values()) != {4}:
        raise ResultValidationError("each JAX process must own exactly four devices")
    if device_ids != set(range(64)):
        raise ResultValidationError("physical device IDs must cover 0 through 63")
    return device_processes


def _validate_output(value: Any) -> None:
    output = _exact_keys(
        value,
        {
            "shapes",
            "dtypes",
            "activation_finite",
            "ids_in_range",
            "ids_unique",
            "weights_finite",
            "weights_positive",
            "weights_sum_scaled",
        },
        "output_validation",
    )
    if output["shapes"] != [
        [128, 1_024, 7_168],
        [128, 1_024, 8],
        [128, 1_024, 8],
    ]:
        raise ResultValidationError("prefix output shapes changed")
    if output["dtypes"] != ["bfloat16", "int32", "float32"]:
        raise ResultValidationError("prefix output dtypes changed")
    for name in (
        "activation_finite",
        "ids_in_range",
        "ids_unique",
        "weights_finite",
        "weights_positive",
        "weights_sum_scaled",
    ):
        if output[name] is not True:
            raise ResultValidationError(f"prefix output invariant failed: {name}")


def _validate_device_memory(
    value: Any,
    *,
    device_processes: dict[int, int],
) -> dict[str, Any]:
    evidence = _exact_keys(
        value,
        {
            "schema",
            "source",
            "sample_phase",
            "documented_v6e_hbm_bytes",
            "maximum_hbm_usage",
            "bounded_gather",
            "devices",
            "passes_15_percent_headroom",
        },
        "device_memory",
    )
    if evidence["schema"] != DEVICE_MEMORY_STATS_SCHEMA:
        raise ResultValidationError("device-memory schema changed")
    if evidence["source"] != DEVICE_MEMORY_STATS_SOURCE:
        raise ResultValidationError("device-memory source is not Device.memory_stats")
    if evidence["sample_phase"] != DEVICE_MEMORY_SAMPLE_PHASE:
        raise ResultValidationError("device-memory sample phase changed")
    _require_exact_int(
        evidence["documented_v6e_hbm_bytes"],
        V6E_DOCUMENTED_HBM_BYTES,
        "documented v6e HBM capacity",
    )
    maximum_usage = _exact_keys(
        evidence["maximum_hbm_usage"],
        {"numerator", "denominator"},
        "maximum_hbm_usage",
    )
    _require_exact_int(
        maximum_usage["numerator"],
        MAXIMUM_HBM_USAGE.numerator,
        "maximum HBM usage numerator",
    )
    _require_exact_int(
        maximum_usage["denominator"],
        MAXIMUM_HBM_USAGE.denominator,
        "maximum HBM usage denominator",
    )
    gather = _exact_keys(
        evidence["bounded_gather"],
        {
            "process_count",
            "records_per_process",
            "uint32_words_per_record",
            "total_bytes",
        },
        "device_memory.bounded_gather",
    )
    for name, expected in (
        ("process_count", 16),
        ("records_per_process", 4),
        ("uint32_words_per_record", 12),
        ("total_bytes", 16 * 4 * 12 * 4),
    ):
        _require_exact_int(
            gather[name],
            expected,
            f"device-memory gather {name}",
        )

    devices = evidence["devices"]
    if type(devices) is not list or len(devices) != 64:
        raise ResultValidationError(
            "device-memory evidence must contain exactly 64 records"
        )
    derived_passes = True
    headroom_fractions: list[Fraction] = []
    for expected_device_id, raw_device in enumerate(devices):
        device = _exact_keys(
            raw_device,
            {
                "device_id",
                "process_index",
                "memory_stats_status",
                "peak_bytes_in_use_key_present",
                "peak_bytes_in_use",
                "bytes_limit_key_present",
                "bytes_limit",
                "bytes_in_use_key_present",
                "bytes_in_use",
            },
            f"device_memory.devices[{expected_device_id}]",
        )
        _require_exact_int(
            device["device_id"],
            expected_device_id,
            "device-memory device ID",
        )
        expected_process_index = device_processes.get(expected_device_id)
        _require_exact_int(
            device["process_index"],
            expected_process_index,
            "device-memory process identity",
        )
        status = device["memory_stats_status"]
        if status not in {
            "available",
            "returned-none",
            "call-error",
            "invalid-return-type",
        }:
            raise ResultValidationError("device-memory backend status is invalid")

        stat_values: dict[str, int | None] = {}
        for stat_name in (
            "peak_bytes_in_use",
            "bytes_limit",
            "bytes_in_use",
        ):
            key_present = device[f"{stat_name}_key_present"]
            stat_value = device[stat_name]
            if type(key_present) is not bool:
                raise ResultValidationError(
                    f"{stat_name} key-presence evidence is not boolean"
                )
            if not key_present and stat_value is not None:
                raise ResultValidationError(
                    f"{stat_name} has a value without backend key presence"
                )
            if stat_value is not None and type(stat_value) is not int:
                raise ResultValidationError(
                    f"{stat_name} is not an exact integer or null"
                )
            stat_values[stat_name] = stat_value
        if status != "available" and any(
            device[f"{name}_key_present"]
            for name in (
                "peak_bytes_in_use",
                "bytes_limit",
                "bytes_in_use",
            )
        ):
            raise ResultValidationError(
                "unavailable device-memory backend reports key presence"
            )

        peak = stat_values["peak_bytes_in_use"]
        limit = stat_values["bytes_limit"]
        current = stat_values["bytes_in_use"]
        if (
            status != "available"
            or device["peak_bytes_in_use_key_present"] is not True
            or peak is None
            or peak <= 0
        ):
            derived_passes = False
            continue
        if device["bytes_limit_key_present"]:
            if limit is None or limit <= 0:
                derived_passes = False
                continue
            effective_capacity = min(limit, V6E_DOCUMENTED_HBM_BYTES)
        else:
            effective_capacity = V6E_DOCUMENTED_HBM_BYTES
        if device["bytes_in_use_key_present"] and (
            current is None or current < 0 or current > peak
        ):
            derived_passes = False
        if peak * MAXIMUM_HBM_USAGE.denominator > (
            effective_capacity * MAXIMUM_HBM_USAGE.numerator
        ):
            derived_passes = False
        headroom_fractions.append(
            Fraction(effective_capacity - peak, effective_capacity)
        )

    reported_passes = evidence["passes_15_percent_headroom"]
    if type(reported_passes) is not bool or reported_passes is not derived_passes:
        raise ResultValidationError("reported HBM headroom gate is not derived")
    if not derived_passes or len(headroom_fractions) != 64:
        raise ResultValidationError(
            "Device.memory_stats does not prove 15% HBM headroom on all 64 devices"
        )
    minimum_headroom = min(headroom_fractions)
    return {
        "minimum_hbm_headroom": float(minimum_headroom),
        "maximum_peak_hbm_bytes": max(
            device["peak_bytes_in_use"] for device in devices
        ),
        "passes_15_percent_hbm_headroom": True,
    }


def _validate_benchmark(value: Any) -> dict[str, Any]:
    benchmark = _exact_keys(
        value,
        {
            "schema",
            "warmup_steps",
            "measurement_steps",
            "valid_tokens_per_step",
            "useful_flops_per_step",
            "peak_flops_per_second",
            "durations_ns",
            "total_duration_ns",
            "total_valid_tokens",
            "total_useful_flops",
            "aggregate_mfu",
            "tokens_per_second",
            "median_step_ns",
            "p95_step_ns",
            "passes_51_percent",
        },
        "benchmark",
    )
    if benchmark["schema"] != "moe-sae.r1.prefix-benchmark-result.v1":
        raise ResultValidationError("benchmark schema changed")
    _require_exact_int(
        benchmark["warmup_steps"], EXPECTED_WARMUP_STEPS, "benchmark warmup count"
    )
    _require_exact_int(
        benchmark["measurement_steps"],
        EXPECTED_MEASUREMENT_STEPS,
        "benchmark measurement count",
    )
    _require_exact_int(
        benchmark["valid_tokens_per_step"],
        EXPECTED_VALID_TOKENS,
        "benchmark valid-token count",
    )
    _require_exact_int(
        benchmark["useful_flops_per_step"],
        EXPECTED_USEFUL_FLOPS,
        "benchmark useful-FLOP count",
    )
    _require_exact_int(
        benchmark["peak_flops_per_second"],
        EXPECTED_PEAK_FLOPS,
        "benchmark peak denominator",
    )
    durations = benchmark["durations_ns"]
    if (
        type(durations) is not list
        or len(durations) != EXPECTED_MEASUREMENT_STEPS
        or any(type(item) is not int or item <= 0 for item in durations)
    ):
        raise ResultValidationError("benchmark durations are invalid")
    total_duration = sum(durations)
    _require_exact_int(
        benchmark["total_duration_ns"],
        total_duration,
        "benchmark total duration",
    )
    expected_total_tokens = EXPECTED_VALID_TOKENS * EXPECTED_MEASUREMENT_STEPS
    expected_total_flops = EXPECTED_USEFUL_FLOPS * EXPECTED_MEASUREMENT_STEPS
    _require_exact_int(
        benchmark["total_valid_tokens"],
        expected_total_tokens,
        "benchmark total tokens",
    )
    _require_exact_int(
        benchmark["total_useful_flops"],
        expected_total_flops,
        "benchmark total FLOPs",
    )
    aggregate_mfu = Fraction(
        expected_total_flops * 1_000_000_000,
        EXPECTED_PEAK_FLOPS * total_duration,
    )
    if aggregate_mfu > 1:
        raise ResultValidationError("derived aggregate MFU exceeds 100%")
    tokens_per_second = Fraction(
        expected_total_tokens * 1_000_000_000,
        total_duration,
    )
    if _fraction_record(benchmark["aggregate_mfu"], "aggregate_mfu") != aggregate_mfu:
        raise ResultValidationError("reported aggregate MFU is not derived")
    if (
        _fraction_record(benchmark["tokens_per_second"], "tokens_per_second")
        != tokens_per_second
    ):
        raise ResultValidationError("reported token rate is not derived")
    ordered = sorted(durations)
    middle = len(ordered) // 2
    expected_median = (ordered[middle - 1] + ordered[middle]) // 2
    _require_exact_int(
        benchmark["median_step_ns"],
        expected_median,
        "reported median step time",
    )
    expected_p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    _require_exact_int(
        benchmark["p95_step_ns"],
        expected_p95,
        "reported p95 step time",
    )
    passed = aggregate_mfu >= TARGET_MFU
    if benchmark["passes_51_percent"] is not passed:
        raise ResultValidationError("reported MFU gate is not derived")
    summary = {
        "aggregate_mfu": float(aggregate_mfu),
        "median_step_ms": benchmark["median_step_ns"] / 1_000_000,
        "p95_step_ms": benchmark["p95_step_ns"] / 1_000_000,
        "tokens_per_second": float(tokens_per_second),
        "passes_51_percent": passed,
    }
    if not passed:
        raise ResultValidationError(
            "measured useful-prefix MFU is below 51%: "
            f"{100 * float(aggregate_mfu):.6f}%"
        )
    return summary


def validate_result(path: Path, layout_profile: str) -> dict[str, Any]:
    profile = _layout_profile(layout_profile)
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_strict_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResultValidationError("result is not strict UTF-8 JSON") from error
    result = _exact_keys(
        value,
        {
            "schema",
            "mesh",
            "seed",
            "process_count",
            "device_count",
            "physical_mesh",
            "prepare_seconds",
            "output_validation",
            "benchmark",
            "device_memory",
        },
        "result",
    )
    if result["schema"] != "moe-sae.r1.synthetic-prefix-qualification.v2":
        raise ResultValidationError("result schema changed")
    if result["mesh"] != profile["mesh"]:
        raise ResultValidationError(
            f"result mesh is not identity-bound {profile['mesh']}"
        )
    _require_exact_int(result["seed"], 0, "result seed")
    _require_exact_int(result["process_count"], 16, "result process count")
    _require_exact_int(result["device_count"], 64, "result device count")
    prepare_seconds = result["prepare_seconds"]
    if (
        type(prepare_seconds) not in {int, float}
        or not math.isfinite(prepare_seconds)
        or prepare_seconds <= 0
    ):
        raise ResultValidationError("prepare timing is invalid")
    device_processes = _validate_physical_mesh(result["physical_mesh"], profile)
    _validate_output(result["output_validation"])
    summary = _validate_benchmark(result["benchmark"])
    memory_summary = _validate_device_memory(
        result["device_memory"],
        device_processes=device_processes,
    )
    return {
        "schema": "moe-sae.r1.legacy-v6e-result-validation.v2",
        "layout_profile": layout_profile,
        "mesh": profile["mesh"],
        "process_count": 16,
        "device_count": 64,
        "useful_flops_per_step": EXPECTED_USEFUL_FLOPS,
        **summary,
        **memory_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--layout-profile",
        choices=tuple(LAYOUT_PROFILES),
        required=True,
    )
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    try:
        summary = validate_result(args.result, args.layout_profile)
    except (OSError, ResultValidationError) as error:
        raise SystemExit(f"legacy v6e result rejected: {error}") from error
    print(
        json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
