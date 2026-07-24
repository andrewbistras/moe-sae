"""One-command synthetic v6e-64 qualification for the R1 prefix."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .maxtext_prefix_runtime import (
    PREFIX_MESH_CANDIDATES,
    PrefixRuntimeSettings,
    assert_real_v6e64_runtime,
    create_compile_synthetic_prefix,
    create_prefix_mesh,
    create_synthetic_prefix_batch,
    initialize_maxtext_config,
    initialize_v6e_distributed_runtime,
    synchronize_prefix_output,
)
from .prefix_benchmark import (
    PackedStepShape,
    reduce_max_durations_across_jax_processes,
    run_synchronized_prefix_benchmark,
)


class PrefixBenchmarkRunnerError(RuntimeError):
    """The real-device benchmark runner failed its executable boundary."""


_DEVICE_MEMORY_STATS_SCHEMA = "moe-sae.r1.v6e-device-memory-stats.v1"
_DEVICE_MEMORY_STATS_SOURCE = "jax.Device.memory_stats"
_DEVICE_MEMORY_SAMPLE_PHASE = (
    "after-final-synchronized-measurement-step-before-duration-and-evidence-gathers"
)
_V6E_DOCUMENTED_HBM_BYTES = 32_000_000_000
_MAXIMUM_HBM_USAGE_NUMERATOR = 17
_MAXIMUM_HBM_USAGE_DENOMINATOR = 20
_EXPECTED_PROCESS_COUNT = 16
_EXPECTED_LOCAL_DEVICE_COUNT = 4
_MEMORY_WORDS_PER_DEVICE = 12
_MEMORY_STATUS_NAMES = {
    0: "returned-none",
    1: "available",
    2: "call-error",
    3: "invalid-return-type",
}


def _mesh_argument(value: str) -> int:
    names = tuple(item.name for item in PREFIX_MESH_CANDIDATES)
    try:
        return names.index(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"must be one of {', '.join(names)}"
        ) from error


def _seed_argument(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 0 <= seed < 2**31:
        raise argparse.ArgumentTypeError("must be in [0, 2^31)")
    return seed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--maxtext-base-yml", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--mesh",
        type=_mesh_argument,
        default=0,
        metavar="EP/FSDP",
        help="EP16/FSDP4, EP32/FSDP2, or EP8/FSDP8",
    )
    parser.add_argument("--seed", type=_seed_argument, default=0)
    parser.add_argument("--result-json", type=Path)
    return parser.parse_args(argv)


def _validate_output(output: Any) -> dict[str, Any]:
    """Reduce the complete output ABI to bounded device-side invariants."""

    try:
        import jax
        import jax.numpy as jnp
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - TPU image
        raise PrefixBenchmarkRunnerError("JAX is unavailable") from error
    if not isinstance(output, tuple) or len(output) != 3:
        raise PrefixBenchmarkRunnerError("prefix output must be one exact triple")
    activation, expert_ids, router_weights = output
    expected = (
        ((128, 1_024, 7_168), "bfloat16"),
        ((128, 1_024, 8), "int32"),
        ((128, 1_024, 8), "float32"),
    )
    for value, (shape, dtype) in zip(output, expected, strict=True):
        if tuple(getattr(value, "shape", ())) != shape or str(value.dtype) != dtype:
            raise PrefixBenchmarkRunnerError("prefix output ABI changed")

    sorted_ids = jnp.sort(expert_ids, axis=-1)
    checks = {
        "activation_finite": jnp.all(jnp.isfinite(activation)),
        "ids_in_range": jnp.all((expert_ids >= 0) & (expert_ids < 256)),
        "ids_unique": jnp.all(sorted_ids[..., 1:] != sorted_ids[..., :-1]),
        "weights_finite": jnp.all(jnp.isfinite(router_weights)),
        "weights_positive": jnp.all(router_weights > 0),
        "weights_sum_scaled": jnp.all(
            jnp.abs(jnp.sum(router_weights, axis=-1) - 2.5) <= 2e-5
        ),
    }
    observed = {
        name: bool(value)
        for name, value in zip(
            checks,
            jax.device_get(tuple(checks.values())),
            strict=True,
        )
    }
    if not all(observed.values()):
        raise PrefixBenchmarkRunnerError(
            f"prefix output invariants failed: {observed!r}"
        )
    return {
        "shapes": [list(item[0]) for item in expected],
        "dtypes": [item[1] for item in expected],
        **observed,
    }


def _physical_mesh_evidence(mesh: Any, config: Any) -> dict[str, Any]:
    """Describe the exact logical-to-physical assignment used by MaxText."""

    try:
        import numpy as np
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - image
        raise PrefixBenchmarkRunnerError("NumPy is unavailable") from error
    axis_names = tuple(getattr(mesh, "axis_names", ()))
    if axis_names != ("data", "fsdp", "expert", "context"):
        raise PrefixBenchmarkRunnerError("prefix mesh axis order changed")
    devices = np.asarray(getattr(mesh, "devices", ()), dtype=object)
    expected_shape = (
        1,
        int(getattr(config, "ici_fsdp_parallelism", 0)),
        int(getattr(config, "ici_expert_parallelism", 0)),
        1,
    )
    if tuple(devices.shape) != expected_shape or devices.size != 64:
        raise PrefixBenchmarkRunnerError("prefix mesh device assignment changed")

    records: list[dict[str, Any]] = []
    device_ids: set[int] = set()
    for logical_index in np.ndindex(devices.shape):
        device = devices[logical_index]
        try:
            device_id = int(device.id)
            process_index = int(device.process_index)
            coordinates = [int(value) for value in device.coords]
        except (AttributeError, TypeError, ValueError) as error:
            raise PrefixBenchmarkRunnerError(
                "v6e device physical identity is unavailable"
            ) from error
        if not coordinates or device_id in device_ids:
            raise PrefixBenchmarkRunnerError(
                "v6e device physical identity is not one-to-one"
            )
        device_ids.add(device_id)
        records.append(
            {
                "logical_index": [int(value) for value in logical_index],
                "device_id": device_id,
                "process_index": process_index,
                "physical_coordinates": coordinates,
            }
        )
    return {
        "axis_names": list(axis_names),
        "logical_shape": list(expected_shape),
        "allow_split_physical_axes": bool(
            getattr(config, "allow_split_physical_axes", False)
        ),
        "devices": records,
    }


def _signed_int_words(value: int) -> tuple[int, int]:
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise PrefixBenchmarkRunnerError(
            "device memory statistic is not a signed 64-bit integer"
        )
    encoded = value & ((1 << 64) - 1)
    return encoded >> 32, encoded & ((1 << 32) - 1)


def _signed_int_from_words(high: int, low: int) -> int:
    encoded = (high << 32) | low
    if encoded >= 1 << 63:
        encoded -= 1 << 64
    return encoded


def _memory_stat_words(
    stats: Mapping[str, Any] | None,
    key: str,
) -> tuple[int, int, int]:
    if stats is None or key not in stats:
        return 0, 0, 0
    value = stats[key]
    if type(value) is not int:
        return 2, 0, 0
    high, low = _signed_int_words(value)
    return 1, high, low


def _local_device_memory_words(
    devices: Sequence[Any],
    *,
    process_index: int,
) -> Any:
    """Call ``Device.memory_stats`` once per local device and encode fixed words."""

    try:
        import numpy as np
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - image
        raise PrefixBenchmarkRunnerError("NumPy is unavailable") from error
    if (
        type(process_index) is not int
        or not 0 <= process_index < _EXPECTED_PROCESS_COUNT
    ):
        raise PrefixBenchmarkRunnerError("JAX process index is invalid")
    if len(devices) != _EXPECTED_LOCAL_DEVICE_COUNT:
        raise PrefixBenchmarkRunnerError(
            "each JAX process must expose exactly four local devices"
        )

    words = np.zeros(
        (_EXPECTED_LOCAL_DEVICE_COUNT, _MEMORY_WORDS_PER_DEVICE),
        dtype=np.uint32,
    )
    for row, device in enumerate(devices):
        try:
            device_id = int(device.id)
            device_process_index = int(device.process_index)
        except (AttributeError, TypeError, ValueError) as error:
            raise PrefixBenchmarkRunnerError(
                "local device identity is unavailable"
            ) from error
        if (
            not 0 <= device_id < 2**32
            or device_process_index != process_index
        ):
            raise PrefixBenchmarkRunnerError("local device identity is inconsistent")

        stats: Mapping[str, Any] | None
        try:
            raw_stats = device.memory_stats()
        except Exception:  # noqa: BLE001 - diagnostic failure is evidence
            memory_status = 2
            stats = None
        else:
            if raw_stats is None:
                memory_status = 0
                stats = None
            elif isinstance(raw_stats, Mapping):
                memory_status = 1
                stats = raw_stats
            else:
                memory_status = 3
                stats = None

        peak = _memory_stat_words(stats, "peak_bytes_in_use")
        limit = _memory_stat_words(stats, "bytes_limit")
        current = _memory_stat_words(stats, "bytes_in_use")
        words[row] = (
            device_id,
            process_index,
            memory_status,
            *peak,
            *limit,
            *current,
        )
    return words


def _decode_device_memory_words(gathered: Any) -> dict[str, Any]:
    """Validate and decode the exact 64-device bounded gather."""

    try:
        import numpy as np
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - image
        raise PrefixBenchmarkRunnerError("NumPy is unavailable") from error
    words = np.asarray(gathered)
    expected_shape = (
        _EXPECTED_PROCESS_COUNT,
        _EXPECTED_LOCAL_DEVICE_COUNT,
        _MEMORY_WORDS_PER_DEVICE,
    )
    if words.shape != expected_shape or words.dtype != np.dtype(np.uint32):
        raise PrefixBenchmarkRunnerError(
            "device memory gather has the wrong fixed shape or dtype"
        )

    devices: list[dict[str, Any]] = []
    observed_device_ids: set[int] = set()
    for gathered_process_index in range(_EXPECTED_PROCESS_COUNT):
        for local_index in range(_EXPECTED_LOCAL_DEVICE_COUNT):
            row = [int(value) for value in words[gathered_process_index, local_index]]
            device_id, process_index, memory_status = row[:3]
            if process_index != gathered_process_index:
                raise PrefixBenchmarkRunnerError(
                    "device memory gather process identity changed"
                )
            if device_id in observed_device_ids:
                raise PrefixBenchmarkRunnerError(
                    "device memory gather contains a duplicate device"
                )
            observed_device_ids.add(device_id)
            try:
                status_name = _MEMORY_STATUS_NAMES[memory_status]
            except KeyError as error:
                raise PrefixBenchmarkRunnerError(
                    "device memory status is invalid"
                ) from error

            fields: dict[str, Any] = {}
            any_stat_present = False
            for name, offset in (
                ("peak_bytes_in_use", 3),
                ("bytes_limit", 6),
                ("bytes_in_use", 9),
            ):
                value_status, high, low = row[offset : offset + 3]
                if value_status not in {0, 1, 2}:
                    raise PrefixBenchmarkRunnerError(
                        "device memory key status is invalid"
                    )
                key_present = value_status != 0
                any_stat_present = any_stat_present or key_present
                fields[f"{name}_key_present"] = key_present
                fields[name] = (
                    _signed_int_from_words(high, low)
                    if value_status == 1
                    else None
                )
            if status_name != "available" and any_stat_present:
                raise PrefixBenchmarkRunnerError(
                    "unavailable memory statistics contain reported keys"
                )
            devices.append(
                {
                    "device_id": device_id,
                    "process_index": process_index,
                    "memory_stats_status": status_name,
                    **fields,
                }
            )

    if observed_device_ids != set(range(64)):
        raise PrefixBenchmarkRunnerError(
            "device memory gather must cover device IDs 0 through 63"
        )
    devices.sort(key=lambda item: item["device_id"])

    passes_headroom = True
    for device in devices:
        peak = device["peak_bytes_in_use"]
        limit = device["bytes_limit"]
        current = device["bytes_in_use"]
        if (
            device["memory_stats_status"] != "available"
            or device["peak_bytes_in_use_key_present"] is not True
            or type(peak) is not int
            or peak <= 0
        ):
            passes_headroom = False
            continue
        if device["bytes_limit_key_present"]:
            if type(limit) is not int or limit <= 0:
                passes_headroom = False
                continue
            effective_capacity = min(limit, _V6E_DOCUMENTED_HBM_BYTES)
        else:
            effective_capacity = _V6E_DOCUMENTED_HBM_BYTES
        if device["bytes_in_use_key_present"] and (
            type(current) is not int or current < 0 or current > peak
        ):
            passes_headroom = False
        if (
            peak * _MAXIMUM_HBM_USAGE_DENOMINATOR
            > effective_capacity * _MAXIMUM_HBM_USAGE_NUMERATOR
        ):
            passes_headroom = False

    return {
        "schema": _DEVICE_MEMORY_STATS_SCHEMA,
        "source": _DEVICE_MEMORY_STATS_SOURCE,
        "sample_phase": _DEVICE_MEMORY_SAMPLE_PHASE,
        "documented_v6e_hbm_bytes": _V6E_DOCUMENTED_HBM_BYTES,
        "maximum_hbm_usage": {
            "numerator": _MAXIMUM_HBM_USAGE_NUMERATOR,
            "denominator": _MAXIMUM_HBM_USAGE_DENOMINATOR,
        },
        "bounded_gather": {
            "process_count": _EXPECTED_PROCESS_COUNT,
            "records_per_process": _EXPECTED_LOCAL_DEVICE_COUNT,
            "uint32_words_per_record": _MEMORY_WORDS_PER_DEVICE,
            "total_bytes": int(words.nbytes),
        },
        "devices": devices,
        "passes_15_percent_headroom": passes_headroom,
    }


def _sample_local_device_memory_words() -> Any:
    """Sample local device statistics without performing any collective."""

    try:
        import jax
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - image
        raise PrefixBenchmarkRunnerError("JAX is unavailable") from error
    if jax.process_count() != _EXPECTED_PROCESS_COUNT:
        raise PrefixBenchmarkRunnerError(
            "device memory evidence requires exactly 16 JAX processes"
        )
    return _local_device_memory_words(
        tuple(jax.local_devices()),
        process_index=jax.process_index(),
    )


def _gather_device_memory_evidence(local_words: Any) -> dict[str, Any]:
    """Gather and decode statistics already sampled on all 16 processes."""

    try:
        import jax
        from jax.experimental import multihost_utils
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - image
        raise PrefixBenchmarkRunnerError("JAX is unavailable") from error
    if jax.process_count() != _EXPECTED_PROCESS_COUNT:
        raise PrefixBenchmarkRunnerError(
            "device memory evidence requires exactly 16 JAX processes"
        )
    gathered = multihost_utils.process_allgather(local_words, tiled=False)
    return _decode_device_memory_words(gathered)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def run(argv: Sequence[str] | None = None) -> dict[str, Any] | None:
    """Compile, validate, and benchmark one exact synthetic mesh candidate."""

    args = parse_args(argv)
    initialize_v6e_distributed_runtime()
    assert_real_v6e64_runtime()
    mesh_candidate = PREFIX_MESH_CANDIDATES[args.mesh]
    settings = PrefixRuntimeSettings(
        maxtext_base_yml=args.maxtext_base_yml.absolute(),
        output_directory=args.output_directory.absolute(),
        mesh=mesh_candidate,
        run_name=(
            "deepseek-r1-prefix-synthetic-"
            + mesh_candidate.name.lower().replace("/", "-")
        ),
    )
    config = initialize_maxtext_config(settings)
    mesh = create_prefix_mesh(config)
    mesh_evidence = _physical_mesh_evidence(mesh, config)
    batch = create_synthetic_prefix_batch(mesh)

    compile_started = time.monotonic_ns()
    compiled = create_compile_synthetic_prefix(
        config,
        mesh,
        batch,
        seed=args.seed,
    )
    compile_stopped = time.monotonic_ns()
    local_prepare_ns = compile_stopped - compile_started
    prepare_ns = reduce_max_durations_across_jax_processes((local_prepare_ns,))[0]

    validation = _validate_output(compiled(batch))
    sampled_memory_words: Any | None = None

    def reduce_durations_after_memory_sample(
        local_durations: tuple[int, ...],
    ) -> tuple[int, ...]:
        nonlocal sampled_memory_words
        if sampled_memory_words is not None:
            raise PrefixBenchmarkRunnerError(
                "device memory statistics were sampled more than once"
            )
        sampled_memory_words = _sample_local_device_memory_words()
        return reduce_max_durations_across_jax_processes(local_durations)

    result = run_synchronized_prefix_benchmark(
        compiled,
        batch,
        packed_shape=PackedStepShape.full_sequences(),
        synchronize=synchronize_prefix_output,
        reduce_max_durations_ns=reduce_durations_after_memory_sample,
    )
    if sampled_memory_words is None:
        raise PrefixBenchmarkRunnerError(
            "benchmark did not sample post-measurement device memory"
        )
    device_memory = _gather_device_memory_evidence(sampled_memory_words)
    payload = {
        "schema": "moe-sae.r1.synthetic-prefix-qualification.v2",
        "mesh": mesh_candidate.name,
        "seed": args.seed,
        "process_count": 16,
        "device_count": 64,
        "physical_mesh": mesh_evidence,
        "prepare_seconds": prepare_ns / 1_000_000_000,
        "output_validation": validation,
        "benchmark": result.as_dict(),
        "device_memory": device_memory,
    }
    if not math.isfinite(payload["prepare_seconds"]):
        raise PrefixBenchmarkRunnerError("prepare timing is not finite")

    import jax

    if jax.process_index() == 0:
        if args.result_json is not None:
            _atomic_json(args.result_json, payload)
        print(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
        return payload
    return None


def main(argv: Sequence[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover - executable module
    raise SystemExit(main())
