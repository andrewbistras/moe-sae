"""Diagnostic-only multi-host profiler for the Gemma 4 prefix adapter.

This module deliberately does not participate in the production benchmark
supervisor.  It runs one short synthetic candidate, writes a distinct profile
schema below a caller-supplied ``diagnostics`` directory, and can never produce
a benchmark selection record.  Invoke it concurrently on every TPU controller
with ``python -m moe_sae.profile_worker``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np

from moe_sae.config import Gemma4Config, load_config
from moe_sae.harvest import (
    _block_until_ready,
    _capture_output_shardings,
    _close_maxtext_runtime,
    _jit_cache_size,
    compile_adapter_step,
    extract_grain_batch,
)
from moe_sae.model import CaptureBatch, Gemma4PrefixAdapter, checkpoint_inventory_sha256
from moe_sae.runtime import (
    HARVEST_MESHES,
    PER_DEVICE_SEQUENCE_BATCHES,
    V4_32_BF16_PEAK_FLOPS,
    atomic_write_json,
    build_maxtext_argv,
    create_maxtext_runtime,
    environment_fingerprint,
    estimate_useful_prefix_flops,
    maxtext_mesh_context,
    require_v4_32_runtime,
)


PROFILE_SCHEMA = "gemma4-prefix-profile-v1"
MAX_DIAGNOSTIC_STEPS = 50
MAX_TRACE_STEPS = 20
_KERNEL_CONFIG_FIELDS = (
    "use_ragged_sort",
    "ragged_buffer_factor",
    "megablox",
    "sparse_matmul",
    "use_gmm_v2",
    "use_tokamax_gmm",
    "attention",
    "model_call_mode",
    "prefuse_moe_weights",
)


def _sha256_argument(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError(
            "must be a lowercase 64-character SHA-256 digest"
        )
    return value


def _mesh_by_name(name: str):
    for candidate in HARVEST_MESHES:
        if candidate.name == name:
            return candidate
    raise ValueError(f"unknown harvest mesh {name!r}")


def _process_context() -> tuple[int, int, int]:
    import jax

    return int(jax.process_index()), int(jax.process_count()), int(jax.device_count())


def _validate_diagnostic_destination(path: Path) -> None:
    if "diagnostics" not in {part.lower() for part in path.parts}:
        raise ValueError("profile output must be below a directory named diagnostics")
    if path.suffix:
        raise ValueError("profile output must be a directory, not a file")


def _coordinated_digest(digest: str, *, purpose: str, process_count: int) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError(f"{purpose} is missing its SHA-256 digest")
    if process_count == 1:
        return digest
    from jax.experimental import multihost_utils

    payload = np.frombuffer(bytes.fromhex(digest), dtype=np.uint8).copy()
    gathered = np.asarray(
        multihost_utils.process_allgather(payload, tiled=False), dtype=np.uint8
    ).reshape(process_count, 32)
    if not np.all(gathered == gathered[0]):
        raise RuntimeError(f"TPU controllers disagree on {purpose}")
    return gathered[0].tobytes().hex()


def _verify_checkpoint(
    checkpoint: str, claimed_sha256: str, *, process_count: int
) -> tuple[str, float]:
    started = time.perf_counter()
    local_error: Exception | None = None
    try:
        local_digest = checkpoint_inventory_sha256(checkpoint)
    except Exception as error:  # exchange failure state before raising locally
        local_error = error
        local_digest = "0" * 64
    if process_count == 1:
        if local_error is not None:
            raise RuntimeError("could not inventory the staged checkpoint") from local_error
        actual = local_digest
    else:
        from jax.experimental import multihost_utils

        local = np.concatenate(
            (
                np.asarray([local_error is None], dtype=np.uint8),
                np.frombuffer(bytes.fromhex(local_digest), dtype=np.uint8),
            )
        )
        rows = np.asarray(
            multihost_utils.process_allgather(local, tiled=False), dtype=np.uint8
        ).reshape(process_count, 33)
        if not np.all(rows[:, 0] == 1):
            raise RuntimeError(
                "checkpoint inventory failed on at least one TPU controller"
            )
        if not np.all(rows[:, 1:] == rows[0, 1:]):
            raise RuntimeError(
                "TPU controllers observe different checkpoint inventories"
            )
        actual = rows[0, 1:].tobytes().hex()
    if actual != claimed_sha256:
        raise RuntimeError(
            "--checkpoint-sha256 does not match the staged checkpoint inventory"
        )
    return actual, time.perf_counter() - started


def _sharding_descriptor(value: Any) -> dict[str, Any]:
    sharding = getattr(value, "sharding", None)
    device_set = getattr(sharding, "device_set", ()) if sharding is not None else ()
    return {
        "shape": [int(size) for size in getattr(value, "shape", ())],
        "dtype": str(getattr(value, "dtype", type(value).__name__)),
        "sharding_type": type(sharding).__name__ if sharding is not None else None,
        "partition_spec": (
            repr(getattr(sharding, "spec"))
            if sharding is not None and hasattr(sharding, "spec")
            else None
        ),
        "memory_kind": (
            str(getattr(sharding, "memory_kind", None))
            if sharding is not None
            else None
        ),
        "device_count": len(device_set),
        "addressable_shard_count": len(getattr(value, "addressable_shards", ())),
    }


def _effective_kernel_config(config: Any) -> dict[str, Any]:
    """Return the JSON-native MaxText switches that select MoE/attention kernels."""

    result: dict[str, Any] = {}
    for name in _KERNEL_CONFIG_FIELDS:
        try:
            value = getattr(config, name)
        except (AttributeError, KeyError, ValueError):
            value = None
        if isinstance(value, np.generic):
            value = value.item()
        if value is not None and not isinstance(value, (bool, int, float, str)):
            value = str(value)
        result[name] = value
    return result


def _parameter_sharding_summary(model: Any) -> dict[str, Any]:
    import jax
    from flax import nnx

    groups: Counter[str] = Counter()
    group_bytes: Counter[str] = Counter()
    arrays = 0
    elements = 0
    total_bytes = 0
    for leaf in jax.tree_util.tree_leaves(nnx.state(model)):
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            continue
        arrays += 1
        size = int(math.prod(int(axis) for axis in leaf.shape))
        # JAX typed PRNG keys use an extended dtype such as ``key<fry>`` that
        # NumPy deliberately cannot interpret.  JAX arrays expose the correct
        # logical byte count directly for both ordinary and extended dtypes.
        nbytes = int(leaf.nbytes)
        elements += size
        total_bytes += nbytes
        descriptor = _sharding_descriptor(leaf)
        key = json.dumps(
            {
                "sharding_type": descriptor["sharding_type"],
                "partition_spec": descriptor["partition_spec"],
                "memory_kind": descriptor["memory_kind"],
                "device_count": descriptor["device_count"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        groups[key] += 1
        group_bytes[key] += nbytes
    return {
        "arrays": arrays,
        "elements": elements,
        "bytes": total_bytes,
        "groups": [
            {
                **json.loads(key),
                "arrays": groups[key],
                "bytes": group_bytes[key],
            }
            for key in sorted(groups)
        ],
    }


def _capture_descriptors(output: CaptureBatch) -> dict[str, Any]:
    return {
        name: _sharding_descriptor(getattr(output, name))
        for name in CaptureBatch._fields
    }


def _validate_capture_on_device(
    output: CaptureBatch, config: Gemma4Config, *, expected_rows: int
) -> dict[str, Any]:
    """Validate the global capture without copying its large activation to host."""

    expected_shapes = {
        "activation": (expected_rows, config.hidden_size),
        "expert_ids": (expected_rows, config.experts_per_token),
        "router_weights": (expected_rows, config.experts_per_token),
        "sequence_id": (expected_rows,),
        "position": (expected_rows,),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(int(size) for size in getattr(output, name).shape)
        if actual != expected:
            raise RuntimeError(f"{name} shape drifted: {actual} != {expected}")
    if str(output.activation.dtype) != "bfloat16":
        raise RuntimeError("profile capture activation is not bfloat16")
    if str(output.router_weights.dtype) != "bfloat16":
        raise RuntimeError("profile capture router_weights are not bfloat16")
    if str(output.expert_ids.dtype) != "uint8":
        raise RuntimeError("profile capture expert_ids are not uint8")

    import jax
    import jax.numpy as jnp

    ids = output.expert_ids.astype(jnp.int16)
    weights = output.router_weights.astype(jnp.float32)
    positions = output.position.astype(jnp.int32)
    sorted_ids = jnp.sort(ids, axis=1)
    router_error = jnp.max(jnp.abs(jnp.sum(weights, axis=1) - 1.0))
    scalars = jax.device_get(
        (
            jnp.all(jnp.isfinite(output.activation)),
            jnp.all(jnp.isfinite(weights)),
            jnp.min(ids),
            jnp.max(ids),
            jnp.all(jnp.diff(sorted_ids, axis=1) > 0),
            jnp.all(weights > 0.0),
            router_error,
            jnp.min(positions),
            jnp.max(positions),
        )
    )
    (
        activation_finite,
        weights_finite,
        min_id,
        max_id,
        unique_ids,
        positive_weights,
        max_router_error,
        min_position,
        max_position,
    ) = (np.asarray(value).item() for value in scalars)
    if not bool(activation_finite) or not bool(weights_finite):
        raise RuntimeError("profile capture contains a non-finite value")
    if int(min_id) < 0 or int(max_id) >= config.num_routed_experts:
        raise RuntimeError("profile expert IDs lie outside [0, 127]")
    if not bool(unique_ids):
        raise RuntimeError("profile capture contains duplicate routed experts")
    if not bool(positive_weights):
        raise RuntimeError("profile router weights are not strictly positive")
    if float(max_router_error) > 0.02:
        raise RuntimeError("profile router weights do not sum to one")
    if int(min_position) < 0 or int(max_position) >= config.sequence_length:
        raise RuntimeError("profile positions lie outside the sequence")
    return {
        "activation_finite": True,
        "router_weights_finite": True,
        "min_expert_id": int(min_id),
        "max_expert_id": int(max_id),
        "unique_experts_per_token": True,
        "strictly_positive_router_weights": True,
        "max_router_sum_error": float(max_router_error),
        "min_position": int(min_position),
        "max_position": int(max_position),
    }


def _allgather_rows(local: Sequence[float], *, process_count: int) -> np.ndarray:
    row = np.asarray(local, dtype=np.float64)
    if process_count == 1:
        return row.reshape(1, -1)
    from jax.experimental import multihost_utils

    return np.asarray(
        multihost_utils.process_allgather(row, tiled=False), dtype=np.float64
    ).reshape(process_count, len(row))


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _profile(args: argparse.Namespace) -> int:
    output_root = Path(args.output).resolve()
    _validate_diagnostic_destination(output_root)
    config = load_config(args.config)
    config.validate_production()
    mesh = _mesh_by_name(args.mesh)
    require_v4_32_runtime()
    process_index, process_count, global_chips = _process_context()
    config_sha256 = _coordinated_digest(
        config.config_hash, purpose="configuration", process_count=process_count
    )
    environment = environment_fingerprint(Path(__file__).resolve().parents[2])
    environment_sha256 = _coordinated_digest(
        str(environment["sha256"]),
        purpose="environment fingerprint",
        process_count=process_count,
    )
    checkpoint_sha256, checkpoint_seconds = _verify_checkpoint(
        args.checkpoint,
        args.checkpoint_sha256,
        process_count=process_count,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_root / "profile.json"
    local_exists = float(aggregate_path.exists())
    if np.any(_allgather_rows([local_exists], process_count=process_count)):
        raise RuntimeError(f"diagnostic profile already exists: {aggregate_path}")

    argv = build_maxtext_argv(
        checkpoint_path=args.checkpoint,
        grain_train_files="",
        grain_eval_files="",
        mesh=mesh,
        per_device_sequence_batch=args.per_device_sequence_batch,
        synthetic=True,
    )
    runtime_started = time.perf_counter()
    runtime = create_maxtext_runtime(argv)
    runtime_seconds = time.perf_counter() - runtime_started
    try:
        effective_kernel_config = _effective_kernel_config(runtime.config)
        kernel_config_digest = hashlib.sha256(
            json.dumps(
                effective_kernel_config, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        _coordinated_digest(
            kernel_config_digest,
            purpose="effective MaxText kernel configuration",
            process_count=process_count,
        )
        adapter = Gemma4PrefixAdapter.from_maxtext(runtime.model, config)
        step = compile_adapter_step(
            adapter, out_shardings=_capture_output_shardings(runtime.mesh)
        )
        source = iter(runtime.train_iterator)
        try:
            first_raw = next(source)
        except StopIteration as error:
            raise RuntimeError("MaxText returned an empty synthetic iterator") from error
        first = extract_grain_batch(
            first_raw, expected_sequence_length=config.sequence_length
        )

        with maxtext_mesh_context(runtime):
            compile_started = time.perf_counter()
            output = step(
                first.token_ids,
                first.positions,
                first.segment_ids,
                first.segment_ids,
            )
            _block_until_ready(output)
            compile_seconds = time.perf_counter() - compile_started

            parameter_started = time.perf_counter()
            parameter_shardings = _parameter_sharding_summary(runtime.model)
            parameter_summary_seconds = time.perf_counter() - parameter_started
            parameter_digest = hashlib.sha256(
                json.dumps(
                    parameter_shardings, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            _coordinated_digest(
                parameter_digest,
                purpose="parameter sharding summary",
                process_count=process_count,
            )

            batches = itertools.chain((first_raw,), source)
            warmup_seconds: list[float] = []
            for _ in range(args.warmup_steps):
                grain = extract_grain_batch(
                    next(batches), expected_sequence_length=config.sequence_length
                )
                started = time.perf_counter()
                output = step(
                    grain.token_ids,
                    grain.positions,
                    grain.segment_ids,
                    grain.segment_ids,
                )
                _block_until_ready(output)
                warmup_seconds.append(time.perf_counter() - started)

            cache_before = _jit_cache_size(step)
            measured_device_seconds: list[float] = []
            measured_input_seconds: list[float] = []
            last_grain = first
            for _ in range(args.measured_steps):
                input_started = time.perf_counter()
                last_grain = extract_grain_batch(
                    next(batches), expected_sequence_length=config.sequence_length
                )
                measured_input_seconds.append(time.perf_counter() - input_started)
                device_started = time.perf_counter()
                output = step(
                    last_grain.token_ids,
                    last_grain.positions,
                    last_grain.segment_ids,
                    last_grain.segment_ids,
                )
                _block_until_ready(output)
                measured_device_seconds.append(time.perf_counter() - device_started)
            cache_after = _jit_cache_size(step)

            trace_seconds = 0.0
            trace_path: str | None = None
            if args.trace_root is not None:
                import jax

                trace_directory = (
                    Path(args.trace_root).resolve()
                    / output_root.name
                    / f"host-{process_index:05d}"
                )
                trace_directory.mkdir(parents=True, exist_ok=True)
                trace_path = str(trace_directory)
                trace_started = time.perf_counter()
                with jax.profiler.trace(trace_path):
                    for _ in range(args.trace_steps):
                        grain = extract_grain_batch(
                            next(batches),
                            expected_sequence_length=config.sequence_length,
                        )
                        traced = step(
                            grain.token_ids,
                            grain.positions,
                            grain.segment_ids,
                            grain.segment_ids,
                        )
                        _block_until_ready(traced)
                trace_seconds = time.perf_counter() - trace_started

            validation_started = time.perf_counter()
            expected_rows = (
                args.per_device_sequence_batch
                * global_chips
                * config.sequence_length
            )
            validation = _validate_capture_on_device(
                output, config, expected_rows=expected_rows
            )
            validation_seconds = time.perf_counter() - validation_started
            input_shardings = {
                "token_ids": _sharding_descriptor(last_grain.token_ids),
                "positions": _sharding_descriptor(last_grain.positions),
                "segment_ids": _sharding_descriptor(last_grain.segment_ids),
            }
            output_shardings = _capture_descriptors(output)

        cache_before_value = -1 if cache_before is None else cache_before
        cache_after_value = -1 if cache_after is None else cache_after
        fixed_metrics = [
            checkpoint_seconds,
            runtime_seconds,
            compile_seconds,
            parameter_summary_seconds,
            sum(warmup_seconds),
            sum(measured_device_seconds),
            sum(measured_input_seconds),
            trace_seconds,
            validation_seconds,
            float(cache_before_value),
            float(cache_after_value),
        ]
        gathered = _allgather_rows(
            [*fixed_metrics, *measured_device_seconds, *measured_input_seconds],
            process_count=process_count,
        )
        fixed_count = len(fixed_metrics)
        device_rows = gathered[
            :, fixed_count : fixed_count + args.measured_steps
        ]
        input_rows = gathered[:, fixed_count + args.measured_steps :]
        global_step_seconds = np.max(device_rows, axis=0)
        global_device_seconds = float(np.sum(global_step_seconds))
        useful_flops_per_step = estimate_useful_prefix_flops(
            batch_sequences=args.per_device_sequence_batch * global_chips,
            sequence_length=config.sequence_length,
        )
        useful_mfu = (
            args.measured_steps * useful_flops_per_step
        ) / (V4_32_BF16_PEAK_FLOPS * global_device_seconds)
        host_records = []
        for index, row in enumerate(gathered):
            host_records.append(
                {
                    "process_index": index,
                    "checkpoint_inventory_seconds": float(row[0]),
                    "runtime_create_seconds": float(row[1]),
                    "compile_seconds": float(row[2]),
                    "parameter_summary_seconds": float(row[3]),
                    "warmup_device_seconds": float(row[4]),
                    "measured_device_seconds": float(row[5]),
                    "measured_input_seconds": float(row[6]),
                    "trace_seconds": float(row[7]),
                    "validation_seconds": float(row[8]),
                    "jit_cache_size_before": (
                        None if int(row[9]) < 0 else int(row[9])
                    ),
                    "jit_cache_size_after": (
                        None if int(row[10]) < 0 else int(row[10])
                    ),
                }
            )
        known_cache_sizes = [
            (int(row[9]), int(row[10]))
            for row in gathered
            if int(row[9]) >= 0 and int(row[10]) >= 0
        ]
        recompilations = max(
            (after - before for before, after in known_cache_sizes), default=0
        )
        if recompilations != 0:
            raise RuntimeError("adapter JIT cache grew during measured profile steps")

        local_record = {
            "schema_version": PROFILE_SCHEMA,
            "diagnostic_only": True,
            "eligible_for_benchmark_selection": False,
            "process_index": process_index,
            "process_count": process_count,
            "measured_device_step_seconds": measured_device_seconds,
            "measured_input_step_seconds": measured_input_seconds,
            "trace_path": trace_path,
            "validation": validation,
        }
        atomic_write_json(output_root / f"host-{process_index:05d}.json", local_record)

        if process_index == 0:
            mesh_shape = {
                str(name): int(size) for name, size in runtime.mesh.shape.items()
            }
            payload = {
                "schema_version": PROFILE_SCHEMA,
                "diagnostic_only": True,
                "eligible_for_benchmark_selection": False,
                "production_hard_gate_evaluated": False,
                "note": (
                    "Short diagnostic timing without persistence; rerun the full "
                    "200-step benchmark before selecting a production candidate."
                ),
                "config_sha256": config_sha256,
                "environment_sha256": environment_sha256,
                "checkpoint": args.checkpoint,
                "checkpoint_sha256": checkpoint_sha256,
                "mesh": {
                    "name": mesh.name,
                    "expert_parallelism": mesh.expert_parallelism,
                    "fsdp_parallelism": mesh.fsdp_parallelism,
                    "tensor_parallelism": mesh.tensor_parallelism,
                    "runtime_shape": mesh_shape,
                    "axis_names": [str(name) for name in runtime.mesh.axis_names],
                },
                "process_count": process_count,
                "global_chips": global_chips,
                "per_device_sequence_batch": args.per_device_sequence_batch,
                "sequence_length": config.sequence_length,
                "effective_kernel_config": effective_kernel_config,
                "effective_kernel_config_sha256": kernel_config_digest,
                "warmup_steps": args.warmup_steps,
                "measured_steps": args.measured_steps,
                "trace_steps": args.trace_steps if args.trace_root else 0,
                "useful_flops_per_step": useful_flops_per_step,
                "global_device_seconds": global_device_seconds,
                "device_tokens_per_second": (
                    args.measured_steps * expected_rows / global_device_seconds
                ),
                "diagnostic_useful_mfu": useful_mfu,
                "global_step_seconds": global_step_seconds.tolist(),
                "global_step_summary_seconds": _percentiles(global_step_seconds),
                "global_input_step_summary_seconds": _percentiles(
                    np.max(input_rows, axis=0)
                ),
                "recompilations": recompilations,
                "capture_validation": validation,
                "shardings": {
                    "inputs": input_shardings,
                    "outputs": output_shardings,
                    "parameters": parameter_shardings,
                },
                "hosts": host_records,
            }
            atomic_write_json(aggregate_path, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        _close_maxtext_runtime(runtime)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m moe_sae.profile_worker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", type=_sha256_argument, required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="new diagnostic directory; a path component must be named diagnostics",
    )
    parser.add_argument(
        "--mesh", choices=tuple(candidate.name for candidate in HARVEST_MESHES), required=True
    )
    parser.add_argument(
        "--per-device-sequence-batch",
        type=int,
        choices=PER_DEVICE_SEQUENCE_BATCHES,
        required=True,
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=10)
    parser.add_argument(
        "--trace-root",
        help="optional per-host JAX trace root; tracing runs after untimed measurements",
    )
    parser.add_argument("--trace-steps", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.warmup_steps <= MAX_DIAGNOSTIC_STEPS:
        raise SystemExit(
            f"--warmup-steps must lie in [1, {MAX_DIAGNOSTIC_STEPS}]"
        )
    if not 1 <= args.measured_steps <= MAX_DIAGNOSTIC_STEPS:
        raise SystemExit(
            f"--measured-steps must lie in [1, {MAX_DIAGNOSTIC_STEPS}]"
        )
    if not 1 <= args.trace_steps <= MAX_TRACE_STEPS:
        raise SystemExit(f"--trace-steps must lie in [1, {MAX_TRACE_STEPS}]")
    try:
        return _profile(args)
    except Exception as error:
        try:
            process_index, _, _ = _process_context()
        except Exception:
            process_index = 0
        try:
            output_root = Path(args.output).resolve()
            _validate_diagnostic_destination(output_root)
            atomic_write_json(
                output_root / "failures" / f"host-{process_index:05d}.json",
                {
                    "schema_version": "gemma4-prefix-profile-failure-v1",
                    "diagnostic_only": True,
                    "eligible_for_benchmark_selection": False,
                    "process_index": process_index,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        except Exception:
            pass
        print(
            f"prefix profile failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
