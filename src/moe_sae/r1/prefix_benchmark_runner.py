"""One-command synthetic v6e-64 qualification for the R1 prefix."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

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
    result = run_synchronized_prefix_benchmark(
        compiled,
        batch,
        packed_shape=PackedStepShape.full_sequences(),
        synchronize=synchronize_prefix_output,
        reduce_max_durations_ns=reduce_max_durations_across_jax_processes,
    )
    payload = {
        "schema": "moe-sae.r1.synthetic-prefix-qualification.v1",
        "mesh": mesh_candidate.name,
        "seed": args.seed,
        "process_count": 16,
        "device_count": 64,
        "prepare_seconds": prepare_ns / 1_000_000_000,
        "output_validation": validation,
        "benchmark": result.as_dict(),
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
