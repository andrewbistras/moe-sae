"""Private fresh-process workers for the multi-host harvest benchmark.

The public ``benchmark-harvest`` command is deliberately a JAX-cold
supervisor.  This module is its exec boundary: every invocation initializes
one distributed JAX runtime, performs one bounded action, and exits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from moe_sae.cli import (
    _aggregate_benchmark_observation,
    _artifact_root,
    _coordinated_environment,
    _coordinator_write_json,
    _global_barrier,
    _load_conversion_attestation,
    _load_storage_attestation,
    _load_storage_attestation_payload,
    _mesh_by_name,
    _new_run_instance_id,
    _process_context,
    _require_production_artifact_root,
    _sha256_argument,
    _storage_layout,
    _verify_checkpoint_inventory,
)
from moe_sae.config import load_config
from moe_sae.object_store import join_gcs_uri
from moe_sae.runtime import atomic_write_json, require_v4_32_runtime


def _paths(args: argparse.Namespace) -> tuple[Any, Path, Path]:
    config = load_config(args.config)
    config.validate_production()
    artifact_root = _artifact_root(config, args.artifact_root).resolve()
    _require_production_artifact_root(artifact_root)
    output = Path(args.output).resolve()
    try:
        output.relative_to(artifact_root)
    except ValueError as error:
        raise RuntimeError(
            f"benchmark worker output must be below {artifact_root}"
        ) from error
    output.mkdir(parents=True, exist_ok=True)
    return config, artifact_root, output


def _bootstrap(args: argparse.Namespace) -> int:
    require_v4_32_runtime()
    config, artifact_root, output = _paths(args)
    _verify_checkpoint_inventory(args)
    conversion_attestation_sha256 = _load_conversion_attestation(
        artifact_root, config, args.checkpoint_sha256
    )
    environment = _coordinated_environment()
    storage_attestation_sha256 = _load_storage_attestation(
        artifact_root, config, environment=environment
    )
    process_index, process_count = _process_context()
    sweep_id = _new_run_instance_id()
    sweep_root = output / ".supervisor" / "sweeps" / sweep_id
    mapping_path = (
        output / ".supervisor" / "bootstrap" / f"{args.supervisor_id}.json"
    )
    atomic_write_json(
        mapping_path,
        {
            "schema_version": "gemma4-benchmark-bootstrap-map-v1",
            "sweep_id": sweep_id,
            "process_index": process_index,
            "process_count": process_count,
        },
    )
    _global_barrier(f"gemma4-benchmark-bootstrap-maps-{sweep_id}")
    _coordinator_write_json(
        sweep_root / "bootstrap.json",
        {
            "schema_version": "gemma4-benchmark-bootstrap-v1",
            "sweep_id": sweep_id,
            "config_sha256": config.config_hash,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "conversion_attestation_sha256": conversion_attestation_sha256,
            "checkpoint_verification_seconds": getattr(
                args, "_checkpoint_verification_seconds", 0.0
            ),
            "environment": environment,
            "storage_attestation_sha256": storage_attestation_sha256,
            "process_count": process_count,
        },
        barrier=f"gemma4-benchmark-bootstrap-record-{sweep_id}",
    )
    return 0


def _candidate(args: argparse.Namespace) -> int:
    process_index = args.expected_process_index
    try:
        require_v4_32_runtime()
        config, artifact_root, output = _paths(args)
        actual_index, actual_count = _process_context()
        process_index = actual_index
        if (
            actual_index != args.expected_process_index
            or actual_count != args.expected_process_count
        ):
            raise RuntimeError("candidate worker TPU controller identity changed")
        # The checkpoint path is intentionally verified again in every fresh
        # process.  A bootstrap proof cannot make a mutable local/GCS prefix
        # immutable across a nine-candidate sweep.
        _verify_checkpoint_inventory(args)
        bootstrap = json.loads(
            Path(args.bootstrap_record).read_text(encoding="utf-8")
        )
        if not isinstance(bootstrap, dict):
            raise RuntimeError("benchmark bootstrap record must be an object")
        environment = _coordinated_environment()
        storage_payload = _load_storage_attestation_payload(
            artifact_root, config, environment=environment
        )
        storage_attestation_sha256 = str(storage_payload["attestation_sha256"])
        storage_layout = _storage_layout(storage_payload)
        conversion_attestation_sha256 = _load_conversion_attestation(
            artifact_root, config, args.checkpoint_sha256
        )
        if (
            bootstrap.get("schema_version") != "gemma4-benchmark-bootstrap-v1"
            or bootstrap.get("sweep_id") != args.sweep_id
            or bootstrap.get("config_sha256") != config.config_hash
            or bootstrap.get("checkpoint") != str(args.checkpoint)
            or bootstrap.get("checkpoint_sha256") != args.checkpoint_sha256
            or bootstrap.get("conversion_attestation_sha256")
            != conversion_attestation_sha256
            or bootstrap.get("storage_attestation_sha256")
            != storage_attestation_sha256
            or int(bootstrap.get("process_count", 0)) != actual_count
            or not isinstance(bootstrap.get("environment"), dict)
            or bootstrap["environment"].get("sha256")
            != environment.get("sha256")
        ):
            raise RuntimeError("candidate worker does not match its bootstrap record")

        from moe_sae.harvest import benchmark_production_candidate

        mesh = _mesh_by_name(args.mesh)
        observation = benchmark_production_candidate(
            checkpoint_path=args.checkpoint,
            mesh_candidate=mesh,
            per_device_sequence_batch=args.per_device_sequence_batch,
            config=config,
            measured_steps=max(args.steps, config.benchmark_steps),
            output_directory=output,
            storage_attestation_sha256=storage_attestation_sha256,
            object_prefix=join_gcs_uri(
                storage_layout["bulk_gcs_root"],
                "benchmarks",
                config.config_hash,
                args.sweep_id,
                f"{args.mesh}-b{args.per_device_sequence_batch}",
            ),
            local_buffer_root=storage_layout["local_buffer_root"],
        )
        observation = _aggregate_benchmark_observation(observation)
        _coordinator_write_json(
            Path(args.attempt_root) / "raw.json",
            {
                "schema_version": "gemma4-benchmark-worker-result-v2",
                "sweep_id": args.sweep_id,
                "attempt_id": args.attempt_id,
                "observation": observation.as_dict(),
            },
            barrier=f"gemma4-benchmark-worker-{args.sweep_id}-{args.attempt_id}",
        )
        return 0
    except Exception as error:
        atomic_write_json(
            Path(args.attempt_root)
            / "failures"
            / f"host-{process_index:05d}.json",
            {
                "schema_version": "gemma4-benchmark-failure-v1",
                "sweep_id": args.sweep_id,
                "attempt_id": args.attempt_id,
                "mesh": args.mesh,
                "per_device_sequence_batch": args.per_device_sequence_batch,
                "process_index": process_index,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        print(
            f"benchmark candidate failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m moe_sae.benchmark_worker")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--config", required=True)
        target.add_argument("--artifact-root", required=True)
        target.add_argument("--checkpoint", required=True)
        target.add_argument(
            "--checkpoint-sha256", type=_sha256_argument, required=True
        )
        target.add_argument("--output", required=True)

    bootstrap = subparsers.add_parser("bootstrap")
    common(bootstrap)
    bootstrap.add_argument("--supervisor-id", required=True)
    bootstrap.set_defaults(handler=_bootstrap)

    candidate = subparsers.add_parser("candidate")
    common(candidate)
    candidate.add_argument("--bootstrap-record", required=True)
    candidate.add_argument("--sweep-id", required=True)
    candidate.add_argument("--attempt-id", required=True)
    candidate.add_argument("--attempt-root", required=True)
    candidate.add_argument("--mesh", required=True)
    candidate.add_argument("--per-device-sequence-batch", type=int, required=True)
    candidate.add_argument("--steps", type=int, required=True)
    candidate.add_argument("--expected-process-index", type=int, required=True)
    candidate.add_argument("--expected-process-count", type=int, required=True)
    candidate.set_defaults(handler=_candidate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
