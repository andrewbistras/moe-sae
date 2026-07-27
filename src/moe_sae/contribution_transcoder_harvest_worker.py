"""Executable v4-32 worker for the bounded Experiment-3 source harvest.

Launch the same command on all four TPU VM workers.  Each JAX process derives
its lane from the sealed contribution target, replays the exact source-v1
stream, publishes only its owned overlay ranges, and finalizes its host index.
Process zero seals the manifest after a global barrier.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import tempfile
import time
from typing import Any, Mapping, Sequence

from moe_sae.artifacts import (
    ArtifactDataset,
    load_manifest,
    verify_staged_dataset_manifest,
)
from moe_sae.contribution_artifacts import canonical_json_bytes, canonical_sha256
from moe_sae.contribution_maxtext import build_contribution_maxtext_argv
from moe_sae.contribution_storage import load_contribution_manifest
from moe_sae.contribution_transcoder_artifacts import RoutedInputMetadata
from moe_sae.contribution_transcoder_config import (
    load_contribution_transcoder_config,
)
from moe_sae.contribution_transcoder_harvest import (
    _process_target_entries,
    harvest_routed_input_host,
)
from moe_sae.contribution_transcoder_model import (
    Gemma4RoutedInputPrefixAdapter,
    RoutedInputCaptureBatch,
)
from moe_sae.contribution_transcoder_storage import (
    RoutedInputShardWriter,
    load_routed_input_manifest,
    publish_routed_input_manifest,
)
from moe_sae.harvest import (
    _close_maxtext_runtime,
    compile_adapter_step,
    extract_grain_batch,
)
from moe_sae.model import checkpoint_inventory_sha256
from moe_sae.object_store import (
    GCSObjectStore,
    ShardCache,
    object_identity_from_mapping,
)
from moe_sae.runtime import (
    HARVEST_MESHES,
    create_maxtext_runtime,
    environment_fingerprint,
    initialize_jax_distributed,
    maxtext_mesh_context,
)


MAXIMUM_RUNTIME_SECONDS = 600


class RoutedInputHarvestWorkerError(RuntimeError):
    """The bounded TPU worker failed before sealing a valid overlay."""


def _mesh(name: str):
    for candidate in HARVEST_MESHES:
        if candidate.name == name:
            return candidate
    raise RoutedInputHarvestWorkerError(f"unknown harvest mesh {name!r}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(dict(value)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _replay_first(first: Any, iterator: Any):
    yield first
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return


def _output_shardings(mesh: Any) -> RoutedInputCaptureBatch:
    try:
        from jax.sharding import NamedSharding, PartitionSpec
    except ImportError as error:
        raise RoutedInputHarvestWorkerError(
            "TPU output sharding requires JAX"
        ) from error
    row_axes = tuple(mesh.axis_names)
    if not row_axes:
        raise RoutedInputHarvestWorkerError("MaxText mesh has no named axes")
    return RoutedInputCaptureBatch(
        routed_input=NamedSharding(mesh, PartitionSpec(row_axes, None)),
        sequence_id=NamedSharding(mesh, PartitionSpec(row_axes)),
        position=NamedSharding(mesh, PartitionSpec(row_axes)),
    )


def _compile_source_step(
    adapter: Gemma4RoutedInputPrefixAdapter,
    mesh: Any,
    first_batch: Any,
):
    step = compile_adapter_step(adapter, out_shardings=_output_shardings(mesh))
    transformed = getattr(step, "_compiled_step", None)
    lower = getattr(transformed, "lower", None)
    if not callable(lower):
        raise RoutedInputHarvestWorkerError(
            "source prefix transform exposes no lower() boundary"
        )
    grain = extract_grain_batch(first_batch, expected_sequence_length=1_024)
    lowered = lower(
        grain.token_ids,
        grain.positions,
        grain.segment_ids,
        grain.segment_ids,
    )
    compile_method = getattr(lowered, "compile", None)
    if not callable(compile_method):
        raise RoutedInputHarvestWorkerError(
            "lowered source prefix exposes no compile() boundary"
        )
    return compile_method()


def _batch_collective(
    local_complete_after_batch: bool,
    local_needs_execution: bool,
) -> tuple[bool, bool]:
    import numpy as np
    from jax.experimental import multihost_utils

    local = np.asarray(
        [local_complete_after_batch, local_needs_execution], dtype=np.uint8
    )
    gathered = np.asarray(
        multihost_utils.process_allgather(local, tiled=False), dtype=np.uint8
    ).reshape((-1, 2))
    return bool(np.all(gathered[:, 0])), bool(np.any(gathered[:, 1]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m moe_sae.contribution_transcoder_harvest_worker"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--grain-files", required=True)
    parser.add_argument("--staged-manifest", required=True)
    parser.add_argument("--source-v1-root", required=True)
    parser.add_argument("--source-cache-root", required=True)
    parser.add_argument("--source-cache-max-bytes", type=int, required=True)
    parser.add_argument("--target-control-root", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--local-buffer-root", required=True)
    parser.add_argument("--object-prefix", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--jax-cache-dir", required=True)
    parser.add_argument(
        "--mesh",
        choices=tuple(candidate.name for candidate in HARVEST_MESHES),
        required=True,
    )
    parser.add_argument("--per-device-sequence-batch", type=int, required=True)
    parser.add_argument(
        "--maximum-runtime-seconds",
        type=int,
        default=MAXIMUM_RUNTIME_SECONDS,
    )
    return parser


def run_worker(arguments: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if not 0 < arguments.maximum_runtime_seconds <= MAXIMUM_RUNTIME_SECONDS:
        raise RoutedInputHarvestWorkerError(
            "source harvest runtime must lie in [1,600] seconds"
        )
    config = load_contribution_transcoder_config(arguments.config)
    target = load_contribution_manifest(arguments.target_control_root)
    expected_parent = config.parents[
        f"{arguments.split}_target_manifest_sha256"
    ]
    expected_tokens = config.parents[f"{arguments.split}_tokens"]
    if (
        target["manifest_sha256"] != expected_parent
        or target["total_tokens"] != expected_tokens
    ):
        raise RoutedInputHarvestWorkerError(
            "target manifest differs from the frozen Experiment-3 parent"
        )
    checkpoint = checkpoint_inventory_sha256(arguments.checkpoint)
    if checkpoint != arguments.checkpoint_sha256:
        raise RoutedInputHarvestWorkerError("checkpoint inventory SHA-256 changed")

    initialize_jax_distributed()
    import jax
    from jax.experimental import multihost_utils

    process_count = int(jax.process_count())
    process_index = int(jax.process_index())
    if process_count != 4 or int(jax.device_count()) != 16:
        raise RoutedInputHarvestWorkerError(
            "source harvest requires the allocated four-host v4-32"
        )
    object_store = GCSObjectStore()
    provenance = verify_staged_dataset_manifest(
        arguments.staged_manifest,
        arguments.grain_files,
        expected_split=(
            config.raw["provenance"]["train_split"]
            if arguments.split == "train"
            else config.raw["provenance"]["eval_split"]
        ),
        expected_tokens=expected_tokens,
        tokenizer_id=config.raw["provenance"]["model_id"],
        tokenizer_revision=config.raw["provenance"]["model_revision"],
        sequence_length=1_024,
        verify_file_hashes=False,
        verify_arrayrecord_count=False,
    )
    for entry in provenance.manifest["files"]:
        identity = entry.get("object")
        if identity is not None:
            object_store.verify(object_identity_from_mapping(identity))
    if provenance.fingerprint_sha256 != target["metadata"]["dataset_fingerprint"]:
        raise RoutedInputHarvestWorkerError(
            "staged data fingerprint differs from the sealed target"
        )

    source_manifest = load_manifest(arguments.source_v1_root)
    if (
        source_manifest["manifest_sha256"]
        != target["source_v1_manifest"]["manifest_sha256"]
    ):
        raise RoutedInputHarvestWorkerError(
            "source-v1 root differs from the target's parent"
        )
    remote = any("object" in entry for entry in source_manifest["shards"])
    source_cache = (
        ShardCache(
            arguments.source_cache_root,
            object_store,
            max_bytes=arguments.source_cache_max_bytes,
        )
        if remote
        else None
    )
    source = ArtifactDataset(
        arguments.source_v1_root,
        verify_hashes=True,
        verify_statistics=True,
        decode_bf16=False,
        cache=source_cache,
    )
    environment = environment_fingerprint(Path(arguments.repo_root))
    metadata = RoutedInputMetadata(
        checkpoint_sha256=checkpoint,
        maxtext_fork_commit=config.provenance.maxtext_fork_commit,
        dataset_fingerprint=provenance.fingerprint_sha256,
        split=arguments.split,
        config_sha256=config.config_hash,
        environment_sha256=environment["sha256"],
        target_manifest_sha256=target["manifest_sha256"],
    )
    host_id, _entries = _process_target_entries(
        target, process_count=process_count, process_index=process_index
    )
    writer = RoutedInputShardWriter(
        arguments.control_root,
        host_id,
        metadata,
        target_manifest=target,
        local_buffer_root=arguments.local_buffer_root,
        object_prefix=arguments.object_prefix,
        object_store=object_store,
    )

    mesh = _mesh(arguments.mesh)
    argv = build_contribution_maxtext_argv(
        checkpoint_path=arguments.checkpoint,
        grain_train_files=arguments.grain_files,
        grain_eval_files=arguments.grain_files,
        mesh=mesh,
        per_device_sequence_batch=arguments.per_device_sequence_batch,
        attention=str(config.raw["model"]["attention"]),
        jax_cache_dir=arguments.jax_cache_dir,
        synthetic=False,
    )
    runtime = create_maxtext_runtime(
        argv,
        packed_train_files=tuple(provenance.files),
        packed_eval_files=tuple(provenance.files),
        packed_lane_prefix_replay_records=4,
    )
    body_error: BaseException | None = None
    try:
        adapter = Gemma4RoutedInputPrefixAdapter.from_installed_maxtext(
            runtime.model
        )
        iterator_value = (
            runtime.train_iterator
            if arguments.split == "train"
            else runtime.eval_iterator
        )
        if iterator_value is None:
            raise RoutedInputHarvestWorkerError("MaxText built no selected iterator")
        iterator = iter(iterator_value)
        try:
            first = next(iterator)
        except StopIteration as error:
            raise RoutedInputHarvestWorkerError("staged iterator is empty") from error
        with maxtext_mesh_context(runtime):
            compile_started = time.perf_counter()
            executable = _compile_source_step(adapter, runtime.mesh, first)
            compile_seconds = time.perf_counter() - compile_started
            remaining = arguments.maximum_runtime_seconds - (
                time.perf_counter() - started
            )
            if remaining <= 0:
                raise RoutedInputHarvestWorkerError(
                    "compilation exhausted the 600-second authorization"
                )
            result = harvest_routed_input_host(
                _replay_first(first, iterator),
                executable,
                writer=writer,
                source_v1_dataset=source,
                target_manifest=target,
                process_count=process_count,
                process_index=process_index,
                maximum_runtime_seconds=remaining,
                batch_collective=_batch_collective,
            )
        host_evidence_unsigned = {
            "schema_version": "gemma4-routed-input-harvest-worker-v1",
            "config_sha256": config.config_hash,
            "environment_sha256": environment["sha256"],
            "checkpoint_sha256": checkpoint,
            "compile_seconds": compile_seconds,
            "result": result.to_dict(),
        }
        host_evidence = {
            **host_evidence_unsigned,
            "evidence_sha256": canonical_sha256(host_evidence_unsigned),
        }
        _atomic_json(
            Path(arguments.control_root)
            / "evidence"
            / f"host-{process_index}.json",
            host_evidence,
        )
        multihost_utils.sync_global_devices("exp3-overlay-hosts-finalized")
        if process_index == 0:
            publish_routed_input_manifest(
                arguments.control_root,
                metadata,
                target_manifest=target,
                object_prefix=arguments.object_prefix,
                object_store=object_store,
                expected_hosts=process_count,
            )
        multihost_utils.sync_global_devices("exp3-overlay-manifest-sealed")
        manifest = load_routed_input_manifest(
            arguments.control_root, target_manifest=target
        )
        if time.perf_counter() - started > arguments.maximum_runtime_seconds:
            raise RoutedInputHarvestWorkerError(
                "complete worker exceeded its 600-second authorization"
            )
        return {
            "status": "complete",
            "process_index": process_index,
            "manifest_sha256": manifest["manifest_sha256"],
            "target_manifest_sha256": target["manifest_sha256"],
            "total_tokens": manifest["total_tokens"],
        }
    except BaseException as error:
        body_error = error
        raise
    finally:
        try:
            _close_maxtext_runtime(runtime)
        except BaseException as error:
            if body_error is None:
                raise
            try:
                body_error.add_note(f"runtime cleanup also failed: {error}")
            except AttributeError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)

    def deadline(_signum, _frame):
        raise TimeoutError("Experiment-3 source harvest hit its 600-second deadline")

    prior_handler = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, arguments.maximum_runtime_seconds)
    try:
        result = run_worker(arguments)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
