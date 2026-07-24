"""Collective, identity-bound execution of one Gemma Gate-1 matrix cell.

This module is deliberately narrower than the production evaluator.  It
restores one immutable historical checkpoint, evaluates one protocol-declared
split/route cell on the original 4x4 SAE mesh, and writes one deterministic
JSON result per JAX process.  It has no command-line, GCS-control, or report
publication surface.

The checkpoint restore configuration and the diagnostic scoring
configuration are intentionally distinct for ``all_banks``:

* the checkpoint is always restored with its original condition and training
  capacity; and
* an A checkpoint may then be scored with F visibility because A and F have
  the same total dictionary width and parameter layout.

For true/shuffled A scoring, capacity is measured from the exact transformed
static local microbatches, max-reduced across all four hosts, expanded by 10%,
and rounded to a multiple of eight.  Overflow is fatal.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import jax
import numpy as np

from moe_sae.artifacts import (
    DEFAULT_CAPTURE_SITE,
    HIDDEN_SIZE,
    ArtifactDataset,
    bf16_bits_to_float32,
)
from moe_sae.artifacts import (
    SCHEMA_VERSION as CAPTURE_SCHEMA_VERSION,
)
from moe_sae.audit_checkpoint import restore_audit_checkpoint_payload
from moe_sae.config import Gemma4Config
from moe_sae.diagnostics import (
    SequenceSSEAccumulator,
    apply_route_mode,
    merge_sequence_sse_summaries,
)
from moe_sae.distributed import (
    DistributedBatch,
    DistributedConfig,
    _checkpoint_phase_decision,
    create_mesh,
    init_distributed_state,
)
from moe_sae.gate1_distributed import (
    Gate1DistributedScorer,
    process_local_feature_state,
    process_local_token_scores,
)
from moe_sae.gate1_protocol import (
    GATE1_ROUTE_MODES,
    Gate1Protocol,
    SplitSpec,
    TokenRange,
    validate_gate1_protocol,
)
from moe_sae.pipeline import TrainingConfig
from moe_sae.runtime import (
    atomic_write_json,
    environment_fingerprint,
    initialize_jax_distributed,
)
from moe_sae.sae import SAEConfig

GATE1_PROCESS_RESULT_SCHEMA = "gemma4-gate1-process-result-v1"
GATE1_CELL_RESULT_SCHEMA = "gemma4-gate1-cell-result-v1"
GATE1_DATA_AXIS_SIZE = 4
GATE1_FEATURE_AXIS_SIZE = 4
GATE1_PROCESS_COUNT = 4
GATE1_LOCAL_DEVICE_COUNT = 4
GATE1_GLOBAL_DEVICE_COUNT = 16
GATE1_MICROBATCHES_PER_GROUP = 4
GATE1_LOCAL_MICROBATCH_TOKENS = 1_024
ORIGINAL_A_TRAINING_CAPACITY = 1_128
A_CAPACITY_MARGIN = 0.10
A_CAPACITY_ROUND_TO = 8
_MAX_EXISTING_RESULT_BYTES = 64 * 1024 * 1024
_VISIBILITY_TIMEOUT_SECONDS = 90.0
_VISIBILITY_POLL_SECONDS = 0.25

RouteMode = Literal["true", "shuffled", "all_banks"]


class Gate1AuditError(RuntimeError):
    """One Gate-1 cell cannot be executed without violating its contract."""


@dataclass(frozen=True, slots=True)
class Gate1AuditCell:
    """One exact member of the protocol-declared audit matrix."""

    checkpoint_label: str
    split_name: str
    route_mode: RouteMode

    def to_mapping(self) -> dict[str, str]:
        return {
            "checkpoint_label": self.checkpoint_label,
            "split": self.split_name,
            "route_mode": self.route_mode,
        }


@dataclass(frozen=True, slots=True)
class Gate1Runtime:
    """Validated physical runtime needed by the collective scorer."""

    mesh: Any
    distributed: DistributedConfig
    process_index: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "process_count": GATE1_PROCESS_COUNT,
            "process_index": self.process_index,
            "local_device_count": GATE1_LOCAL_DEVICE_COUNT,
            "global_device_count": GATE1_GLOBAL_DEVICE_COUNT,
            "mesh_shape": {
                self.distributed.data_axis: GATE1_DATA_AXIS_SIZE,
                self.distributed.feature_axis: GATE1_FEATURE_AXIS_SIZE,
            },
            "platform": "tpu",
        }


@dataclass(frozen=True, slots=True)
class LocalAuditGroup:
    """Four static process-local microbatches and their cluster identities."""

    batch: DistributedBatch
    sequence_ids: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class CapacityEvidence:
    """Exact transformed-route capacity derivation."""

    local_observed_max: int
    global_observed_max: int
    capacity: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "margin": A_CAPACITY_MARGIN,
            "round_to": A_CAPACITY_ROUND_TO,
            "local_observed_max": self.local_observed_max,
            "global_observed_max": self.global_observed_max,
            "capacity": self.capacity,
        }


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Gate1AuditError("Gate-1 result identity is not canonical JSON") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@lru_cache(maxsize=1)
def _consumer_environment() -> dict[str, Any]:
    """Bind reuse to runtime packages, devices, and the scoring source closure."""

    package = Path(__file__).resolve().parent
    source_names = (
        "artifacts.py",
        "audit_checkpoint.py",
        "config.py",
        "diagnostics.py",
        "distributed.py",
        "gate1_audit.py",
        "gate1_distributed.py",
        "gate1_protocol.py",
        "pipeline.py",
        "sae.py",
    )
    sources: dict[str, str] = {}
    for name in source_names:
        path = package / name
        try:
            sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise Gate1AuditError(
                f"cannot hash Gate-1 source dependency: {name}"
            ) from error
    environment = environment_fingerprint()
    supplied_sha256 = environment.pop("sha256", None)
    if not isinstance(supplied_sha256, str) or len(supplied_sha256) != 64:
        raise Gate1AuditError("runtime environment fingerprint is not bound")
    identity = {
        "environment": environment,
        "environment_sha256": supplied_sha256,
        "source_sha256": sources,
    }
    identity["sha256"] = _sha256(identity)
    return identity


def _matrix_cells(protocol: Gate1Protocol) -> frozenset[Gate1AuditCell]:
    matrix = protocol.matrix
    cells = {
        Gate1AuditCell(label, matrix.learning_curve.split, "true")
        for label in matrix.learning_curve.checkpoints
    }
    cells.update(
        Gate1AuditCell(label, split, "true")
        for label in matrix.final_baselines.checkpoints
        for split in matrix.final_baselines.splits
    )
    cells.update(
        Gate1AuditCell(label, matrix.router_ablation.split, mode)
        for label in matrix.router_ablation.checkpoints
        for mode in matrix.router_ablation.route_modes
    )
    return frozenset(cells)


def _validated_protocol_and_cell(
    protocol: Gate1Protocol,
    *,
    checkpoint_label: str,
    split_name: str,
    route_mode: str,
) -> tuple[Gate1Protocol, Gate1AuditCell]:
    if type(protocol) is not Gate1Protocol:
        raise Gate1AuditError("protocol must be a validated Gate1Protocol")
    # Revalidate the public dataclass at the execution boundary so constructing
    # it directly cannot bypass the frozen protocol schema.
    try:
        canonical = validate_gate1_protocol(protocol.to_mapping())
    except Exception as error:
        raise Gate1AuditError(
            "Gate-1 protocol failed execution-time validation"
        ) from error
    if canonical.sha256 != protocol.sha256:
        raise Gate1AuditError("Gate-1 protocol changed during validation")
    if (
        not isinstance(checkpoint_label, str)
        or not isinstance(split_name, str)
        or route_mode not in GATE1_ROUTE_MODES
    ):
        raise Gate1AuditError("checkpoint, split, and route must be canonical strings")
    cell = Gate1AuditCell(
        checkpoint_label=checkpoint_label,
        split_name=split_name,
        route_mode=route_mode,  # type: ignore[arg-type]
    )
    if cell not in _matrix_cells(canonical):
        raise Gate1AuditError(
            "requested checkpoint/split/route is not a declared Gate-1 matrix cell"
        )
    return canonical, cell


def _sae_config(config: Gemma4Config, condition: str) -> SAEConfig:
    return SAEConfig(
        condition=condition,
        hidden_dim=config.hidden_size,
        num_experts=config.num_routed_experts,
        top_routed=config.experts_per_token,
        bank_width=config.bank_width,
        k=config.topk_features,
        parameter_dtype=config.dtype,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_eps=config.adam_epsilon,
        grad_clip=config.grad_clip,
        peak_lr=config.learning_rate,
        min_lr=config.min_learning_rate,
        warmup_steps=config.warmup_updates,
        total_steps=config.planned_updates,
    )


def _original_training_config(
    protocol: Gate1Protocol,
    cell: Gate1AuditCell,
) -> TrainingConfig:
    checkpoint = protocol.checkpoints[cell.checkpoint_label]
    config = Gemma4Config()
    if config.config_hash != protocol.capture.config_sha256:
        raise Gate1AuditError("protocol does not identify the current Gemma config")
    training = TrainingConfig(
        sae=_sae_config(config, checkpoint.condition),
        seed=checkpoint.seed,
        total_tokens=config.train_tokens,
        effective_batch_tokens=config.effective_batch_tokens,
        microbatch_tokens=config.microbatch_tokens,
        checkpoint_every_tokens=config.checkpoint_every_tokens,
        health_gate_tokens=config.health_gate_tokens,
        live_feature_fraction=config.min_live_feature_fraction,
        a_capacity=(
            ORIGINAL_A_TRAINING_CAPACITY if checkpoint.condition == "A" else None
        ),
    )
    if (
        _sha256(dataclasses.asdict(training))
        != checkpoint.expectation.training_config_sha256
    ):
        raise Gate1AuditError(
            "reconstructed TrainingConfig does not identify the checkpoint payload"
        )
    return training


def _scoring_sae_config(
    training: TrainingConfig,
    route_mode: RouteMode,
) -> SAEConfig:
    if route_mode != "all_banks":
        return training.sae
    if training.sae.condition != "A":
        raise Gate1AuditError("all_banks is defined only for an A checkpoint")
    scoring = dataclasses.replace(training.sae, condition="F")
    if (
        scoring.total_features != training.sae.total_features
        or scoring.parameter_count != training.sae.parameter_count
    ):
        raise Gate1AuditError("A/F all-bank parameter layouts are not identical")
    return scoring


def _manifest_identity(
    dataset: ArtifactDataset,
    *,
    expected_sha256: str,
    expected_tokens: int,
    expected_split: str,
    protocol: Gate1Protocol,
    label: str,
) -> dict[str, Any]:
    if type(dataset) is not ArtifactDataset:
        raise Gate1AuditError(f"{label} must be an exact ArtifactDataset")
    manifest = dataset.manifest
    if not isinstance(manifest, Mapping):
        raise Gate1AuditError(f"{label} manifest is not a mapping")
    metadata = manifest.get("metadata")
    statistics = manifest.get("statistics")
    if not isinstance(metadata, Mapping) or not isinstance(statistics, Mapping):
        raise Gate1AuditError(f"{label} manifest lacks metadata/statistics")
    expected = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "sealed": True,
        "manifest_sha256": expected_sha256,
        "total_tokens": expected_tokens,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise Gate1AuditError(f"{label} manifest {name} mismatch")
    unsigned_manifest = {
        name: value for name, value in manifest.items() if name != "manifest_sha256"
    }
    if _sha256(unsigned_manifest) != expected_sha256:
        raise Gate1AuditError(f"{label} manifest content hash mismatch")
    metadata_sha256 = manifest.get("metadata_sha256")
    if (
        not isinstance(metadata_sha256, str)
        or _sha256(dict(metadata)) != metadata_sha256
    ):
        raise Gate1AuditError(f"{label} capture metadata hash mismatch")
    if dataset.manifest_sha256 != expected_sha256:
        raise Gate1AuditError(f"{label} ArtifactDataset identity mismatch")
    if dataset.total_tokens != expected_tokens:
        raise Gate1AuditError(f"{label} ArtifactDataset token count mismatch")
    config = Gemma4Config()
    metadata_expected = {
        "config_sha256": protocol.capture.config_sha256,
        "checkpoint_sha256": protocol.capture.checkpoint_sha256,
        "split": expected_split,
        "maxtext_commit": config.maxtext_commit,
        "target_layer_one_indexed": config.target_layer_one_indexed,
        "capture_site": DEFAULT_CAPTURE_SITE,
    }
    for name, value in metadata_expected.items():
        if metadata.get(name) != value:
            raise Gate1AuditError(f"{label} capture metadata {name} mismatch")
    if statistics.get("count") != expected_tokens:
        raise Gate1AuditError(f"{label} sufficient-statistics count mismatch")
    return {
        "manifest_sha256": expected_sha256,
        "total_tokens": expected_tokens,
        "metadata_sha256": manifest.get("metadata_sha256"),
        "config_sha256": metadata["config_sha256"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "split": metadata["split"],
        "maxtext_commit": metadata["maxtext_commit"],
        "target_layer_one_indexed": metadata["target_layer_one_indexed"],
        "capture_site": metadata["capture_site"],
    }


def _path_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_datasets_and_output(
    protocol: Gate1Protocol,
    train_dataset: ArtifactDataset,
    eval_dataset: ArtifactDataset,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, np.ndarray]:
    if not isinstance(output_root, Path):
        raise Gate1AuditError("shared output root must be a pathlib.Path")
    if str(output_root).startswith(("gs:", "s3:")):
        raise Gate1AuditError("Gate-1 output must be a local shared POSIX path")
    expanded_output = output_root.expanduser()
    if expanded_output.is_symlink():
        raise Gate1AuditError("shared output root may not be a symlink")
    output = expanded_output.resolve(strict=False)
    if output.exists() and not output.is_dir():
        raise Gate1AuditError(
            "shared output root must be a directory and may not be a symlink"
        )
    train_identity = _manifest_identity(
        train_dataset,
        expected_sha256=protocol.capture.train_manifest_sha256,
        expected_tokens=protocol.capture.train_tokens,
        expected_split="train",
        protocol=protocol,
        label="train",
    )
    eval_identity = _manifest_identity(
        eval_dataset,
        expected_sha256=protocol.capture.eval_manifest_sha256,
        expected_tokens=protocol.capture.eval_tokens,
        expected_split="eval",
        protocol=protocol,
        label="eval",
    )
    if train_dataset is eval_dataset:
        raise Gate1AuditError("train and eval ArtifactDataset objects must be distinct")
    train_root = train_dataset.root.expanduser().resolve(strict=False)
    eval_root = eval_dataset.root.expanduser().resolve(strict=False)
    if train_root == eval_root:
        raise Gate1AuditError("train and eval capture roots must be distinct")
    for capture_root in (train_root, eval_root):
        if _path_within(output, capture_root) or _path_within(capture_root, output):
            raise Gate1AuditError(
                "Gate-1 output root must be disjoint from immutable captures"
            )
    mean = np.asarray(train_dataset.mean, dtype=np.float32)
    if mean.shape != (HIDDEN_SIZE,) or not np.isfinite(mean).all():
        raise Gate1AuditError("training capture mean is not finite [2816]")
    return train_identity, eval_identity, output, mean


def _create_exact_runtime() -> Gate1Runtime:
    # This must precede every backend/process/device query.  Cloud TPU
    # multi-controller discovery is performed inside the initializer.
    initialize_jax_distributed()
    local_error: Exception | None = None
    runtime: Gate1Runtime | None = None
    try:
        if (
            jax.process_count() != GATE1_PROCESS_COUNT
            or jax.local_device_count() != GATE1_LOCAL_DEVICE_COUNT
            or jax.device_count() != GATE1_GLOBAL_DEVICE_COUNT
        ):
            raise Gate1AuditError(
                "Gate-1 requires exactly four processes, four local TPU devices, "
                "and sixteen global devices"
            )
        devices = tuple(jax.devices())
        if len(devices) != GATE1_GLOBAL_DEVICE_COUNT or any(
            str(device.platform).lower() != "tpu" for device in devices
        ):
            raise Gate1AuditError("Gate-1 requires exactly sixteen TPU devices")
        distributed = DistributedConfig(
            data_axis_size=GATE1_DATA_AXIS_SIZE,
            feature_axis_size=GATE1_FEATURE_AXIS_SIZE,
            microbatches_per_update=GATE1_MICROBATCHES_PER_GROUP,
        )
        mesh = create_mesh(distributed, devices=devices)
        if dict(mesh.shape) != {
            distributed.data_axis: GATE1_DATA_AXIS_SIZE,
            distributed.feature_axis: GATE1_FEATURE_AXIS_SIZE,
        }:
            raise Gate1AuditError("Gate-1 mesh is not exactly data=4, feature=4")
        process_index = jax.process_index()
        grid = np.asarray(mesh.devices, dtype=object)
        owned_rows = [
            row
            for row in grid
            if all(int(device.process_index) == process_index for device in row)
        ]
        if len(owned_rows) != 1 or len(owned_rows[0]) != GATE1_FEATURE_AXIS_SIZE:
            raise Gate1AuditError(
                "each Gate-1 process must own one complete four-device feature row"
            )
        runtime = Gate1Runtime(mesh, distributed, process_index)
    except Exception as error:
        local_error = error
    _checkpoint_phase_decision("Gate-1 physical runtime validation", local_error)
    assert runtime is not None
    return runtime


def _digest_consensus(value: str, *, operation: str) -> None:
    if len(value) != 64:
        raise Gate1AuditError(f"{operation} is not a SHA-256")
    from jax.experimental import multihost_utils

    local = np.frombuffer(bytes.fromhex(value), dtype=np.uint8).copy()
    gathered = np.asarray(
        multihost_utils.process_allgather(local, tiled=False), dtype=np.uint8
    ).reshape(GATE1_PROCESS_COUNT, 32)
    error: Exception | None = None
    if not np.all(gathered == gathered[0]):
        error = Gate1AuditError(f"hosts disagree on {operation}")
    _checkpoint_phase_decision(f"Gate-1 {operation} consensus", error)


def _collective_max_int(value: int, *, operation: str) -> int:
    from jax.experimental import multihost_utils

    gathered = np.asarray(
        multihost_utils.process_allgather(np.asarray(value, np.int32), tiled=False),
        dtype=np.int32,
    )
    if gathered.size != GATE1_PROCESS_COUNT:
        raise Gate1AuditError(f"{operation} collective returned the wrong size")
    return int(np.max(gathered))


def _collective_all(value: bool, *, operation: str) -> bool:
    from jax.experimental import multihost_utils

    gathered = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(bool(value), np.uint8), tiled=False
        ),
        dtype=np.uint8,
    )
    if gathered.size != GATE1_PROCESS_COUNT:
        raise Gate1AuditError(f"{operation} collective returned the wrong size")
    return bool(np.all(gathered))


def _barrier(label: str) -> None:
    from jax.experimental import multihost_utils

    multihost_utils.sync_global_devices(label)


def _partition_ranges(
    split: SplitSpec,
    *,
    process_index: int,
) -> tuple[TokenRange, ...]:
    if process_index not in range(GATE1_PROCESS_COUNT):
        raise Gate1AuditError("Gate-1 process index is outside [0,4)")
    selected: list[TokenRange] = []
    for item in split.ranges:
        if item.tokens % GATE1_PROCESS_COUNT:
            raise Gate1AuditError("split range does not divide across four hosts")
        local = item.tokens // GATE1_PROCESS_COUNT
        start = item.start + process_index * local
        selected.append(TokenRange(start, start + local))
    return tuple(selected)


def _expected_groups(ranges: Sequence[TokenRange]) -> int:
    total = 0
    for item in ranges:
        local_microbatches = math.ceil(item.tokens / GATE1_LOCAL_MICROBATCH_TOKENS)
        total += math.ceil(local_microbatches / GATE1_MICROBATCHES_PER_GROUP)
    return total


def _prefetch_local_ranges(
    dataset: ArtifactDataset,
    ranges: Sequence[TokenRange],
) -> dict[str, Any]:
    observations: list[dict[str, int]] = []
    local_error: Exception | None = None
    try:
        for item in ranges:
            observed = dataset.prefetch_token_range(item.start, item.stop)
            if (
                not isinstance(observed, Mapping)
                or isinstance(observed.get("objects"), bool)
                or not isinstance(observed.get("objects"), int)
                or isinstance(observed.get("bytes"), bool)
                or not isinstance(observed.get("bytes"), int)
                or int(observed["objects"]) < 0
                or int(observed["bytes"]) < 0
            ):
                raise Gate1AuditError("capture prefetch returned invalid evidence")
            observations.append(
                {
                    "objects": int(observed["objects"]),
                    "bytes": int(observed["bytes"]),
                }
            )
    except Exception as error:
        local_error = error
    _checkpoint_phase_decision("Gate-1 local capture prefetch", local_error)
    return {
        "ranges": [item.to_list() for item in ranges],
        "objects": sum(item["objects"] for item in observations),
        "bytes": sum(item["bytes"] for item in observations),
        "performance_claim": False,
    }


def _padding_ids(rows: int, config: SAEConfig) -> np.ndarray:
    token = np.arange(rows, dtype=np.int64)[:, None]
    slot = np.arange(config.top_routed, dtype=np.int64)[None, :]
    return ((token * config.top_routed + slot) % config.num_experts).astype(np.int32)


def _validated_rows(
    batch: Any,
    config: SAEConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    activation = np.asarray(batch.activation)
    ids = np.asarray(batch.expert_ids)
    sequences = np.asarray(batch.sequence_id)
    rows = activation.shape[0] if activation.ndim else -1
    if activation.ndim != 2 or activation.shape[1] != config.hidden_dim:
        raise Gate1AuditError("capture activation has the wrong shape")
    if ids.shape != (rows, config.top_routed):
        raise Gate1AuditError("capture expert IDs have the wrong shape")
    if sequences.shape != (rows,) or not np.issubdtype(
        sequences.dtype, np.signedinteger
    ):
        raise Gate1AuditError("capture sequence IDs must be a signed vector")
    if not np.issubdtype(ids.dtype, np.integer):
        raise Gate1AuditError("capture expert IDs must use an integer dtype")
    if activation.dtype == np.uint16:
        activation = bf16_bits_to_float32(activation)
    else:
        activation = activation.astype(np.float32, copy=False)
    if not np.isfinite(activation).all():
        raise Gate1AuditError("capture activation contains non-finite values")
    # ``apply_route_mode`` performs the strict ID range/uniqueness checks.
    return (
        activation,
        ids.astype(np.int32, copy=False),
        sequences.astype(np.int64, copy=False),
    )


def _transform_routes(
    ids: np.ndarray,
    sequence_ids: np.ndarray,
    *,
    route_mode: RouteMode,
    key: int,
    num_experts: int,
) -> np.ndarray:
    diagnostic_mode = "shuffled" if route_mode == "shuffled" else "true"
    return apply_route_mode(
        ids,
        sequence_ids,
        mode=diagnostic_mode,
        seed=key,
        num_experts=num_experts,
    ).astype(np.int32, copy=False)


def _iter_local_groups(
    dataset: ArtifactDataset,
    ranges: Sequence[TokenRange],
    config: SAEConfig,
    *,
    route_mode: RouteMode,
    route_key: int,
) -> Iterator[LocalAuditGroup]:
    """Pack each local range independently into one fixed static shape."""

    micro = GATE1_LOCAL_MICROBATCH_TOKENS

    def padding_microbatch() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        activation = np.zeros((micro, config.hidden_dim), dtype=np.float32)
        ids = _padding_ids(micro, config)
        sequences = np.zeros((micro,), dtype=np.int64)
        ids = _transform_routes(
            ids,
            sequences,
            route_mode=route_mode,
            key=route_key,
            num_experts=config.num_experts,
        )
        return activation, ids, sequences, np.zeros((micro,), dtype=bool)

    for token_range in ranges:
        activation_chunks: list[np.ndarray] = []
        id_chunks: list[np.ndarray] = []
        sequence_chunks: list[np.ndarray] = []
        buffered = 0
        consumed = 0
        microbatches: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        def emit(
            target: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
            *,
            tail: bool,
        ) -> None:
            nonlocal activation_chunks, id_chunks, sequence_chunks, buffered
            if buffered <= 0:
                raise Gate1AuditError("cannot emit an empty Gate-1 microbatch")
            activation = np.concatenate(activation_chunks, axis=0)
            ids = np.concatenate(id_chunks, axis=0)
            sequences = np.concatenate(sequence_chunks, axis=0)
            valid = np.ones((buffered,), dtype=bool)
            if tail and buffered < micro:
                padding = micro - buffered
                activation = np.pad(activation, ((0, padding), (0, 0)))
                ids = np.concatenate((ids, _padding_ids(padding, config)), axis=0)
                sequences = np.pad(sequences, (0, padding))
                valid = np.pad(valid, (0, padding), constant_values=False)
            if activation.shape != (micro, config.hidden_dim):
                raise Gate1AuditError("Gate-1 microbatch is not static")
            transformed = _transform_routes(
                ids,
                sequences,
                route_mode=route_mode,
                key=route_key,
                num_experts=config.num_experts,
            )
            target.append((activation, transformed, sequences, valid))
            activation_chunks = []
            id_chunks = []
            sequence_chunks = []
            buffered = 0

        for batch in dataset.iter_token_range(token_range.start, token_range.stop):
            activation, ids, sequences = _validated_rows(batch, config)
            offset = 0
            consumed += activation.shape[0]
            while offset < activation.shape[0]:
                take = min(micro - buffered, activation.shape[0] - offset)
                stop = offset + take
                activation_chunks.append(activation[offset:stop])
                id_chunks.append(ids[offset:stop])
                sequence_chunks.append(sequences[offset:stop])
                buffered += take
                offset = stop
                if buffered == micro:
                    emit(microbatches, tail=False)
                    if len(microbatches) == GATE1_MICROBATCHES_PER_GROUP:
                        yield _stack_local_group(microbatches)
                        microbatches.clear()
        if consumed != token_range.tokens:
            raise Gate1AuditError(
                f"capture yielded {consumed}/{token_range.tokens} local rows"
            )
        if buffered:
            emit(microbatches, tail=True)
        while microbatches and len(microbatches) < GATE1_MICROBATCHES_PER_GROUP:
            microbatches.append(padding_microbatch())
        if microbatches:
            yield _stack_local_group(microbatches)


def _stack_local_group(
    microbatches: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> LocalAuditGroup:
    if len(microbatches) != GATE1_MICROBATCHES_PER_GROUP:
        raise Gate1AuditError("Gate-1 group must contain exactly four microbatches")
    activation = np.stack([item[0] for item in microbatches])
    expert_ids = np.stack([item[1] for item in microbatches]).astype(
        np.int32, copy=False
    )
    sequences = np.stack([item[2] for item in microbatches]).astype(
        np.int64, copy=False
    )
    valid = np.stack([item[3] for item in microbatches]).astype(bool, copy=False)
    return LocalAuditGroup(
        batch=DistributedBatch(activation, expert_ids, valid),
        sequence_ids=sequences,
        valid_mask=valid,
    )


def _derive_a_capacity(
    groups: Iterator[LocalAuditGroup],
    *,
    num_experts: int,
) -> CapacityEvidence:
    local_worst = 0
    local_error: Exception | None = None
    try:
        for group in groups:
            for ids in group.batch.expert_ids:
                counts = np.bincount(
                    np.asarray(ids, dtype=np.int32).reshape(-1),
                    minlength=num_experts,
                )
                local_worst = max(local_worst, int(counts.max(initial=0)))
        if local_worst <= 0:
            raise Gate1AuditError("transformed A capacity scan observed no routes")
    except Exception as error:
        local_error = error
    _checkpoint_phase_decision("Gate-1 transformed A capacity scan", local_error)
    global_worst = _collective_max_int(
        local_worst, operation="Gate-1 transformed A capacity"
    )
    capacity = int(
        math.ceil((global_worst * (1.0 + A_CAPACITY_MARGIN)) / A_CAPACITY_ROUND_TO)
        * A_CAPACITY_ROUND_TO
    )
    if capacity < global_worst or capacity % A_CAPACITY_ROUND_TO:
        raise Gate1AuditError("derived transformed A capacity is invalid")
    return CapacityEvidence(local_worst, global_worst, capacity)


def _cell_identity(
    protocol: Gate1Protocol,
    cell: Gate1AuditCell,
    training: TrainingConfig,
    scoring: SAEConfig,
    train_capture: Mapping[str, Any],
    eval_capture: Mapping[str, Any],
    consumer_environment: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = protocol.checkpoints[cell.checkpoint_label]
    split = protocol.splits[cell.split_name]
    return {
        "schema_version": GATE1_PROCESS_RESULT_SCHEMA,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "cell": cell.to_mapping(),
        "checkpoint": {
            "label": checkpoint.label,
            "uri": checkpoint.checkpoint_uri,
            "expectation_sha256": checkpoint.expectation.sha256,
            "condition": checkpoint.condition,
            "seed": checkpoint.seed,
            "steps": checkpoint.steps,
            "tokens_seen": checkpoint.tokens_seen,
            "training_config_sha256": checkpoint.expectation.training_config_sha256,
        },
        "capture": {
            "config_sha256": protocol.capture.config_sha256,
            "checkpoint_sha256": protocol.capture.checkpoint_sha256,
            "train": dict(train_capture),
            "eval": dict(eval_capture),
        },
        "split": split.to_mapping(),
        "route": {
            "mode": cell.route_mode,
            "permutation": protocol.route_permutation.to_mapping(),
        },
        "restore_training_config": dataclasses.asdict(training),
        "scoring_sae_config": dataclasses.asdict(scoring),
        "consumer_environment": dict(consumer_environment),
    }


def _process_identity(
    *,
    cell_sha256: str,
    runtime: Gate1Runtime,
    ranges: Sequence[TokenRange],
) -> dict[str, Any]:
    return {
        "cell_sha256": cell_sha256,
        "process_index": runtime.process_index,
        "process_count": GATE1_PROCESS_COUNT,
        "process_ranges": [item.to_list() for item in ranges],
        "process_tokens": sum(item.tokens for item in ranges),
    }


def _result_path(
    output_root: Path,
    protocol: Gate1Protocol,
    cell_sha256: str,
    process_index: int,
) -> Path:
    return (
        output_root
        / protocol.protocol_id
        / protocol.sha256
        / cell_sha256
        / f"process-{process_index:02d}.json"
    )


def _validate_hashed_result(
    value: Any,
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Gate1AuditError("existing Gate-1 result is not a JSON object")
    record = dict(value)
    if record.get("schema_version") != GATE1_PROCESS_RESULT_SCHEMA:
        raise Gate1AuditError("existing Gate-1 result schema mismatch")
    supplied = record.get("result_sha256")
    if not isinstance(supplied, str) or len(supplied) != 64:
        raise Gate1AuditError("existing Gate-1 result has no canonical hash")
    unsigned = dict(record)
    del unsigned["result_sha256"]
    if _sha256(unsigned) != supplied:
        raise Gate1AuditError("existing Gate-1 result hash mismatch")
    if record.get("identity") != dict(expected_identity):
        raise Gate1AuditError("existing Gate-1 result belongs to another invocation")
    return record


def _read_existing(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise Gate1AuditError("cannot read existing Gate-1 result") from error
    if path.is_symlink():
        raise Gate1AuditError("existing Gate-1 result is not a regular file")
    if len(raw) > _MAX_EXISTING_RESULT_BYTES:
        raise Gate1AuditError("existing Gate-1 result is unexpectedly large")
    try:
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gate1AuditError("cannot read existing Gate-1 result") from error
    return _validate_hashed_result(value, expected_identity=expected_identity)


def _wait_for_existing(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
    timeout_seconds: float = _VISIBILITY_TIMEOUT_SECONDS,
    poll_seconds: float = _VISIBILITY_POLL_SECONDS,
    clock: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    if (
        not math.isfinite(timeout_seconds)
        or timeout_seconds < 0.0
        or not math.isfinite(poll_seconds)
        or poll_seconds <= 0.0
    ):
        raise Gate1AuditError("Gate-1 visibility retry bounds are invalid")
    deadline = clock() + timeout_seconds
    while True:
        result = _read_existing(path, expected_identity=expected_identity)
        if result is not None:
            return result
        remaining = deadline - clock()
        if remaining <= 0.0:
            raise Gate1AuditError(
                "Gate-1 process result did not become visible before timeout"
            )
        sleep(min(poll_seconds, remaining))


def _write_once(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_identity = payload["identity"]
    existing = _read_existing(path, expected_identity=expected_identity)
    if existing is not None:
        if existing != dict(payload):
            raise Gate1AuditError(
                "an existing Gate-1 result differs from deterministic rerun output"
            )
        return existing
    atomic_write_json(path, payload)
    persisted = _wait_for_existing(path, expected_identity=expected_identity)
    if persisted != dict(payload):
        raise Gate1AuditError("persisted Gate-1 result failed read-after-write")
    return persisted


_CELL_RESULT_FIELDS = {
    "schema_version",
    "protocol_sha256",
    "cell_identity_sha256",
    "consumer_environment_sha256",
    "process_result_sha256s",
    "checkpoint_label",
    "checkpoint_expectation_sha256",
    "checkpoint_descriptor_sha256",
    "checkpoint_uri",
    "capture_config_sha256",
    "capture_checkpoint_sha256",
    "capture_manifest_sha256",
    "split",
    "source",
    "ranges",
    "route_mode",
    "route_permutation_domain",
    "route_permutation_key",
    "tokens",
    "exact_l0",
    "topk_valid",
    "dead_features",
    "total_features",
    "fired_mask_sha256",
    "audit_capacity",
    "sequence_sse",
}


def _read_cell_result(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise Gate1AuditError("cannot read merged Gate-1 cell result") from error
    if path.is_symlink():
        raise Gate1AuditError("merged Gate-1 cell result is not a regular file")
    if len(raw) > _MAX_EXISTING_RESULT_BYTES:
        raise Gate1AuditError("merged Gate-1 cell result is unexpectedly large")
    try:
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gate1AuditError("cannot read merged Gate-1 cell result") from error
    if not isinstance(value, dict) or set(value) != _CELL_RESULT_FIELDS:
        raise Gate1AuditError("merged Gate-1 cell result fields do not match schema")
    if value.get("schema_version") != GATE1_CELL_RESULT_SCHEMA:
        raise Gate1AuditError("merged Gate-1 cell result schema mismatch")
    return value


def _wait_for_cell_result(
    path: Path,
    *,
    timeout_seconds: float = _VISIBILITY_TIMEOUT_SECONDS,
    poll_seconds: float = _VISIBILITY_POLL_SECONDS,
    clock: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    if (
        not math.isfinite(timeout_seconds)
        or timeout_seconds < 0.0
        or not math.isfinite(poll_seconds)
        or poll_seconds <= 0.0
    ):
        raise Gate1AuditError("Gate-1 visibility retry bounds are invalid")
    deadline = clock() + timeout_seconds
    while True:
        result = _read_cell_result(path)
        if result is not None:
            return result
        remaining = deadline - clock()
        if remaining <= 0.0:
            raise Gate1AuditError(
                "merged Gate-1 cell did not become visible before timeout"
            )
        sleep(min(poll_seconds, remaining))


def _write_cell_once(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    existing = _read_cell_result(path)
    if existing is not None:
        if existing != dict(payload):
            raise Gate1AuditError(
                "existing merged Gate-1 cell differs from deterministic rerun"
            )
        return existing
    atomic_write_json(path, payload)
    persisted = _wait_for_cell_result(path)
    if persisted != dict(payload):
        raise Gate1AuditError("merged Gate-1 cell failed read-after-write")
    return persisted


def _merged_cell_payload(
    protocol: Gate1Protocol,
    cell: Gate1AuditCell,
    cell_sha256: str,
    scoring: SAEConfig,
    fragments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(fragments) != GATE1_PROCESS_COUNT:
        raise Gate1AuditError("Gate-1 cell requires exactly four process fragments")
    if any(
        not isinstance(fragment.get("cell_identity"), Mapping)
        or _sha256(fragment["cell_identity"]) != cell_sha256
        for fragment in fragments
    ):
        raise Gate1AuditError("process fragment cell identity hash mismatch")
    consumer_environment_sha256s = {
        str(fragment["cell_identity"].get("consumer_environment", {}).get("sha256", ""))
        for fragment in fragments
    }
    if (
        len(consumer_environment_sha256s) != 1
        or len(next(iter(consumer_environment_sha256s))) != 64
        or any(
            character not in "0123456789abcdef"
            for character in next(iter(consumer_environment_sha256s))
        )
    ):
        raise Gate1AuditError(
            "process fragments disagree on consumer environment identity"
        )
    consumer_environment_sha256 = next(iter(consumer_environment_sha256s))
    process_result_sha256s = [
        str(fragment.get("result_sha256", "")) for fragment in fragments
    ]
    if any(
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in process_result_sha256s
    ):
        raise Gate1AuditError("process fragment result hash is not a SHA-256")
    descriptor_sha256s = {
        str(fragment["checkpoint"]["descriptor_sha256"]) for fragment in fragments
    }
    if len(descriptor_sha256s) != 1:
        raise Gate1AuditError("process fragments disagree on checkpoint descriptor")
    checkpoint_descriptor_sha256 = next(iter(descriptor_sha256s))
    if len(checkpoint_descriptor_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in checkpoint_descriptor_sha256
    ):
        raise Gate1AuditError("process checkpoint descriptor is not a SHA-256")
    shared_invariants = {
        (
            bool(fragment["invariants"]["exact_l0"]),
            bool(fragment["invariants"]["topk_valid"]),
            bool(fragment["invariants"]["capacity_overflow"]),
            int(fragment["invariants"]["dead_features"]),
            int(fragment["invariants"]["total_features"]),
            str(fragment["invariants"]["fired_mask_sha256"]),
        )
        for fragment in fragments
    }
    if len(shared_invariants) != 1:
        raise Gate1AuditError("process fragments disagree on global feature state")
    (
        exact_l0,
        topk_valid,
        overflow,
        dead_features,
        total_features,
        fired_mask_sha256,
    ) = next(iter(shared_invariants))
    if (
        not exact_l0
        or not topk_valid
        or overflow
        or total_features != scoring.total_features
        or not 0 <= dead_features <= total_features
    ):
        raise Gate1AuditError("process fragments failed Gate-1 scoring invariants")
    if len(fired_mask_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in fired_mask_sha256
    ):
        raise Gate1AuditError("process fired-mask hash is not a SHA-256")
    audit_capacity: dict[str, Any] | None = None
    raw_capacities = [fragment.get("audit_capacity") for fragment in fragments]
    if scoring.condition == "A":
        if any(not isinstance(value, Mapping) for value in raw_capacities):
            raise Gate1AuditError("routed A scoring lacks capacity evidence")
        margins = {float(value["margin"]) for value in raw_capacities}
        round_to = {int(value["round_to"]) for value in raw_capacities}
        global_maxima = {int(value["global_observed_max"]) for value in raw_capacities}
        capacities = {int(value["capacity"]) for value in raw_capacities}
        if (
            margins != {A_CAPACITY_MARGIN}
            or round_to != {A_CAPACITY_ROUND_TO}
            or len(global_maxima) != 1
            or len(capacities) != 1
        ):
            raise Gate1AuditError("process fragments disagree on A capacity")
        global_observed_max = next(iter(global_maxima))
        selected_capacity = next(iter(capacities))
        local_observed = [int(value["local_observed_max"]) for value in raw_capacities]
        expected_capacity = int(
            math.ceil(
                global_observed_max * (1.0 + A_CAPACITY_MARGIN) / A_CAPACITY_ROUND_TO
            )
            * A_CAPACITY_ROUND_TO
        )
        if (
            min(local_observed) < 0
            or global_observed_max != max(local_observed)
            or selected_capacity != expected_capacity
        ):
            raise Gate1AuditError("A capacity evidence is internally inconsistent")
        audit_capacity = {
            "margin": A_CAPACITY_MARGIN,
            "round_to": A_CAPACITY_ROUND_TO,
            "local_observed_max_by_process": local_observed,
            "global_observed_max": global_observed_max,
            "capacity": selected_capacity,
        }
    elif any(value is not None for value in raw_capacities):
        raise Gate1AuditError("global scoring unexpectedly has A capacity evidence")
    sequence_sse = merge_sequence_sse_summaries(
        fragment["sequence_summary"] for fragment in fragments
    )
    split = protocol.splits[cell.split_name]
    if int(sequence_sse["tokens"]) != split.tokens:
        raise Gate1AuditError("merged sequence summary does not cover the split")
    checkpoint = protocol.checkpoints[cell.checkpoint_label]
    selected_manifest = (
        protocol.capture.eval_manifest_sha256
        if split.source == "eval"
        else protocol.capture.train_manifest_sha256
    )
    shuffled = cell.route_mode == "shuffled"
    result = {
        "schema_version": GATE1_CELL_RESULT_SCHEMA,
        "protocol_sha256": protocol.sha256,
        "cell_identity_sha256": cell_sha256,
        "consumer_environment_sha256": consumer_environment_sha256,
        "process_result_sha256s": process_result_sha256s,
        "checkpoint_label": checkpoint.label,
        "checkpoint_expectation_sha256": checkpoint.expectation.sha256,
        "checkpoint_descriptor_sha256": checkpoint_descriptor_sha256,
        "checkpoint_uri": checkpoint.checkpoint_uri,
        "capture_config_sha256": protocol.capture.config_sha256,
        "capture_checkpoint_sha256": protocol.capture.checkpoint_sha256,
        "capture_manifest_sha256": selected_manifest,
        "split": split.name,
        "source": split.source,
        "ranges": [item.to_list() for item in split.ranges],
        "route_mode": cell.route_mode,
        "route_permutation_domain": (
            protocol.route_permutation.domain if shuffled else None
        ),
        "route_permutation_key": (protocol.route_permutation.key if shuffled else None),
        "tokens": split.tokens,
        "exact_l0": True,
        "topk_valid": True,
        "dead_features": dead_features,
        "total_features": total_features,
        "fired_mask_sha256": fired_mask_sha256,
        "audit_capacity": audit_capacity,
        "sequence_sse": sequence_sse,
    }
    if set(result) != _CELL_RESULT_FIELDS:
        raise Gate1AuditError("internal merged Gate-1 cell schema drift")
    # The path already binds this digest; assert each fragment belongs there.
    if any(
        fragment.get("identity", {}).get("cell_sha256") != cell_sha256
        for fragment in fragments
    ):
        raise Gate1AuditError("process fragment belongs to another Gate-1 cell")
    return result


def _publish_merged_cell(
    output_root: Path,
    protocol: Gate1Protocol,
    cell: Gate1AuditCell,
    cell_sha256: str,
    scoring: SAEConfig,
    runtime: Gate1Runtime,
) -> dict[str, Any]:
    cell_directory = output_root / protocol.protocol_id / protocol.sha256 / cell_sha256
    destination = cell_directory / "cell.json"
    coordinator_result: dict[str, Any] | None = None
    coordinator_error: Exception | None = None
    if runtime.process_index == 0:
        try:
            fragments: list[dict[str, Any]] = []
            for process_index in range(GATE1_PROCESS_COUNT):
                ranges = _partition_ranges(
                    protocol.splits[cell.split_name],
                    process_index=process_index,
                )
                identity = _process_identity(
                    cell_sha256=cell_sha256,
                    runtime=dataclasses.replace(runtime, process_index=process_index),
                    ranges=ranges,
                )
                fragment = _wait_for_existing(
                    cell_directory / f"process-{process_index:02d}.json",
                    expected_identity=identity,
                )
                fragments.append(fragment)
            payload = _merged_cell_payload(
                protocol, cell, cell_sha256, scoring, fragments
            )
            coordinator_result = _write_cell_once(destination, payload)
        except Exception as error:
            coordinator_error = error
    _checkpoint_phase_decision("Gate-1 merged cell publication", coordinator_error)
    _barrier(f"gate1-cell-{cell_sha256[:16]}-merged")

    local_result: dict[str, Any] | None = None
    local_error: Exception | None = None
    try:
        local_result = _wait_for_cell_result(destination)
        # Reconstructing the expected payload on non-coordinator hosts would
        # require rereading all fragments.  These immutable scalar bindings
        # catch a wrong-path publication before the report layer sees it.
        checkpoint = protocol.checkpoints[cell.checkpoint_label]
        if (
            local_result["protocol_sha256"] != protocol.sha256
            or local_result["checkpoint_label"] != checkpoint.label
            or local_result["checkpoint_expectation_sha256"]
            != checkpoint.expectation.sha256
            or local_result["split"] != cell.split_name
            or local_result["route_mode"] != cell.route_mode
        ):
            raise Gate1AuditError("merged Gate-1 cell result identity mismatch")
    except Exception as error:
        local_error = error
    _checkpoint_phase_decision("Gate-1 merged cell observation", local_error)
    assert local_result is not None
    if runtime.process_index == 0 and coordinator_result != local_result:
        raise Gate1AuditError("coordinator observed a different merged cell result")
    return local_result


def _score_groups(
    scorer: Gate1DistributedScorer,
    params: Any,
    groups: Iterator[LocalAuditGroup],
    *,
    expected_groups: int,
    mean: np.ndarray,
    k: int,
    total_features: int,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    accumulator = SequenceSSEAccumulator(label=label)
    valid_tokens = 0
    padded_tokens = 0
    fired_mask = np.zeros((total_features,), dtype=bool)
    for group_index in range(expected_groups):
        group: LocalAuditGroup | None = None
        load_error: Exception | None = None
        try:
            group = next(groups)
        except Exception as error:
            load_error = error
        _checkpoint_phase_decision(
            f"Gate-1 group {group_index:05d} input readiness", load_error
        )
        assert group is not None

        scores: Any | None = None
        score_error: Exception | None = None
        try:
            scores = scorer.score(
                params,
                group.batch,
                mean,
                process_local=True,
            )
            token_sse, token_sst, token_l0, topk_valid = process_local_token_scores(
                scores
            )
            group_fired, capacity_overflow = process_local_feature_state(scores)
            if group_fired.shape != (total_features,):
                raise Gate1AuditError(
                    "Gate-1 scorer returned the wrong fired-mask shape"
                )
            fired_mask |= group_fired
            shape = (
                GATE1_MICROBATCHES_PER_GROUP,
                GATE1_LOCAL_MICROBATCH_TOKENS,
            )
            if any(
                np.asarray(value).shape != shape
                for value in (token_sse, token_sst, token_l0, topk_valid)
            ):
                raise Gate1AuditError("Gate-1 scorer returned the wrong token shape")
            valid = group.valid_mask
            if not np.all(np.asarray(topk_valid, dtype=bool)):
                raise Gate1AuditError("Gate-1 scorer reported an invalid TopK row")
            if not np.all(np.asarray(token_l0)[valid] == k):
                raise Gate1AuditError("Gate-1 scorer did not select exact L0=k")
            if not np.all(np.asarray(token_l0)[~valid] == 0):
                raise Gate1AuditError("Gate-1 padding rows have nonzero L0")
            if not np.all(np.asarray(token_sse)[~valid] == 0) or not np.all(
                np.asarray(token_sst)[~valid] == 0
            ):
                raise Gate1AuditError("Gate-1 padding rows contributed SSE/SST")
            if capacity_overflow:
                raise Gate1AuditError(
                    "transformed A capacity overflowed; no result was accepted"
                )
            accumulator.update(
                group.sequence_ids.reshape(-1),
                np.asarray(token_sse).reshape(-1),
                np.asarray(token_sst).reshape(-1),
                valid_mask=valid.reshape(-1),
            )
            valid_tokens += int(np.count_nonzero(valid))
            padded_tokens += int(valid.size - np.count_nonzero(valid))
        except Exception as error:
            score_error = error
        _checkpoint_phase_decision(
            f"Gate-1 group {group_index:05d} scoring", score_error
        )
    exhaustion_error: Exception | None = None
    try:
        next(groups)
        raise Gate1AuditError("Gate-1 input produced more groups than expected")
    except StopIteration:
        pass
    except Exception as error:
        exhaustion_error = error
    _checkpoint_phase_decision("Gate-1 input exhaustion", exhaustion_error)
    summary: dict[str, Any] | None = None
    summary_error: Exception | None = None
    try:
        summary = accumulator.finalize()
        if int(summary["tokens"]) != valid_tokens:
            raise Gate1AuditError("sequence summary token count mismatch")
    except Exception as error:
        summary_error = error
    _checkpoint_phase_decision("Gate-1 sequence summary finalization", summary_error)
    assert summary is not None
    packed_fired = np.packbits(fired_mask, bitorder="little")
    return summary, {
        "groups": expected_groups,
        "valid_tokens": valid_tokens,
        "padded_tokens": padded_tokens,
        "exact_l0": True,
        "topk_valid": True,
        "capacity_overflow": False,
        "dead_features": int(total_features - np.count_nonzero(fired_mask)),
        "total_features": total_features,
        "fired_mask_sha256": hashlib.sha256(packed_fired.tobytes()).hexdigest(),
    }


def run_gate1_audit_cell(
    protocol: Gate1Protocol,
    train_dataset: ArtifactDataset,
    eval_dataset: ArtifactDataset,
    output_root: Path,
    *,
    checkpoint_label: str,
    split_name: str,
    route_mode: str,
) -> dict[str, Any]:
    """Collectively execute or idempotently reuse one exact Gate-1 cell.

    All four processes must call this function with identical arguments.
    Checkpoint and activation objects are read-only.  The sole write is this
    process's deterministic JSON file below ``output_root``.
    """

    runtime = _create_exact_runtime()
    prepared: (
        tuple[
            Gate1Protocol,
            Gate1AuditCell,
            dict[str, Any],
            dict[str, Any],
            Path,
            np.ndarray,
            TrainingConfig,
            SAEConfig,
            dict[str, Any],
        ]
        | None
    ) = None
    preflight_error: Exception | None = None
    try:
        validated, cell = _validated_protocol_and_cell(
            protocol,
            checkpoint_label=checkpoint_label,
            split_name=split_name,
            route_mode=route_mode,
        )
        train_identity, eval_identity, output, mean = _validate_datasets_and_output(
            validated, train_dataset, eval_dataset, output_root
        )
        training = _original_training_config(validated, cell)
        scoring = _scoring_sae_config(training, cell.route_mode)
        consumer_environment = _consumer_environment()
        prepared = (
            validated,
            cell,
            train_identity,
            eval_identity,
            output,
            mean,
            training,
            scoring,
            consumer_environment,
        )
    except Exception as error:
        preflight_error = error
    _checkpoint_phase_decision("Gate-1 invocation preflight", preflight_error)
    assert prepared is not None
    (
        validated,
        cell,
        train_identity,
        eval_identity,
        output,
        mean,
        training,
        scoring,
        consumer_environment,
    ) = prepared
    split = validated.splits[cell.split_name]
    ranges = _partition_ranges(split, process_index=runtime.process_index)
    cell_identity = _cell_identity(
        validated,
        cell,
        training,
        scoring,
        train_identity,
        eval_identity,
        consumer_environment,
    )
    cell_sha256 = _sha256(cell_identity)
    _digest_consensus(cell_sha256, operation="cell identity")
    process_identity = _process_identity(
        cell_sha256=cell_sha256,
        runtime=runtime,
        ranges=ranges,
    )
    path = _result_path(output, validated, cell_sha256, runtime.process_index)
    existing_error: Exception | None = None
    existing: dict[str, Any] | None = None
    try:
        existing = _read_existing(path, expected_identity=process_identity)
    except Exception as error:
        existing_error = error
    _checkpoint_phase_decision("Gate-1 existing result validation", existing_error)
    if _collective_all(existing is not None, operation="existing result"):
        assert existing is not None
        _barrier(f"gate1-cell-{cell_sha256[:16]}-reused")
        return _publish_merged_cell(
            output,
            validated,
            cell,
            cell_sha256,
            scoring,
            runtime,
        )

    dataset = eval_dataset if split.source == "eval" else train_dataset
    prefetch = _prefetch_local_ranges(dataset, ranges)
    template: Any | None = None
    template_error: Exception | None = None
    try:
        template = init_distributed_state(
            jax.random.PRNGKey(training.seed),
            runtime.mesh,
            training.sae,
            runtime.distributed,
        )
    except Exception as error:
        template_error = error
    _checkpoint_phase_decision("Gate-1 restore-template initialization", template_error)
    assert template is not None
    checkpoint = validated.checkpoints[cell.checkpoint_label]
    restored = restore_audit_checkpoint_payload(
        checkpoint.checkpoint_uri,
        template,
        runtime.mesh,
        training,
        checkpoint.expectation,
    )

    capacity: CapacityEvidence | None = None
    if scoring.condition == "A":
        capacity = _derive_a_capacity(
            _iter_local_groups(
                dataset,
                ranges,
                scoring,
                route_mode=cell.route_mode,
                route_key=validated.route_permutation.key,
            ),
            num_experts=scoring.num_experts,
        )
    scorer: Gate1DistributedScorer | None = None
    scorer_error: Exception | None = None
    try:
        scorer = Gate1DistributedScorer(
            runtime.mesh,
            scoring,
            runtime.distributed,
            a_capacity=(capacity.capacity if capacity is not None else None),
        )
    except Exception as error:
        scorer_error = error
    _checkpoint_phase_decision("Gate-1 scorer construction", scorer_error)
    assert scorer is not None
    group_count = _expected_groups(ranges)
    summary, invariants = _score_groups(
        scorer,
        restored.state.params,
        iter(
            _iter_local_groups(
                dataset,
                ranges,
                scoring,
                route_mode=cell.route_mode,
                route_key=validated.route_permutation.key,
            )
        ),
        expected_groups=group_count,
        mean=mean,
        k=scoring.k,
        total_features=scoring.total_features,
        label=cell_sha256,
    )
    payload: dict[str, Any] | None = None
    acceptance_error: Exception | None = None
    try:
        if invariants["valid_tokens"] != process_identity["process_tokens"]:
            raise Gate1AuditError("scored token count differs from process partition")
        descriptor = restored.descriptor
        checkpoint_evidence = {
            "uri": checkpoint.checkpoint_uri,
            "expectation_sha256": checkpoint.expectation.sha256,
            "descriptor_sha256": descriptor["descriptor_sha256"],
            "checkpoint_header_sha256": descriptor["checkpoint_header_sha256"],
            "checkpoint_identity_sha256": descriptor["checkpoint_identity_sha256"],
            "custom_metadata_sha256": descriptor["custom_metadata_sha256"],
            "condition": checkpoint.condition,
            "seed": checkpoint.seed,
            "steps": checkpoint.steps,
            "tokens_seen": checkpoint.tokens_seen,
            "training_config_sha256": (checkpoint.expectation.training_config_sha256),
        }
        payload = {
            "schema_version": GATE1_PROCESS_RESULT_SCHEMA,
            "identity": process_identity,
            "cell_identity": cell_identity,
            "runtime": runtime.to_mapping(),
            "prefetch": prefetch,
            "checkpoint": checkpoint_evidence,
            "audit_capacity": (capacity.to_mapping() if capacity is not None else None),
            "invariants": invariants,
            "sequence_summary": summary,
        }
        payload["result_sha256"] = _sha256(payload)
    except Exception as error:
        acceptance_error = error
    _checkpoint_phase_decision("Gate-1 result acceptance", acceptance_error)
    assert payload is not None

    write_error: Exception | None = None
    persisted: dict[str, Any] | None = None
    try:
        persisted = _write_once(path, payload)
    except Exception as error:
        write_error = error
    _checkpoint_phase_decision("Gate-1 process result publication", write_error)
    assert persisted is not None
    _barrier(f"gate1-cell-{cell_sha256[:16]}-published")
    if not _collective_all(True, operation="result publication"):
        raise Gate1AuditError("not every process published a Gate-1 result")
    return _publish_merged_cell(
        output,
        validated,
        cell,
        cell_sha256,
        scoring,
        runtime,
    )


__all__ = [
    "A_CAPACITY_MARGIN",
    "A_CAPACITY_ROUND_TO",
    "GATE1_CELL_RESULT_SCHEMA",
    "GATE1_PROCESS_RESULT_SCHEMA",
    "ORIGINAL_A_TRAINING_CAPACITY",
    "Gate1AuditCell",
    "Gate1AuditError",
    "Gate1Runtime",
    "run_gate1_audit_cell",
]
