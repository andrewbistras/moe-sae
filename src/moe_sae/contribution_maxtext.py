"""Bounded, performance-qualified MaxText runtime for Experiment 2.

This module joins the pieces that must be identical between the ten-minute
qualification and a longer contribution harvest:

* the published one-commit MaxText fork;
* the prefix-pruned NNX transform;
* static, device-resident packed inputs;
* Gemma's ragged sparse-MoE path;
* the rank-three selected-expert output sharding; and
* the useful-FLOP ledger for blocks 1--14 plus block-15 attention, routed
  experts, router, and shared expert.

It intentionally does not provision a TPU.  A separate launcher owns process
deadlines and cluster lifetime; this module produces the exact-identity
evidence that launcher must require before authorizing work beyond 600
seconds.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from moe_sae.config import Gemma4Config
from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    build_recomposition_sidecar,
    canonical_sha256,
)
from moe_sae.contribution_config import (
    MAXTEXT_FORK_COMMIT,
    MAXTEXT_ROUTED_MOE_PATCHED_SHA256,
    Experiment2Config,
)
from moe_sae.contribution_model import (
    ContributionCaptureBatch,
    Gemma4ContributionPrefixAdapter,
)
from moe_sae.contribution_performance import (
    MAX_QUALIFICATION_SECONDS,
    MINIMUM_USEFUL_PREFIX_MFU,
    ExactPerformanceIdentity,
    QualificationEvidence,
    build_qualification_receipt,
)
from moe_sae.contribution_recomposition import SUPPORTED_ACCUMULATION_ORDER
from moe_sae.harvest import (
    GrainBatch,
    _block_until_ready,
    _process_local_array,
    compile_adapter_step,
    extract_grain_batch,
)
from moe_sae.runtime import (
    V4_32_BF16_PEAK_FLOPS,
    MeshCandidate,
    build_maxtext_argv,
    estimate_useful_prefix_flops,
)

QUALIFICATION_HOST_SCHEMA_VERSION = (
    "gemma4-contribution-qualification-host-observation-v2"
)
CONTRIBUTION_GRAPH_SCHEMA_VERSION = "gemma4-contribution-prefix-graph-v1"
CONTRIBUTION_HARDWARE_SCHEMA_VERSION = "gemma4-contribution-hardware-v1"
CONTRIBUTION_LAUNCH_CONFIG_SCHEMA_VERSION = (
    "gemma4-contribution-launch-config-v1"
)
OBSERVED_PERFORMANCE_LAUNCH_SCHEMA_VERSION = (
    "gemma4-contribution-observed-performance-launch-v1"
)
USEFUL_FLOPS_LEDGER_SCHEMA_VERSION = (
    "gemma4-contribution-useful-flops-ledger-v1"
)

_PERFORMANCE_RUNTIME_CONFIG_FIELDS = (
    "model_name",
    "use_multimodal",
    "scan_layers",
    "pure_nnx",
    "enable_nnx",
    "pure_nnx_decoder",
    "enable_dropout",
    "load_balance_loss_weight",
    "use_ragged_sort",
    "ragged_buffer_factor",
    "megablox",
    "sparse_matmul",
    "use_gmm_v2",
    "use_tokamax_gmm",
    "use_ring_of_experts",
    "float32_weight_sum",
    "matmul_precision",
    "attention",
    "model_call_mode",
    "prefuse_moe_weights",
    "dtype",
    "weight_dtype",
    "max_target_length",
    "per_device_batch_size",
    "ici_expert_parallelism",
    "ici_fsdp_parallelism",
    "ici_tensor_parallelism",
)


class ContributionMaxTextError(RuntimeError):
    """The exact Experiment-2 MaxText runtime contract was not satisfied."""


@dataclass(frozen=True, slots=True)
class QualificationPlan:
    """A short benchmark that is always legal without a prior MFU receipt."""

    warmup_steps: int = 5
    writer_disabled_steps: int = 10
    measured_steps: int = 10
    maximum_seconds: int = MAX_QUALIFICATION_SECONDS

    def __post_init__(self) -> None:
        for name in ("warmup_steps", "writer_disabled_steps", "measured_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ContributionMaxTextError(f"{name} must be positive")
        if (
            isinstance(self.maximum_seconds, bool)
            or not isinstance(self.maximum_seconds, int)
            or not 0 < self.maximum_seconds <= MAX_QUALIFICATION_SECONDS
        ):
            raise ContributionMaxTextError(
                "qualification maximum_seconds must lie in [1, 600]"
            )


DEFAULT_QUALIFICATION_PLAN = QualificationPlan()


@dataclass(frozen=True, slots=True)
class QualificationDiagnostics:
    """Fail-closed counters observed by one controller."""

    recompilations_after_warmup: int = 0
    oom_events: int = 0
    nonfinite_events: int = 0
    capacity_overflow_events: int = 0
    dropped_routed_tokens: int = 0
    writer_failures: int = 0
    static_shapes: bool = True
    parameters_resident_before_timing: bool = True
    inputs_resident_before_timing: bool = True
    synchronized_device_timing: bool = True

    def __post_init__(self) -> None:
        for name in (
            "recompilations_after_warmup",
            "oom_events",
            "nonfinite_events",
            "capacity_overflow_events",
            "dropped_routed_tokens",
            "writer_failures",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContributionMaxTextError(
                    f"qualification diagnostic {name} must be nonnegative"
                )
        for name in (
            "static_shapes",
            "parameters_resident_before_timing",
            "inputs_resident_before_timing",
            "synchronized_device_timing",
        ):
            if type(getattr(self, name)) is not bool:
                raise ContributionMaxTextError(
                    f"qualification diagnostic {name} must be boolean"
                )


@dataclass(frozen=True, slots=True)
class InstrumentationObservation:
    """Per-controller digest proof for unmodified versus instrumented output."""

    disabled_output_sha256: str
    enabled_output_sha256: str
    equal: bool

    def __post_init__(self) -> None:
        for name in ("disabled_output_sha256", "enabled_output_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ContributionMaxTextError(f"{name} is not a SHA-256")
        if type(self.equal) is not bool:
            raise ContributionMaxTextError("instrumentation equality must be boolean")
        if self.equal and (
            self.disabled_output_sha256 != self.enabled_output_sha256
        ):
            raise ContributionMaxTextError(
                "instrumentation claims equality but output digests differ"
            )


@dataclass(frozen=True, slots=True)
class PreparedContributionExecutable:
    """One lowered and compiled prefix executable plus its exact graph record."""

    executable: Callable[[Any, Any, Any, Any], ContributionCaptureBatch]
    graph: Mapping[str, Any]
    compile_seconds: float


@dataclass(frozen=True, slots=True)
class QualificationHostObservation:
    """One controller's synchronized timing and bounded-persistence evidence."""

    identity: ExactPerformanceIdentity
    process_count: int
    process_index: int
    qualification_wall_seconds: float
    compile_seconds: float
    measured_device_seconds: float
    persisted_window_seconds: float
    writer_disabled_device_seconds: float
    global_output_tokens: int
    local_persisted_tokens: int
    writer_disabled_global_output_tokens: int
    warmup_steps: int
    synchronized_measurement_steps: int
    measured_capture_compile_count: int
    parity_oracle_compile_count: int
    total_compile_count: int
    max_writer_backlog_batches: int
    writer_backlog_limit_batches: int
    writer_queue_drained: bool
    instrumentation: InstrumentationObservation
    diagnostics: QualificationDiagnostics

    def to_dict(self) -> dict[str, Any]:
        unsigned = {
            "schema_version": QUALIFICATION_HOST_SCHEMA_VERSION,
            "identity": self.identity.to_dict(),
            "identity_sha256": self.identity.identity_sha256,
            "process_count": self.process_count,
            "process_index": self.process_index,
            "qualification_wall_seconds": self.qualification_wall_seconds,
            "compile_seconds": self.compile_seconds,
            "measured_device_seconds": self.measured_device_seconds,
            "persisted_window_seconds": self.persisted_window_seconds,
            "writer_disabled_device_seconds": (
                self.writer_disabled_device_seconds
            ),
            "global_output_tokens": self.global_output_tokens,
            "local_persisted_tokens": self.local_persisted_tokens,
            "writer_disabled_global_output_tokens": (
                self.writer_disabled_global_output_tokens
            ),
            "warmup_steps": self.warmup_steps,
            "synchronized_measurement_steps": (
                self.synchronized_measurement_steps
            ),
            "measured_capture_compile_count": (
                self.measured_capture_compile_count
            ),
            "parity_oracle_compile_count": self.parity_oracle_compile_count,
            "total_compile_count": self.total_compile_count,
            "max_writer_backlog_batches": self.max_writer_backlog_batches,
            "writer_backlog_limit_batches": self.writer_backlog_limit_batches,
            "writer_queue_drained": self.writer_queue_drained,
            "instrumentation": asdict(self.instrumentation),
            "diagnostics": asdict(self.diagnostics),
        }
        return {
            **unsigned,
            "observation_sha256": canonical_sha256(unsigned),
        }


def estimate_contribution_prefix_flops(
    *,
    batch_sequences: int,
    sequence_length: int = 1_024,
) -> int:
    """Useful Experiment-2 prefix FLOPs, including target shared expert.

    Experiment 1 stopped before the target shared expert.  Experiment 2 must
    capture its raw output, so the target block contributes one additional
    gated MLP: three projections in the forward pass, each counted as two
    FLOPs per multiply-accumulate.
    """

    base = estimate_useful_prefix_flops(
        batch_sequences=batch_sequences,
        sequence_length=sequence_length,
    )
    tokens = batch_sequences * sequence_length
    target_shared = 6 * tokens * HIDDEN_SIZE * 2_112
    return base + target_shared


def useful_flops_ledger(
    *,
    batch_sequences: int,
    sequence_length: int,
    measured_steps: int,
) -> dict[str, Any]:
    """Return the versioned arithmetic behind the MFU numerator."""

    if measured_steps <= 0:
        raise ContributionMaxTextError("measured_steps must be positive")
    per_step = estimate_contribution_prefix_flops(
        batch_sequences=batch_sequences,
        sequence_length=sequence_length,
    )
    return {
        "schema_version": USEFUL_FLOPS_LEDGER_SCHEMA_VERSION,
        "model": "google/gemma-4-26B-A4B",
        "target_layer_one_indexed": 15,
        "executed_full_blocks": 14,
        "target_block_components": [
            "local_attention",
            "router",
            "top8_routed_ffn",
            "shared_ffn",
        ],
        "excluded": [
            "embedding_lookup",
            "normalization",
            "elementwise_activations",
            "target_outer_norm",
            "target_residual_add",
            "blocks_16_through_30",
            "final_norm",
            "logits",
            "padding",
        ],
        "batch_sequences": batch_sequences,
        "sequence_length": sequence_length,
        "measured_steps": measured_steps,
        "useful_flops_per_step": per_step,
        "useful_flops": per_step * measured_steps,
    }


def _replace_setting(argv: Sequence[str], name: str, value: Any) -> list[str]:
    prefix = f"{name}="
    result = [item for item in argv if not item.startswith(prefix)]
    result.append(f"{name}={value}")
    return result


def build_contribution_maxtext_argv(
    *,
    checkpoint_path: str,
    grain_train_files: str,
    grain_eval_files: str,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
    attention: str,
    jax_cache_dir: str | Path,
    synthetic: bool = False,
    base_config_path: str | None = None,
) -> list[str]:
    """Build the exact fork-compatible capture argv.

    ``use_ragged_sort=true`` comes from the shared builder.  ``attention`` is
    required rather than inherited from MaxText's mutable default so the
    configured kernel is identical in qualification and production.  A
    nonempty local persistent cache is mandatory so retries on the same TPU VM
    do not pay the XLA compilation cost again; the cache is not placed in GCS.
    """

    cache = Path(jax_cache_dir).expanduser()
    if not str(cache):
        raise ContributionMaxTextError("jax_cache_dir must be nonempty")
    argv = build_maxtext_argv(
        checkpoint_path=checkpoint_path,
        grain_train_files=grain_train_files,
        grain_eval_files=grain_eval_files,
        mesh=mesh,
        per_device_sequence_batch=per_device_sequence_batch,
        synthetic=synthetic,
        base_config_path=base_config_path,
    )
    argv = _replace_setting(argv, "run_name", "gemma4-contribution-v1")
    argv = _replace_setting(argv, "attention", attention)
    argv = _replace_setting(argv, "jax_cache_dir", str(cache))
    return argv


def contribution_output_shardings(mesh: Any) -> ContributionCaptureBatch:
    """Shard every capture tensor by its leading token-row dimension."""

    try:
        from jax.sharding import NamedSharding, PartitionSpec
    except ImportError as error:  # pragma: no cover - TPU dependency boundary
        raise ContributionMaxTextError(
            "output sharding requires the JAX TPU runtime"
        ) from error
    row_axes = tuple(mesh.axis_names)
    if not row_axes:
        raise ContributionMaxTextError("MaxText mesh has no named axes")
    cube = NamedSharding(mesh, PartitionSpec(row_axes, None, None))
    matrix = NamedSharding(mesh, PartitionSpec(row_axes, None))
    vector = NamedSharding(mesh, PartitionSpec(row_axes))
    return ContributionCaptureBatch(
        expert_output=cube,
        expert_ids=matrix,
        router_weights=matrix,
        shared_output=matrix,
        sequence_id=vector,
        position=vector,
    )


def _stablehlo_text(lowered: Any) -> str:
    compiler_ir = getattr(lowered, "compiler_ir", None)
    if not callable(compiler_ir):
        raise ContributionMaxTextError(
            "lowered contribution graph does not expose compiler_ir"
        )
    try:
        value = compiler_ir(dialect="stablehlo")
    except TypeError:
        value = compiler_ir()
    text = str(value)
    if not text.strip():
        raise ContributionMaxTextError("lowered StableHLO text is empty")
    return text


def prepare_contribution_executable(
    adapter: Gemma4ContributionPrefixAdapter,
    mesh: Any,
    batch: GrainBatch | Mapping[str, Any],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> PreparedContributionExecutable:
    """Lower exactly once, hash StableHLO, then compile that same lowering."""

    grain = (
        batch
        if isinstance(batch, GrainBatch)
        else extract_grain_batch(batch, expected_sequence_length=1_024)
    )
    step = compile_adapter_step(
        adapter,
        out_shardings=contribution_output_shardings(mesh),
    )
    transformed = getattr(step, "_compiled_step", None)
    lower = getattr(transformed, "lower", None)
    if not callable(lower):
        raise ContributionMaxTextError(
            "prefix transform does not expose a lower() boundary"
        )
    started = clock()
    lowered = lower(
        grain.token_ids,
        grain.positions,
        grain.segment_ids,
        grain.segment_ids,
    )
    stablehlo = _stablehlo_text(lowered)
    compile_method = getattr(lowered, "compile", None)
    if not callable(compile_method):
        raise ContributionMaxTextError(
            "lowered contribution graph does not expose compile()"
        )
    executable = compile_method()
    compile_seconds = clock() - started
    if not math.isfinite(compile_seconds) or compile_seconds <= 0.0:
        raise ContributionMaxTextError("compile clock did not advance")
    graph = {
        "schema_version": CONTRIBUTION_GRAPH_SCHEMA_VERSION,
        "stablehlo_sha256": hashlib.sha256(
            stablehlo.encode("utf-8")
        ).hexdigest(),
        "adapter": (
            "moe_sae.contribution_model.Gemma4ContributionPrefixAdapter"
        ),
        "maxtext_fork_commit": MAXTEXT_FORK_COMMIT,
        "output_fields": [
            "expert_output_bf16_Bx8x2816",
            "expert_ids_uint8_Bx8",
            "router_weights_bf16_Bx8",
            "shared_output_bf16_Bx2816",
            "sequence_id_int64_B",
            "position_uint16_B",
        ],
        "suffix_pruned": True,
        "nnx_graph_updates": False,
        "ragged_sort": True,
        "route_ordering_sorts": 1,
    }
    return PreparedContributionExecutable(
        executable=executable,
        graph=graph,
        compile_seconds=compile_seconds,
    )


def hardware_manifest(
    *,
    devices: Iterable[Any],
    process_count: int,
    mesh: MeshCandidate,
    peak_flops_per_second: int = int(V4_32_BF16_PEAK_FLOPS),
) -> dict[str, Any]:
    """Build a deterministic inventory; Experiment 2 accepts only v4-32."""

    inventory: list[dict[str, Any]] = []
    for device in devices:
        inventory.append(
            {
                "id": int(device.id),
                "process_index": int(device.process_index),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "slice_index": int(getattr(device, "slice_index", 0)),
            }
        )
    inventory.sort(key=lambda item: (item["process_index"], item["id"]))
    if (
        len(inventory) != 16
        or process_count != 4
        or any(item["platform"] != "tpu" for item in inventory)
        or any("v4" not in item["device_kind"].lower() for item in inventory)
    ):
        raise ContributionMaxTextError(
            "authoritative Gemma contribution capture requires exactly one "
            "four-controller v4-32 slice"
        )
    if peak_flops_per_second != int(V4_32_BF16_PEAK_FLOPS):
        raise ContributionMaxTextError(
            "v4-32 BF16 peak must be the pinned 4.4 PFLOP/s"
        )
    return {
        "schema_version": CONTRIBUTION_HARDWARE_SCHEMA_VERSION,
        "accelerator": "v4-32",
        "global_chip_count": 16,
        "process_count": process_count,
        "peak_bf16_flops_per_second": peak_flops_per_second,
        "mesh": asdict(mesh),
        "devices": inventory,
    }


def launch_config_manifest(
    config: Experiment2Config,
    *,
    checkpoint_sha256: str,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
    sequence_length: int,
    writer_backlog_limit_batches: int,
    jax_cache_dir: str | Path,
) -> dict[str, Any]:
    """Bind every performance-relevant launch choice into one hash."""

    config.validate()
    if len(checkpoint_sha256) != 64:
        raise ContributionMaxTextError("checkpoint_sha256 is invalid")
    if writer_backlog_limit_batches != 2:
        raise ContributionMaxTextError("writer backlog limit must remain two")
    if sequence_length != config.model.sequence_length:
        raise ContributionMaxTextError("launch sequence length differs from protocol")
    return {
        "schema_version": CONTRIBUTION_LAUNCH_CONFIG_SCHEMA_VERSION,
        "experiment_config_sha256": config.config_hash,
        "checkpoint_sha256": checkpoint_sha256,
        "maxtext_fork_commit": config.provenance.maxtext_fork_commit,
        "mesh": asdict(mesh),
        "per_device_sequence_batch": per_device_sequence_batch,
        "global_sequence_batch": per_device_sequence_batch * 16,
        "sequence_length": sequence_length,
        "writer_backlog_limit_batches": writer_backlog_limit_batches,
        "jax_cache_location_kind": "host_local",
        "jax_cache_enabled": bool(str(Path(jax_cache_dir).expanduser())),
        "static_shapes": True,
        "parameters_resident": True,
        "inputs_resident": True,
        "use_ragged_sort": True,
        "nnx_prefix_pruning": True,
        "selected_route_ordering_sorts": 1,
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_scalar(child)
            for key, child in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_json_scalar(child) for child in value]
    return str(value)


def effective_performance_runtime_manifest(
    runtime_config: Any,
    runtime_mesh: Any,
    *,
    config: Experiment2Config,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
) -> dict[str, Any]:
    """Extract only graph/performance-relevant effective MaxText settings.

    Dataset iterator mode, Grain paths, qualification plan, writer
    implementation, scratch paths, and the persistent-cache path are
    deliberately absent.  They are execution evidence, not compiled capture
    identity, and differ between a synthetic qualifier and real-data harvest.
    """

    values: dict[str, Any] = {}
    for name in _PERFORMANCE_RUNTIME_CONFIG_FIELDS:
        try:
            value = getattr(runtime_config, name)
        except (AttributeError, KeyError, ValueError):
            value = None
        values[name] = _json_scalar(value)
    try:
        axis_names = tuple(str(item) for item in runtime_mesh.axis_names)
        device_shape = tuple(int(item) for item in runtime_mesh.devices.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise ContributionMaxTextError(
            "effective MaxText mesh identity is unavailable"
        ) from error
    if len(axis_names) != len(device_shape):
        raise ContributionMaxTextError(
            "effective MaxText mesh axes and shape disagree"
        )
    values["mesh_axis_names"] = list(axis_names)
    values["mesh_shape"] = {
        name: size
        for name, size in zip(axis_names, device_shape, strict=True)
    }
    required = {
        "use_multimodal": False,
        "scan_layers": False,
        "pure_nnx_decoder": True,
        "enable_dropout": False,
        "use_ragged_sort": True,
        "sparse_matmul": True,
        "use_ring_of_experts": False,
        "float32_weight_sum": True,
        "attention": config.model.attention,
        "max_target_length": config.model.sequence_length,
        "per_device_batch_size": per_device_sequence_batch,
        "ici_expert_parallelism": mesh.expert_parallelism,
        "ici_fsdp_parallelism": mesh.fsdp_parallelism,
        "ici_tensor_parallelism": mesh.tensor_parallelism,
    }
    failures = {
        name: {"expected": expected, "observed": values.get(name)}
        for name, expected in required.items()
        if values.get(name) != expected
    }
    precision = str(values["matmul_precision"]).lower()
    if not precision.endswith("default"):
        failures["matmul_precision"] = {
            "expected": "default",
            "observed": values["matmul_precision"],
        }
    if failures:
        raise ContributionMaxTextError(
            "effective MaxText graph runtime differs from the qualification "
            f"contract: {failures}"
        )
    return {
        "schema_version": "gemma4-contribution-effective-graph-runtime-v1",
        **values,
    }


def observed_performance_launch_manifest(
    config: Experiment2Config,
    *,
    checkpoint_sha256: str,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
    writer_backlog_limit_batches: int,
    jax_cache_dir: str | Path,
    environment: Mapping[str, Any],
    maxtext_provenance: Mapping[str, str],
    runtime_config: Any,
    runtime_mesh: Any,
) -> dict[str, Any]:
    """Build the qualifier/harvester-common launch identity component."""

    base = launch_config_manifest(
        config,
        checkpoint_sha256=checkpoint_sha256,
        mesh=mesh,
        per_device_sequence_batch=per_device_sequence_batch,
        sequence_length=config.model.sequence_length,
        writer_backlog_limit_batches=writer_backlog_limit_batches,
        jax_cache_dir=jax_cache_dir,
    )
    environment_sha256 = environment.get("sha256")
    if (
        not isinstance(environment_sha256, str)
        or len(environment_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in environment_sha256
        )
    ):
        raise ContributionMaxTextError(
            "environment fingerprint lacks its SHA-256"
        )
    if set(maxtext_provenance) != {
        "mode",
        "commit",
        "routed_moe_sha256",
    }:
        raise ContributionMaxTextError(
            "MaxText runtime provenance fields are not exact"
        )
    if (
        maxtext_provenance["commit"] != MAXTEXT_FORK_COMMIT
        or maxtext_provenance["routed_moe_sha256"]
        != MAXTEXT_ROUTED_MOE_PATCHED_SHA256
        or maxtext_provenance["mode"]
        not in {"locked_pep610_install", "explicit_clean_checkout"}
    ):
        raise ContributionMaxTextError(
            "MaxText runtime provenance differs from the published fork"
        )
    effective = effective_performance_runtime_manifest(
        runtime_config,
        runtime_mesh,
        config=config,
        mesh=mesh,
        per_device_sequence_batch=per_device_sequence_batch,
    )
    return {
        "schema_version": OBSERVED_PERFORMANCE_LAUNCH_SCHEMA_VERSION,
        "requested_launch": base,
        "environment_sha256": environment_sha256,
        "maxtext_provenance": dict(maxtext_provenance),
        "effective_graph_runtime": effective,
    }


def exact_performance_identity(
    *,
    graph: Mapping[str, Any],
    hardware: Mapping[str, Any],
    launch_config: Mapping[str, Any],
) -> ExactPerformanceIdentity:
    """Hash the exact compiled graph, physical slice, and runtime choices."""

    return ExactPerformanceIdentity.from_components(
        graph=graph,
        hardware=hardware,
        config=launch_config,
    )


def _remaining_seconds(
    *,
    qualification_started: float,
    maximum_seconds: int,
    clock: Callable[[], float],
) -> float:
    remaining = qualification_started + maximum_seconds - clock()
    if remaining <= 0.0:
        raise ContributionMaxTextError(
            "qualification exceeded its 600-second hard deadline"
        )
    return remaining


def _output_rows(output: Any) -> int:
    rows = getattr(output, "rows", None)
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ContributionMaxTextError("capture output has no positive row count")
    return rows


def _qualification_sequence_ids(
    segment_ids: Any,
    *,
    process_count: int,
    process_index: int,
) -> Any:
    """Build unique packed-sequence IDs without changing graph input identity.

    MaxText's synthetic input supplies packed segment labels, not sequence
    identities.  Passing those labels directly as the adapter's fourth input
    makes every batch row emit the same ``(segment, position)`` token keys.
    Construct process/row/segment IDs in the existing signed-int32 input wire
    instead.  The result has exactly the same global shape, dtype, and
    sharding as ``segment_ids``, so the already-lowered executable and its
    StableHLO identity are unchanged.

    The 31 non-sign bits are ``process:7 | local-row:8 | segment:16``.  The
    qualification reuses one static batch, so it deliberately needs no batch
    ordinal.  Production replaces adapter-emitted IDs with authoritative v1
    source keys before persistence.
    """

    if (
        isinstance(process_count, bool)
        or not isinstance(process_count, int)
        or not 0 < process_count <= 2**7
        or isinstance(process_index, bool)
        or not isinstance(process_index, int)
        or not 0 <= process_index < process_count
    ):
        raise ContributionMaxTextError(
            "qualification process identity does not fit the token-key wire"
        )
    try:
        local_segments = _process_local_array(
            segment_ids,
            "qualification segmentation",
            process_count=process_count,
            process_index=process_index,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise ContributionMaxTextError(
            "cannot assemble process-local qualification segmentation"
        ) from error
    if local_segments.ndim != 2 or local_segments.shape[0] >= 2**8:
        raise ContributionMaxTextError(
            "qualification segmentation must contain fewer than 256 local rows"
        )
    if (
        not np.issubdtype(local_segments.dtype, np.signedinteger)
        or local_segments.dtype.itemsize < 4
    ):
        raise ContributionMaxTextError(
            "qualification segmentation must use a signed integer dtype "
            "at least as wide as int32"
        )
    if local_segments.size and (
        int(local_segments.min()) < 0
        or int(local_segments.max()) >= 2**16
    ):
        raise ContributionMaxTextError(
            "qualification packed segment IDs must lie in [0,65535]"
        )

    process_bits = np.int64(process_index) << np.int64(24)
    row_bits = np.arange(
        local_segments.shape[0],
        dtype=np.int64,
    )[:, None] << np.int64(16)
    local_ids = np.ascontiguousarray(
        (
            process_bits
            | row_bits
            | local_segments.astype(np.int64, copy=False)
        ).astype(local_segments.dtype)
    )
    if getattr(segment_ids, "addressable_shards", None) is None:
        return local_ids

    try:
        import jax

        result = jax.make_array_from_process_local_data(
            segment_ids.sharding,
            local_ids,
            global_shape=tuple(int(size) for size in segment_ids.shape),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise ContributionMaxTextError(
            "cannot assemble globally sharded qualification sequence IDs"
        ) from error
    if result.shape != segment_ids.shape or result.dtype != segment_ids.dtype:
        raise ContributionMaxTextError(
            "qualification sequence IDs changed graph input shape or dtype"
        )
    return result


def _timed_step(
    executable: Callable[[Any, Any, Any, Any], Any],
    grain: GrainBatch,
    sequence_ids: Any,
    *,
    clock: Callable[[], float],
) -> tuple[Any, float]:
    started = clock()
    output = executable(
        grain.token_ids,
        grain.positions,
        grain.segment_ids,
        sequence_ids,
    )
    _block_until_ready(output)
    elapsed = clock() - started
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise ContributionMaxTextError("synchronized device clock did not advance")
    return output, elapsed


def run_qualification_host(
    prepared: PreparedContributionExecutable,
    batch: GrainBatch | Mapping[str, Any],
    *,
    identity: ExactPerformanceIdentity,
    process_count: int,
    process_index: int,
    persist: Callable[[Any], int],
    instrumentation_probe: Callable[[], InstrumentationObservation],
    qualification_started: float,
    plan: QualificationPlan = DEFAULT_QUALIFICATION_PLAN,
    diagnostics: Callable[[], QualificationDiagnostics] = (
        QualificationDiagnostics
    ),
    clock: Callable[[], float] = time.perf_counter,
) -> QualificationHostObservation:
    """Run one controller's bounded warmup, baseline, and persisted windows."""

    if not 0 <= process_index < process_count:
        raise ContributionMaxTextError("invalid JAX process identity")
    grain = (
        batch
        if isinstance(batch, GrainBatch)
        else extract_grain_batch(batch, expected_sequence_length=1_024)
    )
    sequence_ids = _qualification_sequence_ids(
        grain.segment_ids,
        process_count=process_count,
        process_index=process_index,
    )
    _block_until_ready(sequence_ids)
    _remaining_seconds(
        qualification_started=qualification_started,
        maximum_seconds=plan.maximum_seconds,
        clock=clock,
    )
    instrumentation = instrumentation_probe()
    if type(instrumentation) is not InstrumentationObservation:
        raise ContributionMaxTextError(
            "instrumentation probe returned the wrong evidence type"
        )
    if not instrumentation.equal:
        raise ContributionMaxTextError(
            "instrumented target block differs from the unmodified block"
        )

    static_rows: int | None = None
    for _ in range(plan.warmup_steps):
        _remaining_seconds(
            qualification_started=qualification_started,
            maximum_seconds=plan.maximum_seconds,
            clock=clock,
        )
        output, _ = _timed_step(
            prepared.executable,
            grain,
            sequence_ids,
            clock=clock,
        )
        rows = _output_rows(output)
        if static_rows is None:
            static_rows = rows
        elif rows != static_rows:
            raise ContributionMaxTextError("capture output shape changed in warmup")

    baseline_seconds = 0.0
    baseline_tokens = 0
    for _ in range(plan.writer_disabled_steps):
        _remaining_seconds(
            qualification_started=qualification_started,
            maximum_seconds=plan.maximum_seconds,
            clock=clock,
        )
        output, elapsed = _timed_step(
            prepared.executable,
            grain,
            sequence_ids,
            clock=clock,
        )
        rows = _output_rows(output)
        if rows != static_rows:
            raise ContributionMaxTextError("capture output shape changed after warmup")
        baseline_seconds += elapsed
        baseline_tokens += rows

    device_seconds = 0.0
    global_output_tokens = 0
    local_persisted_tokens = 0
    persisted_started = clock()
    for _ in range(plan.measured_steps):
        _remaining_seconds(
            qualification_started=qualification_started,
            maximum_seconds=plan.maximum_seconds,
            clock=clock,
        )
        output, elapsed = _timed_step(
            prepared.executable,
            grain,
            sequence_ids,
            clock=clock,
        )
        rows = _output_rows(output)
        if rows != static_rows:
            raise ContributionMaxTextError("capture output shape changed while timed")
        persisted = persist(output)
        if (
            isinstance(persisted, bool)
            or not isinstance(persisted, int)
            or not 0 <= persisted <= rows
        ):
            raise ContributionMaxTextError(
                f"persist returned invalid local rows {persisted}/{rows}"
            )
        device_seconds += elapsed
        global_output_tokens += rows
        local_persisted_tokens += persisted
    flush = getattr(persist, "flush", None)
    if not callable(flush):
        raise ContributionMaxTextError(
            "qualification persistence callback must expose flush()"
        )
    flush()
    persisted_window_seconds = clock() - persisted_started
    if persisted_window_seconds <= 0.0:
        raise ContributionMaxTextError("persistence clock did not advance")

    details = diagnostics()
    if type(details) is not QualificationDiagnostics:
        raise ContributionMaxTextError(
            "qualification diagnostics returned the wrong type"
        )
    maximum_backlog = getattr(persist, "max_observed_backlog", None)
    backlog_limit = getattr(persist, "backlog_limit", None)
    queue_drained = getattr(persist, "queue_drained", None)
    if (
        isinstance(maximum_backlog, bool)
        or not isinstance(maximum_backlog, int)
        or maximum_backlog < 0
        or isinstance(backlog_limit, bool)
        or not isinstance(backlog_limit, int)
        or backlog_limit <= 0
        or type(queue_drained) is not bool
    ):
        raise ContributionMaxTextError(
            "persistence callback lacks bounded-queue diagnostics"
        )
    wall = clock() - qualification_started
    if not 0.0 < wall <= plan.maximum_seconds:
        raise ContributionMaxTextError(
            "qualification exceeded its 600-second hard deadline"
        )
    return QualificationHostObservation(
        identity=identity,
        process_count=process_count,
        process_index=process_index,
        qualification_wall_seconds=wall,
        compile_seconds=prepared.compile_seconds,
        measured_device_seconds=device_seconds,
        persisted_window_seconds=persisted_window_seconds,
        writer_disabled_device_seconds=baseline_seconds,
        global_output_tokens=global_output_tokens,
        local_persisted_tokens=local_persisted_tokens,
        writer_disabled_global_output_tokens=baseline_tokens,
        warmup_steps=plan.warmup_steps,
        synchronized_measurement_steps=plan.measured_steps,
        measured_capture_compile_count=1,
        parity_oracle_compile_count=1,
        total_compile_count=2,
        max_writer_backlog_batches=maximum_backlog,
        writer_backlog_limit_batches=backlog_limit,
        writer_queue_drained=queue_drained,
        instrumentation=instrumentation,
        diagnostics=details,
    )


def _equal_field(
    observations: Sequence[QualificationHostObservation],
    name: str,
) -> Any:
    values = [getattr(item, name) for item in observations]
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ContributionMaxTextError(
            f"qualification controllers disagree on {name}"
        )
    return first


def aggregate_qualification_hosts(
    observations: Sequence[QualificationHostObservation],
    *,
    batch_sequences: int,
    sequence_length: int,
    peak_flops_per_second: int = int(V4_32_BF16_PEAK_FLOPS),
    minimum_useful_mfu: float = MINIMUM_USEFUL_PREFIX_MFU,
) -> tuple[QualificationEvidence, dict[str, Any]]:
    """Aggregate controllers and seal the Experiment-2 admission receipt."""

    if not observations:
        raise ContributionMaxTextError("qualification has no host observations")
    ordered = tuple(sorted(observations, key=lambda item: item.process_index))
    process_count = _equal_field(ordered, "process_count")
    if len(ordered) != process_count or tuple(
        item.process_index for item in ordered
    ) != tuple(range(process_count)):
        raise ContributionMaxTextError(
            "qualification host inventory is incomplete or duplicated"
        )
    identity = _equal_field(ordered, "identity")
    warmups = _equal_field(ordered, "warmup_steps")
    steps = _equal_field(ordered, "synchronized_measurement_steps")
    global_tokens = _equal_field(ordered, "global_output_tokens")
    baseline_tokens = _equal_field(
        ordered, "writer_disabled_global_output_tokens"
    )
    measured_capture_compile_count = _equal_field(
        ordered,
        "measured_capture_compile_count",
    )
    parity_oracle_compile_count = _equal_field(
        ordered,
        "parity_oracle_compile_count",
    )
    total_compile_count = _equal_field(ordered, "total_compile_count")
    backlog_limit = _equal_field(ordered, "writer_backlog_limit_batches")
    if (
        measured_capture_compile_count != 1
        or parity_oracle_compile_count != 1
        or total_compile_count != 2
        or total_compile_count
        != measured_capture_compile_count + parity_oracle_compile_count
    ):
        raise ContributionMaxTextError(
            "qualification must compile one measured capture executable and "
            "one separate parity oracle"
        )

    host_records = [item.to_dict() for item in ordered]
    source_evidence = {
        "schema_version": QUALIFICATION_HOST_SCHEMA_VERSION,
        "identity_sha256": identity.identity_sha256,
        "hosts": host_records,
    }
    source_evidence_sha256 = canonical_sha256(source_evidence)
    ledger = useful_flops_ledger(
        batch_sequences=batch_sequences,
        sequence_length=sequence_length,
        measured_steps=steps,
    )
    reference_hash = canonical_sha256(
        [
            item.instrumentation.disabled_output_sha256
            for item in ordered
        ]
    )
    instrumented_hash = canonical_sha256(
        [
            item.instrumentation.enabled_output_sha256
            for item in ordered
        ]
    )
    diagnostics = [item.diagnostics for item in ordered]
    evidence = QualificationEvidence(
        identity=identity,
        source_evidence_sha256=source_evidence_sha256,
        useful_flops_ledger_sha256=canonical_sha256(ledger),
        qualification_wall_seconds=max(
            item.qualification_wall_seconds for item in ordered
        ),
        measured_device_seconds=max(
            item.measured_device_seconds for item in ordered
        ),
        persisted_window_seconds=max(
            item.persisted_window_seconds for item in ordered
        ),
        useful_flops=int(ledger["useful_flops"]),
        peak_flops_per_second=peak_flops_per_second,
        warmup_steps=warmups,
        synchronized_measurement_steps=steps,
        measured_graph_compile_count=measured_capture_compile_count,
        auxiliary_validation_compile_count=parity_oracle_compile_count,
        total_compile_count=total_compile_count,
        recompilations_after_warmup=sum(
            item.recompilations_after_warmup for item in diagnostics
        ),
        oom_events=sum(item.oom_events for item in diagnostics),
        nonfinite_events=sum(item.nonfinite_events for item in diagnostics),
        capacity_overflow_events=sum(
            item.capacity_overflow_events for item in diagnostics
        ),
        dropped_routed_tokens=sum(
            item.dropped_routed_tokens for item in diagnostics
        ),
        output_tokens=global_tokens,
        persisted_tokens=sum(item.local_persisted_tokens for item in ordered),
        max_writer_backlog_batches=max(
            item.max_writer_backlog_batches for item in ordered
        ),
        writer_backlog_limit_batches=backlog_limit,
        writer_failures=sum(item.writer_failures for item in diagnostics),
        writer_queue_drained=all(
            item.writer_queue_drained for item in ordered
        ),
        static_shapes=all(item.static_shapes for item in diagnostics),
        parameters_resident_before_timing=all(
            item.parameters_resident_before_timing for item in diagnostics
        ),
        inputs_resident_before_timing=all(
            item.inputs_resident_before_timing for item in diagnostics
        ),
        synchronized_device_timing=all(
            item.synchronized_device_timing for item in diagnostics
        ),
        instrumentation_disabled_output_sha256=reference_hash,
        instrumentation_enabled_output_sha256=instrumented_hash,
        writer_disabled_output_tokens=baseline_tokens,
        writer_disabled_device_seconds=max(
            item.writer_disabled_device_seconds for item in ordered
        ),
    )
    receipt = build_qualification_receipt(
        evidence,
        minimum_useful_mfu=minimum_useful_mfu,
        qualification_max_seconds=MAX_QUALIFICATION_SECONDS,
    )
    bundle = {
        "source_evidence": source_evidence,
        "useful_flops_ledger": ledger,
        "qualification_receipt": receipt,
    }
    return evidence, bundle


def _parameter_value(value: Any) -> Any:
    get_value = getattr(value, "get_value", None)
    if callable(get_value):
        return get_value()
    if hasattr(value, "value"):
        return value.value
    return value


def _effective_norm_scale(norm: Any, *, name: str) -> Any:
    if getattr(norm, "with_scale", True) is not True:
        raise ContributionMaxTextError(f"{name} has no learned scale")
    epsilon = float(getattr(norm, "epsilon", math.nan))
    if not math.isclose(epsilon, 1e-6, rel_tol=0.0, abs_tol=0.0):
        raise ContributionMaxTextError(f"{name} uses unexpected epsilon {epsilon}")
    offset = float(getattr(norm, "scale_offset", 0.0))
    if offset != 0.0:
        raise ContributionMaxTextError(f"{name} uses unsupported scale offset")
    scale = getattr(norm, "scale", None)
    if scale is None:
        raise ContributionMaxTextError(f"{name} scale is absent")
    return _parameter_value(scale)


def _bf16_bits_from_device(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:  # pragma: no cover - TPU dependency boundary
        raise ContributionMaxTextError(
            "sidecar extraction requires JAX"
        ) from error
    array = jnp.asarray(value, dtype=jnp.bfloat16)
    bits = np.asarray(
        jax.device_get(jax.lax.bitcast_convert_type(array, jnp.uint16)),
        dtype=np.uint16,
    )
    if bits.shape != shape:
        raise ContributionMaxTextError(
            f"{name} has shape {bits.shape}, expected {shape}"
        )
    return np.ascontiguousarray(bits)


def extract_recomposition_sidecar(
    adapter: Gemma4ContributionPrefixAdapter,
    *,
    checkpoint_sha256: str,
    source_fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    """Extract the four tiny exact-BF16 tensors needed off accelerator."""

    if type(adapter) is not Gemma4ContributionPrefixAdapter:
        raise ContributionMaxTextError(
            "sidecar extraction requires the exact contribution adapter"
        )
    target = adapter.layers[adapter.config.target_layer_index]
    routed_scale = _effective_norm_scale(
        target.mlp.post_feedforward_layernorm_2,
        name="routed post-FFN RMSNorm",
    )
    shared_scale = _effective_norm_scale(
        target.mlp.post_feedforward_layernorm_1,
        name="shared post-FFN RMSNorm",
    )
    outer = getattr(target, "post_ffw_norm", None)
    if outer is None:
        raise ContributionMaxTextError("target outer FFN RMSNorm is absent")
    outer_scale = _effective_norm_scale(outer, name="outer FFN RMSNorm")
    scalar = _parameter_value(target.layer_scalar)
    return build_recomposition_sidecar(
        checkpoint_sha256=checkpoint_sha256,
        maxtext_fork_commit=MAXTEXT_FORK_COMMIT,
        source_fingerprints=source_fingerprints,
        routed_rmsnorm_scale=_bf16_bits_from_device(
            routed_scale,
            shape=(HIDDEN_SIZE,),
            name="routed_rmsnorm_scale",
        ),
        shared_rmsnorm_scale=_bf16_bits_from_device(
            shared_scale,
            shape=(HIDDEN_SIZE,),
            name="shared_rmsnorm_scale",
        ),
        outer_ffn_rmsnorm_scale=_bf16_bits_from_device(
            outer_scale,
            shape=(HIDDEN_SIZE,),
            name="outer_ffn_rmsnorm_scale",
        ),
        layer_scalar=_bf16_bits_from_device(
            scalar,
            shape=(1,),
            name="layer_scalar",
        ),
        accumulation_order=SUPPORTED_ACCUMULATION_ORDER,
    )


def contribution_adapter_from_runtime(
    model: Any,
    *,
    maxtext_root: str | Path | None = None,
    config: Gemma4Config | None = None,
) -> Gemma4ContributionPrefixAdapter:
    """Construct through locked-install or explicit clean-checkout verification."""

    if maxtext_root is None:
        return Gemma4ContributionPrefixAdapter.from_installed_maxtext(
            model,
            config,
        )
    return Gemma4ContributionPrefixAdapter.from_maxtext(
        model,
        config,
        maxtext_root=maxtext_root,
    )


__all__ = [
    "CONTRIBUTION_GRAPH_SCHEMA_VERSION",
    "CONTRIBUTION_HARDWARE_SCHEMA_VERSION",
    "CONTRIBUTION_LAUNCH_CONFIG_SCHEMA_VERSION",
    "OBSERVED_PERFORMANCE_LAUNCH_SCHEMA_VERSION",
    "QUALIFICATION_HOST_SCHEMA_VERSION",
    "USEFUL_FLOPS_LEDGER_SCHEMA_VERSION",
    "ContributionMaxTextError",
    "InstrumentationObservation",
    "PreparedContributionExecutable",
    "QualificationDiagnostics",
    "QualificationHostObservation",
    "QualificationPlan",
    "aggregate_qualification_hosts",
    "build_contribution_maxtext_argv",
    "contribution_adapter_from_runtime",
    "contribution_output_shardings",
    "effective_performance_runtime_manifest",
    "estimate_contribution_prefix_flops",
    "exact_performance_identity",
    "extract_recomposition_sidecar",
    "hardware_manifest",
    "launch_config_manifest",
    "observed_performance_launch_manifest",
    "prepare_contribution_executable",
    "run_qualification_host",
    "useful_flops_ledger",
]
