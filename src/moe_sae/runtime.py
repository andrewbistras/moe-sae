"""TPU runtime boundaries for the Gemma 4 activation harvester.

MaxText is deliberately imported only inside :func:`create_maxtext_runtime`.
This keeps checkpoint conversion dependencies and TPU-only libraries out of
the CPU test/training environment while leaving the production path explicit.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from importlib import metadata, resources
from pathlib import Path
from typing import Any

import numpy as np

from moe_sae.config import MAXTEXT_COMMIT, MODEL_ID

V4_32_CHIPS = 16
V4_32_BF16_PEAK_FLOPS = 4.4e15
V6E_16_CHIPS = 16
V6E_16_BF16_PEAK_FLOPS = 14.688e15
V5LITEPOD_16_CHIPS = 16
V5LITEPOD_16_BF16_PEAK_FLOPS = 3.152e15
V5LITEPOD_DEVICE_KIND_ENV = "MOE_SAE_V5LITEPOD_DEVICE_KIND"
SUPPORTED_TRAINING_ACCELERATOR_TYPES = frozenset(
    {"v4-32", "v6e-16", "v5litepod-16"}
)
REGIONAL_ACCELERATOR_TYPES = frozenset({"v6e-16", "v5litepod-16"})
TARGET_SEQUENCE_LENGTH = 1_024
TARGET_LAYER_ONE_INDEXED = 15
TARGET_TRAIN_TOKENS = 50_000_000
TARGET_EVAL_TOKENS = 2_000_000
BENCHMARK_TIMING_SCHEMA = "gemma4-benchmark-timing-v2"


@dataclass(frozen=True, slots=True)
class MeshCandidate:
    """One MaxText intra-chip mesh candidate, in required benchmark order."""

    name: str
    expert_parallelism: int
    fsdp_parallelism: int
    tensor_parallelism: int


HARVEST_MESHES = (
    MeshCandidate("ep16-fsdp1-tp1", 16, 1, 1),
    MeshCandidate("ep8-fsdp2-tp1", 8, 2, 1),
    MeshCandidate("ep8-fsdp1-tp2", 8, 1, 2),
)
PER_DEVICE_SEQUENCE_BATCHES = (1, 2, 4, 8)


@dataclass(frozen=True, slots=True)
class BenchmarkObservation:
    """Synchronized result for one mesh/batch harvester benchmark.

    ``device_elapsed_seconds`` contains only calls to the compiled prefix step
    through their synchronization point. ``elapsed_seconds`` also contains
    input handling and durable persistence, and is therefore the denominator
    for persisted goodput rather than device MFU.
    """

    mesh: MeshCandidate
    per_device_sequence_batch: int
    synchronized_steps: int
    elapsed_seconds: float
    device_elapsed_seconds: float
    persistence_seconds: float
    useful_flops: float
    output_tokens: int
    persisted_tokens: int
    input_wait_seconds: float = 0.0
    recompilations: int = 0
    dropped_routed_tokens: int = 0
    max_writer_backlog: int = 0
    instrumentation_equal: bool = True
    failure: str | None = None

    @property
    def useful_mfu(self) -> float:
        return self.useful_flops / (
            V4_32_BF16_PEAK_FLOPS * self.device_elapsed_seconds
        )

    @property
    def device_tokens_per_second(self) -> float:
        return self.output_tokens / self.device_elapsed_seconds

    @property
    def persisted_tokens_per_second(self) -> float:
        return self.persisted_tokens / self.elapsed_seconds

    @property
    def input_wait_fraction(self) -> float:
        return self.input_wait_seconds / self.elapsed_seconds

    @property
    def persistence_fraction(self) -> float:
        return self.persistence_seconds / self.elapsed_seconds

    def passes_hard_gate(self, *, minimum_mfu: float = 0.50) -> bool:
        return (
            self.synchronized_steps >= 200
            and self.useful_mfu >= minimum_mfu
            and self.recompilations == 0
            and self.dropped_routed_tokens == 0
            and self.max_writer_backlog < 2
            and self.instrumentation_equal
            and self.persisted_tokens == self.output_tokens
            and self.failure is None
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["timing_schema"] = BENCHMARK_TIMING_SCHEMA
        result["useful_mfu"] = self.useful_mfu
        result["device_tokens_per_second"] = self.device_tokens_per_second
        result["persisted_tokens_per_second"] = self.persisted_tokens_per_second
        result["input_wait_fraction"] = self.input_wait_fraction
        result["persistence_fraction"] = self.persistence_fraction
        result["passes_hard_gate"] = self.passes_hard_gate()
        return result


@dataclass(frozen=True, slots=True)
class CanaryObservation:
    """Real-data canary compared with the chosen synthetic benchmark."""

    captured_tokens: int
    real_goodput: float
    synthetic_goodput: float
    input_wait_fraction: float
    persisted_goodput: float
    device_output_goodput: float
    max_writer_backlog: int

    def passes(self) -> bool:
        return (
            self.captured_tokens == 2_000_000
            and self.real_goodput >= 0.90 * self.synthetic_goodput
            and self.input_wait_fraction < 0.05
            and self.persisted_goodput >= 0.90 * self.device_output_goodput
            and self.max_writer_backlog < 2
        )


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str


class PreflightError(RuntimeError):
    """Raised when any production prerequisite is not exactly satisfied."""


class RuntimeDependencyError(RuntimeError):
    """Raised when the optional MaxText/TPU environment is unavailable."""


def initialize_jax_distributed() -> None:
    """Initialize TPU multi-controller JAX before any backend query."""

    try:
        import jax
    except ImportError as error:
        raise RuntimeDependencyError("JAX is unavailable") from error
    if jax.distributed.is_initialized():
        return
    try:
        # Cloud TPU coordinator/process metadata is auto-discovered by JAX.
        jax.distributed.initialize()
    except (RuntimeError, ValueError) as error:
        raise RuntimeDependencyError(
            "could not initialize JAX distributed before backend discovery"
        ) from error


def v4_32_runtime_status() -> tuple[bool, str]:
    """Return whether JAX exposes exactly sixteen TPU v4 chips."""

    try:
        import jax
    except ImportError:
        return False, "JAX unavailable"
    devices = list(jax.devices())
    platforms = sorted({str(device.platform) for device in devices})
    kinds = sorted({str(device.device_kind) for device in devices})
    correct = (
        len(devices) == V4_32_CHIPS
        and all(device.platform == "tpu" for device in devices)
        and all("v4" in str(device.device_kind).lower() for device in devices)
    )
    return correct, f"{len(devices)} devices; platforms={platforms}; kinds={kinds}"


def require_v4_32_runtime() -> None:
    """Reject authoritative benchmark/training timing on any other runtime."""

    initialize_jax_distributed()
    passed, detail = v4_32_runtime_status()
    if not passed:
        raise RuntimeDependencyError(
            f"authoritative execution requires a v4-32: {detail}"
        )


def accelerator_peak_flops_per_second(accelerator_type: str) -> int:
    """Return the exact published aggregate BF16 peak for one slice."""

    expected_peaks = {
        "v4-32": int(V4_32_BF16_PEAK_FLOPS),
        "v6e-16": int(V6E_16_BF16_PEAK_FLOPS),
        "v5litepod-16": int(V5LITEPOD_16_BF16_PEAK_FLOPS),
    }
    try:
        return expected_peaks[accelerator_type]
    except KeyError as error:
        raise RuntimeDependencyError(
            f"unsupported accelerator type: {accelerator_type!r}"
        ) from error


def v5litepod_device_kind_from_environment(
    environment: Mapping[str, str] | None = None,
) -> str:
    """Return the operator-supplied exact JAX v5litepod device kind."""

    source = os.environ if environment is None else environment
    expected_kind = source.get(V5LITEPOD_DEVICE_KIND_ENV, "")
    if (
        not expected_kind
        or len(expected_kind) > 128
        or any(ord(character) < 32 for character in expected_kind)
    ):
        raise RuntimeDependencyError(
            f"{V5LITEPOD_DEVICE_KIND_ENV} must be set to the exact "
            "device_kind observed by the fresh controller identity probe"
        )
    return expected_kind


def accelerator_runtime_status(
    accelerator_type: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """Return exact training topology status for supported 16-chip slices.

    Historical v4 callers retain :func:`require_v4_32_runtime` unchanged.
    New regional training/evaluation paths additionally require the observed
    four-process by four-local-device topology used by regional 16-chip
    ``v6e-16`` and ``v5litepod-16`` slices.
    """

    if accelerator_type == "v4-32":
        return v4_32_runtime_status()
    if accelerator_type not in REGIONAL_ACCELERATOR_TYPES:
        return False, f"unsupported accelerator_type={accelerator_type!r}"
    try:
        import jax
    except ImportError:
        return False, "JAX unavailable"
    devices = tuple(jax.devices())
    platforms = sorted({str(device.platform) for device in devices})
    kinds = sorted({str(device.device_kind) for device in devices})
    if accelerator_type == "v6e-16":
        kind_matches = all(
            "v6e" in str(device.device_kind).lower()
            or "v6 lite" in str(device.device_kind).lower()
            for device in devices
        )
        expected_chips = V6E_16_CHIPS
    else:
        try:
            expected_kind = v5litepod_device_kind_from_environment(environment)
        except RuntimeDependencyError as error:
            return False, str(error)
        kind_matches = all(
            str(device.device_kind) == expected_kind for device in devices
        )
        expected_chips = V5LITEPOD_16_CHIPS
    correct = (
        len(devices) == expected_chips
        and int(jax.process_count()) == 4
        and int(jax.local_device_count()) == 4
        and all(device.platform == "tpu" for device in devices)
        and kind_matches
    )
    return (
        correct,
        f"{len(devices)} devices; process_count={int(jax.process_count())}; "
        f"local_device_count={int(jax.local_device_count())}; "
        f"platforms={platforms}; kinds={kinds}",
    )


def require_accelerator_runtime(
    accelerator_type: str,
    *,
    peak_flops_per_second: int | None = None,
) -> None:
    """Require one exact supported runtime and, when supplied, its peak."""

    initialize_jax_distributed()
    expected_peak = accelerator_peak_flops_per_second(accelerator_type)
    if (
        peak_flops_per_second is not None
        and peak_flops_per_second != expected_peak
    ):
        raise RuntimeDependencyError(
            f"{accelerator_type} BF16 peak must be exactly {expected_peak}"
        )
    if accelerator_type == "v4-32":
        # Preserve the historical v4 acceptance surface byte-for-byte.
        require_v4_32_runtime()
        return
    passed, detail = accelerator_runtime_status(accelerator_type)
    if not passed:
        raise RuntimeDependencyError(
            f"authoritative execution requires a {accelerator_type}: {detail}"
        )


@dataclass(slots=True)
class MaxTextRuntime:
    """Loaded MaxText model, mesh, and deterministic data iterators."""

    config: Any
    mesh: Any
    model: Any
    train_iterator: Any
    eval_iterator: Any


def choose_harvest_benchmark(
    observations: Sequence[BenchmarkObservation], *, minimum_mfu: float = 0.50
) -> BenchmarkObservation:
    """Return the highest persisted-throughput candidate passing every gate."""

    passing = [
        observation
        for observation in observations
        if observation.passes_hard_gate(minimum_mfu=minimum_mfu)
    ]
    if not passing:
        summaries = ", ".join(
            f"{item.mesh.name}/b{item.per_device_sequence_batch}:"
            f"mfu={item.useful_mfu:.3f}"
            for item in observations
        )
        raise RuntimeError(f"no harvester benchmark passed the hard gate ({summaries})")
    return max(passing, key=lambda item: item.persisted_tokens_per_second)


def _attention_flops(
    *,
    batch_sequences: int,
    sequence_length: int,
    global_attention: bool,
    emb_dim: int = 2_816,
    query_heads: int = 16,
) -> int:
    """Count useful attention matmul FLOPs for one Gemma 4 block.

    The count is versioned here instead of borrowing MaxText's full-model
    estimator. It includes Q/K/V/O projections and QK/AV matmuls, and excludes
    padding, embedding lookup, normalizations, and elementwise activations.
    """

    tokens = batch_sequences * sequence_length
    if global_attention:
        head_dim, kv_heads, attended = 512, 2, sequence_length
    else:
        head_dim, kv_heads, attended = 256, 8, min(sequence_length, 1_024)
    query_dim = query_heads * head_dim
    kv_dim = kv_heads * head_dim
    # Gemma 4 global-attention blocks share their K/V projection; local blocks
    # use distinct K and V projections.  Q and O are always distinct.
    kv_projection_count = 1 if global_attention else 2
    projection = 2 * tokens * emb_dim * (2 * query_dim + kv_projection_count * kv_dim)
    qk_and_av = (
        4 * batch_sequences * query_heads * sequence_length * attended * head_dim
    )
    return projection + qk_and_av


def estimate_useful_prefix_flops(
    *,
    batch_sequences: int,
    sequence_length: int = TARGET_SEQUENCE_LENGTH,
    emb_dim: int = 2_816,
    routed_experts_per_token: int = 8,
    routed_mlp_dim: int = 704,
    shared_mlp_dim: int = 2_112,
    num_experts: int = 128,
) -> int:
    """Useful FLOPs for blocks 1--14 plus target attention/router/routed FFN.

    Blocks 6 and 12 are global; all other executed blocks are local. Full
    blocks include routed and shared expert matmuls. Block 15 stops after the
    post-normalized routed aggregate, so its shared expert and residual tail
    are intentionally absent.
    """

    if batch_sequences <= 0 or sequence_length <= 0:
        raise ValueError("batch_sequences and sequence_length must be positive")
    tokens = batch_sequences * sequence_length
    routed_ffn = 6 * tokens * emb_dim * routed_mlp_dim * routed_experts_per_token
    shared_ffn = 6 * tokens * emb_dim * shared_mlp_dim
    router = 2 * tokens * emb_dim * num_experts

    total = 0
    for block in range(1, 15):
        total += _attention_flops(
            batch_sequences=batch_sequences,
            sequence_length=sequence_length,
            global_attention=block in (6, 12),
            emb_dim=emb_dim,
        )
        total += routed_ffn + shared_ffn + router
    total += _attention_flops(
        batch_sequences=batch_sequences,
        sequence_length=sequence_length,
        global_attention=False,
        emb_dim=emb_dim,
    )
    total += routed_ffn + router
    return total


def token_quota(total: int, process_count: int, process_index: int) -> int:
    """Deterministically partition an exact scientific-token total by host."""

    if total < 0 or process_count <= 0 or not 0 <= process_index < process_count:
        raise ValueError("invalid token quota arguments")
    quotient, remainder = divmod(total, process_count)
    return quotient + (process_index < remainder)


def immutable_range_id(split: str, process_index: int, batch_index: int) -> str:
    """Stable input-range identity; never derived from existing output files."""

    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")
    if process_index < 0 or batch_index < 0:
        raise ValueError("range coordinates must be non-negative")
    return f"{split}-p{process_index:05d}-b{batch_index:012d}"


def scientific_token_mask(
    token_ids: np.ndarray,
    segmentation: np.ndarray,
    *,
    excluded_token_ids: Iterable[int],
) -> np.ndarray:
    """Exclude padding and configured BOS/EOS IDs from scientific counts."""

    token_ids = np.asarray(token_ids)
    segmentation = np.asarray(segmentation)
    if token_ids.shape != segmentation.shape:
        raise ValueError("token_ids and segmentation must have identical shape")
    excluded = np.asarray(tuple(excluded_token_ids), dtype=token_ids.dtype)
    mask = segmentation != 0
    if excluded.size:
        mask &= ~np.isin(token_ids, excluded)
    return mask


def compact_capture(
    arrays: Mapping[str, np.ndarray], mask: np.ndarray, *, limit: int | None = None
) -> dict[str, np.ndarray]:
    """Flatten batch/sequence axes and retain the exact common token set."""

    flat_mask = np.asarray(mask, dtype=bool).reshape(-1)
    indices = np.flatnonzero(flat_mask)
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        indices = indices[:limit]
    result: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.shape[: mask.ndim] != mask.shape:
            raise ValueError(f"{name} does not share token axes {mask.shape}")
        trailing = value.shape[mask.ndim :]
        result[name] = value.reshape((-1, *trailing))[indices]
    return result


class BoundedArtifactSink:
    """Single-worker sink whose backlog remains strictly below two batches."""

    def __init__(self, write: Callable[..., Any], *, max_pending: int = 1):
        if max_pending != 1:
            raise ValueError("production writer backlog is fixed below two batches")
        self._write = write
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="artifact-writer"
        )
        self._pending: list[Future[Any]] = []
        self.max_observed_backlog = 0

    def submit(self, *args: Any, **kwargs: Any) -> None:
        self._reap_finished()
        # Back-pressure here is intentional: device-output and persisted
        # goodput are measured separately, so a slow writer cannot hide behind
        # an ever-growing host queue.
        if self._pending:
            self._pending.pop(0).result()
        self._pending.append(self._executor.submit(self._write, *args, **kwargs))
        self.max_observed_backlog = max(self.max_observed_backlog, len(self._pending))

    def _reap_finished(self) -> None:
        remaining: list[Future[Any]] = []
        for future in self._pending:
            if future.done():
                future.result()
            else:
                remaining.append(future)
        self._pending = remaining

    def flush(self) -> None:
        for future in self._pending:
            future.result()
        self._pending.clear()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "BoundedArtifactSink":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def maxtext_base_config_path() -> str:
    """Resolve the base config shipped by the pinned MaxText distribution."""

    try:
        return str(resources.files("maxtext").joinpath("configs", "base.yml"))
    except (ModuleNotFoundError, TypeError) as error:
        raise RuntimeDependencyError(
            "MaxText is not installed; run `uv sync --extra tpu` on a TPU host"
        ) from error


def build_maxtext_argv(
    *,
    checkpoint_path: str,
    grain_train_files: str,
    grain_eval_files: str,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
    synthetic: bool = False,
    base_config_path: str | None = None,
) -> list[str]:
    """Build the only supported MaxText configuration for harvesting."""

    if not checkpoint_path:
        raise ValueError("checkpoint_path is required")
    if per_device_sequence_batch not in PER_DEVICE_SEQUENCE_BATCHES:
        raise ValueError("per-device sequence batch must be one of 1, 2, 4, or 8")
    if not synthetic and (not grain_train_files or not grain_eval_files):
        raise ValueError("pre-staged Grain train and eval files are required")
    argv = [base_config_path or maxtext_base_config_path()]
    settings: dict[str, Any] = {
        "model_name": "gemma4-26b",
        "override_model_config": "true",
        # This path is inert for our custom pretokenized iterator and the
        # synthetic benchmark, but keep it coupled to the single model pin.
        "tokenizer_path": MODEL_ID,
        "tokenizer_type": "huggingface",
        "load_parameters_path": checkpoint_path,
        "use_multimodal": "false",
        "scan_layers": "false",
        "pure_nnx": "true",
        "enable_nnx": "true",
        "pure_nnx_decoder": "true",
        "enable_dropout": "false",
        "load_balance_loss_weight": 0,
        # Dropless EP uses a worst-case receive buffer (16x the balanced
        # assignment count for Gemma 4 at EP16).  Restrict the local
        # gather/unsort work to its valid prefix while retaining that
        # worst-case allocation, so this changes neither routing nor overflow
        # semantics.  Without it MaxText repeatedly processes the padded
        # 131,072-row buffer in every routed layer.
        "use_ragged_sort": "true",
        "dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "max_target_length": TARGET_SEQUENCE_LENGTH,
        # Real data is already tokenized and packed offline.  These flags are
        # also a fail-safe if MaxText's stock input path is reached by mistake.
        "tokenize_train_data": "false",
        "tokenize_eval_data": "false",
        "add_bos": "false",
        "add_eos": "false",
        "packing": "false",
        "grain_file_type": "arrayrecord",
        "enable_data_shuffling": "false",
        "per_device_batch_size": per_device_sequence_batch,
        "eval_per_device_batch_size": per_device_sequence_batch,
        "ici_expert_parallelism": mesh.expert_parallelism,
        "ici_fsdp_parallelism": mesh.fsdp_parallelism,
        "ici_tensor_parallelism": mesh.tensor_parallelism,
        "dataset_type": "synthetic" if synthetic else "grain",
        "eval_interval": 0 if synthetic else 1,
        # MaxText requires checkpointing enabled whenever parameters are
        # restored, and its TPU bootstrap uses this path to initialize JAX
        # distributed.  No saver is constructed by this capture runtime.
        "enable_checkpointing": "true",
        "run_name": "gemma4-v1-harvest",
    }
    if not synthetic:
        settings["grain_train_files"] = grain_train_files
        settings["grain_eval_files"] = grain_eval_files
    argv.extend(f"{key}={value}" for key, value in settings.items())
    return argv


def create_maxtext_runtime(
    argv: Sequence[str],
    *,
    packed_train_files: Sequence[str] | None = None,
    packed_eval_files: Sequence[str] | None = None,
    packed_lane_prefix_replay_records: int = 0,
) -> MaxTextRuntime:
    """Load the pinned MaxText model and iterators without importing Torch."""

    try:
        import jax
        from maxtext.common.common_types import MODEL_MODE_TRAIN
        from maxtext.configs import pyconfig
        from maxtext.input_pipeline import input_pipeline_interface
        from maxtext.utils import maxtext_utils, model_creation_utils
    except ImportError as error:
        raise RuntimeDependencyError(
            "TPU runtime dependencies are unavailable; run `uv sync --extra tpu`"
        ) from error

    # pyconfig treats argv[0] as a program name and argv[1] as the YAML path.
    config = pyconfig.initialize(["", *argv])
    devices = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices, config.mesh_axes)
    model = model_creation_utils.from_pretrained(
        config, mesh=mesh, model_mode=MODEL_MODE_TRAIN
    )
    if packed_train_files is None and packed_eval_files is None:
        train_iterator, eval_iterator = input_pipeline_interface.create_data_iterator(
            config, mesh
        )
    else:
        from moe_sae.data import build_maxtext_packed_iterator

        if not packed_train_files or not packed_eval_files:
            raise ValueError("packed runtime requires exact train and eval file inventories")
        per_device_sequence_batch = int(config.per_device_batch_size)
        if per_device_sequence_batch not in PER_DEVICE_SEQUENCE_BATCHES:
            raise ValueError(
                "packed runtime per-device sequence batch was not qualified"
            )
        train_iterator = build_maxtext_packed_iterator(
            packed_train_files,
            config=config,
            mesh=mesh,
            global_batch_size=int(config.global_batch_size_to_load),
            qualified_per_device_sequence_batch=per_device_sequence_batch,
            replay_lane_prefix_records=packed_lane_prefix_replay_records,
        )
        eval_batch = int(
            getattr(config, "global_batch_size_to_load_eval", 0)
            or config.global_batch_size_to_load
        )
        eval_iterator = build_maxtext_packed_iterator(
            packed_eval_files,
            config=config,
            mesh=mesh,
            global_batch_size=eval_batch,
            qualified_per_device_sequence_batch=per_device_sequence_batch,
            replay_lane_prefix_records=packed_lane_prefix_replay_records,
        )
    return MaxTextRuntime(config, mesh, model, train_iterator, eval_iterator)


@contextlib.contextmanager
def maxtext_mesh_context(runtime: MaxTextRuntime):
    """Install the exact MaxText mesh and logical-axis contexts for lazy NNX JIT."""

    try:
        import jax
        from flax.linen import partitioning as nn_partitioning
    except ImportError as error:
        raise RuntimeDependencyError(
            "MaxText mesh contexts require JAX and Flax"
        ) from error
    with (
        jax.set_mesh(runtime.mesh),
        runtime.mesh,
        nn_partitioning.axis_rules(runtime.config.logical_axis_rules),
    ):
        yield


def installed_maxtext_commit() -> str | None:
    """Read PEP 610 provenance without importing MaxText or Git."""

    try:
        direct_url = metadata.distribution("maxtext").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not direct_url:
        return None
    try:
        payload = json.loads(direct_url)
        return payload.get("vcs_info", {}).get("commit_id")
    except (TypeError, json.JSONDecodeError):
        return None


def environment_fingerprint(repo_root: Path | None = None) -> dict[str, Any]:
    """Return a secret-free, JSON-native execution-environment fingerprint."""

    packages: dict[str, str | None] = {}
    for name in (
        "jax",
        "jaxlib",
        "numpy",
        "safetensors",
        "maxtext",
        "orbax-checkpoint",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    devices: list[dict[str, Any]] = []
    try:
        import jax

        for device in jax.devices():
            devices.append(
                {
                    "id": int(device.id),
                    "process_index": int(device.process_index),
                    "platform": str(device.platform),
                    "device_kind": str(device.device_kind),
                }
            )
        process_count = int(jax.process_count())
    except ImportError:
        process_count = 1
    git_head: str | None = None
    repository_content_sha256: str | None = None
    repository_dirty: bool | None = None
    if repo_root is not None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            git_head = completed.stdout.strip() or None
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            git_head = None
        try:
            listed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "-z",
                ],
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout.split(b"\0")
            digest = hashlib.sha256()
            for encoded in sorted(item for item in listed if item):
                relative = encoded.decode("utf-8", errors="surrogateescape")
                path = repo_root / relative
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                if path.is_file():
                    with path.open("rb") as handle:
                        while chunk := handle.read(8 << 20):
                            digest.update(chunk)
                else:
                    digest.update(b"<deleted>")
            repository_content_sha256 = digest.hexdigest()
            status = subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain=v1", "-z"],
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout
            repository_dirty = bool(status)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            repository_content_sha256 = None
            repository_dirty = None
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "maxtext_commit": installed_maxtext_commit(),
        "git_head": git_head,
        "repository_content_sha256": repository_content_sha256,
        "repository_dirty": repository_dirty,
        "jax_process_count": process_count,
        "devices": devices,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


_TOKEN_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
)


def scan_token_shaped_secrets(root: Path) -> list[Path]:
    """Return text files containing obvious token shapes, never token values."""

    findings: list[Path] = []
    ignored_parts = {
        ".git",
        ".venv",
        "artifacts",
        "captures",
        "runs",
        "stats",
        "__pycache__",
    }
    for path in root.rglob("*"):
        if not path.is_file() or any(part in ignored_parts for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(pattern.search(text) for pattern in _TOKEN_PATTERNS):
            findings.append(path.relative_to(root))
    return findings


def _staged_path_accessible(value: str) -> tuple[bool, str]:
    """Check one local or GCS checkpoint/data specification without reading it."""

    if not value:
        return False, "missing path"
    if value.startswith("gs://"):
        try:
            completed = subprocess.run(
                ["gcloud", "storage", "ls", value],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError:
            return False, "gcloud CLI is unavailable"
        except subprocess.TimeoutExpired:
            return False, "GCS accessibility check timed out"
        if completed.returncode != 0:
            return False, "GCS path is absent or inaccessible"
        if not completed.stdout.strip():
            return False, "GCS pattern matched no objects"
        return True, "GCS path is accessible"
    matches = glob.glob(os.path.expanduser(value))
    if not matches:
        return False, "local path/pattern matched no files"
    return True, f"local path/pattern matched {len(matches)} entries"


def _shared_artifact_probe(path: Path) -> tuple[bool, str]:
    """Prove that every JAX controller observes the same writable directory."""

    try:
        import jax
        from jax.experimental import multihost_utils
    except ImportError:
        return False, "JAX multi-host utilities are unavailable"
    process_count = int(jax.process_count())
    process_index = int(jax.process_index())
    if process_count == 1:
        return True, "single-process artifact directory"

    path.mkdir(parents=True, exist_ok=True)
    local_nonce = os.urandom(16) if process_index == 0 else bytes(16)
    nonce = np.asarray(
        multihost_utils.broadcast_one_to_all(
            np.frombuffer(local_nonce, dtype=np.uint8).copy(),
            is_source=process_index == 0,
        ),
        dtype=np.uint8,
    ).tobytes()
    probe = path / f".shared-preflight-{nonce.hex()}"
    write_ok = True
    if process_index == 0:
        temporary = probe.with_suffix(".tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(nonce)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, probe)
        except OSError:
            write_ok = False
            temporary.unlink(missing_ok=True)
    multihost_utils.sync_global_devices("gemma4-preflight-artifact-written")
    try:
        visible = write_ok and probe.read_bytes() == nonce
    except OSError:
        visible = False
    visible_everywhere = bool(
        np.all(
            np.asarray(
                multihost_utils.process_allgather(np.asarray(visible, dtype=np.uint8))
            )
        )
    )
    multihost_utils.sync_global_devices("gemma4-preflight-artifact-observed")
    if process_index == 0:
        probe.unlink(missing_ok=True)
    multihost_utils.sync_global_devices("gemma4-preflight-artifact-cleaned")
    if not visible_everywhere:
        return (
            False,
            "directory is not a common writable filesystem on every controller",
        )
    return True, f"shared writable directory visible to {process_count} controllers"


def run_preflight(
    *,
    repo_root: Path,
    artifact_root: Path,
    checkpoint_path: str = "",
    grain_train_files: str = "",
    grain_eval_files: str = "",
    require_tpu: bool = True,
) -> list[PreflightCheck]:
    """Run fail-closed production checks and raise with all failures."""

    repo_root = repo_root.resolve()
    distributed_error: str | None = None
    if require_tpu:
        try:
            initialize_jax_distributed()
        except RuntimeDependencyError as error:
            distributed_error = str(error)
    resolved_artifact = (
        (repo_root / artifact_root).resolve()
        if not artifact_root.is_absolute()
        else artifact_root.resolve()
    )
    required_artifact = (repo_root / "artifacts" / "gemma4_v1").resolve()
    try:
        resolved_artifact.relative_to(required_artifact)
        artifact_ok = True
    except ValueError:
        artifact_ok = False

    secret_files = scan_token_shaped_secrets(repo_root)
    commit = installed_maxtext_commit()
    checkpoint_access = (bool(checkpoint_path), "access deferred in local mode")
    train_access = (bool(grain_train_files), "access deferred in local mode")
    eval_access = (bool(grain_eval_files), "access deferred in local mode")
    shared_artifacts = (True, "shared-store probe deferred in local mode")
    if require_tpu:
        checkpoint_access = _staged_path_accessible(checkpoint_path)
        train_access = _staged_path_accessible(grain_train_files)
        eval_access = _staged_path_accessible(grain_eval_files)
        if distributed_error is None:
            try:
                shared_artifacts = _shared_artifact_probe(resolved_artifact)
            except (OSError, RuntimeError, ValueError) as error:
                shared_artifacts = (False, f"shared-store probe failed: {error}")
        else:
            shared_artifacts = (False, distributed_error)
    checks = [
        PreflightCheck(
            "python",
            sys.version_info[:2] == (3, 12),
            f"found {sys.version_info.major}.{sys.version_info.minor}; require 3.12",
        ),
        PreflightCheck(
            "artifact_root",
            artifact_ok,
            f"resolved to {resolved_artifact}; must be under {required_artifact}",
        ),
        PreflightCheck(
            "secret_scan",
            not secret_files,
            "clean"
            if not secret_files
            else "token-shaped secret in " + ", ".join(map(str, secret_files)),
        ),
        PreflightCheck(
            "checkpoint",
            checkpoint_access[0],
            checkpoint_access[1],
        ),
        PreflightCheck(
            "grain_train",
            train_access[0],
            train_access[1],
        ),
        PreflightCheck(
            "grain_eval",
            eval_access[0],
            eval_access[1],
        ),
        PreflightCheck(
            "shared_artifact_store",
            shared_artifacts[0],
            shared_artifacts[1],
        ),
    ]
    if require_tpu:
        checks.append(
            PreflightCheck(
                "jax_distributed",
                distributed_error is None,
                "initialized before backend discovery"
                if distributed_error is None
                else distributed_error,
            )
        )
        checks.append(
            PreflightCheck(
                "maxtext_commit",
                commit == MAXTEXT_COMMIT,
                f"found {commit or 'not installed'}; require {MAXTEXT_COMMIT}",
            )
        )
        tpu_ok, detail = v4_32_runtime_status()
        checks.append(PreflightCheck("v4_32", tpu_ok, detail))

    failures = [check for check in checks if not check.passed]
    if failures:
        raise PreflightError(
            "; ".join(f"{check.name}: {check.detail}" for check in failures)
        )
    return checks


def timed_synchronized_steps(
    step: Callable[[Any], Any],
    batches: Iterable[Any],
    *,
    warmups: int = 5,
    measured_steps: int = 200,
) -> tuple[float, list[Any]]:
    """Compile/run warmups, then time synchronized device steps."""

    if warmups != 5 or measured_steps < 200:
        raise ValueError(
            "benchmark contract requires five warmups and at least 200 measured steps"
        )
    iterator = iter(batches)
    for _ in range(warmups):
        output = step(next(iterator))
        _block_until_ready(output)
    outputs: list[Any] = []
    started = time.perf_counter()
    for _ in range(measured_steps):
        output = step(next(iterator))
        _block_until_ready(output)
        outputs.append(output)
    elapsed = time.perf_counter() - started
    return elapsed, outputs


def _block_until_ready(value: Any) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _block_until_ready(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _block_until_ready(item)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Small atomic metric sink used before artifact modules are available."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def updates_for_tokens(tokens: int, effective_batch: int) -> tuple[int, int]:
    """Return update count and final-update padding for an exact token target."""

    if tokens <= 0 or effective_batch <= 0:
        raise ValueError("tokens and effective_batch must be positive")
    updates = math.ceil(tokens / effective_batch)
    return updates, updates * effective_batch - tokens
