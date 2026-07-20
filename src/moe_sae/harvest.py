"""Production orchestration for Gemma 4 activation harvesting.

This module owns ordering, quotas, resume, host compaction, and performance
accounting.  Model execution is injected as one static-shape callable, so the
control plane remains CPU-testable and imports neither JAX nor MaxText at module
load time.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from moe_sae.artifacts import (
    MANIFEST_NAME,
    ArtifactDataset,
    ArtifactWriter,
    CaptureMetadata,
    load_manifest,
    publish_manifest,
)
from moe_sae.config import Gemma4Config
from moe_sae.harvest_control import HarvestCampaignControl
from moe_sae.model import CaptureBatch, Gemma4PrefixAdapter, validate_capture_batch
from moe_sae.object_store import (
    GCSObjectStore,
    ImmutableObjectStore,
    object_identity_from_mapping,
)
from moe_sae.runtime import (
    BoundedArtifactSink,
    BenchmarkObservation,
    CanaryObservation,
    MeshCandidate,
    V4_32_BF16_PEAK_FLOPS,
    atomic_write_json,
    build_maxtext_argv,
    create_maxtext_runtime,
    estimate_useful_prefix_flops,
    environment_fingerprint,
    immutable_range_id,
    maxtext_mesh_context,
    scientific_token_mask,
    token_quota,
)


EXCLUDED_TOKEN_IDS = (0, 1, 2)
DEFAULT_SEQUENCE_LENGTH = 1_024
DEFAULT_CANARY_TOKENS = 2_000_000
_PART_RE = re.compile(
    r"^(?P<split>train|eval)-p(?P<process>\d{5})-b(?P<batch>\d{12})"
    r"-o(?P<offset>\d{9})-n(?P<count>\d{9})$"
)


class HarvestError(RuntimeError):
    """The deterministic harvest contract was violated."""


class HarvestDependencyError(ImportError):
    """A TPU-only dependency was requested from a minimal installation."""


class GrainBatch(NamedTuple):
    """The three MaxText Grain fields consumed by the prefix adapter."""

    token_ids: Any
    positions: Any
    segment_ids: Any

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(size) for size in self.token_ids.shape)


@dataclass(frozen=True, slots=True)
class HarvestResult:
    split: str
    process_count: int
    process_index: int
    host_id: str
    total_tokens: int
    stop_after_tokens: int
    host_quota: int
    new_tokens: int
    resumed_tokens: int
    durable_host_tokens: int
    batches_scanned: int
    batches_executed: int
    elapsed_seconds: float
    device_seconds: float
    input_wait_seconds: float
    max_writer_backlog: int
    manifest_path: str | None = None
    executed_scientific_tokens: int = 0
    compile_seconds: float = 0.0

    @property
    def persisted_goodput(self) -> float:
        return (
            self.host_quota / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0
        )

    @property
    def device_output_goodput(self) -> float:
        return self.new_tokens / self.device_seconds if self.device_seconds > 0 else 0.0

    @property
    def input_wait_fraction(self) -> float:
        return (
            self.input_wait_seconds / self.elapsed_seconds
            if self.elapsed_seconds > 0
            else 0.0
        )


@dataclass(frozen=True, slots=True)
class CampaignSummary:
    """One globally synchronized harvest phase and its cumulative cost.

    For the 50M training completion, cumulative fields cover the disjoint
    canary prefix plus the resumed suffix.  Token counts always describe the
    durable ``[0, stop_after_tokens)`` prefix and therefore never double-count
    canary rows.
    """

    schema_version: str
    config_sha256: str
    checkpoint_sha256: str
    conversion_attestation_sha256: str
    dataset_fingerprint: str
    environment_sha256: str
    capture_manifest_sha256: str | None
    split: str
    phase: str
    stop_after_tokens: int
    total_tokens: int
    process_count: int
    global_chips: int
    mesh_name: str
    per_device_sequence_batch: int
    captured_tokens: int
    new_tokens: int
    resumed_tokens: int
    phase_wall_seconds: float
    wall_seconds: float
    phase_device_seconds: float
    device_seconds: float
    phase_compile_seconds: float
    compile_seconds: float
    publication_seconds: float
    publication_chip_seconds: float
    phase_chip_seconds: float
    chip_seconds: float
    real_goodput: float
    synthetic_goodput: float | None
    input_wait_fraction: float
    persisted_goodput: float
    device_output_goodput: float
    max_writer_backlog: int
    useful_flops: float
    useful_mfu: float
    benchmark_useful_mfu: float | None
    dropped_routed_tokens: int
    recompilations: int
    instrumentation_equal: bool
    canary_passed: bool | None
    manifest_path: str | None
    host_result_paths: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CampaignSummary":
        payload = dict(value)
        payload["host_result_paths"] = tuple(payload.get("host_result_paths", ()))
        try:
            result = cls(**payload)
        except TypeError as exc:
            raise HarvestError("campaign summary has an incompatible schema") from exc
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != "gemma4-harvest-campaign-v2":
            raise HarvestError(f"unsupported campaign schema {self.schema_version!r}")
        for name in (
            "config_sha256",
            "checkpoint_sha256",
            "conversion_attestation_sha256",
            "dataset_fingerprint",
            "environment_sha256",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(getattr(self, name))):
                raise HarvestError(f"campaign {name} is not a SHA-256")
        if self.phase == "production":
            if not re.fullmatch(
                r"[0-9a-f]{64}", str(self.capture_manifest_sha256 or "")
            ):
                raise HarvestError("production campaign lacks its manifest SHA-256")
        elif self.capture_manifest_sha256 is not None:
            raise HarvestError("canary campaign cannot claim an unpublished manifest")
        if self.split not in {"train", "eval"} or self.phase not in {
            "canary",
            "production",
        }:
            raise HarvestError("invalid campaign split/phase")
        if self.captured_tokens != self.stop_after_tokens:
            raise HarvestError(
                "campaign does not contain the exact requested token prefix"
            )
        if self.process_count <= 0 or len(self.host_result_paths) != self.process_count:
            raise HarvestError("campaign host telemetry is incomplete")
        numeric = (
            self.phase_wall_seconds,
            self.wall_seconds,
            self.phase_device_seconds,
            self.device_seconds,
            self.phase_chip_seconds,
            self.chip_seconds,
            self.phase_compile_seconds,
            self.compile_seconds,
            self.publication_seconds,
            self.publication_chip_seconds,
            self.real_goodput,
            self.input_wait_fraction,
            self.persisted_goodput,
            self.device_output_goodput,
            self.useful_flops,
            self.useful_mfu,
        )
        if any(not math.isfinite(float(item)) or float(item) < 0.0 for item in numeric):
            raise HarvestError("campaign telemetry must be finite and nonnegative")
        if self.wall_seconds <= 0.0 or self.chip_seconds <= 0.0:
            raise HarvestError("campaign wall/chip-seconds must be positive")
        expected_chip_seconds = self.global_chips * self.wall_seconds
        if not math.isclose(self.chip_seconds, expected_chip_seconds, rel_tol=1e-12):
            raise HarvestError("campaign chip-seconds are not global_chips * max wall")
        if not math.isclose(
            self.publication_chip_seconds,
            self.global_chips * self.publication_seconds,
            rel_tol=1e-12,
        ):
            raise HarvestError("publication chip-seconds are inconsistent")
        if not math.isclose(
            self.phase_chip_seconds,
            self.global_chips * self.phase_wall_seconds,
            rel_tol=1e-12,
        ):
            raise HarvestError("phase chip-seconds are inconsistent")


@dataclass(frozen=True, slots=True)
class BenchmarkDiagnostics:
    recompilations: int = 0
    dropped_routed_tokens: int = 0
    instrumentation_equal: bool = True
    max_writer_backlog: int = 0


def extract_grain_batch(
    batch: Mapping[str, Any],
    *,
    expected_sequence_length: int | None = DEFAULT_SEQUENCE_LENGTH,
) -> GrainBatch:
    """Extract the exact MaxText Grain training fields with no alias guessing."""

    if not isinstance(batch, Mapping):
        raise TypeError("a Grain batch must be a mapping")
    required = ("inputs", "inputs_position", "inputs_segmentation")
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError("Grain batch missing fields: " + ", ".join(missing))
    result = GrainBatch(*(batch[name] for name in required))
    shapes = [tuple(getattr(value, "shape", ())) for value in result]
    if not shapes[0] or len(shapes[0]) != 2:
        raise ValueError(f"Grain inputs must be rank two, got {shapes[0]}")
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError(
            f"Grain token, position, and segmentation shapes differ: {shapes}"
        )
    if (
        expected_sequence_length is not None
        and shapes[0][1] != expected_sequence_length
    ):
        raise ValueError(
            f"Grain sequence length must be {expected_sequence_length}, got {shapes[0][1]}"
        )
    return result


def make_sequence_ids(
    segmentation: Any,
    *,
    process_index: int,
    batch_index: int,
) -> np.ndarray:
    """Make stable signed-int64 IDs for packed sequences.

    The 63 payload bits are ``process:7 | batch:32 | row:8 | segment:16``.
    Segment zero is padding; its generated ID is never persisted.
    """

    segments = _host_array(segmentation, "inputs_segmentation")
    if segments.ndim != 2:
        raise ValueError("segmentation must have shape [batch, sequence]")
    rows, _ = segments.shape
    if not 0 <= process_index < 2**7:
        raise ValueError("process_index must fit in seven bits")
    if not 0 <= batch_index < 2**32:
        raise ValueError("batch_index must fit in 32 bits")
    if rows >= 2**8:
        raise ValueError("per-process sequence batch must contain fewer than 256 rows")
    if not np.issubdtype(segments.dtype, np.integer):
        raise TypeError("inputs_segmentation must have an integer dtype")
    if segments.size and (segments.min() < 0 or segments.max() >= 2**16):
        raise ValueError("packed segment IDs must lie in [0, 65535]")

    prefix = (np.int64(process_index) << np.int64(56)) | (
        np.int64(batch_index) << np.int64(24)
    )
    row_bits = np.arange(rows, dtype=np.int64)[:, None] << np.int64(16)
    return np.broadcast_to(prefix | row_bits, segments.shape) | segments.astype(
        np.int64
    )


def compile_adapter_step(
    adapter: Callable[..., CaptureBatch],
    *,
    jit: Callable[[Callable[..., CaptureBatch]], Callable[..., CaptureBatch]]
    | None = None,
) -> Callable[..., CaptureBatch]:
    """Compile a full-shape adapter call, importing Flax NNX only on demand."""

    if jit is None:
        try:
            jit = import_module("flax.nnx").jit
        except (ImportError, ModuleNotFoundError) as exc:
            raise HarvestDependencyError(
                "compiling the Gemma 4 adapter requires the pinned MaxText/Flax TPU environment"
            ) from exc

    model = getattr(adapter, "model", None)
    config = getattr(adapter, "config", None)
    if model is not None and config is not None:
        adapter_type = type(adapter)

        def explicit_model_step(
            dynamic_model: Any,
            token_ids: Any,
            positions: Any,
            segment_ids: Any,
            sequence_ids: Any,
        ) -> CaptureBatch:
            # NNX graph nodes must be explicit transform arguments.  Rebinding
            # the lightweight adapter at trace time avoids capturing a module
            # in a Python closure.
            rebound = adapter_type(dynamic_model, config)
            return rebound(
                token_ids,
                positions,
                segment_ids,
                sequence_ids,
                model_mode="train",
            )

        compiled = jit(explicit_model_step)

        def bound_step(
            token_ids: Any,
            positions: Any,
            segment_ids: Any,
            sequence_ids: Any,
        ) -> CaptureBatch:
            return compiled(model, token_ids, positions, segment_ids, sequence_ids)

        setattr(bound_step, "_compiled_step", compiled)
        return bound_step

    def injected_step(
        token_ids: Any,
        positions: Any,
        segment_ids: Any,
        sequence_ids: Any,
    ) -> CaptureBatch:
        return adapter(
            token_ids, positions, segment_ids, sequence_ids, model_mode="train"
        )

    compiled = jit(injected_step)
    try:
        setattr(compiled, "_compiled_step", compiled)
    except (AttributeError, TypeError):
        pass
    return compiled


def _close_maxtext_runtime(runtime: Any) -> None:
    """Close Grain workers explicitly before the next large MaxText runtime."""

    seen: set[int] = set()
    for iterator in (runtime.train_iterator, runtime.eval_iterator):
        if iterator is None or id(iterator) in seen:
            continue
        seen.add(id(iterator))
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def _run_instrumentation_parity(
    adapter: Gemma4PrefixAdapter,
    grain: GrainBatch,
    *,
    process_count: int,
) -> bool:
    """Require a black-box full-branch parity probe from the model adapter."""

    parity = getattr(adapter, "instrumentation_parity", None)
    if not callable(parity):
        raise HarvestError(
            "Gemma4PrefixAdapter lacks instrumentation_parity; benchmark cannot "
            "claim instrumented/full-branch equality"
        )
    model = getattr(adapter, "model", None)
    config = getattr(adapter, "config", None)
    if model is not None and config is not None:
        try:
            nnx_jit = import_module("flax.nnx").jit
        except (ImportError, ModuleNotFoundError) as exc:
            raise HarvestDependencyError(
                "instrumentation parity requires Flax NNX"
            ) from exc
        adapter_type = type(adapter)

        def explicit_model_parity(
            dynamic_model: Any,
            token_ids: Any,
            positions: Any,
            segment_ids: Any,
            sequence_ids: Any,
        ) -> Any:
            rebound = adapter_type(dynamic_model, config)
            return rebound.instrumentation_parity(
                token_ids,
                positions,
                segment_ids,
                sequence_ids,
                model_mode="train",
            )

        compiled_parity = nnx_jit(explicit_model_parity)
        result = compiled_parity(
            model,
            grain.token_ids,
            grain.positions,
            grain.segment_ids,
            grain.segment_ids,
        )
    else:
        result = parity(
            grain.token_ids,
            grain.positions,
            grain.segment_ids,
            grain.segment_ids,
            model_mode="train",
        )
    _block_until_ready(result)
    try:
        local = np.asarray(result)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HarvestError(
            "instrumentation parity result is not host-addressable"
        ) from exc
    if local.size != 1:
        raise HarvestError("instrumentation_parity must return one boolean scalar")
    local_equal = bool(local.reshape(()).item())
    # Do not rendezvous here.  The CLI aggregates the complete candidate
    # observation after every controller has either succeeded or recorded a
    # failure, which lets a local setup/writer error fail the candidate without
    # stranding peers inside an internal barrier.
    del process_count
    if not local_equal:
        raise HarvestError(
            "instrumented prefix output differs from the unmodified full branch"
        )
    return True


def compile_warm_and_replay(
    batches: Iterable[Mapping[str, Any]],
    step: Callable[[Any, Any, Any, Any], Any],
    *,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[Iterable[Mapping[str, Any]], float]:
    """Lazy-JIT on the first static batch, then replay it as immutable range zero."""

    source = iter(batches)
    try:
        first_raw = next(source)
    except StopIteration as exc:
        raise HarvestError("cannot compile-warm from an empty Grain iterator") from exc
    first = extract_grain_batch(first_raw, expected_sequence_length=sequence_length)
    started = clock()
    output = step(
        first.token_ids,
        first.positions,
        first.segment_ids,
        first.segment_ids,
    )
    _block_until_ready(output)
    elapsed = clock() - started
    del output
    if elapsed < 0.0 or not math.isfinite(elapsed):
        raise HarvestError("compile-warm clock produced an invalid duration")

    def replay() -> Iterable[Mapping[str, Any]]:
        yield first_raw
        yield from source

    return replay(), elapsed


def host_token_offset(total_tokens: int, process_count: int, process_index: int) -> int:
    """Global scientific-token start assigned to one process."""

    if not 0 <= process_index < process_count:
        raise ValueError("invalid process index")
    return sum(
        token_quota(total_tokens, process_count, index)
        for index in range(process_index)
    )


def capture_metadata(
    config: Gemma4Config,
    *,
    checkpoint_sha256: str,
    conversion_attestation_sha256: str,
    dataset_fingerprint: str,
    split: str,
    storage_attestation_sha256: str | None = None,
    object_prefix: str | None = None,
    environment: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> CaptureMetadata:
    """Construct complete capture identity from config and environment hashes."""

    config.validate()
    fingerprint = (
        dict(environment)
        if environment is not None
        else environment_fingerprint(None if repo_root is None else Path(repo_root))
    )
    environment_sha = fingerprint.get("sha256")
    if not isinstance(environment_sha, str):
        raise ValueError("environment fingerprint must contain a sha256 string")
    return CaptureMetadata(
        checkpoint_sha256=checkpoint_sha256,
        maxtext_commit=config.maxtext_commit,
        dataset_fingerprint=dataset_fingerprint,
        split=split,
        config_sha256=config.config_hash,
        environment_sha256=environment_sha,
        conversion_attestation_sha256=conversion_attestation_sha256,
        storage_attestation_sha256=storage_attestation_sha256,
        object_prefix=object_prefix,
    )


_JOURNAL_COUNTERS = (
    "elapsed_seconds",
    "device_seconds",
    "input_wait_seconds",
    "batches_scanned",
    "batches_executed",
    "executed_scientific_tokens",
)


class _PhaseTelemetryJournal:
    """Small crash journal coupled to every immutable range commit.

    A pending record is published before the safetensor write.  On restart it
    is promoted when the immutable range exists, or discarded after retaining
    its spent-time counters when it does not.  This closes the otherwise fatal
    shard-written/telemetry-missing crash window without rewriting a growing
    range ledger (the host manifest remains the range source of truth).
    """

    def __init__(
        self,
        root: Path,
        *,
        writer: ArtifactWriter,
        metadata: CaptureMetadata,
        split: str,
        stop_after_tokens: int,
        process_index: int,
        baseline_resumed_tokens: int,
        existing_tokens: int,
        attempt_compile_seconds: float,
    ) -> None:
        if baseline_resumed_tokens < 0 or existing_tokens < 0:
            raise ValueError("journal token baselines must be nonnegative")
        self.path = _telemetry_directory(root) / (
            f"journal-p{process_index:05d}-stop-{stop_after_tokens:012d}.json"
        )
        self.writer = writer
        self._lock = threading.Lock()
        identity_payload = json.dumps(
            asdict(metadata), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        identity = hashlib.sha256(identity_payload).hexdigest()
        if self.path.exists():
            state = _read_json_mapping(self.path)
            expected = {
                "schema_version": "gemma4-harvest-phase-journal-v1",
                "identity_sha256": identity,
                "split": split,
                "stop_after_tokens": stop_after_tokens,
                "process_index": process_index,
                "baseline_resumed_tokens": baseline_resumed_tokens,
            }
            for name, value in expected.items():
                if state.get(name) != value:
                    raise HarvestError(f"phase journal identity changed at {self.path}")
        else:
            if existing_tokens != baseline_resumed_tokens:
                raise HarvestError(
                    "durable ranges exist without their phase telemetry journal; "
                    "refusing to invent performance counters"
                )
            state = {
                "schema_version": "gemma4-harvest-phase-journal-v1",
                "identity_sha256": identity,
                "split": split,
                "stop_after_tokens": stop_after_tokens,
                "process_index": process_index,
                "baseline_resumed_tokens": baseline_resumed_tokens,
                "durable_new_tokens": 0,
                "compile_seconds": 0.0,
                "complete": False,
                "pending": None,
                **{
                    name: 0.0 if name.endswith("seconds") else 0
                    for name in _JOURNAL_COUNTERS
                },
            }
        self.state = state
        self._reconcile_pending()
        accounted = int(self.state["baseline_resumed_tokens"]) + int(
            self.state["durable_new_tokens"]
        )
        if accounted != existing_tokens:
            raise HarvestError(
                f"phase journal accounts for {accounted} durable tokens, "
                f"host has {existing_tokens}"
            )
        if attempt_compile_seconds < 0.0 or not math.isfinite(attempt_compile_seconds):
            raise ValueError("attempt_compile_seconds must be finite and nonnegative")
        self.state["compile_seconds"] = float(
            self.state.get("compile_seconds", 0.0)
        ) + float(attempt_compile_seconds)
        self._persist()

    def _persist(self) -> None:
        atomic_write_json(self.path, self.state)

    def _merge_cost(self, snapshot: Mapping[str, float | int]) -> None:
        for name in _JOURNAL_COUNTERS:
            value = snapshot[name]
            if float(value) < 0 or not math.isfinite(float(value)):
                raise HarvestError(
                    f"journal counter {name} must be finite and nonnegative"
                )
            self.state[name] = max(self.state.get(name, 0), value)

    def _reconcile_pending(self) -> None:
        pending = self.state.get("pending")
        if pending is None:
            return
        if not isinstance(pending, Mapping):
            raise HarvestError(f"phase journal pending record is invalid: {self.path}")
        self._merge_cost(pending["snapshot"])
        exists = self.writer.has_range(
            str(pending["range_id"]),
            int(pending["range_start"]),
            int(pending["range_end"]),
        )
        if exists:
            self.state["durable_new_tokens"] = (
                int(self.state["durable_new_tokens"])
                + int(pending["range_end"])
                - int(pending["range_start"])
            )
        self.state["pending"] = None
        self._persist()

    def counters(self) -> dict[str, float | int]:
        return {name: self.state[name] for name in _JOURNAL_COUNTERS}

    @property
    def durable_new_tokens(self) -> int:
        return int(self.state["durable_new_tokens"])

    @property
    def compile_seconds(self) -> float:
        return float(self.state["compile_seconds"])

    def checkpoint(self, snapshot: Mapping[str, float | int]) -> None:
        with self._lock:
            self._merge_cost(snapshot)
            self._persist()

    def write_range(
        self,
        range_id: str,
        start: int,
        end: int,
        batch: CaptureBatch,
        snapshot: Mapping[str, float | int],
    ) -> dict[str, Any]:
        with self._lock:
            if self.state.get("complete") is True:
                raise HarvestError("completed phase journal cannot accept a range")
            if self.state.get("pending") is not None:
                raise HarvestError("phase journal already has a pending range")
            self._merge_cost(snapshot)
            self.state["pending"] = {
                "range_id": range_id,
                "range_start": start,
                "range_end": end,
                "snapshot": dict(snapshot),
            }
            self._persist()
        result = self.writer.write_range(range_id, start, end, batch)
        with self._lock:
            pending = self.state.get("pending")
            if not isinstance(pending, Mapping) or pending.get("range_id") != range_id:
                raise HarvestError(
                    "phase journal pending identity changed during write"
                )
            self.state["durable_new_tokens"] = int(self.state["durable_new_tokens"]) + (
                end - start
            )
            self.state["pending"] = None
            self._persist()
        return result

    def finish(
        self,
        snapshot: Mapping[str, float | int],
        *,
        exact_phase_tokens: int,
    ) -> None:
        with self._lock:
            self._merge_cost(snapshot)
            if self.state.get("pending") is not None:
                raise HarvestError(
                    "cannot finish a phase with a pending artifact write"
                )
            accounted = int(self.state["baseline_resumed_tokens"]) + int(
                self.state["durable_new_tokens"]
            )
            if accounted != exact_phase_tokens:
                raise HarvestError(
                    f"phase journal accounts for {accounted}/{exact_phase_tokens} tokens"
                )
            self.state["complete"] = True
            self._persist()


def _check_campaign_abort(
    campaign_control: HarvestCampaignControl,
    abort_collective: Callable[[bool], bool] | None,
) -> None:
    """Take one synchronized abort branch without stranding healthy peers.

    A corrupt/unreadable marker is itself a local stop request.  The affected
    controller still participates in the collective OR before surfacing its
    integrity error, so every other live controller leaves the batch boundary
    instead of entering the next model collective alone.
    """

    marker_error: Exception | None = None
    try:
        local_abort = campaign_control.abort_requested()
    except Exception as error:
        marker_error = error
        local_abort = True
    globally_aborted = (
        abort_collective(local_abort) if abort_collective is not None else local_abort
    )
    if marker_error is not None:
        raise marker_error
    campaign_control.raise_if_aborted(globally_requested=globally_aborted)


def _harvest_host_impl(
    batches: Iterable[Mapping[str, Any]],
    adapter_step: Callable[[Any, Any, Any, Any], CaptureBatch],
    *,
    artifact_root: str | Path,
    metadata: CaptureMetadata,
    split: str,
    total_tokens: int,
    process_count: int,
    process_index: int,
    stop_after_tokens: int | None = None,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    batch_collective: Callable[[bool, bool], tuple[bool, bool]] | None = None,
    baseline_resumed_tokens: int = 0,
    compile_seconds: float = 0.0,
    local_buffer_root: str | Path | None = None,
    object_prefix: str | None = None,
    object_store: ImmutableObjectStore | None = None,
    campaign_control: HarvestCampaignControl,
    abort_collective: Callable[[bool], bool] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> HarvestResult:
    """Harvest one process's deterministic quota and durably finalize its host.

    ``stop_after_tokens`` is a global prefix target.  Setting it to 2M creates
    an exact canary prefix in the same unsealed 50M artifact store.  A later
    call with the full target resumes completed batches and appends the suffix
    of the one quota-truncated boundary batch.
    """

    _validate_campaign(
        split, total_tokens, process_count, process_index, stop_after_tokens
    )
    stop = total_tokens if stop_after_tokens is None else stop_after_tokens
    quota = token_quota(stop, process_count, process_index)
    if quota <= 0:
        raise HarvestError(
            "every active process must receive at least one scientific token"
        )
    full_quota = token_quota(total_tokens, process_count, process_index)
    global_start = host_token_offset(total_tokens, process_count, process_index)
    host_id = f"p{process_index:05d}"
    root = Path(artifact_root)
    if (root / MANIFEST_NAME).exists():
        raise HarvestError(
            "artifact store is already sealed; no host harvesting is permitted"
        )

    writer = ArtifactWriter(
        root,
        host_id,
        metadata,
        local_buffer_root=local_buffer_root,
        object_prefix=object_prefix,
        object_store=object_store,
    )
    committed = _committed_parts(root, host_id, split, process_index)
    existing_manifest = (
        _read_host_manifest(root, host_id)
        if (root / "hosts" / f"host-{host_id}.json").exists()
        else {"shards": []}
    )
    existing_tokens = sum(
        int(entry["tokens"]) for entry in existing_manifest.get("shards", ())
    )
    journal = _PhaseTelemetryJournal(
        root,
        writer=writer,
        metadata=metadata,
        split=split,
        stop_after_tokens=stop,
        process_index=process_index,
        baseline_resumed_tokens=baseline_resumed_tokens,
        existing_tokens=existing_tokens,
        attempt_compile_seconds=compile_seconds,
    )
    campaign_control.heartbeat(
        "running",
        progress={
            "durable_host_tokens": existing_tokens,
            "host_quota": quota,
            "batches_scanned": int(journal.counters()["batches_scanned"]),
            "batches_executed": int(journal.counters()["batches_executed"]),
        },
        force=True,
    )
    if journal.state.get("complete") is True:
        if existing_tokens != quota:
            raise HarvestError(
                "completed phase journal does not match the exact host quota"
            )
        if stop == total_tokens and existing_manifest.get("complete") is not True:
            raise HarvestError(
                "completed production journal has an unfinalized host manifest"
            )
        if process_count == 1:
            counters = journal.counters()
            return HarvestResult(
                split=split,
                process_count=process_count,
                process_index=process_index,
                host_id=host_id,
                total_tokens=total_tokens,
                stop_after_tokens=stop,
                host_quota=quota,
                new_tokens=journal.durable_new_tokens,
                resumed_tokens=baseline_resumed_tokens,
                durable_host_tokens=existing_tokens,
                batches_scanned=int(counters["batches_scanned"]),
                batches_executed=int(counters["batches_executed"]),
                elapsed_seconds=float(counters["elapsed_seconds"]),
                device_seconds=float(counters["device_seconds"]),
                input_wait_seconds=float(counters["input_wait_seconds"]),
                max_writer_backlog=1 if journal.durable_new_tokens else 0,
                executed_scientific_tokens=int(
                    counters["executed_scientific_tokens"]
                ),
                compile_seconds=journal.compile_seconds,
            )
        # A completed lane cannot skip directly to the caller's global barrier:
        # another lane may have crashed before finishing this same prefix. Replay
        # the immutable committed prefix from local_tokens=0 so every controller
        # takes the same abort, batch-decision, and (when any peer needs it) model
        # collectives. No range is rewritten because every local part is already
        # present and validated below.
    iterator = iter(batches)
    local_tokens = 0
    prior = journal.counters()
    batches_scanned = int(prior["batches_scanned"])
    batches_executed = int(prior["batches_executed"])
    input_wait = float(prior["input_wait_seconds"])
    device_seconds = float(prior["device_seconds"])
    executed_scientific_tokens = int(prior["executed_scientific_tokens"])
    prior_elapsed = float(prior["elapsed_seconds"])
    static_shape: tuple[int, int] | None = None
    started = clock()

    def telemetry_snapshot() -> dict[str, float | int]:
        return {
            "elapsed_seconds": prior_elapsed + (clock() - started),
            "device_seconds": device_seconds,
            "input_wait_seconds": input_wait,
            "batches_scanned": batches_scanned,
            "batches_executed": batches_executed,
            "executed_scientific_tokens": executed_scientific_tokens,
        }

    with BoundedArtifactSink(journal.write_range, max_pending=1) as sink:
        for batch_index in range(2**32):
            local_complete = local_tokens >= quota
            if local_complete and batch_collective is None:
                break
            wait_started = clock()
            try:
                raw = next(iterator)
            except StopIteration as exc:
                input_wait += clock() - wait_started
                journal.checkpoint(telemetry_snapshot())
                raise HarvestError(
                    f"input exhausted at {local_tokens}/{quota} scientific tokens for {host_id}"
                ) from exc
            except BaseException:
                input_wait += clock() - wait_started
                journal.checkpoint(telemetry_snapshot())
                raise
            input_wait += clock() - wait_started
            grain = extract_grain_batch(raw, expected_sequence_length=sequence_length)
            batches_scanned += 1
            _check_campaign_abort(campaign_control, abort_collective)
            if static_shape is None:
                static_shape = grain.shape
            elif grain.shape != static_shape:
                raise HarvestError(
                    f"static Grain shape changed from {static_shape} to {grain.shape}"
                )

            host_tokens = _process_local_array(
                grain.token_ids,
                "inputs",
                process_count=process_count,
                process_index=process_index,
            )
            host_positions = _process_local_array(
                grain.positions,
                "inputs_position",
                process_count=process_count,
                process_index=process_index,
            )
            host_segments = _process_local_array(
                grain.segment_ids,
                "inputs_segmentation",
                process_count=process_count,
                process_index=process_index,
            )
            _validate_host_grain(
                host_tokens, host_positions, host_segments, sequence_length
            )
            mask = scientific_token_mask(
                host_tokens,
                host_segments,
                excluded_token_ids=EXCLUDED_TOKEN_IDS,
            )
            valid_indices = np.flatnonzero(mask.reshape(-1))
            valid_count = int(valid_indices.size)
            if valid_count == 0 and committed.get(batch_index):
                raise HarvestError(
                    f"batch {batch_index} became empty after it was committed"
                )

            desired = 0 if local_complete else min(valid_count, quota - local_tokens)
            parts = committed.get(batch_index, ())
            existing = _validate_committed_prefix(
                parts,
                batch_index=batch_index,
                expected_global_start=global_start + local_tokens,
                valid_count=valid_count,
                writer=writer,
            )
            needs_execution = existing < desired
            local_complete_after_batch = (
                local_complete or local_tokens + desired >= quota
            )
            if batch_collective is None:
                all_complete, any_needs_execution = (
                    local_complete_after_batch,
                    needs_execution,
                )
            else:
                all_complete, any_needs_execution = batch_collective(
                    local_complete_after_batch,
                    needs_execution,
                )

            full_capture: CaptureBatch | None = None
            if any_needs_execution:
                host_sequence_ids = make_sequence_ids(
                    host_segments,
                    process_index=process_index,
                    batch_index=batch_index,
                )
                device_started = clock()
                try:
                    full_capture = adapter_step(
                        grain.token_ids,
                        grain.positions,
                        grain.segment_ids,
                        grain.segment_ids,  # placeholder; host IDs are authoritative
                    )
                    _block_until_ready(full_capture)
                except BaseException:
                    device_seconds += clock() - device_started
                    journal.checkpoint(telemetry_snapshot())
                    raise
                device_seconds += clock() - device_started
                batches_executed += 1
                executed_scientific_tokens += valid_count
                journal.checkpoint(telemetry_snapshot())

            if needs_execution:
                if full_capture is None:
                    raise HarvestError(
                        "collective skipped a locally required adapter execution"
                    )
                addition = desired - existing
                compacted = _compact_adapter_output(
                    full_capture,
                    valid_indices=valid_indices[existing:desired],
                    token_shape=host_tokens.shape,
                    sequence_ids=host_sequence_ids,
                    positions=host_positions,
                    process_count=process_count,
                    process_index=process_index,
                    sequence_length=sequence_length,
                )
                range_start = global_start + local_tokens + existing
                range_end = range_start + addition
                range_id = _part_range_id(
                    split,
                    process_index,
                    batch_index,
                    existing,
                    addition,
                )
                if writer.has_range(range_id, range_start, range_end):
                    raise HarvestError(
                        "new suffix unexpectedly appeared after resume validation"
                    )
                sink.submit(
                    range_id,
                    range_start,
                    range_end,
                    compacted,
                    telemetry_snapshot(),
                )

            local_tokens += desired
            journal.checkpoint(telemetry_snapshot())
            campaign_control.heartbeat(
                "running",
                progress={
                    "durable_host_tokens": (
                        baseline_resumed_tokens + journal.durable_new_tokens
                    ),
                    "planned_host_tokens": local_tokens,
                    "host_quota": quota,
                    "batches_scanned": batches_scanned,
                    "batches_executed": batches_executed,
                    "max_writer_backlog": sink.max_observed_backlog,
                },
            )
            if all_complete:
                break

    _check_campaign_abort(campaign_control, abort_collective)
    campaign_control.heartbeat(
        "finalizing",
        progress={
            "durable_host_tokens": baseline_resumed_tokens + journal.durable_new_tokens,
            "host_quota": quota,
            "batches_scanned": batches_scanned,
            "batches_executed": batches_executed,
            "max_writer_backlog": sink.max_observed_backlog,
        },
        force=True,
    )
    if stop == total_tokens:
        host_manifest = writer.finalize_host()
    else:
        # Canary ranges are already durable, but marking a host complete would
        # seal it against appending the remaining 48M-token production suffix.
        host_manifest = _read_host_manifest(root, host_id)
    durable_tokens = sum(
        int(entry["tokens"]) for entry in host_manifest.get("shards", ())
    )
    if durable_tokens < quota:
        raise HarvestError(
            f"host finalized only {durable_tokens}/{quota} required tokens"
        )
    if stop == total_tokens and durable_tokens != full_quota:
        raise HarvestError(
            f"full host capture has {durable_tokens} tokens; exact quota is {full_quota}"
        )
    journal.finish(telemetry_snapshot(), exact_phase_tokens=quota)
    final_counters = journal.counters()
    elapsed = float(final_counters["elapsed_seconds"])
    device_seconds = float(final_counters["device_seconds"])
    input_wait = float(final_counters["input_wait_seconds"])
    batches_scanned = int(final_counters["batches_scanned"])
    batches_executed = int(final_counters["batches_executed"])
    executed_scientific_tokens = int(final_counters["executed_scientific_tokens"])
    new_tokens = journal.durable_new_tokens
    resumed_tokens = quota - new_tokens
    if resumed_tokens != baseline_resumed_tokens:
        raise HarvestError(
            f"phase resumed-token baseline changed: {resumed_tokens} != {baseline_resumed_tokens}"
        )

    return HarvestResult(
        split=split,
        process_count=process_count,
        process_index=process_index,
        host_id=host_id,
        total_tokens=total_tokens,
        stop_after_tokens=stop,
        host_quota=quota,
        new_tokens=new_tokens,
        resumed_tokens=resumed_tokens,
        durable_host_tokens=durable_tokens,
        batches_scanned=batches_scanned,
        batches_executed=batches_executed,
        elapsed_seconds=elapsed,
        device_seconds=device_seconds,
        input_wait_seconds=input_wait,
        max_writer_backlog=max(
            sink.max_observed_backlog,
            1 if journal.durable_new_tokens else 0,
        ),
        executed_scientific_tokens=executed_scientific_tokens,
        compile_seconds=journal.compile_seconds,
    )


def harvest_host(
    batches: Iterable[Mapping[str, Any]],
    adapter_step: Callable[[Any, Any, Any, Any], CaptureBatch],
    *,
    artifact_root: str | Path,
    metadata: CaptureMetadata,
    split: str,
    total_tokens: int,
    process_count: int,
    process_index: int,
    stop_after_tokens: int | None = None,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    batch_collective: Callable[[bool, bool], tuple[bool, bool]] | None = None,
    abort_collective: Callable[[bool], bool] | None = None,
    baseline_resumed_tokens: int = 0,
    compile_seconds: float = 0.0,
    local_buffer_root: str | Path | None = None,
    object_prefix: str | None = None,
    object_store: ImmutableObjectStore | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> HarvestResult:
    """Harvest one lane under a lifetime lease with durable operator status.

    The lease is process-specific, so all controllers can work concurrently
    while a duplicate launch for the same process index fails immediately.
    Abort markers are observed only at batch boundaries.  Multi-controller
    callers supply ``abort_collective`` so every controller takes the same
    branch before another collective model step.
    """

    if process_count > 1 and batch_collective is None:
        raise ValueError("multi-process harvest requires a batch collective")
    if process_count > 1 and abort_collective is None:
        raise ValueError("multi-process harvest requires an abort collective")
    stop = total_tokens if stop_after_tokens is None else stop_after_tokens
    control = HarvestCampaignControl(
        artifact_root,
        metadata.to_dict(),
        split=split,
        total_tokens=total_tokens,
        stop_after_tokens=stop,
        process_count=process_count,
        process_index=process_index,
    )
    with control:
        result = _harvest_host_impl(
            batches,
            adapter_step,
            artifact_root=artifact_root,
            metadata=metadata,
            split=split,
            total_tokens=total_tokens,
            process_count=process_count,
            process_index=process_index,
            stop_after_tokens=stop_after_tokens,
            sequence_length=sequence_length,
            batch_collective=batch_collective,
            baseline_resumed_tokens=baseline_resumed_tokens,
            compile_seconds=compile_seconds,
            local_buffer_root=local_buffer_root,
            object_prefix=object_prefix,
            object_store=object_store,
            campaign_control=control,
            abort_collective=abort_collective,
            clock=clock,
        )
        control.complete(
            {
                "durable_host_tokens": result.durable_host_tokens,
                "host_quota": result.host_quota,
                "new_tokens": result.new_tokens,
                "resumed_tokens": result.resumed_tokens,
                "batches_scanned": result.batches_scanned,
                "batches_executed": result.batches_executed,
                "max_writer_backlog": result.max_writer_backlog,
            }
        )
        return result


def publish_completed_capture(
    artifact_root: str | Path,
    metadata: CaptureMetadata,
    *,
    process_count: int,
    total_tokens: int,
    object_store: ImmutableObjectStore | None = None,
) -> Path:
    """Seal only an exact, complete set of finalized host quotas."""

    path = publish_manifest(
        artifact_root,
        metadata,
        expected_hosts=process_count,
        expected_tokens=total_tokens,
        object_store=object_store,
    )
    manifest = load_manifest(artifact_root)
    if manifest["total_tokens"] != total_tokens:
        raise HarvestError(
            f"published manifest has {manifest['total_tokens']} tokens, expected {total_tokens}"
        )
    return path


def _publish_production_after_telemetry(
    artifact_root: str | Path,
    metadata: CaptureMetadata,
    *,
    split: str,
    total_tokens: int,
    process_count: int,
    process_index: int,
    object_store: ImmutableObjectStore | None = None,
) -> float:
    status_path = _telemetry_directory(artifact_root) / (
        f"publication-stop-{total_tokens:012d}.json"
    )
    publication_started_unix: float | None = None
    prior_publication_seconds = 0.0
    if process_index == 0:
        if status_path.exists():
            prior_status = _read_json_mapping(status_path)
            prior_publication_seconds = float(
                prior_status.get("publication_seconds", 0.0)
            )
            if (
                not math.isfinite(prior_publication_seconds)
                or prior_publication_seconds < 0.0
            ):
                raise HarvestError("prior publication timing is invalid")
        publication_started_unix = time.time()
        atomic_write_json(
            status_path,
            {
                "ok": False,
                "state": "publishing",
                "publication_seconds": prior_publication_seconds,
                "attempt_started_unix": publication_started_unix,
            },
        )
        try:
            host_results = [
                _load_host_result(_host_result_path(artifact_root, total_tokens, index))
                for index in range(process_count)
            ]
            _validate_host_result_set(
                host_results,
                split=split,
                stop_after_tokens=total_tokens,
            )
            expected_manifest = str(Path(artifact_root) / MANIFEST_NAME)
            if any(item.manifest_path != expected_manifest for item in host_results):
                raise HarvestError(
                    "host telemetry was not prepared for final publication"
                )
            publish_completed_capture(
                artifact_root,
                metadata,
                process_count=process_count,
                total_tokens=total_tokens,
                object_store=object_store,
            )
            atomic_write_json(
                status_path,
                {
                    "ok": True,
                    "state": "published",
                    "publication_seconds": prior_publication_seconds
                    + time.time()
                    - publication_started_unix,
                    "attempt_started_unix": publication_started_unix,
                },
            )
        except Exception as exc:
            atomic_write_json(
                status_path,
                {
                    "ok": False,
                    "state": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "publication_seconds": prior_publication_seconds
                    + time.time()
                    - publication_started_unix,
                    "attempt_started_unix": publication_started_unix,
                },
            )
    if process_count > 1:
        multihost_barrier(f"gemma4-{split}-{total_tokens}-publication-stage")
    if process_index == 0:
        status = _read_json_mapping(status_path)
        status["publication_seconds"] = (
            prior_publication_seconds + time.time() - float(publication_started_unix)
        )
        atomic_write_json(status_path, status)
    if process_count > 1:
        multihost_barrier(f"gemma4-{split}-{total_tokens}-publication-final")
    status = _read_json_mapping(status_path)
    if status.get("ok") is not True:
        raise HarvestError(
            f"capture publication failed: {status.get('error_type', 'Error')}: "
            f"{status.get('error', 'unknown error')}"
        )
    manifest = load_manifest(artifact_root)
    if int(manifest.get("total_tokens", -1)) != total_tokens:
        raise HarvestError("published manifest has the wrong exact token count")
    publication_seconds = float(status.get("publication_seconds", -1.0))
    if not math.isfinite(publication_seconds) or publication_seconds < 0.0:
        raise HarvestError("publication timing is missing or invalid")
    return publication_seconds


def run_harvest(
    batches: Iterable[Mapping[str, Any]],
    adapter_step: Callable[[Any, Any, Any, Any], CaptureBatch],
    *,
    artifact_root: str | Path,
    metadata: CaptureMetadata,
    split: str,
    total_tokens: int,
    process_count: int,
    process_index: int,
    stop_after_tokens: int | None = None,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    barrier: Callable[[str], None] | None = None,
    batch_collective: Callable[[bool, bool], tuple[bool, bool]] | None = None,
    abort_collective: Callable[[bool], bool] | None = None,
    baseline_resumed_tokens: int = 0,
    compile_seconds: float = 0.0,
    publish: bool = True,
    local_buffer_root: str | Path | None = None,
    object_prefix: str | None = None,
    object_store: ImmutableObjectStore | None = None,
) -> HarvestResult:
    """Run one host, synchronize hosts, and coordinator-publish a full capture."""

    if process_count > 1 and barrier is None:
        raise ValueError("multi-process harvest requires an explicit global barrier")
    synchronize = barrier or (lambda _name: None)
    stop = total_tokens if stop_after_tokens is None else stop_after_tokens

    result = harvest_host(
        batches,
        adapter_step,
        artifact_root=artifact_root,
        metadata=metadata,
        split=split,
        total_tokens=total_tokens,
        process_count=process_count,
        process_index=process_index,
        stop_after_tokens=stop,
        sequence_length=sequence_length,
        batch_collective=batch_collective,
        abort_collective=abort_collective,
        baseline_resumed_tokens=baseline_resumed_tokens,
        compile_seconds=compile_seconds,
        local_buffer_root=local_buffer_root,
        object_prefix=object_prefix,
        object_store=object_store,
    )
    synchronize(f"gemma4-{split}-{stop}-hosts")

    manifest_path: Path | None = None
    if stop == total_tokens and publish:
        publication_status = _telemetry_directory(artifact_root) / (
            f"publication-stop-{stop:012d}.json"
        )
        if process_index == 0:
            try:
                manifest_path = publish_completed_capture(
                    artifact_root,
                    metadata,
                    process_count=process_count,
                    total_tokens=total_tokens,
                    object_store=object_store,
                )
                atomic_write_json(publication_status, {"ok": True})
            except Exception as exc:
                atomic_write_json(
                    publication_status,
                    {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
        synchronize(f"gemma4-{split}-{stop}-published")
        try:
            status = json.loads(publication_status.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarvestError(
                "capture publication status is missing or invalid"
            ) from exc
        if status.get("ok") is not True:
            raise HarvestError(
                f"capture publication failed: {status.get('error_type', 'Error')}: "
                f"{status.get('error', 'unknown error')}"
            )
        manifest = load_manifest(artifact_root)
        if manifest["total_tokens"] != total_tokens:
            raise HarvestError(
                "global manifest count changed after publication barrier"
            )
        manifest_path = str(Path(artifact_root) / MANIFEST_NAME)
    return _replace_result_manifest(result, manifest_path)


def benchmark_production_candidate(
    *,
    checkpoint_path: str,
    mesh_candidate: MeshCandidate,
    per_device_sequence_batch: int,
    config: Gemma4Config,
    output_directory: str | Path,
    measured_steps: int = 200,
    storage_attestation_sha256: str | None = None,
    object_prefix: str | None = None,
    local_buffer_root: str | Path | None = None,
    object_store: ImmutableObjectStore | None = None,
) -> BenchmarkObservation:
    """Create the native MaxText runtime and benchmark one v4-32 candidate."""

    config.validate()
    remote_values = (storage_attestation_sha256, object_prefix, local_buffer_root)
    if any(value is not None for value in remote_values) and not all(
        value is not None for value in remote_values
    ):
        raise ValueError(
            "benchmark remote persistence requires attestation, GCS prefix, and local buffer"
        )
    if object_prefix is not None and object_store is None:
        object_store = GCSObjectStore()
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    argv = build_maxtext_argv(
        checkpoint_path=checkpoint_path,
        grain_train_files="",
        grain_eval_files="",
        mesh=mesh_candidate,
        per_device_sequence_batch=per_device_sequence_batch,
        synthetic=True,
    )
    runtime = create_maxtext_runtime(argv)
    scratch: Path | None = None
    persistence: _ProductionBenchmarkPersist | None = None
    try:
        adapter = Gemma4PrefixAdapter.from_maxtext(runtime.model, config)
        step = compile_adapter_step(adapter)
        try:
            jax = import_module("jax")
            process_index = int(jax.process_index())
            process_count = int(jax.process_count())
            global_chips = int(jax.device_count())
        except (
            ImportError,
            ModuleNotFoundError,
        ) as exc:  # pragma: no cover - runtime already imports JAX
            raise HarvestDependencyError(
                "production benchmarking requires JAX"
            ) from exc
        source_iterator = iter(runtime.train_iterator)
        try:
            parity_raw_batch = next(source_iterator)
        except StopIteration as exc:
            raise HarvestError(
                "MaxText constructed an empty synthetic iterator"
            ) from exc
        parity_grain = extract_grain_batch(
            parity_raw_batch,
            expected_sequence_length=config.sequence_length,
        )
        with maxtext_mesh_context(runtime):
            instrumentation_equal = _run_instrumentation_parity(
                adapter,
                parity_grain,
                process_count=process_count,
            )

        def replay_parity_batch() -> Iterable[Mapping[str, Any]]:
            yield parity_raw_batch
            yield from source_iterator

        scratch = Path(
            tempfile.mkdtemp(
                prefix=(
                    f".{mesh_candidate.name}-b{per_device_sequence_batch}"
                    f"-p{process_index:05d}-"
                ),
                dir=output_directory,
            )
        )
        environment = environment_fingerprint(Path(__file__).resolve().parents[2])
        metadata = CaptureMetadata(
            checkpoint_sha256=hashlib.sha256(
                checkpoint_path.encode("utf-8")
            ).hexdigest(),
            maxtext_commit=config.maxtext_commit,
            dataset_fingerprint="0" * 64,
            split="train",
            config_sha256=config.config_hash,
            environment_sha256=str(environment["sha256"]),
            conversion_attestation_sha256="0" * 64,
            storage_attestation_sha256=storage_attestation_sha256,
            object_prefix=object_prefix,
        )
        persistence = _ProductionBenchmarkPersist(
            scratch,
            metadata=metadata,
            config=config,
            process_count=process_count,
            process_index=process_index,
            reference_instrumentation_equal=instrumentation_equal,
            local_buffer_root=local_buffer_root,
            object_prefix=object_prefix,
            object_store=object_store,
        )
        with maxtext_mesh_context(runtime):
            warmed_benchmark_batches, _benchmark_compile_seconds = (
                compile_warm_and_replay(
                    replay_parity_batch(),
                    step,
                    sequence_length=config.sequence_length,
                )
            )
            local_observation = benchmark_worker(
                warmed_benchmark_batches,
                step,
                mesh=mesh_candidate,
                per_device_sequence_batch=per_device_sequence_batch,
                measured_steps=max(measured_steps, config.benchmark_steps),
                warmup_steps=config.benchmark_warmup_steps,
                sequence_length=config.sequence_length,
                persist=persistence,
                diagnostics=persistence.diagnostics,
            )
    finally:
        try:
            if persistence is not None:
                persistence.close()
                cleanup_remote = getattr(
                    persistence, "cleanup_remote_objects", None
                )
                if cleanup_remote is not None:
                    cleanup_remote()
        finally:
            try:
                _close_maxtext_runtime(runtime)
            finally:
                if scratch is not None:
                    shutil.rmtree(scratch, ignore_errors=True)
    if persistence is None:  # pragma: no cover - successful body always constructs it
        raise AssertionError("benchmark persistence was not initialized")
    expected_output_tokens = (
        local_observation.synchronized_steps
        * per_device_sequence_batch
        * global_chips
        * config.sequence_length
    )
    if local_observation.output_tokens != expected_output_tokens:
        raise HarvestError(
            "MaxText benchmark output is not a global static batch: "
            f"{local_observation.output_tokens} != {expected_output_tokens}"
        )
    if expected_output_tokens % process_count:
        raise HarvestError(
            "benchmark global output does not partition evenly across controllers"
        )
    expected_local_persisted = expected_output_tokens // process_count
    if persistence.local_persisted_rows != expected_local_persisted:
        raise HarvestError(
            "benchmark local writer did not persist its exact static partition: "
            f"{persistence.local_persisted_rows}/{expected_local_persisted}"
        )
    atomic_write_json(
        _benchmark_host_path(
            output_directory,
            mesh_candidate,
            per_device_sequence_batch,
            process_index,
        ),
        {
            "observation": local_observation.as_dict(),
            "local_persisted_rows": persistence.local_persisted_rows,
            "process_count": process_count,
            "process_index": process_index,
            "global_chips": global_chips,
        },
    )
    return local_observation


def run_production_harvest(
    *,
    config: Gemma4Config,
    split: str,
    checkpoint_path: str,
    grain_files: str,
    staged_files: Sequence[str] | None = None,
    artifact_root: str | Path,
    checkpoint_sha256: str,
    conversion_attestation_sha256: str,
    dataset_fingerprint: str,
    storage_attestation_sha256: str,
    capture_object_prefix: str,
    local_buffer_root: str | Path,
    mesh_candidate: MeshCandidate,
    per_device_sequence_batch: int,
    stop_after_tokens: int | None = None,
    benchmark_observation: BenchmarkObservation | Mapping[str, Any] | None = None,
    publish: bool = True,
    repo_root: str | Path | None = None,
) -> CampaignSummary:
    """End-to-end MaxText/Grain entry point used by the single CLI surface."""

    config.validate()
    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")
    if not grain_files:
        raise ValueError("grain_files must be an explicit pre-staged pattern")
    if not staged_files:
        raise ValueError("staged_files must be the verified ordered manifest inventory")
    if not re.fullmatch(r"[0-9a-f]{64}", storage_attestation_sha256):
        raise ValueError("storage_attestation_sha256 must be a SHA-256 digest")
    if not capture_object_prefix.startswith("gs://"):
        raise ValueError("production capture objects require a gs:// prefix")
    if not str(local_buffer_root):
        raise ValueError("production capture requires a host-local buffer root")
    total_tokens = config.train_tokens if split == "train" else config.eval_tokens
    stop = total_tokens if stop_after_tokens is None else stop_after_tokens
    _validate_campaign(split, total_tokens, 1, 0, stop)
    phase = (
        "canary"
        if split == "train" and stop == config.canary_tokens and stop < total_tokens
        else "production"
    )
    try:
        jax = import_module("jax")
        process_count = int(jax.process_count())
        process_index = int(jax.process_index())
        global_chips = int(jax.device_count())
    except (ImportError, ModuleNotFoundError) as exc:
        raise HarvestDependencyError("production harvesting requires JAX") from exc
    fingerprint_root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    environment = environment_fingerprint(fingerprint_root)
    environment["sha256"] = _coordinator_environment_sha(
        str(environment["sha256"]),
        process_index=process_index,
        process_count=process_count,
    )
    metadata = capture_metadata(
        config,
        checkpoint_sha256=checkpoint_sha256,
        conversion_attestation_sha256=conversion_attestation_sha256,
        dataset_fingerprint=dataset_fingerprint,
        split=split,
        storage_attestation_sha256=storage_attestation_sha256,
        object_prefix=capture_object_prefix,
        environment=environment,
    )
    object_store = GCSObjectStore()
    benchmark = _benchmark_telemetry(
        benchmark_observation,
        mesh=mesh_candidate,
        per_device_sequence_batch=per_device_sequence_batch,
    )
    prior_summary: CampaignSummary | None = None
    if phase == "canary":
        if benchmark is None:
            raise HarvestError(
                "the real-data canary requires the selected passing benchmark"
            )
    elif split == "train" and total_tokens > config.canary_tokens:
        prior_summary = _load_campaign_summary(
            _campaign_summary_path(artifact_root, config.canary_tokens)
        )
        _validate_campaign_summary_identity(
            prior_summary,
            split="train",
            stop_after_tokens=config.canary_tokens,
            mesh=mesh_candidate,
            per_device_sequence_batch=per_device_sequence_batch,
            capture_metadata=metadata.to_dict(),
            benchmark=benchmark,
        )
        if prior_summary.canary_passed is not True:
            raise HarvestError(
                "the 2M real-data canary did not pass; refusing the suffix"
            )
        if prior_summary.mesh_name != mesh_candidate.name or (
            prior_summary.per_device_sequence_batch != per_device_sequence_batch
        ):
            raise HarvestError("production mesh/batch differs from the passing canary")
        if benchmark is None:
            benchmark = _benchmark_from_summary(prior_summary)
    if benchmark is None:
        raise HarvestError(
            "production harvesting requires the selected passing benchmark"
        )

    summary_path = _campaign_summary_path(artifact_root, stop)
    if summary_path.exists():
        completed = _load_campaign_summary(summary_path)
        _validate_campaign_summary_identity(
            completed,
            split=split,
            stop_after_tokens=stop,
            mesh=mesh_candidate,
            per_device_sequence_batch=per_device_sequence_batch,
            capture_metadata=metadata.to_dict(),
            capture_manifest_sha256=(
                str(load_manifest(artifact_root)["manifest_sha256"])
                if phase == "production"
                else None
            ),
            benchmark=benchmark,
        )
        if phase == "canary" and completed.canary_passed is not True:
            raise HarvestError("the existing real-data canary did not pass")
        return completed
    if (Path(artifact_root) / MANIFEST_NAME).exists():
        return _recover_sealed_campaign(
            artifact_root=artifact_root,
            config=config,
            split=split,
            phase=phase,
            total_tokens=total_tokens,
            stop_after_tokens=stop,
            mesh=mesh_candidate,
            per_device_sequence_batch=per_device_sequence_batch,
            benchmark=benchmark,
            prior_summary=prior_summary,
            expected_capture_metadata=metadata.to_dict(),
            object_store=object_store,
        )

    argv = build_maxtext_argv(
        checkpoint_path=checkpoint_path,
        # MaxText constructs both iterators in one runtime.  Only the selected
        # iterator is consumed; using the explicit split pattern for both keeps
        # the unrelated iterator from introducing a second input dependency.
        grain_train_files=grain_files,
        grain_eval_files=grain_files,
        mesh=mesh_candidate,
        per_device_sequence_batch=per_device_sequence_batch,
        synthetic=False,
    )
    runtime = create_maxtext_runtime(
        argv,
        packed_train_files=tuple(staged_files),
        packed_eval_files=tuple(staged_files),
    )
    adapter = Gemma4PrefixAdapter.from_maxtext(runtime.model, config)
    step = compile_adapter_step(adapter)
    iterator = runtime.train_iterator if split == "train" else runtime.eval_iterator
    if iterator is None:
        raise HarvestError(f"MaxText did not construct the {split} Grain iterator")
    baseline_resumed_tokens = (
        token_quota(config.canary_tokens, process_count, process_index)
        if split == "train" and phase == "production" and prior_summary is not None
        else 0
    )
    with maxtext_mesh_context(runtime):
        warmed_iterator, compile_seconds = compile_warm_and_replay(
            iterator,
            step,
            sequence_length=config.sequence_length,
        )
        host_result = run_harvest(
            warmed_iterator,
            step,
            artifact_root=artifact_root,
            metadata=metadata,
            split=split,
            total_tokens=total_tokens,
            process_count=process_count,
            process_index=process_index,
            stop_after_tokens=stop,
            sequence_length=config.sequence_length,
            barrier=multihost_barrier if process_count > 1 else None,
            batch_collective=(_jax_batch_collective if process_count > 1 else None),
            abort_collective=(_jax_abort_collective if process_count > 1 else None),
            baseline_resumed_tokens=baseline_resumed_tokens,
            compile_seconds=compile_seconds,
            # High-level orchestration publishes only after every host's
            # phase telemetry is durable, closing the manifest/metrics gap.
            publish=False,
            local_buffer_root=local_buffer_root,
            object_prefix=capture_object_prefix,
            object_store=object_store,
        )
    if stop == total_tokens and publish:
        host_result = _replace_result_manifest(
            host_result,
            str(Path(artifact_root) / MANIFEST_NAME),
        )
    host_path = _host_result_path(artifact_root, stop, process_index)
    atomic_write_json(host_path, asdict(host_result))
    if process_count > 1:
        multihost_barrier(f"gemma4-{split}-{stop}-host-telemetry")
    publication_seconds = 0.0
    if stop == total_tokens and publish:
        publication_seconds = _publish_production_after_telemetry(
            artifact_root,
            metadata,
            split=split,
            total_tokens=total_tokens,
            process_count=process_count,
            process_index=process_index,
            object_store=object_store,
        )

    if process_index == 0:
        try:
            results = [
                _load_host_result(_host_result_path(artifact_root, stop, index))
                for index in range(process_count)
            ]
            if prior_summary is not None:
                prior_results = [
                    _load_host_result(
                        _host_result_path(artifact_root, config.canary_tokens, index)
                    )
                    for index in range(process_count)
                ]
            else:
                prior_results = None
            summary = _aggregate_campaign_summary(
                results,
                artifact_root=artifact_root,
                config=config,
                split=split,
                phase=phase,
                total_tokens=total_tokens,
                stop_after_tokens=stop,
                global_chips=global_chips,
                mesh=mesh_candidate,
                per_device_sequence_batch=per_device_sequence_batch,
                benchmark=benchmark,
                capture_metadata=metadata.to_dict(),
                prior_summary=prior_summary,
                prior_results=prior_results,
                publication_seconds=publication_seconds,
            )
            atomic_write_json(summary_path, summary.as_dict())
        except Exception as exc:  # synchronize the same fail-closed error to every host
            atomic_write_json(
                summary_path,
                {
                    "schema_version": "gemma4-harvest-campaign-error-v1",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
    if process_count > 1:
        multihost_barrier(f"gemma4-{split}-{stop}-campaign-telemetry")
    summary = _load_campaign_summary(summary_path)
    if phase == "canary" and summary.canary_passed is not True:
        raise HarvestError(
            "the real-data canary failed its goodput/input/writer gates; refusing production"
        )
    _close_maxtext_runtime(runtime)
    return summary


class _ProductionBenchmarkPersist:
    """Exercise the production safetensor writer while timing a candidate."""

    def __init__(
        self,
        root: Path,
        *,
        metadata: CaptureMetadata,
        config: Gemma4Config,
        process_count: int,
        process_index: int,
        reference_instrumentation_equal: bool,
        local_buffer_root: str | Path | None = None,
        object_prefix: str | None = None,
        object_store: ImmutableObjectStore | None = None,
    ) -> None:
        self.config = config
        self.process_count = process_count
        self.process_index = process_index
        self.reference_instrumentation_equal = reference_instrumentation_equal
        self.control_root = Path(root)
        self.object_store = object_store
        self.writer = ArtifactWriter(
            root,
            f"p{process_index:05d}",
            metadata,
            local_buffer_root=local_buffer_root,
            object_prefix=object_prefix,
            object_store=object_store,
        )
        self.sink = BoundedArtifactSink(self.writer.write_range, max_pending=1)
        self.offset = 0
        self.calls = 0
        self.validated_calls = 0
        self._cache_after_warmup: int | None = None
        self._compiled_step: Any | None = None
        self._closed = False
        self._finalized = False

    def cleanup_remote_objects(self) -> None:
        """Remove only this disposable benchmark's exact object generations."""

        if self.object_store is None:
            return
        host_path = (
            self.control_root
            / "hosts"
            / f"host-p{self.process_index:05d}.json"
        )
        entries: list[Mapping[str, Any]] = []
        if host_path.exists():
            entries.extend(_read_json_mapping(host_path).get("shards", ()))
        receipt_dir = (
            self.control_root
            / "receipts"
            / f"host-p{self.process_index:05d}"
        )
        for receipt_path in sorted(receipt_dir.glob("*.json")):
            receipt = _read_json_mapping(receipt_path)
            entry = receipt.get("entry")
            if isinstance(entry, Mapping):
                entries.append(entry)
        identities = {}
        for entry in entries:
            if "object" not in entry:
                raise HarvestError("benchmark remote ledger lacks an object identity")
            identity = object_identity_from_mapping(entry["object"])
            identities[(identity.uri, identity.generation)] = identity
        for identity in identities.values():
            self.object_store.delete(identity)

    @property
    def max_observed_backlog(self) -> int:
        return self.sink.max_observed_backlog

    @property
    def local_persisted_rows(self) -> int:
        return self.offset

    def mark_warmup_complete(self, step: Callable[..., Any]) -> None:
        self._compiled_step = getattr(step, "_compiled_step", step)
        self._cache_after_warmup = _jit_cache_size(self._compiled_step)

    def __call__(self, output: CaptureBatch) -> int:
        if self._closed:
            raise HarvestError("benchmark persistence was already closed")
        activation = _process_local_array(
            output.activation,
            "benchmark activation",
            process_count=self.process_count,
            process_index=self.process_index,
            replicated_partition_unit=self.config.sequence_length,
        )
        expert_ids = _process_local_array(
            output.expert_ids,
            "benchmark expert_ids",
            process_count=self.process_count,
            process_index=self.process_index,
            replicated_partition_unit=self.config.sequence_length,
        )
        router_weights = _process_local_array(
            output.router_weights,
            "benchmark router_weights",
            process_count=self.process_count,
            process_index=self.process_index,
            replicated_partition_unit=self.config.sequence_length,
        )
        rows = int(activation.shape[0])
        if rows <= 0 or rows % self.config.sequence_length:
            raise HarvestError(
                "benchmark output is not a whole process-local sequence batch"
            )
        if expert_ids.shape != (rows, self.config.experts_per_token) or (
            router_weights.shape != (rows, self.config.experts_per_token)
        ):
            raise HarvestError(
                "benchmark routing output has an incompatible process-local shape"
            )
        sequence_rows = rows // self.config.sequence_length
        base = (np.int64(self.process_index) << np.int64(56)) | (
            np.int64(self.calls) << np.int64(24)
        )
        sequence_id = np.repeat(
            base | (np.arange(sequence_rows, dtype=np.int64) << np.int64(16)),
            self.config.sequence_length,
        )
        position = np.tile(
            np.arange(self.config.sequence_length, dtype=np.uint16),
            sequence_rows,
        )
        capture = CaptureBatch(
            activation=activation,
            expert_ids=expert_ids,
            router_weights=router_weights,
            sequence_id=sequence_id,
            position=position,
        )
        validate_capture_batch(capture, self.config, require_storage_dtypes=True)
        range_id = f"benchmark-p{self.process_index:05d}-b{self.calls:012d}"
        start, end = self.offset, self.offset + rows
        self.sink.submit(range_id, start, end, capture)
        self.offset = end
        self.calls += 1
        self.validated_calls += 1
        # Every process performs and validates its addressable fraction.  The
        # global benchmark row count is the union of those fractions.
        return _capture_rows(output)

    def flush(self) -> None:
        if not self._closed:
            self.sink.close()
            self._closed = True
        if not self._finalized:
            self.writer.finalize_host()
            self._finalized = True

    def close(self) -> None:
        self.flush()

    def diagnostics(self) -> BenchmarkDiagnostics:
        final_cache = _jit_cache_size(self._compiled_step)
        if self._cache_after_warmup is None or final_cache is None:
            recompilations = 1  # unavailable diagnostics fail the hard gate
        else:
            recompilations = max(0, final_cache - self._cache_after_warmup)
        return BenchmarkDiagnostics(
            recompilations=recompilations,
            # validate_model_architecture requires capacity_factor <= 0, so
            # this MaxText route has no finite capacity at which tokens drop.
            dropped_routed_tokens=0,
            instrumentation_equal=(
                self.reference_instrumentation_equal
                and self._finalized
                and self.validated_calls == self.calls
                and self.calls > 0
            ),
            max_writer_backlog=self.max_observed_backlog,
        )


def _benchmark_host_path(
    output_directory: str | Path,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
    process_index: int,
) -> Path:
    return (
        Path(output_directory)
        / "hosts"
        / (f"{mesh.name}-b{per_device_sequence_batch}-p{process_index:05d}.json")
    )


def _benchmark_summary_path(
    output_directory: str | Path,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
) -> Path:
    return Path(output_directory) / (
        f"{mesh.name}-b{per_device_sequence_batch}-summary.json"
    )


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarvestError(f"cannot load JSON telemetry {path}") from exc
    if not isinstance(payload, dict):
        raise HarvestError(f"JSON telemetry must be an object: {path}")
    return payload


def _benchmark_observation_from_mapping(
    value: Mapping[str, Any],
) -> BenchmarkObservation:
    raw_mesh = value.get("mesh")
    if not isinstance(raw_mesh, Mapping):
        raise HarvestError("benchmark telemetry lacks structured mesh identity")
    try:
        mesh = MeshCandidate(
            name=str(raw_mesh["name"]),
            expert_parallelism=int(raw_mesh["expert_parallelism"]),
            fsdp_parallelism=int(raw_mesh["fsdp_parallelism"]),
            tensor_parallelism=int(raw_mesh["tensor_parallelism"]),
        )
        return BenchmarkObservation(
            mesh=mesh,
            per_device_sequence_batch=int(value["per_device_sequence_batch"]),
            synchronized_steps=int(value["synchronized_steps"]),
            elapsed_seconds=float(value["elapsed_seconds"]),
            useful_flops=float(value["useful_flops"]),
            output_tokens=int(value["output_tokens"]),
            persisted_tokens=int(value["persisted_tokens"]),
            input_wait_seconds=float(value.get("input_wait_seconds", 0.0)),
            recompilations=int(value.get("recompilations", -1)),
            dropped_routed_tokens=int(value.get("dropped_routed_tokens", -1)),
            max_writer_backlog=int(value.get("max_writer_backlog", -1)),
            instrumentation_equal=value.get("instrumentation_equal") is True,
            failure=(
                str(value["failure"]) if value.get("failure") is not None else None
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HarvestError("benchmark telemetry has an incompatible schema") from exc


def _aggregate_benchmark_hosts(
    host_payloads: Sequence[Mapping[str, Any]],
    *,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
    expected_output_tokens: int,
    global_chips: int,
) -> BenchmarkObservation:
    if not host_payloads:
        raise HarvestError("benchmark host telemetry is empty")
    process_count = int(host_payloads[0].get("process_count", 0))
    indices = {int(item.get("process_index", -1)) for item in host_payloads}
    if len(host_payloads) != process_count or indices != set(range(process_count)):
        raise HarvestError(
            "benchmark host telemetry has duplicate or missing processes"
        )
    if any(int(item.get("global_chips", -1)) != global_chips for item in host_payloads):
        raise HarvestError("benchmark hosts disagree about the global chip count")
    observations = []
    for item in host_payloads:
        raw = item.get("observation")
        if not isinstance(raw, Mapping):
            raise HarvestError("benchmark host telemetry lacks an observation")
        observation = _benchmark_observation_from_mapping(raw)
        if observation.mesh != mesh or (
            observation.per_device_sequence_batch != per_device_sequence_batch
        ):
            raise HarvestError("benchmark host telemetry mixes mesh/batch candidates")
        observations.append(observation)
    reference = observations[0]
    invariant_fields = (
        "synchronized_steps",
        "useful_flops",
        "output_tokens",
        "persisted_tokens",
    )
    for name in invariant_fields:
        values = {getattr(item, name) for item in observations}
        if len(values) != 1:
            raise HarvestError(f"benchmark hosts disagree about {name}")
    if reference.output_tokens != expected_output_tokens:
        raise HarvestError(
            "benchmark global output-token count is not the static target"
        )
    local_persisted = sum(
        int(item.get("local_persisted_rows", -1)) for item in host_payloads
    )
    if local_persisted != expected_output_tokens:
        raise HarvestError(
            f"benchmark writers durably handled {local_persisted}/{expected_output_tokens} rows"
        )
    return BenchmarkObservation(
        mesh=mesh,
        per_device_sequence_batch=per_device_sequence_batch,
        synchronized_steps=reference.synchronized_steps,
        elapsed_seconds=max(item.elapsed_seconds for item in observations),
        useful_flops=reference.useful_flops,
        output_tokens=reference.output_tokens,
        persisted_tokens=reference.persisted_tokens,
        input_wait_seconds=max(item.input_wait_seconds for item in observations),
        recompilations=max(item.recompilations for item in observations),
        dropped_routed_tokens=sum(item.dropped_routed_tokens for item in observations),
        max_writer_backlog=max(item.max_writer_backlog for item in observations),
        instrumentation_equal=all(item.instrumentation_equal for item in observations),
        failure=(
            "candidate failed on at least one TPU controller"
            if any(item.failure is not None for item in observations)
            else None
        ),
    )


def benchmark_worker(
    batches: Iterable[Mapping[str, Any]],
    step: Callable[[Any, Any, Any, Any], Any],
    *,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
    persist: Callable[[Any], int] | None = None,
    diagnostics: Callable[[], BenchmarkDiagnostics] | None = None,
    warmup_steps: int = 5,
    measured_steps: int = 200,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkObservation:
    """Benchmark one mesh/batch candidate with compile-excluded synchronized steps."""

    if warmup_steps != 5 or measured_steps < 200:
        raise ValueError(
            "benchmark requires five warmups and at least 200 synchronized steps"
        )
    if per_device_sequence_batch not in (1, 2, 4):
        raise ValueError("per-device sequence batch must be 1, 2, or 4")
    iterator = iter(batches)
    persist_output = persist or _persisted_row_count

    for index in range(warmup_steps):
        grain = extract_grain_batch(
            next(iterator), expected_sequence_length=sequence_length
        )
        output = step(
            grain.token_ids, grain.positions, grain.segment_ids, grain.segment_ids
        )
        _block_until_ready(output)
    mark_warmup_complete = getattr(persist_output, "mark_warmup_complete", None)
    if callable(mark_warmup_complete):
        mark_warmup_complete(step)

    output_tokens = 0
    persisted_tokens = 0
    input_wait = 0.0
    started = clock()
    for offset in range(measured_steps):
        wait_started = clock()
        grain = extract_grain_batch(
            next(iterator), expected_sequence_length=sequence_length
        )
        input_wait += clock() - wait_started
        output = step(
            grain.token_ids, grain.positions, grain.segment_ids, grain.segment_ids
        )
        _block_until_ready(output)
        rows = _capture_rows(output)
        output_tokens += rows
        persisted = int(persist_output(output))
        if persisted < 0 or persisted > rows:
            raise HarvestError(
                f"persist callback returned invalid token count {persisted}/{rows}"
            )
        persisted_tokens += persisted
    flush = getattr(persist_output, "flush", None)
    if callable(flush):
        flush()
    elapsed = clock() - started
    if elapsed <= 0.0:
        raise HarvestError("benchmark clock did not advance")

    details = diagnostics() if diagnostics is not None else BenchmarkDiagnostics()
    callback_backlog = int(getattr(persist_output, "max_observed_backlog", 0))
    global_batch_sequences = per_device_sequence_batch * 16
    useful_flops = measured_steps * estimate_useful_prefix_flops(
        batch_sequences=global_batch_sequences,
        sequence_length=sequence_length,
    )
    return BenchmarkObservation(
        mesh=mesh,
        per_device_sequence_batch=per_device_sequence_batch,
        synchronized_steps=measured_steps,
        elapsed_seconds=elapsed,
        useful_flops=float(useful_flops),
        output_tokens=output_tokens,
        persisted_tokens=persisted_tokens,
        input_wait_seconds=input_wait,
        recompilations=details.recompilations,
        dropped_routed_tokens=details.dropped_routed_tokens,
        max_writer_backlog=max(details.max_writer_backlog, callback_backlog),
        instrumentation_equal=details.instrumentation_equal,
    )


def canary_observation(
    results: Sequence[HarvestResult],
    *,
    synthetic_goodput: float,
    expected_tokens: int = DEFAULT_CANARY_TOKENS,
) -> CanaryObservation:
    """Aggregate fresh per-host 2M canary timing into the runtime gate type."""

    if not results:
        raise ValueError("canary results cannot be empty")
    if synthetic_goodput <= 0.0:
        raise ValueError("synthetic_goodput must be positive")
    process_counts = {result.process_count for result in results}
    process_indices = {result.process_index for result in results}
    if len(process_counts) != 1 or len(results) != next(iter(process_counts)):
        raise ValueError("canary results must contain exactly one result per process")
    if process_indices != set(range(len(results))):
        raise ValueError("canary process indices are incomplete or duplicated")
    if any(result.resumed_tokens for result in results):
        raise ValueError(
            "performance canary metrics require a fresh, non-resumed prefix"
        )
    captured = sum(result.new_tokens for result in results)
    if captured != expected_tokens:
        raise ValueError(
            f"canary captured {captured} tokens, expected exactly {expected_tokens}"
        )
    elapsed = max(result.elapsed_seconds for result in results)
    device_elapsed = max(result.device_seconds for result in results)
    if elapsed <= 0.0 or device_elapsed <= 0.0:
        raise ValueError("canary timings must be positive")
    return CanaryObservation(
        captured_tokens=captured,
        real_goodput=captured / elapsed,
        synthetic_goodput=synthetic_goodput,
        input_wait_fraction=max(result.input_wait_fraction for result in results),
        persisted_goodput=captured / elapsed,
        device_output_goodput=captured / device_elapsed,
        max_writer_backlog=max(result.max_writer_backlog for result in results),
    )


def _benchmark_telemetry(
    value: BenchmarkObservation | Mapping[str, Any] | None,
    *,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, BenchmarkObservation):
        if (
            value.mesh != mesh
            or value.per_device_sequence_batch != per_device_sequence_batch
        ):
            raise HarvestError(
                "selected benchmark does not match the requested mesh/batch"
            )
        payload = value.as_dict()
    elif isinstance(value, Mapping):
        payload = dict(value)
        raw_mesh = payload.get("mesh")
        mesh_name = raw_mesh.get("name") if isinstance(raw_mesh, Mapping) else raw_mesh
        if mesh_name != mesh.name or int(
            payload.get("per_device_sequence_batch", -1)
        ) != (per_device_sequence_batch):
            raise HarvestError(
                "selected benchmark does not match the requested mesh/batch"
            )
    else:
        raise TypeError(
            "benchmark_observation must be a BenchmarkObservation or mapping"
        )

    if not bool(payload.get("passes_hard_gate", False)):
        raise HarvestError("selected synthetic benchmark did not pass every hard gate")
    synthetic = float(payload.get("persisted_tokens_per_second", 0.0))
    useful_mfu = float(payload.get("useful_mfu", 0.0))
    if not math.isfinite(synthetic) or synthetic <= 0.0:
        raise HarvestError("selected benchmark has invalid persisted goodput")
    if not math.isfinite(useful_mfu) or useful_mfu < 0.50:
        raise HarvestError("selected benchmark has invalid useful MFU")
    dropped = int(payload.get("dropped_routed_tokens", -1))
    recompilations = int(payload.get("recompilations", -1))
    instrumentation_equal = payload.get("instrumentation_equal") is True
    if dropped != 0 or recompilations != 0 or not instrumentation_equal:
        raise HarvestError("selected benchmark diagnostics are not fail-closed clean")
    return {
        "synthetic_goodput": synthetic,
        "useful_mfu": useful_mfu,
        "dropped_routed_tokens": dropped,
        "recompilations": recompilations,
        "instrumentation_equal": instrumentation_equal,
    }


def _benchmark_from_summary(summary: CampaignSummary) -> dict[str, Any]:
    if summary.synthetic_goodput is None or summary.benchmark_useful_mfu is None:
        raise HarvestError("passing canary lacks selected benchmark telemetry")
    return {
        "synthetic_goodput": summary.synthetic_goodput,
        "useful_mfu": summary.benchmark_useful_mfu,
        "dropped_routed_tokens": summary.dropped_routed_tokens,
        "recompilations": summary.recompilations,
        "instrumentation_equal": summary.instrumentation_equal,
    }


def _telemetry_directory(artifact_root: str | Path) -> Path:
    return Path(artifact_root) / "telemetry"


def _host_result_path(
    artifact_root: str | Path,
    stop_after_tokens: int,
    process_index: int,
) -> Path:
    return _telemetry_directory(artifact_root) / (
        f"host-p{process_index:05d}-stop-{stop_after_tokens:012d}.json"
    )


def _campaign_summary_path(artifact_root: str | Path, stop_after_tokens: int) -> Path:
    return _telemetry_directory(artifact_root) / (
        f"campaign-stop-{stop_after_tokens:012d}.json"
    )


def _load_host_result(path: Path) -> HarvestResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = HarvestResult(**payload)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise HarvestError(f"cannot load host harvest telemetry {path}") from exc
    if result.process_index < 0 or result.process_count <= result.process_index:
        raise HarvestError(f"invalid process identity in {path}")
    return result


def _load_campaign_summary(path: Path) -> CampaignSummary:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarvestError(f"cannot load campaign summary {path}") from exc
    if not isinstance(payload, Mapping):
        raise HarvestError(f"campaign summary must be an object: {path}")
    if payload.get("schema_version") == "gemma4-harvest-campaign-error-v1":
        raise HarvestError(
            f"campaign aggregation failed: {payload.get('error_type', 'Error')}: "
            f"{payload.get('error', 'unknown error')}"
        )
    return CampaignSummary.from_mapping(payload)


def _validate_campaign_summary_identity(
    summary: CampaignSummary,
    *,
    split: str,
    stop_after_tokens: int,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
    capture_metadata: Mapping[str, Any],
    capture_manifest_sha256: str | None = None,
    benchmark: Mapping[str, Any] | None = None,
) -> None:
    identity = dict(capture_metadata)
    if (
        summary.split != split
        or summary.stop_after_tokens != stop_after_tokens
        or summary.mesh_name != mesh.name
        or summary.per_device_sequence_batch != per_device_sequence_batch
        or summary.config_sha256 != identity.get("config_sha256")
        or summary.checkpoint_sha256 != identity.get("checkpoint_sha256")
        or summary.conversion_attestation_sha256
        != identity.get("conversion_attestation_sha256")
        or summary.dataset_fingerprint != identity.get("dataset_fingerprint")
        or summary.environment_sha256 != identity.get("environment_sha256")
        or summary.capture_manifest_sha256 != capture_manifest_sha256
    ):
        raise HarvestError("existing campaign summary does not match this invocation")
    if benchmark is not None and (
        not math.isclose(
            float(summary.synthetic_goodput or -1.0),
            float(benchmark.get("synthetic_goodput", -2.0)),
            rel_tol=1e-12,
        )
        or not math.isclose(
            float(summary.benchmark_useful_mfu or -1.0),
            float(benchmark.get("useful_mfu", -2.0)),
            rel_tol=1e-12,
        )
        or summary.dropped_routed_tokens
        != int(benchmark.get("dropped_routed_tokens", -1))
        or summary.recompilations != int(benchmark.get("recompilations", -1))
        or summary.instrumentation_equal
        is not bool(benchmark.get("instrumentation_equal", False))
    ):
        raise HarvestError("existing campaign used a different selected benchmark")


def _recover_interrupted_publication(
    artifact_root: str | Path,
    status_path: Path,
    *,
    total_tokens: int,
    process_count: int,
    process_index: int,
    split: str,
    object_store: ImmutableObjectStore | None = None,
) -> dict[str, Any]:
    """Finish telemetry after a crash between manifest sealing and status commit."""

    status = _read_json_mapping(status_path)
    if status.get("ok") is True:
        return status
    state = status.get("state")
    if state not in {"publishing", "error"}:
        raise HarvestError("sealed capture lacks recoverable publication telemetry")
    prior_seconds = float(status.get("publication_seconds", -1.0))
    if not math.isfinite(prior_seconds) or prior_seconds < 0.0:
        raise HarvestError("interrupted publication timing is invalid")
    if state == "publishing":
        raise HarvestError(
            "publication was interrupted after manifest sealing; authoritative "
            "chip-seconds are indeterminate, so this campaign cannot be recovered"
        )
    recovery_started = time.time()
    if process_index == 0:
        try:
            manifest = load_manifest(artifact_root)
            if any("object" in entry for entry in manifest["shards"]):
                if object_store is None:
                    raise HarvestError(
                        "remote publication recovery requires object-store verification"
                    )
                for entry in manifest["shards"]:
                    object_store.verify(
                        object_identity_from_mapping(entry["object"])
                    )
                verified_tokens = sum(
                    int(entry["tokens"]) for entry in manifest["shards"]
                )
            else:
                dataset = ArtifactDataset(
                    artifact_root,
                    verify_hashes=True,
                    verify_statistics=True,
                    decode_bf16=False,
                )
                verified_tokens = sum(
                    shard.batch.rows for shard in dataset.iter_shards()
                )
            if verified_tokens != total_tokens:
                raise HarvestError(
                    f"sealed recovery verified {verified_tokens}/{total_tokens} tokens"
                )
            publication_seconds = prior_seconds + time.time() - recovery_started
            atomic_write_json(
                status_path,
                {
                    "ok": True,
                    "state": "recovered",
                    "publication_seconds": publication_seconds,
                    "recovered_at_unix": time.time(),
                },
            )
        except Exception as exc:
            atomic_write_json(
                status_path,
                {
                    "ok": False,
                    "state": "recovery-error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "publication_seconds": prior_seconds
                    + time.time()
                    - recovery_started,
                },
            )
    if process_count > 1:
        multihost_barrier(f"gemma4-{split}-{total_tokens}-recover-publication")
    recovered = _read_json_mapping(status_path)
    if recovered.get("ok") is not True:
        raise HarvestError(
            "interrupted publication recovery failed: "
            f"{recovered.get('error_type', 'Error')}: "
            f"{recovered.get('error', 'unknown error')}"
        )
    return recovered


def _recover_sealed_campaign(
    *,
    artifact_root: str | Path,
    config: Gemma4Config,
    split: str,
    phase: str,
    total_tokens: int,
    stop_after_tokens: int,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
    benchmark: Mapping[str, Any],
    prior_summary: CampaignSummary | None,
    expected_capture_metadata: Mapping[str, Any],
    object_store: ImmutableObjectStore | None = None,
) -> CampaignSummary:
    """Rebuild only a missing summary after exact manifest publication."""

    manifest = load_manifest(artifact_root)
    if int(manifest.get("total_tokens", -1)) != total_tokens:
        raise HarvestError("sealed capture token count does not match this campaign")
    if manifest.get("metadata") != dict(expected_capture_metadata):
        raise HarvestError(
            "sealed capture metadata does not match the current invocation"
        )
    try:
        jax = import_module("jax")
        process_count = int(jax.process_count())
        process_index = int(jax.process_index())
        global_chips = int(jax.device_count())
    except (ImportError, ModuleNotFoundError) as exc:
        raise HarvestDependencyError("sealed campaign recovery requires JAX") from exc
    summary_path = _campaign_summary_path(artifact_root, stop_after_tokens)
    publication_status_path = _telemetry_directory(artifact_root) / (
        f"publication-stop-{total_tokens:012d}.json"
    )
    publication_status = _recover_interrupted_publication(
        artifact_root,
        publication_status_path,
        total_tokens=total_tokens,
        process_count=process_count,
        process_index=process_index,
        split=split,
        object_store=object_store,
    )
    publication_seconds = float(publication_status.get("publication_seconds", -1.0))
    if not math.isfinite(publication_seconds) or publication_seconds < 0.0:
        raise HarvestError("sealed capture lacks valid publication timing")
    if process_index == 0:
        try:
            results = [
                _load_host_result(
                    _host_result_path(artifact_root, stop_after_tokens, index)
                )
                for index in range(process_count)
            ]
            prior_results = (
                [
                    _load_host_result(
                        _host_result_path(artifact_root, config.canary_tokens, index)
                    )
                    for index in range(process_count)
                ]
                if prior_summary is not None
                else None
            )
            summary = _aggregate_campaign_summary(
                results,
                artifact_root=artifact_root,
                config=config,
                split=split,
                phase=phase,
                total_tokens=total_tokens,
                stop_after_tokens=stop_after_tokens,
                global_chips=global_chips,
                mesh=mesh,
                per_device_sequence_batch=per_device_sequence_batch,
                benchmark=benchmark,
                capture_metadata=manifest["metadata"],
                prior_summary=prior_summary,
                prior_results=prior_results,
                publication_seconds=publication_seconds,
            )
            atomic_write_json(summary_path, summary.as_dict())
        except Exception as exc:
            atomic_write_json(
                summary_path,
                {
                    "schema_version": "gemma4-harvest-campaign-error-v1",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
    if process_count > 1:
        multihost_barrier(f"gemma4-{split}-{stop_after_tokens}-recover-summary")
    recovered = _load_campaign_summary(summary_path)
    _validate_campaign_summary_identity(
        recovered,
        split=split,
        stop_after_tokens=stop_after_tokens,
        mesh=mesh,
        per_device_sequence_batch=per_device_sequence_batch,
        capture_metadata=manifest["metadata"],
        capture_manifest_sha256=str(manifest["manifest_sha256"]),
        benchmark=benchmark,
    )
    if phase == "canary" and recovered.canary_passed is not True:
        raise HarvestError("the recovered real-data canary did not pass")
    return recovered


def _validate_host_result_set(
    results: Sequence[HarvestResult],
    *,
    split: str,
    stop_after_tokens: int,
) -> None:
    if not results:
        raise HarvestError("host telemetry set is empty")
    process_count = results[0].process_count
    if len(results) != process_count or {item.process_index for item in results} != set(
        range(process_count)
    ):
        raise HarvestError("host telemetry has duplicate or missing process indices")
    for item in results:
        if (
            item.split != split
            or item.process_count != process_count
            or item.stop_after_tokens != stop_after_tokens
        ):
            raise HarvestError("host telemetry does not describe one campaign phase")
        if item.new_tokens + item.resumed_tokens != item.host_quota:
            raise HarvestError("host telemetry does not account for its exact quota")


def _aggregate_campaign_summary(
    results: Sequence[HarvestResult],
    *,
    artifact_root: str | Path,
    config: Gemma4Config,
    split: str,
    phase: str,
    total_tokens: int,
    stop_after_tokens: int,
    global_chips: int,
    mesh: MeshCandidate,
    per_device_sequence_batch: int,
    benchmark: Mapping[str, Any] | None,
    capture_metadata: Mapping[str, Any],
    prior_summary: CampaignSummary | None,
    prior_results: Sequence[HarvestResult] | None,
    publication_seconds: float = 0.0,
) -> CampaignSummary:
    _validate_host_result_set(
        results,
        split=split,
        stop_after_tokens=stop_after_tokens,
    )
    if global_chips <= 0:
        raise HarvestError("global chip count must be positive")
    if benchmark is None:
        raise HarvestError(
            "production harvest summary requires selected benchmark telemetry"
        )
    identity = dict(capture_metadata)
    required_identity = (
        "config_sha256",
        "checkpoint_sha256",
        "conversion_attestation_sha256",
        "dataset_fingerprint",
        "environment_sha256",
    )
    if any(not identity.get(name) for name in required_identity):
        raise HarvestError("capture metadata lacks campaign provenance")
    capture_manifest_sha256: str | None = None
    if phase == "production":
        manifest = load_manifest(artifact_root)
        if manifest.get("metadata") != identity:
            raise HarvestError("campaign metadata differs from the sealed manifest")
        capture_manifest_sha256 = str(manifest.get("manifest_sha256", ""))
    if not math.isfinite(publication_seconds) or publication_seconds < 0.0:
        raise HarvestError("publication_seconds must be finite and nonnegative")
    process_count = results[0].process_count
    captured_tokens = sum(item.host_quota for item in results)
    if captured_tokens != stop_after_tokens:
        raise HarvestError(
            f"host quotas sum to {captured_tokens}, require exactly {stop_after_tokens}"
        )
    durable_tokens = sum(item.durable_host_tokens for item in results)
    if durable_tokens != stop_after_tokens:
        raise HarvestError(
            f"durable host tokens sum to {durable_tokens}, require {stop_after_tokens}"
        )

    phase_wall = max(item.elapsed_seconds for item in results)
    phase_device = max(item.device_seconds for item in results)
    phase_compile = max(item.compile_seconds for item in results)
    phase_input_wait = max(item.input_wait_seconds for item in results)
    prior_wall = 0.0
    prior_device = 0.0
    prior_compile = 0.0
    prior_input_wait = 0.0
    useful_flops = 0.0
    if prior_summary is not None or prior_results is not None:
        if prior_summary is None or prior_results is None:
            raise HarvestError(
                "cumulative production telemetry requires both canary inputs"
            )
        _validate_host_result_set(
            prior_results,
            split=split,
            stop_after_tokens=config.canary_tokens,
        )
        if prior_summary.canary_passed is not True:
            raise HarvestError("cannot aggregate a suffix after a failed canary")
        if any(
            getattr(prior_summary, name) != identity[name] for name in required_identity
        ):
            raise HarvestError("canary and production capture provenance differ")
        prior_wall = prior_summary.wall_seconds
        prior_device = prior_summary.device_seconds
        prior_compile = prior_summary.compile_seconds
        prior_input_wait = (
            prior_summary.input_wait_fraction * prior_summary.wall_seconds
        )
        useful_flops += prior_summary.useful_flops

    step_counts = {item.batches_executed for item in results}
    if len(step_counts) != 1:
        raise HarvestError("SPMD hosts executed different numbers of adapter steps")
    phase_steps = next(iter(step_counts))
    if phase_steps:
        global_sequences = per_device_sequence_batch * global_chips
        full_tokens = phase_steps * global_sequences * config.sequence_length
        scientific_tokens = sum(item.executed_scientific_tokens for item in results)
        if not 0 < scientific_tokens <= full_tokens:
            raise HarvestError("executed scientific-token accounting is inconsistent")
        full_step_flops = estimate_useful_prefix_flops(
            batch_sequences=global_sequences,
            sequence_length=config.sequence_length,
        )
        useful_flops += float(full_step_flops * phase_steps) * (
            scientific_tokens / full_tokens
        )

    capture_wall = prior_wall + phase_wall
    wall = capture_wall + publication_seconds
    device = prior_device + phase_device
    compile_seconds = prior_compile + phase_compile
    if phase_wall <= 0.0 or wall <= 0.0 or device <= 0.0:
        raise HarvestError("harvest timing must advance on every completed campaign")
    input_wait_fraction = (prior_input_wait + phase_input_wait) / capture_wall
    persisted_goodput = captured_tokens / wall
    device_output_goodput = captured_tokens / device
    synthetic_goodput = float(benchmark["synthetic_goodput"])
    canary_passed: bool | None = None
    if phase == "canary":
        observation = canary_observation(
            results,
            synthetic_goodput=synthetic_goodput,
            expected_tokens=config.canary_tokens,
        )
        canary_passed = (
            observation.passes()
            and observation.real_goodput
            >= config.canary_goodput_fraction * observation.synthetic_goodput
            and observation.input_wait_fraction < config.max_input_wait_fraction
            and observation.persisted_goodput
            >= config.canary_goodput_fraction * observation.device_output_goodput
            and observation.max_writer_backlog < config.max_writer_backlog_batches
        )
    elif prior_summary is not None:
        canary_passed = prior_summary.canary_passed

    manifest_paths = {item.manifest_path for item in results}
    if len(manifest_paths) != 1:
        raise HarvestError("hosts disagree about the published capture manifest")
    manifest_path = next(iter(manifest_paths))
    if phase == "production" and manifest_path is None:
        raise HarvestError(
            "production campaign completed without publishing a manifest"
        )
    host_paths = tuple(
        str(_host_result_path(artifact_root, stop_after_tokens, index))
        for index in range(process_count)
    )
    summary = CampaignSummary(
        schema_version="gemma4-harvest-campaign-v2",
        config_sha256=str(identity["config_sha256"]),
        checkpoint_sha256=str(identity["checkpoint_sha256"]),
        conversion_attestation_sha256=str(
            identity["conversion_attestation_sha256"]
        ),
        dataset_fingerprint=str(identity["dataset_fingerprint"]),
        environment_sha256=str(identity["environment_sha256"]),
        capture_manifest_sha256=capture_manifest_sha256,
        split=split,
        phase=phase,
        stop_after_tokens=stop_after_tokens,
        total_tokens=total_tokens,
        process_count=process_count,
        global_chips=global_chips,
        mesh_name=mesh.name,
        per_device_sequence_batch=per_device_sequence_batch,
        captured_tokens=captured_tokens,
        new_tokens=sum(item.new_tokens for item in results),
        resumed_tokens=sum(item.resumed_tokens for item in results),
        phase_wall_seconds=phase_wall + publication_seconds,
        wall_seconds=wall,
        phase_device_seconds=phase_device,
        device_seconds=device,
        phase_compile_seconds=phase_compile,
        compile_seconds=compile_seconds,
        publication_seconds=publication_seconds,
        publication_chip_seconds=global_chips * publication_seconds,
        phase_chip_seconds=global_chips * (phase_wall + publication_seconds),
        chip_seconds=global_chips * wall,
        real_goodput=captured_tokens / wall,
        synthetic_goodput=synthetic_goodput,
        input_wait_fraction=input_wait_fraction,
        persisted_goodput=persisted_goodput,
        device_output_goodput=device_output_goodput,
        max_writer_backlog=max(
            [item.max_writer_backlog for item in results]
            + (
                []
                if prior_results is None
                else [item.max_writer_backlog for item in prior_results]
            )
        ),
        useful_flops=useful_flops,
        useful_mfu=useful_flops / (V4_32_BF16_PEAK_FLOPS * wall),
        benchmark_useful_mfu=float(benchmark["useful_mfu"]),
        dropped_routed_tokens=int(benchmark["dropped_routed_tokens"]),
        recompilations=int(benchmark["recompilations"]),
        instrumentation_equal=bool(benchmark["instrumentation_equal"]),
        canary_passed=canary_passed,
        manifest_path=manifest_path,
        host_result_paths=host_paths,
    )
    return summary


def multihost_barrier(name: str) -> None:
    """Synchronize JAX processes, importing the TPU runtime only when called."""

    try:
        multihost = import_module("jax.experimental.multihost_utils")
    except (ImportError, ModuleNotFoundError) as exc:
        raise HarvestDependencyError("multi-host synchronization requires JAX") from exc
    multihost.sync_global_devices(name)


def _jax_batch_collective(
    local_complete_after_batch: bool,
    local_needs_execution: bool,
) -> tuple[bool, bool]:
    """Keep host control flow identical around collective MaxText steps."""

    try:
        multihost = import_module("jax.experimental.multihost_utils")
    except (ImportError, ModuleNotFoundError) as exc:
        raise HarvestDependencyError(
            "multi-host batch coordination requires JAX"
        ) from exc
    local = np.asarray(
        [local_complete_after_batch, local_needs_execution],
        dtype=np.uint8,
    )
    gathered = np.asarray(
        multihost.process_allgather(local, tiled=False), dtype=np.uint8
    )
    values = gathered.reshape((-1, 2))
    return bool(np.all(values[:, 0])), bool(np.any(values[:, 1]))


def _jax_abort_collective(local_abort_requested: bool) -> bool:
    """Make an operator abort decision identical on every live controller."""

    try:
        multihost = import_module("jax.experimental.multihost_utils")
    except (ImportError, ModuleNotFoundError) as exc:
        raise HarvestDependencyError(
            "multi-host abort coordination requires JAX"
        ) from exc
    gathered = np.asarray(
        multihost.process_allgather(
            np.asarray(local_abort_requested, dtype=np.uint8), tiled=False
        ),
        dtype=np.uint8,
    )
    return bool(np.any(gathered))


def _validate_campaign(
    split: str,
    total_tokens: int,
    process_count: int,
    process_index: int,
    stop_after_tokens: int | None,
) -> None:
    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")
    if (
        total_tokens <= 0
        or process_count <= 0
        or not 0 <= process_index < process_count
    ):
        raise ValueError("invalid campaign token/process values")
    if stop_after_tokens is not None and not 0 < stop_after_tokens <= total_tokens:
        raise ValueError("stop_after_tokens must lie in (0, total_tokens]")


def _part_range_id(
    split: str,
    process_index: int,
    batch_index: int,
    offset: int,
    count: int,
) -> str:
    if offset < 0 or not 0 < count < 10**9:
        raise ValueError("invalid within-batch range")
    return f"{immutable_range_id(split, process_index, batch_index)}-o{offset:09d}-n{count:09d}"


def _committed_parts(
    root: Path,
    host_id: str,
    split: str,
    process_index: int,
) -> dict[int, tuple[dict[str, Any], ...]]:
    path = root / "hosts" / f"host-{host_id}.json"
    if not path.exists():
        return {}
    manifest = _read_host_manifest(root, host_id)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for entry in manifest.get("shards", ()):
        match = _PART_RE.fullmatch(str(entry.get("range_id", "")))
        if match is None:
            raise HarvestError(
                f"host manifest contains an unsupported range ID: {entry.get('range_id')!r}"
            )
        if match["split"] != split or int(match["process"]) != process_index:
            raise HarvestError(
                "host manifest range identity does not match this process/split"
            )
        copied = dict(entry)
        copied["part_offset"] = int(match["offset"])
        copied["part_count"] = int(match["count"])
        grouped.setdefault(int(match["batch"]), []).append(copied)
    return {
        batch_index: tuple(sorted(parts, key=lambda entry: entry["part_offset"]))
        for batch_index, parts in grouped.items()
    }


def _read_host_manifest(root: Path, host_id: str) -> dict[str, Any]:
    path = root / "hosts" / f"host-{host_id}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarvestError(f"cannot read resumable host manifest {path}") from exc
    if not isinstance(value, dict):
        raise HarvestError(f"host manifest must be an object: {path}")
    return value


def _validate_committed_prefix(
    parts: Sequence[Mapping[str, Any]],
    *,
    batch_index: int,
    expected_global_start: int,
    valid_count: int,
    writer: ArtifactWriter,
) -> int:
    offset = 0
    for entry in parts:
        part_offset = int(entry["part_offset"])
        part_count = int(entry["part_count"])
        if part_offset != offset or part_count <= 0:
            raise HarvestError(
                f"committed parts for batch {batch_index} are not a contiguous prefix"
            )
        if int(entry.get("tokens", -1)) != part_count:
            raise HarvestError(
                f"committed batch {batch_index} part token count changed"
            )
        start = expected_global_start + part_offset
        end = start + part_count
        if (
            int(entry.get("range_start", -1)) != start
            or int(entry.get("range_end", -1)) != end
        ):
            raise HarvestError(f"committed batch {batch_index} global range changed")
        if not writer.has_range(str(entry["range_id"]), start, end):
            raise HarvestError(
                f"committed batch {batch_index} disappeared during resume"
            )
        offset += part_count
    if offset > valid_count:
        raise HarvestError(
            f"batch {batch_index} now has {valid_count} valid tokens but {offset} were committed"
        )
    return offset


def _compact_adapter_output(
    capture: CaptureBatch,
    *,
    valid_indices: np.ndarray,
    token_shape: tuple[int, int],
    sequence_ids: np.ndarray,
    positions: np.ndarray,
    process_count: int,
    process_index: int,
    sequence_length: int,
) -> CaptureBatch:
    rows = token_shape[0] * token_shape[1]
    activation = _process_local_array(
        capture.activation,
        "activation",
        process_count=process_count,
        process_index=process_index,
        replicated_partition_unit=sequence_length,
    )
    expert_ids = _process_local_array(
        capture.expert_ids,
        "expert_ids",
        process_count=process_count,
        process_index=process_index,
        replicated_partition_unit=sequence_length,
    )
    router_weights = _process_local_array(
        capture.router_weights,
        "router_weights",
        process_count=process_count,
        process_index=process_index,
        replicated_partition_unit=sequence_length,
    )
    if activation.ndim != 2 or activation.shape[0] != rows:
        raise HarvestError(
            f"adapter activation must be full static shape [{rows}, D], got {activation.shape}"
        )
    if expert_ids.shape != (rows, 8) or router_weights.shape != (rows, 8):
        raise HarvestError(
            "adapter routing fields must have full static shape [batch*sequence, 8]"
        )
    flat_positions = np.asarray(positions).reshape((rows,))
    flat_sequence_ids = np.asarray(sequence_ids, dtype=np.int64).reshape((rows,))
    return CaptureBatch(
        activation=activation[valid_indices],
        expert_ids=expert_ids[valid_indices],
        router_weights=router_weights[valid_indices],
        sequence_id=flat_sequence_ids[valid_indices].astype(np.int64, copy=False),
        position=flat_positions[valid_indices].astype(np.uint16, copy=False),
    )


def _host_array(value: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HarvestError(
            f"{name} is not host-addressable; inject a step returning this process's local arrays"
        ) from exc


def _process_local_array(
    value: Any,
    name: str,
    *,
    process_count: int,
    process_index: int,
    replicated_partition_unit: int = 1,
) -> np.ndarray:
    """Assemble this process's addressable JAX shards without an all-gather.

    Replicated global arrays are partitioned deterministically on axis zero so
    multiple controllers never persist duplicate rows.  NumPy/test arrays are
    assumed to be process-local already.
    """

    shards = getattr(value, "addressable_shards", None)
    if shards is None:
        return _host_array(value, name)
    shards = list(shards)
    if not shards:
        raise HarvestError(f"{name} has no addressable shard on this process")

    unique: dict[tuple[tuple[int, int], ...], tuple[tuple[slice, ...], np.ndarray]] = {}
    for shard in shards:
        index = tuple(shard.index)
        normalized: list[tuple[int, int]] = []
        normalized_slices: list[slice] = []
        for axis, item in enumerate(index):
            if not isinstance(item, slice):
                raise HarvestError(f"{name} shard axis {axis} is not slice-addressable")
            start = 0 if item.start is None else int(item.start)
            stop = int(value.shape[axis]) if item.stop is None else int(item.stop)
            step = 1 if item.step is None else int(item.step)
            if step != 1 or stop <= start:
                raise HarvestError(f"{name} has an unsupported shard slice {item}")
            normalized.append((start, stop))
            normalized_slices.append(slice(start, stop))
        key = tuple(normalized)
        if key not in unique:
            unique[key] = (tuple(normalized_slices), _host_array(shard.data, name))

    minima = [min(key[axis][0] for key in unique) for axis in range(len(value.shape))]
    maxima = [max(key[axis][1] for key in unique) for axis in range(len(value.shape))]
    local_shape = tuple(maximum - minimum for minimum, maximum in zip(minima, maxima))
    first = next(iter(unique.values()))[1]
    local = np.empty(local_shape, dtype=first.dtype)
    covered = 0
    for slices, data in unique.values():
        relative = tuple(
            slice(item.start - minima[axis], item.stop - minima[axis])
            for axis, item in enumerate(slices)
        )
        expected = tuple(item.stop - item.start for item in slices)
        if data.shape != expected:
            raise HarvestError(
                f"{name} shard data shape {data.shape} != index shape {expected}"
            )
        local[relative] = data
        covered += int(data.size)
    if covered != int(local.size):
        raise HarvestError(
            f"{name} addressable shards are not one contiguous local rectangle"
        )

    global_shape = tuple(int(size) for size in value.shape)
    if process_count > 1 and local.shape == global_shape:
        if replicated_partition_unit <= 0 or local.shape[0] % replicated_partition_unit:
            raise HarvestError(
                f"{name} replicated axis cannot be partitioned by sequence"
            )
        units = local.shape[0] // replicated_partition_unit
        start = host_token_offset(units, process_count, process_index)
        count = token_quota(units, process_count, process_index)
        local = local[
            start * replicated_partition_unit : (start + count)
            * replicated_partition_unit
        ]
    return local


def _validate_host_grain(
    tokens: np.ndarray,
    positions: np.ndarray,
    segments: np.ndarray,
    sequence_length: int,
) -> None:
    if (
        tokens.shape != positions.shape
        or tokens.shape != segments.shape
        or tokens.ndim != 2
    ):
        raise HarvestError("process-local Grain fields must share rank-two token axes")
    for name, value in (
        ("inputs", tokens),
        ("inputs_position", positions),
        ("inputs_segmentation", segments),
    ):
        if not np.issubdtype(value.dtype, np.integer):
            raise HarvestError(f"{name} must have an integer dtype")
    if tokens.size and tokens.min() < 0:
        raise HarvestError("token IDs cannot be negative")
    if positions.size and (positions.min() < 0 or positions.max() >= sequence_length):
        raise HarvestError(f"positions must lie in [0, {sequence_length - 1}]")
    if segments.size and (segments.min() < 0 or segments.max() >= 2**16):
        raise HarvestError("packed segment IDs must lie in [0, 65535]")


def _block_until_ready(value: Any) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elif isinstance(value, Mapping):
        for child in value.values():
            _block_until_ready(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _block_until_ready(child)


def _capture_rows(output: Any) -> int:
    if isinstance(output, CaptureBatch):
        return output.rows
    if hasattr(output, "rows"):
        return int(output.rows)
    if hasattr(output, "shape") and len(output.shape) >= 1:
        return int(output.shape[0])
    raise TypeError("benchmark step output must expose rows or a leading shape")


def _persisted_row_count(output: Any) -> int:
    return _capture_rows(output)


def _jit_cache_size(compiled: Any) -> int | None:
    pending = [compiled]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        getter = getattr(candidate, "_cache_size", None)
        if callable(getter):
            try:
                value = int(getter())
            except (TypeError, ValueError, RuntimeError):
                return None
            return value if value >= 0 else None
        for name in ("jitted_fn", "__wrapped__", "wrapped", "_fun"):
            child = getattr(candidate, name, None)
            if child is not None:
                pending.append(child)
    return None


def _coordinator_environment_sha(
    value: str, *, process_index: int, process_count: int
) -> str:
    """Require one identical environment fingerprint on every controller."""

    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("environment SHA-256 must be lowercase hexadecimal")
    if process_count == 1:
        return value
    try:
        multihost = import_module("jax.experimental.multihost_utils")
    except (ImportError, ModuleNotFoundError) as exc:
        raise HarvestDependencyError(
            "environment fingerprint broadcast requires JAX"
        ) from exc
    del process_index
    payload = np.frombuffer(bytes.fromhex(value), dtype=np.uint8).copy()
    gathered = np.asarray(multihost.process_allgather(payload, tiled=False))
    rows = gathered.reshape(process_count, -1)
    if rows.shape != (process_count, 32) or not np.all(rows == rows[0]):
        raise HarvestError(
            "TPU controllers have different code/package/environment fingerprints"
        )
    return rows[0].tobytes().hex()


def _replace_result_manifest(result: HarvestResult, path: str | None) -> HarvestResult:
    values = {field: getattr(result, field) for field in result.__dataclass_fields__}
    values["manifest_path"] = path
    return HarvestResult(**values)


__all__ = [
    "BenchmarkDiagnostics",
    "CampaignSummary",
    "DEFAULT_CANARY_TOKENS",
    "EXCLUDED_TOKEN_IDS",
    "GrainBatch",
    "HarvestDependencyError",
    "HarvestError",
    "HarvestResult",
    "benchmark_worker",
    "benchmark_production_candidate",
    "canary_observation",
    "capture_metadata",
    "compile_warm_and_replay",
    "compile_adapter_step",
    "extract_grain_batch",
    "harvest_host",
    "host_token_offset",
    "make_sequence_ids",
    "multihost_barrier",
    "publish_completed_capture",
    "run_harvest",
    "run_production_harvest",
]
