"""Bounded asynchronous handoff from JAX capture outputs to the host writer.

The synchronized model thread first acquires one of two end-to-end buffer
permits, then validates one exact B128 output ABI, snapshots only the four
local host-metadata slices, and calls ``copy_to_host_async``.  It never waits
for device transfer, packing, or storage.  A coordinator thread later
materializes the four addressable shards by their global batch slices—not
device order—and submits them to the existing four-stream host writer.

This bridge remains production-unqualified until exercised with real
``jax.Array`` objects on v6e-64 and included in the MFU/persistence windows.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import queue
import threading
from typing import Any

import ml_dtypes
import numpy as np

from .capture_data_plane_contract import (
    CaptureDataPlaneContractError,
    CaptureWorkPlanV1,
    PINNED_CAPTURE_DATA_PLANE_CONTRACT_SHA256,
    PINNED_FRAGMENTS_PER_HOST,
    PINNED_MAX_QUEUED_COMPLETE_STEPS,
)
from .capture_fragment_codec import AddressableCaptureShardV1
from .capture_host_writer import (
    AddressableCaptureSliceV1,
    AsyncCaptureHostWriterV1,
    CaptureHostStepResultV1,
    CaptureStepTicketV1,
    HostCaptureStepV1,
    build_host_capture_step,
)
from .prefix_batchsplit_contract import (
    PINNED_CAPTURE_OUTPUTS,
    PINNED_GLOBAL_BATCH_SIZE,
    PINNED_SEQUENCE_LENGTH,
)
from .protocol import canonical_json_bytes


_STOP = object()
_TICKET_CAPABILITY = object()
_BRIDGE_CONTRACT = "two-slot-copy-to-host-async-global-slice-handoff-v1"
_BF16 = np.dtype(ml_dtypes.bfloat16)
_I32 = np.dtype(np.int32)
_F32 = np.dtype(np.float32)
_I64 = np.dtype(np.int64)
_U16 = np.dtype(np.uint16)
_LOCAL_SHAPES = {
    "routed_activation": (2, PINNED_SEQUENCE_LENGTH, 7_168),
    "global_expert_ids": (2, PINNED_SEQUENCE_LENGTH, 8),
    "scaled_router_weights": (2, PINNED_SEQUENCE_LENGTH, 8),
}
_DTYPES = {
    "routed_activation": _BF16,
    "global_expert_ids": _I32,
    "scaled_router_weights": _F32,
}


class JAXCaptureBridgeError(CaptureDataPlaneContractError):
    """The JAX output ABI, transfer lifecycle, or downstream handoff failed."""


class JAXCaptureBridgeBackpressure(JAXCaptureBridgeError):
    """Both end-to-end transfer/writer slots are occupied."""


class JAXCaptureBridgeFatal(JAXCaptureBridgeError):
    """An accepted transfer failed and permanently latched the bridge."""


@dataclasses.dataclass(frozen=True, slots=True)
class _PendingCaptureTransferV1:
    work_plan: CaptureWorkPlanV1
    step_index: int
    routed_activation: tuple[tuple[int, int, Any], ...]
    global_expert_ids: tuple[tuple[int, int, Any], ...]
    scaled_router_weights: tuple[tuple[int, int, Any], ...]
    sequence_id: tuple[tuple[int, int, np.ndarray], ...]
    position: tuple[tuple[int, int, np.ndarray], ...]


class JAXCaptureBridgeTicketV1:
    """Completion handle; callers must not wait from the model thread."""

    __slots__ = ("_error", "_event", "_result", "step_index")

    def __init__(
        self,
        step_index: int,
        *,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _TICKET_CAPABILITY:
            raise JAXCaptureBridgeError(
                "bridge tickets are available only through submit_nowait"
            )
        self.step_index = step_index
        self._event = threading.Event()
        self._result: CaptureHostStepResultV1 | None = None
        self._error: BaseException | None = None

    @property
    def done(self) -> bool:
        return self._event.is_set()

    def _succeed(self, result: CaptureHostStepResultV1) -> None:
        if self._event.is_set():
            raise JAXCaptureBridgeError("bridge ticket completed more than once")
        self._result = result
        self._event.set()

    def _fail(self, error: BaseException) -> None:
        if self._event.is_set():
            raise JAXCaptureBridgeError("bridge ticket completed more than once")
        self._error = error
        self._event.set()

    def result(self, timeout: float | None = None) -> CaptureHostStepResultV1:
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout < 0
        ):
            raise JAXCaptureBridgeError(
                "bridge ticket timeout must be finite and nonnegative"
            )
        if not self._event.wait(timeout):
            raise TimeoutError(f"JAX capture step {self.step_index} is not complete")
        if self._error is not None:
            raise JAXCaptureBridgeFatal(
                f"JAX capture step {self.step_index} failed"
            ) from self._error
        if self._result is None:  # pragma: no cover - internal event invariant
            raise JAXCaptureBridgeError("bridge ticket has no terminal result")
        return self._result


@dataclasses.dataclass(frozen=True, slots=True)
class JAXCaptureBridgeSnapshotV1:
    submitted_steps: int
    completed_steps: int
    failed_steps: int
    active_step_index: int | None
    queued_complete_steps: int
    maximum_observed_queue_depth: int
    fatal: bool
    closed: bool


class JAXCaptureBridgeV1:
    """One active transfer/handoff plus one waiting transfer."""

    def __init__(
        self,
        writer: AsyncCaptureHostWriterV1,
        *,
        array_type: type[Any] | None = None,
        array_adapter_id: str | None = None,
    ) -> None:
        if type(writer) is not AsyncCaptureHostWriterV1:
            raise JAXCaptureBridgeError(
                "JAX capture bridge requires an exact asynchronous host writer"
            )
        if array_type is None:
            try:
                import jax
            except ImportError as error:  # pragma: no cover - base dependency pinned
                raise JAXCaptureBridgeError("JAX is unavailable") from error
            array_type = jax.Array
            adapter_id = f"jax.Array@{jax.__version__}"
        else:
            if type(array_type) is not type:
                raise JAXCaptureBridgeError("array_type must be one exact class")
            if type(array_adapter_id) is not str or not array_adapter_id:
                raise JAXCaptureBridgeError(
                    "an injected array type requires an exact adapter identity"
                )
            adapter_id = f"injected-test-adapter:{array_adapter_id}"
        identity = {
            "schema": "moe-sae.r1.jax-capture-bridge-identity.v1",
            "bridge_contract": _BRIDGE_CONTRACT,
            "capture_data_plane_contract_sha256": (
                PINNED_CAPTURE_DATA_PLANE_CONTRACT_SHA256
            ),
            "array_adapter_id": adapter_id,
        }
        self._writer = writer
        self._array_type = array_type
        self._array_adapter_id = adapter_id
        self._bridge_id = (
            "jax-capture-bridge-v1:"
            + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        )
        self._production_identity_qualified = False
        self._queue: queue.Queue[
            tuple[_PendingCaptureTransferV1, JAXCaptureBridgeTicketV1] | object
        ] = queue.Queue()
        self._available_slots = threading.BoundedSemaphore(
            PINNED_MAX_QUEUED_COMPLETE_STEPS + 1
        )
        self._lock = threading.Lock()
        self._closed = False
        self._fatal_error: BaseException | None = None
        self._active_step_index: int | None = None
        self._submitted_steps = 0
        self._completed_steps = 0
        self._failed_steps = 0
        self._outstanding_steps = 0
        self._maximum_observed_queue_depth = 0
        self._thread = threading.Thread(
            target=self._run,
            name="r1-jax-capture-bridge",
            daemon=True,
        )
        self._thread.start()

    @property
    def bridge_id(self) -> str:
        return self._bridge_id

    @property
    def production_identity_qualified(self) -> bool:
        return self._production_identity_qualified

    def submit_nowait(
        self,
        *,
        work_plan: CaptureWorkPlanV1,
        step_index: int,
        routed_activation: Any,
        global_expert_ids: Any,
        scaled_router_weights: Any,
        sequence_id: np.ndarray,
        position: np.ndarray,
    ) -> JAXCaptureBridgeTicketV1:
        """Schedule D2H and return without waiting for any accepted work."""

        if type(work_plan) is not CaptureWorkPlanV1:
            raise JAXCaptureBridgeError("submit requires an exact capture work plan")
        if type(step_index) is not int or not 0 <= step_index < work_plan.step_count:
            raise JAXCaptureBridgeError("step_index is outside the capture work plan")
        with self._lock:
            if self._closed:
                raise JAXCaptureBridgeError("JAX capture bridge is closed")
            if not self._thread.is_alive():
                raise JAXCaptureBridgeFatal("bridge coordinator is not alive")
            if self._fatal_error is not None:
                raise JAXCaptureBridgeFatal("bridge is fatally latched") from (
                    self._fatal_error
                )
            if not self._available_slots.acquire(blocking=False):
                raise JAXCaptureBridgeBackpressure(
                    "both end-to-end JAX capture slots are occupied"
                )
            copy_started = False
            try:
                arrays = {
                    "routed_activation": routed_activation,
                    "global_expert_ids": global_expert_ids,
                    "scaled_router_weights": scaled_router_weights,
                }
                shard_sets = {
                    schema.name: self._validate_output_array(
                        arrays[schema.name],
                        name=schema.name,
                        global_shape=schema.shape,
                        dtype=np.dtype(schema.device_dtype),
                    )
                    for schema in PINNED_CAPTURE_OUTPUTS
                }
                starts = tuple(item[0] for item in shard_sets["routed_activation"])
                if any(
                    tuple(item[0] for item in shard_sets[name]) != starts
                    for name in ("global_expert_ids", "scaled_router_weights")
                ):
                    raise JAXCaptureBridgeError(
                        "capture output fields have different addressable global slices"
                    )
                intervals = tuple(
                    (item[0], item[1]) for item in shard_sets["routed_activation"]
                )
                sequence_snapshot = self._snapshot_metadata(
                    sequence_id,
                    name="sequence_id",
                    dtype=_I64,
                    intervals=intervals,
                )
                position_snapshot = self._snapshot_metadata(
                    position,
                    name="position",
                    dtype=_U16,
                    intervals=intervals,
                )
                for value in arrays.values():
                    copy_started = True
                    value.copy_to_host_async()
                ticket = JAXCaptureBridgeTicketV1(
                    step_index,
                    _capability=_TICKET_CAPABILITY,
                )
                pending = _PendingCaptureTransferV1(
                    work_plan=work_plan,
                    step_index=step_index,
                    routed_activation=shard_sets["routed_activation"],
                    global_expert_ids=shard_sets["global_expert_ids"],
                    scaled_router_weights=shard_sets["scaled_router_weights"],
                    sequence_id=sequence_snapshot,
                    position=position_snapshot,
                )
                self._queue.put_nowait((pending, ticket))
                self._submitted_steps += 1
                self._outstanding_steps += 1
                self._maximum_observed_queue_depth = max(
                    self._maximum_observed_queue_depth,
                    max(0, self._outstanding_steps - 1),
                )
                return ticket
            except BaseException as error:
                self._available_slots.release()
                if not copy_started:
                    raise
                fatal = JAXCaptureBridgeFatal(
                    "capture submission failed after copy_to_host_async began "
                    f"for step {step_index}"
                )
                fatal.__cause__ = error
                self._fatal_error = fatal
                raise fatal

    def snapshot(self) -> JAXCaptureBridgeSnapshotV1:
        with self._lock:
            return JAXCaptureBridgeSnapshotV1(
                submitted_steps=self._submitted_steps,
                completed_steps=self._completed_steps,
                failed_steps=self._failed_steps,
                active_step_index=self._active_step_index,
                queued_complete_steps=max(0, self._outstanding_steps - 1),
                maximum_observed_queue_depth=self._maximum_observed_queue_depth,
                fatal=self._fatal_error is not None,
                closed=self._closed,
            )

    def close(self) -> None:
        """Drain accepted transfers; the caller separately owns the writer."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(_STOP)
        self._thread.join()
        if self._thread.is_alive():  # pragma: no cover - unbounded join
            raise JAXCaptureBridgeError("bridge coordinator did not stop")

    def __enter__(self) -> JAXCaptureBridgeV1:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _validate_output_array(
        self,
        value: Any,
        *,
        name: str,
        global_shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> tuple[tuple[int, int, Any], ...]:
        if not isinstance(value, self._array_type):
            raise JAXCaptureBridgeError(f"{name} is not the bound JAX array type")
        if (
            tuple(getattr(value, "shape", ())) != global_shape
            or np.dtype(getattr(value, "dtype", None)) != dtype
        ):
            raise JAXCaptureBridgeError(
                f"{name} differs from the frozen global output ABI"
            )
        copy_to_host_async = getattr(value, "copy_to_host_async", None)
        shards = getattr(value, "addressable_shards", None)
        if not callable(copy_to_host_async) or type(shards) is not list:
            raise JAXCaptureBridgeError(
                f"{name} does not expose the exact JAX asynchronous shard API"
            )
        if len(shards) != PINNED_FRAGMENTS_PER_HOST:
            raise JAXCaptureBridgeError(
                f"{name} must have exactly four addressable shards on this host"
            )
        result = []
        for shard in shards:
            if type(getattr(shard, "replica_id", None)) is not int or (
                shard.replica_id != 0
            ):
                raise JAXCaptureBridgeError(
                    f"{name} addressable shard is replicated or ambiguous"
                )
            data = getattr(shard, "data", None)
            if not isinstance(data, self._array_type):
                raise JAXCaptureBridgeError(
                    f"{name} addressable shard data is not the bound array type"
                )
            start, stop = self._global_batch_interval(
                getattr(shard, "index", None),
                global_shape=global_shape,
                local_shape=_LOCAL_SHAPES[name],
            )
            if (
                tuple(getattr(data, "shape", ())) != _LOCAL_SHAPES[name]
                or np.dtype(getattr(data, "dtype", None)) != _DTYPES[name]
            ):
                raise JAXCaptureBridgeError(
                    f"{name} local shard differs from the frozen output ABI"
                )
            result.append((start, stop, data))
        ordered = tuple(sorted(result, key=lambda item: item[0]))
        starts = tuple(item[0] for item in ordered)
        if len(set(starts)) != PINNED_FRAGMENTS_PER_HOST:
            raise JAXCaptureBridgeError(
                f"{name} addressable global batch intervals are duplicated"
            )
        return ordered

    @staticmethod
    def _global_batch_interval(
        index: Any,
        *,
        global_shape: tuple[int, ...],
        local_shape: tuple[int, ...],
    ) -> tuple[int, int]:
        if (
            type(index) is not tuple
            or len(index) != len(global_shape)
            or any(type(item) is not slice for item in index)
        ):
            raise JAXCaptureBridgeError(
                "addressable shard index must be an exact tuple of slices"
            )
        bounds = []
        for item, global_dimension, local_dimension in zip(
            index,
            global_shape,
            local_shape,
            strict=True,
        ):
            start = 0 if item.start is None else item.start
            stop = global_dimension if item.stop is None else item.stop
            step = 1 if item.step is None else item.step
            if (
                type(start) is not int
                or type(stop) is not int
                or type(step) is not int
                or step != 1
                or not 0 <= start < stop <= global_dimension
                or stop - start != local_dimension
            ):
                raise JAXCaptureBridgeError(
                    "addressable shard slice differs from its exact local shape"
                )
            bounds.append((start, stop))
        start, stop = bounds[0]
        if (
            stop - start != 2
            or start % 2
            or bounds[1:] != [(0, dimension) for dimension in global_shape[1:]]
        ):
            raise JAXCaptureBridgeError(
                "capture output must shard only aligned two-sequence batch slices"
            )
        return start, stop

    @staticmethod
    def _snapshot_metadata(
        value: np.ndarray,
        *,
        name: str,
        dtype: np.dtype,
        intervals: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int, np.ndarray], ...]:
        expected_shape = (PINNED_GLOBAL_BATCH_SIZE, PINNED_SEQUENCE_LENGTH)
        if (
            type(value) is not np.ndarray
            or value.shape != expected_shape
            or value.dtype != dtype
            or not value.flags.c_contiguous
        ):
            raise JAXCaptureBridgeError(
                f"{name} must be one exact global C-contiguous host metadata array"
            )
        snapshots = []
        for start, stop in intervals:
            snapshot = np.array(
                value[start:stop],
                dtype=dtype,
                order="C",
                copy=True,
            )
            if snapshot.shape != (2, PINNED_SEQUENCE_LENGTH):
                raise JAXCaptureBridgeError(
                    f"{name} local metadata slice differs from the output sharding"
                )
            snapshot.setflags(write=False)
            snapshots.append((start, stop, snapshot))
        return tuple(snapshots)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                if type(item) is not tuple or len(item) != 2:
                    raise JAXCaptureBridgeError("bridge queue item is corrupt")
                pending, ticket = item
                if (
                    type(pending) is not _PendingCaptureTransferV1
                    or type(ticket) is not JAXCaptureBridgeTicketV1
                ):
                    raise JAXCaptureBridgeError(
                        "bridge queue item has invalid exact types"
                    )
                with self._lock:
                    fatal = self._fatal_error
                    self._active_step_index = pending.step_index
                if fatal is not None:
                    self._release_step_slot(completed=False)
                    ticket._fail(fatal)
                    continue
                try:
                    host_step = self._materialize_host_step(pending)
                    writer_ticket = self._writer.submit_nowait(host_step)
                    if type(writer_ticket) is not CaptureStepTicketV1:
                        raise JAXCaptureBridgeError(
                            "host writer returned an invalid ticket type"
                        )
                    result = writer_ticket.result()
                except BaseException as error:
                    fatal_error = JAXCaptureBridgeFatal(
                        f"JAX capture handoff failed for step {pending.step_index}"
                    )
                    fatal_error.__cause__ = error
                    with self._lock:
                        if self._fatal_error is None:
                            self._fatal_error = fatal_error
                        latched = self._fatal_error
                    self._release_step_slot(completed=False)
                    ticket._fail(latched)
                else:
                    self._release_step_slot(completed=True)
                    ticket._succeed(result)
            finally:
                self._queue.task_done()

    def _release_step_slot(self, *, completed: bool) -> None:
        with self._lock:
            if self._outstanding_steps <= 0 or self._active_step_index is None:
                raise JAXCaptureBridgeError("bridge slot accounting is corrupt")
            self._outstanding_steps -= 1
            self._active_step_index = None
            if completed:
                self._completed_steps += 1
            else:
                self._failed_steps += 1
        self._available_slots.release()

    @staticmethod
    def _host_array(
        value: Any,
        *,
        name: str,
    ) -> np.ndarray:
        result = np.asarray(value)
        if (
            type(result) is not np.ndarray
            or result.shape != _LOCAL_SHAPES[name]
            or result.dtype != _DTYPES[name]
            or not result.flags.c_contiguous
        ):
            raise JAXCaptureBridgeError(
                f"{name} D2H result differs from the exact contiguous local ABI"
            )
        return result

    def _materialize_host_step(
        self,
        pending: _PendingCaptureTransferV1,
    ) -> HostCaptureStepV1:
        ids = {item[0]: item for item in pending.global_expert_ids}
        weights = {item[0]: item for item in pending.scaled_router_weights}
        sequence_ids = {item[0]: item for item in pending.sequence_id}
        positions = {item[0]: item for item in pending.position}
        slices = []
        for start, stop, activation_value in pending.routed_activation:
            if (
                start not in ids
                or start not in weights
                or start not in sequence_ids
                or start not in positions
            ):
                raise JAXCaptureBridgeError(
                    "capture output global intervals changed during handoff"
                )
            shard = AddressableCaptureShardV1(
                routed_activation=self._host_array(
                    activation_value,
                    name="routed_activation",
                ),
                global_expert_ids=self._host_array(
                    ids[start][2],
                    name="global_expert_ids",
                ),
                scaled_router_weights=self._host_array(
                    weights[start][2],
                    name="scaled_router_weights",
                ),
                sequence_id=sequence_ids[start][2],
                position=positions[start][2],
            )
            slices.append(AddressableCaptureSliceV1(start, stop, shard))
        return build_host_capture_step(
            work_plan=pending.work_plan,
            step_index=pending.step_index,
            addressable_slices=tuple(slices),
        )


__all__ = [
    "JAXCaptureBridgeBackpressure",
    "JAXCaptureBridgeError",
    "JAXCaptureBridgeFatal",
    "JAXCaptureBridgeSnapshotV1",
    "JAXCaptureBridgeTicketV1",
    "JAXCaptureBridgeV1",
]
