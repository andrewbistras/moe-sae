"""Nonblocking two-slot host writer for provisional R1 capture fragments.

The caller transfers exclusive ownership of four host-resident addressable
shards to :class:`AsyncCaptureHostWriterV1` and receives a ticket immediately.
One step may be active and one complete step may wait.  A third submission
fails synchronously; the numerical path never sleeps for storage.

Four worker threads independently pack and create the host's four fragments.
Any codec, sink, or receipt mismatch latches a fatal writer state.  Other
in-flight creates may leave immutable provisional objects, but no failed step
returns a result and no later step is accepted.  Full readback, manifest
publication, qualification, and production admission are deliberately outside
this module.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
import queue
import threading
import time
from typing import Any, ClassVar, Protocol

from .capture_data_plane_contract import (
    CaptureDataPlaneContractError,
    CaptureFragmentPlanV1,
    CaptureWorkPlanV1,
    PINNED_FRAGMENTS_PER_HOST,
    PINNED_MAX_QUEUED_COMPLETE_STEPS,
    PINNED_UPLOAD_STREAMS_PER_HOST,
)
from .capture_fragment_codec import (
    AddressableCaptureShardV1,
    CaptureFragmentCodecError,
    PackedCaptureFragmentV1,
    pack_capture_fragment,
)
from .gcs_immutable_byte_store import GCSPhysicalObjectReceiptV1
from .protocol import (
    CanonicalRecord,
    ProtocolError,
    SHA256_PATTERN,
    require_string,
)


_STOP = object()
_RESULT_CAPABILITY = object()
_TICKET_CAPABILITY = object()
_RECEIPT_CONTRACT = "provisional-create-ack-with-deferred-full-readback-v1"


class CaptureHostWriterError(CaptureDataPlaneContractError):
    """A host writer lifecycle, sink, or receipt invariant failed."""


class CaptureHostWriterBackpressure(CaptureHostWriterError):
    """The two writer slots are occupied; the caller must fail the window."""


class CaptureHostWriterFatal(CaptureHostWriterError):
    """The writer latched a fatal asynchronous pack or create failure."""


class ProvisionalCaptureFragmentSink(Protocol):
    """Future fast GCS adapter; a create acknowledgement is not readback."""

    def create_provisional(
        self,
        packed: PackedCaptureFragmentV1,
    ) -> ProvisionalCaptureFragmentReceiptV1: ...


@dataclass(frozen=True, slots=True)
class ProvisionalCaptureFragmentReceiptV1(CanonicalRecord):
    """One exact object-create acknowledgement with hard-false later gates."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.provisional-capture-fragment-receipt.v1"

    fragment_plan_sha256: str
    physical_receipt: GCSPhysicalObjectReceiptV1
    object_create_acknowledged: bool = True
    full_payload_readback_verified: bool = False
    manifest_visible: bool = False
    production_sealable: bool = False
    receipt_contract: str = _RECEIPT_CONTRACT

    def __post_init__(self) -> None:
        if type(self.fragment_plan_sha256) is not str:
            raise CaptureHostWriterError(
                "fragment_plan_sha256 must be an exact immutable string"
            )
        try:
            require_string(
                self.fragment_plan_sha256,
                "fragment_plan_sha256",
                pattern=SHA256_PATTERN,
            )
        except ProtocolError as error:
            raise CaptureHostWriterError(
                "fragment_plan_sha256 is not a SHA-256"
            ) from error
        if type(self.physical_receipt) is not GCSPhysicalObjectReceiptV1:
            raise CaptureHostWriterError(
                "physical_receipt must be an exact GCS receipt"
            )
        if (
            type(self.object_create_acknowledged) is not bool
            or self.object_create_acknowledged is not True
        ):
            raise CaptureHostWriterError(
                "provisional receipt requires an exact create acknowledgement"
            )
        for name in (
            "full_payload_readback_verified",
            "manifest_visible",
            "production_sealable",
        ):
            if (
                type(getattr(self, name)) is not bool
                or getattr(self, name) is not False
            ):
                raise CaptureHostWriterError(
                    f"{name} must remain exact false for a provisional receipt"
                )
        if (
            type(self.receipt_contract) is not str
            or self.receipt_contract != _RECEIPT_CONTRACT
        ):
            raise CaptureHostWriterError("provisional receipt contract changed")
        if self.physical_receipt.content_type != "application/octet-stream":
            raise CaptureHostWriterError(
                "capture fragment physical receipt must describe binary content"
            )

    @classmethod
    def from_packed(
        cls,
        packed: PackedCaptureFragmentV1,
        physical_receipt: GCSPhysicalObjectReceiptV1,
    ) -> ProvisionalCaptureFragmentReceiptV1:
        if type(packed) is not PackedCaptureFragmentV1:
            raise CaptureHostWriterError("packed fragment has an invalid type")
        result = cls(
            fragment_plan_sha256=packed.plan.digest,
            physical_receipt=physical_receipt,
        )
        result.require_matches(packed)
        return result

    def require_matches(self, packed: PackedCaptureFragmentV1) -> None:
        if type(packed) is not PackedCaptureFragmentV1:
            raise CaptureHostWriterError("packed fragment has an invalid type")
        receipt = self.physical_receipt
        if (
            self.fragment_plan_sha256 != packed.plan.digest
            or receipt.object_id != packed.object_id
            or receipt.sha256 != packed.sha256
            or receipt.crc32c != packed.crc32c
            or receipt.size_bytes != len(packed.payload)
        ):
            raise CaptureHostWriterError(
                "provisional physical receipt differs from the packed fragment"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "fragment_plan_sha256": self.fragment_plan_sha256,
            "physical_receipt": self.physical_receipt.as_dict(),
            "object_create_acknowledged": self.object_create_acknowledged,
            "full_payload_readback_verified": (self.full_payload_readback_verified),
            "manifest_visible": self.manifest_visible,
            "production_sealable": self.production_sealable,
            "receipt_contract": self.receipt_contract,
        }


@dataclass(frozen=True, slots=True)
class AddressableCaptureSliceV1:
    """One observed global batch slice and its exclusively owned host buffer."""

    global_batch_start: int
    global_batch_stop: int
    shard: AddressableCaptureShardV1

    def __post_init__(self) -> None:
        if (
            type(self.global_batch_start) is not int
            or type(self.global_batch_stop) is not int
            or self.global_batch_start < 0
            or self.global_batch_stop - self.global_batch_start != 2
            or self.global_batch_start % 2
            or self.global_batch_stop > 128
        ):
            raise CaptureHostWriterError(
                "addressable slice must be one aligned two-sequence interval"
            )
        if type(self.shard) is not AddressableCaptureShardV1:
            raise CaptureHostWriterError(
                "addressable slice must hold an exact capture shard"
            )


@dataclass(frozen=True, slots=True)
class HostCaptureStepV1:
    """Four addressable slices; host/device identity is intentionally absent."""

    work_plan: CaptureWorkPlanV1
    step_index: int
    addressable_slices: tuple[AddressableCaptureSliceV1, ...]

    def __post_init__(self) -> None:
        if type(self.work_plan) is not CaptureWorkPlanV1:
            raise CaptureHostWriterError("host step requires an exact work plan")
        if (
            type(self.step_index) is not int
            or not 0 <= self.step_index < self.work_plan.step_count
        ):
            raise CaptureHostWriterError("host step index is outside its work plan")
        if (
            type(self.addressable_slices) is not tuple
            or len(self.addressable_slices) != PINNED_FRAGMENTS_PER_HOST
            or any(
                type(item) is not AddressableCaptureSliceV1
                for item in self.addressable_slices
            )
        ):
            raise CaptureHostWriterError(
                "host step must contain exactly four addressable slices"
            )
        starts = tuple(item.global_batch_start for item in self.addressable_slices)
        if starts != tuple(sorted(starts)) or len(set(starts)) != len(starts):
            raise CaptureHostWriterError(
                "host addressable slices must be unique and canonically sorted"
            )

    @property
    def fragment_plans(self) -> tuple[CaptureFragmentPlanV1, ...]:
        result = []
        for item in self.addressable_slices:
            plan = self.work_plan.fragment_for_global_batch_slice(
                self.step_index,
                item.global_batch_start,
                item.global_batch_stop,
            )
            if plan is not None:
                result.append(plan)
        return tuple(result)


def build_host_capture_step(
    *,
    work_plan: CaptureWorkPlanV1,
    step_index: int,
    addressable_slices: tuple[AddressableCaptureSliceV1, ...],
) -> HostCaptureStepV1:
    """Canonicalize local device enumeration by global batch interval."""

    if (
        type(addressable_slices) is not tuple
        or len(addressable_slices) != PINNED_FRAGMENTS_PER_HOST
        or any(
            type(item) is not AddressableCaptureSliceV1 for item in addressable_slices
        )
    ):
        raise CaptureHostWriterError(
            "addressable_slices must be an exact four-item tuple"
        )
    ordered = tuple(
        sorted(addressable_slices, key=lambda item: item.global_batch_start)
    )
    return HostCaptureStepV1(
        work_plan=work_plan,
        step_index=step_index,
        addressable_slices=ordered,
    )


@dataclass(frozen=True, slots=True, init=False)
class CaptureHostStepResultV1:
    """One complete host step of provisional receipts, never a manifest seal."""

    step_index: int
    fragment_plan_sha256s: tuple[str, ...]
    receipts: tuple[ProvisionalCaptureFragmentReceiptV1, ...]
    payload_bytes: int
    duration_ns: int
    full_payload_readback_verified: bool
    manifest_visible: bool
    production_sealable: bool

    def __init__(
        self,
        step: HostCaptureStepV1,
        receipts: tuple[ProvisionalCaptureFragmentReceiptV1, ...],
        *,
        payload_bytes: int,
        duration_ns: int,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _RESULT_CAPABILITY:
            raise CaptureHostWriterError(
                "host results are available only through the async writer"
            )
        if (
            type(step) is not HostCaptureStepV1
            or type(receipts) is not tuple
            or any(
                type(item) is not ProvisionalCaptureFragmentReceiptV1
                for item in receipts
            )
        ):
            raise CaptureHostWriterError("host result state has invalid types")
        plans = step.fragment_plans
        expected_digests = tuple(item.digest for item in plans)
        actual_digests = tuple(item.fragment_plan_sha256 for item in receipts)
        if actual_digests != expected_digests:
            raise CaptureHostWriterError(
                "host receipts do not exactly cover canonical fragment plans"
            )
        if type(payload_bytes) is not int or payload_bytes != sum(
            item.physical_receipt.size_bytes for item in receipts
        ):
            raise CaptureHostWriterError("host result byte count is not exact")
        if type(duration_ns) is not int or duration_ns <= 0:
            raise CaptureHostWriterError("host result duration must be positive")
        object.__setattr__(self, "step_index", step.step_index)
        object.__setattr__(self, "fragment_plan_sha256s", expected_digests)
        object.__setattr__(self, "receipts", receipts)
        object.__setattr__(self, "payload_bytes", payload_bytes)
        object.__setattr__(self, "duration_ns", duration_ns)
        object.__setattr__(self, "full_payload_readback_verified", False)
        object.__setattr__(self, "manifest_visible", False)
        object.__setattr__(self, "production_sealable", False)


class CaptureStepTicketV1:
    """Thread-safe completion handle; waiting is forbidden in the model path."""

    __slots__ = ("_error", "_event", "_result", "step_index")

    def __init__(
        self,
        step_index: int,
        *,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _TICKET_CAPABILITY:
            raise CaptureHostWriterError(
                "capture tickets are available only through submit_nowait"
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
            raise CaptureHostWriterError("capture ticket completed more than once")
        self._result = result
        self._event.set()

    def _fail(self, error: BaseException) -> None:
        if self._event.is_set():
            raise CaptureHostWriterError("capture ticket completed more than once")
        self._error = error
        self._event.set()

    def result(self, timeout: float | None = None) -> CaptureHostStepResultV1:
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout < 0
        ):
            raise CaptureHostWriterError(
                "ticket timeout must be finite and nonnegative"
            )
        if not self._event.wait(timeout):
            raise TimeoutError(f"capture step {self.step_index} is not complete")
        if self._error is not None:
            raise CaptureHostWriterFatal(
                f"capture step {self.step_index} failed"
            ) from self._error
        if self._result is None:  # pragma: no cover - event state is internal
            raise CaptureHostWriterError("capture ticket has no terminal state")
        return self._result


@dataclass(frozen=True, slots=True)
class CaptureHostWriterSnapshotV1:
    submitted_steps: int
    completed_steps: int
    failed_steps: int
    active_step_index: int | None
    queued_complete_steps: int
    maximum_observed_queue_depth: int
    fatal: bool
    closed: bool


class AsyncCaptureHostWriterV1:
    """One active host step plus one waiting step, with four upload workers."""

    def __init__(self, sink: ProvisionalCaptureFragmentSink) -> None:
        create = getattr(sink, "create_provisional", None)
        if not callable(create):
            raise CaptureHostWriterError(
                "provisional sink must implement create_provisional"
            )
        self._sink = sink
        self._queue: queue.Queue[
            tuple[HostCaptureStepV1, CaptureStepTicketV1] | object
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
            name="r1-capture-host-writer",
            daemon=True,
        )
        self._thread.start()

    def submit_nowait(self, step: HostCaptureStepV1) -> CaptureStepTicketV1:
        """Transfer one host step without waiting for queue or storage."""

        if type(step) is not HostCaptureStepV1:
            raise CaptureHostWriterError("submit requires an exact host step")
        with self._lock:
            if self._closed:
                raise CaptureHostWriterError("capture writer is closed")
            if not self._thread.is_alive():
                raise CaptureHostWriterFatal("capture writer coordinator is not alive")
            if self._fatal_error is not None:
                raise CaptureHostWriterFatal("capture writer is fatally latched") from (
                    self._fatal_error
                )
            ticket = CaptureStepTicketV1(
                step.step_index,
                _capability=_TICKET_CAPABILITY,
            )
            if not self._available_slots.acquire(blocking=False):
                raise CaptureHostWriterBackpressure(
                    "both host writer slots are occupied"
                )
            self._queue.put_nowait((step, ticket))
            self._submitted_steps += 1
            self._outstanding_steps += 1
            self._maximum_observed_queue_depth = max(
                self._maximum_observed_queue_depth,
                max(0, self._outstanding_steps - 1),
            )
            return ticket

    def snapshot(self) -> CaptureHostWriterSnapshotV1:
        with self._lock:
            return CaptureHostWriterSnapshotV1(
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
        """Drain accepted work and stop; call only outside the numerical path."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(_STOP)
        self._thread.join()
        if self._thread.is_alive():  # pragma: no cover - join has no timeout
            raise CaptureHostWriterError("capture writer did not stop")

    def __enter__(self) -> AsyncCaptureHostWriterV1:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _run(self) -> None:
        with ThreadPoolExecutor(
            max_workers=PINNED_UPLOAD_STREAMS_PER_HOST,
            thread_name_prefix="r1-capture-fragment",
        ) as pool:
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        return
                    if type(item) is not tuple or len(item) != 2:
                        raise CaptureHostWriterError(
                            "capture writer queue item is corrupt"
                        )
                    step, ticket = item
                    if (
                        type(step) is not HostCaptureStepV1
                        or type(ticket) is not CaptureStepTicketV1
                    ):
                        raise CaptureHostWriterError(
                            "capture writer queue item has invalid types"
                        )
                    with self._lock:
                        fatal = self._fatal_error
                        self._active_step_index = step.step_index
                    if fatal is not None:
                        self._release_step_slot(completed=False)
                        ticket._fail(fatal)
                        continue
                    try:
                        result = self._process_step(step, pool)
                    except BaseException as error:
                        fatal_error = CaptureHostWriterFatal(
                            f"host capture step {step.step_index} failed"
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
                raise CaptureHostWriterError(
                    "capture writer slot accounting is corrupt"
                )
            self._outstanding_steps -= 1
            self._active_step_index = None
            if completed:
                self._completed_steps += 1
            else:
                self._failed_steps += 1
        self._available_slots.release()

    def _process_step(
        self,
        step: HostCaptureStepV1,
        pool: ThreadPoolExecutor,
    ) -> CaptureHostStepResultV1:
        started_ns = time.monotonic_ns()
        work: list[
            tuple[
                CaptureFragmentPlanV1,
                Future[ProvisionalCaptureFragmentReceiptV1],
            ]
        ] = []
        for item in step.addressable_slices:
            plan = step.work_plan.fragment_for_global_batch_slice(
                step.step_index,
                item.global_batch_start,
                item.global_batch_stop,
            )
            if plan is None:
                continue
            work.append(
                (
                    plan,
                    pool.submit(self._pack_and_create, plan, item.shard),
                )
            )
        receipts: list[ProvisionalCaptureFragmentReceiptV1] = []
        try:
            for _plan, future in work:
                receipts.append(future.result())
        except BaseException:
            for _plan, future in work:
                future.cancel()
            raise
        duration_ns = max(1, time.monotonic_ns() - started_ns)
        receipt_tuple = tuple(receipts)
        return CaptureHostStepResultV1(
            step,
            receipt_tuple,
            payload_bytes=sum(
                item.physical_receipt.size_bytes for item in receipt_tuple
            ),
            duration_ns=duration_ns,
            _capability=_RESULT_CAPABILITY,
        )

    def _pack_and_create(
        self,
        plan: CaptureFragmentPlanV1,
        shard: AddressableCaptureShardV1,
    ) -> ProvisionalCaptureFragmentReceiptV1:
        try:
            packed = pack_capture_fragment(plan, shard)
        except (CaptureFragmentCodecError, CaptureDataPlaneContractError) as error:
            raise CaptureHostWriterError("capture fragment packing failed") from error
        receipt = self._sink.create_provisional(packed)
        if type(receipt) is not ProvisionalCaptureFragmentReceiptV1:
            raise CaptureHostWriterError(
                "provisional sink returned an invalid receipt type"
            )
        receipt.require_matches(packed)
        return receipt


__all__ = [
    "AddressableCaptureSliceV1",
    "AsyncCaptureHostWriterV1",
    "CaptureHostStepResultV1",
    "CaptureHostWriterBackpressure",
    "CaptureHostWriterError",
    "CaptureHostWriterFatal",
    "CaptureHostWriterSnapshotV1",
    "CaptureStepTicketV1",
    "HostCaptureStepV1",
    "ProvisionalCaptureFragmentReceiptV1",
    "ProvisionalCaptureFragmentSink",
    "build_host_capture_step",
]
