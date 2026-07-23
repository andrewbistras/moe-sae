from __future__ import annotations

from dataclasses import replace
import math
import threading
import time

import ml_dtypes
import numpy as np
import pytest

from moe_sae.r1.capture_data_plane_contract import build_capture_work_plan
from moe_sae.r1.capture_fragment_codec import (
    AddressableCaptureShardV1,
    PackedCaptureFragmentV1,
    pack_capture_fragment,
)
from moe_sae.r1.capture_host_writer import (
    AddressableCaptureSliceV1,
    AsyncCaptureHostWriterV1,
    CaptureHostWriterBackpressure,
    CaptureHostWriterError,
    CaptureHostWriterFatal,
    HostCaptureStepV1,
    ProvisionalCaptureFragmentReceiptV1,
    build_host_capture_step,
)
from moe_sae.r1.gcs_immutable_byte_store import GCSPhysicalObjectReceiptV1


_BF16 = np.dtype(ml_dtypes.bfloat16)


@pytest.fixture(scope="module")
def addressable_shard() -> AddressableCaptureShardV1:
    activation = np.zeros((2, 1_024, 7_168), dtype=_BF16)
    ids = np.broadcast_to(
        np.arange(8, dtype=np.int32),
        (2, 1_024, 8),
    ).copy()
    weights = np.full(
        (2, 1_024, 8),
        np.float32(2.5 / 8),
        dtype=np.float32,
    )
    sequence_id = np.broadcast_to(
        np.asarray([100, 101], dtype=np.int64)[:, None],
        (2, 1_024),
    ).copy()
    position = np.broadcast_to(
        np.arange(1_024, dtype=np.uint16),
        (2, 1_024),
    ).copy()
    return AddressableCaptureShardV1(
        activation,
        ids,
        weights,
        sequence_id,
        position,
    )


def _host_step(
    addressable_shard: AddressableCaptureShardV1,
    *,
    token_count: int = 131_072,
    step_index: int = 0,
    starts: tuple[int, ...] = (0, 2, 4, 6),
) -> HostCaptureStepV1:
    work = build_capture_work_plan(
        work_id=f"writer-range-{token_count}",
        input_range_sha256="a" * 64,
        start_token_ordinal=0,
        token_count=token_count,
    )
    slices = tuple(
        AddressableCaptureSliceV1(start, start + 2, addressable_shard)
        for start in reversed(starts)
    )
    return build_host_capture_step(
        work_plan=work,
        step_index=step_index,
        addressable_slices=slices,
    )


class _Sink:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self.calls: list[str] = []

    def create_provisional(
        self,
        packed: PackedCaptureFragmentV1,
    ) -> ProvisionalCaptureFragmentReceiptV1:
        with self._lock:
            self._generation += 1
            generation = str(self._generation)
            self.calls.append(packed.plan.digest)
        physical = GCSPhysicalObjectReceiptV1(
            store_id="writer-test-store",
            object_id=packed.object_id,
            uri=f"gs://writer-test-bucket/attempt/{packed.object_id}",
            region="us-central1",
            generation=generation,
            metageneration="1",
            size_bytes=len(packed.payload),
            crc32c=packed.crc32c,
            sha256=packed.sha256,
            content_type="application/octet-stream",
        )
        return ProvisionalCaptureFragmentReceiptV1.from_packed(packed, physical)


class _BlockingSink(_Sink):
    def __init__(self) -> None:
        super().__init__()
        self.all_entered = threading.Event()
        self.release = threading.Event()
        self.active = 0
        self.maximum_active = 0

    def create_provisional(
        self,
        packed: PackedCaptureFragmentV1,
    ) -> ProvisionalCaptureFragmentReceiptV1:
        with self._lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            if self.active == 4:
                self.all_entered.set()
        if not self.release.wait(10):
            raise RuntimeError("test sink release timed out")
        try:
            return super().create_provisional(packed)
        finally:
            with self._lock:
                self.active -= 1


class _FailingSink(_Sink):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def create_provisional(
        self,
        packed: PackedCaptureFragmentV1,
    ) -> ProvisionalCaptureFragmentReceiptV1:
        with self._lock:
            if not self.failed:
                self.failed = True
                fail = True
            else:
                fail = False
        if fail:
            self.entered.set()
            if not self.release.wait(10):
                raise RuntimeError("test failure release timed out")
            raise RuntimeError("injected provisional create failure")
        return super().create_provisional(packed)


class _MismatchedSink(_Sink):
    def create_provisional(
        self,
        packed: PackedCaptureFragmentV1,
    ) -> ProvisionalCaptureFragmentReceiptV1:
        receipt = super().create_provisional(packed)
        return replace(
            receipt,
            fragment_plan_sha256="f" * 64,
        )


def test_host_step_canonicalizes_device_order_without_host_identity(
    addressable_shard: AddressableCaptureShardV1,
) -> None:
    step = _host_step(addressable_shard)
    assert tuple(item.global_batch_start for item in step.addressable_slices) == (
        0,
        2,
        4,
        6,
    )
    assert len(step.fragment_plans) == 4
    with pytest.raises(CaptureHostWriterError, match="four"):
        build_host_capture_step(
            work_plan=step.work_plan,
            step_index=0,
            addressable_slices=step.addressable_slices[:3],
        )
    with pytest.raises(CaptureHostWriterError, match="four"):
        build_host_capture_step(
            work_plan=step.work_plan,
            step_index=0,
            addressable_slices=(
                step.addressable_slices[0],
                step.addressable_slices[1],
                step.addressable_slices[2],
                object(),  # type: ignore[arg-type]
            ),
        )


def test_writer_uses_four_streams_and_returns_only_provisional_receipts(
    addressable_shard: AddressableCaptureShardV1,
) -> None:
    sink = _BlockingSink()
    writer = AsyncCaptureHostWriterV1(sink)
    try:
        ticket = writer.submit_nowait(_host_step(addressable_shard))
        with pytest.raises(CaptureHostWriterError, match="finite"):
            ticket.result(timeout=math.nan)
        assert sink.all_entered.wait(10)
        assert sink.maximum_active == 4
        assert not ticket.done
        sink.release.set()
        result = ticket.result(timeout=10)
        assert len(result.receipts) == 4
        assert result.payload_bytes == 4 * 29_446_144
        assert result.full_payload_readback_verified is False
        assert result.manifest_visible is False
        assert result.production_sealable is False
        assert all(
            receipt.object_create_acknowledged
            and not receipt.full_payload_readback_verified
            and not receipt.manifest_visible
            and not receipt.production_sealable
            for receipt in result.receipts
        )
    finally:
        sink.release.set()
        writer.close()


def test_third_step_fails_immediately_when_active_and_waiting_slots_are_full(
    addressable_shard: AddressableCaptureShardV1,
) -> None:
    sink = _BlockingSink()
    writer = AsyncCaptureHostWriterV1(sink)
    step = _host_step(addressable_shard)
    try:
        first = writer.submit_nowait(step)
        second = writer.submit_nowait(step)
        started = time.monotonic()
        with pytest.raises(CaptureHostWriterBackpressure, match="both"):
            writer.submit_nowait(step)
        assert time.monotonic() - started < 0.1
        assert sink.all_entered.wait(10)
        snapshot = writer.snapshot()
        assert snapshot.queued_complete_steps == 1
        assert snapshot.maximum_observed_queue_depth == 1
        sink.release.set()
        assert len(first.result(timeout=10).receipts) == 4
        assert len(second.result(timeout=10).receipts) == 4
    finally:
        sink.release.set()
        writer.close()
    snapshot = writer.snapshot()
    assert snapshot.submitted_steps == 2
    assert snapshot.completed_steps == 2
    assert snapshot.failed_steps == 0
    assert snapshot.closed is True


def test_failure_latches_writer_and_rejects_later_scientific_work(
    addressable_shard: AddressableCaptureShardV1,
) -> None:
    sink = _FailingSink()
    writer = AsyncCaptureHostWriterV1(sink)
    step = _host_step(addressable_shard)
    try:
        first = writer.submit_nowait(step)
        assert sink.entered.wait(10)
        second = writer.submit_nowait(step)
        sink.release.set()
        with pytest.raises(CaptureHostWriterFatal) as raised:
            first.result(timeout=10)
        assert isinstance(raised.value.__cause__, CaptureHostWriterFatal)
        with pytest.raises(CaptureHostWriterFatal):
            second.result(timeout=10)
        with pytest.raises(CaptureHostWriterFatal, match="latched"):
            writer.submit_nowait(step)
        snapshot = writer.snapshot()
        assert snapshot.fatal is True
        assert snapshot.failed_steps == 2
        assert snapshot.completed_steps == 0
    finally:
        sink.release.set()
        writer.close()


def test_receipt_mismatch_is_a_fatal_writer_failure(
    addressable_shard: AddressableCaptureShardV1,
) -> None:
    writer = AsyncCaptureHostWriterV1(_MismatchedSink())
    try:
        ticket = writer.submit_nowait(_host_step(addressable_shard))
        with pytest.raises(CaptureHostWriterFatal):
            ticket.result(timeout=10)
        assert writer.snapshot().fatal is True
    finally:
        writer.close()


def test_padding_only_slices_create_no_objects(
    addressable_shard: AddressableCaptureShardV1,
) -> None:
    sink = _Sink()
    writer = AsyncCaptureHostWriterV1(sink)
    try:
        step = _host_step(
            addressable_shard,
            token_count=1,
            starts=(2, 4, 6, 8),
        )
        assert step.fragment_plans == ()
        result = writer.submit_nowait(step).result(timeout=10)
        assert result.receipts == ()
        assert result.payload_bytes == 0
        assert sink.calls == []
    finally:
        writer.close()


def test_public_result_ticket_and_receipt_claims_fail_closed(
    addressable_shard: AddressableCaptureShardV1,
) -> None:
    step = _host_step(addressable_shard)
    with pytest.raises(CaptureHostWriterError, match="implement"):
        AsyncCaptureHostWriterV1(object())
    receipt = _Sink().create_provisional(
        # The codec result is intentionally created only inside the writer in
        # production; this direct test exercises immutable receipt flags.
        pack_capture_fragment(step.fragment_plans[0], addressable_shard)
    )
    with pytest.raises(CaptureHostWriterError, match="exact false"):
        replace(receipt, full_payload_readback_verified=True)
    with pytest.raises(CaptureHostWriterError, match="exact false"):
        replace(receipt, manifest_visible=True)
    with pytest.raises(CaptureHostWriterError, match="exact false"):
        replace(receipt, production_sealable=True)
