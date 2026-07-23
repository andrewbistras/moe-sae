from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading
from typing import Callable

import ml_dtypes
import numpy as np
import pytest

from moe_sae.r1.capture_data_plane_contract import (
    CaptureWorkPlanV1,
    build_capture_work_plan,
)
from moe_sae.r1.capture_fragment_codec import (
    PackedCaptureFragmentV1,
    decode_capture_fragment,
)
from moe_sae.r1.capture_host_writer import (
    AsyncCaptureHostWriterV1,
    ProvisionalCaptureFragmentReceiptV1,
)
from moe_sae.r1.jax_capture_bridge import (
    JAXCaptureBridgeBackpressure,
    JAXCaptureBridgeError,
    JAXCaptureBridgeFatal,
    JAXCaptureBridgeTicketV1,
    JAXCaptureBridgeV1,
)
from moe_sae.r1.prefix_batchsplit_contract import PINNED_CAPTURE_OUTPUTS
from tests.r1.test_capture_host_writer import _Sink


_BF16 = np.dtype(ml_dtypes.bfloat16)


class _RecordingSink(_Sink):
    def __init__(self) -> None:
        super().__init__()
        self.packed: list[PackedCaptureFragmentV1] = []

    def create_provisional(
        self,
        packed: PackedCaptureFragmentV1,
    ) -> ProvisionalCaptureFragmentReceiptV1:
        self.packed.append(packed)
        return super().create_provisional(packed)


@dataclass
class _FakeShard:
    index: tuple[slice, ...]
    data: "_FakeArray"
    replica_id: int = 0


class _FakeArray:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype,
        *,
        shards: list[_FakeShard] | None = None,
        factory: Callable[[], np.ndarray] | None = None,
        entered: threading.Event | None = None,
        release: threading.Event | None = None,
        fail_copy: bool = False,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.addressable_shards = [] if shards is None else shards
        self._factory = factory
        self._entered = entered
        self._release = release
        self._fail_copy = fail_copy
        self.copy_calls = 0
        self.array_calls = 0

    def copy_to_host_async(self) -> None:
        self.copy_calls += 1
        if self._fail_copy:
            raise RuntimeError("injected async-copy failure")

    def __array__(
        self,
        dtype: np.dtype | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        self.array_calls += 1
        if self._entered is not None:
            self._entered.set()
        if self._release is not None and not self._release.wait(10):
            raise RuntimeError("test D2H release timed out")
        if self._factory is None:
            raise RuntimeError("global fake array cannot materialize")
        result = self._factory()
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        if copy is True:
            result = result.copy()
        return result


def _local_factory(name: str) -> Callable[[], np.ndarray]:
    if name == "routed_activation":
        return lambda: np.zeros((2, 1_024, 7_168), dtype=_BF16)
    if name == "global_expert_ids":
        return lambda: np.broadcast_to(
            np.arange(8, dtype=np.int32),
            (2, 1_024, 8),
        ).copy()
    return lambda: np.full(
        (2, 1_024, 8),
        np.float32(2.5 / 8),
        dtype=np.float32,
    )


def _outputs(
    *,
    starts: tuple[int, ...] = (6, 0, 4, 2),
    block_first_activation: tuple[threading.Event, threading.Event] | None = None,
    fail_copy_name: str | None = None,
) -> dict[str, _FakeArray]:
    result = {}
    for schema in PINNED_CAPTURE_OUTPUTS:
        shards = []
        for start in starts:
            entered = None
            release = None
            if (
                schema.name == "routed_activation"
                and start == 0
                and block_first_activation is not None
            ):
                entered, release = block_first_activation
            local = _FakeArray(
                (
                    (2, 1_024, 7_168)
                    if schema.name == "routed_activation"
                    else (2, 1_024, 8)
                ),
                np.dtype(schema.device_dtype),
                factory=_local_factory(schema.name),
                entered=entered,
                release=release,
            )
            shards.append(
                _FakeShard(
                    (
                        slice(start, start + 2),
                        slice(None),
                        slice(None),
                    ),
                    local,
                )
            )
        result[schema.name] = _FakeArray(
            schema.shape,
            np.dtype(schema.device_dtype),
            shards=shards,
            fail_copy=schema.name == fail_copy_name,
        )
    return result


def _metadata() -> tuple[np.ndarray, np.ndarray]:
    sequence_id = np.broadcast_to(
        np.arange(128, dtype=np.int64)[:, None],
        (128, 1_024),
    ).copy()
    position = np.broadcast_to(
        np.arange(1_024, dtype=np.uint16),
        (128, 1_024),
    ).copy()
    return sequence_id, position


def _work(name: str) -> CaptureWorkPlanV1:
    return build_capture_work_plan(
        work_id=name,
        input_range_sha256=hashlib.sha256(name.encode("utf-8")).hexdigest(),
        start_token_ordinal=0,
        token_count=1,
    )


def _submit(
    bridge: JAXCaptureBridgeV1,
    outputs: dict[str, _FakeArray],
    *,
    work_name: str,
) -> JAXCaptureBridgeTicketV1:
    sequence_id, position = _metadata()
    return bridge.submit_nowait(
        work_plan=_work(work_name),
        step_index=0,
        routed_activation=outputs["routed_activation"],
        global_expert_ids=outputs["global_expert_ids"],
        scaled_router_weights=outputs["scaled_router_weights"],
        sequence_id=sequence_id,
        position=position,
    )


def test_bridge_async_handoff_uses_global_slices_and_exact_output_abi() -> None:
    sink = _Sink()
    writer = AsyncCaptureHostWriterV1(sink)
    bridge = JAXCaptureBridgeV1(
        writer,
        array_type=_FakeArray,
        array_adapter_id="unit-fake-v1",
    )
    outputs = _outputs()
    try:
        ticket = _submit(bridge, outputs, work_name="bridge-work-a")
        result = ticket.result(10)
    finally:
        bridge.close()
        writer.close()

    expected_plan = next(_work("bridge-work-a").iter_fragments())
    assert result.step_index == 0
    assert result.fragment_plan_sha256s == (expected_plan.digest,)
    assert len(result.receipts) == 1
    assert all(value.copy_calls == 1 for value in outputs.values())
    assert bridge.production_identity_qualified is False
    assert bridge.bridge_id.startswith("jax-capture-bridge-v1:")
    assert bridge.snapshot().completed_steps == 1


def test_bridge_has_two_end_to_end_slots_and_never_waits_for_a_third() -> None:
    entered = threading.Event()
    release = threading.Event()
    outputs = _outputs(block_first_activation=(entered, release))
    writer = AsyncCaptureHostWriterV1(_Sink())
    bridge = JAXCaptureBridgeV1(
        writer,
        array_type=_FakeArray,
        array_adapter_id="unit-fake-v1",
    )
    try:
        first = _submit(bridge, outputs, work_name="bridge-work-b")
        assert entered.wait(10)
        second = _submit(bridge, outputs, work_name="bridge-work-c")
        invalid_third = _outputs(starts=(0, 2, 4))
        with pytest.raises(JAXCaptureBridgeBackpressure, match="both"):
            _submit(bridge, invalid_third, work_name="bridge-work-d")
        assert all(value.copy_calls == 0 for value in invalid_third.values())
        snapshot = bridge.snapshot()
        assert snapshot.queued_complete_steps == 1
        assert snapshot.maximum_observed_queue_depth == 1
        release.set()
        first.result(10)
        second.result(10)
    finally:
        release.set()
        bridge.close()
        writer.close()


def test_bridge_snapshots_only_owned_host_metadata_before_return() -> None:
    entered = threading.Event()
    release = threading.Event()
    outputs = _outputs(block_first_activation=(entered, release))
    sequence_id, position = _metadata()
    sink = _RecordingSink()
    writer = AsyncCaptureHostWriterV1(sink)
    bridge = JAXCaptureBridgeV1(
        writer,
        array_type=_FakeArray,
        array_adapter_id="unit-fake-v1",
    )
    work = _work("bridge-work-i")
    try:
        ticket = bridge.submit_nowait(
            work_plan=work,
            step_index=0,
            routed_activation=outputs["routed_activation"],
            global_expert_ids=outputs["global_expert_ids"],
            scaled_router_weights=outputs["scaled_router_weights"],
            sequence_id=sequence_id,
            position=position,
        )
        assert entered.wait(10)
        sequence_id[0, 0] = 999
        position[0, 0] = 77
        release.set()
        ticket.result(10)
    finally:
        release.set()
        bridge.close()
        writer.close()

    assert len(sink.packed) == 1
    packed = sink.packed[0]
    decoded = decode_capture_fragment(
        packed.plan,
        packed.payload,
        expected_sha256=packed.sha256,
    )
    assert decoded.fields["sequence_id"][0] == 0
    assert decoded.fields["position"][0] == 0


def test_bridge_rejects_shape_and_slice_drift_before_async_copy() -> None:
    outputs = _outputs(starts=(0, 2, 4))
    writer = AsyncCaptureHostWriterV1(_Sink())
    bridge = JAXCaptureBridgeV1(
        writer,
        array_type=_FakeArray,
        array_adapter_id="unit-fake-v1",
    )
    try:
        with pytest.raises(JAXCaptureBridgeError, match="four"):
            _submit(bridge, outputs, work_name="bridge-work-e")
        assert all(value.copy_calls == 0 for value in outputs.values())
    finally:
        bridge.close()
        writer.close()


def test_async_copy_failure_latches_bridge_and_no_ticket_is_issued() -> None:
    outputs = _outputs(fail_copy_name="global_expert_ids")
    writer = AsyncCaptureHostWriterV1(_Sink())
    bridge = JAXCaptureBridgeV1(
        writer,
        array_type=_FakeArray,
        array_adapter_id="unit-fake-v1",
    )
    try:
        with pytest.raises(JAXCaptureBridgeFatal, match="copy_to_host_async"):
            _submit(bridge, outputs, work_name="bridge-work-f")
        with pytest.raises(JAXCaptureBridgeFatal, match="latched"):
            _submit(bridge, _outputs(), work_name="bridge-work-g")
        assert bridge.snapshot().fatal is True
    finally:
        bridge.close()
        writer.close()


def test_default_adapter_binds_real_jax_array_without_claiming_geometry() -> None:
    import jax.numpy as jnp

    writer = AsyncCaptureHostWriterV1(_Sink())
    bridge = JAXCaptureBridgeV1(writer)
    sequence_id, position = _metadata()
    tiny = jnp.zeros((1,), dtype=jnp.bfloat16)
    try:
        with pytest.raises(JAXCaptureBridgeError, match="global output ABI"):
            bridge.submit_nowait(
                work_plan=_work("bridge-work-h"),
                step_index=0,
                routed_activation=tiny,
                global_expert_ids=tiny,
                scaled_router_weights=tiny,
                sequence_id=sequence_id,
                position=position,
            )
        snapshot = bridge.snapshot()
        assert snapshot.submitted_steps == 0
        assert snapshot.fatal is False
        assert bridge.production_identity_qualified is False
    finally:
        bridge.close()
        writer.close()


def test_bridge_ticket_cannot_be_constructed_or_waited_ambiguously() -> None:
    with pytest.raises(JAXCaptureBridgeError, match="only through"):
        JAXCaptureBridgeTicketV1(0)
