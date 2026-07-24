from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace

import pytest

from moe_sae.r1.legacy_slice_runtime import (
    LEGACY_PROCESS_COUNT,
    LegacySliceBindingV1,
    LegacySliceCommandV1,
    LegacySliceRuntimeError,
    LegacyV6e64WholeSliceRuntimeV1,
)
from moe_sae.r1.preemption_v2 import AttemptAuthorityV2
from moe_sae.r1.protocol import ProtocolError
from moe_sae.r1.runtime_supervisor import (
    RuntimeSupervisorV1,
    SliceMemberLost,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _binding() -> LegacySliceBindingV1:
    return LegacySliceBindingV1(
        launch_token_sha256=_sha("token"),
        launch_admission_sha256=_sha("admission"),
        deployment_envelope_sha256=_sha("deployment"),
        implementation_identity_sha256=_sha("implementation"),
        attempt_id="r1-attempt-0001",
        expected_fence=1,
        maximum_step_permits=64,
        process_count=16,
        device_count=64,
    )


class MemoryCoordination:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.values: dict[str, bytes] = {}
        self.live = {index: 10_000 + index for index in range(16)}
        self.barriers: dict[str, set[int]] = {}


class MemoryClient:
    production_qualified = True

    def __init__(self, backend: MemoryCoordination, process_index: int) -> None:
        self.backend = backend
        self._process_index = process_index

    @property
    def process_index(self) -> int:
        return self._process_index

    @property
    def process_count(self) -> int:
        return 16

    def put_once(self, key: str, value: bytes) -> None:
        with self.backend.changed:
            if key in self.backend.values:
                raise RuntimeError("write-once conflict")
            self.backend.values[key] = bytes(value)
            self.backend.changed.notify_all()

    def wait_get(self, key: str, *, timeout_seconds: float) -> bytes:
        deadline = time.monotonic() + timeout_seconds
        with self.backend.changed:
            while key not in self.backend.values:
                if set(self.backend.live) != set(range(16)):
                    raise RuntimeError("command member was lost")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("command wait timed out")
                self.backend.changed.wait(remaining)
            return self.backend.values[key]

    def barrier(
        self,
        barrier_id: str,
        *,
        timeout_seconds: float,
        process_ids: Sequence[int],
    ) -> None:
        expected = set(process_ids)
        deadline = time.monotonic() + timeout_seconds
        with self.backend.changed:
            arrived = self.backend.barriers.setdefault(barrier_id, set())
            if self.process_index in arrived:
                raise RuntimeError("process entered a barrier twice")
            arrived.add(self.process_index)
            self.backend.changed.notify_all()
            while arrived != expected:
                if not expected.issubset(self.backend.live):
                    raise RuntimeError("barrier member was lost")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("barrier timed out")
                self.backend.changed.wait(remaining)

    def live_nodes(self, process_ids: Sequence[int]) -> Mapping[int, int]:
        with self.backend.lock:
            return {
                index: self.backend.live[index]
                for index in process_ids
                if index in self.backend.live
            }


def _runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    MemoryCoordination,
    list[LegacyV6e64WholeSliceRuntimeV1],
    list[str],
]:
    # These tests exercise the distributed execution boundary independently
    # from the already-covered launch/admission record constructors.
    monkeypatch.setattr(
        LegacySliceBindingV1,
        "verify_admission",
        lambda _self, _admission: None,
    )
    monkeypatch.setattr(
        LegacySliceBindingV1,
        "verify_authority",
        lambda _self, _authority: None,
    )
    backend = MemoryCoordination()
    inventory = _sha("exact-64-device-inventory")
    runtimes = [
        LegacyV6e64WholeSliceRuntimeV1(
            binding=_binding(),
            coordination=MemoryClient(backend, process_index),
            inventory_sha256=lambda _binding: inventory,
            poll_interval_seconds=0.01,
            command_timeout_seconds=2.0,
            barrier_timeout_seconds=2.0,
        )
        for process_index in range(16)
    ]
    stop_calls: list[str] = []
    runtimes[0].bind_stop_request(stop_calls.append)
    for runtime in runtimes:
        assert (
            runtime.initialize_device_inventory_sha256(
                object()  # type: ignore[arg-type]
            )
            == inventory
        )
    return backend, runtimes, stop_calls


def _start_followers(
    backend: MemoryCoordination,
    runtimes: list[LegacyV6e64WholeSliceRuntimeV1],
    callbacks: dict[int, list[int]],
) -> tuple[list[threading.Thread], dict[int, str], dict[int, BaseException]]:
    outcomes: dict[int, str] = {}
    errors: dict[int, BaseException] = {}

    def serve(index: int) -> None:
        try:
            outcomes[index] = runtimes[index].serve_follower(
                lambda permit: callbacks[index].append(permit)
            )
        except SliceMemberLost as error:
            errors[index] = error
            with backend.changed:
                backend.live.pop(index, None)
                backend.changed.notify_all()

    threads = [
        threading.Thread(target=serve, args=(index,), daemon=True)
        for index in range(1, LEGACY_PROCESS_COUNT)
    ]
    for thread in threads:
        thread.start()
    return threads, outcomes, errors


def test_binding_and_wire_records_reject_changed_slice_identity() -> None:
    binding = _binding()
    assert LegacySliceBindingV1.from_json(binding.canonical_bytes()) == binding
    command = LegacySliceCommandV1(
        slice_binding_sha256=binding.digest,
        sequence=1,
        kind="STEP",
        permit=1,
        outcome=None,
    )
    assert LegacySliceCommandV1.from_json(command.canonical_bytes()) == command
    with pytest.raises(ProtocolError, match="maximum_step_permits"):
        replace(binding, maximum_step_permits=1_000_001)
    with pytest.raises(ProtocolError, match="permit=sequence"):
        LegacySliceCommandV1(
            slice_binding_sha256=binding.digest,
            sequence=2,
            kind="STEP",
            permit=1,
            outcome=None,
        )
    with pytest.raises(ProtocolError, match="16 processes"):
        LegacySliceBindingV1(
            launch_token_sha256=_sha("token"),
            launch_admission_sha256=_sha("admission"),
            deployment_envelope_sha256=_sha("deployment"),
            implementation_identity_sha256=_sha("implementation"),
            attempt_id="r1-attempt-0001",
            expected_fence=1,
            maximum_step_permits=64,
            process_count=15,
            device_count=64,
        )


def test_binding_rejects_a_stale_attempt_fence_before_permit_publication() -> None:
    binding = _binding()

    def authority(*, attempt: str, fence: int) -> AttemptAuthorityV2:
        value = object.__new__(AttemptAuthorityV2)
        object.__setattr__(
            value,
            "lease",
            SimpleNamespace(attempt_id=attempt, fence=fence),
        )
        object.__setattr__(
            value,
            "deployment_identity",
            SimpleNamespace(
                implementation_identity_sha256=(
                    binding.implementation_identity_sha256
                )
            ),
        )
        object.__setattr__(
            value,
            "deployment",
            SimpleNamespace(digest=binding.deployment_envelope_sha256),
        )
        return value

    binding.verify_authority(
        authority(attempt=binding.attempt_id, fence=binding.expected_fence)
    )
    with pytest.raises(SliceMemberLost, match="authority differs"):
        binding.verify_authority(
            authority(attempt=binding.attempt_id, fence=2)
        )
    with pytest.raises(SliceMemberLost, match="authority differs"):
        binding.verify_authority(
            authority(attempt="r1-attempt-0002", fence=1)
        )


def test_process_zero_fans_out_monotonic_permits_and_waits_for_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, runtimes, stop_calls = _runtimes(monkeypatch)
    callbacks = {index: [] for index in range(1, 16)}
    threads, outcomes, errors = _start_followers(backend, runtimes, callbacks)
    leader_steps: list[int] = []

    first = runtimes[0].run_authorized_step(
        permit=1,
        authority=object(),  # type: ignore[arg-type]
        accelerator_step=lambda: leader_steps.append(1) or "first",
    )
    second = runtimes[0].run_authorized_step(
        permit=2,
        authority=object(),  # type: ignore[arg-type]
        accelerator_step=lambda: leader_steps.append(2) or "second",
    )
    runtimes[0].terminate(outcome="SUCCEEDED", incident_sha256=None)
    for thread in threads:
        thread.join(timeout=2)

    assert first == "first"
    assert second == "second"
    assert leader_steps == [1, 2]
    assert callbacks == {index: [1, 2] for index in range(1, 16)}
    assert outcomes == {index: "SUCCEEDED" for index in range(1, 16)}
    assert errors == {}
    assert stop_calls == []
    assert all(not thread.is_alive() for thread in threads)
    assert len(backend.values) == 3
    assert len(backend.barriers) == 5
    assert all(
        processes == set(range(16))
        for processes in backend.barriers.values()
    )


def test_completed_barrier_waits_for_every_callback_result_pytree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _backend, runtimes, _stop_calls = _runtimes(monkeypatch)
    completed: set[int] = set()
    completed_lock = threading.Lock()

    class DeferredLeaf:
        def __init__(self, process_index: int) -> None:
            self.process_index = process_index

        def block_until_ready(self) -> DeferredLeaf:
            with completed_lock:
                completed.add(self.process_index)
            return self

    outcomes: dict[int, str] = {}

    def serve(index: int) -> None:
        outcomes[index] = runtimes[index].serve_follower(
            lambda _permit: {"nested": [DeferredLeaf(index)]}
        )

    threads = [
        threading.Thread(target=serve, args=(index,), daemon=True)
        for index in range(1, 16)
    ]
    for thread in threads:
        thread.start()

    result = runtimes[0].run_authorized_step(
        permit=1,
        authority=object(),  # type: ignore[arg-type]
        accelerator_step=lambda: {"nested": [DeferredLeaf(0)]},
    )
    assert isinstance(result["nested"][0], DeferredLeaf)
    assert completed == set(range(16))

    runtimes[0].terminate(outcome="SUCCEEDED", incident_sha256=None)
    for thread in threads:
        thread.join(timeout=2)
    assert outcomes == {index: "SUCCEEDED" for index in range(1, 16)}
    assert all(not thread.is_alive() for thread in threads)


def test_leader_callback_failure_releases_all_followers_with_next_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, runtimes, stop_calls = _runtimes(monkeypatch)
    callbacks = {index: [] for index in range(1, 16)}
    threads, outcomes, errors = _start_followers(
        backend,
        runtimes,
        callbacks,
    )

    def fail_leader() -> None:
        raise RuntimeError("leader callback failed")

    with pytest.raises(RuntimeError, match="leader callback failed"):
        runtimes[0].run_authorized_step(
            permit=1,
            authority=object(),  # type: ignore[arg-type]
            accelerator_step=fail_leader,
        )
    runtimes[0].terminate(outcome="FAILED", incident_sha256=None)
    for thread in threads:
        thread.join(timeout=2)

    assert callbacks == {index: [1] for index in range(1, 16)}
    assert outcomes == {index: "FAILED" for index in range(1, 16)}
    assert errors == {}
    assert stop_calls == []
    assert all(not thread.is_alive() for thread in threads)
    assert len(backend.values) == 2
    command_keys = sorted(backend.values)
    assert command_keys[0].endswith("00000000000000000001")
    assert command_keys[1].endswith("00000000000000000002")


def test_member_loss_during_step_closes_gate_and_no_new_permit_is_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, runtimes, stop_calls = _runtimes(monkeypatch)
    callbacks = {index: [] for index in range(1, 16)}
    threads, _outcomes, errors = _start_followers(
        backend,
        runtimes,
        callbacks,
    )

    def lose_member() -> None:
        with backend.changed:
            del backend.live[7]
            backend.changed.notify_all()
        time.sleep(0.03)

    with pytest.raises(SliceMemberLost):
        runtimes[0].run_authorized_step(
            permit=1,
            authority=object(),  # type: ignore[arg-type]
            accelerator_step=lose_member,
        )
    with pytest.raises(SliceMemberLost, match="latched"):
        runtimes[0].run_authorized_step(
            permit=2,
            authority=object(),  # type: ignore[arg-type]
            accelerator_step=lambda: None,
        )
    runtimes[0].terminate(
        outcome="PREEMPTED",
        incident_sha256=_sha("incident"),
    )
    for thread in threads:
        thread.join(timeout=2)

    assert stop_calls == ["SLICE_MEMBER_LOST"]
    assert all(not thread.is_alive() for thread in threads)
    assert errors
    assert all(isinstance(error, SliceMemberLost) for error in errors.values())
    assert not any("/commands/00000000000000000002" in key for key in backend.values)


def test_fake_coordination_member_loss_closes_actual_supervisor_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        LegacySliceBindingV1,
        "verify_admission",
        lambda _self, _admission: None,
    )
    backend = MemoryCoordination()
    runtime = LegacyV6e64WholeSliceRuntimeV1(
        binding=_binding(),
        coordination=MemoryClient(backend, 0),
        inventory_sha256=lambda _binding: _sha("inventory"),
        poll_interval_seconds=0.01,
        command_timeout_seconds=2.0,
        barrier_timeout_seconds=2.0,
    )

    class QualifiedInput:
        production_qualified = True

    supervisor = RuntimeSupervisorV1(
        resolver=object(),  # type: ignore[arg-type]
        control_store=object(),  # type: ignore[arg-type]
        pinned_reader=object(),  # type: ignore[arg-type]
        regional_input_resolver=QualifiedInput(),  # type: ignore[arg-type]
        slice_runtime=runtime,
        workload=object(),  # type: ignore[arg-type]
        content_verifier=lambda _content: True,
    )
    runtime.initialize_device_inventory_sha256(
        object()  # type: ignore[arg-type]
    )
    with backend.changed:
        backend.live.pop(7)
        backend.changed.notify_all()
    deadline = time.monotonic() + 1.0
    while not supervisor.stop_requested and time.monotonic() < deadline:
        time.sleep(0.01)
    runtime.terminate(
        outcome="PREEMPTED",
        incident_sha256=_sha("incident"),
    )

    assert supervisor.stop_requested


def test_follower_callback_failure_is_a_whole_slice_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, runtimes, stop_calls = _runtimes(monkeypatch)
    outcomes: dict[int, str] = {}
    errors: dict[int, BaseException] = {}

    def serve(index: int) -> None:
        try:
            outcomes[index] = runtimes[index].serve_follower(
                lambda _permit: (
                    (_ for _ in ()).throw(RuntimeError("fixed step failed"))
                    if index == 9
                    else None
                )
            )
        except SliceMemberLost as error:
            errors[index] = error
            with backend.changed:
                backend.live.pop(index, None)
                backend.changed.notify_all()

    threads = [
        threading.Thread(target=serve, args=(index,), daemon=True)
        for index in range(1, 16)
    ]
    for thread in threads:
        thread.start()

    with pytest.raises(SliceMemberLost):
        runtimes[0].run_authorized_step(
            permit=1,
            authority=object(),  # type: ignore[arg-type]
            accelerator_step=lambda: None,
        )
    runtimes[0].terminate(
        outcome="PREEMPTED",
        incident_sha256=_sha("incident"),
    )
    for thread in threads:
        thread.join(timeout=2)

    assert stop_calls == ["SLICE_MEMBER_LOST"]
    assert isinstance(errors.get(9), SliceMemberLost)
    assert outcomes == {}
    assert all(not thread.is_alive() for thread in threads)


def test_runtime_rejects_non_exact_coordination_shape() -> None:
    backend = MemoryCoordination()
    client = MemoryClient(backend, 0)
    client._process_index = 16
    with pytest.raises(LegacySliceRuntimeError, match="0..15"):
        LegacyV6e64WholeSliceRuntimeV1(
            binding=_binding(),
            coordination=client,
        )
