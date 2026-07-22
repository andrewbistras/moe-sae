from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import threading
import time
from unittest import mock

import pytest

import moe_sae.sae_operator as sae_operator_module
from moe_sae.sae_operator import (
    CONTROL_POLL_SECONDS,
    EXPECTED_LANES,
    HEARTBEAT_INTERVAL_SECONDS,
    HEARTBEAT_OBSERVATION_POLLS,
    LANE_FRESH_SECONDS,
    LANE_STATUS_PUBLICATION_GRACE_SECONDS,
    MISSED_HEARTBEATS_BEFORE_STALE,
    RENDEZVOUS_TIMEOUT_SECONDS,
    SAEAbortRequested,
    SAEControl,
    SAELaneLeaseHeld,
    SAEOperationMismatch,
    SAEOperationSpec,
    SAEOperatorError,
    SAEOrphanedLane,
    SAEPeerFailure,
    SAEPhaseAlreadyPassed,
    SAE_PHASES,
    SAETamperError,
    canonical_sha256,
)


@pytest.fixture(autouse=True)
def _fast_local_startup_lock_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local tmp_path locks need no NFSv3 recovery backoff."""

    monkeypatch.setattr(
        sae_operator_module, "LANE_STATUS_PUBLICATION_POLL_SECONDS", 0.001
    )


def digest(character: str) -> str:
    return character * 64


def operation_spec(**changes: object) -> SAEOperationSpec:
    values: dict[str, object] = {
        "config_sha256": digest("1"),
        "repository_content_sha256": digest("2"),
        "consumer_environment_sha256": digest("3"),
        "train_capture_manifest_sha256": digest("4"),
        "eval_capture_manifest_sha256": digest("5"),
        "activation_storage_attestation_sha256": digest("6"),
        "sae_storage_attestation_sha256": digest("7"),
        "capture_compatibility_sha256": digest("8"),
        "smoke_campaign_sha256": digest("e"),
        "smoke_suite_sha256": digest("f"),
        "checkpoint_payload_root": "gs://test-bucket/sae-checkpoints",
        "canary_waiver_sha256": digest("9"),
        "sealed_recovery_waiver_sha256": digest("c"),
        "performance_waiver_sha256": digest("a"),
    }
    values.update(changes)
    return SAEOperationSpec(**values)  # type: ignore[arg-type]


def control_for(tmp_path: Path, spec: SAEOperationSpec | None = None) -> SAEControl:
    selected = operation_spec() if spec is None else spec
    control = SAEControl(tmp_path)
    control.begin(selected)
    return control


def acquire_lanes(stack: ExitStack, control: SAEControl):
    return [
        stack.enter_context(control.lane_lease(f"controller-{index}"))
        for index in range(EXPECTED_LANES)
    ]


def begin_on_all(control: SAEControl, leases, phase: str):
    with ThreadPoolExecutor(max_workers=EXPECTED_LANES) as executor:
        attempts = list(
            executor.map(
                lambda lease: control.begin_phase(
                    lease,
                    phase,
                    timeout_seconds=3.0,
                    poll_seconds=0.001,
                ),
                leases,
            )
        )
    assert len({attempt.attempt_id for attempt in attempts}) == 1
    assert len({attempt.cohort_sha256 for attempt in attempts}) == 1
    return attempts[0]


def publish_all(
    control: SAEControl,
    leases,
    attempt,
    *,
    outcomes: tuple[str, ...] | None = None,
    results: tuple[str | None, ...] | None = None,
):
    chosen_outcomes = outcomes or ("passed",) * EXPECTED_LANES
    chosen_results = results or (digest("b"),) * EXPECTED_LANES
    for lease, outcome, result in zip(
        leases, chosen_outcomes, chosen_results, strict=True
    ):
        control.publish_attempt_result(
            lease,
            attempt,
            outcome=outcome,  # type: ignore[arg-type]
            result_sha256=result,
        )


def test_operation_spec_hash_binds_the_exact_fixed_plan(tmp_path: Path) -> None:
    spec = operation_spec()
    assert len(SAE_PHASES) == 26
    assert SAE_PHASES[:7] == (
        "validate-prerequisites",
        "smoke-F",
        "smoke-A",
        "smoke-B",
        "gate-A-seed-1",
        "gate-B-seed-1",
        "gate-F-seed-1",
    )
    assert SAE_PHASES[-1] == "report"
    assert spec.sha256 == canonical_sha256(spec.to_mapping())
    assert spec.to_mapping()["sealed_recovery_waiver_sha256"] == digest("c")
    assert spec.to_mapping()["smoke_campaign_sha256"] == digest("e")
    assert spec.to_mapping()["smoke_suite_sha256"] == digest("f")

    control = control_for(tmp_path, spec)
    state = control.load_state()
    assert state["operation_sha256"] == spec.sha256
    assert state["phase_plan"] == list(SAE_PHASES)
    assert control.next_phase() == SAE_PHASES[0]

    changed = operation_spec(checkpoint_payload_root="gs://elsewhere/checkpoints")
    with pytest.raises(SAEOperationMismatch):
        control.begin(changed)

    assert operation_spec(
        sealed_recovery_waiver_sha256=digest("d")
    ).sha256 != spec.sha256
    assert operation_spec(smoke_campaign_sha256=digest("d")).sha256 != spec.sha256
    assert operation_spec(smoke_suite_sha256=digest("d")).sha256 != spec.sha256


def test_nfs_control_timing_budget_is_derived_and_bounded() -> None:
    assert CONTROL_POLL_SECONDS >= 5.0
    assert HEARTBEAT_INTERVAL_SECONDS >= 4.0 * CONTROL_POLL_SECONDS
    assert LANE_FRESH_SECONDS == (
        MISSED_HEARTBEATS_BEFORE_STALE * HEARTBEAT_INTERVAL_SECONDS
        + HEARTBEAT_OBSERVATION_POLLS * CONTROL_POLL_SECONDS
    )
    assert LANE_STATUS_PUBLICATION_GRACE_SECONDS >= (
        2.0 * HEARTBEAT_INTERVAL_SECONDS
    )
    assert RENDEZVOUS_TIMEOUT_SECONDS >= 4.0 * LANE_FRESH_SECONDS


def test_phase_rendezvous_uses_requested_poll_and_times_out_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = control_for(tmp_path)
    monotonic = 0.0
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return monotonic

    def fake_sleep(seconds: float) -> None:
        nonlocal monotonic
        sleeps.append(seconds)
        monotonic += seconds

    monkeypatch.setattr(sae_operator_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(sae_operator_module.time, "sleep", fake_sleep)

    with control.lane_lease("controller-0") as lease:
        with pytest.raises(SAEOperatorError, match="rendezvous timed out"):
            control.begin_phase(
                lease,
                SAE_PHASES[0],
                timeout_seconds=1.0,
                poll_seconds=CONTROL_POLL_SECONDS,
            )

    assert sleeps == [CONTROL_POLL_SECONDS]


@pytest.mark.parametrize("poll_seconds", (0.0, -1.0, float("inf"), float("nan")))
def test_phase_rendezvous_rejects_invalid_polling(
    tmp_path: Path, poll_seconds: float
) -> None:
    control = control_for(tmp_path)
    with control.lane_lease("controller-0") as lease:
        with pytest.raises(ValueError, match="poll_seconds"):
            control.begin_phase(
                lease,
                SAE_PHASES[0],
                poll_seconds=poll_seconds,
            )


def test_operation_spec_rejects_invalid_sealed_recovery_waiver() -> None:
    with pytest.raises(ValueError, match="sealed_recovery_waiver_sha256"):
        operation_spec(sealed_recovery_waiver_sha256="not-a-sha256")


@pytest.mark.parametrize("field", ["smoke_campaign_sha256", "smoke_suite_sha256"])
def test_operation_spec_requires_valid_smoke_handoff_hashes(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        operation_spec(**{field: "not-a-sha256"})


def test_four_lane_attempt_passes_and_skip_requires_exact_result(tmp_path: Path) -> None:
    control = control_for(tmp_path)
    with ExitStack() as stack:
        leases = acquire_lanes(stack, control)
        attempt = begin_on_all(control, leases, SAE_PHASES[0])
        assert attempt.attempt_number == 1
        assert control.status()["campaign_status"] == "running"

        publish_all(control, leases, attempt)
        outcome = control.finalize_attempt(attempt, timeout_seconds=1.0)
        assert outcome.outcome == "passed"
        assert outcome.result_sha256 == digest("b")

        record = control.validate_passed_phase(SAE_PHASES[0], digest("b"))
        assert record["status"] == "passed"
        with pytest.raises(SAEPhaseAlreadyPassed):
            control.validate_passed_phase(SAE_PHASES[0], digest("c"))
        assert control.next_phase() == SAE_PHASES[1]

    assert control.status()["campaign_status"] == "partial"


def test_failed_attempt_restarts_with_new_cohort_and_preserves_history(
    tmp_path: Path,
) -> None:
    control = control_for(tmp_path)
    phase = SAE_PHASES[0]
    with ExitStack() as stack:
        leases = acquire_lanes(stack, control)
        first = begin_on_all(control, leases, phase)
        publish_all(
            control,
            leases,
            first,
            outcomes=("failed", "aborted", "aborted", "aborted"),
            results=(None, None, None, None),
        )
        result = control.finalize_attempt(first, timeout_seconds=1.0)
        assert result.outcome == "failed"
        assert control.load_state()["phases"][phase]["status"] == "failed"

    with ExitStack() as stack:
        leases = acquire_lanes(stack, control)
        second = begin_on_all(control, leases, phase)
        assert second.attempt_number == 2
        assert second.attempt_id != first.attempt_id
        assert second.cohort_sha256 != first.cohort_sha256
        running = control.load_state()["phases"][phase]
        assert running["history"] == [
            {
                "status": "failed",
                "attempt_id": first.attempt_id,
                "cohort_sha256": first.cohort_sha256,
                "members": list(first.members),
                "started_at": running["history"][0]["started_at"],
                "completed_at": running["history"][0]["completed_at"],
                "result": running["history"][0]["result"],
            }
        ]
        publish_all(control, leases, second)
        assert control.finalize_attempt(second).outcome == "passed"


def test_lease_exception_preserves_a_committed_aborted_lane_verdict(
    tmp_path: Path,
) -> None:
    control = control_for(tmp_path)
    phase = SAE_PHASES[0]

    with pytest.raises(RuntimeError, match="supervisor returns collective abort"):
        with ExitStack() as stack:
            leases = acquire_lanes(stack, control)
            attempt = begin_on_all(control, leases, phase)
            publish_all(
                control,
                leases,
                attempt,
                outcomes=("aborted",) * EXPECTED_LANES,
                results=(None,) * EXPECTED_LANES,
            )
            assert control.finalize_attempt(attempt).outcome == "aborted"
            raise RuntimeError("supervisor returns collective abort")

    record = control.load_state()["phases"][phase]
    assert record["status"] == "aborted"
    lanes = control.status()["lanes"]
    assert len(lanes) == EXPECTED_LANES
    assert {lane["state"] for lane in lanes} == {"aborted"}
    assert {lane["attempt_id"] for lane in lanes} == {attempt.attempt_id}
    assert control.status()["vanished_lanes"] == []
    assert control.status()["campaign_status"] == "aborted"

    # A released terminal record is durable history, not a fresh vanished
    # supervisor.  A new exact four-lane cohort can retry immediately without
    # waiting for the freshness window to expire.
    with ExitStack() as stack:
        restarted = acquire_lanes(stack, control)
        retry = begin_on_all(control, restarted, phase)
        assert retry.attempt_number == 2
        publish_all(control, restarted, retry)
        assert control.finalize_attempt(
            retry, timeout_seconds=1.0, poll_seconds=0.001
        ).outcome == "passed"


def test_interrupted_running_attempt_restarts_only_after_old_lanes_release(
    tmp_path: Path,
) -> None:
    control = control_for(tmp_path)
    phase = SAE_PHASES[0]
    with ExitStack() as stack:
        leases = acquire_lanes(stack, control)
        first = begin_on_all(control, leases, phase)
        assert control.load_state()["phases"][phase]["status"] == "running"

    # The lane contexts record their premature exits, but only a newly formed
    # exact four-member cohort may supersede the durable running attempt.
    with ExitStack() as stack:
        leases = acquire_lanes(stack, control)
        second = begin_on_all(control, leases, phase)
        assert second.attempt_number == 2
        assert second.attempt_id != first.attempt_id
        record = control.load_state()["phases"][phase]
        assert record["history"][0]["status"] == "interrupted"
        assert record["history"][0]["attempt_id"] == first.attempt_id
        publish_all(control, leases, second)
        assert control.finalize_attempt(second).outcome == "passed"


def test_fixed_plan_rejects_out_of_order_phase(tmp_path: Path) -> None:
    control = control_for(tmp_path)
    with ExitStack() as stack:
        leases = acquire_lanes(stack, control)
        with ThreadPoolExecutor(max_workers=EXPECTED_LANES) as executor:
            futures = [
                executor.submit(
                    control.begin_phase,
                    lease,
                    SAE_PHASES[1],
                    timeout_seconds=1.0,
                )
                for lease in leases
            ]
            for future in futures:
                with pytest.raises(SAEOperatorError):
                    future.result()
        assert control.next_phase() == SAE_PHASES[0]
        assert all(
            record["status"] == "pending"
            for record in control.load_state()["phases"].values()
        )


def test_inconsistent_success_digests_fail_the_phase_closed(tmp_path: Path) -> None:
    control = control_for(tmp_path)
    with ExitStack() as stack:
        leases = acquire_lanes(stack, control)
        attempt = begin_on_all(control, leases, SAE_PHASES[0])
        publish_all(
            control,
            leases,
            attempt,
            results=(digest("b"), digest("b"), digest("b"), digest("c")),
        )
        with pytest.raises(SAEPeerFailure, match="inconsistent results"):
            control.finalize_attempt(attempt, timeout_seconds=1.0)
        state = control.load_state()
        record = state["phases"][SAE_PHASES[0]]
        assert record["status"] == "failed"
        assert record["result"]["result_sha256"] is None


def test_split_phase_cohort_is_rejected_before_attempt_creation(tmp_path: Path) -> None:
    control = control_for(tmp_path)
    with ExitStack() as stack:
        leases = acquire_lanes(stack, control)
        for lease in leases:
            control.heartbeat(
                lease,
                "ready-to-begin",
                phase=SAE_PHASES[0],
                force=True,
            )
        control.heartbeat(
            leases[-1],
            "ready-to-begin",
            phase=SAE_PHASES[1],
            force=True,
        )
        with pytest.raises(SAEOperatorError, match="split across phase cohorts"):
            control.begin_phase(leases[0], SAE_PHASES[0], timeout_seconds=0.5)
        assert control.load_state()["phases"][SAE_PHASES[0]]["status"] == "pending"


def test_boundary_abort_is_latched_idempotent_and_never_clears_live_lanes(
    tmp_path: Path,
) -> None:
    control = control_for(tmp_path)
    request = control.request_abort("stop after this child", requested_by="test")
    assert control.request_abort("different text", requested_by="test") == request
    assert control.status()["campaign_status"] == "abort-requested"

    with ExitStack() as stack:
        lease = stack.enter_context(control.lane_lease("controller-0"))
        with pytest.raises(SAEAbortRequested, match="stop after this child"):
            control.begin_phase(lease, SAE_PHASES[0], timeout_seconds=0.1)
        with pytest.raises(SAELaneLeaseHeld):
            control.clear_abort()

    acknowledged = control.acknowledge_abort(SAE_PHASES[0])
    assert acknowledged["last_abort_sha256"] == request["abort_sha256"]
    assert control.clear_abort()
    assert not control.clear_abort()
    assert control.pending_abort() is None


def test_abort_cannot_be_acknowledged_during_a_running_phase(tmp_path: Path) -> None:
    control = control_for(tmp_path)
    with ExitStack() as stack:
        leases = acquire_lanes(stack, control)
        begin_on_all(control, leases, SAE_PHASES[0])
        control.request_abort("stop at boundary")
        with pytest.raises(SAEOperatorError, match="phase is running"):
            control.acknowledge_abort(SAE_PHASES[0])


def test_tampered_state_and_lane_status_fail_closed(tmp_path: Path) -> None:
    control = control_for(tmp_path)
    state = json.loads(control.state_path.read_text(encoding="utf-8"))
    state["phase_plan"][-1] = "not-report"
    control.state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(SAETamperError):
        control.load_state()

    other = control_for(tmp_path / "lane")
    with pytest.raises(SAETamperError):
        with other.lane_lease("controller-0") as lease:
            lane = json.loads(lease.status_path.read_text(encoding="utf-8"))
            lane["hostname"] = "tampered"
            lease.status_path.write_text(json.dumps(lane), encoding="utf-8")
            other.status()


def test_stale_kernel_held_lane_is_reported_as_orphan_and_blocks_progress(
    tmp_path: Path,
) -> None:
    control = control_for(tmp_path)
    with control.lane_lease("controller-0"):
        future = time.time() + LANE_FRESH_SECONDS + 1.0
        with mock.patch("moe_sae.sae_operator.time.time", return_value=future):
            status = control.status()
            assert status["campaign_status"] == "orphaned"
            assert len(status["orphaned_lanes"]) == 1
            with pytest.raises(SAEOrphanedLane):
                control.assert_no_orphaned_lanes()


def test_lane_observer_waits_for_initial_status_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force lock acquisition to race the first atomic heartbeat."""

    control = control_for(tmp_path)
    heartbeat_entered = threading.Event()
    allow_heartbeat = threading.Event()
    body_entered = threading.Event()
    release_lane = threading.Event()
    observer_entered = threading.Event()
    original_heartbeat = control.heartbeat
    original_observe_lanes = control._observe_lanes
    delayed = False

    def delayed_heartbeat(lease, state, **kwargs):
        nonlocal delayed
        if state == "starting" and not delayed:
            delayed = True
            heartbeat_entered.set()
            assert allow_heartbeat.wait(2.0)
        return original_heartbeat(lease, state, **kwargs)

    def observed_lanes():
        observer_entered.set()
        return original_observe_lanes()

    monkeypatch.setattr(control, "heartbeat", delayed_heartbeat)
    monkeypatch.setattr(control, "_observe_lanes", observed_lanes)

    def own_lane() -> None:
        with control.lane_lease("controller-0"):
            body_entered.set()
            assert release_lane.wait(2.0)

    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(own_lane)
        assert heartbeat_entered.wait(2.0)
        observer = executor.submit(control.status)
        assert observer_entered.wait(2.0)
        with pytest.raises(FutureTimeoutError):
            observer.result(timeout=0.05)
        allow_heartbeat.set()
        assert body_entered.wait(2.0)
        status = observer.result(timeout=2.0)
        assert status["orphaned_lanes"] == []
        assert len(status["lanes"]) == 1
        assert status["lanes"][0]["state"] == "starting"
        assert status["lanes"][0]["lease_held"] is True
        release_lane.set()
        owner.result(timeout=2.0)


def test_lane_observer_never_pairs_reacquired_lock_with_previous_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = control_for(tmp_path)
    with control.lane_lease("controller-0") as previous:
        previous_lease_id = previous.lease_id
    old_status = control._read_lane_if_valid(previous.status_path)
    assert old_status is not None and old_status["state"] == "exited"

    heartbeat_entered = threading.Event()
    allow_heartbeat = threading.Event()
    release_lane = threading.Event()
    original_heartbeat = control.heartbeat
    delayed = False

    def delayed_heartbeat(lease, state, **kwargs):
        nonlocal delayed
        if state == "starting" and not delayed:
            delayed = True
            heartbeat_entered.set()
            assert allow_heartbeat.wait(2.0)
        return original_heartbeat(lease, state, **kwargs)

    monkeypatch.setattr(control, "heartbeat", delayed_heartbeat)

    def reacquire_lane() -> str:
        with control.lane_lease("controller-0") as current:
            assert release_lane.wait(2.0)
            return current.lease_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(reacquire_lane)
        assert heartbeat_entered.wait(2.0)
        observer = executor.submit(control.status)
        with pytest.raises(FutureTimeoutError):
            observer.result(timeout=0.05)
        allow_heartbeat.set()
        status = observer.result(timeout=2.0)
        assert len(status["lanes"]) == 1
        observed = status["lanes"][0]
        assert observed["lease_id"] != previous_lease_id
        assert observed["state"] == "starting"
        assert observed["lease_held"] is True
        release_lane.set()
        assert owner.result(timeout=2.0) == observed["lease_id"]


def test_blocked_running_heartbeat_cannot_regress_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = control_for(tmp_path)
    running_write_entered = threading.Event()
    allow_running_write = threading.Event()
    original_atomic_json = sae_operator_module._atomic_json

    def blocked_atomic_json(path: Path, payload) -> None:
        if payload.get("state") == "running" and not running_write_entered.is_set():
            running_write_entered.set()
            assert allow_running_write.wait(2.0)
        original_atomic_json(path, payload)

    monkeypatch.setattr(sae_operator_module, "_atomic_json", blocked_atomic_json)

    with control.lane_lease("controller-0") as lease:
        with ThreadPoolExecutor(max_workers=2) as executor:
            running = executor.submit(
                control.heartbeat,
                lease,
                "running",
                phase=SAE_PHASES[0],
                attempt_id="a" * 32,
                force=True,
            )
            assert running_write_entered.wait(2.0)
            terminal = executor.submit(
                control.heartbeat,
                lease,
                "succeeded",
                phase=SAE_PHASES[0],
                attempt_id="a" * 32,
                result_sha256=digest("b"),
                force=True,
            )
            with pytest.raises(FutureTimeoutError):
                terminal.result(timeout=0.05)
            allow_running_write.set()
            running.result(timeout=2.0)
            terminal.result(timeout=2.0)

        observed = control._read_lane_if_valid(lease.status_path)
        assert observed is not None
        assert observed["state"] == "succeeded"
        assert observed["result_sha256"] == digest("b")


def test_held_lane_without_status_fails_after_bounded_publication_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = control_for(tmp_path)
    lane_id = sae_operator_module.hashlib.sha256(b"controller-0").hexdigest()[:24]
    lock_path = control.root / "lanes" / f"lane-{lane_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        sae_operator_module, "LANE_STATUS_PUBLICATION_GRACE_SECONDS", 0.0
    )
    with lock_path.open("a+b", buffering=0) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        with pytest.raises(SAEOrphanedLane, match="did not publish"):
            control.status()


def test_duplicate_lane_lease_is_rejected_until_inherited_lock_closes(
    tmp_path: Path,
) -> None:
    control = control_for(tmp_path)
    with control.lane_lease("controller-0"):
        with pytest.raises(SAELaneLeaseHeld):
            with control.lane_lease("controller-0"):
                pass
    with control.lane_lease("controller-0"):
        pass


def test_fresh_active_status_without_kernel_lease_fails_closed(tmp_path: Path) -> None:
    control = control_for(tmp_path)
    with control.lane_lease("controller-0") as lease:
        control.heartbeat(
            lease,
            "ready-to-begin",
            phase=SAE_PHASES[0],
            force=True,
        )
    # The normal context exit replaces an active record with `failed`; recreate
    # the crash shape by changing only the state and recomputing its self-hash.
    path = next((control.root / "lanes").glob("lane-*.json"))
    lane = json.loads(path.read_text(encoding="utf-8"))
    lane.update(state="ready-to-begin", phase=SAE_PHASES[0], attempt_id=None)
    lane["lane_status_sha256"] = canonical_sha256(
        {key: value for key, value in lane.items() if key != "lane_status_sha256"}
    )
    path.write_text(json.dumps(lane), encoding="utf-8")
    status = control.status()
    assert status["campaign_status"] == "inconsistent-lanes"
    assert status["vanished_lanes"] == [lane["lane_id"]]
    with pytest.raises(SAEPeerFailure):
        control.assert_no_orphaned_lanes()
