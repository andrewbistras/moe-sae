from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import json
from typing import Any

import pytest

from moe_sae.r1.preemption import (
    AmbiguousMetadata,
    AttemptBindingMismatch,
    AttemptBindingV1,
    CONSERVATIVE_DRAIN_BUDGET_MS,
    CheckpointTimingEvidenceV1,
    IncidentSinkError,
    LeaseExpiredDuringDrain,
    MIN_CHECKPOINT_TIMING_SAMPLES,
    MetadataPreemptionObservationV1,
    NewStepsForbidden,
    PREEMPTED_METADATA_URL,
    PreemptionClockError,
    PreemptionError,
    PreemptionFailClosed,
    PreemptionFailClosedEvidenceV1,
    SliceDrainDecisionV1,
    SlicePreemptionGuard,
)
from moe_sae.r1.protocol import AttemptLeaseV1, ProtocolError


DEPLOYMENT = "d" * 64
MEASUREMENT = "e" * 64
CHECKPOINT = "c" * 64


class FakeClock:
    def __init__(self, now_ms: object = 1_100) -> None:
        self.now_ms = now_ms
        self.calls = 0
        self.error: Exception | None = None

    def __call__(self) -> Any:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.now_ms


def _lease(
    *,
    attempt_id: str = "attempt-1",
    fence: int = 1,
    expires_at: int = 100_000,
) -> AttemptLeaseV1:
    return AttemptLeaseV1(
        campaign_id="r1-study",
        work_id="sae-work-1",
        attempt_id=attempt_id,
        fence=fence,
        executor_id=f"executor-{attempt_id}",
        deployment_identity_sha256=DEPLOYMENT,
        acquired_at_unix_ms=1_000,
        expires_at_unix_ms=expires_at,
    )


def _timing(
    *, p99_ms: int = 60_000, margin_ms: int = 15_000
) -> CheckpointTimingEvidenceV1:
    return CheckpointTimingEvidenceV1(
        measurement_evidence_sha256=MEASUREMENT,
        measurement_sample_count=200,
        full_save_finalize_upload_cas_p99_ms=p99_ms,
        explicit_safety_margin_ms=margin_ms,
    )


def _guard(
    *,
    lease: AttemptLeaseV1 | None = None,
    timing: CheckpointTimingEvidenceV1 | None = None,
    clock: FakeClock | None = None,
    sink: Any | None = None,
) -> tuple[SlicePreemptionGuard, AttemptLeaseV1, FakeClock, list[Any]]:
    lease = _lease() if lease is None else lease
    timing = _timing() if timing is None else timing
    clock = FakeClock() if clock is None else clock
    incidents: list[Any] = []
    actual_sink = incidents.append if sink is None else sink
    return (
        SlicePreemptionGuard(
            lease,
            checkpoint_timing=timing,
            clock_ms=clock,
            incident_sink=actual_sink,
        ),
        lease,
        clock,
        incidents,
    )


def _poll(
    guard: SlicePreemptionGuard,
    lease: AttemptLeaseV1,
    *,
    body: object,
    status: object = 200,
    flavor: object = "Google",
    checkpoint: object = CHECKPOINT,
) -> SliceDrainDecisionV1 | None:
    return guard.observe_metadata(
        lease,
        observer_id="r1-host-00",
        http_status=status,
        metadata_flavor=flavor,
        body=body,
        last_sealed_checkpoint_manifest_sha256=checkpoint,  # type: ignore[arg-type]
    )


def test_attempt_binding_is_stable_across_renewal_but_not_takeover() -> None:
    lease = _lease()
    binding = AttemptBindingV1.from_lease(lease)
    renewed = replace(lease, expires_at_unix_ms=200_000)
    takeover = _lease(attempt_id="attempt-2", fence=2)

    assert binding.matches(renewed)
    assert not binding.matches(takeover)
    assert AttemptBindingV1.from_json(binding.canonical_json()) == binding
    with pytest.raises(FrozenInstanceError):
        binding.fence = 2  # type: ignore[misc]


def test_checkpoint_timing_uses_fixed_inclusive_75_second_rule() -> None:
    exact = _timing(p99_ms=60_000, margin_ms=15_000)
    over = _timing(p99_ms=60_001, margin_ms=15_000)

    assert exact.required_drain_ms == CONSERVATIVE_DRAIN_BUDGET_MS
    assert exact.measurement_sample_count == MIN_CHECKPOINT_TIMING_SAMPLES
    assert exact.fits_conservative_budget
    assert not over.fits_conservative_budget
    assert CheckpointTimingEvidenceV1.from_json(exact.canonical_json()) == exact


@pytest.mark.parametrize(
    "mutation",
    [
        {"measurement_sample_count": 0},
        {"measurement_sample_count": 199},
        {"measurement_sample_count": True},
        {"full_save_finalize_upload_cas_p99_ms": 0},
        {"explicit_safety_margin_ms": 0},
    ],
)
def test_checkpoint_timing_rejects_unmeasured_or_implicit_margin(
    mutation: dict[str, object],
) -> None:
    with pytest.raises(ProtocolError):
        replace(_timing(), **mutation)


def test_false_observations_are_strict_chained_evidence_and_leave_gate_open() -> None:
    guard, lease, clock, incidents = _guard()

    assert _poll(guard, lease, body=b"FALSE") is None
    first = guard.last_observation
    assert first is not None
    assert first.raw_value == "FALSE"
    assert first.observation_index == 1
    assert first.previous_observation_sha256 is None
    assert first.metadata_url == PREEMPTED_METADATA_URL
    assert MetadataPreemptionObservationV1.from_json(first.canonical_json()) == first

    clock.now_ms = 1_200
    assert _poll(guard, lease, body=b"FALSE") is None
    second = guard.last_observation
    assert second is not None
    assert second.observation_index == 2
    assert second.previous_observation_sha256 == first.digest
    assert guard.new_steps_allowed
    assert not incidents


def test_first_true_closes_gate_before_sink_and_selects_emergency_at_boundary() -> None:
    lease = _lease()
    clock = FakeClock()
    published: list[SliceDrainDecisionV1] = []
    gate_was_closed: list[bool] = []
    guard: SlicePreemptionGuard

    def sink(evidence: Any) -> None:
        gate_was_closed.append(not guard.new_steps_allowed)
        published.append(evidence)

    guard = SlicePreemptionGuard(
        lease,
        checkpoint_timing=_timing(),
        clock_ms=clock,
        incident_sink=sink,
    )

    assert _poll(guard, lease, body=b"FALSE") is None
    first = guard.last_observation
    assert first is not None
    clock.now_ms = 1_200
    decision = _poll(guard, lease, body=b"TRUE")

    assert decision is not None
    assert not guard.new_steps_allowed
    assert gate_was_closed == [True]
    assert published == [decision]
    assert guard.incident_published
    assert decision.action == "EMERGENCY_CHECKPOINT"
    assert decision.reason == "MEASURED_P99_PLUS_MARGIN_FITS_75S_BUDGET"
    assert decision.estimated_checkpoint_completion_ms == 75_000
    assert decision.emergency_checkpoint_deadline_unix_ms == 76_200
    assert decision.trigger.previous_observation_sha256 == first.digest
    assert decision.last_sealed_checkpoint_manifest_sha256 == CHECKPOINT
    assert SliceDrainDecisionV1.from_json(decision.canonical_json()) == decision

    calls = clock.calls
    repeated = guard.observe_metadata(
        lease,
        observer_id="ignored-after-terminal",
        http_status=503,
        metadata_flavor=None,
        body=b"not-a-value",
        last_sealed_checkpoint_manifest_sha256=None,
    )
    assert repeated is decision
    assert repeated.digest == decision.digest
    assert clock.calls == calls
    assert published == [decision]
    with pytest.raises(NewStepsForbidden) as blocked:
        guard.authorize_new_step(
            lease,
            last_sealed_checkpoint_manifest_sha256=CHECKPOINT,
        )
    assert blocked.value.evidence is decision


def test_checkpoint_is_not_attempted_when_p99_plus_margin_misses_by_one_ms() -> None:
    guard, lease, _clock, incidents = _guard(
        timing=_timing(p99_ms=60_001, margin_ms=15_000)
    )

    decision = _poll(guard, lease, body=b"TRUE")

    assert decision is not None
    assert decision.estimated_checkpoint_completion_ms == 75_001
    assert decision.action == "PRESERVE_LAST_SEALED_CHECKPOINT"
    assert decision.reason == "MEASURED_P99_PLUS_MARGIN_EXCEEDS_75S_BUDGET"
    assert decision.emergency_checkpoint_deadline_unix_ms is None
    assert decision.last_sealed_checkpoint_manifest_sha256 == CHECKPOINT
    assert incidents == [decision]


def test_preserve_action_honestly_records_absence_of_a_prior_seal() -> None:
    guard, lease, _clock, _incidents = _guard(
        timing=_timing(p99_ms=74_999, margin_ms=2)
    )
    decision = guard.observe_metadata(
        lease,
        observer_id="r1-host-00",
        http_status=200,
        metadata_flavor="Google",
        body=b"TRUE",
        last_sealed_checkpoint_manifest_sha256=None,
    )

    assert decision is not None
    assert decision.action == "PRESERVE_LAST_SEALED_CHECKPOINT"
    assert decision.last_sealed_checkpoint_manifest_sha256 is None


@pytest.mark.parametrize(
    ("status", "flavor", "body", "fault_code"),
    [
        (True, "Google", b"TRUE", "METADATA_HTTP_STATUS_AMBIGUOUS"),
        (503, "Google", b"TRUE", "METADATA_HTTP_STATUS_AMBIGUOUS"),
        (200, None, b"TRUE", "METADATA_FLAVOR_AMBIGUOUS"),
        (200, "google", b"TRUE", "METADATA_FLAVOR_AMBIGUOUS"),
        (200, "Google", "TRUE", "METADATA_BODY_TYPE_AMBIGUOUS"),
        (200, "Google", bytearray(b"TRUE"), "METADATA_BODY_TYPE_AMBIGUOUS"),
        (200, "Google", b"true", "METADATA_VALUE_AMBIGUOUS"),
        (200, "Google", b"TRUE\n", "METADATA_VALUE_AMBIGUOUS"),
    ],
)
def test_ambiguous_metadata_fails_closed_without_emergency_checkpoint(
    status: object,
    flavor: object,
    body: object,
    fault_code: str,
) -> None:
    guard, lease, clock, incidents = _guard()

    with pytest.raises(AmbiguousMetadata) as raised:
        _poll(
            guard,
            lease,
            status=status,
            flavor=flavor,
            body=body,
        )

    evidence = raised.value.evidence
    assert isinstance(evidence, PreemptionFailClosedEvidenceV1)
    assert evidence.fault_code == fault_code
    assert evidence.action == "PRESERVE_LAST_SEALED_CHECKPOINT"
    assert evidence.new_steps_forbidden
    assert evidence.last_sealed_checkpoint_manifest_sha256 == CHECKPOINT
    assert not guard.new_steps_allowed
    assert incidents == [evidence]
    assert (
        PreemptionFailClosedEvidenceV1.from_json(evidence.canonical_json()) == evidence
    )

    calls = clock.calls
    with pytest.raises(AmbiguousMetadata) as repeated:
        _poll(guard, lease, body=b"TRUE")
    assert repeated.value.evidence is evidence
    assert clock.calls == calls
    assert incidents == [evidence]


@pytest.mark.parametrize(
    ("sample", "fault_code"),
    [
        (True, "CLOCK_VALUE_INVALID"),
        (1.5, "CLOCK_VALUE_INVALID"),
        (-1, "CLOCK_VALUE_INVALID"),
        (999, "CLOCK_BEFORE_ATTEMPT"),
    ],
)
def test_malformed_or_pre_acquisition_clock_fails_closed(
    sample: object, fault_code: str
) -> None:
    clock = FakeClock(sample)
    guard, lease, _clock, incidents = _guard(clock=clock)

    with pytest.raises(PreemptionClockError) as raised:
        guard.authorize_new_step(
            lease,
            last_sealed_checkpoint_manifest_sha256=CHECKPOINT,
        )

    assert raised.value.evidence.fault_code == fault_code
    assert raised.value.evidence.action == "PRESERVE_LAST_SEALED_CHECKPOINT"
    assert not guard.new_steps_allowed
    assert incidents == [raised.value.evidence]


def test_clock_exception_fails_closed_without_embedding_untrusted_message() -> None:
    clock = FakeClock()
    clock.error = RuntimeError("sensitive and nondeterministic detail")
    guard, lease, _clock, _incidents = _guard(clock=clock)

    with pytest.raises(PreemptionClockError) as raised:
        _poll(guard, lease, body=b"FALSE")

    assert raised.value.evidence.fault_code == "CLOCK_READ_FAILED"
    assert "sensitive" not in raised.value.evidence.canonical_json()


def test_regressing_clock_after_valid_observation_fails_closed() -> None:
    guard, lease, clock, incidents = _guard()
    assert _poll(guard, lease, body=b"FALSE") is None
    clock.now_ms = 1_099

    with pytest.raises(PreemptionClockError) as raised:
        guard.authorize_new_step(
            lease,
            last_sealed_checkpoint_manifest_sha256=CHECKPOINT,
        )

    evidence = raised.value.evidence
    assert evidence.fault_code == "CLOCK_REGRESSED"
    assert evidence.trusted_at_or_before_unix_ms == 1_100
    assert evidence.rejected_clock_sample_unix_ms == 1_099
    assert incidents == [evidence]


def test_equal_clock_samples_are_not_falsely_treated_as_regression() -> None:
    guard, lease, _clock, _incidents = _guard()
    assert _poll(guard, lease, body=b"FALSE") is None
    assert (
        guard.authorize_new_step(
            lease,
            last_sealed_checkpoint_manifest_sha256=CHECKPOINT,
        )
        == 1_100
    )
    assert guard.new_steps_allowed


def test_expired_lease_fails_closed_and_names_the_expiry() -> None:
    lease = _lease(expires_at=1_100)
    guard, _lease_value, _clock, incidents = _guard(lease=lease)

    with pytest.raises(LeaseExpiredDuringDrain) as raised:
        _poll(guard, lease, body=b"FALSE")

    assert raised.value.evidence.fault_code == "LEASE_EXPIRED"
    assert raised.value.evidence.lease_expires_at_unix_ms == 1_100
    assert incidents == [raised.value.evidence]


def test_wrong_attempt_or_fence_cannot_observe_or_poison_current_guard() -> None:
    guard, lease, clock, incidents = _guard()
    takeover = _lease(attempt_id="attempt-2", fence=2)

    with pytest.raises(AttemptBindingMismatch):
        _poll(guard, takeover, body=b"TRUE")

    assert guard.new_steps_allowed
    assert guard.incident is None
    assert clock.calls == 0
    assert not incidents
    assert _poll(guard, lease, body=b"FALSE") is None


def test_renewed_lease_for_same_attempt_and_fence_is_accepted() -> None:
    lease = _lease(expires_at=1_050)
    guard, _lease_value, clock, _incidents = _guard(lease=lease)
    renewed = replace(lease, expires_at_unix_ms=2_000)
    clock.now_ms = 1_100

    assert _poll(guard, renewed, body=b"FALSE") is None
    assert guard.last_observation is not None
    assert guard.last_observation.lease_expires_at_unix_ms == 2_000


def test_true_observation_is_linearized_once_under_concurrency() -> None:
    guard, lease, clock, incidents = _guard()

    def observe(_index: int) -> SliceDrainDecisionV1 | None:
        return _poll(guard, lease, body=b"TRUE")

    with ThreadPoolExecutor(max_workers=16) as executor:
        decisions = list(executor.map(observe, range(32)))

    assert all(item is decisions[0] for item in decisions)
    assert isinstance(decisions[0], SliceDrainDecisionV1)
    assert clock.calls == 1
    assert incidents == [decisions[0]]


def test_transient_sink_failure_requires_explicit_same_evidence_retry() -> None:
    calls: list[Any] = []

    def transient_sink(evidence: Any) -> None:
        calls.append(evidence)
        if len(calls) == 1:
            raise RuntimeError("object store unavailable")

    guard, lease, clock, _incidents = _guard(sink=transient_sink)

    with pytest.raises(IncidentSinkError) as first:
        _poll(guard, lease, body=b"TRUE")

    assert first.value.evidence is guard.incident
    assert not guard.new_steps_allowed
    assert not guard.incident_published
    assert len(calls) == 1
    clock_calls = clock.calls
    with pytest.raises(IncidentSinkError) as repeated:
        _poll(guard, lease, body=b"TRUE")
    assert repeated.value.evidence is first.value.evidence
    assert len(calls) == 1
    assert clock.calls == clock_calls

    assert guard.retry_incident_publication()
    assert calls == [first.value.evidence, first.value.evidence]
    assert calls[0] is calls[1]
    assert guard.incident is first.value.evidence
    assert guard.incident_published
    assert not guard.new_steps_allowed
    assert clock.calls == clock_calls

    assert guard.retry_incident_publication() is False
    assert len(calls) == 2
    assert clock.calls == clock_calls
    assert _poll(guard, lease, body=b"FALSE") is first.value.evidence
    assert len(calls) == 2
    with pytest.raises(NewStepsForbidden):
        guard.authorize_new_step(
            lease,
            last_sealed_checkpoint_manifest_sha256=CHECKPOINT,
        )


def test_explicit_publication_retry_can_fail_repeatedly_without_mutation() -> None:
    calls: list[Any] = []

    def broken_sink(evidence: Any) -> None:
        calls.append(evidence)
        raise RuntimeError(f"failure-{len(calls)}")

    guard, lease, clock, _incidents = _guard(sink=broken_sink)
    with pytest.raises(IncidentSinkError) as first:
        _poll(guard, lease, body=b"TRUE")
    incident = first.value.evidence
    digest = incident.digest
    clock_calls = clock.calls

    with pytest.raises(IncidentSinkError) as second:
        guard.retry_incident_publication()

    assert second.value.evidence is incident
    assert guard.incident is incident
    assert guard.incident.digest == digest
    assert calls == [incident, incident]
    assert not guard.new_steps_allowed
    assert not guard.incident_published
    assert clock.calls == clock_calls


def test_publication_retry_before_an_incident_is_rejected_without_clock_read() -> None:
    guard, _lease_value, clock, incidents = _guard()

    with pytest.raises(PreemptionError, match="before a terminal incident"):
        guard.retry_incident_publication()

    assert guard.new_steps_allowed
    assert guard.incident is None
    assert not incidents
    assert clock.calls == 0


def test_invalid_checkpoint_reference_fails_closed_and_is_not_echoed() -> None:
    guard, lease, _clock, incidents = _guard()

    with pytest.raises(PreemptionFailClosed) as raised:
        _poll(guard, lease, body=b"FALSE", checkpoint="not-a-digest")

    evidence = raised.value.evidence
    assert evidence.fault_code == "CHECKPOINT_REFERENCE_INVALID"
    assert evidence.last_sealed_checkpoint_manifest_sha256 is None
    assert "not-a-digest" not in evidence.canonical_json()
    assert incidents == [evidence]


def test_decision_parser_rejects_unknown_fields_and_inconsistent_action() -> None:
    guard, lease, _clock, _incidents = _guard()
    decision = _poll(guard, lease, body=b"TRUE")
    assert decision is not None
    payload = decision.as_dict()
    payload["unknown"] = "forbidden"
    with pytest.raises(ProtocolError, match="unknown fields"):
        SliceDrainDecisionV1.from_dict(payload)

    payload = decision.as_dict()
    payload["action"] = "PRESERVE_LAST_SEALED_CHECKPOINT"
    with pytest.raises(ProtocolError, match="fixed timing rule"):
        SliceDrainDecisionV1.from_dict(payload)


def test_duplicate_json_fields_are_rejected_at_incident_boundary() -> None:
    guard, lease, _clock, _incidents = _guard()
    decision = _poll(guard, lease, body=b"TRUE")
    assert decision is not None
    encoded = decision.canonical_json()
    duplicate = encoded.replace(
        '"schema":"moe-sae.r1.slice-drain-decision.v1"',
        '"schema":"moe-sae.r1.slice-drain-decision.v1",'
        '"schema":"moe-sae.r1.slice-drain-decision.v1"',
        1,
    )
    with pytest.raises(ProtocolError, match="duplicate JSON field"):
        SliceDrainDecisionV1.from_json(duplicate)


def test_incident_bytes_are_deterministic_for_identical_inputs() -> None:
    first_guard, first_lease, _clock, _incidents = _guard()
    second_guard, second_lease, _clock2, _incidents2 = _guard()
    first = _poll(first_guard, first_lease, body=b"TRUE")
    second = _poll(second_guard, second_lease, body=b"TRUE")

    assert first is not None and second is not None
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.digest == second.digest
    assert json.loads(first.canonical_json())["new_steps_forbidden"] is True
