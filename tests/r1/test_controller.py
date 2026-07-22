from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any

import pytest

from moe_sae.r1.control_store import CASConflict, MemoryJSONCASStore
from moe_sae.r1.controller import (
    CampaignConflict,
    CampaignController,
    ContentVerificationFailed,
    InvalidProgress,
    InvalidReceipt,
    LeaseUnavailable,
    RetryLimitExceeded,
    StaleAttempt,
)
from moe_sae.r1.protocol import (
    AttemptLeaseV1,
    AttemptReceiptV1,
    ContentRefV1,
    CostLedgerV1,
    HarvestRangeV1,
    ProgressV1,
    RegionalReplicaV1,
    SAEJobV1,
)


SCIENCE = "a" * 64
CORPUS = "b" * 64
CAPTURE = "c" * 64
ORDER = "d" * 64
DEPLOYMENT = "e" * 64


class FakeClock:
    def __init__(self, now_ms: int = 1_000) -> None:
        self.now_ms = now_ms
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        return self.now_ms

    def advance(self, milliseconds: int) -> None:
        self.now_ms += milliseconds


def _harvest(*, record_start: int = 0) -> HarvestRangeV1:
    return HarvestRangeV1(
        scientific_identity_sha256=SCIENCE,
        corpus_fingerprint_sha256=CORPUS,
        split="en/train",
        record_start=record_start,
        record_end=record_start + 100,
        scientific_token_start=record_start * 1_000,
        scientific_token_end=(record_start + 100) * 1_000,
    )


def _sae(*, condition: str = "A", seed: int = 1) -> SAEJobV1:
    return SAEJobV1(
        scientific_identity_sha256=SCIENCE,
        capture_manifest_sha256=CAPTURE,
        activation_order_sha256=ORDER,
        condition=condition,
        seed=seed,
    )


def _content(
    character: str = "1",
    *,
    region: str = "us-central1",
    generation: int = 7,
) -> ContentRefV1:
    sha = character * 64
    replica = RegionalReplicaV1(
        region=region,
        uri=f"gs://r1-{region}/objects/{sha}",
        generation=generation,
        size_bytes=4096,
        crc32c="AAAAAA==",
        sha256=sha,
    )
    return ContentRefV1(
        sha256=sha,
        size_bytes=4096,
        media_type="application/x-safetensors",
        replicas=(replica,),
    )


def _cost() -> CostLedgerV1:
    return CostLedgerV1(
        successful_chip_seconds=640.0,
        attempted_chip_seconds=704.0,
        compilation_chip_seconds=32.0,
        checkpoint_seconds=4.0,
        queue_seconds=10.0,
        egress_bytes=0,
        replayed_units=0,
    )


def _controller(
    *,
    store: Any | None = None,
    clock: FakeClock | None = None,
    verifier: Any | None = None,
    campaign_id: str = "r1-study",
    max_cas_retries: int = 8,
) -> tuple[CampaignController, MemoryJSONCASStore, FakeClock]:
    memory = MemoryJSONCASStore() if store is None else store
    clock = FakeClock() if clock is None else clock
    controller = CampaignController(
        memory,
        base_uri="memory://control",
        campaign_id=campaign_id,
        clock_ms=clock,
        content_verifier=(lambda _content: True) if verifier is None else verifier,
        max_cas_retries=max_cas_retries,
    )
    return controller, memory, clock


def _acquire(
    controller: CampaignController,
    work_id: str,
    *,
    attempt: str = "attempt-1",
    duration: int = 10_000,
) -> AttemptLeaseV1:
    return controller.acquire(
        work_id,
        attempt_id=attempt,
        executor_id=f"executor-{attempt}",
        deployment_identity_sha256=DEPLOYMENT,
        lease_duration_ms=duration,
    )


def _progress(
    lease: AttemptLeaseV1,
    *,
    sequence: int,
    completed: int,
    total: int = 100,
    updated: int = 1_100,
) -> ProgressV1:
    return ProgressV1(
        work_id=lease.work_id,
        attempt_id=lease.attempt_id,
        fence=lease.fence,
        sequence=sequence,
        phase="train",
        completed_units=completed,
        total_units=total,
        updated_at_unix_ms=updated,
    )


def _receipt(
    lease: AttemptLeaseV1,
    output: ContentRefV1,
    *,
    sequence: int,
    outcome: str = "SUCCEEDED",
) -> AttemptReceiptV1:
    return AttemptReceiptV1(
        work_id=lease.work_id,
        attempt_id=lease.attempt_id,
        fence=lease.fence,
        final_progress_sequence=sequence,
        outcome=outcome,
        outputs=(output,),
        cost=_cost(),
        termination_reason=None if outcome == "SUCCEEDED" else "executor failed",
        finished_at_unix_ms=2_000,
    )


def test_initialize_mixed_immutable_inventory_and_idempotent_reopen() -> None:
    controller, _store, _clock = _controller()
    harvest = _harvest()
    sae = _sae()

    first = controller.initialize((sae, harvest))
    second = controller.initialize((harvest, sae))

    assert first.inventory_sha256 == second.inventory_sha256
    assert first.revision == 0
    assert first.event_count == 1
    assert [item.work_id for item in first.works] == sorted(
        (harvest.work_id, sae.work_id)
    )
    assert all(item.status == "PENDING" for item in first.works)
    assert controller.replay() == controller.view()


def test_same_campaign_name_rejects_different_inventory() -> None:
    controller, store, clock = _controller()
    controller.initialize((_harvest(),))
    other = CampaignController(
        store,
        base_uri="memory://control",
        campaign_id="r1-study",
        clock_ms=clock,
        content_verifier=lambda _content: True,
    )
    with pytest.raises(CampaignConflict, match="different inventory"):
        other.initialize((_harvest(record_start=100),))


def test_concurrent_acquire_has_exactly_one_winner() -> None:
    controller, _store, _clock = _controller()
    work = _harvest()
    controller.initialize((work,))
    barrier = threading.Barrier(2)

    def contender(number: int) -> tuple[str, int | None]:
        barrier.wait()
        try:
            lease = _acquire(controller, work.work_id, attempt=f"attempt-{number}")
            return "won", lease.fence
        except LeaseUnavailable:
            return "lost", None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(contender, (1, 2)))

    assert sorted(result for result, _fence in outcomes) == ["lost", "won"]
    assert {fence for result, fence in outcomes if result == "won"} == {1}
    view = controller.replay()
    assert view.revision == 1
    assert view.event_count == 2
    assert view.work(work.work_id).status == "LEASED"


def test_expiry_is_clock_deterministic_and_takeover_increments_fence() -> None:
    controller, _store, clock = _controller()
    work = _harvest()
    controller.initialize((work,))
    first = _acquire(controller, work.work_id, duration=100)

    clock.advance(99)
    with pytest.raises(LeaseUnavailable, match="not expired"):
        controller.expire_and_requeue(work.work_id)
    clock.advance(1)
    pending = controller.expire_and_requeue(work.work_id)
    second = _acquire(controller, work.work_id, attempt="attempt-2")

    assert pending.status == "PENDING"
    assert pending.fence == 1
    assert second.fence == first.fence + 1
    assert controller.replay().work(work.work_id).lease == second


def test_each_transition_samples_controller_clock_exactly_once() -> None:
    controller, _store, clock = _controller()
    work = _harvest()
    controller.initialize((work,))
    calls_before = clock.calls

    lease = _acquire(controller, work.work_id)

    assert clock.calls == calls_before + 1
    assert lease.acquired_at_unix_ms == clock.now_ms


def test_renew_requires_current_unexpired_exact_lease_and_extends_expiry() -> None:
    controller, _store, clock = _controller()
    work = _harvest()
    controller.initialize((work,))
    lease = _acquire(controller, work.work_id, duration=100)
    clock.advance(50)

    with pytest.raises(LeaseUnavailable, match="extend"):
        controller.renew(lease, lease_duration_ms=25)
    renewed = controller.renew(lease, lease_duration_ms=100)
    assert renewed.expires_at_unix_ms == clock.now_ms + 100

    with pytest.raises(StaleAttempt):
        controller.renew(lease, lease_duration_ms=200)
    clock.advance(100)
    with pytest.raises(StaleAttempt, match="expired"):
        controller.renew(renewed, lease_duration_ms=200)


def test_progress_requires_sequence_and_completed_units_to_strictly_increase() -> None:
    controller, _store, _clock = _controller()
    work = _sae()
    controller.initialize((work,))
    lease = _acquire(controller, work.work_id)
    first = controller.record_progress(
        _progress(lease, sequence=1, completed=10, total=100)
    )

    with pytest.raises(InvalidProgress, match="sequence"):
        controller.record_progress(
            _progress(lease, sequence=1, completed=20, total=100, updated=1_200)
        )
    with pytest.raises(InvalidProgress, match="completed_units"):
        controller.record_progress(
            _progress(lease, sequence=2, completed=10, total=100, updated=1_200)
        )
    with pytest.raises(InvalidProgress, match="total_units"):
        controller.record_progress(
            _progress(lease, sequence=2, completed=20, total=101, updated=1_200)
        )
    accepted = controller.record_progress(
        _progress(lease, sequence=2, completed=20, total=100, updated=1_200)
    )

    assert first.sequence == 1
    assert accepted.sequence == 2
    assert controller.view().work(work.work_id).progress == accepted


def test_stale_progress_and_commit_are_rejected_after_takeover() -> None:
    controller, _store, clock = _controller()
    work = _sae()
    controller.initialize((work,))
    first = _acquire(controller, work.work_id, duration=100)
    controller.record_progress(_progress(first, sequence=1, completed=100, total=100))
    old_receipt = _receipt(first, _content(), sequence=1)
    clock.advance(100)
    controller.expire_and_requeue(work.work_id)
    _acquire(controller, work.work_id, attempt="attempt-2")

    with pytest.raises(StaleAttempt):
        controller.record_progress(
            _progress(first, sequence=2, completed=100, total=100)
        )
    with pytest.raises(StaleAttempt):
        controller.commit(old_receipt)
    with pytest.raises(StaleAttempt):
        controller.adopt_orphan(old_receipt)


def test_progress_totals_and_monotonic_floors_survive_takeover() -> None:
    controller, _store, clock = _controller()
    work = _sae()
    controller.initialize((work,))
    first = _acquire(controller, work.work_id, duration=100)
    controller.record_progress(_progress(first, sequence=5, completed=40, total=100))
    clock.advance(100)
    controller.expire_and_requeue(work.work_id)
    second = _acquire(controller, work.work_id, attempt="attempt-2")

    with pytest.raises(InvalidProgress, match="total_units"):
        controller.record_progress(
            _progress(second, sequence=6, completed=50, total=101, updated=1_300)
        )
    with pytest.raises(InvalidProgress, match="sequence"):
        controller.record_progress(
            _progress(second, sequence=5, completed=50, total=100, updated=1_300)
        )
    with pytest.raises(InvalidProgress, match="completed_units"):
        controller.record_progress(
            _progress(second, sequence=6, completed=40, total=100, updated=1_300)
        )
    accepted = controller.record_progress(
        _progress(second, sequence=6, completed=50, total=100, updated=1_300)
    )

    view = controller.replay().work(work.work_id)
    assert view.progress == accepted
    assert view.last_progress_sequence == 6
    assert view.completed_units == 50
    assert view.total_units == 100


def test_commit_requires_current_complete_progress_and_verified_content() -> None:
    verified: list[str] = []

    def verifier(content: ContentRefV1) -> bool:
        verified.append(content.sha256)
        return True

    controller, _store, _clock = _controller(verifier=verifier)
    work = _sae()
    controller.initialize((work,))
    lease = _acquire(controller, work.work_id)
    output = _content()
    controller.record_progress(_progress(lease, sequence=1, completed=50, total=100))
    with pytest.raises(InvalidReceipt, match="completed"):
        controller.commit(_receipt(lease, output, sequence=1))

    controller.record_progress(
        _progress(lease, sequence=2, completed=100, total=100, updated=1_200)
    )
    with pytest.raises(InvalidReceipt, match="progress sequence"):
        controller.commit(_receipt(lease, output, sequence=1))
    receipt = _receipt(lease, output, sequence=2)
    assert controller.commit(receipt) == receipt
    assert verified == [output.sha256]
    assert controller.replay().work(work.work_id).status == "COMMITTED"


def test_content_verifier_fails_closed_without_mutating_state() -> None:
    controller, _store, _clock = _controller(verifier=lambda _content: False)
    work = _sae()
    controller.initialize((work,))
    lease = _acquire(controller, work.work_id)
    controller.record_progress(_progress(lease, sequence=1, completed=100, total=100))
    revision = controller.view().revision

    with pytest.raises(ContentVerificationFailed, match="rejected"):
        controller.commit(_receipt(lease, _content(), sequence=1))
    assert controller.view().revision == revision
    assert controller.view().work(work.work_id).status == "LEASED"


def test_orphan_adoption_matches_exact_logical_identity_not_replica_location() -> None:
    expected = _content(region="us-central1", generation=7)
    regional_replica = _content(region="europe-west4", generation=91)
    controller, _store, clock = _controller()
    work = _harvest()
    controller.initialize((work,), expected_outputs={work.work_id: (expected,)})
    lease = _acquire(controller, work.work_id, duration=100)
    controller.record_progress(_progress(lease, sequence=1, completed=100, total=100))
    clock.advance(100)
    controller.expire_and_requeue(work.work_id)

    receipt = _receipt(lease, regional_replica, sequence=1)
    assert controller.adopt_orphan(receipt) == receipt
    assert controller.replay().work(work.work_id).status == "COMMITTED"


def test_orphan_adoption_rejects_wrong_or_uncommitted_logical_identity() -> None:
    expected = _content("1")
    wrong = _content("2")
    controller, _store, clock = _controller()
    work = _harvest()
    controller.initialize((work,), expected_outputs={work.work_id: (expected,)})
    lease = _acquire(controller, work.work_id, duration=100)
    controller.record_progress(_progress(lease, sequence=1, completed=100, total=100))
    clock.advance(100)
    controller.expire_and_requeue(work.work_id)
    with pytest.raises(InvalidReceipt, match="exactly match"):
        controller.adopt_orphan(_receipt(lease, wrong, sequence=1))

    other, _other_store, other_clock = _controller(campaign_id="without-expectation")
    other_work = _harvest()
    other.initialize((other_work,))
    other_lease = _acquire(other, other_work.work_id, duration=100)
    other.record_progress(_progress(other_lease, sequence=1, completed=100, total=100))
    other_clock.advance(100)
    other.expire_and_requeue(other_work.work_id)
    with pytest.raises(InvalidReceipt, match="precommitted"):
        other.adopt_orphan(_receipt(other_lease, expected, sequence=1))


class TracingStore:
    def __init__(self) -> None:
        self.delegate = MemoryJSONCASStore()
        self.calls: list[tuple[str, str]] = []

    def read(self, uri: str) -> Any:
        return self.delegate.read(uri)

    def create(self, uri: str, value: Any) -> Any:
        self.calls.append(("create", uri))
        return self.delegate.create(uri, value)

    def compare_and_swap(self, uri: str, expected_generation: int, value: Any) -> Any:
        self.calls.append(("cas", uri))
        return self.delegate.compare_and_swap(uri, expected_generation, value)

    def put_immutable(self, uri: str, value: Any) -> Any:
        self.calls.append(("immutable", uri))
        return self.delegate.put_immutable(uri, value)


def test_event_and_snapshot_are_immutable_before_authority_cas() -> None:
    store = TracingStore()
    controller, _unused, _clock = _controller(store=store)
    work = _harvest()
    controller.initialize((work,))
    store.calls.clear()
    _acquire(controller, work.work_id)

    assert [kind for kind, _uri in store.calls] == ["immutable", "immutable", "cas"]
    assert "/events/" in store.calls[0][1]
    assert "/snapshots/" in store.calls[1][1]
    assert store.calls[2][1] == controller.head_uri


class ConflictingCASStore(TracingStore):
    def __init__(self) -> None:
        super().__init__()
        self.force_conflict = False
        self.cas_attempts = 0

    def compare_and_swap(self, uri: str, expected_generation: int, value: Any) -> Any:
        self.cas_attempts += 1
        if self.force_conflict:
            current = self.delegate.read(uri)
            raise CASConflict(
                uri,
                expected_generation=expected_generation,
                actual_generation=None if current is None else current.generation,
            )
        return super().compare_and_swap(uri, expected_generation, value)


def test_head_cas_retries_are_bounded_and_losers_never_change_head() -> None:
    store = ConflictingCASStore()
    controller, _unused, _clock = _controller(store=store, max_cas_retries=3)
    work = _harvest()
    controller.initialize((work,))
    accepted = controller.view()
    store.force_conflict = True

    with pytest.raises(RetryLimitExceeded, match="3 attempts"):
        _acquire(controller, work.work_id)
    assert store.cas_attempts == 3
    assert controller.view() == accepted
    assert controller.replay() == accepted


def test_replay_follows_head_chain_after_full_lifecycle() -> None:
    controller, _store, clock = _controller()
    first_work = _harvest()
    second_work = _sae()
    controller.initialize((first_work, second_work))
    first_lease = _acquire(controller, first_work.work_id, duration=100)
    controller.record_progress(
        _progress(first_lease, sequence=1, completed=10, total=100)
    )
    clock.advance(100)
    controller.expire_and_requeue(first_work.work_id)
    second_lease = _acquire(controller, second_work.work_id)
    controller.record_progress(
        _progress(second_lease, sequence=1, completed=100, total=100)
    )
    controller.commit(_receipt(second_lease, _content(), sequence=1))

    direct = controller.view()
    replayed = controller.replay()
    assert replayed == direct
    assert replayed.revision == 6
    assert replayed.event_count == 7
    assert replayed.work(first_work.work_id).status == "PENDING"
    assert replayed.work(second_work.work_id).status == "COMMITTED"
