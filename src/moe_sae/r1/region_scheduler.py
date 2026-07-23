"""Bounded region-local scheduling before v6e-64 provisioning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Callable, Sequence

from .control_store import CASConflict, JSONCASStore
from .preemption_v2 import AttemptAuthorityV2
from .protocol import (
    ContentRefV1,
    IDENTIFIER_PATTERN,
    ProtocolError,
    canonical_json_bytes,
    require_integer,
    require_record_object,
    require_string,
)


MAX_CANDIDATES = 64
MAX_BULK_INPUTS = 8
MAX_TOTAL_REPLICAS = 64
MAX_CAS_RETRIES = 8
MAX_HEAD_BYTES = 2048
_HEAD_SCHEMA = "moe-sae.r1.region-schedule-head.v1"
_ZONE = re.compile(r"[a-z]+-[a-z0-9]+[0-9]-[a-z][a-z0-9]*")


class RegionSchedulingError(RuntimeError):
    pass


class NoLocalCandidate(RegionSchedulingError):
    pass


class ClaimConflict(RegionSchedulingError):
    pass


def _identifier(value: Any, name: str) -> str:
    return require_string(value, name, pattern=IDENTIFIER_PATTERN)


def _zone(value: Any) -> str:
    return require_string(value, "zone", pattern=_ZONE)


def _region(zone: str) -> str:
    return _zone(zone).rsplit("-", 1)[0]


def _records(value: Any, name: str, kind: type, maximum: int) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise RegionSchedulingError(f"{name} must be a bounded list or tuple")
    items = tuple(value)
    if not 1 <= len(items) <= maximum or any(type(item) is not kind for item in items):
        raise RegionSchedulingError(f"{name} is empty, oversized, or malformed")
    return items


@dataclass(frozen=True, slots=True)
class V6E64Candidate:
    """One already-eligible complete Spot v6e-64 slice."""

    zone: str
    hourly_slice_cost_microusd: int
    queue_priority: int = 0

    def __post_init__(self) -> None:
        _zone(self.zone)
        require_integer(
            self.hourly_slice_cost_microusd,
            "hourly_slice_cost_microusd",
            minimum=0,
        )
        require_integer(self.queue_priority, "queue_priority", minimum=0)

    @property
    def region(self) -> str:
        return _region(self.zone)


@dataclass(frozen=True, slots=True)
class BulkLocality:
    """A sealed content-addressed index and the payload bytes it unlocks."""

    sealed_index: ContentRefV1
    payload_bytes: int

    def __post_init__(self) -> None:
        if type(self.sealed_index) is not ContentRefV1:
            raise ProtocolError("sealed_index must be ContentRefV1")
        require_integer(self.payload_bytes, "payload_bytes", minimum=1)
        digest = self.sealed_index.sha256
        is_json = self.sealed_index.media_type == "application/json"
        stem, suffix = ("records", "json") if is_json else ("objects", "bin")
        ending = f"/{stem}/sha256/{digest[:2]}/{digest}.{suffix}"
        regions = tuple(replica.region for replica in self.sealed_index.replicas)
        if len(regions) != len(set(regions)):
            raise ProtocolError("sealed index has two replicas in one region")
        if any(
            not replica.uri.endswith(ending)
            for replica in self.sealed_index.replicas
        ):
            raise ProtocolError("sealed index is not content addressed")


@dataclass(frozen=True, slots=True)
class ScheduledClaim:
    """The winning zonal claim plus exact generation-pinned local inputs."""

    campaign_id: str
    work_id: str
    claimant_id: str
    source_fence: int
    attempt_epoch: int
    attempt_id: str
    candidate: V6E64Candidate
    local_inputs: tuple[ContentRefV1, ...]
    local_payload_bytes: int
    created_at_unix_ms: int

class RegionScheduler:
    """Deterministically rank locality, then CAS one small claim head."""

    def __init__(
        self,
        store: JSONCASStore,
        *,
        base_uri: str,
        campaign_id: str,
        clock_ms: Callable[[], int],
        max_cas_retries: int = MAX_CAS_RETRIES,
    ) -> None:
        methods = ("read", "create", "compare_and_swap")
        if any(not callable(getattr(store, name, None)) for name in methods):
            raise TypeError("store lacks the JSON CAS interface")
        base = require_string(base_uri, "base_uri").rstrip("/")
        if not base.startswith("gs://") or base == "gs://":
            raise ValueError("base_uri must be a gs:// prefix")
        self._store = store
        self._campaign = _identifier(campaign_id, "campaign_id")
        self._root = f"{base}/campaigns/{self._campaign}/region-schedules"
        self._clock = clock_ms
        self._retries = require_integer(
            max_cas_retries, "max_cas_retries", minimum=1, maximum=MAX_CAS_RETRIES
        )

    def _head_uri(self, work_id: str) -> str:
        digest = hashlib.sha256(work_id.encode()).hexdigest()
        return f"{self._root}/{digest[:2]}/{digest}/head.json"

    @staticmethod
    def _select(
        candidates: tuple[V6E64Candidate, ...],
        inputs: tuple[BulkLocality, ...],
        excluded: frozenset[str],
    ) -> tuple[V6E64Candidate, tuple[ContentRefV1, ...], int]:
        rank = lambda item: (  # noqa: E731
            item.hourly_slice_cost_microusd,
            item.queue_priority,
            item.zone,
        )
        for candidate in sorted(candidates, key=rank):
            if candidate.zone in excluded:
                continue
            local = []
            for item in inputs:
                replicas = tuple(
                    replica
                    for replica in item.sealed_index.replicas
                    if replica.region == candidate.region
                )
                if len(replicas) != 1:
                    break
                local.append(
                    ContentRefV1(
                        item.sealed_index.sha256,
                        item.sealed_index.size_bytes,
                        item.sealed_index.media_type,
                        replicas,
                    )
                )
            else:
                return (
                    candidate,
                    tuple(sorted(local, key=lambda item: item.sha256)),
                    sum(item.payload_bytes for item in inputs),
                )
        raise NoLocalCandidate("no candidate has every sealed bulk input locally")

    def claim(
        self,
        *,
        work_id: str,
        claimant_id: str,
        source_fence: int,
        candidates: Sequence[V6E64Candidate],
        bulk_inputs: Sequence[BulkLocality],
        excluded_zones: Sequence[str] = (),
    ) -> ScheduledClaim:
        """Claim one epoch; each contender must supply a unique executor ID."""

        work = _identifier(work_id, "work_id")
        claimant = _identifier(claimant_id, "claimant_id")
        fence = require_integer(source_fence, "source_fence", minimum=0)
        offers = _records(candidates, "candidates", V6E64Candidate, MAX_CANDIDATES)
        inputs = _records(bulk_inputs, "bulk_inputs", BulkLocality, MAX_BULK_INPUTS)
        if sum(len(item.sealed_index.replicas) for item in inputs) > MAX_TOTAL_REPLICAS:
            raise RegionSchedulingError("too many regional replicas")
        if not isinstance(excluded_zones, (list, tuple)):
            raise RegionSchedulingError("excluded_zones must be bounded")
        excluded = frozenset(_zone(zone) for zone in excluded_zones)
        if len(excluded) != len(excluded_zones) or len(excluded) > MAX_CANDIDATES:
            raise RegionSchedulingError("excluded_zones is duplicated or oversized")
        if len({item.zone for item in offers}) != len(offers):
            raise RegionSchedulingError("candidate zones must be unique")
        candidate, local, payload_bytes = self._select(offers, inputs, excluded)
        epoch = fence + 1
        request = {
            "campaign_id": self._campaign,
            "work_id": work,
            "claimant_id": claimant,
            "attempt_epoch": epoch,
            "candidate": {
                "zone": candidate.zone,
                "cost": candidate.hourly_slice_cost_microusd,
                "priority": candidate.queue_priority,
            },
            "local_inputs": [item.as_dict() for item in local],
            "local_payload_bytes": payload_bytes,
        }
        request_sha = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
        attempt_id = f"r1e{epoch}-{request_sha[:32]}"
        head_uri = self._head_uri(work)

        for _ in range(self._retries):
            version = self._store.read(head_uri)
            if version is not None:
                head = require_record_object(
                    version.decoded(),
                    schema=_HEAD_SCHEMA,
                    fields=(
                        "campaign_id",
                        "work_id",
                        "source_fence",
                        "attempt_epoch",
                        "attempt_id",
                        "request_sha256",
                        "created_at_unix_ms",
                    ),
                )
                if head["campaign_id"] != self._campaign or head["work_id"] != work:
                    raise RegionSchedulingError("claim head identity changed")
                old_source = require_integer(
                    head["source_fence"], "source_fence", minimum=0
                )
                old_epoch = require_integer(
                    head["attempt_epoch"], "attempt_epoch", minimum=1
                )
                if old_epoch != old_source + 1:
                    raise RegionSchedulingError("claim head epoch is corrupt")
                if old_source == fence:
                    if (
                        head["request_sha256"] != request_sha
                        or head["attempt_id"] != attempt_id
                    ):
                        raise ClaimConflict("another claimant owns this source fence")
                    return ScheduledClaim(
                        self._campaign,
                        work,
                        claimant,
                        fence,
                        epoch,
                        attempt_id,
                        candidate,
                        local,
                        payload_bytes,
                        head["created_at_unix_ms"],
                    )
                if old_epoch != fence:
                    raise ClaimConflict("source fence does not continue claim epoch")

            created = require_integer(self._clock(), "clock_ms", minimum=0)
            claim = ScheduledClaim(
                self._campaign,
                work,
                claimant,
                fence,
                epoch,
                attempt_id,
                candidate,
                local,
                payload_bytes,
                created,
            )
            head = {
                "schema": _HEAD_SCHEMA,
                "campaign_id": self._campaign,
                "work_id": work,
                "source_fence": fence,
                "attempt_epoch": epoch,
                "attempt_id": attempt_id,
                "request_sha256": request_sha,
                "created_at_unix_ms": created,
            }
            if len(canonical_json_bytes(head)) > MAX_HEAD_BYTES:
                raise RegionSchedulingError("claim head exceeds 2 KiB")
            try:
                if version is None:
                    self._store.create(head_uri, head)
                else:
                    self._store.compare_and_swap(head_uri, version.generation, head)
            except CASConflict:
                continue
            return claim
        raise ClaimConflict(f"claim CAS failed after {self._retries} attempts")


def require_attempt_binding(
    claim: ScheduledClaim, authority: AttemptAuthorityV2
) -> None:
    """Bind the schedule to the existing fenced controller/deployment chain."""

    if type(claim) is not ScheduledClaim or type(authority) is not AttemptAuthorityV2:
        raise RegionSchedulingError("claim or authority has the wrong type")
    lease, deployment = authority.lease, authority.deployment
    if (
        lease.campaign_id != claim.campaign_id
        or lease.work_id != claim.work_id
        or lease.attempt_id != claim.attempt_id
        or lease.executor_id != claim.claimant_id
        or lease.fence != claim.attempt_epoch
        or claim.attempt_epoch != claim.source_fence + 1
        or deployment.slice_attempt_id != claim.attempt_id
        or deployment.zone != claim.candidate.zone
        or deployment.region != claim.candidate.region
        or any(
            len(item.replicas) != 1
            or item.replicas[0].region != claim.candidate.region
            for item in claim.local_inputs
        )
    ):
        raise RegionSchedulingError("controller authority does not bind claim")
