"""Fenced, object-store-native controller for independently preemptible work.

The mutable root of a campaign is one generation-matched head object.  Every
accepted head names an immutable inventory, immutable state snapshot, and the
tip of an immutable event chain.  No recovery operation depends on listing an
object-store prefix.

For every transition the controller publishes the candidate event and
snapshot *before* attempting the head CAS.  A race loser can consequently
leave content-addressed objects behind, but they are unreachable from the
accepted head and have no authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import secrets
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from .control_store import CASConflict, JSONCASStore, VersionedJSON
from .protocol import (
    AttemptLeaseV1,
    AttemptReceiptV1,
    CanonicalRecord,
    ContentRefV1,
    HarvestRangeV1,
    ProgressV1,
    ProtocolError,
    SAEJobV1,
    canonical_sha256,
)
from .mfu_v2 import PrefixMFURunContractV2

if TYPE_CHECKING:
    from .mfu_authority import (
        ControllerMFUEvidenceRegistryV2,
        TrustedMFUAuthorityPolicyV2,
    )


_HEAD_SCHEMA = "moe-sae.r1.campaign-head.v1"
_INVENTORY_SCHEMA = "moe-sae.r1.campaign-inventory.v1"
_SNAPSHOT_SCHEMA = "moe-sae.r1.campaign-snapshot.v1"
_EVENT_SCHEMA = "moe-sae.r1.campaign-event.v1"
_MFU_ADMISSION_SCHEMA = "moe-sae.r1.mfu-admission.v1"
_MFU_ADMISSION_PROOF_SCHEMA = "moe-sae.r1.mfu-admission-proof.v1"
_MFU_ADMISSION_EVENT_SCHEMA = "moe-sae.r1.mfu-admission-event.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]*")


class ControllerError(RuntimeError):
    """Base class for campaign-controller errors."""


class CampaignNotInitialized(ControllerError):
    """The campaign head does not exist."""


class CampaignConflict(ControllerError):
    """A campaign name already refers to a different immutable inventory."""


class CorruptCampaign(ControllerError):
    """An accepted object is missing, malformed, or hash-inconsistent."""


class WorkNotFound(ControllerError):
    """The requested work ID is not in the immutable inventory."""


class LeaseUnavailable(ControllerError):
    """Work cannot presently be leased."""


class StaleAttempt(ControllerError):
    """An attempt no longer holds the current fence."""


class InvalidProgress(ControllerError):
    """Progress is non-monotonic or inconsistent with the current attempt."""


class InvalidReceipt(ControllerError):
    """A completion receipt is not admissible for the current work state."""


class ContentVerificationFailed(ControllerError):
    """An output could not be independently verified."""


class RetryLimitExceeded(ControllerError):
    """The bounded number of head-CAS attempts was exhausted."""


@dataclass(frozen=True, slots=True)
class _ObjectRef:
    uri: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri:
            raise CorruptCampaign("object reference URI is invalid")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise CorruptCampaign("object reference SHA-256 is invalid")

    def as_dict(self) -> dict[str, str]:
        return {"uri": self.uri, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: Any, name: str) -> _ObjectRef:
        obj = _exact_object(value, name, ("uri", "sha256"))
        return cls(
            uri=_text(obj["uri"], f"{name}.uri"), sha256=_sha(obj["sha256"], name)
        )


@dataclass(frozen=True, slots=True, order=True)
class _LogicalContent:
    sha256: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        _sha(self.sha256, "logical content")
        _integer(self.size_bytes, "logical content size", minimum=1)
        _text(self.media_type, "logical content media type")

    @classmethod
    def from_content(cls, value: ContentRefV1) -> _LogicalContent:
        return cls(value.sha256, value.size_bytes, value.media_type)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
        }

    @classmethod
    def from_dict(cls, value: Any) -> _LogicalContent:
        obj = _exact_object(
            value,
            "logical content",
            ("sha256", "size_bytes", "media_type"),
        )
        return cls(
            sha256=_sha(obj["sha256"], "logical content"),
            size_bytes=_integer(obj["size_bytes"], "size_bytes", minimum=1),
            media_type=_text(obj["media_type"], "media_type"),
        )


WorkItem = HarvestRangeV1 | SAEJobV1


@dataclass(frozen=True, slots=True)
class _InventoryEntry:
    kind: str
    work_id: str
    scientific_identity_sha256: str
    specification: WorkItem
    expected_outputs: tuple[_LogicalContent, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"HARVEST", "SAE"}:
            raise CorruptCampaign("inventory work kind is invalid")
        _identifier(self.work_id, "work_id")
        _sha(self.scientific_identity_sha256, "scientific identity")
        expected = tuple(sorted(self.expected_outputs))
        if len({item.sha256 for item in expected}) != len(expected):
            raise CorruptCampaign("expected outputs contain duplicate SHA-256")
        object.__setattr__(self, "expected_outputs", expected)

    @classmethod
    def build(
        cls,
        work: WorkItem,
        expected_outputs: Sequence[ContentRefV1],
    ) -> _InventoryEntry:
        if isinstance(work, HarvestRangeV1):
            kind = "HARVEST"
        elif isinstance(work, SAEJobV1):
            kind = "SAE"
        else:  # pragma: no cover - guarded by public initialize validation
            raise TypeError("unsupported work item")
        logical = tuple(_LogicalContent.from_content(item) for item in expected_outputs)
        return cls(
            kind=kind,
            work_id=work.work_id,
            scientific_identity_sha256=work.scientific_identity_sha256,
            specification=work,
            expected_outputs=logical,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "work_id": self.work_id,
            "scientific_identity_sha256": self.scientific_identity_sha256,
            "specification": self.specification.as_dict(),
            "expected_outputs": [item.as_dict() for item in self.expected_outputs],
        }

    @classmethod
    def from_dict(cls, value: Any) -> _InventoryEntry:
        obj = _exact_object(
            value,
            "inventory entry",
            (
                "kind",
                "work_id",
                "scientific_identity_sha256",
                "specification",
                "expected_outputs",
            ),
        )
        kind = _text(obj["kind"], "kind")
        spec_value = obj["specification"]
        try:
            if kind == "HARVEST":
                specification: WorkItem = HarvestRangeV1.from_dict(spec_value)
            elif kind == "SAE":
                specification = SAEJobV1.from_dict(spec_value)
            else:
                raise CorruptCampaign("inventory work kind is invalid")
        except ProtocolError as error:
            raise CorruptCampaign("inventory work specification is invalid") from error
        raw_expected = obj["expected_outputs"]
        if not isinstance(raw_expected, list):
            raise CorruptCampaign("expected_outputs must be an array")
        entry = cls(
            kind=kind,
            work_id=_identifier(obj["work_id"], "work_id"),
            scientific_identity_sha256=_sha(
                obj["scientific_identity_sha256"], "scientific identity"
            ),
            specification=specification,
            expected_outputs=tuple(
                _LogicalContent.from_dict(item) for item in raw_expected
            ),
        )
        if entry.work_id != specification.work_id:
            raise CorruptCampaign("inventory work ID does not match specification")
        if entry.scientific_identity_sha256 != specification.scientific_identity_sha256:
            raise CorruptCampaign(
                "inventory scientific identity does not match specification"
            )
        return entry


@dataclass(frozen=True, slots=True)
class _Inventory:
    campaign_id: str
    scientific_identity_sha256: str
    entries: tuple[_InventoryEntry, ...]

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, "campaign_id")
        _sha(self.scientific_identity_sha256, "scientific identity")
        entries = tuple(sorted(self.entries, key=lambda item: item.work_id))
        if not entries:
            raise CampaignConflict("campaign inventory must not be empty")
        if len({item.work_id for item in entries}) != len(entries):
            raise CampaignConflict("campaign work IDs must be unique")
        if any(
            item.scientific_identity_sha256 != self.scientific_identity_sha256
            for item in entries
        ):
            raise CampaignConflict("all work must share one scientific identity")
        object.__setattr__(self, "entries", entries)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _INVENTORY_SCHEMA,
            "campaign_id": self.campaign_id,
            "scientific_identity_sha256": self.scientific_identity_sha256,
            "works": [item.as_dict() for item in self.entries],
        }

    @classmethod
    def from_dict(cls, value: Any) -> _Inventory:
        obj = _versioned_object(
            value,
            _INVENTORY_SCHEMA,
            ("campaign_id", "scientific_identity_sha256", "works"),
        )
        raw_works = obj["works"]
        if not isinstance(raw_works, list):
            raise CorruptCampaign("inventory works must be an array")
        return cls(
            campaign_id=_identifier(obj["campaign_id"], "campaign_id"),
            scientific_identity_sha256=_sha(
                obj["scientific_identity_sha256"], "scientific identity"
            ),
            entries=tuple(_InventoryEntry.from_dict(item) for item in raw_works),
        )

    def entry(self, work_id: str) -> _InventoryEntry:
        for item in self.entries:
            if item.work_id == work_id:
                return item
        raise WorkNotFound(work_id)


@dataclass(frozen=True, slots=True)
class _WorkState:
    work_id: str
    status: str
    fence: int
    last_attempt_id: str | None
    lease: AttemptLeaseV1 | None
    progress: ProgressV1 | None
    receipt: AttemptReceiptV1 | None
    last_progress_sequence: int | None
    completed_units: int | None
    total_units: int | None

    def __post_init__(self) -> None:
        _identifier(self.work_id, "work_id")
        if self.status not in {"PENDING", "LEASED", "COMMITTED"}:
            raise CorruptCampaign("work status is invalid")
        _integer(self.fence, "fence", minimum=0)
        if self.last_attempt_id is not None:
            _identifier(self.last_attempt_id, "last_attempt_id")
        for record, name in (
            (self.lease, "lease"),
            (self.progress, "progress"),
            (self.receipt, "receipt"),
        ):
            if record is not None and record.work_id != self.work_id:
                raise CorruptCampaign(f"{name} belongs to another work item")
        if self.status == "LEASED":
            if self.lease is None or self.receipt is not None:
                raise CorruptCampaign("leased work must have only an active lease")
            if self.last_attempt_id != self.lease.attempt_id:
                raise CorruptCampaign("active lease attempt does not match work state")
            if self.fence != self.lease.fence:
                raise CorruptCampaign("active lease fence does not match work state")
        elif self.lease is not None:
            raise CorruptCampaign("non-leased work cannot have an active lease")
        if self.progress is not None:
            if self.progress.attempt_id != self.last_attempt_id:
                raise CorruptCampaign("progress attempt does not match work state")
            if self.progress.fence != self.fence:
                raise CorruptCampaign("progress fence does not match work state")
        counters = (
            self.last_progress_sequence,
            self.completed_units,
            self.total_units,
        )
        if any(item is None for item in counters) != all(
            item is None for item in counters
        ):
            raise CorruptCampaign("progress counters must be all present or all absent")
        if self.last_progress_sequence is not None:
            _integer(self.last_progress_sequence, "last_progress_sequence", minimum=0)
            assert self.completed_units is not None and self.total_units is not None
            _integer(self.completed_units, "completed_units", minimum=0)
            _integer(self.total_units, "total_units", minimum=1)
            if self.completed_units > self.total_units:
                raise CorruptCampaign("completed_units exceeds total_units")
        if self.progress is not None and (
            self.progress.sequence != self.last_progress_sequence
            or self.progress.completed_units != self.completed_units
            or self.progress.total_units != self.total_units
        ):
            raise CorruptCampaign("current progress disagrees with global counters")
        if self.status == "COMMITTED":
            if self.receipt is None:
                raise CorruptCampaign("committed work requires a receipt")
            if self.receipt.attempt_id != self.last_attempt_id:
                raise CorruptCampaign("receipt attempt does not match work state")
            if self.receipt.fence != self.fence:
                raise CorruptCampaign("receipt fence does not match work state")
        elif self.receipt is not None:
            raise CorruptCampaign("uncommitted work cannot have a receipt")

    @classmethod
    def pending(cls, work_id: str) -> _WorkState:
        return cls(work_id, "PENDING", 0, None, None, None, None, None, None, None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "status": self.status,
            "fence": self.fence,
            "last_attempt_id": self.last_attempt_id,
            "lease": None if self.lease is None else self.lease.as_dict(),
            "progress": None if self.progress is None else self.progress.as_dict(),
            "receipt": None if self.receipt is None else self.receipt.as_dict(),
            "last_progress_sequence": self.last_progress_sequence,
            "completed_units": self.completed_units,
            "total_units": self.total_units,
        }

    @classmethod
    def from_dict(cls, value: Any) -> _WorkState:
        obj = _exact_object(
            value,
            "work state",
            (
                "work_id",
                "status",
                "fence",
                "last_attempt_id",
                "lease",
                "progress",
                "receipt",
                "last_progress_sequence",
                "completed_units",
                "total_units",
            ),
        )
        try:
            lease = (
                None if obj["lease"] is None else AttemptLeaseV1.from_dict(obj["lease"])
            )
            progress = (
                None
                if obj["progress"] is None
                else ProgressV1.from_dict(obj["progress"])
            )
            receipt = (
                None
                if obj["receipt"] is None
                else AttemptReceiptV1.from_dict(obj["receipt"])
            )
        except ProtocolError as error:
            raise CorruptCampaign("work state contains an invalid record") from error
        last_attempt = obj["last_attempt_id"]
        if last_attempt is not None:
            last_attempt = _identifier(last_attempt, "last_attempt_id")
        last_progress_sequence = obj["last_progress_sequence"]
        completed_units = obj["completed_units"]
        total_units = obj["total_units"]
        if last_progress_sequence is not None:
            last_progress_sequence = _integer(
                last_progress_sequence, "last_progress_sequence", minimum=0
            )
        if completed_units is not None:
            completed_units = _integer(completed_units, "completed_units", minimum=0)
        if total_units is not None:
            total_units = _integer(total_units, "total_units", minimum=1)
        return cls(
            work_id=_identifier(obj["work_id"], "work_id"),
            status=_text(obj["status"], "status"),
            fence=_integer(obj["fence"], "fence", minimum=0),
            last_attempt_id=last_attempt,
            lease=lease,
            progress=progress,
            receipt=receipt,
            last_progress_sequence=last_progress_sequence,
            completed_units=completed_units,
            total_units=total_units,
        )


@dataclass(frozen=True, slots=True)
class _Snapshot:
    campaign_id: str
    scientific_identity_sha256: str
    inventory_sha256: str
    revision: int
    works: tuple[_WorkState, ...]

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, "campaign_id")
        _sha(self.scientific_identity_sha256, "scientific identity")
        _sha(self.inventory_sha256, "inventory")
        _integer(self.revision, "revision", minimum=0)
        works = tuple(sorted(self.works, key=lambda item: item.work_id))
        if not works or len({item.work_id for item in works}) != len(works):
            raise CorruptCampaign("snapshot work inventory is empty or duplicated")
        object.__setattr__(self, "works", works)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _SNAPSHOT_SCHEMA,
            "campaign_id": self.campaign_id,
            "scientific_identity_sha256": self.scientific_identity_sha256,
            "inventory_sha256": self.inventory_sha256,
            "revision": self.revision,
            "works": [item.as_dict() for item in self.works],
        }

    @classmethod
    def from_dict(cls, value: Any) -> _Snapshot:
        obj = _versioned_object(
            value,
            _SNAPSHOT_SCHEMA,
            (
                "campaign_id",
                "scientific_identity_sha256",
                "inventory_sha256",
                "revision",
                "works",
            ),
        )
        raw_works = obj["works"]
        if not isinstance(raw_works, list):
            raise CorruptCampaign("snapshot works must be an array")
        return cls(
            campaign_id=_identifier(obj["campaign_id"], "campaign_id"),
            scientific_identity_sha256=_sha(
                obj["scientific_identity_sha256"], "scientific identity"
            ),
            inventory_sha256=_sha(obj["inventory_sha256"], "inventory"),
            revision=_integer(obj["revision"], "revision", minimum=0),
            works=tuple(_WorkState.from_dict(item) for item in raw_works),
        )

    def work(self, work_id: str) -> _WorkState:
        for item in self.works:
            if item.work_id == work_id:
                return item
        raise WorkNotFound(work_id)

    def replacing(self, state: _WorkState) -> _Snapshot:
        if state.work_id not in {item.work_id for item in self.works}:
            raise WorkNotFound(state.work_id)
        return _Snapshot(
            campaign_id=self.campaign_id,
            scientific_identity_sha256=self.scientific_identity_sha256,
            inventory_sha256=self.inventory_sha256,
            revision=self.revision + 1,
            works=tuple(
                state if item.work_id == state.work_id else item for item in self.works
            ),
        )


@dataclass(frozen=True, slots=True)
class _Head:
    campaign_id: str
    scientific_identity_sha256: str
    inventory: _ObjectRef
    snapshot: _ObjectRef
    last_event: _ObjectRef
    revision: int
    event_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _HEAD_SCHEMA,
            "campaign_id": self.campaign_id,
            "scientific_identity_sha256": self.scientific_identity_sha256,
            "inventory": self.inventory.as_dict(),
            "snapshot": self.snapshot.as_dict(),
            "last_event": self.last_event.as_dict(),
            "revision": self.revision,
            "event_count": self.event_count,
        }

    @classmethod
    def from_dict(cls, value: Any) -> _Head:
        obj = _versioned_object(
            value,
            _HEAD_SCHEMA,
            (
                "campaign_id",
                "scientific_identity_sha256",
                "inventory",
                "snapshot",
                "last_event",
                "revision",
                "event_count",
            ),
        )
        result = cls(
            campaign_id=_identifier(obj["campaign_id"], "campaign_id"),
            scientific_identity_sha256=_sha(
                obj["scientific_identity_sha256"], "scientific identity"
            ),
            inventory=_ObjectRef.from_dict(obj["inventory"], "inventory"),
            snapshot=_ObjectRef.from_dict(obj["snapshot"], "snapshot"),
            last_event=_ObjectRef.from_dict(obj["last_event"], "last_event"),
            revision=_integer(obj["revision"], "revision", minimum=0),
            event_count=_integer(obj["event_count"], "event_count", minimum=1),
        )
        return result


@dataclass(frozen=True, slots=True)
class MFUAdmissionV1(CanonicalRecord):
    """Controller-owned admission of one exact MFU run under a live lease.

    The complete run contract is embedded rather than merely named.  This
    makes the accepted event self-contained while its canonical digest still
    provides a compact identity for downstream evidence.  The lease interval
    is immutable: renewing the same attempt/fence never extends this admission.

    ``controller_challenge`` is presently binding/correlation material only.
    No trusted measurement channel echoes or authenticates it yet, so neither
    this record nor :class:`MFUAdmissionProofV1` can authorize production PASS.
    Closing that challenge-response loop remains a future authority blocker.
    """

    SCHEMA = _MFU_ADMISSION_SCHEMA

    campaign_id: str
    work_id: str
    attempt_id: str
    executor_id: str
    fence: int
    lease_acquired_at_unix_ms: int
    lease_expires_at_unix_ms: int
    controller_observed_at_unix_ms: int
    scientific_identity_sha256: str
    deployment_identity_sha256: str
    run_contract: PrefixMFURunContractV2
    controller_challenge: str

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, "campaign_id")
        _identifier(self.work_id, "work_id")
        _identifier(self.attempt_id, "attempt_id")
        _identifier(self.executor_id, "executor_id")
        _integer(self.fence, "fence", minimum=1)
        acquired = _integer(
            self.lease_acquired_at_unix_ms,
            "lease_acquired_at_unix_ms",
            minimum=0,
        )
        expires = _integer(
            self.lease_expires_at_unix_ms,
            "lease_expires_at_unix_ms",
            minimum=1,
        )
        observed = _integer(
            self.controller_observed_at_unix_ms,
            "controller_observed_at_unix_ms",
            minimum=0,
        )
        if not acquired <= observed < expires:
            raise CorruptCampaign(
                "MFU admission was not observed inside the accepted lease interval"
            )
        _sha(self.scientific_identity_sha256, "scientific identity")
        _sha(self.deployment_identity_sha256, "deployment identity")
        _sha(self.controller_challenge, "controller challenge")
        if not isinstance(self.run_contract, PrefixMFURunContractV2):
            raise CorruptCampaign(
                "MFU admission run contract has the wrong record type"
            )
        if self.run_contract.attempt_id != self.attempt_id:
            raise CorruptCampaign("MFU admission attempt differs from its run contract")
        if (
            self.run_contract.scientific_identity.sha256
            != self.scientific_identity_sha256
        ):
            raise CorruptCampaign(
                "MFU admission scientific identity differs from its run contract"
            )
        if (
            self.run_contract.deployment_identity.sha256
            != self.deployment_identity_sha256
        ):
            raise CorruptCampaign(
                "MFU admission deployment identity differs from its run contract"
            )

    @property
    def run_contract_sha256(self) -> str:
        return self.run_contract.digest

    @property
    def window_plan_sha256(self) -> str:
        return self.run_contract.window_plan.digest

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "campaign_id": self.campaign_id,
            "work_id": self.work_id,
            "attempt_id": self.attempt_id,
            "executor_id": self.executor_id,
            "fence": self.fence,
            "lease_acquired_at_unix_ms": self.lease_acquired_at_unix_ms,
            "lease_expires_at_unix_ms": self.lease_expires_at_unix_ms,
            "controller_observed_at_unix_ms": self.controller_observed_at_unix_ms,
            "scientific_identity_sha256": self.scientific_identity_sha256,
            "deployment_identity_sha256": self.deployment_identity_sha256,
            "run_contract": self.run_contract.as_dict(),
            "controller_challenge": self.controller_challenge,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MFUAdmissionV1:
        obj = _versioned_object(
            value,
            cls.SCHEMA,
            (
                "campaign_id",
                "work_id",
                "attempt_id",
                "executor_id",
                "fence",
                "lease_acquired_at_unix_ms",
                "lease_expires_at_unix_ms",
                "controller_observed_at_unix_ms",
                "scientific_identity_sha256",
                "deployment_identity_sha256",
                "run_contract",
                "controller_challenge",
            ),
        )
        try:
            run_contract = PrefixMFURunContractV2.from_dict(obj["run_contract"])
        except (ProtocolError, TypeError, ValueError) as error:
            raise CorruptCampaign(
                "MFU admission contains an invalid run contract"
            ) from error
        return cls(
            campaign_id=_identifier(obj["campaign_id"], "campaign_id"),
            work_id=_identifier(obj["work_id"], "work_id"),
            attempt_id=_identifier(obj["attempt_id"], "attempt_id"),
            executor_id=_identifier(obj["executor_id"], "executor_id"),
            fence=_integer(obj["fence"], "fence", minimum=1),
            lease_acquired_at_unix_ms=_integer(
                obj["lease_acquired_at_unix_ms"],
                "lease_acquired_at_unix_ms",
                minimum=0,
            ),
            lease_expires_at_unix_ms=_integer(
                obj["lease_expires_at_unix_ms"],
                "lease_expires_at_unix_ms",
                minimum=1,
            ),
            controller_observed_at_unix_ms=_integer(
                obj["controller_observed_at_unix_ms"],
                "controller_observed_at_unix_ms",
                minimum=0,
            ),
            scientific_identity_sha256=_sha(
                obj["scientific_identity_sha256"], "scientific identity"
            ),
            deployment_identity_sha256=_sha(
                obj["deployment_identity_sha256"], "deployment identity"
            ),
            run_contract=run_contract,
            controller_challenge=_sha(
                obj["controller_challenge"], "controller challenge"
            ),
        )


@dataclass(frozen=True, slots=True)
class MFUAdmissionProofV1(CanonicalRecord):
    """Portable handle whose authority comes only from accepted-chain replay.

    This proves controller admission, not measurement provenance.  A future
    closed authority must also require trusted measurement evidence to echo
    the admitted challenge before this proof can participate in PASS.
    """

    SCHEMA = _MFU_ADMISSION_PROOF_SCHEMA

    campaign_id: str
    event_uri: str
    event_sha256: str
    event_revision: int
    admission: MFUAdmissionV1

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, "campaign_id")
        _text(self.event_uri, "event_uri")
        _sha(self.event_sha256, "event")
        _integer(self.event_revision, "event_revision", minimum=1)
        if not isinstance(self.admission, MFUAdmissionV1):
            raise CorruptCampaign("MFU admission proof has the wrong admission type")
        if self.admission.campaign_id != self.campaign_id:
            raise CorruptCampaign("MFU admission proof crosses campaign identities")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "campaign_id": self.campaign_id,
            "event_uri": self.event_uri,
            "event_sha256": self.event_sha256,
            "event_revision": self.event_revision,
            "admission": self.admission.as_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MFUAdmissionProofV1:
        obj = _versioned_object(
            value,
            cls.SCHEMA,
            (
                "campaign_id",
                "event_uri",
                "event_sha256",
                "event_revision",
                "admission",
            ),
        )
        return cls(
            campaign_id=_identifier(obj["campaign_id"], "campaign_id"),
            event_uri=_text(obj["event_uri"], "event_uri"),
            event_sha256=_sha(obj["event_sha256"], "event"),
            event_revision=_integer(obj["event_revision"], "event_revision", minimum=1),
            admission=MFUAdmissionV1.from_dict(obj["admission"]),
        )


@dataclass(frozen=True, slots=True)
class _Event:
    campaign_id: str
    kind: str
    revision: int
    occurred_at_unix_ms: int
    work_id: str | None
    previous_event: _ObjectRef | None
    before_snapshot: _ObjectRef | None
    after_snapshot: _ObjectRef
    before_work: _WorkState | None
    after_work: _WorkState | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _EVENT_SCHEMA,
            "campaign_id": self.campaign_id,
            "kind": self.kind,
            "revision": self.revision,
            "occurred_at_unix_ms": self.occurred_at_unix_ms,
            "work_id": self.work_id,
            "previous_event": (
                None if self.previous_event is None else self.previous_event.as_dict()
            ),
            "before_snapshot": (
                None if self.before_snapshot is None else self.before_snapshot.as_dict()
            ),
            "after_snapshot": self.after_snapshot.as_dict(),
            "before_work": None
            if self.before_work is None
            else self.before_work.as_dict(),
            "after_work": None
            if self.after_work is None
            else self.after_work.as_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> _Event:
        obj = _versioned_object(
            value,
            _EVENT_SCHEMA,
            (
                "campaign_id",
                "kind",
                "revision",
                "occurred_at_unix_ms",
                "work_id",
                "previous_event",
                "before_snapshot",
                "after_snapshot",
                "before_work",
                "after_work",
            ),
        )
        work_id = obj["work_id"]
        if work_id is not None:
            work_id = _identifier(work_id, "work_id")
        event = cls(
            campaign_id=_identifier(obj["campaign_id"], "campaign_id"),
            kind=_identifier(obj["kind"], "event kind"),
            revision=_integer(obj["revision"], "revision", minimum=0),
            occurred_at_unix_ms=_integer(
                obj["occurred_at_unix_ms"], "event timestamp", minimum=0
            ),
            work_id=work_id,
            previous_event=(
                None
                if obj["previous_event"] is None
                else _ObjectRef.from_dict(obj["previous_event"], "previous_event")
            ),
            before_snapshot=(
                None
                if obj["before_snapshot"] is None
                else _ObjectRef.from_dict(obj["before_snapshot"], "before_snapshot")
            ),
            after_snapshot=_ObjectRef.from_dict(
                obj["after_snapshot"], "after_snapshot"
            ),
            before_work=(
                None
                if obj["before_work"] is None
                else _WorkState.from_dict(obj["before_work"])
            ),
            after_work=(
                None
                if obj["after_work"] is None
                else _WorkState.from_dict(obj["after_work"])
            ),
        )
        if event.kind == "CAMPAIGN_INITIALIZED":
            if any(
                item is not None
                for item in (
                    event.work_id,
                    event.previous_event,
                    event.before_snapshot,
                    event.before_work,
                    event.after_work,
                )
            ):
                raise CorruptCampaign("initialization event has transition fields")
            if event.revision != 0:
                raise CorruptCampaign("initialization event revision must be zero")
        elif event.kind == "MFU_ADMITTED":
            raise CorruptCampaign(
                "MFU admission must use the dedicated admission-event schema"
            )
        elif (
            event.work_id is None
            or event.previous_event is None
            or event.before_snapshot is None
            or event.before_work is None
            or event.after_work is None
        ):
            raise CorruptCampaign("transition event is missing required fields")
        return event


@dataclass(frozen=True, slots=True)
class _MFUAdmissionEvent:
    campaign_id: str
    kind: str
    revision: int
    occurred_at_unix_ms: int
    work_id: str
    previous_event: _ObjectRef
    before_snapshot: _ObjectRef
    after_snapshot: _ObjectRef
    before_work: _WorkState
    after_work: _WorkState
    admission: MFUAdmissionV1

    def __post_init__(self) -> None:
        if self.kind != "MFU_ADMITTED":
            raise CorruptCampaign("MFU admission event has an invalid kind")
        _identifier(self.campaign_id, "campaign_id")
        _integer(self.revision, "revision", minimum=1)
        _integer(self.occurred_at_unix_ms, "event timestamp", minimum=0)
        _identifier(self.work_id, "work_id")
        if self.before_work != self.after_work:
            raise CorruptCampaign("MFU admission event must preserve leased work state")
        if self.before_work.work_id != self.work_id:
            raise CorruptCampaign("MFU admission event names another work state")
        if self.before_work.status != "LEASED" or self.before_work.lease is None:
            raise CorruptCampaign("MFU admission event lacks an active leased state")
        admission = self.admission
        lease = self.before_work.lease
        if admission.campaign_id != self.campaign_id:
            raise CorruptCampaign("MFU admission event crosses campaign identities")
        if admission.work_id != self.work_id:
            raise CorruptCampaign("MFU admission event crosses work identities")
        if admission.controller_observed_at_unix_ms != self.occurred_at_unix_ms:
            raise CorruptCampaign("MFU admission and event timestamps differ")
        if (
            admission.attempt_id != lease.attempt_id
            or admission.executor_id != lease.executor_id
            or admission.fence != lease.fence
            or admission.lease_acquired_at_unix_ms != lease.acquired_at_unix_ms
            or admission.lease_expires_at_unix_ms != lease.expires_at_unix_ms
            or admission.deployment_identity_sha256 != lease.deployment_identity_sha256
        ):
            raise CorruptCampaign("MFU admission does not exactly bind the event lease")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _MFU_ADMISSION_EVENT_SCHEMA,
            "campaign_id": self.campaign_id,
            "kind": self.kind,
            "revision": self.revision,
            "occurred_at_unix_ms": self.occurred_at_unix_ms,
            "work_id": self.work_id,
            "previous_event": self.previous_event.as_dict(),
            "before_snapshot": self.before_snapshot.as_dict(),
            "after_snapshot": self.after_snapshot.as_dict(),
            "before_work": self.before_work.as_dict(),
            "after_work": self.after_work.as_dict(),
            "admission": self.admission.as_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> _MFUAdmissionEvent:
        obj = _versioned_object(
            value,
            _MFU_ADMISSION_EVENT_SCHEMA,
            (
                "campaign_id",
                "kind",
                "revision",
                "occurred_at_unix_ms",
                "work_id",
                "previous_event",
                "before_snapshot",
                "after_snapshot",
                "before_work",
                "after_work",
                "admission",
            ),
        )
        return cls(
            campaign_id=_identifier(obj["campaign_id"], "campaign_id"),
            kind=_identifier(obj["kind"], "event kind"),
            revision=_integer(obj["revision"], "revision", minimum=1),
            occurred_at_unix_ms=_integer(
                obj["occurred_at_unix_ms"], "event timestamp", minimum=0
            ),
            work_id=_identifier(obj["work_id"], "work_id"),
            previous_event=_ObjectRef.from_dict(
                obj["previous_event"], "previous_event"
            ),
            before_snapshot=_ObjectRef.from_dict(
                obj["before_snapshot"], "before_snapshot"
            ),
            after_snapshot=_ObjectRef.from_dict(
                obj["after_snapshot"], "after_snapshot"
            ),
            before_work=_WorkState.from_dict(obj["before_work"]),
            after_work=_WorkState.from_dict(obj["after_work"]),
            admission=MFUAdmissionV1.from_dict(obj["admission"]),
        )


_CampaignEvent = _Event | _MFUAdmissionEvent


@dataclass(frozen=True, slots=True)
class WorkView:
    work_id: str
    status: str
    fence: int
    last_attempt_id: str | None
    lease: AttemptLeaseV1 | None
    progress: ProgressV1 | None
    receipt: AttemptReceiptV1 | None
    last_progress_sequence: int | None
    completed_units: int | None
    total_units: int | None

    @classmethod
    def from_state(cls, state: _WorkState) -> WorkView:
        return cls(
            state.work_id,
            state.status,
            state.fence,
            state.last_attempt_id,
            state.lease,
            state.progress,
            state.receipt,
            state.last_progress_sequence,
            state.completed_units,
            state.total_units,
        )


@dataclass(frozen=True, slots=True)
class CampaignView:
    campaign_id: str
    scientific_identity_sha256: str
    revision: int
    event_count: int
    inventory_sha256: str
    snapshot_sha256: str
    last_event_sha256: str
    works: tuple[WorkView, ...]

    def work(self, work_id: str) -> WorkView:
        for item in self.works:
            if item.work_id == work_id:
                return item
        raise WorkNotFound(work_id)


ContentVerifier = Callable[[ContentRefV1], bool]
Clock = Callable[[], int]
ChallengeFactory = Callable[[], str]


class CampaignController:
    """Coordinate one immutable campaign through a single CAS head."""

    def __init__(
        self,
        store: JSONCASStore,
        *,
        base_uri: str,
        campaign_id: str,
        clock_ms: Clock | None = None,
        challenge_factory: ChallengeFactory | None = None,
        content_verifier: ContentVerifier | None = None,
        max_cas_retries: int = 8,
    ) -> None:
        if not isinstance(base_uri, str) or not base_uri.rstrip("/"):
            raise ValueError("base_uri must be non-empty")
        self._store = store
        self._base_uri = base_uri.rstrip("/")
        self._campaign_id = _identifier(campaign_id, "campaign_id")
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._challenge_factory = challenge_factory or (lambda: secrets.token_hex(32))
        self._content_verifier = content_verifier or (lambda _content: False)
        self._max_cas_retries = _integer(max_cas_retries, "max_cas_retries", minimum=1)

    @property
    def head_uri(self) -> str:
        return f"{self._base_uri}/campaigns/{self._campaign_id}/head.json"

    def mfu_evidence_registry_v2(
        self,
        *,
        policy: TrustedMFUAuthorityPolicyV2,
        replica_region: str,
    ) -> ControllerMFUEvidenceRegistryV2:
        """Bind an MFU receipt publisher to the currently accepted head.

        The import is local so the V1 HARVEST/SAE controller wire schema stays
        independent of the optional V2 MFU evidence layer.  Each registry is
        frozen to the exact head generation and revision observed here.
        """

        from .mfu_authority import ControllerMFUEvidenceRegistryV2

        head, version = self._load_head()
        return ControllerMFUEvidenceRegistryV2(
            self._store,
            policy=policy,
            campaign_id=self._campaign_id,
            campaign_head_uri=self.head_uri,
            campaign_head_generation=version.generation,
            campaign_head_revision=head.revision,
            replica_region=replica_region,
            clock_ns=lambda: self._now() * 1_000_000,
        )

    def admit_mfu_v2(
        self,
        *,
        work_id: str,
        run_contract: PrefixMFURunContractV2,
    ) -> MFUAdmissionProofV1:
        """Atomically admit one exact V2 MFU contract under the active lease.

        An admission becomes authoritative only when its event wins the same
        generation-matched head transition as the corresponding snapshot
        revision.  Retries use a fresh controller challenge.  A repeated call
        for the same lease and exact contract is idempotent; a second contract
        for that lease is rejected.
        """

        if not isinstance(run_contract, PrefixMFURunContractV2):
            raise InvalidReceipt("MFU run contract has the wrong record type")
        work_id = _identifier(work_id, "work_id")
        candidate_challenges: set[str] = set()
        for _attempt in range(self._max_cas_retries):
            head, version = self._load_head()
            inventory = self._load_inventory(head.inventory)
            snapshot = self._load_snapshot(head.snapshot)
            _view, events = self._replay_accepted(
                head,
                inventory,
                snapshot,
            )
            entry = inventory.entry(work_id)
            before = snapshot.work(work_id)
            now = self._now()
            lease = before.lease
            if before.status != "LEASED" or lease is None:
                raise LeaseUnavailable("MFU admission requires an active lease")
            if run_contract.attempt_id != lease.attempt_id:
                raise StaleAttempt("MFU run attempt does not hold the active lease")
            if (
                run_contract.deployment_identity.sha256
                != lease.deployment_identity_sha256
            ):
                raise InvalidReceipt("MFU deployment does not match the active lease")
            if (
                run_contract.scientific_identity.sha256
                != entry.scientific_identity_sha256
            ):
                raise InvalidReceipt(
                    "MFU scientific identity does not match campaign work"
                )

            existing = self._admission_for_lease(
                events,
                work_id=work_id,
                attempt_id=lease.attempt_id,
                fence=lease.fence,
            )
            if existing is not None:
                existing_ref, existing_event = existing
                admitted = existing_event.admission
                if (
                    admitted.lease_acquired_at_unix_ms != lease.acquired_at_unix_ms
                    or admitted.lease_expires_at_unix_ms != lease.expires_at_unix_ms
                ):
                    raise InvalidReceipt(
                        "active lease interval differs from its accepted MFU "
                        "admission; renewal cannot extend admission authority"
                    )
            if now >= lease.expires_at_unix_ms:
                raise LeaseUnavailable("MFU admission requires an active lease")
            if existing is not None:
                existing_ref, existing_event = existing
                admitted = existing_event.admission
                if (
                    admitted.run_contract.canonical_bytes()
                    != run_contract.canonical_bytes()
                ):
                    raise InvalidReceipt(
                        "active lease already admitted a different MFU run contract"
                    )
                return self._admission_proof(existing_ref, existing_event)

            challenge = _sha(
                self._challenge_factory(),
                "controller challenge",
            )
            accepted_challenges = {
                event.admission.controller_challenge
                for _event_ref, event in events
                if isinstance(event, _MFUAdmissionEvent)
            }
            if challenge in candidate_challenges or challenge in accepted_challenges:
                raise ControllerError("controller challenge source reused an MFU nonce")
            candidate_challenges.add(challenge)
            admission = MFUAdmissionV1(
                campaign_id=self._campaign_id,
                work_id=work_id,
                attempt_id=lease.attempt_id,
                executor_id=lease.executor_id,
                fence=lease.fence,
                lease_acquired_at_unix_ms=lease.acquired_at_unix_ms,
                lease_expires_at_unix_ms=lease.expires_at_unix_ms,
                controller_observed_at_unix_ms=now,
                scientific_identity_sha256=entry.scientific_identity_sha256,
                deployment_identity_sha256=lease.deployment_identity_sha256,
                run_contract=run_contract,
                controller_challenge=challenge,
            )
            next_snapshot = snapshot.replacing(before)
            snapshot_ref = self._ref_for("snapshots", next_snapshot.as_dict())
            event = _MFUAdmissionEvent(
                campaign_id=self._campaign_id,
                kind="MFU_ADMITTED",
                revision=next_snapshot.revision,
                occurred_at_unix_ms=now,
                work_id=work_id,
                previous_event=head.last_event,
                before_snapshot=head.snapshot,
                after_snapshot=snapshot_ref,
                before_work=before,
                after_work=before,
                admission=admission,
            )
            # Required ordering: event first, snapshot second, authority CAS last.
            event_ref = self._publish("events", event.as_dict())
            published_snapshot = self._publish("snapshots", next_snapshot.as_dict())
            if published_snapshot != snapshot_ref:
                raise CorruptCampaign("published admission snapshot identity changed")
            new_head = _Head(
                campaign_id=self._campaign_id,
                scientific_identity_sha256=head.scientific_identity_sha256,
                inventory=head.inventory,
                snapshot=snapshot_ref,
                last_event=event_ref,
                revision=next_snapshot.revision,
                event_count=head.event_count + 1,
            )
            try:
                self._store.compare_and_swap(
                    self.head_uri,
                    version.generation,
                    new_head.as_dict(),
                )
            except CASConflict:
                continue
            return self._admission_proof(event_ref, event)
        raise RetryLimitExceeded(
            "MFU admission head CAS did not succeed after "
            f"{self._max_cas_retries} attempts"
        )

    def verify_mfu_admission_v2(
        self,
        proof: MFUAdmissionProofV1,
        *,
        work_id: str,
        run_contract: PrefixMFURunContractV2,
    ) -> MFUAdmissionV1:
        """Replay a proof from the accepted head; never trust a bare event URI."""

        if not isinstance(proof, MFUAdmissionProofV1):
            raise InvalidReceipt("MFU admission proof has the wrong record type")
        if not isinstance(run_contract, PrefixMFURunContractV2):
            raise InvalidReceipt("MFU run contract has the wrong record type")
        work_id = _identifier(work_id, "work_id")
        if proof.campaign_id != self._campaign_id:
            raise InvalidReceipt("MFU admission proof belongs to another campaign")
        if proof.admission.work_id != work_id:
            raise InvalidReceipt("MFU admission proof belongs to another work item")
        if (
            proof.admission.run_contract.canonical_bytes()
            != run_contract.canonical_bytes()
        ):
            raise InvalidReceipt("MFU admission proof names another run contract")

        head, _version = self._load_head()
        inventory = self._load_inventory(head.inventory)
        snapshot = self._load_snapshot(head.snapshot)
        _view, events = self._replay_accepted(head, inventory, snapshot)
        expected_ref = _ObjectRef(proof.event_uri, proof.event_sha256)
        for event_ref, event in events:
            if event_ref != expected_ref:
                continue
            if not isinstance(event, _MFUAdmissionEvent):
                raise InvalidReceipt("proof reference is not an MFU admission event")
            if event.revision != proof.event_revision:
                raise InvalidReceipt("MFU admission proof revision is inconsistent")
            if event.admission.canonical_bytes() != proof.admission.canonical_bytes():
                raise InvalidReceipt("MFU admission proof payload is inconsistent")
            if (
                event.admission.run_contract.canonical_bytes()
                != run_contract.canonical_bytes()
            ):
                raise InvalidReceipt("accepted admission names another run contract")
            return event.admission
        raise InvalidReceipt("MFU admission event is not reachable from accepted head")

    def predeclare_mfu_v2(
        self,
        *,
        work_id: str,
        run_contract: PrefixMFURunContractV2,
        policy: TrustedMFUAuthorityPolicyV2,
        replica_region: str,
        measurement_not_before_unix_ns: int,
    ) -> ContentRefV1:
        """Publish a legacy receipt derived from an accepted admission event.

        The receipt remains available to the protocol oracle, but it grants no
        authority by itself.  The accepted admission event and replay proof are
        the fail-closed seam for any future production verifier.
        """

        from .mfu_authority import ControllerMFUEvidenceRegistryV2

        if not isinstance(run_contract, PrefixMFURunContractV2):
            raise InvalidReceipt("MFU run contract has the wrong record type")
        proof = self.admit_mfu_v2(work_id=work_id, run_contract=run_contract)
        admission = self.verify_mfu_admission_v2(
            proof,
            work_id=work_id,
            run_contract=run_contract,
        )
        head, version = self._load_head()
        registry = ControllerMFUEvidenceRegistryV2(
            self._store,
            policy=policy,
            campaign_id=self._campaign_id,
            campaign_head_uri=self.head_uri,
            campaign_head_generation=version.generation,
            campaign_head_revision=head.revision,
            replica_region=replica_region,
            clock_ns=lambda: self._now() * 1_000_000,
        )
        return registry._predeclare_from_controller(
            run_contract,
            work_id=admission.work_id,
            controller_fence=admission.fence,
            executor_id=admission.executor_id,
            lease_acquired_at_unix_ns=(admission.lease_acquired_at_unix_ms * 1_000_000),
            lease_expires_at_unix_ns=(admission.lease_expires_at_unix_ms * 1_000_000),
            measurement_not_before_unix_ns=measurement_not_before_unix_ns,
        )

    def _content_uri(self, kind: str, sha256: str) -> str:
        return f"{self._base_uri}/campaigns/{self._campaign_id}/{kind}/{sha256}.json"

    def _now(self) -> int:
        return _integer(self._clock_ms(), "controller clock", minimum=0)

    def initialize(
        self,
        works: Sequence[WorkItem],
        *,
        expected_outputs: Mapping[str, Sequence[ContentRefV1]] | None = None,
    ) -> CampaignView:
        work_items = tuple(works)
        if not work_items or not all(
            isinstance(item, (HarvestRangeV1, SAEJobV1)) for item in work_items
        ):
            raise CampaignConflict(
                "works must contain at least one HarvestRangeV1 or SAEJobV1"
            )
        expected_outputs = {} if expected_outputs is None else expected_outputs
        work_ids = {item.work_id for item in work_items}
        unknown_expected = set(expected_outputs) - work_ids
        if unknown_expected:
            raise CampaignConflict(
                f"expected outputs reference unknown work IDs: {sorted(unknown_expected)}"
            )
        entries = tuple(
            _InventoryEntry.build(item, tuple(expected_outputs.get(item.work_id, ())))
            for item in work_items
        )
        inventory = _Inventory(
            campaign_id=self._campaign_id,
            scientific_identity_sha256=entries[0].scientific_identity_sha256,
            entries=entries,
        )
        inventory_ref = self._publish("inventories", inventory.as_dict())
        initial_snapshot = _Snapshot(
            campaign_id=self._campaign_id,
            scientific_identity_sha256=inventory.scientific_identity_sha256,
            inventory_sha256=inventory_ref.sha256,
            revision=0,
            works=tuple(_WorkState.pending(item.work_id) for item in entries),
        )
        snapshot_ref = self._ref_for("snapshots", initial_snapshot.as_dict())
        event = _Event(
            campaign_id=self._campaign_id,
            kind="CAMPAIGN_INITIALIZED",
            revision=0,
            occurred_at_unix_ms=self._now(),
            work_id=None,
            previous_event=None,
            before_snapshot=None,
            after_snapshot=snapshot_ref,
            before_work=None,
            after_work=None,
        )
        event_ref = self._publish("events", event.as_dict())
        published_snapshot = self._publish("snapshots", initial_snapshot.as_dict())
        if published_snapshot != snapshot_ref:
            raise CorruptCampaign("published initial snapshot identity changed")
        head = _Head(
            campaign_id=self._campaign_id,
            scientific_identity_sha256=inventory.scientific_identity_sha256,
            inventory=inventory_ref,
            snapshot=snapshot_ref,
            last_event=event_ref,
            revision=0,
            event_count=1,
        )
        try:
            self._store.create(self.head_uri, head.as_dict())
        except CASConflict:
            existing, _version = self._load_head()
            if existing.inventory.sha256 != inventory_ref.sha256:
                raise CampaignConflict(
                    "campaign already exists with a different inventory"
                ) from None
        return self.replay()

    def view(self) -> CampaignView:
        head, _version = self._load_head()
        snapshot = self._load_snapshot(head.snapshot)
        self._validate_head_snapshot(head, snapshot)
        return self._view(head, snapshot)

    def replay(self) -> CampaignView:
        """Reconstruct accepted state using only head-followed object references."""

        head, _version = self._load_head()
        inventory = self._load_inventory(head.inventory)
        accepted_snapshot = self._load_snapshot(head.snapshot)
        view, _events = self._replay_accepted(head, inventory, accepted_snapshot)
        return view

    def acquire(
        self,
        work_id: str,
        *,
        attempt_id: str,
        executor_id: str,
        deployment_identity_sha256: str,
        lease_duration_ms: int,
    ) -> AttemptLeaseV1:
        work_id = _identifier(work_id, "work_id")
        attempt_id = _identifier(attempt_id, "attempt_id")
        executor_id = _identifier(executor_id, "executor_id")
        deployment_identity_sha256 = _sha(
            deployment_identity_sha256, "deployment identity"
        )
        duration = _integer(lease_duration_ms, "lease_duration_ms", minimum=1)

        def mutate(state: _WorkState, now: int, _entry: _InventoryEntry) -> _WorkState:
            if state.status == "COMMITTED":
                raise LeaseUnavailable("work is already committed")
            if state.status == "LEASED":
                assert state.lease is not None
                if now >= state.lease.expires_at_unix_ms:
                    raise LeaseUnavailable(
                        "lease has expired but must be explicitly requeued"
                    )
                raise LeaseUnavailable("work already has an active lease")
            lease = AttemptLeaseV1(
                campaign_id=self._campaign_id,
                work_id=work_id,
                attempt_id=attempt_id,
                fence=state.fence + 1,
                executor_id=executor_id,
                deployment_identity_sha256=deployment_identity_sha256,
                acquired_at_unix_ms=now,
                expires_at_unix_ms=now + duration,
            )
            return _WorkState(
                work_id=work_id,
                status="LEASED",
                fence=lease.fence,
                last_attempt_id=attempt_id,
                lease=lease,
                progress=None,
                receipt=None,
                last_progress_sequence=state.last_progress_sequence,
                completed_units=state.completed_units,
                total_units=state.total_units,
            )

        state = self._transition("LEASE_ACQUIRED", work_id, mutate)
        assert state.lease is not None
        return state.lease

    def renew(
        self,
        lease: AttemptLeaseV1,
        *,
        lease_duration_ms: int,
    ) -> AttemptLeaseV1:
        if lease.campaign_id != self._campaign_id:
            raise StaleAttempt("lease belongs to another campaign")
        duration = _integer(lease_duration_ms, "lease_duration_ms", minimum=1)

        def mutate(state: _WorkState, now: int, _entry: _InventoryEntry) -> _WorkState:
            current = self._require_current_lease(state, lease, now)
            new_expiry = now + duration
            if new_expiry <= current.expires_at_unix_ms:
                raise LeaseUnavailable("renewal must extend the current expiry")
            renewed = AttemptLeaseV1(
                campaign_id=current.campaign_id,
                work_id=current.work_id,
                attempt_id=current.attempt_id,
                fence=current.fence,
                executor_id=current.executor_id,
                deployment_identity_sha256=current.deployment_identity_sha256,
                acquired_at_unix_ms=current.acquired_at_unix_ms,
                expires_at_unix_ms=new_expiry,
            )
            return _WorkState(
                work_id=state.work_id,
                status="LEASED",
                fence=state.fence,
                last_attempt_id=state.last_attempt_id,
                lease=renewed,
                progress=state.progress,
                receipt=None,
                last_progress_sequence=state.last_progress_sequence,
                completed_units=state.completed_units,
                total_units=state.total_units,
            )

        state = self._transition("LEASE_RENEWED", lease.work_id, mutate)
        assert state.lease is not None
        return state.lease

    def record_progress(self, progress: ProgressV1) -> ProgressV1:
        """Accept coarse checkpoint-scale progress, not per-step telemetry.

        Each accepted transition writes a complete immutable campaign snapshot,
        whose size is linear in the work inventory. High-frequency metrics must
        use a separate telemetry path rather than inflating the authority log.
        """

        def mutate(state: _WorkState, now: int, _entry: _InventoryEntry) -> _WorkState:
            self._require_attempt_fields(
                state,
                progress.attempt_id,
                progress.fence,
                now,
            )
            if progress.work_id != state.work_id:
                raise StaleAttempt("progress belongs to another work item")
            if state.total_units is not None:
                if progress.total_units != state.total_units:
                    raise InvalidProgress("total_units must remain fixed")
                assert state.last_progress_sequence is not None
                assert state.completed_units is not None
                if progress.sequence <= state.last_progress_sequence:
                    raise InvalidProgress("progress sequence must increase strictly")
                if progress.completed_units <= state.completed_units:
                    raise InvalidProgress(
                        "completed_units must increase strictly with every progress event"
                    )
            previous = state.progress
            if previous is not None:
                if progress.updated_at_unix_ms < previous.updated_at_unix_ms:
                    raise InvalidProgress("progress timestamp must not move backwards")
            return _WorkState(
                work_id=state.work_id,
                status="LEASED",
                fence=state.fence,
                last_attempt_id=state.last_attempt_id,
                lease=state.lease,
                progress=progress,
                receipt=None,
                last_progress_sequence=progress.sequence,
                completed_units=progress.completed_units,
                total_units=progress.total_units,
            )

        state = self._transition("PROGRESS_RECORDED", progress.work_id, mutate)
        assert state.progress is not None
        return state.progress

    def expire_and_requeue(self, work_id: str) -> WorkView:
        work_id = _identifier(work_id, "work_id")

        def mutate(state: _WorkState, now: int, _entry: _InventoryEntry) -> _WorkState:
            if state.status != "LEASED" or state.lease is None:
                raise LeaseUnavailable("work has no active lease to expire")
            if now < state.lease.expires_at_unix_ms:
                raise LeaseUnavailable("lease has not expired")
            return _WorkState(
                work_id=state.work_id,
                status="PENDING",
                fence=state.fence,
                last_attempt_id=state.last_attempt_id,
                lease=None,
                progress=state.progress,
                receipt=None,
                last_progress_sequence=state.last_progress_sequence,
                completed_units=state.completed_units,
                total_units=state.total_units,
            )

        state = self._transition("LEASE_EXPIRED", work_id, mutate)
        return WorkView.from_state(state)

    def commit(self, receipt: AttemptReceiptV1) -> AttemptReceiptV1:
        verified: set[_LogicalContent] = set()

        def mutate(state: _WorkState, now: int, entry: _InventoryEntry) -> _WorkState:
            self._validate_receipt_for_state(
                receipt, state, now, require_active_lease=True
            )
            self._validate_expected_outputs(receipt, entry, required=False)
            self._verify_outputs(receipt.outputs, verified)
            return _WorkState(
                work_id=state.work_id,
                status="COMMITTED",
                fence=state.fence,
                last_attempt_id=state.last_attempt_id,
                lease=None,
                progress=state.progress,
                receipt=receipt,
                last_progress_sequence=state.last_progress_sequence,
                completed_units=state.completed_units,
                total_units=state.total_units,
            )

        state = self._transition("WORK_COMMITTED", receipt.work_id, mutate)
        assert state.receipt is not None
        return state.receipt

    def adopt_orphan(self, receipt: AttemptReceiptV1) -> AttemptReceiptV1:
        """Adopt an expired attempt only against precommitted logical outputs."""

        verified: set[_LogicalContent] = set()

        def mutate(state: _WorkState, now: int, entry: _InventoryEntry) -> _WorkState:
            if (
                receipt.attempt_id != state.last_attempt_id
                or receipt.fence != state.fence
            ):
                raise StaleAttempt("orphan receipt attempt or fence is stale")
            if state.status != "PENDING" or state.lease is not None:
                raise InvalidReceipt("orphan adoption requires requeued pending work")
            self._validate_receipt_for_state(
                receipt, state, now, require_active_lease=False
            )
            self._validate_expected_outputs(receipt, entry, required=True)
            self._verify_outputs(receipt.outputs, verified)
            return _WorkState(
                work_id=state.work_id,
                status="COMMITTED",
                fence=state.fence,
                last_attempt_id=state.last_attempt_id,
                lease=None,
                progress=state.progress,
                receipt=receipt,
                last_progress_sequence=state.last_progress_sequence,
                completed_units=state.completed_units,
                total_units=state.total_units,
            )

        state = self._transition("ORPHAN_ADOPTED", receipt.work_id, mutate)
        assert state.receipt is not None
        return state.receipt

    def _transition(
        self,
        kind: str,
        work_id: str,
        mutate: Callable[[_WorkState, int, _InventoryEntry], _WorkState],
    ) -> _WorkState:
        for _attempt in range(self._max_cas_retries):
            head, version = self._load_head()
            inventory = self._load_inventory(head.inventory)
            snapshot = self._load_snapshot(head.snapshot)
            self._validate_campaign_identity(head, inventory, snapshot)
            before = snapshot.work(work_id)
            now = self._now()
            after = mutate(before, now, inventory.entry(work_id))
            if after.work_id != work_id or after == before:
                raise ControllerError("transition did not produce a new matching state")
            next_snapshot = snapshot.replacing(after)
            snapshot_ref = self._ref_for("snapshots", next_snapshot.as_dict())
            event = _Event(
                campaign_id=self._campaign_id,
                kind=kind,
                revision=next_snapshot.revision,
                occurred_at_unix_ms=now,
                work_id=work_id,
                previous_event=head.last_event,
                before_snapshot=head.snapshot,
                after_snapshot=snapshot_ref,
                before_work=before,
                after_work=after,
            )
            # Required ordering: event first, snapshot second, authority CAS last.
            event_ref = self._publish("events", event.as_dict())
            published_snapshot = self._publish("snapshots", next_snapshot.as_dict())
            if published_snapshot != snapshot_ref:
                raise CorruptCampaign("published transition snapshot identity changed")
            new_head = _Head(
                campaign_id=self._campaign_id,
                scientific_identity_sha256=head.scientific_identity_sha256,
                inventory=head.inventory,
                snapshot=snapshot_ref,
                last_event=event_ref,
                revision=next_snapshot.revision,
                event_count=head.event_count + 1,
            )
            try:
                self._store.compare_and_swap(
                    self.head_uri,
                    version.generation,
                    new_head.as_dict(),
                )
            except CASConflict:
                continue
            return after
        raise RetryLimitExceeded(
            f"head CAS did not succeed after {self._max_cas_retries} attempts"
        )

    def _require_current_lease(
        self,
        state: _WorkState,
        supplied: AttemptLeaseV1,
        now: int,
    ) -> AttemptLeaseV1:
        if state.status != "LEASED" or state.lease is None:
            raise StaleAttempt("work is not currently leased")
        current = state.lease
        if (
            supplied.work_id != current.work_id
            or supplied.attempt_id != current.attempt_id
            or supplied.fence != current.fence
            or supplied.executor_id != current.executor_id
            or supplied.deployment_identity_sha256 != current.deployment_identity_sha256
            or supplied.acquired_at_unix_ms != current.acquired_at_unix_ms
            or supplied.expires_at_unix_ms != current.expires_at_unix_ms
        ):
            raise StaleAttempt("lease is not the current fenced lease")
        if now >= current.expires_at_unix_ms:
            raise StaleAttempt("lease has expired")
        return current

    def _require_attempt_fields(
        self,
        state: _WorkState,
        attempt_id: str,
        fence: int,
        now: int,
    ) -> None:
        if state.status != "LEASED" or state.lease is None:
            raise StaleAttempt("work is not currently leased")
        if state.last_attempt_id != attempt_id or state.fence != fence:
            raise StaleAttempt("attempt or fence is stale")
        if now >= state.lease.expires_at_unix_ms:
            raise StaleAttempt("lease has expired")

    def _validate_receipt_for_state(
        self,
        receipt: AttemptReceiptV1,
        state: _WorkState,
        now: int,
        *,
        require_active_lease: bool,
    ) -> None:
        if receipt.outcome != "SUCCEEDED":
            raise InvalidReceipt("only successful receipts can commit work")
        if receipt.work_id != state.work_id:
            raise StaleAttempt("receipt belongs to another work item")
        if receipt.attempt_id != state.last_attempt_id or receipt.fence != state.fence:
            raise StaleAttempt("receipt attempt or fence is stale")
        if require_active_lease:
            self._require_attempt_fields(state, receipt.attempt_id, receipt.fence, now)
        progress = state.progress
        if progress is None:
            raise InvalidReceipt("receipt requires accepted numerical progress")
        if progress.completed_units != progress.total_units:
            raise InvalidReceipt("receipt requires completed numerical work")
        if receipt.final_progress_sequence != progress.sequence:
            raise InvalidReceipt("receipt does not name current progress sequence")

    def _validate_expected_outputs(
        self,
        receipt: AttemptReceiptV1,
        entry: _InventoryEntry,
        *,
        required: bool,
    ) -> None:
        actual = tuple(
            sorted(_LogicalContent.from_content(item) for item in receipt.outputs)
        )
        if required and not entry.expected_outputs:
            raise InvalidReceipt(
                "orphan adoption requires precommitted expected output identities"
            )
        if entry.expected_outputs and actual != entry.expected_outputs:
            raise InvalidReceipt(
                "receipt logical output identities do not exactly match expectation"
            )

    def _verify_outputs(
        self,
        outputs: Sequence[ContentRefV1],
        verified: set[_LogicalContent],
    ) -> None:
        for output in outputs:
            logical = _LogicalContent.from_content(output)
            if logical in verified:
                continue
            try:
                accepted = self._content_verifier(output)
            except Exception as error:
                raise ContentVerificationFailed(
                    f"content verifier raised for {output.sha256}"
                ) from error
            if accepted is not True:
                raise ContentVerificationFailed(
                    f"content verifier rejected {output.sha256}"
                )
            verified.add(logical)

    def _publish(self, kind: str, value: Mapping[str, Any]) -> _ObjectRef:
        reference = self._ref_for(kind, value)
        published = self._store.put_immutable(reference.uri, value)
        if published.sha256 != reference.sha256:
            raise CorruptCampaign("object store returned an unexpected content digest")
        return reference

    def _ref_for(self, kind: str, value: Mapping[str, Any]) -> _ObjectRef:
        digest = canonical_sha256(value)
        return _ObjectRef(self._content_uri(kind, digest), digest)

    def _read_ref(self, reference: _ObjectRef) -> Mapping[str, Any]:
        value = self._store.read(reference.uri)
        if value is None:
            raise CorruptCampaign(f"referenced object is missing: {reference.uri}")
        if value.sha256 != reference.sha256:
            raise CorruptCampaign(f"referenced object hash mismatch: {reference.uri}")
        return value.decoded()

    def _load_head(self) -> tuple[_Head, VersionedJSON]:
        value = self._store.read(self.head_uri)
        if value is None:
            raise CampaignNotInitialized(self._campaign_id)
        head = _Head.from_dict(value.decoded())
        if head.campaign_id != self._campaign_id:
            raise CorruptCampaign("head belongs to another campaign")
        return head, value

    def _load_inventory(self, reference: _ObjectRef) -> _Inventory:
        inventory = _Inventory.from_dict(self._read_ref(reference))
        if inventory.campaign_id != self._campaign_id:
            raise CorruptCampaign("inventory belongs to another campaign")
        return inventory

    def _load_snapshot(self, reference: _ObjectRef) -> _Snapshot:
        snapshot = _Snapshot.from_dict(self._read_ref(reference))
        if snapshot.campaign_id != self._campaign_id:
            raise CorruptCampaign("snapshot belongs to another campaign")
        return snapshot

    def _load_event(self, reference: _ObjectRef) -> _CampaignEvent:
        value = self._read_ref(reference)
        schema = value.get("schema")
        if schema == _EVENT_SCHEMA:
            return _Event.from_dict(value)
        if schema == _MFU_ADMISSION_EVENT_SCHEMA:
            return _MFUAdmissionEvent.from_dict(value)
        raise CorruptCampaign(f"unsupported campaign event schema: {schema!r}")

    def _replay_accepted(
        self,
        head: _Head,
        inventory: _Inventory,
        accepted_snapshot: _Snapshot,
    ) -> tuple[CampaignView, tuple[tuple[_ObjectRef, _CampaignEvent], ...]]:
        self._validate_campaign_identity(head, inventory, accepted_snapshot)

        reverse_events: list[tuple[_ObjectRef, _CampaignEvent]] = []
        event_ref: _ObjectRef | None = head.last_event
        expected_after = head.snapshot
        while event_ref is not None:
            event = self._load_event(event_ref)
            if event.campaign_id != self._campaign_id:
                raise CorruptCampaign("event belongs to another campaign")
            if event.after_snapshot != expected_after:
                raise CorruptCampaign("event chain has a discontinuous snapshot link")
            reverse_events.append((event_ref, event))
            if event.previous_event is None:
                if event.before_snapshot is not None:
                    raise CorruptCampaign(
                        "root event unexpectedly has a prior snapshot"
                    )
                event_ref = None
            else:
                if event.before_snapshot is None:
                    raise CorruptCampaign("event chain is missing a prior snapshot")
                event_ref = event.previous_event
                expected_after = event.before_snapshot
            if len(reverse_events) > head.event_count:
                raise CorruptCampaign("event chain is longer than the head claims")
        if len(reverse_events) != head.event_count:
            raise CorruptCampaign("event count does not match the accepted chain")

        events = tuple(reversed(reverse_events))
        if (
            not events
            or not isinstance(events[0][1], _Event)
            or events[0][1].kind != "CAMPAIGN_INITIALIZED"
        ):
            raise CorruptCampaign("accepted chain lacks an initialization event")
        snapshot = _Snapshot(
            campaign_id=self._campaign_id,
            scientific_identity_sha256=inventory.scientific_identity_sha256,
            inventory_sha256=head.inventory.sha256,
            revision=0,
            works=tuple(_WorkState.pending(item.work_id) for item in inventory.entries),
        )
        initial_ref = self._ref_for("snapshots", snapshot.as_dict())
        if events[0][1].after_snapshot != initial_ref:
            raise CorruptCampaign("initialization event does not name initial state")

        previous_event_ref = events[0][0]
        previous_snapshot_ref = initial_ref
        previous_occurred_at = events[0][1].occurred_at_unix_ms
        admission_keys: set[tuple[str, str, int]] = set()
        challenges: set[str] = set()
        for current_event_ref, event in events[1:]:
            if event.previous_event != previous_event_ref:
                raise CorruptCampaign("event predecessor reference is invalid")
            if event.before_snapshot != previous_snapshot_ref:
                raise CorruptCampaign("event before-snapshot reference is invalid")
            if event.revision != snapshot.revision + 1:
                raise CorruptCampaign("event revision is non-contiguous")
            if event.occurred_at_unix_ms < previous_occurred_at:
                raise CorruptCampaign("event timestamps regress in accepted order")
            if (
                event.work_id is None
                or event.before_work is None
                or event.after_work is None
            ):
                raise CorruptCampaign("transition event lacks a work state")
            if snapshot.work(event.work_id) != event.before_work:
                raise CorruptCampaign("event before-work state does not replay")
            if event.after_work.work_id != event.work_id:
                raise CorruptCampaign("event after-work state belongs to another item")
            if isinstance(event, _MFUAdmissionEvent):
                admission = event.admission
                if (
                    admission.scientific_identity_sha256
                    != inventory.entry(event.work_id).scientific_identity_sha256
                ):
                    raise CorruptCampaign(
                        "MFU admission scientific identity differs from inventory"
                    )
                key = (event.work_id, admission.attempt_id, admission.fence)
                if key in admission_keys:
                    raise CorruptCampaign("lease has multiple accepted MFU admissions")
                if admission.controller_challenge in challenges:
                    raise CorruptCampaign("MFU admission challenge was reused")
                admission_keys.add(key)
                challenges.add(admission.controller_challenge)
            snapshot = snapshot.replacing(event.after_work)
            after_ref = self._ref_for("snapshots", snapshot.as_dict())
            if event.after_snapshot != after_ref:
                raise CorruptCampaign("event after-work state does not match snapshot")
            previous_event_ref = current_event_ref
            previous_snapshot_ref = after_ref
            previous_occurred_at = event.occurred_at_unix_ms

        if snapshot != accepted_snapshot or previous_snapshot_ref != head.snapshot:
            raise CorruptCampaign("replayed state does not match accepted snapshot")
        if snapshot.revision != head.revision:
            raise CorruptCampaign("replayed revision does not match head")
        return self._view(head, snapshot), events

    @staticmethod
    def _admission_for_lease(
        events: Sequence[tuple[_ObjectRef, _CampaignEvent]],
        *,
        work_id: str,
        attempt_id: str,
        fence: int,
    ) -> tuple[_ObjectRef, _MFUAdmissionEvent] | None:
        for event_ref, event in events:
            if not isinstance(event, _MFUAdmissionEvent):
                continue
            admission = event.admission
            if (
                admission.work_id == work_id
                and admission.attempt_id == attempt_id
                and admission.fence == fence
            ):
                return event_ref, event
        return None

    @staticmethod
    def _admission_proof(
        event_ref: _ObjectRef,
        event: _MFUAdmissionEvent,
    ) -> MFUAdmissionProofV1:
        return MFUAdmissionProofV1(
            campaign_id=event.campaign_id,
            event_uri=event_ref.uri,
            event_sha256=event_ref.sha256,
            event_revision=event.revision,
            admission=event.admission,
        )

    def _validate_campaign_identity(
        self,
        head: _Head,
        inventory: _Inventory,
        snapshot: _Snapshot,
    ) -> None:
        if not (
            head.scientific_identity_sha256
            == inventory.scientific_identity_sha256
            == snapshot.scientific_identity_sha256
        ):
            raise CorruptCampaign("scientific identity changes within campaign state")
        if snapshot.inventory_sha256 != head.inventory.sha256:
            raise CorruptCampaign("snapshot names a different inventory")
        if snapshot.revision != head.revision:
            raise CorruptCampaign("snapshot revision differs from head")
        inventory_ids = tuple(item.work_id for item in inventory.entries)
        snapshot_ids = tuple(item.work_id for item in snapshot.works)
        if inventory_ids != snapshot_ids:
            raise CorruptCampaign("snapshot work set differs from inventory")

    def _validate_head_snapshot(self, head: _Head, snapshot: _Snapshot) -> None:
        inventory = self._load_inventory(head.inventory)
        self._validate_campaign_identity(head, inventory, snapshot)

    @staticmethod
    def _view(head: _Head, snapshot: _Snapshot) -> CampaignView:
        return CampaignView(
            campaign_id=head.campaign_id,
            scientific_identity_sha256=head.scientific_identity_sha256,
            revision=head.revision,
            event_count=head.event_count,
            inventory_sha256=head.inventory.sha256,
            snapshot_sha256=head.snapshot.sha256,
            last_event_sha256=head.last_event.sha256,
            works=tuple(WorkView.from_state(item) for item in snapshot.works),
        )


def _exact_object(
    value: Any,
    name: str,
    fields: tuple[str, ...],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CorruptCampaign(f"{name} must be an object")
    expected = set(fields)
    actual = set(value)
    if actual != expected:
        raise CorruptCampaign(
            f"{name} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def _versioned_object(
    value: Any,
    schema: str,
    fields: tuple[str, ...],
) -> Mapping[str, Any]:
    obj = _exact_object(value, schema, ("schema", *fields))
    if obj["schema"] != schema:
        raise CorruptCampaign(
            f"schema mismatch: expected {schema!r}, got {obj['schema']!r}"
        )
    return obj


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CorruptCampaign(f"{name} must be a non-empty string")
    return value


def _identifier(value: Any, name: str) -> str:
    result = _text(value, name)
    if _IDENTIFIER.fullmatch(result) is None:
        raise CorruptCampaign(f"{name} has an invalid format")
    return result


def _sha(value: Any, name: str) -> str:
    result = _text(value, name)
    if _SHA256.fullmatch(result) is None:
        raise CorruptCampaign(f"{name} SHA-256 is invalid")
    return result


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise CorruptCampaign(
            f"{name} must be an integer >= {minimum} (booleans are forbidden)"
        )
    return value


__all__ = [
    "CampaignConflict",
    "CampaignController",
    "CampaignNotInitialized",
    "CampaignView",
    "ContentVerificationFailed",
    "ControllerError",
    "CorruptCampaign",
    "InvalidProgress",
    "InvalidReceipt",
    "LeaseUnavailable",
    "MFUAdmissionProofV1",
    "MFUAdmissionV1",
    "RetryLimitExceeded",
    "StaleAttempt",
    "WorkNotFound",
    "WorkView",
]
