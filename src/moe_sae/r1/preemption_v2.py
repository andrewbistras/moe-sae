"""Zero-notice Spot recovery protocol for DeepSeek-R1 v6e-64 slices.

Version 2 is deliberately additive: the historical 75-second drain protocol
in :mod:`moe_sae.r1.preemption` remains unchanged.  This protocol assumes the
only notice guaranteed to correctness is **zero milliseconds**.  A metadata
``TRUE`` response, ambiguous metadata, lease expiry, or loss of one slice
member may close the local step gate, but none can authorize an emergency
checkpoint.  Recovery uses only the last checkpoint whose complete seal was
published through the controller's compare-and-swap pointer before the loss.

The classes below perform no polling, clock reads, network I/O, checkpoint
writes, or process shutdown.  Those effects belong to adapters.  Records and
transitions are pure; ``ZeroNoticeStepLatchV2`` adds only one lock-protected
linearization point and an injected immutable-incident publication sink.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading
from typing import Any, Callable, ClassVar, Mapping, Self

from .placement_v2 import DeploymentEnvelopeV2, GCSObjectVersionRefV2
from .protocol import (
    AttemptLeaseV1,
    CanonicalRecord,
    DeploymentIdentityV1,
    IDENTIFIER_PATTERN,
    ProtocolError,
    SHA256_PATTERN,
    require_integer,
    require_record_object,
    require_string,
)


_TRIGGERS = frozenset(
    {
        "METADATA_TRUE",
        "METADATA_AMBIGUOUS",
        "LEASE_EXPIRED",
        "SLICE_MEMBER_LOST",
        "VM_TERMINATED",
    }
)
_UNPUBLISHED_STATES = frozenset(
    {"IN_FLIGHT", "PARTIAL", "FULLY_SEALED_NOT_CAS_PUBLISHED"}
)
_CHECKPOINT_RULE = "LAST_FULLY_SEALED_CAS_PUBLISHED_POINTER_ONLY"


class ZeroNoticePreemptionError(RuntimeError):
    """Base class for local V2 gate failures."""


class AttemptBindingMismatchV2(ZeroNoticePreemptionError):
    """A caller supplied a stale or different attempt/fence authority."""


class NewStepsForbiddenV2(ZeroNoticePreemptionError):
    """The one-way local step gate is already closed."""

    def __init__(self, incident: ZeroNoticeSliceIncidentV2) -> None:
        self.incident = incident
        super().__init__(f"new steps are forbidden; incident={incident.digest}")


class TerminalIncidentImmutableV2(ZeroNoticePreemptionError):
    """A different terminal incident was proposed after the latch closed."""


class IncidentPublicationErrorV2(ZeroNoticePreemptionError):
    """The incident is latched safely but its injected publisher failed."""

    def __init__(
        self, incident: ZeroNoticeSliceIncidentV2, cause: BaseException
    ) -> None:
        self.incident = incident
        self.cause = cause
        super().__init__(f"incident publication failed; incident={incident.digest}")


def _sha256(value: Any, name: str) -> str:
    return require_string(value, name, pattern=SHA256_PATTERN)


def _identifier(value: Any, name: str) -> str:
    return require_string(value, name, pattern=IDENTIFIER_PATTERN)


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ProtocolError(f"{name} must be a boolean")
    return value


def _optional_sha256(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, name)


def _record_array(
    value: Any,
    name: str,
    record_type: type[CanonicalRecord],
) -> tuple[CanonicalRecord, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProtocolError(f"{name} must be an array")
    result = tuple(value)
    if any(type(item) is not record_type for item in result):
        raise ProtocolError(f"{name} contains the wrong record type")
    return result


@dataclass(frozen=True, slots=True)
class AttemptAuthorityV2(CanonicalRecord):
    """Exact lease -> deployment identity -> V2 envelope authority chain."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.zero-notice-attempt-authority.v2"

    lease: AttemptLeaseV1
    deployment_identity: DeploymentIdentityV1
    deployment: DeploymentEnvelopeV2

    def __post_init__(self) -> None:
        if type(self.lease) is not AttemptLeaseV1:
            raise ProtocolError("lease must be AttemptLeaseV1")
        if type(self.deployment_identity) is not DeploymentIdentityV1:
            raise ProtocolError("deployment_identity must be DeploymentIdentityV1")
        if type(self.deployment) is not DeploymentEnvelopeV2:
            raise ProtocolError("deployment must be DeploymentEnvelopeV2")
        if self.lease.deployment_identity_sha256 != self.deployment_identity.digest:
            raise ProtocolError("lease does not bind the deployment identity")
        if (
            self.deployment_identity.deployment_envelope_sha256
            != self.deployment.digest
        ):
            raise ProtocolError("deployment identity does not bind the V2 envelope")
        if self.deployment_identity.attempt_id != self.lease.attempt_id:
            raise ProtocolError("deployment identity and lease attempt IDs differ")
        if self.deployment.slice_attempt_id != self.lease.attempt_id:
            raise ProtocolError("deployment envelope and lease attempt IDs differ")

    @property
    def fence(self) -> int:
        return self.lease.fence

    @property
    def attempt_id(self) -> str:
        return self.lease.attempt_id

    def same_binding(self, other: object) -> bool:
        return type(other) is AttemptAuthorityV2 and self.canonical_bytes() == (
            other.canonical_bytes()
        )

    def same_attempt_binding(self, other: object) -> bool:
        """Compare the stable lease fields, permitting only expiry renewal."""

        if type(other) is not AttemptAuthorityV2:
            return False
        stable_lease_fields = (
            "campaign_id",
            "work_id",
            "attempt_id",
            "fence",
            "executor_id",
            "deployment_identity_sha256",
            "acquired_at_unix_ms",
        )
        return (
            all(
                getattr(self.lease, field) == getattr(other.lease, field)
                for field in stable_lease_fields
            )
            and self.deployment_identity.canonical_bytes()
            == other.deployment_identity.canonical_bytes()
            and self.deployment.canonical_bytes() == other.deployment.canonical_bytes()
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "lease": self.lease.as_dict(),
            "deployment_identity": self.deployment_identity.as_dict(),
            "deployment": self.deployment.as_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = require_record_object(
            value,
            schema=cls.SCHEMA,
            fields=("lease", "deployment_identity", "deployment"),
        )
        return cls(
            lease=AttemptLeaseV1.from_dict(obj["lease"]),
            deployment_identity=DeploymentIdentityV1.from_dict(
                obj["deployment_identity"]
            ),
            deployment=DeploymentEnvelopeV2.from_dict(obj["deployment"]),
        )


@dataclass(frozen=True, slots=True)
class FullySealedCheckpointV2(CanonicalRecord):
    """Immutable checkpoint seal payload; publication is a separate fact."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.fully-sealed-checkpoint.v2"

    campaign_id: str
    work_id: str
    source_attempt_id: str
    source_fence: int
    checkpoint_id: str
    completed_steps: int
    manifest: GCSObjectVersionRefV2
    seal_receipt_sha256: str
    sealed_at_unix_ms: int
    seal_state: str

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, "campaign_id")
        _identifier(self.work_id, "work_id")
        _identifier(self.source_attempt_id, "source_attempt_id")
        require_integer(self.source_fence, "source_fence", minimum=1)
        _identifier(self.checkpoint_id, "checkpoint_id")
        require_integer(self.completed_steps, "completed_steps", minimum=0)
        if type(self.manifest) is not GCSObjectVersionRefV2:
            raise ProtocolError("manifest must be GCSObjectVersionRefV2")
        _sha256(self.seal_receipt_sha256, "seal_receipt_sha256")
        require_integer(self.sealed_at_unix_ms, "sealed_at_unix_ms", minimum=0)
        if self.seal_state != "FULLY_SEALED":
            raise ProtocolError("seal_state must be FULLY_SEALED")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "campaign_id": self.campaign_id,
            "work_id": self.work_id,
            "source_attempt_id": self.source_attempt_id,
            "source_fence": self.source_fence,
            "checkpoint_id": self.checkpoint_id,
            "completed_steps": self.completed_steps,
            "manifest": self.manifest.as_dict(),
            "seal_receipt_sha256": self.seal_receipt_sha256,
            "sealed_at_unix_ms": self.sealed_at_unix_ms,
            "seal_state": self.seal_state,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "campaign_id",
            "work_id",
            "source_attempt_id",
            "source_fence",
            "checkpoint_id",
            "completed_steps",
            "manifest",
            "seal_receipt_sha256",
            "sealed_at_unix_ms",
            "seal_state",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            campaign_id=_identifier(obj["campaign_id"], "campaign_id"),
            work_id=_identifier(obj["work_id"], "work_id"),
            source_attempt_id=_identifier(
                obj["source_attempt_id"], "source_attempt_id"
            ),
            source_fence=require_integer(
                obj["source_fence"], "source_fence", minimum=1
            ),
            checkpoint_id=_identifier(obj["checkpoint_id"], "checkpoint_id"),
            completed_steps=require_integer(
                obj["completed_steps"], "completed_steps", minimum=0
            ),
            manifest=GCSObjectVersionRefV2.from_dict(obj["manifest"]),
            seal_receipt_sha256=_sha256(
                obj["seal_receipt_sha256"], "seal_receipt_sha256"
            ),
            sealed_at_unix_ms=require_integer(
                obj["sealed_at_unix_ms"], "sealed_at_unix_ms", minimum=0
            ),
            seal_state=require_string(obj["seal_state"], "seal_state"),
        )


@dataclass(frozen=True, slots=True)
class CASPublishedCheckpointPointerV2(CanonicalRecord):
    """The sole V2 recovery authority: exact checkpoint bytes at a CAS generation.

    ``pointer_object`` must name an object generation whose bytes are exactly
    ``checkpoint.canonical_bytes()``.  Fetching those bytes with the recorded
    generation precondition is an adapter responsibility; ``verify_bytes``
    performs the content check after the read.
    """

    SCHEMA: ClassVar[str] = "moe-sae.r1.cas-published-checkpoint-pointer.v2"

    checkpoint: FullySealedCheckpointV2
    pointer_object: GCSObjectVersionRefV2
    cas_expected_generation: int
    published_at_unix_ms: int
    publication_state: str

    def __post_init__(self) -> None:
        if type(self.checkpoint) is not FullySealedCheckpointV2:
            raise ProtocolError("checkpoint must be FullySealedCheckpointV2")
        if type(self.pointer_object) is not GCSObjectVersionRefV2:
            raise ProtocolError("pointer_object must be GCSObjectVersionRefV2")
        payload = self.checkpoint.canonical_bytes()
        if self.pointer_object.size_bytes != len(payload):
            raise ProtocolError("pointer object size does not match checkpoint payload")
        if self.pointer_object.sha256 != hashlib.sha256(payload).hexdigest():
            raise ProtocolError("pointer object hash does not match checkpoint payload")
        expected = require_integer(
            self.cas_expected_generation,
            "cas_expected_generation",
            minimum=0,
        )
        if expected >= self.pointer_object.generation:
            raise ProtocolError(
                "CAS expected generation must precede published object generation"
            )
        published = require_integer(
            self.published_at_unix_ms,
            "published_at_unix_ms",
            minimum=0,
        )
        if published < self.checkpoint.sealed_at_unix_ms:
            raise ProtocolError("pointer publication cannot precede checkpoint seal")
        if self.publication_state != "CAS_PUBLISHED":
            raise ProtocolError("publication_state must be CAS_PUBLISHED")

    def verify_bytes(self, value: bytes) -> None:
        self.pointer_object.verify_bytes(value)
        if value != self.checkpoint.canonical_bytes():
            raise ProtocolError(
                "pointer bytes are not the canonical checkpoint payload"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "checkpoint": self.checkpoint.as_dict(),
            "pointer_object": self.pointer_object.as_dict(),
            "cas_expected_generation": self.cas_expected_generation,
            "published_at_unix_ms": self.published_at_unix_ms,
            "publication_state": self.publication_state,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "checkpoint",
            "pointer_object",
            "cas_expected_generation",
            "published_at_unix_ms",
            "publication_state",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            checkpoint=FullySealedCheckpointV2.from_dict(obj["checkpoint"]),
            pointer_object=GCSObjectVersionRefV2.from_dict(obj["pointer_object"]),
            cas_expected_generation=require_integer(
                obj["cas_expected_generation"],
                "cas_expected_generation",
                minimum=0,
            ),
            published_at_unix_ms=require_integer(
                obj["published_at_unix_ms"],
                "published_at_unix_ms",
                minimum=0,
            ),
            publication_state=require_string(
                obj["publication_state"], "publication_state"
            ),
        )


@dataclass(frozen=True, slots=True)
class UnpublishedCheckpointV2(CanonicalRecord):
    """Diagnostic-only checkpoint state that is never recovery authority."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.unpublished-checkpoint.v2"

    campaign_id: str
    work_id: str
    attempt_id: str
    fence: int
    checkpoint_id: str
    state: str
    diagnostic_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, "campaign_id")
        _identifier(self.work_id, "work_id")
        _identifier(self.attempt_id, "attempt_id")
        require_integer(self.fence, "fence", minimum=1)
        _identifier(self.checkpoint_id, "checkpoint_id")
        if self.state not in _UNPUBLISHED_STATES:
            raise ProtocolError("state must remain explicitly unpublished")
        _sha256(self.diagnostic_sha256, "diagnostic_sha256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "campaign_id": self.campaign_id,
            "work_id": self.work_id,
            "attempt_id": self.attempt_id,
            "fence": self.fence,
            "checkpoint_id": self.checkpoint_id,
            "state": self.state,
            "diagnostic_sha256": self.diagnostic_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "campaign_id",
            "work_id",
            "attempt_id",
            "fence",
            "checkpoint_id",
            "state",
            "diagnostic_sha256",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            campaign_id=_identifier(obj["campaign_id"], "campaign_id"),
            work_id=_identifier(obj["work_id"], "work_id"),
            attempt_id=_identifier(obj["attempt_id"], "attempt_id"),
            fence=require_integer(obj["fence"], "fence", minimum=1),
            checkpoint_id=_identifier(obj["checkpoint_id"], "checkpoint_id"),
            state=require_string(obj["state"], "state"),
            diagnostic_sha256=_sha256(obj["diagnostic_sha256"], "diagnostic_sha256"),
        )


@dataclass(frozen=True, slots=True)
class ZeroNoticeSliceIncidentV2(CanonicalRecord):
    """Immutable whole-slice terminal incident; it never requests a save."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.zero-notice-slice-incident.v2"

    authority: AttemptAuthorityV2
    trigger_kind: str
    guaranteed_notice_ms: int
    termination_action: str
    loss_scope: str
    lost_vm_count: int
    lost_chip_count: int
    checkpoint_selection_rule: str
    recovery_checkpoint: CASPublishedCheckpointPointerV2 | None
    ignored_unpublished_checkpoints: tuple[UnpublishedCheckpointV2, ...]
    trigger_authorizes_checkpoint: bool
    emergency_checkpoint_authorized: bool
    new_steps_forbidden: bool
    replacement_required: bool
    authorized_steps_before_latch: int

    def __post_init__(self) -> None:
        if type(self.authority) is not AttemptAuthorityV2:
            raise ProtocolError("authority must be AttemptAuthorityV2")
        if self.trigger_kind not in _TRIGGERS:
            raise ProtocolError("trigger_kind is unsupported")
        if (
            require_integer(
                self.guaranteed_notice_ms, "guaranteed_notice_ms", minimum=0
            )
            != 0
        ):
            raise ProtocolError("guaranteed_notice_ms must be zero")
        if self.termination_action != "STOP":
            raise ProtocolError("termination_action must be STOP")
        if self.loss_scope != "WHOLE_SLICE":
            raise ProtocolError("loss_scope must be WHOLE_SLICE")
        if require_integer(self.lost_vm_count, "lost_vm_count", minimum=1) != 16:
            raise ProtocolError("whole-slice loss must contain 16 TPU VMs")
        if require_integer(self.lost_chip_count, "lost_chip_count", minimum=1) != 64:
            raise ProtocolError("whole-slice loss must contain 64 TPU chips")
        if self.checkpoint_selection_rule != _CHECKPOINT_RULE:
            raise ProtocolError("checkpoint selection rule is not fail-closed")
        pointer = self.recovery_checkpoint
        if pointer is not None and type(pointer) is not CASPublishedCheckpointPointerV2:
            raise ProtocolError(
                "recovery_checkpoint must be CASPublishedCheckpointPointerV2 or null"
            )
        lease = self.authority.lease
        if pointer is not None:
            checkpoint = pointer.checkpoint
            if not pointer.pointer_object.uri.startswith(
                self.authority.deployment.controller_root_uri
            ):
                raise ProtocolError(
                    "recovery pointer is outside the deployment controller root"
                )
            if (checkpoint.campaign_id, checkpoint.work_id) != (
                lease.campaign_id,
                lease.work_id,
            ):
                raise ProtocolError("recovery checkpoint belongs to another work item")
            if checkpoint.source_fence > lease.fence:
                raise ProtocolError("recovery checkpoint comes from a future fence")
            if (
                checkpoint.source_fence == lease.fence
                and checkpoint.source_attempt_id != lease.attempt_id
            ):
                raise ProtocolError(
                    "recovery checkpoint attempt conflicts with the current fence"
                )
            if pointer.published_at_unix_ms > lease.expires_at_unix_ms:
                raise ProtocolError(
                    "recovery pointer was published after the current lease expiry"
                )
        ignored = _record_array(
            self.ignored_unpublished_checkpoints,
            "ignored_unpublished_checkpoints",
            UnpublishedCheckpointV2,
        )
        for item in ignored:
            assert isinstance(item, UnpublishedCheckpointV2)
            if (item.campaign_id, item.work_id) != (
                lease.campaign_id,
                lease.work_id,
            ):
                raise ProtocolError("ignored checkpoint belongs to another work item")
            if item.fence > lease.fence:
                raise ProtocolError("ignored checkpoint comes from a future fence")
        if len({item.digest for item in ignored}) != len(ignored):
            raise ProtocolError("ignored checkpoints must not contain duplicates")
        object.__setattr__(
            self,
            "ignored_unpublished_checkpoints",
            tuple(sorted(ignored, key=lambda item: item.digest)),
        )
        if _boolean(
            self.trigger_authorizes_checkpoint, "trigger_authorizes_checkpoint"
        ):
            raise ProtocolError("a terminal trigger cannot authorize a checkpoint")
        if _boolean(
            self.emergency_checkpoint_authorized,
            "emergency_checkpoint_authorized",
        ):
            raise ProtocolError("emergency checkpoints are forbidden")
        if not _boolean(self.new_steps_forbidden, "new_steps_forbidden"):
            raise ProtocolError("new_steps_forbidden must be true")
        if not _boolean(self.replacement_required, "replacement_required"):
            raise ProtocolError("replacement_required must be true")
        require_integer(
            self.authorized_steps_before_latch,
            "authorized_steps_before_latch",
            minimum=0,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "authority": self.authority.as_dict(),
            "trigger_kind": self.trigger_kind,
            "guaranteed_notice_ms": self.guaranteed_notice_ms,
            "termination_action": self.termination_action,
            "loss_scope": self.loss_scope,
            "lost_vm_count": self.lost_vm_count,
            "lost_chip_count": self.lost_chip_count,
            "checkpoint_selection_rule": self.checkpoint_selection_rule,
            "recovery_checkpoint": (
                None
                if self.recovery_checkpoint is None
                else self.recovery_checkpoint.as_dict()
            ),
            "ignored_unpublished_checkpoints": [
                item.as_dict() for item in self.ignored_unpublished_checkpoints
            ],
            "trigger_authorizes_checkpoint": self.trigger_authorizes_checkpoint,
            "emergency_checkpoint_authorized": self.emergency_checkpoint_authorized,
            "new_steps_forbidden": self.new_steps_forbidden,
            "replacement_required": self.replacement_required,
            "authorized_steps_before_latch": self.authorized_steps_before_latch,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "authority",
            "trigger_kind",
            "guaranteed_notice_ms",
            "termination_action",
            "loss_scope",
            "lost_vm_count",
            "lost_chip_count",
            "checkpoint_selection_rule",
            "recovery_checkpoint",
            "ignored_unpublished_checkpoints",
            "trigger_authorizes_checkpoint",
            "emergency_checkpoint_authorized",
            "new_steps_forbidden",
            "replacement_required",
            "authorized_steps_before_latch",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        raw_pointer = obj["recovery_checkpoint"]
        if raw_pointer is not None and not isinstance(raw_pointer, Mapping):
            raise ProtocolError("recovery_checkpoint must be an object or null")
        raw_ignored = obj["ignored_unpublished_checkpoints"]
        if not isinstance(raw_ignored, list):
            raise ProtocolError("ignored_unpublished_checkpoints must be an array")
        return cls(
            authority=AttemptAuthorityV2.from_dict(obj["authority"]),
            trigger_kind=require_string(obj["trigger_kind"], "trigger_kind"),
            guaranteed_notice_ms=require_integer(
                obj["guaranteed_notice_ms"], "guaranteed_notice_ms", minimum=0
            ),
            termination_action=require_string(
                obj["termination_action"], "termination_action"
            ),
            loss_scope=require_string(obj["loss_scope"], "loss_scope"),
            lost_vm_count=require_integer(
                obj["lost_vm_count"], "lost_vm_count", minimum=1
            ),
            lost_chip_count=require_integer(
                obj["lost_chip_count"], "lost_chip_count", minimum=1
            ),
            checkpoint_selection_rule=require_string(
                obj["checkpoint_selection_rule"], "checkpoint_selection_rule"
            ),
            recovery_checkpoint=(
                None
                if raw_pointer is None
                else CASPublishedCheckpointPointerV2.from_dict(raw_pointer)
            ),
            ignored_unpublished_checkpoints=tuple(
                UnpublishedCheckpointV2.from_dict(item) for item in raw_ignored
            ),
            trigger_authorizes_checkpoint=_boolean(
                obj["trigger_authorizes_checkpoint"],
                "trigger_authorizes_checkpoint",
            ),
            emergency_checkpoint_authorized=_boolean(
                obj["emergency_checkpoint_authorized"],
                "emergency_checkpoint_authorized",
            ),
            new_steps_forbidden=_boolean(
                obj["new_steps_forbidden"], "new_steps_forbidden"
            ),
            replacement_required=_boolean(
                obj["replacement_required"], "replacement_required"
            ),
            authorized_steps_before_latch=require_integer(
                obj["authorized_steps_before_latch"],
                "authorized_steps_before_latch",
                minimum=0,
            ),
        )


@dataclass(frozen=True, slots=True)
class ZeroNoticeRecoveryPlanV2(CanonicalRecord):
    """Pure successor transition after lease expiry and whole-MIG replacement."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.zero-notice-recovery-plan.v2"

    incident: ZeroNoticeSliceIncidentV2
    replacement_authority: AttemptAuthorityV2
    expired_lease_sha256: str
    expired_fence: int
    replacement_fence: int
    restore_pointer_sha256: str | None
    restore_action: str
    old_lease_expired: bool
    old_fence_invalidated: bool
    fresh_whole_mig_required: bool

    def __post_init__(self) -> None:
        if type(self.incident) is not ZeroNoticeSliceIncidentV2:
            raise ProtocolError("incident must be ZeroNoticeSliceIncidentV2")
        if type(self.replacement_authority) is not AttemptAuthorityV2:
            raise ProtocolError("replacement_authority must be AttemptAuthorityV2")
        old = self.incident.authority
        new = self.replacement_authority
        if (
            _sha256(self.expired_lease_sha256, "expired_lease_sha256")
            != old.lease.digest
        ):
            raise ProtocolError("expired lease digest does not match incident")
        if require_integer(self.expired_fence, "expired_fence", minimum=1) != old.fence:
            raise ProtocolError("expired fence does not match incident")
        if (
            require_integer(self.replacement_fence, "replacement_fence", minimum=1)
            != new.fence
        ):
            raise ProtocolError("replacement fence does not match successor")
        if (new.lease.campaign_id, new.lease.work_id) != (
            old.lease.campaign_id,
            old.lease.work_id,
        ):
            raise ProtocolError("replacement belongs to another work item")
        if new.attempt_id == old.attempt_id:
            raise ProtocolError("replacement must use a new attempt ID")
        if new.fence <= old.fence:
            raise ProtocolError("replacement fence must advance")
        if new.lease.acquired_at_unix_ms < old.lease.expires_at_unix_ms:
            raise ProtocolError("replacement cannot precede old lease expiry")
        new.deployment.verify_fresh_replacement_of(old.deployment)
        pointer = self.incident.recovery_checkpoint
        expected_digest = None if pointer is None else pointer.digest
        if (
            _optional_sha256(self.restore_pointer_sha256, "restore_pointer_sha256")
            != expected_digest
        ):
            raise ProtocolError("restore pointer does not match terminal incident")
        expected_action = (
            "RESTART_FROM_INITIAL_STATE"
            if pointer is None
            else "RESTORE_LAST_CAS_PUBLISHED"
        )
        if self.restore_action != expected_action:
            raise ProtocolError(
                "restore action is inconsistent with recovery authority"
            )
        if not _boolean(self.old_lease_expired, "old_lease_expired"):
            raise ProtocolError("old_lease_expired must be true")
        if not _boolean(self.old_fence_invalidated, "old_fence_invalidated"):
            raise ProtocolError("old_fence_invalidated must be true")
        if not _boolean(self.fresh_whole_mig_required, "fresh_whole_mig_required"):
            raise ProtocolError("fresh_whole_mig_required must be true")

    @classmethod
    def build(
        cls,
        incident: ZeroNoticeSliceIncidentV2,
        replacement_authority: AttemptAuthorityV2,
    ) -> Self:
        if type(incident) is not ZeroNoticeSliceIncidentV2:
            raise ProtocolError("incident must be ZeroNoticeSliceIncidentV2")
        if type(replacement_authority) is not AttemptAuthorityV2:
            raise ProtocolError("replacement_authority must be AttemptAuthorityV2")
        pointer = incident.recovery_checkpoint
        return cls(
            incident=incident,
            replacement_authority=replacement_authority,
            expired_lease_sha256=incident.authority.lease.digest,
            expired_fence=incident.authority.fence,
            replacement_fence=replacement_authority.fence,
            restore_pointer_sha256=None if pointer is None else pointer.digest,
            restore_action=(
                "RESTART_FROM_INITIAL_STATE"
                if pointer is None
                else "RESTORE_LAST_CAS_PUBLISHED"
            ),
            old_lease_expired=True,
            old_fence_invalidated=True,
            fresh_whole_mig_required=True,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "incident": self.incident.as_dict(),
            "replacement_authority": self.replacement_authority.as_dict(),
            "expired_lease_sha256": self.expired_lease_sha256,
            "expired_fence": self.expired_fence,
            "replacement_fence": self.replacement_fence,
            "restore_pointer_sha256": self.restore_pointer_sha256,
            "restore_action": self.restore_action,
            "old_lease_expired": self.old_lease_expired,
            "old_fence_invalidated": self.old_fence_invalidated,
            "fresh_whole_mig_required": self.fresh_whole_mig_required,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "incident",
            "replacement_authority",
            "expired_lease_sha256",
            "expired_fence",
            "replacement_fence",
            "restore_pointer_sha256",
            "restore_action",
            "old_lease_expired",
            "old_fence_invalidated",
            "fresh_whole_mig_required",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            incident=ZeroNoticeSliceIncidentV2.from_dict(obj["incident"]),
            replacement_authority=AttemptAuthorityV2.from_dict(
                obj["replacement_authority"]
            ),
            expired_lease_sha256=_sha256(
                obj["expired_lease_sha256"], "expired_lease_sha256"
            ),
            expired_fence=require_integer(
                obj["expired_fence"], "expired_fence", minimum=1
            ),
            replacement_fence=require_integer(
                obj["replacement_fence"], "replacement_fence", minimum=1
            ),
            restore_pointer_sha256=_optional_sha256(
                obj["restore_pointer_sha256"], "restore_pointer_sha256"
            ),
            restore_action=require_string(obj["restore_action"], "restore_action"),
            old_lease_expired=_boolean(obj["old_lease_expired"], "old_lease_expired"),
            old_fence_invalidated=_boolean(
                obj["old_fence_invalidated"], "old_fence_invalidated"
            ),
            fresh_whole_mig_required=_boolean(
                obj["fresh_whole_mig_required"], "fresh_whole_mig_required"
            ),
        )


IncidentSinkV2 = Callable[[ZeroNoticeSliceIncidentV2], object]


class ZeroNoticeStepLatchV2:
    """Thread-safe one-way local step/terminal-incident linearization point.

    A successful ``authorize_new_step`` means only that the step linearized
    before this local latch.  Whole-slice propagation is a controller/runtime
    concern.  No clock, metadata client, checkpoint writer, or emergency-save
    callback is accepted by this class.
    """

    def __init__(
        self,
        authority: AttemptAuthorityV2,
        *,
        incident_sink: IncidentSinkV2,
    ) -> None:
        if type(authority) is not AttemptAuthorityV2:
            raise ProtocolError("authority must be AttemptAuthorityV2")
        if not callable(incident_sink):
            raise TypeError("incident_sink must be callable")
        self._authority = authority
        self._incident_sink = incident_sink
        self._lock = threading.RLock()
        self._authorized_steps = 0
        self._incident: ZeroNoticeSliceIncidentV2 | None = None
        self._sink_succeeded = False
        self._sink_error: BaseException | None = None

    @property
    def authority(self) -> AttemptAuthorityV2:
        with self._lock:
            return self._authority

    @property
    def new_steps_allowed(self) -> bool:
        with self._lock:
            return self._incident is None

    @property
    def authorized_step_count(self) -> int:
        with self._lock:
            return self._authorized_steps

    @property
    def incident(self) -> ZeroNoticeSliceIncidentV2 | None:
        with self._lock:
            return self._incident

    @property
    def incident_published(self) -> bool:
        with self._lock:
            return self._sink_succeeded

    def authorize_new_step(self, authority: AttemptAuthorityV2) -> int:
        """Return a local monotonic permit number, without reading a clock."""

        with self._lock:
            if self._incident is not None:
                self._require_exact_authority(authority)
                raise NewStepsForbiddenV2(self._incident)
            self._require_open_authority(authority)
            self._authorized_steps += 1
            return self._authorized_steps

    def latch(
        self,
        authority: AttemptAuthorityV2,
        *,
        trigger_kind: str,
        recovery_checkpoint: CASPublishedCheckpointPointerV2 | None,
        ignored_unpublished_checkpoints: tuple[UnpublishedCheckpointV2, ...] = (),
    ) -> ZeroNoticeSliceIncidentV2:
        """Close the gate and publish one immutable no-save incident."""

        with self._lock:
            if self._incident is not None:
                self._require_exact_authority(authority)
                candidate = self._build_incident_locked(
                    trigger_kind=trigger_kind,
                    recovery_checkpoint=recovery_checkpoint,
                    ignored_unpublished_checkpoints=ignored_unpublished_checkpoints,
                )
                if candidate.canonical_bytes() != self._incident.canonical_bytes():
                    raise TerminalIncidentImmutableV2(
                        "first terminal incident is immutable"
                    )
                self._raise_publication_error_locked()
                return self._incident
            self._require_open_authority(authority)
            candidate = self._build_incident_locked(
                trigger_kind=trigger_kind,
                recovery_checkpoint=recovery_checkpoint,
                ignored_unpublished_checkpoints=ignored_unpublished_checkpoints,
            )
            self._incident = candidate
            self._invoke_sink_locked()
            self._raise_publication_error_locked()
            return candidate

    def retry_incident_publication(self) -> bool:
        """Retry only the exact latched incident; never reopen the gate."""

        with self._lock:
            if self._incident is None:
                raise ZeroNoticePreemptionError(
                    "cannot retry publication before a terminal incident"
                )
            if self._sink_succeeded:
                return False
            incident = self._incident
            self._invoke_sink_locked()
            if self._incident is not incident:
                raise AssertionError("terminal incident changed during retry")
            self._raise_publication_error_locked()
            return True

    def _build_incident_locked(
        self,
        *,
        trigger_kind: str,
        recovery_checkpoint: CASPublishedCheckpointPointerV2 | None,
        ignored_unpublished_checkpoints: tuple[UnpublishedCheckpointV2, ...],
    ) -> ZeroNoticeSliceIncidentV2:
        return ZeroNoticeSliceIncidentV2(
            authority=self._authority,
            trigger_kind=trigger_kind,
            guaranteed_notice_ms=0,
            termination_action="STOP",
            loss_scope="WHOLE_SLICE",
            lost_vm_count=16,
            lost_chip_count=64,
            checkpoint_selection_rule=_CHECKPOINT_RULE,
            recovery_checkpoint=recovery_checkpoint,
            ignored_unpublished_checkpoints=ignored_unpublished_checkpoints,
            trigger_authorizes_checkpoint=False,
            emergency_checkpoint_authorized=False,
            new_steps_forbidden=True,
            replacement_required=True,
            authorized_steps_before_latch=self._authorized_steps,
        )

    def _require_exact_authority(self, authority: AttemptAuthorityV2) -> None:
        if not self._authority.same_binding(authority):
            raise AttemptBindingMismatchV2("terminal incident authority is immutable")

    def _require_open_authority(self, authority: AttemptAuthorityV2) -> None:
        if self._authority.same_binding(authority):
            return
        if (
            self._authority.same_attempt_binding(authority)
            and authority.lease.expires_at_unix_ms
            > self._authority.lease.expires_at_unix_ms
        ):
            self._authority = authority
            return
        raise AttemptBindingMismatchV2(
            "authority does not match the latch attempt/fence/deployment"
        )

    def _invoke_sink_locked(self) -> None:
        assert self._incident is not None
        try:
            self._incident_sink(self._incident)
        except Exception as error:
            self._sink_error = error
            return
        self._sink_error = None
        self._sink_succeeded = True

    def _raise_publication_error_locked(self) -> None:
        if self._sink_error is not None:
            assert self._incident is not None
            raise IncidentPublicationErrorV2(
                self._incident, self._sink_error
            ) from self._sink_error


__all__ = [
    "AttemptAuthorityV2",
    "AttemptBindingMismatchV2",
    "CASPublishedCheckpointPointerV2",
    "FullySealedCheckpointV2",
    "IncidentPublicationErrorV2",
    "NewStepsForbiddenV2",
    "TerminalIncidentImmutableV2",
    "UnpublishedCheckpointV2",
    "ZeroNoticePreemptionError",
    "ZeroNoticeRecoveryPlanV2",
    "ZeroNoticeSliceIncidentV2",
    "ZeroNoticeStepLatchV2",
]
