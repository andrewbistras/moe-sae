"""Exact-range sealing for fully read-back R1 capture fragments.

This is a pure control-plane step.  It accepts only canonical fragment order
for one immutable work range and only receipts that already passed
generation-pinned full-payload validation.  The output is a complete,
content-addressable manifest candidate; it is not yet published and remains
ineligible for production admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from .capture_data_plane_contract import (
    CaptureDataPlaneContractError,
    CaptureFragmentPlanV1,
    CaptureWorkPlanV1,
    PINNED_CAPTURE_DATA_PLANE_CONTRACT_SHA256,
    fragment_content_object_id,
)
from .capture_fragment_validator import ValidatedCaptureFragmentReceiptV1
from .gcs_immutable_byte_store import GCSPhysicalObjectReceiptV1
from .protocol import CanonicalRecord


_ENTRY_CAPABILITY = object()
_MANIFEST_CAPABILITY = object()
_MANIFEST_CONTRACT = "canonical-exact-range-full-readback-manifest-v1"


class CaptureManifestError(CaptureDataPlaneContractError):
    """Validated fragments cannot form one exact immutable range."""


@dataclass(frozen=True, slots=True, init=False)
class CaptureManifestEntryV1(CanonicalRecord):
    """One plan plus its exact generation-pinned validated object."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.capture-manifest-entry.v1"

    fragment_plan: CaptureFragmentPlanV1
    validation_receipt_sha256: str
    physical_receipt: GCSPhysicalObjectReceiptV1
    validator_id: str

    def __init__(
        self,
        *,
        plan: CaptureFragmentPlanV1,
        validated: ValidatedCaptureFragmentReceiptV1,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _ENTRY_CAPABILITY:
            raise CaptureManifestError(
                "manifest entries are available only through exact-range sealing"
            )
        if (
            type(plan) is not CaptureFragmentPlanV1
            or type(validated) is not ValidatedCaptureFragmentReceiptV1
            or validated.fragment_plan_sha256 != plan.digest
            or validated.full_payload_readback_verified is not True
            or validated.manifest_visible is not False
            or validated.production_sealable is not False
            or validated.physical_receipt.size_bytes != plan.payload_bytes
            or validated.physical_receipt.object_id
            != fragment_content_object_id(validated.physical_receipt.sha256)
        ):
            raise CaptureManifestError(
                "manifest entry is not one exact fully read-back fragment"
            )
        object.__setattr__(self, "fragment_plan", plan)
        object.__setattr__(
            self,
            "validation_receipt_sha256",
            validated.digest,
        )
        object.__setattr__(
            self,
            "physical_receipt",
            validated.physical_receipt,
        )
        object.__setattr__(self, "validator_id", validated.validator_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "fragment_plan": self.fragment_plan.as_dict(),
            "validation_receipt_sha256": self.validation_receipt_sha256,
            "physical_receipt": self.physical_receipt.as_dict(),
            "validator_id": self.validator_id,
        }


@dataclass(frozen=True, slots=True, init=False)
class SealedCaptureManifestV1(CanonicalRecord):
    """A complete exact-range manifest candidate, not a publication receipt."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.sealed-capture-manifest.v1"

    work_plan: CaptureWorkPlanV1
    capture_data_plane_contract_sha256: str
    entries: tuple[CaptureManifestEntryV1, ...]
    store_id: str
    region: str
    validator_id: str
    fragment_count: int
    token_count: int
    payload_bytes: int
    exact_range_verified: bool
    full_payload_readback_verified: bool
    manifest_complete: bool
    manifest_published: bool
    production_sealable: bool
    manifest_contract: str

    def __init__(
        self,
        *,
        work_plan: CaptureWorkPlanV1,
        entries: tuple[CaptureManifestEntryV1, ...],
        _capability: object | None = None,
    ) -> None:
        if _capability is not _MANIFEST_CAPABILITY:
            raise CaptureManifestError(
                "sealed manifests are available only through exact-range sealing"
            )
        if (
            type(work_plan) is not CaptureWorkPlanV1
            or type(entries) is not tuple
            or not entries
            or any(type(item) is not CaptureManifestEntryV1 for item in entries)
        ):
            raise CaptureManifestError("sealed manifest state has invalid exact types")
        expected = tuple(work_plan.iter_fragments())
        if (
            len(entries) != work_plan.fragment_count
            or tuple(item.fragment_plan for item in entries) != expected
        ):
            raise CaptureManifestError(
                "manifest entries do not cover the canonical exact work range"
            )
        store_ids = {item.physical_receipt.store_id for item in entries}
        regions = {item.physical_receipt.region for item in entries}
        validator_ids = {item.validator_id for item in entries}
        object_ids = tuple(item.physical_receipt.object_id for item in entries)
        physical_identities = tuple(
            (
                item.physical_receipt.uri,
                item.physical_receipt.generation,
                item.physical_receipt.metageneration,
            )
            for item in entries
        )
        if (
            len(store_ids) != 1
            or len(regions) != 1
            or len(validator_ids) != 1
            or len(set(object_ids)) != len(object_ids)
            or len(set(physical_identities)) != len(physical_identities)
        ):
            raise CaptureManifestError(
                "manifest objects are not one unique store/region/validator cohort"
            )
        payload_bytes = sum(item.physical_receipt.size_bytes for item in entries)
        if payload_bytes != work_plan.payload_bytes:
            raise CaptureManifestError(
                "manifest payload total differs from the exact work range"
            )
        object.__setattr__(self, "work_plan", work_plan)
        object.__setattr__(
            self,
            "capture_data_plane_contract_sha256",
            PINNED_CAPTURE_DATA_PLANE_CONTRACT_SHA256,
        )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "store_id", next(iter(store_ids)))
        object.__setattr__(self, "region", next(iter(regions)))
        object.__setattr__(self, "validator_id", next(iter(validator_ids)))
        object.__setattr__(self, "fragment_count", len(entries))
        object.__setattr__(self, "token_count", work_plan.token_count)
        object.__setattr__(self, "payload_bytes", payload_bytes)
        object.__setattr__(self, "exact_range_verified", True)
        object.__setattr__(self, "full_payload_readback_verified", True)
        object.__setattr__(self, "manifest_complete", True)
        object.__setattr__(self, "manifest_published", False)
        object.__setattr__(self, "production_sealable", False)
        object.__setattr__(self, "manifest_contract", _MANIFEST_CONTRACT)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "work_plan": self.work_plan.as_dict(),
            "capture_data_plane_contract_sha256": (
                self.capture_data_plane_contract_sha256
            ),
            "entries": [item.as_dict() for item in self.entries],
            "store_id": self.store_id,
            "region": self.region,
            "validator_id": self.validator_id,
            "fragment_count": self.fragment_count,
            "token_count": self.token_count,
            "payload_bytes": self.payload_bytes,
            "exact_range_verified": self.exact_range_verified,
            "full_payload_readback_verified": (self.full_payload_readback_verified),
            "manifest_complete": self.manifest_complete,
            "manifest_published": self.manifest_published,
            "production_sealable": self.production_sealable,
            "manifest_contract": self.manifest_contract,
        }


def seal_capture_manifest(
    work_plan: CaptureWorkPlanV1,
    planned_receipts: tuple[
        tuple[CaptureFragmentPlanV1, ValidatedCaptureFragmentReceiptV1],
        ...,
    ],
) -> SealedCaptureManifestV1:
    """Require canonical complete coverage; never sort away producer mistakes."""

    if type(work_plan) is not CaptureWorkPlanV1:
        raise CaptureManifestError("manifest requires an exact capture work plan")
    if (
        type(planned_receipts) is not tuple
        or not planned_receipts
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not CaptureFragmentPlanV1
            or type(item[1]) is not ValidatedCaptureFragmentReceiptV1
            for item in planned_receipts
        )
    ):
        raise CaptureManifestError(
            "planned receipts must be an exact nonempty tuple of exact pairs"
        )
    expected = tuple(work_plan.iter_fragments())
    plans = tuple(item[0] for item in planned_receipts)
    if plans != expected:
        raise CaptureManifestError(
            "planned receipts do not match canonical exact-range order"
        )
    entries = tuple(
        CaptureManifestEntryV1(
            plan=plan,
            validated=validated,
            _capability=_ENTRY_CAPABILITY,
        )
        for plan, validated in planned_receipts
    )
    return SealedCaptureManifestV1(
        work_plan=work_plan,
        entries=entries,
        _capability=_MANIFEST_CAPABILITY,
    )


__all__ = [
    "CaptureManifestEntryV1",
    "CaptureManifestError",
    "SealedCaptureManifestV1",
    "seal_capture_manifest",
]
