"""Off-window generation-pinned validation for provisional capture objects.

The numerical path ends at a provisional create acknowledgement.  This module
is the independent second half: it binds that acknowledgement to the exact
topology-neutral fragment plan and asks the immutable byte store to HEAD and
stream every byte from the acknowledged generation.  The resulting receipt
still cannot enter a manifest; exact-range sealing is a later, separate gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, ClassVar

from .capture_data_plane_contract import (
    CaptureDataPlaneContractError,
    CaptureFragmentPlanV1,
    PINNED_CAPTURE_DATA_PLANE_CONTRACT_SHA256,
    fragment_content_object_id,
)
from .capture_host_writer import ProvisionalCaptureFragmentReceiptV1
from .gcs_immutable_byte_store import (
    GCSImmutableByteStore,
    GCSImmutableByteStoreError,
    GCSPhysicalObjectReceiptV1,
)
from .protocol import CanonicalRecord, canonical_json_bytes


_VALIDATED_CAPABILITY = object()
_VALIDATION_CONTRACT = "generation-pinned-full-sha256-crc32c-readback-v1"


class CaptureFragmentValidationError(CaptureDataPlaneContractError):
    """A provisional receipt could not earn full-readback status."""


@dataclass(frozen=True, slots=True, init=False)
class ValidatedCaptureFragmentReceiptV1(CanonicalRecord):
    """One fully read-back fragment that remains invisible to manifests."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.validated-capture-fragment-receipt.v1"

    fragment_plan_sha256: str
    provisional_receipt_sha256: str
    physical_receipt: GCSPhysicalObjectReceiptV1
    validator_id: str
    object_create_acknowledged: bool
    full_payload_readback_verified: bool
    manifest_visible: bool
    production_sealable: bool
    validation_contract: str

    def __init__(
        self,
        *,
        plan: CaptureFragmentPlanV1,
        provisional: ProvisionalCaptureFragmentReceiptV1,
        physical_receipt: GCSPhysicalObjectReceiptV1,
        validator_id: str,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _VALIDATED_CAPABILITY:
            raise CaptureFragmentValidationError(
                "validated receipts are available only through the validator"
            )
        if (
            type(plan) is not CaptureFragmentPlanV1
            or type(provisional) is not ProvisionalCaptureFragmentReceiptV1
            or type(physical_receipt) is not GCSPhysicalObjectReceiptV1
            or type(validator_id) is not str
            or not validator_id.startswith("capture-fragment-validator-v1:")
        ):
            raise CaptureFragmentValidationError(
                "validated receipt state has invalid exact types or identity"
            )
        if (
            plan.digest != provisional.fragment_plan_sha256
            or physical_receipt != provisional.physical_receipt
            or physical_receipt.size_bytes != plan.payload_bytes
            or physical_receipt.object_id
            != fragment_content_object_id(physical_receipt.sha256)
        ):
            raise CaptureFragmentValidationError(
                "validated receipt does not match its plan and provisional receipt"
            )
        object.__setattr__(self, "fragment_plan_sha256", plan.digest)
        object.__setattr__(
            self,
            "provisional_receipt_sha256",
            provisional.digest,
        )
        object.__setattr__(self, "physical_receipt", physical_receipt)
        object.__setattr__(self, "validator_id", validator_id)
        object.__setattr__(self, "object_create_acknowledged", True)
        object.__setattr__(self, "full_payload_readback_verified", True)
        object.__setattr__(self, "manifest_visible", False)
        object.__setattr__(self, "production_sealable", False)
        object.__setattr__(self, "validation_contract", _VALIDATION_CONTRACT)

    def require_matches(
        self,
        plan: CaptureFragmentPlanV1,
        provisional: ProvisionalCaptureFragmentReceiptV1,
    ) -> None:
        if (
            type(plan) is not CaptureFragmentPlanV1
            or type(provisional) is not ProvisionalCaptureFragmentReceiptV1
            or self.fragment_plan_sha256 != plan.digest
            or self.provisional_receipt_sha256 != provisional.digest
            or self.physical_receipt != provisional.physical_receipt
        ):
            raise CaptureFragmentValidationError(
                "validated receipt differs from its exact source records"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "fragment_plan_sha256": self.fragment_plan_sha256,
            "provisional_receipt_sha256": self.provisional_receipt_sha256,
            "physical_receipt": self.physical_receipt.as_dict(),
            "validator_id": self.validator_id,
            "object_create_acknowledged": self.object_create_acknowledged,
            "full_payload_readback_verified": (self.full_payload_readback_verified),
            "manifest_visible": self.manifest_visible,
            "production_sealable": self.production_sealable,
            "validation_contract": self.validation_contract,
        }


class CaptureFragmentValidatorV1:
    """Immutable validator over one exact generation-pinned byte store."""

    __slots__ = (
        "_production_identity_qualified",
        "_store",
        "_validator_id",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__ and hasattr(self, name):
            raise AttributeError(f"{name} is an immutable validator snapshot")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in self.__slots__:
            raise AttributeError(f"{name} is an immutable validator snapshot")
        object.__delattr__(self, name)

    def __init__(self, store: GCSImmutableByteStore) -> None:
        if type(store) is not GCSImmutableByteStore:
            raise CaptureFragmentValidationError(
                "validator requires an exact GCS immutable byte store"
            )
        identity = {
            "schema": "moe-sae.r1.capture-fragment-validator-identity.v1",
            "validation_contract": _VALIDATION_CONTRACT,
            "capture_data_plane_contract_sha256": (
                PINNED_CAPTURE_DATA_PLANE_CONTRACT_SHA256
            ),
            "store_id": store.store_id,
            "store_prefix": store.prefix,
            "store_region": store.region,
        }
        self._store = store
        self._validator_id = (
            "capture-fragment-validator-v1:"
            + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        )
        # Code existence and an internally constructed cloud client are not a
        # live readback/chaos/throughput qualification.
        self._production_identity_qualified = False

    @property
    def store(self) -> GCSImmutableByteStore:
        return self._store

    @property
    def validator_id(self) -> str:
        return self._validator_id

    @property
    def production_identity_qualified(self) -> bool:
        return self._production_identity_qualified

    def validate(
        self,
        plan: CaptureFragmentPlanV1,
        provisional: ProvisionalCaptureFragmentReceiptV1,
    ) -> ValidatedCaptureFragmentReceiptV1:
        if (
            type(plan) is not CaptureFragmentPlanV1
            or type(provisional) is not ProvisionalCaptureFragmentReceiptV1
        ):
            raise CaptureFragmentValidationError(
                "validator requires exact plan and provisional receipt types"
            )
        physical = provisional.physical_receipt
        if (
            plan.digest != provisional.fragment_plan_sha256
            or plan.payload_bytes != physical.size_bytes
            or physical.object_id != fragment_content_object_id(physical.sha256)
        ):
            raise CaptureFragmentValidationError(
                "provisional receipt differs from the exact fragment plan"
            )
        try:
            observed = self._store.verify(physical)
        except GCSImmutableByteStoreError as error:
            raise CaptureFragmentValidationError(
                "generation-pinned full-payload readback failed"
            ) from error
        if observed != physical:
            raise CaptureFragmentValidationError(
                "generation-pinned readback changed the physical receipt"
            )
        return ValidatedCaptureFragmentReceiptV1(
            plan=plan,
            provisional=provisional,
            physical_receipt=observed,
            validator_id=self.validator_id,
            _capability=_VALIDATED_CAPABILITY,
        )


__all__ = [
    "CaptureFragmentValidationError",
    "CaptureFragmentValidatorV1",
    "ValidatedCaptureFragmentReceiptV1",
]
