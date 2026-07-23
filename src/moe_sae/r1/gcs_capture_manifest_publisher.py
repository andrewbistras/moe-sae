"""Publish a sealed capture manifest after all fragment readbacks finish.

Manifest publication is deliberately last.  The publisher writes the
canonical exact-range manifest to a separate attempt-scoped immutable GCS
prefix using the ordinary create-or-verify path, which performs full
generation-pinned readback.  Publication still does not qualify the runtime or
admit the capture to production.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, ClassVar

from .capture_manifest import SealedCaptureManifestV1
from .gcs_immutable_byte_store import (
    MAX_APPLICATION_STREAM_SLICE_BYTES,
    MAX_IMMUTABLE_SNAPSHOT_BYTES,
    GCSImmutableByteStore,
    GCSImmutableByteStoreError,
    GCSPhysicalObjectReceiptV1,
)
from .protocol import CanonicalRecord, canonical_json_bytes


_PUBLISHED_CAPABILITY = object()
_PUBLICATION_CONTRACT = "manifest-last-create-or-verify-full-readback-v1"
_ATTEMPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class GCSCaptureManifestPublicationError(GCSImmutableByteStoreError):
    """A sealed manifest could not become one immutable published object."""


@dataclass(frozen=True, slots=True, init=False)
class PublishedCaptureManifestReceiptV1(CanonicalRecord):
    """Publication proof with production admission still hard false."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.published-capture-manifest-receipt.v1"

    manifest_sha256: str
    manifest_physical_receipt: GCSPhysicalObjectReceiptV1
    data_store_id: str
    publisher_id: str
    fragment_count: int
    token_count: int
    payload_bytes: int
    exact_range_verified: bool
    fragment_full_payload_readback_verified: bool
    manifest_full_payload_readback_verified: bool
    fragments_manifest_visible: bool
    production_sealable: bool
    publication_contract: str

    def __init__(
        self,
        *,
        manifest: SealedCaptureManifestV1,
        manifest_sha256: str,
        manifest_size_bytes: int,
        physical_receipt: GCSPhysicalObjectReceiptV1,
        publisher_id: str,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _PUBLISHED_CAPABILITY:
            raise GCSCaptureManifestPublicationError(
                "published receipts are available only through the GCS publisher"
            )
        if type(manifest) is not SealedCaptureManifestV1:
            raise GCSCaptureManifestPublicationError(
                "published receipt requires an exact sealed manifest"
            )
        if (
            type(manifest_sha256) is not str
            or type(manifest_size_bytes) is not int
            or type(physical_receipt) is not GCSPhysicalObjectReceiptV1
            or type(publisher_id) is not str
            or not publisher_id.startswith("gcs-capture-manifest-publisher-v1:")
            or physical_receipt.sha256 != manifest_sha256
            or physical_receipt.size_bytes != manifest_size_bytes
            or physical_receipt.region != manifest.region
        ):
            raise GCSCaptureManifestPublicationError(
                "published receipt differs from the sealed manifest"
            )
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(self, "manifest_physical_receipt", physical_receipt)
        object.__setattr__(self, "data_store_id", manifest.store_id)
        object.__setattr__(self, "publisher_id", publisher_id)
        object.__setattr__(self, "fragment_count", manifest.fragment_count)
        object.__setattr__(self, "token_count", manifest.token_count)
        object.__setattr__(self, "payload_bytes", manifest.payload_bytes)
        object.__setattr__(self, "exact_range_verified", True)
        object.__setattr__(
            self,
            "fragment_full_payload_readback_verified",
            True,
        )
        object.__setattr__(
            self,
            "manifest_full_payload_readback_verified",
            True,
        )
        object.__setattr__(self, "fragments_manifest_visible", True)
        object.__setattr__(self, "production_sealable", False)
        object.__setattr__(self, "publication_contract", _PUBLICATION_CONTRACT)

    def require_matches(self, manifest: SealedCaptureManifestV1) -> None:
        if (
            type(manifest) is not SealedCaptureManifestV1
            or self.manifest_sha256 != manifest.digest
            or self.data_store_id != manifest.store_id
            or self.fragment_count != manifest.fragment_count
            or self.token_count != manifest.token_count
            or self.payload_bytes != manifest.payload_bytes
            or self.manifest_physical_receipt.region != manifest.region
        ):
            raise GCSCaptureManifestPublicationError(
                "published receipt differs from its exact sealed manifest"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "manifest_sha256": self.manifest_sha256,
            "manifest_physical_receipt": (self.manifest_physical_receipt.as_dict()),
            "data_store_id": self.data_store_id,
            "publisher_id": self.publisher_id,
            "fragment_count": self.fragment_count,
            "token_count": self.token_count,
            "payload_bytes": self.payload_bytes,
            "exact_range_verified": self.exact_range_verified,
            "fragment_full_payload_readback_verified": (
                self.fragment_full_payload_readback_verified
            ),
            "manifest_full_payload_readback_verified": (
                self.manifest_full_payload_readback_verified
            ),
            "fragments_manifest_visible": self.fragments_manifest_visible,
            "production_sealable": self.production_sealable,
            "publication_contract": self.publication_contract,
        }


class GCSCaptureManifestPublisherV1:
    """Immutable attempt-scoped publisher for one or more work manifests."""

    __slots__ = (
        "_attempt_id",
        "_production_identity_qualified",
        "_publisher_id",
        "_store",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__ and hasattr(self, name):
            raise AttributeError(f"{name} is an immutable publisher snapshot")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in self.__slots__:
            raise AttributeError(f"{name} is an immutable publisher snapshot")
        object.__delattr__(self, name)

    def __init__(
        self,
        store: GCSImmutableByteStore,
        *,
        attempt_id: str,
    ) -> None:
        if type(store) is not GCSImmutableByteStore:
            raise GCSCaptureManifestPublicationError(
                "manifest publisher requires an exact immutable byte store"
            )
        if type(attempt_id) is not str or _ATTEMPT_ID.fullmatch(attempt_id) is None:
            raise GCSCaptureManifestPublicationError(
                "attempt_id must be one exact safe path component"
            )
        suffix = f"capture-manifests/attempts/{attempt_id}"
        if store.prefix != suffix and not store.prefix.endswith(f"/{suffix}"):
            raise GCSCaptureManifestPublicationError(
                "manifest store prefix is not scoped to the exact unique attempt"
            )
        if (
            store.maximum_object_bytes != MAX_IMMUTABLE_SNAPSHOT_BYTES
            or store.stream_buffer_bytes != MAX_APPLICATION_STREAM_SLICE_BYTES
        ):
            raise GCSCaptureManifestPublicationError(
                "manifest store does not use the frozen bounded object geometry"
            )
        identity = {
            "schema": "moe-sae.r1.gcs-capture-manifest-publisher-identity.v1",
            "publication_contract": _PUBLICATION_CONTRACT,
            "store_id": store.store_id,
            "store_prefix": store.prefix,
            "store_region": store.region,
            "attempt_id": attempt_id,
        }
        self._store = store
        self._attempt_id = attempt_id
        self._publisher_id = (
            "gcs-capture-manifest-publisher-v1:"
            + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        )
        self._production_identity_qualified = False

    @property
    def publisher_id(self) -> str:
        return self._publisher_id

    @property
    def production_identity_qualified(self) -> bool:
        return self._production_identity_qualified

    def publish(
        self,
        manifest: SealedCaptureManifestV1,
    ) -> PublishedCaptureManifestReceiptV1:
        if type(manifest) is not SealedCaptureManifestV1:
            raise GCSCaptureManifestPublicationError(
                "publisher requires an exact sealed capture manifest"
            )
        if (
            manifest.manifest_complete is not True
            or manifest.manifest_published is not False
            or manifest.production_sealable is not False
            or manifest.region != self._store.region
        ):
            raise GCSCaptureManifestPublicationError(
                "manifest is incomplete, already published, or in another region"
            )
        payload = manifest.canonical_bytes()
        if not 1 <= len(payload) <= self._store.maximum_object_bytes:
            raise GCSCaptureManifestPublicationError(
                "canonical manifest exceeds the bounded immutable object"
            )
        sha256 = hashlib.sha256(payload).hexdigest()
        try:
            physical = self._store.create_content_or_verify(
                payload,
                expected_sha256=sha256,
                expected_size_bytes=len(payload),
            )
        except GCSImmutableByteStoreError as error:
            raise GCSCaptureManifestPublicationError(
                "immutable manifest publication or readback failed"
            ) from error
        return PublishedCaptureManifestReceiptV1(
            manifest=manifest,
            manifest_sha256=sha256,
            manifest_size_bytes=len(payload),
            physical_receipt=physical,
            publisher_id=self.publisher_id,
            _capability=_PUBLISHED_CAPABILITY,
        )


__all__ = [
    "GCSCaptureManifestPublicationError",
    "GCSCaptureManifestPublisherV1",
    "PublishedCaptureManifestReceiptV1",
]
