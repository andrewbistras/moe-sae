"""Fast create-only GCS sink for provisional R1 capture fragments.

This adapter is the only bridge from the asynchronous host writer to
``GCSImmutableByteStore.create_content_provisional_ack``.  It requires a
unique attempt-scoped prefix and wraps the server response in a receipt whose
readback, manifest-visibility, and sealing claims remain hard false.

No upload exception is resolved or adopted here.  A conflict or ambiguous
post-commit error latches the host writer and may leave an unreachable object
under the unique attempt prefix.  An independent later validator must perform
generation-pinned full-payload readback before any manifest can see it.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .capture_data_plane_contract import (
    PINNED_CAPTURE_DATA_PLANE_CONTRACT,
    PINNED_CAPTURE_DATA_PLANE_CONTRACT_SHA256,
    PINNED_FULL_FRAGMENT_BYTES,
)
from .capture_fragment_codec import PackedCaptureFragmentV1
from .capture_host_writer import (
    CaptureHostWriterError,
    ProvisionalCaptureFragmentReceiptV1,
)
from .gcs_immutable_byte_store import (
    GCSImmutableByteStore,
    GCSImmutableByteStoreError,
)
from .protocol import canonical_json_bytes


GCS_PROVISIONAL_CAPTURE_SINK_CONTRACT = (
    "unique-attempt-create-only-server-crc-deferred-readback-v1"
)
_ATTEMPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class GCSProvisionalCaptureSinkError(CaptureHostWriterError):
    """The provisional sink identity or create acknowledgement failed."""


class GCSProvisionalCaptureSinkV1:
    """Immutable attempt-scoped adapter; safe for four concurrent workers."""

    __slots__ = (
        "_attempt_id",
        "_production_identity_qualified",
        "_sink_id",
        "_store",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__ and hasattr(self, name):
            raise AttributeError(f"{name} is an immutable sink snapshot")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in self.__slots__:
            raise AttributeError(f"{name} is an immutable sink snapshot")
        object.__delattr__(self, name)

    def __init__(
        self,
        store: GCSImmutableByteStore,
        *,
        attempt_id: str,
    ) -> None:
        if type(store) is not GCSImmutableByteStore:
            raise GCSProvisionalCaptureSinkError(
                "provisional sink requires an exact GCS immutable byte store"
            )
        if type(attempt_id) is not str or _ATTEMPT_ID.fullmatch(attempt_id) is None:
            raise GCSProvisionalCaptureSinkError(
                "attempt_id must be one exact safe path component"
            )
        suffix = f"attempts/{attempt_id}"
        if store.prefix != suffix and not store.prefix.endswith(f"/{suffix}"):
            raise GCSProvisionalCaptureSinkError(
                "GCS store prefix is not scoped to the exact unique attempt"
            )
        if (
            store.maximum_object_bytes < PINNED_FULL_FRAGMENT_BYTES
            or store.stream_buffer_bytes
            != PINNED_CAPTURE_DATA_PLANE_CONTRACT.maximum_application_slice_bytes
        ):
            raise GCSProvisionalCaptureSinkError(
                "GCS store bounds cannot carry the pinned capture fragment"
            )
        identity = {
            "schema": "moe-sae.r1.gcs-provisional-capture-sink-identity.v1",
            "sink_contract": GCS_PROVISIONAL_CAPTURE_SINK_CONTRACT,
            "capture_data_plane_contract_sha256": (
                PINNED_CAPTURE_DATA_PLANE_CONTRACT_SHA256
            ),
            "store_id": store.store_id,
            "store_prefix": store.prefix,
            "store_region": store.region,
            "attempt_id": attempt_id,
        }
        sink_id = (
            "gcs-provisional-capture-sink-v1:"
            + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        )
        self._store = store
        self._attempt_id = attempt_id
        self._sink_id = sink_id
        # Even an exact production client is only a necessary input.  The
        # adapter remains unqualified until live throughput/chaos/readback.
        self._production_identity_qualified = False

    @property
    def store(self) -> GCSImmutableByteStore:
        return self._store

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    @property
    def sink_id(self) -> str:
        return self._sink_id

    @property
    def distributed_store_identity_qualified(self) -> bool:
        return self._store.distributed_identity_qualified

    @property
    def production_identity_qualified(self) -> bool:
        return self._production_identity_qualified

    def create_provisional(
        self,
        packed: PackedCaptureFragmentV1,
    ) -> ProvisionalCaptureFragmentReceiptV1:
        if type(packed) is not PackedCaptureFragmentV1:
            raise GCSProvisionalCaptureSinkError(
                "create_provisional requires an exact packed fragment"
            )
        if (
            packed.persisted is not False
            or packed.readback_verified is not False
            or packed.production_sealable is not False
        ):
            raise GCSProvisionalCaptureSinkError(
                "packed fragment carries an impossible prior evidence claim"
            )
        try:
            physical = self._store.create_content_provisional_ack(
                packed.payload,
                expected_sha256=packed.sha256,
                expected_size_bytes=len(packed.payload),
            )
            return ProvisionalCaptureFragmentReceiptV1.from_packed(
                packed,
                physical,
            )
        except (GCSImmutableByteStoreError, CaptureHostWriterError) as error:
            raise GCSProvisionalCaptureSinkError(
                "provisional capture create failed without adoption"
            ) from error


__all__ = [
    "GCS_PROVISIONAL_CAPTURE_SINK_CONTRACT",
    "GCSProvisionalCaptureSinkError",
    "GCSProvisionalCaptureSinkV1",
]
