"""Executable, no-list runtime control for one DeepSeek-R1 slice attempt.

The immutable instance-template metadata value is deliberately small: it is
the canonical JSON encoding of :class:`GCSObjectVersionRefV2`, pointing at one
pre-placement :class:`RuntimeLaunchTokenV1`.  The token is created before the
instance template and therefore cannot contain the deployment envelope (the
envelope itself binds that launch reference).  After Compute placement is
observed, the controller publishes one deterministic admission snapshot at a
name derived from the token digest.  This breaks the otherwise circular hash
dependency without object listing or a mutable ``latest`` name.

This module owns the live, JAX-free supervisor semantics:

* generation/SHA-pinned metadata, admission, evidence, and recovery reads;
* one existing :class:`CampaignController` lease and monotonic fence;
* lease renewal and preemption polling immediately before accelerator steps;
* zero-notice one-way step latching;
* checkpoint manifest verification followed by pointer CAS **last**; and
* immutable bounded V2 incident publication on a terminal Spot signal.

The numerical workload and whole-slice collective are fixed injected adapters,
never metadata-derived commands.  A production caller must provide adapters
whose qualification flags are exact ``True``.  The local adapters in tests
exercise the full state machine without pretending to qualify 16-host JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import signal
import threading
import time
from types import TracebackType
from typing import Any, Callable, ClassVar, Mapping, Protocol, Self
import urllib.error
import urllib.request

from .control_store import CASConflict, JSONCASStore, VersionedJSON
from .controller import CampaignController, LeaseUnavailable, StaleAttempt
from .gcs_immutable_byte_store import GCSPhysicalObjectReceiptV1
from .placement_v2 import (
    ControllerPlacementReceiptV2,
    DeploymentEnvelopeV2,
    GCSObjectVersionRefV2,
)
from .preemption import PREEMPTED_METADATA_URL
from .preemption_v2 import (
    AttemptAuthorityV2,
    CASPublishedCheckpointPointerV2,
    FullySealedCheckpointV2,
    IncidentPublicationErrorV2,
    ZeroNoticeSliceIncidentV2,
    ZeroNoticeStepLatchV2,
)
from .prefix_checkpoint_v2 import (
    GenerationPinnedPrefixObjectReader,
    LazyRegionalPrefixCheckpointReaderV1,
    MAX_REGIONAL_ENVELOPE_BYTES,
)
from .protocol import (
    AttemptReceiptV1,
    CanonicalRecord,
    ContentRefV1,
    DeploymentIdentityV1,
    IDENTIFIER_PATTERN,
    ProgressV1,
    ProtocolError,
    SHA256_PATTERN,
    require_integer,
    require_record_object,
    require_string,
)


LAUNCH_METADATA_KEY = "moe-sae-r1-launch-ref"
LAUNCH_METADATA_URL_PREFIX = (
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/"
)
MAX_METADATA_BYTES = 4096
MAX_LAUNCH_TOKEN_BYTES = 64 * 1024
MAX_ADMISSION_BYTES = 2 * 1024 * 1024
MAX_API_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_CHECKPOINT_POINTER_BYTES = 2 * 1024 * 1024
MAX_CHECKPOINT_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_INCIDENT_BYTES = 2 * 1024 * 1024
MAX_CAS_RETRIES = 8
MAX_LEASE_DURATION_MS = 24 * 60 * 60 * 1000

_REGION = re.compile(r"[a-z]+-[a-z0-9]+[0-9]")
_EMAIL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*@[A-Za-z0-9.-]+")
_SERVICE_ACCOUNT_ID = re.compile(r"[0-9]{6,32}")
_TERMINAL_TRIGGERS = frozenset(
    {
        "METADATA_TRUE",
        "METADATA_AMBIGUOUS",
        "LEASE_EXPIRED",
        "SLICE_MEMBER_LOST",
        "VM_TERMINATED",
    }
)


class RuntimeSupervisorError(RuntimeError):
    """A production runtime invariant failed closed."""


class MetadataBoundaryError(RuntimeSupervisorError):
    """The GCE metadata response was not exact and authenticated."""


class LaunchAdmissionError(RuntimeSupervisorError):
    """The launch token/admission/placement chain is inconsistent."""


class CheckpointPublicationError(RuntimeSupervisorError):
    """A sealed checkpoint could not monotonically become recovery authority."""


class RuntimeStopRequested(RuntimeSupervisorError):
    """No additional accelerator step or checkpoint publication is authorized."""


class SliceMemberLost(RuntimeSupervisorError):
    """The fixed whole-slice adapter observed a lost member."""


def _identifier(value: Any, name: str) -> str:
    return require_string(value, name, pattern=IDENTIFIER_PATTERN)


def _sha256(value: Any, name: str) -> str:
    return require_string(value, name, pattern=SHA256_PATTERN)


def _gcs_prefix(value: Any, name: str) -> str:
    result = require_string(value, name).rstrip("/")
    if (
        not result.startswith("gs://")
        or result == "gs://"
        or any(ord(character) < 0x20 for character in result)
    ):
        raise ProtocolError(f"{name} must be a non-empty gs:// prefix")
    bucket, separator, suffix = result[5:].partition("/")
    if not bucket or (
        separator
        and any(part in {"", ".", ".."} for part in suffix.split("/"))
    ):
        raise ProtocolError(f"{name} is not a safe gs:// prefix")
    return result


def _region(value: Any, name: str = "region") -> str:
    return require_string(value, name, pattern=_REGION)


def _service_account_email(value: Any, name: str) -> str:
    return require_string(value, name, pattern=_EMAIL)


def _service_account_id(value: Any, name: str) -> str:
    return require_string(value, name, pattern=_SERVICE_ACCOUNT_ID)


@dataclass(frozen=True, slots=True)
class RuntimeLaunchTokenV1(CanonicalRecord):
    """Pre-template, generation-pinned authority for one runtime attempt."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.runtime-launch-token.v1"

    controller_base_uri: str
    campaign_id: str
    work_id: str
    attempt_id: str
    executor_id: str
    expected_fence: int
    implementation_identity_sha256: str
    controller_service_account_email: str
    controller_service_account_unique_id: str
    controller_challenge: str
    region: str
    regional_envelope_receipt: GCSPhysicalObjectReceiptV1
    expected_logical_manifest_sha256: str
    recovery_checkpoint: CASPublishedCheckpointPointerV2 | None
    lease_duration_ms: int
    renewal_margin_ms: int
    control_poll_interval_ms: int

    def __post_init__(self) -> None:
        base = _gcs_prefix(self.controller_base_uri, "controller_base_uri")
        object.__setattr__(self, "controller_base_uri", base)
        _identifier(self.campaign_id, "campaign_id")
        _identifier(self.work_id, "work_id")
        _identifier(self.attempt_id, "attempt_id")
        _identifier(self.executor_id, "executor_id")
        fence = require_integer(self.expected_fence, "expected_fence", minimum=1)
        _sha256(
            self.implementation_identity_sha256,
            "implementation_identity_sha256",
        )
        _service_account_email(
            self.controller_service_account_email,
            "controller_service_account_email",
        )
        _service_account_id(
            self.controller_service_account_unique_id,
            "controller_service_account_unique_id",
        )
        _sha256(self.controller_challenge, "controller_challenge")
        region = _region(self.region)
        if type(self.regional_envelope_receipt) is not GCSPhysicalObjectReceiptV1:
            raise ProtocolError(
                "regional_envelope_receipt must be GCSPhysicalObjectReceiptV1"
            )
        receipt = self.regional_envelope_receipt
        if receipt.region != region or receipt.content_type != "application/json":
            raise ProtocolError(
                "regional envelope receipt must be JSON in the selected region"
            )
        _sha256(
            self.expected_logical_manifest_sha256,
            "expected_logical_manifest_sha256",
        )
        pointer = self.recovery_checkpoint
        if pointer is not None:
            if type(pointer) is not CASPublishedCheckpointPointerV2:
                raise ProtocolError(
                    "recovery_checkpoint must be a published V2 pointer or null"
                )
            checkpoint = pointer.checkpoint
            if (
                checkpoint.campaign_id != self.campaign_id
                or checkpoint.work_id != self.work_id
                or checkpoint.source_fence >= fence
            ):
                raise ProtocolError(
                    "recovery checkpoint does not precede this exact work fence"
                )
            if not pointer.pointer_object.uri.startswith(self.controller_root_uri):
                raise ProtocolError(
                    "recovery pointer is outside the controller campaign root"
                )
        duration = require_integer(
            self.lease_duration_ms,
            "lease_duration_ms",
            minimum=1,
            maximum=MAX_LEASE_DURATION_MS,
        )
        margin = require_integer(
            self.renewal_margin_ms,
            "renewal_margin_ms",
            minimum=1,
            maximum=duration - 1,
        )
        interval = require_integer(
            self.control_poll_interval_ms,
            "control_poll_interval_ms",
            minimum=1,
            maximum=margin,
        )
        if margin * 2 > duration:
            raise ProtocolError(
                "renewal_margin_ms must not exceed half the lease duration"
            )
        if interval > margin:
            raise ProtocolError(
                "control polling must occur no later than the renewal margin"
            )

    @property
    def controller_root_uri(self) -> str:
        return (
            f"{self.controller_base_uri}/campaigns/{self.campaign_id}/"
        )

    def admission_uri(self, launch_ref: GCSObjectVersionRefV2) -> str:
        if type(launch_ref) is not GCSObjectVersionRefV2:
            raise ProtocolError("launch_ref must be GCSObjectVersionRefV2")
        return (
            f"{self.controller_root_uri}runtime-admissions/"
            f"{launch_ref.sha256}.json"
        )

    def launch_uri(self) -> str:
        return f"{self.controller_root_uri}launch/{self.digest}.json"

    def incident_uri(self) -> str:
        digest = hashlib.sha256(self.attempt_id.encode("ascii")).hexdigest()
        return f"{self.controller_root_uri}attempts/{digest}/incident.json"

    def checkpoint_pointer_uri(self) -> str:
        digest = hashlib.sha256(self.work_id.encode("ascii")).hexdigest()
        return f"{self.controller_root_uri}checkpoints/{digest}/head.json"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "controller_base_uri": self.controller_base_uri,
            "campaign_id": self.campaign_id,
            "work_id": self.work_id,
            "attempt_id": self.attempt_id,
            "executor_id": self.executor_id,
            "expected_fence": self.expected_fence,
            "implementation_identity_sha256": self.implementation_identity_sha256,
            "controller_service_account_email": (
                self.controller_service_account_email
            ),
            "controller_service_account_unique_id": (
                self.controller_service_account_unique_id
            ),
            "controller_challenge": self.controller_challenge,
            "region": self.region,
            "regional_envelope_receipt": self.regional_envelope_receipt.as_dict(),
            "expected_logical_manifest_sha256": (
                self.expected_logical_manifest_sha256
            ),
            "recovery_checkpoint": (
                None
                if self.recovery_checkpoint is None
                else self.recovery_checkpoint.as_dict()
            ),
            "lease_duration_ms": self.lease_duration_ms,
            "renewal_margin_ms": self.renewal_margin_ms,
            "control_poll_interval_ms": self.control_poll_interval_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "controller_base_uri",
            "campaign_id",
            "work_id",
            "attempt_id",
            "executor_id",
            "expected_fence",
            "implementation_identity_sha256",
            "controller_service_account_email",
            "controller_service_account_unique_id",
            "controller_challenge",
            "region",
            "regional_envelope_receipt",
            "expected_logical_manifest_sha256",
            "recovery_checkpoint",
            "lease_duration_ms",
            "renewal_margin_ms",
            "control_poll_interval_ms",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        raw_pointer = obj["recovery_checkpoint"]
        if raw_pointer is not None and not isinstance(raw_pointer, Mapping):
            raise ProtocolError("recovery_checkpoint must be an object or null")
        return cls(
            controller_base_uri=require_string(
                obj["controller_base_uri"], "controller_base_uri"
            ),
            campaign_id=_identifier(obj["campaign_id"], "campaign_id"),
            work_id=_identifier(obj["work_id"], "work_id"),
            attempt_id=_identifier(obj["attempt_id"], "attempt_id"),
            executor_id=_identifier(obj["executor_id"], "executor_id"),
            expected_fence=require_integer(
                obj["expected_fence"], "expected_fence", minimum=1
            ),
            implementation_identity_sha256=_sha256(
                obj["implementation_identity_sha256"],
                "implementation_identity_sha256",
            ),
            controller_service_account_email=_service_account_email(
                obj["controller_service_account_email"],
                "controller_service_account_email",
            ),
            controller_service_account_unique_id=_service_account_id(
                obj["controller_service_account_unique_id"],
                "controller_service_account_unique_id",
            ),
            controller_challenge=_sha256(
                obj["controller_challenge"], "controller_challenge"
            ),
            region=_region(obj["region"]),
            regional_envelope_receipt=GCSPhysicalObjectReceiptV1.from_dict(
                obj["regional_envelope_receipt"]
            ),
            expected_logical_manifest_sha256=_sha256(
                obj["expected_logical_manifest_sha256"],
                "expected_logical_manifest_sha256",
            ),
            recovery_checkpoint=(
                None
                if raw_pointer is None
                else CASPublishedCheckpointPointerV2.from_dict(raw_pointer)
            ),
            lease_duration_ms=require_integer(
                obj["lease_duration_ms"], "lease_duration_ms", minimum=1
            ),
            renewal_margin_ms=require_integer(
                obj["renewal_margin_ms"], "renewal_margin_ms", minimum=1
            ),
            control_poll_interval_ms=require_integer(
                obj["control_poll_interval_ms"],
                "control_poll_interval_ms",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeLaunchAdmissionV1(CanonicalRecord):
    """Post-placement immutable snapshot at the token-derived exact name."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.runtime-launch-admission.v1"

    launch_ref: GCSObjectVersionRefV2
    deployment: DeploymentEnvelopeV2
    placement: ControllerPlacementReceiptV2

    def __post_init__(self) -> None:
        if type(self.launch_ref) is not GCSObjectVersionRefV2:
            raise ProtocolError("launch_ref must be GCSObjectVersionRefV2")
        if type(self.deployment) is not DeploymentEnvelopeV2:
            raise ProtocolError("deployment must be DeploymentEnvelopeV2")
        if type(self.placement) is not ControllerPlacementReceiptV2:
            raise ProtocolError("placement must be ControllerPlacementReceiptV2")
        if self.deployment.launch_ref != self.launch_ref:
            raise ProtocolError("deployment does not bind the exact launch reference")
        self.placement.verify_deployment(self.deployment)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "launch_ref": self.launch_ref.as_dict(),
            "deployment": self.deployment.as_dict(),
            "placement": self.placement.as_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = require_record_object(
            value,
            schema=cls.SCHEMA,
            fields=("launch_ref", "deployment", "placement"),
        )
        return cls(
            launch_ref=GCSObjectVersionRefV2.from_dict(obj["launch_ref"]),
            deployment=DeploymentEnvelopeV2.from_dict(obj["deployment"]),
            placement=ControllerPlacementReceiptV2.from_dict(obj["placement"]),
        )


def publish_runtime_launch_token(
    store: JSONCASStore,
    token: RuntimeLaunchTokenV1,
) -> GCSObjectVersionRefV2:
    """Create the pre-template token once at its content-addressed exact name."""

    if type(token) is not RuntimeLaunchTokenV1:
        raise TypeError("token must be RuntimeLaunchTokenV1")
    encoded = token.canonical_bytes()
    if len(encoded) > MAX_LAUNCH_TOKEN_BYTES:
        raise LaunchAdmissionError("runtime launch token exceeds its byte bound")
    version = store.put_immutable(token.launch_uri(), token)
    reference = _version_ref(version)
    if reference.sha256 != token.digest:
        raise LaunchAdmissionError("published launch token identity changed")
    return reference


def publish_runtime_launch_admission(
    store: JSONCASStore,
    token: RuntimeLaunchTokenV1,
    admission: RuntimeLaunchAdmissionV1,
) -> VersionedJSON:
    """Publish the post-placement admission once at the token-derived name."""

    if type(token) is not RuntimeLaunchTokenV1:
        raise TypeError("token must be RuntimeLaunchTokenV1")
    if type(admission) is not RuntimeLaunchAdmissionV1:
        raise TypeError("admission must be RuntimeLaunchAdmissionV1")
    if admission.launch_ref.uri != token.launch_uri():
        raise LaunchAdmissionError("admission launch reference names another token")
    encoded = admission.canonical_bytes()
    if len(encoded) > MAX_ADMISSION_BYTES:
        raise LaunchAdmissionError("runtime admission exceeds its byte bound")
    return store.put_immutable(
        token.admission_uri(admission.launch_ref),
        admission,
    )


@dataclass(frozen=True, slots=True)
class MetadataHTTPResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes

    def __post_init__(self) -> None:
        if type(self.status) is not int:
            raise TypeError("metadata status must be an integer")
        if not isinstance(self.headers, Mapping) or any(
            type(key) is not str or type(value) is not str
            for key, value in self.headers.items()
        ):
            raise TypeError("metadata headers must be a string mapping")
        if type(self.body) is not bytes:
            raise TypeError("metadata body must be exact bytes")


MetadataTransport = Callable[[str, int, float], MetadataHTTPResponse]


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def _metadata_transport(
    url: str,
    maximum_bytes: int,
    timeout_seconds: float,
) -> MetadataHTTPResponse:
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _RejectRedirects(),
    )
    request = urllib.request.Request(
        url,
        headers={"Metadata-Flavor": "Google"},
        method="GET",
    )
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            body = response.read(maximum_bytes + 1)
            headers = {str(key): str(value) for key, value in response.headers.items()}
            return MetadataHTTPResponse(
                status=int(response.status),
                headers=headers,
                body=body,
            )
    except urllib.error.HTTPError as error:
        body = error.read(maximum_bytes + 1)
        headers = {str(key): str(value) for key, value in error.headers.items()}
        return MetadataHTTPResponse(int(error.code), headers, body)
    except (OSError, ValueError) as error:
        raise MetadataBoundaryError("GCE metadata request failed") from error


class GCEMetadataClientV1:
    """Exact fixed-URL GCE metadata reader; values are data, never commands."""

    def __init__(
        self,
        *,
        transport: MetadataTransport | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0.0
        ):
            raise ValueError("metadata timeout must be finite and positive")
        self._transport = transport or _metadata_transport
        self._timeout = float(timeout_seconds)

    def _get(self, url: str, *, maximum_bytes: int) -> bytes:
        response = self._transport(url, maximum_bytes, self._timeout)
        flavor = next(
            (
                value
                for key, value in response.headers.items()
                if key.lower() == "metadata-flavor"
            ),
            None,
        )
        if (
            response.status != 200
            or flavor != "Google"
            or len(response.body) > maximum_bytes
        ):
            raise MetadataBoundaryError(
                "metadata response is unauthenticated, non-200, or oversized"
            )
        return response.body

    def read_launch_reference(self, metadata_key: str) -> GCSObjectVersionRefV2:
        if metadata_key != LAUNCH_METADATA_KEY:
            raise MetadataBoundaryError("runtime launch metadata key is not fixed")
        raw = self._get(
            LAUNCH_METADATA_URL_PREFIX + LAUNCH_METADATA_KEY,
            maximum_bytes=MAX_METADATA_BYTES,
        )
        try:
            reference = GCSObjectVersionRefV2.from_json(raw)
        except (ProtocolError, ValueError) as error:
            raise MetadataBoundaryError(
                "launch metadata is not a canonical generation-pinned reference"
            ) from error
        if raw != reference.canonical_bytes():
            raise MetadataBoundaryError("launch metadata is not canonical JSON")
        return reference

    def preempted(self) -> bool:
        raw = self._get(PREEMPTED_METADATA_URL, maximum_bytes=5)
        if raw == b"FALSE":
            return False
        if raw == b"TRUE":
            return True
        raise MetadataBoundaryError("preemption metadata is not exact TRUE/FALSE")


class VersionPinnedByteReader(Protocol):
    def read_exact(
        self,
        reference: GCSObjectVersionRefV2,
        *,
        maximum_bytes: int,
    ) -> bytes: ...


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise LaunchAdmissionError("version-pinned object is not in GCS")
    bucket, separator, name = uri[5:].partition("/")
    if (
        not separator
        or not bucket
        or not name
        or name.endswith("/")
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise LaunchAdmissionError("version-pinned GCS object URI is unsafe")
    return bucket, name


class GCSVersionPinnedByteReaderV1:
    """One exact-generation GCS GET with no listing or latest discovery."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        project: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if client is None:
            try:
                from google.cloud import storage
            except ImportError as error:
                raise RuntimeSupervisorError(
                    "GCS pinned reads require google-cloud-storage"
                ) from error
            client = storage.Client(project=project)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0.0
        ):
            raise ValueError("GCS timeout must be finite and positive")
        self._client = client
        self._timeout = float(timeout_seconds)

    def read_exact(
        self,
        reference: GCSObjectVersionRefV2,
        *,
        maximum_bytes: int,
    ) -> bytes:
        if type(reference) is not GCSObjectVersionRefV2:
            raise TypeError("reference must be GCSObjectVersionRefV2")
        if (
            type(maximum_bytes) is not int
            or maximum_bytes <= 0
            or reference.size_bytes > maximum_bytes
        ):
            raise LaunchAdmissionError("version-pinned read exceeds its bound")
        bucket, name = _parse_gcs_uri(reference.uri)
        blob = self._client.bucket(bucket).blob(
            name,
            generation=reference.generation,
        )
        try:
            blob.reload(
                if_generation_match=reference.generation,
                timeout=self._timeout,
            )
            if (
                type(blob.generation) not in {int, str}
                or int(blob.generation) != reference.generation
                or isinstance(blob.size, bool)
                or int(blob.size) != reference.size_bytes
            ):
                raise LaunchAdmissionError(
                    "GCS object metadata differs from its pinned reference"
                )
            value = blob.download_as_bytes(
                if_generation_match=reference.generation,
                checksum="crc32c",
                timeout=self._timeout,
            )
        except LaunchAdmissionError:
            raise
        except Exception as error:
            raise LaunchAdmissionError(
                "generation-pinned GCS read failed"
            ) from error
        if type(value) is not bytes:
            raise LaunchAdmissionError("GCS reader returned non-bytes")
        try:
            reference.verify_bytes(value)
        except ProtocolError as error:
            raise LaunchAdmissionError(
                "GCS bytes differ from their generation-pinned reference"
            ) from error
        return value


@dataclass(frozen=True, slots=True)
class ResolvedRuntimeLaunchV1:
    launch_ref: GCSObjectVersionRefV2
    launch_token: RuntimeLaunchTokenV1
    admission_version: VersionedJSON
    admission: RuntimeLaunchAdmissionV1


class RuntimeLaunchResolverV1:
    """Resolve exactly one metadata token and its deterministic admission."""

    def __init__(
        self,
        *,
        metadata: GCEMetadataClientV1,
        pinned_reader: VersionPinnedByteReader,
        control_store: JSONCASStore,
        allowed_controller_base_uri: str,
        trusted_controller_service_account_email: str,
        trusted_controller_service_account_unique_id: str,
    ) -> None:
        if not callable(getattr(metadata, "read_launch_reference", None)):
            raise TypeError("metadata adapter lacks read_launch_reference")
        if not callable(getattr(pinned_reader, "read_exact", None)):
            raise TypeError("pinned_reader lacks read_exact")
        if not callable(getattr(control_store, "read", None)):
            raise TypeError("control_store lacks read")
        self._metadata = metadata
        self._reader = pinned_reader
        self._store = control_store
        try:
            self._allowed_base = _gcs_prefix(
                allowed_controller_base_uri,
                "allowed_controller_base_uri",
            )
            self._trusted_controller_email = _service_account_email(
                trusted_controller_service_account_email,
                "trusted_controller_service_account_email",
            )
            self._trusted_controller_id = _service_account_id(
                trusted_controller_service_account_unique_id,
                "trusted_controller_service_account_unique_id",
            )
        except ProtocolError as error:
            raise ValueError("trusted controller authority is malformed") from error

    @property
    def metadata(self) -> GCEMetadataClientV1:
        return self._metadata

    def resolve(
        self,
        metadata_key: str = LAUNCH_METADATA_KEY,
    ) -> ResolvedRuntimeLaunchV1:
        launch_ref = self._metadata.read_launch_reference(metadata_key)
        token_raw = self._reader.read_exact(
            launch_ref,
            maximum_bytes=MAX_LAUNCH_TOKEN_BYTES,
        )
        try:
            token = RuntimeLaunchTokenV1.from_json(token_raw)
        except (ProtocolError, ValueError) as error:
            raise LaunchAdmissionError("runtime launch token is malformed") from error
        if token_raw != token.canonical_bytes():
            raise LaunchAdmissionError("runtime launch token is not canonical")
        if (
            token.controller_base_uri != self._allowed_base
            or token.controller_service_account_email
            != self._trusted_controller_email
            or token.controller_service_account_unique_id
            != self._trusted_controller_id
        ):
            raise LaunchAdmissionError(
                "launch token is outside the fixed controller trust boundary"
            )
        if launch_ref.uri != token.launch_uri():
            raise LaunchAdmissionError(
                "launch token object is not at its content-addressed campaign name"
            )
        admission_uri = token.admission_uri(launch_ref)
        version = self._store.read(admission_uri)
        if version is None:
            raise LaunchAdmissionError("runtime admission snapshot is absent")
        if len(version.value) > MAX_ADMISSION_BYTES:
            raise LaunchAdmissionError("runtime admission exceeds its byte bound")
        try:
            admission = RuntimeLaunchAdmissionV1.from_json(version.value)
        except (ProtocolError, ValueError) as error:
            raise LaunchAdmissionError(
                "runtime admission snapshot is malformed"
            ) from error
        if version.value != admission.canonical_bytes():
            raise LaunchAdmissionError("runtime admission is not canonical")
        deployment = admission.deployment
        if (
            admission.launch_ref != launch_ref
            or deployment.controller_root_uri != token.controller_root_uri
            or deployment.slice_attempt_id != token.attempt_id
            or deployment.region != token.region
            or token.regional_envelope_receipt.region != deployment.region
        ):
            raise LaunchAdmissionError(
                "admission does not bind token attempt, root, and regional input"
            )
        try:
            admission.placement.verify_authority(
                controller_service_account_email=(
                    token.controller_service_account_email
                ),
                controller_service_account_unique_id=(
                    token.controller_service_account_unique_id
                ),
                controller_challenge=token.controller_challenge,
            )
            evidence = self._reader.read_exact(
                admission.placement.api_evidence,
                maximum_bytes=MAX_API_EVIDENCE_BYTES,
            )
            admission.placement.api_evidence.verify_bytes(evidence)
        except (ProtocolError, RuntimeSupervisorError) as error:
            raise LaunchAdmissionError(
                "controller placement authority/evidence failed verification"
            ) from error
        return ResolvedRuntimeLaunchV1(
            launch_ref=launch_ref,
            launch_token=token,
            admission_version=version,
            admission=admission,
        )


class RegionalInputResolver(Protocol):
    @property
    def production_qualified(self) -> bool: ...

    def open(
        self,
        receipt: GCSPhysicalObjectReceiptV1,
        *,
        expected_logical_manifest_sha256: str,
    ) -> object: ...


class PrefixRegionalInputResolverV1:
    """Open the existing two-control-read regional prefix checkpoint."""

    def __init__(
        self,
        store: GenerationPinnedPrefixObjectReader,
        *,
        maximum_read_bytes: int,
    ) -> None:
        if (
            type(maximum_read_bytes) is not int
            or maximum_read_bytes <= 0
        ):
            raise ValueError("maximum_read_bytes must be positive")
        self._store = store
        self._maximum_read_bytes = maximum_read_bytes

    @property
    def production_qualified(self) -> bool:
        return self._store.distributed_identity_qualified is True

    def open(
        self,
        receipt: GCSPhysicalObjectReceiptV1,
        *,
        expected_logical_manifest_sha256: str,
    ) -> LazyRegionalPrefixCheckpointReaderV1:
        return LazyRegionalPrefixCheckpointReaderV1.open(
            self._store,
            receipt,
            expected_logical_manifest_sha256=(
                expected_logical_manifest_sha256
            ),
            maximum_control_bytes=MAX_REGIONAL_ENVELOPE_BYTES,
            maximum_read_bytes=self._maximum_read_bytes,
        )


class WholeSliceRuntime(Protocol):
    """The fixed collective seam; metadata never chooses its implementation."""

    @property
    def production_qualified(self) -> bool: ...

    def initialize_device_inventory_sha256(
        self,
        admission: RuntimeLaunchAdmissionV1,
    ) -> str: ...

    def terminate(
        self,
        *,
        outcome: str,
        incident_sha256: str | None,
    ) -> None: ...


class ImmutableWorkload(Protocol):
    """Fixed checkpoint-scale workload; no executable value comes from metadata."""

    @property
    def total_units(self) -> int: ...

    def restore(
        self,
        *,
        regional_input: object,
        recovery_checkpoint: FullySealedCheckpointV2 | None,
        recovery_manifest_bytes: bytes | None,
    ) -> None: ...

    def run_unit(
        self,
        unit_index: int,
        context: RuntimeStepContextV1,
    ) -> FullySealedCheckpointV2: ...

    def successful_receipt(
        self,
        *,
        authority: AttemptAuthorityV2,
        final_checkpoint: CASPublishedCheckpointPointerV2,
        final_progress_sequence: int,
        finished_at_unix_ms: int,
    ) -> AttemptReceiptV1: ...


def _version_ref(version: VersionedJSON) -> GCSObjectVersionRefV2:
    return GCSObjectVersionRefV2(
        uri=version.uri,
        generation=version.generation,
        size_bytes=len(version.value),
        sha256=version.sha256,
    )


def verify_recovery_checkpoint(
    pointer: CASPublishedCheckpointPointerV2 | None,
    *,
    token: RuntimeLaunchTokenV1,
    pinned_reader: VersionPinnedByteReader,
) -> bytes | None:
    """Verify both the published pointer generation and its sealed manifest."""

    if pointer is None:
        return None
    checkpoint = pointer.checkpoint
    if (
        checkpoint.campaign_id != token.campaign_id
        or checkpoint.work_id != token.work_id
        or checkpoint.source_fence >= token.expected_fence
        or pointer.pointer_object.uri != token.checkpoint_pointer_uri()
    ):
        raise CheckpointPublicationError(
            "recovery pointer is not the exact predecessor authority"
        )
    pointer_bytes = pinned_reader.read_exact(
        pointer.pointer_object,
        maximum_bytes=MAX_CHECKPOINT_POINTER_BYTES,
    )
    try:
        pointer.verify_bytes(pointer_bytes)
    except ProtocolError as error:
        raise CheckpointPublicationError(
            "recovery pointer bytes differ from published authority"
        ) from error
    manifest = pinned_reader.read_exact(
        checkpoint.manifest,
        maximum_bytes=MAX_CHECKPOINT_MANIFEST_BYTES,
    )
    try:
        checkpoint.manifest.verify_bytes(manifest)
    except ProtocolError as error:
        raise CheckpointPublicationError(
            "recovery manifest differs from its sealed generation"
        ) from error
    return manifest


class CASCheckpointPointerPublisherV2:
    """Publish an already-sealed checkpoint through one monotonic CAS head."""

    def __init__(
        self,
        store: JSONCASStore,
        *,
        token: RuntimeLaunchTokenV1,
        pinned_reader: VersionPinnedByteReader,
        clock_ms: Callable[[], int],
        max_cas_retries: int = MAX_CAS_RETRIES,
    ) -> None:
        self._store = store
        self._token = token
        self._reader = pinned_reader
        self._clock = clock_ms
        self._retries = require_integer(
            max_cas_retries,
            "max_cas_retries",
            minimum=1,
            maximum=MAX_CAS_RETRIES,
        )

    def publish(
        self,
        authority: AttemptAuthorityV2,
        checkpoint: FullySealedCheckpointV2,
    ) -> CASPublishedCheckpointPointerV2:
        if type(authority) is not AttemptAuthorityV2:
            raise TypeError("authority must be AttemptAuthorityV2")
        if type(checkpoint) is not FullySealedCheckpointV2:
            raise TypeError("checkpoint must be FullySealedCheckpointV2")
        lease = authority.lease
        if (
            checkpoint.campaign_id != lease.campaign_id
            or checkpoint.work_id != lease.work_id
            or checkpoint.source_attempt_id != lease.attempt_id
            or checkpoint.source_fence != lease.fence
        ):
            raise CheckpointPublicationError(
                "sealed checkpoint does not belong to the active attempt/fence"
            )
        now = require_integer(self._clock(), "clock_ms", minimum=0)
        if (
            checkpoint.sealed_at_unix_ms > now
            or now >= lease.expires_at_unix_ms
        ):
            raise CheckpointPublicationError(
                "checkpoint publication requires an unexpired lease after sealing"
            )
        # A workload may claim FULLY_SEALED only after publishing its exact
        # manifest.  Verify that generation before making the mutable pointer
        # authoritative.
        manifest_bytes = self._reader.read_exact(
            checkpoint.manifest,
            maximum_bytes=MAX_CHECKPOINT_MANIFEST_BYTES,
        )
        try:
            checkpoint.manifest.verify_bytes(manifest_bytes)
        except ProtocolError as error:
            raise CheckpointPublicationError(
                "sealed checkpoint manifest differs from its exact generation"
            ) from error
        uri = self._token.checkpoint_pointer_uri()
        encoded = checkpoint.canonical_bytes()
        if len(encoded) > MAX_CHECKPOINT_POINTER_BYTES:
            raise CheckpointPublicationError("checkpoint pointer payload is oversized")

        attempted_expected_generation: int | None = None
        for _attempt in range(self._retries):
            current = self._store.read(uri)
            expected_generation = 0
            if current is not None:
                expected_generation = current.generation
                if len(current.value) > MAX_CHECKPOINT_POINTER_BYTES:
                    raise CheckpointPublicationError(
                        "existing checkpoint pointer is oversized"
                    )
                try:
                    previous = FullySealedCheckpointV2.from_json(current.value)
                except (ProtocolError, ValueError) as error:
                    raise CheckpointPublicationError(
                        "existing checkpoint pointer is malformed"
                    ) from error
                if (
                    previous.campaign_id != checkpoint.campaign_id
                    or previous.work_id != checkpoint.work_id
                ):
                    raise CheckpointPublicationError(
                        "checkpoint pointer identity changed"
                    )
                if current.value == encoded:
                    if (
                        attempted_expected_generation is not None
                        and current.generation > attempted_expected_generation
                    ):
                        return CASPublishedCheckpointPointerV2(
                            checkpoint=checkpoint,
                            pointer_object=_version_ref(current),
                            cas_expected_generation=(
                                attempted_expected_generation
                            ),
                            published_at_unix_ms=now,
                            publication_state="CAS_PUBLISHED",
                        )
                    raise CheckpointPublicationError(
                        "checkpoint is already published; reuse its exact "
                        "CASPublishedCheckpointPointerV2 authority"
                    )
                if previous.completed_steps >= checkpoint.completed_steps:
                    raise CheckpointPublicationError(
                        "checkpoint completed_steps did not increase"
                    )
                if previous.source_fence > checkpoint.source_fence:
                    raise CheckpointPublicationError(
                        "checkpoint pointer would move to an older fence"
                    )
                if (
                    previous.source_fence == checkpoint.source_fence
                    and previous.source_attempt_id
                    != checkpoint.source_attempt_id
                ):
                    raise CheckpointPublicationError(
                        "checkpoint pointer has two attempts at one fence"
                    )
            attempted_expected_generation = expected_generation
            try:
                if current is None:
                    version = self._store.create(uri, checkpoint)
                else:
                    version = self._store.compare_and_swap(
                        uri,
                        current.generation,
                        checkpoint,
                    )
            except CASConflict:
                continue
            if version.generation <= expected_generation:
                raise CheckpointPublicationError(
                    "checkpoint CAS did not advance generation"
                )
            return CASPublishedCheckpointPointerV2(
                checkpoint=checkpoint,
                pointer_object=_version_ref(version),
                cas_expected_generation=expected_generation,
                published_at_unix_ms=now,
                publication_state="CAS_PUBLISHED",
            )
        raise CheckpointPublicationError(
            f"checkpoint CAS failed after {self._retries} attempts"
        )


@dataclass(frozen=True, slots=True)
class RuntimeSupervisorResultV1:
    outcome: str
    authority: AttemptAuthorityV2
    completed_units: int
    final_checkpoint: CASPublishedCheckpointPointerV2 | None
    incident: ZeroNoticeSliceIncidentV2 | None
    receipt: AttemptReceiptV1 | None

    def __post_init__(self) -> None:
        if self.outcome not in {"SUCCEEDED", "PREEMPTED"}:
            raise ValueError("runtime result outcome is invalid")
        if type(self.authority) is not AttemptAuthorityV2:
            raise TypeError("runtime result authority is invalid")
        require_integer(self.completed_units, "completed_units", minimum=0)
        if self.outcome == "SUCCEEDED":
            if self.final_checkpoint is None or self.receipt is None:
                raise ValueError("successful result requires checkpoint and receipt")
            if self.incident is not None:
                raise ValueError("successful result cannot contain an incident")
        else:
            if self.incident is None or self.receipt is not None:
                raise ValueError("preempted result requires only an incident")


class RuntimeStepContextV1:
    """Workload-visible gate called immediately before every accelerator step."""

    def __init__(self, supervisor: RuntimeSupervisorV1) -> None:
        self._supervisor = supervisor

    @property
    def stop_requested(self) -> bool:
        return self._supervisor.stop_requested

    @property
    def authority(self) -> AttemptAuthorityV2:
        return self._supervisor._require_authority()

    def authorize_step(self) -> int:
        return self._supervisor._authorize_step()


class _SignalScope:
    def __init__(self, stop: Callable[[str], None]) -> None:
        self._stop = stop
        self._previous: Any = None
        self._installed = False

    def __enter__(self) -> _SignalScope:
        if threading.current_thread() is threading.main_thread():
            self._previous = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._handle)
            self._installed = True
        return self

    def _handle(self, signum: int, frame: Any) -> None:
        del frame
        if signum == signal.SIGTERM:
            self._stop("VM_TERMINATED")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self._installed:
            signal.signal(signal.SIGTERM, self._previous)


class RuntimeSupervisorV1:
    """Run checkpoint-scale immutable units under one fenced slice lease."""

    def __init__(
        self,
        *,
        resolver: RuntimeLaunchResolverV1,
        control_store: JSONCASStore,
        pinned_reader: VersionPinnedByteReader,
        regional_input_resolver: RegionalInputResolver,
        slice_runtime: WholeSliceRuntime,
        workload: ImmutableWorkload,
        content_verifier: Callable[[ContentRefV1], bool],
        clock_ms: Callable[[], int] | None = None,
        monotonic_ms: Callable[[], int] | None = None,
    ) -> None:
        if regional_input_resolver.production_qualified is not True:
            raise RuntimeSupervisorError(
                "regional input resolver is not production-qualified"
            )
        if slice_runtime.production_qualified is not True:
            raise RuntimeSupervisorError(
                "whole-slice runtime is not production-qualified"
            )
        if not callable(content_verifier):
            raise TypeError("content_verifier must be callable")
        self._resolver = resolver
        self._store = control_store
        self._reader = pinned_reader
        self._input_resolver = regional_input_resolver
        self._slice = slice_runtime
        self._workload = workload
        self._content_verifier = content_verifier
        self._clock = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._monotonic = monotonic_ms or (lambda: time.monotonic_ns() // 1_000_000)
        self._stop = threading.Event()
        self._stop_lock = threading.RLock()
        self._stop_trigger: str | None = None
        self._authority: AttemptAuthorityV2 | None = None
        self._latch: ZeroNoticeStepLatchV2 | None = None
        self._controller: CampaignController | None = None
        self._token: RuntimeLaunchTokenV1 | None = None
        self._last_control_poll_ms: int | None = None
        self._last_checkpoint: CASPublishedCheckpointPointerV2 | None = None
        self._metadata: GCEMetadataClientV1 | None = None

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def request_stop(self, trigger_kind: str) -> None:
        if trigger_kind not in _TERMINAL_TRIGGERS:
            raise ValueError("unsupported runtime stop trigger")
        with self._stop_lock:
            if self._stop_trigger is None:
                self._stop_trigger = trigger_kind
                self._stop.set()

    def _incident_sink(self, incident: ZeroNoticeSliceIncidentV2) -> None:
        token = self._require_token()
        encoded = incident.canonical_bytes()
        if len(encoded) > MAX_INCIDENT_BYTES:
            raise RuntimeSupervisorError("terminal incident exceeds its byte bound")
        self._store.put_immutable(token.incident_uri(), incident)

    def _require_token(self) -> RuntimeLaunchTokenV1:
        if self._token is None:
            raise RuntimeSupervisorError("runtime launch is not resolved")
        return self._token

    def _require_authority(self) -> AttemptAuthorityV2:
        if self._authority is None:
            raise RuntimeSupervisorError("runtime attempt has no lease authority")
        return self._authority

    def _poll_controls(self) -> None:
        token = self._require_token()
        now_mono = require_integer(
            self._monotonic(),
            "monotonic_ms",
            minimum=0,
        )
        if (
            self._last_control_poll_ms is not None
            and now_mono < self._last_control_poll_ms
        ):
            self.request_stop("METADATA_AMBIGUOUS")
            raise RuntimeStopRequested("monotonic control clock regressed")
        if (
            self._last_control_poll_ms is not None
            and now_mono - self._last_control_poll_ms
            < token.control_poll_interval_ms
        ):
            return
        self._last_control_poll_ms = now_mono
        if self._metadata is not None:
            try:
                if self._metadata.preempted():
                    self.request_stop("METADATA_TRUE")
            except MetadataBoundaryError:
                self.request_stop("METADATA_AMBIGUOUS")
        if self.stop_requested:
            raise RuntimeStopRequested("runtime control gate is closed")

    def _maybe_renew(self) -> None:
        token = self._require_token()
        authority = self._require_authority()
        now = require_integer(self._clock(), "clock_ms", minimum=0)
        lease = authority.lease
        if now >= lease.expires_at_unix_ms:
            self.request_stop("LEASE_EXPIRED")
            raise RuntimeStopRequested("attempt lease expired")
        if now < lease.expires_at_unix_ms - token.renewal_margin_ms:
            return
        assert self._controller is not None
        try:
            renewed = self._controller.renew(
                lease,
                lease_duration_ms=token.lease_duration_ms,
            )
        except (LeaseUnavailable, StaleAttempt):
            self.request_stop("LEASE_EXPIRED")
            raise RuntimeStopRequested("attempt lease renewal failed") from None
        self._authority = AttemptAuthorityV2(
            lease=renewed,
            deployment_identity=authority.deployment_identity,
            deployment=authority.deployment,
        )

    def _authorize_step(self) -> int:
        if self.stop_requested:
            raise RuntimeStopRequested("runtime stop was requested")
        self._poll_controls()
        self._maybe_renew()
        # Linearize the final stop observation and the local permit against
        # request_stop().  This matters when a fixed workload drives steps
        # from a helper thread.  On the main thread, temporarily masking
        # SIGTERM also prevents Python's re-entrant signal handler from
        # interrupting the lock-protected permit operation; any pending signal
        # is delivered as soon as the prior mask is restored.
        previous_mask: set[signal.Signals] | None = None
        if threading.current_thread() is threading.main_thread():
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                {signal.SIGTERM},
            )
        try:
            with self._stop_lock:
                if self.stop_requested:
                    raise RuntimeStopRequested("runtime stop was requested")
                latch = self._latch
                if latch is None:
                    raise RuntimeSupervisorError(
                        "runtime step latch is unavailable"
                    )
                permit = latch.authorize_new_step(self._require_authority())
        finally:
            if previous_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        if self.stop_requested:
            raise RuntimeStopRequested("runtime stop raced accelerator permit")
        return permit

    def _terminal_incident(self) -> ZeroNoticeSliceIncidentV2:
        trigger = self._stop_trigger or "SLICE_MEMBER_LOST"
        latch = self._latch
        if latch is None:
            raise RuntimeSupervisorError("cannot terminalize before step latch")
        try:
            return latch.latch(
                self._require_authority(),
                trigger_kind=trigger,
                recovery_checkpoint=self._last_checkpoint,
            )
        except IncidentPublicationErrorV2 as error:
            try:
                latch.retry_incident_publication()
            except IncidentPublicationErrorV2 as retry_error:
                raise RuntimeSupervisorError(
                    "terminal gate closed but bounded incident publication failed"
                ) from retry_error
            return error.incident

    def run(
        self,
        *,
        metadata_key: str = LAUNCH_METADATA_KEY,
    ) -> RuntimeSupervisorResultV1:
        resolved = self._resolver.resolve(metadata_key)
        token = resolved.launch_token
        self._token = token
        self._metadata = self._resolver.metadata
        total_units = require_integer(
            self._workload.total_units,
            "workload.total_units",
            minimum=1,
        )

        inventory_sha256 = _sha256(
            self._slice.initialize_device_inventory_sha256(resolved.admission),
            "device_inventory_sha256",
        )
        deployment = resolved.admission.deployment
        identity = DeploymentIdentityV1(
            implementation_identity_sha256=token.implementation_identity_sha256,
            deployment_envelope_sha256=deployment.digest,
            device_inventory_sha256=inventory_sha256,
            attempt_id=token.attempt_id,
        )
        controller = CampaignController(
            self._store,
            base_uri=token.controller_base_uri,
            campaign_id=token.campaign_id,
            clock_ms=self._clock,
            content_verifier=self._content_verifier,
        )
        self._controller = controller
        before = controller.view().work(token.work_id)
        if (
            before.status != "PENDING"
            or before.lease is not None
            or before.fence + 1 != token.expected_fence
        ):
            raise LaunchAdmissionError(
                "campaign state cannot acquire the token's exact next fence"
            )
        recovery = token.recovery_checkpoint
        recovered_units = (
            0 if recovery is None else recovery.checkpoint.completed_steps
        )
        if recovered_units > total_units:
            raise LaunchAdmissionError(
                "recovery checkpoint exceeds workload total units"
            )
        if (
            before.completed_units is not None
            and before.completed_units > recovered_units
        ):
            raise LaunchAdmissionError(
                "controller progress is ahead of sealed recovery authority"
            )
        if before.total_units is not None and before.total_units != total_units:
            raise LaunchAdmissionError("workload total units changed across attempts")

        lease = controller.acquire(
            token.work_id,
            attempt_id=token.attempt_id,
            executor_id=token.executor_id,
            deployment_identity_sha256=identity.digest,
            lease_duration_ms=token.lease_duration_ms,
        )
        if lease.fence != token.expected_fence:
            raise LaunchAdmissionError("controller acquired an unexpected fence")
        authority = AttemptAuthorityV2(
            lease=lease,
            deployment_identity=identity,
            deployment=deployment,
        )
        self._authority = authority
        self._last_checkpoint = recovery
        self._latch = ZeroNoticeStepLatchV2(
            authority,
            incident_sink=self._incident_sink,
        )
        pointer_publisher = CASCheckpointPointerPublisherV2(
            self._store,
            token=token,
            pinned_reader=self._reader,
            clock_ms=self._clock,
        )
        completed = recovered_units
        sequence = before.last_progress_sequence or 0
        context = RuntimeStepContextV1(self)
        try:
            with _SignalScope(self.request_stop):
                regional_input = self._input_resolver.open(
                    token.regional_envelope_receipt,
                    expected_logical_manifest_sha256=(
                        token.expected_logical_manifest_sha256
                    ),
                )
                recovery_manifest = verify_recovery_checkpoint(
                    recovery,
                    token=token,
                    pinned_reader=self._reader,
                )
                self._workload.restore(
                    regional_input=regional_input,
                    recovery_checkpoint=(
                        None if recovery is None else recovery.checkpoint
                    ),
                    recovery_manifest_bytes=recovery_manifest,
                )
                controller_completed = before.completed_units or 0
                if recovered_units > controller_completed:
                    self._poll_controls()
                    self._maybe_renew()
                    authority = self._require_authority()
                    sequence += 1
                    try:
                        controller.record_progress(
                            ProgressV1(
                                work_id=token.work_id,
                                attempt_id=authority.attempt_id,
                                fence=authority.fence,
                                sequence=sequence,
                                phase="R1_RUNTIME_RECOVERY_ADOPTED",
                                completed_units=recovered_units,
                                total_units=total_units,
                                updated_at_unix_ms=require_integer(
                                    self._clock(), "clock_ms", minimum=0
                                ),
                            )
                        )
                    except StaleAttempt:
                        self.request_stop("LEASE_EXPIRED")
                        raise RuntimeStopRequested(
                            "lease became stale while adopting recovery progress"
                        ) from None
                while completed < total_units:
                    if self.stop_requested:
                        raise RuntimeStopRequested("stop requested at unit boundary")
                    # Close known control-plane failures before entering the
                    # fixed workload without claiming an accelerator permit.
                    self._poll_controls()
                    self._maybe_renew()
                    try:
                        checkpoint = self._workload.run_unit(completed, context)
                    except RuntimeStopRequested:
                        raise
                    except SliceMemberLost:
                        self.request_stop("SLICE_MEMBER_LOST")
                        raise RuntimeStopRequested("slice member was lost") from None
                    # A signal that races unit completion never starts a save.
                    # The sealed candidate may remain immutable diagnostic data.
                    if self.stop_requested:
                        raise RuntimeStopRequested(
                            "stop raced immutable unit completion"
                        )
                    self._maybe_renew()
                    authority = self._require_authority()
                    if (
                        type(checkpoint) is not FullySealedCheckpointV2
                        or checkpoint.completed_steps != completed + 1
                    ):
                        raise RuntimeSupervisorError(
                            "workload returned a non-monotonic sealed checkpoint"
                        )
                    # A publication already in flight when SIGTERM arrives may
                    # finish.  Its CAS result is the recovery authority, so
                    # mirror the pointer before observing the terminal latch.
                    pointer = pointer_publisher.publish(authority, checkpoint)
                    self._last_checkpoint = pointer
                    completed = pointer.checkpoint.completed_steps
                    if self.stop_requested:
                        raise RuntimeStopRequested(
                            "stop raced checkpoint pointer publication"
                        )
                    sequence += 1
                    try:
                        controller.record_progress(
                            ProgressV1(
                                work_id=token.work_id,
                                attempt_id=authority.attempt_id,
                                fence=authority.fence,
                                sequence=sequence,
                                phase="R1_RUNTIME_UNIT_SEALED",
                                completed_units=completed,
                                total_units=total_units,
                                updated_at_unix_ms=require_integer(
                                    self._clock(), "clock_ms", minimum=0
                                ),
                            )
                        )
                    except StaleAttempt:
                        self.request_stop("LEASE_EXPIRED")
                        raise RuntimeStopRequested(
                            "lease became stale after checkpoint publication"
                        ) from None
                final_checkpoint = self._last_checkpoint
                if final_checkpoint is None:
                    raise RuntimeSupervisorError(
                        "successful workload has no checkpoint"
                    )
                authority = self._require_authority()
                if self.stop_requested:
                    raise RuntimeStopRequested(
                        "stop requested before successful receipt"
                    )
                receipt = self._workload.successful_receipt(
                    authority=authority,
                    final_checkpoint=final_checkpoint,
                    final_progress_sequence=sequence,
                    finished_at_unix_ms=require_integer(
                        self._clock(), "clock_ms", minimum=0
                    ),
                )
                if self.stop_requested:
                    raise RuntimeStopRequested(
                        "stop raced successful receipt construction"
                    )
                committed = controller.commit(receipt)
        except RuntimeStopRequested:
            try:
                incident = self._terminal_incident()
            except Exception:
                self._slice.terminate(outcome="FAILED", incident_sha256=None)
                raise
            authority = self._require_authority()
            self._slice.terminate(
                outcome="PREEMPTED",
                incident_sha256=incident.digest,
            )
            return RuntimeSupervisorResultV1(
                outcome="PREEMPTED",
                authority=authority,
                completed_units=completed,
                final_checkpoint=self._last_checkpoint,
                incident=incident,
                receipt=None,
            )
        except Exception:
            self._slice.terminate(outcome="FAILED", incident_sha256=None)
            raise

        self._slice.terminate(outcome="SUCCEEDED", incident_sha256=None)
        return RuntimeSupervisorResultV1(
            outcome="SUCCEEDED",
            authority=authority,
            completed_units=completed,
            final_checkpoint=final_checkpoint,
            incident=None,
            receipt=committed,
        )


__all__ = [
    "CASCheckpointPointerPublisherV2",
    "CheckpointPublicationError",
    "GCEMetadataClientV1",
    "GCSVersionPinnedByteReaderV1",
    "ImmutableWorkload",
    "LAUNCH_METADATA_KEY",
    "LaunchAdmissionError",
    "MetadataBoundaryError",
    "MetadataHTTPResponse",
    "PrefixRegionalInputResolverV1",
    "RegionalInputResolver",
    "ResolvedRuntimeLaunchV1",
    "RuntimeLaunchAdmissionV1",
    "RuntimeLaunchResolverV1",
    "RuntimeLaunchTokenV1",
    "RuntimeStepContextV1",
    "RuntimeStopRequested",
    "RuntimeSupervisorError",
    "RuntimeSupervisorResultV1",
    "RuntimeSupervisorV1",
    "SliceMemberLost",
    "VersionPinnedByteReader",
    "WholeSliceRuntime",
    "publish_runtime_launch_admission",
    "publish_runtime_launch_token",
    "verify_recovery_checkpoint",
]
