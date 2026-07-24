"""Fail-closed v6e-64 placement records for the DeepSeek-R1 controller.

This additive protocol intentionally does not reinterpret
``DeploymentEnvelopeV1``.  Version 2 describes the deployment shape used by
the preemptible DeepSeek-R1 campaign: one *zonal* v6e-64 slice represented by
one 16-VM managed instance group (MIG), created and replaced atomically.

The module has no cloud-client dependency.  A privileged controller is
responsible for reading authenticated Compute and IAM API responses (including
``instances.get``, ``disks.get``, and source-image resolution), storing the raw
response bundle in a generation-pinned GCS object, and constructing a
``ControllerPlacementReceiptV2`` from those responses.  Guest metadata and
host attestations are deliberately not placement authority.

Google API field names are not guessed here.  In particular,
``target_size_policy == "BULK"`` is the protocol's normalized value for the
documented full-slice target-size operation.  A future cloud adapter must map
an observed API response to that value explicitly and fail on an unknown API
shape.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, ClassVar, Mapping, Self

from .image_protocol import ImagePublicationV1
from .protocol import (
    CanonicalRecord,
    ProtocolError,
    SHA256_PATTERN,
    require_integer,
    require_record_object,
    require_string,
)


_GCE_ID_MAX = (1 << 64) - 1
_GCE_PREFIX = "https://www.googleapis.com/compute/v1/projects/"
_GCE_NAME = re.compile(r"[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?")
_PROJECT = re.compile(r"[a-z][a-z0-9-]{0,61}[a-z0-9]")
_REGION = re.compile(r"[a-z]+-[a-z0-9]+[0-9]")
_ZONE = re.compile(r"[a-z]+-[a-z0-9]+[0-9]-[a-z][a-z0-9]*")
_SERVICE_ACCOUNT = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._+-]*@[A-Za-z0-9.-]+\.gserviceaccount\.com"
)
_SERVICE_ACCOUNT_UNIQUE_ID = re.compile(r"[0-9]{6,32}")
_ATTEMPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_LAUNCH_METADATA_KEY = "moe-sae-r1-launch-ref"

_BASE_IMAGE_PROJECT = "ubuntu-os-accelerator-images"
_BASE_IMAGE_FAMILY = "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e"
_TPU_LICENSE = (
    "https://www.googleapis.com/compute/v1/projects/ubuntu-os-cloud/"
    "global/licenses/ubuntu-2204-lts-tpu"
)


def _sha256(value: Any, name: str) -> str:
    return require_string(value, name, pattern=SHA256_PATTERN)


def _gce_id(value: Any, name: str) -> int:
    return require_integer(value, name, minimum=1, maximum=_GCE_ID_MAX)


def _gce_name(value: Any, name: str) -> str:
    return require_string(value, name, pattern=_GCE_NAME)


def _project(value: Any, name: str = "project") -> str:
    return require_string(value, name, pattern=_PROJECT)


def _region(value: Any, name: str = "region") -> str:
    return require_string(value, name, pattern=_REGION)


def _zone(value: Any, name: str = "zone") -> str:
    return require_string(value, name, pattern=_ZONE)


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ProtocolError(f"{name} must be a boolean")
    return value


def _string_array(
    value: Any, name: str, *, count: int | None = None
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ProtocolError(f"{name} must be an array")
    result = tuple(require_string(item, f"{name}[]") for item in value)
    if count is not None and len(result) != count:
        raise ProtocolError(f"{name} must contain exactly {count} entries")
    if not result:
        raise ProtocolError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ProtocolError(f"{name} must not contain duplicates")
    return result


def _zone_region(zone: str) -> str:
    return zone.rsplit("-", 1)[0]


def _global_link(
    value: Any,
    name: str,
    *,
    project: str,
    collection: str,
    resource_name: str,
) -> str:
    link = require_string(value, name)
    expected = f"{_GCE_PREFIX}{project}/global/{collection}/{resource_name}"
    if link != expected:
        raise ProtocolError(f"{name} must equal canonical GCE selfLink {expected!r}")
    return link


def _scoped_link(
    value: Any,
    name: str,
    *,
    project: str,
    scope: str,
    location: str,
    collection: str,
    resource_name: str,
) -> str:
    link = require_string(value, name)
    expected = f"{_GCE_PREFIX}{project}/{scope}/{location}/{collection}/{resource_name}"
    if link != expected:
        raise ProtocolError(f"{name} must equal canonical GCE selfLink {expected!r}")
    return link


def _parse_image_link(value: Any, name: str) -> tuple[str, str, str]:
    link = require_string(value, name)
    match = re.fullmatch(
        re.escape(_GCE_PREFIX)
        + rf"(?P<project>{_PROJECT.pattern})/global/images/"
        + rf"(?P<image>{_GCE_NAME.pattern})",
        link,
    )
    if match is None:
        raise ProtocolError(f"{name} must be a canonical GCE image selfLink")
    return link, match.group("project"), match.group("image")


def _parse_license_link(value: Any, name: str) -> str:
    link = require_string(value, name)
    match = re.fullmatch(
        re.escape(_GCE_PREFIX)
        + rf"{_PROJECT.pattern}/global/licenses/{_GCE_NAME.pattern}",
        link,
    )
    if match is None:
        raise ProtocolError(f"{name} must be a canonical GCE license selfLink")
    return link


def _gcs_object_uri(value: Any, name: str) -> str:
    uri = require_string(value, name)
    match = re.fullmatch(
        r"gs://(?P<bucket>[a-z0-9][a-z0-9._-]{1,220}[a-z0-9])/"
        r"(?P<object>[^?#\x00-\x1f\x7f]+)",
        uri,
    )
    if match is None:
        raise ProtocolError(f"{name} must be an absolute gs:// object URI")
    bucket = match.group("bucket")
    object_name = match.group("object")
    if (
        ".." in bucket
        or ".-" in bucket
        or "-." in bucket
        or "//" in object_name
        or object_name.endswith("/")
        or any(part in {".", ".."} for part in object_name.split("/"))
    ):
        raise ProtocolError(f"{name} must be a normalized gs:// object URI")
    return uri


def _controller_root(value: Any) -> str:
    uri = require_string(value, "controller_root_uri")
    if not uri.endswith("/"):
        raise ProtocolError("controller_root_uri must end with '/'")
    # Validate the prefix by appending a sentinel object name.
    _gcs_object_uri(f"{uri}sentinel", "controller_root_uri")
    return uri


def _service_account_email(value: Any, name: str) -> str:
    return require_string(value, name, pattern=_SERVICE_ACCOUNT)


def _service_account_unique_id(value: Any, name: str) -> str:
    # IAM unique IDs are strings even when they contain decimal digits.  This
    # prevents JSON decoders and languages from losing precision.
    return require_string(value, name, pattern=_SERVICE_ACCOUNT_UNIQUE_ID)


@dataclass(frozen=True, slots=True)
class GCSObjectVersionRefV2(CanonicalRecord):
    """One immutable object generation containing authenticated raw evidence."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.gcs-object-version-ref.v2"

    uri: str
    generation: int
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _gcs_object_uri(self.uri, "uri")
        require_integer(self.generation, "generation", minimum=1)
        require_integer(self.size_bytes, "size_bytes", minimum=1)
        _sha256(self.sha256, "sha256")

    def verify_bytes(self, value: bytes) -> None:
        """Verify bytes fetched with an external ``generation`` precondition."""

        if not isinstance(value, bytes):
            raise ProtocolError("evidence value must be bytes")
        if len(value) != self.size_bytes:
            raise ProtocolError("evidence byte size does not match reference")
        if hashlib.sha256(value).hexdigest() != self.sha256:
            raise ProtocolError("evidence SHA-256 does not match reference")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "uri": self.uri,
            "generation": self.generation,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = require_record_object(
            value,
            schema=cls.SCHEMA,
            fields=("uri", "generation", "size_bytes", "sha256"),
        )
        return cls(
            uri=_gcs_object_uri(obj["uri"], "uri"),
            generation=require_integer(obj["generation"], "generation", minimum=1),
            size_bytes=require_integer(obj["size_bytes"], "size_bytes", minimum=1),
            sha256=_sha256(obj["sha256"], "sha256"),
        )


@dataclass(frozen=True, slots=True)
class BaseImageFamilyResolutionReceiptV2(CanonicalRecord):
    """Controller receipt resolving the certified v6e family exactly once.

    The family alias is never a Packer input after this receipt is accepted;
    the builder consumes the dated image numeric ID and selfLink.  The raw
    authenticated ``images.getFromFamily``/``images.get`` response bundle is
    retained through ``api_evidence``.
    """

    SCHEMA: ClassVar[str] = "moe-sae.r1.base-image-family-resolution-receipt.v2"

    controller_service_account_email: str
    controller_service_account_unique_id: str
    controller_challenge: str
    api_evidence: GCSObjectVersionRefV2
    family_project: str
    family_name: str
    family_alias: str
    resolved_image_name: str
    resolved_image_id: int
    resolved_image_self_link: str
    resolved_status: str
    architecture: str
    license_self_links: tuple[str, ...]
    observed_at_unix_ms: int

    def __post_init__(self) -> None:
        _service_account_email(
            self.controller_service_account_email,
            "controller_service_account_email",
        )
        _service_account_unique_id(
            self.controller_service_account_unique_id,
            "controller_service_account_unique_id",
        )
        _sha256(self.controller_challenge, "controller_challenge")
        if type(self.api_evidence) is not GCSObjectVersionRefV2:
            raise ProtocolError("api_evidence must be GCSObjectVersionRefV2")
        project = _project(self.family_project, "family_project")
        family = _gce_name(self.family_name, "family_name")
        if project != _BASE_IMAGE_PROJECT or family != _BASE_IMAGE_FAMILY:
            raise ProtocolError(
                "base image family must be the certified Ubuntu 22.04 v5e/v5p/v6e family"
            )
        expected_alias = f"{_GCE_PREFIX}{project}/global/images/family/{family}"
        if self.family_alias != expected_alias:
            raise ProtocolError(
                f"family_alias must equal canonical family alias {expected_alias!r}"
            )
        image_name = _gce_name(self.resolved_image_name, "resolved_image_name")
        if re.fullmatch(re.escape(family) + r"-v[0-9]{8}", image_name) is None:
            raise ProtocolError("resolved_image_name must be a dated family image")
        _gce_id(self.resolved_image_id, "resolved_image_id")
        _, image_project, linked_name = _parse_image_link(
            self.resolved_image_self_link,
            "resolved_image_self_link",
        )
        if image_project != project or linked_name != image_name:
            raise ProtocolError(
                "resolved image selfLink does not match family project/name"
            )
        if self.resolved_status != "READY":
            raise ProtocolError("resolved_status must be READY")
        if self.architecture != "X86_64":
            raise ProtocolError("architecture must be X86_64")
        licenses = _string_array(self.license_self_links, "license_self_links")
        for index, link in enumerate(licenses):
            _parse_license_link(link, f"license_self_links[{index}]")
        if _TPU_LICENSE not in licenses:
            raise ProtocolError(
                "resolved image must carry the Ubuntu 22.04 TPU license"
            )
        object.__setattr__(self, "license_self_links", tuple(sorted(licenses)))
        require_integer(self.observed_at_unix_ms, "observed_at_unix_ms", minimum=0)

    def verify_authority(
        self,
        *,
        controller_service_account_email: str,
        controller_service_account_unique_id: str,
        controller_challenge: str,
    ) -> None:
        """Bind the self-described receipt to controller-precommitted authority."""

        expected = {
            "controller_service_account_email": _service_account_email(
                controller_service_account_email,
                "controller_service_account_email",
            ),
            "controller_service_account_unique_id": _service_account_unique_id(
                controller_service_account_unique_id,
                "controller_service_account_unique_id",
            ),
            "controller_challenge": _sha256(
                controller_challenge,
                "controller_challenge",
            ),
        }
        for field, expected_value in expected.items():
            if getattr(self, field) != expected_value:
                raise ProtocolError(
                    f"base image receipt {field} does not match authority"
                )

    def verify_admission(
        self,
        *,
        controller_service_account_email: str,
        controller_service_account_unique_id: str,
        controller_challenge: str,
        api_evidence_bytes: bytes,
    ) -> None:
        """Atomically check external authority and exact fetched evidence bytes."""

        self.verify_authority(
            controller_service_account_email=controller_service_account_email,
            controller_service_account_unique_id=controller_service_account_unique_id,
            controller_challenge=controller_challenge,
        )
        self.api_evidence.verify_bytes(api_evidence_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "controller_service_account_email": self.controller_service_account_email,
            "controller_service_account_unique_id": self.controller_service_account_unique_id,
            "controller_challenge": self.controller_challenge,
            "api_evidence": self.api_evidence.as_dict(),
            "family_project": self.family_project,
            "family_name": self.family_name,
            "family_alias": self.family_alias,
            "resolved_image_name": self.resolved_image_name,
            "resolved_image_id": self.resolved_image_id,
            "resolved_image_self_link": self.resolved_image_self_link,
            "resolved_status": self.resolved_status,
            "architecture": self.architecture,
            "license_self_links": list(self.license_self_links),
            "observed_at_unix_ms": self.observed_at_unix_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "controller_service_account_email",
            "controller_service_account_unique_id",
            "controller_challenge",
            "api_evidence",
            "family_project",
            "family_name",
            "family_alias",
            "resolved_image_name",
            "resolved_image_id",
            "resolved_image_self_link",
            "resolved_status",
            "architecture",
            "license_self_links",
            "observed_at_unix_ms",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        raw_licenses = obj["license_self_links"]
        if not isinstance(raw_licenses, list):
            raise ProtocolError("license_self_links must be an array")
        return cls(
            controller_service_account_email=_service_account_email(
                obj["controller_service_account_email"],
                "controller_service_account_email",
            ),
            controller_service_account_unique_id=_service_account_unique_id(
                obj["controller_service_account_unique_id"],
                "controller_service_account_unique_id",
            ),
            controller_challenge=_sha256(
                obj["controller_challenge"], "controller_challenge"
            ),
            api_evidence=GCSObjectVersionRefV2.from_dict(obj["api_evidence"]),
            family_project=_project(obj["family_project"], "family_project"),
            family_name=_gce_name(obj["family_name"], "family_name"),
            family_alias=require_string(obj["family_alias"], "family_alias"),
            resolved_image_name=_gce_name(
                obj["resolved_image_name"], "resolved_image_name"
            ),
            resolved_image_id=_gce_id(obj["resolved_image_id"], "resolved_image_id"),
            resolved_image_self_link=require_string(
                obj["resolved_image_self_link"], "resolved_image_self_link"
            ),
            resolved_status=require_string(obj["resolved_status"], "resolved_status"),
            architecture=require_string(obj["architecture"], "architecture"),
            license_self_links=tuple(raw_licenses),
            observed_at_unix_ms=require_integer(
                obj["observed_at_unix_ms"], "observed_at_unix_ms", minimum=0
            ),
        )


@dataclass(frozen=True, slots=True)
class DeploymentEnvelopeV2(CanonicalRecord):
    """Exact policy for one fresh zonal v6e-64 Spot MIG attempt.

    ``vm_count`` counts the 16 Compute Engine TPU VM members (and eventual
    JAX processes), not physical hosts.  Google may place multiple TPU VMs on
    one physical host; this protocol makes no physical-host-count claim.
    """

    SCHEMA: ClassVar[str] = "moe-sae.r1.deployment-envelope.v2"

    base_image_resolution_sha256: str
    image_publication_sha256: str
    project: str
    region: str
    zone: str
    slice_attempt_id: str
    accelerator_type: str
    machine_type: str
    machine_type_self_link: str
    topology: str
    vm_count: int
    chips_per_vm: int
    chip_count: int
    image_name: str
    image_id: int
    image_self_link: str
    instance_template_name: str
    instance_template_id: int
    instance_template_self_link: str
    resource_policy_name: str
    resource_policy_id: int
    resource_policy_self_link: str
    mig_name: str
    mig_id: int
    mig_self_link: str
    mig_scope: str
    target_size: int
    target_size_policy: str
    default_action_on_failure: str
    replacement_policy: str
    provisioning_model: str
    termination_action: str
    maintenance_policy: str
    service_account_email: str
    service_account_unique_id: str
    controller_root_uri: str
    launch_ref: GCSObjectVersionRefV2

    def __post_init__(self) -> None:
        _sha256(self.base_image_resolution_sha256, "base_image_resolution_sha256")
        _sha256(self.image_publication_sha256, "image_publication_sha256")
        project = _project(self.project)
        region = _region(self.region)
        zone = _zone(self.zone)
        if _zone_region(zone) != region:
            raise ProtocolError("zone must belong to region")
        require_string(self.slice_attempt_id, "slice_attempt_id", pattern=_ATTEMPT_ID)
        if self.accelerator_type != "v6e-64":
            raise ProtocolError("accelerator_type must be v6e-64")
        if self.machine_type != "ct6e-standard-4t":
            raise ProtocolError("machine_type must be ct6e-standard-4t")
        _scoped_link(
            self.machine_type_self_link,
            "machine_type_self_link",
            project=project,
            scope="zones",
            location=zone,
            collection="machineTypes",
            resource_name=self.machine_type,
        )
        if self.topology != "8x8":
            raise ProtocolError("topology must be 8x8")
        if require_integer(self.vm_count, "vm_count", minimum=1) != 16:
            raise ProtocolError("vm_count must be 16")
        if require_integer(self.chips_per_vm, "chips_per_vm", minimum=1) != 4:
            raise ProtocolError("chips_per_vm must be 4")
        if require_integer(self.chip_count, "chip_count", minimum=1) != 64:
            raise ProtocolError("chip_count must be 64")
        if self.vm_count * self.chips_per_vm != self.chip_count:
            raise ProtocolError("VM and chip counts are inconsistent")
        image_name = _gce_name(self.image_name, "image_name")
        _gce_id(self.image_id, "image_id")
        _, image_project, linked_image = _parse_image_link(
            self.image_self_link, "image_self_link"
        )
        if image_project != project or linked_image != image_name:
            raise ProtocolError("image selfLink does not match deployment project/name")
        template = _gce_name(self.instance_template_name, "instance_template_name")
        _gce_id(self.instance_template_id, "instance_template_id")
        _global_link(
            self.instance_template_self_link,
            "instance_template_self_link",
            project=project,
            collection="instanceTemplates",
            resource_name=template,
        )
        policy = _gce_name(self.resource_policy_name, "resource_policy_name")
        _gce_id(self.resource_policy_id, "resource_policy_id")
        _scoped_link(
            self.resource_policy_self_link,
            "resource_policy_self_link",
            project=project,
            scope="regions",
            location=region,
            collection="resourcePolicies",
            resource_name=policy,
        )
        mig = _gce_name(self.mig_name, "mig_name")
        _gce_id(self.mig_id, "mig_id")
        _scoped_link(
            self.mig_self_link,
            "mig_self_link",
            project=project,
            scope="zones",
            location=zone,
            collection="instanceGroupManagers",
            resource_name=mig,
        )
        if self.mig_scope != "ZONAL":
            raise ProtocolError("mig_scope must be ZONAL in deployment envelope v2")
        if require_integer(self.target_size, "target_size", minimum=0) != 16:
            raise ProtocolError("target_size must equal the full 16-VM slice")
        if self.target_size_policy != "BULK":
            raise ProtocolError("target_size_policy must be BULK")
        if self.default_action_on_failure != "DO_NOTHING":
            raise ProtocolError("default_action_on_failure must be DO_NOTHING")
        if self.replacement_policy != "FRESH_WHOLE_MIG":
            raise ProtocolError("replacement_policy must be FRESH_WHOLE_MIG")
        if self.provisioning_model != "SPOT":
            raise ProtocolError("provisioning_model must be SPOT")
        if self.termination_action != "STOP":
            raise ProtocolError("termination_action must be STOP")
        if self.maintenance_policy != "TERMINATE":
            raise ProtocolError("maintenance_policy must be TERMINATE")
        _service_account_email(self.service_account_email, "service_account_email")
        _service_account_unique_id(
            self.service_account_unique_id, "service_account_unique_id"
        )
        _controller_root(self.controller_root_uri)
        if type(self.launch_ref) is not GCSObjectVersionRefV2:
            raise ProtocolError("launch_ref must be GCSObjectVersionRefV2")
        if not self.launch_ref.uri.startswith(self.controller_root_uri):
            raise ProtocolError("launch_ref is outside the deployment controller root")

    def verify_base_image_resolution(
        self, resolution: BaseImageFamilyResolutionReceiptV2
    ) -> None:
        if type(resolution) is not BaseImageFamilyResolutionReceiptV2:
            raise ProtocolError("resolution must be BaseImageFamilyResolutionReceiptV2")
        if resolution.digest != self.base_image_resolution_sha256:
            raise ProtocolError(
                "base image resolution digest does not match deployment"
            )
        if not resolution.api_evidence.uri.startswith(self.controller_root_uri):
            raise ProtocolError(
                "base image API evidence is outside the deployment controller root"
            )

    def verify_publication(self, publication: ImagePublicationV1) -> None:
        if type(publication) is not ImagePublicationV1:
            raise ProtocolError("publication must be ImagePublicationV1")
        checks = {
            "digest": self.image_publication_sha256,
            "image_name": self.image_name,
            "image_id": self.image_id,
            "image_self_link": self.image_self_link,
        }
        for field, expected in checks.items():
            actual = (
                publication.digest if field == "digest" else getattr(publication, field)
            )
            if actual != expected:
                raise ProtocolError(f"publication {field} does not match deployment")

    def verify_image_chain(
        self,
        resolution: BaseImageFamilyResolutionReceiptV2,
        publication: ImagePublicationV1,
    ) -> None:
        """Bind family resolution, custom image publication, and deployment."""

        self.verify_base_image_resolution(resolution)
        self.verify_publication(publication)
        if publication.base_image_id != resolution.resolved_image_id:
            raise ProtocolError(
                "published image base numeric ID does not match family resolution"
            )
        if publication.base_image_self_link != resolution.resolved_image_self_link:
            raise ProtocolError(
                "published image base selfLink does not match family resolution"
            )

    def verify_fresh_replacement_of(self, previous: DeploymentEnvelopeV2) -> None:
        """Prove this envelope names a new whole-MIG attempt, not a resize/reuse.

        Cross-attempt ordering is controller state, not a Compute API property;
        this method therefore checks identity separation and an unchanged slice
        contract.  Region, zone, instance-template identity, regional placement
        policy identity, and the zone-scoped machine-type selfLink may change
        during global Spot failover.  The controller must call this before
        accepting a successor.
        """

        if type(previous) is not DeploymentEnvelopeV2:
            raise ProtocolError("previous must be DeploymentEnvelopeV2")
        stable_fields = (
            "base_image_resolution_sha256",
            "image_publication_sha256",
            "project",
            "accelerator_type",
            "machine_type",
            "topology",
            "vm_count",
            "chips_per_vm",
            "chip_count",
            "image_name",
            "image_id",
            "image_self_link",
            "mig_scope",
            "target_size",
            "target_size_policy",
            "default_action_on_failure",
            "replacement_policy",
            "provisioning_model",
            "termination_action",
            "maintenance_policy",
            "service_account_email",
            "service_account_unique_id",
            "controller_root_uri",
        )
        for field in stable_fields:
            if getattr(self, field) != getattr(previous, field):
                raise ProtocolError(f"replacement changed immutable field {field}")
        if self.slice_attempt_id == previous.slice_attempt_id:
            raise ProtocolError("replacement must use a fresh slice_attempt_id")
        if self.mig_name == previous.mig_name:
            raise ProtocolError("replacement must use a fresh MIG name")
        if self.mig_id == previous.mig_id:
            raise ProtocolError("replacement must use a fresh MIG numeric ID")
        if self.mig_self_link == previous.mig_self_link:
            raise ProtocolError("replacement must use a fresh MIG selfLink")
        if self.launch_ref == previous.launch_ref:
            raise ProtocolError("replacement must use a fresh launch_ref")
        if self.instance_template_name == previous.instance_template_name:
            raise ProtocolError("replacement must use a fresh instance template name")
        if self.instance_template_id == previous.instance_template_id:
            raise ProtocolError(
                "replacement must use a fresh instance template numeric ID"
            )
        if self.instance_template_self_link == previous.instance_template_self_link:
            raise ProtocolError(
                "replacement must use a fresh instance template selfLink"
            )

    def as_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "launch_ref": self.launch_ref.as_dict()} | {
            field: getattr(self, field)
            for field in (
                "base_image_resolution_sha256",
                "image_publication_sha256",
                "project",
                "region",
                "zone",
                "slice_attempt_id",
                "accelerator_type",
                "machine_type",
                "machine_type_self_link",
                "topology",
                "vm_count",
                "chips_per_vm",
                "chip_count",
                "image_name",
                "image_id",
                "image_self_link",
                "instance_template_name",
                "instance_template_id",
                "instance_template_self_link",
                "resource_policy_name",
                "resource_policy_id",
                "resource_policy_self_link",
                "mig_name",
                "mig_id",
                "mig_self_link",
                "mig_scope",
                "target_size",
                "target_size_policy",
                "default_action_on_failure",
                "replacement_policy",
                "provisioning_model",
                "termination_action",
                "maintenance_policy",
                "service_account_email",
                "service_account_unique_id",
                "controller_root_uri",
            )
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "base_image_resolution_sha256",
            "image_publication_sha256",
            "project",
            "region",
            "zone",
            "slice_attempt_id",
            "accelerator_type",
            "machine_type",
            "machine_type_self_link",
            "topology",
            "vm_count",
            "chips_per_vm",
            "chip_count",
            "image_name",
            "image_id",
            "image_self_link",
            "instance_template_name",
            "instance_template_id",
            "instance_template_self_link",
            "resource_policy_name",
            "resource_policy_id",
            "resource_policy_self_link",
            "mig_name",
            "mig_id",
            "mig_self_link",
            "mig_scope",
            "target_size",
            "target_size_policy",
            "default_action_on_failure",
            "replacement_policy",
            "provisioning_model",
            "termination_action",
            "maintenance_policy",
            "service_account_email",
            "service_account_unique_id",
            "controller_root_uri",
            "launch_ref",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        string_fields = {
            field: require_string(obj[field], field)
            for field in fields
            if field
            not in {
                "vm_count",
                "chips_per_vm",
                "chip_count",
                "image_id",
                "instance_template_id",
                "resource_policy_id",
                "mig_id",
                "target_size",
                "launch_ref",
            }
        }
        return cls(
            **string_fields,
            vm_count=require_integer(obj["vm_count"], "vm_count", minimum=1),
            chips_per_vm=require_integer(
                obj["chips_per_vm"], "chips_per_vm", minimum=1
            ),
            chip_count=require_integer(obj["chip_count"], "chip_count", minimum=1),
            image_id=_gce_id(obj["image_id"], "image_id"),
            instance_template_id=_gce_id(
                obj["instance_template_id"], "instance_template_id"
            ),
            resource_policy_id=_gce_id(obj["resource_policy_id"], "resource_policy_id"),
            mig_id=_gce_id(obj["mig_id"], "mig_id"),
            target_size=require_integer(obj["target_size"], "target_size", minimum=0),
            launch_ref=GCSObjectVersionRefV2.from_dict(obj["launch_ref"]),
        )


@dataclass(frozen=True, slots=True)
class InstanceTemplateObservationV2(CanonicalRecord):
    """Controller-normalized observation of the exact instance template.

    The cloud adapter must parse the sole metadata value as canonical
    ``GCSObjectVersionRefV2`` JSON.  ``metadata_keys`` is retained explicitly
    so an added ``startup-script``/``startup-script-url`` or any other metadata
    key cannot disappear during normalization.
    """

    SCHEMA: ClassVar[str] = "moe-sae.r1.instance-template-observation.v2"

    project: str
    zone: str
    name: str
    resource_id: int
    self_link: str
    machine_type: str
    machine_type_self_link: str
    image_name: str
    image_id: int
    image_self_link: str
    provisioning_model: str
    termination_action: str
    maintenance_policy: str
    automatic_restart: bool
    service_account_email: str
    metadata_keys: tuple[str, ...]
    launch_ref: GCSObjectVersionRefV2

    def __post_init__(self) -> None:
        project = _project(self.project)
        zone = _zone(self.zone)
        name = _gce_name(self.name, "name")
        _gce_id(self.resource_id, "resource_id")
        _global_link(
            self.self_link,
            "self_link",
            project=project,
            collection="instanceTemplates",
            resource_name=name,
        )
        if self.machine_type != "ct6e-standard-4t":
            raise ProtocolError("template machine_type must be ct6e-standard-4t")
        _scoped_link(
            self.machine_type_self_link,
            "machine_type_self_link",
            project=project,
            scope="zones",
            location=zone,
            collection="machineTypes",
            resource_name=self.machine_type,
        )
        image_name = _gce_name(self.image_name, "image_name")
        _gce_id(self.image_id, "image_id")
        _, image_project, linked_name = _parse_image_link(
            self.image_self_link, "image_self_link"
        )
        if image_project != project or linked_name != image_name:
            raise ProtocolError("template image selfLink does not match project/name")
        if self.provisioning_model != "SPOT":
            raise ProtocolError("template provisioning_model must be SPOT")
        if self.termination_action != "STOP":
            raise ProtocolError("template termination_action must be STOP")
        if self.maintenance_policy != "TERMINATE":
            raise ProtocolError("template maintenance_policy must be TERMINATE")
        if _boolean(self.automatic_restart, "automatic_restart"):
            raise ProtocolError("template automatic_restart must be false")
        _service_account_email(self.service_account_email, "service_account_email")
        metadata_keys = _string_array(self.metadata_keys, "metadata_keys")
        if metadata_keys != (_LAUNCH_METADATA_KEY,):
            raise ProtocolError(
                "template metadata must contain only moe-sae-r1-launch-ref; "
                "startup scripts and arbitrary metadata are forbidden"
            )
        object.__setattr__(self, "metadata_keys", metadata_keys)
        if type(self.launch_ref) is not GCSObjectVersionRefV2:
            raise ProtocolError("template launch_ref must be GCSObjectVersionRefV2")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "metadata_keys": list(self.metadata_keys),
            "launch_ref": self.launch_ref.as_dict(),
        } | {
            field: getattr(self, field)
            for field in (
                "project",
                "zone",
                "name",
                "resource_id",
                "self_link",
                "machine_type",
                "machine_type_self_link",
                "image_name",
                "image_id",
                "image_self_link",
                "provisioning_model",
                "termination_action",
                "maintenance_policy",
                "automatic_restart",
                "service_account_email",
            )
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "project",
            "zone",
            "name",
            "resource_id",
            "self_link",
            "machine_type",
            "machine_type_self_link",
            "image_name",
            "image_id",
            "image_self_link",
            "provisioning_model",
            "termination_action",
            "maintenance_policy",
            "automatic_restart",
            "service_account_email",
            "metadata_keys",
            "launch_ref",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        raw_metadata_keys = obj["metadata_keys"]
        if not isinstance(raw_metadata_keys, list):
            raise ProtocolError("metadata_keys must be an array")
        return cls(
            project=_project(obj["project"]),
            zone=_zone(obj["zone"]),
            name=_gce_name(obj["name"], "name"),
            resource_id=_gce_id(obj["resource_id"], "resource_id"),
            self_link=require_string(obj["self_link"], "self_link"),
            machine_type=require_string(obj["machine_type"], "machine_type"),
            machine_type_self_link=require_string(
                obj["machine_type_self_link"], "machine_type_self_link"
            ),
            image_name=_gce_name(obj["image_name"], "image_name"),
            image_id=_gce_id(obj["image_id"], "image_id"),
            image_self_link=require_string(obj["image_self_link"], "image_self_link"),
            provisioning_model=require_string(
                obj["provisioning_model"], "provisioning_model"
            ),
            termination_action=require_string(
                obj["termination_action"], "termination_action"
            ),
            maintenance_policy=require_string(
                obj["maintenance_policy"], "maintenance_policy"
            ),
            automatic_restart=_boolean(obj["automatic_restart"], "automatic_restart"),
            service_account_email=_service_account_email(
                obj["service_account_email"], "service_account_email"
            ),
            metadata_keys=tuple(raw_metadata_keys),
            launch_ref=GCSObjectVersionRefV2.from_dict(obj["launch_ref"]),
        )


@dataclass(frozen=True, slots=True)
class ResourcePolicyObservationV2(CanonicalRecord):
    """Controller-normalized high-throughput TPU placement policy."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.resource-policy-observation.v2"

    project: str
    region: str
    name: str
    resource_id: int
    self_link: str
    policy_type: str
    topology: str

    def __post_init__(self) -> None:
        project = _project(self.project)
        region = _region(self.region)
        name = _gce_name(self.name, "name")
        _gce_id(self.resource_id, "resource_id")
        _scoped_link(
            self.self_link,
            "self_link",
            project=project,
            scope="regions",
            location=region,
            collection="resourcePolicies",
            resource_name=name,
        )
        if self.policy_type != "HIGH_THROUGHPUT":
            raise ProtocolError("resource policy type must be HIGH_THROUGHPUT")
        if self.topology != "8x8":
            raise ProtocolError("resource policy topology must be 8x8")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "project": self.project,
            "region": self.region,
            "name": self.name,
            "resource_id": self.resource_id,
            "self_link": self.self_link,
            "policy_type": self.policy_type,
            "topology": self.topology,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "project",
            "region",
            "name",
            "resource_id",
            "self_link",
            "policy_type",
            "topology",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            project=_project(obj["project"]),
            region=_region(obj["region"]),
            name=_gce_name(obj["name"], "name"),
            resource_id=_gce_id(obj["resource_id"], "resource_id"),
            self_link=require_string(obj["self_link"], "self_link"),
            policy_type=require_string(obj["policy_type"], "policy_type"),
            topology=require_string(obj["topology"], "topology"),
        )


@dataclass(frozen=True, slots=True)
class ManagedInstanceObservationV2(CanonicalRecord):
    """One member joined from MIG, instance, disk, and image API evidence.

    ``instances.get`` does not by itself prove the boot source-image numeric
    identity.  The cloud adapter must join ``listManagedInstances``,
    ``instances.get``, ``disks.get`` (including source-image ID), and the exact
    ``images.get`` result before constructing this normalized observation.
    """

    SCHEMA: ClassVar[str] = "moe-sae.r1.managed-instance-observation.v2"

    project: str
    zone: str
    name: str
    resource_id: int
    self_link: str
    mig_self_link: str
    instance_template_self_link: str
    machine_type_self_link: str
    status: str
    current_action: str
    service_account_email: str
    boot_source_image_id: int
    boot_source_image_self_link: str

    def __post_init__(self) -> None:
        project = _project(self.project)
        zone = _zone(self.zone)
        name = _gce_name(self.name, "name")
        _gce_id(self.resource_id, "resource_id")
        _scoped_link(
            self.self_link,
            "self_link",
            project=project,
            scope="zones",
            location=zone,
            collection="instances",
            resource_name=name,
        )
        require_string(self.mig_self_link, "mig_self_link")
        require_string(self.instance_template_self_link, "instance_template_self_link")
        _scoped_link(
            self.machine_type_self_link,
            "machine_type_self_link",
            project=project,
            scope="zones",
            location=zone,
            collection="machineTypes",
            resource_name="ct6e-standard-4t",
        )
        if self.status != "RUNNING":
            raise ProtocolError("managed instance status must be RUNNING")
        if self.current_action != "NONE":
            raise ProtocolError("managed instance current_action must be NONE")
        _service_account_email(self.service_account_email, "service_account_email")
        _gce_id(self.boot_source_image_id, "boot_source_image_id")
        _parse_image_link(
            self.boot_source_image_self_link,
            "boot_source_image_self_link",
        )

    def as_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA} | {
            field: getattr(self, field)
            for field in (
                "project",
                "zone",
                "name",
                "resource_id",
                "self_link",
                "mig_self_link",
                "instance_template_self_link",
                "machine_type_self_link",
                "status",
                "current_action",
                "service_account_email",
                "boot_source_image_id",
                "boot_source_image_self_link",
            )
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "project",
            "zone",
            "name",
            "resource_id",
            "self_link",
            "mig_self_link",
            "instance_template_self_link",
            "machine_type_self_link",
            "status",
            "current_action",
            "service_account_email",
            "boot_source_image_id",
            "boot_source_image_self_link",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            project=_project(obj["project"]),
            zone=_zone(obj["zone"]),
            name=_gce_name(obj["name"], "name"),
            resource_id=_gce_id(obj["resource_id"], "resource_id"),
            self_link=require_string(obj["self_link"], "self_link"),
            mig_self_link=require_string(obj["mig_self_link"], "mig_self_link"),
            instance_template_self_link=require_string(
                obj["instance_template_self_link"], "instance_template_self_link"
            ),
            machine_type_self_link=require_string(
                obj["machine_type_self_link"], "machine_type_self_link"
            ),
            status=require_string(obj["status"], "status"),
            current_action=require_string(obj["current_action"], "current_action"),
            service_account_email=_service_account_email(
                obj["service_account_email"], "service_account_email"
            ),
            boot_source_image_id=_gce_id(
                obj["boot_source_image_id"], "boot_source_image_id"
            ),
            boot_source_image_self_link=require_string(
                obj["boot_source_image_self_link"], "boot_source_image_self_link"
            ),
        )


@dataclass(frozen=True, slots=True)
class MIGObservationV2(CanonicalRecord):
    """Controller-normalized MIG state after it reaches a quiescent full slice."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.mig-observation.v2"

    project: str
    zone: str
    name: str
    resource_id: int
    self_link: str
    instance_template_id: int
    instance_template_self_link: str
    resource_policy_id: int
    resource_policy_self_link: str
    target_size: int
    target_size_policy: str
    default_action_on_failure: str
    is_stable: bool
    member_instance_self_links: tuple[str, ...]

    def __post_init__(self) -> None:
        project = _project(self.project)
        zone = _zone(self.zone)
        name = _gce_name(self.name, "name")
        _gce_id(self.resource_id, "resource_id")
        _scoped_link(
            self.self_link,
            "self_link",
            project=project,
            scope="zones",
            location=zone,
            collection="instanceGroupManagers",
            resource_name=name,
        )
        _gce_id(self.instance_template_id, "instance_template_id")
        require_string(self.instance_template_self_link, "instance_template_self_link")
        _gce_id(self.resource_policy_id, "resource_policy_id")
        require_string(self.resource_policy_self_link, "resource_policy_self_link")
        if require_integer(self.target_size, "target_size", minimum=0) != 16:
            raise ProtocolError("MIG target_size must be 16")
        if self.target_size_policy != "BULK":
            raise ProtocolError("MIG target_size_policy must be BULK")
        if self.default_action_on_failure != "DO_NOTHING":
            raise ProtocolError("MIG default_action_on_failure must be DO_NOTHING")
        if not _boolean(self.is_stable, "is_stable"):
            raise ProtocolError("MIG is_stable must be true")
        members = _string_array(
            self.member_instance_self_links,
            "member_instance_self_links",
            count=16,
        )
        for index, member in enumerate(members):
            match = re.fullmatch(
                re.escape(_GCE_PREFIX)
                + re.escape(project)
                + "/zones/"
                + re.escape(zone)
                + rf"/instances/{_GCE_NAME.pattern}",
                member,
            )
            if match is None:
                raise ProtocolError(
                    f"member_instance_self_links[{index}] is not in the MIG zone/project"
                )
        object.__setattr__(self, "member_instance_self_links", tuple(sorted(members)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "project": self.project,
            "zone": self.zone,
            "name": self.name,
            "resource_id": self.resource_id,
            "self_link": self.self_link,
            "instance_template_id": self.instance_template_id,
            "instance_template_self_link": self.instance_template_self_link,
            "resource_policy_id": self.resource_policy_id,
            "resource_policy_self_link": self.resource_policy_self_link,
            "target_size": self.target_size,
            "target_size_policy": self.target_size_policy,
            "default_action_on_failure": self.default_action_on_failure,
            "is_stable": self.is_stable,
            "member_instance_self_links": list(self.member_instance_self_links),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "project",
            "zone",
            "name",
            "resource_id",
            "self_link",
            "instance_template_id",
            "instance_template_self_link",
            "resource_policy_id",
            "resource_policy_self_link",
            "target_size",
            "target_size_policy",
            "default_action_on_failure",
            "is_stable",
            "member_instance_self_links",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        members = obj["member_instance_self_links"]
        if not isinstance(members, list):
            raise ProtocolError("member_instance_self_links must be an array")
        return cls(
            project=_project(obj["project"]),
            zone=_zone(obj["zone"]),
            name=_gce_name(obj["name"], "name"),
            resource_id=_gce_id(obj["resource_id"], "resource_id"),
            self_link=require_string(obj["self_link"], "self_link"),
            instance_template_id=_gce_id(
                obj["instance_template_id"], "instance_template_id"
            ),
            instance_template_self_link=require_string(
                obj["instance_template_self_link"], "instance_template_self_link"
            ),
            resource_policy_id=_gce_id(obj["resource_policy_id"], "resource_policy_id"),
            resource_policy_self_link=require_string(
                obj["resource_policy_self_link"], "resource_policy_self_link"
            ),
            target_size=require_integer(obj["target_size"], "target_size", minimum=0),
            target_size_policy=require_string(
                obj["target_size_policy"], "target_size_policy"
            ),
            default_action_on_failure=require_string(
                obj["default_action_on_failure"], "default_action_on_failure"
            ),
            is_stable=_boolean(obj["is_stable"], "is_stable"),
            member_instance_self_links=tuple(members),
        )


@dataclass(frozen=True, slots=True)
class ControllerPlacementReceiptV2(CanonicalRecord):
    """Trusted controller's joined API observation of one ready v6e-64 slice.

    Acceptance requires all three independent bindings:

    * ``verify_deployment`` against the precommitted envelope;
    * ``verify_authority`` against a challenge and controller identity held
      outside this self-described record; and
    * ``api_evidence.verify_bytes`` after fetching the exact GCS generation.

    The raw response bundle must include MIG/template/resource-policy reads,
    ``listManagedInstances``, every ``instances.get`` and ``disks.get``, exact
    source-image resolution, and the IAM service-account identity lookup.  A
    host-generated copy of these fields is not authoritative.
    """

    SCHEMA: ClassVar[str] = "moe-sae.r1.controller-placement-receipt.v2"

    deployment_envelope_sha256: str
    base_image_resolution_sha256: str
    image_publication_sha256: str
    slice_attempt_id: str
    controller_service_account_email: str
    controller_service_account_unique_id: str
    controller_challenge: str
    api_evidence: GCSObjectVersionRefV2
    worker_service_account_email: str
    worker_service_account_unique_id: str
    instance_template: InstanceTemplateObservationV2
    resource_policy: ResourcePolicyObservationV2
    mig: MIGObservationV2
    instances: tuple[ManagedInstanceObservationV2, ...]
    observed_at_unix_ms: int

    def __post_init__(self) -> None:
        _sha256(self.deployment_envelope_sha256, "deployment_envelope_sha256")
        _sha256(self.base_image_resolution_sha256, "base_image_resolution_sha256")
        _sha256(self.image_publication_sha256, "image_publication_sha256")
        require_string(self.slice_attempt_id, "slice_attempt_id", pattern=_ATTEMPT_ID)
        _service_account_email(
            self.controller_service_account_email,
            "controller_service_account_email",
        )
        _service_account_unique_id(
            self.controller_service_account_unique_id,
            "controller_service_account_unique_id",
        )
        _sha256(self.controller_challenge, "controller_challenge")
        if type(self.api_evidence) is not GCSObjectVersionRefV2:
            raise ProtocolError("api_evidence must be GCSObjectVersionRefV2")
        _service_account_email(
            self.worker_service_account_email,
            "worker_service_account_email",
        )
        _service_account_unique_id(
            self.worker_service_account_unique_id,
            "worker_service_account_unique_id",
        )
        if type(self.instance_template) is not InstanceTemplateObservationV2:
            raise ProtocolError(
                "instance_template must be InstanceTemplateObservationV2"
            )
        if type(self.resource_policy) is not ResourcePolicyObservationV2:
            raise ProtocolError("resource_policy must be ResourcePolicyObservationV2")
        if type(self.mig) is not MIGObservationV2:
            raise ProtocolError("mig must be MIGObservationV2")
        if not isinstance(self.instances, (list, tuple)):
            raise ProtocolError("instances must be an array")
        instances = tuple(self.instances)
        if len(instances) != 16:
            raise ProtocolError("instances must contain exactly 16 observations")
        if any(type(item) is not ManagedInstanceObservationV2 for item in instances):
            raise ProtocolError(
                "instances must contain ManagedInstanceObservationV2 records"
            )
        for field in ("name", "resource_id", "self_link"):
            values = [getattr(item, field) for item in instances]
            if len(set(values)) != len(values):
                raise ProtocolError(f"managed instance {field} values must be unique")
        instances = tuple(sorted(instances, key=lambda item: item.self_link))
        object.__setattr__(self, "instances", instances)

        template = self.instance_template
        policy = self.resource_policy
        mig = self.mig
        if _zone_region(template.zone) != policy.region:
            raise ProtocolError("template zone and resource policy region differ")
        if (mig.project, mig.zone) != (template.project, template.zone):
            raise ProtocolError("MIG and template placement differ")
        if (policy.project, policy.region) != (
            template.project,
            _zone_region(template.zone),
        ):
            raise ProtocolError("resource policy and template placement differ")
        if (
            mig.instance_template_id != template.resource_id
            or mig.instance_template_self_link != template.self_link
        ):
            raise ProtocolError("MIG instance template identity is inconsistent")
        if (
            mig.resource_policy_id != policy.resource_id
            or mig.resource_policy_self_link != policy.self_link
        ):
            raise ProtocolError("MIG resource policy identity is inconsistent")
        if template.service_account_email != self.worker_service_account_email:
            raise ProtocolError("template worker service account is inconsistent")
        observed_members = tuple(item.self_link for item in instances)
        if observed_members != mig.member_instance_self_links:
            raise ProtocolError(
                "MIG membership does not exactly match instance observations"
            )
        for item in instances:
            checks = {
                "project": template.project,
                "zone": template.zone,
                "mig_self_link": mig.self_link,
                "instance_template_self_link": template.self_link,
                "machine_type_self_link": template.machine_type_self_link,
                "service_account_email": self.worker_service_account_email,
                "boot_source_image_id": template.image_id,
                "boot_source_image_self_link": template.image_self_link,
            }
            for field, expected in checks.items():
                if getattr(item, field) != expected:
                    raise ProtocolError(
                        f"managed instance {item.name} {field} is inconsistent"
                    )
        require_integer(self.observed_at_unix_ms, "observed_at_unix_ms", minimum=0)

    def verify_authority(
        self,
        *,
        controller_service_account_email: str,
        controller_service_account_unique_id: str,
        controller_challenge: str,
    ) -> None:
        expected = {
            "controller_service_account_email": _service_account_email(
                controller_service_account_email,
                "controller_service_account_email",
            ),
            "controller_service_account_unique_id": _service_account_unique_id(
                controller_service_account_unique_id,
                "controller_service_account_unique_id",
            ),
            "controller_challenge": _sha256(
                controller_challenge,
                "controller_challenge",
            ),
        }
        for field, expected_value in expected.items():
            if getattr(self, field) != expected_value:
                raise ProtocolError(
                    f"placement receipt {field} does not match authority"
                )

    def verify_admission(
        self,
        deployment: DeploymentEnvelopeV2,
        *,
        controller_service_account_email: str,
        controller_service_account_unique_id: str,
        controller_challenge: str,
        api_evidence_bytes: bytes,
    ) -> None:
        """Atomically check deployment, external authority, and evidence bytes."""

        self.verify_deployment(deployment)
        self.verify_authority(
            controller_service_account_email=controller_service_account_email,
            controller_service_account_unique_id=controller_service_account_unique_id,
            controller_challenge=controller_challenge,
        )
        self.api_evidence.verify_bytes(api_evidence_bytes)

    def verify_deployment(self, deployment: DeploymentEnvelopeV2) -> None:
        if type(deployment) is not DeploymentEnvelopeV2:
            raise ProtocolError("deployment must be DeploymentEnvelopeV2")
        checks = {
            "deployment_envelope_sha256": deployment.digest,
            "base_image_resolution_sha256": deployment.base_image_resolution_sha256,
            "image_publication_sha256": deployment.image_publication_sha256,
            "slice_attempt_id": deployment.slice_attempt_id,
            "worker_service_account_email": deployment.service_account_email,
            "worker_service_account_unique_id": deployment.service_account_unique_id,
        }
        for field, expected in checks.items():
            if getattr(self, field) != expected:
                raise ProtocolError(
                    f"placement receipt {field} does not match deployment"
                )
        if not self.api_evidence.uri.startswith(deployment.controller_root_uri):
            raise ProtocolError(
                "placement API evidence is outside the deployment controller root"
            )
        template_checks = {
            "project": deployment.project,
            "zone": deployment.zone,
            "name": deployment.instance_template_name,
            "resource_id": deployment.instance_template_id,
            "self_link": deployment.instance_template_self_link,
            "machine_type": deployment.machine_type,
            "machine_type_self_link": deployment.machine_type_self_link,
            "image_name": deployment.image_name,
            "image_id": deployment.image_id,
            "image_self_link": deployment.image_self_link,
            "provisioning_model": deployment.provisioning_model,
            "termination_action": deployment.termination_action,
            "maintenance_policy": deployment.maintenance_policy,
            "service_account_email": deployment.service_account_email,
            "launch_ref": deployment.launch_ref,
        }
        for field, expected in template_checks.items():
            if getattr(self.instance_template, field) != expected:
                raise ProtocolError(
                    f"template observation {field} does not match deployment"
                )
        policy_checks = {
            "project": deployment.project,
            "region": deployment.region,
            "name": deployment.resource_policy_name,
            "resource_id": deployment.resource_policy_id,
            "self_link": deployment.resource_policy_self_link,
            "topology": deployment.topology,
        }
        for field, expected in policy_checks.items():
            if getattr(self.resource_policy, field) != expected:
                raise ProtocolError(
                    f"resource policy observation {field} does not match deployment"
                )
        mig_checks = {
            "project": deployment.project,
            "zone": deployment.zone,
            "name": deployment.mig_name,
            "resource_id": deployment.mig_id,
            "self_link": deployment.mig_self_link,
            "target_size": deployment.target_size,
            "target_size_policy": deployment.target_size_policy,
            "default_action_on_failure": deployment.default_action_on_failure,
        }
        for field, expected in mig_checks.items():
            if getattr(self.mig, field) != expected:
                raise ProtocolError(
                    f"MIG observation {field} does not match deployment"
                )

    def verify_fresh_replacement_of(
        self, previous: ControllerPlacementReceiptV2
    ) -> None:
        """Reject reused MIG or member identities across replacement attempts."""

        if type(previous) is not ControllerPlacementReceiptV2:
            raise ProtocolError("previous must be ControllerPlacementReceiptV2")
        if self.slice_attempt_id == previous.slice_attempt_id:
            raise ProtocolError("replacement receipt reused slice_attempt_id")
        if self.mig.resource_id == previous.mig.resource_id:
            raise ProtocolError("replacement receipt reused MIG numeric ID")
        if self.mig.self_link == previous.mig.self_link:
            raise ProtocolError("replacement receipt reused MIG selfLink")
        prior_instance_ids = {item.resource_id for item in previous.instances}
        current_instance_ids = {item.resource_id for item in self.instances}
        if prior_instance_ids & current_instance_ids:
            raise ProtocolError("replacement receipt reused managed instance IDs")
        prior_instance_links = {item.self_link for item in previous.instances}
        current_instance_links = {item.self_link for item in self.instances}
        if prior_instance_links & current_instance_links:
            raise ProtocolError("replacement receipt reused managed instance selfLinks")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "deployment_envelope_sha256": self.deployment_envelope_sha256,
            "base_image_resolution_sha256": self.base_image_resolution_sha256,
            "image_publication_sha256": self.image_publication_sha256,
            "slice_attempt_id": self.slice_attempt_id,
            "controller_service_account_email": self.controller_service_account_email,
            "controller_service_account_unique_id": self.controller_service_account_unique_id,
            "controller_challenge": self.controller_challenge,
            "api_evidence": self.api_evidence.as_dict(),
            "worker_service_account_email": self.worker_service_account_email,
            "worker_service_account_unique_id": self.worker_service_account_unique_id,
            "instance_template": self.instance_template.as_dict(),
            "resource_policy": self.resource_policy.as_dict(),
            "mig": self.mig.as_dict(),
            "instances": [item.as_dict() for item in self.instances],
            "observed_at_unix_ms": self.observed_at_unix_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "deployment_envelope_sha256",
            "base_image_resolution_sha256",
            "image_publication_sha256",
            "slice_attempt_id",
            "controller_service_account_email",
            "controller_service_account_unique_id",
            "controller_challenge",
            "api_evidence",
            "worker_service_account_email",
            "worker_service_account_unique_id",
            "instance_template",
            "resource_policy",
            "mig",
            "instances",
            "observed_at_unix_ms",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        raw_instances = obj["instances"]
        if not isinstance(raw_instances, list):
            raise ProtocolError("instances must be an array")
        return cls(
            deployment_envelope_sha256=_sha256(
                obj["deployment_envelope_sha256"], "deployment_envelope_sha256"
            ),
            base_image_resolution_sha256=_sha256(
                obj["base_image_resolution_sha256"],
                "base_image_resolution_sha256",
            ),
            image_publication_sha256=_sha256(
                obj["image_publication_sha256"], "image_publication_sha256"
            ),
            slice_attempt_id=require_string(
                obj["slice_attempt_id"], "slice_attempt_id", pattern=_ATTEMPT_ID
            ),
            controller_service_account_email=_service_account_email(
                obj["controller_service_account_email"],
                "controller_service_account_email",
            ),
            controller_service_account_unique_id=_service_account_unique_id(
                obj["controller_service_account_unique_id"],
                "controller_service_account_unique_id",
            ),
            controller_challenge=_sha256(
                obj["controller_challenge"], "controller_challenge"
            ),
            api_evidence=GCSObjectVersionRefV2.from_dict(obj["api_evidence"]),
            worker_service_account_email=_service_account_email(
                obj["worker_service_account_email"],
                "worker_service_account_email",
            ),
            worker_service_account_unique_id=_service_account_unique_id(
                obj["worker_service_account_unique_id"],
                "worker_service_account_unique_id",
            ),
            instance_template=InstanceTemplateObservationV2.from_dict(
                obj["instance_template"]
            ),
            resource_policy=ResourcePolicyObservationV2.from_dict(
                obj["resource_policy"]
            ),
            mig=MIGObservationV2.from_dict(obj["mig"]),
            instances=tuple(
                ManagedInstanceObservationV2.from_dict(item) for item in raw_instances
            ),
            observed_at_unix_ms=require_integer(
                obj["observed_at_unix_ms"], "observed_at_unix_ms", minimum=0
            ),
        )


__all__ = [
    "BaseImageFamilyResolutionReceiptV2",
    "ControllerPlacementReceiptV2",
    "DeploymentEnvelopeV2",
    "GCSObjectVersionRefV2",
    "InstanceTemplateObservationV2",
    "MIGObservationV2",
    "ManagedInstanceObservationV2",
    "ResourcePolicyObservationV2",
]
