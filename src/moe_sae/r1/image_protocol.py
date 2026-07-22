"""Canonical provenance records for immutable DeepSeek-R1 TPU images.

The records in this module bridge four different trust boundaries:

* the declarative recipe used by Packer;
* observations made inside the completed image;
* the GCE image resource that was actually published; and
* the TPU hosts and JAX devices that booted from that publication.

GCE resources are identified by their numeric resource ID and canonical
``selfLink``.  In particular, a boot persistent disk is *not* assigned a
fictional content digest.  A host instead attests the disk's GCE identity, the
numeric source-image identity reported by GCE, and the canonical image
manifest digest observed in the guest.

This module deliberately has no Google Cloud, JAX, or Packer dependency.  It
uses the strict JSON and canonical hashing machinery from :mod:`protocol` so
the same bytes can be checked by a controller, image builder, or executor.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, ClassVar, Mapping, Self

from .protocol import (
    CanonicalRecord,
    DeploymentIdentityV1,
    ImplementationIdentityV1,
    ProtocolError,
    _REVISION,
    _SHA256,
    _integer,
    _strict_object,
    _string,
)


_GCE_ID_MAX = (1 << 64) - 1
_GCE_PREFIX = "https://www.googleapis.com/compute/v1/projects/"
_GCE_NAME = re.compile(r"[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?")
_PROJECT = re.compile(r"[a-z][a-z0-9-]{0,61}[a-z0-9]")
_REGION = re.compile(r"[a-z]+-[a-z0-9]+[0-9]")
_ZONE = re.compile(r"[a-z]+-[a-z0-9]+[0-9]-[a-z][a-z0-9]*")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}")
_LINUX_USER = re.compile(r"[a-z_][a-z0-9_-]{0,31}")
_SERVICE_UNIT = re.compile(r"[A-Za-z0-9@_.-]+\.service")
_VFIO_DEVICE = re.compile(r"/dev/vfio/(?:0|[1-9][0-9]*)")
_SERVICE_ACCOUNT = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._+-]*@[A-Za-z0-9.-]+\.gserviceaccount\.com"
)


def _sha256(value: Any, name: str) -> str:
    return _string(value, name, pattern=_SHA256)


def _revision(value: Any, name: str) -> str:
    return _string(value, name, pattern=_REVISION)


def _gce_id(value: Any, name: str) -> int:
    return _integer(value, name, minimum=1, maximum=_GCE_ID_MAX)


def _gce_name(value: Any, name: str) -> str:
    return _string(value, name, pattern=_GCE_NAME)


def _project(value: Any, name: str = "project") -> str:
    return _string(value, name, pattern=_PROJECT)


def _region(value: Any, name: str = "region") -> str:
    return _string(value, name, pattern=_REGION)


def _zone(value: Any, name: str = "zone") -> str:
    return _string(value, name, pattern=_ZONE)


def _zone_region(zone: str) -> str:
    return zone.rsplit("-", 1)[0]


def _canonical_gce_link(
    value: Any,
    name: str,
    *,
    project: str,
    scope_kind: str,
    scope_name: str,
    collection: str,
    resource_name: str,
) -> str:
    link = _string(value, name)
    expected = (
        f"{_GCE_PREFIX}{project}/{scope_kind}/{scope_name}/{collection}/{resource_name}"
    )
    if link != expected:
        raise ProtocolError(f"{name} must equal canonical GCE selfLink {expected!r}")
    return link


def _expected_global_link(*, project: str, collection: str, resource_name: str) -> str:
    return f"{_GCE_PREFIX}{project}/global/{collection}/{resource_name}"


def _validate_global_link(
    value: Any,
    name: str,
    *,
    project: str,
    collection: str,
    resource_name: str,
) -> str:
    link = _string(value, name)
    expected = _expected_global_link(
        project=project,
        collection=collection,
        resource_name=resource_name,
    )
    if link != expected:
        raise ProtocolError(f"{name} must equal canonical GCE selfLink {expected!r}")
    return link


def _parse_image_link(value: Any, name: str) -> tuple[str, str, str]:
    link = _string(value, name)
    match = re.fullmatch(
        re.escape(_GCE_PREFIX)
        + rf"(?P<project>{_PROJECT.pattern})/global/images/"
        + rf"(?P<image>{_GCE_NAME.pattern})",
        link,
    )
    if match is None:
        raise ProtocolError(f"{name} must be a canonical GCE image selfLink")
    return link, match.group("project"), match.group("image")


def _controller_root(value: Any, name: str = "controller_root_uri") -> str:
    uri = _string(value, name)
    match = re.fullmatch(
        r"gs://(?P<bucket>[a-z0-9][a-z0-9._-]{1,220}[a-z0-9])/"
        r"(?P<path>[A-Za-z0-9._~!$&'()*+,;=:@/-]+)/",
        uri,
    )
    if match is None:
        raise ProtocolError(
            f"{name} must be an absolute gs:// prefix with a non-empty path and trailing slash"
        )
    bucket = match.group("bucket")
    path = match.group("path")
    if (
        ".." in bucket
        or ".-" in bucket
        or "-." in bucket
        or "//" in path
        or any(part in {".", ".."} for part in path.split("/"))
    ):
        raise ProtocolError(f"{name} must be a normalized gs:// prefix")
    return uri


def _string_array(value: Any, name: str, *, count: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProtocolError(f"{name} must be an array")
    result = tuple(_string(item, f"{name}[]") for item in value)
    if len(result) != count:
        raise ProtocolError(f"{name} must contain exactly {count} entries")
    if len(set(result)) != len(result):
        raise ProtocolError(f"{name} must not contain duplicates")
    return result


def _device_ids(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProtocolError(f"{name} must be an array")
    result = tuple(_integer(item, f"{name}[]", minimum=0) for item in value)
    if len(result) != 4:
        raise ProtocolError(f"{name} must contain exactly 4 entries")
    if len(set(result)) != len(result):
        raise ProtocolError(f"{name} must not contain duplicate devices")
    return result


def _device_coordinates(value: Any, name: str) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(value, (list, tuple)):
        raise ProtocolError(f"{name} must be an array")
    result: list[tuple[int, int, int]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise ProtocolError(f"{name}[{index}] must be a three-integer coordinate")
        result.append(
            (
                _integer(raw[0], f"{name}[{index}][0]", minimum=0),
                _integer(raw[1], f"{name}[{index}][1]", minimum=0),
                _integer(raw[2], f"{name}[{index}][2]", minimum=0),
            )
        )
    if len(result) != 4:
        raise ProtocolError(f"{name} must contain exactly 4 entries")
    return tuple(result)


def _core_on_chip(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProtocolError(f"{name} must be an array")
    result = tuple(_integer(item, f"{name}[]", minimum=0) for item in value)
    if len(result) != 4:
        raise ProtocolError(f"{name} must contain exactly 4 entries")
    return result


@dataclass(frozen=True, slots=True)
class ImageRecipeV1(CanonicalRecord):
    """Inputs whose exact bytes define one reproducible image build."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.image-recipe.v1"

    base_image_id: int
    base_image_self_link: str
    repository_commit: str
    repository_archive_sha256: str
    pyproject_sha256: str
    uv_lock_sha256: str
    maxtext_commit: str
    python_version: str
    uv_version: str
    packer_version: str
    packer_template_sha256: str
    provisioner_sha256: str
    service_unit_name: str
    service_unit_sha256: str

    def __post_init__(self) -> None:
        _gce_id(self.base_image_id, "base_image_id")
        _parse_image_link(self.base_image_self_link, "base_image_self_link")
        _revision(self.repository_commit, "repository_commit")
        _sha256(self.repository_archive_sha256, "repository_archive_sha256")
        _sha256(self.pyproject_sha256, "pyproject_sha256")
        _sha256(self.uv_lock_sha256, "uv_lock_sha256")
        _revision(self.maxtext_commit, "maxtext_commit")
        _string(self.python_version, "python_version", pattern=_VERSION)
        _string(self.uv_version, "uv_version", pattern=_VERSION)
        _string(self.packer_version, "packer_version", pattern=_VERSION)
        _sha256(self.packer_template_sha256, "packer_template_sha256")
        _sha256(self.provisioner_sha256, "provisioner_sha256")
        _string(self.service_unit_name, "service_unit_name", pattern=_SERVICE_UNIT)
        _sha256(self.service_unit_sha256, "service_unit_sha256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "base_image_id": self.base_image_id,
            "base_image_self_link": self.base_image_self_link,
            "repository_commit": self.repository_commit,
            "repository_archive_sha256": self.repository_archive_sha256,
            "pyproject_sha256": self.pyproject_sha256,
            "uv_lock_sha256": self.uv_lock_sha256,
            "maxtext_commit": self.maxtext_commit,
            "python_version": self.python_version,
            "uv_version": self.uv_version,
            "packer_version": self.packer_version,
            "packer_template_sha256": self.packer_template_sha256,
            "provisioner_sha256": self.provisioner_sha256,
            "service_unit_name": self.service_unit_name,
            "service_unit_sha256": self.service_unit_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "base_image_id",
            "base_image_self_link",
            "repository_commit",
            "repository_archive_sha256",
            "pyproject_sha256",
            "uv_lock_sha256",
            "maxtext_commit",
            "python_version",
            "uv_version",
            "packer_version",
            "packer_template_sha256",
            "provisioner_sha256",
            "service_unit_name",
            "service_unit_sha256",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            base_image_id=_gce_id(obj["base_image_id"], "base_image_id"),
            base_image_self_link=_string(
                obj["base_image_self_link"], "base_image_self_link"
            ),
            repository_commit=_revision(obj["repository_commit"], "repository_commit"),
            repository_archive_sha256=_sha256(
                obj["repository_archive_sha256"], "repository_archive_sha256"
            ),
            pyproject_sha256=_sha256(obj["pyproject_sha256"], "pyproject_sha256"),
            uv_lock_sha256=_sha256(obj["uv_lock_sha256"], "uv_lock_sha256"),
            maxtext_commit=_revision(obj["maxtext_commit"], "maxtext_commit"),
            python_version=_string(
                obj["python_version"], "python_version", pattern=_VERSION
            ),
            uv_version=_string(obj["uv_version"], "uv_version", pattern=_VERSION),
            packer_version=_string(
                obj["packer_version"], "packer_version", pattern=_VERSION
            ),
            packer_template_sha256=_sha256(
                obj["packer_template_sha256"], "packer_template_sha256"
            ),
            provisioner_sha256=_sha256(obj["provisioner_sha256"], "provisioner_sha256"),
            service_unit_name=_string(
                obj["service_unit_name"],
                "service_unit_name",
                pattern=_SERVICE_UNIT,
            ),
            service_unit_sha256=_sha256(
                obj["service_unit_sha256"], "service_unit_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class ImageManifestV1(CanonicalRecord):
    """Measurements made inside the completed image before publication."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.image-manifest.v1"

    recipe_sha256: str
    os_release_id: str
    os_release_version_id: str
    kernel_release: str
    architecture: str
    repository_commit: str
    installed_repository_archive_sha256: str
    maxtext_commit: str
    python_version: str
    uv_version: str
    jax_version: str
    libtpu_version: str
    installed_distribution_sha256: str
    sbom_sha256: str
    runtime_user: str
    runtime_user_spec_sha256: str
    service_unit_name: str
    service_unit_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.recipe_sha256, "recipe_sha256")
        _string(self.os_release_id, "os_release_id", pattern=_VERSION)
        _string(self.os_release_version_id, "os_release_version_id", pattern=_VERSION)
        _string(self.kernel_release, "kernel_release", pattern=_VERSION)
        _string(self.architecture, "architecture", pattern=_VERSION)
        _revision(self.repository_commit, "repository_commit")
        _sha256(
            self.installed_repository_archive_sha256,
            "installed_repository_archive_sha256",
        )
        _revision(self.maxtext_commit, "maxtext_commit")
        _string(self.python_version, "python_version", pattern=_VERSION)
        _string(self.uv_version, "uv_version", pattern=_VERSION)
        _string(self.jax_version, "jax_version", pattern=_VERSION)
        _string(self.libtpu_version, "libtpu_version", pattern=_VERSION)
        _sha256(self.installed_distribution_sha256, "installed_distribution_sha256")
        _sha256(self.sbom_sha256, "sbom_sha256")
        _string(self.runtime_user, "runtime_user", pattern=_LINUX_USER)
        _sha256(self.runtime_user_spec_sha256, "runtime_user_spec_sha256")
        _string(self.service_unit_name, "service_unit_name", pattern=_SERVICE_UNIT)
        _sha256(self.service_unit_sha256, "service_unit_sha256")

    def verify_recipe(self, recipe: ImageRecipeV1) -> None:
        """Fail unless the observations bind to the exact declared recipe."""

        if not isinstance(recipe, ImageRecipeV1):
            raise ProtocolError("recipe must be ImageRecipeV1")
        expected = {
            "recipe_sha256": recipe.digest,
            "repository_commit": recipe.repository_commit,
            "installed_repository_archive_sha256": recipe.repository_archive_sha256,
            "maxtext_commit": recipe.maxtext_commit,
            "python_version": recipe.python_version,
            "uv_version": recipe.uv_version,
            "service_unit_name": recipe.service_unit_name,
            "service_unit_sha256": recipe.service_unit_sha256,
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ProtocolError(f"image manifest {field} does not match recipe")

    def verify_implementation(self, implementation: ImplementationIdentityV1) -> None:
        """Fail unless an implementation selects these observed image bytes."""

        if not isinstance(implementation, ImplementationIdentityV1):
            raise ProtocolError("implementation must be ImplementationIdentityV1")
        expected = {
            "image_manifest_sha256": self.digest,
            "repository_commit": self.repository_commit,
            "maxtext_commit": self.maxtext_commit,
            "python_version": self.python_version,
            "jax_version": self.jax_version,
            "libtpu_version": self.libtpu_version,
        }
        for field, value in expected.items():
            if getattr(implementation, field) != value:
                raise ProtocolError(
                    f"implementation {field} does not match image manifest"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "recipe_sha256": self.recipe_sha256,
            "os_release_id": self.os_release_id,
            "os_release_version_id": self.os_release_version_id,
            "kernel_release": self.kernel_release,
            "architecture": self.architecture,
            "repository_commit": self.repository_commit,
            "installed_repository_archive_sha256": self.installed_repository_archive_sha256,
            "maxtext_commit": self.maxtext_commit,
            "python_version": self.python_version,
            "uv_version": self.uv_version,
            "jax_version": self.jax_version,
            "libtpu_version": self.libtpu_version,
            "installed_distribution_sha256": self.installed_distribution_sha256,
            "sbom_sha256": self.sbom_sha256,
            "runtime_user": self.runtime_user,
            "runtime_user_spec_sha256": self.runtime_user_spec_sha256,
            "service_unit_name": self.service_unit_name,
            "service_unit_sha256": self.service_unit_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "recipe_sha256",
            "os_release_id",
            "os_release_version_id",
            "kernel_release",
            "architecture",
            "repository_commit",
            "installed_repository_archive_sha256",
            "maxtext_commit",
            "python_version",
            "uv_version",
            "jax_version",
            "libtpu_version",
            "installed_distribution_sha256",
            "sbom_sha256",
            "runtime_user",
            "runtime_user_spec_sha256",
            "service_unit_name",
            "service_unit_sha256",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            recipe_sha256=_sha256(obj["recipe_sha256"], "recipe_sha256"),
            os_release_id=_string(
                obj["os_release_id"], "os_release_id", pattern=_VERSION
            ),
            os_release_version_id=_string(
                obj["os_release_version_id"],
                "os_release_version_id",
                pattern=_VERSION,
            ),
            kernel_release=_string(
                obj["kernel_release"], "kernel_release", pattern=_VERSION
            ),
            architecture=_string(obj["architecture"], "architecture", pattern=_VERSION),
            repository_commit=_revision(obj["repository_commit"], "repository_commit"),
            installed_repository_archive_sha256=_sha256(
                obj["installed_repository_archive_sha256"],
                "installed_repository_archive_sha256",
            ),
            maxtext_commit=_revision(obj["maxtext_commit"], "maxtext_commit"),
            python_version=_string(
                obj["python_version"], "python_version", pattern=_VERSION
            ),
            uv_version=_string(obj["uv_version"], "uv_version", pattern=_VERSION),
            jax_version=_string(obj["jax_version"], "jax_version", pattern=_VERSION),
            libtpu_version=_string(
                obj["libtpu_version"], "libtpu_version", pattern=_VERSION
            ),
            installed_distribution_sha256=_sha256(
                obj["installed_distribution_sha256"],
                "installed_distribution_sha256",
            ),
            sbom_sha256=_sha256(obj["sbom_sha256"], "sbom_sha256"),
            runtime_user=_string(
                obj["runtime_user"], "runtime_user", pattern=_LINUX_USER
            ),
            runtime_user_spec_sha256=_sha256(
                obj["runtime_user_spec_sha256"], "runtime_user_spec_sha256"
            ),
            service_unit_name=_string(
                obj["service_unit_name"],
                "service_unit_name",
                pattern=_SERVICE_UNIT,
            ),
            service_unit_sha256=_sha256(
                obj["service_unit_sha256"], "service_unit_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class ImagePublicationV1(CanonicalRecord):
    """The completed GCE custom-image resource and its build evidence."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.image-publication.v1"

    image_project: str
    image_name: str
    image_id: int
    image_self_link: str
    base_image_id: int
    base_image_self_link: str
    manifest_sha256: str
    creation_operation_name: str
    creation_operation_id: int
    creation_operation_self_link: str
    creation_operation_status: str

    def __post_init__(self) -> None:
        project = _project(self.image_project, "image_project")
        image = _gce_name(self.image_name, "image_name")
        _gce_id(self.image_id, "image_id")
        _validate_global_link(
            self.image_self_link,
            "image_self_link",
            project=project,
            collection="images",
            resource_name=image,
        )
        _gce_id(self.base_image_id, "base_image_id")
        _parse_image_link(self.base_image_self_link, "base_image_self_link")
        _sha256(self.manifest_sha256, "manifest_sha256")
        operation = _gce_name(self.creation_operation_name, "creation_operation_name")
        _gce_id(self.creation_operation_id, "creation_operation_id")
        _validate_global_link(
            self.creation_operation_self_link,
            "creation_operation_self_link",
            project=project,
            collection="operations",
            resource_name=operation,
        )
        if self.creation_operation_status != "DONE":
            raise ProtocolError("creation_operation_status must be DONE")

    def verify_manifest(
        self, manifest: ImageManifestV1, recipe: ImageRecipeV1 | None = None
    ) -> None:
        if not isinstance(manifest, ImageManifestV1):
            raise ProtocolError("manifest must be ImageManifestV1")
        if manifest.digest != self.manifest_sha256:
            raise ProtocolError("published manifest digest does not match manifest")
        if recipe is not None:
            manifest.verify_recipe(recipe)
            if (
                self.base_image_id != recipe.base_image_id
                or self.base_image_self_link != recipe.base_image_self_link
            ):
                raise ProtocolError("published base image does not match recipe")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "image_project": self.image_project,
            "image_name": self.image_name,
            "image_id": self.image_id,
            "image_self_link": self.image_self_link,
            "base_image_id": self.base_image_id,
            "base_image_self_link": self.base_image_self_link,
            "manifest_sha256": self.manifest_sha256,
            "creation_operation_name": self.creation_operation_name,
            "creation_operation_id": self.creation_operation_id,
            "creation_operation_self_link": self.creation_operation_self_link,
            "creation_operation_status": self.creation_operation_status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "image_project",
            "image_name",
            "image_id",
            "image_self_link",
            "base_image_id",
            "base_image_self_link",
            "manifest_sha256",
            "creation_operation_name",
            "creation_operation_id",
            "creation_operation_self_link",
            "creation_operation_status",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            image_project=_project(obj["image_project"], "image_project"),
            image_name=_gce_name(obj["image_name"], "image_name"),
            image_id=_gce_id(obj["image_id"], "image_id"),
            image_self_link=_string(obj["image_self_link"], "image_self_link"),
            base_image_id=_gce_id(obj["base_image_id"], "base_image_id"),
            base_image_self_link=_string(
                obj["base_image_self_link"], "base_image_self_link"
            ),
            manifest_sha256=_sha256(obj["manifest_sha256"], "manifest_sha256"),
            creation_operation_name=_gce_name(
                obj["creation_operation_name"], "creation_operation_name"
            ),
            creation_operation_id=_gce_id(
                obj["creation_operation_id"], "creation_operation_id"
            ),
            creation_operation_self_link=_string(
                obj["creation_operation_self_link"],
                "creation_operation_self_link",
            ),
            creation_operation_status=_string(
                obj["creation_operation_status"], "creation_operation_status"
            ),
        )


@dataclass(frozen=True, slots=True)
class DeploymentEnvelopeV1(CanonicalRecord):
    """Immutable GCE placement policy accepted for one TPU slice."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.deployment-envelope.v1"

    image_publication_sha256: str
    project: str
    region: str
    accelerator_type: str
    instance_template_name: str
    instance_template_id: int
    instance_template_self_link: str
    workload_policy_name: str
    workload_policy_id: int
    workload_policy_self_link: str
    mig_name: str
    mig_id: int
    mig_scope: str
    mig_location: str
    mig_self_link: str
    provisioning_model: str
    preemption_notice_seconds: int
    termination_action: str
    maintenance_policy: str
    service_account: str
    controller_root_uri: str

    def __post_init__(self) -> None:
        _sha256(self.image_publication_sha256, "image_publication_sha256")
        project = _project(self.project)
        region = _region(self.region)
        accelerator_type = _string(
            self.accelerator_type, "accelerator_type", pattern=_GCE_NAME
        )
        if accelerator_type != "v6e-64":
            raise ProtocolError("accelerator_type must be v6e-64")
        template = _gce_name(self.instance_template_name, "instance_template_name")
        _gce_id(self.instance_template_id, "instance_template_id")
        _validate_global_link(
            self.instance_template_self_link,
            "instance_template_self_link",
            project=project,
            collection="instanceTemplates",
            resource_name=template,
        )
        policy = _gce_name(self.workload_policy_name, "workload_policy_name")
        _gce_id(self.workload_policy_id, "workload_policy_id")
        _canonical_gce_link(
            self.workload_policy_self_link,
            "workload_policy_self_link",
            project=project,
            scope_kind="regions",
            scope_name=region,
            collection="resourcePolicies",
            resource_name=policy,
        )
        mig = _gce_name(self.mig_name, "mig_name")
        _gce_id(self.mig_id, "mig_id")
        if self.mig_scope not in {"regional", "zonal"}:
            raise ProtocolError("mig_scope must be regional or zonal")
        if self.mig_scope == "regional":
            location = _region(self.mig_location, "mig_location")
            if location != region:
                raise ProtocolError("regional MIG location must equal region")
            scope_kind = "regions"
        else:
            location = _zone(self.mig_location, "mig_location")
            if _zone_region(location) != region:
                raise ProtocolError("zonal MIG location must belong to region")
            scope_kind = "zones"
        _canonical_gce_link(
            self.mig_self_link,
            "mig_self_link",
            project=project,
            scope_kind=scope_kind,
            scope_name=location,
            collection="instanceGroupManagers",
            resource_name=mig,
        )
        if self.provisioning_model != "SPOT":
            raise ProtocolError("provisioning_model must be SPOT")
        if (
            _integer(
                self.preemption_notice_seconds,
                "preemption_notice_seconds",
                minimum=0,
            )
            != 120
        ):
            raise ProtocolError("preemption_notice_seconds must be 120")
        if self.termination_action != "STOP":
            raise ProtocolError("termination_action must be STOP")
        if self.maintenance_policy != "TERMINATE":
            raise ProtocolError("maintenance_policy must be TERMINATE")
        _string(self.service_account, "service_account", pattern=_SERVICE_ACCOUNT)
        _controller_root(self.controller_root_uri)

    def verify_publication(self, publication: ImagePublicationV1) -> None:
        if not isinstance(publication, ImagePublicationV1):
            raise ProtocolError("publication must be ImagePublicationV1")
        if publication.digest != self.image_publication_sha256:
            raise ProtocolError("deployment image publication digest does not match")

    def accepts_zone(self, zone: str) -> bool:
        checked = _zone(zone)
        if self.mig_scope == "zonal":
            return checked == self.mig_location
        return _zone_region(checked) == self.region

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "image_publication_sha256": self.image_publication_sha256,
            "project": self.project,
            "region": self.region,
            "accelerator_type": self.accelerator_type,
            "instance_template_name": self.instance_template_name,
            "instance_template_id": self.instance_template_id,
            "instance_template_self_link": self.instance_template_self_link,
            "workload_policy_name": self.workload_policy_name,
            "workload_policy_id": self.workload_policy_id,
            "workload_policy_self_link": self.workload_policy_self_link,
            "mig_name": self.mig_name,
            "mig_id": self.mig_id,
            "mig_scope": self.mig_scope,
            "mig_location": self.mig_location,
            "mig_self_link": self.mig_self_link,
            "provisioning_model": self.provisioning_model,
            "preemption_notice_seconds": self.preemption_notice_seconds,
            "termination_action": self.termination_action,
            "maintenance_policy": self.maintenance_policy,
            "service_account": self.service_account,
            "controller_root_uri": self.controller_root_uri,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "image_publication_sha256",
            "project",
            "region",
            "accelerator_type",
            "instance_template_name",
            "instance_template_id",
            "instance_template_self_link",
            "workload_policy_name",
            "workload_policy_id",
            "workload_policy_self_link",
            "mig_name",
            "mig_id",
            "mig_scope",
            "mig_location",
            "mig_self_link",
            "provisioning_model",
            "preemption_notice_seconds",
            "termination_action",
            "maintenance_policy",
            "service_account",
            "controller_root_uri",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            image_publication_sha256=_sha256(
                obj["image_publication_sha256"], "image_publication_sha256"
            ),
            project=_project(obj["project"]),
            region=_region(obj["region"]),
            accelerator_type=_string(
                obj["accelerator_type"], "accelerator_type", pattern=_GCE_NAME
            ),
            instance_template_name=_gce_name(
                obj["instance_template_name"], "instance_template_name"
            ),
            instance_template_id=_gce_id(
                obj["instance_template_id"], "instance_template_id"
            ),
            instance_template_self_link=_string(
                obj["instance_template_self_link"], "instance_template_self_link"
            ),
            workload_policy_name=_gce_name(
                obj["workload_policy_name"], "workload_policy_name"
            ),
            workload_policy_id=_gce_id(obj["workload_policy_id"], "workload_policy_id"),
            workload_policy_self_link=_string(
                obj["workload_policy_self_link"], "workload_policy_self_link"
            ),
            mig_name=_gce_name(obj["mig_name"], "mig_name"),
            mig_id=_gce_id(obj["mig_id"], "mig_id"),
            mig_scope=_string(obj["mig_scope"], "mig_scope"),
            mig_location=_string(obj["mig_location"], "mig_location"),
            mig_self_link=_string(obj["mig_self_link"], "mig_self_link"),
            provisioning_model=_string(obj["provisioning_model"], "provisioning_model"),
            preemption_notice_seconds=_integer(
                obj["preemption_notice_seconds"],
                "preemption_notice_seconds",
                minimum=0,
            ),
            termination_action=_string(obj["termination_action"], "termination_action"),
            maintenance_policy=_string(obj["maintenance_policy"], "maintenance_policy"),
            service_account=_string(obj["service_account"], "service_account"),
            controller_root_uri=_string(
                obj["controller_root_uri"], "controller_root_uri"
            ),
        )


@dataclass(frozen=True, slots=True)
class HostAttestationV1(CanonicalRecord):
    """Boot, runtime, VFIO, JAX, and cache evidence for one TPU VM."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.host-attestation.v1"

    deployment_envelope_sha256: str
    project: str
    zone: str
    instance_name: str
    instance_id: int
    instance_self_link: str
    machine_type: str
    machine_type_self_link: str
    accelerator_type: str
    service_account: str
    boot_disk_name: str
    boot_disk_id: int
    boot_disk_self_link: str
    boot_source_image_id: int
    boot_source_image_self_link: str
    image_manifest_sha256: str
    vfio_devices: tuple[str, ...]
    jax_process_index: int
    jax_process_count: int
    jax_global_device_ids: tuple[int, ...]
    jax_device_coordinates: tuple[tuple[int, int, int], ...]
    jax_core_on_chip: tuple[int, ...]
    cache_attestation_sha256: str
    attested_at_unix_ms: int

    def __post_init__(self) -> None:
        _sha256(self.deployment_envelope_sha256, "deployment_envelope_sha256")
        project = _project(self.project)
        zone = _zone(self.zone)
        instance = _gce_name(self.instance_name, "instance_name")
        _gce_id(self.instance_id, "instance_id")
        _canonical_gce_link(
            self.instance_self_link,
            "instance_self_link",
            project=project,
            scope_kind="zones",
            scope_name=zone,
            collection="instances",
            resource_name=instance,
        )
        machine = _gce_name(self.machine_type, "machine_type")
        _canonical_gce_link(
            self.machine_type_self_link,
            "machine_type_self_link",
            project=project,
            scope_kind="zones",
            scope_name=zone,
            collection="machineTypes",
            resource_name=machine,
        )
        _string(self.accelerator_type, "accelerator_type", pattern=_GCE_NAME)
        _string(self.service_account, "service_account", pattern=_SERVICE_ACCOUNT)
        disk = _gce_name(self.boot_disk_name, "boot_disk_name")
        _gce_id(self.boot_disk_id, "boot_disk_id")
        _canonical_gce_link(
            self.boot_disk_self_link,
            "boot_disk_self_link",
            project=project,
            scope_kind="zones",
            scope_name=zone,
            collection="disks",
            resource_name=disk,
        )
        _gce_id(self.boot_source_image_id, "boot_source_image_id")
        _parse_image_link(
            self.boot_source_image_self_link, "boot_source_image_self_link"
        )
        _sha256(self.image_manifest_sha256, "image_manifest_sha256")

        vfio = _string_array(self.vfio_devices, "vfio_devices", count=4)
        if any(_VFIO_DEVICE.fullmatch(item) is None for item in vfio):
            raise ProtocolError("vfio_devices entries must be /dev/vfio/<integer>")
        object.__setattr__(self, "vfio_devices", tuple(sorted(vfio)))

        process_index = _integer(self.jax_process_index, "jax_process_index", minimum=0)
        process_count = _integer(self.jax_process_count, "jax_process_count", minimum=1)
        if process_count != 16:
            raise ProtocolError("jax_process_count must be 16 for v6e-64")
        if process_index >= 16:
            raise ProtocolError("jax_process_index must be in 0..15")
        device_ids = _device_ids(self.jax_global_device_ids, "jax_global_device_ids")
        coordinates = _device_coordinates(
            self.jax_device_coordinates, "jax_device_coordinates"
        )
        cores = _core_on_chip(self.jax_core_on_chip, "jax_core_on_chip")
        coordinate_cores = tuple(zip(coordinates, cores, strict=True))
        if len(set(coordinate_cores)) != 4:
            raise ProtocolError(
                "jax_device_coordinates/core_on_chip pairs must be unique"
            )
        paired = sorted(zip(device_ids, coordinates, cores, strict=True))
        object.__setattr__(self, "jax_global_device_ids", tuple(x[0] for x in paired))
        object.__setattr__(self, "jax_device_coordinates", tuple(x[1] for x in paired))
        object.__setattr__(self, "jax_core_on_chip", tuple(x[2] for x in paired))
        _sha256(self.cache_attestation_sha256, "cache_attestation_sha256")
        _integer(self.attested_at_unix_ms, "attested_at_unix_ms", minimum=0)

    def verify_deployment(self, deployment: DeploymentEnvelopeV1) -> None:
        if not isinstance(deployment, DeploymentEnvelopeV1):
            raise ProtocolError("deployment must be DeploymentEnvelopeV1")
        checks = {
            "deployment_envelope_sha256": deployment.digest,
            "project": deployment.project,
            "accelerator_type": deployment.accelerator_type,
            "service_account": deployment.service_account,
        }
        for field, value in checks.items():
            if getattr(self, field) != value:
                raise ProtocolError(f"host {field} does not match deployment")
        if not deployment.accepts_zone(self.zone):
            raise ProtocolError("host zone is outside the deployment MIG scope")

    def verify_publication(self, publication: ImagePublicationV1) -> None:
        if not isinstance(publication, ImagePublicationV1):
            raise ProtocolError("publication must be ImagePublicationV1")
        if self.boot_source_image_id != publication.image_id:
            raise ProtocolError("host boot source image ID does not match publication")
        if self.boot_source_image_self_link != publication.image_self_link:
            raise ProtocolError(
                "host boot source image selfLink does not match publication"
            )
        if self.image_manifest_sha256 != publication.manifest_sha256:
            raise ProtocolError("host image manifest does not match publication")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "deployment_envelope_sha256": self.deployment_envelope_sha256,
            "project": self.project,
            "zone": self.zone,
            "instance_name": self.instance_name,
            "instance_id": self.instance_id,
            "instance_self_link": self.instance_self_link,
            "machine_type": self.machine_type,
            "machine_type_self_link": self.machine_type_self_link,
            "accelerator_type": self.accelerator_type,
            "service_account": self.service_account,
            "boot_disk_name": self.boot_disk_name,
            "boot_disk_id": self.boot_disk_id,
            "boot_disk_self_link": self.boot_disk_self_link,
            "boot_source_image_id": self.boot_source_image_id,
            "boot_source_image_self_link": self.boot_source_image_self_link,
            "image_manifest_sha256": self.image_manifest_sha256,
            "vfio_devices": list(self.vfio_devices),
            "jax_process_index": self.jax_process_index,
            "jax_process_count": self.jax_process_count,
            "jax_global_device_ids": list(self.jax_global_device_ids),
            "jax_device_coordinates": [
                [x, y, z] for x, y, z in self.jax_device_coordinates
            ],
            "jax_core_on_chip": list(self.jax_core_on_chip),
            "cache_attestation_sha256": self.cache_attestation_sha256,
            "attested_at_unix_ms": self.attested_at_unix_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "deployment_envelope_sha256",
            "project",
            "zone",
            "instance_name",
            "instance_id",
            "instance_self_link",
            "machine_type",
            "machine_type_self_link",
            "accelerator_type",
            "service_account",
            "boot_disk_name",
            "boot_disk_id",
            "boot_disk_self_link",
            "boot_source_image_id",
            "boot_source_image_self_link",
            "image_manifest_sha256",
            "vfio_devices",
            "jax_process_index",
            "jax_process_count",
            "jax_global_device_ids",
            "jax_device_coordinates",
            "jax_core_on_chip",
            "cache_attestation_sha256",
            "attested_at_unix_ms",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            deployment_envelope_sha256=_sha256(
                obj["deployment_envelope_sha256"], "deployment_envelope_sha256"
            ),
            project=_project(obj["project"]),
            zone=_zone(obj["zone"]),
            instance_name=_gce_name(obj["instance_name"], "instance_name"),
            instance_id=_gce_id(obj["instance_id"], "instance_id"),
            instance_self_link=_string(obj["instance_self_link"], "instance_self_link"),
            machine_type=_gce_name(obj["machine_type"], "machine_type"),
            machine_type_self_link=_string(
                obj["machine_type_self_link"], "machine_type_self_link"
            ),
            accelerator_type=_string(
                obj["accelerator_type"], "accelerator_type", pattern=_GCE_NAME
            ),
            service_account=_string(obj["service_account"], "service_account"),
            boot_disk_name=_gce_name(obj["boot_disk_name"], "boot_disk_name"),
            boot_disk_id=_gce_id(obj["boot_disk_id"], "boot_disk_id"),
            boot_disk_self_link=_string(
                obj["boot_disk_self_link"], "boot_disk_self_link"
            ),
            boot_source_image_id=_gce_id(
                obj["boot_source_image_id"], "boot_source_image_id"
            ),
            boot_source_image_self_link=_string(
                obj["boot_source_image_self_link"],
                "boot_source_image_self_link",
            ),
            image_manifest_sha256=_sha256(
                obj["image_manifest_sha256"], "image_manifest_sha256"
            ),
            vfio_devices=_string_array(obj["vfio_devices"], "vfio_devices", count=4),
            jax_process_index=_integer(
                obj["jax_process_index"], "jax_process_index", minimum=0
            ),
            jax_process_count=_integer(
                obj["jax_process_count"], "jax_process_count", minimum=1
            ),
            jax_global_device_ids=_device_ids(
                obj["jax_global_device_ids"], "jax_global_device_ids"
            ),
            jax_device_coordinates=_device_coordinates(
                obj["jax_device_coordinates"], "jax_device_coordinates"
            ),
            jax_core_on_chip=_core_on_chip(obj["jax_core_on_chip"], "jax_core_on_chip"),
            cache_attestation_sha256=_sha256(
                obj["cache_attestation_sha256"], "cache_attestation_sha256"
            ),
            attested_at_unix_ms=_integer(
                obj["attested_at_unix_ms"], "attested_at_unix_ms", minimum=0
            ),
        )


@dataclass(frozen=True, slots=True)
class DeviceInventoryV1(CanonicalRecord):
    """A 16-host/64-device v6e-64 inventory with raw JAX coordinates."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.device-inventory.v1"

    deployment_envelope_sha256: str
    topology: str
    hosts: tuple[HostAttestationV1, ...]

    def __post_init__(self) -> None:
        deployment_digest = _sha256(
            self.deployment_envelope_sha256, "deployment_envelope_sha256"
        )
        if self.topology != "8x8":
            raise ProtocolError("topology must be exactly 8x8 for v6e-64")
        if not isinstance(self.hosts, (list, tuple)):
            raise ProtocolError("hosts must be an array")
        hosts = tuple(self.hosts)
        if len(hosts) != 16 or not all(
            isinstance(host, HostAttestationV1) for host in hosts
        ):
            raise ProtocolError(
                "hosts must contain exactly 16 HostAttestationV1 records"
            )

        def require_unique(values: list[Any], label: str) -> None:
            if len(set(values)) != len(values):
                raise ProtocolError(f"device inventory contains duplicate {label}")

        require_unique([host.instance_id for host in hosts], "instance IDs")
        require_unique([host.instance_name for host in hosts], "instance names")
        require_unique(
            [host.instance_self_link for host in hosts], "instance selfLinks"
        )
        require_unique([host.boot_disk_id for host in hosts], "boot disk IDs")
        require_unique(
            [host.boot_disk_self_link for host in hosts], "boot disk selfLinks"
        )
        require_unique(
            [host.jax_process_index for host in hosts], "JAX process indexes"
        )
        require_unique(
            [host.cache_attestation_sha256 for host in hosts],
            "cache attestations",
        )

        if {host.jax_process_index for host in hosts} != set(range(16)):
            raise ProtocolError("JAX process indexes must cover exactly 0..15")
        if any(host.jax_process_count != 16 for host in hosts):
            raise ProtocolError("every host must attest jax_process_count=16")
        if any(host.deployment_envelope_sha256 != deployment_digest for host in hosts):
            raise ProtocolError("every host must bind the same deployment envelope")

        for values, label in (
            ({host.project for host in hosts}, "project"),
            ({host.zone for host in hosts}, "zone"),
            ({host.machine_type for host in hosts}, "machine type"),
            ({host.accelerator_type for host in hosts}, "accelerator type"),
            ({host.service_account for host in hosts}, "service account"),
        ):
            if len(values) != 1:
                raise ProtocolError(f"every host must attest one {label}")
        if {host.accelerator_type for host in hosts} != {"v6e-64"}:
            raise ProtocolError("every host accelerator type must be v6e-64")

        image_ids = {host.boot_source_image_id for host in hosts}
        image_links = {host.boot_source_image_self_link for host in hosts}
        manifest_digests = {host.image_manifest_sha256 for host in hosts}
        if len(image_ids) != 1 or len(image_links) != 1 or len(manifest_digests) != 1:
            raise ProtocolError(
                "every host must attest one image identity and manifest"
            )

        device_ids = [
            device_id for host in hosts for device_id in host.jax_global_device_ids
        ]
        coordinate_cores = [
            (coordinate, core)
            for host in hosts
            for coordinate, core in zip(
                host.jax_device_coordinates,
                host.jax_core_on_chip,
                strict=True,
            )
        ]
        require_unique(device_ids, "JAX device IDs")
        require_unique(coordinate_cores, "topology coordinate/core pairs")

        object.__setattr__(
            self, "hosts", tuple(sorted(hosts, key=lambda host: host.jax_process_index))
        )

    @property
    def device_count(self) -> int:
        return sum(len(host.jax_global_device_ids) for host in self.hosts)

    def verify_deployment(
        self,
        deployment: DeploymentEnvelopeV1,
        publication: ImagePublicationV1,
    ) -> None:
        if not isinstance(deployment, DeploymentEnvelopeV1):
            raise ProtocolError("deployment must be DeploymentEnvelopeV1")
        if not isinstance(publication, ImagePublicationV1):
            raise ProtocolError("publication must be ImagePublicationV1")
        if deployment.digest != self.deployment_envelope_sha256:
            raise ProtocolError("inventory deployment digest does not match deployment")
        deployment.verify_publication(publication)
        for host in self.hosts:
            host.verify_deployment(deployment)
            host.verify_publication(publication)

    def verify_identity(
        self,
        identity: DeploymentIdentityV1,
        implementation: ImplementationIdentityV1,
        deployment: DeploymentEnvelopeV1,
    ) -> None:
        """Verify the compact attempt identity against its content records."""

        if not isinstance(identity, DeploymentIdentityV1):
            raise ProtocolError("identity must be DeploymentIdentityV1")
        if not isinstance(implementation, ImplementationIdentityV1):
            raise ProtocolError("implementation must be ImplementationIdentityV1")
        if not isinstance(deployment, DeploymentEnvelopeV1):
            raise ProtocolError("deployment must be DeploymentEnvelopeV1")
        expected = {
            "implementation_identity_sha256": implementation.digest,
            "deployment_envelope_sha256": deployment.digest,
            "device_inventory_sha256": self.digest,
        }
        for field, value in expected.items():
            if getattr(identity, field) != value:
                raise ProtocolError(f"deployment identity {field} does not match")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "deployment_envelope_sha256": self.deployment_envelope_sha256,
            "topology": self.topology,
            "hosts": [host.as_dict() for host in self.hosts],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _strict_object(
            value,
            schema=cls.SCHEMA,
            fields=("deployment_envelope_sha256", "topology", "hosts"),
        )
        raw_hosts = obj["hosts"]
        if not isinstance(raw_hosts, list):
            raise ProtocolError("hosts must be an array")
        return cls(
            deployment_envelope_sha256=_sha256(
                obj["deployment_envelope_sha256"], "deployment_envelope_sha256"
            ),
            topology=_string(obj["topology"], "topology"),
            hosts=tuple(HostAttestationV1.from_dict(host) for host in raw_hosts),
        )


__all__ = [
    "DeploymentEnvelopeV1",
    "DeviceInventoryV1",
    "HostAttestationV1",
    "ImageManifestV1",
    "ImagePublicationV1",
    "ImageRecipeV1",
]
