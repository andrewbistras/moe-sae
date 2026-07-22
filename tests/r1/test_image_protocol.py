from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib

import pytest

from moe_sae.r1.image_protocol import (
    DeploymentEnvelopeV1,
    DeviceInventoryV1,
    HostAttestationV1,
    ImageManifestV1,
    ImagePublicationV1,
    ImageRecipeV1,
)
from moe_sae.r1.protocol import (
    DeploymentIdentityV1,
    ImplementationIdentityV1,
    ProtocolError,
    canonical_json_bytes,
)


PROJECT = "neural-dragnet"
REGION = "us-central1"
ZONE = "us-central1-b"
SERVICE_ACCOUNT = "r1-worker@neural-dragnet.iam.gserviceaccount.com"
BASE_IMAGE_ID = 1_012_345_678_901_234_567
IMAGE_ID = 2_012_345_678_901_234_567
BASE_IMAGE_LINK = (
    "https://www.googleapis.com/compute/v1/projects/ubuntu-os-accelerator-images/"
    "global/images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e-v20260719"
)
IMAGE_NAME = "r1-v6e-20260722"
IMAGE_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
    f"global/images/{IMAGE_NAME}"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _recipe() -> ImageRecipeV1:
    return ImageRecipeV1(
        base_image_id=BASE_IMAGE_ID,
        base_image_self_link=BASE_IMAGE_LINK,
        repository_commit="1" * 40,
        repository_archive_sha256=_sha("repository-archive"),
        pyproject_sha256=_sha("pyproject"),
        uv_lock_sha256=_sha("uv-lock"),
        maxtext_commit="2" * 40,
        python_version="3.12.11",
        uv_version="0.8.15",
        packer_version="1.14.1",
        packer_template_sha256=_sha("packer-template"),
        provisioner_sha256=_sha("provisioner"),
        service_unit_name="moe-sae-r1.service",
        service_unit_sha256=_sha("service-unit"),
    )


def _manifest(recipe: ImageRecipeV1 | None = None) -> ImageManifestV1:
    recipe = recipe or _recipe()
    return ImageManifestV1(
        recipe_sha256=recipe.digest,
        os_release_id="ubuntu",
        os_release_version_id="22.04",
        kernel_release="6.8.0-1030-gcp",
        architecture="x86_64",
        repository_commit=recipe.repository_commit,
        installed_repository_archive_sha256=recipe.repository_archive_sha256,
        maxtext_commit=recipe.maxtext_commit,
        python_version=recipe.python_version,
        uv_version=recipe.uv_version,
        jax_version="0.8.2",
        libtpu_version="0.0.36",
        installed_distribution_sha256=_sha("installed-distribution"),
        sbom_sha256=_sha("sbom"),
        runtime_user="moe_sae",
        runtime_user_spec_sha256=_sha("runtime-user-spec"),
        service_unit_name=recipe.service_unit_name,
        service_unit_sha256=recipe.service_unit_sha256,
    )


def _publication(
    manifest: ImageManifestV1 | None = None,
) -> ImagePublicationV1:
    manifest = manifest or _manifest()
    operation = "operation-172161"
    return ImagePublicationV1(
        image_project=PROJECT,
        image_name=IMAGE_NAME,
        image_id=IMAGE_ID,
        image_self_link=IMAGE_LINK,
        base_image_id=BASE_IMAGE_ID,
        base_image_self_link=BASE_IMAGE_LINK,
        manifest_sha256=manifest.digest,
        creation_operation_name=operation,
        creation_operation_id=3_012_345_678_901_234_567,
        creation_operation_self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
            f"global/operations/{operation}"
        ),
        creation_operation_status="DONE",
    )


def _deployment(
    publication: ImagePublicationV1 | None = None,
    *,
    mig_scope: str = "zonal",
) -> DeploymentEnvelopeV1:
    publication = publication or _publication()
    if mig_scope == "zonal":
        mig_location = ZONE
        mig_segment = f"zones/{ZONE}"
    else:
        mig_location = REGION
        mig_segment = f"regions/{REGION}"
    return DeploymentEnvelopeV1(
        image_publication_sha256=publication.digest,
        project=PROJECT,
        region=REGION,
        accelerator_type="v6e-64",
        instance_template_name="r1-v6e-template",
        instance_template_id=4_012_345_678_901_234_567,
        instance_template_self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
            "global/instanceTemplates/r1-v6e-template"
        ),
        workload_policy_name="r1-high-throughput",
        workload_policy_id=5_012_345_678_901_234_567,
        workload_policy_self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
            f"regions/{REGION}/resourcePolicies/r1-high-throughput"
        ),
        mig_name="r1-v6e-slice",
        mig_id=6_012_345_678_901_234_567,
        mig_scope=mig_scope,
        mig_location=mig_location,
        mig_self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
            f"{mig_segment}/instanceGroupManagers/r1-v6e-slice"
        ),
        provisioning_model="SPOT",
        preemption_notice_seconds=120,
        termination_action="STOP",
        maintenance_policy="TERMINATE",
        service_account=SERVICE_ACCOUNT,
        controller_root_uri="gs://r1-control/campaigns/headline-01/",
    )


def _host(
    process_index: int,
    deployment: DeploymentEnvelopeV1 | None = None,
    publication: ImagePublicationV1 | None = None,
) -> HostAttestationV1:
    publication = publication or _publication()
    deployment = deployment or _deployment(publication)
    instance = f"r1-v6e-slice-{process_index:02d}"
    disk = f"{instance}-boot"
    first_device = process_index * 4
    device_ids = tuple(range(first_device, first_device + 4))
    coordinates = tuple((device // 8, device % 8, 0) for device in device_ids)
    return HostAttestationV1(
        deployment_envelope_sha256=deployment.digest,
        project=PROJECT,
        zone=ZONE,
        instance_name=instance,
        instance_id=7_000_000_000_000_000_000 + process_index,
        instance_self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
            f"zones/{ZONE}/instances/{instance}"
        ),
        machine_type="ct6e-standard-4t",
        machine_type_self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
            f"zones/{ZONE}/machineTypes/ct6e-standard-4t"
        ),
        accelerator_type="v6e-64",
        service_account=SERVICE_ACCOUNT,
        boot_disk_name=disk,
        boot_disk_id=8_000_000_000_000_000_000 + process_index,
        boot_disk_self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
            f"zones/{ZONE}/disks/{disk}"
        ),
        boot_source_image_id=publication.image_id,
        boot_source_image_self_link=publication.image_self_link,
        image_manifest_sha256=publication.manifest_sha256,
        vfio_devices=("/dev/vfio/0", "/dev/vfio/1", "/dev/vfio/2", "/dev/vfio/3"),
        jax_process_index=process_index,
        jax_process_count=16,
        jax_global_device_ids=device_ids,
        jax_device_coordinates=coordinates,
        jax_core_on_chip=(0, 0, 0, 0),
        cache_attestation_sha256=_sha(f"cache-attestation-{process_index}"),
        attested_at_unix_ms=1_753_201_000_000 + process_index,
    )


def _inventory(
    deployment: DeploymentEnvelopeV1 | None = None,
    publication: ImagePublicationV1 | None = None,
) -> DeviceInventoryV1:
    publication = publication or _publication()
    deployment = deployment or _deployment(publication)
    return DeviceInventoryV1(
        deployment_envelope_sha256=deployment.digest,
        topology="8x8",
        hosts=tuple(_host(index, deployment, publication) for index in range(16)),
    )


def _implementation(
    manifest: ImageManifestV1 | None = None,
) -> ImplementationIdentityV1:
    manifest = manifest or _manifest()
    return ImplementationIdentityV1(
        repository_commit=manifest.repository_commit,
        maxtext_commit=manifest.maxtext_commit,
        image_manifest_sha256=manifest.digest,
        python_version=manifest.python_version,
        jax_version=manifest.jax_version,
        libtpu_version=manifest.libtpu_version,
        adapter_version="deepseek-r1-prefix-v1",
        converter_version="deepseek-r1-converter-v1",
    )


def _deployment_identity(
    implementation: ImplementationIdentityV1,
    deployment: DeploymentEnvelopeV1,
    inventory: DeviceInventoryV1,
) -> DeploymentIdentityV1:
    return DeploymentIdentityV1(
        implementation_identity_sha256=implementation.digest,
        deployment_envelope_sha256=deployment.digest,
        device_inventory_sha256=inventory.digest,
        attempt_id="attempt-0001",
    )


def test_full_provenance_chain_round_trips_and_verifies() -> None:
    recipe = _recipe()
    manifest = _manifest(recipe)
    publication = _publication(manifest)
    deployment = _deployment(publication)
    inventory = _inventory(deployment, publication)
    implementation = _implementation(manifest)
    deployment_identity = _deployment_identity(implementation, deployment, inventory)

    manifest.verify_recipe(recipe)
    manifest.verify_implementation(implementation)
    publication.verify_manifest(manifest, recipe)
    deployment.verify_publication(publication)
    inventory.verify_deployment(deployment, publication)
    inventory.verify_identity(deployment_identity, implementation, deployment)

    records = (
        recipe,
        manifest,
        publication,
        implementation,
        deployment,
        inventory.hosts[0],
        inventory,
        deployment_identity,
    )
    for record in records:
        decoded = type(record).from_json(record.canonical_bytes())
        assert decoded == record
        assert decoded.digest == record.digest
        assert record.canonical_bytes() == canonical_json_bytes(record.as_dict())
    assert inventory.device_count == 64


def test_public_r1_package_exports_all_image_records() -> None:
    import moe_sae.r1 as r1

    for record_type in (
        ImageRecipeV1,
        ImageManifestV1,
        ImagePublicationV1,
        DeploymentEnvelopeV1,
        HostAttestationV1,
        DeviceInventoryV1,
    ):
        assert getattr(r1, record_type.__name__) is record_type
        assert record_type.__name__ in r1.__all__


def test_compact_identities_reject_mismatched_manifest_envelope_or_inventory() -> None:
    manifest = _manifest()
    implementation = _implementation(manifest)
    publication = _publication(manifest)
    deployment = _deployment(publication)
    inventory = _inventory(deployment, publication)
    identity = _deployment_identity(implementation, deployment, inventory)

    with pytest.raises(ProtocolError, match="image_manifest_sha256"):
        manifest.verify_implementation(
            replace(implementation, image_manifest_sha256=_sha("wrong-manifest"))
        )
    with pytest.raises(ProtocolError, match="implementation_identity_sha256"):
        inventory.verify_identity(
            replace(identity, implementation_identity_sha256=_sha("wrong-impl")),
            implementation,
            deployment,
        )
    with pytest.raises(ProtocolError, match="deployment_envelope_sha256"):
        inventory.verify_identity(
            replace(identity, deployment_envelope_sha256=_sha("wrong-envelope")),
            implementation,
            deployment,
        )
    with pytest.raises(ProtocolError, match="device_inventory_sha256"):
        inventory.verify_identity(
            replace(identity, device_inventory_sha256=_sha("wrong-inventory")),
            implementation,
            deployment,
        )


def test_image_identity_is_numeric_id_plus_manifest_not_fake_disk_digest() -> None:
    publication = _publication()
    host = _host(0)

    assert type(publication.image_id) is int
    assert type(host.boot_source_image_id) is int
    assert host.boot_source_image_id == publication.image_id
    assert host.image_manifest_sha256 == publication.manifest_sha256
    assert "boot_disk_sha256" not in host.as_dict()
    assert "image_digest" not in publication.as_dict()


def test_recipe_rejects_image_family_alias_and_malformed_hash() -> None:
    with pytest.raises(ProtocolError, match="canonical GCE image selfLink"):
        replace(_recipe(), base_image_self_link="ubuntu-os-cloud:ubuntu-2404-lts")
    with pytest.raises(ProtocolError, match="invalid format"):
        replace(_recipe(), uv_lock_sha256="A" * 64)


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (_recipe, "base_image_id"),
        (_manifest, "recipe_sha256"),
        (_publication, "image_id"),
        (_deployment, "instance_template_id"),
        (lambda: _host(0), "instance_id"),
        (_inventory, "topology"),
    ],
)
def test_unknown_and_missing_fields_fail_closed(factory, field: str) -> None:
    record = factory()
    payload = record.as_dict()
    payload["unexpected"] = "surprise"
    with pytest.raises(ProtocolError, match="unknown fields"):
        type(record).from_dict(payload)

    payload = record.as_dict()
    del payload[field]
    with pytest.raises(ProtocolError, match="missing fields"):
        type(record).from_dict(payload)


def test_unknown_field_in_nested_host_fails_closed() -> None:
    payload = _inventory().as_dict()
    payload["hosts"][0]["unknown_attestation"] = "unsafe"
    with pytest.raises(ProtocolError, match="unknown fields"):
        DeviceInventoryV1.from_dict(payload)


@pytest.mark.parametrize(
    ("record", "field"),
    [
        (_recipe(), "base_image_id"),
        (_publication(), "creation_operation_id"),
        (_deployment(), "preemption_notice_seconds"),
        (_host(0), "boot_source_image_id"),
        (_host(0), "jax_process_index"),
    ],
)
def test_boolean_is_never_accepted_as_numeric_identity(record, field: str) -> None:
    payload = record.as_dict()
    payload[field] = True
    with pytest.raises(ProtocolError, match="booleans are forbidden"):
        type(record).from_dict(payload)


def test_duplicate_json_field_fails_before_record_construction() -> None:
    recipe = _recipe().canonical_json()
    duplicate = recipe[:-1] + ',"base_image_id":123}'
    with pytest.raises(ProtocolError, match="duplicate JSON field"):
        ImageRecipeV1.from_json(duplicate)


def test_manifest_recipe_mismatch_fails_for_every_bound_observation() -> None:
    recipe = _recipe()
    mutations = (
        {"recipe_sha256": _sha("wrong-recipe")},
        {"repository_commit": "3" * 40},
        {"installed_repository_archive_sha256": _sha("wrong-archive")},
        {"maxtext_commit": "4" * 40},
        {"python_version": "3.12.12"},
        {"uv_version": "0.8.16"},
        {"service_unit_name": "wrong.service"},
        {"service_unit_sha256": _sha("wrong-unit")},
    )
    for mutation in mutations:
        with pytest.raises(ProtocolError, match="does not match recipe"):
            replace(_manifest(recipe), **mutation).verify_recipe(recipe)


def test_publication_rejects_mismatched_self_links_and_unfinished_operation() -> None:
    publication = _publication()
    with pytest.raises(ProtocolError, match="image_self_link must equal"):
        replace(publication, image_self_link=publication.image_self_link + "-other")
    with pytest.raises(ProtocolError, match="creation_operation_self_link must equal"):
        replace(
            publication,
            creation_operation_self_link=(
                publication.creation_operation_self_link.replace(
                    PROJECT, "other-project"
                )
            ),
        )
    with pytest.raises(ProtocolError, match="must be DONE"):
        replace(publication, creation_operation_status="RUNNING")


@pytest.mark.parametrize(
    ("record", "field"),
    [
        (_deployment(), "instance_template_self_link"),
        (_deployment(), "workload_policy_self_link"),
        (_deployment(), "mig_self_link"),
        (_host(0), "instance_self_link"),
        (_host(0), "machine_type_self_link"),
        (_host(0), "boot_disk_self_link"),
        (_host(0), "boot_source_image_self_link"),
    ],
)
def test_every_gce_self_link_is_exact(record, field: str) -> None:
    with pytest.raises(ProtocolError, match="selfLink"):
        replace(record, **{field: getattr(record, field) + "?mutable=true"})


def test_publication_verification_catches_manifest_and_base_substitution() -> None:
    recipe = _recipe()
    manifest = _manifest(recipe)
    publication = _publication(manifest)
    with pytest.raises(ProtocolError, match="manifest digest"):
        publication.verify_manifest(replace(manifest, sbom_sha256=_sha("different")))
    with pytest.raises(ProtocolError, match="base image"):
        replace(publication, base_image_id=BASE_IMAGE_ID + 1).verify_manifest(
            manifest, recipe
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"provisioning_model": "STANDARD"}, "must be SPOT"),
        ({"preemption_notice_seconds": 30}, "must be 120"),
        ({"termination_action": "DELETE"}, "must be STOP"),
        ({"maintenance_policy": "MIGRATE"}, "must be TERMINATE"),
        ({"accelerator_type": "v5p-64"}, "must be v6e-64"),
        ({"controller_root_uri": "https://example.test/control"}, "gs:// prefix"),
        ({"mig_location": "us-east5-a"}, "belong to region"),
    ],
)
def test_deployment_envelope_freezes_spot_lifecycle(mutation, message: str) -> None:
    with pytest.raises(ProtocolError, match=message):
        replace(_deployment(), **mutation)


def test_regional_and_zonal_mig_self_links_are_exact() -> None:
    zonal = _deployment(mig_scope="zonal")
    regional = _deployment(mig_scope="regional")
    assert "/zones/us-central1-b/instanceGroupManagers/" in zonal.mig_self_link
    assert "/regions/us-central1/instanceGroupManagers/" in regional.mig_self_link
    assert regional.accepts_zone("us-central1-f")
    assert not regional.accepts_zone("us-east5-a")
    with pytest.raises(ProtocolError, match="mig_self_link must equal"):
        replace(regional, mig_self_link=zonal.mig_self_link)


def test_host_requires_four_unique_vfio_devices_and_four_jax_devices() -> None:
    host = _host(0)
    with pytest.raises(ProtocolError, match="exactly 4"):
        replace(host, vfio_devices=host.vfio_devices[:3])
    with pytest.raises(ProtocolError, match="duplicates"):
        replace(host, vfio_devices=("/dev/vfio/0",) * 4)
    with pytest.raises(ProtocolError, match="/dev/vfio"):
        replace(
            host,
            vfio_devices=("/dev/vfio/0", "/dev/vfio/1", "/dev/vfio/2", "/dev/tpu/3"),
        )
    with pytest.raises(ProtocolError, match="duplicate devices"):
        replace(host, jax_global_device_ids=(0, 0, 1, 2))
    with pytest.raises(ProtocolError, match="coordinate.*pairs must be unique"):
        replace(
            host,
            jax_device_coordinates=((0, 0, 0),) * 4,
            jax_core_on_chip=(0, 0, 0, 0),
        )
    with pytest.raises(ProtocolError, match="booleans are forbidden"):
        replace(host, jax_core_on_chip=(0, 0, 0, True))


def test_host_canonicalizes_local_enumeration_without_losing_pairing() -> None:
    host = _host(0)
    reordered = replace(
        host,
        vfio_devices=tuple(reversed(host.vfio_devices)),
        jax_global_device_ids=tuple(reversed(host.jax_global_device_ids)),
        jax_device_coordinates=tuple(reversed(host.jax_device_coordinates)),
        jax_core_on_chip=tuple(reversed(host.jax_core_on_chip)),
    )
    assert reordered == host
    assert reordered.digest == host.digest


def test_raw_jax_coords_and_core_are_preserved_without_invented_grid_mapping() -> None:
    host = _host(0)
    raw = replace(
        host,
        jax_device_coordinates=(
            (80, 0, 4),
            (80, 0, 4),
            (81, 3, 9),
            (99, 2, 1),
        ),
        jax_core_on_chip=(0, 1, 0, 7),
    )
    assert raw.jax_device_coordinates[0] == (80, 0, 4)
    assert raw.jax_core_on_chip == (0, 1, 0, 7)


def test_host_verification_rejects_boot_or_placement_substitution() -> None:
    publication = _publication()
    deployment = _deployment(publication)
    host = _host(0, deployment, publication)

    with pytest.raises(ProtocolError, match="source image ID"):
        replace(host, boot_source_image_id=publication.image_id + 1).verify_publication(
            publication
        )
    with pytest.raises(ProtocolError, match="manifest"):
        replace(
            host, image_manifest_sha256=_sha("different-manifest")
        ).verify_publication(publication)
    with pytest.raises(ProtocolError, match="service_account"):
        replace(
            host,
            service_account="other-worker@neural-dragnet.iam.gserviceaccount.com",
        ).verify_deployment(deployment)


def test_inventory_is_order_independent_and_exactly_16_by_4() -> None:
    inventory = _inventory()
    reversed_inventory = DeviceInventoryV1(
        deployment_envelope_sha256=inventory.deployment_envelope_sha256,
        topology="8x8",
        hosts=tuple(reversed(inventory.hosts)),
    )
    assert reversed_inventory == inventory
    assert reversed_inventory.digest == inventory.digest
    assert tuple(host.jax_process_index for host in inventory.hosts) == tuple(range(16))
    assert inventory.device_count == 64


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda hosts: hosts[:15],
            "exactly 16",
        ),
        (
            lambda hosts: (
                hosts[:15] + (replace(hosts[15], instance_id=hosts[0].instance_id),)
            ),
            "duplicate instance IDs",
        ),
        (
            lambda hosts: (
                hosts[:15]
                + (replace(hosts[15], jax_process_index=hosts[0].jax_process_index),)
            ),
            "duplicate JAX process indexes",
        ),
        (
            lambda hosts: (
                hosts[:15]
                + (
                    replace(
                        hosts[15], jax_global_device_ids=hosts[0].jax_global_device_ids
                    ),
                )
            ),
            "duplicate JAX device IDs",
        ),
        (
            lambda hosts: (
                hosts[:15]
                + (
                    replace(
                        hosts[15],
                        jax_device_coordinates=hosts[0].jax_device_coordinates,
                    ),
                )
            ),
            "duplicate topology coordinate/core pairs",
        ),
    ],
)
def test_inventory_rejects_missing_or_duplicate_hosts_devices(
    mutator, message: str
) -> None:
    inventory = _inventory()
    with pytest.raises(ProtocolError, match=message):
        DeviceInventoryV1(
            deployment_envelope_sha256=inventory.deployment_envelope_sha256,
            topology="8x8",
            hosts=mutator(inventory.hosts),
        )


def test_inventory_rejects_wrong_topology_declaration_or_negative_raw_coordinate() -> (
    None
):
    inventory = _inventory()
    with pytest.raises(ProtocolError, match="topology must be exactly 8x8"):
        replace(inventory, topology="4x16")

    last = inventory.hosts[-1]
    with pytest.raises(ProtocolError, match="must be >= 0"):
        replace(
            last,
            jax_device_coordinates=(
                last.jax_device_coordinates[0],
                last.jax_device_coordinates[1],
                last.jax_device_coordinates[2],
                (-1, 0, 0),
            ),
        )


def test_inventory_requires_uniqueness_but_does_not_guess_v6e_coordinate_layout() -> (
    None
):
    inventory = _inventory()
    hosts = []
    for host in inventory.hosts:
        coordinates = tuple(
            (10_000 + device_id, 17, 23) for device_id in host.jax_global_device_ids
        )
        hosts.append(
            replace(
                host,
                jax_device_coordinates=coordinates,
                jax_core_on_chip=(3, 3, 3, 3),
            )
        )
    observed = replace(inventory, hosts=tuple(hosts))
    assert observed.device_count == 64
    assert observed.topology == "8x8"


def test_inventory_rejects_mixed_image_and_deployment_attestations() -> None:
    inventory = _inventory()
    mixed_image = replace(
        inventory.hosts[-1],
        boot_source_image_id=IMAGE_ID + 1,
    )
    with pytest.raises(ProtocolError, match="one image identity"):
        replace(inventory, hosts=inventory.hosts[:-1] + (mixed_image,))

    mixed_deployment = replace(
        inventory.hosts[-1],
        deployment_envelope_sha256=_sha("other-deployment"),
    )
    with pytest.raises(ProtocolError, match="same deployment envelope"):
        replace(inventory, hosts=inventory.hosts[:-1] + (mixed_deployment,))


def test_records_are_frozen_and_slotted() -> None:
    recipe = _recipe()
    with pytest.raises(FrozenInstanceError):
        recipe.python_version = "3.13"  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        recipe.extra = "not allowed"  # type: ignore[attr-defined]
