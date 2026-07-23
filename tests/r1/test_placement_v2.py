from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from moe_sae.r1.image_protocol import ImagePublicationV1
from moe_sae.r1.placement_v2 import (
    BaseImageFamilyResolutionReceiptV2,
    ControllerPlacementReceiptV2,
    DeploymentEnvelopeV2,
    GCSObjectVersionRefV2,
    InstanceTemplateObservationV2,
    MIGObservationV2,
    ManagedInstanceObservationV2,
    ResourcePolicyObservationV2,
)
from moe_sae.r1.protocol import ProtocolError


PROJECT = "neural-dragnet"
REGION = "us-central1"
ZONE = "us-central1-b"
WORKER_EMAIL = "r1-worker@neural-dragnet.iam.gserviceaccount.com"
WORKER_ID = "123456789012345678901"
CONTROLLER_EMAIL = "r1-controller@neural-dragnet.iam.gserviceaccount.com"
CONTROLLER_ID = "223456789012345678901"
CHALLENGE = "c" * 64
IMAGE_NAME = "r1-v6e-runtime-20260723"
IMAGE_ID = 2_012_345_678_901_234_567
IMAGE_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
    f"global/images/{IMAGE_NAME}"
)
BASE_IMAGE_NAME = "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e-v20260719"
BASE_IMAGE_ID = 1_012_345_678_901_234_567
BASE_IMAGE_LINK = (
    "https://www.googleapis.com/compute/v1/projects/"
    "ubuntu-os-accelerator-images/global/images/"
    f"{BASE_IMAGE_NAME}"
)
TPU_LICENSE = (
    "https://www.googleapis.com/compute/v1/projects/ubuntu-os-cloud/"
    "global/licenses/ubuntu-2204-lts-tpu"
)
TEMPLATE_NAME = "r1-v6e-template"
TEMPLATE_ID = 4_012_345_678_901_234_567
TEMPLATE_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
    f"global/instanceTemplates/{TEMPLATE_NAME}"
)
POLICY_NAME = "r1-v6e-high-throughput"
POLICY_ID = 5_012_345_678_901_234_567
POLICY_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
    f"regions/{REGION}/resourcePolicies/{POLICY_NAME}"
)
MACHINE_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
    f"zones/{ZONE}/machineTypes/ct6e-standard-4t"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _evidence(label: str = "placement") -> tuple[bytes, GCSObjectVersionRefV2]:
    value = json.dumps(
        {"source": "authenticated-controller-api", "label": label},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return value, GCSObjectVersionRefV2(
        uri=f"gs://r1-control/campaigns/headline-01/evidence/{label}.json",
        generation=17,
        size_bytes=len(value),
        sha256=hashlib.sha256(value).hexdigest(),
    )


def _resolution() -> BaseImageFamilyResolutionReceiptV2:
    _, evidence = _evidence("base-image-resolution")
    return BaseImageFamilyResolutionReceiptV2(
        controller_service_account_email=CONTROLLER_EMAIL,
        controller_service_account_unique_id=CONTROLLER_ID,
        controller_challenge=CHALLENGE,
        api_evidence=evidence,
        family_project="ubuntu-os-accelerator-images",
        family_name="ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e",
        family_alias=(
            "https://www.googleapis.com/compute/v1/projects/"
            "ubuntu-os-accelerator-images/global/images/family/"
            "ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e"
        ),
        resolved_image_name=BASE_IMAGE_NAME,
        resolved_image_id=BASE_IMAGE_ID,
        resolved_image_self_link=BASE_IMAGE_LINK,
        resolved_status="READY",
        architecture="X86_64",
        license_self_links=(TPU_LICENSE,),
        observed_at_unix_ms=1_753_220_000_000,
    )


def _publication() -> ImagePublicationV1:
    operation = "publish-r1-v6e-runtime"
    return ImagePublicationV1(
        image_project=PROJECT,
        image_name=IMAGE_NAME,
        image_id=IMAGE_ID,
        image_self_link=IMAGE_LINK,
        base_image_id=BASE_IMAGE_ID,
        base_image_self_link=BASE_IMAGE_LINK,
        manifest_sha256=_sha("image-manifest"),
        creation_operation_name=operation,
        creation_operation_id=3_012_345_678_901_234_567,
        creation_operation_self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
            f"global/operations/{operation}"
        ),
        creation_operation_status="DONE",
    )


def _mig_identity(attempt: int) -> tuple[str, int, str]:
    name = f"r1-v6e-slice-a{attempt:04d}"
    resource_id = 6_000_000_000_000_000_000 + attempt
    self_link = (
        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
        f"zones/{ZONE}/instanceGroupManagers/{name}"
    )
    return name, resource_id, self_link


def _template_identity(attempt: int) -> tuple[str, int, str]:
    if attempt == 1:
        return TEMPLATE_NAME, TEMPLATE_ID, TEMPLATE_LINK
    name = f"r1-v6e-template-a{attempt:04d}"
    resource_id = TEMPLATE_ID + attempt - 1
    self_link = (
        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
        f"global/instanceTemplates/{name}"
    )
    return name, resource_id, self_link


def _launch_ref(attempt: int = 1) -> GCSObjectVersionRefV2:
    value = json.dumps(
        {"slice_attempt_id": f"slice-attempt-{attempt:04d}"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return GCSObjectVersionRefV2(
        uri=(
            "gs://r1-control/campaigns/headline-01/launch/"
            f"slice-attempt-{attempt:04d}.json"
        ),
        generation=100 + attempt,
        size_bytes=len(value),
        sha256=hashlib.sha256(value).hexdigest(),
    )


def _deployment(attempt: int = 1) -> DeploymentEnvelopeV2:
    resolution = _resolution()
    publication = _publication()
    mig_name, mig_id, mig_link = _mig_identity(attempt)
    template_name, template_id, template_link = _template_identity(attempt)
    return DeploymentEnvelopeV2(
        base_image_resolution_sha256=resolution.digest,
        image_publication_sha256=publication.digest,
        project=PROJECT,
        region=REGION,
        zone=ZONE,
        slice_attempt_id=f"slice-attempt-{attempt:04d}",
        accelerator_type="v6e-64",
        machine_type="ct6e-standard-4t",
        machine_type_self_link=MACHINE_LINK,
        topology="8x8",
        vm_count=16,
        chips_per_vm=4,
        chip_count=64,
        image_name=IMAGE_NAME,
        image_id=IMAGE_ID,
        image_self_link=IMAGE_LINK,
        instance_template_name=template_name,
        instance_template_id=template_id,
        instance_template_self_link=template_link,
        resource_policy_name=POLICY_NAME,
        resource_policy_id=POLICY_ID,
        resource_policy_self_link=POLICY_LINK,
        mig_name=mig_name,
        mig_id=mig_id,
        mig_self_link=mig_link,
        mig_scope="ZONAL",
        target_size=16,
        target_size_policy="BULK",
        default_action_on_failure="DO_NOTHING",
        replacement_policy="FRESH_WHOLE_MIG",
        provisioning_model="SPOT",
        termination_action="DELETE",
        maintenance_policy="TERMINATE",
        service_account_email=WORKER_EMAIL,
        service_account_unique_id=WORKER_ID,
        controller_root_uri="gs://r1-control/campaigns/headline-01/",
        launch_ref=_launch_ref(attempt),
    )


def _template(attempt: int = 1) -> InstanceTemplateObservationV2:
    name, resource_id, self_link = _template_identity(attempt)
    return InstanceTemplateObservationV2(
        project=PROJECT,
        zone=ZONE,
        name=name,
        resource_id=resource_id,
        self_link=self_link,
        machine_type="ct6e-standard-4t",
        machine_type_self_link=MACHINE_LINK,
        image_name=IMAGE_NAME,
        image_id=IMAGE_ID,
        image_self_link=IMAGE_LINK,
        provisioning_model="SPOT",
        termination_action="DELETE",
        maintenance_policy="TERMINATE",
        automatic_restart=False,
        service_account_email=WORKER_EMAIL,
        metadata_keys=("moe-sae-r1-launch-ref",),
        launch_ref=_launch_ref(attempt),
    )


def _policy() -> ResourcePolicyObservationV2:
    return ResourcePolicyObservationV2(
        project=PROJECT,
        region=REGION,
        name=POLICY_NAME,
        resource_id=POLICY_ID,
        self_link=POLICY_LINK,
        policy_type="HIGH_THROUGHPUT",
        topology="8x8",
    )


def _instances(attempt: int = 1) -> tuple[ManagedInstanceObservationV2, ...]:
    mig_name, _, mig_link = _mig_identity(attempt)
    _, _, template_link = _template_identity(attempt)
    result = []
    for index in range(16):
        name = f"{mig_name}-{index:04d}"
        result.append(
            ManagedInstanceObservationV2(
                project=PROJECT,
                zone=ZONE,
                name=name,
                resource_id=7_000_000_000_000_000_000 + attempt * 100 + index,
                self_link=(
                    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
                    f"zones/{ZONE}/instances/{name}"
                ),
                mig_self_link=mig_link,
                instance_template_self_link=template_link,
                machine_type_self_link=MACHINE_LINK,
                status="RUNNING",
                current_action="NONE",
                service_account_email=WORKER_EMAIL,
                boot_source_image_id=IMAGE_ID,
                boot_source_image_self_link=IMAGE_LINK,
            )
        )
    return tuple(result)


def _mig(attempt: int = 1) -> MIGObservationV2:
    name, resource_id, self_link = _mig_identity(attempt)
    _, template_id, template_link = _template_identity(attempt)
    instances = _instances(attempt)
    return MIGObservationV2(
        project=PROJECT,
        zone=ZONE,
        name=name,
        resource_id=resource_id,
        self_link=self_link,
        instance_template_id=template_id,
        instance_template_self_link=template_link,
        resource_policy_id=POLICY_ID,
        resource_policy_self_link=POLICY_LINK,
        target_size=16,
        target_size_policy="BULK",
        default_action_on_failure="DO_NOTHING",
        is_stable=True,
        member_instance_self_links=tuple(item.self_link for item in instances),
    )


def _receipt(attempt: int = 1) -> ControllerPlacementReceiptV2:
    deployment = _deployment(attempt)
    _, evidence = _evidence(f"placement-{attempt}")
    return ControllerPlacementReceiptV2(
        deployment_envelope_sha256=deployment.digest,
        base_image_resolution_sha256=deployment.base_image_resolution_sha256,
        image_publication_sha256=deployment.image_publication_sha256,
        slice_attempt_id=deployment.slice_attempt_id,
        controller_service_account_email=CONTROLLER_EMAIL,
        controller_service_account_unique_id=CONTROLLER_ID,
        controller_challenge=CHALLENGE,
        api_evidence=evidence,
        worker_service_account_email=WORKER_EMAIL,
        worker_service_account_unique_id=WORKER_ID,
        instance_template=_template(attempt),
        resource_policy=_policy(),
        mig=_mig(attempt),
        instances=_instances(attempt),
        observed_at_unix_ms=1_753_220_100_000 + attempt,
    )


def test_full_placement_chain_round_trips_and_verifies() -> None:
    resolution = _resolution()
    publication = _publication()
    deployment = _deployment()
    receipt = _receipt()

    resolution.verify_admission(
        controller_service_account_email=CONTROLLER_EMAIL,
        controller_service_account_unique_id=CONTROLLER_ID,
        controller_challenge=CHALLENGE,
        api_evidence_bytes=_evidence("base-image-resolution")[0],
    )
    deployment.verify_image_chain(resolution, publication)
    receipt.verify_admission(
        deployment,
        controller_service_account_email=CONTROLLER_EMAIL,
        controller_service_account_unique_id=CONTROLLER_ID,
        controller_challenge=CHALLENGE,
        api_evidence_bytes=_evidence("placement-1")[0],
    )

    records = (
        resolution.api_evidence,
        resolution,
        deployment,
        deployment.launch_ref,
        receipt.instance_template,
        receipt.resource_policy,
        receipt.mig,
        receipt.instances[0],
        receipt,
    )
    for record in records:
        decoded = type(record).from_json(record.canonical_bytes())
        assert decoded == record
        assert decoded.canonical_bytes() == record.canonical_bytes()
        assert decoded.digest == record.digest


def test_evidence_is_generation_pinned_and_content_verified() -> None:
    value, evidence = _evidence()
    evidence.verify_bytes(value)
    with pytest.raises(ProtocolError, match="size"):
        evidence.verify_bytes(value + b"x")
    corrupt = bytearray(value)
    corrupt[0] ^= 1
    with pytest.raises(ProtocolError, match="SHA-256"):
        evidence.verify_bytes(bytes(corrupt))
    with pytest.raises(ProtocolError, match="must be bytes"):
        evidence.verify_bytes("not-bytes")  # type: ignore[arg-type]

    with pytest.raises(ProtocolError, match="generation"):
        replace(evidence, generation=True)
    with pytest.raises(ProtocolError, match="normalized"):
        replace(evidence, uri="gs://r1-control/evidence/../forged.json")
    with pytest.raises(ProtocolError, match="object URI"):
        replace(evidence, uri="gs://r1-control/evidence/file.json?generation=17")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("family_project", "ubuntu-os-cloud", "certified Ubuntu 22.04"),
        ("family_name", "ubuntu-2404-lts-amd64", "certified Ubuntu 22.04"),
        ("family_alias", "projects/family/latest", "canonical family alias"),
        ("resolved_image_name", "ubuntu-accel-latest", "dated family image"),
        ("resolved_image_status", "CREATING", "unused"),
    ],
)
def test_base_image_resolution_rejects_alias_drift(
    field: str, value: object, message: str
) -> None:
    resolution = _resolution()
    if field == "resolved_image_status":
        with pytest.raises(ProtocolError, match="resolved_status"):
            replace(resolution, resolved_status=value)
        return
    with pytest.raises(ProtocolError, match=message):
        replace(resolution, **{field: value})


def test_base_image_resolution_requires_ready_x86_tpu_image() -> None:
    resolution = _resolution()
    with pytest.raises(ProtocolError, match="architecture"):
        replace(resolution, architecture="ARM64")
    with pytest.raises(ProtocolError, match="TPU license"):
        replace(
            resolution,
            license_self_links=(
                "https://www.googleapis.com/compute/v1/projects/ubuntu-os-cloud/"
                "global/licenses/ubuntu-2204-lts",
            ),
        )
    with pytest.raises(ProtocolError, match="resolved_image_id"):
        replace(resolution, resolved_image_id=True)


def test_base_image_authority_is_not_self_asserted() -> None:
    resolution = _resolution()
    with pytest.raises(ProtocolError, match="challenge"):
        resolution.verify_authority(
            controller_service_account_email=CONTROLLER_EMAIL,
            controller_service_account_unique_id=CONTROLLER_ID,
            controller_challenge="d" * 64,
        )
    with pytest.raises(ProtocolError, match="unique_id"):
        resolution.verify_authority(
            controller_service_account_email=CONTROLLER_EMAIL,
            controller_service_account_unique_id="9" * 21,
            controller_challenge=CHALLENGE,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("accelerator_type", "v6e-32", "v6e-64"),
        ("machine_type", "ct6e-standard-8t", "ct6e-standard-4t"),
        ("topology", "4x8", "8x8"),
        ("vm_count", 8, "vm_count"),
        ("chips_per_vm", 8, "chips_per_vm"),
        ("chip_count", 32, "chip_count"),
        ("mig_scope", "REGIONAL", "ZONAL"),
        ("target_size", 0, "full 16-VM"),
        ("target_size_policy", "ANY", "BULK"),
        ("default_action_on_failure", "RECREATE", "DO_NOTHING"),
        ("replacement_policy", "REUSE_MIG", "FRESH_WHOLE_MIG"),
        ("provisioning_model", "STANDARD", "SPOT"),
        ("termination_action", "STOP", "DELETE"),
        ("maintenance_policy", "MIGRATE", "TERMINATE"),
    ],
)
def test_deployment_envelope_hard_codes_atomic_spot_slice(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ProtocolError, match=message):
        replace(_deployment(), **{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "vm_count",
        "chips_per_vm",
        "chip_count",
        "image_id",
        "instance_template_id",
        "resource_policy_id",
        "mig_id",
        "target_size",
    ],
)
def test_deployment_rejects_boolean_integer_fields(field: str) -> None:
    with pytest.raises(ProtocolError, match="integer|ID"):
        replace(_deployment(), **{field: True})


def test_deployment_rejects_identity_and_location_mismatches() -> None:
    deployment = _deployment()
    with pytest.raises(ProtocolError, match="zone must belong"):
        replace(deployment, zone="us-east5-a")
    with pytest.raises(ProtocolError, match="machine_type_self_link"):
        replace(
            deployment, machine_type_self_link=MACHINE_LINK.replace(ZONE, "us-east5-a")
        )
    with pytest.raises(ProtocolError, match="image selfLink"):
        replace(deployment, image_name="other-image")
    with pytest.raises(ProtocolError, match="instance_template_self_link"):
        replace(deployment, instance_template_name="other-template")
    with pytest.raises(ProtocolError, match="resource_policy_self_link"):
        replace(deployment, resource_policy_name="other-policy")
    with pytest.raises(ProtocolError, match="mig_self_link"):
        replace(deployment, mig_name="other-mig")
    with pytest.raises(ProtocolError, match="service_account_unique_id"):
        replace(deployment, service_account_unique_id=123)  # type: ignore[arg-type]


def test_deployment_binds_exact_base_and_published_image() -> None:
    deployment = _deployment()
    with pytest.raises(ProtocolError, match="resolution digest"):
        deployment.verify_base_image_resolution(
            replace(_resolution(), observed_at_unix_ms=1_753_220_000_001)
        )
    changed_publication = replace(_publication(), image_id=IMAGE_ID + 1)
    with pytest.raises(ProtocolError, match="image_id"):
        replace(
            deployment,
            image_publication_sha256=changed_publication.digest,
        ).verify_publication(changed_publication)

    changed_base = replace(_publication(), base_image_id=BASE_IMAGE_ID + 1)
    with pytest.raises(ProtocolError, match="base numeric ID"):
        replace(
            deployment,
            image_publication_sha256=changed_base.digest,
        ).verify_image_chain(_resolution(), changed_base)


def test_api_evidence_must_live_under_controller_root() -> None:
    resolution = _resolution()
    outside_resolution = replace(
        resolution,
        api_evidence=replace(
            resolution.api_evidence,
            uri="gs://other-control/evidence/base-image.json",
        ),
    )
    deployment = replace(
        _deployment(),
        base_image_resolution_sha256=outside_resolution.digest,
    )
    with pytest.raises(ProtocolError, match="outside.*controller root"):
        deployment.verify_base_image_resolution(outside_resolution)

    receipt = _receipt()
    outside_receipt = replace(
        receipt,
        api_evidence=replace(
            receipt.api_evidence,
            uri="gs://other-control/evidence/placement.json",
        ),
    )
    with pytest.raises(ProtocolError, match="outside.*controller root"):
        outside_receipt.verify_deployment(_deployment())


def test_observations_require_quiescent_full_slice() -> None:
    template = _template()
    with pytest.raises(ProtocolError, match="automatic_restart"):
        replace(template, automatic_restart=True)
    with pytest.raises(ProtocolError, match="boolean"):
        replace(template, automatic_restart=0)
    with pytest.raises(ProtocolError, match="termination_action"):
        replace(template, termination_action="STOP")
    with pytest.raises(ProtocolError, match="only moe-sae-r1-launch-ref"):
        replace(template, metadata_keys=("operator-command",))
    with pytest.raises(ProtocolError, match="startup scripts"):
        replace(
            template,
            metadata_keys=("moe-sae-r1-launch-ref", "startup-script"),
        )

    instance = _instances()[0]
    with pytest.raises(ProtocolError, match="RUNNING"):
        replace(instance, status="TERMINATED")
    with pytest.raises(ProtocolError, match="current_action"):
        replace(instance, current_action="RECREATING")

    mig = _mig()
    with pytest.raises(ProtocolError, match="is_stable"):
        replace(mig, is_stable=False)
    with pytest.raises(ProtocolError, match="boolean"):
        replace(mig, is_stable=1)
    with pytest.raises(ProtocolError, match="exactly 16"):
        replace(mig, member_instance_self_links=mig.member_instance_self_links[:-1])


def test_receipt_requires_exact_unique_membership() -> None:
    receipt = _receipt()
    with pytest.raises(ProtocolError, match="exactly 16"):
        replace(receipt, instances=receipt.instances[:-1])

    duplicate_id = replace(
        receipt.instances[-1], resource_id=receipt.instances[0].resource_id
    )
    with pytest.raises(ProtocolError, match="resource_id values must be unique"):
        replace(receipt, instances=receipt.instances[:-1] + (duplicate_id,))

    other = replace(
        receipt.instances[-1],
        name="unmanaged-instance",
        resource_id=7_999_999_999_999_999_999,
        self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
            f"zones/{ZONE}/instances/unmanaged-instance"
        ),
    )
    with pytest.raises(ProtocolError, match="membership"):
        replace(receipt, instances=receipt.instances[:-1] + (other,))


def test_receipt_rejects_cross_resource_join_mismatches() -> None:
    receipt = _receipt()
    wrong_template = replace(
        receipt.instance_template,
        name="other-template",
        resource_id=TEMPLATE_ID + 1,
        self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
            "global/instanceTemplates/other-template"
        ),
    )
    with pytest.raises(ProtocolError, match="MIG instance template"):
        replace(receipt, instance_template=wrong_template)

    wrong_policy = replace(
        receipt.resource_policy,
        name="other-policy",
        resource_id=POLICY_ID + 1,
        self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
            f"regions/{REGION}/resourcePolicies/other-policy"
        ),
    )
    with pytest.raises(ProtocolError, match="MIG resource policy"):
        replace(receipt, resource_policy=wrong_policy)

    wrong_instance = replace(
        receipt.instances[-1], service_account_email=CONTROLLER_EMAIL
    )
    wrong_members = list(receipt.mig.member_instance_self_links)
    changed = receipt.instances[-1].self_link
    assert changed in wrong_members
    with pytest.raises(ProtocolError, match="service_account_email"):
        replace(receipt, instances=receipt.instances[:-1] + (wrong_instance,))


def test_receipt_authority_requires_external_controller_identity_and_challenge() -> (
    None
):
    receipt = _receipt()
    with pytest.raises(ProtocolError, match="challenge"):
        receipt.verify_authority(
            controller_service_account_email=CONTROLLER_EMAIL,
            controller_service_account_unique_id=CONTROLLER_ID,
            controller_challenge="0" * 64,
        )
    with pytest.raises(ProtocolError, match="email"):
        receipt.verify_authority(
            controller_service_account_email="guest@neural-dragnet.iam.gserviceaccount.com",
            controller_service_account_unique_id=CONTROLLER_ID,
            controller_challenge=CHALLENGE,
        )


def test_receipt_verification_detects_every_exact_deployment_identity() -> None:
    receipt = _receipt()
    baseline = _deployment()
    valid_mutations = {
        "image_id": IMAGE_ID + 1,
        "instance_template_id": TEMPLATE_ID + 1,
        "resource_policy_id": POLICY_ID + 1,
        "mig_id": baseline.mig_id + 1,
        "service_account_unique_id": "9" * 21,
    }
    for field, value in valid_mutations.items():
        changed = replace(baseline, **{field: value})
        with pytest.raises(ProtocolError):
            receipt.verify_deployment(changed)

    substituted_launch_ref = replace(
        receipt.instance_template,
        launch_ref=replace(receipt.instance_template.launch_ref, generation=999),
    )
    with pytest.raises(ProtocolError, match="launch_ref"):
        replace(receipt, instance_template=substituted_launch_ref).verify_deployment(
            baseline
        )


def test_replacement_requires_fresh_whole_mig_and_instance_set() -> None:
    first_deployment = _deployment(1)
    second_deployment = _deployment(2)
    second_deployment.verify_fresh_replacement_of(first_deployment)

    with pytest.raises(ProtocolError, match="slice_attempt_id"):
        first_deployment.verify_fresh_replacement_of(first_deployment)
    with pytest.raises(ProtocolError, match="immutable field image_id"):
        replace(second_deployment, image_id=IMAGE_ID + 1).verify_fresh_replacement_of(
            first_deployment
        )

    first_receipt = _receipt(1)
    second_receipt = _receipt(2)
    second_receipt.verify_fresh_replacement_of(first_receipt)
    with pytest.raises(ProtocolError, match="slice_attempt_id"):
        first_receipt.verify_fresh_replacement_of(first_receipt)


def test_replacement_can_fail_over_across_regions_with_fresh_placement() -> None:
    previous = _deployment(1)
    successor = _deployment(2)
    east_region = "us-east5"
    east_zone = "us-east5-a"
    east_machine_link = (
        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
        f"zones/{east_zone}/machineTypes/ct6e-standard-4t"
    )
    east_policy_name = "r1-v6e-high-throughput-east"
    east_policy_id = POLICY_ID + 100
    east_policy_link = (
        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
        f"regions/{east_region}/resourcePolicies/{east_policy_name}"
    )
    east_mig_name = "r1-v6e-slice-east-a0002"
    east_mig_link = (
        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
        f"zones/{east_zone}/instanceGroupManagers/{east_mig_name}"
    )
    successor = replace(
        successor,
        region=east_region,
        zone=east_zone,
        machine_type_self_link=east_machine_link,
        resource_policy_name=east_policy_name,
        resource_policy_id=east_policy_id,
        resource_policy_self_link=east_policy_link,
        mig_name=east_mig_name,
        mig_id=successor.mig_id + 100,
        mig_self_link=east_mig_link,
    )
    successor.verify_fresh_replacement_of(previous)


def test_replacement_rejects_scientific_runtime_policy_drift() -> None:
    previous = _deployment(1)
    successor = _deployment(2)
    with pytest.raises(ProtocolError, match="immutable field image_id"):
        replace(successor, image_id=IMAGE_ID + 1).verify_fresh_replacement_of(previous)
    with pytest.raises(
        ProtocolError, match="immutable field service_account_unique_id"
    ):
        replace(
            successor,
            service_account_unique_id="9" * 21,
        ).verify_fresh_replacement_of(previous)


def test_replacement_requires_fresh_generation_pinned_launch_ref() -> None:
    previous = _deployment(1)
    successor = _deployment(2)
    with pytest.raises(ProtocolError, match="fresh launch_ref"):
        replace(successor, launch_ref=previous.launch_ref).verify_fresh_replacement_of(
            previous
        )


def test_replacement_launch_ref_requires_fresh_immutable_template() -> None:
    previous = _deployment(1)
    successor = _deployment(2)
    with pytest.raises(ProtocolError, match="fresh instance template name"):
        replace(
            successor,
            instance_template_name=previous.instance_template_name,
            instance_template_id=previous.instance_template_id,
            instance_template_self_link=previous.instance_template_self_link,
        ).verify_fresh_replacement_of(previous)


def test_wire_schema_rejects_unknown_fields_duplicate_keys_and_host_claims() -> None:
    receipt = _receipt()
    value = receipt.as_dict()
    value["host_attestation"] = {"claimed_ready": True}
    with pytest.raises(ProtocolError, match="unknown fields"):
        ControllerPlacementReceiptV2.from_dict(value)

    encoded = receipt.canonical_json()
    duplicate = encoded[:-1] + ',"observed_at_unix_ms":1}'
    with pytest.raises(ProtocolError, match="duplicate JSON field"):
        ControllerPlacementReceiptV2.from_json(duplicate)

    wrong_schema = receipt.as_dict()
    wrong_schema["schema"] = "moe-sae.r1.controller-placement-receipt.v1"
    with pytest.raises(ProtocolError, match="schema mismatch"):
        ControllerPlacementReceiptV2.from_dict(wrong_schema)


def test_nested_integer_fields_reject_booleans_during_decode() -> None:
    receipt = _receipt().as_dict()
    receipt["api_evidence"]["generation"] = True
    with pytest.raises(ProtocolError, match="booleans are forbidden"):
        ControllerPlacementReceiptV2.from_dict(receipt)

    receipt = _receipt().as_dict()
    receipt["instances"][0]["resource_id"] = False
    with pytest.raises(ProtocolError, match="booleans are forbidden"):
        ControllerPlacementReceiptV2.from_dict(receipt)


def test_collection_order_is_canonical_and_records_are_frozen() -> None:
    receipt = _receipt()
    reversed_receipt = replace(receipt, instances=tuple(reversed(receipt.instances)))
    assert reversed_receipt == receipt
    assert reversed_receipt.canonical_bytes() == receipt.canonical_bytes()

    resolution = _resolution()
    extra_license = (
        "https://www.googleapis.com/compute/v1/projects/ubuntu-os-cloud/"
        "global/licenses/ubuntu-2204-lts"
    )
    left = replace(resolution, license_self_links=(TPU_LICENSE, extra_license))
    right = replace(resolution, license_self_links=(extra_license, TPU_LICENSE))
    assert left == right
    assert left.canonical_bytes() == right.canonical_bytes()

    with pytest.raises(FrozenInstanceError):
        receipt.slice_attempt_id = "forged"  # type: ignore[misc]
