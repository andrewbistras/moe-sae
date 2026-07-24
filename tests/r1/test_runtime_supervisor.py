from __future__ import annotations

import hashlib
import signal
from typing import Any, Mapping

import pytest

from moe_sae.r1.control_store import MemoryJSONCASStore
from moe_sae.r1.controller import CampaignController
from moe_sae.r1.gcs_immutable_byte_store import (
    GCSPhysicalObjectReceiptV1,
    content_addressed_record_object_id,
)
from moe_sae.r1.placement_v2 import (
    ControllerPlacementReceiptV2,
    DeploymentEnvelopeV2,
    GCSObjectVersionRefV2,
    InstanceTemplateObservationV2,
    MIGObservationV2,
    ManagedInstanceObservationV2,
    ResourcePolicyObservationV2,
)
from moe_sae.r1.preemption_v2 import (
    AttemptAuthorityV2,
    CASPublishedCheckpointPointerV2,
    FullySealedCheckpointV2,
    ZeroNoticeStepLatchV2,
)
from moe_sae.r1.protocol import (
    AttemptReceiptV1,
    ContentRefV1,
    CostLedgerV1,
    HarvestRangeV1,
    RegionalReplicaV1,
)
from moe_sae.r1.runtime_supervisor import (
    GCEMetadataClientV1,
    GCSVersionPinnedByteReaderV1,
    LAUNCH_METADATA_KEY,
    LaunchAdmissionError,
    MetadataBoundaryError,
    MetadataHTTPResponse,
    RuntimeLaunchAdmissionV1,
    RuntimeLaunchResolverV1,
    RuntimeLaunchTokenV1,
    RuntimeStepContextV1,
    RuntimeSupervisorV1,
    publish_runtime_launch_admission,
    publish_runtime_launch_token,
)


PROJECT = "neural-dragnet"
REGION = "europe-west4"
ZONE = "europe-west4-a"
CAMPAIGN = "r1-headline"
BASE_URI = "gs://r1-control"
ROOT = f"{BASE_URI}/campaigns/{CAMPAIGN}/"
CONTROLLER_EMAIL = "r1-controller@neural-dragnet.iam.gserviceaccount.com"
CONTROLLER_ID = "223456789012345678901"
WORKER_EMAIL = "r1-worker@neural-dragnet.iam.gserviceaccount.com"
WORKER_ID = "123456789012345678901"
CHALLENGE = "c" * 64
IMPLEMENTATION = hashlib.sha256(b"implementation").hexdigest()
INVENTORY = "d" * 64
SCIENTIFIC = hashlib.sha256(b"scientific").hexdigest()
CORPUS = "f" * 64
LOGICAL_MANIFEST = "a" * 64
IMAGE_NAME = "r1-v6e-runtime-20260724"
IMAGE_ID = 2_012_345_678_901_234_567
IMAGE_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
    f"global/images/{IMAGE_NAME}"
)
MACHINE_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}/"
    "machineTypes/ct6e-standard-4t"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _version_ref(version: Any) -> GCSObjectVersionRefV2:
    return GCSObjectVersionRefV2(
        uri=version.uri,
        generation=version.generation,
        size_bytes=len(version.value),
        sha256=version.sha256,
    )


def _regional_envelope_receipt() -> GCSPhysicalObjectReceiptV1:
    digest = _sha("regional-prefix-envelope")
    object_id = content_addressed_record_object_id(digest)
    return GCSPhysicalObjectReceiptV1(
        store_id=f"gcs-prefix-byte-store-v2:{_sha('store')}",
        object_id=object_id,
        uri=f"gs://r1-europe-west4/prefix/{object_id}",
        region=REGION,
        generation="101",
        metageneration="1",
        size_bytes=4096,
        crc32c="AAAAAA==",
        sha256=digest,
        content_type="application/json",
    )


def _output() -> ContentRefV1:
    digest = _sha("final-output")
    replica = RegionalReplicaV1(
        region=REGION,
        uri=f"gs://r1-europe-west4/outputs/{digest}.json",
        generation=77,
        size_bytes=128,
        crc32c="AAAAAA==",
        sha256=digest,
    )
    return ContentRefV1(
        sha256=digest,
        size_bytes=128,
        media_type="application/json",
        replicas=(replica,),
    )


OUTPUT = _output()


class Clock:
    def __init__(self, value: int = 10_000) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


class SignalAfterPointerStore(MemoryJSONCASStore):
    """Inject SIGTERM after one checkpoint-head mutation has committed."""

    def __init__(self) -> None:
        super().__init__()
        self.pointer_signal_count = 0

    def _signal_after_pointer(self, uri: str) -> None:
        if "/checkpoints/" in uri and uri.endswith("/head.json"):
            if self.pointer_signal_count == 0:
                self.pointer_signal_count += 1
                signal.raise_signal(signal.SIGTERM)

    def create(self, uri: str, value: Any) -> Any:
        version = super().create(uri, value)
        self._signal_after_pointer(uri)
        return version

    def compare_and_swap(
        self,
        uri: str,
        expected_generation: int,
        value: Any,
    ) -> Any:
        version = super().compare_and_swap(uri, expected_generation, value)
        self._signal_after_pointer(uri)
        return version


class PinnedMemoryReader:
    def __init__(self, store: MemoryJSONCASStore) -> None:
        self.store = store

    def read_exact(
        self,
        reference: GCSObjectVersionRefV2,
        *,
        maximum_bytes: int,
    ) -> bytes:
        value = self.store.read(reference.uri)
        if value is None:
            raise LaunchAdmissionError("fake pinned object is absent")
        assert len(value.value) <= maximum_bytes
        assert value.generation == reference.generation
        reference.verify_bytes(value.value)
        return value.value


class MetadataTransport:
    def __init__(self, launch_ref: GCSObjectVersionRefV2) -> None:
        self.launch_ref = launch_ref
        self.preempted_body = b"FALSE"
        self.calls: list[str] = []

    def __call__(
        self,
        url: str,
        maximum_bytes: int,
        timeout_seconds: float,
    ) -> MetadataHTTPResponse:
        del maximum_bytes, timeout_seconds
        self.calls.append(url)
        body = (
            self.launch_ref.canonical_bytes()
            if url.endswith(f"/attributes/{LAUNCH_METADATA_KEY}")
            else self.preempted_body
        )
        return MetadataHTTPResponse(
            status=200,
            headers={"Metadata-Flavor": "Google"},
            body=body,
        )


def _token(
    *,
    attempt: int,
    fence: int,
    recovery: CASPublishedCheckpointPointerV2 | None = None,
) -> RuntimeLaunchTokenV1:
    return RuntimeLaunchTokenV1(
        controller_base_uri=BASE_URI,
        campaign_id=CAMPAIGN,
        work_id=_work().work_id,
        attempt_id=f"r1-attempt-{attempt:04d}",
        executor_id=f"r1-controller-{attempt:04d}",
        expected_fence=fence,
        implementation_identity_sha256=IMPLEMENTATION,
        controller_service_account_email=CONTROLLER_EMAIL,
        controller_service_account_unique_id=CONTROLLER_ID,
        controller_challenge=CHALLENGE,
        region=REGION,
        regional_envelope_receipt=_regional_envelope_receipt(),
        expected_logical_manifest_sha256=LOGICAL_MANIFEST,
        recovery_checkpoint=recovery,
        lease_duration_ms=1_000,
        renewal_margin_ms=400,
        control_poll_interval_ms=100,
    )


def _work() -> HarvestRangeV1:
    return HarvestRangeV1(
        scientific_identity_sha256=SCIENTIFIC,
        corpus_fingerprint_sha256=CORPUS,
        split="train",
        record_start=0,
        record_end=10,
        scientific_token_start=0,
        scientific_token_end=1024,
    )


def _deployment(
    launch_ref: GCSObjectVersionRefV2,
    *,
    attempt: int,
) -> DeploymentEnvelopeV2:
    template_name = f"r1-template-a{attempt:04d}"
    template_id = 4_012_345_678_901_234_000 + attempt
    template_link = (
        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/global/"
        f"instanceTemplates/{template_name}"
    )
    policy_name = "r1-v6e-high-throughput"
    policy_id = 5_012_345_678_901_234_567
    policy_link = (
        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/regions/"
        f"{REGION}/resourcePolicies/{policy_name}"
    )
    mig_name = f"r1-slice-a{attempt:04d}"
    mig_id = 6_012_345_678_901_234_000 + attempt
    mig_link = (
        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}/"
        f"instanceGroupManagers/{mig_name}"
    )
    return DeploymentEnvelopeV2(
        base_image_resolution_sha256=_sha("base-image-resolution"),
        image_publication_sha256=_sha("image-publication"),
        project=PROJECT,
        region=REGION,
        zone=ZONE,
        slice_attempt_id=f"r1-attempt-{attempt:04d}",
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
        resource_policy_name=policy_name,
        resource_policy_id=policy_id,
        resource_policy_self_link=policy_link,
        mig_name=mig_name,
        mig_id=mig_id,
        mig_self_link=mig_link,
        mig_scope="ZONAL",
        target_size=16,
        target_size_policy="BULK",
        default_action_on_failure="DO_NOTHING",
        replacement_policy="FRESH_WHOLE_MIG",
        provisioning_model="SPOT",
        termination_action="STOP",
        maintenance_policy="TERMINATE",
        service_account_email=WORKER_EMAIL,
        service_account_unique_id=WORKER_ID,
        controller_root_uri=ROOT,
        launch_ref=launch_ref,
    )


def _placement(
    deployment: DeploymentEnvelopeV2,
    api_ref: GCSObjectVersionRefV2,
) -> ControllerPlacementReceiptV2:
    template = InstanceTemplateObservationV2(
        project=PROJECT,
        zone=ZONE,
        name=deployment.instance_template_name,
        resource_id=deployment.instance_template_id,
        self_link=deployment.instance_template_self_link,
        machine_type=deployment.machine_type,
        machine_type_self_link=MACHINE_LINK,
        image_name=IMAGE_NAME,
        image_id=IMAGE_ID,
        image_self_link=IMAGE_LINK,
        provisioning_model="SPOT",
        termination_action="STOP",
        maintenance_policy="TERMINATE",
        automatic_restart=False,
        service_account_email=WORKER_EMAIL,
        metadata_keys=(LAUNCH_METADATA_KEY,),
        launch_ref=deployment.launch_ref,
    )
    policy = ResourcePolicyObservationV2(
        project=PROJECT,
        region=REGION,
        name=deployment.resource_policy_name,
        resource_id=deployment.resource_policy_id,
        self_link=deployment.resource_policy_self_link,
        policy_type="HIGH_THROUGHPUT",
        topology="8x8",
    )
    instances = tuple(
        ManagedInstanceObservationV2(
            project=PROJECT,
            zone=ZONE,
            name=f"{deployment.mig_name}-{index:04d}",
            resource_id=7_012_345_678_901_230_000 + index,
            self_link=(
                f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
                f"zones/{ZONE}/instances/{deployment.mig_name}-{index:04d}"
            ),
            mig_self_link=deployment.mig_self_link,
            instance_template_self_link=deployment.instance_template_self_link,
            machine_type_self_link=MACHINE_LINK,
            status="RUNNING",
            current_action="NONE",
            service_account_email=WORKER_EMAIL,
            boot_source_image_id=IMAGE_ID,
            boot_source_image_self_link=IMAGE_LINK,
        )
        for index in range(16)
    )
    mig = MIGObservationV2(
        project=PROJECT,
        zone=ZONE,
        name=deployment.mig_name,
        resource_id=deployment.mig_id,
        self_link=deployment.mig_self_link,
        instance_template_id=deployment.instance_template_id,
        instance_template_self_link=deployment.instance_template_self_link,
        resource_policy_id=deployment.resource_policy_id,
        resource_policy_self_link=deployment.resource_policy_self_link,
        target_size=16,
        target_size_policy="BULK",
        default_action_on_failure="DO_NOTHING",
        is_stable=True,
        member_instance_self_links=tuple(item.self_link for item in instances),
    )
    return ControllerPlacementReceiptV2(
        deployment_envelope_sha256=deployment.digest,
        base_image_resolution_sha256=deployment.base_image_resolution_sha256,
        image_publication_sha256=deployment.image_publication_sha256,
        slice_attempt_id=deployment.slice_attempt_id,
        controller_service_account_email=CONTROLLER_EMAIL,
        controller_service_account_unique_id=CONTROLLER_ID,
        controller_challenge=CHALLENGE,
        api_evidence=api_ref,
        worker_service_account_email=WORKER_EMAIL,
        worker_service_account_unique_id=WORKER_ID,
        instance_template=template,
        resource_policy=policy,
        mig=mig,
        instances=instances,
        observed_at_unix_ms=9_900,
    )


def _publish_launch(
    store: MemoryJSONCASStore,
    token: RuntimeLaunchTokenV1,
    *,
    attempt: int,
) -> tuple[GCSObjectVersionRefV2, MetadataTransport]:
    launch_ref = publish_runtime_launch_token(store, token)
    evidence = {
        "schema": "moe-sae.r1.test-placement-api-evidence.v1",
        "attempt_id": token.attempt_id,
    }
    evidence_version = store.put_immutable(
        f"{ROOT}evidence/{token.attempt_id}.json",
        evidence,
    )
    deployment = _deployment(launch_ref, attempt=attempt)
    admission = RuntimeLaunchAdmissionV1(
        launch_ref=launch_ref,
        deployment=deployment,
        placement=_placement(deployment, _version_ref(evidence_version)),
    )
    publish_runtime_launch_admission(store, token, admission)
    return launch_ref, MetadataTransport(launch_ref)


class FakeRegionalInputResolver:
    production_qualified = True

    def __init__(self) -> None:
        self.calls: list[tuple[GCSPhysicalObjectReceiptV1, str]] = []
        self.value = object()

    def open(
        self,
        receipt: GCSPhysicalObjectReceiptV1,
        *,
        expected_logical_manifest_sha256: str,
    ) -> object:
        self.calls.append((receipt, expected_logical_manifest_sha256))
        return self.value


class FakeWholeSlice:
    production_qualified = True

    def __init__(self) -> None:
        self.admissions: list[RuntimeLaunchAdmissionV1] = []
        self.terminations: list[tuple[str, str | None]] = []

    def initialize_device_inventory_sha256(
        self,
        admission: RuntimeLaunchAdmissionV1,
    ) -> str:
        self.admissions.append(admission)
        return INVENTORY

    def terminate(
        self,
        *,
        outcome: str,
        incident_sha256: str | None,
    ) -> None:
        self.terminations.append((outcome, incident_sha256))


class FakeWorkload:
    def __init__(
        self,
        store: MemoryJSONCASStore,
        clock: Clock,
        *,
        total_units: int = 3,
        signal_on_unit: int | None = None,
        advance_clock_on_unit: Mapping[int, int] | None = None,
    ) -> None:
        self.store = store
        self.clock = clock
        self._total_units = total_units
        self.signal_on_unit = signal_on_unit
        self.advance_clock_on_unit = dict(advance_clock_on_unit or {})
        self.restore_calls: list[
            tuple[object, FullySealedCheckpointV2 | None, bytes | None]
        ] = []
        self.run_calls: list[int] = []
        self.authorized_permits: list[int] = []

    @property
    def total_units(self) -> int:
        return self._total_units

    def restore(
        self,
        *,
        regional_input: object,
        recovery_checkpoint: FullySealedCheckpointV2 | None,
        recovery_manifest_bytes: bytes | None,
    ) -> None:
        self.restore_calls.append(
            (regional_input, recovery_checkpoint, recovery_manifest_bytes)
        )

    def run_unit(
        self,
        unit_index: int,
        context: RuntimeStepContextV1,
    ) -> FullySealedCheckpointV2:
        self.run_calls.append(unit_index)
        if unit_index in self.advance_clock_on_unit:
            self.clock.value = self.advance_clock_on_unit[unit_index]
        self.authorized_permits.append(context.authorize_step())
        authority = context.authority
        manifest = {
            "schema": "moe-sae.r1.test-training-checkpoint-manifest.v1",
            "attempt_id": authority.attempt_id,
            "completed_steps": unit_index + 1,
        }
        version = self.store.put_immutable(
            (
                f"{ROOT}test-manifests/{authority.attempt_id}/"
                f"{unit_index + 1}.json"
            ),
            manifest,
        )
        checkpoint = FullySealedCheckpointV2(
            campaign_id=authority.lease.campaign_id,
            work_id=authority.lease.work_id,
            source_attempt_id=authority.attempt_id,
            source_fence=authority.fence,
            checkpoint_id=(
                f"{authority.attempt_id}-checkpoint-{unit_index + 1:04d}"
            ),
            completed_steps=unit_index + 1,
            manifest=_version_ref(version),
            seal_receipt_sha256=_sha(
                f"{authority.attempt_id}-seal-{unit_index + 1}"
            ),
            sealed_at_unix_ms=self.clock(),
            seal_state="FULLY_SEALED",
        )
        if self.signal_on_unit == unit_index:
            signal.raise_signal(signal.SIGTERM)
        return checkpoint

    def successful_receipt(
        self,
        *,
        authority: AttemptAuthorityV2,
        final_checkpoint: CASPublishedCheckpointPointerV2,
        final_progress_sequence: int,
        finished_at_unix_ms: int,
    ) -> AttemptReceiptV1:
        assert final_checkpoint.checkpoint.completed_steps == self.total_units
        return AttemptReceiptV1(
            work_id=authority.lease.work_id,
            attempt_id=authority.attempt_id,
            fence=authority.fence,
            final_progress_sequence=final_progress_sequence,
            outcome="SUCCEEDED",
            outputs=(OUTPUT,),
            cost=CostLedgerV1(
                successful_chip_seconds=1.0,
                attempted_chip_seconds=1.0,
                compilation_chip_seconds=0.0,
                checkpoint_seconds=0.0,
                queue_seconds=0.0,
                egress_bytes=0,
                replayed_units=0,
            ),
            termination_reason=None,
            finished_at_unix_ms=finished_at_unix_ms,
        )


def _initialize_campaign(
    store: MemoryJSONCASStore,
    clock: Clock,
) -> CampaignController:
    controller = CampaignController(
        store,
        base_uri=BASE_URI,
        campaign_id=CAMPAIGN,
        clock_ms=clock,
        content_verifier=lambda content: content == OUTPUT,
    )
    controller.initialize(
        (_work(),),
        expected_outputs={_work().work_id: (OUTPUT,)},
    )
    return controller


def _supervisor(
    store: MemoryJSONCASStore,
    clock: Clock,
    token: RuntimeLaunchTokenV1,
    *,
    attempt: int,
    workload: FakeWorkload,
) -> tuple[RuntimeSupervisorV1, MetadataTransport, FakeWholeSlice]:
    _launch_ref, transport = _publish_launch(store, token, attempt=attempt)
    metadata = GCEMetadataClientV1(transport=transport)
    reader = PinnedMemoryReader(store)
    resolver = RuntimeLaunchResolverV1(
        metadata=metadata,
        pinned_reader=reader,
        control_store=store,
        allowed_controller_base_uri=BASE_URI,
        trusted_controller_service_account_email=CONTROLLER_EMAIL,
        trusted_controller_service_account_unique_id=CONTROLLER_ID,
    )
    slice_runtime = FakeWholeSlice()
    supervisor = RuntimeSupervisorV1(
        resolver=resolver,
        control_store=store,
        pinned_reader=reader,
        regional_input_resolver=FakeRegionalInputResolver(),
        slice_runtime=slice_runtime,
        workload=workload,
        content_verifier=lambda content: content == OUTPUT,
        clock_ms=clock,
        monotonic_ms=lambda: 0,
    )
    return supervisor, transport, slice_runtime


def test_metadata_launch_reference_is_exact_fixed_and_non_executable() -> None:
    raw = b'{"schema":"moe-sae.r1.test-launch.v1","value":"fixed"}'
    reference = GCSObjectVersionRefV2(
        uri=f"{ROOT}launch/r1-attempt-0001.json",
        generation=17,
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    transport = MetadataTransport(reference)
    client = GCEMetadataClientV1(transport=transport)

    assert client.read_launch_reference(LAUNCH_METADATA_KEY) == reference
    assert transport.calls == [
        (
            "http://metadata.google.internal/computeMetadata/v1/"
            f"instance/attributes/{LAUNCH_METADATA_KEY}"
        )
    ]
    with pytest.raises(MetadataBoundaryError, match="not fixed"):
        client.read_launch_reference("startup-script")

    def injected(
        url: str,
        maximum_bytes: int,
        timeout_seconds: float,
    ) -> MetadataHTTPResponse:
        del url, maximum_bytes, timeout_seconds
        return MetadataHTTPResponse(
            200,
            {"Metadata-Flavor": "Google"},
            b"$(touch /tmp/metadata-was-executed)",
        )

    with pytest.raises(MetadataBoundaryError, match="canonical"):
        GCEMetadataClientV1(transport=injected).read_launch_reference(
            LAUNCH_METADATA_KEY
        )


def test_generation_pinned_gcs_reader_uses_exact_get_without_listing() -> None:
    payload = b'{"schema":"moe-sae.r1.test-pinned-read.v1"}'
    reference = GCSObjectVersionRefV2(
        uri="gs://r1-control/exact/object.json",
        generation=91,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    class Blob:
        generation: int | None = None
        size: int | None = None

        def __init__(self) -> None:
            self.reload_calls: list[dict[str, Any]] = []
            self.download_calls: list[dict[str, Any]] = []

        def reload(self, **kwargs: Any) -> None:
            self.reload_calls.append(kwargs)
            self.generation = reference.generation
            self.size = reference.size_bytes

        def download_as_bytes(self, **kwargs: Any) -> bytes:
            self.download_calls.append(kwargs)
            return payload

    blob = Blob()

    class Bucket:
        def blob(self, name: str, *, generation: int) -> Blob:
            assert name == "exact/object.json"
            assert generation == reference.generation
            return blob

    class Client:
        def bucket(self, name: str) -> Bucket:
            assert name == "r1-control"
            return Bucket()

    reader = GCSVersionPinnedByteReaderV1(Client())
    assert reader.read_exact(reference, maximum_bytes=1024) == payload
    assert blob.reload_calls == [
        {"if_generation_match": 91, "timeout": 60.0}
    ]
    assert blob.download_calls == [
        {
            "if_generation_match": 91,
            "checksum": "crc32c",
            "timeout": 60.0,
        }
    ]
    assert not hasattr(Client(), "list_blobs")


@pytest.mark.parametrize(
    "response",
    [
        MetadataHTTPResponse(503, {"Metadata-Flavor": "Google"}, b"FALSE"),
        MetadataHTTPResponse(200, {"Metadata-Flavor": "google"}, b"FALSE"),
        MetadataHTTPResponse(200, {"Metadata-Flavor": "Google"}, b"FALSE\n"),
    ],
)
def test_preemption_metadata_ambiguity_fails_closed(
    response: MetadataHTTPResponse,
) -> None:
    client = GCEMetadataClientV1(
        transport=lambda _url, _maximum, _timeout: response
    )
    with pytest.raises(MetadataBoundaryError):
        client.preempted()


def test_resolver_requires_existing_token_derived_admission() -> None:
    store = MemoryJSONCASStore()
    token = _token(attempt=1, fence=1)
    launch_ref = publish_runtime_launch_token(store, token)
    metadata = GCEMetadataClientV1(transport=MetadataTransport(launch_ref))
    resolver = RuntimeLaunchResolverV1(
        metadata=metadata,
        pinned_reader=PinnedMemoryReader(store),
        control_store=store,
        allowed_controller_base_uri=BASE_URI,
        trusted_controller_service_account_email=CONTROLLER_EMAIL,
        trusted_controller_service_account_unique_id=CONTROLLER_ID,
    )

    with pytest.raises(LaunchAdmissionError, match="absent"):
        resolver.resolve()


def test_lease_renews_before_expiry_at_an_accelerator_boundary() -> None:
    store = MemoryJSONCASStore()
    clock = Clock()
    _initialize_campaign(store, clock)
    workload = FakeWorkload(
        store,
        clock,
        total_units=1,
        advance_clock_on_unit={0: 10_650},
    )
    supervisor, transport, slice_runtime = _supervisor(
        store,
        clock,
        _token(attempt=1, fence=1),
        attempt=1,
        workload=workload,
    )

    result = supervisor.run()

    assert result.outcome == "SUCCEEDED"
    assert result.authority.lease.expires_at_unix_ms == 11_650
    assert workload.authorized_permits
    assert any(url.endswith("/preempted") for url in transport.calls)
    assert slice_runtime.terminations == [("SUCCEEDED", None)]


def test_local_sigterm_preemption_resumes_only_from_published_pointer() -> None:
    store = MemoryJSONCASStore()
    clock = Clock()
    campaign = _initialize_campaign(store, clock)

    first_workload = FakeWorkload(
        store,
        clock,
        total_units=3,
        signal_on_unit=1,
    )
    first, _transport, first_slice = _supervisor(
        store,
        clock,
        _token(attempt=1, fence=1),
        attempt=1,
        workload=first_workload,
    )
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    first_result = first.run()

    assert first_result.outcome == "PREEMPTED"
    assert first_result.completed_units == 1
    assert first_result.final_checkpoint is not None
    assert first_result.final_checkpoint.checkpoint.completed_steps == 1
    assert first_result.incident is not None
    assert first_result.incident.trigger_kind == "VM_TERMINATED"
    assert first_result.incident.authorized_steps_before_latch == 2
    assert (
        first_result.incident.recovery_checkpoint
        == first_result.final_checkpoint
    )
    assert first_workload.run_calls == [0, 1]
    incident_version = store.read(_token(attempt=1, fence=1).incident_uri())
    assert incident_version is not None
    assert incident_version.sha256 == first_result.incident.digest
    assert first_slice.terminations == [
        ("PREEMPTED", first_result.incident.digest)
    ]
    assert signal.getsignal(signal.SIGTERM) is previous_sigterm

    clock.value = 12_000
    expired = campaign.expire_and_requeue(_work().work_id)
    assert expired.status == "PENDING"
    assert expired.fence == 1

    second_workload = FakeWorkload(store, clock, total_units=3)
    second, _transport, second_slice = _supervisor(
        store,
        clock,
        _token(
            attempt=2,
            fence=2,
            recovery=first_result.final_checkpoint,
        ),
        attempt=2,
        workload=second_workload,
    )
    second_result = second.run()

    assert second_result.outcome == "SUCCEEDED"
    assert second_result.completed_units == 3
    assert second_workload.run_calls == [1, 2]
    assert len(second_workload.restore_calls) == 1
    restored = second_workload.restore_calls[0]
    assert restored[1] == first_result.final_checkpoint.checkpoint
    assert restored[2] is not None
    assert second_result.final_checkpoint is not None
    assert second_result.final_checkpoint.checkpoint.completed_steps == 3
    view = campaign.replay().work(_work().work_id)
    assert view.status == "COMMITTED"
    assert view.fence == 2
    assert view.completed_units == 3
    assert second_slice.terminations == [("SUCCEEDED", None)]


def test_sigterm_after_pointer_cas_reconciles_progress_on_fresh_fence() -> None:
    store = SignalAfterPointerStore()
    clock = Clock()
    campaign = _initialize_campaign(store, clock)

    first_workload = FakeWorkload(store, clock, total_units=1)
    first, _transport, first_slice = _supervisor(
        store,
        clock,
        _token(attempt=1, fence=1),
        attempt=1,
        workload=first_workload,
    )
    first_result = first.run()

    assert first_result.outcome == "PREEMPTED"
    assert first_result.completed_units == 1
    assert first_result.final_checkpoint is not None
    assert first_result.final_checkpoint.checkpoint.completed_steps == 1
    assert first_result.incident is not None
    assert first_result.incident.recovery_checkpoint == first_result.final_checkpoint
    assert first_result.incident.authorized_steps_before_latch == 1
    assert first_slice.terminations == [
        ("PREEMPTED", first_result.incident.digest)
    ]
    before_recovery = campaign.replay().work(_work().work_id)
    assert before_recovery.completed_units is None

    clock.value = 12_000
    campaign.expire_and_requeue(_work().work_id)
    second_workload = FakeWorkload(store, clock, total_units=1)
    second, _transport, second_slice = _supervisor(
        store,
        clock,
        _token(
            attempt=2,
            fence=2,
            recovery=first_result.final_checkpoint,
        ),
        attempt=2,
        workload=second_workload,
    )
    second_result = second.run()

    assert second_result.outcome == "SUCCEEDED"
    assert second_result.completed_units == 1
    assert second_workload.run_calls == []
    committed = campaign.replay().work(_work().work_id)
    assert committed.status == "COMMITTED"
    assert committed.completed_units == 1
    assert committed.last_progress_sequence == 1
    assert second_slice.terminations == [("SUCCEEDED", None)]


def test_sigterm_racing_local_permit_cannot_reach_accelerator_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryJSONCASStore()
    clock = Clock()
    _initialize_campaign(store, clock)
    workload = FakeWorkload(store, clock, total_units=1)
    supervisor, _transport, slice_runtime = _supervisor(
        store,
        clock,
        _token(attempt=1, fence=1),
        attempt=1,
        workload=workload,
    )
    original = ZeroNoticeStepLatchV2.authorize_new_step

    def signal_after_permit(
        latch: ZeroNoticeStepLatchV2,
        authority: AttemptAuthorityV2,
    ) -> int:
        permit = original(latch, authority)
        signal.raise_signal(signal.SIGTERM)
        return permit

    monkeypatch.setattr(
        ZeroNoticeStepLatchV2,
        "authorize_new_step",
        signal_after_permit,
    )

    result = supervisor.run()

    assert result.outcome == "PREEMPTED"
    assert result.completed_units == 0
    assert result.incident is not None
    assert result.incident.trigger_kind == "VM_TERMINATED"
    assert result.incident.authorized_steps_before_latch == 1
    assert workload.run_calls == [0]
    assert workload.authorized_permits == []
    assert slice_runtime.terminations == [
        ("PREEMPTED", result.incident.digest)
    ]


def test_ambiguous_live_preemption_poll_publishes_bounded_v2_incident() -> None:
    store = MemoryJSONCASStore()
    clock = Clock()
    _initialize_campaign(store, clock)
    workload = FakeWorkload(store, clock, total_units=1)
    supervisor, transport, slice_runtime = _supervisor(
        store,
        clock,
        _token(attempt=1, fence=1),
        attempt=1,
        workload=workload,
    )
    transport.preempted_body = b"UNKNOWN"

    result = supervisor.run()

    assert result.outcome == "PREEMPTED"
    assert result.completed_units == 0
    assert result.final_checkpoint is None
    assert result.incident is not None
    assert result.incident.trigger_kind == "METADATA_AMBIGUOUS"
    assert workload.run_calls == []
    assert slice_runtime.terminations == [
        ("PREEMPTED", result.incident.digest)
    ]
