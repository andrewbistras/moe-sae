from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import hashlib
import inspect
import json
import threading

import pytest

from moe_sae.r1.placement_v2 import DeploymentEnvelopeV2, GCSObjectVersionRefV2
from moe_sae.r1.preemption_v2 import (
    AttemptAuthorityV2,
    AttemptBindingMismatchV2,
    CASPublishedCheckpointPointerV2,
    FullySealedCheckpointV2,
    IncidentPublicationErrorV2,
    NewStepsForbiddenV2,
    TerminalIncidentImmutableV2,
    UnpublishedCheckpointV2,
    ZeroNoticePreemptionError,
    ZeroNoticeRecoveryPlanV2,
    ZeroNoticeSliceIncidentV2,
    ZeroNoticeStepLatchV2,
)
from moe_sae.r1.protocol import AttemptLeaseV1, DeploymentIdentityV1, ProtocolError


PROJECT = "neural-dragnet"
REGION = "us-central1"
ZONE = "us-central1-b"
WORKER_EMAIL = "r1-worker@neural-dragnet.iam.gserviceaccount.com"
WORKER_ID = "123456789012345678901"
IMAGE_NAME = "r1-v6e-runtime-20260723"
IMAGE_ID = 2_012_345_678_901_234_567
IMAGE_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
    f"global/images/{IMAGE_NAME}"
)
POLICY_NAME = "r1-v6e-high-throughput"
POLICY_ID = 5_012_345_678_901_234_567
POLICY_LINK = (
    f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
    f"regions/{REGION}/resourcePolicies/{POLICY_NAME}"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _object_ref(path: str, generation: int, payload: bytes) -> GCSObjectVersionRefV2:
    return GCSObjectVersionRefV2(
        uri=f"gs://r1-control/campaigns/headline-01/{path}",
        generation=generation,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _deployment(attempt: int = 1) -> DeploymentEnvelopeV2:
    attempt_id = f"attempt-{attempt}"
    template_name = f"r1-v6e-template-a{attempt:04d}"
    template_id = 4_012_345_678_901_234_000 + attempt
    template_link = (
        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
        f"global/instanceTemplates/{template_name}"
    )
    mig_name = f"r1-v6e-slice-a{attempt:04d}"
    mig_id = 6_012_345_678_901_234_000 + attempt
    mig_link = (
        f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
        f"zones/{ZONE}/instanceGroupManagers/{mig_name}"
    )
    launch_payload = json.dumps(
        {"attempt_id": attempt_id}, sort_keys=True, separators=(",", ":")
    ).encode()
    return DeploymentEnvelopeV2(
        base_image_resolution_sha256=_sha("base-image-resolution"),
        image_publication_sha256=_sha("image-publication"),
        project=PROJECT,
        region=REGION,
        zone=ZONE,
        slice_attempt_id=attempt_id,
        accelerator_type="v6e-64",
        machine_type="ct6e-standard-4t",
        machine_type_self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/"
            f"zones/{ZONE}/machineTypes/ct6e-standard-4t"
        ),
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
        termination_action="STOP",
        maintenance_policy="TERMINATE",
        service_account_email=WORKER_EMAIL,
        service_account_unique_id=WORKER_ID,
        controller_root_uri="gs://r1-control/campaigns/headline-01/",
        launch_ref=_object_ref(
            f"launch/{attempt_id}.json", 100 + attempt, launch_payload
        ),
    )


def _authority_for_deployment(
    deployment: DeploymentEnvelopeV2,
    *,
    fence: int,
    acquired: int,
    expires: int,
) -> AttemptAuthorityV2:
    attempt_id = deployment.slice_attempt_id
    identity = DeploymentIdentityV1(
        implementation_identity_sha256=_sha("implementation"),
        deployment_envelope_sha256=deployment.digest,
        device_inventory_sha256=_sha(f"inventory-{attempt_id}"),
        attempt_id=attempt_id,
    )
    lease = AttemptLeaseV1(
        campaign_id="r1-study",
        work_id="sae-work-1",
        attempt_id=attempt_id,
        fence=fence,
        executor_id=f"executor-{attempt_id}",
        deployment_identity_sha256=identity.digest,
        acquired_at_unix_ms=acquired,
        expires_at_unix_ms=expires,
    )
    return AttemptAuthorityV2(
        lease=lease,
        deployment_identity=identity,
        deployment=deployment,
    )


def _authority(
    attempt: int = 1,
    *,
    fence: int | None = None,
    acquired: int | None = None,
    expires: int | None = None,
) -> AttemptAuthorityV2:
    fence = attempt if fence is None else fence
    acquired = (1_000 if attempt == 1 else 2_000) if acquired is None else acquired
    expires = (2_000 if attempt == 1 else 3_000) if expires is None else expires
    return _authority_for_deployment(
        _deployment(attempt),
        fence=fence,
        acquired=acquired,
        expires=expires,
    )


def _sealed(
    *,
    source_attempt_id: str = "attempt-1",
    source_fence: int = 1,
) -> FullySealedCheckpointV2:
    manifest_payload = b'{"orbax":"complete","step":100}'
    return FullySealedCheckpointV2(
        campaign_id="r1-study",
        work_id="sae-work-1",
        source_attempt_id=source_attempt_id,
        source_fence=source_fence,
        checkpoint_id="checkpoint-000100",
        completed_steps=100,
        manifest=_object_ref(
            "checkpoints/checkpoint-000100/manifest.json",
            501,
            manifest_payload,
        ),
        seal_receipt_sha256=_sha("seal-receipt"),
        sealed_at_unix_ms=1_500,
        seal_state="FULLY_SEALED",
    )


def _pointer(
    checkpoint: FullySealedCheckpointV2 | None = None,
) -> CASPublishedCheckpointPointerV2:
    checkpoint = _sealed() if checkpoint is None else checkpoint
    payload = checkpoint.canonical_bytes()
    return CASPublishedCheckpointPointerV2(
        checkpoint=checkpoint,
        pointer_object=_object_ref("checkpoints/latest.json", 700, payload),
        cas_expected_generation=699,
        published_at_unix_ms=1_600,
        publication_state="CAS_PUBLISHED",
    )


def _unpublished(state: str = "IN_FLIGHT") -> UnpublishedCheckpointV2:
    return UnpublishedCheckpointV2(
        campaign_id="r1-study",
        work_id="sae-work-1",
        attempt_id="attempt-1",
        fence=1,
        checkpoint_id="checkpoint-000101",
        state=state,
        diagnostic_sha256=_sha(f"diagnostic-{state}"),
    )


def _incident(
    *,
    trigger: str = "METADATA_TRUE",
    pointer: CASPublishedCheckpointPointerV2 | None = None,
    ignored: tuple[UnpublishedCheckpointV2, ...] = (),
) -> ZeroNoticeSliceIncidentV2:
    authority = _authority()
    latch = ZeroNoticeStepLatchV2(authority, incident_sink=lambda _: None)
    return latch.latch(
        authority,
        trigger_kind=trigger,
        recovery_checkpoint=pointer,
        ignored_unpublished_checkpoints=ignored,
    )


def test_authority_pointer_incident_and_plan_are_canonical() -> None:
    authority = _authority()
    pointer = _pointer()
    incident = _incident(pointer=pointer, ignored=(_unpublished(),))
    plan = ZeroNoticeRecoveryPlanV2.build(incident, _authority(2))

    pointer.verify_bytes(pointer.checkpoint.canonical_bytes())
    for record in (
        authority,
        pointer.checkpoint,
        pointer,
        _unpublished(),
        incident,
        plan,
    ):
        decoded = type(record).from_json(record.canonical_bytes())
        assert decoded == record
        assert decoded.canonical_bytes() == record.canonical_bytes()
        assert decoded.digest == record.digest


def test_attempt_authority_binds_lease_identity_and_v2_envelope() -> None:
    authority = _authority()
    assert (
        authority.lease.deployment_identity_sha256
        == authority.deployment_identity.digest
    )
    assert (
        authority.deployment_identity.deployment_envelope_sha256
        == authority.deployment.digest
    )

    with pytest.raises(ProtocolError, match="lease does not bind"):
        replace(
            authority,
            lease=replace(
                authority.lease,
                deployment_identity_sha256=_sha("other-deployment-identity"),
            ),
        )
    changed_identity = replace(
        authority.deployment_identity,
        deployment_envelope_sha256=_sha("other-envelope"),
    )
    with pytest.raises(ProtocolError, match="V2 envelope"):
        replace(
            authority,
            lease=replace(
                authority.lease,
                deployment_identity_sha256=changed_identity.digest,
            ),
            deployment_identity=changed_identity,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("guaranteed_notice_ms", 1, "must be zero"),
        ("termination_action", "DELETE", "STOP"),
        ("loss_scope", "ONE_VM", "WHOLE_SLICE"),
        ("lost_vm_count", 1, "16 TPU VMs"),
        ("lost_chip_count", 4, "64 TPU chips"),
        ("trigger_authorizes_checkpoint", True, "cannot authorize"),
        ("emergency_checkpoint_authorized", True, "forbidden"),
        ("new_steps_forbidden", False, "must be true"),
        ("replacement_required", False, "must be true"),
    ],
)
def test_zero_notice_incident_has_no_emergency_save_semantics(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ProtocolError, match=message):
        replace(_incident(pointer=_pointer()), **{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "guaranteed_notice_ms",
        "lost_vm_count",
        "lost_chip_count",
        "authorized_steps_before_latch",
    ],
)
def test_incident_rejects_boolean_integer_fields(field: str) -> None:
    with pytest.raises(ProtocolError, match="booleans are forbidden"):
        replace(_incident(), **{field: True})


@pytest.mark.parametrize(
    "field",
    [
        "trigger_authorizes_checkpoint",
        "emergency_checkpoint_authorized",
        "new_steps_forbidden",
        "replacement_required",
    ],
)
def test_incident_rejects_integer_boolean_fields(field: str) -> None:
    with pytest.raises(ProtocolError, match="boolean"):
        replace(_incident(), **{field: 1})


def test_every_terminal_trigger_only_preserves_prepublished_pointer() -> None:
    pointer = _pointer()
    for trigger in (
        "METADATA_TRUE",
        "METADATA_AMBIGUOUS",
        "LEASE_EXPIRED",
        "SLICE_MEMBER_LOST",
        "VM_TERMINATED",
    ):
        incident = _incident(trigger=trigger, pointer=pointer)
        assert incident.recovery_checkpoint == pointer
        assert incident.guaranteed_notice_ms == 0
        assert not incident.trigger_authorizes_checkpoint
        assert not incident.emergency_checkpoint_authorized
        assert incident.new_steps_forbidden


def test_unsealed_or_unpublished_checkpoint_cannot_be_recovery_authority() -> None:
    authority = _authority()
    latch = ZeroNoticeStepLatchV2(authority, incident_sink=lambda _: None)
    with pytest.raises(ProtocolError, match="CASPublishedCheckpointPointerV2"):
        latch.latch(
            authority,
            trigger_kind="METADATA_TRUE",
            recovery_checkpoint=_unpublished(),  # type: ignore[arg-type]
        )
    with pytest.raises(ProtocolError, match="CASPublishedCheckpointPointerV2"):
        latch.latch(
            authority,
            trigger_kind="METADATA_TRUE",
            recovery_checkpoint=_sealed(),  # type: ignore[arg-type]
        )

    with pytest.raises(ProtocolError, match="FULLY_SEALED"):
        replace(_sealed(), seal_state="PARTIAL")
    with pytest.raises(ProtocolError, match="CAS_PUBLISHED"):
        replace(_pointer(), publication_state="IN_FLIGHT")


def test_partial_and_inflight_checkpoints_are_recorded_but_ignored() -> None:
    pointer = _pointer()
    ignored = (
        _unpublished("PARTIAL"),
        replace(
            _unpublished("IN_FLIGHT"),
            checkpoint_id="checkpoint-000102",
            diagnostic_sha256=_sha("other-in-flight"),
        ),
        _unpublished("FULLY_SEALED_NOT_CAS_PUBLISHED"),
    )
    incident = _incident(pointer=pointer, ignored=tuple(reversed(ignored)))
    assert incident.recovery_checkpoint == pointer
    assert {item.state for item in incident.ignored_unpublished_checkpoints} == {
        "PARTIAL",
        "IN_FLIGHT",
        "FULLY_SEALED_NOT_CAS_PUBLISHED",
    }
    plan = ZeroNoticeRecoveryPlanV2.build(incident, _authority(2))
    assert plan.restore_pointer_sha256 == pointer.digest
    assert plan.restore_action == "RESTORE_LAST_CAS_PUBLISHED"

    no_pointer_plan = ZeroNoticeRecoveryPlanV2.build(_incident(), _authority(2))
    assert no_pointer_plan.restore_pointer_sha256 is None
    assert no_pointer_plan.restore_action == "RESTART_FROM_INITIAL_STATE"


def test_pointer_binds_exact_canonical_payload_and_cas_generation() -> None:
    pointer = _pointer()
    with pytest.raises(ProtocolError, match="size"):
        replace(
            pointer,
            pointer_object=replace(
                pointer.pointer_object,
                size_bytes=pointer.pointer_object.size_bytes + 1,
            ),
        )
    with pytest.raises(ProtocolError, match="hash"):
        replace(
            pointer,
            pointer_object=replace(pointer.pointer_object, sha256=_sha("forged")),
        )
    with pytest.raises(ProtocolError, match="booleans are forbidden"):
        replace(pointer, cas_expected_generation=True)
    with pytest.raises(ProtocolError, match="must precede"):
        replace(pointer, cas_expected_generation=pointer.pointer_object.generation)
    with pytest.raises(ProtocolError, match="size|SHA-256|canonical"):
        pointer.verify_bytes(pointer.checkpoint.canonical_bytes() + b" ")


def test_latch_constructor_has_no_clock_or_checkpoint_writer_path() -> None:
    parameters = inspect.signature(ZeroNoticeStepLatchV2.__init__).parameters
    assert set(parameters) == {"self", "authority", "incident_sink"}
    assert not any(
        fragment in name
        for name in parameters
        for fragment in ("clock", "save", "checkpoint", "emergency", "metadata")
    )
    assert not any("emergency" in name.lower() for name in dir(ZeroNoticeStepLatchV2))


def test_stale_attempt_or_fence_cannot_step_or_latch() -> None:
    authority = _authority()
    stale = _authority(2)
    latch = ZeroNoticeStepLatchV2(authority, incident_sink=lambda _: None)
    with pytest.raises(AttemptBindingMismatchV2):
        latch.authorize_new_step(stale)
    with pytest.raises(AttemptBindingMismatchV2):
        latch.latch(
            stale,
            trigger_kind="METADATA_TRUE",
            recovery_checkpoint=None,
        )
    assert latch.new_steps_allowed
    assert latch.authorized_step_count == 0


def test_monotonic_lease_renewal_updates_terminal_expiry_but_stale_lease_fails() -> (
    None
):
    authority = _authority()
    renewed = replace(
        authority,
        lease=replace(authority.lease, expires_at_unix_ms=2_500),
    )
    latch = ZeroNoticeStepLatchV2(authority, incident_sink=lambda _: None)
    assert latch.authorize_new_step(renewed) == 1
    assert latch.authority == renewed
    with pytest.raises(AttemptBindingMismatchV2):
        latch.authorize_new_step(authority)
    incident = latch.latch(
        renewed,
        trigger_kind="LEASE_EXPIRED",
        recovery_checkpoint=_pointer(),
    )
    assert incident.authority.lease.expires_at_unix_ms == 2_500


def test_post_latch_renewal_cannot_mutate_authority_through_either_method() -> None:
    authority = _authority()
    renewed = replace(
        authority,
        lease=replace(authority.lease, expires_at_unix_ms=2_500),
    )
    published: list[ZeroNoticeSliceIncidentV2] = []
    latch = ZeroNoticeStepLatchV2(authority, incident_sink=published.append)
    incident = latch.latch(
        authority,
        trigger_kind="METADATA_TRUE",
        recovery_checkpoint=_pointer(),
    )
    incident_bytes = incident.canonical_bytes()
    authority_bytes = latch.authority.canonical_bytes()

    with pytest.raises(AttemptBindingMismatchV2, match="immutable"):
        latch.authorize_new_step(renewed)
    with pytest.raises(AttemptBindingMismatchV2, match="immutable"):
        latch.latch(
            renewed,
            trigger_kind="METADATA_TRUE",
            recovery_checkpoint=_pointer(),
        )

    assert latch.authority.canonical_bytes() == authority_bytes
    assert latch.incident is incident
    assert latch.incident.canonical_bytes() == incident_bytes
    assert latch.authorized_step_count == 0
    assert published == [incident]
    with pytest.raises(NewStepsForbiddenV2):
        latch.authorize_new_step(authority)
    assert (
        latch.latch(
            authority,
            trigger_kind="METADATA_TRUE",
            recovery_checkpoint=_pointer(),
        )
        is incident
    )


def test_concurrent_renewal_vs_old_authority_latch_never_splits_terminal_binding() -> (
    None
):
    for _ in range(64):
        authority = _authority()
        renewed = replace(
            authority,
            lease=replace(authority.lease, expires_at_unix_ms=2_500),
        )
        latch = ZeroNoticeStepLatchV2(authority, incident_sink=lambda _: None)
        barrier = threading.Barrier(2)

        def renew_and_step() -> str:
            barrier.wait()
            try:
                latch.authorize_new_step(renewed)
            except AttemptBindingMismatchV2:
                return "renewal-rejected"
            return "renewal-won"

        def latch_old() -> str:
            barrier.wait()
            try:
                latch.latch(
                    authority,
                    trigger_kind="SLICE_MEMBER_LOST",
                    recovery_checkpoint=_pointer(),
                )
            except AttemptBindingMismatchV2:
                return "old-latch-rejected"
            return "old-latch-won"

        with ThreadPoolExecutor(max_workers=2) as pool:
            renewal_result = pool.submit(renew_and_step)
            latch_result = pool.submit(latch_old)
            outcomes = {renewal_result.result(), latch_result.result()}

        if latch.incident is not None:
            assert outcomes == {"old-latch-won", "renewal-rejected"}
            assert latch.authority.same_binding(authority)
        else:
            assert outcomes == {"renewal-won", "old-latch-rejected"}
            assert latch.authority.same_binding(renewed)
            latch.latch(
                renewed,
                trigger_kind="SLICE_MEMBER_LOST",
                recovery_checkpoint=_pointer(),
            )
        assert latch.incident is not None
        assert latch.authority.same_binding(latch.incident.authority)


def test_first_terminal_incident_is_immutable_and_forbids_later_steps() -> None:
    authority = _authority()
    published: list[ZeroNoticeSliceIncidentV2] = []
    latch = ZeroNoticeStepLatchV2(authority, incident_sink=published.append)
    assert latch.authorize_new_step(authority) == 1
    incident = latch.latch(
        authority,
        trigger_kind="METADATA_TRUE",
        recovery_checkpoint=_pointer(),
    )
    assert incident.authorized_steps_before_latch == 1
    assert published == [incident]
    assert (
        latch.latch(
            authority,
            trigger_kind="METADATA_TRUE",
            recovery_checkpoint=_pointer(),
        )
        is incident
    )
    with pytest.raises(TerminalIncidentImmutableV2):
        latch.latch(
            authority,
            trigger_kind="METADATA_AMBIGUOUS",
            recovery_checkpoint=_pointer(),
        )
    with pytest.raises(NewStepsForbiddenV2) as captured:
        latch.authorize_new_step(authority)
    assert captured.value.incident is incident


def test_failed_incident_publication_retries_exact_object_only() -> None:
    authority = _authority()
    calls: list[ZeroNoticeSliceIncidentV2] = []

    def flaky(incident: ZeroNoticeSliceIncidentV2) -> None:
        calls.append(incident)
        if len(calls) == 1:
            raise RuntimeError("transient")

    latch = ZeroNoticeStepLatchV2(authority, incident_sink=flaky)
    with pytest.raises(IncidentPublicationErrorV2) as failed:
        latch.latch(
            authority,
            trigger_kind="METADATA_AMBIGUOUS",
            recovery_checkpoint=_pointer(),
        )
    terminal = failed.value.incident
    assert latch.incident is terminal
    assert not latch.new_steps_allowed
    assert not latch.incident_published
    with pytest.raises(NewStepsForbiddenV2):
        latch.authorize_new_step(authority)

    assert latch.retry_incident_publication()
    assert calls == [terminal, terminal]
    assert calls[0] is calls[1]
    assert latch.incident_published
    assert not latch.retry_incident_publication()


def test_retry_before_terminal_incident_is_forbidden() -> None:
    latch = ZeroNoticeStepLatchV2(_authority(), incident_sink=lambda _: None)
    with pytest.raises(ZeroNoticePreemptionError, match="before"):
        latch.retry_incident_publication()


def test_concurrent_steps_and_latch_have_one_linearization_order() -> None:
    authority = _authority()
    latch = ZeroNoticeStepLatchV2(authority, incident_sink=lambda _: None)
    worker_count = 64
    barrier = threading.Barrier(worker_count + 1)

    def start_step() -> int | None:
        barrier.wait()
        try:
            return latch.authorize_new_step(authority)
        except NewStepsForbiddenV2:
            return None

    def terminal() -> ZeroNoticeSliceIncidentV2:
        barrier.wait()
        return latch.latch(
            authority,
            trigger_kind="SLICE_MEMBER_LOST",
            recovery_checkpoint=_pointer(),
        )

    with ThreadPoolExecutor(max_workers=worker_count + 1) as pool:
        step_futures = [pool.submit(start_step) for _ in range(worker_count)]
        incident_future = pool.submit(terminal)
        results = [future.result() for future in step_futures]
        incident = incident_future.result()

    permits = sorted(item for item in results if item is not None)
    assert permits == list(range(1, len(permits) + 1))
    assert incident.authorized_steps_before_latch == len(permits)
    assert latch.authorized_step_count == len(permits)
    with pytest.raises(NewStepsForbiddenV2):
        latch.authorize_new_step(authority)


def test_recovery_requires_expired_lease_advanced_fence_and_fresh_whole_mig() -> None:
    incident = _incident(pointer=_pointer())
    replacement = _authority(2)
    plan = ZeroNoticeRecoveryPlanV2.build(incident, replacement)
    assert plan.expired_fence == 1
    assert plan.replacement_fence == 2
    assert plan.old_lease_expired
    assert plan.old_fence_invalidated
    assert plan.fresh_whole_mig_required

    stale_fence = _authority(2, fence=1)
    with pytest.raises(ProtocolError, match="fence must advance"):
        ZeroNoticeRecoveryPlanV2.build(incident, stale_fence)

    premature = _authority(2, acquired=1_999, expires=3_000)
    with pytest.raises(ProtocolError, match="lease expiry"):
        ZeroNoticeRecoveryPlanV2.build(incident, premature)

    successor = _deployment(2)
    old_deployment = incident.authority.deployment
    reused_mig = replace(
        successor,
        mig_name=old_deployment.mig_name,
        mig_id=old_deployment.mig_id,
        mig_self_link=old_deployment.mig_self_link,
    )
    reused_authority = _authority_for_deployment(
        reused_mig,
        fence=2,
        acquired=2_000,
        expires=3_000,
    )
    with pytest.raises(ProtocolError, match="fresh MIG"):
        ZeroNoticeRecoveryPlanV2.build(incident, reused_authority)


def test_future_fence_and_other_work_checkpoint_are_rejected() -> None:
    future = _pointer(_sealed(source_attempt_id="attempt-2", source_fence=2))
    with pytest.raises(ProtocolError, match="future fence"):
        _incident(pointer=future)

    other = _pointer(replace(_sealed(), work_id="different-work"))
    with pytest.raises(ProtocolError, match="another work item"):
        _incident(pointer=other)

    pointer = _pointer()
    outside = replace(
        pointer,
        pointer_object=replace(
            pointer.pointer_object,
            uri="gs://attacker-control/checkpoints/latest.json",
        ),
    )
    with pytest.raises(ProtocolError, match="outside.*controller root"):
        _incident(pointer=outside)


def test_strict_wire_schema_and_nested_bool_rejection() -> None:
    incident = _incident(pointer=_pointer())
    value = incident.as_dict()
    value["emergency_callback"] = "save-now"
    with pytest.raises(ProtocolError, match="unknown fields"):
        ZeroNoticeSliceIncidentV2.from_dict(value)

    encoded = incident.canonical_json()
    duplicate = encoded[:-1] + ',"guaranteed_notice_ms":120000}'
    with pytest.raises(ProtocolError, match="duplicate JSON field"):
        ZeroNoticeSliceIncidentV2.from_json(duplicate)

    value = incident.as_dict()
    value["authority"]["lease"]["fence"] = True
    with pytest.raises(ProtocolError, match="booleans are forbidden"):
        ZeroNoticeSliceIncidentV2.from_dict(value)


def test_records_are_frozen() -> None:
    incident = _incident()
    with pytest.raises(FrozenInstanceError):
        incident.guaranteed_notice_ms = 75_000  # type: ignore[misc]
