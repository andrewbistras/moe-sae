from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import threading
from typing import Any

import pytest

from moe_sae.r1.control_store import CASConflict, MemoryJSONCASStore
from moe_sae.r1.placement_v2 import DeploymentEnvelopeV2, GCSObjectVersionRefV2
from moe_sae.r1.preemption_v2 import AttemptAuthorityV2
from moe_sae.r1.protocol import (
    AttemptLeaseV1,
    ContentRefV1,
    DeploymentIdentityV1,
    ProtocolError,
    RegionalReplicaV1,
)
from moe_sae.r1.region_scheduler import (
    BulkLocality,
    ClaimConflict,
    NoLocalCandidate,
    RegionScheduler,
    RegionSchedulingError,
    ScheduledClaim,
    V6E64Candidate,
    require_attempt_binding,
)


CHECKPOINT_BYTES = 649_674_223_616


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _bulk(label: str, payload_bytes: int, *regions: str) -> BulkLocality:
    digest = _sha(f"{label}-sealed-index")
    replicas = tuple(
        RegionalReplicaV1(
            region=region,
            uri=(
                f"gs://r1-{region}/records/sha256/"
                f"{digest[:2]}/{digest}.json"
            ),
            generation=index + 1,
            size_bytes=4096,
            crc32c="AAAAAA==",
            sha256=digest,
        )
        for index, region in enumerate(regions)
    )
    return BulkLocality(
        ContentRefV1(
            sha256=digest,
            size_bytes=4096,
            media_type="application/json",
            replicas=replicas,
        ),
        payload_bytes,
    )


INPUTS = (
    _bulk(
        "deepseek-prefix-checkpoint",
        CHECKPOINT_BYTES,
        "us-central1",
        "us-east1",
    ),
    _bulk("token-input", 8_589_934_592, "us-central1", "us-east1"),
)
CANDIDATES = (
    V6E64Candidate("us-central1-b", 1_000_000),
    # TRC allocation has zero cash compute cost; zero must remain honest.
    V6E64Candidate("us-east1-d", 0),
    V6E64Candidate("europe-west4-a", 800_000),
)


class TracingStore:
    def __init__(self) -> None:
        self.memory = MemoryJSONCASStore()
        self.calls = dict.fromkeys(("read", "create", "cas"), 0)

    def read(self, uri: str) -> Any:
        self.calls["read"] += 1
        return self.memory.read(uri)

    def create(self, uri: str, value: Any) -> Any:
        self.calls["create"] += 1
        return self.memory.create(uri, value)

    def compare_and_swap(
        self, uri: str, expected_generation: int, value: Any
    ) -> Any:
        self.calls["cas"] += 1
        return self.memory.compare_and_swap(uri, expected_generation, value)

def _allocator(
    store: Any | None = None, *, retries: int = 8
) -> RegionScheduler:
    return RegionScheduler(
        MemoryJSONCASStore() if store is None else store,
        base_uri="gs://r1-control",
        campaign_id="r1-headline",
        clock_ms=lambda: 1_750_000_000_000,
        max_cas_retries=retries,
    )


def _claim(
    allocator: RegionScheduler,
    *,
    claimant: str = "controller-a",
    fence: int = 0,
    candidates: tuple[V6E64Candidate, ...] = CANDIDATES,
    inputs: tuple[BulkLocality, ...] = INPUTS,
    excluded: tuple[str, ...] = (),
) -> ScheduledClaim:
    return allocator.claim(
        work_id="harvest-range-0001",
        claimant_id=claimant,
        source_fence=fence,
        candidates=candidates,
        bulk_inputs=inputs,
        excluded_zones=excluded,
    )


def test_cheapest_fully_local_candidate_wins_without_list_or_copy() -> None:
    store = TracingStore()
    claim = _claim(_allocator(store))

    assert claim.candidate.zone == "us-east1-d"  # Europe lacks both inputs
    assert {item.replicas[0].region for item in claim.local_inputs} == {
        "us-east1"
    }
    assert claim.local_payload_bytes > CHECKPOINT_BYTES
    assert claim.attempt_epoch == 1
    assert store.calls == {"read": 1, "create": 1, "cas": 0}
    assert not hasattr(store, "list")
    assert not hasattr(store, "copy")


def test_claims_are_idempotent_monotonic_region_movable_and_single_winner() -> None:
    allocator = _allocator()
    first = _claim(allocator)
    assert _claim(allocator) == first
    second = _claim(
        allocator,
        claimant="controller-b",
        fence=first.attempt_epoch,
        excluded=(first.candidate.zone,),
    )
    assert (second.attempt_epoch, second.candidate.zone) == (2, "us-central1-b")
    with pytest.raises(ClaimConflict):
        _claim(allocator, fence=0)
    with pytest.raises(ClaimConflict):
        _claim(
            allocator,
            claimant="controller-c",
            fence=second.source_fence,
            excluded=(first.candidate.zone,),
        )

    competing = _allocator()
    barrier = threading.Barrier(2)

    def contend(claimant: str) -> str:
        barrier.wait()
        try:
            return _claim(competing, claimant=claimant).claimant_id
        except ClaimConflict:
            return "lost"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(contend, ("controller-a", "controller-b")))
    assert results.count("lost") == 1


def test_missing_local_input_fails_before_control_store_io() -> None:
    store = TracingStore()
    split = (
        _bulk("checkpoint", CHECKPOINT_BYTES, "us-central1"),
        _bulk("input", 1024, "us-east1"),
    )
    with pytest.raises(NoLocalCandidate):
        _claim(_allocator(store), inputs=split)
    assert all(count == 0 for count in store.calls.values())


def test_bulk_locality_requires_content_addressed_generation_pinned_index() -> None:
    good = INPUTS[0].sealed_index
    foreign = replace(
        good.replicas[0],
        uri="gs://r1-us-central1/checkpoints/latest.json",
    )
    with pytest.raises(ProtocolError, match="content addressed"):
        BulkLocality(replace(good, replicas=(foreign,)), CHECKPOINT_BYTES)


class ConflictingCASStore(TracingStore):
    def compare_and_swap(
        self, uri: str, expected_generation: int, value: Any
    ) -> Any:
        self.calls["cas"] += 1
        current = self.memory.read(uri)
        raise CASConflict(
            uri,
            expected_generation=expected_generation,
            actual_generation=current.generation,
        )


def test_generation_cas_retry_is_bounded() -> None:
    store = ConflictingCASStore()
    allocator = _allocator(store, retries=3)
    first = _claim(allocator)
    with pytest.raises(ClaimConflict, match="after 3 attempts"):
        _claim(
            allocator,
            claimant="controller-b",
            fence=first.attempt_epoch,
            excluded=(first.candidate.zone,),
        )
    assert store.calls["cas"] == 3


def _authority(
    claim: ScheduledClaim, *, executor_id: str | None = None
) -> AttemptAuthorityV2:
    project = "neural-dragnet"
    zone, region = claim.candidate.zone, claim.candidate.region
    attempt = claim.attempt_id
    root = "gs://r1-control/campaigns/r1-headline/"
    launch = f'{{"attempt":"{attempt}"}}'.encode()
    deployment = DeploymentEnvelopeV2(
        base_image_resolution_sha256=_sha("base-image"),
        image_publication_sha256=_sha("image-publication"),
        project=project,
        region=region,
        zone=zone,
        slice_attempt_id=attempt,
        accelerator_type="v6e-64",
        machine_type="ct6e-standard-4t",
        machine_type_self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{project}/"
            f"zones/{zone}/machineTypes/ct6e-standard-4t"
        ),
        topology="8x8",
        vm_count=16,
        chips_per_vm=4,
        chip_count=64,
        image_name="r1-v6e-runtime-20260723",
        image_id=2_012_345_678_901_234_567,
        image_self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{project}/"
            "global/images/r1-v6e-runtime-20260723"
        ),
        instance_template_name=f"r1-template-e{claim.attempt_epoch}",
        instance_template_id=4_012_345_678_901_234_567,
        instance_template_self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{project}/"
            f"global/instanceTemplates/r1-template-e{claim.attempt_epoch}"
        ),
        resource_policy_name="r1-v6e-high-throughput",
        resource_policy_id=5_012_345_678_901_234_567,
        resource_policy_self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{project}/"
            f"regions/{region}/resourcePolicies/r1-v6e-high-throughput"
        ),
        mig_name=f"r1-slice-e{claim.attempt_epoch}",
        mig_id=6_012_345_678_901_234_567,
        mig_self_link=(
            f"https://www.googleapis.com/compute/v1/projects/{project}/"
            f"zones/{zone}/instanceGroupManagers/r1-slice-e{claim.attempt_epoch}"
        ),
        mig_scope="ZONAL",
        target_size=16,
        target_size_policy="BULK",
        default_action_on_failure="DO_NOTHING",
        replacement_policy="FRESH_WHOLE_MIG",
        provisioning_model="SPOT",
        termination_action="DELETE",
        maintenance_policy="TERMINATE",
        service_account_email="r1-worker@neural-dragnet.iam.gserviceaccount.com",
        service_account_unique_id="123456789012345678901",
        controller_root_uri=root,
        launch_ref=GCSObjectVersionRefV2(
            uri=f"{root}launch/{attempt}.json",
            generation=101,
            size_bytes=len(launch),
            sha256=hashlib.sha256(launch).hexdigest(),
        ),
    )
    identity = DeploymentIdentityV1(
        implementation_identity_sha256=_sha("implementation"),
        deployment_envelope_sha256=deployment.digest,
        device_inventory_sha256=_sha("inventory"),
        attempt_id=attempt,
    )
    return AttemptAuthorityV2(
        lease=AttemptLeaseV1(
            campaign_id=claim.campaign_id,
            work_id=claim.work_id,
            attempt_id=attempt,
            fence=claim.attempt_epoch,
            executor_id=claim.claimant_id if executor_id is None else executor_id,
            deployment_identity_sha256=identity.digest,
            acquired_at_unix_ms=1_750_000_000_100,
            expires_at_unix_ms=1_750_003_600_100,
        ),
        deployment_identity=identity,
        deployment=deployment,
    )


def test_execution_requires_exact_existing_controller_authority() -> None:
    claim = _claim(_allocator())
    require_attempt_binding(claim, _authority(claim))
    with pytest.raises(RegionSchedulingError, match="does not bind"):
        require_attempt_binding(
            claim, _authority(claim, executor_id="another-controller")
        )
