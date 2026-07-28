from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from moe_sae.object_store import ObjectIdentity, ObjectStoreError
from moe_sae.regional_execution import (
    MirrorDataset,
    RegionalExecutionError,
    RegionalMirrorStore,
    assert_spot_tpu_description,
    atomic_write_evidence,
    build_accelerator_contract,
    build_cleanup_plan,
    build_controller_observation,
    build_delivery_ledger,
    build_local_cache_observation,
    build_regional_mirror_specification,
    build_regional_placement,
    prewarm_regional_cache,
    seal_controller_mapping,
    seal_local_cache_inventory,
    seal_regional_mirror,
    validate_controller_mapping_receipt,
    validate_local_cache_inventory,
    validate_regional_mirror_specification,
    validate_sealed_regional_mirror,
)


CRC32C = "AAAAAA=="


def object_identity(
    uri: str,
    *,
    generation: int,
    payload: bytes,
) -> ObjectIdentity:
    import hashlib

    return ObjectIdentity(
        uri=uri,
        generation=generation,
        size_bytes=len(payload),
        crc32c=CRC32C,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def datasets() -> tuple[MirrorDataset, MirrorDataset, dict[str, bytes]]:
    payloads = {
        "gs://origin/target/shards/host-0/a.safetensors": b"a" * 31,
        "gs://origin/target/shards/host-1/b.safetensors": b"b" * 37,
        "gs://origin/overlay/shards/host-0/c.safetensors": b"c" * 41,
    }
    target = MirrorDataset(
        name="target",
        manifest_sha256="1" * 64,
        source_object_prefix="gs://origin/target",
        objects=tuple(
            object_identity(uri, generation=index + 1, payload=payloads[uri])
            for index, uri in enumerate(tuple(payloads)[:2])
        ),
    )
    overlay_uri = tuple(payloads)[2]
    overlay = MirrorDataset(
        name="overlay",
        manifest_sha256="2" * 64,
        source_object_prefix="gs://origin/overlay",
        objects=(
            object_identity(
                overlay_uri,
                generation=3,
                payload=payloads[overlay_uri],
            ),
        ),
    )
    return target, overlay, payloads


def sealed_mirror(
    *,
    region: str = "us-east1",
    selected: tuple[str, ...] = ("target", "overlay"),
):
    target, overlay, payloads = datasets()
    by_name = {"target": target, "overlay": overlay}
    plan = build_regional_mirror_specification(
        run_id="experiment4",
        region=region,
        mirror_prefix=f"gs://mirror-{region}/experiment4",
        datasets=tuple(by_name[name] for name in selected),
    )
    mirrors = {}
    for index, item in enumerate(plan["objects"]):
        source = ObjectIdentity(**item["source"])
        mirrors[item["mirror_uri"]] = ObjectIdentity(
            uri=item["mirror_uri"],
            generation=100 + index,
            size_bytes=source.size_bytes,
            crc32c=source.crc32c,
            sha256=source.sha256,
        )
    return plan, seal_regional_mirror(plan, mirrors), payloads


def accelerator_contract(
    *,
    model: str = "sae",
    condition: str = "A",
    region: str = "us-east1",
    zone: str = "us-east1-d",
):
    return build_accelerator_contract(
        run_id="experiment4",
        scientific_model=model,
        condition=condition,
        project="neural-dragnet",
        tpu_name=f"experiment4-{model}-{condition.lower()}",
        resource_create_time="2026-07-28T10:11:12.123Z",
        region=region,
        zone=zone,
    )


def controller_receipts(contract):
    process_order = (2, 0, 3, 1)
    observations = [
        build_controller_observation(
            contract=contract,
            probe_id="probe-20260728",
            controller_index=controller,
            process_index=process,
            process_count=4,
            global_device_count=16,
            local_device_count=4,
            platforms=("tpu",),
            device_kinds=("TPU v6 lite",),
            boot_id=f"00000000-0000-0000-0000-{controller + 1:012d}",
            hostname=f"worker-{controller}",
        )
        for controller, process in enumerate(process_order)
    ]
    mapping = seal_controller_mapping(
        contract=contract,
        probe_id="probe-20260728",
        observations=observations,
    )
    return observations, mapping


def test_mirror_plan_counts_each_exact_source_generation_once() -> None:
    target, overlay, _ = datasets()
    plan = build_regional_mirror_specification(
        run_id="experiment4",
        region="us-east1",
        mirror_prefix="gs://mirror-use1/experiment4",
        datasets=(target, overlay),
    )

    assert validate_regional_mirror_specification(plan) == plan
    assert plan["unique_source_objects"] == 3
    assert plan["source_transfer_bytes"] == 31 + 37 + 41
    assert all(
        item["copy_preconditions"]["source_generation_match"]
        == item["source"]["generation"]
        and item["copy_preconditions"]["destination_generation_match"] == 0
        for item in plan["objects"]
    )
    assert all(
        item["mirror_uri"].startswith(
            "gs://mirror-use1/experiment4/objects/origin/"
        )
        for item in plan["objects"]
    )


def test_sealed_mirror_requires_exact_content_and_destination_generation() -> None:
    plan, seal, _ = sealed_mirror()
    assert validate_sealed_regional_mirror(seal) == seal

    observations = {
        item["mirror"]["uri"]: ObjectIdentity(**item["mirror"])
        for item in seal["objects"]
    }
    first_uri = next(iter(observations))
    wrong = observations[first_uri]
    observations[first_uri] = ObjectIdentity(
        uri=wrong.uri,
        generation=wrong.generation + 1,
        size_bytes=wrong.size_bytes + 1,
        crc32c=wrong.crc32c,
        sha256=wrong.sha256,
    )
    with pytest.raises(RegionalExecutionError, match="differs"):
        seal_regional_mirror(plan, observations)


class MirrorBackingStore:
    def __init__(self, seal, source_payloads):
        self.payloads = {}
        self.downloaded: list[ObjectIdentity] = []
        for item in seal["objects"]:
            source = ObjectIdentity(**item["source"])
            mirror = ObjectIdentity(**item["mirror"])
            self.payloads[(mirror.uri, mirror.generation)] = source_payloads[
                source.uri
            ]

    def download(self, identity, destination):
        self.downloaded.append(identity)
        destination.write_bytes(
            self.payloads[(identity.uri, identity.generation)]
        )

    def verify(self, identity):
        if (identity.uri, identity.generation) not in self.payloads:
            raise RuntimeError("absent")

    def put_if_absent(self, *_args, **_kwargs):
        raise AssertionError("read-only test")

    def delete(self, _identity):
        raise AssertionError("read-only test")


def test_regional_prewarm_uses_mirror_but_retains_original_cache_identity(
    tmp_path: Path,
) -> None:
    _plan, seal, payloads = sealed_mirror()
    backing = MirrorBackingStore(seal, payloads)
    target_bytes = sum(
        item["source"]["size_bytes"]
        for item in seal["objects"]
        if "target" in item["datasets"]
    )

    receipt = prewarm_regional_cache(
        sealed_mirror=seal,
        backing_store=backing,
        cache_root=tmp_path / "cache",
        cache_max_bytes=target_bytes,
        datasets=("target",),
        max_workers=2,
    )

    assert receipt["scientific_manifests"] == [
        {"name": "target", "manifest_sha256": "1" * 64}
    ]
    assert receipt["object_bytes"] == target_bytes
    assert all(
        identity.uri.startswith("gs://mirror-us-east1/")
        for identity in backing.downloaded
    )
    assert {
        path.stem
        for path in (tmp_path / "cache").glob("*.safetensors")
    } == {
        item["source"]["sha256"]
        for item in seal["objects"]
        if "target" in item["datasets"]
    }


def test_mirror_store_refuses_unplanned_original_object(tmp_path: Path) -> None:
    _plan, seal, payloads = sealed_mirror()
    store = RegionalMirrorStore(seal, MirrorBackingStore(seal, payloads))
    unknown = object_identity(
        "gs://origin/target/shards/host-9/unknown.safetensors",
        generation=99,
        payload=b"unknown",
    )

    with pytest.raises(ObjectStoreError, match="absent"):
        store.download(unknown, tmp_path / "unknown")
    with pytest.raises(ObjectStoreError, match="read-only"):
        store.delete(unknown)


def test_delivery_ledger_rejects_two_independent_copies_to_one_region() -> None:
    _plan, seal, _ = sealed_mirror()
    with pytest.raises(RegionalExecutionError, match="one combined mirror"):
        build_delivery_ledger((seal, seal))

    _eu_plan, eu, _ = sealed_mirror(
        region="europe-west4",
        selected=("target",),
    )
    ledger = build_delivery_ledger((seal, eu))
    assert ledger["total_cross_region_bytes"] == (
        seal["source_transfer_bytes"] + eu["source_transfer_bytes"]
    )


def test_spot_attestation_rejects_on_demand_wrong_shape_and_old_incarnation():
    contract = accelerator_contract()
    description = {
        "name": (
            "projects/neural-dragnet/locations/us-east1-d/nodes/"
            "experiment4-sae-a"
        ),
        "acceleratorType": "v6e-16",
        "state": "READY",
        "createTime": "2026-07-28T10:11:12.123Z",
        "schedulingConfig": {"spot": True, "preemptible": True},
    }
    attestation = assert_spot_tpu_description(description, contract)
    assert attestation["spot"] is True

    for mutation in (
        lambda value: value["schedulingConfig"].update(spot=False),
        lambda value: value.update(acceleratorType="v6e-64"),
        lambda value: value.update(createTime="2026-07-28T10:11:13Z"),
    ):
        changed = copy.deepcopy(description)
        mutation(changed)
        with pytest.raises(RegionalExecutionError, match="exact ready Spot"):
            assert_spot_tpu_description(changed, contract)


def test_fresh_controller_mapping_is_exact_permutation_and_incarnation_bound():
    contract = accelerator_contract()
    observations, mapping = controller_receipts(contract)

    assert validate_controller_mapping_receipt(
        mapping,
        expected_contract_sha256=contract["contract_sha256"],
    ) == mapping
    assert [
        row["process_index"]
        for row in mapping["controller_to_jax_process_index"]
    ] == [2, 0, 3, 1]

    duplicated = list(observations)
    duplicated[-1] = observations[0]
    with pytest.raises(RegionalExecutionError, match="one-to-one"):
        seal_controller_mapping(
            contract=contract,
            probe_id="probe-20260728",
            observations=duplicated,
        )

    another = accelerator_contract(condition="R")
    with pytest.raises(RegionalExecutionError, match="contract changed"):
        validate_controller_mapping_receipt(
            mapping,
            expected_contract_sha256=another["contract_sha256"],
        )


def test_cache_inventory_requires_10m_lane_headroom_on_every_worker() -> None:
    contract = accelerator_contract()
    _observations, mapping = controller_receipts(contract)
    process_by_controller = [2, 0, 3, 1]
    caches = [
        build_local_cache_observation(
            contract=contract,
            controller_mapping=mapping,
            controller_index=controller,
            cache_root="/dev/shm/moe-sae",
            required_lane_bytes=147_000_000_000,
            required_free_bytes=210_000_000_000,
            available_bytes=300_000_000_000,
            mem_available_bytes=600_000_000_000,
            mount_target="/dev/shm",
            mount_source="tmpfs",
            filesystem_type="tmpfs",
            mount_options=("rw", "nodev", "nosuid"),
            boot_id=f"10000000-0000-0000-0000-{controller + 1:012d}",
        )
        for controller in range(4)
    ]
    inventory = seal_local_cache_inventory(
        contract=contract,
        controller_mapping=mapping,
        observations=caches,
    )

    assert validate_local_cache_inventory(
        inventory,
        expected_contract_sha256=contract["contract_sha256"],
        expected_controller_mapping_sha256=mapping["mapping_sha256"],
    ) == inventory
    assert [
        worker["process_index"] for worker in inventory["workers"]
    ] == process_by_controller

    with pytest.raises(RegionalExecutionError, match="10M process lane"):
        build_local_cache_observation(
            contract=contract,
            controller_mapping=mapping,
            controller_index=0,
            cache_root="/dev/shm/moe-sae",
            required_lane_bytes=147_000_000_000,
            required_free_bytes=161_700_000_000,
            available_bytes=300_000_000_000,
            mem_available_bytes=600_000_000_000,
            mount_target="/dev/shm",
            mount_source="tmpfs",
            filesystem_type="tmpfs",
            mount_options=("rw", "nodev"),
            boot_id="20000000-0000-0000-0000-000000000001",
        )


def test_regional_placement_and_cleanup_are_region_local_and_plan_only() -> None:
    contract = accelerator_contract()
    _plan, mirror, _ = sealed_mirror()
    placement = build_regional_placement(
        contract=contract,
        mirror=mirror,
        control_root="/regional/control/experiment4/sae/A",
        control_zone="us-east1-d",
        checkpoint_prefix="gs://checkpoint-use1/experiment4/sae/A",
        checkpoint_bucket_region="us-east1",
    )
    cleanup = build_cleanup_plan(
        run_id="experiment4",
        central_result_prefix="gs://central/results/experiment4",
        placements=(placement,),
        resources=(
            {
                "resource_type": "tpu",
                "resource_name": "experiment4-sae-a",
                "region": "us-east1",
                "run_label": "experiment4",
            },
            {
                "resource_type": "gcs_mirror",
                "resource_name": "mirror-us-east1",
                "region": "us-east1",
                "run_label": "experiment4",
            },
        ),
    )
    assert cleanup["execution_policy"] == "plan-only-no-implicit-cloud-deletion"
    assert cleanup["ordered_stages"][-1].endswith("inventory-empty")

    with pytest.raises(RegionalExecutionError, match="regional"):
        build_regional_placement(
            contract=contract,
            mirror=mirror,
            control_root="/regional/control/experiment4/sae/A",
            control_zone="us-east1-d",
            checkpoint_prefix="gs://checkpoint-eu/experiment4/sae/A",
            checkpoint_bucket_region="europe-west4",
        )


def test_atomic_evidence_never_overwrites_a_different_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    atomic_write_evidence(path, {"schema": "test.v1", "value": 1})
    atomic_write_evidence(path, {"schema": "test.v1", "value": 1})
    assert json.loads(path.read_text())["value"] == 1
    with pytest.raises(RegionalExecutionError, match="differs"):
        atomic_write_evidence(path, {"schema": "test.v1", "value": 2})


def test_fresh_controller_probe_dry_run_declares_identity_and_cache_fanout():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "tpu"
        / "probe_controller_identity.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--project",
            "neural-dragnet",
            "--tpu-name",
            "experiment4-sae-a",
            "--zone",
            "us-east1-d",
            "--project-commit",
            "a" * 40,
            "--project-root",
            "/opt/moe-sae",
            "--remote-uv",
            "/usr/local/bin/uv",
            "--contract",
            "/regional/control/contract.json",
            "--probe-id",
            "probe-20260728",
            "--probe-root",
            "/regional/control/probe-20260728",
            "--cache-root",
            "/dev/shm/moe-sae",
            "--required-lane-bytes",
            "147000000000",
            "--internal-ip",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert len(payload["commands"]) == 4
    assert len(payload["cache_commands"]) == 4
    assert payload["cloud_mutation"] is False
    assert all(
        "--internal-ip" in command for command in payload["commands"]
    )
    assert all(
        "observe-controller" in " ".join(command)
        for command in payload["commands"]
    )
    assert all(
        "observe-cache" in " ".join(command)
        for command in payload["cache_commands"]
    )
