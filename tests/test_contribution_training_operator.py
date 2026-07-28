from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    build_recomposition_sidecar,
    canonical_sha256,
)
from moe_sae.contribution_config import MAXTEXT_FORK_COMMIT
from moe_sae.contribution_performance import (
    ExactPerformanceIdentity,
    QualificationEvidence,
    build_qualification_receipt,
)
from moe_sae.contribution_recomposition import (
    SUPPORTED_ACCUMULATION_ORDER,
)
from moe_sae.object_store import ObjectIdentity
from moe_sae.regional_execution import (
    MirrorDataset,
    build_accelerator_contract,
    build_controller_observation,
    build_local_cache_observation,
    build_regional_mirror_specification,
    seal_controller_mapping,
    seal_local_cache_inventory,
    seal_regional_mirror,
)

ROOT = Path(__file__).resolve().parents[1]
OPERATOR_PATH = ROOT / "scripts" / "tpu" / "train_contribution.py"
CONFIG_PATH = ROOT / "configs" / "gemma4_contribution_v1.yaml"
HORIZON_TEMPLATE = ROOT / "configs" / "gemma4_contribution_horizon_v1.yaml"


def _load(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def operator():
    return _load(OPERATOR_PATH, "_test_contribution_training_operator")


def _identity() -> ExactPerformanceIdentity:
    return ExactPerformanceIdentity("a" * 64, "b" * 64, "c" * 64)


def _qualification_receipt() -> dict:
    identity = _identity()
    return build_qualification_receipt(
        QualificationEvidence(
            identity=identity,
            source_evidence_sha256="d" * 64,
            useful_flops_ledger_sha256="e" * 64,
            qualification_wall_seconds=10.0,
            measured_device_seconds=1.0,
            persisted_window_seconds=1.0,
            useful_flops=200,
            peak_flops_per_second=1_000,
            warmup_steps=1,
            synchronized_measurement_steps=3,
            measured_graph_compile_count=1,
            auxiliary_validation_compile_count=1,
            total_compile_count=2,
            recompilations_after_warmup=0,
            oom_events=0,
            nonfinite_events=0,
            capacity_overflow_events=0,
            dropped_routed_tokens=0,
            output_tokens=100,
            persisted_tokens=100,
            max_writer_backlog_batches=0,
            writer_backlog_limit_batches=2,
            writer_failures=0,
            writer_queue_drained=True,
            static_shapes=True,
            parameters_resident_before_timing=True,
            inputs_resident_before_timing=True,
            synchronized_device_timing=True,
            instrumentation_disabled_output_sha256="f" * 64,
            instrumentation_enabled_output_sha256="f" * 64,
            writer_disabled_output_tokens=100,
            writer_disabled_device_seconds=1.0,
        )
    )


def _sidecar() -> dict:
    scale = np.full((HIDDEN_SIZE,), 0x3F80, dtype=np.uint16)
    return build_recomposition_sidecar(
        checkpoint_sha256="1" * 64,
        maxtext_fork_commit=MAXTEXT_FORK_COMMIT,
        source_fingerprints={"environment": "2" * 64},
        routed_rmsnorm_scale=scale,
        shared_rmsnorm_scale=scale,
        outer_ffn_rmsnorm_scale=scale,
        layer_scalar=np.asarray([0x3F80], dtype=np.uint16),
        accumulation_order=SUPPORTED_ACCUMULATION_ORDER,
    )


def _bundle(operator, sidecar: dict) -> dict:
    receipt = _qualification_receipt()
    identity = _identity()
    unsigned = {
        "schema_version": operator.COORDINATOR_BUNDLE_SCHEMA_VERSION,
        "identity_relationship": (
            "receipt-binds-observed-executable-identity-not-"
            "authorization-contract-identity"
        ),
        "authorization_contract_sha256": "3" * 64,
        "host_evidence_sha256": [
            f"{index + 4:x}" * 64 for index in range(4)
        ],
        "controller_to_jax_process_index": [
            {"controller_index": 0, "process_index": 2},
            {"controller_index": 1, "process_index": 0},
            {"controller_index": 2, "process_index": 3},
            {"controller_index": 3, "process_index": 1},
        ],
        "observed_executable_identity": {
            **identity.to_dict(),
            "identity_sha256": identity.identity_sha256,
        },
        "recomposition_sidecar_sha256": sidecar["sidecar_sha256"],
        "recomposition_sidecar_canonical_sha256": canonical_sha256(sidecar),
        "source_evidence": {"schema_version": "test-source"},
        "useful_flops_ledger": {"schema_version": "test-ledger"},
        "qualification_receipt": receipt,
    }
    return {
        **unsigned,
        "coordinator_bundle_sha256": canonical_sha256(unsigned),
    }


def _write_inputs(operator, root: Path) -> dict[str, Path]:
    sidecar = _sidecar()
    sidecar_path = root / "recomposition-sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    bundle_path = root / "coordinator-bundle.json"
    bundle_path.write_text(
        json.dumps(_bundle(operator, sidecar)),
        encoding="utf-8",
    )
    capture = root / "capture"
    capture.mkdir()
    manifest_unsigned = {
        "schema_version": "gemma4-expert-contribution-v2",
        "sealed": True,
    }
    (capture / "manifest.json").write_text(
        json.dumps(
            {
                **manifest_unsigned,
                "manifest_sha256": canonical_sha256(manifest_unsigned),
            }
        ),
        encoding="utf-8",
    )
    return {
        "sidecar": sidecar_path,
        "bundle": bundle_path,
        "capture": capture,
    }


def _write_v6e_incarnation_evidence(
    root: Path,
    *,
    tpu_name: str,
    resource_create_time: str,
    process_order: tuple[int, int, int, int],
) -> tuple[Path, Path]:
    contract = build_accelerator_contract(
        run_id="phase2-A-seed0",
        scientific_model="sae",
        condition="A",
        project="neural-dragnet",
        tpu_name=tpu_name,
        resource_create_time=resource_create_time,
        region="us-east1",
        zone="us-east1-d",
    )
    observations = [
        build_controller_observation(
            contract=contract,
            probe_id=f"probe-{root.name}",
            controller_index=controller,
            process_index=process,
            process_count=4,
            global_device_count=16,
            local_device_count=4,
            platforms=("tpu",),
            device_kinds=("TPU v6 lite",),
            boot_id=(
                f"{controller + 1:08x}-0000-0000-0000-"
                f"{controller + 1:012d}"
            ),
            hostname=f"{root.name}-worker-{controller}",
        )
        for controller, process in enumerate(process_order)
    ]
    mapping = seal_controller_mapping(
        contract=contract,
        probe_id=f"probe-{root.name}",
        observations=observations,
    )
    root.mkdir()
    mapping_path = root / "controller-mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    cache_observations = [
        build_local_cache_observation(
            contract=contract,
            controller_mapping=mapping,
            controller_index=controller,
            cache_root="/dev/shm/contribution-sae",
            required_lane_bytes=147_000_000_000,
            required_free_bytes=210_000_000_000,
            available_bytes=300_000_000_000,
            mem_available_bytes=600_000_000_000,
            mount_target="/dev/shm",
            mount_source="tmpfs",
            filesystem_type="tmpfs",
            mount_options=("rw", "nodev", "nosuid"),
            boot_id=(
                f"{controller + 5:08x}-0000-0000-0000-"
                f"{controller + 5:012d}"
            ),
        )
        for controller in range(4)
    ]
    inventory = seal_local_cache_inventory(
        contract=contract,
        controller_mapping=mapping,
        observations=cache_observations,
    )
    inventory_path = root / "local-cache-inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    return mapping_path, inventory_path


def _write_regional_mirror(root: Path) -> Path:
    payload = b"immutable test capture"
    source = ObjectIdentity(
        uri="gs://origin-bucket/capture/shards/host-0/shard.safetensors",
        generation=1,
        size_bytes=len(payload),
        crc32c="AAAAAA==",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    dataset = MirrorDataset(
        name="target",
        manifest_sha256="a" * 64,
        source_object_prefix="gs://origin-bucket/capture",
        objects=(source,),
    )
    specification = build_regional_mirror_specification(
        run_id="phase2-A-seed0",
        region="us-east1",
        mirror_prefix="gs://regional-bucket/phase2-A-seed0",
        datasets=(dataset,),
    )
    destination_uri = specification["objects"][0]["mirror_uri"]
    mirror = seal_regional_mirror(
        specification,
        {
            destination_uri: ObjectIdentity(
                uri=destination_uri,
                generation=2,
                size_bytes=source.size_bytes,
                crc32c=source.crc32c,
                sha256=source.sha256,
            )
        },
    )
    path = root / "regional-mirror.json"
    path.write_text(json.dumps(mirror), encoding="utf-8")
    return path


def _settings(
    operator,
    tmp_path: Path,
    *,
    action: str = "qualify",
    maximum_runtime_seconds: int | None = None,
):
    return operator.Settings(
        gcp_project="neural-dragnet",
        tpu_name="gemma4-v4-32",
        tpu_zone="us-central2-b",
        project_commit="9" * 40,
        project_root="/home/researcher/src/moe-sae",
        remote_uv="/home/researcher/.local/bin/uv",
        capture_control_root="/nfs/contributions/phase2/train",
        object_prefix="gs://bucket/contributions/train",
        recomposition_sidecar="/nfs/qualification/recomposition-sidecar.json",
        qualification_bundle="/nfs/qualification/coordinator-bundle.json",
        cache_root="/dev/shm/contribution-sae",
        cache_max_bytes=40_000_000_000,
        sae_control_root="/nfs/experiment2/sae",
        checkpoint_payload_root="gs://bucket/sae-checkpoints",
        ssh_key_file=None,
        action=action,
        phase="phase2_mechanism_pilot",
        condition="A",
        seed=0,
        run_id="phase2-A-seed0",
        attempt_id=f"{action}-001",
        maximum_runtime_seconds=(
            maximum_runtime_seconds
            if maximum_runtime_seconds is not None
            else (600 if action == "qualify" else 7_200)
        ),
        local_log_root=tmp_path / f"{action}-logs",
    )


def test_internal_ip_transport_is_explicit_and_v6e_requires_it(
    operator,
    tmp_path: Path,
) -> None:
    settings = _settings(operator, tmp_path)
    external = operator._gcloud(settings, worker=0, remote_command="true")
    internal = operator._gcloud(
        replace(settings, internal_ip=True),
        worker=0,
        remote_command="true",
    )
    assert "--internal-ip" not in external
    assert "--internal-ip" in internal
    with pytest.raises(operator.OperatorError, match="TPU_USE_INTERNAL_IP=1"):
        operator._internal_ip_mode({}, accelerator_type="v6e-16")
    assert (
        operator._internal_ip_mode(
            {"TPU_USE_INTERNAL_IP": "1"},
            accelerator_type="v6e-16",
        )
        is True
    )


def test_explicit_tpu_ssh_user_is_validated_and_rendered(
    operator,
    tmp_path: Path,
) -> None:
    settings = replace(_settings(operator, tmp_path), ssh_user="abistras")
    command = operator._gcloud(settings, worker=0, remote_command="true")
    assert command[command.index("ssh") + 1] == "abistras@gemma4-v4-32"
    args = argparse.Namespace(
        action="train",
        scientific_model="sae",
        condition="A",
        phase="learning_horizon_10m",
        run_id="horizon-a",
        attempt_id="segment-1",
        seed=0,
        maximum_runtime_seconds=3_600,
        config="/nfs/contracts/sae.yaml",
    )
    environment = {
        "GCP_PROJECT": "neural-dragnet",
        "TPU_NAME": "gemma4-v4-32",
        "TPU_ZONE": "us-central2-b",
        "PROJECT_COMMIT": "a" * 40,
        "TPU_SSH_USER": "bad;user",
    }
    with pytest.raises(operator.OperatorError, match="TPU_SSH_USER"):
        operator._settings(environment, args)


def test_horizon_prefetch_and_direct_limits_cover_audited_v4_v6_only(
    operator,
    tmp_path: Path,
) -> None:
    finalized = tmp_path / "arbitrarily-renamed-finalized-config.yaml"
    finalized.write_text(
        HORIZON_TEMPLATE.read_text(encoding="utf-8").replace(
            "PENDING_SEALED_10M_TARGET_MANIFEST",
            "a" * 64,
        ),
        encoding="utf-8",
    )
    settings = replace(
        _settings(operator, tmp_path),
        accelerator_type="v6e-16",
        selected_config_path=str(finalized),
        phase="learning_horizon_10m",
    )
    assert settings.prefetch_hard_deadline_seconds == 1_800
    command = operator._prefetch_command(
        settings,
        controller_index=0,
        process_index=2,
    )
    joined = " ".join(command)
    assert "--accelerator-type v6e-16" in joined
    assert "--maximum-runtime-seconds 1800" in joined

    historical = replace(
        settings,
        selected_config_path=str(CONFIG_PATH),
    )
    assert historical.prefetch_hard_deadline_seconds == 600

    v4 = replace(
        settings,
        accelerator_type="v4-32",
        action="train",
        phase="learning_horizon_10m",
        maximum_runtime_seconds=43_200,
    )
    assert v4.prefetch_hard_deadline_seconds == 1_800
    assert v4.direct_runtime_limit_seconds == 43_200
    assert v4.admission_mode == operator.DIRECT_BOUNDED_ADMISSION
    assert operator._is_horizon_direct(
        action="train",
        maximum_runtime_seconds=43_200,
        accelerator_type="v4-32",
        config_path=str(finalized),
        phase="learning_horizon_10m",
    )

    v5 = replace(v4, accelerator_type="v5litepod-16")
    assert v5.prefetch_hard_deadline_seconds == 600
    assert v5.direct_runtime_limit_seconds == 3_600
    assert v5.admission_mode == operator.QUALIFIED_ADMISSION
    assert not operator._is_horizon_direct(
        action="train",
        maximum_runtime_seconds=43_200,
        accelerator_type="v5litepod-16",
        config_path=str(finalized),
        phase="learning_horizon_10m",
    )


@pytest.mark.parametrize(
    ("accelerator_type", "device_kind"),
    [
        ("v4-32", "TPU v4"),
        ("v6e-16", "TPU v6 lite"),
    ],
)
def test_audited_horizon_runtime_continuous_direct_run_uses_no_qualification(
    operator,
    tmp_path: Path,
    accelerator_type: str,
    device_kind: str,
) -> None:
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(json.dumps(_sidecar()), encoding="utf-8")
    contract = build_accelerator_contract(
        run_id="experiment4",
        scientific_model="sae",
        condition="F",
        project="neural-dragnet",
        tpu_name="experiment4-sae-f",
        resource_create_time="2026-07-28T10:11:12.123Z",
        region="europe-west4",
        zone="europe-west4-a",
        accelerator_type=accelerator_type,
    )
    observations = [
        build_controller_observation(
            contract=contract,
            probe_id="probe-direct",
            controller_index=index,
            process_index=index,
            process_count=4,
            global_device_count=16,
            local_device_count=4,
            platforms=("tpu",),
            device_kinds=(device_kind,),
            boot_id=f"00000000-0000-0000-0000-{index + 1:012d}",
            hostname=f"worker-{index}",
        )
        for index in range(4)
    ]
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            seal_controller_mapping(
                contract=contract,
                probe_id="probe-direct",
                observations=observations,
            )
        ),
        encoding="utf-8",
    )
    lineage = operator._fresh_direct_lineage(
        sidecar_path,
        fresh_controller_mapping_path=mapping_path,
        expected_tpu_name="experiment4-sae-f",
        expected_tpu_zone="europe-west4-a",
        expected_accelerator_type=accelerator_type,
    )
    assert lineage["qualification_bundle_sha256"] is None
    assert lineage["qualification_receipt_sha256"] is None
    assert lineage["admission_evidence_policy"].endswith(
        "no-qualification"
    )

    finalized = tmp_path / "finalized-horizon.yaml"
    finalized.write_text(
        HORIZON_TEMPLATE.read_text(encoding="utf-8").replace(
            "PENDING_SEALED_10M_TARGET_MANIFEST",
            "a" * 64,
        ),
        encoding="utf-8",
    )
    settings = replace(
        _settings(
            operator,
            tmp_path,
            action="train",
            maximum_runtime_seconds=43_200,
        ),
        tpu_name="experiment4-sae-f",
        tpu_zone="europe-west4-a",
        accelerator_type=accelerator_type,
        selected_config_path=str(finalized),
        phase="learning_horizon_10m",
        qualification_bundle=None,
        fresh_controller_mapping=str(mapping_path),
        regional_mirror_seal="/regional/control/mirror-seal.json",
        local_cache_inventory="/regional/control/cache-inventory.json",
        internal_ip=True,
    )
    command = operator._init_command(settings)
    remote = next(
        item for item in command if item.startswith("--command=")
    )
    assert "--qualification-bundle" not in remote
    assert settings.prefetch_hard_deadline_seconds == 1_800
    assert settings.direct_runtime_limit_seconds == 43_200
    assert settings.admission_mode == operator.DIRECT_BOUNDED_ADMISSION
    action = " ".join(
        operator._action_command(
            settings,
            controller_index=0,
            process_index=0,
        )
    )
    assert "--maximum-runtime-seconds 43200" in action
    assert "--direct-bounded-resident" in action


def test_v6e_action_child_receives_exact_accelerator_type(
    operator,
    tmp_path: Path,
) -> None:
    settings = replace(
        _settings(
            operator,
            tmp_path,
            action="train",
            maximum_runtime_seconds=3_600,
        ),
        accelerator_type="v6e-16",
        internal_ip=True,
    )
    command = operator._action_command(
        settings,
        controller_index=0,
        process_index=2,
    )
    joined = " ".join(command)
    assert "--accelerator-type v6e-16" in joined


def _init_args(
    operator,
    tmp_path: Path,
    inputs: dict[str, Path],
    *,
    action: str,
    attempt_id: str,
    maximum_runtime_seconds: int | None = None,
) -> argparse.Namespace:
    run_root = tmp_path / "sae" / "runs" / "phase2-A-seed0"
    attempt_root = (
        run_root / "operator-attempts" / f"{action}-{attempt_id}"
    )
    return argparse.Namespace(
        action=action,
        run_id="phase2-A-seed0",
        attempt_id=attempt_id,
        project_commit="9" * 40,
        config=str(CONFIG_PATH),
        phase="phase2_mechanism_pilot",
        condition="A",
        seed=0,
        capture_control_root=str(inputs["capture"]),
        object_prefix="gs://bucket/contributions/train",
        recomposition_sidecar=str(inputs["sidecar"]),
        qualification_bundle=str(inputs["bundle"]),
        cache_root="/dev/shm/contribution-sae",
        cache_max_bytes=40_000_000_000,
        sae_control_root=str(tmp_path / "sae"),
        run_root=str(run_root),
        attempt_root=str(attempt_root),
        checkpoint_payload_root="gs://bucket/sae-checkpoints",
        maximum_runtime_seconds=(
            maximum_runtime_seconds
            if maximum_runtime_seconds is not None
            else (600 if action == "qualify" else 7_200)
        ),
    )


def test_operator_is_executable_and_compiles(tmp_path: Path) -> None:
    assert os.access(OPERATOR_PATH, os.X_OK)
    py_compile.compile(
        str(OPERATOR_PATH),
        cfile=str(tmp_path / "train-contribution.pyc"),
        doraise=True,
    )


def test_lineage_accepts_nonidentity_permutation_and_rejects_tampering(
    operator,
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(operator, tmp_path)
    lineage = operator._qualification_lineage(
        inputs["bundle"],
        inputs["sidecar"],
    )
    assert lineage["controller_to_jax_process_index"] == [
        {"controller_index": 0, "process_index": 2},
        {"controller_index": 1, "process_index": 0},
        {"controller_index": 2, "process_index": 3},
        {"controller_index": 3, "process_index": 1},
    ]
    assert (
        lineage["qualification_receipt_sha256"]
        == _qualification_receipt()["receipt_sha256"]
    )

    bundle = json.loads(inputs["bundle"].read_text(encoding="utf-8"))
    bundle["controller_to_jax_process_index"][3]["process_index"] = 3
    unsigned = dict(bundle)
    unsigned.pop("coordinator_bundle_sha256")
    bundle["coordinator_bundle_sha256"] = canonical_sha256(unsigned)
    inputs["bundle"].write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(operator.OperatorError, match="not a permutation"):
        operator._qualification_lineage(
            inputs["bundle"],
            inputs["sidecar"],
        )


def test_attempt_contract_is_create_only_and_train_requires_qualification(
    operator,
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(operator, tmp_path)
    qualify_args = _init_args(
        operator,
        tmp_path,
        inputs,
        action="qualify",
        attempt_id="q001",
    )
    plan = operator.initialize_attempt(qualify_args)
    assert plan["controller_to_jax_process_index"][0] == {
        "controller_index": 0,
        "process_index": 2,
    }
    assert plan["config"] == str(CONFIG_PATH)
    assert plan["config_file_sha256"] == hashlib.sha256(
        CONFIG_PATH.read_bytes()
    ).hexdigest()
    assert plan["prefetch_hard_deadline_seconds"] == 600
    with pytest.raises(operator.OperatorError, match="already exists"):
        operator.initialize_attempt(qualify_args)

    train_args = _init_args(
        operator,
        tmp_path,
        inputs,
        action="train",
        attempt_id="t001",
    )
    with pytest.raises(operator.OperatorError, match="mode-selection"):
        operator.initialize_attempt(train_args)

    run_root = Path(train_args.run_root)
    (run_root / "mode-selection.json").write_text("{}", encoding="utf-8")
    receipt_root = run_root / "qualified-prefetch"
    receipt_root.mkdir()
    for item in plan["controller_to_jax_process_index"]:
        (receipt_root / (
            f"controller-c{item['controller_index']:05d}-"
            f"p{item['process_index']:05d}.json"
        )).write_text("{}", encoding="utf-8")
    trained = operator.initialize_attempt(train_args)
    assert trained["action"] == "train"
    assert trained["maximum_runtime_seconds"] == 7_200


def test_bounded_train_creates_direct_resident_contract_without_selection(
    operator,
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(operator, tmp_path)
    train_args = _init_args(
        operator,
        tmp_path,
        inputs,
        action="train",
        attempt_id="direct001",
        maximum_runtime_seconds=3_600,
    )
    plan = operator.initialize_attempt(train_args)
    assert plan["action"] == "train"
    assert plan["maximum_runtime_seconds"] == 3_600
    assert plan["admission_mode"] == operator.DIRECT_BOUNDED_ADMISSION
    run_root = Path(train_args.run_root)
    assert not (run_root / "mode-selection.json").exists()
    contract = json.loads(
        (run_root / "run-contract.json").read_text(encoding="utf-8")
    )
    assert contract["admission_mode"] == operator.DIRECT_BOUNDED_ADMISSION


def test_v6e_replacement_slice_reuses_one_scientific_run_contract(
    operator,
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(operator, tmp_path)
    finalized = tmp_path / "finalized-horizon.yaml"
    finalized.write_text(
        HORIZON_TEMPLATE.read_text(encoding="utf-8").replace(
            "PENDING_SEALED_10M_TARGET_MANIFEST",
            "a" * 64,
        ),
        encoding="utf-8",
    )
    mirror_path = _write_regional_mirror(tmp_path)
    first_mapping, first_inventory = _write_v6e_incarnation_evidence(
        tmp_path / "slice-one",
        tpu_name="horizon-sae-a-first",
        resource_create_time="2026-07-28T10:11:12.123Z",
        process_order=(2, 0, 3, 1),
    )
    second_mapping, second_inventory = _write_v6e_incarnation_evidence(
        tmp_path / "slice-two",
        tpu_name="horizon-sae-a-replacement",
        resource_create_time="2026-07-28T12:34:56.789Z",
        process_order=(1, 3, 0, 2),
    )

    def replacement_args(
        *,
        attempt_id: str,
        tpu_name: str,
        mapping_path: Path,
        inventory_path: Path,
    ) -> argparse.Namespace:
        args = _init_args(
            operator,
            tmp_path,
            inputs,
            action="train",
            attempt_id=attempt_id,
            maximum_runtime_seconds=3_600,
        )
        args.accelerator_type = "v6e-16"
        args.tpu_name = tpu_name
        args.tpu_zone = "us-east1-d"
        args.tpu_ssh_user = "abistras"
        args.config = str(finalized)
        args.phase = "learning_horizon_10m"
        args.cache_max_bytes = 160 * 1024**3
        args.fresh_controller_mapping = str(mapping_path)
        args.regional_mirror_seal = str(mirror_path)
        args.local_cache_inventory = str(inventory_path)
        return args

    first_args = replacement_args(
        attempt_id="segment001",
        tpu_name="horizon-sae-a-first",
        mapping_path=first_mapping,
        inventory_path=first_inventory,
    )
    first_plan = operator.initialize_attempt(first_args)
    contract_path = Path(first_args.run_root) / "run-contract.json"
    first_contract = json.loads(contract_path.read_text(encoding="utf-8"))

    second_args = replacement_args(
        attempt_id="segment002",
        tpu_name="horizon-sae-a-replacement",
        mapping_path=second_mapping,
        inventory_path=second_inventory,
    )
    second_plan = operator.initialize_attempt(second_args)
    second_contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert second_contract == first_contract
    assert (
        second_plan["run_contract_sha256"]
        == first_plan["run_contract_sha256"]
        == first_contract["run_contract_sha256"]
    )
    assert {
        "controller_to_jax_process_index",
        "fresh_controller_mapping_sha256",
        "fresh_controller_contract_sha256",
        "regional_mirror_sha256",
        "local_cache_inventory_sha256",
        "config",
        "capture_control_root",
        "recomposition_sidecar",
        "qualification_bundle",
        "cache_root",
        "cache_max_bytes",
    }.isdisjoint(first_contract)
    assert first_contract["run_contract_scope"].startswith(
        "stable-across-spot-slice-incarnations"
    )

    first_attestation = first_plan["operational_evidence"]
    second_attestation = second_plan["operational_evidence"]
    assert (
        first_attestation["execution_attestation_sha256"]
        != second_attestation["execution_attestation_sha256"]
    )
    for plan, attestation in (
        (first_plan, first_attestation),
        (second_plan, second_attestation),
    ):
        unsigned = dict(attestation)
        claimed = unsigned.pop("execution_attestation_sha256")
        assert claimed == canonical_sha256(unsigned)
        assert (
            attestation["controller_to_jax_process_index"]
            == plan["controller_to_jax_process_index"]
        )
        assert attestation["tpu_ssh_user"] == "abistras"

    for args, plan in (
        (first_args, first_plan),
        (second_args, second_plan),
    ):
        settings = replace(
            _settings(
                operator,
                tmp_path,
                action="train",
                maximum_runtime_seconds=3_600,
            ),
            tpu_name=args.tpu_name,
            tpu_zone=args.tpu_zone,
            accelerator_type="v6e-16",
            selected_config_path=str(finalized),
            phase=args.phase,
            sae_control_root=args.sae_control_root,
            capture_control_root=args.capture_control_root,
            recomposition_sidecar=args.recomposition_sidecar,
            qualification_bundle=None,
            cache_root=args.cache_root,
            cache_max_bytes=args.cache_max_bytes,
            checkpoint_payload_root=args.checkpoint_payload_root,
            attempt_id=args.attempt_id,
            regional_mirror_seal=args.regional_mirror_seal,
            fresh_controller_mapping=args.fresh_controller_mapping,
            local_cache_inventory=args.local_cache_inventory,
            internal_ip=True,
            ssh_user=args.tpu_ssh_user,
        )
        validated, mapping = operator._validate_attempt_plan(plan, settings)
        assert validated == plan
        assert mapping == operator._controller_mapping(plan)


def test_transcoder_operator_binds_and_forwards_source_overlay(
    operator,
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(operator, tmp_path)
    source = tmp_path / "source-overlay"
    source.mkdir()
    source_unsigned = {
        "schema_version": "gemma4-routed-input-overlay-v1",
        "sealed": True,
    }
    (source / "manifest.json").write_text(
        json.dumps(
            {
                **source_unsigned,
                "manifest_sha256": canonical_sha256(source_unsigned),
            }
        ),
        encoding="utf-8",
    )
    args = _init_args(
        operator,
        tmp_path,
        inputs,
        action="train",
        attempt_id="transcoder001",
        maximum_runtime_seconds=3_600,
    )
    args.scientific_model = "transcoder"
    args.config = str(ROOT / "configs" / "gemma4_transcoder_v2.yaml")
    args.phase = "mechanism_pilot"
    args.source_overlay_control_root = str(source)
    args.source_overlay_object_prefix = "gs://bucket/experiment3/source"
    plan = operator.initialize_attempt(args)
    contract = json.loads(
        (Path(args.run_root) / "run-contract.json").read_text(encoding="utf-8")
    )
    assert plan["admission_mode"] == operator.DIRECT_BOUNDED_ADMISSION
    assert contract["scientific_model"] == "transcoder"
    assert contract["source_overlay_control_root"] == str(source)
    assert (
        contract["source_overlay_manifest_sha256"]
        == canonical_sha256(source_unsigned)
    )

    settings = replace(
        _settings(
            operator,
            tmp_path,
            action="train",
            maximum_runtime_seconds=3_600,
        ),
        scientific_model="transcoder",
        source_overlay_control_root="/nfs/experiment3/source/train",
        source_overlay_object_prefix="gs://bucket/experiment3/source/train",
        phase="mechanism_pilot",
    )
    assert settings.config_path.endswith("configs/gemma4_transcoder_v2.yaml")
    for command in (
        operator._prefetch_command(
            settings, controller_index=1, process_index=3
        ),
        operator._action_command(
            settings, controller_index=1, process_index=3
        ),
    ):
        joined = " ".join(command)
        assert "--scientific-model transcoder" in joined
        assert "--source-overlay-control-root" in joined
        assert "/nfs/experiment3/source/train" in joined
        assert "--source-overlay-object-prefix" in joined


def test_commands_keep_controller_transport_and_jax_lane_distinct(
    operator,
    tmp_path: Path,
) -> None:
    settings = _settings(operator, tmp_path)
    prefetch = operator._prefetch_command(
        settings,
        controller_index=1,
        process_index=3,
    )
    joined = " ".join(prefetch)
    assert "--worker=1" in prefetch
    assert "moe_sae.contribution_training_worker" in joined
    assert "prefetch" in joined
    assert "--process-index 3" in joined
    assert "/process-p00003" in joined
    assert "--maximum-runtime-seconds 600" in joined
    assert "--supervision-dir" in joined

    qualify = operator._action_command(
        settings,
        controller_index=1,
        process_index=3,
    )
    joined = " ".join(qualify)
    assert "--worker=1" in qualify
    assert "qualify" in joined
    assert "/process-p00003" in joined
    assert "controller-c00001-p00003" in joined
    assert "--maximum-runtime-seconds" not in joined

    training = operator._action_command(
        _settings(operator, tmp_path, action="train"),
        controller_index=1,
        process_index=3,
    )
    joined = " ".join(training)
    assert "--maximum-runtime-seconds 7200" in joined
    assert "--checkpoint-control-root" in joined
    assert "--metric-path" in joined
    assert "--direct-bounded-resident" not in joined

    direct = operator._action_command(
        _settings(
            operator,
            tmp_path,
            action="train",
            maximum_runtime_seconds=3_600,
        ),
        controller_index=1,
        process_index=3,
    )
    joined = " ".join(direct)
    assert "--maximum-runtime-seconds 3600" in joined
    assert "--direct-bounded-resident" in joined


def test_qualification_runtime_is_exactly_600(operator, tmp_path: Path) -> None:
    settings = _settings(operator, tmp_path)
    assert settings.maximum_runtime_seconds == 600
    environment = {
        "GCP_PROJECT": settings.gcp_project,
        "TPU_NAME": settings.tpu_name,
        "TPU_ZONE": settings.tpu_zone,
        "PROJECT_COMMIT": settings.project_commit,
        "TPU_PROJECT_ROOT": settings.project_root,
        "REMOTE_UV": settings.remote_uv,
        "CONTRIBUTION_TRAIN_CAPTURE_ROOT": settings.capture_control_root,
        "CONTRIBUTION_OBJECT_PREFIX": settings.object_prefix,
        "CONTRIBUTION_RECOMPOSITION_SIDECAR": (
            settings.recomposition_sidecar
        ),
        "CONTRIBUTION_QUALIFICATION_BUNDLE": settings.qualification_bundle,
        "CONTRIBUTION_SAE_CACHE_ROOT": settings.cache_root,
        "CONTRIBUTION_SAE_CACHE_MAX_BYTES": str(settings.cache_max_bytes),
        "CONTRIBUTION_SAE_CONTROL_ROOT": settings.sae_control_root,
        "CONTRIBUTION_SAE_CHECKPOINT_ROOT": (
            settings.checkpoint_payload_root
        ),
    }
    parser = operator._public_parser()
    args = parser.parse_args(
        [
            "--action",
            "qualify",
            "--phase",
            settings.phase,
            "--condition",
            settings.condition,
            "--seed",
            "0",
            "--run-id",
            settings.run_id,
            "--attempt-id",
            settings.attempt_id,
            "--maximum-runtime-seconds",
            "599",
        ]
    )
    with pytest.raises(operator.OperatorError, match="fixed at 600"):
        operator._settings(environment, args)


def test_public_config_override_is_absolute_and_forwarded(
    operator,
    tmp_path: Path,
) -> None:
    historical = _settings(operator, tmp_path)
    environment = {
        "GCP_PROJECT": historical.gcp_project,
        "TPU_NAME": historical.tpu_name,
        "TPU_ZONE": historical.tpu_zone,
        "PROJECT_COMMIT": historical.project_commit,
        "TPU_PROJECT_ROOT": historical.project_root,
        "REMOTE_UV": historical.remote_uv,
        "CONTRIBUTION_TRAIN_CAPTURE_ROOT": historical.capture_control_root,
        "CONTRIBUTION_OBJECT_PREFIX": historical.object_prefix,
        "CONTRIBUTION_RECOMPOSITION_SIDECAR": historical.recomposition_sidecar,
        "CONTRIBUTION_QUALIFICATION_BUNDLE": historical.qualification_bundle,
        "CONTRIBUTION_SAE_CACHE_ROOT": historical.cache_root,
        "CONTRIBUTION_SAE_CACHE_MAX_BYTES": str(historical.cache_max_bytes),
        "CONTRIBUTION_SAE_CONTROL_ROOT": historical.sae_control_root,
        "CONTRIBUTION_SAE_CHECKPOINT_ROOT": historical.checkpoint_payload_root,
    }
    arguments = [
        "--action",
        "qualify",
        "--config",
        "/nfs/contracts/finalized-contribution-horizon.yaml",
        "--phase",
        historical.phase,
        "--condition",
        historical.condition,
        "--seed",
        "0",
        "--run-id",
        historical.run_id,
        "--attempt-id",
        historical.attempt_id,
        "--maximum-runtime-seconds",
        "600",
    ]
    selected = operator._settings(
        environment,
        operator._public_parser().parse_args(arguments),
    )
    assert (
        selected.config_path
        == "/nfs/contracts/finalized-contribution-horizon.yaml"
    )
    assert selected.config_path in " ".join(operator._init_command(selected))
    with pytest.raises(operator.OperatorError, match="absolute POSIX path"):
        operator._settings(
            environment,
            operator._public_parser().parse_args(
                [
                    *arguments[:3],
                    "relative/config.yaml",
                    *arguments[4:],
                ]
            ),
        )


def test_dry_run_is_read_only_and_declares_the_four_mapped_stages(
    operator,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _settings(operator, tmp_path)
    assert operator._dry_run(settings) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"].endswith(
        "configs/gemma4_contribution_v1.yaml"
    )
    assert payload["qualification_hard_deadline_seconds"] == 600
    assert payload["prefetch_hard_deadline_seconds"] == 600
    assert len(payload["fanout_controllers"]) == 4
    assert {
        tuple(item["stages"]) for item in payload["fanout_controllers"]
    } == {("prefetch", "qualify")}
    assert all(
        item["process_index"] == "<from-sealed-v4-bundle>"
        for item in payload["fanout_controllers"]
    )


def test_prefetch_failure_never_launches_accelerator_action(
    operator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(operator, tmp_path)
    plan_body = {
        "schema_version": operator.ATTEMPT_PLAN_SCHEMA_VERSION,
        "action": "qualify",
        "attempt_id": settings.attempt_id,
        "attempt_root": settings.attempt_root,
        "run_root": settings.run_root,
        "run_contract_sha256": "a" * 64,
        "config": settings.config_path,
        "config_file_sha256": "b" * 64,
        "admission_mode": operator.QUALIFIED_ADMISSION,
        "controller_to_jax_process_index": [
            {"controller_index": 0, "process_index": 2},
            {"controller_index": 1, "process_index": 0},
            {"controller_index": 2, "process_index": 3},
            {"controller_index": 3, "process_index": 1},
        ],
        "maximum_runtime_seconds": 600,
        "qualification_hard_deadline_seconds": 600,
        "prefetch_hard_deadline_seconds": 600,
        "terminal_policy": (
            "four_mapped_prefetches_then_four_concurrent_supervised_workers"
        ),
    }
    plan = {
        **plan_body,
        "attempt_plan_sha256": canonical_sha256(plan_body),
    }
    completed = subprocess.CompletedProcess(
        args=["gcloud"],
        returncode=0,
        stdout=(json.dumps(plan) + "\n").encode("utf-8"),
    )
    monkeypatch.setattr(operator.shutil, "which", lambda _name: "/bin/gcloud")
    monkeypatch.setattr(operator, "_verify_local_checkout", lambda _settings: None)
    monkeypatch.setattr(operator, "_run_capture", lambda *_args: completed)
    stages: list[str] = []

    def fanout(_commands, logs, *, maximum_wait_seconds, stage):
        del maximum_wait_seconds
        stages.append(stage)
        assert stage == "prefetch"
        statuses = [0, 1, 0, 0]
        mapping = [(0, 2), (1, 0), (2, 3), (3, 1)]
        for log, (controller, process), status in zip(
            logs,
            mapping,
            statuses,
            strict=True,
        ):
            log.write_text(
                f"terminal prefetch controller={controller} "
                f"process={process} status={status}\n",
                encoding="utf-8",
            )
        return [1, 1, 1, 1]

    monkeypatch.setattr(operator, "_fanout", fanout)
    with pytest.raises(operator.OperatorError, match="prefetches failed"):
        operator.run(settings)
    assert stages == ["prefetch"]


def test_nonzero_gcloud_transport_accepts_exact_bound_worker_markers(
    operator,
    tmp_path: Path,
) -> None:
    settings = _settings(operator, tmp_path, action="train")
    mapping = {0: 2, 1: 0, 2: 3, 3: 1}
    prefetch_logs = []
    action_logs = []
    for controller, process in mapping.items():
        prefetch = tmp_path / f"prefetch-{controller}.log"
        prefetch.write_text(
            f"transport diagnostic\nterminal prefetch "
            f"controller={controller} process={process} status=0\n",
            encoding="utf-8",
        )
        prefetch_logs.append(prefetch)
        action = tmp_path / f"action-{controller}.log"
        action.write_text(
            f"terminal action=train controller={controller} "
            f"process={process} status=0\n",
            encoding="utf-8",
        )
        action_logs.append(action)

    assert operator._validated_fanout_completion(
        settings=settings,
        mapping=mapping,
        stage="prefetch",
        returncodes=[1, 1, 1, 1],
        log_paths=prefetch_logs,
    ) == [0, 0, 0, 0]
    assert operator._validated_fanout_completion(
        settings=settings,
        mapping=mapping,
        stage="action",
        returncodes=[1, 1, 1, 1],
        log_paths=action_logs,
    ) == [0, 0, 0, 0]


def test_zero_gcloud_transport_still_requires_every_terminal_marker(
    operator,
    tmp_path: Path,
) -> None:
    settings = _settings(operator, tmp_path, action="train")
    mapping = {0: 2, 1: 0, 2: 3, 3: 1}
    logs = []
    for controller, process in mapping.items():
        log = tmp_path / f"prefetch-{controller}.log"
        log.write_text(
            (
                f"terminal prefetch controller={controller} "
                f"process={process} status=0\n"
                if controller != 3
                else "gcloud transport exited zero without remote marker\n"
            ),
            encoding="utf-8",
        )
        logs.append(log)

    with pytest.raises(operator.OperatorError, match="exactly one"):
        operator._validated_fanout_completion(
            settings=settings,
            mapping=mapping,
            stage="prefetch",
            returncodes=[0, 0, 0, 0],
            log_paths=logs,
        )


def test_worker_terminal_marker_is_exactly_bound_and_unique(
    operator,
    tmp_path: Path,
) -> None:
    settings = _settings(operator, tmp_path, action="train")
    log = tmp_path / "worker.log"
    log.write_text(
        "terminal prefetch controller=0 process=2 status=0\n"
        "terminal prefetch controller=0 process=2 status=0\n",
        encoding="utf-8",
    )
    with pytest.raises(operator.OperatorError, match="exactly one"):
        operator._terminal_worker_status(
            log,
            settings=settings,
            stage="prefetch",
            controller_index=0,
            process_index=2,
        )

    log.write_text(
        "terminal action=train controller=0 process=3 status=0\n",
        encoding="utf-8",
    )
    with pytest.raises(operator.OperatorError, match="does not match"):
        operator._terminal_worker_status(
            log,
            settings=settings,
            stage="action",
            controller_index=0,
            process_index=2,
        )


@pytest.mark.parametrize("status", ["", "-1", "00", "256", "0 trailing"])
def test_worker_terminal_marker_rejects_noncanonical_status(
    operator,
    tmp_path: Path,
    status: str,
) -> None:
    settings = _settings(operator, tmp_path, action="train")
    log = tmp_path / "worker.log"
    log.write_text(
        f"terminal action=train controller=0 process=2 status={status}\n",
        encoding="utf-8",
    )
    with pytest.raises(operator.OperatorError, match="worker status|range"):
        operator._terminal_worker_status(
            log,
            settings=settings,
            stage="action",
            controller_index=0,
            process_index=2,
        )


@pytest.mark.parametrize(
    ("action_statuses", "failure"),
    [
        ([0, 0, 0, 0], None),
        ([0, 7, 0, 0], "train workers failed"),
    ],
)
def test_run_uses_exact_remote_status_after_transport_teardown(
    operator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action_statuses: list[int],
    failure: str | None,
) -> None:
    settings = _settings(
        operator,
        tmp_path,
        action="train",
        maximum_runtime_seconds=600,
    )
    mapping = {0: 2, 1: 0, 2: 3, 3: 1}
    plan_body = {
        "schema_version": operator.ATTEMPT_PLAN_SCHEMA_VERSION,
        "action": settings.action,
        "attempt_id": settings.attempt_id,
        "attempt_root": settings.attempt_root,
        "run_root": settings.run_root,
        "run_contract_sha256": "a" * 64,
        "config": settings.config_path,
        "config_file_sha256": "b" * 64,
        "admission_mode": operator.DIRECT_BOUNDED_ADMISSION,
        "controller_to_jax_process_index": [
            {"controller_index": controller, "process_index": process}
            for controller, process in mapping.items()
        ],
        "maximum_runtime_seconds": 600,
        "qualification_hard_deadline_seconds": 600,
        "prefetch_hard_deadline_seconds": 600,
        "terminal_policy": (
            "four_mapped_prefetches_then_four_concurrent_supervised_workers"
        ),
    }
    plan = {
        **plan_body,
        "attempt_plan_sha256": canonical_sha256(plan_body),
    }
    completed = subprocess.CompletedProcess(
        args=["gcloud"],
        returncode=0,
        stdout=(json.dumps(plan) + "\n").encode("utf-8"),
    )
    monkeypatch.setattr(operator.shutil, "which", lambda _name: "/bin/gcloud")
    monkeypatch.setattr(operator, "_verify_local_checkout", lambda _settings: None)
    monkeypatch.setattr(operator, "_run_capture", lambda *_args: completed)
    stages = []

    def fanout(_commands, logs, *, maximum_wait_seconds, stage):
        del maximum_wait_seconds
        stages.append(stage)
        marker_stage = "prefetch" if stage == "prefetch" else "action=train"
        statuses = [0, 0, 0, 0] if stage == "prefetch" else action_statuses
        for log, (controller, process), status in zip(
            logs,
            mapping.items(),
            statuses,
            strict=True,
        ):
            log.write_text(
                f"terminal {marker_stage} controller={controller} "
                f"process={process} status={status}\n",
                encoding="utf-8",
            )
        return [1, 1, 1, 1]

    monkeypatch.setattr(operator, "_fanout", fanout)
    if failure is None:
        assert operator.run(settings) == 0
    else:
        with pytest.raises(operator.OperatorError, match=failure):
            operator.run(settings)
    assert stages == ["prefetch", "train"]
