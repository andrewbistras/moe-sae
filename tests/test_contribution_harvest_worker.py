from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import moe_sae.contribution_harvest_worker as worker
from moe_sae.artifacts import CaptureMetadata, StagedDatasetProvenance
from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    build_recomposition_sidecar,
    canonical_json_bytes,
    canonical_sha256,
    float32_to_bf16_bits,
)
from moe_sae.contribution_config import (
    MAXTEXT_FORK_COMMIT,
    MAXTEXT_ROUTED_MOE_PATCHED_SHA256,
    load_contribution_config,
)
from moe_sae.contribution_harvest import ContributionHarvestResult
from moe_sae.contribution_maxtext import (
    PreparedContributionExecutable,
    exact_performance_identity,
    hardware_manifest,
    observed_performance_launch_manifest,
)
from moe_sae.contribution_performance import (
    QualificationEvidence,
    authorize_accelerator_run,
    build_qualification_receipt,
)
from moe_sae.contribution_recomposition import SUPPORTED_ACCUMULATION_ORDER
from moe_sae.contribution_storage import (
    ContributionAdoptionResult,
    ContributionShardWriter,
)
from moe_sae.contribution_supervisor import build_execution_record
from moe_sae.object_store import LocalObjectStore
from moe_sae.runtime import HARVEST_MESHES

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "gemma4_contribution_v1.yaml"
)
CHECKPOINT_SHA = "1" * 64
DATASET_SHA = "2" * 64
ENVIRONMENT_SHA = "3" * 64
SOURCE_MANIFEST_SHA = "4" * 64
SIDE_FINGERPRINT = "5" * 64
OBJECT_PREFIX = "gs://experiment-2-test/contributions"


class FakeDevice:
    def __init__(self, device_id: int) -> None:
        self.id = device_id
        self.process_index = device_id // 4
        self.platform = "tpu"
        self.device_kind = "TPU v4"
        self.slice_index = 0


class FakeSource:
    def __init__(self, root: Path, manifest: dict) -> None:
        self.root = root
        self.manifest = manifest
        self.manifest_sha256 = manifest["manifest_sha256"]


class FakeWriter:
    def __init__(self, host_id: str) -> None:
        self.host_id = host_id
        self.finalize_calls = 0

    def finalize_host(self) -> dict:
        self.finalize_calls += 1
        return {"index_sha256": "6" * 64}


def _sidecar() -> dict:
    config = load_contribution_config(CONFIG_PATH)
    scale = float32_to_bf16_bits(np.ones(HIDDEN_SIZE, dtype=np.float32))
    return build_recomposition_sidecar(
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_fork_commit=MAXTEXT_FORK_COMMIT,
        source_fingerprints={
            "authorization_contract": "7" * 64,
            "environment": ENVIRONMENT_SHA,
            "experiment_config": config.config_hash,
            "maxtext_routed_moe": MAXTEXT_ROUTED_MOE_PATCHED_SHA256,
            "source_graph_contract": SIDE_FINGERPRINT,
        },
        routed_rmsnorm_scale=scale,
        shared_rmsnorm_scale=scale,
        outer_ffn_rmsnorm_scale=scale,
        layer_scalar=float32_to_bf16_bits(np.asarray([1.0], dtype=np.float32)),
        accumulation_order=SUPPORTED_ACCUMULATION_ORDER,
    )


def _source_manifest(config) -> dict:
    metadata = CaptureMetadata(
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_commit=config.provenance.maxtext_base_commit,
        dataset_fingerprint=DATASET_SHA,
        split="train",
        config_sha256="8" * 64,
        environment_sha256="9" * 64,
        conversion_attestation_sha256="a" * 64,
    ).to_dict()
    lane = config.phases[-1].train_tokens // 4
    return {
        "manifest_sha256": SOURCE_MANIFEST_SHA,
        "host_count": 4,
        "total_tokens": config.phases[-1].train_tokens,
        "metadata": metadata,
        "shards": [
            {
                "host_id": f"p{process:05d}",
                "range_start": process * lane,
                "range_end": (process + 1) * lane,
            }
            for process in range(4)
        ],
    }


def _receipt(identity) -> dict:
    return build_qualification_receipt(
        QualificationEvidence(
            identity=identity,
            source_evidence_sha256="b" * 64,
            useful_flops_ledger_sha256="c" * 64,
            qualification_wall_seconds=2.0,
            measured_device_seconds=1.0,
            persisted_window_seconds=1.0,
            useful_flops=1_000_000,
            peak_flops_per_second=1_000_000,
            warmup_steps=5,
            synchronized_measurement_steps=200,
            measured_graph_compile_count=1,
            auxiliary_validation_compile_count=0,
            total_compile_count=1,
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
            instrumentation_disabled_output_sha256="d" * 64,
            instrumentation_enabled_output_sha256="d" * 64,
            writer_disabled_output_tokens=100,
            writer_disabled_device_seconds=1.0,
        )
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _raw_batch() -> dict[str, np.ndarray]:
    shape = (16, 1_024)
    return {
        "inputs": np.full(shape, 3, dtype=np.int32),
        "inputs_position": np.broadcast_to(
            np.arange(1_024, dtype=np.int32),
            shape,
        ).copy(),
        "inputs_segmentation": np.ones(shape, dtype=np.int32),
    }


def _case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase_name: str = "phase1_smoke",
    prepared_graph: dict | None = None,
    receipt_graph: dict | None = None,
    controller_index: int = 0,
    observed_process_index: int = 0,
) -> SimpleNamespace:
    config = load_contribution_config(CONFIG_PATH)
    source_manifest = _source_manifest(config)
    source = FakeSource(tmp_path / "source", source_manifest)
    monkeypatch.setattr(worker, "load_manifest", lambda _root: source_manifest)

    sidecar = _sidecar()
    sidecar_path = tmp_path / "sidecar.json"
    _write_json(sidecar_path, sidecar)
    staged = StagedDatasetProvenance(
        fingerprint_sha256=DATASET_SHA,
        split="train",
        files=("/immutable/train-000.array_record",),
        scientific_tokens=config.phases[-1].train_tokens,
        manifest={"files": []},
    )
    mesh = HARVEST_MESHES[1]
    devices = tuple(FakeDevice(index) for index in range(16))
    hardware = hardware_manifest(
        devices=devices,
        process_count=4,
        mesh=mesh,
    )
    performance_launch = {
        "schema_version": "test-performance-launch-v1",
        "config_sha256": config.config_hash,
    }
    qualified_graph = receipt_graph or {
        "schema_version": "test-graph-v1",
        "stablehlo_sha256": "e" * 64,
    }
    actual_graph = prepared_graph or qualified_graph
    identity = exact_performance_identity(
        graph=qualified_graph,
        hardware=hardware,
        launch_config=performance_launch,
    )
    receipt = _receipt(identity)
    receipt_path = tmp_path / "qualification-receipt.json"
    _write_json(receipt_path, receipt)
    maximum_runtime = 3_600
    authorization = authorize_accelerator_run(
        identity=identity,
        maximum_runtime_seconds=maximum_runtime,
        purpose="workload",
        qualification_receipt=receipt,
    )
    execution_path = tmp_path / "execution.json"
    execution = build_execution_record(
        authorization=authorization,
        command=("fake-harvest-child",),
        environment={},
        working_directory=tmp_path,
        log_path=tmp_path / "stdout.log",
        execution_record_path=execution_path,
        result_path=tmp_path / "result.json",
        termination_grace_seconds=10.0,
        qualification_receipt=receipt,
    )
    _write_json(execution_path, execution)

    invocation = worker.HarvestInvocation(
        config_path=CONFIG_PATH,
        phase=phase_name,
        split="train",
        checkpoint_path="gs://model/checkpoint",
        checkpoint_sha256=CHECKPOINT_SHA,
        grain_files="/immutable/*.array_record",
        staged_manifest="/immutable/manifest.json",
        source_v1_root=tmp_path / "source",
        source_cache_root=tmp_path / "source-cache",
        source_cache_max_bytes=1 << 30,
        control_root=tmp_path / f"{phase_name}-control",
        prior_control_root=None,
        prior_phase_evidence_path=None,
        local_buffer_root=tmp_path / "local-buffer",
        object_prefix=OBJECT_PREFIX,
        recomposition_sidecar_path=sidecar_path,
        qualification_receipt_path=receipt_path,
        repo_root=tmp_path,
        jax_cache_dir=tmp_path / "jax-cache",
        maxtext_root=None,
        mesh=mesh,
        per_device_sequence_batch=8,
        controller_index=controller_index,
        maximum_runtime_seconds=maximum_runtime,
        child_evidence_path=tmp_path / "child-evidence.json",
        execution_record_path=execution_path,
    )
    runtime = SimpleNamespace(
        model=object(),
        mesh=object(),
        config=SimpleNamespace(dataset_type="grain"),
        train_iterator=iter((_raw_batch(),)),
        eval_iterator=iter((_raw_batch(),)),
    )

    def executable(*_args):
        return None

    prepared = PreparedContributionExecutable(
        executable=executable,
        graph=actual_graph,
        compile_seconds=1.0,
    )
    writer_state: dict[str, object] = {"created": False}

    def create_writer(_invocation, host_id, _metadata, _store, _max_tokens):
        writer_state["created"] = True
        value = FakeWriter(host_id)
        writer_state["writer"] = value
        return value

    def fake_harvest(
        _batches,
        adapter_step,
        *,
        range_plan,
        phase_total_tokens,
        process_count,
        process_index,
        writer,
        **_kwargs,
    ):
        assert adapter_step.executable is executable
        assert writer_state["created"] is True
        host_id = range_plan.for_process(
            phase_total_tokens,
            process_index,
        )[0].host_id
        return ContributionHarvestResult(
            schema_version="gemma4-contribution-harvest-host-v1",
            range_plan_sha256=range_plan.plan_sha256,
            source_v1_manifest_sha256=range_plan.source_manifest_sha256,
            split="train",
            phase_total_tokens=phase_total_tokens,
            process_count=process_count,
            process_index=process_index,
            host_id=host_id,
            host_tokens=phase_total_tokens // process_count,
            new_tokens=phase_total_tokens // process_count,
            resumed_tokens=0,
            batches_scanned=1,
            batches_executed=1,
            executed_scientific_tokens=16_384,
            wall_seconds=1.0,
            device_seconds=0.8,
            input_wait_seconds=0.01,
            max_writer_backlog_batches=0,
        )

    dependencies = worker.HarvestWorkerDependencies(
        initialize_distributed=lambda: None,
        process_context=lambda: (
            devices,
            4,
            observed_process_index,
            16,
        ),
        verify_maxtext=lambda _root: {
            "mode": "locked_pep610_install",
            "commit": MAXTEXT_FORK_COMMIT,
            "routed_moe_sha256": MAXTEXT_ROUTED_MOE_PATCHED_SHA256,
        },
        verify_checkpoint=lambda _path: CHECKPOINT_SHA,
        create_object_store=lambda: LocalObjectStore(tmp_path / "objects"),
        verify_staged=lambda *_args: staged,
        create_source_dataset=lambda *_args: source,
        create_runtime=lambda _argv, _files: runtime,
        close_runtime=lambda _runtime: None,
        mesh_context=lambda _runtime: nullcontext(),
        create_adapter=lambda _model, _root: object(),
        prepare=lambda _adapter, _mesh, _batch, _clock: prepared,
        effective_runtime=lambda *_args: {
            "schema_version": "test-effective-runtime-v1"
        },
        performance_launch=lambda *_args: performance_launch,
        verify_sidecar=lambda _adapter, _checkpoint, value: value,
        environment=lambda _root: {"sha256": ENVIRONMENT_SHA},
        create_writer=create_writer,
        adopt_phase=lambda *_args: pytest.fail("unexpected phase adoption"),
        harvest=fake_harvest,
        batch_collective=lambda complete, execute: (complete, execute),
        clock=time.perf_counter,
    )
    return SimpleNamespace(
        config=config,
        source=source,
        source_manifest=source_manifest,
        staged=staged,
        sidecar=sidecar,
        invocation=invocation,
        dependencies=dependencies,
        writer_state=writer_state,
        receipt=receipt,
        identity=identity,
        performance_launch=performance_launch,
    )


def test_long_harvest_reuses_qualified_prepared_executable_and_seals_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)

    evidence = worker.run_harvest_child(
        case.invocation,
        dependencies=case.dependencies,
    )

    checked = worker.validate_child_evidence(evidence)
    assert checked["status"] == "complete"
    assert checked["qualification_receipt_sha256"] == case.receipt["receipt_sha256"]
    assert checked["observed_identity_sha256"] == case.identity.identity_sha256
    assert checked["authorization"]["basis"] == "qualified_exact_identity"
    assert checked["phase_total_tokens"] == 100_000
    assert case.writer_state["created"] is True
    assert case.invocation.child_evidence_path.read_bytes() == canonical_json_bytes(
        evidence
    )


def test_controller_ordinal_does_not_override_observed_jax_process_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Live v4-32 observation: gcloud controller 1 was JAX process 2.
    case = _case(
        tmp_path,
        monkeypatch,
        controller_index=1,
        observed_process_index=2,
    )

    evidence = worker.run_harvest_child(
        case.invocation,
        dependencies=case.dependencies,
    )

    checked = worker.validate_child_evidence(evidence)
    assert checked["controller_index"] == 1
    assert checked["process_index"] == 2
    assert checked["host_id"] == "p00002"
    assert checked["harvest"]["process_index"] == 2
    assert case.writer_state["writer"].host_id == "p00002"
    command = worker._child_command(case.invocation)
    assert command[command.index("--controller-index") + 1] == "1"
    assert "--process-index" not in command


def test_identity_mismatch_fails_before_writer_or_harvest_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(
        tmp_path,
        monkeypatch,
        prepared_graph={
            "schema_version": "test-graph-v1",
            "stablehlo_sha256": "f" * 64,
        },
        receipt_graph={
            "schema_version": "test-graph-v1",
            "stablehlo_sha256": "e" * 64,
        },
    )

    with pytest.raises(ValueError, match="exact graph/hardware/configuration"):
        worker.run_harvest_child(
            case.invocation,
            dependencies=case.dependencies,
        )

    assert case.writer_state["created"] is False
    assert not case.invocation.child_evidence_path.exists()


def test_phase2_adopts_exact_phase1_parent_before_appending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(
        tmp_path,
        monkeypatch,
        phase_name="phase2_mechanism_pilot",
    )
    phase1 = case.config.phase("phase1_smoke")
    range_plan = worker.build_contribution_range_plan(
        case.source_manifest,
        worker._phase_totals(case.config, "train"),
        process_count=4,
        max_shard_tokens=case.config.runtime.max_shard_tokens,
    )
    metadata = worker._metadata(
        invocation=case.invocation,
        config=case.config,
        source=case.source,
        staged=case.staged,
        sidecar=case.sidecar,
        environment_sha256=ENVIRONMENT_SHA,
    )
    prior_manifest_sha = "0" * 64
    prior_manifest = {
        "manifest_sha256": prior_manifest_sha,
        "total_tokens": phase1.train_tokens,
        "metadata": metadata,
    }
    monkeypatch.setattr(
        worker,
        "load_contribution_manifest",
        lambda _root: prior_manifest,
    )
    phase_evidence_unsigned = {
        "schema_version": worker.PHASE_EVIDENCE_SCHEMA_VERSION,
        "status": "sealed",
        "protocol_id": case.config.protocol_id,
        "config_sha256": case.config.config_hash,
        "phase": phase1.name,
        "phase_ordinal": phase1.ordinal,
        "split": "train",
        "phase_total_tokens": phase1.train_tokens,
        "process_count": 4,
        "source_v1_manifest_sha256": SOURCE_MANIFEST_SHA,
        "range_plan_sha256": range_plan.plan_sha256,
        "metadata_sha256": canonical_sha256(metadata),
        "recomposition_sidecar_sha256": case.sidecar["sidecar_sha256"],
        "adopted_parent_manifest_sha256": None,
        "parent_phase_evidence_sha256": None,
        "contribution_manifest_sha256": prior_manifest_sha,
        "qualification_receipt_sha256": case.receipt["receipt_sha256"],
        "observed_identity_sha256": case.identity.identity_sha256,
        # Live mapping inverted to process order:
        # controller->process 0->0, 1->2, 2->3, 3->1.
        "controller_index_by_process": [0, 3, 1, 2],
        "host_evidence_sha256_by_process": [f"{index + 10:064x}" for index in range(4)],
    }
    phase_evidence = worker._self_hash(
        phase_evidence_unsigned,
        hash_field="phase_evidence_sha256",
    )
    worker.validate_phase_evidence(phase_evidence)
    for invalid_controllers in (
        [0, 3, 1, 1],
        [False, 1, 2, 3],
        ["0", 1, 2, 3],
    ):
        invalid_mapping = worker._self_hash(
            {
                **phase_evidence_unsigned,
                "controller_index_by_process": invalid_controllers,
            },
            hash_field="phase_evidence_sha256",
        )
        with pytest.raises(
            worker.ContributionHarvestWorkerError,
            match="controller mapping",
        ):
            worker.validate_phase_evidence(invalid_mapping)
    prior_evidence_path = tmp_path / "phase1-evidence.json"
    _write_json(prior_evidence_path, phase_evidence)
    prior_root = tmp_path / "phase1-control"
    invocation = replace(
        case.invocation,
        prior_control_root=prior_root,
        prior_phase_evidence_path=prior_evidence_path,
    )
    adopted: list[str] = []

    def adopt(writer, selected_root, _sidecar):
        adopted.append(str(selected_root))
        return ContributionAdoptionResult(
            prior_manifest_sha256=prior_manifest_sha,
            host_id=writer.host_id,
            adopted_shards=7,
            adopted_tokens=phase1.train_tokens // 4,
        )

    dependencies = replace(case.dependencies, adopt_phase=adopt)

    evidence = worker.run_harvest_child(
        invocation,
        dependencies=dependencies,
    )

    assert adopted == [str(prior_root)]
    assert evidence["phase_total_tokens"] == 2_000_000
    assert evidence["adopted_parent"] == {
        "prior_manifest_sha256": prior_manifest_sha,
        "host_id": "p00000",
        "adopted_shards": 7,
        "adopted_tokens": 25_000,
    }
    assert (
        evidence["parent_phase_evidence_sha256"]
        == phase_evidence["phase_evidence_sha256"]
    )


@pytest.mark.parametrize(
    ("controller_indices", "process_indices"),
    (
        ([0, 0, 2, 3], [0, 1, 2, 3]),
        ([0, 1, 2, 3], [0, 1, 1, 3]),
    ),
)
def test_publication_requires_independent_controller_and_process_permutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controller_indices: list[int],
    process_indices: list[int],
) -> None:
    records = [
        {
            "status": "complete",
            "controller_index": controller_index,
            "process_index": process_index,
        }
        for controller_index, process_index in zip(
            controller_indices,
            process_indices,
            strict=True,
        )
    ]
    monkeypatch.setattr(
        worker,
        "validate_supervised_evidence",
        lambda value: value,
    )
    with pytest.raises(
        worker.ContributionHarvestWorkerError,
        match="independent exact controller-index and JAX-process-index",
    ):
        worker.publish_harvest_phase(
            config_path=CONFIG_PATH,
            phase_name="phase1_smoke",
            split="train",
            control_root=tmp_path / "control",
            source_v1_dataset=object(),
            object_prefix=OBJECT_PREFIX,
            object_store=LocalObjectStore(tmp_path / "objects"),
            recomposition_sidecar={},
            supervised_host_evidence=records,
            phase_evidence_path=tmp_path / "phase-evidence.json",
        )


def test_production_writer_factory_is_exact_and_worker_has_no_stale_fork(
    tmp_path: Path,
) -> None:
    config = load_contribution_config(CONFIG_PATH)
    sidecar = _sidecar()
    metadata = worker.ContributionMetadata(
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_fork_commit=MAXTEXT_FORK_COMMIT,
        dataset_fingerprint=DATASET_SHA,
        split="train",
        config_sha256=config.config_hash,
        environment_sha256=ENVIRONMENT_SHA,
        source_v1_manifest_sha256=SOURCE_MANIFEST_SHA,
        recomposition_sidecar_sha256=sidecar["sidecar_sha256"],
    ).to_dict()
    invocation = worker.HarvestInvocation(
        config_path=CONFIG_PATH,
        phase="phase1_smoke",
        split="train",
        checkpoint_path="unused",
        checkpoint_sha256=CHECKPOINT_SHA,
        grain_files="unused",
        staged_manifest="unused",
        source_v1_root=tmp_path,
        source_cache_root=tmp_path / "cache",
        source_cache_max_bytes=1,
        control_root=tmp_path / "control",
        prior_control_root=None,
        prior_phase_evidence_path=None,
        local_buffer_root=tmp_path / "local",
        object_prefix=OBJECT_PREFIX,
        recomposition_sidecar_path=tmp_path / "sidecar",
        qualification_receipt_path=tmp_path / "receipt",
        repo_root=tmp_path,
        jax_cache_dir=tmp_path / "jax-cache",
        maxtext_root=None,
        mesh=HARVEST_MESHES[1],
        per_device_sequence_batch=8,
        controller_index=0,
        maximum_runtime_seconds=3_600,
        child_evidence_path=tmp_path / "child",
        execution_record_path=tmp_path / "execution",
    )
    writer_value = worker._default_create_writer(
        invocation,
        "p00000",
        metadata,
        LocalObjectStore(tmp_path / "objects"),
        config.runtime.max_shard_tokens,
    )

    assert type(writer_value) is ContributionShardWriter
    source_text = Path(worker.__file__).read_text(encoding="utf-8")
    assert "d2f8a3e3d8392f9aef3d3ee6c7262deaf22ee294" not in source_text
    assert MAXTEXT_FORK_COMMIT == "73bafe53569eb17dc18542f2568dd591a8db9e28"


def test_qualifier_and_real_harvest_share_one_performance_launch_identity(
    tmp_path: Path,
) -> None:
    config = load_contribution_config(CONFIG_PATH)
    mesh = HARVEST_MESHES[1]
    common = {
        "use_multimodal": False,
        "scan_layers": False,
        "pure_nnx_decoder": True,
        "enable_dropout": False,
        "use_ragged_sort": True,
        "sparse_matmul": True,
        "use_ring_of_experts": False,
        "float32_weight_sum": True,
        "matmul_precision": "default",
        "max_target_length": 1_024,
        "per_device_batch_size": 8,
        "ici_expert_parallelism": 8,
        "ici_fsdp_parallelism": 2,
        "ici_tensor_parallelism": 1,
    }
    qualifier_runtime = SimpleNamespace(
        **common,
        dataset_type="synthetic",
        jax_cache_dir="/qualifier/cache",
    )
    harvest_runtime = SimpleNamespace(
        **common,
        dataset_type="grain",
        grain_train_files="gs://immutable/train/*",
        jax_cache_dir="/harvest/cache",
    )
    runtime_mesh = SimpleNamespace(
        axis_names=("expert", "fsdp"),
        devices=np.empty((8, 2), dtype=object),
    )
    environment = {"sha256": ENVIRONMENT_SHA}
    provenance = {
        "mode": "locked_pep610_install",
        "commit": MAXTEXT_FORK_COMMIT,
        "routed_moe_sha256": MAXTEXT_ROUTED_MOE_PATCHED_SHA256,
    }

    qualified = observed_performance_launch_manifest(
        config,
        checkpoint_sha256=CHECKPOINT_SHA,
        mesh=mesh,
        per_device_sequence_batch=8,
        writer_backlog_limit_batches=2,
        jax_cache_dir=tmp_path / "qualifier-cache",
        environment=environment,
        maxtext_provenance=provenance,
        runtime_config=qualifier_runtime,
        runtime_mesh=runtime_mesh,
    )
    harvested = observed_performance_launch_manifest(
        config,
        checkpoint_sha256=CHECKPOINT_SHA,
        mesh=mesh,
        per_device_sequence_batch=8,
        writer_backlog_limit_batches=2,
        jax_cache_dir=tmp_path / "harvest-cache",
        environment=environment,
        maxtext_provenance=provenance,
        runtime_config=harvest_runtime,
        runtime_mesh=runtime_mesh,
    )

    assert qualified == harvested
    assert canonical_sha256(qualified) == canonical_sha256(harvested)
    encoded = canonical_json_bytes(qualified)
    assert b"synthetic" not in encoded
    assert b"grain_train_files" not in encoded
    assert str(tmp_path).encode() not in encoded
