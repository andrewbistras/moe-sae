from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    build_recomposition_sidecar,
    canonical_sha256,
    validate_recomposition_sidecar,
)
from moe_sae.contribution_config import (
    MAXTEXT_FORK_COMMIT,
    MAXTEXT_ROUTED_MOE_PATCHED_SHA256,
    load_contribution_config,
)
from moe_sae.contribution_maxtext import (
    InstrumentationObservation,
    PreparedContributionExecutable,
    QualificationDiagnostics,
    QualificationHostObservation,
    QualificationPlan,
    run_qualification_host,
)
from moe_sae.contribution_performance import (
    ContributionPerformanceError,
    ExactPerformanceIdentity,
    QualificationEvidence,
    authorize_accelerator_run,
    build_qualification_receipt,
)
from moe_sae.contribution_qualification_worker import (
    QUALIFICATION_WRITER_SCHEMA_VERSION,
    ChildInvocation,
    ContributionQualificationWorkerError,
    LocalQualificationPersist,
    QualificationWorkerDependencies,
    _child_record,
    _contract_identity,
    _observation_from_dict,
    _supervised_host_record,
    _validate_effective_runtime,
    aggregate_supervised_host_evidence,
    authorize_extended_workload_after_lowering,
    build_authorization_contract,
    launch_qualification_host,
    run_qualification_child,
    validate_authorization_contract,
    validate_child_evidence,
    validate_supervised_host_evidence,
)
from moe_sae.contribution_recomposition import SUPPORTED_ACCUMULATION_ORDER
from moe_sae.contribution_supervisor import (
    SUPERVISION_RESULT_SCHEMA_VERSION,
    build_execution_record,
)
from moe_sae.harvest import GrainBatch
from moe_sae.runtime import MeshCandidate

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "gemma4_contribution_v1.yaml"
MESH = MeshCandidate("ep16-fsdp1-tp1", 16, 1, 1)
CHECKPOINT_SHA = "c" * 64
OUTPUT_SHA = "d" * 64


def _sidecar(source_fingerprints: dict[str, str] | None = None) -> dict:
    ones = np.full((HIDDEN_SIZE,), 0x3F80, dtype=np.uint16)
    return build_recomposition_sidecar(
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_fork_commit=MAXTEXT_FORK_COMMIT,
        source_fingerprints=source_fingerprints or {"test": "e" * 64},
        routed_rmsnorm_scale=ones,
        shared_rmsnorm_scale=ones,
        outer_ffn_rmsnorm_scale=ones,
        layer_scalar=np.asarray([0x3F80], dtype=np.uint16),
        accumulation_order=SUPPORTED_ACCUMULATION_ORDER,
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _contract(tmp_path: Path, *, plan: QualificationPlan | None = None) -> dict:
    return build_authorization_contract(
        load_contribution_config(CONFIG_PATH),
        checkpoint_sha256=CHECKPOINT_SHA,
        mesh=MESH,
        per_device_sequence_batch=1,
        jax_cache_dir=tmp_path / "cache",
        repo_root=ROOT,
        plan=plan or QualificationPlan(),
    )


def _execution(
    tmp_path: Path,
    contract: dict,
    *,
    maximum_seconds: int = 600,
) -> tuple[dict, Path]:
    identity = _contract_identity(contract)
    authorization = authorize_accelerator_run(
        identity=identity,
        maximum_runtime_seconds=maximum_seconds,
        purpose="qualification",
    )
    execution_path = tmp_path / "execution.json"
    execution = build_execution_record(
        authorization=authorization,
        command=("python", "-m", "fake"),
        environment={"PATH": "/usr/bin"},
        working_directory=ROOT,
        log_path=tmp_path / "run.log",
        execution_record_path=execution_path,
        result_path=tmp_path / "result.json",
        termination_grace_seconds=min(10.0, maximum_seconds / 2),
    )
    _write_json(execution_path, execution)
    return execution, execution_path


def _writer_summary(
    *,
    process_index: int,
    tokens: int = 40_960,
) -> dict:
    unsigned = {
        "schema_version": QUALIFICATION_WRITER_SCHEMA_VERSION,
        "process_count": 4,
        "process_index": process_index,
        "backlog_limit": 2,
        "max_observed_backlog": 1,
        "queue_drained": True,
        "files": [],
        "tokens": tokens,
        "bytes": tokens * 10,
    }
    return {
        **unsigned,
        "writer_evidence_sha256": canonical_sha256(unsigned),
    }


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, value: float) -> None:
        self.value += value


class _Output:
    rows = 16_384

    def block_until_ready(self) -> _Output:
        return self


class _Persist:
    backlog_limit = 2
    max_observed_backlog = 1
    queue_drained = False

    def __init__(self, clock: _Clock, process_index: int) -> None:
        self.clock = clock
        self.process_index = process_index
        self.rows = 0

    def __call__(self, output: _Output) -> int:
        assert output.rows == 16_384
        self.clock.advance(0.001)
        self.rows += output.rows // 4
        return output.rows // 4

    def flush(self) -> None:
        self.clock.advance(0.001)
        self.queue_drained = True

    def close(self) -> None:
        self.flush()

    def summary(self) -> dict:
        assert self.queue_drained
        return _writer_summary(
            process_index=self.process_index,
            tokens=self.rows,
        )


class _Device:
    def __init__(self, ordinal: int) -> None:
        self.id = ordinal % 4
        self.process_index = ordinal // 4
        self.platform = "tpu"
        self.device_kind = "TPU v4"
        self.slice_index = 0


def _runtime() -> SimpleNamespace:
    effective = {
        "model_name": "gemma4-26b",
        "load_parameters_path": "/checkpoint",
        "use_multimodal": False,
        "scan_layers": False,
        "pure_nnx": True,
        "enable_nnx": True,
        "pure_nnx_decoder": True,
        "enable_dropout": False,
        "load_balance_loss_weight": 0.0,
        "use_ragged_sort": True,
        "ragged_buffer_factor": -1.0,
        "sparse_matmul": True,
        "use_ring_of_experts": False,
        "float32_weight_sum": True,
        "matmul_precision": "default",
        "attention": "autoselected",
        "dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "max_target_length": 1_024,
        "per_device_batch_size": 1,
        "eval_per_device_batch_size": 1,
        "ici_expert_parallelism": 16,
        "ici_fsdp_parallelism": 1,
        "ici_tensor_parallelism": 1,
        "dataset_type": "synthetic",
        "jax_cache_dir": "/cache",
    }
    mesh = SimpleNamespace(
        axis_names=("expert", "fsdp", "tensor"),
        devices=np.empty((16, 1, 1), dtype=object),
    )
    batch = {
        "inputs": np.ones((1, 1_024), dtype=np.int32),
        "inputs_position": np.broadcast_to(
            np.arange(1_024, dtype=np.int32),
            (1, 1_024),
        ),
        "inputs_segmentation": np.ones((1, 1_024), dtype=np.int32),
    }
    return SimpleNamespace(
        config=SimpleNamespace(**effective),
        mesh=mesh,
        model=object(),
        train_iterator=iter((batch,)),
        eval_iterator=None,
    )


def _fake_dependencies(
    clock: _Clock,
    events: list[str],
    *,
    process_index: int = 0,
) -> QualificationWorkerDependencies:
    def initialize() -> None:
        events.append("initialize")

    def process_context():
        events.append("process_context")
        return (
            tuple(_Device(index) for index in range(16)),
            4,
            process_index,
            16,
        )

    def verify_maxtext(_path: Path | None) -> dict[str, str]:
        events.append("verify_maxtext")
        return {
            "mode": (
                "locked_pep610_install" if _path is None else "explicit_clean_checkout"
            ),
            "commit": MAXTEXT_FORK_COMMIT,
            "routed_moe_sha256": MAXTEXT_ROUTED_MOE_PATCHED_SHA256,
        }

    def verify_checkpoint(_path: str) -> str:
        events.append("verify_checkpoint")
        return CHECKPOINT_SHA

    def create_runtime(_argv):
        events.append("create_runtime")
        return _runtime()

    def close_runtime(_runtime) -> None:
        events.append("close_runtime")

    def mesh_context(_runtime):
        class _Context:
            def __enter__(self):
                events.append("mesh_enter")

            def __exit__(self, *_args):
                events.append("mesh_exit")

        return _Context()

    def create_adapter(_model, _root):
        events.append("create_adapter")
        return object()

    def extract_sidecar(_adapter, _checkpoint, source_fingerprints):
        events.append("extract_sidecar")
        return _sidecar(dict(source_fingerprints))

    def prepare(_adapter, _mesh, _batch, _clock):
        events.append("prepare")

        def execute(*_args):
            clock.advance(0.05)
            return _Output()

        return PreparedContributionExecutable(
            executable=execute,
            graph={
                "schema_version": "graph-test-v1",
                "stablehlo_sha256": "a" * 64,
            },
            compile_seconds=0.2,
        )

    def probe(_adapter, _batch, _count, _index):
        events.append("build_probe")
        return lambda: InstrumentationObservation(
            disabled_output_sha256=OUTPUT_SHA,
            enabled_output_sha256=OUTPUT_SHA,
            equal=True,
        )

    def persist(_root, _count, index, _sequence):
        events.append("create_persist")
        return _Persist(clock, index)

    def environment(_root):
        events.append("environment")
        return {"runtime": "fake", "sha256": "e" * 64}

    return QualificationWorkerDependencies(
        initialize_distributed=initialize,
        process_context=process_context,
        verify_maxtext=verify_maxtext,
        verify_checkpoint=verify_checkpoint,
        create_runtime=create_runtime,
        close_runtime=close_runtime,
        mesh_context=mesh_context,
        create_adapter=create_adapter,
        extract_sidecar=extract_sidecar,
        prepare=prepare,
        instrumentation_probe=probe,
        create_persist=persist,
        environment=environment,
        clock=clock,
    )


def _invocation_and_records(
    tmp_path: Path,
    *,
    controller_index: int = 0,
    plan: QualificationPlan | None = None,
) -> tuple[ChildInvocation, dict, dict]:
    selected_plan = plan or QualificationPlan()
    contract = _contract(tmp_path, plan=selected_plan)
    execution, execution_path = _execution(
        tmp_path,
        contract,
        maximum_seconds=selected_plan.maximum_seconds,
    )
    contract_path = tmp_path / "contract.json"
    _write_json(contract_path, contract)
    invocation = ChildInvocation(
        config_path=CONFIG_PATH,
        checkpoint_path="/checkpoint",
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_root=None,
        repo_root=ROOT,
        jax_cache_dir=tmp_path / "cache",
        scratch_root=tmp_path / "scratch",
        contract_path=contract_path,
        execution_record_path=execution_path,
        child_evidence_path=tmp_path / "child.json",
        recomposition_sidecar_path=tmp_path / "recomposition-sidecar.json",
        mesh=MESH,
        per_device_sequence_batch=1,
        controller_index=controller_index,
        plan=selected_plan,
    )
    return invocation, contract, execution


def test_authorization_contract_is_self_hashed_and_source_bound(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    checked = validate_authorization_contract(contract)
    identity = _contract_identity(checked)
    assert (
        checked["identity_semantics"]
        == "prelaunch-source-hardware-config-contract-not-observed-stablehlo"
    )
    assert (
        checked["authorization_contract_identity"]["identity_sha256"]
        == identity.identity_sha256
    )
    assert checked["graph_contract"]["maxtext_fork_commit"] == MAXTEXT_FORK_COMMIT
    assert {item["path"] for item in checked["graph_contract"]["source_files"]} >= {
        "src/moe_sae/runtime.py",
        "src/moe_sae/contribution_model.py",
        "patches/maxtext/0001-gemma4-return-selected-outputs.patch",
    }
    tampered = json.loads(json.dumps(contract))
    tampered["launch_contract"]["qualification_plan"]["measured_steps"] += 1
    with pytest.raises(
        ContributionQualificationWorkerError,
        match="self-hash",
    ):
        validate_authorization_contract(tampered)


def test_qualification_runtime_rejects_attention_drift() -> None:
    config = load_contribution_config(CONFIG_PATH)
    runtime = _runtime()
    runtime.config.attention = "dot_product"
    with pytest.raises(ContributionQualificationWorkerError, match="attention"):
        _validate_effective_runtime(
            runtime,
            config=config,
            mesh=MESH,
            per_device_sequence_batch=1,
        )


def test_child_initializes_distributed_first_and_seals_observation(
    tmp_path: Path,
) -> None:
    invocation, contract, execution = _invocation_and_records(tmp_path)
    clock = _Clock()
    events: list[str] = []
    record = run_qualification_child(
        invocation,
        dependencies=_fake_dependencies(clock, events),
        qualification_started=0.0,
    )
    checked = validate_child_evidence(record)
    observation = _observation_from_dict(checked["observation"])

    assert events[0:2] == ["initialize", "process_context"]
    assert events.index("initialize") < events.index("create_runtime")
    assert events.count("prepare") == 1
    assert events[-1] == "close_runtime"
    assert observation.process_count == 4
    assert observation.process_index == 0
    assert checked["controller_index"] == 0
    assert observation.measured_capture_compile_count == 1
    assert observation.parity_oracle_compile_count == 1
    assert observation.total_compile_count == 2
    assert observation.global_output_tokens == 10 * 16_384
    assert observation.local_persisted_tokens == 10 * 4_096
    assert (
        checked["authorization_contract_sha256"]
        == contract["authorization_contract_sha256"]
    )
    assert checked["execution_sha256"] == execution["execution_sha256"]
    assert (
        checked["authorization_contract_identity"]
        != checked["observed_executable_identity"]
    )
    assert invocation.child_evidence_path.exists()
    published_sidecar = validate_recomposition_sidecar(
        json.loads(invocation.recomposition_sidecar_path.read_text())
    )
    assert (
        checked["recomposition_sidecar_sha256"]
        == published_sidecar["sidecar_sha256"]
    )
    assert events.index("create_adapter") < events.index("extract_sidecar")
    assert events.index("extract_sidecar") < events.index("prepare")


def test_child_binds_transport_controller_separately_from_jax_process(
    tmp_path: Path,
) -> None:
    invocation, _, _ = _invocation_and_records(
        tmp_path,
        controller_index=1,
    )
    record = run_qualification_child(
        invocation,
        dependencies=_fake_dependencies(
            _Clock(),
            [],
            process_index=2,
        ),
        qualification_started=0.0,
    )
    checked = validate_child_evidence(record)
    observation = _observation_from_dict(checked["observation"])

    assert checked["controller_index"] == 1
    assert observation.process_index == 2
    assert checked["persistence"]["process_index"] == 2


def test_child_rejects_contract_drift_before_distributed_initialization(
    tmp_path: Path,
) -> None:
    invocation, _, _ = _invocation_and_records(tmp_path)
    changed = replace(invocation, checkpoint_sha256="f" * 64)
    events: list[str] = []
    with pytest.raises(
        ContributionQualificationWorkerError,
        match="no longer match",
    ):
        run_qualification_child(
            changed,
            dependencies=_fake_dependencies(_Clock(), events),
            qualification_started=0.0,
        )
    assert events == []


def _result(execution_sha256: str, *, elapsed: float = 0.5) -> dict:
    unsigned = {
        "schema_version": SUPERVISION_RESULT_SCHEMA_VERSION,
        "execution_sha256": execution_sha256,
        "status": "completed",
        "pid": 123,
        "returncode": 0,
        "term_sent": False,
        "kill_sent": False,
        "process_group_reaped": True,
        "elapsed_seconds": elapsed,
        "term_requested_elapsed_seconds": None,
        "kill_requested_elapsed_seconds": None,
        "log_size_bytes": 0,
        "log_sha256": "0" * 64,
        "error_type": None,
        "error_message": None,
    }
    return {**unsigned, "result_sha256": canonical_sha256(unsigned)}


def test_launcher_uses_contract_identity_and_labels_observed_identity(
    tmp_path: Path,
) -> None:
    plan = QualificationPlan(
        warmup_steps=1,
        writer_disabled_steps=1,
        measured_steps=1,
        maximum_seconds=30,
    )
    observed_identity = ExactPerformanceIdentity.from_components(
        graph={"observed": "stablehlo"},
        hardware={"observed": "v4-32"},
        config={"observed": "launch"},
    )
    seen: dict = {}

    def fake_supervisor(**kwargs):
        seen.update(kwargs)
        execution = build_execution_record(
            authorization=kwargs["authorization"],
            command=kwargs["command"],
            environment=kwargs["environment"],
            working_directory=kwargs["working_directory"],
            log_path=kwargs["log_path"],
            execution_record_path=kwargs["execution_record_path"],
            result_path=kwargs["result_path"],
            termination_grace_seconds=kwargs["termination_grace_seconds"],
        )
        _write_json(Path(kwargs["execution_record_path"]), execution)
        command = list(kwargs["command"])
        child_path = Path(command[command.index("--child-evidence") + 1])
        contract_path = Path(command[command.index("--contract") + 1])
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        observation = QualificationHostObservation(
            identity=observed_identity,
            process_count=4,
            process_index=2,
            qualification_wall_seconds=0.4,
            compile_seconds=0.1,
            measured_device_seconds=0.01,
            persisted_window_seconds=0.011,
            writer_disabled_device_seconds=0.01,
            global_output_tokens=16_384,
            local_persisted_tokens=4_096,
            writer_disabled_global_output_tokens=16_384,
            warmup_steps=1,
            synchronized_measurement_steps=1,
            measured_capture_compile_count=1,
            parity_oracle_compile_count=1,
            total_compile_count=2,
            max_writer_backlog_batches=1,
            writer_backlog_limit_batches=2,
            writer_queue_drained=True,
            instrumentation=InstrumentationObservation(
                disabled_output_sha256=OUTPUT_SHA,
                enabled_output_sha256=OUTPUT_SHA,
                equal=True,
            ),
            diagnostics=QualificationDiagnostics(),
        )
        child = _child_record(
            contract=contract,
            execution_sha256=execution["execution_sha256"],
            controller_index=1,
            observation=observation,
            persistence=_writer_summary(process_index=2, tokens=4_096),
            checkpoint_sha256=CHECKPOINT_SHA,
            maxtext_fork_commit=MAXTEXT_FORK_COMMIT,
            effective_runtime={"fake": True},
            recomposition_sidecar=_sidecar(),
        )
        _write_json(child_path, child)
        return _result(execution["execution_sha256"])

    record = launch_qualification_host(
        config_path=CONFIG_PATH,
        checkpoint_path="/checkpoint",
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_root=None,
        repo_root=ROOT,
        jax_cache_dir=tmp_path / "cache",
        control_root=tmp_path / "control",
        scratch_root=tmp_path / "scratch",
        mesh=MESH,
        per_device_sequence_batch=1,
        controller_index=1,
        plan=plan,
        environment={"PATH": "/usr/bin"},
        supervisor=fake_supervisor,
    )
    checked = validate_supervised_host_evidence(record)
    assert seen["authorization"].purpose == "qualification"
    assert "--maxtext-root" not in seen["command"]
    assert (
        seen["authorization"].identity_sha256
        == checked["authorization_contract_identity"]["identity_sha256"]
    )
    assert (
        checked["observed_executable_identity"]["identity_sha256"]
        == observed_identity.identity_sha256
    )
    assert (
        checked["authorization_contract_identity"]["identity_sha256"]
        != checked["observed_executable_identity"]["identity_sha256"]
    )
    assert checked["observation"]["qualification_wall_seconds"] == 0.5
    assert checked["controller_index"] == 1
    assert checked["process_index"] == 2
    assert "--controller-index" in seen["command"]
    assert "--process-index" not in seen["command"]
    assert (
        tmp_path
        / "control"
        / "controller-c00001"
        / "host-evidence.json"
    ).is_file()


def test_four_supervised_hosts_aggregate_on_observed_identity(
    tmp_path: Path,
) -> None:
    invocation, contract, execution = _invocation_and_records(tmp_path)
    base = run_qualification_child(
        invocation,
        dependencies=_fake_dependencies(_Clock(), []),
        qualification_started=0.0,
    )
    base_observation = _observation_from_dict(base["observation"])
    result = _result(execution["execution_sha256"], elapsed=0.5)
    hosts = []
    observed_process_indices = (0, 2, 3, 1)
    for controller_index, process_index in enumerate(observed_process_indices):
        observation = replace(base_observation, process_index=process_index)
        child = _child_record(
            contract=contract,
            execution_sha256=execution["execution_sha256"],
            controller_index=controller_index,
            observation=observation,
            persistence=_writer_summary(process_index=process_index),
            checkpoint_sha256=CHECKPOINT_SHA,
            maxtext_fork_commit=MAXTEXT_FORK_COMMIT,
            effective_runtime={"fake": True},
            recomposition_sidecar=_sidecar(),
        )
        hosts.append(
            _supervised_host_record(
                contract=contract,
                child=child,
                execution=execution,
                result=result,
                expected_controller_index=controller_index,
            )
        )
    bundle = aggregate_supervised_host_evidence(
        hosts,
        batch_sequences=16,
        sequence_length=1_024,
    )
    assert bundle["qualification_receipt"]["status"] == "passed"
    assert (
        bundle["qualification_receipt"]["identity_sha256"]
        == base_observation.identity.identity_sha256
    )
    assert (
        bundle["observed_executable_identity"]["identity_sha256"]
        != bundle["authorization_contract_sha256"]
    )
    assert bundle["recomposition_sidecar_sha256"] == _sidecar()["sidecar_sha256"]
    assert bundle["recomposition_sidecar_canonical_sha256"] == canonical_sha256(
        _sidecar()
    )
    assert bundle["controller_to_jax_process_index"] == [
        {"controller_index": 0, "process_index": 0},
        {"controller_index": 1, "process_index": 2},
        {"controller_index": 2, "process_index": 3},
        {"controller_index": 3, "process_index": 1},
    ]

    duplicated_process_observation = replace(
        base_observation,
        process_index=0,
    )
    duplicated_process_child = _child_record(
        contract=contract,
        execution_sha256=execution["execution_sha256"],
        controller_index=3,
        observation=duplicated_process_observation,
        persistence=_writer_summary(process_index=0),
        checkpoint_sha256=CHECKPOINT_SHA,
        maxtext_fork_commit=MAXTEXT_FORK_COMMIT,
        effective_runtime={"fake": True},
        recomposition_sidecar=_sidecar(),
    )
    duplicated_process_host = _supervised_host_record(
        contract=contract,
        child=duplicated_process_child,
        execution=execution,
        result=result,
        expected_controller_index=3,
    )
    with pytest.raises(
        ContributionQualificationWorkerError,
        match="JAX process inventory",
    ):
        aggregate_supervised_host_evidence(
            [*hosts[:3], duplicated_process_host],
            batch_sequences=16,
            sequence_length=1_024,
        )

    with pytest.raises(
        ContributionQualificationWorkerError,
        match="controller inventory",
    ):
        aggregate_supervised_host_evidence(
            [*hosts[:3], hosts[2]],
            batch_sequences=16,
            sequence_length=1_024,
        )


def test_extended_workload_is_callable_only_after_observed_receipt_match() -> None:
    graph = {"graph": "observed-stablehlo"}
    hardware = {"hardware": "observed-v4-32"}
    launch = {"launch": "observed-runtime"}
    identity = ExactPerformanceIdentity.from_components(
        graph=graph,
        hardware=hardware,
        config=launch,
    )
    device_seconds = 100.0
    peak = int(4.4e15)
    evidence = QualificationEvidence(
        identity=identity,
        source_evidence_sha256="1" * 64,
        useful_flops_ledger_sha256="2" * 64,
        qualification_wall_seconds=180.0,
        measured_device_seconds=device_seconds,
        persisted_window_seconds=device_seconds,
        useful_flops=int(0.11 * peak * device_seconds),
        peak_flops_per_second=peak,
        warmup_steps=5,
        synchronized_measurement_steps=10,
        measured_graph_compile_count=1,
        auxiliary_validation_compile_count=1,
        total_compile_count=2,
        recompilations_after_warmup=0,
        oom_events=0,
        nonfinite_events=0,
        capacity_overflow_events=0,
        dropped_routed_tokens=0,
        output_tokens=1_000,
        persisted_tokens=1_000,
        max_writer_backlog_batches=1,
        writer_backlog_limit_batches=2,
        writer_failures=0,
        writer_queue_drained=True,
        static_shapes=True,
        parameters_resident_before_timing=True,
        inputs_resident_before_timing=True,
        synchronized_device_timing=True,
        instrumentation_disabled_output_sha256=OUTPUT_SHA,
        instrumentation_enabled_output_sha256=OUTPUT_SHA,
        writer_disabled_output_tokens=1_000,
        writer_disabled_device_seconds=device_seconds,
    )
    receipt = build_qualification_receipt(evidence)
    prepared = PreparedContributionExecutable(
        executable=lambda *_args: "ran",
        graph=graph,
        compile_seconds=1.0,
    )
    authorized = authorize_extended_workload_after_lowering(
        prepared,
        hardware=hardware,
        launch_config=launch,
        qualification_receipt=receipt,
        maximum_runtime_seconds=601,
        config=load_contribution_config(CONFIG_PATH),
    )
    assert authorized.authorization.basis == "qualified_exact_identity"
    assert authorized.authorization.identity_sha256 == identity.identity_sha256
    assert authorized(None, None, None, None) == "ran"

    drifted = replace(prepared, graph={"graph": "drifted"})
    with pytest.raises(ContributionPerformanceError, match="exact"):
        authorize_extended_workload_after_lowering(
            drifted,
            hardware=hardware,
            launch_config=launch,
            qualification_receipt=receipt,
            maximum_runtime_seconds=601,
            config=load_contribution_config(CONFIG_PATH),
        )


def test_local_writer_uses_create_only_safetensors_and_self_hashed_summary(
    tmp_path: Path,
) -> None:
    rows = 2
    output = SimpleNamespace(
        expert_output=np.zeros((rows, 8, 2_816), dtype=np.float32),
        expert_ids=np.broadcast_to(
            np.arange(8, dtype=np.uint8),
            (rows, 8),
        ),
        router_weights=np.full((rows, 8), 0.125, dtype=np.float32),
        shared_output=np.zeros((rows, 2_816), dtype=np.float32),
        sequence_id=np.arange(rows, dtype=np.int64),
        position=np.arange(rows, dtype=np.uint16),
    )
    writer = LocalQualificationPersist(
        tmp_path,
        process_count=1,
        process_index=0,
        sequence_length=1,
    )
    try:
        assert writer(output) == rows
        writer.flush()
        summary = writer.summary()
    finally:
        writer.close()
    assert summary["tokens"] == rows
    assert summary["max_observed_backlog"] == 1
    assert summary["queue_drained"] is True
    assert (tmp_path / "step-00000-part-000.safetensors").is_file()
    unsigned = dict(summary)
    claimed = unsigned.pop("writer_evidence_sha256")
    assert canonical_sha256(unsigned) == claimed

    duplicate = LocalQualificationPersist(
        tmp_path,
        process_count=1,
        process_index=0,
        sequence_length=1,
    )
    try:
        duplicate(output)
        with pytest.raises(
            ContributionQualificationWorkerError,
            match="already exists",
        ):
            duplicate.flush()
    finally:
        with pytest.raises(ContributionQualificationWorkerError):
            duplicate.close()


def test_local_writer_chunks_large_logical_steps_without_growing_backlog(
    tmp_path: Path,
) -> None:
    rows = 2
    output = SimpleNamespace(
        expert_output=np.zeros((rows, 8, 2_816), dtype=np.float32),
        expert_ids=np.broadcast_to(
            np.arange(8, dtype=np.uint8),
            (rows, 8),
        ),
        router_weights=np.full((rows, 8), 0.125, dtype=np.float32),
        shared_output=np.zeros((rows, 2_816), dtype=np.float32),
        sequence_id=np.arange(rows, dtype=np.int64),
        position=np.arange(rows, dtype=np.uint16),
    )
    writer = LocalQualificationPersist(
        tmp_path,
        process_count=1,
        process_index=0,
        sequence_length=1,
        wire_chunk_tokens=1,
    )
    try:
        assert writer(output) == rows
        writer.flush()
        summary = writer.summary()
    finally:
        writer.close()

    assert summary["tokens"] == rows
    assert summary["max_observed_backlog"] == 1
    assert [item["part"] for item in summary["files"]] == [0, 1]
    assert [item["parts"] for item in summary["files"]] == [2, 2]
    assert (tmp_path / "step-00000-part-000.safetensors").is_file()
    assert (tmp_path / "step-00000-part-001.safetensors").is_file()


def test_qualification_disambiguates_packed_segments_across_rows_before_chunking(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    segments = np.asarray(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.int32,
    )
    positions = np.asarray(
        [
            [0, 1, 0, 1],
            [0, 1, 0, 1],
        ],
        dtype=np.int32,
    )
    batch = GrainBatch(
        token_ids=np.ones_like(segments),
        positions=positions,
        segment_ids=segments,
    )
    observed_sequence_ids: list[np.ndarray] = []

    def execute(
        _token_ids: np.ndarray,
        _positions: np.ndarray,
        _segment_ids: np.ndarray,
        sequence_ids: np.ndarray,
    ) -> SimpleNamespace:
        clock.advance(0.1)
        observed_sequence_ids.append(np.asarray(sequence_ids).copy())
        rows = int(segments.size)
        return SimpleNamespace(
            rows=rows,
            expert_output=np.zeros((rows, 8, HIDDEN_SIZE), dtype=np.float32),
            expert_ids=np.broadcast_to(
                np.arange(8, dtype=np.uint8),
                (rows, 8),
            ),
            router_weights=np.full((rows, 8), 0.125, dtype=np.float32),
            shared_output=np.zeros((rows, HIDDEN_SIZE), dtype=np.float32),
            sequence_id=np.asarray(sequence_ids).reshape(-1),
            position=positions.astype(np.uint16).reshape(-1),
        )

    writer = LocalQualificationPersist(
        tmp_path,
        process_count=4,
        process_index=2,
        sequence_length=4,
        wire_chunk_tokens=5,
    )
    try:
        observation = run_qualification_host(
            PreparedContributionExecutable(
                executable=execute,
                graph={"stablehlo": "test"},
                compile_seconds=0.1,
            ),
            batch,
            identity=ExactPerformanceIdentity.from_components(
                graph={"graph": "test"},
                hardware={"hardware": "test"},
                config={"config": "test"},
            ),
            process_count=4,
            process_index=2,
            persist=writer,
            instrumentation_probe=lambda: InstrumentationObservation(
                disabled_output_sha256=OUTPUT_SHA,
                enabled_output_sha256=OUTPUT_SHA,
                equal=True,
            ),
            qualification_started=0.0,
            plan=QualificationPlan(
                warmup_steps=1,
                writer_disabled_steps=1,
                measured_steps=1,
            ),
            diagnostics=QualificationDiagnostics,
            clock=clock,
        )
        summary = writer.summary()
    finally:
        writer.close()

    assert observation.local_persisted_tokens == segments.size
    assert summary["tokens"] == segments.size
    assert [item["tokens"] for item in summary["files"]] == [5, 3]
    assert len(observed_sequence_ids) == 3
    for sequence_ids in observed_sequence_ids:
        assert sequence_ids.dtype == segments.dtype
        assert sequence_ids.shape == segments.shape
        np.testing.assert_array_equal(
            sequence_ids & np.int32(0xFFFF),
            segments,
        )
        np.testing.assert_array_equal(
            sequence_ids >> np.int32(24),
            np.full_like(sequence_ids, 2),
        )
        token_keys = np.rec.fromarrays(
            [
                sequence_ids.reshape(-1).astype(np.int64),
                positions.reshape(-1).astype(np.uint16),
            ],
            names=("sequence_id", "position"),
        )
        assert np.unique(token_keys).size == segments.size
