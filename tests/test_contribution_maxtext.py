from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from moe_sae.config import Gemma4Config
from moe_sae.contribution_artifacts import validate_recomposition_sidecar
from moe_sae.contribution_config import MAXTEXT_FORK_COMMIT, Experiment2Config
from moe_sae.contribution_maxtext import (
    ContributionMaxTextError,
    InstrumentationObservation,
    PreparedContributionExecutable,
    QualificationDiagnostics,
    QualificationPlan,
    aggregate_qualification_hosts,
    build_contribution_maxtext_argv,
    effective_performance_runtime_manifest,
    estimate_contribution_prefix_flops,
    exact_performance_identity,
    extract_recomposition_sidecar,
    hardware_manifest,
    launch_config_manifest,
    run_qualification_host,
    useful_flops_ledger,
)
from moe_sae.contribution_model import Gemma4ContributionPrefixAdapter
from moe_sae.contribution_performance import (
    MINIMUM_USEFUL_PREFIX_MFU,
    ContributionPerformanceError,
)
from moe_sae.harvest import GrainBatch
from moe_sae.runtime import (
    MeshCandidate,
    estimate_useful_prefix_flops,
)

_SHA = "a" * 64
_MESH = MeshCandidate("ep16-fsdp1-tp1", 16, 1, 1)


def test_contribution_flops_add_exact_target_shared_expert() -> None:
    batch_sequences = 16
    sequence_length = 1_024
    base = estimate_useful_prefix_flops(
        batch_sequences=batch_sequences,
        sequence_length=sequence_length,
    )
    expected_shared = 6 * batch_sequences * sequence_length * 2_816 * 2_112
    assert (
        estimate_contribution_prefix_flops(
            batch_sequences=batch_sequences,
            sequence_length=sequence_length,
        )
        == base + expected_shared
    )
    ledger = useful_flops_ledger(
        batch_sequences=batch_sequences,
        sequence_length=sequence_length,
        measured_steps=10,
    )
    assert ledger["useful_flops"] == 10 * (base + expected_shared)
    assert "shared_ffn" in ledger["target_block_components"]
    assert "padding" in ledger["excluded"]


def test_contribution_argv_accepts_b128_and_keeps_route_optimizations(
    tmp_path,
) -> None:
    config = Experiment2Config()
    argv = build_contribution_maxtext_argv(
        checkpoint_path="gs://bucket/model/items",
        grain_train_files="gs://bucket/data/train",
        grain_eval_files="gs://bucket/data/eval",
        mesh=_MESH,
        per_device_sequence_batch=8,
        attention=config.model.attention,
        jax_cache_dir=tmp_path / "jax-cache",
    )
    assert "use_ragged_sort=true" in argv
    assert "per_device_batch_size=8" in argv
    assert f"attention={config.model.attention}" in argv
    assert "run_name=gemma4-contribution-v1" in argv
    assert f"jax_cache_dir={tmp_path / 'jax-cache'}" in argv
    assert sum(value.startswith("attention=") for value in argv) == 1
    assert sum(value.startswith("jax_cache_dir=") for value in argv) == 1


def test_effective_runtime_rejects_attention_drift() -> None:
    config = Experiment2Config()
    runtime_config = SimpleNamespace(
        use_multimodal=False,
        scan_layers=False,
        pure_nnx_decoder=True,
        enable_dropout=False,
        use_ragged_sort=True,
        sparse_matmul=True,
        use_ring_of_experts=False,
        float32_weight_sum=True,
        attention=config.model.attention,
        matmul_precision="default",
        max_target_length=config.model.sequence_length,
        per_device_batch_size=8,
        ici_expert_parallelism=_MESH.expert_parallelism,
        ici_fsdp_parallelism=_MESH.fsdp_parallelism,
        ici_tensor_parallelism=_MESH.tensor_parallelism,
    )
    runtime_mesh = SimpleNamespace(
        axis_names=("expert", "fsdp", "tensor"),
        devices=np.empty((16, 1, 1), dtype=object),
    )
    manifest = effective_performance_runtime_manifest(
        runtime_config,
        runtime_mesh,
        config=config,
        mesh=_MESH,
        per_device_sequence_batch=8,
    )
    assert manifest["attention"] == config.model.attention

    runtime_config.attention = "dot_product"
    with pytest.raises(ContributionMaxTextError, match="attention"):
        effective_performance_runtime_manifest(
            runtime_config,
            runtime_mesh,
            config=config,
            mesh=_MESH,
            per_device_sequence_batch=8,
        )


class _Device:
    def __init__(self, identifier: int):
        self.id = identifier % 4
        self.process_index = identifier // 4
        self.platform = "tpu"
        self.device_kind = "TPU v4"
        self.slice_index = 0


def test_exact_identity_binds_v4_hardware_and_launch_config(tmp_path) -> None:
    hardware = hardware_manifest(
        devices=[_Device(index) for index in range(16)],
        process_count=4,
        mesh=_MESH,
    )
    launch = launch_config_manifest(
        Experiment2Config(),
        checkpoint_sha256=_SHA,
        mesh=_MESH,
        per_device_sequence_batch=1,
        sequence_length=1_024,
        writer_backlog_limit_batches=2,
        jax_cache_dir=tmp_path,
    )
    identity = exact_performance_identity(
        graph={"schema": "test", "stablehlo_sha256": _SHA},
        hardware=hardware,
        launch_config=launch,
    )
    assert len(identity.identity_sha256) == 64
    assert launch["selected_route_ordering_sorts"] == 1
    assert hardware["global_chip_count"] == 16

    with pytest.raises(ContributionMaxTextError, match="v4-32"):
        hardware_manifest(
            devices=[_Device(index) for index in range(8)],
            process_count=2,
            mesh=_MESH,
        )


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _Output:
    rows = 16_384

    def block_until_ready(self) -> _Output:
        return self


class _Persist:
    backlog_limit = 2
    max_observed_backlog = 1
    queue_drained = False

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock

    def __call__(self, output: _Output) -> int:
        assert output.rows == 16_384
        self.clock.advance(0.005)
        return 8_192

    def flush(self) -> None:
        self.clock.advance(0.005)
        self.queue_drained = True


def _host_observation(process_index: int):
    clock = _Clock()

    def execute(*_args):
        clock.advance(0.1)
        return _Output()

    identity = exact_performance_identity(
        graph={"stablehlo": "x"},
        hardware={"hardware": "v4-32"},
        launch_config={"config": "test"},
    )
    prepared = PreparedContributionExecutable(
        executable=execute,
        graph={"stablehlo": "x"},
        compile_seconds=0.5,
    )
    batch = GrainBatch(
        token_ids=np.ones((1, 1_024), dtype=np.int32),
        positions=np.broadcast_to(
            np.arange(1_024, dtype=np.int32),
            (1, 1_024),
        ),
        segment_ids=np.ones((1, 1_024), dtype=np.int32),
    )
    persist = _Persist(clock)
    return run_qualification_host(
        prepared,
        batch,
        identity=identity,
        process_count=2,
        process_index=process_index,
        persist=persist,
        instrumentation_probe=lambda: InstrumentationObservation(
            disabled_output_sha256=_SHA,
            enabled_output_sha256=_SHA,
            equal=True,
        ),
        qualification_started=0.0,
        plan=QualificationPlan(),
        diagnostics=QualificationDiagnostics,
        clock=clock,
    )


def test_host_qualification_aggregates_into_passing_mfu_receipt() -> None:
    hosts = [_host_observation(0), _host_observation(1)]
    evidence, bundle = aggregate_qualification_hosts(
        hosts,
        batch_sequences=16,
        sequence_length=1_024,
    )
    assert evidence.useful_prefix_mfu >= MINIMUM_USEFUL_PREFIX_MFU
    assert evidence.persisted_tokens == evidence.output_tokens
    assert evidence.auxiliary_throughput_fraction == pytest.approx(1.0)
    assert bundle["qualification_receipt"]["status"] == "passed"
    assert (
        bundle["qualification_receipt"]["identity_sha256"]
        == hosts[0].identity.identity_sha256
    )


def test_low_mfu_and_missing_host_fail_closed() -> None:
    hosts = [_host_observation(0), _host_observation(1)]
    slow = [
        replace(
            item,
            measured_device_seconds=item.measured_device_seconds * 10,
            writer_disabled_device_seconds=(
                item.writer_disabled_device_seconds * 10
            ),
            persisted_window_seconds=item.measured_device_seconds * 10.05,
            qualification_wall_seconds=max(
                item.qualification_wall_seconds,
                item.measured_device_seconds * 10.1,
            ),
        )
        for item in hosts
    ]
    with pytest.raises(ContributionPerformanceError, match="3%"):
        aggregate_qualification_hosts(
            slow,
            batch_sequences=16,
            sequence_length=1_024,
        )
    with pytest.raises(ContributionMaxTextError, match="inventory"):
        aggregate_qualification_hosts(
            hosts[:1],
            batch_sequences=16,
            sequence_length=1_024,
        )


class _Param:
    def __init__(self, value):
        self.value = value


class _Norm:
    with_scale = True
    epsilon = 1e-6
    scale_offset = 0.0

    def __init__(self, value: float):
        self.scale = _Param(jnp.full((2_816,), value, dtype=jnp.bfloat16))


def test_sidecar_is_extracted_from_exact_target_parameters() -> None:
    adapter = object.__new__(Gemma4ContributionPrefixAdapter)
    adapter.config = Gemma4Config()
    target = SimpleNamespace(
        mlp=SimpleNamespace(
            post_feedforward_layernorm_2=_Norm(1.25),
            post_feedforward_layernorm_1=_Norm(0.75),
        ),
        post_ffw_norm=_Norm(0.5),
        layer_scalar=_Param(jnp.asarray([0.875], dtype=jnp.bfloat16)),
    )
    adapter.layers = [SimpleNamespace() for _ in range(30)]
    adapter.layers[14] = target
    sidecar = extract_recomposition_sidecar(
        adapter,
        checkpoint_sha256=_SHA,
        source_fingerprints={"checkpoint_manifest": "b" * 64},
    )
    checked = validate_recomposition_sidecar(sidecar)
    assert checked["maxtext_fork_commit"] == MAXTEXT_FORK_COMMIT
    assert checked["tensors"]["layer_scalar"]["shape"] == [1]
