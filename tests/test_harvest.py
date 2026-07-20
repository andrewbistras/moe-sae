from __future__ import annotations

from dataclasses import replace
import json
import time

import numpy as np
import pytest

import moe_sae.harvest as harvest_module
from moe_sae.artifacts import ArtifactDataset, CaptureMetadata, load_manifest
from moe_sae.config import Gemma4Config
from moe_sae.harvest import (
    BenchmarkDiagnostics,
    CampaignSummary,
    HarvestError,
    HarvestResult,
    benchmark_worker,
    canary_observation,
    capture_metadata,
    compile_adapter_step,
    extract_grain_batch,
    harvest_host,
    make_sequence_ids,
    publish_completed_capture,
    run_harvest,
)
from moe_sae.harvest_control import (
    HarvestAbortRequested,
    HarvestControlError,
    campaign_identity,
    load_campaign_control,
    request_campaign_abort,
)
from moe_sae.model import CaptureBatch
from moe_sae.runtime import HARVEST_MESHES, BenchmarkObservation


HIDDEN = 2_816


def metadata(split: str = "train") -> CaptureMetadata:
    return CaptureMetadata(
        checkpoint_sha256="a" * 64,
        maxtext_commit="48c7da3e28e8f57fe36ddf09d00eec53072a2ea0",
        dataset_fingerprint="3" * 64,
        split=split,
        config_sha256="b" * 64,
        environment_sha256="c" * 64,
        conversion_attestation_sha256="9" * 64,
    )


def grain(tokens: list[list[int]], segments: list[list[int]]) -> dict[str, np.ndarray]:
    token_array = np.asarray(tokens, dtype=np.int32)
    return {
        "inputs": token_array,
        "inputs_position": np.broadcast_to(
            np.arange(token_array.shape[1], dtype=np.int32), token_array.shape
        ).copy(),
        "inputs_segmentation": np.asarray(segments, dtype=np.int32),
        "targets": token_array.copy(),  # ignored deterministically
    }


class FakeStep:
    def __init__(self):
        self.calls: list[tuple[tuple[int, ...], np.ndarray]] = []

    def __call__(self, tokens, positions, segments, sequence_ids):
        del positions, segments
        tokens = np.asarray(tokens)
        self.calls.append((tokens.shape, np.asarray(sequence_ids).copy()))
        rows = tokens.size
        activation = np.repeat(
            tokens.reshape(rows, 1).astype(np.float32), HIDDEN, axis=1
        )
        return CaptureBatch(
            activation=activation,
            expert_ids=np.tile(np.arange(8, dtype=np.uint8), (rows, 1)),
            router_weights=np.full((rows, 8), 1 / 8, dtype=np.float32),
            sequence_id=np.full(rows, -1, dtype=np.int32),
            position=np.full(rows, -1, dtype=np.int32),
        )


def test_grain_extraction_is_strict_and_sequence_ids_are_stable():
    raw = grain([[3, 4, 0, 0], [5, 6, 7, 0]], [[1, 1, 0, 0], [1, 2, 2, 0]])
    extracted = extract_grain_batch(raw, expected_sequence_length=4)
    assert extracted.shape == (2, 4)
    ids = make_sequence_ids(extracted.segment_ids, process_index=3, batch_index=9)
    assert ids.dtype == np.int64 and ids.shape == (2, 4)
    assert ids[0, 0] == ids[0, 1]
    assert ids[1, 1] == ids[1, 2]
    assert ids[0, 0] != ids[1, 0]
    assert np.array_equal(
        ids,
        make_sequence_ids(extracted.segment_ids, process_index=3, batch_index=9),
    )

    with pytest.raises(KeyError, match="inputs_position"):
        extract_grain_batch(
            {"inputs": raw["inputs"], "inputs_segmentation": raw["inputs_segmentation"]}
        )
    with pytest.raises(ValueError, match="sequence length"):
        extract_grain_batch(raw, expected_sequence_length=1_024)


def test_host_harvest_compacts_exact_fields_and_publishes(tmp_path):
    raw = grain(
        [[0, 3, 4, 1], [2, 5, 6, 7]],
        [[0, 1, 1, 1], [2, 2, 2, 2]],
    )
    step = FakeStep()
    result = run_harvest(
        [raw],
        step,
        artifact_root=tmp_path,
        metadata=metadata(),
        split="train",
        total_tokens=5,
        process_count=1,
        process_index=0,
        sequence_length=4,
    )

    assert result.new_tokens == 5 and result.resumed_tokens == 0
    assert result.batches_executed == 1
    assert result.max_writer_backlog == 1
    assert result.manifest_path == str(tmp_path / "manifest.json")
    assert step.calls[0][0] == (2, 4)  # full static shape, not five compact rows

    stored = next(ArtifactDataset(tmp_path).iter_batches())
    assert stored.activation.shape == (5, HIDDEN)
    assert stored.activation[:, 0].tolist() == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert stored.sequence_id.dtype == np.int64
    assert stored.position.dtype == np.uint16
    assert stored.position.tolist() == [1, 2, 1, 2, 3]
    assert np.all(stored.sequence_id >= 0)
    assert load_manifest(tmp_path)["total_tokens"] == 5
    campaign_id = campaign_identity(
        metadata().to_dict(),
        split="train",
        total_tokens=5,
        stop_after_tokens=5,
        process_count=1,
    )
    control = load_campaign_control(tmp_path, campaign_id)
    assert control["hosts"][0]["state"] == "completed"
    assert control["hosts"][0]["progress"]["durable_host_tokens"] == 5


def test_abort_is_taken_before_device_execution_at_a_safe_batch_boundary(tmp_path):
    campaign_id = campaign_identity(
        metadata().to_dict(),
        split="train",
        total_tokens=2,
        stop_after_tokens=2,
        process_count=1,
    )
    request_campaign_abort(tmp_path, campaign_id, reason="test stop")
    step = FakeStep()
    with pytest.raises(HarvestAbortRequested, match="safe boundary"):
        harvest_host(
            [grain([[3, 4, 0, 0]], [[1, 1, 0, 0]])],
            step,
            artifact_root=tmp_path,
            metadata=metadata(),
            split="train",
            total_tokens=2,
            process_count=1,
            process_index=0,
            sequence_length=4,
        )
    assert step.calls == []
    assert not (tmp_path / "manifest.json").exists()
    assert load_campaign_control(tmp_path, campaign_id)["hosts"][0]["state"] == (
        "aborted"
    )


def test_completed_multi_process_lane_replays_collectives_for_incomplete_peer(
    tmp_path,
):
    raw = grain(
        [[3, 4, 0, 0], [5, 6, 0, 0]],
        [[1, 1, 0, 0], [1, 1, 0, 0]],
    )
    first_step = FakeStep()
    harvest_host(
        [raw],
        first_step,
        artifact_root=tmp_path,
        metadata=metadata(),
        split="train",
        total_tokens=4,
        process_count=2,
        process_index=0,
        sequence_length=4,
        batch_collective=lambda complete, execute: (complete, execute),
        abort_collective=lambda requested: requested,
    )

    decisions: list[tuple[bool, bool]] = []
    aborts: list[bool] = []

    def peer_decision(local_complete: bool, local_execute: bool) -> tuple[bool, bool]:
        decisions.append((local_complete, local_execute))
        # This lane is complete, but the simulated peer still needs the shared
        # model collective for this batch and completes immediately afterward.
        return True, True

    def collective_abort(local_requested: bool) -> bool:
        aborts.append(local_requested)
        return local_requested

    replay_step = FakeStep()
    resumed = harvest_host(
        [raw],
        replay_step,
        artifact_root=tmp_path,
        metadata=metadata(),
        split="train",
        total_tokens=4,
        process_count=2,
        process_index=0,
        sequence_length=4,
        batch_collective=peer_decision,
        abort_collective=collective_abort,
    )

    assert decisions == [(True, False)]
    assert aborts == [False, False]
    assert len(replay_step.calls) == 1
    assert resumed.durable_host_tokens == 2
    assert resumed.new_tokens == 2


def test_multi_process_harvest_requires_collective_abort_coordination(tmp_path):
    with pytest.raises(ValueError, match="batch collective"):
        harvest_host(
            [],
            FakeStep(),
            artifact_root=tmp_path,
            metadata=metadata(),
            split="train",
            total_tokens=2,
            process_count=2,
            process_index=0,
            sequence_length=4,
        )
    with pytest.raises(ValueError, match="abort collective"):
        harvest_host(
            [],
            FakeStep(),
            artifact_root=tmp_path,
            metadata=metadata(),
            split="train",
            total_tokens=2,
            process_count=2,
            process_index=0,
            sequence_length=4,
            batch_collective=lambda complete, execute: (complete, execute),
        )


def test_collectively_observed_abort_controls_the_branch(tmp_path):
    collective_inputs: list[bool] = []

    def collective(local_requested: bool) -> bool:
        collective_inputs.append(local_requested)
        return True  # another controller observed the shared marker

    step = FakeStep()
    with pytest.raises(HarvestAbortRequested, match="safe boundary"):
        harvest_host(
            [grain([[3, 4, 0, 0]], [[1, 1, 0, 0]])],
            step,
            artifact_root=tmp_path,
            metadata=metadata(),
            split="train",
            total_tokens=2,
            process_count=1,
            process_index=0,
            sequence_length=4,
            abort_collective=collective,
        )
    assert collective_inputs == [False]
    assert step.calls == []


def test_corrupt_abort_marker_still_enters_collective_before_failing(tmp_path):
    campaign_id = campaign_identity(
        metadata().to_dict(),
        split="train",
        total_tokens=2,
        stop_after_tokens=2,
        process_count=1,
    )
    abort_path = request_campaign_abort(tmp_path, campaign_id, reason="test stop")
    payload = json.loads(abort_path.read_text(encoding="utf-8"))
    payload["reason"] = "tampered"
    abort_path.write_text(json.dumps(payload), encoding="utf-8")
    collective_inputs: list[bool] = []

    def collective(local_requested: bool) -> bool:
        collective_inputs.append(local_requested)
        return local_requested

    step = FakeStep()
    with pytest.raises(HarvestControlError, match="SHA-256 mismatch"):
        harvest_host(
            [grain([[3, 4, 0, 0]], [[1, 1, 0, 0]])],
            step,
            artifact_root=tmp_path,
            metadata=metadata(),
            split="train",
            total_tokens=2,
            process_count=1,
            process_index=0,
            sequence_length=4,
            abort_collective=collective,
        )
    assert collective_inputs == [True]
    assert step.calls == []


def test_collective_abort_after_writer_drain_prevents_finalization(tmp_path):
    collective_inputs: list[bool] = []

    def collective(local_requested: bool) -> bool:
        collective_inputs.append(local_requested)
        # The first call admits the one model step.  The second represents a
        # peer seeing the marker after every writer queue has drained but before
        # any host seals its manifest.
        return len(collective_inputs) == 2

    step = FakeStep()
    with pytest.raises(HarvestAbortRequested, match="safe boundary"):
        harvest_host(
            [grain([[3, 4, 0, 0]], [[1, 1, 0, 0]])],
            step,
            artifact_root=tmp_path,
            metadata=metadata(),
            split="train",
            total_tokens=2,
            process_count=1,
            process_index=0,
            sequence_length=4,
            abort_collective=collective,
        )

    assert collective_inputs == [False, False]
    assert len(step.calls) == 1
    assert not (tmp_path / "manifest.json").exists()
    host = json.loads(
        (tmp_path / "hosts" / "host-p00000.json").read_text(encoding="utf-8")
    )
    assert host["complete"] is False
    assert sum(int(shard["tokens"]) for shard in host["shards"]) == 2


def test_canary_prefix_resumes_full_batches_and_partial_boundary(tmp_path):
    batches = [
        grain([[3, 4, 0, 0]], [[1, 1, 0, 0]]),
        grain([[5, 6, 7, 8]], [[1, 1, 2, 2]]),
    ]
    step = FakeStep()
    canary = harvest_host(
        batches,
        step,
        artifact_root=tmp_path,
        metadata=metadata(),
        split="train",
        total_tokens=6,
        stop_after_tokens=3,
        process_count=1,
        process_index=0,
        sequence_length=4,
    )
    assert canary.new_tokens == 3
    assert canary.batches_executed == 2
    assert len(step.calls) == 2

    production = harvest_host(
        batches,
        step,
        artifact_root=tmp_path,
        metadata=metadata(),
        split="train",
        total_tokens=6,
        process_count=1,
        process_index=0,
        sequence_length=4,
        baseline_resumed_tokens=3,
    )
    assert production.resumed_tokens == 3
    assert production.new_tokens == 3
    assert production.batches_scanned == 2
    assert production.batches_executed == 1  # batch zero skipped before execution
    assert len(step.calls) == 3
    publish_completed_capture(tmp_path, metadata(), process_count=1, total_tokens=6)
    manifest = load_manifest(tmp_path)
    assert manifest["total_tokens"] == 6
    assert len(manifest["shards"]) == 3
    assert [entry["range_start"] for entry in manifest["shards"]] == [0, 2, 3]


def test_interrupted_phase_restores_durable_tokens_and_cumulative_telemetry(tmp_path):
    first = grain([[3, 4, 0, 0]], [[1, 1, 0, 0]])
    second = grain([[5, 6, 0, 0]], [[1, 1, 0, 0]])

    def interrupted():
        yield first
        raise RuntimeError("injected interruption")

    step = FakeStep()
    with pytest.raises(RuntimeError, match="injected interruption"):
        harvest_host(
            interrupted(),
            step,
            artifact_root=tmp_path,
            metadata=metadata(),
            split="train",
            total_tokens=4,
            process_count=1,
            process_index=0,
            sequence_length=4,
            compile_seconds=1.5,
        )
    journal_path = tmp_path / "telemetry" / "journal-p00000-stop-000000000004.json"
    interrupted_state = harvest_module._read_json_mapping(journal_path)
    assert interrupted_state["durable_new_tokens"] == 2
    assert interrupted_state["complete"] is False

    resumed = harvest_host(
        [first, second],
        step,
        artifact_root=tmp_path,
        metadata=metadata(),
        split="train",
        total_tokens=4,
        process_count=1,
        process_index=0,
        sequence_length=4,
        compile_seconds=2.0,
    )
    assert resumed.new_tokens == 4
    assert resumed.resumed_tokens == 0  # journal attributes all rows to this phase
    assert resumed.batches_executed == 2
    assert resumed.batches_scanned >= 3  # includes the replayed immutable first range
    assert resumed.elapsed_seconds >= interrupted_state["elapsed_seconds"]
    assert resumed.compile_seconds == 3.5
    final_state = harvest_module._read_json_mapping(journal_path)
    assert final_state["durable_new_tokens"] == 4
    assert final_state["complete"] is True
    observation = canary_observation(
        [resumed], synthetic_goodput=1.0, expected_tokens=4
    )
    assert observation.captured_tokens == 4


def test_interrupted_suffix_preserves_canary_baseline_and_prior_suffix_work(tmp_path):
    batches = [
        grain([[3, 4, 0, 0]], [[1, 1, 0, 0]]),
        grain([[5, 6, 0, 0]], [[1, 1, 0, 0]]),
        grain([[7, 8, 0, 0]], [[1, 1, 0, 0]]),
    ]
    step = FakeStep()
    canary = harvest_host(
        batches,
        step,
        artifact_root=tmp_path,
        metadata=metadata(),
        split="train",
        total_tokens=6,
        stop_after_tokens=2,
        process_count=1,
        process_index=0,
        sequence_length=4,
    )
    assert canary.new_tokens == 2 and canary.resumed_tokens == 0

    def interrupted_suffix():
        yield batches[0]
        yield batches[1]
        raise RuntimeError("suffix interruption")

    with pytest.raises(RuntimeError, match="suffix interruption"):
        harvest_host(
            interrupted_suffix(),
            step,
            artifact_root=tmp_path,
            metadata=metadata(),
            split="train",
            total_tokens=6,
            process_count=1,
            process_index=0,
            sequence_length=4,
            baseline_resumed_tokens=2,
        )
    interrupted_state = harvest_module._read_json_mapping(
        tmp_path / "telemetry" / "journal-p00000-stop-000000000006.json"
    )
    assert interrupted_state["baseline_resumed_tokens"] == 2
    assert interrupted_state["durable_new_tokens"] == 2

    production = harvest_host(
        batches,
        step,
        artifact_root=tmp_path,
        metadata=metadata(),
        split="train",
        total_tokens=6,
        process_count=1,
        process_index=0,
        sequence_length=4,
        baseline_resumed_tokens=2,
    )
    assert production.new_tokens == 4
    assert production.resumed_tokens == 2
    assert production.batches_executed == 2
    assert production.elapsed_seconds >= interrupted_state["elapsed_seconds"]


def test_static_shape_change_and_input_exhaustion_fail_without_publication(tmp_path):
    step = FakeStep()
    with pytest.raises(HarvestError, match="input exhausted"):
        harvest_host(
            [grain([[3, 0]], [[1, 0]])],
            step,
            artifact_root=tmp_path / "short",
            metadata=metadata(),
            split="train",
            total_tokens=2,
            process_count=1,
            process_index=0,
            sequence_length=2,
        )
    assert not (tmp_path / "short" / "manifest.json").exists()

    changing = [
        grain([[3, 0]], [[1, 0]]),
        grain([[4, 0], [5, 0]], [[1, 0], [1, 0]]),
    ]
    with pytest.raises(HarvestError, match="static Grain shape changed"):
        harvest_host(
            changing,
            step,
            artifact_root=tmp_path / "shape",
            metadata=metadata(),
            split="train",
            total_tokens=2,
            process_count=1,
            process_index=0,
            sequence_length=2,
        )


class BenchmarkPersist:
    max_observed_backlog = 1

    def __init__(self):
        self.calls = 0
        self.flushed = False

    def __call__(self, output):
        self.calls += 1
        return output.rows

    def flush(self):
        self.flushed = True


def test_benchmark_worker_uses_five_warmups_and_synchronized_200_steps():
    raw = grain([[3, 4, 5, 6]], [[1, 1, 1, 1]])
    step = FakeStep()
    persist = BenchmarkPersist()
    observation = benchmark_worker(
        [raw] * 205,
        step,
        mesh=HARVEST_MESHES[0],
        per_device_sequence_batch=1,
        persist=persist,
        diagnostics=lambda: BenchmarkDiagnostics(instrumentation_equal=True),
        sequence_length=4,
    )
    assert len(step.calls) == 205
    assert persist.calls == 200 and persist.flushed
    assert observation.synchronized_steps == 200
    assert observation.output_tokens == observation.persisted_tokens == 800
    assert observation.max_writer_backlog == 1
    assert observation.useful_flops > 0
    with pytest.raises(ValueError, match="five warmups"):
        benchmark_worker(
            [raw] * 204,
            step,
            mesh=HARVEST_MESHES[0],
            per_device_sequence_batch=1,
            warmup_steps=4,
            sequence_length=4,
        )


def test_production_benchmark_closes_runtime_on_adapter_setup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    class Runtime:
        model = object()

    runtime = Runtime()
    closed: list[object] = []

    class FailingAdapter:
        @staticmethod
        def from_maxtext(model, config):
            del model, config
            raise HarvestError("adapter setup failed")

    monkeypatch.setattr(harvest_module, "build_maxtext_argv", lambda **kwargs: [])
    monkeypatch.setattr(harvest_module, "create_maxtext_runtime", lambda argv: runtime)
    monkeypatch.setattr(harvest_module, "Gemma4PrefixAdapter", FailingAdapter)
    monkeypatch.setattr(
        harvest_module, "_close_maxtext_runtime", lambda value: closed.append(value)
    )

    with pytest.raises(HarvestError, match="adapter setup failed"):
        harvest_module.benchmark_production_candidate(
            checkpoint_path="/checkpoint",
            mesh_candidate=HARVEST_MESHES[0],
            per_device_sequence_batch=1,
            config=Gemma4Config(),
            output_directory=tmp_path,
        )

    assert closed == [runtime]


def test_production_benchmark_cleans_persistence_and_scratch_on_worker_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    class Runtime:
        model = object()
        train_iterator = iter([grain([list(range(1_024))], [[1] * 1_024])])

    class AdapterFactory:
        @staticmethod
        def from_maxtext(model, config):
            del model, config
            return object()

    class NoopContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback
            return False

    class Persistence:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

        def diagnostics(self):
            return BenchmarkDiagnostics()

    runtime = Runtime()
    persistence = Persistence()
    closed: list[object] = []
    monkeypatch.setattr(harvest_module, "build_maxtext_argv", lambda **kwargs: [])
    monkeypatch.setattr(harvest_module, "create_maxtext_runtime", lambda argv: runtime)
    monkeypatch.setattr(harvest_module, "Gemma4PrefixAdapter", AdapterFactory)
    monkeypatch.setattr(
        harvest_module, "compile_adapter_step", lambda adapter: object()
    )
    monkeypatch.setattr(
        harvest_module, "maxtext_mesh_context", lambda value: NoopContext()
    )
    monkeypatch.setattr(
        harvest_module, "_run_instrumentation_parity", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        harvest_module,
        "environment_fingerprint",
        lambda root: {"sha256": "d" * 64},
    )
    monkeypatch.setattr(
        harvest_module,
        "_ProductionBenchmarkPersist",
        lambda *args, **kwargs: persistence,
    )
    monkeypatch.setattr(
        harvest_module,
        "compile_warm_and_replay",
        lambda *args, **kwargs: (iter(()), 0.0),
    )

    def fail_worker(*args, **kwargs):
        del args, kwargs
        raise HarvestError("worker failed")

    monkeypatch.setattr(harvest_module, "benchmark_worker", fail_worker)
    monkeypatch.setattr(
        harvest_module, "_close_maxtext_runtime", lambda value: closed.append(value)
    )

    with pytest.raises(HarvestError, match="worker failed"):
        harvest_module.benchmark_production_candidate(
            checkpoint_path="/checkpoint",
            mesh_candidate=HARVEST_MESHES[0],
            per_device_sequence_batch=1,
            config=Gemma4Config(),
            output_directory=tmp_path,
        )

    assert persistence.closed
    assert closed == [runtime]
    assert list(tmp_path.iterdir()) == []


def test_benchmark_host_aggregation_uses_slowest_host_and_exact_durable_union():
    observation = BenchmarkObservation(
        mesh=HARVEST_MESHES[0],
        per_device_sequence_batch=1,
        synchronized_steps=200,
        elapsed_seconds=4.0,
        useful_flops=10.0,
        output_tokens=800,
        persisted_tokens=800,
        input_wait_seconds=0.1,
        recompilations=0,
        dropped_routed_tokens=0,
        max_writer_backlog=1,
        instrumentation_equal=True,
    )
    slower = replace(observation, elapsed_seconds=5.0, input_wait_seconds=0.2)
    payloads = [
        {
            "observation": item.as_dict(),
            "local_persisted_rows": 400,
            "process_count": 2,
            "process_index": index,
            "global_chips": 16,
        }
        for index, item in enumerate((observation, slower))
    ]
    result = harvest_module._aggregate_benchmark_hosts(
        payloads,
        mesh=HARVEST_MESHES[0],
        per_device_sequence_batch=1,
        expected_output_tokens=800,
        global_chips=16,
    )
    assert result.elapsed_seconds == 5.0
    assert result.input_wait_seconds == 0.2
    assert result.persisted_tokens == result.output_tokens == 800


def test_nnx_jit_cache_diagnostic_unwraps_jitted_fn():
    class Compiled:
        def _cache_size(self):
            return 3

    class JitWrapped:
        jitted_fn = Compiled()

    assert harvest_module._jit_cache_size(JitWrapped()) == 3
    assert harvest_module._jit_cache_size(object()) is None


def _result(index: int) -> HarvestResult:
    return HarvestResult(
        split="train",
        process_count=2,
        process_index=index,
        host_id=f"p{index:05d}",
        total_tokens=50_000_000,
        stop_after_tokens=2_000_000,
        host_quota=1_000_000,
        new_tokens=1_000_000,
        resumed_tokens=0,
        durable_host_tokens=1_000_000,
        batches_scanned=10,
        batches_executed=10,
        elapsed_seconds=10.0,
        device_seconds=10.0,
        input_wait_seconds=0.2,
        max_writer_backlog=1,
    )


def test_canary_metrics_require_exact_fresh_two_million_tokens():
    observation = canary_observation(
        [_result(0), _result(1)], synthetic_goodput=210_000
    )
    assert observation.captured_tokens == 2_000_000
    assert observation.real_goodput == 200_000
    assert observation.persisted_goodput == 200_000
    assert observation.device_output_goodput == 200_000
    assert observation.input_wait_fraction == pytest.approx(0.02)
    assert observation.passes()
    with pytest.raises(ValueError, match="fresh"):
        canary_observation(
            [_result(0), replace(_result(1), resumed_tokens=1)],
            synthetic_goodput=210_000,
        )


def test_compile_adapter_step_is_injectable_without_tpu_dependencies():
    calls = []

    def adapter(*args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

    step = compile_adapter_step(adapter, jit=lambda function: function)
    assert step(1, 2, 3, 4) == "ok"
    assert calls[0][1] == {"model_mode": "train"}


def test_nnx_model_is_an_explicit_jit_argument():
    model = object()
    config = object()
    transformed_calls = []

    class Adapter:
        def __init__(self, dynamic_model, static_config):
            self.model = dynamic_model
            self.config = static_config

        def __call__(self, *args, **kwargs):
            return self.model, self.config, args, kwargs

    def fake_jit(function):
        def compiled(*args):
            transformed_calls.append(args)
            return function(*args)

        return compiled

    step = compile_adapter_step(Adapter(model, config), jit=fake_jit)
    result = step(1, 2, 3, 4)
    assert transformed_calls == [(model, 1, 2, 3, 4)]
    assert result[:2] == (model, config)
    assert result[3] == {"model_mode": "train"}


def test_instrumentation_parity_is_required_and_fail_closed():
    raw = grain([[3, 4, 0, 0]], [[1, 1, 0, 0]])
    extracted = extract_grain_batch(raw, expected_sequence_length=4)

    class ParityAdapter:
        def __init__(self, value):
            self.value = value

        def instrumentation_parity(self, *args, **kwargs):
            assert len(args) == 4
            assert kwargs == {"model_mode": "train"}
            return np.asarray(self.value)

    assert harvest_module._run_instrumentation_parity(
        ParityAdapter(True), extracted, process_count=1
    )
    with pytest.raises(HarvestError, match="differs"):
        harvest_module._run_instrumentation_parity(
            ParityAdapter(False), extracted, process_count=1
        )
    with pytest.raises(HarvestError, match="lacks instrumentation_parity"):
        harvest_module._run_instrumentation_parity(object(), extracted, process_count=1)


def test_instrumentation_parity_jit_receives_nnx_model_explicitly(monkeypatch):
    extracted = extract_grain_batch(
        grain([[3, 4, 0, 0]], [[1, 1, 0, 0]]),
        expected_sequence_length=4,
    )
    model = object()
    config = object()
    transformed_calls = []

    class Adapter:
        def __init__(self, dynamic_model, static_config):
            self.model = dynamic_model
            self.config = static_config

        def instrumentation_parity(self, *args, **kwargs):
            assert self.model is model and self.config is config
            assert kwargs == {"model_mode": "train"}
            return np.asarray(True)

    class FakeNNX:
        @staticmethod
        def jit(function):
            def compiled(*args):
                transformed_calls.append(args)
                return function(*args)

            return compiled

    original_import = harvest_module.import_module
    monkeypatch.setattr(
        harvest_module,
        "import_module",
        lambda name: FakeNNX if name == "flax.nnx" else original_import(name),
    )
    assert harvest_module._run_instrumentation_parity(
        Adapter(model, config), extracted, process_count=1
    )
    assert transformed_calls[0][0] is model


def test_compile_warm_replays_first_batch_without_charging_campaign_order():
    batches = [
        grain([[3, 4, 0, 0]], [[1, 1, 0, 0]]),
        grain([[5, 6, 0, 0]], [[1, 1, 0, 0]]),
    ]
    step = FakeStep()
    ticks = iter((10.0, 12.5))
    replay, compile_seconds = harvest_module.compile_warm_and_replay(
        batches,
        step,
        sequence_length=4,
        clock=lambda: next(ticks),
    )
    assert compile_seconds == 2.5
    assert len(step.calls) == 1
    replayed = list(replay)
    assert replayed[0] is batches[0]
    assert replayed[1] is batches[1]


class _Shard:
    def __init__(self, index, data):
        self.index = index
        self.data = data


class _GlobalArray:
    def __init__(self, shape, shards):
        self.shape = shape
        self.addressable_shards = shards


def test_process_local_array_assembles_feature_shards_and_partitions_replication():
    full = np.arange(16, dtype=np.int32).reshape(4, 4)
    sharded = _GlobalArray(
        full.shape,
        [
            _Shard((slice(0, 4), slice(0, 2)), full[:, :2]),
            _Shard((slice(0, 4), slice(2, 4)), full[:, 2:]),
        ],
    )
    assembled = harvest_module._process_local_array(
        sharded,
        "test",
        process_count=1,
        process_index=0,
    )
    assert np.array_equal(assembled, full)

    replicated = _GlobalArray(
        full.shape,
        [_Shard((slice(0, 4), slice(0, 4)), full)],
    )
    process_one = harvest_module._process_local_array(
        replicated,
        "test",
        process_count=2,
        process_index=1,
    )
    assert np.array_equal(process_one, full[2:])


def test_capture_metadata_records_config_and_coordinated_environment_hashes():
    config = Gemma4Config()
    result = capture_metadata(
        config,
        checkpoint_sha256="a" * 64,
        conversion_attestation_sha256="9" * 64,
        dataset_fingerprint="4" * 64,
        split="train",
        environment={"sha256": "d" * 64},
    )
    assert result.config_sha256 == config.config_hash
    assert result.environment_sha256 == "d" * 64


def test_campaign_summary_uses_max_host_wall_and_exact_chip_seconds(tmp_path):
    config = Gemma4Config()
    results = [_result(0), replace(_result(1), elapsed_seconds=11.0)]
    for result in results:
        object.__setattr__(result, "executed_scientific_tokens", 1_000_000)
        object.__setattr__(result, "batches_executed", 123)
    summary = harvest_module._aggregate_campaign_summary(
        results,
        artifact_root=tmp_path,
        config=config,
        split="train",
        phase="canary",
        total_tokens=config.train_tokens,
        stop_after_tokens=config.canary_tokens,
        global_chips=16,
        mesh=HARVEST_MESHES[0],
        per_device_sequence_batch=1,
        benchmark={
            "synthetic_goodput": 190_000.0,
            "useful_mfu": 0.55,
            "dropped_routed_tokens": 0,
            "recompilations": 0,
            "instrumentation_equal": True,
        },
        capture_metadata=capture_metadata(
            config,
            checkpoint_sha256="a" * 64,
            conversion_attestation_sha256="9" * 64,
            dataset_fingerprint="4" * 64,
            split="train",
            environment={"sha256": "d" * 64},
        ).to_dict(),
        prior_summary=None,
        prior_results=None,
    )
    assert isinstance(summary, CampaignSummary)
    assert summary.captured_tokens == 2_000_000
    assert summary.wall_seconds == 11.0
    assert summary.chip_seconds == 176.0
    assert summary.canary_passed
    summary.validate()


def test_campaign_cost_sums_synchronized_phase_stragglers(tmp_path, monkeypatch):
    """A different slow host in each phase costs max(canary)+max(suffix)."""

    config = Gemma4Config()
    metadata_value = capture_metadata(
        config,
        checkpoint_sha256="a" * 64,
        conversion_attestation_sha256="9" * 64,
        dataset_fingerprint="4" * 64,
        split="train",
        environment={"sha256": "d" * 64},
    ).to_dict()
    benchmark = {
        "synthetic_goodput": 19_000.0,
        "useful_mfu": 0.55,
        "dropped_routed_tokens": 0,
        "recompilations": 0,
        "instrumentation_equal": True,
    }
    canary_results = [
        replace(
            _result(0),
            elapsed_seconds=100.0,
            device_seconds=100.0,
            batches_executed=123,
            executed_scientific_tokens=1_000_000,
        ),
        replace(
            _result(1),
            elapsed_seconds=10.0,
            device_seconds=10.0,
            batches_executed=123,
            executed_scientific_tokens=1_000_000,
        ),
    ]
    canary = harvest_module._aggregate_campaign_summary(
        canary_results,
        artifact_root=tmp_path,
        config=config,
        split="train",
        phase="canary",
        total_tokens=config.train_tokens,
        stop_after_tokens=config.canary_tokens,
        global_chips=16,
        mesh=HARVEST_MESHES[0],
        per_device_sequence_batch=1,
        benchmark=benchmark,
        capture_metadata=metadata_value,
        prior_summary=None,
        prior_results=None,
    )
    assert canary.canary_passed
    production_results = [
        replace(
            _result(0),
            stop_after_tokens=config.train_tokens,
            host_quota=config.train_tokens // 2,
            new_tokens=24_000_000,
            resumed_tokens=1_000_000,
            durable_host_tokens=config.train_tokens // 2,
            elapsed_seconds=10.0,
            device_seconds=10.0,
            batches_executed=2_930,
            executed_scientific_tokens=24_000_000,
            manifest_path=str(tmp_path / "manifest.json"),
        ),
        replace(
            _result(1),
            stop_after_tokens=config.train_tokens,
            host_quota=config.train_tokens // 2,
            new_tokens=24_000_000,
            resumed_tokens=1_000_000,
            durable_host_tokens=config.train_tokens // 2,
            elapsed_seconds=100.0,
            device_seconds=100.0,
            batches_executed=2_930,
            executed_scientific_tokens=24_000_000,
            manifest_path=str(tmp_path / "manifest.json"),
        ),
    ]
    monkeypatch.setattr(
        harvest_module,
        "load_manifest",
        lambda root: {
            "metadata": metadata_value,
            "manifest_sha256": "e" * 64,
        },
    )
    production = harvest_module._aggregate_campaign_summary(
        production_results,
        artifact_root=tmp_path,
        config=config,
        split="train",
        phase="production",
        total_tokens=config.train_tokens,
        stop_after_tokens=config.train_tokens,
        global_chips=16,
        mesh=HARVEST_MESHES[0],
        per_device_sequence_batch=1,
        benchmark=benchmark,
        capture_metadata=metadata_value,
        prior_summary=canary,
        prior_results=canary_results,
    )
    assert production.phase_wall_seconds == 100.0
    assert production.wall_seconds == 200.0
    assert production.chip_seconds == 3_200.0


def test_sealed_manifest_rejects_indeterminate_publication_telemetry(tmp_path):
    raw = grain(
        [[0, 3, 4, 1], [2, 5, 6, 7]],
        [[0, 1, 1, 1], [2, 2, 2, 2]],
    )
    run_harvest(
        [raw],
        FakeStep(),
        artifact_root=tmp_path,
        metadata=metadata(),
        split="train",
        total_tokens=5,
        process_count=1,
        process_index=0,
        sequence_length=4,
    )
    status_path = tmp_path / "telemetry" / "publication-stop-000000000005.json"
    harvest_module.atomic_write_json(
        status_path,
        {
            "ok": False,
            "state": "publishing",
            "publication_seconds": 3.0,
            "attempt_started_unix": time.time() - 2.0,
        },
    )

    with pytest.raises(HarvestError, match="indeterminate"):
        harvest_module._recover_interrupted_publication(
            tmp_path,
            status_path,
            total_tokens=5,
            process_count=1,
            process_index=0,
            split="train",
        )


def test_sealed_manifest_recovers_measured_failed_publication_attempt(tmp_path):
    raw = grain(
        [[0, 3, 4, 1], [2, 5, 6, 7]],
        [[0, 1, 1, 1], [2, 2, 2, 2]],
    )
    run_harvest(
        [raw],
        FakeStep(),
        artifact_root=tmp_path,
        metadata=metadata(),
        split="train",
        total_tokens=5,
        process_count=1,
        process_index=0,
        sequence_length=4,
    )
    status_path = tmp_path / "telemetry" / "publication-stop-000000000005.json"
    harvest_module.atomic_write_json(
        status_path,
        {
            "ok": False,
            "state": "error",
            "publication_seconds": 3.0,
        },
    )

    recovered = harvest_module._recover_interrupted_publication(
        tmp_path,
        status_path,
        total_tokens=5,
        process_count=1,
        process_index=0,
        split="train",
    )

    assert recovered["ok"] is True
    assert recovered["state"] == "recovered"
    assert recovered["publication_seconds"] >= 3.0
