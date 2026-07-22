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
    publication_status = json.loads(
        (tmp_path / "telemetry" / "publication-stop-000000000005.json").read_text()
    )
    assert publication_status["total_tokens"] == 5
    assert publication_status["capture_manifest_sha256"] == load_manifest(tmp_path)[
        "manifest_sha256"
    ]
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
        shard_tokens=2,
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
        shard_tokens=2,
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
    assert len(manifest["shards"]) == 4
    assert [entry["range_start"] for entry in manifest["shards"]] == [0, 2, 3, 5]
    assert [entry["tokens"] for entry in manifest["shards"]] == [2, 1, 2, 1]


def test_harvest_coalesces_model_batches_into_canonical_bounded_shards(tmp_path):
    batches = [
        grain([[3, 4, 0, 0]], [[1, 1, 0, 0]]),
        grain([[5, 6, 7, 0]], [[1, 1, 1, 0]]),
        grain([[8, 9, 10, 11]], [[1, 1, 1, 1]]),
    ]
    result = harvest_host(
        batches,
        FakeStep(),
        artifact_root=tmp_path,
        metadata=metadata(),
        split="train",
        total_tokens=9,
        process_count=1,
        process_index=0,
        sequence_length=4,
        shard_tokens=5,
    )
    assert result.new_tokens == 9
    assert result.max_writer_backlog == 1

    host = json.loads(
        (tmp_path / "hosts" / "host-p00000.json").read_text(encoding="utf-8")
    )
    assert [entry["tokens"] for entry in host["shards"]] == [5, 4]
    assert [entry["range_id"] for entry in host["shards"]] == [
        "train-p00000-r000000000000-n000000005",
        "train-p00000-r000000000005-n000000004",
    ]

    publish_completed_capture(tmp_path, metadata(), process_count=1, total_tokens=9)
    manifest = load_manifest(tmp_path)
    expected = np.arange(3, 12, dtype=np.float64)
    assert manifest["statistics"]["count"] == 9
    assert manifest["statistics"]["sum"][0] == pytest.approx(expected.sum())
    assert manifest["statistics"]["squared_norm"] == pytest.approx(
        HIDDEN * np.square(expected).sum()
    )
    stored = list(ArtifactDataset(tmp_path).iter_batches())
    assert [batch.rows for batch in stored] == [5, 4]
    assert np.concatenate([batch.activation[:, 0] for batch in stored]).tolist() == (
        expected.tolist()
    )


def test_harvest_skips_empty_scientific_batches_before_coalescing(tmp_path):
    batches = [
        grain([[0, 1, 2, 0]], [[0, 0, 0, 0]]),
        grain([[3, 4, 0, 0]], [[1, 1, 0, 0]]),
    ]
    step = FakeStep()

    result = harvest_host(
        batches,
        step,
        artifact_root=tmp_path,
        metadata=metadata(),
        split="train",
        total_tokens=2,
        process_count=1,
        process_index=0,
        sequence_length=4,
        shard_tokens=2,
    )

    assert result.batches_scanned == 2
    assert result.batches_executed == 1
    assert len(step.calls) == 1
    publish_completed_capture(tmp_path, metadata(), process_count=1, total_tokens=2)
    stored = next(ArtifactDataset(tmp_path).iter_batches())
    assert stored.activation[:, 0].tolist() == [3.0, 4.0]


def test_resume_reconciles_crash_after_partial_boundary_commit(tmp_path, monkeypatch):
    batches = [grain([[3, 4, 5, 0]], [[1, 1, 1, 0]])]
    original = harvest_module.ArtifactWriter.write_range
    crashed = False

    def commit_then_crash(self, *args, **kwargs):
        nonlocal crashed
        result = original(self, *args, **kwargs)
        if not crashed:
            crashed = True
            raise RuntimeError("crash after immutable commit")
        return result

    monkeypatch.setattr(
        harvest_module.ArtifactWriter, "write_range", commit_then_crash
    )
    with pytest.raises(RuntimeError, match="crash after immutable commit"):
        harvest_host(
            batches,
            FakeStep(),
            artifact_root=tmp_path,
            metadata=metadata(),
            split="train",
            total_tokens=3,
            process_count=1,
            process_index=0,
            sequence_length=4,
            shard_tokens=5,
        )
    state = harvest_module._read_json_mapping(
        tmp_path / "telemetry" / "journal-p00000-stop-000000000003.json"
    )
    assert state["pending"]["range_end"] == 3

    monkeypatch.setattr(harvest_module.ArtifactWriter, "write_range", original)
    resumed_step = FakeStep()
    resumed = harvest_host(
        batches,
        resumed_step,
        artifact_root=tmp_path,
        metadata=metadata(),
        split="train",
        total_tokens=3,
        process_count=1,
        process_index=0,
        sequence_length=4,
        shard_tokens=5,
    )
    assert resumed.durable_host_tokens == 3
    assert resumed_step.calls == []
    final = harvest_module._read_json_mapping(
        tmp_path / "telemetry" / "journal-p00000-stop-000000000003.json"
    )
    assert final["pending"] is None and final["complete"] is True


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (
            [
                {
                    "range_id": "train-p00000-r000000000001-n000000002",
                    "range_start": 1,
                    "range_end": 3,
                    "tokens": 2,
                }
            ],
            "canonical contiguous",
        ),
        (
            [
                {
                    "range_id": "train-p00000-r000000000000-n000000002",
                    "range_start": 0,
                    "range_end": 2,
                    "tokens": 2,
                },
                {
                    "range_id": "train-p00000-r000000000001-n000000002",
                    "range_start": 1,
                    "range_end": 3,
                    "tokens": 2,
                },
            ],
            "canonical contiguous",
        ),
        (
            [
                {
                    "range_id": "train-p00000-b000000000000-o000000000-n000000002",
                    "range_start": 0,
                    "range_end": 2,
                    "tokens": 2,
                }
            ],
            "unsupported range ID",
        ),
        (
            [
                {
                    "range_id": "train-p00000-r000000000000-n000000005",
                    "range_start": 0,
                    "range_end": 5,
                    "tokens": 5,
                }
            ],
            "canonical contiguous",
        ),
        (
            [
                {
                    "range_id": "train-p00000-r000000000000-n000000003",
                    "range_start": 0,
                    "range_end": 3,
                    "tokens": 3,
                }
            ],
            "not a phase boundary",
        ),
    ],
)
def test_resume_rejects_noncanonical_coalesced_ledgers(entries, message):
    class PresentWriter:
        def has_range(self, range_id, start, end):
            del range_id, start, end
            return True

    with pytest.raises(HarvestError, match=message):
        harvest_module._validate_committed_shards(
            entries,
            split="train",
            process_index=0,
            expected_global_start=0,
            max_tokens=4,
            allowed_partial_ends={2, 10},
            writer=PresentWriter(),
        )


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
            shard_tokens=2,
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
        shard_tokens=2,
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
    publish_completed_capture(tmp_path, metadata(), process_count=1, total_tokens=4)
    restored = np.concatenate(
        [batch.activation[:, 0] for batch in ArtifactDataset(tmp_path).iter_batches()]
    )
    assert restored.tolist() == [3.0, 4.0, 5.0, 6.0]
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
        shard_tokens=2,
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
            shard_tokens=2,
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
        shard_tokens=2,
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


def test_benchmark_worker_separates_device_and_persistence_wall_time():
    raw = grain([[3, 4, 5, 6]], [[1, 1, 1, 1]])

    class Clock:
        value = 0.0

        def __call__(self):
            return self.value

        def advance(self, seconds):
            self.value += seconds

    clock = Clock()

    class Batches:
        remaining = 205

        def __iter__(self):
            return self

        def __next__(self):
            if self.remaining == 0:
                raise StopIteration
            self.remaining -= 1
            clock.advance(1.0)
            return raw

    delegate = FakeStep()

    def step(*args):
        clock.advance(2.0)
        return delegate(*args)

    class Persist:
        max_observed_backlog = 1

        def __call__(self, output):
            clock.advance(3.0)
            return output.rows

        def flush(self):
            clock.advance(4.0)

    observation = benchmark_worker(
        Batches(),
        step,
        mesh=HARVEST_MESHES[0],
        per_device_sequence_batch=1,
        persist=Persist(),
        diagnostics=lambda: BenchmarkDiagnostics(instrumentation_equal=True),
        sequence_length=4,
        clock=clock,
    )

    assert observation.device_elapsed_seconds == 400.0
    assert observation.persistence_seconds == 604.0
    assert observation.input_wait_seconds == 200.0
    assert observation.elapsed_seconds == 1_204.0
    assert observation.device_tokens_per_second == 2.0
    assert observation.persisted_tokens_per_second == pytest.approx(800 / 1_204)


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


def test_production_benchmark_abort_does_not_finalize_empty_writer(tmp_path):
    persistence = harvest_module._ProductionBenchmarkPersist(
        tmp_path,
        metadata=metadata(),
        config=Gemma4Config(),
        process_count=4,
        process_index=0,
        reference_instrumentation_equal=True,
    )

    persistence.abort()
    persistence.abort()

    assert persistence._closed
    assert not persistence._finalized
    assert not (tmp_path / "hosts" / "host-p00000.json").exists()


def test_production_benchmark_coalesces_steps_to_production_shard_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    import ml_dtypes

    writes = []

    class Writer:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.finalized = False

        def write_range(self, range_id, start, end, capture):
            writes.append((range_id, start, end, capture.activation.shape[0]))

        def finalize_host(self):
            self.finalized = True

    writer = Writer()
    monkeypatch.setattr(
        harvest_module,
        "ArtifactWriter",
        lambda *args, **kwargs: writer,
    )
    config = replace(Gemma4Config(), shard_tokens=2_048)
    persistence = harvest_module._ProductionBenchmarkPersist(
        tmp_path,
        metadata=metadata(),
        config=config,
        process_count=1,
        process_index=0,
        reference_instrumentation_equal=True,
    )
    capture = CaptureBatch(
        activation=np.zeros((1_024, HIDDEN), dtype=ml_dtypes.bfloat16),
        expert_ids=np.tile(np.arange(8, dtype=np.uint8), (1_024, 1)),
        router_weights=np.full((1_024, 8), 0.125, dtype=ml_dtypes.bfloat16),
        sequence_id=np.arange(1_024, dtype=np.int64),
        position=np.arange(1_024, dtype=np.uint16),
    )

    assert persistence(capture) == 1_024
    assert persistence(capture) == 1_024
    assert persistence(capture) == 1_024
    persistence.flush()

    assert [(start, end, rows) for _, start, end, rows in writes] == [
        (0, 2_048, 2_048),
        (2_048, 3_072, 1_024),
    ]
    assert persistence.local_persisted_rows == 3_072
    assert persistence.calls == persistence.validated_calls == 3
    assert writer.finalized


def test_production_benchmark_cleans_persistence_and_scratch_on_worker_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    class Runtime:
        model = object()
        mesh = object()
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
            self.aborted = False
            self.remote_cleaned = False

        def close(self):
            self.closed = True
            raise RuntimeError("host has committed no ranges")

        def abort(self):
            self.aborted = True
            raise RuntimeError("abort failed")

        def cleanup_remote_objects(self):
            self.remote_cleaned = True

        def diagnostics(self):
            return BenchmarkDiagnostics()

    runtime = Runtime()
    persistence = Persistence()
    closed: list[object] = []
    monkeypatch.setattr(harvest_module, "build_maxtext_argv", lambda **kwargs: [])
    monkeypatch.setattr(harvest_module, "create_maxtext_runtime", lambda argv: runtime)
    monkeypatch.setattr(harvest_module, "Gemma4PrefixAdapter", AdapterFactory)
    monkeypatch.setattr(
        harvest_module, "_capture_output_shardings", lambda mesh: None
    )
    monkeypatch.setattr(
        harvest_module, "compile_adapter_step", lambda adapter, **kwargs: object()
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

    with pytest.raises(HarvestError, match="worker failed") as raised:
        harvest_module.benchmark_production_candidate(
            checkpoint_path="/checkpoint",
            mesh_candidate=HARVEST_MESHES[0],
            per_device_sequence_batch=1,
            config=Gemma4Config(),
            output_directory=tmp_path,
        )

    assert persistence.aborted
    assert not persistence.closed
    assert persistence.remote_cleaned
    assert raised.value.__notes__ == [
        "suppressed benchmark persistence error: RuntimeError: abort failed"
    ]
    assert closed == [runtime]
    assert list(tmp_path.iterdir()) == []


def test_production_benchmark_surfaces_cleanup_failure_after_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    class Runtime:
        model = object()
        mesh = object()
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
            self.remote_cleaned = False

        def close(self):
            self.closed = True
            raise RuntimeError("successful cleanup failed")

        def abort(self):
            raise AssertionError("successful benchmark must not abort persistence")

        def cleanup_remote_objects(self):
            self.remote_cleaned = True

        def diagnostics(self):
            return BenchmarkDiagnostics()

    runtime = Runtime()
    persistence = Persistence()
    closed: list[object] = []
    monkeypatch.setattr(harvest_module, "build_maxtext_argv", lambda **kwargs: [])
    monkeypatch.setattr(harvest_module, "create_maxtext_runtime", lambda argv: runtime)
    monkeypatch.setattr(harvest_module, "Gemma4PrefixAdapter", AdapterFactory)
    monkeypatch.setattr(
        harvest_module, "_capture_output_shardings", lambda mesh: None
    )
    monkeypatch.setattr(
        harvest_module, "compile_adapter_step", lambda adapter, **kwargs: object()
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
    monkeypatch.setattr(
        harvest_module, "benchmark_worker", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        harvest_module, "_close_maxtext_runtime", lambda value: closed.append(value)
    )

    with pytest.raises(RuntimeError, match="successful cleanup failed"):
        harvest_module.benchmark_production_candidate(
            checkpoint_path="/checkpoint",
            mesh_candidate=HARVEST_MESHES[0],
            per_device_sequence_batch=1,
            config=Gemma4Config(),
            output_directory=tmp_path,
        )

    assert persistence.closed
    assert persistence.remote_cleaned
    assert closed == [runtime]
    assert list(tmp_path.iterdir()) == []


def test_benchmark_host_aggregation_uses_slowest_host_and_exact_durable_union():
    observation = BenchmarkObservation(
        mesh=HARVEST_MESHES[0],
        per_device_sequence_batch=1,
        synchronized_steps=200,
        elapsed_seconds=4.0,
        device_elapsed_seconds=2.0,
        persistence_seconds=1.0,
        useful_flops=10.0,
        output_tokens=800,
        persisted_tokens=800,
        input_wait_seconds=0.1,
        recompilations=0,
        dropped_routed_tokens=0,
        max_writer_backlog=1,
        instrumentation_equal=True,
    )
    slower = replace(
        observation,
        elapsed_seconds=5.0,
        device_elapsed_seconds=3.0,
        persistence_seconds=1.5,
        input_wait_seconds=0.2,
    )
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
    assert result.device_elapsed_seconds == 3.0
    assert result.persistence_seconds == 1.5
    assert result.input_wait_seconds == 0.2
    assert result.persisted_tokens == result.output_tokens == 800


def test_selected_benchmark_rejects_obsolete_combined_timing_schema():
    with pytest.raises(HarvestError, match="obsolete timing schema"):
        harvest_module._benchmark_telemetry(
            {
                "mesh": HARVEST_MESHES[0].name,
                "per_device_sequence_batch": 1,
                "passes_hard_gate": True,
                "useful_mfu": 0.55,
                "persisted_tokens_per_second": 1.0,
                "dropped_routed_tokens": 0,
                "recompilations": 0,
                "instrumentation_equal": True,
            },
            mesh=HARVEST_MESHES[0],
            per_device_sequence_batch=1,
        )


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


def test_compile_adapter_step_forwards_explicit_capture_shardings():
    sentinel = object()
    transformed = []

    def fake_jit(function, **kwargs):
        transformed.append(kwargs)
        return function

    step = compile_adapter_step(
        lambda *args, **kwargs: (args, kwargs),
        jit=fake_jit,
        out_shardings=sentinel,  # type: ignore[arg-type]
    )

    step(1, 2, 3, 4)
    assert transformed == [{"out_shardings": sentinel}]


def test_capture_output_shardings_use_every_mesh_axis_for_rows():
    import jax
    from jax.sharding import Mesh, PartitionSpec as P

    axis_names = ("data", "fsdp", "tensor", "expert")
    mesh = Mesh(np.asarray(jax.devices()).reshape((1, 1, 1, 1)), axis_names)
    shardings = harvest_module._capture_output_shardings(mesh)

    assert shardings.activation.spec == P(axis_names, None)
    assert shardings.expert_ids.spec == P(axis_names, None)
    assert shardings.router_weights.spec == P(axis_names, None)
    assert shardings.sequence_id.spec == P(axis_names)
    assert shardings.position.spec == P(axis_names)


def test_nnx_model_is_prebound_to_inference_pure_jit():
    model = object()
    config = object()
    sentinel = object()
    partial_arguments = []
    transformed_calls = []
    transformed_options = []

    class Adapter:
        def __init__(self, dynamic_model, static_config):
            self.model = dynamic_model
            self.config = static_config

        def __call__(self, *args, **kwargs):
            return self.model, self.config, args, kwargs

    def fake_jit(function, *partial_args, **kwargs):
        partial_arguments.append(partial_args)
        transformed_options.append(kwargs)

        def compiled(*args):
            transformed_calls.append(args)
            return function(*partial_args, *args)

        return compiled

    step = compile_adapter_step(
        Adapter(model, config),
        jit=fake_jit,
        out_shardings=sentinel,  # type: ignore[arg-type]
    )
    result = step(1, 2, 3, 4)
    assert partial_arguments == [(model,)]
    assert transformed_calls == [(1, 2, 3, 4)]
    assert transformed_options == [
        {
            "graph": True,
            "graph_updates": False,
            "out_shardings": sentinel,
        }
    ]
    assert result[:2] == (model, config)
    assert result[3] == {"model_mode": "train"}


def test_nnx_partial_adapter_preserves_output_and_prunes_unused_state():
    nnx = pytest.importorskip("flax.nnx")
    jnp = pytest.importorskip("jax.numpy")

    class TinyModel(nnx.Module):
        def __init__(self):
            self.used = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
            self.unused = nnx.Param(jnp.ones((19, 23), dtype=jnp.float32))

    class TinyAdapter:
        def __init__(self, model, config):
            self.model = model
            self.config = config

        def __call__(
            self,
            token_ids,
            positions,
            segment_ids,
            sequence_ids,
            *,
            model_mode,
        ):
            assert model_mode == "train"
            return (
                self.model.used(token_ids)
                + positions
                + segment_ids
                + sequence_ids
            )

    adapter = TinyAdapter(TinyModel(), object())
    token_ids = jnp.asarray([[1.0, 2.0]], dtype=jnp.float32)
    zeros = jnp.zeros((1, 2), dtype=jnp.float32)
    expected = adapter(token_ids, zeros, zeros, zeros, model_mode="train")

    step = compile_adapter_step(adapter)
    actual = step(token_ids, zeros, zeros, zeros)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))

    # jit_partial has already bound the graph, so lowering accepts only the
    # four runtime arrays.  The intentionally distinctive unused parameter
    # shape must not survive JAX's unused-input pruning.
    compiled = getattr(step, "_compiled_step")
    lowered_text = compiled.lower(token_ids, zeros, zeros, zeros).as_text()
    assert "19x23xf32" not in lowered_text


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
    def __init__(self, shape, shards, sharding=None):
        self.shape = shape
        self.addressable_shards = shards
        self.sharding = sharding


class _Device:
    def __init__(self, process_index):
        self.process_index = process_index


class _Sharding:
    def __init__(self, device_map):
        self.device_map = device_map

    def devices_indices_map(self, shape):
        del shape
        return self.device_map


def test_process_local_array_assembles_feature_shards_and_rejects_replication():
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
        _Sharding(
            {
                _Device(0): (slice(0, 4), slice(0, 4)),
                _Device(1): (slice(0, 4), slice(0, 4)),
            }
        ),
    )
    with pytest.raises(HarvestError, match="data-lane ownership is ambiguous"):
        harvest_module._process_local_array(
            replicated,
            "test",
            process_count=2,
            process_index=1,
        )


def test_process_local_array_concatenates_noncontiguous_complete_row_slabs():
    full = np.arange(32, dtype=np.int32).reshape(8, 4)
    interleaved = _GlobalArray(
        full.shape,
        [
            _Shard((slice(0, 2), slice(0, 2)), full[0:2, :2]),
            _Shard((slice(0, 1), slice(2, 4)), full[0:1, 2:]),
            _Shard((slice(1, 2), slice(2, 4)), full[1:2, 2:]),
            _Shard((slice(6, 8), slice(0, 4)), full[6:8]),
        ],
        _Sharding(
            {
                _Device(0): (slice(0, 2), slice(0, 2)),
                _Device(0): (slice(0, 1), slice(2, 4)),
                _Device(0): (slice(1, 2), slice(2, 4)),
                _Device(1): (slice(2, 6), slice(0, 4)),
                _Device(0): (slice(6, 8), slice(0, 4)),
            }
        ),
    )

    assembled = harvest_module._process_local_array(
        interleaved,
        "test",
        process_count=2,
        process_index=0,
    )

    assert np.array_equal(assembled, np.concatenate((full[0:2], full[6:8])))


def test_process_local_array_matches_v4_32_interleaved_process_geometry():
    full = np.arange(64, dtype=np.int32).reshape(16, 4)
    devices = [_Device(slot % 4) for slot in range(16)]
    device_map = {
        device: (slice(slot, slot + 1), slice(0, 4))
        for slot, device in enumerate(devices)
    }
    sharding = _Sharding(device_map)

    for process_index in range(4):
        slots = list(range(process_index, 16, 4))
        value = _GlobalArray(
            full.shape,
            [
                _Shard((slice(slot, slot + 1), slice(0, 4)), full[slot : slot + 1])
                for slot in slots
            ],
            sharding,
        )

        assembled = harvest_module._process_local_array(
            value,
            "test",
            process_count=4,
            process_index=process_index,
        )

        assert np.array_equal(assembled, full[slots])


def test_capture_row_ownership_rejects_tp2_grain_output_mismatch():
    output = _GlobalArray(
        (16, 4),
        [
            _Shard((slice(0, 2), slice(0, 4)), np.zeros((2, 4))),
            _Shard((slice(8, 10), slice(0, 4)), np.zeros((2, 4))),
        ],
    )
    grain_coverage = ((0, 1), (4, 5), (8, 9), (12, 13))

    with pytest.raises(HarvestError, match="process-owned rows"):
        harvest_module._require_leading_coverage(
            output,
            "activation",
            grain_coverage,
        )


def test_capture_row_ownership_accepts_forced_input_layout():
    output = _GlobalArray(
        (16, 4),
        [
            _Shard((slice(slot, slot + 1), slice(0, 4)), np.zeros((1, 4)))
            for slot in (0, 4, 8, 12)
        ],
    )

    harvest_module._require_leading_coverage(
        output,
        "activation",
        ((0, 1), (4, 5), (8, 9), (12, 13)),
    )


@pytest.mark.parametrize(
    ("value", "unit", "message"),
    [
        (
            _GlobalArray(
                (4, 4),
                [_Shard((slice(0, 2), slice(0, 4)), np.zeros((2, 4)))],
                _Sharding(
                    {
                        _Device(0): (slice(0, 2), slice(0, 4)),
                        _Device(1): (slice(0, 2), slice(0, 4)),
                        _Device(0): (slice(2, 4), slice(0, 4)),
                    }
                ),
            ),
            1,
            "partial shards are replicated across processes",
        ),
        (
            _GlobalArray(
                (4, 4),
                [_Shard((slice(0, 1), slice(0, 4)), np.zeros((1, 4)))],
                _Sharding(
                    {
                        _Device(0): (slice(0, 1), slice(0, 4)),
                        _Device(1): (slice(2, 4), slice(0, 4)),
                    }
                ),
            ),
            1,
            "shards do not tile the global shape",
        ),
        (
            _GlobalArray(
                (8, 4),
                [_Shard((slice(1, 3), slice(0, 4)), np.zeros((2, 4)))],
                _Sharding(
                    {
                        _Device(1): (slice(0, 1), slice(0, 4)),
                        _Device(0): (slice(1, 3), slice(0, 4)),
                        _Device(1): (slice(3, 8), slice(0, 4)),
                    }
                ),
            ),
            2,
            "row slab splits a partition unit",
        ),
    ],
)
def test_process_local_array_rejects_unsafe_global_or_unit_geometry(
    value, unit, message
):
    with pytest.raises(HarvestError, match=message):
        harvest_module._process_local_array(
            value,
            "test",
            process_count=2,
            process_index=0,
            replicated_partition_unit=unit,
        )


@pytest.mark.parametrize(
    ("shards", "message"),
    [
        (
            [
                _Shard(
                    (slice(0, 2), slice(0, 3)),
                    np.arange(6, dtype=np.int32).reshape(2, 3),
                )
            ],
            "do not tile complete local rows",
        ),
        (
            [
                _Shard(
                    (slice(0, 2), slice(0, 3)),
                    np.arange(6, dtype=np.int32).reshape(2, 3),
                ),
                _Shard(
                    (slice(0, 2), slice(2, 4)),
                    np.arange(4, dtype=np.int32).reshape(2, 2),
                ),
            ],
            "feature rectangles overlap",
        ),
    ],
)
def test_process_local_array_rejects_incomplete_or_overlapping_feature_tiles(
    shards, message
):
    value = _GlobalArray((2, 4), shards)

    with pytest.raises(HarvestError, match=message):
        harvest_module._process_local_array(
            value,
            "test",
            process_count=1,
            process_index=0,
        )


def test_process_local_array_requires_global_map_for_multi_process_arrays():
    full = np.arange(16, dtype=np.int32).reshape(4, 4)
    value = _GlobalArray(
        full.shape,
        [_Shard((slice(0, 2), slice(0, 4)), full[:2])],
    )

    with pytest.raises(HarvestError, match="global sharding map is required"):
        harvest_module._process_local_array(
            value,
            "test",
            process_count=2,
            process_index=0,
        )


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
    assert recovered["total_tokens"] == 5
    assert recovered["capture_manifest_sha256"] == load_manifest(tmp_path)[
        "manifest_sha256"
    ]


def test_manifest_visibility_retry_waits_for_enoent_then_checks_identity(tmp_path):
    expected_metadata = metadata().to_dict()
    expected_sha256 = "d" * 64
    visibility = iter((False, False, True))
    sleeps: list[float] = []

    manifest = harvest_module._load_expected_manifest_with_retry(
        tmp_path,
        expected_tokens=5,
        expected_metadata=expected_metadata,
        expected_manifest_sha256=expected_sha256,
        timeout_seconds=1.0,
        poll_seconds=0.1,
        path_visible=lambda _path: next(visibility),
        sleep=sleeps.append,
        loader=lambda _root: {
            "total_tokens": 5,
            "metadata": expected_metadata,
            "manifest_sha256": expected_sha256,
        },
    )

    assert manifest["manifest_sha256"] == expected_sha256
    assert sleeps == [0.1, 0.1]


def test_manifest_visibility_retry_is_bounded_and_never_accepts_wrong_identity(
    tmp_path,
):
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    with pytest.raises(HarvestError, match="bounded retry expired"):
        harvest_module._load_expected_manifest_with_retry(
            tmp_path,
            expected_tokens=5,
            expected_metadata=metadata().to_dict(),
            timeout_seconds=0.5,
            poll_seconds=0.2,
            clock=lambda: now[0],
            sleep=sleep,
            path_visible=lambda _path: False,
        )

    loader_calls = 0

    def wrong_loader(_root):
        nonlocal loader_calls
        loader_calls += 1
        return {
            "total_tokens": 5,
            "metadata": metadata("eval").to_dict(),
            "manifest_sha256": "d" * 64,
        }

    with pytest.raises(HarvestError, match="different capture identity"):
        harvest_module._load_expected_manifest_with_retry(
            tmp_path,
            expected_tokens=5,
            expected_metadata=metadata().to_dict(),
            expected_manifest_sha256="d" * 64,
            path_visible=lambda _path: True,
            loader=wrong_loader,
        )
    assert loader_calls == 1
