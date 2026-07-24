from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import moe_sae.gate1_audit as audit
from moe_sae.artifacts import (
    DEFAULT_CAPTURE_SITE,
    HIDDEN_SIZE,
    ArtifactDataset,
)
from moe_sae.artifacts import (
    SCHEMA_VERSION as CAPTURE_SCHEMA_VERSION,
)
from moe_sae.audit_checkpoint import RestoredAuditCheckpoint
from moe_sae.config import Gemma4Config
from moe_sae.diagnostics import SequenceSSEAccumulator, apply_route_mode
from moe_sae.distributed import DistributedBatch, DistributedConfig
from moe_sae.gate1_protocol import (
    Gate1Protocol,
    TokenRange,
    load_gate1_protocol,
    validate_gate1_protocol,
)
from moe_sae.model import CaptureBatch
from moe_sae.sae import SAEConfig

REPOSITORY = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPOSITORY / "configs/gemma4_gate1_audit.yaml"


def _fixture_manifest(
    protocol: Gate1Protocol,
    *,
    source: str,
    metadata_split: str | None = None,
    capture_site: str = DEFAULT_CAPTURE_SITE,
) -> dict[str, Any]:
    config = Gemma4Config()
    train = source == "train"
    tokens = protocol.capture.train_tokens if train else protocol.capture.eval_tokens
    metadata = {
        "config_sha256": protocol.capture.config_sha256,
        "checkpoint_sha256": protocol.capture.checkpoint_sha256,
        "split": metadata_split or source,
        "maxtext_commit": config.maxtext_commit,
        "target_layer_one_indexed": config.target_layer_one_indexed,
        "capture_site": capture_site,
    }
    manifest = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "sealed": True,
        "metadata_sha256": audit._sha256(metadata),
        "metadata": metadata,
        "total_tokens": tokens,
        "statistics": {
            "count": tokens,
            "sum": [0.0] * HIDDEN_SIZE,
        },
        "shards": [],
    }
    manifest["manifest_sha256"] = audit._sha256(manifest)
    return manifest


def _small_protocol(
    *,
    eval_metadata_split: str = "eval",
    eval_capture_site: str = DEFAULT_CAPTURE_SITE,
) -> Gate1Protocol:
    value = load_gate1_protocol(PROTOCOL_PATH).to_mapping()
    value["splits"]["train_stratified"]["ranges"] = [[0, 16]]
    provisional = validate_gate1_protocol(value)
    train_manifest = _fixture_manifest(provisional, source="train")
    eval_manifest = _fixture_manifest(
        provisional,
        source="eval",
        metadata_split=eval_metadata_split,
        capture_site=eval_capture_site,
    )
    value["capture"]["train_manifest_sha256"] = train_manifest["manifest_sha256"]
    value["capture"]["eval_manifest_sha256"] = eval_manifest["manifest_sha256"]
    for checkpoint in value["checkpoints"].values():
        checkpoint["checkpoint_identity"]["capture_manifest_sha256"] = train_manifest[
            "manifest_sha256"
        ]
    return validate_gate1_protocol(value)


def _dataset(
    root: Path,
    protocol: Gate1Protocol,
    *,
    source: str,
    metadata_split: str | None = None,
    capture_site: str = DEFAULT_CAPTURE_SITE,
) -> ArtifactDataset:
    train = source == "train"
    expected_manifest_sha256 = (
        protocol.capture.train_manifest_sha256
        if train
        else protocol.capture.eval_manifest_sha256
    )
    manifest = _fixture_manifest(
        protocol,
        source=source,
        metadata_split=metadata_split,
        capture_site=capture_site,
    )
    assert manifest["manifest_sha256"] == expected_manifest_sha256
    return ArtifactDataset(
        root,
        manifest=manifest,
        verify_hashes=False,
        verify_statistics=False,
    )


def _summary(label: str, sequence: int, tokens: int) -> dict[str, Any]:
    accumulator = SequenceSSEAccumulator(label=label)
    accumulator.update(
        np.full((tokens,), sequence, dtype=np.int64),
        np.ones((tokens,), dtype=np.float64),
        np.full((tokens,), 2.0, dtype=np.float64),
    )
    return accumulator.finalize()


def test_original_training_configs_match_every_frozen_checkpoint() -> None:
    protocol = load_gate1_protocol(PROTOCOL_PATH)
    for label, checkpoint in protocol.checkpoints.items():
        cell = audit.Gate1AuditCell(label, "heldout", "true")
        training = audit._original_training_config(protocol, cell)
        assert training.sae.condition == checkpoint.condition
        assert training.seed == checkpoint.seed
        assert training.a_capacity == (
            audit.ORIGINAL_A_TRAINING_CAPACITY if checkpoint.condition == "A" else None
        )
        assert audit._sha256(dataclasses.asdict(training)) == (
            checkpoint.expectation.training_config_sha256
        )


def test_all_banks_changes_visibility_without_changing_parameter_layout() -> None:
    protocol = load_gate1_protocol(PROTOCOL_PATH)
    cell = audit.Gate1AuditCell("A_seed1_50m", "heldout", "all_banks")
    training = audit._original_training_config(protocol, cell)
    scoring = audit._scoring_sae_config(training, "all_banks")

    assert training.sae.condition == "A"
    assert scoring.condition == "F"
    assert scoring.total_features == training.sae.total_features == 131_072
    assert scoring.parameter_count == training.sae.parameter_count
    assert scoring.candidates_per_token == 131_072


def test_only_declared_matrix_cells_are_accepted() -> None:
    protocol = load_gate1_protocol(PROTOCOL_PATH)
    _, selected = audit._validated_protocol_and_cell(
        protocol,
        checkpoint_label="A_seed1_50m",
        split_name="heldout",
        route_mode="shuffled",
    )
    assert selected.route_mode == "shuffled"

    with pytest.raises(audit.Gate1AuditError, match="not a declared"):
        audit._validated_protocol_and_cell(
            protocol,
            checkpoint_label="B_seed1_50m",
            split_name="heldout",
            route_mode="shuffled",
        )


def test_capture_identity_uses_logical_eval_and_binds_capture_site(
    tmp_path: Path,
) -> None:
    protocol = _small_protocol()
    train = _dataset(tmp_path / "train", protocol, source="train")
    evaluation = _dataset(tmp_path / "eval", protocol, source="eval")

    train_identity, eval_identity, output, mean = audit._validate_datasets_and_output(
        protocol,
        train,
        evaluation,
        tmp_path / "output",
    )
    assert train_identity["split"] == "train"
    assert eval_identity["split"] == "eval"
    assert eval_identity["capture_site"] == DEFAULT_CAPTURE_SITE
    assert output == (tmp_path / "output").resolve()
    assert mean.shape == (HIDDEN_SIZE,)

    wrong_split_protocol = _small_protocol(eval_metadata_split="validation")
    wrong_logical_split = _dataset(
        tmp_path / "wrong-eval",
        wrong_split_protocol,
        source="eval",
        metadata_split="validation",
    )
    with pytest.raises(audit.Gate1AuditError, match="metadata split mismatch"):
        audit._validate_datasets_and_output(
            wrong_split_protocol,
            _dataset(
                tmp_path / "wrong-train",
                wrong_split_protocol,
                source="train",
            ),
            wrong_logical_split,
            tmp_path / "output-2",
        )

    wrong_site_protocol = _small_protocol(eval_capture_site="wrong.site")
    wrong_site = _dataset(
        tmp_path / "wrong-site",
        wrong_site_protocol,
        source="eval",
        capture_site="wrong.site",
    )
    with pytest.raises(audit.Gate1AuditError, match="capture_site mismatch"):
        audit._validate_datasets_and_output(
            wrong_site_protocol,
            _dataset(
                tmp_path / "wrong-site-train",
                wrong_site_protocol,
                source="train",
            ),
            wrong_site,
            tmp_path / "output-3",
        )


def test_runtime_initializes_distributed_before_any_backend_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = False

    def initialize() -> None:
        nonlocal initialized
        initialized = True

    class Device:
        platform = "tpu"

        def __init__(self, process_index: int) -> None:
            self.process_index = process_index

    devices = tuple(Device(process) for process in range(4) for _ in range(4))

    def require_initialized(value: Any) -> Any:
        assert initialized
        return value

    monkeypatch.setattr(audit, "initialize_jax_distributed", initialize)
    monkeypatch.setattr(audit.jax, "process_count", lambda: require_initialized(4))
    monkeypatch.setattr(audit.jax, "local_device_count", lambda: require_initialized(4))
    monkeypatch.setattr(audit.jax, "device_count", lambda: require_initialized(16))
    monkeypatch.setattr(audit.jax, "devices", lambda: require_initialized(devices))
    monkeypatch.setattr(audit.jax, "process_index", lambda: require_initialized(0))
    monkeypatch.setattr(
        audit,
        "create_mesh",
        lambda _distributed, *, devices: SimpleNamespace(
            shape={"data": 4, "feature": 4},
            devices=np.asarray(devices, dtype=object).reshape(4, 4),
        ),
    )
    monkeypatch.setattr(audit, "_checkpoint_phase_decision", lambda *_args: None)

    runtime = audit._create_exact_runtime()

    assert initialized is True
    assert runtime.process_index == 0
    assert runtime.to_mapping()["mesh_shape"] == {"data": 4, "feature": 4}


class _TinyDataset:
    def __init__(self, batches: list[CaptureBatch]) -> None:
        self.batches = batches
        self.requests: list[tuple[int, int]] = []

    def iter_token_range(self, start: int, stop: int):
        self.requests.append((start, stop))
        yield from self.batches


def _tiny_capture(rows: int, *, offset: int = 0) -> CaptureBatch:
    activation = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4) / 100.0
    ids = np.stack(
        (
            np.arange(rows, dtype=np.int32) % 8,
            (np.arange(rows, dtype=np.int32) + 3) % 8,
        ),
        axis=1,
    )
    return CaptureBatch(
        activation=activation,
        expert_ids=ids,
        router_weights=np.full((rows, 2), 0.5, dtype=np.float32),
        sequence_id=(offset + np.arange(rows, dtype=np.int64)) // 7,
        position=np.arange(rows, dtype=np.uint16),
    )


def test_static_group_packing_preserves_sequences_and_shuffles_routes() -> None:
    config = SAEConfig(
        condition="A",
        hidden_dim=4,
        num_experts=8,
        top_routed=2,
        bank_width=4,
        k=2,
        parameter_dtype="float32",
        warmup_steps=0,
        total_steps=1,
    )
    first = _tiny_capture(700)
    second = _tiny_capture(327, offset=700)
    dataset = _TinyDataset([first, second])
    groups = list(
        audit._iter_local_groups(
            dataset,  # type: ignore[arg-type]
            (TokenRange(0, 1_027),),
            config,
            route_mode="shuffled",
            route_key=17,
        )
    )

    assert len(groups) == audit._expected_groups((TokenRange(0, 1_027),)) == 1
    group = groups[0]
    assert group.batch.activation.shape == (4, 1_024, 4)
    assert group.batch.expert_ids.shape == (4, 1_024, 2)
    assert group.sequence_ids.shape == (4, 1_024)
    assert int(np.count_nonzero(group.valid_mask)) == 1_027
    expected_sequences = np.concatenate(
        (np.asarray(first.sequence_id), np.asarray(second.sequence_id))
    )
    assert np.array_equal(group.sequence_ids[group.valid_mask], expected_sequences)
    true_ids = np.concatenate(
        (np.asarray(first.expert_ids), np.asarray(second.expert_ids))
    )
    expected_ids = apply_route_mode(
        true_ids,
        expected_sequences,
        mode="shuffled",
        seed=17,
        num_experts=8,
    )
    assert np.array_equal(group.batch.expert_ids[group.valid_mask], expected_ids)
    assert dataset.requests == [(0, 1_027)]


def test_transformed_capacity_uses_global_max_margin_and_rounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = np.tile(np.arange(8, dtype=np.int32), (4, 1_024, 1))
    group = audit.LocalAuditGroup(
        batch=DistributedBatch(
            np.zeros((4, 1_024, 1), dtype=np.float32),
            ids,
            np.ones((4, 1_024), dtype=bool),
        ),
        sequence_ids=np.zeros((4, 1_024), dtype=np.int64),
        valid_mask=np.ones((4, 1_024), dtype=bool),
    )
    monkeypatch.setattr(audit, "_checkpoint_phase_decision", lambda *_args: None)
    monkeypatch.setattr(
        audit,
        "_collective_max_int",
        lambda _value, *, operation: 1_030,
    )

    evidence = audit._derive_a_capacity(iter((group,)), num_experts=128)

    assert evidence.local_observed_max == 1_024
    assert evidence.global_observed_max == 1_030
    assert evidence.capacity == 1_136
    assert evidence.capacity % 8 == 0


def test_prefetches_every_process_local_range_collectively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class Dataset:
        def prefetch_token_range(self, start: int, stop: int) -> dict[str, int]:
            calls.append((start, stop))
            return {"objects": 2, "bytes": stop - start}

    monkeypatch.setattr(audit, "_checkpoint_phase_decision", lambda *_args: None)
    evidence = audit._prefetch_local_ranges(
        Dataset(),  # type: ignore[arg-type]
        (TokenRange(0, 4), TokenRange(20, 28)),
    )

    assert calls == [(0, 4), (20, 28)]
    assert evidence == {
        "ranges": [[0, 4], [20, 28]],
        "objects": 4,
        "bytes": 12,
        "performance_claim": False,
    }


def test_merged_cell_matches_frozen_report_schema() -> None:
    protocol = _small_protocol()
    cell = audit.Gate1AuditCell("B_seed1_50m", "train_stratified", "true")
    training = audit._original_training_config(protocol, cell)
    cell_identity = {
        "test_identity": "shared",
        "consumer_environment": {"sha256": "a" * 64},
    }
    cell_sha256 = audit._sha256(cell_identity)
    fragments = []
    for process_index in range(4):
        fragments.append(
            {
                "identity": {"cell_sha256": cell_sha256},
                "cell_identity": cell_identity,
                "result_sha256": f"{process_index + 1:064x}",
                "checkpoint": {"descriptor_sha256": "d" * 64},
                "audit_capacity": None,
                "invariants": {
                    "exact_l0": True,
                    "topk_valid": True,
                    "capacity_overflow": False,
                    "dead_features": 10,
                    "total_features": training.sae.total_features,
                    "fired_mask_sha256": "e" * 64,
                },
                "sequence_summary": _summary(cell_sha256, process_index, 4),
            }
        )

    result = audit._merged_cell_payload(
        protocol,
        cell,
        cell_sha256,
        training.sae,
        fragments,
    )

    assert set(result) == audit._CELL_RESULT_FIELDS
    assert result["schema_version"] == audit.GATE1_CELL_RESULT_SCHEMA
    assert result["checkpoint_descriptor_sha256"] == "d" * 64
    assert result["capture_manifest_sha256"] == (protocol.capture.train_manifest_sha256)
    assert result["ranges"] == [[0, 16]]
    assert result["tokens"] == 16
    assert result["route_permutation_domain"] is None
    assert result["route_permutation_key"] is None
    assert result["sequence_sse"]["tokens"] == 16
    assert result["dead_features"] == 10


def test_process_fragment_write_is_idempotent_and_tamper_evident(
    tmp_path: Path,
) -> None:
    identity = {
        "cell_sha256": "a" * 64,
        "process_index": 0,
        "process_count": 4,
        "process_ranges": [[0, 4]],
        "process_tokens": 4,
    }
    payload = {
        "schema_version": audit.GATE1_PROCESS_RESULT_SCHEMA,
        "identity": identity,
        "value": 1,
    }
    payload["result_sha256"] = audit._sha256(payload)
    path = tmp_path / "result.json"

    assert audit._write_once(path, payload) == payload
    assert audit._write_once(path, payload) == payload
    changed = dict(payload)
    changed["value"] = 2
    path.write_text(
        __import__("json").dumps(changed),
        encoding="utf-8",
    )
    with pytest.raises(audit.Gate1AuditError, match="hash mismatch"):
        audit._read_existing(path, expected_identity=identity)


def test_runner_restores_scores_writes_and_reuses_one_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _small_protocol()
    train = _dataset(tmp_path / "train", protocol, source="train")
    evaluation = _dataset(tmp_path / "eval", protocol, source="eval")
    distributed = DistributedConfig(
        data_axis_size=4,
        feature_axis_size=4,
        microbatches_per_update=4,
    )
    runtime = audit.Gate1Runtime(
        mesh=SimpleNamespace(shape={"data": 4, "feature": 4}),
        distributed=distributed,
        process_index=0,
    )
    restored_calls: list[tuple[Any, ...]] = []
    scored_batches: list[DistributedBatch] = []
    merged_calls: list[str] = []
    events: list[str] = []

    def iter_token_range(self: ArtifactDataset, start: int, stop: int):
        assert self is train
        assert (start, stop) == (0, 4)
        yield CaptureBatch(
            activation=np.ones((4, HIDDEN_SIZE), dtype=np.float32),
            expert_ids=np.tile(np.arange(8, dtype=np.int32), (4, 1)),
            router_weights=np.full((4, 8), 0.125, dtype=np.float32),
            sequence_id=np.arange(100, 104, dtype=np.int64),
            position=np.arange(4, dtype=np.uint16),
        )

    def restore(*args: Any, **kwargs: Any) -> RestoredAuditCheckpoint:
        events.append("restore")
        restored_calls.append((*args, kwargs))
        return RestoredAuditCheckpoint(
            state=SimpleNamespace(params="restored-params"),
            metadata={},
            descriptor={
                "descriptor_sha256": "1" * 64,
                "checkpoint_header_sha256": "2" * 64,
                "checkpoint_identity_sha256": "3" * 64,
                "custom_metadata_sha256": "4" * 64,
            },
        )

    class Scorer:
        def __init__(
            self,
            _mesh: Any,
            config: SAEConfig,
            _distributed: DistributedConfig,
            *,
            a_capacity: int | None,
        ) -> None:
            assert config.condition == "B"
            assert a_capacity is None
            self.total_features = config.total_features

        def score(
            self,
            params: Any,
            batch: DistributedBatch,
            _mean: Any,
            *,
            process_local: bool,
        ) -> Any:
            assert params == "restored-params"
            assert process_local is True
            scored_batches.append(batch)
            return SimpleNamespace(batch=batch, total_features=self.total_features)

    def token_scores(scores: Any):
        valid = np.asarray(scores.batch.valid_mask)
        return (
            valid.astype(np.float32),
            valid.astype(np.float32) * 2.0,
            valid.astype(np.int32) * 64,
            np.ones(valid.shape, dtype=bool),
        )

    def feature_state(scores: Any):
        fired = np.zeros((scores.total_features,), dtype=bool)
        fired[:4] = True
        return fired, False

    def prefetch(_dataset: ArtifactDataset, ranges: Any) -> dict[str, Any]:
        events.append("prefetch")
        return {
            "ranges": [item.to_list() for item in ranges],
            "objects": 1,
            "bytes": 1,
            "performance_claim": False,
        }

    monkeypatch.setattr(ArtifactDataset, "iter_token_range", iter_token_range)
    monkeypatch.setattr(audit, "_create_exact_runtime", lambda: runtime)
    monkeypatch.setattr(audit, "_checkpoint_phase_decision", lambda *_args: None)
    monkeypatch.setattr(audit, "_digest_consensus", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        audit, "_collective_all", lambda value, *, operation: bool(value)
    )
    monkeypatch.setattr(audit, "_barrier", lambda _label: None)
    monkeypatch.setattr(audit, "_prefetch_local_ranges", prefetch)
    monkeypatch.setattr(
        audit,
        "init_distributed_state",
        lambda *_args, **_kwargs: "template",
    )
    monkeypatch.setattr(audit, "restore_audit_checkpoint_payload", restore)
    monkeypatch.setattr(audit, "Gate1DistributedScorer", Scorer)
    monkeypatch.setattr(audit, "process_local_token_scores", token_scores)
    monkeypatch.setattr(audit, "process_local_feature_state", feature_state)
    monkeypatch.setattr(
        audit,
        "_publish_merged_cell",
        lambda _output, _protocol, _cell, digest, _scoring, _runtime: (
            merged_calls.append(digest) or {"merged": digest}
        ),
    )

    kwargs = {
        "checkpoint_label": "B_seed1_50m",
        "split_name": "train_stratified",
        "route_mode": "true",
    }
    first = audit.run_gate1_audit_cell(
        protocol,
        train,
        evaluation,
        tmp_path / "output",
        **kwargs,
    )
    second = audit.run_gate1_audit_cell(
        protocol,
        train,
        evaluation,
        tmp_path / "output",
        **kwargs,
    )

    assert first == second
    assert events == ["prefetch", "restore"]
    assert len(restored_calls) == 1
    assert restored_calls[0][1] == "template"
    restored_training = restored_calls[0][3]
    assert restored_training.sae.condition == "B"
    assert restored_training.seed == 1
    assert restored_training.a_capacity is None
    assert len(scored_batches) == 1
    assert scored_batches[0].activation.shape == (4, 1_024, HIDDEN_SIZE)
    assert int(np.count_nonzero(scored_batches[0].valid_mask)) == 4
    assert len(merged_calls) == 2

    fragments = list((tmp_path / "output").rglob("process-00.json"))
    assert len(fragments) == 1
    payload = audit._read_existing(
        fragments[0],
        expected_identity={
            "cell_sha256": merged_calls[0],
            "process_index": 0,
            "process_count": 4,
            "process_ranges": [[0, 4]],
            "process_tokens": 4,
        },
    )
    assert payload is not None
    assert payload["checkpoint"]["descriptor_sha256"] == "1" * 64
    assert payload["invariants"]["valid_tokens"] == 4
    assert payload["invariants"]["dead_features"] == 8_188
    assert payload["sequence_summary"]["tokens"] == 4
