from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    SCHEMA_VERSION,
    TOP_K_EXPERTS,
    ContributionBatch,
    canonical_sha256,
    float32_to_bf16_bits,
)
from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_transcoder_config import (
    PHASE_NAME,
    PROTOCOL_ID,
    SCHEMA_VERSION as RUN_CONFIG_SCHEMA_VERSION,
    ContributionTranscoderConfigError,
    load_contribution_transcoder_config,
)
from moe_sae.contribution_transcoder_input import (
    SOURCE_SCHEMA_VERSION,
    ContributionTranscoderInputError,
    ContributionTranscoderProcessInput,
    routed_input_mean,
)
from moe_sae.contribution_training_worker import (
    TRANSCODER_PREFETCH_RECEIPT_SCHEMA_VERSION,
    ContributionTrainingWorkerError,
    prefetch_transcoder_process_lane,
    validate_transcoder_prefetch_receipt,
)

TRAIN_SOURCE_OVERLAY_SHA256 = (
    "8d947629e57d64d054a3086e2729090eed1e9f6afc4568b76e5c90da5f378e3c"
)
EVAL_SOURCE_OVERLAY_SHA256 = (
    "f68e2fe6e4cf568e694925697dd5d65fe10160a52d357c641892de85ee5c3da7"
)


def _target_batch(tokens: int) -> ContributionBatch:
    expert_ids = np.stack(
        [
            (np.arange(TOP_K_EXPERTS, dtype=np.uint8) + row * 8) % 128
            for row in range(tokens)
        ]
    )
    weights = np.full(
        (tokens, TOP_K_EXPERTS),
        1.0 / TOP_K_EXPERTS,
        np.float32,
    )
    return ContributionBatch(
        expert_output=float32_to_bf16_bits(
            np.arange(
                tokens * TOP_K_EXPERTS * HIDDEN_SIZE,
                dtype=np.float32,
            ).reshape(tokens, TOP_K_EXPERTS, HIDDEN_SIZE)
            / 1_000,
        ),
        expert_ids=expert_ids,
        router_weights=float32_to_bf16_bits(weights),
        shared_output=float32_to_bf16_bits(np.zeros((tokens, HIDDEN_SIZE), np.float32)),
        sequence_id=np.asarray([10 + index // 2 for index in range(tokens)], np.int64),
        position=np.arange(tokens, dtype=np.uint16),
    )


class _TargetDataset:
    def __init__(self, value: ContributionBatch) -> None:
        self.value = value
        self.total_tokens = value.tokens
        self.manifest_sha256 = "a" * 64
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "manifest_sha256": self.manifest_sha256,
        }

    def iter_token_range(self, start: int, end: int):
        yield ContributionBatch(
            **{
                name: np.ascontiguousarray(
                    np.asarray(getattr(self.value, name))[start:end]
                )
                for name in ContributionBatch.__dataclass_fields__
            }
        )


class _SourceBatch:
    def __init__(
        self,
        routed_input: np.ndarray,
        sequence_id: np.ndarray,
        position: np.ndarray,
    ) -> None:
        self.routed_input = routed_input
        self.sequence_id = sequence_id
        self.position = position


class _SourceDataset:
    def __init__(self, target: _TargetDataset) -> None:
        source = (
            np.arange(target.total_tokens * HIDDEN_SIZE, dtype=np.float32).reshape(
                target.total_tokens, HIDDEN_SIZE
            )
            / 100
        )
        self.wire = float32_to_bf16_bits(source)
        self.sequence_id = target.value.sequence_id.copy()
        self.position = target.value.position.copy()
        self.total_tokens = target.total_tokens
        self.manifest_sha256 = "b" * 64
        decoded = source.astype(np.float64)
        self.manifest = {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "manifest_sha256": self.manifest_sha256,
            "target_manifest": {
                "manifest_sha256": target.manifest_sha256,
            },
            "metadata": {
                "target_manifest_sha256": target.manifest_sha256,
            },
            "statistics": {
                "count": target.total_tokens,
                "sum": decoded.sum(axis=0, dtype=np.float64).tolist(),
                "squared_norm": float(np.square(decoded).sum(dtype=np.float64)),
            },
        }

    def iter_token_range(self, start: int, end: int):
        yield _SourceBatch(
            np.ascontiguousarray(self.wire[start:end]),
            np.ascontiguousarray(self.sequence_id[start:end]),
            np.ascontiguousarray(self.position[start:end]),
        )


def _paired(tokens: int = 6) -> tuple[_TargetDataset, _SourceDataset]:
    target = _TargetDataset(_target_batch(tokens))
    return target, _SourceDataset(target)


def test_paired_input_uses_only_overlay_as_encoder_operand_and_binds_both_hashes():
    target, source = _paired()
    layout = ContributionSAEConfig(
        condition="A",
        bank_width=8,
        k_per_contribution=2,
    )
    adapter = ContributionTranscoderProcessInput(
        source,
        target,
        layout,
        total_tokens=6,
        global_microbatch_tokens=4,
        microbatches_per_update=2,
    )

    first, tail = next(adapter.iter_updates())

    assert first.routed_input.shape == (4, HIDDEN_SIZE)
    assert first.expert_output.shape == (4, TOP_K_EXPERTS, HIDDEN_SIZE)
    np.testing.assert_array_equal(first.bank_ids, target.value.expert_ids[:4])
    assert not hasattr(first, "expert_ids")
    assert adapter.receipt.source_manifest_sha256 == "b" * 64
    assert adapter.receipt.target_manifest_sha256 == "a" * 64
    assert int(tail.valid_mask.sum()) == 2
    assert np.count_nonzero(tail.routed_input[2:]) == 0


@pytest.mark.parametrize("field", ["sequence_id", "position"])
def test_pair_fails_on_elementwise_token_key_mismatch(field: str):
    target, source = _paired()
    value = getattr(source, field).copy()
    value[[1, 2]] = value[[2, 1]]
    setattr(source, field, value)
    adapter = ContributionTranscoderProcessInput(
        source,
        target,
        ContributionSAEConfig(
            condition="B",
            bank_width=8,
            k_per_contribution=2,
        ),
        total_tokens=6,
        global_microbatch_tokens=4,
    )

    with pytest.raises(
        ContributionTranscoderInputError,
        match=f"source and target {field} differ elementwise",
    ):
        next(adapter.iter_updates())


def test_pair_rejects_wrong_parent_even_when_tokens_match():
    target, source = _paired()
    source.manifest["target_manifest"]["manifest_sha256"] = "c" * 64

    with pytest.raises(
        ContributionTranscoderInputError,
        match="does not bind",
    ):
        ContributionTranscoderProcessInput(
            source,
            target,
            ContributionSAEConfig(
                condition="A",
                bank_width=8,
                k_per_contribution=2,
            ),
            total_tokens=6,
        )


def test_capacity_receipt_is_rebound_to_paired_input_identity():
    target, source = _paired()
    adapter = ContributionTranscoderProcessInput(
        source,
        target,
        ContributionSAEConfig(
            condition="A",
            bank_width=8,
            k_per_contribution=2,
        ),
        total_tokens=6,
        global_microbatch_tokens=4,
    )

    receipt = adapter.scan_local_capacity(margin_fraction=0.10)

    assert receipt.input_receipt_sha256 == adapter.receipt.receipt_sha256
    assert receipt.capacity >= receipt.observed_maximum


def test_source_and_target_means_are_independent_identities():
    target, source = _paired()

    source_mean, source_sha = routed_input_mean(source)
    changed_target = replace(
        target.value,
        expert_output=float32_to_bf16_bits(
            np.full(
                target.value.expert_output.shape,
                7.0,
                dtype=np.float32,
            )
        ),
    )
    target.value = changed_target
    source_mean_after, source_sha_after = routed_input_mean(source)

    np.testing.assert_array_equal(source_mean_after, source_mean)
    assert source_sha_after == source_sha


def test_strict_transcoder_config_exposes_one_pilot_and_canonical_hash():
    config = load_contribution_transcoder_config("configs/gemma4_transcoder_v2.yaml")
    assert config.encoder_activation == "softplus"
    assert (
        config.raw["schema_version"]
        == "gemma4-contribution-transcoder-run-config-v2"
        == RUN_CONFIG_SCHEMA_VERSION
    )
    assert (
        config.raw["protocol_id"]
        == "gemma4-contribution-transcoder-v2-20260727"
        == PROTOCOL_ID
    )
    assert config.raw["source_overlay"]["schema_version"] == SOURCE_SCHEMA_VERSION
    assert (
        config.parents["train_source_overlay_manifest_sha256"]
        == TRAIN_SOURCE_OVERLAY_SHA256
    )
    assert (
        config.parents["eval_source_overlay_manifest_sha256"]
        == EVAL_SOURCE_OVERLAY_SHA256
    )

    assert config.phase(PHASE_NAME).planned_updates == 123
    assert config.phase(PHASE_NAME).checkpoint_tokens == (2_000_000,)
    assert [item.condition for item in config.layouts_for_phase(PHASE_NAME)] == [
        "A",
        "R",
        "B",
        "F",
    ]
    assert config.runtime.accelerator_min_useful_mfu == 0.10
    assert len(config.config_hash) == 64


def test_v2_config_rejects_a_different_overlay_manifest(tmp_path: Path):
    checked_in = Path("configs/gemma4_transcoder_v2.yaml").read_text(
        encoding="utf-8"
    )
    changed = checked_in.replace(
        TRAIN_SOURCE_OVERLAY_SHA256,
        "0" * 64,
    )
    assert changed != checked_in
    path = tmp_path / "different-overlay.yaml"
    path.write_text(changed, encoding="utf-8")

    with pytest.raises(
        ContributionTranscoderConfigError,
        match="parents.train_source_overlay_manifest_sha256",
    ):
        load_contribution_transcoder_config(path)


def test_training_prefetch_rejects_nonpinned_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = load_contribution_transcoder_config(
        "configs/gemma4_transcoder_v2.yaml"
    )

    class Target:
        total_tokens = 2_000_000
        manifest_sha256 = config.parents["train_target_manifest_sha256"]
        manifest = {"manifest_sha256": manifest_sha256}

        def __init__(self, *_args, **_kwargs):
            pass

    class Source:
        total_tokens = 2_000_000
        manifest_sha256 = "0" * 64

        def __init__(self, *_args, **_kwargs):
            pass

    class Cache:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(
        "moe_sae.contribution_storage.ContributionDataset",
        Target,
    )
    monkeypatch.setattr(
        "moe_sae.contribution_transcoder_storage.RoutedInputDataset",
        Source,
    )
    monkeypatch.setattr("moe_sae.object_store.ShardCache", Cache)

    with pytest.raises(
        ContributionTrainingWorkerError,
        match="config-pinned parent",
    ):
        prefetch_transcoder_process_lane(
            config=config,
            phase=PHASE_NAME,
            source_revision="1" * 40,
            capture_control_root=tmp_path / "target",
            object_prefix="gs://bucket/target",
            source_overlay_control_root=tmp_path / "source",
            source_overlay_object_prefix="gs://bucket/source",
            cache_root=tmp_path / "cache",
            cache_max_bytes=1_000,
            process_count=4,
            process_index=0,
            recomposition_sidecar=None,
            object_store=object(),
        )


def test_paired_prefetch_receipt_names_both_immutable_object_sets():
    unsigned = {
        "schema_version": TRANSCODER_PREFETCH_RECEIPT_SCHEMA_VERSION,
        "config_sha256": "1" * 64,
        "phase": PHASE_NAME,
        "source_revision": "2" * 40,
        "target_manifest_sha256": "3" * 64,
        "source_manifest_sha256": "4" * 64,
        "target_object_prefix": "gs://bucket/targets",
        "source_object_prefix": "gs://bucket/sources",
        "cache_root": "/tmp/cache",
        "cache_max_bytes": 1_000,
        "process_partition": {
            "strategy": "contiguous_process_lanes_v1",
            "process_count": 4,
            "process_index": 0,
            "range_start": 0,
            "range_end": 10,
            "tokens": 10,
        },
        "target_objects": {
            "object_count": 2,
            "object_bytes": 600,
            "object_set_sha256": "5" * 64,
        },
        "source_objects": {
            "object_count": 1,
            "object_bytes": 200,
            "object_set_sha256": "6" * 64,
        },
        "combined_object_count": 3,
        "combined_object_bytes": 800,
        "combined_object_set_sha256": canonical_sha256(
            {"target": "5" * 64, "source": "6" * 64}
        ),
        "elapsed_seconds": 1.5,
        "resident_only_complete": True,
    }
    receipt = {
        **unsigned,
        "receipt_sha256": canonical_sha256(unsigned),
    }

    assert (
        validate_transcoder_prefetch_receipt(receipt)["source_manifest_sha256"]
        == "4" * 64
    )
    corrupted = dict(receipt)
    corrupted["source_manifest_sha256"] = "8" * 64
    with pytest.raises(ContributionTrainingWorkerError, match="self-hash"):
        validate_transcoder_prefetch_receipt(corrupted)
