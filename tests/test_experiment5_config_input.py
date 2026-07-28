from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    SCHEMA_VERSION,
    TOP_K_EXPERTS,
    ContributionBatch,
    bf16_bits_to_float32,
    build_recomposition_sidecar,
    float32_to_bf16_bits,
)
from moe_sae.contribution_recomposition import (
    SUPPORTED_ACCUMULATION_ORDER,
    recompose_gemma4,
)
from moe_sae.experiment5_config import (
    Experiment5ConfigError,
    experiment5_randomization_key,
    load_experiment5_config,
)
from moe_sae.experiment5_hybrid import production_experiment5_layout
from moe_sae.experiment5_input import (
    Experiment5ProcessInput,
    merge_experiment5_preprocessing,
)
from moe_sae.contribution_training_worker import (
    TrainingWorkerPlan,
    direct_resident_runtime_limit_seconds,
    prefetch_runtime_limit_seconds,
)


def _sidecar() -> dict:
    ones = float32_to_bf16_bits(np.ones((HIDDEN_SIZE,), np.float32))
    return build_recomposition_sidecar(
        checkpoint_sha256="1" * 64,
        maxtext_fork_commit="2" * 40,
        source_fingerprints={
            "checkpoint": "1" * 64,
            "fork_tree": "3" * 64,
        },
        routed_rmsnorm_scale=ones,
        shared_rmsnorm_scale=ones,
        outer_ffn_rmsnorm_scale=ones,
        layer_scalar=float32_to_bf16_bits(np.ones((1,), np.float32)),
        accumulation_order=SUPPORTED_ACCUMULATION_ORDER,
    )


def _wire(tokens: int) -> ContributionBatch:
    base = np.linspace(0.1, 1.0, HIDDEN_SIZE, dtype=np.float32)
    expert = np.stack(
        [
            np.stack(
                [base * np.float32(1 + row + slot / 8) for slot in range(8)]
            )
            for row in range(tokens)
        ]
    )
    shared = np.stack(
        [base * np.float32(0.5 + row / 10) for row in range(tokens)]
    )
    weights = np.asarray(
        [0.25, 0.1875, 0.15625, 0.125, 0.09375, 0.078125, 0.0625, 0.046875],
        np.float32,
    )
    ids = np.stack(
        [
            (np.arange(TOP_K_EXPERTS, dtype=np.uint8) + 3 * row) % 128
            for row in range(tokens)
        ]
    )
    return ContributionBatch(
        expert_output=float32_to_bf16_bits(expert),
        expert_ids=ids,
        router_weights=float32_to_bf16_bits(
            np.broadcast_to(weights, (tokens, TOP_K_EXPERTS))
        ),
        shared_output=float32_to_bf16_bits(shared),
        sequence_id=np.asarray([100 + row // 2 for row in range(tokens)], np.int64),
        position=np.arange(tokens, dtype=np.uint16),
    )


class _Dataset:
    def __init__(self, value: ContributionBatch):
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
                name: np.ascontiguousarray(np.asarray(getattr(self.value, name))[start:end])
                for name in ContributionBatch.__dataclass_fields__
            }
        )


def test_checked_in_configs_load_as_strict_worker_views():
    sae = load_experiment5_config(
        "configs/gemma4_hybrid_sae_v1.yaml",
        expected_panel="sae",
    )
    transcoder = load_experiment5_config(
        "configs/gemma4_hybrid_transcoder_v1.yaml",
        expected_panel="transcoder",
    )

    assert sae.phase("routed_shared_hybrid_10m").planned_updates == 611
    assert transcoder.phase("routed_shared_hybrid_10m").seeds == (0, 1, 2)
    assert tuple(item.condition for item in sae.layouts) == ("H", "R", "B", "F")
    assert sae.layouts[0].total_features == sae.layouts[-1].total_features
    plan = TrainingWorkerPlan.from_config(
        transcoder,
        phase="routed_shared_hybrid_10m",
        condition="H",
        seed=2,
    )
    assert plan.total_tokens == 10_000_000
    assert plan.checkpoint_tokens[-1] == 10_000_000
    assert plan.randomization_key == experiment5_randomization_key(2)
    assert experiment5_randomization_key(0) != experiment5_randomization_key(1)
    assert (
        direct_resident_runtime_limit_seconds(
            config_path="configs/gemma4_hybrid_transcoder_v1.yaml",
            phase="routed_shared_hybrid_10m",
            accelerator_type="v5litepod-16",
        )
        == 43_200
    )
    assert (
        prefetch_runtime_limit_seconds(
            config_path="configs/gemma4_hybrid_transcoder_v1.yaml",
            phase="routed_shared_hybrid_10m",
            accelerator_type="v5litepod-16",
        )
        == 1_800
    )


def test_config_rejects_scientific_layout_drift(tmp_path):
    raw = yaml.safe_load(
        open("configs/gemma4_hybrid_sae_v1.yaml", encoding="utf-8")
    )
    changed = copy.deepcopy(raw)
    changed["layouts"][0]["routed_bank_width"] = 256
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")

    with pytest.raises(Experiment5ConfigError, match="routed_bank_width"):
        load_experiment5_config(path, expected_panel="sae")


def test_sae_adapter_derives_shared_rows_and_stable_randomized_banks():
    wire = _wire(5)
    sidecar = _sidecar()
    source = Experiment5ProcessInput(
        _Dataset(wire),
        sidecar,
        production_experiment5_layout("R", randomization_key=7),
        panel="sae",
        total_tokens=5,
        global_microbatch_tokens=4,
        microbatches_per_update=2,
    )

    update = next(source.iter_updates())
    first, tail = update
    assert first.expert_output.shape == (4, 8, HIDDEN_SIZE)
    assert int(first.valid_mask.sum()) == 4
    assert int(tail.valid_mask.sum()) == 1
    assert not np.array_equal(first.bank_ids, wire.expert_ids[:4])
    assert np.count_nonzero(first.routed_input) == 0
    assert np.isfinite(first.shared_normalized).all()
    assert np.count_nonzero(tail.shared_normalized[1:]) == 0
    # One packed sequence uses one expert->random bank bijection.
    for expert in np.intersect1d(wire.expert_ids[0], wire.expert_ids[1]):
        left = first.bank_ids[0, np.flatnonzero(wire.expert_ids[0] == expert)[0]]
        right = first.bank_ids[1, np.flatnonzero(wire.expert_ids[1] == expert)[0]]
        assert left == right


def test_preprocessing_receipts_merge_to_exact_full_target_mean_and_sst():
    wire = _wire(6)
    dataset = _Dataset(wire)
    sidecar = _sidecar()
    layout = production_experiment5_layout("H", randomization_key=0)
    sources = [
        Experiment5ProcessInput(
            dataset,
            sidecar,
            layout,
            panel="sae",
            total_tokens=6,
            global_microbatch_tokens=4,
            microbatches_per_update=1,
            process_count=2,
            process_index=index,
        )
        for index in range(2)
    ]
    merged = merge_experiment5_preprocessing(
        tuple(source.scan_local_preprocessing() for source in sources),
        margin_fraction=0.10,
    )
    target = bf16_bits_to_float32(
        recompose_gemma4(wire, sidecar).outer_ffn_post_norm_bf16_bits
    )

    np.testing.assert_allclose(
        merged["full_target_mean"],
        np.mean(target, axis=0, dtype=np.float64).astype(np.float32),
        rtol=0,
        atol=0,
    )
    centered = target.astype(np.float64) - np.mean(
        target,
        axis=0,
        dtype=np.float64,
    )
    assert merged["full_target_sst"] == pytest.approx(
        float(np.sum(centered * centered, dtype=np.float64)),
        rel=1e-9,
    )
    assert merged["bank_capacity"] is not None
    assert len(merged["routed_bank_component_counts"]) == 128
    assert sum(merged["routed_bank_component_counts"]) == 6 * 8
    milestones = merged["routed_bank_component_counts_by_training_tokens"]
    assert [tokens for tokens, _counts in milestones] == [6]
    assert sum(milestones[0][1]) == 6 * 8
