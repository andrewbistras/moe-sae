from pathlib import Path

import numpy as np
import pytest

from moe_sae.runtime import (
    BenchmarkObservation,
    HARVEST_MESHES,
    _staged_path_accessible,
    build_maxtext_argv,
    choose_harvest_benchmark,
    compact_capture,
    environment_fingerprint,
    estimate_useful_prefix_flops,
    immutable_range_id,
    scientific_token_mask,
    scan_token_shaped_secrets,
    token_quota,
    updates_for_tokens,
)


def _observation(mesh_index: int, *, mfu: float, tokens: int) -> BenchmarkObservation:
    elapsed = 2.0
    peak = 4.4e15
    return BenchmarkObservation(
        mesh=HARVEST_MESHES[mesh_index],
        per_device_sequence_batch=1,
        synchronized_steps=200,
        elapsed_seconds=elapsed,
        device_elapsed_seconds=elapsed,
        persistence_seconds=0.0,
        useful_flops=mfu * peak * elapsed,
        output_tokens=tokens,
        persisted_tokens=tokens,
        max_writer_backlog=1,
    )


def test_benchmark_selection_requires_gate_and_maximizes_persisted_goodput():
    too_low = _observation(0, mfu=0.49, tokens=99_000)
    slower = _observation(1, mfu=0.55, tokens=100_000)
    faster = _observation(2, mfu=0.51, tokens=120_000)
    assert choose_harvest_benchmark([too_low, slower, faster]) == faster
    with pytest.raises(RuntimeError, match="no harvester benchmark"):
        choose_harvest_benchmark([too_low])


def test_benchmark_uses_device_time_for_mfu_and_total_time_for_persistence():
    observation = BenchmarkObservation(
        mesh=HARVEST_MESHES[0],
        per_device_sequence_batch=1,
        synchronized_steps=200,
        elapsed_seconds=10.0,
        device_elapsed_seconds=2.0,
        persistence_seconds=7.0,
        useful_flops=4.4e15,
        output_tokens=1_000,
        persisted_tokens=1_000,
    )
    assert observation.useful_mfu == pytest.approx(0.5)
    assert observation.device_tokens_per_second == pytest.approx(500.0)
    assert observation.persisted_tokens_per_second == pytest.approx(100.0)
    assert observation.persistence_fraction == pytest.approx(0.7)


def test_prefix_flops_are_explicit_and_scale_with_batch():
    one = estimate_useful_prefix_flops(batch_sequences=1)
    four = estimate_useful_prefix_flops(batch_sequences=4)
    assert one > 0
    assert four == 4 * one


def test_maxtext_argv_locks_scientific_capture_settings():
    argv = build_maxtext_argv(
        checkpoint_path="gs://bucket/checkpoint",
        grain_train_files="gs://bucket/c4/train-*",
        grain_eval_files="gs://bucket/c4/validation-*",
        mesh=HARVEST_MESHES[0],
        per_device_sequence_batch=2,
        base_config_path="base.yml",
    )
    joined = "\n".join(argv)
    assert argv[0] == "base.yml"
    assert "model_name=gemma4-26b" in joined
    assert "tokenizer_path=google/gemma-4-26B-A4B" in joined
    assert "max_target_length=1024" in joined
    assert "scan_layers=false" in joined
    assert "use_multimodal=false" in joined
    assert "load_balance_loss_weight=0" in joined
    assert "ici_expert_parallelism=16" in joined


def test_exact_host_quota_and_immutable_ids():
    quotas = [token_quota(10, 3, index) for index in range(3)]
    assert quotas == [4, 3, 3]
    assert sum(quotas) == 10
    assert immutable_range_id("train", 2, 7) == "train-p00002-b000000000007"


def test_scientific_mask_and_common_compaction():
    token_ids = np.array([[2, 10, 11, 1], [0, 12, 13, 14]], dtype=np.int32)
    segmentation = np.array([[1, 1, 1, 1], [0, 2, 2, 0]], dtype=np.int32)
    mask = scientific_token_mask(token_ids, segmentation, excluded_token_ids=(0, 1, 2))
    assert mask.tolist() == [[False, True, True, False], [False, True, True, False]]
    activation = np.arange(8 * 3).reshape(2, 4, 3)
    expert_ids = np.arange(8 * 8).reshape(2, 4, 8)
    compact = compact_capture(
        {"activation": activation, "expert_ids": expert_ids}, mask, limit=3
    )
    assert compact["activation"].shape == (3, 3)
    assert compact["expert_ids"].shape == (3, 8)
    np.testing.assert_array_equal(
        compact["activation"], activation.reshape(-1, 3)[[1, 2, 5]]
    )


def test_token_shaped_secret_scan_reports_path_only(tmp_path: Path):
    clean = tmp_path / "clean.txt"
    clean.write_text("no credentials", encoding="utf-8")
    assert scan_token_shaped_secrets(tmp_path) == []
    suspect = tmp_path / "suspect.txt"
    suspect.write_text("hf_" + "a" * 24, encoding="utf-8")
    assert scan_token_shaped_secrets(tmp_path) == [Path("suspect.txt")]


def test_staged_local_path_probe_requires_a_real_match(tmp_path: Path):
    shard = tmp_path / "train-000.array_record"
    shard.write_bytes(b"staged")
    assert _staged_path_accessible(str(tmp_path / "*.array_record"))[0]
    passed, detail = _staged_path_accessible(str(tmp_path / "missing-*"))
    assert not passed and "matched no files" in detail


def test_update_plan_masks_only_the_final_padding():
    assert updates_for_tokens(50_000_000, 16_384) == (3_052, 3_968)


def test_environment_fingerprint_is_secret_free_and_hashed(tmp_path: Path):
    fingerprint = environment_fingerprint(tmp_path)
    assert len(fingerprint["sha256"]) == 64
    assert "packages" in fingerprint and "devices" in fingerprint
    assert "environment" not in fingerprint
