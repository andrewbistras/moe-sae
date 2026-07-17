from __future__ import annotations

import dataclasses
import json

import pytest

from moe_sae.config import (
    CONDITIONS,
    MAXTEXT_COMMIT,
    MODEL_ID,
    Gemma4Config,
    load_config,
    save_config,
)


def test_approved_defaults_and_derived_contract():
    cfg = Gemma4Config()

    assert cfg.model_id == MODEL_ID
    assert cfg.maxtext_commit == MAXTEXT_COMMIT
    assert cfg.conditions == CONDITIONS
    assert cfg.target_layer_one_indexed == 15
    assert cfg.target_layer_index == 14
    assert cfg.total_features == 131_072
    assert cfg.visible_features == 8_192
    assert cfg.candidate_exposure == pytest.approx(0.0625)
    assert cfg.planned_updates == 3_052
    assert cfg.final_padding_tokens == 3_968
    assert cfg.parameter_count("A") == 738_328_576
    assert cfg.parameter_count("F") == 738_328_576
    assert cfg.width_for_condition("B") == 8_192
    assert cfg.candidates_per_token("A") == cfg.candidates_per_token("B") == 8_192
    assert cfg.candidates_per_token("F") == 131_072
    assert cfg.min_control_free_bytes == 10_000_000_000
    assert cfg.min_local_buffer_free_bytes == 128 * 1024**3
    assert cfg.capture_cache_max_bytes == 100 * 1024**3


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model_id": "wrong/model"}, "model_id is pinned"),
        ({"maxtext_commit": "deadbeef"}, "maxtext_commit is pinned"),
        ({"target_layer_one_indexed": 14}, "target_layer_one_indexed is pinned"),
        ({"scan_layers": True}, "scan_layers is pinned"),
        ({"use_multimodal": True}, "use_multimodal is pinned"),
        ({"conditions": ("A", "F")}, "conditions are exactly"),
        ({"topk_features": 9_000}, "cannot exceed"),
        ({"effective_batch_tokens": 16_000}, "must be divisible"),
        ({"min_harvester_mfu": 0.49}, "50% hard gate"),
        ({"benchmark_steps": 199}, "at least 200"),
        ({"max_writer_backlog_batches": 3}, "fixed at two"),
        ({"min_local_buffer_free_bytes": 100_000_000_000}, "128 GiB gate"),
        ({"capture_cache_max_bytes": 99 * 1024**3}, "at least 100 GiB"),
    ],
)
def test_validation_rejects_drift(change, message):
    with pytest.raises(ValueError, match=message):
        Gemma4Config(**change)


def test_small_scale_configuration_remains_possible():
    cfg = Gemma4Config(
        train_tokens=32_768,
        eval_tokens=1_024,
        shard_tokens=1_024,
        effective_batch_tokens=1_024,
        microbatch_tokens=256,
        warmup_updates=2,
        canary_tokens=1_024,
        health_gate_tokens=2_048,
        checkpoint_every_tokens=4_096,
    )
    assert cfg.planned_updates == 32
    assert cfg.final_padding_tokens == 0
    with pytest.raises(
        ValueError, match="authoritative production requires train_tokens"
    ):
        cfg.validate_production()


def test_default_configuration_is_the_exact_authoritative_contract():
    Gemma4Config().validate_production()


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_id": MODEL_ID, "typo_field": 3}))
    with pytest.raises(ValueError, match="unknown configuration keys: typo_field"):
        load_config(path)


def test_json_and_yaml_round_trip_are_stable(tmp_path):
    pytest.importorskip("yaml")
    cfg = Gemma4Config(artifacts_root=str(tmp_path / "artifacts"))
    for suffix in (".json", ".yaml"):
        path = tmp_path / f"config{suffix}"
        save_config(cfg, path)
        restored = load_config(path)
        assert restored == cfg
        assert restored.config_hash == cfg.config_hash


def test_config_is_immutable():
    cfg = Gemma4Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.bank_width = 12
