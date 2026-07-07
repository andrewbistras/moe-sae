"""Resume machinery for interrupt-safe capture (driver progress protocol +
resume-safe ShardWriter numbering). The end-to-end resume path is additionally
exercised by a kill/restart drill on the real capture run (RUNBOOK)."""

import json

import pytest
import torch

from moe_sae.capture.driver import PROGRESS_FILE, CaptureProgress, _resume_state
from moe_sae.capture.writer import ShardWriter, iter_shards, shard_paths

CONFIG = {
    "visual_cap_per_example": 8,
    "seed": 7,
    "max_seq_len": 2048,
    "text_control": False,
    "shard_size": 32,
}


def add_tokens(w: ShardWriter, n: int, val: float = 1.0) -> None:
    w.add(
        c=torch.full((n, 6, 4), val, dtype=torch.bfloat16),
        experts=torch.zeros(n, 6, dtype=torch.uint8),
        gates=torch.zeros(n, 6),
        example_id=torch.zeros(n, dtype=torch.int32),
        domain=torch.zeros(n, dtype=torch.uint8),
    )


def test_progress_roundtrip_atomic(tmp_path):
    p = CaptureProgress(stream_consumed=12, captured=10, failed=1, config=CONFIG)
    p.tokens = {"visual": 100, "text": 50}
    p.shards = {"visual": 2, "text": 1}
    p.save(tmp_path)
    assert not (tmp_path / (PROGRESS_FILE + ".tmp")).exists()  # atomic rename
    q = CaptureProgress.load(tmp_path)
    assert q == p


def test_fresh_dir_without_resume(tmp_path):
    p = _resume_state(tmp_path, resume=False, run_config=CONFIG)
    assert p.stream_consumed == 0 and p.config == CONFIG


def test_existing_capture_without_resume_refuses(tmp_path):
    CaptureProgress(config=CONFIG).save(tmp_path)
    with pytest.raises(RuntimeError, match="resume"):
        _resume_state(tmp_path, resume=False, run_config=CONFIG)


def test_resume_config_mismatch_refuses(tmp_path):
    CaptureProgress(config=CONFIG).save(tmp_path)
    changed = dict(CONFIG, seed=8)
    with pytest.raises(RuntimeError, match="mismatch"):
        _resume_state(tmp_path, resume=True, run_config=changed)


def test_resume_deletes_stray_unrecorded_shards(tmp_path):
    w = ShardWriter(tmp_path, "visual", shard_size=32)
    add_tokens(w, 64)  # 2 full shards
    w.flush_partial()
    prog = CaptureProgress(stream_consumed=4, captured=4, config=CONFIG)
    prog.shards = {"visual": 2, "text": 0}
    prog.tokens = {"visual": 64, "text": 0}
    prog.save(tmp_path)
    # crash window: one more shard written after the checkpoint
    add_tokens(w, 10)
    w.flush_partial()
    assert len(shard_paths(tmp_path, "visual")) == 3

    resumed = _resume_state(tmp_path, resume=True, run_config=CONFIG)
    assert resumed.stream_consumed == 4
    assert len(shard_paths(tmp_path, "visual")) == 2  # stray deleted
    # a fresh writer continues numbering after the recorded shards
    w2 = ShardWriter(tmp_path, "visual", shard_size=32)
    assert w2.shard_count == 2
    add_tokens(w2, 5, val=2.0)
    w2.close()
    names = [p.name for p in shard_paths(tmp_path, "visual")]
    assert names == [f"shard_{i:05d}.safetensors" for i in range(3)]
    total = sum(s["domain"].shape[0] for s in iter_shards(tmp_path, "visual"))
    assert total == 64 + 5  # no duplicates, no loss of checkpointed data


def test_completed_progress_short_circuits(tmp_path):
    prog = CaptureProgress(config=CONFIG, complete=True)
    prog.save(tmp_path)
    out = _resume_state(tmp_path, resume=True, run_config=CONFIG)
    assert out.complete


def test_flush_partial_shards_are_readable(tmp_path):
    w = ShardWriter(tmp_path, "text", shard_size=1000)
    add_tokens(w, 7)
    w.flush_partial()
    add_tokens(w, 5)
    w.close()
    sizes = [s["domain"].shape[0] for s in iter_shards(tmp_path, "text")]
    assert sizes == [7, 5]
