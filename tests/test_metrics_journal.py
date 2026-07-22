from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import moe_sae.metrics_journal as journal_module
from moe_sae.metrics_journal import (
    JOURNAL_SCHEMA_VERSION,
    TRAINING_JOURNAL_SNAPSHOT_SCHEMA,
    JournalIdentity,
    MetricsJournal,
    MetricsJournalError,
    training_journal_snapshot,
)


def _identity(*, run: str = "a" * 32) -> JournalIdentity:
    return JournalIdentity(run_instance_id=run, condition="A", seed=1)


def _append(journal: MetricsJournal, step: int) -> None:
    journal.append(
        {
            "step": step,
            "tokens_seen": step * 16_384,
            "loss": 1.0 / step,
        }
    )


def test_append_is_identity_bound_canonical_fsynced_and_reopenable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metrics.jsonl"
    real_fsync = os.fsync
    fsynced: list[int] = []

    def observed_fsync(descriptor: int) -> None:
        fsynced.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(journal_module.os, "fsync", observed_fsync)
    journal = MetricsJournal(path, _identity())
    binding = journal.append(
        {"tokens_seen": 16_384, "loss": 0.75, "step": 1}
    )

    assert fsynced
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    record = json.loads(raw)
    assert record == {
        "condition": "A",
        "journal_schema_version": JOURNAL_SCHEMA_VERSION,
        "loss": 0.75,
        "run_instance_id": "a" * 32,
        "seed": 1,
        "step": 1,
        "tokens_seen": 16_384,
    }
    assert raw == (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    assert binding.row_count == 1
    assert binding.last_step == 1
    assert binding.last_tokens_seen == 16_384
    assert binding.file_sha256
    assert binding.journal_sha256

    reopened = MetricsJournal(path, _identity()).snapshot()
    assert reopened == binding


@pytest.mark.parametrize(
    "record,match",
    [
        ({"step": True, "tokens_seen": 1}, "step must be an integer"),
        ({"step": 1, "tokens_seen": False}, "tokens_seen must be an integer"),
        ({"step": 0, "tokens_seen": 1}, "step must be an integer"),
        ({"step": 1, "tokens_seen": 0}, "tokens_seen must be an integer"),
        ({"step": 1, "tokens_seen": 1, "loss": float("nan")}, "strict JSON"),
        (
            {"step": 1, "tokens_seen": 1, "run_instance_id": "b" * 32},
            "identity conflict",
        ),
    ],
)
def test_append_rejects_invalid_rows_without_changing_journal(
    tmp_path: Path, record: dict[str, object], match: str
) -> None:
    journal = MetricsJournal(tmp_path / "metrics.jsonl", _identity())
    before = journal.snapshot()
    with pytest.raises(MetricsJournalError, match=match):
        journal.append(record)
    assert journal.snapshot() == before


def test_progress_must_increase_in_both_dimensions(tmp_path: Path) -> None:
    journal = MetricsJournal(tmp_path / "metrics.jsonl", _identity())
    journal.append({"step": 2, "tokens_seen": 20, "loss": 1.0})
    for record in (
        {"step": 2, "tokens_seen": 21},
        {"step": 3, "tokens_seen": 20},
        {"step": 1, "tokens_seen": 19},
        {"step": 2, "tokens_seen": 20},
    ):
        with pytest.raises(MetricsJournalError, match="strictly increasing"):
            journal.append(record)
    assert journal.snapshot().row_count == 1


def test_reconcile_removes_only_rows_ahead_then_allows_reappend(
    tmp_path: Path,
) -> None:
    journal = MetricsJournal(tmp_path / "metrics.jsonl", _identity())
    for step in (1, 2, 3):
        _append(journal, step)

    binding = journal.reconcile(step=2, tokens_seen=2 * 16_384)
    assert binding.row_count == 2
    assert binding.last_step == 2
    assert [row["step"] for row in journal.records()] == [1, 2]

    _append(journal, 3)
    assert [row["step"] for row in journal.records()] == [1, 2, 3]


def test_reconcile_zero_is_the_only_unrecorded_boundary(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    journal = MetricsJournal(path, _identity())
    _append(journal, 1)
    _append(journal, 2)

    with pytest.raises(MetricsJournalError, match="fully positive"):
        journal.reconcile(step=0, tokens_seen=1)
    with pytest.raises(MetricsJournalError, match="known journal boundary"):
        journal.reconcile(step=1, tokens_seen=99)
    assert journal.snapshot().row_count == 2

    binding = journal.reconcile(step=0, tokens_seen=0)
    assert binding.row_count == 0
    assert binding.last_step == 0
    assert binding.last_tokens_seen == 0
    assert path.read_bytes() == b""


def test_invalid_existing_tail_fails_before_reconciliation(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    journal = MetricsJournal(path, _identity())
    _append(journal, 1)
    first = path.read_bytes()
    duplicate = json.loads(first)
    path.write_bytes(
        first
        + json.dumps(duplicate, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )

    with pytest.raises(MetricsJournalError, match="strictly increasing"):
        MetricsJournal(path, _identity())
    # Validation failure never "repairs" or truncates suspect input.
    assert path.read_bytes() != first


def test_existing_identity_conflict_and_partial_line_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    journal = MetricsJournal(path, _identity())
    _append(journal, 1)
    valid = path.read_bytes()

    conflict = json.loads(valid)
    conflict["run_instance_id"] = "b" * 32
    path.write_bytes(
        (json.dumps(conflict, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    with pytest.raises(MetricsJournalError, match="identity conflict"):
        MetricsJournal(path, _identity())

    path.write_bytes(valid.rstrip(b"\n"))
    with pytest.raises(MetricsJournalError, match="partial JSONL"):
        MetricsJournal(path, _identity())


def test_atomic_reconcile_failure_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metrics.jsonl"
    journal = MetricsJournal(path, _identity())
    for step in (1, 2, 3):
        _append(journal, step)
    before = path.read_bytes()

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(journal_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        journal.reconcile(step=2, tokens_seen=2 * 16_384)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".metrics.jsonl.*.tmp"))


def test_independent_instances_refresh_under_the_process_lock(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    first = MetricsJournal(path, _identity())
    second = MetricsJournal(path, _identity())
    _append(first, 1)
    _append(second, 2)

    assert first.snapshot() == second.snapshot()
    assert [row["step"] for row in first.records()] == [1, 2]


def test_digest_is_independent_of_input_mapping_order(tmp_path: Path) -> None:
    left = MetricsJournal(tmp_path / "left.jsonl", _identity())
    right = MetricsJournal(tmp_path / "right.jsonl", _identity())
    left_binding = left.append(
        {"step": 1, "tokens_seen": 16_384, "loss": 0.5, "l0": 64.0}
    )
    right_binding = right.append(
        {"l0": 64.0, "loss": 0.5, "tokens_seen": 16_384, "step": 1}
    )

    assert left_binding.file_sha256 == right_binding.file_sha256
    assert left_binding.journal_sha256 == right_binding.journal_sha256


def _append_training_row(
    journal: MetricsJournal,
    step: int,
    *,
    topk_valid: bool = True,
    l0: float = 64.0,
) -> None:
    previous_tokens = min((step - 1) * 4, 10)
    tokens_seen = min(step * 4, 10)
    journal.append(
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "valid_tokens": tokens_seen - previous_tokens,
            "l0": l0,
            "topk_valid": topk_valid,
            "optimizer_wall_seconds": step * 0.25,
        }
    )


def test_training_snapshot_binds_complete_prefix_and_preserves_health(
    tmp_path: Path,
) -> None:
    journal = MetricsJournal(tmp_path / "training.jsonl", _identity())
    _append_training_row(journal, 1)
    _append_training_row(journal, 2, topk_valid=False)
    _append_training_row(journal, 3, l0=63.0)

    snapshot = training_journal_snapshot(
        journal,
        total_tokens=10,
        effective_batch_tokens=4,
        expected_l0=64,
    )

    assert snapshot["schema_version"] == TRAINING_JOURNAL_SNAPSHOT_SCHEMA
    assert snapshot["binding"]["row_count"] == 3
    assert snapshot["binding"]["last_step"] == 3
    assert snapshot["binding"]["last_tokens_seen"] == 10
    assert snapshot["complete_prefix_topk_valid"] is False
    assert snapshot["complete_prefix_exact_l0"] is False
    assert snapshot["last_cumulative_optimizer_wall_seconds"] == 0.75


@pytest.mark.parametrize(
    "rows,match",
    [
        (
            [
                {
                    "step": 1,
                    "tokens_seen": 4,
                    "valid_tokens": 3,
                    "l0": 64.0,
                    "topk_valid": True,
                    "optimizer_wall_seconds": 0.25,
                }
            ],
            "step/token progression",
        ),
        (
            [
                {
                    "step": 1,
                    "tokens_seen": 4,
                    "valid_tokens": 4,
                    "l0": 64.0,
                    "topk_valid": 1,
                    "optimizer_wall_seconds": 0.25,
                }
            ],
            "topk_valid must be boolean",
        ),
    ],
)
def test_training_snapshot_rejects_noncanonical_update_rows(
    tmp_path: Path,
    rows: list[dict[str, object]],
    match: str,
) -> None:
    journal = MetricsJournal(tmp_path / "training.jsonl", _identity())
    for row in rows:
        journal.append(row)

    with pytest.raises(MetricsJournalError, match=match):
        training_journal_snapshot(
            journal,
            total_tokens=10,
            effective_batch_tokens=4,
            expected_l0=64,
        )
