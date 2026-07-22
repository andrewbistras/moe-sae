"""Crash-safe, identity-bound JSONL metrics journals.

The training checkpoint is authoritative for scientific progress.  A metrics
journal is diagnostic state which may get ahead of that checkpoint when a
process dies between an optimizer update and checkpoint publication.  This
module makes that relationship explicit:

* every row is bound to one run instance, condition, and seed;
* progress is strictly increasing in both step and scientific-token count;
* appends are flushed and fsynced before returning;
* resume may atomically discard only the suffix ahead of an exact checkpoint
  boundary (with ``(0, 0)`` as the sole boundary not represented by a row).

The class is intended to have one writer (the distributed coordinator).  A
POSIX advisory lock still serializes independent instances and reconciliation
so an accidental second writer fails on progress validation rather than
silently interleaving records.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


JOURNAL_SCHEMA_VERSION = "gemma4-sae-metrics-journal-v1"
TRAINING_JOURNAL_SNAPSHOT_SCHEMA = "gemma4-sae-training-journal-snapshot-v1"
_RUN_INSTANCE_RE = re.compile(r"[0-9a-f]{32}")
_IDENTITY_FIELDS = ("run_instance_id", "condition", "seed")
_RESERVED_FIELDS = frozenset(("journal_schema_version", *_IDENTITY_FIELDS))


class MetricsJournalError(RuntimeError):
    """Raised when journal durability, identity, or progress is invalid."""


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MetricsJournalError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _canonical_json(value: Any, *, label: str) -> bytes:
    """Encode an exactly JSON-native value with one canonical representation."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise MetricsJournalError(f"{label} must be strict JSON") from error
    if decoded != value:
        raise MetricsJournalError(
            f"{label} must round-trip through canonical JSON without coercion"
        )
    return encoded


@dataclass(frozen=True, slots=True)
class JournalIdentity:
    """Immutable identity shared by every row in one journal."""

    run_instance_id: str
    condition: str
    seed: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_instance_id, str)
            or _RUN_INSTANCE_RE.fullmatch(self.run_instance_id) is None
        ):
            raise ValueError(
                "run_instance_id must be 32 lowercase hexadecimal characters"
            )
        if not isinstance(self.condition, str) or not self.condition:
            raise ValueError("condition must be a nonempty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_instance_id": self.run_instance_id,
            "condition": self.condition,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class JournalBinding:
    """Canonical identity and content digest for a validated journal prefix."""

    schema_version: str
    identity: JournalIdentity
    identity_sha256: str
    file_sha256: str
    journal_sha256: str
    row_count: int
    last_step: int
    last_tokens_seen: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.as_dict(),
            "identity_sha256": self.identity_sha256,
            "file_sha256": self.file_sha256,
            "journal_sha256": self.journal_sha256,
            "row_count": self.row_count,
            "last_step": self.last_step,
            "last_tokens_seen": self.last_tokens_seen,
        }


@dataclass(frozen=True, slots=True)
class _ValidatedRow:
    record: dict[str, Any]
    encoded: bytes


def _stat_signature(path: Path) -> tuple[int, int, int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:  # pragma: no cover - defensive OS contract check
            raise OSError("journal write made no progress")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class MetricsJournal:
    """One append-only metrics stream reconciled against checkpoint progress."""

    def __init__(self, path: str | Path, identity: JournalIdentity) -> None:
        if not isinstance(identity, JournalIdentity):
            raise TypeError("identity must be a JournalIdentity")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.lock_path = self.path.parent / f".{self.path.name}.lock"
        self._rows: list[_ValidatedRow] = []
        self._signature: tuple[int, int, int, int, int] | None = None
        identity_value = self.identity.as_dict()
        self._identity_bytes = _canonical_json(
            identity_value, label="journal identity"
        )
        self._digest_header = _canonical_json(
            {
                "schema_version": JOURNAL_SCHEMA_VERSION,
                "identity": identity_value,
            },
            label="journal digest header",
        )
        self._file_hasher = hashlib.sha256()
        self._journal_hasher = hashlib.sha256()
        with self._exclusive_lock():
            self._reload()

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _validate_record(
        self,
        value: Mapping[str, Any],
        *,
        previous_step: int,
        previous_tokens_seen: int,
    ) -> _ValidatedRow:
        if not isinstance(value, Mapping):
            raise MetricsJournalError("journal row must be a mapping")
        record = dict(value)
        if not all(isinstance(key, str) for key in record):
            raise MetricsJournalError("journal row keys must be strings")
        expected_identity = self.identity.as_dict()
        if record.get("journal_schema_version") != JOURNAL_SCHEMA_VERSION:
            raise MetricsJournalError("journal row schema version mismatch")
        for field, expected in expected_identity.items():
            if record.get(field) != expected:
                raise MetricsJournalError(
                    f"journal row identity conflict for {field}"
                )
        step = _require_int("step", record.get("step"), minimum=1)
        tokens_seen = _require_int(
            "tokens_seen", record.get("tokens_seen"), minimum=1
        )
        if step <= previous_step or tokens_seen <= previous_tokens_seen:
            raise MetricsJournalError(
                "journal progress must be strictly increasing and unique"
            )
        encoded = _canonical_json(record, label="journal row") + b"\n"
        return _ValidatedRow(record, encoded)

    def _reload(self) -> None:
        try:
            payload = self.path.read_bytes()
        except FileNotFoundError:
            self._rows = []
            self._signature = None
            self._reset_hashers(b"")
            return
        if payload and not payload.endswith(b"\n"):
            raise MetricsJournalError("journal ends with a partial JSONL row")
        rows: list[_ValidatedRow] = []
        previous_step = 0
        previous_tokens = 0
        for index, raw_line in enumerate(payload.splitlines(), start=1):
            try:
                decoded = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise MetricsJournalError(
                    f"journal row {index} is not valid UTF-8 JSON"
                ) from error
            row = self._validate_record(
                decoded,
                previous_step=previous_step,
                previous_tokens_seen=previous_tokens,
            )
            if row.encoded[:-1] != raw_line:
                raise MetricsJournalError(
                    f"journal row {index} is not canonically encoded"
                )
            rows.append(row)
            previous_step = int(row.record["step"])
            previous_tokens = int(row.record["tokens_seen"])
        self._rows = rows
        self._signature = _stat_signature(self.path)
        self._reset_hashers(payload)

    def _reset_hashers(self, file_bytes: bytes) -> None:
        self._file_hasher = hashlib.sha256(file_bytes)
        self._journal_hasher = hashlib.sha256(
            self._digest_header + b"\n" + file_bytes
        )

    def _refresh(self) -> None:
        if _stat_signature(self.path) != self._signature:
            self._reload()

    def _binding(self) -> JournalBinding:
        last = self._rows[-1].record if self._rows else None
        return JournalBinding(
            schema_version=JOURNAL_SCHEMA_VERSION,
            identity=self.identity,
            identity_sha256=hashlib.sha256(self._identity_bytes).hexdigest(),
            file_sha256=self._file_hasher.copy().hexdigest(),
            journal_sha256=self._journal_hasher.copy().hexdigest(),
            row_count=len(self._rows),
            last_step=0 if last is None else int(last["step"]),
            last_tokens_seen=0 if last is None else int(last["tokens_seen"]),
        )

    def snapshot(self) -> JournalBinding:
        """Validate external changes and return the current canonical binding."""

        with self._exclusive_lock():
            self._refresh()
            return self._binding()

    def records(self) -> tuple[dict[str, Any], ...]:
        """Return defensive copies of all validated rows."""

        with self._exclusive_lock():
            self._refresh()
            return tuple(dict(row.record) for row in self._rows)

    def snapshot_records(
        self,
    ) -> tuple[JournalBinding, tuple[dict[str, Any], ...]]:
        """Return one binding and its rows from the same locked file view."""

        with self._exclusive_lock():
            self._refresh()
            return self._binding(), tuple(
                dict(row.record) for row in self._rows
            )

    def append(self, record: Mapping[str, Any]) -> JournalBinding:
        """Append one identity-bound progress row and fsync it before returning.

        Callers supply ``step`` and ``tokens_seen`` plus arbitrary strict-JSON
        metrics.  Identity fields may be omitted; if supplied, they must match
        this journal exactly.  Journal-reserved fields are always canonicalized
        to their expected values.
        """

        if not isinstance(record, Mapping):
            raise MetricsJournalError("journal row must be a mapping")
        supplied = dict(record)
        expected = {
            "journal_schema_version": JOURNAL_SCHEMA_VERSION,
            **self.identity.as_dict(),
        }
        for field in _RESERVED_FIELDS & supplied.keys():
            if supplied[field] != expected[field]:
                raise MetricsJournalError(
                    f"journal row identity conflict for {field}"
                )
        supplied.update(expected)

        with self._exclusive_lock():
            self._refresh()
            previous = self._rows[-1].record if self._rows else None
            row = self._validate_record(
                supplied,
                previous_step=0 if previous is None else int(previous["step"]),
                previous_tokens_seen=(
                    0 if previous is None else int(previous["tokens_seen"])
                ),
            )
            created = not self.path.exists()
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                0o644,
            )
            try:
                _write_all(descriptor, row.encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if created:
                _fsync_directory(self.path.parent)
            self._rows.append(row)
            self._file_hasher.update(row.encoded)
            self._journal_hasher.update(row.encoded)
            self._signature = _stat_signature(self.path)
            return self._binding()

    def _atomic_replace(self, rows: list[_ValidatedRow]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            payload = b"".join(row.encoded for row in rows)
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def reconcile(self, *, step: int, tokens_seen: int) -> JournalBinding:
        """Atomically retain exactly the prefix committed by a checkpoint.

        A nonzero checkpoint must match one existing row exactly.  ``(0, 0)``
        denotes the initial state and may truncate the entire journal even
        though it has no row.  Mixed-zero or unknown boundaries fail without
        changing the file.
        """

        restored_step = _require_int("checkpoint step", step, minimum=0)
        restored_tokens = _require_int(
            "checkpoint tokens_seen", tokens_seen, minimum=0
        )
        if (restored_step == 0) != (restored_tokens == 0):
            raise MetricsJournalError(
                "checkpoint progress must be either (0, 0) or fully positive"
            )

        with self._exclusive_lock():
            self._refresh()
            if restored_step == 0:
                keep = 0
            else:
                matches = [
                    index
                    for index, row in enumerate(self._rows)
                    if int(row.record["step"]) == restored_step
                    and int(row.record["tokens_seen"]) == restored_tokens
                ]
                if len(matches) != 1:
                    raise MetricsJournalError(
                        "checkpoint does not match a known journal boundary"
                    )
                keep = matches[0] + 1
            if keep < len(self._rows):
                retained = self._rows[:keep]
                self._atomic_replace(retained)
                self._rows = retained
                self._reset_hashers(b"".join(row.encoded for row in retained))
                self._signature = _stat_signature(self.path)
            return self._binding()


def training_journal_snapshot(
    journal: MetricsJournal,
    *,
    total_tokens: int,
    effective_batch_tokens: int,
    expected_l0: int,
) -> dict[str, Any]:
    """Validate a complete update prefix and bind its exact journal bytes.

    The checkpoint controls the retained boundary; this function proves that
    every update from step one through that boundary has exactly one metrics
    row with the corresponding scientific-token count. TopK/L0 validity are
    summarized without discarding earlier failures, so a resumed segment
    cannot accidentally reset the health gate.
    """

    if not isinstance(journal, MetricsJournal):
        raise TypeError("journal must be a MetricsJournal")
    for name, value in (
        ("total_tokens", total_tokens),
        ("effective_batch_tokens", effective_batch_tokens),
        ("expected_l0", expected_l0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    binding, records = journal.snapshot_records()
    if binding.row_count != len(records) or binding.last_step != len(records):
        raise MetricsJournalError(
            "training journal must contain exactly one row for every update"
        )

    all_topk_valid = True
    all_exact_l0 = True
    previous_optimizer_seconds = -1.0
    previous_tokens = 0
    for expected_step, record in enumerate(records, start=1):
        expected_tokens = min(
            expected_step * effective_batch_tokens, total_tokens
        )
        if (
            record.get("step") != expected_step
            or record.get("tokens_seen") != expected_tokens
            or record.get("valid_tokens") != expected_tokens - previous_tokens
        ):
            raise MetricsJournalError(
                "training journal step/token progression is incomplete"
            )
        previous_tokens = expected_tokens
        topk_valid = record.get("topk_valid")
        if not isinstance(topk_valid, bool):
            raise MetricsJournalError("training journal topk_valid must be boolean")
        try:
            l0 = float(record["l0"])
            optimizer_seconds = float(record["optimizer_wall_seconds"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise MetricsJournalError(
                "training journal lacks numeric L0/optimizer timing"
            ) from error
        if (
            not math.isfinite(l0)
            or not math.isfinite(optimizer_seconds)
            or optimizer_seconds <= previous_optimizer_seconds
        ):
            raise MetricsJournalError(
                "training journal L0/timing is non-finite or non-increasing"
            )
        previous_optimizer_seconds = optimizer_seconds
        all_topk_valid = all_topk_valid and topk_valid
        all_exact_l0 = all_exact_l0 and math.isclose(
            l0, float(expected_l0), rel_tol=0.0, abs_tol=1e-6
        )

    if (
        binding.last_tokens_seen != previous_tokens
        or binding.last_step != len(records)
    ):
        raise MetricsJournalError("training journal binding disagrees with its rows")
    return {
        "schema_version": TRAINING_JOURNAL_SNAPSHOT_SCHEMA,
        "path": str(journal.path.expanduser().resolve()),
        "binding": binding.as_dict(),
        "total_tokens": total_tokens,
        "effective_batch_tokens": effective_batch_tokens,
        "expected_l0": expected_l0,
        "complete_prefix_topk_valid": all_topk_valid,
        "complete_prefix_exact_l0": all_exact_l0,
        "last_cumulative_optimizer_wall_seconds": (
            0.0 if not records else previous_optimizer_seconds
        ),
    }


__all__ = [
    "JOURNAL_SCHEMA_VERSION",
    "TRAINING_JOURNAL_SNAPSHOT_SCHEMA",
    "JournalBinding",
    "JournalIdentity",
    "MetricsJournal",
    "MetricsJournalError",
    "training_journal_snapshot",
]
