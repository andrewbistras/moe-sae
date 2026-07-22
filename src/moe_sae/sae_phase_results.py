"""Immutable result envelopes for supervised SAE phases.

The four JAX-free farm supervisors need a small value they can independently
revalidate after a numerical child exits.  Numerical artifacts themselves are
not all immutable for their entire lifetime: for example, the seed-1 gate's
``run.json`` and ``latest.json`` are legitimately replaced when that seed
continues from 5M to 50M tokens.  A phase result therefore records the exact
evidence validated at the boundary and seals that evidence into an immutable,
self-hashed envelope.

This module deliberately does not decide whether evidence is scientifically
valid.  The phase child must construct evidence only after performing the
condition-specific checks.  It does guarantee that every controller reports
the same deterministic result digest and that a passed phase can later be
skipped only when its original envelope still validates byte-for-byte.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from moe_sae.sae_operator import SAE_PHASES
from moe_sae.sae_paths import SAEOperationPaths


PHASE_RESULT_SCHEMA = "gemma4-sae-phase-result-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class SAEPhaseResultError(RuntimeError):
    """A phase result is missing, conflicting, or fails validation."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        if json.loads(encoded) != value:
            raise ValueError("canonical JSON round trip changed the value")
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise SAEPhaseResultError("phase evidence must be strict JSON") from error
    return encoded


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _without(value: Mapping[str, Any], *names: str) -> dict[str, Any]:
    excluded = frozenset(names)
    return {name: item for name, item in value.items() if name not in excluded}


def _phase(value: str) -> str:
    if not isinstance(value, str) or value not in SAE_PHASES:
        raise SAEPhaseResultError(f"unknown SAE phase: {value!r}")
    return value


def _result_path(paths: SAEOperationPaths, phase: str) -> Path:
    selected = _phase(phase)
    root = paths.operation_root.resolve()
    path = (root / "phase-results" / f"{selected}.json").resolve()
    try:
        path.relative_to(root)
    except ValueError as error:  # pragma: no cover - fixed names are defensive
        raise SAEPhaseResultError("phase result path escapes the operation") from error
    return path


def phase_result_path(paths: SAEOperationPaths, phase: str) -> Path:
    """Return the validated immutable envelope path for ``phase``."""

    if not isinstance(paths, SAEOperationPaths):
        raise TypeError("paths must be an SAEOperationPaths")
    return _result_path(paths, phase)


def _deterministic_payload(
    paths: SAEOperationPaths, phase: str, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise TypeError("phase evidence must be a mapping")
    material = dict(evidence)
    if not all(isinstance(name, str) for name in material):
        raise SAEPhaseResultError("phase evidence keys must be strings")
    _canonical_bytes(material)
    evidence_sha256 = _sha256(material)
    return {
        "schema_version": PHASE_RESULT_SCHEMA,
        "operation_sha256": paths.operation_sha256,
        "phase": _phase(phase),
        "evidence": material,
        "evidence_sha256": evidence_sha256,
    }


def _validate_record(
    value: Any,
    *,
    paths: SAEOperationPaths,
    phase: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SAEPhaseResultError("phase result is not a JSON object")
    record = dict(value)
    expected_fields = {
        "schema_version",
        "operation_sha256",
        "phase",
        "evidence",
        "evidence_sha256",
        "created_at_utc",
        "result_sha256",
        "record_sha256",
    }
    if set(record) != expected_fields:
        raise SAEPhaseResultError("phase result fields do not match the schema")
    if (
        record.get("schema_version") != PHASE_RESULT_SCHEMA
        or record.get("operation_sha256") != paths.operation_sha256
        or record.get("phase") != _phase(phase)
        or not isinstance(record.get("evidence"), Mapping)
        or not isinstance(record.get("created_at_utc"), str)
        or not record["created_at_utc"]
    ):
        raise SAEPhaseResultError("phase result identity is invalid")
    deterministic = _deterministic_payload(
        paths, phase, dict(record["evidence"])
    )
    if record.get("evidence_sha256") != deterministic["evidence_sha256"]:
        raise SAEPhaseResultError("phase evidence SHA-256 mismatch")
    result_sha256 = record.get("result_sha256")
    record_sha256 = record.get("record_sha256")
    if (
        not isinstance(result_sha256, str)
        or _SHA256_RE.fullmatch(result_sha256) is None
        or result_sha256 != _sha256(deterministic)
    ):
        raise SAEPhaseResultError("phase deterministic result SHA-256 mismatch")
    if (
        not isinstance(record_sha256, str)
        or _SHA256_RE.fullmatch(record_sha256) is None
        or record_sha256 != _sha256(_without(record, "record_sha256"))
    ):
        raise SAEPhaseResultError("phase result self-hash mismatch")
    _canonical_bytes(record)
    return record


def validate_phase_result(paths: SAEOperationPaths, phase: str) -> str:
    """Revalidate an immutable envelope and return its deterministic digest."""

    path = phase_result_path(paths, phase)
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SAEPhaseResultError(f"cannot read phase result: {path}") from error
    record = _validate_record(value, paths=paths, phase=phase)
    expected_bytes = (
        json.dumps(
            record,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if raw != expected_bytes:
        raise SAEPhaseResultError("phase result is not canonically encoded")
    return str(record["result_sha256"])


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_phase_result(
    paths: SAEOperationPaths,
    phase: str,
    evidence: Mapping[str, Any],
    *,
    created_at_utc: str | None = None,
) -> str:
    """Publish once, or idempotently accept the same deterministic evidence.

    The timestamp is diagnostic and intentionally excluded from
    ``result_sha256``.  A concurrent publisher which wins the no-replace race
    is accepted only when its fully validated deterministic result is equal.
    """

    path = phase_result_path(paths, phase)
    deterministic = _deterministic_payload(paths, phase, evidence)
    result_sha256 = _sha256(deterministic)
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()
    if not isinstance(timestamp, str) or not timestamp:
        raise SAEPhaseResultError("phase result timestamp is invalid")
    record: dict[str, Any] = {
        **deterministic,
        "created_at_utc": timestamp,
        "result_sha256": result_sha256,
    }
    record["record_sha256"] = _sha256(record)
    encoded = (
        json.dumps(
            record,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".unpublished", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            observed = validate_phase_result(paths, phase)
            if observed != result_sha256:
                raise SAEPhaseResultError(
                    f"conflicting immutable phase result already exists: {path}"
                )
            return observed
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return validate_phase_result(paths, phase)


__all__ = [
    "PHASE_RESULT_SCHEMA",
    "SAEPhaseResultError",
    "phase_result_path",
    "publish_phase_result",
    "validate_phase_result",
]
