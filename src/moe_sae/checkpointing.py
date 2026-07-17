"""Crash-safe control pointers for immutable Orbax checkpoint payloads.

The shared POSIX control plane contains only small, atomically replaced JSON
pointers.  Checkpoint payloads may live on a separate local filesystem or in
GCS.  A payload is never overwritten: every save gets a unique attempt path,
and resume/evaluation follows only a validated control pointer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from moe_sae.runtime import atomic_write_json


POINTER_SCHEMA = "gemma4-sae-checkpoint-pointer-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ATTEMPT_RE = re.compile(r"[0-9a-f]{32}")
_PAYLOAD_NAME_RE = re.compile(
    r"step-(?P<step>[0-9]{8})-tokens-(?P<tokens>[0-9]{12})-"
    r"attempt-(?P<attempt>[0-9a-f]{32})"
)
_POINTER_KEYS = {
    "schema_version",
    "kind",
    "payload_uri",
    "step",
    "tokens_seen",
    "attempt_id",
    "run_identity",
    "run_identity_sha256",
    "pointer_sha256",
}


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint identity must be canonical JSON") from error
    return encoded.encode("utf-8")


def identity_sha256(identity: Mapping[str, Any]) -> str:
    """Return the canonical digest which namespaces a run's payloads."""

    if not all(isinstance(key, str) for key in identity):
        raise ValueError("checkpoint identity keys must be strings")
    value = dict(identity)
    encoded = _canonical_json(value)
    if json.loads(encoded) != value:
        raise ValueError("checkpoint identity must round-trip through JSON exactly")
    return hashlib.sha256(encoded).hexdigest()


def _pointer_sha256(pointer_without_hash: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(pointer_without_hash)).hexdigest()


def _split_gcs_uri(value: str) -> tuple[str, tuple[str, ...]]:
    if not value.startswith("gs://"):
        raise ValueError("not a GCS URI")
    remainder = value[5:]
    bucket, separator, object_name = remainder.partition("/")
    if not bucket or bucket in {".", ".."}:
        raise ValueError("GCS URI must include a bucket")
    components = tuple(part for part in object_name.split("/") if part)
    if any(part in {".", ".."} for part in object_name.split("/")):
        raise ValueError("GCS checkpoint paths may not contain dot segments")
    if not separator:
        components = ()
    return bucket, components


def normalize_payload_root(root: str | Path) -> str:
    """Canonicalize a local or ``gs://`` immutable-payload root."""

    value = str(root)
    if not value:
        raise ValueError("checkpoint payload root may not be empty")
    if "://" in value and not value.startswith("gs://"):
        raise ValueError("checkpoint payload root must be local or gs://")
    if value.startswith("gs://"):
        bucket, components = _split_gcs_uri(value)
        suffix = "/".join(components)
        return f"gs://{bucket}" + (f"/{suffix}" if suffix else "")
    return str(Path(value).expanduser().resolve(strict=False))


def _join_payload(root: str, *parts: str) -> str:
    if root.startswith("gs://"):
        return "/".join((root.rstrip("/"), *(part.strip("/") for part in parts)))
    return str(Path(root).joinpath(*parts))


def payload_namespace(
    payload_root: str | Path, run_identity: Mapping[str, Any]
) -> str:
    """Return the immutable namespace reserved for exactly one run identity."""

    return _join_payload(
        normalize_payload_root(payload_root),
        "runs",
        identity_sha256(run_identity),
    )


def new_attempt_id() -> str:
    return secrets.token_hex(16)


def checkpoint_payload_uri(
    payload_root: str | Path,
    run_identity: Mapping[str, Any],
    *,
    step: int,
    tokens_seen: int,
    attempt_id: str,
) -> str:
    """Construct one unique immutable payload location."""

    if not 0 <= step <= 99_999_999:
        raise ValueError("checkpoint step must fit the eight-digit payload field")
    if not 0 <= tokens_seen <= 999_999_999_999:
        raise ValueError("checkpoint tokens must fit the twelve-digit payload field")
    if _ATTEMPT_RE.fullmatch(attempt_id) is None:
        raise ValueError("checkpoint attempt_id must be 32 lowercase hex digits")
    name = f"step-{step:08d}-tokens-{tokens_seen:012d}-attempt-{attempt_id}"
    normalized_root = normalize_payload_root(payload_root)
    result = _join_payload(
        payload_namespace(normalized_root, run_identity),
        name,
    )
    if not result.startswith("gs://"):
        root_path = Path(normalized_root).resolve(strict=False)
        resolved = Path(result).resolve(strict=False)
        try:
            resolved.relative_to(root_path)
        except ValueError as error:
            raise ValueError(
                "local checkpoint payload path escapes its configured root"
            ) from error
    return result


def pointer_path(
    control_root: str | Path, kind: Literal["latest", "final"]
) -> Path:
    """Return the POSIX control pointer path, rejecting object-store roots."""

    value = str(control_root)
    if "://" in value:
        raise ValueError("checkpoint control root must be a POSIX path")
    if kind not in {"latest", "final"}:
        raise ValueError("checkpoint pointer kind must be latest or final")
    return Path(value) / f"{kind}.json"


def make_pointer(
    *,
    kind: Literal["latest", "final"],
    payload_uri: str,
    step: int,
    tokens_seen: int,
    attempt_id: str,
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a self-hashed pointer; publication is handled separately."""

    identity = dict(run_identity)
    identity_digest = identity_sha256(identity)
    pointer: dict[str, Any] = {
        "schema_version": POINTER_SCHEMA,
        "kind": kind,
        "payload_uri": payload_uri,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "attempt_id": attempt_id,
        "run_identity": identity,
        "run_identity_sha256": identity_digest,
    }
    pointer["pointer_sha256"] = _pointer_sha256(pointer)
    return pointer


def validate_pointer(
    pointer: Mapping[str, Any],
    *,
    expected_kind: Literal["latest", "final"],
    payload_root: str | Path,
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate schema, self-hash, identity, and exact payload containment."""

    record = dict(pointer)
    if set(record) != _POINTER_KEYS:
        raise RuntimeError("checkpoint pointer fields do not match the schema")
    if record.get("schema_version") != POINTER_SCHEMA:
        raise RuntimeError("unsupported checkpoint pointer schema")
    if record.get("kind") != expected_kind:
        raise RuntimeError("checkpoint pointer kind mismatch")

    supplied_hash = record.get("pointer_sha256")
    if (
        not isinstance(supplied_hash, str)
        or _SHA256_RE.fullmatch(supplied_hash) is None
    ):
        raise RuntimeError("checkpoint pointer has an invalid self-hash")
    unhashed = dict(record)
    del unhashed["pointer_sha256"]
    if _pointer_sha256(unhashed) != supplied_hash:
        raise RuntimeError("checkpoint pointer self-hash mismatch")

    expected_identity = dict(run_identity)
    expected_identity_digest = identity_sha256(expected_identity)
    if record.get("run_identity") != expected_identity:
        raise RuntimeError("checkpoint pointer belongs to another run")
    if record.get("run_identity_sha256") != expected_identity_digest:
        raise RuntimeError("checkpoint pointer identity hash mismatch")

    step = record.get("step")
    tokens_seen = record.get("tokens_seen")
    attempt_id = record.get("attempt_id")
    payload_uri = record.get("payload_uri")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise RuntimeError("checkpoint pointer has an invalid step")
    if (
        isinstance(tokens_seen, bool)
        or not isinstance(tokens_seen, int)
        or tokens_seen < 0
    ):
        raise RuntimeError("checkpoint pointer has an invalid token count")
    if not isinstance(attempt_id, str) or _ATTEMPT_RE.fullmatch(attempt_id) is None:
        raise RuntimeError("checkpoint pointer has an invalid attempt ID")
    if not isinstance(payload_uri, str):
        raise RuntimeError("checkpoint pointer has an invalid payload URI")

    expected_uri = checkpoint_payload_uri(
        payload_root,
        expected_identity,
        step=step,
        tokens_seen=tokens_seen,
        attempt_id=attempt_id,
    )
    if payload_uri != expected_uri:
        raise RuntimeError(
            "checkpoint payload URI escapes its configured run namespace"
        )
    name = (
        PurePosixPath(payload_uri).name
        if payload_uri.startswith("gs://")
        else Path(payload_uri).name
    )
    match = _PAYLOAD_NAME_RE.fullmatch(name)
    if match is None:
        raise RuntimeError("checkpoint payload URI has an invalid immutable name")
    if (
        int(match.group("step")) != step
        or int(match.group("tokens")) != tokens_seen
        or match.group("attempt") != attempt_id
    ):
        raise RuntimeError("checkpoint payload URI disagrees with pointer progress")
    return record


def publish_pointer(
    control_root: str | Path,
    pointer: Mapping[str, Any],
) -> Path:
    """Atomically publish a validated-shape pointer on the POSIX control plane."""

    kind = pointer.get("kind")
    if kind not in {"latest", "final"}:
        raise ValueError("checkpoint pointer kind must be latest or final")
    destination = pointer_path(control_root, kind)
    atomic_write_json(destination, dict(pointer))
    # Persist the directory entry as well as the file contents.  Some network
    # filesystems do not support directory fsync; an explicit failure is safer
    # than claiming a durable checkpoint pointer that is not durable.
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def load_pointer(
    control_root: str | Path,
    *,
    kind: Literal["latest", "final"],
    payload_root: str | Path,
    run_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Read and validate a control pointer, returning ``None`` only if absent."""

    source = pointer_path(control_root, kind)
    if not source.exists():
        return None
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read checkpoint pointer: {source}") from error
    if not isinstance(raw, dict):
        raise RuntimeError("checkpoint pointer must be a JSON object")
    return validate_pointer(
        raw,
        expected_kind=kind,
        payload_root=payload_root,
        run_identity=run_identity,
    )


def final_pointer_from_latest(latest: Mapping[str, Any]) -> dict[str, Any]:
    """Create a final pointer which reuses an already-final latest payload."""

    record = dict(latest)
    if record.get("kind") != "latest":
        raise ValueError("final pointer source must be a latest pointer")
    record["kind"] = "final"
    record.pop("pointer_sha256", None)
    record["pointer_sha256"] = _pointer_sha256(record)
    return record
