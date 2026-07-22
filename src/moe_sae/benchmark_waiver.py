"""Auditable, MFU-only exception policy for activation harvesting.

The benchmark observation remains immutable and truthful.  A waiver is a
separate, self-hashed record that may excuse only the configured useful-MFU
minimum (and therefore the derived ``passes_hard_gate`` flag).  Every other
benchmark safety condition remains mandatory.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any

from moe_sae.runtime import (
    BENCHMARK_TIMING_SCHEMA,
    HARVEST_MESHES,
    BenchmarkObservation,
)


PERFORMANCE_WAIVER_SCHEMA = "gemma4-harvester-performance-waiver-v1"
PERFORMANCE_WAIVER_SCOPE = "minimum-useful-mfu-only"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IDENTITY_FIELDS = (
    "config_sha256",
    "checkpoint_sha256",
    "conversion_attestation_sha256",
    "storage_attestation_sha256",
    "environment_sha256",
)


class BenchmarkWaiverError(RuntimeError):
    """The performance exception or its benchmark evidence is invalid."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return sha256_json({key: item for key, item in value.items() if key != field})


def extract_observation_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Extract one observation from a direct, raw, or supervised record."""

    if value.get("timing_schema") == BENCHMARK_TIMING_SCHEMA:
        return dict(value)
    candidates = [
        item
        for key in ("observation", "selected")
        if isinstance((item := value.get(key)), Mapping)
    ]
    if len(candidates) != 1:
        raise BenchmarkWaiverError(
            "benchmark source must contain exactly one observation or selected record"
        )
    return dict(candidates[0])


def normalize_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Parse and recompute all derived benchmark fields fail-closed."""

    raw = extract_observation_mapping(value)
    encoded_mesh = raw.get("mesh")
    if not isinstance(encoded_mesh, Mapping):
        raise BenchmarkWaiverError("benchmark observation lacks structured mesh identity")
    try:
        mesh = next(
            (
                candidate
                for candidate in HARVEST_MESHES
                if candidate.name == encoded_mesh.get("name")
                and candidate.expert_parallelism
                == int(encoded_mesh.get("expert_parallelism", -1))
                and candidate.fsdp_parallelism
                == int(encoded_mesh.get("fsdp_parallelism", -1))
                and candidate.tensor_parallelism
                == int(encoded_mesh.get("tensor_parallelism", -1))
            ),
            None,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise BenchmarkWaiverError("benchmark observation has invalid mesh identity") from error
    if mesh is None:
        raise BenchmarkWaiverError("benchmark observation uses an unknown mesh identity")
    try:
        observation = BenchmarkObservation(
            mesh=mesh,
            per_device_sequence_batch=int(raw["per_device_sequence_batch"]),
            synchronized_steps=int(raw["synchronized_steps"]),
            elapsed_seconds=float(raw["elapsed_seconds"]),
            device_elapsed_seconds=float(raw["device_elapsed_seconds"]),
            persistence_seconds=float(raw["persistence_seconds"]),
            useful_flops=float(raw["useful_flops"]),
            output_tokens=int(raw["output_tokens"]),
            persisted_tokens=int(raw["persisted_tokens"]),
            input_wait_seconds=float(raw.get("input_wait_seconds", 0.0)),
            recompilations=int(raw.get("recompilations", -1)),
            dropped_routed_tokens=int(raw.get("dropped_routed_tokens", -1)),
            max_writer_backlog=int(raw.get("max_writer_backlog", -1)),
            instrumentation_equal=raw.get("instrumentation_equal") is True,
            failure=(str(raw["failure"]) if raw.get("failure") is not None else None),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise BenchmarkWaiverError("benchmark observation has an incompatible schema") from error

    normalized = observation.as_dict()
    for field in (
        "useful_mfu",
        "device_tokens_per_second",
        "persisted_tokens_per_second",
        "input_wait_fraction",
        "persistence_fraction",
    ):
        try:
            claimed = float(raw[field])
            recomputed = float(normalized[field])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise BenchmarkWaiverError(
                f"benchmark observation lacks derived field {field}"
            ) from error
        if not math.isfinite(claimed) or not math.isclose(
            claimed, recomputed, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise BenchmarkWaiverError(
                f"benchmark observation has inconsistent derived field {field}"
            )
    if raw.get("passes_hard_gate") is not normalized["passes_hard_gate"]:
        raise BenchmarkWaiverError(
            "benchmark observation has an inconsistent passes_hard_gate flag"
        )
    return normalized


def require_mfu_only_failure(
    selected: Mapping[str, Any], *, minimum_mfu: float
) -> dict[str, Any]:
    """Return normalized evidence iff MFU is its sole failed hard check."""

    normalized = normalize_observation(selected)
    observation = BenchmarkObservation(
        mesh=next(
            item for item in HARVEST_MESHES if item.name == normalized["mesh"]["name"]
        ),
        per_device_sequence_batch=int(normalized["per_device_sequence_batch"]),
        synchronized_steps=int(normalized["synchronized_steps"]),
        elapsed_seconds=float(normalized["elapsed_seconds"]),
        device_elapsed_seconds=float(normalized["device_elapsed_seconds"]),
        persistence_seconds=float(normalized["persistence_seconds"]),
        useful_flops=float(normalized["useful_flops"]),
        output_tokens=int(normalized["output_tokens"]),
        persisted_tokens=int(normalized["persisted_tokens"]),
        input_wait_seconds=float(normalized["input_wait_seconds"]),
        recompilations=int(normalized["recompilations"]),
        dropped_routed_tokens=int(normalized["dropped_routed_tokens"]),
        max_writer_backlog=int(normalized["max_writer_backlog"]),
        instrumentation_equal=normalized["instrumentation_equal"] is True,
        failure=normalized["failure"],
    )
    finite_timings = (
        observation.elapsed_seconds,
        observation.device_elapsed_seconds,
        observation.persistence_seconds,
        observation.input_wait_seconds,
        observation.useful_flops,
        observation.useful_mfu,
        observation.persisted_tokens_per_second,
        observation.device_tokens_per_second,
    )
    if (
        not math.isfinite(minimum_mfu)
        or minimum_mfu < 0.50
        or any(not math.isfinite(item) for item in finite_timings)
        or observation.elapsed_seconds <= 0.0
        or observation.device_elapsed_seconds <= 0.0
        or observation.device_elapsed_seconds > observation.elapsed_seconds
        or observation.persistence_seconds < 0.0
        or observation.persistence_seconds > observation.elapsed_seconds
        or observation.input_wait_seconds < 0.0
        or observation.input_wait_seconds > observation.elapsed_seconds
        or observation.useful_flops <= 0.0
        or observation.output_tokens <= 0
        or observation.persisted_tokens != observation.output_tokens
        or observation.per_device_sequence_batch not in (1, 2, 4)
        or not observation.passes_hard_gate(minimum_mfu=0.0)
        or observation.useful_mfu >= minimum_mfu
        or normalized["passes_hard_gate"] is not False
        or not math.isfinite(observation.persisted_tokens_per_second)
        or observation.persisted_tokens_per_second <= 0.0
    ):
        raise BenchmarkWaiverError(
            "performance waiver may excuse only an unmet useful-MFU minimum; "
            "all timing, persistence, drop, recompile, backlog, parity, and failure "
            "checks must remain clean"
        )
    return normalized


def create_performance_waiver(
    selected: Mapping[str, Any],
    *,
    minimum_mfu: float,
    identities: Mapping[str, str],
    reason: str,
    operator_identity: str,
    operator_authorization: str,
    source_record: str,
    source_record_sha256: str,
    observation_producer: Mapping[str, Any],
    cross_environment_evidence_authorized: bool,
    supplementary_evidence: list[Mapping[str, Any]] | None = None,
    authorized_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create a self-hashed exception bound to one normalized observation."""

    normalized = require_mfu_only_failure(selected, minimum_mfu=minimum_mfu)
    missing = [
        field
        for field in _IDENTITY_FIELDS
        if not _SHA256_RE.fullmatch(str(identities.get(field, "")))
    ]
    if missing:
        raise BenchmarkWaiverError(
            f"performance waiver lacks valid identities: {', '.join(missing)}"
        )
    for label, value in (
        ("reason", reason),
        ("operator identity", operator_identity),
        ("operator authorization", operator_authorization),
        ("source record", source_record),
    ):
        if not str(value).strip():
            raise BenchmarkWaiverError(f"{label} cannot be empty")
    if not _SHA256_RE.fullmatch(source_record_sha256):
        raise BenchmarkWaiverError("source record SHA-256 is invalid")
    timestamp = authorized_at_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload: dict[str, Any] = {
        "schema_version": PERFORMANCE_WAIVER_SCHEMA,
        "scope": PERFORMANCE_WAIVER_SCOPE,
        "authorized_at_utc": timestamp,
        "operator_identity": str(operator_identity).strip(),
        "operator_authorization": str(operator_authorization).strip(),
        "reason": str(reason).strip(),
        "identities": {field: str(identities[field]) for field in _IDENTITY_FIELDS},
        "selected_mesh": normalized["mesh"],
        "per_device_sequence_batch": normalized["per_device_sequence_batch"],
        "actual_useful_mfu": normalized["useful_mfu"],
        "minimum_useful_mfu": float(minimum_mfu),
        "actual_passes_hard_gate": normalized["passes_hard_gate"],
        "waived_checks": ["minimum_useful_mfu", "passes_hard_gate"],
        "selected_observation_sha256": sha256_json(normalized),
        "source_record": str(source_record),
        "source_record_sha256": source_record_sha256,
        "observation_producer": dict(observation_producer),
        "cross_environment_evidence_authorized": bool(
            cross_environment_evidence_authorized
        ),
        "supplementary_evidence": [
            dict(item) for item in (supplementary_evidence or [])
        ],
    }
    payload["waiver_sha256"] = _self_hash(payload, "waiver_sha256")
    validate_performance_waiver(
        payload,
        normalized,
        minimum_mfu=minimum_mfu,
        expected_identities=identities,
    )
    return payload


def validate_performance_waiver(
    waiver: Mapping[str, Any],
    selected: Mapping[str, Any],
    *,
    minimum_mfu: float,
    expected_identities: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate a self-hashed waiver and return its plain JSON mapping."""

    normalized = require_mfu_only_failure(selected, minimum_mfu=minimum_mfu)
    value = dict(waiver)
    identities = value.get("identities")
    identity_mapping = dict(identities) if isinstance(identities, Mapping) else {}
    mesh = value.get("selected_mesh")
    timestamp = value.get("authorized_at_utc")
    supplementary = value.get("supplementary_evidence")
    producer = value.get("observation_producer")
    try:
        parsed_timestamp = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        encoded_batch = int(value.get("per_device_sequence_batch", -1))
        encoded_actual_mfu = float(value.get("actual_useful_mfu", math.nan))
        encoded_minimum_mfu = float(value.get("minimum_useful_mfu", math.nan))
    except (TypeError, ValueError, OverflowError) as error:
        raise BenchmarkWaiverError("performance waiver has invalid scalar fields") from error
    identity_valid = isinstance(identities, Mapping) and all(
        _SHA256_RE.fullmatch(str(identity_mapping.get(field, "")))
        for field in _IDENTITY_FIELDS
    )
    expected_identity_mapping = (
        dict(expected_identities) if isinstance(expected_identities, Mapping) else {}
    )
    expected_valid = expected_identities is None or all(
        str(identity_mapping.get(field, ""))
        == str(expected_identity_mapping.get(field, ""))
        for field in _IDENTITY_FIELDS
    )
    supplementary_valid = isinstance(supplementary, list) and all(
        isinstance(item, Mapping)
        and item.get("kind") in {"file", "directory-inventory"}
        and bool(str(item.get("path", "")).strip())
        and _SHA256_RE.fullmatch(str(item.get("sha256", ""))) is not None
        and isinstance(item.get("bytes"), int)
        and not isinstance(item.get("bytes"), bool)
        and int(item["bytes"]) >= 0
        and isinstance(item.get("files"), int)
        and not isinstance(item.get("files"), bool)
        and int(item["files"]) >= 1
        for item in (supplementary or [])
    )
    producer_mapping = dict(producer) if isinstance(producer, Mapping) else {}
    producer_identity_valid = isinstance(producer, Mapping) and all(
        _SHA256_RE.fullmatch(str(producer_mapping.get(field, "")))
        for field in _IDENTITY_FIELDS
    )
    producer_core_matches = all(
        producer_mapping.get(field) == identity_mapping.get(field)
        for field in (
            "config_sha256",
            "checkpoint_sha256",
            "conversion_attestation_sha256",
        )
    )
    cross_environment = any(
        producer_mapping.get(field) != identity_mapping.get(field)
        for field in ("storage_attestation_sha256", "environment_sha256")
    )
    if (
        value.get("schema_version") != PERFORMANCE_WAIVER_SCHEMA
        or value.get("scope") != PERFORMANCE_WAIVER_SCOPE
        or not _SHA256_RE.fullmatch(str(value.get("waiver_sha256", "")))
        or value.get("waiver_sha256") != _self_hash(value, "waiver_sha256")
        or parsed_timestamp.tzinfo is None
        or not str(value.get("operator_identity", "")).strip()
        or not str(value.get("operator_authorization", "")).strip()
        or not str(value.get("reason", "")).strip()
        or not identity_valid
        or not expected_valid
        or mesh != normalized["mesh"]
        or encoded_batch != int(normalized["per_device_sequence_batch"])
        or not math.isclose(
            encoded_actual_mfu,
            float(normalized["useful_mfu"]),
            rel_tol=1e-12,
        )
        or not math.isclose(
            encoded_minimum_mfu,
            float(minimum_mfu),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or value.get("actual_passes_hard_gate") is not False
        or value.get("waived_checks")
        != ["minimum_useful_mfu", "passes_hard_gate"]
        or value.get("selected_observation_sha256") != sha256_json(normalized)
        or not str(value.get("source_record", "")).strip()
        or not _SHA256_RE.fullmatch(str(value.get("source_record_sha256", "")))
        or not producer_identity_valid
        or not producer_core_matches
        or not str(producer_mapping.get("benchmark_sweep_id", "")).strip()
        or not str(producer_mapping.get("bootstrap_record", "")).strip()
        or not _SHA256_RE.fullmatch(
            str(producer_mapping.get("bootstrap_record_sha256", ""))
        )
        or (
            cross_environment
            and value.get("cross_environment_evidence_authorized") is not True
        )
        or (
            not cross_environment
            and value.get("cross_environment_evidence_authorized") is not False
        )
        or not supplementary_valid
    ):
        raise BenchmarkWaiverError(
            "performance waiver is invalid, tampered, or bound to another run"
        )
    return value


def selection_gate_status(
    envelope: Mapping[str, Any],
    *,
    minimum_mfu: float,
    expected_identities: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Validate normal selection or the sole allowed performance exception."""

    selected_raw = envelope.get("selected")
    if not isinstance(selected_raw, Mapping):
        raise BenchmarkWaiverError("selected benchmark envelope lacks an observation")
    selected = normalize_observation(selected_raw)
    if (
        selected["passes_hard_gate"] is True
        and float(selected["useful_mfu"]) >= minimum_mfu
    ):
        if envelope.get("performance_waiver") is not None:
            raise BenchmarkWaiverError("a passing benchmark must not carry a waiver")
        return selected, None
    waiver_raw = envelope.get("performance_waiver")
    if not isinstance(waiver_raw, Mapping):
        raise BenchmarkWaiverError(
            "selected benchmark did not pass the MFU gate and has no valid waiver"
        )
    waiver = validate_performance_waiver(
        waiver_raw,
        selected,
        minimum_mfu=minimum_mfu,
        expected_identities=expected_identities,
    )
    return selected, waiver


__all__ = [
    "BenchmarkWaiverError",
    "PERFORMANCE_WAIVER_SCHEMA",
    "create_performance_waiver",
    "extract_observation_mapping",
    "normalize_observation",
    "require_mfu_only_failure",
    "selection_gate_status",
    "sha256_json",
    "validate_performance_waiver",
]
