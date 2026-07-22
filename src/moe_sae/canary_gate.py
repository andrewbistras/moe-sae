"""Validate the real performance-only canary waiver and recovery chain.

The production authorization predates :mod:`moe_sae.canary_waiver` and has a
different payload despite sharing its historical schema name.  Consumers must
therefore validate the artifact that was actually used: an immutable failed
2M canary, a self-hashed performance-only waiver, and a second self-hashed
authorization binding the evolved production ledgers to the sealed 50M
manifest.  No durable record is rewritten to claim that the canary passed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


CANARY_PERFORMANCE_WAIVER_SCHEMA = "gemma4-canary-performance-waiver-v1"
SEALED_RECOVERY_WAIVER_SCHEMA = "gemma4-sealed-capture-recovery-waiver-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class CanaryGateError(RuntimeError):
    """Canary evidence is missing, mutable, or broader than performance."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(8 << 20):
                digest.update(block)
    except OSError as error:
        raise CanaryGateError(f"cannot read waiver-bound evidence: {path}") from error
    return digest.hexdigest()


def _self_hashed(path: Path, schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CanaryGateError(f"cannot read {schema}: {path}") from error
    if not isinstance(value, dict):
        raise CanaryGateError(f"{schema} must be a JSON object")
    claimed = value.get("waiver_sha256")
    unsigned = {key: item for key, item in value.items() if key != "waiver_sha256"}
    actual = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if (
        value.get("schema_version") != schema
        or not isinstance(claimed, str)
        or not _SHA256_RE.fullmatch(claimed)
        or claimed != actual
    ):
        raise CanaryGateError(f"{schema} has an invalid schema/self-hash")
    return value


def _bound_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CanaryGateError(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise CanaryGateError(f"{label} path escapes the artifact root") from error
    return resolved


def _failed_performance_checks(summary: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        synthetic = float(summary["synthetic_goodput"])
        real = float(summary["real_goodput"])
        persisted = float(summary["persisted_goodput"])
        device = float(summary["device_output_goodput"])
        input_wait = float(summary["input_wait_fraction"])
    except (KeyError, TypeError, ValueError) as error:
        raise CanaryGateError("canary performance telemetry is malformed") from error
    if any(
        not math.isfinite(item) or item <= 0.0
        for item in (synthetic, real, persisted, device)
    ) or not math.isfinite(input_wait):
        raise CanaryGateError("canary performance telemetry is non-finite")
    failures: list[str] = []
    if real < 0.90 * synthetic:
        failures.append("real_to_synthetic_goodput")
    if input_wait >= 0.05:
        failures.append("input_wait_fraction")
    if persisted < 0.90 * device:
        failures.append("persisted_to_device_goodput")
    return tuple(failures)


def _validate_failed_canary(
    summary: Mapping[str, Any], train_metadata: Mapping[str, Any]
) -> tuple[str, ...]:
    if (
        summary.get("schema_version") != "gemma4-harvest-campaign-v2"
        or summary.get("split") != "train"
        or summary.get("phase") != "canary"
        or int(summary.get("stop_after_tokens", -1)) != 2_000_000
        or int(summary.get("captured_tokens", -1)) != 2_000_000
        or int(summary.get("new_tokens", -1)) != 2_000_000
        or int(summary.get("resumed_tokens", -1)) != 0
        or int(summary.get("process_count", -1)) != 4
        or int(summary.get("global_chips", -1)) != 16
        or summary.get("canary_passed") is not False
        or summary.get("capture_manifest_sha256") is not None
        or summary.get("manifest_path") is not None
        or int(summary.get("dropped_routed_tokens", -1)) != 0
        or int(summary.get("recompilations", -1)) != 0
        or summary.get("instrumentation_equal") is not True
        or int(summary.get("max_writer_backlog", 99)) >= 2
    ):
        raise CanaryGateError(
            "waiver source is not the exact safe, failed 2M-token canary"
        )
    for source, capture in (
        ("config_sha256", "config_sha256"),
        ("checkpoint_sha256", "checkpoint_sha256"),
        ("conversion_attestation_sha256", "conversion_attestation_sha256"),
        ("dataset_fingerprint", "dataset_fingerprint"),
        ("environment_sha256", "environment_sha256"),
    ):
        if summary.get(source) != train_metadata.get(capture):
            raise CanaryGateError(f"canary {source} differs from the sealed capture")
    failures = _failed_performance_checks(summary)
    if not failures:
        raise CanaryGateError("performance waiver source has no failed performance gate")
    return failures


def _semantic_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    # Match CampaignSummary.as_dict() without importing the TPU-heavy harvester.
    # Its JSON records already contain every dataclass field; normalizing tuples
    # to lists through a canonical JSON round trip yields the same mapping.
    return json.loads(_canonical(dict(summary)))


def validate_canary_gate(
    artifact_root: str | Path,
    *,
    train_manifest: Mapping[str, Any],
    production_campaign: Mapping[str, Any],
    selected_benchmark: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a passed or explicitly waived canary gate with durable evidence."""

    root = Path(artifact_root).resolve()
    train_metadata = train_manifest.get("metadata")
    if not isinstance(train_metadata, Mapping):
        raise CanaryGateError("sealed training capture lacks metadata")
    if production_campaign.get("canary_passed") is True:
        return {
            "status": "passed",
            "canary_passed": True,
            "waiver_sha256": None,
            "recovery_waiver_sha256": None,
            "failed_performance_checks": [],
            "headline_eligible": True,
        }
    if production_campaign.get("canary_passed") is not False:
        raise CanaryGateError("production campaign has no truthful canary outcome")

    waiver_path = root / "harvest" / "train-canary-waiver.json"
    recovery_path = root / "harvest" / "train-sealed-recovery-waiver.json"
    waiver = _self_hashed(waiver_path, CANARY_PERFORMANCE_WAIVER_SCHEMA)
    recovery = _self_hashed(recovery_path, SEALED_RECOVERY_WAIVER_SCHEMA)
    if (
        waiver.get("authorization_scope") != "canary-performance-only"
        or waiver.get("preserves_canary_passed_false") is not True
        or recovery.get("authorization_scope")
        != "sealed-summary-and-eval-recovery-only"
        or recovery.get("preserves_canary_passed_false") is not True
    ):
        raise CanaryGateError("canary authorization scope is broader than allowed")
    if waiver.get("thresholds") != {
        "real_to_synthetic_goodput": 0.9,
        "max_input_wait_fraction": 0.05,
        "persisted_to_device_goodput": 0.9,
        "max_writer_backlog_exclusive": 2,
    }:
        raise CanaryGateError("canary waiver changed the fixed performance gates")

    source_path = _bound_path(root, waiver.get("source_record"), "canary source")
    if _sha256_file(source_path) != waiver.get("source_record_sha256"):
        raise CanaryGateError("failed canary source record changed")
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CanaryGateError("failed canary source record is invalid") from error
    if not isinstance(source, Mapping):
        raise CanaryGateError("failed canary source record is not an object")
    failures = _validate_failed_canary(source, train_metadata)
    source_summary_sha256 = hashlib.sha256(
        _canonical(_semantic_summary(source))
    ).hexdigest()
    if waiver.get("source_summary_sha256") != source_summary_sha256:
        raise CanaryGateError("canary semantic summary differs from its waiver")
    observed_fields = (
        "captured_tokens",
        "synthetic_goodput",
        "real_goodput",
        "input_wait_fraction",
        "device_output_goodput",
        "persisted_goodput",
        "max_writer_backlog",
        "dropped_routed_tokens",
        "recompilations",
        "instrumentation_equal",
    )
    if waiver.get("observed") != {
        name: source.get(name) for name in observed_fields
    }:
        raise CanaryGateError("waiver observations differ from the failed canary")
    source_identity = {
        "source_config_sha256": "config_sha256",
        "source_checkpoint_sha256": "checkpoint_sha256",
        "source_conversion_attestation_sha256": "conversion_attestation_sha256",
        "source_dataset_fingerprint": "dataset_fingerprint",
        "source_environment_sha256": "environment_sha256",
    }
    if waiver.get("source_canary_passed") is not False or any(
        waiver.get(waiver_name) != source.get(source_name)
        for waiver_name, source_name in source_identity.items()
    ):
        raise CanaryGateError("waiver source identity differs from the failed canary")
    allowed = waiver.get("allowed_failed_performance_checks")
    if not isinstance(allowed, Sequence) or isinstance(allowed, (str, bytes)):
        raise CanaryGateError("waiver has no bounded performance-failure list")
    if sorted(str(item) for item in allowed) != sorted(failures):
        raise CanaryGateError("waiver does not match the observed performance failures")

    mutable_host_paths = {
        str((root / f"captures/train/hosts/host-p{index:05d}.json").resolve())
        for index in range(4)
    }
    bound_files = waiver.get("bound_file_sha256")
    if not isinstance(bound_files, Mapping):
        raise CanaryGateError("canary waiver lacks bound evidence files")
    normalized_bound: dict[str, str] = {}
    for value, expected in bound_files.items():
        path = _bound_path(root, value, "waiver evidence")
        if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
            raise CanaryGateError("waiver evidence has an invalid SHA-256")
        normalized_bound[str(path)] = expected
        if str(path) not in mutable_host_paths and _sha256_file(path) != expected:
            raise CanaryGateError(f"immutable canary evidence changed: {path}")
    if not mutable_host_paths.issubset(normalized_bound):
        raise CanaryGateError("canary waiver did not bind all four source host ledgers")
    storage_path = (root / "preflight" / "storage.json").resolve()
    storage_file_sha256 = normalized_bound.get(str(storage_path))
    if (
        not isinstance(storage_file_sha256, str)
        or not _SHA256_RE.fullmatch(storage_file_sha256)
        or _sha256_file(storage_path) != storage_file_sha256
    ):
        raise CanaryGateError(
            "canary waiver does not bind the exact storage attestation file"
        )
    try:
        storage_payload = json.loads(storage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CanaryGateError("bound storage attestation is invalid") from error
    if not isinstance(storage_payload, Mapping):
        raise CanaryGateError("bound storage attestation is not an object")
    storage_attestation_sha256 = storage_payload.get("attestation_sha256")
    unsigned_storage = {
        name: value
        for name, value in storage_payload.items()
        if name != "attestation_sha256"
    }
    if (
        not isinstance(storage_attestation_sha256, str)
        or not _SHA256_RE.fullmatch(storage_attestation_sha256)
        or hashlib.sha256(_canonical(unsigned_storage)).hexdigest()
        != storage_attestation_sha256
        or waiver.get("storage_attestation_sha256")
        != storage_attestation_sha256
        or train_metadata.get("storage_attestation_sha256")
        != storage_attestation_sha256
    ):
        raise CanaryGateError(
            "waiver, sealed capture, and bound storage attestation differ"
        )
    selected_path = (root / "benchmark" / "selected.json").resolve()
    if (
        normalized_bound.get(str(selected_path))
        != waiver.get("selected_benchmark_record_sha256")
        or _sha256_file(selected_path)
        != waiver.get("selected_benchmark_record_sha256")
    ):
        raise CanaryGateError("canary waiver does not bind the selected benchmark")

    inventory = waiver.get("verified_capture_objects")
    if not isinstance(inventory, Sequence) or isinstance(inventory, (str, bytes)):
        raise CanaryGateError("canary waiver lacks its verified object inventory")
    final_objects = {
        _canonical(entry["object"])
        for entry in train_manifest.get("shards", ())
        if isinstance(entry, Mapping) and isinstance(entry.get("object"), Mapping)
    }
    waived_objects: list[bytes] = []
    for value in inventory:
        if not isinstance(value, Mapping):
            raise CanaryGateError("canary object inventory is malformed")
        encoded = _canonical(dict(value))
        if encoded not in final_objects:
            raise CanaryGateError("canary object is absent from the sealed capture")
        waived_objects.append(encoded)
    if len(waived_objects) != 16 or len(set(waived_objects)) != 16:
        raise CanaryGateError("canary waiver must bind exactly 16 unique objects")

    if (
        recovery.get("source_canary_waiver_file_sha256") != _sha256_file(waiver_path)
        or recovery.get("source_canary_waiver_sha256") != waiver.get("waiver_sha256")
    ):
        raise CanaryGateError("sealed recovery does not descend from the canary waiver")
    production_ledgers = recovery.get("production_host_ledger_sha256")
    if not isinstance(production_ledgers, Mapping):
        raise CanaryGateError("sealed recovery lacks production host ledgers")
    normalized_ledgers = {
        str(_bound_path(root, path, "production host ledger")): digest
        for path, digest in production_ledgers.items()
    }
    if set(normalized_ledgers) != mutable_host_paths:
        raise CanaryGateError("sealed recovery must bind exactly four current host ledgers")
    for path, expected in normalized_ledgers.items():
        if not isinstance(expected, str) or _sha256_file(Path(path)) != expected:
            raise CanaryGateError(f"production host ledger changed: {path}")
    telemetry = recovery.get("production_telemetry_sha256")
    if not isinstance(telemetry, Mapping) or not telemetry:
        raise CanaryGateError("sealed recovery lacks production telemetry")
    for path, expected in telemetry.items():
        resolved = _bound_path(root, path, "production telemetry")
        if not isinstance(expected, str) or _sha256_file(resolved) != expected:
            raise CanaryGateError(f"production telemetry changed: {resolved}")

    manifest_path = root / "captures" / "train" / "manifest.json"
    manifest_sha256 = train_manifest.get("manifest_sha256")
    if (
        train_manifest.get("sealed") is not True
        or int(train_manifest.get("total_tokens", -1)) != 50_000_000
        or recovery.get("manifest_sha256") != manifest_sha256
        or _bound_path(root, recovery.get("manifest_path"), "sealed manifest")
        != manifest_path.resolve()
        or recovery.get("manifest_file_sha256") != _sha256_file(manifest_path)
        or int(recovery.get("manifest_total_tokens", -1)) != 50_000_000
        or int(recovery.get("manifest_shard_count", -1))
        != len(train_manifest.get("shards", ()))
        or production_campaign.get("capture_manifest_sha256") != manifest_sha256
    ):
        raise CanaryGateError("sealed recovery does not bind the exact 50M capture")
    benchmark_waiver = selected_benchmark.get("performance_waiver")
    benchmark_waiver_sha = (
        benchmark_waiver.get("waiver_sha256")
        if isinstance(benchmark_waiver, Mapping)
        else selected_benchmark.get("performance_waiver_sha256")
    )
    if source.get("performance_waiver_sha256") != benchmark_waiver_sha:
        raise CanaryGateError("canary and selected benchmark use different MFU waivers")
    if (
        waiver.get("source_benchmark_performance_waiver_sha256")
        != benchmark_waiver_sha
    ):
        raise CanaryGateError("waiver and selected benchmark use different MFU waivers")

    return {
        "status": "waived-unmet",
        "canary_passed": False,
        "waiver_sha256": waiver["waiver_sha256"],
        "waiver_file_sha256": _sha256_file(waiver_path),
        "recovery_waiver_sha256": recovery["waiver_sha256"],
        "recovery_waiver_file_sha256": _sha256_file(recovery_path),
        "source_record_sha256": waiver["source_record_sha256"],
        "failed_performance_checks": list(failures),
        "headline_eligible": False,
        "label": (
            "The durable 2M canary remains failed; only its measured performance "
            "checks were explicitly waived for this exact sealed capture."
        ),
    }


__all__ = [
    "CANARY_PERFORMANCE_WAIVER_SCHEMA",
    "CanaryGateError",
    "SEALED_RECOVERY_WAIVER_SCHEMA",
    "validate_canary_gate",
]
