"""Independent qualification for the SAE bounded streaming cache.

Activation capture and SAE consumption have different local-storage needs.
The former uses a rolling writer buffer; the latter streams a roughly 294 GB
immutable capture through a deliberately much smaller cache.  This module
qualifies that consumer cache without mutating the activation-capture storage
attestation already embedded in every shard.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any
import uuid

from moe_sae.object_store import (
    ImmutableObjectStore,
    ShardCache,
    object_identity_from_mapping,
)
from moe_sae.storage import CoordinationHooks, local_coordination_hooks


SAE_STORAGE_ATTESTATION_SCHEMA = "gemma4-sae-storage-qualification-v1"
SAE_STORAGE_SCOPE = "sae-training"
DEFAULT_STREAMING_CACHE_MAX_BYTES = 100 * 1024**3
MINIMUM_STREAMING_CACHE_MAX_BYTES = 100 * 1024**3
DEFAULT_STREAMING_CACHE_MINIMUM_FREE_AFTER_BYTES = 128 * 1024**3
SAE_CACHE_MATERIALIZATION_WORKERS = 4
SAE_GCS_DOWNLOAD_WORKERS = 16
SAE_GCS_DOWNLOAD_CHUNK_BYTES = 16 * 1024**2
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class SAEStorageQualificationError(RuntimeError):
    """The bounded SAE consumer cache is unsafe or insufficient."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sae_storage_attestation_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("attestation_sha256", None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _gather(
    hooks: CoordinationHooks, process_count: int, record: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    values = tuple(hooks.allgather(dict(record)))
    if len(values) != process_count:
        raise SAEStorageQualificationError("coordination returned the wrong process count")
    by_process: dict[int, dict[str, Any]] = {}
    for raw in values:
        if not isinstance(raw, Mapping):
            raise SAEStorageQualificationError("coordination returned a non-record")
        index = raw.get("process_index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < process_count
            or index in by_process
        ):
            raise SAEStorageQualificationError("coordination process indexes are invalid")
        by_process[index] = dict(raw)
    if len(by_process) != process_count:
        raise SAEStorageQualificationError("coordination omitted a process")
    return tuple(by_process[index] for index in range(process_count))


def _process_identity(hooks: CoordinationHooks) -> tuple[int, int]:
    index = hooks.process_index()
    count = hooks.process_count()
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or not 0 <= index < count
    ):
        raise SAEStorageQualificationError("invalid storage-qualification process identity")
    return index, count


def _run_id(
    hooks: CoordinationHooks, process_index: int, process_count: int
) -> str:
    proposal = uuid.uuid4().hex if process_index == 0 else ""
    values = tuple(hooks.allgather(proposal))
    if (
        len(values) != process_count
        or not isinstance(values[0], str)
        or re.fullmatch(r"[0-9a-f]{32}", values[0]) is None
        or any(value not in ("", values[0]) for value in values)
    ):
        raise SAEStorageQualificationError("cannot establish a qualification run ID")
    return values[0]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _qualify_host_isolation(
    root: Path,
    run_id: str,
    hooks: CoordinationHooks,
    process_index: int,
    process_count: int,
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Prove equal cache paths are process-private, not a shared mount."""

    path = root / f".sae-cache-isolation-{run_id}.json"
    payload = _canonical(
        {
            "schema_version": "gemma4-sae-cache-isolation-v1",
            "run_id": run_id,
            "process_index": process_index,
            "nonce": uuid.uuid4().hex,
        }
    )
    expected = hashlib.sha256(payload).hexdigest()
    error = ""
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(root)
    except OSError as exc:
        error = f"{type(exc).__name__}: {exc}"
    published = _gather(
        hooks,
        process_count,
        {"process_index": process_index, "expected": expected, "error": error},
    )
    hooks.barrier("sae-cache-isolation-published")
    observed = ""
    observe_error = ""
    try:
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        observe_error = f"{type(exc).__name__}: {exc}"
    observations = _gather(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "observed": observed,
            "error": observe_error,
        },
    )
    hooks.barrier("sae-cache-isolation-observed")
    try:
        path.unlink(missing_ok=True)
        _fsync_directory(root)
        cleanup_error = ""
    except OSError as exc:
        cleanup_error = f"{type(exc).__name__}: {exc}"
    cleaned = _gather(
        hooks,
        process_count,
        {"process_index": process_index, "error": cleanup_error},
    )
    expected_values = tuple(str(item["expected"]) for item in published)
    observed_values = tuple(str(item["observed"]) for item in observations)
    errors = tuple(
        f"process {item['process_index']}: {item['error']}"
        for records in (published, observations, cleaned)
        for item in records
        if item.get("error")
    )
    passed = (
        not errors
        and len(set(expected_values)) == process_count
        and expected_values == observed_values
    )
    return passed, expected_values, observed_values


def _token_partition(total: int, process_count: int, process_index: int) -> tuple[int, int]:
    """Mirror distributed._process_local_source without importing JAX."""

    if total < 0 or process_count <= 0 or not 0 <= process_index < process_count:
        raise ValueError("invalid token partition")
    return (
        (total * process_index) // process_count,
        (total * (process_index + 1)) // process_count,
    )


def _assigned_objects(
    manifests: Sequence[Mapping[str, Any]], process_index: int, process_count: int
) -> tuple[Any, ...]:
    """Return objects overlapping this process's exact contiguous token ranges."""

    objects: list[Any] = []
    seen: set[tuple[str, int]] = set()
    for manifest in manifests:
        shards = manifest.get("shards")
        total = manifest.get("total_tokens")
        if (
            not isinstance(shards, Sequence)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total <= 0
        ):
            raise SAEStorageQualificationError("capture manifest inventory is invalid")
        start, end = _token_partition(total, process_count, process_index)
        offset = 0
        for entry in shards:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("object"), Mapping):
                raise SAEStorageQualificationError(
                    "SAE cache qualification requires immutable remote shards"
                )
            tokens = entry.get("tokens")
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
                raise SAEStorageQualificationError("capture shard token count is invalid")
            shard_start, shard_end = offset, offset + tokens
            offset = shard_end
            if max(start, shard_start) >= min(end, shard_end):
                continue
            identity = object_identity_from_mapping(entry["object"])
            key = (identity.uri, identity.generation)
            if key not in seen:
                seen.add(key)
                objects.append(identity)
        if offset != total:
            raise SAEStorageQualificationError("capture shard counts do not cover the manifest")
    if not objects:
        raise SAEStorageQualificationError("a process has no assigned capture objects")
    return tuple(objects)


def qualify_sae_streaming_cache(
    cache_root: str | Path,
    manifests: Sequence[Mapping[str, Any]],
    object_store: ImmutableObjectStore,
    *,
    cache_max_bytes: int = DEFAULT_STREAMING_CACHE_MAX_BYTES,
    minimum_free_after_bytes: int = DEFAULT_STREAMING_CACHE_MINIMUM_FREE_AFTER_BYTES,
    minimum_aggregate_download_bytes_per_second: float,
    coordination: CoordinationHooks | None = None,
) -> dict[str, Any]:
    """Exercise real capture objects through a cache smaller than the corpus."""

    if (
        isinstance(cache_max_bytes, bool)
        or not isinstance(cache_max_bytes, int)
        or cache_max_bytes <= 0
        or isinstance(minimum_free_after_bytes, bool)
        or not isinstance(minimum_free_after_bytes, int)
        or minimum_free_after_bytes <= 0
        or not math.isfinite(minimum_aggregate_download_bytes_per_second)
        or minimum_aggregate_download_bytes_per_second < 0.0
    ):
        raise ValueError("SAE streaming-cache qualification bounds are invalid")
    hooks = coordination or local_coordination_hooks()
    process_index, process_count = _process_identity(hooks)
    run_id = _run_id(hooks, process_index, process_count)
    root = Path(cache_root).expanduser().resolve()
    prepare_error = ""
    free_before = -1
    filesystem = {"path": str(root), "device": -1, "inode": -1}
    try:
        root.mkdir(parents=True, exist_ok=True)
        stat = root.stat()
        free_before = int(shutil.disk_usage(root).free)
    except OSError as error:
        prepare_error = f"{type(error).__name__}: {error}"
    else:
        filesystem = {
            "path": str(root),
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
        }
    filesystem_records = _gather(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "cache_root": str(root),
            "filesystem": filesystem,
            "free_before_bytes": free_before,
            "error": prepare_error,
        },
    )
    preparation_failures = [
        f"process {record['process_index']}: {record['error']}"
        for record in filesystem_records
        if record.get("error")
    ]
    if preparation_failures:
        raise SAEStorageQualificationError(
            "cannot prepare every SAE cache root: "
            + "; ".join(preparation_failures)
        )
    isolation, expected_isolation, observed_isolation = _qualify_host_isolation(
        root, run_id, hooks, process_index, process_count
    )

    selected: tuple[Any, ...] = ()
    assignment_error = ""
    try:
        selected = _assigned_objects(manifests, process_index, process_count)
    except Exception as error:
        assignment_error = f"{type(error).__name__}: {error}"
    assignment_records = _gather(
        hooks,
        process_count,
        {"process_index": process_index, "error": assignment_error},
    )
    assignment_failures = [
        f"process {record['process_index']}: {record['error']}"
        for record in assignment_records
        if record.get("error")
    ]
    if assignment_failures:
        raise SAEStorageQualificationError(
            "cannot assign every SAE cache partition: "
            + "; ".join(assignment_failures)
        )
    assigned_bytes = sum(identity.size_bytes for identity in selected)
    concurrent_temporary_margin = sum(
        sorted(
            (identity.size_bytes for identity in selected), reverse=True
        )[:SAE_CACHE_MATERIALIZATION_WORKERS]
    )
    # The runtime cache bound remains the config-sealed 100 GiB.  Each host
    # owns only its contiguous train+eval partition (~68.6 GiB), so all assigned
    # objects must remain resident.  No host ever materializes the full corpus.
    required_free = (
        assigned_bytes + minimum_free_after_bytes + concurrent_temporary_margin
    )
    local_failure = ""
    try:
        existing_safetensors = tuple(root.glob("*.safetensors"))
        existing_partials = tuple(root.glob("*.partial"))
        if existing_safetensors or existing_partials:
            local_failure = (
                "cache root already contains materialized or partial objects; "
                "qualification requires a fresh root so measured download "
                "throughput is authoritative"
            )
        elif assigned_bytes > cache_max_bytes:
            local_failure = (
                f"assigned capture partition is {assigned_bytes} bytes, exceeding the "
                f"{cache_max_bytes}-byte cache bound"
            )
        elif free_before < required_free:
            local_failure = (
                f"only {free_before} bytes free; assigned-cache qualification requires "
                f"{required_free}"
            )
    except OSError as error:
        local_failure = f"{type(error).__name__}: {error}"

    materialized_bytes = 0
    elapsed = 0.0
    retained = 0
    all_assigned_resident = False
    failure_evidence: list[dict[str, Any]] = []
    if not local_failure:
        try:
            cache = ShardCache(root, object_store, max_bytes=cache_max_bytes)
            started = time.perf_counter()
            materialized_paths = list(
                cache.materialize_many(
                    selected,
                    max_workers=SAE_CACHE_MATERIALIZATION_WORKERS,
                    preserve_failure_evidence=True,
                )
            )
            elapsed = time.perf_counter() - started
            materialized_bytes = assigned_bytes
            retained = sum(
                item.stat().st_size
                for item in root.glob("*.safetensors")
                if item.is_file()
            )
            all_assigned_resident = all(path.is_file() for path in materialized_paths)
            if (
                not all_assigned_resident
                or retained != assigned_bytes
                or retained > cache_max_bytes
            ):
                raise SAEStorageQualificationError(
                    "not every assigned immutable capture object remained resident"
                )
        except Exception as error:  # aggregate every controller before failing
            elapsed = time.perf_counter() - started if "started" in locals() else 0.0
            local_failure = f"{type(error).__name__}: {error}"
            retained = sum(
                item.stat().st_size
                for item in root.glob("*.safetensors")
                if item.is_file()
            )
            materialized_bytes = retained
    try:
        for pattern, state in (
            ("*.safetensors", "committed"),
            ("*.partial", "partial"),
        ):
            for evidence_path in sorted(root.glob(pattern)):
                try:
                    evidence_size = evidence_path.stat().st_size
                except OSError:
                    evidence_size = -1
                failure_evidence.append(
                    {
                        "name": evidence_path.name,
                        "state": state,
                        "size_bytes": int(evidence_size),
                    }
                )
    except OSError as error:
        local_failure = local_failure or f"{type(error).__name__}: {error}"
    hooks.barrier("sae-cache-assignment-materialized")
    try:
        free_after = int(shutil.disk_usage(root).free)
    except OSError:
        free_after = -1
    records = _gather(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "cache_root": str(root),
            "filesystem": filesystem,
            "free_before_bytes": free_before,
            "free_after_bytes": free_after,
            "required_free_bytes": required_free,
            "concurrent_temporary_margin_bytes": concurrent_temporary_margin,
            "assigned_objects": len(selected),
            "assigned_bytes": assigned_bytes,
            "assigned_object_identities": [
                identity.as_dict() for identity in selected
            ],
            "materialized_bytes": materialized_bytes,
            "materialization_seconds": elapsed,
            "retained_bytes": retained,
            "all_assigned_resident": all_assigned_resident,
            "failure_evidence": failure_evidence if local_failure else [],
            "error": local_failure,
        },
    )
    errors = [
        f"process {record['process_index']}: {record['error']}"
        for record in records
        if record["error"]
    ]
    max_seconds = max(float(record["materialization_seconds"]) for record in records)
    total_download = sum(int(record["materialized_bytes"]) for record in records)
    aggregate_rate = total_download / max_seconds if max_seconds > 0.0 else 0.0
    if not isolation:
        errors.append("cache roots are not process-private")
    if any(
        int(record["free_after_bytes"]) < minimum_free_after_bytes
        for record in records
    ):
        errors.append("resident cache leaves less than the required free-space gate")
    if aggregate_rate < minimum_aggregate_download_bytes_per_second:
        errors.append(
            "aggregate verified capture download throughput is below the hard gate"
        )
    return {
        "run_id": run_id,
        "process_count": process_count,
        "passed": not errors,
        "failures": errors,
        "cache": {
            "directories": [str(record["cache_root"]) for record in records],
            "filesystems": [record["filesystem"] for record in filesystem_records],
            "host_isolation_proven": isolation,
            "expected_isolation_sha256_by_process": list(expected_isolation),
            "observed_isolation_sha256_by_process": list(observed_isolation),
            "max_bytes": cache_max_bytes,
            "minimum_free_after_bytes": minimum_free_after_bytes,
            "assignment_policy": "contiguous-token-range-v1",
            "materialization_workers": SAE_CACHE_MATERIALIZATION_WORKERS,
            "all_assigned_resident": all(
                bool(record["all_assigned_resident"]) for record in records
            ),
            "per_process": records,
        },
        "download": {
            "total_bytes": total_download,
            "max_elapsed_seconds": max_seconds,
            "aggregate_bytes_per_second": aggregate_rate,
            "minimum_aggregate_bytes_per_second": (
                minimum_aggregate_download_bytes_per_second
            ),
            "generation_pinned": True,
            "sha256_verified": True,
            "cold_cache_required": True,
            "outer_object_workers": SAE_CACHE_MATERIALIZATION_WORKERS,
            "inner_object_workers": getattr(
                object_store, "parallel_download_workers", None
            ),
            "chunk_bytes": getattr(
                object_store, "parallel_download_chunk_bytes", None
            ),
        },
    }


def build_sae_storage_attestation(
    qualification: Mapping[str, Any],
    *,
    config_sha256: str,
    activation_storage_attestation_sha256: str,
    producer_environment_sha256: str,
    consumer_environment: Mapping[str, Any],
    global_device_count: int,
    local_device_count: int,
    bulk_gcs_root: str,
    sae_checkpoint_root: str,
    capture_compatibility: Mapping[str, Any],
    created_at_utc: str,
) -> dict[str, Any]:
    """Build the immutable consumer attestation without altering its producer."""

    for label, value in (
        ("config", config_sha256),
        ("activation storage", activation_storage_attestation_sha256),
        ("producer environment", producer_environment_sha256),
        ("consumer environment", consumer_environment.get("sha256")),
        ("capture compatibility", capture_compatibility.get("compatibility_sha256")),
    ):
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise SAEStorageQualificationError(f"{label} identity is invalid")
    if not isinstance(created_at_utc, str) or not created_at_utc:
        raise SAEStorageQualificationError("coordinated creation timestamp is invalid")
    if qualification.get("passed") is not True or qualification.get("failures") not in (
        [],
        (),
    ):
        raise SAEStorageQualificationError("cannot attest a failed SAE cache qualification")
    payload = {
        "schema_version": SAE_STORAGE_ATTESTATION_SCHEMA,
        "scope": SAE_STORAGE_SCOPE,
        "created_at_utc": created_at_utc,
        "config_sha256": config_sha256,
        "producer": {
            "environment_sha256": producer_environment_sha256,
            "storage_attestation_sha256": activation_storage_attestation_sha256,
        },
        "consumer": {
            "environment": dict(consumer_environment),
            "environment_sha256": consumer_environment["sha256"],
        },
        "topology": {
            "process_count": int(qualification["process_count"]),
            "local_device_count": int(local_device_count),
            "global_device_count": int(global_device_count),
            "mesh": {"data": 4, "feature": 4},
        },
        "layout": {
            "cache_directories": list(qualification["cache"]["directories"]),
            "cache_max_bytes": int(qualification["cache"]["max_bytes"]),
            "bulk_gcs_root": bulk_gcs_root.rstrip("/"),
            "sae_checkpoint_root": sae_checkpoint_root.rstrip("/"),
        },
        "qualification": dict(qualification),
        "capture_compatibility": dict(capture_compatibility),
    }
    payload["attestation_sha256"] = sae_storage_attestation_digest(payload)
    return payload


def validate_sae_storage_attestation(
    payload: Mapping[str, Any],
    *,
    expected_config_sha256: str,
    expected_activation_storage_attestation_sha256: str,
    expected_consumer_environment_sha256: str,
    expected_process_count: int,
    expected_local_device_count: int,
    expected_global_device_count: int,
) -> str:
    """Validate self-hash, dual identities, topology, and bounded-cache proof."""

    claimed = payload.get("attestation_sha256")
    producer = payload.get("producer")
    consumer = payload.get("consumer")
    topology = payload.get("topology")
    layout = payload.get("layout")
    qualification = payload.get("qualification")
    compatibility = payload.get("capture_compatibility")
    if (
        payload.get("schema_version") != SAE_STORAGE_ATTESTATION_SCHEMA
        or payload.get("scope") != SAE_STORAGE_SCOPE
        or not isinstance(claimed, str)
        or not _SHA256_RE.fullmatch(claimed)
        or sae_storage_attestation_digest(payload) != claimed
        or payload.get("config_sha256") != expected_config_sha256
        or not all(
            isinstance(value, Mapping)
            for value in (producer, consumer, topology, layout, qualification, compatibility)
        )
    ):
        raise SAEStorageQualificationError("SAE storage attestation schema/self-hash is invalid")
    if (
        producer.get("storage_attestation_sha256")
        != expected_activation_storage_attestation_sha256
        or consumer.get("environment_sha256")
        != expected_consumer_environment_sha256
        or not isinstance(consumer.get("environment"), Mapping)
        or consumer["environment"].get("sha256")
        != expected_consumer_environment_sha256
        or topology.get("process_count") != expected_process_count
        or topology.get("local_device_count") != expected_local_device_count
        or topology.get("global_device_count") != expected_global_device_count
        or topology.get("mesh") != {"data": 4, "feature": 4}
        or expected_process_count != 4
        or expected_local_device_count != 4
        or expected_global_device_count != 16
    ):
        raise SAEStorageQualificationError("SAE storage producer/consumer identity changed")
    directories = layout.get("cache_directories")
    cache_max = layout.get("cache_max_bytes")
    cache = qualification.get("cache")
    download = qualification.get("download")
    per_process = cache.get("per_process", ()) if isinstance(cache, Mapping) else ()
    if (
        qualification.get("passed") is not True
        or qualification.get("failures") not in ([], ())
        or not isinstance(cache, Mapping)
        or not isinstance(download, Mapping)
        or not isinstance(directories, list)
        or len(directories) != expected_process_count
        or directories != cache.get("directories")
        or isinstance(cache_max, bool)
        or not isinstance(cache_max, int)
        or cache_max < MINIMUM_STREAMING_CACHE_MAX_BYTES
        or cache.get("max_bytes") != cache_max
        or cache_max != DEFAULT_STREAMING_CACHE_MAX_BYTES
        or cache.get("minimum_free_after_bytes")
        != DEFAULT_STREAMING_CACHE_MINIMUM_FREE_AFTER_BYTES
        or cache.get("assignment_policy") != "contiguous-token-range-v1"
        or cache.get("materialization_workers")
        != SAE_CACHE_MATERIALIZATION_WORKERS
        or cache.get("all_assigned_resident") is not True
        or cache.get("host_isolation_proven") is not True
        or not isinstance(per_process, (list, tuple))
        or len(per_process) != expected_process_count
        or any(
            not isinstance(record, Mapping)
            or record.get("process_index") != index
            or record.get("cache_root") != directories[index]
            or record.get("error") not in (None, "")
            or record.get("all_assigned_resident") is not True
            or int(record.get("assigned_objects", -1)) <= 0
            or int(record.get("assigned_bytes", -1)) <= 0
            or int(record.get("materialized_bytes", -1))
            != int(record.get("assigned_bytes", -2))
            or int(record.get("retained_bytes", -1))
            != int(record.get("assigned_bytes", -2))
            or int(record.get("required_free_bytes", -1))
            != int(record.get("assigned_bytes", -2))
            + int(record.get("concurrent_temporary_margin_bytes", -3))
            + DEFAULT_STREAMING_CACHE_MINIMUM_FREE_AFTER_BYTES
            for index, record in enumerate(per_process)
        )
        or download.get("generation_pinned") is not True
        or download.get("sha256_verified") is not True
        or download.get("cold_cache_required") is not True
        or download.get("outer_object_workers")
        != SAE_CACHE_MATERIALIZATION_WORKERS
        or download.get("inner_object_workers") != SAE_GCS_DOWNLOAD_WORKERS
        or download.get("chunk_bytes") != SAE_GCS_DOWNLOAD_CHUNK_BYTES
        or any(
            int(record.get("free_after_bytes", -1))
            < DEFAULT_STREAMING_CACHE_MINIMUM_FREE_AFTER_BYTES
            for record in per_process
        )
        or float(download.get("aggregate_bytes_per_second", -1.0))
        < float(download.get("minimum_aggregate_bytes_per_second", math.inf))
        or compatibility.get("compatible") is not True
        or not _SHA256_RE.fullmatch(
            str(compatibility.get("compatibility_sha256", ""))
        )
    ):
        raise SAEStorageQualificationError("SAE bounded streaming-cache proof is invalid")
    return claimed


def validate_sae_storage_assignments(
    payload: Mapping[str, Any], manifests: Sequence[Mapping[str, Any]]
) -> None:
    """Recompute every controller's exact contiguous immutable-object inventory."""

    qualification = payload.get("qualification")
    cache = qualification.get("cache") if isinstance(qualification, Mapping) else None
    records = cache.get("per_process") if isinstance(cache, Mapping) else None
    process_count = qualification.get("process_count") if isinstance(qualification, Mapping) else None
    if (
        process_count != 4
        or not isinstance(records, (list, tuple))
        or len(records) != process_count
    ):
        raise SAEStorageQualificationError("SAE assignment evidence is malformed")
    for process_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise SAEStorageQualificationError("SAE assignment record is malformed")
        expected = _assigned_objects(manifests, process_index, process_count)
        expected_mappings = [identity.as_dict() for identity in expected]
        assigned_bytes = sum(identity.size_bytes for identity in expected)
        margin = sum(
            sorted(
                (identity.size_bytes for identity in expected), reverse=True
            )[:SAE_CACHE_MATERIALIZATION_WORKERS]
        )
        if (
            record.get("assigned_object_identities") != expected_mappings
            or record.get("assigned_objects") != len(expected)
            or record.get("assigned_bytes") != assigned_bytes
            or record.get("concurrent_temporary_margin_bytes") != margin
        ):
            raise SAEStorageQualificationError(
                f"SAE cache assignment changed for process {process_index}"
            )


def validate_local_sae_cache_residency(
    payload: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]],
    *,
    process_index: int,
) -> dict[str, int]:
    """Require this host's attested cache inventory to remain fully resident.

    The qualification is a statement about a particular process-private local
    cache, not permission to refill that cache later.  This lightweight check
    verifies the complete filename/size inventory and current free-space gate.
    SHA-256 is checked by the resident-only cache immediately before each
    train/eval partition is consumed.
    """

    qualification = payload.get("qualification")
    cache = qualification.get("cache") if isinstance(qualification, Mapping) else None
    layout = payload.get("layout")
    records = cache.get("per_process") if isinstance(cache, Mapping) else None
    directories = layout.get("cache_directories") if isinstance(layout, Mapping) else None
    process_count = (
        qualification.get("process_count")
        if isinstance(qualification, Mapping)
        else None
    )
    if (
        isinstance(process_index, bool)
        or not isinstance(process_index, int)
        or not isinstance(process_count, int)
        or not 0 <= process_index < process_count
        or not isinstance(records, (list, tuple))
        or len(records) != process_count
        or not isinstance(directories, list)
        or len(directories) != process_count
        or not isinstance(cache, Mapping)
    ):
        raise SAEStorageQualificationError("SAE cache residency evidence is malformed")

    expected = _assigned_objects(manifests, process_index, process_count)
    root = Path(str(directories[process_index])).expanduser().resolve()
    record = records[process_index]
    if not isinstance(record, Mapping) or record.get("cache_root") != str(root):
        raise SAEStorageQualificationError("SAE cache root changed after qualification")
    try:
        if not root.is_dir():
            raise SAEStorageQualificationError(
                "qualified SAE cache directory is absent; requalify a fresh cache"
            )
        expected_sizes = {
            f"{identity.sha256}.safetensors": identity.size_bytes
            for identity in expected
        }
        actual_paths = tuple(root.glob("*.safetensors"))
        actual_names = {path.name for path in actual_paths if path.is_file()}
        if actual_names != set(expected_sizes) or len(actual_paths) != len(actual_names):
            raise SAEStorageQualificationError(
                "qualified SAE cache inventory is absent or changed; requalify a "
                "fresh cache before continuing"
            )
        for path in actual_paths:
            if path.is_symlink() or path.stat().st_size != expected_sizes[path.name]:
                raise SAEStorageQualificationError(
                    "qualified SAE cache object size/type changed; requalify a fresh "
                    "cache before continuing"
                )
        if tuple(root.glob("*.partial")):
            raise SAEStorageQualificationError(
                "qualified SAE cache contains partial transfer evidence"
            )
        free_bytes = int(shutil.disk_usage(root).free)
    except OSError as error:
        raise SAEStorageQualificationError(
            "cannot inspect the qualified SAE cache"
        ) from error
    minimum_free = int(cache.get("minimum_free_after_bytes", -1))
    if free_bytes < minimum_free:
        raise SAEStorageQualificationError(
            f"qualified SAE cache now has only {free_bytes} free bytes; require "
            f"{minimum_free}"
        )
    resident_bytes = sum(expected_sizes.values())
    if (
        resident_bytes != int(record.get("assigned_bytes", -1))
        or resident_bytes > int(cache.get("max_bytes", -1))
    ):
        raise SAEStorageQualificationError("qualified SAE cache byte inventory changed")
    return {
        "objects": len(expected_sizes),
        "bytes": resident_bytes,
        "free_bytes": free_bytes,
    }


__all__ = [
    "DEFAULT_STREAMING_CACHE_MAX_BYTES",
    "DEFAULT_STREAMING_CACHE_MINIMUM_FREE_AFTER_BYTES",
    "MINIMUM_STREAMING_CACHE_MAX_BYTES",
    "SAE_CACHE_MATERIALIZATION_WORKERS",
    "SAE_GCS_DOWNLOAD_CHUNK_BYTES",
    "SAE_GCS_DOWNLOAD_WORKERS",
    "SAE_STORAGE_ATTESTATION_SCHEMA",
    "SAE_STORAGE_SCOPE",
    "SAEStorageQualificationError",
    "build_sae_storage_attestation",
    "qualify_sae_streaming_cache",
    "sae_storage_attestation_digest",
    "validate_sae_storage_attestation",
    "validate_sae_storage_assignments",
    "validate_local_sae_cache_residency",
]
