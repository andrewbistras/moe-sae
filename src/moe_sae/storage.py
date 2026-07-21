"""Qualification for the split control, buffer, and object-storage planes.

The pipeline deliberately keeps only small coordination records on a shared
POSIX filesystem.  Activation shards are serialized to host-local storage and
published as generation-pinned immutable GCS objects.  This module qualifies
those independent contracts without importing JAX until production collective
hooks are explicitly requested.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import base64
import errno
import fcntl
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
    ObjectIdentity,
    SourceFingerprint,
    _fingerprint_from_components,
    join_gcs_uri,
    object_identity_from_mapping,
    sha256_file,
)


STORAGE_ATTESTATION_SCHEMA = "gemma4-storage-planes-qualification-v6"
ORBAX_QUALIFICATION_SCHEMA = "gemma4-orbax-storage-qualification-v1"
ACTIVATION_CAPTURE_SCOPE = "activation-capture"
MINIMUM_ACTIVATION_LOCAL_FREE_BYTES = 24 * 1024**3
MINIMUM_ACTIVATION_LOCAL_WRITE_BYTES_PER_SECOND = 4_000_000_000.0
MINIMUM_ACTIVATION_GCS_DOWNLOAD_BYTES_PER_SECOND = 0.0
DEFAULT_MINIMUM_CONTROL_FREE_BYTES = 10_000_000_000
DEFAULT_MINIMUM_LOCAL_BUFFER_FREE_BYTES = 128 * 1024**3
DEFAULT_GCS_PROBE_BYTES_PER_PROCESS = 8 * 1024**3
DEFAULT_GCS_PROBE_BLOCK_BYTES = 8 * 1024**2
DEFAULT_MINIMUM_GCS_BYTES_PER_SECOND = 4_000_000_000.0
DEFAULT_MINIMUM_LOCAL_WRITE_BYTES_PER_SECOND = 4_000_000_000.0

_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,128}$")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class StorageQualificationError(RuntimeError):
    """One or more storage planes cannot safely support production."""


@dataclass(frozen=True, slots=True)
class CoordinationHooks:
    """Collectives used by :func:`qualify_storage_planes`.

    ``allgather`` returns one JSON-compatible value per process in arbitrary
    order; records are normalized by their explicit process index.  Every
    process must call the collectives in the same order.
    """

    process_index: Callable[[], int]
    process_count: Callable[[], int]
    barrier: Callable[[str], None]
    allgather: Callable[[Any], Sequence[Any]]


@dataclass(frozen=True, slots=True)
class StoragePlanesQualificationConfig:
    """Hard gates and probe size for the three-plane storage contract."""

    minimum_control_free_bytes: int = DEFAULT_MINIMUM_CONTROL_FREE_BYTES
    minimum_local_buffer_free_bytes: int = DEFAULT_MINIMUM_LOCAL_BUFFER_FREE_BYTES
    minimum_aggregate_local_write_bytes_per_second: float = (
        DEFAULT_MINIMUM_LOCAL_WRITE_BYTES_PER_SECOND
    )
    gcs_probe_bytes_per_process: int = DEFAULT_GCS_PROBE_BYTES_PER_PROCESS
    gcs_probe_block_bytes: int = DEFAULT_GCS_PROBE_BLOCK_BYTES
    minimum_aggregate_upload_bytes_per_second: float = (
        DEFAULT_MINIMUM_GCS_BYTES_PER_SECOND
    )
    minimum_aggregate_download_bytes_per_second: float = (
        DEFAULT_MINIMUM_GCS_BYTES_PER_SECOND
    )
    cleanup: bool = True

    def __post_init__(self) -> None:
        integers = {
            "minimum_control_free_bytes": self.minimum_control_free_bytes,
            "minimum_local_buffer_free_bytes": self.minimum_local_buffer_free_bytes,
            "gcs_probe_bytes_per_process": self.gcs_probe_bytes_per_process,
            "gcs_probe_block_bytes": self.gcs_probe_block_bytes,
        }
        for name, value in integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.gcs_probe_bytes_per_process <= 0:
            raise ValueError("gcs_probe_bytes_per_process must be positive")
        if self.gcs_probe_block_bytes <= 0:
            raise ValueError("gcs_probe_block_bytes must be positive")
        if self.gcs_probe_block_bytes > self.gcs_probe_bytes_per_process:
            raise ValueError("gcs_probe_block_bytes cannot exceed the probe size")
        for name, value in (
            (
                "minimum_aggregate_local_write_bytes_per_second",
                self.minimum_aggregate_local_write_bytes_per_second,
            ),
            (
                "minimum_aggregate_upload_bytes_per_second",
                self.minimum_aggregate_upload_bytes_per_second,
            ),
            (
                "minimum_aggregate_download_bytes_per_second",
                self.minimum_aggregate_download_bytes_per_second,
            ),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TransferMeasurement:
    """One synchronized aggregate storage-I/O measurement."""

    total_bytes: int
    elapsed_seconds: float
    bytes_per_second: float
    per_process_bytes: tuple[int, ...]
    per_process_seconds: tuple[float, ...]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ControlPlaneQualification:
    directory: str
    work_directory: str
    filesystem_identities: tuple[Mapping[str, Any], ...]
    free_bytes_by_process: tuple[int, ...]
    minimum_free_bytes: int
    flock_correct: bool
    atomic_rename_visible: bool
    disjoint_files_visible: bool


@dataclass(frozen=True, slots=True)
class LocalBufferQualification:
    directories: tuple[str, ...]
    filesystem_identities: tuple[Mapping[str, Any], ...]
    free_bytes_by_process: tuple[int, ...]
    minimum_free_bytes: int
    required_probe_free_bytes: int
    host_isolation_proven: bool
    expected_probe_sha256_by_process: tuple[str, ...]
    observed_probe_sha256_by_process: tuple[str, ...]
    write: TransferMeasurement | None
    write_cleanup_complete: bool


@dataclass(frozen=True, slots=True)
class ObjectPlaneQualification:
    prefix: str
    objects: tuple[Mapping[str, Any], ...]
    upload: TransferMeasurement | None
    download: TransferMeasurement | None
    cleanup_complete: bool


@dataclass(frozen=True, slots=True)
class OrbaxCheckpointQualification:
    """Collective save/finalize/restore proof for the checkpoint prefix."""

    payload_uri: str
    process_count: int
    global_device_count: int
    expected_local_sha256_by_process: tuple[str, ...]
    restored_local_sha256_by_process: tuple[str, ...]
    save_seconds_by_process: tuple[float, ...]
    restore_seconds_by_process: tuple[float, ...]
    max_save_seconds: float
    max_restore_seconds: float
    finalized: bool
    metadata_round_trip: bool
    failures: tuple[str, ...]
    schema_version: str = ORBAX_QUALIFICATION_SCHEMA

    @property
    def passed(self) -> bool:
        return not self.failures

    def require_pass(self) -> OrbaxCheckpointQualification:
        if self.failures:
            raise StorageQualificationError("; ".join(self.failures))
        return self


@dataclass(frozen=True, slots=True)
class StoragePlanesQualificationResult:
    """An identical result returned on every participating process."""

    run_id: str
    process_count: int
    control: ControlPlaneQualification
    local_buffer: LocalBufferQualification
    gcs: ObjectPlaneQualification
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def require_pass(self) -> StoragePlanesQualificationResult:
        if self.failures:
            raise StorageQualificationError("; ".join(self.failures))
        return self


def local_coordination_hooks() -> CoordinationHooks:
    """Return single-process hooks for local smoke tests."""

    return CoordinationHooks(
        process_index=lambda: 0,
        process_count=lambda: 1,
        barrier=lambda _label: None,
        allgather=lambda value: (value,),
    )


def jax_coordination_hooks() -> CoordinationHooks:
    """Return production hooks backed by JAX multi-host collectives."""

    try:
        import jax
        import numpy as np
        from jax.experimental import multihost_utils
    except ImportError as exc:  # pragma: no cover - optional production path
        raise StorageQualificationError(
            "JAX multi-host support is unavailable"
        ) from exc

    collective_index = 0

    def barrier(label: str) -> None:
        nonlocal collective_index
        current = collective_index
        collective_index += 1
        multihost_utils.sync_global_devices(
            f"moe-sae-storage-{current:04d}-{label}"
        )

    def allgather(value: Any) -> tuple[Any, ...]:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        lengths = np.asarray(
            multihost_utils.process_allgather(
                np.asarray([len(payload)], dtype=np.int32), tiled=False
            )
        ).reshape(-1)
        count = int(jax.process_count())
        if lengths.size != count:
            raise StorageQualificationError(
                "JAX allgather returned an unexpected process count"
            )
        maximum = int(lengths.max(initial=0))
        encoded = np.zeros((maximum,), dtype=np.uint8)
        if payload:
            encoded[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        gathered = np.asarray(
            multihost_utils.process_allgather(encoded, tiled=False)
        ).reshape(count, maximum)
        return tuple(
            json.loads(bytes(gathered[index, : int(length)]))
            for index, length in enumerate(lengths)
        )

    return CoordinationHooks(
        process_index=jax.process_index,
        process_count=jax.process_count,
        barrier=barrier,
        allgather=allgather,
    )


def qualify_storage_planes(
    control_directory: str | os.PathLike[str],
    local_buffer_directory: str | os.PathLike[str],
    object_prefix: str,
    object_store: ImmutableObjectStore,
    *,
    config: StoragePlanesQualificationConfig | None = None,
    coordination: CoordinationHooks | None = None,
) -> StoragePlanesQualificationResult:
    """Qualify the control volume, local buffers, and immutable object plane.

    The control volume sees only tiny semantic probes.  Each process creates an
    incompressible local object, conditionally uploads it, and then downloads a
    different process's generation-pinned object (cyclically).  Transfer timing
    includes synchronization and integrity verification but excludes local
    probe generation.  Probe objects are deleted by exact generation.
    """

    config = config or StoragePlanesQualificationConfig()
    hooks = coordination or local_coordination_hooks()
    process_index, process_count = _validate_process_identity(hooks)
    control_root = Path(control_directory).expanduser().resolve()
    local_root = Path(local_buffer_directory).expanduser().resolve()
    object_prefix = object_prefix.rstrip("/")
    # Validate the bucket and every existing prefix component without requiring
    # the prefix itself to identify an object.
    join_gcs_uri(object_prefix, "qualification-probe")

    _prepare_directories(
        control_root, local_root, hooks, process_index, process_count
    )
    run_id = _coordinated_run_id(hooks, process_index, process_count)
    work = control_root / f".moe-sae-storage-qualification-{run_id}"
    _prepare_shared_work_directory(
        work, hooks, process_index, process_count
    )

    failures: list[str] = []
    control_free, control_free_errors = _measure_free_space(
        control_root, hooks, process_index, process_count
    )
    local_free, local_free_errors = _measure_free_space(
        local_root, hooks, process_index, process_count
    )
    control_identities = _filesystem_identities(
        control_root, hooks, process_index, process_count
    )
    local_identities = _filesystem_identities(
        local_root, hooks, process_index, process_count
    )
    (
        local_isolation_proven,
        expected_local_probe_sha256,
        observed_local_probe_sha256,
        local_isolation_errors,
    ) = _qualify_local_buffer_isolation(
        local_root, run_id, hooks, process_index, process_count
    )

    if control_free_errors:
        failures.append(
            "control-plane free-space query failed: "
            + "; ".join(control_free_errors)
        )
    elif min(control_free) < config.minimum_control_free_bytes:
        failures.append(
            "control-plane free space below threshold: "
            f"minimum={min(control_free)} "
            f"required={config.minimum_control_free_bytes}"
        )
    required_local = 2 * config.gcs_probe_bytes_per_process
    if local_free_errors:
        failures.append(
            "local-buffer free-space query failed: " + "; ".join(local_free_errors)
        )
    elif min(local_free) < config.minimum_local_buffer_free_bytes:
        failures.append(
            "local-buffer free space below threshold: "
            f"minimum={min(local_free)} "
            f"required={config.minimum_local_buffer_free_bytes}"
        )
    elif min(local_free) < required_local:
        failures.append(
            "local buffer cannot hold upload and download probes: "
            f"minimum={min(local_free)} required={required_local}"
        )
    if not local_isolation_proven:
        failures.append(
            "local-buffer host isolation failed"
            + _error_suffix(local_isolation_errors)
        )

    local_capacity_ok = (
        not local_free_errors
        and min(local_free)
        >= max(config.minimum_local_buffer_free_bytes, required_local)
    )
    if local_capacity_ok and local_isolation_proven:
        local_write, local_write_cleanup_complete = _qualify_local_write(
            local_root,
            run_id,
            config.gcs_probe_bytes_per_process,
            config.gcs_probe_block_bytes,
            hooks,
            process_index,
            process_count,
        )
        if local_write.errors:
            failures.append(
                "local-buffer write probe failed: "
                + "; ".join(local_write.errors)
            )
        elif (
            local_write.bytes_per_second
            < config.minimum_aggregate_local_write_bytes_per_second
        ):
            failures.append(
                "aggregate local-buffer write throughput below threshold: "
                f"actual={local_write.bytes_per_second:.3f} "
                "required="
                f"{config.minimum_aggregate_local_write_bytes_per_second:.3f} B/s"
            )
        if not local_write_cleanup_complete:
            failures.append("local-buffer write-probe cleanup failed")
    else:
        local_write = None
        local_write_cleanup_complete = True

    flock_correct, flock_errors = _qualify_flock(
        work, hooks, process_index, process_count
    )
    if not flock_correct:
        failures.append("cross-process flock failed" + _error_suffix(flock_errors))
    rename_visible, rename_errors = _qualify_atomic_rename(
        work, run_id, hooks, process_index, process_count
    )
    if not rename_visible:
        failures.append(
            "atomic rename/visibility failed" + _error_suffix(rename_errors)
        )
    disjoint_visible, disjoint_errors = _qualify_disjoint_files(
        work, run_id, hooks, process_index, process_count
    )
    if not disjoint_visible:
        failures.append(
            "per-process disjoint-file visibility failed"
            + _error_suffix(disjoint_errors)
        )

    enough_local = (
        local_capacity_ok
        and local_isolation_proven
        and local_write is not None
        and not local_write.errors
        and local_write.bytes_per_second
        >= config.minimum_aggregate_local_write_bytes_per_second
        and local_write_cleanup_complete
    )
    if enough_local:
        objects, upload, download, cleanup_complete, object_failures = (
            _qualify_object_plane(
                local_root,
                object_prefix,
                object_store,
                run_id,
                config,
                hooks,
                process_index,
                process_count,
            )
        )
        failures.extend(object_failures)
    else:
        objects = ()
        upload = None
        download = None
        cleanup_complete = True

    hooks.barrier("control-cleanup-ready")
    cleanup_error = ""
    if config.cleanup and process_index == 0:
        try:
            shutil.rmtree(work)
        except OSError as exc:
            cleanup_error = _format_error(exc)
    cleanup_records = _gather_records(
        hooks,
        process_count,
        {"process_index": process_index, "error": cleanup_error},
    )
    hooks.barrier("control-cleanup-complete")
    cleanup_errors = _record_errors(cleanup_records)
    if cleanup_errors:
        failures.append(
            "control-plane qualification cleanup failed: "
            + "; ".join(cleanup_errors)
        )

    return StoragePlanesQualificationResult(
        run_id=run_id,
        process_count=process_count,
        control=ControlPlaneQualification(
            directory=str(control_root),
            work_directory=str(work),
            filesystem_identities=control_identities,
            free_bytes_by_process=control_free,
            minimum_free_bytes=config.minimum_control_free_bytes,
            flock_correct=flock_correct,
            atomic_rename_visible=rename_visible,
            disjoint_files_visible=disjoint_visible,
        ),
        local_buffer=LocalBufferQualification(
            directories=tuple(str(item["path"]) for item in local_identities),
            filesystem_identities=local_identities,
            free_bytes_by_process=local_free,
            minimum_free_bytes=config.minimum_local_buffer_free_bytes,
            required_probe_free_bytes=required_local,
            host_isolation_proven=local_isolation_proven,
            expected_probe_sha256_by_process=expected_local_probe_sha256,
            observed_probe_sha256_by_process=observed_local_probe_sha256,
            write=local_write,
            write_cleanup_complete=local_write_cleanup_complete,
        ),
        gcs=ObjectPlaneQualification(
            prefix=object_prefix,
            objects=objects,
            upload=upload,
            download=download,
            cleanup_complete=cleanup_complete,
        ),
        failures=tuple(failures),
    )


def build_storage_attestation(
    result: StoragePlanesQualificationResult,
    config: StoragePlanesQualificationConfig,
    *,
    config_sha256: str,
    environment_sha256: str,
    global_device_count: int,
    scope: str = ACTIVATION_CAPTURE_SCOPE,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical, self-hashed production storage attestation.

    The current production qualifier proves only the rolling local buffer used
    while farming activations.  It deliberately does not authorize the much
    larger local cache required by SAE training and evaluation.
    """

    result.require_pass()
    for label, digest in (
        ("config_sha256", config_sha256),
        ("environment_sha256", environment_sha256),
    ):
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"{label} must be lowercase 64-character hex")
    if (
        isinstance(global_device_count, bool)
        or not isinstance(global_device_count, int)
        or global_device_count <= 0
    ):
        raise ValueError("global_device_count must be a positive integer")
    if scope != ACTIVATION_CAPTURE_SCOPE:
        raise ValueError(
            f"storage attestation scope must be {ACTIVATION_CAPTURE_SCOPE!r}"
        )
    payload: dict[str, Any] = {
        "schema_version": STORAGE_ATTESTATION_SCHEMA,
        "scope": scope,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "environment_sha256": environment_sha256,
        "topology": {
            "process_count": result.process_count,
            "global_device_count": global_device_count,
        },
        "contract": asdict(config),
        "qualification": asdict(result),
        "metadata": dict(metadata or {}),
    }
    # Prove that all payload values are canonical-JSON compatible before the
    # caller attempts an atomic control-plane publication.
    _canonical_json(payload)
    payload["attestation_sha256"] = storage_attestation_digest(payload)
    return payload


def storage_attestation_digest(payload: Mapping[str, Any]) -> str:
    """Hash every attestation field except the self-hash itself."""

    content = dict(payload)
    content.pop("attestation_sha256", None)
    return hashlib.sha256(_canonical_json(content)).hexdigest()


def validate_storage_attestation(
    payload: Mapping[str, Any],
    *,
    expected_config_sha256: str | None = None,
    expected_environment_sha256: str | None = None,
    expected_control_directory: str | os.PathLike[str] | None = None,
    expected_local_directories: Sequence[str | os.PathLike[str]] | None = None,
    expected_object_prefix: str | None = None,
    expected_scope: str | None = None,
) -> str:
    """Validate the self-hash, pass state, and optional invocation bindings."""

    if payload.get("schema_version") != STORAGE_ATTESTATION_SCHEMA:
        raise StorageQualificationError("storage attestation schema is incompatible")
    scope = payload.get("scope")
    if scope != ACTIVATION_CAPTURE_SCOPE:
        raise StorageQualificationError("storage attestation scope is incompatible")
    if expected_scope is not None and scope != expected_scope:
        raise StorageQualificationError("storage attestation scope mismatch")
    claimed = payload.get("attestation_sha256")
    if not isinstance(claimed, str) or not _SHA256_RE.fullmatch(claimed):
        raise StorageQualificationError("storage attestation has no valid self-hash")
    if storage_attestation_digest(payload) != claimed:
        raise StorageQualificationError("storage attestation self-hash mismatch")
    contract = payload.get("contract")
    qualification = payload.get("qualification")
    if not isinstance(qualification, Mapping) or qualification.get("failures") not in (
        [],
        (),
    ):
        raise StorageQualificationError("storage attestation did not pass")
    if not isinstance(contract, Mapping):
        raise StorageQualificationError("storage attestation contract is malformed")
    expected_scalars = (
        ("config_sha256", expected_config_sha256),
        ("environment_sha256", expected_environment_sha256),
    )
    for field, expected in expected_scalars:
        if expected is not None and payload.get(field) != expected:
            raise StorageQualificationError(f"storage attestation {field} mismatch")
    control = qualification.get("control")
    local_buffer = qualification.get("local_buffer")
    gcs = qualification.get("gcs")
    if not all(isinstance(item, Mapping) for item in (control, local_buffer, gcs)):
        raise StorageQualificationError("storage attestation plane records are malformed")
    expected_local = tuple(
        local_buffer.get("expected_probe_sha256_by_process", ())
    )
    observed_local = tuple(
        local_buffer.get("observed_probe_sha256_by_process", ())
    )
    process_count = qualification.get("process_count")
    if (
        local_buffer.get("host_isolation_proven") is not True
        or isinstance(process_count, bool)
        or not isinstance(process_count, int)
        or process_count <= 0
        or len(expected_local) != process_count
        or expected_local != observed_local
        or any(not isinstance(item, str) or not _SHA256_RE.fullmatch(item) for item in expected_local)
        or len(set(expected_local)) != process_count
    ):
        raise StorageQualificationError(
            "storage attestation does not prove host-local buffer isolation"
        )
    local_write = local_buffer.get("write")
    try:
        local_probe_bytes = int(contract["gcs_probe_bytes_per_process"])
        minimum_local_write = float(
            contract["minimum_aggregate_local_write_bytes_per_second"]
        )
        local_write_total = int(local_write["total_bytes"])
        local_write_seconds = float(local_write["elapsed_seconds"])
        local_write_rate = float(local_write["bytes_per_second"])
        local_write_process_bytes = tuple(
            int(item) for item in local_write["per_process_bytes"]
        )
        local_write_process_seconds = tuple(
            float(item) for item in local_write["per_process_seconds"]
        )
        local_write_errors = tuple(local_write["errors"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise StorageQualificationError(
            "storage attestation local-write proof is malformed"
        ) from exc
    if (
        not isinstance(local_write, Mapping)
        or local_buffer.get("write_cleanup_complete") is not True
        or local_probe_bytes <= 0
        or not math.isfinite(minimum_local_write)
        or minimum_local_write < 0.0
        or local_write_total != process_count * local_probe_bytes
        or local_write_process_bytes != (local_probe_bytes,) * process_count
        or len(local_write_process_seconds) != process_count
        or any(
            not math.isfinite(item) or item <= 0.0
            for item in local_write_process_seconds
        )
        or not math.isfinite(local_write_seconds)
        or local_write_seconds <= 0.0
        or not math.isclose(
            local_write_seconds,
            max(local_write_process_seconds),
            rel_tol=1e-12,
        )
        or not math.isfinite(local_write_rate)
        or not math.isclose(
            local_write_rate,
            local_write_total / local_write_seconds,
            rel_tol=1e-12,
        )
        or local_write_rate < minimum_local_write
        or local_write_errors
    ):
        raise StorageQualificationError(
            "storage attestation does not prove local-buffer write throughput"
        )
    if expected_control_directory is not None and control.get("directory") != str(
        Path(expected_control_directory).expanduser().resolve()
    ):
        raise StorageQualificationError("storage attestation control root mismatch")
    if expected_local_directories is not None:
        expected_paths = tuple(
            str(Path(item).expanduser().resolve()) for item in expected_local_directories
        )
        if tuple(local_buffer.get("directories", ())) != expected_paths:
            raise StorageQualificationError("storage attestation local roots mismatch")
    if expected_object_prefix is not None and gcs.get("prefix") != (
        expected_object_prefix.rstrip("/")
    ):
        raise StorageQualificationError("storage attestation object prefix mismatch")
    return claimed


def qualify_orbax_checkpoint_storage(
    checkpoint_root: str | os.PathLike[str],
    run_id: str,
    *,
    coordination: CoordinationHooks | None = None,
) -> OrbaxCheckpointQualification:
    """Collectively prove Orbax can save, finalize, and restore a tiny tree.

    The probe deliberately uses Orbax rather than the raw object-store client:
    checkpoint payloads are directories of TensorStore/OCDBT objects and can
    fail because of IAM or backend behavior even when ordinary blob uploads
    succeed.  The finalized probe is retained as immutable qualification
    evidence and its exact URI is sealed into the storage attestation.
    """

    hooks = coordination or local_coordination_hooks()
    process_index, process_count = _validate_process_identity(hooks)
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError("run_id contains unsafe checkpoint path characters")

    root_value = str(checkpoint_root)
    if root_value.startswith("gs://"):
        payload_uri = join_gcs_uri(
            root_value.rstrip("/"), "qualification", f"orbax-{run_id}"
        )
    elif "://" in root_value:
        raise ValueError("checkpoint qualification root must be POSIX or gs://")
    else:
        payload_uri = str(
            Path(root_value).expanduser().resolve()
            / "qualification"
            / f"orbax-{run_id}"
        )

    try:
        import jax
        import numpy as np
        import orbax.checkpoint as ocp
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    except ImportError as exc:  # pragma: no cover - optional production path
        raise StorageQualificationError(
            "Orbax checkpoint qualification requires the TPU runtime extra"
        ) from exc

    global_devices = tuple(jax.devices())
    global_device_count = len(global_devices)
    if (
        int(jax.process_index()) != process_index
        or int(jax.process_count()) != process_count
    ):
        raise StorageQualificationError(
            "checkpoint coordination hooks disagree with the JAX topology"
        )
    if global_device_count <= 0:
        raise StorageQualificationError("Orbax qualification found no JAX devices")
    if global_device_count % process_count:
        raise StorageQualificationError(
            "global JAX devices do not divide evenly across processes"
        )
    local_device_count = global_device_count // process_count
    mesh = Mesh(
        np.asarray(global_devices).reshape(process_count, local_device_count),
        ("data", "feature"),
    )
    sharded = NamedSharding(mesh, P("data", "feature"))
    replicated = NamedSharding(mesh, P())
    expected = {
        "device_index": jax.device_put(
            np.arange(global_device_count, dtype=np.int32).reshape(
                process_count, local_device_count
            ),
            sharded,
        ),
        "sentinel": jax.device_put(
            np.asarray([0x13579BDF, 0x2468ACE], dtype=np.int32), replicated
        ),
    }
    expected_local_digest = _local_jax_tree_sha256(expected)
    metadata = {
        "schema_version": ORBAX_QUALIFICATION_SCHEMA,
        "run_id": run_id,
        "process_count": process_count,
        "global_device_count": global_device_count,
        "payload_uri": payload_uri,
    }
    def options(operation: str) -> Any:
        return ocp.options.MultiprocessingOptions(
            primary_host=0,
            active_processes=set(range(process_count)),
            barrier_sync_key_prefix=f"moe-sae-orbax-{run_id}-{operation}",
        )

    save_error = ""
    save_seconds = 0.0
    finalized = False
    hooks.barrier("orbax-checkpoint-save-ready")
    save_started = time.perf_counter()
    checkpointer = None
    try:
        checkpointer = ocp.StandardCheckpointer(
            multiprocessing_options=options("save")
        )
        checkpointer.save(
            payload_uri,
            expected,
            force=False,
            custom_metadata=metadata,
        )
        checkpointer.wait_until_finished()
        finalized = _orbax_path_finalized(ocp, payload_uri)
        if not finalized:
            raise StorageQualificationError(
                "Orbax checkpoint probe did not finalize"
            )
    except Exception as exc:  # keep every process on the collective protocol
        save_error = _format_error(exc)
    finally:
        if checkpointer is not None:
            try:
                checkpointer.close()
            except Exception as exc:
                if not save_error:
                    save_error = _format_error(exc)
    hooks.barrier("orbax-checkpoint-save-complete")
    save_seconds = time.perf_counter() - save_started
    save_records = _gather_records(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "seconds": save_seconds,
            "finalized": finalized,
            "expected_local_sha256": expected_local_digest,
            "error": save_error,
        },
    )

    restored_local_digest = ""
    restore_error = ""
    restore_seconds = 0.0
    metadata_round_trip = False
    if not _record_errors(save_records) and all(
        bool(record["finalized"]) for record in save_records
    ):
        hooks.barrier("orbax-checkpoint-restore-ready")
        restore_started = time.perf_counter()
        checkpointer = None
        try:
            checkpointer = ocp.StandardCheckpointer(
                multiprocessing_options=options("restore")
            )
            observed_metadata = checkpointer.metadata(payload_uri)
            metadata_round_trip = observed_metadata.custom_metadata == metadata
            if not metadata_round_trip:
                raise StorageQualificationError(
                    "Orbax checkpoint probe metadata changed"
                )
            restored = checkpointer.restore(payload_uri, target=expected)
            restored_local_digest = _local_jax_tree_sha256(restored)
            if restored_local_digest != expected_local_digest:
                raise StorageQualificationError(
                    "Orbax checkpoint probe restored different local shards"
                )
            if not _orbax_path_finalized(ocp, payload_uri):
                raise StorageQualificationError(
                    "Orbax checkpoint probe lost its finalized marker"
                )
        except Exception as exc:
            restore_error = _format_error(exc)
        finally:
            if checkpointer is not None:
                try:
                    checkpointer.close()
                except Exception as exc:
                    if not restore_error:
                        restore_error = _format_error(exc)
        hooks.barrier("orbax-checkpoint-restore-complete")
        restore_seconds = time.perf_counter() - restore_started
    restore_records = _gather_records(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "seconds": restore_seconds,
            "restored_local_sha256": restored_local_digest,
            "metadata_round_trip": metadata_round_trip,
            "error": restore_error,
        },
    )

    failures: list[str] = []
    save_errors = _record_errors(save_records)
    restore_errors = _record_errors(restore_records)
    if save_errors:
        failures.append("Orbax checkpoint save/finalize failed: " + "; ".join(save_errors))
    elif not all(bool(record["finalized"]) for record in save_records):
        failures.append("Orbax checkpoint payload was not finalized on every process")
    if restore_errors:
        failures.append("Orbax checkpoint restore failed: " + "; ".join(restore_errors))
    elif not all(bool(record["metadata_round_trip"]) for record in restore_records):
        failures.append("Orbax checkpoint metadata did not round-trip on every process")
    expected_digests = tuple(
        str(record["expected_local_sha256"]) for record in save_records
    )
    restored_digests = tuple(
        str(record["restored_local_sha256"]) for record in restore_records
    )
    if not failures and restored_digests != expected_digests:
        failures.append("Orbax checkpoint local-shard digests did not round-trip")
    save_timings = tuple(float(record["seconds"]) for record in save_records)
    restore_timings = tuple(
        float(record["seconds"]) for record in restore_records
    )
    return OrbaxCheckpointQualification(
        payload_uri=payload_uri,
        process_count=process_count,
        global_device_count=global_device_count,
        expected_local_sha256_by_process=expected_digests,
        restored_local_sha256_by_process=restored_digests,
        save_seconds_by_process=save_timings,
        restore_seconds_by_process=restore_timings,
        max_save_seconds=max(save_timings),
        max_restore_seconds=max(restore_timings),
        finalized=all(bool(record["finalized"]) for record in save_records),
        metadata_round_trip=all(
            bool(record["metadata_round_trip"]) for record in restore_records
        ),
        failures=tuple(failures),
    )


def _local_jax_tree_sha256(tree: Mapping[str, Any]) -> str:
    """Hash exact addressable shard indexes and bytes for one process."""

    digest = hashlib.sha256()
    for name in sorted(tree):
        value = tree[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        shards = sorted(value.addressable_shards, key=lambda shard: int(shard.device.id))
        for shard in shards:
            digest.update(str(int(shard.device.id)).encode("ascii"))
            digest.update(repr(shard.index).encode("ascii"))
            local = shard.data
            try:
                import jax
                import numpy as np

                host = np.asarray(jax.device_get(local))
            except (ImportError, TypeError, ValueError) as exc:
                raise StorageQualificationError(
                    "cannot inspect an addressable Orbax probe shard"
                ) from exc
            digest.update(host.dtype.str.encode("ascii"))
            digest.update(np.ascontiguousarray(host).tobytes())
    return digest.hexdigest()


def _orbax_path_finalized(ocp: Any, path: str) -> bool:
    finalized = getattr(ocp.utils, "is_path_finalized", None)
    if finalized is None:
        finalized = getattr(ocp.utils, "is_checkpoint_finalized", None)
    if finalized is None:
        raise StorageQualificationError(
            "installed Orbax does not expose checkpoint finalization checks"
        )
    return bool(finalized(path))


def _qualify_local_write(
    local_root: Path,
    run_id: str,
    size: int,
    block_bytes: int,
    hooks: CoordinationHooks,
    process_index: int,
    process_count: int,
) -> tuple[TransferMeasurement, bool]:
    """Measure the synchronized host-local write path and remove its probe.

    Block creation is outside the timed region. Repeating one incompressible
    block avoids measuring a pseudorandom generator while still defeating
    transparent filesystem compression. The probe is removed before the two
    simultaneous GCS source/download files are allocated.
    """

    path = local_root / (
        f".moe-sae-local-write-qualification-{run_id}-{process_index:05d}.probe"
    )
    prepare_error = ""
    block = b""
    try:
        block = hashlib.shake_256(
            f"{run_id}:{process_index}:local-write".encode("ascii")
        ).digest(block_bytes)
    except Exception as exc:
        prepare_error = _format_error(exc)

    write_error = prepare_error
    written_bytes = 0
    hooks.barrier("local-write-ready")
    write_started = time.perf_counter()
    if not write_error:
        try:
            remaining = size
            with path.open("xb", buffering=0) as handle:
                while remaining:
                    portion = min(block_bytes, remaining)
                    _write_all(handle.fileno(), block[:portion])
                    remaining -= portion
                os.fsync(handle.fileno())
            written_bytes = size
        except Exception as exc:
            write_error = _format_error(exc)
    hooks.barrier("local-write-complete")
    write_seconds = time.perf_counter() - write_started
    write_records = _gather_records(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "bytes": written_bytes,
            "seconds": write_seconds,
            "error": write_error,
        },
    )
    measurement = _transfer_measurement(write_records)

    cleanup_error = ""
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:
        cleanup_error = _format_error(exc)
    cleanup_records = _gather_records(
        hooks,
        process_count,
        {"process_index": process_index, "error": cleanup_error},
    )
    hooks.barrier("local-write-cleanup-complete")
    cleanup_errors = _record_errors(cleanup_records)
    return measurement, not cleanup_errors


def _qualify_object_plane(
    local_root: Path,
    object_prefix: str,
    store: ImmutableObjectStore,
    run_id: str,
    config: StoragePlanesQualificationConfig,
    hooks: CoordinationHooks,
    process_index: int,
    process_count: int,
) -> tuple[
    tuple[Mapping[str, Any], ...],
    TransferMeasurement,
    TransferMeasurement | None,
    bool,
    tuple[str, ...],
]:
    source = local_root / (
        f".moe-sae-gcs-qualification-{run_id}-{process_index:05d}.source"
    )
    destination = local_root / (
        f".moe-sae-gcs-qualification-{run_id}-{process_index:05d}.download"
    )
    generation_error = ""
    digest = ""
    source_fingerprint: SourceFingerprint | None = None
    try:
        source_fingerprint = _write_incompressible_probe(
            source,
            size=config.gcs_probe_bytes_per_process,
            block_bytes=config.gcs_probe_block_bytes,
            seed=f"{run_id}:{process_index}".encode("ascii"),
        )
        digest = source_fingerprint.sha256
    except Exception as exc:
        # Every host must reach the same gather even if one local producer or
        # fingerprint operation fails; the coordinated result then fails shut.
        generation_error = _format_error(exc)
    generated = _gather_records(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "sha256": digest,
            "error": generation_error,
        },
    )

    identity: ObjectIdentity | None = None
    upload_error = generation_error
    upload_bytes = 0
    hooks.barrier("gcs-upload-ready")
    upload_started = time.perf_counter()
    if not upload_error:
        uri = join_gcs_uri(
            object_prefix,
            "qualification",
            run_id,
            f"host-{process_index:05d}",
            f"{digest}.bin",
        )
        try:
            identity = store.put_if_absent(
                source,
                uri,
                sha256=digest,
                source_fingerprint=source_fingerprint,
            )
            store.verify(identity)
            if identity.size_bytes != config.gcs_probe_bytes_per_process:
                raise StorageQualificationError(
                    "published object size differs from the probe size"
                )
            upload_bytes = identity.size_bytes
        except Exception as exc:  # keep every process on the collective protocol
            upload_error = _format_error(exc)
    hooks.barrier("gcs-upload-complete")
    upload_seconds = time.perf_counter() - upload_started
    upload_records = _gather_records(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "bytes": upload_bytes,
            "seconds": upload_seconds,
            "identity": identity.as_dict() if identity is not None else None,
            "error": upload_error,
        },
    )
    upload = _transfer_measurement(upload_records)
    identities: tuple[Mapping[str, Any], ...] = tuple(
        item["identity"] for item in upload_records if item["identity"] is not None
    )

    download: TransferMeasurement | None = None
    if len(identities) == process_count and not upload.errors:
        target_mapping = upload_records[(process_index + 1) % process_count]["identity"]
        download_error = ""
        download_bytes = 0
        hooks.barrier("gcs-download-ready")
        download_started = time.perf_counter()
        try:
            target_identity = object_identity_from_mapping(target_mapping)
            store.download(target_identity, destination)
            actual_digest, actual_size = sha256_file(destination)
            if (
                actual_digest != target_identity.sha256
                or actual_size != target_identity.size_bytes
            ):
                raise StorageQualificationError(
                    "downloaded probe failed end-to-end SHA-256/size verification"
                )
            download_bytes = actual_size
        except Exception as exc:
            download_error = _format_error(exc)
        hooks.barrier("gcs-download-complete")
        download_seconds = time.perf_counter() - download_started
        download_records = _gather_records(
            hooks,
            process_count,
            {
                "process_index": process_index,
                "bytes": download_bytes,
                "seconds": download_seconds,
                "error": download_error,
            },
        )
        download = _transfer_measurement(download_records)

    failures: list[str] = []
    generation_errors = _record_errors(generated)
    if generation_errors:
        failures.append(
            "local incompressible probe generation failed: "
            + "; ".join(generation_errors)
        )
    if upload.errors:
        failures.append("immutable GCS upload failed: " + "; ".join(upload.errors))
    elif upload.bytes_per_second < config.minimum_aggregate_upload_bytes_per_second:
        failures.append(
            "aggregate GCS upload throughput below threshold: "
            f"actual={upload.bytes_per_second:.3f} "
            f"required={config.minimum_aggregate_upload_bytes_per_second:.3f} B/s"
        )
    if download is None:
        failures.append("generation-pinned GCS download was not attempted")
    elif download.errors:
        failures.append(
            "generation-pinned GCS download failed: " + "; ".join(download.errors)
        )
    elif download.bytes_per_second < config.minimum_aggregate_download_bytes_per_second:
        failures.append(
            "aggregate GCS download throughput below threshold: "
            f"actual={download.bytes_per_second:.3f} "
            f"required={config.minimum_aggregate_download_bytes_per_second:.3f} B/s"
        )

    cleanup_error = ""
    if config.cleanup:
        cleanup_failures: list[str] = []
        for label, path in (("source", source), ("destination", destination)):
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:
                cleanup_failures.append(f"{label}: {_format_error(exc)}")
        if identity is not None:
            try:
                store.delete(identity)
            except Exception as exc:
                cleanup_failures.append(f"object: {_format_error(exc)}")
        cleanup_error = "; ".join(cleanup_failures)
    cleanup_records = _gather_records(
        hooks,
        process_count,
        {"process_index": process_index, "error": cleanup_error},
    )
    hooks.barrier("gcs-probe-cleanup-complete")
    cleanup_errors = _record_errors(cleanup_records)
    cleanup_complete = not cleanup_errors
    if cleanup_errors:
        failures.append("GCS probe cleanup failed: " + "; ".join(cleanup_errors))
    return identities, upload, download, cleanup_complete, tuple(failures)


def _validate_process_identity(hooks: CoordinationHooks) -> tuple[int, int]:
    index = hooks.process_index()
    count = hooks.process_count()
    if isinstance(index, bool) or not isinstance(index, int):
        raise StorageQualificationError("process_index must return an integer")
    if isinstance(count, bool) or not isinstance(count, int):
        raise StorageQualificationError("process_count must return an integer")
    if count <= 0 or not 0 <= index < count:
        raise StorageQualificationError(f"invalid process identity {index}/{count}")
    return index, count


def _prepare_directories(
    control_root: Path,
    local_root: Path,
    hooks: CoordinationHooks,
    process_index: int,
    process_count: int,
) -> None:
    error = ""
    try:
        control_root.mkdir(parents=True, exist_ok=True)
        local_root.mkdir(parents=True, exist_ok=True)
        if not control_root.is_dir() or not local_root.is_dir():
            raise NotADirectoryError("storage root is not a directory")
    except OSError as exc:
        error = _format_error(exc)
    records = _gather_records(
        hooks,
        process_count,
        {"process_index": process_index, "error": error},
    )
    errors = _record_errors(records)
    if errors:
        raise StorageQualificationError(
            "cannot prepare storage qualification directories ("
            + "; ".join(errors)
            + ")"
        )


def _coordinated_run_id(
    hooks: CoordinationHooks, process_index: int, process_count: int
) -> str:
    proposal = uuid.uuid4().hex if process_index == 0 else ""
    proposals = tuple(hooks.allgather(proposal))
    if len(proposals) != process_count:
        raise StorageQualificationError("allgather returned an unexpected process count")
    run_id = proposals[0]
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise StorageQualificationError("process 0 supplied an invalid run id")
    return run_id


def _prepare_shared_work_directory(
    work: Path,
    hooks: CoordinationHooks,
    process_index: int,
    process_count: int,
) -> None:
    error = ""
    if process_index == 0:
        try:
            work.mkdir(mode=0o700)
        except OSError as exc:
            error = _format_error(exc)
    created = _gather_records(
        hooks,
        process_count,
        {"process_index": process_index, "error": error},
    )
    errors = _record_errors(created)
    if errors:
        raise StorageQualificationError(
            "cannot create shared qualification work directory ("
            + "; ".join(errors)
            + ")"
        )
    hooks.barrier("work-directory-created")
    visible_error = "" if work.is_dir() else "work directory is not visible"
    visible = _gather_records(
        hooks,
        process_count,
        {"process_index": process_index, "error": visible_error},
    )
    errors = _record_errors(visible)
    if errors:
        raise StorageQualificationError(
            "qualification work directory is not shared ("
            + "; ".join(errors)
            + ")"
        )


def _gather_records(
    hooks: CoordinationHooks,
    process_count: int,
    record: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    gathered = tuple(hooks.allgather(dict(record)))
    if len(gathered) != process_count:
        raise StorageQualificationError("allgather returned an unexpected process count")
    by_index: dict[int, Mapping[str, Any]] = {}
    for item in gathered:
        if not isinstance(item, Mapping):
            raise StorageQualificationError("allgather returned a non-record value")
        index = item.get("process_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise StorageQualificationError("allgather record has no process index")
        if index in by_index or not 0 <= index < process_count:
            raise StorageQualificationError("allgather record process indexes are invalid")
        by_index[index] = item
    if len(by_index) != process_count:
        raise StorageQualificationError("allgather omitted a process record")
    return tuple(by_index[index] for index in range(process_count))


def _record_errors(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        f"process {item['process_index']}: {item['error']}"
        for item in records
        if item.get("error")
    )


def _measure_free_space(
    root: Path,
    hooks: CoordinationHooks,
    process_index: int,
    process_count: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    value = -1
    error = ""
    try:
        value = int(shutil.disk_usage(root).free)
    except OSError as exc:
        error = _format_error(exc)
    records = _gather_records(
        hooks,
        process_count,
        {"process_index": process_index, "free_bytes": value, "error": error},
    )
    return tuple(int(item["free_bytes"]) for item in records), _record_errors(records)


def _filesystem_identities(
    root: Path,
    hooks: CoordinationHooks,
    process_index: int,
    process_count: int,
) -> tuple[Mapping[str, Any], ...]:
    error = ""
    identity: dict[str, Any] = {"path": str(root), "device": -1, "inode": -1}
    try:
        stat = root.stat()
        identity.update(device=int(stat.st_dev), inode=int(stat.st_ino))
    except OSError as exc:
        error = _format_error(exc)
    records = _gather_records(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "identity": identity,
            "error": error,
        },
    )
    errors = _record_errors(records)
    if errors:
        raise StorageQualificationError(
            "cannot identify storage filesystems (" + "; ".join(errors) + ")"
        )
    return tuple(dict(item["identity"]) for item in records)


def _qualify_local_buffer_isolation(
    root: Path,
    run_id: str,
    hooks: CoordinationHooks,
    process_index: int,
    process_count: int,
) -> tuple[bool, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Prove equal local-buffer paths resolve to process-private storage.

    Every controller atomically publishes a different payload under the same
    basename.  A genuinely host-local mount returns that controller's payload;
    a shared POSIX/NFS mount converges on one writer's payload and fails.
    """

    path = root / f".moe-sae-local-isolation-{run_id}.probe"
    temporary = root / (
        f".{path.name}.process-{process_index:05d}-{uuid.uuid4().hex}.tmp"
    )
    payload = _canonical_json(
        {
            "schema_version": "gemma4-local-buffer-isolation-probe-v1",
            "run_id": run_id,
            "process_index": process_index,
            "nonce": uuid.uuid4().hex,
        }
    )
    expected_digest = hashlib.sha256(payload).hexdigest()
    publish_error = ""
    try:
        _write_bytes_durable(temporary, payload, exclusive=True)
        os.replace(temporary, path)
        _fsync_directory(root)
    except OSError as exc:
        publish_error = _format_error(exc)
    finally:
        temporary.unlink(missing_ok=True)
    published = _gather_records(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "expected_sha256": expected_digest,
            "error": publish_error,
        },
    )
    hooks.barrier("local-isolation-published")

    observed_digest = ""
    observe_error = ""
    try:
        observed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        observe_error = _format_error(exc)
    observed = _gather_records(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "observed_sha256": observed_digest,
            "error": observe_error,
        },
    )
    hooks.barrier("local-isolation-observed")

    cleanup_error = ""
    try:
        path.unlink(missing_ok=True)
        _fsync_directory(root)
    except OSError as exc:
        cleanup_error = _format_error(exc)
    cleaned = _gather_records(
        hooks,
        process_count,
        {"process_index": process_index, "error": cleanup_error},
    )
    hooks.barrier("local-isolation-cleaned")

    expected = tuple(str(item["expected_sha256"]) for item in published)
    actual = tuple(str(item["observed_sha256"]) for item in observed)
    errors = tuple(
        f"process {item['process_index']}: {item['error']}"
        for records in (published, observed, cleaned)
        for item in records
        if item["error"]
    )
    isolated = (
        not errors
        and expected == actual
        and len(set(expected)) == process_count
    )
    return isolated, expected, actual, errors


def _qualify_flock(
    work: Path,
    hooks: CoordinationHooks,
    process_index: int,
    process_count: int,
) -> tuple[bool, tuple[str, ...]]:
    lock_path = work / "flock.lock"
    holder = None
    hold_error = ""
    if process_index == 0:
        try:
            holder = lock_path.open("a+b", buffering=0)
            fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            hold_error = _format_error(exc)
    held = _gather_records(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "held": holder is not None if process_index == 0 else True,
            "error": hold_error,
        },
    )
    holder_ready = bool(held[0]["held"]) and not held[0]["error"]
    hooks.barrier("flock-held")

    blocked = process_index == 0
    probe_error = ""
    if process_index != 0 and holder_ready:
        probe = None
        try:
            probe = lock_path.open("a+b", buffering=0)
            try:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    blocked = True
                else:
                    raise
            else:
                fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            probe_error = _format_error(exc)
        finally:
            if probe is not None:
                probe.close()
    probes = _gather_records(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "blocked": blocked,
            "error": probe_error,
        },
    )
    hooks.barrier("flock-probed")
    if holder is not None:
        try:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        finally:
            holder.close()
    hooks.barrier("flock-released")

    contender = None
    acquired = False
    acquire_error = ""
    try:
        contender = lock_path.open("a+b", buffering=0)
        try:
            fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
        else:
            acquired = True
    except OSError as exc:
        acquire_error = _format_error(exc)
    winners = _gather_records(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "acquired": acquired,
            "error": acquire_error,
        },
    )
    hooks.barrier("flock-winner-observed")
    if contender is not None:
        try:
            if acquired:
                fcntl.flock(contender.fileno(), fcntl.LOCK_UN)
        finally:
            contender.close()
    errors = tuple(
        f"process {item['process_index']}: {item['error']}"
        for records in (held, probes, winners)
        for item in records
        if item["error"]
    )
    correct = (
        holder_ready
        and all(bool(item["blocked"]) for item in probes)
        and sum(bool(item["acquired"]) for item in winners) == 1
        and not errors
    )
    return correct, errors


def _qualify_atomic_rename(
    work: Path,
    run_id: str,
    hooks: CoordinationHooks,
    process_index: int,
    process_count: int,
) -> tuple[bool, tuple[str, ...]]:
    final_path = work / "rename-visible.bin"
    temporary_path = work / "rename-visible.tmp"
    old_payload = hashlib.shake_256(f"{run_id}:old".encode()).digest(4096)
    new_payload = hashlib.shake_256(f"{run_id}:new".encode()).digest(65536)
    prepare_error = ""
    if process_index == 0:
        try:
            _write_bytes_durable(final_path, old_payload, exclusive=True)
            _write_bytes_durable(temporary_path, new_payload, exclusive=True)
            _fsync_directory(work)
        except OSError as exc:
            prepare_error = _format_error(exc)
    prepared = _gather_records(
        hooks,
        process_count,
        {"process_index": process_index, "error": prepare_error},
    )
    hooks.barrier("rename-prepared")
    before_error = ""
    try:
        before_visible = (
            final_path.read_bytes() == old_payload
            and temporary_path.read_bytes() == new_payload
        )
    except OSError as exc:
        before_visible = False
        before_error = _format_error(exc)
    before = _gather_records(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "visible": before_visible,
            "error": before_error,
        },
    )
    replace_error = ""
    if process_index == 0:
        try:
            os.replace(temporary_path, final_path)
            _fsync_directory(work)
        except OSError as exc:
            replace_error = _format_error(exc)
    replaced = _gather_records(
        hooks,
        process_count,
        {"process_index": process_index, "error": replace_error},
    )
    hooks.barrier("rename-published")
    after_error = ""
    try:
        after_visible = (
            final_path.read_bytes() == new_payload and not temporary_path.exists()
        )
    except OSError as exc:
        after_visible = False
        after_error = _format_error(exc)
    after = _gather_records(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "visible": after_visible,
            "error": after_error,
        },
    )
    errors = tuple(
        f"process {item['process_index']}: {item['error']}"
        for records in (prepared, before, replaced, after)
        for item in records
        if item["error"]
    )
    correct = (
        all(not item["error"] for item in prepared)
        and all(bool(item["visible"]) for item in before)
        and all(not item["error"] for item in replaced)
        and all(bool(item["visible"]) for item in after)
        and not errors
    )
    return correct, errors


def _qualify_disjoint_files(
    work: Path,
    run_id: str,
    hooks: CoordinationHooks,
    process_index: int,
    process_count: int,
) -> tuple[bool, tuple[str, ...]]:
    path = work / f"host-{process_index:05d}.sentinel"
    payload = f"{run_id}:{process_index}\n".encode("ascii")
    publish_error = ""
    try:
        _write_bytes_durable(path, payload, exclusive=True)
    except OSError as exc:
        publish_error = _format_error(exc)
    published = _gather_records(
        hooks,
        process_count,
        {"process_index": process_index, "error": publish_error},
    )
    hooks.barrier("disjoint-files-published")
    visibility_error = ""
    try:
        found = tuple(sorted(work.glob("host-*.sentinel")))
        expected = tuple(
            f"host-{index:05d}.sentinel" for index in range(process_count)
        )
        visible = (
            tuple(item.name for item in found) == expected
            and all(
                item.read_bytes() == f"{run_id}:{index}\n".encode("ascii")
                for index, item in enumerate(found)
            )
        )
    except OSError as exc:
        visible = False
        visibility_error = _format_error(exc)
    observed = _gather_records(
        hooks,
        process_count,
        {
            "process_index": process_index,
            "visible": visible,
            "error": visibility_error,
        },
    )
    errors = tuple(
        f"process {item['process_index']}: {item['error']}"
        for records in (published, observed)
        for item in records
        if item["error"]
    )
    correct = (
        all(not item["error"] for item in published)
        and all(bool(item["visible"]) for item in observed)
        and not errors
    )
    return correct, errors


def _write_incompressible_probe(
    path: Path, *, size: int, block_bytes: int, seed: bytes
) -> SourceFingerprint:
    """Write deterministic pseudorandom blocks without a repeating pattern."""

    try:
        import google_crc32c
    except ImportError as error:  # pragma: no cover - required core dependency
        raise StorageQualificationError(
            "whole-object probe verification requires google-crc32c"
        ) from error
    digest = hashlib.sha256()
    crc32c = google_crc32c.Checksum()
    remaining = size
    counter = 0
    with path.open("xb", buffering=0) as handle:
        while remaining:
            portion = min(block_bytes, remaining)
            block = hashlib.shake_256(
                seed + counter.to_bytes(8, "little")
            ).digest(portion)
            _write_all(handle.fileno(), block)
            digest.update(block)
            crc32c.update(block)
            remaining -= portion
            counter += 1
        os.fsync(handle.fileno())
    return _fingerprint_from_components(
        path,
        sha256=digest.hexdigest(),
        crc32c=base64.b64encode(crc32c.digest()).decode("ascii"),
        size_bytes=size,
    )


def _transfer_measurement(
    records: Sequence[Mapping[str, Any]],
) -> TransferMeasurement:
    byte_counts = tuple(int(item["bytes"]) for item in records)
    seconds = tuple(float(item["seconds"]) for item in records)
    elapsed = max(seconds, default=0.0)
    total = sum(byte_counts)
    return TransferMeasurement(
        total_bytes=total,
        elapsed_seconds=elapsed,
        bytes_per_second=total / elapsed if elapsed > 0 else 0.0,
        per_process_bytes=byte_counts,
        per_process_seconds=seconds,
        errors=_record_errors(records),
    )


def _write_all(file_descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(file_descriptor, view)
        if written <= 0:  # pragma: no cover - defensive POSIX boundary
            raise OSError("write made no progress")
        view = view[written:]


def _write_bytes_durable(path: Path, payload: bytes, *, exclusive: bool) -> None:
    with path.open("xb" if exclusive else "wb", buffering=0) as handle:
        _write_all(handle.fileno(), payload)
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        # Checkpoint pointers require a durable directory entry after atomic
        # replace.  Qualification must exercise that exact contract rather
        # than accepting a filesystem that reports directory fsync unsupported.
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StorageQualificationError(
            "storage attestation is not canonical-JSON compatible"
        ) from exc


def _format_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _error_suffix(errors: Sequence[str]) -> str:
    return ": " + "; ".join(errors) if errors else ""


__all__ = [
    "ACTIVATION_CAPTURE_SCOPE",
    "ControlPlaneQualification",
    "CoordinationHooks",
    "LocalBufferQualification",
    "MINIMUM_ACTIVATION_GCS_DOWNLOAD_BYTES_PER_SECOND",
    "MINIMUM_ACTIVATION_LOCAL_FREE_BYTES",
    "MINIMUM_ACTIVATION_LOCAL_WRITE_BYTES_PER_SECOND",
    "ObjectPlaneQualification",
    "OrbaxCheckpointQualification",
    "ORBAX_QUALIFICATION_SCHEMA",
    "STORAGE_ATTESTATION_SCHEMA",
    "StoragePlanesQualificationConfig",
    "StoragePlanesQualificationResult",
    "StorageQualificationError",
    "TransferMeasurement",
    "build_storage_attestation",
    "jax_coordination_hooks",
    "local_coordination_hooks",
    "qualify_storage_planes",
    "qualify_orbax_checkpoint_storage",
    "storage_attestation_digest",
    "validate_storage_attestation",
]
