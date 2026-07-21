"""Single command-line surface for the Gemma 4 reconstruction experiment."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import dataclasses
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np

from moe_sae.artifacts import (
    ArtifactDataset,
    StagedDatasetProvenance,
    load_manifest,
    verify_staged_dataset_manifest,
)
from moe_sae.config import Gemma4Config, load_config
from moe_sae.metrics import (
    MetricsSink,
    SeedEvaluation,
    build_efficiency_report,
    render_headline,
    write_report_json,
)
from moe_sae.model import (
    CaptureBatch,
    assert_hf_snapshot_unchanged,
    build_checkpoint_conversion_command,
    checkpoint_inventory_sha256,
    checkpoint_conversion_spec,
    checkpoint_validation_command,
    expected_hf_repository_inventory_sha256,
    expected_hf_snapshot_inventory_sha256,
    materialize_hf_snapshot,
    maxtext_checkout_environment,
    verify_maxtext_checkout,
)
from moe_sae.object_store import (
    GCSObjectStore,
    ShardCache,
    join_gcs_uri,
    parse_gcs_uri,
)
from moe_sae.runtime import (
    BenchmarkObservation,
    HARVEST_MESHES,
    PER_DEVICE_SEQUENCE_BATCHES,
    PreflightError,
    atomic_write_json,
    choose_harvest_benchmark,
    environment_fingerprint,
    require_v4_32_runtime,
    run_preflight,
)
from moe_sae.storage import (
    ACTIVATION_CAPTURE_SCOPE,
    MINIMUM_ACTIVATION_GCS_DOWNLOAD_BYTES_PER_SECOND,
    MINIMUM_ACTIVATION_LOCAL_FREE_BYTES,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "gemma4_26b_a4b.yaml"
_BENCHMARK_RENDEZVOUS_SECONDS = 300.0
_BENCHMARK_KILL_GRACE_SECONDS = 15.0
_ACTIVATION_CHILD_KILL_GRACE_SECONDS = 30.0


def _sha256_argument(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("must be a lowercase 64-character SHA-256")
    return value


def _config(args: argparse.Namespace) -> Gemma4Config:
    return load_config(args.config)


def _artifact_root(config: Gemma4Config, override: str | None) -> Path:
    path = Path(override or config.artifacts_root)
    return path if path.is_absolute() else REPO_ROOT / path


def _require_production_artifact_root(root: Path) -> None:
    """Keep all new production state below the one v1 artifact namespace."""

    resolved = root.resolve()
    required = (REPO_ROOT / "artifacts" / "gemma4_v1").resolve()
    try:
        resolved.relative_to(required)
    except ValueError as error:
        raise RuntimeError(
            f"production artifact root must be below {required}, got {resolved}"
        ) from error


def _capture_root(root: Path, split: str) -> Path:
    return root / "captures" / split


def _conversion_attestation_path(root: Path) -> Path:
    return root / "checkpoint" / "conversion.json"


def _conversion_attestation_digest(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("attestation_sha256", None)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _storage_attestation_path(root: Path) -> Path:
    return root / "preflight" / "storage.json"


def _validate_gcs_storage_roots(
    bulk_gcs_root: str, checkpoint_root: str
) -> tuple[str, str]:
    """Return normalized roots after proving checkpoint storage is contained."""

    from moe_sae.object_store import join_gcs_uri

    bulk = str(bulk_gcs_root).rstrip("/")
    checkpoint = str(checkpoint_root).rstrip("/")
    try:
        join_gcs_uri(bulk, "identity-probe")
        join_gcs_uri(checkpoint, "identity-probe")
    except ValueError as error:
        raise RuntimeError("storage roots must be safe gs:// prefixes") from error
    bulk_bucket, _, bulk_prefix = bulk[5:].partition("/")
    checkpoint_bucket, _, checkpoint_prefix = checkpoint[5:].partition("/")
    if (
        checkpoint_bucket != bulk_bucket
        or not checkpoint_prefix
        or (
            bulk_prefix
            and not checkpoint_prefix.startswith(f"{bulk_prefix}/")
        )
    ):
        raise RuntimeError(
            "checkpoint payload root must be below the qualified bulk GCS root"
        )
    return bulk, checkpoint


def _validate_orbax_qualification(
    value: Any,
    *,
    checkpoint_root: str,
    run_id: str,
    process_count: int,
    global_device_count: int,
) -> None:
    """Validate the checkpoint-backend proof sealed into an attestation."""

    from moe_sae.object_store import join_gcs_uri
    from moe_sae.storage import ORBAX_QUALIFICATION_SCHEMA

    if not isinstance(value, Mapping):
        raise RuntimeError("storage attestation lacks Orbax qualification")
    expected_uri = join_gcs_uri(
        checkpoint_root, "qualification", f"orbax-{run_id}"
    )
    expected_digests = value.get("expected_local_sha256_by_process")
    restored_digests = value.get("restored_local_sha256_by_process")
    save_seconds = value.get("save_seconds_by_process")
    restore_seconds = value.get("restore_seconds_by_process")
    if (
        value.get("schema_version") != ORBAX_QUALIFICATION_SCHEMA
        or value.get("payload_uri") != expected_uri
        or value.get("process_count") != process_count
        or value.get("global_device_count") != global_device_count
        or value.get("failures") not in ([], ())
        or value.get("finalized") is not True
        or value.get("metadata_round_trip") is not True
        or not isinstance(expected_digests, (list, tuple))
        or len(expected_digests) != process_count
        or not isinstance(restored_digests, (list, tuple))
        or len(restored_digests) != process_count
        or tuple(expected_digests) != tuple(restored_digests)
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
            for digest in expected_digests
        )
        or not isinstance(save_seconds, (list, tuple))
        or len(save_seconds) != process_count
        or not isinstance(restore_seconds, (list, tuple))
        or len(restore_seconds) != process_count
    ):
        raise RuntimeError("storage attestation Orbax qualification is invalid")
    for label, timings, maximum in (
        ("save", save_seconds, value.get("max_save_seconds")),
        ("restore", restore_seconds, value.get("max_restore_seconds")),
    ):
        try:
            numbers = tuple(float(item) for item in timings)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"storage attestation Orbax {label} timings are invalid"
            ) from error
        if (
            any(not math.isfinite(item) or item <= 0.0 for item in numbers)
            or not isinstance(maximum, (int, float))
            or isinstance(maximum, bool)
            or not math.isclose(float(maximum), max(numbers), rel_tol=1e-12)
        ):
            raise RuntimeError(
                f"storage attestation Orbax {label} timings are invalid"
            )


def _validate_storage_attestation_payload(
    payload: Mapping[str, Any],
    *,
    root: Path,
    config: Gemma4Config,
    environment_sha256: str,
    process_count: int,
    global_device_count: int,
) -> str:
    from moe_sae.storage import (
        StoragePlanesQualificationConfig,
        validate_storage_attestation,
    )

    qualification = payload.get("qualification")
    topology = payload.get("topology")
    metadata = payload.get("metadata")
    expected_contract = dataclasses.asdict(
        StoragePlanesQualificationConfig(
            minimum_control_free_bytes=config.min_control_free_bytes,
            minimum_local_buffer_free_bytes=MINIMUM_ACTIVATION_LOCAL_FREE_BYTES,
            gcs_probe_bytes_per_process=config.gcs_probe_bytes_per_process,
            gcs_probe_block_bytes=config.gcs_probe_block_bytes,
            minimum_aggregate_upload_bytes_per_second=(
                config.min_gcs_upload_bytes_per_second
            ),
            minimum_aggregate_download_bytes_per_second=(
                MINIMUM_ACTIVATION_GCS_DOWNLOAD_BYTES_PER_SECOND
            ),
            cleanup=True,
        )
    )
    if (
        payload.get("contract") != expected_contract
        or not isinstance(qualification, Mapping)
        or qualification.get("process_count") != process_count
        or not isinstance(topology, Mapping)
        or topology.get("process_count") != process_count
        or topology.get("global_device_count") != global_device_count
        or not isinstance(metadata, Mapping)
    ):
        raise RuntimeError(
            "storage attestation does not prove the current config/environment/topology"
        )
    control = qualification.get("control")
    local = qualification.get("local_buffer")
    gcs = qualification.get("gcs")
    bulk_gcs_root = metadata.get("bulk_gcs_root")
    checkpoint_root = metadata.get("sae_checkpoint_root")
    if (
        payload.get("scope") != ACTIVATION_CAPTURE_SCOPE
        or metadata.get("sae_cache_qualified") is not False
        or metadata.get("configured_sae_capture_cache_max_bytes")
        != config.capture_cache_max_bytes
        or metadata.get("configured_sae_min_gcs_download_bytes_per_second")
        != config.min_gcs_download_bytes_per_second
        or not isinstance(control, Mapping)
        or not isinstance(local, Mapping)
        or not isinstance(gcs, Mapping)
        or not isinstance(bulk_gcs_root, str)
        or not isinstance(checkpoint_root, str)
    ):
        raise RuntimeError("storage attestation deployment layout is incomplete")
    bulk_gcs_root, checkpoint_root = _validate_gcs_storage_roots(
        bulk_gcs_root, checkpoint_root
    )
    if gcs.get("prefix") != bulk_gcs_root:
        raise RuntimeError("storage attestation GCS root changed")
    run_id = qualification.get("run_id")
    if not isinstance(run_id, str):
        raise RuntimeError("storage attestation qualification has no run ID")
    _validate_orbax_qualification(
        metadata.get("orbax_checkpoint_qualification"),
        checkpoint_root=checkpoint_root,
        run_id=run_id,
        process_count=process_count,
        global_device_count=global_device_count,
    )
    local_directories = local.get("directories")
    if (
        not isinstance(local_directories, (list, tuple))
        or len(local_directories) != process_count
        or any(not isinstance(path, str) or not path for path in local_directories)
    ):
        raise RuntimeError("storage attestation local-buffer layout is incompatible")
    try:
        return validate_storage_attestation(
            payload,
            expected_config_sha256=config.config_hash,
            expected_environment_sha256=environment_sha256,
            expected_control_directory=root,
            expected_local_directories=local_directories,
            expected_object_prefix=str(gcs.get("prefix", "")),
            expected_scope=ACTIVATION_CAPTURE_SCOPE,
        )
    except RuntimeError as error:
        raise RuntimeError("storage attestation validation failed") from error


def _read_storage_attestation_payload(root: Path) -> dict[str, Any]:
    """Read the storage record without initializing a JAX backend."""

    path = _storage_attestation_path(root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"missing/invalid storage attestation: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("storage attestation must be an object")
    return payload


def _load_storage_attestation_payload(
    root: Path,
    config: Gemma4Config,
    *,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and bind the sealed three-plane deployment record."""

    payload = _read_storage_attestation_payload(root)
    current_environment = dict(environment or _coordinated_environment())
    _, process_count = _process_context()
    try:
        import jax

        global_device_count = int(jax.device_count())
    except ImportError:
        global_device_count = 1
    _validate_storage_attestation_payload(
        payload,
        root=root,
        config=config,
        environment_sha256=str(current_environment.get("sha256", "")),
        process_count=process_count,
        global_device_count=global_device_count,
    )
    return payload


def _load_storage_attestation(
    root: Path,
    config: Gemma4Config,
    *,
    environment: Mapping[str, Any] | None = None,
) -> str:
    return str(
        _load_storage_attestation_payload(root, config, environment=environment)[
            "attestation_sha256"
        ]
    )


def _storage_layout(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve this controller's attested local root and shared GCS roots."""

    process_index, process_count = _process_context()
    qualification = payload["qualification"]
    directories = qualification["local_buffer"]["directories"]
    if len(directories) != process_count:
        raise RuntimeError("storage layout no longer matches the process topology")
    metadata = payload["metadata"]
    return {
        "local_buffer_root": Path(directories[process_index]),
        "capture_cache_root": Path(directories[process_index]) / "capture-cache",
        "bulk_gcs_root": str(metadata["bulk_gcs_root"]).rstrip("/"),
        "sae_checkpoint_root": str(metadata["sae_checkpoint_root"]).rstrip("/"),
        "attestation_sha256": str(payload["attestation_sha256"]),
    }


def _require_sae_cache_qualification(payload: Mapping[str, Any]) -> None:
    """Reject activation-only storage evidence on SAE consumer paths."""

    metadata = payload.get("metadata")
    if (
        payload.get("scope") != "sae-training"
        or not isinstance(metadata, Mapping)
        or metadata.get("sae_cache_qualified") is not True
    ):
        raise RuntimeError(
            "storage attestation is scoped to activation-capture only; "
            "SAE training/evaluation requires a separate cache qualification "
            "or streaming-cache redesign"
        )


def _load_conversion_attestation(
    root: Path, config: Gemma4Config, checkpoint_sha256: str
) -> str:
    """Require a self-hashed successful HF/MaxText conversion parity record."""

    path = _conversion_attestation_path(root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"missing/invalid checkpoint conversion attestation: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint conversion attestation must be an object")
    claimed = str(payload.get("attestation_sha256", ""))
    actual = _conversion_attestation_digest(payload)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", claimed)
        or claimed != actual
        or payload.get("schema_version") != "gemma4-checkpoint-conversion-v2"
        or payload.get("parity_passed") is not True
        or payload.get("config_sha256") != config.config_hash
        or payload.get("model_id") != config.model_id
        or payload.get("model_revision") != config.model_revision
        or payload.get("hf_source_mode") != "immutable-local-snapshot"
        or payload.get("hf_repository_inventory_sha256")
        != expected_hf_repository_inventory_sha256()
        or payload.get("hf_snapshot_inventory_sha256")
        != expected_hf_snapshot_inventory_sha256()
        or payload.get("maxtext_commit") != config.maxtext_commit
        or payload.get("maxtext_checkout_clean") is not True
        or payload.get("maxtext_import_mode") != "verified-checkout-source"
        or payload.get("checkpoint_sha256") != checkpoint_sha256
        or not math.isclose(
            float(payload.get("max_kl_div_threshold", math.nan)),
            0.03,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise RuntimeError(
            "checkpoint conversion attestation does not prove the required parity run"
        )
    return claimed


def _sealed_dataset(
    root: Path,
    *,
    cache_root: Path | None = None,
    cache_max_bytes: int | None = None,
) -> ArtifactDataset:
    """Open a sealed capture and hash each shard as the consumer reads it.

    Publication is not treated as a permanent integrity waiver: training and
    evaluation detect post-publication corruption.  Sufficient statistics do
    not need to be recomputed: they are serialized into the hashed shard, and
    the identical ledger values are bound by a self-hashed upload receipt that
    the coordinator validates before sealing.
    """

    manifest = load_manifest(root)
    remote = any("object" in entry for entry in manifest["shards"])
    cache = None
    if remote:
        if (cache_root is None) != (cache_max_bytes is None):
            raise RuntimeError("capture cache root and bound must be supplied together")
        if cache_root is not None and cache_max_bytes is not None:
            cache = ShardCache(
                cache_root,
                GCSObjectStore(),
                max_bytes=cache_max_bytes,
            )
    return ArtifactDataset(
        root,
        verify_hashes=True,
        verify_statistics=False,
        cache=cache,
    )


def _prefetch_process_partition(
    dataset: ArtifactDataset, total_tokens: int, *, label: str
) -> tuple[dict[str, int], float]:
    """Warm this controller's immutable capture partition outside SAE timing."""

    process_index, process_count = _process_context()
    start = (total_tokens * process_index) // process_count
    end = (total_tokens * (process_index + 1)) // process_count
    _global_barrier(f"gemma4-{label}-prefetch-start")
    started = time.perf_counter()
    result = dataset.prefetch_token_range(start, end)
    elapsed = _global_max_seconds(time.perf_counter() - started)
    _global_barrier(f"gemma4-{label}-prefetch-complete")
    return result, elapsed


def _run_root(root: Path, condition: str, seed: int) -> Path:
    return root / "training" / condition / f"seed-{seed}"


def _run_identity_expected(
    config: Gemma4Config,
    *,
    capture_manifest_sha256: str,
    capture_checkpoint_sha256: str,
    environment_sha256: str,
    condition: str,
    seed: int,
    storage_attestation_sha256: str | None = None,
    checkpoint_payload_root: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "gemma4-sae-run-identity-v1",
        "config_sha256": config.config_hash,
        "capture_manifest_sha256": capture_manifest_sha256,
        "capture_checkpoint_sha256": capture_checkpoint_sha256,
        "environment_sha256": environment_sha256,
        "condition": condition,
        "seed": seed,
        "storage_attestation_sha256": storage_attestation_sha256,
        "checkpoint_payload_root": checkpoint_payload_root,
    }


def _checkpoint_identity_expected(
    config: Gemma4Config,
    *,
    capture_manifest_sha256: str,
    capture_checkpoint_sha256: str,
    environment_sha256: str,
    condition: str,
    seed: int,
    run_instance_id: str,
    storage_attestation_sha256: str | None,
    checkpoint_payload_root: str | None,
) -> dict[str, Any]:
    """Build the exact immutable identity shared by save and restore."""

    identity = _run_identity_expected(
        config,
        capture_manifest_sha256=capture_manifest_sha256,
        capture_checkpoint_sha256=capture_checkpoint_sha256,
        environment_sha256=environment_sha256,
        condition=condition,
        seed=seed,
        storage_attestation_sha256=storage_attestation_sha256,
        checkpoint_payload_root=checkpoint_payload_root,
    )
    identity.pop("schema_version")
    identity["run_instance_id"] = run_instance_id
    return identity


def _load_run_identity(path: Path, expected: dict[str, Any]) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read immutable run identity: {path}") from error
    if any(payload.get(name) != value for name, value in expected.items()):
        raise RuntimeError(f"run identity belongs to another experiment: {path}")
    run_instance_id = str(payload.get("run_instance_id", ""))
    if not re.fullmatch(r"[0-9a-f]{32}", run_instance_id):
        raise RuntimeError(f"run identity has an invalid instance ID: {path}")
    return run_instance_id


def _evaluation_path(root: Path, condition: str, seed: int) -> Path:
    return root / "evaluations" / f"{condition}-seed-{seed}.json"


def _sae_config(config: Gemma4Config, condition: str) -> Any:
    from moe_sae.sae import SAEConfig

    return SAEConfig(
        condition=condition,
        hidden_dim=config.hidden_size,
        num_experts=config.num_routed_experts,
        top_routed=config.experts_per_token,
        bank_width=config.bank_width,
        k=config.topk_features,
        parameter_dtype=config.dtype,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_eps=config.adam_epsilon,
        grad_clip=config.grad_clip,
        peak_lr=config.learning_rate,
        min_lr=config.min_learning_rate,
        warmup_steps=config.warmup_updates,
        total_steps=config.planned_updates,
    )


def _training_config(
    config: Gemma4Config, condition: str, seed: int, *, a_capacity: int | None
) -> Any:
    from moe_sae.pipeline import TrainingConfig

    return TrainingConfig(
        sae=_sae_config(config, condition),
        seed=seed,
        total_tokens=config.train_tokens,
        effective_batch_tokens=config.effective_batch_tokens,
        microbatch_tokens=config.microbatch_tokens,
        checkpoint_every_tokens=config.checkpoint_every_tokens,
        health_gate_tokens=config.health_gate_tokens,
        live_feature_fraction=config.min_live_feature_fraction,
        a_capacity=a_capacity,
    )


def _balanced_padding_ids(rows: int, config: Gemma4Config) -> np.ndarray:
    token = np.arange(rows, dtype=np.int64)[:, None]
    slot = np.arange(config.experts_per_token, dtype=np.int64)[None, :]
    return (
        (token * config.experts_per_token + slot) % config.num_routed_experts
    ).astype(np.int32)


def _reference_evaluation_batches(dataset: ArtifactDataset, microbatch_tokens: int):
    """Slice physical capture shards to the same logical eval microbatches."""

    if microbatch_tokens <= 0:
        raise ValueError("evaluation microbatch size must be positive")
    for batch in dataset.iter_batches():
        rows = int(np.asarray(batch.activation).shape[0])
        for start in range(0, rows, microbatch_tokens):
            end = min(rows, start + microbatch_tokens)
            yield CaptureBatch(*(np.asarray(field)[start:end] for field in batch))


def _precompute_a_capacity(
    dataset: ArtifactDataset,
    config: Gemma4Config,
    *,
    expected_tokens: int | None = None,
) -> int:
    """Scan the fixed token order once and return a rounded 10% hard capacity."""

    micro = config.microbatch_tokens
    buffered = np.empty((0, config.experts_per_token), dtype=np.int32)
    worst = 0

    def observe(ids: np.ndarray) -> None:
        nonlocal worst
        counts = np.bincount(ids.reshape(-1), minlength=config.num_routed_experts)
        worst = max(worst, int(counts.max(initial=0)))

    consumed = 0
    for batch in dataset.iter_batches():
        ids = np.asarray(batch.expert_ids, dtype=np.int32)
        consumed += ids.shape[0]
        if buffered.size:
            ids = np.concatenate((buffered, ids), axis=0)
        whole = (ids.shape[0] // micro) * micro
        for offset in range(0, whole, micro):
            observe(ids[offset : offset + micro])
        buffered = ids[whole:]
    expected = config.train_tokens if expected_tokens is None else expected_tokens
    if consumed != expected:
        raise RuntimeError(
            f"capacity capture contains {consumed} tokens; expected {expected}"
        )
    if buffered.shape[0]:
        padded = np.concatenate(
            (buffered, _balanced_padding_ids(micro - buffered.shape[0], config)), axis=0
        )
        observe(padded)
    capacity = math.ceil((worst * 1.10) / 8) * 8
    if capacity <= 0:
        raise RuntimeError("could not derive an A routing capacity")
    return capacity


def _capacity_cache_identity(
    dataset: ArtifactDataset,
    config: Gemma4Config,
    process_count: int,
    seed: int,
    scope: str,
    total_tokens: int,
) -> dict[str, Any]:
    return {
        "config_sha256": config.config_hash,
        "capture_manifest_sha256": dataset.manifest_sha256,
        "repository_content_sha256": _coordinated_environment().get(
            "repository_content_sha256"
        ),
        "process_count": process_count,
        "seed": seed,
        "scope": scope,
        "total_tokens": total_tokens,
        "microbatch_tokens": config.microbatch_tokens,
        "margin": 0.10,
        "round_to": 8,
    }


def _production_a_capacity(
    root: Path,
    dataset: ArtifactDataset,
    config: Gemma4Config,
    training_config: Any,
    *,
    scope: str = "train",
    total_tokens: int | None = None,
) -> tuple[int, float]:
    import jax
    from jax.experimental import multihost_utils
    from moe_sae.distributed import precompute_distributed_a_capacity

    token_count = config.train_tokens if total_tokens is None else total_tokens
    identity = _capacity_cache_identity(
        dataset,
        config,
        jax.process_count(),
        training_config.seed,
        scope,
        token_count,
    )
    path = root / "routing" / f"a_capacity-{scope}-seed-{training_config.seed}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("identity") != identity:
            raise RuntimeError(
                "cached A capacity does not match this capture/config/mesh"
            )
        capacity = int(cached["capacity"])
        if capacity <= 0 or capacity % 8:
            raise RuntimeError("cached A capacity is invalid")
        wall_seconds = float(cached.get("precompute_wall_seconds", -1.0))
        if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
            raise RuntimeError("cached A capacity lacks measured precompute wall time")
        return capacity, wall_seconds

    started = time.perf_counter()
    capacity = precompute_distributed_a_capacity(
        dataset, training_config, total_tokens=token_count
    )
    wall_seconds = _global_max_seconds(time.perf_counter() - started)
    if jax.process_index() == 0:
        atomic_write_json(
            path,
            {
                "identity": identity,
                "capacity": capacity,
                "precompute_wall_seconds": wall_seconds,
            },
        )
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices("gemma4-a-capacity-published")
    if not path.exists():
        raise RuntimeError("A capacity cache is not visible on every process")
    return capacity, wall_seconds


def _require_exact_capture(dataset: ArtifactDataset, expected: int, name: str) -> None:
    if dataset.total_tokens != expected:
        raise RuntimeError(
            f"{name} capture has {dataset.total_tokens:,} tokens; require exactly {expected:,}"
        )


def _require_capture_identity(
    dataset: ArtifactDataset,
    config: Gemma4Config,
    split: str,
    *,
    checkpoint_sha256: str | None = None,
    conversion_attestation_sha256: str | None = None,
    dataset_fingerprint: str | None = None,
    environment_sha256: str | None = None,
    storage_attestation_sha256: str | None = None,
    object_prefix: str | None = None,
) -> None:
    """Reject a valid v1 capture produced by a different experiment identity."""

    metadata = dataset.manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("capture manifest lacks metadata")
    expected = {
        "config_sha256": config.config_hash,
        "maxtext_commit": config.maxtext_commit,
        "split": split,
        "target_layer_one_indexed": config.target_layer_one_indexed,
        "capture_site": "routed_moe.post_feedforward_layernorm_2",
    }
    if checkpoint_sha256 is not None:
        expected["checkpoint_sha256"] = checkpoint_sha256
    if conversion_attestation_sha256 is not None:
        expected["conversion_attestation_sha256"] = (
            conversion_attestation_sha256
        )
    if dataset_fingerprint is not None:
        expected["dataset_fingerprint"] = dataset_fingerprint
    if environment_sha256 is not None:
        expected["environment_sha256"] = environment_sha256
    if storage_attestation_sha256 is not None:
        expected["storage_attestation_sha256"] = storage_attestation_sha256
    if object_prefix is not None:
        expected["object_prefix"] = object_prefix
    changed = {
        name: (metadata.get(name), value)
        for name, value in expected.items()
        if metadata.get(name) != value
    }
    if changed:
        detail = ", ".join(
            f"{name}={actual!r} (expected {wanted!r})"
            for name, (actual, wanted) in changed.items()
        )
        raise RuntimeError(f"capture identity mismatch: {detail}")


def _verified_dataset_provenance(
    args: argparse.Namespace,
    config: Gemma4Config,
    *,
    split: str,
    grain_files: str,
    manifest_location: str,
) -> StagedDatasetProvenance:
    """Verify and cache one immutable staged split for this CLI process."""

    expected_split = config.train_split if split == "train" else config.eval_split
    expected_tokens = config.train_tokens if split == "train" else config.eval_tokens
    verify_arrayrecord_count = not (
        bool(getattr(args, "local", False))
        or bool(getattr(args, "reuse_verified_dataset", False))
    )
    key = (split, grain_files, manifest_location, verify_arrayrecord_count)
    cache = getattr(args, "_verified_dataset_provenance", None)
    if cache is None:
        cache = {}
        args._verified_dataset_provenance = cache
    if key not in cache:
        value = verify_staged_dataset_manifest(
            manifest_location,
            grain_files,
            expected_split=expected_split,
            expected_tokens=expected_tokens,
            tokenizer_id=config.tokenizer_path,
            tokenizer_revision=config.model_revision,
            sequence_length=config.sequence_length,
            verify_file_hashes=True,
            verify_arrayrecord_count=verify_arrayrecord_count,
        )
        _coordinated_sha256(
            value.fingerprint_sha256, purpose=f"{split} staged dataset fingerprint"
        )
        cache[key] = value
    return cache[key]


def _metric_record(metric: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(metric):
        values = dataclasses.asdict(metric)
    elif hasattr(metric, "_asdict"):
        values = dict(metric._asdict())
    else:
        raise TypeError(f"unsupported metric type {type(metric).__name__}")
    record: dict[str, Any] = {}
    for name, value in values.items():
        array = np.asarray(value)
        record[name] = array.item() if array.ndim == 0 else array.tolist()
    return record


def _process_context() -> tuple[int, int]:
    """Return the active JAX controller identity without requiring JAX locally."""

    try:
        import jax
    except ImportError:
        return 0, 1
    return int(jax.process_index()), int(jax.process_count())


def _coordinated_sha256(digest: str, *, purpose: str) -> str:
    """Require the same lowercase digest on every JAX controller."""

    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError(f"{purpose} is missing its SHA-256")
    _, process_count = _process_context()
    if process_count == 1:
        return digest
    from jax.experimental import multihost_utils

    payload = np.frombuffer(bytes.fromhex(digest), dtype=np.uint8).copy()
    gathered = np.asarray(
        multihost_utils.process_allgather(payload, tiled=False)
    ).reshape(process_count, -1)
    if gathered.shape != (process_count, 32) or not np.all(gathered == gathered[0]):
        raise RuntimeError(f"TPU controllers have different {purpose} values")
    return digest


def _coordinated_environment() -> dict[str, Any]:
    """Require identical source/package/device provenance on every controller."""

    value = environment_fingerprint(REPO_ROOT)
    value["sha256"] = _coordinated_sha256(
        str(value.get("sha256", "")), purpose="environment fingerprint"
    )
    return value


def _verify_checkpoint_inventory(args: argparse.Namespace) -> str:
    """Bind a claimed checkpoint digest to the actual immutable inventory."""

    if getattr(args, "_checkpoint_inventory_verified", False):
        return str(args.checkpoint_sha256)
    started = time.perf_counter()
    process_index, process_count = _process_context()
    del process_index
    local_error: Exception | None = None
    try:
        local_digest = checkpoint_inventory_sha256(str(args.checkpoint))
    except Exception as error:
        local_error = error
        local_digest = "0" * 64
    if process_count == 1:
        if local_error is not None:
            raise RuntimeError(
                "could not inventory the staged checkpoint"
            ) from local_error
        actual = local_digest
    else:
        from jax.experimental import multihost_utils

        local = np.concatenate(
            (
                np.asarray([local_error is None], dtype=np.uint8),
                np.frombuffer(bytes.fromhex(local_digest), dtype=np.uint8),
            )
        )
        rows = np.asarray(
            multihost_utils.process_allgather(local, tiled=False), dtype=np.uint8
        ).reshape(process_count, 33)
        if not np.all(rows[:, 0] == 1):
            raise RuntimeError(
                "checkpoint inventory is unavailable on at least one TPU controller"
            )
        if not np.all(rows[:, 1:] == rows[0, 1:]):
            raise RuntimeError(
                "TPU controllers observe different checkpoint inventories"
            )
        actual = rows[0, 1:].tobytes().hex()
    if actual != args.checkpoint_sha256:
        raise RuntimeError(
            "--checkpoint-sha256 does not match the staged checkpoint inventory"
        )
    args._checkpoint_inventory_verified = True
    args._checkpoint_verification_seconds = _global_max_seconds(
        time.perf_counter() - started
    )
    return actual


def _new_run_instance_id() -> str:
    """Create one opaque run identifier shared by every controller."""

    process_index, process_count = _process_context()
    if process_count == 1:
        return os.urandom(16).hex()
    from jax.experimental import multihost_utils

    local = np.frombuffer(
        os.urandom(16) if process_index == 0 else bytes(16), dtype=np.uint8
    ).copy()
    shared = np.asarray(
        multihost_utils.broadcast_one_to_all(local, is_source=process_index == 0),
        dtype=np.uint8,
    )
    if shared.shape != (16,):
        raise RuntimeError("run instance broadcast returned an invalid identifier")
    return shared.tobytes().hex()


def _global_barrier(name: str) -> None:
    process_index, process_count = _process_context()
    del process_index
    if process_count > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices(name)


def _global_max_seconds(value: float) -> float:
    """Use the slowest controller's wall time for globally synchronized work."""

    _, process_count = _process_context()
    if process_count == 1:
        return float(value)
    from jax.experimental import multihost_utils

    gathered = multihost_utils.process_allgather(np.asarray(value, np.float64))
    return float(np.max(np.asarray(gathered)))


def _aggregate_benchmark_observation(
    observation: BenchmarkObservation,
) -> BenchmarkObservation:
    """Merge controller telemetry conservatively for one global SPMD candidate."""

    _, process_count = _process_context()
    if process_count == 1:
        return observation
    from jax.experimental import multihost_utils

    local = np.asarray(
        [
            observation.synchronized_steps,
            observation.elapsed_seconds,
            observation.useful_flops,
            observation.output_tokens,
            observation.persisted_tokens,
            observation.input_wait_fraction,
            observation.recompilations,
            observation.dropped_routed_tokens,
            observation.max_writer_backlog,
            float(observation.instrumentation_equal),
            float(observation.failure is not None),
        ],
        dtype=np.float64,
    )
    rows = np.asarray(multihost_utils.process_allgather(local)).reshape((-1, 11))
    elapsed = float(np.max(rows[:, 1]))
    if np.any(rows[:, 10] == 1.0):
        return BenchmarkObservation(
            mesh=observation.mesh,
            per_device_sequence_batch=observation.per_device_sequence_batch,
            synchronized_steps=0,
            elapsed_seconds=max(elapsed, 1e-9),
            useful_flops=0.0,
            output_tokens=0,
            persisted_tokens=0,
            recompilations=1,
            instrumentation_equal=False,
            failure="candidate failed on at least one TPU controller",
        )
    for column, name in (
        (0, "steps"),
        (2, "useful FLOPs"),
        (3, "output tokens"),
        (4, "persisted tokens"),
    ):
        if not np.all(rows[:, column] == rows[0, column]):
            raise RuntimeError(f"benchmark controllers disagree on {name}")
    return BenchmarkObservation(
        mesh=observation.mesh,
        per_device_sequence_batch=observation.per_device_sequence_batch,
        synchronized_steps=int(rows[0, 0]),
        elapsed_seconds=elapsed,
        useful_flops=float(rows[0, 2]),
        output_tokens=int(rows[0, 3]),
        persisted_tokens=int(rows[0, 4]),
        input_wait_seconds=float(np.max(rows[:, 5])) * elapsed,
        recompilations=int(np.max(rows[:, 6])),
        dropped_routed_tokens=int(np.max(rows[:, 7])),
        max_writer_backlog=int(np.max(rows[:, 8])),
        instrumentation_equal=bool(np.all(rows[:, 9] == 1.0)),
        failure=None,
    )


def _coordinator_write_json(path: Path, payload: Any, *, barrier: str) -> bool:
    """Write one shared JSON file on process zero, then make it globally visible."""

    process_index, _ = _process_context()
    if process_index == 0:
        atomic_write_json(path, payload)
    _global_barrier(barrier)
    if not path.exists():
        raise RuntimeError(f"coordinator output is not visible after barrier: {path}")
    return process_index == 0


def _release_accelerator_state(label: str) -> None:
    """Synchronously release model/executable caches between large stages."""

    _global_barrier(f"gemma4-release-{label}-before")
    try:
        import jax

        jax.clear_caches()
    except (ImportError, AttributeError):
        pass
    gc.collect()
    _global_barrier(f"gemma4-release-{label}-after")


def _command_stage_c4(args: argparse.Namespace) -> int:
    """Stage one bounded, immutable AllenAI C4 split on a CPU coordinator."""

    from moe_sae.data import publish_staged_dataset, stage_c4_split

    config = _config(args)
    split = args.split
    expected = config.train_tokens if split == config.train_split else config.eval_tokens
    target = expected if args.target_tokens is None else int(args.target_tokens)
    if target != expected and not args.local_smoke:
        raise RuntimeError("production C4 staging uses the configured exact token target")
    root = _artifact_root(config, args.artifact_root)
    output = (
        Path(args.output_directory).expanduser()
        if args.output_directory
        else root / "staging" / "c4" / split
    )
    result = stage_c4_split(
        split=split,
        target_scientific_tokens=target,
        output_directory=output,
        source_cache=args.source_cache_dir,
        tokenizer_id=config.tokenizer_path,
        tokenizer_revision=config.model_revision,
        records_per_shard=args.records_per_shard,
        sequence_length=config.sequence_length,
    )
    if args.publish_gcs:
        result = publish_staged_dataset(result.manifest_location, args.publish_gcs)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


def _command_qualify_storage(args: argparse.Namespace) -> int:
    """Qualify the POSIX control plane, local buffers, and GCS data plane."""

    from moe_sae.object_store import GCSObjectStore, LocalObjectStore
    from moe_sae.storage import (
        ACTIVATION_CAPTURE_SCOPE,
        MINIMUM_ACTIVATION_GCS_DOWNLOAD_BYTES_PER_SECOND,
        MINIMUM_ACTIVATION_LOCAL_FREE_BYTES,
        StoragePlanesQualificationConfig,
        build_storage_attestation,
        jax_coordination_hooks,
        local_coordination_hooks,
        qualify_orbax_checkpoint_storage,
        qualify_storage_planes,
    )

    config = _config(args)
    root = _artifact_root(config, args.artifact_root).resolve()
    local_buffer_root = Path(args.local_buffer_root).expanduser().resolve()
    bulk_gcs_root, sae_checkpoint_root = _validate_gcs_storage_roots(
        str(args.bulk_gcs_root), str(args.sae_checkpoint_root)
    )
    if args.local:
        hooks = local_coordination_hooks()
        qualification_config = StoragePlanesQualificationConfig(
            minimum_control_free_bytes=int(args.minimum_control_free_bytes or 1),
            minimum_local_buffer_free_bytes=int(args.minimum_local_free_bytes or 1),
            gcs_probe_bytes_per_process=int(args.probe_bytes or 8 * 1024**2),
            gcs_probe_block_bytes=int(args.block_bytes or 1024**2),
            minimum_aggregate_upload_bytes_per_second=float(
                args.minimum_bytes_per_second or 0.0
            ),
            minimum_aggregate_download_bytes_per_second=float(
                args.minimum_bytes_per_second or 0.0
            ),
        )
        object_store = LocalObjectStore(
            args.local_object_store_root or root / "preflight" / "object-store-smoke"
        )
        global_device_count = 1
    else:
        config.validate_production()
        _require_production_artifact_root(root)
        require_v4_32_runtime()
        if any(
            value is not None
            for value in (
                args.minimum_control_free_bytes,
                args.minimum_local_free_bytes,
                args.probe_bytes,
                args.block_bytes,
                args.minimum_bytes_per_second,
                args.local_object_store_root,
            )
        ):
            raise RuntimeError("production storage qualification thresholds cannot be overridden")
        hooks = jax_coordination_hooks()
        qualification_config = StoragePlanesQualificationConfig(
            minimum_control_free_bytes=config.min_control_free_bytes,
            minimum_local_buffer_free_bytes=MINIMUM_ACTIVATION_LOCAL_FREE_BYTES,
            gcs_probe_bytes_per_process=config.gcs_probe_bytes_per_process,
            gcs_probe_block_bytes=config.gcs_probe_block_bytes,
            minimum_aggregate_upload_bytes_per_second=(
                config.min_gcs_upload_bytes_per_second
            ),
            minimum_aggregate_download_bytes_per_second=(
                MINIMUM_ACTIVATION_GCS_DOWNLOAD_BYTES_PER_SECOND
            ),
        )
        object_store = GCSObjectStore()
        import jax

        global_device_count = int(jax.device_count())
    result = qualify_storage_planes(
        root,
        local_buffer_root,
        bulk_gcs_root,
        object_store,
        config=qualification_config,
        coordination=hooks,
    )
    result.require_pass()
    if args.local:
        print(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))
        return 0

    orbax_qualification = qualify_orbax_checkpoint_storage(
        sae_checkpoint_root,
        result.run_id,
        coordination=hooks,
    )
    orbax_qualification.require_pass()
    environment = _coordinated_environment()
    payload = build_storage_attestation(
        result,
        qualification_config,
        config_sha256=config.config_hash,
        environment_sha256=str(environment["sha256"]),
        global_device_count=global_device_count,
        scope=ACTIVATION_CAPTURE_SCOPE,
        metadata={
            "bulk_gcs_root": bulk_gcs_root,
            "sae_checkpoint_root": sae_checkpoint_root,
            "sae_cache_qualified": False,
            "configured_sae_capture_cache_max_bytes": config.capture_cache_max_bytes,
            "configured_sae_min_gcs_download_bytes_per_second": (
                config.min_gcs_download_bytes_per_second
            ),
            "orbax_checkpoint_qualification": dataclasses.asdict(
                orbax_qualification
            ),
        },
    )
    coordinator = _coordinator_write_json(
        _storage_attestation_path(root),
        payload,
        barrier="gemma4-storage-qualification-attested",
    )
    if coordinator:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _command_preflight(args: argparse.Namespace) -> int:
    config = _config(args)
    root = _artifact_root(config, args.artifact_root)
    checks = run_preflight(
        repo_root=REPO_ROOT,
        artifact_root=root,
        checkpoint_path=args.checkpoint,
        grain_train_files=args.grain_train,
        grain_eval_files=args.grain_eval,
        require_tpu=not args.local,
    )
    storage_attestation_sha256 = (
        None if args.local else _load_storage_attestation(root, config)
    )
    train_provenance = _verified_dataset_provenance(
        args,
        config,
        split="train",
        grain_files=args.grain_train,
        manifest_location=args.dataset_train_manifest,
    )
    eval_provenance = _verified_dataset_provenance(
        args,
        config,
        split="eval",
        grain_files=args.grain_eval,
        manifest_location=args.dataset_eval_manifest,
    )
    payload = {
        check.name: {"passed": check.passed, "detail": check.detail} for check in checks
    }
    payload["dataset_train_manifest"] = {
        "passed": True,
        "detail": train_provenance.fingerprint_sha256,
    }
    payload["dataset_eval_manifest"] = {
        "passed": True,
        "detail": eval_provenance.fingerprint_sha256,
    }
    if storage_attestation_sha256 is not None:
        payload["storage_attestation"] = {
            "passed": True,
            "detail": storage_attestation_sha256,
        }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for check in checks:
            print(f"PASS {check.name}: {check.detail}")
        print(f"PASS dataset_train_manifest: {train_provenance.fingerprint_sha256}")
        print(f"PASS dataset_eval_manifest: {eval_provenance.fingerprint_sha256}")
        if storage_attestation_sha256 is not None:
            print(f"PASS storage_attestation: {storage_attestation_sha256}")
    return 0


def _command_convert_checkpoint(args: argparse.Namespace) -> int:
    config = _config(args)
    resume_validation = bool(getattr(args, "resume_validation", False))
    if resume_validation and args.dry_run:
        raise ValueError("--resume-validation cannot be combined with --dry-run")
    root = _artifact_root(config, args.artifact_root)
    _require_production_artifact_root(root)
    verify_maxtext_checkout(args.maxtext_root)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "action": "materialize-convert-validate",
                    "model_id": config.model_id,
                    "model_revision": config.model_revision,
                    "hf_repository_inventory_sha256": (
                        expected_hf_repository_inventory_sha256()
                    ),
                    "hf_snapshot_inventory_sha256": (
                        expected_hf_snapshot_inventory_sha256()
                    ),
                    "hf_cache_dir": str(Path(args.hf_cache_dir).expanduser()),
                    "maxtext_root": str(Path(args.maxtext_root).expanduser()),
                    "output": args.output,
                    "max_kl_div_threshold": 0.03,
                    "note": (
                        "dry-run does not download the roughly 52 GB snapshot; "
                        "the real run passes one commit-addressed local snapshot "
                        "to both conversion and parity checking"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    attestation_path = _conversion_attestation_path(root)
    if attestation_path.exists():
        raise RuntimeError(
            "a checkpoint conversion is already sealed at this artifact root; "
            "refusing to overwrite its attestation"
        )
    canonical_items = f"{str(args.output).rstrip('/')}/0/items"
    if resume_validation:
        if args.checkpoint_items is not None and (
            str(args.checkpoint_items).rstrip("/") != canonical_items
        ):
            raise ValueError(
                "--resume-validation requires the canonical checkpoint path "
                f"{canonical_items}, got {args.checkpoint_items}"
            )
        items = canonical_items
        # This both rejects a missing/empty prefix and captures every object's
        # generation/checksum (or every local byte) before validation.  The
        # MaxText restore below remains the authoritative completeness check.
        prevalidation_checkpoint_sha256 = checkpoint_inventory_sha256(items)
    else:
        _require_unused_conversion_output(args.output)
        items = args.checkpoint_items or canonical_items
        prevalidation_checkpoint_sha256 = None
    snapshot_started = time.perf_counter()
    snapshot = materialize_hf_snapshot(
        config.model_id,
        config.model_revision,
        cache_dir=args.hf_cache_dir,
        token=None if resume_validation else os.environ.get("HF_TOKEN"),
        local_files_only=resume_validation,
    )
    snapshot_materialization_seconds = time.perf_counter() - snapshot_started
    spec = checkpoint_conversion_spec(
        args.maxtext_root,
        args.output,
        hf_snapshot=snapshot,
        config=config,
    )
    conversion = (
        None if resume_validation else build_checkpoint_conversion_command(spec)
    )
    validation = checkpoint_validation_command(spec, items, max_kl_div=0.03)
    conversion_environment = maxtext_checkout_environment(args.maxtext_root)
    # The snapshot is already local. Neither subprocess needs a Hub credential,
    # so do not expose one to MaxText, Transformers, or imported test utilities.
    for credential_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN"):
        conversion_environment.pop(credential_name, None)
    conversion_seconds: float | None = None
    if not resume_validation:
        assert conversion is not None
        started = time.perf_counter()
        subprocess.run(
            conversion,
            cwd=spec.maxtext_root,
            check=True,
            env=conversion_environment,
        )
        conversion_seconds = time.perf_counter() - started
    validation_started = time.perf_counter()
    subprocess.run(
        validation,
        cwd=spec.maxtext_root,
        check=True,
        env=conversion_environment,
    )
    validation_seconds = time.perf_counter() - validation_started
    snapshot_revalidation_started = time.perf_counter()
    assert_hf_snapshot_unchanged(snapshot)
    snapshot_revalidation_seconds = (
        time.perf_counter() - snapshot_revalidation_started
    )
    # Fail if either official subprocess modified or generated importable code
    # in the source checkout before the parity attestation is sealed.
    verify_maxtext_checkout(args.maxtext_root)
    digest_started = time.perf_counter()
    checkpoint_sha256 = checkpoint_inventory_sha256(items)
    digest_seconds = time.perf_counter() - digest_started
    if (
        prevalidation_checkpoint_sha256 is not None
        and checkpoint_sha256 != prevalidation_checkpoint_sha256
    ):
        raise RuntimeError(
            "checkpoint inventory changed during resumed parity validation"
        )
    payload = {
        "schema_version": "gemma4-checkpoint-conversion-v2",
        "config_sha256": config.config_hash,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "hf_source_mode": "immutable-local-snapshot",
        "hf_repository_inventory_sha256": (
            snapshot.repository_inventory_sha256
        ),
        "hf_snapshot_inventory_sha256": snapshot.inventory_sha256,
        "maxtext_commit": config.maxtext_commit,
        "maxtext_checkout_clean": True,
        "maxtext_import_mode": "verified-checkout-source",
        "checkpoint": items,
        "checkpoint_sha256": checkpoint_sha256,
        "snapshot_materialization_seconds": snapshot_materialization_seconds,
        "snapshot_revalidation_seconds": snapshot_revalidation_seconds,
        "validation_seconds": validation_seconds,
        "digest_seconds": digest_seconds,
        "max_kl_div_threshold": 0.03,
        "parity_passed": True,
        "environment": _coordinated_environment(),
    }
    if resume_validation:
        payload["conversion_recovery"] = {
            "mode": "resume-validation",
            "converter_rerun": False,
            "conversion_timing_available": False,
            "prevalidation_checkpoint_sha256": prevalidation_checkpoint_sha256,
        }
    else:
        payload["conversion_seconds"] = conversion_seconds
    payload["attestation_sha256"] = _conversion_attestation_digest(payload)
    atomic_write_json(attestation_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _require_unused_conversion_output(output_directory: str) -> None:
    """Refuse to let a conversion attempt share an existing output prefix."""

    normalized = str(output_directory).rstrip("/")
    if not normalized:
        raise ValueError("conversion output must not be empty")
    if normalized.startswith("gs://"):
        bucket_name, prefix = parse_gcs_uri(normalized)
        try:
            store = GCSObjectStore()
            bucket = store.client.bucket(bucket_name)
            exact_exists = bool(bucket.blob(prefix).exists())
            descendants = iter(
                store.client.list_blobs(
                    bucket_name,
                    prefix=prefix + "/",
                    max_results=1,
                )
            )
            descendant_exists = next(descendants, None) is not None
        except Exception as error:
            raise RuntimeError(
                f"cannot prove conversion output prefix is unused: {normalized}"
            ) from error
        if exact_exists or descendant_exists:
            raise RuntimeError(
                "conversion output prefix already contains objects; choose a fresh "
                "attempt-specific --output and never reuse a partial conversion"
            )
        return

    path = Path(normalized).expanduser()
    if path.exists():
        raise RuntimeError(
            "conversion output path already exists; choose a fresh attempt-specific "
            "--output and never reuse a partial conversion"
        )


def _benchmark_output_directory(
    args: argparse.Namespace, artifact_root: Path
) -> Path:
    output = Path(args.output) if args.output else artifact_root / "benchmark"
    output = (
        (REPO_ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    )
    try:
        output.relative_to(artifact_root)
    except ValueError as error:
        raise ValueError(f"benchmark output must be below {artifact_root}") from error
    return output


def _wait_for_paths(
    paths: Sequence[Path], *, deadline: float, description: str
) -> None:
    missing = [path for path in paths if not path.is_file()]
    while missing and time.time() < deadline:
        time.sleep(0.05)
        missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"timed out waiting for {description}: "
            + ", ".join(str(path) for path in missing)
        )


def _terminate_process_group(process: Any) -> int:
    """Terminate one isolated worker and every descendant it may have spawned."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return int(process.wait(timeout=_BENCHMARK_KILL_GRACE_SECONDS))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return int(process.wait())


def _wait_for_worker(process: Any, *, deadline: float) -> tuple[int, bool]:
    try:
        return int(process.wait(timeout=max(0.0, deadline - time.time()))), False
    except subprocess.TimeoutExpired:
        return _terminate_process_group(process), True


def _inherited_activation_lease_fds() -> tuple[int, ...]:
    raw = os.environ.get("MOE_SAE_ACTIVATION_LEASE_FD")
    if raw is None:
        return ()
    try:
        descriptor = int(raw)
        os.fstat(descriptor)
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError("activation supervisor lease descriptor is invalid") from error
    if descriptor < 0:
        raise RuntimeError("activation supervisor lease descriptor is invalid")
    return (descriptor,)


@contextmanager
def _cleanup_worker_on_sigterm(process: Any):
    """Ensure an isolated benchmark worker cannot outlive its supervisor."""

    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum: int, _frame: Any) -> None:
        _terminate_process_group(process)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _benchmark_worker_command(
    mode: str,
    args: argparse.Namespace,
    *,
    artifact_root: Path,
    output: Path,
    extra: Sequence[str],
) -> list[str]:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    return [
        sys.executable,
        "-m",
        "moe_sae.benchmark_worker",
        mode,
        "--config",
        str(config_path),
        "--artifact-root",
        str(artifact_root),
        "--checkpoint",
        str(args.checkpoint),
        "--checkpoint-sha256",
        str(args.checkpoint_sha256),
        "--output",
        str(output),
        *extra,
    ]


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {description}: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return payload


def _benchmark_observation_from_dict(
    payload: Mapping[str, Any], *, mesh: Any, batch: int
) -> BenchmarkObservation:
    encoded_mesh = payload.get("mesh")
    if (
        not isinstance(encoded_mesh, Mapping)
        or encoded_mesh.get("name") != mesh.name
        or int(payload.get("per_device_sequence_batch", -1)) != batch
    ):
        raise RuntimeError("benchmark worker returned the wrong mesh/batch identity")
    return BenchmarkObservation(
        mesh=mesh,
        per_device_sequence_batch=batch,
        synchronized_steps=int(payload["synchronized_steps"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
        useful_flops=float(payload["useful_flops"]),
        output_tokens=int(payload["output_tokens"]),
        persisted_tokens=int(payload["persisted_tokens"]),
        input_wait_seconds=float(payload.get("input_wait_seconds", 0.0)),
        recompilations=int(payload.get("recompilations", 0)),
        dropped_routed_tokens=int(payload.get("dropped_routed_tokens", 0)),
        max_writer_backlog=int(payload.get("max_writer_backlog", 0)),
        instrumentation_equal=bool(payload.get("instrumentation_equal", False)),
        failure=None if payload.get("failure") is None else str(payload["failure"]),
    )


def _bootstrap_benchmark_supervisor(
    args: argparse.Namespace,
    config: Gemma4Config,
    artifact_root: Path,
    output: Path,
) -> dict[str, Any]:
    supervisor_id = os.urandom(16).hex()
    command = _benchmark_worker_command(
        "bootstrap",
        args,
        artifact_root=artifact_root,
        output=output,
        extra=("--supervisor-id", supervisor_id),
    )
    log_path = output / ".supervisor" / "logs" / f"bootstrap-{supervisor_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + config.benchmark_worker_timeout_seconds
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            pass_fds=_inherited_activation_lease_fds(),
        )
        with _cleanup_worker_on_sigterm(process):
            returncode, timed_out = _wait_for_worker(process, deadline=deadline)
    if timed_out or returncode != 0:
        raise RuntimeError(
            "benchmark bootstrap worker failed"
            + (" after timing out" if timed_out else f" with exit {returncode}")
            + f"; see {log_path}"
        )
    mapping_path = (
        output / ".supervisor" / "bootstrap" / f"{supervisor_id}.json"
    )
    mapping = _read_json_object(mapping_path, description="benchmark bootstrap mapping")
    sweep_id = str(mapping.get("sweep_id", ""))
    process_index = int(mapping.get("process_index", -1))
    process_count = int(mapping.get("process_count", 0))
    if (
        mapping.get("schema_version") != "gemma4-benchmark-bootstrap-map-v1"
        or not re.fullmatch(r"[0-9a-f]{32}", sweep_id)
        or process_count <= 0
        or not 0 <= process_index < process_count
    ):
        raise RuntimeError("benchmark bootstrap returned invalid controller identity")
    bootstrap_path = (
        output / ".supervisor" / "sweeps" / sweep_id / "bootstrap.json"
    )
    bootstrap = _read_json_object(bootstrap_path, description="benchmark bootstrap")
    if (
        bootstrap.get("schema_version") != "gemma4-benchmark-bootstrap-v1"
        or bootstrap.get("sweep_id") != sweep_id
        or int(bootstrap.get("process_count", 0)) != process_count
        or bootstrap.get("config_sha256") != config.config_hash
        or bootstrap.get("checkpoint_sha256") != args.checkpoint_sha256
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(bootstrap.get("storage_attestation_sha256", ""))
        )
    ):
        raise RuntimeError("benchmark bootstrap provenance is incompatible")
    return {
        "supervisor_id": supervisor_id,
        "sweep_id": sweep_id,
        "process_index": process_index,
        "process_count": process_count,
        "bootstrap_path": bootstrap_path,
        "bootstrap": bootstrap,
    }


def _run_supervised_benchmark_candidate(
    args: argparse.Namespace,
    config: Gemma4Config,
    artifact_root: Path,
    output: Path,
    supervisor: Mapping[str, Any],
    *,
    mesh: Any,
    batch: int,
) -> BenchmarkObservation:
    process_index = int(supervisor["process_index"])
    process_count = int(supervisor["process_count"])
    sweep_id = str(supervisor["sweep_id"])
    candidate_name = f"{mesh.name}-b{batch}"
    candidate_root = (
        output
        / ".supervisor"
        / "sweeps"
        / sweep_id
        / "candidates"
        / candidate_name
    )
    ready_path = candidate_root / "ready" / f"host-{process_index:05d}.json"
    atomic_write_json(
        ready_path,
        {
            "schema_version": "gemma4-benchmark-ready-v1",
            "sweep_id": sweep_id,
            "mesh": mesh.name,
            "per_device_sequence_batch": batch,
            "process_index": process_index,
        },
    )
    rendezvous_deadline = time.time() + _BENCHMARK_RENDEZVOUS_SECONDS
    control_path = candidate_root / "control.json"
    if process_index == 0:
        _wait_for_paths(
            [
                candidate_root / "ready" / f"host-{index:05d}.json"
                for index in range(process_count)
            ],
            deadline=rendezvous_deadline,
            description=f"{candidate_name} supervisor readiness",
        )
        atomic_write_json(
            control_path,
            {
                "schema_version": "gemma4-benchmark-attempt-v1",
                "sweep_id": sweep_id,
                "attempt_id": os.urandom(16).hex(),
                "mesh": mesh.name,
                "per_device_sequence_batch": batch,
                "deadline_unix": time.time()
                + config.benchmark_worker_timeout_seconds,
            },
        )
    _wait_for_paths(
        [control_path],
        deadline=rendezvous_deadline,
        description=f"{candidate_name} attempt control",
    )
    control = _read_json_object(control_path, description="benchmark attempt control")
    attempt_id = str(control.get("attempt_id", ""))
    deadline = float(control.get("deadline_unix", 0.0))
    if (
        control.get("schema_version") != "gemma4-benchmark-attempt-v1"
        or control.get("sweep_id") != sweep_id
        or control.get("mesh") != mesh.name
        or int(control.get("per_device_sequence_batch", -1)) != batch
        or not re.fullmatch(r"[0-9a-f]{32}", attempt_id)
        or deadline <= time.time()
    ):
        raise RuntimeError(f"invalid shared control for {candidate_name}")

    attempt_root = candidate_root / "attempts" / attempt_id
    command = _benchmark_worker_command(
        "candidate",
        args,
        artifact_root=artifact_root,
        output=output,
        extra=(
            "--bootstrap-record",
            str(supervisor["bootstrap_path"]),
            "--sweep-id",
            sweep_id,
            "--attempt-id",
            attempt_id,
            "--attempt-root",
            str(attempt_root),
            "--mesh",
            mesh.name,
            "--per-device-sequence-batch",
            str(batch),
            "--steps",
            str(max(args.steps, config.benchmark_steps)),
            "--expected-process-index",
            str(process_index),
            "--expected-process-count",
            str(process_count),
        ),
    )
    log_path = attempt_root / "logs" / f"host-{process_index:05d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            pass_fds=_inherited_activation_lease_fds(),
        )
        with _cleanup_worker_on_sigterm(process):
            returncode, timed_out = _wait_for_worker(process, deadline=deadline)
    elapsed = max(time.perf_counter() - started, 1e-9)
    status_path = attempt_root / "status" / f"host-{process_index:05d}.json"
    atomic_write_json(
        status_path,
        {
            "schema_version": "gemma4-benchmark-worker-status-v1",
            "sweep_id": sweep_id,
            "attempt_id": attempt_id,
            "process_index": process_index,
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_seconds": elapsed,
        },
    )
    status_paths = [
        attempt_root / "status" / f"host-{index:05d}.json"
        for index in range(process_count)
    ]
    completion_deadline = (
        deadline + _BENCHMARK_KILL_GRACE_SECONDS + _BENCHMARK_RENDEZVOUS_SECONDS
    )
    result_path = attempt_root / "result.json"
    if process_index == 0:
        _wait_for_paths(
            status_paths,
            deadline=completion_deadline,
            description=f"{candidate_name} worker termination",
        )
        statuses = [
            _read_json_object(path, description="benchmark worker status")
            for path in status_paths
        ]
        valid_statuses = all(
            status.get("schema_version") == "gemma4-benchmark-worker-status-v1"
            and status.get("sweep_id") == sweep_id
            and status.get("attempt_id") == attempt_id
            and int(status.get("process_index", -1)) == index
            for index, status in enumerate(statuses)
        )
        all_succeeded = valid_statuses and all(
            int(status.get("returncode", -1)) == 0
            and status.get("timed_out") is False
            for status in statuses
        )
        raw_path = attempt_root / "raw.json"
        observation: BenchmarkObservation | None = None
        if all_succeeded and raw_path.is_file():
            try:
                raw = _read_json_object(raw_path, description="raw benchmark result")
                if (
                    raw.get("schema_version")
                    != "gemma4-benchmark-worker-result-v1"
                    or raw.get("sweep_id") != sweep_id
                    or raw.get("attempt_id") != attempt_id
                    or not isinstance(raw.get("observation"), Mapping)
                ):
                    raise RuntimeError("raw benchmark result has stale identity")
                observation = _benchmark_observation_from_dict(
                    raw["observation"], mesh=mesh, batch=batch
                )
            except (KeyError, TypeError, ValueError, RuntimeError):
                observation = None
        if observation is None:
            observation = BenchmarkObservation(
                mesh=mesh,
                per_device_sequence_batch=batch,
                synchronized_steps=0,
                elapsed_seconds=max(
                    1e-9,
                    *(
                        float(status.get("elapsed_seconds", 0.0))
                        for status in statuses
                    ),
                ),
                useful_flops=0.0,
                output_tokens=0,
                persisted_tokens=0,
                recompilations=1,
                instrumentation_equal=False,
                failure="candidate subprocess failed or timed out; see attempt telemetry",
            )
        envelope = {
            "schema_version": "gemma4-benchmark-supervised-result-v1",
            "sweep_id": sweep_id,
            "attempt_id": attempt_id,
            "observation": observation.as_dict(),
        }
        atomic_write_json(result_path, envelope)
        atomic_write_json(output / f"{candidate_name}.json", observation.as_dict())
    _wait_for_paths(
        [result_path],
        deadline=completion_deadline,
        description=f"{candidate_name} supervised result",
    )
    result = _read_json_object(result_path, description="supervised benchmark result")
    if (
        result.get("schema_version") != "gemma4-benchmark-supervised-result-v1"
        or result.get("sweep_id") != sweep_id
        or result.get("attempt_id") != attempt_id
        or not isinstance(result.get("observation"), Mapping)
    ):
        raise RuntimeError(f"invalid supervised result for {candidate_name}")
    return _benchmark_observation_from_dict(
        result["observation"], mesh=mesh, batch=batch
    )


def _command_benchmark_harvest(args: argparse.Namespace) -> int:
    """Supervise each mesh/batch in a fresh, coordinated JAX process."""

    config = _config(args)
    config.validate_production()
    artifact_root = _artifact_root(config, args.artifact_root).resolve()
    _require_production_artifact_root(artifact_root)
    output = _benchmark_output_directory(args, artifact_root)
    output.mkdir(parents=True, exist_ok=True)
    supervisor = _bootstrap_benchmark_supervisor(args, config, artifact_root, output)
    observations = [
        _run_supervised_benchmark_candidate(
            args,
            config,
            artifact_root,
            output,
            supervisor,
            mesh=mesh,
            batch=batch,
        )
        for mesh in HARVEST_MESHES
        for batch in PER_DEVICE_SEQUENCE_BATCHES
    ]

    sweep_root = (
        output / ".supervisor" / "sweeps" / str(supervisor["sweep_id"])
    )
    final_path = sweep_root / "final.json"
    if int(supervisor["process_index"]) == 0:
        try:
            chosen = choose_harvest_benchmark(
                observations, minimum_mfu=config.min_harvester_mfu
            )
            bootstrap = supervisor["bootstrap"]
            payload = {
                "schema_version": config.schema_version,
                "config_sha256": config.config_hash,
                "checkpoint_sha256": args.checkpoint_sha256,
                "conversion_attestation_sha256": bootstrap[
                    "conversion_attestation_sha256"
                ],
                "checkpoint_verification_seconds": bootstrap[
                    "checkpoint_verification_seconds"
                ],
                "environment": bootstrap["environment"],
                "storage_attestation_sha256": bootstrap[
                    "storage_attestation_sha256"
                ],
                "benchmark_sweep_id": supervisor["sweep_id"],
                "observations": [item.as_dict() for item in observations],
                "selected": chosen.as_dict(),
            }
            atomic_write_json(output / "selected.json", payload)
            atomic_write_json(
                final_path,
                {
                    "schema_version": "gemma4-benchmark-supervisor-final-v1",
                    "ok": True,
                    "sweep_id": supervisor["sweep_id"],
                },
            )
        except Exception as error:
            atomic_write_json(
                final_path,
                {
                    "schema_version": "gemma4-benchmark-supervisor-final-v1",
                    "ok": False,
                    "sweep_id": supervisor["sweep_id"],
                    "error": str(error),
                },
            )
    _wait_for_paths(
        [final_path],
        deadline=time.time() + _BENCHMARK_RENDEZVOUS_SECONDS,
        description="benchmark sweep finalization",
    )
    final = _read_json_object(final_path, description="benchmark sweep finalization")
    if (
        final.get("schema_version") != "gemma4-benchmark-supervisor-final-v1"
        or final.get("sweep_id") != supervisor["sweep_id"]
        or final.get("ok") is not True
    ):
        raise RuntimeError(str(final.get("error", "benchmark sweep failed")))
    if int(supervisor["process_index"]) == 0:
        print(json.dumps(payload["selected"], indent=2, sort_keys=True))
    return 0


def _mesh_by_name(name: str):
    for candidate in HARVEST_MESHES:
        if candidate.name == name:
            return candidate
    raise ValueError(f"unknown harvest mesh {name!r}")


def _load_selected_benchmark(
    path: Path,
    config: Gemma4Config,
    checkpoint_sha256: str,
    conversion_attestation_sha256: str,
    storage_attestation_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load selected benchmark: {path}") from error
    environment = payload.get("environment")
    current_environment = _coordinated_environment()
    if (
        payload.get("schema_version") != config.schema_version
        or payload.get("config_sha256") != config.config_hash
        or payload.get("checkpoint_sha256") != checkpoint_sha256
        or payload.get("conversion_attestation_sha256")
        != conversion_attestation_sha256
        or (
            storage_attestation_sha256 is not None
            and payload.get("storage_attestation_sha256")
            != storage_attestation_sha256
        )
        or not isinstance(environment, dict)
        or environment.get("sha256") != current_environment.get("sha256")
    ):
        raise RuntimeError(
            "selected benchmark belongs to a different config/checkpoint/environment"
        )
    selected = payload.get("selected")
    if not isinstance(selected, dict) or selected.get("passes_hard_gate") is not True:
        raise RuntimeError("selected benchmark did not pass every hard gate")
    if float(selected.get("useful_mfu", 0.0)) < config.min_harvester_mfu:
        raise RuntimeError("selected benchmark does not meet the configured MFU gate")
    return selected


def _command_harvest(args: argparse.Namespace) -> int:
    from moe_sae.harvest import run_production_harvest

    require_v4_32_runtime()
    config = _config(args)
    config.validate_production()
    root = _artifact_root(config, args.artifact_root)
    _require_production_artifact_root(root)
    _verify_checkpoint_inventory(args)
    conversion_attestation_sha256 = _load_conversion_attestation(
        root, config, args.checkpoint_sha256
    )
    storage_payload = _load_storage_attestation_payload(root, config)
    storage_layout = _storage_layout(storage_payload)
    storage_attestation_sha256 = str(storage_payload["attestation_sha256"])
    split = args.split
    expected = config.train_tokens if split == "train" else config.eval_tokens
    stop_after = config.canary_tokens if args.canary else expected
    provenance = _verified_dataset_provenance(
        args,
        config,
        split=split,
        grain_files=args.grain_files,
        manifest_location=args.dataset_manifest,
    )
    benchmark_path = (
        Path(args.benchmark) if args.benchmark else root / "benchmark" / "selected.json"
    )
    selected = _load_selected_benchmark(
        benchmark_path,
        config,
        args.checkpoint_sha256,
        conversion_attestation_sha256,
        storage_attestation_sha256,
    )
    selected_mesh = selected.get("mesh")
    selected_mesh_name = (
        selected_mesh.get("name") if isinstance(selected_mesh, dict) else selected_mesh
    )
    mesh_name = args.mesh or selected_mesh_name
    sequence_batch = args.per_device_sequence_batch or int(
        selected.get("per_device_sequence_batch", 0)
    )
    if mesh_name != selected_mesh_name or sequence_batch != int(
        selected.get("per_device_sequence_batch", -1)
    ):
        raise RuntimeError("harvest mesh/batch must match the selected benchmark")
    result = run_production_harvest(
        config=config,
        split=split,
        checkpoint_path=args.checkpoint,
        grain_files=args.grain_files,
        staged_files=provenance.files,
        artifact_root=_capture_root(root, split),
        checkpoint_sha256=args.checkpoint_sha256,
        conversion_attestation_sha256=conversion_attestation_sha256,
        dataset_fingerprint=provenance.fingerprint_sha256,
        storage_attestation_sha256=storage_attestation_sha256,
        capture_object_prefix=join_gcs_uri(
            storage_layout["bulk_gcs_root"],
            "captures",
            config.config_hash,
            split,
        ),
        local_buffer_root=storage_layout["local_buffer_root"],
        mesh_candidate=_mesh_by_name(str(mesh_name)),
        per_device_sequence_batch=sequence_batch,
        stop_after_tokens=stop_after,
        benchmark_observation=selected,
        publish=not args.canary,
        repo_root=REPO_ROOT,
    )
    payload = result.as_dict()
    output_name = f"{split}-canary.json" if args.canary else f"{split}.json"
    output = root / "harvest" / output_name
    coordinator = _coordinator_write_json(
        output, payload, barrier=f"gemma4-{split}-{stop_after}-summary"
    )
    if coordinator:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _command_train(args: argparse.Namespace) -> int:
    config = _config(args)
    if not args.reference:
        config.validate_production()
        require_v4_32_runtime()
    if args.condition not in config.conditions or args.seed not in config.seeds:
        raise ValueError(
            "condition/seed is outside the single experiment configuration"
        )
    root = _artifact_root(config, args.artifact_root)
    if not args.reference:
        _require_production_artifact_root(root)
    current_environment = _coordinated_environment()
    if args.reference:
        storage_layout = None
        storage_attestation_sha256 = None
        checkpoint_payload_root = None
    else:
        storage_payload = _load_storage_attestation_payload(
            root, config, environment=current_environment
        )
        _require_sae_cache_qualification(storage_payload)
        storage_layout = _storage_layout(storage_payload)
        storage_attestation_sha256 = str(storage_payload["attestation_sha256"])
        checkpoint_payload_root = str(storage_layout["sae_checkpoint_root"])
    dataset = _sealed_dataset(
        _capture_root(root, "train"),
        cache_root=(
            None if storage_layout is None else storage_layout["capture_cache_root"]
        ),
        cache_max_bytes=(
            None if storage_layout is None else config.capture_cache_max_bytes
        ),
    )
    _require_exact_capture(dataset, config.train_tokens, "training")
    capture_manifest_sha256 = dataset.manifest_sha256
    capture_checkpoint_sha256 = str(dataset.manifest["metadata"]["checkpoint_sha256"])
    conversion_attestation_sha256 = (
        None
        if args.reference
        else _load_conversion_attestation(root, config, capture_checkpoint_sha256)
    )
    _require_capture_identity(
        dataset,
        config,
        "train",
        conversion_attestation_sha256=conversion_attestation_sha256,
        environment_sha256=current_environment["sha256"],
        storage_attestation_sha256=storage_attestation_sha256,
        object_prefix=(
            None
            if storage_layout is None
            else join_gcs_uri(
                storage_layout["bulk_gcs_root"],
                "captures",
                config.config_hash,
                "train",
            )
        ),
    )
    capture_prefetch = {"objects": 0, "bytes": 0}
    capture_prefetch_wall_seconds = 0.0
    if storage_layout is not None:
        capture_prefetch, capture_prefetch_wall_seconds = _prefetch_process_partition(
            dataset, config.train_tokens, label="train-capture"
        )
    capacity = args.a_capacity
    capacity_precompute_wall_seconds = 0.0
    training_config = _training_config(
        config, args.condition, args.seed, a_capacity=capacity
    )
    if args.condition == "A" and capacity is None:
        if args.reference:
            started = time.perf_counter()
            capacity = _precompute_a_capacity(dataset, config)
            capacity_precompute_wall_seconds = time.perf_counter() - started
        else:
            capacity, capacity_precompute_wall_seconds = _production_a_capacity(
                root, dataset, config, training_config
            )
        training_config = dataclasses.replace(training_config, a_capacity=capacity)
    elif args.condition == "A" and not args.reference:
        raise RuntimeError(
            "authoritative A runs must measure their fixed-order capacity prepass; "
            "--a-capacity is reference-only"
        )
    run_root = _run_root(root, args.condition, args.seed)
    run_root.mkdir(parents=True, exist_ok=True)
    sink = MetricsSink(run_root)
    prior: dict[str, Any] = {}
    prior_path = run_root / "run.json"
    identity_path = run_root / "identity.json"
    expected_identity = _run_identity_expected(
        config,
        capture_manifest_sha256=capture_manifest_sha256,
        capture_checkpoint_sha256=capture_checkpoint_sha256,
        environment_sha256=current_environment["sha256"],
        condition=args.condition,
        seed=args.seed,
        storage_attestation_sha256=storage_attestation_sha256,
        checkpoint_payload_root=checkpoint_payload_root,
    )
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if (
            prior.get("config_sha256") != config.config_hash
            or prior.get("condition") != args.condition
            or prior.get("seed") != args.seed
            or prior.get("capture_manifest_sha256") != capture_manifest_sha256
            or prior.get("capture_checkpoint_sha256") != capture_checkpoint_sha256
            or prior.get("storage_attestation_sha256")
            != storage_attestation_sha256
            or prior.get("checkpoint_payload_root") != checkpoint_payload_root
            or not re.fullmatch(r"[0-9a-f]{32}", str(prior.get("run_instance_id", "")))
            or not isinstance(prior.get("environment"), dict)
            or prior["environment"].get("sha256") != current_environment.get("sha256")
        ):
            raise RuntimeError(
                "existing run metadata belongs to a different config/run/environment"
            )
    if identity_path.exists():
        run_instance_id = _load_run_identity(identity_path, expected_identity)
    else:
        if prior:
            raise RuntimeError("existing run metadata lacks its immutable identity")
        run_instance_id = _new_run_instance_id()
        _coordinator_write_json(
            identity_path,
            {**expected_identity, "run_instance_id": run_instance_id},
            barrier=f"gemma4-run-identity-{args.condition}-{args.seed}",
        )
    if prior and prior.get("run_instance_id") != run_instance_id:
        raise RuntimeError("run.json and immutable run identity disagree")
    if prior:
        prior_tokens = int(prior.get("tokens_seen", -1))
        latest_name = "latest.npz" if args.reference else "latest.json"
        if prior_tokens > 0 and not (run_root / latest_name).exists():
            raise RuntimeError(
                "existing run metadata has progress but its latest checkpoint is missing"
            )
        if prior_tokens == config.train_tokens:
            final_name = "final.npz" if args.reference else "final.json"
            if not (run_root / final_name).exists():
                raise RuntimeError(
                    "complete run metadata exists without its final checkpoint"
                )

    if args.reference:
        from moe_sae.pipeline import load_checkpoint, train

        latest = run_root / "latest.npz"
        state = load_checkpoint(latest, training_config)[0] if latest.exists() else None
        start_tokens = 0 if state is None else int(np.asarray(state.tokens_seen))
        started = time.perf_counter()
        result = train(
            training_config,
            dataset.iter_batches(),
            dataset.mean,
            state=state,
            checkpoint_dir=run_root,
            stop_after_tokens=args.stop_after_tokens,
            metric_callback=lambda metric: sink.log(_metric_record(metric)),
        )
        segment_optimizer_seconds = time.perf_counter() - started
        segment_compile_seconds = 0.0
        segment_checkpoint_seconds = 0.0
        segment_tokens = int(np.asarray(result.state.tokens_seen)) - start_tokens
        chip_count = 1
        authoritative = False
        timing_is_cumulative = False
    else:
        from moe_sae.distributed import train_distributed

        result = train_distributed(
            training_config,
            dataset,
            dataset.mean,
            checkpoint_dir=run_root,
            checkpoint_payload_root=checkpoint_payload_root,
            stop_after_tokens=args.stop_after_tokens,
            metric_callback=lambda metric: sink.log(_metric_record(metric)),
            checkpoint_identity=_checkpoint_identity_expected(
                config,
                capture_manifest_sha256=capture_manifest_sha256,
                capture_checkpoint_sha256=capture_checkpoint_sha256,
                environment_sha256=current_environment["sha256"],
                condition=args.condition,
                seed=args.seed,
                run_instance_id=run_instance_id,
                storage_attestation_sha256=storage_attestation_sha256,
                checkpoint_payload_root=checkpoint_payload_root,
            ),
            prior_run_timing=prior,
        )
        segment_optimizer_seconds = result.optimizer_wall_seconds
        segment_compile_seconds = result.compile_seconds
        segment_checkpoint_seconds = result.checkpoint_seconds
        chip_count = result.chip_count
        authoritative = True
        timing_is_cumulative = True

    tokens_seen = int(np.asarray(result.state.tokens_seen))
    if timing_is_cumulative:
        prior_tokens = int(prior.get("tokens_seen", 0)) if prior else 0
        segment_tokens = max(0, tokens_seen - prior_tokens)
    can_accumulate = (
        not timing_is_cumulative
        and bool(prior)
        and int(prior.get("tokens_seen", -1)) == tokens_seen - segment_tokens
        and bool(prior.get("authoritative")) == authoritative
        and int(prior.get("chip_count", -1)) == chip_count
    )
    optimizer_wall_seconds = segment_optimizer_seconds + (
        float(prior.get("optimizer_wall_seconds", 0.0)) if can_accumulate else 0.0
    )
    compile_seconds = segment_compile_seconds + (
        float(prior.get("compile_seconds", 0.0)) if can_accumulate else 0.0
    )
    checkpoint_seconds = segment_checkpoint_seconds + (
        float(prior.get("checkpoint_seconds", 0.0)) if can_accumulate else 0.0
    )
    marginal_wall_seconds = optimizer_wall_seconds + capacity_precompute_wall_seconds
    timing = {
        "schema_version": config.schema_version,
        "config_sha256": config.config_hash,
        "environment": current_environment,
        "capture_manifest_sha256": capture_manifest_sha256,
        "capture_checkpoint_sha256": capture_checkpoint_sha256,
        "condition": args.condition,
        "seed": args.seed,
        "run_instance_id": run_instance_id,
        "tokens_seen": tokens_seen,
        "segment_tokens": segment_tokens,
        "a_capacity": capacity,
        "authoritative": authoritative,
        "optimizer_wall_seconds": optimizer_wall_seconds,
        "compile_seconds": compile_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "storage_attestation_sha256": storage_attestation_sha256,
        "checkpoint_payload_root": checkpoint_payload_root,
        "capture_prefetch": capture_prefetch,
        "capture_prefetch_wall_seconds_excluded": capture_prefetch_wall_seconds,
        "a_capacity_precompute_wall_seconds": capacity_precompute_wall_seconds,
        "marginal_wall_seconds": marginal_wall_seconds,
        "chip_count": chip_count,
        "chip_seconds": marginal_wall_seconds * chip_count,
        "tokens_per_second": tokens_seen / marginal_wall_seconds
        if marginal_wall_seconds
        else 0.0,
        "optimizer_tokens_per_second": (
            tokens_seen / optimizer_wall_seconds if optimizer_wall_seconds else 0.0
        ),
        "health": None if result.health is None else dataclasses.asdict(result.health),
    }
    coordinator = _coordinator_write_json(
        run_root / "run.json",
        timing,
        barrier=f"gemma4-train-{args.condition}-{args.seed}-{tokens_seen}",
    )
    if coordinator:
        print(json.dumps(timing, indent=2, sort_keys=True))
    if result.health is not None and not result.health.passed:
        raise RuntimeError(
            "SAE health gate failed: " + "; ".join(result.health.reasons)
        )
    return 0


def _command_eval_one(
    args: argparse.Namespace, config: Gemma4Config, condition: str, seed: int
) -> dict[str, Any]:
    if not args.reference:
        config.validate_production()
        require_v4_32_runtime()
    root = _artifact_root(config, args.artifact_root)
    if not args.reference:
        _require_production_artifact_root(root)
    current_environment = _coordinated_environment()
    if args.reference:
        storage_layout = None
        storage_attestation_sha256 = None
        checkpoint_payload_root = None
    else:
        storage_payload = _load_storage_attestation_payload(
            root, config, environment=current_environment
        )
        _require_sae_cache_qualification(storage_payload)
        storage_layout = _storage_layout(storage_payload)
        storage_attestation_sha256 = str(storage_payload["attestation_sha256"])
        checkpoint_payload_root = str(storage_layout["sae_checkpoint_root"])
    cache_root = (
        None if storage_layout is None else storage_layout["capture_cache_root"]
    )
    cache_max_bytes = (
        None if storage_layout is None else config.capture_cache_max_bytes
    )
    train_dataset = _sealed_dataset(
        _capture_root(root, "train"),
        cache_root=cache_root,
        cache_max_bytes=cache_max_bytes,
    )
    eval_dataset = _sealed_dataset(
        _capture_root(root, "eval"),
        cache_root=cache_root,
        cache_max_bytes=cache_max_bytes,
    )
    _require_exact_capture(train_dataset, config.train_tokens, "training")
    _require_exact_capture(eval_dataset, config.eval_tokens, "held-out")
    train_checkpoint_sha = str(train_dataset.manifest["metadata"]["checkpoint_sha256"])
    eval_checkpoint_sha = str(eval_dataset.manifest["metadata"]["checkpoint_sha256"])
    if train_checkpoint_sha != eval_checkpoint_sha:
        raise RuntimeError("training and held-out captures use different checkpoints")
    conversion_attestation_sha256 = (
        None
        if args.reference
        else _load_conversion_attestation(root, config, train_checkpoint_sha)
    )
    _require_capture_identity(
        train_dataset,
        config,
        "train",
        conversion_attestation_sha256=conversion_attestation_sha256,
        environment_sha256=current_environment["sha256"],
        storage_attestation_sha256=storage_attestation_sha256,
        object_prefix=(
            None
            if storage_layout is None
            else join_gcs_uri(
                storage_layout["bulk_gcs_root"],
                "captures",
                config.config_hash,
                "train",
            )
        ),
    )
    _require_capture_identity(
        eval_dataset,
        config,
        "eval",
        conversion_attestation_sha256=conversion_attestation_sha256,
        environment_sha256=current_environment["sha256"],
        storage_attestation_sha256=storage_attestation_sha256,
        object_prefix=(
            None
            if storage_layout is None
            else join_gcs_uri(
                storage_layout["bulk_gcs_root"],
                "captures",
                config.config_hash,
                "eval",
            )
        ),
    )
    evaluation_prefetch = {"objects": 0, "bytes": 0}
    evaluation_prefetch_seconds = 0.0
    if storage_layout is not None:
        evaluation_prefetch, evaluation_prefetch_seconds = (
            _prefetch_process_partition(
                eval_dataset, config.eval_tokens, label="eval-capture"
            )
        )
    run_root = _run_root(root, condition, seed)
    timing = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    run_instance_id = _load_run_identity(
        run_root / "identity.json",
        _run_identity_expected(
            config,
            capture_manifest_sha256=train_dataset.manifest_sha256,
            capture_checkpoint_sha256=train_checkpoint_sha,
            environment_sha256=current_environment["sha256"],
            condition=condition,
            seed=seed,
            storage_attestation_sha256=storage_attestation_sha256,
            checkpoint_payload_root=checkpoint_payload_root,
        ),
    )
    if (
        timing.get("config_sha256") != config.config_hash
        or timing.get("condition") != condition
        or timing.get("seed") != seed
        or int(timing.get("tokens_seen", -1)) != config.train_tokens
        or timing.get("capture_manifest_sha256") != train_dataset.manifest_sha256
        or timing.get("capture_checkpoint_sha256") != train_checkpoint_sha
        or timing.get("storage_attestation_sha256")
        != storage_attestation_sha256
        or timing.get("checkpoint_payload_root") != checkpoint_payload_root
        or not re.fullmatch(r"[0-9a-f]{32}", str(timing.get("run_instance_id", "")))
        or timing.get("run_instance_id") != run_instance_id
        or not isinstance(timing.get("environment"), dict)
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(timing["environment"].get("sha256", ""))
        )
    ):
        raise RuntimeError("final evaluation requires a matching complete training run")
    if timing.get("authoritative") is True:
        _validate_authoritative_run_timing(
            timing,
            config,
            condition,
            expected_tokens=config.train_tokens,
        )
    if not timing.get("authoritative") and not args.allow_reference_cost:
        raise RuntimeError("reference timing is not valid for the primary cost report")
    health = timing.get("health")
    if not isinstance(health, dict) or health.get("passed") is not True:
        raise RuntimeError("final evaluation requires a passing SAE health gate")
    if timing.get("authoritative") and int(timing.get("chip_count", -1)) != 16:
        raise RuntimeError("authoritative SAE cost requires exactly 16 TPU v4 chips")
    training_config = _training_config(
        config, condition, seed, a_capacity=timing.get("a_capacity")
    )
    evaluation_capacity_seconds = 0.0
    if condition == "A":
        if args.reference:
            capacity_started = time.perf_counter()
            evaluation_capacity = _precompute_a_capacity(
                eval_dataset, config, expected_tokens=config.eval_tokens
            )
            evaluation_capacity_seconds = time.perf_counter() - capacity_started
        else:
            evaluation_capacity, evaluation_capacity_seconds = _production_a_capacity(
                root,
                eval_dataset,
                config,
                training_config,
                scope="eval",
                total_tokens=config.eval_tokens,
            )
        training_config = dataclasses.replace(
            training_config, a_capacity=evaluation_capacity
        )
    expected_sst = eval_dataset.sst_about(train_dataset.mean)
    started = time.perf_counter()
    if args.reference:
        from moe_sae.pipeline import evaluate, load_checkpoint

        state, _ = load_checkpoint(run_root / "final.npz", training_config)
        if (
            int(np.asarray(state.tokens_seen)) != config.train_tokens
            or int(np.asarray(state.step)) != config.planned_updates
        ):
            raise RuntimeError(
                "reference final checkpoint is not a complete training run"
            )
        result = evaluate(
            state.params,
            training_config,
            _reference_evaluation_batches(
                eval_dataset, training_config.microbatch_tokens
            ),
            train_dataset.mean,
            max_tokens=config.eval_tokens,
            expected_sst=expected_sst,
        )
    else:
        from moe_sae.distributed import evaluate_distributed

        result = evaluate_distributed(
            run_root / "final.json",
            training_config,
            eval_dataset,
            train_dataset.mean,
            max_tokens=config.eval_tokens,
            expected_sst=expected_sst,
            checkpoint_identity=_checkpoint_identity_expected(
                config,
                capture_manifest_sha256=train_dataset.manifest_sha256,
                capture_checkpoint_sha256=train_checkpoint_sha,
                environment_sha256=timing["environment"]["sha256"],
                condition=condition,
                seed=seed,
                run_instance_id=timing["run_instance_id"],
                storage_attestation_sha256=storage_attestation_sha256,
                checkpoint_payload_root=checkpoint_payload_root,
            ),
            checkpoint_payload_root=checkpoint_payload_root,
        )
    evaluation_seconds = time.perf_counter() - started
    if not args.reference:
        evaluation_seconds = _global_max_seconds(evaluation_seconds)
    record = {
        "schema_version": config.schema_version,
        "config_sha256": config.config_hash,
        "environment": _coordinated_environment(),
        "train_capture_manifest_sha256": train_dataset.manifest_sha256,
        "eval_capture_manifest_sha256": eval_dataset.manifest_sha256,
        "capture_checkpoint_sha256": train_checkpoint_sha,
        "training_environment_sha256": timing["environment"]["sha256"],
        "condition": condition,
        "seed": seed,
        "run_instance_id": timing["run_instance_id"],
        "tokens": result.tokens,
        "sse": result.sse,
        "sst": result.sst,
        "ev": result.explained_variance,
        "nmse": result.nmse,
        "l0": result.l0,
        "dead_rate": result.dead_rate,
        "topk_valid": result.topk_valid,
        "chip_seconds": timing["chip_seconds"],
        "wall_seconds": timing.get(
            "marginal_wall_seconds", timing["optimizer_wall_seconds"]
        ),
        "tokens_per_second": timing["tokens_per_second"],
        "evaluation_seconds_excluded": (
            evaluation_seconds
            + evaluation_capacity_seconds
            + evaluation_prefetch_seconds
        ),
        "capture_prefetch": evaluation_prefetch,
        "capture_prefetch_wall_seconds_excluded": evaluation_prefetch_seconds,
        "evaluation_capacity": training_config.a_capacity if condition == "A" else None,
        "evaluation_capacity_precompute_seconds_excluded": evaluation_capacity_seconds,
    }
    SeedEvaluation.from_value(record)
    destination = _evaluation_path(root, condition, seed)
    process_index, _ = _process_context()
    if process_index == 0:
        write_report_json(destination, record)
    _global_barrier(f"gemma4-eval-{condition}-{seed}")
    if not destination.exists():
        raise RuntimeError(
            f"evaluation output is not visible after barrier: {destination}"
        )
    return record


def _command_eval(args: argparse.Namespace) -> int:
    config = _config(args)
    pairs = (
        [(condition, seed) for condition in config.conditions for seed in config.seeds]
        if args.all
        else [(args.condition, args.seed)]
    )
    records = [
        _command_eval_one(args, config, condition, seed) for condition, seed in pairs
    ]
    if _process_context()[0] == 0:
        print(json.dumps(records, indent=2, sort_keys=True))
    return 0


def _load_harvester_report(
    root: Path,
    config: Gemma4Config,
    override: str | None,
    *,
    expected_captures: dict[str, ArtifactDataset] | None = None,
    environment_sha256: str | None = None,
    conversion_attestation_sha256: str | None = None,
    storage_attestation_sha256: str | None = None,
) -> dict[str, Any]:
    """Combine the selected MFU benchmark with measured shared capture cost."""

    from moe_sae.harvest import CampaignSummary

    benchmark_path = (
        Path(override) if override else root / "benchmark" / "selected.json"
    )
    if not benchmark_path.exists():
        raise RuntimeError(f"missing selected harvester benchmark: {benchmark_path}")
    value = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark_environment = value.get("environment")
    if (
        value.get("schema_version") != config.schema_version
        or value.get("config_sha256") != config.config_hash
        or (
            conversion_attestation_sha256 is not None
            and value.get("conversion_attestation_sha256")
            != conversion_attestation_sha256
        )
        or (
            storage_attestation_sha256 is not None
            and value.get("storage_attestation_sha256")
            != storage_attestation_sha256
        )
        or (
            environment_sha256 is not None
            and (
                not isinstance(benchmark_environment, dict)
                or benchmark_environment.get("sha256") != environment_sha256
            )
        )
    ):
        raise RuntimeError(
            "selected harvester benchmark has the wrong config/environment fingerprint"
        )
    selected = value.get("selected", value)
    if not isinstance(selected, dict) or selected.get("passes_hard_gate") is not True:
        raise RuntimeError(f"invalid/failing harvester benchmark: {benchmark_path}")
    if float(selected.get("useful_mfu", 0.0)) < config.min_harvester_mfu:
        raise RuntimeError("selected harvester benchmark is below the MFU hard gate")
    result: dict[str, Any] = dict(selected)
    result["checkpoint_sha256"] = value.get("checkpoint_sha256")
    result["conversion_attestation_sha256"] = value.get(
        "conversion_attestation_sha256"
    )
    result["storage_attestation_sha256"] = value.get(
        "storage_attestation_sha256"
    )
    if not re.fullmatch(r"[0-9a-f]{64}", str(result["checkpoint_sha256"])):
        raise RuntimeError("selected harvester benchmark lacks checkpoint provenance")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(result["conversion_attestation_sha256"])
    ):
        raise RuntimeError(
            "selected harvester benchmark lacks conversion-parity provenance"
        )
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(result["storage_attestation_sha256"])
    ):
        raise RuntimeError("selected harvester benchmark lacks storage provenance")

    campaigns: dict[str, Any] = {}
    capture_chip_seconds = 0.0
    have_capture_cost = False
    for split in ("train", "eval"):
        path = root / "harvest" / f"{split}.json"
        if not path.exists():
            raise RuntimeError(f"missing measured {split} harvest campaign: {path}")
        raw_campaign = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_campaign, dict):
            raise RuntimeError(f"invalid harvest campaign payload: {path}")
        campaign = CampaignSummary.from_mapping(raw_campaign).as_dict()
        expected_tokens = (
            config.train_tokens if split == "train" else config.eval_tokens
        )
        if (
            campaign.get("schema_version") != "gemma4-harvest-campaign-v2"
            or campaign.get("phase") != "production"
            or campaign.get("split") != split
            or int(campaign.get("captured_tokens", -1)) != expected_tokens
            or not campaign.get("manifest_path")
            or int(campaign.get("global_chips", -1)) != 16
            or campaign.get("config_sha256") != config.config_hash
            or campaign.get("checkpoint_sha256") != result["checkpoint_sha256"]
            or campaign.get("conversion_attestation_sha256")
            != result["conversion_attestation_sha256"]
        ):
            raise RuntimeError(f"incomplete/incompatible {split} harvest campaign")
        raw_mesh = selected.get("mesh")
        selected_mesh = raw_mesh.get("name") if isinstance(raw_mesh, dict) else raw_mesh
        if (
            campaign.get("mesh_name") != selected_mesh
            or int(campaign.get("per_device_sequence_batch", -1))
            != int(selected.get("per_device_sequence_batch", -2))
            or not math.isclose(
                float(campaign.get("benchmark_useful_mfu", -1.0)),
                float(selected.get("useful_mfu", -2.0)),
                rel_tol=1e-12,
            )
            or not math.isclose(
                float(campaign.get("synthetic_goodput", -1.0)),
                float(selected.get("persisted_tokens_per_second", -2.0)),
                rel_tol=1e-12,
            )
            or int(campaign.get("dropped_routed_tokens", -1))
            != int(selected.get("dropped_routed_tokens", -2))
            or int(campaign.get("recompilations", -1))
            != int(selected.get("recompilations", -2))
            or campaign.get("instrumentation_equal")
            is not selected.get("instrumentation_equal")
        ):
            raise RuntimeError(f"{split} campaign does not use the selected benchmark")
        if expected_captures is not None:
            capture = expected_captures[split]
            capture_metadata = capture.manifest["metadata"]
            if (
                campaign.get("capture_manifest_sha256") != capture.manifest_sha256
                or campaign.get("dataset_fingerprint")
                != capture_metadata.get("dataset_fingerprint")
                or campaign.get("environment_sha256")
                != capture_metadata.get("environment_sha256")
                or capture_metadata.get("conversion_attestation_sha256")
                != result["conversion_attestation_sha256"]
                or capture_metadata.get("storage_attestation_sha256")
                != result["storage_attestation_sha256"]
            ):
                raise RuntimeError(
                    f"{split} harvest cost is not bound to the sealed capture"
                )
        if split == "train" and campaign.get("canary_passed") is not True:
            raise RuntimeError(
                "training harvest does not descend from a passing canary"
            )
        campaigns[split] = campaign
        if campaign.get("chip_seconds") is None:
            raise RuntimeError(f"{split} harvest lacks measured chip-seconds")
        capture_chip_seconds += float(campaign["chip_seconds"])
        have_capture_cost = True
    result["campaigns"] = campaigns
    if not have_capture_cost:
        raise RuntimeError("shared harvest has no measured cost")
    result["chip_seconds"] = capture_chip_seconds
    return result


def _canonical_evaluation_metrics(
    record: dict[str, Any], authoritative_sst: float
) -> dict[str, Any]:
    """Bind stored reconstruction output to sealed capture statistics."""

    try:
        sse = float(record["sse"])
        stored_sst = float(record["sst"])
        stored_nmse = float(record["nmse"])
        stored_ev = float(record["ev"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("evaluation lacks numeric SSE/SST/NMSE/EV") from error
    if (
        not math.isfinite(sse)
        or sse < 0.0
        or not math.isfinite(authoritative_sst)
        or authoritative_sst <= 0.0
    ):
        raise RuntimeError("evaluation SSE or authoritative SST is invalid")
    nmse = sse / authoritative_sst
    ev = 1.0 - nmse
    if (
        not math.isclose(stored_sst, authoritative_sst, rel_tol=1e-12, abs_tol=0.0)
        or not math.isclose(stored_nmse, nmse, rel_tol=1e-12, abs_tol=1e-15)
        or not math.isclose(stored_ev, ev, rel_tol=1e-12, abs_tol=1e-15)
    ):
        raise RuntimeError(
            "evaluation metrics disagree with SSE and sealed held-out SST"
        )
    canonical = dict(record)
    canonical.update({"sst": authoritative_sst, "nmse": nmse, "ev": ev})
    return canonical


def _validate_authoritative_run_timing(
    record: Mapping[str, Any],
    config: Gemma4Config,
    condition: str,
    *,
    expected_tokens: int | None = None,
) -> None:
    """Recompute every primary cost field from its measured components."""

    try:
        tokens = int(record["tokens_seen"])
        chip_count = int(record["chip_count"])
        optimizer = float(record["optimizer_wall_seconds"])
        compile_seconds = float(record["compile_seconds"])
        checkpoint_seconds = float(record["checkpoint_seconds"])
        capacity = float(record["a_capacity_precompute_wall_seconds"])
        marginal = float(record["marginal_wall_seconds"])
        chip_seconds = float(record["chip_seconds"])
        tokens_per_second = float(record["tokens_per_second"])
        optimizer_tokens_per_second = float(record["optimizer_tokens_per_second"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("run lacks complete numeric cost telemetry") from error

    values = (
        optimizer,
        compile_seconds,
        checkpoint_seconds,
        capacity,
        marginal,
        chip_seconds,
        tokens_per_second,
        optimizer_tokens_per_second,
    )
    expected_marginal = optimizer + capacity
    expected_throughput = tokens / marginal if marginal > 0.0 else math.nan
    expected_optimizer_throughput = (
        tokens / optimizer if optimizer > 0.0 else math.nan
    )
    if (
        condition not in config.conditions
        or record.get("authoritative") is not True
        or chip_count != 16
        or tokens <= 0
        or (expected_tokens is not None and tokens != expected_tokens)
        or any(not math.isfinite(value) for value in values)
        or optimizer <= 0.0
        or compile_seconds < 0.0
        or checkpoint_seconds < 0.0
        or capacity < 0.0
        or marginal <= 0.0
        or chip_seconds <= 0.0
        or tokens_per_second <= 0.0
        or optimizer_tokens_per_second <= 0.0
        or (condition == "A" and capacity <= 0.0)
        or (condition != "A" and capacity != 0.0)
        or not math.isclose(marginal, expected_marginal, rel_tol=1e-12)
        or not math.isclose(chip_seconds, chip_count * marginal, rel_tol=1e-12)
        or not math.isclose(
            tokens_per_second, expected_throughput, rel_tol=1e-12
        )
        or not math.isclose(
            optimizer_tokens_per_second,
            expected_optimizer_throughput,
            rel_tol=1e-12,
        )
    ):
        raise RuntimeError("authoritative SAE cost telemetry is internally inconsistent")


def _command_report(args: argparse.Namespace) -> int:
    config = _config(args)
    config.validate_production()
    root = _artifact_root(config, args.artifact_root)
    _require_production_artifact_root(root)
    current_environment = _coordinated_environment()
    storage_payload = _load_storage_attestation_payload(
        root, config, environment=current_environment
    )
    _require_sae_cache_qualification(storage_payload)
    storage_layout = _storage_layout(storage_payload)
    storage_attestation_sha256 = str(storage_payload["attestation_sha256"])
    pairs = [
        (condition, seed) for condition in config.conditions for seed in config.seeds
    ]
    evaluations = [
        json.loads(_evaluation_path(root, condition, seed).read_text(encoding="utf-8"))
        for condition, seed in pairs
    ]
    train_capture = _sealed_dataset(_capture_root(root, "train"))
    eval_capture = _sealed_dataset(_capture_root(root, "eval"))
    producer_environment_sha256 = str(
        train_capture.manifest["metadata"].get("environment_sha256", "")
    )
    if not re.fullmatch(r"[0-9a-f]{64}", producer_environment_sha256):
        raise RuntimeError("training capture lacks its producer environment")
    _require_exact_capture(train_capture, config.train_tokens, "training")
    _require_exact_capture(eval_capture, config.eval_tokens, "held-out")
    checkpoint_sha = str(train_capture.manifest["metadata"]["checkpoint_sha256"])
    if eval_capture.manifest["metadata"]["checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError("report captures use different base checkpoints")
    conversion_attestation_sha256 = _load_conversion_attestation(
        root, config, checkpoint_sha
    )
    _require_capture_identity(
        train_capture,
        config,
        "train",
        conversion_attestation_sha256=conversion_attestation_sha256,
        environment_sha256=producer_environment_sha256,
        storage_attestation_sha256=storage_attestation_sha256,
        object_prefix=join_gcs_uri(
            storage_layout["bulk_gcs_root"],
            "captures",
            config.config_hash,
            "train",
        ),
    )
    _require_capture_identity(
        eval_capture,
        config,
        "eval",
        conversion_attestation_sha256=conversion_attestation_sha256,
        environment_sha256=producer_environment_sha256,
        storage_attestation_sha256=storage_attestation_sha256,
        object_prefix=join_gcs_uri(
            storage_layout["bulk_gcs_root"],
            "captures",
            config.config_hash,
            "eval",
        ),
    )
    common_sst = eval_capture.sst_about(train_capture.mean)
    canonical_evaluations: list[dict[str, Any]] = []
    for (condition, seed), record in zip(pairs, evaluations, strict=True):
        if (
            record.get("config_sha256") != config.config_hash
            or record.get("condition") != condition
            or record.get("seed") != seed
            or int(record.get("tokens", -1)) != config.eval_tokens
            or record.get("topk_valid") is not True
            or float(record.get("l0", -1.0)) != float(config.topk_features)
            or record.get("train_capture_manifest_sha256")
            != train_capture.manifest_sha256
            or record.get("eval_capture_manifest_sha256")
            != eval_capture.manifest_sha256
            or record.get("capture_checkpoint_sha256") != checkpoint_sha
            or not isinstance(record.get("environment"), dict)
            or record["environment"].get("sha256") != producer_environment_sha256
        ):
            raise RuntimeError(
                f"incomplete/incompatible evaluation for {condition}/seed-{seed}"
            )
        canonical_evaluations.append(_canonical_evaluation_metrics(record, common_sst))
    evaluations = canonical_evaluations
    harvester = _load_harvester_report(
        root,
        config,
        args.harvester,
        expected_captures={"train": train_capture, "eval": eval_capture},
        environment_sha256=producer_environment_sha256,
        conversion_attestation_sha256=conversion_attestation_sha256,
        storage_attestation_sha256=storage_attestation_sha256,
    )
    if harvester.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("benchmark and captures use different checkpoints")
    run_records = [
        json.loads(
            (_run_root(root, condition, seed) / "run.json").read_text(encoding="utf-8")
        )
        for condition, seed in pairs
    ]
    for (condition, seed), run_record, evaluation in zip(
        pairs, run_records, evaluations, strict=True
    ):
        _validate_authoritative_run_timing(
            run_record,
            config,
            condition,
            expected_tokens=config.train_tokens,
        )
        run_instance_id = _load_run_identity(
            _run_root(root, condition, seed) / "identity.json",
            _run_identity_expected(
                config,
                capture_manifest_sha256=train_capture.manifest_sha256,
                capture_checkpoint_sha256=checkpoint_sha,
                environment_sha256=producer_environment_sha256,
                condition=condition,
                seed=seed,
                storage_attestation_sha256=storage_attestation_sha256,
                checkpoint_payload_root=storage_layout["sae_checkpoint_root"],
            ),
        )
        if (
            run_record.get("config_sha256") != config.config_hash
            or run_record.get("condition") != condition
            or run_record.get("seed") != seed
            or run_record.get("authoritative") is not True
            or int(run_record.get("tokens_seen", -1)) != config.train_tokens
            or int(run_record.get("chip_count", -1)) != 16
            or not isinstance(run_record.get("health"), dict)
            or run_record["health"].get("passed") is not True
            or run_record.get("capture_manifest_sha256")
            != train_capture.manifest_sha256
            or run_record.get("capture_checkpoint_sha256") != checkpoint_sha
            or run_record.get("storage_attestation_sha256")
            != storage_attestation_sha256
            or run_record.get("checkpoint_payload_root")
            != storage_layout["sae_checkpoint_root"]
            or not re.fullmatch(
                r"[0-9a-f]{32}", str(run_record.get("run_instance_id", ""))
            )
            or run_record.get("run_instance_id") != run_instance_id
            or evaluation.get("run_instance_id") != run_record.get("run_instance_id")
            or not isinstance(run_record.get("environment"), dict)
            or run_record["environment"].get("sha256") != producer_environment_sha256
            or evaluation.get("training_environment_sha256")
            != run_record["environment"].get("sha256")
            or not math.isclose(
                float(run_record.get("chip_seconds", -1.0)),
                float(evaluation["chip_seconds"]),
                rel_tol=1e-12,
            )
        ):
            raise RuntimeError(
                f"incomplete/non-authoritative run for {condition}/seed-{seed}"
            )
    compile_seconds = sum(float(record["compile_seconds"]) for record in run_records)
    checkpoint_seconds = sum(
        float(record["checkpoint_seconds"]) for record in run_records
    )
    evaluation_seconds = sum(
        float(record.get("evaluation_seconds_excluded", 0.0)) for record in evaluations
    )
    report = build_efficiency_report(
        evaluations,
        harvester=harvester,
        compile_seconds=compile_seconds,
        checkpoint_seconds=checkpoint_seconds,
        evaluation_seconds=evaluation_seconds,
        nominal_candidate_exposure_percent=100.0 * config.candidate_exposure,
    )
    prefetch_by_run = {
        f"{condition}/seed-{seed}": {
            "seconds": float(
                record.get("capture_prefetch_wall_seconds_excluded", 0.0)
            ),
            "objects": int(record.get("capture_prefetch", {}).get("objects", 0)),
            "bytes": int(record.get("capture_prefetch", {}).get("bytes", 0)),
        }
        for (condition, seed), record in zip(pairs, run_records, strict=True)
    }
    report["capture_prefetch_excluded"] = {
        "by_run": prefetch_by_run,
        "sum_wall_seconds": sum(
            item["seconds"] for item in prefetch_by_run.values()
        ),
        "basis": (
            "generation-pinned GCS capture shards are cached before optimizer "
            "timing and reused across A/B/F"
        ),
    }
    storage_qualification = storage_payload["qualification"]
    report["storage_planes"] = {
        "attestation_sha256": storage_attestation_sha256,
        "control": storage_qualification["control"],
        "local_buffer": storage_qualification["local_buffer"],
        "gcs": storage_qualification["gcs"],
        "contract": storage_payload["contract"],
    }
    report["experiment"] = {
        "config_sha256": config.config_hash,
        "model_id": config.model_id,
        "maxtext_commit": config.maxtext_commit,
        "target_layer_one_indexed": config.target_layer_one_indexed,
        "capture_site": "routed_moe.post_feedforward_layernorm_2",
        "train_tokens": config.train_tokens,
        "eval_tokens": config.eval_tokens,
        "seeds": list(config.seeds),
        "conditions": list(config.conditions),
        "checkpoint_sha256": checkpoint_sha,
        "conversion_attestation_sha256": conversion_attestation_sha256,
        "storage_attestation_sha256": storage_attestation_sha256,
        "bulk_gcs_root": storage_layout["bulk_gcs_root"],
        "sae_checkpoint_root": storage_layout["sae_checkpoint_root"],
        "train_capture_manifest_sha256": train_capture.manifest_sha256,
        "eval_capture_manifest_sha256": eval_capture.manifest_sha256,
        "train_dataset_fingerprint": train_capture.manifest["metadata"][
            "dataset_fingerprint"
        ],
        "eval_dataset_fingerprint": eval_capture.manifest["metadata"][
            "dataset_fingerprint"
        ],
        "common_eval_sst": common_sst,
        "report_environment": current_environment,
        "training_environment_sha256_by_run": {
            f"{condition}/seed-{seed}": run_record["environment"]["sha256"]
            for (condition, seed), run_record in zip(pairs, run_records, strict=True)
        },
    }
    destination = Path(args.output) if args.output else root / "report.json"
    process_index, _ = _process_context()
    if process_index == 0:
        write_report_json(destination, report)
    _global_barrier("gemma4-efficiency-report")
    if not destination.exists():
        raise RuntimeError(
            f"efficiency report is not visible after barrier: {destination}"
        )
    if process_index == 0:
        print(render_headline(report))
    return 0


def _load_resume_run(
    root: Path,
    config: Gemma4Config,
    train_capture: ArtifactDataset,
    condition: str,
    seed: int,
    *,
    storage_attestation_sha256: str,
    checkpoint_payload_root: str,
) -> dict[str, Any] | None:
    """Validate existing run state before the production state machine skips it."""

    run_root = _run_root(root, condition, seed)
    path = run_root / "run.json"
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read existing run metadata: {path}") from error
    checkpoint_sha = str(train_capture.manifest["metadata"]["checkpoint_sha256"])
    tokens = int(record.get("tokens_seen", -1))
    try:
        _validate_authoritative_run_timing(record, config, condition)
    except RuntimeError as error:
        raise RuntimeError(f"existing run has invalid cost telemetry: {path}") from error
    current_environment = _coordinated_environment()
    expected_identity = _run_identity_expected(
        config,
        capture_manifest_sha256=train_capture.manifest_sha256,
        capture_checkpoint_sha256=checkpoint_sha,
        environment_sha256=current_environment["sha256"],
        condition=condition,
        seed=seed,
        storage_attestation_sha256=storage_attestation_sha256,
        checkpoint_payload_root=checkpoint_payload_root,
    )
    run_instance_id = _load_run_identity(run_root / "identity.json", expected_identity)
    valid_boundary = tokens == config.train_tokens or (
        0 < tokens < config.train_tokens and tokens % config.effective_batch_tokens == 0
    )
    if (
        record.get("config_sha256") != config.config_hash
        or record.get("condition") != condition
        or record.get("seed") != seed
        or record.get("authoritative") is not True
        or int(record.get("chip_count", -1)) != 16
        or record.get("capture_manifest_sha256") != train_capture.manifest_sha256
        or record.get("capture_checkpoint_sha256") != checkpoint_sha
        or record.get("storage_attestation_sha256")
        != storage_attestation_sha256
        or record.get("checkpoint_payload_root") != checkpoint_payload_root
        or not re.fullmatch(r"[0-9a-f]{32}", str(record.get("run_instance_id", "")))
        or record.get("run_instance_id") != run_instance_id
        or not isinstance(record.get("environment"), dict)
        or record["environment"].get("sha256") != current_environment.get("sha256")
        or not valid_boundary
    ):
        raise RuntimeError(f"existing run is not safely resumable: {path}")
    if tokens >= config.health_gate_tokens:
        health = record.get("health")
        if not isinstance(health, dict) or health.get("passed") is not True:
            raise RuntimeError(f"existing run failed or lacks its health gate: {path}")
    checkpoint = run_root / (
        "final.json" if tokens == config.train_tokens else "latest.json"
    )
    if not checkpoint.exists():
        raise RuntimeError(f"existing run lacks its resumable checkpoint: {checkpoint}")
    return record


def _existing_evaluation(
    root: Path,
    config: Gemma4Config,
    train_capture: ArtifactDataset,
    eval_capture: ArtifactDataset,
    condition: str,
    seed: int,
    run_record: dict[str, Any],
) -> dict[str, Any] | None:
    path = _evaluation_path(root, condition, seed)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        SeedEvaluation.from_value(record)
        record = _canonical_evaluation_metrics(
            record, eval_capture.sst_about(train_capture.mean)
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise RuntimeError(f"existing evaluation is invalid: {path}") from error
    checkpoint_sha = str(train_capture.manifest["metadata"]["checkpoint_sha256"])
    if (
        record.get("config_sha256") != config.config_hash
        or record.get("condition") != condition
        or record.get("seed") != seed
        or int(record.get("tokens", -1)) != config.eval_tokens
        or record.get("topk_valid") is not True
        or float(record.get("l0", -1.0)) != float(config.topk_features)
        or record.get("train_capture_manifest_sha256") != train_capture.manifest_sha256
        or record.get("eval_capture_manifest_sha256") != eval_capture.manifest_sha256
        or record.get("capture_checkpoint_sha256") != checkpoint_sha
        or record.get("run_instance_id") != run_record.get("run_instance_id")
        or not isinstance(record.get("environment"), dict)
        or record["environment"].get("sha256") != run_record["environment"]["sha256"]
        or record.get("training_environment_sha256")
        != run_record["environment"]["sha256"]
        or not math.isclose(
            float(record.get("chip_seconds", -1.0)),
            float(run_record["chip_seconds"]),
            rel_tol=1e-12,
        )
    ):
        raise RuntimeError(f"existing evaluation belongs to another experiment: {path}")
    return record


def _campaign_is_complete(
    path: Path,
    *,
    split: str,
    tokens: int,
    mesh_name: str,
    sequence_batch: int,
    config_sha256: str,
    checkpoint_sha256: str,
    conversion_attestation_sha256: str,
    dataset_fingerprint: str,
    environment_sha256: str,
    capture_manifest_sha256: str,
    benchmark: dict[str, Any],
) -> bool:
    if not path.exists():
        return False
    from moe_sae.harvest import CampaignSummary

    try:
        summary = CampaignSummary.from_mapping(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise RuntimeError(f"existing harvest campaign is invalid: {path}") from error
    if (
        summary.split != split
        or summary.phase != "production"
        or summary.captured_tokens != tokens
        or summary.mesh_name != mesh_name
        or summary.per_device_sequence_batch != sequence_batch
        or summary.global_chips != 16
        or summary.manifest_path is None
        or summary.config_sha256 != config_sha256
        or summary.checkpoint_sha256 != checkpoint_sha256
        or summary.conversion_attestation_sha256
        != conversion_attestation_sha256
        or summary.dataset_fingerprint != dataset_fingerprint
        or summary.environment_sha256 != environment_sha256
        or summary.capture_manifest_sha256 != capture_manifest_sha256
        or not math.isclose(
            float(summary.synthetic_goodput or -1.0),
            float(benchmark.get("persisted_tokens_per_second", -2.0)),
            rel_tol=1e-12,
        )
        or not math.isclose(
            float(summary.benchmark_useful_mfu or -1.0),
            float(benchmark.get("useful_mfu", -2.0)),
            rel_tol=1e-12,
        )
        or summary.dropped_routed_tokens
        != int(benchmark.get("dropped_routed_tokens", -1))
        or summary.recompilations != int(benchmark.get("recompilations", -1))
        or summary.instrumentation_equal
        is not bool(benchmark.get("instrumentation_equal", False))
        or (split == "train" and summary.canary_passed is not True)
    ):
        raise RuntimeError(f"existing harvest campaign is incompatible: {path}")
    return True


def _command_run(args: argparse.Namespace) -> int:
    """Execute the fail-closed production state machine with exact resume."""

    config = _config(args)
    config.validate_production()
    root = _artifact_root(config, args.artifact_root)
    _require_production_artifact_root(root)
    # `run` includes SAE training and evaluation. Check its distinct storage
    # capability before launching even the subprocess benchmark; operators who
    # only qualified the 24 GiB rolling capture buffer must use activation-farm.
    _require_sae_cache_qualification(_read_storage_attestation_payload(root))
    benchmark = root / "benchmark" / "selected.json"
    if not benchmark.exists():
        # The supervisor and all candidate processes must run before this
        # long-lived orchestration process initializes JAX distributed.
        benchmark_args = argparse.Namespace(**vars(args))
        benchmark_args.output = str(root / "benchmark")
        benchmark_args.steps = config.benchmark_steps
        _command_benchmark_harvest(benchmark_args)

    run_preflight(
        repo_root=REPO_ROOT,
        artifact_root=root,
        checkpoint_path=args.checkpoint,
        grain_train_files=args.grain_train,
        grain_eval_files=args.grain_eval,
        require_tpu=True,
    )
    staged = {
        "train": _verified_dataset_provenance(
            args,
            config,
            split="train",
            grain_files=args.grain_train,
            manifest_location=args.dataset_train_manifest,
        ),
        "eval": _verified_dataset_provenance(
            args,
            config,
            split="eval",
            grain_files=args.grain_eval,
            manifest_location=args.dataset_eval_manifest,
        ),
    }
    current_environment = _coordinated_environment()
    storage_payload = _load_storage_attestation_payload(
        root, config, environment=current_environment
    )
    _require_sae_cache_qualification(storage_payload)
    storage_layout = _storage_layout(storage_payload)
    storage_attestation_sha256 = str(storage_payload["attestation_sha256"])
    conversion_attestation_sha256 = _load_conversion_attestation(
        root, config, args.checkpoint_sha256
    )

    selected = _load_selected_benchmark(
        benchmark,
        config,
        args.checkpoint_sha256,
        conversion_attestation_sha256,
        storage_attestation_sha256,
    )
    selected_mesh = selected.get("mesh")
    mesh_name = (
        selected_mesh.get("name")
        if isinstance(selected_mesh, dict)
        else str(selected_mesh)
    )
    sequence_batch = int(selected["per_device_sequence_batch"])

    for split, files, expected in (
        ("train", args.grain_train, config.train_tokens),
        ("eval", args.grain_eval, config.eval_tokens),
    ):
        dataset_fingerprint = staged[split].fingerprint_sha256
        dataset_manifest = (
            args.dataset_train_manifest
            if split == "train"
            else args.dataset_eval_manifest
        )
        capture_root = _capture_root(root, split)
        sealed = (capture_root / "manifest.json").exists()
        if sealed:
            capture_dataset = _sealed_dataset(capture_root)
            _require_exact_capture(capture_dataset, expected, split)
            _require_capture_identity(
                capture_dataset,
                config,
                split,
                checkpoint_sha256=args.checkpoint_sha256,
                conversion_attestation_sha256=conversion_attestation_sha256,
                dataset_fingerprint=dataset_fingerprint,
                environment_sha256=current_environment["sha256"],
                storage_attestation_sha256=storage_attestation_sha256,
                object_prefix=join_gcs_uri(
                    storage_layout["bulk_gcs_root"],
                    "captures",
                    config.config_hash,
                    split,
                ),
            )
        if split == "train" and not sealed:
            canary_args = argparse.Namespace(**vars(args))
            canary_args.split = split
            canary_args.grain_files = files
            canary_args.dataset_manifest = dataset_manifest
            canary_args.canary = True
            canary_args.benchmark = str(benchmark)
            canary_args.mesh = mesh_name
            canary_args.per_device_sequence_batch = sequence_batch
            _command_harvest(canary_args)
            _release_accelerator_state("train-canary")

        campaign_path = root / "harvest" / f"{split}.json"
        campaign_complete = sealed and _campaign_is_complete(
            campaign_path,
            split=split,
            tokens=expected,
            mesh_name=mesh_name,
            sequence_batch=sequence_batch,
            config_sha256=config.config_hash,
            checkpoint_sha256=args.checkpoint_sha256,
            conversion_attestation_sha256=conversion_attestation_sha256,
            dataset_fingerprint=dataset_fingerprint,
            environment_sha256=current_environment["sha256"],
            capture_manifest_sha256=capture_dataset.manifest_sha256,
            benchmark=selected,
        )
        if not campaign_complete:
            harvest_args = argparse.Namespace(**vars(args))
            harvest_args.split = split
            harvest_args.grain_files = files
            harvest_args.dataset_manifest = dataset_manifest
            harvest_args.canary = False
            harvest_args.benchmark = str(benchmark)
            harvest_args.mesh = mesh_name
            harvest_args.per_device_sequence_batch = sequence_batch
            _command_harvest(harvest_args)
            _release_accelerator_state(f"{split}-production-harvest")

    train_capture = _sealed_dataset(_capture_root(root, "train"))
    eval_capture = _sealed_dataset(_capture_root(root, "eval"))
    _require_exact_capture(train_capture, config.train_tokens, "training")
    _require_exact_capture(eval_capture, config.eval_tokens, "held-out")
    _require_capture_identity(
        train_capture,
        config,
        "train",
        checkpoint_sha256=args.checkpoint_sha256,
        conversion_attestation_sha256=conversion_attestation_sha256,
        dataset_fingerprint=staged["train"].fingerprint_sha256,
        environment_sha256=current_environment["sha256"],
        storage_attestation_sha256=storage_attestation_sha256,
        object_prefix=join_gcs_uri(
            storage_layout["bulk_gcs_root"],
            "captures",
            config.config_hash,
            "train",
        ),
    )
    _require_capture_identity(
        eval_capture,
        config,
        "eval",
        checkpoint_sha256=args.checkpoint_sha256,
        conversion_attestation_sha256=conversion_attestation_sha256,
        dataset_fingerprint=staged["eval"].fingerprint_sha256,
        environment_sha256=current_environment["sha256"],
        storage_attestation_sha256=storage_attestation_sha256,
        object_prefix=join_gcs_uri(
            storage_layout["bulk_gcs_root"],
            "captures",
            config.config_hash,
            "eval",
        ),
    )

    def train_to(condition: str, seed: int, stop_after: int | None) -> dict[str, Any]:
        existing = _load_resume_run(
            root,
            config,
            train_capture,
            condition,
            seed,
            storage_attestation_sha256=storage_attestation_sha256,
            checkpoint_payload_root=storage_layout["sae_checkpoint_root"],
        )
        target = config.train_tokens if stop_after is None else stop_after
        if existing is not None and int(existing["tokens_seen"]) >= target:
            return existing
        train_args = argparse.Namespace(**vars(args))
        train_args.condition = condition
        train_args.seed = seed
        train_args.stop_after_tokens = stop_after
        train_args.a_capacity = None
        train_args.reference = False
        _command_train(train_args)
        _release_accelerator_state(f"train-{condition}-{seed}-{target}")
        completed = _load_resume_run(
            root,
            config,
            train_capture,
            condition,
            seed,
            storage_attestation_sha256=storage_attestation_sha256,
            checkpoint_payload_root=storage_layout["sae_checkpoint_root"],
        )
        if completed is None or int(completed["tokens_seen"]) < target:
            raise RuntimeError(
                f"training stage did not reach {condition}/seed-{seed} target"
            )
        return completed

    # No relative evaluation is loaded until every seed-1 condition passes 5M.
    for condition in config.conditions:
        train_to(condition, 1, config.health_gate_tokens)
    for condition in config.conditions:
        train_to(condition, 1, None)
    for seed in config.seeds[1:]:
        for condition in config.conditions:
            train_to(condition, seed, None)

    for condition in config.conditions:
        for seed in config.seeds:
            run_record = _load_resume_run(
                root,
                config,
                train_capture,
                condition,
                seed,
                storage_attestation_sha256=storage_attestation_sha256,
                checkpoint_payload_root=storage_layout["sae_checkpoint_root"],
            )
            if (
                run_record is None
                or int(run_record["tokens_seen"]) != config.train_tokens
            ):
                raise RuntimeError(f"missing complete run for {condition}/seed-{seed}")
            if (
                _existing_evaluation(
                    root,
                    config,
                    train_capture,
                    eval_capture,
                    condition,
                    seed,
                    run_record,
                )
                is None
            ):
                eval_args = argparse.Namespace(**vars(args))
                eval_args.reference = False
                eval_args.allow_reference_cost = False
                _command_eval_one(eval_args, config, condition, seed)
                _release_accelerator_state(f"eval-{condition}-{seed}")

    report_args = argparse.Namespace(**vars(args))
    report_args.harvester = str(benchmark)
    report_args.output = str(root / "report.json")
    return _command_report(report_args)


def _activation_common_child_arguments(args: argparse.Namespace) -> list[str]:
    arguments = [
        "--config",
        str(Path(args.config).expanduser().resolve()),
        "--artifact-root",
        str(Path(args.artifact_root).expanduser().resolve()),
        "--checkpoint",
        str(args.checkpoint),
        "--checkpoint-sha256",
        str(args.checkpoint_sha256),
        "--grain-train",
        str(args.grain_train),
        "--grain-eval",
        str(args.grain_eval),
        "--dataset-train-manifest",
        str(args.dataset_train_manifest),
        "--dataset-eval-manifest",
        str(args.dataset_eval_manifest),
    ]
    if bool(getattr(args, "reuse_verified_dataset", False)):
        arguments.append("--reuse-verified-dataset")
    return arguments


def _activation_child_command(
    args: argparse.Namespace, phase: str
) -> list[str]:
    base = [sys.executable, "-m", "moe_sae.cli"]
    common = _activation_common_child_arguments(args)
    if phase == "preflight":
        # Preflight does not accept or need the claimed checkpoint digest.
        without_digest = common[:]
        index = without_digest.index("--checkpoint-sha256")
        del without_digest[index : index + 2]
        return [*base, "preflight", *without_digest, "--json"]
    if phase == "gate":
        return [sys.executable, "-m", "moe_sae.activation_worker", *common]
    if phase == "benchmark":
        config = _config(args)
        return [
            *base,
            "benchmark-harvest",
            "--config",
            common[1],
            "--artifact-root",
            common[3],
            "--checkpoint",
            str(args.checkpoint),
            "--checkpoint-sha256",
            str(args.checkpoint_sha256),
            "--steps",
            str(config.benchmark_steps),
        ]
    if phase not in {"canary", "train", "eval"}:
        raise ValueError(f"unknown activation subprocess phase {phase!r}")
    split = "train" if phase in {"canary", "train"} else "eval"
    grain = args.grain_train if split == "train" else args.grain_eval
    manifest = (
        args.dataset_train_manifest
        if split == "train"
        else args.dataset_eval_manifest
    )
    command = [
        *base,
        "harvest",
        "--config",
        common[1],
        "--artifact-root",
        common[3],
        "--split",
        split,
        "--checkpoint",
        str(args.checkpoint),
        "--checkpoint-sha256",
        str(args.checkpoint_sha256),
        "--grain-files",
        str(grain),
        "--dataset-manifest",
        str(manifest),
    ]
    if phase == "canary":
        command.append("--canary")
    if bool(getattr(args, "reuse_verified_dataset", False)):
        command.append("--reuse-verified-dataset")
    return command


def _activation_gate_command(
    args: argparse.Namespace, phase: str
) -> list[str]:
    return [
        *_activation_child_command(args, "gate"),
        "--phase",
        phase,
    ]


def _activation_final_record(root: Path, phase: str) -> Path:
    if phase == "benchmark":
        return root / "benchmark" / "selected.json"
    name = {
        "canary": "train-canary.json",
        "train": "train.json",
        "eval": "eval.json",
    }[phase]
    return root / "harvest" / name


def _matching_harvest_controls(
    control: Any, phase: str
) -> list[tuple[Path, str]]:
    if phase not in {"canary", "train", "eval"}:
        return []
    result: list[tuple[Path, str]] = []
    for item in control.status()["harvest_controls"]:
        if item.get("valid") is not True or item.get("phase") != phase:
            continue
        split = str(item["split"])
        result.append(
            (
                control.artifact_root / "captures" / split,
                str(item["campaign_id"]),
            )
        )
    return result


def _mirror_activation_abort_to_harvest(
    control: Any, phase: str, request: Mapping[str, Any]
) -> list[str]:
    """Mirror one coarse request into every matching in-phase safe-stop marker."""

    from moe_sae.harvest_control import request_campaign_abort

    mirrored: list[str] = []
    for capture_root, campaign_id in _matching_harvest_controls(control, phase):
        request_campaign_abort(
            capture_root,
            campaign_id,
            reason=str(request["reason"]),
            requested_by="activation-operator",
        )
        mirrored.append(campaign_id)
    return mirrored


def _run_activation_subprocess(
    command: Sequence[str],
    *,
    label: str,
    control: Any | None = None,
    phase: str | None = None,
    lease_fd: int | None = None,
    supervisor: Any | None = None,
) -> None:
    """Run one fresh child; cooperative aborts never signal a live collective."""

    if control is not None and supervisor is not None:
        control.supervisor_rendezvous(supervisor, label)
    child_environment = os.environ.copy()
    if lease_fd is not None:
        child_environment["MOE_SAE_ACTIVATION_LEASE_FD"] = str(lease_fd)
    process = subprocess.Popen(
        list(command),
        pass_fds=(() if lease_fd is None else (lease_fd,)),
        start_new_session=True,
        env=child_environment,
    )
    if control is not None and supervisor is not None:
        control.supervisor_heartbeat(supervisor, "running", step=label)
    while True:
        returncode = process.poll()
        if returncode is not None:
            break
        if control is not None and supervisor is not None:
            control.supervisor_heartbeat(supervisor, "running", step=label)
            try:
                control.require_supervisor_peers_healthy(supervisor, label)
            except Exception:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.time() + _ACTIVATION_CHILD_KILL_GRACE_SECONDS
                while process.poll() is None and time.time() < deadline:
                    time.sleep(0.25)
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
                raise
        if control is not None and phase in {"canary", "train", "eval"}:
            request = control.pending_abort()
            if request is not None:
                _mirror_activation_abort_to_harvest(control, str(phase), request)
        time.sleep(2.0)
    if control is not None and supervisor is not None:
        local_state = (
            "succeeded"
            if returncode == 0
            else (
                "aborted"
                if phase in {"canary", "train", "eval"}
                and control.pending_abort() is not None
                else "failed"
            )
        )
        outcome = control.supervisor_wait_completion(
            supervisor, label, local_state=local_state
        )
        if outcome == "aborted":
            raise RuntimeError(f"{label} subprocess stopped cooperatively")
    if returncode != 0:
        raise RuntimeError(f"{label} subprocess exited with status {returncode}")


def _activation_gate_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Collectively validate one final phase artifact without rerunning it."""

    from moe_sae.harvest import (
        CampaignSummary,
        _validate_campaign_summary_identity,
        capture_metadata,
    )

    require_v4_32_runtime()
    config = _config(args)
    config.validate_production()
    root = _artifact_root(config, args.artifact_root).resolve()
    _require_production_artifact_root(root)
    _verify_checkpoint_inventory(args)
    environment = _coordinated_environment()
    storage_payload = _load_storage_attestation_payload(
        root, config, environment=environment
    )
    storage = _storage_layout(storage_payload)
    storage_sha256 = str(storage_payload["attestation_sha256"])
    conversion_sha256 = _load_conversion_attestation(
        root, config, args.checkpoint_sha256
    )
    benchmark_path = root / "benchmark" / "selected.json"
    selected = _load_selected_benchmark(
        benchmark_path,
        config,
        args.checkpoint_sha256,
        conversion_sha256,
        storage_sha256,
    )
    selected_mesh = selected.get("mesh")
    mesh_name = (
        selected_mesh.get("name")
        if isinstance(selected_mesh, Mapping)
        else str(selected_mesh)
    )
    sequence_batch = int(selected["per_device_sequence_batch"])
    if args.phase == "benchmark":
        return {
            "phase": "benchmark",
            "path": str(benchmark_path),
            "mesh": mesh_name,
            "per_device_sequence_batch": sequence_batch,
            "useful_mfu": float(selected["useful_mfu"]),
        }

    split = "train" if args.phase in {"canary", "train"} else "eval"
    grain_files = args.grain_train if split == "train" else args.grain_eval
    dataset_manifest = (
        args.dataset_train_manifest
        if split == "train"
        else args.dataset_eval_manifest
    )
    provenance = _verified_dataset_provenance(
        args,
        config,
        split=split,
        grain_files=grain_files,
        manifest_location=dataset_manifest,
    )
    metadata = capture_metadata(
        config,
        checkpoint_sha256=args.checkpoint_sha256,
        conversion_attestation_sha256=conversion_sha256,
        dataset_fingerprint=provenance.fingerprint_sha256,
        split=split,
        storage_attestation_sha256=storage_sha256,
        object_prefix=join_gcs_uri(
            storage["bulk_gcs_root"],
            "captures",
            config.config_hash,
            split,
        ),
        environment=environment,
    )
    summary_path = _activation_final_record(root, args.phase)
    try:
        raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"missing/invalid final {args.phase} record: {summary_path}"
        ) from error
    if not isinstance(raw_summary, Mapping):
        raise RuntimeError(f"final {args.phase} record is not an object")
    summary = CampaignSummary.from_mapping(raw_summary)
    stop_after = config.canary_tokens if args.phase == "canary" else (
        config.train_tokens if split == "train" else config.eval_tokens
    )
    internal_path = (
        _capture_root(root, split)
        / "telemetry"
        / f"campaign-stop-{stop_after:012d}.json"
    )
    try:
        internal_raw = json.loads(internal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"missing/invalid internal {args.phase} campaign: {internal_path}"
        ) from error
    if not isinstance(internal_raw, Mapping):
        raise RuntimeError(f"internal {args.phase} campaign is not an object")
    internal = CampaignSummary.from_mapping(internal_raw)
    if internal != summary:
        raise RuntimeError(
            f"final and internal {args.phase} campaign records disagree"
        )

    _validate_campaign_summary_identity(
        summary,
        split=split,
        stop_after_tokens=stop_after,
        mesh=_mesh_by_name(mesh_name),
        per_device_sequence_batch=sequence_batch,
        capture_metadata=metadata.to_dict(),
        capture_manifest_sha256=(
            None if args.phase == "canary" else summary.capture_manifest_sha256
        ),
        benchmark={
            "synthetic_goodput": selected["persisted_tokens_per_second"],
            "useful_mfu": selected["useful_mfu"],
            "dropped_routed_tokens": selected["dropped_routed_tokens"],
            "recompilations": selected["recompilations"],
            "instrumentation_equal": selected["instrumentation_equal"],
        },
    )
    if args.phase == "canary":
        if (
            summary.phase != "canary"
            or summary.total_tokens != config.train_tokens
            or summary.captured_tokens != config.canary_tokens
            or summary.canary_passed is not True
            or summary.capture_manifest_sha256 is not None
            or summary.manifest_path is not None
            or summary.process_count != 4
            or summary.global_chips != 16
        ):
            raise RuntimeError("final canary did not pass the exact production gate")
    else:
        capture = _sealed_dataset(_capture_root(root, split))
        expected_tokens = (
            config.train_tokens if split == "train" else config.eval_tokens
        )
        _require_exact_capture(capture, expected_tokens, split)
        _require_capture_identity(
            capture,
            config,
            split,
            checkpoint_sha256=args.checkpoint_sha256,
            conversion_attestation_sha256=conversion_sha256,
            dataset_fingerprint=provenance.fingerprint_sha256,
            environment_sha256=environment["sha256"],
            storage_attestation_sha256=storage_sha256,
            object_prefix=metadata.object_prefix,
        )
        if not _campaign_is_complete(
            summary_path,
            split=split,
            tokens=expected_tokens,
            mesh_name=mesh_name,
            sequence_batch=sequence_batch,
            config_sha256=config.config_hash,
            checkpoint_sha256=args.checkpoint_sha256,
            conversion_attestation_sha256=conversion_sha256,
            dataset_fingerprint=provenance.fingerprint_sha256,
            environment_sha256=environment["sha256"],
            capture_manifest_sha256=capture.manifest_sha256,
            benchmark=selected,
        ):
            raise RuntimeError(f"{args.phase} campaign is not complete")
    return {
        "phase": args.phase,
        "path": str(summary_path),
        "captured_tokens": summary.captured_tokens,
        "campaign_environment_sha256": summary.environment_sha256,
        "capture_manifest_sha256": summary.capture_manifest_sha256,
    }


def _command_activation_gate(args: argparse.Namespace) -> int:
    payload = _activation_gate_payload(args)
    if _process_context()[0] == 0:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _activation_spec(args: argparse.Namespace, config: Gemma4Config) -> Any:
    from moe_sae.activation_operator import ActivationRunSpec

    return ActivationRunSpec(
        config_sha256=config.config_hash,
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=str(args.checkpoint_sha256),
        grain_train=str(args.grain_train),
        grain_eval=str(args.grain_eval),
        dataset_train_manifest=str(args.dataset_train_manifest),
        dataset_eval_manifest=str(args.dataset_eval_manifest),
    )


def _raise_if_activation_aborted(control: Any, phase: str) -> None:
    from moe_sae.activation_operator import ActivationAbortRequested

    request = control.pending_abort()
    if request is None:
        return
    state = control.load_state()
    running = (
        []
        if state is None
        else [
            candidate
            for candidate in ("benchmark", "canary", "train", "eval")
            if state["phases"][candidate]["status"] == "running"
        ]
    )
    if len(running) > 1:
        raise RuntimeError("activation state has multiple running phases")
    acknowledged_phase = running[0] if running else phase
    control.acknowledge_abort(acknowledged_phase)
    raise ActivationAbortRequested(
        f"activation farm stopped at the {acknowledged_phase} boundary: "
        f"{request['reason']}; "
        "clear the abort explicitly before resuming"
    )


def _require_activation_phase_commit(
    state: Mapping[str, Any], phase: str, attempt_id: str
) -> None:
    record = state["phases"][phase]
    if (
        record.get("status") != "passed"
        or record.get("attempt_id") != attempt_id
    ):
        raise RuntimeError(
            f"activation phase {phase} did not commit globally for attempt "
            f"{attempt_id}"
        )


def _command_activation_farm(args: argparse.Namespace) -> int:
    """Run only benchmark/canary/train/eval in fresh, gated subprocesses."""

    from moe_sae.activation_operator import (
        ACTIVATION_PHASES,
        ActivationAbortRequested,
        ActivationControl,
    )

    config = _config(args)
    config.validate_production()
    root = _artifact_root(config, args.artifact_root).resolve()
    _require_production_artifact_root(root)
    args.artifact_root = str(root)
    control = ActivationControl(root)
    activation_spec = _activation_spec(args, config)
    control.begin(activation_spec)
    operation_sha256 = hashlib.sha256(
        json.dumps(
            {
                "spec_sha256": activation_spec.sha256,
                "phases": ACTIVATION_PHASES,
                "skip_preflight": bool(getattr(args, "skip_preflight", False)),
                "reuse_verified_dataset": bool(
                    getattr(args, "reuse_verified_dataset", False)
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with control.supervisor_lease(
        operation_sha256=operation_sha256
    ) as supervisor:
        lease_fd = supervisor.descriptor
        _raise_if_activation_aborted(control, "benchmark")
        if bool(getattr(args, "skip_preflight", False)):
            print(
                json.dumps(
                    {
                        "event": "activation_preflight_bypassed",
                        "run_id": supervisor.run_id,
                        "operation_sha256": operation_sha256,
                        "reuse_verified_dataset": bool(
                            getattr(args, "reuse_verified_dataset", False)
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            _run_activation_subprocess(
                _activation_child_command(args, "preflight"),
                label="preflight",
                control=control,
                lease_fd=lease_fd,
                supervisor=supervisor,
            )

        for phase in ACTIVATION_PHASES:
            _raise_if_activation_aborted(control, phase)
            final_record = _activation_final_record(root, phase)
            current = control.load_state()
            if (
                current is not None
                and current["phases"][phase]["status"] == "passed"
                and not final_record.exists()
            ):
                raise RuntimeError(
                    f"activation state says {phase} passed but its final record is missing"
                )
            if final_record.exists():
                phase_state = control.mark_running(phase)
                attempt_id = str(phase_state["phases"][phase]["attempt_id"])
                try:
                    _run_activation_subprocess(
                        _activation_gate_command(args, phase),
                        label=f"existing {phase} gate",
                        control=control,
                        lease_fd=lease_fd,
                        supervisor=supervisor,
                    )
                except Exception as error:
                    control.mark_failed(phase, error, attempt_id=attempt_id)
                    raise
                committed = control.mark_passed(
                    phase,
                    {"validated_existing": True, "path": str(final_record)},
                    attempt_id=attempt_id,
                )
                _require_activation_phase_commit(committed, phase, attempt_id)
                continue
            phase_state = control.mark_running(phase)
            attempt_id = str(phase_state["phases"][phase]["attempt_id"])
            phase_subprocess_completed = False
            try:
                _run_activation_subprocess(
                    _activation_child_command(args, phase),
                    label=phase,
                    control=control,
                    phase=phase,
                    lease_fd=lease_fd,
                    supervisor=supervisor,
                )
                phase_subprocess_completed = True
                _run_activation_subprocess(
                    _activation_gate_command(args, phase),
                    label=f"{phase} completion gate",
                    control=control,
                    lease_fd=lease_fd,
                    supervisor=supervisor,
                )
            except ActivationAbortRequested:
                raise
            except Exception as error:
                request = control.pending_abort()
                if request is not None and not phase_subprocess_completed:
                    control.acknowledge_abort(phase)
                    raise ActivationAbortRequested(
                        f"activation farm stopped safely during {phase}: "
                        f"{request['reason']}"
                    ) from error
                control.mark_failed(phase, error, attempt_id=attempt_id)
                raise
            committed = control.mark_passed(
                phase,
                {"validated_existing": False, "path": str(final_record)},
                attempt_id=attempt_id,
            )
            _require_activation_phase_commit(committed, phase, attempt_id)
            next_index = ACTIVATION_PHASES.index(phase) + 1
            boundary_phase = (
                ACTIVATION_PHASES[next_index]
                if next_index < len(ACTIVATION_PHASES)
                else phase
            )
            _raise_if_activation_aborted(control, boundary_phase)
    print(json.dumps(control.status(), indent=2, sort_keys=True))
    return 0


def _command_activation_status(args: argparse.Namespace) -> int:
    from moe_sae.activation_operator import ActivationControl

    config = _config(args)
    root = _artifact_root(config, args.artifact_root).resolve()
    _require_production_artifact_root(root)
    print(json.dumps(ActivationControl(root).status(), indent=2, sort_keys=True))
    return 0


def _command_activation_abort(args: argparse.Namespace) -> int:
    from moe_sae.activation_operator import ACTIVATION_PHASES, ActivationControl
    from moe_sae.harvest_control import clear_campaign_abort

    config = _config(args)
    root = _artifact_root(config, args.artifact_root).resolve()
    _require_production_artifact_root(root)
    control = ActivationControl(root)
    state = control.load_state()
    if state is None:
        raise RuntimeError("no activation campaign exists at this artifact root")
    if args.clear:
        observed = control.status()["harvest_controls"]
        boundary_cleared = control.clear_abort()
        cleared_campaigns: list[str] = []
        for item in observed:
            if item.get("valid") is not True or item.get("abort_request") is None:
                continue
            split = str(item["split"])
            campaign_id = str(item["campaign_id"])
            if clear_campaign_abort(
                root / "captures" / split, campaign_id
            ):
                cleared_campaigns.append(campaign_id)
        payload = {
            "boundary_abort_cleared": boundary_cleared,
            "harvest_campaign_aborts_cleared": cleared_campaigns,
        }
    else:
        request = control.request_abort(args.reason)
        running = [
            phase
            for phase in ACTIVATION_PHASES
            if state["phases"][phase]["status"] == "running"
        ]
        mirrored = (
            _mirror_activation_abort_to_harvest(control, running[0], request)
            if len(running) == 1
            else []
        )
        payload = {
            "boundary_abort_request": request,
            "in_phase_harvest_campaigns": mirrored,
            "note": (
                "The request is cooperative. Do not kill one controller; wait "
                "for all activation-farm commands to return."
            ),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--artifact-root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moe-sae")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser(
        "stage-c4", help="offline-pack the bounded immutable AllenAI C4 source"
    )
    _add_common(stage)
    stage.add_argument("--split", choices=("train", "validation"), required=True)
    stage.add_argument("--source-cache-dir", required=True)
    stage.add_argument("--output-directory")
    stage.add_argument("--records-per-shard", type=int, default=4_096)
    stage.add_argument("--publish-gcs")
    stage.add_argument("--target-tokens", type=int)
    stage.add_argument(
        "--local-smoke",
        action="store_true",
        help="allow a reduced token target for local validation",
    )
    stage.set_defaults(handler=_command_stage_c4)

    storage = subparsers.add_parser(
        "qualify-storage", help="qualify control POSIX, local buffers, and GCS"
    )
    _add_common(storage)
    storage.add_argument("--local-buffer-root", required=True)
    storage.add_argument("--bulk-gcs-root", required=True)
    storage.add_argument("--sae-checkpoint-root", required=True)
    storage.add_argument(
        "--local", action="store_true", help="single-process smoke qualification"
    )
    storage.add_argument("--local-object-store-root")
    storage.add_argument("--minimum-control-free-bytes", type=int)
    storage.add_argument("--minimum-local-free-bytes", type=int)
    storage.add_argument("--probe-bytes", type=int)
    storage.add_argument("--block-bytes", type=int)
    storage.add_argument("--minimum-bytes-per-second", type=float)
    storage.set_defaults(handler=_command_qualify_storage)

    preflight = subparsers.add_parser(
        "preflight", help="fail-closed environment validation"
    )
    _add_common(preflight)
    preflight.add_argument("--checkpoint", required=True)
    preflight.add_argument("--grain-train", required=True)
    preflight.add_argument("--grain-eval", required=True)
    preflight.add_argument("--dataset-train-manifest", required=True)
    preflight.add_argument("--dataset-eval-manifest", required=True)
    preflight.add_argument(
        "--local", action="store_true", help="skip TPU/MaxText checks"
    )
    preflight.add_argument(
        "--reuse-verified-dataset",
        action="store_true",
        help="reuse a prior semantic ArrayRecord validation while rechecking manifest objects",
    )
    preflight.add_argument("--json", action="store_true")
    preflight.set_defaults(handler=_command_preflight)

    convert = subparsers.add_parser(
        "convert-checkpoint", help="convert and parity-check base/PT weights"
    )
    _add_common(convert)
    convert.add_argument("--maxtext-root", required=True)
    convert.add_argument("--hf-cache-dir", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--checkpoint-items")
    convert.add_argument(
        "--resume-validation",
        action="store_true",
        help=(
            "validate and seal the canonical existing output/0/items after an "
            "interrupted post-conversion parity step; never reruns conversion"
        ),
    )
    convert.add_argument("--dry-run", action="store_true")
    convert.set_defaults(handler=_command_convert_checkpoint)

    benchmark = subparsers.add_parser(
        "benchmark-harvest", help="run the required 3x3 mesh sweep"
    )
    _add_common(benchmark)
    benchmark.add_argument("--checkpoint", required=True)
    benchmark.add_argument("--checkpoint-sha256", type=_sha256_argument, required=True)
    benchmark.add_argument("--output")
    benchmark.add_argument("--steps", type=int, default=200)
    benchmark.set_defaults(handler=_command_benchmark_harvest)

    harvest = subparsers.add_parser(
        "harvest", help="capture one deterministic C4 split"
    )
    _add_common(harvest)
    harvest.add_argument("--split", choices=("train", "eval"), required=True)
    harvest.add_argument("--checkpoint", required=True)
    harvest.add_argument("--checkpoint-sha256", type=_sha256_argument, required=True)
    harvest.add_argument("--grain-files", required=True)
    harvest.add_argument("--dataset-manifest", required=True)
    harvest.add_argument("--benchmark")
    harvest.add_argument("--mesh", choices=tuple(x.name for x in HARVEST_MESHES))
    harvest.add_argument(
        "--per-device-sequence-batch", type=int, choices=PER_DEVICE_SEQUENCE_BATCHES
    )
    harvest.add_argument("--canary", action="store_true")
    harvest.add_argument(
        "--reuse-verified-dataset",
        action="store_true",
        help="reuse a prior semantic ArrayRecord validation while rechecking manifest objects",
    )
    harvest.set_defaults(handler=_command_harvest)

    activation = subparsers.add_parser(
        "activation-farm",
        help="run only benchmark, canary, train capture, and eval capture",
    )
    _add_common(activation)
    activation.add_argument("--checkpoint", required=True)
    activation.add_argument(
        "--checkpoint-sha256", type=_sha256_argument, required=True
    )
    activation.add_argument("--grain-train", required=True)
    activation.add_argument("--grain-eval", required=True)
    activation.add_argument("--dataset-train-manifest", required=True)
    activation.add_argument("--dataset-eval-manifest", required=True)
    activation.add_argument(
        "--skip-preflight",
        action="store_true",
        help="operator-authorized resume that bypasses activation-farm's fresh preflight",
    )
    activation.add_argument(
        "--reuse-verified-dataset",
        action="store_true",
        help="reuse prior semantic ArrayRecord validation in capture subprocesses",
    )
    activation.set_defaults(handler=_command_activation_farm)

    activation_status = subparsers.add_parser(
        "activation-status", help="inspect activation-farm state without JAX"
    )
    _add_common(activation_status)
    activation_status.set_defaults(handler=_command_activation_status)

    activation_abort = subparsers.add_parser(
        "activation-abort", help="request or clear a cooperative activation stop"
    )
    _add_common(activation_abort)
    activation_abort_action = activation_abort.add_mutually_exclusive_group(
        required=True
    )
    activation_abort_action.add_argument("--reason")
    activation_abort_action.add_argument("--clear", action="store_true")
    activation_abort.set_defaults(handler=_command_activation_abort)

    train_parser = subparsers.add_parser("train", help="train or resume one A/B/F seed")
    _add_common(train_parser)
    train_parser.add_argument("--condition", choices=("A", "B", "F"), required=True)
    train_parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    train_parser.add_argument("--stop-after-tokens", type=int)
    train_parser.add_argument("--a-capacity", type=int)
    train_parser.add_argument(
        "--reference",
        action="store_true",
        help="CPU correctness path; timing is non-authoritative",
    )
    train_parser.set_defaults(handler=_command_train)

    eval_parser = subparsers.add_parser(
        "eval", help="evaluate final checkpoint(s) on common held-out SST"
    )
    _add_common(eval_parser)
    eval_group = eval_parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument("--all", action="store_true")
    eval_group.add_argument("--condition", choices=("A", "B", "F"))
    eval_parser.add_argument("--seed", type=int, choices=(1, 2, 3))
    eval_parser.add_argument("--reference", action="store_true")
    eval_parser.add_argument(
        "--allow-reference-cost", action="store_true", help=argparse.SUPPRESS
    )
    eval_parser.set_defaults(handler=_command_eval)

    report = subparsers.add_parser(
        "report", help="build the nine-run efficiency report"
    )
    _add_common(report)
    report.add_argument("--harvester")
    report.add_argument("--output")
    report.set_defaults(handler=_command_report)

    run = subparsers.add_parser(
        "run", help="execute the complete gated production sequence"
    )
    _add_common(run)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--checkpoint-sha256", type=_sha256_argument, required=True)
    run.add_argument("--grain-train", required=True)
    run.add_argument("--grain-eval", required=True)
    run.add_argument("--dataset-train-manifest", required=True)
    run.add_argument("--dataset-eval-manifest", required=True)
    run.set_defaults(handler=_command_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "eval" and not args.all and args.seed is None:
        parser.error("eval --condition requires --seed")
    try:
        return int(args.handler(args))
    except (PreflightError, RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
