"""Versioned capture artifacts for the Gemma 4 router-aggregate pipeline.

The artifact store is deliberately small and fail-closed:

* every shard is a content-addressed safetensors file;
* a shard commits one immutable, half-open input range;
* host manifests are updated atomically under a store-wide range lock;
* the top-level manifest is published only after every shard is verified;
* publication seals the store against further writes; and
* readers accept only :data:`SCHEMA_VERSION` -- legacy captures are never
  guessed or migrated implicitly.

``activation`` and ``router_weights`` are logical BF16 arrays.  Safetensors'
NumPy adapter does not expose BF16 consistently across NumPy versions, so the
wire representation is the exact BF16 bit-pattern stored as ``uint16``.  The
manifest and per-file metadata record the logical dtype explicitly.
"""

from __future__ import annotations

import contextlib
import dataclasses
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from moe_sae.data import (
    ALLENAI_C4_REPO_ID,
    ALLENAI_C4_REVISION,
    ALLENAI_C4_SOURCE_FILES,
    DEFAULT_PER_DEVICE_SEQUENCE_BATCHES,
    DEFAULT_PROCESS_COUNT,
    GEMMA4_TOKENIZER_INVENTORY_SHA256,
    GEMMA4_TOKENIZER_SOURCE_FILES,
    PACKED_DATASET_SCHEMA,
    PACKED_RECORD_FORMAT,
    verify_packed_source,
)
from moe_sae.model import CaptureBatch
from moe_sae.object_store import (
    GCSObjectStore,
    ImmutableObjectStore,
    ObjectStoreError,
    ShardCache,
    fingerprint_file,
    join_gcs_uri,
    object_identity_from_mapping,
    parse_gcs_uri,
)


SCHEMA_VERSION = "gemma4-router-aggregate-v1"
MANIFEST_NAME = "manifest.json"
MAX_SHARD_TOKENS = 131_072
HIDDEN_SIZE = 2_816
NUM_EXPERTS = 128
TOP_K_EXPERTS = 8
TARGET_LAYER_ONE_INDEXED = 15
DEFAULT_CAPTURE_SITE = "routed_moe.post_feedforward_layernorm_2"
STAGED_DATASET_SCHEMA_VERSION = PACKED_DATASET_SCHEMA

_STATS_COUNT = "__stats_count"
_STATS_SUM = "__stats_sum"
_STATS_SQUARED_NORM = "__stats_squared_norm"
_DATA_KEYS = (
    "activation",
    "expert_ids",
    "router_weights",
    "sequence_id",
    "position",
)
_FILE_KEYS = frozenset((*_DATA_KEYS, _STATS_COUNT, _STATS_SUM, _STATS_SQUARED_NORM))
_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ArtifactError(RuntimeError):
    """Base class for artifact contract violations."""


class SchemaError(ArtifactError):
    """The data does not satisfy the Gemma 4 aggregate schema."""


class LegacySchemaError(SchemaError):
    """A missing or old schema was presented to the v1 reader."""


class IntegrityError(ArtifactError):
    """A hash, count, statistic, or manifest invariant failed."""


class RangeConflictError(ArtifactError):
    """Two immutable input ranges overlap or reuse an id inconsistently."""


@dataclasses.dataclass(frozen=True, slots=True)
class StagedDatasetProvenance:
    """Verified identity of one deterministic packed ArrayRecord split."""

    fingerprint_sha256: str
    split: str
    files: tuple[str, ...]
    scientific_tokens: int
    manifest: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _gcs_object_under(prefix: str, uri: str) -> bool:
    """Return whether ``uri`` is lexically below ``prefix`` by bucket/name."""

    try:
        bucket, name = parse_gcs_uri(uri)
        remainder = prefix.removeprefix("gs://").rstrip("/")
        prefix_bucket, separator, prefix_name = remainder.partition("/")
    except ValueError:
        return False
    if not prefix.startswith("gs://") or bucket != prefix_bucket:
        return False
    return bool(name) if not separator else name.startswith(f"{prefix_name}/")


def _read_json_location(location: str) -> Mapping[str, Any]:
    if location.startswith("gs://"):
        try:
            completed = subprocess.run(
                ["gcloud", "storage", "cat", location],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            raise IntegrityError(
                "cannot read staged dataset manifest from GCS"
            ) from exc
        raw = completed.stdout
    else:
        try:
            raw = Path(location).expanduser().read_bytes()
        except OSError as exc:
            raise IntegrityError(
                f"cannot read staged dataset manifest: {location}"
            ) from exc
    if location.startswith("gs://"):
        filename = location.rsplit("/", 1)[-1]
        match = re.fullmatch(r"([0-9a-f]{64})\.json", filename)
        if match is None:
            raise IntegrityError(
                "staged GCS manifest URI is not content-addressed"
            )
        if hashlib.sha256(raw).hexdigest() != match.group(1):
            raise IntegrityError(
                "staged GCS manifest bytes differ from their content address"
            )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError("staged dataset manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise IntegrityError("staged dataset manifest must be a JSON object")
    return value


def _resolve_staged_files(specification: str) -> tuple[str, ...]:
    if specification.startswith("gs://"):
        try:
            completed = subprocess.run(
                ["gcloud", "storage", "ls", specification],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            raise IntegrityError("cannot resolve staged GCS ArrayRecord files") from exc
        files = tuple(
            sorted(
                line.strip() for line in completed.stdout.splitlines() if line.strip()
            )
        )
    else:
        files = tuple(
            sorted(
                str(Path(item).expanduser().resolve())
                for item in glob.glob(specification)
            )
        )
    if not files:
        raise IntegrityError("staged ArrayRecord specification matched no files")
    return files


def _normalize_declared_uri(uri: str, manifest_location: str) -> str:
    if uri.startswith("gs://"):
        return uri
    if manifest_location.startswith("gs://"):
        raise IntegrityError("a GCS dataset manifest must use absolute gs:// file URIs")
    path = Path(uri).expanduser()
    if not path.is_absolute():
        path = Path(manifest_location).expanduser().resolve().parent / path
    return str(path.resolve())


def _hash_location(location: str, chunk_bytes: int = 8 << 20) -> tuple[str, int]:
    if not location.startswith("gs://"):
        path = Path(location)
        try:
            return _sha256_file(path, chunk_bytes), path.stat().st_size
        except OSError as exc:
            raise IntegrityError(
                f"cannot hash staged ArrayRecord file: {location}"
            ) from exc

    try:
        process = subprocess.Popen(
            ["gcloud", "storage", "cat", location],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise IntegrityError("cannot start GCS staged-file verifier") from exc
    if process.stdout is None:  # pragma: no cover - guaranteed by PIPE
        process.kill()
        raise IntegrityError("GCS staged-file verifier has no output stream")
    digest = hashlib.sha256()
    size = 0
    try:
        while chunk := process.stdout.read(chunk_bytes):
            digest.update(chunk)
            size += len(chunk)
    finally:
        process.stdout.close()
    if process.wait() != 0:
        raise IntegrityError("cannot stream a staged GCS ArrayRecord file")
    return digest.hexdigest(), size


def verify_staged_dataset_manifest(
    manifest_location: str,
    grain_files: str,
    *,
    expected_split: str,
    expected_tokens: int,
    tokenizer_id: str,
    tokenizer_revision: str,
    sequence_length: int,
    verify_file_hashes: bool = True,
    verify_arrayrecord_count: bool = False,
    object_store: ImmutableObjectStore | None = None,
) -> StagedDatasetProvenance:
    """Verify a C4 staging manifest and derive its content fingerprint.

    The fingerprint is never supplied independently by an operator.  It hashes
    the complete canonical manifest, whose ordered file inventory is checked
    against the actual Grain specification.  Production additionally hashes
    every ArrayRecord object before any model execution.
    """

    value = dict(_read_json_location(manifest_location))
    required = {
        "schema_version",
        "dataset",
        "source",
        "tokenizer",
        "sequence_length",
        "record_format",
        "packing",
        "consumer",
        "target_scientific_tokens",
        "source_examples",
        "scientific_tokens",
        "record_count",
        "lane_scientific_tokens",
        "files",
    }
    if set(value) != required:
        raise IntegrityError(
            "staged dataset manifest fields differ from the packed v2 contract"
        )
    if value.get("schema_version") != STAGED_DATASET_SCHEMA_VERSION:
        raise IntegrityError("unsupported staged dataset manifest schema")
    expected_dataset = {
        "name": ALLENAI_C4_REPO_ID,
        "config": "en",
        "revision": ALLENAI_C4_REVISION,
        "split": expected_split,
    }
    if value.get("dataset") != expected_dataset:
        raise IntegrityError("staged dataset is not the required C4 English split")
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise IntegrityError("staged dataset source provenance is incomplete")
    expected_source = {
        "provider": "huggingface_hub",
        "repo_type": "dataset",
        "repo_id": ALLENAI_C4_REPO_ID,
        "revision": ALLENAI_C4_REVISION,
        "files": [
            specification.as_dict()
            for specification in ALLENAI_C4_SOURCE_FILES[expected_split]
        ],
    }
    if dict(source) != expected_source:
        raise IntegrityError(
            "staged dataset is not the bounded immutable AllenAI C4 source"
        )
    tokenizer = value.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or set(tokenizer) != {
        "id",
        "revision",
        "repository_inventory_sha256",
        "files",
        "padding_id",
        "eos_id",
        "bos_id",
    }:
        raise IntegrityError("staged tokenizer provenance is incomplete")
    if (
        tokenizer.get("id") != tokenizer_id
        or tokenizer.get("revision") != tokenizer_revision
        or tokenizer.get("repository_inventory_sha256")
        != GEMMA4_TOKENIZER_INVENTORY_SHA256
        or tokenizer.get("files")
        != [entry.as_dict() for entry in GEMMA4_TOKENIZER_SOURCE_FILES]
        or (
            tokenizer.get("padding_id"),
            tokenizer.get("eos_id"),
            tokenizer.get("bos_id"),
        )
        != (0, 1, 2)
    ):
        raise IntegrityError("staged dataset uses a different tokenizer")
    if value.get("sequence_length") != sequence_length:
        raise IntegrityError("staged dataset uses a different packed sequence length")
    if value.get("record_format") != {
        "name": PACKED_RECORD_FORMAT,
        "arrayrecord_options": "group_size:1",
        "token_ids": "int32[1024]",
        "positions": "uint16[1024]",
        "segment_ids": "uint16[1024]",
    }:
        raise IntegrityError("staged ArrayRecord payload format is incompatible")
    packing = value.get("packing")
    if not isinstance(packing, Mapping) or dict(packing) != {
        "strategy": "ordered_next_fit_offline",
        "exclude_from_scientific_count": ["padding", "bos", "eos"],
        "special_token_ids": {"padding": 0, "eos": 1, "bos": 2},
    }:
        raise IntegrityError("staged dataset packing/counting contract is incompatible")
    if value.get("consumer") != {
        "process_count": DEFAULT_PROCESS_COUNT,
        "host_sharding": "record_index_mod_process_count",
        "per_device_sequence_batches": list(DEFAULT_PER_DEVICE_SEQUENCE_BATCHES),
    }:
        raise IntegrityError("staged dataset has an incompatible TPU consumption layout")
    if value.get("target_scientific_tokens") != expected_tokens:
        raise IntegrityError("staged dataset was built for a different token target")
    if (
        isinstance(value.get("source_examples"), bool)
        or not isinstance(value.get("source_examples"), int)
        or int(value["source_examples"]) <= 0
    ):
        raise IntegrityError("staged dataset has an invalid source-example count")
    raw_files = value.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise IntegrityError("staged dataset manifest must inventory at least one file")

    resolved = _resolve_staged_files(grain_files)
    is_remote = manifest_location.startswith("gs://")
    if is_remote and verify_file_hashes and object_store is None:
        object_store = GCSObjectStore()
    declared: list[str] = []
    declared_records = 0
    scientific_tokens = 0
    expected_first_record = 0
    remote_identities = []
    for entry in raw_files:
        expected_entry_fields = {
            "uri",
            "sha256",
            "size_bytes",
            "first_record_index",
            "record_count",
            "scientific_tokens",
        }
        if is_remote:
            expected_entry_fields.add("object")
        if not isinstance(entry, Mapping) or set(entry) != expected_entry_fields:
            raise IntegrityError("staged file inventory entry has incompatible fields")
        uri = _normalize_declared_uri(str(entry.get("uri", "")), manifest_location)
        digest = str(entry.get("sha256", ""))
        size = entry.get("size_bytes")
        records = entry.get("record_count")
        tokens = entry.get("scientific_tokens")
        first_record = entry.get("first_record_index")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise IntegrityError("staged file inventory has an invalid SHA-256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or isinstance(records, bool)
            or not isinstance(records, int)
            or records <= 0
            or isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens < 0
            or isinstance(first_record, bool)
            or not isinstance(first_record, int)
            or first_record != expected_first_record
        ):
            raise IntegrityError("staged file inventory has invalid counts")
        declared.append(uri)
        declared_records += records
        expected_first_record += records
        scientific_tokens += tokens
        identity = None
        if is_remote:
            try:
                identity = object_identity_from_mapping(entry.get("object"))
            except ObjectStoreError as exc:
                raise IntegrityError(
                    "staged GCS object identity is malformed"
                ) from exc
            if (
                identity.uri != uri
                or identity.sha256 != digest
                or identity.size_bytes != size
            ):
                raise IntegrityError(
                    "staged GCS object identity differs from its file inventory"
                )
            remote_identities.append(identity)
        if verify_file_hashes:
            if identity is not None:
                assert object_store is not None
                try:
                    object_store.verify(identity)
                except Exception as exc:
                    raise IntegrityError(
                        f"staged GCS object generation changed: {uri}"
                    ) from exc
            actual_digest, actual_size = _hash_location(uri)
            if actual_digest != digest or actual_size != size:
                raise IntegrityError(f"staged ArrayRecord hash/size mismatch: {uri}")
            if identity is not None:
                try:
                    object_store.verify(identity)
                except Exception as exc:
                    raise IntegrityError(
                        f"staged GCS object changed during verification: {uri}"
                    ) from exc
    if tuple(declared) != resolved:
        raise IntegrityError(
            "ordered staged manifest inventory differs from --grain-files"
        )
    lane_counts = value.get("lane_scientific_tokens")
    if (
        not isinstance(lane_counts, list)
        or len(lane_counts) != DEFAULT_PROCESS_COUNT
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in lane_counts
        )
        or sum(lane_counts) != scientific_tokens
    ):
        raise IntegrityError("staged per-controller scientific-token counts are invalid")
    for process_index, count in enumerate(lane_counts):
        quota = expected_tokens // DEFAULT_PROCESS_COUNT + (
            1 if process_index < expected_tokens % DEFAULT_PROCESS_COUNT else 0
        )
        if count < quota:
            raise IntegrityError(
                "a staged controller lane cannot reach its scientific-token quota"
            )
    if (
        value.get("record_count") != declared_records
        or value.get("scientific_tokens") != scientific_tokens
        or declared_records % 64
    ):
        raise IntegrityError("staged aggregate record/token counts are inconsistent")
    if verify_arrayrecord_count:
        try:
            actual_records, actual_scientific, actual_lanes = verify_packed_source(
                resolved, sequence_length=sequence_length
            )
        except (
            AttributeError,
            ImportError,
            ModuleNotFoundError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise IntegrityError(
                "cannot independently inspect the staged ArrayRecord record count"
            ) from exc
        if actual_records != declared_records:
            raise IntegrityError(
                "staged ArrayRecord count differs from the manifest: "
                f"{actual_records:,} != {declared_records:,}"
            )
        if actual_scientific != scientific_tokens or actual_lanes != tuple(lane_counts):
            raise IntegrityError(
                "decoded packed ArrayRecords differ from manifest scientific counts"
            )
    if scientific_tokens < expected_tokens:
        raise IntegrityError(
            f"staged split contains only {scientific_tokens:,} scientific tokens; "
            f"need at least {expected_tokens:,}"
        )
    if verify_file_hashes and remote_identities:
        assert object_store is not None
        for identity in remote_identities:
            try:
                object_store.verify(identity)
            except Exception as exc:
                raise IntegrityError(
                    "a staged GCS object changed before dataset verification completed: "
                    f"{identity.uri}"
                ) from exc
    fingerprint = _sha256_json(value)
    return StagedDatasetProvenance(
        fingerprint_sha256=fingerprint,
        split=expected_split,
        files=resolved,
        scientific_tokens=scientific_tokens,
        manifest=value,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _fsync_dir(path: Path) -> None:
    """Best-effort directory fsync; unsupported filesystems may reject it."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def _store_lock(root: Path, host_id: str | None = None) -> Iterator[None]:
    """Serialize one host manifest, or the final cross-host publication.

    Capture hosts own disjoint token ranges and distinct shard prefixes.  A
    global per-shard lock would unnecessarily serialize four high-throughput
    writers, so ordinary commits use a host-local lock.  Final publication
    uses the store-wide lock and revalidates every range after the global
    harvest barrier.
    """
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / (".ranges.lock" if host_id is None else f".host-{host_id}.lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl_module: Any | None = None
    try:
        try:
            import fcntl as fcntl_module

            fcntl_module.flock(fd, fcntl_module.LOCK_EX)
        except ImportError as exc:  # pragma: no cover - TPU/Linux and macOS have it
            raise ArtifactError("artifact locking requires a POSIX platform") from exc
        yield
    finally:
        try:
            if fcntl_module is not None:
                fcntl_module.flock(fd, fcntl_module.LOCK_UN)
        finally:
            os.close(fd)


def float32_to_bf16_bits(values: Any) -> np.ndarray:
    """Round FP32 values to BF16 (round-to-nearest-even) and return U16 bits."""
    x = np.asarray(values, dtype=np.float32)
    bits = x.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_float32(values: Any) -> np.ndarray:
    """Expand uint16 BF16 bit-patterns to exactly represented FP32 values."""
    bits = np.asarray(values)
    if bits.dtype != np.uint16:
        raise SchemaError(f"BF16 wire values must be uint16, got {bits.dtype}")
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _as_bf16_bits(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.uint16:
        return np.ascontiguousarray(arr)
    # JAX/ml_dtypes BF16 has two-byte items and the dtype string "bfloat16".
    if arr.dtype.itemsize == 2 and str(arr.dtype).lower() in {"bfloat16", "bf16"}:
        return np.ascontiguousarray(arr.view(np.uint16))
    if np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all():
            raise SchemaError(f"{name} contains non-finite values")
        return np.ascontiguousarray(float32_to_bf16_bits(arr))
    raise SchemaError(
        f"{name} must be floating-point or a uint16 BF16 wire representation, got {arr.dtype}"
    )


def _integer_wire_array(
    name: str, value: Any, dtype: np.dtype[Any], minimum: int, maximum: int
) -> np.ndarray:
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.integer):
        raise SchemaError(f"{name} must have an integer dtype, got {arr.dtype}")
    if arr.size and (np.any(arr < minimum) or np.any(arr > maximum)):
        raise SchemaError(f"{name} must be in [{minimum},{maximum}]")
    return np.ascontiguousarray(arr.astype(dtype, copy=False))


def _normalize_batch(
    batch: CaptureBatch, *, max_tokens: int = MAX_SHARD_TOKENS
) -> CaptureBatch:
    """Normalize the shared model CaptureBatch to exact on-disk dtypes."""
    activation = _as_bf16_bits("activation", batch.activation)
    expert_ids = _integer_wire_array(
        "expert_ids", batch.expert_ids, np.dtype(np.uint8), 0, NUM_EXPERTS - 1
    )
    router_weights = _as_bf16_bits("router_weights", batch.router_weights)
    sequence_id = _integer_wire_array(
        "sequence_id",
        batch.sequence_id,
        np.dtype(np.int64),
        np.iinfo(np.int64).min,
        np.iinfo(np.int64).max,
    )
    position = _integer_wire_array(
        "position", batch.position, np.dtype(np.uint16), 0, 1_023
    )

    if activation.ndim != 2 or activation.shape[1] != HIDDEN_SIZE:
        raise SchemaError(
            f"activation must have shape [N,{HIDDEN_SIZE}], got {activation.shape}"
        )
    n = int(activation.shape[0])
    if not 0 < n <= max_tokens:
        raise SchemaError(f"shard token count must be in [1,{max_tokens}], got {n}")
    expected = {
        "expert_ids": (n, TOP_K_EXPERTS),
        "router_weights": (n, TOP_K_EXPERTS),
        "sequence_id": (n,),
        "position": (n,),
    }
    arrays = {
        "expert_ids": expert_ids,
        "router_weights": router_weights,
        "sequence_id": sequence_id,
        "position": position,
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise SchemaError(
                f"{name} must have shape {shape}, got {arrays[name].shape}"
            )
    sorted_ids = np.sort(expert_ids, axis=1)
    if np.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
        raise SchemaError("each token must have eight unique expert_ids")

    weights = bf16_bits_to_float32(router_weights)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise SchemaError("router_weights must be finite and nonnegative")
    if not np.allclose(weights.sum(axis=1, dtype=np.float32), 1.0, atol=0.02, rtol=0):
        raise SchemaError("router_weights must sum to one for every token")

    return CaptureBatch(activation, expert_ids, router_weights, sequence_id, position)


def _token_count(batch: CaptureBatch) -> int:
    return int(np.asarray(batch.activation).shape[0])


def _batch_tensors(batch: CaptureBatch) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(batch, name)) for name in _DATA_KEYS}


def _decode_batch(batch: CaptureBatch) -> CaptureBatch:
    """Decode logical BF16 fields to FP32 while retaining exact integer fields."""
    return CaptureBatch(
        activation=bf16_bits_to_float32(batch.activation),
        expert_ids=batch.expert_ids,
        router_weights=bf16_bits_to_float32(batch.router_weights),
        sequence_id=batch.sequence_id,
        position=batch.position,
    )


@dataclasses.dataclass(frozen=True)
class CaptureMetadata:
    """Scientific identity common to every shard in one artifact store."""

    checkpoint_sha256: str
    maxtext_commit: str
    dataset_fingerprint: str
    split: str
    config_sha256: str
    environment_sha256: str
    conversion_attestation_sha256: str
    storage_attestation_sha256: str | None = None
    object_prefix: str | None = None
    model_id: str = "google/gemma-4-26B-A4B"
    dataset: str = f"{ALLENAI_C4_REPO_ID}/en@{ALLENAI_C4_REVISION}"
    tokenizer_id: str = "google/gemma-4-26B-A4B"
    target_layer_one_indexed: int = TARGET_LAYER_ONE_INDEXED
    capture_site: str = DEFAULT_CAPTURE_SITE
    sequence_length: int = 1_024

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out.update(
            {
                "hidden_size": HIDDEN_SIZE,
                "num_experts": NUM_EXPERTS,
                "top_k_experts": TOP_K_EXPERTS,
                "activation_dtype": "bfloat16",
                "activation_storage_dtype": "uint16",
                "router_weights_dtype": "bfloat16",
                "router_weights_storage_dtype": "uint16",
            }
        )
        if self.target_layer_one_indexed != TARGET_LAYER_ONE_INDEXED:
            raise SchemaError(
                f"v1 captures target one-indexed layer {TARGET_LAYER_ONE_INDEXED}"
            )
        for name in (
            "checkpoint_sha256",
            "config_sha256",
            "environment_sha256",
            "conversion_attestation_sha256",
        ):
            value = getattr(self, name)
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise SchemaError(f"{name} must be a lowercase SHA-256 hex digest")
        if self.storage_attestation_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.storage_attestation_sha256
        ):
            raise SchemaError(
                "storage_attestation_sha256 must be a lowercase SHA-256 hex digest"
            )
        if self.object_prefix is not None:
            try:
                # Validate the bucket and prefix without pretending that the
                # prefix itself is an object.
                join_gcs_uri(self.object_prefix, "identity-probe")
            except ValueError as error:
                raise SchemaError("object_prefix must be a safe gs:// prefix") from error
        if not self.maxtext_commit:
            raise SchemaError("maxtext_commit is required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.dataset_fingerprint):
            raise SchemaError("dataset_fingerprint must be a lowercase SHA-256")
        if not self.split:
            raise SchemaError("split is required")
        return out


def _metadata_dict(metadata: CaptureMetadata | Mapping[str, Any]) -> dict[str, Any]:
    value = (
        metadata.to_dict() if isinstance(metadata, CaptureMetadata) else dict(metadata)
    )
    try:
        value = json.loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise SchemaError("capture metadata must contain only JSON values") from exc
    required = {
        "model_id",
        "checkpoint_sha256",
        "config_sha256",
        "environment_sha256",
        "conversion_attestation_sha256",
        "maxtext_commit",
        "dataset_fingerprint",
        "dataset",
        "split",
        "tokenizer_id",
        "target_layer_one_indexed",
        "capture_site",
        "sequence_length",
        "hidden_size",
        "num_experts",
        "top_k_experts",
        "activation_dtype",
        "activation_storage_dtype",
        "router_weights_dtype",
        "router_weights_storage_dtype",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise SchemaError(f"capture metadata missing fields: {', '.join(missing)}")
    for name in (
        "checkpoint_sha256",
        "config_sha256",
        "environment_sha256",
        "conversion_attestation_sha256",
    ):
        digest = value.get(name)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SchemaError(f"metadata {name} must be a lowercase SHA-256 hex digest")
    storage_digest = value.get("storage_attestation_sha256")
    if storage_digest is not None and (
        not isinstance(storage_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", storage_digest)
    ):
        raise SchemaError(
            "metadata storage_attestation_sha256 must be a lowercase SHA-256"
        )
    object_prefix = value.get("object_prefix")
    if object_prefix is not None:
        try:
            join_gcs_uri(str(object_prefix), "identity-probe")
        except ValueError as error:
            raise SchemaError("metadata object_prefix must be a safe gs:// prefix") from error
    fixed = {
        "model_id": "google/gemma-4-26B-A4B",
        "maxtext_commit": "48c7da3e28e8f57fe36ddf09d00eec53072a2ea0",
        "dataset": f"{ALLENAI_C4_REPO_ID}/en@{ALLENAI_C4_REVISION}",
        "tokenizer_id": "google/gemma-4-26B-A4B",
        "target_layer_one_indexed": TARGET_LAYER_ONE_INDEXED,
        "capture_site": DEFAULT_CAPTURE_SITE,
        "sequence_length": 1_024,
        "hidden_size": HIDDEN_SIZE,
        "num_experts": NUM_EXPERTS,
        "top_k_experts": TOP_K_EXPERTS,
        "activation_dtype": "bfloat16",
        "activation_storage_dtype": "uint16",
        "router_weights_dtype": "bfloat16",
        "router_weights_storage_dtype": "uint16",
    }
    for key, expected in fixed.items():
        if value.get(key) != expected:
            raise SchemaError(
                f"metadata {key} must be {expected!r}, got {value.get(key)!r}"
            )
    if not re.fullmatch(r"[0-9a-f]{64}", str(value.get("dataset_fingerprint", ""))):
        raise SchemaError("metadata dataset_fingerprint must be a lowercase SHA-256")
    if not str(value.get("split", "")).strip():
        raise SchemaError("metadata split must be nonempty")
    return value


def _batch_stats(
    batch: CaptureBatch, *, chunk_tokens: int = 4_096
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact FP64 sufficient statistics with bounded host memory."""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    activation_bits = np.asarray(batch.activation)
    count = np.asarray([_token_count(batch)], dtype=np.float64)
    total = np.zeros(HIDDEN_SIZE, dtype=np.float64)
    squared_value = 0.0
    for start in range(0, _token_count(batch), chunk_tokens):
        decoded = bf16_bits_to_float32(activation_bits[start : start + chunk_tokens])
        if not np.isfinite(decoded).all():
            raise SchemaError("activation contains non-finite values")
        total += decoded.sum(axis=0, dtype=np.float64)
        # A BF16 value has eight significant bits, so its square is represented
        # exactly in FP32; only the reduction needs the required FP64 precision.
        squared_value += float(np.square(decoded).sum(dtype=np.float64))
    squared_norm = np.asarray([squared_value], dtype=np.float64)
    return count, total, squared_norm


def _host_manifest_path(root: Path, host_id: str) -> Path:
    return root / "hosts" / f"host-{host_id}.json"


def _upload_receipt_path(
    root: Path, host_id: str, range_id: str, start: int, end: int
) -> Path:
    identity = _sha256_json(
        {
            "host_id": host_id,
            "range_id": range_id,
            "range_start": int(start),
            "range_end": int(end),
        }
    )
    return root / "receipts" / f"host-{host_id}" / f"{identity}.json"


def _validate_upload_receipt(
    root: Path,
    entry: Mapping[str, Any],
    *,
    metadata_sha256: str,
) -> dict[str, Any]:
    """Prove a remote ledger entry is exactly its durable self-hashed receipt."""

    path = _upload_receipt_path(
        root,
        str(entry.get("host_id", "")),
        str(entry.get("range_id", "")),
        int(entry.get("range_start", -1)),
        int(entry.get("range_end", -1)),
    )
    receipt = _read_json(path)
    claimed = receipt.get("receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema_version") != "gemma4-capture-upload-receipt-v1"
        or receipt.get("metadata_sha256") != metadata_sha256
        or receipt.get("host_id") != entry.get("host_id")
        or receipt.get("range_id") != entry.get("range_id")
        or receipt.get("range_start") != entry.get("range_start")
        or receipt.get("range_end") != entry.get("range_end")
        or not isinstance(claimed, str)
        or claimed != _sha256_json(unsigned)
        or receipt.get("entry") != dict(entry)
    ):
        raise IntegrityError(
            f"remote shard ledger differs from its upload receipt: {path}"
        )
    return receipt


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read valid JSON from {path}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"manifest must be an object: {path}")
    return value


def _host_manifests(root: Path) -> list[dict[str, Any]]:
    host_dir = root / "hosts"
    if not host_dir.exists():
        return []
    manifests: list[dict[str, Any]] = []
    for path in sorted(host_dir.glob("host-*.json")):
        manifest = _read_json(path)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise LegacySchemaError(
                f"host manifest schema {manifest.get('schema_version')!r} is not {SCHEMA_VERSION}"
            )
        expected_host = path.stem.removeprefix("host-")
        if manifest.get("host_id") != expected_host:
            raise IntegrityError(f"host manifest identity does not match {path.name}")
        manifests.append(manifest)
    return manifests


def _all_entries(root: Path) -> list[dict[str, Any]]:
    return [entry for host in _host_manifests(root) for entry in host.get("shards", [])]


def _find_range_conflict(
    entries: list[dict[str, Any]], range_id: str, start: int, end: int
) -> dict[str, Any] | None:
    for entry in entries:
        if entry["range_id"] == range_id:
            if entry["range_start"] == start and entry["range_end"] == end:
                return entry
            raise RangeConflictError(
                f"range id {range_id!r} is already bound to "
                f"[{entry['range_start']},{entry['range_end']})"
            )
        if max(start, entry["range_start"]) < min(end, entry["range_end"]):
            raise RangeConflictError(
                f"input range [{start},{end}) overlaps {entry['range_id']!r} "
                f"[{entry['range_start']},{entry['range_end']})"
            )
    return None


class ArtifactWriter:
    """Commit immutable capture ranges for one host.

    ``write_range`` is idempotent by ``(range_id, range_start, range_end)``.
    If that identity is already committed, the existing shard entry is
    returned without comparing or rewriting payload bytes; callers should use
    :meth:`has_range` to skip model execution during resume.
    """

    def __init__(
        self,
        root: str | Path,
        host_id: str,
        metadata: CaptureMetadata | Mapping[str, Any],
        *,
        max_tokens: int = MAX_SHARD_TOKENS,
        local_buffer_root: str | Path | None = None,
        object_prefix: str | None = None,
        object_store: ImmutableObjectStore | None = None,
    ) -> None:
        if not _HOST_RE.fullmatch(host_id):
            raise ArtifactError(f"unsafe host_id {host_id!r}")
        if not 0 < max_tokens <= MAX_SHARD_TOKENS:
            raise ArtifactError(f"max_tokens must be in [1,{MAX_SHARD_TOKENS}]")
        self.root = Path(root)
        self.host_id = host_id
        self.metadata = _metadata_dict(metadata)
        self.metadata_sha256 = _sha256_json(self.metadata)
        self.max_tokens = int(max_tokens)
        if (object_prefix is None) != (object_store is None):
            raise ArtifactError("object_prefix and object_store must be supplied together")
        self.object_prefix = None if object_prefix is None else object_prefix.rstrip("/")
        self.object_store = object_store
        if self.object_prefix is None:
            self.staging_dir = self.root / ".staging" / f"host-{host_id}"
            (self.root / "shards" / f"host-{host_id}").mkdir(
                parents=True, exist_ok=True
            )
        else:
            if local_buffer_root is None:
                raise ArtifactError("remote shard publication requires a host-local buffer root")
            self.staging_dir = (
                Path(local_buffer_root).expanduser().resolve()
                / "capture-shards"
                / self.metadata_sha256
                / f"host-{host_id}"
            )
            (self.root / "receipts" / f"host-{host_id}").mkdir(
                parents=True, exist_ok=True
            )
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "hosts").mkdir(parents=True, exist_ok=True)
        with _store_lock(self.root, self.host_id):
            if (self.root / MANIFEST_NAME).exists():
                raise ArtifactError("artifact store is sealed by its final manifest")
            existing = _host_manifest_path(self.root, self.host_id)
            if existing.exists():
                manifest = _read_json(existing)
                if manifest.get("schema_version") != SCHEMA_VERSION:
                    raise LegacySchemaError("cannot resume a non-v1 host manifest")
                if manifest.get("metadata_sha256") != self.metadata_sha256:
                    raise SchemaError(
                        "capture metadata changed while resuming this host"
                    )

    @staticmethod
    def _validate_range(range_id: str, start: int, end: int) -> None:
        if not isinstance(range_id, str) or not range_id.strip():
            raise RangeConflictError("range_id must be a nonempty string")
        if isinstance(start, bool) or isinstance(end, bool):
            raise RangeConflictError("range bounds must be integers")
        if int(start) != start or int(end) != end or start < 0 or end <= start:
            raise RangeConflictError(f"invalid half-open input range [{start},{end})")

    def has_range(self, range_id: str, start: int, end: int) -> bool:
        """Return true for an exact committed identity; reject any conflict."""
        self._validate_range(range_id, start, end)
        existing: dict[str, Any] | None = None
        with _store_lock(self.root, self.host_id):
            path = _host_manifest_path(self.root, self.host_id)
            entries = _read_json(path).get("shards", []) if path.exists() else []
            existing = _find_range_conflict(entries, range_id, start, end)
        if existing is not None:
            if self.object_store is not None:
                receipt = self._load_receipt(range_id, start, end)
                if receipt is None or receipt["entry"] != existing:
                    raise IntegrityError(
                        "committed range differs from its durable upload receipt"
                    )
            return True
        # A process may have died after the object and receipt became durable
        # but before the host ledger advanced.  Verify and adopt that receipt
        # so deterministic resume never reruns the expensive model prefix.
        receipt = self._load_receipt(range_id, start, end)
        if receipt is None:
            return False
        with _store_lock(self.root, self.host_id):
            self._commit_entry_locked(dict(receipt["entry"]))
        return True

    def _receipt_path(self, range_id: str, start: int, end: int) -> Path:
        return _upload_receipt_path(
            self.root, self.host_id, range_id, start, end
        )

    def _load_receipt(
        self, range_id: str, start: int, end: int
    ) -> dict[str, Any] | None:
        if self.object_store is None:
            return None
        path = self._receipt_path(range_id, start, end)
        if not path.exists():
            return None
        receipt = _read_json(path)
        claimed = receipt.get("receipt_sha256")
        unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        entry = receipt.get("entry")
        if (
            receipt.get("schema_version") != "gemma4-capture-upload-receipt-v1"
            or receipt.get("metadata_sha256") != self.metadata_sha256
            or receipt.get("host_id") != self.host_id
            or receipt.get("range_id") != range_id
            or receipt.get("range_start") != start
            or receipt.get("range_end") != end
            or not isinstance(claimed, str)
            or claimed != _sha256_json(unsigned)
            or not isinstance(entry, dict)
            or "object" not in entry
        ):
            raise IntegrityError(f"invalid upload receipt {path}")
        _validate_entry_shape(entry)
        if (
            entry["host_id"] != self.host_id
            or entry["range_id"] != range_id
            or entry["range_start"] != start
            or entry["range_end"] != end
        ):
            raise IntegrityError(f"upload receipt range identity changed: {path}")
        identity = object_identity_from_mapping(entry["object"])
        self.object_store.verify(identity)
        return receipt

    def _commit_entry_locked(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Append one already-durable entry while holding this host's lock."""

        if (self.root / MANIFEST_NAME).exists():
            raise ArtifactError("artifact store was sealed before range commit")
        host_path = _host_manifest_path(self.root, self.host_id)
        host_manifest = _read_json(host_path) if host_path.exists() else None
        entries = host_manifest.get("shards", []) if host_manifest is not None else []
        existing = _find_range_conflict(
            entries,
            str(entry["range_id"]),
            int(entry["range_start"]),
            int(entry["range_end"]),
        )
        if existing is not None:
            if existing != entry:
                raise IntegrityError("committed range differs from its durable receipt")
            return existing
        if host_manifest is None:
            host_manifest = {
                "schema_version": SCHEMA_VERSION,
                "host_id": self.host_id,
                "metadata_sha256": self.metadata_sha256,
                "metadata": self.metadata,
                "complete": False,
                "shards": [],
            }
        elif host_manifest.get("metadata_sha256") != self.metadata_sha256:
            raise SchemaError("capture metadata changed during host commit")
        elif host_manifest.get("complete") is True:
            raise ArtifactError(
                f"host {self.host_id!r} is finalized; no new ranges are allowed"
            )
        host_manifest["shards"].append(entry)
        host_manifest["shards"].sort(
            key=lambda value: (value["range_start"], value["range_id"])
        )
        host_manifest["updated_at"] = _utc_now()
        _atomic_json(host_path, host_manifest)
        return entry

    def write_range(
        self,
        range_id: str,
        start: int,
        end: int,
        batch: CaptureBatch,
    ) -> dict[str, Any]:
        """Atomically commit one content-addressed safetensor shard."""
        self._validate_range(range_id, start, end)
        # Fast resume/finalization check before quantization, statistics, and
        # serialization.  The second locked check below closes the race.
        with _store_lock(self.root, self.host_id):
            if (self.root / MANIFEST_NAME).exists():
                raise ArtifactError("artifact store is sealed by its final manifest")
            host_path = _host_manifest_path(self.root, self.host_id)
            host_entries = (
                _read_json(host_path).get("shards", []) if host_path.exists() else []
            )
            existing = _find_range_conflict(host_entries, range_id, start, end)
            if existing is not None:
                return existing
            if host_path.exists() and _read_json(host_path).get("complete") is True:
                raise ArtifactError(
                    f"host {self.host_id!r} is finalized; no new ranges are allowed"
                )

        batch = _normalize_batch(batch, max_tokens=self.max_tokens)
        count, total, squared_norm = _batch_stats(batch)
        tensors = _batch_tensors(batch)
        tensors.update(
            {
                _STATS_COUNT: count,
                _STATS_SUM: total,
                _STATS_SQUARED_NORM: squared_norm,
            }
        )

        fd, tmp_name = tempfile.mkstemp(
            prefix="shard-", suffix=".safetensors", dir=self.staging_dir
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            save_file(
                tensors,
                str(tmp),
                metadata={
                    "schema_version": SCHEMA_VERSION,
                    "range_id": range_id,
                    "range_start": str(start),
                    "range_end": str(end),
                    "host_id": self.host_id,
                    "activation_logical_dtype": "bfloat16",
                    "router_weights_logical_dtype": "bfloat16",
                },
            )
            with tmp.open("rb") as handle:
                os.fsync(handle.fileno())
            source_fingerprint = fingerprint_file(tmp)
            digest = source_fingerprint.sha256
            relative = Path("shards") / f"host-{self.host_id}" / f"{digest}.safetensors"
            destination = self.root / relative
            entry = {
                "range_id": range_id,
                "range_start": int(start),
                "range_end": int(end),
                "host_id": self.host_id,
                "path": relative.as_posix(),
                "sha256": digest,
                "tokens": _token_count(batch),
                "sum": total.tolist(),
                "squared_norm": float(squared_norm[0]),
            }
            if self.object_prefix is not None:
                assert self.object_store is not None
                uri = join_gcs_uri(
                    self.object_prefix,
                    f"host-{self.host_id}",
                    f"{digest}.safetensors",
                )
                entry["object"] = self.object_store.put_if_absent(
                    tmp,
                    uri,
                    sha256=digest,
                    source_fingerprint=source_fingerprint,
                ).as_dict()
                receipt: dict[str, Any] = {
                    "schema_version": "gemma4-capture-upload-receipt-v1",
                    "metadata_sha256": self.metadata_sha256,
                    "host_id": self.host_id,
                    "range_id": range_id,
                    "range_start": int(start),
                    "range_end": int(end),
                    "entry": entry,
                }
                receipt["receipt_sha256"] = _sha256_json(receipt)
                _atomic_json(self._receipt_path(range_id, start, end), receipt)

            with _store_lock(self.root, self.host_id):
                if (self.root / MANIFEST_NAME).exists():
                    raise ArtifactError("artifact store was sealed before range commit")
                host_path = _host_manifest_path(self.root, self.host_id)
                host_entries = (
                    _read_json(host_path).get("shards", [])
                    if host_path.exists()
                    else []
                )
                existing = _find_range_conflict(host_entries, range_id, start, end)
                if existing is not None:
                    return existing
                if self.object_prefix is None and destination.exists():
                    if _sha256_file(destination) != digest:
                        raise IntegrityError(
                            f"content-address collision at {destination}"
                        )

                if self.object_prefix is None and not destination.exists():
                    os.replace(tmp, destination)
                    _fsync_dir(destination.parent)
                return self._commit_entry_locked(entry)
        finally:
            tmp.unlink(missing_ok=True)

    def finalize_host(self) -> dict[str, Any]:
        """Atomically mark this host complete after its last durable range."""
        path = _host_manifest_path(self.root, self.host_id)
        with _store_lock(self.root, self.host_id):
            if (self.root / MANIFEST_NAME).exists():
                raise ArtifactError("artifact store is already sealed")
            if not path.exists():
                raise ArtifactError(f"host {self.host_id!r} has committed no ranges")
            manifest = _read_json(path)
            if manifest.get("metadata_sha256") != self.metadata_sha256:
                raise SchemaError("capture metadata changed before host finalization")
            shards = manifest.get("shards")
            if not isinstance(shards, list) or not shards:
                raise ArtifactError(f"host {self.host_id!r} has no durable shards")
            if manifest.get("complete") is not True:
                manifest["complete"] = True
                manifest["completed_at"] = _utc_now()
                manifest["tokens"] = sum(int(entry["tokens"]) for entry in shards)
                _atomic_json(path, manifest)
            return manifest


def _validate_entry_shape(entry: Mapping[str, Any]) -> None:
    required = {
        "range_id",
        "range_start",
        "range_end",
        "host_id",
        "path",
        "sha256",
        "tokens",
        "squared_norm",
    }
    missing = sorted(required - entry.keys())
    if missing:
        raise IntegrityError(f"shard entry missing fields: {', '.join(missing)}")
    if not isinstance(entry["range_id"], str) or not entry["range_id"].strip():
        raise IntegrityError("shard range_id must be a nonempty string")
    if not isinstance(entry["host_id"], str) or not _HOST_RE.fullmatch(
        entry["host_id"]
    ):
        raise IntegrityError("shard host_id is invalid")
    for name in ("range_start", "range_end", "tokens"):
        if isinstance(entry[name], bool) or not isinstance(entry[name], int):
            raise IntegrityError(f"shard {name} must be an integer")
    if entry["range_start"] < 0 or entry["range_end"] <= entry["range_start"]:
        raise IntegrityError("shard input range is invalid")
    if not 0 < entry["tokens"] <= MAX_SHARD_TOKENS:
        raise IntegrityError("shard token count is invalid")
    if not isinstance(entry["squared_norm"], (int, float)) or not math.isfinite(
        entry["squared_norm"]
    ):
        raise IntegrityError("shard squared_norm must be finite")
    if entry["squared_norm"] < 0:
        raise IntegrityError("shard squared_norm must be nonnegative")
    if "sum" in entry:
        total = np.asarray(entry["sum"], dtype=np.float64)
        if total.shape != (HIDDEN_SIZE,) or not np.isfinite(total).all():
            raise IntegrityError("shard FP64 sum must be a finite hidden-width vector")
    if not re.fullmatch(r"[0-9a-f]{64}", str(entry["sha256"])):
        raise IntegrityError("invalid shard SHA-256")
    path = Path(str(entry["path"]))
    if path.is_absolute() or ".." in path.parts:
        raise IntegrityError(f"unsafe shard path {path}")
    expected_prefix = Path("shards") / f"host-{entry['host_id']}"
    if path.parent != expected_prefix or path.name != f"{entry['sha256']}.safetensors":
        raise IntegrityError("shard path is not host-specific and content-addressed")
    if "object" in entry:
        try:
            identity = object_identity_from_mapping(entry["object"])
        except (ObjectStoreError, ValueError) as error:
            raise IntegrityError("shard object identity is invalid") from error
        if identity.sha256 != entry["sha256"]:
            raise IntegrityError("shard object and logical path have different hashes")
        _, object_name = identity.uri[5:].split("/", 1)
        expected_suffix = f"/host-{entry['host_id']}/{entry['sha256']}.safetensors"
        if not f"/{object_name}".endswith(expected_suffix):
            raise IntegrityError("shard object URI is not host-specific/content-addressed")
        if "sum" not in entry:
            raise IntegrityError("remote shard entry lacks its publication statistic")


def _load_shard(
    root: Path,
    entry: Mapping[str, Any],
    verify_hash: bool,
    *,
    return_stats: bool = False,
    verify_statistics: bool = True,
    materialize: Any | None = None,
) -> CaptureBatch | tuple[CaptureBatch, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    _validate_entry_shape(entry)
    if "object" in entry:
        if materialize is None:
            raise IntegrityError("remote shard requires a hash-verifying local cache")
        path = Path(materialize(entry["object"]))
    else:
        path = root / str(entry["path"])
    if not path.is_file():
        raise IntegrityError(f"missing shard {path}")
    if verify_hash and _sha256_file(path) != entry["sha256"]:
        raise IntegrityError(f"SHA-256 mismatch for {path}")
    with safe_open(path, framework="numpy") as handle:
        if set(handle.keys()) != _FILE_KEYS:
            raise SchemaError(
                f"{path} tensor keys must be {sorted(_FILE_KEYS)}, got {sorted(handle.keys())}"
            )
        metadata = handle.metadata() or {}
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise LegacySchemaError(f"shard {path} is not schema {SCHEMA_VERSION}")
        for field in ("range_id", "host_id"):
            if metadata.get(field) != str(entry[field]):
                raise IntegrityError(f"shard metadata mismatch for {field}: {path}")
        for field in ("range_start", "range_end"):
            if metadata.get(field) != str(entry[field]):
                raise IntegrityError(f"shard metadata mismatch for {field}: {path}")
    tensors = load_file(path)
    batch = _normalize_batch(
        CaptureBatch(**{name: tensors[name] for name in _DATA_KEYS})
    )
    if _token_count(batch) != entry["tokens"]:
        raise IntegrityError(f"token count mismatch for {path}")
    if verify_statistics or return_stats:
        count, total, squared_norm = _batch_stats(batch)
        if not np.array_equal(tensors[_STATS_COUNT], count):
            raise IntegrityError(f"FP64 count statistic mismatch for {path}")
        if not np.array_equal(tensors[_STATS_SUM], total):
            raise IntegrityError(f"FP64 activation sum mismatch for {path}")
        if not np.array_equal(tensors[_STATS_SQUARED_NORM], squared_norm):
            raise IntegrityError(f"FP64 squared-norm statistic mismatch for {path}")
        if float(squared_norm[0]) != float(entry["squared_norm"]):
            raise IntegrityError(f"manifest squared-norm mismatch for {path}")
    if return_stats:
        return batch, (count, total, squared_norm)
    return batch


def publish_manifest(
    root: str | Path,
    metadata: CaptureMetadata | Mapping[str, Any],
    *,
    expected_hosts: int | None = None,
    expected_tokens: int | None = None,
    object_store: ImmutableObjectStore | None = None,
) -> Path:
    """Verify all host shards and atomically publish the sealing manifest."""
    root = Path(root)
    metadata_dict = _metadata_dict(metadata)
    metadata_sha = _sha256_json(metadata_dict)
    for name, value in (
        ("expected_hosts", expected_hosts),
        ("expected_tokens", expected_tokens),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ArtifactError(f"{name} must be a positive integer when supplied")
    path = root / MANIFEST_NAME
    with _store_lock(root):
        if path.exists():
            existing = load_manifest(root)
            if existing.get("schema_version") != SCHEMA_VERSION:
                raise LegacySchemaError("refusing to replace a legacy final manifest")
            if existing.get("metadata_sha256") != metadata_sha:
                raise SchemaError(
                    "published capture metadata differs from requested metadata"
                )
            if expected_hosts is not None and (
                existing.get("host_count") != expected_hosts
                or existing.get("completed_host_count") != expected_hosts
            ):
                raise IntegrityError(
                    "published completed host count differs from expected_hosts"
                )
            if (
                expected_tokens is not None
                and existing.get("total_tokens") != expected_tokens
            ):
                raise IntegrityError(
                    "published token count differs from expected_tokens"
                )
            return path

        hosts = _host_manifests(root)
        if not hosts:
            raise ArtifactError("cannot publish an empty artifact store")
        host_ids = [host.get("host_id") for host in hosts]
        if len(set(host_ids)) != len(host_ids):
            raise IntegrityError("host manifests contain duplicate host identities")
        if expected_hosts is not None:
            completed_ids = {
                host["host_id"] for host in hosts if host.get("complete") is True
            }
            if len(hosts) != expected_hosts or len(completed_ids) != expected_hosts:
                raise ArtifactError(
                    "cannot publish: found "
                    f"{len(completed_ids)} unique completed hosts across {len(hosts)} "
                    f"manifests, expected exactly {expected_hosts}"
                )
        for host in hosts:
            if (
                host.get("metadata_sha256") != metadata_sha
                or host.get("metadata") != metadata_dict
            ):
                raise SchemaError(
                    "host manifests do not share the publication metadata"
                )
        entries = [entry for host in hosts for entry in host.get("shards", [])]
        if not entries:
            raise ArtifactError("cannot publish a store with no shards")
        remote_flags = {"object" in entry for entry in entries}
        if len(remote_flags) != 1:
            raise IntegrityError("capture cannot mix POSIX and object-store shards")
        remote_capture = True in remote_flags
        if remote_capture:
            object_prefix = metadata_dict.get("object_prefix")
            storage_digest = metadata_dict.get("storage_attestation_sha256")
            if not isinstance(object_prefix, str) or not isinstance(
                storage_digest, str
            ):
                raise SchemaError(
                    "remote capture metadata must bind its object prefix and storage attestation"
                )
            for entry in entries:
                identity = object_identity_from_mapping(entry["object"])
                if not _gcs_object_under(object_prefix, identity.uri):
                    raise IntegrityError(
                        "capture object lies outside the metadata-bound GCS prefix"
                    )
        entries.sort(key=lambda x: (x["range_start"], x["range_id"]))
        seen: list[dict[str, Any]] = []
        count = 0
        range_cursor = 0
        total = np.zeros(HIDDEN_SIZE, dtype=np.float64)
        squared_norm = 0.0
        for entry in entries:
            _validate_entry_shape(entry)
            if int(entry["range_start"]) != range_cursor:
                raise RangeConflictError(
                    f"capture ranges are not contiguous at token {range_cursor}"
                )
            if int(entry["range_end"]) - int(entry["range_start"]) != int(
                entry["tokens"]
            ):
                raise IntegrityError("shard range length differs from its token count")
            range_cursor = int(entry["range_end"])
            if (
                _find_range_conflict(
                    seen, entry["range_id"], entry["range_start"], entry["range_end"]
                )
                is not None
            ):
                raise RangeConflictError(f"duplicate input range {entry['range_id']!r}")
            seen.append(entry)
            if "object" in entry:
                if object_store is None:
                    raise ArtifactError(
                        "remote capture publication requires an object-store verifier"
                    )
                _validate_upload_receipt(
                    root, entry, metadata_sha256=metadata_sha
                )
                identity = object_identity_from_mapping(entry["object"])
                object_store.verify(identity)
                count += int(entry["tokens"])
                total += np.asarray(entry["sum"], dtype=np.float64)
                squared_norm += float(entry["squared_norm"])
            else:
                _, (shard_count, shard_total, shard_sq) = _load_shard(
                    root, entry, verify_hash=True, return_stats=True
                )
                count += int(shard_count[0])
                total += shard_total
                squared_norm += float(shard_sq[0])
        if expected_tokens is not None and count != expected_tokens:
            raise ArtifactError(
                f"cannot publish: captured {count} tokens, expected {expected_tokens}"
            )
        if expected_tokens is not None and range_cursor != expected_tokens:
            raise ArtifactError(
                f"cannot publish: range coverage ends at {range_cursor}, "
                f"expected {expected_tokens}"
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "sealed": True,
            "metadata_sha256": metadata_sha,
            "metadata": metadata_dict,
            "host_count": len(set(host_ids)),
            "completed_host_count": sum(host.get("complete") is True for host in hosts),
            "shard_count": len(entries),
            "total_tokens": count,
            "statistics": {
                "count": count,
                "sum": total.tolist(),
                "squared_norm": squared_norm,
                "centered_sum_of_squares": max(
                    0.0, squared_norm - float(np.dot(total, total)) / count
                ),
            },
            "storage": (
                {
                    "kind": "gcs-immutable",
                    "object_prefix": metadata_dict["object_prefix"],
                    "identity_fields": [
                        "uri",
                        "generation",
                        "size_bytes",
                        "crc32c",
                        "sha256",
                    ],
                }
                if remote_capture
                else {"kind": "posix-local"}
            ),
            "shards": entries,
        }
        manifest["manifest_sha256"] = _sha256_json(manifest)
        _atomic_json(path, manifest)
    return path


def load_manifest(root: str | Path) -> dict[str, Any]:
    """Load a sealed v1 manifest, explicitly rejecting legacy layouts."""
    root = Path(root)
    path = root / MANIFEST_NAME
    if not path.exists():
        if root.exists() and any(root.rglob("*.safetensors")):
            raise LegacySchemaError(
                f"{root} has tensors but no {SCHEMA_VERSION} final manifest; "
                "legacy capture layouts are not accepted"
            )
        raise ArtifactError(f"missing final manifest: {path}")
    manifest = _read_json(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise LegacySchemaError(
            f"schema {manifest.get('schema_version')!r} is not {SCHEMA_VERSION}"
        )
    if manifest.get("sealed") is not True:
        raise IntegrityError("artifact manifest is not sealed")
    recorded_manifest_sha = manifest.get("manifest_sha256")
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if (
        not isinstance(recorded_manifest_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", recorded_manifest_sha)
        or recorded_manifest_sha != _sha256_json(unsigned_manifest)
    ):
        raise IntegrityError("manifest SHA-256 mismatch")
    _metadata_dict(manifest.get("metadata", {}))
    if manifest.get("metadata_sha256") != _sha256_json(manifest["metadata"]):
        raise IntegrityError("manifest metadata SHA-256 mismatch")
    entries = manifest.get("shards")
    if not isinstance(entries, list) or not entries:
        raise IntegrityError("manifest contains no shards")
    seen: list[dict[str, Any]] = []
    range_cursor = 0
    for entry in entries:
        _validate_entry_shape(entry)
        if int(entry["range_start"]) != range_cursor:
            raise IntegrityError(
                f"manifest ranges are not contiguous and ordered at {range_cursor}"
            )
        if int(entry["range_end"]) - int(entry["range_start"]) != int(entry["tokens"]):
            raise IntegrityError("manifest shard range length differs from token count")
        range_cursor = int(entry["range_end"])
        if (
            _find_range_conflict(
                seen, entry["range_id"], entry["range_start"], entry["range_end"]
            )
            is not None
        ):
            raise IntegrityError(f"duplicate input range {entry['range_id']!r}")
        seen.append(entry)
    remote_flags = {"object" in entry for entry in entries}
    if len(remote_flags) != 1:
        raise IntegrityError("manifest mixes POSIX and object-store shards")
    remote_capture = True in remote_flags
    storage = manifest.get("storage")
    if storage is not None:
        expected_kind = "gcs-immutable" if remote_capture else "posix-local"
        if not isinstance(storage, dict) or storage.get("kind") != expected_kind:
            raise IntegrityError("manifest storage layout is inconsistent with shards")
    if remote_capture:
        object_prefix = manifest["metadata"].get("object_prefix")
        if not isinstance(object_prefix, str):
            raise IntegrityError("remote manifest lacks its bound object prefix")
        for entry in entries:
            identity = object_identity_from_mapping(entry["object"])
            if not _gcs_object_under(object_prefix, identity.uri):
                raise IntegrityError("manifest object lies outside its bound GCS prefix")
    if manifest.get("shard_count") != len(entries):
        raise IntegrityError("manifest shard_count mismatch")
    host_count = manifest.get("host_count")
    if not isinstance(host_count, int) or host_count <= 0:
        raise IntegrityError("manifest host_count must be a positive integer")
    if host_count != len({entry["host_id"] for entry in entries}):
        raise IntegrityError("manifest host_count does not match shard hosts")
    completed_host_count = manifest.get("completed_host_count")
    if (
        not isinstance(completed_host_count, int)
        or completed_host_count < 0
        or completed_host_count > host_count
    ):
        raise IntegrityError("manifest completed_host_count is invalid")
    if manifest.get("total_tokens") != sum(int(x["tokens"]) for x in entries):
        raise IntegrityError("manifest total_tokens mismatch")
    if range_cursor != manifest["total_tokens"]:
        raise IntegrityError("manifest ranges do not cover exactly total_tokens")
    stats = manifest.get("statistics", {})
    if stats.get("count") != manifest["total_tokens"]:
        raise IntegrityError("manifest statistics count mismatch")
    if not isinstance(stats.get("sum"), list) or len(stats["sum"]) != HIDDEN_SIZE:
        raise IntegrityError(f"manifest statistics sum must have {HIDDEN_SIZE} entries")
    for name in ("squared_norm", "centered_sum_of_squares"):
        value = stats.get(name)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise IntegrityError(
                f"manifest statistics {name} must be finite and nonnegative"
            )
    return manifest


@dataclasses.dataclass(frozen=True)
class ArtifactShard:
    """A verified shard together with its immutable range identity."""

    range_id: str
    range_start: int
    range_end: int
    host_id: str
    sha256: str
    batch: CaptureBatch


class ArtifactDataset:
    """Reader for a sealed v1 artifact dataset."""

    def __init__(
        self,
        root: str | Path,
        *,
        verify_hashes: bool = True,
        verify_statistics: bool = True,
        decode_bf16: bool = True,
        cache: ShardCache | None = None,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.manifest = (
            load_manifest(self.root) if manifest is None else dict(manifest)
        )
        self.verify_hashes = bool(verify_hashes)
        self.verify_statistics = bool(verify_statistics)
        self.decode_bf16 = bool(decode_bf16)
        self.cache = cache

    def _materialize(self, entry: Mapping[str, Any]) -> Path:
        if "object" not in entry:
            return self.root / str(entry["path"])
        if self.cache is None:  # guarded in __init__; keep the read path fail-closed
            raise ArtifactError("remote capture has no configured local cache")
        return self.cache.materialize(entry["object"])

    def prefetch_token_range(self, start: int, end: int) -> dict[str, int]:
        """Materialize every shard overlapping ``[start,end)`` before SAE timing."""

        for name, value in (("start", start), ("end", end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 0 <= start <= end <= self.total_tokens:
            raise ValueError(
                f"token range must satisfy 0 <= start <= end <= {self.total_tokens}"
            )
        offset = 0
        selected: list[Mapping[str, Any]] = []
        for entry in self.manifest["shards"]:
            shard_start = offset
            shard_end = offset + int(entry["tokens"])
            offset = shard_end
            if max(start, shard_start) >= min(end, shard_end):
                continue
            selected.append(entry)

        remote = [entry["object"] for entry in selected if "object" in entry]
        remote_paths: dict[tuple[str, int], Path] = {}
        if remote:
            if self.cache is None:
                raise ArtifactError("remote capture has no configured local cache")
            paths = self.cache.materialize_many(remote, max_workers=4)
            for value, path in zip(remote, paths, strict=True):
                identity = object_identity_from_mapping(value)
                remote_paths[(identity.uri, identity.generation)] = path

        objects = 0
        bytes_materialized = 0
        for entry in selected:
            if "object" in entry:
                identity = object_identity_from_mapping(entry["object"])
                path = remote_paths[(identity.uri, identity.generation)]
            else:
                path = self._materialize(entry)
            if not path.is_file():
                raise IntegrityError(f"materialized shard is missing: {path}")
            objects += 1
            bytes_materialized += path.stat().st_size
        return {"objects": objects, "bytes": bytes_materialized}

    @property
    def total_tokens(self) -> int:
        return int(self.manifest["total_tokens"])

    @property
    def manifest_sha256(self) -> str:
        """Canonical identity of the sealed manifest and every referenced shard."""

        return str(self.manifest["manifest_sha256"])

    @property
    def mean(self) -> np.ndarray:
        stats = self.manifest["statistics"]
        return np.asarray(stats["sum"], dtype=np.float64) / float(stats["count"])

    @property
    def squared_norm(self) -> float:
        return float(self.manifest["statistics"]["squared_norm"])

    @property
    def centered_sum_of_squares(self) -> float:
        """SST around this dataset's own mean (diagnostic convenience only).

        The A/B/F evaluation denominator must instead use the training mean
        through :meth:`sst_about`, so every condition shares one fixed SST.
        """
        return float(self.manifest["statistics"]["centered_sum_of_squares"])

    def sst_about(self, mean: Any) -> float:
        """Return FP64 ``sum ||x-mean||^2`` from sufficient statistics.

        This does not read activation shards.  It combines the manifest's
        count, vector sum, and squared-norm sum and is therefore the required
        way to evaluate held-out data about the *training* activation mean.
        """
        mean64 = np.asarray(mean, dtype=np.float64)
        if mean64.shape != (HIDDEN_SIZE,) or not np.isfinite(mean64).all():
            raise SchemaError(f"SST mean must be a finite [{HIDDEN_SIZE}] vector")
        stats = self.manifest["statistics"]
        count = float(stats["count"])
        total = np.asarray(stats["sum"], dtype=np.float64)
        squared_norm = float(stats["squared_norm"])
        cross = 2.0 * float(np.dot(mean64, total))
        mean_energy = count * float(np.dot(mean64, mean64))
        sst = squared_norm - cross + mean_energy
        scale = max(1.0, abs(squared_norm) + abs(cross) + abs(mean_energy))
        if sst < -1e-12 * scale:
            raise IntegrityError("sufficient statistics imply a negative SST")
        return max(0.0, sst)

    def iter_shards(self) -> Iterator[ArtifactShard]:
        for entry in self.manifest["shards"]:
            batch = _load_shard(
                self.root,
                entry,
                verify_hash=self.verify_hashes,
                verify_statistics=self.verify_statistics,
                materialize=(self.cache.materialize if self.cache is not None else None),
            )
            assert isinstance(batch, CaptureBatch)
            if self.decode_bf16:
                batch = _decode_batch(batch)
            yield ArtifactShard(
                range_id=entry["range_id"],
                range_start=int(entry["range_start"]),
                range_end=int(entry["range_end"]),
                host_id=entry["host_id"],
                sha256=entry["sha256"],
                batch=batch,
            )

    def iter_batches(self) -> Iterator[CaptureBatch]:
        for shard in self.iter_shards():
            yield shard.batch

    def iter_expert_ids_range(self, start: int, end: int) -> Iterator[np.ndarray]:
        """Read only router IDs for ``[start,end)`` from a sealed capture.

        The publication pass has already hash-verified every complete shard.
        This route-only view deliberately avoids rereading 283 GB of activation
        payload merely to derive condition A's static grouped-GEMM capacity;
        ordinary training still verifies each full shard when it consumes it.
        """

        for name, value in (("start", start), ("end", end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 0 <= start <= end <= self.total_tokens:
            raise ValueError(
                f"token range must satisfy 0 <= start <= end <= {self.total_tokens}"
            )
        offset = 0
        yielded = 0
        for entry in self.manifest["shards"]:
            shard_start = offset
            shard_end = offset + int(entry["tokens"])
            offset = shard_end
            overlap_start = max(start, shard_start)
            overlap_end = min(end, shard_end)
            if overlap_start >= overlap_end:
                continue
            _validate_entry_shape(entry)
            path = self._materialize(entry)
            if not path.is_file():
                raise IntegrityError(f"missing shard {path}")
            with safe_open(path, framework="numpy") as handle:
                metadata = handle.metadata() or {}
                if metadata.get("schema_version") != SCHEMA_VERSION:
                    raise LegacySchemaError(
                        f"shard {path} is not schema {SCHEMA_VERSION}"
                    )
                ids = handle.get_tensor("expert_ids")
            if ids.dtype != np.uint8 or ids.shape != (
                int(entry["tokens"]),
                TOP_K_EXPERTS,
            ):
                raise SchemaError(f"invalid expert_ids tensor in {path}")
            local_start = overlap_start - shard_start
            local_end = overlap_end - shard_start
            selected = np.asarray(ids[local_start:local_end], dtype=np.int32)
            if selected.size and (
                selected.min() < 0
                or selected.max() >= NUM_EXPERTS
                or np.any(np.diff(np.sort(selected, axis=1), axis=1) == 0)
            ):
                raise SchemaError(f"invalid router IDs in {path}")
            yielded += selected.shape[0]
            yield selected
        if yielded != end - start:
            raise IntegrityError(
                f"manifest yielded {yielded} router rows for [{start},{end})"
            )

    def iter_token_range(self, start: int, end: int) -> Iterator[CaptureBatch]:
        """Yield exactly ``[start,end)`` in cumulative manifest token order.

        Shards wholly outside the requested interval are not opened.  Only the
        first and last overlapping batches are sliced, allowing distributed
        callers to partition the scientific token stream without duplicate
        I/O except when two partitions meet inside the same physical shard.
        """
        for name, value in (("start", start), ("end", end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 0 <= start <= end <= self.total_tokens:
            raise ValueError(
                f"token range must satisfy 0 <= start <= end <= {self.total_tokens}"
            )

        offset = 0
        yielded = 0
        for entry in self.manifest["shards"]:
            shard_start = offset
            shard_end = offset + int(entry["tokens"])
            offset = shard_end
            overlap_start = max(start, shard_start)
            overlap_end = min(end, shard_end)
            if overlap_start >= overlap_end:
                continue
            loaded = _load_shard(
                self.root,
                entry,
                verify_hash=self.verify_hashes,
                verify_statistics=self.verify_statistics,
                materialize=(self.cache.materialize if self.cache is not None else None),
            )
            assert isinstance(loaded, CaptureBatch)
            if self.decode_bf16:
                loaded = _decode_batch(loaded)
            local_start = overlap_start - shard_start
            local_end = overlap_end - shard_start
            batch = CaptureBatch(
                *(np.asarray(field)[local_start:local_end] for field in loaded)
            )
            yielded += local_end - local_start
            yield batch
        if yielded != end - start:
            raise IntegrityError(
                f"manifest yielded {yielded} tokens for requested range [{start},{end})"
            )


__all__ = [
    "ArtifactDataset",
    "ArtifactError",
    "ArtifactShard",
    "ArtifactWriter",
    "CaptureBatch",
    "CaptureMetadata",
    "HIDDEN_SIZE",
    "IntegrityError",
    "LegacySchemaError",
    "MANIFEST_NAME",
    "MAX_SHARD_TOKENS",
    "NUM_EXPERTS",
    "RangeConflictError",
    "SCHEMA_VERSION",
    "SchemaError",
    "STAGED_DATASET_SCHEMA_VERSION",
    "StagedDatasetProvenance",
    "TARGET_LAYER_ONE_INDEXED",
    "TOP_K_EXPERTS",
    "bf16_bits_to_float32",
    "float32_to_bf16_bits",
    "load_manifest",
    "publish_manifest",
    "verify_staged_dataset_manifest",
]
