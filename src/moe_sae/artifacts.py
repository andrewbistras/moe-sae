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
from importlib import import_module
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

from moe_sae.model import CaptureBatch


SCHEMA_VERSION = "gemma4-router-aggregate-v1"
MANIFEST_NAME = "manifest.json"
MAX_SHARD_TOKENS = 131_072
HIDDEN_SIZE = 2_816
NUM_EXPERTS = 128
TOP_K_EXPERTS = 8
TARGET_LAYER_ONE_INDEXED = 15
DEFAULT_CAPTURE_SITE = "routed_moe.post_feedforward_layernorm_2"
STAGED_DATASET_SCHEMA_VERSION = "gemma4-c4-arrayrecord-v1"

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
    sequence_length: int,
    verify_file_hashes: bool = True,
    verify_arrayrecord_count: bool = False,
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
        "tokenizer_id",
        "sequence_length",
        "packing",
        "files",
    }
    if set(value) != required:
        raise IntegrityError(
            "staged dataset manifest fields differ from the v1 contract"
        )
    if value.get("schema_version") != STAGED_DATASET_SCHEMA_VERSION:
        raise IntegrityError("unsupported staged dataset manifest schema")
    expected_dataset = {
        "name": "c4",
        "config": "en",
        "version": "3.1.0",
        "split": expected_split,
    }
    if value.get("dataset") != expected_dataset:
        raise IntegrityError("staged dataset is not the required C4 English split")
    if value.get("tokenizer_id") != tokenizer_id:
        raise IntegrityError("staged dataset uses a different tokenizer")
    if value.get("sequence_length") != sequence_length:
        raise IntegrityError("staged dataset uses a different packed sequence length")
    packing = value.get("packing")
    if not isinstance(packing, Mapping) or dict(packing) != {
        "strategy": "packed_fixed_length",
        "exclude_from_scientific_count": ["padding", "bos", "eos"],
        "special_token_ids": {"padding": 0, "eos": 1, "bos": 2},
    }:
        raise IntegrityError("staged dataset packing/counting contract is incompatible")
    raw_files = value.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise IntegrityError("staged dataset manifest must inventory at least one file")

    resolved = _resolve_staged_files(grain_files)
    declared: list[str] = []
    declared_records = 0
    scientific_tokens = 0
    for entry in raw_files:
        if not isinstance(entry, Mapping) or set(entry) != {
            "uri",
            "sha256",
            "size_bytes",
            "record_count",
            "scientific_tokens",
        }:
            raise IntegrityError("staged file inventory entry has incompatible fields")
        uri = _normalize_declared_uri(str(entry.get("uri", "")), manifest_location)
        digest = str(entry.get("sha256", ""))
        size = entry.get("size_bytes")
        records = entry.get("record_count")
        tokens = entry.get("scientific_tokens")
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
            or tokens <= 0
        ):
            raise IntegrityError("staged file inventory has invalid counts")
        declared.append(uri)
        declared_records += records
        scientific_tokens += tokens
        if verify_file_hashes:
            actual_digest, actual_size = _hash_location(uri)
            if actual_digest != digest or actual_size != size:
                raise IntegrityError(f"staged ArrayRecord hash/size mismatch: {uri}")
    if tuple(declared) != resolved:
        raise IntegrityError(
            "ordered staged manifest inventory differs from --grain-files"
        )
    if verify_arrayrecord_count:
        try:
            grain = import_module("grain.python")
            actual_records = len(grain.ArrayRecordDataSource(list(resolved)))
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
    if scientific_tokens < expected_tokens:
        raise IntegrityError(
            f"staged split contains only {scientific_tokens:,} scientific tokens; "
            f"need at least {expected_tokens:,}"
        )
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
    model_id: str = "google/gemma-4-26B-A4B"
    dataset: str = "c4/en:3.1.0"
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
    fixed = {
        "model_id": "google/gemma-4-26B-A4B",
        "maxtext_commit": "48c7da3e28e8f57fe36ddf09d00eec53072a2ea0",
        "dataset": "c4/en:3.1.0",
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
        (self.root / "shards" / f"host-{host_id}").mkdir(parents=True, exist_ok=True)
        (self.root / ".staging" / f"host-{host_id}").mkdir(parents=True, exist_ok=True)
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
        with _store_lock(self.root, self.host_id):
            path = _host_manifest_path(self.root, self.host_id)
            entries = _read_json(path).get("shards", []) if path.exists() else []
            return _find_range_conflict(entries, range_id, start, end) is not None

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

        staging_dir = self.root / ".staging" / f"host-{self.host_id}"
        fd, tmp_name = tempfile.mkstemp(
            prefix="shard-", suffix=".safetensors", dir=staging_dir
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
            digest = _sha256_file(tmp)
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
                "squared_norm": float(squared_norm[0]),
            }

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
                if destination.exists():
                    if _sha256_file(destination) != digest:
                        raise IntegrityError(
                            f"content-address collision at {destination}"
                        )

                if host_path.exists():
                    host_manifest = _read_json(host_path)
                    if host_manifest.get("metadata_sha256") != self.metadata_sha256:
                        raise SchemaError("capture metadata changed during host commit")
                    if host_manifest.get("complete") is True:
                        raise ArtifactError(
                            f"host {self.host_id!r} is finalized; no new ranges are allowed"
                        )
                else:
                    host_manifest = {
                        "schema_version": SCHEMA_VERSION,
                        "host_id": self.host_id,
                        "metadata_sha256": self.metadata_sha256,
                        "metadata": self.metadata,
                        "complete": False,
                        "shards": [],
                    }
                if not destination.exists():
                    os.replace(tmp, destination)
                    _fsync_dir(destination.parent)
                host_manifest["shards"].append(entry)
                host_manifest["shards"].sort(
                    key=lambda x: (x["range_start"], x["range_id"])
                )
                host_manifest["updated_at"] = _utc_now()
                _atomic_json(host_path, host_manifest)
            return entry
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
    if not re.fullmatch(r"[0-9a-f]{64}", str(entry["sha256"])):
        raise IntegrityError("invalid shard SHA-256")
    path = Path(str(entry["path"]))
    if path.is_absolute() or ".." in path.parts:
        raise IntegrityError(f"unsafe shard path {path}")
    expected_prefix = Path("shards") / f"host-{entry['host_id']}"
    if path.parent != expected_prefix or path.name != f"{entry['sha256']}.safetensors":
        raise IntegrityError("shard path is not host-specific and content-addressed")


def _load_shard(
    root: Path,
    entry: Mapping[str, Any],
    verify_hash: bool,
    *,
    return_stats: bool = False,
    verify_statistics: bool = True,
) -> CaptureBatch | tuple[CaptureBatch, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    _validate_entry_shape(entry)
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
    ) -> None:
        self.root = Path(root)
        self.manifest = load_manifest(self.root)
        self.verify_hashes = bool(verify_hashes)
        self.verify_statistics = bool(verify_statistics)
        self.decode_bf16 = bool(decode_bf16)

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
            path = self.root / str(entry["path"])
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
