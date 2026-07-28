"""Immutable object-store data plane for Experiment-2 contribution shards.

The scientific schema lives in :mod:`moe_sae.contribution_artifacts`.  This
module supplies only its storage mechanics:

* six-field v2 payloads and FP64 sufficient statistics are serialized together
  in one safetensors object;
* workers upload directly from bounded host-local staging to host-specific,
  content-addressed object keys;
* the shared POSIX control plane stores small host indexes, self-hashed upload
  receipts, and the sealed manifest;
* resume is keyed by the source-manifest-bound deterministic ``range_id``; and
* readers materialize generation-pinned objects through :class:`ShardCache`
  and revalidate the v2 payload, file metadata, statistics, and SHA-256.

No bucket listing is used to infer progress.  An upload that completes before
its receipt is written is recovered from a durable local intent when possible;
otherwise deterministic replay republishes to the same immutable object key.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import contextlib
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from moe_sae.artifacts import (
    ArtifactDataset,
    ArtifactError,
    load_manifest as load_v1_manifest,
)
from moe_sae.contribution_artifacts import (
    LEGACY_AGGREGATE_SCHEMA_VERSION,
    MAX_SHARD_TOKENS,
    NUM_EXPERTS,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    TOP_K_EXPERTS,
    ContributionBatch,
    ContributionIntegrityError,
    ContributionLegacySchemaError,
    ContributionMetadata,
    ContributionRangeError,
    bf16_bits_to_float32,
    canonical_json_bytes,
    canonical_sha256,
    compute_contribution_statistics,
    deterministic_range_id,
    merge_contribution_statistics,
    seal_contribution_manifest,
    token_identity_sha256,
    validate_contribution_batch,
    validate_contribution_manifest,
    validate_contribution_shard_payload,
    validate_contribution_statistics,
    validate_metadata,
    validate_source_v1_token_keys,
)
from moe_sae.object_store import (
    ImmutableObjectStore,
    ObjectIdentity,
    ShardCache,
    fingerprint_file,
    join_gcs_uri,
    object_identity_from_mapping,
)


MANIFEST_NAME = "manifest.json"
RECEIPT_SCHEMA_VERSION = "gemma4-contribution-upload-receipt-v1"
HOST_INDEX_SCHEMA_VERSION = "gemma4-contribution-host-index-v1"
UPLOAD_INTENT_SCHEMA_VERSION = "gemma4-contribution-upload-intent-v1"

_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONTROL_JSON_BYTES = 64 * 1024 * 1024
_MAX_MANIFEST_JSON_BYTES = 2 * 1024 * 1024 * 1024
_PAYLOAD_FIELDS = (
    "expert_output",
    "expert_ids",
    "router_weights",
    "shared_output",
    "sequence_id",
    "position",
)

# Each statistic is stored as an actual FP64/int64 tensor rather than a string
# in safetensors metadata.  The manifest retains the canonical JSON form
# defined by contribution_artifacts.
_STAT_SPECS: dict[str, tuple[tuple[str, ...], np.dtype[Any], tuple[int, ...]]] = {
    "__stats_token_count": (("token_count",), np.dtype(np.int64), (1,)),
    "__stats_component_count": (("component_count",), np.dtype(np.int64), (1,)),
    "__stats_pooled_count": (
        ("pooled_expert_output", "count"),
        np.dtype(np.int64),
        (1,),
    ),
    "__stats_pooled_sum": (
        ("pooled_expert_output", "sum"),
        np.dtype(np.float64),
        (2_816,),
    ),
    "__stats_pooled_squared_norm": (
        ("pooled_expert_output", "squared_norm"),
        np.dtype(np.float64),
        (1,),
    ),
    "__stats_weight_squared_sum": (
        ("weighted_contribution", "weight_squared_sum"),
        np.dtype(np.float64),
        (1,),
    ),
    "__stats_weight_squared_output_sum": (
        ("weighted_contribution", "weight_squared_output_sum"),
        np.dtype(np.float64),
        (2_816,),
    ),
    "__stats_weighted_squared_norm": (
        ("weighted_contribution", "squared_norm"),
        np.dtype(np.float64),
        (1,),
    ),
    "__stats_weighted_raw_sum": (
        ("weighted_contribution", "raw_sum"),
        np.dtype(np.float64),
        (2_816,),
    ),
    "__stats_router_rank_count": (
        ("router_rank", "count"),
        np.dtype(np.int64),
        (8,),
    ),
    "__stats_router_rank_energy": (
        ("router_rank", "energy"),
        np.dtype(np.float64),
        (8,),
    ),
    "__stats_expert_id_count": (
        ("expert_id", "count"),
        np.dtype(np.int64),
        (128,),
    ),
    "__stats_expert_id_energy": (
        ("expert_id", "energy"),
        np.dtype(np.float64),
        (128,),
    ),
    "__stats_shared_raw_count": (
        ("shared_raw", "count"),
        np.dtype(np.int64),
        (1,),
    ),
    "__stats_shared_raw_sum": (
        ("shared_raw", "sum"),
        np.dtype(np.float64),
        (2_816,),
    ),
    "__stats_shared_raw_squared_norm": (
        ("shared_raw", "squared_norm"),
        np.dtype(np.float64),
        (1,),
    ),
    "__stats_shared_normalized_count": (
        ("shared_normalized", "count"),
        np.dtype(np.int64),
        (1,),
    ),
    "__stats_shared_normalized_sum": (
        ("shared_normalized", "sum"),
        np.dtype(np.float64),
        (2_816,),
    ),
    "__stats_shared_normalized_squared_norm": (
        ("shared_normalized", "squared_norm"),
        np.dtype(np.float64),
        (1,),
    ),
}
_FILE_TENSORS = frozenset((*_PAYLOAD_FIELDS, *_STAT_SPECS))


class ContributionStorageError(RuntimeError):
    """The v2 storage transaction could not complete safely."""


@dataclass(frozen=True, slots=True)
class ContributionShard:
    """One verified contribution shard materialized by the reader."""

    range_id: str
    range_start: int
    range_end: int
    host_id: str
    sha256: str
    object: ObjectIdentity
    statistics: Mapping[str, Any]
    batch: ContributionBatch


@dataclass(frozen=True, slots=True)
class ContributionSharedBatch:
    """The three v2 tensors needed by the common shared-expert SAE.

    Safetensors stores every named tensor at an independently addressable byte
    offset.  This view therefore avoids allocating or decoding the roughly
    eight-times-larger ``expert_output`` tensor when Q is the only consumer.
    The enclosing object is still materialized and hash-verified by
    :class:`ShardCache`; selective GCS range retrieval is deliberately not
    implied by this API.
    """

    shared_output: np.ndarray
    sequence_id: np.ndarray
    position: np.ndarray

    @property
    def tokens(self) -> int:
        return int(self.shared_output.shape[0])


@dataclass(frozen=True, slots=True)
class ContributionSharedShard:
    """One verified contribution shard with only Q-required tensors loaded."""

    range_id: str
    range_start: int
    range_end: int
    host_id: str
    sha256: str
    object: ObjectIdentity
    batch: ContributionSharedBatch


@dataclass(frozen=True, slots=True)
class ContributionCapacityBatch:
    """Minimal routed fields needed by the A/R grouped-GEMM capacity scan.

    ``sequence_id`` is absent for A because true expert IDs already are its
    bank IDs.  R requests it because its bank permutation is keyed by packed
    sequence.  The enclosing immutable safetensors object and its metadata are
    still hash-verified before either tensor is exposed.
    """

    expert_ids: np.ndarray
    sequence_id: np.ndarray | None

    @property
    def tokens(self) -> int:
        return int(self.expert_ids.shape[0])


def _validate_capacity_batch(
    value: ContributionCapacityBatch,
    *,
    expected_tokens: int,
    include_sequence_id: bool,
) -> ContributionCapacityBatch:
    expert_ids = np.asarray(value.expert_ids)
    if expert_ids.dtype != np.uint8 or expert_ids.shape != (
        expected_tokens,
        TOP_K_EXPERTS,
    ):
        raise ContributionIntegrityError(
            "selective expert_ids must be uint8 [tokens,8]"
        )
    if np.any(expert_ids >= NUM_EXPERTS):
        raise ContributionIntegrityError(
            f"selective expert_ids must be in [0,{NUM_EXPERTS - 1}]"
        )
    sorted_ids = np.sort(expert_ids, axis=1)
    if np.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
        raise ContributionIntegrityError(
            "selective expert_ids must be unique within each token"
        )

    sequence_id = value.sequence_id
    if include_sequence_id:
        sequence = np.asarray(sequence_id)
        if sequence.dtype != np.int64 or sequence.shape != (expected_tokens,):
            raise ContributionIntegrityError(
                "selective sequence_id must be int64 [tokens]"
            )
        if np.any(sequence < 0):
            raise ContributionIntegrityError(
                "selective sequence_id must be nonnegative"
            )
        checked_sequence: np.ndarray | None = np.ascontiguousarray(sequence)
    else:
        if sequence_id is not None:
            raise ContributionIntegrityError(
                "A-only capacity reads must not load sequence_id"
            )
        checked_sequence = None

    return ContributionCapacityBatch(
        expert_ids=np.ascontiguousarray(expert_ids),
        sequence_id=checked_sequence,
    )


def _validate_shared_batch(
    value: ContributionSharedBatch,
    *,
    expected_tokens: int,
    expected_token_identity_sha256: str | None,
) -> ContributionSharedBatch:
    shared = np.asarray(value.shared_output)
    sequence = np.asarray(value.sequence_id)
    position = np.asarray(value.position)
    if shared.dtype != np.uint16 or shared.shape != (expected_tokens, 2_816):
        raise ContributionIntegrityError(
            "selective shared_output must be uint16 BF16 bits [tokens,2816]"
        )
    if sequence.dtype != np.int64 or sequence.shape != (expected_tokens,):
        raise ContributionIntegrityError(
            "selective sequence_id must be int64 [tokens]"
        )
    if position.dtype != np.uint16 or position.shape != (expected_tokens,):
        raise ContributionIntegrityError(
            "selective position must be uint16 [tokens]"
        )
    if not np.isfinite(bf16_bits_to_float32(shared)).all():
        raise ContributionIntegrityError(
            "selective shared_output contains non-finite BF16 values"
        )
    observed_identity = token_identity_sha256(sequence, position)
    if (
        expected_token_identity_sha256 is not None
        and observed_identity != expected_token_identity_sha256
    ):
        raise ContributionIntegrityError(
            "selective shared token identity differs from the sealed shard"
        )
    return ContributionSharedBatch(
        shared_output=np.ascontiguousarray(shared),
        sequence_id=np.ascontiguousarray(sequence),
        position=np.ascontiguousarray(position),
    )


def _canonical_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(canonical_json_bytes(dict(value)))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    except OSError:
        # All control files are reconstructible from immutable receipts/objects.
        pass
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(
                dict(value),
                handle,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    expected = _canonical_copy(value)
    if path.exists():
        if _read_json(path) != expected:
            raise ContributionStorageError(
                f"immutable control record differs from replay: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(expected))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Publish without replacement: immutable receipts/manifests may be
            # replayed, but a racing writer can never overwrite the winner.
            os.link(temporary, path)
        except FileExistsError:
            pass
        else:
            _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    if _read_json(path) != expected:
        raise ContributionStorageError(
            f"immutable control record failed read-after-write: {path}"
        )


def _read_json(path: Path, *, manifest: bool = False) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContributionStorageError(
            f"control record is absent, non-regular, or a symlink: {path}"
        )
    limit = _MAX_MANIFEST_JSON_BYTES if manifest else _MAX_CONTROL_JSON_BYTES
    if path.stat().st_size > limit:
        raise ContributionStorageError(f"control record is unexpectedly large: {path}")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContributionStorageError(f"cannot read canonical JSON: {path}") from error
    if not isinstance(value, dict):
        raise ContributionStorageError(f"control record is not an object: {path}")
    return value


@contextlib.contextmanager
def _control_lock(root: Path, host_id: str | None = None) -> Iterator[None]:
    lock_name = ".seal.lock" if host_id is None else f".host-{host_id}.lock"
    lock_dir = root / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / lock_name
    with path.open("a+b", buffering=0) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _nested_get(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    selected: Any = value
    for name in path:
        selected = selected[name]
    return selected


def _statistics_to_tensors(value: Mapping[str, Any]) -> dict[str, np.ndarray]:
    statistics = validate_contribution_statistics(value)
    tensors: dict[str, np.ndarray] = {}
    for name, (path, dtype, shape) in _STAT_SPECS.items():
        array = np.asarray(_nested_get(statistics, path), dtype=dtype)
        if shape == (1,) and array.shape == ():
            array = array.reshape(1)
        if array.shape != shape:
            raise ContributionIntegrityError(
                f"statistic {'.'.join(path)} has the wrong tensor shape"
            )
        tensors[name] = np.ascontiguousarray(array)
    return tensors


def _statistics_from_tensors(
    tensors: Mapping[str, np.ndarray],
    *,
    expected_tokens: int,
) -> dict[str, Any]:
    values: dict[str, np.ndarray] = {}
    for name, (_path, dtype, shape) in _STAT_SPECS.items():
        if name not in tensors:
            raise ContributionIntegrityError(f"safetensors lacks statistic {name}")
        array = np.asarray(tensors[name])
        if array.dtype != dtype or array.shape != shape:
            raise ContributionIntegrityError(
                f"safetensors statistic {name} has dtype/shape "
                f"{array.dtype}/{array.shape}, expected {dtype}/{shape}"
            )
        values[name] = array

    def scalar(name: str) -> float:
        return float(values[name][0])

    def integer(name: str) -> int:
        return int(values[name][0])

    result = {
        "token_count": integer("__stats_token_count"),
        "component_count": integer("__stats_component_count"),
        "pooled_expert_output": {
            "count": integer("__stats_pooled_count"),
            "sum": values["__stats_pooled_sum"].tolist(),
            "squared_norm": scalar("__stats_pooled_squared_norm"),
        },
        "weighted_contribution": {
            "weight_squared_sum": scalar("__stats_weight_squared_sum"),
            "weight_squared_output_sum": values[
                "__stats_weight_squared_output_sum"
            ].tolist(),
            "squared_norm": scalar("__stats_weighted_squared_norm"),
            "raw_sum": values["__stats_weighted_raw_sum"].tolist(),
        },
        "router_rank": {
            "count": values["__stats_router_rank_count"].tolist(),
            "energy": values["__stats_router_rank_energy"].tolist(),
        },
        "expert_id": {
            "count": values["__stats_expert_id_count"].tolist(),
            "energy": values["__stats_expert_id_energy"].tolist(),
        },
        "shared_raw": {
            "count": integer("__stats_shared_raw_count"),
            "sum": values["__stats_shared_raw_sum"].tolist(),
            "squared_norm": scalar("__stats_shared_raw_squared_norm"),
        },
        "shared_normalized": {
            "count": integer("__stats_shared_normalized_count"),
            "sum": values["__stats_shared_normalized_sum"].tolist(),
            "squared_norm": scalar("__stats_shared_normalized_squared_norm"),
        },
    }
    return validate_contribution_statistics(result, expected_tokens=expected_tokens)


def _file_metadata(
    *,
    metadata_sha256: str,
    host_id: str,
    range_id: str,
    range_start: int,
    range_end: int,
    source_v1_range_start: int,
    source_v1_range_end: int,
    token_digest: str,
    source_v1_manifest_sha256: str,
    recomposition_sidecar_sha256: str,
) -> dict[str, str]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "metadata_sha256": metadata_sha256,
        "host_id": host_id,
        "range_id": range_id,
        "range_start": str(range_start),
        "range_end": str(range_end),
        "source_v1_range_start": str(source_v1_range_start),
        "source_v1_range_end": str(source_v1_range_end),
        "token_identity_sha256": token_digest,
        "source_v1_manifest_sha256": source_v1_manifest_sha256,
        "recomposition_sidecar_sha256": recomposition_sidecar_sha256,
        "expert_output_logical_dtype": "bfloat16",
        "router_weights_logical_dtype": "bfloat16",
        "shared_output_logical_dtype": "bfloat16",
        "statistics_reduction_dtype": "float64",
    }


def _receipt_path(
    root: Path,
    *,
    host_id: str,
    metadata_sha256: str,
    range_id: str,
    range_start: int,
    range_end: int,
) -> Path:
    identity = canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "metadata_sha256": metadata_sha256,
            "host_id": host_id,
            "range_id": range_id,
            "range_start": range_start,
            "range_end": range_end,
        }
    )
    return root / "receipts" / f"host-{host_id}" / f"{identity}.json"


def _host_index_path(root: Path, host_id: str) -> Path:
    return root / "hosts" / f"host-{host_id}.json"


def _validate_storage_range(
    range_start: Any,
    range_end: Any,
    *,
    max_tokens: int = MAX_SHARD_TOKENS,
) -> tuple[int, int]:
    if (
        isinstance(range_start, bool)
        or not isinstance(range_start, int)
        or isinstance(range_end, bool)
        or not isinstance(range_end, int)
        or range_start < 0
        or range_end <= range_start
        or range_end - range_start > max_tokens
    ):
        raise ContributionRangeError(
            f"range must be nonempty, nonnegative, and at most {max_tokens} tokens"
        )
    return range_start, range_end


def _self_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise ContributionStorageError(f"{field} is already present")
    unsigned = _canonical_copy(value)
    return {**unsigned, field: canonical_sha256(unsigned)}


def _validate_self_hash(
    value: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> dict[str, Any]:
    record = _canonical_copy(value)
    claimed = record.get(field)
    if not isinstance(claimed, str) or _SHA256_RE.fullmatch(claimed) is None:
        raise ContributionStorageError(f"{label} has no canonical {field}")
    unsigned = dict(record)
    del unsigned[field]
    if canonical_sha256(unsigned) != claimed:
        raise ContributionStorageError(f"{label} self-hash mismatch")
    return record


def _validate_host_index(
    value: Mapping[str, Any],
    *,
    expected_host_id: str,
    expected_metadata_sha256: str,
) -> dict[str, Any]:
    record = _validate_self_hash(
        value, field="index_sha256", label="contribution host index"
    )
    if set(record) != {
        "schema_version",
        "contribution_schema_version",
        "protocol_id",
        "host_id",
        "metadata_sha256",
        "complete",
        "receipts",
        "index_sha256",
    }:
        raise ContributionStorageError("contribution host index fields changed")
    if (
        record["schema_version"] != HOST_INDEX_SCHEMA_VERSION
        or record["contribution_schema_version"] != SCHEMA_VERSION
        or record["protocol_id"] != PROTOCOL_ID
        or record["host_id"] != expected_host_id
        or record["metadata_sha256"] != expected_metadata_sha256
        or type(record["complete"]) is not bool
        or not isinstance(record["receipts"], list)
    ):
        raise ContributionStorageError("contribution host index identity changed")
    expected_reference_keys = {
        "range_id",
        "range_start",
        "range_end",
        "receipt_sha256",
    }
    previous: tuple[int, str] | None = None
    seen_ids: set[str] = set()
    for reference in record["receipts"]:
        if (
            not isinstance(reference, Mapping)
            or set(reference) != expected_reference_keys
        ):
            raise ContributionStorageError("host receipt reference fields changed")
        if (
            not isinstance(reference["range_id"], str)
            or isinstance(reference["range_start"], bool)
            or not isinstance(reference["range_start"], int)
            or isinstance(reference["range_end"], bool)
            or not isinstance(reference["range_end"], int)
            or reference["range_start"] < 0
            or reference["range_end"] <= reference["range_start"]
            or not isinstance(reference["receipt_sha256"], str)
            or _SHA256_RE.fullmatch(reference["receipt_sha256"]) is None
        ):
            raise ContributionStorageError("host receipt reference is invalid")
        if reference["range_id"] in seen_ids:
            raise ContributionRangeError("host index repeats a range_id")
        seen_ids.add(reference["range_id"])
        key = (reference["range_start"], reference["range_id"])
        if previous is not None and key <= previous:
            raise ContributionRangeError(
                "host receipt references are not in deterministic order"
            )
        previous = key
    return record


def _receipt_reference(receipt: Mapping[str, Any]) -> dict[str, Any]:
    entry = receipt["entry"]
    return {
        "range_id": entry["range_id"],
        "range_start": entry["range_start"],
        "range_end": entry["range_end"],
        "receipt_sha256": receipt["receipt_sha256"],
    }


def _validate_receipt(
    value: Mapping[str, Any],
    *,
    metadata_sha256: str,
    host_id: str,
    range_id: str,
    range_start: int,
    range_end: int,
    object_prefix: str,
    object_store: ImmutableObjectStore | None,
) -> dict[str, Any]:
    receipt = _validate_self_hash(
        value, field="receipt_sha256", label="contribution upload receipt"
    )
    if set(receipt) != {
        "schema_version",
        "contribution_schema_version",
        "protocol_id",
        "metadata_sha256",
        "host_id",
        "range_id",
        "range_start",
        "range_end",
        "entry_sha256",
        "entry",
        "object",
        "receipt_sha256",
    }:
        raise ContributionStorageError("contribution receipt fields changed")
    entry = receipt["entry"]
    if not isinstance(entry, Mapping):
        raise ContributionStorageError("contribution receipt entry is not an object")
    if (
        receipt["schema_version"] != RECEIPT_SCHEMA_VERSION
        or receipt["contribution_schema_version"] != SCHEMA_VERSION
        or receipt["protocol_id"] != PROTOCOL_ID
        or receipt["metadata_sha256"] != metadata_sha256
        or receipt["host_id"] != host_id
        or receipt["range_id"] != range_id
        or receipt["range_start"] != range_start
        or receipt["range_end"] != range_end
        or receipt["entry_sha256"] != canonical_sha256(entry)
        or entry.get("metadata_sha256") != metadata_sha256
        or entry.get("host_id") != host_id
        or entry.get("range_id") != range_id
        or entry.get("range_start") != range_start
        or entry.get("range_end") != range_end
    ):
        raise ContributionStorageError("contribution receipt identity changed")
    identity = object_identity_from_mapping(receipt["object"])
    expected_uri = join_gcs_uri(object_prefix, *str(entry["path"]).split("/"))
    if (
        identity.uri != expected_uri
        or identity.sha256 != entry.get("sha256")
        or identity.size_bytes <= 0
    ):
        raise ContributionStorageError(
            "contribution receipt object differs from its shard entry"
        )
    if object_store is not None:
        object_store.verify(identity)
    return receipt


class ContributionShardWriter:
    """Monotonically publish v2 ranges for one capture host."""

    def __init__(
        self,
        root: str | Path,
        host_id: str,
        metadata: ContributionMetadata | Mapping[str, Any],
        *,
        local_buffer_root: str | Path,
        object_prefix: str,
        object_store: ImmutableObjectStore,
        max_tokens: int = MAX_SHARD_TOKENS,
    ) -> None:
        if not isinstance(host_id, str) or _HOST_RE.fullmatch(host_id) is None:
            raise ContributionStorageError(f"unsafe contribution host_id {host_id!r}")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 0 < max_tokens <= MAX_SHARD_TOKENS
        ):
            raise ContributionStorageError(
                f"max_tokens must be in [1,{MAX_SHARD_TOKENS}]"
            )
        required_operations = ("put_if_absent", "download", "verify", "delete")
        if any(
            not callable(getattr(object_store, operation, None))
            for operation in required_operations
        ):
            raise ContributionStorageError(
                "object_store does not implement the immutable object-store contract"
            )
        # join_gcs_uri is the canonical prefix validator.
        join_gcs_uri(object_prefix, "identity-probe")
        self.root = Path(root).expanduser().resolve()
        self.host_id = host_id
        self.metadata = validate_metadata(
            metadata.to_dict()
            if isinstance(metadata, ContributionMetadata)
            else metadata
        )
        self.metadata_sha256 = canonical_sha256(self.metadata)
        self.max_tokens = max_tokens
        self.object_prefix = object_prefix.rstrip("/")
        self.object_store = object_store
        self.staging_dir = (
            Path(local_buffer_root).expanduser().resolve()
            / "contribution-shards"
            / self.metadata_sha256
            / f"host-{host_id}"
        )
        self.intent_dir = self.staging_dir / "intents"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.intent_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "hosts").mkdir(parents=True, exist_ok=True)
        (self.root / "receipts" / f"host-{host_id}").mkdir(parents=True, exist_ok=True)
        with _control_lock(self.root, self.host_id):
            manifest_path = self.root / MANIFEST_NAME
            if manifest_path.exists():
                raise ContributionStorageError(
                    "contribution artifact store is already sealed"
                )
            index_path = _host_index_path(self.root, self.host_id)
            if index_path.exists():
                _validate_host_index(
                    _read_json(index_path),
                    expected_host_id=self.host_id,
                    expected_metadata_sha256=self.metadata_sha256,
                )

    def range_id(
        self,
        range_start: int,
        range_end: int,
        sequence_id: Any,
        position: Any,
        *,
        source_v1_range_start: int,
        source_v1_range_end: int,
    ) -> str:
        _validate_storage_range(range_start, range_end, max_tokens=self.max_tokens)
        _validate_storage_range(
            source_v1_range_start,
            source_v1_range_end,
            max_tokens=self.max_tokens,
        )
        if source_v1_range_end - source_v1_range_start != range_end - range_start:
            raise ContributionRangeError(
                "logical and source-v1 ranges must contain the same token count"
            )
        token_digest = token_identity_sha256(sequence_id, position)
        return deterministic_range_id(
            split=self.metadata["split"],
            range_start=range_start,
            range_end=range_end,
            source_v1_range_start=source_v1_range_start,
            source_v1_range_end=source_v1_range_end,
            source_v1_manifest_sha256=self.metadata["source_v1_manifest_sha256"],
            token_identity_sha256=token_digest,
        )

    def _intent_path(self, range_id: str, range_start: int, range_end: int) -> Path:
        digest = canonical_sha256(
            {
                "range_id": range_id,
                "range_start": range_start,
                "range_end": range_end,
                "metadata_sha256": self.metadata_sha256,
            }
        )
        return self.intent_dir / f"{digest}.json"

    def _staged_path(self, range_id: str) -> Path:
        safe_range = re.sub(r"[^A-Za-z0-9_.-]", "_", range_id)
        return self.staging_dir / f"{safe_range}.safetensors"

    def _read_index(self) -> dict[str, Any] | None:
        path = _host_index_path(self.root, self.host_id)
        if not path.exists():
            return None
        return _validate_host_index(
            _read_json(path),
            expected_host_id=self.host_id,
            expected_metadata_sha256=self.metadata_sha256,
        )

    def _write_index(self, index: Mapping[str, Any]) -> None:
        unsigned = dict(index)
        unsigned.pop("index_sha256", None)
        value = _self_hash(unsigned, "index_sha256")
        _atomic_json(_host_index_path(self.root, self.host_id), value)

    def _find_reference(
        self,
        index: Mapping[str, Any] | None,
        range_id: str,
        range_start: int,
        range_end: int,
    ) -> Mapping[str, Any] | None:
        if index is None:
            return None
        for reference in index["receipts"]:
            same_id = reference["range_id"] == range_id
            overlaps = max(range_start, reference["range_start"]) < min(
                range_end, reference["range_end"]
            )
            if same_id:
                if (
                    reference["range_start"] != range_start
                    or reference["range_end"] != range_end
                ):
                    raise ContributionRangeError(
                        "range_id is already bound to different bounds"
                    )
                return reference
            if overlaps:
                raise ContributionRangeError(
                    "contribution range overlaps an already committed range"
                )
        return None

    def _load_receipt(
        self,
        range_id: str,
        range_start: int,
        range_end: int,
        *,
        verify_object: bool,
    ) -> dict[str, Any] | None:
        path = _receipt_path(
            self.root,
            host_id=self.host_id,
            metadata_sha256=self.metadata_sha256,
            range_id=range_id,
            range_start=range_start,
            range_end=range_end,
        )
        if not path.exists():
            return None
        return _validate_receipt(
            _read_json(path),
            metadata_sha256=self.metadata_sha256,
            host_id=self.host_id,
            range_id=range_id,
            range_start=range_start,
            range_end=range_end,
            object_prefix=self.object_prefix,
            object_store=self.object_store if verify_object else None,
        )

    def _commit_receipt(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        reference = _receipt_reference(receipt)
        with _control_lock(self.root, self.host_id):
            if (self.root / MANIFEST_NAME).exists():
                raise ContributionStorageError(
                    "contribution store sealed before range commit"
                )
            index = self._read_index()
            existing = self._find_reference(
                index,
                reference["range_id"],
                reference["range_start"],
                reference["range_end"],
            )
            if existing is not None:
                if dict(existing) != reference:
                    raise ContributionStorageError(
                        "host index differs from its immutable receipt"
                    )
                return dict(receipt["entry"])
            if index is None:
                index = {
                    "schema_version": HOST_INDEX_SCHEMA_VERSION,
                    "contribution_schema_version": SCHEMA_VERSION,
                    "protocol_id": PROTOCOL_ID,
                    "host_id": self.host_id,
                    "metadata_sha256": self.metadata_sha256,
                    "complete": False,
                    "receipts": [],
                }
            elif index["complete"] is True:
                raise ContributionStorageError(
                    f"contribution host {self.host_id!r} is finalized"
                )
            unsigned = dict(index)
            unsigned.pop("index_sha256", None)
            unsigned["receipts"] = [*index["receipts"], reference]
            unsigned["receipts"].sort(
                key=lambda item: (item["range_start"], item["range_id"])
            )
            self._write_index(unsigned)
            return dict(receipt["entry"])

    def _recover_intent(
        self, range_id: str, range_start: int, range_end: int
    ) -> dict[str, Any] | None:
        path = self._intent_path(range_id, range_start, range_end)
        if not path.exists():
            return None
        intent = _validate_self_hash(
            _read_json(path),
            field="intent_sha256",
            label="contribution upload intent",
        )
        if set(intent) != {
            "schema_version",
            "metadata_sha256",
            "host_id",
            "range_id",
            "range_start",
            "range_end",
            "entry",
            "object_uri",
            "intent_sha256",
        }:
            raise ContributionStorageError("contribution upload intent fields changed")
        entry = intent["entry"]
        if (
            intent["schema_version"] != UPLOAD_INTENT_SCHEMA_VERSION
            or intent["metadata_sha256"] != self.metadata_sha256
            or intent["host_id"] != self.host_id
            or intent["range_id"] != range_id
            or intent["range_start"] != range_start
            or intent["range_end"] != range_end
            or not isinstance(entry, Mapping)
        ):
            raise ContributionStorageError(
                "contribution upload intent identity changed"
            )
        expected_uri = join_gcs_uri(self.object_prefix, *str(entry["path"]).split("/"))
        if intent["object_uri"] != expected_uri:
            raise ContributionStorageError("upload intent object URI changed")
        staged = self._staged_path(range_id)
        if not staged.exists():
            # Host-local staging may disappear with a preempted worker.  The
            # deterministic replay path will regenerate the same content.
            return None
        fingerprint = fingerprint_file(staged)
        if fingerprint.sha256 != entry["sha256"]:
            raise ContributionStorageError("staged contribution intent is corrupt")
        identity = self.object_store.put_if_absent(
            staged,
            expected_uri,
            sha256=fingerprint.sha256,
            source_fingerprint=fingerprint,
        )
        receipt = self._publish_receipt(entry, identity)
        committed = self._commit_receipt(receipt)
        path.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)
        return committed

    def has_range(self, range_id: str, range_start: int, range_end: int) -> bool:
        """Return true only for an exact durable receipt and host-index commit."""

        _validate_storage_range(range_start, range_end, max_tokens=self.max_tokens)
        if not isinstance(range_id, str) or not range_id:
            raise ContributionRangeError("range_id must be nonempty")
        with _control_lock(self.root, self.host_id):
            index = self._read_index()
            reference = self._find_reference(index, range_id, range_start, range_end)
        receipt = self._load_receipt(
            range_id, range_start, range_end, verify_object=True
        )
        if reference is not None:
            if (
                receipt is None
                or reference["receipt_sha256"] != receipt["receipt_sha256"]
            ):
                raise ContributionStorageError(
                    "committed contribution range lacks its exact receipt"
                )
            return True
        if receipt is not None:
            self._commit_receipt(receipt)
            return True
        recovered = self._recover_intent(range_id, range_start, range_end)
        return recovered is not None

    def _publish_receipt(
        self, entry: Mapping[str, Any], identity: ObjectIdentity
    ) -> dict[str, Any]:
        unsigned = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "contribution_schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "metadata_sha256": self.metadata_sha256,
            "host_id": self.host_id,
            "range_id": entry["range_id"],
            "range_start": entry["range_start"],
            "range_end": entry["range_end"],
            "entry_sha256": canonical_sha256(entry),
            "entry": _canonical_copy(entry),
            "object": identity.as_dict(),
        }
        receipt = _self_hash(unsigned, "receipt_sha256")
        path = _receipt_path(
            self.root,
            host_id=self.host_id,
            metadata_sha256=self.metadata_sha256,
            range_id=str(entry["range_id"]),
            range_start=int(entry["range_start"]),
            range_end=int(entry["range_end"]),
        )
        _write_once_json(path, receipt)
        return _validate_receipt(
            _read_json(path),
            metadata_sha256=self.metadata_sha256,
            host_id=self.host_id,
            range_id=str(entry["range_id"]),
            range_start=int(entry["range_start"]),
            range_end=int(entry["range_end"]),
            object_prefix=self.object_prefix,
            object_store=self.object_store,
        )

    def adopt_receipt(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Adopt one prior phase's verified object without uploading it again."""

        if not isinstance(value, Mapping):
            raise ContributionStorageError("adopted receipt must be a mapping")
        try:
            range_id = str(value["range_id"])
            range_start = int(value["range_start"])
            range_end = int(value["range_end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ContributionStorageError(
                "adopted receipt lacks a valid range identity"
            ) from error
        checked = _validate_receipt(
            value,
            metadata_sha256=self.metadata_sha256,
            host_id=self.host_id,
            range_id=range_id,
            range_start=range_start,
            range_end=range_end,
            object_prefix=self.object_prefix,
            object_store=self.object_store,
        )
        entry = dict(checked["entry"])
        identity = object_identity_from_mapping(checked["object"])
        local_receipt = self._publish_receipt(entry, identity)
        return self._commit_receipt(local_receipt)

    def write_range(
        self,
        range_id: str,
        range_start: int,
        range_end: int,
        batch: ContributionBatch | Mapping[str, Any],
        *,
        source_v1_range_start: int,
        source_v1_range_end: int,
        normalized_shared_output: Any,
        source_sequence_id: Any,
        source_position: Any,
    ) -> dict[str, Any]:
        """Serialize, upload, receipt, and monotonically commit one exact range."""

        checked = validate_contribution_batch(batch, max_tokens=self.max_tokens)
        _validate_storage_range(range_start, range_end, max_tokens=self.max_tokens)
        if range_end - range_start != checked.tokens:
            raise ContributionRangeError(
                "contribution range length must equal its payload token count"
            )
        _validate_storage_range(
            source_v1_range_start,
            source_v1_range_end,
            max_tokens=self.max_tokens,
        )
        if source_v1_range_end - source_v1_range_start != checked.tokens:
            raise ContributionRangeError(
                "source-v1 range length must equal its payload token count"
            )
        token_digest = token_identity_sha256(checked.sequence_id, checked.position)
        expected_range_id = deterministic_range_id(
            split=self.metadata["split"],
            range_start=range_start,
            range_end=range_end,
            source_v1_range_start=source_v1_range_start,
            source_v1_range_end=source_v1_range_end,
            source_v1_manifest_sha256=self.metadata["source_v1_manifest_sha256"],
            token_identity_sha256=token_digest,
        )
        if range_id != expected_range_id:
            raise ContributionRangeError(
                "range_id is not bound to this source manifest, range, and token order"
            )
        source_probe = {
            "tokens": checked.tokens,
            "token_identity_sha256": token_digest,
        }
        validate_source_v1_token_keys(source_sequence_id, source_position, source_probe)
        if self.has_range(range_id, range_start, range_end):
            receipt = self._load_receipt(
                range_id, range_start, range_end, verify_object=True
            )
            assert receipt is not None
            return dict(receipt["entry"])

        statistics = compute_contribution_statistics(
            checked, normalized_shared_output=normalized_shared_output
        )
        tensors = {
            name: np.ascontiguousarray(np.asarray(getattr(checked, name)))
            for name in _PAYLOAD_FIELDS
        }
        tensors.update(_statistics_to_tensors(statistics))
        metadata = _file_metadata(
            metadata_sha256=self.metadata_sha256,
            host_id=self.host_id,
            range_id=range_id,
            range_start=range_start,
            range_end=range_end,
            source_v1_range_start=source_v1_range_start,
            source_v1_range_end=source_v1_range_end,
            token_digest=token_digest,
            source_v1_manifest_sha256=self.metadata["source_v1_manifest_sha256"],
            recomposition_sidecar_sha256=self.metadata["recomposition_sidecar_sha256"],
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="contribution-",
            suffix=".safetensors",
            dir=self.staging_dir,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            save_file(tensors, str(temporary), metadata=metadata)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            staged = self._staged_path(range_id)
            if staged.exists():
                # This can only be crash residue from the same deterministic
                # range. Compare the regenerated bytes before adopting it.
                regenerated_fingerprint = fingerprint_file(temporary)
                staged_fingerprint = fingerprint_file(staged)
                if (
                    staged_fingerprint.sha256 != regenerated_fingerprint.sha256
                    or staged_fingerprint.size_bytes
                    != regenerated_fingerprint.size_bytes
                ):
                    raise ContributionStorageError(
                        "deterministic staging path contains different bytes"
                    )
            else:
                os.replace(temporary, staged)
                _fsync_directory(staged.parent)
                # Hash only after the final rename because SourceFingerprint
                # binds ctime as well as inode/size/mtime.
                staged_fingerprint = fingerprint_file(staged)
            digest = staged_fingerprint.sha256
            relative = Path("shards") / f"host-{self.host_id}" / f"{digest}.safetensors"
            entry = {
                "range_id": range_id,
                "range_start": range_start,
                "range_end": range_end,
                "source_v1_range_start": source_v1_range_start,
                "source_v1_range_end": source_v1_range_end,
                "tokens": checked.tokens,
                "host_id": self.host_id,
                "path": relative.as_posix(),
                "sha256": digest,
                "token_identity_sha256": token_digest,
                "metadata_sha256": self.metadata_sha256,
                "source_v1_manifest_sha256": self.metadata["source_v1_manifest_sha256"],
                "recomposition_sidecar_sha256": self.metadata[
                    "recomposition_sidecar_sha256"
                ],
                "statistics": statistics,
            }
            validate_contribution_shard_payload(checked, entry)
            object_uri = join_gcs_uri(
                self.object_prefix, *relative.as_posix().split("/")
            )
            intent = _self_hash(
                {
                    "schema_version": UPLOAD_INTENT_SCHEMA_VERSION,
                    "metadata_sha256": self.metadata_sha256,
                    "host_id": self.host_id,
                    "range_id": range_id,
                    "range_start": range_start,
                    "range_end": range_end,
                    "entry": entry,
                    "object_uri": object_uri,
                },
                "intent_sha256",
            )
            intent_path = self._intent_path(range_id, range_start, range_end)
            _write_once_json(intent_path, intent)
            identity = self.object_store.put_if_absent(
                staged,
                object_uri,
                sha256=digest,
                source_fingerprint=staged_fingerprint,
            )
            receipt = self._publish_receipt(entry, identity)
            committed = self._commit_receipt(receipt)
            intent_path.unlink(missing_ok=True)
            staged.unlink(missing_ok=True)
            return committed
        finally:
            temporary.unlink(missing_ok=True)

    def finalize_host(self) -> dict[str, Any]:
        """Mark this host complete without rewriting full shard statistics."""

        with _control_lock(self.root, self.host_id):
            if (self.root / MANIFEST_NAME).exists():
                raise ContributionStorageError(
                    "contribution artifact store is already sealed"
                )
            index = self._read_index()
            if index is None or not index["receipts"]:
                raise ContributionStorageError(
                    f"contribution host {self.host_id!r} has no committed ranges"
                )
            if index["complete"] is True:
                return index
            unsigned = dict(index)
            unsigned.pop("index_sha256", None)
            unsigned["complete"] = True
            self._write_index(unsigned)
            checked = self._read_index()
            assert checked is not None
            return checked


@dataclass(frozen=True, slots=True)
class ContributionAdoptionResult:
    """Control-plane result of reusing one immutable prior phase."""

    prior_manifest_sha256: str
    host_id: str
    adopted_shards: int
    adopted_tokens: int


def adopt_contribution_phase(
    writer: ContributionShardWriter,
    prior_root: str | Path,
    *,
    recomposition_sidecar: Mapping[str, Any] | None = None,
) -> ContributionAdoptionResult:
    """Copy verified receipts into a new phase root without bulk object I/O."""

    if type(writer) is not ContributionShardWriter:
        raise ContributionStorageError(
            "phase adoption requires an exact ContributionShardWriter"
        )
    prior = load_contribution_manifest(
        prior_root,
        recomposition_sidecar=recomposition_sidecar,
    )
    if canonical_json_bytes(prior["metadata"]) != canonical_json_bytes(
        writer.metadata
    ):
        raise ContributionStorageError(
            "prior phase metadata differs from the extending writer"
        )
    if Path(prior_root).expanduser().resolve() == writer.root:
        raise ContributionStorageError(
            "a sealed phase cannot be adopted into its own control root"
        )
    selected = [
        entry for entry in prior["shards"] if entry["host_id"] == writer.host_id
    ]
    if not selected:
        raise ContributionStorageError(
            f"prior phase has no ranges for host {writer.host_id!r}"
        )
    adopted_tokens = 0
    for entry in selected:
        receipt_path = _receipt_path(
            Path(prior_root).expanduser().resolve(),
            host_id=writer.host_id,
            metadata_sha256=str(prior["metadata_sha256"]),
            range_id=str(entry["range_id"]),
            range_start=int(entry["range_start"]),
            range_end=int(entry["range_end"]),
        )
        receipt = _read_json(receipt_path)
        if receipt.get("entry") != entry:
            raise ContributionStorageError(
                "prior phase receipt differs from its sealed manifest"
            )
        adopted = writer.adopt_receipt(receipt)
        if adopted != entry:
            raise ContributionStorageError(
                "adopted phase entry changed during control-plane replay"
            )
        adopted_tokens += int(entry["tokens"])
    return ContributionAdoptionResult(
        prior_manifest_sha256=str(prior["manifest_sha256"]),
        host_id=writer.host_id,
        adopted_shards=len(selected),
        adopted_tokens=adopted_tokens,
    )


def _read_host_indexes(root: Path, *, metadata_sha256: str) -> list[dict[str, Any]]:
    directory = root / "hosts"
    if not directory.is_dir():
        return []
    indexes: list[dict[str, Any]] = []
    for path in sorted(directory.glob("host-*.json")):
        host_id = path.stem.removeprefix("host-")
        if _HOST_RE.fullmatch(host_id) is None:
            raise ContributionStorageError(f"unsafe host index path: {path}")
        indexes.append(
            _validate_host_index(
                _read_json(path),
                expected_host_id=host_id,
                expected_metadata_sha256=metadata_sha256,
            )
        )
    return indexes


def load_contribution_manifest(
    root: str | Path,
    *,
    recomposition_sidecar: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load only a sealed contribution-v2 manifest; reject aggregate v1."""

    path = Path(root).expanduser().resolve() / MANIFEST_NAME
    if not path.exists():
        raise ContributionStorageError(f"missing contribution manifest: {path}")
    value = _read_json(path, manifest=True)
    return validate_contribution_manifest(
        value, recomposition_sidecar=recomposition_sidecar
    )


def _validated_source_v1(
    source: ArtifactDataset,
    contribution_metadata: Mapping[str, Any],
    *,
    expected_tokens: int,
) -> tuple[ArtifactDataset, dict[str, Any]]:
    """Re-open and bind an authoritative sealed aggregate-v1 dataset.

    A four-field digest/count summary is not evidence that a corresponding
    source manifest exists.  Publication therefore reloads the complete v1
    manifest from the dataset root, ignores any caller-supplied manifest
    mapping, and constructs its own reader over those validated bytes.
    """

    if type(source) is not ArtifactDataset:
        raise ContributionStorageError(
            "source_v1_dataset must be an exact ArtifactDataset"
        )
    try:
        manifest = load_v1_manifest(source.root)
    except ArtifactError as error:
        raise ContributionStorageError(
            "source-v1 root is not a valid sealed aggregate manifest"
        ) from error
    if canonical_json_bytes(manifest) != canonical_json_bytes(source.manifest):
        raise ContributionStorageError(
            "source-v1 dataset manifest differs from its sealed root"
        )

    metadata = manifest["metadata"]
    required_matches = {
        "checkpoint_sha256": contribution_metadata["checkpoint_sha256"],
        "dataset_fingerprint": contribution_metadata["dataset_fingerprint"],
        "split": contribution_metadata["split"],
        "model_id": contribution_metadata["model_id"],
        "target_layer_one_indexed": contribution_metadata[
            "target_layer_one_indexed"
        ],
        "sequence_length": contribution_metadata["sequence_length"],
        "maxtext_commit": contribution_metadata["maxtext_base_commit"],
    }
    for name, expected in required_matches.items():
        if metadata.get(name) != expected:
            raise ContributionStorageError(
                f"source-v1 {name} differs from contribution metadata"
            )
    if (
        contribution_metadata["source_v1_manifest_sha256"]
        != manifest["manifest_sha256"]
    ):
        raise ContributionStorageError(
            "source-v1 manifest SHA-256 differs from contribution metadata"
        )
    if expected_tokens > int(manifest["total_tokens"]):
        raise ContributionStorageError(
            "contribution token count exceeds its sealed source-v1 manifest"
        )

    identity = {
        "schema_version": manifest["schema_version"],
        "manifest_sha256": manifest["manifest_sha256"],
        "split": metadata["split"],
        "total_tokens": manifest["total_tokens"],
    }
    # Reconstruct the reader from the validated root so a forged in-memory
    # mapping or overridden method cannot supply publication token keys.
    verified_reader = ArtifactDataset(
        source.root,
        verify_hashes=source.verify_hashes,
        verify_statistics=source.verify_statistics,
        decode_bf16=False,
        cache=source.cache,
    )
    return verified_reader, identity


def _source_v1_token_keys(
    source: ArtifactDataset,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    pieces = list(source.iter_token_keys_range(start, end))
    if not pieces:
        raise ContributionStorageError(
            f"source-v1 yielded no token keys for [{start},{end})"
        )
    return (
        np.concatenate([piece[0] for piece in pieces]),
        np.concatenate([piece[1] for piece in pieces]),
    )


def publish_contribution_manifest(
    root: str | Path,
    metadata: ContributionMetadata | Mapping[str, Any],
    *,
    source_v1_dataset: ArtifactDataset,
    object_prefix: str,
    object_store: ImmutableObjectStore,
    expected_hosts: int,
    expected_tokens: int,
    recomposition_sidecar: Mapping[str, Any] | None = None,
) -> Path:
    """Coordinator-verify every receipt/object/range/statistic, then seal."""

    if (
        isinstance(expected_hosts, bool)
        or not isinstance(expected_hosts, int)
        or expected_hosts <= 0
        or isinstance(expected_tokens, bool)
        or not isinstance(expected_tokens, int)
        or expected_tokens <= 0
    ):
        raise ContributionStorageError(
            "expected_hosts and expected_tokens must be positive integers"
        )
    join_gcs_uri(object_prefix, "identity-probe")
    selected_root = Path(root).expanduser().resolve()
    selected_metadata = validate_metadata(
        metadata.to_dict() if isinstance(metadata, ContributionMetadata) else metadata
    )
    source_reader, source_v1_manifest = _validated_source_v1(
        source_v1_dataset,
        selected_metadata,
        expected_tokens=expected_tokens,
    )
    metadata_sha256 = canonical_sha256(selected_metadata)
    manifest_path = selected_root / MANIFEST_NAME
    with _control_lock(selected_root):
        if manifest_path.exists():
            existing = load_contribution_manifest(
                selected_root, recomposition_sidecar=recomposition_sidecar
            )
            if (
                existing["metadata_sha256"] != metadata_sha256
                or existing["source_v1_manifest"] != dict(source_v1_manifest)
                or existing["total_tokens"] != expected_tokens
            ):
                raise ContributionStorageError(
                    "sealed contribution manifest differs from this publication"
                )
            return manifest_path

        indexes = _read_host_indexes(selected_root, metadata_sha256=metadata_sha256)
        if (
            len(indexes) != expected_hosts
            or len({index["host_id"] for index in indexes}) != expected_hosts
            or any(index["complete"] is not True for index in indexes)
        ):
            raise ContributionStorageError(
                "cannot seal before every expected host index is complete"
            )
        receipts: list[dict[str, Any]] = []
        for index in indexes:
            for reference in index["receipts"]:
                path = _receipt_path(
                    selected_root,
                    host_id=index["host_id"],
                    metadata_sha256=metadata_sha256,
                    range_id=reference["range_id"],
                    range_start=reference["range_start"],
                    range_end=reference["range_end"],
                )
                receipt = _validate_receipt(
                    _read_json(path),
                    metadata_sha256=metadata_sha256,
                    host_id=index["host_id"],
                    range_id=reference["range_id"],
                    range_start=reference["range_start"],
                    range_end=reference["range_end"],
                    object_prefix=object_prefix.rstrip("/"),
                    object_store=object_store,
                )
                if receipt["receipt_sha256"] != reference["receipt_sha256"]:
                    raise ContributionStorageError(
                        "host index and upload receipt hashes disagree"
                    )
                receipts.append(receipt)
        entries = [dict(receipt["entry"]) for receipt in receipts]
        entries.sort(key=lambda item: (item["range_start"], item["range_id"]))
        if not entries:
            raise ContributionStorageError("cannot seal an empty contribution store")
        for entry in entries:
            try:
                source_sequence, source_position = _source_v1_token_keys(
                    source_reader,
                    int(entry["source_v1_range_start"]),
                    int(entry["source_v1_range_end"]),
                )
            except (ArtifactError, OSError, ValueError) as error:
                raise ContributionStorageError(
                    "cannot read source-v1 token keys for contribution sealing"
                ) from error
            validate_source_v1_token_keys(source_sequence, source_position, entry)
        unsigned = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "sealed": True,
            "metadata": selected_metadata,
            "metadata_sha256": metadata_sha256,
            "source_v1_manifest": _canonical_copy(source_v1_manifest),
            "recomposition_sidecar_sha256": selected_metadata[
                "recomposition_sidecar_sha256"
            ],
            "total_tokens": sum(int(entry["tokens"]) for entry in entries),
            "shard_count": len(entries),
            "statistics": merge_contribution_statistics(
                [entry["statistics"] for entry in entries]
            ),
            "shards": entries,
        }
        sealed = seal_contribution_manifest(unsigned)
        validated = validate_contribution_manifest(
            sealed, recomposition_sidecar=recomposition_sidecar
        )
        if validated["total_tokens"] != expected_tokens:
            raise ContributionStorageError(
                f"captured {validated['total_tokens']} contribution tokens, "
                f"expected {expected_tokens}"
            )
        _write_once_json(manifest_path, validated)
        persisted = load_contribution_manifest(
            selected_root, recomposition_sidecar=recomposition_sidecar
        )
        if canonical_json_bytes(persisted) != canonical_json_bytes(validated):
            raise ContributionStorageError(
                "sealed contribution manifest failed read-after-write"
            )
        return manifest_path


class ContributionDataset:
    """Hash-verifying, cache-backed reader for sealed contribution v2."""

    def __init__(
        self,
        root: str | Path,
        *,
        object_prefix: str,
        cache: ShardCache,
        recomposition_sidecar: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(cache, ShardCache):
            raise ContributionStorageError(
                "contribution reader requires an exact ShardCache"
            )
        join_gcs_uri(object_prefix, "identity-probe")
        self.root = Path(root).expanduser().resolve()
        self.object_prefix = object_prefix.rstrip("/")
        self.cache = cache
        self.manifest = load_contribution_manifest(
            self.root, recomposition_sidecar=recomposition_sidecar
        )
        self.metadata_sha256 = str(self.manifest["metadata_sha256"])
        # ShardCache proves the complete generation-bound SHA-256 before it
        # returns a path.  Training then asks for many adjacent slices of the
        # same ~0.8 GiB shard.  Re-hashing that immutable file for every
        # 1,024-row microbatch would turn a sequential reader into terabytes
        # of redundant host I/O.  Retain only a process-local, stat-bound
        # proof: replacement, mutation, eviction, or truncation changes this
        # state and forces ShardCache to re-prove the object.
        self._verified_local_objects: dict[
            tuple[str, int], tuple[Path, tuple[int, int, int, int, int]]
        ] = {}

    @property
    def total_tokens(self) -> int:
        return int(self.manifest["total_tokens"])

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest["manifest_sha256"])

    def _receipt(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        path = _receipt_path(
            self.root,
            host_id=str(entry["host_id"]),
            metadata_sha256=self.metadata_sha256,
            range_id=str(entry["range_id"]),
            range_start=int(entry["range_start"]),
            range_end=int(entry["range_end"]),
        )
        receipt = _validate_receipt(
            _read_json(path),
            metadata_sha256=self.metadata_sha256,
            host_id=str(entry["host_id"]),
            range_id=str(entry["range_id"]),
            range_start=int(entry["range_start"]),
            range_end=int(entry["range_end"]),
            object_prefix=self.object_prefix,
            object_store=None,
        )
        if receipt["entry"] != dict(entry):
            raise ContributionStorageError(
                "sealed manifest entry differs from its upload receipt"
            )
        return receipt

    def _materialized_shard(
        self,
        entry: Mapping[str, Any],
    ) -> tuple[ObjectIdentity, Path, dict[str, str]]:
        receipt = self._receipt(entry)
        identity = object_identity_from_mapping(receipt["object"])
        key = (identity.uri, identity.generation)
        cached = self._verified_local_objects.get(key)
        path: Path
        if cached is not None and self._local_file_state(cached[0]) == cached[1]:
            path = cached[0]
        else:
            path = self.cache.materialize(identity)
            state = self._local_file_state(path)
            if state is None or state[2] != identity.size_bytes:
                raise ContributionStorageError(
                    "materialized contribution object has an invalid local state"
                )
            self._verified_local_objects[key] = (path, state)
        expected_metadata = self._inspect_materialized_shard(entry, path)
        return identity, path, expected_metadata

    @staticmethod
    def _local_file_state(
        path: Path,
    ) -> tuple[int, int, int, int, int] | None:
        """Return the state to which one in-process SHA proof is bound."""

        try:
            state = path.stat()
        except FileNotFoundError:
            return None
        if not path.is_file() or path.is_symlink():
            return None
        return (
            int(state.st_dev),
            int(state.st_ino),
            int(state.st_size),
            int(state.st_mtime_ns),
            int(state.st_ctime_ns),
        )

    def _inspect_materialized_shard(
        self,
        entry: Mapping[str, Any],
        path: Path,
    ) -> dict[str, str]:
        """Validate the exact v2 tensor set and scientific file metadata."""

        try:
            with safe_open(path, framework="numpy") as handle:
                keys = frozenset(handle.keys())
                metadata = handle.metadata() or {}
        except Exception as error:
            raise ContributionStorageError(
                f"cannot inspect contribution safetensors object: {path}"
            ) from error
        if metadata.get("schema_version") == LEGACY_AGGREGATE_SCHEMA_VERSION:
            raise ContributionLegacySchemaError(
                "aggregate v1 shard cannot be read as contribution v2"
            )
        expected_metadata = _file_metadata(
            metadata_sha256=self.metadata_sha256,
            host_id=str(entry["host_id"]),
            range_id=str(entry["range_id"]),
            range_start=int(entry["range_start"]),
            range_end=int(entry["range_end"]),
            source_v1_range_start=int(entry["source_v1_range_start"]),
            source_v1_range_end=int(entry["source_v1_range_end"]),
            token_digest=str(entry["token_identity_sha256"]),
            source_v1_manifest_sha256=str(entry["source_v1_manifest_sha256"]),
            recomposition_sidecar_sha256=str(entry["recomposition_sidecar_sha256"]),
        )
        if keys != _FILE_TENSORS or metadata != expected_metadata:
            raise ContributionStorageError(
                "contribution safetensors keys or metadata differ from v2"
            )
        return expected_metadata

    def _materialized_capacity_shards(
        self,
        entries: tuple[Mapping[str, Any], ...],
    ) -> tuple[tuple[Mapping[str, Any], Path], ...]:
        """Verify each immutable capacity-scan object exactly once.

        The ordinary training reader deliberately validates one requested
        shard at a time.  A capacity scan knows its complete process-local
        range up front, so batching the cache proof avoids re-hashing the same
        ~0.8 GB object for every 1,024-row microbatch.  ``materialize_many``
        provides the SHA-256/size proof; this method then independently checks
        the exact v2 tensor set and scientific file metadata.
        """

        if not entries:
            return ()
        identities: list[ObjectIdentity] = []
        for entry in entries:
            receipt = self._receipt(entry)
            identities.append(object_identity_from_mapping(receipt["object"]))
        try:
            paths = self.cache.materialize_many(tuple(identities), max_workers=4)
        except Exception as error:
            raise ContributionStorageError(
                "cannot materialize contribution capacity-scan objects"
            ) from error

        verified: list[tuple[Mapping[str, Any], Path]] = []
        for entry, path in zip(entries, paths, strict=True):
            self._inspect_materialized_shard(entry, path)
            verified.append((entry, path))
        return tuple(verified)

    def _load_capacity_slice(
        self,
        entry: Mapping[str, Any],
        path: Path,
        *,
        local_start: int,
        local_end: int,
        include_sequence_id: bool,
    ) -> ContributionCapacityBatch:
        """Memory-map only route IDs and, for R, packed-sequence IDs."""

        tokens = int(entry["tokens"])
        if not 0 <= local_start < local_end <= tokens:
            raise ContributionRangeError(
                "selective capacity shard slice is outside its sealed range"
            )
        try:
            with safe_open(path, framework="numpy") as handle:
                expert_ids = np.asarray(
                    handle.get_slice("expert_ids")[local_start:local_end]
                )
                sequence_id = (
                    np.asarray(handle.get_slice("sequence_id")[local_start:local_end])
                    if include_sequence_id
                    else None
                )
        except Exception as error:
            raise ContributionStorageError(
                f"cannot selectively load contribution safetensors object: {path}"
            ) from error
        return _validate_capacity_batch(
            ContributionCapacityBatch(
                expert_ids=expert_ids,
                sequence_id=sequence_id,
            ),
            expected_tokens=local_end - local_start,
            include_sequence_id=include_sequence_id,
        )

    def _load_shard(self, entry: Mapping[str, Any]) -> ContributionShard:
        identity, path, _metadata = self._materialized_shard(entry)
        try:
            tensors = load_file(path)
        except Exception as error:
            raise ContributionStorageError(
                f"cannot load contribution safetensors object: {path}"
            ) from error
        batch = validate_contribution_batch(
            ContributionBatch(
                **{name: np.asarray(tensors[name]) for name in _PAYLOAD_FIELDS}
            )
        )
        validate_contribution_shard_payload(batch, entry)
        statistics = _statistics_from_tensors(
            tensors, expected_tokens=int(entry["tokens"])
        )
        if canonical_json_bytes(statistics) != canonical_json_bytes(
            entry["statistics"]
        ):
            raise ContributionIntegrityError(
                "safetensors statistics differ from the sealed manifest"
            )
        return ContributionShard(
            range_id=str(entry["range_id"]),
            range_start=int(entry["range_start"]),
            range_end=int(entry["range_end"]),
            host_id=str(entry["host_id"]),
            sha256=str(entry["sha256"]),
            object=identity,
            statistics=statistics,
            batch=batch,
        )

    def _load_slice(
        self,
        entry: Mapping[str, Any],
        *,
        local_start: int,
        local_end: int,
    ) -> ContributionBatch:
        """Read only requested payload rows from one already-proved shard."""

        _identity, path, _metadata = self._materialized_shard(entry)
        tokens = int(entry["tokens"])
        if not 0 <= local_start < local_end <= tokens:
            raise ContributionRangeError(
                "selective contribution shard slice is outside its sealed range"
            )
        try:
            with safe_open(path, framework="numpy") as handle:
                batch = ContributionBatch(
                    **{
                        name: np.asarray(
                            handle.get_slice(name)[local_start:local_end]
                        )
                        for name in _PAYLOAD_FIELDS
                    }
                )
        except Exception as error:
            raise ContributionStorageError(
                f"cannot selectively load contribution safetensors object: {path}"
            ) from error
        return validate_contribution_batch(batch)

    def _load_shared_shard(
        self,
        entry: Mapping[str, Any],
    ) -> ContributionSharedShard:
        """Load only shared_output and scientific token keys from one shard."""

        identity, _path, _metadata = self._materialized_shard(entry)
        checked = self._load_shared_slice(
            entry,
            local_start=0,
            local_end=int(entry["tokens"]),
            expected_token_identity_sha256=str(entry["token_identity_sha256"]),
            materialized_identity=identity,
        )
        return ContributionSharedShard(
            range_id=str(entry["range_id"]),
            range_start=int(entry["range_start"]),
            range_end=int(entry["range_end"]),
            host_id=str(entry["host_id"]),
            sha256=str(entry["sha256"]),
            object=identity,
            batch=checked,
        )

    def _load_shared_slice(
        self,
        entry: Mapping[str, Any],
        *,
        local_start: int,
        local_end: int,
        expected_token_identity_sha256: str | None = None,
        materialized_identity: ObjectIdentity | None = None,
    ) -> ContributionSharedBatch:
        """Memory-map only the requested rows of Q's three safetensors."""

        identity = materialized_identity
        if identity is None:
            identity, path, _metadata = self._materialized_shard(entry)
        else:
            path = self.cache.root / f"{identity.sha256}.safetensors"
        del identity
        tokens = int(entry["tokens"])
        if not 0 <= local_start < local_end <= tokens:
            raise ContributionRangeError(
                "selective shared shard slice is outside its sealed range"
            )
        try:
            with safe_open(path, framework="numpy") as handle:
                batch = ContributionSharedBatch(
                    shared_output=np.asarray(
                        handle.get_slice("shared_output")[local_start:local_end]
                    ),
                    sequence_id=np.asarray(
                        handle.get_slice("sequence_id")[local_start:local_end]
                    ),
                    position=np.asarray(
                        handle.get_slice("position")[local_start:local_end]
                    ),
                )
        except Exception as error:
            raise ContributionStorageError(
                f"cannot selectively load contribution safetensors object: {path}"
            ) from error
        return _validate_shared_batch(
            batch,
            expected_tokens=local_end - local_start,
            expected_token_identity_sha256=expected_token_identity_sha256,
        )

    def iter_shards(self) -> Iterator[ContributionShard]:
        for entry in self.manifest["shards"]:
            yield self._load_shard(entry)

    def iter_batches(self) -> Iterator[ContributionBatch]:
        for shard in self.iter_shards():
            yield shard.batch

    def iter_shared_batches(self) -> Iterator[ContributionSharedBatch]:
        """Yield Q-only views without loading routed-expert payload tensors."""

        for entry in self.manifest["shards"]:
            yield self._load_shared_shard(entry).batch

    def prefetch_token_range(self, start: int, end: int) -> dict[str, int]:
        identities = self.token_range_object_identities(start, end)
        paths = self.cache.materialize_many(identities, max_workers=4)
        # This second pass is deliberately download-incapable.  It proves that
        # the complete range can be consumed under the later resident-only
        # training policy rather than trusting successful transfer futures.
        resident = self.cache.require_resident_many(identities)
        if paths != resident:
            raise ContributionStorageError(
                "prefetched range differs from its resident-only proof"
            )
        return {
            "objects": len(paths),
            "bytes": sum(path.stat().st_size for path in paths),
        }

    def token_range_object_identities(
        self,
        start: int,
        end: int,
    ) -> tuple[ObjectIdentity, ...]:
        """Return the exact immutable objects intersecting one token range."""

        return tuple(
            object_identity_from_mapping(self._receipt(entry)["object"])
            for entry in self._entries_for_range(start, end)
        )

    def require_resident_token_range(
        self,
        start: int,
        end: int,
    ) -> dict[str, int]:
        """Prove a range is complete without permitting an object-store read."""

        identities = self.token_range_object_identities(start, end)
        paths = self.cache.require_resident_many(identities)
        return {
            "objects": len(paths),
            "bytes": sum(path.stat().st_size for path in paths),
        }

    def _entries_for_range(self, start: int, end: int) -> list[Mapping[str, Any]]:
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not 0 <= start <= end <= self.total_tokens
        ):
            raise ContributionRangeError(
                f"token range must satisfy 0 <= start <= end <= {self.total_tokens}"
            )
        return [
            entry
            for entry in self.manifest["shards"]
            if max(start, int(entry["range_start"])) < min(end, int(entry["range_end"]))
        ]

    def iter_token_range(self, start: int, end: int) -> Iterator[ContributionBatch]:
        entries = self._entries_for_range(start, end)
        yielded = 0
        for entry in entries:
            range_start = int(entry["range_start"])
            range_end = int(entry["range_end"])
            local_start = max(start, range_start) - range_start
            local_end = min(end, range_end) - range_start
            selected = self._load_slice(
                entry,
                local_start=local_start,
                local_end=local_end,
            )
            yielded += selected.tokens
            yield validate_contribution_batch(selected)
        if yielded != end - start:
            raise ContributionRangeError(
                f"manifest yielded {yielded} contribution tokens for [{start},{end})"
            )

    def iter_capacity_token_range(
        self,
        start: int,
        end: int,
        *,
        include_sequence_id: bool,
    ) -> Iterator[ContributionCapacityBatch]:
        """Yield an exact range without touching dense contribution tensors.

        Safetensors offsets make ``expert_ids`` and ``sequence_id`` independently
        addressable.  Whole-object SHA-256 verification still binds these
        slices to the sealed contribution payload, so ``position`` is not
        needed merely to re-prove token identity during this derived scan.
        """

        if not isinstance(include_sequence_id, bool):
            raise ContributionRangeError("include_sequence_id must be an exact boolean")
        entries = tuple(self._entries_for_range(start, end))
        yielded = 0
        for entry, path in self._materialized_capacity_shards(entries):
            range_start = int(entry["range_start"])
            range_end = int(entry["range_end"])
            local_start = max(start, range_start) - range_start
            local_end = min(end, range_end) - range_start
            selected = self._load_capacity_slice(
                entry,
                path,
                local_start=local_start,
                local_end=local_end,
                include_sequence_id=include_sequence_id,
            )
            yielded += selected.tokens
            yield selected
        if yielded != end - start:
            raise ContributionRangeError(
                f"manifest yielded {yielded} capacity tokens for [{start},{end})"
            )

    def iter_shared_token_range(
        self,
        start: int,
        end: int,
    ) -> Iterator[ContributionSharedBatch]:
        """Yield a range while materializing only Q-required safetensors."""

        entries = self._entries_for_range(start, end)
        yielded = 0
        for entry in entries:
            range_start = int(entry["range_start"])
            range_end = int(entry["range_end"])
            local_start = max(start, range_start) - range_start
            local_end = min(end, range_end) - range_start
            selected = self._load_shared_slice(
                entry,
                local_start=local_start,
                local_end=local_end,
            )
            yielded += selected.tokens
            yield selected
        if yielded != end - start:
            raise ContributionRangeError(
                f"manifest yielded {yielded} shared tokens for [{start},{end})"
            )


# The longer name mirrors the v1 ArtifactWriter surface and eases migration
# without making either schema auto-detect the other.
ContributionArtifactWriter = ContributionShardWriter


__all__ = [
    "ContributionAdoptionResult",
    "ContributionArtifactWriter",
    "ContributionCapacityBatch",
    "ContributionDataset",
    "ContributionSharedBatch",
    "ContributionSharedShard",
    "ContributionShard",
    "ContributionShardWriter",
    "ContributionStorageError",
    "HOST_INDEX_SCHEMA_VERSION",
    "MANIFEST_NAME",
    "RECEIPT_SCHEMA_VERSION",
    "UPLOAD_INTENT_SCHEMA_VERSION",
    "adopt_contribution_phase",
    "load_contribution_manifest",
    "publish_contribution_manifest",
]
