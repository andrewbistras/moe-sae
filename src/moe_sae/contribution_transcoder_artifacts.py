"""Fail-closed routed-input overlay artifacts for Experiment 3.

The overlay is deliberately a separate scientific object from both the
aggregate-v1 and contribution-v2 captures.  It contains only the exact input
consumed by Gemma's routed experts plus ordered token keys, and binds every
shard one-to-one to a sealed Experiment-2 contribution range.

As in :mod:`moe_sae.contribution_artifacts`, BF16 is represented by exact
``uint16`` bit patterns at the NumPy/storage boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import math
import re
from typing import Any

import numpy as np

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    MAXTEXT_BASE_COMMIT,
    MAX_SHARD_TOKENS,
    MODEL_ID,
    SEQUENCE_LENGTH,
    canonical_json_bytes,
    canonical_sha256,
    bf16_bits_to_float32,
    token_identity_sha256,
    validate_contribution_manifest,
)


SCHEMA_VERSION = "gemma4-routed-input-overlay-v1"
PROTOCOL_ID = "gemma4-contribution-transcoder-v1-20260727"
CAPTURE_SITE = (
    "target.mlp.pre_feedforward_layernorm_2("
    "block15.post_attention_residual)"
)
BF16_LOGICAL_DTYPE = "bfloat16"
BF16_STORAGE_DTYPE = "uint16"

TENSOR_SCHEMA: dict[str, dict[str, Any]] = {
    "routed_input": {
        "logical_dtype": BF16_LOGICAL_DTYPE,
        "storage_dtype": BF16_STORAGE_DTYPE,
        "shape": ["N", HIDDEN_SIZE],
    },
    "sequence_id": {
        "logical_dtype": "int64",
        "storage_dtype": "int64",
        "shape": ["N"],
    },
    "position": {
        "logical_dtype": "uint16",
        "storage_dtype": "uint16",
        "shape": ["N"],
    },
}

_FIELDS = tuple(TENSOR_SCHEMA)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class RoutedInputArtifactError(RuntimeError):
    """Base class for routed-input overlay failures."""


class RoutedInputSchemaError(RoutedInputArtifactError):
    """An object does not satisfy the exact routed-input schema."""


class RoutedInputIntegrityError(RoutedInputArtifactError):
    """A hash, statistic, token key, or parent binding is inconsistent."""


class RoutedInputRangeError(RoutedInputIntegrityError):
    """A routed-input range is not the immutable parent range it claims."""


@dataclasses.dataclass(frozen=True, slots=True)
class RoutedInputBatch:
    """Exact NumPy wire representation of one overlay shard."""

    routed_input: np.ndarray
    sequence_id: np.ndarray
    position: np.ndarray

    @property
    def tokens(self) -> int:
        return int(np.asarray(self.routed_input).shape[0])


@dataclasses.dataclass(frozen=True, slots=True)
class RoutedInputMetadata:
    """Scientific identity shared by every shard in one overlay."""

    checkpoint_sha256: str
    maxtext_fork_commit: str
    dataset_fingerprint: str
    split: str
    config_sha256: str
    environment_sha256: str
    target_manifest_sha256: str
    protocol_id: str = PROTOCOL_ID
    model_id: str = MODEL_ID
    maxtext_base_commit: str = MAXTEXT_BASE_COMMIT
    target_layer_one_indexed: int = 15
    sequence_length: int = SEQUENCE_LENGTH
    capture_site: str = CAPTURE_SITE

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value.update(
            {
                "hidden_size": HIDDEN_SIZE,
                "max_shard_tokens": MAX_SHARD_TOKENS,
                "tensor_schema": TENSOR_SCHEMA,
            }
        )
        return validate_routed_input_metadata(value)


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RoutedInputSchemaError(
            f"{name} must be a lowercase SHA-256 hex digest"
        )
    return value


def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RoutedInputSchemaError(f"{name} must be a positive integer")
    return value


def _canonical_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(canonical_json_bytes(dict(value)))
    except Exception as error:
        raise RoutedInputSchemaError(
            "routed-input identities must be finite canonical JSON"
        ) from error


def _bf16_bits_all_finite(values: np.ndarray) -> bool:
    flat = np.asarray(values).reshape(-1)
    exponent_mask = np.uint16(0x7F80)
    block_elements = 8 * 1024 * 1024
    for start in range(0, int(flat.size), block_elements):
        block = flat[start : start + block_elements]
        if np.any(np.bitwise_and(block, exponent_mask) == exponent_mask):
            return False
    return True


def validate_routed_input_batch(
    value: RoutedInputBatch | Mapping[str, Any],
    *,
    max_tokens: int = MAX_SHARD_TOKENS,
) -> RoutedInputBatch:
    """Validate the exact three-field wire schema and ordered token keys."""

    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 0 < max_tokens <= MAX_SHARD_TOKENS
    ):
        raise RoutedInputSchemaError(
            f"max_tokens must be in [1,{MAX_SHARD_TOKENS}]"
        )
    if isinstance(value, RoutedInputBatch):
        batch = value
    elif isinstance(value, Mapping):
        if set(value) != set(_FIELDS):
            raise RoutedInputSchemaError(
                "routed-input tensor keys must be exactly "
                f"{sorted(_FIELDS)}, got {sorted(str(key) for key in value)}"
            )
        batch = RoutedInputBatch(
            **{name: np.asarray(value[name]) for name in _FIELDS}
        )
    else:
        raise RoutedInputSchemaError(
            "routed-input batch must be a RoutedInputBatch or mapping"
        )

    routed = np.asarray(batch.routed_input)
    sequence = np.asarray(batch.sequence_id)
    position = np.asarray(batch.position)
    if routed.dtype != np.uint16:
        raise RoutedInputSchemaError(
            f"routed_input must have wire dtype uint16, got {routed.dtype}"
        )
    if routed.ndim != 2 or routed.shape[1] != HIDDEN_SIZE:
        raise RoutedInputSchemaError(
            f"routed_input must have shape [N,{HIDDEN_SIZE}], got {routed.shape}"
        )
    tokens = int(routed.shape[0])
    if not 0 < tokens <= max_tokens:
        raise RoutedInputSchemaError(
            f"shard token count must be in [1,{max_tokens}], got {tokens}"
        )
    if sequence.dtype != np.int64 or sequence.shape != (tokens,):
        raise RoutedInputSchemaError(
            f"sequence_id must be int64 [{tokens}], got {sequence.dtype}/{sequence.shape}"
        )
    if position.dtype != np.uint16 or position.shape != (tokens,):
        raise RoutedInputSchemaError(
            f"position must be uint16 [{tokens}], got {position.dtype}/{position.shape}"
        )
    if not _bf16_bits_all_finite(routed):
        raise RoutedInputSchemaError("routed_input contains non-finite BF16 values")
    if np.any(sequence < 0):
        raise RoutedInputSchemaError("sequence_id must be nonnegative")
    if np.any(position >= SEQUENCE_LENGTH):
        raise RoutedInputSchemaError(
            f"position must be in [0,{SEQUENCE_LENGTH - 1}]"
        )
    keys = np.rec.fromarrays(
        [sequence, position], names=("sequence_id", "position")
    )
    if np.unique(keys).size != tokens:
        raise RoutedInputSchemaError(
            "a shard cannot contain duplicate (sequence_id, position) tokens"
        )
    return RoutedInputBatch(
        routed_input=np.ascontiguousarray(routed),
        sequence_id=np.ascontiguousarray(sequence),
        position=np.ascontiguousarray(position),
    )


def routed_input_token_identity_sha256(
    value: RoutedInputBatch | Mapping[str, Any],
) -> str:
    batch = validate_routed_input_batch(value)
    return token_identity_sha256(batch.sequence_id, batch.position)


def compute_routed_input_statistics(
    value: RoutedInputBatch | Mapping[str, Any],
) -> dict[str, Any]:
    """Return FP64 sufficient statistics without a capture-sized FP64 copy."""

    batch = validate_routed_input_batch(value)
    flat = batch.routed_input
    summed = np.zeros(HIDDEN_SIZE, dtype=np.float64)
    squared_norm = np.float64(0.0)
    # Bound the expanded FP32 temporary to roughly 64 MiB.
    rows_per_block = max(1, (64 << 20) // (HIDDEN_SIZE * 4))
    for start in range(0, batch.tokens, rows_per_block):
        decoded = bf16_bits_to_float32(flat[start : start + rows_per_block])
        decoded64 = decoded.astype(np.float64)
        summed += decoded64.sum(axis=0, dtype=np.float64)
        squared_norm += np.square(decoded64).sum(dtype=np.float64)
    return {
        "count": batch.tokens,
        "sum": summed.tolist(),
        "squared_norm": float(squared_norm),
    }


def validate_routed_input_statistics(
    value: Mapping[str, Any],
    *,
    expected_tokens: int | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "count",
        "sum",
        "squared_norm",
    }:
        raise RoutedInputIntegrityError(
            "routed-input statistics fields differ from the v1 contract"
        )
    count = value["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise RoutedInputIntegrityError("statistics count must be positive")
    if expected_tokens is not None and count != expected_tokens:
        raise RoutedInputIntegrityError(
            f"statistics count {count} differs from expected {expected_tokens}"
        )
    summed = np.asarray(value["sum"], dtype=np.float64)
    if summed.shape != (HIDDEN_SIZE,) or not np.isfinite(summed).all():
        raise RoutedInputIntegrityError(
            f"statistics sum must be a finite FP64 vector of length {HIDDEN_SIZE}"
        )
    squared = value["squared_norm"]
    if (
        isinstance(squared, bool)
        or not isinstance(squared, (int, float))
        or not math.isfinite(float(squared))
        or float(squared) < 0
    ):
        raise RoutedInputIntegrityError(
            "statistics squared_norm must be finite and nonnegative"
        )
    return {
        "count": count,
        "sum": summed.tolist(),
        "squared_norm": float(squared),
    }


def merge_routed_input_statistics(
    values: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge statistics in the caller's deterministic target-range order."""

    if not values:
        raise RoutedInputIntegrityError("cannot merge empty statistics")
    checked = [validate_routed_input_statistics(value) for value in values]
    summed = np.zeros(HIDDEN_SIZE, dtype=np.float64)
    for value in checked:
        summed += np.asarray(value["sum"], dtype=np.float64)
    merged = {
        "count": sum(int(value["count"]) for value in checked),
        "sum": summed.tolist(),
        "squared_norm": sum(float(value["squared_norm"]) for value in checked),
    }
    return validate_routed_input_statistics(
        merged, expected_tokens=int(merged["count"])
    )


def validate_routed_input_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RoutedInputSchemaError("routed-input metadata must be an object")
    expected = {
        "checkpoint_sha256",
        "maxtext_fork_commit",
        "dataset_fingerprint",
        "split",
        "config_sha256",
        "environment_sha256",
        "target_manifest_sha256",
        "protocol_id",
        "model_id",
        "maxtext_base_commit",
        "target_layer_one_indexed",
        "sequence_length",
        "capture_site",
        "hidden_size",
        "max_shard_tokens",
        "tensor_schema",
    }
    if set(value) != expected:
        raise RoutedInputSchemaError(
            "routed-input metadata fields differ from the v1 contract"
        )
    for name in (
        "checkpoint_sha256",
        "dataset_fingerprint",
        "config_sha256",
        "environment_sha256",
        "target_manifest_sha256",
    ):
        _require_sha256(name, value[name])
    if (
        not isinstance(value["maxtext_fork_commit"], str)
        or _GIT_COMMIT_RE.fullmatch(value["maxtext_fork_commit"]) is None
    ):
        raise RoutedInputSchemaError(
            "maxtext_fork_commit must be a lowercase 40-character Git commit"
        )
    if (
        not isinstance(value["split"], str)
        or _SAFE_NAME_RE.fullmatch(value["split"]) is None
    ):
        raise RoutedInputSchemaError("split must be a safe nonempty identity")
    fixed = {
        "protocol_id": PROTOCOL_ID,
        "model_id": MODEL_ID,
        "maxtext_base_commit": MAXTEXT_BASE_COMMIT,
        "target_layer_one_indexed": 15,
        "sequence_length": SEQUENCE_LENGTH,
        "capture_site": CAPTURE_SITE,
        "hidden_size": HIDDEN_SIZE,
        "max_shard_tokens": MAX_SHARD_TOKENS,
        "tensor_schema": TENSOR_SCHEMA,
    }
    for name, expected_value in fixed.items():
        if value[name] != expected_value:
            raise RoutedInputSchemaError(
                f"metadata {name} must be {expected_value!r}, got {value[name]!r}"
            )
    return _canonical_copy(value)


def deterministic_overlay_range_id(
    *,
    split: str,
    range_start: int,
    range_end: int,
    target_manifest_sha256: str,
    target_range_id: str,
    token_identity_sha256: str,
) -> str:
    """Bind an overlay range to one exact sealed contribution range."""

    if not isinstance(split, str) or _SAFE_NAME_RE.fullmatch(split) is None:
        raise RoutedInputRangeError("split is not a safe identity")
    if (
        isinstance(range_start, bool)
        or isinstance(range_end, bool)
        or not isinstance(range_start, int)
        or not isinstance(range_end, int)
        or range_start < 0
        or range_end <= range_start
        or range_end - range_start > MAX_SHARD_TOKENS
    ):
        raise RoutedInputRangeError(
            f"invalid overlay range [{range_start},{range_end})"
        )
    if (
        not isinstance(target_range_id, str)
        or _SAFE_NAME_RE.fullmatch(target_range_id) is None
    ):
        raise RoutedInputRangeError("target_range_id is unsafe or empty")
    target_sha = _require_sha256(
        "target_manifest_sha256", target_manifest_sha256
    )
    token_sha = _require_sha256(
        "token_identity_sha256", token_identity_sha256
    )
    digest = canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "range_start": range_start,
            "range_end": range_end,
            "target_manifest_sha256": target_sha,
            "target_range_id": target_range_id,
            "token_identity_sha256": token_sha,
        }
    )
    return f"{split}-{range_start:012d}-{range_end:012d}-{digest[:16]}"


def target_manifest_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    target = validate_contribution_manifest(value)
    return {
        "schema_version": target["schema_version"],
        "manifest_sha256": target["manifest_sha256"],
        "split": target["metadata"]["split"],
        "total_tokens": target["total_tokens"],
        "shard_count": target["shard_count"],
    }


def _validate_target_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "manifest_sha256",
        "split",
        "total_tokens",
        "shard_count",
    }:
        raise RoutedInputSchemaError(
            "target_manifest identity fields differ from the contract"
        )
    _require_sha256("target_manifest.manifest_sha256", value["manifest_sha256"])
    if (
        not isinstance(value["schema_version"], str)
        or not isinstance(value["split"], str)
        or _SAFE_NAME_RE.fullmatch(value["split"]) is None
    ):
        raise RoutedInputSchemaError("target manifest identity is invalid")
    _require_positive_int("target_manifest.total_tokens", value["total_tokens"])
    _require_positive_int("target_manifest.shard_count", value["shard_count"])
    return dict(value)


def _validate_shard_entry(
    value: Any,
    *,
    metadata: Mapping[str, Any],
    metadata_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RoutedInputIntegrityError("overlay shard entry must be an object")
    expected = {
        "range_id",
        "target_range_id",
        "range_start",
        "range_end",
        "tokens",
        "host_id",
        "path",
        "sha256",
        "token_identity_sha256",
        "metadata_sha256",
        "target_manifest_sha256",
        "statistics",
        "object",
    }
    if set(value) != expected:
        raise RoutedInputIntegrityError(
            "overlay shard entry fields differ from the v1 contract"
        )
    start = value["range_start"]
    end = value["range_end"]
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
        or end - start > MAX_SHARD_TOKENS
    ):
        raise RoutedInputRangeError(f"invalid overlay range [{start},{end})")
    if value["tokens"] != end - start:
        raise RoutedInputRangeError("entry tokens differ from its range length")
    token_sha = _require_sha256(
        "shard.token_identity_sha256", value["token_identity_sha256"]
    )
    if value["metadata_sha256"] != metadata_sha256:
        raise RoutedInputIntegrityError("shard metadata identity changed")
    if value["target_manifest_sha256"] != metadata["target_manifest_sha256"]:
        raise RoutedInputIntegrityError("shard target manifest identity changed")
    expected_range_id = deterministic_overlay_range_id(
        split=metadata["split"],
        range_start=start,
        range_end=end,
        target_manifest_sha256=metadata["target_manifest_sha256"],
        target_range_id=value["target_range_id"],
        token_identity_sha256=token_sha,
    )
    if value["range_id"] != expected_range_id:
        raise RoutedInputRangeError(
            "overlay range_id is not bound to its target range and token order"
        )
    if (
        not isinstance(value["host_id"], str)
        or _SAFE_NAME_RE.fullmatch(value["host_id"]) is None
    ):
        raise RoutedInputIntegrityError("shard host_id is invalid")
    digest = _require_sha256("shard.sha256", value["sha256"])
    expected_path = f"shards/host-{value['host_id']}/{digest}.safetensors"
    if value["path"] != expected_path:
        raise RoutedInputIntegrityError(
            "shard path must be host-specific and content-addressed"
        )
    obj = value["object"]
    if not isinstance(obj, Mapping) or set(obj) != {
        "uri",
        "generation",
        "size_bytes",
        "crc32c",
        "sha256",
    }:
        raise RoutedInputIntegrityError("shard object identity fields changed")
    if obj["sha256"] != digest:
        raise RoutedInputIntegrityError("shard and object SHA-256 differ")
    statistics = validate_routed_input_statistics(
        value["statistics"], expected_tokens=value["tokens"]
    )
    return {
        **dict(value),
        "statistics": statistics,
        "object": dict(obj),
    }


def seal_routed_input_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or "manifest_sha256" in value:
        raise RoutedInputSchemaError(
            "only an unsigned routed-input manifest may be sealed"
        )
    unsigned = dict(value)
    return {**unsigned, "manifest_sha256": canonical_sha256(unsigned)}


def validate_routed_input_manifest(
    value: Mapping[str, Any],
    *,
    target_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an overlay and, when supplied, its complete immutable parent."""

    if not isinstance(value, Mapping):
        raise RoutedInputSchemaError("routed-input manifest must be an object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise RoutedInputSchemaError(
            f"unsupported routed-input schema {value.get('schema_version')!r}"
        )
    expected = {
        "schema_version",
        "protocol_id",
        "sealed",
        "metadata",
        "metadata_sha256",
        "target_manifest",
        "total_tokens",
        "shard_count",
        "statistics",
        "shards",
        "manifest_sha256",
    }
    if set(value) != expected:
        raise RoutedInputSchemaError(
            "routed-input manifest fields differ from the v1 contract"
        )
    unsigned = {name: value[name] for name in expected if name != "manifest_sha256"}
    claimed = _require_sha256("manifest_sha256", value["manifest_sha256"])
    if canonical_sha256(unsigned) != claimed:
        raise RoutedInputIntegrityError(
            "routed-input manifest canonical SHA-256 mismatch"
        )
    if value["protocol_id"] != PROTOCOL_ID or value["sealed"] is not True:
        raise RoutedInputIntegrityError(
            "only the frozen protocol's sealed overlay is consumable"
        )
    metadata = validate_routed_input_metadata(value["metadata"])
    metadata_sha = _require_sha256("metadata_sha256", value["metadata_sha256"])
    if canonical_sha256(metadata) != metadata_sha:
        raise RoutedInputIntegrityError("overlay metadata SHA-256 mismatch")
    target_identity = _validate_target_identity(value["target_manifest"])
    if metadata["target_manifest_sha256"] != target_identity["manifest_sha256"]:
        raise RoutedInputIntegrityError(
            "metadata and manifest bind different contribution targets"
        )
    if metadata["split"] != target_identity["split"]:
        raise RoutedInputIntegrityError("overlay and target splits differ")
    total_tokens = _require_positive_int("total_tokens", value["total_tokens"])
    shard_count = _require_positive_int("shard_count", value["shard_count"])
    raw_shards = value["shards"]
    if not isinstance(raw_shards, list) or len(raw_shards) != shard_count:
        raise RoutedInputIntegrityError(
            "manifest shard_count differs from its shard inventory"
        )
    shards = [
        _validate_shard_entry(
            shard, metadata=metadata, metadata_sha256=metadata_sha
        )
        for shard in raw_shards
    ]
    if shards != sorted(
        shards, key=lambda item: (item["range_start"], item["range_id"])
    ):
        raise RoutedInputRangeError(
            "overlay shards are not in deterministic target-range order"
        )
    expected_start = 0
    seen: set[str] = set()
    for shard in shards:
        if shard["target_range_id"] in seen:
            raise RoutedInputRangeError("overlay repeats a target range")
        seen.add(shard["target_range_id"])
        if shard["range_start"] != expected_start:
            raise RoutedInputRangeError("overlay ranges contain a gap or overlap")
        expected_start = shard["range_end"]
    if expected_start != total_tokens:
        raise RoutedInputRangeError("overlay ranges do not cover total_tokens")
    if (
        total_tokens != target_identity["total_tokens"]
        or shard_count != target_identity["shard_count"]
    ):
        raise RoutedInputRangeError(
            "overlay token/shard inventory differs from its target manifest"
        )
    merged = merge_routed_input_statistics(
        [shard["statistics"] for shard in shards]
    )
    recorded = validate_routed_input_statistics(
        value["statistics"], expected_tokens=total_tokens
    )
    if canonical_json_bytes(merged) != canonical_json_bytes(recorded):
        raise RoutedInputIntegrityError(
            "manifest statistics differ from the deterministic shard merge"
        )

    if target_manifest is not None:
        target = validate_contribution_manifest(target_manifest)
        if target["manifest_sha256"] != target_identity["manifest_sha256"]:
            raise RoutedInputIntegrityError(
                "provided contribution target differs from the overlay binding"
            )
        if target_manifest_identity(target) != target_identity:
            raise RoutedInputIntegrityError(
                "provided contribution target identity fields changed"
            )
        for overlay, target_shard in zip(
            shards, target["shards"], strict=True
        ):
            exact = {
                "target_range_id": target_shard["range_id"],
                "range_start": target_shard["range_start"],
                "range_end": target_shard["range_end"],
                "tokens": target_shard["tokens"],
                "token_identity_sha256": target_shard[
                    "token_identity_sha256"
                ],
            }
            for name, expected_value in exact.items():
                if overlay[name] != expected_value:
                    raise RoutedInputRangeError(
                        f"overlay {name} differs from its sealed target range"
                    )
        target_metadata = target["metadata"]
        required = {
            "checkpoint_sha256": target_metadata["checkpoint_sha256"],
            "dataset_fingerprint": target_metadata["dataset_fingerprint"],
            "split": target_metadata["split"],
            "model_id": target_metadata["model_id"],
            "target_layer_one_indexed": target_metadata[
                "target_layer_one_indexed"
            ],
            "maxtext_fork_commit": target_metadata["maxtext_fork_commit"],
        }
        for name, expected_value in required.items():
            if metadata[name] != expected_value:
                raise RoutedInputIntegrityError(
                    f"overlay metadata {name} differs from its contribution target"
                )
    return {
        **dict(value),
        "metadata": metadata,
        "target_manifest": target_identity,
        "statistics": recorded,
        "shards": shards,
    }


def validate_routed_input_shard_payload(
    value: RoutedInputBatch | Mapping[str, Any],
    entry: Mapping[str, Any],
) -> RoutedInputBatch:
    """Prove one payload has the exact ordered keys of its target range."""

    batch = validate_routed_input_batch(value)
    if not isinstance(entry, Mapping) or entry.get("tokens") != batch.tokens:
        raise RoutedInputIntegrityError(
            "routed-input payload count differs from its shard entry"
        )
    expected = _require_sha256(
        "shard.token_identity_sha256", entry.get("token_identity_sha256")
    )
    observed = token_identity_sha256(batch.sequence_id, batch.position)
    if observed != expected:
        raise RoutedInputIntegrityError(
            "routed-input token-key sequence differs from its target range"
        )
    return batch
