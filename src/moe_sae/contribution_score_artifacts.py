"""Durable, content-addressed sequence-score evidence for Experiment 2.

The routed evaluator emits process-local rows.  Those rows are scientific
evidence, not transient logging: paired sequence-cluster intervals cannot be
reconstructed from pooled SSE alone.  This module seals each process/update
chunk as an immutable safetensors envelope, then seals one condition manifest
that binds the complete row inventory to the run, checkpoint, captures,
sidecar, and exact evaluation-input receipts.

Serialization functions return bytes and content-addressed references.  They
perform no filesystem or object-store writes; the existing data plane may
retain the returned bytes under the reference digest.  Report construction
must resolve these retained objects and recompute its sufficient scores.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from safetensors.numpy import load as load_safetensors
from safetensors.numpy import save as save_safetensors

from moe_sae.contribution_artifacts import (
    SEQUENCE_LENGTH,
    canonical_json_bytes,
    canonical_sha256,
)
from moe_sae.contribution_input import (
    CONTRIBUTION_INPUT_SCHEMA_VERSION,
    LEGACY_CONTRIBUTION_INPUT_SCHEMA_VERSION,
    ContributionInputReceipt,
)
from moe_sae.contribution_sae import CONTRIBUTION_CONDITIONS

SCORE_ROW_SCHEMA_VERSION = "gemma4-contribution-sequence-score-row-v1"
SCORE_MANIFEST_SCHEMA_VERSION = "gemma4-contribution-sequence-score-manifest-v1"
SCORE_ROW_MEDIA_TYPE = "application/x.moe-sae-score-row+safetensors"
SCORE_MANIFEST_MEDIA_TYPE = "application/x.moe-sae-score-manifest+json"

_ENVELOPE_MAGIC = b"MOE-SAE-SCORE-ROW-V1\0"
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POSITION_BITMAP_WORDS = SEQUENCE_LENGTH // 64
_ROW_TENSORS = {
    "sequence_id",
    "position_bitmap",
    "token_count",
    "component_sse",
    "component_sst",
    "routed_sse",
    "routed_sst",
}


class ContributionScoreArtifactError(ValueError):
    """A retained score object is missing, corrupt, or identity-incompatible."""


@dataclass(frozen=True, slots=True)
class ScoreArtifactRef:
    """Content address and exact byte length of one retained object."""

    sha256: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        _sha256("artifact sha256", self.sha256)
        _positive_int("artifact size_bytes", self.size_bytes)
        if self.media_type not in {
            SCORE_ROW_MEDIA_TYPE,
            SCORE_MANIFEST_MEDIA_TYPE,
        }:
            raise ContributionScoreArtifactError(
                "score artifact media_type is incompatible"
            )

    @classmethod
    def from_value(
        cls,
        value: ScoreArtifactRef | Mapping[str, Any],
        *,
        expected_media_type: str | None = None,
    ) -> ScoreArtifactRef:
        if isinstance(value, cls):
            selected = value
        else:
            raw = _exact_mapping(
                "score artifact reference",
                value,
                {"sha256", "size_bytes", "media_type"},
            )
            selected = cls(**dict(raw))
        if (
            expected_media_type is not None
            and selected.media_type != expected_media_type
        ):
            raise ContributionScoreArtifactError(
                "score artifact reference has the wrong media type"
            )
        return selected

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ScoreArtifactResolver(Protocol):
    """Read one retained object by its exact content-addressed reference."""

    def __call__(self, reference: ScoreArtifactRef) -> bytes: ...


@dataclass(frozen=True, slots=True)
class ResolvedScoreRows:
    """Validated canonical token rows from one process/update shard."""

    reference: ScoreArtifactRef
    condition: str
    seed: int
    process_index: int
    update_index: int
    evaluation_capture_manifest_sha256: str
    checkpoint_sha256: str
    recomposition_sidecar_sha256: str
    centering_mean_sha256: str
    evaluation_input_receipt_sha256: str
    sequence_id: np.ndarray
    position_bitmap: np.ndarray
    token_count: np.ndarray
    component_sse: np.ndarray
    component_sst: np.ndarray
    routed_sse: np.ndarray
    routed_sst: np.ndarray

    @property
    def tokens(self) -> int:
        return int(self.token_count.sum(dtype=np.int64))

    @property
    def token_identity_sha256(self) -> str:
        return _bitmap_token_identity_sha256(
            self.sequence_id,
            self.position_bitmap,
            self.token_count,
        )


@dataclass(frozen=True, slots=True)
class ResolvedScoreManifest:
    """Complete condition/seed score inventory resolved from retained rows."""

    reference: ScoreArtifactRef
    phase: str
    protocol_sha256: str
    condition: str
    seed: int
    randomization_key: int
    training_capture_manifest_sha256: str
    evaluation_capture_manifest_sha256: str
    recomposition_sidecar_sha256: str
    run_manifest_sha256: str
    config_sha256: str
    checkpoint_sha256: str
    layout_sha256: str
    centering_mean_sha256: str
    training_input_receipt_sha256s: tuple[str, ...]
    evaluation_input_receipts: tuple[ContributionInputReceipt, ...]
    row_references: tuple[ScoreArtifactRef, ...]
    sequence_id: np.ndarray
    position_bitmap: np.ndarray
    token_count: np.ndarray
    component_sse: np.ndarray
    component_sst: np.ndarray
    routed_sse: np.ndarray
    routed_sst: np.ndarray

    @property
    def tokens(self) -> int:
        return int(self.token_count.sum(dtype=np.int64))

    @property
    def token_identity_sha256(self) -> str:
        return _bitmap_token_identity_sha256(
            self.sequence_id,
            self.position_bitmap,
            self.token_count,
        )


def _exact_mapping(
    name: str,
    value: Any,
    fields: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContributionScoreArtifactError(f"{name} must be a mapping")
    if set(value) != fields:
        missing = sorted(fields - set(value))
        extra = sorted(set(value) - fields)
        raise ContributionScoreArtifactError(
            f"{name} fields are not exact; missing={missing}, extra={extra}"
        )
    return value


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContributionScoreArtifactError(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContributionScoreArtifactError(f"{name} must be a nonnegative integer")
    return int(value)


def _positive_int(name: str, value: Any) -> int:
    selected = _nonnegative_int(name, value)
    if selected == 0:
        raise ContributionScoreArtifactError(f"{name} must be a positive integer")
    return selected


def _finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContributionScoreArtifactError(f"{name} must be numeric")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0:
        raise ContributionScoreArtifactError(f"{name} must be finite and nonnegative")
    return selected


def _immutable(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    selected = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    selected.setflags(write=False)
    return selected


def _bitmap_counts(position_bitmap: np.ndarray) -> np.ndarray:
    unpacked = np.unpackbits(
        np.ascontiguousarray(position_bitmap).view(np.uint8),
        axis=1,
    )
    return np.sum(unpacked, axis=1, dtype=np.uint16)


def _bitmap_token_identity_sha256(
    sequence_id: np.ndarray,
    position_bitmap: np.ndarray,
    token_count: np.ndarray,
) -> str:
    """Hash the exact canonical token-key expansion without retaining it."""

    total = int(token_count.sum(dtype=np.int64))
    digest = hashlib.sha256()
    digest.update(b"gemma4-scientific-token-identity-v1\0")
    digest.update(total.to_bytes(8, "little", signed=False))
    record_dtype = np.dtype([("sequence_id", "<i8"), ("position", "<u2")])
    for sequence, bitmap, count in zip(
        sequence_id,
        position_bitmap,
        token_count,
        strict=True,
    ):
        positions = np.flatnonzero(
            np.unpackbits(
                np.ascontiguousarray(bitmap).view(np.uint8),
                bitorder="little",
            )
        ).astype(np.uint16)
        if positions.size != int(count):
            raise ContributionScoreArtifactError(
                "position bitmap population differs from token_count"
            )
        records = np.empty(positions.size, dtype=record_dtype)
        records["sequence_id"] = int(sequence)
        records["position"] = positions
        digest.update(records.tobytes(order="C"))
    return digest.hexdigest()


def _token_rows_to_aggregates(
    sequence_id: np.ndarray,
    position: np.ndarray,
    metrics: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    order = np.lexsort((position.astype(np.int64), sequence_id))
    sequence = np.ascontiguousarray(sequence_id[order], dtype="<i8")
    positions = np.ascontiguousarray(position[order], dtype="<u2")
    records = np.rec.fromarrays(
        (sequence, positions),
        names=("sequence_id", "position"),
    )
    if np.unique(records).size != records.size:
        raise ContributionScoreArtifactError(
            "score row shard contains duplicate scientific token keys"
        )
    unique, inverse = np.unique(sequence, return_inverse=True)
    bitmap = np.zeros(
        (unique.size, _POSITION_BITMAP_WORDS),
        dtype="<u8",
    )
    words = positions.astype(np.int64) // 64
    bits = np.left_shift(
        np.uint64(1),
        (positions.astype(np.int64) % 64).astype(np.uint64),
    )
    np.bitwise_or.at(bitmap, (inverse, words), bits)
    result: dict[str, np.ndarray] = {
        "sequence_id": unique.astype("<i8", copy=False),
        "position_bitmap": bitmap,
        "token_count": np.bincount(
            inverse,
            minlength=unique.size,
        ).astype("<u2"),
    }
    for name, value in metrics.items():
        selected = np.asarray(value, dtype=np.float64)[order]
        result[name] = np.bincount(
            inverse,
            weights=selected,
            minlength=unique.size,
        ).astype("<f8")
    return result


def _validate_rows(
    arrays: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if set(arrays) != _ROW_TENSORS:
        raise ContributionScoreArtifactError(
            "score row tensors differ from the frozen schema"
        )
    sequence = np.asarray(arrays["sequence_id"])
    if sequence.dtype != np.int64 or sequence.ndim != 1 or not sequence.size:
        raise ContributionScoreArtifactError(
            "score sequence_id must be nonempty int64 [sequences]"
        )
    if np.any(sequence < 0) or not np.all(sequence[1:] > sequence[:-1]):
        raise ContributionScoreArtifactError(
            "score sequence IDs must be nonnegative, unique, and increasing"
        )
    bitmap = np.asarray(arrays["position_bitmap"])
    if bitmap.dtype != np.uint64 or bitmap.shape != (
        sequence.size,
        _POSITION_BITMAP_WORDS,
    ):
        raise ContributionScoreArtifactError(
            "position_bitmap must be uint64 [sequences,16]"
        )
    count = np.asarray(arrays["token_count"])
    if count.dtype != np.uint16 or count.shape != sequence.shape:
        raise ContributionScoreArtifactError("token_count must be uint16 [sequences]")
    observed_count = _bitmap_counts(bitmap)
    if (
        np.any(count <= 0)
        or np.any(count > SEQUENCE_LENGTH)
        or not np.array_equal(count, observed_count)
    ):
        raise ContributionScoreArtifactError("position bitmap and token_count disagree")

    result = {
        "sequence_id": _immutable(sequence, np.dtype("<i8")),
        "position_bitmap": _immutable(bitmap, np.dtype("<u8")),
        "token_count": _immutable(count, np.dtype("<u2")),
    }
    for name in _ROW_TENSORS - {
        "sequence_id",
        "position_bitmap",
        "token_count",
    }:
        raw = np.asarray(arrays[name])
        if raw.dtype != np.float64 or raw.shape != sequence.shape:
            raise ContributionScoreArtifactError(
                f"{name} must be float64 [N] aligned with token keys"
            )
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise ContributionScoreArtifactError(
                f"{name} must be finite and nonnegative"
            )
        result[name] = _immutable(raw, np.dtype("<f8"))
    if (
        float(result["component_sst"].sum(dtype=np.float64)) <= 0
        or float(result["routed_sst"].sum(dtype=np.float64)) <= 0
    ):
        raise ContributionScoreArtifactError(
            "score row shard requires positive component and routed SST"
        )
    return result


def _encode_row_envelope(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> bytes:
    metadata_bytes = canonical_json_bytes(dict(metadata))
    if len(metadata_bytes) > _MAX_METADATA_BYTES:
        raise ContributionScoreArtifactError("score row metadata is too large")
    tensor_bytes = save_safetensors(dict(arrays))
    return b"".join(
        (
            _ENVELOPE_MAGIC,
            struct.pack("<Q", len(metadata_bytes)),
            metadata_bytes,
            tensor_bytes,
        )
    )


def _decode_row_envelope(blob: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    prefix = len(_ENVELOPE_MAGIC)
    if not isinstance(blob, bytes) or not blob.startswith(_ENVELOPE_MAGIC):
        raise ContributionScoreArtifactError(
            "score row object has an incompatible envelope"
        )
    if len(blob) < prefix + 8:
        raise ContributionScoreArtifactError("score row object is truncated")
    metadata_size = struct.unpack("<Q", blob[prefix : prefix + 8])[0]
    if metadata_size <= 0 or metadata_size > _MAX_METADATA_BYTES:
        raise ContributionScoreArtifactError("score row metadata length is invalid")
    metadata_end = prefix + 8 + metadata_size
    if metadata_end >= len(blob):
        raise ContributionScoreArtifactError("score row payload is truncated")
    try:
        metadata = json.loads(blob[prefix + 8 : metadata_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContributionScoreArtifactError(
            "score row metadata is not canonical JSON"
        ) from error
    if canonical_json_bytes(metadata) != blob[prefix + 8 : metadata_end]:
        raise ContributionScoreArtifactError(
            "score row metadata is not canonically encoded"
        )
    try:
        arrays = load_safetensors(blob[metadata_end:])
    except Exception as error:
        raise ContributionScoreArtifactError(
            "score row safetensors payload cannot be decoded"
        ) from error
    return metadata, arrays


def _reference(
    blob: bytes,
    *,
    media_type: str,
) -> ScoreArtifactRef:
    return ScoreArtifactRef(
        sha256=hashlib.sha256(blob).hexdigest(),
        size_bytes=len(blob),
        media_type=media_type,
    )


def _resolve_bytes(
    reference: ScoreArtifactRef,
    resolver: ScoreArtifactResolver,
) -> bytes:
    try:
        selected = resolver(reference)
    except Exception as error:
        raise ContributionScoreArtifactError(
            f"cannot resolve retained score artifact {reference.sha256}"
        ) from error
    if not isinstance(selected, bytes):
        raise ContributionScoreArtifactError(
            "score artifact resolver must return bytes"
        )
    if len(selected) != reference.size_bytes:
        raise ContributionScoreArtifactError(
            "resolved score artifact size differs from its reference"
        )
    if hashlib.sha256(selected).hexdigest() != reference.sha256:
        raise ContributionScoreArtifactError(
            "resolved score artifact hash differs from its reference"
        )
    return selected


def _validate_input_receipt(
    value: ContributionInputReceipt | Mapping[str, Any],
) -> ContributionInputReceipt:
    fields = {field.name for field in dataclasses.fields(ContributionInputReceipt)}
    if isinstance(value, ContributionInputReceipt):
        selected = dataclasses.asdict(value)
    else:
        selected = dict(_exact_mapping("evaluation input receipt", value, fields))
    claimed = _sha256(
        "evaluation input receipt_sha256",
        selected.pop("receipt_sha256"),
    )
    if selected.get("schema_version") not in {
        LEGACY_CONTRIBUTION_INPUT_SCHEMA_VERSION,
        CONTRIBUTION_INPUT_SCHEMA_VERSION,
    }:
        raise ContributionScoreArtifactError(
            "evaluation input receipt schema is incompatible"
        )
    if canonical_sha256(selected) != claimed:
        raise ContributionScoreArtifactError(
            "evaluation input receipt self-hash is invalid"
        )
    _sha256(
        "evaluation input capture_manifest_sha256",
        selected["capture_manifest_sha256"],
    )
    _sha256("evaluation input layout_sha256", selected["layout_sha256"])
    if selected["condition"] not in CONTRIBUTION_CONDITIONS:
        raise ContributionScoreArtifactError(
            "evaluation input receipt condition is invalid"
        )
    for name in (
        "randomization_key",
        "process_index",
    ):
        _nonnegative_int(f"evaluation input {name}", selected[name])
    for name in (
        "total_tokens",
        "global_microbatch_tokens",
        "microbatches_per_update",
        "planned_updates",
        "process_count",
        "local_rows_per_microbatch",
    ):
        _positive_int(f"evaluation input {name}", selected[name])
    if selected["process_index"] >= selected["process_count"]:
        raise ContributionScoreArtifactError(
            "evaluation input process_index is outside process_count"
        )
    return ContributionInputReceipt(
        **selected,
        receipt_sha256=claimed,
    )


def seal_aggregated_score_row_shard(
    *,
    condition: str,
    seed: int,
    process_index: int,
    update_index: int,
    evaluation_capture_manifest_sha256: str,
    checkpoint_sha256: str,
    recomposition_sidecar_sha256: str,
    centering_mean_sha256: str,
    evaluation_input_receipt_sha256: str,
    sequence_id: Any,
    position_bitmap: Any,
    token_count: Any,
    component_sse: Any,
    component_sst: Any,
    routed_sse: Any,
    routed_sst: Any,
) -> tuple[ScoreArtifactRef, bytes]:
    """Seal exact packed-sequence aggregates and their token-position bitmaps."""

    if condition not in CONTRIBUTION_CONDITIONS:
        raise ContributionScoreArtifactError("score row condition is invalid")
    selected_seed = _nonnegative_int("score row seed", seed)
    selected_process = _nonnegative_int("score row process_index", process_index)
    selected_update = _nonnegative_int("score row update_index", update_index)
    for name, digest in (
        (
            "evaluation_capture_manifest_sha256",
            evaluation_capture_manifest_sha256,
        ),
        ("checkpoint_sha256", checkpoint_sha256),
        ("recomposition_sidecar_sha256", recomposition_sidecar_sha256),
        ("centering_mean_sha256", centering_mean_sha256),
        (
            "evaluation_input_receipt_sha256",
            evaluation_input_receipt_sha256,
        ),
    ):
        _sha256(name, digest)
    checked = _validate_rows(
        {
            "sequence_id": sequence_id,
            "position_bitmap": position_bitmap,
            "token_count": token_count,
            "component_sse": component_sse,
            "component_sst": component_sst,
            "routed_sse": routed_sse,
            "routed_sst": routed_sst,
        }
    )
    metadata = {
        "schema_version": SCORE_ROW_SCHEMA_VERSION,
        "condition": condition,
        "seed": selected_seed,
        "process_index": selected_process,
        "update_index": selected_update,
        "evaluation_capture_manifest_sha256": evaluation_capture_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "recomposition_sidecar_sha256": recomposition_sidecar_sha256,
        "centering_mean_sha256": centering_mean_sha256,
        "evaluation_input_receipt_sha256": (evaluation_input_receipt_sha256),
        "tokens": int(checked["token_count"].sum(dtype=np.int64)),
        "token_identity_sha256": _bitmap_token_identity_sha256(
            checked["sequence_id"],
            checked["position_bitmap"],
            checked["token_count"],
        ),
    }
    blob = _encode_row_envelope(metadata, checked)
    return _reference(blob, media_type=SCORE_ROW_MEDIA_TYPE), blob


def seal_score_row_shard(
    evidence: Any,
    routed_scores: Any,
    *,
    condition: str,
    seed: int,
    process_index: int,
    evaluation_capture_manifest_sha256: str,
    checkpoint_sha256: str,
    recomposition_sidecar_sha256: str,
    centering_mean_sha256: str,
    evaluation_input_receipt_sha256: str,
) -> tuple[ScoreArtifactRef, bytes]:
    """Seal one evaluator process/update chunk without writing it anywhere."""

    if condition not in CONTRIBUTION_CONDITIONS:
        raise ContributionScoreArtifactError("score row condition is invalid")
    selected_seed = _nonnegative_int("score row seed", seed)
    selected_process = _nonnegative_int("score row process_index", process_index)
    for name, digest in (
        (
            "evaluation_capture_manifest_sha256",
            evaluation_capture_manifest_sha256,
        ),
        ("checkpoint_sha256", checkpoint_sha256),
        ("recomposition_sidecar_sha256", recomposition_sidecar_sha256),
        ("centering_mean_sha256", centering_mean_sha256),
        (
            "evaluation_input_receipt_sha256",
            evaluation_input_receipt_sha256,
        ),
    ):
        _sha256(name, digest)
    try:
        update_index = _nonnegative_int(
            "score row update_index",
            evidence.update_index,
        )
        scores = evidence.scores
        valid = np.asarray(scores.valid_mask)
        routed_valid = np.asarray(routed_scores.valid_mask)
    except AttributeError as error:
        raise ContributionScoreArtifactError(
            "score row input is not evaluation evidence"
        ) from error
    if (
        valid.dtype != np.bool_
        or valid.ndim != 2
        or routed_valid.dtype != np.bool_
        or not np.array_equal(valid, routed_valid)
    ):
        raise ContributionScoreArtifactError(
            "component and routed score masks are not identical"
        )
    if routed_scores.recomposition_sidecar_sha256 != recomposition_sidecar_sha256:
        raise ContributionScoreArtifactError(
            "routed score sidecar differs from the retained row identity"
        )
    sequence = np.asarray(evidence.sequence_id)
    position = np.asarray(evidence.position)
    if sequence.dtype != np.int64 or position.dtype != np.uint16:
        raise ContributionScoreArtifactError(
            "evaluation evidence token-key dtypes are incompatible"
        )
    if sequence.shape != valid.shape or position.shape != valid.shape:
        raise ContributionScoreArtifactError(
            "evaluation evidence token keys do not align with score rows"
        )
    if not np.count_nonzero(valid):
        raise ContributionScoreArtifactError(
            "score row shard cannot contain zero scientific tokens"
        )
    metrics = {
        "component_sse": np.asarray(
            scores.component_token_sse[valid],
            dtype="<f8",
        ),
        "component_sst": np.asarray(
            scores.component_token_sst[valid],
            dtype="<f8",
        ),
        "routed_sse": np.asarray(
            routed_scores.token_sse[valid],
            dtype="<f8",
        ),
        "routed_sst": np.asarray(
            routed_scores.token_sst[valid],
            dtype="<f8",
        ),
    }
    arrays = _token_rows_to_aggregates(
        np.asarray(sequence[valid], dtype="<i8"),
        np.asarray(position[valid], dtype="<u2"),
        metrics,
    )
    return seal_aggregated_score_row_shard(
        condition=condition,
        seed=selected_seed,
        process_index=selected_process,
        update_index=update_index,
        evaluation_capture_manifest_sha256=(evaluation_capture_manifest_sha256),
        checkpoint_sha256=checkpoint_sha256,
        recomposition_sidecar_sha256=recomposition_sidecar_sha256,
        centering_mean_sha256=centering_mean_sha256,
        evaluation_input_receipt_sha256=(evaluation_input_receipt_sha256),
        **arrays,
    )


def resolve_score_row_shard(
    reference: ScoreArtifactRef | Mapping[str, Any],
    resolver: ScoreArtifactResolver,
) -> ResolvedScoreRows:
    """Resolve and validate one content-addressed score row shard."""

    selected_ref = ScoreArtifactRef.from_value(
        reference,
        expected_media_type=SCORE_ROW_MEDIA_TYPE,
    )
    metadata, raw_arrays = _decode_row_envelope(_resolve_bytes(selected_ref, resolver))
    selected = _exact_mapping(
        "score row metadata",
        metadata,
        {
            "schema_version",
            "condition",
            "seed",
            "process_index",
            "update_index",
            "evaluation_capture_manifest_sha256",
            "checkpoint_sha256",
            "recomposition_sidecar_sha256",
            "centering_mean_sha256",
            "evaluation_input_receipt_sha256",
            "tokens",
            "token_identity_sha256",
        },
    )
    if selected["schema_version"] != SCORE_ROW_SCHEMA_VERSION:
        raise ContributionScoreArtifactError("score row schema is incompatible")
    if selected["condition"] not in CONTRIBUTION_CONDITIONS:
        raise ContributionScoreArtifactError("score row condition is invalid")
    seed = _nonnegative_int("score row seed", selected["seed"])
    process_index = _nonnegative_int(
        "score row process_index",
        selected["process_index"],
    )
    update_index = _nonnegative_int(
        "score row update_index",
        selected["update_index"],
    )
    for name in (
        "evaluation_capture_manifest_sha256",
        "checkpoint_sha256",
        "recomposition_sidecar_sha256",
        "centering_mean_sha256",
        "evaluation_input_receipt_sha256",
        "token_identity_sha256",
    ):
        _sha256(f"score row {name}", selected[name])
    arrays = _validate_rows(raw_arrays)
    if selected["tokens"] != arrays["token_count"].sum(dtype=np.int64):
        raise ContributionScoreArtifactError(
            "score row metadata token count differs from its payload"
        )
    identity = _bitmap_token_identity_sha256(
        arrays["sequence_id"],
        arrays["position_bitmap"],
        arrays["token_count"],
    )
    if selected["token_identity_sha256"] != identity:
        raise ContributionScoreArtifactError(
            "score row token identity differs from its payload"
        )
    return ResolvedScoreRows(
        reference=selected_ref,
        condition=selected["condition"],
        seed=seed,
        process_index=process_index,
        update_index=update_index,
        evaluation_capture_manifest_sha256=(
            selected["evaluation_capture_manifest_sha256"]
        ),
        checkpoint_sha256=selected["checkpoint_sha256"],
        recomposition_sidecar_sha256=(selected["recomposition_sidecar_sha256"]),
        centering_mean_sha256=selected["centering_mean_sha256"],
        evaluation_input_receipt_sha256=(selected["evaluation_input_receipt_sha256"]),
        sequence_id=arrays["sequence_id"],
        position_bitmap=arrays["position_bitmap"],
        token_count=arrays["token_count"],
        component_sse=arrays["component_sse"],
        component_sst=arrays["component_sst"],
        routed_sse=arrays["routed_sse"],
        routed_sst=arrays["routed_sst"],
    )


def _normalize_receipt_set(
    values: Sequence[ContributionInputReceipt | Mapping[str, Any]],
    *,
    condition: str,
    randomization_key: int,
    evaluation_capture_manifest_sha256: str,
    layout_sha256: str,
    expected_tokens: int,
) -> tuple[ContributionInputReceipt, ...]:
    receipts = tuple(_validate_input_receipt(value) for value in values)
    if not receipts:
        raise ContributionScoreArtifactError(
            "score manifest requires evaluation input receipts"
        )
    process_counts = {item.process_count for item in receipts}
    if len(process_counts) != 1:
        raise ContributionScoreArtifactError(
            "evaluation input receipts disagree on process_count"
        )
    process_count = next(iter(process_counts))
    if len(receipts) != process_count or {
        item.process_index for item in receipts
    } != set(range(process_count)):
        raise ContributionScoreArtifactError(
            "evaluation input receipt process set is incomplete"
        )
    common_schedule = {
        (
            item.total_tokens,
            item.global_microbatch_tokens,
            item.microbatches_per_update,
            item.planned_updates,
            item.local_rows_per_microbatch,
        )
        for item in receipts
    }
    if len(common_schedule) != 1:
        raise ContributionScoreArtifactError(
            "evaluation input receipts disagree on their schedule"
        )
    for item in receipts:
        if (
            item.condition != condition
            or item.randomization_key != randomization_key
            or item.capture_manifest_sha256 != evaluation_capture_manifest_sha256
            or item.layout_sha256 != layout_sha256
            or item.total_tokens != expected_tokens
        ):
            raise ContributionScoreArtifactError(
                "evaluation input receipt differs from the score manifest identity"
            )
    return tuple(sorted(receipts, key=lambda item: item.process_index))


def _manifest_unsigned(
    *,
    phase: str,
    protocol_sha256: str,
    condition: str,
    seed: int,
    randomization_key: int,
    training_capture_manifest_sha256: str,
    evaluation_capture_manifest_sha256: str,
    recomposition_sidecar_sha256: str,
    run_manifest_sha256: str,
    config_sha256: str,
    checkpoint_sha256: str,
    layout_sha256: str,
    centering_mean_sha256: str,
    training_input_receipt_sha256s: Sequence[str],
    evaluation_input_receipts: Sequence[ContributionInputReceipt],
    row_references: Sequence[ScoreArtifactRef],
    sequence_id: np.ndarray,
    position_bitmap: np.ndarray,
    token_count: np.ndarray,
    component_sse: np.ndarray,
    component_sst: np.ndarray,
    routed_sse: np.ndarray,
    routed_sst: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": SCORE_MANIFEST_SCHEMA_VERSION,
        "phase": phase,
        "protocol_sha256": protocol_sha256,
        "condition": condition,
        "seed": seed,
        "randomization_key": randomization_key,
        "training_capture_manifest_sha256": (training_capture_manifest_sha256),
        "evaluation_capture_manifest_sha256": (evaluation_capture_manifest_sha256),
        "recomposition_sidecar_sha256": recomposition_sidecar_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "layout_sha256": layout_sha256,
        "centering_mean_sha256": centering_mean_sha256,
        "training_input_receipt_sha256s": list(training_input_receipt_sha256s),
        "evaluation_input_receipts": [
            dataclasses.asdict(item) for item in evaluation_input_receipts
        ],
        "row_artifacts": [item.to_dict() for item in row_references],
        "tokens": int(token_count.sum(dtype=np.int64)),
        "token_identity_sha256": _bitmap_token_identity_sha256(
            sequence_id,
            position_bitmap,
            token_count,
        ),
        "component_sse": float(component_sse.sum(dtype=np.float64)),
        "component_sst": float(component_sst.sum(dtype=np.float64)),
        "routed_sse": float(routed_sse.sum(dtype=np.float64)),
        "routed_sst": float(routed_sst.sum(dtype=np.float64)),
    }


def _resolve_and_merge_rows(
    row_references: Sequence[ScoreArtifactRef],
    resolver: ScoreArtifactResolver,
    *,
    condition: str,
    seed: int,
    evaluation_capture_manifest_sha256: str,
    checkpoint_sha256: str,
    recomposition_sidecar_sha256: str,
    centering_mean_sha256: str,
    evaluation_input_receipts: Sequence[ContributionInputReceipt],
) -> tuple[tuple[ResolvedScoreRows, ...], dict[str, np.ndarray]]:
    rows = tuple(
        resolve_score_row_shard(reference, resolver) for reference in row_references
    )
    if not rows:
        raise ContributionScoreArtifactError(
            "score manifest requires at least one row artifact"
        )
    receipt_by_process = {
        item.process_index: item for item in evaluation_input_receipts
    }
    seen_updates: set[tuple[int, int]] = set()
    for row in rows:
        key = (row.process_index, row.update_index)
        if key in seen_updates:
            raise ContributionScoreArtifactError(
                "score row artifacts contain a duplicate process/update"
            )
        seen_updates.add(key)
        receipt = receipt_by_process.get(row.process_index)
        if receipt is None:
            raise ContributionScoreArtifactError(
                "score row artifact names an unknown process"
            )
        if (
            row.condition != condition
            or row.seed != seed
            or row.evaluation_capture_manifest_sha256
            != evaluation_capture_manifest_sha256
            or row.checkpoint_sha256 != checkpoint_sha256
            or row.recomposition_sidecar_sha256 != recomposition_sidecar_sha256
            or row.centering_mean_sha256 != centering_mean_sha256
            or row.evaluation_input_receipt_sha256 != receipt.receipt_sha256
        ):
            raise ContributionScoreArtifactError(
                "score row artifact differs from its condition manifest"
            )
    expected_updates = {
        (receipt.process_index, update)
        for receipt in evaluation_input_receipts
        for update in range(receipt.planned_updates)
    }
    if seen_updates != expected_updates:
        raise ContributionScoreArtifactError(
            "score row process/update coverage is incomplete"
        )

    sequence = np.concatenate([row.sequence_id for row in rows], axis=0)
    order = np.argsort(sequence, kind="stable")
    sequence = sequence[order]
    bitmap = np.concatenate([row.position_bitmap for row in rows], axis=0)[order]
    metric_names = (
        "component_sse",
        "component_sst",
        "routed_sse",
        "routed_sst",
    )
    metrics = {
        name: np.concatenate([getattr(row, name) for row in rows], axis=0)[order]
        for name in metric_names
    }
    unique, inverse = np.unique(sequence, return_inverse=True)
    merged_bitmap = np.zeros(
        (unique.size, _POSITION_BITMAP_WORDS),
        dtype="<u8",
    )
    merged_metrics = {
        name: np.zeros((unique.size,), dtype="<f8") for name in metric_names
    }
    for source_index, destination_index in enumerate(inverse):
        selected_bitmap = bitmap[source_index]
        if np.any(merged_bitmap[destination_index] & selected_bitmap):
            raise ContributionScoreArtifactError(
                "score row artifacts contain duplicate scientific token keys"
            )
        merged_bitmap[destination_index] |= selected_bitmap
        for name in metric_names:
            merged_metrics[name][destination_index] += metrics[name][source_index]
    merged = {
        "sequence_id": unique.astype("<i8", copy=False),
        "position_bitmap": merged_bitmap,
        "token_count": _bitmap_counts(merged_bitmap),
        **merged_metrics,
    }
    checked = _validate_rows(merged)
    return rows, checked


def seal_score_manifest(
    row_references: Sequence[ScoreArtifactRef | Mapping[str, Any]],
    resolver: ScoreArtifactResolver,
    *,
    phase: str,
    protocol_sha256: str,
    condition: str,
    seed: int,
    randomization_key: int,
    training_capture_manifest_sha256: str,
    evaluation_capture_manifest_sha256: str,
    recomposition_sidecar_sha256: str,
    run_manifest_sha256: str,
    config_sha256: str,
    checkpoint_sha256: str,
    layout_sha256: str,
    centering_mean_sha256: str,
    training_input_receipt_sha256s: Sequence[str],
    evaluation_input_receipts: Sequence[ContributionInputReceipt | Mapping[str, Any]],
    expected_tokens: int,
) -> tuple[ScoreArtifactRef, bytes]:
    """Seal one complete condition/seed manifest after resolving every row."""

    if not isinstance(phase, str) or not phase:
        raise ContributionScoreArtifactError("score manifest phase is invalid")
    if condition not in CONTRIBUTION_CONDITIONS:
        raise ContributionScoreArtifactError("score manifest condition is invalid")
    selected_seed = _nonnegative_int("score manifest seed", seed)
    selected_randomization = _nonnegative_int(
        "score manifest randomization_key",
        randomization_key,
    )
    selected_tokens = _positive_int("score manifest expected_tokens", expected_tokens)
    for name, digest in (
        ("protocol_sha256", protocol_sha256),
        (
            "training_capture_manifest_sha256",
            training_capture_manifest_sha256,
        ),
        (
            "evaluation_capture_manifest_sha256",
            evaluation_capture_manifest_sha256,
        ),
        ("recomposition_sidecar_sha256", recomposition_sidecar_sha256),
        ("run_manifest_sha256", run_manifest_sha256),
        ("config_sha256", config_sha256),
        ("checkpoint_sha256", checkpoint_sha256),
        ("layout_sha256", layout_sha256),
        ("centering_mean_sha256", centering_mean_sha256),
    ):
        _sha256(name, digest)
    training_receipts = tuple(
        sorted(
            _sha256("training input receipt SHA-256", item)
            for item in training_input_receipt_sha256s
        )
    )
    if not training_receipts or len(set(training_receipts)) != len(training_receipts):
        raise ContributionScoreArtifactError(
            "training input receipt identity set is empty or duplicated"
        )
    evaluation_receipts = _normalize_receipt_set(
        evaluation_input_receipts,
        condition=condition,
        randomization_key=selected_randomization,
        evaluation_capture_manifest_sha256=(evaluation_capture_manifest_sha256),
        layout_sha256=layout_sha256,
        expected_tokens=selected_tokens,
    )
    references = tuple(
        ScoreArtifactRef.from_value(
            item,
            expected_media_type=SCORE_ROW_MEDIA_TYPE,
        )
        for item in row_references
    )
    if len({item.sha256 for item in references}) != len(references):
        raise ContributionScoreArtifactError(
            "score manifest contains duplicate row artifacts"
        )
    _, merged = _resolve_and_merge_rows(
        references,
        resolver,
        condition=condition,
        seed=selected_seed,
        evaluation_capture_manifest_sha256=(evaluation_capture_manifest_sha256),
        checkpoint_sha256=checkpoint_sha256,
        recomposition_sidecar_sha256=recomposition_sidecar_sha256,
        centering_mean_sha256=centering_mean_sha256,
        evaluation_input_receipts=evaluation_receipts,
    )
    if merged["token_count"].sum(dtype=np.int64) != selected_tokens:
        raise ContributionScoreArtifactError(
            "score row artifacts do not cover the exact evaluation token count"
        )
    unsigned = _manifest_unsigned(
        phase=phase,
        protocol_sha256=protocol_sha256,
        condition=condition,
        seed=selected_seed,
        randomization_key=selected_randomization,
        training_capture_manifest_sha256=(training_capture_manifest_sha256),
        evaluation_capture_manifest_sha256=(evaluation_capture_manifest_sha256),
        recomposition_sidecar_sha256=recomposition_sidecar_sha256,
        run_manifest_sha256=run_manifest_sha256,
        config_sha256=config_sha256,
        checkpoint_sha256=checkpoint_sha256,
        layout_sha256=layout_sha256,
        centering_mean_sha256=centering_mean_sha256,
        training_input_receipt_sha256s=training_receipts,
        evaluation_input_receipts=evaluation_receipts,
        row_references=references,
        sequence_id=merged["sequence_id"],
        position_bitmap=merged["position_bitmap"],
        token_count=merged["token_count"],
        component_sse=merged["component_sse"],
        component_sst=merged["component_sst"],
        routed_sse=merged["routed_sse"],
        routed_sst=merged["routed_sst"],
    )
    blob = canonical_json_bytes(unsigned)
    return _reference(blob, media_type=SCORE_MANIFEST_MEDIA_TYPE), blob


def resolve_score_manifest(
    reference: ScoreArtifactRef | Mapping[str, Any],
    resolver: ScoreArtifactResolver,
) -> ResolvedScoreManifest:
    """Resolve one manifest and every retained row it authorizes."""

    selected_ref = ScoreArtifactRef.from_value(
        reference,
        expected_media_type=SCORE_MANIFEST_MEDIA_TYPE,
    )
    blob = _resolve_bytes(selected_ref, resolver)
    try:
        raw = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContributionScoreArtifactError(
            "score manifest is not canonical JSON"
        ) from error
    if canonical_json_bytes(raw) != blob:
        raise ContributionScoreArtifactError(
            "score manifest is not canonically encoded"
        )
    fields = {
        "schema_version",
        "phase",
        "protocol_sha256",
        "condition",
        "seed",
        "randomization_key",
        "training_capture_manifest_sha256",
        "evaluation_capture_manifest_sha256",
        "recomposition_sidecar_sha256",
        "run_manifest_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "layout_sha256",
        "centering_mean_sha256",
        "training_input_receipt_sha256s",
        "evaluation_input_receipts",
        "row_artifacts",
        "tokens",
        "token_identity_sha256",
        "component_sse",
        "component_sst",
        "routed_sse",
        "routed_sst",
    }
    selected = _exact_mapping("score manifest", raw, fields)
    if selected["schema_version"] != SCORE_MANIFEST_SCHEMA_VERSION:
        raise ContributionScoreArtifactError("score manifest schema is incompatible")
    phase = selected["phase"]
    if not isinstance(phase, str) or not phase:
        raise ContributionScoreArtifactError("score manifest phase is invalid")
    condition = selected["condition"]
    if condition not in CONTRIBUTION_CONDITIONS:
        raise ContributionScoreArtifactError("score manifest condition is invalid")
    seed = _nonnegative_int("score manifest seed", selected["seed"])
    randomization_key = _nonnegative_int(
        "score manifest randomization_key",
        selected["randomization_key"],
    )
    for name in (
        "protocol_sha256",
        "training_capture_manifest_sha256",
        "evaluation_capture_manifest_sha256",
        "recomposition_sidecar_sha256",
        "run_manifest_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "layout_sha256",
        "centering_mean_sha256",
        "token_identity_sha256",
    ):
        _sha256(f"score manifest {name}", selected[name])
    tokens = _positive_int("score manifest tokens", selected["tokens"])
    if not isinstance(selected["training_input_receipt_sha256s"], list):
        raise ContributionScoreArtifactError(
            "training input receipt identities must be a list"
        )
    training_receipts = tuple(
        _sha256("training input receipt SHA-256", item)
        for item in selected["training_input_receipt_sha256s"]
    )
    if (
        not training_receipts
        or tuple(sorted(training_receipts)) != training_receipts
        or len(set(training_receipts)) != len(training_receipts)
    ):
        raise ContributionScoreArtifactError(
            "training input receipt identities are not canonical"
        )
    if not isinstance(selected["evaluation_input_receipts"], list):
        raise ContributionScoreArtifactError("evaluation input receipts must be a list")
    evaluation_receipts = _normalize_receipt_set(
        selected["evaluation_input_receipts"],
        condition=condition,
        randomization_key=randomization_key,
        evaluation_capture_manifest_sha256=(
            selected["evaluation_capture_manifest_sha256"]
        ),
        layout_sha256=selected["layout_sha256"],
        expected_tokens=tokens,
    )
    if not isinstance(selected["row_artifacts"], list):
        raise ContributionScoreArtifactError(
            "score manifest row_artifacts must be a list"
        )
    references = tuple(
        ScoreArtifactRef.from_value(
            item,
            expected_media_type=SCORE_ROW_MEDIA_TYPE,
        )
        for item in selected["row_artifacts"]
    )
    if len({item.sha256 for item in references}) != len(references):
        raise ContributionScoreArtifactError(
            "score manifest contains duplicate row artifacts"
        )
    _, merged = _resolve_and_merge_rows(
        references,
        resolver,
        condition=condition,
        seed=seed,
        evaluation_capture_manifest_sha256=(
            selected["evaluation_capture_manifest_sha256"]
        ),
        checkpoint_sha256=selected["checkpoint_sha256"],
        recomposition_sidecar_sha256=(selected["recomposition_sidecar_sha256"]),
        centering_mean_sha256=selected["centering_mean_sha256"],
        evaluation_input_receipts=evaluation_receipts,
    )
    if merged["token_count"].sum(dtype=np.int64) != tokens:
        raise ContributionScoreArtifactError("score manifest row coverage is not exact")
    if (
        _bitmap_token_identity_sha256(
            merged["sequence_id"],
            merged["position_bitmap"],
            merged["token_count"],
        )
        != selected["token_identity_sha256"]
    ):
        raise ContributionScoreArtifactError(
            "score manifest token identity differs from retained rows"
        )
    for name in ("component_sse", "component_sst", "routed_sse", "routed_sst"):
        claimed = _finite_nonnegative(
            f"score manifest {name}",
            selected[name],
        )
        actual = float(merged[name].sum(dtype=np.float64))
        if not math.isclose(
            claimed,
            actual,
            rel_tol=1e-12,
            abs_tol=1e-10 * max(1.0, abs(actual)),
        ):
            raise ContributionScoreArtifactError(
                f"score manifest {name} differs from retained rows"
            )
    if selected["component_sst"] <= 0 or selected["routed_sst"] <= 0:
        raise ContributionScoreArtifactError(
            "score manifest SST totals must be positive"
        )
    return ResolvedScoreManifest(
        reference=selected_ref,
        phase=phase,
        protocol_sha256=selected["protocol_sha256"],
        condition=condition,
        seed=seed,
        randomization_key=randomization_key,
        training_capture_manifest_sha256=(selected["training_capture_manifest_sha256"]),
        evaluation_capture_manifest_sha256=(
            selected["evaluation_capture_manifest_sha256"]
        ),
        recomposition_sidecar_sha256=(selected["recomposition_sidecar_sha256"]),
        run_manifest_sha256=selected["run_manifest_sha256"],
        config_sha256=selected["config_sha256"],
        checkpoint_sha256=selected["checkpoint_sha256"],
        layout_sha256=selected["layout_sha256"],
        centering_mean_sha256=selected["centering_mean_sha256"],
        training_input_receipt_sha256s=training_receipts,
        evaluation_input_receipts=evaluation_receipts,
        row_references=references,
        sequence_id=merged["sequence_id"],
        position_bitmap=merged["position_bitmap"],
        token_count=merged["token_count"],
        component_sse=merged["component_sse"],
        component_sst=merged["component_sst"],
        routed_sse=merged["routed_sse"],
        routed_sst=merged["routed_sst"],
    )


def resolve_score_manifest_matrix(
    references: Sequence[ScoreArtifactRef | Mapping[str, Any]],
    resolver: ScoreArtifactResolver,
    *,
    expected_seeds: Sequence[int],
) -> dict[tuple[int, str], ResolvedScoreManifest]:
    """Resolve an exact A/R/B/F manifest matrix and verify common held-out rows."""

    resolved = [resolve_score_manifest(item, resolver) for item in references]
    expected = {
        (seed, condition)
        for seed in expected_seeds
        for condition in CONTRIBUTION_CONDITIONS
    }
    result = {(item.seed, item.condition): item for item in resolved}
    if len(result) != len(resolved) or set(result) != expected:
        raise ContributionScoreArtifactError(
            "score manifest matrix is incomplete or contains unexpected cells"
        )
    common_identity: str | None = None
    common_sequence: np.ndarray | None = None
    common_position_bitmap: np.ndarray | None = None
    common_token_count: np.ndarray | None = None
    common_component_sst: np.ndarray | None = None
    common_routed_sst: np.ndarray | None = None
    common_centering: str | None = None
    for key in sorted(result):
        item = result[key]
        if common_identity is None:
            common_identity = item.token_identity_sha256
            common_sequence = item.sequence_id
            common_position_bitmap = item.position_bitmap
            common_token_count = item.token_count
            common_component_sst = item.component_sst
            common_routed_sst = item.routed_sst
            common_centering = item.centering_mean_sha256
            continue
        if (
            item.token_identity_sha256 != common_identity
            or not np.array_equal(item.sequence_id, common_sequence)
            or not np.array_equal(
                item.position_bitmap,
                common_position_bitmap,
            )
            or not np.array_equal(item.token_count, common_token_count)
        ):
            raise ContributionScoreArtifactError(
                "score manifests use different scientific token inventories"
            )
        if not np.array_equal(
            item.component_sst, common_component_sst
        ) or not np.array_equal(item.routed_sst, common_routed_sst):
            raise ContributionScoreArtifactError(
                "score manifests do not share exact component/routed SST rows"
            )
        if item.centering_mean_sha256 != common_centering:
            raise ContributionScoreArtifactError(
                "score manifests use different centering means"
            )
    return result


__all__ = [
    "SCORE_MANIFEST_MEDIA_TYPE",
    "SCORE_MANIFEST_SCHEMA_VERSION",
    "SCORE_ROW_MEDIA_TYPE",
    "SCORE_ROW_SCHEMA_VERSION",
    "ContributionScoreArtifactError",
    "ResolvedScoreManifest",
    "ResolvedScoreRows",
    "ScoreArtifactRef",
    "ScoreArtifactResolver",
    "resolve_score_manifest",
    "resolve_score_manifest_matrix",
    "resolve_score_row_shard",
    "seal_aggregated_score_row_shard",
    "seal_score_manifest",
    "seal_score_row_shard",
]
