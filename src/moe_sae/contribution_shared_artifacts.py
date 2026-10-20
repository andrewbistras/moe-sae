"""Durable shared-Q and full-FFN evidence for Experiment 2.

The numerical evaluators deliberately emit process-local callbacks.  This
module turns those callbacks into immutable, content-addressed evidence:

* shared-Q SSE/SST/L0 rows aggregated by packed sequence;
* oracle-shared and learned-shared full-FFN SSE/SST rows;
* complete row manifests with exact process/update and token-key coverage; and
* a memory-bounded, checkpoint-free training-target pass that seals the one
  common full-FFN mean and SST.

Serialization returns bytes plus a content reference.  ``SharedQScoreRowSink``
and ``FullFFNScoreRowSink`` additionally require caller-supplied durable retain
and resolve functions and perform a read-after-write validation.  This keeps
GCS policy outside the scientific layer without treating an in-memory callback
as evidence.
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
    HIDDEN_SIZE,
    SCHEMA_VERSION,
    SEQUENCE_LENGTH,
    ContributionBatch,
    bf16_bits_to_float32,
    canonical_json_bytes,
    canonical_sha256,
    token_identity_sha256,
    validate_contribution_batch,
    validate_contribution_manifest,
)
from moe_sae.contribution_recomposition import (
    decode_recomposition_sidecar,
    recompose_gemma4,
)
from moe_sae.contribution_score_artifacts import (
    SCORE_MANIFEST_MEDIA_TYPE,
    ScoreArtifactRef,
    resolve_score_manifest,
    resolve_score_row_shard,
)
from moe_sae.contribution_shared import (
    FullFFNRecompositionRows,
    LEGACY_SHARED_Q_INPUT_SCHEMA_VERSION,
    SHARED_Q_INPUT_SCHEMA_VERSION,
    SharedQEvaluationEvidenceChunk,
    SharedQInputReceipt,
    full_ffn_token_scores,
)

FULL_FFN_CENTERING_SCHEMA_VERSION = "gemma4-full-ffn-centering-v1"
SHARED_Q_SCORE_ROW_SCHEMA_VERSION = "gemma4-shared-q-score-row-v1"
SHARED_Q_SCORE_MANIFEST_SCHEMA_VERSION = "gemma4-shared-q-score-manifest-v1"
FULL_FFN_SCORE_ROW_SCHEMA_VERSION = "gemma4-full-ffn-score-row-v1"
FULL_FFN_SCORE_MANIFEST_SCHEMA_VERSION = "gemma4-full-ffn-score-manifest-v1"

FULL_FFN_CENTERING_MEDIA_TYPE = (
    "application/x.moe-sae-full-ffn-centering+safetensors"
)
SHARED_Q_SCORE_ROW_MEDIA_TYPE = "application/x.moe-sae-shared-q-row+safetensors"
SHARED_Q_SCORE_MANIFEST_MEDIA_TYPE = (
    "application/x.moe-sae-shared-q-manifest+json"
)
FULL_FFN_SCORE_ROW_MEDIA_TYPE = "application/x.moe-sae-full-ffn-row+safetensors"
FULL_FFN_SCORE_MANIFEST_MEDIA_TYPE = (
    "application/x.moe-sae-full-ffn-manifest+json"
)

_MEDIA_TYPES = {
    FULL_FFN_CENTERING_MEDIA_TYPE,
    SHARED_Q_SCORE_ROW_MEDIA_TYPE,
    SHARED_Q_SCORE_MANIFEST_MEDIA_TYPE,
    FULL_FFN_SCORE_ROW_MEDIA_TYPE,
    FULL_FFN_SCORE_MANIFEST_MEDIA_TYPE,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POSITION_BITMAP_WORDS = SEQUENCE_LENGTH // 64
_ENVELOPE_MAGIC = b"MOE-SAE-SHARED-EVIDENCE-V1\0"
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_TOKEN_RECORD_DTYPE = np.dtype(
    [("sequence_id", "<i8"), ("position", "<u2")]
)


class SharedEvidenceArtifactError(ValueError):
    """Shared-Q/full-FFN evidence is corrupt or identity-incompatible."""


@dataclass(frozen=True, slots=True)
class SharedEvidenceRef:
    """Content address and exact length of one retained evidence object."""

    sha256: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        _sha256("artifact sha256", self.sha256)
        _positive_int("artifact size_bytes", self.size_bytes)
        if self.media_type not in _MEDIA_TYPES:
            raise SharedEvidenceArtifactError(
                "shared evidence media_type is incompatible"
            )

    @classmethod
    def from_value(
        cls,
        value: SharedEvidenceRef | Mapping[str, Any],
        *,
        expected_media_type: str | None = None,
    ) -> SharedEvidenceRef:
        if isinstance(value, cls):
            selected = value
        else:
            raw = _exact_mapping(
                "shared evidence reference",
                value,
                {"sha256", "size_bytes", "media_type"},
            )
            selected = cls(**dict(raw))
        if (
            expected_media_type is not None
            and selected.media_type != expected_media_type
        ):
            raise SharedEvidenceArtifactError(
                "shared evidence reference has the wrong media type"
            )
        return selected

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class SharedEvidenceResolver(Protocol):
    def __call__(self, reference: SharedEvidenceRef) -> bytes: ...


class SharedEvidenceRetainer(Protocol):
    def __call__(self, reference: SharedEvidenceRef, blob: bytes) -> None: ...


def _exact_mapping(
    name: str,
    value: Any,
    fields: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SharedEvidenceArtifactError(f"{name} must be a mapping")
    if set(value) != fields:
        missing = sorted(fields - set(value))
        extra = sorted(set(value) - fields)
        raise SharedEvidenceArtifactError(
            f"{name} fields are not exact; missing={missing}, extra={extra}"
        )
    return value


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SharedEvidenceArtifactError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SharedEvidenceArtifactError(
            f"{name} must be a nonnegative integer"
        )
    return int(value)


def _positive_int(name: str, value: Any) -> int:
    selected = _nonnegative_int(name, value)
    if selected == 0:
        raise SharedEvidenceArtifactError(f"{name} must be positive")
    return selected


def _finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SharedEvidenceArtifactError(f"{name} must be numeric")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0:
        raise SharedEvidenceArtifactError(
            f"{name} must be finite and nonnegative"
        )
    return selected


def _reference(blob: bytes, *, media_type: str) -> SharedEvidenceRef:
    return SharedEvidenceRef(
        sha256=hashlib.sha256(blob).hexdigest(),
        size_bytes=len(blob),
        media_type=media_type,
    )


def _resolve_bytes(
    reference: SharedEvidenceRef,
    resolver: SharedEvidenceResolver,
) -> bytes:
    try:
        raw = resolver(reference)
    except Exception as error:
        raise SharedEvidenceArtifactError(
            "cannot resolve retained shared evidence"
        ) from error
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise SharedEvidenceArtifactError(
            "shared evidence resolver must return bytes"
        )
    blob = bytes(raw)
    if len(blob) != reference.size_bytes:
        raise SharedEvidenceArtifactError(
            "retained shared evidence size differs from its reference"
        )
    if hashlib.sha256(blob).hexdigest() != reference.sha256:
        raise SharedEvidenceArtifactError(
            "retained shared evidence hash differs from its reference"
        )
    return blob


def _encode_envelope(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray],
) -> bytes:
    metadata_bytes = canonical_json_bytes(dict(metadata))
    if len(metadata_bytes) > _MAX_METADATA_BYTES:
        raise SharedEvidenceArtifactError(
            "shared evidence metadata is unexpectedly large"
        )
    tensor_bytes = save_safetensors(
        {
            name: np.ascontiguousarray(np.asarray(value))
            for name, value in sorted(tensors.items())
        }
    )
    return (
        _ENVELOPE_MAGIC
        + struct.pack("<Q", len(metadata_bytes))
        + metadata_bytes
        + tensor_bytes
    )


def _decode_envelope(blob: bytes) -> tuple[Mapping[str, Any], Mapping[str, np.ndarray]]:
    header = len(_ENVELOPE_MAGIC) + 8
    if len(blob) <= header or not blob.startswith(_ENVELOPE_MAGIC):
        raise SharedEvidenceArtifactError(
            "shared evidence envelope magic is invalid"
        )
    metadata_size = struct.unpack(
        "<Q",
        blob[len(_ENVELOPE_MAGIC) : header],
    )[0]
    if metadata_size <= 0 or metadata_size > _MAX_METADATA_BYTES:
        raise SharedEvidenceArtifactError(
            "shared evidence metadata length is invalid"
        )
    boundary = header + metadata_size
    if boundary >= len(blob):
        raise SharedEvidenceArtifactError(
            "shared evidence envelope is truncated"
        )
    metadata_blob = blob[header:boundary]
    try:
        metadata = json.loads(metadata_blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SharedEvidenceArtifactError(
            "shared evidence metadata is not JSON"
        ) from error
    if canonical_json_bytes(metadata) != metadata_blob:
        raise SharedEvidenceArtifactError(
            "shared evidence metadata is not canonical"
        )
    try:
        tensors = load_safetensors(blob[boundary:])
    except Exception as error:
        raise SharedEvidenceArtifactError(
            "shared evidence safetensors payload is invalid"
        ) from error
    return metadata, tensors


def _immutable(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    selected = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    selected.setflags(write=False)
    return selected


def _bitmap_counts(bitmap: np.ndarray) -> np.ndarray:
    unpacked = np.unpackbits(
        np.ascontiguousarray(bitmap).view(np.uint8),
        axis=1,
        bitorder="little",
    )
    return np.sum(unpacked, axis=1, dtype=np.uint16)


def _bitmap_token_identity_sha256(
    sequence_id: np.ndarray,
    position_bitmap: np.ndarray,
    token_count: np.ndarray,
) -> str:
    total = int(token_count.sum(dtype=np.int64))
    sequence_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
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
            raise SharedEvidenceArtifactError(
                "position bitmap population differs from token_count"
            )
        sequence_parts.append(
            np.full(positions.size, int(sequence), dtype=np.int64)
        )
        position_parts.append(positions)
    if not sequence_parts or sum(part.size for part in sequence_parts) != total:
        raise SharedEvidenceArtifactError("score rows contain no token identity")
    return token_identity_sha256(
        np.concatenate(sequence_parts),
        np.concatenate(position_parts),
    )


def _aggregate_token_metrics(
    sequence_id: Any,
    position: Any,
    metrics: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    sequence = np.asarray(sequence_id)
    positions = np.asarray(position)
    if (
        sequence.dtype != np.int64
        or positions.dtype != np.uint16
        or sequence.ndim != 1
        or positions.shape != sequence.shape
        or not sequence.size
        or np.any(sequence < 0)
        or np.any(positions >= SEQUENCE_LENGTH)
    ):
        raise SharedEvidenceArtifactError(
            "score token keys must be valid aligned int64/uint16 vectors"
        )
    records = np.rec.fromarrays(
        (sequence, positions),
        names=("sequence_id", "position"),
    )
    if np.unique(records).size != records.size:
        raise SharedEvidenceArtifactError(
            "score row contains duplicate scientific token keys"
        )
    order = np.lexsort((positions.astype(np.int64), sequence))
    sequence = np.ascontiguousarray(sequence[order], dtype="<i8")
    positions = np.ascontiguousarray(positions[order], dtype="<u2")
    unique, inverse = np.unique(sequence, return_inverse=True)
    bitmap = np.zeros((unique.size, _POSITION_BITMAP_WORDS), dtype="<u8")
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
        selected = np.asarray(value)
        if selected.shape != order.shape or not np.issubdtype(
            selected.dtype,
            np.number,
        ):
            raise SharedEvidenceArtifactError(
                f"score metric {name} is not aligned numeric data"
            )
        selected64 = selected.astype(np.float64, copy=False)[order]
        if not np.isfinite(selected64).all() or np.any(selected64 < 0):
            raise SharedEvidenceArtifactError(
                f"score metric {name} must be finite and nonnegative"
            )
        result[name] = np.bincount(
            inverse,
            weights=selected64,
            minlength=unique.size,
        ).astype("<f8")
    return result


def _validate_aggregated_rows(
    tensors: Mapping[str, Any],
    metric_names: Sequence[str],
) -> dict[str, np.ndarray]:
    required = {
        "sequence_id",
        "position_bitmap",
        "token_count",
        *metric_names,
    }
    if set(tensors) != required:
        raise SharedEvidenceArtifactError(
            "shared score tensors differ from the frozen schema"
        )
    sequence = np.asarray(tensors["sequence_id"])
    bitmap = np.asarray(tensors["position_bitmap"])
    count = np.asarray(tensors["token_count"])
    if (
        sequence.dtype != np.int64
        or sequence.ndim != 1
        or not sequence.size
        or np.any(sequence < 0)
        or not np.all(sequence[1:] > sequence[:-1])
    ):
        raise SharedEvidenceArtifactError(
            "score sequence IDs must be nonnegative, unique, and increasing"
        )
    if bitmap.dtype != np.uint64 or bitmap.shape != (
        sequence.size,
        _POSITION_BITMAP_WORDS,
    ):
        raise SharedEvidenceArtifactError(
            "score position bitmap has an incompatible dtype or shape"
        )
    if (
        count.dtype != np.uint16
        or count.shape != sequence.shape
        or np.any(count <= 0)
        or np.any(count > SEQUENCE_LENGTH)
        or not np.array_equal(count, _bitmap_counts(bitmap))
    ):
        raise SharedEvidenceArtifactError(
            "score position bitmap and token_count disagree"
        )
    result = {
        "sequence_id": _immutable(sequence, np.dtype("<i8")),
        "position_bitmap": _immutable(bitmap, np.dtype("<u8")),
        "token_count": _immutable(count, np.dtype("<u2")),
    }
    for name in metric_names:
        value = np.asarray(tensors[name])
        if (
            value.dtype != np.float64
            or value.shape != sequence.shape
            or not np.isfinite(value).all()
            or np.any(value < 0)
        ):
            raise SharedEvidenceArtifactError(
                f"score metric {name} must be finite nonnegative float64"
            )
        result[name] = _immutable(value, np.dtype("<f8"))
    return result


def _merge_aggregated_rows(
    rows: Sequence[Any],
    metric_names: Sequence[str],
) -> dict[str, np.ndarray]:
    if not rows:
        raise SharedEvidenceArtifactError("score manifest requires row artifacts")
    sequence = np.concatenate([row.sequence_id for row in rows])
    order = np.argsort(sequence, kind="stable")
    sequence = sequence[order]
    bitmap = np.concatenate([row.position_bitmap for row in rows], axis=0)[order]
    metrics = {
        name: np.concatenate([getattr(row, name) for row in rows])[order]
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
    for source, destination in enumerate(inverse):
        selected_bitmap = bitmap[source]
        if np.any(merged_bitmap[destination] & selected_bitmap):
            raise SharedEvidenceArtifactError(
                "score rows contain duplicate scientific token keys"
            )
        merged_bitmap[destination] |= selected_bitmap
        for name in metric_names:
            merged_metrics[name][destination] += metrics[name][source]
    return _validate_aggregated_rows(
        {
            "sequence_id": unique.astype("<i8", copy=False),
            "position_bitmap": merged_bitmap,
            "token_count": _bitmap_counts(merged_bitmap),
            **merged_metrics,
        },
        metric_names,
    )


def _validate_q_input_receipt(
    value: SharedQInputReceipt | Mapping[str, Any],
) -> SharedQInputReceipt:
    fields = {field.name for field in dataclasses.fields(SharedQInputReceipt)}
    raw = dataclasses.asdict(value) if isinstance(value, SharedQInputReceipt) else value
    selected = _exact_mapping("shared-Q input receipt", raw, fields)
    unsigned = {
        name: selected[name] for name in fields if name != "receipt_sha256"
    }
    claimed = _sha256("shared-Q input receipt SHA-256", selected["receipt_sha256"])
    if canonical_sha256(unsigned) != claimed:
        raise SharedEvidenceArtifactError(
            "shared-Q input receipt self-hash is invalid"
        )
    if selected["schema_version"] not in {
        LEGACY_SHARED_Q_INPUT_SCHEMA_VERSION,
        SHARED_Q_INPUT_SCHEMA_VERSION,
    }:
        raise SharedEvidenceArtifactError("shared-Q input receipt schema is invalid")
    for name in (
        "capture_manifest_sha256",
        "centering_contract_sha256",
        "recomposition_sidecar_sha256",
    ):
        _sha256(f"shared-Q input {name}", selected[name])
    parent = selected["centering_parent_manifest_sha256"]
    if parent is not None:
        _sha256("shared-Q centering parent manifest SHA-256", parent)
    if selected["role"] not in {"train", "evaluation"}:
        raise SharedEvidenceArtifactError("shared-Q input role is invalid")
    for name in (
        "total_tokens",
        "global_microbatch_tokens",
        "microbatches_per_update",
        "planned_updates",
        "process_count",
        "local_rows_per_microbatch",
    ):
        _positive_int(f"shared-Q input {name}", selected[name])
    _nonnegative_int("shared-Q process_index", selected["process_index"])
    _finite_nonnegative("shared-Q sealed_sst", selected["sealed_sst"])
    if selected["process_index"] >= selected["process_count"]:
        raise SharedEvidenceArtifactError(
            "shared-Q input process_index is outside process_count"
        )
    return SharedQInputReceipt(**dict(selected))


@dataclass(frozen=True, slots=True)
class FullFFNCenteringSufficient:
    """One sealed training target's FP64 sufficient statistics."""

    training_capture_manifest_sha256: str
    recomposition_sidecar_sha256: str
    token_count: int
    token_identity_sha256: str
    recomposition_chunk_tokens: int
    target_sum: np.ndarray
    target_squared_norm: float
    mean: np.ndarray
    mean_sha256: str
    sst: float


class FullFFNCenteringAccumulator:
    """Memory-bounded exact-target sufficient-statistics accumulator."""

    def __init__(
        self,
        *,
        training_capture_manifest_sha256: str,
        recomposition_sidecar: Mapping[str, Any],
        expected_tokens: int,
        recomposition_chunk_tokens: int = 128,
    ) -> None:
        self.training_capture_manifest_sha256 = _sha256(
            "training capture manifest SHA-256",
            training_capture_manifest_sha256,
        )
        self.sidecar_mapping = dict(recomposition_sidecar)
        self.sidecar = decode_recomposition_sidecar(recomposition_sidecar)
        self.expected_tokens = _positive_int("expected_tokens", expected_tokens)
        self.recomposition_chunk_tokens = _positive_int(
            "recomposition_chunk_tokens",
            recomposition_chunk_tokens,
        )
        self.count = 0
        self.target_sum = np.zeros((HIDDEN_SIZE,), dtype=np.float64)
        self.target_squared_norm = np.float64(0.0)
        self._identity = hashlib.sha256()
        self._identity.update(b"gemma4-scientific-token-identity-v1\0")
        self._identity.update(
            self.expected_tokens.to_bytes(8, "little", signed=False)
        )

    @staticmethod
    def _slice(batch: ContributionBatch, start: int, end: int) -> ContributionBatch:
        return ContributionBatch(
            **{
                name: np.ascontiguousarray(
                    np.asarray(getattr(batch, name))[start:end]
                )
                for name in ContributionBatch.__dataclass_fields__
            }
        )

    def add(self, batch: ContributionBatch | Mapping[str, Any], *, range_start: int) -> None:
        checked = validate_contribution_batch(batch)
        if range_start != self.count:
            raise SharedEvidenceArtifactError(
                "full-FFN centering ranges are not a contiguous ordered prefix"
            )
        if self.count + checked.tokens > self.expected_tokens:
            raise SharedEvidenceArtifactError(
                "full-FFN centering exceeds its declared token count"
            )
        records = np.empty(checked.tokens, dtype=_TOKEN_RECORD_DTYPE)
        records["sequence_id"] = checked.sequence_id
        records["position"] = checked.position
        self._identity.update(records.tobytes(order="C"))
        for start in range(0, checked.tokens, self.recomposition_chunk_tokens):
            end = min(start + self.recomposition_chunk_tokens, checked.tokens)
            stages = recompose_gemma4(
                self._slice(checked, start, end),
                self.sidecar_mapping,
            )
            target = bf16_bits_to_float32(
                stages.outer_ffn_post_norm_bf16_bits
            ).astype(np.float64)
            self.target_sum += target.sum(axis=0, dtype=np.float64)
            self.target_squared_norm += np.square(
                target,
                dtype=np.float64,
            ).sum(dtype=np.float64)
        self.count += checked.tokens

    def finalize(self) -> FullFFNCenteringSufficient:
        if self.count != self.expected_tokens:
            raise SharedEvidenceArtifactError(
                "full-FFN centering did not consume its exact declared capture"
            )
        mean = np.ascontiguousarray(
            (self.target_sum / self.count).astype(np.float32)
        )
        mean_sha = hashlib.sha256(
            b"gemma4-full-ffn-mean-v1\0"
            + np.asarray(mean, dtype="<f4").tobytes(order="C")
        ).hexdigest()
        mean64 = mean.astype(np.float64)
        sst = float(
            self.target_squared_norm
            - 2.0 * np.dot(mean64, self.target_sum)
            + self.count * np.dot(mean64, mean64)
        )
        if (
            not np.isfinite(self.target_sum).all()
            or not math.isfinite(float(self.target_squared_norm))
            or float(self.target_squared_norm) <= 0
            or not math.isfinite(sst)
            or sst <= 0
        ):
            raise SharedEvidenceArtifactError(
                "full-FFN centering statistics are non-finite or degenerate"
            )
        return FullFFNCenteringSufficient(
            training_capture_manifest_sha256=(
                self.training_capture_manifest_sha256
            ),
            recomposition_sidecar_sha256=self.sidecar.sidecar_sha256,
            token_count=self.count,
            token_identity_sha256=self._identity.hexdigest(),
            recomposition_chunk_tokens=self.recomposition_chunk_tokens,
            target_sum=np.ascontiguousarray(self.target_sum),
            target_squared_norm=float(self.target_squared_norm),
            mean=mean,
            mean_sha256=mean_sha,
            sst=sst,
        )


@dataclass(frozen=True, slots=True)
class ResolvedFullFFNCentering:
    reference: SharedEvidenceRef
    training_capture_manifest_sha256: str
    recomposition_sidecar_sha256: str
    token_count: int
    token_identity_sha256: str
    recomposition_chunk_tokens: int
    target_sum: np.ndarray
    target_squared_norm: float
    mean: np.ndarray
    mean_sha256: str
    sst: float


def seal_full_ffn_centering(
    sufficient: FullFFNCenteringSufficient,
) -> tuple[SharedEvidenceRef, bytes]:
    for name in (
        "training_capture_manifest_sha256",
        "recomposition_sidecar_sha256",
        "token_identity_sha256",
        "mean_sha256",
    ):
        _sha256(name, getattr(sufficient, name))
    count = _positive_int("full-FFN token_count", sufficient.token_count)
    chunk = _positive_int(
        "full-FFN recomposition_chunk_tokens",
        sufficient.recomposition_chunk_tokens,
    )
    target_sum = np.asarray(sufficient.target_sum)
    mean = np.asarray(sufficient.mean)
    if (
        target_sum.dtype != np.float64
        or target_sum.shape != (HIDDEN_SIZE,)
        or mean.dtype != np.float32
        or mean.shape != (HIDDEN_SIZE,)
        or not np.isfinite(target_sum).all()
        or not np.isfinite(mean).all()
    ):
        raise SharedEvidenceArtifactError(
            "full-FFN centering vectors have incompatible dtype, shape, or values"
        )
    squared = _finite_nonnegative(
        "full-FFN target_squared_norm",
        sufficient.target_squared_norm,
    )
    sst = _finite_nonnegative("full-FFN SST", sufficient.sst)
    if squared <= 0 or sst <= 0:
        raise SharedEvidenceArtifactError(
            "full-FFN squared norm and SST must be positive"
        )
    expected_mean = (target_sum / count).astype(np.float32)
    if not np.array_equal(expected_mean, mean):
        raise SharedEvidenceArtifactError(
            "full-FFN mean differs from its FP64 sufficient sum"
        )
    expected_mean_sha = hashlib.sha256(
        b"gemma4-full-ffn-mean-v1\0"
        + np.asarray(mean, dtype="<f4").tobytes(order="C")
    ).hexdigest()
    if expected_mean_sha != sufficient.mean_sha256:
        raise SharedEvidenceArtifactError("full-FFN mean hash is invalid")
    mean64 = mean.astype(np.float64)
    expected_sst = float(
        squared
        - 2.0 * np.dot(mean64, target_sum)
        + count * np.dot(mean64, mean64)
    )
    if not math.isclose(
        sst,
        expected_sst,
        rel_tol=1e-12,
        abs_tol=1e-9 * max(1.0, abs(expected_sst)),
    ):
        raise SharedEvidenceArtifactError(
            "full-FFN SST differs from its sufficient statistics"
        )
    metadata = {
        "schema_version": FULL_FFN_CENTERING_SCHEMA_VERSION,
        "training_capture_manifest_sha256": (
            sufficient.training_capture_manifest_sha256
        ),
        "recomposition_sidecar_sha256": (
            sufficient.recomposition_sidecar_sha256
        ),
        "token_count": count,
        "token_identity_sha256": sufficient.token_identity_sha256,
        "recomposition_chunk_tokens": chunk,
        "mean_sha256": sufficient.mean_sha256,
        "accumulation_order": (
            "logical_range_order_chunked_recompose_bf16_target_fp64_sum_v1"
        ),
    }
    blob = _encode_envelope(
        metadata,
        {
            "target_sum": target_sum,
            "target_squared_norm": np.asarray([squared], dtype="<f8"),
            "mean": mean,
            "sst": np.asarray([sst], dtype="<f8"),
        },
    )
    return _reference(blob, media_type=FULL_FFN_CENTERING_MEDIA_TYPE), blob


def resolve_full_ffn_centering(
    reference: SharedEvidenceRef | Mapping[str, Any],
    resolver: SharedEvidenceResolver,
) -> ResolvedFullFFNCentering:
    selected_ref = SharedEvidenceRef.from_value(
        reference,
        expected_media_type=FULL_FFN_CENTERING_MEDIA_TYPE,
    )
    metadata, tensors = _decode_envelope(_resolve_bytes(selected_ref, resolver))
    selected = _exact_mapping(
        "full-FFN centering metadata",
        metadata,
        {
            "schema_version",
            "training_capture_manifest_sha256",
            "recomposition_sidecar_sha256",
            "token_count",
            "token_identity_sha256",
            "recomposition_chunk_tokens",
            "mean_sha256",
            "accumulation_order",
        },
    )
    if selected["schema_version"] != FULL_FFN_CENTERING_SCHEMA_VERSION:
        raise SharedEvidenceArtifactError("full-FFN centering schema is invalid")
    if selected["accumulation_order"] != (
        "logical_range_order_chunked_recompose_bf16_target_fp64_sum_v1"
    ):
        raise SharedEvidenceArtifactError(
            "full-FFN centering accumulation order is unsupported"
        )
    if set(tensors) != {
        "target_sum",
        "target_squared_norm",
        "mean",
        "sst",
    }:
        raise SharedEvidenceArtifactError(
            "full-FFN centering tensors differ from the frozen schema"
        )
    target_sum = np.asarray(tensors["target_sum"])
    squared_norm = np.asarray(tensors["target_squared_norm"])
    mean = np.asarray(tensors["mean"])
    sst = np.asarray(tensors["sst"])
    if (
        target_sum.dtype != np.float64
        or target_sum.shape != (HIDDEN_SIZE,)
        or squared_norm.dtype != np.float64
        or squared_norm.shape != (1,)
        or mean.dtype != np.float32
        or mean.shape != (HIDDEN_SIZE,)
        or sst.dtype != np.float64
        or sst.shape != (1,)
    ):
        raise SharedEvidenceArtifactError(
            "full-FFN centering tensor dtypes or shapes are incompatible"
        )
    sufficient = FullFFNCenteringSufficient(
        training_capture_manifest_sha256=_sha256(
            "training capture manifest SHA-256",
            selected["training_capture_manifest_sha256"],
        ),
        recomposition_sidecar_sha256=_sha256(
            "recomposition sidecar SHA-256",
            selected["recomposition_sidecar_sha256"],
        ),
        token_count=_positive_int("full-FFN token_count", selected["token_count"]),
        token_identity_sha256=_sha256(
            "full-FFN token identity SHA-256",
            selected["token_identity_sha256"],
        ),
        recomposition_chunk_tokens=_positive_int(
            "full-FFN recomposition chunk",
            selected["recomposition_chunk_tokens"],
        ),
        target_sum=target_sum,
        target_squared_norm=float(squared_norm[0]),
        mean=mean,
        mean_sha256=_sha256("full-FFN mean SHA-256", selected["mean_sha256"]),
        sst=float(sst[0]),
    )
    # The sealer is also the single numerical validator.
    verified_ref, verified_blob = seal_full_ffn_centering(sufficient)
    if verified_blob != _resolve_bytes(selected_ref, resolver):
        raise SharedEvidenceArtifactError(
            "full-FFN centering is not canonically encoded"
        )
    if verified_ref != selected_ref:
        raise SharedEvidenceArtifactError(
            "full-FFN centering canonical reference changed"
        )
    target_sum = _immutable(sufficient.target_sum, np.dtype("<f8"))
    mean = _immutable(sufficient.mean, np.dtype("<f4"))
    return ResolvedFullFFNCentering(
        reference=selected_ref,
        training_capture_manifest_sha256=(
            sufficient.training_capture_manifest_sha256
        ),
        recomposition_sidecar_sha256=(
            sufficient.recomposition_sidecar_sha256
        ),
        token_count=sufficient.token_count,
        token_identity_sha256=sufficient.token_identity_sha256,
        recomposition_chunk_tokens=sufficient.recomposition_chunk_tokens,
        target_sum=target_sum,
        target_squared_norm=sufficient.target_squared_norm,
        mean=mean,
        mean_sha256=sufficient.mean_sha256,
        sst=sufficient.sst,
    )


def seal_full_ffn_centering_from_dataset(
    dataset: Any,
    recomposition_sidecar: Mapping[str, Any],
    *,
    recomposition_chunk_tokens: int = 128,
) -> tuple[SharedEvidenceRef, bytes]:
    """Stream one complete sealed training capture into a centering artifact."""

    manifest = getattr(dataset, "manifest", None)
    try:
        checked = validate_contribution_manifest(manifest)
    except Exception as error:
        raise SharedEvidenceArtifactError(
            "full-FFN centering requires a valid sealed v2 manifest"
        ) from error
    if checked["schema_version"] != SCHEMA_VERSION:
        raise SharedEvidenceArtifactError(
            "full-FFN centering requires exact contribution-v2"
        )
    if checked["metadata"]["split"] != "train":
        raise SharedEvidenceArtifactError(
            "full-FFN centering requires the training split"
        )
    manifest_sha = getattr(dataset, "manifest_sha256", None)
    if manifest_sha is None and isinstance(manifest, Mapping):
        manifest_sha = manifest.get("manifest_sha256")
    if manifest_sha != checked["manifest_sha256"]:
        raise SharedEvidenceArtifactError(
            "full-FFN dataset and manifest hashes differ"
        )
    sidecar = decode_recomposition_sidecar(recomposition_sidecar)
    if checked["recomposition_sidecar_sha256"] != sidecar.sidecar_sha256:
        raise SharedEvidenceArtifactError(
            "full-FFN centering sidecar differs from the capture"
        )
    total = int(checked["total_tokens"])
    accumulator = FullFFNCenteringAccumulator(
        training_capture_manifest_sha256=manifest_sha,
        recomposition_sidecar=recomposition_sidecar,
        expected_tokens=total,
        recomposition_chunk_tokens=recomposition_chunk_tokens,
    )
    try:
        pieces = dataset.iter_token_range(0, total)
        for piece in pieces:
            accumulator.add(piece, range_start=accumulator.count)
    except SharedEvidenceArtifactError:
        raise
    except Exception as error:
        raise SharedEvidenceArtifactError(
            "cannot stream full-FFN training targets from the sealed capture"
        ) from error
    return seal_full_ffn_centering(accumulator.finalize())


_Q_METRICS = ("q_sse", "q_sst", "q_l0_sum")
_FULL_METRICS = (
    "oracle_shared_sse",
    "learned_shared_sse",
    "full_ffn_sst",
)


@dataclass(frozen=True, slots=True)
class ResolvedSharedQScoreRows:
    reference: SharedEvidenceRef
    seed: int
    process_index: int
    update_index: int
    evaluation_capture_manifest_sha256: str
    centering_contract_sha256: str
    checkpoint_sha256: str
    recomposition_sidecar_sha256: str
    evaluation_input_receipt_sha256: str
    topk: int
    sequence_id: np.ndarray
    position_bitmap: np.ndarray
    token_count: np.ndarray
    q_sse: np.ndarray
    q_sst: np.ndarray
    q_l0_sum: np.ndarray

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


def seal_shared_q_score_row(
    evidence: SharedQEvaluationEvidenceChunk,
    *,
    seed: int,
    process_index: int,
    evaluation_input_receipt: SharedQInputReceipt | Mapping[str, Any],
    topk: int = 64,
) -> tuple[SharedEvidenceRef, bytes]:
    receipt = _validate_q_input_receipt(evaluation_input_receipt)
    if receipt.role != "evaluation":
        raise SharedEvidenceArtifactError(
            "shared-Q score row requires an evaluation receipt"
        )
    selected_seed = _nonnegative_int("shared-Q seed", seed)
    selected_process = _nonnegative_int("shared-Q process_index", process_index)
    selected_topk = _positive_int("shared-Q row topk", topk)
    if (
        selected_process != receipt.process_index
        or evidence.capture_manifest_sha256 != receipt.capture_manifest_sha256
        or evidence.centering_contract_sha256 != receipt.centering_contract_sha256
        or evidence.recomposition_sidecar_sha256
        != receipt.recomposition_sidecar_sha256
    ):
        raise SharedEvidenceArtifactError(
            "shared-Q evidence differs from its evaluation receipt"
        )
    valid = np.asarray(evidence.rows.valid_mask)
    if not np.count_nonzero(valid):
        raise SharedEvidenceArtifactError(
            "shared-Q score row cannot contain zero scientific tokens"
        )
    if np.any(np.asarray(evidence.rows.l0)[valid] != selected_topk):
        raise SharedEvidenceArtifactError(
            "shared-Q score row violates exact per-token TopK"
        )
    sequence = np.asarray(evidence.sequence_id)[valid]
    position = np.asarray(evidence.position)[valid]
    arrays = _aggregate_token_metrics(
        sequence,
        position,
        {
            "q_sse": np.asarray(evidence.rows.token_sse)[valid],
            "q_sst": np.asarray(evidence.rows.token_sst)[valid],
            "q_l0_sum": np.asarray(evidence.rows.l0)[valid],
        },
    )
    metadata = {
        "schema_version": SHARED_Q_SCORE_ROW_SCHEMA_VERSION,
        "seed": selected_seed,
        "process_index": selected_process,
        "update_index": _nonnegative_int(
            "shared-Q update_index",
            evidence.update_index,
        ),
        "evaluation_capture_manifest_sha256": (
            evidence.capture_manifest_sha256
        ),
        "centering_contract_sha256": evidence.centering_contract_sha256,
        "checkpoint_sha256": evidence.checkpoint_sha256,
        "recomposition_sidecar_sha256": (
            evidence.recomposition_sidecar_sha256
        ),
        "evaluation_input_receipt_sha256": receipt.receipt_sha256,
        "topk": selected_topk,
        "tokens": int(arrays["token_count"].sum(dtype=np.int64)),
        "token_identity_sha256": _bitmap_token_identity_sha256(
            arrays["sequence_id"],
            arrays["position_bitmap"],
            arrays["token_count"],
        ),
    }
    blob = _encode_envelope(metadata, arrays)
    return _reference(blob, media_type=SHARED_Q_SCORE_ROW_MEDIA_TYPE), blob


def resolve_shared_q_score_row(
    reference: SharedEvidenceRef | Mapping[str, Any],
    resolver: SharedEvidenceResolver,
) -> ResolvedSharedQScoreRows:
    selected_ref = SharedEvidenceRef.from_value(
        reference,
        expected_media_type=SHARED_Q_SCORE_ROW_MEDIA_TYPE,
    )
    metadata, raw = _decode_envelope(_resolve_bytes(selected_ref, resolver))
    selected = _exact_mapping(
        "shared-Q score row metadata",
        metadata,
        {
            "schema_version",
            "seed",
            "process_index",
            "update_index",
            "evaluation_capture_manifest_sha256",
            "centering_contract_sha256",
            "checkpoint_sha256",
            "recomposition_sidecar_sha256",
            "evaluation_input_receipt_sha256",
            "topk",
            "tokens",
            "token_identity_sha256",
        },
    )
    if selected["schema_version"] != SHARED_Q_SCORE_ROW_SCHEMA_VERSION:
        raise SharedEvidenceArtifactError("shared-Q score row schema is invalid")
    arrays = _validate_aggregated_rows(raw, _Q_METRICS)
    tokens = int(arrays["token_count"].sum(dtype=np.int64))
    if tokens != _positive_int("shared-Q row tokens", selected["tokens"]):
        raise SharedEvidenceArtifactError(
            "shared-Q row token count differs from its payload"
        )
    identity = _bitmap_token_identity_sha256(
        arrays["sequence_id"],
        arrays["position_bitmap"],
        arrays["token_count"],
    )
    if identity != _sha256(
        "shared-Q row token identity",
        selected["token_identity_sha256"],
    ):
        raise SharedEvidenceArtifactError(
            "shared-Q row token identity differs from its payload"
        )
    for name in (
        "evaluation_capture_manifest_sha256",
        "centering_contract_sha256",
        "checkpoint_sha256",
        "recomposition_sidecar_sha256",
        "evaluation_input_receipt_sha256",
    ):
        _sha256(f"shared-Q row {name}", selected[name])
    topk = _positive_int("shared-Q row topk", selected["topk"])
    if float(arrays["q_l0_sum"].sum(dtype=np.float64)) != tokens * topk:
        raise SharedEvidenceArtifactError(
            "shared-Q row payload violates exact TopK cardinality"
        )
    return ResolvedSharedQScoreRows(
        reference=selected_ref,
        seed=_nonnegative_int("shared-Q row seed", selected["seed"]),
        process_index=_nonnegative_int(
            "shared-Q row process_index",
            selected["process_index"],
        ),
        update_index=_nonnegative_int(
            "shared-Q row update_index",
            selected["update_index"],
        ),
        evaluation_capture_manifest_sha256=(
            selected["evaluation_capture_manifest_sha256"]
        ),
        centering_contract_sha256=selected["centering_contract_sha256"],
        checkpoint_sha256=selected["checkpoint_sha256"],
        recomposition_sidecar_sha256=(
            selected["recomposition_sidecar_sha256"]
        ),
        evaluation_input_receipt_sha256=(
            selected["evaluation_input_receipt_sha256"]
        ),
        topk=topk,
        **arrays,
    )


@dataclass(frozen=True, slots=True)
class ResolvedSharedQScoreManifest:
    reference: SharedEvidenceRef
    phase: str
    protocol_sha256: str
    seed: int
    training_capture_manifest_sha256: str
    evaluation_capture_manifest_sha256: str
    recomposition_sidecar_sha256: str
    run_manifest_sha256: str
    config_sha256: str
    checkpoint_sha256: str
    centering_contract_sha256: str
    training_tokens: int
    training_input_receipt_sha256s: tuple[str, ...]
    evaluation_input_receipts: tuple[SharedQInputReceipt, ...]
    row_references: tuple[SharedEvidenceRef, ...]
    sequence_id: np.ndarray
    position_bitmap: np.ndarray
    token_count: np.ndarray
    q_sse: np.ndarray
    q_sst: np.ndarray
    q_l0_sum: np.ndarray
    topk: int

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


def _normalize_q_receipts(
    values: Sequence[SharedQInputReceipt | Mapping[str, Any]],
    *,
    role: str,
    capture_manifest_sha256: str,
    centering_contract_sha256: str,
    recomposition_sidecar_sha256: str,
    expected_tokens: int,
) -> tuple[SharedQInputReceipt, ...]:
    receipts = tuple(_validate_q_input_receipt(value) for value in values)
    if not receipts:
        raise SharedEvidenceArtifactError(
            f"shared-Q {role} receipt set is empty"
        )
    process_counts = {item.process_count for item in receipts}
    if len(process_counts) != 1:
        raise SharedEvidenceArtifactError(
            f"shared-Q {role} receipts disagree on process_count"
        )
    process_count = next(iter(process_counts))
    if (
        len(receipts) != process_count
        or {item.process_index for item in receipts} != set(range(process_count))
    ):
        raise SharedEvidenceArtifactError(
            f"shared-Q {role} process receipt set is incomplete"
        )
    schedule = {
        (
            item.total_tokens,
            item.global_microbatch_tokens,
            item.microbatches_per_update,
            item.planned_updates,
            item.local_rows_per_microbatch,
            item.sealed_sst,
        )
        for item in receipts
    }
    if len(schedule) != 1:
        raise SharedEvidenceArtifactError(
            f"shared-Q {role} receipts disagree on schedule or SST"
        )
    for item in receipts:
        if (
            item.role != role
            or item.capture_manifest_sha256 != capture_manifest_sha256
            or item.centering_contract_sha256 != centering_contract_sha256
            or item.recomposition_sidecar_sha256
            != recomposition_sidecar_sha256
            or item.total_tokens != expected_tokens
        ):
            raise SharedEvidenceArtifactError(
                f"shared-Q {role} receipt differs from manifest identity"
            )
    return tuple(sorted(receipts, key=lambda item: item.process_index))


def _resolve_q_rows(
    references: Sequence[SharedEvidenceRef],
    resolver: SharedEvidenceResolver,
    *,
    seed: int,
    checkpoint_sha256: str,
    evaluation_capture_manifest_sha256: str,
    centering_contract_sha256: str,
    recomposition_sidecar_sha256: str,
    receipts: Sequence[SharedQInputReceipt],
) -> tuple[tuple[ResolvedSharedQScoreRows, ...], dict[str, np.ndarray]]:
    rows = tuple(resolve_shared_q_score_row(item, resolver) for item in references)
    receipt_by_process = {item.process_index: item for item in receipts}
    seen: set[tuple[int, int]] = set()
    for row in rows:
        key = (row.process_index, row.update_index)
        receipt = receipt_by_process.get(row.process_index)
        if key in seen:
            raise SharedEvidenceArtifactError(
                "shared-Q rows duplicate a process/update"
            )
        seen.add(key)
        if (
            receipt is None
            or row.seed != seed
            or row.checkpoint_sha256 != checkpoint_sha256
            or row.evaluation_capture_manifest_sha256
            != evaluation_capture_manifest_sha256
            or row.centering_contract_sha256 != centering_contract_sha256
            or row.recomposition_sidecar_sha256
            != recomposition_sidecar_sha256
            or row.evaluation_input_receipt_sha256 != receipt.receipt_sha256
            or row.topk <= 0
        ):
            raise SharedEvidenceArtifactError(
                "shared-Q row differs from its manifest identity"
            )
    expected = {
        (receipt.process_index, update)
        for receipt in receipts
        for update in range(receipt.planned_updates)
    }
    if seen != expected:
        raise SharedEvidenceArtifactError(
            "shared-Q row process/update coverage is incomplete"
        )
    return rows, _merge_aggregated_rows(rows, _Q_METRICS)


def seal_shared_q_score_manifest(
    row_references: Sequence[SharedEvidenceRef | Mapping[str, Any]],
    resolver: SharedEvidenceResolver,
    *,
    phase: str,
    protocol_sha256: str,
    seed: int,
    training_capture_manifest_sha256: str,
    evaluation_capture_manifest_sha256: str,
    recomposition_sidecar_sha256: str,
    run_manifest_sha256: str,
    config_sha256: str,
    checkpoint_sha256: str,
    centering_contract_sha256: str,
    training_input_receipts: Sequence[SharedQInputReceipt | Mapping[str, Any]],
    evaluation_input_receipts: Sequence[SharedQInputReceipt | Mapping[str, Any]],
    expected_tokens: int,
    topk: int = 64,
) -> tuple[SharedEvidenceRef, bytes]:
    if not isinstance(phase, str) or not phase:
        raise SharedEvidenceArtifactError("shared-Q phase must be nonempty")
    for name, value in (
        ("protocol_sha256", protocol_sha256),
        ("training_capture_manifest_sha256", training_capture_manifest_sha256),
        (
            "evaluation_capture_manifest_sha256",
            evaluation_capture_manifest_sha256,
        ),
        ("recomposition_sidecar_sha256", recomposition_sidecar_sha256),
        ("run_manifest_sha256", run_manifest_sha256),
        ("config_sha256", config_sha256),
        ("checkpoint_sha256", checkpoint_sha256),
        ("centering_contract_sha256", centering_contract_sha256),
    ):
        _sha256(name, value)
    selected_seed = _nonnegative_int("shared-Q seed", seed)
    tokens = _positive_int("shared-Q expected_tokens", expected_tokens)
    selected_topk = _positive_int("shared-Q topk", topk)
    if not training_input_receipts:
        raise SharedEvidenceArtifactError(
            "shared-Q training receipt set is empty"
        )
    training = _normalize_q_receipts(
        training_input_receipts,
        role="train",
        capture_manifest_sha256=training_capture_manifest_sha256,
        centering_contract_sha256=centering_contract_sha256,
        recomposition_sidecar_sha256=recomposition_sidecar_sha256,
        expected_tokens=training_input_receipts[0].total_tokens
        if isinstance(training_input_receipts[0], SharedQInputReceipt)
        else int(training_input_receipts[0]["total_tokens"]),
    )
    evaluation = _normalize_q_receipts(
        evaluation_input_receipts,
        role="evaluation",
        capture_manifest_sha256=evaluation_capture_manifest_sha256,
        centering_contract_sha256=centering_contract_sha256,
        recomposition_sidecar_sha256=recomposition_sidecar_sha256,
        expected_tokens=tokens,
    )
    references = tuple(
        SharedEvidenceRef.from_value(
            item,
            expected_media_type=SHARED_Q_SCORE_ROW_MEDIA_TYPE,
        )
        for item in row_references
    )
    if len({item.sha256 for item in references}) != len(references):
        raise SharedEvidenceArtifactError(
            "shared-Q manifest contains duplicate row artifacts"
        )
    resolved_rows, merged = _resolve_q_rows(
        references,
        resolver,
        seed=selected_seed,
        checkpoint_sha256=checkpoint_sha256,
        evaluation_capture_manifest_sha256=evaluation_capture_manifest_sha256,
        centering_contract_sha256=centering_contract_sha256,
        recomposition_sidecar_sha256=recomposition_sidecar_sha256,
        receipts=evaluation,
    )
    if any(row.topk != selected_topk for row in resolved_rows):
        raise SharedEvidenceArtifactError(
            "shared-Q row TopK differs from its manifest"
        )
    if int(merged["token_count"].sum(dtype=np.int64)) != tokens:
        raise SharedEvidenceArtifactError(
            "shared-Q rows do not cover the exact evaluation capture"
        )
    l0_sum = float(merged["q_l0_sum"].sum(dtype=np.float64))
    if l0_sum != tokens * selected_topk:
        raise SharedEvidenceArtifactError(
            "shared-Q rows violate exact TopK cardinality"
        )
    unsigned = {
        "schema_version": SHARED_Q_SCORE_MANIFEST_SCHEMA_VERSION,
        "phase": phase,
        "protocol_sha256": protocol_sha256,
        "seed": selected_seed,
        "training_capture_manifest_sha256": training_capture_manifest_sha256,
        "evaluation_capture_manifest_sha256": (
            evaluation_capture_manifest_sha256
        ),
        "recomposition_sidecar_sha256": recomposition_sidecar_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "centering_contract_sha256": centering_contract_sha256,
        "training_tokens": training[0].total_tokens,
        "training_input_receipts": [
            dataclasses.asdict(item) for item in training
        ],
        "evaluation_input_receipts": [
            dataclasses.asdict(item) for item in evaluation
        ],
        "row_artifacts": [item.to_dict() for item in references],
        "tokens": tokens,
        "token_identity_sha256": _bitmap_token_identity_sha256(
            merged["sequence_id"],
            merged["position_bitmap"],
            merged["token_count"],
        ),
        "topk": selected_topk,
        "q_sse": float(merged["q_sse"].sum(dtype=np.float64)),
        "q_sst": float(merged["q_sst"].sum(dtype=np.float64)),
        "q_l0_sum": l0_sum,
    }
    blob = canonical_json_bytes(unsigned)
    return _reference(blob, media_type=SHARED_Q_SCORE_MANIFEST_MEDIA_TYPE), blob


def resolve_shared_q_score_manifest(
    reference: SharedEvidenceRef | Mapping[str, Any],
    resolver: SharedEvidenceResolver,
) -> ResolvedSharedQScoreManifest:
    selected_ref = SharedEvidenceRef.from_value(
        reference,
        expected_media_type=SHARED_Q_SCORE_MANIFEST_MEDIA_TYPE,
    )
    blob = _resolve_bytes(selected_ref, resolver)
    try:
        raw = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SharedEvidenceArtifactError(
            "shared-Q score manifest is not JSON"
        ) from error
    if canonical_json_bytes(raw) != blob:
        raise SharedEvidenceArtifactError(
            "shared-Q score manifest is not canonical"
        )
    selected = _exact_mapping(
        "shared-Q score manifest",
        raw,
        {
            "schema_version",
            "phase",
            "protocol_sha256",
            "seed",
            "training_capture_manifest_sha256",
            "evaluation_capture_manifest_sha256",
            "recomposition_sidecar_sha256",
            "run_manifest_sha256",
            "config_sha256",
            "checkpoint_sha256",
            "centering_contract_sha256",
            "training_tokens",
            "training_input_receipts",
            "evaluation_input_receipts",
            "row_artifacts",
            "tokens",
            "token_identity_sha256",
            "topk",
            "q_sse",
            "q_sst",
            "q_l0_sum",
        },
    )
    if selected["schema_version"] != SHARED_Q_SCORE_MANIFEST_SCHEMA_VERSION:
        raise SharedEvidenceArtifactError(
            "shared-Q score manifest schema is invalid"
        )
    phase = selected["phase"]
    if not isinstance(phase, str) or not phase:
        raise SharedEvidenceArtifactError("shared-Q manifest phase is invalid")
    for name in (
        "protocol_sha256",
        "training_capture_manifest_sha256",
        "evaluation_capture_manifest_sha256",
        "recomposition_sidecar_sha256",
        "run_manifest_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "centering_contract_sha256",
        "token_identity_sha256",
    ):
        _sha256(f"shared-Q manifest {name}", selected[name])
    seed = _nonnegative_int("shared-Q manifest seed", selected["seed"])
    tokens = _positive_int("shared-Q manifest tokens", selected["tokens"])
    topk = _positive_int("shared-Q manifest topk", selected["topk"])
    training_tokens = _positive_int(
        "shared-Q manifest training_tokens",
        selected["training_tokens"],
    )
    if not isinstance(selected["training_input_receipts"], list) or not isinstance(
        selected["evaluation_input_receipts"],
        list,
    ):
        raise SharedEvidenceArtifactError(
            "shared-Q manifest receipt inventories must be lists"
        )
    if not selected["training_input_receipts"]:
        raise SharedEvidenceArtifactError(
            "shared-Q manifest training receipts are empty"
        )
    training = _normalize_q_receipts(
        selected["training_input_receipts"],
        role="train",
        capture_manifest_sha256=selected["training_capture_manifest_sha256"],
        centering_contract_sha256=selected["centering_contract_sha256"],
        recomposition_sidecar_sha256=selected["recomposition_sidecar_sha256"],
        expected_tokens=training_tokens,
    )
    evaluation = _normalize_q_receipts(
        selected["evaluation_input_receipts"],
        role="evaluation",
        capture_manifest_sha256=selected["evaluation_capture_manifest_sha256"],
        centering_contract_sha256=selected["centering_contract_sha256"],
        recomposition_sidecar_sha256=selected["recomposition_sidecar_sha256"],
        expected_tokens=tokens,
    )
    if not isinstance(selected["row_artifacts"], list):
        raise SharedEvidenceArtifactError(
            "shared-Q manifest row_artifacts must be a list"
        )
    references = tuple(
        SharedEvidenceRef.from_value(
            item,
            expected_media_type=SHARED_Q_SCORE_ROW_MEDIA_TYPE,
        )
        for item in selected["row_artifacts"]
    )
    resolved_rows, merged = _resolve_q_rows(
        references,
        resolver,
        seed=seed,
        checkpoint_sha256=selected["checkpoint_sha256"],
        evaluation_capture_manifest_sha256=(
            selected["evaluation_capture_manifest_sha256"]
        ),
        centering_contract_sha256=selected["centering_contract_sha256"],
        recomposition_sidecar_sha256=selected["recomposition_sidecar_sha256"],
        receipts=evaluation,
    )
    if any(row.topk != topk for row in resolved_rows):
        raise SharedEvidenceArtifactError(
            "shared-Q manifest row TopK is inconsistent"
        )
    if int(merged["token_count"].sum(dtype=np.int64)) != tokens:
        raise SharedEvidenceArtifactError(
            "shared-Q manifest token coverage is incomplete"
        )
    if (
        _bitmap_token_identity_sha256(
            merged["sequence_id"],
            merged["position_bitmap"],
            merged["token_count"],
        )
        != selected["token_identity_sha256"]
    ):
        raise SharedEvidenceArtifactError(
            "shared-Q manifest token identity differs from its rows"
        )
    totals = {
        name: float(merged[name].sum(dtype=np.float64))
        for name in _Q_METRICS
    }
    for name, claimed_name in (
        ("q_sse", "q_sse"),
        ("q_sst", "q_sst"),
        ("q_l0_sum", "q_l0_sum"),
    ):
        claimed = _finite_nonnegative(
            f"shared-Q manifest {claimed_name}",
            selected[claimed_name],
        )
        if not math.isclose(
            claimed,
            totals[name],
            rel_tol=1e-12,
            abs_tol=1e-10 * max(1.0, abs(totals[name])),
        ):
            raise SharedEvidenceArtifactError(
                f"shared-Q manifest {claimed_name} differs from its rows"
            )
    if totals["q_sst"] <= 0 or totals["q_l0_sum"] != tokens * topk:
        raise SharedEvidenceArtifactError(
            "shared-Q manifest SST/L0 is invalid"
        )
    return ResolvedSharedQScoreManifest(
        reference=selected_ref,
        phase=phase,
        protocol_sha256=selected["protocol_sha256"],
        seed=seed,
        training_capture_manifest_sha256=(
            selected["training_capture_manifest_sha256"]
        ),
        evaluation_capture_manifest_sha256=(
            selected["evaluation_capture_manifest_sha256"]
        ),
        recomposition_sidecar_sha256=(
            selected["recomposition_sidecar_sha256"]
        ),
        run_manifest_sha256=selected["run_manifest_sha256"],
        config_sha256=selected["config_sha256"],
        checkpoint_sha256=selected["checkpoint_sha256"],
        centering_contract_sha256=selected["centering_contract_sha256"],
        training_tokens=training_tokens,
        training_input_receipt_sha256s=tuple(
            item.receipt_sha256 for item in training
        ),
        evaluation_input_receipts=evaluation,
        row_references=references,
        topk=topk,
        **merged,
    )


@dataclass(frozen=True, slots=True)
class ResolvedFullFFNScoreRows:
    reference: SharedEvidenceRef
    condition: str
    seed: int
    process_index: int
    update_index: int
    evaluation_capture_manifest_sha256: str
    routed_checkpoint_sha256: str
    shared_q_capture_manifest_sha256: str
    shared_q_centering_contract_sha256: str
    shared_q_checkpoint_sha256: str
    recomposition_sidecar_sha256: str
    full_ffn_centering_sha256: str
    full_ffn_mean_sha256: str
    routed_score_row_sha256: str
    shared_q_score_row_sha256: str
    sequence_id: np.ndarray
    position_bitmap: np.ndarray
    token_count: np.ndarray
    oracle_shared_sse: np.ndarray
    learned_shared_sse: np.ndarray
    full_ffn_sst: np.ndarray

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


def seal_full_ffn_score_row(
    rows: FullFFNRecompositionRows,
    centering: ResolvedFullFFNCentering,
    *,
    condition: str,
    seed: int,
    process_index: int,
    update_index: int,
    evaluation_capture_manifest_sha256: str,
    routed_checkpoint_sha256: str,
    routed_score_row_sha256: str,
    shared_q_score_row_sha256: str,
) -> tuple[SharedEvidenceRef, bytes]:
    if condition not in {"A", "R", "B", "F"}:
        raise SharedEvidenceArtifactError("full-FFN condition is invalid")
    for name, value in (
        (
            "evaluation_capture_manifest_sha256",
            evaluation_capture_manifest_sha256,
        ),
        ("routed_checkpoint_sha256", routed_checkpoint_sha256),
        ("routed_score_row_sha256", routed_score_row_sha256),
        ("shared_q_score_row_sha256", shared_q_score_row_sha256),
    ):
        _sha256(name, value)
    if (
        rows.recomposition_sidecar_sha256
        != centering.recomposition_sidecar_sha256
    ):
        raise SharedEvidenceArtifactError(
            "full-FFN rows and centering use different sidecars"
        )
    if rows.shared_q_capture_manifest_sha256 != evaluation_capture_manifest_sha256:
        raise SharedEvidenceArtifactError(
            "full-FFN rows use a different shared-Q evaluation capture"
        )
    scores = full_ffn_token_scores(rows, centering.mean)
    valid = scores.valid_mask
    if not np.count_nonzero(valid):
        raise SharedEvidenceArtifactError(
            "full-FFN score row cannot contain zero scientific tokens"
        )
    arrays = _aggregate_token_metrics(
        scores.sequence_id[valid],
        scores.position[valid],
        {
            "oracle_shared_sse": scores.oracle_shared_sse[valid],
            "learned_shared_sse": scores.learned_shared_sse[valid],
            "full_ffn_sst": scores.common_sst[valid],
        },
    )
    metadata = {
        "schema_version": FULL_FFN_SCORE_ROW_SCHEMA_VERSION,
        "condition": condition,
        "seed": _nonnegative_int("full-FFN seed", seed),
        "process_index": _nonnegative_int(
            "full-FFN process_index",
            process_index,
        ),
        "update_index": _nonnegative_int(
            "full-FFN update_index",
            update_index,
        ),
        "evaluation_capture_manifest_sha256": (
            evaluation_capture_manifest_sha256
        ),
        "routed_checkpoint_sha256": routed_checkpoint_sha256,
        "shared_q_capture_manifest_sha256": (
            rows.shared_q_capture_manifest_sha256
        ),
        "shared_q_centering_contract_sha256": (
            rows.shared_q_centering_contract_sha256
        ),
        "shared_q_checkpoint_sha256": rows.shared_q_checkpoint_sha256,
        "recomposition_sidecar_sha256": rows.recomposition_sidecar_sha256,
        "full_ffn_centering_sha256": centering.reference.sha256,
        "full_ffn_mean_sha256": centering.mean_sha256,
        "routed_score_row_sha256": routed_score_row_sha256,
        "shared_q_score_row_sha256": shared_q_score_row_sha256,
        "tokens": int(arrays["token_count"].sum(dtype=np.int64)),
        "token_identity_sha256": _bitmap_token_identity_sha256(
            arrays["sequence_id"],
            arrays["position_bitmap"],
            arrays["token_count"],
        ),
    }
    blob = _encode_envelope(metadata, arrays)
    return _reference(blob, media_type=FULL_FFN_SCORE_ROW_MEDIA_TYPE), blob


def resolve_full_ffn_score_row(
    reference: SharedEvidenceRef | Mapping[str, Any],
    resolver: SharedEvidenceResolver,
) -> ResolvedFullFFNScoreRows:
    selected_ref = SharedEvidenceRef.from_value(
        reference,
        expected_media_type=FULL_FFN_SCORE_ROW_MEDIA_TYPE,
    )
    metadata, raw = _decode_envelope(_resolve_bytes(selected_ref, resolver))
    selected = _exact_mapping(
        "full-FFN score row metadata",
        metadata,
        {
            "schema_version",
            "condition",
            "seed",
            "process_index",
            "update_index",
            "evaluation_capture_manifest_sha256",
            "routed_checkpoint_sha256",
            "shared_q_capture_manifest_sha256",
            "shared_q_centering_contract_sha256",
            "shared_q_checkpoint_sha256",
            "recomposition_sidecar_sha256",
            "full_ffn_centering_sha256",
            "full_ffn_mean_sha256",
            "routed_score_row_sha256",
            "shared_q_score_row_sha256",
            "tokens",
            "token_identity_sha256",
        },
    )
    if selected["schema_version"] != FULL_FFN_SCORE_ROW_SCHEMA_VERSION:
        raise SharedEvidenceArtifactError("full-FFN score row schema is invalid")
    condition = selected["condition"]
    if condition not in {"A", "R", "B", "F"}:
        raise SharedEvidenceArtifactError("full-FFN row condition is invalid")
    for name in (
        "evaluation_capture_manifest_sha256",
        "routed_checkpoint_sha256",
        "shared_q_capture_manifest_sha256",
        "shared_q_centering_contract_sha256",
        "shared_q_checkpoint_sha256",
        "recomposition_sidecar_sha256",
        "full_ffn_centering_sha256",
        "full_ffn_mean_sha256",
        "routed_score_row_sha256",
        "shared_q_score_row_sha256",
        "token_identity_sha256",
    ):
        _sha256(f"full-FFN row {name}", selected[name])
    arrays = _validate_aggregated_rows(raw, _FULL_METRICS)
    tokens = int(arrays["token_count"].sum(dtype=np.int64))
    if tokens != _positive_int("full-FFN row tokens", selected["tokens"]):
        raise SharedEvidenceArtifactError(
            "full-FFN row token count differs from its payload"
        )
    if (
        _bitmap_token_identity_sha256(
            arrays["sequence_id"],
            arrays["position_bitmap"],
            arrays["token_count"],
        )
        != selected["token_identity_sha256"]
    ):
        raise SharedEvidenceArtifactError(
            "full-FFN row token identity differs from its payload"
        )
    return ResolvedFullFFNScoreRows(
        reference=selected_ref,
        condition=condition,
        seed=_nonnegative_int("full-FFN row seed", selected["seed"]),
        process_index=_nonnegative_int(
            "full-FFN row process_index",
            selected["process_index"],
        ),
        update_index=_nonnegative_int(
            "full-FFN row update_index",
            selected["update_index"],
        ),
        evaluation_capture_manifest_sha256=(
            selected["evaluation_capture_manifest_sha256"]
        ),
        routed_checkpoint_sha256=selected["routed_checkpoint_sha256"],
        shared_q_capture_manifest_sha256=(
            selected["shared_q_capture_manifest_sha256"]
        ),
        shared_q_centering_contract_sha256=(
            selected["shared_q_centering_contract_sha256"]
        ),
        shared_q_checkpoint_sha256=selected["shared_q_checkpoint_sha256"],
        recomposition_sidecar_sha256=(
            selected["recomposition_sidecar_sha256"]
        ),
        full_ffn_centering_sha256=selected["full_ffn_centering_sha256"],
        full_ffn_mean_sha256=selected["full_ffn_mean_sha256"],
        routed_score_row_sha256=selected["routed_score_row_sha256"],
        shared_q_score_row_sha256=selected["shared_q_score_row_sha256"],
        **arrays,
    )


@dataclass(frozen=True, slots=True)
class ResolvedFullFFNScoreManifest:
    reference: SharedEvidenceRef
    phase: str
    protocol_sha256: str
    condition: str
    seed: int
    evaluation_capture_manifest_sha256: str
    routed_checkpoint_sha256: str
    shared_q_checkpoint_sha256: str
    recomposition_sidecar_sha256: str
    full_ffn_centering_reference: SharedEvidenceRef
    full_ffn_mean_sha256: str
    routed_score_manifest_reference: ScoreArtifactRef
    shared_q_score_manifest_reference: SharedEvidenceRef
    row_references: tuple[SharedEvidenceRef, ...]
    sequence_id: np.ndarray
    position_bitmap: np.ndarray
    token_count: np.ndarray
    oracle_shared_sse: np.ndarray
    learned_shared_sse: np.ndarray
    full_ffn_sst: np.ndarray

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


def seal_full_ffn_score_manifest(
    row_references: Sequence[SharedEvidenceRef | Mapping[str, Any]],
    resolver: SharedEvidenceResolver,
    *,
    phase: str,
    protocol_sha256: str,
    condition: str,
    seed: int,
    evaluation_capture_manifest_sha256: str,
    routed_checkpoint_sha256: str,
    routed_score_manifest_reference: ScoreArtifactRef | Mapping[str, Any],
    shared_q_score_manifest: ResolvedSharedQScoreManifest,
    full_ffn_centering: ResolvedFullFFNCentering,
    expected_process_count: int,
    expected_updates: int,
    expected_tokens: int,
) -> tuple[SharedEvidenceRef, bytes]:
    if condition not in {"A", "R", "B", "F"}:
        raise SharedEvidenceArtifactError("full-FFN manifest condition is invalid")
    if not isinstance(phase, str) or not phase:
        raise SharedEvidenceArtifactError("full-FFN manifest phase is invalid")
    for name, value in (
        ("protocol_sha256", protocol_sha256),
        (
            "evaluation_capture_manifest_sha256",
            evaluation_capture_manifest_sha256,
        ),
        ("routed_checkpoint_sha256", routed_checkpoint_sha256),
    ):
        _sha256(name, value)
    selected_seed = _nonnegative_int("full-FFN manifest seed", seed)
    process_count = _positive_int(
        "full-FFN expected_process_count",
        expected_process_count,
    )
    updates = _positive_int("full-FFN expected_updates", expected_updates)
    tokens = _positive_int("full-FFN expected_tokens", expected_tokens)
    if (
        shared_q_score_manifest.phase != phase
        or shared_q_score_manifest.protocol_sha256 != protocol_sha256
        or shared_q_score_manifest.seed != selected_seed
        or shared_q_score_manifest.evaluation_capture_manifest_sha256
        != evaluation_capture_manifest_sha256
        or shared_q_score_manifest.recomposition_sidecar_sha256
        != full_ffn_centering.recomposition_sidecar_sha256
        or shared_q_score_manifest.tokens != tokens
        or shared_q_score_manifest.training_capture_manifest_sha256
        != full_ffn_centering.training_capture_manifest_sha256
        or shared_q_score_manifest.training_tokens
        != full_ffn_centering.token_count
    ):
        raise SharedEvidenceArtifactError(
            "common shared-Q manifest differs from full-FFN identity"
        )
    selected_routed_reference = ScoreArtifactRef.from_value(
        routed_score_manifest_reference,
        expected_media_type=SCORE_MANIFEST_MEDIA_TYPE,
    )
    try:
        routed_manifest = resolve_score_manifest(
            selected_routed_reference,
            resolver,
        )
    except Exception as error:
        raise SharedEvidenceArtifactError(
            "cannot resolve the routed score manifest dependency"
        ) from error
    if (
        routed_manifest.phase != phase
        or routed_manifest.protocol_sha256 != protocol_sha256
        or routed_manifest.condition != condition
        or routed_manifest.seed != selected_seed
        or routed_manifest.evaluation_capture_manifest_sha256
        != evaluation_capture_manifest_sha256
        or routed_manifest.checkpoint_sha256 != routed_checkpoint_sha256
        or routed_manifest.tokens != tokens
        or routed_manifest.training_capture_manifest_sha256
        != full_ffn_centering.training_capture_manifest_sha256
    ):
        raise SharedEvidenceArtifactError(
            "routed score manifest differs from full-FFN identity"
        )
    references = tuple(
        SharedEvidenceRef.from_value(
            item,
            expected_media_type=FULL_FFN_SCORE_ROW_MEDIA_TYPE,
        )
        for item in row_references
    )
    rows = tuple(resolve_full_ffn_score_row(item, resolver) for item in references)
    if len({item.sha256 for item in references}) != len(references):
        raise SharedEvidenceArtifactError(
            "full-FFN manifest contains duplicate row artifacts"
        )
    q_row_by_key = {}
    for q_ref in shared_q_score_manifest.row_references:
        q_row = resolve_shared_q_score_row(q_ref, resolver)
        q_row_by_key[(q_row.process_index, q_row.update_index)] = q_row
    routed_row_by_key = {}
    for routed_ref in routed_manifest.row_references:
        try:
            routed_row = resolve_score_row_shard(routed_ref, resolver)
        except Exception as error:
            raise SharedEvidenceArtifactError(
                "cannot resolve a routed score row dependency"
            ) from error
        routed_row_by_key[
            (routed_row.process_index, routed_row.update_index)
        ] = routed_row
    seen: set[tuple[int, int]] = set()
    for row in rows:
        key = (row.process_index, row.update_index)
        q_row = q_row_by_key.get(key)
        routed_row = routed_row_by_key.get(key)
        if key in seen:
            raise SharedEvidenceArtifactError(
                "full-FFN rows duplicate a process/update"
            )
        seen.add(key)
        if (
            q_row is None
            or routed_row is None
            or row.condition != condition
            or row.seed != selected_seed
            or row.evaluation_capture_manifest_sha256
            != evaluation_capture_manifest_sha256
            or row.routed_checkpoint_sha256 != routed_checkpoint_sha256
            or row.shared_q_capture_manifest_sha256
            != shared_q_score_manifest.evaluation_capture_manifest_sha256
            or row.shared_q_centering_contract_sha256
            != shared_q_score_manifest.centering_contract_sha256
            or row.shared_q_checkpoint_sha256
            != shared_q_score_manifest.checkpoint_sha256
            or row.recomposition_sidecar_sha256
            != full_ffn_centering.recomposition_sidecar_sha256
            or row.full_ffn_centering_sha256
            != full_ffn_centering.reference.sha256
            or row.full_ffn_mean_sha256 != full_ffn_centering.mean_sha256
            or row.shared_q_score_row_sha256 != q_row.reference.sha256
            or row.routed_score_row_sha256 != routed_row.reference.sha256
            or row.token_identity_sha256 != q_row.token_identity_sha256
            or row.token_identity_sha256 != routed_row.token_identity_sha256
        ):
            raise SharedEvidenceArtifactError(
                "full-FFN row differs from Q/centering/condition identity"
            )
    expected = {
        (process, update)
        for process in range(process_count)
        for update in range(updates)
    }
    if (
        seen != expected
        or set(q_row_by_key) != expected
        or set(routed_row_by_key) != expected
    ):
        raise SharedEvidenceArtifactError(
            "full-FFN/Q process-update coverage is incomplete"
        )
    merged = _merge_aggregated_rows(rows, _FULL_METRICS)
    if int(merged["token_count"].sum(dtype=np.int64)) != tokens:
        raise SharedEvidenceArtifactError(
            "full-FFN rows do not cover the exact evaluation capture"
        )
    if (
        _bitmap_token_identity_sha256(
            merged["sequence_id"],
            merged["position_bitmap"],
            merged["token_count"],
        )
        != shared_q_score_manifest.token_identity_sha256
    ):
        raise SharedEvidenceArtifactError(
            "full-FFN and shared-Q token inventories differ"
        )
    unsigned = {
        "schema_version": FULL_FFN_SCORE_MANIFEST_SCHEMA_VERSION,
        "phase": phase,
        "protocol_sha256": protocol_sha256,
        "condition": condition,
        "seed": selected_seed,
        "evaluation_capture_manifest_sha256": (
            evaluation_capture_manifest_sha256
        ),
        "routed_checkpoint_sha256": routed_checkpoint_sha256,
        "shared_q_checkpoint_sha256": (
            shared_q_score_manifest.checkpoint_sha256
        ),
        "recomposition_sidecar_sha256": (
            full_ffn_centering.recomposition_sidecar_sha256
        ),
        "full_ffn_centering_artifact": (
            full_ffn_centering.reference.to_dict()
        ),
        "full_ffn_mean_sha256": full_ffn_centering.mean_sha256,
        "routed_score_manifest_artifact": (
            selected_routed_reference.to_dict()
        ),
        "shared_q_score_manifest_artifact": (
            shared_q_score_manifest.reference.to_dict()
        ),
        "expected_process_count": process_count,
        "expected_updates": updates,
        "row_artifacts": [item.to_dict() for item in references],
        "tokens": tokens,
        "token_identity_sha256": _bitmap_token_identity_sha256(
            merged["sequence_id"],
            merged["position_bitmap"],
            merged["token_count"],
        ),
        "oracle_shared_sse": float(
            merged["oracle_shared_sse"].sum(dtype=np.float64)
        ),
        "learned_shared_sse": float(
            merged["learned_shared_sse"].sum(dtype=np.float64)
        ),
        "full_ffn_sst": float(
            merged["full_ffn_sst"].sum(dtype=np.float64)
        ),
    }
    if unsigned["full_ffn_sst"] <= 0:
        raise SharedEvidenceArtifactError("full-FFN manifest SST is non-positive")
    blob = canonical_json_bytes(unsigned)
    return _reference(blob, media_type=FULL_FFN_SCORE_MANIFEST_MEDIA_TYPE), blob


def resolve_full_ffn_score_manifest(
    reference: SharedEvidenceRef | Mapping[str, Any],
    resolver: SharedEvidenceResolver,
) -> ResolvedFullFFNScoreManifest:
    """Resolve a full-FFN manifest and re-prove every upstream dependency."""

    selected_ref = SharedEvidenceRef.from_value(
        reference,
        expected_media_type=FULL_FFN_SCORE_MANIFEST_MEDIA_TYPE,
    )
    blob = _resolve_bytes(selected_ref, resolver)
    try:
        raw = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SharedEvidenceArtifactError(
            "full-FFN score manifest is not JSON"
        ) from error
    if canonical_json_bytes(raw) != blob:
        raise SharedEvidenceArtifactError(
            "full-FFN score manifest is not canonical"
        )
    selected = _exact_mapping(
        "full-FFN score manifest",
        raw,
        {
            "schema_version",
            "phase",
            "protocol_sha256",
            "condition",
            "seed",
            "evaluation_capture_manifest_sha256",
            "routed_checkpoint_sha256",
            "shared_q_checkpoint_sha256",
            "recomposition_sidecar_sha256",
            "full_ffn_centering_artifact",
            "full_ffn_mean_sha256",
            "routed_score_manifest_artifact",
            "shared_q_score_manifest_artifact",
            "expected_process_count",
            "expected_updates",
            "row_artifacts",
            "tokens",
            "token_identity_sha256",
            "oracle_shared_sse",
            "learned_shared_sse",
            "full_ffn_sst",
        },
    )
    if selected["schema_version"] != FULL_FFN_SCORE_MANIFEST_SCHEMA_VERSION:
        raise SharedEvidenceArtifactError(
            "full-FFN score manifest schema is invalid"
        )
    phase = selected["phase"]
    condition = selected["condition"]
    if not isinstance(phase, str) or not phase:
        raise SharedEvidenceArtifactError("full-FFN manifest phase is invalid")
    if condition not in {"A", "R", "B", "F"}:
        raise SharedEvidenceArtifactError(
            "full-FFN manifest condition is invalid"
        )
    for name in (
        "protocol_sha256",
        "evaluation_capture_manifest_sha256",
        "routed_checkpoint_sha256",
        "shared_q_checkpoint_sha256",
        "recomposition_sidecar_sha256",
        "full_ffn_mean_sha256",
        "token_identity_sha256",
    ):
        _sha256(f"full-FFN manifest {name}", selected[name])
    seed = _nonnegative_int("full-FFN manifest seed", selected["seed"])
    process_count = _positive_int(
        "full-FFN manifest process_count",
        selected["expected_process_count"],
    )
    updates = _positive_int(
        "full-FFN manifest updates",
        selected["expected_updates"],
    )
    tokens = _positive_int("full-FFN manifest tokens", selected["tokens"])
    centering_ref = SharedEvidenceRef.from_value(
        selected["full_ffn_centering_artifact"],
        expected_media_type=FULL_FFN_CENTERING_MEDIA_TYPE,
    )
    q_manifest_ref = SharedEvidenceRef.from_value(
        selected["shared_q_score_manifest_artifact"],
        expected_media_type=SHARED_Q_SCORE_MANIFEST_MEDIA_TYPE,
    )
    routed_manifest_ref = ScoreArtifactRef.from_value(
        selected["routed_score_manifest_artifact"],
        expected_media_type=SCORE_MANIFEST_MEDIA_TYPE,
    )
    centering = resolve_full_ffn_centering(centering_ref, resolver)
    q_manifest = resolve_shared_q_score_manifest(q_manifest_ref, resolver)
    if not isinstance(selected["row_artifacts"], list):
        raise SharedEvidenceArtifactError(
            "full-FFN manifest row_artifacts must be a list"
        )
    references = tuple(
        SharedEvidenceRef.from_value(
            item,
            expected_media_type=FULL_FFN_SCORE_ROW_MEDIA_TYPE,
        )
        for item in selected["row_artifacts"]
    )
    canonical_ref, canonical_blob = seal_full_ffn_score_manifest(
        references,
        resolver,
        phase=phase,
        protocol_sha256=selected["protocol_sha256"],
        condition=condition,
        seed=seed,
        evaluation_capture_manifest_sha256=(
            selected["evaluation_capture_manifest_sha256"]
        ),
        routed_checkpoint_sha256=selected["routed_checkpoint_sha256"],
        routed_score_manifest_reference=routed_manifest_ref,
        shared_q_score_manifest=q_manifest,
        full_ffn_centering=centering,
        expected_process_count=process_count,
        expected_updates=updates,
        expected_tokens=tokens,
    )
    if canonical_blob != blob or canonical_ref != selected_ref:
        raise SharedEvidenceArtifactError(
            "full-FFN manifest differs from recomputed dependency evidence"
        )
    rows = tuple(resolve_full_ffn_score_row(item, resolver) for item in references)
    merged = _merge_aggregated_rows(rows, _FULL_METRICS)
    return ResolvedFullFFNScoreManifest(
        reference=selected_ref,
        phase=phase,
        protocol_sha256=selected["protocol_sha256"],
        condition=condition,
        seed=seed,
        evaluation_capture_manifest_sha256=(
            selected["evaluation_capture_manifest_sha256"]
        ),
        routed_checkpoint_sha256=selected["routed_checkpoint_sha256"],
        shared_q_checkpoint_sha256=selected["shared_q_checkpoint_sha256"],
        recomposition_sidecar_sha256=(
            selected["recomposition_sidecar_sha256"]
        ),
        full_ffn_centering_reference=centering_ref,
        full_ffn_mean_sha256=selected["full_ffn_mean_sha256"],
        routed_score_manifest_reference=routed_manifest_ref,
        shared_q_score_manifest_reference=q_manifest_ref,
        row_references=references,
        **merged,
    )


class SharedQScoreRowSink:
    """Callable evaluator sink with immutable replay and read-after-write."""

    def __init__(
        self,
        *,
        seed: int,
        process_index: int,
        evaluation_input_receipt: SharedQInputReceipt | Mapping[str, Any],
        retain: SharedEvidenceRetainer,
        resolve: SharedEvidenceResolver,
        topk: int = 64,
    ) -> None:
        self.seed = _nonnegative_int("shared-Q sink seed", seed)
        self.process_index = _nonnegative_int(
            "shared-Q sink process_index",
            process_index,
        )
        self.receipt = _validate_q_input_receipt(evaluation_input_receipt)
        self.topk = _positive_int("shared-Q sink topk", topk)
        if (
            self.receipt.role != "evaluation"
            or self.receipt.process_index != self.process_index
        ):
            raise SharedEvidenceArtifactError(
                "shared-Q sink requires its matching evaluation receipt"
            )
        self.retain = retain
        self.resolve = resolve
        self.references: dict[int, SharedEvidenceRef] = {}

    def __call__(self, evidence: SharedQEvaluationEvidenceChunk) -> None:
        reference, blob = seal_shared_q_score_row(
            evidence,
            seed=self.seed,
            process_index=self.process_index,
            evaluation_input_receipt=self.receipt,
            topk=self.topk,
        )
        previous = self.references.get(evidence.update_index)
        if previous is not None and previous != reference:
            raise SharedEvidenceArtifactError(
                "shared-Q sink replay changed immutable evidence"
            )
        self.retain(reference, blob)
        resolved = resolve_shared_q_score_row(reference, self.resolve)
        if (
            resolved.process_index != self.process_index
            or resolved.update_index != evidence.update_index
        ):
            raise SharedEvidenceArtifactError(
                "shared-Q retained row failed read-after-write identity"
            )
        self.references[evidence.update_index] = reference

    @property
    def row_references(self) -> tuple[SharedEvidenceRef, ...]:
        return tuple(self.references[index] for index in sorted(self.references))


class FullFFNScoreRowSink:
    """Durably retain full-FFN rows paired to routed and common-Q row hashes."""

    def __init__(
        self,
        *,
        condition: str,
        seed: int,
        process_index: int,
        evaluation_capture_manifest_sha256: str,
        routed_checkpoint_sha256: str,
        centering: ResolvedFullFFNCentering,
        dependency_rows: Mapping[int, tuple[str, str]],
        retain: SharedEvidenceRetainer,
        resolve: SharedEvidenceResolver,
    ) -> None:
        if condition not in {"A", "R", "B", "F"}:
            raise SharedEvidenceArtifactError("full-FFN sink condition is invalid")
        self.condition = condition
        self.seed = _nonnegative_int("full-FFN sink seed", seed)
        self.process_index = _nonnegative_int(
            "full-FFN sink process_index",
            process_index,
        )
        self.evaluation_capture_manifest_sha256 = _sha256(
            "full-FFN sink evaluation capture",
            evaluation_capture_manifest_sha256,
        )
        self.routed_checkpoint_sha256 = _sha256(
            "full-FFN sink routed checkpoint",
            routed_checkpoint_sha256,
        )
        self.centering = centering
        self.dependencies = {
            _nonnegative_int("full-FFN dependency update", update): (
                _sha256("routed score row SHA-256", pair[0]),
                _sha256("shared-Q score row SHA-256", pair[1]),
            )
            for update, pair in dependency_rows.items()
        }
        self.retain = retain
        self.resolve = resolve
        self.references: dict[int, SharedEvidenceRef] = {}

    def retain_rows(
        self,
        update_index: int,
        rows: FullFFNRecompositionRows,
    ) -> SharedEvidenceRef:
        selected_update = _nonnegative_int(
            "full-FFN sink update_index",
            update_index,
        )
        try:
            routed_row, q_row = self.dependencies[selected_update]
        except KeyError as error:
            raise SharedEvidenceArtifactError(
                "full-FFN sink lacks dependency rows for this update"
            ) from error
        reference, blob = seal_full_ffn_score_row(
            rows,
            self.centering,
            condition=self.condition,
            seed=self.seed,
            process_index=self.process_index,
            update_index=selected_update,
            evaluation_capture_manifest_sha256=(
                self.evaluation_capture_manifest_sha256
            ),
            routed_checkpoint_sha256=self.routed_checkpoint_sha256,
            routed_score_row_sha256=routed_row,
            shared_q_score_row_sha256=q_row,
        )
        previous = self.references.get(selected_update)
        if previous is not None and previous != reference:
            raise SharedEvidenceArtifactError(
                "full-FFN sink replay changed immutable evidence"
            )
        self.retain(reference, blob)
        resolved = resolve_full_ffn_score_row(reference, self.resolve)
        if (
            resolved.process_index != self.process_index
            or resolved.update_index != selected_update
        ):
            raise SharedEvidenceArtifactError(
                "full-FFN retained row failed read-after-write identity"
            )
        self.references[selected_update] = reference
        return reference

    @property
    def row_references(self) -> tuple[SharedEvidenceRef, ...]:
        return tuple(self.references[index] for index in sorted(self.references))


__all__ = [
    "FULL_FFN_CENTERING_MEDIA_TYPE",
    "FULL_FFN_CENTERING_SCHEMA_VERSION",
    "FULL_FFN_SCORE_MANIFEST_MEDIA_TYPE",
    "FULL_FFN_SCORE_MANIFEST_SCHEMA_VERSION",
    "FULL_FFN_SCORE_ROW_MEDIA_TYPE",
    "FULL_FFN_SCORE_ROW_SCHEMA_VERSION",
    "SHARED_Q_SCORE_MANIFEST_MEDIA_TYPE",
    "SHARED_Q_SCORE_MANIFEST_SCHEMA_VERSION",
    "SHARED_Q_SCORE_ROW_MEDIA_TYPE",
    "SHARED_Q_SCORE_ROW_SCHEMA_VERSION",
    "FullFFNCenteringAccumulator",
    "FullFFNCenteringSufficient",
    "FullFFNScoreRowSink",
    "ResolvedFullFFNCentering",
    "ResolvedFullFFNScoreManifest",
    "ResolvedFullFFNScoreRows",
    "ResolvedSharedQScoreManifest",
    "ResolvedSharedQScoreRows",
    "SharedEvidenceArtifactError",
    "SharedEvidenceRef",
    "SharedEvidenceResolver",
    "SharedEvidenceRetainer",
    "SharedQScoreRowSink",
    "resolve_full_ffn_centering",
    "resolve_full_ffn_score_manifest",
    "resolve_full_ffn_score_row",
    "resolve_shared_q_score_manifest",
    "resolve_shared_q_score_row",
    "seal_full_ffn_centering",
    "seal_full_ffn_centering_from_dataset",
    "seal_full_ffn_score_manifest",
    "seal_full_ffn_score_row",
    "seal_shared_q_score_manifest",
    "seal_shared_q_score_row",
]
