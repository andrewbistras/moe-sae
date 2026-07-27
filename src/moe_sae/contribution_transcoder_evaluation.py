"""Exact host scoring and paired sequence evidence for Experiment 3.

The accelerator evaluator only needs to return the decoded expert predictions
and their sparse-selection diagnostics.  This module owns the scientific
boundary after decode:

* the unchanged router-weighted component objective;
* the exact Gemma routed RMSNorm round points;
* the exact oracle-shared addition and outer FFN RMSNorm round points; and
* one immutable sufficient row per packed sequence.

The source overlay is deliberately absent from the score arithmetic.  Its
manifest identity is nevertheless bound into every retained score artifact,
which proves that a source-only transcoder produced the predictions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    NUM_EXPERTS,
    SEQUENCE_LENGTH,
    TOP_K_EXPERTS,
    bf16_bits_to_float32,
    canonical_sha256,
    float32_to_bf16_bits,
    token_identity_sha256,
)
from moe_sae.contribution_recomposition import (
    decode_recomposition_sidecar,
    maxtext_rmsnorm_bf16,
)
from moe_sae.contribution_distributed import (
    _decode_selected_columns_default_precision,
    _local_grouped_candidates,
    _local_rank_candidates,
    _merge_local_candidates,
    _scientific_input_valid,
    _validate_contribution_mesh,
)
from moe_sae.contribution_evaluation import (
    ContributionEvaluationAux,
    ContributionEvaluationChunk,
    ContributionEvaluationRows,
    _addressable_feature_mask,
    _host_chunk,
)
from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_transcoder import (
    ContributionTranscoderParams,
    transcoder_params_from_sae,
    transcoder_params_to_sae,
)
from moe_sae.contribution_transcoder_distributed import (
    ContributionTranscoderDistributedBatch,
    _validate_stacked_batch,
    contribution_transcoder_batch_shardings,
    shard_contribution_transcoder_batch,
    shard_process_local_contribution_transcoder_batch,
    stack_contribution_transcoder_microbatches,
)
from moe_sae.distributed import (
    DistributedConfig,
    _param_specs,
    replicate_process_local_array,
)

CONDITIONS = ("A", "R", "B", "F")
ENDPOINTS = ("component", "routed", "oracle_shared")
SEQUENCE_SCORE_SCHEMA_VERSION = (
    "gemma4-contribution-transcoder-sequence-scores-v1"
)
TOKEN_SCORE_SCHEMA_VERSION = "gemma4-contribution-transcoder-token-scores-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POSITION_WORDS = SEQUENCE_LENGTH // 64


class ContributionTranscoderEvaluationError(RuntimeError):
    """Experiment-3 evaluation evidence violates its frozen contract."""


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContributionTranscoderEvaluationError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite_vector(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    *,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != shape
        or not np.issubdtype(array.dtype, np.floating)
        or not np.isfinite(array).all()
        or (nonnegative and np.any(array < 0))
    ):
        qualifier = "finite nonnegative " if nonnegative else "finite "
        raise ContributionTranscoderEvaluationError(
            f"{name} must be {qualifier}floating-point {shape}"
        )
    result = np.ascontiguousarray(array, dtype=np.float64)
    result.setflags(write=False)
    return result


def _readonly(value: Any, dtype: Any | None = None) -> np.ndarray:
    selected = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    selected.setflags(write=False)
    return selected


def _bf16_add(left_bits: np.ndarray, right_bits: np.ndarray) -> np.ndarray:
    if (
        left_bits.dtype != np.uint16
        or right_bits.dtype != np.uint16
        or left_bits.shape != right_bits.shape
    ):
        raise ContributionTranscoderEvaluationError(
            "oracle-shared BF16 addition operands are incompatible"
        )
    value = np.add(
        bf16_bits_to_float32(left_bits),
        bf16_bits_to_float32(right_bits),
        dtype=np.float32,
    )
    return float32_to_bf16_bits(value)


@dataclasses.dataclass(frozen=True, slots=True)
class ContributionTranscoderTokenScores:
    """Per-token sufficient scores for one condition.

    Route rows are retained for expert-balanced and router-rank diagnostics.
    All padding rows are exactly zero so flattening a static accelerator batch
    cannot turn padding into scientific evidence.
    """

    component_sse: np.ndarray
    component_sst: np.ndarray
    routed_sse: np.ndarray
    routed_sst: np.ndarray
    oracle_shared_sse: np.ndarray
    oracle_shared_sst: np.ndarray
    route_component_sse: np.ndarray
    route_component_sst: np.ndarray
    valid_mask: np.ndarray
    recomposition_sidecar_sha256: str

    def __post_init__(self) -> None:
        valid = np.asarray(self.valid_mask)
        if valid.ndim != 1 or valid.dtype != np.bool_ or not np.any(valid):
            raise ContributionTranscoderEvaluationError(
                "valid_mask must be nonempty bool [tokens]"
            )
        token_shape = valid.shape
        for name in (
            "component_sse",
            "component_sst",
            "routed_sse",
            "routed_sst",
            "oracle_shared_sse",
            "oracle_shared_sst",
        ):
            array = _finite_vector(
                name,
                getattr(self, name),
                token_shape,
                nonnegative=True,
            )
            if np.any(array[~valid] != 0):
                raise ContributionTranscoderEvaluationError(
                    f"{name} must be exactly zero on padding"
                )
            object.__setattr__(self, name, array)
        route_shape = (*token_shape, TOP_K_EXPERTS)
        for name in ("route_component_sse", "route_component_sst"):
            array = _finite_vector(
                name,
                getattr(self, name),
                route_shape,
                nonnegative=True,
            )
            if np.any(array[~valid] != 0):
                raise ContributionTranscoderEvaluationError(
                    f"{name} must be exactly zero on padding"
                )
            object.__setattr__(self, name, array)
        for name in ("component_sst", "routed_sst", "oracle_shared_sst"):
            if float(np.sum(getattr(self, name)[valid], dtype=np.float64)) <= 0:
                raise ContributionTranscoderEvaluationError(
                    f"{name} must be positive on scientific rows"
                )
        object.__setattr__(self, "valid_mask", _readonly(valid, np.bool_))
        _sha256(
            "recomposition_sidecar_sha256",
            self.recomposition_sidecar_sha256,
        )

    @property
    def tokens(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    def endpoint(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name not in ENDPOINTS:
            raise ContributionTranscoderEvaluationError(
                f"unknown transcoder endpoint {name!r}"
            )
        return (
            getattr(self, f"{name}_sse"),
            getattr(self, f"{name}_sst"),
        )


def _recomposed_token_scores(
    *,
    raw_routed_target: np.ndarray,
    raw_routed_reconstruction: np.ndarray,
    shared_output_bf16_bits: np.ndarray,
    routed_mean: np.ndarray,
    oracle_shared_mean: np.ndarray,
    valid_mask: np.ndarray,
    recomposition_sidecar: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Score raw routed aggregates through the two frozen Gemma norms."""

    decoded_sidecar = decode_recomposition_sidecar(recomposition_sidecar)
    _, routed_target_bits = maxtext_rmsnorm_bf16(
        float32_to_bf16_bits(raw_routed_target),
        decoded_sidecar.routed_rmsnorm_scale_bf16_bits,
        epsilon=float(decoded_sidecar.epsilon),
    )
    _, routed_estimate_bits = maxtext_rmsnorm_bf16(
        float32_to_bf16_bits(raw_routed_reconstruction),
        decoded_sidecar.routed_rmsnorm_scale_bf16_bits,
        epsilon=float(decoded_sidecar.epsilon),
    )
    routed_target = bf16_bits_to_float32(routed_target_bits)
    routed_estimate = bf16_bits_to_float32(routed_estimate_bits)
    routed_sse = np.sum(
        np.square(routed_target - routed_estimate, dtype=np.float32),
        axis=1,
        dtype=np.float64,
    )
    routed_sst = np.sum(
        np.square(
            routed_target - routed_mean[None, :],
            dtype=np.float32,
        ),
        axis=1,
        dtype=np.float64,
    )

    _, shared_post_bits = maxtext_rmsnorm_bf16(
        shared_output_bf16_bits,
        decoded_sidecar.shared_rmsnorm_scale_bf16_bits,
        epsilon=float(decoded_sidecar.epsilon),
    )
    target_sum_bits = _bf16_add(routed_target_bits, shared_post_bits)
    estimate_sum_bits = _bf16_add(routed_estimate_bits, shared_post_bits)
    _, oracle_target_bits = maxtext_rmsnorm_bf16(
        target_sum_bits,
        decoded_sidecar.outer_ffn_rmsnorm_scale_bf16_bits,
        epsilon=float(decoded_sidecar.epsilon),
    )
    _, oracle_estimate_bits = maxtext_rmsnorm_bf16(
        estimate_sum_bits,
        decoded_sidecar.outer_ffn_rmsnorm_scale_bf16_bits,
        epsilon=float(decoded_sidecar.epsilon),
    )
    oracle_target = bf16_bits_to_float32(oracle_target_bits)
    oracle_estimate = bf16_bits_to_float32(oracle_estimate_bits)
    oracle_sse = np.sum(
        np.square(oracle_target - oracle_estimate, dtype=np.float32),
        axis=1,
        dtype=np.float64,
    )
    oracle_sst = np.sum(
        np.square(
            oracle_target - oracle_shared_mean[None, :],
            dtype=np.float32,
        ),
        axis=1,
        dtype=np.float64,
    )
    return (
        np.where(valid_mask, routed_sse, 0.0),
        np.where(valid_mask, routed_sst, 0.0),
        np.where(valid_mask, oracle_sse, 0.0),
        np.where(valid_mask, oracle_sst, 0.0),
        decoded_sidecar.sidecar_sha256,
    )


def score_transcoder_predictions(
    *,
    expert_output: Any,
    reconstruction: Any,
    router_weights: Any,
    shared_output_bf16_bits: Any,
    target_mean: Any,
    routed_mean: Any,
    oracle_shared_mean: Any,
    valid_mask: Any,
    recomposition_sidecar: Mapping[str, Any],
) -> ContributionTranscoderTokenScores:
    """Score source-only expert predictions through all frozen endpoints.

    ``expert_output`` and ``router_weights`` are the decoded BF16 values from
    the sealed v2 target.  ``reconstruction`` is the transcoder prediction.
    The shared branch is always the exact sealed branch, never a learned or
    condition-specific approximation.
    """

    target = np.asarray(expert_output)
    estimate = np.asarray(reconstruction)
    weights = np.asarray(router_weights)
    shared_bits = np.asarray(shared_output_bf16_bits)
    valid = np.asarray(valid_mask)
    if (
        target.ndim != 3
        or target.shape[1:] != (TOP_K_EXPERTS, HIDDEN_SIZE)
        or estimate.shape != target.shape
    ):
        raise ContributionTranscoderEvaluationError(
            "expert target and reconstruction must be [tokens,8,2816]"
        )
    tokens = target.shape[0]
    if (
        weights.shape != (tokens, TOP_K_EXPERTS)
        or shared_bits.dtype != np.uint16
        or shared_bits.shape != (tokens, HIDDEN_SIZE)
        or valid.dtype != np.bool_
        or valid.shape != (tokens,)
        or not np.any(valid)
    ):
        raise ContributionTranscoderEvaluationError(
            "router/shared/mask arrays do not align with transcoder predictions"
        )
    for name, value in (
        ("expert_output", target),
        ("reconstruction", estimate),
        ("router_weights", weights),
    ):
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(
            value
        ).all():
            raise ContributionTranscoderEvaluationError(
                f"{name} must be finite floating-point"
            )
    if np.any(weights < 0) or not np.all(
        np.abs(weights[valid].sum(axis=1, dtype=np.float32) - 1.0) <= 0.005
    ):
        raise ContributionTranscoderEvaluationError(
            "scientific router weights must be nonnegative and sum to one"
        )
    if not np.isfinite(bf16_bits_to_float32(shared_bits)).all():
        raise ContributionTranscoderEvaluationError(
            "shared_output contains non-finite BF16 values"
        )
    means: dict[str, np.ndarray] = {}
    for name, value in (
        ("target_mean", target_mean),
        ("routed_mean", routed_mean),
        ("oracle_shared_mean", oracle_shared_mean),
    ):
        selected = np.asarray(value)
        if (
            selected.shape != (HIDDEN_SIZE,)
            or not np.issubdtype(selected.dtype, np.floating)
            or not np.isfinite(selected).all()
        ):
            raise ContributionTranscoderEvaluationError(
                f"{name} must be finite floating-point [{HIDDEN_SIZE}]"
            )
        means[name] = selected.astype(np.float32)

    target32 = target.astype(np.float32)
    estimate32 = estimate.astype(np.float32)
    weights32 = weights.astype(np.float32)
    weighted_error = weights32[..., None] * (target32 - estimate32)
    weighted_centered = weights32[..., None] * (
        target32 - means["target_mean"][None, None, :]
    )
    route_sse = np.sum(
        np.square(weighted_error, dtype=np.float32),
        axis=2,
        dtype=np.float64,
    )
    route_sst = np.sum(
        np.square(weighted_centered, dtype=np.float32),
        axis=2,
        dtype=np.float64,
    )
    route_sse = np.where(valid[:, None], route_sse, 0.0)
    route_sst = np.where(valid[:, None], route_sst, 0.0)
    component_sse = route_sse.sum(axis=1, dtype=np.float64)
    component_sst = route_sst.sum(axis=1, dtype=np.float64)

    raw_target = np.einsum(
        "nkh,nk->nh",
        target32,
        weights32,
        dtype=np.float32,
        optimize=False,
    )
    raw_estimate = np.einsum(
        "nkh,nk->nh",
        estimate32,
        weights32,
        dtype=np.float32,
        optimize=False,
    )
    routed_sse, routed_sst, oracle_sse, oracle_sst, sidecar_sha = (
        _recomposed_token_scores(
            raw_routed_target=raw_target,
            raw_routed_reconstruction=raw_estimate,
            shared_output_bf16_bits=shared_bits,
            routed_mean=means["routed_mean"],
            oracle_shared_mean=means["oracle_shared_mean"],
            valid_mask=valid,
            recomposition_sidecar=recomposition_sidecar,
        )
    )
    return ContributionTranscoderTokenScores(
        component_sse=component_sse,
        component_sst=component_sst,
        routed_sse=routed_sse,
        routed_sst=routed_sst,
        oracle_shared_sse=oracle_sse,
        oracle_shared_sst=oracle_sst,
        route_component_sse=route_sse,
        route_component_sst=route_sst,
        valid_mask=valid,
        recomposition_sidecar_sha256=sidecar_sha,
    )


def score_transcoder_sufficient_rows(
    *,
    component_sse: Any,
    component_sst: Any,
    route_component_sse: Any,
    route_component_sst: Any,
    raw_routed_target: Any,
    raw_routed_reconstruction: Any,
    shared_output_bf16_bits: Any,
    routed_mean: Any,
    oracle_shared_mean: Any,
    valid_mask: Any,
    recomposition_sidecar: Mapping[str, Any],
) -> ContributionTranscoderTokenScores:
    """Finish exact scoring from feature-sharded accelerator sufficient rows."""

    valid = np.asarray(valid_mask)
    raw_target = np.asarray(raw_routed_target)
    raw_estimate = np.asarray(raw_routed_reconstruction)
    shared_bits = np.asarray(shared_output_bf16_bits)
    if (
        valid.ndim != 1
        or valid.dtype != np.bool_
        or not np.any(valid)
        or raw_target.shape != (valid.size, HIDDEN_SIZE)
        or raw_estimate.shape != raw_target.shape
        or shared_bits.dtype != np.uint16
        or shared_bits.shape != raw_target.shape
    ):
        raise ContributionTranscoderEvaluationError(
            "distributed sufficient rows have incompatible shapes or dtypes"
        )
    for name, value in (
        ("raw_routed_target", raw_target),
        ("raw_routed_reconstruction", raw_estimate),
    ):
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(
            value
        ).all():
            raise ContributionTranscoderEvaluationError(
                f"{name} must be finite floating-point"
            )
    means = {}
    for name, value in (
        ("routed_mean", routed_mean),
        ("oracle_shared_mean", oracle_shared_mean),
    ):
        selected = np.asarray(value)
        if (
            selected.shape != (HIDDEN_SIZE,)
            or not np.issubdtype(selected.dtype, np.floating)
            or not np.isfinite(selected).all()
        ):
            raise ContributionTranscoderEvaluationError(
                f"{name} must be finite floating-point [{HIDDEN_SIZE}]"
            )
        means[name] = selected.astype(np.float32)
    routed_sse, routed_sst, oracle_sse, oracle_sst, sidecar_sha = (
        _recomposed_token_scores(
            raw_routed_target=raw_target.astype(np.float32),
            raw_routed_reconstruction=raw_estimate.astype(np.float32),
            shared_output_bf16_bits=shared_bits,
            routed_mean=means["routed_mean"],
            oracle_shared_mean=means["oracle_shared_mean"],
            valid_mask=valid,
            recomposition_sidecar=recomposition_sidecar,
        )
    )
    return ContributionTranscoderTokenScores(
        component_sse=component_sse,
        component_sst=component_sst,
        routed_sse=routed_sse,
        routed_sst=routed_sst,
        oracle_shared_sse=oracle_sse,
        oracle_shared_sst=oracle_sst,
        route_component_sse=route_component_sse,
        route_component_sst=route_component_sst,
        valid_mask=valid,
        recomposition_sidecar_sha256=sidecar_sha,
    )


def _aggregate_rows(
    sequence_id: np.ndarray,
    position: np.ndarray,
    values: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    order = np.lexsort((position.astype(np.int64), sequence_id))
    sequence = np.asarray(sequence_id[order], dtype=np.int64)
    positions = np.asarray(position[order], dtype=np.uint16)
    records = np.rec.fromarrays(
        (sequence, positions),
        names=("sequence_id", "position"),
    )
    if np.unique(records).size != records.size:
        raise ContributionTranscoderEvaluationError(
            "transcoder score input contains duplicate token keys"
        )
    unique, inverse = np.unique(sequence, return_inverse=True)
    bitmap = np.zeros((unique.size, _POSITION_WORDS), dtype=np.uint64)
    words = positions.astype(np.int64) // 64
    bits = np.left_shift(
        np.uint64(1),
        (positions.astype(np.int64) % 64).astype(np.uint64),
    )
    np.bitwise_or.at(bitmap, (inverse, words), bits)
    count = np.bincount(inverse, minlength=unique.size).astype(np.uint16)
    aggregated = {
        name: np.bincount(
            inverse,
            weights=np.asarray(value, dtype=np.float64)[order],
            minlength=unique.size,
        ).astype(np.float64)
        for name, value in values.items()
    }
    return unique.astype(np.int64), bitmap, count, aggregated


@dataclasses.dataclass(frozen=True, slots=True)
class TranscoderPairedSequenceScores:
    """One immutable paired score matrix over packed sequences."""

    source_manifest_sha256: str
    target_manifest_sha256: str
    recomposition_sidecar_sha256: str
    token_identity_sha256: str
    sequence_id: np.ndarray
    position_bitmap: np.ndarray
    token_count: np.ndarray
    common_sst: Mapping[str, np.ndarray]
    condition_sse: Mapping[str, Mapping[str, np.ndarray]]

    def __post_init__(self) -> None:
        for name in (
            "source_manifest_sha256",
            "target_manifest_sha256",
            "recomposition_sidecar_sha256",
            "token_identity_sha256",
        ):
            _sha256(name, getattr(self, name))
        sequence = np.asarray(self.sequence_id)
        bitmap = np.asarray(self.position_bitmap)
        count = np.asarray(self.token_count)
        if (
            sequence.dtype != np.int64
            or sequence.ndim != 1
            or not sequence.size
            or np.any(sequence < 0)
            or not np.all(sequence[1:] > sequence[:-1])
            or bitmap.dtype != np.uint64
            or bitmap.shape != (sequence.size, _POSITION_WORDS)
            or count.dtype != np.uint16
            or count.shape != sequence.shape
            or np.any(count == 0)
        ):
            raise ContributionTranscoderEvaluationError(
                "packed-sequence key arrays are malformed"
            )
        observed = np.unpackbits(
            bitmap.view(np.uint8),
            axis=1,
            bitorder="little",
        ).sum(axis=1, dtype=np.uint16)
        if not np.array_equal(observed, count):
            raise ContributionTranscoderEvaluationError(
                "position bitmap differs from sequence token counts"
            )
        if set(self.common_sst) != set(ENDPOINTS):
            raise ContributionTranscoderEvaluationError(
                "paired score endpoints differ from the frozen contract"
            )
        common: dict[str, np.ndarray] = {}
        for endpoint in ENDPOINTS:
            common[endpoint] = _finite_vector(
                f"{endpoint} common SST",
                self.common_sst[endpoint],
                sequence.shape,
                nonnegative=True,
            )
            if float(common[endpoint].sum(dtype=np.float64)) <= 0:
                raise ContributionTranscoderEvaluationError(
                    f"{endpoint} common SST is not positive"
                )
        if set(self.condition_sse) != set(CONDITIONS):
            raise ContributionTranscoderEvaluationError(
                "paired score conditions differ from A/R/B/F"
            )
        conditions: dict[str, dict[str, np.ndarray]] = {}
        for condition in CONDITIONS:
            raw = self.condition_sse[condition]
            if set(raw) != set(ENDPOINTS):
                raise ContributionTranscoderEvaluationError(
                    f"{condition} endpoints differ from the frozen contract"
                )
            conditions[condition] = {
                endpoint: _finite_vector(
                    f"{condition} {endpoint} SSE",
                    raw[endpoint],
                    sequence.shape,
                    nonnegative=True,
                )
                for endpoint in ENDPOINTS
            }
        object.__setattr__(self, "sequence_id", _readonly(sequence, np.int64))
        object.__setattr__(
            self,
            "position_bitmap",
            _readonly(bitmap, np.uint64),
        )
        object.__setattr__(self, "token_count", _readonly(count, np.uint16))
        object.__setattr__(self, "common_sst", common)
        object.__setattr__(self, "condition_sse", conditions)

    @property
    def tokens(self) -> int:
        return int(self.token_count.sum(dtype=np.int64))

    @property
    def clusters(self) -> int:
        return int(self.sequence_id.size)

    def endpoint_ev(self, condition: str, endpoint: str) -> float:
        if condition not in CONDITIONS or endpoint not in ENDPOINTS:
            raise ContributionTranscoderEvaluationError(
                "unknown condition or endpoint"
            )
        sse = float(
            self.condition_sse[condition][endpoint].sum(dtype=np.float64)
        )
        sst = float(self.common_sst[endpoint].sum(dtype=np.float64))
        return 1.0 - sse / sst

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SEQUENCE_SCORE_SCHEMA_VERSION,
            "source_manifest_sha256": self.source_manifest_sha256,
            "target_manifest_sha256": self.target_manifest_sha256,
            "recomposition_sidecar_sha256": (
                self.recomposition_sidecar_sha256
            ),
            "token_identity_sha256": self.token_identity_sha256,
            "tokens": self.tokens,
            "clusters": self.clusters,
            "sequence_id": self.sequence_id.tolist(),
            "position_bitmap": self.position_bitmap.tolist(),
            "token_count": self.token_count.tolist(),
            "common_sst": {
                endpoint: self.common_sst[endpoint].tolist()
                for endpoint in ENDPOINTS
            },
            "condition_sse": {
                condition: {
                    endpoint: self.condition_sse[condition][endpoint].tolist()
                    for endpoint in ENDPOINTS
                }
                for condition in CONDITIONS
            },
        }

    @property
    def score_artifact_sha256(self) -> str:
        return canonical_sha256(self._unsigned_payload())

    def to_dict(self) -> dict[str, Any]:
        unsigned = self._unsigned_payload()
        return {
            **unsigned,
            "score_artifact_sha256": canonical_sha256(unsigned),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "TranscoderPairedSequenceScores":
        if not isinstance(value, Mapping):
            raise ContributionTranscoderEvaluationError(
                "paired score artifact must be a mapping"
            )
        selected = dict(value)
        claimed = selected.pop("score_artifact_sha256", None)
        _sha256("score_artifact_sha256", claimed)
        if canonical_sha256(selected) != claimed:
            raise ContributionTranscoderEvaluationError(
                "paired score artifact self-hash mismatch"
            )
        if selected.pop("schema_version", None) != SEQUENCE_SCORE_SCHEMA_VERSION:
            raise ContributionTranscoderEvaluationError(
                "paired score artifact schema is incompatible"
            )
        tokens = selected.pop("tokens", None)
        clusters = selected.pop("clusters", None)
        result = cls(
            source_manifest_sha256=selected.pop("source_manifest_sha256"),
            target_manifest_sha256=selected.pop("target_manifest_sha256"),
            recomposition_sidecar_sha256=selected.pop(
                "recomposition_sidecar_sha256"
            ),
            token_identity_sha256=selected.pop("token_identity_sha256"),
            sequence_id=np.asarray(selected.pop("sequence_id"), dtype=np.int64),
            position_bitmap=np.asarray(
                selected.pop("position_bitmap"),
                dtype=np.uint64,
            ),
            token_count=np.asarray(selected.pop("token_count"), dtype=np.uint16),
            common_sst={
                name: np.asarray(rows, dtype=np.float64)
                for name, rows in selected.pop("common_sst").items()
            },
            condition_sse={
                condition: {
                    name: np.asarray(rows, dtype=np.float64)
                    for name, rows in endpoints.items()
                }
                for condition, endpoints in selected.pop("condition_sse").items()
            },
        )
        if selected or tokens != result.tokens or clusters != result.clusters:
            raise ContributionTranscoderEvaluationError(
                "paired score artifact fields or cardinalities changed"
            )
        return result


def aggregate_paired_sequence_scores(
    *,
    scores: Mapping[str, ContributionTranscoderTokenScores],
    sequence_id: Any,
    position: Any,
    source_manifest_sha256: str,
    target_manifest_sha256: str,
) -> TranscoderPairedSequenceScores:
    """Aggregate four aligned condition score streams by packed sequence."""

    if set(scores) != set(CONDITIONS):
        raise ContributionTranscoderEvaluationError(
            "paired aggregation requires exactly A/R/B/F"
        )
    first = scores["A"]
    valid = np.asarray(first.valid_mask)
    sequence = np.asarray(sequence_id)
    positions = np.asarray(position)
    if (
        sequence.dtype != np.int64
        or positions.dtype != np.uint16
        or sequence.shape != valid.shape
        or positions.shape != valid.shape
    ):
        raise ContributionTranscoderEvaluationError(
            "scientific token keys do not align with score rows"
        )
    if (
        np.any(sequence[valid] < 0)
        or np.any(positions[valid] >= SEQUENCE_LENGTH)
    ):
        raise ContributionTranscoderEvaluationError(
            "scientific token keys are outside the valid domain"
        )
    for condition in CONDITIONS:
        selected = scores[condition]
        if not np.array_equal(selected.valid_mask, valid):
            raise ContributionTranscoderEvaluationError(
                "condition score masks are not paired"
            )
        if (
            selected.recomposition_sidecar_sha256
            != first.recomposition_sidecar_sha256
        ):
            raise ContributionTranscoderEvaluationError(
                "condition scores use different recomposition sidecars"
            )
        for endpoint in ENDPOINTS:
            if not np.array_equal(
                selected.endpoint(endpoint)[1],
                first.endpoint(endpoint)[1],
            ):
                raise ContributionTranscoderEvaluationError(
                    f"{endpoint} SST is not common across conditions"
                )

    scientific_sequence = sequence[valid]
    scientific_position = positions[valid]
    values: dict[str, np.ndarray] = {}
    for endpoint in ENDPOINTS:
        values[f"sst:{endpoint}"] = first.endpoint(endpoint)[1][valid]
        for condition in CONDITIONS:
            values[f"sse:{condition}:{endpoint}"] = (
                scores[condition].endpoint(endpoint)[0][valid]
            )
    unique, bitmap, count, aggregated = _aggregate_rows(
        scientific_sequence,
        scientific_position,
        values,
    )
    return TranscoderPairedSequenceScores(
        source_manifest_sha256=_sha256(
            "source_manifest_sha256",
            source_manifest_sha256,
        ),
        target_manifest_sha256=_sha256(
            "target_manifest_sha256",
            target_manifest_sha256,
        ),
        recomposition_sidecar_sha256=first.recomposition_sidecar_sha256,
        token_identity_sha256=token_identity_sha256(
            scientific_sequence,
            scientific_position,
        ),
        sequence_id=unique,
        position_bitmap=bitmap,
        token_count=count,
        common_sst={
            endpoint: aggregated[f"sst:{endpoint}"] for endpoint in ENDPOINTS
        },
        condition_sse={
            condition: {
                endpoint: aggregated[f"sse:{condition}:{endpoint}"]
                for endpoint in ENDPOINTS
            }
            for condition in CONDITIONS
        },
    )


def expert_rank_diagnostics(
    scores: ContributionTranscoderTokenScores,
    expert_ids: Any,
) -> dict[str, Any]:
    """Return expert-balanced, frequency-quartile, and router-rank metrics."""

    ids = np.asarray(expert_ids)
    valid = scores.valid_mask
    if (
        ids.dtype not in (np.uint8, np.int32, np.int64)
        or ids.shape != (valid.size, TOP_K_EXPERTS)
        or np.any(ids[valid] < 0)
        or np.any(ids[valid] >= NUM_EXPERTS)
    ):
        raise ContributionTranscoderEvaluationError(
            "expert_ids must be integer [tokens,8] in [0,127]"
        )
    scientific_ids = ids[valid].astype(np.int64, copy=False).reshape(-1)
    route_sse = scores.route_component_sse[valid].reshape(-1)
    route_sst = scores.route_component_sst[valid].reshape(-1)
    counts = np.bincount(scientific_ids, minlength=NUM_EXPERTS).astype(np.int64)
    sse = np.bincount(
        scientific_ids,
        weights=route_sse,
        minlength=NUM_EXPERTS,
    ).astype(np.float64)
    sst = np.bincount(
        scientific_ids,
        weights=route_sst,
        minlength=NUM_EXPERTS,
    ).astype(np.float64)
    supported = sst > 0
    expert_ev = np.full((NUM_EXPERTS,), np.nan, dtype=np.float64)
    expert_ev[supported] = 1.0 - sse[supported] / sst[supported]
    macro = float(np.mean(expert_ev[supported])) if np.any(supported) else math.nan

    order = np.lexsort((np.arange(NUM_EXPERTS), counts))
    frequency_bins = []
    for bin_index, experts in enumerate(np.array_split(order, 4)):
        selected_sse = float(sse[experts].sum(dtype=np.float64))
        selected_sst = float(sst[experts].sum(dtype=np.float64))
        frequency_bins.append(
            {
                "bin": bin_index,
                "experts": [int(item) for item in experts],
                "route_count": int(counts[experts].sum(dtype=np.int64)),
                "sse": selected_sse,
                "sst": selected_sst,
                "nmse": (
                    selected_sse / selected_sst
                    if selected_sst > 0
                    else None
                ),
                "ev": (
                    1.0 - selected_sse / selected_sst
                    if selected_sst > 0
                    else None
                ),
            }
        )
    rank_bins = []
    for rank in range(TOP_K_EXPERTS):
        selected_sse = float(
            scores.route_component_sse[valid, rank].sum(dtype=np.float64)
        )
        selected_sst = float(
            scores.route_component_sst[valid, rank].sum(dtype=np.float64)
        )
        rank_bins.append(
            {
                "rank": rank,
                "route_count": int(np.count_nonzero(valid)),
                "sse": selected_sse,
                "sst": selected_sst,
                "nmse": selected_sse / selected_sst,
                "ev": 1.0 - selected_sse / selected_sst,
            }
        )
    return {
        "macro_expert_ev": macro,
        "supported_experts": int(np.count_nonzero(supported)),
        "expert_route_count": counts.tolist(),
        "expert_sse": sse.tolist(),
        "expert_sst": sst.tolist(),
        "expert_ev": [
            float(value) if math.isfinite(float(value)) else None
            for value in expert_ev
        ],
        "frequency_quartiles": frequency_bins,
        "router_rank": rank_bins,
        "diagnostics_sha256": canonical_sha256(
            {
                "counts": counts.tolist(),
                "sse": sse.tolist(),
                "sst": sst.tolist(),
            }
        ),
    }


def _validate_transcoder_parameter_contract(
    params: ContributionTranscoderParams,
    layout: ContributionSAEConfig,
) -> None:
    if not isinstance(params, ContributionTranscoderParams):
        raise ContributionTranscoderEvaluationError(
            "evaluation requires ContributionTranscoderParams"
        )
    expected = (
        (layout.total_features, layout.hidden_dim),
        (layout.hidden_dim, layout.total_features),
        (layout.total_features,),
    )
    observed = tuple(tuple(value.shape) for value in params)
    if observed != expected:
        raise ContributionTranscoderEvaluationError(
            "transcoder evaluation parameter shapes differ from the layout"
        )
    dtypes = {jnp.dtype(value.dtype) for value in params}
    if len(dtypes) != 1 or next(iter(dtypes)) not in {
        jnp.dtype(jnp.bfloat16),
        jnp.dtype(jnp.float32),
    }:
        raise ContributionTranscoderEvaluationError(
            "transcoder evaluation parameters must share BF16 or FP32 dtype"
        )


def _transcoder_param_specs(feature_axis: str) -> ContributionTranscoderParams:
    return transcoder_params_from_sae(_param_specs(feature_axis))


def _named_transcoder_params(
    mesh: Mesh,
    feature_axis: str,
) -> ContributionTranscoderParams:
    return jax.tree_util.tree_map(
        lambda spec: NamedSharding(mesh, spec),
        _transcoder_param_specs(feature_axis),
    )


def _build_transcoder_mapped_evaluator(
    mesh: Mesh,
    layout: ContributionSAEConfig,
    distributed: DistributedConfig,
    bank_capacity: int | None,
):
    data_axis = distributed.data_axis
    feature_axis = distributed.feature_axis
    routed_banks = layout.condition in {"A", "R"}
    if routed_banks:
        if (
            isinstance(bank_capacity, bool)
            or not isinstance(bank_capacity, int)
            or bank_capacity <= 0
        ):
            raise ValueError(
                "A/R transcoder evaluation requires a positive capacity"
            )
    elif bank_capacity is not None:
        raise ValueError("B/F transcoder evaluation does not accept capacity")
    capacity = int(bank_capacity or 1)

    def mapped(
        params: ContributionTranscoderParams,
        routed_input: jax.Array,
        expert_output: jax.Array,
        bank_ids: jax.Array,
        router_weights: jax.Array,
        valid_mask: jax.Array,
        source_mean: jax.Array,
        target_mean: jax.Array,
    ) -> tuple[ContributionEvaluationAux, ContributionEvaluationRows]:
        sae_params = transcoder_params_to_sae(params)
        local_features = params.encoder.shape[0]
        feature_offset = jax.lax.axis_index(feature_axis) * local_features

        def one_micro(_, inputs):
            source, target, banks, weights, valid = inputs
            compute_dtype = params.encoder.dtype
            centered_source = (
                source.astype(compute_dtype)
                - source_mean.astype(compute_dtype)
            )
            centered = jnp.broadcast_to(
                centered_source[:, None, :],
                (
                    centered_source.shape[0],
                    layout.top_routed,
                    layout.hidden_dim,
                ),
            )
            component_valid = jnp.broadcast_to(valid[:, None], banks.shape)
            if routed_banks:
                local_values, local_ids, overflow = _local_grouped_candidates(
                    sae_params,
                    centered,
                    banks,
                    component_valid,
                    layout,
                    feature_axis=feature_axis,
                    capacity=capacity,
                )
            else:
                local_values, local_ids = _local_rank_candidates(
                    sae_params,
                    centered,
                    component_valid,
                    layout,
                    feature_axis=feature_axis,
                )
                overflow = jnp.asarray(False)
            selection = _merge_local_candidates(
                local_values,
                local_ids,
                layout.k_per_contribution,
                feature_axis=feature_axis,
            )
            reconstruction = _decode_selected_columns_default_precision(
                params.decoder,
                selection,
                target_mean,
                feature_offset=feature_offset,
                feature_axis=feature_axis,
            ).reshape(target.shape)

            target32 = target.astype(jnp.float32)
            reconstruction32 = reconstruction.astype(jnp.float32)
            weights32 = weights.astype(jnp.float32)
            target_mean32 = target_mean.astype(jnp.float32)
            weighted_error = weights32[..., None] * (
                target32 - reconstruction32
            )
            weighted_centered = weights32[..., None] * (
                target32 - target_mean32[None, None, :]
            )
            route_sse = jnp.sum(
                jnp.square(weighted_error),
                axis=2,
                dtype=jnp.float32,
            )
            route_sst = jnp.sum(
                jnp.square(weighted_centered),
                axis=2,
                dtype=jnp.float32,
            )
            route_sse = jnp.where(component_valid, route_sse, 0.0)
            route_sst = jnp.where(component_valid, route_sst, 0.0)
            token_sse = jnp.sum(route_sse, axis=1, dtype=jnp.float32)
            token_sst = jnp.sum(route_sst, axis=1, dtype=jnp.float32)

            flat_valid = component_valid.reshape(-1)
            selected_valid = (
                (selection.values > 0)
                & selection.valid[:, None]
                & flat_valid[:, None]
            )
            per_l0 = jnp.sum(
                selected_valid,
                axis=1,
                dtype=jnp.int32,
            ).reshape(banks.shape)
            owned = (selection.latent_ids >= feature_offset) & (
                selection.latent_ids < feature_offset + local_features
            )
            local_ids = jnp.clip(
                selection.latent_ids - feature_offset,
                0,
                local_features - 1,
            )
            fired = (
                jnp.zeros((local_features,), jnp.int32)
                .at[local_ids.reshape(-1)]
                .max((selected_valid & owned).reshape(-1).astype(jnp.int32))
                > 0
            )
            source_valid = jnp.all(
                jnp.isfinite(source) | ~valid[:, None]
            )
            inputs_valid = source_valid & _scientific_input_valid(
                target,
                banks,
                weights,
                valid,
                layout,
            )
            topk_valid = jnp.all(selection.valid | ~flat_valid) & inputs_valid
            raw_target = jnp.einsum(
                "nkh,nk->nh",
                target32,
                weights32,
                preferred_element_type=jnp.float32,
                precision=jax.lax.Precision.DEFAULT,
            )
            raw_reconstruction = jnp.einsum(
                "nkh,nk->nh",
                reconstruction32,
                weights32,
                preferred_element_type=jnp.float32,
                precision=jax.lax.Precision.DEFAULT,
            )
            raw_target = jnp.where(valid[:, None], raw_target, 0.0)
            raw_reconstruction = jnp.where(
                valid[:, None],
                raw_reconstruction,
                0.0,
            )
            rows = ContributionEvaluationRows(
                component_token_sse=token_sse,
                component_token_sst=token_sst,
                component_route_sse=route_sse,
                component_route_sst=route_sst,
                per_contribution_l0=per_l0,
                raw_routed_target=raw_target,
                raw_routed_reconstruction=raw_reconstruction,
                bank_ids=banks.astype(jnp.int32),
                router_weights=weights32,
                valid_mask=valid,
            )
            local = (
                jnp.sum(token_sse, dtype=jnp.float32),
                jnp.sum(token_sst, dtype=jnp.float32),
                jnp.sum(per_l0, dtype=jnp.int32),
                jnp.sum(valid, dtype=jnp.int32),
                jnp.sum(component_valid, dtype=jnp.int32),
                topk_valid,
                fired,
                overflow,
            )
            return None, (local, rows)

        _, (locals_by_micro, rows) = jax.lax.scan(
            one_micro,
            None,
            (
                routed_input,
                expert_output,
                bank_ids,
                router_weights,
                valid_mask,
            ),
        )
        (
            local_sse,
            local_sst,
            local_l0,
            local_tokens,
            local_components,
            topk_by_micro,
            fired_by_micro,
            overflow_by_micro,
        ) = locals_by_micro
        local_sse = jnp.sum(local_sse, dtype=jnp.float32)
        local_sst = jnp.sum(local_sst, dtype=jnp.float32)
        local_l0 = jnp.sum(local_l0, dtype=jnp.int32)
        local_tokens = jnp.sum(local_tokens, dtype=jnp.int32)
        local_components = jnp.sum(local_components, dtype=jnp.int32)
        global_sse = jax.lax.pmean(
            jax.lax.psum(local_sse, data_axis),
            feature_axis,
        )
        global_sst = jax.lax.pmean(
            jax.lax.psum(local_sst, data_axis),
            feature_axis,
        )
        global_l0 = jax.lax.pmax(
            jax.lax.psum(local_l0, data_axis),
            feature_axis,
        )
        global_tokens = jax.lax.pmax(
            jax.lax.psum(local_tokens, data_axis),
            feature_axis,
        )
        global_components = jax.lax.pmax(
            jax.lax.psum(local_components, data_axis),
            feature_axis,
        )
        topk_valid = (
            jax.lax.pmin(
                jnp.all(topk_by_micro).astype(jnp.int32),
                (data_axis, feature_axis),
            )
            > 0
        )
        overflow = (
            jax.lax.pmax(
                jnp.any(overflow_by_micro).astype(jnp.int32),
                (data_axis, feature_axis),
            )
            > 0
        )
        fired = jax.lax.pmax(
            jnp.any(fired_by_micro, axis=0).astype(jnp.int32),
            data_axis,
        ) > 0
        params_finite = jnp.asarray(True)
        for leaf in jax.tree_util.tree_leaves(params):
            params_finite = params_finite & jnp.all(jnp.isfinite(leaf))
        params_finite = (
            params_finite
            & jnp.all(jnp.isfinite(source_mean))
            & jnp.all(jnp.isfinite(target_mean))
        )
        params_finite = (
            jax.lax.pmin(
                params_finite.astype(jnp.int32),
                (data_axis, feature_axis),
            )
            > 0
        )
        return (
            ContributionEvaluationAux(
                component_sse=global_sse,
                component_sst=global_sst,
                l0_sum=global_l0,
                valid_source_tokens=global_tokens,
                valid_components=global_components,
                topk_valid=topk_valid,
                fired_mask=fired,
                capacity_overflow=overflow,
                params_finite=params_finite,
            ),
            rows,
        )

    row_specs = ContributionEvaluationRows(
        P(None, data_axis),
        P(None, data_axis),
        P(None, data_axis, None),
        P(None, data_axis, None),
        P(None, data_axis, None),
        P(None, data_axis, None),
        P(None, data_axis, None),
        P(None, data_axis, None),
        P(None, data_axis, None),
        P(None, data_axis),
    )
    aux_specs = ContributionEvaluationAux(
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(feature_axis),
        P(),
        P(),
    )
    return jax.shard_map(
        mapped,
        mesh=mesh,
        in_specs=(
            _transcoder_param_specs(feature_axis),
            P(None, data_axis, None),
            P(None, data_axis, None, None),
            P(None, data_axis, None),
            P(None, data_axis, None),
            P(None, data_axis),
            P(None),
            P(None),
        ),
        out_specs=(aux_specs, row_specs),
        axis_names={data_axis, feature_axis},
        check_vma=False,
    )


class ContributionTranscoderDistributedEvaluator:
    """Compiled feature-sharded source-only evaluator for A/R/B/F."""

    def __init__(
        self,
        mesh: Mesh,
        layout: ContributionSAEConfig,
        distributed: DistributedConfig | None = None,
        *,
        bank_capacity: int | None = None,
    ) -> None:
        self.mesh = mesh
        self.layout = layout
        self.distributed = distributed or DistributedConfig()
        self.bank_capacity = bank_capacity
        _validate_contribution_mesh(mesh, layout, self.distributed)
        mapped = _build_transcoder_mapped_evaluator(
            mesh,
            layout,
            self.distributed,
            bank_capacity,
        )
        batch_shardings = contribution_transcoder_batch_shardings(
            mesh,
            self.distributed,
        )
        self._compiled = jax.jit(
            mapped,
            in_shardings=(
                _named_transcoder_params(
                    mesh,
                    self.distributed.feature_axis,
                ),
                batch_shardings.routed_input,
                batch_shardings.expert_output,
                batch_shardings.bank_ids,
                batch_shardings.router_weights,
                batch_shardings.valid_mask,
                NamedSharding(mesh, P()),
                NamedSharding(mesh, P()),
            ),
        )

    def _prepare(
        self,
        microbatches: Any,
        source_mean: Any,
        target_mean: Any,
        *,
        process_local: bool,
    ) -> tuple[ContributionTranscoderDistributedBatch, jax.Array, jax.Array]:
        batch = (
            microbatches
            if isinstance(
                microbatches,
                ContributionTranscoderDistributedBatch,
            )
            else stack_contribution_transcoder_microbatches(
                microbatches,
                self.layout,
                self.distributed,
            )
        )
        _validate_stacked_batch(batch, self.layout, self.distributed)
        placed = (
            shard_process_local_contribution_transcoder_batch(
                batch,
                self.mesh,
                self.distributed,
            )
            if process_local
            else shard_contribution_transcoder_batch(
                batch,
                self.mesh,
                self.distributed,
            )
        )
        means = []
        for name, value in (
            ("source_mean", source_mean),
            ("target_mean", target_mean),
        ):
            selected = np.asarray(value)
            if (
                selected.shape != (self.layout.hidden_dim,)
                or not np.issubdtype(selected.dtype, np.floating)
                or not np.isfinite(selected).all()
            ):
                raise ContributionTranscoderEvaluationError(
                    f"{name} must be finite [{self.layout.hidden_dim}]"
                )
            means.append(
                replicate_process_local_array(
                    jnp.asarray(selected, jnp.float32),
                    self.mesh,
                )
            )
        return placed, means[0], means[1]

    def compile(
        self,
        params: ContributionTranscoderParams,
        microbatches: Any,
        source_mean: Any,
        target_mean: Any,
        *,
        process_local: bool = False,
    ) -> None:
        _validate_transcoder_parameter_contract(params, self.layout)
        batch, source, target = self._prepare(
            microbatches,
            source_mean,
            target_mean,
            process_local=process_local,
        )
        self._compiled.lower(
            params,
            batch.routed_input,
            batch.expert_output,
            batch.bank_ids,
            batch.router_weights,
            batch.valid_mask,
            source,
            target,
        ).compile()

    def step(
        self,
        params: ContributionTranscoderParams,
        microbatches: Any,
        source_mean: Any,
        target_mean: Any,
        *,
        process_local: bool = False,
    ) -> tuple[ContributionEvaluationAux, ContributionEvaluationChunk]:
        _validate_transcoder_parameter_contract(params, self.layout)
        batch, source, target = self._prepare(
            microbatches,
            source_mean,
            target_mean,
            process_local=process_local,
        )
        aux, rows = self._compiled(
            params,
            batch.routed_input,
            batch.expert_output,
            batch.bank_ids,
            batch.router_weights,
            batch.valid_mask,
            source,
            target,
        )
        host = jax.device_get(
            (
                aux.component_sse,
                aux.component_sst,
                aux.l0_sum,
                aux.valid_source_tokens,
                aux.valid_components,
                aux.topk_valid,
                aux.capacity_overflow,
                aux.params_finite,
            )
        )
        (
            component_sse,
            component_sst,
            l0_sum,
            tokens,
            components,
            topk_valid,
            overflow,
            params_finite,
        ) = host
        if bool(overflow):
            raise ContributionTranscoderEvaluationError(
                "transcoder evaluation exceeded fixed A/R capacity"
            )
        if not bool(params_finite):
            raise ContributionTranscoderEvaluationError(
                "transcoder parameters or centering means are non-finite"
            )
        if not bool(topk_valid):
            raise ContributionTranscoderEvaluationError(
                "invalid inputs or fewer than two positive features"
            )
        token_count = int(tokens)
        if (
            token_count <= 0
            or int(components) != token_count * self.layout.top_routed
            or int(l0_sum) != token_count * self.layout.total_l0
        ):
            raise ContributionTranscoderEvaluationError(
                "transcoder evaluation cardinality differs from exact L0"
            )
        if (
            not math.isfinite(float(component_sse))
            or not math.isfinite(float(component_sst))
            or float(component_sse) < 0
            or float(component_sst) <= 0
        ):
            raise ContributionTranscoderEvaluationError(
                "transcoder component SSE/SST is invalid"
            )
        chunk = _host_chunk(rows)
        scientific_l0 = chunk.per_contribution_l0[chunk.valid_mask]
        if (
            scientific_l0.shape
            != (chunk.tokens, self.layout.top_routed)
            or np.any(scientific_l0 != self.layout.k_per_contribution)
        ):
            raise ContributionTranscoderEvaluationError(
                "transcoder evaluation failed exact per-slot TopK"
            )
        return aux, chunk

    @staticmethod
    def complete_feature_mask(aux: ContributionEvaluationAux) -> np.ndarray:
        """Materialize the global feature coverage vector without replicas."""

        return _addressable_feature_mask(aux.fired_mask)


__all__ = [
    "CONDITIONS",
    "ENDPOINTS",
    "SEQUENCE_SCORE_SCHEMA_VERSION",
    "ContributionTranscoderEvaluationError",
    "ContributionTranscoderDistributedEvaluator",
    "ContributionTranscoderTokenScores",
    "TranscoderPairedSequenceScores",
    "aggregate_paired_sequence_scores",
    "expert_rank_diagnostics",
    "score_transcoder_predictions",
    "score_transcoder_sufficient_rows",
]
