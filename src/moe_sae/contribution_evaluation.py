"""Fail-closed distributed evaluation for Experiment-2 routed SAEs.

Evaluation deliberately reuses the production sparse execution primitives:
whole-bank grouped GEMMs for A/R, router-rank GEMMs for B/F, feature-sharded
TopK, and sparse selected-column decode.  It never constructs a dense latent
vector.

The device result has two layers:

* globally reduced FP32 component SSE/SST and exact integer cardinalities;
* process-local, data-sharded sufficient rows for sequence-cluster scoring and
  exact checkpoint-free routed recomposition.

The latter contains only per-token errors and the two raw routed aggregates
``sum_s g_s e_s`` and ``sum_s g_s e_hat_s``.  It does not retain the eight
decoded expert tensors after a batch has been scored.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from moe_sae.contribution_artifacts import (
    SEQUENCE_LENGTH,
    bf16_bits_to_float32,
    canonical_sha256,
    float32_to_bf16_bits,
    token_identity_sha256,
    validate_contribution_statistics,
)
from moe_sae.contribution_distributed import (
    ContributionDistributedBatch,
    _decode_selected_columns_default_precision,
    _local_grouped_candidates,
    _local_rank_candidates,
    _merge_local_candidates,
    _scientific_input_valid,
    _validate_contribution_mesh,
    _validate_host_microbatch,
    _validate_stacked_batch_shape,
    contribution_batch_shardings,
    shard_contribution_batch,
    shard_process_local_contribution_batch,
    stack_contribution_microbatches,
)
from moe_sae.contribution_input import (
    CapacityReceipt,
    ContributionHostMicrobatch,
    ContributionProcessInput,
    PADDING_POSITION,
    PADDING_SEQUENCE_ID,
    merge_capacity_receipts,
)
from moe_sae.contribution_recomposition import (
    decode_recomposition_sidecar,
    maxtext_rmsnorm_bf16,
)
from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_training import forward_contributions
from moe_sae.distributed import (
    DistributedConfig,
    _param_specs,
    replicate_process_local_array,
)
from moe_sae.sae import SAEParams

EVALUATION_SCHEMA_VERSION = "gemma4-contribution-evaluation-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContributionEvaluationError(RuntimeError):
    """Evaluation evidence violates the frozen scientific contract."""


class ContributionEvaluationAux(NamedTuple):
    """Globally reduced scalars plus a feature-sharded firing mask."""

    component_sse: jax.Array
    component_sst: jax.Array
    l0_sum: jax.Array
    valid_source_tokens: jax.Array
    valid_components: jax.Array
    topk_valid: jax.Array
    fired_mask: jax.Array
    capacity_overflow: jax.Array
    params_finite: jax.Array


class ContributionEvaluationRows(NamedTuple):
    """Data-sharded rows sufficient for clustered and routed evaluation."""

    component_token_sse: jax.Array
    component_token_sst: jax.Array
    component_route_sse: jax.Array
    component_route_sst: jax.Array
    per_contribution_l0: jax.Array
    raw_routed_target: jax.Array
    raw_routed_reconstruction: jax.Array
    bank_ids: jax.Array
    router_weights: jax.Array
    valid_mask: jax.Array


@dataclasses.dataclass(frozen=True, slots=True)
class ContributionEvaluationChunk:
    """One process's host-addressable evaluation rows.

    Arrays retain the static ``[microbatches, local_rows, ...]`` shape and the
    explicit padding mask.  This keeps row order aligned with
    :class:`ContributionProcessInput`; consumers may attach the corresponding
    sealed sequence IDs before aggregating by packed sequence.
    """

    component_token_sse: np.ndarray
    component_token_sst: np.ndarray
    component_route_sse: np.ndarray
    component_route_sst: np.ndarray
    per_contribution_l0: np.ndarray
    raw_routed_target: np.ndarray
    raw_routed_reconstruction: np.ndarray
    bank_ids: np.ndarray
    router_weights: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        valid = np.asarray(self.valid_mask)
        if valid.ndim != 2 or valid.dtype != np.bool_:
            raise ContributionEvaluationError(
                "evaluation valid_mask must be bool [microbatches,rows]"
            )
        prefix = valid.shape
        token_fields = (
            ("component_token_sse", self.component_token_sse),
            ("component_token_sst", self.component_token_sst),
        )
        for name, value in token_fields:
            array = np.asarray(value)
            if array.shape != prefix or not np.issubdtype(array.dtype, np.floating):
                raise ContributionEvaluationError(
                    f"{name} must be floating-point with shape {prefix}"
                )
            if not np.isfinite(array).all() or np.any(array < 0):
                raise ContributionEvaluationError(
                    f"{name} must be finite and nonnegative"
                )

        route_shape = np.asarray(self.component_route_sse).shape
        if len(route_shape) != 3 or route_shape[:2] != prefix:
            raise ContributionEvaluationError(
                "component_route_sse must be [microbatches,rows,routes]"
            )
        for name, value in (
            ("component_route_sse", self.component_route_sse),
            ("component_route_sst", self.component_route_sst),
        ):
            array = np.asarray(value)
            if (
                array.shape != route_shape
                or not np.issubdtype(array.dtype, np.floating)
                or not np.isfinite(array).all()
                or np.any(array < 0)
            ):
                raise ContributionEvaluationError(
                    f"{name} must be finite nonnegative floating-point {route_shape}"
                )
        l0 = np.asarray(self.per_contribution_l0)
        if l0.shape != route_shape or not np.issubdtype(l0.dtype, np.integer):
            raise ContributionEvaluationError(
                f"per_contribution_l0 must be integer {route_shape}"
            )
        banks = np.asarray(self.bank_ids)
        weights = np.asarray(self.router_weights)
        if banks.shape != route_shape or not np.issubdtype(banks.dtype, np.integer):
            raise ContributionEvaluationError(f"bank_ids must be integer {route_shape}")
        if (
            weights.shape != route_shape
            or not np.issubdtype(weights.dtype, np.floating)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
        ):
            raise ContributionEvaluationError(
                f"router_weights must be finite nonnegative {route_shape}"
            )
        hidden_shape = np.asarray(self.raw_routed_target).shape
        if len(hidden_shape) != 3 or hidden_shape[:2] != prefix:
            raise ContributionEvaluationError(
                "raw_routed_target must be [microbatches,rows,hidden]"
            )
        for name, value in (
            ("raw_routed_target", self.raw_routed_target),
            ("raw_routed_reconstruction", self.raw_routed_reconstruction),
        ):
            array = np.asarray(value)
            if (
                array.shape != hidden_shape
                or not np.issubdtype(array.dtype, np.floating)
                or not np.isfinite(array).all()
            ):
                raise ContributionEvaluationError(
                    f"{name} must be finite floating-point {hidden_shape}"
                )

        # Device rows are explicitly zeroed on padding.  Requiring that here
        # prevents a later sink from accidentally treating padding as evidence.
        padding = ~valid
        for name in (
            "component_token_sse",
            "component_token_sst",
            "component_route_sse",
            "component_route_sst",
            "per_contribution_l0",
            "raw_routed_target",
            "raw_routed_reconstruction",
        ):
            value = np.asarray(getattr(self, name))
            if np.any(value[padding] != 0):
                raise ContributionEvaluationError(
                    f"{name} must be exactly zero on padded rows"
                )

        for field in dataclasses.fields(self):
            array = np.ascontiguousarray(np.asarray(getattr(self, field.name)))
            array.setflags(write=False)
            object.__setattr__(self, field.name, array)

    @property
    def tokens(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def component_sse(self) -> float:
        return float(
            np.sum(
                self.component_token_sse[self.valid_mask],
                dtype=np.float64,
            )
        )

    @property
    def component_sst(self) -> float:
        return float(
            np.sum(
                self.component_token_sst[self.valid_mask],
                dtype=np.float64,
            )
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ContributionEvaluationEvidenceChunk:
    """Identity-bound process-local rows delivered to an evaluation sink."""

    update_index: int
    sequence_id: np.ndarray
    position: np.ndarray
    scores: ContributionEvaluationChunk

    def __post_init__(self) -> None:
        if (
            isinstance(self.update_index, bool)
            or not isinstance(self.update_index, int)
            or self.update_index < 0
        ):
            raise ContributionEvaluationError(
                "evaluation update_index must be a nonnegative integer"
            )
        sequence = np.asarray(self.sequence_id)
        position = np.asarray(self.position)
        valid = np.asarray(self.scores.valid_mask)
        if sequence.dtype != np.int64 or sequence.shape != valid.shape:
            raise ContributionEvaluationError(
                "evaluation sequence_id must be int64 and align with score rows"
            )
        if position.dtype != np.uint16 or position.shape != valid.shape:
            raise ContributionEvaluationError(
                "evaluation position must be uint16 and align with score rows"
            )
        if np.any(sequence[valid] < 0) or np.any(
            position[valid] >= SEQUENCE_LENGTH
        ):
            raise ContributionEvaluationError(
                "scientific evaluation rows contain an invalid token identity"
            )
        if np.any(sequence[~valid] != PADDING_SEQUENCE_ID) or np.any(
            position[~valid] != PADDING_POSITION
        ):
            raise ContributionEvaluationError(
                "padded evaluation identities do not use the exact sentinels"
            )
        if np.count_nonzero(valid):
            records = np.rec.fromarrays(
                (sequence[valid], position[valid]),
                names=("sequence_id", "position"),
            )
            if np.unique(records).size != records.size:
                raise ContributionEvaluationError(
                    "evaluation evidence chunk contains duplicate scientific keys"
                )
        sequence = np.ascontiguousarray(sequence)
        position = np.ascontiguousarray(position)
        sequence.setflags(write=False)
        position.setflags(write=False)
        object.__setattr__(self, "sequence_id", sequence)
        object.__setattr__(self, "position", position)

    @property
    def tokens(self) -> int:
        return self.scores.tokens

    @property
    def token_identity_sha256(self) -> str:
        valid = self.scores.valid_mask
        return token_identity_sha256(
            self.sequence_id[valid],
            self.position[valid],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class RoutedRecompositionScores:
    """Per-token routed-branch SSE/SST after exact Gemma BF16 round points."""

    token_sse: np.ndarray
    token_sst: np.ndarray
    valid_mask: np.ndarray
    recomposition_sidecar_sha256: str

    def __post_init__(self) -> None:
        valid = np.asarray(self.valid_mask)
        sse = np.asarray(self.token_sse)
        sst = np.asarray(self.token_sst)
        if valid.ndim != 2 or valid.dtype != np.bool_:
            raise ContributionEvaluationError(
                "routed score valid_mask must be bool [microbatches,rows]"
            )
        for name, value in (("token_sse", sse), ("token_sst", sst)):
            if (
                value.shape != valid.shape
                or not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all()
                or np.any(value < 0)
                or np.any(value[~valid] != 0)
            ):
                raise ContributionEvaluationError(
                    f"routed {name} must be finite, nonnegative, aligned, and "
                    "zero on padding"
                )
        if (
            not isinstance(self.recomposition_sidecar_sha256, str)
            or len(self.recomposition_sidecar_sha256) != 64
        ):
            raise ContributionEvaluationError(
                "routed scores require a recomposition sidecar SHA-256"
            )
        for name in ("token_sse", "token_sst", "valid_mask"):
            array = np.ascontiguousarray(np.asarray(getattr(self, name)))
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def tokens(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def sse(self) -> float:
        return float(np.sum(self.token_sse[self.valid_mask], dtype=np.float64))

    @property
    def sst(self) -> float:
        return float(np.sum(self.token_sst[self.valid_mask], dtype=np.float64))

    @property
    def ev(self) -> float:
        if not self.sst > 0:
            raise ContributionEvaluationError("routed SST must be positive")
        return 1.0 - self.sse / self.sst


@dataclasses.dataclass(frozen=True, slots=True)
class ContributionEvaluationResult:
    """Hash-bindable aggregate from one full sealed evaluation prefix."""

    schema_version: str
    condition: str
    randomization_key: int
    capture_manifest_sha256: str
    input_receipt_sha256: str
    checkpoint_sha256: str
    centering_mean_sha256: str
    capacity_receipt_sha256: str | None
    tokens: int
    valid_components: int
    component_sse: float
    component_sst: float
    computed_component_sst: float
    l0_sum: int
    l0_per_contribution: int
    total_l0: int
    fired_feature_fraction: float
    dead_rate: float
    updates: int
    local_sufficient_rows: int

    @property
    def component_nmse(self) -> float:
        return self.component_sse / self.component_sst

    @property
    def component_ev(self) -> float:
        return 1.0 - self.component_nmse

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(dataclasses.asdict(self))


def _named_params(mesh: Mesh, feature_axis: str) -> SAEParams:
    return jax.tree_util.tree_map(
        lambda spec: NamedSharding(mesh, spec),
        _param_specs(feature_axis),
    )


def _validate_parameter_contract(
    params: SAEParams,
    layout: ContributionSAEConfig,
) -> None:
    expected = (
        (layout.total_features, layout.hidden_dim),
        (layout.hidden_dim, layout.total_features),
        (layout.total_features,),
    )
    observed = (
        tuple(params.encoder.shape),
        tuple(params.decoder.shape),
        tuple(params.bias.shape),
    )
    if observed != expected:
        raise ContributionEvaluationError(
            "evaluation parameter shapes differ from the frozen layout: "
            f"observed={observed}, expected={expected}"
        )
    dtypes = {jnp.dtype(value.dtype) for value in params}
    if len(dtypes) != 1 or next(iter(dtypes)) not in {
        jnp.dtype(jnp.bfloat16),
        jnp.dtype(jnp.float32),
    }:
        raise ContributionEvaluationError(
            "evaluation parameters must share BF16 or FP32 dtype"
        )


def _build_mapped_evaluator(
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
                "A/R contribution evaluation requires a positive fixed capacity"
            )
    elif bank_capacity is not None:
        raise ValueError("B/F contribution evaluation does not accept a capacity")
    capacity = int(bank_capacity or 1)

    def mapped(
        params: SAEParams,
        expert_output: jax.Array,
        bank_ids: jax.Array,
        router_weights: jax.Array,
        valid_mask: jax.Array,
        mean: jax.Array,
    ) -> tuple[ContributionEvaluationAux, ContributionEvaluationRows]:
        local_features = params.encoder.shape[0]
        feature_offset = jax.lax.axis_index(feature_axis) * local_features

        def one_micro(_, inputs):
            target, banks, weights, valid = inputs
            compute_dtype = params.encoder.dtype
            centered = target.astype(compute_dtype) - mean.astype(compute_dtype)
            component_valid = jnp.broadcast_to(valid[:, None], banks.shape)
            if routed_banks:
                local_values, local_ids, overflow = _local_grouped_candidates(
                    params,
                    centered,
                    banks,
                    component_valid,
                    layout,
                    feature_axis=feature_axis,
                    capacity=capacity,
                )
            else:
                local_values, local_ids = _local_rank_candidates(
                    params,
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
                mean,
                feature_offset=feature_offset,
                feature_axis=feature_axis,
            ).reshape(target.shape)

            target32 = target.astype(jnp.float32)
            reconstruction32 = reconstruction.astype(jnp.float32)
            weights32 = weights.astype(jnp.float32)
            mean32 = mean.astype(jnp.float32)
            weighted_error = weights32[..., None] * (
                target32 - reconstruction32
            )
            weighted_centered = weights32[..., None] * (
                target32 - mean32[None, None, :]
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
            per_l0 = jnp.sum(selected_valid, axis=1, dtype=jnp.int32).reshape(
                banks.shape
            )
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
            inputs_valid = _scientific_input_valid(
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
            (expert_output, bank_ids, router_weights, valid_mask),
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

        # Each feature shard holds the same post-decode scientific totals.  Sum
        # only over data, then choose that identical integer/FP32 value across
        # feature replicas.  Integer pmax avoids an inexact float round-trip.
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
        params_finite = params_finite & jnp.all(jnp.isfinite(mean))
        params_finite = (
            jax.lax.pmin(
                params_finite.astype(jnp.int32),
                (data_axis, feature_axis),
            )
            > 0
        )
        aux = ContributionEvaluationAux(
            component_sse=global_sse,
            component_sst=global_sst,
            l0_sum=global_l0,
            valid_source_tokens=global_tokens,
            valid_components=global_components,
            topk_valid=topk_valid,
            fired_mask=fired,
            capacity_overflow=overflow,
            params_finite=params_finite,
        )
        return aux, rows

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
            _param_specs(feature_axis),
            P(None, data_axis, None, None),
            P(None, data_axis, None),
            P(None, data_axis, None),
            P(None, data_axis),
            P(None),
        ),
        out_specs=(aux_specs, row_specs),
        axis_names={data_axis, feature_axis},
        check_vma=False,
    )


def _addressable_data_rows(value: jax.Array) -> np.ndarray:
    """Materialize each process-local data shard once, dropping replicas."""

    if value.is_fully_addressable:
        return np.asarray(jax.device_get(value))
    unique: dict[str, tuple[Any, np.ndarray]] = {}
    for shard in value.addressable_shards:
        key = repr(shard.index)
        unique.setdefault(key, (shard.index, np.asarray(jax.device_get(shard.data))))
    if not unique:
        raise ContributionEvaluationError(
            "evaluation output has no process-addressable data shards"
        )

    def start(item: tuple[Any, np.ndarray]) -> int:
        index = item[0][1]
        if not isinstance(index, slice):
            return int(index)
        return int(index.start or 0)

    pieces = [item[1] for item in sorted(unique.values(), key=start)]
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)


def _addressable_feature_mask(value: jax.Array) -> np.ndarray:
    """Materialize a complete host-local feature row, dropping data replicas."""

    if value.is_fully_addressable:
        return np.asarray(jax.device_get(value), dtype=np.bool_)
    unique: dict[str, tuple[Any, np.ndarray]] = {}
    for shard in value.addressable_shards:
        key = repr(shard.index)
        unique.setdefault(key, (shard.index, np.asarray(jax.device_get(shard.data))))
    if not unique:
        raise ContributionEvaluationError(
            "evaluation firing mask has no process-addressable shards"
        )

    def start(item: tuple[Any, np.ndarray]) -> int:
        index = item[0][0]
        if not isinstance(index, slice):
            return int(index)
        return int(index.start or 0)

    pieces = [item[1] for item in sorted(unique.values(), key=start)]
    return np.asarray(
        pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0),
        dtype=np.bool_,
    )


def _host_chunk(rows: ContributionEvaluationRows) -> ContributionEvaluationChunk:
    values = {
        field: _addressable_data_rows(getattr(rows, field))
        for field in ContributionEvaluationRows._fields
    }
    return ContributionEvaluationChunk(**values)


class ContributionDistributedEvaluator:
    """Compiled feature-sharded evaluator for one immutable condition layout."""

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
        mapped = _build_mapped_evaluator(
            mesh,
            layout,
            self.distributed,
            bank_capacity,
        )
        self._compiled = jax.jit(
            mapped,
            in_shardings=(
                _named_params(mesh, self.distributed.feature_axis),
                contribution_batch_shardings(mesh, self.distributed).expert_output,
                contribution_batch_shardings(mesh, self.distributed).bank_ids,
                contribution_batch_shardings(mesh, self.distributed).router_weights,
                contribution_batch_shardings(mesh, self.distributed).valid_mask,
                NamedSharding(mesh, P()),
            ),
        )

    def _prepare(
        self,
        microbatches: Sequence[Any] | ContributionDistributedBatch,
        mean: Any,
        *,
        process_local: bool,
    ) -> tuple[ContributionDistributedBatch, jax.Array]:
        batch = (
            microbatches
            if isinstance(microbatches, ContributionDistributedBatch)
            else stack_contribution_microbatches(
                microbatches,
                self.layout,
                self.distributed,
            )
        )
        _validate_stacked_batch_shape(batch, self.layout, self.distributed)
        placed = (
            shard_process_local_contribution_batch(
                batch,
                self.mesh,
                self.distributed,
            )
            if process_local
            else shard_contribution_batch(
                batch,
                self.mesh,
                self.distributed,
            )
        )
        host_mean = np.asarray(mean)
        if (
            host_mean.shape != (self.layout.hidden_dim,)
            or not np.issubdtype(host_mean.dtype, np.floating)
            or not np.isfinite(host_mean).all()
        ):
            raise ValueError(
                f"mean must be finite floating-point [{self.layout.hidden_dim}]"
            )
        mean_array = replicate_process_local_array(
            jnp.asarray(host_mean, jnp.float32),
            self.mesh,
        )
        return placed, mean_array

    def compile(
        self,
        params: SAEParams,
        microbatches: Sequence[Any] | ContributionDistributedBatch,
        mean: Any,
        *,
        process_local: bool = False,
    ) -> None:
        """Compile without consuming or accepting any evaluation evidence."""

        _validate_parameter_contract(params, self.layout)
        batch, mean_array = self._prepare(
            microbatches,
            mean,
            process_local=process_local,
        )
        self._compiled.lower(
            params,
            batch.expert_output,
            batch.bank_ids,
            batch.router_weights,
            batch.valid_mask,
            mean_array,
        ).compile()

    def step(
        self,
        params: SAEParams,
        microbatches: Sequence[Any] | ContributionDistributedBatch,
        mean: Any,
        *,
        process_local: bool = False,
    ) -> tuple[ContributionEvaluationAux, ContributionEvaluationChunk]:
        """Score one static group and reject it before returning bad evidence."""

        _validate_parameter_contract(params, self.layout)
        batch, mean_array = self._prepare(
            microbatches,
            mean,
            process_local=process_local,
        )
        aux, rows = self._compiled(
            params,
            batch.expert_output,
            batch.bank_ids,
            batch.router_weights,
            batch.valid_mask,
            mean_array,
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
        ) = jax.device_get(
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
        if bool(overflow):
            raise ContributionEvaluationError(
                "evaluation exceeded its precomputed A/R grouped-GEMM capacity"
            )
        if not bool(params_finite):
            raise ContributionEvaluationError(
                "evaluation parameters, centering mean, or a remote feature "
                "shard is non-finite"
            )
        if not bool(topk_valid):
            raise ContributionEvaluationError(
                "invalid inputs or fewer than k strictly positive features on "
                "a scientific contribution"
            )
        token_count = int(tokens)
        if token_count <= 0:
            raise ContributionEvaluationError(
                "evaluation group contained zero scientific source tokens"
            )
        if int(components) != token_count * self.layout.top_routed:
            raise ContributionEvaluationError(
                "evaluation component count differs from source-token count"
            )
        expected_l0 = token_count * self.layout.total_l0
        if int(l0_sum) != expected_l0:
            raise ContributionEvaluationError(
                "evaluation TopK cardinality differs from the frozen layout: "
                f"selected={int(l0_sum)}, expected={expected_l0}"
            )
        if (
            not math.isfinite(float(component_sse))
            or not math.isfinite(float(component_sst))
            or float(component_sse) < 0
            or float(component_sst) <= 0
        ):
            raise ContributionEvaluationError(
                "evaluation component SSE/SST is non-finite or non-positive"
            )

        chunk = _host_chunk(rows)
        scientific_l0 = chunk.per_contribution_l0[chunk.valid_mask]
        if (
            scientific_l0.shape
            != (chunk.tokens, self.layout.top_routed)
            or np.any(scientific_l0 != self.layout.k_per_contribution)
        ):
            raise ContributionEvaluationError(
                "evaluation failed exact per-contribution TopK"
            )
        return aux, chunk


def _validate_capacity_receipt(
    receipt: CapacityReceipt,
    layout: ContributionSAEConfig,
) -> None:
    unsigned = {
        "schema_version": receipt.schema_version,
        "input_receipt_sha256": receipt.input_receipt_sha256,
        "condition": receipt.condition,
        "randomization_key": receipt.randomization_key,
        "observed_maximum": receipt.observed_maximum,
        "margin_fraction": receipt.margin_fraction,
        "capacity": receipt.capacity,
    }
    if receipt.schema_version != "gemma4-contribution-capacity-receipt-v1":
        raise ContributionEvaluationError("unknown capacity receipt schema")
    if canonical_sha256(unsigned) != receipt.receipt_sha256:
        raise ContributionEvaluationError("capacity receipt hash is invalid")
    if (
        receipt.condition != layout.condition
        or receipt.randomization_key != layout.randomization_key
        or receipt.observed_maximum <= 0
        or receipt.capacity < receipt.observed_maximum
    ):
        raise ContributionEvaluationError(
            "capacity receipt does not match the evaluation layout"
        )


def _evidence_chunk(
    update_index: int,
    microbatches: Sequence[ContributionHostMicrobatch],
    scores: ContributionEvaluationChunk,
) -> ContributionEvaluationEvidenceChunk:
    sequence = np.stack(
        [np.asarray(item.sequence_id) for item in microbatches],
        axis=0,
    )
    position = np.stack(
        [np.asarray(item.position) for item in microbatches],
        axis=0,
    )
    return ContributionEvaluationEvidenceChunk(
        update_index=update_index,
        sequence_id=sequence,
        position=position,
        scores=scores,
    )


def _sealed_component_sst(
    source: ContributionProcessInput,
    mean: Any,
) -> float:
    """Derive the common FP64 denominator from the sealed capture statistics."""

    if source.total_tokens != source.dataset.total_tokens:
        raise ContributionEvaluationError(
            "authoritative evaluation requires the complete sealed capture, "
            "not a manifest prefix"
        )
    manifest = getattr(source.dataset, "manifest", None)
    statistics = manifest.get("statistics") if isinstance(manifest, Mapping) else None
    if not isinstance(statistics, Mapping):
        raise ContributionEvaluationError(
            "sealed evaluation manifest lacks contribution statistics"
        )
    try:
        checked = validate_contribution_statistics(
            statistics,
            expected_tokens=source.total_tokens,
        )
    except Exception as error:
        raise ContributionEvaluationError(
            "sealed evaluation contribution statistics are invalid"
        ) from error
    pooled_mean = np.asarray(mean)
    if (
        pooled_mean.shape != (source.layout.hidden_dim,)
        or not np.issubdtype(pooled_mean.dtype, np.floating)
        or not np.isfinite(pooled_mean).all()
    ):
        raise ContributionEvaluationError(
            f"mean must be finite floating-point [{source.layout.hidden_dim}]"
        )
    if source.layout.hidden_dim != len(
        checked["weighted_contribution"]["weight_squared_output_sum"]
    ):
        raise ContributionEvaluationError(
            "capture statistics hidden size differs from the evaluation layout"
        )
    mean64 = pooled_mean.astype(np.float64)
    weighted = checked["weighted_contribution"]
    weighted_sum = np.asarray(
        weighted["weight_squared_output_sum"],
        dtype=np.float64,
    )
    sst = (
        float(weighted["squared_norm"])
        - 2.0 * float(np.dot(mean64, weighted_sum))
        + float(weighted["weight_squared_sum"])
        * float(np.dot(mean64, mean64))
    )
    if not math.isfinite(sst) or sst <= 0:
        raise ContributionEvaluationError(
            "sealed statistics imply a non-positive component SST"
        )
    return sst


def evaluate_contribution_input(
    params: SAEParams,
    source: ContributionProcessInput,
    mean: Any,
    mesh: Mesh,
    *,
    checkpoint_sha256: str,
    expected_component_sst: float | None = None,
    capacity_receipts: Sequence[CapacityReceipt] = (),
    distributed: DistributedConfig | None = None,
    row_sink: Callable[[ContributionEvaluationEvidenceChunk], None] | None = None,
) -> ContributionEvaluationResult:
    """Evaluate exactly one sealed/static :class:`ContributionProcessInput`.

    ``row_sink`` is called synchronously for each update with only this
    process's identity-bound rows.  A production sink should immediately
    aggregate them by packed sequence, then discard the raw routed aggregates.
    """

    selected = distributed or DistributedConfig(
        microbatches_per_update=source.microbatches_per_update
    )
    if source.microbatches_per_update != selected.microbatches_per_update:
        raise ContributionEvaluationError(
            "evaluation input and distributed microbatch counts differ"
        )
    if source.process_count != jax.process_count():
        raise ContributionEvaluationError(
            "evaluation input process_count differs from the JAX runtime"
        )
    if source.process_index != jax.process_index():
        raise ContributionEvaluationError(
            "evaluation input process_index differs from the JAX runtime"
        )
    if (
        not isinstance(checkpoint_sha256, str)
        or _SHA256_RE.fullmatch(checkpoint_sha256) is None
    ):
        raise ContributionEvaluationError(
            "evaluation requires the trained checkpoint SHA-256"
        )
    sealed_component_sst = _sealed_component_sst(source, mean)
    if expected_component_sst is not None and not math.isclose(
        float(expected_component_sst),
        sealed_component_sst,
        rel_tol=1e-12,
        abs_tol=1e-9 * max(1.0, sealed_component_sst),
    ):
        raise ContributionEvaluationError(
            "caller component SST disagrees with the sealed capture statistics"
        )

    capacity_sha: str | None = None
    bank_capacity: int | None = None
    receipts = tuple(capacity_receipts)
    if source.layout.condition in {"A", "R"}:
        if len(receipts) != source.process_count:
            raise ContributionEvaluationError(
                "A/R evaluation requires one capacity receipt per process"
            )
        for receipt in receipts:
            _validate_capacity_receipt(receipt, source.layout)
        if len({receipt.input_receipt_sha256 for receipt in receipts}) != len(
            receipts
        ):
            raise ContributionEvaluationError(
                "A/R capacity receipt set contains duplicate process inputs"
            )
        if source.receipt.receipt_sha256 not in {
            receipt.input_receipt_sha256 for receipt in receipts
        }:
            raise ContributionEvaluationError(
                "capacity receipts do not include this process's sealed input"
            )
        merged = merge_capacity_receipts(receipts)
        bank_capacity = merged.capacity
        capacity_sha = merged.receipt_sha256
    elif receipts:
        raise ContributionEvaluationError(
            "B/F evaluation must not receive A/R capacity receipts"
        )

    evaluator = ContributionDistributedEvaluator(
        mesh,
        source.layout,
        selected,
        bank_capacity=bank_capacity,
    )
    component_sse = 0.0
    computed_sst = 0.0
    token_count = 0
    component_count = 0
    l0_sum = 0
    fired = np.zeros((source.layout.total_features,), dtype=np.bool_)
    local_rows = 0
    updates = 0
    process_local = source.process_count > 1
    for update_index, microbatches in enumerate(source.iter_updates()):
        aux, chunk = evaluator.step(
            params,
            microbatches,
            mean,
            process_local=process_local,
        )
        host_aux = jax.device_get(aux)
        component_sse += float(host_aux.component_sse)
        computed_sst += float(host_aux.component_sst)
        token_count += int(host_aux.valid_source_tokens)
        component_count += int(host_aux.valid_components)
        l0_sum += int(host_aux.l0_sum)
        observed_fired = _addressable_feature_mask(aux.fired_mask)
        if observed_fired.shape != fired.shape:
            raise ContributionEvaluationError(
                "evaluation firing mask is not a complete feature vector"
            )
        fired |= observed_fired
        local_rows += chunk.tokens
        updates += 1
        if row_sink is not None:
            row_sink(_evidence_chunk(update_index, microbatches, chunk))

    if updates != source.planned_updates:
        raise ContributionEvaluationError(
            "evaluation did not consume its exact planned update count"
        )
    if token_count != source.total_tokens:
        raise ContributionEvaluationError(
            "evaluation token count differs from the sealed input prefix"
        )
    if component_count != token_count * source.layout.top_routed:
        raise ContributionEvaluationError(
            "evaluation component total differs from token_count × routes"
        )
    if l0_sum != token_count * source.layout.total_l0:
        raise ContributionEvaluationError(
            "evaluation total L0 differs from the frozen contract"
        )
    if not math.isfinite(computed_sst) or computed_sst <= 0:
        raise ContributionEvaluationError(
            "computed component SST must be finite and positive"
        )
    denominator = sealed_component_sst
    if not math.isfinite(denominator) or denominator <= 0:
        raise ContributionEvaluationError(
            "sealed component SST must be finite and positive"
        )
    if not math.isclose(
        computed_sst,
        denominator,
        rel_tol=5e-4,
        abs_tol=1e-6 * max(1.0, denominator),
    ):
        raise ContributionEvaluationError(
            "device-computed component SST disagrees with sealed FP64 statistics"
        )
    fired_fraction = float(np.mean(fired))
    mean32 = np.ascontiguousarray(np.asarray(mean, dtype="<f4"))
    mean_digest = hashlib.sha256()
    mean_digest.update(b"gemma4-contribution-centering-mean-v1\0")
    mean_digest.update(mean32.tobytes(order="C"))
    return ContributionEvaluationResult(
        schema_version=EVALUATION_SCHEMA_VERSION,
        condition=source.layout.condition,
        randomization_key=source.layout.randomization_key,
        capture_manifest_sha256=source.capture_manifest_sha256,
        input_receipt_sha256=source.receipt.receipt_sha256,
        checkpoint_sha256=checkpoint_sha256,
        centering_mean_sha256=mean_digest.hexdigest(),
        capacity_receipt_sha256=capacity_sha,
        tokens=token_count,
        valid_components=component_count,
        component_sse=component_sse,
        component_sst=denominator,
        computed_component_sst=computed_sst,
        l0_sum=l0_sum,
        l0_per_contribution=source.layout.k_per_contribution,
        total_l0=source.layout.total_l0,
        fired_feature_fraction=fired_fraction,
        dead_rate=1.0 - fired_fraction,
        updates=updates,
        local_sufficient_rows=local_rows,
    )


def routed_recomposition_token_scores(
    chunk: ContributionEvaluationChunk,
    recomposition_sidecar: Mapping[str, Any],
    routed_mean: Any,
) -> RoutedRecompositionScores:
    """Apply exact frozen Gemma routed RMSNorm to a sufficient-row chunk."""

    decoded = decode_recomposition_sidecar(recomposition_sidecar)
    hidden = chunk.raw_routed_target.shape[-1]
    if decoded.routed_rmsnorm_scale_bf16_bits.shape != (hidden,):
        raise ContributionEvaluationError(
            "recomposition sidecar hidden size differs from evaluation rows"
        )
    mean = np.asarray(routed_mean)
    if (
        mean.shape != (hidden,)
        or not np.issubdtype(mean.dtype, np.floating)
        or not np.isfinite(mean).all()
    ):
        raise ContributionEvaluationError(
            f"routed_mean must be finite floating-point [{hidden}]"
        )

    shape = chunk.raw_routed_target.shape
    flat_target = np.asarray(chunk.raw_routed_target, np.float32).reshape(-1, hidden)
    flat_estimate = np.asarray(
        chunk.raw_routed_reconstruction,
        np.float32,
    ).reshape(-1, hidden)
    target_bits = float32_to_bf16_bits(flat_target)
    estimate_bits = float32_to_bf16_bits(flat_estimate)
    _, target_post_bits = maxtext_rmsnorm_bf16(
        target_bits,
        decoded.routed_rmsnorm_scale_bf16_bits,
        epsilon=float(decoded.epsilon),
    )
    _, estimate_post_bits = maxtext_rmsnorm_bf16(
        estimate_bits,
        decoded.routed_rmsnorm_scale_bf16_bits,
        epsilon=float(decoded.epsilon),
    )
    target = bf16_bits_to_float32(target_post_bits)
    estimate = bf16_bits_to_float32(estimate_post_bits)
    mean32 = mean.astype(np.float32)
    error = target - estimate
    centered = target - mean32[None, :]
    token_sse = np.sum(
        np.square(error, dtype=np.float32),
        axis=1,
        dtype=np.float64,
    ).reshape(shape[:2])
    token_sst = np.sum(
        np.square(centered, dtype=np.float32),
        axis=1,
        dtype=np.float64,
    ).reshape(shape[:2])
    valid = np.asarray(chunk.valid_mask)
    token_sse = np.where(valid, token_sse, 0.0)
    token_sst = np.where(valid, token_sst, 0.0)
    return RoutedRecompositionScores(
        token_sse=token_sse,
        token_sst=token_sst,
        valid_mask=valid.copy(),
        recomposition_sidecar_sha256=decoded.sidecar_sha256,
    )


def oracle_evaluate_microbatch(
    params: SAEParams,
    microbatch: Any,
    mean: Any,
    layout: ContributionSAEConfig,
) -> ContributionEvaluationChunk:
    """Dense-gather single-device oracle for production evaluator parity."""

    target, banks, weights, valid = _validate_host_microbatch(
        microbatch,
        layout,
        expected_rows=None,
    )
    host_mean = np.asarray(mean)
    if (
        host_mean.shape != (layout.hidden_dim,)
        or not np.issubdtype(host_mean.dtype, np.floating)
        or not np.isfinite(host_mean).all()
    ):
        raise ValueError(f"mean must be finite floating-point [{layout.hidden_dim}]")
    for leaf in jax.tree_util.tree_leaves(params):
        if not np.isfinite(np.asarray(jax.device_get(leaf))).all():
            raise ContributionEvaluationError("oracle parameters must be finite")

    output = forward_contributions(
        params,
        jnp.asarray(target),
        jnp.asarray(banks),
        jnp.asarray(host_mean, jnp.float32),
        layout,
    )
    reconstruction = np.asarray(jax.device_get(output.reconstruction), np.float32)
    selection_values = np.asarray(jax.device_get(output.selection.values))
    selection_valid = np.asarray(jax.device_get(output.selection.valid))
    flat_valid = np.broadcast_to(valid[:, None], banks.shape).reshape(-1)
    selected = (
        (selection_values > 0)
        & selection_valid[:, None]
        & flat_valid[:, None]
    )
    per_l0 = np.sum(selected, axis=1, dtype=np.int32).reshape(banks.shape)
    if np.any(per_l0[valid] != layout.k_per_contribution):
        raise ContributionEvaluationError(
            "oracle found fewer than k positive features"
        )

    target32 = target.astype(np.float32)
    weights32 = weights.astype(np.float32)
    mean32 = host_mean.astype(np.float32)
    error = weights32[..., None] * (target32 - reconstruction)
    centered = weights32[..., None] * (
        target32 - mean32[None, None, :]
    )
    route_sse = np.sum(
        np.square(error, dtype=np.float32),
        axis=2,
        dtype=np.float32,
    )
    route_sst = np.sum(
        np.square(centered, dtype=np.float32),
        axis=2,
        dtype=np.float32,
    )
    route_sse = np.where(valid[:, None], route_sse, 0.0)
    route_sst = np.where(valid[:, None], route_sst, 0.0)
    token_sse = np.sum(route_sse, axis=1, dtype=np.float32)
    token_sst = np.sum(route_sst, axis=1, dtype=np.float32)
    raw_target = np.einsum(
        "nkh,nk->nh",
        target32,
        weights32,
        dtype=np.float32,
        optimize=False,
    )
    raw_estimate = np.einsum(
        "nkh,nk->nh",
        reconstruction,
        weights32,
        dtype=np.float32,
        optimize=False,
    )
    raw_target = np.where(valid[:, None], raw_target, 0.0)
    raw_estimate = np.where(valid[:, None], raw_estimate, 0.0)
    return ContributionEvaluationChunk(
        component_token_sse=token_sse[None, :],
        component_token_sst=token_sst[None, :],
        component_route_sse=route_sse[None, :],
        component_route_sst=route_sst[None, :],
        per_contribution_l0=per_l0[None, :],
        raw_routed_target=raw_target[None, :],
        raw_routed_reconstruction=raw_estimate[None, :],
        bank_ids=banks[None, :],
        router_weights=weights32[None, :],
        valid_mask=valid[None, :],
    )


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "ContributionDistributedEvaluator",
    "ContributionEvaluationAux",
    "ContributionEvaluationChunk",
    "ContributionEvaluationEvidenceChunk",
    "ContributionEvaluationError",
    "ContributionEvaluationResult",
    "ContributionEvaluationRows",
    "RoutedRecompositionScores",
    "evaluate_contribution_input",
    "oracle_evaluate_microbatch",
    "routed_recomposition_token_scores",
]
