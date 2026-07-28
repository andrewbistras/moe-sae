"""Balanced feature/data-sharded production trainer for Experiment 5.

The canonical Experiment-5 parameter tree orders all routed features before
all shared features.  Sharding that vector contiguously would place the entire
shared dictionary on one feature chip.  The production physical layout instead
gives every feature shard an equal number of whole routed banks followed by an
equal shard of the shared dictionary:

``[routed banks owned by shard | shared features owned by shard]``.

Sparse selection continues to use canonical logical latent IDs, so TopK tie
breaking and evidence are independent of physical packing.  Decoder gather and
firing accounting explicitly map those logical IDs back to local physical
columns.  The packing schema and feature-axis size form immutable checkpoint
identity: a checkpoint cannot resume on a different feature-axis size without
an explicit offline repack.

This module parallels :mod:`moe_sae.contribution_distributed`: data rows are
sharded across ``data``; parameters, Adam moments, and firing masks across the
host-local ``feature`` axis; only TopK candidates and decoded hidden vectors
cross the feature axis.  H/R use fixed-capacity grouped GEMMs for routed banks
plus a balanced shared-dictionary GEMM.  B/F use ordinary global
feature-sharded scoring.  Every condition differentiates the same exact
full-FFN normalized-SSE objective.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from moe_sae.contribution_distributed import _encoder_activation
from moe_sae.contribution_training import ContributionOptimizerConfig
from moe_sae.distributed import (
    DistributedConfig,
    _param_specs,
    _validate_exact_distributed_l0,
    init_distributed_state,
    replicate_process_local_array,
    shard_train_state,
    state_shardings,
)
from moe_sae.experiment5_hybrid import (
    Experiment5Batch,
    Experiment5Centering,
    Experiment5EncoderKind,
    Experiment5Error,
    Experiment5Layout,
    Experiment5Recomposition,
    experiment5_optimizer_sae_config,
    jax_recompose_experiment5_full_ffn,
    validate_experiment5_microbatch,
)
from moe_sae.sae import (
    AdamState,
    SAEParams,
    Selection,
    StepMetrics,
    TrainState,
    apply_gradients,
    local_topk_candidates,
    merge_feature_shard_topk,
)

EXPERIMENT5_PHYSICAL_LAYOUT_SCHEMA_VERSION = (
    "gemma4-experiment5-balanced-feature-packing-v1"
)
_EPS = 1e-12
_SENTINEL_ID = np.iinfo(np.int32).max


@dataclass(frozen=True, slots=True)
class Experiment5PhysicalLayout:
    """Immutable logical-to-physical feature packing for one mesh shape."""

    condition: str
    total_features: int
    feature_axis_size: int
    routed_features: int
    routed_banks_per_shard: int
    routed_features_per_shard: int
    shared_features_per_shard: int
    physical_to_logical_sha256: str
    schema_version: str = EXPERIMENT5_PHYSICAL_LAYOUT_SCHEMA_VERSION

    @property
    def local_features(self) -> int:
        return self.total_features // self.feature_axis_size

    @property
    def identity_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(include_identity=False),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(
            b"gemma4-experiment5-physical-layout-identity-v1\0" + payload
        ).hexdigest()

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "condition": self.condition,
            "total_features": self.total_features,
            "feature_axis_size": self.feature_axis_size,
            "local_features": self.local_features,
            "routed_features": self.routed_features,
            "routed_banks_per_shard": self.routed_banks_per_shard,
            "routed_features_per_shard": self.routed_features_per_shard,
            "shared_features_per_shard": self.shared_features_per_shard,
            "physical_to_logical_sha256": self.physical_to_logical_sha256,
        }
        if include_identity:
            result["identity_sha256"] = self.identity_sha256
        return result


def _permutation_sha256(permutation: np.ndarray) -> str:
    canonical = np.ascontiguousarray(permutation, dtype="<i4")
    return hashlib.sha256(
        b"gemma4-experiment5-physical-to-logical-v1\0"
        + canonical.tobytes(order="C")
    ).hexdigest()


def experiment5_physical_to_logical(
    layout: Experiment5Layout,
    feature_axis_size: int,
) -> np.ndarray:
    """Return canonical logical latent ID for each physical feature column."""

    if (
        isinstance(feature_axis_size, bool)
        or not isinstance(feature_axis_size, int)
        or feature_axis_size <= 0
    ):
        raise ValueError("feature_axis_size must be a positive integer")
    if layout.total_features % feature_axis_size:
        raise ValueError("Experiment-5 width must divide across feature shards")
    if not layout.is_hybrid:
        return np.arange(layout.total_features, dtype=np.int32)
    if layout.num_experts % feature_axis_size:
        raise ValueError("H/R experts must divide across feature shards")
    if layout.shared_width % feature_axis_size:
        raise ValueError("H/R shared width must divide across feature shards")
    banks_per_shard = layout.num_experts // feature_axis_size
    routed_per_shard = banks_per_shard * layout.routed_bank_width
    shared_per_shard = layout.shared_width // feature_axis_size
    pieces: list[np.ndarray] = []
    for shard in range(feature_axis_size):
        routed_start = shard * routed_per_shard
        shared_start = layout.routed_features + shard * shared_per_shard
        pieces.extend(
            (
                np.arange(
                    routed_start,
                    routed_start + routed_per_shard,
                    dtype=np.int32,
                ),
                np.arange(
                    shared_start,
                    shared_start + shared_per_shard,
                    dtype=np.int32,
                ),
            )
        )
    result = np.concatenate(pieces)
    if (
        result.shape != (layout.total_features,)
        or not np.array_equal(
            np.sort(result.astype(np.int64)),
            np.arange(layout.total_features, dtype=np.int64),
        )
    ):
        raise ValueError("Experiment-5 physical packing is not a bijection")
    return result


def experiment5_physical_layout(
    layout: Experiment5Layout,
    feature_axis_size: int,
) -> Experiment5PhysicalLayout:
    permutation = experiment5_physical_to_logical(layout, feature_axis_size)
    if layout.is_hybrid:
        banks = layout.num_experts // feature_axis_size
        routed = banks * layout.routed_bank_width
        shared = layout.shared_width // feature_axis_size
    else:
        banks = 0
        routed = 0
        shared = 0
    return Experiment5PhysicalLayout(
        condition=layout.condition,
        total_features=layout.total_features,
        feature_axis_size=feature_axis_size,
        routed_features=layout.routed_features if layout.is_hybrid else 0,
        routed_banks_per_shard=banks,
        routed_features_per_shard=routed,
        shared_features_per_shard=shared,
        physical_to_logical_sha256=_permutation_sha256(permutation),
    )


def require_experiment5_physical_layout(
    expected_identity_sha256: str,
    physical: Experiment5PhysicalLayout,
) -> None:
    """Fail a resume whose mesh-bound physical packing identity changed."""

    if (
        not isinstance(expected_identity_sha256, str)
        or len(expected_identity_sha256) != 64
    ):
        raise ValueError("expected physical-layout identity must be SHA-256")
    if expected_identity_sha256 != physical.identity_sha256:
        raise RuntimeError(
            "Experiment-5 checkpoint physical layout differs from this mesh; "
            "resume requires an explicit offline repack"
        )


def _permute_params(params: SAEParams, permutation: jax.Array) -> SAEParams:
    return SAEParams(
        encoder=jnp.take(params.encoder, permutation, axis=0),
        decoder=jnp.take(params.decoder, permutation, axis=1),
        bias=jnp.take(params.bias, permutation, axis=0),
    )


def pack_experiment5_params(
    params: SAEParams,
    layout: Experiment5Layout,
    feature_axis_size: int,
) -> SAEParams:
    """Convert canonical logical columns to the balanced physical ordering."""

    permutation = jnp.asarray(
        experiment5_physical_to_logical(layout, feature_axis_size),
        jnp.int32,
    )
    return _permute_params(params, permutation)


def unpack_experiment5_params(
    params: SAEParams,
    layout: Experiment5Layout,
    feature_axis_size: int,
) -> SAEParams:
    """Convert a gathered physical parameter tree back to canonical ordering."""

    physical_to_logical = experiment5_physical_to_logical(
        layout,
        feature_axis_size,
    )
    logical_to_physical = jnp.asarray(
        np.argsort(physical_to_logical, kind="stable"),
        jnp.int32,
    )
    return _permute_params(params, logical_to_physical)


def pack_experiment5_train_state(
    state: TrainState,
    layout: Experiment5Layout,
    feature_axis_size: int,
) -> TrainState:
    """Pack params, moments, and firing mask consistently for accelerator use."""

    permutation = jnp.asarray(
        experiment5_physical_to_logical(layout, feature_axis_size),
        jnp.int32,
    )
    return TrainState(
        params=_permute_params(state.params, permutation),
        optimizer=AdamState(
            m=_permute_params(state.optimizer.m, permutation),
            v=_permute_params(state.optimizer.v, permutation),
            step=state.optimizer.step,
        ),
        step=state.step,
        tokens_seen=state.tokens_seen,
        fired_mask=jnp.take(state.fired_mask, permutation, axis=0),
        loss_ema=state.loss_ema,
        initial_loss_ema=state.initial_loss_ema,
        ema_updates=state.ema_updates,
    )


def unpack_experiment5_train_state(
    state: TrainState,
    layout: Experiment5Layout,
    feature_axis_size: int,
) -> TrainState:
    """Gathered/offline inverse of :func:`pack_experiment5_train_state`."""

    physical_to_logical = experiment5_physical_to_logical(
        layout,
        feature_axis_size,
    )
    inverse = jnp.asarray(
        np.argsort(physical_to_logical, kind="stable"),
        jnp.int32,
    )
    return TrainState(
        params=_permute_params(state.params, inverse),
        optimizer=AdamState(
            m=_permute_params(state.optimizer.m, inverse),
            v=_permute_params(state.optimizer.v, inverse),
            step=state.optimizer.step,
        ),
        step=state.step,
        tokens_seen=state.tokens_seen,
        fired_mask=jnp.take(state.fired_mask, inverse, axis=0),
        loss_ema=state.loss_ema,
        initial_loss_ema=state.initial_loss_ema,
        ema_updates=state.ema_updates,
    )


class Experiment5DistributedBatch(NamedTuple):
    routed_input: jax.Array
    expert_output: jax.Array
    shared_normalized: jax.Array
    bank_ids: jax.Array
    router_weights: jax.Array
    valid_mask: jax.Array


class Experiment5DistributedConstants(NamedTuple):
    source_mean: jax.Array
    routed_target_mean: jax.Array
    shared_target_mean: jax.Array
    full_target_mean: jax.Array
    routed_scale: jax.Array
    outer_scale: jax.Array


class Experiment5DistributedLossAux(NamedTuple):
    sse: jax.Array
    sst: jax.Array
    l0_sum: jax.Array
    valid_tokens: jax.Array
    topk_valid: jax.Array
    fired_mask: jax.Array
    capacity_overflow: jax.Array


def experiment5_batch_shardings(
    mesh: Mesh,
    distributed: DistributedConfig,
) -> Experiment5DistributedBatch:
    data = distributed.data_axis
    return Experiment5DistributedBatch(
        NamedSharding(mesh, P(None, data, None)),
        NamedSharding(mesh, P(None, data, None, None)),
        NamedSharding(mesh, P(None, data, None)),
        NamedSharding(mesh, P(None, data, None)),
        NamedSharding(mesh, P(None, data, None)),
        NamedSharding(mesh, P(None, data)),
    )


def _validate_experiment5_mesh(
    mesh: Mesh,
    layout: Experiment5Layout,
    distributed: DistributedConfig,
) -> Experiment5PhysicalLayout:
    shape = mesh.shape
    if distributed.data_axis not in shape or distributed.feature_axis not in shape:
        raise ValueError("mesh does not contain configured data/feature axes")
    if set(mesh.axis_names) != {
        distributed.data_axis,
        distributed.feature_axis,
    }:
        raise ValueError("Experiment-5 training requires an exact two-axis mesh")
    feature_size = int(shape[distributed.feature_axis])
    physical = experiment5_physical_layout(layout, feature_size)
    if layout.is_hybrid:
        if physical.routed_features_per_shard < layout.routed_k:
            raise ValueError("each shard must own enough routed candidates")
        if physical.shared_features_per_shard < layout.shared_k:
            raise ValueError("each shard must own enough shared candidates")
    elif physical.local_features < layout.total_l0:
        raise ValueError("each global feature shard must contain TopK candidates")
    devices = np.asarray(mesh.devices, dtype=object)
    data_dimension = mesh.axis_names.index(distributed.data_axis)
    feature_dimension = mesh.axis_names.index(distributed.feature_axis)
    rows = np.moveaxis(
        devices,
        (data_dimension, feature_dimension),
        (0, 1),
    )
    for row in rows:
        if len({int(device.process_index) for device in row}) != 1:
            raise ValueError("Experiment-5 feature axis must remain host-local")
    return physical


def stack_experiment5_distributed_microbatches(
    microbatches: Sequence[Any],
    layout: Experiment5Layout,
    distributed: DistributedConfig | None = None,
) -> Experiment5DistributedBatch:
    """Validate and stack exact static host microbatches."""

    selected = distributed or DistributedConfig()
    if len(microbatches) != selected.microbatches_per_update:
        raise ValueError(
            f"expected {selected.microbatches_per_update} microbatches, "
            f"got {len(microbatches)}"
        )
    checked: list[Experiment5Batch] = []
    rows: int | None = None
    for item in microbatches:
        batch = validate_experiment5_microbatch(
            item,
            layout,
            expected_rows=rows,
        )
        rows = int(np.asarray(batch.routed_input).shape[0])
        checked.append(batch)
    fields = []
    for name in Experiment5Batch._fields:
        fields.append(
            np.stack([np.asarray(getattr(batch, name)) for batch in checked])
        )
    return Experiment5DistributedBatch(*fields)


def _validate_stacked_batch(
    batch: Experiment5DistributedBatch,
    layout: Experiment5Layout,
    distributed: DistributedConfig,
) -> None:
    source = batch.routed_input
    if (
        source.ndim != 3
        or source.shape[0] != distributed.microbatches_per_update
        or source.shape[2] != layout.hidden_dim
    ):
        raise ValueError("stacked routed_input shape is incompatible")
    expected_expert = (
        source.shape[0],
        source.shape[1],
        layout.top_routed,
        layout.hidden_dim,
    )
    if batch.expert_output.shape != expected_expert:
        raise ValueError("stacked expert_output shape is incompatible")
    if batch.shared_normalized.shape != source.shape:
        raise ValueError("stacked shared_normalized shape is incompatible")
    if batch.bank_ids.shape != expected_expert[:3]:
        raise ValueError("stacked bank_ids shape is incompatible")
    if batch.router_weights.shape != expected_expert[:3]:
        raise ValueError("stacked router_weights shape is incompatible")
    if batch.valid_mask.shape != source.shape[:2]:
        raise ValueError("stacked valid_mask shape is incompatible")
    for name in (
        "routed_input",
        "expert_output",
        "shared_normalized",
        "router_weights",
    ):
        if not jnp.issubdtype(getattr(batch, name).dtype, jnp.floating):
            raise TypeError(f"stacked {name} must be floating-point")
    if not jnp.issubdtype(batch.bank_ids.dtype, jnp.integer):
        raise TypeError("stacked bank_ids must be integer")
    if batch.valid_mask.dtype != jnp.bool_:
        raise TypeError("stacked valid_mask must be boolean")


def shard_experiment5_batch(
    batch: Experiment5DistributedBatch,
    mesh: Mesh,
    distributed: DistributedConfig,
) -> Experiment5DistributedBatch:
    data_size = int(mesh.shape[distributed.data_axis])
    if batch.routed_input.shape[1] % data_size:
        raise ValueError("global microbatch must divide across the data axis")
    return jax.tree_util.tree_map(
        jax.device_put,
        batch,
        experiment5_batch_shardings(mesh, distributed),
    )


def shard_process_local_experiment5_batch(
    batch: Experiment5DistributedBatch,
    mesh: Mesh,
    distributed: DistributedConfig,
) -> Experiment5DistributedBatch:
    """Assemble globally data-sharded arrays from one host lane per process."""

    data_size = int(mesh.shape[distributed.data_axis])
    if data_size != jax.process_count():
        raise ValueError(
            "process-local Experiment-5 assembly requires one data row per process"
        )
    shardings = experiment5_batch_shardings(mesh, distributed)
    local_rows = batch.routed_input.shape[1]

    def place(
        value: Any,
        sharding: NamedSharding,
        global_shape: tuple[int, ...],
    ) -> jax.Array:
        return jax.make_array_from_process_local_data(
            sharding,
            np.asarray(value),
            global_shape=global_shape,
        )

    micro = batch.routed_input.shape[0]
    global_rows = local_rows * data_size
    return Experiment5DistributedBatch(
        place(
            batch.routed_input,
            shardings.routed_input,
            (micro, global_rows, batch.routed_input.shape[2]),
        ),
        place(
            batch.expert_output,
            shardings.expert_output,
            (
                micro,
                global_rows,
                batch.expert_output.shape[2],
                batch.expert_output.shape[3],
            ),
        ),
        place(
            batch.shared_normalized,
            shardings.shared_normalized,
            (micro, global_rows, batch.shared_normalized.shape[2]),
        ),
        place(
            batch.bank_ids,
            shardings.bank_ids,
            (micro, global_rows, batch.bank_ids.shape[2]),
        ),
        place(
            batch.router_weights,
            shardings.router_weights,
            (micro, global_rows, batch.router_weights.shape[2]),
        ),
        place(
            batch.valid_mask,
            shardings.valid_mask,
            (micro, global_rows),
        ),
    )


def init_experiment5_distributed_state(
    key: jax.Array,
    mesh: Mesh,
    layout: Experiment5Layout,
    optimizer: ContributionOptimizerConfig | None = None,
    distributed: DistributedConfig | None = None,
) -> TrainState:
    """Initialize directly into the mesh-bound balanced physical layout."""

    selected = distributed or DistributedConfig()
    _validate_experiment5_mesh(mesh, layout, selected)
    return init_distributed_state(
        key,
        mesh,
        experiment5_optimizer_sae_config(layout, optimizer),
        selected,
    )


def shard_experiment5_train_state(
    logical_state: TrainState,
    mesh: Mesh,
    layout: Experiment5Layout,
    optimizer: ContributionOptimizerConfig | None = None,
    distributed: DistributedConfig | None = None,
) -> TrainState:
    """Pack a canonical oracle state and place it on the production mesh."""

    selected = distributed or DistributedConfig()
    physical = _validate_experiment5_mesh(mesh, layout, selected)
    packed = pack_experiment5_train_state(
        logical_state,
        layout,
        physical.feature_axis_size,
    )
    return shard_train_state(
        packed,
        mesh,
        experiment5_optimizer_sae_config(layout, optimizer),
        selected,
    )


def _merge_local_candidates(
    values: jax.Array,
    logical_ids: jax.Array,
    k: int,
    *,
    feature_axis: str,
) -> Selection:
    gathered_values = jax.lax.all_gather(
        values,
        feature_axis,
        axis=0,
        tiled=False,
    )
    gathered_ids = jax.lax.all_gather(
        logical_ids,
        feature_axis,
        axis=0,
        tiled=False,
    )
    return merge_feature_shard_topk(gathered_values, gathered_ids, k)


def _local_logical_ranges(
    layout: Experiment5Layout,
    physical: Experiment5PhysicalLayout,
    *,
    feature_axis: str,
) -> tuple[jax.Array, jax.Array]:
    shard = jax.lax.axis_index(feature_axis)
    routed_start = shard * physical.routed_features_per_shard
    shared_start = (
        layout.routed_features
        + shard * physical.shared_features_per_shard
    )
    return routed_start, shared_start


def _logical_to_local_physical(
    logical_ids: jax.Array,
    layout: Experiment5Layout,
    physical: Experiment5PhysicalLayout,
    *,
    feature_axis: str,
) -> tuple[jax.Array, jax.Array]:
    if not layout.is_hybrid:
        local_width = physical.local_features
        start = jax.lax.axis_index(feature_axis) * local_width
        owned = (logical_ids >= start) & (logical_ids < start + local_width)
        return jnp.clip(logical_ids - start, 0, local_width - 1), owned
    routed_start, shared_start = _local_logical_ranges(
        layout,
        physical,
        feature_axis=feature_axis,
    )
    routed_owned = (logical_ids >= routed_start) & (
        logical_ids < routed_start + physical.routed_features_per_shard
    )
    shared_owned = (logical_ids >= shared_start) & (
        logical_ids < shared_start + physical.shared_features_per_shard
    )
    routed_local = logical_ids - routed_start
    shared_local = (
        physical.routed_features_per_shard
        + logical_ids
        - shared_start
    )
    local = jnp.where(routed_owned, routed_local, shared_local)
    return (
        jnp.clip(local, 0, physical.local_features - 1),
        routed_owned | shared_owned,
    )


def _decode_logical_selection(
    local_decoder: jax.Array,
    selection: Selection,
    mean: jax.Array,
    layout: Experiment5Layout,
    physical: Experiment5PhysicalLayout,
    *,
    feature_axis: str,
) -> jax.Array:
    local_ids, owned = _logical_to_local_physical(
        selection.latent_ids,
        layout,
        physical,
        feature_axis=feature_axis,
    )
    columns = local_decoder[:, local_ids]
    values = jnp.where(owned, selection.values, 0.0).astype(
        local_decoder.dtype
    )
    partial = jnp.einsum(
        "nk,hnk->nh",
        values,
        columns,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    )
    centered = jax.lax.psum(partial, feature_axis)
    return centered + jnp.asarray(mean, jnp.float32)[None, :]


def _local_grouped_routed_candidates(
    params: SAEParams,
    centered: jax.Array,
    bank_ids: jax.Array,
    component_valid: jax.Array,
    layout: Experiment5Layout,
    physical: Experiment5PhysicalLayout,
    *,
    feature_axis: str,
    capacity: int,
    encoder_activation: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Local per-route TopK after balanced whole-bank grouped GEMMs."""

    local_banks = physical.routed_banks_per_shard
    local_routed_features = physical.routed_features_per_shard
    bank_start = jax.lax.axis_index(feature_axis) * local_banks
    flat_source = centered.reshape(-1, layout.hidden_dim)
    flat_banks = bank_ids.astype(jnp.int32).reshape(-1)
    flat_valid = component_valid.reshape(-1)
    local_bank = flat_banks - bank_start
    owned = flat_valid & (local_bank >= 0) & (local_bank < local_banks)
    safe_bank = jnp.clip(local_bank, 0, local_banks - 1)

    membership = (
        jax.nn.one_hot(safe_bank, local_banks, dtype=jnp.int32)
        * owned[:, None].astype(jnp.int32)
    )
    cumulative = jnp.cumsum(membership, axis=0) - 1
    rank = jnp.sum(cumulative * membership, axis=1)
    overflow = jnp.any(owned & (rank >= capacity))
    in_capacity = owned & (rank < capacity)
    destination = (
        safe_bank * capacity + jnp.clip(rank, 0, capacity - 1)
    )
    padded = (
        jnp.zeros(
            (local_banks * capacity, layout.hidden_dim),
            centered.dtype,
        )
        .at[destination]
        .add(jnp.where(in_capacity[:, None], flat_source, 0))
    )
    occupancy = (
        jnp.zeros((local_banks * capacity,), jnp.int32)
        .at[destination]
        .max(in_capacity.astype(jnp.int32))
        > 0
    )

    encoder = params.encoder[:local_routed_features].reshape(
        local_banks,
        layout.routed_bank_width,
        layout.hidden_dim,
    )
    bias = params.bias[:local_routed_features].reshape(
        local_banks,
        layout.routed_bank_width,
    )
    grouped = jnp.einsum(
        "bcd,bwd->bcw",
        padded.reshape(local_banks, capacity, layout.hidden_dim),
        encoder.astype(centered.dtype),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    ) + bias[:, None, :].astype(jnp.float32)
    grouped = _encoder_activation(
        grouped,
        kind=encoder_activation,
    ).reshape(local_banks * capacity, layout.routed_bank_width)
    grouped = jnp.where(occupancy[:, None], grouped, -jnp.inf)
    global_banks = bank_start + jnp.arange(local_banks, dtype=jnp.int32)
    candidate_ids = (
        global_banks[:, None, None] * layout.routed_bank_width
        + jnp.arange(layout.routed_bank_width, dtype=jnp.int32)[None, None, :]
    )
    candidate_ids = jnp.broadcast_to(
        candidate_ids,
        (local_banks, capacity, layout.routed_bank_width),
    ).reshape(local_banks * capacity, layout.routed_bank_width)
    local = local_topk_candidates(
        grouped,
        candidate_ids,
        layout.routed_k,
        ids_are_sorted=True,
    )
    gathered_values = jnp.where(
        in_capacity[:, None],
        local.values[destination],
        0.0,
    )
    gathered_ids = jnp.where(
        in_capacity[:, None],
        local.latent_ids[destination],
        jnp.asarray(_SENTINEL_ID, jnp.int32),
    )
    return gathered_values, gathered_ids, overflow


def _local_shared_candidates(
    params: SAEParams,
    centered: jax.Array,
    valid_mask: jax.Array,
    layout: Experiment5Layout,
    physical: Experiment5PhysicalLayout,
    *,
    feature_axis: str,
    encoder_activation: str,
) -> tuple[jax.Array, jax.Array]:
    local_start = physical.routed_features_per_shard
    encoder = params.encoder[local_start:]
    bias = params.bias[local_start:]
    pre = jnp.einsum(
        "nd,fd->nf",
        centered,
        encoder.astype(centered.dtype),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    ) + bias.astype(jnp.float32)
    scores = jnp.where(
        valid_mask[:, None],
        _encoder_activation(pre, kind=encoder_activation),
        -jnp.inf,
    )
    _, logical_start = _local_logical_ranges(
        layout,
        physical,
        feature_axis=feature_axis,
    )
    logical_ids = logical_start + jnp.arange(
        physical.shared_features_per_shard,
        dtype=jnp.int32,
    )
    selected = local_topk_candidates(
        scores,
        logical_ids,
        layout.shared_k,
        ids_are_sorted=True,
    )
    return selected.values, selected.latent_ids


def _local_global_candidates(
    params: SAEParams,
    centered: jax.Array,
    valid_mask: jax.Array,
    layout: Experiment5Layout,
    physical: Experiment5PhysicalLayout,
    *,
    feature_axis: str,
    encoder_activation: str,
) -> tuple[jax.Array, jax.Array]:
    pre = jnp.einsum(
        "nd,fd->nf",
        centered,
        params.encoder.astype(centered.dtype),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    ) + params.bias.astype(jnp.float32)
    scores = jnp.where(
        valid_mask[:, None],
        _encoder_activation(pre, kind=encoder_activation),
        -jnp.inf,
    )
    logical_start = (
        jax.lax.axis_index(feature_axis) * physical.local_features
    )
    logical_ids = logical_start + jnp.arange(
        physical.local_features,
        dtype=jnp.int32,
    )
    selected = local_topk_candidates(
        scores,
        logical_ids,
        layout.total_l0,
        ids_are_sorted=True,
    )
    return selected.values, selected.latent_ids


def _local_fired_mask(
    selection: Selection,
    scientific_selection: jax.Array,
    layout: Experiment5Layout,
    physical: Experiment5PhysicalLayout,
    *,
    feature_axis: str,
) -> jax.Array:
    local_ids, owned = _logical_to_local_physical(
        selection.latent_ids,
        layout,
        physical,
        feature_axis=feature_axis,
    )
    return (
        jnp.zeros((physical.local_features,), jnp.int32)
        .at[local_ids.reshape(-1)]
        .max(
            (scientific_selection & owned)
            .reshape(-1)
            .astype(jnp.int32)
        )
        > 0
    )


def _scientific_inputs_valid(
    routed_input: jax.Array,
    expert_output: jax.Array,
    shared_normalized: jax.Array,
    bank_ids: jax.Array,
    router_weights: jax.Array,
    valid_mask: jax.Array,
    layout: Experiment5Layout,
) -> jax.Array:
    scientific = valid_mask[:, None]
    bank_valid = jnp.asarray(True)
    if layout.is_hybrid:
        in_range = (bank_ids >= 0) & (bank_ids < layout.num_experts)
        ordered = jnp.sort(bank_ids, axis=1)
        unique = jnp.all(
            (jnp.diff(ordered, axis=1) != 0) | ~valid_mask[:, None]
        )
        bank_valid = jnp.all(in_range | ~scientific) & unique
    finite = (
        jnp.all(jnp.isfinite(routed_input) | ~valid_mask[:, None])
        & jnp.all(
            jnp.isfinite(expert_output) | ~valid_mask[:, None, None]
        )
        & jnp.all(
            jnp.isfinite(shared_normalized) | ~valid_mask[:, None]
        )
        & jnp.all(jnp.isfinite(router_weights) | ~scientific)
    )
    router_valid = (
        jnp.all((router_weights >= 0) | ~scientific)
        & jnp.all(
            (
                jnp.abs(
                    jnp.sum(
                        router_weights,
                        axis=1,
                        dtype=jnp.float32,
                    )
                    - 1.0
                )
                <= 0.005
            )
            | ~valid_mask
        )
    )
    return bank_valid & finite & router_valid


def _build_experiment5_mapped_objective(
    mesh: Mesh,
    layout: Experiment5Layout,
    physical: Experiment5PhysicalLayout,
    distributed: DistributedConfig,
    bank_capacity: int | None,
    parameter_dtype: str,
    encoder_kind: Experiment5EncoderKind,
    epsilon: float,
    rematerialize_microbatch: bool,
):
    data_axis = distributed.data_axis
    feature_axis = distributed.feature_axis
    if layout.is_hybrid:
        if bank_capacity is None or bank_capacity <= 0:
            raise ValueError(
                "distributed H/R training requires a positive routed-bank capacity"
            )
    elif bank_capacity is not None:
        raise ValueError("B/F global training does not accept a bank capacity")
    capacity = int(bank_capacity or 1)
    compute_dtype = (
        jnp.bfloat16 if parameter_dtype == "bfloat16" else jnp.float32
    )
    activation = "relu" if encoder_kind == "sae" else "softplus"

    def local_objective(
        differentiable_params: SAEParams,
        routed_input: jax.Array,
        expert_output: jax.Array,
        shared_normalized: jax.Array,
        bank_ids: jax.Array,
        router_weights: jax.Array,
        valid_mask: jax.Array,
        source_mean: jax.Array,
        routed_target_mean: jax.Array,
        shared_target_mean: jax.Array,
        full_target_mean: jax.Array,
        routed_scale: jax.Array,
        outer_scale: jax.Array,
    ) -> tuple[jax.Array, Experiment5DistributedLossAux]:
        initial = (
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(True),
            jnp.zeros((physical.local_features,), jnp.bool_),
            jnp.asarray(False),
        )
        recomposition = Experiment5Recomposition(
            routed_scale=routed_scale,
            outer_scale=outer_scale,
            epsilon=epsilon,
        )

        def one_micro(carry, inputs):
            (
                sse,
                sst,
                l0_sum,
                token_count,
                valid_topk,
                fired,
                overflow,
            ) = carry
            source, expert, shared, banks, weights, valid = inputs
            params = jax.tree_util.tree_map(
                lambda value: value.astype(compute_dtype),
                differentiable_params,
            )
            target = jax_recompose_experiment5_full_ffn(
                expert,
                shared,
                weights,
                recomposition,
            )
            if layout.is_hybrid:
                if encoder_kind == "sae":
                    routed_centered = (
                        expert.astype(compute_dtype)
                        - routed_target_mean.astype(compute_dtype)[
                            None, None, :
                        ]
                    )
                    shared_centered = (
                        shared.astype(compute_dtype)
                        - shared_target_mean.astype(compute_dtype)[None, :]
                    )
                else:
                    common_centered = (
                        source.astype(compute_dtype)
                        - source_mean.astype(compute_dtype)[None, :]
                    )
                    routed_centered = jnp.broadcast_to(
                        common_centered[:, None, :],
                        expert.shape,
                    )
                    shared_centered = common_centered
                component_valid = jnp.broadcast_to(valid[:, None], banks.shape)
                local_routed_values, local_routed_ids, overflow_here = (
                    _local_grouped_routed_candidates(
                        params,
                        routed_centered,
                        banks,
                        component_valid,
                        layout,
                        physical,
                        feature_axis=feature_axis,
                        capacity=capacity,
                        encoder_activation=activation,
                    )
                )
                routed_selection = _merge_local_candidates(
                    local_routed_values,
                    local_routed_ids,
                    layout.routed_k,
                    feature_axis=feature_axis,
                )
                local_shared_values, local_shared_ids = (
                    _local_shared_candidates(
                        params,
                        shared_centered,
                        valid,
                        layout,
                        physical,
                        feature_axis=feature_axis,
                        encoder_activation=activation,
                    )
                )
                shared_selection = _merge_local_candidates(
                    local_shared_values,
                    local_shared_ids,
                    layout.shared_k,
                    feature_axis=feature_axis,
                )
                routed_reconstruction = _decode_logical_selection(
                    params.decoder,
                    routed_selection,
                    routed_target_mean,
                    layout,
                    physical,
                    feature_axis=feature_axis,
                ).reshape(expert.shape)
                shared_reconstruction = _decode_logical_selection(
                    params.decoder,
                    shared_selection,
                    shared_target_mean,
                    layout,
                    physical,
                    feature_axis=feature_axis,
                )
                reconstruction = jax_recompose_experiment5_full_ffn(
                    routed_reconstruction,
                    shared_reconstruction,
                    weights,
                    recomposition,
                )
                flat_component_valid = component_valid.reshape(-1)
                routed_scientific = (
                    (routed_selection.values > 0)
                    & routed_selection.valid[:, None]
                    & flat_component_valid[:, None]
                )
                shared_scientific = (
                    (shared_selection.values > 0)
                    & shared_selection.valid[:, None]
                    & valid[:, None]
                )
                fired_here = (
                    _local_fired_mask(
                        routed_selection,
                        routed_scientific,
                        layout,
                        physical,
                        feature_axis=feature_axis,
                    )
                    | _local_fired_mask(
                        shared_selection,
                        shared_scientific,
                        layout,
                        physical,
                        feature_axis=feature_axis,
                    )
                )
                l0_here = (
                    jnp.sum(routed_scientific, dtype=jnp.int32)
                    + jnp.sum(shared_scientific, dtype=jnp.int32)
                )
                topk_here = (
                    jnp.all(
                        routed_selection.valid | ~flat_component_valid
                    )
                    & jnp.all(shared_selection.valid | ~valid)
                )
            else:
                if encoder_kind == "sae":
                    global_source = target
                    global_mean = full_target_mean
                else:
                    global_source = source
                    global_mean = source_mean
                centered = (
                    global_source.astype(compute_dtype)
                    - global_mean.astype(compute_dtype)[None, :]
                )
                local_values, local_ids = _local_global_candidates(
                    params,
                    centered,
                    valid,
                    layout,
                    physical,
                    feature_axis=feature_axis,
                    encoder_activation=activation,
                )
                selection = _merge_local_candidates(
                    local_values,
                    local_ids,
                    layout.total_l0,
                    feature_axis=feature_axis,
                )
                reconstruction = _decode_logical_selection(
                    params.decoder,
                    selection,
                    full_target_mean,
                    layout,
                    physical,
                    feature_axis=feature_axis,
                )
                scientific = (
                    (selection.values > 0)
                    & selection.valid[:, None]
                    & valid[:, None]
                )
                fired_here = _local_fired_mask(
                    selection,
                    scientific,
                    layout,
                    physical,
                    feature_axis=feature_axis,
                )
                l0_here = jnp.sum(scientific, dtype=jnp.int32)
                topk_here = jnp.all(selection.valid | ~valid)
                overflow_here = jnp.asarray(False)

            target32 = target.astype(jnp.float32)
            reconstruction32 = reconstruction.astype(jnp.float32)
            full_mean32 = full_target_mean.astype(jnp.float32)
            valid32 = valid.astype(jnp.float32)
            sse_here = jnp.sum(
                jnp.sum(
                    jax.lax.square(target32 - reconstruction32),
                    axis=1,
                )
                * valid32,
                dtype=jnp.float32,
            )
            sst_here = jnp.sum(
                jnp.sum(
                    jax.lax.square(target32 - full_mean32[None, :]),
                    axis=1,
                )
                * valid32,
                dtype=jnp.float32,
            )
            inputs_valid = _scientific_inputs_valid(
                source,
                expert,
                shared,
                banks,
                weights,
                valid,
                layout,
            )
            constants_finite = (
                jnp.all(jnp.isfinite(source_mean))
                & jnp.all(jnp.isfinite(routed_target_mean))
                & jnp.all(jnp.isfinite(shared_target_mean))
                & jnp.all(jnp.isfinite(full_target_mean))
                & jnp.all(jnp.isfinite(routed_scale))
                & jnp.all(jnp.isfinite(outer_scale))
            )
            return (
                sse + sse_here,
                sst + sst_here,
                l0_sum + l0_here,
                token_count + jnp.sum(valid, dtype=jnp.int32),
                valid_topk
                & topk_here
                & inputs_valid
                & constants_finite,
                fired | fired_here,
                overflow | overflow_here,
            ), None

        microbatch_function = (
            jax.checkpoint(one_micro)
            if rematerialize_microbatch
            else one_micro
        )
        scanned, _ = jax.lax.scan(
            microbatch_function,
            initial,
            (
                routed_input,
                expert_output,
                shared_normalized,
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
            topk_valid,
            fired,
            overflow,
        ) = scanned
        global_sse = jax.lax.pmean(
            jax.lax.psum(local_sse, data_axis),
            feature_axis,
        )
        global_sst = jax.lax.pmean(
            jax.lax.psum(local_sst, data_axis),
            feature_axis,
        )
        global_l0 = jax.lax.pmean(
            jax.lax.psum(local_l0.astype(jnp.float32), data_axis),
            feature_axis,
        ).astype(jnp.int32)
        global_tokens = jax.lax.pmean(
            jax.lax.psum(local_tokens.astype(jnp.float32), data_axis),
            feature_axis,
        ).astype(jnp.int32)
        topk_valid = (
            jax.lax.pmin(
                topk_valid.astype(jnp.int32),
                (data_axis, feature_axis),
            )
            > 0
        )
        overflow = (
            jax.lax.pmax(
                overflow.astype(jnp.int32),
                (data_axis, feature_axis),
            )
            > 0
        )
        fired = jax.lax.pmax(fired.astype(jnp.int32), data_axis) > 0
        aux = Experiment5DistributedLossAux(
            sse=global_sse,
            sst=global_sst,
            l0_sum=global_l0,
            valid_tokens=global_tokens,
            topk_valid=topk_valid,
            fired_mask=fired,
            capacity_overflow=overflow,
        )
        healthy = topk_valid & ~overflow & (global_sst > 0)
        loss = jnp.where(
            healthy,
            global_sse / jnp.maximum(global_sst, _EPS),
            jnp.asarray(jnp.nan, jnp.float32),
        )
        return loss, aux

    aux_specs = Experiment5DistributedLossAux(
        P(),
        P(),
        P(),
        P(),
        P(),
        P(feature_axis),
        P(),
    )
    return jax.shard_map(
        local_objective,
        mesh=mesh,
        in_specs=(
            _param_specs(feature_axis),
            P(None, data_axis, None),
            P(None, data_axis, None, None),
            P(None, data_axis, None),
            P(None, data_axis, None),
            P(None, data_axis, None),
            P(None, data_axis),
            P(None),
            P(None),
            P(None),
            P(None),
            P(None),
            P(None),
        ),
        out_specs=(P(), aux_specs),
        axis_names={data_axis, feature_axis},
        check_vma=False,
    )


class Experiment5DistributedTrainer:
    """One compiled balanced H/R or global B/F optimizer update."""

    def __init__(
        self,
        mesh: Mesh,
        layout: Experiment5Layout,
        optimizer: ContributionOptimizerConfig | None = None,
        distributed: DistributedConfig | None = None,
        *,
        encoder_kind: Experiment5EncoderKind,
        bank_capacity: int | None = None,
        rmsnorm_epsilon: float = 1e-6,
        rematerialize_microbatch: bool = True,
        expected_physical_layout_sha256: str | None = None,
    ) -> None:
        if encoder_kind not in {"sae", "transcoder"}:
            raise ValueError("encoder_kind must be 'sae' or 'transcoder'")
        if (
            isinstance(rmsnorm_epsilon, bool)
            or not isinstance(rmsnorm_epsilon, (int, float))
            or not np.isfinite(rmsnorm_epsilon)
            or rmsnorm_epsilon <= 0
        ):
            raise ValueError("rmsnorm_epsilon must be finite and positive")
        if type(rematerialize_microbatch) is not bool:
            raise TypeError("rematerialize_microbatch must be boolean")
        self.mesh = mesh
        self.layout = layout
        self.optimizer = optimizer or ContributionOptimizerConfig()
        self.optimizer_config = experiment5_optimizer_sae_config(
            layout,
            self.optimizer,
        )
        self.distributed = distributed or DistributedConfig()
        self.encoder_kind = encoder_kind
        self.bank_capacity = bank_capacity
        self.rmsnorm_epsilon = float(rmsnorm_epsilon)
        self.rematerialize_microbatch = rematerialize_microbatch
        self.physical_layout = _validate_experiment5_mesh(
            mesh,
            layout,
            self.distributed,
        )
        if expected_physical_layout_sha256 is not None:
            require_experiment5_physical_layout(
                expected_physical_layout_sha256,
                self.physical_layout,
            )
        mapped = _build_experiment5_mapped_objective(
            mesh,
            layout,
            self.physical_layout,
            self.distributed,
            bank_capacity,
            self.optimizer.parameter_dtype,
            encoder_kind,
            self.rmsnorm_epsilon,
            rematerialize_microbatch,
        )

        def complete_step(
            state: TrainState,
            batch: Experiment5DistributedBatch,
            source_mean: jax.Array,
            routed_target_mean: jax.Array,
            shared_target_mean: jax.Array,
            full_target_mean: jax.Array,
            routed_scale: jax.Array,
            outer_scale: jax.Array,
        ) -> tuple[
            TrainState,
            StepMetrics,
            Experiment5DistributedLossAux,
            jax.Array,
        ]:
            differentiable_params = jax.tree_util.tree_map(
                lambda value: value.astype(jnp.float32),
                state.params,
            )
            (loss, aux), gradients = jax.value_and_grad(
                mapped,
                has_aux=True,
            )(
                differentiable_params,
                batch.routed_input,
                batch.expert_output,
                batch.shared_normalized,
                batch.bank_ids,
                batch.router_weights,
                batch.valid_mask,
                source_mean,
                routed_target_mean,
                shared_target_mean,
                full_target_mean,
                routed_scale,
                outer_scale,
            )
            expected_l0 = aux.valid_tokens * jnp.asarray(
                layout.total_l0,
                aux.l0_sum.dtype,
            )
            raw_l0 = aux.l0_sum.astype(jnp.float32) / jnp.maximum(
                aux.valid_tokens.astype(jnp.float32),
                1.0,
            )
            l0 = jnp.where(
                aux.l0_sum == expected_l0,
                jnp.asarray(layout.total_l0, jnp.float32),
                raw_l0,
            )
            updated, metrics = apply_gradients(
                state,
                gradients,
                self.optimizer_config,
                loss=loss,
                valid_tokens=aux.valid_tokens,
                fired_mask=aux.fired_mask,
                l0=l0,
                sse=aux.sse,
                sst=aux.sst,
                topk_valid=aux.topk_valid & ~aux.capacity_overflow,
            )
            state_finite = jnp.asarray(True)
            for tree in (
                updated.params,
                updated.optimizer.m,
                updated.optimizer.v,
            ):
                for leaf in jax.tree_util.tree_leaves(tree):
                    state_finite = state_finite & jnp.all(jnp.isfinite(leaf))
            return updated, metrics, aux, state_finite

        replicated = NamedSharding(mesh, P())
        self._compiled = jax.jit(
            complete_step,
            in_shardings=(
                state_shardings(mesh, self.distributed),
                experiment5_batch_shardings(mesh, self.distributed),
                replicated,
                replicated,
                replicated,
                replicated,
                replicated,
                replicated,
            ),
        )
        self._executable = None

    @property
    def physical_layout_sha256(self) -> str:
        return self.physical_layout.identity_sha256

    def _prepare(
        self,
        microbatches: Sequence[Any] | Experiment5DistributedBatch,
        centering: Experiment5Centering,
        recomposition: Experiment5Recomposition,
        *,
        process_local: bool,
    ) -> tuple[
        Experiment5DistributedBatch,
        Experiment5DistributedConstants,
    ]:
        if not isinstance(centering, Experiment5Centering):
            raise TypeError("centering must be Experiment5Centering")
        if not isinstance(recomposition, Experiment5Recomposition):
            raise TypeError("recomposition must be Experiment5Recomposition")
        if float(recomposition.epsilon) != self.rmsnorm_epsilon:
            raise ValueError(
                "recomposition epsilon differs from the compiled trainer"
            )
        batch = (
            microbatches
            if isinstance(microbatches, Experiment5DistributedBatch)
            else stack_experiment5_distributed_microbatches(
                microbatches,
                self.layout,
                self.distributed,
            )
        )
        _validate_stacked_batch(batch, self.layout, self.distributed)
        placed = (
            shard_process_local_experiment5_batch(
                batch,
                self.mesh,
                self.distributed,
            )
            if process_local
            else shard_experiment5_batch(
                batch,
                self.mesh,
                self.distributed,
            )
        )

        def replicated_vector(name: str, value: Any) -> jax.Array:
            result = replicate_process_local_array(
                jnp.asarray(value, jnp.float32),
                self.mesh,
            )
            if result.shape != (self.layout.hidden_dim,):
                raise ValueError(
                    f"{name} must have shape [{self.layout.hidden_dim}]"
                )
            return result

        constants = Experiment5DistributedConstants(
            replicated_vector("source_mean", centering.source_mean),
            replicated_vector(
                "routed_target_mean",
                centering.routed_target_mean,
            ),
            replicated_vector(
                "shared_target_mean",
                centering.shared_target_mean,
            ),
            replicated_vector(
                "full_target_mean",
                centering.full_target_mean,
            ),
            replicated_vector("routed_scale", recomposition.routed_scale),
            replicated_vector("outer_scale", recomposition.outer_scale),
        )
        return placed, constants

    def compile(
        self,
        state: TrainState,
        microbatches: Sequence[Any] | Experiment5DistributedBatch,
        centering: Experiment5Centering,
        recomposition: Experiment5Recomposition,
        *,
        process_local: bool = False,
    ) -> float:
        """Collectively compile without applying an optimizer update."""

        batch, constants = self._prepare(
            microbatches,
            centering,
            recomposition,
            process_local=process_local,
        )
        started = time.perf_counter()
        self._executable = self._compiled.lower(
            state,
            batch,
            *constants,
        ).compile()
        return time.perf_counter() - started

    def step(
        self,
        state: TrainState,
        microbatches: Sequence[Any] | Experiment5DistributedBatch,
        centering: Experiment5Centering,
        recomposition: Experiment5Recomposition,
        *,
        process_local: bool = False,
    ) -> tuple[TrainState, StepMetrics]:
        """Apply one update and reject overflow, invalid TopK, or nonfinite state."""

        batch, constants = self._prepare(
            microbatches,
            centering,
            recomposition,
            process_local=process_local,
        )
        executable = self._compiled if self._executable is None else self._executable
        new_state, metrics, aux, state_finite = executable(
            state,
            batch,
            *constants,
        )
        metrics_finite = (
            jnp.isfinite(metrics.loss)
            & jnp.isfinite(metrics.grad_norm)
            & jnp.isfinite(metrics.sse)
            & jnp.isfinite(metrics.sst)
        )
        (
            overflow,
            topk_valid,
            l0_sum,
            valid_tokens,
            metrics_finite,
            state_finite,
        ) = jax.device_get(
            (
                aux.capacity_overflow,
                aux.topk_valid,
                aux.l0_sum,
                aux.valid_tokens,
                metrics_finite,
                state_finite,
            )
        )
        if bool(overflow):
            raise RuntimeError(
                "H/R routed bank exceeded its precomputed grouped-GEMM "
                "capacity; no update was accepted"
            )
        if not bool(topk_valid):
            raise RuntimeError(
                "invalid Experiment-5 inputs or fewer than the exact positive "
                "TopK budget; no update was accepted"
            )
        token_count = int(valid_tokens)
        _validate_exact_distributed_l0(
            l0_sum=int(l0_sum),
            valid_tokens=token_count,
            k=self.layout.total_l0,
        )
        if not bool(metrics_finite):
            raise RuntimeError(
                "non-finite Experiment-5 update metrics; no update was accepted"
            )
        if not bool(state_finite):
            raise RuntimeError(
                "Experiment-5 update produced non-finite parameters or Adam "
                "moments; no update was accepted"
            )
        return new_state, metrics


__all__ = [
    "EXPERIMENT5_PHYSICAL_LAYOUT_SCHEMA_VERSION",
    "Experiment5DistributedBatch",
    "Experiment5DistributedConstants",
    "Experiment5DistributedLossAux",
    "Experiment5DistributedTrainer",
    "Experiment5PhysicalLayout",
    "experiment5_batch_shardings",
    "experiment5_physical_layout",
    "experiment5_physical_to_logical",
    "init_experiment5_distributed_state",
    "pack_experiment5_params",
    "pack_experiment5_train_state",
    "require_experiment5_physical_layout",
    "shard_experiment5_batch",
    "shard_experiment5_train_state",
    "shard_process_local_experiment5_batch",
    "stack_experiment5_distributed_microbatches",
    "unpack_experiment5_params",
    "unpack_experiment5_train_state",
]
