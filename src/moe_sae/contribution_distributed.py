"""Feature- and data-sharded training for Experiment-2 contribution SAEs.

The scientific oracle lives in :mod:`moe_sae.contribution_training`.  This
module preserves that objective while replacing its per-row encoder gather
with whole-bank grouped GEMMs.  Feature shards own contiguous complete banks;
the data axis owns disjoint source tokens.

Each feature shard takes a local TopK over only the bank rows it computed.
Only ``source_components × k`` values and global latent IDs are gathered
across the host-local feature axis.  Sparse decode gathers the selected local
decoder columns and sums their contributions without constructing dense
latents.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_training import (
    ContributionOptimizerConfig,
    optimizer_sae_config,
)
from moe_sae.distributed import (
    DistributedConfig,
    _param_specs,
    _validate_exact_distributed_l0,
    init_distributed_state,
    replicate_process_local_array,
    shard_train_state,
    state_shardings,
)
from moe_sae.sae import (
    SAEParams,
    Selection,
    StepMetrics,
    TrainState,
    apply_gradients,
    local_topk_candidates,
    merge_feature_shard_topk,
)

_EPS = 1e-12
_SENTINEL_ID = np.iinfo(np.int32).max


def _encoder_activation(
    preactivation: jax.Array,
    *,
    kind: str,
) -> jax.Array:
    """Apply the caller-selected encoder nonlinearity in FP32.

    ReLU remains the default for Experiment 2.  Experiment 3 explicitly opts
    into softplus.  The FP32-normal floor preserves mathematical softplus's
    strict positivity when an accelerator flushes subnormal values.
    """

    value = jnp.asarray(preactivation, jnp.float32)
    if kind == "relu":
        return jax.nn.relu(value)
    if kind == "softplus":
        return jnp.maximum(
            jax.nn.softplus(value),
            jnp.finfo(jnp.float32).tiny,
        )
    raise ValueError(f"unsupported encoder activation: {kind!r}")


class ContributionDistributedBatch(NamedTuple):
    """Static stacked microbatches with source-token data sharding."""

    expert_output: jax.Array  # [microbatches, global_microbatch, routes, hidden]
    bank_ids: jax.Array  # [microbatches, global_microbatch, routes]
    router_weights: jax.Array  # [microbatches, global_microbatch, routes]
    valid_mask: jax.Array  # [microbatches, global_microbatch]


class ContributionDistributedLossAux(NamedTuple):
    """Additive global statistics and feature-sharded firing state."""

    sse: jax.Array
    sst: jax.Array
    l0_sum: jax.Array
    valid_source_tokens: jax.Array
    valid_components: jax.Array
    topk_valid: jax.Array
    fired_mask: jax.Array
    capacity_overflow: jax.Array


def _batch_field(batch: Any, name: str, default: Any = None) -> Any:
    if isinstance(batch, Mapping):
        return batch.get(name, default)
    return getattr(batch, name, default)


def _validate_contribution_mesh(
    mesh: Mesh,
    layout: ContributionSAEConfig,
    distributed: DistributedConfig,
) -> None:
    shape = mesh.shape
    if distributed.data_axis not in shape or distributed.feature_axis not in shape:
        raise ValueError("mesh does not contain configured data/feature axes")
    feature_size = int(shape[distributed.feature_axis])
    if layout.bank_count % feature_size:
        raise ValueError(
            "contribution bank count must divide across feature shards; "
            "bank boundaries may not split"
        )
    if layout.total_features % feature_size:
        raise ValueError("contribution features must divide across feature shards")
    local_features = layout.total_features // feature_size
    local_banks = layout.bank_count // feature_size
    if local_features != local_banks * layout.bank_width:
        raise ValueError("feature shard boundary does not align to whole banks")
    if layout.bank_width < layout.k_per_contribution:
        raise ValueError("every local bank must contain at least local TopK features")
    if set(mesh.axis_names) != {
        distributed.data_axis,
        distributed.feature_axis,
    }:
        raise ValueError("contribution training requires an exact two-axis mesh")
    devices = np.asarray(mesh.devices, dtype=object)
    data_dimension = mesh.axis_names.index(distributed.data_axis)
    feature_dimension = mesh.axis_names.index(distributed.feature_axis)
    process_rows = np.moveaxis(
        devices,
        (data_dimension, feature_dimension),
        (0, 1),
    )
    for row in process_rows:
        if len({int(device.process_index) for device in row}) != 1:
            raise ValueError("the contribution feature axis must remain host-local")


def contribution_batch_shardings(
    mesh: Mesh,
    distributed: DistributedConfig,
) -> ContributionDistributedBatch:
    """Return the production batch shardings."""

    data = distributed.data_axis
    return ContributionDistributedBatch(
        NamedSharding(mesh, P(None, data, None, None)),
        NamedSharding(mesh, P(None, data, None)),
        NamedSharding(mesh, P(None, data, None)),
        NamedSharding(mesh, P(None, data)),
    )


def _validate_host_microbatch(
    batch: Any,
    layout: ContributionSAEConfig,
    *,
    expected_rows: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(_batch_field(batch, "expert_output"))
    banks = np.asarray(_batch_field(batch, "bank_ids"))
    weights = np.asarray(_batch_field(batch, "router_weights"))
    if target.ndim != 3 or target.shape[1:] != (
        layout.top_routed,
        layout.hidden_dim,
    ):
        raise ValueError(
            "expert_output must have shape "
            f"[tokens,{layout.top_routed},{layout.hidden_dim}]"
        )
    rows = target.shape[0]
    if expected_rows is not None and rows != expected_rows:
        raise ValueError("all microbatches must have the same static token count")
    if banks.shape != (rows, layout.top_routed):
        raise ValueError("bank_ids must match expert_output token and route axes")
    if weights.shape != banks.shape:
        raise ValueError("router_weights must match bank_ids")
    if not jnp.issubdtype(target.dtype, jnp.floating):
        raise TypeError("expert_output must be floating-point")
    if not np.issubdtype(banks.dtype, np.integer):
        raise TypeError("bank_ids must use an integer dtype")
    if not jnp.issubdtype(weights.dtype, jnp.floating):
        raise TypeError("router_weights must be floating-point")

    mask_value = _batch_field(batch, "valid_mask")
    valid = (
        np.ones((rows,), dtype=np.bool_)
        if mask_value is None
        else np.asarray(mask_value)
    )
    if valid.shape != (rows,) or valid.dtype != np.bool_:
        raise ValueError("valid_mask must be boolean with shape [tokens]")
    if not np.isfinite(target).all() or not np.isfinite(weights).all():
        raise ValueError("expert outputs and router weights must be finite")
    if np.any(weights < 0):
        raise ValueError("router weights must be nonnegative")
    if banks.size and (np.any(banks < 0) or np.any(banks >= layout.bank_count)):
        raise ValueError("bank ID is outside the configured bank range")

    scientific_banks = banks[valid].astype(np.int64, copy=False)
    scientific_weights = weights[valid].astype(np.float64, copy=False)
    if scientific_weights.size and not np.all(
        np.abs(np.sum(scientific_weights, axis=1) - 1.0) <= 0.005
    ):
        raise ValueError("scientific router weights must sum to one")
    if layout.condition in {"A", "R"}:
        if scientific_banks.shape[1] > 1 and np.any(
            np.diff(np.sort(scientific_banks, axis=1), axis=1) == 0
        ):
            raise ValueError("A/R bank IDs must be unique per scientific token")
    else:
        ranks = np.arange(layout.top_routed, dtype=np.int64)
        if scientific_banks.size and not np.all(scientific_banks == ranks[None, :]):
            raise ValueError("B/F bank IDs must equal canonical router ranks")

    return (
        target,
        banks.astype(np.int32, copy=False),
        weights,
        valid,
    )


def stack_contribution_microbatches(
    microbatches: Sequence[Any],
    layout: ContributionSAEConfig,
    distributed: DistributedConfig | None = None,
) -> ContributionDistributedBatch:
    """Validate and stack host microbatches without changing scientific rows."""

    selected = distributed or DistributedConfig()
    if len(microbatches) != selected.microbatches_per_update:
        raise ValueError(
            f"expected {selected.microbatches_per_update} microbatches, "
            f"got {len(microbatches)}"
        )
    targets: list[np.ndarray] = []
    banks: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    rows: int | None = None
    for microbatch in microbatches:
        target, bank, weight, valid = _validate_host_microbatch(
            microbatch,
            layout,
            expected_rows=rows,
        )
        rows = target.shape[0]
        targets.append(target)
        banks.append(bank)
        weights.append(weight)
        masks.append(valid)
    return ContributionDistributedBatch(
        np.stack(targets),
        np.asarray(np.stack(banks), np.int32),
        np.stack(weights),
        np.asarray(np.stack(masks), np.bool_),
    )


def shard_contribution_batch(
    batch: ContributionDistributedBatch,
    mesh: Mesh,
    distributed: DistributedConfig,
) -> ContributionDistributedBatch:
    """Place a global host batch on the data axis."""

    data_size = int(mesh.shape[distributed.data_axis])
    if batch.expert_output.shape[1] % data_size:
        raise ValueError("global microbatch must divide across the data axis")
    return jax.tree_util.tree_map(
        jax.device_put,
        batch,
        contribution_batch_shardings(mesh, distributed),
    )


def _validate_stacked_batch_shape(
    batch: ContributionDistributedBatch,
    layout: ContributionSAEConfig,
    distributed: DistributedConfig,
) -> None:
    target = batch.expert_output
    expected_tail = (layout.top_routed, layout.hidden_dim)
    if target.ndim != 4 or target.shape[2:] != expected_tail:
        raise ValueError(
            "stacked expert_output must be "
            f"[microbatches,tokens,{layout.top_routed},{layout.hidden_dim}]"
        )
    if target.shape[0] != distributed.microbatches_per_update:
        raise ValueError("stacked batch has the wrong number of microbatches")
    expected_routes = target.shape[:3]
    if batch.bank_ids.shape != expected_routes:
        raise ValueError("stacked bank_ids shape mismatch")
    if batch.router_weights.shape != expected_routes:
        raise ValueError("stacked router_weights shape mismatch")
    if batch.valid_mask.shape != target.shape[:2]:
        raise ValueError("stacked valid_mask shape mismatch")
    if not jnp.issubdtype(target.dtype, jnp.floating):
        raise TypeError("stacked expert_output must be floating-point")
    if not jnp.issubdtype(batch.bank_ids.dtype, jnp.integer):
        raise TypeError("stacked bank_ids must use an integer dtype")
    if not jnp.issubdtype(batch.router_weights.dtype, jnp.floating):
        raise TypeError("stacked router_weights must be floating-point")
    if batch.valid_mask.dtype != jnp.bool_:
        raise TypeError("stacked valid_mask must be boolean")


def shard_process_local_contribution_batch(
    batch: ContributionDistributedBatch,
    mesh: Mesh,
    distributed: DistributedConfig,
) -> ContributionDistributedBatch:
    """Assemble a global data-sharded batch from process-local host rows."""

    data_size = int(mesh.shape[distributed.data_axis])
    if data_size != jax.process_count():
        raise ValueError(
            "process-local contribution assembly requires one data row per process"
        )
    local_rows = batch.expert_output.shape[1]
    routes = batch.expert_output.shape[2]
    hidden = batch.expert_output.shape[3]
    shardings = contribution_batch_shardings(mesh, distributed)
    return ContributionDistributedBatch(
        jax.make_array_from_process_local_data(
            shardings.expert_output,
            np.asarray(batch.expert_output),
            global_shape=(
                batch.expert_output.shape[0],
                local_rows * data_size,
                routes,
                hidden,
            ),
        ),
        jax.make_array_from_process_local_data(
            shardings.bank_ids,
            np.asarray(batch.bank_ids),
            global_shape=(
                batch.bank_ids.shape[0],
                local_rows * data_size,
                routes,
            ),
        ),
        jax.make_array_from_process_local_data(
            shardings.router_weights,
            np.asarray(batch.router_weights),
            global_shape=(
                batch.router_weights.shape[0],
                local_rows * data_size,
                routes,
            ),
        ),
        jax.make_array_from_process_local_data(
            shardings.valid_mask,
            np.asarray(batch.valid_mask),
            global_shape=(batch.valid_mask.shape[0], local_rows * data_size),
        ),
    )


def init_contribution_distributed_state(
    key: jax.Array,
    mesh: Mesh,
    layout: ContributionSAEConfig,
    optimizer: ContributionOptimizerConfig | None = None,
    distributed: DistributedConfig | None = None,
) -> TrainState:
    """Initialize the generic SAE optimizer state directly into its shards."""

    selected = distributed or DistributedConfig()
    _validate_contribution_mesh(mesh, layout, selected)
    return init_distributed_state(
        key,
        mesh,
        optimizer_sae_config(layout, optimizer),
        selected,
    )


def shard_contribution_train_state(
    state: TrainState,
    mesh: Mesh,
    layout: ContributionSAEConfig,
    optimizer: ContributionOptimizerConfig | None = None,
    distributed: DistributedConfig | None = None,
) -> TrainState:
    """Place an oracle state on the contribution production mesh."""

    selected = distributed or DistributedConfig()
    _validate_contribution_mesh(mesh, layout, selected)
    return shard_train_state(
        state,
        mesh,
        optimizer_sae_config(layout, optimizer),
        selected,
    )


def _merge_local_candidates(
    values: jax.Array,
    latent_ids: jax.Array,
    k: int,
    *,
    feature_axis: str,
    require_strictly_positive: bool = True,
) -> Selection:
    gathered_values = jax.lax.all_gather(
        values,
        feature_axis,
        axis=0,
        tiled=False,
    )
    gathered_ids = jax.lax.all_gather(
        latent_ids,
        feature_axis,
        axis=0,
        tiled=False,
    )
    # A local shard may emit sentinel IDs when it has fewer than k positive
    # features.  Those sentinels sit between shards after all-gather, so this
    # tiny ``feature_shards × k`` merge retains its deterministic sort.
    return merge_feature_shard_topk(
        gathered_values,
        gathered_ids,
        k,
        require_strictly_positive=require_strictly_positive,
    )


def _decode_selected_columns_default_precision(
    local_decoder: jax.Array,
    selection: Selection,
    mean: jax.Array,
    *,
    feature_offset: int | jax.Array,
    feature_axis: str,
) -> jax.Array:
    """Decode selected local columns with the frozen DEFAULT matmul policy."""

    offset = jnp.asarray(feature_offset, jnp.int32)
    local_width = local_decoder.shape[1]
    owned = (selection.latent_ids >= offset) & (
        selection.latent_ids < offset + local_width
    )
    local_ids = jnp.clip(selection.latent_ids - offset, 0, local_width - 1)
    columns = local_decoder[:, local_ids]
    values = jnp.where(owned, selection.values, 0.0).astype(local_decoder.dtype)
    partial = jnp.einsum(
        "bk,dbk->bd",
        values,
        columns,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    )
    centered = jax.lax.psum(partial, feature_axis)
    return centered + jnp.asarray(mean, jnp.float32)[None, :]


def _local_grouped_candidates(
    params: SAEParams,
    centered: jax.Array,
    bank_ids: jax.Array,
    component_valid: jax.Array,
    layout: ContributionSAEConfig,
    *,
    feature_axis: str,
    capacity: int,
    encoder_activation: str = "relu",
    topk_requires_strictly_positive: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Take local TopK after a fixed-capacity whole-bank A/R GEMM."""

    local_features = params.encoder.shape[0]
    local_banks = local_features // layout.bank_width
    bank_start = jax.lax.axis_index(feature_axis) * local_banks
    flat_target = centered.reshape(-1, layout.hidden_dim)
    flat_banks = bank_ids.astype(jnp.int32).reshape(-1)
    flat_valid = component_valid.reshape(-1)
    local_id = flat_banks - bank_start
    owned = flat_valid & (local_id >= 0) & (local_id < local_banks)
    safe_local_id = jnp.clip(local_id, 0, local_banks - 1)

    membership = jax.nn.one_hot(safe_local_id, local_banks, dtype=jnp.int32) * owned[
        :, None
    ].astype(jnp.int32)
    cumulative = jnp.cumsum(membership, axis=0) - 1
    rank = jnp.sum(cumulative * membership, axis=1)
    overflow = jnp.any(owned & (rank >= capacity))
    in_capacity = owned & (rank < capacity)
    destination = safe_local_id * capacity + jnp.clip(rank, 0, capacity - 1)
    padded = (
        jnp.zeros(
            (local_banks * capacity, layout.hidden_dim),
            centered.dtype,
        )
        .at[destination]
        .add(jnp.where(in_capacity[:, None], flat_target, 0))
    )
    occupancy = (
        jnp.zeros((local_banks * capacity,), jnp.int32)
        .at[destination]
        .max(in_capacity.astype(jnp.int32))
        > 0
    )

    encoder = params.encoder.reshape(
        local_banks,
        layout.bank_width,
        layout.hidden_dim,
    )
    bias = params.bias.reshape(local_banks, layout.bank_width)
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
    ).reshape(
        local_banks * capacity,
        layout.bank_width,
    )
    grouped = jnp.where(occupancy[:, None], grouped, -jnp.inf)
    global_banks = bank_start + jnp.arange(local_banks, dtype=jnp.int32)
    candidate_ids = (
        global_banks[:, None, None] * layout.bank_width
        + jnp.arange(layout.bank_width, dtype=jnp.int32)[None, None, :]
    )
    candidate_ids = jnp.broadcast_to(
        candidate_ids,
        (local_banks, capacity, layout.bank_width),
    ).reshape(local_banks * capacity, layout.bank_width)
    local = local_topk_candidates(
        grouped,
        candidate_ids,
        layout.k_per_contribution,
        ids_are_sorted=True,
        require_strictly_positive=topk_requires_strictly_positive,
    )

    gathered_values = local.values[destination]
    gathered_ids = local.latent_ids[destination]
    gathered_values = jnp.where(
        in_capacity[:, None],
        gathered_values,
        0.0,
    )
    gathered_ids = jnp.where(
        in_capacity[:, None],
        gathered_ids,
        jnp.asarray(_SENTINEL_ID, jnp.int32),
    )
    return gathered_values, gathered_ids, overflow


def _local_rank_candidates(
    params: SAEParams,
    centered: jax.Array,
    component_valid: jax.Array,
    layout: ContributionSAEConfig,
    *,
    feature_axis: str,
    encoder_activation: str = "relu",
    topk_requires_strictly_positive: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Take local TopK for complete B/F router-rank banks."""

    tokens = centered.shape[0]
    local_features = params.encoder.shape[0]
    local_banks = local_features // layout.bank_width
    bank_start = jax.lax.axis_index(feature_axis) * local_banks
    by_rank = jnp.swapaxes(centered, 0, 1)
    local_target = jax.lax.dynamic_slice_in_dim(
        by_rank,
        bank_start,
        local_banks,
        axis=0,
    )
    local_valid = jax.lax.dynamic_slice_in_dim(
        jnp.swapaxes(component_valid, 0, 1),
        bank_start,
        local_banks,
        axis=0,
    )
    encoder = params.encoder.reshape(
        local_banks,
        layout.bank_width,
        layout.hidden_dim,
    )
    bias = params.bias.reshape(local_banks, layout.bank_width)
    grouped = jnp.einsum(
        "bnd,bwd->bnw",
        local_target,
        encoder.astype(centered.dtype),
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    ) + bias[:, None, :].astype(jnp.float32)
    grouped = jnp.where(
        local_valid[..., None],
        _encoder_activation(grouped, kind=encoder_activation),
        -jnp.inf,
    )
    global_banks = bank_start + jnp.arange(local_banks, dtype=jnp.int32)
    candidate_ids = (
        global_banks[:, None, None] * layout.bank_width
        + jnp.arange(layout.bank_width, dtype=jnp.int32)[None, None, :]
    )
    candidate_ids = jnp.broadcast_to(
        candidate_ids,
        (local_banks, tokens, layout.bank_width),
    )
    local = local_topk_candidates(
        grouped.reshape(local_banks * tokens, layout.bank_width),
        candidate_ids.reshape(local_banks * tokens, layout.bank_width),
        layout.k_per_contribution,
        ids_are_sorted=True,
        require_strictly_positive=topk_requires_strictly_positive,
    )

    global_rows = (
        jnp.arange(tokens, dtype=jnp.int32)[None, :] * layout.top_routed
        + global_banks[:, None]
    ).reshape(-1)
    component_rows = tokens * layout.top_routed
    values = (
        jnp.zeros(
            (component_rows, layout.k_per_contribution),
            jnp.float32,
        )
        .at[global_rows]
        .set(local.values)
    )
    latent_ids = (
        jnp.full(
            (component_rows, layout.k_per_contribution),
            _SENTINEL_ID,
            jnp.int32,
        )
        .at[global_rows]
        .set(local.latent_ids)
    )
    return values, latent_ids


def _scientific_input_valid(
    target: jax.Array,
    bank_ids: jax.Array,
    router_weights: jax.Array,
    valid_mask: jax.Array,
    layout: ContributionSAEConfig,
) -> jax.Array:
    scientific = valid_mask[:, None]
    bank_range = (bank_ids >= 0) & (bank_ids < layout.bank_count)
    bank_valid = jnp.all(bank_range | ~scientific)
    if layout.condition in {"A", "R"}:
        ordered = jnp.sort(bank_ids, axis=1)
        unique = jnp.all((jnp.diff(ordered, axis=1) != 0) | ~valid_mask[:, None])
        bank_valid = bank_valid & unique
    else:
        ranks = jnp.arange(layout.top_routed, dtype=bank_ids.dtype)[None, :]
        bank_valid = bank_valid & jnp.all((bank_ids == ranks) | ~scientific)

    finite_target = jnp.all(jnp.isfinite(target) | ~valid_mask[:, None, None])
    finite_weights = jnp.all(jnp.isfinite(router_weights) | ~scientific)
    nonnegative = jnp.all((router_weights >= 0) | ~scientific)
    normalized = jnp.all(
        (jnp.abs(jnp.sum(router_weights, axis=1, dtype=jnp.float32) - 1.0) <= 0.005)
        | ~valid_mask
    )
    return bank_valid & finite_target & finite_weights & nonnegative & normalized


def _build_contribution_mapped_objective(
    mesh: Mesh,
    layout: ContributionSAEConfig,
    distributed: DistributedConfig,
    bank_capacity: int | None,
    parameter_dtype: str,
    rematerialize_microbatch: bool,
    encoder_activation: str,
    topk_requires_strictly_positive: bool,
):
    data_axis = distributed.data_axis
    feature_axis = distributed.feature_axis
    routed_banks = layout.condition in {"A", "R"}
    if routed_banks:
        if bank_capacity is None or bank_capacity <= 0:
            raise ValueError(
                "distributed A/R contribution training requires a positive "
                "fixed per-bank capacity"
            )
    elif bank_capacity is not None:
        raise ValueError("B/F rank-bank training does not accept a bank capacity")
    capacity = int(bank_capacity or 1)
    compute_dtype = jnp.bfloat16 if parameter_dtype == "bfloat16" else jnp.float32

    def local_objective(
        differentiable_params: SAEParams,
        expert_output: jax.Array,
        bank_ids: jax.Array,
        router_weights: jax.Array,
        valid_mask: jax.Array,
        mean: jax.Array,
    ) -> tuple[jax.Array, ContributionDistributedLossAux]:
        # Parameters are stored in BF16, but differentiating directly with
        # respect to a BF16 primal makes JAX return BF16 cotangents.  The FP32
        # shadow is differentiation metadata rather than a second optimizer
        # state.  Cast inside the scanned function—not outside it—so the
        # transpose scan also carries and adds per-microbatch gradients in FP32.
        local_features = differentiable_params.encoder.shape[0]
        feature_offset = jax.lax.axis_index(feature_axis) * local_features
        initial = (
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(True),
            jnp.zeros((local_features,), jnp.bool_),
            jnp.asarray(False),
        )

        def one_micro(carry, inputs):
            (
                sse,
                sst,
                l0_sum,
                source_tokens,
                valid_components,
                valid_topk,
                fired,
                overflow,
            ) = carry
            target, banks, weights, valid = inputs
            params = jax.tree_util.tree_map(
                lambda value: value.astype(compute_dtype),
                differentiable_params,
            )
            centered = target.astype(compute_dtype) - mean.astype(compute_dtype)
            component_valid = jnp.broadcast_to(
                valid[:, None],
                banks.shape,
            )
            if routed_banks:
                local_values, local_ids, overflow_here = _local_grouped_candidates(
                    params,
                    centered,
                    banks,
                    component_valid,
                    layout,
                    feature_axis=feature_axis,
                    capacity=capacity,
                    encoder_activation=encoder_activation,
                    topk_requires_strictly_positive=(
                        topk_requires_strictly_positive
                    ),
                )
            else:
                local_values, local_ids = _local_rank_candidates(
                    params,
                    centered,
                    component_valid,
                    layout,
                    feature_axis=feature_axis,
                    encoder_activation=encoder_activation,
                    topk_requires_strictly_positive=(
                        topk_requires_strictly_positive
                    ),
                )
                overflow_here = jnp.asarray(False)
            selection = _merge_local_candidates(
                local_values,
                local_ids,
                layout.k_per_contribution,
                feature_axis=feature_axis,
                require_strictly_positive=topk_requires_strictly_positive,
            )
            reconstruction = _decode_selected_columns_default_precision(
                params.decoder,
                selection,
                mean,
                feature_offset=feature_offset,
                feature_axis=feature_axis,
            ).reshape(target.shape)

            target32 = target.astype(jnp.float32)
            weights32 = weights.astype(jnp.float32)
            mean32 = mean.astype(jnp.float32)
            weighted_error = weights32[..., None] * (
                target32 - reconstruction.astype(jnp.float32)
            )
            weighted_centered = weights32[..., None] * (
                target32 - mean32[None, None, :]
            )
            valid32 = valid.astype(jnp.float32)
            sse_here = jnp.sum(
                jnp.sum(weighted_error**2, axis=(1, 2)) * valid32,
                dtype=jnp.float32,
            )
            sst_here = jnp.sum(
                jnp.sum(weighted_centered**2, axis=(1, 2)) * valid32,
                dtype=jnp.float32,
            )

            flat_valid = component_valid.reshape(-1)
            selected_valid = (
                (selection.values > 0) & selection.valid[:, None] & flat_valid[:, None]
            )
            owned = (selection.latent_ids >= feature_offset) & (
                selection.latent_ids < feature_offset + local_features
            )
            local_ids = jnp.clip(
                selection.latent_ids - feature_offset,
                0,
                local_features - 1,
            )
            fired_here = (
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
            topk_here = jnp.all(selection.valid | ~flat_valid) & inputs_valid
            tokens_here = jnp.sum(valid, dtype=jnp.int32)
            components_here = jnp.sum(component_valid, dtype=jnp.int32)
            return (
                sse + sse_here,
                sst + sst_here,
                l0_sum + jnp.sum(selected_valid, dtype=jnp.int32),
                source_tokens + tokens_here,
                valid_components + components_here,
                valid_topk & topk_here,
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
        global_components = jax.lax.pmean(
            jax.lax.psum(local_components.astype(jnp.float32), data_axis),
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
        aux = ContributionDistributedLossAux(
            sse=global_sse,
            sst=global_sst,
            l0_sum=global_l0,
            valid_source_tokens=global_tokens,
            valid_components=global_components,
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

    aux_specs = ContributionDistributedLossAux(
        P(),
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
            P(None, data_axis, None, None),
            P(None, data_axis, None),
            P(None, data_axis, None),
            P(None, data_axis),
            P(None),
        ),
        out_specs=(P(), aux_specs),
        axis_names={data_axis, feature_axis},
        check_vma=False,
    )


class ContributionDistributedTrainer:
    """One compiled production-shaped contribution optimizer update."""

    def __init__(
        self,
        mesh: Mesh,
        layout: ContributionSAEConfig,
        optimizer: ContributionOptimizerConfig | None = None,
        distributed: DistributedConfig | None = None,
        *,
        bank_capacity: int | None = None,
        rematerialize_microbatch: bool = True,
        encoder_activation: str = "relu",
        topk_requires_strictly_positive: bool = True,
    ) -> None:
        self.mesh = mesh
        self.layout = layout
        self.optimizer = optimizer or ContributionOptimizerConfig()
        self.optimizer_config = optimizer_sae_config(layout, self.optimizer)
        self.distributed = distributed or DistributedConfig()
        self.bank_capacity = bank_capacity
        if type(rematerialize_microbatch) is not bool:
            raise TypeError("rematerialize_microbatch must be boolean")
        if encoder_activation not in {"relu", "softplus"}:
            raise ValueError("encoder_activation must be 'relu' or 'softplus'")
        if type(topk_requires_strictly_positive) is not bool:
            raise TypeError("topk_requires_strictly_positive must be boolean")
        if (
            encoder_activation == "softplus"
            and not topk_requires_strictly_positive
        ):
            raise ValueError(
                "softplus must retain the strict-positive TopK contract"
            )
        self.rematerialize_microbatch = rematerialize_microbatch
        self.encoder_activation = encoder_activation
        self.topk_requires_strictly_positive = (
            topk_requires_strictly_positive
        )
        _validate_contribution_mesh(mesh, layout, self.distributed)
        mapped_objective = _build_contribution_mapped_objective(
            mesh,
            layout,
            self.distributed,
            bank_capacity,
            self.optimizer.parameter_dtype,
            rematerialize_microbatch,
            encoder_activation,
            topk_requires_strictly_positive,
        )

        def complete_step(
            state: TrainState,
            batch: ContributionDistributedBatch,
            mean: jax.Array,
        ) -> tuple[
            TrainState,
            StepMetrics,
            ContributionDistributedLossAux,
            jax.Array,
        ]:
            differentiable_params = jax.tree_util.tree_map(
                lambda value: value.astype(jnp.float32),
                state.params,
            )
            (loss, aux), gradients = jax.value_and_grad(
                mapped_objective,
                has_aux=True,
            )(
                differentiable_params,
                batch.expert_output,
                batch.bank_ids,
                batch.router_weights,
                batch.valid_mask,
                mean,
            )
            expected_l0 = aux.valid_source_tokens * jnp.asarray(
                layout.total_l0,
                aux.l0_sum.dtype,
            )
            raw_l0 = aux.l0_sum.astype(jnp.float32) / jnp.maximum(
                aux.valid_source_tokens.astype(jnp.float32),
                1.0,
            )
            l0 = jnp.where(
                aux.l0_sum == expected_l0,
                jnp.asarray(layout.total_l0, jnp.float32),
                raw_l0,
            )
            state, metrics = apply_gradients(
                state,
                gradients,
                self.optimizer_config,
                loss=loss,
                valid_tokens=aux.valid_source_tokens,
                fired_mask=aux.fired_mask,
                l0=l0,
                sse=aux.sse,
                sst=aux.sst,
                topk_valid=aux.topk_valid & ~aux.capacity_overflow,
            )
            state_finite = jnp.asarray(True)
            for tree in (state.params, state.optimizer.m, state.optimizer.v):
                for leaf in jax.tree_util.tree_leaves(tree):
                    state_finite = state_finite & jnp.all(jnp.isfinite(leaf))
            return state, metrics, aux, state_finite

        self._compiled = jax.jit(
            complete_step,
            in_shardings=(
                state_shardings(mesh, self.distributed),
                contribution_batch_shardings(mesh, self.distributed),
                NamedSharding(mesh, P()),
            ),
        )
        self._executable = None

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
        mean_array = replicate_process_local_array(
            jnp.asarray(mean, jnp.float32),
            self.mesh,
        )
        if mean_array.shape != (self.layout.hidden_dim,):
            raise ValueError(f"mean must have shape [{self.layout.hidden_dim}]")
        return placed, mean_array

    def compile(
        self,
        state: TrainState,
        microbatches: Sequence[Any] | ContributionDistributedBatch,
        mean: Any,
        *,
        process_local: bool = False,
    ) -> float:
        """Collectively compile without applying an optimizer update."""

        batch, mean_array = self._prepare(
            microbatches,
            mean,
            process_local=process_local,
        )
        started = time.perf_counter()
        self._executable = self._compiled.lower(state, batch, mean_array).compile()
        return time.perf_counter() - started

    def step(
        self,
        state: TrainState,
        microbatches: Sequence[Any] | ContributionDistributedBatch,
        mean: Any,
        *,
        process_local: bool = False,
    ) -> tuple[TrainState, StepMetrics]:
        """Apply one update, failing before the returned state is accepted."""

        batch, mean_array = self._prepare(
            microbatches,
            mean,
            process_local=process_local,
        )
        executable = self._compiled if self._executable is None else self._executable
        new_state, metrics, aux, state_finite = executable(state, batch, mean_array)
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
            source_tokens,
            valid_components,
            metrics_finite,
            state_finite,
        ) = jax.device_get(
            (
                aux.capacity_overflow,
                aux.topk_valid,
                aux.l0_sum,
                aux.valid_source_tokens,
                aux.valid_components,
                metrics_finite,
                state_finite,
            )
        )
        if bool(overflow):
            raise RuntimeError(
                "A/R contribution bank exceeded its precomputed grouped-GEMM "
                "capacity; no update was accepted"
            )
        if not bool(topk_valid):
            raise RuntimeError(
                "invalid contribution inputs or fewer than k strictly positive "
                "features on a scientific component; no update was accepted"
            )
        token_count = int(source_tokens)
        component_count = int(valid_components)
        if component_count != token_count * self.layout.top_routed:
            raise RuntimeError(
                "distributed contribution component count differs from source "
                "token count"
            )
        if self.topk_requires_strictly_positive:
            _validate_exact_distributed_l0(
                l0_sum=int(l0_sum),
                valid_tokens=token_count,
                k=self.layout.total_l0,
            )
        elif not 0 <= int(l0_sum) <= token_count * self.layout.total_l0:
            raise RuntimeError(
                "inclusive-zero ReLU positive L0 is outside its nominal "
                "TopK slot budget"
            )
        if not bool(metrics_finite):
            raise RuntimeError(
                "non-finite distributed contribution update; no update was accepted"
            )
        if not bool(state_finite):
            raise RuntimeError(
                "distributed contribution update produced non-finite parameters "
                "or Adam moments; no update was accepted"
            )
        return new_state, metrics


_TRAINER_CACHE: dict[tuple[Any, ...], ContributionDistributedTrainer] = {}


def _trainer_key(
    mesh: Mesh,
    layout: ContributionSAEConfig,
    optimizer: ContributionOptimizerConfig,
    distributed: DistributedConfig,
    bank_capacity: int | None,
    rematerialize_microbatch: bool,
    topk_requires_strictly_positive: bool,
) -> tuple[Any, ...]:
    devices = tuple(int(device.id) for device in np.asarray(mesh.devices).reshape(-1))
    return (
        devices,
        tuple(mesh.axis_names),
        layout,
        optimizer,
        distributed,
        bank_capacity,
        rematerialize_microbatch,
        topk_requires_strictly_positive,
    )


def distributed_contribution_train_step(
    state: TrainState,
    microbatches: Sequence[Any] | ContributionDistributedBatch,
    mean: Any,
    layout: ContributionSAEConfig,
    mesh: Mesh,
    *,
    optimizer: ContributionOptimizerConfig | None = None,
    distributed: DistributedConfig | None = None,
    bank_capacity: int | None = None,
    process_local: bool = False,
    rematerialize_microbatch: bool = True,
    topk_requires_strictly_positive: bool = True,
) -> tuple[TrainState, StepMetrics]:
    """Cached functional wrapper around the compiled contribution trainer."""

    selected_optimizer = optimizer or ContributionOptimizerConfig()
    selected_distributed = distributed or DistributedConfig()
    key = _trainer_key(
        mesh,
        layout,
        selected_optimizer,
        selected_distributed,
        bank_capacity,
        rematerialize_microbatch,
        topk_requires_strictly_positive,
    )
    trainer = _TRAINER_CACHE.get(key)
    if trainer is None:
        trainer = ContributionDistributedTrainer(
            mesh,
            layout,
            selected_optimizer,
            selected_distributed,
            bank_capacity=bank_capacity,
            rematerialize_microbatch=rematerialize_microbatch,
            topk_requires_strictly_positive=(
                topk_requires_strictly_positive
            ),
        )
        _TRAINER_CACHE[key] = trainer
    return trainer.step(
        state,
        microbatches,
        mean,
        process_local=process_local,
    )


__all__ = [
    "ContributionDistributedBatch",
    "ContributionDistributedLossAux",
    "ContributionDistributedTrainer",
    "contribution_batch_shardings",
    "distributed_contribution_train_step",
    "init_contribution_distributed_state",
    "shard_contribution_batch",
    "shard_contribution_train_state",
    "shard_process_local_contribution_batch",
    "stack_contribution_microbatches",
]
