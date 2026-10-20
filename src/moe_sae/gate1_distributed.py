"""Token-level distributed scoring for the bounded Gemma Gate-1 audit.

The production evaluator intentionally returns only aggregate metrics.  Gate 1
needs one scalar SSE/SST pair per token so host-local code can retain
``sequence_id`` clusters without transferring full reconstructions off TPU.
This module reuses the production scoring, deterministic TopK, and sparse
decoder primitives while leaving the sealed production evaluator unchanged.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from moe_sae.distributed import (
    DistributedBatch,
    DistributedConfig,
    _local_a_scores,
    _local_global_scores,
    _named,
    _param_specs,
    _validate_mesh_model,
    batch_shardings,
    replicate_process_local_array,
    shard_batch,
    shard_process_local_batch,
)
from moe_sae.sae import (
    SAEConfig,
    SAEParams,
    decode_selected_columns_sharded,
    feature_sharded_topk,
)


class Gate1TokenScores(NamedTuple):
    """Globally shaped token errors with feature-sharded firing state."""

    token_sse: jax.Array  # [microbatches, global_microbatch], data sharded
    token_sst: jax.Array  # [microbatches, global_microbatch], data sharded
    token_l0: jax.Array  # [microbatches, global_microbatch], data sharded
    token_topk_valid: jax.Array  # [microbatches, global_microbatch], data sharded
    fired_mask: jax.Array  # [total_features], feature sharded
    capacity_overflow: jax.Array  # global scalar


def _build_gate1_mapped_scorer(
    mesh: Mesh,
    sae: SAEConfig,
    distributed: DistributedConfig,
    a_capacity: int | None,
):
    data_axis = distributed.data_axis
    feature_axis = distributed.feature_axis
    feature_size = int(mesh.shape[feature_axis])
    if sae.condition == "A" and a_capacity is None:
        raise ValueError("condition A Gate-1 scoring requires a fixed capacity")
    capacity = int(a_capacity or 1)

    def local_score(
        params: SAEParams,
        activation: jax.Array,
        expert_ids: jax.Array,
        valid_mask: jax.Array,
        mean: jax.Array,
    ) -> Gate1TokenScores:
        local_width = params.encoder.shape[0]
        feature_offset = jax.lax.axis_index(feature_axis) * local_width
        compute_dtype = params.encoder.dtype

        def one_micro(inputs):
            x, ids, valid = inputs
            centered = x.astype(compute_dtype) - mean.astype(compute_dtype)
            if sae.condition == "A":
                scores, latent_ids, overflow = _local_a_scores(
                    params,
                    centered,
                    ids,
                    sae,
                    feature_axis=feature_axis,
                    feature_size=feature_size,
                    capacity=capacity,
                )
            else:
                scores, latent_ids = _local_global_scores(
                    params, centered, feature_axis=feature_axis
                )
                overflow = jnp.asarray(False)
            selection = feature_sharded_topk(
                scores, latent_ids, sae.k, axis_name=feature_axis
            )
            reconstruction = decode_selected_columns_sharded(
                params.decoder,
                selection,
                mean,
                feature_offset=feature_offset,
                axis_name=feature_axis,
            )
            x32 = x.astype(jnp.float32)
            valid32 = valid.astype(jnp.float32)
            token_sse = (
                jnp.sum(
                    (x32 - reconstruction.astype(jnp.float32)) ** 2,
                    axis=1,
                )
                * valid32
            )
            token_sst = (
                jnp.sum(
                    (x32 - mean.astype(jnp.float32)[None, :]) ** 2,
                    axis=1,
                )
                * valid32
            )
            selected_valid = (
                (selection.values > 0) & selection.valid[:, None] & valid[:, None]
            )
            token_l0 = jnp.sum(selected_valid, axis=1, dtype=jnp.int32)
            token_topk_valid = selection.valid | ~valid
            owned = (selection.latent_ids >= feature_offset) & (
                selection.latent_ids < feature_offset + local_width
            )
            local_ids = jnp.clip(
                selection.latent_ids - feature_offset, 0, local_width - 1
            )
            fired = (
                jnp.zeros((local_width,), jnp.int32)
                .at[local_ids.reshape(-1)]
                .max((selected_valid & owned).reshape(-1).astype(jnp.int32))
                > 0
            )
            return (
                token_sse,
                token_sst,
                token_l0,
                token_topk_valid,
                fired,
                overflow,
            )

        (
            token_sse,
            token_sst,
            token_l0,
            token_topk_valid,
            fired_by_micro,
            overflow_by_micro,
        ) = jax.lax.map(one_micro, (activation, expert_ids, valid_mask))

        # Reconstruction and selection are replicated across feature shards.
        # Make that invariance explicit before returning data-sharded token
        # arrays from shard_map.
        token_sse = jax.lax.pmean(token_sse, feature_axis)
        token_sst = jax.lax.pmean(token_sst, feature_axis)
        token_l0 = jax.lax.pmax(token_l0, feature_axis)
        token_topk_valid = (
            jax.lax.pmin(token_topk_valid.astype(jnp.int32), feature_axis) > 0
        )
        fired = (
            jax.lax.pmax(jnp.any(fired_by_micro, axis=0).astype(jnp.int32), data_axis)
            > 0
        )
        overflow = (
            jax.lax.pmax(
                jnp.any(overflow_by_micro).astype(jnp.int32),
                (data_axis, feature_axis),
            )
            > 0
        )
        return Gate1TokenScores(
            token_sse,
            token_sst,
            token_l0,
            token_topk_valid,
            fired,
            overflow,
        )

    out_specs = Gate1TokenScores(
        P(None, data_axis),
        P(None, data_axis),
        P(None, data_axis),
        P(None, data_axis),
        P(feature_axis),
        P(),
    )
    return jax.shard_map(
        local_score,
        mesh=mesh,
        in_specs=(
            _param_specs(feature_axis),
            P(None, data_axis, None),
            P(None, data_axis, None),
            P(None, data_axis),
            P(None),
        ),
        out_specs=out_specs,
        axis_names={data_axis, feature_axis},
        check_vma=False,
    )


class Gate1DistributedScorer:
    """Compile once and score many fixed-shape Gate-1 microbatch groups."""

    def __init__(
        self,
        mesh: Mesh,
        sae_config: SAEConfig,
        distributed: DistributedConfig | None = None,
        *,
        a_capacity: int | None = None,
    ) -> None:
        self.mesh = mesh
        self.sae_config = sae_config
        self.distributed = distributed or DistributedConfig()
        _validate_mesh_model(mesh, sae_config, self.distributed)
        mapped = _build_gate1_mapped_scorer(
            mesh, sae_config, self.distributed, a_capacity
        )
        self._compiled = jax.jit(
            mapped,
            in_shardings=(
                _named(_param_specs(self.distributed.feature_axis), mesh),
                batch_shardings(mesh, self.distributed).activation,
                batch_shardings(mesh, self.distributed).expert_ids,
                batch_shardings(mesh, self.distributed).valid_mask,
                NamedSharding(mesh, P()),
            ),
        )

    def score(
        self,
        params: SAEParams,
        batch: DistributedBatch,
        mean: Any,
        *,
        process_local: bool,
    ) -> Gate1TokenScores:
        placed = (
            shard_process_local_batch(batch, self.mesh, self.distributed)
            if process_local
            else shard_batch(batch, self.mesh, self.distributed)
        )
        placed_mean = replicate_process_local_array(
            jnp.asarray(mean, jnp.float32), self.mesh
        )
        return self._compiled(
            params,
            placed.activation,
            placed.expert_ids,
            placed.valid_mask,
            placed_mean,
        )


def process_local_token_scores(
    scores: Gate1TokenScores,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Copy this process's one non-replicated token shard to host."""

    arrays: list[np.ndarray] = []
    for name in ("token_sse", "token_sst", "token_l0", "token_topk_valid"):
        value = getattr(scores, name)
        shards = value.addressable_shards
        if not shards:
            raise RuntimeError(f"Gate-1 {name} has no addressable shard")
        # The feature axis replicates token outputs on every local device.  All
        # addressable replicas have the same data-axis index on production
        # process-major meshes; taking one avoids multiplying observations.
        first_index = shards[0].index
        if any(shard.index != first_index for shard in shards[1:]):
            raise RuntimeError(
                "Gate-1 token output has more than one local data-axis shard"
            )
        arrays.append(np.asarray(jax.device_get(shards[0].data)))
    return tuple(arrays)  # type: ignore[return-value]


def process_local_feature_state(
    scores: Gate1TokenScores,
) -> tuple[np.ndarray, bool]:
    """Materialize one complete local replica of firing and overflow state.

    On the production mesh, ``fired_mask`` is partitioned over the host-local
    feature axis and replicated over the cross-host data axis.  A global
    multi-host array is not necessarily directly convertible with
    ``device_get`` even though every process owns one complete logical replica.
    Reassembling its addressable feature slices makes that layout explicit.
    """

    fired_shards = scores.fired_mask.addressable_shards
    if not fired_shards:
        raise RuntimeError("Gate-1 fired_mask has no addressable shard")
    total_features = int(scores.fired_mask.shape[0])
    fired = np.zeros((total_features,), dtype=bool)
    covered = np.zeros((total_features,), dtype=np.int8)
    for shard in fired_shards:
        index = shard.index
        if not isinstance(index, tuple) or len(index) != 1:
            raise RuntimeError("Gate-1 fired_mask shard index is malformed")
        selection = index[0]
        if not isinstance(selection, slice):
            raise RuntimeError("Gate-1 fired_mask must use contiguous feature slices")
        start = 0 if selection.start is None else int(selection.start)
        stop = total_features if selection.stop is None else int(selection.stop)
        step = 1 if selection.step is None else int(selection.step)
        if step != 1 or not 0 <= start < stop <= total_features:
            raise RuntimeError("Gate-1 fired_mask feature slice is invalid")
        value = np.asarray(jax.device_get(shard.data), dtype=bool).reshape(-1)
        if value.shape != (stop - start,):
            raise RuntimeError("Gate-1 fired_mask shard has the wrong shape")
        fired[start:stop] |= value
        covered[start:stop] += 1
    if not np.all(covered == 1):
        raise RuntimeError(
            "Gate-1 process does not own exactly one complete fired-mask replica"
        )

    overflow_shards = scores.capacity_overflow.addressable_shards
    if not overflow_shards:
        raise RuntimeError("Gate-1 capacity_overflow has no addressable shard")
    overflow_values = [
        bool(np.asarray(jax.device_get(shard.data)).item()) for shard in overflow_shards
    ]
    if len(set(overflow_values)) != 1:
        raise RuntimeError("Gate-1 capacity_overflow replicas disagree")
    return fired, overflow_values[0]


__all__ = [
    "Gate1DistributedScorer",
    "Gate1TokenScores",
    "process_local_feature_state",
    "process_local_token_scores",
]
