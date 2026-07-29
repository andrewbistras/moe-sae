"""Feature/data-sharded production trainer for Experiment-3 transcoders.

This module deliberately parallels :mod:`moe_sae.contribution_distributed`
without weakening its SAE defaults.  The encoder operand in this graph is
structurally only ``routed_input - source_mean``.  Targets and router weights
enter after sparse decode, and the fixed target mean is decoder-side only.
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

from moe_sae.contribution_distributed import (
    ContributionDistributedLossAux,
    _decode_selected_columns_default_precision,
    _local_grouped_candidates,
    _local_rank_candidates,
    _merge_local_candidates,
    _scientific_input_valid,
    _validate_contribution_mesh,
)
from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_training import (
    ContributionOptimizerConfig,
    optimizer_sae_config,
)
from moe_sae.contribution_transcoder import (
    ENCODER_ACTIVATION,
    ContributionTranscoderParams,
    ContributionTranscoderTrainState,
    transcoder_params_from_sae,
    transcoder_params_to_sae,
    transcoder_state_from_sae,
    transcoder_state_to_sae,
)
from moe_sae.distributed import (
    DistributedConfig,
    _param_specs,
    _validate_exact_distributed_l0,
    init_distributed_state,
    replicate_process_local_array,
    state_shardings,
)
from moe_sae.sae import StepMetrics, apply_gradients

_EPS = 1e-12


class ContributionTranscoderCentering(NamedTuple):
    """The two fixed means; their roles may not be interchanged."""

    source_mean: Any
    target_mean: Any


class ContributionTranscoderDistributedBatch(NamedTuple):
    """Stacked transcoder microbatches with source-token data sharding."""

    routed_input: jax.Array  # [microbatches, global_microbatch, hidden]
    expert_output: jax.Array  # [microbatches, global_microbatch, routes, hidden]
    bank_ids: jax.Array  # [microbatches, global_microbatch, routes]
    router_weights: jax.Array  # [microbatches, global_microbatch, routes]
    valid_mask: jax.Array  # [microbatches, global_microbatch]


def _batch_field(batch: Any, name: str, default: Any = None) -> Any:
    if isinstance(batch, Mapping):
        return batch.get(name, default)
    return getattr(batch, name, default)


def contribution_transcoder_batch_shardings(
    mesh: Mesh,
    distributed: DistributedConfig,
) -> ContributionTranscoderDistributedBatch:
    data = distributed.data_axis
    return ContributionTranscoderDistributedBatch(
        NamedSharding(mesh, P(None, data, None)),
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(_batch_field(batch, "routed_input"))
    target = np.asarray(_batch_field(batch, "expert_output"))
    banks = np.asarray(_batch_field(batch, "bank_ids"))
    weights = np.asarray(_batch_field(batch, "router_weights"))
    if source.ndim != 2 or source.shape[1:] != (layout.hidden_dim,):
        raise ValueError(f"routed_input must have shape [tokens,{layout.hidden_dim}]")
    rows = source.shape[0]
    if expected_rows is not None and rows != expected_rows:
        raise ValueError("all microbatches must have the same static token count")
    if target.shape != (rows, layout.top_routed, layout.hidden_dim):
        raise ValueError(
            "expert_output must have shape "
            f"[tokens,{layout.top_routed},{layout.hidden_dim}]"
        )
    if banks.shape != (rows, layout.top_routed):
        raise ValueError("bank_ids must match target token and route axes")
    if weights.shape != banks.shape:
        raise ValueError("router_weights must match bank_ids")
    mask_value = _batch_field(batch, "valid_mask")
    valid = (
        np.ones((rows,), dtype=np.bool_)
        if mask_value is None
        else np.asarray(mask_value)
    )
    if valid.shape != (rows,) or valid.dtype != np.bool_:
        raise ValueError("valid_mask must be boolean [tokens]")
    if not np.issubdtype(source.dtype, np.floating):
        raise TypeError("routed_input must be floating-point")
    if not np.issubdtype(target.dtype, np.floating):
        raise TypeError("expert_output must be floating-point")
    if not np.issubdtype(banks.dtype, np.integer):
        raise TypeError("bank_ids must be integer")
    if not np.issubdtype(weights.dtype, np.floating):
        raise TypeError("router_weights must be floating-point")
    if not np.isfinite(source).all():
        raise ValueError("routed_input must be finite")
    if not np.isfinite(target).all() or not np.isfinite(weights).all():
        raise ValueError("targets and router weights must be finite")
    if np.any(weights < 0):
        raise ValueError("router weights must be nonnegative")
    if banks.size and (np.any(banks < 0) or np.any(banks >= layout.bank_count)):
        raise ValueError("bank ID is outside the configured range")
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
            raise ValueError("A/R bank IDs must be unique per token")
    else:
        ranks = np.arange(layout.top_routed, dtype=np.int64)
        if scientific_banks.size and not np.all(scientific_banks == ranks[None, :]):
            raise ValueError("B/F bank IDs must equal canonical router ranks")
    return (
        source,
        target,
        banks.astype(np.int32, copy=False),
        weights,
        valid,
    )


def stack_contribution_transcoder_microbatches(
    microbatches: Sequence[Any],
    layout: ContributionSAEConfig,
    distributed: DistributedConfig | None = None,
) -> ContributionTranscoderDistributedBatch:
    selected = distributed or DistributedConfig()
    if len(microbatches) != selected.microbatches_per_update:
        raise ValueError(
            f"expected {selected.microbatches_per_update} microbatches, "
            f"got {len(microbatches)}"
        )
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    banks: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    rows: int | None = None
    for microbatch in microbatches:
        source, target, bank, weight, valid = _validate_host_microbatch(
            microbatch,
            layout,
            expected_rows=rows,
        )
        rows = source.shape[0]
        sources.append(source)
        targets.append(target)
        banks.append(bank)
        weights.append(weight)
        masks.append(valid)
    return ContributionTranscoderDistributedBatch(
        np.stack(sources),
        np.stack(targets),
        np.asarray(np.stack(banks), np.int32),
        np.stack(weights),
        np.asarray(np.stack(masks), np.bool_),
    )


def _validate_stacked_batch(
    batch: ContributionTranscoderDistributedBatch,
    layout: ContributionSAEConfig,
    distributed: DistributedConfig,
) -> None:
    source = batch.routed_input
    target = batch.expert_output
    if (
        source.ndim != 3
        or source.shape[0] != distributed.microbatches_per_update
        or source.shape[2] != layout.hidden_dim
    ):
        raise ValueError("stacked routed_input shape is incompatible")
    if target.shape != (
        source.shape[0],
        source.shape[1],
        layout.top_routed,
        layout.hidden_dim,
    ):
        raise ValueError("stacked expert_output shape is incompatible")
    if batch.bank_ids.shape != target.shape[:3]:
        raise ValueError("stacked bank_ids shape mismatch")
    if batch.router_weights.shape != target.shape[:3]:
        raise ValueError("stacked router_weights shape mismatch")
    if batch.valid_mask.shape != source.shape[:2]:
        raise ValueError("stacked valid_mask shape mismatch")


def shard_contribution_transcoder_batch(
    batch: ContributionTranscoderDistributedBatch,
    mesh: Mesh,
    distributed: DistributedConfig,
) -> ContributionTranscoderDistributedBatch:
    data_size = int(mesh.shape[distributed.data_axis])
    if batch.routed_input.shape[1] % data_size:
        raise ValueError("global microbatch must divide across the data axis")
    return jax.tree_util.tree_map(
        jax.device_put,
        batch,
        contribution_transcoder_batch_shardings(mesh, distributed),
    )


def shard_process_local_contribution_transcoder_batch(
    batch: ContributionTranscoderDistributedBatch,
    mesh: Mesh,
    distributed: DistributedConfig,
) -> ContributionTranscoderDistributedBatch:
    data_size = int(mesh.shape[distributed.data_axis])
    if data_size != jax.process_count():
        raise ValueError(
            "process-local transcoder assembly requires one data row per process"
        )
    shardings = contribution_transcoder_batch_shardings(mesh, distributed)
    local_rows = batch.routed_input.shape[1]
    global_rows = local_rows * data_size

    def place(value: Any, sharding: Any, global_shape: tuple[int, ...]) -> Any:
        return jax.make_array_from_process_local_data(
            sharding,
            np.asarray(value),
            global_shape=global_shape,
        )

    return ContributionTranscoderDistributedBatch(
        place(
            batch.routed_input,
            shardings.routed_input,
            (
                batch.routed_input.shape[0],
                global_rows,
                batch.routed_input.shape[2],
            ),
        ),
        place(
            batch.expert_output,
            shardings.expert_output,
            (
                batch.expert_output.shape[0],
                global_rows,
                batch.expert_output.shape[2],
                batch.expert_output.shape[3],
            ),
        ),
        place(
            batch.bank_ids,
            shardings.bank_ids,
            (
                batch.bank_ids.shape[0],
                global_rows,
                batch.bank_ids.shape[2],
            ),
        ),
        place(
            batch.router_weights,
            shardings.router_weights,
            (
                batch.router_weights.shape[0],
                global_rows,
                batch.router_weights.shape[2],
            ),
        ),
        place(
            batch.valid_mask,
            shardings.valid_mask,
            (batch.valid_mask.shape[0], global_rows),
        ),
    )


def _transcoder_state_shardings(
    mesh: Mesh,
    distributed: DistributedConfig,
) -> ContributionTranscoderTrainState:
    return transcoder_state_from_sae(state_shardings(mesh, distributed))


def init_contribution_transcoder_distributed_state(
    key: jax.Array,
    mesh: Mesh,
    layout: ContributionSAEConfig,
    optimizer: ContributionOptimizerConfig | None = None,
    distributed: DistributedConfig | None = None,
) -> ContributionTranscoderTrainState:
    selected = distributed or DistributedConfig()
    _validate_contribution_mesh(mesh, layout, selected)
    state = init_distributed_state(
        key,
        mesh,
        optimizer_sae_config(layout, optimizer),
        selected,
    )
    return transcoder_state_from_sae(state)


def _build_mapped_objective(
    mesh: Mesh,
    layout: ContributionSAEConfig,
    distributed: DistributedConfig,
    bank_capacity: int | None,
    parameter_dtype: str,
    rematerialize_microbatch: bool,
    encoder_activation: str,
):
    data_axis = distributed.data_axis
    feature_axis = distributed.feature_axis
    routed_banks = layout.condition in {"A", "R"}
    if routed_banks:
        if bank_capacity is None or bank_capacity <= 0:
            raise ValueError(
                "distributed A/R transcoder training requires positive capacity"
            )
    elif bank_capacity is not None:
        raise ValueError("B/F transcoder training does not accept capacity")
    capacity = int(bank_capacity or 1)
    compute_dtype = jnp.bfloat16 if parameter_dtype == "bfloat16" else jnp.float32

    def local_objective(
        differentiable_params: ContributionTranscoderParams,
        routed_input: jax.Array,
        expert_output: jax.Array,
        bank_ids: jax.Array,
        router_weights: jax.Array,
        valid_mask: jax.Array,
        source_mean: jax.Array,
        target_mean: jax.Array,
    ) -> tuple[jax.Array, ContributionDistributedLossAux]:
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
            source, target, banks, weights, valid = inputs
            transcoder_params = jax.tree_util.tree_map(
                lambda value: value.astype(compute_dtype),
                differentiable_params,
            )
            params = transcoder_params_to_sae(transcoder_params)
            centered_source = source.astype(compute_dtype) - source_mean.astype(
                compute_dtype
            )
            # One source vector is intentionally the common encoder operand
            # for all eight selected slots.  No target-derived value appears
            # in this path.
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
                local_values, local_ids, overflow_here = _local_grouped_candidates(
                    params,
                    centered,
                    banks,
                    component_valid,
                    layout,
                    feature_axis=feature_axis,
                    capacity=capacity,
                    encoder_activation=encoder_activation,
                )
            else:
                local_values, local_ids = _local_rank_candidates(
                    params,
                    centered,
                    component_valid,
                    layout,
                    feature_axis=feature_axis,
                    encoder_activation=encoder_activation,
                )
                overflow_here = jnp.asarray(False)
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
            weights32 = weights.astype(jnp.float32)
            target_mean32 = target_mean.astype(jnp.float32)
            weighted_error = weights32[..., None] * (
                target32 - reconstruction.astype(jnp.float32)
            )
            weighted_centered = weights32[..., None] * (
                target32 - target_mean32[None, None, :]
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
            source_finite = jnp.all(jnp.isfinite(source) | ~valid[:, None])
            inputs_valid = source_finite & _scientific_input_valid(
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

        micro_fn = jax.checkpoint(one_micro) if rematerialize_microbatch else one_micro
        scanned, _ = jax.lax.scan(
            micro_fn,
            initial,
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
            topk_valid,
            fired,
            overflow,
        ) = scanned
        global_sse = jax.lax.pmean(jax.lax.psum(local_sse, data_axis), feature_axis)
        global_sst = jax.lax.pmean(jax.lax.psum(local_sst, data_axis), feature_axis)
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

    param_specs = transcoder_params_from_sae(_param_specs(feature_axis))
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
            param_specs,
            P(None, data_axis, None),
            P(None, data_axis, None, None),
            P(None, data_axis, None),
            P(None, data_axis, None),
            P(None, data_axis),
            P(None),
            P(None),
        ),
        out_specs=(P(), aux_specs),
        axis_names={data_axis, feature_axis},
        check_vma=False,
    )


class ContributionTranscoderDistributedTrainer:
    """One compiled source-only transcoder optimizer update."""

    def __init__(
        self,
        mesh: Mesh,
        layout: ContributionSAEConfig,
        optimizer: ContributionOptimizerConfig | None = None,
        distributed: DistributedConfig | None = None,
        *,
        bank_capacity: int | None = None,
        rematerialize_microbatch: bool = True,
        encoder_activation: str = ENCODER_ACTIVATION,
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
        self.rematerialize_microbatch = rematerialize_microbatch
        self.encoder_activation = encoder_activation
        _validate_contribution_mesh(mesh, layout, self.distributed)
        mapped = _build_mapped_objective(
            mesh,
            layout,
            self.distributed,
            bank_capacity,
            self.optimizer.parameter_dtype,
            rematerialize_microbatch,
            encoder_activation,
        )

        def complete_step(
            state: ContributionTranscoderTrainState,
            batch: ContributionTranscoderDistributedBatch,
            source_mean: jax.Array,
            target_mean: jax.Array,
        ) -> tuple[
            ContributionTranscoderTrainState,
            StepMetrics,
            ContributionDistributedLossAux,
            jax.Array,
        ]:
            differentiable = jax.tree_util.tree_map(
                lambda value: value.astype(jnp.float32),
                state.params,
            )
            (loss, aux), gradients = jax.value_and_grad(
                mapped,
                has_aux=True,
            )(
                differentiable,
                batch.routed_input,
                batch.expert_output,
                batch.bank_ids,
                batch.router_weights,
                batch.valid_mask,
                source_mean,
                target_mean,
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
            sae_state, metrics = apply_gradients(
                transcoder_state_to_sae(state),
                transcoder_params_to_sae(gradients),
                self.optimizer_config,
                loss=loss,
                valid_tokens=aux.valid_source_tokens,
                fired_mask=aux.fired_mask,
                l0=l0,
                sse=aux.sse,
                sst=aux.sst,
                topk_valid=aux.topk_valid & ~aux.capacity_overflow,
            )
            updated = transcoder_state_from_sae(sae_state)
            state_finite = jnp.asarray(True)
            for tree in (
                updated.params,
                updated.optimizer.m,
                updated.optimizer.v,
            ):
                for leaf in jax.tree_util.tree_leaves(tree):
                    state_finite = state_finite & jnp.all(jnp.isfinite(leaf))
            return updated, metrics, aux, state_finite

        self._compiled = jax.jit(
            complete_step,
            in_shardings=(
                _transcoder_state_shardings(mesh, self.distributed),
                contribution_transcoder_batch_shardings(mesh, self.distributed),
                NamedSharding(mesh, P()),
                NamedSharding(mesh, P()),
            ),
        )
        self._executable = None

    def _prepare(
        self,
        microbatches: Sequence[Any] | ContributionTranscoderDistributedBatch,
        centering: ContributionTranscoderCentering,
        *,
        process_local: bool,
    ) -> tuple[
        ContributionTranscoderDistributedBatch,
        ContributionTranscoderCentering,
    ]:
        if not isinstance(centering, ContributionTranscoderCentering):
            raise TypeError(
                "transcoder centering must explicitly name source and target means"
            )
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
        source_mean = replicate_process_local_array(
            jnp.asarray(centering.source_mean, jnp.float32),
            self.mesh,
        )
        target_mean = replicate_process_local_array(
            jnp.asarray(centering.target_mean, jnp.float32),
            self.mesh,
        )
        expected = (self.layout.hidden_dim,)
        if source_mean.shape != expected or target_mean.shape != expected:
            raise ValueError(
                f"both transcoder means must have shape [{self.layout.hidden_dim}]"
            )
        return placed, ContributionTranscoderCentering(
            source_mean,
            target_mean,
        )

    def compile(
        self,
        state: ContributionTranscoderTrainState,
        microbatches: Sequence[Any] | ContributionTranscoderDistributedBatch,
        centering: ContributionTranscoderCentering,
        *,
        process_local: bool = False,
    ) -> float:
        batch, means = self._prepare(
            microbatches,
            centering,
            process_local=process_local,
        )
        started = time.perf_counter()
        self._executable = self._compiled.lower(
            state,
            batch,
            means.source_mean,
            means.target_mean,
        ).compile()
        return time.perf_counter() - started

    def step(
        self,
        state: ContributionTranscoderTrainState,
        microbatches: Sequence[Any] | ContributionTranscoderDistributedBatch,
        centering: ContributionTranscoderCentering,
        *,
        process_local: bool = False,
    ) -> tuple[ContributionTranscoderTrainState, StepMetrics]:
        batch, means = self._prepare(
            microbatches,
            centering,
            process_local=process_local,
        )
        executable = self._compiled if self._executable is None else self._executable
        new_state, metrics, aux, state_finite = executable(
            state,
            batch,
            means.source_mean,
            means.target_mean,
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
            raise RuntimeError("A/R transcoder bank exceeded its precomputed capacity")
        if not bool(topk_valid):
            raise RuntimeError(
                "invalid transcoder inputs or fewer than k positive features"
            )
        token_count = int(source_tokens)
        if int(valid_components) != token_count * self.layout.top_routed:
            raise RuntimeError(
                "transcoder component count differs from source token count"
            )
        _validate_exact_distributed_l0(
            l0_sum=int(l0_sum),
            valid_tokens=token_count,
            k=self.layout.total_l0,
        )
        if not bool(metrics_finite) or not bool(state_finite):
            raise RuntimeError("transcoder update produced non-finite metric or state")
        return new_state, metrics


__all__ = [
    "ContributionTranscoderCentering",
    "ContributionTranscoderDistributedBatch",
    "ContributionTranscoderDistributedTrainer",
    "contribution_transcoder_batch_shardings",
    "init_contribution_transcoder_distributed_state",
    "shard_contribution_transcoder_batch",
    "shard_process_local_contribution_transcoder_batch",
    "stack_contribution_transcoder_microbatches",
]
