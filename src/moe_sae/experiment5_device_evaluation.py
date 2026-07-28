"""TPU-native, parameter-sharded evaluation for Experiment 5.

The production evaluator deliberately shares Experiment 5's distributed
candidate scoring, deterministic TopK, sparse decoder, and exact Gemma
recomposition kernels with training.  Checkpoint parameters remain in their
balanced physical feature layout on device.  The only host-visible values are
compact scientific sufficient rows:

* per-token full SSE/SST and the 64 canonical logical latent IDs;
* for H/R, routed/shared/oracle SSEs and the eight routed bank IDs; and
* per-data-lane routed/shared target sums and squared-norm sums.

No parameter tree, activation target, or reconstruction is gathered to a
host.  ``process_local_experiment5_device_scores`` removes feature-axis
replicas and flattens one process's data lanes for the strict artifact packer
in :mod:`moe_sae.experiment5_evaluation`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from moe_sae.distributed import (
    DistributedConfig,
    _param_specs,
    _validate_exact_distributed_l0,
    replicate_process_local_array,
)
from moe_sae.experiment5_distributed import (
    Experiment5DistributedBatch,
    Experiment5PhysicalLayout,
    _decode_logical_selection,
    _local_fired_mask,
    _local_global_candidates,
    _local_grouped_routed_candidates,
    _local_shared_candidates,
    _merge_local_candidates,
    _scientific_inputs_valid,
    _validate_experiment5_mesh,
    _validate_stacked_batch,
    experiment5_batch_shardings,
    require_experiment5_physical_layout,
    shard_experiment5_batch,
    shard_process_local_experiment5_batch,
    stack_experiment5_distributed_microbatches,
)
from moe_sae.experiment5_hybrid import (
    Experiment5Centering,
    Experiment5EncoderKind,
    Experiment5Layout,
    Experiment5Recomposition,
    jax_experiment5_routed_branch,
    jax_recompose_experiment5_full_ffn,
)
from moe_sae.sae import SAEParams, Selection, TrainState

_INVALID_ID = np.int32(-1)


class Experiment5DeviceScores(NamedTuple):
    """Exact compact device outputs consumed by the evidence packer."""

    selected_latent_ids: Any
    full_sse: Any
    full_sst: Any
    bank_ids: Any | None
    routed_sse: Any | None
    shared_sse: Any | None
    oracle_shared_full_sse: Any | None
    oracle_routed_full_sse: Any | None
    routed_target_sum: Any | None
    routed_target_squared_norm_sum: Any | None
    shared_target_sum: Any | None
    shared_target_squared_norm_sum: Any | None


class Experiment5DeviceEvaluationHealth(NamedTuple):
    """Small replicated fail-closed health record for one evaluation chunk."""

    valid_tokens: jax.Array
    l0_sum: jax.Array
    topk_valid: jax.Array
    inputs_valid: jax.Array
    parameters_finite: jax.Array
    outputs_finite: jax.Array
    capacity_overflow: jax.Array


def _masked_token_sse(
    target: jax.Array,
    estimate: jax.Array,
    valid: jax.Array,
) -> jax.Array:
    value = jnp.sum(
        jax.lax.square(
            target.astype(jnp.float32) - estimate.astype(jnp.float32)
        ),
        axis=-1,
        dtype=jnp.float32,
    )
    return jnp.where(valid, value, jnp.asarray(0.0, jnp.float32))


def _masked_target_sufficient_stats(
    target: jax.Array,
    valid: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    target32 = target.astype(jnp.float32)
    masked = jnp.where(valid[:, None], target32, 0.0)
    return (
        jnp.sum(masked, axis=0, dtype=jnp.float32),
        jnp.sum(jax.lax.square(masked), dtype=jnp.float32),
    )


def _build_experiment5_mapped_evaluator(
    mesh: Mesh,
    layout: Experiment5Layout,
    physical: Experiment5PhysicalLayout,
    distributed: DistributedConfig,
    *,
    encoder_kind: Experiment5EncoderKind,
    bank_capacity: int | None,
    epsilon: float,
):
    data_axis = distributed.data_axis
    feature_axis = distributed.feature_axis
    if layout.is_hybrid:
        if bank_capacity is None or bank_capacity <= 0:
            raise ValueError(
                "distributed H/R evaluation requires positive routed-bank capacity"
            )
    elif bank_capacity is not None:
        raise ValueError("B/F global evaluation does not accept bank capacity")
    capacity = int(bank_capacity or 1)
    activation = "relu" if encoder_kind == "sae" else "softplus"

    def local_evaluate(
        params: SAEParams,
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
    ) -> tuple[Experiment5DeviceScores, Experiment5DeviceEvaluationHealth]:
        recomposition = Experiment5Recomposition(
            routed_scale=routed_scale,
            outer_scale=outer_scale,
            epsilon=epsilon,
        )
        parameter_dtype = params.encoder.dtype

        def one_micro(inputs):
            source, expert, shared, banks, weights, valid = inputs
            target = jax_recompose_experiment5_full_ffn(
                expert,
                shared,
                weights,
                recomposition,
            )
            target32 = target.astype(jnp.float32)
            full_mean32 = full_target_mean.astype(jnp.float32)
            full_sst = jnp.where(
                valid,
                jnp.sum(
                    jax.lax.square(target32 - full_mean32[None, :]),
                    axis=1,
                    dtype=jnp.float32,
                ),
                jnp.asarray(0.0, jnp.float32),
            )

            if layout.is_hybrid:
                if encoder_kind == "sae":
                    routed_centered = (
                        expert.astype(parameter_dtype)
                        - routed_target_mean.astype(parameter_dtype)[
                            None, None, :
                        ]
                    )
                    shared_centered = (
                        shared.astype(parameter_dtype)
                        - shared_target_mean.astype(parameter_dtype)[None, :]
                    )
                else:
                    common_centered = (
                        source.astype(parameter_dtype)
                        - source_mean.astype(parameter_dtype)[None, :]
                    )
                    routed_centered = jnp.broadcast_to(
                        common_centered[:, None, :],
                        expert.shape,
                    )
                    shared_centered = common_centered
                component_valid = jnp.broadcast_to(valid[:, None], banks.shape)
                local_routed_values, local_routed_ids, overflow = (
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
                routed_hat = _decode_logical_selection(
                    params.decoder,
                    routed_selection,
                    routed_target_mean,
                    layout,
                    physical,
                    feature_axis=feature_axis,
                ).reshape(expert.shape)
                shared_hat = _decode_logical_selection(
                    params.decoder,
                    shared_selection,
                    shared_target_mean,
                    layout,
                    physical,
                    feature_axis=feature_axis,
                )
                reconstruction = jax_recompose_experiment5_full_ffn(
                    routed_hat,
                    shared_hat,
                    weights,
                    recomposition,
                )
                routed_target = jax_experiment5_routed_branch(
                    expert,
                    weights,
                    recomposition,
                )
                routed_estimate = jax_experiment5_routed_branch(
                    routed_hat,
                    weights,
                    recomposition,
                )
                oracle_shared = jax_recompose_experiment5_full_ffn(
                    routed_hat,
                    shared,
                    weights,
                    recomposition,
                )
                oracle_routed = jax_recompose_experiment5_full_ffn(
                    expert,
                    shared_hat,
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
                routed_ids = routed_selection.latent_ids.reshape(
                    source.shape[0],
                    layout.top_routed * layout.routed_k,
                )
                selected_ids = jnp.concatenate(
                    (routed_ids, shared_selection.latent_ids),
                    axis=1,
                )
                routed_l0 = jnp.sum(
                    routed_scientific.reshape(
                        source.shape[0],
                        layout.top_routed * layout.routed_k,
                    ),
                    axis=1,
                    dtype=jnp.int32,
                )
                shared_l0 = jnp.sum(
                    shared_scientific,
                    axis=1,
                    dtype=jnp.int32,
                )
                token_l0 = routed_l0 + shared_l0
                topk_valid = (
                    jnp.all(
                        routed_selection.valid | ~flat_component_valid
                    )
                    & jnp.all(shared_selection.valid | ~valid)
                )
                routed_sum, routed_squared = (
                    _masked_target_sufficient_stats(routed_target, valid)
                )
                shared_sum, shared_squared = (
                    _masked_target_sufficient_stats(shared, valid)
                )
                row_scores = (
                    _masked_token_sse(
                        routed_target,
                        routed_estimate,
                        valid,
                    ),
                    _masked_token_sse(shared, shared_hat, valid),
                    _masked_token_sse(target, oracle_shared, valid),
                    _masked_token_sse(target, oracle_routed, valid),
                )
                output_banks = jnp.where(
                    valid[:, None],
                    banks.astype(jnp.int32),
                    jnp.asarray(_INVALID_ID, jnp.int32),
                )
            else:
                if encoder_kind == "sae":
                    global_source = target
                    global_mean = full_target_mean
                else:
                    global_source = source
                    global_mean = source_mean
                centered = (
                    global_source.astype(parameter_dtype)
                    - global_mean.astype(parameter_dtype)[None, :]
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
                selected_ids = selection.latent_ids
                token_l0 = jnp.sum(
                    scientific,
                    axis=1,
                    dtype=jnp.int32,
                )
                topk_valid = jnp.all(selection.valid | ~valid)
                overflow = jnp.asarray(False)
                output_banks = None
                row_scores = (None, None, None, None)
                routed_sum = None
                routed_squared = None
                shared_sum = None
                shared_squared = None

            selected_ids = jnp.where(
                valid[:, None],
                selected_ids.astype(jnp.int32),
                jnp.asarray(_INVALID_ID, jnp.int32),
            )
            full_sse = _masked_token_sse(target, reconstruction, valid)
            inputs_valid = _scientific_inputs_valid(
                source,
                expert,
                shared,
                banks,
                weights,
                valid,
                layout,
            )
            exact_l0 = jnp.all(
                (token_l0 == layout.total_l0) | ~valid
            )
            values_for_finiteness = (
                full_sse,
                full_sst,
                *tuple(value for value in row_scores if value is not None),
            )
            outputs_finite = jnp.asarray(True)
            for value in values_for_finiteness:
                outputs_finite = outputs_finite & jnp.all(jnp.isfinite(value))
            return (
                selected_ids,
                full_sse,
                full_sst,
                output_banks,
                *row_scores,
                routed_sum,
                routed_squared,
                shared_sum,
                shared_squared,
                jnp.sum(valid, dtype=jnp.int32),
                jnp.sum(
                    jnp.where(valid, token_l0, 0),
                    dtype=jnp.int32,
                ),
                topk_valid & exact_l0,
                inputs_valid,
                outputs_finite,
                overflow,
            )

        scanned = jax.lax.map(
            one_micro,
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
            selected_ids,
            full_sse,
            full_sst,
            output_banks,
            routed_sse,
            shared_sse,
            oracle_shared_sse,
            oracle_routed_sse,
            routed_sums,
            routed_squared,
            shared_sums,
            shared_squared,
            valid_tokens,
            l0_sums,
            topk_valid,
            inputs_valid,
            outputs_finite,
            overflow,
        ) = scanned

        if layout.is_hybrid:
            routed_target_sum = jnp.sum(
                routed_sums,
                axis=0,
                dtype=jnp.float32,
            )[None, :]
            routed_target_squared_norm_sum = jnp.sum(
                routed_squared,
                dtype=jnp.float32,
            )[None]
            shared_target_sum = jnp.sum(
                shared_sums,
                axis=0,
                dtype=jnp.float32,
            )[None, :]
            shared_target_squared_norm_sum = jnp.sum(
                shared_squared,
                dtype=jnp.float32,
            )[None]
        else:
            routed_target_sum = None
            routed_target_squared_norm_sum = None
            shared_target_sum = None
            shared_target_squared_norm_sum = None

        scores = Experiment5DeviceScores(
            selected_latent_ids=selected_ids,
            full_sse=full_sse,
            full_sst=full_sst,
            bank_ids=output_banks,
            routed_sse=routed_sse,
            shared_sse=shared_sse,
            oracle_shared_full_sse=oracle_shared_sse,
            oracle_routed_full_sse=oracle_routed_sse,
            routed_target_sum=routed_target_sum,
            routed_target_squared_norm_sum=routed_target_squared_norm_sum,
            shared_target_sum=shared_target_sum,
            shared_target_squared_norm_sum=shared_target_squared_norm_sum,
        )

        local_parameter_finite = jnp.asarray(True)
        for leaf in jax.tree_util.tree_leaves(params):
            local_parameter_finite = (
                local_parameter_finite & jnp.all(jnp.isfinite(leaf))
            )
        global_valid_tokens = jax.lax.pmean(
            jax.lax.psum(
                jnp.sum(valid_tokens, dtype=jnp.float32),
                data_axis,
            ),
            feature_axis,
        ).astype(jnp.int32)
        global_l0_sum = jax.lax.pmean(
            jax.lax.psum(
                jnp.sum(l0_sums, dtype=jnp.float32),
                data_axis,
            ),
            feature_axis,
        ).astype(jnp.int32)

        def global_all(value: jax.Array) -> jax.Array:
            return (
                jax.lax.pmin(
                    jnp.all(value).astype(jnp.int32),
                    (data_axis, feature_axis),
                )
                > 0
            )

        global_overflow = (
            jax.lax.pmax(
                jnp.any(overflow).astype(jnp.int32),
                (data_axis, feature_axis),
            )
            > 0
        )
        health = Experiment5DeviceEvaluationHealth(
            valid_tokens=global_valid_tokens,
            l0_sum=global_l0_sum,
            topk_valid=global_all(topk_valid),
            inputs_valid=global_all(inputs_valid),
            parameters_finite=global_all(local_parameter_finite),
            outputs_finite=global_all(outputs_finite),
            capacity_overflow=global_overflow,
        )
        return scores, health

    data_rows = P(None, data_axis)
    data_ids = P(None, data_axis, None)
    if layout.is_hybrid:
        branch_rows: Any = data_rows
        bank_specs: Any = data_ids
        target_sum_specs: Any = P(data_axis, None)
        target_squared_specs: Any = P(data_axis)
    else:
        branch_rows = None
        bank_specs = None
        target_sum_specs = None
        target_squared_specs = None
    score_specs = Experiment5DeviceScores(
        selected_latent_ids=data_ids,
        full_sse=data_rows,
        full_sst=data_rows,
        bank_ids=bank_specs,
        routed_sse=branch_rows,
        shared_sse=branch_rows,
        oracle_shared_full_sse=branch_rows,
        oracle_routed_full_sse=branch_rows,
        routed_target_sum=target_sum_specs,
        routed_target_squared_norm_sum=target_squared_specs,
        shared_target_sum=target_sum_specs,
        shared_target_squared_norm_sum=target_squared_specs,
    )
    health_specs = Experiment5DeviceEvaluationHealth(
        *(P() for _ in Experiment5DeviceEvaluationHealth._fields)
    )
    return jax.shard_map(
        local_evaluate,
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
        out_specs=(score_specs, health_specs),
        axis_names={data_axis, feature_axis},
        check_vma=False,
    )


def _addressable_concat(value: jax.Array, *, axis: int) -> np.ndarray:
    """Remove feature replicas and concatenate this process's data shards."""

    if value.is_fully_addressable:
        return np.asarray(value)
    unique: dict[tuple[Any, ...], tuple[int, np.ndarray]] = {}
    for shard in value.addressable_shards:
        normalized = tuple(
            (
                item.start,
                item.stop,
                item.step,
            )
            if isinstance(item, slice)
            else int(item)
            for item in shard.index
        )
        axis_index = shard.index[axis]
        start = axis_index.start if isinstance(axis_index, slice) else axis_index
        unique.setdefault(normalized, (int(start or 0), np.asarray(shard.data)))
    if not unique:
        raise RuntimeError("evaluation output has no addressable shards")
    ordered = [item[1] for item in sorted(unique.values(), key=lambda item: item[0])]
    return ordered[0] if len(ordered) == 1 else np.concatenate(ordered, axis=axis)


def process_local_experiment5_device_scores(
    scores: Experiment5DeviceScores,
) -> Experiment5DeviceScores:
    """Materialize only this host's compact rows in microbatch-major order."""

    def rows(value: Any | None) -> np.ndarray | None:
        if value is None:
            return None
        local = _addressable_concat(value, axis=1)
        return np.ascontiguousarray(
            local.reshape((-1, *local.shape[2:])),
        )

    def target_sum(value: Any | None) -> np.ndarray | None:
        if value is None:
            return None
        local = _addressable_concat(value, axis=0)
        return np.ascontiguousarray(
            np.sum(local, axis=0, dtype=np.float64),
            dtype=np.float64,
        )

    def target_squared(value: Any | None) -> float | None:
        if value is None:
            return None
        local = _addressable_concat(value, axis=0)
        return float(np.sum(local, dtype=np.float64))

    return Experiment5DeviceScores(
        selected_latent_ids=rows(scores.selected_latent_ids),
        full_sse=rows(scores.full_sse),
        full_sst=rows(scores.full_sst),
        bank_ids=rows(scores.bank_ids),
        routed_sse=rows(scores.routed_sse),
        shared_sse=rows(scores.shared_sse),
        oracle_shared_full_sse=rows(scores.oracle_shared_full_sse),
        oracle_routed_full_sse=rows(scores.oracle_routed_full_sse),
        routed_target_sum=target_sum(scores.routed_target_sum),
        routed_target_squared_norm_sum=target_squared(
            scores.routed_target_squared_norm_sum
        ),
        shared_target_sum=target_sum(scores.shared_target_sum),
        shared_target_squared_norm_sum=target_squared(
            scores.shared_target_squared_norm_sum
        ),
    )


class Experiment5DistributedEvaluator:
    """Compiled production evaluation with physical-sharded parameters."""

    def __init__(
        self,
        mesh: Mesh,
        layout: Experiment5Layout,
        distributed: DistributedConfig | None = None,
        *,
        encoder_kind: Experiment5EncoderKind,
        bank_capacity: int | None = None,
        rmsnorm_epsilon: float = 1e-6,
        expected_physical_layout_sha256: str | None = None,
        require_pre_sharded_params: bool = True,
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
        if type(require_pre_sharded_params) is not bool:
            raise TypeError("require_pre_sharded_params must be boolean")
        self.mesh = mesh
        self.layout = layout
        self.distributed = distributed or DistributedConfig()
        self.encoder_kind = encoder_kind
        self.bank_capacity = bank_capacity
        self.rmsnorm_epsilon = float(rmsnorm_epsilon)
        self.require_pre_sharded_params = require_pre_sharded_params
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
        mapped = _build_experiment5_mapped_evaluator(
            mesh,
            layout,
            self.physical_layout,
            self.distributed,
            encoder_kind=encoder_kind,
            bank_capacity=bank_capacity,
            epsilon=self.rmsnorm_epsilon,
        )

        def complete(
            params: SAEParams,
            batch: Experiment5DistributedBatch,
            source_mean: jax.Array,
            routed_target_mean: jax.Array,
            shared_target_mean: jax.Array,
            full_target_mean: jax.Array,
            routed_scale: jax.Array,
            outer_scale: jax.Array,
        ):
            return mapped(
                params,
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

        self._param_shardings = jax.tree_util.tree_map(
            lambda spec: NamedSharding(mesh, spec),
            _param_specs(self.distributed.feature_axis),
        )
        replicated = NamedSharding(mesh, P())
        self._compiled = jax.jit(
            complete,
            in_shardings=(
                self._param_shardings,
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

    def _validate_params(self, params: SAEParams) -> None:
        expected_shapes = {
            "encoder": (self.layout.total_features, self.layout.hidden_dim),
            "decoder": (self.layout.hidden_dim, self.layout.total_features),
            "bias": (self.layout.total_features,),
        }
        for name in SAEParams._fields:
            value = getattr(params, name)
            if not isinstance(value, jax.Array):
                raise TypeError(f"{name} must be a JAX array")
            if value.shape != expected_shapes[name]:
                raise ValueError(
                    f"{name} shape {value.shape} differs from "
                    f"{expected_shapes[name]}"
                )
            if not jnp.issubdtype(value.dtype, jnp.floating):
                raise TypeError(f"{name} must be floating-point")
            if self.require_pre_sharded_params:
                expected = getattr(self._param_shardings, name)
                if not value.sharding.is_equivalent_to(expected, value.ndim):
                    raise ValueError(
                        f"{name} is not in the required physical feature sharding"
                    )

    def _prepare(
        self,
        microbatches: Sequence[Any] | Experiment5DistributedBatch,
        centering: Experiment5Centering,
        recomposition: Experiment5Recomposition,
        *,
        process_local: bool,
    ) -> tuple[Experiment5DistributedBatch, tuple[jax.Array, ...]]:
        if not isinstance(centering, Experiment5Centering):
            raise TypeError("centering must be Experiment5Centering")
        if not isinstance(recomposition, Experiment5Recomposition):
            raise TypeError("recomposition must be Experiment5Recomposition")
        if float(recomposition.epsilon) != self.rmsnorm_epsilon:
            raise ValueError(
                "recomposition epsilon differs from the compiled evaluator"
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

        def vector(name: str, value: Any) -> jax.Array:
            result = replicate_process_local_array(
                jnp.asarray(value, jnp.float32),
                self.mesh,
            )
            if result.shape != (self.layout.hidden_dim,):
                raise ValueError(
                    f"{name} must have shape [{self.layout.hidden_dim}]"
                )
            return result

        constants = (
            vector("source_mean", centering.source_mean),
            vector("routed_target_mean", centering.routed_target_mean),
            vector("shared_target_mean", centering.shared_target_mean),
            vector("full_target_mean", centering.full_target_mean),
            vector("routed_scale", recomposition.routed_scale),
            vector("outer_scale", recomposition.outer_scale),
        )
        return placed, constants

    def compile(
        self,
        state_or_params: TrainState | SAEParams,
        microbatches: Sequence[Any] | Experiment5DistributedBatch,
        centering: Experiment5Centering,
        recomposition: Experiment5Recomposition,
        *,
        process_local: bool = False,
    ) -> float:
        """Collectively compile evaluation without materializing score rows."""

        params = (
            state_or_params.params
            if isinstance(state_or_params, TrainState)
            else state_or_params
        )
        self._validate_params(params)
        batch, constants = self._prepare(
            microbatches,
            centering,
            recomposition,
            process_local=process_local,
        )
        started = time.perf_counter()
        self._executable = self._compiled.lower(
            params,
            batch,
            *constants,
        ).compile()
        return time.perf_counter() - started

    def score(
        self,
        state_or_params: TrainState | SAEParams,
        microbatches: Sequence[Any] | Experiment5DistributedBatch,
        centering: Experiment5Centering,
        recomposition: Experiment5Recomposition,
        *,
        process_local: bool = False,
    ) -> Experiment5DeviceScores:
        """Score one chunk and synchronously reject corrupt scientific output."""

        params = (
            state_or_params.params
            if isinstance(state_or_params, TrainState)
            else state_or_params
        )
        self._validate_params(params)
        batch, constants = self._prepare(
            microbatches,
            centering,
            recomposition,
            process_local=process_local,
        )
        executable = self._compiled if self._executable is None else self._executable
        scores, health = executable(params, batch, *constants)
        checked = jax.device_get(health)
        if bool(checked.capacity_overflow):
            raise RuntimeError(
                "H/R routed bank exceeded evaluation grouped-GEMM capacity"
            )
        if not bool(checked.inputs_valid):
            raise RuntimeError("Experiment-5 evaluation inputs are invalid")
        if not bool(checked.topk_valid):
            raise RuntimeError(
                "Experiment-5 evaluation did not select exactly 64 positive "
                "features per scientific token"
            )
        if not bool(checked.parameters_finite):
            raise RuntimeError("Experiment-5 evaluation parameters are non-finite")
        if not bool(checked.outputs_finite):
            raise RuntimeError("Experiment-5 evaluation outputs are non-finite")
        _validate_exact_distributed_l0(
            l0_sum=int(checked.l0_sum),
            valid_tokens=int(checked.valid_tokens),
            k=self.layout.total_l0,
        )
        return scores


__all__ = [
    "Experiment5DeviceEvaluationHealth",
    "Experiment5DeviceScores",
    "Experiment5DistributedEvaluator",
    "process_local_experiment5_device_scores",
]
