"""Production feature-sharded JAX execution for the Gemma 4 SAE.

The default v4-32 mesh is ``process_count × local_device_count``: the data axis
crosses hosts and the feature axis stays within each host.  Parameters and
FP32 Adam moments are sharded on the feature axis and replicated on the data
axis.  This keeps the latency-sensitive TopK all-gather and sparse decoder
psum host-local while using a cross-host reduction only for data-parallel
gradients/statistics.

Orbax is imported lazily because checkpointing is a TPU deployment concern;
the single-device correctness reference remains usable with base dependencies.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from moe_sae.artifacts import ArtifactDataset, bf16_bits_to_float32
from moe_sae.pipeline import (
    ActivationBatch,
    EvaluationResult,
    HealthGateResult,
    TrainingConfig,
    _fixed_microbatches,
    _scientific_rows,
    check_seed_health,
)
from moe_sae.sae import (
    AdamState,
    SAEConfig,
    SAEParams,
    StepMetrics,
    TrainState,
    apply_gradients,
    decode_selected_columns_sharded,
    feature_sharded_topk,
    init_train_state,
)
from moe_sae.runtime import atomic_write_json


@dataclass(frozen=True)
class DistributedConfig:
    data_axis: str = "data"
    feature_axis: str = "feature"
    data_axis_size: int | None = None
    feature_axis_size: int | None = None
    microbatches_per_update: int = 4

    def __post_init__(self) -> None:
        if self.data_axis == self.feature_axis:
            raise ValueError("data and feature axes must have distinct names")
        if self.data_axis_size is not None and self.data_axis_size <= 0:
            raise ValueError("data_axis_size must be positive")
        if self.feature_axis_size is not None and self.feature_axis_size <= 0:
            raise ValueError("feature_axis_size must be positive")
        if self.microbatches_per_update <= 0:
            raise ValueError("microbatches_per_update must be positive")


class DistributedBatch(NamedTuple):
    """Static stacked microbatches, with batch dimension second."""

    activation: jax.Array  # [microbatches, global_microbatch, hidden]
    expert_ids: jax.Array  # [microbatches, global_microbatch, top_routed]
    valid_mask: jax.Array  # [microbatches, global_microbatch]


class DistributedLossAux(NamedTuple):
    sse: jax.Array
    sst: jax.Array
    l0_sum: jax.Array
    valid_tokens: jax.Array
    topk_valid: jax.Array
    fired_mask: jax.Array  # globally shaped, feature sharded
    capacity_overflow: jax.Array


@dataclass(frozen=True)
class DistributedRunTiming:
    compile_seconds: float
    optimizer_wall_seconds: float
    checkpoint_seconds: float
    global_chip_count: int
    tokens: int
    steps: int

    @property
    def chip_seconds(self) -> float:
        """Primary marginal SAE cost; compile/checkpoint time is excluded."""

        return self.optimizer_wall_seconds * self.global_chip_count

    @property
    def tokens_per_second(self) -> float:
        return (
            self.tokens / self.optimizer_wall_seconds
            if self.optimizer_wall_seconds
            else 0.0
        )


@dataclass(frozen=True)
class DistributedTrainingResult:
    state: TrainState
    metrics: tuple[StepMetrics, ...]
    step_optimizer_seconds: tuple[float, ...]
    health: HealthGateResult | None
    timing: DistributedRunTiming
    latest_checkpoint: str | None
    final_checkpoint: str | None

    @property
    def compile_seconds(self) -> float:
        return self.timing.compile_seconds

    @property
    def optimizer_wall_seconds(self) -> float:
        return self.timing.optimizer_wall_seconds

    @property
    def checkpoint_seconds(self) -> float:
        return self.timing.checkpoint_seconds

    @property
    def chip_count(self) -> int:
        return self.timing.global_chip_count


def create_mesh(
    config: DistributedConfig | None = None,
    devices: Sequence[jax.Device] | None = None,
) -> Mesh:
    """Create a process-major 2D mesh whose feature rows are host-local.

    The production default is ``jax.process_count() ×
    jax.local_device_count()``.  Explicit sizes are supported for tests and
    nonstandard slices, but the device ordering still groups each process's
    devices contiguously whenever it matches those sizes.
    """

    config = config or DistributedConfig()
    available = list(jax.devices() if devices is None else devices)
    if not available:
        raise RuntimeError("JAX reported no devices")
    data_size = config.data_axis_size or jax.process_count()
    feature_size = config.feature_axis_size or jax.local_device_count()
    if data_size * feature_size != len(available):
        raise ValueError(
            f"mesh {data_size}×{feature_size} requires {data_size * feature_size} "
            f"devices, found {len(available)}"
        )

    # On a real multi-host slice, explicitly group and sort by process so each
    # feature-axis row is host-local.  Virtual/custom test meshes may place
    # multiple data rows on one process and fall back to supplied order.
    by_process: dict[int, list[jax.Device]] = {}
    for device in available:
        by_process.setdefault(int(device.process_index), []).append(device)
    if len(by_process) == data_size and all(
        len(group) == feature_size for group in by_process.values()
    ):
        rows = []
        for process in sorted(by_process):
            rows.append(sorted(by_process[process], key=lambda device: int(device.id)))
        grid = np.asarray(rows, dtype=object)
    else:
        grid = np.asarray(available, dtype=object).reshape(data_size, feature_size)
    return Mesh(grid, (config.data_axis, config.feature_axis))


def _param_specs(feature_axis: str) -> SAEParams:
    return SAEParams(P(feature_axis, None), P(None, feature_axis), P(feature_axis))


def _state_specs(feature_axis: str) -> TrainState:
    params = _param_specs(feature_axis)
    moments = _param_specs(feature_axis)
    return TrainState(
        params=params,
        optimizer=AdamState(moments, moments, P()),
        step=P(),
        tokens_seen=P(),
        fired_mask=P(feature_axis),
        loss_ema=P(),
        initial_loss_ema=P(),
        ema_updates=P(),
    )


def _named(tree: Any, mesh: Mesh) -> Any:
    return jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), tree)


def state_shardings(mesh: Mesh, config: DistributedConfig) -> TrainState:
    return _named(_state_specs(config.feature_axis), mesh)


def batch_shardings(mesh: Mesh, config: DistributedConfig) -> DistributedBatch:
    return DistributedBatch(
        NamedSharding(mesh, P(None, config.data_axis, None)),
        NamedSharding(mesh, P(None, config.data_axis, None)),
        NamedSharding(mesh, P(None, config.data_axis)),
    )


def shard_train_state(
    state: TrainState,
    mesh: Mesh,
    sae_config: SAEConfig,
    distributed: DistributedConfig | None = None,
) -> TrainState:
    """Place an existing correctness-reference state on the production mesh."""

    distributed = distributed or DistributedConfig()
    _validate_mesh_model(mesh, sae_config, distributed)
    shardings = state_shardings(mesh, distributed)
    return jax.tree_util.tree_map(jax.device_put, state, shardings)


def init_distributed_state(
    key: jax.Array,
    mesh: Mesh,
    sae_config: SAEConfig,
    distributed: DistributedConfig | None = None,
) -> TrainState:
    """Initialize directly into sharded arrays without a full host materialization."""

    distributed = distributed or DistributedConfig()
    _validate_mesh_model(mesh, sae_config, distributed)
    output_shardings = state_shardings(mesh, distributed)
    initialize = jax.jit(
        lambda rng: init_train_state(rng, sae_config),
        in_shardings=NamedSharding(mesh, P()),
        out_shardings=output_shardings,
    )
    return initialize(jax.device_put(key, NamedSharding(mesh, P())))


def _validate_mesh_model(
    mesh: Mesh, sae_config: SAEConfig, distributed: DistributedConfig
) -> None:
    shape = mesh.shape
    if distributed.data_axis not in shape or distributed.feature_axis not in shape:
        raise ValueError("mesh does not contain configured data/feature axes")
    feature_size = int(shape[distributed.feature_axis])
    if sae_config.total_features % feature_size:
        raise ValueError("dictionary width must divide evenly across feature shards")
    local_features = sae_config.total_features // feature_size
    if local_features < sae_config.k:
        raise ValueError("every feature shard must own at least k features")
    if sae_config.condition == "A":
        if sae_config.num_experts % feature_size:
            raise ValueError("condition A experts must divide across feature shards")
        if local_features % sae_config.bank_width:
            raise ValueError("feature shard boundary must align to whole expert banks")


def _batch_field(batch: Any, name: str, default: Any = None) -> Any:
    if isinstance(batch, Mapping):
        return batch.get(name, default)
    return getattr(batch, name, default)


def stack_microbatches(
    microbatches: Sequence[Any],
    sae_config: SAEConfig,
    distributed: DistributedConfig | None = None,
) -> DistributedBatch:
    """Validate/decode already-static microbatches for distributed execution."""

    distributed = distributed or DistributedConfig()
    if len(microbatches) != distributed.microbatches_per_update:
        raise ValueError(
            f"expected {distributed.microbatches_per_update} microbatches, "
            f"got {len(microbatches)}"
        )
    activations: list[np.ndarray] = []
    ids_list: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    batch_size: int | None = None
    for batch in microbatches:
        activation = np.asarray(_batch_field(batch, "activation"))
        if activation.dtype == np.uint16:
            activation = bf16_bits_to_float32(activation)
        if activation.ndim != 2 or activation.shape[1] != sae_config.hidden_dim:
            raise ValueError(
                f"activation must be [tokens,{sae_config.hidden_dim}], got {activation.shape}"
            )
        if batch_size is None:
            batch_size = activation.shape[0]
        if activation.shape[0] != batch_size:
            raise ValueError("all microbatches must have the same static token count")
        mask_value = _batch_field(batch, "valid_mask")
        valid = (
            np.ones((batch_size,), dtype=bool)
            if mask_value is None
            else np.asarray(mask_value, dtype=bool)
        )
        if valid.shape != (batch_size,):
            raise ValueError("valid_mask shape mismatch")

        ids_value = _batch_field(batch, "expert_ids")
        if sae_config.condition == "A":
            if ids_value is None:
                raise ValueError("condition A requires expert_ids")
            ids = np.asarray(ids_value, dtype=np.int32)
            if ids.shape != (batch_size, sae_config.top_routed):
                raise ValueError("expert_ids shape mismatch")
            scientific = ids[valid]
            if scientific.size:
                if scientific.min() < 0 or scientific.max() >= sae_config.num_experts:
                    raise ValueError("expert id outside configured range")
                if np.any(np.diff(np.sort(scientific, axis=1), axis=1) == 0):
                    raise ValueError(
                        "router expert ids must be unique per scientific token"
                    )
        else:
            ids = np.zeros((batch_size, sae_config.top_routed), dtype=np.int32)
        activations.append(activation)
        ids_list.append(ids)
        masks.append(valid)
    return DistributedBatch(
        np.stack(activations),
        np.asarray(np.stack(ids_list), np.int32),
        np.asarray(np.stack(masks), bool),
    )


def shard_batch(
    batch: DistributedBatch, mesh: Mesh, distributed: DistributedConfig
) -> DistributedBatch:
    data_size = int(mesh.shape[distributed.data_axis])
    if batch.activation.shape[1] % data_size:
        raise ValueError("global microbatch must divide across the data axis")
    return jax.tree_util.tree_map(
        jax.device_put, batch, batch_shardings(mesh, distributed)
    )


def shard_process_local_batch(
    batch: DistributedBatch, mesh: Mesh, distributed: DistributedConfig
) -> DistributedBatch:
    """Assemble a global data-sharded batch from process-local host rows."""

    data_size = int(mesh.shape[distributed.data_axis])
    local_rows = batch.activation.shape[1]
    shardings = batch_shardings(mesh, distributed)
    return DistributedBatch(
        jax.make_array_from_process_local_data(
            shardings.activation,
            np.asarray(batch.activation),
            global_shape=(
                batch.activation.shape[0],
                local_rows * data_size,
                batch.activation.shape[2],
            ),
        ),
        jax.make_array_from_process_local_data(
            shardings.expert_ids,
            np.asarray(batch.expert_ids),
            global_shape=(
                batch.expert_ids.shape[0],
                local_rows * data_size,
                batch.expert_ids.shape[2],
            ),
        ),
        jax.make_array_from_process_local_data(
            shardings.valid_mask,
            np.asarray(batch.valid_mask),
            global_shape=(batch.valid_mask.shape[0], local_rows * data_size),
        ),
    )


def _local_a_scores(
    params: SAEParams,
    activation: jax.Array,
    expert_ids: jax.Array,
    sae: SAEConfig,
    *,
    feature_axis: str,
    feature_size: int,
    capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute only locally owned routed banks with a fixed-capacity GEMM."""

    local_features = params.encoder.shape[0]
    local_experts = local_features // sae.bank_width
    expert_start = jax.lax.axis_index(feature_axis) * local_experts
    flat_ids = expert_ids.reshape(-1)
    repeated = jnp.repeat(activation[:, None, :], sae.top_routed, axis=1).reshape(
        -1, sae.hidden_dim
    )
    local_id = flat_ids - expert_start
    owned = (local_id >= 0) & (local_id < local_experts)
    safe_local_id = jnp.clip(local_id, 0, local_experts - 1)

    # Rank assignments within each local expert without a host-side sort.  The
    # [token×route, local_experts] bookkeeping is small (16 experts/chip on the
    # default 4-way feature mesh) and the expensive expert matmul is performed
    # exactly once per owned routed bank.
    membership = jax.nn.one_hot(safe_local_id, local_experts, dtype=jnp.int32) * owned[
        :, None
    ].astype(jnp.int32)
    cumulative = jnp.cumsum(membership, axis=0) - 1
    rank = jnp.sum(cumulative * membership, axis=1)
    overflow = jnp.any(owned & (rank >= capacity))
    in_capacity = owned & (rank < capacity)
    destination = safe_local_id * capacity + jnp.clip(rank, 0, capacity - 1)
    padded = (
        jnp.zeros((local_experts * capacity, sae.hidden_dim), activation.dtype)
        .at[destination]
        .add(jnp.where(in_capacity[:, None], repeated, 0))
    )

    encoder = params.encoder.reshape(local_experts, sae.bank_width, sae.hidden_dim)
    bias = params.bias.reshape(local_experts, sae.bank_width)
    grouped = jnp.einsum(
        "ecd,efd->ecf",
        padded.reshape(local_experts, capacity, sae.hidden_dim),
        encoder.astype(activation.dtype),
        preferred_element_type=jnp.float32,
    ) + bias[:, None, :].astype(jnp.float32)
    gathered = grouped.reshape(local_experts * capacity, sae.bank_width)[destination]
    scores = jnp.where(in_capacity[:, None], jax.nn.relu(gathered), -jnp.inf)
    scores = scores.reshape(activation.shape[0], sae.top_routed * sae.bank_width)
    latent_ids = (
        expert_ids[:, :, None] * sae.bank_width
        + jnp.arange(sae.bank_width, dtype=jnp.int32)[None, None, :]
    ).reshape(activation.shape[0], -1)
    return scores, latent_ids, overflow


def _local_global_scores(
    params: SAEParams,
    activation: jax.Array,
    *,
    feature_axis: str,
) -> tuple[jax.Array, jax.Array]:
    local_width = params.encoder.shape[0]
    offset = jax.lax.axis_index(feature_axis) * local_width
    pre = jnp.einsum(
        "bd,fd->bf",
        activation,
        params.encoder.astype(activation.dtype),
        preferred_element_type=jnp.float32,
    ) + params.bias.astype(jnp.float32)
    ids = offset + jnp.arange(local_width, dtype=jnp.int32)
    return jax.nn.relu(pre), jnp.broadcast_to(ids[None, :], pre.shape)


def _build_mapped_objective(
    mesh: Mesh,
    sae: SAEConfig,
    distributed: DistributedConfig,
    a_capacity: int | None,
):
    data_axis = distributed.data_axis
    feature_axis = distributed.feature_axis
    feature_size = int(mesh.shape[feature_axis])
    if sae.condition == "A" and a_capacity is None:
        raise ValueError(
            "distributed condition A requires a precomputed static per-expert capacity"
        )
    capacity = int(a_capacity or 1)

    def local_objective(
        params: SAEParams,
        activation: jax.Array,
        expert_ids: jax.Array,
        valid_mask: jax.Array,
        mean: jax.Array,
    ) -> tuple[jax.Array, DistributedLossAux]:
        local_width = params.encoder.shape[0]
        feature_offset = jax.lax.axis_index(feature_axis) * local_width
        compute_dtype = params.encoder.dtype

        initial = (
            jnp.asarray(0.0, jnp.float32),  # local SSE
            jnp.asarray(0.0, jnp.float32),  # local SST
            jnp.asarray(0, jnp.int32),  # local l0
            jnp.asarray(0, jnp.int32),  # local valid tokens
            jnp.asarray(True),
            jnp.zeros((local_width,), bool),
            jnp.asarray(False),
        )

        def one_micro(carry, inputs):
            sse, sst, l0_sum, token_count, valid_topk, fired, overflow = carry
            x, ids, valid = inputs
            centered = x.astype(compute_dtype) - mean.astype(compute_dtype)
            if sae.condition == "A":
                scores, latent_ids, overflow_here = _local_a_scores(
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
                overflow_here = jnp.asarray(False)
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
            weights = valid.astype(jnp.float32)
            sse_here = jnp.sum(
                jnp.sum((x32 - reconstruction.astype(jnp.float32)) ** 2, axis=1)
                * weights,
                dtype=jnp.float32,
            )
            sst_here = jnp.sum(
                jnp.sum((x32 - mean.astype(jnp.float32)[None, :]) ** 2, axis=1)
                * weights,
                dtype=jnp.float32,
            )
            selected_valid = (
                (selection.values > 0) & selection.valid[:, None] & valid[:, None]
            )
            owned = (selection.latent_ids >= feature_offset) & (
                selection.latent_ids < feature_offset + local_width
            )
            local_ids = jnp.clip(
                selection.latent_ids - feature_offset, 0, local_width - 1
            )
            fired_here = (
                jnp.zeros((local_width,), jnp.int32)
                .at[local_ids.reshape(-1)]
                .max((selected_valid & owned).reshape(-1).astype(jnp.int32))
                > 0
            )
            return (
                sse + sse_here,
                sst + sst_here,
                l0_sum + jnp.sum(selected_valid, dtype=jnp.int32),
                token_count + jnp.sum(valid, dtype=jnp.int32),
                valid_topk & jnp.all(selection.valid | ~valid),
                fired | fired_here,
                overflow | overflow_here,
            ), None

        # Rematerializing each microbatch during backward keeps activation
        # memory bounded while preserving one optimizer update over their sum.
        scanned, _ = jax.lax.scan(
            jax.checkpoint(one_micro),
            initial,
            (activation, expert_ids, valid_mask),
        )
        local_sse, local_sst, local_l0, local_tokens, valid_topk, fired, overflow = (
            scanned
        )

        # Data shards own disjoint tokens.  Feature shards hold replicas of the
        # final reconstruction/loss, so pmean removes that replica dimension.
        global_sse = jax.lax.pmean(jax.lax.psum(local_sse, data_axis), feature_axis)
        global_sst = jax.lax.pmean(jax.lax.psum(local_sst, data_axis), feature_axis)
        global_l0 = jax.lax.pmean(
            jax.lax.psum(local_l0.astype(jnp.float32), data_axis), feature_axis
        ).astype(jnp.int32)
        global_tokens = jax.lax.pmean(
            jax.lax.psum(local_tokens.astype(jnp.float32), data_axis), feature_axis
        ).astype(jnp.int32)
        valid_topk = (
            jax.lax.pmin(valid_topk.astype(jnp.int32), (data_axis, feature_axis)) > 0
        )
        overflow = (
            jax.lax.pmax(overflow.astype(jnp.int32), (data_axis, feature_axis)) > 0
        )
        fired = jax.lax.pmax(fired.astype(jnp.int32), data_axis) > 0
        aux = DistributedLossAux(
            global_sse,
            global_sst,
            global_l0,
            global_tokens,
            valid_topk,
            fired,
            overflow,
        )
        healthy = valid_topk & ~overflow & (global_sst > 0)
        loss = jnp.where(healthy, global_sse / jnp.maximum(global_sst, 1e-12), jnp.nan)
        return loss, aux

    aux_specs = DistributedLossAux(P(), P(), P(), P(), P(), P(feature_axis), P())
    return jax.shard_map(
        local_objective,
        mesh=mesh,
        in_specs=(
            _param_specs(feature_axis),
            P(None, data_axis, None),
            P(None, data_axis, None),
            P(None, data_axis),
            P(None),
        ),
        out_specs=(P(), aux_specs),
        axis_names={data_axis, feature_axis},
        check_vma=False,
    )


class DistributedTrainer:
    """Compiled feature-sharded update reusable across all training steps."""

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
        self.a_capacity = a_capacity
        _validate_mesh_model(mesh, sae_config, self.distributed)
        mapped_objective = _build_mapped_objective(
            mesh, sae_config, self.distributed, a_capacity
        )

        def complete_step(
            state: TrainState, batch: DistributedBatch, mean: jax.Array
        ) -> tuple[TrainState, StepMetrics, DistributedLossAux]:
            (loss, aux), grads = jax.value_and_grad(mapped_objective, has_aux=True)(
                state.params,
                batch.activation,
                batch.expert_ids,
                batch.valid_mask,
                mean,
            )
            l0 = aux.l0_sum.astype(jnp.float32) / jnp.maximum(
                aux.valid_tokens.astype(jnp.float32), 1.0
            )
            state, metrics = apply_gradients(
                state,
                grads,
                sae_config,
                loss=loss,
                valid_tokens=aux.valid_tokens,
                fired_mask=aux.fired_mask,
                l0=l0,
                sse=aux.sse,
                sst=aux.sst,
                topk_valid=aux.topk_valid,
            )
            return state, metrics, aux

        self._compiled = jax.jit(
            complete_step,
            in_shardings=(
                state_shardings(mesh, self.distributed),
                batch_shardings(mesh, self.distributed),
                NamedSharding(mesh, P()),
            ),
        )
        self._executable = None

    def _prepare(
        self,
        microbatches: Sequence[Any] | DistributedBatch,
        mean: Any,
        *,
        process_local: bool = False,
    ) -> tuple[DistributedBatch, jax.Array]:
        batch = (
            microbatches
            if isinstance(microbatches, DistributedBatch)
            else stack_microbatches(microbatches, self.sae_config, self.distributed)
        )
        batch = (
            shard_process_local_batch(batch, self.mesh, self.distributed)
            if process_local
            else shard_batch(batch, self.mesh, self.distributed)
        )
        mean_array = jax.device_put(
            jnp.asarray(mean, jnp.float32), NamedSharding(self.mesh, P())
        )
        return batch, mean_array

    def compile(
        self,
        state: TrainState,
        microbatches: Sequence[Any] | DistributedBatch,
        mean: Any,
        *,
        process_local: bool = False,
    ) -> float:
        """Collectively compile without executing an optimizer update."""

        batch, mean_array = self._prepare(
            microbatches, mean, process_local=process_local
        )
        started = time.perf_counter()
        self._executable = self._compiled.lower(state, batch, mean_array).compile()
        return time.perf_counter() - started

    def step(
        self,
        state: TrainState,
        microbatches: Sequence[Any] | DistributedBatch,
        mean: Any,
        *,
        process_local: bool = False,
    ) -> tuple[TrainState, StepMetrics]:
        batch, mean_array = self._prepare(
            microbatches, mean, process_local=process_local
        )
        executable = self._compiled if self._executable is None else self._executable
        new_state, metrics, aux = executable(state, batch, mean_array)
        capacity_overflow, topk_valid, l0, valid_tokens = jax.device_get(
            (aux.capacity_overflow, aux.topk_valid, metrics.l0, aux.valid_tokens)
        )
        if bool(capacity_overflow):
            raise RuntimeError(
                "condition A exceeded its precomputed grouped-GEMM capacity; "
                "no update was accepted"
            )
        if not bool(topk_valid):
            raise RuntimeError(
                "fewer than k strictly positive features on a scientific token; "
                "no update was accepted"
            )
        if int(valid_tokens) <= 0:
            raise RuntimeError("distributed update contained zero scientific tokens")
        if not math.isclose(float(l0), self.sae_config.k, rel_tol=0, abs_tol=1e-6):
            raise RuntimeError("distributed TopK produced an unexpected L0")
        finite = jax.device_get(
            jnp.isfinite(metrics.loss)
            & jnp.isfinite(metrics.grad_norm)
            & jnp.isfinite(metrics.sse)
            & jnp.isfinite(metrics.sst)
        )
        if not bool(finite):
            raise RuntimeError("non-finite distributed update; no update was accepted")
        return new_state, metrics


_TRAINER_CACHE: dict[tuple[Any, ...], DistributedTrainer] = {}


def _trainer_key(
    mesh: Mesh,
    sae_config: SAEConfig,
    distributed: DistributedConfig,
    a_capacity: int | None,
) -> tuple[Any, ...]:
    devices = tuple(int(device.id) for device in np.asarray(mesh.devices).reshape(-1))
    return devices, tuple(mesh.axis_names), sae_config, distributed, a_capacity


def distributed_train_step(
    state: TrainState,
    microbatches: Sequence[Any] | DistributedBatch,
    mean: Any,
    sae_config: SAEConfig,
    mesh: Mesh,
    *,
    a_capacity: int | None = None,
    distributed: DistributedConfig | None = None,
) -> tuple[TrainState, StepMetrics]:
    """Cached functional wrapper around :class:`DistributedTrainer`."""

    distributed = distributed or DistributedConfig()
    key = _trainer_key(mesh, sae_config, distributed, a_capacity)
    trainer = _TRAINER_CACHE.get(key)
    if trainer is None:
        trainer = DistributedTrainer(
            mesh, sae_config, distributed, a_capacity=a_capacity
        )
        _TRAINER_CACHE[key] = trainer
    return trainer.step(state, microbatches, mean)


def _global_max_seconds(value: float) -> float:
    if jax.process_count() == 1:
        return float(value)
    from jax.experimental import multihost_utils

    gathered = multihost_utils.process_allgather(np.asarray(value, np.float64))
    return float(np.max(np.asarray(gathered)))


def _checkpoint_child(root: str | Path, name: str) -> str:
    value = str(root)
    if "://" in value:
        return f"{value.rstrip('/')}/{name}"
    return str(Path(value) / name)


def _checkpoint_timing_path(root: str | Path) -> Path:
    value = str(root)
    if "://" in value:
        raise ValueError("SAE timing sidecars require the shared POSIX artifact store")
    return Path(value) / "checkpoint_timing.json"


def _publish_checkpoint_timing(
    checkpoint_dir: str | Path,
    *,
    checkpoint_identity: Mapping[str, Any],
    tokens_seen: int,
    steps: int,
    optimizer_wall_seconds: float,
    compile_seconds: float,
    checkpoint_seconds: float,
) -> None:
    payload = {
        **checkpoint_identity,
        "tokens_seen": tokens_seen,
        "steps": steps,
        "optimizer_wall_seconds": optimizer_wall_seconds,
        "compile_seconds": compile_seconds,
        "checkpoint_seconds": checkpoint_seconds,
    }
    if jax.process_index() == 0:
        atomic_write_json(_checkpoint_timing_path(checkpoint_dir), payload)
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices(
            f"gemma4-sae-checkpoint-timing-{tokens_seen}"
        )


def _load_checkpoint_timing(
    checkpoint_dir: str | Path,
    *,
    checkpoint_identity: Mapping[str, Any],
    restored_tokens: int,
) -> dict[str, Any] | None:
    path = _checkpoint_timing_path(checkpoint_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read checkpoint timing sidecar: {path}") from error
    if any(payload.get(name) != value for name, value in checkpoint_identity.items()):
        raise RuntimeError("checkpoint timing sidecar belongs to another run")
    sidecar_tokens = int(payload.get("tokens_seen", -1))
    if sidecar_tokens > restored_tokens:
        raise RuntimeError("checkpoint timing sidecar is ahead of restored state")
    if sidecar_tokens < restored_tokens:
        return None
    for name in ("optimizer_wall_seconds", "compile_seconds", "checkpoint_seconds"):
        value = float(payload.get(name, -1.0))
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"checkpoint timing sidecar has invalid {name}")
    return payload


def run_distributed_training(
    training_config: TrainingConfig,
    batches: Any,
    mean: Any,
    mesh: Mesh,
    *,
    state: TrainState | None = None,
    distributed: DistributedConfig | None = None,
    a_capacity: int | None = None,
    checkpoint_dir: str | Path | None = None,
    stop_after_tokens: int | None = None,
    require_health_pass: bool = True,
    metric_callback: Any | None = None,
    process_local_inputs: bool = False,
    prior_optimizer_wall_seconds: float = 0.0,
    prior_compile_seconds: float = 0.0,
    prior_checkpoint_seconds: float = 0.0,
    process_local_source_offset: int = 0,
    checkpoint_identity: Mapping[str, Any] | None = None,
) -> DistributedTrainingResult:
    """Complete sharded train path with explicit marginal-cost accounting.

    Compilation is performed and timed separately.  ``optimizer_wall_seconds``
    starts before host batch decoding/device placement and ends only after the
    synchronized update health checks, so it includes input stalls.  Orbax
    time is accumulated separately and never contributes to marginal
    ``chip_seconds``.
    """

    distributed = distributed or DistributedConfig(
        microbatches_per_update=training_config.microbatches_per_update
    )
    checkpoint_identity = dict(checkpoint_identity or {})
    if distributed.microbatches_per_update != training_config.microbatches_per_update:
        raise ValueError("distributed and training microbatch counts disagree")
    _validate_mesh_model(mesh, training_config.sae, distributed)
    if state is None:
        state = init_distributed_state(
            jax.random.PRNGKey(training_config.seed),
            mesh,
            training_config.sae,
            distributed,
        )
    for name, value in (
        ("prior_optimizer_wall_seconds", prior_optimizer_wall_seconds),
        ("prior_compile_seconds", prior_compile_seconds),
        ("prior_checkpoint_seconds", prior_checkpoint_seconds),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    start_tokens = int(np.asarray(jax.device_get(state.tokens_seen)))
    if start_tokens > training_config.total_tokens:
        raise ValueError("checkpoint exceeds configured token budget")
    if (
        start_tokens < training_config.total_tokens
        and start_tokens % training_config.effective_batch_tokens
    ):
        raise ValueError("non-final distributed resume must be on an update boundary")

    target_tokens = training_config.total_tokens
    if stop_after_tokens is not None and stop_after_tokens < target_tokens:
        if stop_after_tokens <= start_tokens:
            raise ValueError("stop_after_tokens must exceed current progress")
        target_tokens = min(
            target_tokens,
            math.ceil(stop_after_tokens / training_config.effective_batch_tokens)
            * training_config.effective_batch_tokens,
        )
    if start_tokens == target_tokens:
        micro_iterator = iter(())
    elif process_local_inputs:
        data_size = int(mesh.shape[distributed.data_axis])
        if (
            training_config.total_tokens % data_size
            or training_config.effective_batch_tokens % data_size
            or training_config.microbatch_tokens % data_size
        ):
            raise ValueError(
                "token budget and global batches must divide across the data axis"
            )
        local_config = dataclasses.replace(
            training_config,
            total_tokens=training_config.total_tokens // data_size,
            effective_batch_tokens=training_config.effective_batch_tokens // data_size,
            microbatch_tokens=training_config.microbatch_tokens // data_size,
        )
        completed_updates = int(np.asarray(jax.device_get(state.step)))
        expected_local_progress = (
            completed_updates * local_config.effective_batch_tokens
        )
        if not 0 <= process_local_source_offset <= expected_local_progress:
            raise ValueError("invalid process-local source resume offset")
        local_skip = expected_local_progress - process_local_source_offset
        local_target = (
            local_config.total_tokens
            if target_tokens == training_config.total_tokens
            else (target_tokens // training_config.effective_batch_tokens)
            * local_config.effective_batch_tokens
        )
        micro_iterator = _fixed_microbatches(
            batches,
            local_config,
            skip_tokens=local_skip,
            take_tokens=local_target - expected_local_progress,
        )
    else:
        micro_iterator = _fixed_microbatches(
            batches,
            training_config,
            skip_tokens=start_tokens,
            take_tokens=target_tokens - start_tokens,
        )
    trainer = DistributedTrainer(
        mesh, training_config.sae, distributed, a_capacity=a_capacity
    )
    metrics: list[StepMetrics] = []
    step_seconds: list[float] = []
    compile_seconds = 0.0
    optimizer_seconds_local = 0.0
    checkpoint_seconds_local = 0.0
    latest_checkpoint: str | None = None
    final_checkpoint: str | None = None
    latest_saved_tokens: int | None = None
    next_checkpoint = (
        (start_tokens // training_config.checkpoint_every_tokens) + 1
    ) * training_config.checkpoint_every_tokens

    first_group: list[ActivationBatch] | None = None
    first_input_seconds = 0.0
    if start_tokens < target_tokens:
        input_started = time.perf_counter()
        first_group = [
            next(micro_iterator) for _ in range(training_config.microbatches_per_update)
        ]
        first_input_seconds = time.perf_counter() - input_started
        compile_seconds = trainer.compile(
            state, first_group, mean, process_local=process_local_inputs
        )

    pending = first_group
    while int(np.asarray(jax.device_get(state.tokens_seen))) < target_tokens:
        started = time.perf_counter()
        group = pending
        if group is None:
            group = [
                next(micro_iterator)
                for _ in range(training_config.microbatches_per_update)
            ]
            input_before_compile = 0.0
        else:
            # The first real batch had to be obtained before lowering.  Charge
            # that artifact/input wait to optimizer wall, never compile time.
            input_before_compile = first_input_seconds
        pending = None
        state, metric = trainer.step(
            state, group, mean, process_local=process_local_inputs
        )
        elapsed = input_before_compile + (time.perf_counter() - started)
        optimizer_seconds_local += elapsed
        step_seconds.append(elapsed)
        metrics.append(metric)
        if metric_callback is not None:
            metric_callback(metric)

        tokens_seen = int(np.asarray(jax.device_get(state.tokens_seen)))
        if checkpoint_dir is not None and tokens_seen >= next_checkpoint:
            latest_checkpoint = _checkpoint_child(checkpoint_dir, "latest")
            cumulative_optimizer = prior_optimizer_wall_seconds + _global_max_seconds(
                optimizer_seconds_local
            )
            cumulative_compile = prior_compile_seconds + _global_max_seconds(
                compile_seconds
            )
            checkpoint_started = time.perf_counter()
            save_distributed_checkpoint(
                latest_checkpoint,
                state,
                mesh,
                training_config,
                metadata={
                    **checkpoint_identity,
                    "tokens_seen": tokens_seen,
                    "steps": int(state.step),
                    "optimizer_wall_seconds": cumulative_optimizer,
                    "compile_seconds": cumulative_compile,
                    "checkpoint_seconds": prior_checkpoint_seconds
                    + checkpoint_seconds_local,
                },
            )
            checkpoint_seconds_local += time.perf_counter() - checkpoint_started
            _publish_checkpoint_timing(
                checkpoint_dir,
                checkpoint_identity=checkpoint_identity,
                tokens_seen=tokens_seen,
                steps=int(state.step),
                optimizer_wall_seconds=cumulative_optimizer,
                compile_seconds=cumulative_compile,
                checkpoint_seconds=prior_checkpoint_seconds
                + _global_max_seconds(checkpoint_seconds_local),
            )
            latest_saved_tokens = tokens_seen
            while next_checkpoint <= tokens_seen:
                next_checkpoint += training_config.checkpoint_every_tokens

    tokens_seen = int(np.asarray(jax.device_get(state.tokens_seen)))
    if checkpoint_dir is not None:
        latest_checkpoint = _checkpoint_child(checkpoint_dir, "latest")
        cumulative_optimizer = prior_optimizer_wall_seconds + _global_max_seconds(
            optimizer_seconds_local
        )
        cumulative_compile = prior_compile_seconds + _global_max_seconds(
            compile_seconds
        )
        if latest_saved_tokens != tokens_seen:
            checkpoint_started = time.perf_counter()
            save_distributed_checkpoint(
                latest_checkpoint,
                state,
                mesh,
                training_config,
                metadata={
                    **checkpoint_identity,
                    "tokens_seen": tokens_seen,
                    "steps": int(state.step),
                    "optimizer_wall_seconds": cumulative_optimizer,
                    "compile_seconds": cumulative_compile,
                    "checkpoint_seconds": prior_checkpoint_seconds
                    + checkpoint_seconds_local,
                },
            )
            checkpoint_seconds_local += time.perf_counter() - checkpoint_started
            _publish_checkpoint_timing(
                checkpoint_dir,
                checkpoint_identity=checkpoint_identity,
                tokens_seen=tokens_seen,
                steps=int(state.step),
                optimizer_wall_seconds=cumulative_optimizer,
                compile_seconds=cumulative_compile,
                checkpoint_seconds=prior_checkpoint_seconds
                + _global_max_seconds(checkpoint_seconds_local),
            )
            latest_saved_tokens = tokens_seen
        if tokens_seen == training_config.total_tokens:
            final_checkpoint = _checkpoint_child(checkpoint_dir, "final")
            checkpoint_started = time.perf_counter()
            save_distributed_checkpoint(
                final_checkpoint,
                state,
                mesh,
                training_config,
                metadata={
                    **checkpoint_identity,
                    "tokens_seen": tokens_seen,
                    "steps": int(state.step),
                    "optimizer_wall_seconds": cumulative_optimizer,
                    "compile_seconds": cumulative_compile,
                    "checkpoint_seconds": prior_checkpoint_seconds
                    + checkpoint_seconds_local,
                },
            )
            checkpoint_seconds_local += time.perf_counter() - checkpoint_started
            _publish_checkpoint_timing(
                checkpoint_dir,
                checkpoint_identity=checkpoint_identity,
                tokens_seen=tokens_seen,
                steps=int(state.step),
                optimizer_wall_seconds=cumulative_optimizer,
                compile_seconds=cumulative_compile,
                checkpoint_seconds=prior_checkpoint_seconds
                + _global_max_seconds(checkpoint_seconds_local),
            )

    health = None
    if tokens_seen >= training_config.health_gate_tokens:
        last_l0 = (
            float(np.asarray(jax.device_get(metrics[-1].l0)))
            if metrics
            else float(training_config.sae.k)
        )
        topk_valid = all(
            bool(np.asarray(jax.device_get(metric.topk_valid))) for metric in metrics
        )
        health = check_seed_health(
            state, training_config, l0=last_l0, topk_valid=topk_valid
        )

    timing = DistributedRunTiming(
        compile_seconds=prior_compile_seconds + _global_max_seconds(compile_seconds),
        optimizer_wall_seconds=prior_optimizer_wall_seconds
        + _global_max_seconds(optimizer_seconds_local),
        checkpoint_seconds=prior_checkpoint_seconds
        + _global_max_seconds(checkpoint_seconds_local),
        global_chip_count=int(np.asarray(mesh.devices).size),
        tokens=tokens_seen,
        steps=int(np.asarray(jax.device_get(state.step))),
    )
    result = DistributedTrainingResult(
        state=state,
        metrics=tuple(metrics),
        step_optimizer_seconds=tuple(step_seconds),
        health=health,
        timing=timing,
        latest_checkpoint=latest_checkpoint,
        final_checkpoint=final_checkpoint,
    )
    if require_health_pass and health is not None and not health.passed:
        raise RuntimeError(
            "distributed seed health gate failed: " + "; ".join(health.reasons)
        )
    return result


def _balanced_padding_ids(rows: int, sae: SAEConfig) -> np.ndarray:
    row = np.arange(rows, dtype=np.int32)[:, None]
    slot = np.arange(sae.top_routed, dtype=np.int32)[None, :]
    return (row * sae.top_routed + slot) % sae.num_experts


def _evaluation_microbatches(
    batches: Any,
    sae: SAEConfig,
    *,
    microbatch_tokens: int,
    max_tokens: int | None,
):
    remaining = max_tokens
    pieces_x: list[np.ndarray] = []
    pieces_ids: list[np.ndarray] = []
    buffered = 0
    dtype = np.dtype(np.float32)

    def emit(pad: bool = False) -> ActivationBatch:
        nonlocal pieces_x, pieces_ids, buffered
        x = np.concatenate(pieces_x, axis=0) if len(pieces_x) > 1 else pieces_x[0]
        ids = None
        if sae.condition == "A":
            ids = (
                np.concatenate(pieces_ids, axis=0)
                if len(pieces_ids) > 1
                else pieces_ids[0]
            )
        valid = np.ones((buffered,), bool)
        if pad and buffered < microbatch_tokens:
            count = microbatch_tokens - buffered
            x = np.pad(x, ((0, count), (0, 0)))
            valid = np.pad(valid, (0, count))
            if ids is not None:
                ids = np.concatenate((ids, _balanced_padding_ids(count, sae)), axis=0)
        result = ActivationBatch(x, ids, valid)
        pieces_x, pieces_ids, buffered = [], [], 0
        return result

    for raw in batches:
        x, ids = _scientific_rows(raw, sae)
        dtype = x.dtype
        if remaining is not None:
            x = x[:remaining]
            if ids is not None:
                ids = ids[:remaining]
        offset = 0
        while offset < x.shape[0]:
            take = min(microbatch_tokens - buffered, x.shape[0] - offset)
            pieces_x.append(x[offset : offset + take])
            if ids is not None:
                pieces_ids.append(ids[offset : offset + take])
            offset += take
            buffered += take
            if remaining is not None:
                remaining -= take
            if buffered == microbatch_tokens:
                yield emit()
            if remaining == 0:
                break
        if remaining == 0:
            break
    if remaining not in (None, 0):
        raise RuntimeError(f"evaluation source ended {remaining} tokens early")
    if buffered:
        yield emit(pad=True)
    del dtype


def distributed_evaluate(
    params: SAEParams,
    batches: Any,
    mean: Any,
    sae_config: SAEConfig,
    mesh: Mesh,
    *,
    expected_sst: float | None = None,
    max_tokens: int | None = None,
    microbatch_tokens: int = 4_096,
    a_capacity: int | None = None,
    distributed: DistributedConfig | None = None,
    process_local_inputs: bool = False,
) -> EvaluationResult:
    """Evaluate sharded A/B/F parameters without gathering them to a host."""

    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    distributed = distributed or DistributedConfig()
    _validate_mesh_model(mesh, sae_config, distributed)
    mapped = _build_mapped_objective(mesh, sae_config, distributed, a_capacity)
    compiled = jax.jit(
        mapped,
        in_shardings=(
            _named(_param_specs(distributed.feature_axis), mesh),
            NamedSharding(mesh, P(None, distributed.data_axis, None)),
            NamedSharding(mesh, P(None, distributed.data_axis, None)),
            NamedSharding(mesh, P(None, distributed.data_axis)),
            NamedSharding(mesh, P()),
        ),
    )
    micro_iterator = iter(
        _evaluation_microbatches(
            batches,
            sae_config,
            microbatch_tokens=microbatch_tokens,
            max_tokens=max_tokens,
        )
    )
    sse = 0.0
    computed_sst = 0.0
    token_count = 0
    l0_sum = 0
    fired = np.zeros((sae_config.total_features,), bool)
    any_group = False
    while True:
        group: list[ActivationBatch] = []
        try:
            for _ in range(distributed.microbatches_per_update):
                group.append(next(micro_iterator))
        except StopIteration:
            pass
        if not group:
            break
        any_group = True
        while len(group) < distributed.microbatches_per_update:
            group.append(
                ActivationBatch(
                    np.zeros((microbatch_tokens, sae_config.hidden_dim), np.float32),
                    _balanced_padding_ids(microbatch_tokens, sae_config)
                    if sae_config.condition == "A"
                    else None,
                    np.zeros((microbatch_tokens,), bool),
                )
            )
        stacked = stack_microbatches(group, sae_config, distributed)
        batch = (
            shard_process_local_batch(stacked, mesh, distributed)
            if process_local_inputs
            else shard_batch(stacked, mesh, distributed)
        )
        mean_array = jax.device_put(
            jnp.asarray(mean, jnp.float32), NamedSharding(mesh, P())
        )
        _, aux = compiled(
            params,
            batch.activation,
            batch.expert_ids,
            batch.valid_mask,
            mean_array,
        )
        aux_host = jax.device_get(aux)
        if bool(aux_host.capacity_overflow):
            raise RuntimeError("evaluation exceeded condition A static capacity")
        if not bool(aux_host.topk_valid):
            raise RuntimeError("evaluation token had fewer than k positive features")
        if int(aux_host.l0_sum) != int(aux_host.valid_tokens) * sae_config.k:
            raise RuntimeError("distributed evaluation L0 differs from configured k")
        sse += float(aux_host.sse)
        computed_sst += float(aux_host.sst)
        token_count += int(aux_host.valid_tokens)
        l0_sum += int(aux_host.l0_sum)
        fired |= np.asarray(aux_host.fired_mask, bool)

    if not any_group or token_count == 0:
        raise RuntimeError("distributed evaluation consumed zero scientific tokens")
    denominator = computed_sst if expected_sst is None else float(expected_sst)
    if not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("expected evaluation SST must be finite and positive")
    if expected_sst is not None and not math.isclose(
        computed_sst,
        denominator,
        rel_tol=5e-4,
        abs_tol=1e-6 * max(1.0, denominator),
    ):
        raise RuntimeError(
            "device-computed evaluation SST disagrees with sealed FP64 statistics"
        )
    nmse = sse / denominator
    return EvaluationResult(
        condition=sae_config.condition,
        tokens=token_count,
        sse=sse,
        sst=denominator,
        computed_sst=computed_sst,
        nmse=nmse,
        explained_variance=1.0 - nmse,
        l0=l0_sum / token_count,
        dead_rate=1.0 - float(np.mean(fired)),
        topk_valid=True,
    )


def _import_orbax():
    try:
        import orbax.checkpoint as ocp
    except ImportError as exc:  # pragma: no cover - depends on installed extra
        raise RuntimeError(
            "distributed checkpoints require the TPU extra: install `moe-sae[tpu]`"
        ) from exc
    return ocp


def _checkpoint_header(
    training_config: TrainingConfig,
    mesh: Mesh,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    identity = dataclasses.asdict(training_config)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return {
        "checkpoint_version": 1,
        "training_config": identity,
        "training_config_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "mesh_shape": {name: int(size) for name, size in mesh.shape.items()},
        "metadata": dict(metadata or {}),
    }


def save_distributed_checkpoint(
    path: str | Path,
    state: TrainState,
    mesh: Mesh,
    training_config: TrainingConfig,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Collectively save sharded state with Orbax; every host must call."""

    ocp = _import_orbax()
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(
        path,
        state,
        force=True,
        custom_metadata=_checkpoint_header(training_config, mesh, metadata),
    )
    checkpointer.wait_until_finished()


def restore_distributed_checkpoint(
    path: str | Path,
    template: TrainState,
    mesh: Mesh,
    training_config: TrainingConfig,
) -> tuple[TrainState, dict[str, Any]]:
    """Collectively restore directly into template shardings on every host."""

    ocp = _import_orbax()
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_metadata = checkpointer.metadata(path).custom_metadata or {}
    expected = _checkpoint_header(training_config, mesh, None)
    if checkpoint_metadata.get("checkpoint_version") != expected["checkpoint_version"]:
        raise ValueError("unsupported distributed checkpoint version")
    if (
        checkpoint_metadata.get("training_config_sha256")
        != expected["training_config_sha256"]
    ):
        raise ValueError("distributed checkpoint configuration mismatch")
    if checkpoint_metadata.get("mesh_shape") != expected["mesh_shape"]:
        raise ValueError("distributed checkpoint mesh shape mismatch")
    restored = checkpointer.restore(path, target=template)
    return restored, dict(checkpoint_metadata.get("metadata", {}))


def _checkpoint_exists(path: str | Path) -> bool:
    _import_orbax()
    from etils import epath

    return bool(epath.Path(path).exists())


def _process_local_source(
    batches: Any, total_tokens: int, *, local_offset: int = 0
) -> tuple[Any, int, int]:
    """Return this process's contiguous token partition and local count."""

    from moe_sae.artifacts import ArtifactDataset

    process_count = jax.process_count()
    if local_offset < 0:
        raise ValueError("local_offset must be nonnegative")
    if process_count == 1:
        if local_offset > total_tokens:
            raise ValueError("local_offset exceeds token partition")
        if isinstance(batches, ArtifactDataset):
            return (
                batches.iter_token_range(local_offset, total_tokens),
                total_tokens,
                local_offset,
            )
        # Generic single-host iterables cannot seek; the runner will skip.
        return batches, total_tokens, 0
    if not isinstance(batches, ArtifactDataset):
        raise TypeError(
            "multi-host distributed input requires ArtifactDataset, not an iterator; "
            "this prevents every host from reading the full capture"
        )
    if not hasattr(batches, "iter_token_range"):
        raise RuntimeError(
            "ArtifactDataset must provide iter_token_range(start, end) for process-local I/O"
        )
    process = jax.process_index()
    start = (total_tokens * process) // process_count
    end = (total_tokens * (process + 1)) // process_count
    partition_tokens = end - start
    if local_offset > partition_tokens:
        raise ValueError("local_offset exceeds process token partition")
    return (
        batches.iter_token_range(start + local_offset, end),
        partition_tokens,
        local_offset,
    )


def precompute_distributed_a_capacity(
    dataset: Any,
    training_config: TrainingConfig,
    distributed: DistributedConfig | None = None,
    *,
    total_tokens: int | None = None,
) -> int:
    """Scan this process's fixed token partition and return global safe capacity.

    Capacity is measured on the actual process-local microbatch shape, then the
    maximum is reduced across hosts before applying the 10% rounded margin.
    This avoids the roughly process-count-fold padding caused by scanning a
    duplicated global microbatch on every host.
    """

    if training_config.sae.condition != "A":
        raise ValueError("routing capacity is defined only for condition A")
    distributed = distributed or DistributedConfig(
        microbatches_per_update=training_config.microbatches_per_update
    )
    data_size = jax.process_count()
    token_count = training_config.total_tokens if total_tokens is None else total_tokens
    if token_count <= 0:
        raise ValueError("capacity token count must be positive")
    if token_count % data_size:
        raise ValueError("training tokens must partition evenly across processes")
    if (
        training_config.effective_batch_tokens % data_size
        or training_config.microbatch_tokens % data_size
    ):
        raise ValueError("training batches must partition evenly across processes")
    process = jax.process_index()
    local_tokens = token_count // data_size
    route_only = isinstance(dataset, ArtifactDataset) and hasattr(
        dataset, "iter_expert_ids_range"
    )
    if route_only:
        start = local_tokens * process
        source = dataset.iter_expert_ids_range(start, start + local_tokens)
    else:
        source, local_tokens, _ = _process_local_source(
            dataset, training_config.total_tokens
        )
    local_microbatch_tokens = training_config.microbatch_tokens // data_size
    worst = 0

    def observe(ids: np.ndarray) -> None:
        nonlocal worst
        counts = np.bincount(
            np.asarray(ids).reshape(-1),
            minlength=training_config.sae.num_experts,
        )
        worst = max(worst, int(counts.max(initial=0)))

    if route_only:
        micro = local_microbatch_tokens
        buffered = np.empty((0, training_config.sae.top_routed), dtype=np.int32)
        consumed = 0
        for raw_ids in source:
            ids = np.asarray(raw_ids, dtype=np.int32)
            consumed += ids.shape[0]
            if buffered.shape[0]:
                ids = np.concatenate((buffered, ids), axis=0)
            whole = (ids.shape[0] // micro) * micro
            for offset in range(0, whole, micro):
                observe(ids[offset : offset + micro])
            buffered = ids[whole:]
        if consumed != local_tokens:
            raise RuntimeError(
                f"route-only capacity source yielded {consumed}/{local_tokens} rows"
            )
        if buffered.shape[0]:
            padding = _balanced_padding_ids(
                micro - buffered.shape[0], training_config.sae
            )
            observe(np.concatenate((buffered, padding), axis=0))
    else:
        if total_tokens is not None and total_tokens != training_config.total_tokens:
            raise TypeError(
                "capacity token override requires ArtifactDataset route-only input"
            )
        local_config = dataclasses.replace(
            training_config,
            total_tokens=local_tokens,
            effective_batch_tokens=training_config.effective_batch_tokens // data_size,
            microbatch_tokens=local_microbatch_tokens,
        )
        for batch in _fixed_microbatches(
            source, local_config, skip_tokens=0, take_tokens=local_tokens
        ):
            observe(np.asarray(batch.expert_ids))
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        worst = int(
            np.max(multihost_utils.process_allgather(np.asarray(worst, np.int32)))
        )
    capacity = int(math.ceil((worst * 1.10) / 8.0) * 8)
    if capacity <= 0:
        raise RuntimeError("could not derive a condition A routing capacity")
    return capacity


def train_distributed(
    training_config: TrainingConfig,
    batches: Any,
    mean: Any,
    *,
    checkpoint_dir: str | Path | None = None,
    stop_after_tokens: int | None = None,
    metric_callback: Any | None = None,
    checkpoint_identity: Mapping[str, Any] | None = None,
    prior_run_timing: Mapping[str, Any] | None = None,
) -> DistributedTrainingResult:
    """CLI-facing multi-host train/resume entrypoint.

    ``batches`` must be an :class:`ArtifactDataset` on multi-host runs so each
    process loads only its contiguous token partition.  ``latest`` is restored
    collectively when present; completed runs write ``<checkpoint_dir>/final``.
    """

    distributed = DistributedConfig(
        microbatches_per_update=training_config.microbatches_per_update
    )
    mesh = create_mesh(distributed)
    if int(mesh.shape[distributed.data_axis]) != jax.process_count():
        raise RuntimeError("production data axis must equal jax.process_count()")
    process_count = jax.process_count()
    if training_config.total_tokens % process_count:
        raise ValueError("training token budget must partition evenly across processes")
    local_tokens = training_config.total_tokens // process_count
    state = init_distributed_state(
        jax.random.PRNGKey(training_config.seed),
        mesh,
        training_config.sae,
        distributed,
    )
    prior_timing: dict[str, Any] = {}
    checkpoint_identity = dict(checkpoint_identity or {})
    if checkpoint_dir is not None:
        latest = _checkpoint_child(checkpoint_dir, "latest")
        if _checkpoint_exists(latest):
            state, prior_timing = restore_distributed_checkpoint(
                latest, state, mesh, training_config
            )
            mismatches = {
                name: (prior_timing.get(name), expected)
                for name, expected in checkpoint_identity.items()
                if prior_timing.get(name) != expected
            }
            if mismatches:
                raise RuntimeError(
                    "distributed checkpoint belongs to a different capture/run: "
                    + ", ".join(sorted(mismatches))
                )
            restored_tokens = int(np.asarray(jax.device_get(state.tokens_seen)))
            sidecar_timing = _load_checkpoint_timing(
                checkpoint_dir,
                checkpoint_identity=checkpoint_identity,
                restored_tokens=restored_tokens,
            )
            if sidecar_timing is not None:
                prior_timing.update(sidecar_timing)
            if prior_run_timing:
                recorded_tokens = int(prior_run_timing.get("tokens_seen", -1))
                if recorded_tokens > restored_tokens:
                    raise RuntimeError(
                        "run.json is ahead of the restored checkpoint boundary"
                    )
                if recorded_tokens == restored_tokens:
                    for name in (
                        "optimizer_wall_seconds",
                        "compile_seconds",
                        "checkpoint_seconds",
                    ):
                        value = float(prior_run_timing.get(name, -1.0))
                        if not math.isfinite(value) or value < 0.0:
                            raise RuntimeError(
                                f"run.json has invalid cumulative {name}"
                            )
                        prior_timing[name] = value
    completed_updates = int(np.asarray(jax.device_get(state.step)))
    local_effective = training_config.effective_batch_tokens // process_count
    local_offset = min(completed_updates * local_effective, local_tokens)
    source, source_partition_tokens, applied_offset = _process_local_source(
        batches, training_config.total_tokens, local_offset=local_offset
    )
    if source_partition_tokens != local_tokens:
        raise RuntimeError("process-local dataset partition size changed")
    return run_distributed_training(
        training_config,
        source,
        mean,
        mesh,
        state=state,
        distributed=distributed,
        a_capacity=training_config.a_capacity,
        checkpoint_dir=checkpoint_dir,
        stop_after_tokens=stop_after_tokens,
        require_health_pass=False,
        # Metrics are logically replicated across hosts; only process zero may
        # mutate the shared JSONL sink.
        metric_callback=metric_callback if jax.process_index() == 0 else None,
        process_local_inputs=True,
        prior_optimizer_wall_seconds=float(
            prior_timing.get("optimizer_wall_seconds", 0.0)
        ),
        prior_compile_seconds=float(prior_timing.get("compile_seconds", 0.0)),
        prior_checkpoint_seconds=float(prior_timing.get("checkpoint_seconds", 0.0)),
        process_local_source_offset=applied_offset,
        checkpoint_identity=checkpoint_identity,
    )


def evaluate_distributed(
    checkpoint_path: str | Path,
    training_config: TrainingConfig,
    batches: Any,
    mean: Any,
    *,
    max_tokens: int | None = None,
    expected_sst: float | None = None,
    checkpoint_identity: Mapping[str, Any] | None = None,
) -> EvaluationResult:
    """CLI-facing sharded restore/evaluation entrypoint."""

    distributed = DistributedConfig(
        microbatches_per_update=training_config.microbatches_per_update
    )
    mesh = create_mesh(distributed)
    template = init_distributed_state(
        jax.random.PRNGKey(training_config.seed),
        mesh,
        training_config.sae,
        distributed,
    )
    state, checkpoint_metadata = restore_distributed_checkpoint(
        checkpoint_path, template, mesh, training_config
    )
    checkpoint_identity = dict(checkpoint_identity or {})
    mismatches = {
        name: (checkpoint_metadata.get(name), expected)
        for name, expected in checkpoint_identity.items()
        if checkpoint_metadata.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            "evaluation checkpoint belongs to a different capture/run: "
            + ", ".join(sorted(mismatches))
        )
    restored_tokens = int(np.asarray(jax.device_get(state.tokens_seen)))
    restored_steps = int(np.asarray(jax.device_get(state.step)))
    if (
        restored_tokens != training_config.total_tokens
        or restored_steps != training_config.updates
    ):
        raise RuntimeError(
            "evaluation checkpoint is not final: "
            f"tokens={restored_tokens}/{training_config.total_tokens}, "
            f"steps={restored_steps}/{training_config.updates}"
        )
    from moe_sae.artifacts import ArtifactDataset

    if max_tokens is None:
        if not isinstance(batches, ArtifactDataset):
            if jax.process_count() > 1:
                raise TypeError("multi-host eval requires ArtifactDataset")
            source, local_max = batches, None
        else:
            max_tokens = batches.total_tokens
            source, local_max, _ = _process_local_source(batches, max_tokens)
    else:
        source, local_max, _ = _process_local_source(batches, max_tokens)
    if local_max is not None and local_max * jax.process_count() != max_tokens:
        raise ValueError(
            "evaluation token budget must partition evenly across processes"
        )
    local_microbatch = training_config.microbatch_tokens // jax.process_count()
    return distributed_evaluate(
        state.params,
        source,
        mean,
        training_config.sae,
        mesh,
        expected_sst=expected_sst,
        max_tokens=local_max,
        microbatch_tokens=local_microbatch,
        a_capacity=training_config.a_capacity,
        distributed=distributed,
        process_local_inputs=True,
    )


__all__ = [
    "DistributedBatch",
    "DistributedConfig",
    "DistributedLossAux",
    "DistributedRunTiming",
    "DistributedTrainingResult",
    "DistributedTrainer",
    "batch_shardings",
    "create_mesh",
    "distributed_train_step",
    "distributed_evaluate",
    "evaluate_distributed",
    "init_distributed_state",
    "precompute_distributed_a_capacity",
    "restore_distributed_checkpoint",
    "run_distributed_training",
    "save_distributed_checkpoint",
    "shard_batch",
    "shard_process_local_batch",
    "shard_train_state",
    "stack_microbatches",
    "state_shardings",
    "train_distributed",
]
