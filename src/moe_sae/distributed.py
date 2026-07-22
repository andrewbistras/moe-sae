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
from moe_sae.checkpointing import (
    checkpoint_payload_uri,
    final_pointer_from_latest,
    make_pointer,
    new_attempt_id,
    normalize_payload_root,
    pointer_path,
    publish_pointer,
    validate_pointer,
)
from moe_sae.pipeline import (
    ActivationBatch,
    EvaluationResult,
    HealthGateResult,
    TrainingConfig,
    _fixed_microbatches,
    _scientific_rows,
    check_seed_health,
)
from moe_sae.metrics_journal import (
    JournalIdentity,
    MetricsJournal,
    training_journal_snapshot,
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


CHECKPOINT_CONTROL_VISIBILITY_TIMEOUT_SECONDS = 90.0
CHECKPOINT_CONTROL_VISIBILITY_POLL_SECONDS = 0.25
CHECKPOINT_CONTROL_MAX_BYTES = 1024 * 1024


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


def _validate_exact_distributed_l0(
    *, l0_sum: int, valid_tokens: int, k: int
) -> None:
    """Validate TopK cardinality from exact counts, never a floating ratio.

    TPU float32 division can round an exactly integral quotient by one ULP.  In
    particular, the production tail has ``794624 / 12416 == 64`` but was
    observed as ``63.999996`` on TPU v4.  Both source values are small exact
    integers, so their equality is the authoritative invariant.
    """

    if valid_tokens <= 0:
        raise RuntimeError("distributed update contained zero scientific tokens")
    expected = valid_tokens * k
    if l0_sum != expected:
        raise RuntimeError(
            "distributed TopK selected an unexpected number of features: "
            f"selected={l0_sum}, valid_tokens={valid_tokens}, k={k}, "
            f"expected={expected}; no update was accepted"
        )


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
    metrics_journal: dict[str, Any] | None = None

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
    # [token×route, local_experts] bookkeeping is small (32 experts per feature
    # device on the production 4-way feature axis); the expensive expert
    # matmul is performed exactly once per owned routed bank.
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
            l0_ratio = aux.l0_sum.astype(jnp.float32) / jnp.maximum(
                aux.valid_tokens.astype(jnp.float32), 1.0
            )
            exact_l0_counts = (
                aux.l0_sum
                == aux.valid_tokens * jnp.asarray(sae_config.k, aux.l0_sum.dtype)
            )
            # Canonicalize a proven-exact result.  Keeping the raw ratio for a
            # mismatch preserves useful diagnostics without allowing TPU
            # division rounding to poison the journal or downstream health gate.
            l0 = jnp.where(
                exact_l0_counts,
                jnp.asarray(sae_config.k, jnp.float32),
                l0_ratio,
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
        capacity_overflow, topk_valid, l0_sum, valid_tokens = jax.device_get(
            (
                aux.capacity_overflow,
                aux.topk_valid,
                aux.l0_sum,
                aux.valid_tokens,
            )
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
        _validate_exact_distributed_l0(
            l0_sum=int(l0_sum),
            valid_tokens=int(valid_tokens),
            k=self.sae_config.k,
        )
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


def _effective_checkpoint_payload_root(
    checkpoint_dir: str | Path,
    checkpoint_payload_root: str | Path | None,
) -> str:
    """Resolve the immutable payload root.

    The local default keeps tests and small deployments self-contained while
    still separating mutable control pointers from immutable Orbax payloads.
    Production callers should pass their qualified ``gs://`` payload root.
    """

    if checkpoint_payload_root is None:
        checkpoint_payload_root = Path(checkpoint_dir) / "checkpoint_payloads"
    return normalize_payload_root(checkpoint_payload_root)


def _collective_attempt_id() -> str:
    """Generate one unpredictable checkpoint attempt ID shared by all hosts."""

    attempt = new_attempt_id() if jax.process_index() == 0 else "0" * 32
    if jax.process_count() == 1:
        return attempt
    from jax.experimental import multihost_utils

    local = np.frombuffer(bytes.fromhex(attempt), dtype=np.uint8)
    gathered = np.asarray(multihost_utils.process_allgather(local), dtype=np.uint8)
    first = gathered.reshape(jax.process_count(), 16)[0]
    result = bytes(first.tolist()).hex()
    if result == "0" * 32:
        raise RuntimeError("coordinator did not publish a checkpoint attempt ID")
    return result


def _checkpoint_barrier(label: str) -> None:
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices(label)


def _checkpoint_collective_failed(local_failed: bool) -> bool:
    """Return one identical checkpoint-control decision on every process."""

    if jax.process_count() == 1:
        return bool(local_failed)
    from jax.experimental import multihost_utils

    gathered = np.asarray(
        multihost_utils.process_allgather(
            np.asarray([local_failed], dtype=np.uint8), tiled=False
        ),
        dtype=np.uint8,
    )
    return bool(np.any(gathered))


def _checkpoint_phase_decision(
    operation: str, local_error: Exception | None
) -> None:
    """Raise only after every process has received the phase outcome."""

    if not _checkpoint_collective_failed(local_error is not None):
        return
    error = RuntimeError(f"distributed checkpoint {operation} failed")
    if local_error is not None:
        raise error from local_error
    raise error


def _coordinator_read_checkpoint_json_if_exists(
    path: Path, *, operation: str
) -> Any | None:
    """Broadcast process zero's exact control-plane observation to all hosts."""

    source: bytes | None = None
    read_error: Exception | None = None
    if jax.process_index() == 0:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            source = None
        except OSError as error:
            read_error = error
        else:
            try:
                value = json.loads(raw)
                source = json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                if len(source) > CHECKPOINT_CONTROL_MAX_BYTES:
                    raise ValueError("checkpoint control JSON exceeds the size bound")
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                read_error = error
    _checkpoint_phase_decision(f"{operation} read", read_error)

    if jax.process_count() == 1:
        shared = source
    else:
        from jax.experimental import multihost_utils

        local_length = (
            -1
            if jax.process_index() == 0 and source is None
            else len(source)
            if jax.process_index() == 0 and source is not None
            else 0
        )
        length_value = np.asarray(
            multihost_utils.broadcast_one_to_all(
                np.asarray([local_length], dtype=np.int32),
                is_source=jax.process_index() == 0,
            ),
            dtype=np.int32,
        )
        length = int(length_value.reshape(-1)[0])
        if length == -1:
            shared = None
        elif not 0 <= length <= CHECKPOINT_CONTROL_MAX_BYTES:
            shared = None
            read_error = RuntimeError("checkpoint control broadcast length is invalid")
        else:
            local = (
                np.frombuffer(source, dtype=np.uint8).copy()
                if jax.process_index() == 0 and source is not None
                else np.zeros((length,), dtype=np.uint8)
            )
            value = np.asarray(
                multihost_utils.broadcast_one_to_all(
                    local, is_source=jax.process_index() == 0
                ),
                dtype=np.uint8,
            )
            if value.shape != (length,):
                shared = None
                read_error = RuntimeError(
                    "checkpoint control broadcast payload is invalid"
                )
            else:
                shared = value.tobytes()
    _checkpoint_phase_decision(f"{operation} broadcast", read_error)
    if shared is None:
        return None
    decoded: Any | None = None
    decode_error: Exception | None = None
    try:
        decoded = json.loads(shared.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        decode_error = error
    _checkpoint_phase_decision(f"{operation} decode", decode_error)
    return decoded


def _collective_load_pointer(
    checkpoint_dir: str | Path,
    *,
    kind: str,
    payload_root: str | Path,
    run_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Choose pointer absence/presence once, then validate it collectively."""

    path: Path | None = None
    prepare_error: Exception | None = None
    try:
        if kind not in {"latest", "final"}:
            raise ValueError("checkpoint pointer kind must be latest or final")
        path = pointer_path(checkpoint_dir, kind)  # type: ignore[arg-type]
    except Exception as error:
        prepare_error = error
    _checkpoint_phase_decision("pointer load preparation", prepare_error)
    assert path is not None
    raw = _coordinator_read_checkpoint_json_if_exists(
        path, operation=f"{kind} pointer"
    )
    if raw is None:
        return None

    pointer: dict[str, Any] | None = None
    validation_error: Exception | None = None
    try:
        if not isinstance(raw, Mapping):
            raise RuntimeError("checkpoint pointer must be a JSON object")
        pointer = validate_pointer(
            raw,
            expected_kind=kind,  # type: ignore[arg-type]
            payload_root=payload_root,
            run_identity=run_identity,
        )
    except Exception as error:
        validation_error = error
    _checkpoint_phase_decision("pointer validation", validation_error)
    assert pointer is not None
    return pointer


def _collectively_require_finalized_checkpoint(payload_uri: str) -> None:
    local_error: Exception | None = None
    try:
        if not _orbax_path_finalized(_import_orbax(), payload_uri):
            raise RuntimeError("checkpoint payload is not finalized")
    except Exception as error:
        local_error = error
    _checkpoint_phase_decision("payload finalization check", local_error)


def _broadcast_coordinator_json(
    source: Mapping[str, Any] | None, *, operation: str
) -> dict[str, Any]:
    """Broadcast one strict coordinator-owned JSON object to every process."""

    encoded: bytes | None = None
    encode_error: Exception | None = None
    if jax.process_index() == 0:
        try:
            if not isinstance(source, Mapping):
                raise TypeError("coordinator JSON source must be a mapping")
            encoded = json.dumps(
                dict(source),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if len(encoded) > CHECKPOINT_CONTROL_MAX_BYTES:
                raise ValueError("coordinator JSON exceeds the size bound")
        except Exception as error:
            encode_error = error
    _checkpoint_phase_decision(f"{operation} encoding", encode_error)

    if jax.process_count() == 1:
        shared = encoded
    else:
        from jax.experimental import multihost_utils

        local_length = len(encoded) if encoded is not None else 0
        length_value = np.asarray(
            multihost_utils.broadcast_one_to_all(
                np.asarray([local_length], dtype=np.int32),
                is_source=jax.process_index() == 0,
            ),
            dtype=np.int32,
        )
        length = int(length_value.reshape(-1)[0])
        broadcast_error: Exception | None = None
        if not 0 < length <= CHECKPOINT_CONTROL_MAX_BYTES:
            shared = None
            broadcast_error = RuntimeError(
                "coordinator JSON broadcast length is invalid"
            )
        else:
            local = (
                np.frombuffer(encoded, dtype=np.uint8).copy()
                if encoded is not None
                else np.zeros((length,), dtype=np.uint8)
            )
            value = np.asarray(
                multihost_utils.broadcast_one_to_all(
                    local, is_source=jax.process_index() == 0
                ),
                dtype=np.uint8,
            )
            if value.shape != (length,):
                shared = None
                broadcast_error = RuntimeError(
                    "coordinator JSON broadcast payload is invalid"
                )
            else:
                shared = value.tobytes()
        _checkpoint_phase_decision(f"{operation} broadcast", broadcast_error)
    assert shared is not None
    result: dict[str, Any] | None = None
    decode_error: Exception | None = None
    try:
        value = json.loads(shared.decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError("broadcast coordinator JSON is not an object")
        result = value
    except Exception as error:
        decode_error = error
    _checkpoint_phase_decision(f"{operation} decoding", decode_error)
    assert result is not None
    return result


def _collective_training_journal_snapshot(
    journal: MetricsJournal | None,
    training_config: TrainingConfig,
    *,
    operation: str,
) -> dict[str, Any]:
    source: dict[str, Any] | None = None
    local_error: Exception | None = None
    if jax.process_index() == 0:
        try:
            if journal is None:
                raise RuntimeError("coordinator metrics journal is unavailable")
            source = training_journal_snapshot(
                journal,
                total_tokens=training_config.total_tokens,
                effective_batch_tokens=training_config.effective_batch_tokens,
                expected_l0=training_config.sae.k,
            )
        except Exception as error:
            local_error = error
    _checkpoint_phase_decision(f"{operation} snapshot", local_error)
    return _broadcast_coordinator_json(source, operation=operation)


def _prepare_training_journal(
    path: str | Path | None,
    identity: JournalIdentity | None,
    training_config: TrainingConfig,
    *,
    restored_step: int,
    restored_tokens: int,
) -> tuple[MetricsJournal | None, dict[str, Any] | None]:
    if (path is None) != (identity is None):
        raise ValueError(
            "metrics journal path and identity must be supplied together"
        )
    if path is None:
        return None, None
    if not isinstance(identity, JournalIdentity):
        raise TypeError("metrics journal identity must be a JournalIdentity")
    if "://" in str(path):
        raise ValueError("metrics journal requires a shared POSIX path")

    journal: MetricsJournal | None = None
    local_error: Exception | None = None
    if jax.process_index() == 0:
        try:
            journal = MetricsJournal(path, identity)
            journal.reconcile(step=restored_step, tokens_seen=restored_tokens)
        except Exception as error:
            local_error = error
    _checkpoint_phase_decision("metrics journal reconciliation", local_error)
    snapshot = _collective_training_journal_snapshot(
        journal, training_config, operation="metrics journal reconciled prefix"
    )
    binding = snapshot.get("binding")
    if (
        not isinstance(binding, Mapping)
        or int(binding.get("last_step", -1)) != restored_step
        or int(binding.get("last_tokens_seen", -1)) != restored_tokens
    ):
        raise RuntimeError("metrics journal does not match restored checkpoint")
    return journal, snapshot


def _metric_journal_record(
    metric: StepMetrics,
    *,
    step: int,
    tokens_seen: int,
    optimizer_wall_seconds: float,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "step": step,
        "tokens_seen": tokens_seen,
        "optimizer_wall_seconds": float(optimizer_wall_seconds),
    }
    for name, value in metric._asdict().items():
        array = np.asarray(jax.device_get(value))
        if array.ndim != 0:
            raise TypeError(f"distributed metric {name} is not scalar")
        record[name] = array.item()
    return record


def _append_training_journal_metric(
    journal: MetricsJournal | None,
    metric: StepMetrics,
    *,
    step: int,
    tokens_seen: int,
    optimizer_wall_seconds: float,
    metric_callback: Any | None,
) -> None:
    local_error: Exception | None = None
    if jax.process_index() == 0:
        try:
            if journal is not None:
                journal.append(
                    _metric_journal_record(
                        metric,
                        step=step,
                        tokens_seen=tokens_seen,
                        optimizer_wall_seconds=optimizer_wall_seconds,
                    )
                )
            if metric_callback is not None:
                metric_callback(metric)
        except Exception as error:
            local_error = error
    _checkpoint_phase_decision("metrics journal append", local_error)


def _load_exact_checkpoint_control_with_retry(
    path: Path,
    expected: Mapping[str, Any],
    *,
    timeout_seconds: float = CHECKPOINT_CONTROL_VISIBILITY_TIMEOUT_SECONDS,
    poll_seconds: float = CHECKPOINT_CONTROL_VISIBILITY_POLL_SECONDS,
    clock: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> None:
    """Wait boundedly until one mutable control path has the exact new value.

    A shared filesystem client can retain an ENOENT or the previous contents
    after another client atomically replaces a file.  Reading the pathname
    directly (without an ``exists()`` branch) and waiting for the exact value
    handles both cases while keeping the surrounding collective schedule fixed.
    Malformed visible JSON is not treated as propagation lag.
    """

    if (
        not math.isfinite(timeout_seconds)
        or timeout_seconds < 0.0
        or not math.isfinite(poll_seconds)
        or poll_seconds <= 0.0
    ):
        raise ValueError("checkpoint visibility retry bounds are invalid")
    deadline = clock() + timeout_seconds
    while True:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            payload = None
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"checkpoint control file contains malformed JSON: {path}"
            ) from error
        except OSError as error:
            raise RuntimeError(
                f"cannot read checkpoint control file: {path}"
            ) from error
        if payload == dict(expected):
            return
        remaining = deadline - clock()
        if remaining <= 0.0:
            raise RuntimeError(
                "checkpoint control publication did not become exactly visible "
                f"before the bounded retry expired: {path}"
            )
        sleep(min(poll_seconds, remaining))


def _publish_checkpoint_pointer(
    checkpoint_dir: str | Path,
    pointer: Mapping[str, Any],
) -> str:
    destination: Path | None = None
    expected: dict[str, Any] | None = None
    prepare_error: Exception | None = None
    try:
        kind = str(pointer["kind"])
        if kind not in {"latest", "final"}:
            raise ValueError("checkpoint pointer kind must be latest or final")
        int(pointer["tokens_seen"])
        destination = pointer_path(checkpoint_dir, kind)
        expected = dict(pointer)
    except Exception as error:  # decision must precede every process raising
        prepare_error = error
    _checkpoint_phase_decision("pointer preparation", prepare_error)
    assert destination is not None and expected is not None

    publish_error: Exception | None = None
    if jax.process_index() == 0:
        try:
            publish_pointer(checkpoint_dir, expected)
        except Exception as error:  # decision must precede every process raising
            publish_error = error
    _checkpoint_phase_decision("pointer publication", publish_error)

    visibility_error: Exception | None = None
    try:
        _load_exact_checkpoint_control_with_retry(destination, expected)
    except Exception as error:  # decision must precede every process raising
        visibility_error = error
    _checkpoint_phase_decision("pointer visibility", visibility_error)
    return str(destination)


def _delete_stale_checkpoint_payload(payload_uri: str, *, tokens_seen: int) -> None:
    """Collectively remove an immutable payload after its pointer is superseded."""

    _checkpoint_barrier(f"gemma4-sae-stale-checkpoint-ready-{tokens_seen}")
    failed = 0
    if jax.process_index() == 0:
        try:
            from etils import epath

            path = epath.Path(payload_uri)
            path.rmtree(missing_ok=True)
            if path.exists():
                raise RuntimeError("stale checkpoint payload still exists")
        except Exception:
            failed = 1
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        failures = np.asarray(
            multihost_utils.process_allgather(
                np.asarray([failed], dtype=np.int32), tiled=False
            )
        )
        failed = int(np.max(failures))
    _checkpoint_barrier(f"gemma4-sae-stale-checkpoint-done-{tokens_seen}")
    if failed:
        raise RuntimeError(
            "new latest pointer is durable, but stale checkpoint cleanup failed: "
            f"{payload_uri}"
        )


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
    destination: Path | None = None
    payload: dict[str, Any] | None = None
    prepare_error: Exception | None = None
    try:
        destination = _checkpoint_timing_path(checkpoint_dir)
        payload = {
            **checkpoint_identity,
            "tokens_seen": tokens_seen,
            "steps": steps,
            "optimizer_wall_seconds": optimizer_wall_seconds,
            "compile_seconds": compile_seconds,
            "checkpoint_seconds": checkpoint_seconds,
        }
        # Prove the payload is JSON-compatible before process zero mutates the
        # shared control plane.  The round trip also gives every process the
        # same exact value used for visibility validation.
        payload = json.loads(json.dumps(payload, allow_nan=False))
    except Exception as error:  # decision must precede every process raising
        prepare_error = error
    _checkpoint_phase_decision("timing preparation", prepare_error)
    assert destination is not None and payload is not None

    publish_error: Exception | None = None
    if jax.process_index() == 0:
        try:
            atomic_write_json(destination, payload)
        except Exception as error:  # decision must precede every process raising
            publish_error = error
    _checkpoint_phase_decision("timing publication", publish_error)

    visibility_error: Exception | None = None
    try:
        _load_exact_checkpoint_control_with_retry(destination, payload)
    except Exception as error:  # decision must precede every process raising
        visibility_error = error
    _checkpoint_phase_decision("timing visibility", visibility_error)


def _load_checkpoint_timing(
    checkpoint_dir: str | Path,
    *,
    checkpoint_identity: Mapping[str, Any],
    restored_tokens: int,
) -> dict[str, Any] | None:
    path: Path | None = None
    prepare_error: Exception | None = None
    try:
        path = _checkpoint_timing_path(checkpoint_dir)
    except Exception as error:
        prepare_error = error
    _checkpoint_phase_decision("timing load preparation", prepare_error)
    assert path is not None
    value = _coordinator_read_checkpoint_json_if_exists(
        path, operation="checkpoint timing"
    )
    if value is None:
        return None
    payload: dict[str, Any] | None = None
    validation_error: Exception | None = None
    try:
        if not isinstance(value, Mapping):
            raise RuntimeError("checkpoint timing sidecar must be a JSON object")
        payload = dict(value)
        if any(
            payload.get(name) != expected
            for name, expected in checkpoint_identity.items()
        ):
            raise RuntimeError("checkpoint timing sidecar belongs to another run")
        sidecar_tokens = int(payload.get("tokens_seen", -1))
        if sidecar_tokens > restored_tokens:
            raise RuntimeError("checkpoint timing sidecar is ahead of restored state")
        if sidecar_tokens < restored_tokens:
            payload = None
        else:
            for name in (
                "optimizer_wall_seconds",
                "compile_seconds",
                "checkpoint_seconds",
            ):
                timing_value = float(payload.get(name, -1.0))
                if not math.isfinite(timing_value) or timing_value < 0.0:
                    raise RuntimeError(
                        f"checkpoint timing sidecar has invalid {name}"
                    )
    except Exception as error:
        validation_error = error
    _checkpoint_phase_decision("timing validation", validation_error)
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
    checkpoint_payload_root: str | Path | None = None,
    stop_after_tokens: int | None = None,
    require_health_pass: bool = True,
    metric_callback: Any | None = None,
    process_local_inputs: bool = False,
    prior_optimizer_wall_seconds: float = 0.0,
    prior_compile_seconds: float = 0.0,
    prior_checkpoint_seconds: float = 0.0,
    process_local_source_offset: int = 0,
    checkpoint_identity: Mapping[str, Any] | None = None,
    metrics_journal_path: str | Path | None = None,
    metrics_journal_identity: JournalIdentity | None = None,
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
    payload_root: str | None = None
    if checkpoint_dir is not None:
        payload_root = _effective_checkpoint_payload_root(
            checkpoint_dir, checkpoint_payload_root
        )
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
    start_step = int(np.asarray(jax.device_get(state.step)))
    existing_latest_pointer: dict[str, Any] | None = None
    if checkpoint_dir is not None:
        assert payload_root is not None
        existing_latest_pointer = _collective_load_pointer(
            checkpoint_dir,
            kind="latest",
            payload_root=payload_root,
            run_identity=checkpoint_identity,
        )
        if existing_latest_pointer is not None:
            _collectively_require_finalized_checkpoint(
                str(existing_latest_pointer["payload_uri"])
            )
            state_error: Exception | None = None
            try:
                if (
                    int(existing_latest_pointer["tokens_seen"]) != start_tokens
                    or int(existing_latest_pointer["step"]) != start_step
                ):
                    raise RuntimeError(
                        "training state does not match the validated latest "
                        "checkpoint pointer"
                    )
            except Exception as error:
                state_error = error
            _checkpoint_phase_decision("resume state validation", state_error)
    if start_tokens > training_config.total_tokens:
        raise ValueError("checkpoint exceeds configured token budget")
    if (
        start_tokens < training_config.total_tokens
        and start_tokens % training_config.effective_batch_tokens
    ):
        raise ValueError("non-final distributed resume must be on an update boundary")
    metrics_journal, journal_prefix = _prepare_training_journal(
        metrics_journal_path,
        metrics_journal_identity,
        training_config,
        restored_step=start_step,
        restored_tokens=start_tokens,
    )

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
    latest_checkpoint: str | None = (
        str(pointer_path(checkpoint_dir, "latest"))
        if existing_latest_pointer is not None and checkpoint_dir is not None
        else None
    )
    final_checkpoint: str | None = None
    latest_pointer = existing_latest_pointer
    latest_saved_tokens: int | None = (
        int(existing_latest_pointer["tokens_seen"])
        if existing_latest_pointer is not None
        else None
    )
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

        tokens_seen = int(np.asarray(jax.device_get(state.tokens_seen)))
        step = int(np.asarray(jax.device_get(state.step)))
        if metrics_journal_path is not None or metric_callback is not None:
            journal_started = time.perf_counter()
            cumulative_before_append = (
                prior_optimizer_wall_seconds
                + _global_max_seconds(optimizer_seconds_local)
                if metrics_journal_path is not None
                else 0.0
            )
            _append_training_journal_metric(
                metrics_journal,
                metric,
                step=step,
                tokens_seen=tokens_seen,
                optimizer_wall_seconds=cumulative_before_append,
                metric_callback=metric_callback,
            )
            journal_phase_seconds = time.perf_counter() - journal_started
            optimizer_seconds_local += journal_phase_seconds
            step_seconds[-1] += journal_phase_seconds
        if checkpoint_dir is not None and tokens_seen >= next_checkpoint:
            assert payload_root is not None
            cumulative_optimizer = prior_optimizer_wall_seconds + _global_max_seconds(
                optimizer_seconds_local
            )
            cumulative_compile = prior_compile_seconds + _global_max_seconds(
                compile_seconds
            )
            checkpoint_started = time.perf_counter()
            attempt_id = _collective_attempt_id()
            payload_uri = checkpoint_payload_uri(
                payload_root,
                checkpoint_identity,
                step=step,
                tokens_seen=tokens_seen,
                attempt_id=attempt_id,
            )
            save_distributed_checkpoint(
                payload_uri,
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
            previous_payload_uri = (
                str(latest_pointer["payload_uri"])
                if latest_pointer is not None
                else None
            )
            new_latest_pointer = make_pointer(
                kind="latest",
                payload_uri=payload_uri,
                step=step,
                tokens_seen=tokens_seen,
                attempt_id=attempt_id,
                run_identity=checkpoint_identity,
            )
            latest_checkpoint = _publish_checkpoint_pointer(
                checkpoint_dir, new_latest_pointer
            )
            if (
                previous_payload_uri is not None
                and previous_payload_uri != payload_uri
            ):
                _delete_stale_checkpoint_payload(
                    previous_payload_uri, tokens_seen=tokens_seen
                )
            latest_pointer = new_latest_pointer
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
        assert payload_root is not None
        cumulative_optimizer = prior_optimizer_wall_seconds + _global_max_seconds(
            optimizer_seconds_local
        )
        cumulative_compile = prior_compile_seconds + _global_max_seconds(
            compile_seconds
        )
        if latest_saved_tokens != tokens_seen:
            checkpoint_started = time.perf_counter()
            step = int(np.asarray(jax.device_get(state.step)))
            attempt_id = _collective_attempt_id()
            payload_uri = checkpoint_payload_uri(
                payload_root,
                checkpoint_identity,
                step=step,
                tokens_seen=tokens_seen,
                attempt_id=attempt_id,
            )
            save_distributed_checkpoint(
                payload_uri,
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
            previous_payload_uri = (
                str(latest_pointer["payload_uri"])
                if latest_pointer is not None
                else None
            )
            new_latest_pointer = make_pointer(
                kind="latest",
                payload_uri=payload_uri,
                step=step,
                tokens_seen=tokens_seen,
                attempt_id=attempt_id,
                run_identity=checkpoint_identity,
            )
            latest_checkpoint = _publish_checkpoint_pointer(
                checkpoint_dir, new_latest_pointer
            )
            if (
                previous_payload_uri is not None
                and previous_payload_uri != payload_uri
            ):
                _delete_stale_checkpoint_payload(
                    previous_payload_uri, tokens_seen=tokens_seen
                )
            latest_pointer = new_latest_pointer
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
            if latest_pointer is None or latest_saved_tokens != tokens_seen:
                raise RuntimeError("final checkpoint has no matching latest payload")
            checkpoint_started = time.perf_counter()
            final_pointer = final_pointer_from_latest(latest_pointer)
            final_checkpoint = _publish_checkpoint_pointer(
                checkpoint_dir, final_pointer
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

    journal_snapshot = journal_prefix
    if metrics_journal_path is not None:
        journal_snapshot = _collective_training_journal_snapshot(
            metrics_journal,
            training_config,
            operation="metrics journal final prefix",
        )
        journal_binding = journal_snapshot.get("binding")
        if (
            not isinstance(journal_binding, Mapping)
            or int(journal_binding.get("last_step", -1))
            != int(np.asarray(jax.device_get(state.step)))
            or int(journal_binding.get("last_tokens_seen", -1)) != tokens_seen
        ):
            raise RuntimeError("final metrics journal does not match training state")

    health = None
    if tokens_seen >= training_config.health_gate_tokens:
        last_l0 = (
            float(np.asarray(jax.device_get(metrics[-1].l0)))
            if metrics
            else float(training_config.sae.k)
        )
        if journal_snapshot is not None:
            if journal_snapshot.get("complete_prefix_exact_l0") is not True:
                last_l0 = math.nan
            topk_valid = (
                journal_snapshot.get("complete_prefix_topk_valid") is True
            )
        else:
            topk_valid = all(
                bool(np.asarray(jax.device_get(metric.topk_valid)))
                for metric in metrics
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
        metrics_journal=journal_snapshot,
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
    """Collectively save one immutable payload; every host must call."""

    bound_metadata = dict(metadata or {})
    reserved = {
        "checkpoint_payload_uri": str(path),
        "tokens_seen": int(np.asarray(jax.device_get(state.tokens_seen))),
        "steps": int(np.asarray(jax.device_get(state.step))),
    }
    conflicts = {
        name: (bound_metadata.get(name), value)
        for name, value in reserved.items()
        if name in bound_metadata and bound_metadata[name] != value
    }
    if conflicts:
        raise ValueError(
            "checkpoint metadata conflicts with payload/state: "
            + ", ".join(sorted(conflicts))
        )
    bound_metadata.update(reserved)
    ocp = _import_orbax()
    checkpointer = ocp.StandardCheckpointer()
    try:
        checkpointer.save(
            path,
            state,
            force=False,
            custom_metadata=_checkpoint_header(
                training_config, mesh, bound_metadata
            ),
        )
        checkpointer.wait_until_finished()
        if not _orbax_path_finalized(ocp, path):
            raise RuntimeError(f"Orbax checkpoint payload is not finalized: {path}")
    finally:
        checkpointer.close()


def _restore_distributed_payload(
    path: str | Path,
    template: TrainState,
    mesh: Mesh,
    training_config: TrainingConfig,
) -> tuple[TrainState, dict[str, Any]]:
    """Restore a payload after its control pointer has already been validated."""

    ocp = _import_orbax()
    checkpointer = ocp.StandardCheckpointer()
    try:
        checkpoint_metadata = checkpointer.metadata(path).custom_metadata or {}
        expected = _checkpoint_header(training_config, mesh, None)
        if (
            checkpoint_metadata.get("checkpoint_version")
            != expected["checkpoint_version"]
        ):
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
    finally:
        checkpointer.close()


def restore_distributed_checkpoint(
    path: str | Path,
    template: TrainState,
    mesh: Mesh,
    training_config: TrainingConfig,
    *,
    checkpoint_payload_root: str | Path | None = None,
    checkpoint_identity: Mapping[str, Any] | None = None,
) -> tuple[TrainState, dict[str, Any]]:
    """Restore only through a validated ``latest.json`` or ``final.json`` pointer."""

    if "://" in str(path):
        raise ValueError("checkpoint control pointer must be a POSIX path")
    pointer_file = Path(path)
    if pointer_file.name not in {"latest.json", "final.json"}:
        raise ValueError(
            "distributed checkpoint restore requires latest.json or final.json"
        )
    kind = "latest" if pointer_file.name == "latest.json" else "final"
    identity = dict(checkpoint_identity or {})
    payload_root = _effective_checkpoint_payload_root(
        pointer_file.parent, checkpoint_payload_root
    )
    pointer = _collective_load_pointer(
        pointer_file.parent,
        kind=kind,
        payload_root=payload_root,
        run_identity=identity,
    )
    if pointer is None:
        raise RuntimeError(f"checkpoint pointer is missing: {pointer_file}")
    _collectively_require_finalized_checkpoint(str(pointer["payload_uri"]))
    restored: TrainState | None = None
    metadata: dict[str, Any] | None = None
    restore_error: Exception | None = None
    try:
        restored, metadata = _restore_distributed_payload(
            str(pointer["payload_uri"]), template, mesh, training_config
        )
        restored_tokens = int(np.asarray(jax.device_get(restored.tokens_seen)))
        restored_step = int(np.asarray(jax.device_get(restored.step)))
        if (
            int(pointer["tokens_seen"]) != restored_tokens
            or int(pointer["step"]) != restored_step
        ):
            raise RuntimeError(
                "checkpoint pointer disagrees with restored payload state"
            )
        if (
            metadata.get("checkpoint_payload_uri") != pointer["payload_uri"]
            or metadata.get("tokens_seen") != restored_tokens
            or metadata.get("steps") != restored_step
        ):
            raise RuntimeError(
                "checkpoint payload metadata disagrees with its pointer"
            )
        mismatches = {
            name: (metadata.get(name), expected)
            for name, expected in identity.items()
            if metadata.get(name) != expected
        }
        if mismatches:
            raise RuntimeError(
                "checkpoint payload belongs to a different run: "
                + ", ".join(sorted(mismatches))
            )
    except Exception as error:
        restore_error = error
    _checkpoint_phase_decision("payload restore", restore_error)
    assert restored is not None and metadata is not None
    return restored, metadata


def _orbax_path_finalized(ocp: Any, path: str | Path) -> bool:
    """Call Orbax's synchronous public wrapper around ``is_path_finalized``."""

    # Orbax 0.12 exposes the underlying `is_path_finalized` implementation as
    # `is_checkpoint_finalized`; support the direct name as well for compatible
    # patch releases without reaching into Orbax private modules.
    finalized = getattr(ocp.utils, "is_path_finalized", None)
    if finalized is None:
        finalized = getattr(ocp.utils, "is_checkpoint_finalized", None)
    if finalized is None:
        raise RuntimeError("installed Orbax does not expose path finalization checks")
    return bool(finalized(path))


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
    local_microbatch_tokens = training_config.microbatch_tokens // data_size
    worst = 0

    def observe(ids: np.ndarray) -> None:
        nonlocal worst
        counts = np.bincount(
            np.asarray(ids).reshape(-1),
            minlength=training_config.sae.num_experts,
        )
        worst = max(worst, int(counts.max(initial=0)))

    scan_error: Exception | None = None
    try:
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
        if route_only:
            micro = local_microbatch_tokens
            buffered = np.empty(
                (0, training_config.sae.top_routed), dtype=np.int32
            )
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
            if (
                total_tokens is not None
                and total_tokens != training_config.total_tokens
            ):
                raise TypeError(
                    "capacity token override requires ArtifactDataset route-only input"
                )
            local_config = dataclasses.replace(
                training_config,
                total_tokens=local_tokens,
                effective_batch_tokens=(
                    training_config.effective_batch_tokens // data_size
                ),
                microbatch_tokens=local_microbatch_tokens,
            )
            for batch in _fixed_microbatches(
                source, local_config, skip_tokens=0, take_tokens=local_tokens
            ):
                observe(np.asarray(batch.expert_ids))
    except Exception as error:
        scan_error = error
    _checkpoint_phase_decision("condition-A capacity scan", scan_error)
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
    checkpoint_payload_root: str | Path | None = None,
    stop_after_tokens: int | None = None,
    metric_callback: Any | None = None,
    checkpoint_identity: Mapping[str, Any] | None = None,
    prior_run_timing: Mapping[str, Any] | None = None,
    metrics_journal_path: str | Path | None = None,
    metrics_journal_identity: JournalIdentity | None = None,
) -> DistributedTrainingResult:
    """CLI-facing multi-host train/resume entrypoint.

    ``batches`` must be an :class:`ArtifactDataset` on multi-host runs so each
    process loads only its contiguous token partition.  Resume follows only a
    validated ``latest.json`` control pointer; completed runs publish
    ``final.json`` to the same immutable payload as the final latest pointer.
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
    payload_root: str | None = None
    if checkpoint_dir is not None:
        payload_root = _effective_checkpoint_payload_root(
            checkpoint_dir, checkpoint_payload_root
        )
        latest = _collective_load_pointer(
            checkpoint_dir,
            kind="latest",
            payload_root=payload_root,
            run_identity=checkpoint_identity,
        )
        if latest is not None:
            state, prior_timing = restore_distributed_checkpoint(
                pointer_path(checkpoint_dir, "latest"),
                state,
                mesh,
                training_config,
                checkpoint_payload_root=payload_root,
                checkpoint_identity=checkpoint_identity,
            )
            if (
                int(latest["tokens_seen"])
                != int(np.asarray(jax.device_get(state.tokens_seen)))
                or int(latest["step"])
                != int(np.asarray(jax.device_get(state.step)))
            ):
                raise RuntimeError(
                    "latest checkpoint pointer disagrees with restored payload state"
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
        checkpoint_payload_root=payload_root,
        stop_after_tokens=stop_after_tokens,
        require_health_pass=False,
        # Metrics are logically replicated across hosts; the inner loop lets
        # only process zero mutate the shared journal/callback and then gives
        # every peer one collective outcome before another update begins.
        metric_callback=metric_callback,
        process_local_inputs=True,
        prior_optimizer_wall_seconds=float(
            prior_timing.get("optimizer_wall_seconds", 0.0)
        ),
        prior_compile_seconds=float(prior_timing.get("compile_seconds", 0.0)),
        prior_checkpoint_seconds=float(prior_timing.get("checkpoint_seconds", 0.0)),
        process_local_source_offset=applied_offset,
        checkpoint_identity=checkpoint_identity,
        metrics_journal_path=metrics_journal_path,
        metrics_journal_identity=metrics_journal_identity,
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
    checkpoint_payload_root: str | Path | None = None,
    evaluation_a_capacity: int | None = None,
) -> EvaluationResult:
    """CLI-facing evaluation restored only through a final control pointer.

    ``training_config`` must remain byte-for-byte identical to the configuration
    stored in the checkpoint.  Condition A may need a different static routing
    capacity for the held-out token order, so that execution-only value is
    supplied separately and is never folded into restore identity.
    """

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
    checkpoint_identity = dict(checkpoint_identity or {})
    if "://" in str(checkpoint_path):
        raise ValueError("evaluation checkpoint pointer must be a POSIX path")
    pointer_file = pointer_path(Path(checkpoint_path).parent, "final")
    if str(pointer_file) != str(Path(checkpoint_path)):
        raise ValueError("evaluation checkpoint path must be final.json")
    state, _ = restore_distributed_checkpoint(
        pointer_file,
        template,
        mesh,
        training_config,
        checkpoint_payload_root=checkpoint_payload_root,
        checkpoint_identity=checkpoint_identity,
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
    effective_a_capacity = (
        evaluation_a_capacity
        if training_config.sae.condition == "A"
        else training_config.a_capacity
    )
    if training_config.sae.condition == "A" and effective_a_capacity is None:
        raise ValueError("condition A evaluation requires a routing capacity")
    return distributed_evaluate(
        state.params,
        source,
        mean,
        training_config.sae,
        mesh,
        expected_sst=expected_sst,
        max_tokens=local_max,
        microbatch_tokens=local_microbatch,
        a_capacity=effective_a_capacity,
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
