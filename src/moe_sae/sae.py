"""Gemma 4 router-conditioned sparse autoencoder primitives.

This module intentionally contains only the scientific hot path.  It has no
knowledge of capture formats, GCS, MaxText, or command-line configuration.  A
small single-device implementation is useful as a correctness oracle; the
selection and sparse-decode APIs are also shaped so that feature-sharded TPU
code can use the same semantics.

The three supported conditions reconstruct one common, globally-centred
activation target:

``A``
    128 banks of 1,024 features.  A token can see the eight banks selected by
    Gemma's router (8,192 candidates).
``B``
    One global dictionary of 8,192 features (candidate matched to A).
``F``
    One global dictionary of 131,072 features (parameter matched to A).

Router weights are deliberately absent from every function signature.  Only
the expert ids define condition A's candidate set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


CONDITIONS = ("A", "B", "F")
DEFAULT_HIDDEN_DIM = 2_816
DEFAULT_NUM_EXPERTS = 128
DEFAULT_TOP_ROUTED = 8
DEFAULT_BANK_WIDTH = 1_024
DEFAULT_TOP_K = 64
DEFAULT_TOTAL_TOKENS = 50_000_000
DEFAULT_BATCH_TOKENS = 16_384
DEFAULT_MICROBATCH_TOKENS = 4_096
DEFAULT_UPDATES = 3_052
DEFAULT_FINAL_PADDING = 3_968
_EPS = 1e-12


def _dtype(name: str) -> jnp.dtype:
    try:
        return {
            "float32": jnp.float32,
            "bfloat16": jnp.bfloat16,
        }[name]
    except KeyError as exc:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unsupported parameter dtype: {name!r}") from exc


@dataclass(frozen=True)
class SAEConfig:
    """Static architecture and optimizer configuration for one condition."""

    condition: str
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_experts: int = DEFAULT_NUM_EXPERTS
    top_routed: int = DEFAULT_TOP_ROUTED
    bank_width: int = DEFAULT_BANK_WIDTH
    k: int = DEFAULT_TOP_K
    parameter_dtype: str = "bfloat16"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    peak_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 250
    total_steps: int = DEFAULT_UPDATES

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(
                f"condition must be one of {CONDITIONS}, got {self.condition!r}"
            )
        for name in ("hidden_dim", "num_experts", "top_routed", "bank_width", "k"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.top_routed > self.num_experts:
            raise ValueError("top_routed cannot exceed num_experts")
        if self.k > self.candidates_per_token:
            raise ValueError("k cannot exceed the visible candidate count")
        if self.parameter_dtype not in ("float32", "bfloat16"):
            raise ValueError("parameter_dtype must be 'float32' or 'bfloat16'")
        if not (0.0 <= self.adam_beta1 < 1.0 and 0.0 <= self.adam_beta2 < 1.0):
            raise ValueError("Adam betas must be in [0, 1)")
        if self.adam_eps <= 0 or self.grad_clip <= 0:
            raise ValueError("adam_eps and grad_clip must be positive")
        if not (0.0 < self.min_lr <= self.peak_lr):
            raise ValueError("learning rates must satisfy 0 < min_lr <= peak_lr")
        if self.warmup_steps < 0 or self.total_steps <= 0:
            raise ValueError("invalid LR schedule length")

    @property
    def total_features(self) -> int:
        if self.condition == "B":
            return self.top_routed * self.bank_width
        return self.num_experts * self.bank_width

    @property
    def candidates_per_token(self) -> int:
        if self.condition in ("A", "B"):
            return self.top_routed * self.bank_width
        return self.num_experts * self.bank_width

    @property
    def parameter_count(self) -> int:
        # encoder + decoder + encoder bias; the centring mean is not trained.
        return 2 * self.hidden_dim * self.total_features + self.total_features


class SAEParams(NamedTuple):
    """Trainable arrays.

    ``encoder`` is [features, hidden], ``decoder`` is [hidden, features], and
    decoder feature vectors are therefore columns (axis 0 is normalized).
    """

    encoder: jax.Array
    decoder: jax.Array
    bias: jax.Array


class AdamState(NamedTuple):
    m: SAEParams
    v: SAEParams
    step: jax.Array


class TrainState(NamedTuple):
    params: SAEParams
    optimizer: AdamState
    step: jax.Array
    tokens_seen: jax.Array
    fired_mask: jax.Array
    loss_ema: jax.Array
    initial_loss_ema: jax.Array
    ema_updates: jax.Array


class Selection(NamedTuple):
    """Exactly-k selected positive features for each token.

    If ``valid[row]`` is false, fewer than k scores were strictly positive for
    that row.  The ids/values in that row are placeholders and callers must
    fail the step (masked padding rows are the sole exception).
    """

    values: jax.Array  # [tokens, k], float32
    latent_ids: jax.Array  # [tokens, k], int32 global ids
    valid: jax.Array  # [tokens], bool


class ForwardOutput(NamedTuple):
    reconstruction: jax.Array  # [tokens, hidden]
    selection: Selection


class LossSums(NamedTuple):
    sse: jax.Array
    sst: jax.Array
    l0_sum: jax.Array
    valid_tokens: jax.Array
    topk_valid: jax.Array
    fired_mask: jax.Array


class StepMetrics(NamedTuple):
    loss: jax.Array
    sse: jax.Array
    sst: jax.Array
    l0: jax.Array
    valid_tokens: jax.Array
    topk_valid: jax.Array
    grad_norm: jax.Array
    clip_coefficient: jax.Array
    learning_rate: jax.Array
    live_fraction: jax.Array


@dataclass(frozen=True)
class GroupingPlan:
    """Capacity-padded routing plan for condition A's grouped GEMM."""

    order: np.ndarray
    destination: np.ndarray
    capacity: int
    num_experts: int


def make_grouping_plan(
    expert_ids: np.ndarray | jax.Array,
    *,
    num_experts: int,
    capacity: int | None = None,
    margin: float = 0.10,
    round_to: int = 8,
) -> GroupingPlan:
    """Build a stable grouped-GEMM plan and fail rather than drop overflow.

    Expert ids are flattened in token-major order.  When ``capacity`` is not
    supplied, this reference chooses the observed maximum plus the configured
    margin.  Production callers should scan their fixed seed order once and
    pass the resulting static capacity so every step has the same XLA shape.
    """

    ids = np.asarray(expert_ids, dtype=np.int64)
    if ids.ndim != 2:
        raise ValueError(f"expert_ids must be rank 2, got shape {ids.shape}")
    if ids.size == 0:
        raise ValueError("expert_ids cannot be empty")
    if ids.min() < 0 or ids.max() >= num_experts:
        raise ValueError("expert id outside configured range")
    if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
        raise ValueError("router expert ids must be unique within every token")

    flat = ids.reshape(-1)
    counts = np.bincount(flat, minlength=num_experts)
    observed = int(counts.max(initial=0))
    if capacity is None:
        required = max(observed, int(math.ceil(observed * (1.0 + margin))))
        capacity = max(round_to, int(math.ceil(required / round_to) * round_to))
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if observed > capacity:
        worst = int(counts.argmax())
        raise RuntimeError(
            f"expert {worst} received {observed} rows > static capacity {capacity}; "
            "increase the precomputed capacity (tokens are never dropped)"
        )

    order = np.argsort(flat, kind="stable")
    sorted_experts = flat[order]
    starts = np.zeros(num_experts, dtype=np.int64)
    starts[1:] = np.cumsum(counts, dtype=np.int64)[:-1]
    rank = np.arange(flat.size, dtype=np.int64) - starts[sorted_experts]
    destination = sorted_experts * int(capacity) + rank
    return GroupingPlan(
        order=order.astype(np.int32),
        destination=destination.astype(np.int32),
        capacity=int(capacity),
        num_experts=num_experts,
    )


def init_params(key: jax.Array, config: SAEConfig) -> SAEParams:
    """Initialize unit decoder columns and the transposed encoder."""

    dtype = _dtype(config.parameter_dtype)
    decoder = jax.random.normal(
        key, (config.hidden_dim, config.total_features), dtype=jnp.float32
    )
    decoder = decoder / jnp.maximum(
        jnp.linalg.norm(decoder, axis=0, keepdims=True), _EPS
    )
    decoder = decoder.astype(dtype)
    return SAEParams(
        encoder=decoder.T.copy(),
        decoder=decoder,
        bias=jnp.zeros((config.total_features,), dtype=dtype),
    )


def _zeros_like_params(params: SAEParams) -> SAEParams:
    return SAEParams(*[jnp.zeros_like(x, dtype=jnp.float32) for x in params])


def init_adam(params: SAEParams) -> AdamState:
    zeros = _zeros_like_params(params)
    return AdamState(m=zeros, v=zeros, step=jnp.asarray(0, jnp.int32))


def init_train_state(key: jax.Array, config: SAEConfig) -> TrainState:
    params = init_params(key, config)
    return TrainState(
        params=params,
        optimizer=init_adam(params),
        step=jnp.asarray(0, jnp.int32),
        tokens_seen=jnp.asarray(0, jnp.int32),
        fired_mask=jnp.zeros((config.total_features,), dtype=bool),
        loss_ema=jnp.asarray(jnp.nan, jnp.float32),
        initial_loss_ema=jnp.asarray(jnp.nan, jnp.float32),
        ema_updates=jnp.asarray(0, jnp.int32),
    )


def learning_rate(step: int | jax.Array, config: SAEConfig) -> jax.Array:
    """250-update linear warmup followed by cosine decay to ``min_lr``."""

    step_f = jnp.asarray(step, jnp.float32)
    update_number = step_f + 1.0
    if config.warmup_steps:
        warm = config.peak_lr * jnp.minimum(
            update_number / float(config.warmup_steps), 1.0
        )
    else:
        warm = jnp.asarray(config.peak_lr, jnp.float32)

    decay_steps = max(config.total_steps - config.warmup_steps, 1)
    progress = jnp.clip(
        (update_number - float(config.warmup_steps)) / float(decay_steps), 0.0, 1.0
    )
    cosine = config.min_lr + 0.5 * (config.peak_lr - config.min_lr) * (
        1.0 + jnp.cos(jnp.pi * progress)
    )
    return jnp.where(update_number <= config.warmup_steps, warm, cosine).astype(
        jnp.float32
    )


def _positive_topk_sorted_ids(
    scores: jax.Array, sorted_ids: jax.Array, k: int
) -> Selection:
    """TopK where candidate ids are strictly ascending on every row.

    ``lax.top_k`` is used only to find the numeric cutoff.  Membership at a
    tied cutoff is then reconstructed explicitly: scores above the cutoff win,
    followed by the lowest global latent ids equal to the cutoff.  This avoids
    backend-dependent tie behavior without perturbing floating-point scores.
    """

    scores32 = jnp.asarray(scores, jnp.float32)
    ids = jnp.asarray(sorted_ids, jnp.int32)
    if scores32.ndim != 2 or ids.shape != scores32.shape:
        raise ValueError("scores and latent ids must have the same rank-2 shape")
    if k <= 0 or k > scores32.shape[1]:
        raise ValueError("k must be in [1, number of candidates]")

    positive = scores32 > 0.0
    ranked = jnp.where(positive, scores32, -jnp.inf)
    cutoff = jax.lax.top_k(ranked, k)[0][:, -1]
    above = ranked > cutoff[:, None]
    tied = positive & (ranked == cutoff[:, None])
    needed = k - jnp.sum(above, axis=1)
    tie_rank = jnp.cumsum(tied.astype(jnp.int32), axis=1)
    selected_mask = above | (tied & (tie_rank <= needed[:, None]))
    valid = jnp.sum(positive, axis=1) >= k

    # IDs are already ascending, so nonzero returns selected membership in the
    # deterministic global-id order.  Invalid rows are placeholders.
    positions = jax.vmap(lambda row: jnp.nonzero(row, size=k, fill_value=0)[0])(
        selected_mask
    )
    selected_values = jnp.take_along_axis(scores32, positions, axis=1)
    selected_ids = jnp.take_along_axis(ids, positions, axis=1)
    selected_values = jnp.where(valid[:, None], selected_values, 0.0)
    return Selection(selected_values, selected_ids, valid)


def positive_topk(scores: jax.Array, latent_ids: jax.Array, k: int) -> Selection:
    """Exact positive per-token TopK with global latent-id tie breaking.

    Candidate ids may arrive in any order; this function first sorts them.  It
    returns a validity bit instead of raising from inside a traced JAX program.
    Training/evaluation orchestration must hard-fail any invalid non-padding
    token.
    """

    scores = jnp.asarray(scores)
    ids = jnp.asarray(latent_ids, jnp.int32)
    if ids.ndim == 1:
        ids = jnp.broadcast_to(ids[None, :], scores.shape)
    if scores.ndim != 2 or ids.shape != scores.shape:
        raise ValueError(
            "scores must be [tokens, candidates] and ids must broadcast to it"
        )
    order = jnp.argsort(ids, axis=1, stable=True)
    sorted_ids = jnp.take_along_axis(ids, order, axis=1)
    sorted_scores = jnp.take_along_axis(scores, order, axis=1)
    return _positive_topk_sorted_ids(sorted_scores, sorted_ids, k)


def local_topk_candidates(
    local_scores: jax.Array, global_latent_ids: jax.Array, k: int
) -> Selection:
    """Return up to k deterministic candidates from one feature shard.

    Invalid slots are encoded as zero values with the largest int32 id.  A
    local shard is allowed to have fewer than k positive features; validity is
    checked only after all shards are merged.
    """

    scores = jnp.asarray(local_scores, jnp.float32)
    ids = jnp.asarray(global_latent_ids, jnp.int32)
    if ids.ndim == 1:
        ids = jnp.broadcast_to(ids[None, :], scores.shape)
    if scores.shape[1] < k:
        raise ValueError("each feature shard must contain at least k candidates")
    order = jnp.argsort(ids, axis=1, stable=True)
    ids = jnp.take_along_axis(ids, order, axis=1)
    scores = jnp.take_along_axis(scores, order, axis=1)

    positive = scores > 0.0
    ranked = jnp.where(positive, scores, -jnp.inf)
    cutoff = jax.lax.top_k(ranked, k)[0][:, -1]
    above = ranked > cutoff[:, None]
    tied = positive & (ranked == cutoff[:, None])
    needed = k - jnp.sum(above, axis=1)
    mask = above | (
        tied & (jnp.cumsum(tied.astype(jnp.int32), axis=1) <= needed[:, None])
    )
    positions = jax.vmap(lambda row: jnp.nonzero(row, size=k, fill_value=0)[0])(mask)
    values = jnp.take_along_axis(scores, positions, axis=1)
    selected_ids = jnp.take_along_axis(ids, positions, axis=1)
    count = jnp.minimum(jnp.sum(positive, axis=1), k)
    slot_valid = jnp.arange(k)[None, :] < count[:, None]
    values = jnp.where(slot_valid, values, 0.0)
    selected_ids = jnp.where(slot_valid, selected_ids, jnp.iinfo(jnp.int32).max)
    return Selection(values, selected_ids, slot_valid)


def merge_feature_shard_topk(
    shard_values: jax.Array, shard_ids: jax.Array, k: int
) -> Selection:
    """Merge ``[shards, tokens, k]`` local candidates into exact global TopK."""

    values = jnp.asarray(shard_values, jnp.float32)
    ids = jnp.asarray(shard_ids, jnp.int32)
    if values.ndim != 3 or values.shape != ids.shape:
        raise ValueError("shard values/ids must have equal [shards, tokens, k] shape")
    values = jnp.swapaxes(values, 0, 1).reshape(values.shape[1], -1)
    ids = jnp.swapaxes(ids, 0, 1).reshape(ids.shape[1], -1)
    # Sentinel ids have zero values and therefore cannot be selected.
    return positive_topk(values, ids, k)


def feature_sharded_topk(
    local_scores: jax.Array,
    global_latent_ids: jax.Array,
    k: int,
    *,
    axis_name: str | None = None,
) -> Selection:
    """Feature-sharded exact TopK helper suitable for ``shard_map``.

    Within a named feature axis, each shard selects its local k, all-gathers
    only those ``tokens × k`` candidates, and merges them deterministically.
    With no axis name it is equivalent to single-shard ``positive_topk``.
    """

    local = local_topk_candidates(local_scores, global_latent_ids, k)
    if axis_name is None:
        values = local.values[None, ...]
        ids = local.latent_ids[None, ...]
    else:
        values = jax.lax.all_gather(local.values, axis_name, axis=0, tiled=False)
        ids = jax.lax.all_gather(local.latent_ids, axis_name, axis=0, tiled=False)
    return merge_feature_shard_topk(values, ids, k)


def _condition_a_scores(
    params: SAEParams,
    centered: jax.Array,
    expert_ids: jax.Array,
    config: SAEConfig,
    plan: GroupingPlan,
) -> tuple[jax.Array, jax.Array]:
    """Grouped/capacity-padded A encoder; returns sorted candidates."""

    ids = jnp.sort(jnp.asarray(expert_ids, jnp.int32), axis=1)
    batch = centered.shape[0]
    slots = config.top_routed
    if ids.shape != (batch, slots):
        raise ValueError(
            f"expert_ids must have shape {(batch, slots)}, got {ids.shape}"
        )
    if plan.num_experts != config.num_experts:
        raise ValueError("grouping plan/config expert mismatch")

    # The plan was built from sorted ids.  Repeat each token activation once
    # per selected bank, group by bank, run one batched GEMM, then undo sort.
    repeated = jnp.repeat(centered[:, None, :], slots, axis=1).reshape(
        -1, config.hidden_dim
    )
    order = jnp.asarray(plan.order, jnp.int32)
    destination = jnp.asarray(plan.destination, jnp.int32)
    padded = (
        jnp.zeros(
            (config.num_experts * plan.capacity, config.hidden_dim),
            dtype=centered.dtype,
        )
        .at[destination]
        .set(repeated[order])
    )
    bank_encoder = params.encoder.reshape(
        config.num_experts, config.bank_width, config.hidden_dim
    )
    bank_bias = params.bias.reshape(config.num_experts, config.bank_width)
    pre = jnp.einsum(
        "ecd,efd->ecf",
        padded.reshape(config.num_experts, plan.capacity, config.hidden_dim),
        bank_encoder.astype(centered.dtype),
        preferred_element_type=jnp.float32,
    ) + bank_bias[:, None, :].astype(jnp.float32)
    grouped = jax.nn.relu(pre).reshape(
        config.num_experts * plan.capacity, config.bank_width
    )
    flat_scores = (
        jnp.zeros((batch * slots, config.bank_width), dtype=grouped.dtype)
        .at[order]
        .set(grouped[destination])
    )
    scores = flat_scores.reshape(batch, slots * config.bank_width)
    latent_ids = (
        ids[:, :, None] * config.bank_width
        + jnp.arange(config.bank_width, dtype=jnp.int32)[None, None, :]
    ).reshape(batch, -1)
    return scores, latent_ids


def candidate_scores(
    params: SAEParams,
    activation: jax.Array,
    mean: jax.Array,
    config: SAEConfig,
    *,
    expert_ids: jax.Array | None = None,
    grouping_plan: GroupingPlan | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute positive candidate scores and their global latent ids."""

    x = jnp.asarray(activation)
    mu = jnp.asarray(mean, jnp.float32)
    if x.ndim != 2 or x.shape[1] != config.hidden_dim:
        raise ValueError(f"activation must be [tokens, {config.hidden_dim}]")
    if mu.shape != (config.hidden_dim,):
        raise ValueError(f"mean must have shape {(config.hidden_dim,)}")
    centered = x.astype(_dtype(config.parameter_dtype)) - mu.astype(
        _dtype(config.parameter_dtype)
    )

    if config.condition == "A":
        if expert_ids is None:
            raise ValueError("condition A requires expert_ids")
        sorted_ids = np.sort(np.asarray(expert_ids, dtype=np.int32), axis=1)
        if grouping_plan is None:
            grouping_plan = make_grouping_plan(
                sorted_ids, num_experts=config.num_experts
            )
        return _condition_a_scores(params, centered, expert_ids, config, grouping_plan)

    pre = jnp.einsum(
        "bd,fd->bf",
        centered,
        params.encoder.astype(centered.dtype),
        preferred_element_type=jnp.float32,
    ) + params.bias.astype(jnp.float32)
    ids = jnp.broadcast_to(
        jnp.arange(config.total_features, dtype=jnp.int32)[None, :], pre.shape
    )
    return jax.nn.relu(pre), ids


def decode_selected_columns(
    decoder: jax.Array,
    selection: Selection,
    mean: jax.Array,
) -> jax.Array:
    """Decode by gathering k columns; a dense latent vector is never formed."""

    columns = decoder[:, selection.latent_ids]  # [hidden, tokens, k]
    centered_reconstruction = jnp.einsum(
        "bk,dbk->bd",
        selection.values.astype(decoder.dtype),
        columns,
        preferred_element_type=jnp.float32,
    )
    return centered_reconstruction + jnp.asarray(mean, jnp.float32)[None, :]


def decode_selected_columns_sharded(
    local_decoder: jax.Array,
    selection: Selection,
    mean: jax.Array,
    *,
    feature_offset: int | jax.Array,
    axis_name: str | None = None,
) -> jax.Array:
    """Sparse decoder contribution for a contiguous feature shard.

    Each shard gathers only selected columns it owns; a feature-axis psum
    produces the full centred reconstruction.
    """

    offset = jnp.asarray(feature_offset, jnp.int32)
    local_width = local_decoder.shape[1]
    owned = (selection.latent_ids >= offset) & (
        selection.latent_ids < offset + local_width
    )
    local_ids = jnp.clip(selection.latent_ids - offset, 0, local_width - 1)
    columns = local_decoder[:, local_ids]
    values = jnp.where(owned, selection.values, 0.0).astype(local_decoder.dtype)
    partial = jnp.einsum(
        "bk,dbk->bd", values, columns, preferred_element_type=jnp.float32
    )
    centered = partial if axis_name is None else jax.lax.psum(partial, axis_name)
    return centered + jnp.asarray(mean, jnp.float32)[None, :]


def forward(
    params: SAEParams,
    activation: jax.Array,
    mean: jax.Array,
    config: SAEConfig,
    *,
    expert_ids: jax.Array | None = None,
    grouping_plan: GroupingPlan | None = None,
) -> ForwardOutput:
    scores, ids = candidate_scores(
        params,
        activation,
        mean,
        config,
        expert_ids=expert_ids,
        grouping_plan=grouping_plan,
    )
    selection = _positive_topk_sorted_ids(scores, ids, config.k)
    reconstruction = decode_selected_columns(params.decoder, selection, mean)
    return ForwardOutput(reconstruction, selection)


def _loss_sums(
    params: SAEParams,
    activation: jax.Array,
    mean: jax.Array,
    config: SAEConfig,
    expert_ids: jax.Array | None,
    valid_mask: jax.Array,
    grouping_plan: GroupingPlan | None,
) -> LossSums:
    output = forward(
        params,
        activation,
        mean,
        config,
        expert_ids=expert_ids,
        grouping_plan=grouping_plan,
    )
    x = jnp.asarray(activation, jnp.float32)
    mu = jnp.asarray(mean, jnp.float32)
    valid = jnp.asarray(valid_mask, bool)
    if valid.shape != (x.shape[0],):
        raise ValueError(f"valid_mask must have shape {(x.shape[0],)}")
    weights = valid.astype(jnp.float32)
    token_sse = jnp.sum((x - output.reconstruction.astype(jnp.float32)) ** 2, axis=1)
    token_sst = jnp.sum((x - mu[None, :]) ** 2, axis=1)
    sse = jnp.sum(token_sse * weights, dtype=jnp.float32)
    sst = jnp.sum(token_sst * weights, dtype=jnp.float32)

    selected_positive = output.selection.values > 0.0
    valid_selected = (
        selected_positive & valid[:, None] & output.selection.valid[:, None]
    )
    fired = (
        jnp.zeros((config.total_features,), dtype=jnp.int32)
        .at[output.selection.latent_ids.reshape(-1)]
        .max(valid_selected.reshape(-1).astype(jnp.int32))
        > 0
    )
    topk_valid = jnp.all(output.selection.valid | ~valid)
    l0_sum = jnp.sum(selected_positive * valid[:, None], dtype=jnp.int32)
    return LossSums(
        sse=sse,
        sst=sst,
        l0_sum=l0_sum,
        valid_tokens=jnp.sum(valid, dtype=jnp.int32),
        topk_valid=topk_valid,
        fired_mask=fired,
    )


def normalized_sse_loss(
    params: SAEParams,
    activation: jax.Array,
    mean: jax.Array,
    config: SAEConfig,
    *,
    expert_ids: jax.Array | None = None,
    valid_mask: jax.Array | None = None,
    grouping_plan: GroupingPlan | None = None,
) -> tuple[jax.Array, LossSums]:
    """Return common normalized SSE and its additive sufficient statistics."""

    if valid_mask is None:
        valid_mask = jnp.ones((activation.shape[0],), dtype=bool)
    sums = _loss_sums(
        params, activation, mean, config, expert_ids, valid_mask, grouping_plan
    )
    loss = sums.sse / jnp.maximum(sums.sst, _EPS)
    # An invalid scientific token is fatal and propagates through compiled
    # training as a non-finite loss; host orchestration also raises explicitly.
    loss = jnp.where(sums.topk_valid, loss, jnp.asarray(jnp.nan, jnp.float32))
    return loss, sums


def loss_and_grad_sums(
    params: SAEParams,
    activation: jax.Array,
    mean: jax.Array,
    config: SAEConfig,
    *,
    expert_ids: jax.Array | None = None,
    valid_mask: jax.Array | None = None,
    grouping_plan: GroupingPlan | None = None,
) -> tuple[LossSums, SAEParams]:
    """Differentiate unnormalized SSE for exact microbatch accumulation."""

    if valid_mask is None:
        valid_mask = jnp.ones((activation.shape[0],), dtype=bool)

    def objective(p: SAEParams) -> tuple[jax.Array, LossSums]:
        sums = _loss_sums(
            p, activation, mean, config, expert_ids, valid_mask, grouping_plan
        )
        return sums.sse, sums

    (_, sums), grads = jax.value_and_grad(objective, has_aux=True)(params)
    return sums, grads


def _project_decoder_gradient(gradient: jax.Array, decoder: jax.Array) -> jax.Array:
    decoder32 = decoder.astype(jnp.float32)
    unit = decoder32 / jnp.maximum(
        jnp.linalg.norm(decoder32, axis=0, keepdims=True), _EPS
    )
    parallel = jnp.sum(gradient * unit, axis=0, keepdims=True)
    return gradient - unit * parallel


def apply_gradients(
    state: TrainState,
    normalized_grads: SAEParams,
    config: SAEConfig,
    *,
    loss: jax.Array,
    valid_tokens: jax.Array,
    fired_mask: jax.Array,
    l0: jax.Array,
    sse: jax.Array,
    sst: jax.Array,
    topk_valid: jax.Array,
) -> tuple[TrainState, StepMetrics]:
    """Clip, project decoder gradient, Adam update, then renormalize columns."""

    grads32 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), normalized_grads)
    grad_sq = sum(
        jnp.sum(x * x, dtype=jnp.float32) for x in jax.tree_util.tree_leaves(grads32)
    )
    grad_norm = jnp.sqrt(grad_sq)
    clip_coefficient = jnp.minimum(
        1.0, jnp.asarray(config.grad_clip, jnp.float32) / (grad_norm + 1e-6)
    )
    clipped = jax.tree_util.tree_map(lambda x: x * clip_coefficient, grads32)
    clipped = SAEParams(
        encoder=clipped.encoder,
        decoder=_project_decoder_gradient(clipped.decoder, state.params.decoder),
        bias=clipped.bias,
    )

    beta1 = jnp.asarray(config.adam_beta1, jnp.float32)
    beta2 = jnp.asarray(config.adam_beta2, jnp.float32)
    opt_step = state.optimizer.step + 1
    new_m = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1.0 - beta1) * g, state.optimizer.m, clipped
    )
    new_v = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), state.optimizer.v, clipped
    )
    correction1 = 1.0 - beta1 ** opt_step.astype(jnp.float32)
    correction2 = 1.0 - beta2 ** opt_step.astype(jnp.float32)
    lr = learning_rate(state.step, config)

    def update(p: jax.Array, m: jax.Array, v: jax.Array) -> jax.Array:
        p32 = p.astype(jnp.float32)
        changed = p32 - lr * (m / correction1) / (
            jnp.sqrt(v / correction2) + config.adam_eps
        )
        return changed.astype(p.dtype)

    params = jax.tree_util.tree_map(update, state.params, new_m, new_v)
    decoder32 = params.decoder.astype(jnp.float32)
    decoder32 = decoder32 / jnp.maximum(
        jnp.linalg.norm(decoder32, axis=0, keepdims=True), _EPS
    )
    params = SAEParams(
        params.encoder, decoder32.astype(params.decoder.dtype), params.bias
    )

    alpha = jnp.asarray(2.0 / 65.0, jnp.float32)  # 64-update EMA
    loss32 = jnp.asarray(loss, jnp.float32)
    loss_ema = jnp.where(
        state.ema_updates == 0,
        loss32,
        (1.0 - alpha) * state.loss_ema + alpha * loss32,
    )
    ema_updates = state.ema_updates + 1
    initial_ema = jnp.where(ema_updates == 64, loss_ema, state.initial_loss_ema)
    new_state = TrainState(
        params=params,
        optimizer=AdamState(new_m, new_v, opt_step),
        step=state.step + 1,
        tokens_seen=state.tokens_seen + valid_tokens.astype(jnp.int32),
        fired_mask=state.fired_mask | jnp.asarray(fired_mask, bool),
        loss_ema=loss_ema,
        initial_loss_ema=initial_ema,
        ema_updates=ema_updates,
    )
    metrics = StepMetrics(
        loss=loss32,
        sse=jnp.asarray(sse, jnp.float32),
        sst=jnp.asarray(sst, jnp.float32),
        l0=jnp.asarray(l0, jnp.float32),
        valid_tokens=valid_tokens.astype(jnp.int32),
        topk_valid=jnp.asarray(topk_valid, bool),
        grad_norm=grad_norm,
        clip_coefficient=clip_coefficient,
        learning_rate=lr,
        live_fraction=jnp.mean(new_state.fired_mask.astype(jnp.float32)),
    )
    return new_state, metrics


def train_step(
    state: TrainState,
    activation: jax.Array,
    mean: jax.Array,
    config: SAEConfig,
    *,
    expert_ids: jax.Array | None = None,
    valid_mask: jax.Array | None = None,
    grouping_plan: GroupingPlan | None = None,
) -> tuple[TrainState, StepMetrics]:
    """Single-microbatch reference update.

    Production orchestration uses ``loss_and_grad_sums`` over four
    microbatches and calls ``apply_gradients`` once with their summed,
    SST-normalized gradient.
    """

    sums, grads = loss_and_grad_sums(
        state.params,
        activation,
        mean,
        config,
        expert_ids=expert_ids,
        valid_mask=valid_mask,
        grouping_plan=grouping_plan,
    )
    normalized_grads = jax.tree_util.tree_map(
        lambda g: g / jnp.maximum(sums.sst, _EPS), grads
    )
    loss = sums.sse / jnp.maximum(sums.sst, _EPS)
    l0 = sums.l0_sum.astype(jnp.float32) / jnp.maximum(
        sums.valid_tokens.astype(jnp.float32), 1.0
    )
    return apply_gradients(
        state,
        normalized_grads,
        config,
        loss=loss,
        valid_tokens=sums.valid_tokens,
        fired_mask=sums.fired_mask,
        l0=l0,
        sse=sums.sse,
        sst=sums.sst,
        topk_valid=sums.topk_valid,
    )


__all__ = [
    "AdamState",
    "CONDITIONS",
    "DEFAULT_BATCH_TOKENS",
    "DEFAULT_FINAL_PADDING",
    "DEFAULT_MICROBATCH_TOKENS",
    "DEFAULT_TOTAL_TOKENS",
    "DEFAULT_UPDATES",
    "ForwardOutput",
    "GroupingPlan",
    "LossSums",
    "SAEConfig",
    "SAEParams",
    "Selection",
    "StepMetrics",
    "TrainState",
    "apply_gradients",
    "candidate_scores",
    "decode_selected_columns",
    "decode_selected_columns_sharded",
    "feature_sharded_topk",
    "forward",
    "init_adam",
    "init_params",
    "init_train_state",
    "learning_rate",
    "local_topk_candidates",
    "loss_and_grad_sums",
    "make_grouping_plan",
    "merge_feature_shard_topk",
    "normalized_sse_loss",
    "positive_topk",
    "train_step",
]
