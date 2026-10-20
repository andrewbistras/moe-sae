"""Single-device JAX oracle for Experiment-2 contribution SAE training.

The production trainer will replace the dense bank gather in this module with
feature-sharded grouped GEMMs.  The scientific computation remains the same:
each of a source token's selected expert outputs is encoded and decoded by one
bank, exact per-contribution TopK is enforced, and errors are weighted by the
square of the native router weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.sae import (
    SAEConfig,
    SAEParams,
    Selection,
    StepMetrics,
    TrainState,
    apply_gradients,
    decode_selected_columns,
    init_train_state,
    positive_topk,
)

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class ContributionOptimizerConfig:
    """Optimizer settings shared by every Experiment-2 condition."""

    parameter_dtype: str = "bfloat16"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    peak_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 250
    total_steps: int = 3_052

    def __post_init__(self) -> None:
        if self.parameter_dtype not in {"bfloat16", "float32"}:
            raise ValueError("parameter_dtype must be bfloat16 or float32")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("Adam betas must be in [0,1)")
        if self.adam_eps <= 0 or self.grad_clip <= 0:
            raise ValueError("Adam epsilon and gradient clip must be positive")
        if not 0 < self.min_lr <= self.peak_lr:
            raise ValueError("learning rates must satisfy 0 < min <= peak")
        if self.warmup_steps < 0 or self.total_steps <= 0:
            raise ValueError("optimizer schedule lengths are invalid")


class ContributionForward(NamedTuple):
    """Eight independently decoded outputs and their flattened selections."""

    reconstruction: jax.Array  # [source_tokens, top_routed, hidden]
    selection: Selection  # flattened [source_tokens * top_routed, k]
    bank_ids: jax.Array  # [source_tokens, top_routed]


class ContributionLossSums(NamedTuple):
    """Additive scientific statistics for one contribution batch."""

    sse: jax.Array
    sst: jax.Array
    l0_sum: jax.Array
    valid_source_tokens: jax.Array
    valid_components: jax.Array
    topk_valid: jax.Array
    fired_mask: jax.Array


def optimizer_sae_config(
    layout: ContributionSAEConfig,
    optimizer: ContributionOptimizerConfig | None = None,
) -> SAEConfig:
    """Translate a contribution layout to the proven generic Adam state.

    The legacy ``SAEConfig.condition`` affects only flattened parameter count
    and is not used for contribution scoring.  F therefore uses eight banks,
    while A/R use all expert banks.  Its ``k`` is total source-token L0 so the
    shared optimizer metrics retain their original meaning.
    """

    selected = optimizer or ContributionOptimizerConfig()
    if layout.condition in {"A", "R"}:
        condition = "A"
        num_experts = layout.bank_count
    elif layout.condition == "B":
        condition = "B"
        num_experts = layout.num_experts
    else:
        condition = "F"
        num_experts = layout.bank_count
    result = SAEConfig(
        condition=condition,
        hidden_dim=layout.hidden_dim,
        num_experts=num_experts,
        top_routed=layout.top_routed,
        bank_width=layout.bank_width,
        k=layout.total_l0,
        parameter_dtype=selected.parameter_dtype,
        adam_beta1=selected.adam_beta1,
        adam_beta2=selected.adam_beta2,
        adam_eps=selected.adam_eps,
        grad_clip=selected.grad_clip,
        peak_lr=selected.peak_lr,
        min_lr=selected.min_lr,
        warmup_steps=selected.warmup_steps,
        total_steps=selected.total_steps,
    )
    if result.total_features != layout.total_features:
        raise ValueError("optimizer parameter layout differs from contribution layout")
    return result


def init_contribution_train_state(
    key: jax.Array,
    layout: ContributionSAEConfig,
    optimizer: ContributionOptimizerConfig | None = None,
) -> TrainState:
    """Initialize the common unit-column SAE state for a contribution layout."""

    return init_train_state(key, optimizer_sae_config(layout, optimizer))


def _validate_forward_shapes(
    params: SAEParams,
    expert_output: jax.Array,
    bank_ids: jax.Array,
    mean: jax.Array,
    layout: ContributionSAEConfig,
) -> None:
    expected = (
        expert_output.shape[0],
        layout.top_routed,
        layout.hidden_dim,
    )
    if expert_output.ndim != 3 or expert_output.shape != expected:
        raise ValueError(
            f"expert_output must have shape [N,{layout.top_routed},{layout.hidden_dim}]"
        )
    if bank_ids.shape != expert_output.shape[:2]:
        raise ValueError("bank_ids must match expert_output token and slot axes")
    if mean.shape != (layout.hidden_dim,):
        raise ValueError(f"mean must have shape [{layout.hidden_dim}]")
    if not jnp.issubdtype(expert_output.dtype, jnp.floating):
        raise TypeError("expert_output must be floating-point")
    if not jnp.issubdtype(bank_ids.dtype, jnp.integer):
        raise TypeError("bank_ids must be integer")
    if not jnp.issubdtype(mean.dtype, jnp.floating):
        raise TypeError("mean must be floating-point")
    expected_parameter_shapes = (
        (layout.total_features, layout.hidden_dim),
        (layout.hidden_dim, layout.total_features),
        (layout.total_features,),
    )
    if (
        params.encoder.shape,
        params.decoder.shape,
        params.bias.shape,
    ) != expected_parameter_shapes:
        raise ValueError("SAE parameter shapes differ from contribution layout")


def contribution_candidate_scores(
    params: SAEParams,
    expert_output: jax.Array,
    bank_ids: jax.Array,
    mean: jax.Array,
    layout: ContributionSAEConfig,
) -> tuple[jax.Array, jax.Array]:
    """Score one bank for every selected expert contribution.

    This deliberately materializes a bank gather and is therefore a correctness
    oracle, not the production A/R hot path.
    """

    target = jnp.asarray(expert_output)
    banks = jnp.asarray(bank_ids)
    pooled_mean = jnp.asarray(mean)
    _validate_forward_shapes(params, target, banks, pooled_mean, layout)
    pooled_mean = pooled_mean.astype(jnp.float32)

    flat_target = target.reshape(-1, layout.hidden_dim)
    flat_banks = banks.astype(jnp.int32).reshape(-1)
    valid_banks = (flat_banks >= 0) & (flat_banks < layout.bank_count)
    safe_banks = jnp.clip(flat_banks, 0, layout.bank_count - 1)
    compute_dtype = params.encoder.dtype
    centered = flat_target.astype(compute_dtype) - pooled_mean.astype(compute_dtype)
    encoder = params.encoder.reshape(
        layout.bank_count,
        layout.bank_width,
        layout.hidden_dim,
    )
    bias = params.bias.reshape(layout.bank_count, layout.bank_width)
    selected_encoder = encoder[safe_banks]
    selected_bias = bias[safe_banks]
    pre = jnp.einsum(
        "md,mwd->mw",
        centered,
        selected_encoder.astype(compute_dtype),
        preferred_element_type=jnp.float32,
    ) + selected_bias.astype(jnp.float32)
    local_ids = jnp.arange(layout.bank_width, dtype=jnp.int32)
    latent_ids = safe_banks[:, None] * layout.bank_width + local_ids[None, :]
    scores = jnp.where(valid_banks[:, None], jax.nn.relu(pre), -jnp.inf)
    return scores, latent_ids


def forward_contributions(
    params: SAEParams,
    expert_output: jax.Array,
    bank_ids: jax.Array,
    mean: jax.Array,
    layout: ContributionSAEConfig,
) -> ContributionForward:
    """Encode and sparsely decode every routed contribution independently."""

    scores, latent_ids = contribution_candidate_scores(
        params,
        expert_output,
        bank_ids,
        mean,
        layout,
    )
    selection = positive_topk(scores, latent_ids, layout.k_per_contribution)
    flat_reconstruction = decode_selected_columns(params.decoder, selection, mean)
    reconstruction = flat_reconstruction.reshape(
        expert_output.shape[0],
        layout.top_routed,
        layout.hidden_dim,
    )
    return ContributionForward(
        reconstruction=reconstruction,
        selection=selection,
        bank_ids=jnp.asarray(bank_ids, jnp.int32),
    )


def _contribution_loss_sums(
    params: SAEParams,
    expert_output: jax.Array,
    bank_ids: jax.Array,
    router_weights: jax.Array,
    mean: jax.Array,
    valid_mask: jax.Array,
    layout: ContributionSAEConfig,
) -> ContributionLossSums:
    output = forward_contributions(
        params,
        expert_output,
        bank_ids,
        mean,
        layout,
    )
    target = jnp.asarray(expert_output, jnp.float32)
    weights = jnp.asarray(router_weights)
    pooled_mean = jnp.asarray(mean, jnp.float32)
    valid = jnp.asarray(valid_mask)
    if weights.shape != target.shape[:2]:
        raise ValueError("router_weights must match expert_output token and slot axes")
    if valid.shape != (target.shape[0],):
        raise ValueError("valid_mask must have shape [source_tokens]")
    if not jnp.issubdtype(weights.dtype, jnp.floating):
        raise TypeError("router_weights must be floating-point")
    if valid.dtype != jnp.bool_:
        raise TypeError("valid_mask must be boolean")
    weights = weights.astype(jnp.float32)

    weighted_error = weights[..., None] * (
        target - output.reconstruction.astype(jnp.float32)
    )
    weighted_centered = weights[..., None] * (target - pooled_mean[None, None, :])
    per_token_sse = jnp.sum(weighted_error**2, axis=(1, 2))
    per_token_sst = jnp.sum(weighted_centered**2, axis=(1, 2))
    valid_weight = valid.astype(jnp.float32)
    sse = jnp.sum(per_token_sse * valid_weight, dtype=jnp.float32)
    sst = jnp.sum(per_token_sst * valid_weight, dtype=jnp.float32)

    selection_valid = output.selection.valid.reshape(
        target.shape[0],
        layout.top_routed,
    )
    selected_positive = (output.selection.values > 0).reshape(
        target.shape[0],
        layout.top_routed,
        layout.k_per_contribution,
    )
    scientific_selection = (
        selected_positive & selection_valid[..., None] & valid[:, None, None]
    )
    router_valid = (
        jnp.all(jnp.isfinite(weights))
        & jnp.all(weights >= 0)
        & jnp.all(jnp.abs(jnp.sum(weights, axis=1, dtype=jnp.float32) - 1.0) <= 0.005)
    )
    topk_valid = jnp.all(selection_valid | ~valid[:, None]) & router_valid
    l0_sum = jnp.sum(scientific_selection, dtype=jnp.int32)

    flat_selected_ids = output.selection.latent_ids.reshape(-1)
    flat_selected_valid = scientific_selection.reshape(-1)
    fired = (
        jnp.zeros((layout.total_features,), jnp.int32)
        .at[flat_selected_ids]
        .max(flat_selected_valid.astype(jnp.int32))
        > 0
    )
    source_tokens = jnp.sum(valid, dtype=jnp.int32)
    return ContributionLossSums(
        sse=sse,
        sst=sst,
        l0_sum=l0_sum,
        valid_source_tokens=source_tokens,
        valid_components=source_tokens * layout.top_routed,
        topk_valid=topk_valid,
        fired_mask=fired,
    )


def normalized_contribution_loss(
    params: SAEParams,
    expert_output: jax.Array,
    bank_ids: jax.Array,
    router_weights: jax.Array,
    mean: jax.Array,
    layout: ContributionSAEConfig,
    *,
    valid_mask: jax.Array | None = None,
) -> tuple[jax.Array, ContributionLossSums]:
    """Return common weighted-contribution NMSE and exact additive sums."""

    if valid_mask is None:
        valid_mask = jnp.ones((expert_output.shape[0],), dtype=bool)
    sums = _contribution_loss_sums(
        params,
        expert_output,
        bank_ids,
        router_weights,
        mean,
        valid_mask,
        layout,
    )
    healthy = sums.topk_valid & (sums.sst > 0)
    loss = jnp.where(
        healthy,
        sums.sse / jnp.maximum(sums.sst, _EPS),
        jnp.asarray(jnp.nan, jnp.float32),
    )
    return loss, sums


def contribution_loss_and_grad_sums(
    params: SAEParams,
    expert_output: jax.Array,
    bank_ids: jax.Array,
    router_weights: jax.Array,
    mean: jax.Array,
    layout: ContributionSAEConfig,
    *,
    valid_mask: jax.Array | None = None,
) -> tuple[ContributionLossSums, SAEParams]:
    """Differentiate unnormalized SSE for exact microbatch accumulation."""

    if valid_mask is None:
        valid_mask = jnp.ones((expert_output.shape[0],), dtype=bool)

    def objective(selected: SAEParams) -> tuple[jax.Array, ContributionLossSums]:
        sums = _contribution_loss_sums(
            selected,
            expert_output,
            bank_ids,
            router_weights,
            mean,
            valid_mask,
            layout,
        )
        return sums.sse, sums

    (_, sums), gradients = jax.value_and_grad(objective, has_aux=True)(params)
    return sums, gradients


def contribution_train_step(
    state: TrainState,
    expert_output: jax.Array,
    bank_ids: jax.Array,
    router_weights: jax.Array,
    mean: jax.Array,
    layout: ContributionSAEConfig,
    optimizer: ContributionOptimizerConfig | None = None,
    *,
    valid_mask: jax.Array | None = None,
) -> tuple[TrainState, StepMetrics]:
    """Apply one reference optimizer update over source-token contributions."""

    optimizer_config = optimizer_sae_config(layout, optimizer)
    sums, gradients = contribution_loss_and_grad_sums(
        state.params,
        expert_output,
        bank_ids,
        router_weights,
        mean,
        layout,
        valid_mask=valid_mask,
    )
    normalized_gradients = jax.tree_util.tree_map(
        lambda gradient: gradient / jnp.maximum(sums.sst, _EPS),
        gradients,
    )
    healthy = sums.topk_valid & (sums.sst > 0)
    loss = jnp.where(
        healthy,
        sums.sse / jnp.maximum(sums.sst, _EPS),
        jnp.asarray(jnp.nan, jnp.float32),
    )
    l0 = sums.l0_sum.astype(jnp.float32) / jnp.maximum(
        sums.valid_source_tokens.astype(jnp.float32),
        1.0,
    )
    return apply_gradients(
        state,
        normalized_gradients,
        optimizer_config,
        loss=loss,
        valid_tokens=sums.valid_source_tokens,
        fired_mask=sums.fired_mask,
        l0=l0,
        sse=sums.sse,
        sst=sums.sst,
        topk_valid=sums.topk_valid,
    )


__all__ = [
    "ContributionForward",
    "ContributionLossSums",
    "ContributionOptimizerConfig",
    "contribution_candidate_scores",
    "contribution_loss_and_grad_sums",
    "contribution_train_step",
    "forward_contributions",
    "init_contribution_train_state",
    "normalized_contribution_loss",
    "optimizer_sae_config",
]
