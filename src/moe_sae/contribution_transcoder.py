"""Pure JAX/NumPy primitives for expert-aligned contribution transcoders.

An Experiment-3 transcoder receives one routed-FFN input per source token and
predicts the eight raw selected-expert outputs independently.  The dictionary
layout and per-contribution sparsity contract are exactly the Experiment-2
``A/R/B/F`` layout, but the encoder and decoder now operate in different
spaces:

* the encoder subtracts one fixed global routed-input mean;
* the decoder adds one distinct fixed global raw-expert-output mean.

The two means are explicit arguments and are never trainable parameters.
Trainable arrays intentionally use a transcoder-specific type even though
their shapes match :class:`moe_sae.sae.SAEParams`.  Named conversion functions
are the only compatibility boundary with the existing initializer and Adam
implementation.

This module contains no capture, checkpoint, storage, or operator code.  Its
dense bank gather is a bounded single-device correctness oracle; a production
trainer may replace that gather without changing the scientific semantics.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from moe_sae.contribution_sae import (
    ContributionSAEConfig,
    ContributionTokenScores,
    condition_bank_ids,
    weighted_contribution_token_scores,
)
from moe_sae.contribution_training import (
    ContributionOptimizerConfig,
    optimizer_sae_config,
)
from moe_sae.sae import (
    AdamState,
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


class ContributionTranscoderParams(NamedTuple):
    """Trainable transcoder arrays.

    ``encoder`` is ``[features, hidden]``, ``decoder`` is
    ``[hidden, features]``, and ``encoder_bias`` is ``[features]``.  There is
    deliberately no trainable output bias: decoding uses the fixed global
    output mean supplied to the forward call.
    """

    encoder: jax.Array
    decoder: jax.Array
    encoder_bias: jax.Array


class ContributionTranscoderAdamState(NamedTuple):
    """Adam moments with transcoder-specific parameter semantics."""

    m: ContributionTranscoderParams
    v: ContributionTranscoderParams
    step: jax.Array


class ContributionTranscoderTrainState(NamedTuple):
    """Training state that cannot be mistaken for an SAE state."""

    params: ContributionTranscoderParams
    optimizer: ContributionTranscoderAdamState
    step: jax.Array
    tokens_seen: jax.Array
    fired_mask: jax.Array
    loss_ema: jax.Array
    initial_loss_ema: jax.Array
    ema_updates: jax.Array


class ContributionTranscoderForward(NamedTuple):
    """Eight independently decoded outputs and flattened sparse selections."""

    reconstruction: jax.Array  # [source_tokens, top_routed, hidden]
    selection: Selection  # flattened [source_tokens * top_routed, k]
    bank_ids: jax.Array  # [source_tokens, top_routed]


class ContributionTranscoderNumpyForward(NamedTuple):
    """Independent host oracle for sparse selection and reconstruction."""

    reconstruction: np.ndarray
    values: np.ndarray
    latent_ids: np.ndarray
    valid: np.ndarray
    bank_ids: np.ndarray


class ContributionTranscoderLossSums(NamedTuple):
    """Additive scientific statistics for one transcoder batch."""

    sse: jax.Array
    sst: jax.Array
    l0_sum: jax.Array
    valid_source_tokens: jax.Array
    valid_components: jax.Array
    topk_valid: jax.Array
    fired_mask: jax.Array


def transcoder_params_from_sae(
    params: SAEParams,
) -> ContributionTranscoderParams:
    """Explicitly reinterpret an SAE-shaped parameter tuple as a transcoder."""

    if not isinstance(params, SAEParams):
        raise TypeError("params must be SAEParams")
    return ContributionTranscoderParams(
        encoder=params.encoder,
        decoder=params.decoder,
        encoder_bias=params.bias,
    )


def transcoder_params_to_sae(
    params: ContributionTranscoderParams,
) -> SAEParams:
    """Explicitly adapt transcoder arrays to the shared optimizer primitive."""

    if not isinstance(params, ContributionTranscoderParams):
        raise TypeError("params must be ContributionTranscoderParams")
    return SAEParams(
        encoder=params.encoder,
        decoder=params.decoder,
        bias=params.encoder_bias,
    )


def transcoder_state_from_sae(
    state: TrainState,
) -> ContributionTranscoderTrainState:
    """Explicitly adapt a freshly initialized or updated shared train state."""

    if not isinstance(state, TrainState):
        raise TypeError("state must be TrainState")
    optimizer = state.optimizer
    return ContributionTranscoderTrainState(
        params=transcoder_params_from_sae(state.params),
        optimizer=ContributionTranscoderAdamState(
            m=transcoder_params_from_sae(optimizer.m),
            v=transcoder_params_from_sae(optimizer.v),
            step=optimizer.step,
        ),
        step=state.step,
        tokens_seen=state.tokens_seen,
        fired_mask=state.fired_mask,
        loss_ema=state.loss_ema,
        initial_loss_ema=state.initial_loss_ema,
        ema_updates=state.ema_updates,
    )


def transcoder_state_to_sae(
    state: ContributionTranscoderTrainState,
) -> TrainState:
    """Explicitly adapt a transcoder state for the shared Adam update."""

    if not isinstance(state, ContributionTranscoderTrainState):
        raise TypeError("state must be ContributionTranscoderTrainState")
    optimizer = state.optimizer
    return TrainState(
        params=transcoder_params_to_sae(state.params),
        optimizer=AdamState(
            m=transcoder_params_to_sae(optimizer.m),
            v=transcoder_params_to_sae(optimizer.v),
            step=optimizer.step,
        ),
        step=state.step,
        tokens_seen=state.tokens_seen,
        fired_mask=state.fired_mask,
        loss_ema=state.loss_ema,
        initial_loss_ema=state.initial_loss_ema,
        ema_updates=state.ema_updates,
    )


def init_contribution_transcoder_train_state(
    key: jax.Array,
    layout: ContributionSAEConfig,
    optimizer: ContributionOptimizerConfig | None = None,
) -> ContributionTranscoderTrainState:
    """Initialize unit decoder columns with the proven shared initializer."""

    sae_state = init_train_state(key, optimizer_sae_config(layout, optimizer))
    return transcoder_state_from_sae(sae_state)


def contribution_transcoder_bank_ids(
    ordered_expert_ids: Any,
    sequence_ids: Any,
    layout: ContributionSAEConfig,
) -> np.ndarray:
    """Return the exact Experiment-2 ``A/R/B/F`` bank assignment.

    Keeping this as a named transcoder entry point makes the architectural
    dependency explicit while delegating all randomization and route-rank
    semantics to the already tested host-side implementation.
    """

    return condition_bank_ids(ordered_expert_ids, sequence_ids, layout)


def contribution_transcoder_token_scores(
    expert_output: Any,
    reconstruction: Any,
    router_weights: Any,
    target_mean: Any,
    *,
    valid_mask: Any,
) -> ContributionTokenScores:
    """Return the existing FP64-accumulated contribution metric.

    ``target_mean`` is the fixed raw-expert-output center.  The routed-input
    mean is intentionally absent from this decoder-side metric.
    """

    return weighted_contribution_token_scores(
        expert_output,
        reconstruction,
        router_weights,
        target_mean,
        valid_mask=valid_mask,
    )


def _validate_parameter_shapes(
    params: ContributionTranscoderParams,
    layout: ContributionSAEConfig,
) -> None:
    if not isinstance(params, ContributionTranscoderParams):
        raise TypeError("params must be ContributionTranscoderParams")
    expected = (
        (layout.total_features, layout.hidden_dim),
        (layout.hidden_dim, layout.total_features),
        (layout.total_features,),
    )
    actual = (
        params.encoder.shape,
        params.decoder.shape,
        params.encoder_bias.shape,
    )
    if actual != expected:
        raise ValueError("transcoder parameter shapes differ from contribution layout")
    for name, value in (
        ("encoder", params.encoder),
        ("decoder", params.decoder),
        ("encoder_bias", params.encoder_bias),
    ):
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError(f"{name} must be floating-point")


def _validate_encoder_inputs(
    params: ContributionTranscoderParams,
    routed_input: jax.Array,
    bank_ids: jax.Array,
    source_mean: jax.Array,
    layout: ContributionSAEConfig,
) -> None:
    _validate_parameter_shapes(params, layout)
    if routed_input.ndim != 2 or routed_input.shape[1] != layout.hidden_dim:
        raise ValueError(f"routed_input must have shape [N,{layout.hidden_dim}]")
    if bank_ids.shape != (routed_input.shape[0], layout.top_routed):
        raise ValueError(
            f"bank_ids must have shape [N,{layout.top_routed}]"
        )
    if source_mean.shape != (layout.hidden_dim,):
        raise ValueError(f"source_mean must have shape [{layout.hidden_dim}]")
    if not jnp.issubdtype(routed_input.dtype, jnp.floating):
        raise TypeError("routed_input must be floating-point")
    if not jnp.issubdtype(bank_ids.dtype, jnp.integer):
        raise TypeError("bank_ids must be integer")
    if not jnp.issubdtype(source_mean.dtype, jnp.floating):
        raise TypeError("source_mean must be floating-point")


def numpy_forward_contribution_transcoder(
    params: ContributionTranscoderParams,
    routed_input: Any,
    bank_ids: Any,
    source_mean: Any,
    target_mean: Any,
    layout: ContributionSAEConfig,
) -> ContributionTranscoderNumpyForward:
    """Independent FP32 NumPy oracle for the bounded transcoder forward pass.

    Selection membership is ordered by descending activation and then global
    latent ID.  Selected values are emitted in ascending latent-ID order,
    matching :func:`moe_sae.sae.positive_topk`.
    """

    if not isinstance(params, ContributionTranscoderParams):
        raise TypeError("params must be ContributionTranscoderParams")
    source = np.asarray(routed_input)
    banks = np.asarray(bank_ids)
    source_center = np.asarray(source_mean)
    target_center = np.asarray(target_mean)
    encoder = np.asarray(params.encoder)
    decoder = np.asarray(params.decoder)
    encoder_bias = np.asarray(params.encoder_bias)
    expected_parameters = (
        (layout.total_features, layout.hidden_dim),
        (layout.hidden_dim, layout.total_features),
        (layout.total_features,),
    )
    if (encoder.shape, decoder.shape, encoder_bias.shape) != expected_parameters:
        raise ValueError("transcoder parameter shapes differ from contribution layout")
    if source.ndim != 2 or source.shape[1] != layout.hidden_dim:
        raise ValueError(f"routed_input must have shape [N,{layout.hidden_dim}]")
    if banks.shape != (source.shape[0], layout.top_routed):
        raise ValueError(f"bank_ids must have shape [N,{layout.top_routed}]")
    if source_center.shape != (layout.hidden_dim,):
        raise ValueError(f"source_mean must have shape [{layout.hidden_dim}]")
    if target_center.shape != (layout.hidden_dim,):
        raise ValueError(f"target_mean must have shape [{layout.hidden_dim}]")
    for name, value in (
        ("encoder", encoder),
        ("decoder", decoder),
        ("encoder_bias", encoder_bias),
        ("routed_input", source),
        ("source_mean", source_center),
        ("target_mean", target_center),
    ):
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError(f"{name} must be floating-point")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if not np.issubdtype(banks.dtype, np.integer):
        raise TypeError("bank_ids must be integer")

    compute_dtype = encoder.dtype
    centered = (
        source.astype(compute_dtype) - source_center.astype(compute_dtype)
    ).astype(np.float32)
    repeated = np.broadcast_to(
        centered[:, None, :],
        (source.shape[0], layout.top_routed, layout.hidden_dim),
    ).reshape(-1, layout.hidden_dim)
    flat_banks = banks.astype(np.int64, copy=False).reshape(-1)
    valid_banks = (flat_banks >= 0) & (flat_banks < layout.bank_count)
    safe_banks = np.clip(flat_banks, 0, layout.bank_count - 1)
    bank_encoder = encoder.astype(np.float32).reshape(
        layout.bank_count,
        layout.bank_width,
        layout.hidden_dim,
    )
    bank_bias = encoder_bias.astype(np.float32).reshape(
        layout.bank_count,
        layout.bank_width,
    )
    pre = np.einsum(
        "md,mwd->mw",
        repeated,
        bank_encoder[safe_banks],
        dtype=np.float32,
    ) + bank_bias[safe_banks]
    scores = np.maximum(pre, np.float32(0))
    scores = np.where(valid_banks[:, None], scores, -np.inf)
    local_ids = np.arange(layout.bank_width, dtype=np.int32)
    candidate_ids = (
        safe_banks.astype(np.int32)[:, None] * layout.bank_width
        + local_ids[None, :]
    )

    component_rows = scores.shape[0]
    k = layout.k_per_contribution
    values = np.zeros((component_rows, k), dtype=np.float32)
    selected_ids = np.repeat(candidate_ids[:, :1], k, axis=1)
    valid = np.zeros((component_rows,), dtype=np.bool_)
    for row in range(component_rows):
        positive = np.flatnonzero(scores[row] > 0)
        if positive.size < k:
            continue
        ranked = np.lexsort(
            (
                candidate_ids[row, positive],
                -scores[row, positive],
            )
        )
        chosen = positive[ranked[:k]]
        chosen = chosen[
            np.argsort(candidate_ids[row, chosen], kind="stable")
        ]
        values[row] = scores[row, chosen]
        selected_ids[row] = candidate_ids[row, chosen]
        valid[row] = True

    columns = decoder.astype(np.float32)[:, selected_ids]
    decode_values = values.astype(decoder.dtype).astype(np.float32)
    centered_reconstruction = np.einsum(
        "mk,hmk->mh",
        decode_values,
        columns,
        dtype=np.float32,
    )
    reconstruction = (
        centered_reconstruction + target_center.astype(np.float32)[None, :]
    ).reshape(source.shape[0], layout.top_routed, layout.hidden_dim)
    return ContributionTranscoderNumpyForward(
        reconstruction=reconstruction,
        values=values,
        latent_ids=selected_ids.astype(np.int32, copy=False),
        valid=valid,
        bank_ids=banks.astype(np.int32, copy=True),
    )


def contribution_transcoder_candidate_scores(
    params: ContributionTranscoderParams,
    routed_input: jax.Array,
    bank_ids: jax.Array,
    source_mean: jax.Array,
    layout: ContributionSAEConfig,
) -> tuple[jax.Array, jax.Array]:
    """Score one assigned bank per selected expert from one shared FFN input.

    The shared input is repeated logically across router slots.  This dense
    gather is a correctness oracle and materializes
    ``N * top_routed * bank_width`` scores, never the full latent dictionary
    for A or R.
    """

    source = jnp.asarray(routed_input)
    banks = jnp.asarray(bank_ids)
    source_mean = jnp.asarray(source_mean)
    _validate_encoder_inputs(params, source, banks, source_mean, layout)

    compute_dtype = params.encoder.dtype
    centered = source.astype(compute_dtype) - source_mean.astype(compute_dtype)
    repeated = jnp.broadcast_to(
        centered[:, None, :],
        (source.shape[0], layout.top_routed, layout.hidden_dim),
    ).reshape(-1, layout.hidden_dim)

    flat_banks = banks.astype(jnp.int32).reshape(-1)
    valid_banks = (flat_banks >= 0) & (flat_banks < layout.bank_count)
    safe_banks = jnp.clip(flat_banks, 0, layout.bank_count - 1)
    encoder = params.encoder.reshape(
        layout.bank_count,
        layout.bank_width,
        layout.hidden_dim,
    )
    encoder_bias = params.encoder_bias.reshape(
        layout.bank_count,
        layout.bank_width,
    )
    selected_encoder = encoder[safe_banks]
    selected_bias = encoder_bias[safe_banks]
    pre = jnp.einsum(
        "md,mwd->mw",
        repeated,
        selected_encoder.astype(compute_dtype),
        preferred_element_type=jnp.float32,
    ) + selected_bias.astype(jnp.float32)

    local_ids = jnp.arange(layout.bank_width, dtype=jnp.int32)
    latent_ids = safe_banks[:, None] * layout.bank_width + local_ids[None, :]
    scores = jnp.where(valid_banks[:, None], jax.nn.relu(pre), -jnp.inf)
    return scores, latent_ids


def forward_contribution_transcoder(
    params: ContributionTranscoderParams,
    routed_input: jax.Array,
    bank_ids: jax.Array,
    source_mean: jax.Array,
    target_mean: jax.Array,
    layout: ContributionSAEConfig,
) -> ContributionTranscoderForward:
    """Map ``[N,H]`` routed inputs to selected raw outputs ``[N,8,H]``."""

    target_mean = jnp.asarray(target_mean)
    if target_mean.shape != (layout.hidden_dim,):
        raise ValueError(f"target_mean must have shape [{layout.hidden_dim}]")
    if not jnp.issubdtype(target_mean.dtype, jnp.floating):
        raise TypeError("target_mean must be floating-point")

    scores, latent_ids = contribution_transcoder_candidate_scores(
        params,
        routed_input,
        bank_ids,
        source_mean,
        layout,
    )
    selection = positive_topk(
        scores,
        latent_ids,
        layout.k_per_contribution,
    )
    flat_reconstruction = decode_selected_columns(
        params.decoder,
        selection,
        target_mean,
    )
    reconstruction = flat_reconstruction.reshape(
        jnp.asarray(routed_input).shape[0],
        layout.top_routed,
        layout.hidden_dim,
    )
    return ContributionTranscoderForward(
        reconstruction=reconstruction,
        selection=selection,
        bank_ids=jnp.asarray(bank_ids, jnp.int32),
    )


def _contribution_transcoder_loss_sums(
    params: ContributionTranscoderParams,
    routed_input: jax.Array,
    expert_output: jax.Array,
    bank_ids: jax.Array,
    router_weights: jax.Array,
    source_mean: jax.Array,
    target_mean: jax.Array,
    valid_mask: jax.Array,
    layout: ContributionSAEConfig,
) -> ContributionTranscoderLossSums:
    output = forward_contribution_transcoder(
        params,
        routed_input,
        bank_ids,
        source_mean,
        target_mean,
        layout,
    )
    source = jnp.asarray(routed_input)
    target = jnp.asarray(expert_output)
    weights = jnp.asarray(router_weights)
    source_mean = jnp.asarray(source_mean)
    target_mean = jnp.asarray(target_mean)
    valid = jnp.asarray(valid_mask)
    expected_target_shape = (
        source.shape[0],
        layout.top_routed,
        layout.hidden_dim,
    )
    if target.shape != expected_target_shape:
        raise ValueError(
            "expert_output must have shape "
            f"[N,{layout.top_routed},{layout.hidden_dim}]"
        )
    if weights.shape != target.shape[:2]:
        raise ValueError("router_weights must match expert_output token and slot axes")
    if valid.shape != (source.shape[0],):
        raise ValueError("valid_mask must have shape [source_tokens]")
    if not jnp.issubdtype(target.dtype, jnp.floating):
        raise TypeError("expert_output must be floating-point")
    if not jnp.issubdtype(weights.dtype, jnp.floating):
        raise TypeError("router_weights must be floating-point")
    if valid.dtype != jnp.bool_:
        raise TypeError("valid_mask must be boolean")

    target32 = target.astype(jnp.float32)
    weights32 = weights.astype(jnp.float32)
    target_mean32 = target_mean.astype(jnp.float32)
    weighted_error = weights32[..., None] * (
        target32 - output.reconstruction.astype(jnp.float32)
    )
    weighted_centered = weights32[..., None] * (
        target32 - target_mean32[None, None, :]
    )
    per_token_sse = jnp.sum(weighted_error**2, axis=(1, 2))
    per_token_sst = jnp.sum(weighted_centered**2, axis=(1, 2))
    valid_weight = valid.astype(jnp.float32)
    sse = jnp.sum(per_token_sse * valid_weight, dtype=jnp.float32)
    sst = jnp.sum(per_token_sst * valid_weight, dtype=jnp.float32)

    selection_valid = output.selection.valid.reshape(
        source.shape[0],
        layout.top_routed,
    )
    selected_positive = (output.selection.values > 0).reshape(
        source.shape[0],
        layout.top_routed,
        layout.k_per_contribution,
    )
    scientific_selection = (
        selected_positive & selection_valid[..., None] & valid[:, None, None]
    )
    per_contribution_l0 = jnp.sum(
        scientific_selection,
        axis=2,
        dtype=jnp.int32,
    )
    exact_l0 = jnp.all(
        jnp.where(
            valid[:, None],
            per_contribution_l0 == layout.k_per_contribution,
            per_contribution_l0 == 0,
        )
    )
    scientific_components = valid[:, None]
    bank_values = jnp.asarray(bank_ids)
    bank_range = (bank_values >= 0) & (bank_values < layout.bank_count)
    bank_valid = jnp.all(bank_range | ~scientific_components)
    if layout.condition in {"A", "R"}:
        ordered_banks = jnp.sort(bank_values, axis=1)
        bank_valid = bank_valid & jnp.all(
            (jnp.diff(ordered_banks, axis=1) != 0) | ~valid[:, None]
        )
    else:
        ranks = jnp.arange(
            layout.top_routed,
            dtype=bank_values.dtype,
        )[None, :]
        bank_valid = bank_valid & jnp.all(
            (bank_values == ranks) | ~scientific_components
        )
    router_valid = (
        jnp.all(jnp.isfinite(weights32) | ~scientific_components)
        & jnp.all((weights32 >= 0) | ~scientific_components)
        & jnp.all(
            (
                jnp.abs(
                    jnp.sum(weights32, axis=1, dtype=jnp.float32) - 1.0
                )
                <= 0.005
            )
            | ~valid
        )
    )
    finite_inputs = (
        jnp.all(jnp.isfinite(source) | ~valid[:, None])
        & jnp.all(jnp.isfinite(target) | ~valid[:, None, None])
        & jnp.all(jnp.isfinite(source_mean))
        & jnp.all(jnp.isfinite(target_mean))
    )
    topk_valid = (
        jnp.all(selection_valid | ~valid[:, None])
        & exact_l0
        & bank_valid
        & router_valid
        & finite_inputs
    )
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
    return ContributionTranscoderLossSums(
        sse=sse,
        sst=sst,
        l0_sum=l0_sum,
        valid_source_tokens=source_tokens,
        valid_components=source_tokens * layout.top_routed,
        topk_valid=topk_valid,
        fired_mask=fired,
    )


def normalized_contribution_transcoder_loss(
    params: ContributionTranscoderParams,
    routed_input: jax.Array,
    expert_output: jax.Array,
    bank_ids: jax.Array,
    router_weights: jax.Array,
    source_mean: jax.Array,
    target_mean: jax.Array,
    layout: ContributionSAEConfig,
    *,
    valid_mask: jax.Array,
) -> tuple[jax.Array, ContributionTranscoderLossSums]:
    """Return common weighted-contribution NMSE and exact additive sums."""

    sums = _contribution_transcoder_loss_sums(
        params,
        routed_input,
        expert_output,
        bank_ids,
        router_weights,
        source_mean,
        target_mean,
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


def contribution_transcoder_loss_and_grad_sums(
    params: ContributionTranscoderParams,
    routed_input: jax.Array,
    expert_output: jax.Array,
    bank_ids: jax.Array,
    router_weights: jax.Array,
    source_mean: jax.Array,
    target_mean: jax.Array,
    layout: ContributionSAEConfig,
    *,
    valid_mask: jax.Array,
) -> tuple[ContributionTranscoderLossSums, ContributionTranscoderParams]:
    """Differentiate unnormalized SSE for exact microbatch accumulation."""

    def objective(
        selected: ContributionTranscoderParams,
    ) -> tuple[jax.Array, ContributionTranscoderLossSums]:
        sums = _contribution_transcoder_loss_sums(
            selected,
            routed_input,
            expert_output,
            bank_ids,
            router_weights,
            source_mean,
            target_mean,
            valid_mask,
            layout,
        )
        return sums.sse, sums

    (_, sums), gradients = jax.value_and_grad(objective, has_aux=True)(params)
    return sums, gradients


def contribution_transcoder_train_step(
    state: ContributionTranscoderTrainState,
    routed_input: jax.Array,
    expert_output: jax.Array,
    bank_ids: jax.Array,
    router_weights: jax.Array,
    source_mean: jax.Array,
    target_mean: jax.Array,
    layout: ContributionSAEConfig,
    optimizer: ContributionOptimizerConfig | None = None,
    *,
    valid_mask: jax.Array,
) -> tuple[ContributionTranscoderTrainState, StepMetrics]:
    """Apply one shared-Adam update through explicit transcoder adapters."""

    if not isinstance(state, ContributionTranscoderTrainState):
        raise TypeError("state must be ContributionTranscoderTrainState")
    optimizer_config = optimizer_sae_config(layout, optimizer)
    sums, gradients = contribution_transcoder_loss_and_grad_sums(
        state.params,
        routed_input,
        expert_output,
        bank_ids,
        router_weights,
        source_mean,
        target_mean,
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
    updated, metrics = apply_gradients(
        transcoder_state_to_sae(state),
        transcoder_params_to_sae(normalized_gradients),
        optimizer_config,
        loss=loss,
        valid_tokens=sums.valid_source_tokens,
        fired_mask=sums.fired_mask,
        l0=l0,
        sse=sums.sse,
        sst=sums.sst,
        topk_valid=sums.topk_valid,
    )
    return transcoder_state_from_sae(updated), metrics


__all__ = [
    "ContributionTranscoderAdamState",
    "ContributionTranscoderForward",
    "ContributionTranscoderLossSums",
    "ContributionTranscoderNumpyForward",
    "ContributionTranscoderParams",
    "ContributionTranscoderTrainState",
    "contribution_transcoder_bank_ids",
    "contribution_transcoder_candidate_scores",
    "contribution_transcoder_loss_and_grad_sums",
    "contribution_transcoder_token_scores",
    "contribution_transcoder_train_step",
    "forward_contribution_transcoder",
    "init_contribution_transcoder_train_state",
    "normalized_contribution_transcoder_loss",
    "numpy_forward_contribution_transcoder",
    "transcoder_params_from_sae",
    "transcoder_params_to_sae",
    "transcoder_state_from_sae",
    "transcoder_state_to_sae",
]
