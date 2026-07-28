"""Scientific core for Experiment 5 routed/shared sparse models.

Experiment 5 asks every condition to reconstruct the same exact Gemma full-FFN
target.  ``H`` and ``R`` mirror Gemma's computation graph: eight routed
dictionaries reconstruct raw expert outputs, one global shared dictionary
reconstructs the already-normalized shared branch, and the estimates are
combined through Gemma's frozen routed and outer RMSNorms.  ``B`` and ``F``
instead reconstruct the full-FFN target with one global dictionary.

The production layout is frozen:

* H/R routed: ``128 x 512`` features, eight visible banks, TopK-4 per route;
* H/R shared: ``4,096`` features, TopK-32;
* B: ``8,192`` global features, TopK-64;
* F: ``69,632`` global features, TopK-64.

All conditions use one flattened :class:`~moe_sae.sae.SAEParams` tree.  For
H/R, routed features occupy the prefix and shared features occupy the suffix.
That representation makes H and F exactly parameter-matched and lets the
existing initializer, Adam implementation, sharding, and checkpoint format be
reused without an adapter.

This module is deliberately independent of storage and orchestration.  It
contains a bounded dense JAX correctness graph, an independent NumPy oracle,
host batch validation/stacking, loss/gradient primitives, and a single-device
train step.  Production grouped/feature-sharded kernels may replace candidate
scoring without changing these scientific semantics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from moe_sae.contribution_artifacts import (
    RMS_NORM_EPSILON,
    ROUTER_WEIGHT_SUM_ATOL,
    bf16_bits_to_float32,
    float32_to_bf16_bits,
)
from moe_sae.contribution_sae import (
    ContributionSAEConfig,
    condition_bank_ids,
)
from moe_sae.contribution_training import ContributionOptimizerConfig
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

Experiment5Condition = Literal["H", "R", "B", "F"]
Experiment5EncoderKind = Literal["sae", "transcoder"]

EXPERIMENT5_CONDITIONS = ("H", "R", "B", "F")
EXPERIMENT5_ENCODER_KINDS = ("sae", "transcoder")
DEFAULT_HIDDEN_DIM = 2_816
DEFAULT_NUM_EXPERTS = 128
DEFAULT_TOP_ROUTED = 8
DEFAULT_ROUTED_BANK_WIDTH = 512
DEFAULT_ROUTED_K = 4
DEFAULT_SHARED_WIDTH = 4_096
DEFAULT_SHARED_K = 32
_EPS = 1e-12
_UINT64_MAX = 2**64 - 1


class Experiment5Error(ValueError):
    """An Experiment-5 scientific input violates the frozen contract."""


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise Experiment5Error(f"{name} must be an integer")
    selected = int(value)
    if selected <= 0 or selected > 2**31 - 1:
        raise Experiment5Error(f"{name} must be in [1, {2**31 - 1}]")
    return selected


@dataclass(frozen=True, slots=True)
class Experiment5Layout:
    """Static H/R/B/F layout with candidate, parameter, and L0 matches."""

    condition: Experiment5Condition
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_experts: int = DEFAULT_NUM_EXPERTS
    top_routed: int = DEFAULT_TOP_ROUTED
    routed_bank_width: int = DEFAULT_ROUTED_BANK_WIDTH
    routed_k: int = DEFAULT_ROUTED_K
    shared_width: int = DEFAULT_SHARED_WIDTH
    shared_k: int = DEFAULT_SHARED_K
    randomization_key: int = 0

    def __post_init__(self) -> None:
        if self.condition not in EXPERIMENT5_CONDITIONS:
            raise Experiment5Error(
                f"condition must be one of {EXPERIMENT5_CONDITIONS}"
            )
        for name in (
            "hidden_dim",
            "num_experts",
            "top_routed",
            "routed_bank_width",
            "routed_k",
            "shared_width",
            "shared_k",
        ):
            _positive_integer(name, getattr(self, name))
        if self.top_routed > self.num_experts:
            raise Experiment5Error("top_routed cannot exceed num_experts")
        if self.routed_k > self.routed_bank_width:
            raise Experiment5Error("routed_k cannot exceed routed_bank_width")
        if self.shared_k > self.shared_width:
            raise Experiment5Error("shared_k cannot exceed shared_width")
        if (
            isinstance(self.randomization_key, (bool, np.bool_))
            or not isinstance(self.randomization_key, (int, np.integer))
            or not 0 <= int(self.randomization_key) <= _UINT64_MAX
        ):
            raise Experiment5Error("randomization_key must be uint64-compatible")

    @property
    def is_hybrid(self) -> bool:
        return self.condition in {"H", "R"}

    @property
    def routed_features(self) -> int:
        return self.num_experts * self.routed_bank_width

    @property
    def hybrid_total_features(self) -> int:
        return self.routed_features + self.shared_width

    @property
    def hybrid_candidates_per_token(self) -> int:
        return self.top_routed * self.routed_bank_width + self.shared_width

    @property
    def total_features(self) -> int:
        if self.condition == "B":
            return self.hybrid_candidates_per_token
        return self.hybrid_total_features

    @property
    def candidates_per_token(self) -> int:
        if self.condition == "F":
            return self.hybrid_total_features
        return self.hybrid_candidates_per_token

    @property
    def total_l0(self) -> int:
        return self.top_routed * self.routed_k + self.shared_k

    @property
    def parameter_count(self) -> int:
        return 2 * self.hidden_dim * self.total_features + self.total_features

    @property
    def candidate_exposure(self) -> float:
        return self.candidates_per_token / self.total_features


def production_experiment5_layout(
    condition: Experiment5Condition,
    *,
    randomization_key: int,
) -> Experiment5Layout:
    """Return the frozen production Experiment-5 layout."""

    return Experiment5Layout(
        condition=condition,
        randomization_key=randomization_key,
    )


class Experiment5Centering(NamedTuple):
    """Fixed source/target means with roles that may not be interchanged."""

    source_mean: Any
    routed_target_mean: Any
    shared_target_mean: Any
    full_target_mean: Any


class Experiment5Recomposition(NamedTuple):
    """Frozen Gemma parameters needed by the differentiable full-FFN graph."""

    routed_scale: Any
    outer_scale: Any
    epsilon: float = RMS_NORM_EPSILON


class Experiment5Batch(NamedTuple):
    """One aligned source/target batch shared by SAE and transcoder panels.

    ``routed_input`` is ignored by the SAE panel but remains present so the
    same paired-prefetch object and static executable signature can serve both
    panels.  ``shared_normalized`` is derived from the existing raw shared
    capture and recomposition sidecar; it is not a new captured tensor.
    """

    routed_input: Any  # [tokens, hidden]
    expert_output: Any  # [tokens, routes, hidden]
    shared_normalized: Any  # [tokens, hidden]
    bank_ids: Any  # [tokens, routes]
    router_weights: Any  # [tokens, routes]
    valid_mask: Any  # [tokens]


class Experiment5Forward(NamedTuple):
    """One condition's full reconstruction and exact sparse selections."""

    full_reconstruction: jax.Array
    routed_reconstruction: jax.Array | None
    shared_reconstruction: jax.Array | None
    routed_selection: Selection | None
    shared_selection: Selection | None
    global_selection: Selection | None


class Experiment5NumpySelection(NamedTuple):
    values: np.ndarray
    latent_ids: np.ndarray
    valid: np.ndarray


class Experiment5NumpyForward(NamedTuple):
    full_reconstruction: np.ndarray
    routed_reconstruction: np.ndarray | None
    shared_reconstruction: np.ndarray | None
    routed_selection: Experiment5NumpySelection | None
    shared_selection: Experiment5NumpySelection | None
    global_selection: Experiment5NumpySelection | None


class Experiment5LossSums(NamedTuple):
    """Additive statistics for the common normalized full-FFN objective."""

    sse: jax.Array
    sst: jax.Array
    l0_sum: jax.Array
    valid_tokens: jax.Array
    topk_valid: jax.Array
    fired_mask: jax.Array


@dataclass(frozen=True, slots=True)
class Experiment5TokenScores:
    """FP64 per-token evaluation rows for paired sequence inference."""

    token_sse: np.ndarray
    token_sst: np.ndarray
    valid_mask: np.ndarray

    @property
    def tokens(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def sse(self) -> float:
        return float(np.sum(self.token_sse, dtype=np.float64))

    @property
    def sst(self) -> float:
        return float(np.sum(self.token_sst, dtype=np.float64))

    @property
    def nmse(self) -> float:
        if not self.sst > 0:
            raise Experiment5Error("full-FFN SST must be positive")
        return self.sse / self.sst

    @property
    def ev(self) -> float:
        return 1.0 - self.nmse


def experiment5_bank_ids(
    ordered_expert_ids: Any,
    sequence_ids: Any,
    layout: Experiment5Layout,
) -> np.ndarray:
    """Map expert IDs to H/R banks with the frozen bijection primitive.

    Production callers provide an Experiment-5 protocol/seed-separated key;
    the primitive then mixes in each packed sequence ID.
    """

    if layout.condition not in {"H", "R"}:
        raise Experiment5Error("only H/R use routed bank identities")
    contribution_layout = ContributionSAEConfig(
        condition="A" if layout.condition == "H" else "R",
        hidden_dim=layout.hidden_dim,
        num_experts=layout.num_experts,
        top_routed=layout.top_routed,
        bank_width=layout.routed_bank_width,
        k_per_contribution=layout.routed_k,
        randomization_key=layout.randomization_key,
    )
    return condition_bank_ids(
        ordered_expert_ids,
        sequence_ids,
        contribution_layout,
    )


def experiment5_optimizer_sae_config(
    layout: Experiment5Layout,
    optimizer: ContributionOptimizerConfig | None = None,
) -> SAEConfig:
    """Represent every condition as one global-width optimizer parameter tree."""

    selected = optimizer or ContributionOptimizerConfig()
    return SAEConfig(
        condition="B",
        hidden_dim=layout.hidden_dim,
        num_experts=1,
        top_routed=1,
        bank_width=layout.total_features,
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


def init_experiment5_train_state(
    key: jax.Array,
    layout: Experiment5Layout,
    optimizer: ContributionOptimizerConfig | None = None,
) -> TrainState:
    """Initialize unit decoder columns, tied-transpose encoder, and Adam."""

    return init_train_state(key, experiment5_optimizer_sae_config(layout, optimizer))


def _batch_field(batch: Any, name: str, default: Any = None) -> Any:
    if isinstance(batch, Mapping):
        return batch.get(name, default)
    return getattr(batch, name, default)


def validate_experiment5_microbatch(
    batch: Any,
    layout: Experiment5Layout,
    *,
    expected_rows: int | None = None,
) -> Experiment5Batch:
    """Validate one host microbatch without changing its scientific rows."""

    source = np.asarray(_batch_field(batch, "routed_input"))
    expert = np.asarray(_batch_field(batch, "expert_output"))
    shared = np.asarray(_batch_field(batch, "shared_normalized"))
    banks = np.asarray(_batch_field(batch, "bank_ids"))
    weights = np.asarray(_batch_field(batch, "router_weights"))
    valid = np.asarray(_batch_field(batch, "valid_mask"))
    if source.ndim != 2 or source.shape[1:] != (layout.hidden_dim,):
        raise Experiment5Error(
            f"routed_input must have shape [tokens,{layout.hidden_dim}]"
        )
    rows = source.shape[0]
    if expected_rows is not None and rows != expected_rows:
        raise Experiment5Error("all microbatches must have equal static rows")
    if expert.shape != (rows, layout.top_routed, layout.hidden_dim):
        raise Experiment5Error(
            "expert_output must have shape "
            f"[tokens,{layout.top_routed},{layout.hidden_dim}]"
        )
    if shared.shape != (rows, layout.hidden_dim):
        raise Experiment5Error(
            f"shared_normalized must have shape [tokens,{layout.hidden_dim}]"
        )
    if banks.shape != (rows, layout.top_routed):
        raise Experiment5Error("bank_ids must match token and route axes")
    if weights.shape != banks.shape:
        raise Experiment5Error("router_weights must match bank_ids")
    if valid.shape != (rows,) or valid.dtype != np.bool_:
        raise Experiment5Error("valid_mask must be boolean [tokens]")
    for name, value in (
        ("routed_input", source),
        ("expert_output", expert),
        ("shared_normalized", shared),
        ("router_weights", weights),
    ):
        if not np.issubdtype(value.dtype, np.floating):
            raise Experiment5Error(f"{name} must be floating-point")
        if not np.isfinite(value).all():
            raise Experiment5Error(f"{name} must be finite")
    if not np.issubdtype(banks.dtype, np.integer):
        raise Experiment5Error("bank_ids must be integer")
    if np.any(weights < 0):
        raise Experiment5Error("router_weights must be nonnegative")
    scientific_weights = weights[valid].astype(np.float64, copy=False)
    if scientific_weights.size and not np.all(
        np.abs(np.sum(scientific_weights, axis=1) - 1.0)
        <= ROUTER_WEIGHT_SUM_ATOL
    ):
        raise Experiment5Error("scientific router weights must sum to one")
    if layout.is_hybrid:
        scientific_banks = banks[valid].astype(np.int64, copy=False)
        if scientific_banks.size and (
            np.any(scientific_banks < 0)
            or np.any(scientific_banks >= layout.num_experts)
        ):
            raise Experiment5Error("hybrid bank ID is out of range")
        if scientific_banks.shape[1] > 1 and np.any(
            np.diff(np.sort(scientific_banks, axis=1), axis=1) == 0
        ):
            raise Experiment5Error("H/R bank IDs must be unique per token")
    return Experiment5Batch(
        source,
        expert,
        shared,
        banks.astype(np.int32, copy=False),
        weights,
        valid,
    )


def stack_experiment5_microbatches(
    microbatches: Sequence[Any],
    layout: Experiment5Layout,
) -> Experiment5Batch:
    """Validate and stack host microbatches for a static compiled update."""

    if not microbatches:
        raise Experiment5Error("at least one microbatch is required")
    checked: list[Experiment5Batch] = []
    rows: int | None = None
    for item in microbatches:
        batch = validate_experiment5_microbatch(
            item,
            layout,
            expected_rows=rows,
        )
        rows = batch.routed_input.shape[0]
        checked.append(batch)
    return Experiment5Batch(
        *[
            np.stack([np.asarray(getattr(item, field)) for item in checked])
            for field in Experiment5Batch._fields
        ]
    )


def _validate_encoder_kind(kind: str) -> Experiment5EncoderKind:
    if kind not in EXPERIMENT5_ENCODER_KINDS:
        raise Experiment5Error(
            f"encoder_kind must be one of {EXPERIMENT5_ENCODER_KINDS}"
        )
    return kind  # type: ignore[return-value]


def _jax_activation(value: jax.Array, kind: Experiment5EncoderKind) -> jax.Array:
    value32 = jnp.asarray(value, jnp.float32)
    if kind == "sae":
        return jax.nn.relu(value32)
    return jnp.maximum(
        jax.nn.softplus(value32),
        jnp.finfo(jnp.float32).tiny,
    )


def _numpy_activation(
    value: np.ndarray,
    kind: Experiment5EncoderKind,
) -> np.ndarray:
    value32 = np.asarray(value, np.float32)
    if kind == "sae":
        return np.maximum(value32, np.float32(0.0))
    return np.maximum(
        np.logaddexp(np.float32(0.0), value32).astype(np.float32, copy=False),
        np.finfo(np.float32).tiny,
    )


def _validate_params(params: SAEParams, layout: Experiment5Layout) -> None:
    if not isinstance(params, SAEParams):
        raise TypeError("params must be SAEParams")
    expected = (
        (layout.total_features, layout.hidden_dim),
        (layout.hidden_dim, layout.total_features),
        (layout.total_features,),
    )
    actual = (params.encoder.shape, params.decoder.shape, params.bias.shape)
    if actual != expected:
        raise Experiment5Error(
            f"parameter shapes {actual} differ from Experiment-5 layout {expected}"
        )
    for name, value in zip(("encoder", "decoder", "bias"), params, strict=True):
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError(f"{name} must be floating-point")


def _validate_centering_and_recomposition(
    centering: Experiment5Centering,
    recomposition: Experiment5Recomposition,
    layout: Experiment5Layout,
) -> None:
    for name, value in zip(
        Experiment5Centering._fields,
        centering,
        strict=True,
    ):
        array = jnp.asarray(value)
        if array.shape != (layout.hidden_dim,):
            raise Experiment5Error(
                f"{name} must have shape [{layout.hidden_dim}]"
            )
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError(f"{name} must be floating-point")
    for name in ("routed_scale", "outer_scale"):
        array = jnp.asarray(getattr(recomposition, name))
        if array.shape != (layout.hidden_dim,):
            raise Experiment5Error(
                f"{name} must have shape [{layout.hidden_dim}]"
            )
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError(f"{name} must be floating-point")
    epsilon = recomposition.epsilon
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0
    ):
        raise Experiment5Error("RMSNorm epsilon must be finite and positive")


def _jax_rmsnorm_bf16(
    value: jax.Array,
    scale: jax.Array,
    *,
    epsilon: float,
) -> jax.Array:
    """Pinned MaxText RMSNorm with both BF16 round points."""

    input_bf16 = jnp.asarray(value, jnp.bfloat16)
    input32 = input_bf16.astype(jnp.float32)
    mean_square = jnp.mean(
        jax.lax.square(input32),
        axis=-1,
        keepdims=True,
        dtype=jnp.float32,
    )
    unit = jnp.asarray(
        input32 * jax.lax.rsqrt(mean_square + jnp.float32(epsilon)),
        jnp.bfloat16,
    )
    return jnp.asarray(
        unit * jnp.asarray(scale, jnp.bfloat16),
        jnp.bfloat16,
    )


def jax_recompose_experiment5_full_ffn(
    expert_output: jax.Array,
    shared_normalized: jax.Array,
    router_weights: jax.Array,
    recomposition: Experiment5Recomposition,
) -> jax.Array:
    """Apply Gemma's routed sum, routed RMSNorm, addition, and outer RMSNorm."""

    expert = jnp.asarray(expert_output)
    shared = jnp.asarray(shared_normalized)
    weights = jnp.asarray(router_weights)
    if expert.ndim != 3:
        raise Experiment5Error("expert_output must be rank three")
    if shared.shape != (expert.shape[0], expert.shape[2]):
        raise Experiment5Error("shared_normalized must match expert token/hidden axes")
    if weights.shape != expert.shape[:2]:
        raise Experiment5Error("router_weights must match expert token/route axes")
    routed_scale = jnp.asarray(recomposition.routed_scale)
    outer_scale = jnp.asarray(recomposition.outer_scale)
    if routed_scale.shape != (expert.shape[2],) or outer_scale.shape != (
        expert.shape[2],
    ):
        raise Experiment5Error("recomposition scales must match hidden dimension")
    routed = jax_experiment5_routed_branch(
        expert,
        weights,
        recomposition,
    )
    branch_sum = jnp.asarray(
        routed + shared.astype(jnp.bfloat16),
        jnp.bfloat16,
    )
    return _jax_rmsnorm_bf16(
        branch_sum,
        outer_scale,
        epsilon=float(recomposition.epsilon),
    )


def jax_experiment5_routed_branch(
    expert_output: jax.Array,
    router_weights: jax.Array,
    recomposition: Experiment5Recomposition,
) -> jax.Array:
    """Return Gemma's exact normalized routed branch before shared addition.

    Evaluation needs this intermediate for the routed-only diagnostic.  It is
    deliberately exposed from the same implementation that defines the
    full-FFN target so the diagnostic cannot silently drift from the training
    objective's BF16 round points.
    """

    expert = jnp.asarray(expert_output)
    weights = jnp.asarray(router_weights)
    if expert.ndim != 3:
        raise Experiment5Error("expert_output must be rank three")
    if weights.shape != expert.shape[:2]:
        raise Experiment5Error("router_weights must match expert token/route axes")
    routed_scale = jnp.asarray(recomposition.routed_scale)
    if routed_scale.shape != (expert.shape[2],):
        raise Experiment5Error("routed scale must match hidden dimension")
    raw_routed = jnp.einsum(
        "nkh,nk->nh",
        expert.astype(jnp.bfloat16).astype(jnp.float32),
        weights.astype(jnp.bfloat16).astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )
    return _jax_rmsnorm_bf16(
        raw_routed,
        routed_scale,
        epsilon=float(recomposition.epsilon),
    )


def _numpy_bf16(value: Any) -> np.ndarray:
    return bf16_bits_to_float32(
        float32_to_bf16_bits(np.asarray(value, dtype=np.float32))
    )


def _numpy_rmsnorm_bf16(
    value: Any,
    scale: Any,
    *,
    epsilon: float,
) -> np.ndarray:
    input_bf16 = _numpy_bf16(value)
    scale_bf16 = _numpy_bf16(scale)
    mean_square = np.mean(
        np.square(input_bf16, dtype=np.float32),
        axis=-1,
        keepdims=True,
        dtype=np.float32,
    )
    inverse = np.reciprocal(
        np.sqrt(mean_square + np.float32(epsilon), dtype=np.float32),
        dtype=np.float32,
    )
    unit = _numpy_bf16(
        np.multiply(input_bf16, inverse, dtype=np.float32)
    )
    return _numpy_bf16(
        np.multiply(unit, scale_bf16, dtype=np.float32)
    )


def numpy_recompose_experiment5_full_ffn(
    expert_output: Any,
    shared_normalized: Any,
    router_weights: Any,
    recomposition: Experiment5Recomposition,
) -> np.ndarray:
    """Independent NumPy oracle for exact full-FFN recomposition."""

    expert = np.asarray(expert_output)
    shared = np.asarray(shared_normalized)
    weights = np.asarray(router_weights)
    if expert.ndim != 3:
        raise Experiment5Error("expert_output must be rank three")
    if shared.shape != (expert.shape[0], expert.shape[2]):
        raise Experiment5Error("shared_normalized must match expert token/hidden axes")
    if weights.shape != expert.shape[:2]:
        raise Experiment5Error("router_weights must match expert token/route axes")
    expert_bf16 = _numpy_bf16(expert)
    weights_bf16 = _numpy_bf16(weights)
    raw_routed = np.einsum(
        "nkh,nk->nh",
        expert_bf16,
        weights_bf16,
        dtype=np.float32,
        optimize=False,
    )
    routed = _numpy_rmsnorm_bf16(
        raw_routed,
        recomposition.routed_scale,
        epsilon=float(recomposition.epsilon),
    )
    branch_sum = _numpy_bf16(
        np.add(routed, _numpy_bf16(shared), dtype=np.float32)
    )
    return _numpy_rmsnorm_bf16(
        branch_sum,
        recomposition.outer_scale,
        epsilon=float(recomposition.epsilon),
    )


def _jax_hybrid_sources(
    batch: Experiment5Batch,
    centering: Experiment5Centering,
    encoder_kind: Experiment5EncoderKind,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    expert = jnp.asarray(batch.expert_output)
    routed_input = jnp.asarray(batch.routed_input)
    shared = jnp.asarray(batch.shared_normalized)
    if encoder_kind == "sae":
        return (
            expert,
            shared,
            jnp.asarray(centering.routed_target_mean),
            jnp.asarray(centering.shared_target_mean),
        )
    return (
        jnp.broadcast_to(
            routed_input[:, None, :],
            expert.shape,
        ),
        routed_input,
        jnp.asarray(centering.source_mean),
        jnp.asarray(centering.source_mean),
    )


def _jax_hybrid_candidate_scores(
    params: SAEParams,
    routed_source: jax.Array,
    shared_source: jax.Array,
    bank_ids: jax.Array,
    routed_source_mean: jax.Array,
    shared_source_mean: jax.Array,
    layout: Experiment5Layout,
    encoder_kind: Experiment5EncoderKind,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Dense correctness graph for visible routed banks and shared dictionary."""

    tokens = routed_source.shape[0]
    flat_source = routed_source.reshape(-1, layout.hidden_dim)
    flat_banks = jnp.asarray(bank_ids, jnp.int32).reshape(-1)
    valid_banks = (flat_banks >= 0) & (flat_banks < layout.num_experts)
    safe_banks = jnp.clip(flat_banks, 0, layout.num_experts - 1)
    compute_dtype = params.encoder.dtype
    centered_routed = (
        flat_source.astype(compute_dtype)
        - jnp.asarray(routed_source_mean).astype(compute_dtype)[None, :]
    )
    routed_encoder = params.encoder[: layout.routed_features].reshape(
        layout.num_experts,
        layout.routed_bank_width,
        layout.hidden_dim,
    )
    routed_bias = params.bias[: layout.routed_features].reshape(
        layout.num_experts,
        layout.routed_bank_width,
    )
    pre_routed = jnp.einsum(
        "md,mwd->mw",
        centered_routed,
        routed_encoder[safe_banks].astype(compute_dtype),
        preferred_element_type=jnp.float32,
    ) + routed_bias[safe_banks].astype(jnp.float32)
    routed_scores = jnp.where(
        valid_banks[:, None],
        _jax_activation(pre_routed, encoder_kind),
        -jnp.inf,
    )
    local_ids = jnp.arange(layout.routed_bank_width, dtype=jnp.int32)
    routed_ids = (
        safe_banks[:, None] * layout.routed_bank_width + local_ids[None, :]
    )

    centered_shared = (
        shared_source.astype(compute_dtype)
        - jnp.asarray(shared_source_mean).astype(compute_dtype)[None, :]
    )
    shared_encoder = params.encoder[layout.routed_features :]
    shared_bias = params.bias[layout.routed_features :]
    pre_shared = jnp.einsum(
        "nd,fd->nf",
        centered_shared,
        shared_encoder.astype(compute_dtype),
        preferred_element_type=jnp.float32,
    ) + shared_bias.astype(jnp.float32)
    shared_scores = _jax_activation(pre_shared, encoder_kind)
    shared_ids = jnp.broadcast_to(
        (
            layout.routed_features
            + jnp.arange(layout.shared_width, dtype=jnp.int32)
        )[None, :],
        (tokens, layout.shared_width),
    )
    return routed_scores, routed_ids, shared_scores, shared_ids


def forward_experiment5_hybrid(
    params: SAEParams,
    batch: Experiment5Batch,
    centering: Experiment5Centering,
    recomposition: Experiment5Recomposition,
    layout: Experiment5Layout,
    *,
    encoder_kind: Experiment5EncoderKind,
) -> Experiment5Forward:
    """Forward H/R with exact per-route/shared TopK and sparse column decode."""

    if not layout.is_hybrid:
        raise Experiment5Error("hybrid forward requires condition H or R")
    selected_kind = _validate_encoder_kind(encoder_kind)
    _validate_params(params, layout)
    _validate_centering_and_recomposition(centering, recomposition, layout)
    source = jnp.asarray(batch.routed_input)
    expert = jnp.asarray(batch.expert_output)
    shared = jnp.asarray(batch.shared_normalized)
    banks = jnp.asarray(batch.bank_ids)
    if source.ndim != 2 or source.shape[1] != layout.hidden_dim:
        raise Experiment5Error(
            f"routed_input must have shape [N,{layout.hidden_dim}]"
        )
    if expert.shape != (
        source.shape[0],
        layout.top_routed,
        layout.hidden_dim,
    ):
        raise Experiment5Error(
            f"expert_output must have shape [N,{layout.top_routed},{layout.hidden_dim}]"
        )
    if shared.shape != source.shape:
        raise Experiment5Error("shared_normalized must match routed_input")
    if banks.shape != expert.shape[:2]:
        raise Experiment5Error("bank_ids must match token and route axes")
    routed_source, shared_source, routed_source_mean, shared_source_mean = (
        _jax_hybrid_sources(batch, centering, selected_kind)
    )
    routed_scores, routed_ids, shared_scores, shared_ids = (
        _jax_hybrid_candidate_scores(
            params,
            routed_source,
            shared_source,
            banks,
            routed_source_mean,
            shared_source_mean,
            layout,
            selected_kind,
        )
    )
    routed_selection = positive_topk(
        routed_scores,
        routed_ids,
        layout.routed_k,
    )
    shared_selection = positive_topk(
        shared_scores,
        shared_ids,
        layout.shared_k,
    )
    routed_reconstruction = decode_selected_columns(
        params.decoder,
        routed_selection,
        jnp.asarray(centering.routed_target_mean),
    ).reshape(
        source.shape[0],
        layout.top_routed,
        layout.hidden_dim,
    )
    shared_reconstruction = decode_selected_columns(
        params.decoder,
        shared_selection,
        jnp.asarray(centering.shared_target_mean),
    )
    full = jax_recompose_experiment5_full_ffn(
        routed_reconstruction,
        shared_reconstruction,
        jnp.asarray(batch.router_weights),
        recomposition,
    )
    return Experiment5Forward(
        full_reconstruction=full,
        routed_reconstruction=routed_reconstruction,
        shared_reconstruction=shared_reconstruction,
        routed_selection=routed_selection,
        shared_selection=shared_selection,
        global_selection=None,
    )


def _jax_global_source_and_mean(
    batch: Experiment5Batch,
    centering: Experiment5Centering,
    recomposition: Experiment5Recomposition,
    encoder_kind: Experiment5EncoderKind,
) -> tuple[jax.Array, jax.Array]:
    if encoder_kind == "sae":
        target = jax_recompose_experiment5_full_ffn(
            jnp.asarray(batch.expert_output),
            jnp.asarray(batch.shared_normalized),
            jnp.asarray(batch.router_weights),
            recomposition,
        )
        return target, jnp.asarray(centering.full_target_mean)
    return (
        jnp.asarray(batch.routed_input),
        jnp.asarray(centering.source_mean),
    )


def forward_experiment5_global(
    params: SAEParams,
    batch: Experiment5Batch,
    centering: Experiment5Centering,
    recomposition: Experiment5Recomposition,
    layout: Experiment5Layout,
    *,
    encoder_kind: Experiment5EncoderKind,
) -> Experiment5Forward:
    """Forward B/F global dictionary on full target (SAE) or common x (TC)."""

    if layout.is_hybrid:
        raise Experiment5Error("global forward requires condition B or F")
    selected_kind = _validate_encoder_kind(encoder_kind)
    _validate_params(params, layout)
    _validate_centering_and_recomposition(centering, recomposition, layout)
    source, source_mean = _jax_global_source_and_mean(
        batch,
        centering,
        recomposition,
        selected_kind,
    )
    compute_dtype = params.encoder.dtype
    centered = (
        source.astype(compute_dtype)
        - source_mean.astype(compute_dtype)[None, :]
    )
    pre = jnp.einsum(
        "nd,fd->nf",
        centered,
        params.encoder.astype(compute_dtype),
        preferred_element_type=jnp.float32,
    ) + params.bias.astype(jnp.float32)
    scores = _jax_activation(pre, selected_kind)
    latent_ids = jnp.broadcast_to(
        jnp.arange(layout.total_features, dtype=jnp.int32)[None, :],
        scores.shape,
    )
    selection = positive_topk(scores, latent_ids, layout.total_l0)
    reconstruction = decode_selected_columns(
        params.decoder,
        selection,
        jnp.asarray(centering.full_target_mean),
    )
    return Experiment5Forward(
        full_reconstruction=reconstruction,
        routed_reconstruction=None,
        shared_reconstruction=None,
        routed_selection=None,
        shared_selection=None,
        global_selection=selection,
    )


def forward_experiment5(
    params: SAEParams,
    batch: Experiment5Batch,
    centering: Experiment5Centering,
    recomposition: Experiment5Recomposition,
    layout: Experiment5Layout,
    *,
    encoder_kind: Experiment5EncoderKind,
) -> Experiment5Forward:
    """Dispatch the frozen H/R hybrid or B/F global scientific graph."""

    if layout.is_hybrid:
        return forward_experiment5_hybrid(
            params,
            batch,
            centering,
            recomposition,
            layout,
            encoder_kind=encoder_kind,
        )
    return forward_experiment5_global(
        params,
        batch,
        centering,
        recomposition,
        layout,
        encoder_kind=encoder_kind,
    )


def _numpy_select(
    scores: np.ndarray,
    latent_ids: np.ndarray,
    k: int,
) -> Experiment5NumpySelection:
    rows = scores.shape[0]
    values = np.zeros((rows, k), dtype=np.float32)
    selected_ids = np.zeros((rows, k), dtype=np.int32)
    valid = np.zeros((rows,), dtype=np.bool_)
    for row in range(rows):
        positive = np.flatnonzero(scores[row] > 0)
        if positive.size < k:
            continue
        ranked = np.lexsort(
            (
                latent_ids[row, positive],
                -scores[row, positive],
            )
        )
        chosen = positive[ranked[:k]]
        chosen = chosen[
            np.argsort(latent_ids[row, chosen], kind="stable")
        ]
        values[row] = scores[row, chosen]
        selected_ids[row] = latent_ids[row, chosen]
        valid[row] = True
    return Experiment5NumpySelection(values, selected_ids, valid)


def _numpy_decode(
    decoder: np.ndarray,
    selection: Experiment5NumpySelection,
    mean: np.ndarray,
    *,
    parameter_is_bf16: bool,
) -> np.ndarray:
    columns = decoder.astype(np.float32)[:, selection.latent_ids]
    decode_values = (
        _numpy_bf16(selection.values)
        if parameter_is_bf16
        else selection.values.astype(np.float32)
    )
    return np.einsum(
        "nk,hnk->nh",
        decode_values,
        columns,
        dtype=np.float32,
    ) + mean.astype(np.float32)[None, :]


def _numpy_full_target(
    batch: Experiment5Batch,
    recomposition: Experiment5Recomposition,
) -> np.ndarray:
    return numpy_recompose_experiment5_full_ffn(
        batch.expert_output,
        batch.shared_normalized,
        batch.router_weights,
        recomposition,
    )


def numpy_forward_experiment5(
    params: SAEParams,
    batch: Experiment5Batch,
    centering: Experiment5Centering,
    recomposition: Experiment5Recomposition,
    layout: Experiment5Layout,
    *,
    encoder_kind: Experiment5EncoderKind,
) -> Experiment5NumpyForward:
    """Independent bounded NumPy oracle for H/R/B/F sparse forward."""

    selected_kind = _validate_encoder_kind(encoder_kind)
    _validate_params(params, layout)
    _validate_centering_and_recomposition(centering, recomposition, layout)
    parameter_is_bf16 = "bfloat16" in str(np.asarray(params.encoder).dtype)
    encoder = np.asarray(params.encoder).astype(np.float32)
    decoder = np.asarray(params.decoder).astype(np.float32)
    bias = np.asarray(params.bias).astype(np.float32)
    source = np.asarray(batch.routed_input, dtype=np.float32)
    expert = np.asarray(batch.expert_output, dtype=np.float32)
    shared = np.asarray(batch.shared_normalized, dtype=np.float32)
    banks = np.asarray(batch.bank_ids, dtype=np.int32)
    source_mean = np.asarray(centering.source_mean, dtype=np.float32)
    routed_mean = np.asarray(centering.routed_target_mean, dtype=np.float32)
    shared_mean = np.asarray(centering.shared_target_mean, dtype=np.float32)
    full_mean = np.asarray(centering.full_target_mean, dtype=np.float32)
    if layout.is_hybrid:
        if selected_kind == "sae":
            routed_source = expert
            shared_source = shared
            routed_source_mean = routed_mean
            shared_source_mean = shared_mean
        else:
            routed_source = np.broadcast_to(source[:, None, :], expert.shape)
            shared_source = source
            routed_source_mean = source_mean
            shared_source_mean = source_mean
        flat_source = routed_source.reshape(-1, layout.hidden_dim)
        flat_banks = banks.reshape(-1).astype(np.int64)
        valid_banks = (flat_banks >= 0) & (flat_banks < layout.num_experts)
        safe_banks = np.clip(flat_banks, 0, layout.num_experts - 1)
        bank_encoder = encoder[: layout.routed_features].reshape(
            layout.num_experts,
            layout.routed_bank_width,
            layout.hidden_dim,
        )
        bank_bias = bias[: layout.routed_features].reshape(
            layout.num_experts,
            layout.routed_bank_width,
        )
        routed_centered = flat_source - routed_source_mean[None, :]
        if parameter_is_bf16:
            routed_centered = _numpy_bf16(
                _numpy_bf16(flat_source)
                - _numpy_bf16(routed_source_mean)[None, :]
            )
        pre_routed = np.einsum(
            "md,mwd->mw",
            routed_centered,
            bank_encoder[safe_banks],
            dtype=np.float32,
        ) + bank_bias[safe_banks]
        routed_scores = np.where(
            valid_banks[:, None],
            _numpy_activation(pre_routed, selected_kind),
            -np.inf,
        )
        local_ids = np.arange(layout.routed_bank_width, dtype=np.int32)
        routed_ids = (
            safe_banks.astype(np.int32)[:, None] * layout.routed_bank_width
            + local_ids[None, :]
        )
        routed_selection = _numpy_select(
            routed_scores,
            routed_ids,
            layout.routed_k,
        )
        routed_reconstruction = _numpy_decode(
            decoder,
            routed_selection,
            routed_mean,
            parameter_is_bf16=parameter_is_bf16,
        ).reshape(
            source.shape[0],
            layout.top_routed,
            layout.hidden_dim,
        )

        shared_encoder = encoder[layout.routed_features :]
        shared_bias = bias[layout.routed_features :]
        shared_centered = shared_source - shared_source_mean[None, :]
        if parameter_is_bf16:
            shared_centered = _numpy_bf16(
                _numpy_bf16(shared_source)
                - _numpy_bf16(shared_source_mean)[None, :]
            )
        pre_shared = np.einsum(
            "nd,fd->nf",
            shared_centered,
            shared_encoder,
            dtype=np.float32,
        ) + shared_bias
        shared_scores = _numpy_activation(pre_shared, selected_kind)
        shared_ids = np.broadcast_to(
            (
                layout.routed_features
                + np.arange(layout.shared_width, dtype=np.int32)
            )[None, :],
            shared_scores.shape,
        )
        shared_selection = _numpy_select(
            shared_scores,
            shared_ids,
            layout.shared_k,
        )
        shared_reconstruction = _numpy_decode(
            decoder,
            shared_selection,
            shared_mean,
            parameter_is_bf16=parameter_is_bf16,
        )
        full = numpy_recompose_experiment5_full_ffn(
            routed_reconstruction,
            shared_reconstruction,
            batch.router_weights,
            recomposition,
        )
        return Experiment5NumpyForward(
            full,
            routed_reconstruction,
            shared_reconstruction,
            routed_selection,
            shared_selection,
            None,
        )

    if selected_kind == "sae":
        global_source = _numpy_full_target(batch, recomposition)
        global_source_mean = full_mean
    else:
        global_source = source
        global_source_mean = source_mean
    global_centered = global_source - global_source_mean[None, :]
    if parameter_is_bf16:
        global_centered = _numpy_bf16(
            _numpy_bf16(global_source)
            - _numpy_bf16(global_source_mean)[None, :]
        )
    pre = np.einsum(
        "nd,fd->nf",
        global_centered,
        encoder,
        dtype=np.float32,
    ) + bias
    scores = _numpy_activation(pre, selected_kind)
    latent_ids = np.broadcast_to(
        np.arange(layout.total_features, dtype=np.int32)[None, :],
        scores.shape,
    )
    selection = _numpy_select(scores, latent_ids, layout.total_l0)
    reconstruction = _numpy_decode(
        decoder,
        selection,
        full_mean,
        parameter_is_bf16=parameter_is_bf16,
    )
    return Experiment5NumpyForward(
        reconstruction,
        None,
        None,
        None,
        None,
        selection,
    )


def _experiment5_loss_sums(
    params: SAEParams,
    batch: Experiment5Batch,
    centering: Experiment5Centering,
    recomposition: Experiment5Recomposition,
    layout: Experiment5Layout,
    encoder_kind: Experiment5EncoderKind,
) -> Experiment5LossSums:
    output = forward_experiment5(
        params,
        batch,
        centering,
        recomposition,
        layout,
        encoder_kind=encoder_kind,
    )
    target = jax_recompose_experiment5_full_ffn(
        jnp.asarray(batch.expert_output),
        jnp.asarray(batch.shared_normalized),
        jnp.asarray(batch.router_weights),
        recomposition,
    ).astype(jnp.float32)
    reconstruction = output.full_reconstruction.astype(jnp.float32)
    valid = jnp.asarray(batch.valid_mask)
    full_mean = jnp.asarray(centering.full_target_mean, jnp.float32)
    if valid.shape != (target.shape[0],) or valid.dtype != jnp.bool_:
        raise Experiment5Error("valid_mask must be boolean [tokens]")
    token_sse = jnp.sum(
        jax.lax.square(target - reconstruction),
        axis=1,
        dtype=jnp.float32,
    )
    token_sst = jnp.sum(
        jax.lax.square(target - full_mean[None, :]),
        axis=1,
        dtype=jnp.float32,
    )
    valid_weight = valid.astype(jnp.float32)
    sse = jnp.sum(token_sse * valid_weight, dtype=jnp.float32)
    sst = jnp.sum(token_sst * valid_weight, dtype=jnp.float32)

    fired = jnp.zeros((layout.total_features,), dtype=jnp.int32)
    l0_sum = jnp.asarray(0, jnp.int32)
    topk_valid = jnp.asarray(True)
    if layout.is_hybrid:
        assert output.routed_selection is not None
        assert output.shared_selection is not None
        routed_valid = output.routed_selection.valid.reshape(
            target.shape[0],
            layout.top_routed,
        )
        routed_positive = (output.routed_selection.values > 0).reshape(
            target.shape[0],
            layout.top_routed,
            layout.routed_k,
        )
        routed_scientific = (
            routed_positive
            & routed_valid[..., None]
            & valid[:, None, None]
        )
        shared_valid = output.shared_selection.valid
        shared_positive = output.shared_selection.values > 0
        shared_scientific = (
            shared_positive & shared_valid[:, None] & valid[:, None]
        )
        topk_valid = (
            jnp.all(routed_valid | ~valid[:, None])
            & jnp.all(shared_valid | ~valid)
        )
        l0_sum = (
            jnp.sum(routed_scientific, dtype=jnp.int32)
            + jnp.sum(shared_scientific, dtype=jnp.int32)
        )
        routed_ids = output.routed_selection.latent_ids.reshape(-1)
        routed_fired = routed_scientific.reshape(-1).astype(jnp.int32)
        shared_ids = output.shared_selection.latent_ids.reshape(-1)
        shared_fired = shared_scientific.reshape(-1).astype(jnp.int32)
        fired = fired.at[routed_ids].max(routed_fired)
        fired = fired.at[shared_ids].max(shared_fired)
    else:
        assert output.global_selection is not None
        selected_valid = output.global_selection.valid
        selected_positive = output.global_selection.values > 0
        scientific = (
            selected_positive & selected_valid[:, None] & valid[:, None]
        )
        topk_valid = jnp.all(selected_valid | ~valid)
        l0_sum = jnp.sum(scientific, dtype=jnp.int32)
        fired = fired.at[
            output.global_selection.latent_ids.reshape(-1)
        ].max(scientific.reshape(-1).astype(jnp.int32))

    weights = jnp.asarray(batch.router_weights, jnp.float32)
    banks = jnp.asarray(batch.bank_ids)
    scientific_routes = valid[:, None]
    router_valid = (
        jnp.all(jnp.isfinite(weights) | ~scientific_routes)
        & jnp.all((weights >= 0) | ~scientific_routes)
        & jnp.all(
            (
                jnp.abs(
                    jnp.sum(weights, axis=1, dtype=jnp.float32) - 1.0
                )
                <= ROUTER_WEIGHT_SUM_ATOL
            )
            | ~valid
        )
    )
    bank_valid = jnp.asarray(True)
    if layout.is_hybrid:
        bank_valid = (
            jnp.all(
                (
                    (banks >= 0)
                    & (banks < layout.num_experts)
                )
                | ~scientific_routes
            )
            & jnp.all(
                (jnp.diff(jnp.sort(banks, axis=1), axis=1) != 0)
                | ~valid[:, None]
            )
        )
    finite_inputs = (
        jnp.all(
            jnp.isfinite(jnp.asarray(batch.routed_input))
            | ~valid[:, None]
        )
        & jnp.all(
            jnp.isfinite(jnp.asarray(batch.expert_output))
            | ~valid[:, None, None]
        )
        & jnp.all(
            jnp.isfinite(jnp.asarray(batch.shared_normalized))
            | ~valid[:, None]
        )
        & jnp.all(jnp.isfinite(full_mean))
        & jnp.all(jnp.isfinite(jnp.asarray(recomposition.routed_scale)))
        & jnp.all(jnp.isfinite(jnp.asarray(recomposition.outer_scale)))
    )
    valid_tokens = jnp.sum(valid, dtype=jnp.int32)
    expected_l0 = valid_tokens * layout.total_l0
    topk_valid = (
        topk_valid
        & router_valid
        & bank_valid
        & finite_inputs
        & (l0_sum == expected_l0)
    )
    return Experiment5LossSums(
        sse=sse,
        sst=sst,
        l0_sum=l0_sum,
        valid_tokens=valid_tokens,
        topk_valid=topk_valid,
        fired_mask=fired > 0,
    )


def normalized_experiment5_loss(
    params: SAEParams,
    batch: Experiment5Batch,
    centering: Experiment5Centering,
    recomposition: Experiment5Recomposition,
    layout: Experiment5Layout,
    *,
    encoder_kind: Experiment5EncoderKind,
) -> tuple[jax.Array, Experiment5LossSums]:
    """Return the common full-FFN NMSE and exact additive sufficient sums."""

    selected_kind = _validate_encoder_kind(encoder_kind)
    sums = _experiment5_loss_sums(
        params,
        batch,
        centering,
        recomposition,
        layout,
        selected_kind,
    )
    healthy = sums.topk_valid & (sums.sst > 0)
    loss = jnp.where(
        healthy,
        sums.sse / jnp.maximum(sums.sst, _EPS),
        jnp.asarray(jnp.nan, jnp.float32),
    )
    return loss, sums


def experiment5_loss_and_grad_sums(
    params: SAEParams,
    batch: Experiment5Batch,
    centering: Experiment5Centering,
    recomposition: Experiment5Recomposition,
    layout: Experiment5Layout,
    *,
    encoder_kind: Experiment5EncoderKind,
) -> tuple[Experiment5LossSums, SAEParams]:
    """Differentiate unnormalized full-FFN SSE for microbatch accumulation."""

    selected_kind = _validate_encoder_kind(encoder_kind)

    def objective(
        selected: SAEParams,
    ) -> tuple[jax.Array, Experiment5LossSums]:
        sums = _experiment5_loss_sums(
            selected,
            batch,
            centering,
            recomposition,
            layout,
            selected_kind,
        )
        return sums.sse, sums

    (_, sums), gradients = jax.value_and_grad(objective, has_aux=True)(params)
    return sums, gradients


def experiment5_train_step(
    state: TrainState,
    batch: Experiment5Batch,
    centering: Experiment5Centering,
    recomposition: Experiment5Recomposition,
    layout: Experiment5Layout,
    optimizer: ContributionOptimizerConfig | None = None,
    *,
    encoder_kind: Experiment5EncoderKind,
) -> tuple[TrainState, StepMetrics]:
    """Apply one existing-Adam update to the common full-FFN objective."""

    if not isinstance(state, TrainState):
        raise TypeError("state must be TrainState")
    optimizer_config = experiment5_optimizer_sae_config(layout, optimizer)
    sums, gradients = experiment5_loss_and_grad_sums(
        state.params,
        batch,
        centering,
        recomposition,
        layout,
        encoder_kind=encoder_kind,
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
    raw_l0 = sums.l0_sum.astype(jnp.float32) / jnp.maximum(
        sums.valid_tokens.astype(jnp.float32),
        1.0,
    )
    l0 = jnp.where(
        sums.l0_sum == sums.valid_tokens * layout.total_l0,
        jnp.asarray(layout.total_l0, jnp.float32),
        raw_l0,
    )
    return apply_gradients(
        state,
        normalized_gradients,
        optimizer_config,
        loss=loss,
        valid_tokens=sums.valid_tokens,
        fired_mask=sums.fired_mask,
        l0=l0,
        sse=sums.sse,
        sst=sums.sst,
        topk_valid=sums.topk_valid,
    )


def experiment5_token_scores(
    target: Any,
    reconstruction: Any,
    full_target_mean: Any,
    *,
    valid_mask: Any | None = None,
) -> Experiment5TokenScores:
    """Return common FP64 held-out SSE/SST rows for one condition."""

    target_array = np.asarray(target)
    estimate = np.asarray(reconstruction)
    mean = np.asarray(full_target_mean)
    if target_array.ndim != 2:
        raise Experiment5Error("target must have shape [tokens,hidden]")
    if estimate.shape != target_array.shape:
        raise Experiment5Error("reconstruction must match target")
    if mean.shape != (target_array.shape[1],):
        raise Experiment5Error(
            f"full_target_mean must have shape [{target_array.shape[1]}]"
        )
    for name, value in (
        ("target", target_array),
        ("reconstruction", estimate),
        ("full_target_mean", mean),
    ):
        if not np.issubdtype(value.dtype, np.floating):
            raise Experiment5Error(f"{name} must be floating-point")
        if not np.isfinite(value).all():
            raise Experiment5Error(f"{name} must be finite")
    if valid_mask is None:
        valid = np.ones((target_array.shape[0],), dtype=np.bool_)
    else:
        valid = np.asarray(valid_mask)
        if valid.shape != (target_array.shape[0],) or valid.dtype != np.bool_:
            raise Experiment5Error("valid_mask must be boolean [tokens]")
    target32 = target_array.astype(np.float32)
    estimate32 = estimate.astype(np.float32)
    mean32 = mean.astype(np.float32)
    token_sse = np.sum(
        np.square(target32 - estimate32, dtype=np.float32),
        axis=1,
        dtype=np.float64,
    )
    token_sst = np.sum(
        np.square(target32 - mean32[None, :], dtype=np.float32),
        axis=1,
        dtype=np.float64,
    )
    return Experiment5TokenScores(
        np.where(valid, token_sse, 0.0),
        np.where(valid, token_sst, 0.0),
        valid.copy(),
    )


__all__ = [
    "EXPERIMENT5_CONDITIONS",
    "EXPERIMENT5_ENCODER_KINDS",
    "Experiment5Batch",
    "Experiment5Centering",
    "Experiment5Condition",
    "Experiment5EncoderKind",
    "Experiment5Error",
    "Experiment5Forward",
    "Experiment5Layout",
    "Experiment5LossSums",
    "Experiment5NumpyForward",
    "Experiment5NumpySelection",
    "Experiment5Recomposition",
    "Experiment5TokenScores",
    "experiment5_bank_ids",
    "experiment5_loss_and_grad_sums",
    "experiment5_optimizer_sae_config",
    "experiment5_token_scores",
    "experiment5_train_step",
    "forward_experiment5",
    "forward_experiment5_global",
    "forward_experiment5_hybrid",
    "init_experiment5_train_state",
    "jax_experiment5_routed_branch",
    "jax_recompose_experiment5_full_ffn",
    "normalized_experiment5_loss",
    "numpy_forward_experiment5",
    "numpy_recompose_experiment5_full_ffn",
    "production_experiment5_layout",
    "stack_experiment5_microbatches",
    "validate_experiment5_microbatch",
]
