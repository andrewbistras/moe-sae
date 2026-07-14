"""Training, checkpoint, health-gate, and evaluation orchestration.

The orchestration deliberately accepts plain iterables of array-like batches.
Artifact readers can therefore remain independent from the JAX scientific
core.  A batch may be a mapping or an object with ``activation``, optional
``expert_ids``, and optional ``valid_mask`` fields.  Extra fields, including
router weights, are ignored.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from moe_sae.artifacts import bf16_bits_to_float32
from moe_sae.sae import (
    DEFAULT_BATCH_TOKENS,
    DEFAULT_MICROBATCH_TOKENS,
    DEFAULT_TOTAL_TOKENS,
    SAEConfig,
    SAEParams,
    TrainState,
    apply_gradients,
    init_train_state,
    loss_and_grad_sums,
    make_grouping_plan,
    normalized_sse_loss,
)


CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class ActivationBatch:
    activation: Any
    expert_ids: Any | None = None
    valid_mask: Any | None = None


@dataclass(frozen=True)
class TrainingConfig:
    sae: SAEConfig
    seed: int
    total_tokens: int = DEFAULT_TOTAL_TOKENS
    effective_batch_tokens: int = DEFAULT_BATCH_TOKENS
    microbatch_tokens: int = DEFAULT_MICROBATCH_TOKENS
    checkpoint_every_tokens: int = 10_000_000
    health_gate_tokens: int = 5_000_000
    live_feature_fraction: float = 0.95
    loss_ratio_threshold: float = 0.80
    a_capacity: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "total_tokens",
            "effective_batch_tokens",
            "microbatch_tokens",
            "checkpoint_every_tokens",
            "health_gate_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.effective_batch_tokens % self.microbatch_tokens:
            raise ValueError(
                "effective batch must be an integer number of microbatches"
            )
        if not 0.0 <= self.live_feature_fraction <= 1.0:
            raise ValueError("live_feature_fraction must be in [0, 1]")
        if self.loss_ratio_threshold <= 0:
            raise ValueError("loss_ratio_threshold must be positive")
        if self.a_capacity is not None and self.a_capacity <= 0:
            raise ValueError("a_capacity must be positive")
        if self.sae.total_steps != self.updates:
            raise ValueError(
                f"SAE LR schedule has {self.sae.total_steps} steps but the token plan "
                f"requires {self.updates}; set SAEConfig.total_steps={self.updates}"
            )

    @property
    def microbatches_per_update(self) -> int:
        return self.effective_batch_tokens // self.microbatch_tokens

    @property
    def updates(self) -> int:
        return math.ceil(self.total_tokens / self.effective_batch_tokens)

    @property
    def final_padding(self) -> int:
        return self.updates * self.effective_batch_tokens - self.total_tokens


@dataclass(frozen=True)
class TrainingMetric:
    step: int
    tokens_seen: int
    loss: float
    sse: float
    sst: float
    l0: float
    valid_tokens: int
    topk_valid: bool
    grad_norm: float
    clip_coefficient: float
    learning_rate: float
    live_fraction: float


@dataclass(frozen=True)
class HealthGateResult:
    passed: bool
    finite: bool
    exact_l0: bool
    topk_valid: bool
    enough_tokens: bool
    loss_ratio: float
    loss_improved: bool
    live_fraction: float
    live_features_ok: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingResult:
    state: TrainState
    metrics: tuple[TrainingMetric, ...]
    health: HealthGateResult | None
    latest_checkpoint: Path | None
    final_checkpoint: Path | None


@dataclass(frozen=True)
class EvaluationResult:
    condition: str
    tokens: int
    sse: float
    sst: float
    computed_sst: float
    nmse: float
    explained_variance: float
    l0: float
    dead_rate: float
    topk_valid: bool


def planned_token_mask(
    update_index: int,
    *,
    total_tokens: int = DEFAULT_TOTAL_TOKENS,
    batch_tokens: int = DEFAULT_BATCH_TOKENS,
) -> np.ndarray:
    """Static-shape mask for an update in the exact token plan."""

    if update_index < 0:
        raise ValueError("update_index must be non-negative")
    start = update_index * batch_tokens
    valid = min(max(total_tokens - start, 0), batch_tokens)
    mask = np.zeros((batch_tokens,), dtype=bool)
    mask[:valid] = True
    return mask


def _field(batch: Any, name: str, default: Any = None) -> Any:
    if isinstance(batch, Mapping):
        return batch.get(name, default)
    return getattr(batch, name, default)


def _scientific_rows(
    batch: Any, config: SAEConfig
) -> tuple[np.ndarray, np.ndarray | None]:
    activation = np.asarray(_field(batch, "activation"))
    # Capture artifacts store logical BF16 tensors as their exact uint16 wire
    # bits.  Treating those integers as activations silently produces enormous
    # bogus values, so decoding is part of this input boundary rather than an
    # optional reader convenience.
    if activation.dtype == np.uint16:
        activation = bf16_bits_to_float32(activation)
    if activation.ndim != 2 or activation.shape[1] != config.hidden_dim:
        raise ValueError(
            f"activation must be [tokens, {config.hidden_dim}], got {activation.shape}"
        )
    mask_value = _field(batch, "valid_mask")
    mask = (
        np.ones((activation.shape[0],), dtype=bool)
        if mask_value is None
        else np.asarray(mask_value, dtype=bool)
    )
    if mask.shape != (activation.shape[0],):
        raise ValueError("valid_mask shape does not match activation rows")
    activation = activation[mask]

    ids_value = _field(batch, "expert_ids")
    expert_ids: np.ndarray | None = None
    if config.condition == "A":
        if ids_value is None:
            raise ValueError("condition A batches require expert_ids")
        ids = np.asarray(ids_value)
        if ids.shape != (mask.shape[0], config.top_routed):
            raise ValueError(
                f"expert_ids must be {(mask.shape[0], config.top_routed)}, got {ids.shape}"
            )
        expert_ids = ids[mask].astype(np.int32, copy=False)
        if expert_ids.size:
            if expert_ids.min() < 0 or expert_ids.max() >= config.num_experts:
                raise ValueError("expert id outside configured range")
            if np.any(np.diff(np.sort(expert_ids, axis=1), axis=1) == 0):
                raise ValueError("router expert ids must be unique within every token")
    return activation, expert_ids


def _fixed_microbatches(
    batches: Iterable[Any],
    config: TrainingConfig,
    *,
    skip_tokens: int,
    take_tokens: int,
) -> Iterator[ActivationBatch]:
    """Pack arbitrary artifact batches into static microbatches.

    The input iterable must begin at scientific token zero.  ``skip_tokens``
    gives deterministic resume semantics; invalid/padded source rows do not
    count.  Only the final emitted microbatch is padded.
    """

    micro = config.microbatch_tokens
    skip_remaining = int(skip_tokens)
    take_remaining = int(take_tokens)
    chunks_x: list[np.ndarray] = []
    chunks_ids: list[np.ndarray] = []
    buffered = 0
    template_dtype: np.dtype[Any] | None = None
    emitted = 0

    def emit(padded: bool = False) -> ActivationBatch:
        nonlocal chunks_x, chunks_ids, buffered, emitted
        x = np.concatenate(chunks_x, axis=0) if len(chunks_x) > 1 else chunks_x[0]
        ids = None
        if config.sae.condition == "A":
            ids = (
                np.concatenate(chunks_ids, axis=0)
                if len(chunks_ids) > 1
                else chunks_ids[0]
            )
        valid = np.ones((buffered,), dtype=bool)
        if padded and buffered < micro:
            pad = micro - buffered
            x = np.pad(x, ((0, pad), (0, 0)), mode="constant")
            valid = np.pad(valid, (0, pad), mode="constant")
            if ids is not None:
                # A legal unique route for masked rows keeps grouping-plan
                # validation independent from scientific validity.  Balance
                # masked assignments over every expert; routing all pad rows
                # to experts 0..7 would inflate capacity to O(microbatch).
                rows = np.arange(pad, dtype=np.int32)[:, None]
                slots = np.arange(config.sae.top_routed, dtype=np.int32)[None, :]
                pad_ids = (
                    rows * config.sae.top_routed + slots
                ) % config.sae.num_experts
                ids = np.concatenate((ids, pad_ids), axis=0)
        result = ActivationBatch(x, ids, valid)
        chunks_x, chunks_ids, buffered = [], [], 0
        emitted += 1
        return result

    def empty_padding() -> ActivationBatch:
        nonlocal emitted
        dtype = np.dtype(np.float32) if template_dtype is None else template_dtype
        x = np.zeros((micro, config.sae.hidden_dim), dtype=dtype)
        ids = None
        if config.sae.condition == "A":
            rows = np.arange(micro, dtype=np.int32)[:, None]
            slots = np.arange(config.sae.top_routed, dtype=np.int32)[None, :]
            ids = (rows * config.sae.top_routed + slots) % config.sae.num_experts
        emitted += 1
        return ActivationBatch(x, ids, np.zeros((micro,), dtype=bool))

    for batch in batches:
        x, ids = _scientific_rows(batch, config.sae)
        template_dtype = x.dtype
        if skip_remaining:
            skipped = min(skip_remaining, x.shape[0])
            x = x[skipped:]
            if ids is not None:
                ids = ids[skipped:]
            skip_remaining -= skipped
        if skip_remaining or not x.shape[0]:
            continue

        offset = 0
        while offset < x.shape[0] and take_remaining > 0:
            take = min(micro - buffered, x.shape[0] - offset, take_remaining)
            chunks_x.append(x[offset : offset + take])
            if ids is not None:
                chunks_ids.append(ids[offset : offset + take])
            buffered += take
            offset += take
            take_remaining -= take
            if buffered == micro:
                yield emit()
            if take_remaining == 0:
                break
        if take_remaining == 0:
            break

    if skip_remaining:
        raise RuntimeError(
            f"batch source ended with {skip_remaining} resume tokens left to skip"
        )
    if take_remaining:
        raise RuntimeError(
            f"batch source ended {take_remaining} scientific tokens early"
        )
    if buffered:
        yield emit(padded=True)
    expected = (
        math.ceil(take_tokens / config.effective_batch_tokens)
        * config.microbatches_per_update
    )
    while emitted < expected:
        yield empty_padding()


def _tree_add(left: SAEParams | None, right: SAEParams) -> SAEParams:
    if left is None:
        return jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), right)
    return jax.tree_util.tree_map(lambda x, y: x + y.astype(jnp.float32), left, right)


def _as_float(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)))


def _as_int(value: Any) -> int:
    return int(np.asarray(jax.device_get(value)))


def _as_bool(value: Any) -> bool:
    return bool(np.asarray(jax.device_get(value)))


def _training_identity(config: TrainingConfig) -> dict[str, Any]:
    return {
        "sae": dataclasses.asdict(config.sae),
        "seed": config.seed,
        "total_tokens": config.total_tokens,
        "effective_batch_tokens": config.effective_batch_tokens,
        "microbatch_tokens": config.microbatch_tokens,
        "a_capacity": config.a_capacity,
    }


def _identity_hash(config: TrainingConfig) -> str:
    encoded = json.dumps(
        _training_identity(config), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def checkpoint_path(directory: str | Path, *, final: bool = False) -> Path:
    return Path(directory) / ("final.npz" if final else "latest.npz")


def save_checkpoint(
    path: str | Path,
    state: TrainState,
    config: TrainingConfig,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save complete parameters, FP32 Adam state, and progress."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "config_hash": _identity_hash(config),
        "config": _training_identity(config),
        "metadata": dict(metadata or {}),
    }
    payload = {
        "header": np.asarray(json.dumps(header, sort_keys=True)),
        "params_encoder": np.asarray(jax.device_get(state.params.encoder)),
        "params_decoder": np.asarray(jax.device_get(state.params.decoder)),
        "params_bias": np.asarray(jax.device_get(state.params.bias)),
        "adam_m_encoder": np.asarray(jax.device_get(state.optimizer.m.encoder)),
        "adam_m_decoder": np.asarray(jax.device_get(state.optimizer.m.decoder)),
        "adam_m_bias": np.asarray(jax.device_get(state.optimizer.m.bias)),
        "adam_v_encoder": np.asarray(jax.device_get(state.optimizer.v.encoder)),
        "adam_v_decoder": np.asarray(jax.device_get(state.optimizer.v.decoder)),
        "adam_v_bias": np.asarray(jax.device_get(state.optimizer.v.bias)),
        "adam_step": np.asarray(jax.device_get(state.optimizer.step)),
        "step": np.asarray(jax.device_get(state.step)),
        "tokens_seen": np.asarray(jax.device_get(state.tokens_seen)),
        "fired_mask": np.asarray(jax.device_get(state.fired_mask), dtype=np.uint8),
        "loss_ema": np.asarray(jax.device_get(state.loss_ema)),
        "initial_loss_ema": np.asarray(jax.device_get(state.initial_loss_ema)),
        "ema_updates": np.asarray(jax.device_get(state.ema_updates)),
    }
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_checkpoint(
    path: str | Path, config: TrainingConfig
) -> tuple[TrainState, dict[str, Any]]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        header = json.loads(str(data["header"]))
        if header.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("unsupported checkpoint version")
        if header.get("config_hash") != _identity_hash(config):
            raise ValueError("checkpoint configuration does not match this run")
        params = SAEParams(
            jnp.asarray(data["params_encoder"]),
            jnp.asarray(data["params_decoder"]),
            jnp.asarray(data["params_bias"]),
        )
        from moe_sae.sae import AdamState  # avoid an unnecessarily wide import surface

        optimizer = AdamState(
            m=SAEParams(
                jnp.asarray(data["adam_m_encoder"], jnp.float32),
                jnp.asarray(data["adam_m_decoder"], jnp.float32),
                jnp.asarray(data["adam_m_bias"], jnp.float32),
            ),
            v=SAEParams(
                jnp.asarray(data["adam_v_encoder"], jnp.float32),
                jnp.asarray(data["adam_v_decoder"], jnp.float32),
                jnp.asarray(data["adam_v_bias"], jnp.float32),
            ),
            step=jnp.asarray(data["adam_step"], jnp.int32),
        )
        state = TrainState(
            params=params,
            optimizer=optimizer,
            step=jnp.asarray(data["step"], jnp.int32),
            tokens_seen=jnp.asarray(data["tokens_seen"], jnp.int32),
            fired_mask=jnp.asarray(data["fired_mask"], bool),
            loss_ema=jnp.asarray(data["loss_ema"], jnp.float32),
            initial_loss_ema=jnp.asarray(data["initial_loss_ema"], jnp.float32),
            ema_updates=jnp.asarray(data["ema_updates"], jnp.int32),
        )
    return state, dict(header.get("metadata", {}))


def _state_is_finite(state: TrainState) -> bool:
    leaves = jax.tree_util.tree_leaves(
        (state.params, state.optimizer.m, state.optimizer.v)
    )
    checks = [jnp.all(jnp.isfinite(leaf)) for leaf in leaves]
    return all(_as_bool(check) for check in checks) and math.isfinite(
        _as_float(state.loss_ema)
    )


def check_seed_health(
    state: TrainState,
    config: TrainingConfig,
    *,
    l0: float,
    topk_valid: bool = True,
) -> HealthGateResult:
    finite = _state_is_finite(state)
    exact_l0 = math.isclose(float(l0), float(config.sae.k), rel_tol=0.0, abs_tol=1e-6)
    enough_tokens = _as_int(state.tokens_seen) >= config.health_gate_tokens
    initial = _as_float(state.initial_loss_ema)
    current = _as_float(state.loss_ema)
    loss_ratio = (
        current / initial if math.isfinite(initial) and initial > 0 else math.inf
    )
    loss_improved = loss_ratio <= config.loss_ratio_threshold
    live_fraction = _as_float(jnp.mean(state.fired_mask.astype(jnp.float32)))
    live_features_ok = (
        True
        if config.sae.condition == "B"
        else live_fraction >= config.live_feature_fraction
    )
    reasons: list[str] = []
    if not finite:
        reasons.append("non-finite parameter, optimizer, or loss state")
    if not exact_l0:
        reasons.append(f"L0 {l0:g} != configured k={config.sae.k}")
    if not topk_valid:
        reasons.append(
            "at least one scientific token had fewer than k positive features"
        )
    if not enough_tokens:
        reasons.append("health gate evaluated before the configured token threshold")
    if not loss_improved:
        reasons.append(
            f"64-update loss EMA ratio {loss_ratio:.4g} exceeds {config.loss_ratio_threshold:g}"
        )
    if not live_features_ok:
        reasons.append(
            f"live feature fraction {live_fraction:.4g} below {config.live_feature_fraction:g}"
        )
    return HealthGateResult(
        passed=not reasons,
        finite=finite,
        exact_l0=exact_l0,
        topk_valid=bool(topk_valid),
        enough_tokens=enough_tokens,
        loss_ratio=loss_ratio,
        loss_improved=loss_improved,
        live_fraction=live_fraction,
        live_features_ok=live_features_ok,
        reasons=tuple(reasons),
    )


def train(
    config: TrainingConfig,
    batches: Iterable[Any],
    mean: Any,
    *,
    state: TrainState | None = None,
    checkpoint_dir: str | Path | None = None,
    stop_after_tokens: int | None = None,
    metric_callback: Callable[[TrainingMetric], None] | None = None,
) -> TrainingResult:
    """Train one A/B/F run with exact four-microbatch accumulation.

    ``batches`` must deterministically begin at token zero even on resume; the
    saved scientific token count is used to skip to the next update boundary.
    A non-final ``stop_after_tokens`` is rounded up to an effective-batch
    boundary so resuming produces the same updates as an uninterrupted run.
    """

    if state is None:
        state = init_train_state(jax.random.PRNGKey(config.seed), config.sae)
    start_tokens = _as_int(state.tokens_seen)
    if start_tokens > config.total_tokens:
        raise ValueError("checkpoint has consumed more than configured total tokens")
    if (
        start_tokens < config.total_tokens
        and start_tokens % config.effective_batch_tokens
    ):
        raise ValueError(
            "non-final resume checkpoint is not on an effective-batch boundary"
        )

    target_tokens = config.total_tokens
    if stop_after_tokens is not None and stop_after_tokens < config.total_tokens:
        if stop_after_tokens <= start_tokens:
            raise ValueError("stop_after_tokens must exceed current progress")
        target_tokens = min(
            config.total_tokens,
            math.ceil(stop_after_tokens / config.effective_batch_tokens)
            * config.effective_batch_tokens,
        )
    take_tokens = target_tokens - start_tokens
    micro_iterator = _fixed_microbatches(
        batches,
        config,
        skip_tokens=start_tokens,
        take_tokens=take_tokens,
    )

    metrics: list[TrainingMetric] = []
    latest: Path | None = None
    final: Path | None = None
    next_checkpoint = (
        (start_tokens // config.checkpoint_every_tokens) + 1
    ) * config.checkpoint_every_tokens

    while _as_int(state.tokens_seen) < target_tokens:
        grad_sum: SAEParams | None = None
        sse = jnp.asarray(0.0, jnp.float32)
        sst = jnp.asarray(0.0, jnp.float32)
        l0_sum = jnp.asarray(0, jnp.int32)
        valid_tokens = jnp.asarray(0, jnp.int32)
        fired = jnp.zeros_like(state.fired_mask)
        topk_valid = True

        for _ in range(config.microbatches_per_update):
            try:
                batch = next(micro_iterator)
            except StopIteration as exc:  # pragma: no cover - guarded by packer
                raise RuntimeError(
                    "microbatch source ended inside an effective batch"
                ) from exc
            plan = None
            if config.sae.condition == "A":
                plan = make_grouping_plan(
                    np.sort(np.asarray(batch.expert_ids), axis=1),
                    num_experts=config.sae.num_experts,
                    capacity=config.a_capacity,
                )
            sums, grads = loss_and_grad_sums(
                state.params,
                jnp.asarray(batch.activation),
                jnp.asarray(mean, jnp.float32),
                config.sae,
                expert_ids=None
                if batch.expert_ids is None
                else jnp.asarray(batch.expert_ids),
                valid_mask=jnp.asarray(batch.valid_mask),
                grouping_plan=plan,
            )
            valid_here = _as_bool(sums.topk_valid)
            if not valid_here:
                raise RuntimeError(
                    "fewer than k strictly positive features on a scientific token"
                )
            if _as_int(sums.l0_sum) != _as_int(sums.valid_tokens) * config.sae.k:
                raise RuntimeError("TopK returned an L0 different from configured k")
            grad_sum = _tree_add(grad_sum, grads)
            sse = sse + sums.sse
            sst = sst + sums.sst
            l0_sum = l0_sum + sums.l0_sum
            valid_tokens = valid_tokens + sums.valid_tokens
            fired = fired | sums.fired_mask
            topk_valid = topk_valid and valid_here

        if grad_sum is None:  # pragma: no cover - config forbids zero microbatches
            raise RuntimeError("no gradients accumulated")
        sst_value = _as_float(sst)
        if not math.isfinite(sst_value) or sst_value <= 0:
            raise RuntimeError("non-positive or non-finite normalized-SSE denominator")
        normalized_grads = jax.tree_util.tree_map(lambda g: g / sst, grad_sum)
        loss = sse / sst
        l0 = l0_sum.astype(jnp.float32) / valid_tokens.astype(jnp.float32)
        state, step_metric = apply_gradients(
            state,
            normalized_grads,
            config.sae,
            loss=loss,
            valid_tokens=valid_tokens,
            fired_mask=fired,
            l0=l0,
            sse=sse,
            sst=sst,
            topk_valid=jnp.asarray(topk_valid),
        )
        metric = TrainingMetric(
            step=_as_int(state.step),
            tokens_seen=_as_int(state.tokens_seen),
            loss=_as_float(step_metric.loss),
            sse=_as_float(step_metric.sse),
            sst=_as_float(step_metric.sst),
            l0=_as_float(step_metric.l0),
            valid_tokens=_as_int(step_metric.valid_tokens),
            topk_valid=_as_bool(step_metric.topk_valid),
            grad_norm=_as_float(step_metric.grad_norm),
            clip_coefficient=_as_float(step_metric.clip_coefficient),
            learning_rate=_as_float(step_metric.learning_rate),
            live_fraction=_as_float(step_metric.live_fraction),
        )
        if not all(
            math.isfinite(value)
            for value in (metric.loss, metric.sse, metric.sst, metric.grad_norm)
        ):
            raise RuntimeError("non-finite training metric; run halted")
        metrics.append(metric)
        if metric_callback is not None:
            metric_callback(metric)

        if checkpoint_dir is not None and metric.tokens_seen >= next_checkpoint:
            latest = save_checkpoint(
                checkpoint_path(checkpoint_dir),
                state,
                config,
                metadata={"last_metric": dataclasses.asdict(metric)},
            )
            while next_checkpoint <= metric.tokens_seen:
                next_checkpoint += config.checkpoint_every_tokens

    if checkpoint_dir is not None:
        latest = save_checkpoint(
            checkpoint_path(checkpoint_dir),
            state,
            config,
            metadata={
                "last_metric": dataclasses.asdict(metrics[-1]) if metrics else None
            },
        )
        if _as_int(state.tokens_seen) == config.total_tokens:
            final = save_checkpoint(
                checkpoint_path(checkpoint_dir, final=True),
                state,
                config,
                metadata={
                    "last_metric": dataclasses.asdict(metrics[-1]) if metrics else None
                },
            )

    health = None
    if _as_int(state.tokens_seen) >= config.health_gate_tokens:
        last_l0 = metrics[-1].l0 if metrics else float(config.sae.k)
        all_topk_valid = all(metric.topk_valid for metric in metrics)
        health = check_seed_health(state, config, l0=last_l0, topk_valid=all_topk_valid)
    return TrainingResult(state, tuple(metrics), health, latest, final)


def evaluate(
    params: SAEParams,
    config: SAEConfig | TrainingConfig,
    batches: Iterable[Any],
    mean: Any,
    *,
    max_tokens: int | None = None,
    expected_sst: float | None = None,
    a_capacity: int | None = None,
) -> EvaluationResult:
    """Accumulate common held-out SSE/SST in host float64."""

    sae = config.sae if isinstance(config, TrainingConfig) else config
    if isinstance(config, TrainingConfig) and a_capacity is None:
        a_capacity = config.a_capacity
    remaining = None if max_tokens is None else int(max_tokens)
    if remaining is not None and remaining <= 0:
        raise ValueError("max_tokens must be positive")

    sse_total = 0.0
    computed_sst = 0.0
    tokens = 0
    l0_sum = 0
    fired = np.zeros((sae.total_features,), dtype=bool)
    all_valid = True
    for raw_batch in batches:
        activation, expert_ids = _scientific_rows(raw_batch, sae)
        if remaining is not None:
            activation = activation[:remaining]
            if expert_ids is not None:
                expert_ids = expert_ids[:remaining]
        if not activation.shape[0]:
            if remaining == 0:
                break
            continue
        valid = np.ones((activation.shape[0],), dtype=bool)
        plan = None
        if sae.condition == "A":
            plan = make_grouping_plan(
                np.sort(expert_ids, axis=1),
                num_experts=sae.num_experts,
                capacity=a_capacity,
            )
        _, sums = normalized_sse_loss(
            params,
            jnp.asarray(activation),
            jnp.asarray(mean, jnp.float32),
            sae,
            expert_ids=None if expert_ids is None else jnp.asarray(expert_ids),
            valid_mask=jnp.asarray(valid),
            grouping_plan=plan,
        )
        valid_topk = _as_bool(sums.topk_valid)
        if not valid_topk:
            raise RuntimeError(
                "fewer than k strictly positive features during evaluation"
            )
        batch_tokens = _as_int(sums.valid_tokens)
        batch_l0 = _as_int(sums.l0_sum)
        if batch_l0 != batch_tokens * sae.k:
            raise RuntimeError("evaluation L0 differs from configured k")
        sse_total += _as_float(sums.sse)
        computed_sst += _as_float(sums.sst)
        tokens += batch_tokens
        l0_sum += batch_l0
        fired |= np.asarray(jax.device_get(sums.fired_mask), dtype=bool)
        all_valid = all_valid and valid_topk
        if remaining is not None:
            remaining -= batch_tokens
            if remaining == 0:
                break

    if tokens == 0:
        raise RuntimeError("evaluation consumed zero scientific tokens")
    if remaining not in (None, 0):
        raise RuntimeError(f"evaluation source ended {remaining} tokens early")
    denominator = computed_sst if expected_sst is None else float(expected_sst)
    if not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("evaluation SST must be finite and positive")
    nmse = sse_total / denominator
    return EvaluationResult(
        condition=sae.condition,
        tokens=tokens,
        sse=sse_total,
        sst=denominator,
        computed_sst=computed_sst,
        nmse=nmse,
        explained_variance=1.0 - nmse,
        l0=l0_sum / tokens,
        dead_rate=1.0 - float(np.mean(fired)),
        topk_valid=all_valid,
    )


__all__ = [
    "ActivationBatch",
    "CHECKPOINT_VERSION",
    "EvaluationResult",
    "HealthGateResult",
    "TrainingConfig",
    "TrainingMetric",
    "TrainingResult",
    "check_seed_health",
    "checkpoint_path",
    "evaluate",
    "load_checkpoint",
    "planned_token_mask",
    "save_checkpoint",
    "train",
]
