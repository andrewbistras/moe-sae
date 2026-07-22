"""TPU numerical callbacks for the isolated full-shape SAE smoke protocol.

This module is the numerical boundary for :mod:`moe_sae.sae_smoke`.  Importing
it does not import JAX, Orbax, the distributed trainer, or Google Cloud.  The
default runtime loads those dependencies lazily; tests can replace every
side-effecting operation with plain Python callbacks.

The adapter runs the *production* 50M-token ``TrainingConfig`` only through
its first 16,384-token update.  It never supplies a checkpoint control
directory to the trainer.  The resulting full state is saved directly to the
attempt's immutable ``sae-smoke`` payload URI, independently inventoried in
GCS, and restored into a freshly initialized sharded template with a fresh
Orbax checkpointer.

All methods are collective in production: every JAX process must call them in
the same order.  The surrounding POSIX smoke transaction remains a separate
coordinator concern.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from moe_sae.sae_smoke import (
    FinalizedCheckpoint,
    SMOKE_NAMESPACE,
    SAESmokeIntegrityError,
    SmokeAttemptIdentity,
    SmokeCampaignIdentity,
    SmokeCheckpointExpectation,
    SmokeNumericalCallbacks,
    SmokeRestoreResult,
    SmokeUpdateResult,
)


__all__ = [
    "GCSObjectVersion",
    "OrbaxPayloadInspection",
    "RestoredPayload",
    "SAESmokeTPUAdapter",
    "StateProgress",
    "TPUSmokeRuntime",
    "checkpoint_inventory_sha256",
    "hash_addressable_train_state",
    "require_identical_host_digests",
]


TPU_CHECKPOINT_SCHEMA = "gemma4-sae-smoke-tpu-checkpoint-v1"
INVENTORY_SCHEMA = "gemma4-sae-smoke-gcs-inventory-v1"
STATE_HASH_SCHEMA = "gemma4-sae-addressable-train-state-v1"

PRODUCTION_TOTAL_TOKENS = 50_000_000
PRODUCTION_EFFECTIVE_BATCH_TOKENS = 16_384
PRODUCTION_MICROBATCH_TOKENS = 4_096
PRODUCTION_UPDATES = 3_052
PRODUCTION_FINAL_PADDING = 3_968
PRODUCTION_CHECKPOINT_EVERY_TOKENS = 10_000_000
PRODUCTION_HEALTH_GATE_TOKENS = 5_000_000
PRODUCTION_HIDDEN_DIM = 2_816
PRODUCTION_NUM_EXPERTS = 128
PRODUCTION_TOP_ROUTED = 8
PRODUCTION_BANK_WIDTH = 1_024
PRODUCTION_TOPK_FEATURES = 64

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TRAIN_STATE_FIELDS = (
    "params",
    "optimizer",
    "step",
    "tokens_seen",
    "fired_mask",
    "loss_ema",
    "initial_loss_ema",
    "ema_updates",
)
_PARAM_FIELDS = ("encoder", "decoder", "bias")
_OPTIMIZER_FIELDS = ("m", "v", "step")
_TRAIN_STATE_LEAF_PATHS = (
    ("params", "encoder"),
    ("params", "decoder"),
    ("params", "bias"),
    ("optimizer", "m", "encoder"),
    ("optimizer", "m", "decoder"),
    ("optimizer", "m", "bias"),
    ("optimizer", "v", "encoder"),
    ("optimizer", "v", "decoder"),
    ("optimizer", "v", "bias"),
    ("optimizer", "step"),
    ("step",),
    ("tokens_seen",),
    ("fired_mask",),
    ("loss_ema",),
    ("initial_loss_ema",),
    ("ema_updates",),
)


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("TPU smoke metadata must be canonical JSON") from error
    return encoded.encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SAESmokeIntegrityError(f"{label} must be a lowercase SHA-256")
    return value


def _require_exact_int(value: Any, expected: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise SAESmokeIntegrityError(f"{label} must equal {expected}")


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _gcs_parts(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise SAESmokeIntegrityError("TPU smoke checkpoint URI must be gs://")
    remainder = uri[5:]
    bucket, separator, prefix = remainder.partition("/")
    if not bucket or not separator or not prefix:
        raise SAESmokeIntegrityError("GCS checkpoint URI must include an object prefix")
    components = prefix.split("/")
    if any(not component or component in {".", ".."} for component in components):
        raise SAESmokeIntegrityError("GCS checkpoint URI contains an unsafe component")
    if SMOKE_NAMESPACE not in components:
        raise SAESmokeIntegrityError("checkpoint URI is outside the sae-smoke namespace")
    return bucket, prefix


def _require_smoke_cache_path(value: str) -> None:
    path = Path(value)
    if SMOKE_NAMESPACE not in path.parts or path.name != "cache":
        raise SAESmokeIntegrityError("attempt cache path is outside its smoke namespace")


@dataclass(frozen=True, slots=True)
class GCSObjectVersion:
    """Immutable GCS identity for one live Orbax object generation."""

    name: str
    generation: int
    size: int
    crc32c: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name.startswith("/"):
            raise ValueError("GCS object name is invalid")
        _require_positive_int(self.generation, "GCS object generation")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("GCS object size must be a nonnegative integer")
        if not isinstance(self.crc32c, str) or not self.crc32c:
            raise ValueError("GCS object CRC32C is required")

    def to_mapping(self, *, prefix: str) -> dict[str, Any]:
        relative = self.name[len(prefix) :].lstrip("/")
        if self.name != prefix and not self.name.startswith(prefix + "/"):
            raise SAESmokeIntegrityError(
                "GCS inventory contains an object outside the checkpoint prefix"
            )
        return {
            "name": relative,
            "generation": self.generation,
            "size": self.size,
            "crc32c": self.crc32c,
        }


@dataclass(frozen=True, slots=True)
class OrbaxPayloadInspection:
    """Fresh Orbax metadata plus an independent live GCS object listing."""

    checkpoint_uri: str
    finalized: bool
    checkpoint_header: Mapping[str, Any]
    objects: tuple[GCSObjectVersion, ...]


@dataclass(frozen=True, slots=True)
class RestoredPayload:
    """State and bound metadata returned by a fresh direct-payload restore."""

    state: object
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StateProgress:
    step: int
    tokens_seen: int
    optimizer_step: int
    ema_updates: int


@dataclass(frozen=True, slots=True)
class TPUSmokeRuntime:
    """Injectable numerical/cloud runtime; defaults import dependencies lazily."""

    run_training: Callable[..., object]
    save_checkpoint: Callable[..., None]
    inspect_payload: Callable[[str], OrbaxPayloadInspection]
    restore_fresh: Callable[..., RestoredPayload]
    state_digest: Callable[[object], str]
    state_progress: Callable[[object], StateProgress]
    consensus_digest: Callable[[str, str], str]

    @classmethod
    def production(cls) -> TPUSmokeRuntime:
        return cls(
            run_training=_default_run_training,
            save_checkpoint=_default_save_checkpoint,
            inspect_payload=_default_inspect_payload,
            restore_fresh=_default_restore_fresh,
            state_digest=_default_state_digest,
            state_progress=_default_state_progress,
            consensus_digest=_default_consensus_digest,
        )


def checkpoint_inventory_sha256(
    checkpoint_uri: str,
    objects: Sequence[GCSObjectVersion],
) -> str:
    """Digest a sorted generation/size/CRC inventory for one GCS payload."""

    bucket, prefix = _gcs_parts(checkpoint_uri)
    if not objects:
        raise SAESmokeIntegrityError("finalized checkpoint has no GCS objects")
    records = [item.to_mapping(prefix=prefix) for item in objects]
    records.sort(key=lambda item: (str(item["name"]), int(item["generation"])))
    names = [str(item["name"]) for item in records]
    if len(names) != len(set(names)):
        raise SAESmokeIntegrityError(
            "finalized checkpoint inventory has duplicate live object names"
        )
    return _canonical_sha256(
        {
            "schema_version": INVENTORY_SCHEMA,
            "checkpoint_uri": checkpoint_uri,
            "bucket": bucket,
            "prefix": prefix,
            "objects": records,
        }
    )


def require_identical_host_digests(
    local_digest: str,
    host_digests: Sequence[str],
    *,
    label: str,
) -> str:
    """Fail unless every data-replicated host reports the same digest."""

    expected = _require_sha256(local_digest, f"local {label} digest")
    if not host_digests:
        raise SAESmokeIntegrityError(f"{label} digest collective returned no hosts")
    normalized = tuple(
        _require_sha256(value, f"host {label} digest") for value in host_digests
    )
    if any(value != expected for value in normalized):
        raise SAESmokeIntegrityError(
            f"data-replicated hosts disagree on the {label} digest"
        )
    return expected


def _fields(value: Any, expected: tuple[str, ...], label: str) -> None:
    observed = getattr(value, "_fields", None)
    if observed is not None and tuple(observed) != expected:
        raise SAESmokeIntegrityError(f"{label} fields do not match TrainState schema")
    for name in expected:
        if not hasattr(value, name):
            raise SAESmokeIntegrityError(f"{label} is missing {name}")


def _require_train_state_structure(state: Any) -> None:
    _fields(state, _TRAIN_STATE_FIELDS, "TrainState")
    _fields(state.params, _PARAM_FIELDS, "TrainState.params")
    _fields(state.optimizer, _OPTIMIZER_FIELDS, "TrainState.optimizer")
    _fields(state.optimizer.m, _PARAM_FIELDS, "TrainState.optimizer.m")
    _fields(state.optimizer.v, _PARAM_FIELDS, "TrainState.optimizer.v")


def _leaf(state: Any, path: tuple[str, ...]) -> Any:
    value = state
    for name in path:
        value = getattr(value, name)
    return value


def _normalized_index(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalized_index(item) for item in value]
    if isinstance(value, slice):
        return {
            "slice": [
                None if value.start is None else int(value.start),
                None if value.stop is None else int(value.stop),
                None if value.step is None else int(value.step),
            ]
        }
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    raise SAESmokeIntegrityError(f"unsupported addressable shard index: {value!r}")


def _canonical_array_bytes(
    value: Any, to_numpy: Callable[[Any], np.ndarray]
) -> tuple[np.ndarray, bytes]:
    array = np.asarray(to_numpy(value))
    if array.dtype.hasobject:
        raise SAESmokeIntegrityError("TrainState shard has object dtype")
    contiguous = np.ascontiguousarray(array)
    byteorder = contiguous.dtype.byteorder
    if byteorder == ">" or (byteorder == "=" and not np.little_endian):
        little_dtype = contiguous.dtype.newbyteorder("<")
        contiguous = contiguous.astype(little_dtype, copy=False)
    return contiguous, contiguous.tobytes(order="C")


def _scalar_value(value: Any, scalar_reader: Callable[[Any], int], label: str) -> int:
    try:
        result = scalar_reader(value)
    except Exception as error:
        raise SAESmokeIntegrityError(f"cannot read TrainState {label}") from error
    if isinstance(result, bool) or not isinstance(result, (int, np.integer)):
        raise SAESmokeIntegrityError(f"TrainState {label} is not an integer scalar")
    return int(result)


def hash_addressable_train_state(
    state: object,
    *,
    to_numpy: Callable[[Any], np.ndarray] | None = None,
    scalar_reader: Callable[[Any], int] | None = None,
) -> str:
    """Hash every ordered addressable shard and explicit progress scalar.

    Replica copies with the same global index are hashed once after proving
    byte equality.  This makes the local inventory identical on every data
    replica while retaining all feature-sharded values.
    """

    _require_train_state_structure(state)
    convert = to_numpy or _default_to_numpy
    read_scalar = scalar_reader or _default_scalar_reader
    leaves: list[dict[str, Any]] = []
    for path in _TRAIN_STATE_LEAF_PATHS:
        array = _leaf(state, path)
        shards = getattr(array, "addressable_shards", None)
        if shards is None:
            raise SAESmokeIntegrityError(
                f"TrainState leaf {'.'.join(path)} has no addressable shards"
            )
        by_index: dict[bytes, dict[str, Any]] = {}
        for shard in shards:
            index = _normalized_index(getattr(shard, "index", None))
            key = _canonical_bytes(index)
            host_array, raw = _canonical_array_bytes(getattr(shard, "data"), convert)
            record = {
                "index": index,
                "shape": list(host_array.shape),
                "dtype": str(host_array.dtype),
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            previous = by_index.get(key)
            if previous is not None and previous != record:
                raise SAESmokeIntegrityError(
                    f"replicas disagree within TrainState leaf {'.'.join(path)}"
                )
            by_index[key] = record
        if not by_index:
            raise SAESmokeIntegrityError(
                f"TrainState leaf {'.'.join(path)} has no local shard data"
            )
        leaves.append(
            {
                "path": ".".join(path),
                "global_shape": list(getattr(array, "shape", ())),
                "global_dtype": str(getattr(array, "dtype", "")),
                "shards": [by_index[key] for key in sorted(by_index)],
            }
        )
    progress = {
        "step": _scalar_value(state.step, read_scalar, "step"),
        "tokens_seen": _scalar_value(
            state.tokens_seen, read_scalar, "tokens_seen"
        ),
        "optimizer_step": _scalar_value(
            state.optimizer.step, read_scalar, "optimizer.step"
        ),
        "ema_updates": _scalar_value(
            state.ema_updates, read_scalar, "ema_updates"
        ),
    }
    return _canonical_sha256(
        {
            "schema_version": STATE_HASH_SCHEMA,
            "leaf_order": [".".join(path) for path in _TRAIN_STATE_LEAF_PATHS],
            "leaves": leaves,
            "progress": progress,
        }
    )


def _training_config_mapping(training_config: object) -> dict[str, Any]:
    if not is_dataclass(training_config) or isinstance(training_config, type):
        raise SAESmokeIntegrityError("production TrainingConfig must be a dataclass")
    value = asdict(training_config)
    if not isinstance(value, dict):
        raise SAESmokeIntegrityError("TrainingConfig identity is not an object")
    _canonical_bytes(value)
    return value


def _training_config_sha256(training_config: object) -> str:
    # Match distributed._checkpoint_header exactly.
    encoded = json.dumps(
        _training_config_mapping(training_config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mesh_shape(mesh: object) -> dict[str, int]:
    shape = getattr(mesh, "shape", None)
    if not isinstance(shape, Mapping):
        raise SAESmokeIntegrityError("mesh does not expose a shape mapping")
    result: dict[str, int] = {}
    for name, size in shape.items():
        if not isinstance(name, str):
            raise SAESmokeIntegrityError("mesh axis name is not a string")
        result[name] = int(size)
    return result


def _dtype_name(value: Any) -> str:
    return str(getattr(value, "dtype", "")).lower()


def _validate_state_shapes(state: object, training_config: object) -> None:
    _require_train_state_structure(state)
    sae = training_config.sae
    features = int(sae.total_features)
    hidden = int(sae.hidden_dim)
    expected_shapes = {
        ("params", "encoder"): (features, hidden),
        ("params", "decoder"): (hidden, features),
        ("params", "bias"): (features,),
        ("optimizer", "m", "encoder"): (features, hidden),
        ("optimizer", "m", "decoder"): (hidden, features),
        ("optimizer", "m", "bias"): (features,),
        ("optimizer", "v", "encoder"): (features, hidden),
        ("optimizer", "v", "decoder"): (hidden, features),
        ("optimizer", "v", "bias"): (features,),
        ("optimizer", "step"): (),
        ("step",): (),
        ("tokens_seen",): (),
        ("fired_mask",): (features,),
        ("loss_ema",): (),
        ("initial_loss_ema",): (),
        ("ema_updates",): (),
    }
    for path, expected in expected_shapes.items():
        observed = tuple(getattr(_leaf(state, path), "shape", ()))
        if observed != expected:
            raise SAESmokeIntegrityError(
                f"TrainState {'.'.join(path)} shape {observed} != {expected}"
            )
    for path in (
        ("params", "encoder"),
        ("params", "decoder"),
        ("params", "bias"),
    ):
        if _dtype_name(_leaf(state, path)) != "bfloat16":
            raise SAESmokeIntegrityError(
                f"TrainState {'.'.join(path)} must be bfloat16"
            )
    for group in ("m", "v"):
        for name in _PARAM_FIELDS:
            if _dtype_name(_leaf(state, ("optimizer", group, name))) != "float32":
                raise SAESmokeIntegrityError(
                    f"TrainState optimizer.{group}.{name} must be float32"
                )


def _validate_production_training_config(
    training_config: object,
    attempt: SmokeAttemptIdentity,
    smoke_a_capacity: int | None,
) -> None:
    _training_config_mapping(training_config)
    sae = getattr(training_config, "sae", None)
    if sae is None:
        raise SAESmokeIntegrityError("TrainingConfig has no SAE configuration")
    exact_training = {
        "total_tokens": PRODUCTION_TOTAL_TOKENS,
        "effective_batch_tokens": PRODUCTION_EFFECTIVE_BATCH_TOKENS,
        "microbatch_tokens": PRODUCTION_MICROBATCH_TOKENS,
        "checkpoint_every_tokens": PRODUCTION_CHECKPOINT_EVERY_TOKENS,
        "health_gate_tokens": PRODUCTION_HEALTH_GATE_TOKENS,
    }
    for name, expected in exact_training.items():
        _require_exact_int(getattr(training_config, name, None), expected, name)
    _require_exact_int(
        getattr(training_config, "updates", None),
        PRODUCTION_UPDATES,
        "TrainingConfig.updates",
    )
    _require_exact_int(
        getattr(training_config, "final_padding", None),
        PRODUCTION_FINAL_PADDING,
        "TrainingConfig.final_padding",
    )
    _require_exact_int(
        getattr(training_config, "microbatches_per_update", None),
        4,
        "TrainingConfig.microbatches_per_update",
    )
    if getattr(training_config, "seed", None) not in {1, 2, 3}:
        raise SAESmokeIntegrityError("production smoke seed must be one of 1, 2, 3")
    if getattr(training_config, "live_feature_fraction", None) != 0.95:
        raise SAESmokeIntegrityError(
            "production TrainingConfig live_feature_fraction must equal 0.95"
        )
    if getattr(training_config, "loss_ratio_threshold", None) != 0.80:
        raise SAESmokeIntegrityError(
            "production TrainingConfig loss_ratio_threshold must equal 0.80"
        )
    exact_sae = {
        "condition": attempt.condition,
        "hidden_dim": PRODUCTION_HIDDEN_DIM,
        "num_experts": PRODUCTION_NUM_EXPERTS,
        "top_routed": PRODUCTION_TOP_ROUTED,
        "bank_width": PRODUCTION_BANK_WIDTH,
        "k": PRODUCTION_TOPK_FEATURES,
        "parameter_dtype": "bfloat16",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "grad_clip": 1.0,
        "peak_lr": 3e-4,
        "min_lr": 3e-5,
        "warmup_steps": 250,
        "total_steps": PRODUCTION_UPDATES,
    }
    for name, expected in exact_sae.items():
        if getattr(sae, name, None) != expected:
            raise SAESmokeIntegrityError(
                f"production SAEConfig {name} must equal {expected!r}"
            )
    expected_total_features = (
        PRODUCTION_TOP_ROUTED * PRODUCTION_BANK_WIDTH
        if attempt.condition == "B"
        else PRODUCTION_NUM_EXPERTS * PRODUCTION_BANK_WIDTH
    )
    expected_candidates = (
        PRODUCTION_TOP_ROUTED * PRODUCTION_BANK_WIDTH
        if attempt.condition in {"A", "B"}
        else PRODUCTION_NUM_EXPERTS * PRODUCTION_BANK_WIDTH
    )
    expected_parameters = expected_total_features * (2 * PRODUCTION_HIDDEN_DIM + 1)
    if (
        attempt.condition_shape.hidden_dim != PRODUCTION_HIDDEN_DIM
        or attempt.condition_shape.total_features != expected_total_features
        or attempt.condition_shape.candidates_per_token != expected_candidates
        or attempt.condition_shape.topk_features != PRODUCTION_TOPK_FEATURES
        or attempt.condition_shape.parameter_count != expected_parameters
    ):
        raise SAESmokeIntegrityError(
            "attempt does not describe the production Gemma 4 SAE full shape"
        )
    if int(getattr(sae, "total_features", -1)) != attempt.condition_shape.total_features:
        raise SAESmokeIntegrityError("production SAE dictionary width mismatch")
    if int(getattr(sae, "candidates_per_token", -1)) != (
        attempt.condition_shape.candidates_per_token
    ):
        raise SAESmokeIntegrityError("production SAE candidate width mismatch")
    if int(getattr(sae, "parameter_count", -1)) != attempt.condition_shape.parameter_count:
        raise SAESmokeIntegrityError("production SAE parameter count mismatch")
    if attempt.expected_final_tokens_seen != PRODUCTION_EFFECTIVE_BATCH_TOKENS:
        raise SAESmokeIntegrityError("smoke attempt is not one production batch")
    configured_capacity = getattr(training_config, "a_capacity", None)
    if attempt.condition == "A":
        capacity = _require_positive_int(smoke_a_capacity, "smoke A capacity")
        if capacity % 8:
            raise SAESmokeIntegrityError(
                "one-batch smoke A capacity must use the production multiple-of-8 rounding"
            )
        if configured_capacity != capacity:
            raise SAESmokeIntegrityError(
                "TrainingConfig.a_capacity must equal the explicit one-batch smoke capacity"
            )
    elif smoke_a_capacity is not None or configured_capacity is not None:
        raise SAESmokeIntegrityError("B/F smoke attempts cannot have an A capacity")


def _validate_distributed_config(distributed_config: object, mesh: object) -> None:
    if getattr(distributed_config, "data_axis", None) != "data":
        raise SAESmokeIntegrityError("smoke distributed data axis must be 'data'")
    if getattr(distributed_config, "feature_axis", None) != "feature":
        raise SAESmokeIntegrityError("smoke distributed feature axis must be 'feature'")
    _require_exact_int(
        getattr(distributed_config, "microbatches_per_update", None),
        4,
        "DistributedConfig.microbatches_per_update",
    )
    shape = _mesh_shape(mesh)
    if "data" not in shape or "feature" not in shape:
        raise SAESmokeIntegrityError("smoke mesh must contain data and feature axes")
    if shape["data"] <= 0 or shape["feature"] <= 0:
        raise SAESmokeIntegrityError("smoke mesh axes must be positive")


def _checkpoint_metadata(
    campaign: SmokeCampaignIdentity,
    attempt: SmokeAttemptIdentity,
    expectation: SmokeCheckpointExpectation,
) -> dict[str, Any]:
    return {
        "smoke_schema_version": TPU_CHECKPOINT_SCHEMA,
        "smoke_campaign_identity": campaign.to_mapping(),
        "smoke_attempt_identity": attempt.to_mapping(),
        "smoke_attempt_sha256": attempt.sha256,
        "smoke_campaign_sha256": attempt.campaign_sha256,
        "smoke_condition_shape_sha256": attempt.condition_shape.sha256,
        "smoke_state_sha256": expectation.state_sha256,
        "smoke_optimizer_updates": expectation.optimizer_updates,
        "checkpoint_payload_uri": expectation.checkpoint_uri,
        "tokens_seen": expectation.tokens_seen,
        "steps": expectation.step,
    }


@dataclass(slots=True)
class _AttemptContext:
    attempt: SmokeAttemptIdentity
    state: object
    state_sha256: str
    saved: bool = False
    checkpoint_sha256: str | None = None


class SAESmokeTPUAdapter:
    """Collective production numerical implementation of smoke callbacks."""

    def __init__(
        self,
        *,
        campaign_identity: SmokeCampaignIdentity,
        training_config: object,
        batches: object,
        mean: object,
        mesh: object,
        distributed_config: object,
        smoke_a_capacity: int | None = None,
        process_local_inputs: bool = True,
        runtime: TPUSmokeRuntime | None = None,
    ) -> None:
        if not isinstance(campaign_identity, SmokeCampaignIdentity):
            raise TypeError("campaign_identity must be a SmokeCampaignIdentity")
        self.campaign_identity = campaign_identity
        self.training_config = training_config
        self.batches = batches
        self.mean = mean
        self.mesh = mesh
        self.distributed_config = distributed_config
        self.smoke_a_capacity = smoke_a_capacity
        self.process_local_inputs = bool(process_local_inputs)
        self.runtime = runtime or TPUSmokeRuntime.production()
        self._contexts: dict[str, _AttemptContext] = {}

    def callbacks(self) -> SmokeNumericalCallbacks:
        return SmokeNumericalCallbacks(
            update_once=self.update_once,
            save_orbax=self.save_orbax,
            inspect_finalized_checkpoint=self.inspect_finalized_checkpoint,
            restore_orbax=self.restore_orbax,
        )

    def _consensus(self, digest: str, label: str) -> str:
        _require_sha256(digest, label)
        agreed = self.runtime.consensus_digest(digest, label)
        return _require_sha256(agreed, f"agreed {label}")

    def _context(self, uri: str) -> _AttemptContext:
        try:
            return self._contexts[uri]
        except KeyError as error:
            raise SAESmokeIntegrityError(
                "checkpoint callback has no matching optimizer update"
            ) from error

    def update_once(self, attempt: SmokeAttemptIdentity) -> SmokeUpdateResult:
        if not isinstance(attempt, SmokeAttemptIdentity):
            raise TypeError("attempt must be a SmokeAttemptIdentity")
        _gcs_parts(attempt.checkpoint_uri)
        _require_smoke_cache_path(attempt.cache_path)
        if attempt.campaign_sha256 != self.campaign_identity.sha256:
            raise SAESmokeIntegrityError("attempt belongs to another smoke campaign")
        if attempt.full_shape_sha256 != self.campaign_identity.full_shape.sha256:
            raise SAESmokeIntegrityError("attempt full shape differs from its campaign")
        _validate_production_training_config(
            self.training_config, attempt, self.smoke_a_capacity
        )
        _validate_distributed_config(self.distributed_config, self.mesh)
        if attempt.checkpoint_uri in self._contexts:
            raise SAESmokeIntegrityError("optimizer update already ran for this attempt")
        result = self.runtime.run_training(
            self.training_config,
            self.batches,
            self.mean,
            self.mesh,
            state=None,
            distributed=self.distributed_config,
            a_capacity=self.smoke_a_capacity,
            checkpoint_dir=None,
            checkpoint_payload_root=None,
            stop_after_tokens=PRODUCTION_EFFECTIVE_BATCH_TOKENS,
            require_health_pass=False,
            metric_callback=None,
            process_local_inputs=self.process_local_inputs,
            prior_optimizer_wall_seconds=0.0,
            prior_compile_seconds=0.0,
            prior_checkpoint_seconds=0.0,
            process_local_source_offset=0,
            checkpoint_identity=None,
        )
        state = getattr(result, "state", None)
        if state is None:
            raise SAESmokeIntegrityError("distributed smoke returned no TrainState")
        metrics = tuple(getattr(result, "metrics", ()))
        if len(metrics) != 1:
            raise SAESmokeIntegrityError("distributed smoke did not run exactly one update")
        if getattr(result, "latest_checkpoint", None) is not None or (
            getattr(result, "final_checkpoint", None) is not None
        ):
            raise SAESmokeIntegrityError("smoke training created a production checkpoint")
        timing = getattr(result, "timing", None)
        if timing is None:
            raise SAESmokeIntegrityError("distributed smoke returned no timing record")
        _require_exact_int(getattr(timing, "steps", None), 1, "smoke timing steps")
        _require_exact_int(
            getattr(timing, "tokens", None),
            PRODUCTION_EFFECTIVE_BATCH_TOKENS,
            "smoke timing tokens",
        )
        _validate_state_shapes(state, self.training_config)
        progress = self.runtime.state_progress(state)
        if not isinstance(progress, StateProgress):
            raise SAESmokeIntegrityError("state_progress returned the wrong type")
        if progress != StateProgress(1, PRODUCTION_EFFECTIVE_BATCH_TOKENS, 1, 1):
            raise SAESmokeIntegrityError(
                "updated TrainState progress is not exactly step 1 / 16,384 tokens"
            )
        state_sha256 = self._consensus(
            self.runtime.state_digest(state), "updated TrainState"
        )
        self._contexts[attempt.checkpoint_uri] = _AttemptContext(
            attempt=attempt,
            state=state,
            state_sha256=state_sha256,
        )
        return SmokeUpdateResult(
            state=state,
            initial_step=0,
            final_step=progress.step,
            initial_tokens_seen=0,
            final_tokens_seen=progress.tokens_seen,
            optimizer_updates=progress.optimizer_step,
            attempt_sha256=attempt.sha256,
            condition_shape_sha256=attempt.condition_shape.sha256,
            state_sha256=state_sha256,
        )

    def _validate_expectation(
        self,
        expectation: SmokeCheckpointExpectation,
        context: _AttemptContext,
    ) -> None:
        expected = SmokeCheckpointExpectation(
            attempt_sha256=context.attempt.sha256,
            condition_shape_sha256=context.attempt.condition_shape.sha256,
            checkpoint_uri=context.attempt.checkpoint_uri,
            step=1,
            tokens_seen=PRODUCTION_EFFECTIVE_BATCH_TOKENS,
            optimizer_updates=1,
            state_sha256=context.state_sha256,
        )
        if expectation != expected:
            raise SAESmokeIntegrityError("checkpoint expectation does not match update")

    def save_orbax(
        self,
        state: object,
        expectation: SmokeCheckpointExpectation,
    ) -> None:
        context = self._context(expectation.checkpoint_uri)
        self._validate_expectation(expectation, context)
        if state is not context.state:
            raise SAESmokeIntegrityError("save did not receive the updated TrainState")
        if context.saved:
            raise SAESmokeIntegrityError("Orbax smoke payload was already saved")
        metadata = _checkpoint_metadata(
            self.campaign_identity, context.attempt, expectation
        )
        self.runtime.save_checkpoint(
            context.attempt.checkpoint_uri,
            state,
            self.mesh,
            self.training_config,
            metadata=metadata,
        )
        context.saved = True

    def _validate_inspection(
        self,
        inspection: OrbaxPayloadInspection,
        context: _AttemptContext,
    ) -> str:
        if not isinstance(inspection, OrbaxPayloadInspection):
            raise SAESmokeIntegrityError("payload inspector returned the wrong type")
        if inspection.checkpoint_uri != context.attempt.checkpoint_uri:
            raise SAESmokeIntegrityError("payload inspector returned another URI")
        if inspection.finalized is not True:
            raise SAESmokeIntegrityError("external Orbax payload is not finalized")
        header = dict(inspection.checkpoint_header)
        expected_header_keys = {
            "checkpoint_version",
            "training_config",
            "training_config_sha256",
            "mesh_shape",
            "metadata",
        }
        if set(header) != expected_header_keys:
            raise SAESmokeIntegrityError("Orbax checkpoint header fields mismatch")
        if header["checkpoint_version"] != 1:
            raise SAESmokeIntegrityError("Orbax checkpoint version mismatch")
        if header["training_config"] != _training_config_mapping(self.training_config):
            raise SAESmokeIntegrityError("Orbax checkpoint TrainingConfig mismatch")
        if header["training_config_sha256"] != _training_config_sha256(
            self.training_config
        ):
            raise SAESmokeIntegrityError("Orbax TrainingConfig digest mismatch")
        if header["mesh_shape"] != _mesh_shape(self.mesh):
            raise SAESmokeIntegrityError("Orbax checkpoint mesh shape mismatch")
        expectation = SmokeCheckpointExpectation(
            attempt_sha256=context.attempt.sha256,
            condition_shape_sha256=context.attempt.condition_shape.sha256,
            checkpoint_uri=context.attempt.checkpoint_uri,
            step=1,
            tokens_seen=PRODUCTION_EFFECTIVE_BATCH_TOKENS,
            optimizer_updates=1,
            state_sha256=context.state_sha256,
        )
        if header["metadata"] != _checkpoint_metadata(
            self.campaign_identity, context.attempt, expectation
        ):
            raise SAESmokeIntegrityError("Orbax checkpoint smoke identity mismatch")
        local_digest = checkpoint_inventory_sha256(
            context.attempt.checkpoint_uri, inspection.objects
        )
        return self._consensus(local_digest, "GCS checkpoint inventory")

    def inspect_finalized_checkpoint(self, uri: str) -> FinalizedCheckpoint:
        context = self._context(uri)
        if not context.saved:
            raise SAESmokeIntegrityError("checkpoint inspection occurred before save")
        digest = self._validate_inspection(
            self.runtime.inspect_payload(uri), context
        )
        context.checkpoint_sha256 = digest
        return FinalizedCheckpoint(
            checkpoint_uri=uri,
            finalized=True,
            step=1,
            tokens_seen=PRODUCTION_EFFECTIVE_BATCH_TOKENS,
            optimizer_updates=1,
            attempt_sha256=context.attempt.sha256,
            condition_shape_sha256=context.attempt.condition_shape.sha256,
            state_sha256=context.state_sha256,
            checkpoint_sha256=digest,
        )

    def restore_orbax(
        self,
        uri: str,
        expectation: SmokeCheckpointExpectation,
    ) -> SmokeRestoreResult:
        context = self._context(uri)
        self._validate_expectation(expectation, context)
        if context.checkpoint_sha256 is None:
            raise SAESmokeIntegrityError("restore occurred before external inspection")
        restored = self.runtime.restore_fresh(
            uri,
            self.mesh,
            self.training_config,
            self.distributed_config,
        )
        if not isinstance(restored, RestoredPayload):
            raise SAESmokeIntegrityError("fresh restore returned the wrong type")
        expected_metadata = _checkpoint_metadata(
            self.campaign_identity, context.attempt, expectation
        )
        if dict(restored.metadata) != expected_metadata:
            raise SAESmokeIntegrityError("restored Orbax metadata identity mismatch")
        _validate_state_shapes(restored.state, self.training_config)
        progress = self.runtime.state_progress(restored.state)
        if progress != StateProgress(1, PRODUCTION_EFFECTIVE_BATCH_TOKENS, 1, 1):
            raise SAESmokeIntegrityError("restored TrainState progress mismatch")
        restored_state_sha256 = self._consensus(
            self.runtime.state_digest(restored.state), "restored TrainState"
        )
        if restored_state_sha256 != context.state_sha256:
            raise SAESmokeIntegrityError("restored full TrainState digest mismatch")
        post_restore_checkpoint_sha256 = self._validate_inspection(
            self.runtime.inspect_payload(uri), context
        )
        if post_restore_checkpoint_sha256 != context.checkpoint_sha256:
            raise SAESmokeIntegrityError(
                "GCS checkpoint inventory changed across restore"
            )
        return SmokeRestoreResult(
            state=restored.state,
            step=progress.step,
            tokens_seen=progress.tokens_seen,
            optimizer_updates=progress.optimizer_step,
            attempt_sha256=context.attempt.sha256,
            condition_shape_sha256=context.attempt.condition_shape.sha256,
            state_sha256=restored_state_sha256,
            checkpoint_sha256=post_restore_checkpoint_sha256,
        )


def _default_run_training(*args: Any, **kwargs: Any) -> object:
    from moe_sae.distributed import run_distributed_training

    return run_distributed_training(*args, **kwargs)


def _default_save_checkpoint(*args: Any, **kwargs: Any) -> None:
    from moe_sae.distributed import save_distributed_checkpoint

    save_distributed_checkpoint(*args, **kwargs)


def _orbax_finalized(ocp: Any, uri: str) -> bool:
    finalized = getattr(ocp.utils, "is_path_finalized", None)
    if finalized is None:
        finalized = getattr(ocp.utils, "is_checkpoint_finalized", None)
    if finalized is None:
        raise RuntimeError("installed Orbax has no public finalization check")
    return bool(finalized(uri))


def _default_inspect_payload(uri: str) -> OrbaxPayloadInspection:
    bucket, prefix = _gcs_parts(uri)
    try:
        import orbax.checkpoint as ocp
    except ImportError as error:  # pragma: no cover - TPU deployment dependency
        raise RuntimeError("TPU smoke inspection requires orbax-checkpoint") from error
    try:
        from google.cloud import storage
    except ImportError as error:  # pragma: no cover - TPU deployment dependency
        raise RuntimeError("TPU smoke inspection requires google-cloud-storage") from error
    finalized = _orbax_finalized(ocp, uri)
    checkpointer = ocp.StandardCheckpointer()
    try:
        metadata = checkpointer.metadata(uri).custom_metadata or {}
        header = dict(metadata)
    finally:
        checkpointer.close()
    client = storage.Client()
    objects: list[GCSObjectVersion] = []
    for blob in client.list_blobs(bucket, prefix=prefix.rstrip("/") + "/"):
        try:
            generation = int(blob.generation)
            size = int(blob.size)
        except (TypeError, ValueError) as error:
            raise SAESmokeIntegrityError(
                f"GCS object lacks generation/size: {blob.name}"
            ) from error
        objects.append(
            GCSObjectVersion(
                name=str(blob.name),
                generation=generation,
                size=size,
                crc32c=str(blob.crc32c or ""),
            )
        )
    return OrbaxPayloadInspection(
        checkpoint_uri=uri,
        finalized=finalized,
        checkpoint_header=header,
        objects=tuple(objects),
    )


def _default_restore_fresh(
    uri: str,
    mesh: object,
    training_config: object,
    distributed_config: object,
) -> RestoredPayload:
    import jax

    from moe_sae.distributed import (
        _restore_distributed_payload,
        init_distributed_state,
    )

    template = init_distributed_state(
        jax.random.PRNGKey(int(training_config.seed)),
        mesh,
        training_config.sae,
        distributed_config,
    )
    state, metadata = _restore_distributed_payload(
        uri, template, mesh, training_config
    )
    return RestoredPayload(state=state, metadata=metadata)


def _default_to_numpy(value: Any) -> np.ndarray:
    import jax

    return np.asarray(jax.device_get(value))


def _default_scalar_reader(value: Any) -> int:
    import jax

    array = np.asarray(jax.device_get(value))
    if array.shape != ():
        raise ValueError("not a scalar")
    return int(array)


def _default_state_digest(state: object) -> str:
    return hash_addressable_train_state(state)


def _default_state_progress(state: object) -> StateProgress:
    return StateProgress(
        step=_default_scalar_reader(state.step),
        tokens_seen=_default_scalar_reader(state.tokens_seen),
        optimizer_step=_default_scalar_reader(state.optimizer.step),
        ema_updates=_default_scalar_reader(state.ema_updates),
    )


def _default_consensus_digest(local_digest: str, label: str) -> str:
    import jax
    from jax.experimental import multihost_utils

    local = np.frombuffer(bytes.fromhex(_require_sha256(local_digest, label)), dtype=np.uint8)
    gathered = np.asarray(multihost_utils.process_allgather(local, tiled=False))
    try:
        rows = gathered.reshape(-1, 32)
    except ValueError as error:
        raise SAESmokeIntegrityError(
            f"{label} digest collective returned an invalid shape"
        ) from error
    if rows.shape[0] != int(jax.process_count()):
        raise SAESmokeIntegrityError(
            f"{label} digest collective returned the wrong process count"
        )
    digests = tuple(bytes(row.tolist()).hex() for row in rows)
    return require_identical_host_digests(local_digest, digests, label=label)
