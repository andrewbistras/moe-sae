"""Common shared-expert SAE and full-FFN evaluation for Experiment 2.

``Q`` is deliberately independent of the routed A/R/B/F condition.  It sees
only Gemma's normalized shared-expert branch, has one global 8,192-feature
dictionary with exact TopK-64, and is trained once per seed.

The module has four boundaries:

* a sealed v2 -> exact normalized-shared static input adapter;
* a corrected feature-sharded Q trainer with FP32 differentiation/reductions;
* a Q evaluator that streams identity-bound sufficient rows;
* an exact host oracle combining those rows with routed-condition rows through
  Gemma's routed and outer RMSNorm round points.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    SEQUENCE_LENGTH,
    bf16_bits_to_float32,
    canonical_sha256,
    float32_to_bf16_bits,
    token_identity_sha256,
    validate_contribution_manifest,
    validate_contribution_statistics,
)
from moe_sae.contribution_distributed import (
    _decode_selected_columns_default_precision,
    _merge_local_candidates,
)
from moe_sae.contribution_evaluation import (
    ContributionEvaluationEvidenceChunk,
    _addressable_data_rows,
)
from moe_sae.contribution_input import (
    PROCESS_PARTITION_STRATEGY,
    PADDING_POSITION,
    PADDING_SEQUENCE_ID,
    contiguous_process_range,
    _dataset_manifest_sha256,
    _dataset_total_tokens,
    _read_exact_range,
)
from moe_sae.contribution_recomposition import (
    RecompositionSidecar,
    decode_recomposition_sidecar,
    maxtext_rmsnorm_bf16,
)
from moe_sae.distributed import (
    DistributedBatch,
    DistributedConfig,
    _param_specs,
    batch_shardings,
    init_distributed_state,
    replicate_process_local_array,
    shard_batch,
    shard_process_local_batch,
    stack_microbatches,
    state_shardings,
)
from moe_sae.sae import (
    SAEConfig,
    SAEParams,
    StepMetrics,
    TrainState,
    apply_gradients,
    local_topk_candidates,
)

LEGACY_SHARED_Q_INPUT_SCHEMA_VERSION = "gemma4-shared-q-input-v1"
SHARED_Q_INPUT_SCHEMA_VERSION = (
    "gemma4-shared-q-input-v2-contiguous-process-lanes"
)
SHARED_Q_CENTERING_SCHEMA_VERSION = "gemma4-shared-q-centering-v1"
SHARED_Q_EVALUATION_SCHEMA_VERSION = "gemma4-shared-q-evaluation-v1"
SHARED_Q_TOTAL_FEATURES = 8_192
SHARED_Q_TOPK = 64
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EPS = 1e-12


class SharedQError(RuntimeError):
    """The common shared-Q evidence contract was violated."""


@dataclasses.dataclass(frozen=True, slots=True)
class SharedQLayout:
    """Static global-dictionary layout; production is exactly 8,192/64."""

    hidden_dim: int = HIDDEN_SIZE
    total_features: int = SHARED_Q_TOTAL_FEATURES
    k: int = SHARED_Q_TOPK

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "total_features", "k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"shared Q {name} must be a positive integer")
        if self.k > self.total_features:
            raise ValueError("shared Q k cannot exceed total_features")

    @property
    def authoritative(self) -> bool:
        return (
            self.hidden_dim == HIDDEN_SIZE
            and self.total_features == SHARED_Q_TOTAL_FEATURES
            and self.k == SHARED_Q_TOPK
        )


def shared_q_sae_config(
    layout: SharedQLayout | None = None,
    *,
    parameter_dtype: str = "bfloat16",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    grad_clip: float = 1.0,
    peak_lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup_steps: int = 250,
    total_steps: int = 3_052,
) -> SAEConfig:
    """Map Q to the proven generic global-SAE parameter representation."""

    selected = layout or SharedQLayout()
    return SAEConfig(
        condition="B",
        hidden_dim=selected.hidden_dim,
        num_experts=1,
        top_routed=1,
        bank_width=selected.total_features,
        k=selected.k,
        parameter_dtype=parameter_dtype,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=adam_eps,
        grad_clip=grad_clip,
        peak_lr=peak_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )


def _mean_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    digest = hashlib.sha256()
    digest.update(b"gemma4-shared-q-centering-mean-v1\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _manifest_and_statistics(
    dataset: Any,
    sidecar: RecompositionSidecar,
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    manifest = getattr(dataset, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise SharedQError("shared Q requires a sealed v2 manifest")
    try:
        checked_manifest = validate_contribution_manifest(manifest)
    except Exception as error:
        raise SharedQError("shared Q requires a valid sealed v2 manifest") from error
    manifest_sha = _dataset_manifest_sha256(dataset)
    if checked_manifest["manifest_sha256"] != manifest_sha:
        raise SharedQError("shared-Q dataset and manifest hashes differ")
    if (
        checked_manifest["recomposition_sidecar_sha256"]
        != sidecar.sidecar_sha256
    ):
        raise SharedQError(
            "shared Q sidecar differs from the sealed capture manifest"
        )
    total = _dataset_total_tokens(dataset)
    try:
        statistics = validate_contribution_statistics(
            checked_manifest.get("statistics"),
            expected_tokens=total,
        )
    except Exception as error:
        raise SharedQError(
            "shared Q manifest statistics are invalid"
        ) from error
    return checked_manifest, statistics, manifest_sha


def _shared_sst(statistics: Mapping[str, Any], mean: np.ndarray) -> float:
    shared = statistics["shared_normalized"]
    mean64 = np.asarray(mean, dtype=np.float64)
    total = (
        float(shared["squared_norm"])
        - 2.0
        * float(
            np.dot(
                mean64,
                np.asarray(shared["sum"], dtype=np.float64),
            )
        )
        + int(shared["count"]) * float(np.dot(mean64, mean64))
    )
    if not math.isfinite(total) or total <= 0:
        raise SharedQError(
            "sealed normalized-shared statistics imply non-positive SST"
        )
    return total


def _read_exact_shared_range(
    dataset: Any,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read Q-only tensors when the dataset exposes a selective safe reader."""

    selective = getattr(dataset, "iter_shared_token_range", None)
    if not callable(selective):
        wire = _read_exact_range(dataset, start, end)
        return wire.shared_output, wire.sequence_id, wire.position
    try:
        pieces = tuple(selective(start, end))
    except Exception as error:
        raise SharedQError(
            f"cannot selectively read shared-Q range [{start},{end})"
        ) from error
    if not pieces:
        raise SharedQError(
            f"shared-Q dataset yielded no selective rows for [{start},{end})"
        )
    try:
        shared = np.ascontiguousarray(
            np.concatenate(
                [np.asarray(piece.shared_output) for piece in pieces],
                axis=0,
            )
        )
        sequence = np.ascontiguousarray(
            np.concatenate(
                [np.asarray(piece.sequence_id) for piece in pieces],
                axis=0,
            )
        )
        position = np.ascontiguousarray(
            np.concatenate(
                [np.asarray(piece.position) for piece in pieces],
                axis=0,
            )
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise SharedQError("selective shared-Q pieces are incompatible") from error
    expected = end - start
    if (
        shared.dtype != np.uint16
        or shared.shape != (expected, HIDDEN_SIZE)
        or sequence.dtype != np.int64
        or sequence.shape != (expected,)
        or position.dtype != np.uint16
        or position.shape != (expected,)
        or not np.isfinite(bf16_bits_to_float32(shared)).all()
        or np.any(sequence < 0)
        or np.any(position >= SEQUENCE_LENGTH)
    ):
        raise SharedQError(
            "selective shared-Q range has invalid shape, dtype, or values"
        )
    keys = np.rec.fromarrays(
        (sequence, position),
        names=("sequence_id", "position"),
    )
    if np.unique(keys).size != expected:
        raise SharedQError("selective shared-Q range contains duplicate token keys")
    return shared, sequence, position


@dataclasses.dataclass(frozen=True, slots=True)
class SharedQCentering:
    """Training-capture mean shared unchanged by every Q consumer."""

    schema_version: str
    training_capture_manifest_sha256: str
    recomposition_sidecar_sha256: str
    token_count: int
    mean: np.ndarray
    mean_sha256: str
    training_sst: float
    contract_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != SHARED_Q_CENTERING_SCHEMA_VERSION:
            raise SharedQError("unknown shared-Q centering schema")
        for name in (
            "training_capture_manifest_sha256",
            "recomposition_sidecar_sha256",
            "mean_sha256",
            "contract_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise SharedQError(f"{name} must be a SHA-256")
        if (
            isinstance(self.token_count, bool)
            or not isinstance(self.token_count, int)
            or self.token_count <= 0
        ):
            raise SharedQError("shared-Q centering token_count must be positive")
        mean = np.asarray(self.mean)
        if (
            mean.shape != (HIDDEN_SIZE,)
            or mean.dtype != np.float32
            or not np.isfinite(mean).all()
        ):
            raise SharedQError(
                f"shared-Q centering mean must be finite FP32 [{HIDDEN_SIZE}]"
            )
        if _mean_sha256(mean) != self.mean_sha256:
            raise SharedQError("shared-Q centering mean hash is invalid")
        if not math.isfinite(self.training_sst) or self.training_sst <= 0:
            raise SharedQError("shared-Q training SST must be finite and positive")
        unsigned = {
            "schema_version": self.schema_version,
            "training_capture_manifest_sha256": (
                self.training_capture_manifest_sha256
            ),
            "recomposition_sidecar_sha256": self.recomposition_sidecar_sha256,
            "token_count": self.token_count,
            "mean_sha256": self.mean_sha256,
            "training_sst": self.training_sst,
        }
        if canonical_sha256(unsigned) != self.contract_sha256:
            raise SharedQError("shared-Q centering contract hash is invalid")
        mean = np.ascontiguousarray(mean)
        mean.setflags(write=False)
        object.__setattr__(self, "mean", mean)


def build_shared_q_centering(
    training_dataset: Any,
    recomposition_sidecar: Mapping[str, Any],
) -> SharedQCentering:
    """Derive Q's one common FP32 mean from sealed training statistics."""

    sidecar = decode_recomposition_sidecar(recomposition_sidecar)
    _manifest, statistics, manifest_sha = _manifest_and_statistics(
        training_dataset,
        sidecar,
    )
    normalized = statistics["shared_normalized"]
    token_count = int(normalized["count"])
    mean = np.asarray(normalized["sum"], dtype=np.float64) / token_count
    mean = np.ascontiguousarray(mean.astype(np.float32))
    mean_sha = _mean_sha256(mean)
    training_sst = _shared_sst(statistics, mean)
    unsigned = {
        "schema_version": SHARED_Q_CENTERING_SCHEMA_VERSION,
        "training_capture_manifest_sha256": manifest_sha,
        "recomposition_sidecar_sha256": sidecar.sidecar_sha256,
        "token_count": token_count,
        "mean_sha256": mean_sha,
        "training_sst": training_sst,
    }
    return SharedQCentering(
        **unsigned,
        mean=mean,
        contract_sha256=canonical_sha256(unsigned),
    )


class SharedQHostMicrobatch(NamedTuple):
    """One static process-local batch of exact normalized shared targets."""

    activation: np.ndarray
    valid_mask: np.ndarray
    sequence_id: np.ndarray
    position: np.ndarray


@dataclasses.dataclass(frozen=True, slots=True)
class SharedQInputReceipt:
    schema_version: str
    role: str
    capture_manifest_sha256: str
    centering_contract_sha256: str
    centering_parent_manifest_sha256: str | None
    recomposition_sidecar_sha256: str
    total_tokens: int
    sealed_sst: float
    global_microbatch_tokens: int
    microbatches_per_update: int
    planned_updates: int
    process_count: int
    process_index: int
    local_rows_per_microbatch: int
    receipt_sha256: str


class SharedQProcessInput:
    """Reiterable sealed v2 -> normalized-shared process-local adapter."""

    def __init__(
        self,
        dataset: Any,
        recomposition_sidecar: Mapping[str, Any],
        centering: SharedQCentering,
        *,
        role: Literal["train", "evaluation"],
        total_tokens: int,
        parent_training_dataset: Any | None = None,
        global_microbatch_tokens: int = 4_096,
        microbatches_per_update: int = 4,
        process_count: int = 1,
        process_index: int = 0,
    ) -> None:
        self.dataset = dataset
        self.sidecar = decode_recomposition_sidecar(recomposition_sidecar)
        self.centering = centering
        _manifest, statistics, manifest_sha = _manifest_and_statistics(
            dataset,
            self.sidecar,
        )
        if role not in {"train", "evaluation"}:
            raise SharedQError("shared-Q input role must be train or evaluation")
        # Contribution manifests bind the artifact role (train/eval), while
        # provenance separately binds the underlying corpus split
        # (train/validation).  Do not compare the artifact field to the corpus
        # name here: the sealed held-out contribution capture is exactly
        # ``split="eval"``.
        expected_split = "train" if role == "train" else "eval"
        if _manifest["metadata"]["split"] != expected_split:
            raise SharedQError(
                f"shared-Q {role} input requires split {expected_split!r}"
            )
        parent_manifest_sha: str | None = None
        if role == "train" and (
            manifest_sha != centering.training_capture_manifest_sha256
        ):
            if parent_training_dataset is None:
                raise SharedQError(
                    "resumed shared-Q training requires its sealed parent capture"
                )
            parent_manifest, _parent_stats, parent_manifest_sha = (
                _manifest_and_statistics(parent_training_dataset, self.sidecar)
            )
            if (
                parent_manifest_sha
                != centering.training_capture_manifest_sha256
            ):
                raise SharedQError(
                    "shared-Q parent capture differs from its centering contract"
                )
            parent_shards = parent_manifest["shards"]
            current_shards = _manifest["shards"]
            if (
                int(parent_manifest["total_tokens"])
                >= int(_manifest["total_tokens"])
                or len(parent_shards) >= len(current_shards)
                or current_shards[: len(parent_shards)] != parent_shards
            ):
                raise SharedQError(
                    "shared-Q training capture is not a monotonic sealed "
                    "extension of its centering parent"
                )
        elif parent_training_dataset is not None:
            raise SharedQError(
                "shared-Q parent capture is allowed only for resumed training"
            )
        available = _dataset_total_tokens(dataset)
        for name, value in (
            ("total_tokens", total_tokens),
            ("global_microbatch_tokens", global_microbatch_tokens),
            ("microbatches_per_update", microbatches_per_update),
            ("process_count", process_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SharedQError(f"{name} must be a positive integer")
        if total_tokens != available:
            raise SharedQError(
                "authoritative shared-Q input requires the complete sealed capture"
            )
        if global_microbatch_tokens % process_count:
            raise SharedQError(
                "global_microbatch_tokens must divide across processes"
            )
        if not 0 <= process_index < process_count:
            raise SharedQError("process_index is outside process_count")
        if (
            centering.recomposition_sidecar_sha256
            != self.sidecar.sidecar_sha256
        ):
            raise SharedQError(
                "shared-Q input and centering use different sidecars"
            )

        self.capture_manifest_sha256 = manifest_sha
        self.role = role
        self.total_tokens = total_tokens
        self.global_microbatch_tokens = global_microbatch_tokens
        self.microbatches_per_update = microbatches_per_update
        self.process_count = process_count
        self.process_index = process_index
        self.process_partition_strategy = PROCESS_PARTITION_STRATEGY
        self.process_range_start, self.process_range_end = contiguous_process_range(
            total_tokens,
            process_count=process_count,
            process_index=process_index,
        )
        self.local_rows = global_microbatch_tokens // process_count
        effective = global_microbatch_tokens * microbatches_per_update
        self.planned_updates = math.ceil(total_tokens / effective)
        self.sealed_sst = _shared_sst(statistics, centering.mean)
        unsigned = {
            "schema_version": SHARED_Q_INPUT_SCHEMA_VERSION,
            "role": role,
            "capture_manifest_sha256": manifest_sha,
            "centering_contract_sha256": centering.contract_sha256,
            "centering_parent_manifest_sha256": parent_manifest_sha,
            "recomposition_sidecar_sha256": self.sidecar.sidecar_sha256,
            "total_tokens": total_tokens,
            "sealed_sst": self.sealed_sst,
            "global_microbatch_tokens": global_microbatch_tokens,
            "microbatches_per_update": microbatches_per_update,
            "planned_updates": self.planned_updates,
            "process_count": process_count,
            "process_index": process_index,
            "local_rows_per_microbatch": self.local_rows,
        }
        self.receipt = SharedQInputReceipt(
            **unsigned,
            receipt_sha256=canonical_sha256(unsigned),
        )

    def _microbatch(self, microbatch_index: int) -> SharedQHostMicrobatch:
        if (
            isinstance(microbatch_index, bool)
            or not isinstance(microbatch_index, int)
            or microbatch_index < 0
        ):
            raise SharedQError(
                "microbatch_index must be a nonnegative integer"
            )
        read_start = self.process_range_start + microbatch_index * self.local_rows
        local_valid = max(
            0,
            min(self.local_rows, self.process_range_end - read_start),
        )
        activation = np.zeros(
            (self.local_rows, HIDDEN_SIZE),
            dtype=np.float32,
        )
        valid = np.zeros((self.local_rows,), dtype=np.bool_)
        sequence_id = np.full(
            (self.local_rows,),
            PADDING_SEQUENCE_ID,
            dtype=np.int64,
        )
        position = np.full(
            (self.local_rows,),
            PADDING_POSITION,
            dtype=np.uint16,
        )
        if local_valid:
            shared_output, selected_sequence, selected_position = (
                _read_exact_shared_range(
                self.dataset,
                read_start,
                read_start + local_valid,
            )
            )
            _, normalized_bits = maxtext_rmsnorm_bf16(
                shared_output,
                self.sidecar.shared_rmsnorm_scale_bf16_bits,
                epsilon=float(self.sidecar.epsilon),
            )
            activation[:local_valid] = bf16_bits_to_float32(normalized_bits)
            valid[:local_valid] = True
            sequence_id[:local_valid] = selected_sequence
            position[:local_valid] = selected_position
        return SharedQHostMicrobatch(
            activation,
            valid,
            sequence_id,
            position,
        )

    def iter_updates(self):
        yield from self.iter_updates_from(0)

    def iter_updates_from(self, start_update: int):
        """Seek directly to an exact shared-Q optimizer boundary."""

        if (
            isinstance(start_update, bool)
            or not isinstance(start_update, int)
            or not 0 <= start_update <= self.planned_updates
        ):
            raise SharedQError(
                "start_update must lie in the fixed update plan"
            )
        for update in range(start_update, self.planned_updates):
            start = update * self.microbatches_per_update
            yield tuple(
                self._microbatch(index)
                for index in range(start, start + self.microbatches_per_update)
            )


class SharedQDistributedAux(NamedTuple):
    sse: jax.Array
    sst: jax.Array
    l0_sum: jax.Array
    valid_tokens: jax.Array
    topk_valid: jax.Array
    firing_count: jax.Array
    params_finite: jax.Array


class SharedQDeviceRows(NamedTuple):
    token_sse: jax.Array
    token_sst: jax.Array
    target: jax.Array
    reconstruction: jax.Array
    l0: jax.Array
    valid_mask: jax.Array


@dataclasses.dataclass(frozen=True, slots=True)
class SharedQEvaluationRows:
    """Process-local normalized-shared targets and Q reconstructions."""

    token_sse: np.ndarray
    token_sst: np.ndarray
    target: np.ndarray
    reconstruction: np.ndarray
    l0: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        valid = np.asarray(self.valid_mask)
        if valid.ndim != 2 or valid.dtype != np.bool_:
            raise SharedQError(
                "shared-Q valid_mask must be bool [microbatches,rows]"
            )
        for name in ("token_sse", "token_sst"):
            value = np.asarray(getattr(self, name))
            if (
                value.shape != valid.shape
                or not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all()
                or np.any(value < 0)
                or np.any(value[~valid] != 0)
            ):
                raise SharedQError(
                    f"shared-Q {name} must be finite/nonnegative and zero on padding"
                )
        l0 = np.asarray(self.l0)
        if (
            l0.shape != valid.shape
            or not np.issubdtype(l0.dtype, np.integer)
            or np.any(l0[~valid] != 0)
        ):
            raise SharedQError(
                "shared-Q L0 must be integer, aligned, and zero on padding"
            )
        target = np.asarray(self.target)
        reconstruction = np.asarray(self.reconstruction)
        if (
            target.ndim != 3
            or target.shape[:2] != valid.shape
            or reconstruction.shape != target.shape
        ):
            raise SharedQError(
                "shared-Q target/reconstruction must align as [micro,rows,hidden]"
            )
        for name, value in (
            ("target", target),
            ("reconstruction", reconstruction),
        ):
            if (
                not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all()
                or np.any(value[~valid] != 0)
            ):
                raise SharedQError(
                    f"shared-Q {name} must be finite and zero on padding"
                )
        for field in dataclasses.fields(self):
            value = np.ascontiguousarray(np.asarray(getattr(self, field.name)))
            value.setflags(write=False)
            object.__setattr__(self, field.name, value)

    @property
    def tokens(self) -> int:
        return int(np.count_nonzero(self.valid_mask))


@dataclasses.dataclass(frozen=True, slots=True)
class SharedQEvaluationEvidenceChunk:
    """Token identities plus Q rows for a single process/update."""

    update_index: int
    capture_manifest_sha256: str
    centering_contract_sha256: str
    checkpoint_sha256: str
    recomposition_sidecar_sha256: str
    sequence_id: np.ndarray
    position: np.ndarray
    rows: SharedQEvaluationRows

    def __post_init__(self) -> None:
        if (
            isinstance(self.update_index, bool)
            or not isinstance(self.update_index, int)
            or self.update_index < 0
        ):
            raise SharedQError("shared-Q update_index must be nonnegative")
        for name in (
            "capture_manifest_sha256",
            "centering_contract_sha256",
            "checkpoint_sha256",
            "recomposition_sidecar_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise SharedQError(f"shared-Q evidence {name} must be SHA-256")
        sequence = np.asarray(self.sequence_id)
        position = np.asarray(self.position)
        valid = self.rows.valid_mask
        if sequence.dtype != np.int64 or sequence.shape != valid.shape:
            raise SharedQError("shared-Q sequence IDs are not aligned int64")
        if position.dtype != np.uint16 or position.shape != valid.shape:
            raise SharedQError("shared-Q positions are not aligned uint16")
        if np.any(sequence[valid] < 0) or np.any(
            position[valid] >= SEQUENCE_LENGTH
        ):
            raise SharedQError("shared-Q scientific token identity is invalid")
        if np.any(sequence[~valid] != PADDING_SEQUENCE_ID) or np.any(
            position[~valid] != PADDING_POSITION
        ):
            raise SharedQError("shared-Q padding identity sentinel changed")
        if np.count_nonzero(valid):
            keys = np.rec.fromarrays(
                (sequence[valid], position[valid]),
                names=("sequence_id", "position"),
            )
            if np.unique(keys).size != keys.size:
                raise SharedQError("shared-Q evidence contains duplicate token keys")
        sequence = np.ascontiguousarray(sequence)
        position = np.ascontiguousarray(position)
        sequence.setflags(write=False)
        position.setflags(write=False)
        object.__setattr__(self, "sequence_id", sequence)
        object.__setattr__(self, "position", position)

    @property
    def tokens(self) -> int:
        return self.rows.tokens

    @property
    def token_identity_sha256(self) -> str:
        valid = self.rows.valid_mask
        return token_identity_sha256(
            self.sequence_id[valid],
            self.position[valid],
        )


def _validate_q_sae_config(config: SAEConfig) -> SharedQLayout:
    if (
        config.condition != "B"
        or config.num_experts != 1
        or config.top_routed != 1
        or config.total_features != config.bank_width
    ):
        raise ValueError(
            "shared Q must use one generic global dictionary, not routed banks"
        )
    return SharedQLayout(
        hidden_dim=config.hidden_dim,
        total_features=config.total_features,
        k=config.k,
    )


def _validate_q_mesh(
    mesh: Mesh,
    config: SAEConfig,
    distributed: DistributedConfig,
) -> None:
    shape = mesh.shape
    if distributed.data_axis not in shape or distributed.feature_axis not in shape:
        raise ValueError("shared-Q mesh lacks data/feature axes")
    if set(mesh.axis_names) != {
        distributed.data_axis,
        distributed.feature_axis,
    }:
        raise ValueError("shared Q requires an exact two-axis mesh")
    feature_size = int(shape[distributed.feature_axis])
    if config.total_features % feature_size:
        raise ValueError("shared-Q features must divide across feature shards")
    if config.total_features // feature_size < config.k:
        raise ValueError("every shared-Q feature shard must contain at least k")
    devices = np.asarray(mesh.devices, dtype=object)
    data_dim = mesh.axis_names.index(distributed.data_axis)
    feature_dim = mesh.axis_names.index(distributed.feature_axis)
    rows = np.moveaxis(devices, (data_dim, feature_dim), (0, 1))
    for row in rows:
        if len({int(device.process_index) for device in row}) != 1:
            raise ValueError("shared-Q feature axis must remain host-local")


def _q_local_forward(
    params: SAEParams,
    target: jax.Array,
    valid: jax.Array,
    mean: jax.Array,
    config: SAEConfig,
    *,
    feature_axis: str,
) -> tuple[jax.Array, Any, jax.Array, jax.Array]:
    """Return sparse reconstruction, selection, local fires, and input health."""

    local_width = params.encoder.shape[0]
    offset = jax.lax.axis_index(feature_axis) * local_width
    centered = target.astype(params.encoder.dtype) - mean.astype(
        params.encoder.dtype
    )
    scores = jnp.einsum(
        "bd,fd->bf",
        centered,
        params.encoder,
        preferred_element_type=jnp.float32,
        precision=jax.lax.Precision.DEFAULT,
    ) + params.bias.astype(jnp.float32)
    local_ids = offset + jnp.arange(local_width, dtype=jnp.int32)
    local = local_topk_candidates(
        jax.nn.relu(scores),
        jnp.broadcast_to(local_ids[None, :], scores.shape),
        config.k,
    )
    selection = _merge_local_candidates(
        local.values,
        local.latent_ids,
        config.k,
        feature_axis=feature_axis,
    )
    reconstruction = _decode_selected_columns_default_precision(
        params.decoder,
        selection,
        mean,
        feature_offset=offset,
        feature_axis=feature_axis,
    )
    selected = (selection.values > 0) & selection.valid[:, None] & valid[:, None]
    owned = (selection.latent_ids >= offset) & (
        selection.latent_ids < offset + local_width
    )
    selected_local = jnp.clip(
        selection.latent_ids - offset,
        0,
        local_width - 1,
    )
    firing_count = (
        jnp.zeros((local_width,), jnp.int32)
        .at[selected_local.reshape(-1)]
        .add((selected & owned).reshape(-1).astype(jnp.int32))
    )
    input_valid = (
        jnp.all(jnp.isfinite(target) | ~valid[:, None])
        & jnp.all(jnp.isfinite(mean))
        & jnp.all(selection.valid | ~valid)
    )
    return reconstruction, selection, firing_count, input_valid


def _build_q_mapped(
    mesh: Mesh,
    config: SAEConfig,
    distributed: DistributedConfig,
    *,
    parameter_dtype: str,
    emit_rows: bool,
    rematerialize_microbatch: bool = False,
):
    data_axis = distributed.data_axis
    feature_axis = distributed.feature_axis
    compute_dtype = (
        jnp.bfloat16 if parameter_dtype == "bfloat16" else jnp.float32
    )

    def mapped(
        differentiable_params: SAEParams,
        activation: jax.Array,
        valid_mask: jax.Array,
        mean: jax.Array,
    ):
        local_width = differentiable_params.encoder.shape[0]
        initial = (
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(True),
            jnp.zeros((local_width,), jnp.int32),
        )

        def one_micro(carry, inputs):
            sse, sst, l0_sum, tokens, healthy, firing = carry
            target, valid = inputs
            params = jax.tree_util.tree_map(
                lambda value: value.astype(compute_dtype),
                differentiable_params,
            )
            reconstruction, selection, fires, valid_here = _q_local_forward(
                params,
                target,
                valid,
                mean,
                config,
                feature_axis=feature_axis,
            )
            target32 = target.astype(jnp.float32)
            reconstruction32 = reconstruction.astype(jnp.float32)
            mean32 = mean.astype(jnp.float32)
            token_sse = jnp.sum(
                jnp.square(target32 - reconstruction32),
                axis=1,
                dtype=jnp.float32,
            )
            token_sst = jnp.sum(
                jnp.square(target32 - mean32[None, :]),
                axis=1,
                dtype=jnp.float32,
            )
            token_sse = jnp.where(valid, token_sse, 0.0)
            token_sst = jnp.where(valid, token_sst, 0.0)
            selected = (
                (selection.values > 0)
                & selection.valid[:, None]
                & valid[:, None]
            )
            l0 = jnp.sum(selected, axis=1, dtype=jnp.int32)
            rows = SharedQDeviceRows(
                token_sse,
                token_sst,
                jnp.where(valid[:, None], target32, 0.0),
                jnp.where(valid[:, None], reconstruction32, 0.0),
                l0,
                valid,
            )
            updated = (
                sse + jnp.sum(token_sse, dtype=jnp.float32),
                sst + jnp.sum(token_sst, dtype=jnp.float32),
                l0_sum + jnp.sum(l0, dtype=jnp.int32),
                tokens + jnp.sum(valid, dtype=jnp.int32),
                healthy & valid_here,
                firing + fires,
            )
            return updated, rows if emit_rows else None

        microbatch_function = (
            jax.checkpoint(one_micro)
            if rematerialize_microbatch and not emit_rows
            else one_micro
        )
        scanned, rows = jax.lax.scan(
            microbatch_function,
            initial,
            (activation, valid_mask),
        )
        local_sse, local_sst, local_l0, local_tokens, topk_valid, firing = scanned
        global_sse = jax.lax.pmean(
            jax.lax.psum(local_sse, data_axis),
            feature_axis,
        )
        global_sst = jax.lax.pmean(
            jax.lax.psum(local_sst, data_axis),
            feature_axis,
        )
        global_l0 = jax.lax.pmax(
            jax.lax.psum(local_l0, data_axis),
            feature_axis,
        )
        global_tokens = jax.lax.pmax(
            jax.lax.psum(local_tokens, data_axis),
            feature_axis,
        )
        topk_valid = (
            jax.lax.pmin(
                topk_valid.astype(jnp.int32),
                (data_axis, feature_axis),
            )
            > 0
        )
        firing = jax.lax.psum(firing, data_axis)
        params_finite = jnp.asarray(True)
        for leaf in jax.tree_util.tree_leaves(differentiable_params):
            params_finite = params_finite & jnp.all(jnp.isfinite(leaf))
        params_finite = (
            jax.lax.pmin(
                params_finite.astype(jnp.int32),
                (data_axis, feature_axis),
            )
            > 0
        )
        aux = SharedQDistributedAux(
            global_sse,
            global_sst,
            global_l0,
            global_tokens,
            topk_valid,
            firing,
            params_finite,
        )
        good = topk_valid & params_finite & (global_sst > 0)
        loss = jnp.where(
            good,
            global_sse / jnp.maximum(global_sst, _EPS),
            jnp.asarray(jnp.nan, jnp.float32),
        )
        return (loss, aux, rows) if emit_rows else (loss, aux)

    aux_specs = SharedQDistributedAux(
        P(),
        P(),
        P(),
        P(),
        P(),
        P(feature_axis),
        P(),
    )
    in_specs = (
        _param_specs(feature_axis),
        P(None, data_axis, None),
        P(None, data_axis),
        P(None),
    )
    if not emit_rows:
        return jax.shard_map(
            mapped,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=(P(), aux_specs),
            axis_names={data_axis, feature_axis},
            check_vma=False,
        )
    row_specs = SharedQDeviceRows(
        P(None, data_axis),
        P(None, data_axis),
        P(None, data_axis, None),
        P(None, data_axis, None),
        P(None, data_axis),
        P(None, data_axis),
    )
    return jax.shard_map(
        mapped,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=(P(), aux_specs, row_specs),
        axis_names={data_axis, feature_axis},
        check_vma=False,
    )


def _named_q_params(mesh: Mesh, feature_axis: str) -> SAEParams:
    return jax.tree_util.tree_map(
        lambda spec: NamedSharding(mesh, spec),
        _param_specs(feature_axis),
    )


def _validate_q_params(params: SAEParams, config: SAEConfig) -> None:
    expected = (
        (config.total_features, config.hidden_dim),
        (config.hidden_dim, config.total_features),
        (config.total_features,),
    )
    observed = (
        tuple(params.encoder.shape),
        tuple(params.decoder.shape),
        tuple(params.bias.shape),
    )
    if observed != expected:
        raise SharedQError(
            f"shared-Q parameter shapes {observed} differ from {expected}"
        )
    dtypes = {jnp.dtype(value.dtype) for value in params}
    if len(dtypes) != 1 or next(iter(dtypes)) not in {
        jnp.dtype(jnp.bfloat16),
        jnp.dtype(jnp.float32),
    }:
        raise SharedQError("shared-Q parameters must share BF16 or FP32 dtype")


class SharedQDistributedTrainer:
    """One corrected feature-sharded optimizer update for common Q."""

    def __init__(
        self,
        mesh: Mesh,
        config: SAEConfig,
        distributed: DistributedConfig | None = None,
        *,
        rematerialize_microbatch: bool = True,
    ) -> None:
        self.mesh = mesh
        self.config = config
        self.layout = _validate_q_sae_config(config)
        self.distributed = distributed or DistributedConfig()
        if type(rematerialize_microbatch) is not bool:
            raise TypeError("rematerialize_microbatch must be boolean")
        self.rematerialize_microbatch = rematerialize_microbatch
        _validate_q_mesh(mesh, config, self.distributed)
        mapped = _build_q_mapped(
            mesh,
            config,
            self.distributed,
            parameter_dtype=config.parameter_dtype,
            emit_rows=False,
            rematerialize_microbatch=rematerialize_microbatch,
        )

        def complete(
            state: TrainState,
            batch: DistributedBatch,
            mean: jax.Array,
        ):
            shadow = jax.tree_util.tree_map(
                lambda value: value.astype(jnp.float32),
                state.params,
            )
            (loss, aux), gradients = jax.value_and_grad(
                mapped,
                has_aux=True,
            )(
                shadow,
                batch.activation,
                batch.valid_mask,
                mean,
            )
            exact_l0 = aux.valid_tokens * jnp.asarray(
                config.k,
                aux.l0_sum.dtype,
            )
            raw_l0 = aux.l0_sum.astype(jnp.float32) / jnp.maximum(
                aux.valid_tokens.astype(jnp.float32),
                1.0,
            )
            l0 = jnp.where(
                aux.l0_sum == exact_l0,
                jnp.asarray(config.k, jnp.float32),
                raw_l0,
            )
            state, metrics = apply_gradients(
                state,
                gradients,
                config,
                loss=loss,
                valid_tokens=aux.valid_tokens,
                fired_mask=aux.firing_count > 0,
                l0=l0,
                sse=aux.sse,
                sst=aux.sst,
                topk_valid=aux.topk_valid & aux.params_finite,
            )
            state_finite = jnp.asarray(True)
            for tree in (state.params, state.optimizer.m, state.optimizer.v):
                for leaf in jax.tree_util.tree_leaves(tree):
                    state_finite = state_finite & jnp.all(jnp.isfinite(leaf))
            return state, metrics, aux, state_finite

        self._compiled = jax.jit(
            complete,
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
        process_local: bool,
    ) -> tuple[DistributedBatch, jax.Array]:
        batch = (
            microbatches
            if isinstance(microbatches, DistributedBatch)
            else stack_microbatches(
                microbatches,
                self.config,
                self.distributed,
            )
        )
        placed = (
            shard_process_local_batch(batch, self.mesh, self.distributed)
            if process_local
            else shard_batch(batch, self.mesh, self.distributed)
        )
        host_mean = np.asarray(mean)
        if (
            host_mean.shape != (self.config.hidden_dim,)
            or not np.issubdtype(host_mean.dtype, np.floating)
            or not np.isfinite(host_mean).all()
        ):
            raise ValueError(
                f"shared-Q mean must be finite [{self.config.hidden_dim}]"
            )
        return placed, replicate_process_local_array(
            jnp.asarray(host_mean, jnp.float32),
            self.mesh,
        )

    def step(
        self,
        state: TrainState,
        microbatches: Sequence[Any] | DistributedBatch,
        mean: Any,
        *,
        process_local: bool = False,
    ) -> tuple[TrainState, StepMetrics]:
        _validate_q_params(state.params, self.config)
        batch, mean_array = self._prepare(
            microbatches,
            mean,
            process_local=process_local,
        )
        executable = self._compiled if self._executable is None else self._executable
        new_state, metrics, aux, state_finite = executable(
            state,
            batch,
            mean_array,
        )
        (
            tokens,
            l0_sum,
            topk_valid,
            params_finite,
            state_finite,
            metrics_finite,
        ) = jax.device_get(
            (
                aux.valid_tokens,
                aux.l0_sum,
                aux.topk_valid,
                aux.params_finite,
                state_finite,
                jnp.isfinite(metrics.loss)
                & jnp.isfinite(metrics.sse)
                & jnp.isfinite(metrics.sst)
                & jnp.isfinite(metrics.grad_norm),
            )
        )
        token_count = int(tokens)
        if token_count <= 0:
            raise SharedQError("shared-Q update contains zero scientific tokens")
        if not bool(params_finite) or not bool(state_finite):
            raise SharedQError(
                "shared-Q update has non-finite parameters or optimizer state"
            )
        if not bool(topk_valid):
            raise SharedQError(
                "shared-Q update has invalid inputs or fewer than k positives"
            )
        if int(l0_sum) != token_count * self.config.k:
            raise SharedQError("shared-Q update violates exact TopK cardinality")
        if not bool(metrics_finite):
            raise SharedQError("shared-Q update produced non-finite metrics")
        return new_state, metrics

    def compile(
        self,
        state: TrainState,
        microbatches: Sequence[Any] | DistributedBatch,
        mean: Any,
        *,
        process_local: bool = False,
    ) -> float:
        """Collectively compile Q without accepting an optimizer update."""

        _validate_q_params(state.params, self.config)
        batch, mean_array = self._prepare(
            microbatches,
            mean,
            process_local=process_local,
        )
        started = time.perf_counter()
        self._executable = self._compiled.lower(
            state,
            batch,
            mean_array,
        ).compile()
        return time.perf_counter() - started


def init_shared_q_state(
    key: jax.Array,
    mesh: Mesh,
    config: SAEConfig,
    distributed: DistributedConfig | None = None,
) -> TrainState:
    """Initialize unit-column Q directly in its feature shards."""

    _validate_q_sae_config(config)
    selected = distributed or DistributedConfig()
    _validate_q_mesh(mesh, config, selected)
    return init_distributed_state(key, mesh, config, selected)


def _feature_vector(value: jax.Array) -> np.ndarray:
    if value.is_fully_addressable:
        return np.asarray(jax.device_get(value))
    unique: dict[str, tuple[Any, np.ndarray]] = {}
    for shard in value.addressable_shards:
        unique.setdefault(
            repr(shard.index),
            (shard.index, np.asarray(jax.device_get(shard.data))),
        )
    if not unique:
        raise SharedQError("shared-Q feature vector has no addressable shards")

    def start(item: tuple[Any, np.ndarray]) -> int:
        index = item[0][0]
        if isinstance(index, slice):
            return int(index.start or 0)
        return int(index)

    pieces = [item[1] for item in sorted(unique.values(), key=start)]
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)


def _q_host_rows(rows: SharedQDeviceRows) -> SharedQEvaluationRows:
    return SharedQEvaluationRows(
        **{
            field: _addressable_data_rows(getattr(rows, field))
            for field in SharedQDeviceRows._fields
        }
    )


class SharedQDistributedEvaluator:
    """Sparse feature-sharded Q evaluator with process-local sufficient rows."""

    def __init__(
        self,
        mesh: Mesh,
        config: SAEConfig,
        distributed: DistributedConfig | None = None,
    ) -> None:
        self.mesh = mesh
        self.config = config
        self.layout = _validate_q_sae_config(config)
        self.distributed = distributed or DistributedConfig()
        _validate_q_mesh(mesh, config, self.distributed)
        mapped = _build_q_mapped(
            mesh,
            config,
            self.distributed,
            parameter_dtype=config.parameter_dtype,
            emit_rows=True,
        )
        shardings = batch_shardings(mesh, self.distributed)
        self._compiled = jax.jit(
            mapped,
            in_shardings=(
                _named_q_params(mesh, self.distributed.feature_axis),
                shardings.activation,
                shardings.valid_mask,
                NamedSharding(mesh, P()),
            ),
        )

    def step(
        self,
        params: SAEParams,
        microbatches: Sequence[Any] | DistributedBatch,
        mean: Any,
        *,
        process_local: bool = False,
    ) -> tuple[SharedQDistributedAux, SharedQEvaluationRows]:
        _validate_q_params(params, self.config)
        batch = (
            microbatches
            if isinstance(microbatches, DistributedBatch)
            else stack_microbatches(
                microbatches,
                self.config,
                self.distributed,
            )
        )
        placed = (
            shard_process_local_batch(batch, self.mesh, self.distributed)
            if process_local
            else shard_batch(batch, self.mesh, self.distributed)
        )
        host_mean = np.asarray(mean)
        if (
            host_mean.shape != (self.config.hidden_dim,)
            or not np.issubdtype(host_mean.dtype, np.floating)
            or not np.isfinite(host_mean).all()
        ):
            raise ValueError(
                f"shared-Q mean must be finite [{self.config.hidden_dim}]"
            )
        mean_array = replicate_process_local_array(
            jnp.asarray(host_mean, jnp.float32),
            self.mesh,
        )
        _loss, aux, device_rows = self._compiled(
            params,
            placed.activation,
            placed.valid_mask,
            mean_array,
        )
        (
            tokens,
            l0_sum,
            topk_valid,
            params_finite,
            sse,
            sst,
        ) = jax.device_get(
            (
                aux.valid_tokens,
                aux.l0_sum,
                aux.topk_valid,
                aux.params_finite,
                aux.sse,
                aux.sst,
            )
        )
        token_count = int(tokens)
        if token_count <= 0:
            raise SharedQError("shared-Q evaluation group has zero tokens")
        if not bool(params_finite):
            raise SharedQError("shared-Q evaluation parameters are non-finite")
        if not bool(topk_valid):
            raise SharedQError(
                "shared-Q evaluation input or positive TopK is invalid"
            )
        if int(l0_sum) != token_count * self.config.k:
            raise SharedQError("shared-Q evaluation L0 differs from exact TopK")
        if (
            not math.isfinite(float(sse))
            or not math.isfinite(float(sst))
            or float(sse) < 0
            or float(sst) <= 0
        ):
            raise SharedQError("shared-Q evaluation SSE/SST is invalid")
        rows = _q_host_rows(device_rows)
        if np.any(rows.l0[rows.valid_mask] != self.config.k):
            raise SharedQError("shared-Q per-token L0 differs from exact TopK")
        return aux, rows


def _q_evidence(
    update_index: int,
    microbatches: Sequence[SharedQHostMicrobatch],
    rows: SharedQEvaluationRows,
    source: SharedQProcessInput,
    checkpoint_sha256: str,
) -> SharedQEvaluationEvidenceChunk:
    return SharedQEvaluationEvidenceChunk(
        update_index=update_index,
        capture_manifest_sha256=source.capture_manifest_sha256,
        centering_contract_sha256=source.centering.contract_sha256,
        checkpoint_sha256=checkpoint_sha256,
        recomposition_sidecar_sha256=source.sidecar.sidecar_sha256,
        sequence_id=np.stack(
            [np.asarray(item.sequence_id) for item in microbatches]
        ),
        position=np.stack([np.asarray(item.position) for item in microbatches]),
        rows=rows,
    )


def _firing_properties(counts: np.ndarray) -> tuple[float, float]:
    values = np.asarray(counts, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        raise SharedQError("shared-Q evaluation observed no feature firings")
    probability = values / total
    positive = probability[probability > 0]
    entropy = float(-np.sum(positive * np.log(positive), dtype=np.float64))
    ordered = np.sort(values)
    n = ordered.size
    gini = float(
        (
            2.0
            * np.dot(
                np.arange(1, n + 1, dtype=np.float64),
                ordered,
            )
            / (n * total)
        )
        - (n + 1.0) / n
    )
    return entropy, max(0.0, min(1.0, gini))


@dataclasses.dataclass(frozen=True, slots=True)
class SharedQEvaluationResult:
    schema_version: str
    capture_manifest_sha256: str
    input_receipt_sha256: str
    centering_contract_sha256: str
    checkpoint_sha256: str
    tokens: int
    sse: float
    sst: float
    computed_sst: float
    l0_sum: int
    l0: int
    fired_feature_fraction: float
    dead_rate: float
    firing_entropy: float
    firing_gini: float
    updates: int
    local_sufficient_rows: int

    @property
    def nmse(self) -> float:
        return self.sse / self.sst

    @property
    def ev(self) -> float:
        return 1.0 - self.nmse

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(dataclasses.asdict(self))


def evaluate_shared_q_input(
    params: SAEParams,
    source: SharedQProcessInput,
    mesh: Mesh,
    config: SAEConfig,
    *,
    checkpoint_sha256: str,
    distributed: DistributedConfig | None = None,
    row_sink: Callable[[SharedQEvaluationEvidenceChunk], None] | None = None,
) -> SharedQEvaluationResult:
    """Evaluate common Q over one complete sealed held-out capture."""

    layout = _validate_q_sae_config(config)
    if source.role != "evaluation":
        raise SharedQError("shared-Q evaluator requires an evaluation-role input")
    if not layout.authoritative:
        raise SharedQError("authoritative shared-Q evaluation requires 8192/64")
    if (
        not isinstance(checkpoint_sha256, str)
        or _SHA256_RE.fullmatch(checkpoint_sha256) is None
    ):
        raise SharedQError("shared-Q evaluation requires checkpoint SHA-256")
    selected = distributed or DistributedConfig(
        microbatches_per_update=source.microbatches_per_update
    )
    if source.microbatches_per_update != selected.microbatches_per_update:
        raise SharedQError("shared-Q input/evaluator microbatch counts differ")
    if (
        source.process_count != jax.process_count()
        or source.process_index != jax.process_index()
    ):
        raise SharedQError("shared-Q input partition differs from JAX runtime")
    evaluator = SharedQDistributedEvaluator(mesh, config, selected)
    sse = 0.0
    computed_sst = 0.0
    tokens = 0
    l0_sum = 0
    firing = np.zeros((config.total_features,), dtype=np.int64)
    local_rows = 0
    updates = 0
    for update_index, microbatches in enumerate(source.iter_updates()):
        aux, rows = evaluator.step(
            params,
            microbatches,
            source.centering.mean,
            process_local=source.process_count > 1,
        )
        host = jax.device_get(aux)
        sse += float(host.sse)
        computed_sst += float(host.sst)
        tokens += int(host.valid_tokens)
        l0_sum += int(host.l0_sum)
        observed = np.asarray(_feature_vector(aux.firing_count), dtype=np.int64)
        if observed.shape != firing.shape:
            raise SharedQError("shared-Q firing vector is incomplete")
        firing += observed
        local_rows += rows.tokens
        updates += 1
        if row_sink is not None:
            row_sink(
                _q_evidence(
                    update_index,
                    microbatches,
                    rows,
                    source,
                    checkpoint_sha256,
                )
            )
    if updates != source.planned_updates or tokens != source.total_tokens:
        raise SharedQError("shared-Q evaluator did not consume its exact input")
    if l0_sum != tokens * config.k or int(firing.sum()) != l0_sum:
        raise SharedQError("shared-Q aggregate firing/L0 accounting is invalid")
    if not math.isclose(
        computed_sst,
        source.sealed_sst,
        rel_tol=5e-4,
        abs_tol=1e-6 * max(1.0, source.sealed_sst),
    ):
        raise SharedQError(
            "device shared-Q SST disagrees with sealed FP64 statistics"
        )
    live = float(np.mean(firing > 0))
    entropy, gini = _firing_properties(firing)
    return SharedQEvaluationResult(
        schema_version=SHARED_Q_EVALUATION_SCHEMA_VERSION,
        capture_manifest_sha256=source.capture_manifest_sha256,
        input_receipt_sha256=source.receipt.receipt_sha256,
        centering_contract_sha256=source.centering.contract_sha256,
        checkpoint_sha256=checkpoint_sha256,
        tokens=tokens,
        sse=sse,
        sst=source.sealed_sst,
        computed_sst=computed_sst,
        l0_sum=l0_sum,
        l0=config.k,
        fired_feature_fraction=live,
        dead_rate=1.0 - live,
        firing_entropy=entropy,
        firing_gini=gini,
        updates=updates,
        local_sufficient_rows=local_rows,
    )


def _rmsnorm_post(
    value: np.ndarray,
    scale_bits: np.ndarray,
    sidecar: RecompositionSidecar,
) -> np.ndarray:
    bits = float32_to_bf16_bits(np.asarray(value, dtype=np.float32))
    _, output = maxtext_rmsnorm_bf16(
        bits,
        scale_bits,
        epsilon=float(sidecar.epsilon),
    )
    return output


def _bf16_add_bits(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        raise SharedQError("full-FFN BF16 addition operands differ in shape")
    return float32_to_bf16_bits(
        np.add(
            bf16_bits_to_float32(left),
            bf16_bits_to_float32(right),
            dtype=np.float32,
        )
    )


@dataclasses.dataclass(frozen=True, slots=True)
class FullFFNRecompositionRows:
    """Exact target and both full-FFN reconstructions for one aligned chunk."""

    sequence_id: np.ndarray
    position: np.ndarray
    valid_mask: np.ndarray
    target: np.ndarray
    oracle_shared_reconstruction: np.ndarray
    learned_shared_reconstruction: np.ndarray
    recomposition_sidecar_sha256: str
    shared_q_capture_manifest_sha256: str
    shared_q_centering_contract_sha256: str
    shared_q_checkpoint_sha256: str

    def __post_init__(self) -> None:
        sequence = np.asarray(self.sequence_id)
        position = np.asarray(self.position)
        valid = np.asarray(self.valid_mask)
        if (
            sequence.dtype != np.int64
            or position.dtype != np.uint16
            or valid.dtype != np.bool_
            or sequence.shape != valid.shape
            or position.shape != valid.shape
            or valid.ndim != 2
        ):
            raise SharedQError("full-FFN token identities/mask are not aligned")
        if np.any(sequence[valid] < 0) or np.any(
            position[valid] >= SEQUENCE_LENGTH
        ):
            raise SharedQError("full-FFN scientific identity is invalid")
        if np.any(sequence[~valid] != PADDING_SEQUENCE_ID) or np.any(
            position[~valid] != PADDING_POSITION
        ):
            raise SharedQError("full-FFN padding sentinel changed")
        target = np.asarray(self.target)
        if target.ndim != 3 or target.shape[:2] != valid.shape:
            raise SharedQError("full-FFN target must be [micro,rows,hidden]")
        for name in (
            "target",
            "oracle_shared_reconstruction",
            "learned_shared_reconstruction",
        ):
            value = np.asarray(getattr(self, name))
            if (
                value.shape != target.shape
                or not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all()
                or np.any(value[~valid] != 0)
            ):
                raise SharedQError(
                    f"full-FFN {name} must be finite/aligned and zero on padding"
                )
        for name in (
            "recomposition_sidecar_sha256",
            "shared_q_capture_manifest_sha256",
            "shared_q_centering_contract_sha256",
            "shared_q_checkpoint_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise SharedQError(f"full-FFN rows require {name}")
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
                value.setflags(write=False)
                object.__setattr__(self, field.name, value)

    @property
    def tokens(self) -> int:
        return int(np.count_nonzero(self.valid_mask))


def recompose_full_ffn_rows(
    routed: ContributionEvaluationEvidenceChunk,
    shared_q: SharedQEvaluationEvidenceChunk,
    recomposition_sidecar: Mapping[str, Any],
) -> FullFFNRecompositionRows:
    """Combine one routed condition and common Q through exact Gemma stages."""

    sidecar = decode_recomposition_sidecar(recomposition_sidecar)
    routed_rows = routed.scores
    q_rows = shared_q.rows
    if routed.update_index != shared_q.update_index:
        raise SharedQError("routed and shared-Q update indices differ")
    if shared_q.recomposition_sidecar_sha256 != sidecar.sidecar_sha256:
        raise SharedQError("shared-Q evidence uses a different sidecar")
    for name, left, right in (
        ("valid mask", routed_rows.valid_mask, q_rows.valid_mask),
        ("sequence ID", routed.sequence_id, shared_q.sequence_id),
        ("position", routed.position, shared_q.position),
    ):
        if not np.array_equal(left, right):
            raise SharedQError(f"routed and shared-Q {name} rows are misaligned")
    if (
        routed_rows.raw_routed_target.shape
        != q_rows.target.shape
        or routed_rows.raw_routed_reconstruction.shape
        != q_rows.reconstruction.shape
        or q_rows.target.shape[-1] != HIDDEN_SIZE
    ):
        raise SharedQError("routed and shared-Q hidden tensors are misaligned")

    shape = q_rows.target.shape
    raw_target = routed_rows.raw_routed_target.reshape(-1, HIDDEN_SIZE)
    raw_estimate = routed_rows.raw_routed_reconstruction.reshape(
        -1,
        HIDDEN_SIZE,
    )
    q_target = q_rows.target.reshape(-1, HIDDEN_SIZE)
    q_estimate = q_rows.reconstruction.reshape(-1, HIDDEN_SIZE)

    routed_target_bits = _rmsnorm_post(
        raw_target,
        sidecar.routed_rmsnorm_scale_bf16_bits,
        sidecar,
    )
    routed_estimate_bits = _rmsnorm_post(
        raw_estimate,
        sidecar.routed_rmsnorm_scale_bf16_bits,
        sidecar,
    )
    # The adapter target is already the exact BF16 normalized shared branch.
    q_target_bits = float32_to_bf16_bits(q_target)
    q_estimate_bits = float32_to_bf16_bits(q_estimate)
    full_target_bits = _rmsnorm_post(
        bf16_bits_to_float32(
            _bf16_add_bits(routed_target_bits, q_target_bits)
        ),
        sidecar.outer_ffn_rmsnorm_scale_bf16_bits,
        sidecar,
    )
    oracle_bits = _rmsnorm_post(
        bf16_bits_to_float32(
            _bf16_add_bits(routed_estimate_bits, q_target_bits)
        ),
        sidecar.outer_ffn_rmsnorm_scale_bf16_bits,
        sidecar,
    )
    learned_bits = _rmsnorm_post(
        bf16_bits_to_float32(
            _bf16_add_bits(routed_estimate_bits, q_estimate_bits)
        ),
        sidecar.outer_ffn_rmsnorm_scale_bf16_bits,
        sidecar,
    )
    valid = q_rows.valid_mask

    def decoded(bits: np.ndarray) -> np.ndarray:
        value = bf16_bits_to_float32(bits).reshape(shape)
        return np.where(valid[..., None], value, 0.0)

    return FullFFNRecompositionRows(
        sequence_id=shared_q.sequence_id.copy(),
        position=shared_q.position.copy(),
        valid_mask=valid.copy(),
        target=decoded(full_target_bits),
        oracle_shared_reconstruction=decoded(oracle_bits),
        learned_shared_reconstruction=decoded(learned_bits),
        recomposition_sidecar_sha256=sidecar.sidecar_sha256,
        shared_q_capture_manifest_sha256=(
            shared_q.capture_manifest_sha256
        ),
        shared_q_centering_contract_sha256=(
            shared_q.centering_contract_sha256
        ),
        shared_q_checkpoint_sha256=shared_q.checkpoint_sha256,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class FullFFNTokenScores:
    sequence_id: np.ndarray
    position: np.ndarray
    valid_mask: np.ndarray
    oracle_shared_sse: np.ndarray
    learned_shared_sse: np.ndarray
    common_sst: np.ndarray
    recomposition_sidecar_sha256: str
    shared_q_capture_manifest_sha256: str
    shared_q_centering_contract_sha256: str
    shared_q_checkpoint_sha256: str

    def __post_init__(self) -> None:
        valid = np.asarray(self.valid_mask)
        if valid.ndim != 2 or valid.dtype != np.bool_:
            raise SharedQError("full-FFN score mask must be boolean rank two")
        for name in (
            "oracle_shared_sse",
            "learned_shared_sse",
            "common_sst",
        ):
            value = np.asarray(getattr(self, name))
            if (
                value.shape != valid.shape
                or not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all()
                or np.any(value < 0)
                or np.any(value[~valid] != 0)
            ):
                raise SharedQError(
                    f"full-FFN {name} is invalid or nonzero on padding"
                )
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
                value.setflags(write=False)
                object.__setattr__(self, field.name, value)
        for name in (
            "recomposition_sidecar_sha256",
            "shared_q_capture_manifest_sha256",
            "shared_q_centering_contract_sha256",
            "shared_q_checkpoint_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise SharedQError(f"full-FFN scores require {name}")

    @property
    def tokens(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def oracle_ev(self) -> float:
        denominator = float(
            np.sum(self.common_sst[self.valid_mask], dtype=np.float64)
        )
        if denominator <= 0:
            raise SharedQError("full-FFN SST must be positive")
        return 1.0 - float(
            np.sum(self.oracle_shared_sse[self.valid_mask], dtype=np.float64)
        ) / denominator

    @property
    def learned_ev(self) -> float:
        denominator = float(
            np.sum(self.common_sst[self.valid_mask], dtype=np.float64)
        )
        if denominator <= 0:
            raise SharedQError("full-FFN SST must be positive")
        return 1.0 - float(
            np.sum(self.learned_shared_sse[self.valid_mask], dtype=np.float64)
        ) / denominator


def full_ffn_token_scores(
    rows: FullFFNRecompositionRows,
    full_ffn_mean: Any,
) -> FullFFNTokenScores:
    """Compute paired oracle/learned shared SSE with one common full-FFN SST."""

    mean = np.asarray(full_ffn_mean)
    hidden = rows.target.shape[-1]
    if (
        mean.shape != (hidden,)
        or not np.issubdtype(mean.dtype, np.floating)
        or not np.isfinite(mean).all()
    ):
        raise SharedQError(
            f"full_ffn_mean must be finite floating-point [{hidden}]"
        )
    target = rows.target.astype(np.float32)
    oracle = rows.oracle_shared_reconstruction.astype(np.float32)
    learned = rows.learned_shared_reconstruction.astype(np.float32)
    mean32 = mean.astype(np.float32)
    oracle_sse = np.sum(
        np.square(target - oracle, dtype=np.float32),
        axis=2,
        dtype=np.float64,
    )
    learned_sse = np.sum(
        np.square(target - learned, dtype=np.float32),
        axis=2,
        dtype=np.float64,
    )
    sst = np.sum(
        np.square(target - mean32[None, None, :], dtype=np.float32),
        axis=2,
        dtype=np.float64,
    )
    valid = rows.valid_mask
    return FullFFNTokenScores(
        sequence_id=rows.sequence_id.copy(),
        position=rows.position.copy(),
        valid_mask=valid.copy(),
        oracle_shared_sse=np.where(valid, oracle_sse, 0.0),
        learned_shared_sse=np.where(valid, learned_sse, 0.0),
        common_sst=np.where(valid, sst, 0.0),
        recomposition_sidecar_sha256=rows.recomposition_sidecar_sha256,
        shared_q_capture_manifest_sha256=(
            rows.shared_q_capture_manifest_sha256
        ),
        shared_q_centering_contract_sha256=(
            rows.shared_q_centering_contract_sha256
        ),
        shared_q_checkpoint_sha256=rows.shared_q_checkpoint_sha256,
    )


__all__ = [
    "LEGACY_SHARED_Q_INPUT_SCHEMA_VERSION",
    "SHARED_Q_CENTERING_SCHEMA_VERSION",
    "SHARED_Q_EVALUATION_SCHEMA_VERSION",
    "SHARED_Q_INPUT_SCHEMA_VERSION",
    "SHARED_Q_TOPK",
    "SHARED_Q_TOTAL_FEATURES",
    "FullFFNRecompositionRows",
    "FullFFNTokenScores",
    "SharedQCentering",
    "SharedQDistributedAux",
    "SharedQDistributedEvaluator",
    "SharedQDistributedTrainer",
    "SharedQError",
    "SharedQEvaluationEvidenceChunk",
    "SharedQEvaluationResult",
    "SharedQEvaluationRows",
    "SharedQHostMicrobatch",
    "SharedQInputReceipt",
    "SharedQLayout",
    "SharedQProcessInput",
    "build_shared_q_centering",
    "evaluate_shared_q_input",
    "full_ffn_token_scores",
    "init_shared_q_state",
    "recompose_full_ffn_rows",
    "shared_q_sae_config",
]
