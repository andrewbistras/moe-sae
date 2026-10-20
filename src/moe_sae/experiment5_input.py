"""Sealed Experiment-4 capture adapter for Experiment 5.

The adapter derives the normalized shared branch from the already-captured raw
shared output and joins the routed-input overlay only for the transcoder
panel.  It never creates or migrates a capture.  All process lanes are
contiguous and resume seeks by optimizer update, matching the existing
production operator.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Literal, Mapping, NamedTuple

import numpy as np

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    TOP_K_EXPERTS,
    bf16_bits_to_float32,
    canonical_sha256,
)
from moe_sae.contribution_input import (
    PADDING_POSITION,
    PADDING_SEQUENCE_ID,
    PROCESS_PARTITION_STRATEGY,
    _read_exact_range,
    contiguous_process_range,
)
from moe_sae.contribution_recomposition import (
    RecompositionSidecar,
    decode_recomposition_sidecar,
    maxtext_rmsnorm_bf16,
)
from moe_sae.contribution_transcoder_input import (
    _read_source_range,
    _validate_pair_identity,
)
from moe_sae.experiment5_hybrid import (
    Experiment5Layout,
    Experiment5Recomposition,
    experiment5_bank_ids,
    numpy_recompose_experiment5_full_ffn,
)

EXPERIMENT5_INPUT_SCHEMA_VERSION = (
    "gemma4-routed-shared-hybrid-input-v1-contiguous-process-lanes"
)
EXPERIMENT5_PREPROCESSING_SCHEMA_VERSION = (
    "gemma4-routed-shared-hybrid-preprocessing-v1"
)


class Experiment5InputError(ValueError):
    """The sealed parent artifacts cannot produce the Experiment-5 stream."""


class Experiment5HostMicrobatch(NamedTuple):
    routed_input: np.ndarray
    expert_output: np.ndarray
    shared_normalized: np.ndarray
    bank_ids: np.ndarray
    router_weights: np.ndarray
    valid_mask: np.ndarray
    sequence_id: np.ndarray
    position: np.ndarray


@dataclass(frozen=True, slots=True)
class Experiment5InputReceipt:
    schema_version: str
    panel: str
    target_manifest_sha256: str
    source_manifest_sha256: str | None
    recomposition_sidecar_sha256: str
    condition: str
    randomization_key: int
    layout_sha256: str
    total_tokens: int
    global_microbatch_tokens: int
    microbatches_per_update: int
    planned_updates: int
    process_count: int
    process_index: int
    local_rows_per_microbatch: int
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Experiment5PreprocessingReceipt:
    schema_version: str
    input_receipt_sha256: str
    process_index: int
    valid_tokens: int
    full_target_sum: tuple[float, ...]
    full_target_squared_norm_sum: float
    observed_maximum_bank_rows: int | None
    routed_bank_component_counts: tuple[int, ...] | None
    routed_bank_component_counts_by_training_tokens: (
        tuple[tuple[int, tuple[int, ...]], ...] | None
    )
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _manifest_sha(dataset: Any, *, role: str) -> str:
    manifest = getattr(dataset, "manifest", None)
    digest = getattr(dataset, "manifest_sha256", None)
    if not isinstance(manifest, Mapping) or not isinstance(digest, str):
        raise Experiment5InputError(f"{role} dataset lacks a sealed manifest")
    if manifest.get("manifest_sha256") != digest:
        raise Experiment5InputError(f"{role} manifest identity is inconsistent")
    return digest


def _sidecar_identity(sidecar: RecompositionSidecar) -> str:
    value = sidecar.sidecar_sha256
    if not isinstance(value, str) or len(value) != 64:
        raise Experiment5InputError("recomposition sidecar identity is invalid")
    return value


class Experiment5ProcessInput:
    """Reiterable process-local view of one exact Experiment-5 token prefix."""

    def __init__(
        self,
        target_dataset: Any,
        recomposition_sidecar: Mapping[str, Any],
        layout: Experiment5Layout,
        *,
        panel: Literal["sae", "transcoder"],
        total_tokens: int,
        source_dataset: Any | None = None,
        global_microbatch_tokens: int = 4_096,
        microbatches_per_update: int = 4,
        process_count: int = 1,
        process_index: int = 0,
    ) -> None:
        if panel not in {"sae", "transcoder"}:
            raise Experiment5InputError("panel must be sae or transcoder")
        if panel == "transcoder":
            if source_dataset is None:
                raise Experiment5InputError(
                    "transcoder input requires the aligned routed-input overlay"
                )
            source_sha, target_sha, available = _validate_pair_identity(
                source_dataset,
                target_dataset,
            )
        else:
            if source_dataset is not None:
                raise Experiment5InputError(
                    "SAE input must not receive a routed-input overlay"
                )
            target_sha = _manifest_sha(target_dataset, role="target")
            source_sha = None
            available = getattr(target_dataset, "total_tokens", None)
        if isinstance(available, bool) or not isinstance(available, int):
            raise Experiment5InputError("target token inventory is invalid")
        for name, value in (
            ("total_tokens", total_tokens),
            ("global_microbatch_tokens", global_microbatch_tokens),
            ("microbatches_per_update", microbatches_per_update),
            ("process_count", process_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise Experiment5InputError(f"{name} must be a positive integer")
        if not 0 <= process_index < process_count:
            raise Experiment5InputError("process_index is outside process_count")
        if total_tokens > available:
            raise Experiment5InputError(
                "requested token prefix exceeds the sealed parent capture"
            )
        if global_microbatch_tokens % process_count:
            raise Experiment5InputError(
                "global_microbatch_tokens must divide across TPU processes"
            )

        self.target_dataset = target_dataset
        self.source_dataset = source_dataset
        self.sidecar_value = dict(recomposition_sidecar)
        self.sidecar = decode_recomposition_sidecar(recomposition_sidecar)
        self.core_recomposition = Experiment5Recomposition(
            routed_scale=bf16_bits_to_float32(
                self.sidecar.routed_rmsnorm_scale_bf16_bits
            ),
            outer_scale=bf16_bits_to_float32(
                self.sidecar.outer_ffn_rmsnorm_scale_bf16_bits
            ),
            epsilon=float(self.sidecar.epsilon),
        )
        self.layout = layout
        self.panel = panel
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
        unsigned = {
            "schema_version": EXPERIMENT5_INPUT_SCHEMA_VERSION,
            "panel": panel,
            "target_manifest_sha256": target_sha,
            "source_manifest_sha256": source_sha,
            "recomposition_sidecar_sha256": _sidecar_identity(self.sidecar),
            "condition": layout.condition,
            "randomization_key": int(layout.randomization_key),
            "layout_sha256": canonical_sha256(asdict(layout)),
            "total_tokens": total_tokens,
            "global_microbatch_tokens": global_microbatch_tokens,
            "microbatches_per_update": microbatches_per_update,
            "planned_updates": self.planned_updates,
            "process_count": process_count,
            "process_index": process_index,
            "local_rows_per_microbatch": self.local_rows,
        }
        self.receipt = Experiment5InputReceipt(
            **unsigned,
            receipt_sha256=canonical_sha256(unsigned),
        )

    def _range(self, microbatch_index: int) -> tuple[int, int]:
        if (
            isinstance(microbatch_index, bool)
            or not isinstance(microbatch_index, int)
            or microbatch_index < 0
        ):
            raise Experiment5InputError(
                "microbatch_index must be a nonnegative integer"
            )
        start = self.process_range_start + microbatch_index * self.local_rows
        valid = max(0, min(self.local_rows, self.process_range_end - start))
        return start, valid

    def _wire(self, microbatch_index: int) -> tuple[Any | None, int]:
        start, valid = self._range(microbatch_index)
        if valid == 0:
            return None, 0
        return (
            _read_exact_range(self.target_dataset, start, start + valid),
            valid,
        )

    def _microbatch(self, microbatch_index: int) -> Experiment5HostMicrobatch:
        start, local_valid = self._range(microbatch_index)
        source = np.zeros((self.local_rows, HIDDEN_SIZE), dtype=np.float32)
        expert = np.zeros(
            (self.local_rows, TOP_K_EXPERTS, HIDDEN_SIZE),
            dtype=np.float32,
        )
        shared = np.zeros((self.local_rows, HIDDEN_SIZE), dtype=np.float32)
        banks = np.zeros((self.local_rows, TOP_K_EXPERTS), dtype=np.int32)
        weights = np.zeros(
            (self.local_rows, TOP_K_EXPERTS),
            dtype=np.float32,
        )
        valid = np.zeros((self.local_rows,), dtype=np.bool_)
        sequence = np.full(
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
            wire = _read_exact_range(
                self.target_dataset,
                start,
                start + local_valid,
            )
            expert[:local_valid] = bf16_bits_to_float32(wire.expert_output)
            weights[:local_valid] = bf16_bits_to_float32(wire.router_weights)
            _, normalized_shared_bits = maxtext_rmsnorm_bf16(
                wire.shared_output,
                self.sidecar.shared_rmsnorm_scale_bf16_bits,
                epsilon=float(self.sidecar.epsilon),
            )
            shared[:local_valid] = bf16_bits_to_float32(normalized_shared_bits)
            if self.layout.is_hybrid:
                banks[:local_valid] = experiment5_bank_ids(
                    wire.expert_ids,
                    wire.sequence_id,
                    self.layout,
                )
            else:
                # Global B/F do not consume this field.  Retaining exact IDs
                # keeps the common batch auditable without inventing ranks.
                banks[:local_valid] = wire.expert_ids.astype(np.int32)
            if self.panel == "transcoder":
                assert self.source_dataset is not None
                routed, source_sequence, source_position = _read_source_range(
                    self.source_dataset,
                    start,
                    start + local_valid,
                )
                if not np.array_equal(source_sequence, wire.sequence_id) or not np.array_equal(
                    source_position, wire.position
                ):
                    raise Experiment5InputError(
                        "source/target token identities differ in requested range"
                    )
                source[:local_valid] = routed
            valid[:local_valid] = True
            sequence[:local_valid] = wire.sequence_id
            position[:local_valid] = wire.position
        return Experiment5HostMicrobatch(
            source,
            expert,
            shared,
            banks,
            weights,
            valid,
            sequence,
            position,
        )

    def iter_updates(self) -> Iterator[tuple[Experiment5HostMicrobatch, ...]]:
        yield from self.iter_updates_from(0)

    def iter_updates_from(
        self,
        start_update: int,
    ) -> Iterator[tuple[Experiment5HostMicrobatch, ...]]:
        if (
            isinstance(start_update, bool)
            or not isinstance(start_update, int)
            or not 0 <= start_update <= self.planned_updates
        ):
            raise Experiment5InputError(
                "start_update must lie in the fixed update plan"
            )
        for update in range(start_update, self.planned_updates):
            first = update * self.microbatches_per_update
            yield tuple(
                self._microbatch(index)
                for index in range(first, first + self.microbatches_per_update)
            )

    def scan_local_preprocessing(self) -> Experiment5PreprocessingReceipt:
        """Scan exact target rows for full-target mean and H/R capacity.

        This intentionally combines the nonlinear full-target statistics and
        capacity passes.  It is run once per trajectory before compilation;
        no training step performs object-store I/O.
        """

        full_sum = np.zeros((HIDDEN_SIZE,), dtype=np.float64)
        squared_norm_sum = np.float64(0.0)
        count = 0
        observed = 0
        routed_bank_component_counts = (
            np.zeros((self.layout.num_experts,), dtype=np.int64)
            if self.layout.is_hybrid
            else None
        )
        effective_batch = (
            self.global_microbatch_tokens * self.microbatches_per_update
        )
        local_effective_batch = effective_batch // self.process_count
        local_lane_tokens = self.process_range_end - self.process_range_start
        checkpoint_targets: list[tuple[int, int]] = []
        for nominal in (2_000_000, 5_000_000, self.total_tokens):
            if nominal > self.total_tokens:
                continue
            actual = min(
                math.ceil(nominal / effective_batch) * effective_batch,
                self.total_tokens,
            )
            step = math.ceil(nominal / effective_batch)
            local_tokens = min(step * local_effective_batch, local_lane_tokens)
            pair = (actual, local_tokens)
            if pair not in checkpoint_targets:
                checkpoint_targets.append(pair)
        checkpoint_targets.sort()
        routed_snapshots: list[tuple[int, tuple[int, ...]]] = []
        microbatch_count = math.ceil(
            (self.process_range_end - self.process_range_start) / self.local_rows
        )
        for index in range(microbatch_count):
            start = self.process_range_start + index * self.local_rows
            rows = min(self.local_rows, self.process_range_end - start)
            if rows <= 0:
                break
            wire = _read_exact_range(self.target_dataset, start, start + rows)
            _, normalized_shared_bits = maxtext_rmsnorm_bf16(
                wire.shared_output,
                self.sidecar.shared_rmsnorm_scale_bf16_bits,
                epsilon=float(self.sidecar.epsilon),
            )
            target = numpy_recompose_experiment5_full_ffn(
                bf16_bits_to_float32(wire.expert_output),
                bf16_bits_to_float32(normalized_shared_bits),
                bf16_bits_to_float32(wire.router_weights),
                self.core_recomposition,
            )
            full_sum += np.sum(target, axis=0, dtype=np.float64)
            squared_norm_sum += np.sum(
                np.square(target.astype(np.float64)),
                dtype=np.float64,
            )
            count += rows
            if self.layout.is_hybrid:
                selected = experiment5_bank_ids(
                    wire.expert_ids,
                    wire.sequence_id,
                    self.layout,
                )
                counts = np.bincount(
                    selected.reshape(-1),
                    minlength=self.layout.num_experts,
                )
                observed = max(observed, int(counts.max(initial=0)))
                assert routed_bank_component_counts is not None
                routed_bank_component_counts += counts.astype(
                    np.int64,
                    copy=False,
                )
                while (
                    len(routed_snapshots) < len(checkpoint_targets)
                    and count >= checkpoint_targets[len(routed_snapshots)][1]
                ):
                    actual_tokens, local_target = checkpoint_targets[
                        len(routed_snapshots)
                    ]
                    if count != local_target:
                        raise Experiment5InputError(
                            "preprocessing chunk crossed a training milestone"
                        )
                    routed_snapshots.append(
                        (
                            actual_tokens,
                            tuple(
                                int(item)
                                for item in routed_bank_component_counts
                            ),
                        )
                    )
        expected = self.process_range_end - self.process_range_start
        if count != expected or count <= 0:
            raise Experiment5InputError(
                f"preprocessing scanned {count} rows, expected {expected}"
            )
        if self.layout.is_hybrid and observed <= 0:
            raise Experiment5InputError("capacity scan observed no routed rows")
        body = {
            "schema_version": EXPERIMENT5_PREPROCESSING_SCHEMA_VERSION,
            "input_receipt_sha256": self.receipt.receipt_sha256,
            "process_index": self.process_index,
            "valid_tokens": count,
            "full_target_sum": tuple(float(item) for item in full_sum),
            "full_target_squared_norm_sum": float(squared_norm_sum),
            "observed_maximum_bank_rows": (
                observed if self.layout.is_hybrid else None
            ),
            "routed_bank_component_counts": (
                tuple(int(item) for item in routed_bank_component_counts)
                if routed_bank_component_counts is not None
                else None
            ),
            "routed_bank_component_counts_by_training_tokens": (
                tuple(routed_snapshots)
                if routed_bank_component_counts is not None
                else None
            ),
        }
        return Experiment5PreprocessingReceipt(
            **body,
            receipt_sha256=canonical_sha256(body),
        )


def merge_experiment5_preprocessing(
    receipts: tuple[Experiment5PreprocessingReceipt, ...],
    *,
    margin_fraction: float,
) -> dict[str, Any]:
    """Merge process-ordered statistics into one exact training centering."""

    if not receipts:
        raise Experiment5InputError("preprocessing receipt set cannot be empty")
    if [item.process_index for item in receipts] != list(range(len(receipts))):
        raise Experiment5InputError("preprocessing receipts are not process ordered")
    if len({item.input_receipt_sha256 for item in receipts}) != len(receipts):
        raise Experiment5InputError("preprocessing receipts contain duplicate lanes")
    if not math.isfinite(float(margin_fraction)) or not 0 <= margin_fraction <= 1:
        raise Experiment5InputError("capacity margin must be in [0,1]")
    tokens = sum(item.valid_tokens for item in receipts)
    full_sum = np.sum(
        np.asarray([item.full_target_sum for item in receipts], dtype=np.float64),
        axis=0,
        dtype=np.float64,
    )
    squared = sum(
        item.full_target_squared_norm_sum for item in receipts
    )
    mean = full_sum / tokens
    sst = squared - float(np.dot(full_sum, full_sum) / tokens)
    if (
        tokens <= 0
        or mean.shape != (HIDDEN_SIZE,)
        or not np.isfinite(mean).all()
        or not math.isfinite(sst)
        or sst <= 0
    ):
        raise Experiment5InputError("merged full-target statistics are invalid")
    maxima = [
        item.observed_maximum_bank_rows
        for item in receipts
        if item.observed_maximum_bank_rows is not None
    ]
    capacity = (
        math.ceil(max(maxima) * (1.0 + margin_fraction))
        if maxima
        else None
    )
    routed_count_rows = [
        item.routed_bank_component_counts
        for item in receipts
        if item.routed_bank_component_counts is not None
    ]
    if routed_count_rows and len(routed_count_rows) != len(receipts):
        raise Experiment5InputError(
            "preprocessing receipts mix hybrid and global layouts"
        )
    routed_counts = (
        np.sum(
            np.asarray(routed_count_rows, dtype=np.int64),
            axis=0,
            dtype=np.int64,
        )
        if routed_count_rows
        else None
    )
    if routed_counts is not None and (
        routed_counts.shape != (128,)
        or int(routed_counts.sum()) != tokens * TOP_K_EXPERTS
        or np.any(routed_counts < 0)
    ):
        raise Experiment5InputError(
            "merged routed-bank component counts are inconsistent"
        )
    routed_snapshot_rows = [
        item.routed_bank_component_counts_by_training_tokens
        for item in receipts
        if item.routed_bank_component_counts_by_training_tokens is not None
    ]
    if routed_snapshot_rows and len(routed_snapshot_rows) != len(receipts):
        raise Experiment5InputError(
            "preprocessing receipts mix hybrid and global milestone counts"
        )
    merged_snapshots: tuple[tuple[int, tuple[int, ...]], ...] | None = None
    if routed_snapshot_rows:
        milestone_keys = tuple(
            int(item[0]) for item in routed_snapshot_rows[0]
        )
        if any(
            tuple(int(item[0]) for item in row) != milestone_keys
            for row in routed_snapshot_rows
        ):
            raise Experiment5InputError(
                "routed-bank milestone keys differ across processes"
            )
        merged_rows: list[tuple[int, tuple[int, ...]]] = []
        for index, actual_tokens in enumerate(milestone_keys):
            counts = np.sum(
                np.asarray(
                    [row[index][1] for row in routed_snapshot_rows],
                    dtype=np.int64,
                ),
                axis=0,
                dtype=np.int64,
            )
            if (
                counts.shape != (128,)
                or np.any(counts < 0)
                or int(counts.sum()) != actual_tokens * TOP_K_EXPERTS
            ):
                raise Experiment5InputError(
                    "routed-bank milestone counts are inconsistent"
                )
            merged_rows.append(
                (actual_tokens, tuple(int(item) for item in counts))
            )
        merged_snapshots = tuple(merged_rows)
    mean32 = np.ascontiguousarray(mean.astype(np.float32))
    digest = hashlib.sha256()
    digest.update(b"gemma4-experiment5-full-target-mean-v1\0")
    digest.update(mean32.astype("<f4", copy=False).tobytes(order="C"))
    body = {
        "schema_version": "gemma4-routed-shared-hybrid-preprocessing-merge-v1",
        "receipt_set_sha256": canonical_sha256(
            [item.receipt_sha256 for item in receipts]
        ),
        "tokens": tokens,
        "full_target_mean": mean32,
        "full_target_mean_sha256": digest.hexdigest(),
        "full_target_sst": float(sst),
        "bank_capacity": capacity,
        "routed_bank_component_counts": (
            tuple(int(item) for item in routed_counts)
            if routed_counts is not None
            else None
        ),
        "routed_bank_component_counts_by_training_tokens": merged_snapshots,
    }
    return body


__all__ = [
    "EXPERIMENT5_INPUT_SCHEMA_VERSION",
    "EXPERIMENT5_PREPROCESSING_SCHEMA_VERSION",
    "Experiment5HostMicrobatch",
    "Experiment5InputError",
    "Experiment5InputReceipt",
    "Experiment5PreprocessingReceipt",
    "Experiment5ProcessInput",
    "merge_experiment5_preprocessing",
]
