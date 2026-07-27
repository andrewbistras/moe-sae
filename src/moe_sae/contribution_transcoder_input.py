"""Fail-closed paired input adaptation for Experiment 3 transcoders.

The routed-input overlay and Experiment-2 contribution capture are separate
immutable artifacts.  This module is the sole seam that joins them for
training and evaluation.  A join is accepted only when:

* the overlay manifest names the exact sealed contribution manifest;
* both datasets expose the same scientific token inventory; and
* ``(sequence_id, position)`` agrees elementwise for every requested range.

The resulting microbatch deliberately exposes ``routed_input`` as the only
encoder operand.  Expert outputs and router weights remain decoder-side loss
inputs; expert IDs are consumed on the host only to construct condition bank
IDs and are not carried into the accelerator batch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
import re
from typing import Any, Iterator, Mapping, NamedTuple

import numpy as np

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    SCHEMA_VERSION as TARGET_SCHEMA_VERSION,
    TOP_K_EXPERTS,
    bf16_bits_to_float32,
    canonical_sha256,
)
from moe_sae.contribution_input import (
    CapacityReceipt,
    ContributionInputError,
    ContributionProcessInput,
    PADDING_POSITION,
    PADDING_SEQUENCE_ID,
)
from moe_sae.contribution_sae import ContributionSAEConfig, condition_bank_ids

SOURCE_SCHEMA_VERSION = "gemma4-routed-input-overlay-v1"
TRANSCODER_INPUT_SCHEMA_VERSION = (
    "gemma4-contribution-transcoder-training-input-v1-paired"
)
PROCESS_PARTITION_STRATEGY = "contiguous_process_lanes_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContributionTranscoderInputError(ContributionInputError):
    """The source/target pair cannot produce the frozen token stream."""


class ContributionTranscoderHostMicrobatch(NamedTuple):
    """One process-local static-shape transcoder microbatch."""

    routed_input: np.ndarray
    expert_output: np.ndarray
    bank_ids: np.ndarray
    router_weights: np.ndarray
    valid_mask: np.ndarray
    sequence_id: np.ndarray
    position: np.ndarray


@dataclass(frozen=True, slots=True)
class ContributionTranscoderInputReceipt:
    """Immutable identity for one paired process lane."""

    schema_version: str
    source_manifest_sha256: str
    target_manifest_sha256: str
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


def _manifest(dataset: Any, *, role: str) -> Mapping[str, Any]:
    value = getattr(dataset, "manifest", None)
    if not isinstance(value, Mapping):
        raise ContributionTranscoderInputError(
            f"{role} dataset lacks its sealed manifest"
        )
    return value


def _manifest_sha256(dataset: Any, *, role: str) -> str:
    manifest = _manifest(dataset, role=role)
    value = getattr(dataset, "manifest_sha256", manifest.get("manifest_sha256"))
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContributionTranscoderInputError(
            f"{role} dataset lacks a sealed manifest SHA-256"
        )
    if manifest.get("manifest_sha256") != value:
        raise ContributionTranscoderInputError(
            f"{role} dataset manifest identity is internally inconsistent"
        )
    return value


def _total_tokens(dataset: Any, *, role: str) -> int:
    value = getattr(dataset, "total_tokens", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContributionTranscoderInputError(
            f"{role} dataset total_tokens must be a positive integer"
        )
    return value


def _validate_pair_identity(
    source_dataset: Any,
    target_dataset: Any,
) -> tuple[str, str, int]:
    source_manifest = _manifest(source_dataset, role="source")
    target_manifest = _manifest(target_dataset, role="target")
    if source_manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ContributionTranscoderInputError(
            "transcoder source requires the exact routed-input overlay schema"
        )
    if target_manifest.get("schema_version") != TARGET_SCHEMA_VERSION:
        raise ContributionTranscoderInputError(
            "transcoder target requires the exact contribution-v2 schema"
        )
    source_sha = _manifest_sha256(source_dataset, role="source")
    target_sha = _manifest_sha256(target_dataset, role="target")
    target_binding = source_manifest.get("target_manifest")
    if (
        not isinstance(target_binding, Mapping)
        or target_binding.get("manifest_sha256") != target_sha
    ):
        raise ContributionTranscoderInputError(
            "routed-input overlay does not bind the selected target manifest"
        )
    metadata = source_manifest.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("target_manifest_sha256") != target_sha
    ):
        raise ContributionTranscoderInputError(
            "routed-input overlay metadata binds a different target manifest"
        )
    source_tokens = _total_tokens(source_dataset, role="source")
    target_tokens = _total_tokens(target_dataset, role="target")
    if source_tokens != target_tokens:
        raise ContributionTranscoderInputError(
            "source and target manifests have different token inventories"
        )
    return source_sha, target_sha, source_tokens


def _read_source_range(dataset: Any, start: int, end: int) -> Any:
    try:
        pieces = tuple(dataset.iter_token_range(start, end))
    except Exception as error:
        raise ContributionTranscoderInputError(
            f"cannot read routed-input range [{start},{end})"
        ) from error
    if not pieces:
        raise ContributionTranscoderInputError(
            f"routed-input dataset yielded no rows for [{start},{end})"
        )
    routed: list[np.ndarray] = []
    sequence: list[np.ndarray] = []
    position: list[np.ndarray] = []
    for piece in pieces:
        routed.append(np.asarray(getattr(piece, "routed_input", None)))
        sequence.append(np.asarray(getattr(piece, "sequence_id", None)))
        position.append(np.asarray(getattr(piece, "position", None)))
    joined_routed = np.ascontiguousarray(np.concatenate(routed, axis=0))
    joined_sequence = np.ascontiguousarray(np.concatenate(sequence, axis=0))
    joined_position = np.ascontiguousarray(np.concatenate(position, axis=0))
    tokens = end - start
    if joined_routed.dtype != np.uint16 or joined_routed.shape != (tokens, HIDDEN_SIZE):
        raise ContributionTranscoderInputError(
            "routed_input must be BF16 wire bits [tokens,2816]"
        )
    if joined_sequence.dtype != np.int64 or joined_sequence.shape != (tokens,):
        raise ContributionTranscoderInputError(
            "source sequence_id must be int64 [tokens]"
        )
    if joined_position.dtype != np.uint16 or joined_position.shape != (tokens,):
        raise ContributionTranscoderInputError(
            "source position must be uint16 [tokens]"
        )
    decoded = bf16_bits_to_float32(joined_routed)
    if not np.isfinite(decoded).all():
        raise ContributionTranscoderInputError(
            "routed_input contains non-finite BF16 values"
        )
    return decoded, joined_sequence, joined_position


class ContributionTranscoderProcessInput:
    """Reiterable paired process-local view of one exact token prefix."""

    def __init__(
        self,
        source_dataset: Any,
        target_dataset: Any,
        layout: ContributionSAEConfig,
        *,
        total_tokens: int,
        global_microbatch_tokens: int = 4_096,
        microbatches_per_update: int = 4,
        process_count: int = 1,
        process_index: int = 0,
    ) -> None:
        (
            self.source_manifest_sha256,
            self.target_manifest_sha256,
            available,
        ) = _validate_pair_identity(source_dataset, target_dataset)
        for name, value in (
            ("total_tokens", total_tokens),
            ("global_microbatch_tokens", global_microbatch_tokens),
            ("microbatches_per_update", microbatches_per_update),
            ("process_count", process_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ContributionTranscoderInputError(
                    f"{name} must be a positive integer"
                )
        if not 0 <= process_index < process_count:
            raise ContributionTranscoderInputError(
                "process_index is outside process_count"
            )
        if total_tokens > available:
            raise ContributionTranscoderInputError(
                "training token prefix exceeds the sealed paired datasets"
            )
        if global_microbatch_tokens % process_count:
            raise ContributionTranscoderInputError(
                "global_microbatch_tokens must divide across TPU processes"
            )

        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.layout = layout
        self.total_tokens = total_tokens
        self.global_microbatch_tokens = global_microbatch_tokens
        self.microbatches_per_update = microbatches_per_update
        self.process_count = process_count
        self.process_index = process_index
        self.process_partition_strategy = PROCESS_PARTITION_STRATEGY
        quotient, remainder = divmod(total_tokens, process_count)
        self.process_range_start = process_index * quotient + min(
            process_index, remainder
        )
        self.process_range_end = (
            self.process_range_start + quotient + int(process_index < remainder)
        )
        self.local_rows = global_microbatch_tokens // process_count
        self.planned_updates = math.ceil(
            total_tokens / (global_microbatch_tokens * microbatches_per_update)
        )
        layout_sha256 = canonical_sha256(asdict(layout))
        unsigned = {
            "schema_version": TRANSCODER_INPUT_SCHEMA_VERSION,
            "source_manifest_sha256": self.source_manifest_sha256,
            "target_manifest_sha256": self.target_manifest_sha256,
            "condition": layout.condition,
            "randomization_key": layout.randomization_key,
            "layout_sha256": layout_sha256,
            "total_tokens": total_tokens,
            "global_microbatch_tokens": global_microbatch_tokens,
            "microbatches_per_update": microbatches_per_update,
            "planned_updates": self.planned_updates,
            "process_count": process_count,
            "process_index": process_index,
            "local_rows_per_microbatch": self.local_rows,
        }
        self.receipt = ContributionTranscoderInputReceipt(
            **unsigned,
            receipt_sha256=canonical_sha256(unsigned),
        )
        # Reuse the proven target reader and capacity scan.  Its output is not
        # returned directly: every capacity receipt is rebound below to the
        # paired source+target input identity.
        self._target_input = ContributionProcessInput(
            target_dataset,
            layout,
            total_tokens=total_tokens,
            global_microbatch_tokens=global_microbatch_tokens,
            microbatches_per_update=microbatches_per_update,
            process_count=process_count,
            process_index=process_index,
        )

    def _microbatch(
        self,
        microbatch_index: int,
    ) -> ContributionTranscoderHostMicrobatch:
        target = self._target_input._microbatch(microbatch_index)
        local_valid = int(np.sum(target.valid_mask, dtype=np.int64))
        routed_input = np.zeros(
            (self.local_rows, HIDDEN_SIZE),
            dtype=np.float32,
        )
        if local_valid:
            start = self.process_range_start + microbatch_index * self.local_rows
            source, source_sequence, source_position = _read_source_range(
                self.source_dataset,
                start,
                start + local_valid,
            )
            target_sequence = target.sequence_id[:local_valid]
            target_position = target.position[:local_valid]
            if not np.array_equal(source_sequence, target_sequence):
                raise ContributionTranscoderInputError(
                    "source and target sequence_id differ elementwise"
                )
            if not np.array_equal(source_position, target_position):
                raise ContributionTranscoderInputError(
                    "source and target position differ elementwise"
                )
            routed_input[:local_valid] = source
        return ContributionTranscoderHostMicrobatch(
            routed_input=routed_input,
            expert_output=target.expert_output,
            bank_ids=target.bank_ids,
            router_weights=target.router_weights,
            valid_mask=target.valid_mask,
            sequence_id=target.sequence_id,
            position=target.position,
        )

    def iter_updates(
        self,
    ) -> Iterator[tuple[ContributionTranscoderHostMicrobatch, ...]]:
        for update in range(self.planned_updates):
            start = update * self.microbatches_per_update
            yield tuple(
                self._microbatch(index)
                for index in range(start, start + self.microbatches_per_update)
            )

    def scan_local_capacity(
        self,
        *,
        margin_fraction: float,
    ) -> CapacityReceipt:
        """Run the proven target-ID scan and bind it to the paired receipt."""

        target_receipt = self._target_input.scan_local_capacity(
            margin_fraction=margin_fraction
        )
        unsigned = {
            **asdict(target_receipt),
            "input_receipt_sha256": self.receipt.receipt_sha256,
        }
        unsigned.pop("receipt_sha256")
        return replace(
            target_receipt,
            input_receipt_sha256=self.receipt.receipt_sha256,
            receipt_sha256=canonical_sha256(unsigned),
        )


def routed_input_mean(dataset: Any) -> tuple[np.ndarray, str]:
    """Derive the one fixed FP32 encoder mean from sealed FP64 statistics."""

    manifest = _manifest(dataset, role="source")
    if manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ContributionTranscoderInputError(
            "routed-input mean requires the exact overlay schema"
        )
    statistics = manifest.get("statistics")
    if not isinstance(statistics, Mapping):
        raise ContributionTranscoderInputError(
            "sealed routed-input manifest lacks statistics"
        )
    count = statistics.get("count")
    total = _total_tokens(dataset, role="source")
    if isinstance(count, bool) or not isinstance(count, int) or count != total:
        raise ContributionTranscoderInputError(
            "routed-input statistic count differs from total_tokens"
        )
    summed = np.asarray(statistics.get("sum"), dtype=np.float64)
    squared_norm = statistics.get("squared_norm")
    if (
        summed.shape != (HIDDEN_SIZE,)
        or not np.isfinite(summed).all()
        or not isinstance(squared_norm, (int, float))
        or not math.isfinite(float(squared_norm))
        or float(squared_norm) < 0
    ):
        raise ContributionTranscoderInputError(
            "routed-input statistics are malformed or non-finite"
        )
    mean = np.ascontiguousarray((summed / count).astype(np.float32))
    digest = hashlib.sha256()
    digest.update(b"gemma4-routed-input-centering-mean-v1\0")
    digest.update(mean.astype("<f4", copy=False).tobytes(order="C"))
    return mean, digest.hexdigest()


__all__ = [
    "ContributionTranscoderHostMicrobatch",
    "ContributionTranscoderInputError",
    "ContributionTranscoderInputReceipt",
    "ContributionTranscoderProcessInput",
    "PROCESS_PARTITION_STRATEGY",
    "SOURCE_SCHEMA_VERSION",
    "TRANSCODER_INPUT_SCHEMA_VERSION",
    "routed_input_mean",
]
