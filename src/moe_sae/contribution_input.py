"""Authoritative capture-to-trainer input adaptation for Experiment 2.

This is the only production seam that turns sealed v2 wire rows into
condition-specific distributed microbatches.  In particular, condition R is
constructed here from ``(expert_id, sequence_id, randomization_key)``; callers
cannot pass precomputed bank IDs and accidentally make R identical to A.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Iterator, NamedTuple

import numpy as np

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    SCHEMA_VERSION,
    TOP_K_EXPERTS,
    ContributionBatch,
    bf16_bits_to_float32,
    canonical_sha256,
    validate_contribution_batch,
)
from moe_sae.contribution_sae import (
    ContributionSAEConfig,
    condition_bank_ids,
)
class ContributionInputError(ValueError):
    """A sealed capture cannot produce the frozen training token stream."""


class ContributionHostMicrobatch(NamedTuple):
    """One process-local, static-shape contribution microbatch."""

    expert_output: np.ndarray
    bank_ids: np.ndarray
    router_weights: np.ndarray
    valid_mask: np.ndarray
    sequence_id: np.ndarray
    position: np.ndarray


PADDING_SEQUENCE_ID = np.int64(-1)
PADDING_POSITION = np.uint16(np.iinfo(np.uint16).max)
LEGACY_CONTRIBUTION_INPUT_SCHEMA_VERSION = (
    "gemma4-contribution-training-input-v1"
)
CONTRIBUTION_INPUT_SCHEMA_VERSION = (
    "gemma4-contribution-training-input-v2-contiguous-process-lanes"
)
PROCESS_PARTITION_STRATEGY = "contiguous_process_lanes_v1"


@dataclass(frozen=True, slots=True)
class ContributionInputReceipt:
    schema_version: str
    capture_manifest_sha256: str
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
class CapacityReceipt:
    schema_version: str
    input_receipt_sha256: str
    condition: str
    randomization_key: int
    observed_maximum: int
    margin_fraction: float
    capacity: int
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dataset_manifest_sha256(dataset: Any) -> str:
    manifest = getattr(dataset, "manifest", None)
    digest = (
        manifest.get("manifest_sha256")
        if isinstance(manifest, dict)
        else getattr(dataset, "manifest_sha256", None)
    )
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ContributionInputError(
            "contribution dataset lacks a sealed manifest SHA-256"
        )
    if isinstance(manifest, dict) and manifest.get("schema_version") != SCHEMA_VERSION:
        raise ContributionInputError(
            "training input requires the exact contribution-v2 schema"
        )
    return digest


def _dataset_total_tokens(dataset: Any) -> int:
    value = getattr(dataset, "total_tokens", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContributionInputError(
            "contribution dataset total_tokens must be a positive integer"
        )
    return value


def _read_exact_range(dataset: Any, start: int, end: int) -> ContributionBatch:
    if start >= end:
        raise ContributionInputError("cannot read an empty scientific range")
    try:
        pieces = tuple(dataset.iter_token_range(start, end))
    except Exception as error:
        raise ContributionInputError(
            f"cannot read contribution range [{start},{end})"
        ) from error
    if not pieces:
        raise ContributionInputError(
            f"contribution dataset yielded no rows for [{start},{end})"
        )
    checked = tuple(validate_contribution_batch(piece) for piece in pieces)
    fields = ContributionBatch.__dataclass_fields__
    merged = ContributionBatch(
        **{
            name: np.ascontiguousarray(
                np.concatenate(
                    [np.asarray(getattr(piece, name)) for piece in checked],
                    axis=0,
                )
            )
            for name in fields
        }
    )
    if merged.tokens != end - start:
        raise ContributionInputError(
            f"dataset yielded {merged.tokens} rows for [{start},{end})"
        )
    return validate_contribution_batch(merged)


def contiguous_process_range(
    total_tokens: int,
    *,
    process_count: int,
    process_index: int,
) -> tuple[int, int]:
    """Return one disjoint, exhaustive, process-contiguous token lane."""

    for name, value in (
        ("total_tokens", total_tokens),
        ("process_count", process_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContributionInputError(f"{name} must be a positive integer")
    if not 0 <= process_index < process_count:
        raise ContributionInputError("process_index is outside process_count")
    quotient, remainder = divmod(total_tokens, process_count)
    offset = process_index * quotient + min(process_index, remainder)
    local_tokens = quotient + int(process_index < remainder)
    return offset, offset + local_tokens


def process_partition_identity(
    total_tokens: int,
    *,
    process_count: int,
    process_index: int,
) -> dict[str, Any]:
    """Canonical receipt payload for a contiguous process lane."""

    start, end = contiguous_process_range(
        total_tokens,
        process_count=process_count,
        process_index=process_index,
    )
    return {
        "strategy": PROCESS_PARTITION_STRATEGY,
        "process_count": process_count,
        "process_index": process_index,
        "range_start": start,
        "range_end": end,
        "tokens": end - start,
    }


class ContributionProcessInput:
    """Reiterable process-local view of one exact training token prefix."""

    def __init__(
        self,
        dataset: Any,
        layout: ContributionSAEConfig,
        *,
        total_tokens: int,
        global_microbatch_tokens: int = 4_096,
        microbatches_per_update: int = 4,
        process_count: int = 1,
        process_index: int = 0,
    ) -> None:
        self.dataset = dataset
        self.layout = layout
        self.capture_manifest_sha256 = _dataset_manifest_sha256(dataset)
        available = _dataset_total_tokens(dataset)
        for name, value in (
            ("total_tokens", total_tokens),
            ("global_microbatch_tokens", global_microbatch_tokens),
            ("microbatches_per_update", microbatches_per_update),
            ("process_count", process_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ContributionInputError(f"{name} must be a positive integer")
        if not 0 <= process_index < process_count:
            raise ContributionInputError("process_index is outside process_count")
        if total_tokens > available:
            raise ContributionInputError(
                "training token prefix exceeds the sealed contribution dataset"
            )
        if global_microbatch_tokens % process_count:
            raise ContributionInputError(
                "global_microbatch_tokens must divide across TPU processes"
            )
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
        # The v2 schema changes the meaning of the already receipt-bound
        # (total_tokens, process_count, process_index) tuple to one contiguous
        # lane.  Keep layout_sha256 scientific-layout-only so all process
        # receipts still agree on the canonical A/R/B/F layout.
        layout_sha256 = canonical_sha256(asdict(layout))
        unsigned = {
            "schema_version": CONTRIBUTION_INPUT_SCHEMA_VERSION,
            "capture_manifest_sha256": self.capture_manifest_sha256,
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
        self.receipt = ContributionInputReceipt(
            **unsigned,
            receipt_sha256=canonical_sha256(unsigned),
        )

    def _microbatch(self, microbatch_index: int) -> ContributionHostMicrobatch:
        if (
            isinstance(microbatch_index, bool)
            or not isinstance(microbatch_index, int)
            or microbatch_index < 0
        ):
            raise ContributionInputError(
                "microbatch_index must be a nonnegative integer"
            )
        read_start = self.process_range_start + microbatch_index * self.local_rows
        local_valid = max(
            0,
            min(self.local_rows, self.process_range_end - read_start),
        )

        expert = np.zeros(
            (self.local_rows, TOP_K_EXPERTS, HIDDEN_SIZE),
            dtype=np.float32,
        )
        weights = np.zeros(
            (self.local_rows, TOP_K_EXPERTS),
            dtype=np.float32,
        )
        banks = np.zeros(
            (self.local_rows, TOP_K_EXPERTS),
            dtype=np.int32,
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
            wire = _read_exact_range(
                self.dataset,
                read_start,
                read_start + local_valid,
            )
            decoded_expert = bf16_bits_to_float32(wire.expert_output)
            decoded_weights = bf16_bits_to_float32(wire.router_weights)
            mapped_banks = condition_bank_ids(
                wire.expert_ids,
                wire.sequence_id,
                self.layout,
            )
            expert[:local_valid] = decoded_expert
            weights[:local_valid] = decoded_weights
            banks[:local_valid] = mapped_banks
            valid[:local_valid] = True
            sequence_id[:local_valid] = wire.sequence_id
            position[:local_valid] = wire.position
        return ContributionHostMicrobatch(
            expert,
            banks,
            weights,
            valid,
            sequence_id,
            position,
        )

    def iter_updates(self) -> Iterator[tuple[ContributionHostMicrobatch, ...]]:
        microbatch_count = self.planned_updates * self.microbatches_per_update
        for update in range(self.planned_updates):
            start = update * self.microbatches_per_update
            yield tuple(
                self._microbatch(index)
                for index in range(start, start + self.microbatches_per_update)
            )
        if microbatch_count <= 0:  # pragma: no cover - constructor proves positive
            raise AssertionError("training input planned no microbatches")

    def scan_local_capacity(
        self,
        *,
        margin_fraction: float,
    ) -> CapacityReceipt:
        """Scan the exact fixed order used by one process's grouped GEMMs."""

        if self.layout.condition not in {"A", "R"}:
            raise ContributionInputError("only A/R require a bank-capacity scan")
        if (
            isinstance(margin_fraction, bool)
            or not isinstance(margin_fraction, (int, float))
            or not math.isfinite(float(margin_fraction))
            or not 0 <= float(margin_fraction) <= 1
        ):
            raise ContributionInputError(
                "capacity margin must be a finite fraction in [0,1]"
            )
        observed = 0
        for update in self.iter_updates():
            for microbatch in update:
                selected = microbatch.bank_ids[microbatch.valid_mask]
                if selected.size:
                    counts = np.bincount(
                        selected.reshape(-1),
                        minlength=self.layout.bank_count,
                    )
                    observed = max(observed, int(counts.max(initial=0)))
        if observed <= 0:
            raise ContributionInputError(
                "capacity scan observed no scientific A/R components"
            )
        capacity = math.ceil(observed * (1.0 + float(margin_fraction)))
        unsigned = {
            "schema_version": "gemma4-contribution-capacity-receipt-v1",
            "input_receipt_sha256": self.receipt.receipt_sha256,
            "condition": self.layout.condition,
            "randomization_key": self.layout.randomization_key,
            "observed_maximum": observed,
            "margin_fraction": float(margin_fraction),
            "capacity": capacity,
        }
        return CapacityReceipt(
            **unsigned,
            receipt_sha256=canonical_sha256(unsigned),
        )


def merge_capacity_receipts(
    receipts: tuple[CapacityReceipt, ...],
) -> CapacityReceipt:
    """Take the collective maximum while preserving all input identities."""

    if not receipts:
        raise ContributionInputError("capacity receipt set cannot be empty")
    first = receipts[0]
    fixed = (
        first.condition,
        first.randomization_key,
        first.margin_fraction,
    )
    if any(
        (
            item.condition,
            item.randomization_key,
            item.margin_fraction,
        )
        != fixed
        for item in receipts
    ):
        raise ContributionInputError("capacity receipts use different conditions")
    observed = max(item.observed_maximum for item in receipts)
    capacity = math.ceil(observed * (1.0 + first.margin_fraction))
    input_set_sha256 = canonical_sha256(
        sorted(item.input_receipt_sha256 for item in receipts)
    )
    unsigned = {
        "schema_version": "gemma4-contribution-capacity-receipt-v1",
        "input_receipt_sha256": input_set_sha256,
        "condition": first.condition,
        "randomization_key": first.randomization_key,
        "observed_maximum": observed,
        "margin_fraction": first.margin_fraction,
        "capacity": capacity,
    }
    return CapacityReceipt(
        **unsigned,
        receipt_sha256=canonical_sha256(unsigned),
    )


__all__ = [
    "CONTRIBUTION_INPUT_SCHEMA_VERSION",
    "LEGACY_CONTRIBUTION_INPUT_SCHEMA_VERSION",
    "PROCESS_PARTITION_STRATEGY",
    "CapacityReceipt",
    "ContributionHostMicrobatch",
    "ContributionInputError",
    "ContributionInputReceipt",
    "ContributionProcessInput",
    "PADDING_POSITION",
    "PADDING_SEQUENCE_ID",
    "contiguous_process_range",
    "merge_capacity_receipts",
    "process_partition_identity",
]
