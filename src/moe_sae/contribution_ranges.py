"""Monotonic multi-host source-range planning for Experiment 2.

The sealed aggregate-v1 manifest is ordered by host lanes: each TPU process
owns one contiguous slice of the scientific token stream.  Experiment 2 must
exercise every host during its small stages while remaining exactly aligned
to those source tokens and reusing every already-persisted v2 shard when a
stage grows.

This module gives every source lane the same phase-local prefix, splits each
increment at the v2 shard limit, and interleaves corresponding lane chunks in
logical v2 order.  Phase boundaries are themselves permanent shard
boundaries.  Consequently a 100k manifest is a byte-for-byte subset of the
2M plan even when 100k is not divisible by the shard size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Sequence

from moe_sae.contribution_artifacts import (
    MAX_SHARD_TOKENS,
    canonical_sha256,
)


class ContributionRangePlanError(ValueError):
    """A source manifest cannot support the requested monotonic phase plan."""


@dataclass(frozen=True, slots=True)
class SourceLane:
    process_index: int
    host_id: str
    source_start: int
    source_end: int

    @property
    def tokens(self) -> int:
        return self.source_end - self.source_start


@dataclass(frozen=True, slots=True)
class ContributionRangeAssignment:
    """One immutable v2 logical shard mapped to one exact v1 source slice."""

    ordinal: int
    process_index: int
    host_id: str
    range_start: int
    range_end: int
    source_v1_range_start: int
    source_v1_range_end: int
    phase_total_tokens: int

    @property
    def tokens(self) -> int:
        return self.range_end - self.range_start

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContributionRangePlan:
    source_manifest_sha256: str
    source_total_tokens: int
    process_count: int
    max_shard_tokens: int
    phase_totals: tuple[int, ...]
    lanes: tuple[SourceLane, ...]
    assignments: tuple[ContributionRangeAssignment, ...]
    plan_sha256: str

    def for_phase(self, total_tokens: int) -> tuple[ContributionRangeAssignment, ...]:
        if total_tokens not in self.phase_totals:
            raise ContributionRangePlanError(
                f"{total_tokens} is not a frozen phase token total"
            )
        selected = tuple(
            assignment
            for assignment in self.assignments
            if assignment.range_end <= total_tokens
        )
        if not selected or selected[-1].range_end != total_tokens:
            raise ContributionRangePlanError(
                f"range plan does not end exactly at {total_tokens}"
            )
        return selected

    def for_process(
        self,
        total_tokens: int,
        process_index: int,
    ) -> tuple[ContributionRangeAssignment, ...]:
        if not 0 <= process_index < self.process_count:
            raise ContributionRangePlanError("process_index is outside the plan")
        return tuple(
            assignment
            for assignment in self.for_phase(total_tokens)
            if assignment.process_index == process_index
        )


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContributionRangePlanError(f"{name} must be a positive integer")
    return value


def _source_lanes(
    source_manifest: Mapping[str, Any],
    *,
    process_count: int,
) -> tuple[SourceLane, ...]:
    if not isinstance(source_manifest, Mapping):
        raise ContributionRangePlanError("source manifest must be a mapping")
    shards = source_manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ContributionRangePlanError("source manifest has no shard inventory")
    if source_manifest.get("host_count") != process_count:
        raise ContributionRangePlanError(
            "source manifest host count differs from the TPU process count"
        )

    by_host: dict[str, list[tuple[int, int]]] = {}
    for entry in shards:
        if not isinstance(entry, Mapping):
            raise ContributionRangePlanError("source shard entry is not a mapping")
        host = entry.get("host_id")
        start = entry.get("range_start")
        end = entry.get("range_end")
        if (
            not isinstance(host, str)
            or not host
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ContributionRangePlanError("source shard identity is invalid")
        by_host.setdefault(host, []).append((start, end))
    if len(by_host) != process_count:
        raise ContributionRangePlanError(
            "source manifest does not contain one lane per process"
        )

    raw_lanes: list[tuple[str, int, int]] = []
    for host, ranges in by_host.items():
        ordered = sorted(ranges)
        cursor = ordered[0][0]
        lane_start = cursor
        for start, end in ordered:
            if start != cursor:
                raise ContributionRangePlanError(
                    f"source host {host!r} does not own one contiguous lane"
                )
            cursor = end
        raw_lanes.append((host, lane_start, cursor))
    raw_lanes.sort(key=lambda item: (item[1], item[0]))

    source_total = _positive_int(
        "source_manifest.total_tokens",
        source_manifest.get("total_tokens"),
    )
    cursor = 0
    lanes: list[SourceLane] = []
    for process_index, (host, start, end) in enumerate(raw_lanes):
        if start != cursor:
            raise ContributionRangePlanError(
                "source host lanes contain a gap, overlap, or interleaving"
            )
        lanes.append(
            SourceLane(
                process_index=process_index,
                host_id=host,
                source_start=start,
                source_end=end,
            )
        )
        cursor = end
    if cursor != source_total:
        raise ContributionRangePlanError(
            "source host lanes do not cover exactly total_tokens"
        )
    lane_tokens = {lane.tokens for lane in lanes}
    if len(lane_tokens) != 1:
        raise ContributionRangePlanError(
            "Experiment-2 staged interleaving requires equal source host lanes"
        )
    return tuple(lanes)


def build_contribution_range_plan(
    source_manifest: Mapping[str, Any],
    phase_totals: Sequence[int],
    *,
    process_count: int,
    max_shard_tokens: int = MAX_SHARD_TOKENS,
) -> ContributionRangePlan:
    """Build the one monotonic logical→source mapping for a split."""

    processes = _positive_int("process_count", process_count)
    shard_limit = _positive_int("max_shard_tokens", max_shard_tokens)
    if shard_limit > MAX_SHARD_TOKENS:
        raise ContributionRangePlanError(
            f"max_shard_tokens exceeds the v2 limit {MAX_SHARD_TOKENS}"
        )
    if not isinstance(phase_totals, Sequence) or isinstance(
        phase_totals, (str, bytes)
    ):
        raise ContributionRangePlanError("phase_totals must be a sequence")
    totals = tuple(
        _positive_int(f"phase_totals[{index}]", value)
        for index, value in enumerate(phase_totals)
    )
    if not totals or tuple(sorted(set(totals))) != totals:
        raise ContributionRangePlanError(
            "phase_totals must be strictly increasing and unique"
        )
    lanes = _source_lanes(source_manifest, process_count=processes)
    source_manifest_sha256 = source_manifest.get("manifest_sha256")
    if (
        not isinstance(source_manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256) is None
    ):
        raise ContributionRangePlanError(
            "source manifest identity must be a lowercase SHA-256"
        )
    source_total = int(source_manifest["total_tokens"])
    if totals[-1] > source_total:
        raise ContributionRangePlanError(
            "final phase exceeds the source manifest token count"
        )
    if any(total % processes for total in totals):
        raise ContributionRangePlanError(
            "every phase total must divide evenly across source host lanes"
        )
    local_totals = tuple(total // processes for total in totals)
    if local_totals[-1] > lanes[0].tokens:
        raise ContributionRangePlanError(
            "phase-local quota exceeds a source host lane"
        )

    assignments: list[ContributionRangeAssignment] = []
    logical_cursor = 0
    prior_local_total = 0
    lane_offsets = [0 for _ in lanes]
    for phase_total, local_total in zip(totals, local_totals, strict=True):
        increment = local_total - prior_local_total
        chunk_lengths: list[int] = []
        remaining = increment
        while remaining:
            length = min(shard_limit, remaining)
            chunk_lengths.append(length)
            remaining -= length
        for length in chunk_lengths:
            for lane in lanes:
                source_start = lane.source_start + lane_offsets[lane.process_index]
                source_end = source_start + length
                assignment = ContributionRangeAssignment(
                    ordinal=len(assignments),
                    process_index=lane.process_index,
                    host_id=lane.host_id,
                    range_start=logical_cursor,
                    range_end=logical_cursor + length,
                    source_v1_range_start=source_start,
                    source_v1_range_end=source_end,
                    phase_total_tokens=phase_total,
                )
                assignments.append(assignment)
                logical_cursor += length
                lane_offsets[lane.process_index] += length
        if logical_cursor != phase_total:
            raise ContributionRangePlanError(
                "phase plan failed to terminate at its exact logical token total"
            )
        if any(offset != local_total for offset in lane_offsets):
            raise ContributionRangePlanError(
                "phase plan failed to advance every source lane equally"
            )
        prior_local_total = local_total

    unsigned = {
        "schema_version": "gemma4-contribution-range-plan-v1",
        "source_manifest_sha256": source_manifest_sha256,
        "source_total_tokens": source_total,
        "process_count": processes,
        "max_shard_tokens": shard_limit,
        "phase_totals": list(totals),
        "lanes": [asdict(lane) for lane in lanes],
        "assignments": [assignment.to_dict() for assignment in assignments],
    }
    plan_sha = canonical_sha256(unsigned)
    return ContributionRangePlan(
        source_manifest_sha256=source_manifest_sha256,
        source_total_tokens=source_total,
        process_count=processes,
        max_shard_tokens=shard_limit,
        phase_totals=totals,
        lanes=lanes,
        assignments=tuple(assignments),
        plan_sha256=plan_sha,
    )


__all__ = [
    "ContributionRangeAssignment",
    "ContributionRangePlan",
    "ContributionRangePlanError",
    "SourceLane",
    "build_contribution_range_plan",
]
