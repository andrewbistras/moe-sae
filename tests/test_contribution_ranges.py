from __future__ import annotations

import copy

import pytest

from moe_sae.contribution_ranges import (
    ContributionRangePlanError,
    build_contribution_range_plan,
)


def source_manifest() -> dict:
    shards = []
    for process in range(4):
        lane_start = process * 100
        for part in range(2):
            start = lane_start + part * 50
            shards.append(
                {
                    "host_id": f"p{process:05d}",
                    "range_start": start,
                    "range_end": start + 50,
                }
            )
    return {
        "manifest_sha256": "a" * 64,
        "host_count": 4,
        "total_tokens": 400,
        "shards": shards,
    }


def test_phase_boundaries_are_permanent_and_every_host_advances_equally():
    plan = build_contribution_range_plan(
        source_manifest(),
        (40, 200, 400),
        process_count=4,
        max_shard_tokens=8,
    )

    phase1 = plan.for_phase(40)
    phase2 = plan.for_phase(200)

    assert phase2[: len(phase1)] == phase1
    assert [item.tokens for item in phase1] == [8] * 4 + [2] * 4
    assert [item.process_index for item in phase1] == list(range(4)) * 2
    assert phase1[-1].range_end == 40
    # The partial two-token phase-1 shard remains immutable. Phase 2 starts a
    # new source slice rather than widening or replacing it.
    process0 = plan.for_process(200, 0)
    assert [(item.source_v1_range_start, item.source_v1_range_end) for item in process0[:3]] == [
        (0, 8),
        (8, 10),
        (10, 18),
    ]
    for process in range(4):
        selected = plan.for_process(200, process)
        assert sum(item.tokens for item in selected) == 50
        source_start = process * 100
        assert selected[0].source_v1_range_start == source_start
        assert selected[-1].source_v1_range_end == source_start + 50


def test_full_plan_is_logically_contiguous_and_uses_every_source_token_once():
    plan = build_contribution_range_plan(
        source_manifest(),
        (40, 200, 400),
        process_count=4,
        max_shard_tokens=8,
    )
    full = plan.for_phase(400)

    assert full[0].range_start == 0
    assert full[-1].range_end == 400
    assert all(
        left.range_end == right.range_start
        for left, right in zip(full, full[1:], strict=False)
    )
    source_ranges = sorted(
        (item.source_v1_range_start, item.source_v1_range_end) for item in full
    )
    assert source_ranges[0][0] == 0
    assert source_ranges[-1][1] == 400
    assert all(
        left[1] == right[0]
        for left, right in zip(source_ranges, source_ranges[1:], strict=False)
    )


def test_plan_identity_is_stable_and_binds_phase_boundaries():
    first = build_contribution_range_plan(
        source_manifest(),
        (40, 200, 400),
        process_count=4,
        max_shard_tokens=8,
    )
    same = build_contribution_range_plan(
        copy.deepcopy(source_manifest()),
        (40, 200, 400),
        process_count=4,
        max_shard_tokens=8,
    )
    changed = build_contribution_range_plan(
        source_manifest(),
        (80, 200, 400),
        process_count=4,
        max_shard_tokens=8,
    )

    assert first.plan_sha256 == same.plan_sha256
    assert first.plan_sha256 != changed.plan_sha256
    with pytest.raises(ContributionRangePlanError, match="not a frozen phase"):
        first.for_phase(41)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value.update(host_count=3),
            "host count",
        ),
        (
            lambda value: value["shards"].__setitem__(
                2,
                {
                    "host_id": "p00001",
                    "range_start": 101,
                    "range_end": 150,
                },
            ),
            "gap, overlap, or interleaving",
        ),
        (
            lambda value: value["shards"].__setitem__(
                -1,
                {
                    "host_id": "p00003",
                    "range_start": 350,
                    "range_end": 399,
                },
            ),
            "cover exactly",
        ),
    ],
)
def test_invalid_source_lane_geometry_fails_closed(mutator, message):
    value = source_manifest()
    mutator(value)
    with pytest.raises(ContributionRangePlanError, match=message):
        build_contribution_range_plan(
            value,
            (40, 200),
            process_count=4,
            max_shard_tokens=8,
        )


def test_phase_totals_must_be_monotonic_balanced_and_within_source():
    value = source_manifest()
    for totals, message in (
        ((40, 40), "strictly increasing"),
        ((41, 200), "divide evenly"),
        ((40, 404), "exceeds"),
    ):
        with pytest.raises(ContributionRangePlanError, match=message):
            build_contribution_range_plan(
                value,
                totals,
                process_count=4,
                max_shard_tokens=8,
            )
