"""Seal the citeable component-target closure report."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from moe_sae.component_closure_config import (
    CHECKPOINT_TOKENS,
    CONDITIONS,
    PANELS,
    PRIMARY_CONTRASTS,
    PROTOCOL_ID,
    SEEDS,
)
from moe_sae.contribution_artifacts import canonical_sha256

EVIDENCE_SCHEMA_VERSION = "gemma4-a512-component-closure-score-grid-v1"
REPORT_SCHEMA_VERSION = "gemma4-a512-component-closure-report-v1"
BOOTSTRAP_ALGORITHM = "hierarchical-paired-seed-packed-sequence-percentile-bootstrap-v1"


class ComponentClosureReportError(ValueError):
    """Closure evidence cannot support the frozen report estimands."""


def _finite_array(value: Any, *, name: str, nonnegative: bool = True) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 1
        or not array.size
        or not np.all(np.isfinite(array))
        or (nonnegative and np.any(array < 0))
    ):
        raise ComponentClosureReportError(f"{name} must be a finite vector")
    return array


def validate_score_grid(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ComponentClosureReportError("score grid must be a mapping")
    selected = copy.deepcopy(dict(value))
    if set(selected) != {
        "schema_version",
        "protocol_id",
        "cells",
        "costs",
        "evidence_sha256",
    }:
        raise ComponentClosureReportError("score-grid fields changed")
    claimed = selected.pop("evidence_sha256")
    if (
        selected["schema_version"] != EVIDENCE_SCHEMA_VERSION
        or selected["protocol_id"] != PROTOCOL_ID
        or not isinstance(claimed, str)
        or canonical_sha256(selected) != claimed
    ):
        raise ComponentClosureReportError("score-grid identity changed")
    cells = selected["cells"]
    expected_roles = [
        (panel, seed, checkpoint, condition)
        for panel in PANELS
        for seed in SEEDS
        for checkpoint in CHECKPOINT_TOKENS
        for condition in CONDITIONS
    ]
    roles: list[tuple[str, int, int, str]] = []
    if not isinstance(cells, list) or len(cells) != len(expected_roles):
        raise ComponentClosureReportError("score grid requires exactly 72 cells")
    paired: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for index, row in enumerate(cells):
        if not isinstance(row, Mapping) or set(row) != {
            "panel",
            "seed",
            "checkpoint_tokens",
            "condition",
            "sequence_id",
            "component_sse",
            "component_sst",
            "evaluation_result_sha256",
        }:
            raise ComponentClosureReportError(f"score cell {index} fields changed")
        role = (
            row["panel"],
            row["seed"],
            row["checkpoint_tokens"],
            row["condition"],
        )
        roles.append(role)
        sequence = np.asarray(row["sequence_id"], dtype=np.int64)
        sse = _finite_array(row["component_sse"], name=f"cell {index} SSE")
        sst = _finite_array(row["component_sst"], name=f"cell {index} SST")
        if (
            sequence.shape != sse.shape
            or sst.shape != sse.shape
            or np.any(sequence < 0)
            or np.unique(sequence).size != sequence.size
            or np.any(sst <= 0)
        ):
            raise ComponentClosureReportError(
                f"score cell {index} sequence/SSE/SST alignment changed"
            )
        key = role[:3]
        reference = paired.setdefault(key, (sequence, sst))
        if not np.array_equal(sequence, reference[0]) or not np.array_equal(
            sst, reference[1]
        ):
            raise ComponentClosureReportError(
                f"conditions are not paired on common SST for {key}"
            )
    if roles != expected_roles:
        raise ComponentClosureReportError("score cell order or roles changed")
    costs = selected["costs"]
    expected_cost_roles = [
        (panel, seed, condition)
        for panel in PANELS
        for seed in SEEDS
        for condition in CONDITIONS
    ]
    if not isinstance(costs, list) or len(costs) != len(expected_cost_roles):
        raise ComponentClosureReportError("cost grid requires 24 trajectories")
    cost_roles = []
    for index, row in enumerate(costs):
        if not isinstance(row, Mapping) or set(row) != {
            "panel",
            "seed",
            "condition",
            "accelerator_type",
            "chip_count",
            "marginal_wall_seconds",
            "chip_seconds",
            "chip_seconds_scope",
            "chip_seconds_excludes",
            "training_source_revision",
        }:
            raise ComponentClosureReportError(f"cost row {index} fields changed")
        cost_roles.append((row["panel"], row["seed"], row["condition"]))
        chip_count = row["chip_count"]
        wall = row["marginal_wall_seconds"]
        chip_seconds = row["chip_seconds"]
        if (
            isinstance(chip_count, bool)
            or not isinstance(chip_count, int)
            or chip_count <= 0
            or not isinstance(wall, (int, float))
            or not isinstance(chip_seconds, (int, float))
            or not math.isfinite(float(wall))
            or not math.isfinite(float(chip_seconds))
            or wall <= 0
            or not math.isclose(
                float(chip_seconds),
                chip_count * float(wall),
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            or not isinstance(row["chip_seconds_excludes"], list)
        ):
            raise ComponentClosureReportError(f"cost row {index} is invalid")
    if cost_roles != expected_cost_roles:
        raise ComponentClosureReportError("cost roles or ordering changed")
    return {**selected, "evidence_sha256": claimed}


def _interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _hierarchical_draws(
    cells: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    totals_sse = {
        condition: np.zeros((replicates,), dtype=np.float64) for condition in CONDITIONS
    }
    totals_sst = np.zeros((replicates,), dtype=np.float64)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(np.asarray(SEEDS), size=len(SEEDS), replace=True)
        for seed in sampled_seeds:
            reference = cells[(int(seed), "A")]
            sequence_count = len(reference["component_sst"])
            sampled = rng.integers(0, sequence_count, size=sequence_count)
            totals_sst[replicate] += np.sum(
                np.asarray(reference["component_sst"], np.float64)[sampled],
                dtype=np.float64,
            )
            for condition in CONDITIONS:
                totals_sse[condition][replicate] += np.sum(
                    np.asarray(
                        cells[(int(seed), condition)]["component_sse"],
                        np.float64,
                    )[sampled],
                    dtype=np.float64,
                )
    ev = {
        condition: 1.0 - totals_sse[condition] / totals_sst for condition in CONDITIONS
    }
    return ev


def _cost_report(costs: list[Mapping[str, Any]], panel: str) -> dict[str, Any]:
    selected = {
        (int(row["seed"]), str(row["condition"])): row
        for row in costs
        if row["panel"] == panel
    }
    rows = []
    for seed in SEEDS:
        a = selected[(seed, "A")]
        for condition in ("R", "B", "F"):
            other = selected[(seed, condition)]
            compatible = (
                a["accelerator_type"] == other["accelerator_type"]
                and a["chip_seconds_scope"] == other["chip_seconds_scope"]
                and a["chip_seconds_excludes"] == other["chip_seconds_excludes"]
            )
            rows.append(
                {
                    "seed": seed,
                    "contrast": f"A/{condition}",
                    "compatible": compatible,
                    "reason": (
                        "same_accelerator_generation_and_timing_boundary"
                        if compatible
                        else "accelerator_generation_or_timing_boundary_differs"
                    ),
                    "chip_seconds_ratio": (
                        float(a["chip_seconds"]) / float(other["chip_seconds"])
                        if compatible
                        else None
                    ),
                }
            )
    return {
        "policy": "same_accelerator_generation_and_identical_timing_scope_only",
        "comparisons": rows,
        "a_over_f_all_seeds_comparable": all(
            row["compatible"] for row in rows if row["contrast"] == "A/F"
        ),
    }


def build_closure_report(
    evidence: Mapping[str, Any],
    *,
    bootstrap_replicates: int = 20_000,
) -> dict[str, Any]:
    checked = validate_score_grid(evidence)
    if bootstrap_replicates < 100:
        raise ComponentClosureReportError("bootstrap requires at least 100 draws")
    cells = {
        (
            row["panel"],
            row["checkpoint_tokens"],
            row["seed"],
            row["condition"],
        ): row
        for row in checked["cells"]
    }
    panels: dict[str, Any] = {}
    for panel_index, panel in enumerate(PANELS):
        milestones: dict[str, Any] = {}
        for checkpoint_index, checkpoint in enumerate(CHECKPOINT_TOKENS):
            selected = {
                (seed, condition): cells[(panel, checkpoint, seed, condition)]
                for seed in SEEDS
                for condition in CONDITIONS
            }
            total_sst = sum(
                float(np.sum(selected[(seed, "A")]["component_sst"])) for seed in SEEDS
            )
            endpoints = {}
            for condition in CONDITIONS:
                total_sse = sum(
                    float(np.sum(selected[(seed, condition)]["component_sse"]))
                    for seed in SEEDS
                )
                endpoints[condition] = {
                    "sse": total_sse,
                    "sst": total_sst,
                    "nmse": total_sse / total_sst,
                    "ev": 1.0 - total_sse / total_sst,
                }
            draws = _hierarchical_draws(
                selected,
                replicates=bootstrap_replicates,
                random_seed=20_260_729 + panel_index * 10 + checkpoint_index,
            )
            effects = {}
            for contrast in PRIMARY_CONTRASTS:
                left, right = contrast.split("-")
                values = draws[left] - draws[right]
                low, high = _interval(values)
                effects[contrast] = {
                    "estimate": endpoints[left]["ev"] - endpoints[right]["ev"],
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_seeds": list(SEEDS),
                    "bootstrap_unit": "packed_sequence_within_seed",
                }
            denominator = endpoints["F"]["ev"] - endpoints["B"]["ev"]
            denominator_draws = draws["F"] - draws["B"]
            numerator = endpoints["A"]["ev"] - endpoints["B"]["ev"]
            denominator_interval = _interval(denominator_draws)
            valid_draws = denominator_draws > 0
            g_status = (
                "undefined_nonpositive_denominator"
                if denominator <= 0
                else (
                    "unstable_interval_includes_zero"
                    if denominator_interval[0] <= 0 <= denominator_interval[1]
                    else "defined"
                )
            )
            g_draws = (
                100.0
                * (draws["A"][valid_draws] - draws["B"][valid_draws])
                / denominator_draws[valid_draws]
            )
            g_interval = (
                (None, None)
                if g_draws.size < max(100, bootstrap_replicates // 2)
                else _interval(g_draws)
            )
            milestones[str(checkpoint)] = {
                "component_endpoint": endpoints,
                "contrasts": effects,
                "g": {
                    "definition": "100*(A-B)/(F-B)",
                    "estimate": (
                        None if denominator <= 0 else 100.0 * numerator / denominator
                    ),
                    "status": g_status,
                    "denominator_estimate": denominator,
                    "denominator_ci95": list(denominator_interval),
                    "ci95_low": g_interval[0],
                    "ci95_high": g_interval[1],
                    "positive_denominator_draw_fraction": float(np.mean(valid_draws)),
                },
            }
        panels[panel] = {
            "milestones": milestones,
            "cost_compatibility": _cost_report(checked["costs"], panel),
        }
    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "evidence_sha256": checked["evidence_sha256"],
        "primary_endpoint": "component_target_explained_variance",
        "bootstrap": {
            "algorithm": BOOTSTRAP_ALGORITHM,
            "replicates": bootstrap_replicates,
            "confidence_level": 0.95,
        },
        "geometry": {
            "A": "128x512_true_expert_id",
            "R": "128x512_per_sequence_randomized_expert_id",
            "B": "8x512_router_rank",
            "F": "8x8192_router_rank",
            "A_R_F_total_features": 65_536,
            "A_R_B_candidates_per_token": 4_096,
            "F_candidates_per_token": 65_536,
            "all_selected_slots_per_token": 32,
        },
        "panels": panels,
    }
    return {**body, "report_sha256": canonical_sha256(body)}


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# A512 component-closure report",
        "",
        f"Evidence: `{report['evidence_sha256']}`",
        "",
        "Primary endpoint: component-target explained variance. Intervals use "
        "paired hierarchical seed/packed-sequence bootstrap.",
        "",
    ]
    for panel in PANELS:
        lines.extend([f"## {panel}", ""])
        lines.append(
            "| tokens | A EV | R EV | B EV | F EV | A-R | A-B | A-F | F-B | G |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for checkpoint in CHECKPOINT_TOKENS:
            row = report["panels"][panel]["milestones"][str(checkpoint)]
            endpoint = row["component_endpoint"]
            effects = row["contrasts"]
            g = row["g"]
            values = [
                f"{checkpoint:,}",
                *(f"{endpoint[c]['ev']:.5f}" for c in CONDITIONS),
                *(f"{effects[c]['estimate']:.5f}" for c in PRIMARY_CONTRASTS),
                (
                    g["status"]
                    if g["estimate"] is None
                    else f"{g['estimate']:.1f}% ({g['status']})"
                ),
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines.extend(
            [
                "",
                "Cost ratios are emitted only where accelerator generation and "
                "timing scope/exclusions match exactly.",
                "",
            ]
        )
    return "\n".join(lines)


def load_score_grid(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComponentClosureReportError("cannot read score grid") from error
    return validate_score_grid(value)


__all__ = [
    "BOOTSTRAP_ALGORITHM",
    "EVIDENCE_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "ComponentClosureReportError",
    "build_closure_report",
    "load_score_grid",
    "render_markdown",
    "validate_score_grid",
]
