"""Seal the citeable component-target closure report."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
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

EVIDENCE_SCHEMA_VERSION = "gemma4-a512-component-closure-score-grid-v2"
REPORT_SCHEMA_VERSION = "gemma4-a512-component-closure-report-v2"
BOOTSTRAP_ALGORITHM = "hierarchical-paired-seed-packed-sequence-percentile-bootstrap-v1"
SST_AGGREGATE_RELATIVE_TOLERANCE = 1e-8
SST_NORMALIZED_L1_TOLERANCE = 1e-7
SST_PER_SEQUENCE_RELATIVE_TOLERANCE = 1e-6
SST_PER_SEQUENCE_RELATIVE_ZERO_POLICY = (
    "abs_A_denominator; A_zero_and_equal_is_zero; "
    "A_zero_and_unequal_is_infinite"
)
COMMON_SST_POLICY = (
    "condition_A_canonical_per_panel_seed_checkpoint_with_exact_sequence_"
    "pairing_and_bounded_numerical_equivalence"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMON_TARGET_SHA256_FIELDS = (
    "evaluation_token_identity_sha256",
    "evaluation_target_manifest_sha256",
    "centering_target_manifest_sha256",
    "target_mean_sha256",
    "endpoint_receipt_sha256",
    "endpoint_cache_key_sha256",
    "routed_mean_sha256",
    "oracle_shared_mean_sha256",
    "recomposition_sidecar_sha256",
)


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


def _sequence_id_array(value: Any, *, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as error:
        raise ComponentClosureReportError(
            f"{name} must be a lossless integer vector"
        ) from error
    if raw.ndim != 1 or raw.size == 0:
        raise ComponentClosureReportError(
            f"{name} must be a lossless integer vector"
        )
    lower = np.iinfo(np.int64).min
    upper = np.iinfo(np.int64).max
    selected: list[int] = []
    for item in raw.tolist():
        if (
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer))
            or int(item) < lower
            or int(item) > upper
        ):
            raise ComponentClosureReportError(
                f"{name} must be a lossless integer vector"
            )
        selected.append(int(item))
    return np.asarray(selected, dtype=np.int64)


def _sha256_value(value: Any, *, name: str, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ComponentClosureReportError(f"{name} must be a lowercase SHA-256")
    return value


def _raw_vector_sha256(value: np.ndarray, *, dtype: str) -> str:
    return hashlib.sha256(
        np.asarray(value, dtype=dtype).tobytes(order="C")
    ).hexdigest()


def _sequence_identity_sha256(sequence: np.ndarray) -> str:
    body = {
        "domain": "gemma4-a512-component-closure-sequence-identity-v1",
        "encoding": "little_endian_int64_c_order",
        "length": int(sequence.size),
        "raw_sha256": _raw_vector_sha256(sequence, dtype="<i8"),
    }
    return canonical_sha256(body)


def _sst_identity_sha256(
    *,
    role: tuple[str, int, int, str],
    evaluation_result_sha256: str,
    sequence_identity_sha256: str,
    sst: np.ndarray,
    raw_sha256: str,
) -> str:
    panel, seed, checkpoint, condition = role
    body = {
        "domain": "gemma4-a512-component-closure-sst-identity-v1",
        "role": {
            "panel": panel,
            "seed": seed,
            "checkpoint_tokens": checkpoint,
            "condition": condition,
        },
        "evaluation_result_sha256": evaluation_result_sha256,
        "sequence_identity_sha256": sequence_identity_sha256,
        "encoding": "little_endian_float64_c_order",
        "length": int(sst.size),
        "raw_sha256": raw_sha256,
    }
    return canonical_sha256(body)


def _sst_equivalence_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float | int]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if (
        left.ndim != 1
        or right.ndim != 1
        or left.size == 0
        or left.shape != right.shape
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ComponentClosureReportError(
            "common SST representations must be equal nonempty finite vectors"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        reference_total = float(left.sum(dtype=np.float64))
        reference_l1 = float(np.abs(left).sum(dtype=np.float64))
        candidate_total = float(right.sum(dtype=np.float64))
    if (
        not math.isfinite(reference_total)
        or reference_total <= 0
        or not math.isfinite(reference_l1)
        or reference_l1 <= 0
    ):
        raise ComponentClosureReportError(
            "common SST reference must have positive finite total and L1 mass"
        )
    if not math.isfinite(candidate_total):
        raise ComponentClosureReportError(
            "common SST candidate must have a finite total"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(right - left)
        reference_absolute = np.abs(left)
        per_sequence_relative = np.divide(
            difference,
            reference_absolute,
            out=np.full_like(difference, np.inf),
            where=reference_absolute != 0,
        )
        per_sequence_relative[
            (reference_absolute == 0) & (difference == 0)
        ] = 0.0
        metrics = {
            "aggregate_relative_difference": (
                abs(candidate_total - reference_total) / abs(reference_total)
            ),
            "normalized_l1_difference": (
                float(difference.sum(dtype=np.float64)) / reference_l1
            ),
            "max_abs_difference": float(difference.max(initial=0.0)),
            "maximum_per_sequence_relative_difference": float(
                per_sequence_relative.max(initial=0.0)
            ),
            "unequal_sequence_count": int(np.count_nonzero(right != left)),
        }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ComponentClosureReportError(
            "common SST drift metrics must be finite"
        )
    if (
        metrics["aggregate_relative_difference"]
        > SST_AGGREGATE_RELATIVE_TOLERANCE
        or metrics["normalized_l1_difference"]
        > SST_NORMALIZED_L1_TOLERANCE
        or metrics["maximum_per_sequence_relative_difference"]
        > SST_PER_SEQUENCE_RELATIVE_TOLERANCE
    ):
        raise ComponentClosureReportError(
            "conditions are not paired on numerically common SST"
        )
    return metrics


def _validated_score_grid(
    value: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    paired: dict[
        tuple[str, int, int],
        tuple[np.ndarray, np.ndarray, str],
    ] = {}
    representations: list[dict[str, Any]] = []
    target_representations: list[dict[str, Any]] = []
    common_target_reference: dict[str, Any] | None = None
    panel_source_references: dict[str, dict[str, str | None]] = {}
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
            "evaluation_token_identity_sha256",
            "evaluation_target_manifest_sha256",
            "evaluation_token_count",
            "evaluation_packed_sequences",
            "evaluation_source_manifest_sha256",
            "centering_receipt_sha256",
            "centering_target_manifest_sha256",
            "target_mean_sha256",
            "endpoint_receipt_sha256",
            "endpoint_cache_key_sha256",
            "routed_mean_sha256",
            "oracle_shared_mean_sha256",
            "recomposition_sidecar_sha256",
            "source_manifest_sha256",
            "source_mean_sha256",
        }:
            raise ComponentClosureReportError(f"score cell {index} fields changed")
        role = (
            row["panel"],
            row["seed"],
            row["checkpoint_tokens"],
            row["condition"],
        )
        roles.append(role)
        sequence = _sequence_id_array(
            row["sequence_id"],
            name=f"cell {index} sequence_id",
        )
        sse = _finite_array(row["component_sse"], name=f"cell {index} SSE")
        sst = _finite_array(row["component_sst"], name=f"cell {index} SST")
        result_sha256 = row["evaluation_result_sha256"]
        _sha256_value(
            result_sha256,
            name=f"score cell {index} evaluation_result_sha256",
        )
        target_identity = {
            field: _sha256_value(
                row[field],
                name=f"score cell {index} {field}",
            )
            for field in _COMMON_TARGET_SHA256_FIELDS
        }
        centering_receipt_sha256 = _sha256_value(
            row["centering_receipt_sha256"],
            name=f"score cell {index} centering_receipt_sha256",
        )
        source_identity = {
            field: _sha256_value(
                row[field],
                name=f"score cell {index} {field}",
                nullable=True,
            )
            for field in (
                "evaluation_source_manifest_sha256",
                "source_manifest_sha256",
                "source_mean_sha256",
            )
        }
        evaluation_token_count = row["evaluation_token_count"]
        evaluation_packed_sequences = row["evaluation_packed_sequences"]
        if (
            sequence.shape != sse.shape
            or sst.shape != sse.shape
            or np.any(sequence < 0)
            or np.unique(sequence).size != sequence.size
            or np.any(sst <= 0)
            or isinstance(evaluation_token_count, bool)
            or not isinstance(evaluation_token_count, int)
            or evaluation_token_count <= 0
            or isinstance(evaluation_packed_sequences, bool)
            or not isinstance(evaluation_packed_sequences, int)
            or evaluation_packed_sequences != sequence.size
            or evaluation_packed_sequences > evaluation_token_count
        ):
            raise ComponentClosureReportError(
                f"score cell {index} sequence/SSE/SST alignment changed"
            )
        shared_target = {
            **target_identity,
            "evaluation_token_count": evaluation_token_count,
            "evaluation_packed_sequences": evaluation_packed_sequences,
        }
        if common_target_reference is None:
            common_target_reference = shared_target
        elif shared_target != common_target_reference:
            raise ComponentClosureReportError(
                "score cells do not share one exact target identity"
            )
        panel = str(row["panel"])
        if panel == "sae":
            if any(value is not None for value in source_identity.values()):
                raise ComponentClosureReportError(
                    "SAE score cells must not carry source identities"
                )
        elif panel == "transcoder":
            if any(value is None for value in source_identity.values()):
                raise ComponentClosureReportError(
                    "transcoder score cells require source identities"
                )
        previous_source = panel_source_references.setdefault(panel, source_identity)
        if source_identity != previous_source:
            raise ComponentClosureReportError(
                f"{panel} score cells do not share one source identity"
            )
        target_representations.append(
            {
                "panel": row["panel"],
                "seed": row["seed"],
                "checkpoint_tokens": row["checkpoint_tokens"],
                "condition": row["condition"],
                "evaluation_result_sha256": result_sha256,
                **shared_target,
                "centering_receipt_sha256": centering_receipt_sha256,
                **source_identity,
            }
        )
        key = role[:3]
        sequence_sha256 = _sequence_identity_sha256(sequence)
        raw_sst_sha256 = _raw_vector_sha256(sst, dtype="<f8")
        sst_identity_sha256 = _sst_identity_sha256(
            role=role,
            evaluation_result_sha256=result_sha256,
            sequence_identity_sha256=sequence_sha256,
            sst=sst,
            raw_sha256=raw_sst_sha256,
        )
        if row["condition"] == "A":
            if key in paired:
                raise ComponentClosureReportError(
                    f"condition A SST reference repeated for {key}"
                )
            paired[key] = (sequence, sst, sst_identity_sha256)
            metrics = _sst_equivalence_metrics(sst, sst)
            canonical_a_sst_sha256 = sst_identity_sha256
        else:
            reference = paired.get(key)
            if reference is None or not np.array_equal(sequence, reference[0]):
                raise ComponentClosureReportError(
                    f"conditions are not paired on exact sequences for {key}"
                )
            metrics = _sst_equivalence_metrics(reference[1], sst)
            canonical_a_sst_sha256 = reference[2]
        representations.append(
            {
                "panel": row["panel"],
                "seed": row["seed"],
                "checkpoint_tokens": row["checkpoint_tokens"],
                "condition": row["condition"],
                "evaluation_result_sha256": result_sha256,
                "per_sequence_sst_length": int(sst.size),
                "sequence_identity_sha256": sequence_sha256,
                "per_sequence_sst_raw_sha256": raw_sst_sha256,
                "per_sequence_sst_identity_sha256": sst_identity_sha256,
                "canonical_a_per_sequence_sst_identity_sha256": (
                    canonical_a_sst_sha256
                ),
                "sst_total": float(sst.sum(dtype=np.float64)),
                **metrics,
            }
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
    common_sst = {
        "policy": COMMON_SST_POLICY,
        "reference_condition": "A",
        "sequence_pairing": "exact_sequence_id_array_equality",
        "aggregate_relative_tolerance": SST_AGGREGATE_RELATIVE_TOLERANCE,
        "normalized_l1_tolerance": SST_NORMALIZED_L1_TOLERANCE,
        "per_sequence_relative_tolerance": (
            SST_PER_SEQUENCE_RELATIVE_TOLERANCE
        ),
        "per_sequence_relative_zero_denominator_policy": (
            SST_PER_SEQUENCE_RELATIVE_ZERO_POLICY
        ),
        "raw_vector_sha256_encoding": (
            "sha256_of_c_order_bytes_with_explicit_little_endian_dtype"
        ),
        "representations": representations,
        "maximum_observed_aggregate_relative_difference": max(
            row["aggregate_relative_difference"] for row in representations
        ),
        "maximum_observed_normalized_l1_difference": max(
            row["normalized_l1_difference"] for row in representations
        ),
        "maximum_observed_max_abs_difference": max(
            row["max_abs_difference"] for row in representations
        ),
        "maximum_observed_per_sequence_relative_difference": max(
            row["maximum_per_sequence_relative_difference"]
            for row in representations
        ),
        "maximum_observed_unequal_sequence_count": max(
            row["unequal_sequence_count"] for row in representations
        ),
    }
    if common_target_reference is None or set(panel_source_references) != set(
        PANELS
    ):
        raise ComponentClosureReportError(
            "score grid lacks its common target identity"
        )
    common_target = {
        "policy": (
            "exact_global_evaluation_tokens_target_and_centering_with_"
            "panel_exact_source_identity"
        ),
        "token_identity_binding": (
            "canonical_sha256_of_ordered_sequence_id_and_position"
        ),
        "shared_identity": common_target_reference,
        "panel_source_identity": {
            panel: panel_source_references[panel] for panel in PANELS
        },
        "representations": target_representations,
    }
    return (
        {**selected, "evidence_sha256": claimed},
        common_sst,
        common_target,
    )


def validate_score_grid(value: Any) -> dict[str, Any]:
    checked, _common_sst, _common_target = _validated_score_grid(value)
    return checked


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
    checked, common_sst, common_target = _validated_score_grid(evidence)
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
        "common_target": common_target,
        "common_sst": common_sst,
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
    common_sst = report["common_sst"]
    common_target = report["common_target"]
    shared_target = common_target["shared_identity"]
    lines = [
        "# A512 component-closure report",
        "",
        f"Evidence: `{report['evidence_sha256']}`",
        "",
        "Primary endpoint: component-target explained variance. Intervals use "
        "paired hierarchical seed/packed-sequence bootstrap.",
        "",
        (
            "Common target identity: all 72 cells share exactly "
            f"{shared_target['evaluation_token_count']:,} ordered evaluation "
            f"tokens in {shared_target['evaluation_packed_sequences']:,} packed "
            "sequences, the same evaluation-target manifest, the same "
            "separately bound training-centering inputs, and recomposition "
            "sidecar. Token identity canonically binds ordered sequence IDs "
            "and positions; evaluation/training source identities are exact "
            "within panel (null for SAE, present for transcoder)."
        ),
        (
            "The JSON common-target audit inventories every cell, including "
            "its result and centering-receipt identities; centering-receipt "
            "hashes may differ because protocol/config provenance differs."
        ),
        "",
        (
            "Common SST policy: condition A is canonical within each "
            "panel/seed/checkpoint; sequence IDs must match exactly. Other "
            "conditions may differ only within aggregate-relative <= "
            f"{common_sst['aggregate_relative_tolerance']:.0e} and "
            "normalized-L1 <= "
            f"{common_sst['normalized_l1_tolerance']:.0e}, with maximum "
            "per-sequence relative difference <= "
            f"{common_sst['per_sequence_relative_tolerance']:.0e}."
        ),
        (
            "Per-sequence relative differences divide by absolute A SST; "
            "a zero A entry maps equal zero to 0 and any unequal value to "
            "infinity. Scientific SST rows are strictly positive."
        ),
        (
            "Maximum observed SST differences: aggregate-relative "
            f"{common_sst['maximum_observed_aggregate_relative_difference']:.3e}; "
            "normalized-L1 "
            f"{common_sst['maximum_observed_normalized_l1_difference']:.3e}; "
            "max-absolute "
            f"{common_sst['maximum_observed_max_abs_difference']:.3e}; "
            "per-sequence-relative "
            f"{common_sst['maximum_observed_per_sequence_relative_difference']:.3e}; "
            "unequal-sequence count "
            f"{common_sst['maximum_observed_unequal_sequence_count']}."
        ),
        (
            f"The JSON audit inventories all {len(common_sst['representations'])} "
            "condition representations with role/source bindings, sequence "
            "identities, canonical-A and domain-separated SST identities, "
            "raw little-endian float64 hashes, totals, and drift metrics."
        ),
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
    "COMMON_SST_POLICY",
    "EVIDENCE_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "SST_AGGREGATE_RELATIVE_TOLERANCE",
    "SST_NORMALIZED_L1_TOLERANCE",
    "SST_PER_SEQUENCE_RELATIVE_TOLERANCE",
    "SST_PER_SEQUENCE_RELATIVE_ZERO_POLICY",
    "ComponentClosureReportError",
    "build_closure_report",
    "load_score_grid",
    "render_markdown",
    "validate_score_grid",
]
