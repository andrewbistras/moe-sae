"""Pure evidence reduction for the Gemma 4 Gate-1 mechanism audit.

This module imports neither JAX nor cloud clients.  It accepts only a validated
Gate-1 protocol and completed, sequence-clustered cell records.  Its job is to
fail closed on identity or matrix drift, form exact paired comparisons, run
the precommitted 20,000-replicate sequence bootstrap, and state only the
bounded inference-time conclusions which Gate 1 can support.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from moe_sae.config import Gemma4Config
from moe_sae.diagnostics import (
    PAIRED_BOOTSTRAP_REPLICATES,
    pair_sequence_sse_summaries,
    paired_sequence_cluster_bootstrap_ci,
    validate_sequence_sse_summary,
)
from moe_sae.gate1_protocol import (
    GATE1_ROUTE_MODES,
    Gate1Protocol,
    SplitSpec,
)

GATE1_CELL_RESULT_SCHEMA = "gemma4-gate1-cell-result-v1"
GATE1_REPORT_SCHEMA = "gemma4-gate1-mechanism-report-v1"

type CellKey = tuple[str, str, str]

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CELL_FIELDS = {
    "schema_version",
    "protocol_sha256",
    "cell_identity_sha256",
    "consumer_environment_sha256",
    "process_result_sha256s",
    "checkpoint_label",
    "checkpoint_expectation_sha256",
    "checkpoint_descriptor_sha256",
    "checkpoint_uri",
    "capture_config_sha256",
    "capture_checkpoint_sha256",
    "capture_manifest_sha256",
    "split",
    "source",
    "ranges",
    "route_mode",
    "route_permutation_domain",
    "route_permutation_key",
    "tokens",
    "exact_l0",
    "topk_valid",
    "dead_features",
    "total_features",
    "fired_mask_sha256",
    "audit_capacity",
    "sequence_sse",
}
_REPORT_FIELDS = {
    "schema_version",
    "protocol_id",
    "protocol_sha256",
    "capture",
    "statistics",
    "matrix_accounting",
    "cell_count",
    "cells",
    "comparisons",
    "seed_summaries",
    "learning_curve",
    "mechanism_decision",
    "report_sha256",
}
_UINT64_MAX = 2**64 - 1


class Gate1ReportError(ValueError):
    """Gate-1 cell evidence or its derived report violates the contract."""


def _canonical_bytes(value: Any, *, label: str) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Gate1ReportError(f"{label} must be canonical JSON") from error
    try:
        decoded = json.loads(encoded)
    except json.JSONDecodeError as error:  # pragma: no cover - encoded above
        raise Gate1ReportError(f"{label} must be canonical JSON") from error
    if decoded != value:
        raise Gate1ReportError(f"{label} must round-trip through JSON exactly")
    return encoded


def _canonical_sha256(value: Any, *, label: str) -> str:
    return hashlib.sha256(_canonical_bytes(value, label=label)).hexdigest()


def _mapping(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Gate1ReportError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise Gate1ReportError(f"{label} keys must be strings")
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        raise Gate1ReportError(
            f"{label} fields do not match schema ({'; '.join(details)})"
        )
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Gate1ReportError(f"{label} must be a nonempty string")
    return value


def _sha256(value: Any, label: str) -> str:
    selected = _string(value, label)
    if _SHA256_RE.fullmatch(selected) is None:
        raise Gate1ReportError(f"{label} must be 64 lowercase hexadecimal characters")
    return selected


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Gate1ReportError(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f" and at most {maximum}" if maximum is not None else ""
        raise Gate1ReportError(f"{label} must be at least {minimum}{suffix}")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Gate1ReportError(f"{label} must be numeric")
    selected = float(value)
    if not math.isfinite(selected):
        raise Gate1ReportError(f"{label} must be finite")
    return selected


def _cell_key(value: Any, label: str) -> CellKey:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise Gate1ReportError(
            f"{label} must be a (checkpoint_label, split, route_mode) tuple"
        )
    return value


def _ranges(value: Any, expected: SplitSpec, label: str) -> list[list[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise Gate1ReportError(f"{label} must be a sequence")
    normalized: list[list[int]] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes, bytearray))
            or len(item) != 2
        ):
            raise Gate1ReportError(f"{label}[{index}] must be a two-item range")
        normalized.append(
            [
                _integer(item[0], f"{label}[{index}][0]", minimum=0),
                _integer(item[1], f"{label}[{index}][1]", minimum=1),
            ]
        )
    expected_ranges = [item.to_list() for item in expected.ranges]
    if normalized != expected_ranges:
        raise Gate1ReportError(f"{label} does not match the frozen split")
    return normalized


def _expanded_gate1_cells(protocol: Gate1Protocol) -> tuple[CellKey, ...]:
    matrix = protocol.matrix
    requests: list[CellKey] = []
    requests.extend(
        (
            checkpoint,
            matrix.learning_curve.split,
            matrix.learning_curve.route_mode,
        )
        for checkpoint in matrix.learning_curve.checkpoints
    )
    requests.extend(
        (checkpoint, split, matrix.final_baselines.route_mode)
        for checkpoint in matrix.final_baselines.checkpoints
        for split in matrix.final_baselines.splits
    )
    requests.extend(
        (checkpoint, matrix.router_ablation.split, route_mode)
        for checkpoint in matrix.router_ablation.checkpoints
        for route_mode in matrix.router_ablation.route_modes
    )
    return tuple(requests)


def required_gate1_cells(protocol: Gate1Protocol) -> tuple[CellKey, ...]:
    """Expand the matrix and collapse repeated cells in first-use order."""

    if not isinstance(protocol, Gate1Protocol):
        raise Gate1ReportError("protocol must be a validated Gate1Protocol")
    return tuple(dict.fromkeys(_expanded_gate1_cells(protocol)))


def _expected_total_features(protocol: Gate1Protocol, checkpoint_label: str) -> int:
    condition = protocol.checkpoints[checkpoint_label].condition
    config = Gemma4Config()
    return config.visible_features if condition == "B" else config.total_features


def _expected_candidates(
    protocol: Gate1Protocol, checkpoint_label: str, route_mode: str
) -> int:
    condition = protocol.checkpoints[checkpoint_label].condition
    config = Gemma4Config()
    if route_mode == "all_banks" or condition == "F":
        return config.total_features
    return config.visible_features


def validate_gate1_cell_record(
    protocol: Gate1Protocol,
    key: CellKey,
    value: Any,
) -> dict[str, Any]:
    """Validate and normalize one unique completed matrix cell."""

    if not isinstance(protocol, Gate1Protocol):
        raise Gate1ReportError("protocol must be a validated Gate1Protocol")
    selected_key = _cell_key(key, "cell key")
    checkpoint_label, split_name, route_mode = selected_key
    if selected_key not in set(required_gate1_cells(protocol)):
        raise Gate1ReportError(f"cell key is not required by the protocol: {key!r}")
    raw = _mapping(value, _CELL_FIELDS, f"cell {key!r}")

    if raw.get("schema_version") != GATE1_CELL_RESULT_SCHEMA:
        raise Gate1ReportError(
            f"cell schema_version must be {GATE1_CELL_RESULT_SCHEMA!r}"
        )
    if raw.get("checkpoint_label") != checkpoint_label:
        raise Gate1ReportError("cell checkpoint_label disagrees with its mapping key")
    if raw.get("split") != split_name:
        raise Gate1ReportError("cell split disagrees with its mapping key")
    if raw.get("route_mode") != route_mode:
        raise Gate1ReportError("cell route_mode disagrees with its mapping key")
    if route_mode not in GATE1_ROUTE_MODES:
        raise Gate1ReportError("cell route_mode is unsupported")

    if _sha256(raw.get("protocol_sha256"), "cell.protocol_sha256") != protocol.sha256:
        raise Gate1ReportError("cell belongs to another Gate-1 protocol")
    cell_identity_sha256 = _sha256(
        raw.get("cell_identity_sha256"), "cell.cell_identity_sha256"
    )
    consumer_environment_sha256 = _sha256(
        raw.get("consumer_environment_sha256"),
        "cell.consumer_environment_sha256",
    )
    raw_process_hashes = raw.get("process_result_sha256s")
    if not isinstance(raw_process_hashes, list) or len(raw_process_hashes) != 4:
        raise Gate1ReportError(
            "cell.process_result_sha256s must contain four process hashes"
        )
    process_result_sha256s = [
        _sha256(value, f"cell.process_result_sha256s[{index}]")
        for index, value in enumerate(raw_process_hashes)
    ]
    checkpoint = protocol.checkpoints[checkpoint_label]
    expectation_sha = _sha256(
        raw.get("checkpoint_expectation_sha256"),
        "cell.checkpoint_expectation_sha256",
    )
    if expectation_sha != checkpoint.expectation.sha256:
        raise Gate1ReportError("cell checkpoint expectation binding mismatch")
    _sha256(
        raw.get("checkpoint_descriptor_sha256"),
        "cell.checkpoint_descriptor_sha256",
    )
    if raw.get("checkpoint_uri") != checkpoint.checkpoint_uri:
        raise Gate1ReportError("cell checkpoint URI binding mismatch")

    capture = protocol.capture
    capture_bindings = {
        "capture_config_sha256": capture.config_sha256,
        "capture_checkpoint_sha256": capture.checkpoint_sha256,
    }
    for name, expected_value in capture_bindings.items():
        if _sha256(raw.get(name), f"cell.{name}") != expected_value:
            raise Gate1ReportError(f"cell {name} binding mismatch")
    split = protocol.splits[split_name]
    expected_manifest = (
        capture.train_manifest_sha256
        if split.source == "train"
        else capture.eval_manifest_sha256
    )
    if (
        _sha256(
            raw.get("capture_manifest_sha256"),
            "cell.capture_manifest_sha256",
        )
        != expected_manifest
    ):
        raise Gate1ReportError("cell capture manifest binding mismatch")
    if raw.get("source") != split.source:
        raise Gate1ReportError("cell source does not match its frozen split")
    normalized_ranges = _ranges(raw.get("ranges"), split, "cell.ranges")

    permutation_domain = raw.get("route_permutation_domain")
    permutation_key = raw.get("route_permutation_key")
    if route_mode == "shuffled":
        if permutation_domain != protocol.route_permutation.domain:
            raise Gate1ReportError("shuffled cell route permutation domain mismatch")
        if (
            _integer(
                permutation_key,
                "cell.route_permutation_key",
                minimum=0,
                maximum=_UINT64_MAX,
            )
            != protocol.route_permutation.key
        ):
            raise Gate1ReportError("shuffled cell route permutation key mismatch")
    elif permutation_domain is not None or permutation_key is not None:
        raise Gate1ReportError(
            "non-shuffled cells must use null route permutation domain/key"
        )

    tokens = _integer(raw.get("tokens"), "cell.tokens", minimum=1)
    if tokens != split.tokens:
        raise Gate1ReportError("cell token count does not match its frozen split")
    config = Gemma4Config()
    if raw.get("exact_l0") is not True:
        raise Gate1ReportError(
            f"cell exact_l0 must prove L0={config.topk_features} on every valid token"
        )
    if raw.get("topk_valid") is not True:
        raise Gate1ReportError("cell topk_valid must be exactly true")
    expected_features = _expected_total_features(protocol, checkpoint_label)
    total_features = _integer(
        raw.get("total_features"), "cell.total_features", minimum=1
    )
    if total_features != expected_features:
        raise Gate1ReportError(
            "cell total_features disagrees with its restored condition"
        )
    if route_mode == "all_banks" and total_features != config.total_features:
        raise Gate1ReportError(
            "all_banks must expose an A checkpoint's 131072 features"
        )
    dead_features = _integer(
        raw.get("dead_features"),
        "cell.dead_features",
        minimum=0,
        maximum=total_features,
    )
    fired_mask_sha256 = _sha256(raw.get("fired_mask_sha256"), "cell.fired_mask_sha256")
    raw_capacity = raw.get("audit_capacity")
    if checkpoint.condition == "A" and route_mode != "all_banks":
        capacity_fields = {
            "margin",
            "round_to",
            "local_observed_max_by_process",
            "global_observed_max",
            "capacity",
        }
        capacity_value = _mapping(raw_capacity, capacity_fields, "cell.audit_capacity")
        if (
            _finite_float(capacity_value.get("margin"), "cell.audit_capacity.margin")
            != 0.10
        ):
            raise Gate1ReportError("cell.audit_capacity margin must be 10%")
        round_to = _integer(
            capacity_value.get("round_to"),
            "cell.audit_capacity.round_to",
            minimum=1,
        )
        if round_to != 8:
            raise Gate1ReportError("cell.audit_capacity round_to must be eight")
        local_raw = capacity_value.get("local_observed_max_by_process")
        if not isinstance(local_raw, list) or len(local_raw) != 4:
            raise Gate1ReportError("cell.audit_capacity requires four local maxima")
        local_maxima = [
            _integer(
                value,
                f"cell.audit_capacity.local_observed_max_by_process[{index}]",
                minimum=0,
            )
            for index, value in enumerate(local_raw)
        ]
        global_maximum = _integer(
            capacity_value.get("global_observed_max"),
            "cell.audit_capacity.global_observed_max",
            minimum=1,
        )
        capacity = _integer(
            capacity_value.get("capacity"),
            "cell.audit_capacity.capacity",
            minimum=global_maximum,
        )
        expected_capacity = (
            math.ceil(global_maximum * (1.0 + 0.10) / round_to) * round_to
        )
        if global_maximum != max(local_maxima) or capacity != expected_capacity:
            raise Gate1ReportError("cell.audit_capacity is internally inconsistent")
        audit_capacity: dict[str, Any] | None = {
            "margin": 0.10,
            "round_to": round_to,
            "local_observed_max_by_process": local_maxima,
            "global_observed_max": global_maximum,
            "capacity": capacity,
        }
    else:
        if raw_capacity is not None:
            raise Gate1ReportError(
                "global/all-banks scoring must have null audit_capacity"
            )
        audit_capacity = None
    try:
        sequence_sse = validate_sequence_sse_summary(raw.get("sequence_sse"))
    except ValueError as error:
        raise Gate1ReportError("cell sequence_sse is invalid") from error
    if sequence_sse["tokens"] != tokens:
        raise Gate1ReportError("cell sequence_sse token count mismatch")

    return {
        "schema_version": GATE1_CELL_RESULT_SCHEMA,
        "protocol_sha256": protocol.sha256,
        "cell_identity_sha256": cell_identity_sha256,
        "consumer_environment_sha256": consumer_environment_sha256,
        "process_result_sha256s": process_result_sha256s,
        "checkpoint_label": checkpoint_label,
        "checkpoint_expectation_sha256": expectation_sha,
        "checkpoint_descriptor_sha256": raw["checkpoint_descriptor_sha256"],
        "checkpoint_uri": checkpoint.checkpoint_uri,
        "capture_config_sha256": capture.config_sha256,
        "capture_checkpoint_sha256": capture.checkpoint_sha256,
        "capture_manifest_sha256": expected_manifest,
        "split": split_name,
        "source": split.source,
        "ranges": normalized_ranges,
        "route_mode": route_mode,
        "route_permutation_domain": permutation_domain,
        "route_permutation_key": permutation_key,
        "tokens": tokens,
        "exact_l0": True,
        "topk_valid": True,
        "dead_features": dead_features,
        "total_features": total_features,
        "fired_mask_sha256": fired_mask_sha256,
        "audit_capacity": audit_capacity,
        "sequence_sse": sequence_sse,
    }


def _assert_common_split_measure(
    records: Mapping[CellKey, Mapping[str, Any]],
) -> None:
    references: dict[str, list[Mapping[str, Any]]] = {}
    for key, record in records.items():
        summary = record["sequence_sse"]
        reference = references.setdefault(key[1], summary["clusters"])
        clusters = summary["clusters"]
        if len(clusters) != len(reference):
            raise Gate1ReportError(
                f"cells on split {key[1]!r} contain different sequences"
            )
        for index, (expected, observed) in enumerate(
            zip(reference, clusters, strict=True)
        ):
            if (
                observed["sequence_id"] != expected["sequence_id"]
                or observed["tokens"] != expected["tokens"]
                or float(observed["sst"]) != float(expected["sst"])
            ):
                raise Gate1ReportError(
                    f"cells on split {key[1]!r} disagree on sequence/SST "
                    f"cluster {index}"
                )


def validate_gate1_cell_records(
    protocol: Gate1Protocol,
    records: Any,
) -> dict[CellKey, dict[str, Any]]:
    """Validate exactly the unique matrix cells required by the protocol."""

    if not isinstance(records, Mapping):
        raise Gate1ReportError("Gate-1 cell records must be a mapping")
    normalized_keys: list[CellKey] = []
    for index, key in enumerate(records):
        normalized_keys.append(_cell_key(key, f"cell key {index}"))
    if len(set(normalized_keys)) != len(normalized_keys):
        raise Gate1ReportError("Gate-1 cell keys must be unique")
    required = required_gate1_cells(protocol)
    supplied = set(normalized_keys)
    expected = set(required)
    if supplied != expected:
        missing = [repr(key) for key in required if key not in supplied]
        extra = [repr(key) for key in normalized_keys if key not in expected]
        details: list[str] = []
        if missing:
            details.append(f"missing cells: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected cells: {', '.join(extra)}")
        raise Gate1ReportError("; ".join(details))
    validated = {
        key: validate_gate1_cell_record(protocol, key, records[key]) for key in required
    }
    _assert_common_split_measure(validated)
    return validated


def _summary_for_pair(record: Mapping[str, Any], key: CellKey) -> dict[str, Any]:
    summary = dict(record["sequence_sse"])
    summary["label"] = "|".join(key)
    return validate_sequence_sse_summary(summary)


def _comparison(
    records: Mapping[CellKey, Mapping[str, Any]],
    *,
    comparison_id: str,
    family: str,
    seed: int,
    split: str,
    baseline_key: CellKey,
    candidate_key: CellKey,
    bootstrap_seed: int,
    material_effect_ev: float,
) -> dict[str, Any]:
    paired = pair_sequence_sse_summaries(
        _summary_for_pair(records[baseline_key], baseline_key),
        _summary_for_pair(records[candidate_key], candidate_key),
    )
    bootstrap = paired_sequence_cluster_bootstrap_ci(paired, seed=bootstrap_seed)
    point = float(paired["candidate_minus_baseline_ev"])
    return {
        "comparison_id": comparison_id,
        "family": family,
        "seed": seed,
        "split": split,
        "baseline_cell": list(baseline_key),
        "candidate_cell": list(candidate_key),
        "effect": "candidate_minus_baseline_ev",
        "material_effect_ev": material_effect_ev,
        "point_ev": point,
        "direction": (
            "positive" if point > 0.0 else "negative" if point < 0.0 else "zero"
        ),
        "material_positive": point >= material_effect_ev,
        "material_negative": point <= -material_effect_ev,
        "paired_sequence_sse": paired,
        "bootstrap_95": bootstrap,
    }


def _build_comparisons(
    protocol: Gate1Protocol,
    records: Mapping[CellKey, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    bootstrap_seed = protocol.statistics.bootstrap_key
    material = protocol.statistics.material_effect_ev
    for split in protocol.matrix.final_baselines.splits:
        for seed in (1, 2, 3):
            result.append(
                _comparison(
                    records,
                    comparison_id=f"A-vs-B:seed-{seed}:{split}",
                    family="A_vs_B",
                    seed=seed,
                    split=split,
                    baseline_key=(f"B_seed{seed}_50m", split, "true"),
                    candidate_key=(f"A_seed{seed}_50m", split, "true"),
                    bootstrap_seed=bootstrap_seed,
                    material_effect_ev=material,
                )
            )
    for seed in (1, 2, 3):
        checkpoint = f"A_seed{seed}_50m"
        result.append(
            _comparison(
                records,
                comparison_id=f"shuffled-to-true:seed-{seed}:heldout",
                family="shuffled_to_true",
                seed=seed,
                split="heldout",
                baseline_key=(checkpoint, "heldout", "shuffled"),
                candidate_key=(checkpoint, "heldout", "true"),
                bootstrap_seed=bootstrap_seed,
                material_effect_ev=material,
            )
        )
        result.append(
            _comparison(
                records,
                comparison_id=f"true-to-all-banks:seed-{seed}:heldout",
                family="true_to_all_banks",
                seed=seed,
                split="heldout",
                baseline_key=(checkpoint, "heldout", "true"),
                candidate_key=(checkpoint, "heldout", "all_banks"),
                bootstrap_seed=bootstrap_seed,
                material_effect_ev=material,
            )
        )
    return result


def _seed_summary(
    family: str,
    split: str,
    comparisons: Sequence[Mapping[str, Any]],
    material_effect_ev: float,
) -> dict[str, Any]:
    selected = [
        item
        for item in comparisons
        if item["family"] == family and item["split"] == split
    ]
    if [item["seed"] for item in selected] != [1, 2, 3]:
        raise Gate1ReportError(
            f"comparison family {family!r}/{split!r} lacks seeds 1, 2, and 3"
        )
    estimates = [
        {
            "seed": item["seed"],
            "point_ev": item["point_ev"],
            "ci_lower": item["bootstrap_95"]["ci_lower"],
            "ci_upper": item["bootstrap_95"]["ci_upper"],
            "material_positive": item["material_positive"],
            "material_negative": item["material_negative"],
        }
        for item in selected
    ]
    points = [float(item["point_ev"]) for item in estimates]
    return {
        "family": family,
        "split": split,
        "aggregation": "unweighted-mean-and-range-of-seed-point-estimates",
        "token_pooling": False,
        "seed_count": 3,
        "per_seed": estimates,
        "mean_point_ev": math.fsum(points) / 3.0,
        "range_point_ev": [min(points), max(points)],
        "all_seeds_positive": all(point > 0.0 for point in points),
        "all_seeds_negative": all(point < 0.0 for point in points),
        "material_positive_seed_count": sum(
            point >= material_effect_ev for point in points
        ),
        "material_negative_seed_count": sum(
            point <= -material_effect_ev for point in points
        ),
        "positive_ci_seed_count": sum(
            float(item["ci_lower"]) > 0.0 for item in estimates
        ),
        "negative_ci_seed_count": sum(
            float(item["ci_upper"]) < 0.0 for item in estimates
        ),
    }


def _build_seed_summaries(
    protocol: Gate1Protocol,
    comparisons: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    material = protocol.statistics.material_effect_ev
    return [
        _seed_summary("A_vs_B", "heldout", comparisons, material),
        _seed_summary("A_vs_B", "train_stratified", comparisons, material),
        _seed_summary("shuffled_to_true", "heldout", comparisons, material),
        _seed_summary("true_to_all_banks", "heldout", comparisons, material),
    ]


def _cell_table(
    protocol: Gate1Protocol,
    records: Mapping[CellKey, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    config = Gemma4Config()
    table: list[dict[str, Any]] = []
    for key in required_gate1_cells(protocol):
        checkpoint_label, split, route_mode = key
        checkpoint = protocol.checkpoints[checkpoint_label]
        record = records[key]
        summary = record["sequence_sse"]
        candidates = _expected_candidates(protocol, checkpoint_label, route_mode)
        table.append(
            {
                "cell": list(key),
                "checkpoint_label": checkpoint_label,
                "condition": checkpoint.condition,
                "seed": checkpoint.seed,
                "checkpoint_tokens_seen": checkpoint.tokens_seen,
                "checkpoint_steps": checkpoint.steps,
                "split": split,
                "source": record["source"],
                "ranges": record["ranges"],
                "route_mode": route_mode,
                "route_permutation_domain": record["route_permutation_domain"],
                "route_permutation_key": record["route_permutation_key"],
                "capture_manifest_sha256": record["capture_manifest_sha256"],
                "tokens": record["tokens"],
                "sequence_count": summary["sequence_count"],
                "sst": summary["sst"],
                "sse": summary["sse"],
                "nmse": summary["nmse"],
                "ev": summary["ev"],
                "exact_l0": record["exact_l0"],
                "l0": config.topk_features,
                "topk_valid": record["topk_valid"],
                "dead_features": record["dead_features"],
                "total_features": record["total_features"],
                "dead_rate": record["dead_features"] / record["total_features"],
                "candidates_per_token": candidates,
                "candidate_exposure_vs_full_dictionary": (
                    candidates / config.total_features
                ),
                "checkpoint_expectation_sha256": record[
                    "checkpoint_expectation_sha256"
                ],
                "checkpoint_descriptor_sha256": record["checkpoint_descriptor_sha256"],
                "cell_identity_sha256": record["cell_identity_sha256"],
                "consumer_environment_sha256": record["consumer_environment_sha256"],
                "process_result_sha256s": record["process_result_sha256s"],
                "fired_mask_sha256": record["fired_mask_sha256"],
                "audit_capacity": record["audit_capacity"],
                "sequence_sse_sha256": _canonical_sha256(
                    summary, label="cell sequence SSE"
                ),
                "cell_record_sha256": _canonical_sha256(
                    record, label="Gate-1 cell record"
                ),
            }
        )
    return table


def _learning_curve(
    protocol: Gate1Protocol,
    records: Mapping[CellKey, Mapping[str, Any]],
) -> dict[str, Any]:
    keys: tuple[CellKey, ...] = (
        ("A_seed1_5m", "heldout", "true"),
        ("B_seed1_5m", "heldout", "true"),
        ("F_seed1_5m", "heldout", "true"),
        ("A_seed1_40m", "heldout", "true"),
        ("A_seed1_50m", "heldout", "true"),
    )
    rows: list[dict[str, Any]] = []
    for key in keys:
        checkpoint = protocol.checkpoints[key[0]]
        summary = records[key]["sequence_sse"]
        rows.append(
            {
                "cell": list(key),
                "condition": checkpoint.condition,
                "seed": checkpoint.seed,
                "checkpoint_tokens_seen": checkpoint.tokens_seen,
                "checkpoint_steps": checkpoint.steps,
                "ev": summary["ev"],
                "nmse": summary["nmse"],
                "reused_final_baseline_cell": key == keys[-1],
            }
        )
    return {
        "split": "heldout",
        "route_mode": "true",
        "common_tokens": protocol.splits["heldout"].tokens,
        "rows": rows,
        "reused_cell": list(keys[-1]),
        "limitations": [
            "B/F 40M and 50M checkpoints were not retained in this audit matrix",
            "these endpoint evaluations are not a dense held-out learning curve",
        ],
    }


def _evidence_grade(summary: Mapping[str, Any]) -> str:
    if (
        summary["material_positive_seed_count"] == 3
        and summary["positive_ci_seed_count"] == 3
    ):
        return "material-positive-in-all-seeds"
    if summary["all_seeds_positive"]:
        return "directionally-positive-exploratory"
    if summary["all_seeds_negative"]:
        return "directionally-negative"
    return "mixed-or-inconclusive"


def _mechanism_decision(
    protocol: Gate1Protocol,
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_key = {(summary["family"], summary["split"]): summary for summary in summaries}
    heldout_ab = by_key[("A_vs_B", "heldout")]
    train_ab = by_key[("A_vs_B", "train_stratified")]
    routing = by_key[("shuffled_to_true", "heldout")]
    all_banks = by_key[("true_to_all_banks", "heldout")]

    overfit_patterns: list[dict[str, Any]] = []
    for seed in (1, 2, 3):
        train = train_ab["per_seed"][seed - 1]
        heldout = heldout_ab["per_seed"][seed - 1]
        train_point = float(train["point_ev"])
        heldout_point = float(heldout["point_ev"])
        if train_point > 0.0 and heldout_point < 0.0:
            pattern = "train-win-heldout-loss-sign-reversal"
        elif train_point <= 0.0 and heldout_point <= 0.0:
            pattern = "A-trails-or-ties-B-on-both"
        else:
            pattern = "mixed"
        overfit_patterns.append(
            {
                "seed": seed,
                "train_A_minus_B_ev": train_point,
                "heldout_A_minus_B_ev": heldout_point,
                "pattern": pattern,
                "material_sign_reversal": (
                    train_point >= protocol.statistics.material_effect_ev
                    and heldout_point <= -protocol.statistics.material_effect_ev
                ),
            }
        )
    pattern_names = [item["pattern"] for item in overfit_patterns]
    if all(name == "A-trails-or-ties-B-on-both" for name in pattern_names):
        overfit_code = "ordinary-overfit-not-supported-by-sampled-train-and-heldout"
    elif all(name == "train-win-heldout-loss-sign-reversal" for name in pattern_names):
        overfit_code = "pattern-consistent-with-generalization-failure"
    else:
        overfit_code = "overfit-diagnostic-inconclusive"

    return {
        "material_effect_ev": protocol.statistics.material_effect_ev,
        "router_identity": {
            "contrast": "A true-route minus A shuffled-route at inference",
            "evidence_grade": _evidence_grade(routing),
            "per_seed_summary": routing,
            "training_time_causality_established": False,
            "caution": (
                "shuffled routing is an inference-time counterfactual and does "
                "not replace matched randomized-route training"
            ),
        },
        "all_banks_visibility": {
            "contrast": "A all-banks minus A true-route at inference",
            "evidence_grade": _evidence_grade(all_banks),
            "per_seed_summary": all_banks,
            "training_distribution_matched": False,
            "caution": (
                "all_banks exposes each learned bank outside its training routes"
            ),
        },
        "ordinary_overfit": {
            "evidence_code": overfit_code,
            "per_seed_patterns": overfit_patterns,
            "ruled_out": False,
            "caution": (
                "the train audit is a fixed stratified sample, not a complete "
                "final-checkpoint training-set evaluation"
            ),
        },
        "original_expert_decomposition_thesis_tested": False,
        "gate2_required": True,
        "causal_scope": (
            "Gate 1 diagnoses trained-checkpoint inference behavior only; it "
            "cannot establish a training-time routing advantage or test the "
            "original per-expert contribution objective"
        ),
    }


def _matrix_accounting(protocol: Gate1Protocol) -> dict[str, int]:
    expanded = _expanded_gate1_cells(protocol)
    unique = required_gate1_cells(protocol)
    return {
        "expanded_requests": len(expanded),
        "unique_cells": len(unique),
        "duplicate_requests_collapsed": len(expanded) - len(unique),
    }


def build_gate1_report(
    protocol: Gate1Protocol,
    cell_records: Mapping[CellKey, Any],
) -> dict[str, Any]:
    """Validate all evidence and build the deterministic Gate-1 report."""

    if protocol.statistics.bootstrap_replicates != PAIRED_BOOTSTRAP_REPLICATES:
        raise Gate1ReportError(
            f"Gate-1 reporting requires exactly "
            f"{PAIRED_BOOTSTRAP_REPLICATES} bootstrap replicates"
        )
    records = validate_gate1_cell_records(protocol, cell_records)
    comparisons = _build_comparisons(protocol, records)
    summaries = _build_seed_summaries(protocol, comparisons)
    report: dict[str, Any] = {
        "schema_version": GATE1_REPORT_SCHEMA,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "capture": protocol.capture.to_mapping(),
        "statistics": protocol.statistics.to_mapping(),
        "matrix_accounting": _matrix_accounting(protocol),
        "cell_count": len(records),
        "cells": _cell_table(protocol, records),
        "comparisons": comparisons,
        "seed_summaries": summaries,
        "learning_curve": _learning_curve(protocol, records),
        "mechanism_decision": _mechanism_decision(protocol, summaries),
    }
    report["report_sha256"] = _canonical_sha256(report, label="Gate-1 report")
    return validate_gate1_report(report, protocol)


def validate_gate1_report(
    value: Any,
    protocol: Gate1Protocol,
) -> dict[str, Any]:
    """Validate the report envelope, canonical hash, and protocol binding."""

    raw = _mapping(value, _REPORT_FIELDS, "Gate-1 report")
    if raw.get("schema_version") != GATE1_REPORT_SCHEMA:
        raise Gate1ReportError("Gate-1 report schema_version mismatch")
    if raw.get("protocol_id") != protocol.protocol_id:
        raise Gate1ReportError("Gate-1 report protocol_id mismatch")
    if _sha256(raw.get("protocol_sha256"), "report.protocol_sha256") != protocol.sha256:
        raise Gate1ReportError("Gate-1 report protocol binding mismatch")
    report_sha = _sha256(raw.get("report_sha256"), "report.report_sha256")
    unsigned = dict(raw)
    unsigned.pop("report_sha256")
    if _canonical_sha256(unsigned, label="Gate-1 report") != report_sha:
        raise Gate1ReportError("Gate-1 report self-hash mismatch")
    if raw.get("capture") != protocol.capture.to_mapping():
        raise Gate1ReportError("Gate-1 report capture binding mismatch")
    if raw.get("statistics") != protocol.statistics.to_mapping():
        raise Gate1ReportError("Gate-1 report statistics binding mismatch")
    accounting = raw.get("matrix_accounting")
    if accounting != _matrix_accounting(protocol):
        raise Gate1ReportError("Gate-1 report matrix accounting mismatch")
    cell_count = _integer(raw.get("cell_count"), "report.cell_count", minimum=1)
    cells = raw.get("cells")
    if not isinstance(cells, list) or len(cells) != cell_count:
        raise Gate1ReportError("Gate-1 report cell table length mismatch")
    if cell_count != len(required_gate1_cells(protocol)):
        raise Gate1ReportError("Gate-1 report cell count is incomplete")
    comparisons = raw.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 12:
        raise Gate1ReportError("Gate-1 report must contain 12 paired comparisons")
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, Mapping):
            raise Gate1ReportError(f"comparison {index} must be a mapping")
        bootstrap = comparison.get("bootstrap_95")
        if (
            not isinstance(bootstrap, Mapping)
            or bootstrap.get("replicates") != PAIRED_BOOTSTRAP_REPLICATES
        ):
            raise Gate1ReportError(
                f"comparison {index} must use the 20k paired bootstrap"
            )
    summaries = raw.get("seed_summaries")
    if not isinstance(summaries, list) or len(summaries) != 4:
        raise Gate1ReportError("Gate-1 report seed summaries are incomplete")
    if any(summary.get("token_pooling") is not False for summary in summaries):
        raise Gate1ReportError("Gate-1 report may not pool tokens across seeds")
    _canonical_bytes(dict(raw), label="Gate-1 report")
    return dict(raw)


__all__ = [
    "GATE1_CELL_RESULT_SCHEMA",
    "GATE1_REPORT_SCHEMA",
    "CellKey",
    "Gate1ReportError",
    "build_gate1_report",
    "required_gate1_cells",
    "validate_gate1_cell_record",
    "validate_gate1_cell_records",
    "validate_gate1_report",
]
