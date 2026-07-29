"""Build and validate the immutable Experiment-5 scientific report.

The assembler consumes eighteen joint H/R/B/F curve-point results and their
eighteen packed-sequence score sidecars: two panels, three seeds, and the
2M/5M/10M milestones. It recomputes the 10M paired effects from additive
sequence scores, performs a hierarchical seed/sequence bootstrap, derives G5
only under a positive F-minus-B denominator, and selects the first *observed*
H checkpoint reaching the paired seed's F-at-10M quality.

Raw cost remains visible on every hardware generation. H/F ratios are
interpreted only when both trajectories used the same accelerator generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from moe_sae.contribution_artifacts import canonical_json_bytes, canonical_sha256

PROTOCOL_ID = "gemma4-routed-shared-hybrid-v1-20260728"
RESULT_SCHEMA_VERSION = "gemma4-routed-shared-hybrid-scientific-result-v1"
PAIRED_SCHEMA_VERSION = "gemma4-routed-shared-hybrid-paired-scores-v1"
REPORT_SCHEMA_VERSION = "gemma4-routed-shared-hybrid-report-v1"
BUDGET_SCHEMA_VERSION = "gemma4-routed-shared-hybrid-budget-ledger-v1"

PANELS = ("sae", "transcoder")
CONDITIONS = ("H", "R", "B", "F")
MILESTONES = (2_000_000, 5_000_000, 10_000_000)
SEEDS = (0, 1, 2)
CONTRASTS = {
    "H-R": ("H", "R"),
    "H-B": ("H", "B"),
    "H-F": ("H", "F"),
    "F-B": ("F", "B"),
}
BOOTSTRAP_REPLICATES = 20_000
CONFIDENCE_LEVEL = 0.95
EVAL_TOKENS = 250_000
HARD_BUDGET_USD = 50.0
HYBRID_EXPOSURE = 8_192 / 69_632

# Canonical targets are identical across conditions, but their FP32 SST
# reductions can differ slightly when the same held-out examples are evaluated
# on different TPU generations.  The per-sequence tolerance covers the
# measured cross-hardware reduction drift; the much tighter aggregate
# tolerance prevents that allowance from hiding a material change in the
# normalization basis.  Both checks are purely relative because every
# per-sequence SST is independently required to be finite and strictly
# positive.
CROSS_HARDWARE_SST_RTOL = 2e-6
CROSS_HARDWARE_SST_SUM_RTOL = 1e-9

TRAIN_TARGET_SHA256 = (
    "f29aaf45d9ff93be08cc3d175cdcea9fa4aecc37fc0e256d09720e88e1ae838c"
)
EVAL_TARGET_SHA256 = (
    "31454e5ce8693521c41a2a95f2ef88d40c5d26c726a0efac5c26d882ed341f5e"
)
TRAIN_SOURCE_SHA256 = (
    "8f27dcf9c1dc0c7f3ffa2e2941daec13d8a72c7204077f09420e15aacb02ffe0"
)
EVAL_SOURCE_SHA256 = (
    "f68e2fe6e4cf568e694925697dd5d65fe10160a52d357c641892de85ee5c3da7"
)
SOURCE_CAPTURE_CONTRACT_SHA256 = (
    "fe0b004ca6e48c9eb8163237a5ea20f45b33acdbd9a3a267a9fb4e670ed2c954"
)
SIDECAR_CONTENT_SHA256 = (
    "37c7b8c012f8fc0c1f63210986c0b3071a261261c88ee4bc1f502220ce3b8526"
)
SIDECAR_FILE_SHA256 = (
    "6486288677e81665b21815879729c424c6acb8c7d0e45be264b961a9d652fe3d"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")


class Experiment5ReportError(ValueError):
    """Experiment-5 result, paired evidence, cost, or report is invalid."""


@dataclass(frozen=True, slots=True)
class ValidatedResult:
    body: dict[str, Any]
    result_sha256: str
    panel: str
    seed: int
    milestone: int
    completed_at: datetime
    scores: dict[str, dict[str, float]]
    costs: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ValidatedPairedScores:
    body: dict[str, Any]
    artifact_sha256: str
    panel: str
    seed: int
    milestone: int
    sequence_ids: np.ndarray
    valid_tokens: np.ndarray
    sst: np.ndarray
    sse: dict[str, np.ndarray]


def _read_json(path: str | Path, *, name: str) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        value = json.loads(selected.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Experiment5ReportError(f"cannot read {name}: {selected}") from error
    if not isinstance(value, dict):
        raise Experiment5ReportError(f"{name} must be a JSON object")
    return value


def _mapping(
    value: Any,
    *,
    name: str,
    keys: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Experiment5ReportError(f"{name} must be an object")
    if keys is not None and set(value) != keys:
        raise Experiment5ReportError(
            f"{name} fields differ: expected {sorted(keys)}, got {sorted(value)}"
        )
    return value


def _finite(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Experiment5ReportError(f"{name} must be numeric")
    selected = float(value)
    if not math.isfinite(selected):
        raise Experiment5ReportError(f"{name} must be finite")
    if positive and selected <= 0:
        raise Experiment5ReportError(f"{name} must be positive")
    if minimum is not None and selected < minimum:
        raise Experiment5ReportError(f"{name} must be at least {minimum}")
    return selected


def _integer(
    value: Any,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Experiment5ReportError(f"{name} must be an integer")
    selected = int(value)
    if minimum is not None and selected < minimum:
        raise Experiment5ReportError(f"{name} must be at least {minimum}")
    return selected


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Experiment5ReportError(f"{name} must be a lowercase SHA-256")
    return value


def _git_sha(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _GIT_RE.fullmatch(value) is None:
        raise Experiment5ReportError(f"{name} must be a lowercase Git SHA")
    return value


def _timestamp(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str):
        raise Experiment5ReportError(f"{name} must be an RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Experiment5ReportError(
            f"{name} must be an RFC3339 timestamp"
        ) from error
    if parsed.tzinfo is None:
        raise Experiment5ReportError(f"{name} must include a timezone")
    return parsed


def _self_hashed(
    value: Mapping[str, Any],
    *,
    field: str,
    name: str,
) -> tuple[dict[str, Any], str]:
    body = dict(value)
    claimed = _sha256(body.pop(field, None), name=f"{name}.{field}")
    if canonical_sha256(body) != claimed:
        raise Experiment5ReportError(f"{name} self-hash is invalid")
    return body, claimed


def _score(value: Any, *, name: str) -> dict[str, float]:
    row = _mapping(
        value,
        name=name,
        keys={"sse", "sst", "nmse", "ev"},
    )
    sse = _finite(row["sse"], name=f"{name}.sse", minimum=0)
    sst = _finite(row["sst"], name=f"{name}.sst", positive=True)
    nmse = _finite(row["nmse"], name=f"{name}.nmse", minimum=0)
    ev = _finite(row["ev"], name=f"{name}.ev")
    expected_nmse = sse / sst
    if not math.isclose(nmse, expected_nmse, rel_tol=1e-10, abs_tol=1e-10):
        raise Experiment5ReportError(f"{name}.nmse differs from SSE/SST")
    if not math.isclose(ev, 1.0 - nmse, rel_tol=1e-10, abs_tol=1e-10):
        raise Experiment5ReportError(f"{name}.ev differs from 1-NMSE")
    return {"sse": sse, "sst": sst, "nmse": nmse, "ev": ev}


def _optional_score(value: Any, *, name: str) -> dict[str, float] | None:
    return None if value is None else _score(value, name=name)


def _coverage(value: Any, *, name: str) -> dict[str, float | int]:
    row = _mapping(
        value,
        name=name,
        keys={"fired_features", "total_features", "firing_fraction"},
    )
    fired = _integer(row["fired_features"], name=f"{name}.fired_features", minimum=0)
    total = _integer(row["total_features"], name=f"{name}.total_features", minimum=1)
    fraction = _finite(
        row["firing_fraction"],
        name=f"{name}.firing_fraction",
        minimum=0,
    )
    if fired > total or fraction > 1:
        raise Experiment5ReportError(f"{name} exceeds its total feature count")
    if not math.isclose(fraction, fired / total, rel_tol=1e-12, abs_tol=1e-12):
        raise Experiment5ReportError(f"{name}.firing_fraction differs from counts")
    return {
        "fired_features": fired,
        "total_features": total,
        "firing_fraction": fraction,
    }


def _heldout_nonfiring(value: Any, *, name: str) -> dict[str, float | int]:
    row = _mapping(
        value,
        name=name,
        keys={"fired_features", "total_features", "nonfiring_fraction"},
    )
    fired = _integer(row["fired_features"], name=f"{name}.fired_features", minimum=0)
    total = _integer(row["total_features"], name=f"{name}.total_features", minimum=1)
    fraction = _finite(
        row["nonfiring_fraction"],
        name=f"{name}.nonfiring_fraction",
        minimum=0,
    )
    if fired > total or fraction > 1:
        raise Experiment5ReportError(f"{name} exceeds its total feature count")
    expected = 1.0 - fired / total
    if not math.isclose(fraction, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise Experiment5ReportError(
            f"{name}.nonfiring_fraction differs from counts"
        )
    return {
        "fired_features": fired,
        "total_features": total,
        "nonfiring_fraction": fraction,
    }


def _expected_parent_values(panel: str) -> dict[str, str | None]:
    return {
        "train_target_manifest_sha256": TRAIN_TARGET_SHA256,
        "eval_target_manifest_sha256": EVAL_TARGET_SHA256,
        "train_source_overlay_manifest_sha256": (
            TRAIN_SOURCE_SHA256 if panel == "transcoder" else None
        ),
        "eval_source_overlay_manifest_sha256": (
            EVAL_SOURCE_SHA256 if panel == "transcoder" else None
        ),
        "source_overlay_capture_contract_sha256": (
            SOURCE_CAPTURE_CONTRACT_SHA256 if panel == "transcoder" else None
        ),
        "recomposition_sidecar_content_sha256": SIDECAR_CONTENT_SHA256,
        "recomposition_sidecar_file_sha256": SIDECAR_FILE_SHA256,
    }


def validate_experiment5_result(value: Mapping[str, Any]) -> ValidatedResult:
    """Validate one self-hashed joint H/R/B/F curve-point result."""

    body, result_sha = _self_hashed(
        value,
        field="result_sha256",
        name="Experiment-5 scientific result",
    )
    _mapping(
        body,
        name="Experiment-5 scientific result",
        keys={
            "schema_version",
            "protocol_id",
            "status",
            "panel",
            "seed",
            "checkpoint_target_tokens",
            "checkpoint_actual_tokens",
            "evaluation_tokens",
            "source_revision",
            "completed_at",
            "config_sha256",
            "parents",
            "common_full_ffn_sst",
            "conditions",
            "paired_effects",
            "predictions_artifact_sha256",
        },
    )
    if (
        body["schema_version"] != RESULT_SCHEMA_VERSION
        or body["protocol_id"] != PROTOCOL_ID
        or body["status"] != "curve_point_measured"
    ):
        raise Experiment5ReportError("scientific result identity changed")
    panel = body["panel"]
    seed = body["seed"]
    milestone = body["checkpoint_target_tokens"]
    if panel not in PANELS or seed not in SEEDS or milestone not in MILESTONES:
        raise Experiment5ReportError("scientific result panel/seed/milestone changed")
    if body["evaluation_tokens"] != EVAL_TOKENS:
        raise Experiment5ReportError("scientific result evaluation count changed")
    _git_sha(body["source_revision"], name="source_revision")
    _sha256(body["config_sha256"], name="config_sha256")
    completed_at = _timestamp(body["completed_at"], name="completed_at")

    actual = _mapping(
        body["checkpoint_actual_tokens"],
        name="checkpoint_actual_tokens",
        keys=set(CONDITIONS),
    )
    for condition in CONDITIONS:
        if _integer(
            actual[condition],
            name=f"checkpoint_actual_tokens.{condition}",
            minimum=1,
        ) < milestone:
            raise Experiment5ReportError(
                f"{condition} checkpoint precedes its target milestone"
            )

    parents = _mapping(
        body["parents"],
        name="parents",
        keys=set(_expected_parent_values(panel)),
    )
    for name, expected in _expected_parent_values(panel).items():
        if parents[name] != expected:
            raise Experiment5ReportError(f"parents.{name} changed")

    common_sst = _finite(
        body["common_full_ffn_sst"],
        name="common_full_ffn_sst",
        positive=True,
    )
    conditions = _mapping(
        body["conditions"],
        name="conditions",
        keys=set(CONDITIONS),
    )
    expected_layouts = {
        "H": (69_632, 8_192, HYBRID_EXPOSURE),
        "R": (69_632, 8_192, HYBRID_EXPOSURE),
        "B": (8_192, 8_192, 1.0),
        "F": (69_632, 69_632, 1.0),
    }
    scores: dict[str, dict[str, float]] = {}
    costs: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        name = f"conditions.{condition}"
        row = _mapping(
            conditions[condition],
            name=name,
            keys={
                "condition",
                "total_features",
                "candidates_per_token",
                "nominal_exposure_fraction",
                "exact_l0",
                "finite",
                "full_ffn",
                "routed",
                "shared",
                "oracle_shared_full_ffn",
                "oracle_routed_full_ffn",
                "error_cancellation_sse_delta",
                "cumulative_training_coverage",
                "heldout_nonfiring",
                "routed_bank_exposure",
                "training",
            },
        )
        if row["condition"] != condition:
            raise Experiment5ReportError(f"{name}.condition changed")
        total, candidates, exposure = expected_layouts[condition]
        if (
            row["total_features"] != total
            or row["candidates_per_token"] != candidates
            or not math.isclose(
                _finite(
                    row["nominal_exposure_fraction"],
                    name=f"{name}.nominal_exposure_fraction",
                    positive=True,
                ),
                exposure,
                rel_tol=1e-15,
                abs_tol=1e-15,
            )
            or row["exact_l0"] != 64
            or row["finite"] is not True
        ):
            raise Experiment5ReportError(f"{name} layout or health changed")
        full = _score(row["full_ffn"], name=f"{name}.full_ffn")
        if not math.isclose(full["sst"], common_sst, rel_tol=1e-12, abs_tol=1e-8):
            raise Experiment5ReportError(f"{name} did not use the common SST")
        scores[condition] = full

        branch_names = (
            "routed",
            "shared",
            "oracle_shared_full_ffn",
            "oracle_routed_full_ffn",
        )
        branch_values = {
            branch: _optional_score(row[branch], name=f"{name}.{branch}")
            for branch in branch_names
        }
        if condition in {"H", "R"}:
            if any(branch_values[branch] is None for branch in branch_names):
                raise Experiment5ReportError(f"{name} lacks hybrid diagnostics")
            for oracle_name in (
                "oracle_shared_full_ffn",
                "oracle_routed_full_ffn",
            ):
                oracle = branch_values[oracle_name]
                if oracle is None:
                    raise Experiment5ReportError(
                        f"{name}.{oracle_name} is missing"
                    )
                if not math.isclose(
                    oracle["sst"],
                    common_sst,
                    rel_tol=1e-12,
                    abs_tol=1e-8,
                ):
                    raise Experiment5ReportError(
                        f"{name}.{oracle_name} did not use the common SST"
                    )
            _finite(
                row["error_cancellation_sse_delta"],
                name=f"{name}.error_cancellation_sse_delta",
            )
        elif any(branch_values[branch] is not None for branch in branch_names):
            raise Experiment5ReportError(f"{name} has inapplicable branch scores")
        elif row["error_cancellation_sse_delta"] is not None:
            raise Experiment5ReportError(
                f"{name} has inapplicable error-cancellation evidence"
            )

        training_coverage = _coverage(
            row["cumulative_training_coverage"],
            name=f"{name}.cumulative_training_coverage",
        )
        heldout = _heldout_nonfiring(
            row["heldout_nonfiring"],
            name=f"{name}.heldout_nonfiring",
        )
        if (
            training_coverage["total_features"] != total
            or heldout["total_features"] != total
        ):
            raise Experiment5ReportError(f"{name} coverage width changed")

        bank_rows = row["routed_bank_exposure"]
        if not isinstance(bank_rows, list):
            raise Experiment5ReportError(f"{name}.routed_bank_exposure must be a list")
        if condition in {"H", "R"}:
            if len(bank_rows) != 128:
                raise Experiment5ReportError(f"{name} must report all 128 banks")
            bank_ids: set[int] = set()
            training_components = 0
            evaluation_components = 0
            routed_training_fired = 0
            routed_heldout_fired = 0
            for index, bank_value in enumerate(bank_rows):
                bank_name = f"{name}.routed_bank_exposure[{index}]"
                bank = _mapping(
                    bank_value,
                    name=bank_name,
                    keys={
                        "bank_id",
                        "routed_training_components",
                        "routed_evaluation_components",
                        "cumulative_training_fired_features",
                        "heldout_fired_features",
                        "bank_features",
                    },
                )
                bank_id = _integer(bank["bank_id"], name=f"{bank_name}.bank_id")
                if not 0 <= bank_id < 128 or bank_id in bank_ids:
                    raise Experiment5ReportError(f"{bank_name}.bank_id is invalid")
                bank_ids.add(bank_id)
                if bank["bank_features"] != 512:
                    raise Experiment5ReportError(f"{bank_name}.bank_features changed")
                for count_name in (
                    "routed_training_components",
                    "routed_evaluation_components",
                    "cumulative_training_fired_features",
                    "heldout_fired_features",
                ):
                    count = _integer(
                        bank[count_name],
                        name=f"{bank_name}.{count_name}",
                        minimum=0,
                    )
                    if "fired_features" in count_name and count > 512:
                        raise Experiment5ReportError(
                            f"{bank_name}.{count_name} exceeds bank width"
                        )
                training_components += bank["routed_training_components"]
                evaluation_components += bank["routed_evaluation_components"]
                routed_training_fired += bank[
                    "cumulative_training_fired_features"
                ]
                routed_heldout_fired += bank["heldout_fired_features"]
            if training_components != actual[condition] * 8:
                raise Experiment5ReportError(
                    f"{name} routed training exposure differs from token count"
                )
            if evaluation_components != EVAL_TOKENS * 8:
                raise Experiment5ReportError(
                    f"{name} routed evaluation exposure differs from token count"
                )
            shared_training_fired = (
                training_coverage["fired_features"] - routed_training_fired
            )
            shared_heldout_fired = (
                heldout["fired_features"] - routed_heldout_fired
            )
            if (
                not 0 <= shared_training_fired <= 4_096
                or not 0 <= shared_heldout_fired <= 4_096
            ):
                raise Experiment5ReportError(
                    f"{name} routed/shared coverage does not reconcile"
                )
        elif bank_rows:
            raise Experiment5ReportError(f"{name} global layout has routed banks")

        training = _mapping(
            row["training"],
            name=f"{name}.training",
            keys={
                "marginal_chip_seconds",
                "wall_seconds",
                "tokens_per_second",
                "accelerator_type",
                "accelerator_generation",
                "chip_count",
                "useful_mfu",
                "training_receipt_sha256",
                "checkpoint_firing_mask_sha256",
            },
        )
        generation = training["accelerator_generation"]
        if generation not in {"v4", "v5e", "v6e"}:
            raise Experiment5ReportError(f"{name} accelerator generation changed")
        mfu = training["useful_mfu"]
        if mfu is not None:
            selected_mfu = _finite(mfu, name=f"{name}.training.useful_mfu", minimum=0)
            if selected_mfu > 1:
                raise Experiment5ReportError(f"{name}.training.useful_mfu exceeds one")
        costs[condition] = {
            "marginal_chip_seconds": _finite(
                training["marginal_chip_seconds"],
                name=f"{name}.training.marginal_chip_seconds",
                positive=True,
            ),
            "wall_seconds": _finite(
                training["wall_seconds"],
                name=f"{name}.training.wall_seconds",
                positive=True,
            ),
            "tokens_per_second": _finite(
                training["tokens_per_second"],
                name=f"{name}.training.tokens_per_second",
                positive=True,
            ),
            "accelerator_type": str(training["accelerator_type"]),
            "accelerator_generation": generation,
            "chip_count": _integer(
                training["chip_count"],
                name=f"{name}.training.chip_count",
                minimum=1,
            ),
            "useful_mfu": mfu,
            "training_receipt_sha256": _sha256(
                training["training_receipt_sha256"],
                name=f"{name}.training.training_receipt_sha256",
            ),
            "checkpoint_firing_mask_sha256": _sha256(
                training["checkpoint_firing_mask_sha256"],
                name=f"{name}.training.checkpoint_firing_mask_sha256",
            ),
        }

    effects = _mapping(
        body["paired_effects"],
        name="paired_effects",
        keys=set(CONTRASTS),
    )
    for contrast, (left, right) in CONTRASTS.items():
        name = f"paired_effects.{contrast}"
        row = _mapping(
            effects[contrast],
            name=name,
            keys={
                "endpoint",
                "estimate",
                "lower",
                "upper",
                "confidence_level",
                "bootstrap_unit",
                "bootstrap_replicates",
            },
        )
        estimate = _finite(row["estimate"], name=f"{name}.estimate")
        lower = _finite(row["lower"], name=f"{name}.lower")
        upper = _finite(row["upper"], name=f"{name}.upper")
        expected = scores[left]["ev"] - scores[right]["ev"]
        if (
            row["endpoint"] != "full_ffn"
            or row["confidence_level"] != CONFIDENCE_LEVEL
            or row["bootstrap_unit"] != "packed_sequence"
            or row["bootstrap_replicates"] != BOOTSTRAP_REPLICATES
            or not lower <= estimate <= upper
            or not math.isclose(
                estimate,
                expected,
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
        ):
            raise Experiment5ReportError(f"{name} is incompatible")
    _sha256(
        body["predictions_artifact_sha256"],
        name="predictions_artifact_sha256",
    )
    return ValidatedResult(
        body=body,
        result_sha256=result_sha,
        panel=panel,
        seed=seed,
        milestone=milestone,
        completed_at=completed_at,
        scores=scores,
        costs=costs,
    )


def validate_experiment5_paired_scores(
    value: Mapping[str, Any],
    *,
    result: ValidatedResult,
) -> ValidatedPairedScores:
    """Validate additive sequence scores and bind them to a curve result."""

    body, artifact_sha = _self_hashed(
        value,
        field="artifact_sha256",
        name="Experiment-5 paired scores",
    )
    _mapping(
        body,
        name="Experiment-5 paired scores",
        keys={
            "schema_version",
            "protocol_id",
            "panel",
            "seed",
            "checkpoint_target_tokens",
            "packed_sequence_ids",
            "valid_tokens",
            "sequence_sst",
            "sequence_sse",
        },
    )
    if (
        body["schema_version"] != PAIRED_SCHEMA_VERSION
        or body["protocol_id"] != PROTOCOL_ID
        or body["panel"] != result.panel
        or body["seed"] != result.seed
        or body["checkpoint_target_tokens"] != result.milestone
        or artifact_sha != result.body["predictions_artifact_sha256"]
    ):
        raise Experiment5ReportError("paired-score identity differs from result")
    ids_raw = body["packed_sequence_ids"]
    tokens_raw = body["valid_tokens"]
    sst_raw = body["sequence_sst"]
    if not all(isinstance(row, list) for row in (ids_raw, tokens_raw, sst_raw)):
        raise Experiment5ReportError("paired-score vectors must be lists")
    rows = len(ids_raw)
    if rows == 0 or len(tokens_raw) != rows or len(sst_raw) != rows:
        raise Experiment5ReportError("paired-score vector lengths differ")
    sequence_ids = np.asarray(ids_raw, dtype=np.int64)
    if np.any(sequence_ids < 0) or len(np.unique(sequence_ids)) != rows:
        raise Experiment5ReportError("packed sequence IDs are invalid")
    valid_tokens = np.asarray(tokens_raw, dtype=np.int64)
    if np.any(valid_tokens <= 0) or int(valid_tokens.sum()) != EVAL_TOKENS:
        raise Experiment5ReportError("paired valid-token count changed")
    sst = np.asarray(sst_raw, dtype=np.float64)
    if not np.all(np.isfinite(sst)) or np.any(sst <= 0):
        raise Experiment5ReportError("per-sequence SST must be finite and positive")
    if not math.isclose(
        float(sst.sum()),
        float(result.body["common_full_ffn_sst"]),
        rel_tol=1e-10,
        abs_tol=1e-7,
    ):
        raise Experiment5ReportError("paired SST differs from result SST")
    sse_rows = _mapping(
        body["sequence_sse"],
        name="sequence_sse",
        keys=set(CONDITIONS),
    )
    sse: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        raw = sse_rows[condition]
        if not isinstance(raw, list) or len(raw) != rows:
            raise Experiment5ReportError(
                f"sequence_sse.{condition} length differs"
            )
        selected = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(selected)) or np.any(selected < 0):
            raise Experiment5ReportError(
                f"sequence_sse.{condition} is invalid"
            )
        if not math.isclose(
            float(selected.sum()),
            result.scores[condition]["sse"],
            rel_tol=1e-10,
            abs_tol=1e-7,
        ):
            raise Experiment5ReportError(
                f"sequence_sse.{condition} differs from result"
            )
        sse[condition] = selected
    return ValidatedPairedScores(
        body=body,
        artifact_sha256=artifact_sha,
        panel=result.panel,
        seed=result.seed,
        milestone=result.milestone,
        sequence_ids=sequence_ids,
        valid_tokens=valid_tokens,
        sst=sst,
        sse=sse,
    )


def build_experiment5_budget_ledger(
    *,
    line_items: Sequence[Mapping[str, Any]],
    quota_funded_tpu_chip_seconds: float,
) -> dict[str, Any]:
    """Build the self-hashed incremental cash ledger consumed by the report."""

    checked_items: list[dict[str, Any]] = []
    for index, value in enumerate(line_items):
        row = _mapping(
            value,
            name=f"line_items[{index}]",
            keys={"category", "amount_usd"},
        )
        category = row["category"]
        if not isinstance(category, str) or not category.strip():
            raise Experiment5ReportError(
                f"line_items[{index}].category must be nonempty"
            )
        checked_items.append(
            {
                "category": category,
                "amount_usd": _finite(
                    row["amount_usd"],
                    name=f"line_items[{index}].amount_usd",
                    minimum=0,
                ),
            }
        )
    spent = math.fsum(row["amount_usd"] for row in checked_items)
    body = {
        "schema_version": BUDGET_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "currency": "USD",
        "hard_limit": HARD_BUDGET_USD,
        "incremental_cash_spent": spent,
        "quota_funded_tpu_chip_seconds": _finite(
            quota_funded_tpu_chip_seconds,
            name="quota_funded_tpu_chip_seconds",
            minimum=0,
        ),
        "line_items": checked_items,
    }
    return {**body, "ledger_sha256": canonical_sha256(body)}


def validate_experiment5_budget_ledger(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a self-hashed budget ledger without hiding an overrun."""

    body, ledger_sha = _self_hashed(
        value,
        field="ledger_sha256",
        name="Experiment-5 budget ledger",
    )
    _mapping(
        body,
        name="Experiment-5 budget ledger",
        keys={
            "schema_version",
            "protocol_id",
            "currency",
            "hard_limit",
            "incremental_cash_spent",
            "quota_funded_tpu_chip_seconds",
            "line_items",
        },
    )
    if (
        body["schema_version"] != BUDGET_SCHEMA_VERSION
        or body["protocol_id"] != PROTOCOL_ID
        or body["currency"] != "USD"
        or body["hard_limit"] != HARD_BUDGET_USD
    ):
        raise Experiment5ReportError("budget ledger identity changed")
    rebuilt = build_experiment5_budget_ledger(
        line_items=body["line_items"],
        quota_funded_tpu_chip_seconds=body["quota_funded_tpu_chip_seconds"],
    )
    if not math.isclose(
        _finite(
            body["incremental_cash_spent"],
            name="incremental_cash_spent",
            minimum=0,
        ),
        rebuilt["incremental_cash_spent"],
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise Experiment5ReportError("budget ledger total differs from line items")
    return {**body, "ledger_sha256": ledger_sha}


def _bootstrap_seed(*components: object) -> int:
    payload = "|".join((PROTOCOL_ID, *(str(value) for value in components)))
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big")


def _paired_bootstrap(
    scores: ValidatedPairedScores,
    *,
    contrast: str,
) -> tuple[float, float, float, np.ndarray]:
    left, right = CONTRASTS[contrast]
    estimate, lower, upper, samples = bootstrap_experiment5_paired_effect(
        panel=scores.panel,
        seed=scores.seed,
        milestone=scores.milestone,
        contrast=contrast,
        sequence_sst=scores.sst,
        left_sequence_sse=scores.sse[left],
        right_sequence_sse=scores.sse[right],
    )
    return estimate, lower, upper, samples


def bootstrap_experiment5_paired_effect(
    *,
    panel: str,
    seed: int,
    milestone: int,
    contrast: str,
    sequence_sst: Any,
    left_sequence_sse: Any,
    right_sequence_sse: Any,
) -> tuple[float, float, float, np.ndarray]:
    """Return the frozen paired sequence bootstrap for one seed/contrast."""

    if panel not in PANELS or seed not in SEEDS or milestone not in MILESTONES:
        raise Experiment5ReportError("paired bootstrap identity is invalid")
    if contrast not in CONTRASTS:
        raise Experiment5ReportError("paired bootstrap contrast is invalid")
    sst = np.asarray(sequence_sst, dtype=np.float64)
    left_sse = np.asarray(left_sequence_sse, dtype=np.float64)
    right_sse = np.asarray(right_sequence_sse, dtype=np.float64)
    if (
        sst.ndim != 1
        or not sst.size
        or left_sse.shape != sst.shape
        or right_sse.shape != sst.shape
        or not np.all(np.isfinite(sst))
        or not np.all(np.isfinite(left_sse))
        or not np.all(np.isfinite(right_sse))
        or np.any(sst <= 0)
        or np.any(left_sse < 0)
        or np.any(right_sse < 0)
    ):
        raise Experiment5ReportError("paired bootstrap rows are invalid")
    delta_sse = right_sse - left_sse
    estimate = float(delta_sse.sum() / sst.sum())
    rng = np.random.Generator(
        np.random.PCG64(
            _bootstrap_seed(
                panel,
                seed,
                milestone,
                contrast,
                "packed-sequence",
            )
        )
    )
    rows = len(sst)
    indices = rng.integers(
        0,
        rows,
        size=(BOOTSTRAP_REPLICATES, rows),
        dtype=np.int32,
    )
    denominator = sst[indices].sum(axis=1, dtype=np.float64)
    samples = delta_sse[indices].sum(axis=1, dtype=np.float64) / denominator
    lower, upper = np.percentile(samples, (2.5, 97.5), method="linear")
    return estimate, float(lower), float(upper), samples


def _hierarchical_effect(
    panel: str,
    contrast: str,
    results: Mapping[int, ValidatedResult],
    paired: Mapping[int, ValidatedPairedScores],
) -> dict[str, Any]:
    seed_samples: list[np.ndarray] = []
    effects: list[float] = []
    for seed in SEEDS:
        estimate, _lower, _upper, samples = _paired_bootstrap(
            paired[seed],
            contrast=contrast,
        )
        expected = (
            results[seed].scores[CONTRASTS[contrast][0]]["ev"]
            - results[seed].scores[CONTRASTS[contrast][1]]["ev"]
        )
        if not math.isclose(estimate, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise Experiment5ReportError(
                f"{panel}.{seed}.{contrast} paired scores differ from EV"
            )
        effects.append(estimate)
        seed_samples.append(samples)
    samples_by_seed = np.stack(seed_samples, axis=0)
    rng = np.random.Generator(
        np.random.PCG64(
            _bootstrap_seed(panel, 10_000_000, contrast, "hierarchical")
        )
    )
    selected_seed = rng.integers(
        0,
        len(SEEDS),
        size=(BOOTSTRAP_REPLICATES, len(SEEDS)),
        dtype=np.int32,
    )
    selected_inner = rng.integers(
        0,
        BOOTSTRAP_REPLICATES,
        size=(BOOTSTRAP_REPLICATES, len(SEEDS)),
        dtype=np.int32,
    )
    hierarchical = samples_by_seed[selected_seed, selected_inner].mean(axis=1)
    lower, upper = np.percentile(hierarchical, (2.5, 97.5), method="linear")
    positive = sum(effect > 0 for effect in effects)
    return {
        "mean": float(np.mean(effects, dtype=np.float64)),
        "lower": float(lower),
        "upper": float(upper),
        "positive_seed_count": positive,
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "passes_headline_rule": positive >= 2 and float(lower) > 0,
    }


def _validate_grid(
    results: Sequence[ValidatedResult],
    paired: Sequence[ValidatedPairedScores],
) -> tuple[
    dict[tuple[str, int, int], ValidatedResult],
    dict[tuple[str, int, int], ValidatedPairedScores],
]:
    expected = {
        (panel, seed, milestone)
        for panel in PANELS
        for seed in SEEDS
        for milestone in MILESTONES
    }
    result_grid = {
        (row.panel, row.seed, row.milestone): row
        for row in results
    }
    paired_grid = {
        (row.panel, row.seed, row.milestone): row
        for row in paired
    }
    if (
        len(result_grid) != len(results)
        or len(paired_grid) != len(paired)
        or set(result_grid) != expected
        or set(paired_grid) != expected
    ):
        raise Experiment5ReportError(
            "report requires exactly both panels x three seeds x three milestones"
        )
    revisions = {row.body["source_revision"] for row in results}
    if len(revisions) != 1:
        raise Experiment5ReportError("evaluation source revision changed across grid")
    for panel in PANELS:
        configs = {
            result_grid[(panel, seed, milestone)].body["config_sha256"]
            for seed in SEEDS
            for milestone in MILESTONES
        }
        parents = {
            canonical_sha256(result_grid[(panel, seed, milestone)].body["parents"])
            for seed in SEEDS
            for milestone in MILESTONES
        }
        if len(configs) != 1 or len(parents) != 1:
            raise Experiment5ReportError(f"{panel} changed config or parent artifacts")
        for seed in SEEDS:
            for condition in CONDITIONS:
                actual = [
                    result_grid[(panel, seed, milestone)].body[
                        "checkpoint_actual_tokens"
                    ][condition]
                    for milestone in MILESTONES
                ]
                costs = [
                    result_grid[(panel, seed, milestone)].costs[condition][
                        "marginal_chip_seconds"
                    ]
                    for milestone in MILESTONES
                ]
                if actual != sorted(actual) or len(set(actual)) != len(actual):
                    raise Experiment5ReportError(
                        f"{panel}.{seed}.{condition} tokens are not increasing"
                    )
                if any(right < left for left, right in zip(costs, costs[1:])):
                    raise Experiment5ReportError(
                        f"{panel}.{seed}.{condition} cumulative cost decreased"
                    )
        for milestone in MILESTONES:
            reference = paired_grid[(panel, 0, milestone)]
            for seed in SEEDS[1:]:
                row = paired_grid[(panel, seed, milestone)]
                basis = f"{panel}.{milestone}"
                if not np.array_equal(
                    row.sequence_ids,
                    reference.sequence_ids,
                ):
                    raise Experiment5ReportError(
                        f"{basis} packed-sequence IDs changed in seed {seed}; "
                        "exact cross-seed equality is required"
                    )
                if not np.array_equal(
                    row.valid_tokens,
                    reference.valid_tokens,
                ):
                    raise Experiment5ReportError(
                        f"{basis} valid-token counts changed in seed {seed}; "
                        "exact cross-seed equality is required"
                    )
                if not np.allclose(
                    row.sst,
                    reference.sst,
                    rtol=CROSS_HARDWARE_SST_RTOL,
                    atol=0.0,
                ):
                    raise Experiment5ReportError(
                        f"{basis} per-sequence SST changed beyond the "
                        f"cross-hardware tolerance in seed {seed} "
                        f"(rtol={CROSS_HARDWARE_SST_RTOL:g}, atol=0)"
                    )
                if not np.isclose(
                    np.sum(row.sst, dtype=np.float64),
                    np.sum(reference.sst, dtype=np.float64),
                    rtol=CROSS_HARDWARE_SST_SUM_RTOL,
                    atol=0.0,
                ):
                    raise Experiment5ReportError(
                        f"{basis} aggregate SST changed beyond the "
                        f"cross-hardware tolerance in seed {seed} "
                        f"(rtol={CROSS_HARDWARE_SST_SUM_RTOL:g}, atol=0)"
                    )
    return result_grid, paired_grid


def _milestone_summary(
    panel: str,
    milestone: int,
    grid: Mapping[tuple[str, int, int], ValidatedResult],
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        ev = [
            grid[(panel, seed, milestone)].scores[condition]["ev"]
            for seed in SEEDS
        ]
        nmse = [
            grid[(panel, seed, milestone)].scores[condition]["nmse"]
            for seed in SEEDS
        ]
        conditions[condition] = {
            "mean_ev": float(np.mean(ev, dtype=np.float64)),
            "mean_nmse": float(np.mean(nmse, dtype=np.float64)),
            "per_seed_ev": ev,
        }
    return {
        "checkpoint_tokens": milestone,
        "conditions": conditions,
    }


def _fixed_quality(
    panel: str,
    grid: Mapping[tuple[str, int, int], ValidatedResult],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    reached = 0
    hardware_mismatch = False
    for seed in SEEDS:
        f_final = grid[(panel, seed, 10_000_000)]
        reference = f_final.scores["F"]["ev"]
        selected_milestone: int | None = None
        for milestone in MILESTONES:
            if grid[(panel, seed, milestone)].scores["H"]["ev"] >= reference:
                selected_milestone = milestone
                break
        ratio: float | None = None
        same_generation = False
        if selected_milestone is not None:
            reached += 1
            h_row = grid[(panel, seed, selected_milestone)]
            same_generation = (
                h_row.costs["H"]["accelerator_generation"]
                == f_final.costs["F"]["accelerator_generation"]
            )
            if same_generation:
                ratio = (
                    100.0
                    * h_row.costs["H"]["marginal_chip_seconds"]
                    / f_final.costs["F"]["marginal_chip_seconds"]
                )
            else:
                hardware_mismatch = True
        rows.append(
            {
                "seed": seed,
                "reference_f_10m_ev": reference,
                "earliest_h_checkpoint_tokens": selected_milestone,
                "q5_percent": ratio,
                "same_accelerator_generation": same_generation,
            }
        )
    ratios = [row["q5_percent"] for row in rows if row["q5_percent"] is not None]
    if hardware_mismatch:
        status = "hardware_incomparable"
        median = None
    elif reached != len(SEEDS):
        status = "not_reached_all_seeds"
        median = None
    else:
        status = "defined"
        median = float(np.median(np.asarray(ratios, dtype=np.float64)))
    return {
        "status": status,
        "reaching_seed_count": reached,
        "per_seed": rows,
        "median_q5_percent": median,
        "interpolation_used": False,
    }


def _equal_token_cost(
    panel: str,
    grid: Mapping[tuple[str, int, int], ValidatedResult],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for milestone in MILESTONES:
        ratios: list[float | None] = []
        for seed in SEEDS:
            row = grid[(panel, seed, milestone)]
            comparable = (
                row.costs["H"]["accelerator_generation"]
                == row.costs["F"]["accelerator_generation"]
            )
            ratios.append(
                (
                    100.0
                    * row.costs["H"]["marginal_chip_seconds"]
                    / row.costs["F"]["marginal_chip_seconds"]
                )
                if comparable
                else None
            )
        all_comparable = all(value is not None for value in ratios)
        output.append(
            {
                "checkpoint_tokens": milestone,
                "per_seed_h_over_f_percent": ratios,
                "median_h_over_f_percent": (
                    float(np.median(np.asarray(ratios, dtype=np.float64)))
                    if all_comparable
                    else None
                ),
                "all_same_generation": all_comparable,
            }
        )
    return output


def _coverage_summary(
    panel: str,
    grid: Mapping[tuple[str, int, int], ValidatedResult],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in CONDITIONS:
        training = [
            grid[(panel, seed, 10_000_000)].body["conditions"][condition][
                "cumulative_training_coverage"
            ]["firing_fraction"]
            for seed in SEEDS
        ]
        heldout = [
            grid[(panel, seed, 10_000_000)].body["conditions"][condition][
                "heldout_nonfiring"
            ]["nonfiring_fraction"]
            for seed in SEEDS
        ]
        output[condition] = {
            "cumulative_training_firing_fraction": float(
                np.mean(training, dtype=np.float64)
            ),
            "heldout_nonfiring_fraction": float(
                np.mean(heldout, dtype=np.float64)
            ),
        }
    return output


def _panel_report(
    panel: str,
    result_grid: Mapping[tuple[str, int, int], ValidatedResult],
    paired_grid: Mapping[tuple[str, int, int], ValidatedPairedScores],
) -> dict[str, Any]:
    ten_results = {
        seed: result_grid[(panel, seed, 10_000_000)] for seed in SEEDS
    }
    ten_paired = {
        seed: paired_grid[(panel, seed, 10_000_000)] for seed in SEEDS
    }
    effects = {
        contrast: _hierarchical_effect(
            panel,
            contrast,
            ten_results,
            ten_paired,
        )
        for contrast in CONTRASTS
    }
    numerator = effects["H-B"]["mean"]
    denominator = effects["F-B"]["mean"]
    denominator_crosses_zero = (
        effects["F-B"]["lower"] <= 0 <= effects["F-B"]["upper"]
    )
    if denominator <= 0:
        gain_status = "undefined_nonpositive_denominator"
        gain = None
    elif denominator_crosses_zero:
        gain_status = "unstable"
        gain = 100.0 * numerator / denominator
    else:
        gain_status = "defined"
        gain = 100.0 * numerator / denominator
    fixed = _fixed_quality(panel, result_grid)
    return {
        "panel": panel,
        "complete": True,
        "milestones": [
            _milestone_summary(panel, milestone, result_grid)
            for milestone in MILESTONES
        ],
        "ten_million_contrasts": effects,
        "gain_recovery": {
            "status": gain_status,
            "g5_percent": gain,
            "numerator_mean": numerator,
            "denominator_mean": denominator,
            "denominator_interval_includes_zero": denominator_crosses_zero,
        },
        "fixed_quality_cost": fixed,
        "equal_token_cost": _equal_token_cost(panel, result_grid),
        "ten_million_coverage": _coverage_summary(panel, result_grid),
        "headline_ready": (
            effects["H-R"]["passes_headline_rule"]
            and effects["H-B"]["passes_headline_rule"]
        ),
    }


def build_experiment5_report(
    *,
    results: Sequence[Mapping[str, Any] | str | Path],
    paired_scores: Sequence[Mapping[str, Any] | str | Path],
    budget_ledger: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Build the self-hashed complete Experiment-5 JSON report."""

    checked_results = [
        validate_experiment5_result(
            _read_json(value, name="scientific result")
            if isinstance(value, (str, Path))
            else value
        )
        for value in results
    ]
    by_identity = {
        (row.panel, row.seed, row.milestone): row for row in checked_results
    }
    checked_paired: list[ValidatedPairedScores] = []
    for value in paired_scores:
        raw = (
            _read_json(value, name="paired scores")
            if isinstance(value, (str, Path))
            else value
        )
        identity = (
            raw.get("panel"),
            raw.get("seed"),
            raw.get("checkpoint_target_tokens"),
        )
        result = by_identity.get(identity)
        if result is None:
            raise Experiment5ReportError(
                f"paired scores have no matching scientific result: {identity}"
            )
        checked_paired.append(
            validate_experiment5_paired_scores(raw, result=result)
        )
    result_grid, paired_grid = _validate_grid(checked_results, checked_paired)
    budget_value = (
        _read_json(budget_ledger, name="budget ledger")
        if isinstance(budget_ledger, (str, Path))
        else budget_ledger
    )
    budget = validate_experiment5_budget_ledger(budget_value)

    panel_rows = {
        panel: _panel_report(panel, result_grid, paired_grid) for panel in PANELS
    }
    within_budget = budget["incremental_cash_spent"] <= HARD_BUDGET_USD
    primary = panel_rows["transcoder"]
    if not within_budget:
        status = "budget_stopped"
    elif primary["headline_ready"]:
        status = "headline_ready"
    elif primary["ten_million_contrasts"]["H-R"]["passes_headline_rule"]:
        status = "mechanism_only"
    else:
        status = "not_established"
    q5 = primary["fixed_quality_cost"]["median_q5_percent"]
    if status == "headline_ready" and q5 is not None:
        headline_text = (
            "The router-aligned routed/shared transcoder reaches the 10M-token "
            f"global full-FFN baseline at {q5:.2f}% of its measured marginal "
            "training cost while exposing 11.76% of the parameter-matched "
            "dictionary per token."
        )
    else:
        headline_text = (
            "Experiment 5 completed; consult the transcoder H-R/H-B effects "
            "and fixed-quality status before making an efficiency claim."
        )
    ordered_results = {
        panel: [
            result_grid[(panel, seed, milestone)].result_sha256
            for seed in SEEDS
            for milestone in MILESTONES
        ]
        for panel in PANELS
    }
    ordered_paired = {
        panel: [
            paired_grid[(panel, seed, milestone)].artifact_sha256
            for seed in SEEDS
            for milestone in MILESTONES
        ]
        for panel in PANELS
    }
    revisions = {row.body["source_revision"] for row in checked_results}
    completed = max(row.completed_at for row in checked_results)
    generated_at = completed.isoformat().replace("+00:00", "Z")
    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "source_revision": next(iter(revisions)),
        "generated_at": generated_at,
        "parents": {
            "sae_result_sha256s": ordered_results["sae"],
            "transcoder_result_sha256s": ordered_results["transcoder"],
            "sae_paired_score_sha256s": ordered_paired["sae"],
            "transcoder_paired_score_sha256s": ordered_paired["transcoder"],
            "sae_config_sha256": result_grid[("sae", 0, MILESTONES[0])].body[
                "config_sha256"
            ],
            "transcoder_config_sha256": result_grid[
                ("transcoder", 0, MILESTONES[0])
            ].body["config_sha256"],
        },
        "panels": panel_rows,
        "budget": {
            "currency": "USD",
            "hard_limit": HARD_BUDGET_USD,
            "incremental_cash_spent": budget["incremental_cash_spent"],
            "quota_funded_tpu_chip_seconds": budget[
                "quota_funded_tpu_chip_seconds"
            ],
            "within_limit": within_budget,
            "ledger_sha256": budget["ledger_sha256"],
        },
        "headline": {
            "primary_panel": "transcoder",
            "text": headline_text,
            "q5_percent": q5,
            "nominal_candidate_exposure_percent": 100.0 * HYBRID_EXPOSURE,
        },
    }
    report = {**body, "report_sha256": canonical_sha256(body)}
    validate_experiment5_report(report)
    return report


def validate_experiment5_report(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate report identity, self-hash, matrix summary, and budget truth."""

    body, report_sha = _self_hashed(
        value,
        field="report_sha256",
        name="Experiment-5 report",
    )
    _mapping(
        body,
        name="Experiment-5 report",
        keys={
            "schema_version",
            "protocol_id",
            "status",
            "source_revision",
            "generated_at",
            "parents",
            "panels",
            "budget",
            "headline",
        },
    )
    if (
        body["schema_version"] != REPORT_SCHEMA_VERSION
        or body["protocol_id"] != PROTOCOL_ID
        or body["status"]
        not in {
            "headline_ready",
            "mechanism_only",
            "not_established",
            "incomplete",
            "budget_stopped",
        }
    ):
        raise Experiment5ReportError("report identity changed")
    _git_sha(body["source_revision"], name="report.source_revision")
    _timestamp(body["generated_at"], name="report.generated_at")
    parents = _mapping(
        body["parents"],
        name="report.parents",
        keys={
            "sae_result_sha256s",
            "transcoder_result_sha256s",
            "sae_paired_score_sha256s",
            "transcoder_paired_score_sha256s",
            "sae_config_sha256",
            "transcoder_config_sha256",
        },
    )
    for name in (
        "sae_result_sha256s",
        "transcoder_result_sha256s",
        "sae_paired_score_sha256s",
        "transcoder_paired_score_sha256s",
    ):
        rows = parents[name]
        if (
            not isinstance(rows, list)
            or len(rows) != 9
            or len(set(rows)) != 9
            or any(_SHA256_RE.fullmatch(str(row)) is None for row in rows)
        ):
            raise Experiment5ReportError(f"report.parents.{name} is invalid")
    _sha256(parents["sae_config_sha256"], name="sae_config_sha256")
    _sha256(parents["transcoder_config_sha256"], name="transcoder_config_sha256")
    panels = _mapping(
        body["panels"],
        name="report.panels",
        keys=set(PANELS),
    )
    for panel in PANELS:
        row = _mapping(panels[panel], name=f"report.panels.{panel}")
        if (
            row.get("panel") != panel
            or row.get("complete") is not True
            or not isinstance(row.get("milestones"), list)
            or [
                point.get("checkpoint_tokens")
                for point in row["milestones"]
                if isinstance(point, Mapping)
            ]
            != list(MILESTONES)
        ):
            raise Experiment5ReportError(f"report.panels.{panel} is incomplete")
    budget = _mapping(
        body["budget"],
        name="report.budget",
        keys={
            "currency",
            "hard_limit",
            "incremental_cash_spent",
            "quota_funded_tpu_chip_seconds",
            "within_limit",
            "ledger_sha256",
        },
    )
    spent = _finite(
        budget["incremental_cash_spent"],
        name="report.budget.incremental_cash_spent",
        minimum=0,
    )
    if (
        budget["currency"] != "USD"
        or budget["hard_limit"] != HARD_BUDGET_USD
        or budget["within_limit"] is not (spent <= HARD_BUDGET_USD)
    ):
        raise Experiment5ReportError("report budget interpretation is false")
    _sha256(budget["ledger_sha256"], name="report.budget.ledger_sha256")
    headline = _mapping(
        body["headline"],
        name="report.headline",
        keys={
            "primary_panel",
            "text",
            "q5_percent",
            "nominal_candidate_exposure_percent",
        },
    )
    if (
        headline["primary_panel"] != "transcoder"
        or not isinstance(headline["text"], str)
        or not math.isclose(
            _finite(
                headline["nominal_candidate_exposure_percent"],
                name="report.headline.nominal_candidate_exposure_percent",
                positive=True,
            ),
            100.0 * HYBRID_EXPOSURE,
            rel_tol=1e-15,
            abs_tol=1e-15,
        )
    ):
        raise Experiment5ReportError("report headline is incompatible")
    if headline["q5_percent"] is not None:
        _finite(
            headline["q5_percent"],
            name="report.headline.q5_percent",
            minimum=0,
        )
    return {**body, "report_sha256": report_sha}


def render_experiment5_report(value: Mapping[str, Any]) -> str:
    """Render a validated Experiment-5 JSON report as concise Markdown."""

    report = validate_experiment5_report(value)
    lines = [
        "# Experiment 5: Routed–Shared Hybrid Sparse Models",
        "",
        report["headline"]["text"],
        "",
        f"Verdict: `{report['status']}`.",
        "",
    ]
    for panel in PANELS:
        selected = report["panels"][panel]
        lines.extend(
            [
                f"## {panel.upper()}",
                "",
                "| train tokens | H EV | R EV | B EV | F EV |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for point in selected["milestones"]:
            conditions = point["conditions"]
            lines.append(
                f"| {point['checkpoint_tokens']:,} | "
                f"{conditions['H']['mean_ev']:.6f} | "
                f"{conditions['R']['mean_ev']:.6f} | "
                f"{conditions['B']['mean_ev']:.6f} | "
                f"{conditions['F']['mean_ev']:.6f} |"
            )
        lines.extend(
            [
                "",
                "### 10M paired effects",
                "",
                "| contrast | mean | paired 95% interval | positive seeds |",
                "|---|---:|---:|---:|",
            ]
        )
        for contrast in CONTRASTS:
            row = selected["ten_million_contrasts"][contrast]
            lines.append(
                f"| {contrast} | {row['mean']:.6f} | "
                f"[{row['lower']:.6f}, {row['upper']:.6f}] | "
                f"{row['positive_seed_count']}/3 |"
            )
        gain = selected["gain_recovery"]
        fixed = selected["fixed_quality_cost"]
        gain_text = (
            "undefined"
            if gain["g5_percent"] is None
            else f"{gain['g5_percent']:.2f}% ({gain['status']})"
        )
        fixed_text = (
            "undefined"
            if fixed["median_q5_percent"] is None
            else f"{fixed['median_q5_percent']:.2f}%"
        )
        lines.extend(
            [
                "",
                f"G5: {gain_text}.",
                (
                    f"Fixed-quality Q5: {fixed_text}; "
                    f"{fixed['reaching_seed_count']}/3 seeds reached F-at-10M."
                ),
                "",
                "### 10M coverage",
                "",
                (
                    "| condition | cumulative training fired | "
                    "held-out nonfiring |"
                ),
                "|---|---:|---:|",
            ]
        )
        for condition in CONDITIONS:
            coverage = selected["ten_million_coverage"][condition]
            lines.append(
                f"| {condition} | "
                f"{100.0 * coverage['cumulative_training_firing_fraction']:.2f}% | "
                f"{100.0 * coverage['heldout_nonfiring_fraction']:.2f}% |"
            )
        lines.append("")
    budget = report["budget"]
    lines.extend(
        [
            "## Cost and identity",
            "",
            (
                f"Incremental cash spend: ${budget['incremental_cash_spent']:.2f} "
                f"of ${budget['hard_limit']:.2f}; within limit: "
                f"`{str(budget['within_limit']).lower()}`."
            ),
            (
                "Quota-funded TPU chip-seconds: "
                f"{budget['quota_funded_tpu_chip_seconds']:.1f}."
            ),
            f"Budget ledger: `{budget['ledger_sha256']}`",
            f"Report SHA-256: `{report['report_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_experiment5_report(
    output_root: str | Path,
    value: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Create one immutable directory containing JSON and Markdown."""

    report = validate_experiment5_report(value)
    markdown = render_experiment5_report(report)
    root = Path(output_root).expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=False, mode=0o700)
    except FileExistsError as error:
        raise Experiment5ReportError(
            f"immutable output root already exists: {root}"
        ) from error
    result_path = root / "experiment5-result.json"
    markdown_path = root / "EXPERIMENT_5.md"
    _create_only(result_path, canonical_json_bytes(report) + b"\n")
    _create_only(markdown_path, markdown.encode("utf-8"))
    return result_path, markdown_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="joint H/R/B/F result; provide exactly 18, in any order",
    )
    parser.add_argument(
        "--paired-scores",
        action="append",
        required=True,
        help="packed-sequence score sidecar; provide exactly 18, in any order",
    )
    parser.add_argument("--budget-ledger", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_experiment5_report(
        results=args.result,
        paired_scores=args.paired_scores,
        budget_ledger=args.budget_ledger,
    )
    result_path, markdown_path = write_experiment5_report(
        args.output_root,
        report,
    )
    print(
        json.dumps(
            {
                "markdown": str(markdown_path),
                "report": str(result_path),
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BUDGET_SCHEMA_VERSION",
    "Experiment5ReportError",
    "PAIRED_SCHEMA_VERSION",
    "PROTOCOL_ID",
    "REPORT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "build_experiment5_budget_ledger",
    "build_experiment5_report",
    "bootstrap_experiment5_paired_effect",
    "main",
    "render_experiment5_report",
    "validate_experiment5_budget_ledger",
    "validate_experiment5_paired_scores",
    "validate_experiment5_report",
    "validate_experiment5_result",
    "write_experiment5_report",
]
