"""Self-hashed Experiment-3 scientific result and human-readable report.

Every confidence interval is recomputed from the retained packed-sequence
score artifact.  Callers supply neither effects nor intervals as assertions.
The primary cost denominator is the measured marginal training ledger, never
an analytical FLOP estimate or an optimizer-only timing.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.contribution_transcoder_config import PROTOCOL_ID
from moe_sae.contribution_transcoder_evaluation import (
    CONDITIONS,
    ENDPOINTS,
    TranscoderPairedSequenceScores,
)

RESULT_SCHEMA_VERSION = "gemma4-contribution-transcoder-scientific-result-v2"
BOOTSTRAP_ALGORITHM = "paired-packed-sequence-percentile-bootstrap-v1"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2_026_072_700
CONTRASTS = (("A", "R"), ("A", "B"), ("A", "F"), ("F", "B"))
EXPERIMENT2_A_MINUS_R = 0.2194296741
EXPERIMENT2_A_MINUS_B = 0.0216359968
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class ContributionTranscoderReportError(ValueError):
    """The requested result is incomplete or scientifically inconsistent."""


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ContributionTranscoderReportError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContributionTranscoderReportError(f"{name} must be numeric")
    selected = float(value)
    if not math.isfinite(selected):
        raise ContributionTranscoderReportError(f"{name} must be finite")
    if minimum is not None and selected < minimum:
        raise ContributionTranscoderReportError(
            f"{name} must be at least {minimum}"
        )
    return selected


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContributionTranscoderReportError(
            f"{name} must be a positive integer"
        )
    return int(value)


def _strict_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ContributionTranscoderReportError(f"{name} must be boolean")
    return value


def _exact_fields(
    name: str,
    value: Any,
    fields: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        observed = sorted(value) if isinstance(value, Mapping) else type(value)
        raise ContributionTranscoderReportError(
            f"{name} fields differ: expected={sorted(fields)}, observed={observed}"
        )
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class TranscoderTrainingLedger:
    """Measured marginal condition cost plus explicitly excluded timings."""

    training_result_sha256: str
    timing_receipt_sha256: str
    chip_count: int
    preprocessing_wall_seconds: float
    optimizer_wall_seconds: float
    marginal_wall_seconds: float
    marginal_chip_seconds: float
    compilation_wall_seconds: float
    checkpoint_serialization_wall_seconds: float
    evaluation_wall_seconds: float
    total_wall_seconds: float
    tokens_per_second: float
    useful_mfu: float

    def __post_init__(self) -> None:
        _sha256("training_result_sha256", self.training_result_sha256)
        _sha256("timing_receipt_sha256", self.timing_receipt_sha256)
        chips = _positive_int("chip_count", self.chip_count)
        for name in (
            "preprocessing_wall_seconds",
            "optimizer_wall_seconds",
            "marginal_wall_seconds",
            "compilation_wall_seconds",
            "checkpoint_serialization_wall_seconds",
            "evaluation_wall_seconds",
            "total_wall_seconds",
        ):
            _finite(name, getattr(self, name), minimum=0.0)
        if self.marginal_wall_seconds <= 0:
            raise ContributionTranscoderReportError(
                "marginal_wall_seconds must be positive"
            )
        marginal = _finite(
            "marginal_chip_seconds",
            self.marginal_chip_seconds,
            minimum=0.0,
        )
        if not math.isclose(
            marginal,
            chips * self.marginal_wall_seconds,
            rel_tol=1e-10,
            abs_tol=1e-6,
        ):
            raise ContributionTranscoderReportError(
                "marginal_chip_seconds must equal chip_count × marginal_wall_seconds"
            )
        if self.total_wall_seconds + 1e-9 < self.marginal_wall_seconds:
            raise ContributionTranscoderReportError(
                "total wall time cannot be below marginal wall time"
            )
        if _finite("tokens_per_second", self.tokens_per_second) <= 0:
            raise ContributionTranscoderReportError(
                "tokens_per_second must be positive"
            )
        mfu = _finite("useful_mfu", self.useful_mfu, minimum=0.0)
        if mfu > 1.0:
            raise ContributionTranscoderReportError(
                "useful_mfu must be expressed as a fraction in [0,1]"
            )

    @classmethod
    def from_value(
        cls,
        value: "TranscoderTrainingLedger | Mapping[str, Any]",
    ) -> "TranscoderTrainingLedger":
        if isinstance(value, cls):
            return value
        fields = {field.name for field in dataclasses.fields(cls)}
        return cls(**dict(_exact_fields("training ledger", value, fields)))


@dataclasses.dataclass(frozen=True, slots=True)
class TranscoderConditionSummary:
    """Hash-bound final checkpoint, evaluation, and sparse health."""

    condition: str
    checkpoint_sha256: str
    evaluation_receipt_sha256: str
    source_manifest_sha256: str
    target_manifest_sha256: str
    tokens: int
    component_sse: float
    component_sst: float
    routed_sse: float
    routed_sst: float
    oracle_shared_sse: float
    oracle_shared_sst: float
    l0_per_contribution: int
    total_l0: int
    fired_feature_fraction: float
    dead_rate: float
    macro_expert_ev: float
    supported_experts: int
    diagnostics_sha256: str
    source_only_encoding: bool
    finite_state: bool
    no_capacity_overflow: bool
    no_token_drops: bool

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ContributionTranscoderReportError("unknown condition")
        for name in (
            "checkpoint_sha256",
            "evaluation_receipt_sha256",
            "source_manifest_sha256",
            "target_manifest_sha256",
            "diagnostics_sha256",
        ):
            _sha256(name, getattr(self, name))
        _positive_int("tokens", self.tokens)
        for endpoint in ENDPOINTS:
            sse = _finite(
                f"{endpoint}_sse",
                getattr(self, f"{endpoint}_sse"),
                minimum=0.0,
            )
            sst = _finite(
                f"{endpoint}_sst",
                getattr(self, f"{endpoint}_sst"),
            )
            if sst <= 0 or not math.isfinite(sse / sst):
                raise ContributionTranscoderReportError(
                    f"{endpoint} requires positive SST and finite NMSE"
                )
        if self.l0_per_contribution != 2 or self.total_l0 != 16:
            raise ContributionTranscoderReportError(
                "Experiment 3 requires exact TopK2 per slot and total L0 16"
            )
        fired = _finite(
            "fired_feature_fraction",
            self.fired_feature_fraction,
            minimum=0.0,
        )
        dead = _finite("dead_rate", self.dead_rate, minimum=0.0)
        if (
            fired > 1
            or dead > 1
            or not math.isclose(fired + dead, 1.0, abs_tol=1e-10)
        ):
            raise ContributionTranscoderReportError(
                "feature coverage and dead rate must be complementary fractions"
            )
        _finite("macro_expert_ev", self.macro_expert_ev)
        if not 0 < self.supported_experts <= 128:
            raise ContributionTranscoderReportError(
                "supported_experts must be in [1,128]"
            )
        for name in (
            "source_only_encoding",
            "finite_state",
            "no_capacity_overflow",
            "no_token_drops",
        ):
            _strict_bool(name, getattr(self, name))

    @classmethod
    def from_value(
        cls,
        value: "TranscoderConditionSummary | Mapping[str, Any]",
    ) -> "TranscoderConditionSummary":
        if isinstance(value, cls):
            return value
        fields = {field.name for field in dataclasses.fields(cls)}
        return cls(**dict(_exact_fields("condition summary", value, fields)))

    def endpoint(self, name: str) -> dict[str, float]:
        sse = float(getattr(self, f"{name}_sse"))
        sst = float(getattr(self, f"{name}_sst"))
        return {
            "sse": sse,
            "sst": sst,
            "nmse": sse / sst,
            "ev": 1.0 - sse / sst,
        }


def build_condition_summary(
    scores: TranscoderPairedSequenceScores,
    *,
    condition: str,
    checkpoint_sha256: str,
    evaluation_receipt_sha256: str,
    fired_feature_fraction: float,
    macro_expert_ev: float,
    supported_experts: int,
    diagnostics_sha256: str,
    source_only_encoding: bool = True,
    finite_state: bool = True,
    no_capacity_overflow: bool = True,
    no_token_drops: bool = True,
) -> TranscoderConditionSummary:
    if condition not in CONDITIONS:
        raise ContributionTranscoderReportError("unknown condition")
    endpoint_values: dict[str, float] = {}
    for endpoint in ENDPOINTS:
        endpoint_values[f"{endpoint}_sse"] = float(
            scores.condition_sse[condition][endpoint].sum(dtype=np.float64)
        )
        endpoint_values[f"{endpoint}_sst"] = float(
            scores.common_sst[endpoint].sum(dtype=np.float64)
        )
    return TranscoderConditionSummary(
        condition=condition,
        checkpoint_sha256=checkpoint_sha256,
        evaluation_receipt_sha256=evaluation_receipt_sha256,
        source_manifest_sha256=scores.source_manifest_sha256,
        target_manifest_sha256=scores.target_manifest_sha256,
        tokens=scores.tokens,
        **endpoint_values,
        l0_per_contribution=2,
        total_l0=16,
        fired_feature_fraction=fired_feature_fraction,
        dead_rate=1.0 - float(fired_feature_fraction),
        macro_expert_ev=macro_expert_ev,
        supported_experts=supported_experts,
        diagnostics_sha256=diagnostics_sha256,
        source_only_encoding=source_only_encoding,
        finite_state=finite_state,
        no_capacity_overflow=no_capacity_overflow,
        no_token_drops=no_token_drops,
    )


def _bootstrap_seed(endpoint: str) -> int:
    digest = canonical_sha256(
        {
            "protocol": RESULT_SCHEMA_VERSION,
            "base_seed": BOOTSTRAP_SEED,
            "endpoint": endpoint,
        }
    )
    return int(digest[:16], 16)


def _paired_endpoint_effects(
    scores: TranscoderPairedSequenceScores,
    endpoint: str,
    *,
    replicates: int,
    chunk_size: int = 128,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], np.ndarray]]:
    if endpoint not in ENDPOINTS:
        raise ContributionTranscoderReportError("unknown endpoint")
    selected_replicates = _positive_int("bootstrap_replicates", replicates)
    selected_chunk = _positive_int("bootstrap_chunk_size", chunk_size)
    clusters = scores.clusters
    common = scores.common_sst[endpoint]
    total_sst = float(common.sum(dtype=np.float64))
    numerators = {
        (left, right): (
            scores.condition_sse[right][endpoint]
            - scores.condition_sse[left][endpoint]
        )
        for left, right in CONTRASTS
    }
    samples = {
        contrast: np.empty(selected_replicates, dtype=np.float64)
        for contrast in CONTRASTS
    }
    rng = np.random.default_rng(_bootstrap_seed(endpoint))
    for start in range(0, selected_replicates, selected_chunk):
        end = min(start + selected_chunk, selected_replicates)
        draws = rng.integers(
            0,
            clusters,
            size=(end - start, clusters),
            endpoint=False,
        )
        denominator = common[draws].sum(axis=1, dtype=np.float64)
        if np.any(denominator <= 0):
            raise ContributionTranscoderReportError(
                "bootstrap draw produced non-positive common SST"
            )
        for contrast, numerator in numerators.items():
            samples[contrast][start:end] = (
                numerator[draws].sum(axis=1, dtype=np.float64) / denominator
            )
    rows = []
    for left, right in CONTRASTS:
        point = float(numerators[(left, right)].sum(dtype=np.float64) / total_sst)
        low, high = np.quantile(
            samples[(left, right)],
            (0.025, 0.975),
            method="linear",
        )
        rows.append(
            {
                "endpoint": endpoint,
                "left": left,
                "right": right,
                "effect": point,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "excludes_zero": bool(low > 0 or high < 0),
                "tokens": scores.tokens,
                "clusters": scores.clusters,
                "replicates": selected_replicates,
                "bootstrap_seed": _bootstrap_seed(endpoint),
                "bootstrap_algorithm": BOOTSTRAP_ALGORITHM,
                "score_artifact_sha256": scores.score_artifact_sha256,
            }
        )
    return rows, samples


def _effect(
    rows: list[dict[str, Any]],
    endpoint: str,
    left: str,
    right: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["endpoint"] == endpoint
        and row["left"] == left
        and row["right"] == right
    ]
    if len(matches) != 1:
        raise ContributionTranscoderReportError(
            f"missing unique {endpoint} {left}-{right} effect"
        )
    return matches[0]


def _ratio_payload(
    rows: list[dict[str, Any]],
    component_samples: Mapping[tuple[str, str], np.ndarray],
) -> dict[str, Any]:
    numerator = _effect(rows, "component", "A", "B")
    denominator = _effect(rows, "component", "F", "B")
    denominator_effect = float(denominator["effect"])
    interval_includes_zero = bool(
        denominator["ci95_low"] <= 0 <= denominator["ci95_high"]
    )
    if denominator_effect <= 0:
        return {
            "gain_recovery_percent": None,
            "gain_ci95_low_percent": None,
            "gain_ci95_high_percent": None,
            "status": "undefined",
            "reason": "component F-minus-B is nonpositive",
            "numerator_a_minus_b": float(numerator["effect"]),
            "denominator_f_minus_b": denominator_effect,
        }
    point = 100.0 * float(numerator["effect"]) / denominator_effect
    if interval_includes_zero:
        return {
            "gain_recovery_percent": point,
            "gain_ci95_low_percent": None,
            "gain_ci95_high_percent": None,
            "status": "unstable_exploratory",
            "reason": "paired component F-minus-B interval includes zero",
            "numerator_a_minus_b": float(numerator["effect"]),
            "denominator_f_minus_b": denominator_effect,
        }
    denominator_samples = component_samples[("F", "B")]
    numerator_samples = component_samples[("A", "B")]
    finite = np.isfinite(denominator_samples) & (denominator_samples != 0)
    ratios = 100.0 * numerator_samples[finite] / denominator_samples[finite]
    if ratios.size < 0.99 * denominator_samples.size:
        raise ContributionTranscoderReportError(
            "stable gain interval lost too many finite bootstrap ratios"
        )
    low, high = np.quantile(ratios, (0.025, 0.975), method="linear")
    return {
        "gain_recovery_percent": point,
        "gain_ci95_low_percent": float(low),
        "gain_ci95_high_percent": float(high),
        "status": "stable",
        "reason": None,
        "numerator_a_minus_b": float(numerator["effect"]),
        "denominator_f_minus_b": denominator_effect,
    }


def _validate_condition_matrix(
    scores: TranscoderPairedSequenceScores,
    summaries: Mapping[str, Any],
    ledgers: Mapping[str, Any],
) -> tuple[
    dict[str, TranscoderConditionSummary],
    dict[str, TranscoderTrainingLedger],
]:
    if set(summaries) != set(CONDITIONS) or set(ledgers) != set(CONDITIONS):
        raise ContributionTranscoderReportError(
            "result requires exactly A/R/B/F summaries and training ledgers"
        )
    checked_summaries = {
        condition: TranscoderConditionSummary.from_value(summaries[condition])
        for condition in CONDITIONS
    }
    checked_ledgers = {
        condition: TranscoderTrainingLedger.from_value(ledgers[condition])
        for condition in CONDITIONS
    }
    for condition, summary in checked_summaries.items():
        if summary.condition != condition:
            raise ContributionTranscoderReportError(
                "condition summary is stored under the wrong key"
            )
        if (
            summary.source_manifest_sha256 != scores.source_manifest_sha256
            or summary.target_manifest_sha256 != scores.target_manifest_sha256
            or summary.tokens != scores.tokens
        ):
            raise ContributionTranscoderReportError(
                "condition summary differs from paired score identity"
            )
        for endpoint in ENDPOINTS:
            expected_sse = float(
                scores.condition_sse[condition][endpoint].sum(dtype=np.float64)
            )
            expected_sst = float(
                scores.common_sst[endpoint].sum(dtype=np.float64)
            )
            if not math.isclose(
                getattr(summary, f"{endpoint}_sse"),
                expected_sse,
                rel_tol=1e-12,
                abs_tol=1e-8 * max(1.0, expected_sse),
            ) or not math.isclose(
                getattr(summary, f"{endpoint}_sst"),
                expected_sst,
                rel_tol=1e-12,
                abs_tol=1e-8 * max(1.0, expected_sst),
            ):
                raise ContributionTranscoderReportError(
                    f"{condition} {endpoint} aggregate differs from retained rows"
                )
    return checked_summaries, checked_ledgers


def build_transcoder_scientific_result(
    *,
    scores: TranscoderPairedSequenceScores,
    summaries: Mapping[
        str,
        TranscoderConditionSummary | Mapping[str, Any],
    ],
    ledgers: Mapping[str, TranscoderTrainingLedger | Mapping[str, Any]],
    source_revision: str,
    config_sha256: str,
    train_source_manifest_sha256: str,
    train_target_manifest_sha256: str,
    common_sst_receipt_sha256: str,
    shared_harvest_receipt_sha256: str,
    shared_harvest_chip_seconds: float,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Build the complete seed-0 transfer verdict from retained evidence."""

    if not isinstance(source_revision, str) or _REVISION_RE.fullmatch(
        source_revision
    ) is None:
        raise ContributionTranscoderReportError(
            "source_revision must be a lowercase 40-character Git commit"
        )
    for name, digest in (
        ("config_sha256", config_sha256),
        ("train_source_manifest_sha256", train_source_manifest_sha256),
        ("train_target_manifest_sha256", train_target_manifest_sha256),
        ("common_sst_receipt_sha256", common_sst_receipt_sha256),
        ("shared_harvest_receipt_sha256", shared_harvest_receipt_sha256),
    ):
        _sha256(name, digest)
    harvest_cost = _finite(
        "shared_harvest_chip_seconds",
        shared_harvest_chip_seconds,
        minimum=0.0,
    )
    checked_summaries, checked_ledgers = _validate_condition_matrix(
        scores,
        summaries,
        ledgers,
    )
    all_effects: list[dict[str, Any]] = []
    component_samples: dict[tuple[str, str], np.ndarray] | None = None
    for endpoint in ENDPOINTS:
        rows, samples = _paired_endpoint_effects(
            scores,
            endpoint,
            replicates=bootstrap_replicates,
        )
        all_effects.extend(rows)
        if endpoint == "component":
            component_samples = samples
    assert component_samples is not None
    gain = _ratio_payload(all_effects, component_samples)
    component_align = _effect(all_effects, "component", "A", "R")
    component_arch = _effect(all_effects, "component", "A", "B")
    routed_arch = _effect(all_effects, "routed", "A", "B")
    denominator = _effect(all_effects, "component", "F", "B")
    c3 = (
        100.0
        * checked_ledgers["A"].marginal_chip_seconds
        / checked_ledgers["F"].marginal_chip_seconds
    )
    checks = {
        "a_minus_r_component_positive_and_excludes_zero": bool(
            component_align["effect"] > 0
            and component_align["ci95_low"] > 0
        ),
        "a_minus_b_component_positive_and_excludes_zero": bool(
            component_arch["effect"] > 0 and component_arch["ci95_low"] > 0
        ),
        "a_minus_b_routed_positive": bool(routed_arch["effect"] > 0),
        "f_minus_b_positive_and_stable": bool(
            denominator["effect"] > 0
            and not (
                denominator["ci95_low"] <= 0 <= denominator["ci95_high"]
            )
        ),
        "g3_at_least_10_percent": bool(
            gain["gain_recovery_percent"] is not None
            and gain["gain_recovery_percent"] >= 10.0
        ),
        "source_only_exact_l0_and_health": all(
            summary.source_only_encoding
            and summary.l0_per_contribution == 2
            and summary.total_l0 == 16
            and summary.finite_state
            and summary.no_capacity_overflow
            and summary.no_token_drops
            for summary in checked_summaries.values()
        ),
    }
    verdict = "transfer_passed" if all(checks.values()) else "transfer_not_established"
    condition_rows = {}
    for condition in CONDITIONS:
        summary = checked_summaries[condition]
        condition_rows[condition] = {
            **dataclasses.asdict(summary),
            "endpoints": {
                endpoint: summary.endpoint(endpoint) for endpoint in ENDPOINTS
            },
            "training": dataclasses.asdict(checked_ledgers[condition]),
        }
    unsigned = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": verdict,
        "seed": 0,
        "source_revision": source_revision,
        "config_sha256": config_sha256,
        "parents": {
            "train_source_manifest_sha256": train_source_manifest_sha256,
            "train_target_manifest_sha256": train_target_manifest_sha256,
            "eval_source_manifest_sha256": scores.source_manifest_sha256,
            "eval_target_manifest_sha256": scores.target_manifest_sha256,
            "recomposition_sidecar_sha256": (
                scores.recomposition_sidecar_sha256
            ),
            "common_sst_receipt_sha256": common_sst_receipt_sha256,
        },
        "paired_scores": {
            "score_artifact_sha256": scores.score_artifact_sha256,
            "token_identity_sha256": scores.token_identity_sha256,
            "tokens": scores.tokens,
            "packed_sequence_clusters": scores.clusters,
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_algorithm": BOOTSTRAP_ALGORITHM,
        },
        "conditions": condition_rows,
        "paired_effects": all_effects,
        "primary": {
            "g3": gain,
            "c3_percent": c3,
            "nominal_candidate_exposure_percent": 6.25,
            "effect_retention": {
                "alignment_ratio": (
                    float(component_align["effect"]) / EXPERIMENT2_A_MINUS_R
                ),
                "architecture_ratio": (
                    float(component_arch["effect"]) / EXPERIMENT2_A_MINUS_B
                ),
                "experiment2_a_minus_r": EXPERIMENT2_A_MINUS_R,
                "experiment2_a_minus_b": EXPERIMENT2_A_MINUS_B,
            },
        },
        "cost": {
            "primary_denominator": "measured_marginal_training_chip_seconds",
            "shared_harvest_receipt_sha256": shared_harvest_receipt_sha256,
            "shared_harvest_chip_seconds": harvest_cost,
            "marginal_a_over_f_percent": c3,
            "end_to_end_one_seed_a_over_f_percent": (
                100.0
                * (
                    harvest_cost
                    + checked_ledgers["A"].marginal_chip_seconds
                )
                / (
                    harvest_cost
                    + checked_ledgers["F"].marginal_chip_seconds
                )
            ),
        },
        "verdict": {
            "checks": checks,
            "passed": bool(all(checks.values())),
            "label": verdict,
        },
    }
    return {
        **unsigned,
        "result_sha256": canonical_sha256(unsigned),
    }


def validate_transcoder_scientific_result(
    value: Mapping[str, Any],
    *,
    scores: TranscoderPairedSequenceScores | None = None,
) -> dict[str, Any]:
    """Validate self-identity and optionally replay every scientific field."""

    if not isinstance(value, Mapping):
        raise ContributionTranscoderReportError("result must be a mapping")
    selected = dict(value)
    claimed = selected.pop("result_sha256", None)
    _sha256("result_sha256", claimed)
    if canonical_sha256(selected) != claimed:
        raise ContributionTranscoderReportError("result self-hash mismatch")
    if selected.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ContributionTranscoderReportError("result schema is incompatible")
    if selected.get("protocol_id") != PROTOCOL_ID:
        raise ContributionTranscoderReportError("result protocol is incompatible")
    if scores is not None:
        if (
            selected.get("paired_scores", {}).get("score_artifact_sha256")
            != scores.score_artifact_sha256
        ):
            raise ContributionTranscoderReportError(
                "result names a different paired score artifact"
            )
        summaries = {
            condition: {
                name: row[name]
                for name in {
                    field.name
                    for field in dataclasses.fields(TranscoderConditionSummary)
                }
            }
            for condition, row in selected["conditions"].items()
        }
        ledgers = {
            condition: row["training"]
            for condition, row in selected["conditions"].items()
        }
        parents = selected["parents"]
        cost = selected["cost"]
        replay = build_transcoder_scientific_result(
            scores=scores,
            summaries=summaries,
            ledgers=ledgers,
            source_revision=selected["source_revision"],
            config_sha256=selected["config_sha256"],
            train_source_manifest_sha256=parents[
                "train_source_manifest_sha256"
            ],
            train_target_manifest_sha256=parents[
                "train_target_manifest_sha256"
            ],
            common_sst_receipt_sha256=parents[
                "common_sst_receipt_sha256"
            ],
            shared_harvest_receipt_sha256=cost[
                "shared_harvest_receipt_sha256"
            ],
            shared_harvest_chip_seconds=cost["shared_harvest_chip_seconds"],
            bootstrap_replicates=selected["paired_scores"][
                "bootstrap_replicates"
            ],
        )
        if replay != dict(value):
            raise ContributionTranscoderReportError(
                "result differs from replayed score/cost evidence"
            )
    return dict(value)


def render_transcoder_report(value: Mapping[str, Any]) -> str:
    """Render the concise operator/scientific handoff from a sealed result."""

    result = validate_transcoder_scientific_result(value)
    primary = result["primary"]
    gain = primary["g3"]
    gain_text = (
        "undefined"
        if gain["gain_recovery_percent"] is None
        else f'{gain["gain_recovery_percent"]:.2f}% ({gain["status"]})'
    )
    lines = [
        "# Experiment 3 result",
        "",
        f"Verdict: **{result['verdict']['label']}**",
        "",
        (
            "Router-conditioned A recovered "
            f"{gain_text} of the full-width F-over-B component gain at "
            f"{primary['c3_percent']:.2f}% of F's measured marginal "
            "training chip-seconds."
        ),
        "",
        "| condition | component EV | routed EV | oracle-shared EV | dead rate | chip-seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = result["conditions"][condition]
        lines.append(
            "| "
            + " | ".join(
                (
                    condition,
                    f"{row['endpoints']['component']['ev']:.6f}",
                    f"{row['endpoints']['routed']['ev']:.6f}",
                    f"{row['endpoints']['oracle_shared']['ev']:.6f}",
                    f"{row['dead_rate']:.4%}",
                    f"{row['training']['marginal_chip_seconds']:.1f}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Paired packed-sequence effects (left minus right):",
            "",
            "| endpoint | contrast | effect | 95% interval |",
            "|---|---|---:|---:|",
        ]
    )
    for row in result["paired_effects"]:
        lines.append(
            f"| {row['endpoint']} | {row['left']}-{row['right']} | "
            f"{row['effect']:.6f} | [{row['ci95_low']:.6f}, "
            f"{row['ci95_high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            f"Result SHA-256: `{result['result_sha256']}`",
            f"Paired-score SHA-256: `{result['paired_scores']['score_artifact_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_transcoder_result(
    result_path: str | Path,
    report_path: str | Path,
    value: Mapping[str, Any],
    *,
    scores: TranscoderPairedSequenceScores | None = None,
) -> tuple[Path, Path]:
    """Atomically publish a validated JSON result and its Markdown rendering."""

    result = validate_transcoder_scientific_result(value, scores=scores)
    destinations = (
        (
            Path(result_path),
            (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "ascii"
            ),
        ),
        (Path(report_path), render_transcoder_report(result).encode("utf-8")),
    )
    written = []
    for destination, payload in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        written.append(destination)
    return written[0], written[1]


__all__ = [
    "BOOTSTRAP_ALGORITHM",
    "BOOTSTRAP_REPLICATES",
    "CONTRASTS",
    "RESULT_SCHEMA_VERSION",
    "ContributionTranscoderReportError",
    "TranscoderConditionSummary",
    "TranscoderTrainingLedger",
    "build_condition_summary",
    "build_transcoder_scientific_result",
    "render_transcoder_report",
    "validate_transcoder_scientific_result",
    "write_transcoder_result",
]
