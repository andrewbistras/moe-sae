"""Evidence-bound Experiment-2 evaluation, gates, and efficiency reports.

The scientific resampling unit is a packed sequence.  Evaluation first seals
one row of sufficient scores per packed sequence; report construction then
computes every paired interval itself.  Callers cannot supply confidence
intervals as assertions.

Only an authoritative, fully gated Phase-4 report may render the confirmatory
headline.  Seed-0 pilots remain explicitly exploratory even when their
escalation gates pass.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from moe_sae.contribution_artifacts import canonical_sha256, token_identity_sha256
from moe_sae.contribution_input import (
    CONTRIBUTION_INPUT_SCHEMA_VERSION,
    LEGACY_CONTRIBUTION_INPUT_SCHEMA_VERSION,
    ContributionInputReceipt,
)
from moe_sae.contribution_performance import (
    MINIMUM_AUXILIARY_THROUGHPUT_FRACTION,
    MINIMUM_PERSISTED_GOODPUT_FRACTION,
)
from moe_sae.contribution_sae import (
    CONTRIBUTION_CONDITIONS,
    pilot_contribution_config,
    production_contribution_config,
)
from moe_sae.contribution_score_artifacts import (
    ContributionScoreArtifactError,
    ScoreArtifactRef,
    ScoreArtifactResolver,
    resolve_score_manifest_matrix,
)

REPORT_SCHEMA_VERSION = "gemma4-contribution-efficiency-report-v3"
PHASE1_RECEIPT_SCHEMA_VERSION = "gemma4-contribution-phase1-receipt-v1"
SCORE_SCHEMA_VERSION = "gemma4-contribution-sequence-scores-v1"
BOOTSTRAP_ALGORITHM = "paired-packed-sequence-percentile-bootstrap-v1"
HIERARCHICAL_BOOTSTRAP_ALGORITHM = (
    "paired-seed-plus-packed-sequence-percentile-bootstrap-v1"
)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2_026_072_400
NOMINAL_A_EXPOSURE_PERCENT = 6.25
MAX_MATERIAL_LATE_COLLAPSE_EV = 0.02

_ENDPOINTS = ("component", "routed", "full_oracle_shared", "full_learned_shared")
_PAIRED_ENDPOINTS = ("component", "routed")
_PHASES = ("phase2", "phase3", "phase4")
_BANK_IDENTITIES = {
    "A": "true_expert_id",
    "R": "per_sequence_randomized_expert_id",
    "B": "router_rank",
    "F": "router_rank",
}
_REQUIRED_EFFECTS = (
    ("component", "A", "R"),
    ("component", "A", "B"),
    ("component", "F", "B"),
    ("routed", "A", "B"),
)
_PHASE_PLAN: dict[str, dict[str, Any]] = {
    "phase2": {
        "scale": "pilot",
        "seeds": (0,),
        "training_tokens": 2_000_000,
        "evaluation_tokens": 250_000,
        "optimizer_updates": 123,
        "warmup_updates": 12,
        "masked_tail_tokens": 15_232,
        "checkpoint_tokens": (2_000_000,),
        "capture_tokens": 2_250_000,
        "parent_phase": None,
    },
    "phase3": {
        "scale": "full",
        "seeds": (0,),
        "training_tokens": 10_000_000,
        "evaluation_tokens": 1_000_000,
        "optimizer_updates": 611,
        "warmup_updates": 61,
        "masked_tail_tokens": 10_624,
        "checkpoint_tokens": (2_000_000, 5_000_000, 10_000_000),
        "capture_tokens": 11_000_000,
        "parent_phase": "phase2",
    },
    "phase4": {
        "scale": "full",
        "seeds": (1, 2, 3),
        "training_tokens": 50_000_000,
        "evaluation_tokens": 2_000_000,
        "optimizer_updates": 3_052,
        "warmup_updates": 250,
        "masked_tail_tokens": 3_968,
        "checkpoint_tokens": (
            2_000_000,
            5_000_000,
            10_000_000,
            20_000_000,
            30_000_000,
            40_000_000,
            50_000_000,
        ),
        "capture_tokens": 52_000_000,
        "parent_phase": "phase3",
    },
}
_PHASE1_CHECKS = (
    "all_capture_exactness_checks_pass",
    "bf16_roundtrip_bit_exact",
    "interrupted_resume_shard_identity",
    "multihost_ranges_no_gaps_or_duplicates",
    "no_routed_token_drops",
    "writer_queue_bounded",
    "persisted_goodput_ge_0.60_device_output",
    "no_recompilation_after_warmup",
    "auxiliary_tokens_per_second_ge_0.85_writer_disabled",
)


class ContributionReportError(ValueError):
    """Report input is incomplete, inconsistent, or scientifically invalid."""


def _finite(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContributionReportError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContributionReportError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ContributionReportError(f"{name} must be at least {minimum}")
    return result


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or int(value) <= 0:
        raise ContributionReportError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or int(value) < 0:
        raise ContributionReportError(f"{name} must be a nonnegative integer")
    return int(value)


def _strict_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ContributionReportError(f"{name} must be boolean")
    return value


def _sha256(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContributionReportError(f"{name} must be a lowercase SHA-256")
    return value


def _exact_mapping(
    name: str,
    value: Any,
    required: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContributionReportError(f"{name} must be a mapping")
    if set(value) != required:
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        raise ContributionReportError(
            f"{name} fields are not exact; missing={missing}, extra={extra}"
        )
    return value


def _endpoint(sse: float, sst: float) -> dict[str, float]:
    selected_sse = _finite("sse", sse, minimum=0.0)
    selected_sst = _finite("sst", sst)
    if selected_sst <= 0:
        raise ContributionReportError("SST must be positive")
    nmse = selected_sse / selected_sst
    return {
        "sse": selected_sse,
        "sst": selected_sst,
        "nmse": nmse,
        "ev": 1.0 - nmse,
    }


def _same_float(left: float, right: float) -> bool:
    scale = max(1.0, abs(float(left)), abs(float(right)))
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-10 * scale)


@dataclass(frozen=True, slots=True)
class CostLedger:
    """Auditable marginal timing and explicitly excluded cost components."""

    timing_receipt_sha256: str
    chip_count: int
    timed_wall_seconds: float
    marginal_chip_seconds: float
    compilation_chip_seconds: float
    checkpoint_serialization_chip_seconds: float
    evaluation_chip_seconds: float

    def __post_init__(self) -> None:
        _sha256("timing_receipt_sha256", self.timing_receipt_sha256)
        chips = _positive_int("chip_count", self.chip_count)
        wall = _finite("timed_wall_seconds", self.timed_wall_seconds)
        if wall <= 0:
            raise ContributionReportError("timed_wall_seconds must be positive")
        marginal = _finite(
            "marginal_chip_seconds",
            self.marginal_chip_seconds,
        )
        if marginal <= 0:
            raise ContributionReportError("marginal_chip_seconds must be positive")
        expected = chips * wall
        if not math.isclose(marginal, expected, rel_tol=1e-9, abs_tol=1e-6):
            raise ContributionReportError(
                "marginal_chip_seconds must equal chip_count * timed_wall_seconds"
            )
        for name in (
            "compilation_chip_seconds",
            "checkpoint_serialization_chip_seconds",
            "evaluation_chip_seconds",
        ):
            _finite(name, getattr(self, name), minimum=0.0)

    @classmethod
    def from_value(cls, value: CostLedger | Mapping[str, Any]) -> CostLedger:
        if isinstance(value, cls):
            return value
        required = {field.name for field in dataclasses.fields(cls)}
        selected = _exact_mapping("cost ledger", value, required)
        return cls(**dict(selected))

    def report_row(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class RunHealthEvidence:
    """Hash-bound numerical and coverage health for one training run."""

    health_receipt_sha256: str
    finite_state: bool
    exact_l0: bool
    no_capacity_overflow: bool
    no_token_drops: bool
    falling_loss: bool
    coverage_passed: bool

    def __post_init__(self) -> None:
        _sha256("health_receipt_sha256", self.health_receipt_sha256)
        for field in dataclasses.fields(self):
            if field.name != "health_receipt_sha256":
                _strict_bool(field.name, getattr(self, field.name))

    @property
    def passed(self) -> bool:
        return all(
            getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "health_receipt_sha256"
        )

    @classmethod
    def from_value(
        cls,
        value: RunHealthEvidence | Mapping[str, Any],
    ) -> RunHealthEvidence:
        if isinstance(value, cls):
            return value
        required = {field.name for field in dataclasses.fields(cls)}
        selected = _exact_mapping("run health evidence", value, required)
        return cls(**dict(selected))


@dataclass(frozen=True, slots=True)
class LearningCurvePoint:
    training_tokens: int
    sse: float
    sst: float
    checkpoint_sha256: str
    evaluation_capture_manifest_sha256: str

    def __post_init__(self) -> None:
        _positive_int("learning_curve.training_tokens", self.training_tokens)
        _endpoint(self.sse, self.sst)
        _sha256("learning_curve.checkpoint_sha256", self.checkpoint_sha256)
        _sha256(
            "learning_curve.evaluation_capture_manifest_sha256",
            self.evaluation_capture_manifest_sha256,
        )

    @property
    def endpoint(self) -> dict[str, float]:
        return _endpoint(self.sse, self.sst)

    @classmethod
    def from_value(
        cls,
        value: LearningCurvePoint | Mapping[str, Any],
    ) -> LearningCurvePoint:
        if isinstance(value, cls):
            return value
        required = {field.name for field in dataclasses.fields(cls)}
        selected = _exact_mapping("learning-curve point", value, required)
        return cls(**dict(selected))

    def report_row(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), **self.endpoint}


def _validate_input_receipt(
    value: ContributionInputReceipt | Mapping[str, Any],
) -> ContributionInputReceipt:
    if isinstance(value, ContributionInputReceipt):
        selected = dataclasses.asdict(value)
    else:
        required = {
            field.name for field in dataclasses.fields(ContributionInputReceipt)
        }
        selected = dict(_exact_mapping("contribution input receipt", value, required))
    claimed = _sha256(
        "input receipt_sha256",
        selected.pop("receipt_sha256"),
    )
    if selected.get("schema_version") not in {
        LEGACY_CONTRIBUTION_INPUT_SCHEMA_VERSION,
        CONTRIBUTION_INPUT_SCHEMA_VERSION,
    }:
        raise ContributionReportError(
            "contribution input receipt schema is incompatible"
        )
    if claimed != canonical_sha256(selected):
        raise ContributionReportError("contribution input receipt self-hash is invalid")
    _sha256(
        "input receipt capture_manifest_sha256",
        selected["capture_manifest_sha256"],
    )
    if selected["condition"] not in CONTRIBUTION_CONDITIONS:
        raise ContributionReportError("input receipt condition is invalid")
    _nonnegative_int("input receipt randomization_key", selected["randomization_key"])
    _sha256("input receipt layout_sha256", selected["layout_sha256"])
    for name in (
        "total_tokens",
        "global_microbatch_tokens",
        "microbatches_per_update",
        "planned_updates",
        "process_count",
        "local_rows_per_microbatch",
    ):
        _positive_int(f"input receipt {name}", selected[name])
    process_index = _nonnegative_int(
        "input receipt process_index",
        selected["process_index"],
    )
    if process_index >= selected["process_count"]:
        raise ContributionReportError(
            "input receipt process_index is outside process_count"
        )
    return ContributionInputReceipt(
        **selected,
        receipt_sha256=claimed,
    )


@dataclass(frozen=True, slots=True)
class ContributionRunEvaluation:
    """One condition/seed evaluated on the common sealed held-out capture."""

    condition: str
    seed: int
    scale: Literal["pilot", "full"]
    evaluation_tokens: int
    training_tokens: int
    optimizer_updates: int
    warmup_updates: int
    masked_tail_tokens: int
    checkpoint_tokens: tuple[int, ...]
    run_manifest_sha256: str
    config_sha256: str
    checkpoint_sha256: str
    shared_q_checkpoint_sha256: str
    input_receipts: tuple[ContributionInputReceipt, ...]
    component_sse: float
    component_sst: float
    routed_sse: float
    routed_sst: float
    full_oracle_shared_sse: float
    full_oracle_shared_sst: float
    full_learned_shared_sse: float
    full_learned_shared_sst: float
    l0_per_contribution: int
    total_l0: int
    dead_rate: float
    fired_feature_fraction: float
    macro_expert_ev: float
    learning_curve: tuple[LearningCurvePoint, ...]
    timing: CostLedger
    health: RunHealthEvidence

    def __post_init__(self) -> None:
        if self.condition not in CONTRIBUTION_CONDITIONS:
            raise ContributionReportError(
                f"condition must be one of {CONTRIBUTION_CONDITIONS}"
            )
        _nonnegative_int("seed", self.seed)
        if self.scale not in {"pilot", "full"}:
            raise ContributionReportError("scale must be 'pilot' or 'full'")
        _positive_int("evaluation_tokens", self.evaluation_tokens)
        _positive_int("training_tokens", self.training_tokens)
        _positive_int("optimizer_updates", self.optimizer_updates)
        _positive_int("warmup_updates", self.warmup_updates)
        _nonnegative_int("masked_tail_tokens", self.masked_tail_tokens)
        checkpoints = tuple(
            _positive_int("checkpoint token", item) for item in self.checkpoint_tokens
        )
        if checkpoints != tuple(sorted(set(checkpoints))):
            raise ContributionReportError(
                "checkpoint_tokens must be strictly increasing and unique"
            )
        if not checkpoints or checkpoints[-1] != self.training_tokens:
            raise ContributionReportError(
                "the final checkpoint must equal training_tokens"
            )
        for name in (
            "run_manifest_sha256",
            "config_sha256",
            "checkpoint_sha256",
            "shared_q_checkpoint_sha256",
        ):
            _sha256(name, getattr(self, name))
        receipts = tuple(_validate_input_receipt(item) for item in self.input_receipts)
        if not receipts:
            raise ContributionReportError(
                "every run requires its complete training input receipt set"
            )
        for endpoint in _ENDPOINTS:
            _endpoint(
                getattr(self, f"{endpoint}_sse"),
                getattr(self, f"{endpoint}_sst"),
            )
        expected_per_contribution = 2 if self.scale == "pilot" else 8
        expected_total = expected_per_contribution * 8
        if self.l0_per_contribution != expected_per_contribution:
            raise ContributionReportError(
                f"{self.scale} L0 per contribution must be {expected_per_contribution}"
            )
        if self.total_l0 != expected_total:
            raise ContributionReportError(
                f"{self.scale} source-token L0 must be {expected_total}"
            )
        dead = _finite("dead_rate", self.dead_rate)
        fired = _finite("fired_feature_fraction", self.fired_feature_fraction)
        if not 0.0 <= dead <= 1.0 or not 0.0 <= fired <= 1.0:
            raise ContributionReportError(
                "dead_rate and fired_feature_fraction must lie in [0,1]"
            )
        if not math.isclose(dead + fired, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ContributionReportError(
                "dead_rate and fired_feature_fraction must be complementary"
            )
        _finite("macro_expert_ev", self.macro_expert_ev)
        curve = tuple(
            LearningCurvePoint.from_value(item) for item in self.learning_curve
        )
        if tuple(item.training_tokens for item in curve) != checkpoints:
            raise ContributionReportError(
                "learning curve must contain exactly the retained checkpoints"
            )
        object.__setattr__(self, "checkpoint_tokens", checkpoints)
        object.__setattr__(self, "learning_curve", curve)
        object.__setattr__(self, "input_receipts", receipts)
        object.__setattr__(self, "timing", CostLedger.from_value(self.timing))
        object.__setattr__(self, "health", RunHealthEvidence.from_value(self.health))

    @classmethod
    def from_value(
        cls,
        value: ContributionRunEvaluation | Mapping[str, Any],
    ) -> ContributionRunEvaluation:
        if isinstance(value, cls):
            return value
        required = {field.name for field in dataclasses.fields(cls)}
        selected = dict(_exact_mapping("run evaluation", value, required))
        selected["checkpoint_tokens"] = tuple(selected["checkpoint_tokens"])
        selected["input_receipts"] = tuple(
            _validate_input_receipt(item) for item in selected["input_receipts"]
        )
        selected["learning_curve"] = tuple(
            LearningCurvePoint.from_value(item) for item in selected["learning_curve"]
        )
        selected["timing"] = CostLedger.from_value(selected["timing"])
        selected["health"] = RunHealthEvidence.from_value(selected["health"])
        return cls(**selected)

    def endpoint(self, name: str) -> dict[str, float]:
        if name not in _ENDPOINTS:
            raise ContributionReportError(f"unknown endpoint {name!r}")
        return _endpoint(
            getattr(self, f"{name}_sse"),
            getattr(self, f"{name}_sst"),
        )

    @property
    def layout(self) -> dict[str, Any]:
        factory = (
            pilot_contribution_config
            if self.scale == "pilot"
            else production_contribution_config
        )
        selected = factory(self.condition, randomization_key=0)
        reference_f = factory("F", randomization_key=0)
        return {
            "target": "raw_selected_expert_output",
            "bank_identity": _BANK_IDENTITIES[self.condition],
            "banks": selected.bank_count,
            "bank_width": selected.bank_width,
            "total_features": selected.total_features,
            "candidates_per_token": selected.candidates_per_token,
            "parameter_count": selected.parameter_count,
            "candidate_exposure_percent_of_f": 100.0
            * selected.candidates_per_token
            / reference_f.candidates_per_token,
            "reference_description": (
                "full-width route-rank-conditioned component dictionary"
                if self.condition == "F"
                else None
            ),
        }

    @property
    def tokens_per_second(self) -> float:
        return self.training_tokens / self.timing.timed_wall_seconds

    def report_row(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "seed": self.seed,
            "scale": self.scale,
            "evaluation_tokens": self.evaluation_tokens,
            "training_tokens": self.training_tokens,
            "optimizer_updates": self.optimizer_updates,
            "warmup_updates": self.warmup_updates,
            "masked_tail_tokens": self.masked_tail_tokens,
            "checkpoint_tokens": list(self.checkpoint_tokens),
            "run_manifest_sha256": self.run_manifest_sha256,
            "config_sha256": self.config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "shared_q_checkpoint_sha256": self.shared_q_checkpoint_sha256,
            "input_receipts": [
                dataclasses.asdict(item) for item in self.input_receipts
            ],
            "endpoints": {endpoint: self.endpoint(endpoint) for endpoint in _ENDPOINTS},
            "l0_per_contribution": self.l0_per_contribution,
            "total_l0": self.total_l0,
            "dead_rate": self.dead_rate,
            "fired_feature_fraction": self.fired_feature_fraction,
            "macro_expert_ev": self.macro_expert_ev,
            "learning_curve": [item.report_row() for item in self.learning_curve],
            "timing": self.timing.report_row(),
            "tokens_per_second": self.tokens_per_second,
            "health": {
                **dataclasses.asdict(self.health),
                "passed": self.health.passed,
            },
            "layout": self.layout,
        }


@dataclass(frozen=True, slots=True)
class SharedQEvaluation:
    """The one common shared-expert SAE trained once for a seed."""

    seed: int
    evaluation_tokens: int
    training_tokens: int
    checkpoint_tokens: tuple[int, ...]
    checkpoint_sha256: str
    config_sha256: str
    training_manifest_sha256: str
    evaluation_capture_manifest_sha256: str
    width: int
    l0: int
    sse: float
    sst: float
    dead_rate: float
    firing_entropy: float
    firing_gini: float
    decoder_coherence: float
    effective_rank: float
    learning_curve: tuple[LearningCurvePoint, ...]
    timing: CostLedger

    def __post_init__(self) -> None:
        _nonnegative_int("seed", self.seed)
        _positive_int("evaluation_tokens", self.evaluation_tokens)
        _positive_int("training_tokens", self.training_tokens)
        checkpoints = tuple(
            _positive_int("shared-Q checkpoint token", item)
            for item in self.checkpoint_tokens
        )
        if checkpoints != tuple(sorted(set(checkpoints))) or not checkpoints:
            raise ContributionReportError(
                "shared-Q checkpoint_tokens must be increasing and unique"
            )
        if checkpoints[-1] != self.training_tokens:
            raise ContributionReportError(
                "shared-Q final checkpoint must equal training_tokens"
            )
        for name in (
            "checkpoint_sha256",
            "config_sha256",
            "training_manifest_sha256",
            "evaluation_capture_manifest_sha256",
        ):
            _sha256(name, getattr(self, name))
        if self.width != 8_192:
            raise ContributionReportError("shared Q must have exactly 8,192 features")
        if self.l0 != 64:
            raise ContributionReportError("shared Q must use exact TopK-64")
        _endpoint(self.sse, self.sst)
        dead = _finite("dead_rate", self.dead_rate)
        gini = _finite("firing_gini", self.firing_gini)
        if not 0.0 <= dead <= 1.0 or not 0.0 <= gini <= 1.0:
            raise ContributionReportError(
                "shared-Q dead_rate and firing_gini must lie in [0,1]"
            )
        for name in ("firing_entropy", "decoder_coherence", "effective_rank"):
            _finite(name, getattr(self, name), minimum=0.0)
        curve = tuple(
            LearningCurvePoint.from_value(item) for item in self.learning_curve
        )
        if tuple(item.training_tokens for item in curve) != checkpoints:
            raise ContributionReportError(
                "shared-Q learning curve must contain retained checkpoints"
            )
        object.__setattr__(self, "checkpoint_tokens", checkpoints)
        object.__setattr__(self, "learning_curve", curve)
        object.__setattr__(self, "timing", CostLedger.from_value(self.timing))

    @classmethod
    def from_value(
        cls,
        value: SharedQEvaluation | Mapping[str, Any],
    ) -> SharedQEvaluation:
        if isinstance(value, cls):
            return value
        required = {field.name for field in dataclasses.fields(cls)}
        selected = dict(_exact_mapping("shared-Q evaluation", value, required))
        selected["checkpoint_tokens"] = tuple(selected["checkpoint_tokens"])
        selected["learning_curve"] = tuple(
            LearningCurvePoint.from_value(item) for item in selected["learning_curve"]
        )
        selected["timing"] = CostLedger.from_value(selected["timing"])
        return cls(**selected)

    @property
    def endpoint(self) -> dict[str, float]:
        return _endpoint(self.sse, self.sst)

    def report_row(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "evaluation_tokens": self.evaluation_tokens,
            "training_tokens": self.training_tokens,
            "checkpoint_tokens": list(self.checkpoint_tokens),
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "training_manifest_sha256": self.training_manifest_sha256,
            "evaluation_capture_manifest_sha256": (
                self.evaluation_capture_manifest_sha256
            ),
            "width": self.width,
            "l0": self.l0,
            **self.endpoint,
            "dead_rate": self.dead_rate,
            "firing_entropy": self.firing_entropy,
            "firing_gini": self.firing_gini,
            "decoder_coherence": self.decoder_coherence,
            "effective_rank": self.effective_rank,
            "learning_curve": [item.report_row() for item in self.learning_curve],
            "timing": self.timing.report_row(),
            "tokens_per_second": (
                self.training_tokens / self.timing.timed_wall_seconds
            ),
        }


def _immutable_array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    selected = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    selected.setflags(write=False)
    return selected


def _sequence_inventory_sha256(sequence_id: np.ndarray, token_count: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"gemma4-contribution-sequence-inventory-v1\0")
    digest.update(np.asarray(sequence_id, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(token_count, dtype="<u2").tobytes(order="C"))
    return digest.hexdigest()


def _score_artifact_sha256(
    *,
    seed: int,
    endpoint: str,
    capture_manifest_sha256: str,
    token_identity: str,
    sequence_id: np.ndarray,
    token_count: np.ndarray,
    common_sst: np.ndarray,
    condition_sse: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(SCORE_SCHEMA_VERSION.encode("ascii") + b"\0")
    digest.update(int(seed).to_bytes(8, "little", signed=False))
    digest.update(endpoint.encode("ascii") + b"\0")
    digest.update(bytes.fromhex(capture_manifest_sha256))
    digest.update(bytes.fromhex(token_identity))
    digest.update(np.asarray(sequence_id, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(token_count, dtype="<u2").tobytes(order="C"))
    digest.update(np.asarray(common_sst, dtype="<f8").tobytes(order="C"))
    for condition in CONTRIBUTION_CONDITIONS:
        digest.update(condition.encode("ascii"))
        digest.update(
            np.asarray(condition_sse[condition], dtype="<f8").tobytes(order="C")
        )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PairedSequenceScores:
    """Sealed per-packed-sequence sufficient scores for one seed/endpoint."""

    seed: int
    endpoint: Literal["component", "routed"]
    capture_manifest_sha256: str
    token_identity_sha256: str
    sequence_id: np.ndarray
    token_count: np.ndarray
    common_sst: np.ndarray
    condition_sse: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        _nonnegative_int("score seed", self.seed)
        if self.endpoint not in _PAIRED_ENDPOINTS:
            raise ContributionReportError("score endpoint must be component or routed")
        _sha256("score capture_manifest_sha256", self.capture_manifest_sha256)
        _sha256("score token_identity_sha256", self.token_identity_sha256)
        sequence = np.asarray(self.sequence_id)
        count = np.asarray(self.token_count)
        if sequence.ndim != 1 or not np.issubdtype(sequence.dtype, np.signedinteger):
            raise ContributionReportError(
                "sequence score IDs must be a signed-integer vector"
            )
        if count.shape != sequence.shape or not np.issubdtype(
            count.dtype,
            np.integer,
        ):
            raise ContributionReportError(
                "sequence token counts must be an aligned integer vector"
            )
        if not sequence.size or np.any(sequence < 0):
            raise ContributionReportError(
                "sequence score IDs must be nonempty and nonnegative"
            )
        if np.unique(sequence).size != sequence.size:
            raise ContributionReportError(
                "sequence score evidence must have one row per packed sequence"
            )
        if np.any(count <= 0) or np.any(count > 1_024):
            raise ContributionReportError(
                "packed-sequence token counts must lie in [1,1024]"
            )
        sst = np.asarray(self.common_sst, dtype=np.float64)
        if (
            sst.shape != sequence.shape
            or not np.isfinite(sst).all()
            or np.any(sst <= 0)
        ):
            raise ContributionReportError(
                "per-sequence common SST must be finite and strictly positive"
            )
        if not isinstance(self.condition_sse, Mapping) or set(
            self.condition_sse
        ) != set(CONTRIBUTION_CONDITIONS):
            raise ContributionReportError(
                "sequence scores require exact A/R/B/F SSE vectors"
            )
        condition_sse: dict[str, np.ndarray] = {}
        for condition in CONTRIBUTION_CONDITIONS:
            selected = np.asarray(self.condition_sse[condition], dtype=np.float64)
            if (
                selected.shape != sequence.shape
                or not np.isfinite(selected).all()
                or np.any(selected < 0)
            ):
                raise ContributionReportError(
                    f"{condition} per-sequence SSE must be finite and nonnegative"
                )
            condition_sse[condition] = _immutable_array(
                selected,
                dtype=np.dtype("<f8"),
            )
        object.__setattr__(
            self,
            "sequence_id",
            _immutable_array(sequence, dtype=np.dtype("<i8")),
        )
        object.__setattr__(
            self,
            "token_count",
            _immutable_array(count, dtype=np.dtype("<u2")),
        )
        object.__setattr__(
            self,
            "common_sst",
            _immutable_array(sst, dtype=np.dtype("<f8")),
        )
        object.__setattr__(self, "condition_sse", condition_sse)

    @classmethod
    def from_value(
        cls,
        value: PairedSequenceScores | Mapping[str, Any],
    ) -> PairedSequenceScores:
        if isinstance(value, cls):
            return value
        required = {field.name for field in dataclasses.fields(cls)}
        selected = _exact_mapping("paired sequence scores", value, required)
        return cls(**dict(selected))

    @property
    def tokens(self) -> int:
        return int(self.token_count.sum(dtype=np.int64))

    @property
    def clusters(self) -> int:
        return int(self.sequence_id.size)

    @property
    def sequence_inventory_sha256(self) -> str:
        return _sequence_inventory_sha256(self.sequence_id, self.token_count)

    @property
    def score_artifact_sha256(self) -> str:
        return _score_artifact_sha256(
            seed=self.seed,
            endpoint=self.endpoint,
            capture_manifest_sha256=self.capture_manifest_sha256,
            token_identity=self.token_identity_sha256,
            sequence_id=self.sequence_id,
            token_count=self.token_count,
            common_sst=self.common_sst,
            condition_sse=self.condition_sse,
        )

    def identity_row(self) -> dict[str, Any]:
        return {
            "schema_version": SCORE_SCHEMA_VERSION,
            "seed": self.seed,
            "endpoint": self.endpoint,
            "capture_manifest_sha256": self.capture_manifest_sha256,
            "token_identity_sha256": self.token_identity_sha256,
            "sequence_inventory_sha256": self.sequence_inventory_sha256,
            "score_artifact_sha256": self.score_artifact_sha256,
            "tokens": self.tokens,
            "clusters": self.clusters,
        }


def aggregate_paired_token_scores(
    sequence_id: Any,
    position: Any,
    common_token_sst: Any,
    condition_token_sse: Mapping[str, Any],
    *,
    seed: int,
    endpoint: Literal["component", "routed"],
    capture_manifest_sha256: str,
) -> PairedSequenceScores:
    """Aggregate aligned token scores into the only accepted bootstrap unit."""

    ids = np.asarray(sequence_id)
    positions = np.asarray(position)
    sst = np.asarray(common_token_sst, dtype=np.float64)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.signedinteger):
        raise ContributionReportError("sequence_id must be a signed-integer vector")
    if positions.shape != ids.shape or not np.issubdtype(positions.dtype, np.integer):
        raise ContributionReportError("position must be an aligned integer vector")
    if sst.shape != ids.shape:
        raise ContributionReportError("token SST must align with token identities")
    if not isinstance(condition_token_sse, Mapping) or set(condition_token_sse) != set(
        CONTRIBUTION_CONDITIONS
    ):
        raise ContributionReportError("token scores require exact A/R/B/F vectors")
    if not ids.size:
        raise ContributionReportError("paired token scores cannot be empty")
    canonical_ids = np.asarray(ids, dtype=np.int64)
    canonical_positions = np.asarray(positions, dtype=np.uint16)
    if np.any(positions < 0) or np.any(positions >= 1_024):
        raise ContributionReportError("token positions must lie in [0,1023]")
    try:
        identity = token_identity_sha256(canonical_ids, canonical_positions)
    except Exception as exc:
        raise ContributionReportError(str(exc)) from exc
    token_keys = np.rec.fromarrays(
        [canonical_ids, canonical_positions],
        names=("sequence_id", "position"),
    )
    if np.unique(token_keys).size != ids.size:
        raise ContributionReportError(
            "paired token scores contain duplicate scientific token identities"
        )
    if not np.isfinite(sst).all() or np.any(sst < 0):
        raise ContributionReportError("token SST must be finite and nonnegative")
    unique, inverse = np.unique(canonical_ids, return_inverse=True)
    counts = np.bincount(inverse, minlength=unique.size)
    cluster_sst = np.bincount(inverse, weights=sst, minlength=unique.size)
    clustered_sse: dict[str, np.ndarray] = {}
    for condition in CONTRIBUTION_CONDITIONS:
        selected = np.asarray(condition_token_sse[condition], dtype=np.float64)
        if (
            selected.shape != ids.shape
            or not np.isfinite(selected).all()
            or np.any(selected < 0)
        ):
            raise ContributionReportError(
                f"{condition} token SSE must be finite and nonnegative"
            )
        clustered_sse[condition] = np.bincount(
            inverse,
            weights=selected,
            minlength=unique.size,
        )
    return PairedSequenceScores(
        seed=seed,
        endpoint=endpoint,
        capture_manifest_sha256=capture_manifest_sha256,
        token_identity_sha256=identity,
        sequence_id=unique,
        token_count=counts,
        common_sst=cluster_sst,
        condition_sse=clustered_sse,
    )


@dataclass(frozen=True, slots=True)
class PairedSequenceEffect:
    """One report-computed paired EV difference and its evidence identity."""

    seed: int
    endpoint: Literal["component", "routed"]
    left: str
    right: str
    effect: float
    ci95_low: float
    ci95_high: float
    tokens: int
    clusters: int
    replicates: int
    bootstrap_seed: int
    bootstrap_algorithm: str
    capture_manifest_sha256: str
    token_identity_sha256: str
    sequence_inventory_sha256: str
    score_artifact_sha256: str

    def __post_init__(self) -> None:
        _nonnegative_int("effect seed", self.seed)
        if self.endpoint not in _PAIRED_ENDPOINTS:
            raise ContributionReportError("paired endpoint must be component or routed")
        if self.left not in CONTRIBUTION_CONDITIONS:
            raise ContributionReportError("left condition is invalid")
        if self.right not in CONTRIBUTION_CONDITIONS or self.right == self.left:
            raise ContributionReportError("right condition is invalid")
        _finite("effect", self.effect)
        low = _finite("ci95_low", self.ci95_low)
        high = _finite("ci95_high", self.ci95_high)
        if low > high:
            raise ContributionReportError("paired CI bounds are reversed")
        _positive_int("effect tokens", self.tokens)
        if _positive_int("clusters", self.clusters) < 2:
            raise ContributionReportError(
                "paired bootstrap requires at least two packed sequences"
            )
        _positive_int("replicates", self.replicates)
        _nonnegative_int("bootstrap_seed", self.bootstrap_seed)
        if self.bootstrap_algorithm != BOOTSTRAP_ALGORITHM:
            raise ContributionReportError("paired bootstrap algorithm is incompatible")
        for name in (
            "capture_manifest_sha256",
            "token_identity_sha256",
            "sequence_inventory_sha256",
            "score_artifact_sha256",
        ):
            _sha256(name, getattr(self, name))


def _derived_bootstrap_seed(seed: int) -> int:
    digest = hashlib.sha256(
        f"{BOOTSTRAP_ALGORITHM}:{BOOTSTRAP_SEED}:{seed}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _effect_key(
    seed: int,
    endpoint: str,
    left: str,
    right: str,
) -> tuple[int, str, str, str]:
    return seed, endpoint, left, right


def _effect_samples_for_seed(
    component: PairedSequenceScores,
    routed: PairedSequenceScores,
    *,
    replicates: int,
    chunk_size: int = 128,
) -> tuple[
    dict[tuple[int, str, str, str], PairedSequenceEffect],
    dict[tuple[int, str, str, str], np.ndarray],
]:
    if component.seed != routed.seed:
        raise ContributionReportError("paired endpoint seeds differ")
    if (
        component.sequence_inventory_sha256 != routed.sequence_inventory_sha256
        or component.token_identity_sha256 != routed.token_identity_sha256
    ):
        raise ContributionReportError(
            "component and routed scores use different sequence inventories"
        )
    selected_replicates = _positive_int("replicates", replicates)
    selected_chunk = _positive_int("chunk_size", chunk_size)
    sources = {"component": component, "routed": routed}
    points: dict[tuple[int, str, str, str], float] = {}
    numerators: dict[tuple[int, str, str, str], np.ndarray] = {}
    samples: dict[tuple[int, str, str, str], np.ndarray] = {}
    constant: dict[tuple[int, str, str, str], bool] = {}
    for endpoint, left, right in _REQUIRED_EFFECTS:
        source = sources[endpoint]
        key = _effect_key(source.seed, endpoint, left, right)
        numerator = source.condition_sse[right] - source.condition_sse[left]
        point = float(
            numerator.sum(dtype=np.float64) / source.common_sst.sum(dtype=np.float64)
        )
        points[key] = point
        numerators[key] = numerator
        samples[key] = np.empty(selected_replicates, dtype=np.float64)
        constant[key] = bool(
            np.allclose(
                numerator,
                point * source.common_sst,
                rtol=1e-13,
                atol=1e-13,
            )
        )
        if constant[key]:
            samples[key].fill(point)

    nonconstant = [key for key, is_constant in constant.items() if not is_constant]
    selected_seed = _derived_bootstrap_seed(component.seed)
    if nonconstant:
        rng = np.random.default_rng(selected_seed)
        clusters = component.clusters
        for start in range(0, selected_replicates, selected_chunk):
            end = min(start + selected_chunk, selected_replicates)
            draws = rng.integers(
                0,
                clusters,
                size=(end - start, clusters),
                endpoint=False,
            )
            for key in nonconstant:
                source = sources[key[1]]
                sampled_sst = source.common_sst[draws].sum(axis=1, dtype=np.float64)
                if np.any(sampled_sst <= 0):
                    raise ContributionReportError(
                        "a sequence-bootstrap replicate has zero common SST"
                    )
                samples[key][start:end] = (
                    numerators[key][draws].sum(
                        axis=1,
                        dtype=np.float64,
                    )
                    / sampled_sst
                )

    effects: dict[tuple[int, str, str, str], PairedSequenceEffect] = {}
    for key, values in samples.items():
        source = sources[key[1]]
        low, high = np.quantile(values, (0.025, 0.975), method="linear")
        effects[key] = PairedSequenceEffect(
            seed=key[0],
            endpoint=key[1],  # type: ignore[arg-type]
            left=key[2],
            right=key[3],
            effect=points[key],
            ci95_low=float(low),
            ci95_high=float(high),
            tokens=source.tokens,
            clusters=source.clusters,
            replicates=selected_replicates,
            bootstrap_seed=selected_seed,
            bootstrap_algorithm=BOOTSTRAP_ALGORITHM,
            capture_manifest_sha256=source.capture_manifest_sha256,
            token_identity_sha256=source.token_identity_sha256,
            sequence_inventory_sha256=source.sequence_inventory_sha256,
            score_artifact_sha256=source.score_artifact_sha256,
        )
    return effects, samples


def paired_sequence_bootstrap(
    scores: PairedSequenceScores | Mapping[str, Any],
    *,
    left: str,
    right: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 128,
) -> PairedSequenceEffect:
    """Bootstrap one comparison from sealed per-sequence evidence.

    This convenience function accepts an explicit seed for standalone
    diagnostics.  Authoritative reports instead use a domain-separated seed
    and shared draws for every contrast in a seed.
    """

    selected = PairedSequenceScores.from_value(scores)
    if left not in CONTRIBUTION_CONDITIONS:
        raise ContributionReportError("left condition is invalid")
    if right not in CONTRIBUTION_CONDITIONS or right == left:
        raise ContributionReportError("right condition is invalid")
    selected_replicates = _positive_int("replicates", replicates)
    selected_seed = _nonnegative_int("bootstrap_seed", bootstrap_seed)
    selected_chunk = _positive_int("chunk_size", chunk_size)
    numerator = selected.condition_sse[right] - selected.condition_sse[left]
    total_sst = float(selected.common_sst.sum(dtype=np.float64))
    point = float(numerator.sum(dtype=np.float64) / total_sst)
    samples = np.empty(selected_replicates, dtype=np.float64)
    if np.allclose(
        numerator,
        point * selected.common_sst,
        rtol=1e-13,
        atol=1e-13,
    ):
        samples.fill(point)
    else:
        rng = np.random.default_rng(selected_seed)
        for start in range(0, selected_replicates, selected_chunk):
            end = min(start + selected_chunk, selected_replicates)
            draws = rng.integers(
                0,
                selected.clusters,
                size=(end - start, selected.clusters),
                endpoint=False,
            )
            sampled_sst = selected.common_sst[draws].sum(axis=1, dtype=np.float64)
            samples[start:end] = (
                numerator[draws].sum(
                    axis=1,
                    dtype=np.float64,
                )
                / sampled_sst
            )
    low, high = np.quantile(samples, (0.025, 0.975), method="linear")
    return PairedSequenceEffect(
        seed=selected.seed,
        endpoint=selected.endpoint,
        left=left,
        right=right,
        effect=point,
        ci95_low=float(low),
        ci95_high=float(high),
        tokens=selected.tokens,
        clusters=selected.clusters,
        replicates=selected_replicates,
        bootstrap_seed=selected_seed,
        bootstrap_algorithm=BOOTSTRAP_ALGORITHM,
        capture_manifest_sha256=selected.capture_manifest_sha256,
        token_identity_sha256=selected.token_identity_sha256,
        sequence_inventory_sha256=selected.sequence_inventory_sha256,
        score_artifact_sha256=selected.score_artifact_sha256,
    )


def build_phase1_receipt(
    *,
    protocol_sha256: str,
    training_capture_manifest_sha256: str,
    evaluation_capture_manifest_sha256: str,
    recomposition_sidecar_sha256: str,
    checks: Mapping[str, Any],
    useful_prefix_mfu: float,
    device_output_tokens_per_second: float,
    persisted_tokens_per_second: float,
    writer_disabled_tokens_per_second: float,
    auxiliary_tokens_per_second: float,
    max_writer_backlog_batches: int,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a self-hashed Phase-1 exactness and writer-performance receipt."""

    for name, value in (
        ("protocol_sha256", protocol_sha256),
        (
            "training_capture_manifest_sha256",
            training_capture_manifest_sha256,
        ),
        (
            "evaluation_capture_manifest_sha256",
            evaluation_capture_manifest_sha256,
        ),
        ("recomposition_sidecar_sha256", recomposition_sidecar_sha256),
    ):
        _sha256(name, value)
    selected_checks = _exact_mapping(
        "Phase-1 checks",
        checks,
        set(_PHASE1_CHECKS),
    )
    normalized_checks = {
        name: _strict_bool(f"Phase-1 check {name}", selected_checks[name])
        for name in _PHASE1_CHECKS
    }
    performance = {
        "useful_prefix_mfu": _finite(
            "Phase-1 useful_prefix_mfu",
            useful_prefix_mfu,
        ),
        "device_output_tokens_per_second": _finite(
            "Phase-1 device_output_tokens_per_second",
            device_output_tokens_per_second,
        ),
        "persisted_tokens_per_second": _finite(
            "Phase-1 persisted_tokens_per_second",
            persisted_tokens_per_second,
        ),
        "writer_disabled_tokens_per_second": _finite(
            "Phase-1 writer_disabled_tokens_per_second",
            writer_disabled_tokens_per_second,
        ),
        "auxiliary_tokens_per_second": _finite(
            "Phase-1 auxiliary_tokens_per_second",
            auxiliary_tokens_per_second,
        ),
        "max_writer_backlog_batches": _nonnegative_int(
            "Phase-1 max_writer_backlog_batches",
            max_writer_backlog_batches,
        ),
    }
    for name in (
        "device_output_tokens_per_second",
        "persisted_tokens_per_second",
        "writer_disabled_tokens_per_second",
        "auxiliary_tokens_per_second",
    ):
        if performance[name] <= 0:
            raise ContributionReportError(f"Phase-1 {name} must be positive")
    if not 0.0 <= performance["useful_prefix_mfu"] <= 1.0:
        raise ContributionReportError("Phase-1 useful_prefix_mfu must lie in [0,1]")
    derived = {
        "persisted_goodput_ge_0.60_device_output": (
            performance["persisted_tokens_per_second"]
            >= MINIMUM_PERSISTED_GOODPUT_FRACTION
            * performance["device_output_tokens_per_second"]
        ),
        "auxiliary_tokens_per_second_ge_0.85_writer_disabled": (
            performance["auxiliary_tokens_per_second"]
            >= MINIMUM_AUXILIARY_THROUGHPUT_FRACTION
            * performance["writer_disabled_tokens_per_second"]
        ),
        "writer_queue_bounded": performance["max_writer_backlog_batches"] <= 2,
    }
    for name, result in derived.items():
        if normalized_checks[name] != result:
            raise ContributionReportError(
                f"Phase-1 check {name} disagrees with measured performance"
            )
    status = "passed" if all(normalized_checks.values()) else "failed"
    unsigned = {
        "schema_version": PHASE1_RECEIPT_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "status": status,
        "protocol_sha256": protocol_sha256,
        "training_capture_manifest_sha256": training_capture_manifest_sha256,
        "evaluation_capture_manifest_sha256": evaluation_capture_manifest_sha256,
        "recomposition_sidecar_sha256": recomposition_sidecar_sha256,
        "train_tokens": 100_000,
        "evaluation_tokens": 25_000,
        "checks": normalized_checks,
        "performance": performance,
    }
    _ensure_json_safe(unsigned)
    return {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}


def validate_phase1_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a Phase-1 receipt strongly enough to authorize Phase 2."""

    required = {
        "schema_version",
        "created_at",
        "status",
        "protocol_sha256",
        "training_capture_manifest_sha256",
        "evaluation_capture_manifest_sha256",
        "recomposition_sidecar_sha256",
        "train_tokens",
        "evaluation_tokens",
        "checks",
        "performance",
        "receipt_sha256",
    }
    selected = dict(_exact_mapping("Phase-1 receipt", value, required))
    claimed = _sha256("Phase-1 receipt_sha256", selected.pop("receipt_sha256"))
    if selected.get("schema_version") != PHASE1_RECEIPT_SCHEMA_VERSION:
        raise ContributionReportError("Phase-1 receipt schema is incompatible")
    if claimed != canonical_sha256(selected):
        raise ContributionReportError("Phase-1 receipt self-hash is invalid")
    for name in (
        "protocol_sha256",
        "training_capture_manifest_sha256",
        "evaluation_capture_manifest_sha256",
        "recomposition_sidecar_sha256",
    ):
        _sha256(f"Phase-1 {name}", selected[name])
    if selected["train_tokens"] != 100_000 or selected["evaluation_tokens"] != 25_000:
        raise ContributionReportError("Phase-1 receipt token counts are not exact")
    checks = _exact_mapping(
        "Phase-1 receipt checks",
        selected["checks"],
        set(_PHASE1_CHECKS),
    )
    if any(
        not _strict_bool(f"Phase-1 check {name}", checks[name])
        for name in _PHASE1_CHECKS
    ):
        if selected["status"] != "failed":
            raise ContributionReportError(
                "Phase-1 receipt status disagrees with failed checks"
            )
    elif selected["status"] != "passed":
        raise ContributionReportError(
            "Phase-1 receipt status disagrees with passing checks"
        )
    performance = _exact_mapping(
        "Phase-1 performance",
        selected["performance"],
        {
            "useful_prefix_mfu",
            "device_output_tokens_per_second",
            "persisted_tokens_per_second",
            "writer_disabled_tokens_per_second",
            "auxiliary_tokens_per_second",
            "max_writer_backlog_batches",
        },
    )
    mfu = _finite("Phase-1 useful_prefix_mfu", performance["useful_prefix_mfu"])
    if not 0.0 <= mfu <= 1.0:
        raise ContributionReportError("Phase-1 MFU must lie in [0,1]")
    device = _finite(
        "Phase-1 device_output_tokens_per_second",
        performance["device_output_tokens_per_second"],
    )
    persisted = _finite(
        "Phase-1 persisted_tokens_per_second",
        performance["persisted_tokens_per_second"],
    )
    disabled = _finite(
        "Phase-1 writer_disabled_tokens_per_second",
        performance["writer_disabled_tokens_per_second"],
    )
    auxiliary = _finite(
        "Phase-1 auxiliary_tokens_per_second",
        performance["auxiliary_tokens_per_second"],
    )
    backlog = _nonnegative_int(
        "Phase-1 max_writer_backlog_batches",
        performance["max_writer_backlog_batches"],
    )
    if device <= 0 or persisted <= 0 or disabled <= 0 or auxiliary <= 0:
        raise ContributionReportError(
            "Phase-1 throughput measurements must be strictly positive"
        )
    derived = {
        "persisted_goodput_ge_0.60_device_output": (
            persisted >= MINIMUM_PERSISTED_GOODPUT_FRACTION * device
        ),
        "auxiliary_tokens_per_second_ge_0.85_writer_disabled": (
            auxiliary >= MINIMUM_AUXILIARY_THROUGHPUT_FRACTION * disabled
        ),
        "writer_queue_bounded": backlog <= 2,
    }
    for name, result in derived.items():
        if checks[name] != result:
            raise ContributionReportError(
                f"Phase-1 check {name} disagrees with performance"
            )
    selected["receipt_sha256"] = claimed
    _ensure_json_safe(selected)
    return selected


def _normalize_runs(
    values: Iterable[ContributionRunEvaluation | Mapping[str, Any]],
    *,
    phase: str,
    training_capture_manifest_sha256: str,
    evaluation_capture_manifest_sha256: str,
) -> list[ContributionRunEvaluation]:
    if phase not in _PHASES:
        raise ContributionReportError(f"phase must be one of {_PHASES}")
    plan = _PHASE_PLAN[phase]
    rows = [ContributionRunEvaluation.from_value(value) for value in values]
    expected_keys = {
        (condition, seed)
        for condition in CONTRIBUTION_CONDITIONS
        for seed in plan["seeds"]
    }
    keys = [(row.condition, row.seed) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ContributionReportError(
            f"{phase} requires exactly condition/seed keys {sorted(expected_keys)}"
        )
    for row in rows:
        expected = {
            "scale": plan["scale"],
            "evaluation_tokens": plan["evaluation_tokens"],
            "training_tokens": plan["training_tokens"],
            "optimizer_updates": plan["optimizer_updates"],
            "warmup_updates": plan["warmup_updates"],
            "masked_tail_tokens": plan["masked_tail_tokens"],
            "checkpoint_tokens": plan["checkpoint_tokens"],
        }
        for name, wanted in expected.items():
            if getattr(row, name) != wanted:
                raise ContributionReportError(
                    f"{phase} {name} must be {wanted!r}, got {getattr(row, name)!r}"
                )
        receipts = row.input_receipts
        process_counts = {item.process_count for item in receipts}
        if len(process_counts) != 1:
            raise ContributionReportError(
                f"{phase} {row.condition} input receipts disagree on process_count"
            )
        process_count = next(iter(process_counts))
        if len(receipts) != process_count or {
            item.process_index for item in receipts
        } != set(range(process_count)):
            raise ContributionReportError(
                f"{phase} {row.condition} input receipt process set is incomplete"
            )
        randomization_keys = {item.randomization_key for item in receipts}
        if len(randomization_keys) != 1:
            raise ContributionReportError(
                f"{phase} {row.condition} input randomization keys differ"
            )
        randomization_key = next(iter(randomization_keys))
        if row.condition != "R" and randomization_key != 0:
            raise ContributionReportError(
                f"{row.condition} must use the canonical zero randomization key"
            )
        factory = (
            pilot_contribution_config
            if row.scale == "pilot"
            else production_contribution_config
        )
        expected_layout_sha256 = canonical_sha256(
            dataclasses.asdict(
                factory(
                    row.condition,
                    randomization_key=randomization_key,
                )
            )
        )
        for receipt in receipts:
            if 4_096 % receipt.process_count != 0:
                raise ContributionReportError(
                    f"{phase} {row.condition} process count does not divide "
                    "the canonical microbatch"
                )
            if receipt.capture_manifest_sha256 != training_capture_manifest_sha256:
                raise ContributionReportError(
                    f"{phase} {row.condition} input is not bound to the "
                    "common training capture"
                )
            if receipt.condition != row.condition:
                raise ContributionReportError(
                    f"{phase} run condition differs from its input receipt"
                )
            if receipt.layout_sha256 != expected_layout_sha256:
                raise ContributionReportError(
                    f"{phase} {row.condition} input layout is not canonical"
                )
            if (
                receipt.total_tokens != plan["training_tokens"]
                or receipt.global_microbatch_tokens != 4_096
                or receipt.microbatches_per_update != 4
                or receipt.planned_updates != plan["optimizer_updates"]
                or receipt.local_rows_per_microbatch != 4_096 // receipt.process_count
            ):
                raise ContributionReportError(
                    f"{phase} {row.condition} input schedule is not canonical"
                )
        for point in row.learning_curve:
            if (
                point.evaluation_capture_manifest_sha256
                != evaluation_capture_manifest_sha256
            ):
                raise ContributionReportError(
                    f"{phase} {row.condition} learning curve uses another "
                    "evaluation capture"
                )
        if row.learning_curve[-1].checkpoint_sha256 != row.checkpoint_sha256:
            raise ContributionReportError(
                f"{phase} {row.condition} final curve point does not bind "
                "the final checkpoint"
            )
    # The held-out capture and targets are common across every condition and
    # every training seed, not merely within a seed.
    if len({row.evaluation_tokens for row in rows}) != 1:
        raise ContributionReportError(
            "all confirmatory rows must use one held-out token count"
        )
    for endpoint in _ENDPOINTS:
        ssts = [getattr(row, f"{endpoint}_sst") for row in rows]
        if any(not _same_float(ssts[0], item) for item in ssts[1:]):
            raise ContributionReportError(
                f"all conditions and seeds must use one common {endpoint} SST"
            )
    if not _same_float(
        rows[0].full_oracle_shared_sst,
        rows[0].full_learned_shared_sst,
    ):
        raise ContributionReportError(
            "oracle-shared and learned-shared endpoints must use the same "
            "exact full-FFN target SST"
        )
    return sorted(rows, key=lambda row: (row.seed, row.condition))


def _normalize_shared_q(
    values: Iterable[SharedQEvaluation | Mapping[str, Any]],
    *,
    rows: Sequence[ContributionRunEvaluation],
    phase: str,
    training_capture_manifest_sha256: str,
    evaluation_capture_manifest_sha256: str,
) -> list[SharedQEvaluation]:
    plan = _PHASE_PLAN[phase]
    q_rows = [SharedQEvaluation.from_value(value) for value in values]
    if sorted(row.seed for row in q_rows) != list(plan["seeds"]):
        raise ContributionReportError(
            "shared Q must be trained exactly once for every routed seed"
        )
    run_by_seed: dict[int, list[ContributionRunEvaluation]] = {}
    for row in rows:
        run_by_seed.setdefault(row.seed, []).append(row)
    for q_row in q_rows:
        if q_row.evaluation_tokens != plan["evaluation_tokens"]:
            raise ContributionReportError(
                f"{phase} shared-Q evaluation token count is not exact"
            )
        if q_row.training_tokens != plan["training_tokens"]:
            raise ContributionReportError(
                f"{phase} shared-Q training token count is not exact"
            )
        if q_row.checkpoint_tokens != plan["checkpoint_tokens"]:
            raise ContributionReportError(
                f"{phase} shared-Q checkpoint cadence is not exact"
            )
        if q_row.training_manifest_sha256 != training_capture_manifest_sha256:
            raise ContributionReportError(
                "shared Q was not trained on the common training capture"
            )
        if (
            q_row.evaluation_capture_manifest_sha256
            != evaluation_capture_manifest_sha256
        ):
            raise ContributionReportError(
                "shared Q was not evaluated on the report capture"
            )
        if any(
            point.evaluation_capture_manifest_sha256
            != evaluation_capture_manifest_sha256
            for point in q_row.learning_curve
        ):
            raise ContributionReportError(
                "shared-Q learning curve uses another evaluation capture"
            )
        if q_row.learning_curve[-1].checkpoint_sha256 != q_row.checkpoint_sha256:
            raise ContributionReportError(
                "shared-Q final curve point does not bind the final checkpoint"
            )
        if any(
            row.shared_q_checkpoint_sha256 != q_row.checkpoint_sha256
            for row in run_by_seed[q_row.seed]
        ):
            raise ContributionReportError(
                f"seed {q_row.seed} conditions do not reference one common Q"
            )
    if any(not _same_float(q_rows[0].sst, item.sst) for item in q_rows[1:]):
        raise ContributionReportError(
            "all shared-Q seeds must use one common held-out SST"
        )
    return sorted(q_rows, key=lambda row: row.seed)


def _scores_from_retained_artifacts(
    references: Iterable[ScoreArtifactRef | Mapping[str, Any]],
    resolver: ScoreArtifactResolver,
    *,
    rows: Sequence[ContributionRunEvaluation],
    phase: str,
    protocol_sha256: str,
    training_capture_manifest_sha256: str,
    evaluation_capture_manifest_sha256: str,
    recomposition_sidecar_sha256: str,
) -> tuple[list[PairedSequenceScores], list[dict[str, Any]]]:
    """Resolve the immutable score matrix and derive report sufficient rows."""

    try:
        selected_refs = tuple(ScoreArtifactRef.from_value(item) for item in references)
        matrix = resolve_score_manifest_matrix(
            selected_refs,
            resolver,
            expected_seeds=_PHASE_PLAN[phase]["seeds"],
        )
    except ContributionScoreArtifactError as error:
        raise ContributionReportError(
            f"retained sequence-score evidence is invalid: {error}"
        ) from error
    by_run = {(row.seed, row.condition): row for row in rows}
    for key, artifact in matrix.items():
        row = by_run[key]
        randomization_keys = {
            receipt.randomization_key for receipt in row.input_receipts
        }
        layout_hashes = {receipt.layout_sha256 for receipt in row.input_receipts}
        training_receipts = tuple(
            sorted(receipt.receipt_sha256 for receipt in row.input_receipts)
        )
        if (
            artifact.phase != phase
            or artifact.protocol_sha256 != protocol_sha256
            or artifact.training_capture_manifest_sha256
            != training_capture_manifest_sha256
            or artifact.evaluation_capture_manifest_sha256
            != evaluation_capture_manifest_sha256
            or artifact.recomposition_sidecar_sha256 != recomposition_sidecar_sha256
            or artifact.run_manifest_sha256 != row.run_manifest_sha256
            or artifact.config_sha256 != row.config_sha256
            or artifact.checkpoint_sha256 != row.checkpoint_sha256
            or len(randomization_keys) != 1
            or artifact.randomization_key != next(iter(randomization_keys))
            or len(layout_hashes) != 1
            or artifact.layout_sha256 != next(iter(layout_hashes))
            or artifact.training_input_receipt_sha256s != training_receipts
            or artifact.tokens != _PHASE_PLAN[phase]["evaluation_tokens"]
        ):
            raise ContributionReportError(
                f"retained score manifest {key} differs from its report run "
                "or frozen experiment identity"
            )
    result: list[PairedSequenceScores] = []
    for seed in _PHASE_PLAN[phase]["seeds"]:
        source = matrix[(seed, "A")]
        for endpoint in _PAIRED_ENDPOINTS:
            result.append(
                PairedSequenceScores(
                    seed=seed,
                    endpoint=endpoint,  # type: ignore[arg-type]
                    capture_manifest_sha256=(evaluation_capture_manifest_sha256),
                    token_identity_sha256=source.token_identity_sha256,
                    sequence_id=source.sequence_id,
                    token_count=source.token_count,
                    common_sst=getattr(source, f"{endpoint}_sst"),
                    condition_sse={
                        condition: getattr(
                            matrix[(seed, condition)],
                            f"{endpoint}_sse",
                        )
                        for condition in CONTRIBUTION_CONDITIONS
                    },
                )
            )
    ordered_references = [
        matrix[(seed, condition)].reference.to_dict()
        for seed in _PHASE_PLAN[phase]["seeds"]
        for condition in CONTRIBUTION_CONDITIONS
    ]
    return result, ordered_references


def _normalize_scores(
    values: Iterable[PairedSequenceScores | Mapping[str, Any]],
    *,
    rows: Sequence[ContributionRunEvaluation],
    phase: str,
    evaluation_capture_manifest_sha256: str,
) -> dict[tuple[int, str], PairedSequenceScores]:
    scores = [PairedSequenceScores.from_value(value) for value in values]
    plan = _PHASE_PLAN[phase]
    required = {
        (seed, endpoint) for seed in plan["seeds"] for endpoint in _PAIRED_ENDPOINTS
    }
    result = {(item.seed, item.endpoint): item for item in scores}
    if len(result) != len(scores) or set(result) != required:
        raise ContributionReportError(
            "paired sequence scores require component and routed evidence "
            "exactly once for every seed"
        )
    by_run = {(row.condition, row.seed): row for row in rows}
    common_identity: str | None = None
    common_inventory: str | None = None
    common_sst: dict[str, np.ndarray] = {}
    for key in sorted(result):
        score = result[key]
        if score.capture_manifest_sha256 != evaluation_capture_manifest_sha256:
            raise ContributionReportError(
                f"paired score {key} is not bound to the report capture"
            )
        if score.tokens != plan["evaluation_tokens"]:
            raise ContributionReportError(
                f"paired score {key} has {score.tokens} tokens; "
                f"{plan['evaluation_tokens']} required"
            )
        if common_identity is None:
            common_identity = score.token_identity_sha256
            common_inventory = score.sequence_inventory_sha256
        elif (
            score.token_identity_sha256 != common_identity
            or score.sequence_inventory_sha256 != common_inventory
        ):
            raise ContributionReportError(
                "all seeds and endpoints must use one scientific token inventory"
            )
        if score.endpoint not in common_sst:
            common_sst[score.endpoint] = score.common_sst
        elif not np.array_equal(common_sst[score.endpoint], score.common_sst):
            raise ContributionReportError(
                f"{score.endpoint} per-sequence SST differs across seeds"
            )
        summed_sst = float(score.common_sst.sum(dtype=np.float64))
        for condition in CONTRIBUTION_CONDITIONS:
            row = by_run[(condition, score.seed)]
            if not _same_float(
                summed_sst,
                getattr(row, f"{score.endpoint}_sst"),
            ):
                raise ContributionReportError(
                    f"{key} common SST does not match {condition} pooled SST"
                )
            summed_sse = float(score.condition_sse[condition].sum(dtype=np.float64))
            if not _same_float(
                summed_sse,
                getattr(row, f"{score.endpoint}_sse"),
            ):
                raise ContributionReportError(
                    f"{key} {condition} sequence SSE does not match pooled SSE"
                )
    return result


def _compute_effects(
    scores: Mapping[tuple[int, str], PairedSequenceScores],
    *,
    replicates: int,
) -> tuple[
    dict[tuple[int, str, str, str], PairedSequenceEffect],
    dict[tuple[int, str, str, str], np.ndarray],
]:
    seeds = sorted({seed for seed, _ in scores})
    effects: dict[tuple[int, str, str, str], PairedSequenceEffect] = {}
    samples: dict[tuple[int, str, str, str], np.ndarray] = {}
    for seed in seeds:
        seed_effects, seed_samples = _effect_samples_for_seed(
            scores[(seed, "component")],
            scores[(seed, "routed")],
            replicates=replicates,
        )
        effects.update(seed_effects)
        samples.update(seed_samples)
    return effects, samples


def _hierarchical_effects(
    effects: Mapping[tuple[int, str, str, str], PairedSequenceEffect],
    samples: Mapping[tuple[int, str, str, str], np.ndarray],
    *,
    replicates: int,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    seeds = sorted({key[0] for key in effects})
    if len(seeds) < 2:
        return {}
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    seed_draws = rng.integers(
        0,
        len(seeds),
        size=(replicates, len(seeds)),
        endpoint=False,
    )
    within_draws = rng.integers(
        0,
        replicates,
        size=(replicates, len(seeds)),
        endpoint=False,
    )
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for endpoint, left, right in _REQUIRED_EFFECTS:
        stacked = np.stack(
            [samples[(seed, endpoint, left, right)] for seed in seeds],
            axis=0,
        )
        selected = stacked[seed_draws, within_draws].mean(axis=1)
        low, high = np.quantile(selected, (0.025, 0.975), method="linear")
        result[(endpoint, left, right)] = {
            "endpoint": endpoint,
            "left": left,
            "right": right,
            "effect": statistics.fmean(
                effects[(seed, endpoint, left, right)].effect for seed in seeds
            ),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "training_seed_clusters": len(seeds),
            "replicates": replicates,
            "bootstrap_seed": BOOTSTRAP_SEED + 1,
            "bootstrap_algorithm": HIERARCHICAL_BOOTSTRAP_ALGORITHM,
        }
    return result


def _primary_summary(
    rows: Sequence[ContributionRunEvaluation],
    effects: Mapping[tuple[int, str, str, str], PairedSequenceEffect],
    hierarchical: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    by_key = {(row.condition, row.seed): row for row in rows}
    seeds = sorted({row.seed for row in rows})
    mean_ev = {
        condition: statistics.fmean(
            by_key[(condition, seed)].endpoint("component")["ev"] for seed in seeds
        )
        for condition in CONTRIBUTION_CONDITIONS
    }
    denominator = mean_ev["F"] - mean_ev["B"]
    paired_denominator = [
        effects[(seed, "component", "F", "B")].effect for seed in seeds
    ]
    if len(seeds) == 1:
        selected = effects[(seeds[0], "component", "F", "B")]
        denominator_ci = (selected.ci95_low, selected.ci95_high)
        interval_basis = "paired packed-sequence bootstrap"
    else:
        selected_hierarchical = hierarchical[("component", "F", "B")]
        denominator_ci = (
            float(selected_hierarchical["ci95_low"]),
            float(selected_hierarchical["ci95_high"]),
        )
        interval_basis = "hierarchical paired seed-plus-sequence bootstrap"

    gain: float | None = None
    gain_status = "undefined"
    gain_reason: str | None = "mean F-minus-B contribution EV is non-positive"
    if denominator > 0:
        gain = 100.0 * (mean_ev["A"] - mean_ev["B"]) / denominator
        every_seed_positive = all(
            effects[(seed, "component", "F", "B")].ci95_low > 0 for seed in seeds
        )
        if denominator_ci[0] <= 0 or not every_seed_positive:
            gain_status = "unstable"
            gain_reason = (
                "a per-seed or hierarchical paired F-minus-B interval includes zero"
            )
        else:
            gain_status = "stable"
            gain_reason = None

    cost_ratios = {
        str(seed): 100.0
        * by_key[("A", seed)].timing.marginal_chip_seconds
        / by_key[("F", seed)].timing.marginal_chip_seconds
        for seed in seeds
    }
    return {
        "mean_component_ev": mean_ev,
        "mean_f_minus_b": denominator,
        "paired_f_minus_b": paired_denominator,
        "paired_f_minus_b_ci95": list(denominator_ci),
        "paired_f_minus_b_interval_basis": interval_basis,
        "gain_recovery_percent": gain,
        "gain_status": gain_status,
        "gain_reason": gain_reason,
        "delta_route": statistics.fmean(
            effects[(seed, "component", "A", "R")].effect for seed in seeds
        ),
        "mean_a_minus_b_component": statistics.fmean(
            effects[(seed, "component", "A", "B")].effect for seed in seeds
        ),
        "mean_a_minus_b_routed": statistics.fmean(
            effects[(seed, "routed", "A", "B")].effect for seed in seeds
        ),
        "cost_ratio_percent": statistics.median(cost_ratios.values()),
        "per_seed_cost_ratio_percent": cost_ratios,
        "nominal_a_candidate_exposure_percent": NOMINAL_A_EXPOSURE_PERCENT,
    }


def _learning_curve_no_material_late_collapse(
    rows: Sequence[ContributionRunEvaluation],
) -> bool:
    """Apply the frozen absolute-EV late-collapse policy to every A seed."""

    selected = [row for row in rows if row.condition == "A"]
    return all(
        row.learning_curve[-1].endpoint["ev"]
        >= max(point.endpoint["ev"] for point in row.learning_curve)
        - MAX_MATERIAL_LATE_COLLAPSE_EV
        for row in selected
    )


def _gate_summary(
    *,
    phase: str,
    rows: Sequence[ContributionRunEvaluation],
    effects: Mapping[tuple[int, str, str, str], PairedSequenceEffect],
    primary: Mapping[str, Any],
    authoritative: bool,
) -> dict[str, Any]:
    seeds = sorted({row.seed for row in rows})
    checks: dict[str, bool] = {
        "authoritative_evidence": authoritative,
        "all_runs_healthy": all(row.health.passed for row in rows),
        "stable_f_minus_b_denominator": primary["gain_status"] == "stable",
        "a_minus_r_positive_with_ci": all(
            effects[(seed, "component", "A", "R")].ci95_low > 0 for seed in seeds
        ),
        "a_minus_b_positive_with_ci": all(
            effects[(seed, "component", "A", "B")].ci95_low > 0 for seed in seeds
        ),
        "f_minus_b_positive_with_ci": all(
            effects[(seed, "component", "F", "B")].ci95_low > 0 for seed in seeds
        ),
        "a_minus_b_routed_same_positive_sign_every_seed": all(
            effects[(seed, "routed", "A", "B")].effect > 0 for seed in seeds
        ),
        "gain_recovery_at_least_10_percent": (
            primary["gain_recovery_percent"] is not None
            and float(primary["gain_recovery_percent"]) >= 10.0
        ),
    }
    if phase != "phase2":
        checks.update(
            {
                "cost_ratio_below_50_percent": (
                    float(primary["cost_ratio_percent"]) < 50.0
                ),
                "learning_curve_no_material_late_collapse": (
                    _learning_curve_no_material_late_collapse(rows)
                ),
            }
        )
    if phase == "phase4":
        checks["three_confirmatory_seeds_complete"] = seeds == [1, 2, 3]
    return {
        "phase": phase,
        "passed": all(checks.values()),
        "checks": checks,
    }


def _conditions_report(
    rows: Sequence[ContributionRunEvaluation],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in CONTRIBUTION_CONDITIONS:
        selected = [row for row in rows if row.condition == condition]
        result[condition] = {
            "layout": selected[0].layout,
            "mean_endpoints": {
                endpoint: {
                    "ev": statistics.fmean(
                        row.endpoint(endpoint)["ev"] for row in selected
                    ),
                    "nmse": statistics.fmean(
                        row.endpoint(endpoint)["nmse"] for row in selected
                    ),
                }
                for endpoint in _ENDPOINTS
            },
            "mean_dead_rate": statistics.fmean(row.dead_rate for row in selected),
            "mean_fired_feature_fraction": statistics.fmean(
                row.fired_feature_fraction for row in selected
            ),
            "mean_macro_expert_ev": statistics.fmean(
                row.macro_expert_ev for row in selected
            ),
            "median_marginal_chip_seconds": statistics.median(
                row.timing.marginal_chip_seconds for row in selected
            ),
            "seeds": [row.report_row() for row in selected],
        }
    return result


def _normalize_harvester(
    value: Mapping[str, Any],
    *,
    phase: str,
    training_capture_manifest_sha256: str,
    evaluation_capture_manifest_sha256: str,
) -> dict[str, Any]:
    required = {
        "training_capture_manifest_sha256",
        "evaluation_capture_manifest_sha256",
        "timing_receipt_sha256",
        "training_tokens",
        "evaluation_tokens",
        "capture_tokens",
        "chip_count",
        "useful_prefix_mfu",
        "device_output_tokens_per_second",
        "persisted_tokens_per_second",
        "input_wait_fraction",
        "max_writer_backlog_batches",
        "wall_seconds",
        "chip_seconds",
        "compilation_chip_seconds",
        "serialization_chip_seconds",
    }
    selected = dict(_exact_mapping("harvester evidence", value, required))
    if (
        selected["training_capture_manifest_sha256"] != training_capture_manifest_sha256
        or selected["evaluation_capture_manifest_sha256"]
        != evaluation_capture_manifest_sha256
    ):
        raise ContributionReportError(
            "harvester evidence is not bound to both report captures"
        )
    _sha256(
        "harvester.training_capture_manifest_sha256",
        selected["training_capture_manifest_sha256"],
    )
    _sha256(
        "harvester.evaluation_capture_manifest_sha256",
        selected["evaluation_capture_manifest_sha256"],
    )
    _sha256(
        "harvester.timing_receipt_sha256",
        selected["timing_receipt_sha256"],
    )
    plan = _PHASE_PLAN[phase]
    if selected["training_tokens"] != plan["training_tokens"]:
        raise ContributionReportError(
            f"{phase} harvester training_tokens must be {plan['training_tokens']}"
        )
    if selected["evaluation_tokens"] != plan["evaluation_tokens"]:
        raise ContributionReportError(
            f"{phase} harvester evaluation_tokens must be {plan['evaluation_tokens']}"
        )
    if (
        selected["capture_tokens"]
        != selected["training_tokens"] + selected["evaluation_tokens"]
    ):
        raise ContributionReportError(
            "harvester capture_tokens must equal training plus evaluation tokens"
        )
    if selected["capture_tokens"] != plan["capture_tokens"]:
        raise ContributionReportError(
            f"{phase} harvester capture_tokens must be "
            f"{_PHASE_PLAN[phase]['capture_tokens']}"
        )
    chips = _positive_int("harvester.chip_count", selected["chip_count"])
    wall = _finite("harvester.wall_seconds", selected["wall_seconds"])
    chip_seconds = _finite("harvester.chip_seconds", selected["chip_seconds"])
    if wall <= 0 or chip_seconds <= 0:
        raise ContributionReportError(
            "harvester wall_seconds and chip_seconds must be positive"
        )
    if not math.isclose(
        chip_seconds,
        chips * wall,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ContributionReportError(
            "harvester chip_seconds must equal chip_count * wall_seconds"
        )
    mfu = _finite("harvester.useful_prefix_mfu", selected["useful_prefix_mfu"])
    if not 0.0 <= mfu <= 1.0:
        raise ContributionReportError("harvester MFU must lie in [0,1]")
    for name in (
        "device_output_tokens_per_second",
        "persisted_tokens_per_second",
    ):
        if _finite(f"harvester.{name}", selected[name]) <= 0:
            raise ContributionReportError(f"harvester.{name} must be positive")
    persisted_goodput = float(selected["persisted_tokens_per_second"])
    observed_goodput = float(selected["capture_tokens"]) / wall
    if not math.isclose(
        persisted_goodput,
        observed_goodput,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ContributionReportError(
            "harvester persisted_tokens_per_second must equal "
            "capture_tokens / wall_seconds"
        )
    if float(selected["device_output_tokens_per_second"]) < persisted_goodput:
        raise ContributionReportError(
            "harvester device-output goodput cannot trail persisted goodput"
        )
    input_wait = _finite(
        "harvester.input_wait_fraction",
        selected["input_wait_fraction"],
    )
    if not 0.0 <= input_wait <= 1.0:
        raise ContributionReportError("harvester.input_wait_fraction must lie in [0,1]")
    _nonnegative_int(
        "harvester.max_writer_backlog_batches",
        selected["max_writer_backlog_batches"],
    )
    for name in ("compilation_chip_seconds", "serialization_chip_seconds"):
        _finite(f"harvester.{name}", selected[name], minimum=0.0)
    return _jsonable(selected)


def _system_cost_summary(
    rows: Sequence[ContributionRunEvaluation],
    q_rows: Sequence[SharedQEvaluation],
    *,
    harvester: Mapping[str, Any],
) -> dict[str, Any]:
    by_key = {(row.condition, row.seed): row for row in rows}
    q_by_seed = {row.seed: row for row in q_rows}
    seeds = sorted(q_by_seed)
    harvest = float(harvester["chip_seconds"])
    one_seed: dict[str, dict[str, float]] = {}
    for seed in seeds:
        q_cost = q_by_seed[seed].timing.marginal_chip_seconds
        a_total = harvest + q_cost + by_key[("A", seed)].timing.marginal_chip_seconds
        f_total = harvest + q_cost + by_key[("F", seed)].timing.marginal_chip_seconds
        one_seed[str(seed)] = {
            "a_total_chip_seconds": a_total,
            "f_total_chip_seconds": f_total,
            "ratio_percent": 100.0 * a_total / f_total,
        }
    all_a = (
        harvest
        + sum(row.timing.marginal_chip_seconds for row in q_rows)
        + sum(by_key[("A", seed)].timing.marginal_chip_seconds for seed in seeds)
    )
    all_f = (
        harvest
        + sum(row.timing.marginal_chip_seconds for row in q_rows)
        + sum(by_key[("F", seed)].timing.marginal_chip_seconds for seed in seeds)
    )
    return {
        "shared_harvest_chip_seconds": harvest,
        "common_shared_q_chip_seconds": {
            str(row.seed): row.timing.marginal_chip_seconds for row in q_rows
        },
        "one_seed": one_seed,
        "median_one_seed_ratio_percent": statistics.median(
            item["ratio_percent"] for item in one_seed.values()
        ),
        "all_available_seeds": {
            "seeds": seeds,
            "a_total_chip_seconds": all_a,
            "f_total_chip_seconds": all_f,
            "ratio_percent": 100.0 * all_a / all_f,
        },
        "excluded_costs": {
            "harvester_compilation_chip_seconds": harvester["compilation_chip_seconds"],
            "harvester_serialization_chip_seconds": harvester[
                "serialization_chip_seconds"
            ],
            "condition_compilation_chip_seconds": sum(
                row.timing.compilation_chip_seconds for row in rows
            ),
            "condition_checkpoint_serialization_chip_seconds": sum(
                row.timing.checkpoint_serialization_chip_seconds for row in rows
            ),
            "condition_evaluation_chip_seconds": sum(
                row.timing.evaluation_chip_seconds for row in rows
            ),
            "shared_q_compilation_chip_seconds": sum(
                row.timing.compilation_chip_seconds for row in q_rows
            ),
            "shared_q_checkpoint_serialization_chip_seconds": sum(
                row.timing.checkpoint_serialization_chip_seconds for row in q_rows
            ),
            "shared_q_evaluation_chip_seconds": sum(
                row.timing.evaluation_chip_seconds for row in q_rows
            ),
        },
        "basis": (
            "shared component harvest once plus one common per-seed Q plus "
            "condition-specific marginal SAE training; compilation, checkpoint "
            "serialization, and evaluation are excluded and itemized"
        ),
    }


def _authorize_stage(
    *,
    phase: str,
    protocol_sha256: str,
    training_capture_manifest_sha256: str,
    evaluation_capture_manifest_sha256: str,
    parent_training_capture_manifest_sha256: str,
    parent_evaluation_capture_manifest_sha256: str,
    training_capture_lineage_receipt_sha256: str,
    evaluation_capture_lineage_receipt_sha256: str,
    recomposition_sidecar_sha256: str,
    phase1_receipt: Mapping[str, Any] | None,
    parent_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    for name, digest in (
        (
            "parent_training_capture_manifest_sha256",
            parent_training_capture_manifest_sha256,
        ),
        (
            "parent_evaluation_capture_manifest_sha256",
            parent_evaluation_capture_manifest_sha256,
        ),
        (
            "training_capture_lineage_receipt_sha256",
            training_capture_lineage_receipt_sha256,
        ),
        (
            "evaluation_capture_lineage_receipt_sha256",
            evaluation_capture_lineage_receipt_sha256,
        ),
    ):
        _sha256(name, digest)
    if (
        parent_training_capture_manifest_sha256 == training_capture_manifest_sha256
        or parent_evaluation_capture_manifest_sha256
        == evaluation_capture_manifest_sha256
    ):
        raise ContributionReportError(
            "each stage split must extend a distinct parent manifest"
        )
    expected_parent_phase = _PHASE_PLAN[phase]["parent_phase"]
    if expected_parent_phase is None:
        if parent_report is not None:
            raise ContributionReportError("Phase 2 cannot accept a parent report")
        if phase1_receipt is None:
            raise ContributionReportError("Phase 2 requires a passing Phase-1 receipt")
        phase1 = validate_phase1_receipt(phase1_receipt)
        if phase1["status"] != "passed":
            raise ContributionReportError(
                "a failed Phase-1 receipt cannot authorize Phase 2"
            )
        if phase1["protocol_sha256"] != protocol_sha256:
            raise ContributionReportError(
                "Phase-1 protocol differs from the Phase-2 protocol"
            )
        if phase1["recomposition_sidecar_sha256"] != recomposition_sidecar_sha256:
            raise ContributionReportError(
                "Phase-1 recomposition sidecar differs from Phase 2"
            )
        if (
            phase1["training_capture_manifest_sha256"]
            != parent_training_capture_manifest_sha256
            or phase1["evaluation_capture_manifest_sha256"]
            != parent_evaluation_capture_manifest_sha256
        ):
            raise ContributionReportError(
                "Phase-2 split parents are not the Phase-1 captures"
            )
        return {
            "phase1_receipt_sha256": phase1["receipt_sha256"],
            "parent_report_sha256": None,
            "parent_phase": "phase1",
            "parent_training_capture_manifest_sha256": (
                parent_training_capture_manifest_sha256
            ),
            "parent_evaluation_capture_manifest_sha256": (
                parent_evaluation_capture_manifest_sha256
            ),
            "training_capture_lineage_receipt_sha256": (
                training_capture_lineage_receipt_sha256
            ),
            "evaluation_capture_lineage_receipt_sha256": (
                evaluation_capture_lineage_receipt_sha256
            ),
        }

    if phase1_receipt is not None:
        raise ContributionReportError(
            f"{phase} derives Phase-1 identity through its parent report"
        )
    if parent_report is None:
        raise ContributionReportError(
            f"{phase} requires a passed {expected_parent_phase} report"
        )
    parent = validate_contribution_report(parent_report)
    if parent["phase"] != expected_parent_phase:
        raise ContributionReportError(f"{phase} parent must be {expected_parent_phase}")
    if parent["status"] != "passed" or not parent["authoritative"]:
        raise ContributionReportError(
            f"{phase} parent report is not an authoritative passed gate"
        )
    if parent["protocol_sha256"] != protocol_sha256:
        raise ContributionReportError("parent report protocol differs")
    if parent["recomposition_sidecar_sha256"] != recomposition_sidecar_sha256:
        raise ContributionReportError("parent report recomposition sidecar differs")
    if (
        parent["training_capture_manifest_sha256"]
        != parent_training_capture_manifest_sha256
        or parent["evaluation_capture_manifest_sha256"]
        != parent_evaluation_capture_manifest_sha256
    ):
        raise ContributionReportError(
            f"{phase} split captures do not extend the parent report captures"
        )
    return {
        "phase1_receipt_sha256": parent["stage_authorization"]["phase1_receipt_sha256"],
        "parent_report_sha256": parent["report_sha256"],
        "parent_phase": expected_parent_phase,
        "parent_training_capture_manifest_sha256": (
            parent_training_capture_manifest_sha256
        ),
        "parent_evaluation_capture_manifest_sha256": (
            parent_evaluation_capture_manifest_sha256
        ),
        "training_capture_lineage_receipt_sha256": (
            training_capture_lineage_receipt_sha256
        ),
        "evaluation_capture_lineage_receipt_sha256": (
            evaluation_capture_lineage_receipt_sha256
        ),
    }


def render_contribution_headline(report: Mapping[str, Any]) -> str:
    """Render a claim whose strength follows the report's actual eligibility."""

    phase = report["phase"]
    status = report["status"]
    authoritative = report["authoritative"]
    primary = report["primary"]
    gain = primary["gain_recovery_percent"]
    if not authoritative:
        return (
            f"Non-authoritative {phase} diagnostic; no scientific headline is "
            "permitted."
        )
    if phase in {"phase2", "phase3"}:
        if gain is None:
            result = "G2 is undefined because F did not beat B"
        else:
            result = (
                f"exploratory G2={float(gain):.1f}% and "
                f"C2={float(primary['cost_ratio_percent']):.1f}%"
            )
        return (
            f"Seed-0 {phase} gate {status}: {result}. Seed 0 is permanently "
            "excluded from the confirmatory headline."
        )
    if status != "headline-ready":
        if gain is None:
            detail = "G2 is undefined because the F-over-B gain is non-positive"
        elif primary["gain_status"] != "stable":
            detail = (
                f"G2={float(gain):.1f}% is exploratory because the "
                "F-over-B denominator is unstable"
            )
        else:
            failed = [
                name for name, passed in report["gate"]["checks"].items() if not passed
            ]
            detail = "failed checks: " + ", ".join(failed)
        return f"Confirmatory report is not headline-ready: {detail}."
    return (
        f"Expert-aligned A recovers {float(gain):.1f}% of the held-out "
        "contribution reconstruction gain of full-width route-rank-conditioned "
        f"F over B at {float(primary['cost_ratio_percent']):.1f}% of F's "
        f"measured marginal SAE training cost "
        f"({NOMINAL_A_EXPOSURE_PERCENT:.2f}% nominal candidate exposure)."
    )


def build_contribution_report(
    evaluations: Iterable[ContributionRunEvaluation | Mapping[str, Any]],
    score_manifest_references: Iterable[ScoreArtifactRef | Mapping[str, Any]],
    shared_q: Iterable[SharedQEvaluation | Mapping[str, Any]],
    *,
    score_artifact_resolver: ScoreArtifactResolver,
    phase: Literal["phase2", "phase3", "phase4"],
    protocol_sha256: str,
    training_capture_manifest_sha256: str,
    evaluation_capture_manifest_sha256: str,
    parent_training_capture_manifest_sha256: str,
    parent_evaluation_capture_manifest_sha256: str,
    training_capture_lineage_receipt_sha256: str,
    evaluation_capture_lineage_receipt_sha256: str,
    recomposition_sidecar_sha256: str,
    harvester: Mapping[str, Any],
    phase1_receipt: Mapping[str, Any] | None = None,
    parent_report: Mapping[str, Any] | None = None,
    authoritative: bool = True,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build one immutable, evidence-bound stage or confirmatory report."""

    if phase not in _PHASES:
        raise ContributionReportError(f"phase must be one of {_PHASES}")
    for name, digest in (
        ("protocol_sha256", protocol_sha256),
        (
            "training_capture_manifest_sha256",
            training_capture_manifest_sha256,
        ),
        (
            "evaluation_capture_manifest_sha256",
            evaluation_capture_manifest_sha256,
        ),
        ("recomposition_sidecar_sha256", recomposition_sidecar_sha256),
    ):
        _sha256(name, digest)
    selected_authoritative = _strict_bool("authoritative", authoritative)
    selected_replicates = _positive_int(
        "bootstrap_replicates",
        bootstrap_replicates,
    )
    if selected_authoritative and selected_replicates != BOOTSTRAP_REPLICATES:
        raise ContributionReportError(
            f"authoritative reports require {BOOTSTRAP_REPLICATES} replicates"
        )

    stage_authorization = _authorize_stage(
        phase=phase,
        protocol_sha256=protocol_sha256,
        training_capture_manifest_sha256=training_capture_manifest_sha256,
        evaluation_capture_manifest_sha256=evaluation_capture_manifest_sha256,
        parent_training_capture_manifest_sha256=(
            parent_training_capture_manifest_sha256
        ),
        parent_evaluation_capture_manifest_sha256=(
            parent_evaluation_capture_manifest_sha256
        ),
        training_capture_lineage_receipt_sha256=(
            training_capture_lineage_receipt_sha256
        ),
        evaluation_capture_lineage_receipt_sha256=(
            evaluation_capture_lineage_receipt_sha256
        ),
        recomposition_sidecar_sha256=recomposition_sidecar_sha256,
        phase1_receipt=phase1_receipt,
        parent_report=parent_report,
    )
    rows = _normalize_runs(
        evaluations,
        phase=phase,
        training_capture_manifest_sha256=training_capture_manifest_sha256,
        evaluation_capture_manifest_sha256=evaluation_capture_manifest_sha256,
    )
    q_rows = _normalize_shared_q(
        shared_q,
        rows=rows,
        phase=phase,
        training_capture_manifest_sha256=training_capture_manifest_sha256,
        evaluation_capture_manifest_sha256=evaluation_capture_manifest_sha256,
    )
    retained_scores, score_manifest_rows = _scores_from_retained_artifacts(
        score_manifest_references,
        score_artifact_resolver,
        rows=rows,
        phase=phase,
        protocol_sha256=protocol_sha256,
        training_capture_manifest_sha256=training_capture_manifest_sha256,
        evaluation_capture_manifest_sha256=evaluation_capture_manifest_sha256,
        recomposition_sidecar_sha256=recomposition_sidecar_sha256,
    )
    scores = _normalize_scores(
        retained_scores,
        rows=rows,
        phase=phase,
        evaluation_capture_manifest_sha256=evaluation_capture_manifest_sha256,
    )
    effects, bootstrap_samples = _compute_effects(
        scores,
        replicates=selected_replicates,
    )
    hierarchical = _hierarchical_effects(
        effects,
        bootstrap_samples,
        replicates=selected_replicates,
    )
    primary = _primary_summary(rows, effects, hierarchical)
    gate = _gate_summary(
        phase=phase,
        rows=rows,
        effects=effects,
        primary=primary,
        authoritative=selected_authoritative,
    )
    if not selected_authoritative:
        status = "non-authoritative"
    elif phase == "phase4":
        status = "headline-ready" if gate["passed"] else "confirmatory-failed"
    else:
        status = "passed" if gate["passed"] else "failed"
    harvester_evidence = _normalize_harvester(
        harvester,
        phase=phase,
        training_capture_manifest_sha256=training_capture_manifest_sha256,
        evaluation_capture_manifest_sha256=evaluation_capture_manifest_sha256,
    )

    unsigned: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "phase": phase,
        "created_at": created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "authoritative": selected_authoritative,
        "status": status,
        "protocol_sha256": protocol_sha256,
        "training_capture_manifest_sha256": training_capture_manifest_sha256,
        "evaluation_capture_manifest_sha256": evaluation_capture_manifest_sha256,
        "recomposition_sidecar_sha256": recomposition_sidecar_sha256,
        "stage_authorization": stage_authorization,
        "primary": primary,
        "gate": gate,
        "conditions": _conditions_report(rows),
        "score_manifest_references": score_manifest_rows,
        "score_evidence": [scores[key].identity_row() for key in sorted(scores)],
        "paired_sequence_effects": [
            dataclasses.asdict(effects[key]) for key in sorted(effects)
        ],
        "hierarchical_seed_sequence_effects": [
            hierarchical[key] for key in sorted(hierarchical)
        ],
        "shared_q": [row.report_row() for row in q_rows],
        "harvester": harvester_evidence,
        "secondary_end_to_end_cost": _system_cost_summary(
            rows,
            q_rows,
            harvester=harvester_evidence,
        ),
    }
    unsigned["headline"] = render_contribution_headline(unsigned)
    _ensure_json_safe(unsigned)
    return {**unsigned, "report_sha256": canonical_sha256(unsigned)}


def _expected_status(
    *,
    phase: str,
    authoritative: bool,
    gate_passed: bool,
) -> str:
    if not authoritative:
        return "non-authoritative"
    if phase == "phase4":
        return "headline-ready" if gate_passed else "confirmatory-failed"
    return "passed" if gate_passed else "failed"


def _require_report_close(name: str, actual: Any, expected: Any) -> None:
    if not _same_float(_finite(name, actual), _finite(name, expected)):
        raise ContributionReportError(
            f"{name} is inconsistent with the report's source evidence"
        )


def _curve_from_report_rows(value: Any) -> tuple[LearningCurvePoint, ...]:
    if not isinstance(value, list):
        raise ContributionReportError("learning curve must be a list")
    result = []
    for raw in value:
        selected = _exact_mapping(
            "learning-curve report point",
            raw,
            {
                "training_tokens",
                "sse",
                "sst",
                "checkpoint_sha256",
                "evaluation_capture_manifest_sha256",
                "nmse",
                "ev",
            },
        )
        derived = _endpoint(selected["sse"], selected["sst"])
        _require_report_close(
            "learning-curve nmse",
            selected["nmse"],
            derived["nmse"],
        )
        _require_report_close(
            "learning-curve ev",
            selected["ev"],
            derived["ev"],
        )
        result.append(
            LearningCurvePoint(
                training_tokens=selected["training_tokens"],
                sse=derived["sse"],
                sst=derived["sst"],
                checkpoint_sha256=selected["checkpoint_sha256"],
                evaluation_capture_manifest_sha256=(
                    selected["evaluation_capture_manifest_sha256"]
                ),
            )
        )
    return tuple(result)


def _run_from_report_row(value: Mapping[str, Any]) -> ContributionRunEvaluation:
    required = {
        "condition",
        "seed",
        "scale",
        "evaluation_tokens",
        "training_tokens",
        "optimizer_updates",
        "warmup_updates",
        "masked_tail_tokens",
        "checkpoint_tokens",
        "run_manifest_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "shared_q_checkpoint_sha256",
        "input_receipts",
        "endpoints",
        "l0_per_contribution",
        "total_l0",
        "dead_rate",
        "fired_feature_fraction",
        "macro_expert_ev",
        "learning_curve",
        "timing",
        "tokens_per_second",
        "health",
        "layout",
    }
    selected = _exact_mapping("condition seed row", value, required)
    endpoints = _exact_mapping(
        "condition endpoints",
        selected["endpoints"],
        set(_ENDPOINTS),
    )
    flattened: dict[str, float] = {}
    for endpoint in _ENDPOINTS:
        endpoint_row = _exact_mapping(
            f"{endpoint} endpoint",
            endpoints[endpoint],
            {"sse", "sst", "nmse", "ev"},
        )
        derived = _endpoint(endpoint_row["sse"], endpoint_row["sst"])
        for metric in ("sse", "sst", "nmse", "ev"):
            _require_report_close(
                f"{endpoint}.{metric}",
                endpoint_row[metric],
                derived[metric],
            )
        flattened[f"{endpoint}_sse"] = derived["sse"]
        flattened[f"{endpoint}_sst"] = derived["sst"]
    health_row = dict(
        _exact_mapping(
            "condition health",
            selected["health"],
            {
                *(field.name for field in dataclasses.fields(RunHealthEvidence)),
                "passed",
            },
        )
    )
    claimed_health_pass = _strict_bool(
        "condition health.passed",
        health_row.pop("passed"),
    )
    health = RunHealthEvidence.from_value(health_row)
    if claimed_health_pass != health.passed:
        raise ContributionReportError("condition health summary is inconsistent")
    run = ContributionRunEvaluation(
        condition=selected["condition"],
        seed=selected["seed"],
        scale=selected["scale"],
        evaluation_tokens=selected["evaluation_tokens"],
        training_tokens=selected["training_tokens"],
        optimizer_updates=selected["optimizer_updates"],
        warmup_updates=selected["warmup_updates"],
        masked_tail_tokens=selected["masked_tail_tokens"],
        checkpoint_tokens=tuple(selected["checkpoint_tokens"]),
        run_manifest_sha256=selected["run_manifest_sha256"],
        config_sha256=selected["config_sha256"],
        checkpoint_sha256=selected["checkpoint_sha256"],
        shared_q_checkpoint_sha256=selected["shared_q_checkpoint_sha256"],
        input_receipts=tuple(selected["input_receipts"]),
        **flattened,
        l0_per_contribution=selected["l0_per_contribution"],
        total_l0=selected["total_l0"],
        dead_rate=selected["dead_rate"],
        fired_feature_fraction=selected["fired_feature_fraction"],
        macro_expert_ev=selected["macro_expert_ev"],
        learning_curve=_curve_from_report_rows(selected["learning_curve"]),
        timing=CostLedger.from_value(selected["timing"]),
        health=health,
    )
    _require_report_close(
        "condition tokens_per_second",
        selected["tokens_per_second"],
        run.tokens_per_second,
    )
    if selected["layout"] != run.layout:
        raise ContributionReportError(
            "condition layout differs from its canonical input receipt"
        )
    return run


def _q_from_report_row(value: Mapping[str, Any]) -> SharedQEvaluation:
    required = {
        "seed",
        "evaluation_tokens",
        "training_tokens",
        "checkpoint_tokens",
        "checkpoint_sha256",
        "config_sha256",
        "training_manifest_sha256",
        "evaluation_capture_manifest_sha256",
        "width",
        "l0",
        "sse",
        "sst",
        "nmse",
        "ev",
        "dead_rate",
        "firing_entropy",
        "firing_gini",
        "decoder_coherence",
        "effective_rank",
        "learning_curve",
        "timing",
        "tokens_per_second",
    }
    selected = _exact_mapping("shared-Q report row", value, required)
    derived = _endpoint(selected["sse"], selected["sst"])
    for metric in ("sse", "sst", "nmse", "ev"):
        _require_report_close(
            f"shared-Q {metric}",
            selected[metric],
            derived[metric],
        )
    q_row = SharedQEvaluation(
        seed=selected["seed"],
        evaluation_tokens=selected["evaluation_tokens"],
        training_tokens=selected["training_tokens"],
        checkpoint_tokens=tuple(selected["checkpoint_tokens"]),
        checkpoint_sha256=selected["checkpoint_sha256"],
        config_sha256=selected["config_sha256"],
        training_manifest_sha256=selected["training_manifest_sha256"],
        evaluation_capture_manifest_sha256=(
            selected["evaluation_capture_manifest_sha256"]
        ),
        width=selected["width"],
        l0=selected["l0"],
        sse=derived["sse"],
        sst=derived["sst"],
        dead_rate=selected["dead_rate"],
        firing_entropy=selected["firing_entropy"],
        firing_gini=selected["firing_gini"],
        decoder_coherence=selected["decoder_coherence"],
        effective_rank=selected["effective_rank"],
        learning_curve=_curve_from_report_rows(selected["learning_curve"]),
        timing=CostLedger.from_value(selected["timing"]),
    )
    _require_report_close(
        "shared-Q tokens_per_second",
        selected["tokens_per_second"],
        q_row.training_tokens / q_row.timing.timed_wall_seconds,
    )
    return q_row


def _semantic_report_evidence(
    result: Mapping[str, Any],
    *,
    score_artifact_resolver: ScoreArtifactResolver | None = None,
) -> tuple[
    list[ContributionRunEvaluation],
    list[SharedQEvaluation],
    dict[tuple[int, str, str, str], PairedSequenceEffect],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, Any],
]:
    """Reconstruct every derived field from the persisted evidence rows."""

    phase = result["phase"]
    plan = _PHASE_PLAN[phase]
    training_capture = _sha256(
        "report.training_capture_manifest_sha256",
        result["training_capture_manifest_sha256"],
    )
    evaluation_capture = _sha256(
        "report.evaluation_capture_manifest_sha256",
        result["evaluation_capture_manifest_sha256"],
    )
    conditions = _exact_mapping(
        "report conditions",
        result["conditions"],
        set(CONTRIBUTION_CONDITIONS),
    )
    raw_rows: list[ContributionRunEvaluation] = []
    for condition in CONTRIBUTION_CONDITIONS:
        condition_report = _exact_mapping(
            f"condition {condition}",
            conditions[condition],
            {
                "layout",
                "mean_endpoints",
                "mean_dead_rate",
                "mean_fired_feature_fraction",
                "mean_macro_expert_ev",
                "median_marginal_chip_seconds",
                "seeds",
            },
        )
        if not isinstance(condition_report["seeds"], list):
            raise ContributionReportError("condition seeds must be a list")
        raw_rows.extend(
            _run_from_report_row(item) for item in condition_report["seeds"]
        )
    rows = _normalize_runs(
        raw_rows,
        phase=phase,
        training_capture_manifest_sha256=training_capture,
        evaluation_capture_manifest_sha256=evaluation_capture,
    )
    if _conditions_report(rows) != conditions:
        raise ContributionReportError(
            "condition summaries are inconsistent with their seed rows"
        )

    if not isinstance(result["shared_q"], list):
        raise ContributionReportError("shared_q must be a list")
    q_rows = _normalize_shared_q(
        [_q_from_report_row(item) for item in result["shared_q"]],
        rows=rows,
        phase=phase,
        training_capture_manifest_sha256=training_capture,
        evaluation_capture_manifest_sha256=evaluation_capture,
    )
    if [item.report_row() for item in q_rows] != result["shared_q"]:
        raise ContributionReportError(
            "shared-Q summaries are inconsistent with their evidence"
        )

    raw_score_references = result["score_manifest_references"]
    if not isinstance(raw_score_references, list):
        raise ContributionReportError("score_manifest_references must be a list")
    try:
        score_references = [
            ScoreArtifactRef.from_value(item) for item in raw_score_references
        ]
    except ContributionScoreArtifactError as error:
        raise ContributionReportError(
            f"score manifest reference is invalid: {error}"
        ) from error
    expected_reference_count = len(plan["seeds"]) * len(CONTRIBUTION_CONDITIONS)
    if (
        len(score_references) != expected_reference_count
        or len({item.sha256 for item in score_references}) != expected_reference_count
    ):
        raise ContributionReportError(
            "score manifest reference set is incomplete or duplicated"
        )
    if (
        not isinstance(result["paired_sequence_effects"], list)
        or not result["paired_sequence_effects"]
    ):
        raise ContributionReportError("paired_sequence_effects must be a nonempty list")

    expected_score_keys = {
        (seed, endpoint) for seed in plan["seeds"] for endpoint in _PAIRED_ENDPOINTS
    }
    score_rows: dict[tuple[int, str], Mapping[str, Any]] = {}
    common_token_identity: str | None = None
    common_inventory: str | None = None
    if not isinstance(result["score_evidence"], list):
        raise ContributionReportError("score_evidence must be a list")
    for raw in result["score_evidence"]:
        score = _exact_mapping(
            "score evidence row",
            raw,
            {
                "schema_version",
                "seed",
                "endpoint",
                "capture_manifest_sha256",
                "token_identity_sha256",
                "sequence_inventory_sha256",
                "score_artifact_sha256",
                "tokens",
                "clusters",
            },
        )
        if score["schema_version"] != SCORE_SCHEMA_VERSION:
            raise ContributionReportError("score evidence schema is incompatible")
        seed = _nonnegative_int("score seed", score["seed"])
        endpoint = score["endpoint"]
        key = (seed, endpoint)
        if key in score_rows:
            raise ContributionReportError("duplicate score evidence row")
        if score["capture_manifest_sha256"] != evaluation_capture:
            raise ContributionReportError(
                "score evidence is not bound to the evaluation capture"
            )
        for name in (
            "capture_manifest_sha256",
            "token_identity_sha256",
            "sequence_inventory_sha256",
            "score_artifact_sha256",
        ):
            _sha256(f"score {name}", score[name])
        if score["tokens"] != plan["evaluation_tokens"]:
            raise ContributionReportError("score evidence token count is not exact")
        if _positive_int("score clusters", score["clusters"]) < 2:
            raise ContributionReportError(
                "score evidence requires at least two sequence clusters"
            )
        if common_token_identity is None:
            common_token_identity = score["token_identity_sha256"]
            common_inventory = score["sequence_inventory_sha256"]
        elif (
            score["token_identity_sha256"] != common_token_identity
            or score["sequence_inventory_sha256"] != common_inventory
        ):
            raise ContributionReportError(
                "persisted score rows use different held-out inventories"
            )
        score_rows[key] = score
    if set(score_rows) != expected_score_keys:
        raise ContributionReportError(
            "persisted score evidence is incomplete or has unexpected keys"
        )

    resolved_effects: (
        dict[
            tuple[int, str, str, str],
            PairedSequenceEffect,
        ]
        | None
    ) = None
    resolved_hierarchical: (
        dict[
            tuple[str, str, str],
            dict[str, Any],
        ]
        | None
    ) = None
    if score_artifact_resolver is not None:
        retained, canonical_references = _scores_from_retained_artifacts(
            score_references,
            score_artifact_resolver,
            rows=rows,
            phase=phase,
            protocol_sha256=result["protocol_sha256"],
            training_capture_manifest_sha256=training_capture,
            evaluation_capture_manifest_sha256=evaluation_capture,
            recomposition_sidecar_sha256=result["recomposition_sidecar_sha256"],
        )
        if canonical_references != raw_score_references:
            raise ContributionReportError(
                "score manifest references are not in canonical matrix order"
            )
        resolved_scores = _normalize_scores(
            retained,
            rows=rows,
            phase=phase,
            evaluation_capture_manifest_sha256=evaluation_capture,
        )
        expected_score_rows = [
            resolved_scores[key].identity_row() for key in sorted(resolved_scores)
        ]
        if expected_score_rows != result["score_evidence"]:
            raise ContributionReportError(
                "score evidence differs from retained sequence-score artifacts"
            )
        replicate_count = (
            BOOTSTRAP_REPLICATES
            if result["authoritative"]
            else result["paired_sequence_effects"][0]["replicates"]
        )
        resolved_effects, resolved_samples = _compute_effects(
            resolved_scores,
            replicates=replicate_count,
        )
        resolved_hierarchical = _hierarchical_effects(
            resolved_effects,
            resolved_samples,
            replicates=replicate_count,
        )

    expected_effect_keys = {
        (seed, endpoint, left, right)
        for seed in plan["seeds"]
        for endpoint, left, right in _REQUIRED_EFFECTS
    }
    effects: dict[tuple[int, str, str, str], PairedSequenceEffect] = {}
    by_run = {(row.condition, row.seed): row for row in rows}
    for raw in result["paired_sequence_effects"]:
        required = {field.name for field in dataclasses.fields(PairedSequenceEffect)}
        selected = _exact_mapping("paired effect", raw, required)
        effect = PairedSequenceEffect(**dict(selected))
        key = (effect.seed, effect.endpoint, effect.left, effect.right)
        if key in effects:
            raise ContributionReportError("duplicate paired effect")
        score = score_rows[(effect.seed, effect.endpoint)]
        for field in (
            "capture_manifest_sha256",
            "token_identity_sha256",
            "sequence_inventory_sha256",
            "score_artifact_sha256",
            "tokens",
            "clusters",
        ):
            if getattr(effect, field) != score[field]:
                raise ContributionReportError(
                    f"paired effect {key} differs from its score artifact"
                )
        left_ev = by_run[(effect.left, effect.seed)].endpoint(effect.endpoint)["ev"]
        right_ev = by_run[(effect.right, effect.seed)].endpoint(effect.endpoint)["ev"]
        _require_report_close(
            f"paired effect {key}",
            effect.effect,
            left_ev - right_ev,
        )
        if result["authoritative"] and effect.replicates != BOOTSTRAP_REPLICATES:
            raise ContributionReportError(
                "authoritative paired effect has the wrong replicate count"
            )
        effects[key] = effect
    if set(effects) != expected_effect_keys:
        raise ContributionReportError(
            "paired effect set is incomplete or has unexpected keys"
        )
    if (
        resolved_effects is not None
        and [
            dataclasses.asdict(resolved_effects[key])
            for key in sorted(resolved_effects)
        ]
        != result["paired_sequence_effects"]
    ):
        raise ContributionReportError(
            "paired intervals differ from retained sequence-score artifacts"
        )

    hierarchical: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not isinstance(result["hierarchical_seed_sequence_effects"], list):
        raise ContributionReportError(
            "hierarchical_seed_sequence_effects must be a list"
        )
    for raw in result["hierarchical_seed_sequence_effects"]:
        selected = dict(
            _exact_mapping(
                "hierarchical effect",
                raw,
                {
                    "endpoint",
                    "left",
                    "right",
                    "effect",
                    "ci95_low",
                    "ci95_high",
                    "training_seed_clusters",
                    "replicates",
                    "bootstrap_seed",
                    "bootstrap_algorithm",
                },
            )
        )
        key = (selected["endpoint"], selected["left"], selected["right"])
        if key in hierarchical or key not in set(_REQUIRED_EFFECTS):
            raise ContributionReportError("hierarchical effect key is invalid")
        if (
            selected["bootstrap_algorithm"] != HIERARCHICAL_BOOTSTRAP_ALGORITHM
            or selected["training_seed_clusters"] != len(plan["seeds"])
            or selected["replicates"]
            != (
                BOOTSTRAP_REPLICATES
                if result["authoritative"]
                else effects[
                    (
                        plan["seeds"][0],
                        selected["endpoint"],
                        selected["left"],
                        selected["right"],
                    )
                ].replicates
            )
        ):
            raise ContributionReportError(
                "hierarchical effect bootstrap identity is inconsistent"
            )
        low = _finite("hierarchical ci95_low", selected["ci95_low"])
        high = _finite("hierarchical ci95_high", selected["ci95_high"])
        if low > high:
            raise ContributionReportError("hierarchical CI bounds are reversed")
        expected_effect = statistics.fmean(
            effects[(seed, *key)].effect for seed in plan["seeds"]
        )
        _require_report_close(
            f"hierarchical effect {key}",
            selected["effect"],
            expected_effect,
        )
        hierarchical[key] = selected
    expected_hierarchical = set(_REQUIRED_EFFECTS) if phase == "phase4" else set()
    if set(hierarchical) != expected_hierarchical:
        raise ContributionReportError(
            "hierarchical effect set is inconsistent with the phase"
        )
    if (
        resolved_hierarchical is not None
        and [resolved_hierarchical[key] for key in sorted(resolved_hierarchical)]
        != result["hierarchical_seed_sequence_effects"]
    ):
        raise ContributionReportError(
            "hierarchical intervals differ from retained score artifacts"
        )

    primary = _primary_summary(rows, effects, hierarchical)
    if primary != result["primary"]:
        raise ContributionReportError(
            "primary G2/C2 metrics are inconsistent with condition evidence"
        )
    expected_gate = _gate_summary(
        phase=phase,
        rows=rows,
        effects=effects,
        primary=primary,
        authoritative=result["authoritative"],
    )
    if expected_gate != result["gate"]:
        raise ContributionReportError(
            "gate checks are inconsistent with paired and health evidence"
        )
    harvester = _normalize_harvester(
        result["harvester"],
        phase=phase,
        training_capture_manifest_sha256=training_capture,
        evaluation_capture_manifest_sha256=evaluation_capture,
    )
    expected_cost = _system_cost_summary(rows, q_rows, harvester=harvester)
    if expected_cost != result["secondary_end_to_end_cost"]:
        raise ContributionReportError(
            "secondary cost totals are inconsistent with timing evidence"
        )
    return rows, q_rows, effects, hierarchical, harvester


def validate_contribution_report(
    value: Mapping[str, Any],
    *,
    score_artifact_resolver: ScoreArtifactResolver | None = None,
) -> dict[str, Any]:
    """Validate both the immutable envelope and headline/gate semantics."""

    required = {
        "schema_version",
        "phase",
        "created_at",
        "authoritative",
        "status",
        "protocol_sha256",
        "training_capture_manifest_sha256",
        "evaluation_capture_manifest_sha256",
        "recomposition_sidecar_sha256",
        "stage_authorization",
        "primary",
        "gate",
        "conditions",
        "score_manifest_references",
        "score_evidence",
        "paired_sequence_effects",
        "hierarchical_seed_sequence_effects",
        "shared_q",
        "harvester",
        "secondary_end_to_end_cost",
        "headline",
        "report_sha256",
    }
    result = dict(_exact_mapping("contribution report", value, required))
    claimed = _sha256("report_sha256", result.pop("report_sha256"))
    if result.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ContributionReportError("contribution report schema is incompatible")
    if claimed != canonical_sha256(result):
        raise ContributionReportError("contribution report self-hash is invalid")
    phase = result.get("phase")
    if phase not in _PHASES:
        raise ContributionReportError("contribution report phase is invalid")
    authoritative = _strict_bool("report.authoritative", result["authoritative"])
    gate = _exact_mapping(
        "report gate",
        result["gate"],
        {"phase", "passed", "checks"},
    )
    if gate["phase"] != phase:
        raise ContributionReportError("report gate phase is inconsistent")
    checks = gate["checks"]
    if not isinstance(checks, Mapping) or not checks:
        raise ContributionReportError("report gate checks must be nonempty")
    normalized_checks = {
        str(name): _strict_bool(f"gate check {name}", passed)
        for name, passed in checks.items()
    }
    gate_passed = _strict_bool("gate.passed", gate["passed"])
    if gate_passed != all(normalized_checks.values()):
        raise ContributionReportError("report gate status is inconsistent")
    expected_status = _expected_status(
        phase=phase,
        authoritative=authoritative,
        gate_passed=gate_passed,
    )
    if result["status"] != expected_status:
        raise ContributionReportError("report status is inconsistent with its gate")
    for name in (
        "protocol_sha256",
        "training_capture_manifest_sha256",
        "evaluation_capture_manifest_sha256",
        "recomposition_sidecar_sha256",
    ):
        _sha256(f"report.{name}", result[name])
    stage = _exact_mapping(
        "stage authorization",
        result["stage_authorization"],
        {
            "phase1_receipt_sha256",
            "parent_report_sha256",
            "parent_phase",
            "parent_training_capture_manifest_sha256",
            "parent_evaluation_capture_manifest_sha256",
            "training_capture_lineage_receipt_sha256",
            "evaluation_capture_lineage_receipt_sha256",
        },
    )
    for name in (
        "phase1_receipt_sha256",
        "parent_training_capture_manifest_sha256",
        "parent_evaluation_capture_manifest_sha256",
        "training_capture_lineage_receipt_sha256",
        "evaluation_capture_lineage_receipt_sha256",
    ):
        _sha256(f"stage_authorization.{name}", stage[name])
    expected_parent_phase = _PHASE_PLAN[phase]["parent_phase"] or "phase1"
    if stage["parent_phase"] != expected_parent_phase:
        raise ContributionReportError("stage authorization parent phase is invalid")
    if phase == "phase2":
        if stage["parent_report_sha256"] is not None:
            raise ContributionReportError(
                "Phase-2 stage authorization cannot name a parent report"
            )
    else:
        _sha256(
            "stage_authorization.parent_report_sha256",
            stage["parent_report_sha256"],
        )
    if (
        stage["parent_training_capture_manifest_sha256"]
        == result["training_capture_manifest_sha256"]
        or stage["parent_evaluation_capture_manifest_sha256"]
        == result["evaluation_capture_manifest_sha256"]
    ):
        raise ContributionReportError(
            "stage authorization does not extend distinct split manifests"
        )
    _semantic_report_evidence(
        result,
        score_artifact_resolver=score_artifact_resolver,
    )
    primary = result["primary"]
    if not isinstance(primary, Mapping):
        raise ContributionReportError("report primary must be a mapping")
    if (
        primary.get("nominal_a_candidate_exposure_percent")
        != NOMINAL_A_EXPOSURE_PERCENT
    ):
        raise ContributionReportError(
            "report nominal candidate exposure is inconsistent"
        )
    expected_headline = render_contribution_headline(result)
    if result["headline"] != expected_headline:
        raise ContributionReportError(
            "report headline is inconsistent with eligibility"
        )
    result["report_sha256"] = claimed
    _ensure_json_safe(result)
    return result


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _ensure_json_safe(value: Any, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _ensure_json_safe(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _ensure_json_safe(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ContributionReportError(f"{path} is non-finite")


def write_contribution_report(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    score_artifact_resolver: ScoreArtifactResolver | None = None,
) -> Path:
    """Validate and atomically persist one strict-JSON report."""

    selected = validate_contribution_report(
        report,
        score_artifact_resolver=score_artifact_resolver,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(selected, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "BOOTSTRAP_ALGORITHM",
    "BOOTSTRAP_REPLICATES",
    "HIERARCHICAL_BOOTSTRAP_ALGORITHM",
    "MAX_MATERIAL_LATE_COLLAPSE_EV",
    "NOMINAL_A_EXPOSURE_PERCENT",
    "PHASE1_RECEIPT_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "SCORE_SCHEMA_VERSION",
    "ContributionReportError",
    "ContributionRunEvaluation",
    "CostLedger",
    "LearningCurvePoint",
    "PairedSequenceEffect",
    "PairedSequenceScores",
    "RunHealthEvidence",
    "SharedQEvaluation",
    "aggregate_paired_token_scores",
    "build_contribution_report",
    "build_phase1_receipt",
    "paired_sequence_bootstrap",
    "render_contribution_headline",
    "validate_contribution_report",
    "validate_phase1_receipt",
    "write_contribution_report",
]
