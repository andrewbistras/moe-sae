"""Immutable primary report for Experiment-5 activation-policy v3.

The report is assembled only from the sealed 36-cell primary evaluation
matrix, the six fresh v3 ReLU training results, and the six completed v2
Softplus trajectory references.  Optional panel-native A256 and H evidence
may be attached as read-only context, but neither can enter the primary
cells, contrasts, verdict, or cost comparison.

Every quality contrast is paired on the retained packed-sequence score rows.
The three-seed interval treats training seed and held-out sequence as crossed
factors: each replicate resamples seeds and applies one common packed-sequence
resample to every selected seed occurrence.  The JSON report is self-hashed,
while its deterministic Markdown rendering is content-hashed by the JSON.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_campaign import _read_self_hashed_json
from moe_sae.experiment5_routed_ladder import (
    ENDPOINTS,
    RoutedLadderTokenScores,
)
from moe_sae.experiment5_routed_ladder_campaign import (
    CompletedTrajectoryCheckpointReference,
    RoutedLadderCampaignPlanError,
    _CHIP_COUNT_BY_ACCELERATOR,
    _validate_h_checkpoint_lineage,
    parse_completed_trajectory_reference_manifest,
    parse_h_reference_manifest,
    validate_completed_checkpoint_lineage,
)
from moe_sae.experiment5_routed_ladder_evaluation_worker import (
    V3_BUNDLE_RESULT_SCHEMA_VERSION,
    V3_RESULT_SCHEMA_VERSION,
    load_merged_scores,
)
from moe_sae.experiment5_routed_ladder_v3_campaign import (
    RoutedLadderV3TrainingPlan,
    inspect_v3_training_completion,
    load_routed_ladder_v3_training_plan,
)
from moe_sae.experiment5_routed_ladder_v3_config import (
    A256_NATIVE_ACTIVATION,
    A256_NATIVE_CONFIG_SHA256,
    A256_NATIVE_LABEL,
    A256_NATIVE_SELECTION_POLICY,
    CHECKPOINT_TOKENS,
    CONDITIONS,
    PANELS,
    PRIMARY_CONTRAST_LABEL,
    PRIMARY_ESTIMAND,
    PROTOCOL_ID,
    SEEDS,
)
from moe_sae.experiment5_routed_ladder_v3_evaluation import (
    CELL_COUNT,
    load_v3_primary_evaluation_plan,
    validate_v3_primary_evaluation_plan,
)
from moe_sae.experiment5_routed_ladder_v3_provenance import (
    parse_a256_provenance_addendum,
    validate_a256_addendum_panel,
)

REPORT_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-policy-primary-report-v1"
)
MARKDOWN_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-policy-primary-report-markdown-v1"
)
REPORT_STATUS = "primary_verdict_sealed"
BOOTSTRAP_REPLICATES = 20_000
CONFIDENCE_LEVEL = 0.95
SEED_BOOTSTRAP_ALGORITHM = (
    "paired-packed-sequence-percentile-bootstrap-v1"
)
HIERARCHICAL_BOOTSTRAP_ALGORITHM = (
    "crossed-paired-seed-and-packed-sequence-percentile-bootstrap-v1"
)
PRIMARY_ENDPOINT = "component"
RELU_CONDITION = "A512ReLUInclusiveTopK"
SOFTPLUS_CONDITION = "A512Softplus"
CONTRAST = f"{SOFTPLUS_CONDITION}-{RELU_CONDITION}"
_SHA256_LENGTH = 64
_CROSSED_BOOTSTRAP_UNIT = "crossed_seed_and_packed_sequence"
_TRAINING_EXECUTION_BY_ORIGIN = {
    "fresh_v3": (
        "routed_ladder_v3_execution",
        "gemma4-experiment5-routed-activation-policy-v3-training-execution-v1",
    ),
    "imported_v2_exact_graph": (
        "routed_ladder_v2_execution",
        "gemma4-experiment5-routed-activation-ladder-training-execution-v1",
    ),
}
_A256_CONTEXT_SCOPE = {
    "condition": "A256Native",
    "evidentiary_role": "panel_native_descriptive_context_only",
    "included_in_primary_a512_inference": False,
    "included_in_primary_a512_estimand": False,
    "cross_panel_pooling_allowed": False,
    "across_seed_inference_allowed": False,
    "allowed_use": "within_panel_historical_scale_context",
    "prohibited_uses": [
        "matched_a512_activation_policy_contrast",
        "cross_panel_activation_policy_contrast",
        "three_seed_inference",
        "replacement_for_any_a512_cell",
    ],
}
_A256_CONTEXT_STATEMENT = (
    "Two panel-native historical A256 lineages and six checkpoint pointers "
    "are hash-pinned. No A256 quality score row is mixed into the v3 A512 "
    "ReLU-versus-Softplus inference."
)


class RoutedLadderV3ReportError(RuntimeError):
    """The v3 primary report or one of its immutable parents is invalid."""


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0)
    ):
        qualifier = "positive " if positive else ""
        raise RoutedLadderV3ReportError(
            f"{name} must be a finite {qualifier}number"
        )
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RoutedLadderV3ReportError(
            f"{name} must be a positive integer"
        )
    return int(value)


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RoutedLadderV3ReportError(f"{name} must be a SHA-256 digest")
    return value


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise RoutedLadderV3ReportError(
                f"cannot replay immutable report artifact: {path}"
            ) from error
        if existing != payload:
            raise RoutedLadderV3ReportError(
                f"refusing to replace immutable report artifact: {path}"
            )
        return
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class BundleTimingEvidence:
    panel: str
    seed: int
    accelerator_type: str
    chip_count: int
    successful_attempt_wall_seconds: float
    successful_attempt_chip_seconds: float
    timing_scope: str
    bundle_result_sha256: str
    bundle_result_file_sha256: str

    def __post_init__(self) -> None:
        if self.panel not in PANELS or self.seed not in SEEDS:
            raise RoutedLadderV3ReportError("bundle timing identity is invalid")
        if not self.accelerator_type or not self.timing_scope:
            raise RoutedLadderV3ReportError(
                "bundle timing hardware/scope is absent"
            )
        _positive_int(self.chip_count, "bundle chip_count")
        expected_chip_count = _CHIP_COUNT_BY_ACCELERATOR.get(
            self.accelerator_type
        )
        if expected_chip_count != self.chip_count:
            raise RoutedLadderV3ReportError(
                "bundle chip count differs from its accelerator type"
            )
        wall = _finite(
            self.successful_attempt_wall_seconds,
            "bundle successful wall seconds",
            positive=True,
        )
        chips = _finite(
            self.successful_attempt_chip_seconds,
            "bundle successful chip seconds",
            positive=True,
        )
        if not math.isclose(
            chips,
            wall * self.chip_count,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise RoutedLadderV3ReportError(
                "bundle chip-seconds do not equal chips times wall-seconds"
            )
        _sha256(self.bundle_result_sha256, "bundle result semantic hash")
        _sha256(self.bundle_result_file_sha256, "bundle result file hash")

    def report_row(self) -> dict[str, Any]:
        return {
            "panel": self.panel,
            "seed": self.seed,
            "accelerator_type": self.accelerator_type,
            "chip_count": self.chip_count,
            "successful_attempt_wall_seconds": (
                self.successful_attempt_wall_seconds
            ),
            "successful_attempt_chip_seconds": (
                self.successful_attempt_chip_seconds
            ),
            "timing_scope": self.timing_scope,
            "bundle_result_sha256": self.bundle_result_sha256,
            "bundle_result_file_sha256": self.bundle_result_file_sha256,
        }


@dataclass(frozen=True, slots=True)
class PrimaryCellEvidence:
    panel: str
    condition: str
    seed: int
    checkpoint_tokens: int
    origin: str
    result_path: str
    result_file_sha256: str
    result: Mapping[str, Any]
    scores: RoutedLadderTokenScores
    bundle_timing: BundleTimingEvidence

    @property
    def identity(self) -> tuple[str, str, int, int]:
        return self.panel, self.condition, self.seed, self.checkpoint_tokens

    def __post_init__(self) -> None:
        if (
            self.panel not in PANELS
            or self.condition not in CONDITIONS
            or self.seed not in SEEDS
            or self.checkpoint_tokens not in CHECKPOINT_TOKENS
            or self.origin not in {"fresh_v3", "imported_v2_exact_graph"}
            or not self.result_path
        ):
            raise RoutedLadderV3ReportError(
                f"primary cell identity is invalid: {self.identity}"
            )
        expected_origin = (
            "fresh_v3"
            if self.condition == RELU_CONDITION
            else "imported_v2_exact_graph"
        )
        if self.origin != expected_origin:
            raise RoutedLadderV3ReportError(
                f"primary cell origin changed: {self.identity}"
            )
        _sha256(self.result_file_sha256, "cell result file hash")
        if (
            self.scores.panel != self.panel
            or self.scores.condition != self.condition
            or self.scores.seed != self.seed
            or self.scores.checkpoint_tokens != self.checkpoint_tokens
            or self.bundle_timing.panel != self.panel
            or self.bundle_timing.seed != self.seed
        ):
            raise RoutedLadderV3ReportError(
                f"cell scores or bundle timing changed: {self.identity}"
            )


@dataclass(frozen=True, slots=True)
class TrainingCostEvidence:
    panel: str
    condition: str
    seed: int
    origin: str
    resource_id: str
    accelerator_type: str
    chip_count: int
    tokens: int
    marginal_wall_seconds: float
    chip_seconds: float
    tokens_per_second: float
    chip_seconds_scope: str
    chip_seconds_excludes: tuple[str, ...]
    compile_seconds: float
    lowering_seconds: float
    preprocessing_wall_seconds: float
    optimizer_wall_seconds: float
    checkpoint_seconds: float
    training_source_revision: str
    training_result_sha256: str
    training_result_file_sha256: str

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.panel, self.condition, self.seed

    def __post_init__(self) -> None:
        if (
            self.panel not in PANELS
            or self.condition not in CONDITIONS
            or self.seed not in SEEDS
            or self.origin not in {"fresh_v3", "imported_v2_exact_graph"}
            or not self.resource_id
            or not self.accelerator_type
            or not self.chip_seconds_scope
            or not self.chip_seconds_excludes
            or len(self.training_source_revision) != 40
        ):
            raise RoutedLadderV3ReportError(
                f"training cost identity is invalid: {self.identity}"
            )
        expected_origin = (
            "fresh_v3"
            if self.condition == RELU_CONDITION
            else "imported_v2_exact_graph"
        )
        if self.origin != expected_origin or self.tokens != 10_000_000:
            raise RoutedLadderV3ReportError(
                f"training cost origin/token count changed: {self.identity}"
            )
        _positive_int(self.chip_count, "training chip_count")
        expected_chip_count = _CHIP_COUNT_BY_ACCELERATOR.get(
            self.accelerator_type
        )
        if expected_chip_count != self.chip_count:
            raise RoutedLadderV3ReportError(
                f"training chip count differs from accelerator type: "
                f"{self.identity}"
            )
        wall = _finite(
            self.marginal_wall_seconds,
            "training marginal_wall_seconds",
            positive=True,
        )
        chips = _finite(
            self.chip_seconds,
            "training chip_seconds",
            positive=True,
        )
        rate = _finite(
            self.tokens_per_second,
            "training tokens_per_second",
            positive=True,
        )
        if (
            not math.isclose(
                chips,
                wall * self.chip_count,
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            or not math.isclose(
                rate,
                self.tokens / wall,
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
        ):
            raise RoutedLadderV3ReportError(
                f"training cost arithmetic changed: {self.identity}"
            )
        for name in (
            "compile_seconds",
            "lowering_seconds",
            "preprocessing_wall_seconds",
            "optimizer_wall_seconds",
            "checkpoint_seconds",
        ):
            value = _finite(getattr(self, name), f"training {name}")
            if value < 0:
                raise RoutedLadderV3ReportError(
                    f"training {name} must be nonnegative"
                )
        if not math.isclose(
            wall,
            self.preprocessing_wall_seconds + self.optimizer_wall_seconds,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise RoutedLadderV3ReportError(
                f"training marginal wall-time scope changed: {self.identity}"
            )
        _sha256(
            self.training_result_sha256,
            "training result semantic hash",
        )
        _sha256(
            self.training_result_file_sha256,
            "training result file hash",
        )

    def report_row(self) -> dict[str, Any]:
        return {
            "panel": self.panel,
            "condition": self.condition,
            "seed": self.seed,
            "origin": self.origin,
            "resource_id": self.resource_id,
            "accelerator_type": self.accelerator_type,
            "chip_count": self.chip_count,
            "tokens": self.tokens,
            "marginal_wall_seconds": self.marginal_wall_seconds,
            "chip_seconds": self.chip_seconds,
            "tokens_per_second": self.tokens_per_second,
            "chip_seconds_scope": self.chip_seconds_scope,
            "chip_seconds_excludes": list(self.chip_seconds_excludes),
            "compile_seconds_excluded": self.compile_seconds,
            "lowering_seconds_excluded": self.lowering_seconds,
            "checkpoint_seconds_excluded": self.checkpoint_seconds,
            "preprocessing_wall_seconds_included": (
                self.preprocessing_wall_seconds
            ),
            "optimizer_wall_seconds_included": self.optimizer_wall_seconds,
            "training_source_revision": self.training_source_revision,
            "training_result_sha256": self.training_result_sha256,
            "training_result_file_sha256": (
                self.training_result_file_sha256
            ),
        }


def _pair_identity(
    left: RoutedLadderTokenScores,
    right: RoutedLadderTokenScores,
    *,
    endpoint: str,
) -> None:
    if (
        left.condition != RELU_CONDITION
        or right.condition != SOFTPLUS_CONDITION
        or left.identity != right.identity
        or endpoint not in ENDPOINTS
        or left.capture_manifest_sha256 != right.capture_manifest_sha256
        or left.recomposition_sidecar_sha256
        != right.recomposition_sidecar_sha256
        or not np.array_equal(left.sequence_id, right.sequence_id)
        or not np.array_equal(left.position, right.position)
        or left.token_identity_sha256 != right.token_identity_sha256
        or not np.array_equal(
            left.endpoint_sst[endpoint],
            right.endpoint_sst[endpoint],
        )
    ):
        raise RoutedLadderV3ReportError(
            "paired ReLU/Softplus token or common-SST identity changed"
        )


def _bootstrap_seed(
    *,
    panel: str,
    seed: int | None,
    checkpoint_tokens: int,
    endpoint: str,
    token_identity_sha256: str,
    level: str,
) -> int:
    payload = "|".join(
        (
            PROTOCOL_ID,
            PRIMARY_CONTRAST_LABEL,
            panel,
            "all-seeds" if seed is None else str(seed),
            str(checkpoint_tokens),
            endpoint,
            token_identity_sha256,
            level,
        )
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("ascii")).digest()[:8],
        "big",
        signed=False,
    )


def _sequence_sufficient_statistics(
    left: RoutedLadderTokenScores,
    right: RoutedLadderTokenScores,
    *,
    endpoint: str,
) -> tuple[np.ndarray, np.ndarray]:
    _pair_identity(left, right, endpoint=endpoint)
    sequence, inverse = np.unique(left.sequence_id, return_inverse=True)
    sequence_sst = np.bincount(
        inverse,
        weights=left.endpoint_sst[endpoint],
        minlength=sequence.size,
    ).astype(np.float64)
    # EV_softplus - EV_relu = (SSE_relu - SSE_softplus) / common SST.
    sequence_delta = np.bincount(
        inverse,
        weights=(
            left.endpoint_sse[endpoint] - right.endpoint_sse[endpoint]
        ),
        minlength=sequence.size,
    ).astype(np.float64)
    if np.any(sequence_sst <= 0):
        raise RoutedLadderV3ReportError(
            "paired bootstrap sequence SST must be positive"
        )
    return sequence_sst, sequence_delta


def paired_v3_bootstrap(
    relu_scores: RoutedLadderTokenScores,
    softplus_scores: RoutedLadderTokenScores,
    *,
    endpoint: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Deterministic packed-sequence interval for one paired seed."""

    if replicates != BOOTSTRAP_REPLICATES:
        raise RoutedLadderV3ReportError(
            "authoritative v3 report requires exactly 20,000 replicates"
        )
    _positive_int(chunk_size, "bootstrap chunk_size")
    denominator_rows, delta_rows = _sequence_sufficient_statistics(
        relu_scores,
        softplus_scores,
        endpoint=endpoint,
    )
    estimate = float(
        delta_rows.sum(dtype=np.float64)
        / denominator_rows.sum(dtype=np.float64)
    )
    seed = _bootstrap_seed(
        panel=relu_scores.panel,
        seed=relu_scores.seed,
        checkpoint_tokens=relu_scores.checkpoint_tokens,
        endpoint=endpoint,
        token_identity_sha256=relu_scores.token_identity_sha256,
        level="packed-sequence",
    )
    samples = np.empty(replicates, dtype=np.float64)
    if np.allclose(
        delta_rows,
        estimate * denominator_rows,
        rtol=1e-13,
        atol=1e-13,
    ):
        samples.fill(estimate)
    else:
        rng = np.random.Generator(np.random.PCG64(seed))
        for start in range(0, replicates, chunk_size):
            end = min(start + chunk_size, replicates)
            draws = rng.integers(
                0,
                denominator_rows.size,
                size=(end - start, denominator_rows.size),
                endpoint=False,
            )
            denominator = denominator_rows[draws].sum(
                axis=1,
                dtype=np.float64,
            )
            samples[start:end] = (
                delta_rows[draws].sum(axis=1, dtype=np.float64)
                / denominator
            )
    lower, upper = np.quantile(samples, (0.025, 0.975), method="linear")
    return {
        "seed": relu_scores.seed,
        "estimate_ev_difference": estimate,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_unit": "packed_sequence",
        "bootstrap_algorithm": SEED_BOOTSTRAP_ALGORITHM,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "evaluation_tokens": relu_scores.tokens,
        "packed_sequences": relu_scores.packed_sequences,
        "token_identity_sha256": relu_scores.token_identity_sha256,
    }


def hierarchical_v3_bootstrap(
    pairs: Sequence[
        tuple[RoutedLadderTokenScores, RoutedLadderTokenScores]
    ],
    *,
    endpoint: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Deterministic crossed seed-by-packed-sequence interval for n=3.

    Every model seed was evaluated on the same held-out packed sequences.
    One sequence resample is therefore shared by all seed occurrences in a
    replicate, preserving the observed cross-seed sequence covariance.
    """

    if replicates != BOOTSTRAP_REPLICATES:
        raise RoutedLadderV3ReportError(
            "authoritative v3 report requires exactly 20,000 replicates"
        )
    _positive_int(chunk_size, "hierarchical bootstrap chunk_size")
    by_seed: dict[
        int,
        tuple[RoutedLadderTokenScores, RoutedLadderTokenScores],
    ] = {}
    for left, right in pairs:
        _pair_identity(left, right, endpoint=endpoint)
        if left.seed in by_seed:
            raise RoutedLadderV3ReportError(
                "hierarchical bootstrap contains a duplicate seed"
            )
        by_seed[left.seed] = (left, right)
    if set(by_seed) != set(SEEDS):
        raise RoutedLadderV3ReportError(
            "hierarchical bootstrap requires exactly paired seeds 0, 1, and 2"
        )
    ordered = [by_seed[seed] for seed in SEEDS]
    first = ordered[0][0]
    for left, _right in ordered[1:]:
        if (
            left.panel != first.panel
            or left.checkpoint_tokens != first.checkpoint_tokens
            or left.capture_manifest_sha256
            != first.capture_manifest_sha256
            or left.recomposition_sidecar_sha256
            != first.recomposition_sidecar_sha256
            or not np.array_equal(left.sequence_id, first.sequence_id)
            or not np.array_equal(left.position, first.position)
            or not np.array_equal(
                left.endpoint_sst[endpoint],
                first.endpoint_sst[endpoint],
            )
        ):
            raise RoutedLadderV3ReportError(
                "three-seed bootstrap cohorts do not share one token/SST parent"
            )
    denominators: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    seed_effects: list[float] = []
    for left, right in ordered:
        denominator, delta = _sequence_sufficient_statistics(
            left,
            right,
            endpoint=endpoint,
        )
        denominators.append(denominator)
        deltas.append(delta)
        seed_effects.append(
            float(
                delta.sum(dtype=np.float64)
                / denominator.sum(dtype=np.float64)
            )
        )
    estimate = float(math.fsum(seed_effects) / len(SEEDS))
    seed = _bootstrap_seed(
        panel=first.panel,
        seed=None,
        checkpoint_tokens=first.checkpoint_tokens,
        endpoint=endpoint,
        token_identity_sha256=first.token_identity_sha256,
        level="crossed-seed-and-packed-sequence",
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(replicates, dtype=np.float64)
    common_denominator_rows = denominators[0]
    for start in range(0, replicates, chunk_size):
        end = min(start + chunk_size, replicates)
        rows = end - start
        seed_draws = rng.integers(
            0,
            len(SEEDS),
            size=(rows, len(SEEDS)),
            endpoint=False,
        )
        sequence_draws = rng.integers(
            0,
            common_denominator_rows.size,
            size=(rows, common_denominator_rows.size),
            endpoint=False,
        )
        denominator = common_denominator_rows[sequence_draws].sum(
            axis=1,
            dtype=np.float64,
        )
        replicate_delta = np.zeros(rows, dtype=np.float64)
        for slot in range(len(SEEDS)):
            selected_seed = seed_draws[:, slot]
            for seed_index in range(len(SEEDS)):
                mask = selected_seed == seed_index
                if not np.any(mask):
                    continue
                delta_rows = deltas[seed_index]
                replicate_delta[mask] += delta_rows[
                    sequence_draws[mask]
                ].sum(
                    axis=1,
                    dtype=np.float64,
                )
        samples[start:end] = (
            replicate_delta / len(SEEDS) / denominator
        )
    lower, upper = np.quantile(samples, (0.025, 0.975), method="linear")
    return {
        "estimate_mean_ev_difference": estimate,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_unit": _CROSSED_BOOTSTRAP_UNIT,
        "bootstrap_algorithm": HIERARCHICAL_BOOTSTRAP_ALGORITHM,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "seeds": list(SEEDS),
        "positive_seed_count": sum(value > 0 for value in seed_effects),
        "seed_effects": [
            {
                "seed": selected_seed,
                "estimate_ev_difference": seed_effects[index],
            }
            for index, selected_seed in enumerate(SEEDS)
        ],
        "evaluation_tokens_per_seed": first.tokens,
        "packed_sequences_per_seed": first.packed_sequences,
        "token_identity_sha256": first.token_identity_sha256,
    }


def _load_json(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RoutedLadderV3ReportError(f"cannot read {name}: {path}") from error
    if not isinstance(value, dict):
        raise RoutedLadderV3ReportError(f"{name} must be a JSON object")
    return value, payload


def _validate_bundle_timing(
    bundle: Mapping[str, Any],
    *,
    panel: str,
    seed: int,
    payload: bytes,
) -> BundleTimingEvidence:
    timing = bundle.get("timing")
    if (
        not isinstance(timing, Mapping)
        or set(timing)
        != {
            "accelerator_type",
            "chip_count",
            "successful_attempt_wall_seconds",
            "successful_attempt_chip_seconds",
            "timing_scope",
        }
    ):
        raise RoutedLadderV3ReportError(
            f"bundle timing fields changed: {panel}/seed{seed}"
        )
    return BundleTimingEvidence(
        panel=panel,
        seed=seed,
        accelerator_type=str(timing["accelerator_type"]),
        chip_count=_positive_int(
            timing["chip_count"],
            f"{panel}/seed{seed} bundle chip_count",
        ),
        successful_attempt_wall_seconds=_finite(
            timing["successful_attempt_wall_seconds"],
            f"{panel}/seed{seed} bundle wall seconds",
            positive=True,
        ),
        successful_attempt_chip_seconds=_finite(
            timing["successful_attempt_chip_seconds"],
            f"{panel}/seed{seed} bundle chip seconds",
            positive=True,
        ),
        timing_scope=str(timing["timing_scope"]),
        bundle_result_sha256=_sha256(
            bundle.get("result_sha256"),
            f"{panel}/seed{seed} bundle semantic hash",
        ),
        bundle_result_file_sha256=hashlib.sha256(payload).hexdigest(),
    )


def collect_primary_cell_evidence(
    evaluation_plan: Mapping[str, Any],
) -> tuple[
    tuple[PrimaryCellEvidence, ...],
    tuple[BundleTimingEvidence, ...],
]:
    """Load and validate every cell and resident-bundle result envelope."""

    plan = validate_v3_primary_evaluation_plan(evaluation_plan)
    cells: list[PrimaryCellEvidence] = []
    timings: list[BundleTimingEvidence] = []
    for raw_bundle in plan["bundles"]:
        panel = str(raw_bundle["panel"])
        seed = int(raw_bundle["seed"])
        manifest = raw_bundle["operator_manifest"]
        bundle_path = Path(str(raw_bundle["bundle_completion_path"]))
        try:
            bundle, bundle_payload = _read_self_hashed_json(
                bundle_path,
                hash_field="result_sha256",
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RoutedLadderV3ReportError(
                f"cannot validate primary bundle result: {panel}/seed{seed}"
            ) from error
        if (
            bundle.get("schema_version") != V3_BUNDLE_RESULT_SCHEMA_VERSION
            or bundle.get("protocol_id") != PROTOCOL_ID
            or bundle.get("status") != "bundle_evaluated"
            or bundle.get("panel") != panel
            or bundle.get("seed") != seed
            or bundle.get("source_revision")
            != plan["evaluation_source_revision"]
            or bundle.get("operator_manifest_sha256")
            != manifest["manifest_sha256"]
        ):
            raise RoutedLadderV3ReportError(
                f"primary bundle identity changed: {panel}/seed{seed}"
            )
        _sha256(
            bundle.get("bundle_spec_sha256"),
            f"{panel}/seed{seed} bundle spec hash",
        )
        timing = _validate_bundle_timing(
            bundle,
            panel=panel,
            seed=seed,
            payload=bundle_payload,
        )
        timings.append(timing)
        expected_bundle_cells: list[dict[str, Any]] = []
        for planned in manifest["cells"]:
            result_path = (
                Path(str(planned["result_root_controller"]))
                / "evaluation-result.json"
            )
            try:
                scores = load_merged_scores(result_path)
                result, result_payload = _read_self_hashed_json(
                    result_path,
                    hash_field="result_sha256",
                )
            except (
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                raise RoutedLadderV3ReportError(
                    f"cannot validate v3 primary cell: "
                    f"{panel}/{planned['condition']}/seed{seed}/"
                    f"{planned['checkpoint_tokens']}"
                ) from error
            condition = str(planned["condition"])
            checkpoint = int(planned["checkpoint_tokens"])
            if (
                result.get("schema_version") != V3_RESULT_SCHEMA_VERSION
                or result.get("protocol_id") != PROTOCOL_ID
                or result.get("status") != "cell_evaluated"
                or result.get("panel") != panel
                or result.get("condition") != condition
                or result.get("seed") != seed
                or result.get("checkpoint_tokens") != checkpoint
                or result.get("source_revision")
                != plan["evaluation_source_revision"]
                or result.get("training_source_revision")
                != planned["training_source_revision"]
                or result.get("parent_training_result_sha256")
                != planned["parent_training_result_sha256"]
                or result.get("parent_training_result_file_sha256")
                != planned["parent_training_result_file_sha256"]
                or result.get("parent_run_contract_sha256")
                != planned["parent_run_contract_sha256"]
                or result.get("parent_run_contract_file_sha256")
                != planned["parent_run_contract_file_sha256"]
                or result.get("parent_checkpoint_pointer_sha256")
                != planned["parent_checkpoint_pointer_sha256"]
                or result.get("parent_checkpoint_pointer_file_sha256")
                != planned["parent_checkpoint_pointer_file_sha256"]
                or scores.panel != panel
                or scores.condition != condition
                or scores.seed != seed
                or scores.checkpoint_tokens != checkpoint
                or scores.source_revision
                != plan["evaluation_source_revision"]
            ):
                raise RoutedLadderV3ReportError(
                    f"primary cell lineage changed: "
                    f"{panel}/{condition}/seed{seed}/{checkpoint}"
                )
            result_file_hash = hashlib.sha256(result_payload).hexdigest()
            expected_bundle_cells.append(
                {
                    "condition": condition,
                    "checkpoint_tokens": checkpoint,
                    "training_source_revision": planned[
                        "training_source_revision"
                    ],
                    "parent_training_result_sha256": planned[
                        "parent_training_result_sha256"
                    ],
                    "parent_training_result_file_sha256": planned[
                        "parent_training_result_file_sha256"
                    ],
                    "parent_run_contract_sha256": planned[
                        "parent_run_contract_sha256"
                    ],
                    "parent_run_contract_file_sha256": planned[
                        "parent_run_contract_file_sha256"
                    ],
                    "parent_checkpoint_pointer_sha256": planned[
                        "parent_checkpoint_pointer_sha256"
                    ],
                    "parent_checkpoint_pointer_file_sha256": planned[
                        "parent_checkpoint_pointer_file_sha256"
                    ],
                    "result_root": planned["result_root_remote"],
                    "result_sha256": result["result_sha256"],
                    "result_file_sha256": result_file_hash,
                    "score_artifact_sha256": result[
                        "score_artifact_sha256"
                    ],
                    "score_data_sha256": result["score_data_sha256"],
                    "score_data_bytes": result["score_data_bytes"],
                }
            )
            cells.append(
                PrimaryCellEvidence(
                    panel=panel,
                    condition=condition,
                    seed=seed,
                    checkpoint_tokens=checkpoint,
                    origin=(
                        "fresh_v3"
                        if planned["reference_kind"] == "trained_v3"
                        else "imported_v2_exact_graph"
                    ),
                    result_path=str(result_path),
                    result_file_sha256=result_file_hash,
                    result=copy.deepcopy(result),
                    scores=scores,
                    bundle_timing=timing,
                )
            )
        if bundle.get("cell_results") != expected_bundle_cells:
            raise RoutedLadderV3ReportError(
                f"bundle cell-result inventory changed: {panel}/seed{seed}"
            )
    identities = {cell.identity for cell in cells}
    expected = {
        (panel, condition, seed, checkpoint)
        for panel in PANELS
        for condition in CONDITIONS
        for seed in SEEDS
        for checkpoint in CHECKPOINT_TOKENS
    }
    if (
        len(cells) != CELL_COUNT
        or identities != expected
        or len(timings) != len(PANELS) * len(SEEDS)
        or {(row.panel, row.seed) for row in timings}
        != {(panel, seed) for panel in PANELS for seed in SEEDS}
    ):
        raise RoutedLadderV3ReportError(
            "primary evidence does not contain the exact 36-cell/six-bundle grid"
        )
    return (
        tuple(
            sorted(
                cells,
                key=lambda row: (
                    PANELS.index(row.panel),
                    row.checkpoint_tokens,
                    row.seed,
                    CONDITIONS.index(row.condition),
                ),
            )
        ),
        tuple(
            sorted(
                timings,
                key=lambda row: (PANELS.index(row.panel), row.seed),
            )
        ),
    )


def _validate_bundle_resource_ownership(
    evaluation_plan: Mapping[str, Any],
    training_plan: RoutedLadderV3TrainingPlan,
    timings: Sequence[BundleTimingEvidence],
) -> None:
    """Bind each measured bundle timing to its immutable training-plan owner."""

    timing_grid = {(row.panel, row.seed): row for row in timings}
    fresh_owner = {
        (task.panel, task.seed): task.eligible_resource_ids[0]
        for task in training_plan.trajectories
    }
    for bundle in evaluation_plan["bundles"]:
        identity = (str(bundle["panel"]), int(bundle["seed"]))
        owner_id = str(bundle["resource_id"])
        if fresh_owner.get(identity) != owner_id:
            raise RoutedLadderV3ReportError(
                f"evaluation bundle owner differs from fresh trajectory: "
                f"{identity}"
            )
        resource = training_plan.resource(owner_id)
        timing = timing_grid.get(identity)
        expected_chips = _CHIP_COUNT_BY_ACCELERATOR.get(
            resource.accelerator_type
        )
        if (
            timing is None
            or timing.accelerator_type != resource.accelerator_type
            or timing.chip_count != expected_chips
        ):
            raise RoutedLadderV3ReportError(
                f"evaluation bundle hardware differs from planned owner: "
                f"{identity}"
            )


def _result_timing_number(
    result: Mapping[str, Any],
    name: str,
    *,
    positive: bool = False,
) -> float:
    return _finite(
        result.get(name),
        f"training result {name}",
        positive=positive,
    )


def _training_cost_from_result(
    result: Mapping[str, Any],
    payload: bytes,
    *,
    panel: str,
    condition: str,
    seed: int,
    origin: str,
    resource_id: str,
    accelerator_type: str,
    training_source_revision: str,
) -> TrainingCostEvidence:
    identity = result.get("run_identity")
    run_plan = identity.get("plan") if isinstance(identity, Mapping) else None
    expected_worker_condition = (
        RELU_CONDITION if condition == RELU_CONDITION else SOFTPLUS_CONDITION
    )
    if (
        result.get("schema_version")
        != "gemma4-contribution-sae-training-result-v1"
        or result.get("status") not in {"completed", "already_complete"}
        or result.get("tokens") != 10_000_000
        or result.get("source_revision") != training_source_revision
        or not isinstance(run_plan, Mapping)
        or run_plan.get("condition") != expected_worker_condition
        or run_plan.get("seed") != seed
        or run_plan.get("total_tokens") != 10_000_000
    ):
        raise RoutedLadderV3ReportError(
            f"training result identity changed: {panel}/{condition}/seed{seed}"
        )
    scientific_model = run_plan.get(
        "scientific_model",
        identity.get("scientific_model")
        if isinstance(identity, Mapping)
        else None,
    )
    if scientific_model is not None and scientific_model != panel:
        raise RoutedLadderV3ReportError(
            f"training result panel changed: {panel}/{condition}/seed{seed}"
        )
    marginal = _result_timing_number(
        result,
        "marginal_wall_seconds",
        positive=True,
    )
    chip_seconds = _result_timing_number(
        result,
        "chip_seconds",
        positive=True,
    )
    raw_chip_count = chip_seconds / marginal
    chip_count = round(raw_chip_count)
    expected_chip_count = _CHIP_COUNT_BY_ACCELERATOR.get(accelerator_type)
    execution_identity = _TRAINING_EXECUTION_BY_ORIGIN.get(origin)
    execution = (
        result.get(execution_identity[0])
        if execution_identity is not None
        else None
    )
    if (
        chip_count <= 0
        or expected_chip_count is None
        or chip_count != expected_chip_count
        or not math.isclose(
            raw_chip_count,
            chip_count,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or execution_identity is None
        or not isinstance(execution, Mapping)
        or execution.get("schema_version") != execution_identity[1]
        or execution.get("accelerator_type") != accelerator_type
        or execution.get("global_chip_count") != expected_chip_count
    ):
        raise RoutedLadderV3ReportError(
            f"training execution hardware differs from its planned owner: "
            f"{panel}/{condition}/seed{seed}"
        )
    scope = result.get("chip_seconds_scope")
    excludes = result.get("chip_seconds_excludes")
    if (
        not isinstance(scope, str)
        or not scope
        or not isinstance(excludes, list)
        or not excludes
        or not all(isinstance(item, str) and item for item in excludes)
    ):
        raise RoutedLadderV3ReportError(
            f"training cost scope changed: {panel}/{condition}/seed{seed}"
        )
    return TrainingCostEvidence(
        panel=panel,
        condition=condition,
        seed=seed,
        origin=origin,
        resource_id=resource_id,
        accelerator_type=accelerator_type,
        chip_count=chip_count,
        tokens=10_000_000,
        marginal_wall_seconds=marginal,
        chip_seconds=chip_seconds,
        tokens_per_second=_result_timing_number(
            result,
            "tokens_per_second",
            positive=True,
        ),
        chip_seconds_scope=scope,
        chip_seconds_excludes=tuple(excludes),
        compile_seconds=_result_timing_number(result, "compile_seconds"),
        lowering_seconds=_result_timing_number(result, "lowering_seconds"),
        preprocessing_wall_seconds=_result_timing_number(
            result,
            "preprocessing_wall_seconds",
        ),
        optimizer_wall_seconds=_result_timing_number(
            result,
            "optimizer_wall_seconds",
        ),
        checkpoint_seconds=_result_timing_number(
            result,
            "checkpoint_seconds",
        ),
        training_source_revision=training_source_revision,
        training_result_sha256=_sha256(
            result.get("result_sha256"),
            "training result semantic hash",
        ),
        training_result_file_sha256=hashlib.sha256(payload).hexdigest(),
    )


def collect_training_cost_evidence(
    training_plan: RoutedLadderV3TrainingPlan,
    completed_manifest: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
) -> tuple[TrainingCostEvidence, ...]:
    """Validate and collect twelve immutable marginal training costs."""

    manifest, references = parse_completed_trajectory_reference_manifest(
        completed_manifest
    )
    if manifest["manifest_sha256"] != expected_manifest_sha256:
        raise RoutedLadderV3ReportError(
            "Softplus reference manifest differs from the evaluation plan"
        )
    costs: list[TrainingCostEvidence] = []
    for task in training_plan.trajectories:
        if inspect_v3_training_completion(training_plan, task) is None:
            raise RoutedLadderV3ReportError(
                f"fresh v3 training is incomplete: {task.task_id}"
            )
        try:
            result, payload = _read_self_hashed_json(
                task.completion_path_controller,
                hash_field="result_sha256",
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RoutedLadderV3ReportError(
                f"cannot validate fresh training result: {task.task_id}"
            ) from error
        resource = training_plan.resource(task.eligible_resource_ids[0])
        costs.append(
            _training_cost_from_result(
                result,
                payload,
                panel=task.panel,
                condition=RELU_CONDITION,
                seed=task.seed,
                origin="fresh_v3",
                resource_id=resource.resource_id,
                accelerator_type=resource.accelerator_type,
                training_source_revision=training_plan.project_commit,
            )
        )
    by_role: dict[
        tuple[str, str, int],
        list[CompletedTrajectoryCheckpointReference],
    ] = {}
    for reference in references:
        validate_completed_checkpoint_lineage(reference)
        by_role.setdefault(reference.trajectory_identity, []).append(reference)
    expected_imports = {
        (panel, SOFTPLUS_CONDITION, seed)
        for panel in PANELS
        for seed in SEEDS
    }
    if set(by_role) != expected_imports:
        raise RoutedLadderV3ReportError(
            "Softplus reference manifest does not contain all six trajectories"
        )
    for role in sorted(
        by_role,
        key=lambda item: (PANELS.index(item[0]), item[2]),
    ):
        rows = by_role[role]
        exemplar = rows[0]
        if (
            len(rows) != len(CHECKPOINT_TOKENS)
            or {row.checkpoint_tokens for row in rows}
            != set(CHECKPOINT_TOKENS)
            or len({row.training_result for row in rows}) != 1
            or len({row.training_result_sha256 for row in rows}) != 1
        ):
            raise RoutedLadderV3ReportError(
                f"Softplus trajectory references are incomplete: {role}"
            )
        try:
            result, payload = _read_self_hashed_json(
                exemplar.training_result.controller_path,
                hash_field="result_sha256",
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RoutedLadderV3ReportError(
                f"cannot validate imported training result: {role}"
            ) from error
        if (
            result.get("result_sha256")
            != exemplar.training_result_sha256
            or hashlib.sha256(payload).hexdigest()
            != exemplar.training_result.sha256
        ):
            raise RoutedLadderV3ReportError(
                f"imported training result pin changed: {role}"
            )
        costs.append(
            _training_cost_from_result(
                result,
                payload,
                panel=exemplar.panel,
                condition=SOFTPLUS_CONDITION,
                seed=exemplar.seed,
                origin="imported_v2_exact_graph",
                resource_id=exemplar.resource_id,
                accelerator_type=exemplar.accelerator_type,
                training_source_revision=exemplar.training_source_revision,
            )
        )
    expected_cost_roles = {
        (panel, condition, seed)
        for panel in PANELS
        for condition in CONDITIONS
        for seed in SEEDS
    }
    if (
        len(costs) != len(expected_cost_roles)
        or {row.identity for row in costs} != expected_cost_roles
    ):
        raise RoutedLadderV3ReportError(
            "training cost evidence does not contain the exact twelve roles"
        )
    scopes = {
        (row.chip_seconds_scope, row.chip_seconds_excludes) for row in costs
    }
    if len(scopes) != 1:
        raise RoutedLadderV3ReportError(
            "paired training costs do not share one measured-cost scope"
        )
    return tuple(
        sorted(
            costs,
            key=lambda row: (
                PANELS.index(row.panel),
                row.seed,
                CONDITIONS.index(row.condition),
            ),
        )
    )


def _validate_common_evaluation(
    cells: Sequence[PrimaryCellEvidence],
) -> dict[str, Any]:
    if len(cells) != CELL_COUNT:
        raise RoutedLadderV3ReportError(
            "common evaluation validation requires all 36 cells"
        )
    first = cells[0].scores
    common_sst = {
        endpoint: np.asarray(first.endpoint_sst[endpoint], dtype=np.float64)
        for endpoint in ENDPOINTS
    }
    for cell in cells[1:]:
        scores = cell.scores
        if (
            scores.tokens != first.tokens
            or scores.packed_sequences != first.packed_sequences
            or scores.token_identity_sha256 != first.token_identity_sha256
            or scores.capture_manifest_sha256
            != first.capture_manifest_sha256
            or scores.recomposition_sidecar_sha256
            != first.recomposition_sidecar_sha256
            or not np.array_equal(scores.sequence_id, first.sequence_id)
            or not np.array_equal(scores.position, first.position)
        ):
            raise RoutedLadderV3ReportError(
                f"common evaluation token identity changed: {cell.identity}"
            )
        for endpoint in ENDPOINTS:
            if not np.array_equal(
                scores.endpoint_sst[endpoint],
                common_sst[endpoint],
            ):
                raise RoutedLadderV3ReportError(
                    f"common {endpoint} SST changed: {cell.identity}"
                )
    return {
        "evaluation_tokens": first.tokens,
        "packed_sequences": first.packed_sequences,
        "token_identity_sha256": first.token_identity_sha256,
        "capture_manifest_sha256": first.capture_manifest_sha256,
        "recomposition_sidecar_sha256": (
            first.recomposition_sidecar_sha256
        ),
        "endpoint_sst": {
            endpoint: {
                "sst": float(
                    common_sst[endpoint].sum(dtype=np.float64)
                ),
                "sst_sha256": first.endpoint_sst_sha256(endpoint),
            }
            for endpoint in ENDPOINTS
        },
        "scope": (
            "all_36_cells_share_exact_token_rows_and_target_only_sst_rows"
        ),
    }


def _sparsity_row(cell: PrimaryCellEvidence) -> dict[str, Any]:
    result = cell.result
    selected = _positive_int(
        result.get("selected_slot_count"),
        f"{cell.identity} selected_slot_count",
    )
    tokens = _positive_int(result.get("tokens"), f"{cell.identity} tokens")
    positive_sum = result.get("positive_l0_sum")
    if (
        isinstance(positive_sum, bool)
        or not isinstance(positive_sum, int)
        or not 0 <= positive_sum <= tokens * selected
    ):
        raise RoutedLadderV3ReportError(
            f"{cell.identity} positive_l0_sum is invalid"
        )
    positive_l0 = _finite(
        result.get("positive_l0"),
        f"{cell.identity} positive_l0",
    )
    zero_fraction = _finite(
        result.get("zero_selected_slot_fraction"),
        f"{cell.identity} zero_selected_slot_fraction",
    )
    if (
        selected != 32
        or not 0 <= positive_l0 <= selected
        or not 0 <= zero_fraction <= 1
        or not math.isclose(
            positive_l0,
            positive_sum / tokens,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            zero_fraction,
            1.0 - positive_l0 / selected,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise RoutedLadderV3ReportError(
            f"{cell.identity} sparsity arithmetic changed"
        )
    contract = result.get("positive_l0_contract")
    if cell.condition == SOFTPLUS_CONDITION:
        if (
            contract != "exact_selected_slots"
            or not math.isclose(positive_l0, 32.0, abs_tol=1e-12)
            or not math.isclose(zero_fraction, 0.0, abs_tol=1e-12)
        ):
            raise RoutedLadderV3ReportError(
                f"{cell.identity} Softplus is not exact-positive"
            )
    elif contract != "bounded_zero_to_selected_slots":
        raise RoutedLadderV3ReportError(
            f"{cell.identity} ReLU inclusive-zero contract changed"
        )
    dead_rate = _finite(
        result.get("dead_rate"),
        f"{cell.identity} dead_rate",
    )
    if not 0 <= dead_rate <= 1:
        raise RoutedLadderV3ReportError(
            f"{cell.identity} dead rate is outside [0,1]"
        )
    return {
        "selected_slot_count": selected,
        "positive_l0_sum": positive_sum,
        "positive_l0": positive_l0,
        "zero_selected_slot_fraction": zero_fraction,
        "positive_l0_contract": contract,
        "dead_rate": dead_rate,
        "dead_rate_scope": result.get("dead_rate_scope"),
        "total_features": _positive_int(
            result.get("total_features"),
            f"{cell.identity} total_features",
        ),
    }


def _cell_report_row(cell: PrimaryCellEvidence) -> dict[str, Any]:
    result = cell.result
    wall = _finite(
        result.get("evaluation_wall_seconds"),
        f"{cell.identity} evaluation wall seconds",
        positive=True,
    )
    scores = cell.scores
    if (
        result.get("tokens") != scores.tokens
        or result.get("packed_sequences") != scores.packed_sequences
        or result.get("token_identity_sha256")
        != scores.token_identity_sha256
    ):
        raise RoutedLadderV3ReportError(
            f"{cell.identity} result and retained rows differ"
        )
    endpoints = {
        endpoint: {
            **scores.aggregate(endpoint),
            "sst_sha256": scores.endpoint_sst_sha256(endpoint),
        }
        for endpoint in ENDPOINTS
    }
    return {
        "panel": cell.panel,
        "condition": cell.condition,
        "seed": cell.seed,
        "checkpoint_tokens": cell.checkpoint_tokens,
        "origin": cell.origin,
        "training_source_revision": result["training_source_revision"],
        "parent_training_result_sha256": result[
            "parent_training_result_sha256"
        ],
        "parent_training_result_file_sha256": result[
            "parent_training_result_file_sha256"
        ],
        "parent_run_contract_sha256": result[
            "parent_run_contract_sha256"
        ],
        "parent_run_contract_file_sha256": result[
            "parent_run_contract_file_sha256"
        ],
        "parent_checkpoint_pointer_sha256": result[
            "parent_checkpoint_pointer_sha256"
        ],
        "parent_checkpoint_pointer_file_sha256": result[
            "parent_checkpoint_pointer_file_sha256"
        ],
        "result_path": cell.result_path,
        "result_sha256": result["result_sha256"],
        "result_file_sha256": cell.result_file_sha256,
        "score_artifact_sha256": result["score_artifact_sha256"],
        "score_data_sha256": result["score_data_sha256"],
        "score_data_bytes": result["score_data_bytes"],
        "evaluation_tokens": scores.tokens,
        "packed_sequences": scores.packed_sequences,
        "token_identity_sha256": scores.token_identity_sha256,
        "capture_manifest_sha256": scores.capture_manifest_sha256,
        "recomposition_sidecar_sha256": (
            scores.recomposition_sidecar_sha256
        ),
        "endpoints": endpoints,
        "sparsity": _sparsity_row(cell),
        "timing": {
            "cell_evaluation_wall_seconds": wall,
            "cell_evaluation_chip_seconds": (
                wall * cell.bundle_timing.chip_count
            ),
            "cell_timing_scope": (
                "cell_evaluator_wall_time_after_resident_bundle_setup"
            ),
            "bundle_accelerator_type": (
                cell.bundle_timing.accelerator_type
            ),
            "bundle_chip_count": cell.bundle_timing.chip_count,
            "bundle_result_sha256": (
                cell.bundle_timing.bundle_result_sha256
            ),
            "bundle_successful_attempt_wall_seconds": (
                cell.bundle_timing.successful_attempt_wall_seconds
            ),
            "bundle_successful_attempt_chip_seconds": (
                cell.bundle_timing.successful_attempt_chip_seconds
            ),
            "bundle_timing_scope": cell.bundle_timing.timing_scope,
            "allocation_note": (
                "cell chip-seconds are measured cell wall-time times bundle "
                "chip-count; bundle setup is reported only at bundle scope"
            ),
        },
    }


def _contrast_rows(
    cells: Sequence[PrimaryCellEvidence],
) -> list[dict[str, Any]]:
    grid = {cell.identity: cell.scores for cell in cells}
    rows: list[dict[str, Any]] = []
    for panel in PANELS:
        for checkpoint in CHECKPOINT_TOKENS:
            for endpoint in ENDPOINTS:
                pairs = [
                    (
                        grid[
                            (
                                panel,
                                RELU_CONDITION,
                                seed,
                                checkpoint,
                            )
                        ],
                        grid[
                            (
                                panel,
                                SOFTPLUS_CONDITION,
                                seed,
                                checkpoint,
                            )
                        ],
                    )
                    for seed in SEEDS
                ]
                per_seed = [
                    paired_v3_bootstrap(
                        left,
                        right,
                        endpoint=endpoint,
                    )
                    for left, right in pairs
                ]
                hierarchical = hierarchical_v3_bootstrap(
                    pairs,
                    endpoint=endpoint,
                )
                rows.append(
                    {
                        "panel": panel,
                        "checkpoint_tokens": checkpoint,
                        "endpoint": endpoint,
                        "contrast": CONTRAST,
                        "effect": (
                            "EV(A512Softplus) - "
                            "EV(A512ReLUInclusiveTopK)"
                        ),
                        "per_seed": per_seed,
                        "across_seed": hierarchical,
                        "verdict": (
                            "softplus_better"
                            if hierarchical["ci95_lower"] > 0
                            else (
                                "relu_inclusive_topk_better"
                                if hierarchical["ci95_upper"] < 0
                                else "inconclusive"
                            )
                        ),
                    }
                )
    return rows


def _training_cost_report(
    costs: Sequence[TrainingCostEvidence],
) -> dict[str, Any]:
    grid = {row.identity: row for row in costs}
    trajectory_rows = [row.report_row() for row in costs]
    paired: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []
    for panel in PANELS:
        relu_values: list[float] = []
        softplus_values: list[float] = []
        ratios: list[float] = []
        for seed in SEEDS:
            relu = grid[(panel, RELU_CONDITION, seed)]
            softplus = grid[(panel, SOFTPLUS_CONDITION, seed)]
            comparable = (
                relu.accelerator_type == softplus.accelerator_type
            )
            ratio = (
                100.0 * softplus.chip_seconds / relu.chip_seconds
                if comparable
                else None
            )
            relu_values.append(relu.chip_seconds)
            softplus_values.append(softplus.chip_seconds)
            if ratio is not None:
                ratios.append(ratio)
            paired.append(
                {
                    "panel": panel,
                    "seed": seed,
                    "relu_accelerator_type": relu.accelerator_type,
                    "softplus_accelerator_type": (
                        softplus.accelerator_type
                    ),
                    "comparison_status": (
                        "comparable_same_accelerator_type"
                        if comparable
                        else "non_comparable_accelerator_type_mismatch"
                    ),
                    "relu_chip_seconds": relu.chip_seconds,
                    "softplus_chip_seconds": softplus.chip_seconds,
                    "softplus_minus_relu_chip_seconds": (
                        softplus.chip_seconds - relu.chip_seconds
                        if comparable
                        else None
                    ),
                    "softplus_to_relu_chip_seconds_ratio_percent": ratio,
                    "relu_to_softplus_chip_seconds_ratio_percent": (
                        100.0 * relu.chip_seconds / softplus.chip_seconds
                        if comparable
                        else None
                    ),
                    "comparison_scope": relu.chip_seconds_scope,
                    "comparison_excludes": list(
                        relu.chip_seconds_excludes
                    ),
                }
            )
        all_comparable = len(ratios) == len(SEEDS)
        panels.append(
            {
                "panel": panel,
                "comparison_status": (
                    "comparable_all_seeds_same_accelerator_type"
                    if all_comparable
                    else (
                        "non_comparable_one_or_more_accelerator_type_"
                        "mismatches"
                    )
                ),
                "comparable_seed_count": len(ratios),
                "relu_median_chip_seconds": float(
                    np.median(np.asarray(relu_values, dtype=np.float64))
                ),
                "softplus_median_chip_seconds": float(
                    np.median(np.asarray(softplus_values, dtype=np.float64))
                ),
                "median_paired_softplus_to_relu_ratio_percent": float(
                    np.median(np.asarray(ratios, dtype=np.float64))
                )
                if all_comparable
                else None,
                "relu_total_chip_seconds": math.fsum(relu_values),
                "softplus_total_chip_seconds": math.fsum(softplus_values),
                "seeds": list(SEEDS),
            }
        )
    return {
        "authority": (
            "immutable marginal SAE/transcoder training-result chip-seconds"
        ),
        "ratio_policy": "same_accelerator_type_only",
        "shared_harvest_excluded": True,
        "trajectory_costs": trajectory_rows,
        "paired_seed_comparisons": paired,
        "panel_summaries": panels,
    }


def _primary_verdict(
    contrasts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = [
        row
        for row in contrasts
        if row["checkpoint_tokens"] == 10_000_000
        and row["endpoint"] == PRIMARY_ENDPOINT
    ]
    if len(selected) != len(PANELS):
        raise RoutedLadderV3ReportError(
            "final primary component verdict is incomplete"
        )
    return {
        "checkpoint_tokens": 10_000_000,
        "endpoint": PRIMARY_ENDPOINT,
        "panels": [
            {
                "panel": row["panel"],
                "estimate_mean_ev_difference": row["across_seed"][
                    "estimate_mean_ev_difference"
                ],
                "ci95_lower": row["across_seed"]["ci95_lower"],
                "ci95_upper": row["across_seed"]["ci95_upper"],
                "positive_seed_count": row["across_seed"][
                    "positive_seed_count"
                ],
                "verdict": row["verdict"],
            }
            for row in selected
        ],
        "claim_boundary": (
            "activation-policy effect at matched geometry and maximum "
            "selected slots; exact positive L0 is measured, not fixed"
        ),
    }


def collect_a256_reference_context(
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate A256 lineage without admitting A256 quality score rows."""

    if manifest is None:
        return {
            "status": "absent_nonblocking",
            "role": "panel_native_historical_scale_context",
            "reason": (
                "optional context is withheld pending an explicit "
                "historical-config provenance addendum"
            ),
            "primary_verdict_blocked": False,
            "checkpoint_reference_count": 0,
            "a256_scores_in_primary_inference": False,
        }
    try:
        selected, panels = parse_a256_provenance_addendum(manifest)
        for panel in panels:
            validate_a256_addendum_panel(panel)
    except (
        OSError,
        RoutedLadderCampaignPlanError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise RoutedLadderV3ReportError(
            "optional A256 provenance addendum is invalid"
        ) from error
    panel_references: list[dict[str, Any]] = []
    checkpoint_count = 0
    for panel in panels:
        checkpoints = [
            {
                "checkpoint_tokens": checkpoint.checkpoint_tokens,
                "checkpoint_pointer_sha256": (
                    checkpoint.checkpoint_pointer_sha256
                ),
                "checkpoint_pointer_file_sha256": (
                    checkpoint.checkpoint_pointer.sha256
                ),
            }
            for checkpoint in panel.checkpoints
        ]
        checkpoint_count += len(checkpoints)
        panel_references.append(
            {
                "panel": panel.panel,
                "source_label": panel.source_label,
                "seed": panel.seed,
                "encoder_activation": panel.encoder_activation,
                "selection_policy": panel.selection_policy,
                "declared_source_config_semantic_sha256": (
                    panel.declared_source_config_semantic_sha256
                ),
                "observed_source_config_semantic_sha256": (
                    panel.observed_source_config_semantic_sha256
                ),
                "observed_source_config_file_sha256": (
                    panel.observed_source_config_file_sha256
                ),
                "declaration_status": panel.declaration_status,
                "disposition": panel.disposition,
                "training_source_revision": panel.training_source_revision,
                "run_contract_sha256": panel.run_contract_sha256,
                "run_contract_file_sha256": panel.run_contract.sha256,
                "training_result_sha256": panel.training_result_sha256,
                "training_result_file_sha256": (
                    panel.training_result.sha256
                ),
                "cost": {
                    "resource_id": panel.cost.resource_id,
                    "accelerator_type": panel.cost.accelerator_type,
                    "chip_count": panel.cost.chip_count,
                    "marginal_wall_seconds": (
                        panel.cost.marginal_wall_seconds
                    ),
                    "chip_seconds": panel.cost.chip_seconds,
                    "chip_seconds_scope": panel.cost.chip_seconds_scope,
                    "chip_seconds_excludes": list(
                        panel.cost.chip_seconds_excludes
                    ),
                },
                "checkpoint_references": checkpoints,
            }
        )
    return {
        "status": "available_read_only_checkpoint_context",
        "role": "panel_native_historical_scale_context",
        "manifest_sha256": selected["manifest_sha256"],
        "source_plan_sha256": selected["source_plan_sha256"],
        "scientific_scope": copy.deepcopy(selected["scientific_scope"]),
        "panel_reference_count": len(panel_references),
        "checkpoint_reference_count": checkpoint_count,
        "panel_references": panel_references,
        "a256_scores_in_primary_inference": False,
        "primary_verdict_blocked": False,
        "statement": _A256_CONTEXT_STATEMENT,
    }


def _validate_a256_report_context(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise RoutedLadderV3ReportError(
            "optional A256 context must be a mapping"
        )
    status = value.get("status")
    if status == "absent_nonblocking":
        if (
            set(value)
            != {
                "status",
                "role",
                "reason",
                "primary_verdict_blocked",
                "checkpoint_reference_count",
                "a256_scores_in_primary_inference",
            }
            or value.get("role")
            != "panel_native_historical_scale_context"
            or value.get("primary_verdict_blocked") is not False
            or value.get("checkpoint_reference_count") != 0
            or value.get("a256_scores_in_primary_inference") is not False
            or "addendum" not in str(value.get("reason"))
        ):
            raise RoutedLadderV3ReportError(
                "absent A256 context metadata changed"
            )
        return
    expected_fields = {
        "status",
        "role",
        "manifest_sha256",
        "source_plan_sha256",
        "scientific_scope",
        "panel_reference_count",
        "checkpoint_reference_count",
        "panel_references",
        "a256_scores_in_primary_inference",
        "primary_verdict_blocked",
        "statement",
    }
    if (
        status != "available_read_only_checkpoint_context"
        or set(value) != expected_fields
        or value.get("role")
        != "panel_native_historical_scale_context"
        or value.get("scientific_scope") != _A256_CONTEXT_SCOPE
        or value.get("panel_reference_count") != 2
        or value.get("checkpoint_reference_count") != 6
        or value.get("a256_scores_in_primary_inference") is not False
        or value.get("primary_verdict_blocked") is not False
        or value.get("statement") != _A256_CONTEXT_STATEMENT
    ):
        raise RoutedLadderV3ReportError(
            "available A256 context identity or scope changed"
        )
    _sha256(value.get("manifest_sha256"), "A256 addendum manifest hash")
    _sha256(value.get("source_plan_sha256"), "A256 source plan hash")
    references = value.get("panel_references")
    if not isinstance(references, list) or len(references) != 2:
        raise RoutedLadderV3ReportError(
            "available A256 context requires exactly two panels"
        )
    expected_panel_fields = {
        "panel",
        "source_label",
        "seed",
        "encoder_activation",
        "selection_policy",
        "declared_source_config_semantic_sha256",
        "observed_source_config_semantic_sha256",
        "observed_source_config_file_sha256",
        "declaration_status",
        "disposition",
        "training_source_revision",
        "run_contract_sha256",
        "run_contract_file_sha256",
        "training_result_sha256",
        "training_result_file_sha256",
        "cost",
        "checkpoint_references",
    }
    expected_identity = {
        "sae": {
            "source_label": A256_NATIVE_LABEL["sae"],
            "encoder_activation": A256_NATIVE_ACTIVATION["sae"],
            "selection_policy": A256_NATIVE_SELECTION_POLICY["sae"],
            "declaration_status": "exact_match",
            "disposition": "contextual_reference_verified",
        },
        "transcoder": {
            "source_label": A256_NATIVE_LABEL["transcoder"],
            "encoder_activation": A256_NATIVE_ACTIVATION["transcoder"],
            "selection_policy": A256_NATIVE_SELECTION_POLICY["transcoder"],
            "declaration_status": "mismatch",
            "disposition": (
                "contextual_reference_corrected_by_observed_lineage"
            ),
        },
    }
    observed_panels: set[str] = set()
    revisions: set[str] = set()
    checkpoint_identities: set[tuple[str, int]] = set()
    for index, row in enumerate(references):
        if not isinstance(row, Mapping) or set(row) != expected_panel_fields:
            raise RoutedLadderV3ReportError(
                f"A256 panel reference {index} fields changed"
            )
        panel = row.get("panel")
        if panel not in PANELS or panel in observed_panels:
            raise RoutedLadderV3ReportError(
                f"A256 panel reference {index} identity changed"
            )
        observed_panels.add(str(panel))
        expected = expected_identity[str(panel)]
        if (
            row.get("seed") != 0
            or any(
                row.get(name) != expected_value
                for name, expected_value in expected.items()
            )
            or row.get("declared_source_config_semantic_sha256")
            != A256_NATIVE_CONFIG_SHA256[str(panel)]
        ):
            raise RoutedLadderV3ReportError(
                f"A256 panel reference {panel} treatment changed"
            )
        declared = _sha256(
            row.get("declared_source_config_semantic_sha256"),
            f"A256 {panel} declared config hash",
        )
        observed = _sha256(
            row.get("observed_source_config_semantic_sha256"),
            f"A256 {panel} observed config hash",
        )
        _sha256(
            row.get("observed_source_config_file_sha256"),
            f"A256 {panel} config file hash",
        )
        if (
            (panel == "sae" and observed != declared)
            or (panel == "transcoder" and observed == declared)
        ):
            raise RoutedLadderV3ReportError(
                f"A256 {panel} declaration status is dishonest"
            )
        for name in (
            "run_contract_sha256",
            "run_contract_file_sha256",
            "training_result_sha256",
            "training_result_file_sha256",
        ):
            _sha256(row.get(name), f"A256 {panel} {name}")
        revision = row.get("training_source_revision")
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise RoutedLadderV3ReportError(
                f"A256 {panel} training revision is invalid"
            )
        revisions.add(revision)
        cost = row.get("cost")
        expected_cost_fields = {
            "resource_id",
            "accelerator_type",
            "chip_count",
            "marginal_wall_seconds",
            "chip_seconds",
            "chip_seconds_scope",
            "chip_seconds_excludes",
        }
        if (
            not isinstance(cost, Mapping)
            or set(cost) != expected_cost_fields
            or not isinstance(cost.get("resource_id"), str)
            or not cost.get("resource_id")
            or not isinstance(cost.get("accelerator_type"), str)
            or not cost.get("accelerator_type")
            or not isinstance(cost.get("chip_seconds_scope"), str)
            or not cost.get("chip_seconds_scope")
            or not isinstance(cost.get("chip_seconds_excludes"), list)
            or not cost.get("chip_seconds_excludes")
            or not all(
                isinstance(item, str) and item
                for item in cost.get("chip_seconds_excludes", [])
            )
        ):
            raise RoutedLadderV3ReportError(
                f"A256 {panel} measured cost metadata changed"
            )
        chip_count = _positive_int(
            cost.get("chip_count"),
            f"A256 {panel} chip count",
        )
        wall = _finite(
            cost.get("marginal_wall_seconds"),
            f"A256 {panel} marginal wall seconds",
            positive=True,
        )
        chip_seconds = _finite(
            cost.get("chip_seconds"),
            f"A256 {panel} chip-seconds",
            positive=True,
        )
        if (
            _CHIP_COUNT_BY_ACCELERATOR.get(
                str(cost["accelerator_type"])
            )
            != chip_count
            or not math.isclose(
                chip_seconds,
                chip_count * wall,
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
        ):
            raise RoutedLadderV3ReportError(
                f"A256 {panel} chip-seconds arithmetic changed"
            )
        checkpoints = row.get("checkpoint_references")
        if not isinstance(checkpoints, list) or len(checkpoints) != 3:
            raise RoutedLadderV3ReportError(
                f"A256 {panel} checkpoint inventory changed"
            )
        for checkpoint in checkpoints:
            if (
                not isinstance(checkpoint, Mapping)
                or set(checkpoint)
                != {
                    "checkpoint_tokens",
                    "checkpoint_pointer_sha256",
                    "checkpoint_pointer_file_sha256",
                }
                or checkpoint.get("checkpoint_tokens")
                not in CHECKPOINT_TOKENS
            ):
                raise RoutedLadderV3ReportError(
                    f"A256 {panel} checkpoint reference changed"
                )
            identity = (str(panel), int(checkpoint["checkpoint_tokens"]))
            if identity in checkpoint_identities:
                raise RoutedLadderV3ReportError(
                    f"A256 {panel} checkpoint is duplicated"
                )
            checkpoint_identities.add(identity)
            _sha256(
                checkpoint.get("checkpoint_pointer_sha256"),
                f"A256 {panel} checkpoint pointer hash",
            )
            _sha256(
                checkpoint.get("checkpoint_pointer_file_sha256"),
                f"A256 {panel} checkpoint pointer file hash",
            )
    expected_checkpoints = {
        (panel, checkpoint)
        for panel in PANELS
        for checkpoint in CHECKPOINT_TOKENS
    }
    if (
        observed_panels != set(PANELS)
        or len(revisions) != 1
        or checkpoint_identities != expected_checkpoints
    ):
        raise RoutedLadderV3ReportError(
            "available A256 panel/checkpoint grid changed"
        )


def collect_h_reference_context(
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate optional H checkpoints without admitting H score rows."""

    if manifest is None:
        return {
            "status": "absent_nonblocking",
            "role": "read_only_descriptive_context",
            "reason": (
                "optional context is withheld pending the same explicit "
                "v3 provenance addendum and is not a randomized arm"
            ),
            "primary_verdict_blocked": False,
            "checkpoint_reference_count": 0,
            "h_scores_in_primary_inference": False,
        }
    try:
        selected, references = parse_h_reference_manifest(manifest)
    except (
        RoutedLadderCampaignPlanError,
        TypeError,
        ValueError,
    ) as error:
        raise RoutedLadderV3ReportError(
            "optional H reference manifest is invalid"
        ) from error
    if len(references) != 18:
        raise RoutedLadderV3ReportError(
            "optional H manifest must contain exactly 18 checkpoint references"
        )
    identities: list[dict[str, Any]] = []
    for cell in references:
        try:
            result_payload = cell.training_result.controller_path.read_bytes()
            pointer_payload = cell.checkpoint_pointer.controller_path.read_bytes()
            _validate_h_checkpoint_lineage(cell)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise RoutedLadderV3ReportError(
                f"optional H lineage changed: {cell.identity}"
            ) from error
        if (
            hashlib.sha256(result_payload).hexdigest()
            != cell.training_result.sha256
            or hashlib.sha256(pointer_payload).hexdigest()
            != cell.checkpoint_pointer.sha256
        ):
            raise RoutedLadderV3ReportError(
                f"optional H file pin changed: {cell.identity}"
            )
        identities.append(
            {
                "panel": cell.panel,
                "seed": cell.seed,
                "checkpoint_tokens": cell.checkpoint_tokens,
                "training_source_revision": cell.training_source_revision,
                "h_config_sha256": cell.h_config_sha256,
                "training_result_sha256": cell.training_result_sha256,
                "training_result_file_sha256": (
                    cell.training_result.sha256
                ),
                "checkpoint_pointer_sha256": (
                    cell.checkpoint_pointer_sha256
                ),
                "checkpoint_pointer_file_sha256": (
                    cell.checkpoint_pointer.sha256
                ),
            }
        )
    return {
        "status": "available_read_only_checkpoint_context",
        "role": "read_only_descriptive_context",
        "manifest_sha256": selected["manifest_sha256"],
        "checkpoint_reference_count": len(references),
        "checkpoint_references": identities,
        "h_scores_in_primary_inference": False,
        "primary_verdict_blocked": False,
        "statement": (
            "Eighteen hash-pinned H checkpoint references exist; no H score "
            "row is mixed into the v3 A512 ReLU-versus-Softplus inference."
        ),
    }


def build_v3_primary_report_from_evidence(
    evaluation_plan: Mapping[str, Any],
    cells: Sequence[PrimaryCellEvidence],
    bundle_timings: Sequence[BundleTimingEvidence],
    training_costs: Sequence[TrainingCostEvidence],
    *,
    a256_provenance_addendum: Mapping[str, Any] | None = None,
    h_reference_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the deterministic scientific report from already loaded evidence."""

    plan = validate_v3_primary_evaluation_plan(evaluation_plan)
    selected_cells = tuple(cells)
    selected_timings = tuple(bundle_timings)
    selected_costs = tuple(training_costs)
    expected_cells = {
        (panel, condition, seed, checkpoint)
        for panel in PANELS
        for condition in CONDITIONS
        for seed in SEEDS
        for checkpoint in CHECKPOINT_TOKENS
    }
    expected_bundles = {
        (panel, seed) for panel in PANELS for seed in SEEDS
    }
    expected_costs = {
        (panel, condition, seed)
        for panel in PANELS
        for condition in CONDITIONS
        for seed in SEEDS
    }
    if (
        len(selected_cells) != CELL_COUNT
        or {row.identity for row in selected_cells} != expected_cells
        or len(selected_timings) != len(expected_bundles)
        or {(row.panel, row.seed) for row in selected_timings}
        != expected_bundles
        or len(selected_costs) != len(expected_costs)
        or {row.identity for row in selected_costs} != expected_costs
    ):
        raise RoutedLadderV3ReportError(
            "report evidence matrix is incomplete or duplicated"
        )
    ordered_cells = tuple(
        sorted(
            selected_cells,
            key=lambda row: (
                PANELS.index(row.panel),
                row.checkpoint_tokens,
                row.seed,
                CONDITIONS.index(row.condition),
            ),
        )
    )
    ordered_timings = tuple(
        sorted(
            selected_timings,
            key=lambda row: (PANELS.index(row.panel), row.seed),
        )
    )
    ordered_costs = tuple(
        sorted(
            selected_costs,
            key=lambda row: (
                PANELS.index(row.panel),
                row.seed,
                CONDITIONS.index(row.condition),
            ),
        )
    )
    common = _validate_common_evaluation(ordered_cells)
    cell_rows = [_cell_report_row(row) for row in ordered_cells]
    contrast_rows = _contrast_rows(ordered_cells)
    training = _training_cost_report(ordered_costs)
    core = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": REPORT_STATUS,
        "source_provenance": {
            "evaluation_plan_sha256": plan["plan_sha256"],
            "training_plan_sha256": plan["training_plan_sha256"],
            "softplus_reference_manifest_sha256": plan[
                "completed_trajectory_manifest_sha256"
            ],
            "v2_source_plan_sha256": plan["v2_source_plan_sha256"],
            "evaluation_source_revision": plan[
                "evaluation_source_revision"
            ],
            "training_source_revision": plan[
                "training_source_revision"
            ],
            "campaign_id": plan["campaign_id"],
        },
        "primary_estimand": {
            "label": PRIMARY_CONTRAST_LABEL,
            "estimand": PRIMARY_ESTIMAND,
            "effect": (
                "EV(A512Softplus) - EV(A512ReLUInclusiveTopK)"
            ),
            "primary_endpoint": PRIMARY_ENDPOINT,
            "paired_seeds": list(SEEDS),
            "checkpoints": list(CHECKPOINT_TOKENS),
            "matched_quantities": {
                "bank_width": True,
                "trainable_parameter_count": True,
                "candidate_count": True,
                "maximum_selected_slot_count": True,
                "exact_positive_l0": False,
            },
            "interval_contract": {
                "confidence_level": CONFIDENCE_LEVEL,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "per_seed_algorithm": SEED_BOOTSTRAP_ALGORITHM,
                "across_seed_algorithm": (
                    HIERARCHICAL_BOOTSTRAP_ALGORITHM
                ),
                "across_seed_n": len(SEEDS),
            },
        },
        "common_evaluation_identity": common,
        "cells": cell_rows,
        "bundle_timings": [row.report_row() for row in ordered_timings],
        "contrasts": contrast_rows,
        "primary_verdict": _primary_verdict(contrast_rows),
        "training_cost_comparison": training,
        "optional_context": {
            "A256Native": collect_a256_reference_context(
                a256_provenance_addendum
            ),
            "H": collect_h_reference_context(h_reference_manifest),
        },
        "interpretation": {
            "authorized_claim": (
                "paired activation-policy effect for Softplus versus genuine "
                "inclusive-zero ReLU at matched A512 geometry and maximum "
                "selected slots"
            ),
            "exact_positive_l0_is_fixed": False,
            "multiplicity_adjustment": None,
            "context_is_causal": False,
            "notes": [
                (
                    "All 36 cells use exactly the same held-out token rows "
                    "and target-only SST rows."
                ),
                (
                    "Per-seed intervals resample packed sequences; the n=3 "
                    "verdict resamples seeds and one common packed-sequence "
                    "draw as crossed factors."
                ),
                (
                    "Training chip-seconds include preprocessing and "
                    "synchronized optimizer work and retain each immutable "
                    "result's explicit exclusions."
                ),
            ],
        },
        "markdown_schema_version": MARKDOWN_SCHEMA_VERSION,
    }
    markdown_payload = _render_v3_primary_markdown_core(core).encode("utf-8")
    body = {
        **core,
        "markdown_sha256": hashlib.sha256(markdown_payload).hexdigest(),
    }
    report = {**body, "report_sha256": canonical_sha256(body)}
    validate_v3_primary_report(report)
    return report


def _render_v3_primary_markdown_core(value: Mapping[str, Any]) -> str:
    """Render Markdown without depending on either output file's own hash."""

    provenance = value["source_provenance"]
    estimand = value["primary_estimand"]
    verdict = value["primary_verdict"]
    lines = [
        "# Experiment 5 v3 primary activation-policy verdict",
        "",
        f"Protocol: `{value['protocol_id']}`",
        "",
        f"Status: `{value['status']}`",
        "",
        "## Estimand",
        "",
        f"`{estimand['label']}`",
        "",
        estimand["estimand"].replace("_", " "),
        "",
        (
            "Effect: `EV(A512Softplus) - "
            "EV(A512ReLUInclusiveTopK)`."
        ),
        "",
        (
            "Bank width, trainable parameter count, candidate count, and "
            "maximum selected slots are fixed. Exact positive L0 is not "
            "fixed; it is measured."
        ),
        "",
        "## Final 10M component verdict",
        "",
        "| panel | mean EV delta | paired hierarchical 95% interval | "
        "positive seeds | verdict |",
        "|---|---:|---:|---:|---|",
    ]
    contrast_index = {
        (
            row["panel"],
            row["checkpoint_tokens"],
            row["endpoint"],
        ): row
        for row in value["contrasts"]
    }
    for panel_row in verdict["panels"]:
        row = contrast_index[(panel_row["panel"], 10_000_000, PRIMARY_ENDPOINT)]
        across = row["across_seed"]
        lines.append(
            f"| {panel_row['panel']} | "
            f"{across['estimate_mean_ev_difference']:.8f} | "
            f"[{across['ci95_lower']:.8f}, "
            f"{across['ci95_upper']:.8f}] | "
            f"{across['positive_seed_count']}/3 | "
            f"{row['verdict']} |"
        )
    lines.extend(
        [
            "",
            (
                "Intervals use 20,000 deterministic crossed seed-by-packed-"
                "sequence bootstrap replicates. No multiplicity adjustment "
                "is applied."
            ),
            "",
            "## Learning-curve contrasts",
            "",
            "| panel | checkpoint | endpoint | mean EV delta | 95% interval "
            "| positive seeds | verdict |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in value["contrasts"]:
        across = row["across_seed"]
        lines.append(
            f"| {row['panel']} | "
            f"{int(row['checkpoint_tokens']) // 1_000_000}M | "
            f"{row['endpoint']} | "
            f"{across['estimate_mean_ev_difference']:.8f} | "
            f"[{across['ci95_lower']:.8f}, "
            f"{across['ci95_upper']:.8f}] | "
            f"{across['positive_seed_count']}/3 | "
            f"{row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Per-cell measurements",
            "",
            "| panel | condition | seed | checkpoint | component EV | "
            "component NMSE | positive L0 | zero selected | dead rate | "
            "eval wall s |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in value["cells"]:
        component = row["endpoints"]["component"]
        sparsity = row["sparsity"]
        timing = row["timing"]
        lines.append(
            f"| {row['panel']} | {row['condition']} | {row['seed']} | "
            f"{int(row['checkpoint_tokens']) // 1_000_000}M | "
            f"{component['ev']:.8f} | {component['nmse']:.8f} | "
            f"{sparsity['positive_l0']:.6f} | "
            f"{sparsity['zero_selected_slot_fraction']:.6f} | "
            f"{sparsity['dead_rate']:.6f} | "
            f"{timing['cell_evaluation_wall_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The JSON artifact retains SSE, SST, all three endpoints, exact "
            "SST hashes, selected-slot counts, positive-L0 sums, cell and "
            "bundle chip-seconds, and every immutable result pin.",
            "",
            "## Marginal training cost",
            "",
            "| panel | seed | ReLU hardware | Softplus hardware | "
            "ReLU chip-s | Softplus chip-s | Softplus/ReLU | status |",
            "|---|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for row in value["training_cost_comparison"][
        "paired_seed_comparisons"
    ]:
        ratio = row["softplus_to_relu_chip_seconds_ratio_percent"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.2f}%"
        lines.append(
            f"| {row['panel']} | {row['seed']} | "
            f"{row['relu_accelerator_type']} | "
            f"{row['softplus_accelerator_type']} | "
            f"{row['relu_chip_seconds']:.3f} | "
            f"{row['softplus_chip_seconds']:.3f} | "
            f"{ratio_text} | {row['comparison_status']} |"
        )
    a256_context = value["optional_context"]["A256Native"]
    a256_sentence = (
        "Two panel-native historical A256 lineages and six checkpoint "
        "pointers are attached as read-only context. The SAE source is "
        "`A256ReLU`; the transcoder source is `A256Softplus`, whose frozen "
        "v3 declaration differs from its observed historical config. No "
        "A256 score row enters the v3 A512 contrast."
        if a256_context["status"]
        == "available_read_only_checkpoint_context"
        else (
            "`A256Native` is absent and nonblocking because its historical "
            "config provenance requires an explicit addendum."
        )
    )
    h_context = value["optional_context"]["H"]
    h_sentence = (
        "Eighteen hash-pinned H checkpoint references are attached as "
        "read-only context. No H score row enters the v3 A512 contrast."
        if h_context["status"]
        == "available_read_only_checkpoint_context"
        else (
            "H is absent and nonblocking pending the explicit provenance "
            "addendum. It is not required for the paired A512 verdict."
        )
    )
    lines.extend(
        [
            "",
            (
                "These are marginal training chip-seconds from immutable "
                "training results. Shared harvesting, compilation, lowering, "
                "checkpoint serialization, and evaluation are not folded "
                "into this comparison. Ratios are emitted only when the "
                "paired trajectories used the same accelerator type."
            ),
            "",
            "## Optional context",
            "",
            a256_sentence,
            "",
            h_sentence,
            "",
            "## Provenance",
            "",
            f"- Evaluation plan: `{provenance['evaluation_plan_sha256']}`",
            f"- Training plan: `{provenance['training_plan_sha256']}`",
            (
                "- Softplus reference manifest: "
                f"`{provenance['softplus_reference_manifest_sha256']}`"
            ),
            f"- Evaluation source: `{provenance['evaluation_source_revision']}`",
            f"- Training source: `{provenance['training_source_revision']}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_v3_primary_report_markdown(
    value: Mapping[str, Any],
) -> str:
    report = validate_v3_primary_report(value)
    rendered = _render_v3_primary_markdown_core(report)
    if (
        hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        != report["markdown_sha256"]
    ):
        raise RoutedLadderV3ReportError(
            "deterministic Markdown differs from its report hash"
        )
    return rendered


def _validate_score_summary(
    value: Any,
    *,
    name: str,
    expected_sst: float,
    expected_sst_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "sse",
        "sst",
        "nmse",
        "ev",
        "sst_sha256",
    }:
        raise RoutedLadderV3ReportError(f"{name} fields changed")
    sse = _finite(value["sse"], f"{name}.sse")
    sst = _finite(value["sst"], f"{name}.sst", positive=True)
    nmse = _finite(value["nmse"], f"{name}.nmse")
    ev = _finite(value["ev"], f"{name}.ev")
    if (
        sse < 0
        or not math.isclose(sst, expected_sst, rel_tol=0, abs_tol=1e-9)
        or value["sst_sha256"] != expected_sst_sha256
        or not math.isclose(nmse, sse / sst, rel_tol=1e-12, abs_tol=1e-12)
        or not math.isclose(ev, 1.0 - nmse, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise RoutedLadderV3ReportError(f"{name} arithmetic changed")
    return dict(value)


def validate_v3_primary_report(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate self-contained integrity and recomputable arithmetic only.

    Confidence intervals cannot be authenticated without retained score
    evidence.  Scientific verification must use
    :func:`verify_v3_primary_report_from_paths`.
    """

    if not isinstance(value, Mapping):
        raise RoutedLadderV3ReportError("v3 report must be a mapping")
    selected = copy.deepcopy(dict(value))
    expected_fields = {
        "schema_version",
        "protocol_id",
        "status",
        "source_provenance",
        "primary_estimand",
        "common_evaluation_identity",
        "cells",
        "bundle_timings",
        "contrasts",
        "primary_verdict",
        "training_cost_comparison",
        "optional_context",
        "interpretation",
        "markdown_schema_version",
        "markdown_sha256",
        "report_sha256",
    }
    if set(selected) != expected_fields:
        raise RoutedLadderV3ReportError("v3 report fields changed")
    claimed = selected.pop("report_sha256")
    if (
        selected["schema_version"] != REPORT_SCHEMA_VERSION
        or selected["protocol_id"] != PROTOCOL_ID
        or selected["status"] != REPORT_STATUS
        or selected["markdown_schema_version"] != MARKDOWN_SCHEMA_VERSION
        or _sha256(claimed, "report_sha256")
        != canonical_sha256(selected)
    ):
        raise RoutedLadderV3ReportError(
            "v3 report identity or self-hash changed"
        )
    report = {**selected, "report_sha256": claimed}
    provenance = report["source_provenance"]
    if not isinstance(provenance, Mapping):
        raise RoutedLadderV3ReportError("source provenance is invalid")
    for name in (
        "evaluation_plan_sha256",
        "training_plan_sha256",
        "softplus_reference_manifest_sha256",
        "v2_source_plan_sha256",
    ):
        _sha256(provenance.get(name), f"source_provenance.{name}")
    for name in ("evaluation_source_revision", "training_source_revision"):
        revision = provenance.get(name)
        if not isinstance(revision, str) or len(revision) != 40:
            raise RoutedLadderV3ReportError(
                f"source_provenance.{name} is invalid"
            )
    estimand = report["primary_estimand"]
    if (
        not isinstance(estimand, Mapping)
        or estimand.get("label") != PRIMARY_CONTRAST_LABEL
        or estimand.get("estimand") != PRIMARY_ESTIMAND
        or estimand.get("primary_endpoint") != PRIMARY_ENDPOINT
        or estimand.get("paired_seeds") != list(SEEDS)
        or estimand.get("checkpoints") != list(CHECKPOINT_TOKENS)
        or estimand.get("matched_quantities")
        != {
            "bank_width": True,
            "trainable_parameter_count": True,
            "candidate_count": True,
            "maximum_selected_slot_count": True,
            "exact_positive_l0": False,
        }
        or estimand.get("interval_contract")
        != {
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "per_seed_algorithm": SEED_BOOTSTRAP_ALGORITHM,
            "across_seed_algorithm": HIERARCHICAL_BOOTSTRAP_ALGORITHM,
            "across_seed_n": 3,
        }
    ):
        raise RoutedLadderV3ReportError(
            "primary estimand or fixed-L0 claim changed"
        )
    common = report["common_evaluation_identity"]
    if not isinstance(common, Mapping):
        raise RoutedLadderV3ReportError(
            "common evaluation identity is invalid"
        )
    tokens = _positive_int(
        common.get("evaluation_tokens"),
        "common evaluation tokens",
    )
    packed = _positive_int(
        common.get("packed_sequences"),
        "common packed sequences",
    )
    if packed > tokens:
        raise RoutedLadderV3ReportError(
            "common packed sequence count exceeds tokens"
        )
    for name in (
        "token_identity_sha256",
        "capture_manifest_sha256",
        "recomposition_sidecar_sha256",
    ):
        _sha256(common.get(name), f"common_evaluation_identity.{name}")
    common_endpoints = common.get("endpoint_sst")
    if (
        not isinstance(common_endpoints, Mapping)
        or set(common_endpoints) != set(ENDPOINTS)
    ):
        raise RoutedLadderV3ReportError("common endpoint SST inventory changed")
    for endpoint in ENDPOINTS:
        row = common_endpoints[endpoint]
        if not isinstance(row, Mapping):
            raise RoutedLadderV3ReportError(
                f"common {endpoint} SST is invalid"
            )
        _finite(row.get("sst"), f"common {endpoint} SST", positive=True)
        _sha256(row.get("sst_sha256"), f"common {endpoint} SST hash")

    raw_cells = report["cells"]
    if not isinstance(raw_cells, list) or len(raw_cells) != CELL_COUNT:
        raise RoutedLadderV3ReportError("report does not contain 36 cells")
    cell_grid: dict[tuple[str, str, int, int], Mapping[str, Any]] = {}
    expected_cell_fields = {
        "panel",
        "condition",
        "seed",
        "checkpoint_tokens",
        "origin",
        "training_source_revision",
        "parent_training_result_sha256",
        "parent_training_result_file_sha256",
        "parent_run_contract_sha256",
        "parent_run_contract_file_sha256",
        "parent_checkpoint_pointer_sha256",
        "parent_checkpoint_pointer_file_sha256",
        "result_path",
        "result_sha256",
        "result_file_sha256",
        "score_artifact_sha256",
        "score_data_sha256",
        "score_data_bytes",
        "evaluation_tokens",
        "packed_sequences",
        "token_identity_sha256",
        "capture_manifest_sha256",
        "recomposition_sidecar_sha256",
        "endpoints",
        "sparsity",
        "timing",
    }
    for index, row in enumerate(raw_cells):
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_cell_fields
        ):
            raise RoutedLadderV3ReportError(f"cells[{index}] is invalid")
        identity = (
            row.get("panel"),
            row.get("condition"),
            row.get("seed"),
            row.get("checkpoint_tokens"),
        )
        if (
            identity[0] not in PANELS
            or identity[1] not in CONDITIONS
            or identity[2] not in SEEDS
            or identity[3] not in CHECKPOINT_TOKENS
            or identity in cell_grid
        ):
            raise RoutedLadderV3ReportError(
                f"cell identity is invalid or duplicated: {identity}"
            )
        cell_grid[identity] = row
        expected_origin = (
            "fresh_v3"
            if identity[1] == RELU_CONDITION
            else "imported_v2_exact_graph"
        )
        training_revision = row.get("training_source_revision")
        _positive_int(
            row.get("score_data_bytes"),
            f"{identity} score data bytes",
        )
        if (
            row.get("origin") != expected_origin
            or not isinstance(training_revision, str)
            or len(training_revision) != 40
            or any(
                character not in "0123456789abcdef"
                for character in training_revision
            )
            or not isinstance(row.get("result_path"), str)
            or not row.get("result_path")
            or row.get("evaluation_tokens") != tokens
            or row.get("packed_sequences") != packed
            or row.get("token_identity_sha256")
            != common["token_identity_sha256"]
            or row.get("capture_manifest_sha256")
            != common["capture_manifest_sha256"]
            or row.get("recomposition_sidecar_sha256")
            != common["recomposition_sidecar_sha256"]
        ):
            raise RoutedLadderV3ReportError(
                f"cell common evaluation identity changed: {identity}"
            )
        endpoints = row.get("endpoints")
        if not isinstance(endpoints, Mapping) or set(endpoints) != set(
            ENDPOINTS
        ):
            raise RoutedLadderV3ReportError(
                f"cell endpoint inventory changed: {identity}"
            )
        for endpoint in ENDPOINTS:
            _validate_score_summary(
                endpoints[endpoint],
                name=f"{identity}.{endpoint}",
                expected_sst=float(common_endpoints[endpoint]["sst"]),
                expected_sst_sha256=str(
                    common_endpoints[endpoint]["sst_sha256"]
                ),
            )
        sparsity = row.get("sparsity")
        if (
            not isinstance(sparsity, Mapping)
            or set(sparsity)
            != {
                "selected_slot_count",
                "positive_l0_sum",
                "positive_l0",
                "zero_selected_slot_fraction",
                "positive_l0_contract",
                "dead_rate",
                "dead_rate_scope",
                "total_features",
            }
        ):
            raise RoutedLadderV3ReportError(
                f"cell sparsity is invalid: {identity}"
            )
        selected_slots = _positive_int(
            sparsity.get("selected_slot_count"),
            f"{identity} selected slots",
        )
        positive_sum = sparsity.get("positive_l0_sum")
        positive_l0 = _finite(
            sparsity.get("positive_l0"),
            f"{identity} positive L0",
        )
        zero_fraction = _finite(
            sparsity.get("zero_selected_slot_fraction"),
            f"{identity} zero selected fraction",
        )
        dead_rate = _finite(
            sparsity.get("dead_rate"),
            f"{identity} held-out dead rate",
        )
        if (
            selected_slots != 32
            or isinstance(positive_sum, bool)
            or not isinstance(positive_sum, int)
            or not 0 <= positive_sum <= tokens * selected_slots
            or not 0 <= positive_l0 <= 32
            or not 0 <= dead_rate <= 1
            or sparsity.get("total_features") != 65_536
            or not isinstance(sparsity.get("dead_rate_scope"), str)
            or not sparsity.get("dead_rate_scope")
            or not math.isclose(
                positive_l0,
                positive_sum / tokens,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                zero_fraction,
                1.0 - positive_l0 / 32.0,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or (
                identity[1] == SOFTPLUS_CONDITION
                and (
                    sparsity.get("positive_l0_contract")
                    != "exact_selected_slots"
                    or not math.isclose(positive_l0, 32.0, abs_tol=1e-12)
                )
            )
            or (
                identity[1] == RELU_CONDITION
                and sparsity.get("positive_l0_contract")
                != "bounded_zero_to_selected_slots"
            )
        ):
            raise RoutedLadderV3ReportError(
                f"cell sparsity policy changed: {identity}"
            )
        timing = row.get("timing")
        if (
            not isinstance(timing, Mapping)
            or set(timing)
            != {
                "cell_evaluation_wall_seconds",
                "cell_evaluation_chip_seconds",
                "cell_timing_scope",
                "bundle_accelerator_type",
                "bundle_chip_count",
                "bundle_result_sha256",
                "bundle_successful_attempt_wall_seconds",
                "bundle_successful_attempt_chip_seconds",
                "bundle_timing_scope",
                "allocation_note",
            }
        ):
            raise RoutedLadderV3ReportError(
                f"cell timing is invalid: {identity}"
            )
        wall = _finite(
            timing.get("cell_evaluation_wall_seconds"),
            f"{identity} cell evaluation wall seconds",
            positive=True,
        )
        chip_count = _positive_int(
            timing.get("bundle_chip_count"),
            f"{identity} bundle chip count",
        )
        accelerator_type = timing.get("bundle_accelerator_type")
        if (
            not isinstance(accelerator_type, str)
            or _CHIP_COUNT_BY_ACCELERATOR.get(accelerator_type)
            != chip_count
            or not isinstance(timing.get("cell_timing_scope"), str)
            or not timing.get("cell_timing_scope")
            or not isinstance(timing.get("bundle_timing_scope"), str)
            or not timing.get("bundle_timing_scope")
            or not isinstance(timing.get("allocation_note"), str)
            or not timing.get("allocation_note")
            or not math.isclose(
                _finite(
                    timing.get("cell_evaluation_chip_seconds"),
                    f"{identity} cell evaluation chip seconds",
                    positive=True,
                ),
                wall * chip_count,
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
        ):
            raise RoutedLadderV3ReportError(
                f"cell timing arithmetic changed: {identity}"
            )
        _sha256(
            timing.get("bundle_result_sha256"),
            f"{identity} cell bundle result hash",
        )
        for name in (
            "result_sha256",
            "result_file_sha256",
            "score_artifact_sha256",
            "score_data_sha256",
            "parent_training_result_sha256",
            "parent_training_result_file_sha256",
            "parent_run_contract_sha256",
            "parent_run_contract_file_sha256",
            "parent_checkpoint_pointer_sha256",
            "parent_checkpoint_pointer_file_sha256",
        ):
            _sha256(row.get(name), f"{identity}.{name}")
    expected_cell_grid = {
        (panel, condition, seed, checkpoint)
        for panel in PANELS
        for condition in CONDITIONS
        for seed in SEEDS
        for checkpoint in CHECKPOINT_TOKENS
    }
    if set(cell_grid) != expected_cell_grid:
        raise RoutedLadderV3ReportError("report cell grid is incomplete")

    raw_bundles = report["bundle_timings"]
    if not isinstance(raw_bundles, list) or len(raw_bundles) != 6:
        raise RoutedLadderV3ReportError(
            "report requires six resident-bundle timing rows"
        )
    bundle_grid: dict[tuple[str, int], Mapping[str, Any]] = {}
    expected_bundle_fields = {
        "panel",
        "seed",
        "accelerator_type",
        "chip_count",
        "successful_attempt_wall_seconds",
        "successful_attempt_chip_seconds",
        "timing_scope",
        "bundle_result_sha256",
        "bundle_result_file_sha256",
    }
    for row in raw_bundles:
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_bundle_fields
        ):
            raise RoutedLadderV3ReportError("bundle timing row is invalid")
        identity = (row.get("panel"), row.get("seed"))
        if (
            identity[0] not in PANELS
            or identity[1] not in SEEDS
            or identity in bundle_grid
        ):
            raise RoutedLadderV3ReportError(
                f"bundle timing identity is invalid: {identity}"
            )
        bundle_grid[identity] = row
        wall = _finite(
            row.get("successful_attempt_wall_seconds"),
            f"{identity} bundle wall seconds",
            positive=True,
        )
        chip_count = _positive_int(
            row.get("chip_count"),
            f"{identity} bundle chip count",
        )
        accelerator_type = row.get("accelerator_type")
        if (
            not isinstance(accelerator_type, str)
            or _CHIP_COUNT_BY_ACCELERATOR.get(accelerator_type)
            != chip_count
            or not isinstance(row.get("timing_scope"), str)
            or not row.get("timing_scope")
            or not math.isclose(
                _finite(
                    row.get("successful_attempt_chip_seconds"),
                    f"{identity} bundle chip seconds",
                    positive=True,
                ),
                wall * chip_count,
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
        ):
            raise RoutedLadderV3ReportError(
                f"bundle timing arithmetic changed: {identity}"
            )
        _sha256(
            row.get("bundle_result_sha256"),
            f"{identity} bundle result hash",
        )
        _sha256(
            row.get("bundle_result_file_sha256"),
            f"{identity} bundle file hash",
        )
    if set(bundle_grid) != {
        (panel, seed) for panel in PANELS for seed in SEEDS
    }:
        raise RoutedLadderV3ReportError("bundle timing grid is incomplete")
    for cell_identity, cell in cell_grid.items():
        bundle = bundle_grid[(str(cell_identity[0]), int(cell_identity[2]))]
        timing = cell["timing"]
        if (
            timing["bundle_accelerator_type"] != bundle["accelerator_type"]
            or timing["bundle_chip_count"] != bundle["chip_count"]
            or timing["bundle_result_sha256"]
            != bundle["bundle_result_sha256"]
            or timing["bundle_successful_attempt_wall_seconds"]
            != bundle["successful_attempt_wall_seconds"]
            or timing["bundle_successful_attempt_chip_seconds"]
            != bundle["successful_attempt_chip_seconds"]
            or timing["bundle_timing_scope"] != bundle["timing_scope"]
        ):
            raise RoutedLadderV3ReportError(
                f"cell timing is not bound to its resident bundle: "
                f"{cell_identity}"
            )

    raw_contrasts = report["contrasts"]
    if (
        not isinstance(raw_contrasts, list)
        or len(raw_contrasts)
        != len(PANELS) * len(CHECKPOINT_TOKENS) * len(ENDPOINTS)
    ):
        raise RoutedLadderV3ReportError("report contrast grid is incomplete")
    seen_contrasts: set[tuple[str, int, str]] = set()
    contrast_grid: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    expected_contrast_fields = {
        "panel",
        "checkpoint_tokens",
        "endpoint",
        "contrast",
        "effect",
        "per_seed",
        "across_seed",
        "verdict",
    }
    expected_per_seed_fields = {
        "seed",
        "estimate_ev_difference",
        "ci95_lower",
        "ci95_upper",
        "confidence_level",
        "bootstrap_unit",
        "bootstrap_algorithm",
        "bootstrap_replicates",
        "bootstrap_seed",
        "evaluation_tokens",
        "packed_sequences",
        "token_identity_sha256",
    }
    expected_across_seed_fields = {
        "estimate_mean_ev_difference",
        "ci95_lower",
        "ci95_upper",
        "confidence_level",
        "bootstrap_unit",
        "bootstrap_algorithm",
        "bootstrap_replicates",
        "bootstrap_seed",
        "seeds",
        "positive_seed_count",
        "seed_effects",
        "evaluation_tokens_per_seed",
        "packed_sequences_per_seed",
        "token_identity_sha256",
    }
    for row in raw_contrasts:
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_contrast_fields
        ):
            raise RoutedLadderV3ReportError("contrast row is invalid")
        identity = (
            row.get("panel"),
            row.get("checkpoint_tokens"),
            row.get("endpoint"),
        )
        if (
            identity[0] not in PANELS
            or identity[1] not in CHECKPOINT_TOKENS
            or identity[2] not in ENDPOINTS
            or identity in seen_contrasts
            or row.get("contrast") != CONTRAST
            or row.get("effect")
            != "EV(A512Softplus) - EV(A512ReLUInclusiveTopK)"
        ):
            raise RoutedLadderV3ReportError(
                f"contrast identity is invalid: {identity}"
            )
        seen_contrasts.add(identity)
        contrast_grid[identity] = row
        per_seed = row.get("per_seed")
        if not isinstance(per_seed, list) or len(per_seed) != len(SEEDS):
            raise RoutedLadderV3ReportError(
                f"contrast lacks three seed rows: {identity}"
            )
        observed_effects: list[float] = []
        for seed, effect in zip(SEEDS, per_seed, strict=True):
            expected_bootstrap_seed = _bootstrap_seed(
                panel=str(identity[0]),
                seed=seed,
                checkpoint_tokens=int(identity[1]),
                endpoint=str(identity[2]),
                token_identity_sha256=common["token_identity_sha256"],
                level="packed-sequence",
            )
            if (
                not isinstance(effect, Mapping)
                or set(effect) != expected_per_seed_fields
                or effect.get("seed") != seed
                or effect.get("confidence_level") != CONFIDENCE_LEVEL
                or effect.get("bootstrap_unit") != "packed_sequence"
                or effect.get("bootstrap_algorithm")
                != SEED_BOOTSTRAP_ALGORITHM
                or effect.get("bootstrap_replicates")
                != BOOTSTRAP_REPLICATES
                or effect.get("bootstrap_seed") != expected_bootstrap_seed
                or effect.get("evaluation_tokens") != tokens
                or effect.get("packed_sequences") != packed
                or effect.get("token_identity_sha256")
                != common["token_identity_sha256"]
            ):
                raise RoutedLadderV3ReportError(
                    f"per-seed interval metadata changed: {identity}/seed{seed}"
                )
            expected_delta = (
                cell_grid[
                    (
                        identity[0],
                        SOFTPLUS_CONDITION,
                        seed,
                        identity[1],
                    )
                ]["endpoints"][identity[2]]["ev"]
                - cell_grid[
                    (
                        identity[0],
                        RELU_CONDITION,
                        seed,
                        identity[1],
                    )
                ]["endpoints"][identity[2]]["ev"]
            )
            estimate = _finite(
                effect.get("estimate_ev_difference"),
                f"{identity}/seed{seed} estimate",
            )
            lower = _finite(
                effect.get("ci95_lower"),
                f"{identity}/seed{seed} lower",
            )
            upper = _finite(
                effect.get("ci95_upper"),
                f"{identity}/seed{seed} upper",
            )
            if (
                not math.isclose(
                    estimate,
                    expected_delta,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or lower > upper
            ):
                raise RoutedLadderV3ReportError(
                    f"per-seed effect arithmetic changed: {identity}/seed{seed}"
                )
            observed_effects.append(estimate)
        across = row.get("across_seed")
        expected_across_seed = _bootstrap_seed(
            panel=str(identity[0]),
            seed=None,
            checkpoint_tokens=int(identity[1]),
            endpoint=str(identity[2]),
            token_identity_sha256=common["token_identity_sha256"],
            level="crossed-seed-and-packed-sequence",
        )
        if (
            not isinstance(across, Mapping)
            or set(across) != expected_across_seed_fields
            or across.get("confidence_level") != CONFIDENCE_LEVEL
            or across.get("bootstrap_unit") != _CROSSED_BOOTSTRAP_UNIT
            or across.get("bootstrap_algorithm")
            != HIERARCHICAL_BOOTSTRAP_ALGORITHM
            or across.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
            or across.get("bootstrap_seed") != expected_across_seed
            or across.get("seeds") != list(SEEDS)
            or across.get("positive_seed_count")
            != sum(value > 0 for value in observed_effects)
            or across.get("evaluation_tokens_per_seed") != tokens
            or across.get("packed_sequences_per_seed") != packed
            or across.get("token_identity_sha256")
            != common["token_identity_sha256"]
        ):
            raise RoutedLadderV3ReportError(
                f"hierarchical interval metadata changed: {identity}"
            )
        if across.get("seed_effects") != [
            {
                "seed": seed,
                "estimate_ev_difference": observed_effects[seed],
            }
            for seed in SEEDS
        ]:
            raise RoutedLadderV3ReportError(
                f"hierarchical seed effects changed: {identity}"
            )
        mean = _finite(
            across.get("estimate_mean_ev_difference"),
            f"{identity} mean effect",
        )
        lower = _finite(across.get("ci95_lower"), f"{identity} lower")
        upper = _finite(across.get("ci95_upper"), f"{identity} upper")
        if (
            not math.isclose(
                mean,
                math.fsum(observed_effects) / len(SEEDS),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or lower > upper
        ):
            raise RoutedLadderV3ReportError(
                f"hierarchical effect arithmetic changed: {identity}"
            )
        expected_verdict = (
            "softplus_better"
            if lower > 0
            else (
                "relu_inclusive_topk_better"
                if upper < 0
                else "inconclusive"
            )
        )
        if row.get("verdict") != expected_verdict:
            raise RoutedLadderV3ReportError(
                f"contrast verdict changed: {identity}"
            )

    primary = report["primary_verdict"]
    if (
        not isinstance(primary, Mapping)
        or primary.get("checkpoint_tokens") != 10_000_000
        or primary.get("endpoint") != PRIMARY_ENDPOINT
        or not isinstance(primary.get("panels"), list)
        or len(primary["panels"]) != len(PANELS)
    ):
        raise RoutedLadderV3ReportError("primary verdict identity changed")
    for panel, row in zip(PANELS, primary["panels"], strict=True):
        source = contrast_grid[(panel, 10_000_000, PRIMARY_ENDPOINT)]
        across = source["across_seed"]
        expected = {
            "panel": panel,
            "estimate_mean_ev_difference": across[
                "estimate_mean_ev_difference"
            ],
            "ci95_lower": across["ci95_lower"],
            "ci95_upper": across["ci95_upper"],
            "positive_seed_count": across["positive_seed_count"],
            "verdict": source["verdict"],
        }
        if row != expected:
            raise RoutedLadderV3ReportError(
                f"primary verdict differs from final contrast: {panel}"
            )

    cost = report["training_cost_comparison"]
    if (
        not isinstance(cost, Mapping)
        or set(cost)
        != {
            "authority",
            "ratio_policy",
            "shared_harvest_excluded",
            "trajectory_costs",
            "paired_seed_comparisons",
            "panel_summaries",
        }
        or cost.get("authority")
        != "immutable marginal SAE/transcoder training-result chip-seconds"
        or cost.get("ratio_policy") != "same_accelerator_type_only"
        or cost.get("shared_harvest_excluded") is not True
    ):
        raise RoutedLadderV3ReportError("training cost report is invalid")
    trajectory = cost.get("trajectory_costs")
    if not isinstance(trajectory, list) or len(trajectory) != 12:
        raise RoutedLadderV3ReportError(
            "training cost report requires twelve trajectories"
        )
    cost_grid: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    expected_trajectory_fields = {
        "panel",
        "condition",
        "seed",
        "origin",
        "resource_id",
        "accelerator_type",
        "chip_count",
        "tokens",
        "marginal_wall_seconds",
        "chip_seconds",
        "tokens_per_second",
        "chip_seconds_scope",
        "chip_seconds_excludes",
        "compile_seconds_excluded",
        "lowering_seconds_excluded",
        "checkpoint_seconds_excluded",
        "preprocessing_wall_seconds_included",
        "optimizer_wall_seconds_included",
        "training_source_revision",
        "training_result_sha256",
        "training_result_file_sha256",
    }
    observed_cost_scopes: set[tuple[str, tuple[str, ...]]] = set()
    for row in trajectory:
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_trajectory_fields
        ):
            raise RoutedLadderV3ReportError(
                "training trajectory cost row is invalid"
            )
        identity = (row.get("panel"), row.get("condition"), row.get("seed"))
        if (
            identity[0] not in PANELS
            or identity[1] not in CONDITIONS
            or identity[2] not in SEEDS
            or identity in cost_grid
        ):
            raise RoutedLadderV3ReportError(
                f"invalid or duplicate training cost row: {identity}"
            )
        cost_grid[identity] = row
        wall = _finite(
            row.get("marginal_wall_seconds"),
            f"{identity} marginal wall seconds",
            positive=True,
        )
        chip_count = _positive_int(
            row.get("chip_count"),
            f"{identity} chip count",
        )
        chips = _finite(
            row.get("chip_seconds"),
            f"{identity} chip seconds",
            positive=True,
        )
        rate = _finite(
            row.get("tokens_per_second"),
            f"{identity} tokens per second",
            positive=True,
        )
        preprocessing = _finite(
            row.get("preprocessing_wall_seconds_included"),
            f"{identity} preprocessing seconds",
        )
        optimizer = _finite(
            row.get("optimizer_wall_seconds_included"),
            f"{identity} optimizer seconds",
        )
        accelerator_type = row.get("accelerator_type")
        excludes = row.get("chip_seconds_excludes")
        scope = row.get("chip_seconds_scope")
        expected_origin = (
            "fresh_v3"
            if identity[1] == RELU_CONDITION
            else "imported_v2_exact_graph"
        )
        revision = row.get("training_source_revision")
        if (
            row.get("origin") != expected_origin
            or not isinstance(row.get("resource_id"), str)
            or not row.get("resource_id")
            or not isinstance(accelerator_type, str)
            or _CHIP_COUNT_BY_ACCELERATOR.get(accelerator_type)
            != chip_count
            or row.get("tokens") != 10_000_000
            or not isinstance(scope, str)
            or not scope
            or not isinstance(excludes, list)
            or not excludes
            or not all(isinstance(item, str) and item for item in excludes)
            or not isinstance(revision, str)
            or len(revision) != 40
            or any(
                character not in "0123456789abcdef"
                for character in revision
            )
            or preprocessing < 0
            or optimizer < 0
            or not math.isclose(
                wall,
                preprocessing + optimizer,
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            or not math.isclose(
                chips,
                wall * chip_count,
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            or not math.isclose(
                rate,
                10_000_000 / wall,
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
        ):
            raise RoutedLadderV3ReportError(
                f"training cost arithmetic changed: {identity}"
            )
        for field in (
            "compile_seconds_excluded",
            "lowering_seconds_excluded",
            "checkpoint_seconds_excluded",
        ):
            if _finite(row.get(field), f"{identity} {field}") < 0:
                raise RoutedLadderV3ReportError(
                    f"training excluded timing is negative: {identity}"
                )
        observed_cost_scopes.add((scope, tuple(excludes)))
        _sha256(
            row.get("training_result_sha256"),
            f"{identity} training result hash",
        )
        _sha256(
            row.get("training_result_file_sha256"),
            f"{identity} training result file hash",
        )
    expected_cost_grid = {
        (panel, condition, seed)
        for panel in PANELS
        for condition in CONDITIONS
        for seed in SEEDS
    }
    if set(cost_grid) != expected_cost_grid:
        raise RoutedLadderV3ReportError("training cost grid is incomplete")
    if len(observed_cost_scopes) != 1:
        raise RoutedLadderV3ReportError(
            "training trajectories do not share one measured-cost scope"
        )
    paired_costs = cost.get("paired_seed_comparisons")
    if not isinstance(paired_costs, list) or len(paired_costs) != 6:
        raise RoutedLadderV3ReportError(
            "training cost comparison requires six paired seeds"
        )
    expected_pair_fields = {
        "panel",
        "seed",
        "relu_accelerator_type",
        "softplus_accelerator_type",
        "comparison_status",
        "relu_chip_seconds",
        "softplus_chip_seconds",
        "softplus_minus_relu_chip_seconds",
        "softplus_to_relu_chip_seconds_ratio_percent",
        "relu_to_softplus_chip_seconds_ratio_percent",
        "comparison_scope",
        "comparison_excludes",
    }
    pair_grid: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in paired_costs:
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_pair_fields
        ):
            raise RoutedLadderV3ReportError(
                "paired training cost row fields changed"
            )
        panel = row.get("panel")
        seed = row.get("seed")
        identity = (panel, seed)
        if (
            panel not in PANELS
            or seed not in SEEDS
            or identity in pair_grid
        ):
            raise RoutedLadderV3ReportError(
                f"paired training cost identity changed: {identity}"
            )
        pair_grid[identity] = row
        relu_row = cost_grid[(panel, RELU_CONDITION, seed)]
        softplus_row = cost_grid[(panel, SOFTPLUS_CONDITION, seed)]
        relu = float(relu_row["chip_seconds"])
        softplus = float(softplus_row["chip_seconds"])
        comparable = (
            relu_row["accelerator_type"]
            == softplus_row["accelerator_type"]
        )
        expected_status = (
            "comparable_same_accelerator_type"
            if comparable
            else "non_comparable_accelerator_type_mismatch"
        )
        difference = row.get("softplus_minus_relu_chip_seconds")
        ratio = row.get("softplus_to_relu_chip_seconds_ratio_percent")
        inverse_ratio = row.get(
            "relu_to_softplus_chip_seconds_ratio_percent"
        )
        if (
            row.get("relu_accelerator_type")
            != relu_row["accelerator_type"]
            or row.get("softplus_accelerator_type")
            != softplus_row["accelerator_type"]
            or row.get("comparison_status") != expected_status
            or row.get("comparison_scope") != relu_row["chip_seconds_scope"]
            or row.get("comparison_excludes")
            != relu_row["chip_seconds_excludes"]
            or not math.isclose(
                _finite(
                    row.get("relu_chip_seconds"),
                    f"{identity} ReLU chip seconds",
                    positive=True,
                ),
                relu,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                _finite(
                    row.get("softplus_chip_seconds"),
                    f"{identity} Softplus chip seconds",
                    positive=True,
                ),
                softplus,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or (
                comparable
                and (
                    not math.isclose(
                        _finite(difference, f"{identity} cost difference"),
                        softplus - relu,
                        rel_tol=0,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        _finite(ratio, f"{identity} cost ratio"),
                        100.0 * softplus / relu,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        _finite(
                            inverse_ratio,
                            f"{identity} inverse cost ratio",
                        ),
                        100.0 * relu / softplus,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
            )
            or (
                not comparable
                and (
                    difference is not None
                    or ratio is not None
                    or inverse_ratio is not None
                )
            )
        ):
            raise RoutedLadderV3ReportError(
                f"paired training cost arithmetic changed: {panel}/seed{seed}"
            )
    if set(pair_grid) != {
        (panel, seed) for panel in PANELS for seed in SEEDS
    }:
        raise RoutedLadderV3ReportError(
            "paired training cost identity grid is incomplete"
        )
    panel_summaries = cost.get("panel_summaries")
    if not isinstance(panel_summaries, list) or len(panel_summaries) != 2:
        raise RoutedLadderV3ReportError(
            "training cost report requires two panel summaries"
        )
    expected_panel_fields = {
        "panel",
        "comparison_status",
        "comparable_seed_count",
        "relu_median_chip_seconds",
        "softplus_median_chip_seconds",
        "median_paired_softplus_to_relu_ratio_percent",
        "relu_total_chip_seconds",
        "softplus_total_chip_seconds",
        "seeds",
    }
    seen_panels: set[str] = set()
    for row in panel_summaries:
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_panel_fields
            or row.get("panel") not in PANELS
            or row.get("panel") in seen_panels
            or row.get("seeds") != list(SEEDS)
        ):
            raise RoutedLadderV3ReportError(
                "training cost panel summary identity changed"
            )
        panel = str(row["panel"])
        seen_panels.add(panel)
        relu_values = [
            float(
                cost_grid[(panel, RELU_CONDITION, seed)]["chip_seconds"]
            )
            for seed in SEEDS
        ]
        softplus_values = [
            float(
                cost_grid[(panel, SOFTPLUS_CONDITION, seed)]["chip_seconds"]
            )
            for seed in SEEDS
        ]
        ratios = [
            pair_grid[(panel, seed)][
                "softplus_to_relu_chip_seconds_ratio_percent"
            ]
            for seed in SEEDS
            if pair_grid[(panel, seed)]["comparison_status"]
            == "comparable_same_accelerator_type"
        ]
        all_comparable = len(ratios) == len(SEEDS)
        expected_panel_status = (
            "comparable_all_seeds_same_accelerator_type"
            if all_comparable
            else "non_comparable_one_or_more_accelerator_type_mismatches"
        )
        expected_ratio = (
            float(np.median(np.asarray(ratios, dtype=np.float64)))
            if all_comparable
            else None
        )
        observed_ratio = row.get(
            "median_paired_softplus_to_relu_ratio_percent"
        )
        if (
            row.get("comparison_status") != expected_panel_status
            or row.get("comparable_seed_count") != len(ratios)
            or not math.isclose(
                _finite(
                    row.get("relu_median_chip_seconds"),
                    f"{panel} median ReLU chip seconds",
                    positive=True,
                ),
                float(np.median(np.asarray(relu_values, dtype=np.float64))),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                _finite(
                    row.get("softplus_median_chip_seconds"),
                    f"{panel} median Softplus chip seconds",
                    positive=True,
                ),
                float(
                    np.median(np.asarray(softplus_values, dtype=np.float64))
                ),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                _finite(
                    row.get("relu_total_chip_seconds"),
                    f"{panel} total ReLU chip seconds",
                    positive=True,
                ),
                math.fsum(relu_values),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                _finite(
                    row.get("softplus_total_chip_seconds"),
                    f"{panel} total Softplus chip seconds",
                    positive=True,
                ),
                math.fsum(softplus_values),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or (
                all_comparable
                and not math.isclose(
                    _finite(observed_ratio, f"{panel} median cost ratio"),
                    float(expected_ratio),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            or (not all_comparable and observed_ratio is not None)
        ):
            raise RoutedLadderV3ReportError(
                f"training cost panel summary arithmetic changed: {panel}"
            )
    if seen_panels != set(PANELS):
        raise RoutedLadderV3ReportError(
            "training cost panel summary grid is incomplete"
        )
    context = report["optional_context"]
    if (
        not isinstance(context, Mapping)
        or set(context) != {"A256Native", "H"}
    ):
        raise RoutedLadderV3ReportError(
            "optional A256/H context inventory changed"
        )
    _validate_a256_report_context(context["A256Native"])
    h_context = context["H"]
    if (
        not isinstance(h_context, Mapping)
        or h_context.get("primary_verdict_blocked") is not False
        or h_context.get("h_scores_in_primary_inference") is not False
    ):
        raise RoutedLadderV3ReportError(
            "H context entered primary inference or blocked the verdict"
        )
    if h_context.get("status") == "absent_nonblocking":
        if (
            h_context.get("checkpoint_reference_count") != 0
            or "addendum" not in str(h_context.get("reason"))
        ):
            raise RoutedLadderV3ReportError(
                "absent H context metadata changed"
            )
    elif h_context.get("status") == "available_read_only_checkpoint_context":
        references = h_context.get("checkpoint_references")
        if (
            h_context.get("checkpoint_reference_count") != 18
            or not isinstance(references, list)
            or len(references) != 18
        ):
            raise RoutedLadderV3ReportError(
                "available H context must pin exactly 18 checkpoints"
            )
        _sha256(
            h_context.get("manifest_sha256"),
            "H reference manifest hash",
        )
        identities = {
            (
                row.get("panel"),
                row.get("seed"),
                row.get("checkpoint_tokens"),
            )
            for row in references
            if isinstance(row, Mapping)
        }
        expected_h = {
            (panel, seed, checkpoint)
            for panel in PANELS
            for seed in SEEDS
            for checkpoint in CHECKPOINT_TOKENS
        }
        if identities != expected_h:
            raise RoutedLadderV3ReportError(
                "available H checkpoint identity grid changed"
            )
        for row in references:
            for name in (
                "h_config_sha256",
                "training_result_sha256",
                "training_result_file_sha256",
                "checkpoint_pointer_sha256",
                "checkpoint_pointer_file_sha256",
            ):
                _sha256(row.get(name), f"H reference {name}")
    else:
        raise RoutedLadderV3ReportError("H context status is invalid")
    rendered = _render_v3_primary_markdown_core(report)
    if (
        _sha256(report["markdown_sha256"], "markdown_sha256")
        != hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    ):
        raise RoutedLadderV3ReportError(
            "Markdown content hash differs from deterministic report rendering"
        )
    return report


def build_v3_primary_report_from_paths(
    *,
    evaluation_plan_path: str | Path,
    training_plan_path: str | Path,
    completed_manifest_path: str | Path,
    a256_provenance_addendum_path: str | Path | None = None,
    h_reference_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    evaluation = load_v3_primary_evaluation_plan(evaluation_plan_path)
    training = load_routed_ladder_v3_training_plan(training_plan_path)
    if (
        evaluation["training_plan_sha256"] != training.plan_sha256
        or evaluation["training_source_revision"] != training.project_commit
    ):
        raise RoutedLadderV3ReportError(
            "evaluation and training plans have different immutable parents"
        )
    try:
        completed = yaml.safe_load(
            Path(completed_manifest_path).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RoutedLadderV3ReportError(
            "cannot read completed Softplus reference manifest"
        ) from error
    if not isinstance(completed, Mapping):
        raise RoutedLadderV3ReportError(
            "completed Softplus reference manifest must be an object"
        )
    a256_addendum: Mapping[str, Any] | None = None
    if a256_provenance_addendum_path is not None:
        try:
            loaded_a256 = yaml.safe_load(
                Path(a256_provenance_addendum_path).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
            raise RoutedLadderV3ReportError(
                "cannot read optional A256 provenance addendum"
            ) from error
        if not isinstance(loaded_a256, Mapping):
            raise RoutedLadderV3ReportError(
                "optional A256 provenance addendum must be an object"
            )
        a256_addendum = loaded_a256
    h_reference: Mapping[str, Any] | None = None
    if h_reference_manifest_path is not None:
        try:
            loaded_h = yaml.safe_load(
                Path(h_reference_manifest_path).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
            raise RoutedLadderV3ReportError(
                "cannot read optional H reference manifest"
            ) from error
        if not isinstance(loaded_h, Mapping):
            raise RoutedLadderV3ReportError(
                "optional H reference manifest must be an object"
            )
        h_reference = loaded_h
    cells, timings = collect_primary_cell_evidence(evaluation)
    _validate_bundle_resource_ownership(evaluation, training, timings)
    costs = collect_training_cost_evidence(
        training,
        completed,
        expected_manifest_sha256=evaluation[
            "completed_trajectory_manifest_sha256"
        ],
    )
    return build_v3_primary_report_from_evidence(
        evaluation,
        cells,
        timings,
        costs,
        a256_provenance_addendum=a256_addendum,
        h_reference_manifest=h_reference,
    )


def write_v3_primary_report(
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    selected = validate_v3_primary_report(report)
    json_destination = Path(json_path).expanduser().resolve()
    markdown_destination = Path(markdown_path).expanduser().resolve()
    if json_destination == markdown_destination:
        raise RoutedLadderV3ReportError(
            "JSON and Markdown report paths must differ"
        )
    json_payload = _json_payload(selected)
    markdown_payload = render_v3_primary_report_markdown(selected).encode(
        "utf-8"
    )
    _create_only(json_destination, json_payload)
    _create_only(markdown_destination, markdown_payload)
    return {
        "status": REPORT_STATUS,
        "report_sha256": selected["report_sha256"],
        "json_path": str(json_destination),
        "json_file_sha256": hashlib.sha256(json_payload).hexdigest(),
        "markdown_path": str(markdown_destination),
        "markdown_file_sha256": hashlib.sha256(
            markdown_payload
        ).hexdigest(),
    }


def load_v3_primary_report(path: str | Path) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    value, _payload = _load_json(selected, "v3 primary report")
    return validate_v3_primary_report(value)


def verify_v3_primary_report_from_paths(
    *,
    report_path: str | Path,
    markdown_path: str | Path,
    evaluation_plan_path: str | Path,
    training_plan_path: str | Path,
    completed_manifest_path: str | Path,
    a256_provenance_addendum_path: str | Path | None = None,
    h_reference_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    observed = load_v3_primary_report(report_path)
    expected = build_v3_primary_report_from_paths(
        evaluation_plan_path=evaluation_plan_path,
        training_plan_path=training_plan_path,
        completed_manifest_path=completed_manifest_path,
        a256_provenance_addendum_path=a256_provenance_addendum_path,
        h_reference_manifest_path=h_reference_manifest_path,
    )
    if observed != expected:
        raise RoutedLadderV3ReportError(
            "sealed report differs from recomputed immutable evidence"
        )
    markdown = Path(markdown_path).expanduser().resolve()
    try:
        payload = markdown.read_bytes()
    except OSError as error:
        raise RoutedLadderV3ReportError(
            f"cannot read v3 Markdown report: {markdown}"
        ) from error
    expected_markdown = render_v3_primary_report_markdown(observed).encode(
        "utf-8"
    )
    if payload != expected_markdown:
        raise RoutedLadderV3ReportError(
            "sealed Markdown differs from deterministic report rendering"
        )
    return {
        "status": "primary_report_verified",
        "report_sha256": observed["report_sha256"],
        "json_file_sha256": hashlib.sha256(
            Path(report_path).read_bytes()
        ).hexdigest(),
        "markdown_file_sha256": hashlib.sha256(payload).hexdigest(),
    }


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "CONFIDENCE_LEVEL",
    "HIERARCHICAL_BOOTSTRAP_ALGORITHM",
    "MARKDOWN_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "REPORT_STATUS",
    "SEED_BOOTSTRAP_ALGORITHM",
    "BundleTimingEvidence",
    "PrimaryCellEvidence",
    "RoutedLadderV3ReportError",
    "TrainingCostEvidence",
    "build_v3_primary_report_from_evidence",
    "build_v3_primary_report_from_paths",
    "collect_a256_reference_context",
    "collect_primary_cell_evidence",
    "collect_training_cost_evidence",
    "hierarchical_v3_bootstrap",
    "load_v3_primary_report",
    "paired_v3_bootstrap",
    "render_v3_primary_report_markdown",
    "validate_v3_primary_report",
    "verify_v3_primary_report_from_paths",
    "write_v3_primary_report",
]
