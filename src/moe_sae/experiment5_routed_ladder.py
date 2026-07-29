"""Scientific scoring for the corrected Experiment-5 routed activation ladder.

This module is deliberately separate from the frozen H/R/B/F implementation.
It scores three routed-only true-ID variants on identical scientific tokens:

* ``A256ReLU`` has 256 features and TopK-2 per visible expert bank, with ReLU;
* ``A512ReLU`` has 512 features and TopK-4 per visible expert bank, with ReLU;
* ``A512Softplus`` has the exact same geometry and uses Softplus; and
* ``H`` is the existing Experiment-5 Softplus routed model plus a learned
  shared dictionary.

The A variants use the exact shared branch when the ``oracle_shared`` endpoint
is formed.  V2 reports two paired effects: A512ReLU minus A256ReLU (the frozen
width/L0 scaling step at constant within-bank activation density), and
A512Softplus minus A512ReLU (encoder activation at fixed geometry).  H evidence
is retained as hash-pinned context and is never silently promoted into a
shared-branch causal contrast.

Experiment 4 supplies only the compatible SAE A256ReLU seed 0.  The report
requires newly trained SAE seeds 1 and 2 before it will emit an across-seed
A256 contrast.  The historical transcoder A256 used Softplus, so protocol v2
trains all three transcoder A256ReLU seeds and never imports that checkpoint.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from moe_sae.contribution_artifacts import (
    canonical_sha256,
    token_identity_sha256,
)

PROTOCOL_ID = "gemma4-experiment5-routed-activation-ladder-v2-20260729"
SCORE_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-token-scores-v2"
)
REPORT_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-report-v2"
)

PANELS = ("sae", "transcoder")
CONDITIONS = ("A256ReLU", "A512ReLU", "A512Softplus", "H")
SCORED_CONDITIONS = ("A256ReLU", "A512ReLU", "A512Softplus")
PAIRED_CONTRASTS = (
    ("A256ReLU", "A512ReLU"),
    ("A512ReLU", "A512Softplus"),
)
ENDPOINTS = ("component", "routed", "oracle_shared")
H_ENDPOINTS = (*ENDPOINTS, "learned_full_ffn")
SEEDS = (0, 1, 2)
CHECKPOINT_TOKENS = (2_000_000, 5_000_000, 10_000_000)
SEQUENCE_LENGTH = 1_024
HIDDEN_SIZE = 2_816
ROUTED_BANKS = 128
VISIBLE_ROUTED_BANKS = 8
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_ALGORITHM = "paired-packed-sequence-percentile-bootstrap-v1"
HIERARCHICAL_BOOTSTRAP_ALGORITHM = (
    "hierarchical-paired-seed-packed-sequence-percentile-bootstrap-v1"
)
CONFIDENCE_LEVEL = 0.95

CAVEATS = (
    "SAE A256ReLU seed 0 is a hash-pinned Experiment-4 trajectory; SAE seeds "
    "1 and 2 are newly trained by protocol v2. The historical transcoder A256 "
    "used Softplus and is excluded; all three transcoder A256ReLU seeds are "
    "new. No A256 across-seed result exists until all three exact cells in a "
    "panel are present.",
    "A256ReLU to A512ReLU doubles routed width, candidates, per-bank TopK, L0, "
    "and parameters while holding TopK/bank-width density at 0.78125%; it is a "
    "frozen scale step, not a pure width-only or compute-matched ablation.",
    "A512ReLU and A512Softplus have identical 128-by-512 routed geometry, "
    "TopK-4 per visible expert bank, L0=32, initialization, token order, "
    "optimizer, and training budget; encoder activation is the only treatment.",
    "Both A conditions use the exact shared branch at the oracle_shared "
    "endpoint; they neither learn nor pay for a shared dictionary.",
    "H uses a learned 4,096-feature TopK-32 shared dictionary, so its full "
    "output quality and cost are not directly exchangeable with oracle-shared "
    "A quality.",
    "H was jointly trained against the full FFN output, while each A arm was "
    "trained against routed expert components; matching H and A512Softplus "
    "activation and routed geometry does not make H a randomized ablation.",
    "A512ReLU-minus-A256ReLU measures the frozen scale step within each panel, "
    "seed, milestone, endpoint, and common scientific token set.",
    "A512Softplus-minus-A512ReLU isolates the encoder nonlinearity within each "
    "panel, seed, milestone, endpoint, and common scientific token set.",
    "H-oracle-minus-A512Softplus-oracle and H-oracle-minus-H-learned are matched "
    "descriptive point gaps. Exact H score rows are retained, but neither gap "
    "is a controlled causal effect or authorized confirmatory interval.",
    "This report authorizes no A-versus-H learned-shared causal or efficiency "
    "headline.",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_KIND_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_TRAJECTORY_ORIGINS = (
    "historical_experiment4_seed0_reference",
    "historical_experiment5_reference",
    "protocol_v2_new_training",
)
_COST_STATUSES = (
    "measured_historical_training_result",
    "measured_v2_training_result",
    "not_supplied_by_h_reference_manifest",
)
_EVALUATION_TIMING_SCOPE = (
    "successful_resident_worker_after_cli_parse_through_cell_completion"
)
_CAMPAIGN_WALL_SCOPE = (
    "durable_campaign_clock_through_evaluation_completion"
)
_SEALED_COMPUTE_COVERAGE = (
    "Includes the 17 sealed protocol-v2 training results and six sealed "
    "successful resident evaluation-bundle invocations. Historical training "
    "compute is reported separately. Failed or preempted attempts are not "
    "reconstructed as chip-seconds; their elapsed exposure remains represented "
    "only by the separately provenance-bound campaign wall/cash projection."
)


class Experiment5RoutedLadderError(ValueError):
    """Routed-ladder evidence or its interpretation is invalid."""


def _nonnegative_int(name: str, value: Any, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Experiment5RoutedLadderError(f"{name} must be an integer")
    selected = int(value)
    if selected < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise Experiment5RoutedLadderError(f"{name} must be {qualifier}")
    return selected


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Experiment5RoutedLadderError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _git_sha(name: str, value: Any) -> str:
    if not isinstance(value, str) or _GIT_RE.fullmatch(value) is None:
        raise Experiment5RoutedLadderError(f"{name} must be a lowercase Git SHA")
    return value


def _exact_mapping(
    name: str,
    value: Any,
    fields: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise Experiment5RoutedLadderError(
            f"{name} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _finite(
    name: str,
    value: Any,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Experiment5RoutedLadderError(f"{name} must be numeric")
    selected = float(value)
    if (
        not math.isfinite(selected)
        or (nonnegative and selected < 0)
        or (positive and selected <= 0)
    ):
        qualifier = (
            "finite and positive"
            if positive
            else "finite and nonnegative"
            if nonnegative
            else "finite"
        )
        raise Experiment5RoutedLadderError(f"{name} must be {qualifier}")
    return selected


def _readonly(value: Any, *, dtype: Any) -> np.ndarray:
    selected = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    selected.setflags(write=False)
    return selected


@dataclass(frozen=True, slots=True)
class ConditionGeometry:
    """One exact activation-ladder dictionary and encoder contract."""

    condition: Literal["A256ReLU", "A512ReLU", "A512Softplus", "H"]
    encoder_activation: Literal["relu", "softplus"]
    routed_bank_width: int
    routed_topk_per_slot: int
    shared_features: int
    shared_topk: int
    shared_treatment: Literal["oracle_exact", "learned_sparse_dictionary"]

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise Experiment5RoutedLadderError("condition geometry is invalid")
        expected_activation = (
            "relu"
            if self.condition in {"A256ReLU", "A512ReLU"}
            else "softplus"
        )
        if self.encoder_activation != expected_activation:
            raise Experiment5RoutedLadderError(
                "encoder activation differs from condition identity"
            )
        _nonnegative_int(
            "routed_bank_width",
            self.routed_bank_width,
            positive=True,
        )
        _nonnegative_int(
            "routed_topk_per_slot",
            self.routed_topk_per_slot,
            positive=True,
        )
        _nonnegative_int("shared_features", self.shared_features)
        _nonnegative_int("shared_topk", self.shared_topk)
        if self.routed_topk_per_slot > self.routed_bank_width:
            raise Experiment5RoutedLadderError(
                "routed TopK exceeds routed bank width"
            )
        if (self.shared_features == 0) is not (self.shared_topk == 0):
            raise Experiment5RoutedLadderError(
                "shared width and TopK must both be zero or both be positive"
            )
        expected_treatment = (
            "oracle_exact"
            if self.shared_features == 0
            else "learned_sparse_dictionary"
        )
        if self.shared_treatment != expected_treatment:
            raise Experiment5RoutedLadderError(
                "shared treatment differs from dictionary geometry"
            )

    @property
    def routed_features(self) -> int:
        return ROUTED_BANKS * self.routed_bank_width

    @property
    def routed_candidates_per_token(self) -> int:
        return VISIBLE_ROUTED_BANKS * self.routed_bank_width

    @property
    def routed_l0(self) -> int:
        return VISIBLE_ROUTED_BANKS * self.routed_topk_per_slot

    @property
    def total_features(self) -> int:
        return self.routed_features + self.shared_features

    @property
    def candidates_per_token(self) -> int:
        return self.routed_candidates_per_token + self.shared_features

    @property
    def exact_l0(self) -> int:
        return self.routed_l0 + self.shared_topk

    @property
    def trainable_parameters(self) -> int:
        return self.total_features * (2 * HIDDEN_SIZE + 1)

    def report_row(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "encoder_activation": self.encoder_activation,
            "routed_banks": ROUTED_BANKS,
            "visible_routed_banks_per_token": VISIBLE_ROUTED_BANKS,
            "routed_bank_width": self.routed_bank_width,
            "routed_features": self.routed_features,
            "routed_candidates_per_token": self.routed_candidates_per_token,
            "routed_topk_per_slot": self.routed_topk_per_slot,
            "routed_l0": self.routed_l0,
            "shared_features": self.shared_features,
            "shared_candidates_per_token": self.shared_features,
            "shared_topk": self.shared_topk,
            "shared_treatment": self.shared_treatment,
            "total_features": self.total_features,
            "candidates_per_token": self.candidates_per_token,
            "nominal_candidate_exposure_fraction": (
                self.candidates_per_token / self.total_features
            ),
            "exact_l0": self.exact_l0,
            "trainable_parameters": self.trainable_parameters,
        }


CONDITION_GEOMETRIES: dict[str, ConditionGeometry] = {
    "A256ReLU": ConditionGeometry(
        condition="A256ReLU",
        encoder_activation="relu",
        routed_bank_width=256,
        routed_topk_per_slot=2,
        shared_features=0,
        shared_topk=0,
        shared_treatment="oracle_exact",
    ),
    "A512ReLU": ConditionGeometry(
        condition="A512ReLU",
        encoder_activation="relu",
        routed_bank_width=512,
        routed_topk_per_slot=4,
        shared_features=0,
        shared_topk=0,
        shared_treatment="oracle_exact",
    ),
    "A512Softplus": ConditionGeometry(
        condition="A512Softplus",
        encoder_activation="softplus",
        routed_bank_width=512,
        routed_topk_per_slot=4,
        shared_features=0,
        shared_topk=0,
        shared_treatment="oracle_exact",
    ),
    "H": ConditionGeometry(
        condition="H",
        encoder_activation="softplus",
        routed_bank_width=512,
        routed_topk_per_slot=4,
        shared_features=4_096,
        shared_topk=32,
        shared_treatment="learned_sparse_dictionary",
    ),
}


def _score_artifact_sha256(scores: RoutedLadderTokenScores) -> str:
    digest = hashlib.sha256()
    digest.update(SCORE_SCHEMA_VERSION.encode("ascii") + b"\0")
    metadata = {
        "condition": scores.condition,
        "panel": scores.panel,
        "seed": scores.seed,
        "checkpoint_tokens": scores.checkpoint_tokens,
        "source_revision": scores.source_revision,
        "capture_manifest_sha256": scores.capture_manifest_sha256,
        "recomposition_sidecar_sha256": scores.recomposition_sidecar_sha256,
        "token_identity_sha256": scores.token_identity_sha256,
    }
    digest.update(bytes.fromhex(canonical_sha256(metadata)))
    digest.update(np.asarray(scores.sequence_id, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(scores.position, dtype="<u2").tobytes(order="C"))
    for endpoint in ENDPOINTS:
        digest.update(endpoint.encode("ascii") + b"\0")
        digest.update(
            np.asarray(scores.endpoint_sse[endpoint], dtype="<f8").tobytes(
                order="C"
            )
        )
        digest.update(
            np.asarray(scores.endpoint_sst[endpoint], dtype="<f8").tobytes(
                order="C"
            )
        )
    return digest.hexdigest()


def _h_score_artifact_sha256(scores: RoutedLadderHTokenScores) -> str:
    digest = hashlib.sha256()
    digest.update(SCORE_SCHEMA_VERSION.encode("ascii") + b"\0")
    metadata = {
        "condition": "H",
        "panel": scores.panel,
        "seed": scores.seed,
        "checkpoint_tokens": scores.checkpoint_tokens,
        "source_revision": scores.source_revision,
        "capture_manifest_sha256": scores.capture_manifest_sha256,
        "recomposition_sidecar_sha256": scores.recomposition_sidecar_sha256,
        "token_identity_sha256": scores.token_identity_sha256,
    }
    digest.update(bytes.fromhex(canonical_sha256(metadata)))
    digest.update(np.asarray(scores.sequence_id, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(scores.position, dtype="<u2").tobytes(order="C"))
    for endpoint in H_ENDPOINTS:
        digest.update(endpoint.encode("ascii") + b"\0")
        digest.update(
            np.asarray(scores.endpoint_sse[endpoint], dtype="<f8").tobytes(
                order="C"
            )
        )
        digest.update(
            np.asarray(scores.endpoint_sst[endpoint], dtype="<f8").tobytes(
                order="C"
            )
        )
    return digest.hexdigest()


def _endpoint_sst_sha256(
    *,
    token_identity: str,
    endpoint: str,
    values: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        b"gemma4-experiment5-routed-activation-ladder-common-sst-v2\0"
    )
    digest.update(bytes.fromhex(token_identity))
    digest.update(endpoint.encode("ascii") + b"\0")
    digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RoutedLadderTokenScores:
    """Per-token sufficient scores for one routed-only A condition."""

    condition: Literal["A256ReLU", "A512ReLU", "A512Softplus"]
    panel: Literal["sae", "transcoder"]
    seed: int
    checkpoint_tokens: int
    source_revision: str
    capture_manifest_sha256: str
    recomposition_sidecar_sha256: str
    sequence_id: np.ndarray
    position: np.ndarray
    endpoint_sse: Mapping[str, np.ndarray]
    endpoint_sst: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if self.condition not in SCORED_CONDITIONS:
            raise Experiment5RoutedLadderError(
                "token scores must be A256ReLU, A512ReLU, or A512Softplus"
            )
        if self.panel not in PANELS:
            raise Experiment5RoutedLadderError("score panel is invalid")
        _nonnegative_int("score seed", self.seed)
        _nonnegative_int(
            "score checkpoint_tokens",
            self.checkpoint_tokens,
            positive=True,
        )
        _git_sha("score source_revision", self.source_revision)
        _sha256("score capture_manifest_sha256", self.capture_manifest_sha256)
        _sha256(
            "score recomposition_sidecar_sha256",
            self.recomposition_sidecar_sha256,
        )

        sequence = np.asarray(self.sequence_id)
        position = np.asarray(self.position)
        if (
            sequence.ndim != 1
            or not sequence.size
            or not np.issubdtype(sequence.dtype, np.signedinteger)
            or np.any(sequence < 0)
        ):
            raise Experiment5RoutedLadderError(
                "sequence_id must be a nonempty nonnegative signed-integer vector"
            )
        if (
            position.shape != sequence.shape
            or not np.issubdtype(position.dtype, np.integer)
            or np.any(position < 0)
            or np.any(position >= SEQUENCE_LENGTH)
        ):
            raise Experiment5RoutedLadderError(
                "position must align with sequence_id and lie in [0,1023]"
            )
        canonical_sequence = _readonly(sequence, dtype=np.dtype("<i8"))
        canonical_position = _readonly(position, dtype=np.dtype("<u2"))
        token_keys = np.empty(
            canonical_sequence.size,
            dtype=np.dtype([("sequence_id", "<i8"), ("position", "<u2")]),
        )
        token_keys["sequence_id"] = canonical_sequence
        token_keys["position"] = canonical_position
        if np.unique(token_keys).size != canonical_sequence.size:
            raise Experiment5RoutedLadderError(
                "token scores contain duplicate scientific token identities"
            )
        if np.unique(canonical_sequence).size < 2:
            raise Experiment5RoutedLadderError(
                "paired bootstrap requires at least two packed sequences"
            )

        if not isinstance(self.endpoint_sse, Mapping) or set(
            self.endpoint_sse
        ) != set(ENDPOINTS):
            raise Experiment5RoutedLadderError(
                "endpoint_sse must contain component/routed/oracle_shared"
            )
        if not isinstance(self.endpoint_sst, Mapping) or set(
            self.endpoint_sst
        ) != set(ENDPOINTS):
            raise Experiment5RoutedLadderError(
                "endpoint_sst must contain component/routed/oracle_shared"
            )
        endpoint_sse: dict[str, np.ndarray] = {}
        endpoint_sst: dict[str, np.ndarray] = {}
        unique_sequence, inverse = np.unique(
            canonical_sequence,
            return_inverse=True,
        )
        for endpoint in ENDPOINTS:
            raw_sse = np.asarray(self.endpoint_sse[endpoint])
            raw_sst = np.asarray(self.endpoint_sst[endpoint])
            for kind, raw in (("SSE", raw_sse), ("SST", raw_sst)):
                if (
                    raw.shape != canonical_sequence.shape
                    or not np.issubdtype(raw.dtype, np.floating)
                    or not np.all(np.isfinite(raw))
                    or np.any(raw < 0)
                ):
                    raise Experiment5RoutedLadderError(
                        f"{endpoint} token {kind} must be aligned finite "
                        "nonnegative floating-point values"
                    )
            selected_sse = _readonly(raw_sse, dtype=np.dtype("<f8"))
            selected_sst = _readonly(raw_sst, dtype=np.dtype("<f8"))
            clustered_sst = np.bincount(
                inverse,
                weights=selected_sst,
                minlength=unique_sequence.size,
            )
            if np.any(clustered_sst <= 0):
                raise Experiment5RoutedLadderError(
                    f"{endpoint} SST must be positive in every packed sequence"
                )
            endpoint_sse[endpoint] = selected_sse
            endpoint_sst[endpoint] = selected_sst

        object.__setattr__(self, "sequence_id", canonical_sequence)
        object.__setattr__(self, "position", canonical_position)
        object.__setattr__(self, "endpoint_sse", endpoint_sse)
        object.__setattr__(self, "endpoint_sst", endpoint_sst)

    @classmethod
    def from_value(
        cls,
        value: RoutedLadderTokenScores | Mapping[str, Any],
    ) -> RoutedLadderTokenScores:
        if isinstance(value, cls):
            return value
        fields = {field.name for field in dataclasses.fields(cls)}
        selected = _exact_mapping("routed-ladder token scores", value, fields)
        return cls(**dict(selected))

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.panel, self.seed, self.checkpoint_tokens

    @property
    def token_identity_sha256(self) -> str:
        return token_identity_sha256(self.sequence_id, self.position)

    @property
    def score_artifact_sha256(self) -> str:
        return _score_artifact_sha256(self)

    def endpoint_sst_sha256(self, endpoint: str) -> str:
        if endpoint not in ENDPOINTS:
            raise Experiment5RoutedLadderError("SST endpoint is invalid")
        return _endpoint_sst_sha256(
            token_identity=self.token_identity_sha256,
            endpoint=endpoint,
            values=self.endpoint_sst[endpoint],
        )

    @property
    def tokens(self) -> int:
        return int(self.sequence_id.size)

    @property
    def packed_sequences(self) -> int:
        return int(np.unique(self.sequence_id).size)

    def aggregate(self, endpoint: str) -> dict[str, float]:
        if endpoint not in ENDPOINTS:
            raise Experiment5RoutedLadderError("aggregate endpoint is invalid")
        sse = float(np.sum(self.endpoint_sse[endpoint], dtype=np.float64))
        sst = float(np.sum(self.endpoint_sst[endpoint], dtype=np.float64))
        if sst <= 0:
            raise Experiment5RoutedLadderError("aggregate SST must be positive")
        nmse = sse / sst
        return {"sse": sse, "sst": sst, "nmse": nmse, "ev": 1.0 - nmse}


@dataclass(frozen=True, slots=True)
class RoutedLadderHTokenScores:
    """Extension-owned per-token H evidence, including learned full output.

    The first three endpoints share the exact A activation-ladder targets. The
    learned full-FFN endpoint uses the same full-output target and therefore
    must have token-for-token identical SST rows to ``oracle_shared``.  This
    class is deliberately separate from :class:`RoutedLadderTokenScores` so
    frozen routed-A paired contrasts cannot silently start treating H as a
    randomized arm.
    """

    panel: Literal["sae", "transcoder"]
    seed: int
    checkpoint_tokens: int
    source_revision: str
    capture_manifest_sha256: str
    recomposition_sidecar_sha256: str
    sequence_id: np.ndarray
    position: np.ndarray
    endpoint_sse: Mapping[str, np.ndarray]
    endpoint_sst: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.endpoint_sse, Mapping)
            or set(self.endpoint_sse) != set(H_ENDPOINTS)
            or not isinstance(self.endpoint_sst, Mapping)
            or set(self.endpoint_sst) != set(H_ENDPOINTS)
        ):
            raise Experiment5RoutedLadderError(
                "H token scores require component/routed/oracle_shared/"
                "learned_full_ffn SSE and SST"
            )
        # Reuse the fully audited A row/key and three-endpoint validator rather
        # than maintaining a subtly different common-target implementation.
        routed = RoutedLadderTokenScores(
            condition="A512Softplus",
            panel=self.panel,
            seed=self.seed,
            checkpoint_tokens=self.checkpoint_tokens,
            source_revision=self.source_revision,
            capture_manifest_sha256=self.capture_manifest_sha256,
            recomposition_sidecar_sha256=self.recomposition_sidecar_sha256,
            sequence_id=self.sequence_id,
            position=self.position,
            endpoint_sse={
                endpoint: self.endpoint_sse[endpoint]
                for endpoint in ENDPOINTS
            },
            endpoint_sst={
                endpoint: self.endpoint_sst[endpoint]
                for endpoint in ENDPOINTS
            },
        )
        learned_sse = np.asarray(self.endpoint_sse["learned_full_ffn"])
        learned_sst = np.asarray(self.endpoint_sst["learned_full_ffn"])
        for name, raw in (
            ("SSE", learned_sse),
            ("SST", learned_sst),
        ):
            if (
                raw.shape != routed.sequence_id.shape
                or not np.issubdtype(raw.dtype, np.floating)
                or not np.all(np.isfinite(raw))
                or np.any(raw < 0)
            ):
                raise Experiment5RoutedLadderError(
                    f"learned_full_ffn token {name} must be aligned finite "
                    "nonnegative floating-point values"
                )
        selected_sse = _readonly(learned_sse, dtype=np.dtype("<f8"))
        selected_sst = _readonly(learned_sst, dtype=np.dtype("<f8"))
        if not np.array_equal(
            selected_sst,
            routed.endpoint_sst["oracle_shared"],
        ):
            raise Experiment5RoutedLadderError(
                "H learned and oracle full-FFN token SST rows differ"
            )
        sequence, inverse = np.unique(
            routed.sequence_id,
            return_inverse=True,
        )
        if np.any(
            np.bincount(
                inverse,
                weights=selected_sst,
                minlength=sequence.size,
            )
            <= 0
        ):
            raise Experiment5RoutedLadderError(
                "H learned full-FFN SST must be positive in every sequence"
            )
        object.__setattr__(self, "sequence_id", routed.sequence_id)
        object.__setattr__(self, "position", routed.position)
        object.__setattr__(
            self,
            "endpoint_sse",
            {
                **dict(routed.endpoint_sse),
                "learned_full_ffn": selected_sse,
            },
        )
        object.__setattr__(
            self,
            "endpoint_sst",
            {
                **dict(routed.endpoint_sst),
                "learned_full_ffn": selected_sst,
            },
        )

    @classmethod
    def from_value(
        cls,
        value: RoutedLadderHTokenScores | Mapping[str, Any],
    ) -> RoutedLadderHTokenScores:
        if isinstance(value, cls):
            return value
        fields = {field.name for field in dataclasses.fields(cls)}
        selected = _exact_mapping("routed-ladder H token scores", value, fields)
        return cls(**dict(selected))

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.panel, self.seed, self.checkpoint_tokens

    @property
    def condition(self) -> Literal["H"]:
        return "H"

    @property
    def token_identity_sha256(self) -> str:
        return token_identity_sha256(self.sequence_id, self.position)

    @property
    def score_artifact_sha256(self) -> str:
        return _h_score_artifact_sha256(self)

    def endpoint_sst_sha256(self, endpoint: str) -> str:
        if endpoint not in H_ENDPOINTS:
            raise Experiment5RoutedLadderError("H SST endpoint is invalid")
        # Learned and oracle shared reconstruct the same full-FFN target.  The
        # proof names the canonical target endpoint, not the estimator.
        proof_endpoint = (
            "oracle_shared"
            if endpoint == "learned_full_ffn"
            else endpoint
        )
        return _endpoint_sst_sha256(
            token_identity=self.token_identity_sha256,
            endpoint=proof_endpoint,
            values=self.endpoint_sst[endpoint],
        )

    @property
    def tokens(self) -> int:
        return int(self.sequence_id.size)

    @property
    def packed_sequences(self) -> int:
        return int(np.unique(self.sequence_id).size)

    def aggregate(self, endpoint: str) -> dict[str, float]:
        if endpoint not in H_ENDPOINTS:
            raise Experiment5RoutedLadderError("H aggregate endpoint is invalid")
        sse = float(np.sum(self.endpoint_sse[endpoint], dtype=np.float64))
        sst = float(np.sum(self.endpoint_sst[endpoint], dtype=np.float64))
        if sst <= 0:
            raise Experiment5RoutedLadderError("H aggregate SST must be positive")
        nmse = sse / sst
        return {"sse": sse, "sst": sst, "nmse": nmse, "ev": 1.0 - nmse}


@dataclass(frozen=True, slots=True)
class ImmutableArtifactReference:
    """A locator whose SHA-256, rather than mutable location, is authoritative."""

    kind: str
    schema_version: str
    uri: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or _KIND_RE.fullmatch(self.kind) is None:
            raise Experiment5RoutedLadderError("artifact kind is invalid")
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise Experiment5RoutedLadderError(
                "artifact schema_version must be nonempty"
            )
        if not isinstance(self.uri, str) or not self.uri.strip():
            raise Experiment5RoutedLadderError("artifact URI must be nonempty")
        _sha256("artifact sha256", self.sha256)

    @classmethod
    def from_value(
        cls,
        value: ImmutableArtifactReference | Mapping[str, Any],
    ) -> ImmutableArtifactReference:
        if isinstance(value, cls):
            return value
        selected = _exact_mapping(
            "immutable artifact reference",
            value,
            {"kind", "schema_version", "uri", "sha256"},
        )
        return cls(**dict(selected))

    def report_row(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class RoutedLadderTrainingCost:
    """Measured cumulative training cost for one 10M-token trajectory."""

    resource_id: str
    accelerator_type: str
    chip_count: int
    marginal_wall_seconds: float
    chip_seconds: float
    chip_seconds_scope: str
    chip_seconds_excludes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("resource_id", self.resource_id),
            ("accelerator_type", self.accelerator_type),
            ("chip_seconds_scope", self.chip_seconds_scope),
        ):
            if (
                not isinstance(value, str)
                or not value.strip()
                or "\x00" in value
            ):
                raise Experiment5RoutedLadderError(
                    f"training cost {name} must be nonempty"
                )
        chips = _nonnegative_int(
            "training cost chip_count",
            self.chip_count,
            positive=True,
        )
        wall = _finite(
            "training cost marginal_wall_seconds",
            self.marginal_wall_seconds,
            positive=True,
        )
        chip_seconds = _finite(
            "training cost chip_seconds",
            self.chip_seconds,
            positive=True,
        )
        if not math.isclose(
            chip_seconds,
            chips * wall,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise Experiment5RoutedLadderError(
                "training cost chip_seconds must equal chip_count times "
                "marginal_wall_seconds"
            )
        if (
            not isinstance(self.chip_seconds_excludes, tuple)
            or not self.chip_seconds_excludes
            or any(
                not isinstance(value, str)
                or not value
                or "\x00" in value
                for value in self.chip_seconds_excludes
            )
            or len(set(self.chip_seconds_excludes))
            != len(self.chip_seconds_excludes)
        ):
            raise Experiment5RoutedLadderError(
                "training cost exclusions must be unique nonempty strings"
            )

    @classmethod
    def from_value(
        cls,
        value: RoutedLadderTrainingCost | Mapping[str, Any],
    ) -> RoutedLadderTrainingCost:
        if isinstance(value, cls):
            return value
        selected = _exact_mapping(
            "training cost",
            value,
            {
                "resource_id",
                "accelerator_type",
                "chip_count",
                "marginal_wall_seconds",
                "chip_seconds",
                "chip_seconds_scope",
                "chip_seconds_excludes",
            },
        )
        exclusions = selected["chip_seconds_excludes"]
        if not isinstance(exclusions, Sequence) or isinstance(
            exclusions,
            (str, bytes),
        ):
            raise Experiment5RoutedLadderError(
                "training cost exclusions must be a sequence"
            )
        return cls(
            resource_id=selected["resource_id"],
            accelerator_type=selected["accelerator_type"],
            chip_count=selected["chip_count"],
            marginal_wall_seconds=selected["marginal_wall_seconds"],
            chip_seconds=selected["chip_seconds"],
            chip_seconds_scope=selected["chip_seconds_scope"],
            chip_seconds_excludes=tuple(exclusions),
        )

    def report_row(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "accelerator_type": self.accelerator_type,
            "chip_count": self.chip_count,
            "marginal_wall_seconds": self.marginal_wall_seconds,
            "chip_seconds": self.chip_seconds,
            "chip_seconds_scope": self.chip_seconds_scope,
            "chip_seconds_excludes": list(self.chip_seconds_excludes),
        }


@dataclass(frozen=True, slots=True)
class RoutedLadderTrajectoryAccounting:
    """Explicit immutable origin, parent, hardware, and cost for one trajectory."""

    panel: Literal["sae", "transcoder"]
    condition: Literal["A256ReLU", "A512ReLU", "A512Softplus", "H"]
    seed: int
    origin: str
    training_source_revision: str
    reference_manifest_kind: str | None
    reference_manifest_sha256: str | None
    parent_training_result_sha256: str
    parent_training_result_file_sha256: str
    cost_status: str
    training_cost: RoutedLadderTrainingCost | None

    def __post_init__(self) -> None:
        if self.panel not in PANELS or self.condition not in CONDITIONS:
            raise Experiment5RoutedLadderError(
                "trajectory accounting identity is invalid"
            )
        if self.seed not in SEEDS:
            raise Experiment5RoutedLadderError(
                "trajectory accounting seed is invalid"
            )
        _git_sha(
            "trajectory accounting training_source_revision",
            self.training_source_revision,
        )
        for name in (
            "parent_training_result_sha256",
            "parent_training_result_file_sha256",
        ):
            _sha256(f"trajectory accounting {name}", getattr(self, name))
        if self.origin not in _TRAJECTORY_ORIGINS:
            raise Experiment5RoutedLadderError(
                "trajectory accounting origin is invalid"
            )
        if self.cost_status not in _COST_STATUSES:
            raise Experiment5RoutedLadderError(
                "trajectory accounting cost status is invalid"
            )
        historical_a256 = (
            self.panel == "sae"
            and self.condition == "A256ReLU"
            and self.seed == 0
        )
        if historical_a256:
            expected_origin = "historical_experiment4_seed0_reference"
            expected_manifest_kind = "a256_reference_manifest"
            expected_cost_status = "measured_historical_training_result"
        elif self.condition == "H":
            expected_origin = "historical_experiment5_reference"
            expected_manifest_kind = "h_reference_manifest"
            expected_cost_status = None
        else:
            expected_origin = "protocol_v2_new_training"
            expected_manifest_kind = None
            expected_cost_status = "measured_v2_training_result"
        if (
            self.origin != expected_origin
            or self.reference_manifest_kind != expected_manifest_kind
        ):
            raise Experiment5RoutedLadderError(
                "trajectory accounting origin or reference manifest differs "
                "from its explicit role"
            )
        if expected_manifest_kind is None:
            if self.reference_manifest_sha256 is not None:
                raise Experiment5RoutedLadderError(
                    "new v2 trajectory cannot claim a historical manifest"
                )
        else:
            _sha256(
                "trajectory accounting reference_manifest_sha256",
                self.reference_manifest_sha256,
            )
        if self.condition == "H":
            allowed = {
                "measured_historical_training_result",
                "not_supplied_by_h_reference_manifest",
            }
            if self.cost_status not in allowed:
                raise Experiment5RoutedLadderError(
                    "H historical cost status is invalid"
                )
        elif self.cost_status != expected_cost_status:
            raise Experiment5RoutedLadderError(
                "trajectory accounting cost status differs from its origin"
            )
        cost = (
            None
            if self.training_cost is None
            else RoutedLadderTrainingCost.from_value(self.training_cost)
        )
        if (
            self.cost_status == "not_supplied_by_h_reference_manifest"
        ) is not (cost is None):
            raise Experiment5RoutedLadderError(
                "trajectory cost presence differs from its cost status"
            )
        object.__setattr__(self, "training_cost", cost)

    @classmethod
    def from_value(
        cls,
        value: RoutedLadderTrajectoryAccounting | Mapping[str, Any],
    ) -> RoutedLadderTrajectoryAccounting:
        if isinstance(value, cls):
            return value
        selected = _exact_mapping(
            "trajectory accounting",
            value,
            {
                "panel",
                "condition",
                "seed",
                "origin",
                "training_source_revision",
                "reference_manifest_kind",
                "reference_manifest_sha256",
                "parent_training_result_sha256",
                "parent_training_result_file_sha256",
                "cost_status",
                "training_cost",
                "trajectory_accounting_sha256",
            },
        )
        unsigned = {
            name: selected[name]
            for name in selected
            if name != "trajectory_accounting_sha256"
        }
        if canonical_sha256(unsigned) != _sha256(
            "trajectory accounting self-hash",
            selected["trajectory_accounting_sha256"],
        ):
            raise Experiment5RoutedLadderError(
                "trajectory accounting self-hash is invalid"
            )
        cost = selected["training_cost"]
        return cls(
            panel=selected["panel"],
            condition=selected["condition"],
            seed=selected["seed"],
            origin=selected["origin"],
            training_source_revision=selected["training_source_revision"],
            reference_manifest_kind=selected["reference_manifest_kind"],
            reference_manifest_sha256=selected[
                "reference_manifest_sha256"
            ],
            parent_training_result_sha256=selected[
                "parent_training_result_sha256"
            ],
            parent_training_result_file_sha256=selected[
                "parent_training_result_file_sha256"
            ],
            cost_status=selected["cost_status"],
            training_cost=(
                None
                if cost is None
                else RoutedLadderTrainingCost.from_value(cost)
            ),
        )

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.panel, self.condition, self.seed

    def report_row(self) -> dict[str, Any]:
        body = {
            "panel": self.panel,
            "condition": self.condition,
            "seed": self.seed,
            "origin": self.origin,
            "training_source_revision": self.training_source_revision,
            "reference_manifest_kind": self.reference_manifest_kind,
            "reference_manifest_sha256": self.reference_manifest_sha256,
            "parent_training_result_sha256": (
                self.parent_training_result_sha256
            ),
            "parent_training_result_file_sha256": (
                self.parent_training_result_file_sha256
            ),
            "cost_status": self.cost_status,
            "training_cost": (
                None
                if self.training_cost is None
                else self.training_cost.report_row()
            ),
        }
        return {
            **body,
            "trajectory_accounting_sha256": canonical_sha256(body),
        }


@dataclass(frozen=True, slots=True)
class RoutedLadderEvaluationBundleAccounting:
    """Hardware and successful-attempt timing for one resident eval bundle."""

    panel: Literal["sae", "transcoder"]
    seed: int
    source_revision: str
    operator_manifest_sha256: str
    bundle_spec_sha256: str
    bundle_result_sha256: str
    bundle_result_file_sha256: str
    accelerator_type: str
    chip_count: int
    successful_attempt_wall_seconds: float
    successful_attempt_chip_seconds: float
    timing_scope: str = _EVALUATION_TIMING_SCOPE

    def __post_init__(self) -> None:
        if self.panel not in PANELS or self.seed not in SEEDS:
            raise Experiment5RoutedLadderError(
                "evaluation bundle accounting identity is invalid"
            )
        _git_sha(
            "evaluation bundle source_revision",
            self.source_revision,
        )
        for name in (
            "operator_manifest_sha256",
            "bundle_spec_sha256",
            "bundle_result_sha256",
            "bundle_result_file_sha256",
        ):
            _sha256(f"evaluation bundle {name}", getattr(self, name))
        if (
            not isinstance(self.accelerator_type, str)
            or not self.accelerator_type
            or "\x00" in self.accelerator_type
        ):
            raise Experiment5RoutedLadderError(
                "evaluation bundle accelerator_type must be nonempty"
            )
        chips = _nonnegative_int(
            "evaluation bundle chip_count",
            self.chip_count,
            positive=True,
        )
        wall = _finite(
            "evaluation bundle successful_attempt_wall_seconds",
            self.successful_attempt_wall_seconds,
            positive=True,
        )
        chip_seconds = _finite(
            "evaluation bundle successful_attempt_chip_seconds",
            self.successful_attempt_chip_seconds,
            positive=True,
        )
        if not math.isclose(
            chip_seconds,
            chips * wall,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise Experiment5RoutedLadderError(
                "evaluation bundle chip-seconds do not equal wall time times "
                "chips"
            )
        if self.timing_scope != _EVALUATION_TIMING_SCOPE:
            raise Experiment5RoutedLadderError(
                "evaluation bundle timing scope changed"
            )

    @classmethod
    def from_value(
        cls,
        value: RoutedLadderEvaluationBundleAccounting | Mapping[str, Any],
    ) -> RoutedLadderEvaluationBundleAccounting:
        if isinstance(value, cls):
            return value
        selected = _exact_mapping(
            "evaluation bundle accounting",
            value,
            {
                "panel",
                "seed",
                "source_revision",
                "operator_manifest_sha256",
                "bundle_spec_sha256",
                "bundle_result_sha256",
                "bundle_result_file_sha256",
                "accelerator_type",
                "chip_count",
                "successful_attempt_wall_seconds",
                "successful_attempt_chip_seconds",
                "timing_scope",
                "bundle_accounting_sha256",
            },
        )
        unsigned = {
            name: selected[name]
            for name in selected
            if name != "bundle_accounting_sha256"
        }
        if canonical_sha256(unsigned) != _sha256(
            "evaluation bundle accounting self-hash",
            selected["bundle_accounting_sha256"],
        ):
            raise Experiment5RoutedLadderError(
                "evaluation bundle accounting self-hash is invalid"
            )
        return cls(
            **{
                name: selected[name]
                for name in selected
                if name != "bundle_accounting_sha256"
            }
        )

    @property
    def identity(self) -> tuple[str, int]:
        return self.panel, self.seed

    def report_row(self) -> dict[str, Any]:
        body = dataclasses.asdict(self)
        return {
            **body,
            "bundle_accounting_sha256": canonical_sha256(body),
        }


@dataclass(frozen=True, slots=True)
class RoutedLadderWallCashAccounting:
    """Campaign elapsed wall time and cash projection, kept separate from FLOPs."""

    campaign_wall_seconds: float
    wall_scope: str
    cash_usd: float
    cash_currency: str
    cash_basis: str
    accounting_source: ImmutableArtifactReference

    def __post_init__(self) -> None:
        _finite(
            "campaign wall seconds",
            self.campaign_wall_seconds,
            nonnegative=True,
        )
        _finite("campaign cash USD", self.cash_usd, nonnegative=True)
        if self.wall_scope != _CAMPAIGN_WALL_SCOPE:
            raise Experiment5RoutedLadderError(
                "campaign wall accounting scope changed"
            )
        if self.cash_currency != "USD":
            raise Experiment5RoutedLadderError(
                "campaign cash currency must be USD"
            )
        if (
            not isinstance(self.cash_basis, str)
            or not self.cash_basis
            or "\x00" in self.cash_basis
        ):
            raise Experiment5RoutedLadderError(
                "campaign cash basis must be nonempty"
            )
        source = ImmutableArtifactReference.from_value(self.accounting_source)
        if source.kind != "cost_projection":
            raise Experiment5RoutedLadderError(
                "campaign wall/cash source must be a cost projection"
            )
        object.__setattr__(self, "accounting_source", source)

    @classmethod
    def from_value(
        cls,
        value: RoutedLadderWallCashAccounting | Mapping[str, Any],
    ) -> RoutedLadderWallCashAccounting:
        if isinstance(value, cls):
            return value
        selected = _exact_mapping(
            "wall/cash accounting",
            value,
            {
                "campaign_wall_seconds",
                "wall_scope",
                "cash_usd",
                "cash_currency",
                "cash_basis",
                "accounting_source",
            },
        )
        return cls(
            campaign_wall_seconds=selected["campaign_wall_seconds"],
            wall_scope=selected["wall_scope"],
            cash_usd=selected["cash_usd"],
            cash_currency=selected["cash_currency"],
            cash_basis=selected["cash_basis"],
            accounting_source=ImmutableArtifactReference.from_value(
                selected["accounting_source"]
            ),
        )

    def report_row(self) -> dict[str, Any]:
        return {
            "campaign_wall_seconds": self.campaign_wall_seconds,
            "wall_scope": self.wall_scope,
            "cash_usd": self.cash_usd,
            "cash_currency": self.cash_currency,
            "cash_basis": self.cash_basis,
            "accounting_source": self.accounting_source.report_row(),
        }


@dataclass(frozen=True, slots=True)
class ExistingHReferences:
    """Hash-pinned existing H result, point metrics, and diagnostics."""

    panel: Literal["sae", "transcoder"]
    seed: int
    checkpoint_tokens: int
    source_revision: str
    training_source_revision: str
    h_config_sha256: str
    training_result_sha256: str
    training_result_file_sha256: str
    checkpoint_pointer_sha256: str
    checkpoint_pointer_file_sha256: str
    score_artifact_sha256: str
    token_identity_sha256: str
    evaluation_tokens: int
    packed_sequences: int
    endpoint_scores: Mapping[str, Mapping[str, float] | None]
    endpoint_sst_sha256: Mapping[str, str | None]
    scientific_result: ImmutableArtifactReference
    diagnostics: tuple[ImmutableArtifactReference, ...]

    def __post_init__(self) -> None:
        if self.panel not in PANELS:
            raise Experiment5RoutedLadderError("H reference panel is invalid")
        _nonnegative_int("H reference seed", self.seed)
        _nonnegative_int(
            "H reference checkpoint_tokens",
            self.checkpoint_tokens,
            positive=True,
        )
        _git_sha("H reference source_revision", self.source_revision)
        _git_sha(
            "H reference training_source_revision",
            self.training_source_revision,
        )
        for name in (
            "h_config_sha256",
            "training_result_sha256",
            "training_result_file_sha256",
            "checkpoint_pointer_sha256",
            "checkpoint_pointer_file_sha256",
            "score_artifact_sha256",
        ):
            _sha256(f"H reference {name}", getattr(self, name))
        _sha256(
            "H reference token_identity_sha256",
            self.token_identity_sha256,
        )
        evaluation_tokens = _nonnegative_int(
            "H reference evaluation_tokens",
            self.evaluation_tokens,
            positive=True,
        )
        packed_sequences = _nonnegative_int(
            "H reference packed_sequences",
            self.packed_sequences,
            positive=True,
        )
        if packed_sequences > evaluation_tokens:
            raise Experiment5RoutedLadderError(
                "H reference has more sequences than tokens"
            )
        if not isinstance(self.endpoint_scores, Mapping) or set(
            self.endpoint_scores
        ) != set(H_ENDPOINTS):
            raise Experiment5RoutedLadderError(
                "H endpoint scores must contain the exact H endpoint set"
            )
        if not isinstance(self.endpoint_sst_sha256, Mapping) or set(
            self.endpoint_sst_sha256
        ) != set(H_ENDPOINTS):
            raise Experiment5RoutedLadderError(
                "H SST proofs must contain the exact H endpoint set"
            )
        endpoint_scores: dict[str, dict[str, float] | None] = {}
        endpoint_sst: dict[str, str | None] = {}
        for endpoint in H_ENDPOINTS:
            score = self.endpoint_scores[endpoint]
            proof = self.endpoint_sst_sha256[endpoint]
            if endpoint == "learned_full_ffn" and score is None and proof is None:
                endpoint_scores[endpoint] = None
                endpoint_sst[endpoint] = None
                continue
            if score is None or proof is None:
                raise Experiment5RoutedLadderError(
                    f"H {endpoint} score and SST proof must both be present"
                )
            endpoint_scores[endpoint] = _validate_score_summary(
                f"H endpoint_scores.{endpoint}",
                score,
            )
            endpoint_sst[endpoint] = _sha256(
                f"H endpoint_sst_sha256.{endpoint}",
                proof,
            )
        learned = endpoint_scores["learned_full_ffn"]
        if learned is not None and (
            not math.isclose(
                learned["sst"],
                endpoint_scores["oracle_shared"]["sst"],  # type: ignore[index]
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or endpoint_sst["learned_full_ffn"]
            != endpoint_sst["oracle_shared"]
        ):
            raise Experiment5RoutedLadderError(
                "H learned and oracle full-FFN scores must use one common SST"
            )
        scientific = ImmutableArtifactReference.from_value(
            self.scientific_result
        )
        if scientific.kind != "scientific_result":
            raise Experiment5RoutedLadderError(
                "H scientific result reference has the wrong kind"
            )
        if (
            not isinstance(self.diagnostics, tuple)
            or not self.diagnostics
        ):
            raise Experiment5RoutedLadderError(
                "H references require at least one diagnostic artifact"
            )
        diagnostics = tuple(
            ImmutableArtifactReference.from_value(value)
            for value in self.diagnostics
        )
        kinds = [value.kind for value in diagnostics]
        if "scientific_result" in kinds or len(set(kinds)) != len(kinds):
            raise Experiment5RoutedLadderError(
                "H diagnostic artifact kinds must be unique and non-result"
            )
        object.__setattr__(self, "scientific_result", scientific)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "endpoint_scores", endpoint_scores)
        object.__setattr__(self, "endpoint_sst_sha256", endpoint_sst)

    @classmethod
    def from_value(
        cls,
        value: ExistingHReferences | Mapping[str, Any],
    ) -> ExistingHReferences:
        if isinstance(value, cls):
            return value
        selected = _exact_mapping(
            "existing H references",
            value,
            {
                "panel",
                "seed",
                "checkpoint_tokens",
                "source_revision",
                "training_source_revision",
                "h_config_sha256",
                "training_result_sha256",
                "training_result_file_sha256",
                "checkpoint_pointer_sha256",
                "checkpoint_pointer_file_sha256",
                "score_artifact_sha256",
                "token_identity_sha256",
                "evaluation_tokens",
                "packed_sequences",
                "endpoint_scores",
                "endpoint_sst_sha256",
                "scientific_result",
                "diagnostics",
            },
        )
        diagnostics = selected["diagnostics"]
        if not isinstance(diagnostics, Sequence) or isinstance(
            diagnostics,
            (str, bytes),
        ):
            raise Experiment5RoutedLadderError(
                "H diagnostics must be a sequence"
            )
        return cls(
            panel=selected["panel"],
            seed=selected["seed"],
            checkpoint_tokens=selected["checkpoint_tokens"],
            source_revision=selected["source_revision"],
            training_source_revision=selected["training_source_revision"],
            h_config_sha256=selected["h_config_sha256"],
            training_result_sha256=selected["training_result_sha256"],
            training_result_file_sha256=selected[
                "training_result_file_sha256"
            ],
            checkpoint_pointer_sha256=selected[
                "checkpoint_pointer_sha256"
            ],
            checkpoint_pointer_file_sha256=selected[
                "checkpoint_pointer_file_sha256"
            ],
            score_artifact_sha256=selected["score_artifact_sha256"],
            token_identity_sha256=selected["token_identity_sha256"],
            evaluation_tokens=selected["evaluation_tokens"],
            packed_sequences=selected["packed_sequences"],
            endpoint_scores=selected["endpoint_scores"],
            endpoint_sst_sha256=selected["endpoint_sst_sha256"],
            scientific_result=ImmutableArtifactReference.from_value(
                selected["scientific_result"]
            ),
            diagnostics=tuple(
                ImmutableArtifactReference.from_value(row)
                for row in diagnostics
            ),
        )

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.panel, self.seed, self.checkpoint_tokens

    def report_row(self) -> dict[str, Any]:
        return {
            "panel": self.panel,
            "seed": self.seed,
            "checkpoint_tokens": self.checkpoint_tokens,
            "condition": "H",
            "source_revision": self.source_revision,
            "training_source_revision": self.training_source_revision,
            "h_config_sha256": self.h_config_sha256,
            "training_result_sha256": self.training_result_sha256,
            "training_result_file_sha256": self.training_result_file_sha256,
            "checkpoint_pointer_sha256": self.checkpoint_pointer_sha256,
            "checkpoint_pointer_file_sha256": (
                self.checkpoint_pointer_file_sha256
            ),
            "score_artifact_sha256": self.score_artifact_sha256,
            "token_identity_sha256": self.token_identity_sha256,
            "evaluation_tokens": self.evaluation_tokens,
            "packed_sequences": self.packed_sequences,
            "encoder_activation": "softplus",
            "shared_treatment": "learned_sparse_dictionary",
            "endpoints": self.endpoint_scores,
            "endpoint_sst_sha256": self.endpoint_sst_sha256,
            "scientific_result": self.scientific_result.report_row(),
            "diagnostics": [row.report_row() for row in self.diagnostics],
        }


def h_token_scores_to_existing_reference(
    scores: RoutedLadderHTokenScores | Mapping[str, Any],
    *,
    training_source_revision: str,
    h_config_sha256: str,
    training_result_sha256: str,
    training_result_file_sha256: str,
    checkpoint_pointer_sha256: str,
    checkpoint_pointer_file_sha256: str,
    scientific_result: ImmutableArtifactReference | Mapping[str, Any],
    diagnostics: Sequence[
        ImmutableArtifactReference | Mapping[str, Any]
    ],
) -> ExistingHReferences:
    """Derive the report-compatible H summary from retained token rows.

    This is the only adapter from extension-owned H rescoring into the legacy
    point-reference boundary.  It computes every point and common-SST proof
    from the strict token envelope; aggregate-only Experiment-5 results cannot
    pass as routed-ladder evidence.
    """

    selected = RoutedLadderHTokenScores.from_value(scores)
    return ExistingHReferences(
        panel=selected.panel,
        seed=selected.seed,
        checkpoint_tokens=selected.checkpoint_tokens,
        source_revision=selected.source_revision,
        training_source_revision=training_source_revision,
        h_config_sha256=h_config_sha256,
        training_result_sha256=training_result_sha256,
        training_result_file_sha256=training_result_file_sha256,
        checkpoint_pointer_sha256=checkpoint_pointer_sha256,
        checkpoint_pointer_file_sha256=checkpoint_pointer_file_sha256,
        score_artifact_sha256=selected.score_artifact_sha256,
        token_identity_sha256=selected.token_identity_sha256,
        evaluation_tokens=selected.tokens,
        packed_sequences=selected.packed_sequences,
        endpoint_scores={
            endpoint: selected.aggregate(endpoint) for endpoint in H_ENDPOINTS
        },
        endpoint_sst_sha256={
            endpoint: selected.endpoint_sst_sha256(endpoint)
            for endpoint in H_ENDPOINTS
        },
        scientific_result=ImmutableArtifactReference.from_value(
            scientific_result
        ),
        diagnostics=tuple(
            ImmutableArtifactReference.from_value(value)
            for value in diagnostics
        ),
    )


def _validate_pair(
    left: RoutedLadderTokenScores,
    right: RoutedLadderTokenScores,
) -> str:
    if (left.condition, right.condition) not in PAIRED_CONTRASTS:
        raise Experiment5RoutedLadderError(
            "routed pair must be A256ReLU/A512ReLU or "
            "A512ReLU/A512Softplus"
        )
    contrast = f"{right.condition}-{left.condition}"
    if left.identity != right.identity:
        raise Experiment5RoutedLadderError(
            f"{contrast} curve identities differ"
        )
    if left.source_revision != right.source_revision:
        raise Experiment5RoutedLadderError(
            f"{contrast} producer revisions differ"
        )
    if (
        left.capture_manifest_sha256 != right.capture_manifest_sha256
        or left.recomposition_sidecar_sha256
        != right.recomposition_sidecar_sha256
    ):
        raise Experiment5RoutedLadderError(
            f"{contrast} scientific parents differ"
        )
    if (
        not np.array_equal(left.sequence_id, right.sequence_id)
        or not np.array_equal(left.position, right.position)
        or left.token_identity_sha256 != right.token_identity_sha256
    ):
        raise Experiment5RoutedLadderError(
            f"{contrast} token identities are not exactly aligned"
        )
    for endpoint in ENDPOINTS:
        if not np.array_equal(
            left.endpoint_sst[endpoint],
            right.endpoint_sst[endpoint],
        ):
            raise Experiment5RoutedLadderError(
                f"{contrast} {endpoint} target SST rows differ"
            )
    return contrast


def _validate_h_context(
    a512_softplus: RoutedLadderTokenScores,
    h_reference: ExistingHReferences,
) -> None:
    if h_reference.identity != a512_softplus.identity:
        raise Experiment5RoutedLadderError(
            "H and A512Softplus curve identities differ"
        )
    if (
        h_reference.token_identity_sha256 != a512_softplus.token_identity_sha256
        or h_reference.evaluation_tokens != a512_softplus.tokens
        or h_reference.packed_sequences != a512_softplus.packed_sequences
    ):
        raise Experiment5RoutedLadderError(
            "H and A512Softplus token identity proofs differ"
        )
    for endpoint in ENDPOINTS:
        h_score = h_reference.endpoint_scores[endpoint]
        if h_score is None:
            raise Experiment5RoutedLadderError(
                f"H {endpoint} point score is missing"
            )
        a_score = a512_softplus.aggregate(endpoint)
        if (
            not math.isclose(
                h_score["sst"],
                a_score["sst"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or h_reference.endpoint_sst_sha256[endpoint]
            != a512_softplus.endpoint_sst_sha256(endpoint)
        ):
            raise Experiment5RoutedLadderError(
                f"H and A512Softplus {endpoint} common SST proofs differ"
            )


def _descriptive_h_gaps(
    a512_softplus: RoutedLadderTokenScores,
    h_reference: ExistingHReferences,
) -> dict[str, Any]:
    h_oracle = h_reference.endpoint_scores["oracle_shared"]
    if h_oracle is None:
        raise Experiment5RoutedLadderError("H oracle-shared point is missing")
    a512_softplus_oracle = a512_softplus.aggregate("oracle_shared")
    learned = h_reference.endpoint_scores["learned_full_ffn"]
    return {
        "h_oracle_minus_a512_softplus_oracle": {
            "status": "descriptive_noncausal",
            "endpoint": "oracle_shared",
            "estimate_ev_difference": h_oracle["ev"] - a512_softplus_oracle["ev"],
            "paired_interval_available": False,
        },
        "h_oracle_minus_h_learned": (
            {
                "status": "descriptive_noncausal",
                "endpoint": "full_ffn_common_sst",
                "estimate_ev_difference": h_oracle["ev"] - learned["ev"],
                "paired_interval_available": False,
            }
            if learned is not None
            else {
                "status": "unavailable_learned_full_ffn_not_defined",
                "endpoint": "full_ffn_common_sst",
                "estimate_ev_difference": None,
                "paired_interval_available": False,
            }
        ),
    }


def _bootstrap_seed(
    left: RoutedLadderTokenScores,
    *,
    endpoint: str,
    contrast: str,
) -> int:
    payload = "|".join(
        (
            PROTOCOL_ID,
            left.panel,
            str(left.seed),
            str(left.checkpoint_tokens),
            endpoint,
            contrast,
            left.token_identity_sha256,
        )
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("ascii")).digest()[:8],
        "big",
        signed=False,
    )


def paired_routed_ladder_bootstrap(
    left_scores: RoutedLadderTokenScores | Mapping[str, Any],
    right_scores: RoutedLadderTokenScores | Mapping[str, Any],
    *,
    endpoint: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Compute a packed-sequence paired EV contrast from aligned token scores."""

    left = RoutedLadderTokenScores.from_value(left_scores)
    right = RoutedLadderTokenScores.from_value(right_scores)
    contrast = _validate_pair(left, right)
    if endpoint not in ENDPOINTS:
        raise Experiment5RoutedLadderError("bootstrap endpoint is invalid")
    selected_replicates = _nonnegative_int(
        "bootstrap replicates",
        replicates,
        positive=True,
    )
    selected_chunk = _nonnegative_int(
        "bootstrap chunk_size",
        chunk_size,
        positive=True,
    )

    sequence, inverse = np.unique(left.sequence_id, return_inverse=True)
    sequence_sst = np.bincount(
        inverse,
        weights=left.endpoint_sst[endpoint],
        minlength=sequence.size,
    )
    # EV_right - EV_left = (SSE_left - SSE_right) / common SST.
    token_delta_sse = (
        left.endpoint_sse[endpoint] - right.endpoint_sse[endpoint]
    )
    sequence_delta_sse = np.bincount(
        inverse,
        weights=token_delta_sse,
        minlength=sequence.size,
    )
    if np.any(sequence_sst <= 0):
        raise Experiment5RoutedLadderError(
            "bootstrap sequence SST must be positive"
        )
    estimate = float(
        sequence_delta_sse.sum(dtype=np.float64)
        / sequence_sst.sum(dtype=np.float64)
    )
    samples = np.empty(selected_replicates, dtype=np.float64)
    seed = _bootstrap_seed(left, endpoint=endpoint, contrast=contrast)
    if np.allclose(
        sequence_delta_sse,
        estimate * sequence_sst,
        rtol=1e-13,
        atol=1e-13,
    ):
        samples.fill(estimate)
    else:
        rng = np.random.Generator(np.random.PCG64(seed))
        for start in range(0, selected_replicates, selected_chunk):
            end = min(start + selected_chunk, selected_replicates)
            draws = rng.integers(
                0,
                sequence.size,
                size=(end - start, sequence.size),
                endpoint=False,
            )
            denominator = sequence_sst[draws].sum(axis=1, dtype=np.float64)
            if np.any(denominator <= 0):
                raise Experiment5RoutedLadderError(
                    "bootstrap replicate has nonpositive SST"
                )
            samples[start:end] = (
                sequence_delta_sse[draws].sum(axis=1, dtype=np.float64)
                / denominator
            )
    lower, upper = np.quantile(samples, (0.025, 0.975), method="linear")
    return {
        "contrast": contrast,
        "endpoint": endpoint,
        "estimate_ev_difference": estimate,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_unit": "packed_sequence",
        "bootstrap_algorithm": BOOTSTRAP_ALGORITHM,
        "bootstrap_replicates": selected_replicates,
        "bootstrap_seed": seed,
        "evaluation_tokens": left.tokens,
        "packed_sequences": left.packed_sequences,
        "token_identity_sha256": left.token_identity_sha256,
    }


def paired_a512_relu_minus_a256_relu_bootstrap(
    a256_relu: RoutedLadderTokenScores | Mapping[str, Any],
    a512_relu: RoutedLadderTokenScores | Mapping[str, Any],
    *,
    endpoint: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Paired EV effect of the frozen 256-to-512 routed scale step."""

    return paired_routed_ladder_bootstrap(
        a256_relu,
        a512_relu,
        endpoint=endpoint,
        replicates=replicates,
        chunk_size=chunk_size,
    )


def paired_a512_softplus_minus_a512_relu_bootstrap(
    a512_relu: RoutedLadderTokenScores | Mapping[str, Any],
    a512_softplus: RoutedLadderTokenScores | Mapping[str, Any],
    *,
    endpoint: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Paired EV effect of activation at fixed A512 geometry."""

    return paired_routed_ladder_bootstrap(
        a512_relu,
        a512_softplus,
        endpoint=endpoint,
        replicates=replicates,
        chunk_size=chunk_size,
    )


def _hierarchical_bootstrap_seed(
    *,
    panel: str,
    checkpoint_tokens: int,
    endpoint: str,
    contrast: str,
) -> int:
    payload = "|".join(
        (
            PROTOCOL_ID,
            panel,
            str(checkpoint_tokens),
            endpoint,
            contrast,
            "hierarchical",
        )
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("ascii")).digest()[:8],
        "big",
        signed=False,
    )


def hierarchical_routed_ladder_bootstrap(
    pairs: Sequence[
        tuple[
            RoutedLadderTokenScores | Mapping[str, Any],
            RoutedLadderTokenScores | Mapping[str, Any],
        ]
    ],
    *,
    endpoint: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Bootstrap the three-seed mean after resampling each seed's sequences.

    Each replicate first samples three seeds with replacement.  For every
    sampled seed occurrence, it then samples that seed's packed sequences with
    replacement and computes the common-SST EV difference.  The replicate is
    the arithmetic mean of those three independently resampled seed effects.
    """

    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise Experiment5RoutedLadderError(
            "hierarchical bootstrap pairs must be a sequence"
        )
    checked: list[tuple[RoutedLadderTokenScores, RoutedLadderTokenScores]] = []
    contrast: str | None = None
    for index, raw_pair in enumerate(pairs):
        if (
            not isinstance(raw_pair, Sequence)
            or isinstance(raw_pair, (str, bytes))
            or len(raw_pair) != 2
        ):
            raise Experiment5RoutedLadderError(
                f"hierarchical bootstrap pair {index} must contain "
                "one authorized routed-ladder pair"
            )
        left = RoutedLadderTokenScores.from_value(raw_pair[0])
        right = RoutedLadderTokenScores.from_value(raw_pair[1])
        selected_contrast = _validate_pair(left, right)
        if contrast is None:
            contrast = selected_contrast
        elif selected_contrast != contrast:
            raise Experiment5RoutedLadderError(
                "hierarchical bootstrap mixes routed-ladder contrasts"
            )
        checked.append((left, right))
    if contrast is None:
        raise Experiment5RoutedLadderError(
            "hierarchical bootstrap requires routed-ladder pairs"
        )
    if endpoint not in ENDPOINTS:
        raise Experiment5RoutedLadderError(
            "hierarchical bootstrap endpoint is invalid"
        )
    selected_replicates = _nonnegative_int(
        "hierarchical bootstrap replicates",
        replicates,
        positive=True,
    )
    selected_chunk = _nonnegative_int(
        "hierarchical bootstrap chunk_size",
        chunk_size,
        positive=True,
    )
    by_seed = {left.seed: (left, right) for left, right in checked}
    if len(checked) != len(SEEDS) or set(by_seed) != set(SEEDS):
        raise Experiment5RoutedLadderError(
            "hierarchical bootstrap requires exactly seeds 0, 1, and 2"
        )
    ordered = [by_seed[seed] for seed in SEEDS]
    first = ordered[0][0]
    for left, _right in ordered[1:]:
        if (
            left.panel != first.panel
            or left.checkpoint_tokens != first.checkpoint_tokens
            or left.source_revision != first.source_revision
            or left.capture_manifest_sha256 != first.capture_manifest_sha256
            or left.recomposition_sidecar_sha256
            != first.recomposition_sidecar_sha256
            or not np.array_equal(left.sequence_id, first.sequence_id)
            or not np.array_equal(left.position, first.position)
            or not np.array_equal(
                left.endpoint_sst[endpoint],
                first.endpoint_sst[endpoint],
            )
        ):
            raise Experiment5RoutedLadderError(
                "hierarchical bootstrap seed cohorts do not share one "
                "scientific token/SST parent"
            )

    sequence_sst: list[np.ndarray] = []
    sequence_delta_sse: list[np.ndarray] = []
    seed_effects: list[float] = []
    for left, right in ordered:
        sequence, inverse = np.unique(left.sequence_id, return_inverse=True)
        selected_sst = np.bincount(
            inverse,
            weights=left.endpoint_sst[endpoint],
            minlength=sequence.size,
        ).astype(np.float64)
        selected_delta = np.bincount(
            inverse,
            weights=(
                left.endpoint_sse[endpoint] - right.endpoint_sse[endpoint]
            ),
            minlength=sequence.size,
        ).astype(np.float64)
        if np.any(selected_sst <= 0):
            raise Experiment5RoutedLadderError(
                "hierarchical bootstrap sequence SST must be positive"
            )
        sequence_sst.append(selected_sst)
        sequence_delta_sse.append(selected_delta)
        seed_effects.append(
            float(
                selected_delta.sum(dtype=np.float64)
                / selected_sst.sum(dtype=np.float64)
            )
        )

    estimate = float(math.fsum(seed_effects) / len(SEEDS))
    seed = _hierarchical_bootstrap_seed(
        panel=first.panel,
        checkpoint_tokens=first.checkpoint_tokens,
        endpoint=endpoint,
        contrast=contrast,
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(selected_replicates, dtype=np.float64)
    for start in range(0, selected_replicates, selected_chunk):
        end = min(start + selected_chunk, selected_replicates)
        rows = end - start
        seed_draws = rng.integers(
            0,
            len(SEEDS),
            size=(rows, len(SEEDS)),
            endpoint=False,
        )
        replicate_sum = np.zeros((rows,), dtype=np.float64)
        for slot in range(len(SEEDS)):
            selected_seed = seed_draws[:, slot]
            slot_effect = np.empty((rows,), dtype=np.float64)
            for seed_index in range(len(SEEDS)):
                mask = selected_seed == seed_index
                count = int(np.count_nonzero(mask))
                if count == 0:
                    continue
                denominator_rows = sequence_sst[seed_index]
                delta_rows = sequence_delta_sse[seed_index]
                draws = rng.integers(
                    0,
                    denominator_rows.size,
                    size=(count, denominator_rows.size),
                    endpoint=False,
                )
                denominator = denominator_rows[draws].sum(
                    axis=1,
                    dtype=np.float64,
                )
                if np.any(denominator <= 0):
                    raise Experiment5RoutedLadderError(
                        "hierarchical bootstrap replicate has nonpositive SST"
                    )
                slot_effect[mask] = (
                    delta_rows[draws].sum(axis=1, dtype=np.float64)
                    / denominator
                )
            replicate_sum += slot_effect
        samples[start:end] = replicate_sum / len(SEEDS)
    lower, upper = np.quantile(samples, (0.025, 0.975), method="linear")
    return {
        "contrast": f"mean_seed({contrast})",
        "endpoint": endpoint,
        "estimate_mean_ev_difference": estimate,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_unit": "seed_then_packed_sequence",
        "bootstrap_algorithm": HIERARCHICAL_BOOTSTRAP_ALGORITHM,
        "bootstrap_replicates": selected_replicates,
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


def hierarchical_a512_relu_minus_a256_relu_bootstrap(
    pairs: Sequence[
        tuple[
            RoutedLadderTokenScores | Mapping[str, Any],
            RoutedLadderTokenScores | Mapping[str, Any],
        ]
    ],
    *,
    endpoint: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Three-seed hierarchical effect of the 256-to-512 scale step."""

    return hierarchical_routed_ladder_bootstrap(
        pairs,
        endpoint=endpoint,
        replicates=replicates,
        chunk_size=chunk_size,
    )


def hierarchical_a512_softplus_minus_a512_relu_bootstrap(
    pairs: Sequence[
        tuple[
            RoutedLadderTokenScores | Mapping[str, Any],
            RoutedLadderTokenScores | Mapping[str, Any],
        ]
    ],
    *,
    endpoint: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Three-seed hierarchical effect of activation at A512 geometry."""

    return hierarchical_routed_ladder_bootstrap(
        pairs,
        endpoint=endpoint,
        replicates=replicates,
        chunk_size=chunk_size,
    )


def _prepare_accounting(
    *,
    trajectory_accounting: Sequence[
        RoutedLadderTrajectoryAccounting | Mapping[str, Any]
    ],
    evaluation_bundle_accounting: Sequence[
        RoutedLadderEvaluationBundleAccounting | Mapping[str, Any]
    ],
    a256_reference_manifest_sha256: str,
    h_reference_manifest_sha256: str,
    wall_cash_accounting: RoutedLadderWallCashAccounting | Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str, int], RoutedLadderTrajectoryAccounting],
    dict[tuple[str, int], RoutedLadderEvaluationBundleAccounting],
    dict[str, Any],
]:
    """Validate and summarize the immutable compute/provenance inventory."""

    a256_manifest_sha = _sha256(
        "A256 reference manifest SHA-256",
        a256_reference_manifest_sha256,
    )
    h_manifest_sha = _sha256(
        "H reference manifest SHA-256",
        h_reference_manifest_sha256,
    )
    trajectories = [
        RoutedLadderTrajectoryAccounting.from_value(value)
        for value in trajectory_accounting
    ]
    trajectory_grid = {row.identity: row for row in trajectories}
    expected_trajectories = {
        (panel, condition, seed)
        for panel in PANELS
        for condition in CONDITIONS
        for seed in SEEDS
    }
    if (
        len(trajectory_grid) != len(trajectories)
        or set(trajectory_grid) != expected_trajectories
    ):
        raise Experiment5RoutedLadderError(
            "report requires the exact 24-trajectory accounting grid"
        )
    for row in trajectories:
        if (
            row.reference_manifest_kind == "a256_reference_manifest"
            and row.reference_manifest_sha256 != a256_manifest_sha
        ) or (
            row.reference_manifest_kind == "h_reference_manifest"
            and row.reference_manifest_sha256 != h_manifest_sha
        ):
            raise Experiment5RoutedLadderError(
                "trajectory reference manifest differs from report accounting"
            )
    new_rows = [
        row
        for row in trajectories
        if row.origin == "protocol_v2_new_training"
    ]
    if len(new_rows) != 17:
        raise Experiment5RoutedLadderError(
            "report accounting requires exactly 17 new v2 trajectories"
        )

    bundles = [
        RoutedLadderEvaluationBundleAccounting.from_value(value)
        for value in evaluation_bundle_accounting
    ]
    bundle_grid = {row.identity: row for row in bundles}
    expected_bundles = {
        (panel, seed) for panel in PANELS for seed in SEEDS
    }
    if (
        len(bundle_grid) != len(bundles)
        or set(bundle_grid) != expected_bundles
    ):
        raise Experiment5RoutedLadderError(
            "report accounting requires the exact six evaluation bundles"
        )
    wall_cash = RoutedLadderWallCashAccounting.from_value(
        wall_cash_accounting
    )
    new_training_chip_seconds = math.fsum(
        row.training_cost.chip_seconds  # type: ignore[union-attr]
        for row in new_rows
    )
    evaluation_chip_seconds = math.fsum(
        row.successful_attempt_chip_seconds for row in bundles
    )
    historical_rows = [
        row
        for row in trajectories
        if row.origin != "protocol_v2_new_training"
    ]
    known_historical_rows = [
        row for row in historical_rows if row.training_cost is not None
    ]
    missing_historical_rows = [
        row for row in historical_rows if row.training_cost is None
    ]
    historical_chip_seconds = math.fsum(
        row.training_cost.chip_seconds  # type: ignore[union-attr]
        for row in known_historical_rows
    )
    trajectory_rows = [
        row.report_row()
        for row in sorted(
            trajectories,
            key=lambda item: (item.panel, item.condition, item.seed),
        )
    ]
    bundle_rows = [
        row.report_row()
        for row in sorted(
            bundles,
            key=lambda item: (item.panel, item.seed),
        )
    ]
    accounting = {
        "reference_manifests": {
            "a256_reference_manifest_sha256": a256_manifest_sha,
            "h_reference_manifest_sha256": h_manifest_sha,
        },
        "trajectories": trajectory_rows,
        "evaluation_bundles": bundle_rows,
        "sealed_successful_compute": {
            "new_v2_training_chip_seconds": new_training_chip_seconds,
            "evaluation_bundle_chip_seconds": evaluation_chip_seconds,
            "total_chip_seconds": (
                new_training_chip_seconds + evaluation_chip_seconds
            ),
            "coverage": _SEALED_COMPUTE_COVERAGE,
        },
        "historical_training_compute": {
            "known_chip_seconds": historical_chip_seconds,
            "known_trajectory_count": len(known_historical_rows),
            "missing_trajectory_count": len(missing_historical_rows),
            "complete": not missing_historical_rows,
            "missing_identities": [
                {
                    "panel": row.panel,
                    "condition": row.condition,
                    "seed": row.seed,
                }
                for row in sorted(
                    missing_historical_rows,
                    key=lambda item: (item.panel, item.condition, item.seed),
                )
            ],
        },
        "wall_cash": wall_cash.report_row(),
    }
    return trajectory_grid, bundle_grid, accounting


def _score_report_row(
    scores: RoutedLadderTokenScores,
    trajectory: RoutedLadderTrajectoryAccounting,
) -> dict[str, Any]:
    if (
        trajectory.panel,
        trajectory.condition,
        trajectory.seed,
    ) != (
        scores.panel,
        scores.condition,
        scores.seed,
    ):
        raise Experiment5RoutedLadderError(
            "score trajectory accounting identity differs"
        )
    return {
        "condition": scores.condition,
        "trajectory_origin": trajectory.origin,
        "trajectory_accounting_sha256": trajectory.report_row()[
            "trajectory_accounting_sha256"
        ],
        "encoder_activation": (
            CONDITION_GEOMETRIES[scores.condition].encoder_activation
        ),
        "source_revision": scores.source_revision,
        "score_artifact_sha256": scores.score_artifact_sha256,
        "capture_manifest_sha256": scores.capture_manifest_sha256,
        "recomposition_sidecar_sha256": (
            scores.recomposition_sidecar_sha256
        ),
        "token_identity_sha256": scores.token_identity_sha256,
        "evaluation_tokens": scores.tokens,
        "packed_sequences": scores.packed_sequences,
        "shared_treatment": "oracle_exact",
        "endpoints": {
            endpoint: scores.aggregate(endpoint) for endpoint in ENDPOINTS
        },
        "endpoint_sst_sha256": {
            endpoint: scores.endpoint_sst_sha256(endpoint)
            for endpoint in ENDPOINTS
        },
    }


def build_experiment5_routed_ladder_report(
    *,
    score_sets: Sequence[
        RoutedLadderTokenScores | Mapping[str, Any]
    ],
    h_references: Sequence[ExistingHReferences | Mapping[str, Any]],
    report_inputs: ImmutableArtifactReference | Mapping[str, Any],
    trajectory_accounting: Sequence[
        RoutedLadderTrajectoryAccounting | Mapping[str, Any]
    ],
    evaluation_bundle_accounting: Sequence[
        RoutedLadderEvaluationBundleAccounting | Mapping[str, Any]
    ],
    a256_reference_manifest_sha256: str,
    h_reference_manifest_sha256: str,
    wall_cash_accounting: RoutedLadderWallCashAccounting | Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic, self-hashed A256/A512/H ladder report."""

    checked_scores = [
        RoutedLadderTokenScores.from_value(value) for value in score_sets
    ]
    if not checked_scores:
        raise Experiment5RoutedLadderError(
            "routed-ladder report requires score evidence"
        )
    score_grid = {
        (row.condition, *row.identity): row for row in checked_scores
    }
    if len(score_grid) != len(checked_scores):
        raise Experiment5RoutedLadderError(
            "routed-ladder scores contain duplicate curve identities"
        )
    identities_by_condition = {
        condition: {
            row.identity for row in checked_scores if row.condition == condition
        }
        for condition in SCORED_CONDITIONS
    }
    expected_identities = {
        (panel, seed, checkpoint_tokens)
        for panel in PANELS
        for seed in SEEDS
        for checkpoint_tokens in CHECKPOINT_TOKENS
    }
    if any(
        identities != expected_identities
        for identities in identities_by_condition.values()
    ):
        raise Experiment5RoutedLadderError(
            "report requires the exact 18-cohort "
            "A256ReLU/A512ReLU/A512Softplus grid"
        )

    checked_h = [ExistingHReferences.from_value(value) for value in h_references]
    h_grid = {row.identity: row for row in checked_h}
    if (
        len(h_grid) != len(checked_h)
        or set(h_grid) != expected_identities
    ):
        raise Experiment5RoutedLadderError(
            "report requires one existing H reference at every curve identity"
        )
    if any(
        row.endpoint_scores["learned_full_ffn"] is None
        for row in checked_h
    ):
        raise Experiment5RoutedLadderError(
            "complete routed-ladder report requires every H learned-full endpoint"
        )
    report_inputs_reference = ImmutableArtifactReference.from_value(
        report_inputs
    )
    if report_inputs_reference.kind != "report_inputs":
        raise Experiment5RoutedLadderError(
            "routed-ladder report inputs reference has the wrong kind"
        )
    (
        trajectory_grid,
        _bundle_grid,
        accounting,
    ) = _prepare_accounting(
        trajectory_accounting=trajectory_accounting,
        evaluation_bundle_accounting=evaluation_bundle_accounting,
        a256_reference_manifest_sha256=(
            a256_reference_manifest_sha256
        ),
        h_reference_manifest_sha256=h_reference_manifest_sha256,
        wall_cash_accounting=wall_cash_accounting,
    )
    for identity, h_reference in h_grid.items():
        trajectory = trajectory_grid[(identity[0], "H", identity[1])]
        if (
            h_reference.training_source_revision
            != trajectory.training_source_revision
            or h_reference.training_result_sha256
            != trajectory.parent_training_result_sha256
            or h_reference.training_result_file_sha256
            != trajectory.parent_training_result_file_sha256
        ):
            raise Experiment5RoutedLadderError(
                "H reference parent differs from trajectory accounting"
            )

    cohorts: list[dict[str, Any]] = []
    for panel, seed, checkpoint_tokens in sorted(expected_identities):
        identity = panel, seed, checkpoint_tokens
        a256_relu = score_grid[("A256ReLU", *identity)]
        a512_relu = score_grid[("A512ReLU", *identity)]
        a512_softplus = score_grid[("A512Softplus", *identity)]
        _validate_pair(a256_relu, a512_relu)
        _validate_pair(a512_relu, a512_softplus)
        h_reference = h_grid[identity]
        _validate_h_context(a512_softplus, h_reference)
        cohorts.append(
            {
                "identity": {
                    "panel": panel,
                    "seed": seed,
                    "checkpoint_tokens": checkpoint_tokens,
                },
                "conditions": {
                    "A256ReLU": _score_report_row(
                        a256_relu,
                        trajectory_grid[(panel, "A256ReLU", seed)],
                    ),
                    "A512ReLU": _score_report_row(
                        a512_relu,
                        trajectory_grid[(panel, "A512ReLU", seed)],
                    ),
                    "A512Softplus": _score_report_row(
                        a512_softplus,
                        trajectory_grid[(panel, "A512Softplus", seed)],
                    ),
                    "H": {
                        **h_reference.report_row(),
                        "trajectory_origin": trajectory_grid[
                            (panel, "H", seed)
                        ].origin,
                        "trajectory_accounting_sha256": trajectory_grid[
                            (panel, "H", seed)
                        ].report_row()["trajectory_accounting_sha256"],
                    },
                },
                "paired_a512_relu_minus_a256_relu": {
                    endpoint: paired_a512_relu_minus_a256_relu_bootstrap(
                        a256_relu,
                        a512_relu,
                        endpoint=endpoint,
                    )
                    for endpoint in ENDPOINTS
                },
                "paired_a512_softplus_minus_a512_relu": {
                    endpoint: paired_a512_softplus_minus_a512_relu_bootstrap(
                        a512_relu,
                        a512_softplus,
                        endpoint=endpoint,
                    )
                    for endpoint in ENDPOINTS
                },
                "descriptive_noncausal_h_gaps": _descriptive_h_gaps(
                    a512_softplus,
                    h_reference,
                ),
            }
        )

    across_seed_scale: list[dict[str, Any]] = []
    across_seed_activation: list[dict[str, Any]] = []
    for panel in PANELS:
        for checkpoint_tokens in CHECKPOINT_TOKENS:
            scale_pairs = [
                (
                    score_grid[
                        ("A256ReLU", panel, seed, checkpoint_tokens)
                    ],
                    score_grid[
                        ("A512ReLU", panel, seed, checkpoint_tokens)
                    ],
                )
                for seed in SEEDS
            ]
            activation_pairs = [
                (
                    score_grid[
                        ("A512ReLU", panel, seed, checkpoint_tokens)
                    ],
                    score_grid[
                        ("A512Softplus", panel, seed, checkpoint_tokens)
                    ],
                )
                for seed in SEEDS
            ]
            across_seed_scale.append(
                {
                    "identity": {
                        "panel": panel,
                        "checkpoint_tokens": checkpoint_tokens,
                    },
                    "paired_mean_a512_relu_minus_a256_relu": {
                        endpoint: hierarchical_a512_relu_minus_a256_relu_bootstrap(
                            scale_pairs,
                            endpoint=endpoint,
                        )
                        for endpoint in ENDPOINTS
                    },
                }
            )
            across_seed_activation.append(
                {
                    "identity": {
                        "panel": panel,
                        "checkpoint_tokens": checkpoint_tokens,
                    },
                    "paired_mean_a512_softplus_minus_a512_relu": {
                        endpoint: hierarchical_a512_softplus_minus_a512_relu_bootstrap(
                            activation_pairs,
                            endpoint=endpoint,
                        )
                        for endpoint in ENDPOINTS
                    },
                }
            )

    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "diagnostic_routed_activation_ladder",
        "conditions": {
            condition: CONDITION_GEOMETRIES[condition].report_row()
            for condition in CONDITIONS
        },
        "endpoints": list(ENDPOINTS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "report_inputs": report_inputs_reference.report_row(),
        "accounting": accounting,
        "cohorts": cohorts,
        "across_seed_a512_relu_minus_a256_relu": across_seed_scale,
        "across_seed_a512_softplus_minus_a512_relu": across_seed_activation,
        "interpretation": {
            "authorized_contrasts": [
                "paired A512ReLU-minus-A256ReLU within endpoint",
                "paired A512Softplus-minus-A512ReLU within endpoint",
            ],
            "headline_authorized": False,
            "a256_seed_provenance": {
                "sae_seed_0": "historical Experiment-4 immutable reference",
                "sae_seeds_1_2": "new protocol-v2 training",
                "transcoder_seeds_0_1_2": "new protocol-v2 training",
                "excluded_transcoder_reference": (
                    "Experiment-4 transcoder A256 seed 0 used Softplus"
                ),
                "three_seed_summary_policy": (
                    "unavailable until all three exact cells in a panel validate"
                ),
            },
            "h_role": (
                "hash-pinned matched point context; H gaps are descriptive "
                "and noncausal"
            ),
            "caveats": list(CAVEATS),
        },
    }
    report = {**body, "report_sha256": canonical_sha256(body)}
    validate_experiment5_routed_ladder_report(report)
    return report


def _validate_score_summary(name: str, value: Any) -> dict[str, float]:
    selected = _exact_mapping(name, value, {"sse", "sst", "nmse", "ev"})
    sse = _finite(f"{name}.sse", selected["sse"], nonnegative=True)
    sst = _finite(f"{name}.sst", selected["sst"], positive=True)
    nmse = _finite(f"{name}.nmse", selected["nmse"], nonnegative=True)
    ev = _finite(f"{name}.ev", selected["ev"])
    if not math.isclose(nmse, sse / sst, rel_tol=1e-12, abs_tol=1e-12):
        raise Experiment5RoutedLadderError(f"{name}.nmse differs from SSE/SST")
    if not math.isclose(ev, 1.0 - nmse, rel_tol=1e-12, abs_tol=1e-12):
        raise Experiment5RoutedLadderError(f"{name}.ev differs from 1-NMSE")
    return {"sse": sse, "sst": sst, "nmse": nmse, "ev": ev}


def _validate_across_seed_collection(
    rows: Any,
    *,
    field: str,
    contrast: str,
    left_condition: str,
    observed_seed_effects: Mapping[
        tuple[str, str, int, str],
        Mapping[int, float],
    ],
    cohorts: Sequence[Mapping[str, Any]],
) -> None:
    """Validate one complete six-row hierarchical contrast collection."""

    if not isinstance(rows, list) or len(rows) != (
        len(PANELS) * len(CHECKPOINT_TOKENS)
    ):
        raise Experiment5RoutedLadderError(
            f"{field} requires six across-seed summaries"
        )
    seen: set[tuple[str, int]] = set()
    expected_effect_fields = {
        "contrast",
        "endpoint",
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
    for index, raw in enumerate(rows):
        name = f"{field}[{index}]"
        effect_field = field.replace("across_seed_", "paired_mean_")
        row = _exact_mapping(name, raw, {"identity", effect_field})
        identity = _exact_mapping(
            f"{name}.identity",
            row["identity"],
            {"panel", "checkpoint_tokens"},
        )
        panel = identity["panel"]
        checkpoint = identity["checkpoint_tokens"]
        if (
            panel not in PANELS
            or checkpoint not in CHECKPOINT_TOKENS
            or (panel, checkpoint) in seen
        ):
            raise Experiment5RoutedLadderError(
                f"{name} has an invalid or duplicate identity"
            )
        seen.add((panel, checkpoint))
        endpoint_rows = _exact_mapping(
            f"{name}.{effect_field}",
            row[effect_field],
            set(ENDPOINTS),
        )
        representative = next(
            cohort
            for cohort in cohorts
            if cohort["identity"]
            == {
                "panel": panel,
                "seed": SEEDS[0],
                "checkpoint_tokens": checkpoint,
            }
        )["conditions"][left_condition]
        for endpoint in ENDPOINTS:
            effect_name = f"{name}.{effect_field}.{endpoint}"
            effect = _exact_mapping(
                effect_name,
                endpoint_rows[endpoint],
                expected_effect_fields,
            )
            seed_effects = observed_seed_effects.get(
                (contrast, panel, checkpoint, endpoint),
            )
            if seed_effects is None or set(seed_effects) != set(SEEDS):
                raise Experiment5RoutedLadderError(
                    f"{effect_name} lacks its three per-seed effects"
            )
            raw_seed_effects = effect["seed_effects"]
            if (
                not isinstance(raw_seed_effects, list)
                or len(raw_seed_effects) != len(SEEDS)
            ):
                raise Experiment5RoutedLadderError(
                    f"{effect_name}.seed_effects must contain exactly three rows"
                )
            checked_seed_effects: dict[int, float] = {}
            for seed_index, raw_seed_effect in enumerate(raw_seed_effects):
                seed_row = _exact_mapping(
                    f"{effect_name}.seed_effects[{seed_index}]",
                    raw_seed_effect,
                    {"seed", "estimate_ev_difference"},
                )
                selected_seed = _nonnegative_int(
                    f"{effect_name}.seed_effects[{seed_index}].seed",
                    seed_row["seed"],
                )
                if (
                    selected_seed != SEEDS[seed_index]
                    or selected_seed in checked_seed_effects
                ):
                    raise Experiment5RoutedLadderError(
                        f"{effect_name}.seed_effects must be ordered seeds 0/1/2"
                    )
                checked_seed_effects[selected_seed] = _finite(
                    (
                        f"{effect_name}.seed_effects[{seed_index}]"
                        ".estimate_ev_difference"
                    ),
                    seed_row["estimate_ev_difference"],
                )
            estimate = _finite(
                f"{effect_name}.estimate_mean_ev_difference",
                effect["estimate_mean_ev_difference"],
            )
            lower = _finite(f"{effect_name}.ci95_lower", effect["ci95_lower"])
            upper = _finite(f"{effect_name}.ci95_upper", effect["ci95_upper"])
            expected_mean = math.fsum(seed_effects.values()) / len(SEEDS)
            if (
                effect["contrast"] != f"mean_seed({contrast})"
                or effect["endpoint"] != endpoint
                or not math.isclose(
                    estimate,
                    expected_mean,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or lower > upper
                or effect["confidence_level"] != CONFIDENCE_LEVEL
                or effect["bootstrap_unit"] != "seed_then_packed_sequence"
                or effect["bootstrap_algorithm"]
                != HIERARCHICAL_BOOTSTRAP_ALGORITHM
                or effect["bootstrap_replicates"] != BOOTSTRAP_REPLICATES
                or _nonnegative_int(
                    f"{effect_name}.bootstrap_seed",
                    effect["bootstrap_seed"],
                )
                != effect["bootstrap_seed"]
                or effect["seeds"] != list(SEEDS)
                or _nonnegative_int(
                    f"{effect_name}.positive_seed_count",
                    effect["positive_seed_count"],
                )
                != sum(value > 0 for value in seed_effects.values())
                or checked_seed_effects != dict(seed_effects)
                or effect["evaluation_tokens_per_seed"]
                != representative["evaluation_tokens"]
                or effect["packed_sequences_per_seed"]
                != representative["packed_sequences"]
                or effect["token_identity_sha256"]
                != representative["token_identity_sha256"]
            ):
                raise Experiment5RoutedLadderError(
                    f"{effect_name} is incompatible with per-seed evidence"
                )
    expected = {
        (panel, checkpoint_tokens)
        for panel in PANELS
        for checkpoint_tokens in CHECKPOINT_TOKENS
    }
    if seen != expected:
        raise Experiment5RoutedLadderError(
            f"{field} summary identities are incomplete"
        )


def validate_experiment5_routed_ladder_report(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate report self-hash, exact geometry, scores, and claim boundary."""

    if not isinstance(value, Mapping):
        raise Experiment5RoutedLadderError("routed-ladder report must be an object")
    body = dict(value)
    claimed = _sha256(
        "routed-ladder report.report_sha256",
        body.pop("report_sha256", None),
    )
    if canonical_sha256(body) != claimed:
        raise Experiment5RoutedLadderError(
            "routed-ladder report self-hash is invalid"
        )
    _exact_mapping(
        "routed-ladder report",
        body,
        {
            "schema_version",
            "protocol_id",
            "status",
            "conditions",
            "endpoints",
            "bootstrap_replicates",
            "report_inputs",
            "accounting",
            "cohorts",
            "across_seed_a512_relu_minus_a256_relu",
            "across_seed_a512_softplus_minus_a512_relu",
            "interpretation",
        },
    )
    if (
        body["schema_version"] != REPORT_SCHEMA_VERSION
        or body["protocol_id"] != PROTOCOL_ID
        or body["status"] != "diagnostic_routed_activation_ladder"
        or body["endpoints"] != list(ENDPOINTS)
        or body["bootstrap_replicates"] != BOOTSTRAP_REPLICATES
    ):
        raise Experiment5RoutedLadderError("routed-ladder report identity changed")
    expected_geometry = {
        condition: CONDITION_GEOMETRIES[condition].report_row()
        for condition in CONDITIONS
    }
    if body["conditions"] != expected_geometry:
        raise Experiment5RoutedLadderError(
            "routed-ladder condition geometry changed"
        )
    report_inputs_reference = ImmutableArtifactReference.from_value(
        body["report_inputs"]
    )
    if report_inputs_reference.kind != "report_inputs":
        raise Experiment5RoutedLadderError(
            "routed-ladder report inputs reference has the wrong kind"
        )
    raw_accounting = _exact_mapping(
        "routed-ladder accounting",
        body["accounting"],
        {
            "reference_manifests",
            "trajectories",
            "evaluation_bundles",
            "sealed_successful_compute",
            "historical_training_compute",
            "wall_cash",
        },
    )
    reference_manifests = _exact_mapping(
        "routed-ladder reference manifests",
        raw_accounting["reference_manifests"],
        {
            "a256_reference_manifest_sha256",
            "h_reference_manifest_sha256",
        },
    )
    raw_trajectories = raw_accounting["trajectories"]
    raw_bundles = raw_accounting["evaluation_bundles"]
    if not isinstance(raw_trajectories, list) or not isinstance(
        raw_bundles,
        list,
    ):
        raise Experiment5RoutedLadderError(
            "routed-ladder accounting inventories must be lists"
        )
    (
        trajectory_grid,
        _bundle_grid,
        expected_accounting,
    ) = _prepare_accounting(
        trajectory_accounting=raw_trajectories,
        evaluation_bundle_accounting=raw_bundles,
        a256_reference_manifest_sha256=reference_manifests[
            "a256_reference_manifest_sha256"
        ],
        h_reference_manifest_sha256=reference_manifests[
            "h_reference_manifest_sha256"
        ],
        wall_cash_accounting=raw_accounting["wall_cash"],
    )
    if raw_accounting != expected_accounting:
        raise Experiment5RoutedLadderError(
            "routed-ladder accounting summary differs from immutable rows"
        )
    interpretation = _exact_mapping(
        "routed-ladder interpretation",
        body["interpretation"],
        {
            "authorized_contrasts",
            "headline_authorized",
            "a256_seed_provenance",
            "h_role",
            "caveats",
        },
    )
    if (
        interpretation["authorized_contrasts"]
        != [
            "paired A512ReLU-minus-A256ReLU within endpoint",
            "paired A512Softplus-minus-A512ReLU within endpoint",
        ]
        or interpretation["headline_authorized"] is not False
        or interpretation["a256_seed_provenance"]
        != {
            "sae_seed_0": "historical Experiment-4 immutable reference",
            "sae_seeds_1_2": "new protocol-v2 training",
            "transcoder_seeds_0_1_2": "new protocol-v2 training",
            "excluded_transcoder_reference": (
                "Experiment-4 transcoder A256 seed 0 used Softplus"
            ),
            "three_seed_summary_policy": (
                "unavailable until all three exact cells in a panel validate"
            ),
        }
        or interpretation["h_role"]
        != (
            "hash-pinned matched point context; H gaps are descriptive "
            "and noncausal"
        )
        or interpretation["caveats"] != list(CAVEATS)
    ):
        raise Experiment5RoutedLadderError(
            "routed-ladder claim boundary changed"
        )
    cohorts = body["cohorts"]
    if not isinstance(cohorts, list) or len(cohorts) != (
        len(PANELS) * len(SEEDS) * len(CHECKPOINT_TOKENS)
    ):
        raise Experiment5RoutedLadderError(
            "routed-ladder report requires the exact 18-cohort grid"
        )
    seen: set[tuple[str, int, int]] = set()
    observed_seed_effects: dict[
        tuple[str, str, int, str],
        dict[int, float],
    ] = {}
    for index, raw in enumerate(cohorts):
        name = f"cohorts[{index}]"
        cohort = _exact_mapping(
            name,
            raw,
            {
                "identity",
                "conditions",
                "paired_a512_relu_minus_a256_relu",
                "paired_a512_softplus_minus_a512_relu",
                "descriptive_noncausal_h_gaps",
            },
        )
        identity = _exact_mapping(
            f"{name}.identity",
            cohort["identity"],
            {"panel", "seed", "checkpoint_tokens"},
        )
        panel = identity["panel"]
        if panel not in PANELS:
            raise Experiment5RoutedLadderError(f"{name} panel is invalid")
        seed = _nonnegative_int(f"{name}.seed", identity["seed"])
        checkpoint = _nonnegative_int(
            f"{name}.checkpoint_tokens",
            identity["checkpoint_tokens"],
            positive=True,
        )
        identity_key = panel, seed, checkpoint
        if identity_key in seen:
            raise Experiment5RoutedLadderError(
                "routed-ladder report contains duplicate cohorts"
            )
        seen.add(identity_key)
        conditions = _exact_mapping(
            f"{name}.conditions",
            cohort["conditions"],
            set(CONDITIONS),
        )
        summaries: dict[str, dict[str, dict[str, float]]] = {}
        common_token_identity: str | None = None
        common_tokens: int | None = None
        common_clusters: int | None = None
        common_source_revision: str | None = None
        common_capture: str | None = None
        common_sidecar: str | None = None
        common_sst_proofs: Mapping[str, Any] | None = None
        for condition in SCORED_CONDITIONS:
            score_name = f"{name}.conditions.{condition}"
            row = _exact_mapping(
                score_name,
                conditions[condition],
                {
                    "condition",
                    "trajectory_origin",
                    "trajectory_accounting_sha256",
                    "encoder_activation",
                    "source_revision",
                    "score_artifact_sha256",
                    "capture_manifest_sha256",
                    "recomposition_sidecar_sha256",
                    "token_identity_sha256",
                    "evaluation_tokens",
                    "packed_sequences",
                    "shared_treatment",
                    "endpoints",
                    "endpoint_sst_sha256",
                },
            )
            trajectory = trajectory_grid[(panel, condition, seed)]
            if (
                row["condition"] != condition
                or row["trajectory_origin"] != trajectory.origin
                or row["trajectory_accounting_sha256"]
                != trajectory.report_row()["trajectory_accounting_sha256"]
                or row["encoder_activation"]
                != CONDITION_GEOMETRIES[condition].encoder_activation
                or row["shared_treatment"] != "oracle_exact"
            ):
                raise Experiment5RoutedLadderError(
                    f"{score_name} identity changed"
                )
            _git_sha(f"{score_name}.source_revision", row["source_revision"])
            for field in (
                "score_artifact_sha256",
                "capture_manifest_sha256",
                "recomposition_sidecar_sha256",
                "token_identity_sha256",
            ):
                _sha256(f"{score_name}.{field}", row[field])
            tokens = _nonnegative_int(
                f"{score_name}.evaluation_tokens",
                row["evaluation_tokens"],
                positive=True,
            )
            clusters = _nonnegative_int(
                f"{score_name}.packed_sequences",
                row["packed_sequences"],
                positive=True,
            )
            if clusters > tokens:
                raise Experiment5RoutedLadderError(
                    f"{score_name} has more sequences than tokens"
                )
            endpoints = _exact_mapping(
                f"{score_name}.endpoints",
                row["endpoints"],
                set(ENDPOINTS),
            )
            summaries[condition] = {
                endpoint: _validate_score_summary(
                    f"{score_name}.endpoints.{endpoint}",
                    endpoints[endpoint],
                )
                for endpoint in ENDPOINTS
            }
            sst_proofs = _exact_mapping(
                f"{score_name}.endpoint_sst_sha256",
                row["endpoint_sst_sha256"],
                set(ENDPOINTS),
            )
            for endpoint in ENDPOINTS:
                _sha256(
                    f"{score_name}.endpoint_sst_sha256.{endpoint}",
                    sst_proofs[endpoint],
                )
            if common_token_identity is None:
                common_token_identity = row["token_identity_sha256"]
                common_tokens = tokens
                common_clusters = clusters
                common_source_revision = row["source_revision"]
                common_capture = row["capture_manifest_sha256"]
                common_sidecar = row["recomposition_sidecar_sha256"]
                common_sst_proofs = sst_proofs
            elif (
                row["token_identity_sha256"] != common_token_identity
                or tokens != common_tokens
                or clusters != common_clusters
                or row["source_revision"] != common_source_revision
                or row["capture_manifest_sha256"] != common_capture
                or row["recomposition_sidecar_sha256"] != common_sidecar
                or sst_proofs != common_sst_proofs
            ):
                raise Experiment5RoutedLadderError(
                    f"{name} A score identities or common SST proofs differ"
                )
        h_row = _exact_mapping(
            f"{name}.conditions.H",
            conditions["H"],
            {
                "panel",
                "seed",
                "checkpoint_tokens",
                "condition",
                "trajectory_origin",
                "trajectory_accounting_sha256",
                "source_revision",
                "training_source_revision",
                "h_config_sha256",
                "training_result_sha256",
                "training_result_file_sha256",
                "checkpoint_pointer_sha256",
                "checkpoint_pointer_file_sha256",
                "score_artifact_sha256",
                "token_identity_sha256",
                "evaluation_tokens",
                "packed_sequences",
                "encoder_activation",
                "shared_treatment",
                "endpoints",
                "endpoint_sst_sha256",
                "scientific_result",
                "diagnostics",
            },
        )
        h_reference = ExistingHReferences.from_value(
            {
                "panel": h_row["panel"],
                "seed": h_row["seed"],
                "checkpoint_tokens": h_row["checkpoint_tokens"],
                "source_revision": h_row["source_revision"],
                "training_source_revision": h_row[
                    "training_source_revision"
                ],
                "h_config_sha256": h_row["h_config_sha256"],
                "training_result_sha256": h_row[
                    "training_result_sha256"
                ],
                "training_result_file_sha256": h_row[
                    "training_result_file_sha256"
                ],
                "checkpoint_pointer_sha256": h_row[
                    "checkpoint_pointer_sha256"
                ],
                "checkpoint_pointer_file_sha256": h_row[
                    "checkpoint_pointer_file_sha256"
                ],
                "score_artifact_sha256": h_row[
                    "score_artifact_sha256"
                ],
                "token_identity_sha256": h_row["token_identity_sha256"],
                "evaluation_tokens": h_row["evaluation_tokens"],
                "packed_sequences": h_row["packed_sequences"],
                "endpoint_scores": h_row["endpoints"],
                "endpoint_sst_sha256": h_row["endpoint_sst_sha256"],
                "scientific_result": h_row["scientific_result"],
                "diagnostics": h_row["diagnostics"],
            }
        )
        h_trajectory = trajectory_grid[(panel, "H", seed)]
        if (
            h_reference.identity != identity_key
            or h_row["condition"] != "H"
            or h_row["trajectory_origin"] != h_trajectory.origin
            or h_row["trajectory_accounting_sha256"]
            != h_trajectory.report_row()["trajectory_accounting_sha256"]
            or h_reference.training_source_revision
            != h_trajectory.training_source_revision
            or h_reference.training_result_sha256
            != h_trajectory.parent_training_result_sha256
            or h_reference.training_result_file_sha256
            != h_trajectory.parent_training_result_file_sha256
            or h_row["encoder_activation"] != "softplus"
            or h_row["shared_treatment"] != "learned_sparse_dictionary"
        ):
            raise Experiment5RoutedLadderError(
                f"{name} H reference identity changed"
            )
        if (
            h_reference.token_identity_sha256 != common_token_identity
            or h_reference.evaluation_tokens != common_tokens
            or h_reference.packed_sequences != common_clusters
        ):
            raise Experiment5RoutedLadderError(
                f"{name} H token identity proof differs from A"
            )
        if common_sst_proofs is None:
            raise Experiment5RoutedLadderError(
                f"{name} lacks A common-SST proofs"
            )
        for endpoint in ENDPOINTS:
            h_score = h_reference.endpoint_scores[endpoint]
            if h_score is None or (
                not math.isclose(
                    h_score["sst"],
                    summaries["A512Softplus"][endpoint]["sst"],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or h_reference.endpoint_sst_sha256[endpoint]
                != common_sst_proofs[endpoint]
            ):
                raise Experiment5RoutedLadderError(
                    f"{name} H {endpoint} common SST proof differs from A"
                )
        for field, left_condition, right_condition in (
            (
                "paired_a512_relu_minus_a256_relu",
                "A256ReLU",
                "A512ReLU",
            ),
            (
                "paired_a512_softplus_minus_a512_relu",
                "A512ReLU",
                "A512Softplus",
            ),
        ):
            contrast = f"{right_condition}-{left_condition}"
            paired = _exact_mapping(
                f"{name}.{field}",
                cohort[field],
                set(ENDPOINTS),
            )
            for endpoint in ENDPOINTS:
                effect_name = f"{name}.{field}.{endpoint}"
                effect = _exact_mapping(
                    effect_name,
                    paired[endpoint],
                    {
                        "contrast",
                        "endpoint",
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
                    },
                )
                estimate = _finite(
                    f"{effect_name}.estimate_ev_difference",
                    effect["estimate_ev_difference"],
                )
                lower = _finite(
                    f"{effect_name}.ci95_lower",
                    effect["ci95_lower"],
                )
                upper = _finite(
                    f"{effect_name}.ci95_upper",
                    effect["ci95_upper"],
                )
                expected = (
                    summaries[right_condition][endpoint]["ev"]
                    - summaries[left_condition][endpoint]["ev"]
                )
                if (
                    effect["contrast"] != contrast
                    or effect["endpoint"] != endpoint
                    or not math.isclose(
                        estimate,
                        expected,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    or lower > upper
                    or effect["confidence_level"] != CONFIDENCE_LEVEL
                    or effect["bootstrap_unit"] != "packed_sequence"
                    or effect["bootstrap_algorithm"] != BOOTSTRAP_ALGORITHM
                    or effect["bootstrap_replicates"] != BOOTSTRAP_REPLICATES
                    or _nonnegative_int(
                        f"{effect_name}.bootstrap_seed",
                        effect["bootstrap_seed"],
                    )
                    != effect["bootstrap_seed"]
                    or effect["evaluation_tokens"] != common_tokens
                    or effect["packed_sequences"] != common_clusters
                    or effect["token_identity_sha256"] != common_token_identity
                ):
                    raise Experiment5RoutedLadderError(
                        f"{effect_name} is incompatible with score evidence"
                    )
                observed_seed_effects.setdefault(
                    (contrast, panel, checkpoint, endpoint),
                    {},
                )[seed] = estimate
        descriptive = _exact_mapping(
            f"{name}.descriptive_noncausal_h_gaps",
            cohort["descriptive_noncausal_h_gaps"],
            {
                "h_oracle_minus_a512_softplus_oracle",
                "h_oracle_minus_h_learned",
            },
        )
        h_oracle = h_reference.endpoint_scores["oracle_shared"]
        if h_oracle is None:
            raise Experiment5RoutedLadderError(
                f"{name} lacks H oracle-shared score"
            )
        hybrid_context = _exact_mapping(
            f"{name}.h_oracle_minus_a512_softplus_oracle",
            descriptive["h_oracle_minus_a512_softplus_oracle"],
            {
                "status",
                "endpoint",
                "estimate_ev_difference",
                "paired_interval_available",
            },
        )
        hybrid_estimate = _finite(
            f"{name}.h_oracle_minus_a512_softplus_oracle.estimate_ev_difference",
            hybrid_context["estimate_ev_difference"],
        )
        if (
            hybrid_context["status"] != "descriptive_noncausal"
            or hybrid_context["endpoint"] != "oracle_shared"
            or hybrid_context["paired_interval_available"] is not False
            or not math.isclose(
                hybrid_estimate,
                h_oracle["ev"]
                - summaries["A512Softplus"]["oracle_shared"]["ev"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise Experiment5RoutedLadderError(
                f"{name} H-oracle/A512Softplus descriptive gap is invalid"
            )
        learned_context = _exact_mapping(
            f"{name}.h_oracle_minus_h_learned",
            descriptive["h_oracle_minus_h_learned"],
            {
                "status",
                "endpoint",
                "estimate_ev_difference",
                "paired_interval_available",
            },
        )
        learned = h_reference.endpoint_scores["learned_full_ffn"]
        if learned is None:
            raise Experiment5RoutedLadderError(
                f"{name} complete report lacks H learned-full evidence"
            )
        learned_estimate = _finite(
            f"{name}.h_oracle_minus_h_learned.estimate_ev_difference",
            learned_context["estimate_ev_difference"],
        )
        if (
            learned_context["status"] != "descriptive_noncausal"
            or not math.isclose(
                learned_estimate,
                h_oracle["ev"] - learned["ev"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise Experiment5RoutedLadderError(
                f"{name} H oracle/learned descriptive gap is invalid"
            )
        if (
            learned_context["endpoint"] != "full_ffn_common_sst"
            or learned_context["paired_interval_available"] is not False
        ):
            raise Experiment5RoutedLadderError(
                f"{name} H oracle/learned gap interpretation changed"
            )
    expected_identities = {
        (panel, seed, checkpoint_tokens)
        for panel in PANELS
        for seed in SEEDS
        for checkpoint_tokens in CHECKPOINT_TOKENS
    }
    if seen != expected_identities:
        raise Experiment5RoutedLadderError(
            "routed-ladder report cohort identities are incomplete"
        )

    _validate_across_seed_collection(
        body["across_seed_a512_relu_minus_a256_relu"],
        field="across_seed_a512_relu_minus_a256_relu",
        contrast="A512ReLU-A256ReLU",
        left_condition="A256ReLU",
        observed_seed_effects=observed_seed_effects,
        cohorts=cohorts,
    )

    _validate_across_seed_collection(
        body["across_seed_a512_softplus_minus_a512_relu"],
        field="across_seed_a512_softplus_minus_a512_relu",
        contrast="A512Softplus-A512ReLU",
        left_condition="A512ReLU",
        observed_seed_effects=observed_seed_effects,
        cohorts=cohorts,
    )
    return {**body, "report_sha256": claimed}


__all__ = [
    "BOOTSTRAP_ALGORITHM",
    "BOOTSTRAP_REPLICATES",
    "CAVEATS",
    "CHECKPOINT_TOKENS",
    "CONDITIONS",
    "CONDITION_GEOMETRIES",
    "ENDPOINTS",
    "H_ENDPOINTS",
    "HIERARCHICAL_BOOTSTRAP_ALGORITHM",
    "PAIRED_CONTRASTS",
    "PANELS",
    "ExistingHReferences",
    "Experiment5RoutedLadderError",
    "ImmutableArtifactReference",
    "PROTOCOL_ID",
    "REPORT_SCHEMA_VERSION",
    "RoutedLadderEvaluationBundleAccounting",
    "RoutedLadderHTokenScores",
    "RoutedLadderTrainingCost",
    "RoutedLadderTokenScores",
    "RoutedLadderTrajectoryAccounting",
    "RoutedLadderWallCashAccounting",
    "SCORE_SCHEMA_VERSION",
    "SCORED_CONDITIONS",
    "SEEDS",
    "build_experiment5_routed_ladder_report",
    "h_token_scores_to_existing_reference",
    "hierarchical_a512_relu_minus_a256_relu_bootstrap",
    "hierarchical_a512_softplus_minus_a512_relu_bootstrap",
    "hierarchical_routed_ladder_bootstrap",
    "paired_a512_relu_minus_a256_relu_bootstrap",
    "paired_a512_softplus_minus_a512_relu_bootstrap",
    "paired_routed_ladder_bootstrap",
    "validate_experiment5_routed_ladder_report",
]
