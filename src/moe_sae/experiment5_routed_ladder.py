"""Scientific scoring for the Experiment-5 routed activation ladder.

This module is deliberately separate from the frozen H/R/B/F implementation.
It scores two full-sized routed-only true-ID variants on identical scientific
tokens:

* ``A512ReLU`` has 512 features and TopK-4 per visible expert bank, with ReLU;
* ``A512Softplus`` has the exact same geometry and uses Softplus; and
* ``H`` is the existing Experiment-5 Softplus routed model plus a learned
  shared dictionary.

The A variants use the exact shared branch when the ``oracle_shared`` endpoint
is formed.  H instead learns its shared branch.  Consequently the only paired,
interval-bearing effect computed here is A512Softplus minus A512ReLU. Existing
H evidence is retained as hash-pinned context and is never silently promoted
into a shared-branch causal contrast.
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

PROTOCOL_ID = (
    "gemma4-experiment5-routed-activation-ladder-v1-20260729"
)
SCORE_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-token-scores-v1"
)
REPORT_SCHEMA_VERSION = (
    "gemma4-experiment5-routed-activation-ladder-report-v1"
)

PANELS = ("sae", "transcoder")
CONDITIONS = ("A512ReLU", "A512Softplus", "H")
SCORED_CONDITIONS = ("A512ReLU", "A512Softplus")
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

    condition: Literal["A512ReLU", "A512Softplus", "H"]
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
            "relu" if self.condition == "A512ReLU" else "softplus"
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
        b"gemma4-experiment5-routed-activation-ladder-common-sst-v1\0"
    )
    digest.update(bytes.fromhex(token_identity))
    digest.update(endpoint.encode("ascii") + b"\0")
    digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RoutedLadderTokenScores:
    """Per-token sufficient scores for one routed-only A condition."""

    condition: Literal["A512ReLU", "A512Softplus"]
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
                "token scores must be A512ReLU or A512Softplus"
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
    frozen A512ReLU/A512Softplus paired contrast cannot silently start treating
    H as a third randomized arm.
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
    a512_relu: RoutedLadderTokenScores,
    a512_softplus: RoutedLadderTokenScores,
) -> None:
    if a512_relu.condition != "A512ReLU" or a512_softplus.condition != "A512Softplus":
        raise Experiment5RoutedLadderError(
            "routed pair must be A512ReLU then A512Softplus"
        )
    if a512_relu.identity != a512_softplus.identity:
        raise Experiment5RoutedLadderError(
            "A512ReLU and A512Softplus curve identities differ"
        )
    if a512_relu.source_revision != a512_softplus.source_revision:
        raise Experiment5RoutedLadderError(
            "A512ReLU and A512Softplus producer revisions differ"
        )
    if (
        a512_relu.capture_manifest_sha256 != a512_softplus.capture_manifest_sha256
        or a512_relu.recomposition_sidecar_sha256
        != a512_softplus.recomposition_sidecar_sha256
    ):
        raise Experiment5RoutedLadderError(
            "A512ReLU and A512Softplus scientific parents differ"
        )
    if (
        not np.array_equal(a512_relu.sequence_id, a512_softplus.sequence_id)
        or not np.array_equal(a512_relu.position, a512_softplus.position)
        or a512_relu.token_identity_sha256 != a512_softplus.token_identity_sha256
    ):
        raise Experiment5RoutedLadderError(
            "A512ReLU and A512Softplus token identities are not exactly aligned"
        )
    for endpoint in ENDPOINTS:
        if not np.array_equal(
            a512_relu.endpoint_sst[endpoint],
            a512_softplus.endpoint_sst[endpoint],
        ):
            raise Experiment5RoutedLadderError(
                f"A512ReLU and A512Softplus {endpoint} target SST rows differ"
            )


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
    a512_relu: RoutedLadderTokenScores,
    *,
    endpoint: str,
) -> int:
    payload = "|".join(
        (
            PROTOCOL_ID,
            a512_relu.panel,
            str(a512_relu.seed),
            str(a512_relu.checkpoint_tokens),
            endpoint,
            "A512Softplus-A512ReLU",
            a512_relu.token_identity_sha256,
        )
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("ascii")).digest()[:8],
        "big",
        signed=False,
    )


def paired_a512_softplus_minus_a512_relu_bootstrap(
    a512_relu: RoutedLadderTokenScores | Mapping[str, Any],
    a512_softplus: RoutedLadderTokenScores | Mapping[str, Any],
    *,
    endpoint: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Compute a packed-sequence paired EV contrast from aligned token scores."""

    left = RoutedLadderTokenScores.from_value(a512_relu)
    right = RoutedLadderTokenScores.from_value(a512_softplus)
    _validate_pair(left, right)
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
    # EV_A512Softplus - EV_A512ReLU = (SSE_A512ReLU - SSE_A512Softplus) / common_SST.
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
    seed = _bootstrap_seed(left, endpoint=endpoint)
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
        "contrast": "A512Softplus-A512ReLU",
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


def _hierarchical_bootstrap_seed(
    *,
    panel: str,
    checkpoint_tokens: int,
    endpoint: str,
) -> int:
    payload = "|".join(
        (
            PROTOCOL_ID,
            panel,
            str(checkpoint_tokens),
            endpoint,
            "A512Softplus-A512ReLU",
            "hierarchical",
        )
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("ascii")).digest()[:8],
        "big",
        signed=False,
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
    for index, raw_pair in enumerate(pairs):
        if (
            not isinstance(raw_pair, Sequence)
            or isinstance(raw_pair, (str, bytes))
            or len(raw_pair) != 2
        ):
            raise Experiment5RoutedLadderError(
                f"hierarchical bootstrap pair {index} must contain "
                "A512ReLU/A512Softplus"
            )
        left = RoutedLadderTokenScores.from_value(raw_pair[0])
        right = RoutedLadderTokenScores.from_value(raw_pair[1])
        _validate_pair(left, right)
        checked.append((left, right))
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
        "contrast": "mean_seed(A512Softplus-A512ReLU)",
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


def _score_report_row(scores: RoutedLadderTokenScores) -> dict[str, Any]:
    return {
        "condition": scores.condition,
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
) -> dict[str, Any]:
    """Build a deterministic, self-hashed A512ReLU/A512Softplus/H ladder report."""

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
    a512_relu_identities = {
        row.identity for row in checked_scores if row.condition == "A512ReLU"
    }
    a512_softplus_identities = {
        row.identity for row in checked_scores if row.condition == "A512Softplus"
    }
    expected_identities = {
        (panel, seed, checkpoint_tokens)
        for panel in PANELS
        for seed in SEEDS
        for checkpoint_tokens in CHECKPOINT_TOKENS
    }
    if (
        a512_relu_identities != expected_identities
        or a512_softplus_identities != expected_identities
    ):
        raise Experiment5RoutedLadderError(
            "report requires the exact 18-cohort A512ReLU/A512Softplus grid"
        )

    checked_h = [ExistingHReferences.from_value(value) for value in h_references]
    h_grid = {row.identity: row for row in checked_h}
    if (
        len(h_grid) != len(checked_h)
        or set(h_grid) != a512_relu_identities
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

    cohorts: list[dict[str, Any]] = []
    for panel, seed, checkpoint_tokens in sorted(a512_relu_identities):
        identity = panel, seed, checkpoint_tokens
        a512_relu = score_grid[("A512ReLU", *identity)]
        a512_softplus = score_grid[("A512Softplus", *identity)]
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
                    "A512ReLU": _score_report_row(a512_relu),
                    "A512Softplus": _score_report_row(a512_softplus),
                    "H": h_reference.report_row(),
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

    across_seed: list[dict[str, Any]] = []
    for panel in PANELS:
        for checkpoint_tokens in CHECKPOINT_TOKENS:
            pairs = [
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
            across_seed.append(
                {
                    "identity": {
                        "panel": panel,
                        "checkpoint_tokens": checkpoint_tokens,
                    },
                    "paired_mean_a512_softplus_minus_a512_relu": {
                        endpoint: hierarchical_a512_softplus_minus_a512_relu_bootstrap(
                            pairs,
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
        "cohorts": cohorts,
        "across_seed_a512_softplus_minus_a512_relu": across_seed,
        "interpretation": {
            "authorized_contrast": "paired A512Softplus-minus-A512ReLU within endpoint",
            "headline_authorized": False,
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
            "cohorts",
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
    interpretation = _exact_mapping(
        "routed-ladder interpretation",
        body["interpretation"],
        {
            "authorized_contrast",
            "headline_authorized",
            "h_role",
            "caveats",
        },
    )
    if (
        interpretation["authorized_contrast"]
        != "paired A512Softplus-minus-A512ReLU within endpoint"
        or interpretation["headline_authorized"] is not False
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
        tuple[str, int, str],
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
            if (
                row["condition"] != condition
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
        if (
            h_reference.identity != identity_key
            or h_row["condition"] != "H"
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
        paired = _exact_mapping(
            f"{name}.paired_a512_softplus_minus_a512_relu",
            cohort["paired_a512_softplus_minus_a512_relu"],
            set(ENDPOINTS),
        )
        for endpoint in ENDPOINTS:
            effect_name = f"{name}.paired_a512_softplus_minus_a512_relu.{endpoint}"
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
            lower = _finite(f"{effect_name}.ci95_lower", effect["ci95_lower"])
            upper = _finite(f"{effect_name}.ci95_upper", effect["ci95_upper"])
            expected = (
                summaries["A512Softplus"][endpoint]["ev"]
                - summaries["A512ReLU"][endpoint]["ev"]
            )
            if (
                effect["contrast"] != "A512Softplus-A512ReLU"
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
                (panel, checkpoint, endpoint),
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

    across_seed = body["across_seed_a512_softplus_minus_a512_relu"]
    if not isinstance(across_seed, list) or len(across_seed) != (
        len(PANELS) * len(CHECKPOINT_TOKENS)
    ):
        raise Experiment5RoutedLadderError(
            "routed-ladder report requires six across-seed summaries"
        )
    seen_across: set[tuple[str, int]] = set()
    for index, raw in enumerate(across_seed):
        name = f"across_seed_a512_softplus_minus_a512_relu[{index}]"
        row = _exact_mapping(
            name,
            raw,
            {"identity", "paired_mean_a512_softplus_minus_a512_relu"},
        )
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
            or (panel, checkpoint) in seen_across
        ):
            raise Experiment5RoutedLadderError(
                f"{name} has an invalid or duplicate identity"
            )
        seen_across.add((panel, checkpoint))
        endpoint_rows = _exact_mapping(
            f"{name}.paired_mean_a512_softplus_minus_a512_relu",
            row["paired_mean_a512_softplus_minus_a512_relu"],
            set(ENDPOINTS),
        )
        for endpoint in ENDPOINTS:
            effect_name = (
                f"{name}.paired_mean_a512_softplus_minus_a512_relu.{endpoint}"
            )
            effect = _exact_mapping(
                effect_name,
                endpoint_rows[endpoint],
                {
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
                },
            )
            seed_effects = observed_seed_effects.get(
                (panel, checkpoint, endpoint),
            )
            if seed_effects is None or set(seed_effects) != set(SEEDS):
                raise Experiment5RoutedLadderError(
                    f"{effect_name} lacks its three per-seed effects"
                )
            expected_mean = math.fsum(seed_effects.values()) / len(SEEDS)
            estimate = _finite(
                f"{effect_name}.estimate_mean_ev_difference",
                effect["estimate_mean_ev_difference"],
            )
            lower = _finite(
                f"{effect_name}.ci95_lower",
                effect["ci95_lower"],
            )
            upper = _finite(
                f"{effect_name}.ci95_upper",
                effect["ci95_upper"],
            )
            raw_seed_effects = effect["seed_effects"]
            if not isinstance(raw_seed_effects, list):
                raise Experiment5RoutedLadderError(
                    f"{effect_name}.seed_effects must be a list"
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
                checked_seed_effects[selected_seed] = _finite(
                    (
                        f"{effect_name}.seed_effects[{seed_index}]"
                        ".estimate_ev_difference"
                    ),
                    seed_row["estimate_ev_difference"],
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
            )
            representative_a = representative["conditions"]["A512ReLU"]
            if (
                effect["contrast"] != "mean_seed(A512Softplus-A512ReLU)"
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
                or checked_seed_effects != seed_effects
                or effect["evaluation_tokens_per_seed"]
                != representative_a["evaluation_tokens"]
                or effect["packed_sequences_per_seed"]
                != representative_a["packed_sequences"]
                or effect["token_identity_sha256"]
                != representative_a["token_identity_sha256"]
            ):
                raise Experiment5RoutedLadderError(
                    f"{effect_name} is incompatible with per-seed evidence"
                )
    expected_across = {
        (panel, checkpoint_tokens)
        for panel in PANELS
        for checkpoint_tokens in CHECKPOINT_TOKENS
    }
    if seen_across != expected_across:
        raise Experiment5RoutedLadderError(
            "routed-ladder across-seed summary identities are incomplete"
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
    "ExistingHReferences",
    "Experiment5RoutedLadderError",
    "ImmutableArtifactReference",
    "PROTOCOL_ID",
    "REPORT_SCHEMA_VERSION",
    "RoutedLadderHTokenScores",
    "RoutedLadderTokenScores",
    "SCORE_SCHEMA_VERSION",
    "SCORED_CONDITIONS",
    "SEEDS",
    "build_experiment5_routed_ladder_report",
    "h_token_scores_to_existing_reference",
    "hierarchical_a512_softplus_minus_a512_relu_bootstrap",
    "paired_a512_softplus_minus_a512_relu_bootstrap",
    "validate_experiment5_routed_ladder_report",
]
