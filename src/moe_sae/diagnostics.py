"""Pure diagnostic primitives for the Gemma 4 mechanism audit.

This module deliberately contains no checkpoint, JAX, or storage code.  It
provides two small pieces needed by the first retrospective gate:

* a reproducible sequence-clustered random-routing control; and
* paired reconstruction errors retained at the sequence level.

The random-routing control follows the historical condition-C construction.
For every sequence, a keyed permutation of all expert labels is constructed
and applied to every token in that sequence.  Router slots and their weights
therefore remain aligned, while expert identity is randomized consistently
within a sequence.  The mapping is based on BLAKE2b rather than Python's
process-randomized ``hash`` or a library RNG, so it is stable across hosts and
processes.

Sequence-level error records are intentionally preserved in the output.  A
paired cluster bootstrap can consequently resample whole sequences without
pretending that adjacent packed tokens are independent observations.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

PAIRED_SSE_SCHEMA_VERSION = "gemma4-gate1-paired-sequence-sse-v1"
SEQUENCE_SSE_SCHEMA_VERSION = "gemma4-gate1-sequence-sse-v1"
PAIRED_BOOTSTRAP_SCHEMA_VERSION = "gemma4-gate1-paired-sequence-bootstrap-v1"
PAIRED_BOOTSTRAP_REPLICATES = 20_000
ROUTE_PERMUTATION_DOMAIN = "gemma4-gate1-router-permutation-v1"
ROUTE_MODES = frozenset({"true", "shuffled"})
RouteMode = Literal["true", "shuffled"]

_PERMUTATION_DOMAIN = ROUTE_PERMUTATION_DOMAIN.encode("ascii") + b"\0"
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_UINT64_MAX = 2**64 - 1
_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
_BOOTSTRAP_CHUNK_REPLICATES = 256
_BOOTSTRAP_ENGINE = "numpy.random.PCG64"
_BOOTSTRAP_INTERVAL_METHOD = "paired-sequence-cluster-percentile"
_BOOTSTRAP_QUANTILE_METHOD = "linear"


class DiagnosticError(ValueError):
    """A Gate-1 diagnostic input or output violates its contract."""


def _integer(
    name: str,
    value: Any,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise DiagnosticError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise DiagnosticError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _finite_float(name: str, value: Any, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise DiagnosticError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DiagnosticError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise DiagnosticError(f"{name} must be finite")
    if nonnegative and result < 0:
        raise DiagnosticError(f"{name} must be nonnegative")
    return result


def sequence_expert_permutation(
    sequence_id: int,
    *,
    seed: int,
    num_experts: int = 128,
    domain: str = ROUTE_PERMUTATION_DOMAIN,
) -> np.ndarray:
    """Return the stable keyed expert-label permutation for one sequence.

    The returned vector satisfies ``result[original_expert_id] ==
    shuffled_expert_id``.  Digest collisions are resolved by the original
    label, making the construction a total deterministic ordering even in the
    vanishingly unlikely collision case.
    """

    selected_sequence = _integer(
        "sequence_id", sequence_id, minimum=_INT64_MIN, maximum=_INT64_MAX
    )
    selected_seed = _integer("seed", seed, minimum=0, maximum=_UINT64_MAX)
    selected_experts = _integer(
        "num_experts", num_experts, minimum=2, maximum=2**31 - 1
    )
    if domain != ROUTE_PERMUTATION_DOMAIN:
        raise DiagnosticError(
            f"route permutation domain must be {ROUTE_PERMUTATION_DOMAIN!r}"
        )
    prefix = (
        _PERMUTATION_DOMAIN
        + selected_seed.to_bytes(8, "little", signed=False)
        + selected_sequence.to_bytes(8, "little", signed=True)
    )
    labels = list(range(selected_experts))
    labels.sort(
        key=lambda expert: (
            hashlib.blake2b(
                prefix + expert.to_bytes(4, "little", signed=False),
                digest_size=16,
            ).digest(),
            expert,
        )
    )
    permutation = np.empty(selected_experts, dtype=np.int64)
    permutation[np.asarray(labels, dtype=np.int64)] = np.arange(
        selected_experts, dtype=np.int64
    )
    return permutation


def apply_route_mode(
    expert_ids: Any,
    sequence_ids: Any,
    *,
    mode: RouteMode,
    seed: int,
    num_experts: int = 128,
    domain: str = ROUTE_PERMUTATION_DOMAIN,
) -> np.ndarray:
    """Return true or sequence-keyed shuffled expert IDs.

    ``expert_ids`` must be an integer ``[N,K]`` matrix containing unique,
    in-range IDs in every token row.  ``sequence_ids`` is the corresponding
    signed integer ``[N]`` vector.  A shuffled route uses one bijection per
    sequence, so every occurrence of an expert inside that sequence maps to
    the same randomized bank.  The output preserves shape, integer dtype,
    per-token uniqueness, and the valid expert range.

    ``seed`` is required for both modes so the full diagnostic invocation has
    an explicit randomization identity even when ``mode="true"``.
    """

    if mode not in ROUTE_MODES:
        raise DiagnosticError(
            f"mode must be one of {sorted(ROUTE_MODES)}, got {mode!r}"
        )
    selected_seed = _integer("seed", seed, minimum=0, maximum=_UINT64_MAX)
    selected_experts = _integer(
        "num_experts", num_experts, minimum=2, maximum=2**31 - 1
    )
    if domain != ROUTE_PERMUTATION_DOMAIN:
        raise DiagnosticError(
            f"route permutation domain must be {ROUTE_PERMUTATION_DOMAIN!r}"
        )
    ids = np.asarray(expert_ids)
    sequences = np.asarray(sequence_ids)
    if ids.ndim != 2 or ids.shape[1] == 0:
        raise DiagnosticError(f"expert_ids must have shape [N,K], got {ids.shape}")
    if sequences.shape != (ids.shape[0],):
        raise DiagnosticError(
            f"sequence_ids must have shape [{ids.shape[0]}], got {sequences.shape}"
        )
    if not np.issubdtype(ids.dtype, np.integer):
        raise DiagnosticError("expert_ids must use an integer dtype")
    if not np.issubdtype(sequences.dtype, np.signedinteger):
        raise DiagnosticError("sequence_ids must use a signed integer dtype")
    if np.iinfo(ids.dtype).max < selected_experts - 1:
        raise DiagnosticError(
            "expert_ids dtype cannot represent the configured expert range"
        )
    if ids.size and (np.any(ids < 0) or np.any(ids >= selected_experts)):
        raise DiagnosticError("expert_ids are outside the configured expert range")
    if ids.shape[1] > selected_experts:
        raise DiagnosticError("a token cannot route to more experts than exist")
    if ids.shape[1] > 1 and np.any(
        np.diff(np.sort(ids.astype(np.int64, copy=False), axis=1), axis=1) == 0
    ):
        raise DiagnosticError("every token must contain unique expert IDs")

    # The input schema uses uint8 today, but retaining the caller's integer
    # dtype makes the helper correct for wider future expert inventories.
    output = ids.copy()
    if mode == "true" or ids.shape[0] == 0:
        return output

    for sequence in np.unique(sequences):
        sequence_value = _integer(
            "sequence_id", sequence, minimum=_INT64_MIN, maximum=_INT64_MAX
        )
        permutation = sequence_expert_permutation(
            sequence_value,
            seed=selected_seed,
            num_experts=selected_experts,
            domain=domain,
        )
        selected = sequences == sequence
        output[selected] = permutation[ids[selected]].astype(ids.dtype, copy=False)

    # These checks are cheap relative to an SAE evaluation and make the
    # counterfactual fail closed if this implementation is ever changed.
    if output.size and (np.any(output < 0) or np.any(output >= selected_experts)):
        raise DiagnosticError("shuffled expert IDs escaped the configured range")
    if output.shape[1] > 1 and np.any(
        np.diff(np.sort(output.astype(np.int64, copy=False), axis=1), axis=1) == 0
    ):
        raise DiagnosticError("shuffling did not preserve per-token uniqueness")
    return output


@dataclasses.dataclass(slots=True)
class _SequenceErrors:
    tokens: int = 0
    sst: float = 0.0
    baseline_sse: float = 0.0
    candidate_sse: float = 0.0


@dataclasses.dataclass(slots=True)
class _SingleSequenceErrors:
    tokens: int = 0
    sst: float = 0.0
    sse: float = 0.0


class SequenceSSEAccumulator:
    """Accumulate device-computed per-token SSE/SST by sequence.

    TPU diagnostics return scalar errors rather than transferring full
    2,816-dimensional reconstructions to the host.  This accumulator preserves
    the same sequence-clustered statistical unit as
    :class:`PairedSequenceSSEAccumulator`.
    """

    def __init__(self, *, label: str) -> None:
        if not isinstance(label, str) or not label:
            raise DiagnosticError("label must be a nonempty string")
        self.label = label
        self._sequences: dict[int, _SingleSequenceErrors] = {}

    def update(
        self,
        sequence_ids: Any,
        token_sse: Any,
        token_sst: Any,
        *,
        valid_mask: Any | None = None,
    ) -> None:
        sequences = np.asarray(sequence_ids)
        sse = np.asarray(token_sse, dtype=np.float64)
        sst = np.asarray(token_sst, dtype=np.float64)
        if sequences.ndim != 1:
            raise DiagnosticError("sequence_ids must be rank one")
        if not np.issubdtype(sequences.dtype, np.signedinteger):
            raise DiagnosticError("sequence_ids must use a signed integer dtype")
        if sse.shape != sequences.shape or sst.shape != sequences.shape:
            raise DiagnosticError(
                "sequence_ids, token_sse, and token_sst must have identical shapes"
            )
        if not np.isfinite(sse).all() or not np.isfinite(sst).all():
            raise DiagnosticError("token SSE/SST must be finite")
        if np.any(sse < 0) or np.any(sst < 0):
            raise DiagnosticError("token SSE/SST must be nonnegative")
        valid = (
            np.ones(sequences.shape, dtype=bool)
            if valid_mask is None
            else np.asarray(valid_mask)
        )
        if valid.shape != sequences.shape or valid.dtype != np.dtype(bool):
            raise DiagnosticError("valid_mask must be a boolean vector")
        for sequence in np.unique(sequences[valid]):
            sequence_value = _integer(
                "sequence_id", sequence, minimum=_INT64_MIN, maximum=_INT64_MAX
            )
            selected = valid & (sequences == sequence)
            totals = self._sequences.setdefault(sequence_value, _SingleSequenceErrors())
            totals.tokens += int(np.count_nonzero(selected))
            totals.sst += float(sst[selected].sum(dtype=np.float64))
            totals.sse += float(sse[selected].sum(dtype=np.float64))

    def finalize(self) -> dict[str, Any]:
        if not self._sequences:
            raise DiagnosticError("cannot finalize an empty sequence SSE accumulator")
        clusters = [
            {
                "sequence_id": sequence,
                "tokens": totals.tokens,
                "sst": totals.sst,
                "sse": totals.sse,
            }
            for sequence, totals in sorted(self._sequences.items())
        ]
        tokens = sum(row["tokens"] for row in clusters)
        sst = math.fsum(row["sst"] for row in clusters)
        sse = math.fsum(row["sse"] for row in clusters)
        if sst <= 0:
            raise DiagnosticError("target SST must be positive")
        return validate_sequence_sse_summary(
            {
                "schema_version": SEQUENCE_SSE_SCHEMA_VERSION,
                "label": self.label,
                "tokens": tokens,
                "sequence_count": len(clusters),
                "sst": sst,
                "sse": sse,
                "nmse": sse / sst,
                "ev": 1.0 - sse / sst,
                "clusters": clusters,
            }
        )


class PairedSequenceSSEAccumulator:
    """Stream paired reconstruction errors while retaining sequence clusters.

    ``candidate_minus_baseline_ev`` is positive exactly when the candidate has
    lower SSE on the common target.  Both reconstructions always share the
    supplied training mean and hence one SST.
    """

    def __init__(
        self,
        mean: Any,
        *,
        baseline_label: str,
        candidate_label: str,
    ) -> None:
        selected_mean = np.asarray(mean, dtype=np.float64)
        if (
            selected_mean.ndim != 1
            or selected_mean.size == 0
            or not np.isfinite(selected_mean).all()
        ):
            raise DiagnosticError("mean must be a finite nonempty vector")
        if not isinstance(baseline_label, str) or not baseline_label:
            raise DiagnosticError("baseline_label must be a nonempty string")
        if not isinstance(candidate_label, str) or not candidate_label:
            raise DiagnosticError("candidate_label must be a nonempty string")
        if baseline_label == candidate_label:
            raise DiagnosticError("baseline and candidate labels must differ")
        self.mean = selected_mean
        self.baseline_label = baseline_label
        self.candidate_label = candidate_label
        self._sequences: dict[int, _SequenceErrors] = {}

    def update(
        self,
        sequence_ids: Any,
        target: Any,
        baseline_reconstruction: Any,
        candidate_reconstruction: Any,
    ) -> None:
        """Accumulate one batch, allowing a sequence to span many batches."""

        sequences = np.asarray(sequence_ids)
        target64 = np.asarray(target, dtype=np.float64)
        baseline64 = np.asarray(baseline_reconstruction, dtype=np.float64)
        candidate64 = np.asarray(candidate_reconstruction, dtype=np.float64)
        expected_shape = target64.shape
        if target64.ndim != 2 or target64.shape[1] != self.mean.size:
            raise DiagnosticError(
                f"target must have shape [N,{self.mean.size}], got {target64.shape}"
            )
        if baseline64.shape != expected_shape or candidate64.shape != expected_shape:
            raise DiagnosticError(
                "target, baseline reconstruction, and candidate reconstruction "
                "must have identical shapes"
            )
        if sequences.shape != (target64.shape[0],):
            raise DiagnosticError(
                f"sequence_ids must have shape [{target64.shape[0]}], "
                f"got {sequences.shape}"
            )
        if not np.issubdtype(sequences.dtype, np.signedinteger):
            raise DiagnosticError("sequence_ids must use a signed integer dtype")
        if not (
            np.isfinite(target64).all()
            and np.isfinite(baseline64).all()
            and np.isfinite(candidate64).all()
        ):
            raise DiagnosticError("paired reconstruction inputs must all be finite")
        if target64.shape[0] == 0:
            return

        centered_squared = np.square(target64 - self.mean)
        baseline_squared = np.square(target64 - baseline64)
        candidate_squared = np.square(target64 - candidate64)
        for sequence in np.unique(sequences):
            sequence_value = _integer(
                "sequence_id", sequence, minimum=_INT64_MIN, maximum=_INT64_MAX
            )
            selected = sequences == sequence
            totals = self._sequences.setdefault(sequence_value, _SequenceErrors())
            totals.tokens += int(np.count_nonzero(selected))
            totals.sst += float(centered_squared[selected].sum(dtype=np.float64))
            totals.baseline_sse += float(
                baseline_squared[selected].sum(dtype=np.float64)
            )
            totals.candidate_sse += float(
                candidate_squared[selected].sum(dtype=np.float64)
            )

    def finalize(self) -> dict[str, Any]:
        """Return a strict JSON-safe, cluster-bootstrap-ready summary."""

        if not self._sequences:
            raise DiagnosticError("cannot finalize an empty paired SSE accumulator")
        clusters = [
            {
                "sequence_id": sequence,
                "tokens": totals.tokens,
                "sst": totals.sst,
                "baseline_sse": totals.baseline_sse,
                "candidate_sse": totals.candidate_sse,
                "candidate_minus_baseline_sse": (
                    totals.candidate_sse - totals.baseline_sse
                ),
            }
            for sequence, totals in sorted(self._sequences.items())
        ]
        tokens = sum(cluster["tokens"] for cluster in clusters)
        sst = math.fsum(cluster["sst"] for cluster in clusters)
        baseline_sse = math.fsum(cluster["baseline_sse"] for cluster in clusters)
        candidate_sse = math.fsum(cluster["candidate_sse"] for cluster in clusters)
        if sst <= 0:
            raise DiagnosticError("paired target SST must be positive")
        baseline_nmse = baseline_sse / sst
        candidate_nmse = candidate_sse / sst
        result = {
            "schema_version": PAIRED_SSE_SCHEMA_VERSION,
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "tokens": tokens,
            "sequence_count": len(clusters),
            "sst": sst,
            "baseline_sse": baseline_sse,
            "candidate_sse": candidate_sse,
            "candidate_minus_baseline_sse": candidate_sse - baseline_sse,
            "baseline_nmse": baseline_nmse,
            "candidate_nmse": candidate_nmse,
            "baseline_ev": 1.0 - baseline_nmse,
            "candidate_ev": 1.0 - candidate_nmse,
            "candidate_minus_baseline_ev": (baseline_sse - candidate_sse) / sst,
            "clusters": clusters,
        }
        return validate_paired_sequence_sse_summary(result)


_SUMMARY_FIELDS = {
    "schema_version",
    "baseline_label",
    "candidate_label",
    "tokens",
    "sequence_count",
    "sst",
    "baseline_sse",
    "candidate_sse",
    "candidate_minus_baseline_sse",
    "baseline_nmse",
    "candidate_nmse",
    "baseline_ev",
    "candidate_ev",
    "candidate_minus_baseline_ev",
    "clusters",
}
_CLUSTER_FIELDS = {
    "sequence_id",
    "tokens",
    "sst",
    "baseline_sse",
    "candidate_sse",
    "candidate_minus_baseline_sse",
}


def _close(name: str, observed: float, expected: float) -> None:
    if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise DiagnosticError(f"{name} is inconsistent with the paired clusters")


_SINGLE_SUMMARY_FIELDS = {
    "schema_version",
    "label",
    "tokens",
    "sequence_count",
    "sst",
    "sse",
    "nmse",
    "ev",
    "clusters",
}
_SINGLE_CLUSTER_FIELDS = {"sequence_id", "tokens", "sst", "sse"}


def validate_sequence_sse_summary(value: Any) -> dict[str, Any]:
    """Validate one checkpoint/mode's sequence-clustered reconstruction errors."""

    if not isinstance(value, Mapping) or set(value) != _SINGLE_SUMMARY_FIELDS:
        raise DiagnosticError("sequence SSE summary fields do not match the schema")
    if value.get("schema_version") != SEQUENCE_SSE_SCHEMA_VERSION:
        raise DiagnosticError(
            f"sequence SSE summary schema must be {SEQUENCE_SSE_SCHEMA_VERSION}"
        )
    label = value.get("label")
    if not isinstance(label, str) or not label:
        raise DiagnosticError("label must be a nonempty string")
    tokens = _integer("tokens", value.get("tokens"), minimum=1, maximum=2**63 - 1)
    count = _integer(
        "sequence_count",
        value.get("sequence_count"),
        minimum=1,
        maximum=2**63 - 1,
    )
    raw_clusters = value.get("clusters")
    if not isinstance(raw_clusters, list) or len(raw_clusters) != count:
        raise DiagnosticError("clusters must contain exactly sequence_count records")
    clusters: list[dict[str, int | float]] = []
    previous: int | None = None
    for index, raw in enumerate(raw_clusters):
        if not isinstance(raw, Mapping) or set(raw) != _SINGLE_CLUSTER_FIELDS:
            raise DiagnosticError(
                f"cluster {index} fields do not match the sequence SSE schema"
            )
        sequence = _integer(
            f"clusters[{index}].sequence_id",
            raw.get("sequence_id"),
            minimum=_INT64_MIN,
            maximum=_INT64_MAX,
        )
        if previous is not None and sequence <= previous:
            raise DiagnosticError("clusters must have unique increasing sequence IDs")
        previous = sequence
        clusters.append(
            {
                "sequence_id": sequence,
                "tokens": _integer(
                    f"clusters[{index}].tokens",
                    raw.get("tokens"),
                    minimum=1,
                    maximum=2**63 - 1,
                ),
                "sst": _finite_float(
                    f"clusters[{index}].sst",
                    raw.get("sst"),
                    nonnegative=True,
                ),
                "sse": _finite_float(
                    f"clusters[{index}].sse",
                    raw.get("sse"),
                    nonnegative=True,
                ),
            }
        )
    if sum(int(row["tokens"]) for row in clusters) != tokens:
        raise DiagnosticError("summary tokens do not equal clustered token counts")
    sst = _finite_float("sst", value.get("sst"), nonnegative=True)
    sse = _finite_float("sse", value.get("sse"), nonnegative=True)
    if sst <= 0:
        raise DiagnosticError("target SST must be positive")
    _close("sst", sst, math.fsum(float(row["sst"]) for row in clusters))
    _close("sse", sse, math.fsum(float(row["sse"]) for row in clusters))
    nmse = _finite_float("nmse", value.get("nmse"))
    ev = _finite_float("ev", value.get("ev"))
    _close("nmse", nmse, sse / sst)
    _close("ev", ev, 1.0 - sse / sst)
    return {
        "schema_version": SEQUENCE_SSE_SCHEMA_VERSION,
        "label": label,
        "tokens": tokens,
        "sequence_count": count,
        "sst": sst,
        "sse": sse,
        "nmse": nmse,
        "ev": ev,
        "clusters": clusters,
    }


def merge_sequence_sse_summaries(summaries: Any) -> dict[str, Any]:
    """Merge host/range fragments while preserving whole sequence clusters.

    A packed sequence may cross a device batch, a process-local range, or an
    operator output fragment.  Every input is validated independently, all
    labels must be identical, and repeated ``sequence_id`` rows are summed
    before aggregate EV/NMSE is recomputed.
    """

    if isinstance(summaries, (str, bytes, bytearray, Mapping)):
        raise DiagnosticError(
            "sequence SSE summaries must be a nonempty iterable of summaries"
        )
    try:
        parts = tuple(summaries)
    except TypeError as error:
        raise DiagnosticError(
            "sequence SSE summaries must be a nonempty iterable of summaries"
        ) from error
    if not parts:
        raise DiagnosticError("cannot merge an empty sequence SSE summary set")
    validated = tuple(validate_sequence_sse_summary(part) for part in parts)
    label = validated[0]["label"]
    if any(part["label"] != label for part in validated[1:]):
        raise DiagnosticError("sequence SSE fragments must have the same label")

    token_parts: dict[int, list[int]] = {}
    sst_parts: dict[int, list[float]] = {}
    sse_parts: dict[int, list[float]] = {}
    for part in validated:
        for cluster in part["clusters"]:
            sequence_id = int(cluster["sequence_id"])
            token_parts.setdefault(sequence_id, []).append(int(cluster["tokens"]))
            sst_parts.setdefault(sequence_id, []).append(float(cluster["sst"]))
            sse_parts.setdefault(sequence_id, []).append(float(cluster["sse"]))

    clusters: list[dict[str, int | float]] = []
    for sequence_id in sorted(token_parts):
        cluster_tokens = sum(token_parts[sequence_id])
        _integer(
            f"merged sequence {sequence_id} tokens",
            cluster_tokens,
            minimum=1,
            maximum=_INT64_MAX,
        )
        cluster_sst = math.fsum(sst_parts[sequence_id])
        cluster_sse = math.fsum(sse_parts[sequence_id])
        if not math.isfinite(cluster_sst) or not math.isfinite(cluster_sse):
            raise DiagnosticError("merged sequence SSE/SST overflowed")
        clusters.append(
            {
                "sequence_id": sequence_id,
                "tokens": cluster_tokens,
                "sst": cluster_sst,
                "sse": cluster_sse,
            }
        )

    tokens = sum(int(cluster["tokens"]) for cluster in clusters)
    _integer("merged tokens", tokens, minimum=1, maximum=_INT64_MAX)
    sst = math.fsum(float(cluster["sst"]) for cluster in clusters)
    sse = math.fsum(float(cluster["sse"]) for cluster in clusters)
    if not math.isfinite(sst) or not math.isfinite(sse):
        raise DiagnosticError("merged sequence SSE/SST aggregate overflowed")
    if sst <= 0:
        raise DiagnosticError("merged target SST must be positive")
    return validate_sequence_sse_summary(
        {
            "schema_version": SEQUENCE_SSE_SCHEMA_VERSION,
            "label": label,
            "tokens": tokens,
            "sequence_count": len(clusters),
            "sst": sst,
            "sse": sse,
            "nmse": sse / sst,
            "ev": 1.0 - sse / sst,
            "clusters": clusters,
        }
    )


def pair_sequence_sse_summaries(baseline: Any, candidate: Any) -> dict[str, Any]:
    """Join two summaries evaluated on exactly the same sequence/token measure."""

    baseline_value = validate_sequence_sse_summary(baseline)
    candidate_value = validate_sequence_sse_summary(candidate)
    if baseline_value["label"] == candidate_value["label"]:
        raise DiagnosticError("baseline and candidate labels must differ")
    baseline_clusters = baseline_value["clusters"]
    candidate_clusters = candidate_value["clusters"]
    if len(baseline_clusters) != len(candidate_clusters):
        raise DiagnosticError("paired summaries contain different sequences")
    clusters: list[dict[str, int | float]] = []
    for index, (base, cand) in enumerate(
        zip(baseline_clusters, candidate_clusters, strict=True)
    ):
        if (
            base["sequence_id"] != cand["sequence_id"]
            or base["tokens"] != cand["tokens"]
        ):
            raise DiagnosticError(
                f"paired summaries disagree on sequence cluster {index}"
            )
        if float(cand["sst"]) != float(base["sst"]):
            raise DiagnosticError(
                f"paired summaries disagree on exact SST for sequence cluster {index}"
            )
        baseline_sse = float(base["sse"])
        candidate_sse = float(cand["sse"])
        clusters.append(
            {
                "sequence_id": int(base["sequence_id"]),
                "tokens": int(base["tokens"]),
                "sst": float(base["sst"]),
                "baseline_sse": baseline_sse,
                "candidate_sse": candidate_sse,
                "candidate_minus_baseline_sse": candidate_sse - baseline_sse,
            }
        )
    result = {
        "schema_version": PAIRED_SSE_SCHEMA_VERSION,
        "baseline_label": baseline_value["label"],
        "candidate_label": candidate_value["label"],
        "tokens": baseline_value["tokens"],
        "sequence_count": baseline_value["sequence_count"],
        "sst": baseline_value["sst"],
        "baseline_sse": baseline_value["sse"],
        "candidate_sse": candidate_value["sse"],
        "candidate_minus_baseline_sse": (
            candidate_value["sse"] - baseline_value["sse"]
        ),
        "baseline_nmse": baseline_value["nmse"],
        "candidate_nmse": candidate_value["nmse"],
        "baseline_ev": baseline_value["ev"],
        "candidate_ev": candidate_value["ev"],
        "candidate_minus_baseline_ev": (baseline_value["sse"] - candidate_value["sse"])
        / baseline_value["sst"],
        "clusters": clusters,
    }
    return validate_paired_sequence_sse_summary(result)


def merge_and_pair_sequence_sse_summaries(
    baseline_summaries: Any,
    candidate_summaries: Any,
) -> dict[str, Any]:
    """Merge partial clusters independently, then form one exact paired sample."""

    return pair_sequence_sse_summaries(
        merge_sequence_sse_summaries(baseline_summaries),
        merge_sequence_sse_summaries(candidate_summaries),
    )


def validate_paired_sequence_sse_summary(value: Any) -> dict[str, Any]:
    """Validate and normalize a paired sequence-SSE output record.

    Validation is strict: fields are exact, clusters are unique and sorted,
    aggregate values must reproduce the cluster rows, and EV/NMSE identities
    must hold.  The returned mapping consists only of ordinary JSON-safe
    Python scalars and containers.
    """

    if not isinstance(value, Mapping):
        raise DiagnosticError("paired SSE summary must be a mapping")
    if set(value) != _SUMMARY_FIELDS:
        raise DiagnosticError("paired SSE summary fields do not match the schema")
    if value.get("schema_version") != PAIRED_SSE_SCHEMA_VERSION:
        raise DiagnosticError(
            f"paired SSE summary schema must be {PAIRED_SSE_SCHEMA_VERSION}"
        )
    baseline_label = value.get("baseline_label")
    candidate_label = value.get("candidate_label")
    if not isinstance(baseline_label, str) or not baseline_label:
        raise DiagnosticError("baseline_label must be a nonempty string")
    if not isinstance(candidate_label, str) or not candidate_label:
        raise DiagnosticError("candidate_label must be a nonempty string")
    if baseline_label == candidate_label:
        raise DiagnosticError("baseline and candidate labels must differ")
    tokens = _integer("tokens", value.get("tokens"), minimum=1, maximum=2**63 - 1)
    sequence_count = _integer(
        "sequence_count",
        value.get("sequence_count"),
        minimum=1,
        maximum=2**63 - 1,
    )
    clusters_value = value.get("clusters")
    if not isinstance(clusters_value, list) or len(clusters_value) != sequence_count:
        raise DiagnosticError("clusters must contain exactly sequence_count records")

    clusters: list[dict[str, int | float]] = []
    previous_sequence: int | None = None
    for index, raw_cluster in enumerate(clusters_value):
        if not isinstance(raw_cluster, Mapping) or set(raw_cluster) != _CLUSTER_FIELDS:
            raise DiagnosticError(
                f"cluster {index} fields do not match the paired SSE schema"
            )
        sequence = _integer(
            f"clusters[{index}].sequence_id",
            raw_cluster.get("sequence_id"),
            minimum=_INT64_MIN,
            maximum=_INT64_MAX,
        )
        if previous_sequence is not None and sequence <= previous_sequence:
            raise DiagnosticError("clusters must have unique increasing sequence IDs")
        previous_sequence = sequence
        cluster_tokens = _integer(
            f"clusters[{index}].tokens",
            raw_cluster.get("tokens"),
            minimum=1,
            maximum=2**63 - 1,
        )
        cluster_sst = _finite_float(
            f"clusters[{index}].sst", raw_cluster.get("sst"), nonnegative=True
        )
        cluster_baseline = _finite_float(
            f"clusters[{index}].baseline_sse",
            raw_cluster.get("baseline_sse"),
            nonnegative=True,
        )
        cluster_candidate = _finite_float(
            f"clusters[{index}].candidate_sse",
            raw_cluster.get("candidate_sse"),
            nonnegative=True,
        )
        cluster_delta = _finite_float(
            f"clusters[{index}].candidate_minus_baseline_sse",
            raw_cluster.get("candidate_minus_baseline_sse"),
        )
        _close(
            f"clusters[{index}].candidate_minus_baseline_sse",
            cluster_delta,
            cluster_candidate - cluster_baseline,
        )
        clusters.append(
            {
                "sequence_id": sequence,
                "tokens": cluster_tokens,
                "sst": cluster_sst,
                "baseline_sse": cluster_baseline,
                "candidate_sse": cluster_candidate,
                "candidate_minus_baseline_sse": cluster_delta,
            }
        )

    if sum(int(cluster["tokens"]) for cluster in clusters) != tokens:
        raise DiagnosticError("summary tokens do not equal clustered token counts")
    summed_sst = math.fsum(float(cluster["sst"]) for cluster in clusters)
    summed_baseline = math.fsum(float(cluster["baseline_sse"]) for cluster in clusters)
    summed_candidate = math.fsum(
        float(cluster["candidate_sse"]) for cluster in clusters
    )
    sst = _finite_float("sst", value.get("sst"), nonnegative=True)
    baseline_sse = _finite_float(
        "baseline_sse", value.get("baseline_sse"), nonnegative=True
    )
    candidate_sse = _finite_float(
        "candidate_sse", value.get("candidate_sse"), nonnegative=True
    )
    delta_sse = _finite_float(
        "candidate_minus_baseline_sse",
        value.get("candidate_minus_baseline_sse"),
    )
    if sst <= 0:
        raise DiagnosticError("paired target SST must be positive")
    _close("sst", sst, summed_sst)
    _close("baseline_sse", baseline_sse, summed_baseline)
    _close("candidate_sse", candidate_sse, summed_candidate)
    _close("candidate_minus_baseline_sse", delta_sse, candidate_sse - baseline_sse)

    scalar_metrics = {
        name: _finite_float(name, value.get(name))
        for name in (
            "baseline_nmse",
            "candidate_nmse",
            "baseline_ev",
            "candidate_ev",
            "candidate_minus_baseline_ev",
        )
    }
    _close("baseline_nmse", scalar_metrics["baseline_nmse"], baseline_sse / sst)
    _close("candidate_nmse", scalar_metrics["candidate_nmse"], candidate_sse / sst)
    _close("baseline_ev", scalar_metrics["baseline_ev"], 1.0 - baseline_sse / sst)
    _close(
        "candidate_ev",
        scalar_metrics["candidate_ev"],
        1.0 - candidate_sse / sst,
    )
    _close(
        "candidate_minus_baseline_ev",
        scalar_metrics["candidate_minus_baseline_ev"],
        (baseline_sse - candidate_sse) / sst,
    )
    return {
        "schema_version": PAIRED_SSE_SCHEMA_VERSION,
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "tokens": tokens,
        "sequence_count": sequence_count,
        "sst": sst,
        "baseline_sse": baseline_sse,
        "candidate_sse": candidate_sse,
        "candidate_minus_baseline_sse": delta_sse,
        **scalar_metrics,
        "clusters": clusters,
    }


_BOOTSTRAP_FIELDS = {
    "schema_version",
    "paired_summary_sha256",
    "baseline_label",
    "candidate_label",
    "tokens",
    "sequence_count",
    "seed",
    "replicates",
    "confidence_level",
    "engine",
    "interval_method",
    "quantile_method",
    "point_candidate_minus_baseline_ev",
    "bootstrap_mean_candidate_minus_baseline_ev",
    "bootstrap_standard_error",
    "ci_lower",
    "ci_upper",
}


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DiagnosticError("diagnostic summary is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _sha256_value(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DiagnosticError(f"{name} must be 64 lowercase hexadecimal digits")
    return value


def _json_number(name: str, value: Any, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise DiagnosticError(f"{name} must be a JSON number")
    return _finite_float(name, value, nonnegative=nonnegative)


def validate_paired_sequence_bootstrap_ci(
    value: Any,
    paired_summary: Any,
) -> dict[str, Any]:
    """Validate a strict JSON-safe bootstrap result and its source binding."""

    if not isinstance(value, Mapping) or set(value) != _BOOTSTRAP_FIELDS:
        raise DiagnosticError("paired bootstrap fields do not match the schema")
    if value.get("schema_version") != PAIRED_BOOTSTRAP_SCHEMA_VERSION:
        raise DiagnosticError(
            f"paired bootstrap schema must be {PAIRED_BOOTSTRAP_SCHEMA_VERSION}"
        )
    paired = validate_paired_sequence_sse_summary(paired_summary)
    paired_sha256 = _sha256_value(
        "paired_summary_sha256", value.get("paired_summary_sha256")
    )
    if paired_sha256 != _canonical_sha256(paired):
        raise DiagnosticError("paired bootstrap source summary hash mismatch")
    baseline_label = value.get("baseline_label")
    candidate_label = value.get("candidate_label")
    if baseline_label != paired["baseline_label"]:
        raise DiagnosticError("paired bootstrap baseline label mismatch")
    if candidate_label != paired["candidate_label"]:
        raise DiagnosticError("paired bootstrap candidate label mismatch")
    tokens = _integer("tokens", value.get("tokens"), minimum=1, maximum=_INT64_MAX)
    sequence_count = _integer(
        "sequence_count",
        value.get("sequence_count"),
        minimum=1,
        maximum=_INT64_MAX,
    )
    if tokens != paired["tokens"] or sequence_count != paired["sequence_count"]:
        raise DiagnosticError("paired bootstrap source dimensions mismatch")
    seed = _integer("seed", value.get("seed"), minimum=0, maximum=_UINT64_MAX)
    replicates = _integer(
        "replicates", value.get("replicates"), minimum=1, maximum=_INT64_MAX
    )
    if replicates != PAIRED_BOOTSTRAP_REPLICATES:
        raise DiagnosticError(
            f"paired bootstrap requires exactly {PAIRED_BOOTSTRAP_REPLICATES} replicates"
        )
    confidence_level = _json_number("confidence_level", value.get("confidence_level"))
    if confidence_level != _BOOTSTRAP_CONFIDENCE_LEVEL:
        raise DiagnosticError("paired bootstrap confidence level must be exactly 0.95")
    if value.get("engine") != _BOOTSTRAP_ENGINE:
        raise DiagnosticError("paired bootstrap engine mismatch")
    if value.get("interval_method") != _BOOTSTRAP_INTERVAL_METHOD:
        raise DiagnosticError("paired bootstrap interval method mismatch")
    if value.get("quantile_method") != _BOOTSTRAP_QUANTILE_METHOD:
        raise DiagnosticError("paired bootstrap quantile method mismatch")

    point = _json_number(
        "point_candidate_minus_baseline_ev",
        value.get("point_candidate_minus_baseline_ev"),
    )
    _close(
        "point_candidate_minus_baseline_ev",
        point,
        float(paired["candidate_minus_baseline_ev"]),
    )
    bootstrap_mean = _json_number(
        "bootstrap_mean_candidate_minus_baseline_ev",
        value.get("bootstrap_mean_candidate_minus_baseline_ev"),
    )
    standard_error = _json_number(
        "bootstrap_standard_error",
        value.get("bootstrap_standard_error"),
        nonnegative=True,
    )
    ci_lower = _json_number("ci_lower", value.get("ci_lower"))
    ci_upper = _json_number("ci_upper", value.get("ci_upper"))
    if ci_lower > ci_upper:
        raise DiagnosticError("paired bootstrap CI bounds are reversed")
    (
        expected_mean,
        expected_standard_error,
        expected_lower,
        expected_upper,
    ) = _paired_bootstrap_statistics(paired, seed)
    _close(
        "bootstrap_mean_candidate_minus_baseline_ev",
        bootstrap_mean,
        expected_mean,
    )
    _close(
        "bootstrap_standard_error",
        standard_error,
        expected_standard_error,
    )
    _close("ci_lower", ci_lower, expected_lower)
    _close("ci_upper", ci_upper, expected_upper)
    result = {
        "schema_version": PAIRED_BOOTSTRAP_SCHEMA_VERSION,
        "paired_summary_sha256": paired_sha256,
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "tokens": tokens,
        "sequence_count": sequence_count,
        "seed": seed,
        "replicates": replicates,
        "confidence_level": confidence_level,
        "engine": _BOOTSTRAP_ENGINE,
        "interval_method": _BOOTSTRAP_INTERVAL_METHOD,
        "quantile_method": _BOOTSTRAP_QUANTILE_METHOD,
        "point_candidate_minus_baseline_ev": point,
        "bootstrap_mean_candidate_minus_baseline_ev": bootstrap_mean,
        "bootstrap_standard_error": standard_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as error:  # pragma: no cover - guarded above
        raise DiagnosticError("paired bootstrap result is not JSON-safe") from error
    return result


def _paired_bootstrap_statistics(
    paired: Mapping[str, Any], selected_seed: int
) -> tuple[float, float, float, float]:
    cluster_sst = np.asarray(
        [cluster["sst"] for cluster in paired["clusters"]], dtype=np.float64
    )
    cluster_ev_numerator = np.asarray(
        [
            float(cluster["baseline_sse"]) - float(cluster["candidate_sse"])
            for cluster in paired["clusters"]
        ],
        dtype=np.float64,
    )
    sequence_count = cluster_sst.size
    if sequence_count == 0:  # pragma: no cover - paired validator rejects it
        raise DiagnosticError("paired bootstrap requires at least one sequence")
    generator = np.random.Generator(np.random.PCG64(selected_seed))
    estimates = np.empty(PAIRED_BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, PAIRED_BOOTSTRAP_REPLICATES, _BOOTSTRAP_CHUNK_REPLICATES):
        end = min(
            start + _BOOTSTRAP_CHUNK_REPLICATES,
            PAIRED_BOOTSTRAP_REPLICATES,
        )
        indexes = generator.integers(
            0,
            sequence_count,
            size=(end - start, sequence_count),
            dtype=np.int64,
        )
        sampled_sst = cluster_sst[indexes].sum(axis=1, dtype=np.float64)
        sampled_numerator = cluster_ev_numerator[indexes].sum(axis=1, dtype=np.float64)
        if (
            not np.isfinite(sampled_sst).all()
            or not np.isfinite(sampled_numerator).all()
            or np.any(sampled_sst <= 0.0)
        ):
            raise DiagnosticError(
                "paired bootstrap produced a non-finite or zero-SST replicate"
            )
        estimates[start:end] = sampled_numerator / sampled_sst
    if not np.isfinite(estimates).all():  # pragma: no cover - guarded above
        raise DiagnosticError("paired bootstrap produced a non-finite estimate")
    alpha = (1.0 - _BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    ci_lower, ci_upper = np.quantile(
        estimates,
        [alpha, 1.0 - alpha],
        method=_BOOTSTRAP_QUANTILE_METHOD,
    )
    return (
        float(estimates.mean(dtype=np.float64)),
        float(estimates.std(dtype=np.float64, ddof=1)),
        float(ci_lower),
        float(ci_upper),
    )


def paired_sequence_cluster_bootstrap_ci(
    paired_summary: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    """Return a deterministic 95% paired sequence-cluster percentile CI.

    Each replicate samples ``sequence_count`` whole clusters uniformly with
    replacement.  Candidate-minus-baseline EV is recomputed as a ratio of the
    resampled additive SSE difference and the resampled common SST; sequence
    point estimates are never averaged directly.
    """

    paired = validate_paired_sequence_sse_summary(paired_summary)
    selected_seed = _integer("seed", seed, minimum=0, maximum=_UINT64_MAX)
    bootstrap_mean, standard_error, ci_lower, ci_upper = _paired_bootstrap_statistics(
        paired, selected_seed
    )
    result = {
        "schema_version": PAIRED_BOOTSTRAP_SCHEMA_VERSION,
        "paired_summary_sha256": _canonical_sha256(paired),
        "baseline_label": paired["baseline_label"],
        "candidate_label": paired["candidate_label"],
        "tokens": paired["tokens"],
        "sequence_count": paired["sequence_count"],
        "seed": selected_seed,
        "replicates": PAIRED_BOOTSTRAP_REPLICATES,
        "confidence_level": _BOOTSTRAP_CONFIDENCE_LEVEL,
        "engine": _BOOTSTRAP_ENGINE,
        "interval_method": _BOOTSTRAP_INTERVAL_METHOD,
        "quantile_method": _BOOTSTRAP_QUANTILE_METHOD,
        "point_candidate_minus_baseline_ev": paired["candidate_minus_baseline_ev"],
        "bootstrap_mean_candidate_minus_baseline_ev": bootstrap_mean,
        "bootstrap_standard_error": standard_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
    return validate_paired_sequence_bootstrap_ci(result, paired)


__all__ = [
    "PAIRED_BOOTSTRAP_REPLICATES",
    "PAIRED_BOOTSTRAP_SCHEMA_VERSION",
    "PAIRED_SSE_SCHEMA_VERSION",
    "ROUTE_MODES",
    "ROUTE_PERMUTATION_DOMAIN",
    "SEQUENCE_SSE_SCHEMA_VERSION",
    "DiagnosticError",
    "PairedSequenceSSEAccumulator",
    "RouteMode",
    "SequenceSSEAccumulator",
    "apply_route_mode",
    "merge_and_pair_sequence_sse_summaries",
    "merge_sequence_sse_summaries",
    "pair_sequence_sse_summaries",
    "paired_sequence_cluster_bootstrap_ci",
    "sequence_expert_permutation",
    "validate_paired_sequence_bootstrap_ci",
    "validate_paired_sequence_sse_summary",
    "validate_sequence_sse_summary",
]
